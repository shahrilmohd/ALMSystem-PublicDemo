"""
engine/ifrs17/_gmm_jit.py — JIT-compiled pure inner step for GmmEngine.

Implements the arithmetic core of one GMM period as a pure function that is
JIT-compiled with JAX.  GmmEngine calls this function and then updates its
mutable tracker state from the returned values.

Design (DECISIONS.md §32 Step 1)
----------------------------------
The function is pure — same inputs always produce the same outputs with no
side effects.  This is the prerequisite for @jax.jit.  All state mutation
(CsmTracker._csm, LossComponentTracker._balance, prior_bel_locked) stays
outside the JIT boundary in GmmEngine.step().

float64 precision
-----------------
JAX defaults to float32.  Actuarial projections require float64 to avoid
rounding errors in long-horizon CSM accretion chains.  jax_enable_x64 is
set to True at module import time.  This must be set before any JAX array
is created, so it lives here at the top level.

Fallback
--------
If JAX is not installed, _jit is replaced by an identity decorator and
jnp is aliased to numpy.  The function executes as plain Python/NumPy
with identical numerical results.  GmmEngine is unaware of which path
is active.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# JAX import with fallback
# ---------------------------------------------------------------------------

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    _jit = jax.jit
    JAX_AVAILABLE: bool = True

except ImportError:  # pragma: no cover
    import numpy as jnp  # type: ignore[no-redef]

    def _jit(f):  # type: ignore[misc]
        """Identity decorator — no JIT when JAX is unavailable."""
        return f

    JAX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pure inner step function
# ---------------------------------------------------------------------------

@_jit
def _gmm_step_inner(
    csm_opening,
    locked_in_rate,
    year_fraction,
    units_consumed,
    units_remaining_opening,
    fcf_change_non_financial,
    lc_opening,
    actual_outgo,
    total_remaining_outgo,
    bel_current,
    risk_adjustment,
):
    """
    Pure arithmetic core for one GMM period.

    All inputs are scalars (Python float / JAX 0-d array).  Under @jax.jit
    these are traced as abstract values and compiled to XLA; under the NumPy
    fallback they execute as ordinary Python arithmetic.

    Parameters
    ----------
    csm_opening : float
        CSM balance at the start of the period.
    locked_in_rate : float
        Annual discount rate frozen at contract group inception.
    year_fraction : float
        Period length in years (from ProjectionPeriod.year_fraction).
    units_consumed : float
        Coverage units consumed this period.
    units_remaining_opening : float
        Coverage units remaining at the start of this period (before
        consuming units_consumed). Must be > 0 — validated by caller.
    fcf_change_non_financial : float
        Non-financial FCF change for future service (positive = worsened).
    lc_opening : float
        Loss component balance at the START of this period, before any
        onerous excess generated in this same step is added.
    actual_outgo : float
        Expected cash outgo this period.
    total_remaining_outgo : float
        Total expected outgo from this period to end. Must be > 0.
    bel_current : float
        Current-period BEL (used for LRC assembly only).
    risk_adjustment : float
        Risk Adjustment this period (used for LRC assembly only).

    Returns
    -------
    tuple of 8 scalars (JAX 0-d arrays or plain floats):
        (0)  csm_accretion
        (1)  csm_adjustment_non_financial
        (2)  csm_release
        (3)  csm_closing
        (4)  loss_component_opening_after_addition
        (5)  loss_component_release
        (6)  loss_component_closing
        (7)  lrc
    """
    # ------------------------------------------------------------------
    # CSM arithmetic
    # ------------------------------------------------------------------
    accretion         = csm_opening * locked_in_rate * year_fraction
    csm_after_acc     = csm_opening + accretion
    csm_after_adj_raw = csm_after_acc - fcf_change_non_financial

    # Floor at zero; report onerous excess for loss component.
    onerous_excess = jnp.where(csm_after_adj_raw < 0.0, -csm_after_adj_raw, 0.0)
    csm_after_adj  = jnp.maximum(csm_after_adj_raw, 0.0)
    applied_adj    = csm_after_acc - csm_after_adj  # always >= 0

    release_fraction = jnp.minimum(units_consumed / units_remaining_opening, 1.0)
    csm_release      = csm_after_adj * release_fraction
    csm_closing      = jnp.maximum(csm_after_adj - csm_release, 0.0)

    # Sign convention matches CsmTracker: negative = CSM reduced by FCF change.
    csm_adj_reported = -applied_adj

    # ------------------------------------------------------------------
    # Loss component arithmetic
    # ------------------------------------------------------------------
    # Onerous excess from this period's CSM step is added to the opening
    # balance before the release calculation — replicating the sequence:
    #   lc_tracker.add_onerous_excess(onerous_excess)
    #   lc_result = lc_tracker.step(...)
    lc_after_addition = lc_opening + onerous_excess

    lc_rf      = jnp.minimum(actual_outgo / total_remaining_outgo, 1.0)
    lc_release = jnp.where(lc_after_addition > 0.0, lc_after_addition * lc_rf, 0.0)
    lc_closing = jnp.maximum(lc_after_addition - lc_release, 0.0)

    # ------------------------------------------------------------------
    # Balance sheet assembly
    # ------------------------------------------------------------------
    lrc = bel_current + risk_adjustment + csm_closing

    return (
        accretion,          # (0) csm_accretion
        csm_adj_reported,   # (1) csm_adjustment_non_financial
        csm_release,        # (2) csm_release
        csm_closing,        # (3) csm_closing
        lc_after_addition,  # (4) loss_component_opening (after onerous excess)
        lc_release,         # (5) loss_component_release
        lc_closing,         # (6) loss_component_closing
        lrc,                # (7) lrc
    )
