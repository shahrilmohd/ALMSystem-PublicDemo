"""
engine/matching_adjustment/
============================
Matching Adjustment (MA) computation layer for BPA portfolios.

Modules
-------
fundamental_spread  : FS table lookup and per-asset MA contribution (DECISIONS.md §23)
eligibility         : Four-condition eligibility assessment (DECISIONS.md §22)
ma_calculator       : MACalculator orchestrator → MAResult (DECISIONS.md §21–23)

This package is a pure library.  It takes DataFrames in and returns MAResult out.
It has no knowledge of DeterministicRun, ResultStore, or the projection loop.
Step 20 wires it into _calibrate_ma() on DeterministicRun.
"""
