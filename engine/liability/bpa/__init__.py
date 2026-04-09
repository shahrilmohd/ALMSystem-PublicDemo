"""
BPA liability classes — Phase 3, Step 18.

Class hierarchy:
    BaseLiability (abstract)
    └── MultiDecrementLiability (abstract)
        ├── InPaymentLiability   — in-payment pensioners
        ├── DeferredLiability    — deferred members (4-decrement)
        ├── DependantLiability   — contingent dependants (convolution)
        └── EnhancedLiability    — impaired lives (age-rated, subclasses InPaymentLiability)

See DECISIONS.md §15, §18, §19, §25, §26, §27.
"""
