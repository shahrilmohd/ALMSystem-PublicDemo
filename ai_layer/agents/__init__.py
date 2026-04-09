"""
agents — specialist AI agents for the ALM system.

Phase 2 (conventional products):
  RunAnalystAgent    — explains BEL/TVOG/cash flows for a given run
  ConfigAdvisorAgent — proposes revised RunConfig assumptions
  ReviewerAgent      — cross-checks proposed configs for actuarial consistency
  ModellingAgent     — explains how the engine calculates BEL/TVOG/cash flows
                       (injects full engine source code into the system prompt)

Phase 3 stubs (raise NotImplementedError until Phase 3 Step 23):
  IFRS17Agent        — CSM/RA/LRC/LIC explanation
  SolvencyIIAgent    — SCR stress, MA offset, capital adequacy
  BPAAgent           — matching adjustment, eligibility, fundamental spread
"""
