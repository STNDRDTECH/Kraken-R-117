---
name: Bounded plasticity resilience
description: Rules for invalidating learned candidate preferences without creating new authority or replay gaps.
---

Adaptive recovery may only reverse a retained prior candidate update when a
later, independently reverified grounded task failure supplies the basis. The
reducer restores the nearest retained pre-update checkpoint, retains the
original and invalidating record identities, records the invalidation audit,
and has a deterministic replay surface.

**Why:** A later failure must be able to weaken contaminated, stale, or
non-generalizing reinforcement, but retrying or silently deleting old credit
would allow replay and identity-reuse errors. Infrastructure and ambiguous
outcomes are non-creditable and therefore cannot trigger recovery.

**How to apply:** Keep recovery bounded by retained checkpoints and generation
limits. Require exact grounded lineage for the invalidating record, reject
stale selections rather than repairing them implicitly, and preserve route
caps and tactic escape as candidate-only anti-lock-in controls. Do not use
pressure, model output, or undeclared evidence as a recovery trigger.