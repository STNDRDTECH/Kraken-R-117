---
name: Settlement-grounded plastic routing
description: Durable authority and safety boundary for Round 6 candidate route preference.
---

Route plasticity is a pure, caller-owned, candidate-only reducer. Credit is
allowed only from a validated immutable constitutional trace whose settlement,
grounded observed execution evidence, transaction, objective, authorized task
state/version, route selection, and provenance all agree.

For a grounded trace, the route record must also carry the sealed execution
request, the verified execution record, and an independent verifier. Credit
requires re-verifying that proof against the same authorized state and matching
the record hash to both the cycle observation and every evidence item; a
`grounded_execution` source label is never sufficient.

**Why:** Legacy cascade/plasticity, pathway, topology, and online-learning
mechanisms use mutable stores, clock-driven behavior, broad credit assignment,
signals, confidence, or background authority. Those paths can self-reinforce
without constitutional evidence and cannot be replayed safely.

**Why:** Provenance labels can be copied into an otherwise valid immutable
trace. Reverification at the reducer boundary prevents a direct caller from
turning a claimed grounded source into route credit without the independently
verifiable observation it claims to represent.

**How to apply:** Keep route selection advisory and in-memory. Withhold or
reject signals, confidence, physiology, raw claims, contradiction, insufficient
evidence, malformed provenance, stale selections, and non-settled results.
Use fixed bounded route-local changes only. A reset must advance topology
generation; retain all accepted settlement identities within a hard per-
generation update budget rather than evicting them and risking duplicate credit.
For grounded learning, bind the exact trace and its original verified inputs;
do not substitute a same-objective execution, transaction, or record hash.