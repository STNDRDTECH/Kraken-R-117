---
name: Stage 10.8 explicit inactivity
description: Defines the boundary between a truly inactive dynamical tick and a non-creditable tick that must retain adaptive topology.
---

An automatic, no-credit candidate-route move toward neutral preference is allowed
only when the caller explicitly supplies an empty event batch. A tick containing
signals, observations, withheld settlements, stale outcomes, contradictory
outcomes, or physiology-inhibited outcomes is not inactive and must not change
adaptive topology by that mechanism.

**Why:** “No successful adaptive audit” conflates an absence of activity with an
activity that was deliberately withheld from adaptation. Treating both as
inactivity would let non-evidentiary outcomes indirectly reshape candidate
topology, violating the constitutional boundary.

**How to apply:** Keep the reducer check tied to an empty ordered event batch,
and test any future no-credit or decay behavior against both an empty tick and
a non-empty withheld/inhibited tick. Do not turn this decay into settlement,
audit, update-budget, connection, tactic, or learning credit.