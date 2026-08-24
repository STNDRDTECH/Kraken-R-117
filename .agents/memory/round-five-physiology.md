---
name: Bounded candidate physiology
description: Authority and replay constraints for the candidate-only physiology layer.
---

Candidate physiology must consume only explicit immutable, transaction- and
task-state-bound condition snapshots. It may classify a bounded internal regime
and conservatively inhibit an already candidate-authorized action before
execution; it must not authorize a new action.

**Why:** Legacy pressure, HOP, homeostasis, queue, and resource mechanisms have
live mutable state, event routing, side effects, or lifecycle ownership. Reusing
them would turn an observability/regulation concept into competing runtime
authority and break deterministic causal replay.

**How to apply:** Preserve pure deterministic replay from the same snapshot.
Keep any future adaptation advisory-only: no evidence, settlement, learning,
goal selection, persistence, subscriptions, live sensors, or autonomous loops
without a separately approved authority migration.