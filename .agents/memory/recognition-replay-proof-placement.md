---
name: Recognition replay proof placement
description: Why full recognition ancestry proofs stay outside focused model context.
---

Keep complete recognition proofs in structurally hashed trace records. Put only
a compact canonical envelope and proof commitment in focused model context.
Validate the complete proof first, then derive and compare the compact context
before deterministic capability selection or replay.

**Why:** Full ancestry, episode, ambiguity, and expansion proofs exceed the
cognition kernel's deliberately small aggregate-item and per-string model
context limits. Embedding them in focused context either breaks valid replay or
leaks unnecessary detailed memory into provider input.

**How to apply:** Any future recognition field must be derived canonically from
the separately bounded proof, included in the proof commitment, and checked
against the exact downstream payload limits using the emitted serialization.