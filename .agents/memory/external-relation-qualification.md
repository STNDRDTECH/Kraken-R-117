---
name: External relation qualification
description: Durable fail-closed rule for turning attested external material into candidate claim revisions.
---

External relation qualification is intentionally narrow: normalized quoted
material must equal the complete signed source content, then equal either the
complete claim assertion or one complete declared falsification discriminator.
Substring matches, extracted spans, contextualized statements, and material
with both dispositions cannot revise a claim.

**Why:** Substring qualification lets negated, quoted, modal, or disputed
passages inherit the opposite semantic meaning while still containing the
target text. That would let exact quote presence masquerade as independent
semantic validation.

**How to apply:** Preserve whole-source deterministic reconstruction during
live validation and replay. If richer documents need support later, add a
separately specified signed span/semantic grammar with equivalent adversarial
coverage rather than weakening this guard.