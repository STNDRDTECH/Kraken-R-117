---
name: Counterfactual shadow boundary
description: Durable authority and lifecycle rules for Kraken-R counterfactual diagnostics.
---

Counterfactual shadow intent is embedded in the exact grounded request before
prediction sealing. The durable intent claim must occur before checking whether
primary execution already exists; this ordering makes races fail conservatively
and ensures every accepted execution follows the persisted intent.

**Why:** A status-check-then-claim sequence is raceable. Claiming first turns the
existing delivery ledger into a one-shot ancestry anchor, while the later
execution attestation covers the exact request-bound intent. Exact historical
reconstruction remains valid; substituted post-outcome history does not.

**How to apply:** Keep structural shadows diagnostic and unmeasured. Existing
verified execution receipts may prove that bounded work ran and report measured
resources, but cannot prove a semantic counterfactual effect without a
verifier-checkable transformation witness. Cold replay must receive its trusted
executor identity out of band, never from the serialized record itself. Shadow
diagnostics never grant or veto grounded adaptive eligibility.