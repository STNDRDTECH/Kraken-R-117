---
name: Counterfactual shadow boundary
description: Durable authority and lifecycle rules for Kraken-R counterfactual diagnostics.
---

Counterfactual shadows are created only inside the canonical Round 5 ordering
after primary commitment and before grounded execution. They are immutable
diagnostic records, not alternate work paths or evidence.

**Why:** Offline records cannot independently prove wall-clock ordering without
adding a trusted clock, signer, or mutable ledger. Kraken-R deliberately avoids
creating those new authorities, so temporal safety comes from the bounded
orchestration boundary and hash-bound lineage.

**How to apply:** Keep shadows free of executable requests and authority. Bind
every shadow to the primary task/request/cognition boundary, change exactly one
validated contributor, replay without live calls, and allow the diagnostic only
to preserve or withhold eligibility already established by grounded outcome
learning.