---
name: GitHub publication verification
description: Reliable verification pattern for authenticated GitHub Contents API publication
---

GitHub Contents API updates can complete even when the surrounding durable client reports an interrupted or incomplete callback replay. Treat the client result as unconfirmed until the remote ref, required blobs, ancestry, and a fresh checkout have been checked independently.

**Why:** A publication attempt reported callback replay termination, but the remote branch had already advanced and contained the intended files. Trusting the client error alone would have caused an unnecessary second publication attempt.

**How to apply:** After Contents API writes, read the branch ref directly, compare required Git blob hashes, verify protected refs and ancestry, then clone the published branch and run its own validation commands.