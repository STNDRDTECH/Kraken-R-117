---
name: Grounded child runtime availability
description: Environment boundary observed when validating grounded execution from clean checkouts
---

The parent test environment and the interpreter selected for the isolated grounded child are separate. A clean checkout can run the parent pytest suite while the selected child interpreter has no pytest dependency, producing a verified `infrastructure_setup_failure` with zero tests.

**Why:** This is an environment prerequisite, not evidence that the candidate reducer or its replay semantics are wrong. Treating setup failure as task failure would violate the grounded epistemic boundary.

**How to apply:** During fresh-checkout validation, distinguish pure substrate tests from grounded execution prerequisites. Keep grounded-path tests strict when the child dependency is present, and skip only those tests when the verifier explicitly reports unavailable child pytest; the runtime-reliability work should address installation/environment provisioning separately.