# Round 8 grounded execution publication verification

Round 8 is published only on `round-8-grounded-execution`. The accepted
`round-7-controlled-llm` branch and `main` are not update targets.

The Round 8 manifest binds the candidate's constitutive source, tests,
documentation, bounded runner, grounded execution adapter, and Round 8 tests
to Git blob hashes. The verification script checks every listed hash, rejects
runtime `.rogal` state and database sidecars anywhere in the published tree,
enforces that the complete published tree is unchanged from Round 7 except for
declared Round 8 files, runs the cumulative Kraken-R tests and standalone
validator, and—in a clean published checkout—requires the published Round 7
commit to be the direct parent.

Run:

```bash
python scripts/verify_round_8_publication.py
python scripts/verify_round_8_publication.py --published-checkout
```

The second command always clones the declared canonical GitHub branch at depth
two and repeats
the hash, runtime-state, test, validator, and direct-parent checks without
using the local working tree.

## Verification result

The independent fresh-clone verification passed: all 34 manifest artifacts
matched, the published commit had the accepted Round 7 GitHub commit as its
only parent, 111 cumulative Kraken-R tests passed, and the standalone validator
reported `PASS (68 mechanisms, 16 categories, 11 planned)`. The recursive
publication tree contained 48 files and matched the accepted Round 7 tree plus
the 36 declared Round 8 publication files, with the two required
runtime/session removals. No runtime state remains in the published tree.
`main` and the accepted Round 7 ref were not modified.