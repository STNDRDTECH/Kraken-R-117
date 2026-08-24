# Round 7 publication verification

`ROUND_7_PUBLICATION_MANIFEST.json` records the complete candidate artifact
set expected on the isolated `round-7-controlled-llm` ref, including the
original Rounds 1–3 files, Rounds 4–7 additions, and the current signal-lineage
implementation.

Run the verifier from a fresh checkout of that ref:

```bash
python scripts/verify_round_7_publication.py --published-checkout
```

It fails if any recorded artifact is absent or differs from the approved Git
blob, if that isolated GitHub checkout is not directly parented by its recorded
accepted Round 6 commit, or if the Kraken-R test suite or JSON CLI validation
fails. The manifest records the equivalent local and isolated-publication
Round 6 object IDs because they belong to separate repositories. With
`--published-checkout`, the command verifies the isolated GitHub clone directly.
When invoked elsewhere, it creates a temporary fresh clone of the published ref
first; inability to clone or verify that ref is an error, never a skipped
lineage check.

The verifier was run against a fresh Round 7 checkout after the restoration:

```text
artifacts: 33 verified
lineage: direct Round 6 parent verified
pytest: 96 passed
python -m kraken_r --json: ok
```