# Kraken-R foundation

This is an inspectable, candidate-only foundation. It is intentionally not a
second runtime.

## Contents

- `kraken_r/contracts.py` — immutable domain-agnostic vocabulary records.
- `kraken_r/architecture_registry.json` — living census and authority map.
- `kraken_r/registry.py` — read-only loader and structural validator.
- `kraken_r/validate.py` — standalone validator command.
- `kraken_r/cycle.py` — deterministic, in-memory candidate lifecycle fixture.
- `kraken_r/replay.py` — read-only recorded-execution replay adapter.
- `kraken_r/constitution.json` — machine-readable constitution metadata
  validated alongside the registry.
- `kraken_r/architecture_registry.schema.json` — machine-readable JSON schema.
- `docs/kraken_r/CONSTITUTION.md` — authority, evidence, lifecycle, and
  preservation rules.
- `docs/kraken_r/CATEGORIES.md` — 16 census areas and planned mechanisms.
- `docs/kraken_r/SALVAGE_DECISIONS.md` — retained artifact and nervous-system
  salvage decisions.

## Validate independently

From the project root:

```text
python -m kraken_r --json
pytest -q \
  tests/test_kraken_r_foundation.py \
  tests/test_kraken_r_cycle.py \
  tests/test_kraken_r_replay.py
```

The command reads the bundled JSON, imports the isolated package, and executes
the bounded success fixture in memory. The focused suite contains the accepted
37-test baseline. It does not import `rogal_core`, start a workflow, contact an
LLM, open a live store, or write runtime state.

The cycle can also be exercised directly:

```python
from kraken_r import Objective, CycleMode, run_constitutional_cycle

trace = run_constitutional_cycle(
    Objective("demo", "observe one bounded behavior"),
    mode=CycleMode.SUCCESS,
)
assert trace.stop_decision.outcome == "stop"
```

Recorded observations can be replayed without live runtime access:

```python
from kraken_r import (
    Objective,
    RecordedExecution,
    RecordedExecutionMode,
    replay_recorded_execution,
)

objective = Objective(
    "recorded-demo",
    "inspect one recorded observation",
    provenance={"transaction_id": "recorded-demo-tx"},
)
record = RecordedExecution.fixture(
    RecordedExecutionMode.SUCCESS,
    transaction_id="recorded-demo-tx",
    objective_id=objective.objective_id,
)
trace = replay_recorded_execution(objective, record)
assert trace.execution.execution_id == record.execution_id
```