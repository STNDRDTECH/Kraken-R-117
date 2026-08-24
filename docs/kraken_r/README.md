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
- `kraken_r/nervous_system.py` — bounded deterministic propagation of canonical
  declared signals; no runtime subscription, persistence, or adaptive routing.
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
  tests/test_kraken_r_replay.py \
  tests/test_kraken_r_nervous_system.py
```

The command reads the bundled JSON, imports the isolated package, and executes
the bounded fixture matrix in memory. The focused suite retains the accepted
Rounds 1–3 coverage and adds adversarial Round 4 signal coverage. It does not
import `rogal_core`, start a workflow, contact an LLM, open a live store, or
write runtime state.

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

Bounded signal propagation can only influence a candidate action before
execution. It cannot create evidence:

```python
from kraken_r import Objective, make_bound_signal, run_constitutional_cycle

objective = Objective(
    "signal-demo",
    "show an uncertainty signal inhibiting a candidate action",
    provenance={"transaction_id": "signal-demo-tx"},
)
uncertainty = make_bound_signal(
    "signal-demo-uncertainty",
    "candidate.uncertainty",
    transaction_id="signal-demo-tx",
    objective_id=objective.objective_id,
    task_state_id="signal-demo-state-4",
    task_state_version=4,
    source="demo-input",
    cause="demo-uncertainty",
    priority=5,
)
trace = run_constitutional_cycle(objective, signals=(uncertainty,))
assert trace.decision.outcome == "insufficient_evidence"
assert trace.evidence == ()
```

Candidate propagation requires explicit source/cause identities and an integer
priority. Within a tick, higher priority delivers first; equal priorities sort
by signal ID. A topic absent from the static rule graph is rejected before
delivery.