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
- `kraken_r/physiology.py` — immutable, deterministic advisory regulation over
  explicit internal-condition snapshots; it can only inhibit a candidate action.
- `kraken_r/plastic_routing.py` — immutable settlement/evidence-gated candidate
  route preference; it never dispatches, persists, executes, or selects goals.
- `kraken_r/llm_adapter.py` — strict, injected-provider proposal/reasoning leaf;
  output remains declared-only and structurally replayable.
- `kraken_r/controlled_evaluation.py` — base/mediated/reset held-out comparison
  with fixed provider/model/prompt-budget controls and no performance claim.
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
  tests/test_kraken_r_nervous_system.py \
  tests/test_kraken_r_physiology.py \
   tests/test_kraken_r_plastic_routing.py \
   tests/test_kraken_r_llm_adapter.py \
   tests/test_kraken_r_controlled_evaluation.py
```

The command reads the bundled JSON, imports the isolated package, and executes
the bounded fixture matrix in memory. The focused suite retains the accepted
Rounds 1–3 coverage, adversarial Round 4 signal coverage, Round 5 physiology
regime/ablation/replay coverage, and Round 6 settlement-grounded route
plasticity coverage. It does not import `rogal_core`,
start a workflow, contact an LLM, open a live store, or write runtime state.
The Round 7 provider is deterministic fixture-only unless a caller explicitly
injects a configured provider.

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

Bounded physiology is caller-provided internal condition data, not a live
controller. It can only inhibit the already candidate-owned action before
execution, never create evidence or choose a goal:

```python
from kraken_r import (
    Objective,
    PhysiologySnapshot,
    run_constitutional_cycle,
)

objective = Objective(
    "physiology-demo",
    "show a safe candidate physiology stop",
    provenance={"transaction_id": "physiology-demo-tx"},
)
snapshot = PhysiologySnapshot(
    "physiology-demo-critical",
    "physiology-demo-tx",
    objective.objective_id,
    "physiology-demo-state-4",
    4,
    contradiction_density=0.9,
    protected_reserve=0.05,
)
trace = run_constitutional_cycle(objective, physiology=snapshot)
assert trace.execution.status == "not_observed"
assert trace.evidence == ()
```

Settlement-grounded route preference is an explicit post-settlement reducer,
not a live router. It can only credit grounded observed execution evidence:

```python
from kraken_r import (
    Objective,
    RouteTopology,
    SettlementRouteRecord,
    apply_settlement_learning,
    run_constitutional_cycle,
    select_candidate_route,
)

objective = Objective(
    "routing-demo",
    "produce one settled observation before candidate route learning",
    provenance={"transaction_id": "routing-demo-tx"},
)
cycle = run_constitutional_cycle(objective)
topology = RouteTopology.fixture()
selection = select_candidate_route(
    topology,
    "candidate-work",
    transaction_id=cycle.transaction_id,
    objective_id=objective.objective_id,
    task_state_id=cycle.states[4].state_id,
    task_state_version=cycle.states[4].version,
)
record = SettlementRouteRecord(
    "routing-demo-record",
    selection,
    cycle,
    provenance={
        "transaction_id": cycle.transaction_id,
        "objective_id": objective.objective_id,
        "task_state_id": cycle.states[4].state_id,
        "task_state_version": cycle.states[4].version,
        "route_id": selection.route_id,
        "settlement_id": cycle.settlement.settlement_id,
        "evidence_ids": tuple(item.evidence_id for item in cycle.evidence),
    },
)
topology, trace = apply_settlement_learning(topology, record)
assert trace.disposition == "accepted"
```

The model adapter is proposal-only and does not turn a model answer into
evidence or learning:

```python
from kraken_r import (
    CandidateModelContext,
    CycleMode,
    FixtureModelProvider,
    ModelAdapter,
    Objective,
    run_constitutional_cycle,
)

trace = run_constitutional_cycle(
    Objective("llm-demo", "suggest a bounded candidate route"),
    mode=CycleMode.INSUFFICIENT_EVIDENCE,
)
context = CandidateModelContext.from_cycle_trace(
    trace,
    context_id="llm-demo-context",
    allowed_route_ids=("path-alpha", "path-beta"),
    route_scores=(("path-alpha", 0.50), ("path-beta", 0.50)),
)
provider = FixtureModelProvider(
    lambda _: '{"proposal":"inspect alpha","reasoning":"declared only","route_hint":"path-alpha"}'
)
result = ModelAdapter(provider).invoke(
    context,
    request_id="llm-demo-request",
    prompt="Return only the bounded proposal JSON.",
    model_id="fixture-model",
)
assert result.proposal.declared_only
assert not hasattr(result.proposal, "evidence")
```