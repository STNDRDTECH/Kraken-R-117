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
- `kraken_r/grounded_execution.py` — sealed, disposable-workspace candidate test
  observation with explicit limits, reproducible child-runtime provenance,
  executor attestation, independent verification, and structural replay.
- `kraken_r/interactions.py` — stateless validation that existing bounded
  modules compose without creating a new runtime authority.
- `kraken_r/task_integrity.py` — immutable original-task preservation,
  bounded structured interpretation and beliefs, information-loss accounting,
  and selective proposal-only adversarial review.
- `kraken_r/dynamical_substrate.py` — explicit bounded event batches reduced
  into immutable, replayable fast/medium/slow candidate state; it is not a
  scheduler, daemon, persistent cognitive store, or controller.
- `kraken_r/metastability.py` — bounded, deterministic full-versus-ablation
  measurements over caller-owned dynamical ticks; reports are disposable
  observations, never evidence, credit, promotion, or runtime authority.
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
   tests/test_kraken_r_controlled_evaluation.py \
   tests/test_kraken_r_grounded_execution.py \
   tests/test_kraken_r_interactions.py \
   tests/test_kraken_r_adaptive_substrate.py \
     tests/test_kraken_r_task_integrity.py \
     tests/test_kraken_r_dynamical_substrate.py \
     tests/test_kraken_r_metastability.py
```

The command reads the bundled JSON, imports the isolated package, and executes
the bounded fixture matrix in memory. The focused suite retains the accepted
Rounds 1–3 coverage, adversarial Round 4 signal coverage, Round 5 physiology
regime/ablation/replay coverage, and Round 6 settlement-grounded route
plasticity coverage. It does not import `rogal_core`,
start a workflow, contact an LLM, open a live store, or write runtime state.
The Round 7 provider is deterministic fixture-only unless a caller explicitly
injects a configured provider.

## Bounded grounded execution

The fixture cycle remains the default for deterministic unit tests. When a
caller needs a real local observation, it must first provide a candidate-owned
`run_bounded_pytest` action and the exact authorized state. The bounded adapter
accepts only declared relative text files and test paths, runs them in a
disposable child workspace under explicit resource limits, and emits an
attested record. A separate verifier recomputes the sealed request/output
hashes and derives the outcome from test facts; executor output and model
claims are not trusted as evidence by themselves.

```python
from kraken_r import (
    GroundedExecutionExecutor,
    GroundedDeliveryLedger,
    GroundedExecutionRequest,
    GroundedExecutionVerifier,
    Objective,
    TaskState,
    make_grounded_action,
    run_constitutional_cycle,
)

objective = Objective(
    "grounded-demo",
    "run one bounded real test",
    provenance={"transaction_id": "grounded-demo-tx"},
)
action = make_grounded_action(objective.objective_id)
state = TaskState(
    "grounded-demo-state-5", objective.objective_id, 5, "authorized",
    values={"action_id": action.action_id},
)
request = GroundedExecutionRequest(
    "grounded-demo-request", "grounded-demo-tx", objective.objective_id,
    state.state_id, state.version, action,
    {
        "subject.py": "def add(a, b):\n    return a + b\n",
        "test_subject.py": (
            "from subject import add\n\n"
            "def test_add():\n    assert add(20, 22) == 42\n"
        ),
    },
    ("test_subject.py",),
)
ledger = GroundedDeliveryLedger("/trusted-runtime/grounded-receipts.json")
executor = GroundedExecutionExecutor(delivery_ledger=ledger)
record = executor.execute(request, authorized_state=state)
verifier = executor.verifier()
verified = verifier.verify(
    record, request=request, authorized_state=state,
)
trace = run_constitutional_cycle(
    objective,
    grounded_execution=verified,
    grounded_request=request,
    grounded_verifier=verifier,
)
assert trace.evidence[0].grade.value == "grounded"
```

This is an execution-source ablation, not a model evaluation or performance
claim. A timeout, failed setup, malformed test collection, zero-test run,
stale state, workspace escape, or invalid attestation produces no grounded
evidence or learning.

Records carry an Ed25519 signature, but the public key inside a record is never
trusted by itself. Persist the request, authorized state, record, and a pinned
`TrustedExecutorIdentity` through their JSON serializers; a restarted verifier
must be constructed with the external trust anchor:

```python
trusted = executor.trusted_executor()  # distribute independently of the record
fresh = GroundedExecutionVerifier.from_record(
    GroundedExecutionRecord.from_json(record.to_json()),
    trusted_executor=trusted,
)
```

## Stage 10 adaptive substrate and Stage 11.1 causal integrity

`kraken_r.adaptive_substrate` adds bounded ordinary adaptation as immutable,
caller-owned candidate state.  It requires a sealed, independently verified
grounded constitutional settlement before it can adjust a selected route, form
or weaken a limited candidate connection, switch a predeclared tactic, decay or
recover a route, or retain a rollback checkpoint.  It has no store, daemon,
event subscription, provider loop, controller, dispatch, execution, or
promotion authority.

```python
from kraken_r import AdaptiveState, HomeostaticSnapshot, run_orzhaal_experiment

state = AdaptiveState.fixture()
pressure = HomeostaticSnapshot(
    "pressure-1", "tx-1", "objective-1", "objective-1-state-4", 4,
    contradiction=0.3, uncertainty=0.2, repeated_failure=0.0,
    novelty=0.4, resource_expenditure=0.2,
)
assert all(signal.payload["advisory_only"] for signal in pressure.signals())

# Orzhaal only compares a disposable immutable fork.  The result cannot
# promote itself or mutate the canonical candidate state.
result = run_orzhaal_experiment(state, experiment_id="compare-rule-1")
assert result.promotable is False and result.canonical_mutation is False
```

All ordinary changes require exact grounded trace, request, authorized-state,
record, evidence, and provenance bindings. Model declarations, confidence,
signals, and physiology never earn learning credit or create topology.
Stage 11.1 makes the evidence authority split explicit: operational evidence can
shape only a returned, short-horizon `AdvisoryCognition`, while grounded
evidence remains mandatory for every durable adaptive change. The advisory
result reads bounded tactic scores and non-retired connection weights but is
non-dispatchable and cannot authorize execution, evidence, settlement, credit,
or promotion.

Adaptive audits now retain immutable route-causal lineage. Invalidation requires
the exact target audit envelope in the request hash plus the signed
child-runtime witness, rather than accepting a parent ID alone; it also replays
only the checkpoint's current live branch after rollback. Rollback is restricted
to the latest trusted checkpoint and keeps all consumed identities, preventing
duplicate credit. Candidate connections weaken into
recoverable dormant state, can recover under later useful grounded evidence,
and become terminal only through a separate explicit grounded retirement.
Higher-order circuits, autonomous recursion, workers, controllers, alternate
loops, and all other Stage 11 mechanisms remain unstarted and excluded.

`GroundedDeliveryLedger(path)` persists bounded expiry-stamped active receipts
and non-expiring identity tombstones. It rejects duplicate execution, evidence,
settlement, and learning delivery after restart even when an active receipt has
expired; expiry never permits identity reuse. It grants no controller authority.
The host runner measures the applied resource limits and confirms the sandbox
process group is gone before marking cleanup provenance as verified.
Stage 10.5 additionally selects an explicit local Python runtime, requires
pytest inside the child, binds its declared interpreter/pytest facts to that
selection, and rejects absolute, traversal, or symlink-escaping test paths.
Grounded observations
are classed as `task_success`, `task_failure`, setup/infrastructure failure,
timeout/resource failure, execution failure, contradiction, or insufficient
evidence. Only the first two classes can become learning credit; every
adaptive audit retains the exact creditable class and child-runtime
fingerprint. Other Stage 11 mechanisms remain unstarted.

## Stage 10.6 plasticity resilience

Stage 10.6 keeps adaptation domain-agnostic and immutable while adding bounded
recovery from bad reinforcement. A later independently verified grounded task
failure may invalidate one retained ordinary update and restore its nearest
retained pre-update checkpoint; both the original and invalidating record
identities remain consumed. Route preference is capped below the generic route
maximum, stale selections are rejected instead of being repaired implicitly,
and a bounded tactic streak forces a deterministic alternate tactic. Advisory
physiology remains non-evidentiary, and setup/infrastructure failures cannot
trigger recovery, topology changes, or learning credit. Stage 11.1 adds only
the causal-integrity boundaries described above; other Stage 11 mechanisms
remain excluded.

## Stage 10.7 task integrity

`kraken_r.task_integrity` is a pure inspection boundary for retaining the
original caller task throughout structured interpretation, bounded competing
hypotheses, a declared plan, candidate results, and adversarial review. It does
not create an action, provider request, execution, evidence, settlement,
learning update, authority, topology change, adaptive state, durable memory, or
runtime wiring.

```python
from kraken_r import (
    BeliefState,
    CandidateResult,
    OriginalTask,
    RequirementKind,
    TaskHypothesis,
    TaskPlan,
    TaskRequirement,
    TaskSpecification,
    replay_task_integrity,
)

task = OriginalTask(
    "integrity-demo",
    "Preserve all requirements and verify independently.",
    "caller",
    {},
)
specification = TaskSpecification.from_task(
    "integrity-demo-spec",
    task,
    (
        TaskRequirement("ask", RequirementKind.ASK, "Preserve all requirements."),
        TaskRequirement("evidence", RequirementKind.REQUIRED_EVIDENCE, "Verify independently."),
    ),
    "Keep both clauses explicit.",
)
hypothesis = TaskHypothesis(
    "integrity-demo-hypothesis",
    specification.specification_id,
    "The candidate can preserve both clauses.",
    specification.requirement_ids,
)
belief = BeliefState("integrity-demo-belief", specification.specification_id, (hypothesis,))
plan = TaskPlan(
    "integrity-demo-plan",
    specification.specification_id,
    (hypothesis.hypothesis_id,),
    specification.requirement_ids,
    ("Preserve clause lineage.",),
)
result = CandidateResult(
    "integrity-demo-result",
    specification.specification_id,
    plan.plan_id,
    (hypothesis.hypothesis_id,),
    specification.requirement_ids,
)
trace = replay_task_integrity(task, specification, belief, plan, result)
assert trace.to_dict()["creates_evidence"] is False
assert trace.to_dict()["changes_adaptive_state"] is False
```

Loss accounting reports omissions, declared compression, underweighting, and
unsupported reverse dependencies without repairing or mutating the candidate.
Angel gets only requirement coverage, Nemesis gets only declared hypothesis
status/dependencies, and Antimetabole gets only conclusion-to-claim structure.
Their bounded findings and verification questions are proposal-only and require
later independent grounding. Ambiguity, infrastructure/setup, timeout/resource,
execution, contradiction, and insufficient-evidence remain non-creditable.
Stage 11 provider reputation/routing, probes, higher-order circuits, workers,
recursion, and automatic promotion remain excluded.

## Stage 10.8 dynamical substrate

`kraken_r.dynamical_substrate` composes the already-approved candidate
mechanisms into an explicit immutable transition:

```text
state at tick t + caller-owned events at tick t+1 -> state at tick t+1
```

There is no clock, scheduler, event subscription, worker, daemon, store,
subprocess, provider, primitive creation, model knowledge, or promotion path.
Every event batch is finite, ordered deterministically by its typed phase and
event identity, retained only in a bounded audit window, and replayed by passing
the same records to the same pure reducer.

Fast state contains only activation, inhibition, surprise, resource pressure,
delivered signal topics, and a bounded carried physiology cooldown. Medium state
wraps the existing immutable settlement-gated routes, connections, tactics,
decay, and rollback reducers. On a caller-supplied inactive tick, a previously
selected route may move one fixed small step toward neutral preference; this is
explicit traceable candidate turnover, not evidence, settlement credit, or a
learning update.
Slow state records a minimal coherence/recurrence observation only; it cannot
be used as an evidence source, controller, or learning credit. Endogenous
physiology is derived only from bounded candidate state and remains advisory.

```python
from kraken_r import (
    DynamicalEvent,
    DynamicalState,
    DynamicalTick,
    make_bound_signal,
    replay_dynamical_ticks,
)

state = DynamicalState.fixture()
urgency = make_bound_signal(
    "dynamical-demo-urgency",
    "candidate.urgency",
    transaction_id=state.transaction_id,
    objective_id=state.objective_id,
    task_state_id=state.task_state_id,
    task_state_version=state.task_state_version,
    source="demo",
    cause="declared-input",
)
ticks = (
    DynamicalTick(
        1,
        (
            DynamicalEvent.signal_event("dynamical-demo-signal", urgency),
            DynamicalEvent.observation_event(
                "dynamical-demo-observation", 0.50, 0.50
            ),
        ),
    ),
)
next_state, traces = replay_dynamical_ticks(state, ticks)
assert next_state.tick == 1
assert traces[0].adaptive_audits == ()
```

Prediction/observation mismatch is typed and bounded. Surprise, resource
pressure, homeostatic inhibition, decay, and rollback change the candidate
trajectory and bounded instrumentation rather than merely producing logs.
Declared signals, mismatch observations, physiology, coherence, reviews, and
non-creditable settlements never earn adaptive credit. Setup/infrastructure,
timeout/resource, execution, contradiction, insufficient-evidence, stale, and
inhibited outcomes retain provenance but cannot reshape topology, connections,
tactics, homeostasis, or learning.

Every public tick/state collection checks its fixed retention budget
incrementally before freezing it. Retained tick counters saturate at documented
finite bounds, and a candidate connection remains present at its minimum weight
floor rather than being removed implicitly. The physiology cooldown is carried
only as immutable fast state from one explicit caller-supplied tick to the next;
there is no timer, background controller, or hidden state.

## Stage 10.9 metastability experiments

`kraken_r.metastability` is a measurement layer over the Stage 10.8 pure
reducer, not a new dynamical runtime. It accepts a finite caller-owned tick
stream, starts every arm from the same immutable tick-zero state, and records
descriptive route diversity/entropy, dominant-route share, topology turnover,
plasticity, decay, rollback, inhibition, surprise, recurrence, coherence,
stagnation, resource pressure, grounded task-performance observations, and
regime transitions.

The six deterministic scenario factories are `stable`, `regime_change`,
`noisy_noncreditable`, `oscillatory`, `monopolizing`, and `stagnating`.
They make no claim that any one regime, entropy, topology, or threshold is
ideal. `operating_region` and `failure_modes` are bounded descriptive labels
for the supplied trace, not criticality targets or optimization objectives.

```python
from kraken_r import (
    DynamicalState,
    ExperimentScenario,
    compare_metastability,
    make_metastability_scenario,
)

initial = DynamicalState.fixture("metastability-demo")
ticks = make_metastability_scenario(
    initial, ExperimentScenario.OSCILLATORY, ticks=32
)
comparison = compare_metastability(
    initial, ticks, scenario=ExperimentScenario.OSCILLATORY
)
assert comparison.to_dict()["matched_budget"] is True
assert comparison.baseline.ablation.label == "full"
```

The comparison set has a full arm and one selectively neutralized arm each for
homeostasis, inactivity decay, action inhibition, mismatch surprise, and the
anti-monopoly route-selection control. Arms retain the same raw tick digest and
tick budget. A private comparison-only reducer path applies the same bounded
reducer logic to those raw records while retaining the original physiology
decision as the authority gate: an inhibition ablation may alter its measured
candidate-action state, but it can never allow an inhibited settlement or
rollback to receive credit. Anti-monopoly is an explicit route-choice comparison
and never changes the canonical topology cap.

No experiment report is evidence, a settlement, a learning update, an adaptive
audit, or a proposal for promotion. Only pre-existing independently verified
grounded `task_success`/`task_failure` settlement records can reach the
adaptive reducer. Setup, infrastructure, ambiguous, timeout, execution,
contradiction, insufficient-evidence, non-creditable, and physiology-inhibited
outcomes remain withheld. There are no daemons, schedulers, persistence,
subprocesses, providers, recursive modifications, consolidation, primitive
promotion, or hidden mutable state.

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