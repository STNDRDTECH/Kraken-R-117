# Kraken-R Constitution

**Status:** candidate-only foundation  
**Version:** 1.10.9
**Effective boundary:** documentation, standalone validation, deterministic fixture
execution, and independently verified bounded candidate execution observations

## Mission

Kraken-R is a domain-agnostic architectural direction for reasoning from
objectives through evidence-backed settlement and learning. The foundation
defines shared vocabulary, records the existing ROGAL architecture, and
executes one deterministic in-memory candidate cycle so the causal contracts
can be validated without guessing which mechanism is authoritative.

## Authority boundary

`rogal_core.daemon.ROGALDaemon` and
`rogal_core.autonomous_cycle.AutonomousCycle` remain the legacy reference
runtime. Existing EventBuses, routers, ledgers, homeostasis controllers,
executors, sandboxes, persistence stores, capability systems, and mutation
actuators remain unchanged and authoritative for their current behavior.

The `kraken_r` package is candidate-only. Its deterministic cycle,
recorded-execution replay, bounded signal propagation, bounded advisory
physiology, settlement-grounded route-preference reducer, immutable replayable
dynamical substrate, bounded metastability experiment observations, immutable task-integrity inspection, proposal-only LLM
adapter, narrow verified execution observer, and static interaction validator
have no daemon entrypoint, runtime writer, event subscription, general command
surface, router, ledger, homeostasis controller, or mutation actuator. Loading,
validating, or running them must not start ROGAL, contact an LLM, open live
stores, or change `.rogal/` runtime state.

## Canonical lifecycle

The future lifecycle is:

1. acquire an **Objective** and versioned **TaskState**;
2. form falsifiable **Hypothesis** records and a bounded **Plan**;
3. observe **Signal/Event** inputs and issue bounded **Action** requests;
4. return an **ExecutionResult** through a domain adapter;
5. attach **Evidence/GroundTruth** with provenance and an honest evidence grade;
6. record a **Decision** and compare it with observation in **Settlement**;
7. propose a **LearningUpdate** against a named capability, route, memory, or
   constraint;
8. govern **Capability/Authority**, **Mutation/Lineage**, and **Regression**
   before any future promotion.

The cycle is a bounded fixture, not an autonomous loop: it ends with an
explicit `Decision(outcome="stop")`.

## Evidence-first rules

- Declared intent is not execution evidence.
- A retrieved or referenced pattern is not proof that it caused an outcome.
- Evidence must identify its subject, source, provenance, and (when applicable)
  execution boundary.
- Confidence never outranks contradictory observed ground truth.
- A declared `Signal`/`Event` and an `ExecutionResult` are not evidence by
  themselves; evidence is created only from the cycle's observed execution
  fixture.
- A fixture observation remains operational evidence. `GROUNDED` evidence
  requires a sealed execution request, disposable-workspace observation,
  complete provenance and hashes, executor attestation, and separate verifier
  acceptance.
- A grounded child must declare facts that match the executor-selected Python
  and pytest runtime exactly. Only `task_success` and `task_failure` are
  creditable classes. Setup/infrastructure, timeout/resource, execution,
  contradiction, and insufficient-evidence classes are preserved in
  provenance but cannot create evidence, settlement learning, route credit,
  topology changes, tactic changes, or homeostatic pressure.
- Regression comparisons require a named baseline and evidence references.

## Authority and mutation rules

- Every authority is explicit and singular for its scope.
- Aliases in the canonical vocabulary do not create second buses, stores, or
  authorities.
- A mutation is a governed proposal with lineage, rollback information, and
  regression evidence; it is never an implicit write.
- Kraken-R cannot promote or wire itself during this phase.

## Preservation policy

Legacy, alternate, dormant, experimental, attached, and forgotten mechanisms
are preserved and classified in
`kraken_r/architecture_registry.json`. A `merge`, `quarantine`, or
`rebuild` disposition is a future decision marker, not permission to delete or
replace code now. Orzhaal Bubble and recursive stacks are explicitly retained.

## Planned capabilities

Metaplasticity, neuromodulation, reservoir-style dynamics, criticality control,
organizational engrams, offline consolidation, causal lesion/shadow
experiments, developmental specialization, hyperdimensional associative
state, and hierarchical learning levels are catalogued as planned-only. They
are not implemented, reachable, authoritative, or evidence-backed by this
foundation.

## Bounded replayable dynamical substrate

Stage 10.8 permits only an explicit, caller-owned, finite tick transition:
immutable candidate state and declared events at one tick produce a new
immutable candidate state at the next tick. It is not permission to start a
clock, scheduler, daemon, queue, subscriber, bus, store, worker, subprocess,
controller, provider, persistent cognitive memory, circuit/motif mechanism,
primitive promotion path, or alternate runtime.

The substrate has three strictly bounded timescales. Fast state records
activation, inhibition, typed prediction/observation mismatch surprise,
resource pressure, declared signal delivery, and an explicit bounded physiology
cooldown. Medium state can invoke only the pre-existing settlement-grounded
immutable reducers for routes, connections, tactics, bounded decay, and
rollback. A caller-supplied inactive tick may move a previously selected
candidate route one fixed small step toward neutral preference; this
non-evidentiary turnover does not create a settlement, audit credit, new
connection, tactic, or learning update. Slow state records minimal coherence
and recurrence observations; it has no authority over evidence, topology,
tactics, physiology, learning, or promotion.

Signals, surprise, advisory endogenous physiology, coherence, and
instrumentation are non-evidentiary. A task success or failure can affect
medium state only by satisfying the existing independently verified
constitutional settlement and execution-class gates. Setup/infrastructure,
timeout/resource, execution, contradiction, insufficient-evidence, stale, and
physiology-inhibited outcomes preserve their provenance but cannot gain credit
or reshape candidate topology, tactics, connections, homeostasis, or learning.

## Bounded metastability experiment observations

Stage 10.9 adds only a deterministic comparison and measurement layer over the
Stage 10.8 reducer. It accepts caller-owned finite immutable ticks and a
tick-zero immutable candidate state; it returns bounded trace-derived
observations. It is not a target-seeking criticality controller, a topology
optimizer, an evidence source, a new learning boundary, or a second runtime.

Full and ablated arms must retain the same raw inputs and tick budget. The
allowed counterfactual controls selectively neutralize advisory homeostasis,
empty-batch inactivity decay, action inhibition, mismatch surprise, or the
experiment's explicit anti-monopoly route-choice guard. They are comparison
inputs only. They cannot rewrite the canonical route cap, mutate a caller's
state, create a settlement, or bypass the existing grounded-credit gates.

Route entropy/diversity, dominance, turnover, plasticity, decay, rollback,
inhibition, surprise, recurrence, coherence, stagnation, resource pressure,
grounded task-performance observations, and regime transitions are descriptive
measurements only. Labels such as settling, oscillatory, stagnating, or
monopolizing do not establish an ideal topology or a magic criticality target.
Experiment output is not evidence, truth, confidence, promotion authority, or
adaptive credit. Non-creditable, setup/infrastructure, timeout/resource,
execution, contradiction, insufficient-evidence, and inhibited outcomes remain
non-creditable in every arm.

## Acceptance

An architectural change is accepted only when its files physically exist, its
registry entry is current, its validator passes, focused tests pass, and the
legacy non-interference boundary is checked. The registry and category notes
are the living record for future bounded changes.

## Deterministic candidate cycle

`kraken_r.cycle.run_constitutional_cycle` accepts one `Objective` and a
deterministic mode: `success`, `failure`, `contradiction`, or
`insufficient_evidence`. Every run advances immutable `TaskState` versions
from objective acquisition through an explicit stop. The action must carry
`Authority.KRAKEN_CANDIDATE`; missing or other authority fails closed.

Success and failure produce operational evidence from the observed execution
and may produce a learning update after settlement. Contradictory observations
produce evidence but withhold learning. An unobserved execution produces no
evidence and no learning. No declared claim or self-report is promoted to
evidence.

## Bounded candidate nervous-system slice

Round 4 extends the existing canonical `Signal`/`Event` alias with optional
task-state identity, task-state version, provenance, TTL, creation-tick,
source, cause, and priority fields. Candidate propagation accepts only bound,
declared signals with explicit source/cause identities and an integer priority.
It processes each fresh in-memory run by tick, then higher priority, then stable
signal ID order; equal priorities therefore replay identically. Any topic not
named by the static rule graph is rejected before delivery rather than counted
as an inert success. Hard limits remain in force for ticks, deliveries, and
fan-out.

Rules are static named edges. They can pass a message, explicitly amplify a
message by a fixed factor, or inhibit one named candidate action topic. They do
not learn, reinforce, persist, subscribe to a deployed dispatcher, call a
handler, change routing weights, or become an alternate runtime authority.

An uncertainty signal may therefore inhibit the candidate action before the
fixture execution boundary. That causes an honest `not_observed` result and
`insufficient_evidence`; it never manufactures `Evidence`, `GroundTruth`,
settlement support, or a learning update. The enabled and ablated paths are
replayable from immutable input records, and their first state divergence is
the authorization state.

## Recorded-execution replay

`kraken_r.replay` accepts one immutable `RecordedExecution` record and feeds it
through the same candidate lifecycle as the deterministic cycle. It is a
read-only fixture adapter, not an event subscription or execution engine.

Replay fails closed unless the record preserves its transaction, objective,
authorized task-state version, action, execution, evidence, decision, and
settlement identities, as well as its stop decision and any learning update.
Its provenance must name those identities and the recorded observation
boundary. Missing provenance, stale task-state versions, mismatched observed
identities, status/outcome conflicts, and self-report-only success are rejected
before a trace is produced. Only observed execution evidence can support
settlement or learning.

The replay adapter has no live inputs. It accepts only caller-provided
in-memory records, writes nothing, and never imports ROGAL runtime machinery.

## Independently verified bounded execution

Round 8 adds `kraken_r.grounded_execution`, a deliberately narrow external
observation adapter. It accepts exactly one candidate-owned
`run_bounded_pytest` action, a version-5 authorized `TaskState`, declared
relative text files, declared Python test paths, and explicit wall-clock, CPU,
memory, and output limits. It materializes those files only in a disposable
child workspace and has no arbitrary command, repository clone, network,
provider, persistence, goal-selection, mutation, retry, subscriber, or
controller path.

The executor produces an immutable observation record with sealed input,
workspace, output, and record hashes plus disposal and resource-limit
provenance. That record is **not evidence** and it has no success field. Each
 record carries an Ed25519 signature and public-key metadata; the private
 signing key never serializes. A record's public key is never a trust root:
 restart verification requires an independently supplied, pinned
 `TrustedExecutorIdentity`. Canonical JSON reconstruction is available for the
 sealed request, authorized state, and record before that verifier can recheck
 bindings and hashes. It rejects stale authorization, workspace escape,
 malformed or self-reported records, and derives success or failure only from
 completed test facts. A timeout, setup failure, malformed collection, or
 zero-test run withholds evidence and learning.

The bounded pytest transport uses a runner-owned empty configuration and
disables candidate `conftest.py` discovery. Candidate-controlled pytest hooks
or configuration therefore cannot rewrite the runner's call facts; any test
that depends on such local setup fails closed rather than becoming grounded
evidence.

Only a `VerifiedGroundedExecution` can enter the constitutional cycle. Its
translated `Evidence` is `GROUNDED`, not merely operational, and is bound to
the execution-record and record hashes. Structural replay revalidates the
immutable record without launching another child process. The existing fixture
path remains the default for unit tests; comparing fixture and grounded traces
is a source/authority ablation only and carries no performance claim. A
 caller-owned bounded receipt ledger persists bounded, expiry-stamped receipts
 with a file lock, durable replace, and directory sync. It rejects duplicate
 execution, evidence, settlement, and learning delivery across restarts and
 concurrent processes until their explicit expiry. It is an idempotency guard,
 not a controller. Resource-limit and cleanup provenance are accepted only when
 measured by the host runner: the sandbox is a fresh user/mount/network/PID
 namespace whose parent uses `unshare --kill-child=SIGKILL`; the runner confirms
 the whole namespace parent group has disappeared before it reports cleanup.
Stage 10.5 binds the child runtime declaration to the executor-selected
interpreter and pytest facts, rejects duplicate or mismatched declarations,
and rejects absolute, traversal, and symlink-escaping test paths before the
child is started.

## Bounded advisory physiology

Round 5 adapts only four safe concepts: normalized multidimensional pressure,
threshold and hysteresis behavior, contradiction/backlog regulation, and
protected resource reserves. `kraken_r.physiology` accepts one immutable,
caller-provided `PhysiologySnapshot` bound to the same transaction, objective,
and signaled task-state version as the candidate cycle.

The evaluator deterministically classifies the snapshot as `productive`,
`cautious`, `recovery`, or `critical`. Its decision is advisory-only and has
one permitted effect: it may conservatively inhibit the already candidate-owned
action before fixture execution. A physiology inhibition produces an honest
`not_observed` result and therefore `insufficient_evidence`; it cannot create
evidence, ground truth, settlement support, a learning update, a new action,
or a goal selection.

There are no live sensors, feedback workers, queues, cooldown timers,
subscriptions, or persisted regulator state. Hysteresis is explicit input
(`prior_regime` plus a bounded `cooldown_remaining`), so replay consumes the
same immutable record and produces the same trace. The legacy pressure field,
HOP, homeostasis controllers, goal queue, and resource pools are neither
imported nor called and remain preserved/quarantined references.

## Settlement-grounded plastic route preference

Round 6 adapts only the safe ideas from the quarantined cascade router,
pathway tracker, topology tracker, plasticity store, and online-learning
integration: route-local preference, explicit source/target topology, bounded
strengthening and weakening, audit lineage, reset/ablation, and an observable
later route choice. It does not import those modules or reconnect their SQLite
stores, global singletons, clocks, background thread, signal subscriptions,
confidence handling, broad credit assignment, or persistent state.

`kraken_r.plastic_routing` is a pure reducer over an immutable caller-provided
`RouteTopology`. A `RouteSelection` is a deterministic candidate preference
only: it cannot dispatch a message, execute an action, select a goal, settle a
transaction, or alter the constitutional lifecycle. The caller can submit one
`SettlementRouteRecord` only after the constitutional trace exists.

An update is accepted only when all of the following agree:

1. the selected route is the deterministic highest-weight route at the recorded
   topology version;
2. transaction, objective, authorized task state/version, route, settlement,
   and evidence identities match the explicit provenance binding;
3. the settlement is `settled` with an observed `success` or `failure`; and
4. every cited `Evidence` is operational/grounded observed execution evidence.

A grounded success strengthens only the selected route; a grounded failure
weakens only the selected route. Contradiction, insufficient evidence,
unsettled results, declared signals, confidence, physiology state, raw model
claims, malformed provenance, stale selections, and duplicate in-window
settlements cannot earn credit. Every topology generation accepts at most 16
unique settlement identities; the bounded budget rejects a seventeenth update
rather than evicting old IDs. Updates use a fixed ±0.10 delta constrained to
the `[0.25, 0.75]` interval; no automatic promotion, cross-route credit, or
adaptive controller exists. Resetting a topology ablates learned state while
retaining the declared route graph and advances its generation, so prior
selections cannot be reused. Replaying the same ordered immutable record
history produces the same topology and audit traces.

## Controlled LLM proposal adapter

Round 7 adds `kraken_r.llm_adapter` as a narrow, stateless leaf behind an
injected provider protocol. It accepts an immutable `CandidateModelContext`
projected only after the canonical cycle trace validates and its authorized
task state is confirmed, plus a bounded prompt and model configuration. The
context deliberately excludes evidence, execution observations, settlement,
confidence, signals, and physiology. A model can return exactly three declared
fields: `proposal`, `reasoning`, and an optional bounded `route_hint`.

The adapter has no fallback model, retry loop, cache, budget authority, event
emission, store, runtime wiring, queue, controller state, execution path, goal
selection, or persistence. It cannot create an Action, Evidence, Settlement,
LearningUpdate, or Mutation. Top-level model claims about success, failure,
confidence, evidence, execution, settlement, action, goal, or learning are
rejected. A valid proposal is still declared-only; no model self-report can
support learning. The existing settlement-grounded reducer continues to accept
only a complete constitutional trace with grounded execution evidence.

Provider identity and model provenance must exactly match the injected request;
provider failures, enforced invocation deadlines, malformed JSON, invalid route
hints, and malformed context bindings fail closed with no proposal. Received
malformed output is retained only in the immutable invocation envelope and
hashed so its failed disposition can also be structurally replayed. Generation
is explicitly nondeterministic, but the request configuration/input hash,
raw-output hash, parsed proposal shape, and failure disposition can be
structurally replayed without another provider call.

`kraken_r.controlled_evaluation` evaluates the same held-out task set in three
conditions: base model with a masked organizational slot, Kraken-R-mediated
with the current bounded route selection, and Kraken-R with route learning
reset. It fixes the provider, model ID, task set, prompt template, temperature,
and output-token budget. It verifies provider identity, model configuration,
and normalized prompt length per task; where a provider reports input-token
counts, unequal counts are surfaced as a failed control rather than hidden.
The report is descriptive only: it records observed task-label deltas but makes
no performance claim because proposal output is not independent execution
settlement.

## Stage 9 interaction validation

`kraken_r.interactions` is a stateless, standalone acceptance validator for
existing immutable records. It does not add a circuit, scheduler, persistent
ledger, controller, or execution path. It rechecks that independently verified
grounded execution, constitutional settlement, route-credit provenance,
signal/physiology inhibition, replay, and model proposal provenance can compose
without granting new authority.

The validator accepts only caller-supplied traces and records. It requires a
grounded trace to be independently reverified against the same sealed request
and authorized state, checks its evidence lineage against the record hash, and
requires the only representable delivery sequence to be
`execution → evidence → settlement → learning` (with later stages absent when
there is no observed evidence). Replays remain read-only; the existing bounded
receipt ledger remains the only duplicate-delivery guard.

Any signal or physiology inhibition is conservative: neither source can
authorize execution, evidence, settlement support, or learning. A model
invocation can be structurally replayed only as declared proposal provenance;
proposal output cannot appear in execution evidence or influence route credit.
For a grounded settlement, the route reducer itself also requires the same
sealed request, verified record, and verifier before it can issue bounded
credit; it rejects a ground-evidence label without that proof. The reducer
remains a pure, bounded caller-owned transformation.

The deployed main instance remains responsible for all live runtime work:
daemon lifecycle, buses, stores, queues, retries, controllers, executors,
actuators, live observations, and any future promotion. The following remain
quarantined or planned-only after Stage 9: permanent specialist daemons,
autonomous recursion, ALU/resonant-lattice composition, organ evolution,
chemistry, HOP/Redstone stacks, alternate reasoning runtimes, and broad
cognitive features.

The provider-backed [controlled trial archive](CONTROLLED_MODEL_TRIAL_COMPARABLE.md)
stores immutable invocation envelopes and structural replay records beside its
descriptive labels. Structural replay verifies the stored request/output hashes
without a second provider call. Any provider or schema failure remains visible
in the archive and does not become evidence, execution, settlement, learning,
or a performance claim.

## Stage 10 adaptive substrate

Stage 10 adds only ordinary, bounded candidate adaptation. It is a pure reducer
over caller-owned immutable `AdaptiveState`, not a new Kraken runtime. Every
route, connection, recovery, decay, tactic, or rollback operation is
generation-scoped, bounded by explicit hard limits, and recorded in finite
immutable audit lineage.

Learning credit is stricter than an ordinary candidate cycle: it requires the
exact constitutional trace, a settled success or failure, an independently
reverified execution record, the sealed request, the matching authorized
candidate state, and matching request, record, evidence, and provenance hashes.
Signals, physiology, uncertainty, confidence, novelty, resource pressure, and
model declarations may conservatively advise or inhibit; none can create
evidence, settlement, learning credit, route preference, connection topology,
or tactic authority.

Route preference changes only the selected eligible route. Weights remain
within fixed bounds, decay and recovery move toward the neutral baseline in
small fixed steps, and finite update budgets prevent lock-in. Candidate
connections are deterministic source-target identities with fixed total and
degree limits; they never dispatch, execute, subscribe, persist, or become a
runtime edge. Tactic switching only chooses among predeclared labels using a
grounded settled result plus a task-state-bound `HomeostaticSnapshot`; it cannot
select a goal or bypass authorization and stop behavior.

Rollback restores a retained immutable checkpoint into a new generation while
retaining all already-consumed record identities, so an old outcome cannot earn
credit a second time. Homeostatic observations explicitly cover contradiction,
uncertainty, repeated failure, novelty, and resource expenditure as bounded
declared signals. They are advisory-only and remain outside evidence and
learning authority.

### Orzhaal boundary

Orzhaal is not a Kraken-R organ, controller, recursive loop, or persistent
experimental runtime. It is a minimal disposable comparison boundary for
adaptation-rule experiments: it reads an immutable candidate state, returns a
structured non-promotable result, and provides no canonical write or promotion
operation. Adaptation of adaptation rules therefore remains isolated from
ordinary Kraken adaptation.

Stage 11 is explicitly unstarted. Higher-order circuits, autonomous recursion,
specialist daemons, event buses, background workers, live controllers,
permanent self-modification, organ evolution, broad lattice composition,
chemistry, Redstone, HOP, ALU systems, and alternate reasoning runtimes remain
excluded.

## Stage 10.7 task integrity

Stage 10.7 adds a pure, bounded inspection boundary for preserving the exact
caller-supplied task while making interpretation losses visible. `OriginalTask`
is immutable and hash-bound to a `TaskSpecification`; every structured
requirement keeps a source clause identity and can be traced through declared
competing hypotheses, a proposal-only plan, and a candidate result.

`BeliefState` keeps a finite set of candidate interpretations with declared
support, contradiction, unresolved dependencies, and status. Those fields are
not evidence grades and cannot settle a task, alter authority, create topology,
or change adaptive state. `InformationLossReport` deterministically reports
omitted, compressed, underweighted, and unsupported reverse dependencies across
the specification, belief, plan, and result. It never repairs or mutates those
inputs.

Angel receives only requirement identities, kinds, weights, and declared
coverage to look for omissions. Nemesis receives only hypothesis status and
declared contradiction/dependency identities to request falsification.
Antimetabole receives only conclusion-to-claim dependency structure to request
reverse checks. The original raw task, provenance, execution, evidence,
settlement, authority, and adaptive context are excluded from all projections.
Each role is selective and returns only bounded proposal-only findings plus
questions for a later independent tool, execution, or retrieval boundary.

Neither a review finding nor a verification question is evidence. Ambiguous,
infrastructure/setup, timeout/resource, execution, contradiction, and
insufficient-evidence outcomes remain non-creditable. This stage does not add
a provider, execution path, persistent memory, adaptive reducer, controller,
daemon, routing reputation, open-weight probe, recursive circuit, or any Stage
11 functionality.
