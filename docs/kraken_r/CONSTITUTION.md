# Kraken-R Constitution

**Status:** candidate-only foundation  
**Version:** 1.5
**Effective boundary:** documentation, standalone validation, and bounded in-memory candidate-cycle execution

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
recorded-execution replay, bounded signal propagation, and bounded advisory
physiology have no daemon entrypoint, runtime writer, event subscription,
external executor, router, ledger, homeostasis controller, or mutation
actuator. Loading, validating, or running them must not start ROGAL, contact an
LLM, open live stores, or change `.rogal/` runtime state.

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
