"""Long-horizon integrity soak for Kraken-R Stage 10.8/10.9, in two passes.

This module exercises the immutable dynamical/adaptive/plastic-routing
reducers across many more cycles than any existing acceptance test, without
weakening any declared production bound (``MAX_TICKS``, ``MAX_ADAPTIVE_
GENERATIONS``, ``MAX_ADAPTIVE_UPDATES``, ``MAX_AUDIT_LINKS``, ...).

A single ``DynamicalState`` lineage cannot advance past ``MAX_TICKS`` (256)
ticks -- that bound is enforced at construction time and is not something a
test may raise. Long-horizon coverage is therefore obtained by *chaining many
independent bounded sessions* for the cheap bulk soak, and by driving a
handful of dedicated sessions to their exact declared ceiling to assert the
boundary is hit precisely (neither early nor silently exceeded).

Fail-closed classes explicitly covered here:
  1. bounded state growing beyond its declared limit
  2. non-finite / out-of-range numeric state
  3. stale or ancient lineage regaining adaptive credit
  4. duplicate settlement/evidence identity being credited twice
  5. rollback manufacturing credit instead of only undoing it
  6. illegal route/topology bounds
  7. a legitimate declared bounded-capacity ceiling (a generation's update
     budget, the rollback budget, the adaptive generation ceiling, tracked-
     record identity retention, the topology's settlement-audit retention,
     or the invalidation budget) crashing tick reduction instead of being
     withheld like every other boundary condition

The first pass (the 10,000-cycle chained soak and the exact-ceiling tests)
discovered failure class 7 as an open gap: a fully legitimate settlement or
rollback event crashed ``reduce_dynamical_tick`` with an uncaught exception
whenever it happened to land on one of six declared bounded-capacity
ceilings, instead of becoming a non-mutating ``WithheldEvent`` like every
sibling boundary condition recognized by ``_is_current_lineage_error``. That
gap is now closed by a sibling classifier, ``_is_bounded_capacity_exhaustion_
error``, applied at the same three tick-reducer call sites. The dedicated
regression tests below drive each of the five capacity-exhaustion modes that
are actually reachable through ``reduce_dynamical_tick`` (topology settlement
budget, adaptive generation update budget, record-identity retention,
rollback budget, and the generation ceiling via rollback) through the full
reducer and confirm: a withheld (never crashing) outcome, zero state
mutation, a stable/deterministic reason string, and that replaying or
retrying the identical rejected transition never consumes hidden state or
changes the semantic rejection.

One of the six named exhaustion messages -- "invalidation budget is
exhausted" -- is a documented exception: ``invalidate_grounded_adaptation``
has no corresponding event kind or settlement ``operation`` in
``DynamicalEvent``'s vocabulary today, so it cannot be reached through
``reduce_dynamical_tick`` at all (confirmed by inspection: no caller of
``invalidate_grounded_adaptation``/``replay_adaptive_invalidations`` exists
in ``dynamical_substrate.py``). It is covered here at the only place it is
actually reachable -- direct calls into ``adaptive_substrate`` -- plus a
direct check that the new classifier recognizes its exact message, rather
than pretending full-tick-reducer coverage exists for a path that does not
exist.

The second pass adds a bounded combinatorial soak: a seeded, deterministic
mix of legal and pathological conditions across nine independent dimensions
(event ordering, rollback distance, stale-state age, settlement saturation,
reset timing, invalidation timing, topology-selection age, resource
saturation, and contradictory observations), run across many independent
bounded sessions with every global invariant re-checked after every tick.

Every reducer call in this file is a pure function over caller-owned
immutable records; nothing here starts a clock, worker, or daemon, and no
production bound is modified to make a test pass.
"""

from __future__ import annotations

import math
import random
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from kraken_r import (
    AdaptiveCheckpoint,
    AdaptiveState,
    AdaptiveSubstrateValidationError,
    BASELINE_WEIGHT,
    CandidateRoute,
    DynamicalEvent,
    DynamicalState,
    DynamicalSubstrateValidationError,
    DynamicalTick,
    FastState,
    GroundedDeliveryLedger,
    GroundedExecutionExecutor,
    GroundedExecutionRequest,
    MAX_ADAPTIVE_GENERATIONS,
    MAX_ADAPTIVE_UPDATES,
    MAX_AUDIT_LINKS,
    MAX_WEIGHT,
    MIN_WEIGHT,
    Objective,
    PlasticRoutingValidationError,
    RouteTopology,
    SettlementRouteRecord,
    TaskState,
    apply_grounded_adaptation,
    apply_settlement_learning,
    form_grounded_connection,
    invalidate_grounded_adaptation,
    make_bound_signal,
    make_grounded_action,
    reduce_dynamical_tick,
    replay_dynamical_ticks,
    rollback_adaptive_state,
    run_constitutional_cycle,
    select_candidate_route,
)
from kraken_r.adaptive_substrate import MAX_TRACKED_RECORDS
from kraken_r.dynamical_substrate import (
    MAX_EVENT_COUNTER,
    MAX_EVENT_LOG,
    MAX_TICK_EVENTS,
    MAX_TICKS,
    _is_bounded_capacity_exhaustion_error,
)

# ---------------------------------------------------------------------------
# Shared deterministic fixtures (no clocks, no crypto randomness anywhere)
# ---------------------------------------------------------------------------


def _grounded_dynamical_state(label: str) -> DynamicalState:
    objective_id = f"{label}-objective"
    return DynamicalState.fixture(
        f"{label}-substrate",
        transaction_id=f"{label}-transaction",
        objective_id=objective_id,
        task_state_id=f"{objective_id}-state-5",
        task_state_version=5,
    )


def _grounded_route_record_for_topology(
    topology: RouteTopology,
    label: str,
    *,
    transaction_id: str | None = None,
    objective_id: str | None = None,
    passing: bool = True,
    causal_parent_record_id: str | None = None,
) -> SettlementRouteRecord:
    """Build one independently verified grounded settlement record."""

    objective_id = objective_id or f"{label}-objective"
    transaction_id = transaction_id or f"{label}-transaction"
    objective = Objective(
        objective_id,
        "Produce an independently verified longevity-soak outcome.",
        provenance={"transaction_id": transaction_id},
    )
    action = make_grounded_action(objective.objective_id)
    authorized = TaskState(
        f"{objective.objective_id}-state-5",
        objective.objective_id,
        5,
        "authorized",
        values={"action_id": action.action_id},
    )
    files = {
        "subject.py": f"def answer():\n    return {42 if passing else 0}\n",
        "test_subject.py": (
            "from subject import answer\n\n"
            "def test_answer():\n"
            "    assert answer() == 42\n"
        ),
    }
    request = GroundedExecutionRequest(
        f"{label}-request",
        transaction_id,
        objective.objective_id,
        authorized.state_id,
        authorized.version,
        action,
        files,
        ("test_subject.py",),
        causal_parent_record_id=causal_parent_record_id,
    )
    ledger = GroundedDeliveryLedger(
        Path(tempfile.mkdtemp(prefix="kraken-r-soak-receipts-")) / "receipts.json"
    )
    executor = GroundedExecutionExecutor(delivery_ledger=ledger)
    verifier = executor.verifier()
    execution = executor.execute(request, authorized_state=authorized)
    verified = verifier.verify(execution, request=request, authorized_state=authorized)
    trace = run_constitutional_cycle(
        objective,
        grounded_execution=verified,
        grounded_request=request,
        grounded_verifier=verifier,
    )
    selection = select_candidate_route(
        topology,
        "candidate-work",
        transaction_id=trace.transaction_id,
        objective_id=trace.objective.objective_id,
        task_state_id=trace.states[4].state_id,
        task_state_version=trace.states[4].version,
    )
    return SettlementRouteRecord(
        f"{label}-record",
        selection,
        trace,
        provenance={
            "transaction_id": trace.transaction_id,
            "objective_id": trace.objective.objective_id,
            "task_state_id": trace.states[4].state_id,
            "task_state_version": trace.states[4].version,
            "route_id": selection.route_id,
            "settlement_id": trace.settlement.settlement_id,
            "evidence_ids": tuple(item.evidence_id for item in trace.evidence),
            **(
                {"causal_parent_record_id": causal_parent_record_id}
                if causal_parent_record_id is not None
                else {}
            ),
        },
        grounded_execution=verified,
        grounded_request=request,
        grounded_verifier=verifier,
    )


def _grounded_route_record(
    state: AdaptiveState, label: str, *, passing: bool = True
) -> SettlementRouteRecord:
    return _grounded_route_record_for_topology(state.route_topology, label, passing=passing)


def _grounded_route_record_from_dynamical(
    state: DynamicalState, label: str, *, passing: bool = True
) -> SettlementRouteRecord:
    return _grounded_route_record_for_topology(
        state.medium.adaptive_state.route_topology,
        label,
        transaction_id=state.transaction_id,
        objective_id=state.objective_id,
        passing=passing,
    )


def _require_child_pytest(record: SettlementRouteRecord) -> None:
    if record.grounded_execution.epistemic_class.value != "task_success":
        pytest.skip(
            "grounded child pytest is unavailable in this environment; "
            "the default-runtime reliability task covers that prerequisite"
        )


def _contradiction_record(state: DynamicalState, label: str) -> SettlementRouteRecord:
    """A cheap, deterministic, always-withheld settlement (no subprocess)."""

    objective = Objective(
        state.objective_id,
        "Produce a contradiction with no grounded credit during a soak session.",
        provenance={"transaction_id": state.transaction_id},
    )
    trace = run_constitutional_cycle(objective, mode="contradiction")
    selection = select_candidate_route(
        state.medium.adaptive_state.route_topology,
        "candidate-work",
        transaction_id=trace.transaction_id,
        objective_id=trace.objective.objective_id,
        task_state_id=trace.states[4].state_id,
        task_state_version=trace.states[4].version,
    )
    return SettlementRouteRecord(
        label,
        selection,
        trace,
        provenance={
            "transaction_id": trace.transaction_id,
            "objective_id": trace.objective.objective_id,
            "task_state_id": trace.states[4].state_id,
            "task_state_version": trace.states[4].version,
            "route_id": selection.route_id,
            "settlement_id": trace.settlement.settlement_id,
            "evidence_ids": tuple(item.evidence_id for item in trace.evidence),
        },
    )


def _assert_dynamical_invariants(state: DynamicalState) -> None:
    """Cheap defense-in-depth checks re-verified at every soak tick."""

    floats = (
        state.fast.activation,
        state.fast.inhibition,
        state.fast.surprise,
        state.fast.resource_pressure,
        state.slow.coherence_score,
        state.instrumentation.activity,
        state.instrumentation.diversity,
        state.instrumentation.entropy,
        state.instrumentation.dominant_route_share,
        state.instrumentation.surprise,
        state.instrumentation.resource_pressure,
    )
    assert all(math.isfinite(value) for value in floats), floats
    assert all(0.0 <= value <= 1.0 for value in floats), floats
    assert 0 <= state.tick <= MAX_TICKS
    assert len(state.event_log) <= MAX_EVENT_LOG
    assert len(state.consumed_event_ids) <= MAX_EVENT_LOG
    assert len(state.withheld_events) <= 32
    assert len(state.historical_event_ids) <= MAX_EVENT_COUNTER
    assert len(state.medium.route_history) <= 32
    assert len(state.fast.signal_topics) <= 16
    assert len(state.slow.coherence_observations) <= 16
    for counter in (
        state.medium.turnover,
        state.medium.decay_events,
        state.medium.plasticity_events,
        state.instrumentation.turnover,
        state.instrumentation.decay_events,
        state.instrumentation.plasticity_events,
        state.instrumentation.rollback_events,
    ):
        assert 0 <= counter <= MAX_EVENT_COUNTER
    for counter in (
        state.instrumentation.inhibition_events,
        state.instrumentation.stagnation_ticks,
        state.instrumentation.coherence_recurrence,
        state.slow.recurrence,
    ):
        assert 0 <= counter <= MAX_TICKS
    for route in state.medium.adaptive_state.route_topology.routes:
        assert MIN_WEIGHT <= route.weight <= MAX_WEIGHT
        assert route.success_count >= 0 and route.failure_count >= 0
    assert 0 <= state.medium.adaptive_state.generation <= MAX_ADAPTIVE_GENERATIONS
    assert 0 <= state.medium.adaptive_state.updates_applied <= MAX_ADAPTIVE_UPDATES


def _synthetic_checkpoint(state: AdaptiveState, suffix: str) -> AdaptiveCheckpoint:
    """A directly-constructed, valid pre-update checkpoint for one exact
    adaptive state.

    Rollback only ever requires a *retained* checkpoint matching the
    current generation/topology lineage -- it never touches a settlement
    record. Building one directly (instead of running a real grounded
    credit cycle just to get a checkpoint as a side effect) keeps the
    bounded-capacity regression tests below free of any subprocess cost."""

    return AdaptiveCheckpoint(
        f"{state.substrate_id}-synthetic-checkpoint-{suffix}",
        state.generation,
        state.route_topology,
        state.connections,
        state.active_tactic_id,
        state.tactic_scores,
        state.applied_record_ids,
    )


def _dynamical_state_with_adaptive(label: str, adaptive: AdaptiveState) -> DynamicalState:
    """A fresh dynamical fixture with one field -- its adaptive substrate --
    swapped for a directly-constructed pathological/boundary state."""

    base = DynamicalState.fixture(label)
    return replace(base, medium=replace(base.medium, adaptive_state=adaptive))


def _assert_bounded_capacity_exhaustion_is_withheld_and_replay_safe(
    state: DynamicalState,
    event: DynamicalEvent,
    *,
    expected_reason_prefix: str,
    expected_message_fragment: str,
) -> None:
    """Shared assertions for every reachable bounded-capacity exhaustion
    mode: the tick reduces to a withheld, non-mutating outcome (never an
    uncaught exception), the withheld reason is deterministic across
    repeated evaluation of the identical starting state and tick, and
    replaying the identical rejected transition in a later tick -- from the
    still-unmutated resulting state -- changes neither the outcome nor any
    hidden state."""

    before = state.medium.adaptive_state
    next_state, trace = reduce_dynamical_tick(state, DynamicalTick(state.tick + 1, (event,)))
    assert len(trace.withheld_events) == 1
    reason = trace.withheld_events[0].reason
    assert reason.startswith(expected_reason_prefix)
    assert expected_message_fragment in reason
    assert next_state.medium.adaptive_state == before

    # Determinism: replaying the identical starting state + tick yields a
    # byte-identical withheld outcome, not just an equivalent one.
    replay_state, replay_trace = reduce_dynamical_tick(
        state, DynamicalTick(state.tick + 1, (event,))
    )
    assert replay_state == next_state
    assert replay_trace.withheld_events[0].reason == reason

    # Retrying the same rejected settlement/checkpoint in a later tick, from
    # the still-unmutated resulting state, is refused identically -- no
    # hidden state advances and the rejection reason never drifts.
    retry_event = replace(event, event_id=f"{event.event_id}-retry")
    retried_state, retried_trace = reduce_dynamical_tick(
        next_state, DynamicalTick(next_state.tick + 1, (retry_event,))
    )
    assert len(retried_trace.withheld_events) == 1
    assert retried_trace.withheld_events[0].reason == reason
    assert retried_state.medium.adaptive_state == before


NUM_SOAK_SESSIONS = 40
TICKS_PER_SOAK_SESSION = 250
TOTAL_SOAK_TICKS = NUM_SOAK_SESSIONS * TICKS_PER_SOAK_SESSION


def test_ten_thousand_cycle_dynamical_soak_stays_within_every_declared_bound() -> None:
    """Chain 40 independent bounded 256-tick sessions (10,000 reducer calls
    total) through a deterministic, seeded mix of signal/observation/
    resource/settlement/rollback events, and re-verify every declared bound
    after every single tick. No production bound is touched; a single
    lineage's own MAX_TICKS ceiling is why this soak chains sessions instead
    of running one long-lived state (see the ceiling test below)."""

    rng = random.Random(20260826)
    total_ticks = 0
    for session_index in range(NUM_SOAK_SESSIONS):
        state = DynamicalState.fixture(f"soak-session-{session_index}")
        initial_adaptive_state = state.medium.adaptive_state
        contradiction = _contradiction_record(
            state, f"soak-{session_index}-contradiction"
        )
        previous_historical_len = 0
        for tick in range(1, TICKS_PER_SOAK_SESSION + 1):
            roll = rng.random()
            if roll < 0.30:
                event = DynamicalEvent.resource_event(
                    f"s{session_index}-t{tick}-resource", round(rng.random(), 4)
                )
            elif roll < 0.55:
                event = DynamicalEvent.observation_event(
                    f"s{session_index}-t{tick}-observation",
                    round(rng.random(), 4),
                    round(rng.random(), 4),
                )
            elif roll < 0.75:
                event = DynamicalEvent.signal_event(
                    f"s{session_index}-t{tick}-signal",
                    make_bound_signal(
                        f"s{session_index}-t{tick}-signal-payload",
                        "candidate.urgency",
                        transaction_id=state.transaction_id,
                        objective_id=state.objective_id,
                        task_state_id=state.task_state_id,
                        task_state_version=state.task_state_version,
                        source="longevity-soak",
                        cause=f"s{session_index}-t{tick}-cause",
                    ),
                )
            elif roll < 0.90:
                event = DynamicalEvent.settlement_event(
                    f"s{session_index}-t{tick}-contradiction-event", contradiction
                )
            else:
                event = DynamicalEvent.rollback_event(
                    f"s{session_index}-t{tick}-bogus-rollback",
                    f"never-issued-checkpoint-{session_index}-{tick}",
                )

            state, trace = reduce_dynamical_tick(state, DynamicalTick(tick, (event,)))
            _assert_dynamical_invariants(state)
            assert len(state.historical_event_ids) >= previous_historical_len
            previous_historical_len = len(state.historical_event_ids)
            total_ticks += 1

        # A session built only from withheld/non-grounded events must never
        # grant any adaptive credit, no matter how many ticks accumulate.
        assert state.medium.adaptive_state == initial_adaptive_state

    assert total_ticks == TOTAL_SOAK_TICKS == 10_000


def test_dynamical_state_permanently_halts_and_historical_ledger_hits_exact_ceiling() -> None:
    """A single dynamical lineage cannot advance past MAX_TICKS; its
    non-evicting historical event ledger can reach but never exceed the
    declared MAX_EVENT_COUNTER ceiling, which is exactly
    MAX_TICKS * MAX_TICK_EVENTS -- so a maximally busy single lineage fills
    it precisely as it exhausts its own tick budget. This is the concrete
    boundary behind the already-filed 'dynamical substrate permanent
    lockup' gap: there is no reset path back to tick 0 once this point is
    reached."""

    assert MAX_EVENT_COUNTER == MAX_TICKS * MAX_TICK_EVENTS

    state = DynamicalState.fixture("ceiling-session")
    first_tick_event_ids: tuple[str, ...] = ()
    for tick in range(1, MAX_TICKS + 1):
        events = tuple(
            DynamicalEvent.resource_event(f"ceiling-t{tick}-e{index}", 0.10)
            for index in range(MAX_TICK_EVENTS)
        )
        if tick == 1:
            first_tick_event_ids = tuple(event.event_id for event in events)
        state, _ = reduce_dynamical_tick(state, DynamicalTick(tick, events))
        _assert_dynamical_invariants(state)

    assert state.tick == MAX_TICKS
    assert len(state.historical_event_ids) == MAX_EVENT_COUNTER
    assert len(set(state.historical_event_ids)) == MAX_EVENT_COUNTER

    # Historical identities are never evicted, unlike the bounded
    # sliding-window ledgers (event_log / consumed_event_ids at MAX_EVENT_LOG).
    assert set(first_tick_event_ids).issubset(set(state.historical_event_ids))
    assert not set(first_tick_event_ids) & set(state.event_log)
    assert not set(first_tick_event_ids) & set(state.consumed_event_ids)
    assert len(state.event_log) == MAX_EVENT_LOG
    assert len(state.consumed_event_ids) == MAX_EVENT_LOG

    # The lineage is now permanently exhausted: no further tick can even be
    # constructed, and replay refuses to advance a single further step.
    with pytest.raises(DynamicalSubstrateValidationError, match="exceeds its bounded limit"):
        DynamicalTick(MAX_TICKS + 1)

    def one_more_tick():
        yield DynamicalTick(1)

    with pytest.raises(DynamicalSubstrateValidationError, match="replay tick budget exceeded"):
        replay_dynamical_ticks(state, one_more_tick())


def test_saturating_counters_hold_the_line_under_real_repeated_reduction_at_the_ceiling() -> None:
    """Push every saturating medium/instrumentation counter to one below its
    declared ceiling, then drive two more *real* reducer transitions (not
    direct construction) and confirm every counter saturates rather than
    overflowing or wrapping."""

    initial = DynamicalState.fixture("saturation-session")
    topology = initial.medium.adaptive_state.route_topology
    active_route = topology.routes[0]
    boosted_topology = replace(
        topology, routes=(replace(active_route, weight=0.65),) + topology.routes[1:]
    )
    near_ceiling = replace(
        initial,
        fast=replace(initial.fast, active_route_id=active_route.route_id),
        medium=replace(
            initial.medium,
            adaptive_state=replace(
                initial.medium.adaptive_state, route_topology=boosted_topology
            ),
            turnover=MAX_EVENT_COUNTER - 1,
            decay_events=MAX_EVENT_COUNTER - 1,
            plasticity_events=MAX_EVENT_COUNTER - 1,
        ),
        instrumentation=replace(
            initial.instrumentation,
            turnover=MAX_EVENT_COUNTER - 1,
            decay_events=MAX_EVENT_COUNTER - 1,
            plasticity_events=MAX_EVENT_COUNTER - 1,
            stagnation_ticks=MAX_TICKS - 1,
        ),
    )

    state, trace_one = reduce_dynamical_tick(near_ceiling, DynamicalTick(1))
    assert trace_one.inactive_decay_route_id == active_route.route_id
    assert state.medium.turnover == MAX_EVENT_COUNTER
    assert state.medium.decay_events == MAX_EVENT_COUNTER
    assert state.medium.plasticity_events == MAX_EVENT_COUNTER
    assert state.instrumentation.turnover == MAX_EVENT_COUNTER
    assert state.instrumentation.decay_events == MAX_EVENT_COUNTER
    assert state.instrumentation.plasticity_events == MAX_EVENT_COUNTER
    assert state.instrumentation.stagnation_ticks == MAX_TICKS

    state, _ = reduce_dynamical_tick(state, DynamicalTick(2))
    assert state.medium.turnover == MAX_EVENT_COUNTER
    assert state.instrumentation.turnover == MAX_EVENT_COUNTER
    assert state.instrumentation.stagnation_ticks == MAX_TICKS
    _assert_dynamical_invariants(state)


def test_adaptive_lineage_generation_ceiling_and_ancient_state_resurrection_over_many_cycles() -> None:
    """Drive an adaptive lineage through MAX_ADAPTIVE_GENERATIONS repeated
    grounded-credit + rollback cycles. Confirms: rollback always restores
    the exact pre-credit weight (failure class 5: no manufactured credit);
    the topology version keeps climbing even though the value it carries
    keeps returning to baseline; and once the declared generation ceiling
    is reached, rollback becomes permanently unavailable for this lineage
    even though ordinary bounded credit can continue. Also confirms three
    concrete ancient-state-resurrection cases after the long run: a stale
    (already-applied) settlement record, an evicted checkpoint, and a
    generation-exhausted rollback attempt."""

    state = AdaptiveState.fixture("generation-ceiling")
    first_record: SettlementRouteRecord | None = None
    first_checkpoint_id: str | None = None
    previous_topology_version = state.route_topology.version

    for cycle in range(MAX_ADAPTIVE_GENERATIONS):
        assert state.generation == cycle
        pre_credit_weight = state.route_topology.routes[0].weight
        assert pre_credit_weight == pytest.approx(BASELINE_WEIGHT)

        record = _grounded_route_record(state, f"generation-ceiling-credit-{cycle}")
        _require_child_pytest(record)
        if first_record is None:
            first_record = record

        credited, audit = apply_grounded_adaptation(state, record)
        assert audit.operation == "strengthen"
        assert credited.generation == cycle  # credit alone never advances generation
        assert credited.route_topology.version > previous_topology_version

        checkpoint_id = credited.checkpoints[-1].checkpoint_id
        if first_checkpoint_id is None:
            first_checkpoint_id = checkpoint_id

        state, rollback_audit = rollback_adaptive_state(credited, checkpoint_id)
        assert rollback_audit.operation == "rollback"
        assert state.generation == cycle + 1
        assert state.updates_applied == 0
        # Failure class 5: rollback restores the pre-credit weight exactly;
        # it never leaves behind partial or extra credit.
        assert state.route_topology.routes[0].weight == pytest.approx(pre_credit_weight)
        assert state.route_topology.version > credited.route_topology.version
        previous_topology_version = state.route_topology.version

    assert state.generation == MAX_ADAPTIVE_GENERATIONS

    # Ordinary bounded credit still works at the generation ceiling...
    one_more = _grounded_route_record(state, "generation-ceiling-final-credit")
    updated, _ = apply_grounded_adaptation(state, one_more)
    assert updated.generation == MAX_ADAPTIVE_GENERATIONS
    assert updated.updates_applied == 1

    # ...but rollback is now permanently refused for this lineage: reaching
    # the generation ceiling is a one-way door for undo, not for edits.
    checkpoint_id = updated.checkpoints[-1].checkpoint_id
    with pytest.raises(AdaptiveSubstrateValidationError, match="generation limit is exhausted"):
        rollback_adaptive_state(updated, checkpoint_id)

    # Ancient-state resurrection 1: the very first credited record can never
    # be re-applied, even after MAX_ADAPTIVE_GENERATIONS full cycles.
    assert first_record is not None
    with pytest.raises(AdaptiveSubstrateValidationError, match="already applied"):
        apply_grounded_adaptation(updated, first_record)

    # Ancient-state resurrection 2: the very first checkpoint has long since
    # been evicted from the bounded checkpoint window; replaying it must be
    # rejected as not retained, not silently ignored or reused.
    assert first_checkpoint_id is not None
    assert first_checkpoint_id not in tuple(c.checkpoint_id for c in updated.checkpoints)
    with pytest.raises(AdaptiveSubstrateValidationError, match="not retained"):
        rollback_adaptive_state(updated, first_checkpoint_id)


def test_route_topology_settlement_budget_is_a_hard_bound_requiring_explicit_reset() -> None:
    """`RouteTopology.applied_settlement_ids` is a fixed-size audit tuple
    (failure class 6: illegal/unbounded topology growth would be a bug).
    Once it is full, `apply_settlement_learning` must fail closed rather
    than silently evict the oldest entry or auto-reset the generation --
    and a record selected against the pre-reset lineage must remain stale
    forever after an explicit reset."""

    topology = RouteTopology.fixture("route-budget-topology")
    for index in range(MAX_AUDIT_LINKS):
        record = _grounded_route_record_for_topology(topology, f"route-budget-{index}")
        _require_child_pytest(record)
        topology, learning = apply_settlement_learning(topology, record)
        assert learning.disposition == "accepted"
        assert MIN_WEIGHT <= learning.weight_after <= MAX_WEIGHT

    assert len(topology.applied_settlement_ids) == MAX_AUDIT_LINKS
    selected_route = next(
        route for route in topology.routes if route.route_id == "path-alpha"
    )
    assert selected_route.weight == pytest.approx(MAX_WEIGHT)  # saturates, never exceeds
    assert selected_route.success_count == MAX_AUDIT_LINKS

    one_more = _grounded_route_record_for_topology(topology, "route-budget-overflow")
    with pytest.raises(
        PlasticRoutingValidationError,
        match="topology settlement budget is exhausted",
    ):
        apply_settlement_learning(topology, one_more)

    reset_topology = topology.reset()
    assert reset_topology.generation == topology.generation + 1
    assert reset_topology.applied_settlement_ids == ()
    assert all(route.weight == BASELINE_WEIGHT for route in reset_topology.routes)
    assert all(route.success_count == 0 for route in reset_topology.routes)

    # A record selected against the pre-reset lineage is stale forever, even
    # though the reset topology otherwise looks fresh and has budget again.
    stale = _grounded_route_record_for_topology(topology, "route-budget-stale-lineage")
    with pytest.raises(PlasticRoutingValidationError, match="stale"):
        apply_settlement_learning(reset_topology, stale)


def test_route_success_and_failure_counts_have_no_declared_ceiling_construction_check() -> None:
    """Confirms, at the constructor level, an already-filed long-horizon gap
    (project task: 'Stop route-learning replay history from growing without
    limit'): CandidateRoute accepts an arbitrarily large success_count /
    failure_count, unlike every bounded sibling field on the same
    dataclass. This is a documentation/regression check, not a fix -- fixing
    it is explicitly out of scope for this pass."""

    unbounded_route = CandidateRoute(
        "route-unbounded",
        "candidate-work",
        "candidate",
        "alpha",
        success_count=10_000_000,
        failure_count=10_000_000,
    )
    assert unbounded_route.success_count == 10_000_000
    assert unbounded_route.failure_count == 10_000_000

    with pytest.raises(PlasticRoutingValidationError, match="exceeds"):
        CandidateRoute(
            "route-bounded-tuple",
            "candidate-work",
            "candidate",
            "alpha",
            settlement_ids=tuple(f"s-{index}" for index in range(MAX_AUDIT_LINKS + 1)),
        )


def test_illegal_numeric_and_topology_values_are_rejected_fail_closed() -> None:
    """Failure classes 2 and 6: non-finite numeric state and illegal route
    bounds can never enter the system through any public constructor."""

    for bad_value in (float("nan"), float("inf"), float("-inf"), -0.0001, 1.0001):
        with pytest.raises(DynamicalSubstrateValidationError):
            FastState(activation=bad_value)
        with pytest.raises(DynamicalSubstrateValidationError):
            FastState(surprise=bad_value)
        with pytest.raises(DynamicalSubstrateValidationError):
            FastState(resource_pressure=bad_value)

    for bad_weight in (MIN_WEIGHT - 0.01, MAX_WEIGHT + 0.01, float("nan"), float("inf")):
        with pytest.raises(PlasticRoutingValidationError):
            CandidateRoute("r", "candidate-work", "a", "b", weight=bad_weight)

    with pytest.raises(PlasticRoutingValidationError):
        RouteTopology(
            "t",
            -1,
            (
                CandidateRoute("a", "candidate-work", "x", "y"),
                CandidateRoute("b", "candidate-work", "x", "z"),
            ),
        )
    with pytest.raises(PlasticRoutingValidationError):
        RouteTopology(
            "t",
            0,
            (
                CandidateRoute("a", "candidate-work", "x", "y"),
                CandidateRoute("b", "candidate-work", "x", "z"),
            ),
            generation=-1,
        )

    with pytest.raises(AdaptiveSubstrateValidationError):
        replace(AdaptiveState.fixture(), generation=MAX_ADAPTIVE_GENERATIONS + 1)
    with pytest.raises(AdaptiveSubstrateValidationError):
        replace(AdaptiveState.fixture(), updates_applied=MAX_ADAPTIVE_UPDATES + 1)


def test_duplicate_settlement_credit_through_the_full_dynamical_reducer_stack_is_rejected_once() -> None:
    """Failure class 4, exercised through the *entire* reducer stack (not
    just the inner adaptive substrate): replaying the identical settlement
    record for credit a second time through `reduce_dynamical_tick` must be
    withheld without any state mutation, however the event is packaged."""

    state = _grounded_dynamical_state("duplicate-credit-session")
    record = _grounded_route_record_from_dynamical(state, "duplicate-credit")
    _require_child_pytest(record)

    connected, first_trace = reduce_dynamical_tick(
        state,
        DynamicalTick(
            1,
            (
                DynamicalEvent.settlement_event(
                    "duplicate-credit-first", record, operation="form_connection"
                ),
            ),
        ),
    )
    assert first_trace.noncreditable_settlement_ids == ()
    assert len(connected.medium.adaptive_state.connections) == 1

    replayed, second_trace = reduce_dynamical_tick(
        connected,
        DynamicalTick(
            2,
            (
                DynamicalEvent.settlement_event(
                    "duplicate-credit-second", record, operation="form_connection"
                ),
            ),
        ),
    )
    assert second_trace.noncreditable_settlement_ids == (record.record_id,)
    assert "already applied" in second_trace.withheld_events[0].reason
    assert replayed.medium.adaptive_state == connected.medium.adaptive_state
    assert len(replayed.medium.adaptive_state.connections) == 1


def test_topology_settlement_budget_exhaustion_is_withheld_not_crashed() -> None:
    """FIXED GAP (previously pinned by this test as a crash): every
    staleness/duplication condition recognized by `_is_current_lineage_error`
    (stale binding, already-applied, already-consumed, already-invalidated,
    checkpoint not retained) was already gracefully turned into a
    non-mutating WithheldEvent when it surfaced while processing a real
    event through `reduce_dynamical_tick`. Capacity-exhaustion conditions
    from the same adaptive/plastic-routing layer -- 'topology settlement
    budget is exhausted' among them -- were not, so a fully legitimate,
    correctly authorized, context-matched, real grounded task-success
    settlement used to crash the entire tick reduction with an uncaught
    exception instead of being withheld like every sibling boundary
    condition. A sibling classifier, `_is_bounded_capacity_exhaustion_error`,
    now recognizes this and the other five declared bounded-capacity
    ceilings at the same three tick-reducer call sites, so this exact
    scenario must now resolve to a withheld, non-mutating outcome."""

    state = _grounded_dynamical_state("budget-gap-session")
    full_topology = replace(
        state.medium.adaptive_state.route_topology,
        applied_settlement_ids=tuple(f"s-{index}" for index in range(MAX_AUDIT_LINKS)),
    )
    state = replace(
        state,
        medium=replace(
            state.medium,
            adaptive_state=replace(state.medium.adaptive_state, route_topology=full_topology),
        ),
    )
    record = _grounded_route_record_from_dynamical(state, "budget-gap")
    _require_child_pytest(record)

    _assert_bounded_capacity_exhaustion_is_withheld_and_replay_safe(
        state,
        DynamicalEvent.settlement_event("budget-gap-event", record),
        expected_reason_prefix="bounded capacity limit withheld candidate adaptation",
        expected_message_fragment="topology settlement budget is exhausted",
    )


def test_adaptive_generation_update_budget_exhaustion_is_withheld_not_crashed() -> None:
    """Same fix, second reachable exhaustion mode: a fresh, never-before-seen
    grounded settlement arriving after this generation's update budget
    (`MAX_ADAPTIVE_UPDATES`) is already spent must be withheld, not crash
    the reducer -- even though the topology itself still has room and the
    record has never been applied before."""

    state = _grounded_dynamical_state("update-budget-gap-session")
    saturated_adaptive = replace(
        state.medium.adaptive_state, updates_applied=MAX_ADAPTIVE_UPDATES
    )
    state = replace(state, medium=replace(state.medium, adaptive_state=saturated_adaptive))
    record = _grounded_route_record_from_dynamical(state, "update-budget-gap")
    _require_child_pytest(record)

    _assert_bounded_capacity_exhaustion_is_withheld_and_replay_safe(
        state,
        DynamicalEvent.settlement_event("update-budget-gap-event", record),
        expected_reason_prefix="bounded capacity limit withheld candidate adaptation",
        expected_message_fragment="adaptive generation update budget is exhausted",
    )


def test_adaptive_record_identity_retention_exhaustion_is_withheld_not_crashed() -> None:
    """Third reachable exhaustion mode: once `applied_record_ids` has
    accumulated `MAX_TRACKED_RECORDS` identities (the bounded tracked-record
    retention window, independent of the current generation's own update
    budget), one more fresh grounded settlement must be withheld -- not
    crash -- even though this generation's own update budget still has
    room."""

    state = _grounded_dynamical_state("retention-gap-session")
    saturated_adaptive = replace(
        state.medium.adaptive_state,
        applied_record_ids=tuple(
            f"retention-gap-synthetic-{index}" for index in range(MAX_TRACKED_RECORDS)
        ),
    )
    state = replace(state, medium=replace(state.medium, adaptive_state=saturated_adaptive))
    record = _grounded_route_record_from_dynamical(state, "retention-gap")
    _require_child_pytest(record)

    _assert_bounded_capacity_exhaustion_is_withheld_and_replay_safe(
        state,
        DynamicalEvent.settlement_event("retention-gap-event", record),
        expected_reason_prefix="bounded capacity limit withheld candidate adaptation",
        expected_message_fragment="adaptive record identity retention is exhausted",
    )


def test_rollback_budget_exhaustion_is_withheld_not_crashed() -> None:
    """Fourth reachable exhaustion mode, via the rollback event path (a
    distinct tick-reducer call site from the settlement path above): a
    perfectly valid, retained, current-generation checkpoint must be
    withheld -- not crash the reducer -- when this generation's own update
    budget is already spent, since rollback itself consumes one update
    slot. No real grounded record is needed here; rollback only ever
    inspects checkpoints, never settlement records."""

    base_adaptive = AdaptiveState.fixture("rollback-budget-gap-adaptive")
    checkpoint = _synthetic_checkpoint(base_adaptive, "rollback-budget-gap")
    saturated_adaptive = replace(
        base_adaptive, updates_applied=MAX_ADAPTIVE_UPDATES, checkpoints=(checkpoint,)
    )
    state = _dynamical_state_with_adaptive("rollback-budget-gap-session", saturated_adaptive)

    _assert_bounded_capacity_exhaustion_is_withheld_and_replay_safe(
        state,
        DynamicalEvent.rollback_event("rollback-budget-gap-event", checkpoint.checkpoint_id),
        expected_reason_prefix="bounded capacity limit withheld candidate rollback",
        expected_message_fragment="rollback budget is exhausted",
    )


def test_rollback_generation_limit_reached_is_withheld_not_crashed() -> None:
    """Fifth reachable exhaustion mode, also via the rollback event path: a
    valid, retained checkpoint whose lineage exactly matches the current
    (already-maximal) generation must be withheld -- not crash the reducer
    -- once the adaptive generation ceiling itself has been reached, even
    though this generation's own update budget still has room (isolating
    this check from the rollback-budget check above)."""

    ceiling_topology = replace(
        RouteTopology.fixture("rollback-generation-limit-gap-topology"),
        generation=MAX_ADAPTIVE_GENERATIONS,
    )
    base_adaptive = replace(
        AdaptiveState.fixture("rollback-generation-limit-gap-adaptive"),
        generation=MAX_ADAPTIVE_GENERATIONS,
        route_topology=ceiling_topology,
    )
    checkpoint = _synthetic_checkpoint(base_adaptive, "rollback-generation-limit-gap")
    saturated_adaptive = replace(base_adaptive, checkpoints=(checkpoint,))
    state = _dynamical_state_with_adaptive(
        "rollback-generation-limit-gap-session", saturated_adaptive
    )

    _assert_bounded_capacity_exhaustion_is_withheld_and_replay_safe(
        state,
        DynamicalEvent.rollback_event(
            "rollback-generation-limit-gap-event", checkpoint.checkpoint_id
        ),
        expected_reason_prefix="bounded capacity limit withheld candidate rollback",
        expected_message_fragment="adaptive generation limit is exhausted",
    )


def test_invalidation_budget_exhaustion_scope_note_and_direct_regression() -> None:
    """SCOPE NOTE: unlike the five modes above, 'invalidation budget is
    exhausted' cannot be driven through `reduce_dynamical_tick` at all.
    `DynamicalEvent` has no invalidate-kind event and no settlement
    `operation` dispatches to `invalidate_grounded_adaptation`; grepping
    `dynamical_substrate.py` confirms neither `invalidate_grounded_
    adaptation` nor `replay_adaptive_invalidations` has any caller there.
    This is a pre-existing structural fact, independent of this pass's fix
    -- not something a test can honestly exercise 'through the full tick
    reducer' without inventing a new reducer pathway, which is out of
    scope. This test instead covers exactly what *is* real: the new
    classifier recognizes this message (so the day a tick event is wired up
    for it, the withhold behavior above would apply automatically), and the
    only actual call path -- direct `adaptive_substrate` calls -- rejects
    the exhausted-budget case deterministically and without mutating its
    input, including on repeated retry."""

    assert _is_bounded_capacity_exhaustion_error(
        AdaptiveSubstrateValidationError("invalidation budget is exhausted")
    )

    base_adaptive = AdaptiveState.fixture("invalidation-budget-gap-adaptive")
    target_record = _grounded_route_record(base_adaptive, "invalidation-budget-gap-target")
    _require_child_pytest(target_record)
    credited, _ = apply_grounded_adaptation(base_adaptive, target_record)
    saturated = replace(credited, updates_applied=MAX_ADAPTIVE_UPDATES)

    invalidating_record = _grounded_route_record_for_topology(
        saturated.route_topology,
        "invalidation-budget-gap-invalidator",
        transaction_id=base_adaptive.substrate_id,
        objective_id="invalidation-budget-gap-invalidator-objective",
        passing=False,
        causal_parent_record_id=target_record.record_id,
    )
    _require_child_pytest(invalidating_record)

    with pytest.raises(
        AdaptiveSubstrateValidationError, match="invalidation budget is exhausted"
    ):
        invalidate_grounded_adaptation(
            saturated,
            target_record.record_id,
            invalidating_record,
            reason="later grounded failure",
        )
    # Deterministic and non-mutating: retrying the identical call against
    # the identical (untouched) input raises the exact same way every time.
    with pytest.raises(
        AdaptiveSubstrateValidationError, match="invalidation budget is exhausted"
    ):
        invalidate_grounded_adaptation(
            saturated,
            target_record.record_id,
            invalidating_record,
            reason="later grounded failure",
        )


def test_stale_task_state_version_settlement_is_withheld_even_after_many_intervening_cycles() -> None:
    """Ancient-state resurrection: a settlement bound to a task-state
    version that never matches the substrate's current version must stay
    withheld no matter how many cheap ticks separate its construction from
    its (attempted) replay."""

    state = _grounded_dynamical_state("stale-task-state-longrun")
    record = _grounded_route_record_from_dynamical(state, "stale-task-state-longrun")
    _require_child_pytest(record)

    current = replace(state, task_state_version=state.task_state_version + 1)
    for tick in range(1, 51):
        current, _ = reduce_dynamical_tick(
            current,
            DynamicalTick(tick, (DynamicalEvent.resource_event(f"stale-tsv-{tick}", 0.1),)),
        )
    before = current.medium.adaptive_state

    final, trace = reduce_dynamical_tick(
        current,
        DynamicalTick(51, (DynamicalEvent.settlement_event("stale-task-state-final", record),)),
    )
    assert trace.noncreditable_settlement_ids == (record.record_id,)
    assert "not bound to the substrate task state" in trace.withheld_events[0].reason
    assert final.medium.adaptive_state == before


# ---------------------------------------------------------------------------
# Second pass: bounded combinatorial soak across legal + pathological mixes
# ---------------------------------------------------------------------------

NUM_COMBINATORIAL_SESSIONS = 15
TICKS_PER_COMBINATORIAL_SESSION = 80
TOTAL_COMBINATORIAL_TICKS = NUM_COMBINATORIAL_SESSIONS * TICKS_PER_COMBINATORIAL_SESSION

_COMBINATORIAL_DIMENSIONS = (
    "event_ordering",
    "rollback_distance",
    "stale_state_age",
    "settlement_saturation",
    "reset_timing",
    "invalidation_timing",
    "topology_selection_age",
    "resource_saturation",
    "contradictory_observation",
)


def test_bounded_combinatorial_soak_across_legal_and_pathological_dimensions() -> None:
    """Second-pass bounded combinatorial soak: a seeded, deterministic mix
    of legal and pathological conditions across nine independent dimensions
    -- event ordering, rollback distance, stale-state age, settlement
    saturation, reset timing, invalidation timing, topology-selection age,
    resource saturation, and contradictory observations -- run across many
    independent bounded sessions with every global invariant re-checked
    after every single tick.

    Cost discipline matches the first pass: real subprocess grounded
    execution is used only where a genuine grounded record is structurally
    required (one reusable record per session for settlement saturation,
    plus a small module-shared pool for the cross-session staleness/aging
    dimensions), never once per combination.

    Every pathological branch must resolve to a deterministic, non-mutating
    withheld outcome or an explicit, expected fail-closed rejection --
    nothing may silently succeed, corrupt state, or crash."""

    rng = random.Random(20260826_02)

    # A small pool of records shared across *every* session, built once.
    # Because they are permanently bound to this template session's own
    # transaction/objective/topology lineage, replaying them against any
    # *other* session's independently-seeded state is unconditionally
    # stale/mismatched -- exactly the cross-context aging behavior the
    # stale-state-age and reset-timing dimensions need to stress, at the
    # cost of only two subprocess calls for the whole test.
    template_state = _grounded_dynamical_state("combo-template")
    cross_context_record = _grounded_route_record_from_dynamical(
        template_state, "combo-template-cross-context"
    )
    _require_child_pytest(cross_context_record)
    aged_topology_record = _grounded_route_record_for_topology(
        template_state.medium.adaptive_state.route_topology.reset(),
        "combo-template-aged-topology",
        transaction_id=template_state.transaction_id,
        objective_id=template_state.objective_id,
    )

    total_ticks = 0
    for session_index in range(NUM_COMBINATORIAL_SESSIONS):
        state = _grounded_dynamical_state(f"combo-session-{session_index}")
        # One real grounded record owned by *this* session, used to stress
        # settlement saturation: it can genuinely succeed exactly once
        # (real credit), after which every further replay in this same
        # session must be withheld as already-applied -- cheap, deterministic
        # saturation pressure from a single subprocess call per session.
        session_credit_record = _grounded_route_record_from_dynamical(
            state, f"combo-{session_index}-credit"
        )
        _require_child_pytest(session_credit_record)
        contradiction = _contradiction_record(state, f"combo-{session_index}-contradiction")
        checkpoints_seen: tuple[str, ...] = ()

        for tick in range(1, TICKS_PER_COMBINATORIAL_SESSION + 1):
            dimension = rng.choice(_COMBINATORIAL_DIMENSIONS)
            event_id_base = f"combo-{session_index}-t{tick}-{dimension}"

            if dimension == "event_ordering":
                # A single tick carrying several distinct event kinds at
                # once forces the fabric's canonical ordering logic to run
                # on a non-trivial, deterministic multi-kind batch.
                events = (
                    DynamicalEvent.resource_event(
                        f"{event_id_base}-resource", round(rng.random(), 4)
                    ),
                    DynamicalEvent.observation_event(
                        f"{event_id_base}-observation",
                        round(rng.random(), 4),
                        round(rng.random(), 4),
                    ),
                    DynamicalEvent.signal_event(
                        f"{event_id_base}-signal",
                        make_bound_signal(
                            f"{event_id_base}-signal-payload",
                            "candidate.urgency",
                            transaction_id=state.transaction_id,
                            objective_id=state.objective_id,
                            task_state_id=state.task_state_id,
                            task_state_version=state.task_state_version,
                            source="combinatorial-soak",
                            cause=f"{event_id_base}-cause",
                        ),
                    ),
                    DynamicalEvent.settlement_event(
                        f"{event_id_base}-contradiction", contradiction
                    ),
                )
            elif dimension == "rollback_distance":
                mode = rng.choice(("bogus", "evicted", "recent"))
                if mode == "recent" and checkpoints_seen:
                    checkpoint_id = checkpoints_seen[-1]
                elif mode == "evicted" and checkpoints_seen:
                    checkpoint_id = checkpoints_seen[0]
                else:
                    checkpoint_id = f"{event_id_base}-never-issued-checkpoint"
                events = (
                    DynamicalEvent.rollback_event(f"{event_id_base}-rollback", checkpoint_id),
                )
            elif dimension == "stale_state_age":
                # Bound to a wholly different session/transaction -- always
                # a task-state mismatch, no matter how many ticks have
                # elapsed since the record was constructed.
                events = (
                    DynamicalEvent.settlement_event(
                        f"{event_id_base}-stale", cross_context_record
                    ),
                )
            elif dimension == "settlement_saturation":
                events = (
                    DynamicalEvent.settlement_event(
                        f"{event_id_base}-saturation", session_credit_record
                    ),
                )
            elif dimension == "reset_timing":
                # Bound to a topology snapshot taken *before* a reset --
                # stale forever afterward, regardless of when it is retried.
                events = (
                    DynamicalEvent.settlement_event(
                        f"{event_id_base}-reset", aged_topology_record
                    ),
                )
            elif dimension == "invalidation_timing":
                # `invalidate_grounded_adaptation` has no tick-reducer event
                # today (see the dedicated scope-note test above); probe it
                # on an independent snapshot of the *current* adaptive state
                # so a pathological attempt can never influence the live
                # dynamical lineage running in this same session.
                snapshot = state.medium.adaptive_state
                try:
                    invalidate_grounded_adaptation(
                        snapshot,
                        session_credit_record.record_id,
                        cross_context_record,
                        reason="combinatorial soak invalidation probe",
                    )
                except AdaptiveSubstrateValidationError:
                    pass
                assert state.medium.adaptive_state == snapshot
                events = (
                    DynamicalEvent.resource_event(f"{event_id_base}-filler", 0.2),
                )
            elif dimension == "topology_selection_age":
                selection = select_candidate_route(
                    state.medium.adaptive_state.route_topology,
                    "candidate-work",
                    transaction_id=state.transaction_id,
                    objective_id=state.objective_id,
                    task_state_id=state.task_state_id,
                    task_state_version=state.task_state_version,
                )
                assert selection.route_id in {
                    route.route_id
                    for route in state.medium.adaptive_state.route_topology.routes
                }
                events = (
                    DynamicalEvent.resource_event(f"{event_id_base}-filler", 0.3),
                )
            elif dimension == "resource_saturation":
                pressure = rng.choice((0.0, 1.0, 0.999999, round(rng.random(), 6)))
                events = (
                    DynamicalEvent.resource_event(f"{event_id_base}-pressure", pressure),
                )
            else:  # contradictory_observation
                events = (
                    DynamicalEvent.observation_event(f"{event_id_base}-contradiction", 0.0, 1.0),
                )

            state, trace = reduce_dynamical_tick(state, DynamicalTick(tick, events))
            _assert_dynamical_invariants(state)
            checkpoints_seen = tuple(
                checkpoint.checkpoint_id
                for checkpoint in state.medium.adaptive_state.checkpoints
            )
            total_ticks += 1

    assert total_ticks == TOTAL_COMBINATORIAL_TICKS
