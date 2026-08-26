"""First-pass long-horizon integrity soak for Kraken-R Stage 10.8/10.9.

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
    make_bound_signal,
    make_grounded_action,
    reduce_dynamical_tick,
    replay_dynamical_ticks,
    rollback_adaptive_state,
    run_constitutional_cycle,
    select_candidate_route,
)
from kraken_r.dynamical_substrate import (
    MAX_EVENT_COUNTER,
    MAX_EVENT_LOG,
    MAX_TICK_EVENTS,
    MAX_TICKS,
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


def test_topology_settlement_budget_exhaustion_crashes_the_dynamical_reducer() -> None:
    """DISCOVERED GAP, pinned but not fixed by this pass (see the filed
    follow-up task for the fix).

    Every staleness/duplication condition recognized by
    `_is_current_lineage_error` (stale binding, already-applied, already-
    consumed, already-invalidated, checkpoint not retained) is gracefully
    turned into a non-mutating WithheldEvent when it surfaces while
    processing a real event through `reduce_dynamical_tick`. Capacity-
    exhaustion conditions from the same adaptive/plastic-routing layer --
    'topology settlement budget is exhausted', and by the same code path
    'adaptive generation update budget is exhausted', 'rollback budget is
    exhausted', 'adaptive generation limit is exhausted', 'adaptive record
    identity retention is exhausted', 'invalidation budget is exhausted' --
    are NOT in that phrase list. When one of them fires, a fully
    legitimate, correctly authorized, context-matched, real grounded
    task-success settlement crashes the entire tick reduction with an
    uncaught exception instead of being withheld like every sibling
    boundary condition. This test pins today's actual (undesirable) crash
    so that fixing the classifier is a visible, deliberate change rather
    than a silent one."""

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

    with pytest.raises(
        PlasticRoutingValidationError, match="topology settlement budget is exhausted"
    ):
        reduce_dynamical_tick(
            state, DynamicalTick(1, (DynamicalEvent.settlement_event("budget-gap-event", record),))
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
