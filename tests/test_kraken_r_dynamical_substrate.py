"""Acceptance tests for the immutable Stage 10.8 dynamical substrate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import pytest

from kraken_r import (
    AdaptiveSubstrateValidationError,
    DynamicalEvent,
    DynamicalState,
    DynamicalSubstrateValidationError,
    DynamicalTick,
    GroundedDeliveryLedger,
    GroundedExecutionExecutor,
    GroundedExecutionRequest,
    Objective,
    SettlementRouteRecord,
    TaskState,
    make_bound_signal,
    make_grounded_action,
    reduce_dynamical_tick,
    replay_dynamical_ticks,
    run_constitutional_cycle,
    select_candidate_route,
)
from kraken_r.dynamical_substrate import MAX_TICKS
from kraken_r.dynamical_substrate import (
    MAX_EVENT_COUNTER,
    MAX_EVENT_LOG,
    MAX_SIGNAL_TOPICS,
    DynamicalEventFabric,
)
from kraken_r.physiology import OperatingRegime


def _grounded_record(
    state: DynamicalState, label: str, *, passing: bool = True
) -> SettlementRouteRecord:
    """Create one independently verified record bound to the fixture context."""

    objective = Objective(
        state.objective_id,
        "Produce an independently verified dynamical substrate fixture.",
        provenance={"transaction_id": state.transaction_id},
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
        "test_subject.py": "from subject import answer\n\ndef test_answer():\n    assert answer() == 42\n",
    }
    request = GroundedExecutionRequest(
        f"{label}-request",
        state.transaction_id,
        objective.objective_id,
        authorized.state_id,
        authorized.version,
        action,
        files,
        ("test_subject.py",),
    )
    ledger = GroundedDeliveryLedger(
        Path(tempfile.mkdtemp(prefix="kraken-r-dynamical-receipts-")) / "receipts.json"
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
        state.medium.adaptive_state.route_topology,
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


def _signal(state: DynamicalState, signal_id: str = "dynamical-urgency"):
    return make_bound_signal(
        signal_id,
        "candidate.urgency",
        transaction_id=state.transaction_id,
        objective_id=state.objective_id,
        task_state_id=state.task_state_id,
        task_state_version=state.task_state_version,
        source="dynamical-test",
        cause=f"{signal_id}-cause",
    )


def _grounded_state(label: str) -> DynamicalState:
    objective_id = f"{label}-objective"
    return DynamicalState.fixture(
        f"{label}-substrate",
        transaction_id=f"{label}-transaction",
        objective_id=objective_id,
        task_state_id=f"{objective_id}-state-5",
        task_state_version=5,
    )


def _require_child_pytest(record: SettlementRouteRecord) -> None:
    if record.grounded_execution.epistemic_class.value != "task_success":
        pytest.skip(
            "grounded child pytest is unavailable in this environment; "
            "the default-runtime reliability task covers that prerequisite"
        )


def test_explicit_tick_replay_is_deterministic_and_non_authoritative() -> None:
    initial = DynamicalState.fixture()
    ticks = (
        DynamicalTick(
            1,
            (
                DynamicalEvent.signal_event("tick-1-signal", _signal(initial)),
                DynamicalEvent.observation_event("tick-1-observation", 0.40, 0.40),
            ),
        ),
        DynamicalTick(2, (DynamicalEvent.resource_event("tick-2-resource", 0.20),)),
    )

    first, first_traces = replay_dynamical_ticks(initial, ticks)
    second, second_traces = replay_dynamical_ticks(initial, ticks)

    assert first.to_dict() == second.to_dict()
    assert tuple(item.to_dict() for item in first_traces) == tuple(
        item.to_dict() for item in second_traces
    )
    assert first.tick == 2
    assert first.medium.adaptive_state == initial.medium.adaptive_state
    assert first.fast.signal_topics == ("candidate.urgency",)
    assert first.instrumentation.activity > 0


def test_surprise_and_homeostasis_change_dynamics_under_ablation() -> None:
    initial = DynamicalState.fixture()
    stable, stable_trace = reduce_dynamical_tick(
        initial,
        DynamicalTick(
            1,
            (
                DynamicalEvent.signal_event("stable-signal", _signal(initial, "stable")),
                DynamicalEvent.observation_event("stable-observation", 0.50, 0.50),
            ),
        ),
    )
    surprised, surprised_trace = reduce_dynamical_tick(
        initial,
        DynamicalTick(
            1,
            (
                DynamicalEvent.signal_event("surprised-signal", _signal(initial, "surprised")),
                DynamicalEvent.observation_event("surprised-observation", 0.0, 1.0),
                DynamicalEvent.resource_event("surprised-resource", 1.0),
            ),
        ),
    )

    assert stable_trace.inhibited is False
    assert surprised_trace.inhibited is True
    assert surprised.fast.surprise == 1.0
    assert surprised.fast.activation < stable.fast.activation
    assert surprised.instrumentation.inhibition_events == 1
    assert surprised.instrumentation.resource_pressure == 1.0


def test_noncreditable_settlement_preserves_provenance_without_learning() -> None:
    state = DynamicalState.fixture()
    objective = Objective(
        state.objective_id,
        "Produce a contradiction with no grounded credit.",
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
    record = SettlementRouteRecord(
        "noncreditable-record",
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

    next_state, tick_trace = reduce_dynamical_tick(
        state,
        DynamicalTick(
            1, (DynamicalEvent.settlement_event("noncreditable-event", record),)
        ),
    )

    assert tick_trace.noncreditable_settlement_ids == ("noncreditable-record",)
    assert next_state.medium.adaptive_state == state.medium.adaptive_state
    assert next_state.event_log == ("noncreditable-event",)
    assert next_state.withheld_events[0].to_dict() == {
        "event_id": "noncreditable-event",
        "kind": "settlement",
        "reason": tick_trace.withheld_events[0].reason,
        "record_id": "noncreditable-record",
        "settlement_id": trace.settlement.settlement_id,
        "transaction_id": state.transaction_id,
        "objective_id": state.objective_id,
        "task_state_id": state.task_state_id,
        "task_state_version": state.task_state_version,
        "route_id": selection.route_id,
        "evidence_ids": [item.evidence_id for item in trace.evidence],
        "epistemic_class": None,
        "operation": "credit",
        "checkpoint_id": None,
    }


def test_grounded_route_connection_decay_and_rollback_are_bounded() -> None:
    state = _grounded_state("connection-success")
    connection_record = _grounded_record(state, "connection-success")
    _require_child_pytest(connection_record)
    connected, connection_trace = reduce_dynamical_tick(
        state,
        DynamicalTick(
            1,
            (
                DynamicalEvent.settlement_event(
                    "connection-event", connection_record, operation="form_connection"
                ),
            ),
        ),
    )
    assert [item.operation for item in connection_trace.adaptive_audits] == [
        "form_connection"
    ], connection_trace.to_dict()
    assert len(connected.medium.adaptive_state.connections) == 1

    restored, rollback_trace = reduce_dynamical_tick(
        connected,
        DynamicalTick(
            2,
            (
                DynamicalEvent.rollback_event(
                    "rollback-event",
                    connected.medium.adaptive_state.checkpoints[-1].checkpoint_id,
                ),
            ),
        ),
    )
    assert rollback_trace.adaptive_audits[0].operation == "rollback"
    assert restored.instrumentation.rollback_events == 1
    assert restored.medium.adaptive_state.connections == ()

    decay_state = _grounded_state("decay-success")
    decay_record = _grounded_record(decay_state, "decay-success")
    decayed, decay_trace = reduce_dynamical_tick(
        decay_state,
        DynamicalTick(
            1,
            (
                DynamicalEvent.settlement_event(
                    "decay-event", decay_record, operation="decay"
                ),
            ),
        ),
    )
    assert decay_trace.adaptive_audits[0].operation == "decay"
    assert decayed.instrumentation.decay_events == 1
    assert decayed.instrumentation.plasticity_events == 1


def test_inhibition_withholds_rollback_and_grounded_credit_with_provenance() -> None:
    state = _grounded_state("inhibition-grounded-success")
    record = _grounded_record(state, "inhibition-grounded-success")
    _require_child_pytest(record)
    connected, _ = reduce_dynamical_tick(
        state,
        DynamicalTick(
            1,
            (
                DynamicalEvent.settlement_event(
                    "inhibition-connect", record, operation="form_connection"
                ),
            ),
        ),
    )
    assert connected.medium.adaptive_state.checkpoints, "grounded connection was withheld"
    checkpoint_id = connected.medium.adaptive_state.checkpoints[-1].checkpoint_id
    next_state, trace = reduce_dynamical_tick(
        connected,
        DynamicalTick(
            2,
            (
                DynamicalEvent.observation_event("inhibition-surprise", 0.0, 1.0),
                DynamicalEvent.rollback_event("inhibition-rollback", checkpoint_id),
            ),
        ),
    )

    assert trace.inhibited is True
    assert next_state.medium.adaptive_state == connected.medium.adaptive_state
    assert next_state.instrumentation.rollback_events == 0
    assert trace.withheld_events[0].to_dict()["checkpoint_id"] == checkpoint_id
    assert trace.withheld_events[0].reason == "advisory physiology inhibited candidate rollback"


def test_event_identity_and_tick_order_fail_closed() -> None:
    initial = DynamicalState.fixture()
    event = DynamicalEvent.resource_event("only-once", 0.20)
    state, _ = reduce_dynamical_tick(initial, DynamicalTick(1, (event,)))

    with pytest.raises(DynamicalSubstrateValidationError, match="already consumed"):
        reduce_dynamical_tick(state, DynamicalTick(2, (event,)))
    with pytest.raises(DynamicalSubstrateValidationError, match="exactly one"):
        reduce_dynamical_tick(initial, DynamicalTick(2))


def test_same_kind_events_are_canonically_ordered_regardless_of_submission_order() -> None:
    """Two same-kind events in one tick must reduce identically no matter what
    order the caller supplies them in -- ``DynamicalEventFabric.order`` must
    break ties by ``event_id``, not by submission order, or an ambiguous
    same-kind ordering could make the tick's outcome caller-dependent."""

    first = DynamicalEvent.resource_event("resource-aaa", 0.20)
    second = DynamicalEvent.resource_event("resource-bbb", 0.40)

    forward = DynamicalEventFabric.order((first, second))
    backward = DynamicalEventFabric.order((second, first))
    assert forward == backward == (first, second)

    initial = DynamicalState.fixture()
    state_forward, trace_forward = reduce_dynamical_tick(
        initial, DynamicalTick(1, (first, second))
    )
    state_backward, trace_backward = reduce_dynamical_tick(
        initial, DynamicalTick(1, (second, first))
    )
    assert state_forward == state_backward
    assert (
        trace_forward.event_ids
        == trace_backward.event_ids
        == ("resource-aaa", "resource-bbb")
    )


def test_stale_settlements_and_checkpoints_are_provenance_only() -> None:
    initial = _grounded_state("stale-lineage")
    topology = initial.medium.adaptive_state.route_topology
    boosted = replace(
        initial,
        fast=replace(initial.fast, active_route_id=topology.routes[0].route_id),
        medium=replace(
            initial.medium,
            adaptive_state=replace(
                initial.medium.adaptive_state,
                route_topology=replace(
                    topology,
                    routes=(replace(topology.routes[0], weight=0.65), *topology.routes[1:]),
                ),
            ),
        ),
    )
    stale_record = _grounded_record(boosted, "stale-lineage-settlement")
    _require_child_pytest(stale_record)
    decayed, _ = reduce_dynamical_tick(boosted, DynamicalTick(1))
    before = decayed.to_dict()
    withheld, trace = reduce_dynamical_tick(
        decayed,
        DynamicalTick(
            2,
            (DynamicalEvent.settlement_event("stale-lineage-event", stale_record),),
        ),
    )
    assert trace.noncreditable_settlement_ids == (stale_record.record_id,)
    assert "current-state lineage withheld" in trace.withheld_events[0].reason
    assert withheld.medium == decayed.medium
    assert withheld.fast == decayed.fast
    assert withheld.slow == decayed.slow
    assert withheld.instrumentation == decayed.instrumentation
    assert withheld.event_log == ("stale-lineage-event",)
    assert before["medium"] == withheld.to_dict()["medium"]

    connected_record = _grounded_record(initial, "stale-lineage-checkpoint")
    connected, _ = reduce_dynamical_tick(
        initial,
        DynamicalTick(
            1,
            (
                DynamicalEvent.settlement_event(
                    "stale-lineage-connect", connected_record, operation="form_connection"
                ),
            ),
        ),
    )
    checkpoint_id = connected.medium.adaptive_state.checkpoints[-1].checkpoint_id
    rolled_back, _ = reduce_dynamical_tick(
        connected,
        DynamicalTick(2, (DynamicalEvent.rollback_event("stale-lineage-rollback", checkpoint_id),)),
    )
    before = rolled_back.to_dict()
    rejected, trace = reduce_dynamical_tick(
        rolled_back,
        DynamicalTick(
            3,
            (DynamicalEvent.rollback_event("stale-lineage-reused-checkpoint", checkpoint_id),),
        ),
    )
    assert "checkpoint lineage is stale" in trace.withheld_events[0].reason
    assert rejected.medium == rolled_back.medium
    assert rejected.fast == rolled_back.fast
    assert rejected.slow == rolled_back.slow
    assert rejected.instrumentation == rolled_back.instrumentation
    assert before["medium"] == rejected.to_dict()["medium"]


def test_event_identity_cannot_be_reused_after_short_audit_retention() -> None:
    state = DynamicalState.fixture("historical-event-identities")
    first = DynamicalEvent.resource_event("historical-event-0", 0.20)
    state, _ = reduce_dynamical_tick(state, DynamicalTick(1, (first,)))
    for tick in range(2, MAX_EVENT_LOG + 3):
        state, _ = reduce_dynamical_tick(
            state,
            DynamicalTick(
                tick,
                (DynamicalEvent.resource_event(f"historical-event-{tick}", 0.20),),
            ),
        )
    assert "historical-event-0" not in state.consumed_event_ids
    assert "historical-event-0" in state.historical_event_ids
    with pytest.raises(DynamicalSubstrateValidationError, match="already consumed"):
        reduce_dynamical_tick(state, DynamicalTick(MAX_EVENT_LOG + 3, (first,)))


def test_older_task_state_settlement_is_withheld_without_dynamical_mutation() -> None:
    state = _grounded_state("stale-task-state")
    record = _grounded_record(state, "stale-task-state-record")
    _require_child_pytest(record)
    newer_task_state = replace(state, task_state_version=state.task_state_version + 1)
    before = newer_task_state.to_dict()
    next_state, trace = reduce_dynamical_tick(
        newer_task_state,
        DynamicalTick(
            1,
            (DynamicalEvent.settlement_event("stale-task-state-event", record),),
        ),
    )
    assert trace.noncreditable_settlement_ids == (record.record_id,)
    assert "not bound to the substrate task state" in trace.withheld_events[0].reason
    assert next_state.medium == newer_task_state.medium
    assert next_state.fast == newer_task_state.fast
    assert next_state.slow == newer_task_state.slow
    assert next_state.instrumentation == newer_task_state.instrumentation
    assert before["medium"] == next_state.to_dict()["medium"]


def test_generator_boundaries_fail_closed_without_eager_materialization() -> None:
    def too_many_events():
        for index in range(17):
            yield DynamicalEvent.resource_event(f"generator-event-{index}", 0.20)
        raise AssertionError("event generator was consumed beyond its bounded budget")

    with pytest.raises(DynamicalSubstrateValidationError, match="tick event budget exceeded"):
        DynamicalTick(1, too_many_events())

    initial = DynamicalState.fixture()

    def too_many_ticks():
        for tick in range(1, MAX_TICKS + 1):
            yield DynamicalTick(tick)
        yield object()
        raise AssertionError("tick generator was consumed beyond its bounded budget")

    with pytest.raises(DynamicalSubstrateValidationError, match="replay tick budget exceeded"):
        replay_dynamical_ticks(initial, too_many_ticks())


def test_public_iterable_boundaries_reject_oversized_generators_incrementally() -> None:
    initial = DynamicalState.fixture()

    def oversized_events():
        for index in range(17):
            yield DynamicalEvent.resource_event(f"fabric-event-{index}", 0.20)
        raise AssertionError("event fabric read beyond its bounded budget")

    with pytest.raises(DynamicalSubstrateValidationError, match="event fabric budget exceeded"):
        DynamicalEventFabric.order(oversized_events())

    def oversized_topics():
        for index in range(MAX_SIGNAL_TOPICS + 1):
            yield f"topic-{index}"
        raise AssertionError("fast state read beyond its bounded budget")

    with pytest.raises(DynamicalSubstrateValidationError, match="signal topics budget exceeded"):
        replace(initial.fast, signal_topics=oversized_topics())

    def oversized_event_log():
        for index in range(MAX_EVENT_LOG + 1):
            yield f"event-{index}"
        raise AssertionError("state read beyond its bounded budget")

    with pytest.raises(DynamicalSubstrateValidationError, match="event_log budget exceeded"):
        replace(initial, event_log=oversized_event_log())

    def oversized_connections():
        for _ in range(9):
            yield object()
        raise AssertionError("nested adaptive state read beyond its bounded budget")

    with pytest.raises(
        AdaptiveSubstrateValidationError, match="connections exceeds its fixed retention"
    ):
        replace(initial.medium.adaptive_state, connections=oversized_connections())


def test_inactive_candidate_decay_is_explicit_neutral_and_noncrediting() -> None:
    initial = DynamicalState.fixture()
    topology = initial.medium.adaptive_state.route_topology
    active_route = topology.routes[0]
    strengthened = replace(active_route, weight=0.65)
    adaptive = replace(
        initial.medium.adaptive_state,
        route_topology=replace(
            topology,
            routes=(strengthened,) + topology.routes[1:],
        ),
    )
    state = replace(
        initial,
        fast=replace(initial.fast, active_route_id=active_route.route_id),
        medium=replace(initial.medium, adaptive_state=adaptive),
    )

    next_state, trace = reduce_dynamical_tick(state, DynamicalTick(1))

    decayed_route = next(
        route
        for route in next_state.medium.adaptive_state.route_topology.routes
        if route.route_id == active_route.route_id
    )
    assert trace.inactive_decay_route_id == active_route.route_id
    assert trace.adaptive_audits == ()
    assert decayed_route.weight == pytest.approx(0.60)
    assert next_state.medium.adaptive_state.applied_record_ids == ()
    assert next_state.medium.decay_events == 1
    # Idle decay must remain observable (invariant: unaudited inactive-route
    # decay) without ever becoming a credit-bearing AdaptiveAudit -- it
    # carries its own before/after weight on the tick trace instead.
    assert trace.inactive_decay_weight_before == pytest.approx(0.65)
    assert trace.inactive_decay_weight_after == pytest.approx(0.60)
    assert trace.to_dict()["inactive_decay_weight_before"] == pytest.approx(0.65)


def test_inactive_decay_audit_fields_are_present_or_absent_together() -> None:
    initial = DynamicalState.fixture()
    next_state, trace = reduce_dynamical_tick(
        initial,
        DynamicalTick(1, (DynamicalEvent.observation_event("obs", 0.5, 0.5),)),
    )
    # No idle-route decay happened this tick (an event was delivered), so
    # none of the three decay-audit fields should be populated.
    assert trace.inactive_decay_route_id is None
    assert trace.inactive_decay_weight_before is None
    assert trace.inactive_decay_weight_after is None

    with pytest.raises(
        DynamicalSubstrateValidationError, match="present together or absent together"
    ):
        replace(trace, inactive_decay_route_id="some-route")


def test_noninactive_ticks_cannot_trigger_automatic_candidate_decay() -> None:
    initial = DynamicalState.fixture()
    topology = initial.medium.adaptive_state.route_topology
    active_route = topology.routes[0]
    adaptive = replace(
        initial.medium.adaptive_state,
        route_topology=replace(
            topology,
            routes=(replace(active_route, weight=0.65),) + topology.routes[1:],
        ),
    )
    state = replace(
        initial,
        fast=replace(initial.fast, active_route_id=active_route.route_id),
        medium=replace(initial.medium, adaptive_state=adaptive),
    )

    observed, observed_trace = reduce_dynamical_tick(
        state,
        DynamicalTick(
            1,
            (DynamicalEvent.observation_event("active-observation", 0.50, 0.50),),
        ),
    )

    assert observed_trace.inactive_decay_route_id is None
    assert observed.medium.adaptive_state.route_topology == adaptive.route_topology
    decayed, decayed_trace = reduce_dynamical_tick(observed, DynamicalTick(2))
    assert decayed_trace.inactive_decay_route_id == active_route.route_id
    assert decayed.medium.adaptive_state.route_topology.routes[0].weight == pytest.approx(0.60)


def test_hysteresis_cooldown_carries_only_across_explicit_ticks() -> None:
    initial = DynamicalState.fixture()
    state = replace(
        initial,
        fast=replace(
            initial.fast,
            last_regime=OperatingRegime.CRITICAL,
            cooldown_remaining=2,
        ),
    )

    first, first_trace = reduce_dynamical_tick(state, DynamicalTick(1))
    second, second_trace = reduce_dynamical_tick(first, DynamicalTick(2))
    third, third_trace = reduce_dynamical_tick(second, DynamicalTick(3))

    assert first_trace.physiology.decision.cooldown_applied is True
    assert second_trace.physiology.decision.cooldown_applied is True
    assert first.fast.cooldown_remaining == 1
    assert second.fast.cooldown_remaining == 0
    assert third_trace.physiology.decision.cooldown_applied is False
    assert third.fast.last_regime is OperatingRegime.CAUTIOUS
    assert third_trace.inhibited is False


def test_retained_counters_saturate_and_reject_unbounded_constructor_values() -> None:
    initial = DynamicalState.fixture()
    topology = initial.medium.adaptive_state.route_topology
    adaptive = replace(
        initial.medium.adaptive_state,
        route_topology=replace(
            topology,
            routes=(replace(topology.routes[0], weight=0.65),) + topology.routes[1:],
        ),
    )
    saturated = replace(
        initial,
        fast=replace(initial.fast, active_route_id=topology.routes[0].route_id),
        medium=replace(
            initial.medium,
            adaptive_state=adaptive,
            turnover=MAX_EVENT_COUNTER,
            decay_events=MAX_EVENT_COUNTER,
            plasticity_events=MAX_EVENT_COUNTER,
        ),
    )
    next_state, _ = reduce_dynamical_tick(saturated, DynamicalTick(1))

    assert next_state.medium.turnover == MAX_EVENT_COUNTER
    assert next_state.medium.decay_events == MAX_EVENT_COUNTER
    assert next_state.medium.plasticity_events == MAX_EVENT_COUNTER
    with pytest.raises(DynamicalSubstrateValidationError, match="turnover exceeds"):
        replace(initial.medium, turnover=MAX_EVENT_COUNTER + 1)
    with pytest.raises(DynamicalSubstrateValidationError, match="recurrence exceeds"):
        replace(initial.slow, recurrence=MAX_TICKS + 1)
    with pytest.raises(DynamicalSubstrateValidationError, match="inhibition_events exceeds"):
        replace(initial.instrumentation, inhibition_events=MAX_TICKS + 1)


def test_saturated_failure_streak_stays_replayable_and_bounded_as_pressure() -> None:
    initial = DynamicalState.fixture()
    state = replace(
        initial,
        medium=replace(
            initial.medium,
            adaptive_state=replace(initial.medium.adaptive_state, failure_streak=4),
        ),
    )
    ticks = (DynamicalTick(1), DynamicalTick(2))

    first, first_traces = replay_dynamical_ticks(state, ticks)
    second, second_traces = replay_dynamical_ticks(state, ticks)

    assert first.to_dict() == second.to_dict()
    assert tuple(item.to_dict() for item in first_traces) == tuple(
        item.to_dict() for item in second_traces
    )
    assert first_traces[0].physiology.snapshot.routing_pressure == 1.0