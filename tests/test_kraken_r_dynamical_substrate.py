"""Acceptance tests for the immutable Stage 10.8 dynamical substrate."""

from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from kraken_r import (
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