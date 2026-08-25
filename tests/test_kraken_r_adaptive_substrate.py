"""Stage 10 acceptance tests for the bounded adaptive substrate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import pytest

from kraken_r import (
    AdaptiveState,
    AdaptiveSubstrateValidationError,
    CandidateConnection,
    GroundedDeliveryLedger,
    GroundedExecutionExecutor,
    GroundedExecutionRequest,
    EpistemicOutcomeClass,
    HomeostaticSnapshot,
    MAX_ADAPTIVE_ROUTE_WEIGHT,
    MAX_ADAPTIVE_GENERATIONS,
    MAX_TACTIC_STREAK,
    Objective,
    SettlementRouteRecord,
    TaskState,
    apply_grounded_adaptation,
    form_grounded_connection,
    invalidate_grounded_adaptation,
    make_grounded_action,
    replay_adaptive_updates,
    replay_adaptive_invalidations,
    rollback_adaptive_state,
    run_constitutional_cycle,
    run_orzhaal_experiment,
    select_candidate_route,
    switch_grounded_tactic,
    validate_interaction_chain,
    weaken_grounded_connection,
)


def _grounded_record(
    label: str,
    *,
    passing: bool = True,
    files: dict[str, str] | None = None,
):
    objective = Objective(
        f"{label}-objective",
        "Produce an independently verified bounded outcome.",
        provenance={"transaction_id": f"{label}-transaction"},
    )
    action = make_grounded_action(objective.objective_id)
    authorized_state = TaskState(
        f"{objective.objective_id}-state-5",
        objective.objective_id,
        5,
        "authorized",
        values={"action_id": action.action_id},
    )
    workspace_files = files or {
        "subject.py": f"def answer():\n    return {42 if passing else 0}\n",
        "test_subject.py": (
            "from subject import answer\n\n"
            "def test_answer():\n"
            "    assert answer() == 42\n"
        ),
    }
    request = GroundedExecutionRequest(
        f"{label}-request",
        f"{label}-transaction",
        objective.objective_id,
        authorized_state.state_id,
        authorized_state.version,
        action,
        workspace_files,
        tuple(path for path in workspace_files if path.startswith("test")),
    )
    ledger = GroundedDeliveryLedger(
        Path(tempfile.mkdtemp(prefix="kraken-r-stage10-receipts-")) / "receipts.json"
    )
    executor = GroundedExecutionExecutor(delivery_ledger=ledger)
    verifier = executor.verifier()
    execution = executor.execute(request, authorized_state=authorized_state)
    verified = verifier.verify(
        execution, request=request, authorized_state=authorized_state
    )
    trace = run_constitutional_cycle(
        objective,
        grounded_execution=verified,
        grounded_request=request,
        grounded_verifier=verifier,
    )
    return request, verifier, verified, trace


def _route_record(state: AdaptiveState, label: str, *, passing: bool = True):
    request, verifier, verified, trace = _grounded_record(label, passing=passing)
    selection = select_candidate_route(
        state.route_topology,
        "candidate-work",
        transaction_id=trace.transaction_id,
        objective_id=trace.objective.objective_id,
        task_state_id=trace.states[4].state_id,
        task_state_version=trace.states[4].version,
    )
    return SettlementRouteRecord(
        f"{label}-adaptive-record",
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


def _route_record_from_execution(
    state: AdaptiveState, label: str, *, files: dict[str, str]
) -> SettlementRouteRecord:
    request, verifier, verified, trace = _grounded_record(label, files=files)
    selection = select_candidate_route(
        state.route_topology,
        "candidate-work",
        transaction_id=trace.transaction_id,
        objective_id=trace.objective.objective_id,
        task_state_id=trace.states[4].state_id,
        task_state_version=trace.states[4].version,
    )
    return SettlementRouteRecord(
        f"{label}-adaptive-record",
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
def _pressure(record: SettlementRouteRecord, *, high: bool = False) -> HomeostaticSnapshot:
    trace = record.constitutional_trace
    return HomeostaticSnapshot(
        f"{record.record_id}-pressure",
        trace.transaction_id,
        trace.objective.objective_id,
        trace.states[4].state_id,
        trace.states[4].version,
        contradiction=0.8 if high else 0.1,
        uncertainty=0.7 if high else 0.1,
        repeated_failure=0.8 if high else 0.0,
        novelty=0.4,
        resource_expenditure=0.8 if high else 0.1,
    )


def test_only_grounded_settlement_can_change_route_and_replay_is_deterministic() -> None:
    initial = AdaptiveState.fixture()
    record = _route_record(initial, "adaptive-success")
    updated, audit = apply_grounded_adaptation(initial, record)

    selected = next(
        route for route in updated.route_topology.routes
        if route.route_id == record.selection.route_id
    )
    assert audit.operation == "strengthen"
    assert selected.weight == pytest.approx(0.60)
    assert replay_adaptive_updates(
        initial, (("credit", record, None),)
    ) == updated

    declared_trace = replace(
        record.constitutional_trace,
        provenance={"source": "deterministic_fixture"},
    )
    with pytest.raises(AdaptiveSubstrateValidationError, match="grounded"):
        apply_grounded_adaptation(
            initial,
            replace(
                record,
                constitutional_trace=declared_trace,
                grounded_execution=None,
                grounded_request=None,
                grounded_verifier=None,
            ),
        )


def test_decay_recovery_and_anti_lock_in_remain_bounded() -> None:
    initial = AdaptiveState.fixture()
    first = _route_record(initial, "adaptive-decay-first")
    strengthened, _ = apply_grounded_adaptation(initial, first)
    second = _route_record(strengthened, "adaptive-decay-second")
    decayed, decay_audit = apply_grounded_adaptation(
        strengthened, second, operation="decay"
    )
    assert decay_audit.operation == "decay"
    route = next(route for route in decayed.route_topology.routes if route.route_id == "path-alpha")
    assert route.weight == pytest.approx(0.55)

    third = _route_record(decayed, "adaptive-recovery-third", passing=False)
    weakened, _ = apply_grounded_adaptation(decayed, third)
    fourth = _route_record(weakened, "adaptive-recovery-fourth")
    recovered, recovery_audit = apply_grounded_adaptation(
        weakened, fourth, operation="recover"
    )
    assert recovery_audit.operation == "recover"
    assert all(0.25 <= route.weight <= 0.75 for route in recovered.route_topology.routes)

    failed = _route_record(initial, "adaptive-direction-failure", passing=False)
    with pytest.raises(AdaptiveSubstrateValidationError, match="cannot contradict"):
        apply_grounded_adaptation(initial, failed, operation="strengthen")
    with pytest.raises(AdaptiveSubstrateValidationError, match="cannot contradict"):
        apply_grounded_adaptation(initial, first, operation="weaken")


def test_connection_limits_tactic_switch_and_advisory_signals() -> None:
    state = AdaptiveState.fixture()
    success = _route_record(state, "adaptive-connection-success")
    state, formed = form_grounded_connection(state, success)
    assert formed.operation == "form_connection"
    assert state.connections[0].connection_id == CandidateConnection.for_route(
        state.route_topology.routes[0], generation=0
    ).connection_id

    failure = _route_record(state, "adaptive-connection-failure", passing=False)
    state, weakened = weaken_grounded_connection(
        state, failure, state.connections[0].connection_id
    )
    assert weakened.operation == "weaken_connection"
    assert state.connections[0].weight == pytest.approx(0.40)

    unrelated = CandidateConnection(
        "candidate-connection-6f11590ea0b3b512", "other", "edge"
    )
    malformed = replace(state, connections=state.connections + (unrelated,))
    unrelated_failure = _route_record(
        malformed, "adaptive-unrelated-connection-failure", passing=False
    )
    with pytest.raises(AdaptiveSubstrateValidationError, match="selected route"):
        weaken_grounded_connection(
            malformed, unrelated_failure, unrelated.connection_id
        )

    tactic_record = _route_record(state, "adaptive-tactic-failure", passing=False)
    pressure = _pressure(tactic_record, high=True)
    state, tactic = switch_grounded_tactic(state, tactic_record, pressure)
    assert tactic.operation == "switch_tactic"
    assert state.active_tactic_id == "alternate"
    signals = pressure.signals()
    assert len(signals) == 5
    assert all(signal.evidence_grade.value == "declared" for signal in signals)
    assert all(signal.payload["advisory_only"] is True for signal in signals)


def test_duplicate_record_remains_rejected_after_rollback() -> None:
    state = AdaptiveState.fixture()
    record = _route_record(state, "adaptive-rollback")
    updated, _ = apply_grounded_adaptation(state, record)
    checkpoint_id = updated.checkpoints[-1].checkpoint_id
    rolled_back, rollback = rollback_adaptive_state(updated, checkpoint_id)

    assert rollback.operation == "rollback"
    assert rolled_back.generation == updated.generation + 1
    assert record.record_id in rolled_back.applied_record_ids
    with pytest.raises(AdaptiveSubstrateValidationError, match="already applied"):
        apply_grounded_adaptation(rolled_back, record)
    with pytest.raises(AdaptiveSubstrateValidationError, match="generation limit"):
        replace(rolled_back, generation=MAX_ADAPTIVE_GENERATIONS + 1)


def test_infrastructure_and_execution_failures_cannot_reshape_adaptation() -> None:
    state = AdaptiveState.fixture("non-creditable-adaptive")
    malformed = _route_record_from_execution(
        state,
        "adaptive-malformed-child",
        files={"test_broken.py": "def test_broken(:\n    pass\n"},
    )

    assert (
        malformed.grounded_execution.epistemic_class
        is EpistemicOutcomeClass.EXECUTION_FAILURE
    )
    assert malformed.constitutional_trace.evidence == ()
    assert malformed.constitutional_trace.learning_update is None
    before = state.to_dict()
    with pytest.raises(AdaptiveSubstrateValidationError, match="non-creditable"):
        apply_grounded_adaptation(state, malformed)
    with pytest.raises(AdaptiveSubstrateValidationError, match="non-creditable"):
        form_grounded_connection(state, malformed)
    with pytest.raises(AdaptiveSubstrateValidationError, match="non-creditable"):
        switch_grounded_tactic(state, malformed, _pressure(malformed, high=True))
    assert state.to_dict() == before


def test_later_grounded_failure_can_invalidate_bad_reinforcement() -> None:
    initial = AdaptiveState.fixture("resilience-invalidation")
    success = _route_record(initial, "resilience-prior-success")
    reinforced, reinforcement_audit = apply_grounded_adaptation(initial, success)
    assert reinforcement_audit.operation == "strengthen"
    assert reinforced.route_topology.routes[0].weight == pytest.approx(0.60)

    later_failure = _route_record(
        reinforced, "resilience-later-failure", passing=False
    )
    recovered, invalidation = invalidate_grounded_adaptation(
        reinforced,
        success.record_id,
        later_failure,
        reason="later grounded outcome no longer supports prior reinforcement",
    )

    assert invalidation.operation == "invalidate_evidence"
    assert invalidation.epistemic_class == EpistemicOutcomeClass.TASK_FAILURE.value
    assert success.record_id in recovered.invalidated_record_ids
    assert success.record_id in recovered.applied_record_ids
    assert later_failure.record_id in recovered.applied_record_ids
    assert recovered.route_topology.routes[0].weight == pytest.approx(0.50)
    assert replay_adaptive_invalidations(
        reinforced,
        ((success.record_id, later_failure, "later grounded outcome no longer supports prior reinforcement"),),
    ) == recovered
    with pytest.raises(AdaptiveSubstrateValidationError, match="already invalidated"):
        invalidate_grounded_adaptation(
            recovered,
            success.record_id,
            _route_record(recovered, "resilience-repeat-failure", passing=False),
            reason="repeat",
        )
    with pytest.raises(AdaptiveSubstrateValidationError, match="already applied"):
        apply_grounded_adaptation(recovered, success)


def test_resilience_caps_monopoly_escapes_lock_in_and_rejects_stale_selection() -> None:
    state = AdaptiveState.fixture("resilience-bounds")
    for label in ("first", "second", "third"):
        state, _ = apply_grounded_adaptation(
            state, _route_record(state, f"resilience-success-{label}")
        )
    selected = state.route_topology.routes[0]
    assert selected.weight == pytest.approx(MAX_ADAPTIVE_ROUTE_WEIGHT)

    stale = _route_record(state, "resilience-stale-selection")
    advanced, _ = apply_grounded_adaptation(
        state, _route_record(state, "resilience-advance", passing=False)
    )
    with pytest.raises(AdaptiveSubstrateValidationError, match="stale"):
        apply_grounded_adaptation(advanced, stale)

    locked = replace(advanced, tactic_streak=MAX_TACTIC_STREAK)
    escape = _route_record(locked, "resilience-lock-in-escape")
    escaped, audit = switch_grounded_tactic(locked, escape, _pressure(escape))
    assert audit.operation == "switch_tactic"
    assert escaped.active_tactic_id == "alternate"
    assert escaped.tactic_streak == 0


def test_orzhaal_is_disposable_and_never_promotable() -> None:
    state = AdaptiveState.fixture()
    before = state.to_dict()
    result = run_orzhaal_experiment(state, experiment_id="orzhaal-compare")

    assert result.promotable is False
    assert result.canonical_mutation is False
    assert state.to_dict() == before
    with pytest.raises(AdaptiveSubstrateValidationError, match="bounded"):
        run_orzhaal_experiment(
            state, experiment_id="orzhaal-invalid", weight_delta=0.10
        )


def test_interaction_validation_composes_grounded_adaptation_without_authority() -> None:
    state = AdaptiveState.fixture("interaction-adaptive")
    record = _route_record(state, "adaptive-interaction")
    report = validate_interaction_chain(
        record.constitutional_trace,
        grounded_execution=record.grounded_execution,
        grounded_request=record.grounded_request,
        grounded_verifier=record.grounded_verifier,
        adaptive_state=state,
        adaptive_record=record,
        homeostatic_snapshot=_pressure(record),
    )

    assert report.grounded_execution is True
    assert report.adaptive_audit is not None
    assert report.adaptive_audit.operation == "strengthen"