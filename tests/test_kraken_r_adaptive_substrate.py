"""Stage 10 acceptance tests for the bounded adaptive substrate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import pytest

from kraken_r import (
    ADAPTIVE_FIELD_CONSUMERS,
    AdaptiveState,
    AdaptiveSubstrateValidationError,
    CandidateConnection,
    ConnectionLifecycle,
    EvidenceAuthorityTier,
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
    derive_advisory_cognition,
    form_grounded_connection,
    invalidate_grounded_adaptation,
    make_grounded_action,
    replay_adaptive_updates,
    replay_adaptive_invalidations,
    recover_grounded_connection,
    retire_grounded_connection,
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
    causal_parent_record_id: str | None = None,
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
        causal_parent_record_id=causal_parent_record_id,
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


def _route_record(
    state: AdaptiveState,
    label: str,
    *,
    passing: bool = True,
    causal_parent_record_id: str | None = None,
):
    request, verifier, verified, trace = _grounded_record(
        label,
        passing=passing,
        causal_parent_record_id=causal_parent_record_id,
    )
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
    with pytest.raises(
        AdaptiveSubstrateValidationError, match="grounded lifecycle lineage"
    ):
        replace(
            state,
            connections=state.connections + (unrelated,),
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


def test_connection_floor_is_retained_and_cannot_consume_noop_credit() -> None:
    state = AdaptiveState.fixture()
    success = _route_record(state, "adaptive-floor-success")
    state, _ = form_grounded_connection(state, success)
    connection_id = state.connections[0].connection_id

    for index in range(3):
        failure = _route_record(state, f"adaptive-floor-failure-{index}", passing=False)
        state, audit = weaken_grounded_connection(state, failure, connection_id)
        assert audit.operation == "weaken_connection"

    assert len(state.connections) == 1
    assert state.connections[0].weight == pytest.approx(0.25)
    at_floor = _route_record(state, "adaptive-floor-noop", passing=False)
    with pytest.raises(AdaptiveSubstrateValidationError, match="retained at its minimum"):
        weaken_grounded_connection(state, at_floor, connection_id)


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


def test_settlement_and_checkpoint_lineage_fail_closed_after_state_advances() -> None:
    state = AdaptiveState.fixture("lineage-settlement")
    record = _route_record(state, "lineage-settlement-success")
    updated, _ = apply_grounded_adaptation(state, record)
    checkpoint_id = updated.checkpoints[-1].checkpoint_id

    replay_selection = select_candidate_route(
        updated.route_topology,
        "candidate-work",
        transaction_id=record.selection.transaction_id,
        objective_id=record.selection.objective_id,
        task_state_id=record.selection.task_state_id,
        task_state_version=record.selection.task_state_version,
    )
    reused_settlement = replace(
        record,
        record_id="lineage-settlement-reused-record",
        selection=replay_selection,
        provenance={
            **dict(record.provenance),
            "route_id": replay_selection.route_id,
        },
    )
    before = updated.to_dict()
    with pytest.raises(AdaptiveSubstrateValidationError, match="settlement identity"):
        apply_grounded_adaptation(updated, reused_settlement)
    assert updated.to_dict() == before

    rolled_back, _ = rollback_adaptive_state(updated, checkpoint_id)
    before = rolled_back.to_dict()
    with pytest.raises(AdaptiveSubstrateValidationError, match="checkpoint lineage is stale"):
        rollback_adaptive_state(rolled_back, checkpoint_id)
    assert rolled_back.to_dict() == before


def test_stale_topology_generation_and_replay_after_invalidation_fail_closed() -> None:
    initial = AdaptiveState.fixture("lineage-generation")
    stale = _route_record(initial, "lineage-generation-stale")
    advanced = replace(
        initial,
        route_topology=replace(
            initial.route_topology,
            generation=initial.route_topology.generation + 1,
        ),
    )
    before = advanced.to_dict()
    with pytest.raises(AdaptiveSubstrateValidationError, match="generation is stale"):
        apply_grounded_adaptation(advanced, stale)
    assert advanced.to_dict() == before

    success = _route_record(initial, "lineage-replay-success")
    reinforced, _ = apply_grounded_adaptation(initial, success)
    failure = _route_record(
        reinforced,
        "lineage-replay-failure",
        passing=False,
        causal_parent_record_id=success.record_id,
    )
    invalidated, _ = invalidate_grounded_adaptation(
        reinforced, success.record_id, failure, reason="verified contrary outcome"
    )
    before = invalidated.to_dict()
    with pytest.raises(AdaptiveSubstrateValidationError, match="already applied"):
        replay_adaptive_updates(invalidated, (("credit", success, None),))
    assert invalidated.to_dict() == before


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
        reinforced,
        "resilience-later-failure",
        passing=False,
        causal_parent_record_id=success.record_id,
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
    before = recovered.to_dict()
    with pytest.raises(AdaptiveSubstrateValidationError, match="settlement identity"):
        apply_grounded_adaptation(
            recovered,
            replace(
                later_failure,
                record_id="resilience-reused-invalidating-settlement",
                selection=select_candidate_route(
                    recovered.route_topology,
                    "candidate-work",
                    transaction_id=later_failure.selection.transaction_id,
                    objective_id=later_failure.selection.objective_id,
                    task_state_id=later_failure.selection.task_state_id,
                    task_state_version=later_failure.selection.task_state_version,
                ),
            ),
        )
    assert recovered.to_dict() == before


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


def test_operational_evidence_is_provisional_and_cannot_become_durable_state() -> None:
    state = AdaptiveState.fixture("stage-11-operational")
    before = state.to_dict()
    objective = Objective(
        "stage-11-operational-objective",
        "Observe one deterministic provisional result.",
        provenance={"transaction_id": "stage-11-operational-transaction"},
    )
    trace = run_constitutional_cycle(objective)
    assert trace.evidence and all(
        item.grade.value == "operational" for item in trace.evidence
    )

    cognition = derive_advisory_cognition(state, trace.evidence)

    assert cognition.evidence_tier is EvidenceAuthorityTier.OPERATIONAL_PROVISIONAL
    assert cognition.advisory_only is True
    assert cognition.dispatchable is False
    assert cognition.authorizes_execution is False
    assert state.to_dict() == before
    assert set(ADAPTIVE_FIELD_CONSUMERS) == {
        "route_topology",
        "connections",
        "tactic_scores",
        "active_tactic_id",
        "applied_record_ids",
        "consumed_settlement_ids",
        "audits",
        "checkpoints",
        "failure_streak",
        "tactic_streak",
        "invalidated_record_ids",
    }


def test_causal_invalidation_rejects_explicitly_unrelated_failure() -> None:
    initial = AdaptiveState.fixture("stage-11-causal")
    success = _route_record(initial, "stage-11-causal-success")
    reinforced, _ = apply_grounded_adaptation(initial, success)
    failure = _route_record(reinforced, "stage-11-causal-failure", passing=False)
    unrelated = _route_record(
        reinforced,
        "stage-11-causal-unrelated",
        passing=False,
        causal_parent_record_id="unrelated-adaptive-record",
    )
    before = reinforced.to_dict()

    with pytest.raises(AdaptiveSubstrateValidationError, match="causally target"):
        invalidate_grounded_adaptation(
            reinforced,
            success.record_id,
            unrelated,
            reason="temporally later is not causally related",
        )
    assert reinforced.to_dict() == before

    tampered = replace(
        failure,
        provenance={
            **dict(failure.provenance),
            "causal_parent_record_id": success.record_id,
        },
    )
    with pytest.raises(AdaptiveSubstrateValidationError, match="verified grounded"):
        invalidate_grounded_adaptation(
            reinforced,
            success.record_id,
            tampered,
            reason="wrapper-only parent injection is not attested",
        )

    with pytest.raises(AdaptiveSubstrateValidationError, match="explicit causal_parent"):
        invalidate_grounded_adaptation(
            reinforced,
            success.record_id,
            failure,
            reason="temporal order alone is insufficient",
        )


def test_causal_invalidation_replays_later_valid_route_suffix() -> None:
    initial = AdaptiveState.fixture("stage-11-suffix")
    bad = _route_record(initial, "stage-11-suffix-bad")
    after_bad, _ = apply_grounded_adaptation(initial, bad)
    valid = _route_record(after_bad, "stage-11-suffix-valid")
    after_both, _ = apply_grounded_adaptation(after_bad, valid)
    invalidator = _route_record(
        after_both,
        "stage-11-suffix-invalidator",
        passing=False,
        causal_parent_record_id=bad.record_id,
    )

    recovered, _ = invalidate_grounded_adaptation(
        after_both,
        bad.record_id,
        invalidator,
        reason="the first reinforcement was causally falsified",
    )

    route = recovered.route_topology.routes[0]
    assert route.weight == pytest.approx(0.60)
    assert route.success_count == 1
    assert valid.constitutional_trace.settlement.settlement_id in route.settlement_ids
    assert bad.constitutional_trace.settlement.settlement_id not in route.settlement_ids
    assert valid.record_id in recovered.applied_record_ids
    with pytest.raises(AdaptiveSubstrateValidationError, match="already applied"):
        apply_grounded_adaptation(recovered, valid)


def test_rollback_accepts_only_latest_trusted_checkpoint_and_preserves_prefix() -> None:
    initial = AdaptiveState.fixture("stage-11-rollback")
    first = _route_record(initial, "stage-11-rollback-first")
    first_state, _ = apply_grounded_adaptation(initial, first)
    second = _route_record(first_state, "stage-11-rollback-second", passing=False)
    second_state, _ = apply_grounded_adaptation(first_state, second)

    with pytest.raises(AdaptiveSubstrateValidationError, match="most recent trusted"):
        rollback_adaptive_state(second_state, second_state.checkpoints[0].checkpoint_id)

    restored, audit = rollback_adaptive_state(
        second_state, second_state.checkpoints[-1].checkpoint_id
    )
    assert audit.operation == "rollback"
    assert restored.route_topology.routes == first_state.route_topology.routes
    assert first.record_id in restored.applied_record_ids
    assert second.record_id in restored.applied_record_ids
    with pytest.raises(AdaptiveSubstrateValidationError, match="already applied"):
        apply_grounded_adaptation(restored, second)


def test_dormant_connection_recovers_and_retirement_is_separate_and_terminal() -> None:
    state = AdaptiveState.fixture("stage-11-recovery")
    formed_record = _route_record(state, "stage-11-recovery-form")
    state, _ = form_grounded_connection(state, formed_record)
    connection_id = state.connections[0].connection_id
    lineage_id = state.connections[0].lineage_id
    for index in range(3):
        failure = _route_record(
            state, f"stage-11-recovery-weaken-{index}", passing=False
        )
        state, _ = weaken_grounded_connection(state, failure, connection_id)
    assert state.connections[0].lifecycle == ConnectionLifecycle.DORMANT.value

    recovery = _route_record(state, "stage-11-recovery-useful")
    recovered, audit = recover_grounded_connection(state, recovery, connection_id)
    assert audit.operation == "recover_connection"
    assert recovered.connections[0].weight == pytest.approx(0.30)
    assert recovered.connections[0].lifecycle == ConnectionLifecycle.WEAKENED.value
    assert recovered.connections[0].lineage_id == lineage_id

    dormant = state
    retirement = _route_record(
        dormant, "stage-11-recovery-retirement", passing=False
    )
    retired, audit = retire_grounded_connection(
        dormant,
        retirement,
        connection_id,
        reason="repeated causally grounded failure",
    )
    assert audit.operation == "retire_connection"
    assert retired.connections[0].lifecycle == ConnectionLifecycle.RETIRED.value
    later_success = _route_record(retired, "stage-11-recovery-after-retirement")
    with pytest.raises(AdaptiveSubstrateValidationError, match="cannot recover"):
        recover_grounded_connection(retired, later_success, connection_id)


def test_grounded_advisory_cognition_reads_tactics_and_connections_without_authority() -> None:
    state = AdaptiveState.fixture("stage-11-cognition")
    record = _route_record(state, "stage-11-cognition-grounded")
    connected, _ = form_grounded_connection(state, record)
    before = connected.to_dict()

    cognition = derive_advisory_cognition(
        connected, record.evidence, pressure=_pressure(record, high=True)
    )

    assert cognition.evidence_tier is EvidenceAuthorityTier.GROUNDED_DURABLE
    assert cognition.active_tactic_id == connected.active_tactic_id
    assert cognition.tactic_scores == connected.tactic_scores
    assert cognition.preferred_connection_id == connected.connections[0].connection_id
    assert cognition.authorizes_execution is False
    assert connected.to_dict() == before