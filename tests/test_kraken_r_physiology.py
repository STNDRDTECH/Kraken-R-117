"""Adversarial tests for Round 5 bounded candidate physiology."""

from __future__ import annotations

from dataclasses import replace

import pytest

from kraken_r import (
    Authority,
    CycleInvariantError,
    Objective,
    OperatingRegime,
    PhysiologyReplayRecord,
    PhysiologySnapshot,
    PhysiologyValidationError,
    RegulationEffect,
    evaluate_physiology,
    replay_constitutional_signal_path,
    replay_physiology,
    run_constitutional_cycle,
)


def _objective() -> Objective:
    return Objective(
        "round5-objective",
        "Demonstrate bounded candidate physiology without runtime authority.",
        provenance={"transaction_id": "round5-transaction"},
    )


def _snapshot(**overrides: object) -> PhysiologySnapshot:
    context: dict[str, object] = {
        "snapshot_id": "round5-snapshot",
        "transaction_id": "round5-transaction",
        "objective_id": "round5-objective",
        "task_state_id": "round5-objective-state-4",
        "task_state_version": 4,
        "memory_pressure": 0.0,
        "routing_pressure": 0.0,
        "backlog_pressure": 0.0,
        "contradiction_density": 0.0,
        "resource_pressure": 0.0,
        "protected_reserve": 1.0,
    }
    context.update(overrides)
    return PhysiologySnapshot(**context)  # type: ignore[arg-type]


def test_identical_transaction_changes_regime_only_from_internal_conditions() -> None:
    objective = _objective()
    healthy = run_constitutional_cycle(objective, physiology=_snapshot())
    stressed = run_constitutional_cycle(
        objective,
        physiology=_snapshot(
            snapshot_id="round5-critical",
            contradiction_density=0.85,
            backlog_pressure=0.90,
            resource_pressure=0.95,
            protected_reserve=0.05,
        ),
    )

    assert healthy.transaction_id == stressed.transaction_id
    assert healthy.action.action_id == stressed.action.action_id
    assert healthy.physiology_trace is not None
    assert stressed.physiology_trace is not None
    assert healthy.physiology_trace.decision.regime is OperatingRegime.PRODUCTIVE
    assert stressed.physiology_trace.decision.regime is OperatingRegime.CRITICAL
    assert healthy.states[4].phase == "authorized"
    assert stressed.states[4].phase == "inhibited"
    assert healthy.execution.status == "completed"
    assert stressed.execution.status == "not_observed"


def test_physiology_ablation_changes_only_candidate_execution_path() -> None:
    objective = _objective()
    critical = _snapshot(
        contradiction_density=0.90,
        protected_reserve=0.05,
    )

    enabled = run_constitutional_cycle(objective, physiology=critical)
    ablated = run_constitutional_cycle(objective)

    assert enabled.physiology_trace is not None
    assert enabled.physiology_trace.decision.effect is RegulationEffect.INHIBIT_ACTION
    assert enabled.states[4].phase == "inhibited"
    assert enabled.execution.observations["physiology_inhibited"] is True
    assert enabled.execution.status == "not_observed"
    assert enabled.evidence == ()
    assert enabled.decision.outcome == "insufficient_evidence"
    assert enabled.settlement.status == "insufficient_evidence"
    assert enabled.learning_update is None
    assert ablated.physiology_trace is None
    assert ablated.states[4].phase == "authorized"
    assert ablated.decision.outcome == "success"


def test_physiology_replay_is_deterministic_and_cycle_replays() -> None:
    snapshot = _snapshot(
        snapshot_id="round5-replay",
        backlog_pressure=0.72,
        contradiction_density=0.61,
    )
    record = PhysiologyReplayRecord(snapshot)

    first = replay_physiology(record)
    second = replay_physiology(record)
    direct = evaluate_physiology(
        snapshot,
        transaction_id=snapshot.transaction_id,
        objective_id=snapshot.objective_id,
        task_state_id=snapshot.task_state_id,
        task_state_version=snapshot.task_state_version,
    )
    cycle = run_constitutional_cycle(_objective(), physiology=snapshot)
    replayed_cycle = replay_constitutional_signal_path(
        _objective(), physiology=snapshot
    )

    assert first.to_dict() == second.to_dict() == direct.to_dict()
    assert first.decision.regime is OperatingRegime.RECOVERY
    assert first.decision.action_inhibited is True
    assert cycle.to_dict() == replayed_cycle.to_dict()


def test_regulation_is_bounded_and_hysteretic_without_hidden_state() -> None:
    retained = _snapshot(
        snapshot_id="round5-cooldown",
        prior_regime=OperatingRegime.CRITICAL,
        cooldown_remaining=2,
    )
    trace = replay_physiology(PhysiologyReplayRecord(retained))

    assert trace.decision.composite_pressure == 0.0
    assert trace.decision.cooldown_applied is True
    assert trace.decision.regime is OperatingRegime.CRITICAL
    assert trace.decision.action_inhibited is True

    with pytest.raises(PhysiologyValidationError, match="between 0 and 1"):
        _snapshot(memory_pressure=1.01)
    with pytest.raises(PhysiologyValidationError, match="0 through 8"):
        _snapshot(cooldown_remaining=9)


def test_physiology_cannot_create_evidence_or_override_authority() -> None:
    trace = run_constitutional_cycle(
        _objective(),
        physiology=_snapshot(
            contradiction_density=0.9,
            protected_reserve=0.05,
        ),
    )

    assert trace.physiology_trace is not None
    assert trace.physiology_trace.decision.advisory_only is True
    assert trace.action.authority is Authority.KRAKEN_CANDIDATE
    assert trace.evidence == ()
    assert trace.ground_truth is None
    assert trace.settlement.evidence_ids == ()
    assert trace.learning_update is None
    assert trace.final_state.phase == "stopped"


def test_malformed_or_stale_physiology_fails_closed_before_execution() -> None:
    snapshot = _snapshot()
    stale = replace(snapshot, task_state_version=3)

    with pytest.raises(CycleInvariantError, match="candidate physiology was rejected"):
        run_constitutional_cycle(_objective(), physiology=stale)
    with pytest.raises(PhysiologyValidationError, match="between 0 and 1"):
        replace(snapshot, contradiction_density=-0.1)
    with pytest.raises(PhysiologyValidationError, match="replay requires"):
        replay_physiology(object())  # type: ignore[arg-type]