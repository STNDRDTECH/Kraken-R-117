"""Focused acceptance tests for read-only recorded-execution replay."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from kraken_r import (
    Objective,
    RecordedExecution,
    RecordedExecutionMode,
    ReplayValidationError,
    replay_recorded_execution,
)


ROOT = Path(__file__).resolve().parents[1]


def fixture(mode: RecordedExecutionMode) -> tuple[Objective, RecordedExecution]:
    transaction_id = f"replay-{mode.value}"
    objective = Objective(
        f"{transaction_id}-objective",
        "Replay one immutable recorded execution.",
        provenance={"transaction_id": transaction_id},
    )
    record = RecordedExecution.fixture(
        mode, transaction_id=transaction_id, objective_id=objective.objective_id
    )
    return objective, record


@pytest.mark.parametrize(
    ("mode", "decision", "observed", "evidence_count", "learns"),
    [
        (RecordedExecutionMode.SUCCESS, "success", "success", 1, True),
        (RecordedExecutionMode.FAILURE, "failure", "failure", 1, True),
        (RecordedExecutionMode.CONTRADICTION, "contradiction", "contradiction", 2, False),
        (
            RecordedExecutionMode.INSUFFICIENT_EVIDENCE,
            "insufficient_evidence",
            "not_observed",
            0,
            False,
        ),
    ],
)
def test_replay_is_deterministic_and_preserves_record_identity(
    mode: RecordedExecutionMode,
    decision: str,
    observed: str,
    evidence_count: int,
    learns: bool,
) -> None:
    objective, record = fixture(mode)
    trace = replay_recorded_execution(objective, record)

    assert trace.to_dict() == replay_recorded_execution(objective, record).to_dict()
    assert trace.transaction_id == record.transaction_id
    assert trace.objective.objective_id == record.objective_id
    assert trace.states[4].state_id == record.task_state_id
    assert trace.states[4].version == record.task_state_version
    assert trace.action.action_id == record.action_id
    assert trace.execution.execution_id == record.execution_id
    assert tuple(item.evidence_id for item in trace.evidence) == record.evidence_ids
    assert trace.settlement.settlement_id == record.settlement_id
    assert trace.provenance["record_id"] == record.provenance["record_id"]
    assert all(
        item.provenance["record_id"] == record.provenance["record_id"]
        for item in trace.evidence
    )
    assert trace.decision.outcome == decision
    assert trace.settlement.observed_outcome == observed
    assert len(trace.evidence) == evidence_count
    assert (trace.learning_update is not None) is learns
    assert trace.stop_decision.outcome == "stop"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: replace(record, provenance={}), "provenance"),
        (lambda record: replace(record, task_state_version=4), "stale"),
        (
            lambda record: replace(
                record, observations={**record.observations, "action_id": "wrong-action"}
            ),
            "mismatched",
        ),
        (
            lambda record: replace(
                record,
                status="not_observed",
                observations={
                    **record.observations,
                    "observed": False,
                    "self_reported_outcome": "success",
                },
                evidence_ids=(),
                learning_update_id=None,
                    provenance={
                        **record.provenance,
                        "evidence_ids": (),
                        "learning_update_id": None,
                    },
            ),
            "self-report",
        ),
        (
            lambda record: replace(record, status="failed"),
            "status conflicts",
        ),
    ],
)
def test_replay_rejects_ungrounded_or_incoherent_records(mutation, message: str) -> None:
    objective, record = fixture(RecordedExecutionMode.SUCCESS)

    with pytest.raises(ReplayValidationError, match=message):
        replay_recorded_execution(objective, mutation(record))


@pytest.mark.parametrize(
    "identity",
    [
        "evidence_ids",
        "decision_id",
        "settlement_id",
        "stop_decision_id",
        "learning_update_id",
    ],
)
def test_replay_requires_provenance_for_every_derived_identity(identity: str) -> None:
    objective, record = fixture(RecordedExecutionMode.SUCCESS)
    provenance = dict(record.provenance)
    provenance.pop(identity)

    with pytest.raises(ReplayValidationError, match=identity):
        replay_recorded_execution(objective, replace(record, provenance=provenance))


def test_recorded_execution_is_immutable_and_candidate_only() -> None:
    objective, record = fixture(RecordedExecutionMode.SUCCESS)

    with pytest.raises(TypeError):
        record.observations["observed"] = False

    legacy_paths = [
        ROOT / "rogal_core/daemon.py",
        ROOT / "rogal_core/autonomous_cycle.py",
        ROOT / ".replit",
    ]
    legacy_paths = [path for path in legacy_paths if path.exists()]
    before = {path: path.read_bytes() for path in legacy_paths}
    replay_recorded_execution(objective, record)
    assert all(path.read_bytes() == content for path, content in before.items())
    source = (ROOT / "kraken_r/replay.py").read_text()
    for forbidden in ("import rogal_core", "from rogal_core", "EventBus", "sqlite3"):
        assert forbidden not in source


def test_replay_rejects_unknown_and_cross_mode_semantic_observation_fields() -> None:
    objective, record = fixture(RecordedExecutionMode.SUCCESS)
    for observations in (
        {**record.observations, "provider_claim": "success"},
        {**record.observations, "positive_observation": False},
    ):
        with pytest.raises(ReplayValidationError, match="semantic"):
            replay_recorded_execution(
                objective, replace(record, observations=observations)
            )