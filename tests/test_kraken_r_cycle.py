"""Acceptance tests for the deterministic candidate-only Kraken-R cycle."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataclasses import replace

from kraken_r import (
    Authority,
    ConstitutionalCycle,
    CycleInvariantError,
    CycleMode,
    EvidenceGrade,
    Objective,
    run_constitutional_cycle,
)


ROOT = Path(__file__).resolve().parents[1]


def objective() -> Objective:
    return Objective(
        "cycle-objective",
        "Observe one bounded candidate behavior",
        success_criteria=("the observed result is recorded",),
    )


def test_success_trace_is_complete_versioned_and_stops() -> None:
    trace = run_constitutional_cycle(objective())

    assert trace.signal is trace.event
    assert trace.action.authority is Authority.KRAKEN_CANDIDATE
    assert [state.version for state in trace.states] == list(range(1, 12))
    assert trace.states[-1].phase == "stopped"
    assert trace.decision.outcome == "success"
    assert trace.execution.status == "completed"
    assert len(trace.evidence) == 1
    assert trace.evidence[0].grade is EvidenceGrade.OPERATIONAL
    assert trace.ground_truth is trace.evidence[0]
    assert trace.evidence[0].provenance["observation_origin"] == "execution_result"
    assert trace.settlement.prediction == "success"
    assert trace.learning_update is not None
    assert tuple(trace.learning_update.evidence_ids) == trace.settlement.evidence_ids
    assert trace.stop_decision.outcome == "stop"


@pytest.mark.parametrize(
    ("mode", "decision", "observed", "evidence_count", "learning"),
    [
        (CycleMode.FAILURE, "failure", "failure", 1, True),
        (CycleMode.CONTRADICTION, "contradiction", "contradiction", 2, False),
        (
            CycleMode.INSUFFICIENT_EVIDENCE,
            "insufficient_evidence",
            "not_observed",
            0,
            False,
        ),
    ],
)
def test_non_success_paths_remain_honest(
    mode: CycleMode,
    decision: str,
    observed: str,
    evidence_count: int,
    learning: bool,
) -> None:
    trace = run_constitutional_cycle(objective(), mode=mode)

    assert trace.decision.outcome == decision
    assert trace.settlement.observed_outcome == observed
    assert trace.settlement.prediction == "success"
    assert len(trace.evidence) == evidence_count
    assert trace.learning_update is not None if learning else trace.learning_update is None
    assert trace.signal.evidence_grade is EvidenceGrade.DECLARED
    assert all(
        evidence.source == "deterministic_observation"
        for evidence in trace.evidence
    )
    assert all(evidence.provenance for evidence in trace.evidence)
    assert trace.stop_decision.outcome == "stop"


@pytest.mark.parametrize("authority", [None, Authority.NONE, Authority.LEGACY_REFERENCE])
def test_action_authority_is_required_and_candidate_only(authority) -> None:
    with pytest.raises(CycleInvariantError, match="authority"):
        run_constitutional_cycle(objective(), action_authority=authority)


def test_validate_trace_rejects_an_out_of_order_phase() -> None:
    """A hand-tampered trace that skips ``authorized`` straight to ``observed``
    from ``signaled`` must be rejected even though version numbers stay
    contiguous."""

    trace = run_constitutional_cycle(objective())
    states = list(trace.states)
    # states[4] is normally "authorized"; replace it with an out-of-sequence
    # phase that "signaled" (states[3]) is not permitted to transition into.
    states[4] = replace(states[4], phase="settled")
    tampered = replace(trace, states=tuple(states))
    with pytest.raises(CycleInvariantError, match="invalid phase transition"):
        ConstitutionalCycle._validate_trace(tampered)


def test_validate_trace_rejects_broken_identity_bindings() -> None:
    trace = run_constitutional_cycle(objective())

    bad_settlement = replace(trace.settlement, decision_id="some-other-decision")
    with pytest.raises(CycleInvariantError, match="not bound to the cycle decision"):
        ConstitutionalCycle._validate_trace(replace(trace, settlement=bad_settlement))

    bad_learning = replace(trace.learning_update, settlement_id="some-other-settlement")
    with pytest.raises(CycleInvariantError, match="not bound to the cycle settlement"):
        ConstitutionalCycle._validate_trace(replace(trace, learning_update=bad_learning))

    bad_decision = replace(trace.decision, objective_id="some-other-objective")
    with pytest.raises(CycleInvariantError, match="not bound to the cycle objective"):
        ConstitutionalCycle._validate_trace(replace(trace, decision=bad_decision))


def test_validate_trace_rejects_evidence_ids_outside_the_cycle() -> None:
    """State-to-record continuity: a record referencing an evidence id that
    was never produced by this cycle must fail, not pass silently."""

    trace = run_constitutional_cycle(objective())
    bad_capability = replace(trace.capability, evidence_ids=("forged-evidence-id",))
    with pytest.raises(CycleInvariantError, match="evidence outside this cycle"):
        ConstitutionalCycle._validate_trace(replace(trace, capability=bad_capability))

    states = list(trace.states)
    states[-1] = replace(states[-1], evidence_ids=("forged-evidence-id",))
    with pytest.raises(CycleInvariantError, match="evidence outside this cycle"):
        ConstitutionalCycle._validate_trace(replace(trace, states=tuple(states)))


def test_cycle_does_not_import_or_mutate_legacy_runtime() -> None:
    legacy_files = [
        ROOT / "rogal_core/daemon.py",
        ROOT / "rogal_core/autonomous_cycle.py",
        ROOT / ".replit",
    ]
    legacy_files = [path for path in legacy_files if path.exists()]
    before = {path: path.read_bytes() for path in legacy_files}
    trace = run_constitutional_cycle(objective(), mode=CycleMode.SUCCESS)
    assert trace.to_dict()["stop_decision"]["outcome"] == "stop"
    assert all(path.read_bytes() == content for path, content in before.items())
    source = (ROOT / "kraken_r/cycle.py").read_text()
    assert "import rogal_core" not in source
    assert "from rogal_core" not in source