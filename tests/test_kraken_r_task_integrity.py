"""Stage 10.7 adversarial acceptance tests for task-integrity boundaries."""

from __future__ import annotations

from dataclasses import replace

import pytest

from kraken_r import (
    BeliefState,
    BeliefStatus,
    CandidateClaim,
    CandidateResult,
    LossKind,
    MAX_HYPOTHESES,
    MAX_LOSS_FINDINGS,
    OriginalTask,
    RequirementKind,
    ReviewProjection,
    ReviewRole,
    TaskHypothesis,
    TaskIntegrityValidationError,
    TaskPlan,
    TaskRequirement,
    TaskSpecification,
    account_information_loss,
    project_review,
    replay_task_integrity,
)


def _fixture(*, ambiguous: bool = False):
    original = OriginalTask(
        "task-integrity-original",
        "Preserve every request, avoid runtime wiring, and verify independently.",
        "user_request",
        {"channel": "candidate"},
    )
    requirements = (
        TaskRequirement("req-ask", RequirementKind.ASK, "Preserve every request."),
        TaskRequirement(
            "req-constraint",
            RequirementKind.CONSTRAINT,
            "Avoid runtime wiring.",
        ),
        TaskRequirement(
            "req-evidence",
            RequirementKind.REQUIRED_EVIDENCE,
            "Verify independently.",
        ),
    ) + (
        (
            TaskRequirement(
                "req-ambiguity",
                RequirementKind.AMBIGUITY,
                "The exact verification boundary is ambiguous.",
            ),
        )
        if ambiguous
        else ()
    )
    specification = TaskSpecification.from_task(
        "task-integrity-spec",
        original,
        requirements,
        "Interpret all clauses as candidate-only requirements.",
        provenance={"declared_by": "caller"},
    )
    requirement_ids = specification.requirement_ids
    hypothesis = TaskHypothesis(
        "hypothesis-main",
        specification.specification_id,
        "A pure immutable boundary can preserve each requirement.",
        requirement_ids,
        status=BeliefStatus.AMBIGUOUS if ambiguous else BeliefStatus.OPEN,
    )
    belief = BeliefState("task-integrity-belief", specification.specification_id, (hypothesis,))
    plan = TaskPlan(
        "task-integrity-plan",
        specification.specification_id,
        (hypothesis.hypothesis_id,),
        requirement_ids,
        ("Preserve clauses.", "Inspect candidate output."),
    )
    claim = CandidateClaim("claim-main", "Every requirement is addressed.", requirement_ids)
    result = CandidateResult(
        "task-integrity-result",
        specification.specification_id,
        plan.plan_id,
        (hypothesis.hypothesis_id,),
        requirement_ids,
        claims=(claim,),
        conclusion_claim_ids=(claim.claim_id,),
    )
    return original, specification, belief, plan, result


def test_correct_candidate_replays_deterministically_without_review_mutation() -> None:
    original, specification, belief, plan, result = _fixture()
    before = result.to_dict()

    first = replay_task_integrity(original, specification, belief, plan, result)
    second = replay_task_integrity(original, specification, belief, plan, result)

    assert first.to_dict() == second.to_dict()
    assert first.loss_report.findings == ()
    assert first.review_findings == ()
    assert first.verification_questions == ()
    assert result.to_dict() == before
    assert first.to_dict()["creates_evidence"] is False
    assert first.to_dict()["changes_adaptive_state"] is False


def test_loss_accounting_detects_omission_compression_and_underweighting() -> None:
    original, specification, belief, plan, result = _fixture()
    result = replace(
        result,
        addressed_requirement_ids=("req-ask",),
        compressed_requirement_ids=("req-constraint",),
        underweighted_requirement_ids=("req-evidence",),
    )

    report = account_information_loss(specification, belief, plan, result)
    findings = {(item.kind, item.requirement_id) for item in report.findings}
    trace = replay_task_integrity(
        original, specification, belief, plan, result, roles=(ReviewRole.ANGEL,)
    )

    assert (LossKind.OMITTED, "req-constraint") in findings
    assert (LossKind.COMPRESSED, "req-constraint") in findings
    assert (LossKind.UNDERWEIGHTED, "req-evidence") in findings
    assert {item.role for item in trace.review_findings} == {ReviewRole.ANGEL}
    assert all(item.to_dict()["proposal_only"] for item in trace.review_findings)


def test_antimetabole_reports_unsupported_reverse_dependency_without_credit() -> None:
    original, specification, belief, plan, result = _fixture()
    result = replace(
        result,
        claims=(
            CandidateClaim(
                "claim-main",
                "A conclusion relies on an absent support.",
                ("req-ask",),
                dependency_claim_ids=("claim-absent",),
            ),
        ),
    )
    trace = replay_task_integrity(
        original, specification, belief, plan, result, roles=(ReviewRole.ANTIMETABOLE,)
    )

    assert any(
        item.kind is LossKind.UNSUPPORTED_DEPENDENCY
        for item in trace.loss_report.findings
    )
    assert len(trace.review_findings) == len(trace.verification_questions) == 1
    assert trace.review_findings[0].role is ReviewRole.ANTIMETABOLE
    assert trace.verification_questions[0].to_dict()["evidence_grade"] == "none"


def test_role_projections_are_selective_and_do_not_leak_correlated_context() -> None:
    _, specification, belief, _, result = _fixture()
    angel = project_review(ReviewRole.ANGEL, specification, belief, result)
    nemesis = project_review(ReviewRole.NEMESIS, specification, belief, result)
    antimetabole = project_review(ReviewRole.ANTIMETABOLE, specification, belief, result)

    assert set(angel.payload) == {"requirements", "addressed_requirement_ids"}
    assert set(nemesis.payload) == {"hypotheses"}
    assert set(antimetabole.payload) == {"claims", "conclusion_claim_ids"}
    assert "original_text" not in angel.payload
    with pytest.raises(TaskIntegrityValidationError, match="forbidden context"):
        ReviewProjection(
            "invalid-projection",
            ReviewRole.ANGEL,
            specification.specification_id,
            result.result_id,
            {"original_text": "leaked"},
        )


def test_false_objections_do_not_appear_without_declared_contradiction() -> None:
    original, specification, belief, plan, result = _fixture()
    trace = replay_task_integrity(
        original, specification, belief, plan, result, roles=(ReviewRole.NEMESIS,)
    )
    assert trace.review_findings == ()

    challenged = BeliefState(
        belief.belief_state_id,
        belief.specification_id,
        (
            replace(
                belief.hypotheses[0],
                contradiction_ids=("declared-counterexample",),
                unresolved_dependency_ids=("external-check",),
            ),
        ),
    )
    trace = replay_task_integrity(
        original, specification, challenged, plan, result, roles=(ReviewRole.NEMESIS,)
    )
    assert len(trace.review_findings) == 1
    assert trace.review_findings[0].role is ReviewRole.NEMESIS


def test_ambiguous_task_is_preserved_as_declared_status_not_a_mutation() -> None:
    original, specification, belief, plan, result = _fixture(ambiguous=True)
    trace = replay_task_integrity(original, specification, belief, plan, result)

    assert belief.hypotheses[0].status is BeliefStatus.AMBIGUOUS
    assert "req-ambiguity" in trace.result.addressed_requirement_ids
    assert trace.result == result
    assert trace.belief_state == belief


def test_declared_early_success_and_later_failure_cannot_become_credit() -> None:
    original, specification, belief, plan, result = _fixture()
    declared_success = replace(
        result,
        claims=(
            CandidateClaim(
                "claim-early-success",
                "The candidate succeeded before independent verification.",
                specification.requirement_ids,
            ),
        ),
        conclusion_claim_ids=("claim-early-success",),
    )
    declared_later_failure = replace(
        declared_success,
        result_id="task-integrity-later-failure",
        claims=(
            CandidateClaim(
                "claim-later-failure",
                "A later declared outcome says the candidate failed.",
                specification.requirement_ids,
            ),
        ),
        conclusion_claim_ids=("claim-later-failure",),
    )

    early = replay_task_integrity(
        original, specification, belief, plan, declared_success
    )
    later = replay_task_integrity(
        original, specification, belief, plan, declared_later_failure
    )

    for trace in (early, later):
        assert trace.to_dict()["creates_evidence"] is False
        assert trace.to_dict()["changes_adaptive_state"] is False
        assert trace.result.to_dict()["evidence_grade"] == "none"


def test_original_task_and_nested_provenance_are_immutable_and_hash_bound() -> None:
    original, specification, belief, plan, result = _fixture()
    with pytest.raises(TypeError):
        original.provenance["channel"] = "mutated"  # type: ignore[index]
    with pytest.raises(TaskIntegrityValidationError, match="does not bind"):
        replay_task_integrity(
            replace(original, text="Changed task"),
            specification,
            belief,
            plan,
            result,
        )


def test_fixed_bounds_and_invalid_review_role_sets_fail_closed() -> None:
    original, specification, belief, plan, result = _fixture()
    hypothesis = belief.hypotheses[0]
    with pytest.raises(TaskIntegrityValidationError, match="fixed limit"):
        BeliefState(
            "overfull-belief",
            specification.specification_id,
            tuple(replace(hypothesis, hypothesis_id=f"hypothesis-{index}") for index in range(MAX_HYPOTHESES + 1)),
        )
    with pytest.raises(TaskIntegrityValidationError, match="cannot repeat"):
        replay_task_integrity(
            original,
            specification,
            belief,
            plan,
            result,
            roles=(ReviewRole.ANGEL, ReviewRole.ANGEL),
        )


def test_maximum_loss_surface_is_bounded_with_explicit_truncation() -> None:
    _, specification, belief, plan, result = _fixture()
    requirement_ids = specification.requirement_ids
    noisy = replace(
        result,
        addressed_requirement_ids=(),
        compressed_requirement_ids=requirement_ids,
        underweighted_requirement_ids=requirement_ids,
        claims=tuple(
            CandidateClaim(
                f"claim-{index}",
                "Declared claim with bounded missing dependencies.",
                ("req-ask",),
                dependency_claim_ids=tuple(
                    f"missing-{index}-{dependency}" for dependency in range(8)
                ),
            )
            for index in range(16)
        ),
    )
    report = account_information_loss(specification, belief, plan, noisy)

    assert len(report.findings) == MAX_LOSS_FINDINGS
    assert report.truncated_finding_count > 0