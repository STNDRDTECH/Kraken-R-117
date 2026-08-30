"""Behavioral qualification for the candidate-only cognition kernel."""

from dataclasses import replace
from copy import deepcopy

import pytest

from kraken_r.adaptive_substrate import AdaptiveState
from kraken_r.cognition_kernel import (
    CognitionValidationError,
    EdgeType,
    EpistemicClaim,
    EpistemicStatus,
    MaterialState,
    MAX_MATERIALS,
    ObligationKind,
    OperationResult,
    ProcessingCapability,
    ProcessingBudget,
    ProcessingEvent,
    ProcessingEventKind,
    ProcessingOperation,
    ProcessingTrace,
    ProblemBranch,
    ProblemGraph,
    ProblemMaterial,
    ProblemRequirement,
    ProblemSubtask,
    SatisfactionState,
    SourceLineage,
    TypedReasoningEdge,
    assess_synthesis,
    replay_processing_trace,
    run_connected_processing,
)
from kraken_r.contracts import Objective
from kraken_r.cycle import run_constitutional_cycle
from kraken_r.plastic_routing import CandidateRoute, RouteTopology
from kraken_r.task_integrity import OriginalTask


def _problem() -> ProblemGraph:
    original = OriginalTask(
        "task-cognition",
        "Determine whether alpha or beta explains the observation without losing constraints.",
        "user",
        {"conversation": "fixture"},
    )
    return ProblemGraph(
        "problem-cognition",
        original,
        "Which candidate branch should be investigated first?",
        (
            ProblemRequirement(
                "requirement-answer",
                "Answer the central question.",
                status=SatisfactionState.UNRESOLVED,
            ),
            ProblemRequirement(
                "requirement-units",
                "Preserve units and accounting.",
                user_constraint=True,
                status=SatisfactionState.UNRESOLVED,
            ),
        ),
        (
            ProblemSubtask(
                "subtask-alpha",
                "Check alpha.",
                ("requirement-answer",),
                branch_id="branch-alpha",
            ),
            ProblemSubtask(
                "subtask-beta",
                "Check beta.",
                ("requirement-answer",),
                branch_id="branch-beta",
            ),
        ),
        (
            ProblemMaterial(
                "material-active",
                "Shared observation.",
                state=MaterialState.ACTIVE,
            ),
            ProblemMaterial(
                "material-deferred",
                "Deferred caveat.",
                state=MaterialState.DEFERRED,
                environment_id="environment-alpha",
            ),
            ProblemMaterial(
                "material-dormant",
                "Dormant alternative.",
                state=MaterialState.DORMANT,
                environment_id="environment-beta",
            ),
            ProblemMaterial(
                "material-reactivated",
                "Reactivated requirement.",
                state=MaterialState.REACTIVATED,
            ),
            ProblemMaterial(
                "material-branch-alpha",
                "Declared alpha output slot.",
                state=MaterialState.DORMANT,
                environment_id="environment-alpha",
            ),
            ProblemMaterial(
                "material-branch-beta",
                "Declared beta output slot.",
                state=MaterialState.DORMANT,
                environment_id="environment-beta",
            ),
            ProblemMaterial(
                "material-stable",
                "Declared stable alpha output slot.",
                state=MaterialState.DORMANT,
                environment_id="environment-alpha",
            ),
        ),
        (
            ProblemBranch(
                "branch-alpha",
                "Alpha frame",
                "environment-alpha",
                ("subtask-alpha",),
            ),
            ProblemBranch(
                "branch-beta",
                "Beta frame",
                "environment-beta",
                ("subtask-beta",),
            ),
        ),
        ("environment-alpha", "environment-beta", "environment-shared"),
        {
            "requirement-answer": SatisfactionState.UNRESOLVED,
            "requirement-units": SatisfactionState.UNRESOLVED,
            "subtask-alpha": SatisfactionState.UNRESOLVED,
            "subtask-beta": SatisfactionState.UNRESOLVED,
        },
        {"original_task_hash": original.task_hash, "candidate_only": True},
        (
            SourceLineage(
                "lineage-shared",
                "source-shared",
                "fixture",
                "group-shared",
            ),
            SourceLineage(
                "lineage-alpha",
                "source-alpha",
                "fixture",
                "group-alpha",
            ),
        ),
    )


def _capabilities() -> tuple[ProcessingCapability, ...]:
    return (
        ProcessingCapability(
            "capability-alpha",
            ProcessingOperation.CHECK_LOGIC,
            "path-alpha",
            "branch-alpha",
            ("material-active", "subtask-alpha"),
            ("material-branch-alpha",),
            ("subtask-alpha",),
            cost=2,
        ),
        ProcessingCapability(
            "capability-beta",
            ProcessingOperation.CHECK_QUANTITATIVE,
            "path-beta",
            "branch-beta",
            ("material-active", "subtask-beta"),
            ("material-branch-beta",),
            ("subtask-beta",),
            cost=2,
        ),
    )


def _adaptive(alpha: float, beta: float, version: int) -> AdaptiveState:
    state = AdaptiveState.fixture(f"adaptive-{version}")
    topology = RouteTopology(
        state.route_topology.topology_id,
        version,
        (
            CandidateRoute("path-alpha", "candidate-work", "candidate", "alpha", alpha),
            CandidateRoute("path-beta", "candidate-work", "candidate", "beta", beta),
        ),
        generation=version,
    )
    return replace(state, generation=version, route_topology=topology)


def _result_for(request):
    return OperationResult(
        f"{request.request_id}-result",
        request.request_id,
        output={"operation": request.operation.value, "branch": request.branch_id},
        downstream_material_ids=(f"material-{request.branch_id}",),
        downstream_satisfaction={
            "subtask-alpha" if request.branch_id == "branch-alpha" else "subtask-beta":
            SatisfactionState.SATISFIED
        },
        questions=("What independent observation can falsify this branch?",),
        investigative_needs=("Obtain an independent grounded check.",),
    )


def test_topology_changes_effective_processing_and_structural_replay_is_identical():
    problem = _problem()
    alpha_state = _adaptive(0.65, 0.25, 1)
    beta_state = _adaptive(0.25, 0.65, 2)
    callbacks = {
        ProcessingOperation.CHECK_LOGIC: _result_for,
        ProcessingOperation.CHECK_QUANTITATIVE: _result_for,
    }

    alpha = run_connected_processing(
        problem, _capabilities(), alpha_state, operation_callbacks=callbacks
    )
    beta = run_connected_processing(
        problem, _capabilities(), beta_state, operation_callbacks=callbacks
    )

    assert alpha.projection.available_capability_ids == ("capability-alpha",)
    assert beta.projection.available_capability_ids == ("capability-beta",)
    assert alpha.decision.operation is ProcessingOperation.CHECK_LOGIC
    assert beta.decision.operation is ProcessingOperation.CHECK_QUANTITATIVE
    assert alpha.decision.branch_id == "branch-alpha"
    assert beta.decision.branch_id == "branch-beta"
    assert alpha.context.output_hash != beta.context.output_hash
    assert alpha.events[0].causal_links == (alpha.projection.projection_id,)
    assert replay_processing_trace(alpha).to_dict() == alpha.to_dict()
    assert replay_processing_trace(beta).to_dict() == beta.to_dict()
    assert replay_processing_trace(alpha.to_dict()).to_dict() == alpha.to_dict()


def test_unverified_claim_cannot_create_evidence_or_adaptive_credit():
    claim = EpistemicClaim(
        "claim-unverified",
        _problem().problem_id,
        "A model proposes alpha.",
        status=EpistemicStatus.PROPOSAL,
    )
    with pytest.raises(CognitionValidationError, match="cannot create evidence"):
        claim.to_evidence()
    with pytest.raises(CognitionValidationError, match="cannot create adaptive credit"):
        claim.to_adaptive_credit()
    with pytest.raises(CognitionValidationError, match="cannot create evidence_ids"):
        OperationResult(
            "result-illegal",
            "request-illegal",
            evidence_ids=("fake-evidence",),
        )


def test_analogy_cannot_masquerade_as_causal_or_quantitative_relationship():
    analogy = TypedReasoningEdge.create(
        "edge-analogy",
        _problem().problem_id,
        "claim-a",
        "claim-b",
        EdgeType.ANALOGY,
    )
    assert analogy.required_obligation_kinds == (
        ObligationKind.LOGIC,
        ObligationKind.ANALOGY_SCOPE,
    )
    causal = TypedReasoningEdge.create(
        "edge-causal",
        _problem().problem_id,
        "claim-a",
        "claim-b",
        EdgeType.CAUSATION,
    )
    with pytest.raises(CognitionValidationError, match="must create exactly"):
        replace(analogy, obligations=causal.obligations)


@pytest.mark.parametrize(
    "edge_type,required",
    (
        (EdgeType.DEDUCTION, ObligationKind.LOGIC),
        (EdgeType.QUANTITATIVE, ObligationKind.UNITS_DIMENSIONS),
        (EdgeType.QUANTITATIVE, ObligationKind.CONSERVATION_ACCOUNTING),
        (EdgeType.CAUSATION, ObligationKind.REGIME_APPLICABILITY),
    ),
)
def test_missing_hard_obligations_block_candidate_synthesis(edge_type, required):
    problem = _problem()
    first = EpistemicClaim("claim-first", problem.problem_id, "First candidate claim.")
    second = EpistemicClaim("claim-second", problem.problem_id, "Second candidate claim.")
    edge = TypedReasoningEdge.create(
        f"edge-{edge_type.value}",
        problem.problem_id,
        first.claim_id,
        second.claim_id,
        edge_type,
    )
    assert required in edge.required_obligation_kinds
    assessment = assess_synthesis(problem, (first, second), (edge,), ())
    assert assessment.allowed is False
    assert assessment.blocking_obligation_ids


def test_materially_identical_processing_under_changed_topology_is_causal_noop():
    problem = _problem()
    capabilities = (
        ProcessingCapability(
            "capability-alpha",
            ProcessingOperation.CHECK_LOGIC,
            "path-alpha",
            "branch-alpha",
            ("material-active",),
            ("material-stable",),
            cost=2,
        ),
    )

    def stable_result(request):
        return OperationResult(
            f"{request.request_id}-result",
            request.request_id,
            output={"answer": "same"},
            downstream_material_ids=("material-stable",),
            questions=("Same question.",),
            investigative_needs=("Same need.",),
        )

    first = run_connected_processing(
        problem,
        capabilities,
        _adaptive(0.50, 0.25, 3),
        operation_callbacks={ProcessingOperation.CHECK_LOGIC: stable_result},
    )
    second = run_connected_processing(
        problem,
        capabilities,
        _adaptive(0.65, 0.25, 4),
        operation_callbacks={ProcessingOperation.CHECK_LOGIC: stable_result},
        prior_trace=first,
    )

    assert first.projection.topology_read_hash != second.projection.topology_read_hash
    assert first.effective_computation_hash == second.effective_computation_hash
    assert second.causal_noop is True
    assert second.events[-1].kind is ProcessingEventKind.NOOP


def test_bounds_provenance_duplicates_branch_separation_and_unresolved_preservation():
    problem = _problem()
    with pytest.raises(CognitionValidationError, match="materials exceeds"):
        replace(
            problem,
            materials=tuple(
                ProblemMaterial(f"material-{index}", "bounded")
                for index in range(MAX_MATERIALS + 1)
            ),
        )

    altered_original = OriginalTask(
        problem.original_task.task_id,
        problem.original_task.text + " Changed.",
        problem.original_task.source,
        problem.original_task.provenance,
    )
    assert replace(problem, original_task=altered_original).graph_hash != problem.graph_hash

    trace = run_connected_processing(problem, _capabilities(), _adaptive(0.65, 0.25, 5))
    assert trace.context.branch_id == "branch-alpha"
    assert trace.context.satisfaction["subtask-beta"] is SatisfactionState.UNRESOLVED
    assert "material-dormant" not in trace.context.active_material_ids
    assert "material-reactivated" in trace.context.active_material_ids

    duplicate = ProcessingEvent(
        "duplicate-event",
        ProcessingEventKind.ACTIVATION,
        "capability-alpha",
        "input-hash",
        "output-hash",
    )
    with pytest.raises(CognitionValidationError, match="event identities"):
        replace(trace, events=(duplicate, duplicate))


def test_candidate_processing_does_not_interfere_with_frozen_cycle():
    objective = Objective(
        "frozen-cycle-objective",
        "Prove the candidate cognition projection leaves the qualified cycle unchanged.",
    )
    before = run_constitutional_cycle(objective).to_dict()
    run_connected_processing(_problem(), _capabilities(), _adaptive(0.65, 0.25, 6))
    after = run_constitutional_cycle(objective).to_dict()
    assert before == after


def test_replay_rejects_tampered_hashes_and_causal_links():
    trace = run_connected_processing(
        _problem(), _capabilities(), _adaptive(0.65, 0.25, 7)
    )
    tampered = deepcopy(trace.to_dict())
    tampered["events"][1]["output_hash"] = "forged-output-hash"
    with pytest.raises(CognitionValidationError, match="event chain|structural replay"):
        replay_processing_trace(tampered)

    tampered = deepcopy(trace.to_dict())
    tampered["problem"]["central_question"] = "Forged question."
    with pytest.raises(CognitionValidationError, match="problem graph replay hash"):
        replay_processing_trace(tampered)


def test_branch_scope_and_event_budget_fail_closed():
    problem = _problem()
    cross_branch = (
        ProcessingCapability(
            "capability-cross-branch",
            ProcessingOperation.CHECK_LOGIC,
            "path-alpha",
            "branch-alpha",
            ("subtask-beta",),
        ),
    )
    with pytest.raises(CognitionValidationError, match="subtask branch scope"):
        run_connected_processing(problem, cross_branch, _adaptive(0.65, 0.25, 8))
    cross_requirement = (
        ProcessingCapability(
            "capability-cross-requirement",
            ProcessingOperation.CHECK_LOGIC,
            "path-alpha",
            "branch-alpha",
            ("subtask-alpha",),
            declared_satisfaction_ids=("requirement-deferred",),
        ),
    )
    with pytest.raises(CognitionValidationError, match="satisfaction branch scope"):
        run_connected_processing(
            problem, cross_requirement, _adaptive(0.65, 0.25, 8)
        )
    with pytest.raises(CognitionValidationError, match="event budget"):
        run_connected_processing(
            problem,
            _capabilities(),
            _adaptive(0.65, 0.25, 8),
            budget=ProcessingBudget(max_events=2),
        )


def test_noop_requires_changed_topology_and_compatible_problem():
    problem = _problem()
    first = run_connected_processing(
        problem, _capabilities(), _adaptive(0.65, 0.25, 1)
    )
    same = run_connected_processing(
        problem,
        _capabilities(),
        _adaptive(0.65, 0.25, 1),
        prior_trace=first,
    )
    assert same.causal_noop is False
    assert all(item.kind is not ProcessingEventKind.NOOP for item in same.events)

    other_problem = replace(problem, central_question="A different bounded question.")
    with pytest.raises(CognitionValidationError, match="same problem"):
        run_connected_processing(
            other_problem,
            _capabilities(),
            _adaptive(0.65, 0.25, 2),
            prior_trace=first,
        )


def test_blocked_trace_and_proof_graph_replay_are_fail_closed():
    problem = _problem()
    inhibited = (
        ProcessingCapability(
            "capability-inhibited",
            ProcessingOperation.CHECK_LOGIC,
            "path-alpha",
            "branch-alpha",
            ("subtask-alpha",),
            min_route_weight=0.70,
        ),
    )
    blocked = run_connected_processing(
        problem, inhibited, _adaptive(0.65, 0.25, 1)
    )
    assert blocked.request is None
    assert replay_processing_trace(blocked.to_dict()).to_dict() == blocked.to_dict()
    tampered = deepcopy(blocked.to_dict())
    tampered["events"][0]["causal_links"] = ["forged-projection"]
    with pytest.raises(CognitionValidationError, match="inhibition event"):
        replay_processing_trace(tampered)

    first = EpistemicClaim("claim-proof-first", problem.problem_id, "First.")
    second = EpistemicClaim("claim-proof-second", problem.problem_id, "Second.")
    edge = TypedReasoningEdge.create(
        "edge-proof",
        problem.problem_id,
        first.claim_id,
        second.claim_id,
        EdgeType.DEDUCTION,
    )
    proof_trace = run_connected_processing(
        problem,
        _capabilities(),
        _adaptive(0.65, 0.25, 2),
        claims=(first, second),
        reasoning_edges=(edge,),
    )
    assert proof_trace.obligations == edge.obligations
    assert replay_processing_trace(proof_trace.to_dict()).to_dict() == proof_trace.to_dict()

    invented_endpoint = TypedReasoningEdge.create(
        "edge-invented",
        problem.problem_id,
        first.claim_id,
        "claim-not-declared",
        EdgeType.DEDUCTION,
    )
    with pytest.raises(CognitionValidationError, match="endpoint is undeclared"):
        assess_synthesis(problem, (first,), (invented_endpoint,), ())
    with pytest.raises(CognitionValidationError, match="endpoint is undeclared"):
        run_connected_processing(
            problem,
            _capabilities(),
            _adaptive(0.65, 0.25, 3),
            claims=(first,),
            reasoning_edges=(invented_endpoint,),
        )


def test_callback_payload_depth_and_size_are_bounded():
    problem = _problem()

    def deep_result(request):
        value = "leaf"
        for _ in range(10):
            value = {"nested": value}
        return OperationResult(
            f"{request.request_id}-result",
            request.request_id,
            output={"value": value},
        )

    with pytest.raises(CognitionValidationError, match="depth limit"):
        run_connected_processing(
            problem,
            _capabilities(),
            _adaptive(0.65, 0.25, 4),
            operation_callbacks={ProcessingOperation.CHECK_LOGIC: deep_result},
        )

    def large_result(request):
        return OperationResult(
            f"{request.request_id}-result",
            request.request_id,
            output={"value": "x" * 40_000},
        )

    with pytest.raises(CognitionValidationError, match="oversized text|byte limit"):
        run_connected_processing(
            problem,
            _capabilities(),
            _adaptive(0.65, 0.25, 5),
            operation_callbacks={ProcessingOperation.CHECK_LOGIC: large_result},
        )


def test_material_lineage_and_user_constraints_cannot_be_omitted():
    problem = _problem()
    forged_material = replace(
        problem.materials[0],
        source_lineage_ids=("lineage-not-declared",),
    )
    with pytest.raises(CognitionValidationError, match="lineage is undeclared"):
        replace(
            problem,
            materials=(forged_material,) + problem.materials[1:],
        )

    assessment = assess_synthesis(problem, (), (), ())
    assert assessment.allowed is False
    assert (
        "requirement-units-user-constraint-obligation"
        in assessment.blocking_obligation_ids
    )