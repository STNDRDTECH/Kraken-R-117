"""Acceptance and adversarial tests for typed semantic model cognition."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from kraken_r.adaptive_substrate import AdaptiveState
from kraken_r.cognition_kernel import (
    CognitionValidationError,
    MaterialState,
    ProcessingCapability,
    ProcessingOperation,
    ProblemBranch,
    ProblemGraph,
    ProblemMaterial,
    ProblemRequirement,
    ProblemSubtask,
    SatisfactionState,
    SemanticJob,
    SourceLineage,
    replay_processing_trace,
    run_connected_processing,
)
from kraken_r.llm_adapter import (
    FixtureModelProvider,
    ModelAdapter,
    ModelProviderError,
)
from kraken_r.plastic_routing import CandidateRoute, RouteTopology
from kraken_r.semantic_capability import (
    MAX_SEMANTIC_CANDIDATES,
    CandidateCognition,
    SemanticCapabilityError,
    SemanticModelCapability,
    validate_semantic_operation_result,
)
from kraken_r.task_integrity import OriginalTask


def _problem() -> ProblemGraph:
    original = OriginalTask(
        "semantic-task",
        "Explain the observation while preserving the declared constraints.",
        "user",
        {"conversation": "semantic-fixture"},
    )
    return ProblemGraph(
        "semantic-problem",
        original,
        "Which mechanism best distinguishes alpha from beta?",
        (
            ProblemRequirement(
                "semantic-requirement",
                "Preserve the conserved quantity.",
                user_constraint=True,
            ),
        ),
        (
            ProblemSubtask(
                "semantic-alpha",
                "Interpret alpha.",
                ("semantic-requirement",),
                branch_id="semantic-branch-alpha",
            ),
            ProblemSubtask(
                "semantic-beta",
                "Interpret beta.",
                ("semantic-requirement",),
                branch_id="semantic-branch-beta",
            ),
        ),
        (
            ProblemMaterial(
                "semantic-material-alpha",
                "Alpha rises after the valve opens.",
                state=MaterialState.ACTIVE,
                source_lineage_ids=("semantic-lineage-alpha",),
                environment_id="semantic-environment-alpha",
            ),
            ProblemMaterial(
                "semantic-material-beta",
                "Beta remains stable after the valve opens.",
                state=MaterialState.ACTIVE,
                source_lineage_ids=("semantic-lineage-beta",),
                environment_id="semantic-environment-beta",
            ),
        ),
        (
            ProblemBranch(
                "semantic-branch-alpha",
                "Alpha explanation",
                "semantic-environment-alpha",
                ("semantic-alpha",),
            ),
            ProblemBranch(
                "semantic-branch-beta",
                "Beta explanation",
                "semantic-environment-beta",
                ("semantic-beta",),
            ),
        ),
        ("semantic-environment-alpha", "semantic-environment-beta"),
        {
            "semantic-requirement": SatisfactionState.UNRESOLVED,
            "semantic-alpha": SatisfactionState.UNRESOLVED,
            "semantic-beta": SatisfactionState.UNRESOLVED,
        },
        {"original_task_hash": original.task_hash, "candidate_only": True},
        (
            SourceLineage(
                "semantic-lineage-alpha",
                "semantic-source-alpha",
                "fixture",
                "semantic-group-alpha",
            ),
            SourceLineage(
                "semantic-lineage-beta",
                "semantic-source-beta",
                "fixture",
                "semantic-group-beta",
            ),
        ),
    )


def _capability(
    job: SemanticJob,
    *,
    capability_id: str = "semantic-capability-alpha",
    route_id: str = "semantic-route-alpha",
    branch_id: str = "semantic-branch-alpha",
    material_id: str = "semantic-material-alpha",
) -> ProcessingCapability:
    return ProcessingCapability(
        capability_id,
        ProcessingOperation.MODEL_PROPOSAL,
        route_id,
        branch_id,
        (material_id,),
        semantic_job=job,
    )


def _state(alpha: float = 0.6, beta: float = 0.25, version: int = 1) -> AdaptiveState:
    state = AdaptiveState.fixture(f"semantic-adaptive-{version}")
    topology = RouteTopology(
        state.route_topology.topology_id,
        version,
        (
            CandidateRoute(
                "semantic-route-alpha", "candidate", "candidate", "alpha", alpha
            ),
            CandidateRoute(
                "semantic-route-beta", "candidate", "candidate", "beta", beta
            ),
        ),
        generation=version,
    )
    return replace(state, generation=version, route_topology=topology)


def _provider_output(
    job: str,
    use_class: str,
    *,
    extra_root=None,
    candidate_updates=None,
    candidate_count: int = 1,
) -> str:
    candidate = {
        "statement": "Pressure-driven transport is a candidate mechanism.",
        "rationale": "It could explain the declared observation but needs testing.",
        "structured_payload": {
            "variables": ["pressure", "flow"],
            "equations": ["flow = conductance * pressure_difference"],
        },
        "self_reported_confidence": 0.99,
        "source_refs": ["semantic-lineage-alpha"],
    }
    candidate.update(candidate_updates or {})
    value = {
        "job": job,
        "use_class": use_class,
        "candidates": [deepcopy(candidate) for _ in range(candidate_count)],
        "questions": ["Does the response disappear when pressure is clamped?"],
        "missing_information": ["Independent pressure measurements are missing."],
    }
    if extra_root:
        value.update(extra_root)
    return json.dumps(value)


def _semantic(job: SemanticJob, response_factory=None):
    def default_response(request):
        prompt = json.loads(request.prompt)
        source_refs = prompt["focused_context"]["focused_items"][0].get(
            "source_lineage_ids", []
        )
        return _provider_output(
            prompt["semantic_job"],
            prompt["use_class"],
            candidate_updates={"source_refs": source_refs},
        )

    provider = FixtureModelProvider(response_factory or default_response)
    operation = SemanticModelCapability(
        ModelAdapter(provider), model_id="fixture-model"
    )
    return provider, operation


@pytest.mark.parametrize("job", tuple(SemanticJob))
def test_all_declared_semantic_jobs_return_typed_candidate_cognition(job):
    provider, operation = _semantic(job)
    trace = run_connected_processing(
        _problem(),
        (_capability(job),),
        _state(),
        operation_callbacks={ProcessingOperation.MODEL_PROPOSAL: operation},
    )

    semantic = validate_semantic_operation_result(trace.request, trace.result)
    assert semantic.job is job
    assert semantic.use_class == f"{job.value}_candidate"
    assert isinstance(semantic.candidates[0], CandidateCognition)
    assert semantic.candidates[0].self_reported_confidence == 0.99
    assert semantic.candidates[0].candidate_only is True
    assert semantic.candidates[0].evidence_authority is False
    assert trace.result.evidence_ids == ()
    assert trace.result.settlement_ids == ()
    assert trace.result.adaptive_update_ids == ()
    assert len(provider.calls) == 1


def test_topology_changes_selected_semantic_job_and_focused_model_input():
    capabilities = (
        _capability(SemanticJob.RECALL),
        _capability(
            SemanticJob.MECHANISM_GENERATION,
            capability_id="semantic-capability-beta",
            route_id="semantic-route-beta",
            branch_id="semantic-branch-beta",
            material_id="semantic-material-beta",
        ),
    )
    provider, operation = _semantic(SemanticJob.RECALL)
    callbacks = {ProcessingOperation.MODEL_PROPOSAL: operation}

    alpha = run_connected_processing(
        _problem(), capabilities, _state(), operation_callbacks=callbacks
    )
    beta = run_connected_processing(
        _problem(),
        capabilities,
        _state(0.25, 0.6, 2),
        operation_callbacks=callbacks,
    )
    alpha_prompt = json.loads(provider.calls[0].prompt)
    beta_prompt = json.loads(provider.calls[1].prompt)

    assert alpha.request.semantic_job is SemanticJob.RECALL
    assert beta.request.semantic_job is SemanticJob.MECHANISM_GENERATION
    assert alpha_prompt["semantic_job"] == "recall"
    assert beta_prompt["semantic_job"] == "mechanism_generation"
    assert alpha_prompt["focused_context"] != beta_prompt["focused_context"]
    assert alpha_prompt["focused_context"]["focus_ids"] == [
        "semantic-material-alpha"
    ]
    assert beta_prompt["focused_context"]["focus_ids"] == [
        "semantic-material-beta"
    ]
    assert "semantic-material-beta" not in provider.calls[0].prompt
    assert "semantic-material-alpha" not in provider.calls[1].prompt


def test_structural_replay_validates_semantics_without_calling_provider():
    provider, operation = _semantic(SemanticJob.FALSIFIER_GENERATION)
    trace = run_connected_processing(
        _problem(),
        (_capability(SemanticJob.FALSIFIER_GENERATION),),
        _state(),
        operation_callbacks={ProcessingOperation.MODEL_PROPOSAL: operation},
    )
    calls_before = len(provider.calls)

    replayed = replay_processing_trace(trace.to_dict())

    assert replayed.to_dict() == trace.to_dict()
    assert len(provider.calls) == calls_before


@pytest.mark.parametrize(
    "response_factory",
    (
        lambda _: "{not-json",
        lambda _: _provider_output(
            "mechanism_generation", "mechanism_generation_candidate"
        ),
        lambda _: _provider_output("recall", "wrong_candidate"),
        lambda _: _provider_output(
            "recall",
            "recall_candidate",
            extra_root={"evidence": ["invented"]},
        ),
        lambda _: _provider_output(
            "recall",
            "recall_candidate",
            candidate_updates={"source_refs": ["undeclared-lineage"]},
        ),
        lambda _: _provider_output(
            "recall",
            "recall_candidate",
            candidate_count=MAX_SEMANTIC_CANDIDATES + 1,
        ),
    ),
)
def test_malformed_unauthorized_and_unbound_outputs_fail_closed(response_factory):
    _, operation = _semantic(SemanticJob.RECALL, response_factory)

    with pytest.raises(SemanticCapabilityError):
        run_connected_processing(
            _problem(),
            (_capability(SemanticJob.RECALL),),
            _state(),
            operation_callbacks={ProcessingOperation.MODEL_PROPOSAL: operation},
        )


def test_oversized_output_and_provider_error_fail_closed():
    for response_factory in (
        lambda _: "x" * 40_000,
        lambda _: (_ for _ in ()).throw(ModelProviderError("offline")),
    ):
        _, operation = _semantic(SemanticJob.RECALL, response_factory)
        with pytest.raises(SemanticCapabilityError):
            run_connected_processing(
                _problem(),
                (_capability(SemanticJob.RECALL),),
                _state(),
                operation_callbacks={ProcessingOperation.MODEL_PROPOSAL: operation},
            )


def test_tampered_semantic_provenance_fails_structural_replay():
    _, operation = _semantic(SemanticJob.RECALL)
    trace = run_connected_processing(
        _problem(),
        (_capability(SemanticJob.RECALL),),
        _state(),
        operation_callbacks={ProcessingOperation.MODEL_PROPOSAL: operation},
    )
    record = trace.to_dict()
    record["result"]["output"]["semantic_result"]["candidates"][0]["provenance"][
        "branch_id"
    ] = "semantic-branch-beta"

    with pytest.raises(CognitionValidationError):
        replay_processing_trace(record)


def test_semantic_capability_uses_existing_adapter_boundary_only():
    source = __import__("inspect").getsource(
        __import__(
            "kraken_r.semantic_capability", fromlist=["SemanticModelCapability"]
        )
    )
    assert "OpenAICompatibleProvider" not in source
    assert "urlopen" not in source
    assert "threading" not in source
    assert ".complete_raw(provider_request)" in source