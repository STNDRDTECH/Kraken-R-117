"""Acceptance and adversarial tests for typed semantic model cognition."""

from copy import deepcopy
from dataclasses import replace
import hashlib
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
    _hash,
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
        semantic_configuration={
            "model_id": "fixture-model",
            "max_tokens": 768,
            "temperature": 0.0,
            "prompt_template_version": "semantic-capability-v2",
        },
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
        "self_reported_confidence": 0.99,
        "source_refs": ["semantic-lineage-alpha"],
    }
    candidate.update(candidate_updates or {})
    source_ref = (
        candidate["source_refs"][0]
        if candidate["source_refs"]
        else "semantic-lineage-alpha"
    )
    payloads = {
        "recall": {
            "recollections": [
                {
                    "content": "A pressure gradient can drive flow.",
                    "relevance": "The valve changes the pressure path.",
                }
            ],
            "verification_obligations": [
                "Verify the recollection against declared material."
            ],
        },
        "mechanism_generation": {
            "components": [
                {
                    "name": "valve",
                    "role": "changes conductance",
                    "inputs": ["pressure difference"],
                    "outputs": ["flow"],
                }
            ],
            "causal_steps": ["Opening the valve increases conductance."],
            "assumptions": ["Pressure difference remains nonzero."],
            "verification_obligations": ["Measure each causal dependency."],
        },
        "competing_explanations": {
            "alternatives": [
                {
                    "label": "pressure-driven",
                    "explanation": "Pressure drives the observed flow.",
                    "discriminators": ["Clamp pressure while opening the valve."],
                },
                {
                    "label": "sensor-artifact",
                    "explanation": "The reading changes without real flow.",
                    "discriminators": ["Use an independent flow sensor."],
                },
            ],
            "discriminators": ["Compare independent flow and pressure readings."],
            "verification_obligations": [
                "Test alternatives under the same conditions."
            ],
        },
        "cross_domain_correspondence": {
            "correspondences": [
                {
                    "source_concept": "electrical conductance",
                    "target_concept": "fluid conductance",
                    "mapping": "Both relate flow to a driving potential.",
                }
            ],
            "scope_limits": ["Compressibility has no direct electrical analogue."],
            "verification_obligations": ["Verify the analogy within its scope."],
        },
        "variable_equation_extraction": {
            "variables": [
                {
                    "name": "flow",
                    "description": "Volumetric transport rate.",
                    "unit": "m3/s",
                    "role": "output",
                },
            ],
            "equations": [
                {
                    "expression": "flow = conductance * pressure_difference",
                    "variable_names": ["flow"],
                    "assumptions": ["Linear response regime."],
                }
            ],
            "verification_obligations": [
                "Check dimensions and the linear-regime assumption."
            ],
        },
        "source_interpretation": {
            "interpretations": [
                {
                    "source_ref": source_ref,
                    "passage": "Alpha rises after the valve opens.",
                    "interpretation": "The observation is consistent with increased flow.",
                }
            ],
            "limitations": ["The material does not state measurement uncertainty."],
            "verification_obligations": [
                "Compare the interpretation with the cited source."
            ],
        },
        "falsifier_generation": {
            "falsifiers": [
                {
                    "test": "Clamp pressure while opening the valve.",
                    "if_supported": "Flow does not rise without a pressure gradient.",
                    "if_disconfirmed": "Flow rises despite zero pressure difference.",
                    "discriminates": "Pressure-driven mechanism.",
                }
            ],
            "verification_obligations": ["Run an independent pressure-clamp test."],
        },
        "missing_information_detection": {
            "missing_items": [
                {
                    "item": "Independent pressure measurement.",
                    "why_needed": "The proposed mechanism requires a pressure gradient.",
                    "blocks": "Mechanism discrimination.",
                }
            ],
            "blocking_questions": ["What was the pressure across the valve?"],
            "verification_obligations": [
                "Obtain a calibrated independent pressure measurement."
            ],
        },
    }
    candidate.setdefault("structured_payload", payloads[job])
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


def _rehash_trace_record(record):
    result = record["result"]
    result_without_hash = {
        key: value for key, value in result.items() if key != "output_hash"
    }
    result["output_hash"] = _hash(result_without_hash)
    context = record["context"]
    context["output_hash"] = _hash(
        {
            "problem": record["problem"]["graph_hash"],
            "branch": context["branch_id"],
            "materials": context["active_material_ids"],
            "satisfaction": context["satisfaction"],
            "result": result["output_hash"],
        }
    )
    for event in record["events"]:
        if event["kind"] == "operation":
            event["output_hash"] = result["output_hash"]
        elif event["kind"] == "motif_observation":
            event["input_hash"] = result["output_hash"]
            event["output_hash"] = context["output_hash"]
    record["structural_hash"] = _hash(
        {key: value for key, value in record.items() if key != "structural_hash"}
    )
    return record


def _semantic_digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


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
    assert semantic.candidates[0].structured_payload["verification_obligations"]
    assert (
        set(semantic.candidates[0].structured_payload)
        != {"verification_obligations"}
    )
    assert trace.result.evidence_ids == ()
    assert trace.result.settlement_ids == ()
    assert trace.result.adaptive_update_ids == ()
    assert len(provider.calls) == 1


@pytest.mark.parametrize(
    ("job", "wrong_job"),
    tuple(
        (job, tuple(SemanticJob)[(index + 1) % len(SemanticJob)])
        for index, job in enumerate(SemanticJob)
    ),
)
def test_each_job_rejects_another_jobs_typed_payload(job, wrong_job):
    wrong_output = json.loads(
        _provider_output(wrong_job.value, f"{wrong_job.value}_candidate")
    )
    wrong_payload = wrong_output["candidates"][0]["structured_payload"]

    def response(request):
        prompt = json.loads(request.prompt)
        return _provider_output(
            prompt["semantic_job"],
            prompt["use_class"],
            candidate_updates={"structured_payload": wrong_payload},
        )


@pytest.mark.parametrize(
    ("job", "empty_field"),
    (
        (SemanticJob.RECALL, "verification_obligations"),
        (SemanticJob.MECHANISM_GENERATION, "causal_steps"),
        (SemanticJob.COMPETING_EXPLANATIONS, "discriminators"),
        (SemanticJob.CROSS_DOMAIN_CORRESPONDENCE, "scope_limits"),
        (SemanticJob.VARIABLE_EQUATION_EXTRACTION, "verification_obligations"),
        (SemanticJob.SOURCE_INTERPRETATION, "limitations"),
        (SemanticJob.FALSIFIER_GENERATION, "verification_obligations"),
        (SemanticJob.MISSING_INFORMATION_DETECTION, "blocking_questions"),
    ),
)
def test_each_job_rejects_missing_required_reasoning_content(job, empty_field):
    def response(request):
        prompt = json.loads(request.prompt)
        output = json.loads(
            _provider_output(prompt["semantic_job"], prompt["use_class"])
        )
        output["candidates"][0]["structured_payload"][empty_field] = []
        return json.dumps(output)

    _, operation = _semantic(job, response)
    with pytest.raises(SemanticCapabilityError):
        run_connected_processing(
            _problem(),
            (_capability(job),),
            _state(),
            operation_callbacks={ProcessingOperation.MODEL_PROPOSAL: operation},
        )

    _, operation = _semantic(job, response)
    with pytest.raises(SemanticCapabilityError):
        run_connected_processing(
            _problem(),
            (_capability(job),),
            _state(),
            operation_callbacks={ProcessingOperation.MODEL_PROPOSAL: operation},
        )


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


def test_candidate_identity_is_deterministic_and_bound_to_provider_output():
    _, operation = _semantic(SemanticJob.RECALL)
    first = run_connected_processing(
        _problem(),
        (_capability(SemanticJob.RECALL),),
        _state(),
        operation_callbacks={ProcessingOperation.MODEL_PROPOSAL: operation},
    )
    second = run_connected_processing(
        _problem(),
        (_capability(SemanticJob.RECALL),),
        _state(),
        operation_callbacks={ProcessingOperation.MODEL_PROPOSAL: operation},
    )
    first_semantic = validate_semantic_operation_result(
        first.request, first.result
    )
    second_semantic = validate_semantic_operation_result(
        second.request, second.result
    )

    expected = f"semantic-{first_semantic.provider_output_hash[:16]}-1"
    assert first_semantic.candidates[0].candidate_id == expected
    assert second_semantic.candidates[0].candidate_id == expected


@pytest.mark.parametrize(
    "tamper",
    (
        lambda candidate: candidate.update({"unknown_field": "forged"}),
        lambda candidate: candidate["provenance"].update(
            {"source_lineage_ids": ["undeclared-lineage"]}
        ),
        lambda candidate: candidate.update({"candidate_id": "semantic-forged-1"}),
    ),
)
def test_structurally_rehashed_semantic_records_still_fail_live_contract(tamper):
    _, operation = _semantic(SemanticJob.RECALL)
    trace = run_connected_processing(
        _problem(),
        (_capability(SemanticJob.RECALL),),
        _state(),
        operation_callbacks={ProcessingOperation.MODEL_PROPOSAL: operation},
    )
    record = trace.to_dict()
    candidate = record["result"]["output"]["semantic_result"]["candidates"][0]
    tamper(candidate)
    _rehash_trace_record(record)

    with pytest.raises(CognitionValidationError):
        replay_processing_trace(record)


def test_rehashed_invocation_prompt_and_configuration_tampering_fail_closed():
    _, operation = _semantic(SemanticJob.RECALL)
    trace = run_connected_processing(
        _problem(),
        (_capability(SemanticJob.RECALL),),
        _state(),
        operation_callbacks={ProcessingOperation.MODEL_PROPOSAL: operation},
    )
    for field, value in (
        ("max_tokens", 769),
        ("prompt_hash", "0" * 64),
        ("prompt_template_version", "semantic-capability-forged"),
    ):
        record = trace.to_dict()
        invocation = record["result"]["output"]["semantic_result"]["invocation"]
        invocation[field] = value
        _rehash_trace_record(record)
        with pytest.raises(CognitionValidationError):
            replay_processing_trace(record)


def test_self_consistently_rehashed_generation_configuration_is_rejected():
    _, operation = _semantic(SemanticJob.RECALL)
    trace = run_connected_processing(
        _problem(),
        (_capability(SemanticJob.RECALL),),
        _state(),
        operation_callbacks={ProcessingOperation.MODEL_PROPOSAL: operation},
    )
    record = trace.to_dict()
    invocation = record["result"]["output"]["semantic_result"]["invocation"]
    invocation["max_tokens"] = 769
    invocation["configuration_hash"] = _semantic_digest(
        {
            "model_id": invocation["model_id"],
            "max_tokens": invocation["max_tokens"],
            "temperature": invocation["temperature"],
            "prompt_template_version": invocation["prompt_template_version"],
        }
    )
    invocation["request_hash"] = _semantic_digest(
        {
            "provider_request_id": invocation["provider_request_id"],
            "operation_request_id": invocation["operation_request_id"],
            "prompt_hash": invocation["prompt_hash"],
            "configuration_hash": invocation["configuration_hash"],
        }
    )
    _rehash_trace_record(record)

    with pytest.raises(CognitionValidationError):
        replay_processing_trace(record)


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
        lambda _: (_ for _ in ()).throw(TimeoutError("deadline")),
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