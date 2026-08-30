"""Typed, candidate-only LLM semantics inside Kraken's cognition loop.

This module owns no provider, network, scheduler, persistence, execution, or
settlement path.  It can only call ``ModelAdapter.complete_raw`` and translate
one bounded provider response into an ``OperationResult`` whose authority is
explicitly nil.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

from .cognition_kernel import (
    CognitionValidationError,
    OperationRequest,
    OperationResult,
    ProcessingOperation,
    SemanticJob,
)
from .llm_adapter import (
    ModelAdapter,
    ModelAdapterValidationError,
    ModelProviderError,
    ProviderRequest,
)


MAX_SEMANTIC_CANDIDATES = 8
MAX_SEMANTIC_QUESTIONS = 8
MAX_SEMANTIC_MISSING_ITEMS = 8
MAX_SEMANTIC_OUTPUT_CHARS = 32_768
MAX_SEMANTIC_TEXT_CHARS = 4_096
MAX_SEMANTIC_PAYLOAD_ITEMS = 32
MAX_SEMANTIC_PROMPT_CHARS = 12_000


class SemanticCapabilityError(CognitionValidationError):
    """Fail-closed semantic model boundary."""


_USE_CLASSES = {
    job: f"{job.value}_candidate"
    for job in SemanticJob
}
_ROOT_KEYS = frozenset(
    {"job", "use_class", "candidates", "questions", "missing_information"}
)
_CANDIDATE_KEYS = frozenset(
    {
        "statement",
        "rationale",
        "structured_payload",
        "self_reported_confidence",
        "source_refs",
    }
)
_FORBIDDEN_KEYS = frozenset(
    {
        "action",
        "authority",
        "adaptive_credit",
        "adaptive_update",
        "evidence",
        "evidence_ids",
        "execution",
        "execution_truth",
        "goal",
        "learning_update",
        "settlement",
        "settlement_ids",
        "success",
        "truth",
        "verified",
    }
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _identifier(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(character.isspace() for character in value)
    ):
        raise SemanticCapabilityError(
            f"{field_name} must be a non-empty identifier without whitespace"
        )
    return value


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticCapabilityError(f"{field_name} must be non-empty text")
    if len(value) > MAX_SEMANTIC_TEXT_CHARS:
        raise SemanticCapabilityError(f"{field_name} exceeds its text bound")
    return value


def _text_items(value: Any, field_name: str, limit: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > limit:
        raise SemanticCapabilityError(f"{field_name} must be a bounded list")
    return tuple(_text(item, field_name) for item in value)


def _validate_payload(
    value: Any,
    *,
    path: str = "structured_payload",
    depth: int = 0,
    counter: list[int] | None = None,
) -> Any:
    if counter is None:
        counter = [0]
    if depth > 4:
        raise SemanticCapabilityError(f"{path} exceeds nesting bound")
    counter[0] += 1
    if counter[0] > MAX_SEMANTIC_PAYLOAD_ITEMS:
        raise SemanticCapabilityError(f"{path} exceeds item bound")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > MAX_SEMANTIC_TEXT_CHARS:
            raise SemanticCapabilityError(f"{path} text exceeds bound")
        return value
    if isinstance(value, list):
        return tuple(
            _validate_payload(
                item, path=f"{path}[{index}]", depth=depth + 1, counter=counter
            )
            for index, item in enumerate(value)
        )
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 128:
                raise SemanticCapabilityError(f"{path} key is invalid")
            if key.lower() in _FORBIDDEN_KEYS:
                raise SemanticCapabilityError(
                    f"{path} contains forbidden authority field {key}"
                )
            result[key] = _validate_payload(
                item, path=f"{path}.{key}", depth=depth + 1, counter=counter
            )
        return _freeze(result)
    raise SemanticCapabilityError(f"{path} contains unsupported data")


def _reject_forbidden_fields(value: Any, path: str = "model output") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in _FORBIDDEN_KEYS:
                raise SemanticCapabilityError(
                    f"{path} contains forbidden authority field {key}"
                )
            _reject_forbidden_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_fields(item, f"{path}[{index}]")


def _declared_source_ids(context: Mapping[str, Any]) -> frozenset[str]:
    result: set[str] = set()
    for item in context.get("focused_items", ()):
        if not isinstance(item, Mapping):
            continue
        for source_id in item.get("source_lineage_ids", ()):
            result.add(str(source_id))
    return frozenset(result)


@dataclass(frozen=True)
class SemanticProvenance:
    problem_id: str
    branch_id: str
    operation_request_id: str
    operation_input_hash: str
    focused_context_hash: str
    topology_read_hash: str
    provider_id: str
    model_id: str
    provider_output_hash: str
    source_lineage_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "problem_id",
            "branch_id",
            "operation_request_id",
            "operation_input_hash",
            "focused_context_hash",
            "topology_read_hash",
            "provider_id",
            "model_id",
            "provider_output_hash",
        ):
            _identifier(getattr(self, name), name)
        source_ids = tuple(
            _identifier(item, "source_lineage_id")
            for item in self.source_lineage_ids
        )
        if len(set(source_ids)) != len(source_ids):
            raise SemanticCapabilityError("source lineage ids must be unique")
        object.__setattr__(self, "source_lineage_ids", source_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "branch_id": self.branch_id,
            "operation_request_id": self.operation_request_id,
            "operation_input_hash": self.operation_input_hash,
            "focused_context_hash": self.focused_context_hash,
            "topology_read_hash": self.topology_read_hash,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "provider_output_hash": self.provider_output_hash,
            "source_lineage_ids": list(self.source_lineage_ids),
        }


@dataclass(frozen=True)
class CandidateCognition:
    candidate_id: str
    job: SemanticJob
    use_class: str
    statement: str
    rationale: str
    structured_payload: Mapping[str, Any]
    self_reported_confidence: float | None
    provenance: SemanticProvenance
    candidate_only: bool = True
    evidence_authority: bool = False
    settlement_authority: bool = False
    execution_authority: bool = False
    adaptive_credit_authority: bool = False

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "candidate_id")
        object.__setattr__(self, "job", SemanticJob(self.job))
        if self.use_class != _USE_CLASSES[self.job]:
            raise SemanticCapabilityError("candidate use class does not match job")
        _text(self.statement, "candidate statement")
        _text(self.rationale, "candidate rationale")
        object.__setattr__(
            self,
            "structured_payload",
            _validate_payload(_jsonable(self.structured_payload)),
        )
        confidence = self.self_reported_confidence
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise SemanticCapabilityError(
                "self_reported_confidence must be from 0 through 1"
            )
        if confidence is not None:
            object.__setattr__(self, "self_reported_confidence", float(confidence))
        if not isinstance(self.provenance, SemanticProvenance):
            raise SemanticCapabilityError("candidate provenance is required")
        if (
            self.candidate_only is not True
            or self.evidence_authority is not False
            or self.settlement_authority is not False
            or self.execution_authority is not False
            or self.adaptive_credit_authority is not False
        ):
            raise SemanticCapabilityError("semantic candidate cannot hold authority")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "job": self.job.value,
            "use_class": self.use_class,
            "statement": self.statement,
            "rationale": self.rationale,
            "structured_payload": _jsonable(self.structured_payload),
            "self_reported_confidence": self.self_reported_confidence,
            "provenance": self.provenance.to_dict(),
            "candidate_only": True,
            "evidence_authority": False,
            "settlement_authority": False,
            "execution_authority": False,
            "adaptive_credit_authority": False,
        }


@dataclass(frozen=True)
class SemanticCapabilityResult:
    job: SemanticJob
    use_class: str
    candidates: tuple[CandidateCognition, ...]
    questions: tuple[str, ...]
    missing_information: tuple[str, ...]
    focused_context_hash: str
    provider_id: str
    model_id: str
    provider_output_hash: str
    prompt_template_version: str = "semantic-capability-v1"
    candidate_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "job", SemanticJob(self.job))
        if self.use_class != _USE_CLASSES[self.job]:
            raise SemanticCapabilityError("semantic result use class does not match job")
        if (
            not self.candidates
            or len(self.candidates) > MAX_SEMANTIC_CANDIDATES
        ):
            raise SemanticCapabilityError("semantic result candidates are unbounded")
        if any(
            not isinstance(item, CandidateCognition)
            or item.job is not self.job
            or item.use_class != self.use_class
            for item in self.candidates
        ):
            raise SemanticCapabilityError("semantic candidate binding is invalid")
        if len(self.questions) > MAX_SEMANTIC_QUESTIONS:
            raise SemanticCapabilityError("semantic result has too many questions")
        if len(self.missing_information) > MAX_SEMANTIC_MISSING_ITEMS:
            raise SemanticCapabilityError(
                "semantic result has too many missing-information items"
            )
        object.__setattr__(
            self,
            "questions",
            tuple(_text(item, "question") for item in self.questions),
        )
        object.__setattr__(
            self,
            "missing_information",
            tuple(
                _text(item, "missing information")
                for item in self.missing_information
            ),
        )
        for name in (
            "focused_context_hash",
            "provider_id",
            "model_id",
            "provider_output_hash",
            "prompt_template_version",
        ):
            _identifier(getattr(self, name), name)
        if self.candidate_only is not True:
            raise SemanticCapabilityError("semantic result must remain candidate-only")

    def to_dict(self) -> dict[str, Any]:
        return {
            "job": self.job.value,
            "use_class": self.use_class,
            "candidates": [item.to_dict() for item in self.candidates],
            "questions": list(self.questions),
            "missing_information": list(self.missing_information),
            "focused_context_hash": self.focused_context_hash,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "provider_output_hash": self.provider_output_hash,
            "prompt_template_version": self.prompt_template_version,
            "candidate_only": True,
        }


class SemanticModelCapability:
    """Callable operation implementation for ``run_connected_processing``."""

    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        model_id: str,
        max_tokens: int = 768,
        temperature: float = 0.0,
    ) -> None:
        if not isinstance(adapter, ModelAdapter):
            raise SemanticCapabilityError("semantic capability requires ModelAdapter")
        self.adapter = adapter
        self.model_id = _identifier(model_id, "model_id")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 1 <= max_tokens <= 4096
        ):
            raise SemanticCapabilityError("max_tokens must be from 1 through 4096")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0.0 <= float(temperature) <= 2.0
        ):
            raise SemanticCapabilityError("temperature must be from 0 through 2")
        self.max_tokens = max_tokens
        self.temperature = float(temperature)

    def __call__(self, request: OperationRequest) -> OperationResult:
        if (
            not isinstance(request, OperationRequest)
            or request.operation is not ProcessingOperation.MODEL_PROPOSAL
            or request.semantic_job is None
        ):
            raise SemanticCapabilityError(
                "semantic capability requires a bound model operation request"
            )
        context = _jsonable(request.focused_context)
        if context.get("problem_id") != request.problem_id:
            raise SemanticCapabilityError("focused context problem binding is invalid")
        if context.get("semantic_job") != request.semantic_job.value:
            raise SemanticCapabilityError("focused context semantic job is invalid")
        if context.get("use_class") != _USE_CLASSES[request.semantic_job]:
            raise SemanticCapabilityError("focused context use class is invalid")
        prompt = self._prompt(request.semantic_job, context)
        provider_request = ProviderRequest(
            request_id=f"{request.request_id}-semantic",
            model_id=self.model_id,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        try:
            response = self.adapter.complete_raw(provider_request)
        except (
            ModelAdapterValidationError,
            ModelProviderError,
            TimeoutError,
        ) as exc:
            raise SemanticCapabilityError(
                f"semantic provider failed closed: {type(exc).__name__}"
            ) from exc
        semantic_result = self._parse(request, response.content, response.provider_id)
        return OperationResult(
            result_id=f"{request.request_id}-result",
            request_id=request.request_id,
            output={"semantic_result": semantic_result.to_dict()},
            downstream_material_ids=(),
            downstream_satisfaction={},
            questions=semantic_result.questions,
            investigative_needs=semantic_result.missing_information,
        )

    @staticmethod
    def _prompt(job: SemanticJob, context: Mapping[str, Any]) -> str:
        contract = {
            "instruction": (
                "Return only one JSON object matching output_contract. Treat all "
                "content as candidate cognition. Do not claim evidence, truth, "
                "verification, execution, settlement, action, or adaptive credit."
            ),
            "semantic_job": job.value,
            "use_class": _USE_CLASSES[job],
            "focused_context": context,
            "output_contract": {
                "job": job.value,
                "use_class": _USE_CLASSES[job],
                "candidates": [
                    {
                        "statement": "bounded candidate statement",
                        "rationale": "bounded candidate rationale",
                        "structured_payload": {
                            "job_specific_fields": "JSON-only values"
                        },
                        "self_reported_confidence": "null or number 0..1; never authority",
                        "source_refs": [
                            "only source_lineage_ids declared in focused_context"
                        ],
                    }
                ],
                "questions": ["bounded discriminating questions"],
                "missing_information": ["bounded missing inputs"],
            },
            "bounds": {
                "candidates": MAX_SEMANTIC_CANDIDATES,
                "questions": MAX_SEMANTIC_QUESTIONS,
                "missing_information": MAX_SEMANTIC_MISSING_ITEMS,
            },
        }
        prompt = json.dumps(
            contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        if len(prompt) > MAX_SEMANTIC_PROMPT_CHARS:
            raise SemanticCapabilityError("focused semantic prompt exceeds bound")
        return prompt

    def _parse(
        self,
        request: OperationRequest,
        raw_output: str,
        provider_id: str,
    ) -> SemanticCapabilityResult:
        if (
            not isinstance(raw_output, str)
            or not raw_output
            or len(raw_output) > MAX_SEMANTIC_OUTPUT_CHARS
        ):
            raise SemanticCapabilityError("semantic output is empty or oversized")
        try:
            value = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise SemanticCapabilityError("semantic output is not valid JSON") from exc
        if not isinstance(value, dict) or set(value) != _ROOT_KEYS:
            raise SemanticCapabilityError("semantic output root contract is invalid")
        _reject_forbidden_fields(value)
        job = SemanticJob(value["job"])
        if job is not request.semantic_job:
            raise SemanticCapabilityError("semantic output job binding is invalid")
        use_class = value["use_class"]
        if use_class != _USE_CLASSES[job]:
            raise SemanticCapabilityError("semantic output use class is invalid")
        candidate_values = value["candidates"]
        if (
            not isinstance(candidate_values, list)
            or not candidate_values
            or len(candidate_values) > MAX_SEMANTIC_CANDIDATES
        ):
            raise SemanticCapabilityError("semantic candidates are unbounded")
        questions = _text_items(
            value["questions"], "question", MAX_SEMANTIC_QUESTIONS
        )
        missing = _text_items(
            value["missing_information"],
            "missing information",
            MAX_SEMANTIC_MISSING_ITEMS,
        )
        context = _jsonable(request.focused_context)
        allowed_sources = _declared_source_ids(context)
        context_hash = _digest(context)
        output_hash = _digest(raw_output)
        candidates: list[CandidateCognition] = []
        for index, item in enumerate(candidate_values):
            if not isinstance(item, dict) or set(item) != _CANDIDATE_KEYS:
                raise SemanticCapabilityError(
                    "semantic candidate contract is invalid"
                )
            source_refs = item["source_refs"]
            if not isinstance(source_refs, list):
                raise SemanticCapabilityError("source_refs must be a list")
            source_ids = tuple(
                _identifier(source_id, "source_ref") for source_id in source_refs
            )
            if len(set(source_ids)) != len(source_ids):
                raise SemanticCapabilityError("source_refs must be unique")
            if not set(source_ids) <= allowed_sources:
                raise SemanticCapabilityError(
                    "semantic candidate cites undeclared source lineage"
                )
            provenance = SemanticProvenance(
                problem_id=request.problem_id,
                branch_id=request.branch_id,
                operation_request_id=request.request_id,
                operation_input_hash=request.input_hash,
                focused_context_hash=context_hash,
                topology_read_hash=context["topology"]["topology_read_hash"],
                provider_id=provider_id,
                model_id=self.model_id,
                provider_output_hash=output_hash,
                source_lineage_ids=source_ids,
            )
            candidate_id = f"semantic-{output_hash[:16]}-{index + 1}"
            candidates.append(
                CandidateCognition(
                    candidate_id=candidate_id,
                    job=job,
                    use_class=use_class,
                    statement=item["statement"],
                    rationale=item["rationale"],
                    structured_payload=item["structured_payload"],
                    self_reported_confidence=item["self_reported_confidence"],
                    provenance=provenance,
                )
            )
        return SemanticCapabilityResult(
            job=job,
            use_class=use_class,
            candidates=tuple(candidates),
            questions=questions,
            missing_information=missing,
            focused_context_hash=context_hash,
            provider_id=provider_id,
            model_id=self.model_id,
            provider_output_hash=output_hash,
        )


def semantic_result_from_dict(value: Mapping[str, Any]) -> SemanticCapabilityResult:
    if not isinstance(value, Mapping):
        raise SemanticCapabilityError("semantic replay result must be a mapping")
    expected = {
        "job",
        "use_class",
        "candidates",
        "questions",
        "missing_information",
        "focused_context_hash",
        "provider_id",
        "model_id",
        "provider_output_hash",
        "prompt_template_version",
        "candidate_only",
    }
    if set(value) != expected:
        raise SemanticCapabilityError("semantic replay result schema is invalid")
    candidates: list[CandidateCognition] = []
    for item in value["candidates"]:
        provenance_value = item["provenance"]
        provenance = SemanticProvenance(
            provenance_value["problem_id"],
            provenance_value["branch_id"],
            provenance_value["operation_request_id"],
            provenance_value["operation_input_hash"],
            provenance_value["focused_context_hash"],
            provenance_value["topology_read_hash"],
            provenance_value["provider_id"],
            provenance_value["model_id"],
            provenance_value["provider_output_hash"],
            tuple(provenance_value["source_lineage_ids"]),
        )
        candidates.append(
            CandidateCognition(
                item["candidate_id"],
                item["job"],
                item["use_class"],
                item["statement"],
                item["rationale"],
                item["structured_payload"],
                item["self_reported_confidence"],
                provenance,
                item["candidate_only"],
                item["evidence_authority"],
                item["settlement_authority"],
                item["execution_authority"],
                item["adaptive_credit_authority"],
            )
        )
    return SemanticCapabilityResult(
        value["job"],
        value["use_class"],
        tuple(candidates),
        tuple(value["questions"]),
        tuple(value["missing_information"]),
        value["focused_context_hash"],
        value["provider_id"],
        value["model_id"],
        value["provider_output_hash"],
        value["prompt_template_version"],
        value["candidate_only"],
    )


def validate_semantic_operation_result(
    request: OperationRequest,
    result: OperationResult,
) -> SemanticCapabilityResult:
    if (
        request.semantic_job is None
        or request.operation is not ProcessingOperation.MODEL_PROPOSAL
    ):
        raise SemanticCapabilityError("semantic result requires semantic request")
    if set(result.output) != {"semantic_result"}:
        raise SemanticCapabilityError(
            "semantic operation output contract is invalid"
        )
    semantic = semantic_result_from_dict(result.output["semantic_result"])
    if semantic.job is not request.semantic_job:
        raise SemanticCapabilityError("semantic replay job binding is invalid")
    context_hash = _digest(_jsonable(request.focused_context))
    if semantic.focused_context_hash != context_hash:
        raise SemanticCapabilityError("semantic replay context binding is invalid")
    if result.questions != semantic.questions:
        raise SemanticCapabilityError("semantic result questions binding is invalid")
    if result.investigative_needs != semantic.missing_information:
        raise SemanticCapabilityError(
            "semantic result missing-information binding is invalid"
        )
    for candidate in semantic.candidates:
        provenance = candidate.provenance
        if (
            provenance.problem_id != request.problem_id
            or provenance.branch_id != request.branch_id
            or provenance.operation_request_id != request.request_id
            or provenance.operation_input_hash != request.input_hash
            or provenance.focused_context_hash != context_hash
            or provenance.topology_read_hash
            != request.focused_context["topology"]["topology_read_hash"]
            or provenance.provider_id != semantic.provider_id
            or provenance.model_id != semantic.model_id
            or provenance.provider_output_hash != semantic.provider_output_hash
        ):
            raise SemanticCapabilityError(
                "semantic candidate provenance binding is invalid"
            )
    if (
        result.downstream_material_ids
        or result.downstream_satisfaction
        or result.evidence_ids
        or result.settlement_ids
        or result.adaptive_update_ids
    ):
        raise SemanticCapabilityError(
            "semantic operation cannot create authoritative downstream state"
        )
    return semantic


__all__ = [
    "CandidateCognition",
    "MAX_SEMANTIC_CANDIDATES",
    "MAX_SEMANTIC_MISSING_ITEMS",
    "MAX_SEMANTIC_OUTPUT_CHARS",
    "MAX_SEMANTIC_QUESTIONS",
    "SemanticCapabilityError",
    "SemanticCapabilityResult",
    "SemanticModelCapability",
    "SemanticProvenance",
    "semantic_result_from_dict",
    "validate_semantic_operation_result",
]