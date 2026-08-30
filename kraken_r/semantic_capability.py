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
MAX_SEMANTIC_VERIFICATION_OBLIGATIONS = 6


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
_REPLAY_CANDIDATE_KEYS = frozenset(
    {
        "candidate_id",
        "job",
        "use_class",
        "statement",
        "rationale",
        "structured_payload",
        "self_reported_confidence",
        "provenance",
        "candidate_only",
        "evidence_authority",
        "settlement_authority",
        "execution_authority",
        "adaptive_credit_authority",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "problem_id",
        "branch_id",
        "operation_request_id",
        "operation_input_hash",
        "focused_context_hash",
        "topology_read_hash",
        "provider_id",
        "model_id",
        "provider_output_hash",
        "source_lineage_ids",
    }
)
_INVOCATION_KEYS = frozenset(
    {
        "provider_request_id",
        "operation_request_id",
        "model_id",
        "max_tokens",
        "temperature",
        "prompt_template_version",
        "prompt_hash",
        "configuration_hash",
        "request_hash",
    }
)
_JOB_PAYLOAD_KEYS = {
    SemanticJob.RECALL: frozenset(
        {"recollections", "verification_obligations"}
    ),
    SemanticJob.MECHANISM_GENERATION: frozenset(
        {
            "components",
            "causal_steps",
            "assumptions",
            "verification_obligations",
        }
    ),
    SemanticJob.COMPETING_EXPLANATIONS: frozenset(
        {"alternatives", "discriminators", "verification_obligations"}
    ),
    SemanticJob.CROSS_DOMAIN_CORRESPONDENCE: frozenset(
        {"correspondences", "scope_limits", "verification_obligations"}
    ),
    SemanticJob.VARIABLE_EQUATION_EXTRACTION: frozenset(
        {"variables", "equations", "verification_obligations"}
    ),
    SemanticJob.SOURCE_INTERPRETATION: frozenset(
        {"interpretations", "limitations", "verification_obligations"}
    ),
    SemanticJob.FALSIFIER_GENERATION: frozenset(
        {"falsifiers", "verification_obligations"}
    ),
    SemanticJob.MISSING_INFORMATION_DETECTION: frozenset(
        {"missing_items", "blocking_questions", "verification_obligations"}
    ),
}
_JOB_ITEM_KEYS = {
    "recollections": frozenset({"content", "relevance"}),
    "components": frozenset({"name", "role", "inputs", "outputs"}),
    "alternatives": frozenset(
        {"label", "explanation", "discriminators"}
    ),
    "correspondences": frozenset(
        {"source_concept", "target_concept", "mapping"}
    ),
    "variables": frozenset({"name", "description", "unit", "role"}),
    "equations": frozenset(
        {"expression", "variable_names", "assumptions"}
    ),
    "interpretations": frozenset(
        {"source_ref", "passage", "interpretation"}
    ),
    "falsifiers": frozenset(
        {"test", "if_supported", "if_disconfirmed", "discriminates"}
    ),
    "missing_items": frozenset({"item", "why_needed", "blocks"}),
}
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
    if not isinstance(value, (list, tuple)) or len(value) > limit:
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


def _record_list(
    value: Any,
    field_name: str,
    item_keys: frozenset[str],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= limit:
        raise SemanticCapabilityError(
            f"{field_name} must contain from 1 through {limit} records"
        )
    records: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != item_keys:
            raise SemanticCapabilityError(
                f"{field_name}[{index}] has the wrong typed schema"
            )
        records.append(item)
    return records


def _validate_job_payload(
    value: Any,
    job: SemanticJob,
    allowed_sources: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != _JOB_PAYLOAD_KEYS[job]:
        raise SemanticCapabilityError(
            f"{job.value} payload does not match its required schema"
        )
    verification = _text_items(
        value["verification_obligations"],
        "verification obligation",
        MAX_SEMANTIC_VERIFICATION_OBLIGATIONS,
    )
    if not verification:
        raise SemanticCapabilityError(
            f"{job.value} requires a verification obligation"
        )
    normalized: dict[str, Any] = {
        "verification_obligations": verification,
    }

    def bounded_text(value: Any, field_name: str, limit: int = 8) -> tuple[str, ...]:
        return _text_items(value, field_name, limit)

    def required_text(
        value: Any, field_name: str, limit: int = 8
    ) -> tuple[str, ...]:
        items = bounded_text(value, field_name, limit)
        if not items:
            raise SemanticCapabilityError(f"{field_name} cannot be empty")
        return items

    if job is SemanticJob.RECALL:
        records = _record_list(
            value["recollections"], "recollections", _JOB_ITEM_KEYS["recollections"]
        )
        normalized["recollections"] = tuple(
            {
                "content": _text(item["content"], "recollection content"),
                "relevance": _text(item["relevance"], "recollection relevance"),
            }
            for item in records
        )
    elif job is SemanticJob.MECHANISM_GENERATION:
        records = _record_list(
            value["components"], "components", _JOB_ITEM_KEYS["components"]
        )
        normalized["components"] = tuple(
            {
                "name": _text(item["name"], "component name"),
                "role": _text(item["role"], "component role"),
                "inputs": required_text(item["inputs"], "component inputs"),
                "outputs": required_text(item["outputs"], "component outputs"),
            }
            for item in records
        )
        normalized["causal_steps"] = required_text(
            value["causal_steps"], "causal step"
        )
        normalized["assumptions"] = required_text(
            value["assumptions"], "mechanism assumption"
        )
    elif job is SemanticJob.COMPETING_EXPLANATIONS:
        records = _record_list(
            value["alternatives"],
            "alternatives",
            _JOB_ITEM_KEYS["alternatives"],
            limit=4,
        )
        if len(records) < 2:
            raise SemanticCapabilityError(
                "competing explanations require at least two alternatives"
            )
        normalized["alternatives"] = tuple(
            {
                "label": _text(item["label"], "alternative label"),
                "explanation": _text(item["explanation"], "alternative explanation"),
                "discriminators": required_text(
                    item["discriminators"], "alternative discriminator", 4
                ),
            }
            for item in records
        )
        normalized["discriminators"] = required_text(
            value["discriminators"], "explanation discriminator"
        )
    elif job is SemanticJob.CROSS_DOMAIN_CORRESPONDENCE:
        records = _record_list(
            value["correspondences"],
            "correspondences",
            _JOB_ITEM_KEYS["correspondences"],
        )
        normalized["correspondences"] = tuple(
            {
                "source_concept": _text(
                    item["source_concept"], "source concept"
                ),
                "target_concept": _text(
                    item["target_concept"], "target concept"
                ),
                "mapping": _text(item["mapping"], "concept mapping"),
            }
            for item in records
        )
        normalized["scope_limits"] = required_text(
            value["scope_limits"], "correspondence scope limit"
        )
    elif job is SemanticJob.VARIABLE_EQUATION_EXTRACTION:
        variables = _record_list(
            value["variables"], "variables", _JOB_ITEM_KEYS["variables"]
        )
        normalized["variables"] = tuple(
            {
                "name": _text(item["name"], "variable name"),
                "description": _text(item["description"], "variable description"),
                "unit": _text(item["unit"], "variable unit"),
                "role": _text(item["role"], "variable role"),
            }
            for item in variables
        )
        equations = _record_list(
            value["equations"], "equations", _JOB_ITEM_KEYS["equations"]
        )
        normalized["equations"] = tuple(
            {
                "expression": _text(item["expression"], "equation expression"),
                "variable_names": required_text(
                    item["variable_names"], "equation variable name", 8
                ),
                "assumptions": required_text(
                    item["assumptions"], "equation assumption", 4
                ),
            }
            for item in equations
        )
    elif job is SemanticJob.SOURCE_INTERPRETATION:
        records = _record_list(
            value["interpretations"],
            "interpretations",
            _JOB_ITEM_KEYS["interpretations"],
        )
        normalized["interpretations"] = tuple(
            {
                "source_ref": _identifier(item["source_ref"], "source_ref"),
                "passage": _text(item["passage"], "source passage"),
                "interpretation": _text(
                    item["interpretation"], "source interpretation"
                ),
            }
            for item in records
        )
        if not all(
            item["source_ref"] in allowed_sources
            for item in normalized["interpretations"]
        ):
            raise SemanticCapabilityError(
                "source interpretation cites undeclared source lineage"
            )
        normalized["limitations"] = required_text(
            value["limitations"], "interpretation limitation"
        )
    elif job is SemanticJob.FALSIFIER_GENERATION:
        records = _record_list(
            value["falsifiers"], "falsifiers", _JOB_ITEM_KEYS["falsifiers"]
        )
        normalized["falsifiers"] = tuple(
            {
                "test": _text(item["test"], "falsifier test"),
                "if_supported": _text(
                    item["if_supported"], "supported prediction"
                ),
                "if_disconfirmed": _text(
                    item["if_disconfirmed"], "disconfirmed prediction"
                ),
                "discriminates": _text(
                    item["discriminates"], "falsifier discriminator"
                ),
            }
            for item in records
        )
    elif job is SemanticJob.MISSING_INFORMATION_DETECTION:
        records = _record_list(
            value["missing_items"],
            "missing_items",
            _JOB_ITEM_KEYS["missing_items"],
        )
        normalized["missing_items"] = tuple(
            {
                "item": _text(item["item"], "missing item"),
                "why_needed": _text(item["why_needed"], "missing item reason"),
                "blocks": _text(item["blocks"], "missing item blocker"),
            }
            for item in records
        )
        normalized["blocking_questions"] = required_text(
            value["blocking_questions"], "blocking question"
        )
    else:
        raise SemanticCapabilityError("unsupported semantic job")
    return _freeze(normalized)


def _declared_source_ids(context: Mapping[str, Any]) -> frozenset[str]:
    result: set[str] = set()
    for item in context.get("focused_items", ()):
        if not isinstance(item, Mapping):
            continue
        for source_id in item.get("source_lineage_ids", ()):
            result.add(str(source_id))
    return frozenset(result)


def _job_payload_example(job: SemanticJob) -> Mapping[str, Any]:
    examples: dict[SemanticJob, Mapping[str, Any]] = {
        SemanticJob.RECALL: {
            "recollections": [
                {"content": "candidate recollection", "relevance": "why it may apply"}
            ],
            "verification_obligations": ["verify recollection against a declared source"],
        },
        SemanticJob.MECHANISM_GENERATION: {
            "components": [
                {
                    "name": "component",
                    "role": "candidate role",
                    "inputs": ["input"],
                    "outputs": ["output"],
                }
            ],
            "causal_steps": ["candidate causal step"],
            "assumptions": ["declared assumption"],
            "verification_obligations": ["test each causal dependency"],
        },
        SemanticJob.COMPETING_EXPLANATIONS: {
            "alternatives": [
                {
                    "label": "alternative-a",
                    "explanation": "candidate explanation",
                    "discriminators": ["observation that distinguishes it"],
                }
            ],
            "discriminators": ["cross-alternative discriminating observation"],
            "verification_obligations": ["compare alternatives under the same evidence"],
        },
        SemanticJob.CROSS_DOMAIN_CORRESPONDENCE: {
            "correspondences": [
                {
                    "source_concept": "source concept",
                    "target_concept": "target concept",
                    "mapping": "bounded candidate correspondence",
                }
            ],
            "scope_limits": ["where the analogy stops"],
            "verification_obligations": ["verify mapping and scope independently"],
        },
        SemanticJob.VARIABLE_EQUATION_EXTRACTION: {
            "variables": [
                {
                    "name": "x",
                    "description": "candidate variable",
                    "unit": "declared unit or dimensionless",
                    "role": "input, output, parameter, or state",
                }
            ],
            "equations": [
                {
                    "expression": "candidate equation",
                    "variable_names": ["x"],
                    "assumptions": ["equation assumption"],
                }
            ],
            "verification_obligations": ["check dimensions, quantities, and dependencies"],
        },
        SemanticJob.SOURCE_INTERPRETATION: {
            "interpretations": [
                {
                    "source_ref": "declared source_lineage_id",
                    "passage": "bounded source passage",
                    "interpretation": "candidate interpretation",
                }
            ],
            "limitations": ["source limitation or ambiguity"],
            "verification_obligations": ["compare interpretation with the cited source"],
        },
        SemanticJob.FALSIFIER_GENERATION: {
            "falsifiers": [
                {
                    "test": "candidate falsifying test",
                    "if_supported": "observation expected if candidate survives",
                    "if_disconfirmed": "observation expected if candidate fails",
                    "discriminates": "claim or alternatives distinguished",
                }
            ],
            "verification_obligations": ["run an independent discriminating check"],
        },
        SemanticJob.MISSING_INFORMATION_DETECTION: {
            "missing_items": [
                {
                    "item": "missing information",
                    "why_needed": "why it matters",
                    "blocks": "claim, dependency, or decision it blocks",
                }
            ],
            "blocking_questions": ["question required before synthesis"],
            "verification_obligations": ["obtain the missing information from a valid source"],
        },
    }
    return examples[job]


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
        if not isinstance(self.provenance, SemanticProvenance):
            raise SemanticCapabilityError("candidate provenance is required")
        object.__setattr__(
            self,
            "structured_payload",
            _validate_job_payload(
                _jsonable(self.structured_payload),
                self.job,
                frozenset(self.provenance.source_lineage_ids),
            ),
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
class SemanticInvocationEnvelope:
    provider_request_id: str
    operation_request_id: str
    model_id: str
    max_tokens: int
    temperature: float
    prompt_template_version: str
    prompt_hash: str
    configuration_hash: str
    request_hash: str

    def __post_init__(self) -> None:
        for name in (
            "provider_request_id",
            "operation_request_id",
            "model_id",
            "prompt_template_version",
            "prompt_hash",
            "configuration_hash",
            "request_hash",
        ):
            _identifier(getattr(self, name), name)
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or not 1 <= self.max_tokens <= 4096
        ):
            raise SemanticCapabilityError(
                "invocation max_tokens must be from 1 through 4096"
            )
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not 0.0 <= float(self.temperature) <= 2.0
        ):
            raise SemanticCapabilityError(
                "invocation temperature must be from 0 through 2"
            )
        object.__setattr__(self, "temperature", float(self.temperature))
        expected_configuration_hash = _digest(
            {
                "model_id": self.model_id,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "prompt_template_version": self.prompt_template_version,
            }
        )
        if self.configuration_hash != expected_configuration_hash:
            raise SemanticCapabilityError(
                "semantic invocation configuration hash is invalid"
            )
        expected_request_hash = _digest(
            {
                "provider_request_id": self.provider_request_id,
                "operation_request_id": self.operation_request_id,
                "prompt_hash": self.prompt_hash,
                "configuration_hash": self.configuration_hash,
            }
        )
        if self.request_hash != expected_request_hash:
            raise SemanticCapabilityError(
                "semantic invocation request hash is invalid"
            )

    @classmethod
    def create(
        cls,
        *,
        provider_request_id: str,
        operation_request_id: str,
        model_id: str,
        max_tokens: int,
        temperature: float,
        prompt_template_version: str,
        prompt_hash: str,
    ) -> "SemanticInvocationEnvelope":
        configuration_hash = _digest(
            {
                "model_id": model_id,
                "max_tokens": max_tokens,
                "temperature": float(temperature),
                "prompt_template_version": prompt_template_version,
            }
        )
        request_hash = _digest(
            {
                "provider_request_id": provider_request_id,
                "operation_request_id": operation_request_id,
                "prompt_hash": prompt_hash,
                "configuration_hash": configuration_hash,
            }
        )
        return cls(
            provider_request_id,
            operation_request_id,
            model_id,
            max_tokens,
            float(temperature),
            prompt_template_version,
            prompt_hash,
            configuration_hash,
            request_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_request_id": self.provider_request_id,
            "operation_request_id": self.operation_request_id,
            "model_id": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "prompt_template_version": self.prompt_template_version,
            "prompt_hash": self.prompt_hash,
            "configuration_hash": self.configuration_hash,
            "request_hash": self.request_hash,
        }


@dataclass(frozen=True)
class SemanticCapabilityResult:
    job: SemanticJob
    use_class: str
    candidates: tuple[CandidateCognition, ...]
    questions: tuple[str, ...]
    missing_information: tuple[str, ...]
    focused_context_hash: str
    invocation: SemanticInvocationEnvelope
    provider_id: str
    model_id: str
    provider_output_hash: str
    prompt_template_version: str = "semantic-capability-v2"
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
        if not isinstance(self.invocation, SemanticInvocationEnvelope):
            raise SemanticCapabilityError("semantic invocation envelope is required")
        if self.invocation.model_id != self.model_id:
            raise SemanticCapabilityError(
                "semantic result model does not match invocation"
            )
        if self.invocation.prompt_template_version != self.prompt_template_version:
            raise SemanticCapabilityError(
                "semantic result template does not match invocation"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "job": self.job.value,
            "use_class": self.use_class,
            "candidates": [item.to_dict() for item in self.candidates],
            "questions": list(self.questions),
            "missing_information": list(self.missing_information),
            "focused_context_hash": self.focused_context_hash,
            "invocation": self.invocation.to_dict(),
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "provider_output_hash": self.provider_output_hash,
            "prompt_template_version": self.prompt_template_version,
            "candidate_only": True,
        }


class SemanticModelCapability:
    """Callable operation implementation for ``run_connected_processing``."""

    PROMPT_TEMPLATE_VERSION = "semantic-capability-v2"

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
        expected_configuration = {
            "model_id": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "prompt_template_version": self.PROMPT_TEMPLATE_VERSION,
        }
        if _jsonable(request.semantic_configuration) != expected_configuration:
            raise SemanticCapabilityError(
                "semantic capability configuration does not match its declaration"
            )
        prompt = self._prompt(request.semantic_job, context)
        prompt_hash = _digest(prompt)
        provider_request_id = f"{request.request_id}-semantic"
        invocation = SemanticInvocationEnvelope.create(
            provider_request_id=provider_request_id,
            operation_request_id=request.request_id,
            model_id=self.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            prompt_template_version=self.PROMPT_TEMPLATE_VERSION,
            prompt_hash=prompt_hash,
        )
        provider_request = ProviderRequest(
            request_id=provider_request_id,
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
        semantic_result = self._parse(
            request,
            response.content,
            response.provider_id,
            invocation,
        )
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
                        "structured_payload": _job_payload_example(job),
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
        invocation: SemanticInvocationEnvelope,
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
        output_hash = _digest(value)
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
                    structured_payload=_validate_job_payload(
                        item["structured_payload"], job, allowed_sources
                    ),
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
            invocation=invocation,
            provider_id=provider_id,
            model_id=self.model_id,
            provider_output_hash=output_hash,
        )


def semantic_result_from_dict(
    value: Mapping[str, Any],
    *,
    request: OperationRequest,
) -> SemanticCapabilityResult:
    if not isinstance(value, Mapping):
        raise SemanticCapabilityError("semantic replay result must be a mapping")
    expected = {
        "job",
        "use_class",
        "candidates",
        "questions",
        "missing_information",
        "focused_context_hash",
        "invocation",
        "provider_id",
        "model_id",
        "provider_output_hash",
        "prompt_template_version",
        "candidate_only",
    }
    if set(value) != expected:
        raise SemanticCapabilityError("semantic replay result schema is invalid")
    if not isinstance(request, OperationRequest) or request.semantic_job is None:
        raise SemanticCapabilityError(
            "semantic replay requires its bound operation request"
        )
    try:
        job = SemanticJob(value["job"])
    except (TypeError, ValueError) as exc:
        raise SemanticCapabilityError("semantic replay job is invalid") from exc
    if job is not request.semantic_job:
        raise SemanticCapabilityError("semantic replay job binding is invalid")
    if value["use_class"] != _USE_CLASSES[job]:
        raise SemanticCapabilityError("semantic replay use class is invalid")
    questions = _text_items(
        value["questions"], "question", MAX_SEMANTIC_QUESTIONS
    )
    missing = _text_items(
        value["missing_information"],
        "missing information",
        MAX_SEMANTIC_MISSING_ITEMS,
    )
    context = _jsonable(request.focused_context)
    context_hash = _digest(context)
    if value["focused_context_hash"] != context_hash:
        raise SemanticCapabilityError("semantic replay context binding is invalid")
    invocation_value = value["invocation"]
    if (
        not isinstance(invocation_value, Mapping)
        or set(invocation_value) != _INVOCATION_KEYS
    ):
        raise SemanticCapabilityError(
            "semantic invocation replay schema is invalid"
        )
    invocation = SemanticInvocationEnvelope(
        invocation_value["provider_request_id"],
        invocation_value["operation_request_id"],
        invocation_value["model_id"],
        invocation_value["max_tokens"],
        invocation_value["temperature"],
        invocation_value["prompt_template_version"],
        invocation_value["prompt_hash"],
        invocation_value["configuration_hash"],
        invocation_value["request_hash"],
    )
    if (
        invocation.operation_request_id != request.request_id
        or invocation.provider_request_id != f"{request.request_id}-semantic"
        or invocation.model_id
        != request.semantic_configuration["model_id"]
        or invocation.max_tokens
        != request.semantic_configuration["max_tokens"]
        or invocation.temperature
        != request.semantic_configuration["temperature"]
        or invocation.prompt_template_version
        != request.semantic_configuration["prompt_template_version"]
        or invocation.prompt_template_version
        != SemanticModelCapability.PROMPT_TEMPLATE_VERSION
        or invocation.prompt_hash
        != _digest(SemanticModelCapability._prompt(job, context))
    ):
        raise SemanticCapabilityError(
            "semantic invocation does not match the deterministic request"
        )
    if value["prompt_template_version"] != invocation.prompt_template_version:
        raise SemanticCapabilityError(
            "semantic replay prompt template binding is invalid"
        )
    if value["model_id"] != invocation.model_id:
        raise SemanticCapabilityError("semantic replay model binding is invalid")
    allowed_sources = _declared_source_ids(context)
    candidate_values = value["candidates"]
    if (
        not isinstance(candidate_values, (list, tuple))
        or not candidate_values
        or len(candidate_values) > MAX_SEMANTIC_CANDIDATES
    ):
        raise SemanticCapabilityError("semantic replay candidates are unbounded")
    candidates: list[CandidateCognition] = []
    provider_candidates: list[dict[str, Any]] = []
    for index, item in enumerate(candidate_values):
        if not isinstance(item, Mapping) or set(item) != _REPLAY_CANDIDATE_KEYS:
            raise SemanticCapabilityError(
                "semantic replay candidate schema is invalid"
            )
        provenance_value = item["provenance"]
        if (
            not isinstance(provenance_value, Mapping)
            or set(provenance_value) != _PROVENANCE_KEYS
        ):
            raise SemanticCapabilityError(
                "semantic replay provenance schema is invalid"
            )
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
        if not set(provenance.source_lineage_ids) <= allowed_sources:
            raise SemanticCapabilityError(
                "semantic replay cites undeclared source lineage"
            )
        if item["candidate_id"] != (
            f"semantic-{value['provider_output_hash'][:16]}-{index + 1}"
        ):
            raise SemanticCapabilityError(
                "semantic replay candidate identity is invalid"
            )
        if item["job"] != job.value or item["use_class"] != _USE_CLASSES[job]:
            raise SemanticCapabilityError(
                "semantic replay candidate job binding is invalid"
            )
        typed_payload = _validate_job_payload(
            _jsonable(item["structured_payload"]),
            job,
            allowed_sources,
        )
        candidates.append(
            CandidateCognition(
                item["candidate_id"],
                item["job"],
                item["use_class"],
                item["statement"],
                item["rationale"],
                typed_payload,
                item["self_reported_confidence"],
                provenance,
                item["candidate_only"],
                item["evidence_authority"],
                item["settlement_authority"],
                item["execution_authority"],
                item["adaptive_credit_authority"],
            )
        )
        provider_candidates.append(
            {
                "statement": item["statement"],
                "rationale": item["rationale"],
                "structured_payload": _jsonable(typed_payload),
                "self_reported_confidence": item["self_reported_confidence"],
                "source_refs": list(provenance.source_lineage_ids),
            }
        )
    provider_value = {
        "job": job.value,
        "use_class": _USE_CLASSES[job],
        "candidates": provider_candidates,
        "questions": list(questions),
        "missing_information": list(missing),
    }
    if value["provider_output_hash"] != _digest(provider_value):
        raise SemanticCapabilityError(
            "semantic replay provider output hash is invalid"
        )
    return SemanticCapabilityResult(
        job=job,
        use_class=value["use_class"],
        candidates=tuple(candidates),
        questions=questions,
        missing_information=missing,
        focused_context_hash=value["focused_context_hash"],
        invocation=invocation,
        provider_id=value["provider_id"],
        model_id=value["model_id"],
        provider_output_hash=value["provider_output_hash"],
        prompt_template_version=value["prompt_template_version"],
        candidate_only=value["candidate_only"],
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
    semantic = semantic_result_from_dict(
        result.output["semantic_result"],
        request=request,
    )
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
        semantic.invocation.operation_request_id != request.request_id
        or semantic.invocation.model_id != semantic.model_id
    ):
        raise SemanticCapabilityError(
            "semantic invocation/result binding is invalid"
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