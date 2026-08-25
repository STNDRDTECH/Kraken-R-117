"""Bounded, proposal-only task-integrity infrastructure for Kraken-R Stage 10.7.

The records in this module preserve a caller-supplied task while making a
structured interpretation, competing declared hypotheses, information-loss
findings, and adversarial review inspectable.  They are deliberately not an
executor, provider adapter, evidence store, controller, or adaptive reducer.
Review output is a bounded request for later independent verification only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping


class TaskIntegrityValidationError(ValueError):
    """Raised when a bounded task-integrity record is malformed or unbound."""


MAX_TASK_TEXT = 8_192
MAX_REQUIREMENTS = 16
MAX_HYPOTHESES = 8
MAX_PLAN_STEPS = 12
MAX_CLAIM_DEPENDENCIES = 8
MAX_RESULT_CLAIMS = 16
MAX_LOSS_FINDINGS = 32
MAX_REVIEW_FINDINGS = MAX_LOSS_FINDINGS + MAX_HYPOTHESES
MAX_VERIFICATION_QUESTIONS = MAX_REVIEW_FINDINGS
MAX_REVIEW_ROLES = 3


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(
        item.isspace() for item in value
    ):
        raise TaskIntegrityValidationError(
            f"{field_name} must be a non-empty identifier without whitespace"
        )
    return value


def _text(value: str, field_name: str, *, maximum: int = MAX_TASK_TEXT) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskIntegrityValidationError(f"{field_name} must be non-empty text")
    if len(value) > maximum:
        raise TaskIntegrityValidationError(f"{field_name} exceeds its bounded length")
    return value


def _bounded(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskIntegrityValidationError(f"{field_name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise TaskIntegrityValidationError(f"{field_name} must be between 0 and 1")
    return numeric


def _tuple(
    value: tuple[Any, ...] | list[Any],
    field_name: str,
    *,
    maximum: int,
) -> tuple[Any, ...]:
    if not isinstance(value, (tuple, list)):
        raise TaskIntegrityValidationError(f"{field_name} must be a tuple or list")
    values = tuple(value)
    if len(values) > maximum:
        raise TaskIntegrityValidationError(f"{field_name} exceeds its fixed limit")
    return values


def _identifiers(
    value: tuple[str, ...] | list[str],
    field_name: str,
    *,
    maximum: int,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    values = _tuple(value, field_name, maximum=maximum)
    if not allow_empty and not values:
        raise TaskIntegrityValidationError(f"{field_name} cannot be empty")
    for item in values:
        _identifier(item, field_name)
    if len(set(values)) != len(values):
        raise TaskIntegrityValidationError(f"{field_name} cannot repeat identities")
    return values


def _mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TaskIntegrityValidationError(f"{field_name} must be a mapping")
    return MappingProxyType(
        {str(key): _freeze(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _mapping(value, "nested mapping")
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RequirementKind(str, Enum):
    """The task role a preserved clause plays."""

    ASK = "ask"
    CONSTRAINT = "constraint"
    SUCCESS_CONDITION = "success_condition"
    AMBIGUITY = "ambiguity"
    REQUIRED_EVIDENCE = "required_evidence"
    CONTEXT = "context"


class BeliefStatus(str, Enum):
    """Declared status of an interpretation; never an evidence grade."""

    OPEN = "open"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    BLOCKED = "blocked"
    AMBIGUOUS = "ambiguous"


class LossKind(str, Enum):
    """A non-evidentiary way task meaning may have been lost."""

    OMITTED = "omitted"
    COMPRESSED = "compressed"
    UNDERWEIGHTED = "underweighted"
    UNSUPPORTED_DEPENDENCY = "unsupported_dependency"


class ReviewRole(str, Enum):
    """Purpose-limited review perspectives."""

    ANGEL = "angel"
    NEMESIS = "nemesis"
    ANTIMETABOLE = "antimetabole"


class VerificationTarget(str, Enum):
    """A later independent boundary that could answer a review question."""

    TOOL = "tool"
    EXECUTION = "execution"
    RETRIEVAL = "retrieval"


@dataclass(frozen=True)
class OriginalTask:
    """Immutable caller-owned task text and declared source provenance."""

    task_id: str
    text: str
    source: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        _identifier(self.task_id, "task_id")
        _text(self.text, "text")
        _text(self.source, "source", maximum=256)
        object.__setattr__(self, "provenance", _mapping(self.provenance, "provenance"))

    @property
    def task_hash(self) -> str:
        return _canonical_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "task_id": self.task_id,
            "text": self.text,
            "source": self.source,
            "provenance": _jsonable(self.provenance),
        }
        if include_hash:
            result["task_hash"] = self.task_hash
        return result


@dataclass(frozen=True)
class TaskRequirement:
    """One source-addressable task clause, preserved without interpretation."""

    requirement_id: str
    kind: RequirementKind
    source_text: str
    importance: float = 1.0

    def __post_init__(self) -> None:
        _identifier(self.requirement_id, "requirement_id")
        object.__setattr__(self, "kind", RequirementKind(self.kind))
        _text(self.source_text, "source_text", maximum=1_024)
        object.__setattr__(self, "importance", _bounded(self.importance, "importance"))

    @property
    def source_hash(self) -> str:
        return _canonical_hash(
            {"requirement_id": self.requirement_id, "kind": self.kind.value, "source_text": self.source_text}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "kind": self.kind.value,
            "source_text": self.source_text,
            "source_hash": self.source_hash,
            "importance": self.importance,
        }


@dataclass(frozen=True)
class TaskSpecification:
    """Bounded structured interpretation, explicitly bound to one original task."""

    specification_id: str
    original_task_id: str
    original_task_hash: str
    requirements: tuple[TaskRequirement, ...]
    interpretation: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        _identifier(self.specification_id, "specification_id")
        _identifier(self.original_task_id, "original_task_id")
        if not isinstance(self.original_task_hash, str) or len(self.original_task_hash) != 64:
            raise TaskIntegrityValidationError("original_task_hash must be a SHA-256 hash")
        requirements = _tuple(self.requirements, "requirements", maximum=MAX_REQUIREMENTS)
        if not requirements or any(not isinstance(item, TaskRequirement) for item in requirements):
            raise TaskIntegrityValidationError("requirements must contain TaskRequirement records")
        if len({item.requirement_id for item in requirements}) != len(requirements):
            raise TaskIntegrityValidationError("requirement identities must be unique")
        _text(self.interpretation, "interpretation", maximum=2_048)
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "provenance", _mapping(self.provenance, "provenance"))

    @classmethod
    def from_task(
        cls,
        specification_id: str,
        original_task: OriginalTask,
        requirements: tuple[TaskRequirement, ...],
        interpretation: str,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> "TaskSpecification":
        return cls(
            specification_id,
            original_task.task_id,
            original_task.task_hash,
            requirements,
            interpretation,
            provenance or {},
        )

    def validate_original_task(self, original_task: OriginalTask) -> None:
        if (
            original_task.task_id != self.original_task_id
            or original_task.task_hash != self.original_task_hash
        ):
            raise TaskIntegrityValidationError(
                "task specification does not bind the supplied original task"
            )

    @property
    def requirement_ids(self) -> tuple[str, ...]:
        return tuple(item.requirement_id for item in self.requirements)

    def to_dict(self) -> dict[str, Any]:
        return {
            "specification_id": self.specification_id,
            "original_task_id": self.original_task_id,
            "original_task_hash": self.original_task_hash,
            "requirements": [item.to_dict() for item in self.requirements],
            "interpretation": self.interpretation,
            "provenance": _jsonable(self.provenance),
        }


@dataclass(frozen=True)
class TaskHypothesis:
    """One declared candidate interpretation with bounded uncertainty lineage."""

    hypothesis_id: str
    specification_id: str
    statement: str
    requirement_ids: tuple[str, ...]
    support_ids: tuple[str, ...] = ()
    contradiction_ids: tuple[str, ...] = ()
    unresolved_dependency_ids: tuple[str, ...] = ()
    status: BeliefStatus = BeliefStatus.OPEN

    def __post_init__(self) -> None:
        _identifier(self.hypothesis_id, "hypothesis_id")
        _identifier(self.specification_id, "specification_id")
        _text(self.statement, "statement", maximum=2_048)
        object.__setattr__(
            self, "requirement_ids", _identifiers(
                self.requirement_ids, "requirement_ids", maximum=MAX_REQUIREMENTS, allow_empty=False
            )
        )
        for name in ("support_ids", "contradiction_ids", "unresolved_dependency_ids"):
            object.__setattr__(
                self, name, _identifiers(getattr(self, name), name, maximum=MAX_REQUIREMENTS)
            )
        object.__setattr__(self, "status", BeliefStatus(self.status))

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "specification_id": self.specification_id,
            "statement": self.statement,
            "requirement_ids": list(self.requirement_ids),
            "support_ids": list(self.support_ids),
            "contradiction_ids": list(self.contradiction_ids),
            "unresolved_dependency_ids": list(self.unresolved_dependency_ids),
            "status": self.status.value,
            "declared_only": True,
        }


@dataclass(frozen=True)
class BeliefState:
    """Finite competing interpretations; its labels are not evidence."""

    belief_state_id: str
    specification_id: str
    hypotheses: tuple[TaskHypothesis, ...]

    def __post_init__(self) -> None:
        _identifier(self.belief_state_id, "belief_state_id")
        _identifier(self.specification_id, "specification_id")
        hypotheses = _tuple(self.hypotheses, "hypotheses", maximum=MAX_HYPOTHESES)
        if not hypotheses or any(not isinstance(item, TaskHypothesis) for item in hypotheses):
            raise TaskIntegrityValidationError("hypotheses must contain TaskHypothesis records")
        if len({item.hypothesis_id for item in hypotheses}) != len(hypotheses):
            raise TaskIntegrityValidationError("hypothesis identities must be unique")
        if any(item.specification_id != self.specification_id for item in hypotheses):
            raise TaskIntegrityValidationError("hypotheses must bind this specification")
        object.__setattr__(self, "hypotheses", hypotheses)

    def validate_specification(self, specification: TaskSpecification) -> None:
        if self.specification_id != specification.specification_id:
            raise TaskIntegrityValidationError("belief state does not bind the specification")
        requirement_ids = set(specification.requirement_ids)
        for hypothesis in self.hypotheses:
            if not set(hypothesis.requirement_ids).issubset(requirement_ids):
                raise TaskIntegrityValidationError("hypothesis names an undeclared requirement")

    def to_dict(self) -> dict[str, Any]:
        return {
            "belief_state_id": self.belief_state_id,
            "specification_id": self.specification_id,
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "evidence_grade": "none",
        }


@dataclass(frozen=True)
class TaskPlan:
    """A declared bounded plan projection, not an action or execution request."""

    plan_id: str
    specification_id: str
    hypothesis_ids: tuple[str, ...]
    requirement_ids: tuple[str, ...]
    steps: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.plan_id, "plan_id")
        _identifier(self.specification_id, "specification_id")
        object.__setattr__(
            self, "hypothesis_ids", _identifiers(
                self.hypothesis_ids, "hypothesis_ids", maximum=MAX_HYPOTHESES, allow_empty=False
            )
        )
        object.__setattr__(
            self, "requirement_ids", _identifiers(
                self.requirement_ids, "requirement_ids", maximum=MAX_REQUIREMENTS
            )
        )
        steps = _tuple(self.steps, "steps", maximum=MAX_PLAN_STEPS)
        if not steps:
            raise TaskIntegrityValidationError("steps cannot be empty")
        for step in steps:
            _text(step, "step", maximum=1_024)
        object.__setattr__(self, "steps", steps)

    def validate_inputs(self, specification: TaskSpecification, belief_state: BeliefState) -> None:
        if (
            self.specification_id != specification.specification_id
            or belief_state.specification_id != self.specification_id
        ):
            raise TaskIntegrityValidationError("plan does not bind task specification and belief")
        if not set(self.hypothesis_ids).issubset(
            {item.hypothesis_id for item in belief_state.hypotheses}
        ):
            raise TaskIntegrityValidationError("plan names an undeclared hypothesis")
        if not set(self.requirement_ids).issubset(set(specification.requirement_ids)):
            raise TaskIntegrityValidationError("plan names an undeclared requirement")

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "specification_id": self.specification_id,
            "hypothesis_ids": list(self.hypothesis_ids),
            "requirement_ids": list(self.requirement_ids),
            "steps": list(self.steps),
            "proposal_only": True,
        }


@dataclass(frozen=True)
class CandidateClaim:
    """A declared candidate claim and its stated reverse dependencies."""

    claim_id: str
    statement: str
    requirement_ids: tuple[str, ...]
    dependency_claim_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.claim_id, "claim_id")
        _text(self.statement, "statement", maximum=2_048)
        object.__setattr__(
            self, "requirement_ids", _identifiers(
                self.requirement_ids, "requirement_ids", maximum=MAX_REQUIREMENTS
            )
        )
        dependencies = _identifiers(
            self.dependency_claim_ids,
            "dependency_claim_ids",
            maximum=MAX_CLAIM_DEPENDENCIES,
        )
        if self.claim_id in dependencies:
            raise TaskIntegrityValidationError("a claim cannot depend on itself")
        object.__setattr__(self, "dependency_claim_ids", dependencies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "requirement_ids": list(self.requirement_ids),
            "dependency_claim_ids": list(self.dependency_claim_ids),
            "declared_only": True,
        }


@dataclass(frozen=True)
class CandidateResult:
    """A proposed result bound to task requirements, not a settled outcome."""

    result_id: str
    specification_id: str
    plan_id: str
    hypothesis_ids: tuple[str, ...]
    addressed_requirement_ids: tuple[str, ...]
    compressed_requirement_ids: tuple[str, ...] = ()
    underweighted_requirement_ids: tuple[str, ...] = ()
    claims: tuple[CandidateClaim, ...] = ()
    conclusion_claim_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value, name in (
            (self.result_id, "result_id"),
            (self.specification_id, "specification_id"),
            (self.plan_id, "plan_id"),
        ):
            _identifier(value, name)
        object.__setattr__(
            self, "hypothesis_ids", _identifiers(
                self.hypothesis_ids, "hypothesis_ids", maximum=MAX_HYPOTHESES, allow_empty=False
            )
        )
        for name in (
            "addressed_requirement_ids",
            "compressed_requirement_ids",
            "underweighted_requirement_ids",
        ):
            object.__setattr__(
                self, name, _identifiers(getattr(self, name), name, maximum=MAX_REQUIREMENTS)
            )
        claims = _tuple(self.claims, "claims", maximum=MAX_RESULT_CLAIMS)
        if any(not isinstance(item, CandidateClaim) for item in claims):
            raise TaskIntegrityValidationError("claims must contain CandidateClaim records")
        if len({item.claim_id for item in claims}) != len(claims):
            raise TaskIntegrityValidationError("claim identities must be unique")
        object.__setattr__(self, "claims", claims)
        object.__setattr__(
            self, "conclusion_claim_ids", _identifiers(
                self.conclusion_claim_ids,
                "conclusion_claim_ids",
                maximum=MAX_RESULT_CLAIMS,
            )
        )

    def validate_inputs(
        self, specification: TaskSpecification, belief_state: BeliefState, plan: TaskPlan
    ) -> None:
        if (
            self.specification_id != specification.specification_id
            or plan.specification_id != self.specification_id
            or self.plan_id != plan.plan_id
        ):
            raise TaskIntegrityValidationError("candidate result does not bind task plan")
        requirement_ids = set(specification.requirement_ids)
        if not (
            set(self.addressed_requirement_ids)
            | set(self.compressed_requirement_ids)
            | set(self.underweighted_requirement_ids)
        ).issubset(requirement_ids):
            raise TaskIntegrityValidationError("candidate result names an undeclared requirement")
        if not set(self.hypothesis_ids).issubset(
            {item.hypothesis_id for item in belief_state.hypotheses}
        ):
            raise TaskIntegrityValidationError("candidate result names an undeclared hypothesis")
        for claim in self.claims:
            if not set(claim.requirement_ids).issubset(requirement_ids):
                raise TaskIntegrityValidationError("claim names an undeclared requirement")

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "specification_id": self.specification_id,
            "plan_id": self.plan_id,
            "hypothesis_ids": list(self.hypothesis_ids),
            "addressed_requirement_ids": list(self.addressed_requirement_ids),
            "compressed_requirement_ids": list(self.compressed_requirement_ids),
            "underweighted_requirement_ids": list(self.underweighted_requirement_ids),
            "claims": [item.to_dict() for item in self.claims],
            "conclusion_claim_ids": list(self.conclusion_claim_ids),
            "proposal_only": True,
            "evidence_grade": "none",
        }


@dataclass(frozen=True)
class InformationLossFinding:
    """One deterministic, non-evidentiary loss or dependency concern."""

    finding_id: str
    kind: LossKind
    requirement_id: str | None
    source_id: str
    detail: str

    def __post_init__(self) -> None:
        _identifier(self.finding_id, "finding_id")
        object.__setattr__(self, "kind", LossKind(self.kind))
        if self.requirement_id is not None:
            _identifier(self.requirement_id, "requirement_id")
        _identifier(self.source_id, "source_id")
        _text(self.detail, "detail", maximum=1_024)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "kind": self.kind.value,
            "requirement_id": self.requirement_id,
            "source_id": self.source_id,
            "detail": self.detail,
            "proposal_only": True,
        }


@dataclass(frozen=True)
class InformationLossReport:
    """Bounded deterministic accounting over task, belief, plan, and result."""

    report_id: str
    specification_id: str
    result_id: str
    findings: tuple[InformationLossFinding, ...]
    truncated_finding_count: int = 0

    def __post_init__(self) -> None:
        _identifier(self.report_id, "report_id")
        _identifier(self.specification_id, "specification_id")
        _identifier(self.result_id, "result_id")
        findings = _tuple(self.findings, "findings", maximum=MAX_LOSS_FINDINGS)
        if any(not isinstance(item, InformationLossFinding) for item in findings):
            raise TaskIntegrityValidationError("loss findings are invalid")
        if len({item.finding_id for item in findings}) != len(findings):
            raise TaskIntegrityValidationError("loss finding identities must be unique")
        if (
            isinstance(self.truncated_finding_count, bool)
            or not isinstance(self.truncated_finding_count, int)
            or self.truncated_finding_count < 0
        ):
            raise TaskIntegrityValidationError(
                "truncated_finding_count must be a non-negative integer"
            )
        object.__setattr__(self, "findings", findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "specification_id": self.specification_id,
            "result_id": self.result_id,
            "findings": [item.to_dict() for item in self.findings],
            "truncated_finding_count": self.truncated_finding_count,
            "proposal_only": True,
            "evidence_grade": "none",
        }


@dataclass(frozen=True)
class ReviewProjection:
    """Minimal role-specific view; it intentionally excludes original raw context."""

    projection_id: str
    role: ReviewRole
    specification_id: str
    result_id: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _identifier(self.projection_id, "projection_id")
        object.__setattr__(self, "role", ReviewRole(self.role))
        _identifier(self.specification_id, "specification_id")
        _identifier(self.result_id, "result_id")
        payload = _mapping(self.payload, "payload")
        forbidden = {
            "original_task",
            "original_text",
            "provenance",
            "evidence",
            "settlement",
            "adaptive_state",
            "execution",
            "authority",
        }
        if forbidden & set(payload):
            raise TaskIntegrityValidationError("review projection includes forbidden context")
        object.__setattr__(self, "payload", payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "projection_id": self.projection_id,
            "role": self.role.value,
            "specification_id": self.specification_id,
            "result_id": self.result_id,
            "payload": _jsonable(self.payload),
            "minimum_context": True,
        }


@dataclass(frozen=True)
class VerificationQuestion:
    """A bounded request for later independent verification, never evidence."""

    question_id: str
    finding_id: str
    target: VerificationTarget
    question: str

    def __post_init__(self) -> None:
        _identifier(self.question_id, "question_id")
        _identifier(self.finding_id, "finding_id")
        object.__setattr__(self, "target", VerificationTarget(self.target))
        _text(self.question, "question", maximum=1_024)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "finding_id": self.finding_id,
            "target": self.target.value,
            "question": self.question,
            "requires_independent_grounding": True,
            "evidence_grade": "none",
        }


@dataclass(frozen=True)
class ReviewFinding:
    """A proposal-only role finding that cannot change the candidate result."""

    finding_id: str
    projection_id: str
    role: ReviewRole
    kind: LossKind
    detail: str
    requirement_ids: tuple[str, ...] = ()
    question_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.finding_id, "finding_id")
        _identifier(self.projection_id, "projection_id")
        object.__setattr__(self, "role", ReviewRole(self.role))
        object.__setattr__(self, "kind", LossKind(self.kind))
        _text(self.detail, "detail", maximum=1_024)
        object.__setattr__(
            self, "requirement_ids", _identifiers(
                self.requirement_ids, "requirement_ids", maximum=MAX_REQUIREMENTS
            )
        )
        object.__setattr__(
            self, "question_ids", _identifiers(
                self.question_ids, "question_ids", maximum=MAX_VERIFICATION_QUESTIONS
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "projection_id": self.projection_id,
            "role": self.role.value,
            "kind": self.kind.value,
            "detail": self.detail,
            "requirement_ids": list(self.requirement_ids),
            "question_ids": list(self.question_ids),
            "proposal_only": True,
            "evidence_grade": "none",
        }


@dataclass(frozen=True)
class TaskIntegrityTrace:
    """Read-only deterministic output of bounded task-integrity inspection."""

    trace_id: str
    original_task: OriginalTask
    specification: TaskSpecification
    belief_state: BeliefState
    plan: TaskPlan
    result: CandidateResult
    loss_report: InformationLossReport
    projections: tuple[ReviewProjection, ...]
    review_findings: tuple[ReviewFinding, ...]
    verification_questions: tuple[VerificationQuestion, ...]

    def __post_init__(self) -> None:
        _identifier(self.trace_id, "trace_id")
        if not all(
            isinstance(item, expected)
            for item, expected in (
                (self.original_task, OriginalTask),
                (self.specification, TaskSpecification),
                (self.belief_state, BeliefState),
                (self.plan, TaskPlan),
                (self.result, CandidateResult),
                (self.loss_report, InformationLossReport),
            )
        ):
            raise TaskIntegrityValidationError("task-integrity trace records are invalid")
        projections = _tuple(self.projections, "projections", maximum=MAX_REVIEW_ROLES)
        findings = _tuple(self.review_findings, "review_findings", maximum=MAX_REVIEW_FINDINGS)
        questions = _tuple(
            self.verification_questions,
            "verification_questions",
            maximum=MAX_VERIFICATION_QUESTIONS,
        )
        if any(not isinstance(item, ReviewProjection) for item in projections):
            raise TaskIntegrityValidationError("review projections are invalid")
        if any(not isinstance(item, ReviewFinding) for item in findings):
            raise TaskIntegrityValidationError("review findings are invalid")
        if any(not isinstance(item, VerificationQuestion) for item in questions):
            raise TaskIntegrityValidationError("verification questions are invalid")
        if len({item.role for item in projections}) != len(projections):
            raise TaskIntegrityValidationError("review roles may be invoked at most once")
        projection_ids = {item.projection_id for item in projections}
        question_ids = {item.question_id for item in questions}
        if any(item.projection_id not in projection_ids for item in findings):
            raise TaskIntegrityValidationError("review finding does not bind an invoked role")
        if any(
            not set(item.question_ids).issubset(question_ids) for item in findings
        ):
            raise TaskIntegrityValidationError("review finding names an unknown question")
        finding_ids = {item.finding_id for item in findings}
        if any(item.finding_id not in finding_ids for item in questions):
            raise TaskIntegrityValidationError("verification question lacks a review finding")
        object.__setattr__(self, "projections", projections)
        object.__setattr__(self, "review_findings", findings)
        object.__setattr__(self, "verification_questions", questions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "original_task": self.original_task.to_dict(),
            "specification": self.specification.to_dict(),
            "belief_state": self.belief_state.to_dict(),
            "plan": self.plan.to_dict(),
            "result": self.result.to_dict(),
            "loss_report": self.loss_report.to_dict(),
            "projections": [item.to_dict() for item in self.projections],
            "review_findings": [item.to_dict() for item in self.review_findings],
            "verification_questions": [item.to_dict() for item in self.verification_questions],
            "read_only": True,
            "creates_evidence": False,
            "changes_adaptive_state": False,
        }


def account_information_loss(
    specification: TaskSpecification,
    belief_state: BeliefState,
    plan: TaskPlan,
    result: CandidateResult,
    *,
    report_id: str | None = None,
) -> InformationLossReport:
    """Return deterministic declared loss findings without repairing any input."""

    belief_state.validate_specification(specification)
    plan.validate_inputs(specification, belief_state)
    result.validate_inputs(specification, belief_state, plan)
    findings: list[InformationLossFinding] = []
    hypothesis_requirements = {
        item for hypothesis in belief_state.hypotheses for item in hypothesis.requirement_ids
    }
    planned = set(plan.requirement_ids)
    addressed = set(result.addressed_requirement_ids)
    for requirement in specification.requirements:
        requirement_id = requirement.requirement_id
        if requirement_id not in hypothesis_requirements:
            findings.append(
                InformationLossFinding(
                    f"{result.result_id}-belief-omitted-{requirement_id}",
                    LossKind.OMITTED,
                    requirement_id,
                    belief_state.belief_state_id,
                    "requirement is absent from every candidate interpretation",
                )
            )
        if requirement_id not in planned:
            findings.append(
                InformationLossFinding(
                    f"{result.result_id}-plan-omitted-{requirement_id}",
                    LossKind.OMITTED,
                    requirement_id,
                    plan.plan_id,
                    "requirement is absent from the declared plan",
                )
            )
        if requirement_id not in addressed:
            findings.append(
                InformationLossFinding(
                    f"{result.result_id}-result-omitted-{requirement_id}",
                    LossKind.OMITTED,
                    requirement_id,
                    result.result_id,
                    "requirement is absent from the candidate result",
                )
            )
        if requirement_id in result.compressed_requirement_ids:
            findings.append(
                InformationLossFinding(
                    f"{result.result_id}-compressed-{requirement_id}",
                    LossKind.COMPRESSED,
                    requirement_id,
                    result.result_id,
                    "candidate result declares this requirement compressed",
                )
            )
        if requirement_id in result.underweighted_requirement_ids:
            findings.append(
                InformationLossFinding(
                    f"{result.result_id}-underweighted-{requirement_id}",
                    LossKind.UNDERWEIGHTED,
                    requirement_id,
                    result.result_id,
                    "candidate result declares this requirement underweighted",
                )
            )
    claim_ids = {item.claim_id for item in result.claims}
    for claim in result.claims:
        for dependency_id in claim.dependency_claim_ids:
            if dependency_id not in claim_ids:
                findings.append(
                    InformationLossFinding(
                        f"{result.result_id}-dependency-{claim.claim_id}-{dependency_id}",
                        LossKind.UNSUPPORTED_DEPENDENCY,
                        None,
                        claim.claim_id,
                        "declared claim dependency is absent from the candidate result",
                    )
                )
    for claim_id in result.conclusion_claim_ids:
        if claim_id not in claim_ids:
            findings.append(
                InformationLossFinding(
                    f"{result.result_id}-conclusion-{claim_id}",
                    LossKind.UNSUPPORTED_DEPENDENCY,
                    None,
                    result.result_id,
                    "declared conclusion does not resolve to a candidate claim",
                )
            )
    truncated_finding_count = max(0, len(findings) - MAX_LOSS_FINDINGS)
    return InformationLossReport(
        report_id or f"{result.result_id}-loss-report",
        specification.specification_id,
        result.result_id,
        tuple(findings[:MAX_LOSS_FINDINGS]),
        truncated_finding_count,
    )


def project_review(
    role: ReviewRole,
    specification: TaskSpecification,
    belief_state: BeliefState,
    result: CandidateResult,
) -> ReviewProjection:
    """Create one narrow role projection without raw task or runtime context."""

    role = ReviewRole(role)
    belief_state.validate_specification(specification)
    if result.specification_id != specification.specification_id:
        raise TaskIntegrityValidationError("review result does not bind specification")
    if role is ReviewRole.ANGEL:
        payload: Mapping[str, Any] = {
            "requirements": tuple(
                (item.requirement_id, item.kind.value, item.importance)
                for item in specification.requirements
            ),
            "addressed_requirement_ids": result.addressed_requirement_ids,
        }
    elif role is ReviewRole.NEMESIS:
        payload = {
            "hypotheses": tuple(
                (
                    item.hypothesis_id,
                    item.status.value,
                    item.requirement_ids,
                    item.contradiction_ids,
                    item.unresolved_dependency_ids,
                )
                for item in belief_state.hypotheses
            ),
        }
    else:
        payload = {
            "claims": tuple(
                (item.claim_id, item.requirement_ids, item.dependency_claim_ids)
                for item in result.claims
            ),
            "conclusion_claim_ids": result.conclusion_claim_ids,
        }
    return ReviewProjection(
        f"{result.result_id}-{role.value}-projection",
        role,
        specification.specification_id,
        result.result_id,
        payload,
    )


def replay_task_integrity(
    original_task: OriginalTask,
    specification: TaskSpecification,
    belief_state: BeliefState,
    plan: TaskPlan,
    result: CandidateResult,
    *,
    roles: tuple[ReviewRole, ...] = (
        ReviewRole.ANGEL,
        ReviewRole.NEMESIS,
        ReviewRole.ANTIMETABOLE,
    ),
    trace_id: str | None = None,
) -> TaskIntegrityTrace:
    """Inspect fixed immutable input deterministically and return proposal-only output."""

    specification.validate_original_task(original_task)
    selected_roles = _tuple(roles, "roles", maximum=MAX_REVIEW_ROLES)
    selected_roles = tuple(ReviewRole(role) for role in selected_roles)
    if len(set(selected_roles)) != len(selected_roles):
        raise TaskIntegrityValidationError("review roles cannot repeat")
    loss_report = account_information_loss(specification, belief_state, plan, result)
    projections = tuple(
        project_review(role, specification, belief_state, result) for role in selected_roles
    )
    projection_by_role = {item.role: item for item in projections}
    questions: list[VerificationQuestion] = []
    findings: list[ReviewFinding] = []
    for loss in loss_report.findings:
        if loss.kind in {
            LossKind.OMITTED,
            LossKind.COMPRESSED,
            LossKind.UNDERWEIGHTED,
        } and ReviewRole.ANGEL in projection_by_role:
            finding_id = f"{loss.finding_id}-angel"
            question_id = f"{finding_id}-question"
            questions.append(
                VerificationQuestion(
                    question_id,
                    finding_id,
                    VerificationTarget.RETRIEVAL,
                    "Independently verify whether the preserved requirement is fully covered.",
                )
            )
            findings.append(
                ReviewFinding(
                    finding_id,
                    projection_by_role[ReviewRole.ANGEL].projection_id,
                    ReviewRole.ANGEL,
                    loss.kind,
                    loss.detail,
                    (loss.requirement_id,) if loss.requirement_id else (),
                    (question_id,),
                )
            )
        if loss.kind is LossKind.UNSUPPORTED_DEPENDENCY and ReviewRole.ANTIMETABOLE in projection_by_role:
            finding_id = f"{loss.finding_id}-antimetabole"
            question_id = f"{finding_id}-question"
            questions.append(
                VerificationQuestion(
                    question_id,
                    finding_id,
                    VerificationTarget.TOOL,
                    "Independently check whether the conclusion dependency is present and supported.",
                )
            )
            findings.append(
                ReviewFinding(
                    finding_id,
                    projection_by_role[ReviewRole.ANTIMETABOLE].projection_id,
                    ReviewRole.ANTIMETABOLE,
                    loss.kind,
                    loss.detail,
                    (),
                    (question_id,),
                )
            )
    if ReviewRole.NEMESIS in projection_by_role:
        for hypothesis in belief_state.hypotheses:
            if hypothesis.contradiction_ids or hypothesis.unresolved_dependency_ids:
                finding_id = f"{result.result_id}-nemesis-{hypothesis.hypothesis_id}"
                question_id = f"{finding_id}-question"
                questions.append(
                    VerificationQuestion(
                        question_id,
                        finding_id,
                        VerificationTarget.EXECUTION,
                        "Independently test the competing interpretation and its unresolved dependency.",
                    )
                )
                findings.append(
                    ReviewFinding(
                        finding_id,
                        projection_by_role[ReviewRole.NEMESIS].projection_id,
                        ReviewRole.NEMESIS,
                        LossKind.UNSUPPORTED_DEPENDENCY,
                        "hypothesis has declared contradiction or unresolved dependency",
                        hypothesis.requirement_ids,
                        (question_id,),
                    )
                )
    return TaskIntegrityTrace(
        trace_id or f"{result.result_id}-integrity-trace",
        original_task,
        specification,
        belief_state,
        plan,
        result,
        loss_report,
        projections,
        tuple(findings),
        tuple(questions),
    )


__all__ = [
    "BeliefState",
    "BeliefStatus",
    "CandidateClaim",
    "CandidateResult",
    "InformationLossFinding",
    "InformationLossReport",
    "LossKind",
    "MAX_CLAIM_DEPENDENCIES",
    "MAX_HYPOTHESES",
    "MAX_LOSS_FINDINGS",
    "MAX_PLAN_STEPS",
    "MAX_REQUIREMENTS",
    "MAX_RESULT_CLAIMS",
    "MAX_REVIEW_FINDINGS",
    "MAX_REVIEW_ROLES",
    "MAX_TASK_TEXT",
    "MAX_VERIFICATION_QUESTIONS",
    "OriginalTask",
    "RequirementKind",
    "ReviewFinding",
    "ReviewProjection",
    "ReviewRole",
    "TaskHypothesis",
    "TaskIntegrityTrace",
    "TaskIntegrityValidationError",
    "TaskPlan",
    "TaskRequirement",
    "TaskSpecification",
    "VerificationQuestion",
    "VerificationTarget",
    "account_information_loss",
    "project_review",
    "replay_task_integrity",
]