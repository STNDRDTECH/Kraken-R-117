"""Domain-agnostic Kraken-R constitutional contract vocabulary.

These immutable records describe a future shared lifecycle. They deliberately
have no persistence, event bus, executor, LLM, or legacy ROGAL dependency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional


class ContractValidationError(ValueError):
    """Raised when a constitutional contract is malformed."""


class EvidenceGrade(str, Enum):
    """Evidence strength from assertion to repeated ground truth."""

    NONE = "none"
    DECLARED = "declared"
    IMPLEMENTED = "implemented"
    WIRED_UNVERIFIED = "wired_unverified"
    OPERATIONAL = "operational"
    GROUNDED = "grounded"


class Authority(str, Enum):
    """Who may treat a contract as authoritative."""

    NONE = "none"
    LEGACY_REFERENCE = "legacy_reference"
    KRAKEN_CANDIDATE = "kraken_candidate"
    DOMAIN_ADAPTER = "domain_adapter"
    DOCUMENTATION_ONLY = "documentation_only"


def _text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field_name} must be a non-empty string")
    return value


def _identifier(value: str, field_name: str) -> str:
    value = _text(value, field_name)
    if any(char.isspace() for char in value):
        raise ContractValidationError(f"{field_name} must not contain whitespace")
    return value


def _freeze_value(value: Any) -> Any:
    """Recursively freeze containers supplied to a constitutional record."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _frozen_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field_name} must be a mapping")
    return _freeze_value(value)


def _frozen_tuple(value: tuple[Any, ...], field_name: str) -> tuple[Any, ...]:
    if not isinstance(value, (tuple, list)):
        raise ContractValidationError(f"{field_name} must be a tuple or list")
    return tuple(_freeze_value(item) for item in value)


def _enum_value(value: Any, enum_type: type[Enum], field_name: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ContractValidationError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            contract_field.name: _jsonable(getattr(value, contract_field.name))
            for contract_field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class _Contract:
    """Shared serialization behavior for all constitutional records."""

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class Objective(_Contract):
    """A domain-agnostic statement of what should change or be learned."""

    objective_id: str
    description: str
    domain: str = "domain_agnostic"
    success_criteria: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.objective_id, "objective_id")
        _text(self.description, "description")
        _text(self.domain, "domain")
        object.__setattr__(
            self, "success_criteria", _frozen_tuple(self.success_criteria, "success_criteria")
        )
        object.__setattr__(self, "constraints", _frozen_tuple(self.constraints, "constraints"))
        object.__setattr__(self, "provenance", _frozen_mapping(self.provenance, "provenance"))


@dataclass(frozen=True)
class TaskState(_Contract):
    """Versioned state owned by one future objective execution."""

    state_id: str
    objective_id: str
    version: int
    phase: str
    values: Mapping[str, Any] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    authority: Authority = Authority.KRAKEN_CANDIDATE

    def __post_init__(self) -> None:
        _identifier(self.state_id, "state_id")
        _identifier(self.objective_id, "objective_id")
        _text(self.phase, "phase")
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version < 1
        ):
            raise ContractValidationError("version must be an integer at least 1")
        object.__setattr__(self, "values", _frozen_mapping(self.values, "values"))
        object.__setattr__(
            self, "evidence_ids", _frozen_tuple(self.evidence_ids, "evidence_ids")
        )
        object.__setattr__(
            self, "authority", _enum_value(self.authority, Authority, "authority")
        )


# State is an alias, not a second state authority.
State = TaskState


@dataclass(frozen=True)
class Hypothesis(_Contract):
    """A falsifiable explanation or candidate route."""

    hypothesis_id: str
    objective_id: str
    statement: str
    confidence: float = 0.0
    basis_evidence_ids: tuple[str, ...] = ()
    status: str = "open"

    def __post_init__(self) -> None:
        _identifier(self.hypothesis_id, "hypothesis_id")
        _identifier(self.objective_id, "objective_id")
        _text(self.statement, "statement")
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractValidationError("confidence must be between 0 and 1")
        _text(self.status, "status")
        object.__setattr__(
            self,
            "basis_evidence_ids",
            _frozen_tuple(self.basis_evidence_ids, "basis_evidence_ids"),
        )


@dataclass(frozen=True)
class Plan(_Contract):
    """A proposed sequence or graph of actions with safeguards."""

    plan_id: str
    objective_id: str
    hypothesis_ids: tuple[str, ...] = ()
    action_ids: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    rollback_plan: Optional[str] = None

    def __post_init__(self) -> None:
        _identifier(self.plan_id, "plan_id")
        _identifier(self.objective_id, "objective_id")
        if self.rollback_plan is not None:
            _text(self.rollback_plan, "rollback_plan")
        object.__setattr__(
            self, "hypothesis_ids", _frozen_tuple(self.hypothesis_ids, "hypothesis_ids")
        )
        object.__setattr__(self, "action_ids", _frozen_tuple(self.action_ids, "action_ids"))
        object.__setattr__(
            self, "preconditions", _frozen_tuple(self.preconditions, "preconditions")
        )


@dataclass(frozen=True)
class Signal(_Contract):
    """A provenance-bearing message between future organs."""

    signal_id: str
    topic: str
    correlation_id: str
    producer: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    evidence_grade: EvidenceGrade = EvidenceGrade.DECLARED
    authority: Authority = Authority.KRAKEN_CANDIDATE
    task_state_id: Optional[str] = None
    task_state_version: Optional[int] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    ttl: Optional[int] = None
    created_tick: int = 0
    source: Optional[str] = None
    cause: Optional[str] = None
    priority: int = 0

    def __post_init__(self) -> None:
        _identifier(self.signal_id, "signal_id")
        _identifier(self.correlation_id, "correlation_id")
        _text(self.topic, "topic")
        _text(self.producer, "producer")
        object.__setattr__(self, "payload", _frozen_mapping(self.payload, "payload"))
        object.__setattr__(
            self,
            "evidence_grade",
            _enum_value(self.evidence_grade, EvidenceGrade, "evidence_grade"),
        )
        object.__setattr__(
            self, "authority", _enum_value(self.authority, Authority, "authority")
        )
        if self.task_state_id is not None:
            _identifier(self.task_state_id, "task_state_id")
        if self.task_state_version is not None:
            if self.task_state_id is None:
                raise ContractValidationError(
                    "task_state_version requires task_state_id"
                )
            if (
                isinstance(self.task_state_version, bool)
                or not isinstance(self.task_state_version, int)
                or self.task_state_version < 1
            ):
                raise ContractValidationError(
                    "task_state_version must be a positive integer"
                )
        if self.ttl is not None and (
            isinstance(self.ttl, bool)
            or not isinstance(self.ttl, int)
            or self.ttl < 0
        ):
            raise ContractValidationError("ttl must be a non-negative integer")
        if (
            isinstance(self.created_tick, bool)
            or not isinstance(self.created_tick, int)
            or self.created_tick < 0
        ):
            raise ContractValidationError("created_tick must be a non-negative integer")
        if self.source is not None:
            _identifier(self.source, "source")
        if self.cause is not None:
            _identifier(self.cause, "cause")
        if (
            isinstance(self.priority, bool)
            or not isinstance(self.priority, int)
            or not -100 <= self.priority <= 100
        ):
            raise ContractValidationError(
                "priority must be an integer from -100 through 100"
            )
        object.__setattr__(
            self, "provenance", _frozen_mapping(self.provenance, "provenance")
        )

    def dedup_key(self) -> str:
        """Return a stable identity for bounded delivery deduplication."""

        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


# Event is a vocabulary alias, not a second event bus.
Event = Signal


@dataclass(frozen=True)
class Action(_Contract):
    """A bounded request to affect a target or produce an artifact."""

    action_id: str
    objective_id: str
    operation: str
    target: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    preconditions: tuple[str, ...] = ()
    requested_by: str = "kraken_r_candidate"
    authority: Authority = Authority.KRAKEN_CANDIDATE

    def __post_init__(self) -> None:
        _identifier(self.action_id, "action_id")
        _identifier(self.objective_id, "objective_id")
        _text(self.operation, "operation")
        _text(self.target, "target")
        _text(self.requested_by, "requested_by")
        object.__setattr__(
            self, "parameters", _frozen_mapping(self.parameters, "parameters")
        )
        object.__setattr__(
            self, "preconditions", _frozen_tuple(self.preconditions, "preconditions")
        )
        object.__setattr__(
            self, "authority", _enum_value(self.authority, Authority, "authority")
        )


@dataclass(frozen=True)
class ExecutionResult(_Contract):
    """Observed facts returned by a future adapter or execution boundary."""

    execution_id: str
    action_id: str
    status: str
    artifact_ref: Optional[str] = None
    exit_code: Optional[int] = None
    observations: Mapping[str, Any] = field(default_factory=dict)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def __post_init__(self) -> None:
        _identifier(self.execution_id, "execution_id")
        _identifier(self.action_id, "action_id")
        _text(self.status, "status")
        if self.exit_code is not None and (
            isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int)
        ):
            raise ContractValidationError("exit_code must be an integer when present")
        object.__setattr__(
            self, "observations", _frozen_mapping(self.observations, "observations")
        )


@dataclass(frozen=True)
class Evidence(_Contract):
    """An auditable observation that can advance an evidence grade."""

    evidence_id: str
    subject_id: str
    grade: EvidenceGrade
    source: str
    observations: Mapping[str, Any] = field(default_factory=dict)
    execution_id: Optional[str] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.evidence_id, "evidence_id")
        _identifier(self.subject_id, "subject_id")
        _text(self.source, "source")
        object.__setattr__(
            self, "observations", _frozen_mapping(self.observations, "observations")
        )
        object.__setattr__(
            self, "provenance", _frozen_mapping(self.provenance, "provenance")
        )
        object.__setattr__(
            self, "grade", _enum_value(self.grade, EvidenceGrade, "grade")
        )


# GroundTruth is a vocabulary alias, not a second evidence store.
GroundTruth = Evidence


@dataclass(frozen=True)
class Decision(_Contract):
    """A selected or rejected alternative tied to evidence."""

    decision_id: str
    objective_id: str
    outcome: str
    rationale: str
    selected_action_id: Optional[str] = None
    evidence_ids: tuple[str, ...] = ()
    authority: Authority = Authority.KRAKEN_CANDIDATE

    def __post_init__(self) -> None:
        _identifier(self.decision_id, "decision_id")
        _identifier(self.objective_id, "objective_id")
        _text(self.outcome, "outcome")
        _text(self.rationale, "rationale")
        object.__setattr__(
            self, "evidence_ids", _frozen_tuple(self.evidence_ids, "evidence_ids")
        )
        object.__setattr__(
            self, "authority", _enum_value(self.authority, Authority, "authority")
        )


@dataclass(frozen=True)
class Settlement(_Contract):
    """A comparison between a decision prediction and observed outcome."""

    settlement_id: str
    decision_id: str
    prediction: str
    observed_outcome: str
    evidence_ids: tuple[str, ...] = ()
    status: str = "unsettled"

    def __post_init__(self) -> None:
        _identifier(self.settlement_id, "settlement_id")
        _identifier(self.decision_id, "decision_id")
        _text(self.prediction, "prediction")
        _text(self.observed_outcome, "observed_outcome")
        _text(self.status, "status")
        object.__setattr__(
            self, "evidence_ids", _frozen_tuple(self.evidence_ids, "evidence_ids")
        )


@dataclass(frozen=True)
class LearningUpdate(_Contract):
    """A proposed change to a capability, memory, route, or constraint."""

    update_id: str
    settlement_id: str
    target_kind: str
    target_id: str
    change: Mapping[str, Any] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    disposition: str = "proposed"

    def __post_init__(self) -> None:
        _identifier(self.update_id, "update_id")
        _identifier(self.settlement_id, "settlement_id")
        _text(self.target_kind, "target_kind")
        _identifier(self.target_id, "target_id")
        _text(self.disposition, "disposition")
        object.__setattr__(self, "change", _frozen_mapping(self.change, "change"))
        object.__setattr__(
            self, "evidence_ids", _frozen_tuple(self.evidence_ids, "evidence_ids")
        )


@dataclass(frozen=True)
class Capability(_Contract):
    """A reusable ability with explicit authority and evidence."""

    capability_id: str
    name: str
    authority: Authority = Authority.KRAKEN_CANDIDATE
    evidence_grade: EvidenceGrade = EvidenceGrade.NONE
    evidence_ids: tuple[str, ...] = ()
    status: str = "candidate"

    def __post_init__(self) -> None:
        _identifier(self.capability_id, "capability_id")
        _text(self.name, "name")
        _text(self.status, "status")
        object.__setattr__(
            self, "evidence_ids", _frozen_tuple(self.evidence_ids, "evidence_ids")
        )
        object.__setattr__(
            self, "authority", _enum_value(self.authority, Authority, "authority")
        )
        object.__setattr__(
            self,
            "evidence_grade",
            _enum_value(self.evidence_grade, EvidenceGrade, "evidence_grade"),
        )


@dataclass(frozen=True)
class Mutation(_Contract):
    """A governed change proposal, never an implicit write."""

    mutation_id: str
    operation: str
    parent_ids: tuple[str, ...] = ()
    artifact_hash: Optional[str] = None
    authority: Authority = Authority.KRAKEN_CANDIDATE
    status: str = "proposed"
    rollback_ref: Optional[str] = None

    def __post_init__(self) -> None:
        _identifier(self.mutation_id, "mutation_id")
        _text(self.operation, "operation")
        _text(self.status, "status")
        object.__setattr__(
            self, "parent_ids", _frozen_tuple(self.parent_ids, "parent_ids")
        )
        if self.mutation_id in self.parent_ids:
            raise ContractValidationError(
                "mutation cannot list itself among its own parent_ids"
            )
        if len(set(self.parent_ids)) != len(self.parent_ids):
            raise ContractValidationError("mutation parent_ids must not contain duplicates")
        object.__setattr__(
            self, "authority", _enum_value(self.authority, Authority, "authority")
        )


@dataclass(frozen=True)
class Lineage(_Contract):
    """An ancestry record for a capability or mutation."""

    lineage_id: str
    subject_id: str
    parent_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    created_by: str = "kraken_r_candidate"

    def __post_init__(self) -> None:
        _identifier(self.lineage_id, "lineage_id")
        _identifier(self.subject_id, "subject_id")
        _text(self.created_by, "created_by")
        object.__setattr__(
            self, "parent_ids", _frozen_tuple(self.parent_ids, "parent_ids")
        )
        if self.subject_id in self.parent_ids:
            raise ContractValidationError("lineage subject cannot be its own parent")
        if len(set(self.parent_ids)) != len(self.parent_ids):
            raise ContractValidationError("lineage parent_ids must not contain duplicates")
        object.__setattr__(
            self, "evidence_ids", _frozen_tuple(self.evidence_ids, "evidence_ids")
        )


@dataclass(frozen=True)
class Regression(_Contract):
    """A comparison against a known baseline after a proposed change."""

    regression_id: str
    mutation_id: str
    baseline_ref: str
    result: str
    test_refs: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.regression_id, "regression_id")
        _identifier(self.mutation_id, "mutation_id")
        _text(self.baseline_ref, "baseline_ref")
        _text(self.result, "result")
        object.__setattr__(self, "test_refs", _frozen_tuple(self.test_refs, "test_refs"))
        object.__setattr__(
            self, "evidence_ids", _frozen_tuple(self.evidence_ids, "evidence_ids")
        )


def ancestry_cycle_errors(
    records: Iterable[Any], *, id_field: str, parent_field: str
) -> list[str]:
    """Detect circular ancestry across a caller-supplied collection.

    Pure and read-only: each individual record already rejects a
    self-reference and duplicate parents in its own ``__post_init__``, but a
    cycle can only be seen across a whole collection (A parents B, B parents
    A). This performs the same bounded Kahn's-algorithm check used by
    ``ArchitectureRegistry._dependency_cycle_errors`` -- it is not a
    persisted registry of its own; callers pass in whatever bounded
    collection of ``Mutation`` or ``Lineage`` records they already hold and
    nothing here is retained across calls.
    """

    records = tuple(records)
    ids = {getattr(record, id_field) for record in records}
    indegree: dict[str, int] = {record_id: 0 for record_id in ids}
    dependents: dict[str, list[str]] = {record_id: [] for record_id in ids}
    for record in records:
        record_id = getattr(record, id_field)
        for parent_id in getattr(record, parent_field):
            if parent_id in ids:
                indegree[record_id] += 1
                dependents[parent_id].append(record_id)

    queue = deque(record_id for record_id, degree in indegree.items() if degree == 0)
    processed = 0
    while queue:
        record_id = queue.popleft()
        processed += 1
        for dependent in dependents[record_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                queue.append(dependent)
    if processed == len(ids):
        return []
    cycle_nodes = sorted(record_id for record_id, degree in indegree.items() if degree > 0)
    return ["Ancestry cycle detected: " + ", ".join(cycle_nodes)]


__all__ = [
    "Action",
    "ancestry_cycle_errors",
    "Authority",
    "Capability",
    "ContractValidationError",
    "Decision",
    "Event",
    "Evidence",
    "EvidenceGrade",
    "ExecutionResult",
    "GroundTruth",
    "Hypothesis",
    "LearningUpdate",
    "Lineage",
    "Mutation",
    "Objective",
    "Plan",
    "Regression",
    "Settlement",
    "Signal",
    "State",
    "TaskState",
]