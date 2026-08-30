"""Bounded, proof-carrying cognition for the Kraken-R candidate layer.

This module is intentionally an in-memory projection above the frozen 11.1
contracts.  It owns no persistence, executor, provider, event bus, scheduler,
evidence store, settlement reducer, or adaptive writer.  Adaptive state is
read-only advisory input; only declared candidate operations may run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from .task_integrity import OriginalTask


class CognitionValidationError(ValueError):
    """Raised when a candidate cognition contract is malformed."""


MAX_GRAPH_NODES = 32
MAX_REQUIREMENTS = 24
MAX_SUBTASKS = 24
MAX_MATERIALS = 32
MAX_BRANCHES = 8
MAX_CLAIMS = 48
MAX_SOURCE_LINEAGE = 64
MAX_EDGES = 96
MAX_OBLIGATIONS = 128
MAX_BLOCKERS = 128
MAX_EVENTS = 32
MAX_CONTEXT_ITEMS = 32
MAX_OPERATION_COST = 1_000
MAX_PROCESSING_OPERATIONS = 8
MIN_ROUTE_WEIGHT = 0.35
MAX_PAYLOAD_DEPTH = 8
MAX_PAYLOAD_ITEMS = 64
MAX_PAYLOAD_BYTES = 32_768


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(c.isspace() for c in value):
        raise CognitionValidationError(f"{name} must be a non-empty identifier without whitespace")
    return value


def _text(value: str, name: str, maximum: int = 8_192) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CognitionValidationError(f"{name} must be non-empty text")
    if len(value) > maximum:
        raise CognitionValidationError(f"{name} exceeds its bounded length")
    return value


def _bounded(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CognitionValidationError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise CognitionValidationError(f"{name} must be between 0 and 1")
    return round(value, 6)


def _positive_int(value: int, name: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CognitionValidationError(f"{name} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise CognitionValidationError(f"{name} exceeds its bounded limit")
    return value


def _bounded_items(value: Iterable[Any], name: str, maximum: int) -> tuple[Any, ...]:
    try:
        items = tuple(value)
    except TypeError as exc:
        raise CognitionValidationError(f"{name} must be iterable") from exc
    if len(items) > maximum:
        raise CognitionValidationError(f"{name} exceeds its bounded limit")
    return items


def _ids(value: Iterable[str], name: str, maximum: int, *, allow_empty: bool = True) -> tuple[str, ...]:
    items = _bounded_items(value, name, maximum)
    if not allow_empty and not items:
        raise CognitionValidationError(f"{name} cannot be empty")
    result = tuple(_identifier(item, name) for item in items)
    if len(set(result)) != len(result):
        raise CognitionValidationError(f"{name} cannot contain duplicate identities")
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _bounded_payload(value: Any, name: str) -> Any:
    """Reject unbounded or non-JSON callback/provenance structures."""

    item_count = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal item_count
        if depth > MAX_PAYLOAD_DEPTH:
            raise CognitionValidationError(f"{name} exceeds the payload depth limit")
        if item is None or isinstance(item, (bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise CognitionValidationError(f"{name} contains a non-finite number")
            return
        if isinstance(item, str):
            if len(item) > 8_192:
                raise CognitionValidationError(f"{name} contains oversized text")
            return
        if isinstance(item, Mapping):
            item_count += len(item)
            if len(item) > MAX_PAYLOAD_ITEMS or item_count > MAX_PAYLOAD_ITEMS:
                raise CognitionValidationError(f"{name} exceeds the payload item limit")
            for key, child in item.items():
                if not isinstance(key, str) or len(key) > 256:
                    raise CognitionValidationError(f"{name} contains an invalid key")
                visit(child, depth + 1)
            return
        if isinstance(item, (tuple, list)):
            item_count += len(item)
            if len(item) > MAX_PAYLOAD_ITEMS or item_count > MAX_PAYLOAD_ITEMS:
                raise CognitionValidationError(f"{name} exceeds the payload item limit")
            for child in item:
                visit(child, depth + 1)
            return
        raise CognitionValidationError(f"{name} must contain only bounded JSON values")

    visit(value, 0)
    try:
        raw = json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    except (TypeError, ValueError) as exc:
        raise CognitionValidationError(f"{name} is not JSON serializable") from exc
    if len(raw.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise CognitionValidationError(f"{name} exceeds the payload byte limit")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _hash(value: Any) -> str:
    raw = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class MaterialState(str, Enum):
    ACTIVE = "active"
    DEFERRED = "deferred"
    DORMANT = "dormant"
    REACTIVATED = "reactivated"


class SatisfactionState(str, Enum):
    UNSATISFIED = "unsatisfied"
    SATISFIED = "satisfied"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"


class EpistemicStatus(str, Enum):
    PROPOSAL = "proposal"
    SUPPORT = "support"
    CONTRADICTION = "contradiction"
    UNRESOLVED = "unresolved"


class CompetenceOrigin(str, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"


class EdgeType(str, Enum):
    ASSUMPTION = "assumption"
    DEDUCTION = "deduction"
    CAUSATION = "causation"
    CORRELATION = "correlation"
    ANALOGY = "analogy"
    QUANTITATIVE = "quantitative"
    SOURCE_ASSERTION = "source_assertion"
    PREDICTION = "prediction"
    REQUIREMENT = "requirement"
    COUNTERFACTUAL = "counterfactual"


class ObligationKind(str, Enum):
    LOGIC = "logic"
    MISSING_DEPENDENCY = "missing_dependency"
    UNITS_DIMENSIONS = "units_dimensions"
    QUANTITATIVE_VALIDITY = "quantitative_validity"
    CONSERVATION_ACCOUNTING = "conservation_accounting"
    REGIME_APPLICABILITY = "regime_applicability"
    TEMPORAL_ORDERING = "temporal_ordering"
    CONTRADICTION = "contradiction"
    STALE_CORRELATED_SOURCE = "stale_or_correlated_source"
    UNRESOLVED_USER_CONSTRAINT = "unresolved_user_constraint"
    ANALOGY_SCOPE = "analogy_scope"


class ObligationState(str, Enum):
    OPEN = "open"
    SATISFIED = "satisfied"
    BLOCKED = "blocked"


class BlockerKind(str, Enum):
    LOGIC = "logic"
    MISSING_DEPENDENCY = "missing_dependency"
    UNITS_DIMENSIONS = "units_dimensions"
    QUANTITATIVE_VALIDITY = "quantitative_validity"
    CONSERVATION_ACCOUNTING = "conservation_accounting"
    REGIME_APPLICABILITY = "regime_applicability"
    TEMPORAL_ORDERING = "temporal_ordering"
    CONTRADICTION = "contradiction"
    STALE_CORRELATED_SOURCE = "stale_or_correlated_source"
    UNRESOLVED_USER_CONSTRAINT = "unresolved_user_constraint"
    ANALOGY_SCOPE = "analogy_scope"


class ProcessingOperation(str, Enum):
    QUESTION = "question"
    CHECK_LOGIC = "check_logic"
    CHECK_QUANTITATIVE = "check_quantitative"
    RETRIEVE = "retrieve"
    MODEL_PROPOSAL = "model_proposal"
    TOOL_PROPOSAL = "tool_proposal"
    SIMULATION_PROPOSAL = "simulation_proposal"
    ANGEL = "angel"
    NEMESIS = "nemesis"
    ANTIMETABOLE = "antimetabole"


class SemanticJob(str, Enum):
    RECALL = "recall"
    MECHANISM_GENERATION = "mechanism_generation"
    COMPETING_EXPLANATIONS = "competing_explanations"
    CROSS_DOMAIN_CORRESPONDENCE = "cross_domain_correspondence"
    VARIABLE_EQUATION_EXTRACTION = "variable_equation_extraction"
    SOURCE_INTERPRETATION = "source_interpretation"
    FALSIFIER_GENERATION = "falsifier_generation"
    MISSING_INFORMATION_DETECTION = "missing_information_detection"


class ProcessingEventKind(str, Enum):
    DISCOVERED_NEED = "discovered_need"
    ACTIVATION = "activation"
    INHIBITION = "inhibition"
    EDGE_TRAVERSAL = "edge_traversal"
    BRANCH_FORK = "branch_fork"
    BRANCH_MERGE = "branch_merge"
    DEPENDENCY_GAP = "dependency_gap"
    ENVIRONMENT_SPLIT = "environment_split"
    CONTRADICTION = "contradiction"
    RESOURCE_REALLOCATION = "resource_reallocation"
    PREDICTION_MISMATCH = "prediction_mismatch"
    MOTIF_OBSERVATION = "motif_observation"
    OPERATION = "operation"
    NOOP = "causal_noop"


@dataclass(frozen=True)
class ProblemRequirement:
    requirement_id: str
    text: str
    user_constraint: bool = False
    status: SatisfactionState = SatisfactionState.UNRESOLVED

    def __post_init__(self) -> None:
        _identifier(self.requirement_id, "requirement_id")
        _text(self.text, "requirement text", 2_048)
        object.__setattr__(self, "status", SatisfactionState(self.status))

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "text": self.text,
            "user_constraint": self.user_constraint,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class ProblemSubtask:
    subtask_id: str
    question: str
    requirement_ids: tuple[str, ...] = ()
    status: SatisfactionState = SatisfactionState.UNRESOLVED
    branch_id: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.subtask_id, "subtask_id")
        _text(self.question, "subtask question", 2_048)
        object.__setattr__(self, "requirement_ids", _ids(self.requirement_ids, "subtask requirement_ids", MAX_REQUIREMENTS))
        object.__setattr__(self, "status", SatisfactionState(self.status))
        if self.branch_id is not None:
            _identifier(self.branch_id, "branch_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "question": self.question,
            "requirement_ids": list(self.requirement_ids),
            "status": self.status.value,
            "branch_id": self.branch_id,
        }


@dataclass(frozen=True)
class ProblemMaterial:
    material_id: str
    content: str
    kind: str = "declared"
    state: MaterialState = MaterialState.ACTIVE
    source_lineage_ids: tuple[str, ...] = ()
    environment_id: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.material_id, "material_id")
        _text(self.content, "material content", 4_096)
        _identifier(self.kind, "material kind")
        object.__setattr__(self, "state", MaterialState(self.state))
        object.__setattr__(self, "source_lineage_ids", _ids(self.source_lineage_ids, "source_lineage_ids", MAX_CLAIMS))
        if self.environment_id is not None:
            _identifier(self.environment_id, "environment_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "material_id": self.material_id,
            "content": self.content,
            "kind": self.kind,
            "state": self.state.value,
            "source_lineage_ids": list(self.source_lineage_ids),
            "environment_id": self.environment_id,
        }


@dataclass(frozen=True)
class ProblemBranch:
    branch_id: str
    label: str
    environment_id: str
    subtask_ids: tuple[str, ...] = ()
    status: SatisfactionState = SatisfactionState.UNRESOLVED
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.branch_id, "branch_id")
        _text(self.label, "branch label", 512)
        _identifier(self.environment_id, "environment_id")
        object.__setattr__(self, "subtask_ids", _ids(self.subtask_ids, "branch subtask_ids", MAX_SUBTASKS))
        object.__setattr__(self, "status", SatisfactionState(self.status))
        object.__setattr__(self, "context", _freeze(_bounded_payload(self.context, "branch context")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "label": self.label,
            "environment_id": self.environment_id,
            "subtask_ids": list(self.subtask_ids),
            "status": self.status.value,
            "context": _jsonable(self.context),
        }


@dataclass(frozen=True)
class ProblemGraph:
    """Lossless bounded task projection; no field is silently discarded."""

    problem_id: str
    original_task: Any
    central_question: str
    requirements: tuple[ProblemRequirement, ...]
    subtasks: tuple[ProblemSubtask, ...] = ()
    materials: tuple[ProblemMaterial, ...] = ()
    branches: tuple[ProblemBranch, ...] = ()
    environments: tuple[str, ...] = ()
    satisfaction: Mapping[str, SatisfactionState] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    source_lineage: tuple[SourceLineage, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.problem_id, "problem_id")
        if not hasattr(self.original_task, "task_hash") or not hasattr(self.original_task, "to_dict"):
            raise CognitionValidationError("original_task must be an OriginalTask-compatible record")
        _text(self.central_question, "central_question", 2_048)
        requirements = _bounded_items(self.requirements, "requirements", MAX_REQUIREMENTS)
        subtasks = _bounded_items(self.subtasks, "subtasks", MAX_SUBTASKS)
        materials = _bounded_items(self.materials, "materials", MAX_MATERIALS)
        branches = _bounded_items(self.branches, "branches", MAX_BRANCHES)
        if not requirements or not all(isinstance(item, ProblemRequirement) for item in requirements):
            raise CognitionValidationError("requirements must contain ProblemRequirement records")
        if not all(isinstance(item, ProblemSubtask) for item in subtasks):
            raise CognitionValidationError("subtasks must contain ProblemSubtask records")
        if not all(isinstance(item, ProblemMaterial) for item in materials):
            raise CognitionValidationError("materials must contain ProblemMaterial records")
        if not all(isinstance(item, ProblemBranch) for item in branches):
            raise CognitionValidationError("branches must contain ProblemBranch records")
        for records, name in ((requirements, "requirement"), (subtasks, "subtask"), (materials, "material"), (branches, "branch")):
            ids = [getattr(item, f"{name}_id") for item in records]
            if len(set(ids)) != len(ids):
                raise CognitionValidationError(f"{name} identities must be unique")
        environment_ids = _ids(self.environments, "environments", MAX_BRANCHES)
        if any(branch.environment_id not in environment_ids for branch in branches):
            raise CognitionValidationError("each branch must bind a declared environment")
        if any(
            material.environment_id is not None
            and material.environment_id not in environment_ids
            for material in materials
        ):
            raise CognitionValidationError("each material environment must be declared")
        requirement_ids = {item.requirement_id for item in requirements}
        subtask_ids = {item.subtask_id for item in subtasks}
        branch_ids = {item.branch_id for item in branches}
        if any(set(item.requirement_ids) - requirement_ids for item in subtasks):
            raise CognitionValidationError("subtask references an undeclared requirement")
        if any(item.branch_id is not None and item.branch_id not in branch_ids for item in subtasks):
            raise CognitionValidationError("subtask references an undeclared branch")
        if any(set(item.subtask_ids) - subtask_ids for item in branches):
            raise CognitionValidationError("branch references an undeclared subtask")
        branch_subtasks = {
            branch.branch_id: set(branch.subtask_ids) for branch in branches
        }
        if any(
            item.branch_id is not None
            and item.subtask_id not in branch_subtasks[item.branch_id]
            for item in subtasks
        ):
            raise CognitionValidationError("subtask and branch membership must be reciprocal")
        satisfaction = {
            _identifier(key, "satisfaction key"): SatisfactionState(value)
            for key, value in self.satisfaction.items()
        }
        valid_material_ids = {item.material_id for item in materials}
        if set(satisfaction) - (subtask_ids | requirement_ids | valid_material_ids):
            raise CognitionValidationError("satisfaction names an undeclared item")
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "subtasks", subtasks)
        object.__setattr__(self, "materials", materials)
        object.__setattr__(self, "branches", branches)
        object.__setattr__(self, "environments", environment_ids)
        object.__setattr__(self, "satisfaction", _freeze(satisfaction))
        object.__setattr__(self, "provenance", _freeze(_bounded_payload(self.provenance, "problem provenance")))
        source_lineage = _bounded_items(
            self.source_lineage, "problem source_lineage", MAX_SOURCE_LINEAGE
        )
        if not all(isinstance(item, SourceLineage) for item in source_lineage):
            raise CognitionValidationError("problem source lineage is invalid")
        lineage_ids = {item.lineage_id for item in source_lineage}
        if len(lineage_ids) != len(source_lineage):
            raise CognitionValidationError(
                "problem source lineage identities must be unique"
            )
        if any(
            set(item.parent_lineage_ids) - lineage_ids for item in source_lineage
        ):
            raise CognitionValidationError(
                "problem source lineage parent is undeclared"
            )
        if any(
            set(material.source_lineage_ids) - lineage_ids
            for material in materials
        ):
            raise CognitionValidationError(
                "problem material source lineage is undeclared"
            )
        object.__setattr__(self, "source_lineage", source_lineage)

    @property
    def graph_hash(self) -> str:
        return _hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "problem_id": self.problem_id,
            "original_task": self.original_task.to_dict(),
            "central_question": self.central_question,
            "requirements": [item.to_dict() for item in self.requirements],
            "subtasks": [item.to_dict() for item in self.subtasks],
            "materials": [item.to_dict() for item in self.materials],
            "branches": [item.to_dict() for item in self.branches],
            "environments": list(self.environments),
            "satisfaction": {key: value.value for key, value in self.satisfaction.items()},
            "provenance": _jsonable(self.provenance),
            "source_lineage": [item.to_dict() for item in self.source_lineage],
        }
        if include_hash:
            result["graph_hash"] = self.graph_hash
        return result


@dataclass(frozen=True)
class SourceLineage:
    lineage_id: str
    source_id: str
    source_kind: str
    independent_group: str
    parent_lineage_ids: tuple[str, ...] = ()
    current: bool = True
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, name in ((self.lineage_id, "lineage_id"), (self.source_id, "source_id"), (self.independent_group, "independent_group")):
            _identifier(value, name)
        _identifier(self.source_kind, "source_kind")
        object.__setattr__(self, "parent_lineage_ids", _ids(self.parent_lineage_ids, "parent_lineage_ids", MAX_CLAIMS))
        object.__setattr__(self, "provenance", _freeze(_bounded_payload(self.provenance, "source provenance")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "lineage_id": self.lineage_id,
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "independent_group": self.independent_group,
            "parent_lineage_ids": list(self.parent_lineage_ids),
            "current": self.current,
            "provenance": _jsonable(self.provenance),
        }


@dataclass(frozen=True)
class VerificationPressure:
    stakes: float = 0.0
    volatility: float = 0.0
    locality: float = 0.0
    precision: float = 0.0
    novelty: float = 0.0
    contestability: float = 0.0
    actionability: float = 0.0

    def __post_init__(self) -> None:
        for name in ("stakes", "volatility", "locality", "precision", "novelty", "contestability", "actionability"):
            object.__setattr__(self, name, _bounded(getattr(self, name), name))

    @property
    def score(self) -> float:
        return round(sum(getattr(self, name) for name in (
            "stakes", "volatility", "locality", "precision", "novelty", "contestability", "actionability"
        )) / 7.0, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stakes": self.stakes,
            "volatility": self.volatility,
            "locality": self.locality,
            "precision": self.precision,
            "novelty": self.novelty,
            "contestability": self.contestability,
            "actionability": self.actionability,
            "score": self.score,
        }


@dataclass(frozen=True)
class FalsificationCondition:
    condition_id: str
    statement: str
    discriminator: str

    def __post_init__(self) -> None:
        _identifier(self.condition_id, "condition_id")
        _text(self.statement, "falsification statement", 2_048)
        _text(self.discriminator, "falsification discriminator", 2_048)

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "statement": self.statement,
            "discriminator": self.discriminator,
        }


@dataclass(frozen=True)
class EpistemicClaim:
    """A proposal-bearing claim; it can never itself become Evidence."""

    claim_id: str
    problem_id: str
    statement: str
    status: EpistemicStatus = EpistemicStatus.PROPOSAL
    assumption_ids: tuple[str, ...] = ()
    support_claim_ids: tuple[str, ...] = ()
    contradiction_claim_ids: tuple[str, ...] = ()
    unresolved_ids: tuple[str, ...] = ()
    source_lineage: tuple[SourceLineage, ...] = ()
    use_site: str = "analysis"
    verification_pressure: VerificationPressure = field(default_factory=VerificationPressure)
    falsification_conditions: tuple[FalsificationCondition, ...] = ()
    competence_origin: CompetenceOrigin = CompetenceOrigin.INTERNAL
    evidence_ids: tuple[str, ...] = ()
    declared_only: bool = True

    def __post_init__(self) -> None:
        _identifier(self.claim_id, "claim_id")
        _identifier(self.problem_id, "problem_id")
        _text(self.statement, "claim statement", 4_096)
        object.__setattr__(self, "status", EpistemicStatus(self.status))
        object.__setattr__(self, "assumption_ids", _ids(self.assumption_ids, "assumption_ids", MAX_OBLIGATIONS))
        object.__setattr__(self, "support_claim_ids", _ids(self.support_claim_ids, "support_claim_ids", MAX_CLAIMS))
        object.__setattr__(self, "contradiction_claim_ids", _ids(self.contradiction_claim_ids, "contradiction_claim_ids", MAX_CLAIMS))
        object.__setattr__(self, "unresolved_ids", _ids(self.unresolved_ids, "unresolved_ids", MAX_OBLIGATIONS))
        lineages = _bounded_items(self.source_lineage, "source_lineage", MAX_CLAIMS)
        if not all(isinstance(item, SourceLineage) for item in lineages):
            raise CognitionValidationError("source_lineage must contain SourceLineage records")
        object.__setattr__(self, "source_lineage", lineages)
        _identifier(self.use_site, "use_site")
        conditions = _bounded_items(self.falsification_conditions, "falsification_conditions", MAX_OBLIGATIONS)
        if not all(isinstance(item, FalsificationCondition) for item in conditions):
            raise CognitionValidationError("falsification conditions are invalid")
        object.__setattr__(self, "falsification_conditions", conditions)
        object.__setattr__(self, "competence_origin", CompetenceOrigin(self.competence_origin))
        object.__setattr__(self, "evidence_ids", _ids(self.evidence_ids, "evidence_ids", MAX_CLAIMS))
        if not self.declared_only:
            raise CognitionValidationError("cognition claims must remain declared-only")

    def to_evidence(self) -> Any:
        raise CognitionValidationError("epistemic claims cannot create evidence")

    def to_adaptive_credit(self) -> Any:
        raise CognitionValidationError("epistemic claims cannot create adaptive credit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "problem_id": self.problem_id,
            "statement": self.statement,
            "status": self.status.value,
            "assumption_ids": list(self.assumption_ids),
            "support_claim_ids": list(self.support_claim_ids),
            "contradiction_claim_ids": list(self.contradiction_claim_ids),
            "unresolved_ids": list(self.unresolved_ids),
            "source_lineage": [item.to_dict() for item in self.source_lineage],
            "use_site": self.use_site,
            "verification_pressure": self.verification_pressure.to_dict(),
            "falsification_conditions": [item.to_dict() for item in self.falsification_conditions],
            "competence_origin": self.competence_origin.value,
            "evidence_ids": list(self.evidence_ids),
            "declared_only": True,
        }


_EDGE_OBLIGATIONS: Mapping[EdgeType, tuple[ObligationKind, ...]] = {
    EdgeType.ASSUMPTION: (ObligationKind.LOGIC, ObligationKind.CONTRADICTION),
    EdgeType.DEDUCTION: (ObligationKind.LOGIC, ObligationKind.MISSING_DEPENDENCY),
    EdgeType.CAUSATION: (
        ObligationKind.LOGIC, ObligationKind.QUANTITATIVE_VALIDITY,
        ObligationKind.REGIME_APPLICABILITY, ObligationKind.TEMPORAL_ORDERING,
        ObligationKind.CONTRADICTION,
    ),
    EdgeType.CORRELATION: (ObligationKind.LOGIC, ObligationKind.CONTRADICTION),
    EdgeType.ANALOGY: (ObligationKind.LOGIC, ObligationKind.ANALOGY_SCOPE),
    EdgeType.QUANTITATIVE: (
        ObligationKind.LOGIC, ObligationKind.UNITS_DIMENSIONS,
        ObligationKind.QUANTITATIVE_VALIDITY, ObligationKind.CONSERVATION_ACCOUNTING,
    ),
    EdgeType.SOURCE_ASSERTION: (ObligationKind.LOGIC, ObligationKind.STALE_CORRELATED_SOURCE),
    EdgeType.PREDICTION: (
        ObligationKind.LOGIC, ObligationKind.QUANTITATIVE_VALIDITY,
        ObligationKind.TEMPORAL_ORDERING, ObligationKind.CONTRADICTION,
    ),
    EdgeType.REQUIREMENT: (ObligationKind.LOGIC, ObligationKind.UNRESOLVED_USER_CONSTRAINT),
    EdgeType.COUNTERFACTUAL: (
        ObligationKind.LOGIC, ObligationKind.REGIME_APPLICABILITY,
        ObligationKind.TEMPORAL_ORDERING, ObligationKind.CONTRADICTION,
    ),
}


@dataclass(frozen=True)
class VerificationObligation:
    obligation_id: str
    problem_id: str
    subject_id: str
    kind: ObligationKind
    description: str
    state: ObligationState = ObligationState.OPEN
    blocker_ids: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, name in ((self.obligation_id, "obligation_id"), (self.problem_id, "problem_id"), (self.subject_id, "subject_id")):
            _identifier(value, name)
        object.__setattr__(self, "kind", ObligationKind(self.kind))
        _text(self.description, "obligation description", 2_048)
        object.__setattr__(self, "state", ObligationState(self.state))
        object.__setattr__(self, "blocker_ids", _ids(self.blocker_ids, "blocker_ids", MAX_BLOCKERS))
        object.__setattr__(self, "provenance", _freeze(_bounded_payload(self.provenance, "obligation provenance")))

    @property
    def blocking(self) -> bool:
        return self.state is not ObligationState.SATISFIED or bool(self.blocker_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "problem_id": self.problem_id,
            "subject_id": self.subject_id,
            "kind": self.kind.value,
            "description": self.description,
            "state": self.state.value,
            "blocker_ids": list(self.blocker_ids),
            "provenance": _jsonable(self.provenance),
        }


@dataclass(frozen=True)
class Blocker:
    blocker_id: str
    obligation_id: str
    kind: BlockerKind
    reason: str
    source_ids: tuple[str, ...] = ()
    hard: bool = True

    def __post_init__(self) -> None:
        _identifier(self.blocker_id, "blocker_id")
        _identifier(self.obligation_id, "obligation_id")
        object.__setattr__(self, "kind", BlockerKind(self.kind))
        _text(self.reason, "blocker reason", 2_048)
        object.__setattr__(self, "source_ids", _ids(self.source_ids, "blocker source_ids", MAX_CLAIMS))

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocker_id": self.blocker_id,
            "obligation_id": self.obligation_id,
            "kind": self.kind.value,
            "reason": self.reason,
            "source_ids": list(self.source_ids),
            "hard": self.hard,
        }


@dataclass(frozen=True)
class TypedReasoningEdge:
    edge_id: str
    problem_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    obligations: tuple[VerificationObligation, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, name in ((self.edge_id, "edge_id"), (self.problem_id, "problem_id"), (self.source_id, "source_id"), (self.target_id, "target_id")):
            _identifier(value, name)
        if self.source_id == self.target_id:
            raise CognitionValidationError("reasoning edges cannot be self-loops")
        object.__setattr__(self, "edge_type", EdgeType(self.edge_type))
        obligations = _bounded_items(self.obligations, "edge obligations", MAX_OBLIGATIONS)
        if not all(isinstance(item, VerificationObligation) for item in obligations):
            raise CognitionValidationError("edge obligations are invalid")
        expected = _EDGE_OBLIGATIONS[self.edge_type]
        actual = tuple(item.kind for item in obligations)
        if set(actual) != set(expected) or len(actual) != len(expected):
            raise CognitionValidationError(
                f"{self.edge_type.value} edge must create exactly its declared verification obligations"
            )
        if any(item.problem_id != self.problem_id or item.subject_id != self.edge_id for item in obligations):
            raise CognitionValidationError("edge obligations must bind the edge and problem")
        object.__setattr__(self, "obligations", obligations)
        object.__setattr__(self, "provenance", _freeze(_bounded_payload(self.provenance, "edge provenance")))

    @property
    def required_obligation_kinds(self) -> tuple[ObligationKind, ...]:
        return _EDGE_OBLIGATIONS[self.edge_type]

    @classmethod
    def create(
        cls,
        edge_id: str,
        problem_id: str,
        source_id: str,
        target_id: str,
        edge_type: EdgeType | str,
        *,
        description_prefix: str = "Verify",
        provenance: Mapping[str, Any] | None = None,
    ) -> "TypedReasoningEdge":
        selected = EdgeType(edge_type)
        obligations = tuple(
            VerificationObligation(
                f"{edge_id}-obligation-{index + 1}",
                problem_id,
                edge_id,
                kind,
                f"{description_prefix} {kind.value.replace('_', ' ')} for {selected.value} relationship.",
                provenance={"edge_id": edge_id, "edge_type": selected.value},
            )
            for index, kind in enumerate(_EDGE_OBLIGATIONS[selected])
        )
        return cls(edge_id, problem_id, source_id, target_id, selected, obligations, provenance or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "problem_id": self.problem_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "obligations": [item.to_dict() for item in self.obligations],
            "provenance": _jsonable(self.provenance),
        }


@dataclass(frozen=True)
class SynthesisAssessment:
    allowed: bool
    claim_ids: tuple[str, ...]
    blocking_obligation_ids: tuple[str, ...]
    blocker_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "claim_ids": list(self.claim_ids),
            "blocking_obligation_ids": list(self.blocking_obligation_ids),
            "blocker_ids": list(self.blocker_ids),
            "reason": self.reason,
            "candidate_only": True,
        }


def _validate_proof_graph(
    claims: tuple[EpistemicClaim, ...],
    edges: tuple[TypedReasoningEdge, ...],
    obligations: tuple[VerificationObligation, ...],
    blockers: tuple[Blocker, ...],
    *,
    material_ids: Iterable[str] = (),
    source_lineage: Iterable[SourceLineage] = (),
    allowed_subject_ids: Iterable[str] = (),
) -> None:
    for values, name, identity in (
        (claims, "claim", "claim_id"),
        (edges, "reasoning edge", "edge_id"),
        (obligations, "obligation", "obligation_id"),
        (blockers, "blocker", "blocker_id"),
    ):
        ids = [getattr(item, identity) for item in values]
        if len(set(ids)) != len(ids):
            raise CognitionValidationError(f"{name} identities must be unique")
    problem_ids = {
        item.problem_id for item in claims + edges + obligations
    }
    if len(problem_ids) > 1:
        raise CognitionValidationError("proof records must bind one problem")
    claim_ids = {item.claim_id for item in claims}
    edge_ids = {item.edge_id for item in edges}
    obligation_by_id = {item.obligation_id: item for item in obligations}
    blocker_by_id = {item.blocker_id: item for item in blockers}
    lineages = tuple(source_lineage) or tuple(
        lineage for claim in claims for lineage in claim.source_lineage
    )
    lineage_ids = {item.lineage_id for item in lineages}
    if len(lineage_ids) != len(lineages):
        raise CognitionValidationError("lineage identities must be unique")
    for claim in claims:
        lineage_by_id = {item.lineage_id: item for item in lineages}
        if any(
            lineage_by_id.get(item.lineage_id) != item
            for item in claim.source_lineage
        ):
            raise CognitionValidationError(
                "claim lineage is absent or conflicts with the canonical registry"
            )
        if (
            set(claim.support_claim_ids) - claim_ids
            or set(claim.contradiction_claim_ids) - claim_ids
        ):
            raise CognitionValidationError("claim references an undeclared claim")
        if (
            set(claim.assumption_ids) - set(obligation_by_id)
            or set(claim.unresolved_ids) - set(obligation_by_id)
        ):
            raise CognitionValidationError(
                "claim assumption or unresolved reference is undeclared"
            )
    if any(set(item.parent_lineage_ids) - lineage_ids for item in lineages):
        raise CognitionValidationError("source lineage parent is undeclared")
    for edge in edges:
        if edge.source_id not in claim_ids or edge.target_id not in claim_ids:
            raise CognitionValidationError("reasoning edge endpoint is undeclared")
        for edge_obligation in edge.obligations:
            if obligation_by_id.get(edge_obligation.obligation_id) != edge_obligation:
                raise CognitionValidationError(
                    "edge obligation is absent or conflicts with proof graph"
                )
    valid_subject_ids = claim_ids | edge_ids | set(allowed_subject_ids)
    if any(item.subject_id not in valid_subject_ids for item in obligations):
        raise CognitionValidationError("obligation subject is undeclared")
    source_ids = claim_ids | lineage_ids | set(material_ids)
    for obligation in obligations:
        if set(obligation.blocker_ids) - set(blocker_by_id):
            raise CognitionValidationError("obligation blocker is undeclared")
        if any(
            blocker_by_id[blocker_id].obligation_id != obligation.obligation_id
            for blocker_id in obligation.blocker_ids
        ):
            raise CognitionValidationError("obligation blocker binding is invalid")
    for blocker in blockers:
        obligation = obligation_by_id.get(blocker.obligation_id)
        if obligation is None or blocker.blocker_id not in obligation.blocker_ids:
            raise CognitionValidationError("blocker is not reciprocally bound")
        if set(blocker.source_ids) - source_ids:
            raise CognitionValidationError("blocker source is undeclared")


def assess_synthesis(
    problem: ProblemGraph,
    claims: Iterable[EpistemicClaim],
    edges: Iterable[TypedReasoningEdge],
    obligations: Iterable[VerificationObligation],
    blockers: Iterable[Blocker] = (),
) -> SynthesisAssessment:
    if not isinstance(problem, ProblemGraph):
        raise CognitionValidationError("synthesis requires a bound ProblemGraph")
    claims = _bounded_items(claims, "claims", MAX_CLAIMS)
    edges = _bounded_items(edges, "edges", MAX_EDGES)
    supplied_obligations = _bounded_items(obligations, "obligations", MAX_OBLIGATIONS)
    blockers = _bounded_items(blockers, "blockers", MAX_BLOCKERS)
    edge_obligations = tuple(
        obligation for edge in edges for obligation in edge.obligations
    )
    obligations_by_id: dict[str, VerificationObligation] = {}
    for item in edge_obligations + supplied_obligations:
        existing = obligations_by_id.get(item.obligation_id)
        if existing is not None and existing != item:
            raise CognitionValidationError(
                "synthesis obligation identity has conflicting records"
            )
        obligations_by_id[item.obligation_id] = item
    for requirement in problem.requirements:
        if not requirement.user_constraint:
            continue
        obligation = VerificationObligation(
            f"{requirement.requirement_id}-user-constraint-obligation",
            problem.problem_id,
            requirement.requirement_id,
            ObligationKind.UNRESOLVED_USER_CONSTRAINT,
            f"User constraint must be satisfied: {requirement.text}",
            (
                ObligationState.SATISFIED
                if requirement.status is SatisfactionState.SATISFIED
                else ObligationState.OPEN
            ),
            provenance={"requirement_id": requirement.requirement_id},
        )
        existing = obligations_by_id.get(obligation.obligation_id)
        if existing is not None and existing != obligation:
            raise CognitionValidationError(
                "user-constraint obligation conflicts with supplied proof"
            )
        obligations_by_id[obligation.obligation_id] = obligation
    obligations = tuple(obligations_by_id.values())
    _validate_proof_graph(
        claims,
        edges,
        obligations,
        blockers,
        source_lineage=problem.source_lineage,
        allowed_subject_ids=(
            item.requirement_id for item in problem.requirements
        ),
    )
    claim_ids = tuple(item.claim_id for item in claims)
    linked_edge_ids = {item.edge_id for item in edges if item.source_id in claim_ids or item.target_id in claim_ids}
    relevant_subject_ids = (
        set(claim_ids)
        | linked_edge_ids
        | {
            item.requirement_id
            for item in problem.requirements
            if item.user_constraint
        }
    )
    relevant = tuple(
        item for item in obligations if item.subject_id in relevant_subject_ids
    )
    relevant_obligation_ids = {item.obligation_id for item in relevant}
    hard_blockers = tuple(
        item for item in blockers if item.hard and item.obligation_id in relevant_obligation_ids
    )
    # The explicit state check is intentionally independent of blocker presence:
    # an open obligation is already a fail-closed synthesis boundary.
    open_obligations = tuple(item for item in relevant if item.blocking)
    allowed = not open_obligations and not hard_blockers and all(
        item.status is not EpistemicStatus.CONTRADICTION for item in claims
    )
    return SynthesisAssessment(
        allowed,
        claim_ids,
        tuple(item.obligation_id for item in open_obligations),
        tuple(item.blocker_id for item in hard_blockers),
        "candidate synthesis permitted" if allowed else "candidate synthesis blocked by unresolved proof obligations",
    )


@dataclass(frozen=True)
class ProcessingCapability:
    capability_id: str
    operation: ProcessingOperation
    route_id: str
    branch_id: str
    declared_input_ids: tuple[str, ...]
    declared_output_material_ids: tuple[str, ...] = ()
    declared_satisfaction_ids: tuple[str, ...] = ()
    cost: int = 1
    priority: int = 0
    min_route_weight: float = MIN_ROUTE_WEIGHT
    semantic_job: SemanticJob | None = None
    semantic_configuration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.capability_id, "capability_id")
        object.__setattr__(self, "operation", ProcessingOperation(self.operation))
        _identifier(self.route_id, "route_id")
        _identifier(self.branch_id, "branch_id")
        object.__setattr__(self, "declared_input_ids", _ids(self.declared_input_ids, "declared_input_ids", MAX_CONTEXT_ITEMS))
        object.__setattr__(
            self,
            "declared_output_material_ids",
            _ids(
                self.declared_output_material_ids,
                "declared_output_material_ids",
                MAX_CONTEXT_ITEMS,
            ),
        )
        object.__setattr__(
            self,
            "declared_satisfaction_ids",
            _ids(
                self.declared_satisfaction_ids,
                "declared_satisfaction_ids",
                MAX_CONTEXT_ITEMS,
            ),
        )
        _positive_int(self.cost, "capability cost", MAX_OPERATION_COST)
        if not -100 <= self.priority <= 100:
            raise CognitionValidationError("capability priority must be from -100 through 100")
        object.__setattr__(self, "min_route_weight", _bounded(self.min_route_weight, "min_route_weight"))
        if self.semantic_job is not None:
            object.__setattr__(self, "semantic_job", SemanticJob(self.semantic_job))
        if (
            self.operation is ProcessingOperation.MODEL_PROPOSAL
            and self.semantic_job is None
        ):
            raise CognitionValidationError(
                "model proposal capability requires a declared semantic job"
            )
        if (
            self.operation is not ProcessingOperation.MODEL_PROPOSAL
            and self.semantic_job is not None
        ):
            raise CognitionValidationError(
                "semantic jobs are restricted to model proposal capabilities"
            )
        if not isinstance(self.semantic_configuration, Mapping):
            raise CognitionValidationError(
                "semantic_configuration must be a mapping"
            )
        semantic_configuration = dict(self.semantic_configuration)
        expected_configuration_fields = {
            "model_id",
            "max_tokens",
            "temperature",
            "prompt_template_version",
        }
        if self.semantic_job is not None:
            if set(semantic_configuration) != expected_configuration_fields:
                raise CognitionValidationError(
                    "model proposal capability requires exact semantic configuration"
                )
            _identifier(
                semantic_configuration["model_id"],
                "semantic configuration model_id",
            )
            _positive_int(
                semantic_configuration["max_tokens"],
                "semantic configuration max_tokens",
                4096,
            )
            temperature = semantic_configuration["temperature"]
            if (
                isinstance(temperature, bool)
                or not isinstance(temperature, (int, float))
                or not 0.0 <= float(temperature) <= 2.0
            ):
                raise CognitionValidationError(
                    "semantic configuration temperature must be from 0 through 2"
                )
            semantic_configuration["temperature"] = float(temperature)
            _identifier(
                semantic_configuration["prompt_template_version"],
                "semantic configuration prompt_template_version",
            )
        elif semantic_configuration:
            raise CognitionValidationError(
                "non-model capability cannot carry semantic configuration"
            )
        object.__setattr__(
            self,
            "semantic_configuration",
            _freeze(semantic_configuration),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "operation": self.operation.value,
            "route_id": self.route_id,
            "branch_id": self.branch_id,
            "declared_input_ids": list(self.declared_input_ids),
            "declared_output_material_ids": list(self.declared_output_material_ids),
            "declared_satisfaction_ids": list(self.declared_satisfaction_ids),
            "cost": self.cost,
            "priority": self.priority,
            "min_route_weight": self.min_route_weight,
            "semantic_job": self.semantic_job.value if self.semantic_job else None,
            "semantic_configuration": _jsonable(self.semantic_configuration),
        }


@dataclass(frozen=True)
class AdaptiveProcessingProjection:
    projection_id: str
    topology_id: str
    topology_version: int
    topology_generation: int
    active_tactic_id: str
    available_capability_ids: tuple[str, ...]
    inhibited_capability_ids: tuple[str, ...]
    route_weights: tuple[tuple[str, float], ...]
    read_links: tuple[str, ...]
    topology_read_hash: str

    def __post_init__(self) -> None:
        _identifier(self.projection_id, "projection_id")
        _identifier(self.topology_id, "topology_id")
        _identifier(self.active_tactic_id, "active_tactic_id")
        _positive_int(self.topology_version, "topology_version")
        _positive_int(self.topology_generation, "topology_generation")
        object.__setattr__(self, "available_capability_ids", _ids(self.available_capability_ids, "available_capability_ids", MAX_GRAPH_NODES))
        object.__setattr__(self, "inhibited_capability_ids", _ids(self.inhibited_capability_ids, "inhibited_capability_ids", MAX_GRAPH_NODES))
        object.__setattr__(self, "read_links", _ids(self.read_links, "read_links", MAX_GRAPH_NODES))
        if set(self.available_capability_ids) & set(self.inhibited_capability_ids):
            raise CognitionValidationError("a capability cannot be both available and inhibited")
        object.__setattr__(self, "route_weights", tuple((str(route_id), _bounded(weight, "route weight")) for route_id, weight in self.route_weights))

    def to_dict(self) -> dict[str, Any]:
        return {
            "projection_id": self.projection_id,
            "topology_id": self.topology_id,
            "topology_version": self.topology_version,
            "topology_generation": self.topology_generation,
            "active_tactic_id": self.active_tactic_id,
            "available_capability_ids": list(self.available_capability_ids),
            "inhibited_capability_ids": list(self.inhibited_capability_ids),
            "route_weights": [list(item) for item in self.route_weights],
            "read_links": list(self.read_links),
            "topology_read_hash": self.topology_read_hash,
        }


def project_adaptive_processing(
    problem: ProblemGraph,
    capabilities: Iterable[ProcessingCapability],
    adaptive_state: Any,
) -> AdaptiveProcessingProjection:
    """Read existing adaptive topology without importing or mutating its reducer."""

    if adaptive_state is None or not hasattr(adaptive_state, "route_topology"):
        raise CognitionValidationError("adaptive_state must expose a candidate route_topology")
    capabilities = _bounded_items(capabilities, "capabilities", MAX_GRAPH_NODES)
    if not all(isinstance(item, ProcessingCapability) for item in capabilities):
        raise CognitionValidationError("capabilities are invalid")
    topology = adaptive_state.route_topology
    routes = {item.route_id: item for item in topology.routes}
    weights = tuple(sorted((route_id, float(route.weight)) for route_id, route in routes.items()))
    available: list[str] = []
    inhibited: list[str] = []
    for capability in capabilities:
        route = routes.get(capability.route_id)
        if route is None or route.weight < capability.min_route_weight:
            inhibited.append(capability.capability_id)
        else:
            available.append(capability.capability_id)
    active_tactic = getattr(adaptive_state, "active_tactic_id", "candidate-default")
    _identifier(active_tactic, "active_tactic_id")
    read_links = tuple(
        [f"topology:{topology.topology_id}", f"topology-version:{topology.version}", f"topology-generation:{topology.generation}"]
        + [f"route:{route_id}" for route_id, _ in weights]
        + [f"capability:{capability_id}" for capability_id in available + inhibited]
    )
    projection_id = f"{problem.problem_id}-projection-{topology.version}-{topology.generation}"
    read_hash = _hash({
        "problem_hash": problem.graph_hash,
        "topology_id": topology.topology_id,
        "topology_version": topology.version,
        "topology_generation": topology.generation,
        "active_tactic_id": active_tactic,
        "available": available,
        "inhibited": inhibited,
        "weights": weights,
    })
    return AdaptiveProcessingProjection(
        projection_id,
        topology.topology_id,
        topology.version,
        topology.generation,
        active_tactic,
        tuple(available),
        tuple(inhibited),
        weights,
        read_links,
        read_hash,
    )


@dataclass(frozen=True)
class ProcessingBudget:
    max_operations: int = 1
    max_cost: int = 8
    max_events: int = MAX_EVENTS

    def __post_init__(self) -> None:
        _positive_int(self.max_operations, "max_operations", MAX_PROCESSING_OPERATIONS)
        _positive_int(self.max_cost, "max_cost", MAX_OPERATION_COST)
        _positive_int(self.max_events, "max_events", MAX_EVENTS)
        if self.max_operations == 0 or self.max_cost == 0 or self.max_events == 0:
            raise CognitionValidationError("processing budget must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_operations": self.max_operations,
            "max_cost": self.max_cost,
            "max_events": self.max_events,
        }


@dataclass(frozen=True)
class ProcessingDecision:
    decision_id: str
    projection_id: str
    selected_capability_id: str | None
    operation: ProcessingOperation | None
    branch_id: str | None
    input_ids: tuple[str, ...]
    available_capability_ids: tuple[str, ...]
    inhibited_capability_ids: tuple[str, ...]
    topology_read_hash: str
    budget_before: int
    cost_reserved: int
    causal_consumer_links: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.decision_id, "decision_id")
        _identifier(self.projection_id, "projection_id")
        object.__setattr__(self, "input_ids", _ids(self.input_ids, "decision input_ids", MAX_CONTEXT_ITEMS))
        object.__setattr__(self, "available_capability_ids", _ids(self.available_capability_ids, "decision available capabilities", MAX_GRAPH_NODES))
        object.__setattr__(self, "inhibited_capability_ids", _ids(self.inhibited_capability_ids, "decision inhibited capabilities", MAX_GRAPH_NODES))
        _positive_int(self.budget_before, "budget_before")
        _positive_int(self.cost_reserved, "cost_reserved")
        object.__setattr__(self, "causal_consumer_links", _ids(self.causal_consumer_links, "causal_consumer_links", MAX_GRAPH_NODES))
        if self.operation is not None:
            object.__setattr__(self, "operation", ProcessingOperation(self.operation))
        if self.selected_capability_id is None and self.operation is not None:
            raise CognitionValidationError("operation requires selected capability")
        if self.selected_capability_id is not None:
            _identifier(self.selected_capability_id, "selected_capability_id")
            if self.operation is None or self.branch_id is None:
                raise CognitionValidationError("selected operation requires operation and branch")
            _identifier(self.branch_id, "branch_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "projection_id": self.projection_id,
            "selected_capability_id": self.selected_capability_id,
            "operation": self.operation.value if self.operation else None,
            "branch_id": self.branch_id,
            "input_ids": list(self.input_ids),
            "available_capability_ids": list(self.available_capability_ids),
            "inhibited_capability_ids": list(self.inhibited_capability_ids),
            "topology_read_hash": self.topology_read_hash,
            "budget_before": self.budget_before,
            "cost_reserved": self.cost_reserved,
            "causal_consumer_links": list(self.causal_consumer_links),
        }


@dataclass(frozen=True)
class OperationRequest:
    request_id: str
    problem_id: str
    operation: ProcessingOperation
    capability_id: str
    branch_id: str
    input_ids: tuple[str, ...]
    allowed_output_material_ids: tuple[str, ...]
    allowed_satisfaction_ids: tuple[str, ...]
    semantic_job: SemanticJob | None
    semantic_configuration: Mapping[str, Any]
    focused_context: Mapping[str, Any]
    input_hash: str
    budget_cost: int

    def __post_init__(self) -> None:
        _identifier(self.request_id, "request_id")
        _identifier(self.problem_id, "problem_id")
        object.__setattr__(self, "operation", ProcessingOperation(self.operation))
        _identifier(self.capability_id, "capability_id")
        _identifier(self.branch_id, "branch_id")
        object.__setattr__(self, "input_ids", _ids(self.input_ids, "operation input_ids", MAX_CONTEXT_ITEMS))
        object.__setattr__(
            self,
            "allowed_output_material_ids",
            _ids(
                self.allowed_output_material_ids,
                "allowed_output_material_ids",
                MAX_CONTEXT_ITEMS,
            ),
        )
        object.__setattr__(
            self,
            "allowed_satisfaction_ids",
            _ids(
                self.allowed_satisfaction_ids,
                "allowed_satisfaction_ids",
                MAX_CONTEXT_ITEMS,
            ),
        )
        if self.semantic_job is not None:
            object.__setattr__(self, "semantic_job", SemanticJob(self.semantic_job))
        if (
            self.operation is ProcessingOperation.MODEL_PROPOSAL
            and self.semantic_job is None
        ):
            raise CognitionValidationError(
                "model proposal request requires a semantic job"
            )
        if (
            self.operation is not ProcessingOperation.MODEL_PROPOSAL
            and self.semantic_job is not None
        ):
            raise CognitionValidationError(
                "non-model requests cannot carry a semantic job"
            )
        if not isinstance(self.semantic_configuration, Mapping):
            raise CognitionValidationError(
                "semantic_configuration must be a mapping"
            )
        semantic_configuration = dict(self.semantic_configuration)
        if self.semantic_job is not None:
            if set(semantic_configuration) != {
                "model_id",
                "max_tokens",
                "temperature",
                "prompt_template_version",
            }:
                raise CognitionValidationError(
                    "semantic request configuration is incomplete"
                )
        elif semantic_configuration:
            raise CognitionValidationError(
                "non-model request cannot carry semantic configuration"
            )
        object.__setattr__(
            self,
            "semantic_configuration",
            _freeze(
                _bounded_payload(
                    semantic_configuration,
                    "semantic configuration",
                )
            ),
        )
        object.__setattr__(
            self,
            "focused_context",
            _freeze(_bounded_payload(self.focused_context, "focused context")),
        )
        _identifier(self.input_hash, "input_hash")
        _positive_int(self.budget_cost, "budget_cost", MAX_OPERATION_COST)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "problem_id": self.problem_id,
            "operation": self.operation.value,
            "capability_id": self.capability_id,
            "branch_id": self.branch_id,
            "input_ids": list(self.input_ids),
            "allowed_output_material_ids": list(self.allowed_output_material_ids),
            "allowed_satisfaction_ids": list(self.allowed_satisfaction_ids),
            "semantic_job": self.semantic_job.value if self.semantic_job else None,
            "semantic_configuration": _jsonable(self.semantic_configuration),
            "focused_context": _jsonable(self.focused_context),
            "input_hash": self.input_hash,
            "budget_cost": self.budget_cost,
        }


@dataclass(frozen=True)
class OperationResult:
    result_id: str
    request_id: str
    output: Mapping[str, Any] = field(default_factory=dict)
    downstream_material_ids: tuple[str, ...] = ()
    downstream_satisfaction: Mapping[str, SatisfactionState] = field(default_factory=dict)
    questions: tuple[str, ...] = ()
    investigative_needs: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    settlement_ids: tuple[str, ...] = ()
    adaptive_update_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.result_id, "result_id")
        _identifier(self.request_id, "request_id")
        object.__setattr__(self, "output", _freeze(_bounded_payload(self.output, "operation output")))
        object.__setattr__(self, "downstream_material_ids", _ids(self.downstream_material_ids, "downstream_material_ids", MAX_CONTEXT_ITEMS))
        object.__setattr__(self, "downstream_satisfaction", _freeze({
            _identifier(key, "downstream satisfaction key"): SatisfactionState(value)
            for key, value in self.downstream_satisfaction.items()
        }))
        object.__setattr__(self, "questions", tuple(_text(item, "question", 2_048) for item in _bounded_items(self.questions, "questions", MAX_CONTEXT_ITEMS)))
        object.__setattr__(self, "investigative_needs", tuple(_text(item, "investigative need", 2_048) for item in _bounded_items(self.investigative_needs, "investigative_needs", MAX_CONTEXT_ITEMS)))
        for name in ("evidence_ids", "settlement_ids", "adaptive_update_ids"):
            values = _ids(getattr(self, name), name, MAX_CONTEXT_ITEMS)
            if values:
                raise CognitionValidationError(f"processing results cannot create {name}")
            object.__setattr__(self, name, values)

    @property
    def output_hash(self) -> str:
        return _hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "result_id": self.result_id,
            "request_id": self.request_id,
            "output": _jsonable(self.output),
            "downstream_material_ids": list(self.downstream_material_ids),
            "downstream_satisfaction": {key: value.value for key, value in self.downstream_satisfaction.items()},
            "questions": list(self.questions),
            "investigative_needs": list(self.investigative_needs),
            "evidence_ids": [],
            "settlement_ids": [],
            "adaptive_update_ids": [],
        }
        if include_hash:
            result["output_hash"] = self.output_hash
        return result


@dataclass(frozen=True)
class ProcessingEvent:
    event_id: str
    kind: ProcessingEventKind
    subject_id: str
    input_hash: str
    output_hash: str
    causal_links: tuple[str, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, name in ((self.event_id, "event_id"), (self.subject_id, "subject_id"), (self.input_hash, "input_hash"), (self.output_hash, "output_hash")):
            _identifier(value, name)
        object.__setattr__(self, "kind", ProcessingEventKind(self.kind))
        object.__setattr__(self, "causal_links", _ids(self.causal_links, "event causal_links", MAX_GRAPH_NODES))
        object.__setattr__(self, "payload", _freeze(_bounded_payload(self.payload, "event payload")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind.value,
            "subject_id": self.subject_id,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "causal_links": list(self.causal_links),
            "payload": _jsonable(self.payload),
        }


@dataclass(frozen=True)
class ContextSnapshot:
    context_id: str
    problem_hash: str
    branch_id: str
    active_material_ids: tuple[str, ...]
    satisfaction: Mapping[str, SatisfactionState]
    output_hash: str
    changed: bool

    def __post_init__(self) -> None:
        _identifier(self.context_id, "context_id")
        _identifier(self.problem_hash, "problem_hash")
        _identifier(self.branch_id, "branch_id")
        object.__setattr__(self, "active_material_ids", _ids(self.active_material_ids, "active_material_ids", MAX_CONTEXT_ITEMS))
        object.__setattr__(self, "satisfaction", _freeze({
            _identifier(key, "context satisfaction key"): SatisfactionState(value)
            for key, value in self.satisfaction.items()
        }))
        _identifier(self.output_hash, "context output_hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "problem_hash": self.problem_hash,
            "branch_id": self.branch_id,
            "active_material_ids": list(self.active_material_ids),
            "satisfaction": {key: value.value for key, value in self.satisfaction.items()},
            "output_hash": self.output_hash,
            "changed": self.changed,
        }


def _validate_capability_scope(
    problem: ProblemGraph, capabilities: tuple[ProcessingCapability, ...]
) -> None:
    branch_by_id = {item.branch_id: item for item in problem.branches}
    requirement_ids = {item.requirement_id for item in problem.requirements}
    subtask_by_id = {item.subtask_id: item for item in problem.subtasks}
    material_by_id = {item.material_id: item for item in problem.materials}
    all_input_ids = requirement_ids | set(subtask_by_id) | set(material_by_id)
    for capability in capabilities:
        branch = branch_by_id.get(capability.branch_id)
        if branch is None:
            raise CognitionValidationError("capability names an undeclared branch")
        if set(capability.declared_input_ids) - all_input_ids:
            raise CognitionValidationError("capability names an undeclared input")
        for input_id in capability.declared_input_ids:
            subtask = subtask_by_id.get(input_id)
            material = material_by_id.get(input_id)
            if subtask is not None and subtask.branch_id not in (
                None,
                capability.branch_id,
            ):
                raise CognitionValidationError("capability crosses subtask branch scope")
            if (
                material is not None
                and material.environment_id is not None
                and material.environment_id != branch.environment_id
            ):
                raise CognitionValidationError(
                    "capability crosses material environment scope"
                )
        if set(capability.declared_output_material_ids) - set(material_by_id):
            raise CognitionValidationError(
                "capability names an undeclared output material"
            )
        for material_id in capability.declared_output_material_ids:
            material = material_by_id[material_id]
            if (
                material.environment_id is not None
                and material.environment_id != branch.environment_id
            ):
                raise CognitionValidationError(
                    "capability output crosses environment scope"
                )
        branch_subtasks = tuple(
            subtask_by_id[subtask_id] for subtask_id in branch.subtask_ids
        )
        branch_requirement_ids = {
            requirement_id
            for subtask in branch_subtasks
            for requirement_id in subtask.requirement_ids
        }
        allowed_satisfaction = branch_requirement_ids | set(branch.subtask_ids)
        if set(capability.declared_satisfaction_ids) - allowed_satisfaction:
            raise CognitionValidationError(
                "capability crosses satisfaction branch scope"
            )


def _focused_context_values(
    problem: ProblemGraph,
    capability: ProcessingCapability,
    projection: AdaptiveProcessingProjection,
) -> Mapping[str, Any]:
    requirements = {item.requirement_id: item for item in problem.requirements}
    subtasks = {item.subtask_id: item for item in problem.subtasks}
    materials = {item.material_id: item for item in problem.materials}
    branch = next(
        item for item in problem.branches if item.branch_id == capability.branch_id
    )
    focused_items: list[dict[str, Any]] = []
    for item_id in capability.declared_input_ids:
        if item_id in requirements:
            item = requirements[item_id]
            focused_items.append(
                {
                    "item_id": item.requirement_id,
                    "kind": "requirement",
                    "text": item.text,
                    "status": item.status.value,
                    "user_constraint": item.user_constraint,
                }
            )
        elif item_id in subtasks:
            item = subtasks[item_id]
            focused_items.append(
                {
                    "item_id": item.subtask_id,
                    "kind": "subtask",
                    "question": item.question,
                    "requirement_ids": list(item.requirement_ids),
                    "status": item.status.value,
                }
            )
        else:
            item = materials[item_id]
            focused_items.append(
                {
                    "item_id": item.material_id,
                    "kind": "material",
                    "content": item.content,
                    "material_kind": item.kind,
                    "state": item.state.value,
                    "source_lineage_ids": list(item.source_lineage_ids),
                    "environment_id": item.environment_id,
                }
            )
    return _freeze(
        _bounded_payload(
            {
                "problem_id": problem.problem_id,
                "problem_hash": problem.graph_hash,
                "central_question": problem.central_question,
                "branch": {
                    "branch_id": branch.branch_id,
                    "label": branch.label,
                    "environment_id": branch.environment_id,
                    "status": branch.status.value,
                },
                "semantic_job": (
                    capability.semantic_job.value
                    if capability.semantic_job is not None
                    else None
                ),
                "use_class": (
                    f"{capability.semantic_job.value}_candidate"
                    if capability.semantic_job is not None
                    else "candidate_processing"
                ),
                "model_configuration": _jsonable(
                    capability.semantic_configuration
                ),
                "focus_ids": list(capability.declared_input_ids),
                "focused_items": focused_items,
                "topology": {
                    "projection_id": projection.projection_id,
                    "topology_read_hash": projection.topology_read_hash,
                    "active_tactic_id": projection.active_tactic_id,
                    "route_id": capability.route_id,
                    "route_weight": _route_weight(projection, capability.route_id),
                },
                "candidate_only": True,
            },
            "focused context",
        )
    )


def _canonical_context_values(
    problem: ProblemGraph,
    capability: ProcessingCapability,
    result: OperationResult,
) -> tuple[tuple[str, ...], dict[str, SatisfactionState], str, bool]:
    if set(result.downstream_material_ids) - set(
        capability.declared_output_material_ids
    ):
        raise CognitionValidationError(
            "operation returned an undeclared output material"
        )
    if set(result.downstream_satisfaction) - set(
        capability.declared_satisfaction_ids
    ):
        raise CognitionValidationError(
            "operation changed satisfaction outside its branch scope"
        )
    before_materials = tuple(
        item.material_id
        for item in problem.materials
        if item.state in (MaterialState.ACTIVE, MaterialState.REACTIVATED)
    )
    after_materials = tuple(
        dict.fromkeys(before_materials + result.downstream_material_ids)
    )
    satisfaction = dict(problem.satisfaction)
    satisfaction.update(result.downstream_satisfaction)
    output_hash = _hash({
        "problem": problem.graph_hash,
        "branch": capability.branch_id,
        "materials": after_materials,
        "satisfaction": {
            key: value.value for key, value in satisfaction.items()
        },
        "result": result.output_hash,
    })
    changed = (
        after_materials != before_materials
        or bool(result.downstream_satisfaction)
        or bool(result.output)
        or bool(result.questions)
        or bool(result.investigative_needs)
    )
    return after_materials, satisfaction, output_hash, changed


@dataclass(frozen=True)
class ProcessingTrace:
    problem: ProblemGraph
    projection: AdaptiveProcessingProjection
    decision: ProcessingDecision
    request: OperationRequest | None
    result: OperationResult | None
    context: ContextSnapshot
    events: tuple[ProcessingEvent, ...]
    claims: tuple[EpistemicClaim, ...] = ()
    reasoning_edges: tuple[TypedReasoningEdge, ...] = ()
    obligations: tuple[VerificationObligation, ...] = ()
    blockers: tuple[Blocker, ...] = ()
    capabilities: tuple[ProcessingCapability, ...] = ()
    budget: ProcessingBudget = field(default_factory=ProcessingBudget)
    causal_noop: bool = False
    candidate_only: bool = True

    def __post_init__(self) -> None:
        if not self.candidate_only:
            raise CognitionValidationError("processing trace must remain candidate-only")
        events = _bounded_items(self.events, "processing events", MAX_EVENTS)
        if not all(isinstance(item, ProcessingEvent) for item in events):
            raise CognitionValidationError("processing events are invalid")
        ids = [item.event_id for item in events]
        if len(set(ids)) != len(ids):
            raise CognitionValidationError("processing event identities must be unique")
        object.__setattr__(self, "events", events)
        if self.request is None and self.result is not None:
            raise CognitionValidationError("operation result requires operation request")
        claims = _bounded_items(self.claims, "trace claims", MAX_CLAIMS)
        edges = _bounded_items(self.reasoning_edges, "trace reasoning_edges", MAX_EDGES)
        obligations = _bounded_items(self.obligations, "trace obligations", MAX_OBLIGATIONS)
        blockers = _bounded_items(self.blockers, "trace blockers", MAX_BLOCKERS)
        if not all(isinstance(item, EpistemicClaim) for item in claims):
            raise CognitionValidationError("trace claims are invalid")
        if not all(isinstance(item, TypedReasoningEdge) for item in edges):
            raise CognitionValidationError("trace reasoning edges are invalid")
        if not all(isinstance(item, VerificationObligation) for item in obligations):
            raise CognitionValidationError("trace obligations are invalid")
        if not all(isinstance(item, Blocker) for item in blockers):
            raise CognitionValidationError("trace blockers are invalid")
        for values, name, identity in (
            (claims, "claim", "claim_id"),
            (edges, "reasoning edge", "edge_id"),
            (obligations, "obligation", "obligation_id"),
            (blockers, "blocker", "blocker_id"),
        ):
            ids = [getattr(item, identity) for item in values]
            if len(set(ids)) != len(ids):
                raise CognitionValidationError(f"trace {name} identities must be unique")
        if any(item.problem_id != self.problem.problem_id for item in claims + edges + obligations):
            raise CognitionValidationError("trace proof records must bind the problem")
        obligation_ids = {item.obligation_id for item in obligations}
        if any(item.obligation_id not in obligation_ids for item in blockers):
            raise CognitionValidationError("trace blocker names an undeclared obligation")
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "reasoning_edges", edges)
        object.__setattr__(self, "obligations", obligations)
        object.__setattr__(self, "blockers", blockers)
        claim_ids = {item.claim_id for item in claims}
        edge_ids = {item.edge_id for item in edges}
        obligation_by_id = {item.obligation_id: item for item in obligations}
        blocker_by_id = {item.blocker_id: item for item in blockers}
        lineages = tuple(
            lineage for claim in claims for lineage in claim.source_lineage
        )
        lineage_registry = {
            item.lineage_id: item for item in self.problem.source_lineage
        }
        if any(
            lineage_registry.get(item.lineage_id) != item for item in lineages
        ):
            raise CognitionValidationError(
                "claim lineage is absent or conflicts with problem provenance"
            )
        lineage_ids = {item.lineage_id for item in lineages}
        if len(lineage_ids) != len(lineages):
            raise CognitionValidationError("trace lineage identities must be unique")
        for claim in claims:
            if (
                set(claim.support_claim_ids) - claim_ids
                or set(claim.contradiction_claim_ids) - claim_ids
            ):
                raise CognitionValidationError("claim references an undeclared claim")
            if (
                set(claim.assumption_ids) - set(obligation_by_id)
                or set(claim.unresolved_ids) - set(obligation_by_id)
            ):
                raise CognitionValidationError(
                    "claim assumption or unresolved reference is undeclared"
                )
        if any(set(item.parent_lineage_ids) - lineage_ids for item in lineages):
            raise CognitionValidationError("source lineage parent is undeclared")
        for edge in edges:
            if edge.source_id not in claim_ids or edge.target_id not in claim_ids:
                raise CognitionValidationError("reasoning edge endpoint is undeclared")
            for edge_obligation in edge.obligations:
                if obligation_by_id.get(edge_obligation.obligation_id) != edge_obligation:
                    raise CognitionValidationError(
                        "edge obligation is absent or conflicts with trace proof"
                    )
        if any(item.subject_id not in claim_ids | edge_ids for item in obligations):
            raise CognitionValidationError("obligation subject is undeclared")
        source_ids = (
            claim_ids
            | lineage_ids
            | {item.material_id for item in self.problem.materials}
        )
        for obligation in obligations:
            if set(obligation.blocker_ids) - set(blocker_by_id):
                raise CognitionValidationError("obligation blocker is undeclared")
            if any(
                blocker_by_id[blocker_id].obligation_id != obligation.obligation_id
                for blocker_id in obligation.blocker_ids
            ):
                raise CognitionValidationError("obligation blocker binding is invalid")
        for blocker in blockers:
            obligation = obligation_by_id.get(blocker.obligation_id)
            if obligation is None or blocker.blocker_id not in obligation.blocker_ids:
                raise CognitionValidationError("blocker is not reciprocally bound")
            if set(blocker.source_ids) - source_ids:
                raise CognitionValidationError("blocker source is undeclared")
        capabilities = _bounded_items(
            self.capabilities, "trace capabilities", MAX_GRAPH_NODES
        )
        if not all(isinstance(item, ProcessingCapability) for item in capabilities):
            raise CognitionValidationError("trace capabilities are invalid")
        if len({item.capability_id for item in capabilities}) != len(capabilities):
            raise CognitionValidationError("trace capability identities must be unique")
        object.__setattr__(self, "capabilities", capabilities)
        _validate_capability_scope(self.problem, capabilities)
        if not isinstance(self.budget, ProcessingBudget):
            raise CognitionValidationError("trace budget is invalid")
        if len(events) > self.budget.max_events:
            raise CognitionValidationError("processing events exceed the trace budget")
        expected_projection_hash = _hash({
            "problem_hash": self.problem.graph_hash,
            "topology_id": self.projection.topology_id,
            "topology_version": self.projection.topology_version,
            "topology_generation": self.projection.topology_generation,
            "active_tactic_id": self.projection.active_tactic_id,
            "available": list(self.projection.available_capability_ids),
            "inhibited": list(self.projection.inhibited_capability_ids),
            "weights": self.projection.route_weights,
        })
        if self.projection.topology_read_hash != expected_projection_hash:
            raise CognitionValidationError("projection topology read hash is invalid")
        if (
            self.decision.projection_id != self.projection.projection_id
            or self.decision.topology_read_hash != self.projection.topology_read_hash
            or self.decision.available_capability_ids
            != self.projection.available_capability_ids
            or self.decision.inhibited_capability_ids
            != self.projection.inhibited_capability_ids
        ):
            raise CognitionValidationError("decision does not bind the topology projection")
        capability_by_id = {item.capability_id: item for item in capabilities}
        weights = dict(self.projection.route_weights)
        expected_available = tuple(
            item.capability_id
            for item in capabilities
            if weights.get(item.route_id, 0.0) >= item.min_route_weight
        )
        expected_inhibited = tuple(
            item.capability_id
            for item in capabilities
            if item.capability_id not in expected_available
        )
        if (
            self.projection.available_capability_ids != expected_available
            or self.projection.inhibited_capability_ids != expected_inhibited
        ):
            raise CognitionValidationError("projection capability availability is invalid")
        selected = (
            capability_by_id.get(self.decision.selected_capability_id)
            if self.decision.selected_capability_id is not None
            else None
        )
        canonical_selected = _select_capability(
            capabilities, self.projection, self.budget
        )
        if (
            canonical_selected.capability_id
            if canonical_selected is not None
            else None
        ) != self.decision.selected_capability_id:
            raise CognitionValidationError(
                "decision does not match deterministic topology selection"
            )
        if self.decision.selected_capability_id is not None and (
            selected is None
            or selected.capability_id not in self.projection.available_capability_ids
        ):
            raise CognitionValidationError("decision selected an unavailable capability")
        if selected is not None:
            if (
                self.request is None
                or self.result is None
                or self.decision.operation is not selected.operation
                or self.decision.branch_id != selected.branch_id
                or self.decision.input_ids != selected.declared_input_ids
                or self.decision.cost_reserved != selected.cost
                or self.decision.cost_reserved > self.budget.max_cost
            ):
                raise CognitionValidationError("decision does not bind its declared capability")
            expected_input_hash = _hash({
                "problem": self.problem.graph_hash,
                "capability": selected.to_dict(),
                "branch": selected.branch_id,
                "topology_read": self.projection.topology_read_hash,
                "focused_context": _jsonable(
                    _focused_context_values(
                        self.problem, selected, self.projection
                    )
                ),
            })
            if (
                self.request.problem_id != self.problem.problem_id
                or self.request.operation is not selected.operation
                or self.request.capability_id != selected.capability_id
                or self.request.branch_id != selected.branch_id
                or self.request.input_ids != selected.declared_input_ids
                or self.request.allowed_output_material_ids
                != selected.declared_output_material_ids
                or self.request.allowed_satisfaction_ids
                != selected.declared_satisfaction_ids
                or self.request.semantic_job != selected.semantic_job
                or self.request.semantic_configuration
                != selected.semantic_configuration
                or self.request.focused_context
                != _focused_context_values(
                    self.problem, selected, self.projection
                )
                or self.request.input_hash != expected_input_hash
                or self.request.budget_cost != selected.cost
                or self.result.request_id != self.request.request_id
            ):
                raise CognitionValidationError("operation request/result binding is invalid")
            if selected.semantic_job is not None:
                from .semantic_capability import validate_semantic_operation_result

                validate_semantic_operation_result(self.request, self.result)
            (
                expected_materials,
                expected_satisfaction,
                expected_context_hash,
                expected_changed,
            ) = _canonical_context_values(self.problem, selected, self.result)
            if (
                self.context.problem_hash != self.problem.graph_hash
                or self.context.branch_id != selected.branch_id
                or self.context.active_material_ids != expected_materials
                or dict(self.context.satisfaction) != expected_satisfaction
                or self.context.output_hash != expected_context_hash
                or self.context.changed != expected_changed
            ):
                raise CognitionValidationError("downstream context binding is invalid")
            operation_events = [
                item for item in events if item.kind is ProcessingEventKind.OPERATION
            ]
            if len(operation_events) != 1 or (
                operation_events[0].subject_id != self.request.request_id
                or operation_events[0].input_hash != self.request.input_hash
                or operation_events[0].output_hash != self.result.output_hash
                or operation_events[0].causal_links
                != (self.decision.decision_id, selected.capability_id)
            ):
                raise CognitionValidationError("processing event chain is invalid")
            activation_events = [
                item for item in events if item.kind is ProcessingEventKind.ACTIVATION
            ]
            context_events = [
                item
                for item in events
                if item.kind is ProcessingEventKind.MOTIF_OBSERVATION
            ]
            if len(activation_events) != 1 or (
                activation_events[0].subject_id != selected.capability_id
                or activation_events[0].input_hash
                != self.projection.topology_read_hash
                or activation_events[0].output_hash != self.request.input_hash
                or activation_events[0].causal_links
                != (self.projection.projection_id,)
            ):
                raise CognitionValidationError("activation event binding is invalid")
            if len(context_events) != 1 or (
                context_events[0].subject_id != self.context.context_id
                or context_events[0].input_hash != self.result.output_hash
                or context_events[0].output_hash != self.context.output_hash
                or context_events[0].causal_links != (self.request.request_id,)
            ):
                raise CognitionValidationError("context event binding is invalid")
        elif self.request is not None or self.result is not None:
            raise CognitionValidationError("unselected processing cannot have operation records")
        else:
            expected_branch = (
                self.problem.branches[0].branch_id
                if self.problem.branches
                else "unbranched"
            )
            expected_materials = tuple(
                item.material_id
                for item in self.problem.materials
                if item.state in (MaterialState.ACTIVE, MaterialState.REACTIVATED)
            )
            expected_context_hash = _hash(
                {"problem": self.problem.graph_hash, "blocked": True}
            )
            if (
                self.context.problem_hash != self.problem.graph_hash
                or self.context.branch_id != expected_branch
                or self.context.active_material_ids != expected_materials
                or dict(self.context.satisfaction)
                != dict(self.problem.satisfaction)
                or self.context.output_hash != expected_context_hash
                or self.context.changed
            ):
                raise CognitionValidationError("blocked context binding is invalid")
            inhibition_events = [
                item for item in events if item.kind is ProcessingEventKind.INHIBITION
            ]
            if len(events) != 1 or len(inhibition_events) != 1:
                raise CognitionValidationError(
                    "blocked trace requires exactly one inhibition event"
                )
            inhibition = inhibition_events[0]
            if (
                inhibition.subject_id != self.decision.decision_id
                or inhibition.input_hash != self.projection.topology_read_hash
                or inhibition.output_hash != self.context.output_hash
                or inhibition.causal_links != (self.projection.projection_id,)
            ):
                raise CognitionValidationError("inhibition event binding is invalid")
        noop_events = [item for item in events if item.kind is ProcessingEventKind.NOOP]
        if self.causal_noop != (len(noop_events) == 1):
            raise CognitionValidationError("causal no-op state and event must agree")
        if len(noop_events) > 1:
            raise CognitionValidationError("causal no-op event cannot repeat")
        if noop_events:
            noop = noop_events[0]
            if (
                noop.subject_id != self.decision.decision_id
                or noop.causal_links[-1:] != (self.projection.projection_id,)
                or len(noop.causal_links) != 2
                or noop.causal_links[0] == noop.causal_links[1]
                or noop.input_hash != noop.output_hash
                or noop.input_hash != self.effective_computation_hash
                or noop.payload.get("topology_changed") is not True
                or noop.payload.get("current_topology_read_hash")
                != self.projection.topology_read_hash
                or noop.payload.get("prior_topology_read_hash")
                == self.projection.topology_read_hash
                or noop.payload.get("prior_effective_computation_hash")
                != noop.input_hash
            ):
                raise CognitionValidationError("causal no-op proof is invalid")

    @property
    def structural_hash(self) -> str:
        return _hash(self.to_dict(include_hash=False))

    @property
    def effective_computation_hash(self) -> str:
        return _hash({
            "operation": self.decision.operation.value if self.decision.operation else None,
            "capability": self.decision.selected_capability_id,
            "branch": self.decision.branch_id,
            "inputs": self.decision.input_ids,
            "budget": self.decision.cost_reserved,
            "result": (
                {
                    "output": _jsonable(self.result.output),
                    "downstream_material_ids": self.result.downstream_material_ids,
                    "downstream_satisfaction": {
                        key: value.value
                        for key, value in self.result.downstream_satisfaction.items()
                    },
                    "questions": self.result.questions,
                    "investigative_needs": self.result.investigative_needs,
                }
                if self.result
                else None
            ),
            "context": {
                "branch_id": self.context.branch_id,
                "active_material_ids": self.context.active_material_ids,
                "satisfaction": {
                    key: value.value for key, value in self.context.satisfaction.items()
                },
                "changed": self.context.changed,
            },
        })

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "problem": self.problem.to_dict(),
            "projection": self.projection.to_dict(),
            "decision": self.decision.to_dict(),
            "request": self.request.to_dict() if self.request else None,
            "result": self.result.to_dict() if self.result else None,
            "context": self.context.to_dict(),
            "events": [item.to_dict() for item in self.events],
            "claims": [item.to_dict() for item in self.claims],
            "reasoning_edges": [item.to_dict() for item in self.reasoning_edges],
            "obligations": [item.to_dict() for item in self.obligations],
            "blockers": [item.to_dict() for item in self.blockers],
            "capabilities": [item.to_dict() for item in self.capabilities],
            "budget": self.budget.to_dict(),
            "causal_noop": self.causal_noop,
            "candidate_only": True,
        }
        if include_hash:
            result["structural_hash"] = self.structural_hash
        return result


def _route_weight(projection: AdaptiveProcessingProjection, route_id: str) -> float:
    return dict(projection.route_weights).get(route_id, 0.0)


def _select_capability(
    capabilities: tuple[ProcessingCapability, ...],
    projection: AdaptiveProcessingProjection,
    budget: ProcessingBudget,
) -> ProcessingCapability | None:
    available = set(projection.available_capability_ids)
    choices = [item for item in capabilities if item.capability_id in available and item.cost <= budget.max_cost]
    return sorted(choices, key=lambda item: (-_route_weight(projection, item.route_id), -item.priority, item.capability_id))[0] if choices else None


def _default_result(request: OperationRequest) -> OperationResult:
    return OperationResult(
        f"{request.request_id}-result",
        request.request_id,
        output={
            "operation": request.operation.value,
            "capability_id": request.capability_id,
            "branch_id": request.branch_id,
            "candidate_only": True,
        },
        downstream_material_ids=request.allowed_output_material_ids,
        downstream_satisfaction={},
        questions=(f"Verify the declared result of {request.operation.value}.",),
        investigative_needs=(f"Independent check for {request.capability_id}.",),
    )


def run_connected_processing(
    problem: ProblemGraph,
    capabilities: Iterable[ProcessingCapability],
    adaptive_state: Any,
    *,
    budget: ProcessingBudget | None = None,
    operation_callbacks: Mapping[ProcessingOperation | str, Callable[[OperationRequest], OperationResult]] | None = None,
    prior_trace: ProcessingTrace | None = None,
    claims: Iterable[EpistemicClaim] = (),
    reasoning_edges: Iterable[TypedReasoningEdge] = (),
    obligations: Iterable[VerificationObligation] = (),
    blockers: Iterable[Blocker] = (),
) -> ProcessingTrace:
    """Select and run one declared candidate operation under a finite budget."""

    if not isinstance(problem, ProblemGraph):
        raise CognitionValidationError("processing requires a ProblemGraph")
    capabilities = _bounded_items(capabilities, "capabilities", MAX_GRAPH_NODES)
    if not all(isinstance(item, ProcessingCapability) for item in capabilities):
        raise CognitionValidationError("capabilities are invalid")
    if len({item.capability_id for item in capabilities}) != len(capabilities):
        raise CognitionValidationError("capability identities must be unique")
    budget = budget or ProcessingBudget()
    if budget.max_operations < 1:
        raise CognitionValidationError("processing budget cannot run an operation")
    if budget.max_events < (4 if prior_trace is not None else 3):
        raise CognitionValidationError("processing event budget is insufficient")
    _validate_capability_scope(problem, tuple(capabilities))
    claims = _bounded_items(claims, "claims", MAX_CLAIMS)
    reasoning_edges = _bounded_items(reasoning_edges, "reasoning_edges", MAX_EDGES)
    explicit_obligations = _bounded_items(obligations, "obligations", MAX_OBLIGATIONS)
    blockers = _bounded_items(blockers, "blockers", MAX_BLOCKERS)
    edge_obligations = tuple(
        obligation for edge in reasoning_edges for obligation in edge.obligations
    )
    obligations_by_id: dict[str, VerificationObligation] = {}
    for item in edge_obligations + explicit_obligations:
        existing = obligations_by_id.get(item.obligation_id)
        if existing is not None and existing != item:
            raise CognitionValidationError("obligation identity has conflicting records")
        obligations_by_id[item.obligation_id] = item
    obligations = tuple(obligations_by_id.values())
    projection = project_adaptive_processing(problem, capabilities, adaptive_state)
    selected = _select_capability(tuple(capabilities), projection, budget)
    decision_id = f"{problem.problem_id}-processing-decision-{projection.topology_version}-{projection.topology_generation}"
    if selected is None:
        decision = ProcessingDecision(
            decision_id, projection.projection_id, None, None, None, (), projection.available_capability_ids,
            projection.inhibited_capability_ids, projection.topology_read_hash, budget.max_cost, 0,
            (f"projection:{projection.projection_id}",),
        )
        context = ContextSnapshot(
            f"{decision_id}-context", problem.graph_hash, problem.branches[0].branch_id if problem.branches else "unbranched",
            tuple(item.material_id for item in problem.materials if item.state in (MaterialState.ACTIVE, MaterialState.REACTIVATED)),
            problem.satisfaction, _hash({"problem": problem.graph_hash, "blocked": True}), False,
        )
        event = ProcessingEvent(
            f"{decision_id}-event", ProcessingEventKind.INHIBITION, decision_id,
            projection.topology_read_hash, context.output_hash,
            causal_links=(projection.projection_id,), payload={"reason": "no capability within budget"},
        )
        return ProcessingTrace(
            problem=problem,
            projection=projection,
            decision=decision,
            request=None,
            result=None,
            context=context,
            events=(event,),
            claims=claims,
            reasoning_edges=reasoning_edges,
            obligations=obligations,
            blockers=blockers,
            capabilities=capabilities,
            budget=budget,
        )
    input_hash = _hash({
        "problem": problem.graph_hash,
        "capability": selected.to_dict(),
        "branch": selected.branch_id,
        "topology_read": projection.topology_read_hash,
        "focused_context": _jsonable(
            _focused_context_values(problem, selected, projection)
        ),
    })
    focused_context = _focused_context_values(problem, selected, projection)
    request = OperationRequest(
        f"{problem.problem_id}-operation-{selected.capability_id}-{projection.topology_version}-{projection.topology_generation}",
        problem.problem_id, selected.operation, selected.capability_id, selected.branch_id,
        selected.declared_input_ids,
        selected.declared_output_material_ids,
        selected.declared_satisfaction_ids,
        selected.semantic_job,
        selected.semantic_configuration,
        focused_context,
        input_hash,
        selected.cost,
    )
    callback = None
    if operation_callbacks:
        callback = operation_callbacks.get(selected.operation) or operation_callbacks.get(selected.operation.value)
    result = callback(request) if callback is not None else _default_result(request)
    if not isinstance(result, OperationResult) or result.request_id != request.request_id:
        raise CognitionValidationError("operation callback must return a bound OperationResult")
    (
        after_materials,
        satisfaction,
        context_output_hash,
        context_changed,
    ) = _canonical_context_values(problem, selected, result)
    context = ContextSnapshot(
        f"{request.request_id}-context", problem.graph_hash, selected.branch_id,
        after_materials, satisfaction, context_output_hash, context_changed,
    )
    decision = ProcessingDecision(
        decision_id, projection.projection_id, selected.capability_id, selected.operation, selected.branch_id,
        selected.declared_input_ids, projection.available_capability_ids, projection.inhibited_capability_ids,
        projection.topology_read_hash, budget.max_cost, selected.cost,
        (projection.projection_id, f"capability:{selected.capability_id}", f"operation:{selected.operation.value}", f"branch:{selected.branch_id}", f"context:{context.context_id}"),
    )
    events = (
        ProcessingEvent(
            f"{decision_id}-activation", ProcessingEventKind.ACTIVATION, selected.capability_id,
            projection.topology_read_hash, input_hash, causal_links=(projection.projection_id,),
            payload={"available": True, "route_id": selected.route_id},
        ),
        ProcessingEvent(
            f"{decision_id}-operation", ProcessingEventKind.OPERATION, request.request_id,
            request.input_hash, result.output_hash, causal_links=(decision_id, selected.capability_id),
            payload={"operation": selected.operation.value, "branch_id": selected.branch_id},
        ),
        ProcessingEvent(
            f"{decision_id}-context", ProcessingEventKind.MOTIF_OBSERVATION, context.context_id,
            result.output_hash, context.output_hash, causal_links=(request.request_id,),
            payload={"changed": context.changed, "candidate_only": True},
        ),
    )
    trace = ProcessingTrace(
        problem=problem,
        projection=projection,
        decision=decision,
        request=request,
        result=result,
        context=context,
        events=events,
        claims=claims,
        reasoning_edges=reasoning_edges,
        obligations=obligations,
        blockers=blockers,
        capabilities=capabilities,
        budget=budget,
    )
    if prior_trace is not None:
        if prior_trace.problem.graph_hash != problem.graph_hash:
            raise CognitionValidationError("prior trace must bind the same problem")
        topology_changed = (
            prior_trace.projection.topology_read_hash
            != projection.topology_read_hash
        )
        causal_noop = (
            topology_changed
            and prior_trace.effective_computation_hash
            == trace.effective_computation_hash
        )
        trace = ProcessingTrace(
            problem=problem,
            projection=projection,
            decision=decision,
            request=request,
            result=result,
            context=context,
            events=events + ((ProcessingEvent(
                f"{decision_id}-noop", ProcessingEventKind.NOOP, decision_id,
                prior_trace.effective_computation_hash, trace.effective_computation_hash,
                causal_links=(prior_trace.projection.projection_id, projection.projection_id),
                payload={
                    "topology_changed": True,
                    "prior_topology_read_hash": prior_trace.projection.topology_read_hash,
                    "current_topology_read_hash": projection.topology_read_hash,
                    "prior_effective_computation_hash": prior_trace.effective_computation_hash,
                },
            ),) if causal_noop else ()),
            claims=claims,
            reasoning_edges=reasoning_edges,
            obligations=obligations,
            blockers=blockers,
            capabilities=capabilities,
            budget=budget,
            causal_noop=causal_noop,
        )
    return trace


def _problem_from_dict(value: Mapping[str, Any]) -> ProblemGraph:
    original_value = value["original_task"]
    original = OriginalTask(
        original_value["task_id"],
        original_value["text"],
        original_value["source"],
        original_value["provenance"],
    )
    problem = ProblemGraph(
        value["problem_id"],
        original,
        value["central_question"],
        tuple(
            ProblemRequirement(
                item["requirement_id"],
                item["text"],
                item["user_constraint"],
                item["status"],
            )
            for item in value["requirements"]
        ),
        tuple(
            ProblemSubtask(
                item["subtask_id"],
                item["question"],
                tuple(item["requirement_ids"]),
                item["status"],
                item["branch_id"],
            )
            for item in value["subtasks"]
        ),
        tuple(
            ProblemMaterial(
                item["material_id"],
                item["content"],
                item["kind"],
                item["state"],
                tuple(item["source_lineage_ids"]),
                item["environment_id"],
            )
            for item in value["materials"]
        ),
        tuple(
            ProblemBranch(
                item["branch_id"],
                item["label"],
                item["environment_id"],
                tuple(item["subtask_ids"]),
                item["status"],
                item["context"],
            )
            for item in value["branches"]
        ),
        tuple(value["environments"]),
        value["satisfaction"],
        value["provenance"],
        tuple(
            SourceLineage(
                item["lineage_id"],
                item["source_id"],
                item["source_kind"],
                item["independent_group"],
                tuple(item["parent_lineage_ids"]),
                item["current"],
                item["provenance"],
            )
            for item in value["source_lineage"]
        ),
    )
    if value.get("graph_hash") != problem.graph_hash:
        raise CognitionValidationError("problem graph replay hash mismatch")
    return problem


def _obligation_from_dict(value: Mapping[str, Any]) -> VerificationObligation:
    return VerificationObligation(
        value["obligation_id"],
        value["problem_id"],
        value["subject_id"],
        value["kind"],
        value["description"],
        value["state"],
        tuple(value["blocker_ids"]),
        value["provenance"],
    )


def _claim_from_dict(value: Mapping[str, Any]) -> EpistemicClaim:
    pressure = value["verification_pressure"]
    return EpistemicClaim(
        value["claim_id"],
        value["problem_id"],
        value["statement"],
        value["status"],
        tuple(value["assumption_ids"]),
        tuple(value["support_claim_ids"]),
        tuple(value["contradiction_claim_ids"]),
        tuple(value["unresolved_ids"]),
        tuple(
            SourceLineage(
                item["lineage_id"],
                item["source_id"],
                item["source_kind"],
                item["independent_group"],
                tuple(item["parent_lineage_ids"]),
                item["current"],
                item["provenance"],
            )
            for item in value["source_lineage"]
        ),
        value["use_site"],
        VerificationPressure(
            pressure["stakes"],
            pressure["volatility"],
            pressure["locality"],
            pressure["precision"],
            pressure["novelty"],
            pressure["contestability"],
            pressure["actionability"],
        ),
        tuple(
            FalsificationCondition(
                item["condition_id"], item["statement"], item["discriminator"]
            )
            for item in value["falsification_conditions"]
        ),
        value["competence_origin"],
        tuple(value["evidence_ids"]),
        value["declared_only"],
    )


def _edge_from_dict(value: Mapping[str, Any]) -> TypedReasoningEdge:
    return TypedReasoningEdge(
        value["edge_id"],
        value["problem_id"],
        value["source_id"],
        value["target_id"],
        value["edge_type"],
        tuple(_obligation_from_dict(item) for item in value["obligations"]),
        value["provenance"],
    )


def _capability_from_dict(value: Mapping[str, Any]) -> ProcessingCapability:
    return ProcessingCapability(
        value["capability_id"],
        value["operation"],
        value["route_id"],
        value["branch_id"],
        tuple(value["declared_input_ids"]),
        tuple(value["declared_output_material_ids"]),
        tuple(value["declared_satisfaction_ids"]),
        value["cost"],
        value["priority"],
        value["min_route_weight"],
        value.get("semantic_job"),
        value.get("semantic_configuration", {}),
    )


def _processing_trace_from_dict(value: Mapping[str, Any]) -> ProcessingTrace:
    expected_fields = {
        "problem", "projection", "decision", "request", "result", "context",
        "events", "claims", "reasoning_edges", "obligations", "blockers",
        "capabilities", "budget", "causal_noop", "candidate_only",
        "structural_hash",
    }
    if set(value) != expected_fields:
        raise CognitionValidationError("processing replay record schema is invalid")
    problem = _problem_from_dict(value["problem"])
    projection_value = value["projection"]
    projection = AdaptiveProcessingProjection(
        projection_value["projection_id"],
        projection_value["topology_id"],
        projection_value["topology_version"],
        projection_value["topology_generation"],
        projection_value["active_tactic_id"],
        tuple(projection_value["available_capability_ids"]),
        tuple(projection_value["inhibited_capability_ids"]),
        tuple(tuple(item) for item in projection_value["route_weights"]),
        tuple(projection_value["read_links"]),
        projection_value["topology_read_hash"],
    )
    decision_value = value["decision"]
    decision = ProcessingDecision(
        decision_value["decision_id"],
        decision_value["projection_id"],
        decision_value["selected_capability_id"],
        decision_value["operation"],
        decision_value["branch_id"],
        tuple(decision_value["input_ids"]),
        tuple(decision_value["available_capability_ids"]),
        tuple(decision_value["inhibited_capability_ids"]),
        decision_value["topology_read_hash"],
        decision_value["budget_before"],
        decision_value["cost_reserved"],
        tuple(decision_value["causal_consumer_links"]),
    )
    request_value = value["request"]
    request = (
        OperationRequest(
            request_value["request_id"],
            request_value["problem_id"],
            request_value["operation"],
            request_value["capability_id"],
            request_value["branch_id"],
            tuple(request_value["input_ids"]),
            tuple(request_value["allowed_output_material_ids"]),
            tuple(request_value["allowed_satisfaction_ids"]),
            request_value.get("semantic_job"),
            request_value.get("semantic_configuration", {}),
            request_value.get("focused_context", {}),
            request_value["input_hash"],
            request_value["budget_cost"],
        )
        if request_value is not None
        else None
    )
    result_value = value["result"]
    result = (
        OperationResult(
            result_value["result_id"],
            result_value["request_id"],
            result_value["output"],
            tuple(result_value["downstream_material_ids"]),
            result_value["downstream_satisfaction"],
            tuple(result_value["questions"]),
            tuple(result_value["investigative_needs"]),
            tuple(result_value["evidence_ids"]),
            tuple(result_value["settlement_ids"]),
            tuple(result_value["adaptive_update_ids"]),
        )
        if result_value is not None
        else None
    )
    if result is not None and result_value.get("output_hash") != result.output_hash:
        raise CognitionValidationError("operation result replay hash mismatch")
    context_value = value["context"]
    context = ContextSnapshot(
        context_value["context_id"],
        context_value["problem_hash"],
        context_value["branch_id"],
        tuple(context_value["active_material_ids"]),
        context_value["satisfaction"],
        context_value["output_hash"],
        context_value["changed"],
    )
    events = tuple(
        ProcessingEvent(
            item["event_id"],
            item["kind"],
            item["subject_id"],
            item["input_hash"],
            item["output_hash"],
            tuple(item["causal_links"]),
            item["payload"],
        )
        for item in value["events"]
    )
    blockers = tuple(
        Blocker(
            item["blocker_id"],
            item["obligation_id"],
            item["kind"],
            item["reason"],
            tuple(item["source_ids"]),
            item["hard"],
        )
        for item in value["blockers"]
    )
    budget_value = value["budget"]
    trace = ProcessingTrace(
        problem=problem,
        projection=projection,
        decision=decision,
        request=request,
        result=result,
        context=context,
        events=events,
        claims=tuple(_claim_from_dict(item) for item in value["claims"]),
        reasoning_edges=tuple(
            _edge_from_dict(item) for item in value["reasoning_edges"]
        ),
        obligations=tuple(
            _obligation_from_dict(item) for item in value["obligations"]
        ),
        blockers=blockers,
        capabilities=tuple(
            _capability_from_dict(item) for item in value["capabilities"]
        ),
        budget=ProcessingBudget(
            budget_value["max_operations"],
            budget_value["max_cost"],
            budget_value["max_events"],
        ),
        causal_noop=value["causal_noop"],
        candidate_only=value["candidate_only"],
    )
    if value["structural_hash"] != trace.structural_hash:
        raise CognitionValidationError("processing structural replay hash mismatch")
    return trace


def replay_processing_trace(record: ProcessingTrace | Mapping[str, Any]) -> ProcessingTrace:
    """Replay structure and hashes only; provider/tool callbacks are never called."""

    if isinstance(record, Mapping):
        return _processing_trace_from_dict(record)
    if not isinstance(record, ProcessingTrace):
        raise CognitionValidationError("processing replay requires a trace or mapping")
    expected = record.structural_hash
    reconstructed = ProcessingTrace(
        problem=record.problem,
        projection=record.projection,
        decision=record.decision,
        request=record.request,
        result=record.result,
        context=record.context,
        events=record.events,
        claims=record.claims,
        reasoning_edges=record.reasoning_edges,
        obligations=record.obligations,
        blockers=record.blockers,
        capabilities=record.capabilities,
        budget=record.budget,
        causal_noop=record.causal_noop,
        candidate_only=record.candidate_only,
    )
    if reconstructed.structural_hash != expected:
        raise CognitionValidationError("processing structural replay hash mismatch")
    return reconstructed


def antimetabole_review(
    problem: ProblemGraph,
    claims: Iterable[EpistemicClaim],
    edges: Iterable[TypedReasoningEdge],
) -> Mapping[str, Any]:
    """Bounded forward consequences and reverse requirements, diagnostics only."""

    claims = tuple(claims)
    edges = tuple(edges)
    if len(claims) > MAX_CLAIMS or len(edges) > MAX_EDGES:
        raise CognitionValidationError("Antimetabole input exceeds its bound")
    claim_ids = {item.claim_id for item in claims}
    forward = {item.claim_id: [] for item in claims}
    reverse = {item.claim_id: [] for item in claims}
    gaps: list[str] = []
    for edge in edges:
        if edge.source_id not in claim_ids or edge.target_id not in claim_ids:
            gaps.append(f"{edge.edge_id}: missing dependency")
            continue
        forward[edge.source_id].append(edge.target_id)
        reverse[edge.target_id].append(edge.source_id)
        if edge.edge_type is EdgeType.CORRELATION and any(ob.kind is ObligationKind.QUANTITATIVE_VALIDITY for ob in edge.obligations):
            gaps.append(f"{edge.edge_id}: correlation cannot masquerade as quantitative relation")
    unresolved = tuple(item.claim_id for item in claims if item.status is EpistemicStatus.UNRESOLVED)
    conflicts = tuple(item.claim_id for item in claims if item.status is EpistemicStatus.CONTRADICTION)
    return {
        "problem_id": problem.problem_id,
        "forward_consequences": {key: tuple(value) for key, value in forward.items()},
        "reverse_requirements": {key: tuple(value) for key, value in reverse.items()},
        "dependency_gaps": tuple(gaps),
        "conflicts": conflicts,
        "unsatisfied_subtasks": tuple(item.subtask_id for item in problem.subtasks if item.status is not SatisfactionState.SATISFIED),
        "unresolved_claims": unresolved,
        "proposal_only": True,
        "creates_evidence": False,
        "creates_adaptive_credit": False,
    }


def angel_projection(problem: ProblemGraph, claim: EpistemicClaim) -> Mapping[str, Any]:
    return {
        "role": "angel",
        "problem_id": problem.problem_id,
        "claim_id": claim.claim_id,
        "survival_questions": tuple(item.discriminator for item in claim.falsification_conditions),
        "investigative_needs": ("Identify the strongest declared support for the hypothesis.",),
        "proposal_only": True,
    }


def nemesis_projection(problem: ProblemGraph, claim: EpistemicClaim) -> Mapping[str, Any]:
    return {
        "role": "nemesis",
        "problem_id": problem.problem_id,
        "claim_id": claim.claim_id,
        "break_point_questions": tuple(item.statement for item in claim.falsification_conditions),
        "investigative_needs": ("Find an independent discriminating test or contradiction.",),
        "proposal_only": True,
    }


__all__ = [
    "AdaptiveProcessingProjection",
    "Blocker",
    "BlockerKind",
    "CompetenceOrigin",
    "ContextSnapshot",
    "CognitionValidationError",
    "EdgeType",
    "EpistemicClaim",
    "EpistemicStatus",
    "FalsificationCondition",
    "MaterialState",
    "MAX_BRANCHES",
    "MAX_CLAIMS",
    "MAX_EDGES",
    "MAX_EVENTS",
    "MAX_MATERIALS",
    "MAX_OBLIGATIONS",
    "ObligationKind",
    "ObligationState",
    "OperationRequest",
    "OperationResult",
    "ProcessingBudget",
    "ProcessingCapability",
    "ProcessingDecision",
    "ProcessingEvent",
    "ProcessingEventKind",
    "ProcessingOperation",
    "SemanticJob",
    "ProcessingTrace",
    "ProblemBranch",
    "ProblemGraph",
    "ProblemMaterial",
    "ProblemRequirement",
    "ProblemSubtask",
    "SatisfactionState",
    "SourceLineage",
    "SynthesisAssessment",
    "TypedReasoningEdge",
    "VerificationObligation",
    "VerificationPressure",
    "angel_projection",
    "antimetabole_review",
    "assess_synthesis",
    "nemesis_projection",
    "project_adaptive_processing",
    "replay_processing_trace",
    "run_connected_processing",
]