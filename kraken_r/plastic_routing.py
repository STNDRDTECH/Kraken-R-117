"""Settlement-grounded, candidate-only plastic route preference.

This module adapts the useful *ideas* behind the quarantined legacy
plasticity, pathway, and topology trackers without importing their runtime
authority.  It is a pure transformation over caller-owned immutable route
state.  It has no store, clock, worker, event subscription, signal handling,
goal selection, execution, or mutation authority.

Only a constitutionally settled success or failure supported by grounded
execution evidence can change a selected candidate route.  Contradictory and
insufficient-evidence results are explicitly withheld rather than credited.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping

from .contracts import Evidence, EvidenceGrade, LearningUpdate, Settlement
from .cycle import ConstitutionalCycle, CycleInvariantError, CycleTrace
from .grounded_execution import (
    GroundedExecutionRejected,
    GroundedExecutionRequest,
    GroundedExecutionVerifier,
    VerifiedGroundedExecution,
)


class PlasticRoutingValidationError(ValueError):
    """Raised when a route-learning record crosses a candidate boundary."""


BASELINE_WEIGHT = 0.50
MIN_WEIGHT = 0.25
MAX_WEIGHT = 0.75
MAX_WEIGHT_CHANGE = 0.10
MAX_AUDIT_LINKS = 16


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(char.isspace() for char in value):
        raise PlasticRoutingValidationError(
            f"{field_name} must be a non-empty identifier without whitespace"
        )
    return value


def _bounded_weight(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlasticRoutingValidationError(f"{field_name} must be a number")
    result = float(value)
    if not MIN_WEIGHT <= result <= MAX_WEIGHT:
        raise PlasticRoutingValidationError(
            f"{field_name} must be from {MIN_WEIGHT} through {MAX_WEIGHT}"
        )
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _audit_append(existing: tuple[str, ...], value: str) -> tuple[str, ...]:
    """Append one immutable audit identity with a fixed retention bound."""

    return (existing + (value,))[-MAX_AUDIT_LINKS:]


@dataclass(frozen=True)
class CandidateRoute:
    """One bounded candidate organizational edge, not an executable route."""

    route_id: str
    context_id: str
    source: str
    target: str
    weight: float = BASELINE_WEIGHT
    success_count: int = 0
    failure_count: int = 0
    settlement_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    lineage_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("route_id", "context_id", "source", "target"):
            _identifier(getattr(self, field_name), field_name)
        object.__setattr__(self, "weight", _bounded_weight(self.weight, "weight"))
        for field_name in ("success_count", "failure_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PlasticRoutingValidationError(
                    f"{field_name} must be a non-negative integer"
                )
        for field_name in ("settlement_ids", "evidence_ids", "lineage_ids"):
            values = getattr(self, field_name)
            if not isinstance(values, (tuple, list)) or not all(
                isinstance(item, str) and item for item in values
            ):
                raise PlasticRoutingValidationError(
                    f"{field_name} must be identifiers"
                )
            if len(values) > MAX_AUDIT_LINKS:
                raise PlasticRoutingValidationError(
                    f"{field_name} exceeds the bounded audit retention"
                )
            object.__setattr__(self, field_name, tuple(values))


@dataclass(frozen=True)
class RouteTopology:
    """Caller-owned immutable route preference and bounded audit state."""

    topology_id: str
    version: int
    routes: tuple[CandidateRoute, ...]
    generation: int = 0
    applied_settlement_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.topology_id, "topology_id")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 0:
            raise PlasticRoutingValidationError("version must be a non-negative integer")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise PlasticRoutingValidationError(
                "generation must be a non-negative integer"
            )
        if not isinstance(self.routes, (tuple, list)) or len(self.routes) < 2:
            raise PlasticRoutingValidationError("topology requires at least two routes")
        if not all(isinstance(route, CandidateRoute) for route in self.routes):
            raise PlasticRoutingValidationError("routes must be CandidateRoute records")
        route_ids = tuple(route.route_id for route in self.routes)
        if len(set(route_ids)) != len(route_ids):
            raise PlasticRoutingValidationError("route ids must be unique")
        if not isinstance(self.applied_settlement_ids, (tuple, list)) or not all(
            isinstance(item, str) and item for item in self.applied_settlement_ids
        ):
            raise PlasticRoutingValidationError(
                "applied_settlement_ids must be settlement identifiers"
            )
        if len(self.applied_settlement_ids) > MAX_AUDIT_LINKS:
            raise PlasticRoutingValidationError(
                "applied_settlement_ids exceeds the bounded audit retention"
            )
        object.__setattr__(self, "routes", tuple(self.routes))
        object.__setattr__(
            self, "applied_settlement_ids", tuple(self.applied_settlement_ids)
        )

    @classmethod
    def fixture(cls, topology_id: str = "candidate-topology") -> "RouteTopology":
        """Build a neutral two-path candidate organization for tests and examples."""

        return cls(
            topology_id,
            0,
            (
                CandidateRoute("path-alpha", "candidate-work", "candidate", "alpha"),
                CandidateRoute("path-beta", "candidate-work", "candidate", "beta"),
            ),
        )

    def reset(self) -> "RouteTopology":
        """Ablate learned preference while retaining the declared topology."""

        return RouteTopology(
            self.topology_id,
            0,
            tuple(
                CandidateRoute(
                    route.route_id,
                    route.context_id,
                    route.source,
                    route.target,
                )
                for route in self.routes
            ),
            self.generation + 1,
        )


@dataclass(frozen=True)
class RouteSelection:
    """A deterministic candidate preference; it does not dispatch or execute."""

    topology_id: str
    topology_version: int
    topology_generation: int
    context_id: str
    route_id: str
    transaction_id: str
    objective_id: str
    task_state_id: str
    task_state_version: int
    candidate_scores: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        for field_name in (
            "topology_id",
            "context_id",
            "route_id",
            "transaction_id",
            "objective_id",
            "task_state_id",
        ):
            _identifier(getattr(self, field_name), field_name)
        for field_name in (
            "topology_version",
            "topology_generation",
            "task_state_version",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PlasticRoutingValidationError(
                    f"{field_name} must be a non-negative integer"
                )
        if not isinstance(self.candidate_scores, (tuple, list)) or not self.candidate_scores:
            raise PlasticRoutingValidationError("candidate_scores must be non-empty")
        score_ids: list[str] = []
        for score in self.candidate_scores:
            if not isinstance(score, (tuple, list)) or len(score) != 2:
                raise PlasticRoutingValidationError(
                    "candidate_scores must contain route and weight pairs"
                )
            route_id, weight = score
            _identifier(route_id, "candidate score route_id")
            _bounded_weight(weight, "candidate score weight")
            score_ids.append(route_id)
        if len(set(score_ids)) != len(score_ids):
            raise PlasticRoutingValidationError("candidate_scores route ids must be unique")
        object.__setattr__(
            self,
            "candidate_scores",
            tuple((str(route_id), float(weight)) for route_id, weight in self.candidate_scores),
        )


@dataclass(frozen=True)
class SettlementRouteRecord:
    """A provenance-bound attempt to credit exactly one selected candidate route."""

    record_id: str
    selection: RouteSelection
    constitutional_trace: CycleTrace
    provenance: Mapping[str, Any] = field(default_factory=dict)
    grounded_execution: VerifiedGroundedExecution | None = None
    grounded_request: GroundedExecutionRequest | None = None
    grounded_verifier: GroundedExecutionVerifier | None = None

    def __post_init__(self) -> None:
        _identifier(self.record_id, "record_id")
        if not isinstance(self.selection, RouteSelection):
            raise PlasticRoutingValidationError("record requires a RouteSelection")
        if not isinstance(self.constitutional_trace, CycleTrace):
            raise PlasticRoutingValidationError("record requires a CycleTrace")
        if not isinstance(self.provenance, Mapping):
            raise PlasticRoutingValidationError("record provenance must be a mapping")
        grounded_inputs = (
            self.grounded_execution,
            self.grounded_request,
            self.grounded_verifier,
        )
        is_grounded_trace = (
            self.constitutional_trace.provenance.get("source")
            == "grounded_execution_verifier"
        )
        if is_grounded_trace != any(item is not None for item in grounded_inputs):
            raise PlasticRoutingValidationError(
                "grounded trace and grounded route inputs must agree"
            )
        if is_grounded_trace and not all(item is not None for item in grounded_inputs):
            raise PlasticRoutingValidationError(
                "grounded route record requires execution, request, and verifier"
            )
        object.__setattr__(self, "provenance", _freeze(self.provenance))

    @property
    def settlement(self) -> Settlement:
        """The settlement is read only from the validated constitutional trace."""

        return self.constitutional_trace.settlement

    @property
    def evidence(self) -> tuple[Evidence, ...]:
        """The evidence is read only from the validated constitutional trace."""

        return self.constitutional_trace.evidence


@dataclass(frozen=True)
class RouteLearningTrace:
    """Immutable audit trace of one allowed or withheld plasticity decision."""

    record_id: str
    settlement_id: str
    route_id: str
    disposition: str
    effect: str
    reason: str
    topology_before: int
    topology_after: int
    weight_before: float
    weight_after: float
    evidence_ids: tuple[str, ...] = ()
    lineage_id: str | None = None
    learning_update: LearningUpdate | None = None

    def __post_init__(self) -> None:
        for field_name in ("record_id", "settlement_id", "route_id"):
            _identifier(getattr(self, field_name), field_name)
        if self.disposition not in {"accepted", "withheld"}:
            raise PlasticRoutingValidationError("disposition must be accepted or withheld")
        if self.effect not in {"strengthen", "weaken", "none"}:
            raise PlasticRoutingValidationError("effect must be strengthen, weaken, or none")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise PlasticRoutingValidationError("reason must be a non-empty string")
        if self.topology_before < 0 or self.topology_after < 0:
            raise PlasticRoutingValidationError("topology versions must be non-negative")
        _bounded_weight(self.weight_before, "weight_before")
        _bounded_weight(self.weight_after, "weight_after")
        if abs(self.weight_after - self.weight_before) > MAX_WEIGHT_CHANGE + 1e-12:
            raise PlasticRoutingValidationError("one route update exceeded the fixed bound")
        if not isinstance(self.evidence_ids, (tuple, list)) or not all(
            isinstance(item, str) and item for item in self.evidence_ids
        ):
            raise PlasticRoutingValidationError("evidence_ids must be evidence identifiers")
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        if self.lineage_id is not None:
            _identifier(self.lineage_id, "lineage_id")
        if self.disposition == "accepted":
            if self.learning_update is None or self.lineage_id is None or not self.evidence_ids:
                raise PlasticRoutingValidationError(
                    "accepted learning requires lineage, update, and grounded evidence"
                )
        elif self.learning_update is not None or self.lineage_id is not None:
            raise PlasticRoutingValidationError("withheld learning cannot emit learning records")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "settlement_id": self.settlement_id,
            "route_id": self.route_id,
            "disposition": self.disposition,
            "effect": self.effect,
            "reason": self.reason,
            "topology_before": self.topology_before,
            "topology_after": self.topology_after,
            "weight_before": self.weight_before,
            "weight_after": self.weight_after,
            "evidence_ids": list(self.evidence_ids),
            "lineage_id": self.lineage_id,
            "learning_update": (
                self.learning_update.to_dict() if self.learning_update is not None else None
            ),
        }


def select_candidate_route(
    topology: RouteTopology,
    context_id: str,
    *,
    transaction_id: str,
    objective_id: str,
    task_state_id: str,
    task_state_version: int,
) -> RouteSelection:
    """Choose the highest-weight route with a stable route-id tie break."""

    if not isinstance(topology, RouteTopology):
        raise PlasticRoutingValidationError("selection requires a RouteTopology")
    _identifier(context_id, "context_id")
    candidates = sorted(
        (route for route in topology.routes if route.context_id == context_id),
        key=lambda route: (-route.weight, route.route_id),
    )
    if not candidates:
        raise PlasticRoutingValidationError("topology has no route for this context")
    selected = candidates[0]
    return RouteSelection(
        topology.topology_id,
        topology.version,
        topology.generation,
        context_id,
        selected.route_id,
        transaction_id,
        objective_id,
        task_state_id,
        task_state_version,
        tuple((route.route_id, route.weight) for route in candidates),
    )


def _route_for(topology: RouteTopology, route_id: str) -> CandidateRoute:
    for route in topology.routes:
        if route.route_id == route_id:
            return route
    raise PlasticRoutingValidationError("selected route does not exist in topology")


def _validate_record_binding(topology: RouteTopology, record: SettlementRouteRecord) -> CandidateRoute:
    if not isinstance(record, SettlementRouteRecord):
        raise PlasticRoutingValidationError("learning requires a SettlementRouteRecord")
    selection = record.selection
    if selection.topology_id != topology.topology_id:
        raise PlasticRoutingValidationError("selection topology identity is stale")
    if selection.topology_version != topology.version:
        raise PlasticRoutingValidationError("selection topology version is stale")
    if selection.topology_generation != topology.generation:
        raise PlasticRoutingValidationError("selection topology generation is stale")
    expected_selection = select_candidate_route(
        topology,
        selection.context_id,
        transaction_id=selection.transaction_id,
        objective_id=selection.objective_id,
        task_state_id=selection.task_state_id,
        task_state_version=selection.task_state_version,
    )
    if selection != expected_selection:
        raise PlasticRoutingValidationError(
            "record does not credit the deterministically selected route"
        )
    route = _route_for(topology, selection.route_id)
    if route.context_id != selection.context_id:
        raise PlasticRoutingValidationError("selection context does not match route")
    if record.settlement.settlement_id in topology.applied_settlement_ids:
        raise PlasticRoutingValidationError("settlement was already applied to topology")
    if len(topology.applied_settlement_ids) >= MAX_AUDIT_LINKS:
        raise PlasticRoutingValidationError(
            "topology settlement budget is exhausted; reset begins a new generation"
        )
    trace = record.constitutional_trace
    try:
        ConstitutionalCycle._validate_trace(trace)
    except CycleInvariantError as exc:
        raise PlasticRoutingValidationError(
            "record does not contain a valid constitutional trace"
        ) from exc
    if (
        trace.transaction_id != selection.transaction_id
        or trace.objective.objective_id != selection.objective_id
        or trace.states[4].state_id != selection.task_state_id
        or trace.states[4].version != selection.task_state_version
    ):
        raise PlasticRoutingValidationError(
            "selection identity does not match constitutional trace"
        )
    if trace.provenance.get("source") not in {
        "deterministic_fixture",
        "recorded_execution_replay",
        "grounded_execution_verifier",
    }:
        raise PlasticRoutingValidationError(
            "constitutional trace provenance does not name an accepted observation source"
        )
    required = {
        "transaction_id": selection.transaction_id,
        "objective_id": selection.objective_id,
        "task_state_id": selection.task_state_id,
        "task_state_version": selection.task_state_version,
        "route_id": selection.route_id,
        "settlement_id": record.settlement.settlement_id,
    }
    for key, expected in required.items():
        if record.provenance.get(key) != expected:
            raise PlasticRoutingValidationError(f"record provenance is missing or stale: {key}")
    evidence_ids = tuple(item.evidence_id for item in record.evidence)
    if tuple(record.provenance.get("evidence_ids", ())) != evidence_ids:
        raise PlasticRoutingValidationError("record provenance evidence ids do not match evidence")
    if tuple(record.settlement.evidence_ids) != evidence_ids:
        raise PlasticRoutingValidationError("settlement evidence ids do not match evidence")
    _validate_grounded_route_binding(trace, record)
    return route


def _validate_grounded_route_binding(
    trace: CycleTrace, record: SettlementRouteRecord
) -> None:
    """Require independently verified record lineage before grounded credit."""

    is_grounded_trace = (
        trace.provenance.get("source") == "grounded_execution_verifier"
    )
    grounded_inputs = (
        record.grounded_execution,
        record.grounded_request,
        record.grounded_verifier,
    )
    if not is_grounded_trace:
        if any(item is not None for item in grounded_inputs):
            raise PlasticRoutingValidationError(
                "non-grounded route record cannot carry grounded execution inputs"
            )
        return
    if not all(item is not None for item in grounded_inputs):
        raise PlasticRoutingValidationError(
            "grounded route record requires execution, request, and verifier"
        )
    assert record.grounded_execution is not None
    assert record.grounded_request is not None
    assert record.grounded_verifier is not None
    request = record.grounded_request
    if (
        request.transaction_id != trace.transaction_id
        or request.objective_id != trace.objective.objective_id
        or request.task_state_id != trace.states[4].state_id
        or request.task_state_version != trace.states[4].version
    ):
        raise PlasticRoutingValidationError(
            "grounded request identity does not match constitutional trace"
        )
    try:
        verified = record.grounded_verifier.verify(
            record.grounded_execution.record,
            request=request,
            authorized_state=trace.states[4],
        )
    except GroundedExecutionRejected as exc:
        raise PlasticRoutingValidationError(
            f"grounded route record cannot be independently reverified: {exc}"
        ) from exc
    if verified != record.grounded_execution:
        raise PlasticRoutingValidationError(
            "grounded route verification result changed during learning"
        )
    record_hash = record.grounded_execution.record.record_hash
    if trace.execution.observations.get("record_hash") != record_hash:
        raise PlasticRoutingValidationError(
            "grounded execution lineage does not match cycle observation"
        )
    if any(
        item.grade is not EvidenceGrade.GROUNDED
        or item.provenance.get("observation_origin") != "grounded_execution"
        or item.provenance.get("record_hash") != record_hash
        for item in record.evidence
    ):
        raise PlasticRoutingValidationError(
            "grounded route evidence does not match the verified record"
        )


def _is_grounded_evidence(evidence: tuple[Evidence, ...]) -> bool:
    return bool(evidence) and all(
        item.grade in {EvidenceGrade.OPERATIONAL, EvidenceGrade.GROUNDED}
        and item.execution_id is not None
        and item.provenance.get("observation_origin")
        in {"execution_result", "recorded_execution", "grounded_execution"}
        for item in evidence
    )


def apply_settlement_learning(
    topology: RouteTopology, record: SettlementRouteRecord
) -> tuple[RouteTopology, RouteLearningTrace]:
    """Apply one bounded route update, or explicitly withhold non-eligible credit."""

    if not isinstance(topology, RouteTopology):
        raise PlasticRoutingValidationError("learning requires a RouteTopology")
    route = _validate_record_binding(topology, record)
    settlement = record.settlement
    evidence_ids = tuple(item.evidence_id for item in record.evidence)

    if settlement.status != "settled":
        return topology, RouteLearningTrace(
            record.record_id, settlement.settlement_id, route.route_id, "withheld", "none",
            "settlement is not constitutionally settled", topology.version, topology.version,
            route.weight, route.weight,
        )
    if settlement.observed_outcome not in {"success", "failure"}:
        return topology, RouteLearningTrace(
            record.record_id, settlement.settlement_id, route.route_id, "withheld", "none",
            "only uncontradicted success or failure may affect a route",
            topology.version, topology.version, route.weight, route.weight,
        )
    if not _is_grounded_evidence(record.evidence):
        return topology, RouteLearningTrace(
            record.record_id, settlement.settlement_id, route.route_id, "withheld", "none",
            "settlement lacks grounded execution evidence",
            topology.version, topology.version, route.weight, route.weight,
        )

    effect = "strengthen" if settlement.observed_outcome == "success" else "weaken"
    direction = MAX_WEIGHT_CHANGE if effect == "strengthen" else -MAX_WEIGHT_CHANGE
    next_weight = min(MAX_WEIGHT, max(MIN_WEIGHT, route.weight + direction))
    lineage_id = f"{record.record_id}-lineage"
    next_route = replace(
        route,
        weight=next_weight,
        success_count=route.success_count + (1 if effect == "strengthen" else 0),
        failure_count=route.failure_count + (1 if effect == "weaken" else 0),
        settlement_ids=_audit_append(route.settlement_ids, settlement.settlement_id),
        evidence_ids=_audit_append(route.evidence_ids, evidence_ids[-1]),
        lineage_ids=_audit_append(route.lineage_ids, lineage_id),
    )
    next_topology = RouteTopology(
        topology.topology_id,
        topology.version + 1,
        tuple(next_route if item.route_id == route.route_id else item for item in topology.routes),
        topology.generation,
        topology.applied_settlement_ids + (settlement.settlement_id,),
    )
    update = LearningUpdate(
        f"{record.record_id}-learning",
        settlement.settlement_id,
        "candidate_route",
        route.route_id,
        change={
            "effect": effect,
            "weight_before": route.weight,
            "weight_after": next_weight,
            "topology_before": topology.version,
            "topology_after": next_topology.version,
            "lineage_id": lineage_id,
        },
        evidence_ids=evidence_ids,
        disposition="accepted",
    )
    return next_topology, RouteLearningTrace(
        record.record_id,
        settlement.settlement_id,
        route.route_id,
        "accepted",
        effect,
        "settled grounded execution outcome",
        topology.version,
        next_topology.version,
        route.weight,
        next_weight,
        evidence_ids,
        lineage_id,
        update,
    )


def replay_settlement_learning(
    initial_topology: RouteTopology,
    history: tuple[SettlementRouteRecord, ...],
) -> tuple[RouteTopology, tuple[RouteLearningTrace, ...]]:
    """Replay immutable learning history through the same bounded reducer."""

    if not isinstance(initial_topology, RouteTopology):
        raise PlasticRoutingValidationError("replay requires a RouteTopology")
    if not isinstance(history, (tuple, list)) or not all(
        isinstance(item, SettlementRouteRecord) for item in history
    ):
        raise PlasticRoutingValidationError(
            "replay history must contain SettlementRouteRecord records"
        )
    topology = initial_topology
    traces: list[RouteLearningTrace] = []
    for record in history:
        topology, trace = apply_settlement_learning(topology, record)
        traces.append(trace)
    return topology, tuple(traces)


__all__ = [
    "BASELINE_WEIGHT",
    "MAX_AUDIT_LINKS",
    "MAX_WEIGHT",
    "MAX_WEIGHT_CHANGE",
    "MIN_WEIGHT",
    "CandidateRoute",
    "PlasticRoutingValidationError",
    "RouteLearningTrace",
    "RouteSelection",
    "RouteTopology",
    "SettlementRouteRecord",
    "apply_settlement_learning",
    "replay_settlement_learning",
    "select_candidate_route",
]