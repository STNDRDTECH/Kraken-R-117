"""Bounded, grounded adaptive substrate for Kraken-R Stage 10.

This module is a pure reducer over caller-owned immutable state.  It adds
ordinary candidate adaptation without creating a daemon, bus, store, provider,
controller, executor, or actuator.  Every state-changing operation is bound to
one independently reverified constitutional settlement and consumes one finite
generation budget.

Orzhaal is intentionally only a disposable fork for comparing adaptation-rule
proposals.  Its result has no promotion or canonical-write operation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any, Mapping

from .contracts import Authority, EvidenceGrade, Signal
from .grounded_execution import (
    EpistemicOutcomeClass,
    GroundedExecutionRejected,
    GroundedExecutionRequest,
    GroundedExecutionVerifier,
    VerifiedGroundedExecution,
)
from .nervous_system import make_bound_signal
from .plastic_routing import (
    MAX_WEIGHT,
    MIN_WEIGHT,
    CandidateRoute,
    RouteTopology,
    SettlementRouteRecord,
    apply_settlement_learning,
)


class AdaptiveSubstrateValidationError(ValueError):
    """Raised when an adaptive input crosses the candidate boundary."""


MAX_ADAPTIVE_UPDATES = 8
MAX_ADAPTIVE_GENERATIONS = 8
MAX_CONNECTIONS = 8
MAX_CONNECTION_DEGREE = 2
MAX_CONNECTION_CHANGE = 0.10
MAX_ROUTE_DECAY = 0.05
MAX_ROUTE_RECOVERY = 0.05
MAX_ADAPTIVE_ROUTE_WEIGHT = 0.65
MAX_TACTIC_STREAK = 3
MAX_ADAPTIVE_AUDIT = 32
MAX_CHECKPOINTS = 4
MAX_TRACKED_RECORDS = 64
BASELINE_CONNECTION_WEIGHT = 0.50


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(
        character.isspace() for character in value
    ):
        raise AdaptiveSubstrateValidationError(
            f"{field_name} must be a non-empty identifier without whitespace"
        )
    return value


def _bounded(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdaptiveSubstrateValidationError(f"{field_name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise AdaptiveSubstrateValidationError(
            f"{field_name} must be between 0 and 1"
        )
    return value


def _weight(value: float, field_name: str) -> float:
    value = _bounded(value, field_name)
    if not MIN_WEIGHT <= value <= MAX_WEIGHT:
        raise AdaptiveSubstrateValidationError(
            f"{field_name} must be from {MIN_WEIGHT} through {MAX_WEIGHT}"
        )
    return value


def _positive_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AdaptiveSubstrateValidationError(
            f"{field_name} must be a non-negative integer"
        )


def _connection_id(source: str, target: str) -> str:
    digest = hashlib.sha256(f"{source}->{target}".encode("utf-8")).hexdigest()[:16]
    return f"candidate-connection-{digest}"


@dataclass(frozen=True)
class HomeostaticSnapshot:
    """Bounded current pressures; it is advisory and never learning authority."""

    snapshot_id: str
    transaction_id: str
    objective_id: str
    task_state_id: str
    task_state_version: int
    contradiction: float = 0.0
    uncertainty: float = 0.0
    repeated_failure: float = 0.0
    novelty: float = 0.0
    resource_expenditure: float = 0.0

    def __post_init__(self) -> None:
        for value, name in (
            (self.snapshot_id, "snapshot_id"),
            (self.transaction_id, "transaction_id"),
            (self.objective_id, "objective_id"),
            (self.task_state_id, "task_state_id"),
        ):
            _identifier(value, name)
        _positive_integer(self.task_state_version, "task_state_version")
        if self.task_state_version < 1:
            raise AdaptiveSubstrateValidationError(
                "task_state_version must be positive"
            )
        for name in (
            "contradiction",
            "uncertainty",
            "repeated_failure",
            "novelty",
            "resource_expenditure",
        ):
            object.__setattr__(self, name, _bounded(getattr(self, name), name))

    @property
    def pressure(self) -> float:
        """Return a deterministic mean pressure, not a confidence score."""

        return round(
            (
                self.contradiction
                + self.uncertainty
                + self.repeated_failure
                + self.novelty
                + self.resource_expenditure
            )
            / 5.0,
            6,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "transaction_id": self.transaction_id,
            "objective_id": self.objective_id,
            "task_state_id": self.task_state_id,
            "task_state_version": self.task_state_version,
            "contradiction": self.contradiction,
            "uncertainty": self.uncertainty,
            "repeated_failure": self.repeated_failure,
            "novelty": self.novelty,
            "resource_expenditure": self.resource_expenditure,
            "pressure": self.pressure,
            "advisory_only": True,
        }

    def signals(self) -> tuple[Signal, ...]:
        """Create declared bounded signals; callers may not treat them as facts."""

        values = (
            ("contradiction", self.contradiction),
            ("uncertainty", self.uncertainty),
            ("repeated_failure", self.repeated_failure),
            ("novelty", self.novelty),
            ("resource_expenditure", self.resource_expenditure),
        )
        return tuple(
            make_bound_signal(
                f"{self.snapshot_id}-{name}",
                f"candidate.homeostasis.{name}",
                transaction_id=self.transaction_id,
                objective_id=self.objective_id,
                task_state_id=self.task_state_id,
                task_state_version=self.task_state_version,
                payload={"bounded_value": value, "advisory_only": True},
                cause=self.snapshot_id,
            )
            for name, value in values
        )


@dataclass(frozen=True)
class CandidateConnection:
    """One bounded organizational edge; it is never dispatchable."""

    connection_id: str
    source: str
    target: str
    weight: float = BASELINE_CONNECTION_WEIGHT
    generation: int = 0

    def __post_init__(self) -> None:
        _identifier(self.source, "source")
        _identifier(self.target, "target")
        if self.source == self.target:
            raise AdaptiveSubstrateValidationError("a connection cannot target itself")
        _identifier(self.connection_id, "connection_id")
        if self.connection_id != _connection_id(self.source, self.target):
            raise AdaptiveSubstrateValidationError(
                "connection identity is not the deterministic source-target identity"
            )
        _weight(self.weight, "connection weight")
        _positive_integer(self.generation, "generation")
        if self.generation > MAX_ADAPTIVE_GENERATIONS:
            raise AdaptiveSubstrateValidationError(
                "adaptive generation limit is exhausted"
            )

    @classmethod
    def for_route(cls, route: CandidateRoute, *, generation: int) -> "CandidateConnection":
        return cls(
            _connection_id(route.source, route.target),
            route.source,
            route.target,
            generation=generation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "generation": self.generation,
            "dispatchable": False,
        }


@dataclass(frozen=True)
class CandidateTactic:
    """A predeclared tactic label, not a goal or action selector."""

    tactic_id: str
    ordinal: int

    def __post_init__(self) -> None:
        _identifier(self.tactic_id, "tactic_id")
        _positive_integer(self.ordinal, "ordinal")


@dataclass(frozen=True)
class AdaptiveCheckpoint:
    """A bounded pre-update snapshot used only for safe candidate rollback."""

    checkpoint_id: str
    generation: int
    route_topology: RouteTopology
    connections: tuple[CandidateConnection, ...]
    active_tactic_id: str
    tactic_scores: tuple[tuple[str, float], ...]
    applied_record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.checkpoint_id, "checkpoint_id")
        _positive_integer(self.generation, "generation")
        if not isinstance(self.route_topology, RouteTopology):
            raise AdaptiveSubstrateValidationError("checkpoint route topology is invalid")
        object.__setattr__(self, "connections", tuple(self.connections))
        object.__setattr__(self, "tactic_scores", tuple(self.tactic_scores))
        object.__setattr__(self, "applied_record_ids", tuple(self.applied_record_ids))


@dataclass(frozen=True)
class AdaptiveAudit:
    """Immutable lineage for one bounded adaptive reducer decision."""

    audit_id: str
    record_id: str
    operation: str
    generation_before: int
    generation_after: int
    route_id: str
    connection_id: str | None
    weight_before: float | None
    weight_after: float | None
    reason: str
    evidence_ids: tuple[str, ...]
    epistemic_class: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.audit_id, "audit_id"),
            (self.record_id, "record_id"),
            (self.route_id, "route_id"),
        ):
            _identifier(value, name)
        _positive_integer(self.generation_before, "generation_before")
        _positive_integer(self.generation_after, "generation_after")
        if self.operation not in {
            "strengthen",
            "weaken",
            "decay",
            "recover",
            "form_connection",
            "weaken_connection",
            "switch_tactic",
            "rollback",
            "invalidate_evidence",
        }:
            raise AdaptiveSubstrateValidationError("unsupported adaptive operation")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise AdaptiveSubstrateValidationError("audit reason is required")
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        if not all(isinstance(item, str) and item for item in self.evidence_ids):
            raise AdaptiveSubstrateValidationError("audit evidence ids are invalid")
        if self.operation == "rollback":
            if self.epistemic_class is not None:
                raise AdaptiveSubstrateValidationError(
                    "rollback audit cannot claim an execution epistemic class"
                )
        elif self.epistemic_class not in {
            EpistemicOutcomeClass.TASK_SUCCESS.value,
            EpistemicOutcomeClass.TASK_FAILURE.value,
        }:
            raise AdaptiveSubstrateValidationError(
                "adaptive audit requires a creditable verified epistemic class"
            )
        if self.weight_before is not None:
            _weight(self.weight_before, "weight_before")
        if self.weight_after is not None:
            _weight(self.weight_after, "weight_after")
        if (
            self.weight_before is not None
            and self.weight_after is not None
            and abs(self.weight_after - self.weight_before)
            > MAX_CONNECTION_CHANGE + 1e-12
        ):
            raise AdaptiveSubstrateValidationError("adaptive weight change exceeded bound")

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "record_id": self.record_id,
            "operation": self.operation,
            "generation_before": self.generation_before,
            "generation_after": self.generation_after,
            "route_id": self.route_id,
            "connection_id": self.connection_id,
            "weight_before": self.weight_before,
            "weight_after": self.weight_after,
            "reason": self.reason,
            "evidence_ids": list(self.evidence_ids),
            "epistemic_class": self.epistemic_class,
        }


@dataclass(frozen=True)
class AdaptiveState:
    """Caller-owned immutable bounded adaptive state."""

    substrate_id: str
    generation: int
    updates_applied: int
    route_topology: RouteTopology
    connections: tuple[CandidateConnection, ...]
    tactics: tuple[CandidateTactic, ...]
    active_tactic_id: str
    tactic_scores: tuple[tuple[str, float], ...]
    applied_record_ids: tuple[str, ...] = ()
    audits: tuple[AdaptiveAudit, ...] = ()
    checkpoints: tuple[AdaptiveCheckpoint, ...] = ()
    failure_streak: int = 0
    tactic_streak: int = 0
    invalidated_record_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.substrate_id, "substrate_id")
        _positive_integer(self.generation, "generation")
        if self.generation > MAX_ADAPTIVE_GENERATIONS:
            raise AdaptiveSubstrateValidationError(
                "adaptive generation limit is exhausted"
            )
        if not 0 <= self.updates_applied <= MAX_ADAPTIVE_UPDATES:
            raise AdaptiveSubstrateValidationError(
                "updates_applied exceeds the generation update budget"
            )
        if not isinstance(self.route_topology, RouteTopology):
            raise AdaptiveSubstrateValidationError("route_topology is invalid")
        if any(
            route.weight > MAX_ADAPTIVE_ROUTE_WEIGHT
            for route in self.route_topology.routes
        ):
            raise AdaptiveSubstrateValidationError(
                "route preference exceeds the adaptive anti-monopoly bound"
            )
        connections = tuple(self.connections)
        if len(connections) > MAX_CONNECTIONS:
            raise AdaptiveSubstrateValidationError("connection count exceeds the hard limit")
        if any(not isinstance(item, CandidateConnection) for item in connections):
            raise AdaptiveSubstrateValidationError("connections are invalid")
        if len({item.connection_id for item in connections}) != len(connections):
            raise AdaptiveSubstrateValidationError("connection identities must be unique")
        degree: dict[str, int] = {}
        for item in connections:
            degree[item.source] = degree.get(item.source, 0) + 1
            degree[item.target] = degree.get(item.target, 0) + 1
        if any(value > MAX_CONNECTION_DEGREE for value in degree.values()):
            raise AdaptiveSubstrateValidationError("connection degree exceeds the hard limit")
        tactics = tuple(self.tactics)
        if not tactics or any(not isinstance(item, CandidateTactic) for item in tactics):
            raise AdaptiveSubstrateValidationError("at least one tactic is required")
        if len({item.tactic_id for item in tactics}) != len(tactics):
            raise AdaptiveSubstrateValidationError("tactic identities must be unique")
        tactic_ids = {item.tactic_id for item in tactics}
        _identifier(self.active_tactic_id, "active_tactic_id")
        if self.active_tactic_id not in tactic_ids:
            raise AdaptiveSubstrateValidationError("active tactic is not declared")
        scores = tuple(self.tactic_scores)
        if {item[0] for item in scores} != tactic_ids or len(scores) != len(tactic_ids):
            raise AdaptiveSubstrateValidationError("tactic scores must cover declared tactics")
        for tactic_id, score in scores:
            _identifier(tactic_id, "tactic score id")
            _bounded(score, "tactic score")
        for name, values, limit in (
            ("applied_record_ids", self.applied_record_ids, MAX_TRACKED_RECORDS),
            ("invalidated_record_ids", self.invalidated_record_ids, MAX_TRACKED_RECORDS),
            ("audits", self.audits, MAX_ADAPTIVE_AUDIT),
            ("checkpoints", self.checkpoints, MAX_CHECKPOINTS),
        ):
            if len(values) > limit:
                raise AdaptiveSubstrateValidationError(f"{name} exceeds its fixed retention")
        if len(set(self.applied_record_ids)) != len(self.applied_record_ids):
            raise AdaptiveSubstrateValidationError("applied record identities must not repeat")
        if any(not isinstance(item, str) or not item for item in self.applied_record_ids):
            raise AdaptiveSubstrateValidationError("applied record identities are invalid")
        if len(set(self.invalidated_record_ids)) != len(self.invalidated_record_ids):
            raise AdaptiveSubstrateValidationError(
                "invalidated record identities must not repeat"
            )
        if any(
            not isinstance(item, str) or not item
            for item in self.invalidated_record_ids
        ):
            raise AdaptiveSubstrateValidationError(
                "invalidated record identities are invalid"
            )
        if not set(self.invalidated_record_ids).issubset(
            set(self.applied_record_ids)
        ):
            raise AdaptiveSubstrateValidationError(
                "invalidated identities must remain applied identities"
            )
        if any(not isinstance(item, AdaptiveAudit) for item in self.audits):
            raise AdaptiveSubstrateValidationError("adaptive audit entries are invalid")
        if any(not isinstance(item, AdaptiveCheckpoint) for item in self.checkpoints):
            raise AdaptiveSubstrateValidationError("adaptive checkpoints are invalid")
        _positive_integer(self.failure_streak, "failure_streak")
        if self.tactic_streak < 0 or self.tactic_streak > MAX_TACTIC_STREAK:
            raise AdaptiveSubstrateValidationError(
                "tactic streak exceeds the anti-lock-in bound"
            )
        object.__setattr__(self, "connections", connections)
        object.__setattr__(self, "tactics", tactics)
        object.__setattr__(self, "tactic_scores", scores)
        object.__setattr__(self, "applied_record_ids", tuple(self.applied_record_ids))
        object.__setattr__(self, "audits", tuple(self.audits))
        object.__setattr__(self, "checkpoints", tuple(self.checkpoints))

    @classmethod
    def fixture(cls, substrate_id: str = "candidate-adaptive-substrate") -> "AdaptiveState":
        tactics = (CandidateTactic("inspect", 0), CandidateTactic("alternate", 1))
        return cls(
            substrate_id,
            0,
            0,
            RouteTopology.fixture(f"{substrate_id}-topology"),
            (),
            tactics,
            tactics[0].tactic_id,
            tuple((item.tactic_id, 0.50) for item in tactics),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "substrate_id": self.substrate_id,
            "generation": self.generation,
            "updates_applied": self.updates_applied,
            "route_topology": {
                "topology_id": self.route_topology.topology_id,
                "version": self.route_topology.version,
                "generation": self.route_topology.generation,
                "routes": [
                    {
                        "route_id": route.route_id,
                        "context_id": route.context_id,
                        "source": route.source,
                        "target": route.target,
                        "weight": route.weight,
                        "success_count": route.success_count,
                        "failure_count": route.failure_count,
                    }
                    for route in self.route_topology.routes
                ],
            },
            "connections": [item.to_dict() for item in self.connections],
            "tactics": [
                {"tactic_id": item.tactic_id, "ordinal": item.ordinal}
                for item in self.tactics
            ],
            "active_tactic_id": self.active_tactic_id,
            "tactic_scores": [list(item) for item in self.tactic_scores],
            "applied_record_ids": list(self.applied_record_ids),
            "audits": [item.to_dict() for item in self.audits],
            "checkpoint_ids": [item.checkpoint_id for item in self.checkpoints],
            "failure_streak": self.failure_streak,
            "tactic_streak": self.tactic_streak,
            "invalidated_record_ids": list(self.invalidated_record_ids),
            "authority": Authority.KRAKEN_CANDIDATE.value,
        }


def _validate_pressure(
    pressure: HomeostaticSnapshot | None, record: SettlementRouteRecord
) -> None:
    if pressure is None:
        return
    trace = record.constitutional_trace
    if (
        pressure.transaction_id != trace.transaction_id
        or pressure.objective_id != trace.objective.objective_id
        or pressure.task_state_id != trace.states[4].state_id
        or pressure.task_state_version != trace.states[4].version
    ):
        raise AdaptiveSubstrateValidationError(
            "homeostatic pressure is not bound to the grounded authorized state"
        )


def _require_grounded_credit(
    state: AdaptiveState, record: SettlementRouteRecord
) -> tuple[CandidateRoute, tuple[str, ...], EpistemicOutcomeClass]:
    """Require exact constitutional, execution, state, and hash lineage."""

    if not isinstance(record, SettlementRouteRecord):
        raise AdaptiveSubstrateValidationError("adaptive credit requires a route record")
    trace = record.constitutional_trace
    if trace.provenance.get("source") != "grounded_execution_verifier":
        raise AdaptiveSubstrateValidationError(
            "adaptive credit requires a grounded constitutional trace"
        )
    if (
        trace.states[4].phase != "authorized"
        or trace.states[4].authority is not Authority.KRAKEN_CANDIDATE
    ):
        raise AdaptiveSubstrateValidationError(
            "adaptive credit requires the authorized candidate state"
        )
    if record.grounded_execution is None or record.grounded_request is None:
        raise AdaptiveSubstrateValidationError(
            "adaptive credit requires a sealed request and verified execution"
        )
    if record.grounded_verifier is None:
        raise AdaptiveSubstrateValidationError(
            "adaptive credit requires an independent verifier"
        )
    execution = record.grounded_execution
    request = record.grounded_request
    verifier = record.grounded_verifier
    if not isinstance(execution, VerifiedGroundedExecution):
        raise AdaptiveSubstrateValidationError("verified execution input is invalid")
    if not isinstance(request, GroundedExecutionRequest):
        raise AdaptiveSubstrateValidationError("grounded request input is invalid")
    try:
        verified = verifier.verify(
            execution.record,
            request=request,
            authorized_state=trace.states[4],
        )
    except GroundedExecutionRejected as exc:
        raise AdaptiveSubstrateValidationError(
            f"grounded execution could not be independently reverified: {exc}"
        ) from exc
    if verified != execution:
        raise AdaptiveSubstrateValidationError(
            "grounded execution changed during adaptive validation"
        )
    epistemic_class = execution.epistemic_class
    if epistemic_class not in {
        EpistemicOutcomeClass.TASK_SUCCESS,
        EpistemicOutcomeClass.TASK_FAILURE,
    }:
        raise AdaptiveSubstrateValidationError(
            "non-creditable grounded execution class cannot alter adaptive state"
        )
    if (
        trace.execution.observations.get("epistemic_class") != epistemic_class.value
        or trace.provenance.get("epistemic_class") != epistemic_class.value
        or execution.record.provenance.child_runtime_verified is not True
        or not execution.record.provenance.child_runtime_fingerprint
    ):
        raise AdaptiveSubstrateValidationError(
            "adaptive credit requires matching verified child-runtime epistemic provenance"
        )
    if execution.record.input_hash != request.input_hash:
        raise AdaptiveSubstrateValidationError(
            "grounded record input hash does not match the sealed request"
        )
    if trace.execution.observations.get("record_hash") != execution.record.record_hash:
        raise AdaptiveSubstrateValidationError(
            "grounded record hash does not match the constitutional trace"
        )
    if trace.execution.observations.get("input_hash") != request.input_hash:
        raise AdaptiveSubstrateValidationError(
            "grounded request hash does not match the constitutional trace"
        )
    if trace.settlement.status != "settled" or trace.settlement.observed_outcome not in {
        "success",
        "failure",
    }:
        raise AdaptiveSubstrateValidationError(
            "adaptive credit requires a settled success or failure"
        )
    evidence_ids = tuple(item.evidence_id for item in trace.evidence)
    if not evidence_ids or any(
        item.grade is not EvidenceGrade.GROUNDED
        or item.provenance.get("record_hash") != execution.record.record_hash
        or item.provenance.get("observation_origin") != "grounded_execution"
        for item in trace.evidence
    ):
        raise AdaptiveSubstrateValidationError(
            "adaptive credit requires grounded evidence for the same record hash"
        )
    try:
        _, route_trace = apply_settlement_learning(state.route_topology, record)
    except ValueError as exc:
        raise AdaptiveSubstrateValidationError(
            f"grounded route binding is invalid: {exc}"
        ) from exc
    if route_trace.disposition != "accepted":
        raise AdaptiveSubstrateValidationError(
            "grounded settlement did not produce eligible route credit"
        )
    route = next(
        item for item in state.route_topology.routes
        if item.route_id == record.selection.route_id
    )
    return route, evidence_ids, epistemic_class


def _checkpoint(state: AdaptiveState) -> AdaptiveCheckpoint:
    checkpoint_id = (
        f"{state.substrate_id}-checkpoint-{state.generation}-"
        f"{state.route_topology.version}-{state.updates_applied}"
    )
    return AdaptiveCheckpoint(
        checkpoint_id,
        state.generation,
        state.route_topology,
        state.connections,
        state.active_tactic_id,
        state.tactic_scores,
        state.applied_record_ids,
    )


def _prepare_update(
    state: AdaptiveState, record_id: str
) -> tuple[AdaptiveCheckpoint, int]:
    if record_id in state.applied_record_ids:
        raise AdaptiveSubstrateValidationError(
            "adaptive record identity was already applied, including after rollback"
        )
    if state.updates_applied >= MAX_ADAPTIVE_UPDATES:
        raise AdaptiveSubstrateValidationError(
            "adaptive generation update budget is exhausted"
        )
    if len(state.applied_record_ids) >= MAX_TRACKED_RECORDS:
        raise AdaptiveSubstrateValidationError(
            "adaptive record identity retention is exhausted; reset is required"
        )
    _identifier(record_id, "record_id")
    return _checkpoint(state), state.updates_applied + 1


def _reject_duplicate_record(state: AdaptiveState, record: object) -> None:
    """Reject a consumed identity before any stale-binding detail can mask it."""

    if not isinstance(record, SettlementRouteRecord):
        raise AdaptiveSubstrateValidationError("adaptive credit requires a route record")
    if record.record_id in state.applied_record_ids:
        raise AdaptiveSubstrateValidationError(
            "adaptive record identity was already applied, including after rollback"
        )


def _finish_update(
    state: AdaptiveState,
    checkpoint: AdaptiveCheckpoint,
    record_id: str,
    operation: str,
    *,
    route_id: str,
    connection_id: str | None = None,
    weight_before: float | None = None,
    weight_after: float | None = None,
    reason: str,
    evidence_ids: tuple[str, ...],
    epistemic_class: EpistemicOutcomeClass,
    **changes: Any,
) -> tuple[AdaptiveState, AdaptiveAudit]:
    next_generation = state.generation
    audit = AdaptiveAudit(
        f"{state.substrate_id}-audit-{state.generation}-{state.updates_applied + 1}",
        record_id,
        operation,
        state.generation,
        next_generation,
        route_id,
        connection_id,
        weight_before,
        weight_after,
        reason,
        evidence_ids,
        epistemic_class.value,
    )
    next_checkpoints = (state.checkpoints + (checkpoint,))[-MAX_CHECKPOINTS:]
    next_state = replace(
        state,
        audits=(state.audits + (audit,))[-MAX_ADAPTIVE_AUDIT:],
        checkpoints=next_checkpoints,
        applied_record_ids=state.applied_record_ids + (record_id,),
        updates_applied=state.updates_applied + 1,
        **changes,
    )
    return next_state, audit


def _replace_route(
    topology: RouteTopology,
    route: CandidateRoute,
    *,
    weight: float,
    settlement_id: str,
) -> RouteTopology:
    return RouteTopology(
        topology.topology_id,
        topology.version + 1,
        tuple(replace(item, weight=weight) if item.route_id == route.route_id else item
              for item in topology.routes),
        topology.generation,
        topology.applied_settlement_ids + (settlement_id,),
    )


def apply_grounded_adaptation(
    state: AdaptiveState,
    record: SettlementRouteRecord,
    *,
    operation: str = "credit",
    pressure: HomeostaticSnapshot | None = None,
) -> tuple[AdaptiveState, AdaptiveAudit]:
    """Apply one bounded grounded route adaptation.

    ``credit`` strengthens a settled success and weakens a settled failure.
    Decay and recovery are explicit, bounded alternatives and never inferred
    from a pressure snapshot or confidence value.
    """

    if not isinstance(state, AdaptiveState):
        raise AdaptiveSubstrateValidationError("adaptation requires AdaptiveState")
    _reject_duplicate_record(state, record)
    _validate_pressure(pressure, record)
    route, evidence_ids, epistemic_class = _require_grounded_credit(state, record)
    if operation == "credit":
        operation = (
            "strengthen"
            if record.settlement.observed_outcome == "success"
            else "weaken"
        )
    if operation not in {"strengthen", "weaken", "decay", "recover"}:
        raise AdaptiveSubstrateValidationError("operation is not a route adaptation")
    outcome = record.settlement.observed_outcome
    permitted_outcomes = {
        "strengthen": "success",
        "weaken": "failure",
        "decay": "success",
        "recover": "success",
    }
    if outcome != permitted_outcomes[operation]:
        raise AdaptiveSubstrateValidationError(
            f"{operation} cannot contradict the settled grounded {outcome} outcome"
        )
    checkpoint, _ = _prepare_update(state, record.record_id)
    if operation == "strengthen":
        next_weight = min(
            MAX_ADAPTIVE_ROUTE_WEIGHT, route.weight + MAX_CONNECTION_CHANGE
        )
    elif operation == "weaken":
        next_weight = max(MIN_WEIGHT, route.weight - MAX_CONNECTION_CHANGE)
    elif operation == "decay":
        delta = min(MAX_ROUTE_DECAY, abs(route.weight - 0.50))
        next_weight = route.weight - delta if route.weight > 0.50 else route.weight + delta
    else:
        delta = min(MAX_ROUTE_RECOVERY, abs(route.weight - 0.50))
        next_weight = route.weight + delta if route.weight < 0.50 else route.weight - delta
    next_weight = min(MAX_ADAPTIVE_ROUTE_WEIGHT, round(next_weight, 6))
    next_topology = _replace_route(
        state.route_topology,
        route,
        weight=next_weight,
        settlement_id=record.settlement.settlement_id,
    )
    next_state, audit = _finish_update(
        state,
        checkpoint,
        record.record_id,
        operation,
        route_id=route.route_id,
        weight_before=route.weight,
        weight_after=next_weight,
        reason=f"settled grounded {outcome} outcome",
        evidence_ids=evidence_ids,
        epistemic_class=epistemic_class,
        route_topology=next_topology,
        failure_streak=(
            state.failure_streak + 1
            if outcome == "failure"
            else 0
        ),
        tactic_streak=0 if outcome == "failure" else min(
            MAX_TACTIC_STREAK, state.tactic_streak + 1
        ),
    )
    return next_state, audit


def form_grounded_connection(
    state: AdaptiveState, record: SettlementRouteRecord
) -> tuple[AdaptiveState, AdaptiveAudit]:
    """Form at most one deterministic candidate edge after grounded success."""

    if not isinstance(state, AdaptiveState):
        raise AdaptiveSubstrateValidationError("connection formation requires AdaptiveState")
    _reject_duplicate_record(state, record)
    route, evidence_ids, epistemic_class = _require_grounded_credit(state, record)
    if record.settlement.observed_outcome != "success":
        raise AdaptiveSubstrateValidationError(
            "candidate connections form only after grounded success"
        )
    connection_id = _connection_id(route.source, route.target)
    if any(item.connection_id == connection_id for item in state.connections):
        raise AdaptiveSubstrateValidationError("candidate connection identity already exists")
    if len(state.connections) >= MAX_CONNECTIONS:
        raise AdaptiveSubstrateValidationError("candidate connection count limit reached")
    degree: dict[str, int] = {}
    for item in state.connections:
        degree[item.source] = degree.get(item.source, 0) + 1
        degree[item.target] = degree.get(item.target, 0) + 1
    if (
        degree.get(route.source, 0) >= MAX_CONNECTION_DEGREE
        or degree.get(route.target, 0) >= MAX_CONNECTION_DEGREE
    ):
        raise AdaptiveSubstrateValidationError("candidate connection degree limit reached")
    checkpoint, _ = _prepare_update(state, record.record_id)
    connection = CandidateConnection.for_route(route, generation=state.generation)
    next_state, audit = _finish_update(
        state,
        checkpoint,
        record.record_id,
        "form_connection",
        route_id=route.route_id,
        connection_id=connection.connection_id,
        reason="settled grounded success formed one bounded candidate edge",
        evidence_ids=evidence_ids,
        epistemic_class=epistemic_class,
        connections=state.connections + (connection,),
    )
    return next_state, audit


def weaken_grounded_connection(
    state: AdaptiveState, record: SettlementRouteRecord, connection_id: str
) -> tuple[AdaptiveState, AdaptiveAudit]:
    """Weaken one local candidate edge after grounded failure; remove at floor."""

    if not isinstance(state, AdaptiveState):
        raise AdaptiveSubstrateValidationError("connection weakening requires AdaptiveState")
    _reject_duplicate_record(state, record)
    route, evidence_ids, epistemic_class = _require_grounded_credit(state, record)
    if record.settlement.observed_outcome != "failure":
        raise AdaptiveSubstrateValidationError(
            "candidate connections weaken only after grounded failure"
        )
    _identifier(connection_id, "connection_id")
    connection = next(
        (item for item in state.connections if item.connection_id == connection_id),
        None,
    )
    if connection is None:
        raise AdaptiveSubstrateValidationError("candidate connection does not exist")
    if connection.connection_id != _connection_id(route.source, route.target):
        raise AdaptiveSubstrateValidationError(
            "grounded failure can weaken only the selected route connection"
        )
    checkpoint, _ = _prepare_update(state, record.record_id)
    next_weight = max(MIN_WEIGHT, round(connection.weight - MAX_CONNECTION_CHANGE, 6))
    next_connections = tuple(
        replace(item, weight=next_weight)
        if item.connection_id == connection_id
        else item
        for item in state.connections
    )
    next_state, audit = _finish_update(
        state,
        checkpoint,
        record.record_id,
        "weaken_connection",
        route_id=route.route_id,
        connection_id=connection_id,
        weight_before=connection.weight,
        weight_after=next_weight,
        reason="settled grounded failure weakened one bounded candidate edge",
        evidence_ids=evidence_ids,
        epistemic_class=epistemic_class,
        connections=next_connections,
    )
    return next_state, audit


def switch_grounded_tactic(
    state: AdaptiveState,
    record: SettlementRouteRecord,
    pressure: HomeostaticSnapshot,
) -> tuple[AdaptiveState, AdaptiveAudit]:
    """Switch deterministically only on grounded outcomes and bound pressure."""

    if not isinstance(state, AdaptiveState):
        raise AdaptiveSubstrateValidationError("tactic switching requires AdaptiveState")
    _reject_duplicate_record(state, record)
    _validate_pressure(pressure, record)
    _, evidence_ids, epistemic_class = _require_grounded_credit(state, record)
    checkpoint, _ = _prepare_update(state, record.record_id)
    ordered = sorted(state.tactics, key=lambda item: (item.ordinal, item.tactic_id))
    current_index = next(
        index for index, item in enumerate(ordered)
        if item.tactic_id == state.active_tactic_id
    )
    should_switch = (
        record.settlement.observed_outcome == "failure"
        or pressure.pressure >= 0.70
        or state.tactic_streak >= MAX_TACTIC_STREAK
    )
    target = (
        ordered[(current_index + 1) % len(ordered)]
        if should_switch and len(ordered) > 1
        else ordered[current_index]
    )
    scores = dict(state.tactic_scores)
    if should_switch and target.tactic_id != state.active_tactic_id:
        scores[target.tactic_id] = min(1.0, round(scores[target.tactic_id] + 0.05, 6))
    next_state, audit = _finish_update(
        state,
        checkpoint,
        record.record_id,
        "switch_tactic",
        route_id=record.selection.route_id,
        reason=(
            "grounded failure or bounded pressure selected the next predeclared tactic"
            if should_switch
            else "grounded outcome and bounded pressure retained the current tactic"
        ),
        evidence_ids=evidence_ids,
        epistemic_class=epistemic_class,
        active_tactic_id=target.tactic_id,
        tactic_scores=tuple((item.tactic_id, scores[item.tactic_id]) for item in ordered),
        tactic_streak=(
            0
            if target.tactic_id != state.active_tactic_id
            else min(MAX_TACTIC_STREAK, state.tactic_streak + 1)
        ),
    )
    return next_state, audit


def rollback_adaptive_state(
    state: AdaptiveState, checkpoint_id: str
) -> tuple[AdaptiveState, AdaptiveAudit]:
    """Restore a prior checkpoint while retaining duplicate identities.

    Rollback starts a new generation, retains all record identities already
    consumed by the current state, and consumes one bounded update slot.  It
    cannot make an old grounded record applicable a second time.
    """

    if not isinstance(state, AdaptiveState):
        raise AdaptiveSubstrateValidationError("rollback requires AdaptiveState")
    _identifier(checkpoint_id, "checkpoint_id")
    checkpoint = next(
        (item for item in state.checkpoints if item.checkpoint_id == checkpoint_id),
        None,
    )
    if checkpoint is None:
        raise AdaptiveSubstrateValidationError("rollback checkpoint is not retained")
    if state.updates_applied >= MAX_ADAPTIVE_UPDATES:
        raise AdaptiveSubstrateValidationError("rollback budget is exhausted")
    if state.generation >= MAX_ADAPTIVE_GENERATIONS:
        raise AdaptiveSubstrateValidationError("adaptive generation limit is exhausted")
    audit = AdaptiveAudit(
        f"{state.substrate_id}-rollback-{state.generation}-{state.updates_applied + 1}",
        f"rollback-{checkpoint_id}",
        "rollback",
        state.generation,
        state.generation + 1,
        "rollback",
        None,
        None,
        None,
        "restored immutable checkpoint without evicting applied identities",
        (),
        None,
    )
    topology = replace(
        checkpoint.route_topology,
        version=max(checkpoint.route_topology.version, state.route_topology.version) + 1,
        generation=state.route_topology.generation + 1,
    )
    next_state = AdaptiveState(
        state.substrate_id,
        state.generation + 1,
        0,
        topology,
        checkpoint.connections,
        state.tactics,
        checkpoint.active_tactic_id,
        checkpoint.tactic_scores,
        state.applied_record_ids,
        (state.audits + (audit,))[-MAX_ADAPTIVE_AUDIT:],
        state.checkpoints,
        state.failure_streak,
    )
    return next_state, audit


def invalidate_grounded_adaptation(
    state: AdaptiveState,
    target_record_id: str,
    invalidating_record: SettlementRouteRecord,
    *,
    reason: str,
) -> tuple[AdaptiveState, AdaptiveAudit]:
    """Reverse one retained adaptive update after a newer grounded failure.

    This reducer is deliberately conservative: it does not create replacement
    credit, and it only restores a retained pre-update checkpoint.  Both the
    original and invalidating identities remain consumed.
    """

    if not isinstance(state, AdaptiveState):
        raise AdaptiveSubstrateValidationError("invalidation requires AdaptiveState")
    _identifier(target_record_id, "target_record_id")
    if not isinstance(reason, str) or not reason.strip():
        raise AdaptiveSubstrateValidationError("invalidation reason is required")
    if target_record_id in state.invalidated_record_ids:
        raise AdaptiveSubstrateValidationError("adaptive record is already invalidated")
    target_audit = next(
        (item for item in state.audits if item.record_id == target_record_id),
        None,
    )
    if target_audit is None or target_audit.operation in {"rollback", "invalidate_evidence"}:
        raise AdaptiveSubstrateValidationError(
            "invalidation requires one retained ordinary adaptive audit"
        )
    _reject_duplicate_record(state, invalidating_record)
    _, invalidating_evidence, epistemic_class = _require_grounded_credit(
        state, invalidating_record
    )
    if (
        invalidating_record.settlement.observed_outcome != "failure"
        or epistemic_class is not EpistemicOutcomeClass.TASK_FAILURE
    ):
        raise AdaptiveSubstrateValidationError(
            "invalidation requires a later grounded task failure"
        )
    checkpoint_candidates = tuple(
        item
        for item in state.checkpoints
        if target_record_id not in item.applied_record_ids
    )
    if not checkpoint_candidates:
        raise AdaptiveSubstrateValidationError(
            "target pre-update checkpoint is no longer retained"
        )
    if state.generation >= MAX_ADAPTIVE_GENERATIONS:
        raise AdaptiveSubstrateValidationError("adaptive generation limit is exhausted")
    if state.updates_applied >= MAX_ADAPTIVE_UPDATES:
        raise AdaptiveSubstrateValidationError("invalidation budget is exhausted")
    checkpoint = max(
        checkpoint_candidates, key=lambda item: len(item.applied_record_ids)
    )
    restored_topology = RouteTopology(
        checkpoint.route_topology.topology_id,
        state.route_topology.version + 1,
        checkpoint.route_topology.routes,
        state.route_topology.generation + 1,
        state.route_topology.applied_settlement_ids,
    )
    audit = AdaptiveAudit(
        f"{state.substrate_id}-invalidate-{state.generation}-{state.updates_applied + 1}",
        invalidating_record.record_id,
        "invalidate_evidence",
        state.generation,
        state.generation + 1,
        target_audit.route_id,
        None,
        None,
        None,
        f"reversed {target_record_id} after later grounded failure: {reason.strip()}",
        tuple(dict.fromkeys(target_audit.evidence_ids + invalidating_evidence)),
        epistemic_class.value,
    )
    return AdaptiveState(
        state.substrate_id,
        state.generation + 1,
        0,
        restored_topology,
        checkpoint.connections,
        state.tactics,
        checkpoint.active_tactic_id,
        checkpoint.tactic_scores,
        state.applied_record_ids + (invalidating_record.record_id,),
        (state.audits + (audit,))[-MAX_ADAPTIVE_AUDIT:],
        state.checkpoints,
        0,
        0,
        state.invalidated_record_ids + (target_record_id,),
    ), audit


def replay_adaptive_updates(
    initial_state: AdaptiveState,
    history: tuple[tuple[str, SettlementRouteRecord, HomeostaticSnapshot | None], ...],
) -> AdaptiveState:
    """Replay a deterministic sequence of named grounded route reducers."""

    if not isinstance(initial_state, AdaptiveState):
        raise AdaptiveSubstrateValidationError("replay requires AdaptiveState")
    state = initial_state
    for operation, record, pressure in history:
        state, _ = apply_grounded_adaptation(
            state, record, operation=operation, pressure=pressure
        )
    return state


def replay_adaptive_invalidations(
    initial_state: AdaptiveState,
    history: tuple[tuple[str, SettlementRouteRecord, str], ...],
) -> AdaptiveState:
    """Replay bounded grounded invalidations without introducing live state."""

    if not isinstance(initial_state, AdaptiveState):
        raise AdaptiveSubstrateValidationError("replay requires AdaptiveState")
    state = initial_state
    for target_record_id, invalidating_record, reason in history:
        state, _ = invalidate_grounded_adaptation(
            state,
            target_record_id,
            invalidating_record,
            reason=reason,
        )
    return state


@dataclass(frozen=True)
class OrzhaalExperimentResult:
    """Disposable comparison output; never canonical learning authority."""

    experiment_id: str
    fork_generation: int
    operation: str
    canonical_state_digest: str
    result_digest: str
    changed: bool
    promotable: bool = False
    canonical_mutation: bool = False

    def __post_init__(self) -> None:
        for value, name in (
            (self.experiment_id, "experiment_id"),
            (self.canonical_state_digest, "canonical_state_digest"),
            (self.result_digest, "result_digest"),
        ):
            _identifier(value, name)
        _positive_integer(self.fork_generation, "fork_generation")
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise AdaptiveSubstrateValidationError("experiment operation is required")
        if self.promotable is not False or self.canonical_mutation is not False:
            raise AdaptiveSubstrateValidationError(
                "Orzhaal results cannot be promoted or mutate canonical state"
            )


def _state_digest(state: AdaptiveState) -> str:
    return hashlib.sha256(
        repr(state.to_dict()).encode("utf-8")
    ).hexdigest()


def run_orzhaal_experiment(
    canonical_state: AdaptiveState,
    *,
    experiment_id: str,
    operation: str = "observe_rule_variant",
    weight_delta: float = 0.02,
) -> OrzhaalExperimentResult:
    """Compare a disposable fork without exposing any canonical write method."""

    if not isinstance(canonical_state, AdaptiveState):
        raise AdaptiveSubstrateValidationError("Orzhaal requires AdaptiveState")
    _identifier(experiment_id, "experiment_id")
    if not isinstance(operation, str) or not operation.strip():
        raise AdaptiveSubstrateValidationError("experiment operation is required")
    if not math.isfinite(float(weight_delta)) or abs(float(weight_delta)) > 0.05:
        raise AdaptiveSubstrateValidationError("Orzhaal experiment delta is bounded")
    before = _state_digest(canonical_state)
    route = canonical_state.route_topology.routes[0]
    simulated_weight = min(MAX_WEIGHT, max(MIN_WEIGHT, route.weight + float(weight_delta)))
    result_digest = hashlib.sha256(
        f"{experiment_id}|{operation}|{canonical_state.generation}|{simulated_weight}".encode(
            "utf-8"
        )
    ).hexdigest()
    return OrzhaalExperimentResult(
        experiment_id,
        canonical_state.generation,
        operation,
        before,
        result_digest,
        simulated_weight != route.weight,
    )


__all__ = [
    "AdaptiveAudit",
    "AdaptiveCheckpoint",
    "AdaptiveState",
    "AdaptiveSubstrateValidationError",
    "BASELINE_CONNECTION_WEIGHT",
    "CandidateConnection",
    "CandidateTactic",
    "HomeostaticSnapshot",
    "MAX_ADAPTIVE_AUDIT",
    "MAX_ADAPTIVE_ROUTE_WEIGHT",
    "MAX_ADAPTIVE_GENERATIONS",
    "MAX_ADAPTIVE_UPDATES",
    "MAX_CONNECTION_DEGREE",
    "MAX_CONNECTIONS",
    "MAX_ROUTE_DECAY",
    "MAX_ROUTE_RECOVERY",
    "MAX_TACTIC_STREAK",
    "OrzhaalExperimentResult",
    "apply_grounded_adaptation",
    "form_grounded_connection",
    "invalidate_grounded_adaptation",
    "replay_adaptive_updates",
    "replay_adaptive_invalidations",
    "rollback_adaptive_state",
    "run_orzhaal_experiment",
    "switch_grounded_tactic",
    "weaken_grounded_connection",
]