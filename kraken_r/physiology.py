"""Bounded, candidate-only physiology and regulation for Kraken-R Round 5.

This module adapts safe concepts from the preserved legacy mechanisms:
normalized multidimensional pressure, setpoint-style thresholds, hysteresis,
protected reserves, and explicit advisory regulation.  It does not import or
call HOP, homeostasis, pressure fields, queues, resource pools, event dispatchers,
routers, persistence, or runtime lifecycle code.

Physiology is an internal condition evaluator, not a source of evidence or a
second controller.  Its only permitted cycle effect is conservative inhibition
of a candidate action before execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any


class PhysiologyValidationError(ValueError):
    """Raised when a candidate physiology input or replay is malformed."""


class RegulationEffect(str, Enum):
    """The only candidate-cycle effect physiology may produce."""

    ALLOW = "allow"
    INHIBIT_ACTION = "inhibit_action"


class OperatingRegime(str, Enum):
    """Bounded internal operating regimes, ordered from calm to stressed."""

    PRODUCTIVE = "productive"
    CAUTIOUS = "cautious"
    RECOVERY = "recovery"
    CRITICAL = "critical"


SALVAGED_CONCEPTS = (
    "pressure_field.normalized_dimensions",
    "homeostasis.thresholds_and_hysteresis",
    "contradiction.backlog_and_resource_regulation",
    "resource_regulation.protected_reserves",
)

_REGIME_LEVEL = {
    OperatingRegime.PRODUCTIVE: 0,
    OperatingRegime.CAUTIOUS: 1,
    OperatingRegime.RECOVERY: 2,
    OperatingRegime.CRITICAL: 3,
}


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(
        char.isspace() for char in value
    ):
        raise PhysiologyValidationError(
            f"{field_name} must be a non-empty identifier"
        )
    return value


def _nonempty_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhysiologyValidationError(f"{field_name} must be non-empty text")
    return value


def _bounded(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhysiologyValidationError(f"{field_name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise PhysiologyValidationError(f"{field_name} must be between 0 and 1")
    return numeric


@dataclass(frozen=True)
class PhysiologySnapshot:
    """Immutable internal conditions for one candidate transaction."""

    snapshot_id: str
    transaction_id: str
    objective_id: str
    task_state_id: str
    task_state_version: int
    memory_pressure: float = 0.0
    routing_pressure: float = 0.0
    backlog_pressure: float = 0.0
    contradiction_density: float = 0.0
    resource_pressure: float = 0.0
    protected_reserve: float = 1.0
    prior_regime: OperatingRegime = OperatingRegime.PRODUCTIVE
    cooldown_remaining: int = 0

    def __post_init__(self) -> None:
        for value, name in (
            (self.snapshot_id, "snapshot_id"),
            (self.transaction_id, "transaction_id"),
            (self.objective_id, "objective_id"),
            (self.task_state_id, "task_state_id"),
        ):
            _identifier(value, name)
        if (
            isinstance(self.task_state_version, bool)
            or not isinstance(self.task_state_version, int)
            or self.task_state_version < 1
        ):
            raise PhysiologyValidationError(
                "task_state_version must be a positive integer"
            )
        for name in (
            "memory_pressure",
            "routing_pressure",
            "backlog_pressure",
            "contradiction_density",
            "resource_pressure",
            "protected_reserve",
        ):
            object.__setattr__(self, name, _bounded(getattr(self, name), name))
        try:
            prior = OperatingRegime(self.prior_regime)
        except (TypeError, ValueError) as exc:
            raise PhysiologyValidationError("prior_regime is unsupported") from exc
        object.__setattr__(self, "prior_regime", prior)
        if (
            isinstance(self.cooldown_remaining, bool)
            or not isinstance(self.cooldown_remaining, int)
            or not 0 <= self.cooldown_remaining <= 8
        ):
            raise PhysiologyValidationError(
                "cooldown_remaining must be an integer from 0 through 8"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "transaction_id": self.transaction_id,
            "objective_id": self.objective_id,
            "task_state_id": self.task_state_id,
            "task_state_version": self.task_state_version,
            "memory_pressure": self.memory_pressure,
            "routing_pressure": self.routing_pressure,
            "backlog_pressure": self.backlog_pressure,
            "contradiction_density": self.contradiction_density,
            "resource_pressure": self.resource_pressure,
            "protected_reserve": self.protected_reserve,
            "prior_regime": self.prior_regime.value,
            "cooldown_remaining": self.cooldown_remaining,
        }


@dataclass(frozen=True)
class RegulationDecision:
    """Auditable advisory result with no evidence or settlement meaning."""

    regime: OperatingRegime
    composite_pressure: float
    effect: RegulationEffect
    action_inhibited: bool
    reason: str
    cooldown_applied: bool = False
    advisory_only: bool = True

    def __post_init__(self) -> None:
        try:
            regime = OperatingRegime(self.regime)
            effect = RegulationEffect(self.effect)
        except (TypeError, ValueError) as exc:
            raise PhysiologyValidationError(
                "regulation decision contains an unsupported enum"
            ) from exc
        object.__setattr__(self, "regime", regime)
        object.__setattr__(self, "effect", effect)
        if not 0.0 <= self.composite_pressure <= 1.0:
            raise PhysiologyValidationError(
                "composite_pressure must be between 0 and 1"
            )
        if not isinstance(self.action_inhibited, bool):
            raise PhysiologyValidationError("action_inhibited must be boolean")
        if not isinstance(self.cooldown_applied, bool):
            raise PhysiologyValidationError("cooldown_applied must be boolean")
        if self.advisory_only is not True:
            raise PhysiologyValidationError(
                "physiology decisions must remain advisory-only"
            )
        _nonempty_text(self.reason, "reason")
        if self.action_inhibited != (self.effect is RegulationEffect.INHIBIT_ACTION):
            raise PhysiologyValidationError(
                "regulation effect and action_inhibited disagree"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime.value,
            "composite_pressure": self.composite_pressure,
            "effect": self.effect.value,
            "action_inhibited": self.action_inhibited,
            "reason": self.reason,
            "cooldown_applied": self.cooldown_applied,
            "advisory_only": self.advisory_only,
        }


@dataclass(frozen=True)
class PhysiologyTrace:
    """Immutable physiology evaluation bound to a candidate task state."""

    transaction_id: str
    objective_id: str
    task_state_id: str
    task_state_version: int
    snapshot: PhysiologySnapshot
    decision: RegulationDecision
    salvaged_concepts: tuple[str, ...] = SALVAGED_CONCEPTS

    def __post_init__(self) -> None:
        for value, name in (
            (self.transaction_id, "transaction_id"),
            (self.objective_id, "objective_id"),
            (self.task_state_id, "task_state_id"),
        ):
            _identifier(value, name)
        if (
            isinstance(self.task_state_version, bool)
            or not isinstance(self.task_state_version, int)
            or self.task_state_version < 1
        ):
            raise PhysiologyValidationError(
                "trace task_state_version must be a positive integer"
            )
        if not isinstance(self.snapshot, PhysiologySnapshot):
            raise PhysiologyValidationError("trace snapshot is invalid")
        if not isinstance(self.decision, RegulationDecision):
            raise PhysiologyValidationError("trace decision is invalid")
        object.__setattr__(self, "salvaged_concepts", tuple(self.salvaged_concepts))
        if not all(isinstance(item, str) and item for item in self.salvaged_concepts):
            raise PhysiologyValidationError("salvaged_concepts must be strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "objective_id": self.objective_id,
            "task_state_id": self.task_state_id,
            "task_state_version": self.task_state_version,
            "snapshot": self.snapshot.to_dict(),
            "decision": self.decision.to_dict(),
            "salvaged_concepts": list(self.salvaged_concepts),
        }


@dataclass(frozen=True)
class PhysiologyReplayRecord:
    """Immutable input envelope for deterministic physiology replay."""

    snapshot: PhysiologySnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, PhysiologySnapshot):
            raise PhysiologyValidationError("replay requires a PhysiologySnapshot")

    def to_dict(self) -> dict[str, Any]:
        return {"snapshot": self.snapshot.to_dict()}


def _composite_pressure(snapshot: PhysiologySnapshot) -> float:
    weighted = (
        snapshot.memory_pressure * 0.15
        + snapshot.routing_pressure * 0.15
        + snapshot.backlog_pressure * 0.20
        + snapshot.contradiction_density * 0.30
        + snapshot.resource_pressure * 0.20
    )
    return round(min(1.0, weighted), 6)


def _raw_regime(
    snapshot: PhysiologySnapshot, composite_pressure: float
) -> OperatingRegime:
    if (
        composite_pressure >= 0.85
        or snapshot.contradiction_density >= 0.75
        or snapshot.protected_reserve <= 0.10
    ):
        return OperatingRegime.CRITICAL
    if (
        composite_pressure >= 0.60
        or snapshot.contradiction_density >= 0.50
        or snapshot.backlog_pressure >= 0.70
        or snapshot.resource_pressure >= 0.80
    ):
        return OperatingRegime.RECOVERY
    if (
        composite_pressure >= 0.30
        or snapshot.backlog_pressure >= 0.45
        or snapshot.resource_pressure >= 0.45
        or snapshot.memory_pressure >= 0.45
        or snapshot.routing_pressure >= 0.45
    ):
        return OperatingRegime.CAUTIOUS
    return OperatingRegime.PRODUCTIVE


def evaluate_physiology(
    snapshot: PhysiologySnapshot,
    *,
    transaction_id: str,
    objective_id: str,
    task_state_id: str,
    task_state_version: int,
) -> PhysiologyTrace:
    """Evaluate one bounded snapshot without mutating or consulting live state."""

    if not isinstance(snapshot, PhysiologySnapshot):
        raise PhysiologyValidationError("physiology requires a snapshot")
    if (
        snapshot.transaction_id != transaction_id
        or snapshot.objective_id != objective_id
        or snapshot.task_state_id != task_state_id
        or snapshot.task_state_version != task_state_version
    ):
        raise PhysiologyValidationError(
            "physiology snapshot is not bound to the active task state"
        )
    composite = _composite_pressure(snapshot)
    raw = _raw_regime(snapshot, composite)
    cooldown_applied = (
        snapshot.cooldown_remaining > 0
        and _REGIME_LEVEL[snapshot.prior_regime] > _REGIME_LEVEL[raw]
    )
    regime = snapshot.prior_regime if cooldown_applied else raw
    if regime is OperatingRegime.CRITICAL:
        reason = "critical pressure or protected reserve requires a safe stop"
        inhibit = True
    elif regime is OperatingRegime.RECOVERY and (
        snapshot.contradiction_density >= 0.60
        or snapshot.backlog_pressure >= 0.80
        or snapshot.resource_pressure >= 0.90
        or snapshot.protected_reserve <= 0.20
    ):
        reason = "recovery conditions require conservative action inhibition"
        inhibit = True
    elif cooldown_applied:
        reason = "hysteresis retains the prior conservative regime"
        inhibit = regime in {OperatingRegime.RECOVERY, OperatingRegime.CRITICAL}
    else:
        reason = f"{regime.value} conditions permit the candidate action"
        inhibit = False
    decision = RegulationDecision(
        regime=regime,
        composite_pressure=composite,
        effect=(
            RegulationEffect.INHIBIT_ACTION
            if inhibit
            else RegulationEffect.ALLOW
        ),
        action_inhibited=inhibit,
        reason=reason,
        cooldown_applied=cooldown_applied,
    )
    return PhysiologyTrace(
        transaction_id=transaction_id,
        objective_id=objective_id,
        task_state_id=task_state_id,
        task_state_version=task_state_version,
        snapshot=snapshot,
        decision=decision,
    )


def replay_physiology(record: PhysiologyReplayRecord) -> PhysiologyTrace:
    """Replay an immutable physiology record through the same evaluator."""

    if not isinstance(record, PhysiologyReplayRecord):
        raise PhysiologyValidationError("replay requires a PhysiologyReplayRecord")
    snapshot = record.snapshot
    return evaluate_physiology(
        snapshot,
        transaction_id=snapshot.transaction_id,
        objective_id=snapshot.objective_id,
        task_state_id=snapshot.task_state_id,
        task_state_version=snapshot.task_state_version,
    )


__all__ = [
    "OperatingRegime",
    "PhysiologyReplayRecord",
    "PhysiologySnapshot",
    "PhysiologyTrace",
    "PhysiologyValidationError",
    "RegulationDecision",
    "RegulationEffect",
    "SALVAGED_CONCEPTS",
    "evaluate_physiology",
    "replay_physiology",
]