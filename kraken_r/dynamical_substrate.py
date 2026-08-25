"""Bounded multi-tick dynamical substrate for Kraken-R Stage 10.8.

This is a pure reducer over caller-owned immutable records.  It provides a
small event fabric and three timescales, but it is not a scheduler, daemon,
store, executor, provider, or second runtime.  Every tick is explicit:
``state + events at t -> new_state at t+1``.

Fast state may influence the next candidate tick's activation, inhibition, and
surprise.  Medium state contains only the already-approved adaptive reducers.
Slow state records coherence observations and cannot create authority,
evidence, topology, tactics, or persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Iterable

from .adaptive_substrate import (
    AdaptiveAudit,
    AdaptiveState,
    HomeostaticSnapshot,
    apply_grounded_adaptation,
    form_grounded_connection,
    rollback_adaptive_state,
    switch_grounded_tactic,
    weaken_grounded_connection,
)
from .contracts import Signal
from .nervous_system import (
    PropagationTrace,
    SignalNetwork,
    SignalReplayRecord,
    replay_signal_propagation,
)
from .physiology import (
    OperatingRegime,
    PhysiologySnapshot,
    PhysiologyTrace,
    evaluate_physiology,
)
from .plastic_routing import SettlementRouteRecord, apply_settlement_learning


MAX_TICK_EVENTS = 16
MAX_EVENT_LOG = 64
MAX_SIGNAL_TOPICS = 16
MAX_ROUTE_HISTORY = 32
MAX_COHERENCE_OBSERVATIONS = 16
MAX_AUDIT_RECORDS = 32
MAX_RESOURCE_PRESSURE = 1.0
MAX_SURPRISE = 1.0
MAX_TICKS = 256


class DynamicalSubstrateValidationError(ValueError):
    """Raised when a Stage 10.8 boundary or replay input is malformed."""


class DynamicalEventKind(str, Enum):
    SIGNAL = "signal"
    OBSERVATION = "observation"
    SETTLEMENT = "settlement"
    RESOURCE = "resource"
    ROLLBACK = "rollback"


class MismatchKind(str, Enum):
    MATCH = "match"
    UNDERPREDICTION = "underprediction"
    OVERPREDICTION = "overprediction"
    SURPRISE = "surprise"


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(c.isspace() for c in value):
        raise DynamicalSubstrateValidationError(
            f"{name} must be a non-empty identifier without whitespace"
        )
    return value


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicalSubstrateValidationError(f"{name} must be non-empty text")
    return value


def _unit(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicalSubstrateValidationError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise DynamicalSubstrateValidationError(f"{name} must be between 0 and 1")
    return round(value, 6)


def _integer(value: int, name: str, *, maximum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DynamicalSubstrateValidationError(f"{name} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise DynamicalSubstrateValidationError(f"{name} exceeds its bounded limit")


@dataclass(frozen=True)
class PredictionMismatch:
    """Typed, bounded difference between a declared prediction and observation."""

    mismatch_id: str
    predicted: float
    observed: float
    magnitude: float
    kind: MismatchKind
    source_event_id: str

    def __post_init__(self) -> None:
        _identifier(self.mismatch_id, "mismatch_id")
        _unit(self.predicted, "predicted")
        _unit(self.observed, "observed")
        expected = round(abs(float(self.observed) - float(self.predicted)), 6)
        if _unit(self.magnitude, "magnitude") != expected:
            raise DynamicalSubstrateValidationError(
                "mismatch magnitude must equal the absolute prediction difference"
            )
        try:
            kind = MismatchKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise DynamicalSubstrateValidationError("mismatch kind is unsupported") from exc
        expected_kind = (
            MismatchKind.MATCH
            if expected == 0.0
            else MismatchKind.UNDERPREDICTION
            if self.observed > self.predicted
            else MismatchKind.OVERPREDICTION
        )
        if kind is not expected_kind:
            raise DynamicalSubstrateValidationError("mismatch kind disagrees with values")
        _identifier(self.source_event_id, "source_event_id")
        object.__setattr__(self, "kind", kind)

    @classmethod
    def from_values(
        cls, mismatch_id: str, predicted: float, observed: float, source_event_id: str
    ) -> "PredictionMismatch":
        magnitude = round(abs(float(observed) - float(predicted)), 6)
        kind = (
            MismatchKind.MATCH
            if magnitude == 0.0
            else MismatchKind.UNDERPREDICTION
            if observed > predicted
            else MismatchKind.OVERPREDICTION
        )
        return cls(mismatch_id, predicted, observed, magnitude, kind, source_event_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mismatch_id": self.mismatch_id,
            "predicted": self.predicted,
            "observed": self.observed,
            "magnitude": self.magnitude,
            "kind": self.kind.value,
            "source_event_id": self.source_event_id,
        }


@dataclass(frozen=True)
class DynamicalEvent:
    """One explicitly declared event; exactly one typed payload is permitted."""

    event_id: str
    kind: DynamicalEventKind
    signal: Signal | None = None
    mismatch: PredictionMismatch | None = None
    settlement: SettlementRouteRecord | None = None
    resource_pressure: float | None = None
    checkpoint_id: str | None = None
    operation: str = "credit"
    connection_id: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.event_id, "event_id")
        try:
            kind = DynamicalEventKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise DynamicalSubstrateValidationError("event kind is unsupported") from exc
        object.__setattr__(self, "kind", kind)
        payloads = (
            self.signal is not None,
            self.mismatch is not None,
            self.settlement is not None,
            self.resource_pressure is not None,
            self.checkpoint_id is not None,
        )
        if sum(payloads) != 1:
            raise DynamicalSubstrateValidationError(
                "event must contain exactly one typed payload"
            )
        expected = {
            DynamicalEventKind.SIGNAL: self.signal,
            DynamicalEventKind.OBSERVATION: self.mismatch,
            DynamicalEventKind.SETTLEMENT: self.settlement,
            DynamicalEventKind.RESOURCE: self.resource_pressure,
            DynamicalEventKind.ROLLBACK: self.checkpoint_id,
        }[kind]
        if expected is None:
            raise DynamicalSubstrateValidationError(
                f"{kind.value} event has the wrong payload"
            )
        if self.resource_pressure is not None:
            object.__setattr__(
                self, "resource_pressure", _unit(self.resource_pressure, "resource_pressure")
            )
        if self.operation not in {
            "credit",
            "strengthen",
            "weaken",
            "decay",
            "recover",
            "form_connection",
            "weaken_connection",
            "switch_tactic",
        }:
            raise DynamicalSubstrateValidationError("unsupported settlement operation")
        if self.connection_id is not None:
            _identifier(self.connection_id, "connection_id")
        if self.checkpoint_id is not None:
            _identifier(self.checkpoint_id, "checkpoint_id")
        if kind is not DynamicalEventKind.SETTLEMENT and (
            self.operation != "credit"
            or self.connection_id is not None
        ):
            raise DynamicalSubstrateValidationError(
                "operation and connection_id are settlement-only fields"
            )
        if kind is not DynamicalEventKind.ROLLBACK and self.checkpoint_id is not None:
            raise DynamicalSubstrateValidationError("checkpoint_id is rollback-only")

    @classmethod
    def signal_event(cls, event_id: str, signal: Signal) -> "DynamicalEvent":
        return cls(event_id, DynamicalEventKind.SIGNAL, signal=signal)

    @classmethod
    def observation_event(
        cls, event_id: str, predicted: float, observed: float
    ) -> "DynamicalEvent":
        mismatch = PredictionMismatch.from_values(
            f"{event_id}-mismatch", predicted, observed, event_id
        )
        return cls(event_id, DynamicalEventKind.OBSERVATION, mismatch=mismatch)

    @classmethod
    def settlement_event(
        cls,
        event_id: str,
        settlement: SettlementRouteRecord,
        *,
        operation: str = "credit",
        connection_id: str | None = None,
    ) -> "DynamicalEvent":
        return cls(
            event_id,
            DynamicalEventKind.SETTLEMENT,
            settlement=settlement,
            operation=operation,
            connection_id=connection_id,
        )

    @classmethod
    def resource_event(cls, event_id: str, resource_pressure: float) -> "DynamicalEvent":
        return cls(
            event_id,
            DynamicalEventKind.RESOURCE,
            resource_pressure=resource_pressure,
        )

    @classmethod
    def rollback_event(cls, event_id: str, checkpoint_id: str) -> "DynamicalEvent":
        return cls(
            event_id,
            DynamicalEventKind.ROLLBACK,
            checkpoint_id=checkpoint_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind.value,
            "signal": self.signal.to_dict() if self.signal is not None else None,
            "mismatch": self.mismatch.to_dict() if self.mismatch is not None else None,
            "settlement_record_id": (
                self.settlement.record_id if self.settlement is not None else None
            ),
            "resource_pressure": self.resource_pressure,
            "checkpoint_id": self.checkpoint_id,
            "operation": self.operation,
            "connection_id": self.connection_id,
        }


@dataclass(frozen=True)
class DynamicalTick:
    """A bounded, caller-owned batch of events for exactly one next tick."""

    tick: int
    events: tuple[DynamicalEvent, ...] = ()

    def __post_init__(self) -> None:
        _integer(self.tick, "tick", maximum=MAX_TICKS)
        object.__setattr__(self, "events", tuple(self.events))
        if len(self.events) > MAX_TICK_EVENTS:
            raise DynamicalSubstrateValidationError("tick event budget exceeded")
        if not all(isinstance(item, DynamicalEvent) for item in self.events):
            raise DynamicalSubstrateValidationError("tick events are invalid")
        ids = tuple(item.event_id for item in self.events)
        if len(set(ids)) != len(ids):
            raise DynamicalSubstrateValidationError("tick event ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {"tick": self.tick, "events": [item.to_dict() for item in self.events]}


@dataclass(frozen=True)
class FastState:
    """Short-lived activation, inhibition, surprise, and resource state."""

    activation: float = 0.0
    inhibition: float = 0.0
    surprise: float = 0.0
    resource_pressure: float = 0.0
    last_regime: OperatingRegime = OperatingRegime.PRODUCTIVE
    active_route_id: str | None = None
    signal_topics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("activation", "inhibition", "surprise", "resource_pressure"):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        try:
            object.__setattr__(self, "last_regime", OperatingRegime(self.last_regime))
        except (TypeError, ValueError) as exc:
            raise DynamicalSubstrateValidationError("last_regime is unsupported") from exc
        if self.active_route_id is not None:
            _identifier(self.active_route_id, "active_route_id")
        topics = tuple(self.signal_topics)
        if len(topics) > MAX_SIGNAL_TOPICS or not all(isinstance(item, str) for item in topics):
            raise DynamicalSubstrateValidationError("signal topic retention is bounded")
        object.__setattr__(self, "signal_topics", topics)


@dataclass(frozen=True)
class MediumState:
    """Approved route, connection, and tactic state with bounded turnover."""

    adaptive_state: AdaptiveState
    route_history: tuple[str, ...] = ()
    turnover: int = 0
    decay_events: int = 0
    plasticity_events: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.adaptive_state, AdaptiveState):
            raise DynamicalSubstrateValidationError("medium state requires AdaptiveState")
        for name in ("route_history",):
            values = tuple(getattr(self, name))
            if len(values) > MAX_ROUTE_HISTORY or not all(isinstance(v, str) for v in values):
                raise DynamicalSubstrateValidationError("route history is invalid or unbounded")
            object.__setattr__(self, name, values)
        for name in ("turnover", "decay_events", "plasticity_events"):
            _integer(getattr(self, name), name)


@dataclass(frozen=True)
class SlowState:
    """Minimal coherence observations; no slow field is an authority signal."""

    coherence_score: float = 0.0
    recurrence: int = 0
    coherence_observations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "coherence_score", _unit(self.coherence_score, "coherence_score"))
        _integer(self.recurrence, "recurrence")
        values = tuple(self.coherence_observations)
        if len(values) > MAX_COHERENCE_OBSERVATIONS or not all(
            isinstance(v, str) and v for v in values
        ):
            raise DynamicalSubstrateValidationError("coherence observations are invalid")
        object.__setattr__(self, "coherence_observations", values)


@dataclass(frozen=True)
class SubstrateInstrumentation:
    """Bounded observability only; instrumentation never feeds epistemic credit."""

    activity: float = 0.0
    diversity: float = 0.0
    entropy: float = 0.0
    dominant_route_share: float = 0.0
    turnover: int = 0
    surprise: float = 0.0
    decay_events: int = 0
    plasticity_events: int = 0
    rollback_events: int = 0
    inhibition_events: int = 0
    stagnation_ticks: int = 0
    resource_pressure: float = 0.0
    coherence_recurrence: int = 0

    def __post_init__(self) -> None:
        for name in (
            "activity",
            "diversity",
            "entropy",
            "dominant_route_share",
            "surprise",
            "resource_pressure",
        ):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        for name in (
            "turnover",
            "decay_events",
            "plasticity_events",
            "rollback_events",
            "inhibition_events",
            "stagnation_ticks",
            "coherence_recurrence",
        ):
            _integer(getattr(self, name), name)


@dataclass(frozen=True)
class WithheldEvent:
    """Bounded provenance retained when an event is correctly denied authority."""

    event_id: str
    kind: DynamicalEventKind
    reason: str
    record_id: str | None = None
    settlement_id: str | None = None
    transaction_id: str | None = None
    objective_id: str | None = None
    task_state_id: str | None = None
    task_state_version: int | None = None
    route_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    epistemic_class: str | None = None
    operation: str | None = None
    checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.event_id, "event_id")
        try:
            object.__setattr__(self, "kind", DynamicalEventKind(self.kind))
        except (TypeError, ValueError) as exc:
            raise DynamicalSubstrateValidationError("withheld event kind is unsupported") from exc
        _text(self.reason, "reason")
        for name in (
            "record_id",
            "settlement_id",
            "transaction_id",
            "objective_id",
            "task_state_id",
            "route_id",
            "operation",
            "checkpoint_id",
        ):
            value = getattr(self, name)
            if value is not None:
                _identifier(value, name)
        if self.task_state_version is not None:
            _integer(self.task_state_version, "task_state_version")
            if self.task_state_version < 1:
                raise DynamicalSubstrateValidationError(
                    "withheld task_state_version must be positive"
                )
        values = tuple(self.evidence_ids)
        if not all(isinstance(value, str) and value for value in values):
            raise DynamicalSubstrateValidationError("withheld evidence ids are invalid")
        object.__setattr__(self, "evidence_ids", values)
        if self.epistemic_class is not None:
            _identifier(self.epistemic_class, "epistemic_class")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind.value,
            "reason": self.reason,
            "record_id": self.record_id,
            "settlement_id": self.settlement_id,
            "transaction_id": self.transaction_id,
            "objective_id": self.objective_id,
            "task_state_id": self.task_state_id,
            "task_state_version": self.task_state_version,
            "route_id": self.route_id,
            "evidence_ids": list(self.evidence_ids),
            "epistemic_class": self.epistemic_class,
            "operation": self.operation,
            "checkpoint_id": self.checkpoint_id,
        }


@dataclass(frozen=True)
class DynamicalState:
    """Complete immutable state at one explicit tick."""

    substrate_id: str
    transaction_id: str
    objective_id: str
    task_state_id: str
    task_state_version: int
    tick: int
    fast: FastState
    medium: MediumState
    slow: SlowState
    instrumentation: SubstrateInstrumentation = SubstrateInstrumentation()
    consumed_event_ids: tuple[str, ...] = ()
    event_log: tuple[str, ...] = ()
    withheld_events: tuple[WithheldEvent, ...] = ()

    def __post_init__(self) -> None:
        for value, name in (
            (self.substrate_id, "substrate_id"),
            (self.transaction_id, "transaction_id"),
            (self.objective_id, "objective_id"),
            (self.task_state_id, "task_state_id"),
        ):
            _identifier(value, name)
        _integer(self.task_state_version, "task_state_version")
        if self.task_state_version < 1:
            raise DynamicalSubstrateValidationError("task_state_version must be positive")
        _integer(self.tick, "tick", maximum=MAX_TICKS)
        for name, expected in (
            ("fast", FastState),
            ("medium", MediumState),
            ("slow", SlowState),
            ("instrumentation", SubstrateInstrumentation),
        ):
            if not isinstance(getattr(self, name), expected):
                raise DynamicalSubstrateValidationError(f"{name} is invalid")
        for name in ("consumed_event_ids", "event_log"):
            values = tuple(getattr(self, name))
            if len(values) > MAX_EVENT_LOG or not all(
                isinstance(v, str) and v for v in values
            ):
                raise DynamicalSubstrateValidationError(f"{name} is invalid or unbounded")
            object.__setattr__(self, name, values)
        withheld = tuple(self.withheld_events)
        if len(withheld) > MAX_AUDIT_RECORDS or not all(
            isinstance(item, WithheldEvent) for item in withheld
        ):
            raise DynamicalSubstrateValidationError("withheld event audit is invalid")
        object.__setattr__(self, "withheld_events", withheld)

    @classmethod
    def fixture(
        cls,
        substrate_id: str = "candidate-dynamical-substrate",
        *,
        transaction_id: str = "dynamical-transaction",
        objective_id: str = "dynamical-objective",
        task_state_id: str = "dynamical-objective-state-5",
        task_state_version: int = 5,
    ) -> "DynamicalState":
        return cls(
            substrate_id,
            transaction_id,
            objective_id,
            task_state_id,
            task_state_version,
            0,
            FastState(),
            MediumState(AdaptiveState.fixture(f"{substrate_id}-adaptive")),
            SlowState(),
        )

    def to_dict(self) -> dict[str, Any]:
        adaptive = self.medium.adaptive_state.to_dict()
        return {
            "substrate_id": self.substrate_id,
            "transaction_id": self.transaction_id,
            "objective_id": self.objective_id,
            "task_state_id": self.task_state_id,
            "task_state_version": self.task_state_version,
            "tick": self.tick,
            "fast": {
                "activation": self.fast.activation,
                "inhibition": self.fast.inhibition,
                "surprise": self.fast.surprise,
                "resource_pressure": self.fast.resource_pressure,
                "last_regime": self.fast.last_regime.value,
                "active_route_id": self.fast.active_route_id,
                "signal_topics": list(self.fast.signal_topics),
            },
            "medium": {
                "adaptive_state": adaptive,
                "route_history": list(self.medium.route_history),
                "turnover": self.medium.turnover,
                "decay_events": self.medium.decay_events,
                "plasticity_events": self.medium.plasticity_events,
            },
            "slow": {
                "coherence_score": self.slow.coherence_score,
                "recurrence": self.slow.recurrence,
                "coherence_observations": list(self.slow.coherence_observations),
            },
            "instrumentation": self.instrumentation.__dict__,
            "consumed_event_ids": list(self.consumed_event_ids),
            "event_log": list(self.event_log),
            "withheld_events": [item.to_dict() for item in self.withheld_events],
            "authority": "kraken_candidate",
        }


@dataclass(frozen=True)
class DynamicalTickTrace:
    """Auditable output of one pure tick reduction."""

    tick: int
    event_ids: tuple[str, ...]
    delivered_signal_ids: tuple[str, ...]
    mismatch_ids: tuple[str, ...]
    noncreditable_settlement_ids: tuple[str, ...]
    withheld_events: tuple[WithheldEvent, ...]
    adaptive_audits: tuple[AdaptiveAudit, ...]
    physiology: PhysiologyTrace
    signal_trace: PropagationTrace | None
    activity: float
    inhibited: bool
    reason: str

    def __post_init__(self) -> None:
        _integer(self.tick, "tick")
        for name in (
            "event_ids",
            "delivered_signal_ids",
            "mismatch_ids",
            "noncreditable_settlement_ids",
        ):
            values = tuple(getattr(self, name))
            if not all(isinstance(v, str) and v for v in values):
                raise DynamicalSubstrateValidationError(f"{name} is invalid")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "adaptive_audits", tuple(self.adaptive_audits))
        object.__setattr__(self, "withheld_events", tuple(self.withheld_events))
        if not all(isinstance(item, WithheldEvent) for item in self.withheld_events):
            raise DynamicalSubstrateValidationError("withheld trace events are invalid")
        if not all(isinstance(item, AdaptiveAudit) for item in self.adaptive_audits):
            raise DynamicalSubstrateValidationError("adaptive audits are invalid")
        if not isinstance(self.physiology, PhysiologyTrace):
            raise DynamicalSubstrateValidationError("physiology trace is invalid")
        if self.signal_trace is not None and not isinstance(self.signal_trace, PropagationTrace):
            raise DynamicalSubstrateValidationError("signal trace is invalid")
        object.__setattr__(self, "activity", _unit(self.activity, "activity"))
        if not isinstance(self.inhibited, bool):
            raise DynamicalSubstrateValidationError("inhibited must be boolean")
        _text(self.reason, "reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "event_ids": list(self.event_ids),
            "delivered_signal_ids": list(self.delivered_signal_ids),
            "mismatch_ids": list(self.mismatch_ids),
            "noncreditable_settlement_ids": list(self.noncreditable_settlement_ids),
            "withheld_events": [item.to_dict() for item in self.withheld_events],
            "adaptive_audits": [item.to_dict() for item in self.adaptive_audits],
            "physiology": self.physiology.to_dict(),
            "signal_trace": (
                self.signal_trace.to_dict() if self.signal_trace is not None else None
            ),
            "activity": self.activity,
            "inhibited": self.inhibited,
            "reason": self.reason,
        }


class DynamicalEventFabric:
    """Deterministic event ordering, not a live bus or subscription system."""

    _ORDER = {
        DynamicalEventKind.SIGNAL: 0,
        DynamicalEventKind.OBSERVATION: 1,
        DynamicalEventKind.RESOURCE: 2,
        DynamicalEventKind.ROLLBACK: 3,
        DynamicalEventKind.SETTLEMENT: 4,
    }

    @classmethod
    def order(cls, events: Iterable[DynamicalEvent]) -> tuple[DynamicalEvent, ...]:
        ordered = tuple(events)
        if len(ordered) > MAX_TICK_EVENTS:
            raise DynamicalSubstrateValidationError("event fabric budget exceeded")
        if len({event.event_id for event in ordered}) != len(ordered):
            raise DynamicalSubstrateValidationError("event fabric received duplicate ids")
        return tuple(sorted(ordered, key=lambda item: (cls._ORDER[item.kind], item.event_id)))


def _context_match(state: DynamicalState, record: SettlementRouteRecord) -> None:
    trace = record.constitutional_trace
    if (
        trace.transaction_id != state.transaction_id
        or trace.objective.objective_id != state.objective_id
        or trace.states[4].state_id != state.task_state_id
        or trace.states[4].version != state.task_state_version
    ):
        raise DynamicalSubstrateValidationError(
            "settlement is not bound to the substrate task state"
        )


def _signal_trace(state: DynamicalState, signals: tuple[Signal, ...]) -> PropagationTrace | None:
    if not signals:
        return None
    for signal in signals:
        if (
            signal.correlation_id != state.transaction_id
            or signal.provenance.get("objective_id") != state.objective_id
            or signal.provenance.get("transaction_id") != state.transaction_id
            or signal.task_state_id != state.task_state_id
            or signal.task_state_version != state.task_state_version
        ):
            raise DynamicalSubstrateValidationError("signal is not bound to the substrate task state")
    return replay_signal_propagation(
        SignalReplayRecord(
            state.transaction_id,
            state.objective_id,
            state.task_state_id,
            state.task_state_version,
            signals,
        )
    )


def _entropy(values: tuple[str, ...]) -> float:
    if not values:
        return 0.0
    counts = {value: values.count(value) for value in set(values)}
    if len(counts) <= 1:
        return 0.0
    total = float(len(values))
    raw = -sum((count / total) * math.log(count / total) for count in counts.values())
    return round(raw / math.log(len(counts)), 6)


def _withheld_settlement(
    event: DynamicalEvent, record: SettlementRouteRecord, reason: str
) -> WithheldEvent:
    verified = record.grounded_execution
    return WithheldEvent(
        event.event_id,
        event.kind,
        reason,
        record_id=record.record_id,
        settlement_id=record.settlement.settlement_id,
        transaction_id=record.provenance.get("transaction_id"),
        objective_id=record.provenance.get("objective_id"),
        task_state_id=record.provenance.get("task_state_id"),
        task_state_version=record.provenance.get("task_state_version"),
        route_id=record.provenance.get("route_id"),
        evidence_ids=tuple(record.provenance.get("evidence_ids", ())),
        epistemic_class=(
            verified.epistemic_class.value if verified is not None else None
        ),
        operation=event.operation,
    )


def reduce_dynamical_tick(
    state: DynamicalState,
    tick: DynamicalTick,
    *,
    signal_network: SignalNetwork | None = None,
) -> tuple[DynamicalState, DynamicalTickTrace]:
    """Reduce one explicit tick without clocks, side effects, or persistence."""

    if not isinstance(state, DynamicalState):
        raise DynamicalSubstrateValidationError("reduction requires DynamicalState")
    if not isinstance(tick, DynamicalTick):
        raise DynamicalSubstrateValidationError("reduction requires DynamicalTick")
    if tick.tick != state.tick + 1:
        raise DynamicalSubstrateValidationError("tick must advance by exactly one")
    if signal_network is not None and not isinstance(signal_network, SignalNetwork):
        raise DynamicalSubstrateValidationError("signal_network is invalid")
    ordered = DynamicalEventFabric.order(tick.events)
    if any(event.event_id in state.consumed_event_ids for event in ordered):
        raise DynamicalSubstrateValidationError("event was already consumed")

    signals = tuple(event.signal for event in ordered if event.signal is not None)
    signal_trace = _signal_trace(state, signals)
    if signal_network is not None:
        signal_trace = signal_network.propagate(
            signals,
            transaction_id=state.transaction_id,
            objective_id=state.objective_id,
            task_state_id=state.task_state_id,
            task_state_version=state.task_state_version,
        )
    observations = tuple(event.mismatch for event in ordered if event.mismatch is not None)
    resource_events = tuple(
        event.resource_pressure for event in ordered if event.resource_pressure is not None
    )
    resource = resource_events[-1] if resource_events else state.fast.resource_pressure
    surprise = max(
        (item.magnitude for item in observations),
        default=state.fast.surprise * 0.80,
    )
    topics = tuple(
        dict.fromkeys(
            state.fast.signal_topics
            + tuple(item.signal.topic for item in ordered if item.signal is not None)
        )
    )[-MAX_SIGNAL_TOPICS:]
    activation = min(
        1.0,
        round(
            state.fast.activation * 0.70
            + min(0.30, len(signal_trace.delivered_signal_ids) * 0.05 if signal_trace else 0.0),
            6,
        ),
    )
    provisional_inhibition = state.fast.inhibition * 0.65
    if signal_trace is not None and signal_trace.inhibits("candidate.action.authorize"):
        provisional_inhibition = max(provisional_inhibition, 1.0)
    snapshot = PhysiologySnapshot(
        f"{state.substrate_id}-physiology-{tick.tick}",
        state.transaction_id,
        state.objective_id,
        state.task_state_id,
        state.task_state_version,
        memory_pressure=provisional_inhibition,
        routing_pressure=state.medium.adaptive_state.failure_streak / 3.0,
        backlog_pressure=min(1.0, len(ordered) / MAX_TICK_EVENTS),
        contradiction_density=surprise,
        resource_pressure=resource,
        protected_reserve=1.0 - resource,
        prior_regime=state.fast.last_regime,
    )
    physiology = evaluate_physiology(
        snapshot,
        transaction_id=state.transaction_id,
        objective_id=state.objective_id,
        task_state_id=state.task_state_id,
        task_state_version=state.task_state_version,
    )
    inhibited = physiology.decision.action_inhibited
    adaptive = state.medium.adaptive_state
    audits: list[AdaptiveAudit] = []
    noncreditable: list[str] = []
    withheld: list[WithheldEvent] = []
    route_id = state.fast.active_route_id
    turnover = state.medium.turnover
    decay_events = state.medium.decay_events
    plasticity_events = state.medium.plasticity_events
    rollback_events = state.instrumentation.rollback_events
    for event in ordered:
        if event.checkpoint_id is not None:
            if inhibited:
                withheld.append(
                    WithheldEvent(
                        event.event_id,
                        event.kind,
                        "advisory physiology inhibited candidate rollback",
                        operation="rollback",
                        checkpoint_id=event.checkpoint_id,
                    )
                )
                continue
            try:
                adaptive, audit = rollback_adaptive_state(adaptive, event.checkpoint_id)
            except ValueError as exc:
                raise DynamicalSubstrateValidationError(
                    f"rollback reducer rejected {event.event_id}: {exc}"
                ) from exc
            audits.append(audit)
            rollback_events += 1
            turnover += 1
            continue
        if event.settlement is None:
            continue
        record = event.settlement
        _context_match(state, record)
        learning = apply_settlement_learning(adaptive.route_topology, record)[1]
        if learning.disposition != "accepted":
            noncreditable.append(record.record_id)
            withheld.append(_withheld_settlement(event, record, learning.reason))
            continue
        if inhibited:
            noncreditable.append(record.record_id)
            withheld.append(
                _withheld_settlement(
                    event, record, "advisory physiology inhibited candidate adaptation"
                )
            )
            continue
        try:
            if event.operation == "form_connection":
                adaptive, audit = form_grounded_connection(adaptive, record)
            elif event.operation == "weaken_connection":
                if event.connection_id is None:
                    raise DynamicalSubstrateValidationError(
                        "weaken_connection requires connection_id"
                    )
                adaptive, audit = weaken_grounded_connection(
                    adaptive, record, event.connection_id
                )
            elif event.operation == "switch_tactic":
                pressure = HomeostaticSnapshot(
                    f"{state.substrate_id}-pressure-{tick.tick}",
                    state.transaction_id,
                    state.objective_id,
                    state.task_state_id,
                    state.task_state_version,
                    contradiction=surprise,
                    uncertainty=surprise,
                    repeated_failure=min(1.0, adaptive.failure_streak / 3.0),
                    novelty=min(1.0, len(topics) / MAX_SIGNAL_TOPICS),
                    resource_expenditure=resource,
                )
                adaptive, audit = switch_grounded_tactic(adaptive, record, pressure)
            else:
                adaptive, audit = apply_grounded_adaptation(
                    adaptive, record, operation=event.operation
                )
        except ValueError as exc:
            raise DynamicalSubstrateValidationError(
                f"grounded settlement reducer rejected {record.record_id}: {exc}"
            ) from exc
        audits.append(audit)
        route_id = audit.route_id
        turnover += 1
        plasticity_events += 1
        if audit.operation == "decay":
            decay_events += 1

    route_history = state.medium.route_history
    if route_id is not None:
        route_history = (route_history + (route_id,))[-MAX_ROUTE_HISTORY:]
    active_count = len(route_history)
    dominant_share = (
        max(route_history.count(item) for item in set(route_history)) / active_count
        if active_count
        else 0.0
    )
    diversity = round(len(set(topics)) / MAX_SIGNAL_TOPICS, 6)
    entropy = _entropy(route_history)
    recurrence = (
        state.slow.recurrence + 1
        if route_id is not None and route_history.count(route_id) >= 2
        else 0
    )
    coherence = round(
        min(1.0, 0.5 * dominant_share + 0.3 * (1.0 - surprise) + 0.2 * diversity), 6
    )
    slow = SlowState(
        coherence,
        recurrence,
        (state.slow.coherence_observations + (f"tick-{tick.tick}",))[
            -MAX_COHERENCE_OBSERVATIONS:
        ],
    )
    activity = min(1.0, round((len(ordered) + len(audits)) / MAX_TICK_EVENTS, 6))
    stagnation = state.instrumentation.stagnation_ticks + 1 if not ordered else 0
    instrumentation = SubstrateInstrumentation(
        activity=activity,
        diversity=diversity,
        entropy=entropy,
        dominant_route_share=round(dominant_share, 6),
        turnover=turnover,
        surprise=surprise,
        decay_events=decay_events,
        plasticity_events=plasticity_events,
        rollback_events=rollback_events,
        inhibition_events=state.instrumentation.inhibition_events + int(inhibited),
        stagnation_ticks=stagnation,
        resource_pressure=resource,
        coherence_recurrence=recurrence,
    )
    fast = FastState(
        activation=activation * (0.25 if inhibited else 1.0),
        inhibition=1.0 if inhibited else provisional_inhibition * 0.5,
        surprise=surprise,
        resource_pressure=resource,
        last_regime=physiology.decision.regime,
        active_route_id=route_id,
        signal_topics=topics,
    )
    next_state = DynamicalState(
        state.substrate_id,
        state.transaction_id,
        state.objective_id,
        state.task_state_id,
        state.task_state_version,
        tick.tick,
        fast,
        MediumState(adaptive, route_history, turnover, decay_events, plasticity_events),
        slow,
        instrumentation,
        (state.consumed_event_ids + tuple(event.event_id for event in ordered))[-MAX_EVENT_LOG:],
        (state.event_log + tuple(event.event_id for event in ordered))[-MAX_EVENT_LOG:],
        (state.withheld_events + tuple(withheld))[-MAX_AUDIT_RECORDS:],
    )
    trace = DynamicalTickTrace(
        tick.tick,
        tuple(event.event_id for event in ordered),
        signal_trace.delivered_signal_ids if signal_trace is not None else (),
        tuple(item.mismatch_id for item in observations),
        tuple(noncreditable),
        tuple(withheld),
        tuple(audits),
        physiology,
        signal_trace,
        activity,
        inhibited,
        (
            "advisory physiology inhibited candidate adaptation"
            if inhibited
            else "bounded tick reduced signals, observations, physiology, and candidate state"
        ),
    )
    return next_state, trace


def replay_dynamical_ticks(
    initial: DynamicalState,
    ticks: Iterable[DynamicalTick],
    *,
    signal_network: SignalNetwork | None = None,
) -> tuple[DynamicalState, tuple[DynamicalTickTrace, ...]]:
    """Replay the same explicit tick stream without clocks or hidden state."""

    state = initial
    traces: list[DynamicalTickTrace] = []
    for tick in tuple(ticks):
        state, trace = reduce_dynamical_tick(
            state, tick, signal_network=signal_network
        )
        traces.append(trace)
    return state, tuple(traces)


__all__ = [
    "DynamicalEvent",
    "DynamicalEventFabric",
    "DynamicalEventKind",
    "DynamicalState",
    "DynamicalSubstrateValidationError",
    "DynamicalTick",
    "DynamicalTickTrace",
    "FastState",
    "MAX_EVENT_LOG",
    "MAX_TICK_EVENTS",
    "MediumState",
    "MismatchKind",
    "PredictionMismatch",
    "SlowState",
    "SubstrateInstrumentation",
    "WithheldEvent",
    "reduce_dynamical_tick",
    "replay_dynamical_ticks",
]