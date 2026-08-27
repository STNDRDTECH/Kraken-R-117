"""Bounded candidate-only signal propagation for Kraken-R Round 4.

This module deliberately does not wrap any legacy dispatcher, tick loop,
replay buffer, router, tracker, pressure monitor, or adaptive learning
mechanism. It adapts only the useful concepts: named signals, deterministic
ticks, explicit effects, and finite budgets.

Signals can influence the candidate trajectory before execution.  They are
never execution evidence, truth, settlement, or learning authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Any, Iterable, Mapping

from .contracts import Authority, EvidenceGrade, Signal


class SignalPropagationError(ValueError):
    """Base error for a rejected candidate signal path."""


class SignalValidationError(SignalPropagationError):
    """Raised when a signal is not bound to the active transaction state."""


class PropagationLimitError(SignalPropagationError):
    """Raised when a finite propagation budget would be exceeded."""


class PropagationEffect(str, Enum):
    """Explicit deterministic effect of a signal rule."""

    PASS = "pass"
    AMPLIFY = "amplify"
    INHIBIT = "inhibit"


@dataclass(frozen=True)
class SignalRule:
    """A static, named edge in the candidate signal graph."""

    rule_id: str
    source_topic: str
    target_topic: str
    effect: PropagationEffect = PropagationEffect.PASS
    amplification_factor: int = 1

    def __post_init__(self) -> None:
        for value, name in (
            (self.rule_id, "rule_id"),
            (self.source_topic, "source_topic"),
            (self.target_topic, "target_topic"),
        ):
            if not isinstance(value, str) or not value.strip() or any(
                char.isspace() for char in value
            ):
                raise SignalPropagationError(f"{name} must be a non-empty identifier")
        try:
            effect = PropagationEffect(self.effect)
        except (TypeError, ValueError) as exc:
            raise SignalPropagationError("effect is not supported") from exc
        object.__setattr__(self, "effect", effect)
        if (
            not isinstance(self.amplification_factor, int)
            or self.amplification_factor < 1
            or self.amplification_factor > 4
        ):
            raise SignalPropagationError(
                "amplification_factor must be an integer from 1 through 4"
            )
        if effect is PropagationEffect.AMPLIFY and self.amplification_factor == 1:
            raise SignalPropagationError(
                "amplify rules must declare a factor greater than one"
            )


@dataclass(frozen=True)
class SignalStep:
    """One deterministic delivery or explicit inhibition decision.

    ``source`` and ``cause`` describe the signal represented by
    ``signal_id``.  When the step creates a derived signal,
    ``derived_source`` and ``derived_cause`` describe that derived signal
    without requiring a reviewer to reconstruct it from the signal network.
    """

    tick: int
    signal_id: str
    topic: str
    priority: int
    rule_id: str | None
    effect: PropagationEffect | None
    disposition: str
    derived_signal_id: str | None = None
    source: str | None = None
    cause: str | None = None
    derived_source: str | None = None
    derived_cause: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "signal_id": self.signal_id,
            "topic": self.topic,
            "priority": self.priority,
            "rule_id": self.rule_id,
            "effect": self.effect.value if self.effect else None,
            "disposition": self.disposition,
            "derived_signal_id": self.derived_signal_id,
            "source": self.source,
            "cause": self.cause,
            "derived_source": self.derived_source,
            "derived_cause": self.derived_cause,
        }


@dataclass(frozen=True)
class PropagationTrace:
    """Immutable record of a bounded signal propagation run."""

    transaction_id: str
    objective_id: str
    task_state_id: str
    task_state_version: int
    initial_signal_ids: tuple[str, ...]
    delivered_signal_ids: tuple[str, ...]
    steps: tuple[SignalStep, ...]
    inhibited_topics: tuple[str, ...] = ()
    amplified_signal_ids: tuple[str, ...] = ()
    duplicate_signal_ids: tuple[str, ...] = ()
    expired_signal_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_signal_ids", tuple(self.initial_signal_ids))
        object.__setattr__(
            self, "delivered_signal_ids", tuple(self.delivered_signal_ids)
        )
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "inhibited_topics", tuple(self.inhibited_topics))
        object.__setattr__(
            self, "amplified_signal_ids", tuple(self.amplified_signal_ids)
        )
        object.__setattr__(
            self, "duplicate_signal_ids", tuple(self.duplicate_signal_ids)
        )
        object.__setattr__(self, "expired_signal_ids", tuple(self.expired_signal_ids))

    def inhibits(self, topic: str) -> bool:
        return topic in self.inhibited_topics

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "objective_id": self.objective_id,
            "task_state_id": self.task_state_id,
            "task_state_version": self.task_state_version,
            "initial_signal_ids": list(self.initial_signal_ids),
            "delivered_signal_ids": list(self.delivered_signal_ids),
            "steps": [step.to_dict() for step in self.steps],
            "inhibited_topics": list(self.inhibited_topics),
            "amplified_signal_ids": list(self.amplified_signal_ids),
            "duplicate_signal_ids": list(self.duplicate_signal_ids),
            "expired_signal_ids": list(self.expired_signal_ids),
        }


@dataclass(frozen=True)
class SignalReplayRecord:
    """Immutable input envelope for deterministic signal-path replay."""

    transaction_id: str
    objective_id: str
    task_state_id: str
    task_state_version: int
    signals: tuple[Signal, ...]
    start_tick: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "signals", tuple(self.signals))
        if not self.transaction_id or not self.objective_id or not self.task_state_id:
            raise SignalValidationError("replay identity fields are required")
        if self.task_state_version < 1:
            raise SignalValidationError("replay task-state version must be positive")
        if self.start_tick < 0:
            raise SignalValidationError("replay start_tick must be non-negative")


def make_bound_signal(
    signal_id: str,
    topic: str,
    *,
    transaction_id: str,
    objective_id: str,
    task_state_id: str,
    task_state_version: int,
    producer: str = "kraken_r_candidate",
    payload: Mapping[str, Any] | None = None,
    ttl: int = 2,
    created_tick: int = 0,
    source: str = "candidate_signal_network",
    cause: str | None = None,
    priority: int = 0,
) -> Signal:
    """Construct one canonical Signal with the Round 4 binding envelope."""

    provenance = {
        "transaction_id": transaction_id,
        "objective_id": objective_id,
        "task_state_id": task_state_id,
        "task_state_version": task_state_version,
        "signal_origin": "candidate_signal_network",
    }
    return Signal(
        signal_id,
        topic,
        transaction_id,
        producer,
        payload=payload or {},
        evidence_grade=EvidenceGrade.DECLARED,
        authority=Authority.KRAKEN_CANDIDATE,
        task_state_id=task_state_id,
        task_state_version=task_state_version,
        provenance=provenance,
        ttl=ttl,
        created_tick=created_tick,
        source=source,
        cause=signal_id if cause is None else cause,
        priority=priority,
    )


class SignalNetwork:
    """Pure in-memory deterministic propagation with finite budgets."""

    DEFAULT_RULES = (
        SignalRule(
            "uncertainty-inhibits-action",
            "candidate.uncertainty",
            "candidate.action.authorize",
            PropagationEffect.INHIBIT,
        ),
        SignalRule(
            "urgency-amplifies-observation",
            "candidate.urgency",
            "candidate.observation.requested",
            PropagationEffect.AMPLIFY,
            amplification_factor=2,
        ),
    )

    def __init__(
        self,
        rules: Iterable[SignalRule] = DEFAULT_RULES,
        *,
        max_ticks: int = 8,
        max_deliveries: int = 32,
        max_fanout: int = 8,
    ) -> None:
        self._rules = tuple(sorted(tuple(rules), key=lambda rule: rule.rule_id))
        if any(not isinstance(rule, SignalRule) for rule in self._rules):
            raise SignalPropagationError("signal rules must be SignalRule records")
        rule_ids = [rule.rule_id for rule in self._rules]
        if len(set(rule_ids)) != len(rule_ids):
            duplicates = sorted({rid for rid in rule_ids if rule_ids.count(rid) > 1})
            raise SignalPropagationError(
                f"duplicate signal rule ids: {', '.join(duplicates)}"
            )
        if max_ticks < 0 or max_deliveries < 1 or max_fanout < 1:
            raise SignalPropagationError("propagation budgets must be positive")
        self._max_ticks = max_ticks
        self._max_deliveries = max_deliveries
        self._max_fanout = max_fanout

    @property
    def rules(self) -> tuple[SignalRule, ...]:
        return self._rules

    def propagate(
        self,
        signals: Iterable[Signal],
        *,
        transaction_id: str,
        objective_id: str,
        task_state_id: str,
        task_state_version: int,
        start_tick: int = 0,
    ) -> PropagationTrace:
        if not transaction_id or not objective_id or not task_state_id:
            raise SignalValidationError("propagation identity fields are required")
        if task_state_version < 1 or start_tick < 0:
            raise SignalValidationError("propagation state or tick is invalid")

        initial = tuple(signals)
        if any(not isinstance(signal, Signal) for signal in initial):
            raise SignalValidationError(
                "propagation accepts canonical Signal records only"
            )
        supported_topics = {
            topic
            for rule in self._rules
            for topic in (rule.source_topic, rule.target_topic)
        }
        unsupported_topics = sorted(
            {signal.topic for signal in initial if signal.topic not in supported_topics}
        )
        if unsupported_topics:
            raise SignalValidationError(
                f"unsupported signal topic: {unsupported_topics[0]}"
            )
        queue: list[tuple[int, int, str, int, Signal]] = [
            (start_tick, -signal.priority, signal.signal_id, index, signal)
            for index, signal in enumerate(initial)
        ]
        queue.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        seen: set[str] = set()
        delivered: list[str] = []
        steps: list[SignalStep] = []
        inhibited: set[str] = set()
        amplified: list[str] = []
        duplicates: list[str] = []
        expired: list[str] = []

        while queue:
            tick, _, _, _, signal = queue.pop(0)
            self._validate_signal(
                signal,
                transaction_id=transaction_id,
                objective_id=objective_id,
                task_state_id=task_state_id,
                task_state_version=task_state_version,
                current_tick=tick,
                start_tick=start_tick,
            )
            if signal.topic not in supported_topics:
                raise SignalValidationError(
                    f"unsupported signal topic: {signal.topic}"
                )
            try:
                dedup_key = signal.dedup_key()
            except (TypeError, ValueError) as exc:
                raise SignalValidationError(
                    "signal must contain deterministic JSON-compatible data"
                ) from exc
            if dedup_key in seen:
                duplicates.append(signal.signal_id)
                steps.append(
                    SignalStep(
                        tick,
                        signal.signal_id,
                        signal.topic,
                        signal.priority,
                        None,
                        None,
                        "duplicate",
                        source=signal.source,
                        cause=signal.cause,
                    )
                )
                continue
            seen.add(dedup_key)
            if len(delivered) >= self._max_deliveries:
                raise PropagationLimitError(
                    f"maximum signal deliveries exceeded ({self._max_deliveries})"
                )
            if tick > start_tick + self._max_ticks:
                raise PropagationLimitError(
                    f"maximum signal ticks exceeded ({self._max_ticks})"
                )
            delivered.append(signal.signal_id)
            matching = tuple(
                rule for rule in self._rules if rule.source_topic == signal.topic
            )
            if len(matching) > self._max_fanout:
                raise PropagationLimitError(
                    f"maximum signal fan-out exceeded ({self._max_fanout})"
                )
            if not matching:
                steps.append(
                    SignalStep(
                        tick,
                        signal.signal_id,
                        signal.topic,
                        signal.priority,
                        None,
                        None,
                        "delivered",
                        source=signal.source,
                        cause=signal.cause,
                    )
                )
                continue

            for rule in matching:
                if rule.effect is PropagationEffect.INHIBIT:
                    inhibited.add(rule.target_topic)
                    steps.append(
                        SignalStep(
                            tick,
                            signal.signal_id,
                            signal.topic,
                            signal.priority,
                            rule.rule_id,
                            rule.effect,
                            "inhibited",
                            source=signal.source,
                            cause=signal.cause,
                        )
                    )
                    continue

                derived_id = self._derived_id(signal, rule, tick)
                payload = dict(signal.payload)
                payload["propagated_from"] = signal.signal_id
                prior_depth = payload.get("propagation_depth", 0)
                if not isinstance(prior_depth, int) or prior_depth < 0:
                    raise SignalValidationError(
                        "signal propagation_depth must be a non-negative integer"
                    )
                payload["propagation_depth"] = prior_depth + 1
                if rule.effect is PropagationEffect.AMPLIFY:
                    payload["amplification_factor"] = rule.amplification_factor
                    amplified.append(signal.signal_id)
                derived = Signal(
                    derived_id,
                    rule.target_topic,
                    signal.correlation_id,
                    "kraken_r_signal_network",
                    payload=payload,
                    evidence_grade=EvidenceGrade.DECLARED,
                    authority=Authority.KRAKEN_CANDIDATE,
                    task_state_id=task_state_id,
                    task_state_version=task_state_version,
                    provenance={
                        **dict(signal.provenance),
                        "propagated_from": signal.signal_id,
                        "signal_origin": "candidate_signal_network",
                    },
                    ttl=max(0, (signal.ttl or 0) - 1),
                    created_tick=tick + 1,
                    source="kraken_r_signal_network",
                    cause=signal.signal_id,
                    priority=signal.priority,
                )
                if derived.created_tick > signal.created_tick + (signal.ttl or 0):
                    expired.append(derived.signal_id)
                    steps.append(
                        SignalStep(
                            tick,
                            signal.signal_id,
                            signal.topic,
                            signal.priority,
                            rule.rule_id,
                            rule.effect,
                            "expired",
                            derived_signal_id=derived.signal_id,
                            source=signal.source,
                            cause=signal.cause,
                            derived_source=derived.source,
                            derived_cause=derived.cause,
                        )
                    )
                    continue
                queue.append(
                    (
                        derived.created_tick,
                        -derived.priority,
                        derived.signal_id,
                        len(queue),
                        derived,
                    )
                )
                queue.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
                steps.append(
                    SignalStep(
                        tick,
                        signal.signal_id,
                        signal.topic,
                        signal.priority,
                        rule.rule_id,
                        rule.effect,
                        "propagated",
                        derived_signal_id=derived.signal_id,
                        source=signal.source,
                        cause=signal.cause,
                        derived_source=derived.source,
                        derived_cause=derived.cause,
                    )
                )

        return PropagationTrace(
            transaction_id=transaction_id,
            objective_id=objective_id,
            task_state_id=task_state_id,
            task_state_version=task_state_version,
            initial_signal_ids=tuple(signal.signal_id for signal in initial),
            delivered_signal_ids=tuple(delivered),
            steps=tuple(steps),
            inhibited_topics=tuple(sorted(inhibited)),
            amplified_signal_ids=tuple(amplified),
            duplicate_signal_ids=tuple(duplicates),
            expired_signal_ids=tuple(expired),
        )

    @staticmethod
    def _derived_id(signal: Signal, rule: SignalRule, tick: int) -> str:
        raw = f"{signal.signal_id}:{rule.rule_id}:{tick}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _validate_signal(
        signal: Signal,
        *,
        transaction_id: str,
        objective_id: str,
        task_state_id: str,
        task_state_version: int,
        current_tick: int,
        start_tick: int,
    ) -> None:
        if not isinstance(signal, Signal):
            raise SignalValidationError("propagation accepts canonical Signal records only")
        if signal.authority is not Authority.KRAKEN_CANDIDATE:
            raise SignalValidationError("signal authority must be kraken_candidate")
        if signal.evidence_grade is not EvidenceGrade.DECLARED:
            raise SignalValidationError(
                "signals are declared messages and cannot claim evidence"
            )
        if signal.source is None or signal.cause is None:
            raise SignalValidationError(
                "candidate signals require explicit source and cause identities"
            )
        if signal.task_state_id != task_state_id:
            raise SignalValidationError("signal task-state binding is stale or mismatched")
        if signal.task_state_version != task_state_version:
            raise SignalValidationError("signal task-state version is stale or mismatched")
        expected = {
            "transaction_id": transaction_id,
            "objective_id": objective_id,
            "task_state_id": task_state_id,
            "task_state_version": task_state_version,
        }
        for key, value in expected.items():
            if signal.provenance.get(key) != value:
                raise SignalValidationError(f"signal provenance mismatch: {key}")
        if signal.ttl is None:
            raise SignalValidationError("signal must declare a TTL")
        if signal.created_tick > current_tick:
            raise SignalValidationError("signal originates after its delivery tick")
        if current_tick > signal.created_tick + signal.ttl:
            raise SignalValidationError("signal TTL has expired")


def replay_signal_propagation(
    record: SignalReplayRecord,
    *,
    network: SignalNetwork | None = None,
) -> PropagationTrace:
    """Replay one immutable signal envelope through a fresh bounded network."""

    return (network or SignalNetwork()).propagate(
        record.signals,
        transaction_id=record.transaction_id,
        objective_id=record.objective_id,
        task_state_id=record.task_state_id,
        task_state_version=record.task_state_version,
        start_tick=record.start_tick,
    )


__all__ = [
    "PropagationEffect",
    "PropagationLimitError",
    "PropagationTrace",
    "SignalNetwork",
    "SignalReplayRecord",
    "SignalPropagationError",
    "SignalRule",
    "SignalStep",
    "SignalValidationError",
    "make_bound_signal",
    "replay_signal_propagation",
]