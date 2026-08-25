"""Bounded Stage 10.9 metastability experiments over the Stage 10.8 reducer.

Experiments in this module are descriptive counterfactuals.  They replay
caller-owned immutable ticks, collect measurements from reducer traces, and
return disposable reports.  They do not create evidence, adaptive credit,
canonical state, a controller, or a preferred topology.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
from typing import Any, Iterable

from .dynamical_substrate import (
    MAX_TICKS,
    MAX_TICK_EVENTS,
    DynamicalEvent,
    DynamicalState,
    DynamicalTick,
    DynamicalTickTrace,
    _reduce_dynamical_tick_counterfactual,
    reduce_dynamical_tick,
)


MAX_EXPERIMENT_TICKS = MAX_TICKS
MAX_SCENARIO_NAME = 64
MAX_FAILURE_MODES = 8
COUNTERFORCES = ("homeostasis", "decay", "inhibition", "surprise")


class MetastabilityValidationError(ValueError):
    """Raised when an experiment crosses its bounded descriptive boundary."""


class ExperimentScenario(str, Enum):
    STABLE = "stable"
    REGIME_CHANGE = "regime_change"
    NOISY_NONCREDITABLE = "noisy_noncreditable"
    OSCILLATORY = "oscillatory"
    MONOPOLIZING = "monopolizing"
    STAGNATING = "stagnating"


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(char.isspace() for char in value):
        raise MetastabilityValidationError(
            f"{name} must be a non-empty identifier without whitespace"
        )
    return value


def _text(value: str, name: str, maximum: int = MAX_SCENARIO_NAME) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise MetastabilityValidationError(f"{name} must be bounded non-empty text")
    return value


def _bounded_tuple(values: Iterable[Any], maximum: int, name: str) -> tuple[Any, ...]:
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise MetastabilityValidationError(f"{name} must be iterable") from exc
    result: list[Any] = []
    for value in iterator:
        if len(result) >= maximum:
            raise MetastabilityValidationError(f"{name} budget exceeded")
        result.append(value)
    return tuple(result)


def _unit(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetastabilityValidationError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise MetastabilityValidationError(f"{name} must be between 0 and 1")
    return round(value, 6)


def _nonnegative(
    value: int, name: str, maximum: int = MAX_EXPERIMENT_TICKS
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise MetastabilityValidationError(f"{name} must be a bounded non-negative integer")
    return value


def _rate(value: float, name: str) -> float:
    return _unit(value, name)


def _entropy(values: tuple[str, ...]) -> float:
    if not values:
        return 0.0
    counts = {value: values.count(value) for value in set(values)}
    if len(counts) <= 1:
        return 0.0
    total = float(len(values))
    raw = -sum((count / total) * math.log(count / total) for count in counts.values())
    return round(raw / math.log(len(counts)), 6)


@dataclass(frozen=True)
class MetastabilityAblation:
    """Explicit counterforce settings for one disposable comparison run."""

    homeostasis: bool = True
    decay: bool = True
    inhibition: bool = True
    surprise: bool = True
    anti_monopoly: bool = True
    monopoly_route_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "homeostasis",
            "decay",
            "inhibition",
            "surprise",
            "anti_monopoly",
        ):
            if not isinstance(getattr(self, name), bool):
                raise MetastabilityValidationError(f"{name} ablation control must be boolean")
        if self.monopoly_route_id is not None:
            _identifier(self.monopoly_route_id, "monopoly_route_id")

    @classmethod
    def full(cls) -> "MetastabilityAblation":
        return cls()

    @classmethod
    def without(cls, counterforce: str, *, monopoly_route_id: str | None = None) -> "MetastabilityAblation":
        if counterforce not in (*COUNTERFORCES, "anti_monopoly"):
            raise MetastabilityValidationError(f"unsupported counterforce: {counterforce}")
        values = {
            "homeostasis": True,
            "decay": True,
            "inhibition": True,
            "surprise": True,
            "anti_monopoly": True,
        }
        values[counterforce] = False
        return cls(**values, monopoly_route_id=monopoly_route_id)

    @classmethod
    def matched_set(cls) -> tuple["MetastabilityAblation", ...]:
        return (cls.full(),) + tuple(cls.without(name) for name in (*COUNTERFORCES, "anti_monopoly"))

    @property
    def label(self) -> str:
        if all(
            getattr(self, name)
            for name in ("homeostasis", "decay", "inhibition", "surprise", "anti_monopoly")
        ):
            return "full"
        disabled = tuple(
            name
            for name in ("homeostasis", "decay", "inhibition", "surprise", "anti_monopoly")
            if not getattr(self, name)
        )
        return "without_" + "_".join(disabled)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "homeostasis": self.homeostasis,
            "decay": self.decay,
            "inhibition": self.inhibition,
            "surprise": self.surprise,
            "anti_monopoly": self.anti_monopoly,
            "monopoly_route_id": self.monopoly_route_id,
        }


@dataclass(frozen=True)
class MetastabilityTick:
    """One bounded reducer input plus an optional experimental route choice."""

    dynamical_tick: DynamicalTick
    route_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dynamical_tick, DynamicalTick):
            raise MetastabilityValidationError("experiment tick requires DynamicalTick")
        if self.route_id is not None:
            _identifier(self.route_id, "route_id")

    @property
    def tick(self) -> int:
        return self.dynamical_tick.tick

    def to_dict(self) -> dict[str, Any]:
        return {
            "dynamical_tick": self.dynamical_tick.to_dict(),
            "route_id": self.route_id,
        }


@dataclass(frozen=True)
class MetastabilityMetrics:
    """Bounded descriptive measurements derived only from immutable traces."""

    tick_count: int
    route_diversity: float
    route_entropy: float
    dominant_route_share: float
    topology_turnover: int
    plasticity_rate: float
    decay_rate: float
    rollback_rate: float
    inhibition_rate: float
    surprise_mean: float
    surprise_peak: float
    recurrence: int
    coherence: float
    stagnation_duration: int
    resource_pressure_mean: float
    resource_pressure_peak: float
    task_performance: float | None
    task_observations: int
    regime_transitions: int
    observed_regimes: tuple[str, ...]
    operating_region: str
    failure_modes: tuple[str, ...]
    noncreditable_events: int = 0

    def __post_init__(self) -> None:
        _nonnegative(self.tick_count, "tick_count")
        for name in (
            "route_diversity",
            "route_entropy",
            "dominant_route_share",
            "plasticity_rate",
            "decay_rate",
            "rollback_rate",
            "inhibition_rate",
            "surprise_mean",
            "surprise_peak",
            "coherence",
            "resource_pressure_mean",
            "resource_pressure_peak",
        ):
            _unit(getattr(self, name), name)
        for name in (
            "topology_turnover",
            "recurrence",
            "stagnation_duration",
            "regime_transitions",
        ):
            _nonnegative(getattr(self, name), name)
        _nonnegative(
            self.task_observations,
            "task_observations",
            MAX_EXPERIMENT_TICKS * MAX_TICK_EVENTS,
        )
        _nonnegative(
            self.noncreditable_events,
            "noncreditable_events",
            MAX_EXPERIMENT_TICKS * MAX_TICK_EVENTS,
        )
        if self.task_performance is not None:
            _unit(self.task_performance, "task_performance")
        regimes = _bounded_tuple(self.observed_regimes, len(ExperimentScenario), "observed_regimes")
        failures = _bounded_tuple(self.failure_modes, MAX_FAILURE_MODES, "failure_modes")
        if not all(isinstance(value, str) and value for value in regimes + failures):
            raise MetastabilityValidationError("metric labels must be non-empty strings")
        _text(self.operating_region, "operating_region")
        object.__setattr__(self, "observed_regimes", regimes)
        object.__setattr__(self, "failure_modes", failures)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick_count": self.tick_count,
            "route_diversity": self.route_diversity,
            "route_entropy": self.route_entropy,
            "dominant_route_share": self.dominant_route_share,
            "topology_turnover": self.topology_turnover,
            "plasticity_rate": self.plasticity_rate,
            "decay_rate": self.decay_rate,
            "rollback_rate": self.rollback_rate,
            "inhibition_rate": self.inhibition_rate,
            "surprise_mean": self.surprise_mean,
            "surprise_peak": self.surprise_peak,
            "recurrence": self.recurrence,
            "coherence": self.coherence,
            "stagnation_duration": self.stagnation_duration,
            "resource_pressure_mean": self.resource_pressure_mean,
            "resource_pressure_peak": self.resource_pressure_peak,
            "task_performance": self.task_performance,
            "task_observations": self.task_observations,
            "regime_transitions": self.regime_transitions,
            "observed_regimes": list(self.observed_regimes),
            "operating_region": self.operating_region,
            "failure_modes": list(self.failure_modes),
            "noncreditable_events": self.noncreditable_events,
        }


@dataclass(frozen=True)
class MetastabilityReport:
    """One immutable run report; it contains no promoted or canonical state."""

    experiment_id: str
    scenario: ExperimentScenario
    ablation: MetastabilityAblation
    input_digest: str
    metrics: MetastabilityMetrics
    final_state: DynamicalState
    traces: tuple[DynamicalTickTrace, ...]
    route_history: tuple[str, ...]
    topology_versions: tuple[int, ...]

    def __post_init__(self) -> None:
        _identifier(self.experiment_id, "experiment_id")
        try:
            scenario = ExperimentScenario(self.scenario)
        except (TypeError, ValueError) as exc:
            raise MetastabilityValidationError("scenario is unsupported") from exc
        object.__setattr__(self, "scenario", scenario)
        if not isinstance(self.ablation, MetastabilityAblation):
            raise MetastabilityValidationError("report ablation is invalid")
        if not isinstance(self.final_state, DynamicalState):
            raise MetastabilityValidationError("report final state is invalid")
        traces = _bounded_tuple(self.traces, MAX_EXPERIMENT_TICKS, "traces")
        routes = _bounded_tuple(self.route_history, MAX_EXPERIMENT_TICKS, "route_history")
        versions = _bounded_tuple(
            self.topology_versions, MAX_EXPERIMENT_TICKS, "topology_versions"
        )
        if not all(isinstance(item, DynamicalTickTrace) for item in traces):
            raise MetastabilityValidationError("report traces are invalid")
        if not all(isinstance(item, str) and item for item in routes):
            raise MetastabilityValidationError("report route history is invalid")
        if not all(isinstance(item, int) and item >= 0 for item in versions):
            raise MetastabilityValidationError("report topology versions are invalid")
        if len(traces) != self.metrics.tick_count:
            raise MetastabilityValidationError("report trace count disagrees with metrics")
        object.__setattr__(self, "traces", traces)
        object.__setattr__(self, "route_history", routes)
        object.__setattr__(self, "topology_versions", versions)
        if not isinstance(self.input_digest, str) or len(self.input_digest) != 64:
            raise MetastabilityValidationError("input_digest must be a SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "scenario": self.scenario.value,
            "ablation": self.ablation.to_dict(),
            "input_digest": self.input_digest,
            "metrics": self.metrics.to_dict(),
            "final_state": self.final_state.to_dict(),
            "traces": [trace.to_dict() for trace in self.traces],
            "route_history": list(self.route_history),
            "topology_versions": list(self.topology_versions),
            "authority": "kraken_candidate_experiment_observation",
            "promotes_canonical_state": False,
        }


@dataclass(frozen=True)
class MetastabilityComparison:
    """Matched-budget full-vs-ablation reports over identical raw inputs."""

    comparison_id: str
    scenario: ExperimentScenario
    reports: tuple[MetastabilityReport, ...]

    def __post_init__(self) -> None:
        _identifier(self.comparison_id, "comparison_id")
        reports = _bounded_tuple(self.reports, 1 + len(COUNTERFORCES) + 1, "reports")
        if not reports:
            raise MetastabilityValidationError("comparison requires a full baseline")
        if not all(isinstance(report, MetastabilityReport) for report in reports):
            raise MetastabilityValidationError("comparison reports are invalid")
        baseline = reports[0]
        expected_labels = ("full",) + tuple(
            f"without_{name}" for name in (*COUNTERFORCES, "anti_monopoly")
        )
        if tuple(report.ablation.label for report in reports) != expected_labels:
            raise MetastabilityValidationError(
                "comparison must contain one full arm and every single-factor ablation"
            )
        if any(
            report.scenario is not baseline.scenario
            or report.input_digest != baseline.input_digest
            or report.metrics.tick_count != baseline.metrics.tick_count
            for report in reports
        ):
            raise MetastabilityValidationError("comparison reports are not matched")
        object.__setattr__(self, "scenario", baseline.scenario)
        object.__setattr__(self, "reports", reports)

    @property
    def baseline(self) -> MetastabilityReport:
        return self.reports[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "scenario": self.scenario.value,
            "input_digest": self.baseline.input_digest,
            "reports": [report.to_dict() for report in self.reports],
            "matched_budget": True,
            "authority": "kraken_candidate_experiment_observation",
        }


def _normalize_ticks(
    ticks: Iterable[MetastabilityTick | DynamicalTick],
) -> tuple[MetastabilityTick, ...]:
    normalized: list[MetastabilityTick] = []
    for item in _bounded_tuple(ticks, MAX_EXPERIMENT_TICKS, "experiment ticks"):
        if isinstance(item, MetastabilityTick):
            normalized.append(item)
        elif isinstance(item, DynamicalTick):
            normalized.append(MetastabilityTick(item))
        else:
            raise MetastabilityValidationError("experiment ticks must be DynamicalTick records")
    for index, item in enumerate(normalized):
        expected_tick = index + 1
        if item.tick != expected_tick:
            raise MetastabilityValidationError(
                f"experiment ticks must start at 1 and advance exactly; expected {expected_tick}"
            )
    return tuple(normalized)


def _input_digest(ticks: tuple[MetastabilityTick, ...]) -> str:
    encoded = json.dumps(
        [item.to_dict() for item in ticks],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _route_ids(state: DynamicalState) -> tuple[str, ...]:
    return tuple(route.route_id for route in state.medium.adaptive_state.route_topology.routes)


def _select_experiment_route(
    state: DynamicalState,
    requested: str | None,
    ablation: MetastabilityAblation,
    route_history: tuple[str, ...],
) -> str | None:
    available = _route_ids(state)
    route_id = requested or state.fast.active_route_id
    if not available:
        return None
    if not ablation.anti_monopoly:
        route_id = ablation.monopoly_route_id or sorted(available)[0]
    else:
        counts = {candidate: route_history.count(candidate) for candidate in available}
        minimum = min(counts.values())
        least_used = tuple(sorted(candidate for candidate, count in counts.items() if count == minimum))
        route_id = route_id if route_id in least_used else least_used[0]
    if route_id is not None and route_id not in available:
        raise MetastabilityValidationError("experiment route is absent from the declared topology")
    return route_id


def _operating_region(
    scenario: ExperimentScenario,
    *,
    tick_count: int,
    route_history: tuple[str, ...],
    regime_transitions: int,
    stagnation_duration: int,
    noncreditable_events: int,
) -> str:
    if tick_count and stagnation_duration == tick_count:
        return "stagnating"
    if scenario is ExperimentScenario.STABLE:
        return "settling"
    if scenario is ExperimentScenario.MONOPOLIZING:
        return "monopolizing"
    if route_history and len(set(route_history)) == 1:
        return "monopolizing"
    if regime_transitions >= 3:
        return "oscillatory"
    if scenario is ExperimentScenario.REGIME_CHANGE and len(set(route_history)) > 1:
        return "regime_change"
    if noncreditable_events and not regime_transitions:
        return "disturbance_resistant"
    return "settling"


def _failure_modes(
    *,
    ablation: MetastabilityAblation,
    tick_count: int,
    route_history: tuple[str, ...],
    topology_turnover: int,
    regime_transitions: int,
    stagnation_duration: int,
    noncreditable_events: int,
    traces: tuple[DynamicalTickTrace, ...],
) -> tuple[str, ...]:
    failures: list[str] = []
    if (
        not ablation.anti_monopoly
        and route_history
        and len(set(route_history)) == 1
        and tick_count > 1
    ):
        failures.append("lock_in")
    if regime_transitions >= 3:
        failures.append("thrashing")
    if tick_count and stagnation_duration == tick_count:
        failures.append("dead_equilibrium")
    if (
        not ablation.decay
        and route_history
        and any(trace.inactive_decay_route_id is None for trace in traces)
    ):
        failures.append("persistent_attractor_without_decay")
    if not ablation.anti_monopoly and route_history:
        failures.append("anti_monopoly_counterforce_removed")
    if not ablation.inhibition and any(trace.inhibited for trace in traces):
        failures.append("inhibition_measurement_retained_after_ablation")
    if not ablation.surprise and any(trace.physiology.snapshot.contradiction_density for trace in traces):
        failures.append("surprise_measurement_retained_after_ablation")
    if noncreditable_events and topology_turnover == 0:
        failures.append("noncreditable_disturbance_withheld")
    return tuple(failures[:MAX_FAILURE_MODES])


def run_metastability_experiment(
    initial: DynamicalState,
    ticks: Iterable[MetastabilityTick | DynamicalTick],
    *,
    experiment_id: str = "metastability-experiment",
    scenario: ExperimentScenario = ExperimentScenario.STABLE,
    ablation: MetastabilityAblation | None = None,
) -> MetastabilityReport:
    """Replay one bounded comparison arm and derive descriptive metrics."""

    if not isinstance(initial, DynamicalState):
        raise MetastabilityValidationError("experiment requires DynamicalState")
    normalized = _normalize_ticks(ticks)
    if initial.tick != 0:
        raise MetastabilityValidationError("experiments must start from tick zero")
    try:
        scenario = ExperimentScenario(scenario)
    except (TypeError, ValueError) as exc:
        raise MetastabilityValidationError("scenario is unsupported") from exc
    ablation = ablation or MetastabilityAblation.full()
    if not isinstance(ablation, MetastabilityAblation):
        raise MetastabilityValidationError("ablation is invalid")

    state = initial
    authority_state = initial
    authority_route_history: list[str] = []
    traces: list[DynamicalTickTrace] = []
    route_history: list[str] = []
    topology_versions: list[int] = []
    surprises: list[float] = []
    resources: list[float] = []
    regimes: list[str] = []
    regime_transitions = 0
    topology_turnover = 0
    plasticity = 0
    decay = 0
    rollback = 0
    inhibition = 0
    noncreditable = 0
    stagnation_duration = 0
    current_stagnation = 0
    task_successes = 0
    task_failures = 0
    previous_regime: str | None = None
    previous_topology = state.medium.adaptive_state.route_topology.version

    for item in normalized:
        authority_route_id = _select_experiment_route(
            authority_state,
            item.route_id,
            MetastabilityAblation.full(),
            tuple(authority_route_history),
        )
        if authority_route_id is not None:
            authority_state = replace(
                authority_state,
                fast=replace(authority_state.fast, active_route_id=authority_route_id),
            )
        authority_state, authority_trace = reduce_dynamical_tick(
            authority_state, item.dynamical_tick
        )
        authority_chosen = authority_route_id or authority_state.fast.active_route_id
        if authority_chosen is not None:
            authority_route_history.append(authority_chosen)
        route_id = _select_experiment_route(
            state, item.route_id, ablation, tuple(route_history)
        )
        if route_id is not None:
            state = replace(state, fast=replace(state.fast, active_route_id=route_id))
        state, trace = _reduce_dynamical_tick_counterfactual(
            state,
            item.dynamical_tick,
            counterforce_controls={
                "homeostasis": ablation.homeostasis,
                "decay": ablation.decay,
                "inhibition": ablation.inhibition,
                "surprise": ablation.surprise,
            },
            authority_inhibited=authority_trace.inhibited,
            canonical_withheld_event_ids=tuple(
                event.event_id for event in authority_trace.withheld_events
            ),
        )
        traces.append(trace)
        chosen = route_id or state.fast.active_route_id
        if chosen is not None:
            route_history.append(chosen)
        topology_versions.append(state.medium.adaptive_state.route_topology.version)
        if state.medium.adaptive_state.route_topology.version != previous_topology:
            topology_turnover += 1
        previous_topology = state.medium.adaptive_state.route_topology.version
        surprise = trace.physiology.snapshot.contradiction_density
        resource = trace.physiology.snapshot.resource_pressure
        surprises.append(surprise)
        resources.append(resource)
        regime = trace.physiology.decision.regime.value
        regimes.append(regime)
        if previous_regime is not None and previous_regime != regime:
            regime_transitions += 1
        previous_regime = regime
        plasticity += len(trace.adaptive_audits)
        decay += sum(audit.operation == "decay" for audit in trace.adaptive_audits)
        decay += trace.inactive_decay_route_id is not None
        rollback += sum(audit.operation == "rollback" for audit in trace.adaptive_audits)
        inhibition += trace.inhibited
        noncreditable += len(trace.noncreditable_settlement_ids)
        if not trace.event_ids:
            current_stagnation += 1
            stagnation_duration = max(stagnation_duration, current_stagnation)
        else:
            current_stagnation = 0
        for audit in trace.adaptive_audits:
            if audit.epistemic_class == "task_success":
                task_successes += 1
            elif audit.epistemic_class == "task_failure":
                task_failures += 1

    tick_count = len(traces)
    route_count = len(_route_ids(initial))
    task_observations = task_successes + task_failures
    event_capacity = max(1, tick_count * MAX_TICK_EVENTS)
    metrics = MetastabilityMetrics(
        tick_count=tick_count,
        route_diversity=(
            round(len(set(route_history)) / route_count, 6)
            if route_count
            else 0.0
        ),
        route_entropy=_entropy(tuple(route_history)),
        dominant_route_share=(
            round(max(route_history.count(route) for route in set(route_history)) / len(route_history), 6)
            if route_history
            else 0.0
        ),
        topology_turnover=topology_turnover,
        plasticity_rate=_rate(plasticity / event_capacity, "plasticity_rate"),
        decay_rate=_rate(decay / event_capacity, "decay_rate"),
        rollback_rate=_rate(rollback / event_capacity, "rollback_rate"),
        inhibition_rate=_rate(inhibition / tick_count if tick_count else 0.0, "inhibition_rate"),
        surprise_mean=_rate(sum(surprises) / tick_count if tick_count else 0.0, "surprise_mean"),
        surprise_peak=max(surprises, default=0.0),
        recurrence=state.slow.recurrence,
        coherence=state.slow.coherence_score,
        stagnation_duration=stagnation_duration,
        resource_pressure_mean=_rate(sum(resources) / tick_count if tick_count else 0.0, "resource_pressure_mean"),
        resource_pressure_peak=max(resources, default=0.0),
        task_performance=(
            round(task_successes / task_observations, 6)
            if task_observations
            else None
        ),
        task_observations=task_observations,
        regime_transitions=regime_transitions,
        observed_regimes=tuple(dict.fromkeys(regimes)),
        operating_region=_operating_region(
            scenario,
            tick_count=tick_count,
            route_history=tuple(route_history),
            regime_transitions=regime_transitions,
            stagnation_duration=stagnation_duration,
            noncreditable_events=noncreditable,
        ),
        failure_modes=_failure_modes(
            ablation=ablation,
            tick_count=tick_count,
            route_history=tuple(route_history),
            topology_turnover=topology_turnover,
            regime_transitions=regime_transitions,
            stagnation_duration=stagnation_duration,
            noncreditable_events=noncreditable,
            traces=tuple(traces),
        ),
        noncreditable_events=noncreditable,
    )
    return MetastabilityReport(
        experiment_id,
        scenario,
        ablation,
        _input_digest(normalized),
        metrics,
        state,
        tuple(traces),
        tuple(route_history),
        tuple(topology_versions),
    )


def compare_metastability(
    initial: DynamicalState,
    ticks: Iterable[MetastabilityTick | DynamicalTick],
    *,
    comparison_id: str = "metastability-comparison",
    scenario: ExperimentScenario = ExperimentScenario.STABLE,
    ablations: Iterable[MetastabilityAblation] | None = None,
) -> MetastabilityComparison:
    """Run full and matched counterforce ablations from the same raw tick stream."""

    normalized = _normalize_ticks(ticks)
    arms = (
        _bounded_tuple(ablations, 1 + len(COUNTERFORCES) + 1, "ablation arms")
        if ablations is not None
        else MetastabilityAblation.matched_set()
    )
    expected_labels = ("full",) + tuple(
        f"without_{name}" for name in (*COUNTERFORCES, "anti_monopoly")
    )
    if tuple(arm.label for arm in arms) != expected_labels:
        raise MetastabilityValidationError(
            "matched comparison must contain one full arm and every single-factor ablation"
        )
    reports = tuple(
        run_metastability_experiment(
            initial,
            normalized,
            experiment_id=f"{comparison_id}-{arm.label}",
            scenario=scenario,
            ablation=arm,
        )
        for arm in arms
    )
    return MetastabilityComparison(comparison_id, scenario, reports)


def make_metastability_scenario(
    initial: DynamicalState,
    scenario: ExperimentScenario,
    *,
    ticks: int = 32,
) -> tuple[MetastabilityTick, ...]:
    """Create a deterministic, bounded input world for a named scenario."""

    if not isinstance(initial, DynamicalState):
        raise MetastabilityValidationError("scenario requires DynamicalState")
    _nonnegative(ticks, "ticks")
    try:
        scenario = ExperimentScenario(scenario)
    except (TypeError, ValueError) as exc:
        raise MetastabilityValidationError("scenario is unsupported") from exc
    route_ids = _route_ids(initial)
    primary = route_ids[0] if route_ids else None
    secondary = route_ids[1] if len(route_ids) > 1 else primary
    result: list[MetastabilityTick] = []
    for number in range(1, ticks + 1):
        route_id = primary
        events = ()
        if scenario is ExperimentScenario.STABLE:
            events = (DynamicalEventPlaceholder.observation(number, stable=True),)
        elif scenario is ExperimentScenario.REGIME_CHANGE:
            route_id = primary if number <= ticks // 2 else secondary
            if number <= ticks // 2:
                event = DynamicalEventPlaceholder.observation(number, stable=True)
            else:
                event = DynamicalEventPlaceholder.resource(number, 1.0)
            events = (event,)
        elif scenario is ExperimentScenario.NOISY_NONCREDITABLE:
            event = (
                DynamicalEventPlaceholder.observation(number, stable=False)
                if number % 2
                else DynamicalEventPlaceholder.resource(number, 0.8)
            )
            events = (event,)
            route_id = primary if number % 2 else secondary
        elif scenario is ExperimentScenario.OSCILLATORY:
            route_id = primary if number % 2 else secondary
            events = (
                DynamicalEventPlaceholder.resource(number, 0.2 if number % 2 else 0.8),
            )
        elif scenario is ExperimentScenario.MONOPOLIZING:
            events = (DynamicalEventPlaceholder.observation(number, stable=True),)
        elif scenario is ExperimentScenario.STAGNATING:
            events = ()
        result.append(MetastabilityTick(DynamicalTick(number, events), route_id))
    return tuple(result)


class DynamicalEventPlaceholder:
    """Private scenario factory kept separate from the public event constructors."""

    @staticmethod
    def observation(number: int, *, stable: bool):
        from .dynamical_substrate import DynamicalEvent

        return DynamicalEvent.observation_event(
            f"metastability-observation-{number}",
            0.50,
            0.50 if stable else (1.0 if number % 3 else 0.0),
        )

    @staticmethod
    def resource(number: int, pressure: float):
        from .dynamical_substrate import DynamicalEvent

        return DynamicalEvent.resource_event(
            f"metastability-resource-{number}",
            pressure,
        )


__all__ = [
    "ExperimentScenario",
    "MAX_EXPERIMENT_TICKS",
    "MetastabilityAblation",
    "MetastabilityComparison",
    "MetastabilityMetrics",
    "MetastabilityReport",
    "MetastabilityTick",
    "MetastabilityValidationError",
    "compare_metastability",
    "make_metastability_scenario",
    "run_metastability_experiment",
]