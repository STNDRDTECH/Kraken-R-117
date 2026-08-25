"""Standalone inspection command for the candidate Kraken-R foundation."""

from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import contracts
from .cycle import (
    CycleInvariantError,
    CycleMode,
    replay_constitutional_signal_path,
    run_constitutional_cycle,
)
from .nervous_system import (
    SignalNetwork,
    SignalReplayRecord,
    SignalRule,
    SignalValidationError,
    make_bound_signal,
    replay_signal_propagation,
)
from .physiology import (
    OperatingRegime,
    PhysiologyReplayRecord,
    PhysiologySnapshot,
    PhysiologyValidationError,
    RegulationEffect,
    replay_physiology,
)
from .plastic_routing import (
    CandidateRoute,
    PlasticRoutingValidationError,
    RouteTopology,
    SettlementRouteRecord,
    apply_settlement_learning,
    replay_settlement_learning,
    select_candidate_route,
)
from .llm_adapter import (
    AdapterFailureCode,
    CandidateModelContext,
    FixtureModelProvider,
    ModelAdapter,
    ModelAdapterValidationError,
    replay_model_invocation,
)
from .controlled_evaluation import (
    ControlledEvaluationHarness,
    EvaluationMode,
    HeldOutTask,
)
from .grounded_execution import (
    EpistemicOutcomeClass,
    GroundedExecutionExecutor,
    GroundedDeliveryLedger,
    GroundedExecutionRejected,
    GroundedExecutionRequest,
    GroundedExecutionVerifier,
    make_grounded_action,
    replay_grounded_execution,
)
from .replay import (
    RecordedExecution,
    RecordedExecutionMode,
    ReplayValidationError,
    replay_recorded_execution,
)
from .interactions import InteractionValidationError, validate_interaction_chain
from .registry import load_default_registry, load_registry
from .adaptive_substrate import (
    AdaptiveState,
    AdaptiveSubstrateValidationError,
    HomeostaticSnapshot,
    run_orzhaal_experiment,
)
from .dynamical_substrate import (
    DynamicalEvent,
    DynamicalState,
    DynamicalTick,
    DynamicalSubstrateValidationError,
    reduce_dynamical_tick,
    replay_dynamical_ticks,
)
from .metastability import (
    ExperimentScenario,
    MetastabilityValidationError,
    compare_metastability,
    make_metastability_scenario,
    run_metastability_experiment,
)
from .task_integrity import (
    BeliefState,
    CandidateClaim,
    CandidateResult,
    LossKind,
    OriginalTask,
    RequirementKind,
    ReviewRole,
    TaskHypothesis,
    TaskIntegrityValidationError,
    TaskPlan,
    TaskRequirement,
    TaskSpecification,
    replay_task_integrity,
)


CANONICAL_CONTRACTS = (
    "Objective",
    "TaskState",
    "Plan",
    "Hypothesis",
    "Signal",
    "Event",
    "Action",
    "ExecutionResult",
    "Evidence",
    "GroundTruth",
    "Decision",
    "Settlement",
    "LearningUpdate",
    "Capability",
    "Authority",
    "Mutation",
    "Lineage",
    "Regression",
)


def validate_legacy_runtime_boundary() -> tuple[str, ...]:
    """Statically reject candidate imports or authority symbols from legacy runtime."""

    errors: list[str] = []
    candidate_dir = Path(__file__).resolve().parent
    legacy_root = "rogal" + "_" + "core"
    forbidden_names = {
        "Event" + "Bus",
        "ROGAL" + "Daemon",
        "Autonomous" + "Cycle",
        "SQL" + "ite",
        "sub" + "process",
        "sock" + "et",
    }
    for path in sorted(candidate_dir.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"{path.name}: cannot parse candidate source: {exc.msg}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported = (node.module or "",)
            else:
                imported = ()
            for name in imported:
                if name == legacy_root or name.startswith(f"{legacy_root}."):
                    errors.append(f"{path.name}: legacy runtime import is forbidden")
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                errors.append(
                    f"{path.name}: runtime authority symbol {node.id} is forbidden"
                )
    return tuple(sorted(set(errors)))
PLANNED_CAPABILITIES = (
    "metaplasticity",
    "neuromodulation",
    "reservoir_dynamics",
    "criticality_control",
    "organizational_engrams",
    "offline_consolidation",
    "causal_lesion_shadow",
    "developmental_specialization",
    "hyperdimensional_state",
    "hierarchical_learning",
)


class ConstitutionValidationError(ValueError):
    """Raised when the machine-readable constitution metadata is malformed."""


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConstitutionValidationError(f"{field_name} must be a non-empty string")
    return value


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConstitutionValidationError(f"{field_name} must be a list of strings")
    return tuple(value)


def load_constitution_metadata(path: str | Path | None = None) -> Mapping[str, Any]:
    """Load and strictly validate the candidate-only constitution metadata."""

    metadata_path = Path(path) if path else Path(__file__).with_name("constitution.json")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConstitutionValidationError(
            f"Could not load constitution metadata {metadata_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConstitutionValidationError("Constitution metadata root must be an object")

    required = {
        "constitution_kind",
        "version",
        "mission",
        "authority_boundary",
        "canonical_lifecycle",
        "planned_capabilities",
        "preservation_policy",
    }
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ConstitutionValidationError("Constitution metadata fields invalid: " + "; ".join(details))
    if payload["constitution_kind"] != "kraken_r_constitution":
        raise ConstitutionValidationError("constitution_kind must be 'kraken_r_constitution'")
    _nonempty_string(payload["version"], "version")
    _nonempty_string(payload["mission"], "mission")
    _nonempty_string(payload["preservation_policy"], "preservation_policy")

    boundary = payload["authority_boundary"]
    if not isinstance(boundary, dict):
        raise ConstitutionValidationError("authority_boundary must be an object")
    boundary_required = {
        "legacy_reference",
        "kraken_candidate",
        "allowed_effects",
        "forbidden_effects",
    }
    if set(boundary) != boundary_required:
        raise ConstitutionValidationError(
            "authority_boundary must contain exactly: "
            + ", ".join(sorted(boundary_required))
        )
    _nonempty_string(boundary["legacy_reference"], "authority_boundary.legacy_reference")
    _nonempty_string(boundary["kraken_candidate"], "authority_boundary.kraken_candidate")
    allowed_effects = _string_list(
        boundary["allowed_effects"], "authority_boundary.allowed_effects"
    )
    forbidden_effects = _string_list(
        boundary["forbidden_effects"], "authority_boundary.forbidden_effects"
    )
    if set(allowed_effects) != {
        "documentation",
        "standalone_validation",
        "candidate_cycle_execution",
    }:
        raise ConstitutionValidationError(
            "allowed_effects must be documentation, standalone_validation, "
            "and candidate_cycle_execution"
        )
    required_forbidden = {
        "runtime_wiring",
        "live_store_write",
        "event_subscription",
        "mutation_actuation",
    }
    if not required_forbidden.issubset(forbidden_effects):
        raise ConstitutionValidationError(
            "forbidden_effects must include all candidate-only runtime boundaries"
        )

    lifecycle = _string_list(payload["canonical_lifecycle"], "canonical_lifecycle")
    if set(lifecycle) != set(CANONICAL_CONTRACTS) or len(lifecycle) != len(CANONICAL_CONTRACTS):
        raise ConstitutionValidationError(
            "canonical_lifecycle must list every canonical contract exactly once"
        )
    planned = _string_list(payload["planned_capabilities"], "planned_capabilities")
    if set(planned) != set(PLANNED_CAPABILITIES) or len(planned) != len(PLANNED_CAPABILITIES):
        raise ConstitutionValidationError(
            "planned_capabilities must list every planned capability exactly once"
        )
    return payload


def validate_contract_catalog() -> tuple[str, ...]:
    errors: list[str] = []
    for name in CANONICAL_CONTRACTS:
        if not hasattr(contracts, name):
            errors.append(f"missing canonical contract: {name}")
    if contracts.State is not contracts.TaskState:
        errors.append("State must be an alias of TaskState, not a second authority")
    if contracts.Event is not contracts.Signal:
        errors.append("Event must be an alias of Signal, not a second event authority")
    if contracts.GroundTruth is not contracts.Evidence:
        errors.append("GroundTruth must be an alias of Evidence, not a second store")
    return tuple(errors)


def validate_cycle_execution() -> tuple[str, ...]:
    """Execute the complete fixture matrix and inspect causal invariants."""

    errors: list[str] = []
    expected = {
        CycleMode.SUCCESS: ("success", "success", 1, True),
        CycleMode.FAILURE: ("failure", "failure", 1, True),
        CycleMode.CONTRADICTION: ("contradiction", "contradiction", 2, False),
        CycleMode.INSUFFICIENT_EVIDENCE: (
            "insufficient_evidence",
            "not_observed",
            0,
            False,
        ),
    }
    for mode, (decision, observed, evidence_count, learns) in expected.items():
        objective = contracts.Objective(
            f"validator-objective-{mode.value}",
            "Validate the deterministic candidate cycle",
            success_criteria=("the cycle stops explicitly",),
        )
        try:
            trace = run_constitutional_cycle(objective, mode=mode)
        except (TypeError, ValueError) as exc:
            errors.append(f"{mode.value}: cycle execution failed: {exc}")
            continue
        if trace.decision.outcome != decision:
            errors.append(f"{mode.value}: decision does not match observation")
        if trace.settlement.prediction != "success":
            errors.append(f"{mode.value}: settlement lacks pre-execution prediction")
        if trace.settlement.observed_outcome != observed:
            errors.append(f"{mode.value}: settlement does not use observed outcome")
        if len(trace.evidence) != evidence_count:
            errors.append(f"{mode.value}: evidence count is incorrect")
        if any(not item.provenance for item in trace.evidence):
            errors.append(f"{mode.value}: observed evidence lacks provenance")
        if (trace.learning_update is not None) != learns:
            errors.append(f"{mode.value}: learning disposition is incorrect")
        if trace.stop_decision.outcome != "stop" or trace.final_state.phase != "stopped":
            errors.append(f"{mode.value}: cycle did not stop explicitly")
        if [state.version for state in trace.states] != list(
            range(1, len(trace.states) + 1)
        ):
            errors.append(f"{mode.value}: task-state versions are not contiguous")

    objective = contracts.Objective(
        "validator-authority-objective",
        "Verify that the cycle rejects non-candidate action authority",
    )
    for authority in (
        None,
        contracts.Authority.NONE,
        contracts.Authority.LEGACY_REFERENCE,
    ):
        try:
            run_constitutional_cycle(objective, action_authority=authority)
        except CycleInvariantError:
            continue
        errors.append(f"unauthorized action authority accepted: {authority}")
    return tuple(errors)


def validate_recorded_execution_replay() -> tuple[str, ...]:
    """Replay the bounded recorded-observation matrix and rejection boundaries."""

    errors: list[str] = []
    expected = {
        RecordedExecutionMode.SUCCESS: ("success", "success", 1, True),
        RecordedExecutionMode.FAILURE: ("failure", "failure", 1, True),
        RecordedExecutionMode.CONTRADICTION: ("contradiction", "contradiction", 2, False),
        RecordedExecutionMode.INSUFFICIENT_EVIDENCE: (
            "insufficient_evidence",
            "not_observed",
            0,
            False,
        ),
    }
    for mode, (decision, observed, evidence_count, learns) in expected.items():
        transaction_id = f"validator-replay-{mode.value}"
        objective = contracts.Objective(
            f"{transaction_id}-objective",
            "Validate immutable recorded execution replay",
            provenance={"transaction_id": transaction_id},
        )
        record = RecordedExecution.fixture(
            mode, transaction_id=transaction_id, objective_id=objective.objective_id
        )
        try:
            trace = replay_recorded_execution(objective, record)
        except (TypeError, ValueError) as exc:
            errors.append(f"{mode.value}: replay failed: {exc}")
            continue
        if trace.to_dict() != replay_recorded_execution(objective, record).to_dict():
            errors.append(f"{mode.value}: replay is not deterministic")
        if (
            trace.transaction_id != record.transaction_id
            or trace.objective.objective_id != record.objective_id
            or trace.states[4].state_id != record.task_state_id
            or trace.action.action_id != record.action_id
            or trace.execution.execution_id != record.execution_id
            or trace.settlement.settlement_id != record.settlement_id
        ):
            errors.append(f"{mode.value}: replay did not preserve identities")
        if tuple(item.evidence_id for item in trace.evidence) != record.evidence_ids:
            errors.append(f"{mode.value}: replay did not preserve evidence identities")
        if trace.decision.outcome != decision or trace.settlement.observed_outcome != observed:
            errors.append(f"{mode.value}: observed result was not settled honestly")
        if len(trace.evidence) != evidence_count or (
            trace.learning_update is not None
        ) != learns:
            errors.append(f"{mode.value}: evidence or learning disposition is incorrect")

    transaction_id = "validator-replay-rejections"
    objective = contracts.Objective(
        f"{transaction_id}-objective",
        "Verify replay rejection boundaries",
        provenance={"transaction_id": transaction_id},
    )
    record = RecordedExecution.fixture(
        RecordedExecutionMode.SUCCESS,
        transaction_id=transaction_id,
        objective_id=objective.objective_id,
    )
    invalid_records = {
        "missing provenance": replace(record, provenance={}),
        "missing settlement provenance": replace(
            record,
            provenance={
                key: value
                for key, value in record.provenance.items()
                if key != "settlement_id"
            },
        ),
        "stale state": replace(record, task_state_version=4),
        "mismatched observation": replace(
            record, observations={**record.observations, "action_id": "wrong-action"}
        ),
        "self-report-only success": replace(
            record,
            status="not_observed",
            observations={
                **record.observations,
                "observed": False,
                "self_reported_outcome": "success",
            },
            evidence_ids=(),
            learning_update_id=None,
            provenance={
                **record.provenance,
                "evidence_ids": (),
                "learning_update_id": None,
            },
        ),
        "incoherent status": replace(record, status="failed"),
    }
    for label, invalid in invalid_records.items():
        try:
            replay_recorded_execution(objective, invalid)
        except ReplayValidationError:
            continue
        errors.append(f"{label}: replay record was accepted")
    return tuple(errors)


def validate_nervous_system() -> tuple[str, ...]:
    """Exercise bounded candidate signal propagation and its evidence boundary."""

    errors: list[str] = []
    transaction_id = "validator-nervous-transaction"
    objective = contracts.Objective(
        "validator-nervous-objective",
        "Validate bounded candidate signal propagation",
        provenance={"transaction_id": transaction_id},
    )
    state_id = f"{objective.objective_id}-state-4"
    uncertainty = make_bound_signal(
        "validator-uncertainty",
        "candidate.uncertainty",
        transaction_id=transaction_id,
        objective_id=objective.objective_id,
        task_state_id=state_id,
        task_state_version=4,
    )
    record = SignalReplayRecord(
        transaction_id=transaction_id,
        objective_id=objective.objective_id,
        task_state_id=state_id,
        task_state_version=4,
        signals=(uncertainty,),
    )
    try:
        propagation = replay_signal_propagation(record)
        enabled = run_constitutional_cycle(objective, signals=(uncertainty,))
        replayed = replay_constitutional_signal_path(
            objective, signals=(uncertainty,)
        )
        ablated = run_constitutional_cycle(objective)
    except (TypeError, ValueError) as exc:
        return (f"candidate signal path failed: {exc}",)

    if not propagation.inhibits("candidate.action.authorize"):
        errors.append("uncertainty signal did not produce explicit inhibition")
    if enabled.to_dict() != replayed.to_dict():
        errors.append("candidate signal path is not deterministic under replay")
    if enabled.states[4].phase != "inhibited" or ablated.states[4].phase != "authorized":
        errors.append("enabled and ablated paths did not diverge at authorization")
    if enabled.execution.status != "not_observed" or enabled.evidence:
        errors.append("signal inhibition became observation evidence")
    if ablated.decision.outcome != "success":
        errors.append("ablated baseline did not retain the normal candidate outcome")

    if uncertainty.source != "candidate_signal_network" or uncertainty.cause != uncertainty.signal_id:
        errors.append("bound signal lacks explicit source/cause identity")
    if not isinstance(uncertainty.priority, int) or isinstance(uncertainty.priority, bool):
        errors.append("bound signal lacks an integer priority")

    priority_network = SignalNetwork(
        (SignalRule("priority-source", "candidate.priority", "candidate.sink"),)
    )
    low = make_bound_signal(
        "validator-priority-low",
        "candidate.priority",
        transaction_id=transaction_id,
        objective_id=objective.objective_id,
        task_state_id=state_id,
        task_state_version=4,
        priority=1,
    )
    high = make_bound_signal(
        "validator-priority-high",
        "candidate.priority",
        transaction_id=transaction_id,
        objective_id=objective.objective_id,
        task_state_id=state_id,
        task_state_version=4,
        priority=9,
    )
    priority_trace = replay_signal_propagation(
        SignalReplayRecord(
            transaction_id=transaction_id,
            objective_id=objective.objective_id,
            task_state_id=state_id,
            task_state_version=4,
            signals=(low, high),
        ),
        network=priority_network,
    )
    if priority_trace.delivered_signal_ids[:2] != (
        "validator-priority-high",
        "validator-priority-low",
    ):
        errors.append("priority ordering is not deterministic")
    tie_b = make_bound_signal(
        "validator-priority-b",
        "candidate.priority",
        transaction_id=transaction_id,
        objective_id=objective.objective_id,
        task_state_id=state_id,
        task_state_version=4,
        priority=5,
    )
    tie_a = make_bound_signal(
        "validator-priority-a",
        "candidate.priority",
        transaction_id=transaction_id,
        objective_id=objective.objective_id,
        task_state_id=state_id,
        task_state_version=4,
        priority=5,
    )
    tie_trace = replay_signal_propagation(
        SignalReplayRecord(
            transaction_id=transaction_id,
            objective_id=objective.objective_id,
            task_state_id=state_id,
            task_state_version=4,
            signals=(tie_b, tie_a),
        ),
        network=priority_network,
    )
    if tie_trace.delivered_signal_ids[:2] != (
        "validator-priority-a",
        "validator-priority-b",
    ):
        errors.append("priority tie ordering is not deterministic")

    unsupported = make_bound_signal(
        "validator-unsupported",
        "candidate.unsupported",
        transaction_id=transaction_id,
        objective_id=objective.objective_id,
        task_state_id=state_id,
        task_state_version=4,
    )
    try:
        replay_signal_propagation(
            SignalReplayRecord(
                transaction_id=transaction_id,
                objective_id=objective.objective_id,
                task_state_id=state_id,
                task_state_version=4,
                signals=(unsupported,),
            )
        )
    except SignalValidationError:
        pass
    else:
        errors.append("unsupported topic was accepted")

    for label, kwargs in (
        ("source", {"source": ""}),
        ("cause", {"cause": ""}),
        ("priority", {"priority": 101}),
    ):
        try:
            make_bound_signal(
                f"validator-malformed-{label}",
                "candidate.uncertainty",
                transaction_id=transaction_id,
                objective_id=objective.objective_id,
                task_state_id=state_id,
                task_state_version=4,
                **kwargs,
            )
        except (TypeError, ValueError):
            continue
        errors.append(f"malformed {label} was accepted")
    return tuple(errors)


def validate_physiology() -> tuple[str, ...]:
    """Exercise bounded advisory physiology without live runtime authority."""

    errors: list[str] = []
    transaction_id = "validator-physiology-transaction"
    objective = contracts.Objective(
        "validator-physiology-objective",
        "Validate bounded candidate physiology",
        provenance={"transaction_id": transaction_id},
    )
    state_id = f"{objective.objective_id}-state-4"
    healthy = PhysiologySnapshot(
        "validator-healthy",
        transaction_id,
        objective.objective_id,
        state_id,
        4,
    )
    critical = PhysiologySnapshot(
        "validator-critical",
        transaction_id,
        objective.objective_id,
        state_id,
        4,
        contradiction_density=0.85,
        backlog_pressure=0.9,
        resource_pressure=0.95,
        protected_reserve=0.05,
    )
    try:
        healthy_replay = replay_physiology(PhysiologyReplayRecord(healthy))
        critical_replay = replay_physiology(PhysiologyReplayRecord(critical))
        replay_repeat = replay_physiology(PhysiologyReplayRecord(critical))
        enabled = run_constitutional_cycle(objective, physiology=critical)
        replayed = replay_constitutional_signal_path(objective, physiology=critical)
        ablated = run_constitutional_cycle(objective)
    except (TypeError, ValueError) as exc:
        return (f"candidate physiology failed: {exc}",)

    if healthy_replay.decision.regime is not OperatingRegime.PRODUCTIVE:
        errors.append("healthy physiology did not enter productive regime")
    if critical_replay.decision.regime is not OperatingRegime.CRITICAL:
        errors.append("critical physiology did not enter critical regime")
    if critical_replay.decision.effect is not RegulationEffect.INHIBIT_ACTION:
        errors.append("critical physiology did not inhibit the candidate action")
    if critical_replay.to_dict() != replay_repeat.to_dict():
        errors.append("physiology replay is not deterministic")
    if enabled.to_dict() != replayed.to_dict():
        errors.append("cycle physiology replay is not deterministic")
    if enabled.states[4].phase != "inhibited" or ablated.states[4].phase != "authorized":
        errors.append("physiology ablation did not diverge at authorization")
    if enabled.execution.status != "not_observed" or enabled.evidence:
        errors.append("physiology inhibition became observation evidence")
    if (
        enabled.physiology_trace is None
        or enabled.physiology_trace.decision.advisory_only is not True
        or enabled.learning_update is not None
    ):
        errors.append("physiology crossed its advisory evidence boundary")

    for label, kwargs in (
        ("pressure", {"memory_pressure": 1.01}),
        ("cooldown", {"cooldown_remaining": 9}),
    ):
        try:
            PhysiologySnapshot(
                f"validator-malformed-{label}",
                transaction_id,
                objective.objective_id,
                state_id,
                4,
                **kwargs,
            )
        except PhysiologyValidationError:
            continue
        errors.append(f"malformed physiology {label} was accepted")

    stale = PhysiologySnapshot(
        "validator-stale",
        transaction_id,
        objective.objective_id,
        state_id,
        3,
    )
    try:
        run_constitutional_cycle(objective, physiology=stale)
    except CycleInvariantError:
        pass
    else:
        errors.append("stale physiology was accepted into execution")
    return tuple(errors)


def _route_record(
    topology: RouteTopology,
    trace: Any,
    *,
    record_id: str,
) -> SettlementRouteRecord:
    """Bind one completed constitutional trace to a candidate route selection."""

    selection = select_candidate_route(
        topology,
        "candidate-work",
        transaction_id=trace.transaction_id,
        objective_id=trace.objective.objective_id,
        task_state_id=trace.states[4].state_id,
        task_state_version=trace.states[4].version,
    )
    evidence_ids = tuple(item.evidence_id for item in trace.evidence)
    return SettlementRouteRecord(
        record_id,
        selection,
        trace,
        provenance={
            "transaction_id": trace.transaction_id,
            "objective_id": trace.objective.objective_id,
            "task_state_id": trace.states[4].state_id,
            "task_state_version": trace.states[4].version,
            "route_id": selection.route_id,
            "settlement_id": trace.settlement.settlement_id,
            "evidence_ids": evidence_ids,
        },
    )


def validate_plastic_routing() -> tuple[str, ...]:
    """Exercise settlement-gated, replayable candidate route plasticity."""

    errors: list[str] = []

    def trace(label: str, mode: CycleMode) -> Any:
        objective = contracts.Objective(
            f"validator-routing-{label}-objective",
            "Validate settlement-grounded candidate route preference",
            provenance={"transaction_id": f"validator-routing-{label}-tx"},
        )
        return run_constitutional_cycle(objective, mode=mode)

    try:
        topology = RouteTopology.fixture("validator-routing")
        first_record = _route_record(
            topology, trace("success-one", CycleMode.SUCCESS), record_id="validator-routing-one"
        )
        after_first, first = apply_settlement_learning(topology, first_record)
        second_record = _route_record(
            after_first,
            trace("success-two", CycleMode.SUCCESS),
            record_id="validator-routing-two",
        )
        after_second, second = apply_settlement_learning(after_first, second_record)
    except (TypeError, ValueError) as exc:
        return (f"candidate plastic routing failed: {exc}",)

    alpha = next(route for route in after_second.routes if route.route_id == "path-alpha")
    later = select_candidate_route(
        after_second,
        "candidate-work",
        transaction_id="validator-routing-later-tx",
        objective_id="validator-routing-later-objective",
        task_state_id="validator-routing-later-state",
        task_state_version=4,
    )
    if first.effect != "strengthen" or second.effect != "strengthen":
        errors.append("settled successes did not strengthen the selected route")
    if alpha.weight != 0.70 or later.candidate_scores[0][1] <= later.candidate_scores[1][1]:
        errors.append("successful history did not measurably change route organization")
    ablated = after_second.reset()
    if any(route.weight != 0.50 for route in ablated.routes):
        errors.append("route-state ablation did not remove learned preference")

    replayed_one = replay_settlement_learning(topology, (first_record, second_record))
    replayed_two = replay_settlement_learning(topology, (first_record, second_record))
    if replayed_one != replayed_two or replayed_one[0] != after_second:
        errors.append("route learning history is not deterministic under replay")

    preferred = RouteTopology(
        "validator-routing-failure",
        0,
        tuple(
            CandidateRoute(
                route.route_id,
                route.context_id,
                route.source,
                route.target,
                weight=0.60 if route.route_id == "path-alpha" else route.weight,
            )
            for route in RouteTopology.fixture("validator-routing-failure").routes
        ),
    )
    failure_one = _route_record(
        preferred,
        trace("failure-one", CycleMode.FAILURE),
        record_id="validator-routing-failure-one",
    )
    after_failure_one, failure_trace_one = apply_settlement_learning(preferred, failure_one)
    failure_two = _route_record(
        after_failure_one,
        trace("failure-two", CycleMode.FAILURE),
        record_id="validator-routing-failure-two",
    )
    after_failures, failure_trace_two = apply_settlement_learning(
        after_failure_one, failure_two
    )
    failure_choice = select_candidate_route(
        after_failures,
        "candidate-work",
        transaction_id="validator-routing-failure-later-tx",
        objective_id="validator-routing-failure-later-objective",
        task_state_id="validator-routing-failure-later-state",
        task_state_version=4,
    )
    if (
        failure_trace_one.effect != "weaken"
        or failure_trace_two.effect != "weaken"
        or failure_choice.route_id != "path-beta"
    ):
        errors.append("settled failures did not weaken and reorganize route preference")

    for label, mode in (
        ("contradiction", CycleMode.CONTRADICTION),
        ("insufficient", CycleMode.INSUFFICIENT_EVIDENCE),
    ):
        withheld_record = _route_record(
            topology, trace(label, mode), record_id=f"validator-routing-{label}"
        )
        unchanged, withheld = apply_settlement_learning(topology, withheld_record)
        if unchanged != topology or withheld.disposition != "withheld":
            errors.append(f"{label} outcome earned route reinforcement")

    try:
        apply_settlement_learning(after_first, first_record)
    except PlasticRoutingValidationError:
        pass
    else:
        errors.append("stale route selection was accepted")

    declared_trace = replace(
        first_record.constitutional_trace,
        evidence=tuple(
            replace(item, grade=contracts.EvidenceGrade.DECLARED)
            for item in first_record.evidence
        ),
    )
    try:
        apply_settlement_learning(
            topology,
            replace(first_record, constitutional_trace=declared_trace),
        )
    except PlasticRoutingValidationError:
        pass
    else:
        errors.append("declared evidence earned route reinforcement")
    if any(
        route.weight < 0.25 or route.weight > 0.75
        for route in after_failures.routes + after_second.routes
    ):
        errors.append("route weight escaped its fixed bounds")
    return tuple(errors)


def validate_controlled_llm_adapter() -> tuple[str, ...]:
    """Exercise proposal-only parsing, structural replay, and evaluation controls."""

    errors: list[str] = []
    objective = contracts.Objective(
        "validator-llm-objective",
        "Validate a bounded candidate model proposal.",
        provenance={"transaction_id": "validator-llm-transaction"},
    )
    trace = run_constitutional_cycle(
        objective, mode=CycleMode.INSUFFICIENT_EVIDENCE
    )
    try:
        context = CandidateModelContext.from_cycle_trace(
            trace,
            context_id="validator-llm-context",
            allowed_route_ids=("path-alpha", "path-beta"),
            route_scores=(("path-alpha", 0.50), ("path-beta", 0.50)),
        )
        provider = FixtureModelProvider(
            lambda _: (
                '{"proposal":"inspect candidate route","reasoning":"declared only",'
                '"route_hint":"path-alpha"}'
            ),
            provider_id="validator-fixture",
        )
        result = ModelAdapter(provider).invoke(
            context,
            request_id="validator-llm-request",
            prompt="Return only the bounded proposal JSON.",
            model_id="validator-fixture-model",
        )
    except (TypeError, ValueError) as exc:
        return (f"candidate LLM adapter failed: {exc}",)
    if not result.accepted or result.proposal is None or result.invocation is None:
        errors.append("valid model proposal was not accepted")
        return tuple(errors)
    if result.proposal.declared_only is not True or any(
        hasattr(result.proposal, field_name)
        for field_name in ("evidence", "settlement", "action", "goal", "learning_update")
    ):
        errors.append("model proposal crossed the candidate authority boundary")
    try:
        replay = replay_model_invocation(result.invocation)
    except (TypeError, ValueError) as exc:
        errors.append(f"model invocation structural replay failed: {exc}")
    else:
        if not replay.structure_replayed or replay.generation_replayed:
            errors.append("model replay did not mark generation nondeterminism honestly")

    self_report = ModelAdapter(
        FixtureModelProvider(
            lambda _: (
                '{"proposal":"claim success","reasoning":"declared",'
                '"route_hint":"path-alpha","success":true}'
            )
        )
    ).invoke(
        context,
        request_id="validator-llm-self-report",
        prompt="Return only the bounded proposal JSON.",
        model_id="validator-fixture-model",
    )
    if self_report.failure_code is not AdapterFailureCode.MALFORMED_OUTPUT:
        errors.append("self-report authority field was accepted")

    def evaluation_output(request: Any) -> str:
        route = "path-beta" if "candidate_route=path-beta" in request.prompt else "path-alpha"
        return (
            '{"proposal":"consider context","reasoning":"declared only",'
            f'"route_hint":"{route}"}}'
        )

    evaluator = ControlledEvaluationHarness(
        FixtureModelProvider(evaluation_output, provider_id="validator-eval-fixture"),
        model_id="validator-eval-model",
        max_tokens=128,
        temperature=0.0,
    )
    try:
        report = evaluator.run(
            (
                HeldOutTask(
                    "validator-heldout",
                    "validator-heldout-objective",
                    "Classify an unseen held-out task.",
                    "path-alpha",
                ),
            ),
            RouteTopology.fixture("validator-llm-topology"),
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"controlled evaluation failed: {exc}")
    else:
        if report.control_errors:
            errors.append("controlled evaluation did not preserve comparable conditions")
        if report.claim_status != "not_claimed":
            errors.append("controlled evaluation made an unsupported performance claim")
        modes = {item.mode for item in report.results}
        if modes != set(EvaluationMode):
            errors.append("controlled evaluation did not include all three conditions")
    return tuple(errors)


def validate_grounded_execution() -> tuple[str, ...]:
    """Verify the bounded external-observation boundary end to end."""

    errors: list[str] = []
    objective = contracts.Objective(
        "validator-grounded-objective",
        "Execute one sealed candidate test in a bounded workspace.",
        provenance={"transaction_id": "validator-grounded-transaction"},
    )
    action = make_grounded_action(objective.objective_id)
    authorized_state = contracts.TaskState(
        f"{objective.objective_id}-state-5",
        objective.objective_id,
        5,
        "authorized",
        values={"action_id": action.action_id},
    )
    request = GroundedExecutionRequest(
        "validator-grounded-request",
        "validator-grounded-transaction",
        objective.objective_id,
        authorized_state.state_id,
        authorized_state.version,
        action,
        {
            "subject.py": "def add(left, right):\n    return left + right\n",
            "test_subject.py": (
                "from subject import add\n\n"
                "def test_bounded_addition():\n"
                "    assert add(20, 22) == 42\n"
            ),
        },
        ("test_subject.py",),
    )
    executor = GroundedExecutionExecutor(
        delivery_ledger=GroundedDeliveryLedger(
            Path(tempfile.mkdtemp(prefix="kraken-r-validator-receipts-"))
            / "receipts.json"
        )
    )
    verifier = executor.verifier()
    try:
        record = executor.execute(request, authorized_state=authorized_state)
        verified = verifier.verify(
            record, request=request, authorized_state=authorized_state
        )
        replay = replay_grounded_execution(
            record,
            request=request,
            authorized_state=authorized_state,
            verifier=verifier,
        )
    except (GroundedExecutionRejected, OSError, RuntimeError) as exc:
        return (f"bounded grounded execution failed: {exc}",)
    if (
        verified.observed_outcome != "success"
        or verified.epistemic_class is not EpistemicOutcomeClass.TASK_SUCCESS
        or not record.provenance.child_runtime_verified
        or not record.provenance.child_runtime_fingerprint
        or replay != verified
    ):
        errors.append("verified execution did not structurally replay as success")
    try:
        fresh_verifier = GroundedExecutionVerifier.from_record(
            record, trusted_executor=executor.trusted_executor()
        )
        fresh_verifier.verify(
            record, request=request, authorized_state=authorized_state
        )
    except GroundedExecutionRejected as exc:
        errors.append(f"fresh public-key verifier rejected a valid record: {exc}")
    try:
        executor.execute(request, authorized_state=authorized_state)
    except GroundedExecutionRejected:
        pass
    else:
        errors.append("duplicate grounded execution was accepted")
    try:
        trace = run_constitutional_cycle(
            objective,
            grounded_execution=verified,
            grounded_request=request,
            grounded_verifier=verifier,
        )
    except CycleInvariantError as exc:
        errors.append(f"verified execution did not enter the candidate cycle: {exc}")
    else:
        if (
            trace.decision.outcome != "success"
            or trace.capability.evidence_grade is not contracts.EvidenceGrade.GROUNDED
            or not trace.learning_update
        ):
            errors.append("verified execution did not produce grounded settlement evidence")
    try:
        verifier.verify(
            replace(record, input_hash="0" * 64),
            request=request,
            authorized_state=authorized_state,
        )
    except GroundedExecutionRejected:
        pass
    else:
        errors.append("tampered grounded record was accepted")
    stale_state = contracts.TaskState(
        f"{objective.objective_id}-state-6",
        objective.objective_id,
        6,
        "authorized",
        values={"action_id": action.action_id},
    )
    try:
        verifier.verify(record, request=request, authorized_state=stale_state)
    except GroundedExecutionRejected:
        pass
    else:
        errors.append("stale task state was accepted for grounded verification")
    return tuple(errors)


def validate_interaction_boundaries() -> tuple[str, ...]:
    """Exercise static cross-module boundary checks without creating a runtime."""

    errors: list[str] = []
    objective = contracts.Objective(
        "validator-interaction-objective",
        "Validate conservative candidate interaction boundaries.",
        provenance={"transaction_id": "validator-interaction-transaction"},
    )
    uncertainty = make_bound_signal(
        "validator-interaction-uncertainty",
        "candidate.uncertainty",
        transaction_id="validator-interaction-transaction",
        objective_id=objective.objective_id,
        task_state_id=f"{objective.objective_id}-state-4",
        task_state_version=4,
    )
    critical = PhysiologySnapshot(
        "validator-interaction-critical",
        "validator-interaction-transaction",
        objective.objective_id,
        f"{objective.objective_id}-state-4",
        4,
        contradiction_density=0.90,
        resource_pressure=0.95,
        protected_reserve=0.05,
    )
    try:
        inhibited = run_constitutional_cycle(
            objective, signals=(uncertainty,), physiology=critical
        )
        report = validate_interaction_chain(inhibited)
    except (CycleInvariantError, InteractionValidationError) as exc:
        return (f"conservative interaction validation failed: {exc}",)
    if (
        not report.action_inhibited
        or report.grounded_execution
        or report.delivery_stages
        or inhibited.evidence
        or inhibited.learning_update is not None
    ):
        errors.append("combined signal and physiology inhibition was not conservative")

    proposal_trace = run_constitutional_cycle(
        contracts.Objective(
            "validator-interaction-proposal-objective",
            "Validate proposal provenance remains non-evidentiary.",
            provenance={"transaction_id": "validator-interaction-proposal-transaction"},
        ),
        mode=CycleMode.INSUFFICIENT_EVIDENCE,
    )
    try:
        context = CandidateModelContext.from_cycle_trace(
            proposal_trace,
            context_id="validator-interaction-proposal-context",
            allowed_route_ids=("path-alpha",),
            route_scores=(("path-alpha", 0.50),),
        )
        result = ModelAdapter(
            FixtureModelProvider(
                lambda _: (
                    '{"proposal":"inspect candidate route",'
                    '"reasoning":"declared-only provenance",'
                    '"route_hint":"path-alpha"}'
                )
            )
        ).invoke(
            context,
            request_id="validator-interaction-proposal-request",
            prompt="Return only the bounded proposal JSON.",
            model_id="validator-interaction-model",
        )
        proposal_report = validate_interaction_chain(
            proposal_trace, model_invocation=result.invocation
        )
    except (ModelAdapterValidationError, InteractionValidationError) as exc:
        errors.append(f"model interaction validation failed: {exc}")
    else:
        if proposal_report.proposal_id is None:
            errors.append("accepted declared proposal lost provenance")
    return tuple(errors)


def validate_adaptive_substrate() -> tuple[str, ...]:
    """Validate Stage 10's bounded advisory and disposable-state boundaries."""

    errors: list[str] = []
    state = AdaptiveState.fixture("validator-adaptive")
    snapshot = HomeostaticSnapshot(
        "validator-adaptive-pressure",
        "validator-adaptive-transaction",
        "validator-adaptive-objective",
        "validator-adaptive-state-4",
        4,
        contradiction=0.8,
        uncertainty=0.7,
        repeated_failure=0.6,
        novelty=0.4,
        resource_expenditure=0.9,
    )
    signals = snapshot.signals()
    if len(signals) != 5 or any(
        item.evidence_grade is not contracts.EvidenceGrade.DECLARED
        or item.payload.get("advisory_only") is not True
        for item in signals
    ):
        errors.append("homeostatic observations escaped their advisory signal boundary")
    before = state.to_dict()
    try:
        experiment = run_orzhaal_experiment(
            state, experiment_id="validator-orzhaal-experiment"
        )
    except AdaptiveSubstrateValidationError as exc:
        errors.append(f"disposable Orzhaal experiment failed: {exc}")
    else:
        if (
            experiment.promotable
            or experiment.canonical_mutation
            or state.to_dict() != before
        ):
            errors.append("Orzhaal experiment crossed the canonical isolation boundary")
    try:
        run_orzhaal_experiment(
            state, experiment_id="validator-orzhaal-too-large", weight_delta=0.10
        )
    except AdaptiveSubstrateValidationError:
        pass
    else:
        errors.append("Orzhaal accepted an unbounded experimental change")
    return tuple(errors)


def validate_task_integrity() -> tuple[str, ...]:
    """Exercise the immutable, proposal-only Stage 10.7 task boundary."""

    errors: list[str] = []
    try:
        original = OriginalTask(
            "validator-integrity-task",
            "Preserve requirements, avoid runtime wiring, and verify independently.",
            "validator_fixture",
            {"scope": "candidate_only"},
        )
        requirements = (
            TaskRequirement("integrity-ask", RequirementKind.ASK, "Preserve requirements."),
            TaskRequirement(
                "integrity-constraint",
                RequirementKind.CONSTRAINT,
                "Avoid runtime wiring.",
            ),
            TaskRequirement(
                "integrity-evidence",
                RequirementKind.REQUIRED_EVIDENCE,
                "Verify independently.",
            ),
        )
        specification = TaskSpecification.from_task(
            "validator-integrity-spec",
            original,
            requirements,
            "Preserve every structured clause as a candidate-only requirement.",
        )
        hypothesis = TaskHypothesis(
            "validator-integrity-hypothesis",
            specification.specification_id,
            "A pure boundary can preserve every requirement.",
            specification.requirement_ids,
        )
        belief = BeliefState(
            "validator-integrity-belief",
            specification.specification_id,
            (hypothesis,),
        )
        plan = TaskPlan(
            "validator-integrity-plan",
            specification.specification_id,
            (hypothesis.hypothesis_id,),
            specification.requirement_ids,
            ("Preserve clause lineage.", "Review only declared output."),
        )
        claim = CandidateClaim(
            "validator-integrity-claim",
            "All requirements are addressed.",
            specification.requirement_ids,
        )
        result = CandidateResult(
            "validator-integrity-result",
            specification.specification_id,
            plan.plan_id,
            (hypothesis.hypothesis_id,),
            specification.requirement_ids,
            claims=(claim,),
            conclusion_claim_ids=(claim.claim_id,),
        )
        before = result.to_dict()
        trace = replay_task_integrity(original, specification, belief, plan, result)
        replayed = replay_task_integrity(original, specification, belief, plan, result)
    except TaskIntegrityValidationError as exc:
        return (f"task-integrity fixture failed: {exc}",)
    if trace.to_dict() != replayed.to_dict():
        errors.append("task-integrity replay is not deterministic")
    if (
        trace.loss_report.findings
        or trace.review_findings
        or trace.verification_questions
        or result.to_dict() != before
    ):
        errors.append("correct candidate changed under non-actionable review")
    if (
        trace.to_dict()["creates_evidence"]
        or trace.to_dict()["changes_adaptive_state"]
        or any(item.to_dict()["evidence_grade"] != "none" for item in trace.review_findings)
    ):
        errors.append("task-integrity review crossed its proposal-only boundary")

    missing_dependency = replace(
        result,
        claims=(
            CandidateClaim(
                claim.claim_id,
                "A conclusion names an absent reverse dependency.",
                ("integrity-ask",),
                dependency_claim_ids=("validator-integrity-absent-claim",),
            ),
        ),
    )
    try:
        reverse_trace = replay_task_integrity(
            original,
            specification,
            belief,
            plan,
            missing_dependency,
            roles=(ReviewRole.ANTIMETABOLE,),
        )
    except TaskIntegrityValidationError as exc:
        errors.append(f"reverse-dependency review failed: {exc}")
    else:
        if not any(
            item.kind is LossKind.UNSUPPORTED_DEPENDENCY
            for item in reverse_trace.loss_report.findings
        ):
            errors.append("unsupported reverse dependency was not surfaced")
        if (
            len(reverse_trace.review_findings) != 1
            or reverse_trace.review_findings[0].role is not ReviewRole.ANTIMETABOLE
            or len(reverse_trace.verification_questions) != 1
        ):
            errors.append("reverse-dependency review did not remain selective and bounded")

    omitted = replace(
        result,
        addressed_requirement_ids=("integrity-ask",),
        compressed_requirement_ids=("integrity-constraint",),
    )
    try:
        angel_trace = replay_task_integrity(
            original, specification, belief, plan, omitted, roles=(ReviewRole.ANGEL,)
        )
    except TaskIntegrityValidationError as exc:
        errors.append(f"omission review failed: {exc}")
    else:
        kinds = {item.kind for item in angel_trace.loss_report.findings}
        if LossKind.OMITTED not in kinds or LossKind.COMPRESSED not in kinds:
            errors.append("omission or compression loss was not reported")
        if any(item.role is not ReviewRole.ANGEL for item in angel_trace.review_findings):
            errors.append("selective Angel review leaked another role")
        if any(
            "original_text" in item.payload or "evidence" in item.payload
            for item in angel_trace.projections
        ):
            errors.append("review projection leaked correlated runtime context")
    return tuple(errors)


def validate_dynamical_substrate() -> tuple[str, ...]:
    """Exercise the immutable, bounded Stage 10.8 tick boundary."""

    errors: list[str] = []
    state = DynamicalState.fixture("validator-dynamical")
    signal = make_bound_signal(
        "validator-dynamical-signal",
        "candidate.urgency",
        transaction_id=state.transaction_id,
        objective_id=state.objective_id,
        task_state_id=state.task_state_id,
        task_state_version=state.task_state_version,
        source="validator",
        cause="bounded-fixture",
    )
    ticks = (
        DynamicalTick(
            1,
            (
                DynamicalEvent.signal_event("validator-signal-event", signal),
                DynamicalEvent.observation_event(
                    "validator-observation-event", 0.50, 0.50
                ),
            ),
        ),
        DynamicalTick(2, (DynamicalEvent.resource_event("validator-resource", 0.20),)),
    )
    try:
        first, first_traces = replay_dynamical_ticks(state, ticks)
        second, second_traces = replay_dynamical_ticks(state, ticks)
    except DynamicalSubstrateValidationError as exc:
        return (f"dynamical substrate fixture failed: {exc}",)
    if first.to_dict() != second.to_dict() or tuple(
        item.to_dict() for item in first_traces
    ) != tuple(item.to_dict() for item in second_traces):
        errors.append("dynamical tick replay is not deterministic")
    if first.medium.adaptive_state != state.medium.adaptive_state:
        errors.append("unsettled signals or observations changed adaptive topology")
    try:
        surprised, surprise_trace = reduce_dynamical_tick(
            state,
            DynamicalTick(
                1,
                (
                    DynamicalEvent.observation_event(
                        "validator-surprise", 0.0, 1.0
                    ),
                ),
            ),
        )
    except DynamicalSubstrateValidationError as exc:
        errors.append(f"surprise fixture failed: {exc}")
    else:
        if (
            not surprise_trace.inhibited
            or surprised.fast.surprise != 1.0
            or surprised.instrumentation.inhibition_events != 1
        ):
            errors.append("surprise failed to alter bounded physiology dynamics")
    try:
        reduce_dynamical_tick(state, DynamicalTick(2))
    except DynamicalSubstrateValidationError:
        pass
    else:
        errors.append("out-of-sequence tick did not fail closed")
    stale_objective = contracts.Objective(
        state.objective_id,
        "Keep stale candidate settlement lineage non-creditable.",
        provenance={"transaction_id": state.transaction_id},
    )
    stale_trace = run_constitutional_cycle(stale_objective)
    topology = state.medium.adaptive_state.route_topology
    stale_state = replace(
        state,
        fast=replace(state.fast, active_route_id=topology.routes[0].route_id),
        medium=replace(
            state.medium,
            adaptive_state=replace(
                state.medium.adaptive_state,
                route_topology=replace(
                    topology,
                    routes=(
                        replace(topology.routes[0], weight=0.65),
                        *topology.routes[1:],
                    ),
                ),
            ),
        ),
    )
    stale_record = _route_record(
        stale_state.medium.adaptive_state.route_topology,
        stale_trace,
        record_id="validator-stale-settlement",
    )
    try:
        decayed, _ = reduce_dynamical_tick(stale_state, DynamicalTick(1))
        withheld, stale_tick = reduce_dynamical_tick(
            decayed,
            DynamicalTick(
                2,
                (
                    DynamicalEvent.settlement_event(
                        "validator-stale-settlement-event", stale_record
                    ),
                ),
            ),
        )
    except DynamicalSubstrateValidationError as exc:
        errors.append(f"stale settlement fixture failed: {exc}")
    else:
        if (
            stale_tick.noncreditable_settlement_ids != (stale_record.record_id,)
            or withheld.medium != decayed.medium
            or withheld.fast != decayed.fast
            or withheld.slow != decayed.slow
            or withheld.instrumentation != decayed.instrumentation
        ):
            errors.append("stale settlement altered current dynamical state")
    return tuple(errors)


def validate_metastability_experiments() -> tuple[str, ...]:
    """Exercise Stage 10.9's bounded, non-authoritative comparison layer."""

    errors: list[str] = []
    state = DynamicalState.fixture("validator-metastability")
    ticks = make_metastability_scenario(
        state, ExperimentScenario.NOISY_NONCREDITABLE, ticks=12
    )
    try:
        first = run_metastability_experiment(
            state,
            ticks,
            experiment_id="validator-metastability-first",
            scenario=ExperimentScenario.NOISY_NONCREDITABLE,
        )
        second = run_metastability_experiment(
            state,
            ticks,
            experiment_id="validator-metastability-second",
            scenario=ExperimentScenario.NOISY_NONCREDITABLE,
        )
        comparison = compare_metastability(
            state,
            ticks,
            comparison_id="validator-metastability",
            scenario=ExperimentScenario.NOISY_NONCREDITABLE,
        )
    except (MetastabilityValidationError, DynamicalSubstrateValidationError) as exc:
        return (f"metastability fixture failed: {exc}",)
    if first.to_dict()["final_state"] != second.to_dict()["final_state"]:
        errors.append("metastability replay is not deterministic")
    if first.metrics.noncreditable_events != 0:
        errors.append("non-creditable scenario invented settlement outcomes")
    if first.final_state.medium.adaptive_state != state.medium.adaptive_state:
        errors.append("non-creditable observations changed adaptive topology")
    if len(comparison.reports) != 6 or not all(
        report.input_digest == comparison.baseline.input_digest
        and report.metrics.tick_count == comparison.baseline.metrics.tick_count
        for report in comparison.reports
    ):
        errors.append("metastability ablations did not retain matched inputs and budgets")
    without_surprise = next(
        report for report in comparison.reports if report.ablation.label == "without_surprise"
    )
    if without_surprise.metrics.surprise_peak != 0.0:
        errors.append("surprise ablation retained mismatch pressure")
    without_inhibition = next(
        report for report in comparison.reports if report.ablation.label == "without_inhibition"
    )
    if without_inhibition.metrics.inhibition_rate != 0.0:
        errors.append("inhibition ablation retained action inhibition")
    try:
        run_metastability_experiment(
            state,
            (item for item in ticks for _ in range(30)),
            scenario=ExperimentScenario.STABLE,
        )
    except MetastabilityValidationError:
        pass
    else:
        errors.append("oversized experiment generator did not fail closed")
    return tuple(errors)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the isolated, candidate-only Kraken-R foundation."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional registry JSON path; defaults to the bundled registry.",
    )
    parser.add_argument(
        "--constitution",
        type=Path,
        help="Optional constitution metadata JSON path; defaults to the bundled metadata.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print the validation report as JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = load_registry(args.registry) if args.registry else load_default_registry()
    constitution = load_constitution_metadata(args.constitution)
    registry_report = registry.validate()
    metadata_errors = (
        ()
        if constitution["version"] == registry.version
        else (
            "constitution and architecture registry versions must match "
            f"({constitution['version']} != {registry.version})",
        )
    )
    contract_errors = validate_contract_catalog()
    cycle_errors = validate_cycle_execution()
    replay_errors = validate_recorded_execution_replay()
    nervous_system_errors = validate_nervous_system()
    physiology_errors = validate_physiology()
    plastic_routing_errors = validate_plastic_routing()
    llm_adapter_errors = validate_controlled_llm_adapter()
    grounded_execution_errors = validate_grounded_execution()
    interaction_errors = validate_interaction_boundaries()
    adaptive_substrate_errors = validate_adaptive_substrate()
    task_integrity_errors = validate_task_integrity()
    dynamical_substrate_errors = validate_dynamical_substrate()
    metastability_errors = validate_metastability_experiments()
    legacy_runtime_errors = validate_legacy_runtime_boundary()
    report = {
        "ok": (
            registry_report.ok
            and not metadata_errors
            and not contract_errors
            and not cycle_errors
            and not replay_errors
            and not nervous_system_errors
            and not physiology_errors
            and not plastic_routing_errors
            and not llm_adapter_errors
            and not grounded_execution_errors
            and not interaction_errors
            and not adaptive_substrate_errors
            and not task_integrity_errors
            and not dynamical_substrate_errors
            and not metastability_errors
            and not legacy_runtime_errors
        ),
        "contracts_ok": not contract_errors,
        "contract_errors": list(contract_errors),
        "metadata_versions": {
            "ok": not metadata_errors,
            "constitution": constitution["version"],
            "registry": registry.version,
            "errors": list(metadata_errors),
        },
        "cycle": {
            "ok": not cycle_errors,
            "errors": list(cycle_errors),
            "mode": CycleMode.SUCCESS.value,
            "modes": [mode.value for mode in CycleMode],
        },
        "replay": {
            "ok": not replay_errors,
            "errors": list(replay_errors),
            "modes": [mode.value for mode in RecordedExecutionMode],
        },
        "nervous_system": {
            "ok": not nervous_system_errors,
            "errors": list(nervous_system_errors),
            "authority": "candidate_only",
        },
        "physiology": {
            "ok": not physiology_errors,
            "errors": list(physiology_errors),
            "authority": "advisory_candidate_only",
        },
        "plastic_routing": {
            "ok": not plastic_routing_errors,
            "errors": list(plastic_routing_errors),
            "authority": "settlement_grounded_candidate_only",
        },
        "controlled_llm_adapter": {
            "ok": not llm_adapter_errors,
            "errors": list(llm_adapter_errors),
            "authority": "proposal_only_candidate",
        },
        "grounded_execution": {
            "ok": not grounded_execution_errors,
            "errors": list(grounded_execution_errors),
            "authority": "verified_candidate_execution_only",
        },
        "interactions": {
            "ok": not interaction_errors,
            "errors": list(interaction_errors),
            "authority": "static_candidate_validation_only",
        },
        "adaptive_substrate": {
            "ok": not adaptive_substrate_errors,
            "errors": list(adaptive_substrate_errors),
            "authority": "grounded_candidate_reducers_and_disposable_experiments_only",
        },
        "task_integrity": {
            "ok": not task_integrity_errors,
            "errors": list(task_integrity_errors),
            "authority": "proposal_only_candidate_task_integrity",
        },
        "dynamical_substrate": {
            "ok": not dynamical_substrate_errors,
            "errors": list(dynamical_substrate_errors),
            "authority": "bounded_immutable_replayable_candidate_ticks_only",
        },
        "metastability": {
            "ok": not metastability_errors,
            "errors": list(metastability_errors),
            "authority": "bounded_candidate_experiment_observation_only",
        },
        "constitution": {
            "ok": True,
            "version": constitution["version"],
            "allowed_effects": constitution["authority_boundary"]["allowed_effects"],
        },
        "registry": registry_report.to_dict(),
        "legacy_runtime_wiring": (
            "absent" if not legacy_runtime_errors else "present"
        ),
        "legacy_runtime_errors": list(legacy_runtime_errors),
    }
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        state = "PASS" if report["ok"] else "FAIL"
        print(
            f"Kraken-R foundation validation: {state} "
            f"({registry_report.mechanism_count} mechanisms, "
            f"{registry_report.category_count} categories, "
            f"{registry_report.planned_count} planned)"
        )
        for error in (
            *metadata_errors,
            *contract_errors,
            *cycle_errors,
            *task_integrity_errors,
            *registry_report.errors,
        ):
            print(f"- {error}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())