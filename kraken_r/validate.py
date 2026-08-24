"""Standalone inspection command for the candidate Kraken-R foundation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
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
from .replay import (
    RecordedExecution,
    RecordedExecutionMode,
    ReplayValidationError,
    replay_recorded_execution,
)
from .registry import load_default_registry, load_registry


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
    contract_errors = validate_contract_catalog()
    cycle_errors = validate_cycle_execution()
    replay_errors = validate_recorded_execution_replay()
    nervous_system_errors = validate_nervous_system()
    report = {
        "ok": (
            registry_report.ok
            and not contract_errors
            and not cycle_errors
            and not replay_errors
            and not nervous_system_errors
        ),
        "contracts_ok": not contract_errors,
        "contract_errors": list(contract_errors),
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
        "constitution": {
            "ok": True,
            "version": constitution["version"],
            "allowed_effects": constitution["authority_boundary"]["allowed_effects"],
        },
        "registry": registry_report.to_dict(),
        "legacy_runtime_wiring": "absent",
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
        for error in (*contract_errors, *cycle_errors, *registry_report.errors):
            print(f"- {error}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())