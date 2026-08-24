"""Read-only replay of immutable recorded execution observations.

The adapter is deliberately narrow: it converts one already-recorded
observation into the existing constitutional-cycle trace without subscribing to
events, calling routers, writing state, or importing the legacy runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .contracts import (
    Action,
    Authority,
    Capability,
    Decision,
    Evidence,
    EvidenceGrade,
    ExecutionResult,
    Hypothesis,
    LearningUpdate,
    Objective,
    Plan,
    Settlement,
    Signal,
    TaskState,
)
from .cycle import ConstitutionalCycle, CycleInvariantError, CycleMode, CycleTrace


class ReplayValidationError(ValueError):
    """Raised when a recorded observation cannot enter the candidate replay."""


class RecordedExecutionMode(str, Enum):
    """The bounded observed-outcome fixture shapes accepted by replay."""

    SUCCESS = "success"
    FAILURE = "failure"
    CONTRADICTION = "contradiction"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(char.isspace() for char in value):
        raise ReplayValidationError(f"{field_name} must be a non-empty identifier")
    return value


@dataclass(frozen=True)
class RecordedExecution:
    """Immutable observation and identity envelope for one replayable execution."""

    transaction_id: str
    objective_id: str
    task_state_id: str
    task_state_version: int
    action_id: str
    execution_id: str
    status: str
    observations: Mapping[str, Any]
    provenance: Mapping[str, Any]
    evidence_ids: tuple[str, ...] = ()
    decision_id: str = ""
    settlement_id: str = ""
    stop_decision_id: str = ""
    learning_update_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "transaction_id",
            "objective_id",
            "task_state_id",
            "action_id",
            "execution_id",
            "decision_id",
            "settlement_id",
            "stop_decision_id",
        ):
            _identifier(getattr(self, field_name), field_name)
        if self.learning_update_id is not None:
            _identifier(self.learning_update_id, "learning_update_id")
        if not isinstance(self.task_state_version, int) or self.task_state_version < 1:
            raise ReplayValidationError("task_state_version must be a positive integer")
        if not isinstance(self.status, str) or not self.status.strip():
            raise ReplayValidationError("status must be a non-empty string")
        if not isinstance(self.observations, Mapping):
            raise ReplayValidationError("observations must be a mapping")
        if not isinstance(self.provenance, Mapping):
            raise ReplayValidationError("provenance must be a mapping")
        if not isinstance(self.evidence_ids, tuple) or not all(
            isinstance(item, str) and item for item in self.evidence_ids
        ):
            raise ReplayValidationError("evidence_ids must be a tuple of identifiers")
        object.__setattr__(self, "observations", _freeze(self.observations))
        object.__setattr__(self, "provenance", _freeze(self.provenance))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))

    @classmethod
    def fixture(
        cls,
        mode: RecordedExecutionMode | CycleMode | str,
        *,
        transaction_id: str,
        objective_id: str,
    ) -> "RecordedExecution":
        """Build one immutable in-memory fixture without reading any live state."""

        selected = RecordedExecutionMode(str(getattr(mode, "value", mode)))
        state_id = f"{objective_id}-state-5"
        action_id = f"{transaction_id}-action"
        execution_id = f"{transaction_id}-execution"
        decision_id = f"{transaction_id}-decision"
        settlement_id = f"{transaction_id}-settlement"
        stop_decision_id = f"{transaction_id}-stop"
        observations: dict[str, Any] = {
            "mode": selected.value,
            "transaction_id": transaction_id,
            "objective_id": objective_id,
            "task_state_id": state_id,
            "task_state_version": 5,
            "action_id": action_id,
            "execution_id": execution_id,
        }
        if selected is RecordedExecutionMode.SUCCESS:
            observations.update({"observed": True, "criteria_met": True})
            status, evidence_ids, learning_update_id = (
                "completed",
                (f"{execution_id}-observed",),
                f"{transaction_id}-learning",
            )
        elif selected is RecordedExecutionMode.FAILURE:
            observations.update({"observed": True, "criteria_met": False})
            status, evidence_ids, learning_update_id = (
                "failed",
                (f"{execution_id}-observed",),
                f"{transaction_id}-learning",
            )
        elif selected is RecordedExecutionMode.CONTRADICTION:
            observations.update(
                {
                    "observed": True,
                    "positive_observation": True,
                    "negative_observation": True,
                }
            )
            status, evidence_ids, learning_update_id = (
                "completed",
                (f"{execution_id}-positive", f"{execution_id}-negative"),
                None,
            )
        else:
            observations["observed"] = False
            status, evidence_ids, learning_update_id = "not_observed", (), None
        provenance = {
            "record_id": f"{transaction_id}-record",
            "observation_origin": "recorded_execution",
            "transaction_id": transaction_id,
            "objective_id": objective_id,
            "task_state_id": state_id,
            "task_state_version": 5,
            "action_id": action_id,
            "execution_id": execution_id,
            "evidence_ids": evidence_ids,
            "decision_id": decision_id,
            "settlement_id": settlement_id,
            "stop_decision_id": stop_decision_id,
            "learning_update_id": learning_update_id,
        }
        return cls(
            transaction_id,
            objective_id,
            state_id,
            5,
            action_id,
            execution_id,
            status,
            observations,
            provenance,
            evidence_ids,
            decision_id,
            settlement_id,
            stop_decision_id,
            learning_update_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "objective_id": self.objective_id,
            "task_state_id": self.task_state_id,
            "task_state_version": self.task_state_version,
            "action_id": self.action_id,
            "execution_id": self.execution_id,
            "status": self.status,
            "observations": dict(self.observations),
            "provenance": dict(self.provenance),
            "evidence_ids": list(self.evidence_ids),
            "decision_id": self.decision_id,
            "settlement_id": self.settlement_id,
            "stop_decision_id": self.stop_decision_id,
            "learning_update_id": self.learning_update_id,
        }


class RecordedExecutionReplay:
    """Replay a single verified record through the existing constitutional trace."""

    AUTHORIZED_STATE_VERSION = 5

    def replay(self, objective: Objective, record: RecordedExecution) -> CycleTrace:
        self._validate_record(objective, record)
        prefix = objective.objective_id
        states: list[TaskState] = [
            TaskState(
                f"{prefix}-state-1",
                objective.objective_id,
                1,
                "objective",
                values={"objective_acquired": True, "transaction_id": record.transaction_id},
            )
        ]
        plan = Plan(
            f"{record.transaction_id}-plan",
            objective.objective_id,
            action_ids=(record.action_id,),
            preconditions=("objective_acquired",),
            rollback_plan="stop-without-side-effects",
        )
        states.append(
            ConstitutionalCycle._advance(
                states[-1],
                "planned",
                {
                    "plan_id": plan.plan_id,
                    "preconditions_checked": True,
                    "predicted_outcome": "success",
                    "transaction_id": record.transaction_id,
                },
            )
        )
        hypothesis = Hypothesis(
            f"{record.transaction_id}-hypothesis",
            objective.objective_id,
            "Recorded observation will be evaluated only against its immutable evidence.",
            confidence=0.5,
        )
        states.append(
            ConstitutionalCycle._advance(
                states[-1],
                "hypothesized",
                {"hypothesis_id": hypothesis.hypothesis_id},
            )
        )
        signal = Signal(
            f"{record.transaction_id}-signal",
            "candidate.recorded_execution.replay",
            objective.objective_id,
            "kraken_r_candidate",
            payload={
                "record_id": record.provenance["record_id"],
                "transaction_id": record.transaction_id,
            },
            evidence_grade=EvidenceGrade.DECLARED,
            source="kraken_r_recorded_replay",
            cause=record.provenance["record_id"],
            priority=0,
        )
        states.append(
            ConstitutionalCycle._advance(
                states[-1], "signaled", {"signal_id": signal.signal_id, "declared_only": True}
            )
        )
        action = Action(
            record.action_id,
            objective.objective_id,
            "replay_recorded_execution",
            "immutable_recorded_observation",
            parameters={
                "record_id": record.provenance["record_id"],
                "transaction_id": record.transaction_id,
                "expected_outcome": "success",
            },
            preconditions=plan.preconditions,
            authority=Authority.KRAKEN_CANDIDATE,
        )
        states.append(
            ConstitutionalCycle._advance(
                states[-1],
                "authorized",
                {"action_id": action.action_id, "authority": action.authority.value},
            )
        )
        if states[-1].state_id != record.task_state_id:
            raise ReplayValidationError("record task_state_id does not match replay authorization")

        execution = ExecutionResult(
            record.execution_id,
            record.action_id,
            record.status,
            observations=record.observations,
        )
        states.append(
            ConstitutionalCycle._advance(
                states[-1],
                "observed",
                {
                    "execution_id": execution.execution_id,
                    "execution_status": execution.status,
                    "record_id": record.provenance["record_id"],
                },
            )
        )
        evidence = self._evidence_for_record(execution, record)
        evidence_ids = tuple(item.evidence_id for item in evidence)
        observed_outcome = ConstitutionalCycle._observed_outcome(execution)
        ground_truth = evidence[0] if observed_outcome in {"success", "failure"} else None
        states.append(
            ConstitutionalCycle._advance(
                states[-1],
                "evidenced" if evidence else "evidence_insufficient",
                {"evidence_count": len(evidence), "record_id": record.provenance["record_id"]},
                evidence_ids=evidence_ids,
            )
        )
        outcome = ConstitutionalCycle._decision_outcome(observed_outcome, evidence)
        decision = Decision(
            record.decision_id,
            objective.objective_id,
            outcome,
            ConstitutionalCycle._decision_rationale(outcome),
            selected_action_id=action.action_id if evidence else None,
            evidence_ids=evidence_ids,
        )
        states.append(
            ConstitutionalCycle._advance(
                states[-1],
                "decided",
                {"decision_id": decision.decision_id, "outcome": decision.outcome},
                evidence_ids=evidence_ids,
            )
        )
        settlement = Settlement(
            record.settlement_id,
            decision.decision_id,
            prediction="success",
            observed_outcome=observed_outcome,
            evidence_ids=evidence_ids,
            status="settled" if evidence else "insufficient_evidence",
        )
        states.append(
            ConstitutionalCycle._advance(
                states[-1],
                "settled",
                {
                    "settlement_id": settlement.settlement_id,
                    "observed_outcome": settlement.observed_outcome,
                },
                evidence_ids=evidence_ids,
            )
        )
        capability = Capability(
            f"{record.transaction_id}-capability",
            "recorded execution replay",
            evidence_grade=EvidenceGrade.OPERATIONAL if evidence else EvidenceGrade.NONE,
            evidence_ids=evidence_ids,
        )
        learning_update = self._learning_for_record(
            record, capability, settlement, observed_outcome, evidence_ids
        )
        states.append(
            ConstitutionalCycle._advance(
                states[-1],
                "learned" if learning_update is not None else "learning_withheld",
                {
                    "learning_update_id": (
                        learning_update.update_id if learning_update is not None else None
                    )
                },
                evidence_ids=evidence_ids,
            )
        )
        stop_decision = Decision(
            record.stop_decision_id,
            objective.objective_id,
            "stop",
            "Recorded execution replay reached its bounded constitutional stop.",
            evidence_ids=evidence_ids,
        )
        states.append(
            ConstitutionalCycle._advance(
                states[-1],
                "stopped",
                {"stop_decision_id": stop_decision.decision_id},
                evidence_ids=evidence_ids,
            )
        )
        trace = CycleTrace(
            transaction_id=record.transaction_id,
            objective=objective,
            states=tuple(states),
            plan=plan,
            hypothesis=hypothesis,
            signal=signal,
            event=signal,
            action=action,
            execution=execution,
            evidence=evidence,
            ground_truth=ground_truth,
            decision=decision,
            settlement=settlement,
            capability=capability,
            learning_update=learning_update,
            stop_decision=stop_decision,
            provenance={
                **dict(record.provenance),
                "replay_mode": "read_only_in_memory",
                "observation_origin": "recorded_execution",
            },
        )
        try:
            ConstitutionalCycle._validate_trace(trace)
        except CycleInvariantError as exc:
            raise ReplayValidationError(str(exc)) from exc
        self._validate_trace_identity(trace, record)
        return trace

    @classmethod
    def _validate_record(cls, objective: Objective, record: RecordedExecution) -> None:
        if not isinstance(objective, Objective):
            raise ReplayValidationError("replay requires an Objective contract")
        if not isinstance(record, RecordedExecution):
            raise ReplayValidationError("replay requires a RecordedExecution record")
        if objective.objective_id != record.objective_id:
            raise ReplayValidationError("record objective_id does not match objective")
        if objective.provenance.get("transaction_id") != record.transaction_id:
            raise ReplayValidationError("objective provenance must name the replay transaction")
        if record.task_state_version != cls.AUTHORIZED_STATE_VERSION:
            raise ReplayValidationError("record references a stale task-state version")
        observed_identity = {
            "record_id": f"{record.transaction_id}-record",
            "observation_origin": "recorded_execution",
            "transaction_id": record.transaction_id,
            "objective_id": record.objective_id,
            "task_state_id": record.task_state_id,
            "task_state_version": record.task_state_version,
            "action_id": record.action_id,
            "execution_id": record.execution_id,
        }
        provenance_identity = {
            **observed_identity,
            "evidence_ids": record.evidence_ids,
            "decision_id": record.decision_id,
            "settlement_id": record.settlement_id,
            "stop_decision_id": record.stop_decision_id,
            "learning_update_id": record.learning_update_id,
        }
        for key, expected in provenance_identity.items():
            if record.provenance.get(key) != expected:
                raise ReplayValidationError(f"missing or mismatched provenance: {key}")
        for key, expected in observed_identity.items():
            if key not in {"record_id", "observation_origin"} and record.observations.get(key) != expected:
                raise ReplayValidationError(f"mismatched recorded observation: {key}")
        observed = record.observations.get("observed")
        if not isinstance(observed, bool):
            raise ReplayValidationError("recorded observation must declare observed as a boolean")
        if not observed and record.observations.get("self_reported_outcome"):
            raise ReplayValidationError("self-report-only success is not replayable evidence")
        if record.status not in {"completed", "failed", "not_observed"}:
            raise ReplayValidationError("recorded status is unsupported")
        if observed and record.status == "not_observed":
            raise ReplayValidationError("observed record cannot have not_observed status")
        if not observed and record.status != "not_observed":
            raise ReplayValidationError("unobserved record must have not_observed status")
        contradictory = bool(record.observations.get("positive_observation")) and bool(
            record.observations.get("negative_observation")
        )
        if observed and not contradictory and not isinstance(
            record.observations.get("criteria_met"), bool
        ):
            raise ReplayValidationError(
                "observed record must contain criteria_met unless explicitly contradictory"
            )
        if contradictory and record.status != "completed":
            raise ReplayValidationError(
                "contradictory observation must retain completed execution status"
            )
        if observed and not contradictory:
            expected_status = (
                "completed" if record.observations["criteria_met"] else "failed"
            )
            if record.status != expected_status:
                raise ReplayValidationError(
                    "recorded status conflicts with the observed outcome"
                )
        expected_evidence_count = 0 if not observed else 2 if contradictory else 1
        if len(record.evidence_ids) != expected_evidence_count:
            raise ReplayValidationError("record evidence identities do not match observations")
        if (record.learning_update_id is not None) != (
            observed and not contradictory
        ):
            raise ReplayValidationError("learning identity must be absent for contradiction or no evidence")

    @staticmethod
    def _evidence_for_record(
        execution: ExecutionResult, record: RecordedExecution
    ) -> tuple[Evidence, ...]:
        if not execution.observations["observed"]:
            return ()
        common = {
            "source_kind": "recorded_execution",
            "execution_status": execution.status,
            "record_id": record.provenance["record_id"],
        }
        provenance = {
            **dict(record.provenance),
            "recorded_source": "recorded_execution",
        }
        contradictory = bool(execution.observations.get("positive_observation")) and bool(
            execution.observations.get("negative_observation")
        )
        if contradictory:
            return (
                Evidence(
                    record.evidence_ids[0],
                    execution.execution_id,
                    EvidenceGrade.OPERATIONAL,
                    "recorded_execution",
                    observations={**common, "criteria_met": True},
                    execution_id=execution.execution_id,
                    provenance=provenance,
                ),
                Evidence(
                    record.evidence_ids[1],
                    execution.execution_id,
                    EvidenceGrade.OPERATIONAL,
                    "recorded_execution",
                    observations={**common, "criteria_met": False},
                    execution_id=execution.execution_id,
                    provenance=provenance,
                ),
            )
        return (
            Evidence(
                record.evidence_ids[0],
                execution.execution_id,
                EvidenceGrade.OPERATIONAL,
                "recorded_execution",
                observations={
                    **common,
                    "criteria_met": execution.observations["criteria_met"],
                },
                execution_id=execution.execution_id,
                provenance=provenance,
            ),
        )

    @staticmethod
    def _learning_for_record(
        record: RecordedExecution,
        capability: Capability,
        settlement: Settlement,
        observed_outcome: str,
        evidence_ids: tuple[str, ...],
    ) -> LearningUpdate | None:
        if record.learning_update_id is None:
            return None
        if observed_outcome not in {"success", "failure"}:
            raise ReplayValidationError("unsettled observation cannot create a learning update")
        return LearningUpdate(
            record.learning_update_id,
            settlement.settlement_id,
            "capability",
            capability.capability_id,
            change={
                "observed_outcome": settlement.observed_outcome,
                "record_id": record.provenance["record_id"],
            },
            evidence_ids=evidence_ids,
            disposition="accepted",
        )

    @staticmethod
    def _validate_trace_identity(trace: CycleTrace, record: RecordedExecution) -> None:
        if trace.transaction_id != record.transaction_id:
            raise ReplayValidationError("replay changed transaction identity")
        if trace.objective.objective_id != record.objective_id:
            raise ReplayValidationError("replay changed objective identity")
        action_state = trace.states[RecordedExecutionReplay.AUTHORIZED_STATE_VERSION - 1]
        if (
            action_state.state_id != record.task_state_id
            or action_state.version != record.task_state_version
        ):
            raise ReplayValidationError("replay changed task-state identity")
        if trace.action.action_id != record.action_id:
            raise ReplayValidationError("replay changed action identity")
        if trace.execution.execution_id != record.execution_id:
            raise ReplayValidationError("replay changed execution identity")
        if tuple(item.evidence_id for item in trace.evidence) != record.evidence_ids:
            raise ReplayValidationError("replay changed evidence identity")
        if trace.settlement.settlement_id != record.settlement_id:
            raise ReplayValidationError("replay changed settlement identity")
        for key, expected in record.provenance.items():
            if trace.provenance.get(key) != expected:
                raise ReplayValidationError(f"replay lost provenance: {key}")
            for evidence in trace.evidence:
                if evidence.provenance.get(key) != expected:
                    raise ReplayValidationError(f"evidence lost provenance: {key}")


def replay_recorded_execution(objective: Objective, record: RecordedExecution) -> CycleTrace:
    """Replay one immutable execution observation through the candidate cycle."""

    return RecordedExecutionReplay().replay(objective, record)


__all__ = [
    "RecordedExecution",
    "RecordedExecutionMode",
    "RecordedExecutionReplay",
    "ReplayValidationError",
    "replay_recorded_execution",
]