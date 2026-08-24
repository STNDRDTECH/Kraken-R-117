"""Deterministic, candidate-only execution of the Kraken-R lifecycle.

The cycle is deliberately an in-memory fixture runner.  It demonstrates the
causal ordering of the round-one contracts without becoming a second daemon,
store, event bus, executor, or mutation authority.
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
from .nervous_system import PropagationTrace, SignalNetwork, SignalPropagationError


class CycleInvariantError(ValueError):
    """Raised when a constitutional cycle boundary is violated."""


class CycleMode(str, Enum):
    """The deterministic observed-execution fixtures supported by the cycle."""

    SUCCESS = "success"
    FAILURE = "failure"
    CONTRADICTION = "contradiction"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def _freeze_trace_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_trace_value(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_trace_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_trace_value(item) for item in value)
    return value


@dataclass(frozen=True)
class CycleTrace:
    """All immutable records produced by one bounded candidate cycle."""

    transaction_id: str
    objective: Objective
    states: tuple[TaskState, ...]
    plan: Plan
    hypothesis: Hypothesis
    signal: Signal
    event: Signal
    action: Action
    execution: ExecutionResult
    evidence: tuple[Evidence, ...]
    ground_truth: Evidence | None
    decision: Decision
    settlement: Settlement
    capability: Capability
    learning_update: LearningUpdate | None
    stop_decision: Decision
    provenance: Mapping[str, Any] = field(default_factory=dict)
    signal_trace: PropagationTrace | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.transaction_id, str) or not self.transaction_id.strip():
            raise CycleInvariantError("trace requires a transaction_id")
        object.__setattr__(self, "provenance", _freeze_trace_value(self.provenance))

    @property
    def final_state(self) -> TaskState:
        return self.states[-1]

    @property
    def mode(self) -> CycleMode:
        return CycleMode(self.execution.observations["mode"])

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trace without exposing mutable internal containers."""

        return {
            "transaction_id": self.transaction_id,
            "objective": self.objective.to_dict(),
            "states": [state.to_dict() for state in self.states],
            "plan": self.plan.to_dict(),
            "hypothesis": self.hypothesis.to_dict(),
            "signal": self.signal.to_dict(),
            "event": self.event.to_dict(),
            "action": self.action.to_dict(),
            "execution": self.execution.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "ground_truth": (
                self.ground_truth.to_dict() if self.ground_truth is not None else None
            ),
            "decision": self.decision.to_dict(),
            "settlement": self.settlement.to_dict(),
            "capability": self.capability.to_dict(),
            "learning_update": (
                self.learning_update.to_dict()
                if self.learning_update is not None
                else None
            ),
            "stop_decision": self.stop_decision.to_dict(),
            "provenance": dict(self.provenance),
            "signal_trace": (
                self.signal_trace.to_dict() if self.signal_trace is not None else None
            ),
        }


class ConstitutionalCycle:
    """Run one deterministic objective without any external authority."""

    def run(
        self,
        objective: Objective,
        *,
        mode: CycleMode | str = CycleMode.SUCCESS,
        action_authority: Authority | str | None = Authority.KRAKEN_CANDIDATE,
        signals: tuple[Signal, ...] = (),
        signal_network: SignalNetwork | None = None,
    ) -> CycleTrace:
        if not isinstance(objective, Objective):
            raise CycleInvariantError("cycle requires an Objective contract")
        try:
            selected_mode = CycleMode(mode)
        except (TypeError, ValueError) as exc:
            allowed = ", ".join(item.value for item in CycleMode)
            raise CycleInvariantError(
                f"mode must be one of: {allowed}"
            ) from exc

        if action_authority is None:
            raise CycleInvariantError("action authority is required")
        try:
            selected_authority = Authority(action_authority)
        except (TypeError, ValueError) as exc:
            raise CycleInvariantError(
                "action authority must be an explicit Authority"
            ) from exc
        if selected_authority is not Authority.KRAKEN_CANDIDATE:
            raise CycleInvariantError(
                "candidate cycle action requires kraken_candidate authority"
            )

        prefix = objective.objective_id
        transaction_id = str(objective.provenance.get("transaction_id", prefix))
        state = TaskState(
            f"{prefix}-state-1",
            prefix,
            1,
            "objective",
            values={"objective_acquired": True},
        )
        states = [state]

        plan = Plan(
            f"{prefix}-plan",
            prefix,
            action_ids=(f"{prefix}-action",),
            preconditions=("objective_acquired",),
            rollback_plan="stop-without-side-effects",
        )
        predicted_outcome = "success"
        state = self._advance(
            state,
            "planned",
            {
                "plan_id": plan.plan_id,
                "preconditions_checked": True,
                "predicted_outcome": predicted_outcome,
            },
        )
        states.append(state)

        hypothesis = Hypothesis(
            f"{prefix}-hypothesis",
            prefix,
            "The bounded candidate action will produce the requested observation.",
            confidence=0.5,
        )
        state = self._advance(
            state,
            "hypothesized",
            {"hypothesis_id": hypothesis.hypothesis_id},
        )
        states.append(state)

        signal = Signal(
            f"{prefix}-signal",
            "candidate.observation.requested",
            prefix,
            "kraken_r_candidate",
            payload={"claim": "execute one bounded action"},
            evidence_grade=EvidenceGrade.DECLARED,
        )
        # Event is a vocabulary alias, not a second event authority.
        event = signal
        state = self._advance(
            state,
            "signaled",
            {"signal_id": signal.signal_id, "declared_only": True},
        )
        states.append(state)

        signal_trace: PropagationTrace | None = None
        if signals or signal_network is not None:
            try:
                signal_trace = (signal_network or SignalNetwork()).propagate(
                    signals,
                    transaction_id=transaction_id,
                    objective_id=objective.objective_id,
                    task_state_id=state.state_id,
                    task_state_version=state.version,
                )
            except SignalPropagationError as exc:
                raise CycleInvariantError(
                    f"candidate signal propagation was rejected: {exc}"
                ) from exc
        action_inhibited = (
            signal_trace.inhibits("candidate.action.authorize")
            if signal_trace is not None
            else False
        )

        action = Action(
            f"{prefix}-action",
            prefix,
            "observe_bounded_target",
            "deterministic_target",
            parameters={"mode": selected_mode.value},
            preconditions=plan.preconditions,
            authority=selected_authority,
        )
        state = self._advance(
            state,
            "inhibited" if action_inhibited else "authorized",
            {
                "action_id": action.action_id,
                "authority": selected_authority.value,
                "signal_inhibited": action_inhibited,
                "inhibiting_topics": (
                    tuple(signal_trace.inhibited_topics) if signal_trace else ()
                ),
            },
        )
        states.append(state)

        execution = self._execute_fixture(
            action, selected_mode, signal_trace=signal_trace
        )
        state = self._advance(
            state,
            "observed",
            {
                "execution_id": execution.execution_id,
                "execution_status": execution.status,
            },
        )
        states.append(state)

        evidence = self._evidence_for(execution)
        evidence_ids = tuple(item.evidence_id for item in evidence)
        observed_outcome = self._observed_outcome(execution)
        ground_truth = evidence[0] if observed_outcome in {"success", "failure"} else None
        state = self._advance(
            state,
            "evidenced" if evidence else "evidence_insufficient",
            {"evidence_count": len(evidence)},
            evidence_ids=evidence_ids,
        )
        states.append(state)

        outcome = self._decision_outcome(observed_outcome, evidence)
        decision = Decision(
            f"{prefix}-decision",
            prefix,
            outcome,
            self._decision_rationale(outcome),
            selected_action_id=action.action_id if evidence else None,
            evidence_ids=evidence_ids,
        )
        state = self._advance(
            state,
            "decided",
            {"decision_id": decision.decision_id, "outcome": outcome},
            evidence_ids=evidence_ids,
        )
        states.append(state)

        settlement = Settlement(
            f"{prefix}-settlement",
            decision.decision_id,
            prediction=predicted_outcome,
            observed_outcome=observed_outcome,
            evidence_ids=evidence_ids,
            status="settled" if evidence else "insufficient_evidence",
        )
        state = self._advance(
            state,
            "settled",
            {
                "settlement_id": settlement.settlement_id,
                "observed_outcome": observed_outcome,
            },
            evidence_ids=evidence_ids,
        )
        states.append(state)

        capability = Capability(
            f"{prefix}-capability",
            "bounded deterministic observation",
            evidence_grade=(
                EvidenceGrade.OPERATIONAL
                if evidence
                else EvidenceGrade.NONE
            ),
            evidence_ids=evidence_ids,
        )
        learning_update = self._learning_update(
            prefix, capability, settlement, observed_outcome, evidence_ids
        )
        state = self._advance(
            state,
            "learned" if learning_update is not None else "learning_withheld",
            {
                "learning_update_id": (
                    learning_update.update_id
                    if learning_update is not None
                    else None
                )
            },
            evidence_ids=evidence_ids,
        )
        states.append(state)

        stop_decision = Decision(
            f"{prefix}-stop",
            prefix,
            "stop",
            f"Stop after {outcome}; this bounded candidate cycle has no next action.",
            evidence_ids=evidence_ids,
        )
        state = self._advance(
            state,
            "stopped",
            {"stop_decision_id": stop_decision.decision_id},
            evidence_ids=evidence_ids,
        )
        states.append(state)

        trace = CycleTrace(
            transaction_id=transaction_id,
            objective=objective,
            states=tuple(states),
            plan=plan,
            hypothesis=hypothesis,
            signal=signal,
            event=event,
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
                "transaction_id": transaction_id,
                "source": "deterministic_fixture",
                "observation_origin": "execution_result",
            },
            signal_trace=signal_trace,
        )
        self._validate_trace(trace)
        return trace

    @staticmethod
    def _advance(
        previous: TaskState,
        phase: str,
        values: dict[str, Any],
        *,
        evidence_ids: tuple[str, ...] = (),
    ) -> TaskState:
        next_state = TaskState(
            f"{previous.objective_id}-state-{previous.version + 1}",
            previous.objective_id,
            previous.version + 1,
            phase,
            values=values,
            evidence_ids=evidence_ids,
            authority=previous.authority,
        )
        if next_state.version != previous.version + 1:
            raise CycleInvariantError("task-state version did not advance by one")
        return next_state

    @staticmethod
    def _execute_fixture(
        action: Action,
        mode: CycleMode,
        *,
        signal_trace: PropagationTrace | None = None,
    ) -> ExecutionResult:
        observations: dict[str, Any] = {"mode": mode.value}
        if signal_trace is not None and signal_trace.inhibits(
            "candidate.action.authorize"
        ):
            observations.update(
                {
                    "observed": False,
                    "signal_inhibited": True,
                    "inhibiting_topics": tuple(signal_trace.inhibited_topics),
                    "inhibiting_signal_ids": tuple(
                        step.signal_id
                        for step in signal_trace.steps
                        if step.disposition == "inhibited"
                    ),
                }
            )
            status, exit_code = "not_observed", None
        elif mode is CycleMode.SUCCESS:
            observations.update({"observed": True, "criteria_met": True})
            status, exit_code = "completed", 0
        elif mode is CycleMode.FAILURE:
            observations.update({"observed": True, "criteria_met": False})
            status, exit_code = "failed", 1
        elif mode is CycleMode.CONTRADICTION:
            observations.update(
                {
                    "observed": True,
                    "positive_observation": True,
                    "negative_observation": True,
                }
            )
            status, exit_code = "completed", 0
        else:
            observations["observed"] = False
            status, exit_code = "not_observed", None
        return ExecutionResult(
            f"{action.action_id}-execution",
            action.action_id,
            status,
            exit_code=exit_code,
            observations=observations,
        )

    @staticmethod
    def _evidence_for(execution: ExecutionResult) -> tuple[Evidence, ...]:
        if not execution.observations.get("observed"):
            # The declared Signal and the ExecutionResult are not evidence.
            return ()
        common = {
            "source_kind": "observed_execution",
            "execution_status": execution.status,
        }
        provenance = {
            "cycle": "kraken_r_deterministic_candidate",
            "adapter": "deterministic_observation",
            "observation_origin": "execution_result",
        }
        if (
            execution.observations.get("positive_observation")
            and execution.observations.get("negative_observation")
        ):
            return (
                Evidence(
                    f"{execution.execution_id}-positive",
                    execution.execution_id,
                    EvidenceGrade.OPERATIONAL,
                    "deterministic_observation",
                    observations={**common, "criteria_met": True},
                    execution_id=execution.execution_id,
                    provenance=provenance,
                ),
                Evidence(
                    f"{execution.execution_id}-negative",
                    execution.execution_id,
                    EvidenceGrade.OPERATIONAL,
                    "deterministic_observation",
                    observations={**common, "criteria_met": False},
                    execution_id=execution.execution_id,
                    provenance=provenance,
                ),
            )
        criteria_met = execution.observations.get("criteria_met")
        if not isinstance(criteria_met, bool):
            raise CycleInvariantError(
                "observed execution must report a boolean criteria_met value"
            )
        return (
            Evidence(
                f"{execution.execution_id}-observed",
                execution.execution_id,
                EvidenceGrade.OPERATIONAL,
                "deterministic_observation",
                observations={**common, "criteria_met": criteria_met},
                execution_id=execution.execution_id,
                provenance=provenance,
            ),
        )

    @staticmethod
    def _decision_outcome(
        observed_outcome: str, evidence: tuple[Evidence, ...]
    ) -> str:
        if not evidence:
            return "insufficient_evidence"
        if observed_outcome == "contradiction":
            return "contradiction"
        return observed_outcome

    @staticmethod
    def _decision_rationale(outcome: str) -> str:
        return {
            "success": "Observed execution evidence supports the objective.",
            "failure": "Observed execution evidence shows the action failed.",
            "contradiction": "Observed evidence contains incompatible results.",
            "insufficient_evidence": (
                "The declared request and execution status do not establish evidence."
            ),
        }[outcome]

    @staticmethod
    def _observed_outcome(execution: ExecutionResult) -> str:
        if not execution.observations.get("observed"):
            return "not_observed"
        if (
            execution.observations.get("positive_observation")
            and execution.observations.get("negative_observation")
        ):
            return "contradiction"
        if execution.observations.get("criteria_met") is True:
            return "success"
        if execution.observations.get("criteria_met") is False:
            return "failure"
        raise CycleInvariantError(
            "observed execution has no interpretable outcome"
        )

    @staticmethod
    def _learning_update(
        prefix: str,
        capability: Capability,
        settlement: Settlement,
        observed_outcome: str,
        evidence_ids: tuple[str, ...],
    ) -> LearningUpdate | None:
        if observed_outcome in {"contradiction", "not_observed"}:
            return None
        return LearningUpdate(
            f"{prefix}-learning",
            settlement.settlement_id,
            "capability",
            capability.capability_id,
            change={"observed_outcome": settlement.observed_outcome},
            evidence_ids=evidence_ids,
            disposition="accepted",
        )

    @staticmethod
    def _validate_trace(trace: CycleTrace) -> None:
        if trace.signal is not trace.event:
            raise CycleInvariantError("Event must remain the Signal vocabulary alias")
        if trace.action.authority is not Authority.KRAKEN_CANDIDATE:
            raise CycleInvariantError("action authority is not candidate-authorized")
        if [state.version for state in trace.states] != list(
            range(1, len(trace.states) + 1)
        ):
            raise CycleInvariantError("task-state versions are not contiguous")
        if any(state.objective_id != trace.objective.objective_id for state in trace.states):
            raise CycleInvariantError("task-state objective changed during cycle")
        if any(
            evidence.execution_id != trace.execution.execution_id
            or evidence.grade in {EvidenceGrade.NONE, EvidenceGrade.DECLARED}
            or not evidence.provenance
            or evidence.provenance.get("observation_origin")
            not in {"execution_result", "recorded_execution"}
            for evidence in trace.evidence
        ):
            raise CycleInvariantError("evidence is not grounded in observed execution")
        if trace.signal_trace is not None:
            if (
                trace.signal_trace.transaction_id != trace.transaction_id
                or trace.signal_trace.objective_id != trace.objective.objective_id
                or trace.signal_trace.task_state_id != trace.states[3].state_id
                or trace.signal_trace.task_state_version != trace.states[3].version
            ):
                raise CycleInvariantError(
                    "signal propagation is not bound to the signaled task state"
                )
            if any(
                evidence.evidence_id in trace.signal_trace.delivered_signal_ids
                for evidence in trace.evidence
            ):
                raise CycleInvariantError("signals cannot become evidence")
            if trace.signal_trace.inhibits("candidate.action.authorize") and (
                trace.execution.observations.get("observed")
            ):
                raise CycleInvariantError(
                    "an inhibited action cannot claim observed execution"
                )
        observed_outcome = ConstitutionalCycle._observed_outcome(trace.execution)
        expected_decision = ConstitutionalCycle._decision_outcome(
            observed_outcome, trace.evidence
        )
        if observed_outcome == "not_observed" and trace.evidence:
            raise CycleInvariantError("insufficient-evidence mode produced evidence")
        if (
            observed_outcome == "contradiction"
            and trace.learning_update is not None
        ):
            raise CycleInvariantError("contradictory evidence cannot teach a route")
        if trace.decision.outcome != expected_decision:
            raise CycleInvariantError("decision did not follow observed evidence")
        if trace.settlement.prediction != "success":
            raise CycleInvariantError("settlement must retain the pre-execution prediction")
        if trace.settlement.observed_outcome != observed_outcome:
            raise CycleInvariantError("settlement did not use observed execution")
        if trace.learning_update is not None:
            if trace.settlement.status != "settled":
                raise CycleInvariantError("learning requires a settled result")
            if not trace.learning_update.evidence_ids:
                raise CycleInvariantError("learning requires settlement evidence")
            if tuple(trace.learning_update.evidence_ids) != tuple(
                trace.settlement.evidence_ids
            ):
                raise CycleInvariantError(
                    "learning evidence must match settlement evidence"
                )
        if trace.final_state.phase != "stopped" or trace.stop_decision.outcome != "stop":
            raise CycleInvariantError("cycle must end with an explicit stop decision")


def run_constitutional_cycle(
    objective: Objective,
    *,
    mode: CycleMode | str = CycleMode.SUCCESS,
    action_authority: Authority | str | None = Authority.KRAKEN_CANDIDATE,
    signals: tuple[Signal, ...] = (),
    signal_network: SignalNetwork | None = None,
) -> CycleTrace:
    """Run the bounded deterministic candidate cycle."""

    return ConstitutionalCycle().run(
        objective,
        mode=mode,
        action_authority=action_authority,
        signals=signals,
        signal_network=signal_network,
    )


def replay_constitutional_signal_path(
    objective: Objective,
    *,
    mode: CycleMode | str = CycleMode.SUCCESS,
    signals: tuple[Signal, ...] = (),
    signal_network: SignalNetwork | None = None,
) -> CycleTrace:
    """Replay an immutable signal input through the bounded candidate cycle."""

    return run_constitutional_cycle(
        objective,
        mode=mode,
        signals=signals,
        signal_network=signal_network,
    )


# Short names make the standalone fixture convenient without creating another
# implementation or authority.
run_cycle = run_constitutional_cycle
DeterministicCycle = ConstitutionalCycle


__all__ = [
    "ConstitutionalCycle",
    "CycleInvariantError",
    "CycleMode",
    "CycleTrace",
    "DeterministicCycle",
    "replay_constitutional_signal_path",
    "run_constitutional_cycle",
    "run_cycle",
]