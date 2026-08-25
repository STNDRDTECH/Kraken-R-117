"""Static validation for bounded Kraken-R module interactions.

This module is deliberately a pure validator.  It does not retain receipts,
invoke providers, execute code, route messages, or mutate a topology.  It
checks that already-created immutable records can coexist without crossing the
candidate authority or evidence boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .cycle import ConstitutionalCycle, CycleInvariantError, CycleTrace
from .grounded_execution import (
    GroundedExecutionRequest,
    GroundedExecutionVerifier,
    GroundedExecutionRejected,
    VerifiedGroundedExecution,
)
from .llm_adapter import (
    ModelAdapterValidationError,
    ModelInvocation,
    replay_model_invocation,
)
from .plastic_routing import (
    RouteLearningTrace,
    RouteTopology,
    SettlementRouteRecord,
    apply_settlement_learning,
)


class InteractionValidationError(ValueError):
    """Raised when otherwise valid records form an invalid combined trajectory."""


class DeliveryStage(str, Enum):
    """The only grounded delivery order that can be represented in a trace."""

    EXECUTION = "execution"
    EVIDENCE = "evidence"
    SETTLEMENT = "settlement"
    LEARNING = "learning"


@dataclass(frozen=True)
class InteractionReport:
    """Read-only report of one validated composition boundary."""

    transaction_id: str
    objective_id: str
    action_inhibited: bool
    grounded_execution: bool
    delivery_stages: tuple[DeliveryStage, ...]
    route_learning: RouteLearningTrace | None = None
    proposal_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "transaction_id": self.transaction_id,
            "objective_id": self.objective_id,
            "action_inhibited": self.action_inhibited,
            "grounded_execution": self.grounded_execution,
            "delivery_stages": [stage.value for stage in self.delivery_stages],
            "route_learning": (
                self.route_learning.to_dict() if self.route_learning is not None else None
            ),
            "proposal_id": self.proposal_id,
        }


def _delivery_stages(
    trace: CycleTrace, *, grounded_execution: bool
) -> tuple[DeliveryStage, ...]:
    if not grounded_execution:
        return ()
    stages: list[DeliveryStage] = [DeliveryStage.EXECUTION]
    if trace.evidence:
        stages.extend((DeliveryStage.EVIDENCE, DeliveryStage.SETTLEMENT))
        if trace.learning_update is not None:
            stages.append(DeliveryStage.LEARNING)
    return tuple(stages)


def _coerce_stages(
    stages: Iterable[DeliveryStage | str],
) -> tuple[DeliveryStage, ...]:
    try:
        return tuple(DeliveryStage(stage) for stage in stages)
    except (TypeError, ValueError) as exc:
        raise InteractionValidationError("delivery stages are invalid") from exc


def _validate_model_provenance(
    trace: CycleTrace, invocation: ModelInvocation
) -> str | None:
    try:
        replay = replay_model_invocation(invocation)
    except ModelAdapterValidationError as exc:
        raise InteractionValidationError(
            f"model invocation is not structurally replayable: {exc}"
        ) from exc
    proposal = replay.proposal
    if proposal is None:
        return None
    if proposal.declared_only is not True:
        raise InteractionValidationError("model proposal is not declared-only")
    if proposal.objective_id != trace.objective.objective_id:
        raise InteractionValidationError(
            "model proposal objective does not match the constitutional trace"
        )
    context = invocation.request.context
    authorized_state = trace.states[4]
    if (
        context.transaction_id != trace.transaction_id
        or context.objective_id != trace.objective.objective_id
        or context.task_state_id != authorized_state.state_id
        or context.task_state_version != authorized_state.version
        or context.task_phase != authorized_state.phase
    ):
        raise InteractionValidationError(
            "model invocation context does not match the authorized trace state"
        )
    for evidence in trace.evidence:
        if any(
            key in evidence.provenance
            for key in ("proposal_id", "model_invocation_id", "model_output")
        ):
            raise InteractionValidationError(
                "model provenance cannot be promoted into execution evidence"
            )
    return proposal.proposal_id


def validate_interaction_chain(
    trace: CycleTrace,
    *,
    grounded_execution: VerifiedGroundedExecution | None = None,
    grounded_request: GroundedExecutionRequest | None = None,
    grounded_verifier: GroundedExecutionVerifier | None = None,
    delivery_stages: Iterable[DeliveryStage | str] | None = None,
    topology: RouteTopology | None = None,
    route_record: SettlementRouteRecord | None = None,
    model_invocation: ModelInvocation | None = None,
) -> InteractionReport:
    """Validate a bounded, caller-supplied interaction trajectory.

    A grounded trace is independently rechecked, signal and physiology
    inhibitions remain conservative, delivery stages must be causal, route
    credit remains a pure reducer call, and model output stays declared-only.
    The function has no side effects and intentionally does not replace the
    receipt ledger, execution verifier, or route reducer.
    """

    if not isinstance(trace, CycleTrace):
        raise InteractionValidationError("interaction validation requires a CycleTrace")
    try:
        ConstitutionalCycle._validate_trace(trace)
    except CycleInvariantError as exc:
        raise InteractionValidationError(
            f"constitutional trace is invalid: {exc}"
        ) from exc

    signal_inhibited = (
        trace.signal_trace is not None
        and trace.signal_trace.inhibits("candidate.action.authorize")
    )
    physiology_inhibited = (
        trace.physiology_trace is not None
        and trace.physiology_trace.decision.action_inhibited
    )
    action_inhibited = signal_inhibited or physiology_inhibited
    if action_inhibited:
        if trace.states[4].phase != "inhibited":
            raise InteractionValidationError(
                "an inhibition did not stop candidate authorization"
            )
        if trace.evidence or trace.learning_update is not None:
            raise InteractionValidationError(
                "an inhibited candidate action cannot create evidence or learning"
            )

    trace_is_grounded = (
        trace.provenance.get("source") == "grounded_execution_verifier"
    )
    supplied_grounded = (
        grounded_execution,
        grounded_request,
        grounded_verifier,
    )
    if trace_is_grounded != any(item is not None for item in supplied_grounded):
        raise InteractionValidationError(
            "grounded trace and grounded verification inputs must agree"
        )
    if trace_is_grounded:
        if not all(item is not None for item in supplied_grounded):
            raise InteractionValidationError(
                "grounded trace requires execution, request, and verifier"
            )
        assert grounded_execution is not None
        assert grounded_request is not None
        assert grounded_verifier is not None
        try:
            verified = grounded_verifier.verify(
                grounded_execution.record,
                request=grounded_request,
                authorized_state=trace.states[4],
            )
        except GroundedExecutionRejected as exc:
            raise InteractionValidationError(
                f"grounded execution cannot be independently reverified: {exc}"
            ) from exc
        if verified != grounded_execution:
            raise InteractionValidationError(
                "grounded verification result changed during interaction validation"
            )
        if action_inhibited:
            raise InteractionValidationError(
                "inhibited candidate action cannot consume grounded execution"
            )
        if (
            trace.execution.observations.get("record_hash")
            != grounded_execution.record.record_hash
        ):
            raise InteractionValidationError(
                "grounded execution lineage does not match the cycle observation"
            )
        if trace.evidence and any(
            item.provenance.get("record_hash")
            != grounded_execution.record.record_hash
            for item in trace.evidence
        ):
            raise InteractionValidationError(
                "grounded evidence lineage does not match the verified record"
            )

    expected_stages = _delivery_stages(
        trace, grounded_execution=trace_is_grounded
    )
    actual_stages = (
        expected_stages
        if delivery_stages is None
        else _coerce_stages(delivery_stages)
    )
    if actual_stages != expected_stages:
        raise InteractionValidationError(
            "grounded delivery stages are out of order, duplicated, or incomplete"
        )

    route_learning: RouteLearningTrace | None = None
    if (topology is None) != (route_record is None):
        raise InteractionValidationError(
            "route validation requires both a topology and a settlement record"
        )
    if topology is not None and route_record is not None:
        if route_record.constitutional_trace != trace:
            raise InteractionValidationError(
                "route record does not bind the validated constitutional trace"
            )
        if trace_is_grounded and (
            route_record.grounded_execution != grounded_execution
            or route_record.grounded_request != grounded_request
            or route_record.grounded_verifier != grounded_verifier
        ):
            raise InteractionValidationError(
                "route record does not bind the reverified grounded execution"
            )
        try:
            _, route_learning = apply_settlement_learning(topology, route_record)
        except ValueError as exc:
            raise InteractionValidationError(
                f"settlement route record is invalid: {exc}"
            ) from exc
        if trace_is_grounded and trace.evidence and (
            route_learning.disposition != "accepted"
        ):
            raise InteractionValidationError(
                "grounded settled evidence did not reach bounded route learning"
            )

    proposal_id = None
    if model_invocation is not None:
        proposal_id = _validate_model_provenance(trace, model_invocation)

    return InteractionReport(
        transaction_id=trace.transaction_id,
        objective_id=trace.objective.objective_id,
        action_inhibited=action_inhibited,
        grounded_execution=trace_is_grounded,
        delivery_stages=actual_stages,
        route_learning=route_learning,
        proposal_id=proposal_id,
    )


__all__ = [
    "DeliveryStage",
    "InteractionReport",
    "InteractionValidationError",
    "validate_interaction_chain",
]