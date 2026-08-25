"""Stage 9 acceptance tests for bounded Kraken-R interaction validation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import pytest

from kraken_r import (
    CandidateModelContext,
    CycleInvariantError,
    DeliveryStage,
    FixtureModelProvider,
    GroundedDeliveryLedger,
    GroundedExecutionExecutor,
    GroundedExecutionRejected,
    GroundedExecutionRequest,
    InteractionValidationError,
    ModelAdapter,
    Objective,
    PhysiologySnapshot,
    RouteTopology,
    SettlementRouteRecord,
    TaskState,
    make_bound_signal,
    make_grounded_action,
    replay_grounded_execution,
    run_constitutional_cycle,
    select_candidate_route,
    validate_interaction_chain,
)
from kraken_r.plastic_routing import (
    PlasticRoutingValidationError,
    apply_settlement_learning,
)


def _grounded_trace(label: str):
    objective = Objective(
        f"{label}-objective",
        "Validate one bounded Stage 9 interaction.",
        provenance={"transaction_id": f"{label}-transaction"},
    )
    action = make_grounded_action(objective.objective_id)
    authorized_state = TaskState(
        f"{objective.objective_id}-state-5",
        objective.objective_id,
        5,
        "authorized",
        values={"action_id": action.action_id},
    )
    request = GroundedExecutionRequest(
        f"{label}-request",
        f"{label}-transaction",
        objective.objective_id,
        authorized_state.state_id,
        authorized_state.version,
        action,
        {
            "subject.py": "def answer():\n    return 42\n",
            "test_subject.py": (
                "from subject import answer\n\n"
                "def test_answer():\n"
                "    assert answer() == 42\n"
            ),
        },
        ("test_subject.py",),
    )
    ledger = GroundedDeliveryLedger(
        Path(tempfile.mkdtemp(prefix="kraken-r-stage9-receipts-")) / "receipts.json"
    )
    executor = GroundedExecutionExecutor(delivery_ledger=ledger)
    verifier = executor.verifier()
    record = executor.execute(request, authorized_state=authorized_state)
    verified = verifier.verify(
        record, request=request, authorized_state=authorized_state
    )
    trace = run_constitutional_cycle(
        objective,
        grounded_execution=verified,
        grounded_request=request,
        grounded_verifier=verifier,
    )
    return objective, request, verifier, verified, trace


def _route_record(
    topology: RouteTopology,
    trace: object,
    request: GroundedExecutionRequest,
    verifier: object,
    verified: object,
) -> SettlementRouteRecord:
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
        f"{trace.transaction_id}-route-record",
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
        grounded_execution=verified,
        grounded_request=request,
        grounded_verifier=verifier,
    )


def _proposal_invocation(trace: object):
    context = CandidateModelContext.from_cycle_trace(
        trace,
        context_id=f"{trace.objective.objective_id}-model-context",
        allowed_route_ids=("path-alpha", "path-beta"),
        route_scores=(("path-alpha", 0.50), ("path-beta", 0.50)),
    )
    return ModelAdapter(
        FixtureModelProvider(
            lambda _: (
                '{"proposal":"inspect the candidate route",'
                '"reasoning":"declared-only provenance",'
                '"route_hint":"path-alpha"}'
            )
        )
    ).invoke(
        context,
        request_id=f"{trace.objective.objective_id}-model-request",
        prompt="Return only the bounded proposal JSON.",
        model_id="fixture-model",
    ).invocation


def test_grounded_settlement_reaches_route_learning_without_promoting_model_output() -> None:
    _, request, verifier, verified, trace = _grounded_trace("interaction-success")
    topology = RouteTopology.fixture("interaction-topology")
    invocation = _proposal_invocation(trace)

    report = validate_interaction_chain(
        trace,
        grounded_execution=verified,
        grounded_request=request,
        grounded_verifier=verifier,
        topology=topology,
        route_record=_route_record(topology, trace, request, verifier, verified),
        model_invocation=invocation,
    )

    assert report.grounded_execution is True
    assert report.delivery_stages == (
        DeliveryStage.EXECUTION,
        DeliveryStage.EVIDENCE,
        DeliveryStage.SETTLEMENT,
        DeliveryStage.LEARNING,
    )
    assert report.route_learning is not None
    assert report.route_learning.disposition == "accepted"
    assert report.route_learning.effect == "strengthen"
    assert report.proposal_id == invocation.proposal.proposal_id
    assert all("proposal_id" not in item.provenance for item in trace.evidence)


def test_signal_and_physiology_inhibition_combine_conservatively() -> None:
    objective = Objective(
        "interaction-deadlock-objective",
        "Do not execute when independent candidate safeguards inhibit.",
        provenance={"transaction_id": "interaction-deadlock-transaction"},
    )
    uncertainty = make_bound_signal(
        "interaction-uncertainty",
        "candidate.uncertainty",
        transaction_id="interaction-deadlock-transaction",
        objective_id=objective.objective_id,
        task_state_id=f"{objective.objective_id}-state-4",
        task_state_version=4,
    )
    critical = PhysiologySnapshot(
        "interaction-critical",
        "interaction-deadlock-transaction",
        objective.objective_id,
        f"{objective.objective_id}-state-4",
        4,
        contradiction_density=0.90,
        resource_pressure=0.95,
        protected_reserve=0.05,
    )

    trace = run_constitutional_cycle(
        objective, signals=(uncertainty,), physiology=critical
    )
    report = validate_interaction_chain(trace)

    assert report.action_inhibited is True
    assert report.grounded_execution is False
    assert report.delivery_stages == ()
    assert trace.states[4].phase == "inhibited"
    assert trace.evidence == ()
    assert trace.learning_update is None


def test_inhibiting_physiology_rejects_a_previously_verified_execution() -> None:
    objective, request, verifier, verified, _ = _grounded_trace("interaction-inhibit")
    critical = PhysiologySnapshot(
        "interaction-inhibit-critical",
        request.transaction_id,
        objective.objective_id,
        f"{objective.objective_id}-state-4",
        4,
        contradiction_density=0.90,
        protected_reserve=0.05,
    )

    with pytest.raises(CycleInvariantError, match="inhibited candidate action"):
        run_constitutional_cycle(
            objective,
            physiology=critical,
            grounded_execution=verified,
            grounded_request=request,
            grounded_verifier=verifier,
        )


def test_replay_is_read_only_but_duplicate_delivery_and_out_of_order_stages_fail() -> None:
    objective, request, verifier, verified, trace = _grounded_trace("interaction-replay")

    assert replay_grounded_execution(
        verified.record,
        request=request,
        authorized_state=trace.states[4],
        verifier=verifier,
    ) == verified
    with pytest.raises(GroundedExecutionRejected, match="duplicate grounded evidence"):
        run_constitutional_cycle(
            objective,
            grounded_execution=verified,
            grounded_request=request,
            grounded_verifier=verifier,
        )
    with pytest.raises(InteractionValidationError, match="out of order"):
        validate_interaction_chain(
            trace,
            grounded_execution=verified,
            grounded_request=request,
            grounded_verifier=verifier,
            delivery_stages=(
                DeliveryStage.EVIDENCE,
                DeliveryStage.EXECUTION,
                DeliveryStage.SETTLEMENT,
                DeliveryStage.LEARNING,
            ),
        )


def test_stale_grounded_lineage_and_tampered_model_provenance_fail_closed() -> None:
    _, request, verifier, verified, trace = _grounded_trace("interaction-lineage")
    stale_request = replace(request, transaction_id="other-transaction")
    with pytest.raises(InteractionValidationError, match="independently reverified"):
        validate_interaction_chain(
            trace,
            grounded_execution=verified,
            grounded_request=stale_request,
            grounded_verifier=verifier,
        )

    invocation = _proposal_invocation(trace)
    tampered = replace(
        invocation,
        proposal=replace(invocation.proposal, objective_id="other-objective"),
    )
    with pytest.raises(InteractionValidationError, match="structurally replayable"):
        validate_interaction_chain(
            trace,
            grounded_execution=verified,
            grounded_request=request,
            grounded_verifier=verifier,
            model_invocation=tampered,
        )


def test_grounded_route_credit_requires_the_exact_trace_and_verified_record() -> None:
    _, request, verifier, verified, trace = _grounded_trace("interaction-route-bound")
    topology = RouteTopology.fixture("interaction-route-bound-topology")
    record = _route_record(topology, trace, request, verifier, verified)

    with pytest.raises(
        PlasticRoutingValidationError, match="grounded route inputs must agree"
    ):
        replace(
            record,
            grounded_execution=None,
            grounded_request=None,
            grounded_verifier=None,
        )

    other_objective, other_request, other_verifier, other_verified, other_trace = (
        _grounded_trace("interaction-other-trace")
    )
    del other_objective
    with pytest.raises(InteractionValidationError, match="does not bind the validated"):
        validate_interaction_chain(
            other_trace,
            grounded_execution=other_verified,
            grounded_request=other_request,
            grounded_verifier=other_verifier,
            topology=topology,
            route_record=record,
        )

    learned, learning = apply_settlement_learning(topology, record)
    assert learned.version == topology.version + 1
    assert learning.disposition == "accepted"


def test_model_context_cannot_be_reused_across_transactions() -> None:
    _, request, verifier, verified, trace = _grounded_trace("interaction-model-bound")
    context = CandidateModelContext.from_cycle_trace(
        trace,
        context_id="interaction-model-stale-context",
        allowed_route_ids=("path-alpha",),
        route_scores=(("path-alpha", 0.50),),
    )
    stale_context = replace(
        context,
        transaction_id="other-transaction",
        provenance={
            "source": "kraken_r_cycle",
            "transaction_id": "other-transaction",
            "objective_id": context.objective_id,
            "task_state_id": context.task_state_id,
            "task_state_version": context.task_state_version,
        },
    )
    invocation = ModelAdapter(
        FixtureModelProvider(
            lambda _: (
                '{"proposal":"inspect candidate route",'
                '"reasoning":"declared-only provenance",'
                '"route_hint":"path-alpha"}'
            )
        )
    ).invoke(
        stale_context,
        request_id="interaction-model-stale-request",
        prompt="Return only the bounded proposal JSON.",
        model_id="fixture-model",
    ).invocation

    with pytest.raises(InteractionValidationError, match="context does not match"):
        validate_interaction_chain(
            trace,
            grounded_execution=verified,
            grounded_request=request,
            grounded_verifier=verifier,
            model_invocation=invocation,
        )