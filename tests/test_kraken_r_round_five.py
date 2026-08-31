"""Cross-layer qualification for the bounded Round 5 composition."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy
import tempfile

import pytest

from kraken_r import (
    AdaptiveState,
    ContributionKind,
    GroundedDeliveryLedger,
    GroundedExecutionExecutor,
    GroundedExecutionRequest,
    HumanAuthority,
    HumanInput,
    HumanInputRole,
    InformationSufficiency,
    Objective,
    ProcessingOperation,
    RoundFiveError,
    RouteTopology,
    SemanticJob,
    SufficiencyDecision,
    TaskState,
    assess_information_sufficiency,
    consumer_atlas,
    make_contribution,
    make_grounded_action,
    replay_round_five,
    run_connected_processing,
    run_round_five,
)


ROOT = Path(__file__).resolve().parents[1]
SEMANTIC = runpy.run_path(str(ROOT / "tests" / "test_kraken_r_semantic_capability.py"))


def _semantic_trace():
    _, callback = SEMANTIC["_semantic"](SemanticJob.MECHANISM_GENERATION)
    problem = SEMANTIC["_problem"]()
    state = SEMANTIC["_state"]()
    trace = run_connected_processing(
        problem,
        (SEMANTIC["_capability"](SemanticJob.MECHANISM_GENERATION),),
        state,
        operation_callbacks={ProcessingOperation.MODEL_PROPOSAL: callback},
    )
    return problem, state, trace


def _inputs(label: str):
    problem, adaptive, trace = _semantic_trace()
    transaction_id = f"{label}-transaction"
    objective = Objective(
        "semantic-task",
        "Ground one semantic candidate in bounded real work.",
        success_criteria=("the independently bounded tests pass",),
        provenance={"transaction_id": transaction_id},
    )
    state = TaskState(
        "semantic-task-state-5",
        objective.objective_id,
        5,
        "authorized",
        values={"action_id": "semantic-task-action"},
    )
    request = GroundedExecutionRequest(
        f"{label}-request",
        transaction_id,
        objective.objective_id,
        state.state_id,
        state.version,
        make_grounded_action(objective.objective_id),
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
    executor = GroundedExecutionExecutor(
        delivery_ledger=GroundedDeliveryLedger(
            Path(tempfile.mkdtemp(prefix="round-five-ledger-")) / "receipts.json"
        )
    )
    return objective, problem, adaptive, trace, request, executor, state


def test_semantic_grounded_success_changes_existing_topology_and_replays():
    objective, problem, adaptive, trace, request, executor, state = _inputs("round-five-full")
    result = run_round_five(
        objective,
        problem,
        adaptive,
        trace,
        request,
        executor=executor,
        authorized_state=state,
        context_id="candidate",
    )
    assert result.learning_episode.attribution.eligible is True
    assert result.adaptive_audit is not None
    assert result.changed_topology is True
    assert result.topology_after.version == result.topology_before.version + 1
    before = dict((route.route_id, route.weight) for route in result.topology_before.routes)
    after = dict(result.later_computation)
    assert after["semantic-route-alpha"] > before["semantic-route-alpha"]

    _, replayed = replay_round_five(result, adaptive_state=adaptive)
    assert replayed.route_topology == result.topology_after


@pytest.mark.parametrize(
    "kind",
    (
        ContributionKind.EXTERNAL_RETRIEVAL,
        ContributionKind.DETERMINISTIC_TOOL,
        ContributionKind.HUMAN_INPUT,
    ),
)
def test_decisive_non_reasoning_inputs_withhold_topology_credit(kind):
    objective, problem, adaptive, trace, request, executor, state = _inputs(
        f"round-five-{kind.value}"
    )
    contribution = make_contribution(
        f"{kind.value}-decisive",
        kind,
        {"answer": "decisive"},
        role="decisive",
        correction=kind is ContributionKind.HUMAN_INPUT,
    )
    result = run_round_five(
        objective,
        problem,
        adaptive,
        trace,
        request,
        executor=executor,
        authorized_state=state,
        context_id="candidate",
        contributions=(contribution,),
    )
    assert result.learning_episode.attribution.eligible is False
    assert result.adaptive_audit is None
    assert result.topology_after == result.topology_before


def test_information_sufficiency_has_conservative_deterministic_precedence():
    assert assess_information_sufficiency(non_identifiable=True).decision is SufficiencyDecision.BLOCKED
    assert assess_information_sufficiency(clarification_needed=True).decision is SufficiencyDecision.CLARIFICATION
    assert assess_information_sufficiency(qualified_retrieval_needed=True).decision is SufficiencyDecision.RETRIEVAL
    assert assess_information_sufficiency(deterministic_calculation_needed=True).decision is SufficiencyDecision.CALCULATION
    assert assess_information_sufficiency(explicit_assumptions=("alpha is stable",)).decision is SufficiencyDecision.ASSUMPTION
    continued = assess_information_sufficiency(
        known_facts=("alpha increased",),
        unresolved_parts=("the cause remains unknown",),
        continue_authorized=True,
    )
    assert continued.decision is SufficiencyDecision.CONTINUE_WITH_UNCERTAINTY
    assert continued.unresolved_parts == ("the cause remains unknown",)


def test_continuation_cannot_substitute_unrelated_uncertainty():
    objective, problem, adaptive, trace, request, executor, state = _inputs(
        "round-five-continue"
    )
    sufficiency = InformationSufficiency(
        SufficiencyDecision.CONTINUE_WITH_UNCERTAINTY,
        known_facts=("the bounded test is identified",),
        unresolved_parts=("the broader cause remains unknown",),
        reasons=("continue only as far as the bounded test supports",),
        continue_authorized=True,
    )
    with pytest.raises(RoundFiveError, match="exactly preserve"):
        run_round_five(
            objective,
            problem,
            adaptive,
            trace,
            request,
            executor=executor,
            authorized_state=state,
            context_id="candidate",
            sufficiency=sufficiency,
        )

    authorization = HumanInput(
        "continue-input",
        HumanInputRole.AUTHORIZATION,
        {
            "instruction": "continue with maximum defensible progress",
            "unresolved_parts": ["the broader cause remains unknown"],
        },
        HumanAuthority.CONTINUE,
    )
    with pytest.raises(RoundFiveError, match="exactly preserve"):
        run_round_five(
            objective,
            problem,
            adaptive,
            trace,
            replace(request, request_id="round-five-continue-authorized-request"),
            executor=executor,
            authorized_state=state,
            context_id="candidate",
            sufficiency=sufficiency,
            human_inputs=(authorization,),
        )


def test_consumer_atlas_exposes_causal_and_declared_noop_fields():
    atlas = consumer_atlas()
    by_field = {item.field: item for item in atlas}
    assert by_field["route_topology"].causal is True
    assert by_field["qualified_retrieval"].causal is False
    assert by_field["human_input"].causal is False
    assert by_field["deterministic_tool"].causal is False
    assert by_field["declared_only_model_fields"].causal is False


def test_tampered_sufficiency_hash_fails_closed():
    original = assess_information_sufficiency(known_facts=("fact",))
    with pytest.raises(RoundFiveError, match="hash"):
        InformationSufficiency(
            original.decision,
            original.known_facts,
            original.unresolved_parts,
            original.explicit_assumptions,
            original.reasons,
            original.continue_authorized,
            "0" * 64,
        )


def test_topology_cannot_credit_a_route_that_did_not_produce_cognition():
    objective, problem, adaptive, trace, request, executor, state = _inputs(
        "round-five-route-mismatch"
    )
    routes = tuple(
        replace(route, weight=0.25 if route.route_id.endswith("alpha") else 0.65)
        for route in adaptive.route_topology.routes
    )
    mismatched = replace(
        adaptive,
        route_topology=RouteTopology(
            adaptive.route_topology.topology_id,
            adaptive.route_topology.version,
            routes,
            adaptive.route_topology.generation,
        ),
    )
    with pytest.raises(RoundFiveError, match="does not match the cognition"):
        run_round_five(
            objective,
            problem,
            mismatched,
            trace,
            request,
            executor=executor,
            authorized_state=state,
            context_id="candidate",
        )


def test_cycle_prediction_mismatch_stops_before_execution():
    objective, problem, adaptive, trace, request, executor, state = _inputs(
        "round-five-failure-prediction"
    )
    with pytest.raises(RoundFiveError, match="only bind a success prediction"):
        run_round_five(
            objective,
            problem,
            adaptive,
            trace,
            request,
            executor=executor,
            authorized_state=state,
            context_id="candidate",
            expected_outcome="failure",
        )