"""Acceptance tests for bounded, settlement-grounded candidate plastic routing."""

from __future__ import annotations

from dataclasses import replace

import pytest

import ast
from pathlib import Path

from kraken_r import (
    CycleMode,
    MAX_WEIGHT,
    MIN_WEIGHT,
    PlasticRoutingValidationError,
    RouteTopology,
    SettlementRouteRecord,
    apply_settlement_learning,
    replay_settlement_learning,
    run_constitutional_cycle,
    select_candidate_route,
)

ROOT = Path(__file__).resolve().parents[1]


def test_no_module_redefines_a_conflicting_route_weight_ceiling() -> None:
    """``plastic_routing.py`` is the single source of ``MIN_WEIGHT``/``MAX_WEIGHT``.
    ``adaptive_substrate.py`` and ``dynamical_substrate.py`` must import these
    constants rather than declaring their own value under the same name --
    two differing ceilings for the same bounded quantity would silently pick
    whichever one a given code path happened to read."""

    assert 0.0 <= MIN_WEIGHT < MAX_WEIGHT <= 1.0

    for module_name in ("adaptive_substrate", "dynamical_substrate"):
        source = (ROOT / "kraken_r" / f"{module_name}.py").read_text()
        tree = ast.parse(source)
        assignments = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        redefined = assignments & {"MIN_WEIGHT", "MAX_WEIGHT"}
        assert not redefined, (
            f"{module_name}.py redefines {redefined}; it must import the "
            "canonical constants from plastic_routing.py instead"
        )
        assert "MIN_WEIGHT" in source and "MAX_WEIGHT" in source


def _trace(label: str, mode: CycleMode) -> object:
    return run_constitutional_cycle(
        __import__("kraken_r").Objective(
            f"{label}-objective",
            "Produce a constitutionally settled observation.",
            provenance={"transaction_id": f"{label}-tx"},
        ),
        mode=mode,
    )


def _record(topology: RouteTopology, trace: object, route_id: str | None = None) -> SettlementRouteRecord:
    selected = select_candidate_route(
        topology,
        "candidate-work",
        transaction_id=trace.transaction_id,
        objective_id=trace.objective.objective_id,
        task_state_id=trace.states[4].state_id,
        task_state_version=trace.states[4].version,
    )
    if route_id is not None:
        selected = replace(selected, route_id=route_id)
    evidence_ids = tuple(item.evidence_id for item in trace.evidence)
    return SettlementRouteRecord(
        f"{trace.transaction_id}-route-record",
        selected,
        trace,
        provenance={
            "transaction_id": trace.transaction_id,
            "objective_id": trace.objective.objective_id,
            "task_state_id": trace.states[4].state_id,
            "task_state_version": trace.states[4].version,
            "route_id": selected.route_id,
            "settlement_id": trace.settlement.settlement_id,
            "evidence_ids": evidence_ids,
        },
    )


def _route(topology: RouteTopology, route_id: str) -> object:
    return next(route for route in topology.routes if route.route_id == route_id)


def _alpha_preferred_topology() -> RouteTopology:
    baseline = RouteTopology.fixture()
    return replace(
        baseline,
        routes=tuple(
            replace(route, weight=0.60) if route.route_id == "path-alpha" else route
            for route in baseline.routes
        ),
    )


def test_settled_success_history_strengthens_and_changes_later_choice() -> None:
    topology = RouteTopology.fixture()
    before = select_candidate_route(
        topology, "candidate-work", transaction_id="before-tx", objective_id="before-objective",
        task_state_id="before-state", task_state_version=4,
    )
    assert before.route_id == "path-alpha"

    trace_one = _trace("success-one", CycleMode.SUCCESS)
    topology, first = apply_settlement_learning(topology, _record(topology, trace_one))
    trace_two = _trace("success-two", CycleMode.SUCCESS)
    topology, second = apply_settlement_learning(topology, _record(topology, trace_two))

    assert first.effect == second.effect == "strengthen"
    assert _route(topology, "path-alpha").weight == pytest.approx(0.70)
    later = select_candidate_route(
        topology, "candidate-work", transaction_id="later-tx", objective_id="later-objective",
        task_state_id="later-state", task_state_version=4,
    )
    assert later.route_id == "path-alpha"
    assert later.candidate_scores[0][1] > later.candidate_scores[1][1]


def test_settled_failure_weakens_and_reorganizes_later_choice() -> None:
    topology = _alpha_preferred_topology()
    failure_one = _trace("failure-one", CycleMode.FAILURE)
    topology, first = apply_settlement_learning(topology, _record(topology, failure_one))
    failure_two = _trace("failure-two", CycleMode.FAILURE)
    topology, second = apply_settlement_learning(topology, _record(topology, failure_two))

    assert first.effect == second.effect == "weaken"
    assert _route(topology, "path-alpha").weight == pytest.approx(0.40)
    later = select_candidate_route(
        topology, "candidate-work", transaction_id="later-tx", objective_id="later-objective",
        task_state_id="later-state", task_state_version=4,
    )
    assert later.route_id == "path-beta"


@pytest.mark.parametrize("mode", (CycleMode.CONTRADICTION, CycleMode.INSUFFICIENT_EVIDENCE))
def test_nongrounded_outcomes_withhold_learning(mode: CycleMode) -> None:
    topology = RouteTopology.fixture()
    unchanged, trace = apply_settlement_learning(topology, _record(topology, _trace(mode.value, mode)))
    assert unchanged == topology
    assert trace.disposition == "withheld"
    assert trace.effect == "none"
    assert trace.learning_update is None


def test_ablation_removes_learned_routing_gain() -> None:
    topology = _alpha_preferred_topology()
    for label in ("ablated-failure-one", "ablated-failure-two"):
        topology, _ = apply_settlement_learning(
            topology, _record(topology, _trace(label, CycleMode.FAILURE))
        )
    learned = select_candidate_route(
        topology, "candidate-work", transaction_id="learned-tx", objective_id="learned-objective",
        task_state_id="learned-state", task_state_version=4,
    )
    ablated = select_candidate_route(
        topology.reset(), "candidate-work", transaction_id="reset-tx", objective_id="reset-objective",
        task_state_id="reset-state", task_state_version=4,
    )
    assert learned.route_id == "path-beta"
    assert learned.candidate_scores[0][1] > learned.candidate_scores[1][1]
    assert ablated.route_id == "path-alpha"
    assert ablated.candidate_scores[0][1] == ablated.candidate_scores[1][1] == 0.50


def test_learning_history_replay_is_deterministic() -> None:
    initial = RouteTopology.fixture()
    first_trace = _trace("replay-success", CycleMode.SUCCESS)
    first_record = _record(initial, first_trace)
    after_first, _ = apply_settlement_learning(initial, first_record)
    second_trace = _trace("replay-failure", CycleMode.FAILURE)
    second_record = _record(after_first, second_trace)

    first = replay_settlement_learning(initial, (first_record, second_record))
    second = replay_settlement_learning(initial, (first_record, second_record))
    assert first == second
    assert _route(first[0], "path-alpha").weight == pytest.approx(0.50)
    assert _route(first[0], "path-alpha").success_count == 1
    assert _route(first[0], "path-alpha").failure_count == 1


def test_learning_is_reversible_and_resists_runaway_reinforcement() -> None:
    topology = RouteTopology.fixture()
    history = []
    for index in range(8):
        trace = _trace(f"bounded-success-{index}", CycleMode.SUCCESS)
        record = _record(topology, trace)
        history.append(record)
        topology, accepted = apply_settlement_learning(topology, record)
        assert accepted.weight_after <= 0.75
    assert _route(topology, "path-alpha").weight == 0.75

    failure = _trace("reversible-failure", CycleMode.FAILURE)
    topology, weakened = apply_settlement_learning(topology, _record(topology, failure))
    assert weakened.weight_after == pytest.approx(0.65)
    assert _route(topology, "path-beta").weight == 0.50


def test_stale_duplicate_and_malformed_provenance_fail_closed() -> None:
    topology = RouteTopology.fixture()
    record = _record(topology, _trace("rejection", CycleMode.SUCCESS))
    updated, _ = apply_settlement_learning(topology, record)
    with pytest.raises(PlasticRoutingValidationError, match="stale"):
        apply_settlement_learning(updated, record)
    with pytest.raises(PlasticRoutingValidationError, match="generation"):
        apply_settlement_learning(topology.reset(), record)

    malformed = replace(record, provenance={"transaction_id": record.selection.transaction_id})
    with pytest.raises(PlasticRoutingValidationError, match="provenance"):
        apply_settlement_learning(topology, malformed)

    wrong_evidence = replace(
        record,
        provenance={**dict(record.provenance), "evidence_ids": ("forged-evidence",)},
    )
    with pytest.raises(PlasticRoutingValidationError, match="evidence"):
        apply_settlement_learning(topology, wrong_evidence)

    forged_route = replace(record.selection, route_id="path-beta")
    forged_record = replace(
        record,
        selection=forged_route,
        provenance={**dict(record.provenance), "route_id": "path-beta"},
    )
    with pytest.raises(PlasticRoutingValidationError, match="selected route"):
        apply_settlement_learning(topology, forged_record)


def test_declared_signal_like_evidence_cannot_reinforce_a_route() -> None:
    topology = RouteTopology.fixture()
    record = _record(topology, _trace("declared-only", CycleMode.SUCCESS))
    declared_trace = replace(
        record.constitutional_trace,
        evidence=tuple(
        replace(item, grade=__import__("kraken_r").EvidenceGrade.DECLARED)
        for item in record.evidence
        ),
    )
    with pytest.raises(PlasticRoutingValidationError, match="constitutional trace"):
        apply_settlement_learning(
            topology, replace(record, constitutional_trace=declared_trace)
        )


def test_hard_history_budget_prevents_evicted_duplicate_credit() -> None:
    topology = RouteTopology.fixture()
    for index in range(16):
        record = _record(topology, _trace(f"budget-{index}", CycleMode.SUCCESS))
        topology, accepted = apply_settlement_learning(topology, record)
        assert accepted.disposition == "accepted"
    overflow = _record(topology, _trace("budget-overflow", CycleMode.SUCCESS))
    with pytest.raises(PlasticRoutingValidationError, match="budget"):
        apply_settlement_learning(topology, overflow)


def test_selection_is_advisory_and_route_state_has_bounded_audit_links() -> None:
    topology = RouteTopology.fixture()
    selection = select_candidate_route(
        topology, "candidate-work", transaction_id="bound-tx", objective_id="bound-objective",
        task_state_id="bound-state", task_state_version=4,
    )
    assert selection.route_id == "path-alpha"
    assert not hasattr(selection, "action_id")
    assert not hasattr(selection, "signal_id")
    assert not hasattr(selection, "goal_id")