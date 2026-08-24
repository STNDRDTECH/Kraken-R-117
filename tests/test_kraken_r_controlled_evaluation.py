"""Acceptance tests for Round 7's comparable held-out evaluation harness."""

from __future__ import annotations

from dataclasses import replace

from kraken_r import (
    CycleMode,
    Objective,
    RouteTopology,
    SettlementRouteRecord,
    apply_settlement_learning,
    run_constitutional_cycle,
    select_candidate_route,
)
from kraken_r.controlled_evaluation import (
    ControlledEvaluationHarness,
    EvaluationMode,
    HeldOutTask,
)
from kraken_r.llm_adapter import FixtureModelProvider


def _failure_record(topology: RouteTopology, label: str) -> SettlementRouteRecord:
    trace = run_constitutional_cycle(
        Objective(
            f"{label}-objective",
            "Create an independently settled fixture failure.",
            provenance={"transaction_id": f"{label}-transaction"},
        ),
        mode=CycleMode.FAILURE,
    )
    selection = select_candidate_route(
        topology,
        "candidate-work",
        transaction_id=trace.transaction_id,
        objective_id=trace.objective.objective_id,
        task_state_id=trace.states[4].state_id,
        task_state_version=trace.states[4].version,
    )
    return SettlementRouteRecord(
        f"{label}-record",
        selection,
        trace,
        provenance={
            "transaction_id": trace.transaction_id,
            "objective_id": trace.objective.objective_id,
            "task_state_id": trace.states[4].state_id,
            "task_state_version": trace.states[4].version,
            "route_id": selection.route_id,
            "settlement_id": trace.settlement.settlement_id,
            "evidence_ids": tuple(item.evidence_id for item in trace.evidence),
        },
    )


def _learned_beta_topology() -> RouteTopology:
    baseline = RouteTopology.fixture()
    topology = replace(
        baseline,
        routes=tuple(
            replace(route, weight=0.60) if route.route_id == "path-alpha" else route
            for route in baseline.routes
        ),
    )
    for label in ("evaluation-failure-one", "evaluation-failure-two"):
        topology, _ = apply_settlement_learning(
            topology, _failure_record(topology, label)
        )
    return topology


def test_evaluation_holds_model_budgets_and_tasks_constant_and_ablates_learning() -> None:
    topology = _learned_beta_topology()

    def provider_output(request: object) -> str:
        prompt = request.prompt
        route_hint = (
            "path-beta"
            if "candidate_route_code=R 0 0 1" in prompt
            else "path-alpha"
        )
        return (
            '{"proposal":"consider the supplied organizational context",'
            '"reasoning":"this is a declared candidate suggestion",'
            f'"route_hint":"{route_hint}"}}'
        )

    provider = FixtureModelProvider(provider_output, provider_id="held-out-fixture")
    harness = ControlledEvaluationHarness(
        provider,
        model_id="controlled-fixture-model",
        max_tokens=192,
        temperature=0.0,
    )
    report = harness.run(
        (
            HeldOutTask("heldout-one", "heldout-objective-one", "Route an unseen task.", "path-beta"),
            HeldOutTask("heldout-two", "heldout-objective-two", "Route another unseen task.", "path-beta"),
        ),
        topology,
    )

    assert report.control_errors == ()
    assert report.mode_rates == {
        "base_model": 0.0,
        "kraken_mediated": 1.0,
        "kraken_ablated": 0.0,
    }
    assert report.observed_deltas == {
        "kraken_mediated_minus_base": 1.0,
        "kraken_mediated_minus_ablated": 1.0,
    }
    assert report.claim_status == "not_claimed"
    assert len(provider.calls) == 6
    assert all(result.invocation is not None for result in report.results)
    assert all(result.structural_replay is not None for result in report.results)
    assert all(
        result.structural_replay.request_hash == result.invocation.input_hash
        and result.structural_replay.status == result.invocation.status
        and result.structural_replay.generation_replayed is False
        and result.structural_replay.structure_replayed is True
        for result in report.results
    )
    assert all(
        result.to_dict()["invocation"]["request"]["input_hash"]
        == result.to_dict()["structural_replay"]["request_hash"]
        for result in report.results
    )

    by_mode = {
        mode: [result for result in report.results if result.mode is mode]
        for mode in EvaluationMode
    }
    assert all(result.selected_route_id == "path-beta" for result in by_mode[EvaluationMode.KRAKEN_MEDIATED])
    assert all(result.selected_route_id == "path-alpha" for result in by_mode[EvaluationMode.KRAKEN_ABLATED])
    for task_id in ("heldout-one", "heldout-two"):
        bucket = [result for result in report.results if result.task_id == task_id]
        assert {result.model_id for result in bucket} == {"controlled-fixture-model"}
        assert {result.max_tokens for result in bucket} == {192}
        assert {result.temperature for result in bucket} == {0.0}
        assert {result.provider_id for result in bucket} == {"held-out-fixture"}
        assert len({result.configuration_fingerprint for result in bucket}) == 1