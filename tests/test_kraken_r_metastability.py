"""Acceptance tests for bounded, descriptive Stage 10.9 experiments."""

from __future__ import annotations

from dataclasses import replace

import pytest

from kraken_r import (
    DynamicalEvent,
    DynamicalState,
    DynamicalTick,
    ExperimentScenario,
    MAX_EXPERIMENT_TICKS,
    MetastabilityAblation,
    MetastabilityValidationError,
    compare_metastability,
    make_metastability_scenario,
    run_metastability_experiment,
)


def test_stable_experiment_replays_deterministically_with_complete_metrics() -> None:
    initial = DynamicalState.fixture("metastability-stable")
    ticks = make_metastability_scenario(initial, ExperimentScenario.STABLE, ticks=12)

    first = run_metastability_experiment(
        initial, ticks, experiment_id="stable-a", scenario=ExperimentScenario.STABLE
    )
    second = run_metastability_experiment(
        initial, ticks, experiment_id="stable-b", scenario=ExperimentScenario.STABLE
    )

    assert first.input_digest == second.input_digest
    assert first.final_state.to_dict() == second.final_state.to_dict()
    assert first.metrics.to_dict() == second.metrics.to_dict()
    assert first.metrics.operating_region == "settling"
    assert first.metrics.route_entropy > 0.0
    assert first.metrics.dominant_route_share < 1.0
    assert first.metrics.recurrence > 0
    assert first.metrics.task_performance is None
    assert first.to_dict()["promotes_canonical_state"] is False


def test_matched_ablations_keep_input_budget_and_change_only_declared_counterforces() -> None:
    initial = DynamicalState.fixture("metastability-ablations")
    ticks = make_metastability_scenario(
        initial, ExperimentScenario.NOISY_NONCREDITABLE, ticks=16
    )

    comparison = compare_metastability(
        initial,
        ticks,
        comparison_id="metastability-ablations",
        scenario=ExperimentScenario.NOISY_NONCREDITABLE,
    )
    reports = {report.ablation.label: report for report in comparison.reports}

    assert len(reports) == 6
    assert {report.input_digest for report in reports.values()} == {
        comparison.baseline.input_digest
    }
    assert {report.metrics.tick_count for report in reports.values()} == {len(ticks)}
    assert reports["without_surprise"].metrics.surprise_peak == 0.0
    assert reports["without_homeostasis"].metrics.resource_pressure_peak == 0.0
    assert reports["without_inhibition"].metrics.inhibition_rate == 0.0
    assert reports["without_anti_monopoly"].metrics.dominant_route_share == 1.0
    assert (
        "anti_monopoly_counterforce_removed"
        in reports["without_anti_monopoly"].metrics.failure_modes
    )


def test_inactivity_decay_is_measured_and_can_be_neutralized_only_in_comparison() -> None:
    initial = DynamicalState.fixture("metastability-decay")
    topology = initial.medium.adaptive_state.route_topology
    boosted_topology = replace(
        topology,
        routes=(replace(topology.routes[0], weight=0.60), *topology.routes[1:]),
    )
    initial = replace(
        initial,
        medium=replace(
            initial.medium,
            adaptive_state=replace(
                initial.medium.adaptive_state, route_topology=boosted_topology
            ),
        ),
    )
    ticks = make_metastability_scenario(initial, ExperimentScenario.STAGNATING, ticks=8)

    full = run_metastability_experiment(
        initial, ticks, scenario=ExperimentScenario.STAGNATING
    )
    without_decay = run_metastability_experiment(
        initial,
        ticks,
        scenario=ExperimentScenario.STAGNATING,
        ablation=MetastabilityAblation.without("decay"),
    )

    assert full.metrics.stagnation_duration == len(ticks)
    assert full.metrics.decay_rate > 0.0
    assert without_decay.metrics.decay_rate == 0.0
    assert without_decay.final_state.medium.adaptive_state == initial.medium.adaptive_state
    assert full.final_state.medium.adaptive_state != initial.medium.adaptive_state


def test_noisy_noncreditable_inputs_cannot_earn_adaptive_credit_or_mutate_topology() -> None:
    initial = DynamicalState.fixture("metastability-noncreditable")
    ticks = make_metastability_scenario(
        initial, ExperimentScenario.NOISY_NONCREDITABLE, ticks=12
    )

    report = run_metastability_experiment(
        initial, ticks, scenario=ExperimentScenario.NOISY_NONCREDITABLE
    )

    assert report.metrics.task_observations == 0
    assert report.metrics.task_performance is None
    assert report.metrics.plasticity_rate == 0.0
    assert report.final_state.medium.adaptive_state == initial.medium.adaptive_state
    assert report.metrics.topology_turnover == 0


def test_experiment_generator_and_ablation_arm_budgets_fail_closed() -> None:
    initial = DynamicalState.fixture("metastability-bounds")
    scenario = make_metastability_scenario(initial, ExperimentScenario.STABLE, ticks=1)

    with pytest.raises(MetastabilityValidationError, match="budget exceeded"):
        run_metastability_experiment(
            initial,
            (item for _ in range(MAX_EXPERIMENT_TICKS + 1) for item in scenario),
        )
    with pytest.raises(MetastabilityValidationError, match="budget exceeded"):
        compare_metastability(
            initial,
            scenario,
            ablations=(
                MetastabilityAblation.full()
                for _ in range(7)
            ),
        )
    with pytest.raises(MetastabilityValidationError, match="every single-factor"):
        compare_metastability(
            initial,
            scenario,
            ablations=(MetastabilityAblation.full(),),
        )
    with pytest.raises(MetastabilityValidationError, match="every single-factor"):
        compare_metastability(
            initial,
            scenario,
            ablations=(
                MetastabilityAblation.full(),
                MetastabilityAblation.without("homeostasis"),
                MetastabilityAblation.without("homeostasis"),
                MetastabilityAblation.without("inhibition"),
                MetastabilityAblation.without("surprise"),
                MetastabilityAblation.without("anti_monopoly"),
            ),
        )


def test_inhibition_ablation_sanitizes_authority_bearing_inputs_before_reduction() -> None:
    initial = DynamicalState.fixture("metastability-inhibition-boundary")
    ticks = (
        DynamicalTick(
            1,
            (
                DynamicalEvent.observation_event("inhibiting-mismatch", 0.0, 1.0),
                DynamicalEvent.resource_event("inhibiting-resource", 1.0),
                DynamicalEvent.rollback_event("rollback-input", "missing"),
            ),
        ),
    )

    report = run_metastability_experiment(
        initial,
        ticks,
        scenario=ExperimentScenario.NOISY_NONCREDITABLE,
        ablation=MetastabilityAblation.without("inhibition"),
    )

    assert report.traces[0].adaptive_audits == ()
    assert report.final_state.medium.adaptive_state == initial.medium.adaptive_state
    assert report.traces[0].event_ids == (
        "inhibiting-mismatch",
        "inhibiting-resource",
        "rollback-input",
    )