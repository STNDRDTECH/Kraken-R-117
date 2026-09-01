"""Adversarial qualification for zero-authority shadow precommitments."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import runpy

import pytest

from kraken_r import (
    CounterfactualShadowError,
    ShadowBundle,
    ShadowExecutionReceipt,
    ShadowIntervention,
    ShadowMode,
    ShadowQualityEffect,
    ShadowResourceUsage,
    ShadowSpecification,
    RoundFiveError,
    replay_round_five,
    replay_round_five_serialized,
    replay_shadow_diagnostic,
    run_round_five,
    shadow_experiment_binding_hash,
    make_contribution,
)


ROOT = Path(__file__).resolve().parents[1]
ROUND_FIVE = runpy.run_path(str(ROOT / "tests" / "test_kraken_r_round_five.py"))


def _spec(trace, expected="failure", *, shadow_id="shadow-one"):
    return ShadowSpecification(
        shadow_id,
        f"{trace.request.request_id}-semantic-model",
        ShadowIntervention.REMOVE,
        expected,
        ShadowResourceUsage(1, 500, 100, 20),
    )


def _run(label: str, expected: str):
    objective, problem, adaptive, trace, request, executor, state = ROUND_FIVE["_inputs"](label)
    result = run_round_five(
        objective,
        problem,
        adaptive,
        trace,
        request,
        executor=executor,
        authorized_state=state,
        context_id="candidate",
        shadow_specifications=(_spec(trace, expected),),
        primary_resources=ShadowResourceUsage(1, 600, 120, 25),
    )
    return adaptive, result


def test_degraded_shadow_preserves_existing_eligibility_and_replays_offline():
    adaptive, result = _run("shadow-degraded", "failure")
    diagnostic = result.shadow_diagnostic
    assert diagnostic is not None
    assert diagnostic.eligible is True
    assert diagnostic.comparisons[0].effect is ShadowQualityEffect.DEGRADED
    assert result.adaptive_audit is not None
    assert not hasattr(result.shadow_bundle.shadows[0], "execution_request")
    assert result.shadow_bundle.primary_resources.model_calls == 1
    assert result.shadow_bundle.shadows[0].resources.input_tokens == 500

    replayed = replay_shadow_diagnostic(
        result.shadow_bundle.to_dict(),
        result.learning_episode,
        diagnostic,
    )
    assert replayed == diagnostic
    _, replayed_state = replay_round_five(result, adaptive_state=adaptive)
    assert replayed_state.route_topology == result.topology_after


def test_unchanged_shadow_reports_without_vetoing_grounded_credit():
    adaptive, result = _run("shadow-unchanged", "success")
    assert result.learning_episode.attribution.eligible is True
    assert result.shadow_diagnostic.eligible is False
    assert result.shadow_diagnostic.comparisons[0].effect is ShadowQualityEffect.UNCHANGED
    assert result.adaptive_audit is not None
    assert result.topology_after != result.topology_before
    _, replayed_state = replay_round_five(result, adaptive_state=adaptive)
    assert replayed_state.route_topology == result.topology_after


def test_no_public_post_outcome_constructor_and_forged_removal_fails_closed():
    _, result = _run("shadow-post-outcome", "failure")
    import kraken_r
    assert not hasattr(kraken_r, "seal_shadow_bundle")

    payload = result.shadow_bundle.to_dict()
    payload["shadows"][0]["retained_contribution_hashes"] = [
        result.commitment.contributions[0].contribution_hash
    ]
    payload["shadows"][0]["shadow_hash"] = ""
    payload["bundle_hash"] = ""
    with pytest.raises(CounterfactualShadowError, match="forges"):
        ShadowBundle.from_dict(payload)


def test_cross_shadow_identity_contamination_and_resource_overflow_are_rejected():
    objective, problem, adaptive, trace, request, executor, state = ROUND_FIVE["_inputs"](
        "shadow-bounds"
    )
    duplicate = _spec(trace, shadow_id="same")
    with pytest.raises(Exception, match="shadow"):
        run_round_five(
            objective,
            problem,
            adaptive,
            trace,
            request,
            executor=executor,
            authorized_state=state,
            context_id="candidate",
            shadow_specifications=(duplicate, duplicate),
        )
    with pytest.raises(CounterfactualShadowError, match="input_tokens"):
        ShadowResourceUsage(input_tokens=12_001)


@pytest.mark.parametrize(
    ("primary_usage", "shadow_usage"),
    (
        (ShadowResourceUsage(model_calls=1), ShadowResourceUsage(model_calls=1)),
        (
            ShadowResourceUsage(input_tokens=12_000, output_tokens=4_000),
            ShadowResourceUsage(input_tokens=11_000),
        ),
        (
            ShadowResourceUsage(compute_units=1_000),
            ShadowResourceUsage(compute_units=1_000),
        ),
    ),
)
def test_aggregate_budget_includes_primary_and_every_shadow(
    primary_usage, shadow_usage
):
    objective, problem, adaptive, trace, request, executor, state = ROUND_FIVE["_inputs"](
        f"shadow-aggregate-{primary_usage.model_calls}-{primary_usage.total_tokens}-{primary_usage.compute_units}"
    )
    extras = (
        make_contribution("aggregate-extra-one", "deterministic_tool", {"value": 1}),
        make_contribution("aggregate-extra-two", "external_retrieval", {"value": 2}),
    )
    targets = (
        f"{trace.request.request_id}-semantic-model",
        "aggregate-extra-one",
        "aggregate-extra-two",
    )
    specs = tuple(
        ShadowSpecification(
            f"aggregate-shadow-{index}",
            target,
            ShadowIntervention.REMOVE,
            "failure",
            shadow_usage,
        )
        for index, target in enumerate(targets)
    )
    with pytest.raises(RoundFiveError) as raised:
        run_round_five(
            objective,
            problem,
            adaptive,
            trace,
            request,
            executor=executor,
            authorized_state=state,
            context_id="candidate",
            contributions=extras,
            shadow_specifications=specs,
            primary_resources=primary_usage,
        )
    assert isinstance(raised.value.__cause__, CounterfactualShadowError)
    assert "primary-and-shadow" in str(raised.value.__cause__)


def test_tampering_and_cross_outcome_binding_are_rejected():
    _, first = _run("shadow-first", "failure")
    _, second = _run("shadow-second", "failure")
    with pytest.raises(CounterfactualShadowError, match="not bound"):
        replay_shadow_diagnostic(first.shadow_bundle, second.learning_episode)

    payload = json.loads(json.dumps(first.shadow_bundle.to_dict()))
    payload["shadows"][0]["expected_outcome"] = "success"
    with pytest.raises(CounterfactualShadowError, match="hash"):
        ShadowBundle.from_dict(payload)


def test_resealed_forged_diagnostic_cannot_preserve_credit():
    _, result = _run("shadow-forged-diagnostic", "failure")
    original_shadow = result.shadow_bundle.shadows[0]
    forged_shadow = replace(
        original_shadow,
        expected_outcome="success",
        shadow_hash="",
    )
    with pytest.raises(CounterfactualShadowError, match="pre-execution intent"):
        replace(
            result.shadow_bundle,
            shadows=(forged_shadow,),
            bundle_hash="",
        )


def test_alteration_must_actually_change_one_declared_contributor():
    objective, problem, adaptive, trace, request, executor, state = ROUND_FIVE["_inputs"](
        "shadow-alter-source"
    )
    target_id = f"{trace.request.request_id}-semantic-model"
    replacement = make_contribution(
        target_id,
        "semantic_model",
        {"different": True},
    )
    spec = ShadowSpecification(
        "alter-shadow",
        target_id,
        ShadowIntervention.ALTER,
        "failure",
        replacement_contribution=replacement,
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
        shadow_specifications=(spec,),
    )
    assert (
        result.shadow_bundle.shadows[0].replacement_contribution.contribution_hash
        == replacement.contribution_hash
    )


def test_full_post_outcome_rehash_and_new_post_outcome_seal_are_rejected():
    _, result = _run("shadow-history-attack", "failure")
    original = result.shadow_bundle.shadows[0]
    post_hoc_rehash_rejected = False
    try:
        forged = replace(original, expected_outcome="success", shadow_hash="")
        replace(result.shadow_bundle, shadows=(forged,), bundle_hash="")
    except CounterfactualShadowError:
        post_hoc_rehash_rejected = True

    objective, problem, adaptive, trace, request, executor, state = ROUND_FIVE["_inputs"](
        "shadow-late-seal"
    )
    completed = run_round_five(
        objective,
        problem,
        adaptive,
        trace,
        request,
        executor=executor,
        authorized_state=state,
        context_id="candidate",
        shadow_specifications=(_spec(trace),),
    )
    from kraken_r.counterfactual_shadows import _seal_shadow_bundle

    post_outcome_seal_rejected = False
    try:
        _seal_shadow_bundle(
            completed.commitment,
            (_spec(trace),),
            bundle_id=completed.shadow_bundle.bundle_id,
            lifecycle_ledger=executor.delivery_ledger,
        )
    except CounterfactualShadowError:
        post_outcome_seal_rejected = True
    assert post_hoc_rehash_rejected is True
    assert post_outcome_seal_rejected is True


def test_cold_serialized_round_five_replay_reconstructs_authority_records():
    _, result = _run("shadow-cold-replay", "failure")
    replayed = replay_round_five_serialized(
        json.dumps(result.to_dict()),
        trusted_executor=result.outcome.trusted_executor,
    )
    assert replayed.trace_hash == result.trace_hash
    assert replayed.learning_episode.to_dict() == result.learning_episode.to_dict()
    assert replayed.shadow_bundle == result.shadow_bundle
    assert replayed.shadow_diagnostic == result.shadow_diagnostic
    assert (
        replayed.grounded_execution.record.to_dict()
        == result.outcome.verified_execution.record.to_dict()
    )
    _, _, _, _, _, other_executor, _ = ROUND_FIVE["_inputs"](
        "shadow-cold-replay-other-trust"
    )
    with pytest.raises(RoundFiveError, match="external trust anchor"):
        replay_round_five_serialized(
            json.dumps(result.to_dict()),
            trusted_executor=other_executor.trusted_executor(),
        )


def test_structural_shadow_cannot_claim_measured_resources():
    measured = ShadowResourceUsage(
        compute_units=1,
        accounting_basis="grounded_execution_receipt",
    )
    with pytest.raises(CounterfactualShadowError, match="measured resources"):
        ShadowSpecification(
            "structural-measured",
            "contributor",
            ShadowIntervention.REMOVE,
            "failure",
            measured,
        )


def test_executed_shadow_requires_exact_verified_receipt():
    objective, problem, adaptive, trace, request, executor, state = ROUND_FIVE["_inputs"](
        "executed-shadow"
    )
    contributor_id = f"{trace.request.request_id}-semantic-model"
    experiment_binding = shadow_experiment_binding_hash(
        shadow_id="executed-shadow-one",
        contributor_id=contributor_id,
        intervention=ShadowIntervention.REMOVE,
        replacement_contribution_hash=None,
    )
    shadow_request = replace(
        request,
        request_id="executed-shadow-receipt-request",
        action=replace(
            request.action,
            parameters={
                **dict(request.action.parameters),
                "shadow_experiment_binding_hash": experiment_binding,
            },
        ),
    )
    record = executor.execute(shadow_request, authorized_state=state)
    receipt = ShadowExecutionReceipt(
        "executed-shadow-receipt",
        "executed-shadow-one",
        contributor_id,
        ShadowIntervention.REMOVE,
        None,
        shadow_request,
        record,
        state,
        executor.trusted_executor(),
    )
    specification = ShadowSpecification(
        "executed-shadow-one",
        contributor_id,
        ShadowIntervention.REMOVE,
        receipt.observed_outcome,
        receipt.resources,
        mode=ShadowMode.EXECUTED,
        execution_receipt=receipt,
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
        shadow_specifications=(specification,),
    )
    comparison = result.shadow_diagnostic.comparisons[0].to_dict()
    assert comparison["claim_basis"] == "receipt_backed_execution_observation"
    assert comparison["causal_effect_supported"] is False
    with pytest.raises(CounterfactualShadowError, match="exact verified receipt"):
        ShadowSpecification(
            "executed-shadow-one",
            "substituted-contributor",
            ShadowIntervention.REMOVE,
            receipt.observed_outcome,
            receipt.resources,
            mode=ShadowMode.EXECUTED,
            execution_receipt=receipt,
        )