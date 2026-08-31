"""Adversarial tests for sealed, grounded real-work outcome learning."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import runpy
import tempfile

import pytest

from kraken_r import (
    CompetenceAttribution,
    ContributionKind,
    ExecutionAttestor,
    GroundedLearningEpisode,
    GroundedDeliveryLedger,
    GroundedExecutionExecutor,
    GroundedOutcome,
    OutcomeKind,
    OutcomeLearningError,
    ProcessingOperation,
    SemanticJob,
    bind_grounded_outcome,
    commit_prediction,
    evaluate_grounded_outcome,
    make_contribution,
    replay_outcome_learning,
    replay_outcome_learning_history,
    run_connected_processing,
    seal_prediction_request,
)


ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_FIXTURES = runpy.run_path(
    str(ROOT / "tests" / "test_kraken_r_semantic_capability.py")
)
GROUNDED_FIXTURES = runpy.run_path(
    str(ROOT / "tests" / "test_kraken_r_grounded_execution.py")
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def semantic_trace():
    _, operation = SEMANTIC_FIXTURES["_semantic"](
        SemanticJob.MECHANISM_GENERATION
    )
    return run_connected_processing(
        SEMANTIC_FIXTURES["_problem"](),
        (
            SEMANTIC_FIXTURES["_capability"](
                SemanticJob.MECHANISM_GENERATION
            ),
        ),
        SEMANTIC_FIXTURES["_state"](),
        operation_callbacks={ProcessingOperation.MODEL_PROPOSAL: operation},
    )


@pytest.fixture(scope="module")
def successful_episode(semantic_trace):
    return _sealed_episode(semantic_trace, "outcome-learning-success")


def _sealed_episode(
    semantic_trace,
    name: str,
    *,
    contributions=(),
    files=None,
):
    _, state, request = GROUNDED_FIXTURES["_request"](
        name, files=files
    )
    commitment_id = f"{name}-commitment"
    criteria = ("the independently bounded tests pass",)
    sealed_request = seal_prediction_request(
        semantic_trace,
        request,
        commitment_id=commitment_id,
        expected_outcome="success",
        success_criteria=criteria,
        contributions=contributions,
        route_id="semantic-route-alpha",
    )
    commitment = commit_prediction(
        semantic_trace,
        commitment_id=commitment_id,
        grounded_request=sealed_request,
        expected_outcome="success",
        success_criteria=criteria,
        contributions=contributions,
        authorized_state=state,
        transaction_id=sealed_request.transaction_id,
        route_id="semantic-route-alpha",
    )
    executor = GroundedExecutionExecutor(
        delivery_ledger=GroundedDeliveryLedger(
            Path(tempfile.mkdtemp(prefix="outcome-learning-receipts-"))
            / "receipts.json"
        )
    )
    verifier = executor.verifier()
    record = executor.execute(sealed_request, authorized_state=state)
    verified = verifier.verify(
        record, request=sealed_request, authorized_state=state
    )
    outcome = bind_grounded_outcome(
        commitment,
        outcome_id=f"{name}-outcome",
        verified_execution=verified,
        request=sealed_request,
        authorized_state=state,
        trusted_executor=executor.trusted_executor(),
    )
    return evaluate_grounded_outcome(commitment, outcome)


def test_semantic_only_grounded_success_is_replayable_and_eligible(
    successful_episode,
) -> None:
    episode = successful_episode
    assert episode.comparison.observed_outcome is OutcomeKind.SUCCESS
    assert episode.comparison.prediction_matches is True
    assert episode.attribution.eligible is True
    assert episode.attribution.accepted_kinds == (
        ContributionKind.SEMANTIC_MODEL,
    )
    replayed, topology = replay_outcome_learning(episode.to_dict())
    assert topology is None
    assert replayed.episode_hash == episode.episode_hash


def test_post_hoc_prediction_and_attribution_edits_fail_closed(
    successful_episode,
) -> None:
    prediction_edit = successful_episode.to_dict()
    prediction_edit["commitment"]["expected_outcome"] = "failure"
    with pytest.raises(
        OutcomeLearningError,
        match="signed pre-outcome cognitive lineage|commitment hash",
    ):
        GroundedLearningEpisode.from_dict(prediction_edit)

    attribution_edit = successful_episode.to_dict()
    attribution_edit["attribution"]["eligible"] = False
    attribution_edit["attribution"]["disposition"] = "withheld"
    with pytest.raises(OutcomeLearningError, match="attribution hash"):
        GroundedLearningEpisode.from_dict(attribution_edit)


def test_rehashed_malformed_cognition_replay_still_fails_closed(
    successful_episode,
) -> None:
    payload = successful_episode.commitment.to_dict()
    payload["processing_trace"]["result"]["output"]["semantic_result"][
        "candidates"
    ][0]["statement"] = "post-hoc rewritten answer"
    payload["commitment_hash"] = _digest(
        {key: value for key, value in payload.items() if key != "commitment_hash"}
    )
    with pytest.raises(OutcomeLearningError, match="not replayable"):
        type(successful_episode.commitment).from_dict(payload)


def test_record_embedded_key_cannot_bootstrap_forged_trust(
    successful_episode,
) -> None:
    outcome = successful_episode.outcome
    forged_anchor = ExecutionAttestor().trusted_identity()
    with pytest.raises(OutcomeLearningError, match="not independently verifiable"):
        GroundedOutcome(
            outcome.outcome_id,
            outcome.commitment_id,
            outcome.commitment_hash,
            outcome.verified_execution,
            outcome.request,
            outcome.authorized_state,
            forged_anchor,
        )


@pytest.mark.parametrize(
    "kind",
    (
        ContributionKind.EXTERNAL_RETRIEVAL,
        ContributionKind.DETERMINISTIC_TOOL,
        ContributionKind.HUMAN_INPUT,
    ),
)
def test_apparent_success_from_non_reasoning_inputs_earns_no_credit(
    semantic_trace, successful_episode, kind
) -> None:
    supplemental = make_contribution(
        f"{kind.value}-input",
        kind,
        {"value": "decisive answer"},
        role="decisive",
        relevant=kind is not ContributionKind.EXTERNAL_RETRIEVAL,
        correction=kind is ContributionKind.HUMAN_INPUT,
    )
    episode = _sealed_episode(
        semantic_trace,
        f"{kind.value}-apparent-success",
        contributions=(supplemental,),
    )
    assert episode.attribution.eligible is False
    assert kind in episode.attribution.withheld_kinds


def test_partial_success_is_localized_and_withheld(semantic_trace) -> None:
    files = {
        "test_partial.py": (
            "def test_pass():\n    assert True\n\n"
            "def test_fail():\n    assert False\n"
        )
    }
    episode = _sealed_episode(
        semantic_trace, "outcome-learning-partial", files=files
    )
    assert episode.comparison.observed_outcome is OutcomeKind.PARTIAL
    assert episode.attribution.eligible is False


def test_duplicate_and_conflicting_outcome_delivery_fails_closed(
    successful_episode,
) -> None:
    with pytest.raises(OutcomeLearningError, match="duplicate outcome"):
        replay_outcome_learning_history(
            (successful_episode, successful_episode.to_dict())
        )

    second_outcome = replace(
        successful_episode.outcome,
        outcome_id="conflicting-outcome",
        outcome_hash="",
    )
    conflicting = evaluate_grounded_outcome(
        successful_episode.commitment, second_outcome
    )
    with pytest.raises(OutcomeLearningError, match="conflicting outcome"):
        replay_outcome_learning_history((successful_episode, conflicting))


def test_cross_bound_outcome_is_rejected(semantic_trace, successful_episode) -> None:
    base = successful_episode
    other_request = replace(
        base.outcome.request,
        request_id="same-task-different-request",
        files={
            "subject.py": "def answer():\n    return 7\n",
            "test_subject.py": (
                "from subject import answer\n\n"
                "def test_answer():\n    assert answer() == 7\n"
            ),
        },
    )
    with pytest.raises(OutcomeLearningError, match="outside the committed task"):
        bind_grounded_outcome(
            base.commitment,
            outcome_id="cross-bound-outcome",
            verified_execution=base.outcome.verified_execution,
            request=replace(other_request, transaction_id="another-transaction"),
            authorized_state=base.outcome.authorized_state,
            trusted_executor=base.outcome.trusted_executor,
        )
    with pytest.raises(OutcomeLearningError, match="differs from the pre-outcome"):
        bind_grounded_outcome(
            base.commitment,
            outcome_id="same-task-different-work",
            verified_execution=base.outcome.verified_execution,
            request=other_request,
            authorized_state=base.outcome.authorized_state,
            trusted_executor=base.outcome.trusted_executor,
        )


def test_historical_unsealed_execution_cannot_be_post_hoc_committed(
    semantic_trace,
) -> None:
    _, state, request, _, _, _, _ = GROUNDED_FIXTURES["_execute"](
        "historical-unsealed-outcome"
    )
    with pytest.raises(OutcomeLearningError, match="signed pre-outcome"):
        commit_prediction(
            semantic_trace,
            commitment_id="post-hoc-commitment",
            grounded_request=request,
            expected_outcome="success",
            success_criteria=("tests pass",),
            authorized_state=state,
            transaction_id=request.transaction_id,
            route_id="semantic-route-alpha",
        )


def test_unrelated_semantic_trace_cannot_claim_a_sealed_request(
    semantic_trace, successful_episode
) -> None:
    _, operation = SEMANTIC_FIXTURES["_semantic"](
        SemanticJob.FALSIFIER_GENERATION
    )
    unrelated = run_connected_processing(
        SEMANTIC_FIXTURES["_problem"](),
        (
            SEMANTIC_FIXTURES["_capability"](
                SemanticJob.FALSIFIER_GENERATION
            ),
        ),
        SEMANTIC_FIXTURES["_state"](),
        operation_callbacks={ProcessingOperation.MODEL_PROPOSAL: operation},
    )
    base = successful_episode
    with pytest.raises(OutcomeLearningError, match="signed pre-outcome"):
        commit_prediction(
            unrelated,
            commitment_id=base.commitment.commitment_id,
            grounded_request=base.outcome.request,
            expected_outcome=base.commitment.expected_outcome,
            success_criteria=base.commitment.success_criteria,
            authorized_state=base.outcome.authorized_state,
            transaction_id=base.outcome.request.transaction_id,
            route_id="semantic-route-alpha",
        )

def test_forged_eligible_attribution_cannot_reach_delegation(
    semantic_trace, successful_episode
) -> None:
    human = make_contribution(
        "forged-gate-human-input",
        ContributionKind.HUMAN_INPUT,
        {"correction": "the decisive answer"},
        role="decisive",
        correction=True,
    )
    legitimate = _sealed_episode(
        semantic_trace,
        "forged-gate",
        contributions=(human,),
    )
    forged = CompetenceAttribution(
        legitimate.attribution.contribution_ids,
        (ContributionKind.SEMANTIC_MODEL,),
        (),
        True,
        "eligible",
        (),
    )
    with pytest.raises(
        OutcomeLearningError, match="not the deterministic outcome-learning gate"
    ):
        GroundedLearningEpisode(
            legitimate.commitment,
            legitimate.outcome,
            legitimate.comparison,
            forged,
        )