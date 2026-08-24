"""Focused adversarial tests for the candidate-only Round 4 signal slice."""

from __future__ import annotations

from dataclasses import replace

import pytest

from kraken_r import (
    CycleInvariantError,
    EvidenceGrade,
    Objective,
    PropagationEffect,
    PropagationLimitError,
    SignalNetwork,
    SignalReplayRecord,
    SignalRule,
    SignalValidationError,
    make_bound_signal,
    replay_constitutional_signal_path,
    replay_signal_propagation,
    run_constitutional_cycle,
)


def _context() -> dict[str, object]:
    return {
        "transaction_id": "round4-transaction",
        "objective_id": "round4-objective",
        "task_state_id": "round4-objective-state-4",
        "task_state_version": 4,
    }


def _signal(
    signal_id: str,
    topic: str,
    **overrides: object,
):
    context = {**_context(), **overrides}
    return make_bound_signal(
        signal_id,
        topic,
        transaction_id=str(context["transaction_id"]),
        objective_id=str(context["objective_id"]),
        task_state_id=str(context["task_state_id"]),
        task_state_version=int(context["task_state_version"]),
        ttl=int(context.pop("ttl", 2)),
        created_tick=int(context.pop("created_tick", 0)),
        payload=context.pop("payload", {}),
    )


def test_signal_replay_is_deterministic_and_amplification_is_explicit() -> None:
    urgency = _signal("urgency-1", "candidate.urgency", payload={"reason": "deadline"})
    record = SignalReplayRecord(**_context(), signals=(urgency,))

    first = replay_signal_propagation(record)
    second = replay_signal_propagation(record)

    assert first.to_dict() == second.to_dict()
    assert first.delivered_signal_ids[0] == urgency.signal_id
    assert first.amplified_signal_ids == (urgency.signal_id,)
    assert any(
        step.effect is PropagationEffect.AMPLIFY
        and step.disposition == "propagated"
        for step in first.steps
    )


def test_uncertainty_signal_changes_only_the_candidate_trajectory_and_replays() -> None:
    objective = Objective(
        "round4-objective",
        "Show a bounded uncertainty signal inhibiting a candidate action.",
        provenance={"transaction_id": "round4-transaction"},
    )
    uncertainty = _signal("uncertainty-1", "candidate.uncertainty")

    enabled = run_constitutional_cycle(objective, signals=(uncertainty,))
    ablated = run_constitutional_cycle(objective, signals=())
    replayed = replay_constitutional_signal_path(objective, signals=(uncertainty,))

    assert enabled.to_dict() == replayed.to_dict()
    assert enabled.signal_trace is not None
    assert enabled.signal_trace.inhibits("candidate.action.authorize")
    assert enabled.states[4].phase == "inhibited"
    assert ablated.states[4].phase == "authorized"
    assert enabled.execution.status == "not_observed"
    assert enabled.decision.outcome == "insufficient_evidence"
    assert enabled.evidence == ()
    assert ablated.decision.outcome == "success"


def test_duplicate_delivery_is_deduplicated_without_changing_propagation() -> None:
    signal = _signal("duplicate-1", "candidate.urgency")
    trace = replay_signal_propagation(
        SignalReplayRecord(**_context(), signals=(signal, signal))
    )

    assert trace.delivered_signal_ids.count(signal.signal_id) == 1
    assert trace.duplicate_signal_ids == (signal.signal_id,)


def test_stale_task_state_binding_is_rejected() -> None:
    stale = _signal(
        "stale-1",
        "candidate.uncertainty",
        task_state_id="round4-objective-state-3",
        task_state_version=3,
    )

    with pytest.raises(SignalValidationError, match="stale or mismatched"):
        replay_signal_propagation(SignalReplayRecord(**_context(), signals=(stale,)))


def test_expired_ttl_is_rejected() -> None:
    expired = _signal("expired-1", "candidate.urgency", ttl=0, created_tick=0)
    record = SignalReplayRecord(**_context(), signals=(expired,), start_tick=1)

    with pytest.raises(SignalValidationError, match="TTL has expired"):
        replay_signal_propagation(record)


def test_provenance_mismatch_is_rejected() -> None:
    valid = _signal("provenance-1", "candidate.uncertainty")
    mismatched = replace(
        valid,
        provenance={**valid.provenance, "transaction_id": "other-transaction"},
    )

    with pytest.raises(SignalValidationError, match="provenance mismatch"):
        replay_signal_propagation(SignalReplayRecord(**_context(), signals=(mismatched,)))


def test_signal_cannot_claim_evidence_or_become_cycle_evidence() -> None:
    signal = _signal("evidence-claim-1", "candidate.uncertainty")
    invalid = replace(signal, evidence_grade=EvidenceGrade.OPERATIONAL)

    with pytest.raises(SignalValidationError, match="cannot claim evidence"):
        replay_signal_propagation(SignalReplayRecord(**_context(), signals=(invalid,)))

    objective = Objective(
        "round4-objective",
        "Verify signals never become evidence.",
        provenance={"transaction_id": "round4-transaction"},
    )
    trace = run_constitutional_cycle(objective, signals=(signal,))
    assert trace.evidence == ()
    assert signal.signal_id not in {
        evidence.evidence_id for evidence in trace.evidence
    }


def test_runaway_fanout_and_feedback_are_stopped_by_explicit_limits() -> None:
    source = _signal("fanout-1", "candidate.source")
    fanout_network = SignalNetwork(
        (
            SignalRule("source-to-one", "candidate.source", "candidate.one"),
            SignalRule("source-to-two", "candidate.source", "candidate.two"),
        ),
        max_fanout=1,
    )
    with pytest.raises(PropagationLimitError, match="fan-out"):
        fanout_network.propagate((source,), **_context())

    feedback_network = SignalNetwork(
        (
            SignalRule("a-to-b", "candidate.a", "candidate.b"),
            SignalRule("b-to-a", "candidate.b", "candidate.a"),
        ),
        max_deliveries=3,
    )
    feedback = _signal("feedback-1", "candidate.a", ttl=10)
    with pytest.raises(PropagationLimitError, match="deliveries"):
        feedback_network.propagate((feedback,), **_context())


def test_cycle_rejects_invalid_signal_input_before_execution() -> None:
    objective = Objective(
        "round4-objective",
        "Reject bad signal provenance.",
        provenance={"transaction_id": "round4-transaction"},
    )
    invalid = _signal("bad-input-1", "candidate.uncertainty")
    invalid = replace(
        invalid,
        provenance={**invalid.provenance, "objective_id": "wrong-objective"},
    )

    with pytest.raises(CycleInvariantError, match="signal propagation was rejected"):
        run_constitutional_cycle(objective, signals=(invalid,))