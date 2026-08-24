"""Acceptance tests for the Round 7 candidate-only proposal adapter."""

from __future__ import annotations

from dataclasses import replace
import json
import time
import threading

import pytest

from kraken_r import Authority, CycleMode, Objective, RouteTopology, run_constitutional_cycle
from kraken_r.llm_adapter import (
    AdapterFailureCode,
    CandidateModelContext,
    FixtureModelProvider,
    ModelAdapter,
    ModelAdapterValidationError,
    ModelProviderError,
    OpenAICompatibleProvider,
    ProviderRequest,
    ProviderResponse,
    replay_model_invocation,
)
from kraken_r.plastic_routing import PlasticRoutingValidationError, apply_settlement_learning


def _context(label: str = "adapter") -> CandidateModelContext:
    trace = run_constitutional_cycle(
        Objective(
            f"{label}-objective",
            "Produce only a bounded candidate suggestion.",
            provenance={"transaction_id": f"{label}-transaction"},
        ),
        mode=CycleMode.INSUFFICIENT_EVIDENCE,
    )
    return CandidateModelContext.from_cycle_trace(
        trace,
        context_id=f"{label}-context",
        allowed_route_ids=("path-alpha", "path-beta"),
        route_scores=(("path-alpha", 0.50), ("path-beta", 0.50)),
    )


def _success_output(route_hint: str = "path-alpha") -> str:
    return (
        '{"proposal":"inspect the bounded candidate route",'
        '"reasoning":"this is a declared suggestion only",'
        f'"route_hint":"{route_hint}"}}'
    )


def _invoke(provider: FixtureModelProvider) -> object:
    return ModelAdapter(provider).invoke(
        _context(),
        request_id="adapter-request",
        prompt="Return the strict Round 7 proposal object.",
        model_id="fixture-model",
    )


def test_typed_proposal_is_declared_only_and_structurally_replayable() -> None:
    result = _invoke(FixtureModelProvider(lambda _: _success_output()))

    assert result.accepted
    assert result.proposal is not None
    assert result.proposal.declared_only is True
    assert result.proposal.route_hint == "path-alpha"
    assert not hasattr(result.proposal, "evidence")
    assert not hasattr(result.proposal, "settlement")
    assert not hasattr(result.proposal, "action")
    assert not hasattr(result.proposal, "goal")
    assert not hasattr(result.proposal, "learning_update")
    assert set(result.proposal.to_dict()) == {
        "proposal_id",
        "request_id",
        "objective_id",
        "proposal",
        "reasoning",
        "route_hint",
        "declared_only",
    }

    replay = replay_model_invocation(result.invocation)
    assert replay.structure_replayed is True
    assert replay.generation_replayed is False
    assert replay.proposal == result.proposal


@pytest.mark.parametrize(
    ("factory", "expected"),
    (
        (lambda _: "not json", AdapterFailureCode.MALFORMED_OUTPUT),
        (
            lambda _: (
                '{"proposal":"do it","reasoning":"declared","route_hint":"path-alpha",'
                '"success":true}'
            ),
            AdapterFailureCode.MALFORMED_OUTPUT,
        ),
        (
            lambda _: ProviderResponse("fixture", "other-model", _success_output()),
            AdapterFailureCode.INVALID_PROVENANCE,
        ),
        (
            lambda _: ProviderResponse("other-fixture", "fixture-model", _success_output()),
            AdapterFailureCode.INVALID_PROVENANCE,
        ),
    ),
)
def test_malformed_or_invalid_provider_output_fails_closed(factory: object, expected: object) -> None:
    result = _invoke(FixtureModelProvider(factory))
    assert not result.accepted
    assert result.proposal is None
    assert result.failure_code is expected
    assert result.invocation.status == "failed"
    assert result.invocation.raw_output is not None
    assert result.invocation.output_hash is not None


def test_timeout_and_provider_exception_fail_closed() -> None:
    def timeout(_: object) -> object:
        raise TimeoutError("fixture timeout")

    def unavailable(_: object) -> object:
        raise ModelProviderError("fixture unavailable")

    timed_out = _invoke(FixtureModelProvider(timeout))
    unavailable_result = _invoke(FixtureModelProvider(unavailable))
    assert timed_out.failure_code is AdapterFailureCode.TIMEOUT
    assert unavailable_result.failure_code is AdapterFailureCode.PROVIDER_ERROR
    assert timed_out.proposal is unavailable_result.proposal is None


def test_adapter_enforces_a_deadline_for_a_noncooperative_provider() -> None:
    class SlowProvider:
        provider_id = "slow-fixture"

        def complete(self, request: object, *, timeout_seconds: float) -> object:
            time.sleep(0.10)
            return ProviderResponse(
                "slow-fixture", request.model_id, _success_output()
            )

    started = time.monotonic()
    result = ModelAdapter(SlowProvider(), timeout_seconds=0.01).invoke(
        _context("slow"),
        request_id="slow-request",
        prompt="Return the strict Round 7 proposal object.",
        model_id="fixture-model",
    )
    assert time.monotonic() - started < 0.08
    assert result.failure_code is AdapterFailureCode.TIMEOUT
    assert result.proposal is None


def test_adapter_rejects_before_dispatch_when_deadline_cannot_be_enforced() -> None:
    calls: list[object] = []

    class ThreadProvider:
        provider_id = "thread-fixture"

        def complete(self, request: object, *, timeout_seconds: float) -> object:
            calls.append(request)
            return ProviderResponse(
                "thread-fixture", request.model_id, _success_output()
            )

    holder: list[object] = []
    thread = threading.Thread(
        target=lambda: holder.append(
            ModelAdapter(ThreadProvider(), timeout_seconds=0.01).invoke(
                _context("thread"),
                request_id="thread-request",
                prompt="Return the strict Round 7 proposal object.",
                model_id="fixture-model",
            )
        )
    )
    thread.start()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert holder[0].failure_code is AdapterFailureCode.TIMEOUT
    assert calls == []


def test_context_provenance_is_exactly_bound_to_cycle_identity() -> None:
    context = _context()
    with pytest.raises(ModelAdapterValidationError, match="provenance"):
        replace(context, provenance={"source": "kraken_r_cycle"})


def test_context_projection_rejects_an_unauthorized_or_tampered_cycle_trace() -> None:
    trace = run_constitutional_cycle(
        Objective(
            "tampered-trace-objective",
            "Reject a trace that is not candidate-authorized.",
            provenance={"transaction_id": "tampered-trace-transaction"},
        )
    )
    tampered_trace = replace(
        trace, action=replace(trace.action, authority=Authority.NONE)
    )
    with pytest.raises(ModelAdapterValidationError, match="candidate-authorized"):
        CandidateModelContext.from_cycle_trace(
            tampered_trace, context_id="tampered-trace-context"
        )


def test_non_null_route_hint_requires_an_explicit_authorized_route() -> None:
    trace = run_constitutional_cycle(
        Objective(
            "empty-route-objective",
            "Do not authorize a route hint.",
            provenance={"transaction_id": "empty-route-transaction"},
        ),
        mode=CycleMode.INSUFFICIENT_EVIDENCE,
    )
    empty_route_context = CandidateModelContext.from_cycle_trace(
        trace, context_id="empty-route-context"
    )
    result = ModelAdapter(
        FixtureModelProvider(lambda _: _success_output("path-alpha"))
    ).invoke(
        empty_route_context,
        request_id="empty-route-request",
        prompt="Return the strict Round 7 proposal object.",
        model_id="fixture-model",
    )
    assert result.failure_code is AdapterFailureCode.MALFORMED_OUTPUT
    assert result.proposal is None


def test_model_self_report_cannot_be_credited_as_route_learning() -> None:
    result = _invoke(FixtureModelProvider(lambda _: _success_output()))
    assert result.proposal is not None
    with pytest.raises(PlasticRoutingValidationError, match="SettlementRouteRecord"):
        apply_settlement_learning(RouteTopology.fixture(), result.proposal)


def test_structural_replay_rejects_tampered_output() -> None:
    result = _invoke(FixtureModelProvider(lambda _: _success_output()))
    assert result.invocation is not None
    with pytest.raises(ModelAdapterValidationError, match="output hash"):
        replace(result.invocation, raw_output=_success_output("path-beta"))


def test_failed_received_output_is_hashed_and_structurally_replayable() -> None:
    result = _invoke(FixtureModelProvider(lambda _: "not JSON"))
    assert result.invocation is not None
    assert result.invocation.raw_output == "not JSON"
    replay = replay_model_invocation(result.invocation)
    assert replay.status == "failed"
    assert replay.output_hash == result.invocation.output_hash
    with pytest.raises(ModelAdapterValidationError, match="failed envelope"):
        replace(result.invocation, failure_code=AdapterFailureCode.TIMEOUT)


def test_openai_compatible_provider_normalizes_a_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": _success_output()},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                }
            ).encode("utf-8")

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return Response()

    monkeypatch.setattr("kraken_r.llm_adapter.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        "https://provider.example/v1/",
        "fixture-model",
        api_key="test-key",
        json_object_mode=True,
    )
    response = provider.complete(
        ProviderRequest("provider-request", "fixture-model", "bounded prompt", 32, 0.0),
        timeout_seconds=4.0,
    )

    assert captured["url"] == "https://provider.example/v1/chat/completions"
    assert captured["timeout"] == 4.0
    assert captured["payload"] == {
        "model": "fixture-model",
        "messages": [{"role": "user", "content": "bounded prompt"}],
        "max_tokens": 32,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    assert response.input_tokens == 11
    assert response.output_tokens == 7
    assert provider.configuration_fingerprint