"""Narrow, candidate-only LLM proposal adapter for Kraken-R Round 7.

The adapter is deliberately subordinate to the constitutional cycle.  It can
ask an injected provider for a structured proposal, but it cannot create an
Action, Evidence, Settlement, LearningUpdate, goal, persistence record, or
runtime dispatch.  Provider generation is nondeterministic; the immutable
request/response envelope and structural replay are deterministic.

Legacy model modules were inspected for provider, timeout, and output-shape
ideas only.  This module does not import them, an event bus, a cost ledger, a
router, a background worker, or a live ROGAL runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import hashlib
import json
import os
import signal
import threading
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.request import Request as URLRequest, urlopen

from .contracts import Objective
from .cycle import ConstitutionalCycle, CycleInvariantError, CycleTrace


class ModelAdapterValidationError(ValueError):
    """Raised when a proposal crosses the model adapter boundary."""


class ModelProviderError(RuntimeError):
    """Raised when a provider cannot produce a provider response."""


class _ProviderDeadlineExpired(TimeoutError):
    """Private signal used to stop a synchronous provider call at its deadline."""


class AdapterFailureCode(str, Enum):
    """Explicit fail-closed outcomes; none is a candidate success."""

    INVALID_CONTEXT = "invalid_context"
    INVALID_REQUEST = "invalid_request"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"
    MALFORMED_OUTPUT = "malformed_output"
    INVALID_PROVENANCE = "invalid_provenance"


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(
        character.isspace() for character in value
    ):
        raise ModelAdapterValidationError(
            f"{field_name} must be a non-empty identifier without whitespace"
        )
    return value


def _text(value: str, field_name: str, *, max_length: int = 12000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelAdapterValidationError(f"{field_name} must be non-empty text")
    if len(value) > max_length:
        raise ModelAdapterValidationError(
            f"{field_name} exceeds the {max_length} character bound"
        )
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return value


def _digest(value: Any) -> str:
    canonical = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _deadline_enforceable() -> bool:
    """Whether this invocation can enforce a synchronous provider deadline."""

    return (
        threading.current_thread() is threading.main_thread()
        and hasattr(signal, "setitimer")
        and hasattr(signal, "ITIMER_REAL")
    )


def _failure_envelope_hash(
    request: "ModelRequest",
    failure_code: AdapterFailureCode,
    failure_message: str,
    output_hash: str | None,
) -> str:
    return _digest(
        {
            "request_hash": request.input_hash,
            "status": "failed",
            "failure_code": failure_code.value,
            "failure_message": failure_message,
            "output_hash": output_hash,
        }
    )


@contextmanager
def _provider_deadline(timeout_seconds: float) -> Any:
    """Enforce a synchronous deadline in the main thread without retained state.

    Injected providers also receive the same timeout explicitly.  The signal
    guard prevents a non-cooperative synchronous provider from holding the
    main candidate call forever on the supported Unix runtime; non-main-thread
    callers still get the provider's explicit cooperative timeout argument.
    """

    if not _deadline_enforceable():
        raise _ProviderDeadlineExpired("provider deadline cannot be enforced")
    def raise_timeout(_: int, __: Any) -> None:
        raise _ProviderDeadlineExpired("provider deadline expired")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


@dataclass(frozen=True)
class CandidateModelContext:
    """The only context a model may receive from a constitutional cycle.

    Evidence, execution observations, settlement, confidence, signals, and
    physiology are intentionally not fields in this record.  A route selection
    is advisory context, not an instruction or a dispatch decision.
    """

    context_id: str
    transaction_id: str
    objective_id: str
    task_state_id: str
    task_state_version: int
    objective_description: str
    task_phase: str
    allowed_route_ids: tuple[str, ...] = ()
    route_scores: tuple[tuple[str, float], ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "context_id",
            "transaction_id",
            "objective_id",
            "task_state_id",
            "task_phase",
        ):
            _identifier(getattr(self, field_name), field_name)
        if (
            isinstance(self.task_state_version, bool)
            or not isinstance(self.task_state_version, int)
            or self.task_state_version < 1
        ):
            raise ModelAdapterValidationError(
                "task_state_version must be a positive integer"
            )
        _text(self.objective_description, "objective_description")
        if not isinstance(self.allowed_route_ids, (tuple, list)):
            raise ModelAdapterValidationError("allowed_route_ids must be a tuple")
        route_ids = tuple(
            _identifier(route_id, "allowed route id")
            for route_id in self.allowed_route_ids
        )
        if len(set(route_ids)) != len(route_ids):
            raise ModelAdapterValidationError("allowed_route_ids must be unique")
        scores: list[tuple[str, float]] = []
        for item in self.route_scores:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ModelAdapterValidationError(
                    "route_scores must contain route id and weight pairs"
                )
            route_id, score = item
            _identifier(route_id, "route score id")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ModelAdapterValidationError("route score must be numeric")
            if not 0.0 <= float(score) <= 1.0:
                raise ModelAdapterValidationError("route score must be from 0 through 1")
            scores.append((str(route_id), float(score)))
        if len({route_id for route_id, _ in scores}) != len(scores):
            raise ModelAdapterValidationError("route_scores must be unique")
        if route_ids and any(route_id not in route_ids for route_id, _ in scores):
            raise ModelAdapterValidationError(
                "route_scores must be restricted to allowed_route_ids"
            )
        if not isinstance(self.provenance, Mapping):
            raise ModelAdapterValidationError("provenance must be a mapping")
        provenance = dict(self.provenance)
        required = {
            "source": "kraken_r_cycle",
            "transaction_id": self.transaction_id,
            "objective_id": self.objective_id,
            "task_state_id": self.task_state_id,
            "task_state_version": self.task_state_version,
        }
        if provenance != required:
            raise ModelAdapterValidationError(
                "context provenance must exactly bind the candidate context"
            )
        object.__setattr__(self, "allowed_route_ids", route_ids)
        object.__setattr__(self, "route_scores", tuple(scores))
        object.__setattr__(self, "provenance", _freeze(provenance))

    @classmethod
    def from_cycle_trace(
        cls,
        trace: CycleTrace,
        *,
        context_id: str,
        allowed_route_ids: tuple[str, ...] = (),
        route_scores: tuple[tuple[str, float], ...] = (),
    ) -> "CandidateModelContext":
        """Project a cycle to safe model context without exposing its outcome."""

        if not isinstance(trace, CycleTrace) or len(trace.states) < 5:
            raise ModelAdapterValidationError(
                "model context requires a complete CycleTrace"
            )
        try:
            ConstitutionalCycle._validate_trace(trace)
        except CycleInvariantError as exc:
            raise ModelAdapterValidationError(
                "model context requires a valid candidate-authorized CycleTrace"
            ) from exc
        state = trace.states[4]
        if state.phase != "authorized":
            raise ModelAdapterValidationError(
                "model context requires the cycle authorized task state"
            )
        if state.objective_id != trace.objective.objective_id:
            raise ModelAdapterValidationError("trace state is not bound to objective")
        return cls(
            context_id=context_id,
            transaction_id=trace.transaction_id,
            objective_id=trace.objective.objective_id,
            task_state_id=state.state_id,
            task_state_version=state.version,
            objective_description=trace.objective.description,
            task_phase=state.phase,
            allowed_route_ids=allowed_route_ids,
            route_scores=route_scores,
            provenance={
                "source": "kraken_r_cycle",
                "transaction_id": trace.transaction_id,
                "objective_id": trace.objective.objective_id,
                "task_state_id": state.state_id,
                "task_state_version": state.version,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "transaction_id": self.transaction_id,
            "objective_id": self.objective_id,
            "task_state_id": self.task_state_id,
            "task_state_version": self.task_state_version,
            "objective_description": self.objective_description,
            "task_phase": self.task_phase,
            "allowed_route_ids": list(self.allowed_route_ids),
            "route_scores": [list(item) for item in self.route_scores],
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class ModelRequest:
    """Immutable, bounded model input with an explicit cycle binding."""

    request_id: str
    context: CandidateModelContext
    prompt: str
    model_id: str
    max_tokens: int = 256
    temperature: float = 0.0
    prompt_template_version: str = "round-7-v1"

    def __post_init__(self) -> None:
        _identifier(self.request_id, "request_id")
        if not isinstance(self.context, CandidateModelContext):
            raise ModelAdapterValidationError("request context is required")
        _text(self.prompt, "prompt")
        _identifier(self.model_id, "model_id")
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or not 1 <= self.max_tokens <= 4096
        ):
            raise ModelAdapterValidationError("max_tokens must be from 1 through 4096")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not 0.0 <= float(self.temperature) <= 2.0
        ):
            raise ModelAdapterValidationError("temperature must be from 0 through 2")
        _identifier(self.prompt_template_version, "prompt_template_version")

    @property
    def configuration_fingerprint(self) -> str:
        return _digest(
            {
                "model_id": self.model_id,
                "max_tokens": self.max_tokens,
                "temperature": float(self.temperature),
                "prompt_template_version": self.prompt_template_version,
            }
        )

    @property
    def input_hash(self) -> str:
        return _digest(
            {
                "request_id": self.request_id,
                "context": self.context.to_dict(),
                "prompt": self.prompt,
                "configuration_fingerprint": self.configuration_fingerprint,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "context": self.context.to_dict(),
            "prompt": self.prompt,
            "model_id": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": float(self.temperature),
            "prompt_template_version": self.prompt_template_version,
            "configuration_fingerprint": self.configuration_fingerprint,
            "input_hash": self.input_hash,
        }


@dataclass(frozen=True)
class ProviderRequest:
    """Provider-facing request; it contains no credential or persistence handle."""

    request_id: str
    model_id: str
    prompt: str
    max_tokens: int
    temperature: float


@dataclass(frozen=True)
class ProviderResponse:
    """Minimal provider result; metadata is restricted to non-authoritative usage."""

    provider_id: str
    model_id: str
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.provider_id, "provider_id")
        _identifier(self.model_id, "model_id")
        if not isinstance(self.content, str):
            raise ModelAdapterValidationError("provider content must be text")
        for field_name in ("input_tokens", "output_tokens"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ModelAdapterValidationError(
                    f"{field_name} must be a non-negative integer"
                )


@runtime_checkable
class ModelProvider(Protocol):
    """The adapter's only provider dependency."""

    provider_id: str

    def complete(
        self, request: ProviderRequest, *, timeout_seconds: float
    ) -> ProviderResponse:
        ...


class FixtureModelProvider:
    """Deterministic provider for tests and controlled evaluation."""

    provider_id = "fixture"

    def __init__(
        self,
        response_factory: Callable[[ProviderRequest], ProviderResponse | str],
        *,
        provider_id: str = "fixture",
    ) -> None:
        if not callable(response_factory):
            raise ModelAdapterValidationError("response_factory must be callable")
        self.response_factory = response_factory
        self.provider_id = _identifier(provider_id, "provider_id")
        self.calls: list[ProviderRequest] = []

    def complete(
        self, request: ProviderRequest, *, timeout_seconds: float
    ) -> ProviderResponse:
        if timeout_seconds <= 0:
            raise TimeoutError("fixture timeout must be positive")
        self.calls.append(request)
        result = self.response_factory(request)
        if isinstance(result, str):
            return ProviderResponse(
                self.provider_id, request.model_id, result
            )
        if not isinstance(result, ProviderResponse):
            raise ModelProviderError("fixture returned a non-provider response")
        return result


class OpenAICompatibleProvider:
    """Optional configured provider using an OpenAI-compatible base or endpoint.

    This is intentionally an injected leaf.  It has no fallback model, retry
    loop, cache, budget authority, event emission, or access to Kraken state.
    """

    def __init__(
        self,
        endpoint: str,
        model_id: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 15.0,
        provider_id: str = "openai_compatible",
        json_object_mode: bool = False,
    ) -> None:
        configured_endpoint = _text(endpoint, "endpoint", max_length=2048).rstrip("/")
        self.endpoint = (
            configured_endpoint
            if configured_endpoint.endswith("/chat/completions")
            else f"{configured_endpoint}/chat/completions"
        )
        self.model_id = _identifier(model_id, "model_id")
        self.api_key = api_key
        if timeout_seconds <= 0:
            raise ModelAdapterValidationError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self.provider_id = _identifier(provider_id, "provider_id")
        if not isinstance(json_object_mode, bool):
            raise ModelAdapterValidationError("json_object_mode must be a boolean")
        self.json_object_mode = json_object_mode

    @property
    def configuration_fingerprint(self) -> str:
        """Bounded provider configuration identity without exposing credentials."""

        return _digest(
            {
                "provider_id": self.provider_id,
                "model_id": self.model_id,
                "endpoint_hash": _digest(self.endpoint),
                "timeout_seconds": self.timeout_seconds,
                "json_object_mode": self.json_object_mode,
            }
        )

    @classmethod
    def from_environment(
        cls,
        *,
        endpoint_env: str = "AI_INTEGRATIONS_OPENAI_BASE_URL",
        model_env: str = "KRAKEN_R_LLM_MODEL",
        key_env: str = "AI_INTEGRATIONS_OPENAI_API_KEY",
    ) -> "OpenAICompatibleProvider":
        endpoint = os.environ.get(endpoint_env)
        model_id = os.environ.get(model_env)
        if not endpoint or not model_id:
            raise ModelProviderError(
                f"configured provider requires {endpoint_env} and {model_env}"
            )
        return cls(endpoint, model_id, api_key=os.environ.get(key_env))

    def complete(
        self, request: ProviderRequest, *, timeout_seconds: float
    ) -> ProviderResponse:
        if request.model_id != self.model_id:
            raise ModelProviderError("request model does not match configured provider")
        if timeout_seconds <= 0:
            raise TimeoutError("provider timeout must be positive")
        request_payload: dict[str, Any] = {
            "model": request.model_id,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if self.json_object_mode:
            request_payload["response_format"] = {"type": "json_object"}
        payload = json.dumps(request_payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url_request = URLRequest(
            self.endpoint, data=payload, headers=headers, method="POST"
        )
        try:
            with urlopen(
                url_request, timeout=min(self.timeout_seconds, timeout_seconds)
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except TimeoutError:
            raise
        except (HTTPError, URLError, OSError) as exc:
            raise ModelProviderError(f"provider request failed: {type(exc).__name__}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelProviderError("provider returned invalid JSON") from exc
        try:
            choice = body["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelProviderError("provider response shape is invalid") from exc
        return ProviderResponse(
            provider_id=self.provider_id,
            model_id=request.model_id,
            content=content,
            input_tokens=(body.get("usage") or {}).get("prompt_tokens"),
            output_tokens=(body.get("usage") or {}).get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
        )


@dataclass(frozen=True)
class ModelProposal:
    """A declared candidate suggestion; it has no evidence-shaped fields."""

    proposal_id: str
    request_id: str
    objective_id: str
    proposal: str
    reasoning: str
    route_hint: str | None
    declared_only: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "proposal_id",
            "request_id",
            "objective_id",
        ):
            _identifier(getattr(self, field_name), field_name)
        _text(self.proposal, "proposal", max_length=4000)
        _text(self.reasoning, "reasoning", max_length=8000)
        if self.route_hint is not None:
            _identifier(self.route_hint, "route_hint")
        if self.declared_only is not True:
            raise ModelAdapterValidationError("model proposals are always declared_only")

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "request_id": self.request_id,
            "objective_id": self.objective_id,
            "proposal": self.proposal,
            "reasoning": self.reasoning,
            "route_hint": self.route_hint,
            "declared_only": True,
        }


@dataclass(frozen=True)
class ModelInvocation:
    """Immutable provenance envelope for one generation attempt."""

    request: ModelRequest
    provider_id: str
    model_id: str
    raw_output: str | None
    proposal: ModelProposal | None
    status: str
    failure_code: AdapterFailureCode | None
    failure_message: str | None
    output_hash: str | None
    failure_envelope_hash: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    nondeterministic_generation: bool = True

    def __post_init__(self) -> None:
        _identifier(self.provider_id, "provider_id")
        _identifier(self.model_id, "model_id")
        if self.model_id != self.request.model_id:
            raise ModelAdapterValidationError("invocation model is not request model")
        if self.status not in {"accepted", "failed"}:
            raise ModelAdapterValidationError("invocation status is invalid")
        if self.status == "accepted":
            if self.proposal is None or self.failure_code is not None:
                raise ModelAdapterValidationError(
                    "accepted invocation requires proposal and no failure"
                )
            if self.raw_output is None or self.output_hash != _digest(self.raw_output):
                raise ModelAdapterValidationError("accepted output hash is invalid")
        else:
            if (
                self.proposal is not None
                or not isinstance(self.failure_code, AdapterFailureCode)
            ):
                raise ModelAdapterValidationError(
                    "failed invocation requires failure and no proposal"
                )
            if self.raw_output is None and self.output_hash is not None:
                raise ModelAdapterValidationError(
                    "failed invocation cannot hash absent provider output"
                )
            if self.raw_output is not None and self.output_hash != _digest(self.raw_output):
                raise ModelAdapterValidationError("failed output hash is invalid")
            if not self.failure_message:
                raise ModelAdapterValidationError("failed invocation requires a message")
            expected_failure_hash = _failure_envelope_hash(
                self.request,
                self.failure_code,
                self.failure_message,
                self.output_hash,
            )
            if self.failure_envelope_hash != expected_failure_hash:
                raise ModelAdapterValidationError("failed envelope hash is invalid")
        if self.status == "accepted" and self.failure_envelope_hash is not None:
            raise ModelAdapterValidationError(
                "accepted invocation cannot contain a failure envelope hash"
            )
        for field_name in ("input_tokens", "output_tokens"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ModelAdapterValidationError(
                    f"{field_name} must be a non-negative integer"
                )
        if self.nondeterministic_generation is not True:
            raise ModelAdapterValidationError(
                "provider generation must be explicitly marked nondeterministic"
            )

    @property
    def input_hash(self) -> str:
        return self.request.input_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "raw_output": self.raw_output,
            "proposal": self.proposal.to_dict() if self.proposal else None,
            "status": self.status,
            "failure_code": self.failure_code.value if self.failure_code else None,
            "failure_message": self.failure_message,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "failure_envelope_hash": self.failure_envelope_hash,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "nondeterministic_generation": True,
        }


@dataclass(frozen=True)
class ModelAdapterResult:
    """Result boundary consumed by candidate code; failures fail closed."""

    invocation: ModelInvocation | None
    proposal: ModelProposal | None
    failure_code: AdapterFailureCode | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        if self.proposal is not None and self.failure_code is not None:
            raise ModelAdapterValidationError("result cannot have proposal and failure")
        if self.invocation is not None:
            if self.proposal is not self.invocation.proposal:
                raise ModelAdapterValidationError("result and invocation disagree")
            if self.failure_code is not self.invocation.failure_code:
                raise ModelAdapterValidationError("result failure and invocation disagree")

    @property
    def accepted(self) -> bool:
        return self.proposal is not None and self.failure_code is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "proposal": self.proposal.to_dict() if self.proposal else None,
            "failure_code": self.failure_code.value if self.failure_code else None,
            "failure_message": self.failure_message,
            "invocation": self.invocation.to_dict() if self.invocation else None,
        }


_FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "action",
        "evidence",
        "settlement",
        "success",
        "failure",
        "confidence",
        "goal",
        "learning_update",
        "execution",
        "self_report",
    }
)
_PROPOSAL_KEYS = frozenset({"proposal", "reasoning", "route_hint"})


class ModelAdapter:
    """Stateless, fail-closed adapter for proposal/reasoning generation."""

    def __init__(self, provider: ModelProvider, *, timeout_seconds: float = 15.0):
        if not isinstance(provider, ModelProvider):
            raise ModelAdapterValidationError(
                "provider must implement the ModelProvider protocol"
            )
        _identifier(provider.provider_id, "provider.provider_id")
        if timeout_seconds <= 0:
            raise ModelAdapterValidationError("timeout_seconds must be positive")
        self.provider = provider
        self.timeout_seconds = float(timeout_seconds)

    def invoke(
        self,
        context: CandidateModelContext,
        *,
        request_id: str,
        prompt: str,
        model_id: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        prompt_template_version: str = "round-7-v1",
    ) -> ModelAdapterResult:
        try:
            request = ModelRequest(
                request_id=request_id,
                context=context,
                prompt=prompt,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                prompt_template_version=prompt_template_version,
            )
        except ModelAdapterValidationError as exc:
            return ModelAdapterResult(None, None, AdapterFailureCode.INVALID_REQUEST, str(exc))

        provider_id = getattr(self.provider, "provider_id", "unknown_provider")
        if not _deadline_enforceable():
            return self._failed(
                request,
                provider_id,
                AdapterFailureCode.TIMEOUT,
                "provider deadline cannot be enforced in this calling thread",
            )
        try:
            provider_request = ProviderRequest(
                request.request_id,
                request.model_id,
                request.prompt,
                request.max_tokens,
                float(request.temperature),
            )
            with _provider_deadline(self.timeout_seconds):
                response = self.provider.complete(
                    provider_request, timeout_seconds=self.timeout_seconds
                )
            if not isinstance(response, ProviderResponse):
                raise ModelProviderError("provider did not return ProviderResponse")
            if response.provider_id != provider_id:
                return self._failed(
                    request,
                    provider_id,
                    AdapterFailureCode.INVALID_PROVENANCE,
                    "provider response identity does not match injected provider",
                    raw_output=response.content,
                )
            if response.model_id != request.model_id:
                return self._failed(
                    request,
                    provider_id,
                    AdapterFailureCode.INVALID_PROVENANCE,
                    "provider response model provenance mismatch",
                    raw_output=response.content,
                )
            try:
                proposal = self._parse_proposal(request, response.content)
            except ModelAdapterValidationError as exc:
                return self._failed(
                    request,
                    response.provider_id,
                    AdapterFailureCode.MALFORMED_OUTPUT,
                    str(exc),
                    raw_output=response.content,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
            invocation = ModelInvocation(
                request=request,
                provider_id=response.provider_id,
                model_id=response.model_id,
                raw_output=response.content,
                proposal=proposal,
                status="accepted",
                failure_code=None,
                failure_message=None,
                output_hash=_digest(response.content),
                failure_envelope_hash=None,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
            return ModelAdapterResult(invocation, proposal)
        except TimeoutError as exc:
            return self._failed(
                request, provider_id, AdapterFailureCode.TIMEOUT, str(exc) or "provider timed out"
            )
        except ModelProviderError as exc:
            return self._failed(
                request, provider_id, AdapterFailureCode.PROVIDER_ERROR, str(exc)
            )
        except Exception as exc:
            return self._failed(
                request,
                provider_id,
                AdapterFailureCode.PROVIDER_ERROR,
                f"{type(exc).__name__}: provider call failed",
            )

    def _failed(
        self,
        request: ModelRequest,
        provider_id: str,
        code: AdapterFailureCode,
        message: str,
        *,
        raw_output: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> ModelAdapterResult:
        normalized_message = message or code.value
        output_hash = _digest(raw_output) if raw_output is not None else None
        invocation = ModelInvocation(
            request=request,
            provider_id=provider_id if provider_id != "unknown_provider" else "unknown",
            model_id=request.model_id,
            raw_output=raw_output,
            proposal=None,
            status="failed",
            failure_code=code,
            failure_message=normalized_message,
            output_hash=output_hash,
            failure_envelope_hash=_failure_envelope_hash(
                request, code, normalized_message, output_hash
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return ModelAdapterResult(invocation, None, code, normalized_message)

    @staticmethod
    def _parse_proposal(request: ModelRequest, content: str) -> ModelProposal:
        if not isinstance(content, str) or not content.strip():
            raise ModelAdapterValidationError("provider output is empty")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ModelAdapterValidationError("provider output is not JSON") from exc
        if not isinstance(payload, dict):
            raise ModelAdapterValidationError("provider output must be a JSON object")
        if _FORBIDDEN_OUTPUT_KEYS.intersection(payload):
            raise ModelAdapterValidationError(
                "provider output contains an authority-bearing field"
            )
        if set(payload) != _PROPOSAL_KEYS:
            raise ModelAdapterValidationError(
                "provider output must contain exactly proposal, reasoning, and route_hint"
            )
        route_hint = payload["route_hint"]
        if route_hint is not None:
            if not isinstance(route_hint, str) or not route_hint.strip():
                raise ModelAdapterValidationError("route_hint must be null or text")
            if route_hint not in request.context.allowed_route_ids:
                raise ModelAdapterValidationError(
                    "route_hint is outside the caller-provided route boundary"
                )
        return ModelProposal(
            proposal_id=f"{request.request_id}-proposal",
            request_id=request.request_id,
            objective_id=request.context.objective_id,
            proposal=payload["proposal"],
            reasoning=payload["reasoning"],
            route_hint=route_hint,
        )


@dataclass(frozen=True)
class StructuralReplay:
    """Replay result for deterministic envelope checks, never a new generation."""

    request_hash: str
    output_hash: str | None
    status: str
    proposal: ModelProposal | None
    generation_replayed: bool = False
    structure_replayed: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize the replay check without implying a second provider call."""

        return {
            "request_hash": self.request_hash,
            "output_hash": self.output_hash,
            "status": self.status,
            "proposal": self.proposal.to_dict() if self.proposal else None,
            "generation_replayed": self.generation_replayed,
            "structure_replayed": self.structure_replayed,
        }


def replay_model_invocation(invocation: ModelInvocation) -> StructuralReplay:
    """Replay hashes and typed shape without calling a provider."""

    if not isinstance(invocation, ModelInvocation):
        raise ModelAdapterValidationError("replay requires a ModelInvocation")
    if invocation.request.input_hash != invocation.input_hash:
        raise ModelAdapterValidationError("request input hash changed")
    if invocation.status == "accepted":
        if invocation.raw_output is None or _digest(invocation.raw_output) != invocation.output_hash:
            raise ModelAdapterValidationError("recorded output hash changed")
        parsed = ModelAdapter._parse_proposal(invocation.request, invocation.raw_output)
        if parsed != invocation.proposal:
            raise ModelAdapterValidationError("recorded proposal shape changed")
    else:
        if invocation.raw_output is None and invocation.output_hash is not None:
            raise ModelAdapterValidationError("failed replay has an absent-output hash")
        if (
            invocation.raw_output is not None
            and _digest(invocation.raw_output) != invocation.output_hash
        ):
            raise ModelAdapterValidationError("failed replay output hash changed")
        if invocation.failure_code is None or not invocation.failure_message:
            raise ModelAdapterValidationError("failed replay disposition is invalid")
        if not isinstance(invocation.failure_code, AdapterFailureCode):
            raise ModelAdapterValidationError("failed replay code is invalid")
        if invocation.failure_envelope_hash != _failure_envelope_hash(
            invocation.request,
            invocation.failure_code,
            invocation.failure_message,
            invocation.output_hash,
        ):
            raise ModelAdapterValidationError("failed replay disposition changed")
        parsed = None
    return StructuralReplay(
        request_hash=invocation.input_hash,
        output_hash=invocation.output_hash,
        status=invocation.status,
        proposal=parsed,
    )
