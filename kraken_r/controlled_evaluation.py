"""Controlled held-out evaluation for the Round 7 proposal adapter.

The three conditions differ only in organizational context:

* ``base_model`` receives a masked route slot;
* ``kraken_mediated`` receives the current bounded candidate selection;
* ``kraken_ablated`` receives a reset topology selection.

The model, prompt template, temperature, output budget, provider, and task set
are held constant.  The evaluator scores proposals independently against a
held-out task label; it never creates evidence, settlement, learning, or
execution records from a model response.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping

from .contracts import Authority, Objective
from .cycle import CycleMode, run_constitutional_cycle
from .llm_adapter import (
    CandidateModelContext,
    ModelAdapter,
    ModelAdapterResult,
    ModelInvocation,
    ModelProvider,
    StructuralReplay,
    replay_model_invocation,
)
from .plastic_routing import RouteSelection, RouteTopology, select_candidate_route


class EvaluationValidationError(ValueError):
    """Raised when a controlled evaluation is not comparable."""


class EvaluationMode(str, Enum):
    BASE_MODEL = "base_model"
    KRAKEN_MEDIATED = "kraken_mediated"
    KRAKEN_ABLATED = "kraken_ablated"


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(char.isspace() for char in value):
        raise EvaluationValidationError(f"{name} must be a non-empty identifier")
    return value


@dataclass(frozen=True)
class HeldOutTask:
    """A task whose expected label is owned by the evaluator, not the model."""

    task_id: str
    objective_id: str
    prompt: str
    expected_route_id: str
    held_out: bool = True

    def __post_init__(self) -> None:
        _identifier(self.task_id, "task_id")
        _identifier(self.objective_id, "objective_id")
        _identifier(self.expected_route_id, "expected_route_id")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise EvaluationValidationError("task prompt must be non-empty")
        if self.held_out is not True:
            raise EvaluationValidationError("evaluation tasks must be held out")


@dataclass(frozen=True)
class EvaluationResult:
    task_id: str
    mode: EvaluationMode
    request_id: str
    selected_route_id: str | None
    model_route_hint: str | None
    passed: bool
    adapter_accepted: bool
    prompt_hash: str
    prompt_length: int
    context_hash: str
    configuration_fingerprint: str
    provider_configuration_fingerprint: str
    model_id: str
    provider_id: str
    max_tokens: int
    temperature: float
    reported_input_tokens: int | None
    reported_output_tokens: int | None
    failure_code: str | None
    failure_message: str | None
    invocation: ModelInvocation | None = None
    structural_replay: StructuralReplay | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "mode": self.mode.value,
            "request_id": self.request_id,
            "selected_route_id": self.selected_route_id,
            "model_route_hint": self.model_route_hint,
            "passed": self.passed,
            "adapter_accepted": self.adapter_accepted,
            "prompt_hash": self.prompt_hash,
            "prompt_length": self.prompt_length,
            "context_hash": self.context_hash,
            "configuration_fingerprint": self.configuration_fingerprint,
            "provider_configuration_fingerprint": self.provider_configuration_fingerprint,
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "reported_input_tokens": self.reported_input_tokens,
            "reported_output_tokens": self.reported_output_tokens,
            "failure_code": self.failure_code,
            "failure_message": self.failure_message,
            "invocation": self.invocation.to_dict() if self.invocation else None,
            "structural_replay": (
                self.structural_replay.to_dict() if self.structural_replay else None
            ),
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Descriptive comparison with an explicit no-claim default."""

    results: tuple[EvaluationResult, ...]
    control_errors: tuple[str, ...]
    mode_rates: Mapping[str, float]
    observed_deltas: Mapping[str, float]
    claim_status: str = "not_claimed"
    claim_reason: str = (
        "Model proposals are candidate declarations; this harness has no "
        "independent execution settlement."
    )

    def __post_init__(self) -> None:
        if self.claim_status != "not_claimed":
            raise EvaluationValidationError("Round 7 reports do not make performance claims")
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "control_errors", tuple(self.control_errors))
        object.__setattr__(self, "mode_rates", dict(self.mode_rates))
        object.__setattr__(self, "observed_deltas", dict(self.observed_deltas))

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [result.to_dict() for result in self.results],
            "control_errors": list(self.control_errors),
            "mode_rates": dict(self.mode_rates),
            "observed_deltas": dict(self.observed_deltas),
            "claim_status": self.claim_status,
            "claim_reason": self.claim_reason,
        }


class ControlledEvaluationHarness:
    """Run three comparable organizational conditions over held-out tasks."""

    ROUTING_CONTEXT_WIDTH = 192

    def __init__(
        self,
        provider: ModelProvider,
        *,
        model_id: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        prompt_template_version: str = "round-7-eval-v1",
    ) -> None:
        self.adapter = ModelAdapter(provider)
        self.provider = provider
        self.model_id = _identifier(model_id, "model_id")
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 4096:
            raise EvaluationValidationError("max_tokens must be from 1 through 4096")
        if not 0.0 <= float(temperature) <= 2.0:
            raise EvaluationValidationError("temperature must be from 0 through 2")
        self.max_tokens = max_tokens
        self.temperature = float(temperature)
        self.prompt_template_version = _identifier(
            prompt_template_version, "prompt_template_version"
        )

    def run(
        self,
        tasks: tuple[HeldOutTask, ...] | list[HeldOutTask],
        learned_topology: RouteTopology,
    ) -> EvaluationReport:
        tasks = tuple(tasks)
        if not tasks:
            raise EvaluationValidationError("evaluation requires at least one held-out task")
        if not isinstance(learned_topology, RouteTopology):
            raise EvaluationValidationError("evaluation requires a RouteTopology")
        if not all(isinstance(task, HeldOutTask) for task in tasks):
            raise EvaluationValidationError("tasks must be HeldOutTask records")
        if len({task.task_id for task in tasks}) != len(tasks):
            raise EvaluationValidationError("task ids must be unique")
        route_ids = tuple(route.route_id for route in learned_topology.routes)
        if any(task.expected_route_id not in route_ids for task in tasks):
            raise EvaluationValidationError("expected task route is absent from topology")

        reset_topology = learned_topology.reset()
        results: list[EvaluationResult] = []
        for task in tasks:
            objective = Objective(
                task.objective_id,
                task.prompt,
                provenance={"transaction_id": f"{task.task_id}-transaction"},
            )
            trace = run_constitutional_cycle(
                objective,
                mode=CycleMode.INSUFFICIENT_EVIDENCE,
                action_authority=Authority.KRAKEN_CANDIDATE,
            )
            for mode, topology in (
                (EvaluationMode.BASE_MODEL, None),
                (EvaluationMode.KRAKEN_MEDIATED, learned_topology),
                (EvaluationMode.KRAKEN_ABLATED, reset_topology),
            ):
                selection = self._selection_for(mode, topology, task, trace, route_ids)
                context = self._context_for(
                    trace, mode, selection, route_ids, task.task_id
                )
                prompt = self._prompt_for(task, mode, selection, route_ids)
                request_id = f"{task.task_id}-{mode.value}"
                result = self.adapter.invoke(
                    context,
                    request_id=request_id,
                    prompt=prompt,
                    model_id=self.model_id,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    prompt_template_version=self.prompt_template_version,
                )
                results.append(
                    self._result_for(
                        task, mode, request_id, selection, prompt, context, result
                    )
                )

        control_errors = self._control_errors(results, tasks)
        rates = {}
        for mode in EvaluationMode:
            bucket = [result for result in results if result.mode is mode]
            rates[mode.value] = (
                sum(result.passed for result in bucket) / len(bucket) if bucket else 0.0
            )
        observed_deltas = {
            "kraken_mediated_minus_base": (
                rates[EvaluationMode.KRAKEN_MEDIATED.value]
                - rates[EvaluationMode.BASE_MODEL.value]
            ),
            "kraken_mediated_minus_ablated": (
                rates[EvaluationMode.KRAKEN_MEDIATED.value]
                - rates[EvaluationMode.KRAKEN_ABLATED.value]
            ),
        }
        return EvaluationReport(
            tuple(results), tuple(control_errors), rates, observed_deltas
        )

    @classmethod
    def _selection_for(
        cls,
        mode: EvaluationMode,
        topology: RouteTopology | None,
        task: HeldOutTask,
        trace: Any,
        route_ids: tuple[str, ...],
    ) -> RouteSelection | None:
        if topology is None:
            return None
        return select_candidate_route(
            topology,
            "candidate-work",
            transaction_id=trace.transaction_id,
            objective_id=task.objective_id,
            task_state_id=trace.states[4].state_id,
            task_state_version=trace.states[4].version,
        )

    @classmethod
    def _routing_text(
        cls,
        mode: EvaluationMode,
        selection: RouteSelection | None,
        route_ids: tuple[str, ...],
    ) -> str:
        route_codes = {route_id: index for index, route_id in enumerate(route_ids)}
        if mode is EvaluationMode.BASE_MODEL or selection is None:
            selected_code = 999
            weights = {route_id: 0 for route_id in route_ids}
        else:
            selected_code = route_codes[selection.route_id]
            weights = {
                route_id: int(round(weight * 100))
                for route_id, weight in selection.candidate_scores
            }
        selected_text = " ".join(f"R{selected_code:03d}")
        scores_text = " ".join(
            character
            for route_id in route_ids
            for character in f"{weights.get(route_id, 0):03d}"
        )
        text = (
            f"candidate_route_code={selected_text}; "
            f"candidate_scores_code={scores_text}"
        )
        if len(text) > cls.ROUTING_CONTEXT_WIDTH:
            raise EvaluationValidationError("routing context exceeds fixed control width")
        return text.ljust(cls.ROUTING_CONTEXT_WIDTH, ".")

    @classmethod
    def _prompt_for(
        cls,
        task: HeldOutTask,
        mode: EvaluationMode,
        selection: RouteSelection | None,
        route_ids: tuple[str, ...],
    ) -> str:
        route_text = cls._routing_text(mode, selection, route_ids)
        route_codes = ", ".join(
            f"{' '.join(f'R{index:03d}')}={route_id}"
            for index, route_id in enumerate(route_ids)
        )
        return (
            "Round 7 controlled proposal task.\n"
            "Return one raw JSON object, without markdown fences, with exactly "
            "proposal, reasoning, and route_hint.\n"
            f"Route-code key (static across all conditions): {route_codes}; "
            "R 9 9 9=masked.\n"
            "route_hint must be exactly path-alpha, path-beta, or null.\n"
            "The route_hint is a suggestion only; do not report evidence, "
            "execution, settlement, success, confidence, action, goal, or learning.\n"
            f"Task: {task.prompt}\n"
            f"Organizational context: {route_text}\n"
        )

    @classmethod
    def _context_for(
        cls,
        trace: Any,
        mode: EvaluationMode,
        selection: RouteSelection | None,
        route_ids: tuple[str, ...],
        task_id: str,
    ) -> CandidateModelContext:
        scores = selection.candidate_scores if selection else tuple(
            (route_id, 0.50) for route_id in route_ids
        )
        return CandidateModelContext.from_cycle_trace(
            trace,
            context_id=f"evaluation-{mode.value}-{task_id}",
            allowed_route_ids=route_ids,
            route_scores=scores,
        )

    def _result_for(
        self,
        task: HeldOutTask,
        mode: EvaluationMode,
        request_id: str,
        selection: RouteSelection | None,
        prompt: str,
        context: CandidateModelContext,
        result: ModelAdapterResult,
    ) -> EvaluationResult:
        invocation = result.invocation
        proposal = result.proposal
        structural_replay = (
            replay_model_invocation(invocation) if invocation is not None else None
        )
        return EvaluationResult(
            task_id=task.task_id,
            mode=mode,
            request_id=request_id,
            selected_route_id=selection.route_id if selection else None,
            model_route_hint=proposal.route_hint if proposal else None,
            passed=bool(proposal and proposal.route_hint == task.expected_route_id),
            adapter_accepted=result.accepted,
            prompt_hash=_digest(prompt),
            prompt_length=len(prompt),
            context_hash=_digest(context.to_dict()),
            configuration_fingerprint=(
                invocation.request.configuration_fingerprint if invocation else ""
            ),
            provider_configuration_fingerprint=getattr(
                self.provider,
                "configuration_fingerprint",
                _digest({"provider_id": self.provider.provider_id}),
            ),
            model_id=self.model_id,
            provider_id=invocation.provider_id if invocation else self.provider.provider_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            reported_input_tokens=invocation.input_tokens if invocation else None,
            reported_output_tokens=invocation.output_tokens if invocation else None,
            failure_code=result.failure_code.value if result.failure_code else None,
            failure_message=result.failure_message,
            invocation=invocation,
            structural_replay=structural_replay,
        )

    def _control_errors(
        self, results: list[EvaluationResult], tasks: tuple[HeldOutTask, ...]
    ) -> list[str]:
        errors: list[str] = []
        expected = len(tasks)
        for mode in EvaluationMode:
            count = sum(result.mode is mode for result in results)
            if count != expected:
                errors.append(f"{mode.value} has {count} results; expected {expected}")
        for task in tasks:
            bucket = [result for result in results if result.task_id == task.task_id]
            if not bucket:
                continue
            if any(result.invocation is None for result in bucket):
                errors.append(f"{task.task_id} is missing an immutable invocation envelope")
            if any(result.structural_replay is None for result in bucket):
                errors.append(f"{task.task_id} is missing a structural replay record")
            if any(
                result.structural_replay is not None
                and (
                    result.structural_replay.generation_replayed
                    or not result.structural_replay.structure_replayed
                )
                for result in bucket
            ):
                errors.append(f"{task.task_id} replay repeated generation or failed structure")
            if any(result.model_id != result.invocation.model_id for result in bucket if result.invocation):
                errors.append(f"{task.task_id} has model provenance different from request")
            if any(
                result.provider_id != result.invocation.provider_id
                for result in bucket
                if result.invocation
            ):
                errors.append(f"{task.task_id} has provider provenance different from request")
            if any(result.model_id != self.model_id for result in bucket):
                errors.append(f"{task.task_id} changed injected model identity")
            if any(result.provider_id != self.provider.provider_id for result in bucket):
                errors.append(f"{task.task_id} changed injected provider identity")
            if any(result.max_tokens != self.max_tokens for result in bucket):
                errors.append(f"{task.task_id} changed output-token budget")
            if len({result.model_id for result in bucket}) != 1:
                errors.append(f"{task.task_id} changed model_id across conditions")
            if len({result.max_tokens for result in bucket}) != 1:
                errors.append(f"{task.task_id} changed max_tokens across conditions")
            if len({result.temperature for result in bucket}) != 1:
                errors.append(f"{task.task_id} changed temperature across conditions")
            if len({result.provider_id for result in bucket}) != 1:
                errors.append(f"{task.task_id} changed provider identity across conditions")
            if len({result.configuration_fingerprint for result in bucket}) != 1:
                errors.append(f"{task.task_id} changed model configuration across conditions")
            if len({result.provider_configuration_fingerprint for result in bucket}) != 1:
                errors.append(f"{task.task_id} changed provider configuration across conditions")
            if len({result.prompt_length for result in bucket}) != 1:
                errors.append(f"{task.task_id} changed normalized prompt length across conditions")
            reported_input_values = [
                result.reported_input_tokens for result in bucket
            ]
            reported_input_tokens = {
                value for value in reported_input_values if value is not None
            }
            if reported_input_tokens and any(
                value is None for value in reported_input_values
            ):
                errors.append(
                    f"{task.task_id} has mixed provider-reported input-token availability"
                )
            elif reported_input_tokens and len(reported_input_tokens) != 1:
                errors.append(f"{task.task_id} changed provider-reported input tokens")
        return errors