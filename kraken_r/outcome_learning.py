"""Bounded real-work prediction, outcome comparison, and competence gating.

This module is an orchestration boundary over existing candidate cognition,
grounded execution, constitutional settlement, and route-learning records.  It
does not execute work, call providers, retrieve material, persist receipts, or
own an adaptive reducer.  A prediction is sealed from a validated immutable
``ProcessingTrace`` before a separately verified execution may be attached.
Only a causally qualified episode may be delegated to the existing
``apply_settlement_learning`` reducer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .contracts import Authority, TaskState
from .cognition_kernel import ProcessingOperation, ProcessingTrace, replay_processing_trace
from .grounded_execution import (
    EpistemicOutcomeClass,
    GroundedExecutionError,
    GroundedExecutionRecord,
    GroundedExecutionRequest,
    GroundedExecutionVerifier,
    TrustedExecutorIdentity,
    VerifiedGroundedExecution,
    task_state_from_dict,
)


MAX_CONTRIBUTIONS = 16
MAX_CRITERIA = 16
MAX_REASON_CODES = 16
MAX_REPLAY_EPISODES = 64
MAX_RECORD_BYTES = 64_000
MAX_ID_LENGTH = 256
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class OutcomeLearningError(ValueError):
    """Raised when a bounded outcome-learning contract fails closed."""


class ContributionKind(str, Enum):
    """The only contribution classes recognized by the learning gate."""

    SEMANTIC_MODEL = "semantic_model"
    RECOGNITION_MEMORY = "recognition_memory"
    EXTERNAL_RETRIEVAL = "external_retrieval"
    DETERMINISTIC_TOOL = "deterministic_tool"
    HUMAN_INPUT = "human_input"


class OutcomeKind(str, Enum):
    """Independent grounded outcome classes used by the comparison reducer."""

    SUCCESS = "success"
    FAILURE = "failure"
    CONTRADICTION = "contradiction"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class ErrorLocation(str, Enum):
    """Bounded discrepancy localization, never a caller-selected credit label."""

    NONE = "none"
    SEMANTIC_MODEL = "semantic_model"
    RECOGNITION_MEMORY = "recognition_memory"
    EXTERNAL_RETRIEVAL = "external_retrieval"
    DETERMINISTIC_CALCULATION = "deterministic_calculation"
    HUMAN_INPUT = "human_input"
    FACTUAL_ASSUMPTION = "factual_assumption"
    REGIME_APPLICABILITY = "regime_applicability"
    EXECUTION = "execution"
    INSUFFICIENT_GROUNDING = "insufficient_grounding"
    UNSUPPORTED_CAUSALITY = "unsupported_causality"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _identifier(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(character.isspace() for character in value)
        or len(value) > MAX_ID_LENGTH
    ):
        raise OutcomeLearningError(
            f"{name} must be a bounded non-empty identifier without whitespace"
        )
    return value


def _text(value: Any, name: str, maximum: int = 4_096) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise OutcomeLearningError(f"{name} must be bounded non-empty text")
    return value


def _digest_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise OutcomeLearningError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _bounded_ids(values: Iterable[str], name: str, limit: int) -> tuple[str, ...]:
    try:
        result = tuple(_identifier(value, name) for value in values)
    except TypeError as exc:
        raise OutcomeLearningError(f"{name} must be iterable") from exc
    if len(result) > limit or len(set(result)) != len(result):
        raise OutcomeLearningError(f"{name} is duplicated or exceeds its bound")
    return result


def _bounded_texts(values: Iterable[str], name: str, limit: int) -> tuple[str, ...]:
    try:
        result = tuple(_text(value, name, 2_048) for value in values)
    except TypeError as exc:
        raise OutcomeLearningError(f"{name} must be iterable") from exc
    if len(result) > limit or len(set(result)) != len(result):
        raise OutcomeLearningError(f"{name} is duplicated or exceeds its bound")
    return result


def _bounded_mapping(value: Mapping[str, Any], name: str) -> MappingProxyType:
    if not isinstance(value, Mapping):
        raise OutcomeLearningError(f"{name} must be a mapping")
    normalized = _freeze(dict(value))
    raw = json.dumps(_jsonable(normalized), sort_keys=True, separators=(",", ":"))
    if len(raw.encode("utf-8")) > MAX_RECORD_BYTES:
        raise OutcomeLearningError(f"{name} exceeds its byte bound")
    return normalized


@dataclass(frozen=True)
class ContributionRecord:
    """One hash-bound input lineage item used by a prediction commitment."""

    contribution_id: str
    kind: ContributionKind
    source_ids: tuple[str, ...]
    source_hash: str
    role: str = "supporting"
    relevant: bool = True
    correction: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)
    contribution_hash: str = ""

    def __post_init__(self) -> None:
        _identifier(self.contribution_id, "contribution_id")
        object.__setattr__(self, "kind", ContributionKind(self.kind))
        object.__setattr__(
            self, "source_ids", _bounded_ids(self.source_ids, "source_ids", 32)
        )
        _digest_value(self.source_hash, "source_hash")
        if self.role not in {"supporting", "decisive", "contextual"}:
            raise OutcomeLearningError("contribution role is unsupported")
        if not isinstance(self.relevant, bool) or not isinstance(self.correction, bool):
            raise OutcomeLearningError("contribution relevance flags must be boolean")
        object.__setattr__(
            self, "provenance", _bounded_mapping(self.provenance, "provenance")
        )
        expected = _digest(self.to_dict(include_hash=False))
        if self.contribution_hash:
            _digest_value(self.contribution_hash, "contribution_hash")
            if self.contribution_hash != expected:
                raise OutcomeLearningError("contribution hash does not match its fields")
        else:
            object.__setattr__(self, "contribution_hash", expected)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "contribution_id": self.contribution_id,
            "kind": self.kind.value,
            "source_ids": list(self.source_ids),
            "source_hash": self.source_hash,
            "role": self.role,
            "relevant": self.relevant,
            "correction": self.correction,
            "provenance": _jsonable(self.provenance),
        }
        if include_hash:
            value["contribution_hash"] = self.contribution_hash
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContributionRecord":
        expected = {
            "contribution_id",
            "kind",
            "source_ids",
            "source_hash",
            "role",
            "relevant",
            "correction",
            "provenance",
            "contribution_hash",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise OutcomeLearningError("contribution record schema is invalid")
        return cls(
            value["contribution_id"],
            value["kind"],
            tuple(value["source_ids"]),
            value["source_hash"],
            value["role"],
            value["relevant"],
            value["correction"],
            value["provenance"],
            value["contribution_hash"],
        )


# Descriptive compatibility name: this is a record, not a second authority.
ContributionAttribution = ContributionRecord


def make_contribution(
    contribution_id: str,
    kind: ContributionKind | str,
    payload: Any,
    *,
    source_ids: Iterable[str] = (),
    role: str = "supporting",
    relevant: bool = True,
    correction: bool = False,
    provenance: Mapping[str, Any] | None = None,
) -> ContributionRecord:
    """Create a bounded supplemental lineage record.

    Supplemental records are deliberately never treated as independently
    verified reasoning.  Human, retrieval, and tool records conservatively
    withhold adaptive credit even when their payload is marked irrelevant.
    """

    return ContributionRecord(
        contribution_id,
        kind,
        tuple(source_ids),
        _digest(payload),
        role,
        relevant,
        correction,
        provenance or {},
    )


def _trace_contributions(trace: ProcessingTrace) -> tuple[ContributionRecord, ...]:
    """Derive contribution records from the validated trace, never from labels."""

    records: list[ContributionRecord] = []
    if trace.request is not None and trace.result is not None:
        operation_payload = {
            "request": trace.request.to_dict(),
            "result": trace.result.to_dict(),
            "context": trace.context.to_dict(),
            "trace_hash": trace.structural_hash,
        }
        if trace.request.semantic_job is not None:
            records.append(
                ContributionRecord(
                    f"{trace.request.request_id}-semantic-model",
                    ContributionKind.SEMANTIC_MODEL,
                    (trace.request.request_id, trace.result.result_id),
                    _digest(operation_payload),
                    provenance={
                        "trace_hash": trace.structural_hash,
                        "operation_input_hash": trace.request.input_hash,
                        "operation_output_hash": trace.result.output_hash,
                    },
                )
            )
        else:
            records.append(
                ContributionRecord(
                    f"{trace.request.request_id}-deterministic-tool",
                    ContributionKind.DETERMINISTIC_TOOL,
                    (trace.request.request_id, trace.result.result_id),
                    _digest(operation_payload),
                    provenance={
                        "trace_hash": trace.structural_hash,
                        "operation_input_hash": trace.request.input_hash,
                        "operation_output_hash": trace.result.output_hash,
                    },
                )
            )
    if trace.recognition_envelope or trace.recognition_proof:
        records.append(
            ContributionRecord(
                f"{trace.problem.problem_id}-recognition-memory",
                ContributionKind.RECOGNITION_MEMORY,
                tuple(
                    item
                    for item in (
                        trace.recognition_envelope.get("projection_id"),
                        trace.recognition_envelope.get("memory_set_id"),
                    )
                    if isinstance(item, str)
                ),
                _digest(
                    {
                        "envelope": trace.recognition_envelope,
                        "proof": trace.recognition_proof,
                        "trace_hash": trace.structural_hash,
                    }
                ),
                provenance={"trace_hash": trace.structural_hash},
            )
        )
    if trace.retrieval_observation or trace.candidate_revisions:
        records.append(
            ContributionRecord(
                f"{trace.problem.problem_id}-external-retrieval",
                ContributionKind.EXTERNAL_RETRIEVAL,
                tuple(
                    item.observation_id
                    for item in trace.candidate_revisions
                    if hasattr(item, "observation_id")
                ),
                _digest(
                    {
                        "observation": trace.retrieval_observation,
                        "revisions": trace.candidate_revisions,
                        "trace_hash": trace.structural_hash,
                    }
                ),
                provenance={"trace_hash": trace.structural_hash},
            )
        )
    if not records:
        raise OutcomeLearningError(
            "prediction requires a validated processing operation contribution"
        )
    return tuple(records)


def _cognitive_input_digest(trace: ProcessingTrace) -> str:
    return _digest(
        {
            "problem_hash": trace.problem.graph_hash,
            "projection_hash": trace.projection.topology_read_hash,
            "decision": trace.decision.to_dict(),
            "request": trace.request.to_dict() if trace.request is not None else None,
            "result": trace.result.to_dict() if trace.result is not None else None,
            "context": trace.context.to_dict(),
            "recognition_envelope": trace.recognition_envelope,
            "recognition_proof": trace.recognition_proof,
            "evidence_needs": trace.evidence_needs,
            "candidate_revisions": trace.candidate_revisions,
        }
    )


@dataclass(frozen=True)
class PredictionCommitment:
    """Immutable expected-result commitment created before outcome binding."""

    commitment_id: str
    episode_id: str
    transaction_id: str
    objective_id: str
    task_state_id: str
    task_state_version: int
    route_id: str
    request_id: str
    execution_request: GroundedExecutionRequest
    expected_outcome: str
    success_criteria: tuple[str, ...]
    cognitive_trace_hash: str
    cognitive_input_hash: str
    planned_work_hash: str
    precommitment_hash: str
    processing_trace: Mapping[str, Any]
    contributions: tuple[ContributionRecord, ...]
    candidate_only: bool = True
    commitment_hash: str = ""

    def __post_init__(self) -> None:
        for value, name in (
            (self.commitment_id, "commitment_id"),
            (self.episode_id, "episode_id"),
            (self.transaction_id, "transaction_id"),
            (self.objective_id, "objective_id"),
            (self.task_state_id, "task_state_id"),
            (self.route_id, "route_id"),
            (self.request_id, "request_id"),
        ):
            _identifier(value, name)
        if not isinstance(self.execution_request, GroundedExecutionRequest):
            raise OutcomeLearningError(
                "prediction must seal the exact grounded execution request"
            )
        if (
            self.execution_request.transaction_id != self.transaction_id
            or self.execution_request.objective_id != self.objective_id
            or self.execution_request.task_state_id != self.task_state_id
            or self.execution_request.task_state_version != self.task_state_version
        ):
            raise OutcomeLearningError(
                "sealed execution request crosses the prediction task boundary"
            )
        if (
            isinstance(self.task_state_version, bool)
            or not isinstance(self.task_state_version, int)
            or self.task_state_version < 1
        ):
            raise OutcomeLearningError("task_state_version must be positive")
        if self.expected_outcome not in {"success", "failure"}:
            raise OutcomeLearningError("expected_outcome must be success or failure")
        object.__setattr__(
            self,
            "success_criteria",
            _bounded_texts(self.success_criteria, "success_criteria", MAX_CRITERIA),
        )
        if not self.success_criteria:
            raise OutcomeLearningError("success_criteria cannot be empty")
        _digest_value(self.cognitive_trace_hash, "cognitive_trace_hash")
        _digest_value(self.cognitive_input_hash, "cognitive_input_hash")
        _digest_value(self.planned_work_hash, "planned_work_hash")
        _digest_value(self.precommitment_hash, "precommitment_hash")
        try:
            replayed = replay_processing_trace(self.processing_trace)
        except Exception as exc:
            raise OutcomeLearningError("commitment cognition record is not replayable") from exc
        if (
            replayed.structural_hash != self.cognitive_trace_hash
            or _cognitive_input_digest(replayed) != self.cognitive_input_hash
            or replayed.request is None
            or replayed.request.request_id != self.request_id
            or self.route_id not in {item[0] for item in replayed.projection.route_weights}
        ):
            raise OutcomeLearningError("commitment cognition binding is invalid")
        object.__setattr__(
            self,
            "processing_trace",
            _bounded_mapping(self.processing_trace, "processing_trace"),
        )
        contributions = tuple(self.contributions)
        if not 1 <= len(contributions) <= MAX_CONTRIBUTIONS or not all(
            isinstance(item, ContributionRecord) for item in contributions
        ):
            raise OutcomeLearningError("contributions are invalid or unbounded")
        if len({item.contribution_id for item in contributions}) != len(contributions):
            raise OutcomeLearningError("contribution identities must be unique")
        derived = _trace_contributions(replayed)
        if contributions[: len(derived)] != derived:
            raise OutcomeLearningError("commitment contribution lineage changed")
        expected_work_hash = _planned_work_hash(self.execution_request)
        expected_precommitment_hash = _precommitment_hash(
            commitment_id=self.commitment_id,
            episode_id=self.episode_id,
            transaction_id=self.transaction_id,
            objective_id=self.objective_id,
            task_state_id=self.task_state_id,
            task_state_version=self.task_state_version,
            route_id=self.route_id,
            request_id=self.request_id,
            expected_outcome=self.expected_outcome,
            success_criteria=self.success_criteria,
            cognitive_trace_hash=self.cognitive_trace_hash,
            cognitive_input_hash=self.cognitive_input_hash,
            cognitive_output_hash=replayed.result.output_hash,
            planned_work_hash=expected_work_hash,
            contributions=contributions,
        )
        expected_target = _prediction_target(
            self.commitment_id,
            expected_precommitment_hash,
            self.cognitive_input_hash,
            self.success_criteria,
            expected_work_hash,
            self.expected_outcome,
        )
        if (
            self.planned_work_hash != expected_work_hash
            or self.precommitment_hash != expected_precommitment_hash
            or self.execution_request.causal_parent_record_id != self.commitment_id
            or self.execution_request.causal_target != expected_target
            or not _action_has_cognition_binding(
                self.execution_request,
                self.commitment_id,
                self.cognitive_trace_hash,
                replayed.result.output_hash,
                self.route_id,
            )
        ):
            raise OutcomeLearningError(
                "execution request lacks signed pre-outcome cognitive lineage"
            )
        object.__setattr__(self, "contributions", contributions)
        if self.candidate_only is not True:
            raise OutcomeLearningError("prediction commitment must remain candidate-only")
        expected = _digest(self.to_dict(include_hash=False))
        if self.commitment_hash:
            _digest_value(self.commitment_hash, "commitment_hash")
            if self.commitment_hash != expected:
                raise OutcomeLearningError("prediction commitment hash is invalid")
        else:
            object.__setattr__(self, "commitment_hash", expected)

    @property
    def prediction_hash(self) -> str:
        return self.commitment_hash

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "commitment_id": self.commitment_id,
            "episode_id": self.episode_id,
            "transaction_id": self.transaction_id,
            "objective_id": self.objective_id,
            "task_state_id": self.task_state_id,
            "task_state_version": self.task_state_version,
            "route_id": self.route_id,
            "request_id": self.request_id,
            "execution_request": self.execution_request.to_dict(),
            "expected_outcome": self.expected_outcome,
            "success_criteria": list(self.success_criteria),
            "cognitive_trace_hash": self.cognitive_trace_hash,
            "cognitive_input_hash": self.cognitive_input_hash,
            "planned_work_hash": self.planned_work_hash,
            "precommitment_hash": self.precommitment_hash,
            "processing_trace": _jsonable(self.processing_trace),
            "contributions": [item.to_dict() for item in self.contributions],
            "candidate_only": self.candidate_only,
        }
        if include_hash:
            value["commitment_hash"] = self.commitment_hash
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PredictionCommitment":
        expected = {
            "commitment_id",
            "episode_id",
            "transaction_id",
            "objective_id",
            "task_state_id",
            "task_state_version",
            "route_id",
            "request_id",
            "execution_request",
            "expected_outcome",
            "success_criteria",
            "cognitive_trace_hash",
            "cognitive_input_hash",
            "planned_work_hash",
            "precommitment_hash",
            "processing_trace",
            "contributions",
            "candidate_only",
            "commitment_hash",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise OutcomeLearningError("prediction commitment schema is invalid")
        return cls(
            value["commitment_id"],
            value["episode_id"],
            value["transaction_id"],
            value["objective_id"],
            value["task_state_id"],
            value["task_state_version"],
            value["route_id"],
            value["request_id"],
            GroundedExecutionRequest.from_dict(value["execution_request"]),
            value["expected_outcome"],
            tuple(value["success_criteria"]),
            value["cognitive_trace_hash"],
            value["cognitive_input_hash"],
            value["planned_work_hash"],
            value["precommitment_hash"],
            value["processing_trace"],
            tuple(ContributionRecord.from_dict(item) for item in value["contributions"]),
            value["candidate_only"],
            value["commitment_hash"],
        )


def _planned_work_hash(request: GroundedExecutionRequest) -> str:
    return _digest(
        {
            "request_id": request.request_id,
            "action": request.action.to_dict(),
            "files": dict(request.files),
            "test_paths": list(request.test_paths),
            "limits": request.limits.to_dict(),
        }
    )


def _precommitment_hash(
    *,
    commitment_id: str,
    episode_id: str,
    transaction_id: str,
    objective_id: str,
    task_state_id: str,
    task_state_version: int,
    route_id: str,
    request_id: str,
    expected_outcome: str,
    success_criteria: tuple[str, ...],
    cognitive_trace_hash: str,
    cognitive_input_hash: str,
    cognitive_output_hash: str,
    planned_work_hash: str,
    contributions: tuple[ContributionRecord, ...],
) -> str:
    return _digest(
        {
            "commitment_id": commitment_id,
            "episode_id": episode_id,
            "transaction_id": transaction_id,
            "objective_id": objective_id,
            "task_state_id": task_state_id,
            "task_state_version": task_state_version,
            "route_id": route_id,
            "cognitive_request_id": request_id,
            "expected_outcome": expected_outcome,
            "success_criteria": success_criteria,
            "cognitive_trace_hash": cognitive_trace_hash,
            "cognitive_input_hash": cognitive_input_hash,
            "cognitive_output_hash": cognitive_output_hash,
            "planned_work_hash": planned_work_hash,
            "contributions": [item.to_dict() for item in contributions],
        }
    )


def _prediction_target(
    commitment_id: str,
    precommitment_hash: str,
    cognitive_input_hash: str,
    success_criteria: tuple[str, ...],
    planned_work_hash: str,
    expected_outcome: str,
) -> Mapping[str, str]:
    return {
        "target_record_id": commitment_id,
        "target_settlement_id": f"{commitment_id}-{expected_outcome}",
        "target_update_lineage_id": cognitive_input_hash,
        "target_audit_hash": precommitment_hash,
        "target_criterion_binding": _digest(
            {
                "success_criteria": success_criteria,
                "planned_work_hash": planned_work_hash,
            }
        ),
    }


def _action_has_cognition_binding(
    request: GroundedExecutionRequest,
    commitment_id: str,
    cognitive_trace_hash: str,
    cognitive_output_hash: str,
    route_id: str,
) -> bool:
    parameters = request.action.parameters
    return all(
        parameters.get(key) == value
        for key, value in {
            "prediction_commitment_id": commitment_id,
            "cognitive_trace_hash": cognitive_trace_hash,
            "cognitive_output_hash": cognitive_output_hash,
            "cognitive_route_id": route_id,
        }.items()
    )


def seal_prediction_request(
    trace: ProcessingTrace | Mapping[str, Any],
    request: GroundedExecutionRequest,
    *,
    commitment_id: str,
    expected_outcome: str,
    success_criteria: Iterable[str],
    episode_id: str | None = None,
    contributions: Iterable[ContributionRecord] = (),
    route_id: str | None = None,
) -> GroundedExecutionRequest:
    """Bind cognition and expected criteria into an unexecuted grounded request."""

    validated = replay_processing_trace(trace)
    if validated.request is None or validated.result is None:
        raise OutcomeLearningError("prediction seal requires an executed cognition operation")
    criteria = _bounded_texts(success_criteria, "success_criteria", MAX_CRITERIA)
    derived = _trace_contributions(validated)
    extras = tuple(contributions)
    if any(
        not isinstance(item, ContributionRecord)
        or item.kind in {
            ContributionKind.SEMANTIC_MODEL,
            ContributionKind.RECOGNITION_MEMORY,
        }
        for item in extras
    ):
        raise OutcomeLearningError("supplemental prediction lineage is invalid")
    all_contributions = derived + extras
    selected_route_id = route_id or sorted(
        validated.projection.route_weights, key=lambda item: (-item[1], item[0])
    )[0][0]
    action = replace(
        request.action,
        parameters={
            **dict(request.action.parameters),
            "prediction_commitment_id": commitment_id,
            "cognitive_trace_hash": validated.structural_hash,
            "cognitive_output_hash": validated.result.output_hash,
            "cognitive_route_id": selected_route_id,
        },
    )
    draft = replace(
        request,
        action=action,
        causal_parent_record_id=None,
        causal_target=None,
    )
    work_hash = _planned_work_hash(draft)
    input_hash = _cognitive_input_digest(validated)
    precommitment_hash = _precommitment_hash(
        commitment_id=commitment_id,
        episode_id=episode_id or commitment_id,
        transaction_id=draft.transaction_id,
        objective_id=draft.objective_id,
        task_state_id=draft.task_state_id,
        task_state_version=draft.task_state_version,
        route_id=selected_route_id,
        request_id=validated.request.request_id,
        expected_outcome=expected_outcome,
        success_criteria=criteria,
        cognitive_trace_hash=validated.structural_hash,
        cognitive_input_hash=input_hash,
        cognitive_output_hash=validated.result.output_hash,
        planned_work_hash=work_hash,
        contributions=all_contributions,
    )
    return replace(
        draft,
        causal_parent_record_id=commitment_id,
        causal_target=_prediction_target(
            commitment_id,
            precommitment_hash,
            input_hash,
            criteria,
            work_hash,
            expected_outcome,
        ),
    )


def commit_prediction(
    trace: ProcessingTrace | Mapping[str, Any],
    *,
    commitment_id: str,
    grounded_request: GroundedExecutionRequest,
    expected_outcome: str,
    success_criteria: Iterable[str],
    episode_id: str | None = None,
    contributions: Iterable[ContributionRecord] = (),
    authorized_state: TaskState | None = None,
    transaction_id: str | None = None,
    route_id: str | None = None,
) -> PredictionCommitment:
    """Seal one expected result from a validated cognition trace."""

    try:
        validated = replay_processing_trace(trace)
    except Exception as exc:
        raise OutcomeLearningError(
            "prediction commitment requires a valid replayable processing trace"
        ) from exc
    if validated.request is None or validated.result is None:
        raise OutcomeLearningError("prediction commitment requires an executed operation")
    criteria = _bounded_texts(success_criteria, "success_criteria", MAX_CRITERIA)
    if not isinstance(grounded_request, GroundedExecutionRequest):
        raise OutcomeLearningError(
            "prediction commitment requires the exact future execution request"
        )
    derived = _trace_contributions(validated)
    extras = tuple(contributions)
    if not all(isinstance(item, ContributionRecord) for item in extras):
        raise OutcomeLearningError("supplemental contributions are invalid")
    if any(
        item.kind in {ContributionKind.SEMANTIC_MODEL, ContributionKind.RECOGNITION_MEMORY}
        for item in extras
    ):
        raise OutcomeLearningError(
            "semantic and recognition lineage must be derived from the processing trace"
        )
    all_contributions = derived + extras
    if authorized_state is not None:
        if (
            not isinstance(authorized_state, TaskState)
            or authorized_state.phase != "authorized"
            or authorized_state.version != 5
            or authorized_state.authority is not Authority.KRAKEN_CANDIDATE
        ):
            raise OutcomeLearningError(
                "prediction task state must be the candidate authorized state"
            )
        objective_id = authorized_state.objective_id
        task_state_id = authorized_state.state_id
        task_state_version = authorized_state.version
    else:
        objective_id = validated.problem.original_task.task_id
        task_state_id = f"{validated.problem.problem_id}-processing-state"
        task_state_version = 5
    bound_transaction_id = transaction_id or str(
        validated.problem.original_task.provenance.get(
            "transaction_id", validated.problem.problem_id
        )
    )
    if (
        grounded_request.transaction_id != bound_transaction_id
        or grounded_request.objective_id != objective_id
        or grounded_request.task_state_id != task_state_id
        or grounded_request.task_state_version != task_state_version
    ):
        raise OutcomeLearningError(
            "future execution request does not match the committed task boundary"
        )
    projected_routes = tuple(validated.projection.route_weights)
    selected_route_id = route_id or sorted(
        projected_routes, key=lambda item: (-item[1], item[0])
    )[0][0]
    if selected_route_id not in {item[0] for item in projected_routes}:
        raise OutcomeLearningError("prediction route is absent from the cognition projection")
    cognitive_input_hash = _cognitive_input_digest(validated)
    planned_work_hash = _planned_work_hash(grounded_request)
    precommitment_hash = _precommitment_hash(
        commitment_id=commitment_id,
        episode_id=episode_id or commitment_id,
        transaction_id=bound_transaction_id,
        objective_id=objective_id,
        task_state_id=task_state_id,
        task_state_version=task_state_version,
        route_id=selected_route_id,
        request_id=validated.request.request_id,
        expected_outcome=expected_outcome,
        success_criteria=criteria,
        cognitive_trace_hash=validated.structural_hash,
        cognitive_input_hash=cognitive_input_hash,
        cognitive_output_hash=validated.result.output_hash,
        planned_work_hash=planned_work_hash,
        contributions=all_contributions,
    )
    return PredictionCommitment(
        commitment_id=commitment_id,
        episode_id=episode_id or commitment_id,
        transaction_id=bound_transaction_id,
        objective_id=objective_id,
        task_state_id=task_state_id,
        task_state_version=task_state_version,
        route_id=selected_route_id,
        request_id=validated.request.request_id,
        execution_request=grounded_request,
        expected_outcome=expected_outcome,
        success_criteria=criteria,
        cognitive_trace_hash=validated.structural_hash,
        cognitive_input_hash=cognitive_input_hash,
        planned_work_hash=planned_work_hash,
        precommitment_hash=precommitment_hash,
        processing_trace=validated.to_dict(),
        contributions=all_contributions,
    )


def _outcome_kind(verified: VerifiedGroundedExecution) -> OutcomeKind:
    observation = verified.record.observation
    epistemic = observation.epistemic_class
    if epistemic is EpistemicOutcomeClass.TASK_SUCCESS:
        return OutcomeKind.SUCCESS
    if epistemic is EpistemicOutcomeClass.TASK_FAILURE:
        if observation.tests_passed > 0 and observation.tests_failed > 0:
            return OutcomeKind.PARTIAL
        return OutcomeKind.FAILURE
    if epistemic is EpistemicOutcomeClass.CONTRADICTION:
        return OutcomeKind.CONTRADICTION
    return OutcomeKind.INSUFFICIENT


@dataclass(frozen=True)
class GroundedOutcome:
    """An independently verified, commitment-bound later observation."""

    outcome_id: str
    commitment_id: str
    commitment_hash: str
    verified_execution: VerifiedGroundedExecution
    request: GroundedExecutionRequest
    authorized_state: TaskState
    trusted_executor: TrustedExecutorIdentity
    outcome_kind: OutcomeKind = field(init=False)
    outcome_hash: str = ""

    def __post_init__(self) -> None:
        for value, name in (
            (self.outcome_id, "outcome_id"),
            (self.commitment_id, "commitment_id"),
        ):
            _identifier(value, name)
        _digest_value(self.commitment_hash, "commitment_hash")
        if not isinstance(self.verified_execution, VerifiedGroundedExecution):
            raise OutcomeLearningError("outcome requires verifier-issued execution")
        if not isinstance(self.request, GroundedExecutionRequest):
            raise OutcomeLearningError("outcome requires its sealed execution request")
        if not isinstance(self.authorized_state, TaskState):
            raise OutcomeLearningError("outcome requires its authorized task state")
        if not isinstance(self.trusted_executor, TrustedExecutorIdentity):
            raise OutcomeLearningError("outcome requires an external executor trust anchor")
        record = self.verified_execution.record
        if (
            record.request_id != self.request.request_id
            or record.transaction_id != self.request.transaction_id
            or record.objective_id != self.request.objective_id
            or record.task_state_id != self.request.task_state_id
            or record.task_state_version != self.request.task_state_version
        ):
            raise OutcomeLearningError("grounded outcome record/request identity disagrees")
        if (
            self.authorized_state.state_id != self.request.task_state_id
            or self.authorized_state.version != self.request.task_state_version
            or self.authorized_state.objective_id != self.request.objective_id
        ):
            raise OutcomeLearningError("grounded outcome state is stale or mismatched")
        try:
            verifier = GroundedExecutionVerifier(self.trusted_executor)
            verified = verifier.verify(
                record, request=self.request, authorized_state=self.authorized_state
            )
        except (GroundedExecutionError, ValueError) as exc:
            raise OutcomeLearningError(
                "grounded outcome is not independently verifiable"
            ) from exc
        if verified != self.verified_execution:
            raise OutcomeLearningError("grounded outcome verification changed")
        object.__setattr__(self, "outcome_kind", _outcome_kind(verified))
        expected = _digest(self.to_dict(include_hash=False))
        if self.outcome_hash:
            _digest_value(self.outcome_hash, "outcome_hash")
            if self.outcome_hash != expected:
                raise OutcomeLearningError("grounded outcome hash is invalid")
        else:
            object.__setattr__(self, "outcome_hash", expected)

    @property
    def record_hash(self) -> str:
        return self.verified_execution.record.record_hash

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "outcome_id": self.outcome_id,
            "commitment_id": self.commitment_id,
            "commitment_hash": self.commitment_hash,
            "verified_execution": {
                "record": self.verified_execution.record.to_dict(),
                "observed_outcome": self.verified_execution.observed_outcome,
                "verified": self.verified_execution.verified,
                "epistemic_class": self.verified_execution.epistemic_class.value,
            },
            "request": self.request.to_dict(),
            "authorized_state": self.authorized_state.to_dict(),
            "trusted_executor": self.trusted_executor.to_dict(),
            "outcome_kind": self.outcome_kind.value,
        }
        if include_hash:
            value["outcome_hash"] = self.outcome_hash
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GroundedOutcome":
        expected = {
            "outcome_id",
            "commitment_id",
            "commitment_hash",
            "verified_execution",
            "request",
            "authorized_state",
            "trusted_executor",
            "outcome_kind",
            "outcome_hash",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise OutcomeLearningError("grounded outcome schema is invalid")
        record_value = value["verified_execution"]["record"]
        record = GroundedExecutionRecord.from_dict(record_value)
        request = GroundedExecutionRequest.from_dict(value["request"])
        state = task_state_from_dict(value["authorized_state"])
        trusted = TrustedExecutorIdentity.from_dict(value["trusted_executor"])
        verified_value = value["verified_execution"]
        verified = VerifiedGroundedExecution(
            record,
            verified_value["observed_outcome"],
            verified_value["verified"],
            verified_value["epistemic_class"],
        )
        return cls(
            value["outcome_id"],
            value["commitment_id"],
            value["commitment_hash"],
            verified,
            request,
            state,
            trusted,
            value["outcome_hash"],
        )


def bind_grounded_outcome(
    commitment: PredictionCommitment,
    *,
    outcome_id: str,
    verified_execution: VerifiedGroundedExecution,
    request: GroundedExecutionRequest,
    authorized_state: TaskState,
    trusted_executor: TrustedExecutorIdentity,
) -> GroundedOutcome:
    """Bind an already independently verified execution to its prior seal."""

    if not isinstance(commitment, PredictionCommitment):
        raise OutcomeLearningError("outcome binding requires a prediction commitment")
    if (
        request.transaction_id != commitment.transaction_id
        or request.objective_id != commitment.objective_id
        or request.task_state_id != commitment.task_state_id
        or request.task_state_version != commitment.task_state_version
    ):
        raise OutcomeLearningError("outcome is outside the committed task boundary")
    if (
        request != commitment.execution_request
        or request.input_hash != commitment.execution_request.input_hash
        or request.to_dict() != commitment.execution_request.to_dict()
    ):
        raise OutcomeLearningError(
            "outcome request differs from the pre-outcome execution seal"
        )
    if verified_execution.record.request_id != request.request_id:
        raise OutcomeLearningError("outcome record does not match its sealed request")
    return GroundedOutcome(
        outcome_id,
        commitment.commitment_id,
        commitment.commitment_hash,
        verified_execution,
        request,
        authorized_state,
        trusted_executor,
    )


@dataclass(frozen=True)
class OutcomeComparison:
    """Deterministic comparison of the sealed prediction and grounded facts."""

    commitment_id: str
    commitment_hash: str
    outcome_id: str
    outcome_hash: str
    expected_outcome: str
    observed_outcome: OutcomeKind
    prediction_matches: bool
    error_locations: tuple[ErrorLocation, ...]
    comparison_hash: str = ""

    def __post_init__(self) -> None:
        _identifier(self.commitment_id, "comparison commitment_id")
        _digest_value(self.commitment_hash, "comparison commitment_hash")
        _identifier(self.outcome_id, "comparison outcome_id")
        _digest_value(self.outcome_hash, "comparison outcome_hash")
        if self.expected_outcome not in {"success", "failure"}:
            raise OutcomeLearningError("comparison expected outcome is invalid")
        object.__setattr__(self, "observed_outcome", OutcomeKind(self.observed_outcome))
        locations = tuple(ErrorLocation(item) for item in self.error_locations)
        if len(locations) > MAX_REASON_CODES or len(set(locations)) != len(locations):
            raise OutcomeLearningError("error locations are duplicated or unbounded")
        if not isinstance(self.prediction_matches, bool):
            raise OutcomeLearningError("prediction_matches must be boolean")
        object.__setattr__(self, "error_locations", locations)
        expected = _digest(self.to_dict(include_hash=False))
        if self.comparison_hash:
            _digest_value(self.comparison_hash, "comparison_hash")
            if self.comparison_hash != expected:
                raise OutcomeLearningError("comparison hash is invalid")
        else:
            object.__setattr__(self, "comparison_hash", expected)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "commitment_id": self.commitment_id,
            "commitment_hash": self.commitment_hash,
            "outcome_id": self.outcome_id,
            "outcome_hash": self.outcome_hash,
            "expected_outcome": self.expected_outcome,
            "observed_outcome": self.observed_outcome.value,
            "prediction_matches": self.prediction_matches,
            "error_locations": [item.value for item in self.error_locations],
        }
        if include_hash:
            value["comparison_hash"] = self.comparison_hash
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OutcomeComparison":
        expected = {
            "commitment_id",
            "commitment_hash",
            "outcome_id",
            "outcome_hash",
            "expected_outcome",
            "observed_outcome",
            "prediction_matches",
            "error_locations",
            "comparison_hash",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise OutcomeLearningError("comparison schema is invalid")
        return cls(
            value["commitment_id"],
            value["commitment_hash"],
            value["outcome_id"],
            value["outcome_hash"],
            value["expected_outcome"],
            value["observed_outcome"],
            value["prediction_matches"],
            tuple(value["error_locations"]),
            value["comparison_hash"],
        )


@dataclass(frozen=True)
class CompetenceAttribution:
    """Derived credit decision; it is not caller-authoritative attribution."""

    contribution_ids: tuple[str, ...]
    accepted_kinds: tuple[ContributionKind, ...]
    withheld_kinds: tuple[ContributionKind, ...]
    eligible: bool
    disposition: str
    reasons: tuple[str, ...]
    attribution_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "contribution_ids", _bounded_ids(self.contribution_ids, "contribution_ids", MAX_CONTRIBUTIONS)
        )
        accepted = tuple(ContributionKind(item) for item in self.accepted_kinds)
        withheld = tuple(ContributionKind(item) for item in self.withheld_kinds)
        object.__setattr__(self, "accepted_kinds", accepted)
        object.__setattr__(self, "withheld_kinds", withheld)
        object.__setattr__(
            self, "reasons", _bounded_texts(self.reasons, "attribution reasons", MAX_REASON_CODES)
        )
        if not isinstance(self.eligible, bool):
            raise OutcomeLearningError("attribution eligibility must be boolean")
        if self.disposition not in {"eligible", "withheld"}:
            raise OutcomeLearningError("attribution disposition is invalid")
        if self.disposition == "eligible" and not self.eligible:
            raise OutcomeLearningError("eligible disposition disagrees with eligibility")
        if self.disposition == "withheld" and self.eligible:
            raise OutcomeLearningError("withheld disposition disagrees with eligibility")
        expected = _digest(self.to_dict(include_hash=False))
        if self.attribution_hash:
            _digest_value(self.attribution_hash, "attribution_hash")
            if self.attribution_hash != expected:
                raise OutcomeLearningError("attribution hash is invalid")
        else:
            object.__setattr__(self, "attribution_hash", expected)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "contribution_ids": list(self.contribution_ids),
            "accepted_kinds": [item.value for item in self.accepted_kinds],
            "withheld_kinds": [item.value for item in self.withheld_kinds],
            "eligible": self.eligible,
            "disposition": self.disposition,
            "reasons": list(self.reasons),
        }
        if include_hash:
            value["attribution_hash"] = self.attribution_hash
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CompetenceAttribution":
        expected = {
            "contribution_ids",
            "accepted_kinds",
            "withheld_kinds",
            "eligible",
            "disposition",
            "reasons",
            "attribution_hash",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise OutcomeLearningError("attribution schema is invalid")
        return cls(
            tuple(value["contribution_ids"]),
            tuple(value["accepted_kinds"]),
            tuple(value["withheld_kinds"]),
            value["eligible"],
            value["disposition"],
            tuple(value["reasons"]),
            value["attribution_hash"],
        )


def _locations(
    comparison_kind: OutcomeKind, contributions: tuple[ContributionRecord, ...]
) -> tuple[ErrorLocation, ...]:
    if comparison_kind is OutcomeKind.SUCCESS:
        return ()
    if comparison_kind is OutcomeKind.INSUFFICIENT:
        return (ErrorLocation.INSUFFICIENT_GROUNDING,)
    if comparison_kind is OutcomeKind.PARTIAL:
        return (ErrorLocation.EXECUTION,)
    if comparison_kind is OutcomeKind.CONTRADICTION:
        return (ErrorLocation.FACTUAL_ASSUMPTION, ErrorLocation.EXECUTION)
    kinds = {item.kind for item in contributions}
    result: list[ErrorLocation] = []
    if ContributionKind.EXTERNAL_RETRIEVAL in kinds:
        result.append(ErrorLocation.EXTERNAL_RETRIEVAL)
    if ContributionKind.HUMAN_INPUT in kinds:
        result.append(ErrorLocation.HUMAN_INPUT)
    if ContributionKind.DETERMINISTIC_TOOL in kinds:
        result.append(ErrorLocation.DETERMINISTIC_CALCULATION)
    if ContributionKind.RECOGNITION_MEMORY in kinds:
        result.append(ErrorLocation.RECOGNITION_MEMORY)
    if not result:
        result.append(ErrorLocation.SEMANTIC_MODEL)
    return tuple(result)


def _derive_attribution(
    commitment: PredictionCommitment, outcome: GroundedOutcome, comparison: OutcomeComparison
) -> CompetenceAttribution:
    contributions = commitment.contributions
    kinds = tuple(dict.fromkeys(item.kind for item in contributions))
    blocked: list[str] = []
    withheld_kinds: list[ContributionKind] = []
    if comparison.observed_outcome not in {OutcomeKind.SUCCESS, OutcomeKind.FAILURE}:
        blocked.append(f"grounded outcome is {comparison.observed_outcome.value}")
    if ContributionKind.SEMANTIC_MODEL not in kinds:
        blocked.append("no validated semantic/model contribution")
    for item in contributions:
        if item.kind in {
            ContributionKind.EXTERNAL_RETRIEVAL,
            ContributionKind.DETERMINISTIC_TOOL,
            ContributionKind.HUMAN_INPUT,
        }:
            if item.kind not in withheld_kinds:
                withheld_kinds.append(item.kind)
            blocked.append(f"{item.kind.value} cannot earn competence credit")
        if item.correction:
            blocked.append("correction input cannot earn competence credit")
        if not item.relevant:
            blocked.append("irrelevant contribution cannot earn competence credit")
    eligible = not blocked
    accepted_kinds = tuple(
        kind
        for kind in kinds
        if kind in {
            ContributionKind.SEMANTIC_MODEL,
            ContributionKind.RECOGNITION_MEMORY,
        }
        and kind not in withheld_kinds
    )
    return CompetenceAttribution(
        tuple(item.contribution_id for item in contributions),
        accepted_kinds,
        tuple(withheld_kinds),
        eligible,
        "eligible" if eligible else "withheld",
        tuple(dict.fromkeys(blocked)),
    )


def _derive_comparison(
    commitment: PredictionCommitment, outcome: GroundedOutcome
) -> OutcomeComparison:
    observed = outcome.outcome_kind
    return OutcomeComparison(
        commitment.commitment_id,
        commitment.commitment_hash,
        outcome.outcome_id,
        outcome.outcome_hash,
        commitment.expected_outcome,
        observed,
        observed.value == commitment.expected_outcome,
        _locations(observed, commitment.contributions),
    )


@dataclass(frozen=True)
class GroundedLearningEpisode:
    """Complete immutable prediction/outcome/attribution learning record."""

    commitment: PredictionCommitment
    outcome: GroundedOutcome
    comparison: OutcomeComparison
    attribution: CompetenceAttribution
    adaptive_learning: Mapping[str, Any] | None = None
    candidate_only: bool = True
    episode_hash: str = ""

    def __post_init__(self) -> None:
        if not all(
            isinstance(item, item_type)
            for item, item_type in (
                (self.commitment, PredictionCommitment),
                (self.outcome, GroundedOutcome),
                (self.comparison, OutcomeComparison),
                (self.attribution, CompetenceAttribution),
            )
        ):
            raise OutcomeLearningError("episode records are invalid")
        if self.outcome.commitment_id != self.commitment.commitment_id:
            raise OutcomeLearningError("outcome is not bound to prediction commitment")
        if self.outcome.commitment_hash != self.commitment.commitment_hash:
            raise OutcomeLearningError("outcome commitment hash is stale")
        if (
            self.comparison.commitment_id != self.commitment.commitment_id
            or self.comparison.commitment_hash != self.commitment.commitment_hash
            or self.comparison.outcome_id != self.outcome.outcome_id
            or self.comparison.outcome_hash != self.outcome.outcome_hash
        ):
            raise OutcomeLearningError("comparison lineage is incomplete")
        expected_comparison = _derive_comparison(self.commitment, self.outcome)
        if self.comparison != expected_comparison:
            raise OutcomeLearningError(
                "comparison is not the deterministic result of commitment and outcome"
            )
        expected_attribution = _derive_attribution(
            self.commitment, self.outcome, expected_comparison
        )
        if self.attribution != expected_attribution:
            raise OutcomeLearningError(
                "attribution is not the deterministic outcome-learning gate"
            )
        if self.candidate_only is not True:
            raise OutcomeLearningError("learning episode must remain candidate-only")
        if self.adaptive_learning is not None:
            object.__setattr__(
                self,
                "adaptive_learning",
                _bounded_mapping(self.adaptive_learning, "adaptive_learning"),
            )
        expected = _digest(self.to_dict(include_hash=False))
        if self.episode_hash:
            _digest_value(self.episode_hash, "episode_hash")
            if self.episode_hash != expected:
                raise OutcomeLearningError("episode hash is invalid")
        else:
            object.__setattr__(self, "episode_hash", expected)

    @property
    def eligible_for_learning(self) -> bool:
        return self.attribution.eligible and self.adaptive_learning is not None

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "commitment": self.commitment.to_dict(),
            "outcome": self.outcome.to_dict(),
            "comparison": self.comparison.to_dict(),
            "attribution": self.attribution.to_dict(),
            "adaptive_learning": _jsonable(self.adaptive_learning),
            "candidate_only": self.candidate_only,
        }
        if include_hash:
            value["episode_hash"] = self.episode_hash
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GroundedLearningEpisode":
        expected = {
            "commitment",
            "outcome",
            "comparison",
            "attribution",
            "adaptive_learning",
            "candidate_only",
            "episode_hash",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise OutcomeLearningError("learning episode schema is invalid")
        return cls(
            PredictionCommitment.from_dict(value["commitment"]),
            GroundedOutcome.from_dict(value["outcome"]),
            OutcomeComparison.from_dict(value["comparison"]),
            CompetenceAttribution.from_dict(value["attribution"]),
            value["adaptive_learning"],
            value["candidate_only"],
            value["episode_hash"],
        )


def evaluate_grounded_outcome(
    commitment: PredictionCommitment, outcome: GroundedOutcome
) -> GroundedLearningEpisode:
    """Compare a sealed prediction with an independently verified outcome."""

    if outcome.commitment_id != commitment.commitment_id:
        raise OutcomeLearningError("outcome belongs to a different prediction")
    if outcome.commitment_hash != commitment.commitment_hash:
        raise OutcomeLearningError("outcome does not carry the sealed prediction hash")
    comparison = _derive_comparison(commitment, outcome)
    attribution = _derive_attribution(commitment, outcome, comparison)
    return GroundedLearningEpisode(commitment, outcome, comparison, attribution)


def apply_outcome_learning(
    topology: Any,
    episode: GroundedLearningEpisode,
    settlement_record: Any,
) -> tuple[Any, GroundedLearningEpisode]:
    """Delegate one eligible episode to the existing route-learning reducer."""

    if not isinstance(episode, GroundedLearningEpisode):
        raise OutcomeLearningError("adaptive delegation requires a learning episode")
    if not episode.attribution.eligible:
        return topology, episode
    from .plastic_routing import (
        SettlementRouteRecord,
        apply_settlement_learning,
    )

    if not isinstance(settlement_record, SettlementRouteRecord):
        raise OutcomeLearningError("adaptive delegation requires an existing route record")
    selection = settlement_record.selection
    if (
        selection.transaction_id != episode.commitment.transaction_id
        or selection.objective_id != episode.commitment.objective_id
        or selection.task_state_id != episode.commitment.task_state_id
        or selection.task_state_version != episode.commitment.task_state_version
        or selection.route_id != episode.commitment.route_id
    ):
        raise OutcomeLearningError("settlement route record crosses the episode boundary")
    if settlement_record.settlement.observed_outcome != episode.comparison.observed_outcome.value:
        raise OutcomeLearningError("settlement outcome disagrees with grounded comparison")
    grounded = settlement_record.grounded_execution
    if grounded is None or grounded.record.record_hash != episode.outcome.record_hash:
        raise OutcomeLearningError("settlement is not bound to the grounded outcome")
    next_topology, learning_trace = apply_settlement_learning(
        topology, settlement_record
    )
    updated = replace(
        episode,
        adaptive_learning={
            "settlement_id": learning_trace.settlement_id,
            "route_id": learning_trace.route_id,
            "disposition": learning_trace.disposition,
            "effect": learning_trace.effect,
            "reason": learning_trace.reason,
            "topology_before": learning_trace.topology_before,
            "topology_after": learning_trace.topology_after,
            "weight_before": learning_trace.weight_before,
            "weight_after": learning_trace.weight_after,
            "learning_update_id": (
                learning_trace.learning_update.update_id
                if learning_trace.learning_update is not None
                else None
            ),
        },
        episode_hash="",
    )
    return next_topology, updated


def replay_outcome_learning(
    record: GroundedLearningEpisode | Mapping[str, Any],
    *,
    topology: Any | None = None,
    settlement_record: Any | None = None,
) -> tuple[GroundedLearningEpisode, Any | None]:
    """Replay hashes and optional delegated learning without repeating work."""

    episode = (
        GroundedLearningEpisode.from_dict(record)
        if isinstance(record, Mapping)
        else record
    )
    if not isinstance(episode, GroundedLearningEpisode):
        raise OutcomeLearningError("replay requires a learning episode")
    reconstructed = evaluate_grounded_outcome(episode.commitment, episode.outcome)
    reconstructed_core = reconstructed.to_dict(include_hash=False)
    recorded_core = episode.to_dict(include_hash=False)
    reconstructed_core["adaptive_learning"] = recorded_core["adaptive_learning"] = None
    if reconstructed_core != recorded_core:
        raise OutcomeLearningError("outcome-learning replay changed the episode")
    if topology is None or settlement_record is None or not episode.attribution.eligible:
        return episode, topology
    replayed_topology, replayed = apply_outcome_learning(
        topology, episode, settlement_record
    )
    if replayed.adaptive_learning != episode.adaptive_learning:
        raise OutcomeLearningError("delegated learning replay changed its result")
    return replayed, replayed_topology


def replay_outcome_learning_history(
    records: Iterable[GroundedLearningEpisode | Mapping[str, Any]],
) -> tuple[GroundedLearningEpisode, ...]:
    """Replay a bounded episode stream and reject duplicate/conflicting delivery.

    This is a pure structural pass over caller-owned immutable records.  It is
    not a ledger or persistence owner.  One prediction may bind exactly one
    outcome, and one outcome identity may appear exactly once in a replay.
    """

    try:
        materialized = tuple(records)
    except TypeError as exc:
        raise OutcomeLearningError("replay history must be iterable") from exc
    if len(materialized) > MAX_REPLAY_EPISODES:
        raise OutcomeLearningError("replay history exceeds its episode bound")
    episodes: list[GroundedLearningEpisode] = []
    outcome_identities: dict[str, str] = {}
    commitment_outcomes: dict[str, tuple[str, str]] = {}
    for record in materialized:
        episode, _ = replay_outcome_learning(record)
        outcome_id = episode.outcome.outcome_id
        outcome_hash = episode.outcome.outcome_hash
        if outcome_id in outcome_identities:
            qualifier = (
                "duplicate"
                if outcome_identities[outcome_id] == outcome_hash
                else "conflicting"
            )
            raise OutcomeLearningError(f"{qualifier} outcome identity in replay")
        outcome_identities[outcome_id] = outcome_hash
        commitment_id = episode.commitment.commitment_id
        binding = (outcome_id, outcome_hash)
        if commitment_id in commitment_outcomes:
            qualifier = (
                "duplicate"
                if commitment_outcomes[commitment_id] == binding
                else "conflicting"
            )
            raise OutcomeLearningError(
                f"{qualifier} outcome for prediction commitment"
            )
        commitment_outcomes[commitment_id] = binding
        episodes.append(episode)
    return tuple(episodes)


__all__ = [
    "ContributionAttribution",
    "ContributionKind",
    "ContributionRecord",
    "CompetenceAttribution",
    "ErrorLocation",
    "GroundedLearningEpisode",
    "GroundedOutcome",
    "MAX_CONTRIBUTIONS",
    "MAX_CRITERIA",
    "MAX_REPLAY_EPISODES",
    "OutcomeKind",
    "OutcomeLearningError",
    "OutcomeComparison",
    "PredictionCommitment",
    "apply_outcome_learning",
    "bind_grounded_outcome",
    "commit_prediction",
    "evaluate_grounded_outcome",
    "make_contribution",
    "replay_outcome_learning",
    "replay_outcome_learning_history",
    "seal_prediction_request",
]