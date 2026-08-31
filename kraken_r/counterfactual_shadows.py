"""Zero-authority counterfactual shadow precommitments.

Shadows are sealed diagnostic predictions only.  They contain no executable
request and have no evidence, settlement, execution, or adaptive authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .outcome_learning import (
    ContributionKind,
    ContributionRecord,
    GroundedLearningEpisode,
    OutcomeKind,
    PredictionCommitment,
)


MAX_SHADOWS = 3
MAX_MODEL_CALLS_PER_SHADOW = 1
MAX_INPUT_TOKENS_PER_SHADOW = 12_000
MAX_OUTPUT_TOKENS_PER_SHADOW = 4_096
MAX_COMPUTE_UNITS_PER_SHADOW = 1_000
MAX_TOTAL_BUNDLE_MODEL_CALLS = 3
MAX_TOTAL_BUNDLE_TOKENS = 48_000
MAX_TOTAL_BUNDLE_COMPUTE_UNITS = 3_000
MAX_RECORD_BYTES = 96_000
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class CounterfactualShadowError(ValueError):
    """Raised when a shadow crosses its diagnostic-only boundary."""


class ShadowIntervention(str, Enum):
    REMOVE = "remove"
    ALTER = "alter"


class ShadowQualityEffect(str, Enum):
    DEGRADED = "degraded"
    IMPROVED = "improved"
    UNCHANGED = "unchanged"


class ShadowDiagnosticDisposition(str, Enum):
    PRESERVED = "preserved"
    WITHHELD = "withheld"


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


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _id(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(char.isspace() for char in value)
        or len(value) > 256
    ):
        raise CounterfactualShadowError(f"{name} must be a bounded identifier")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise CounterfactualShadowError(f"{name} must be a SHA-256 digest")
    return value


def _bounded_int(value: Any, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise CounterfactualShadowError(f"{name} exceeds its bound")
    return value


def _record_size(value: Any) -> None:
    if len(json.dumps(_jsonable(value), sort_keys=True).encode("utf-8")) > MAX_RECORD_BYTES:
        raise CounterfactualShadowError("shadow record exceeds its byte bound")


@dataclass(frozen=True)
class ShadowResourceUsage:
    """Separately accounted bounded resources for one prediction path."""

    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    compute_units: int = 0
    accounting_basis: str = "caller_declared_preoutcome"

    def __post_init__(self) -> None:
        _bounded_int(self.model_calls, "model_calls", MAX_MODEL_CALLS_PER_SHADOW)
        _bounded_int(self.input_tokens, "input_tokens", MAX_INPUT_TOKENS_PER_SHADOW)
        _bounded_int(self.output_tokens, "output_tokens", MAX_OUTPUT_TOKENS_PER_SHADOW)
        _bounded_int(self.compute_units, "compute_units", MAX_COMPUTE_UNITS_PER_SHADOW)
        if self.accounting_basis != "caller_declared_preoutcome":
            raise CounterfactualShadowError("resource accounting basis is invalid")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "compute_units": self.compute_units,
            "accounting_basis": self.accounting_basis,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShadowResourceUsage":
        if not isinstance(value, Mapping) or set(value) != {
            "model_calls", "input_tokens", "output_tokens", "compute_units",
            "accounting_basis",
        }:
            raise CounterfactualShadowError("resource usage schema is invalid")
        return cls(**value)


@dataclass(frozen=True)
class ShadowSpecification:
    """Caller-supplied pre-outcome counterfactual proposal."""

    shadow_id: str
    contributor_id: str
    intervention: ShadowIntervention
    expected_outcome: str
    resources: ShadowResourceUsage = field(default_factory=ShadowResourceUsage)
    replacement_contribution: ContributionRecord | None = None

    def __post_init__(self) -> None:
        _id(self.shadow_id, "shadow_id")
        _id(self.contributor_id, "contributor_id")
        object.__setattr__(self, "intervention", ShadowIntervention(self.intervention))
        if self.expected_outcome not in {"success", "failure"}:
            raise CounterfactualShadowError("shadow prediction must be success or failure")
        if not isinstance(self.resources, ShadowResourceUsage):
            raise CounterfactualShadowError("shadow resources are invalid")
        if self.intervention is ShadowIntervention.ALTER:
            if not isinstance(self.replacement_contribution, ContributionRecord):
                raise CounterfactualShadowError("alteration requires a replacement contribution")
        elif self.replacement_contribution is not None:
            raise CounterfactualShadowError("removal cannot carry a replacement contribution")


@dataclass(frozen=True)
class ShadowPrecommitment:
    """One sealed, non-executable, single-contributor counterfactual."""

    shadow_id: str
    bundle_id: str
    primary_commitment_id: str
    primary_commitment_hash: str
    transaction_id: str
    objective_id: str
    task_state_id: str
    task_state_version: int
    request_id: str
    planned_work_hash: str
    cognitive_input_hash: str
    preoutcome_information_hash: str
    contributor_id: str
    contributor_kind: ContributionKind
    contributor_hash: str
    intervention: ShadowIntervention
    retained_contribution_hashes: tuple[str, ...]
    replacement_contribution: ContributionRecord | None
    transformed_input_hash: str
    expected_outcome: str
    success_criteria: tuple[str, ...]
    resources: ShadowResourceUsage
    candidate_only: bool = True
    execution_authority: bool = False
    evidence_authority: bool = False
    settlement_authority: bool = False
    adaptive_credit_authority: bool = False
    shadow_hash: str = ""

    def __post_init__(self) -> None:
        for value, name in (
            (self.shadow_id, "shadow_id"),
            (self.bundle_id, "bundle_id"),
            (self.primary_commitment_id, "primary_commitment_id"),
            (self.transaction_id, "transaction_id"),
            (self.objective_id, "objective_id"),
            (self.task_state_id, "task_state_id"),
            (self.request_id, "request_id"),
            (self.contributor_id, "contributor_id"),
        ):
            _id(value, name)
        for value, name in (
            (self.primary_commitment_hash, "primary_commitment_hash"),
            (self.planned_work_hash, "planned_work_hash"),
            (self.cognitive_input_hash, "cognitive_input_hash"),
            (self.preoutcome_information_hash, "preoutcome_information_hash"),
            (self.contributor_hash, "contributor_hash"),
            (self.transformed_input_hash, "transformed_input_hash"),
        ):
            _digest(value, name)
        if isinstance(self.task_state_version, bool) or self.task_state_version < 1:
            raise CounterfactualShadowError("task state version is invalid")
        object.__setattr__(self, "contributor_kind", ContributionKind(self.contributor_kind))
        object.__setattr__(self, "intervention", ShadowIntervention(self.intervention))
        retained = tuple(self.retained_contribution_hashes)
        if len(set(retained)) != len(retained):
            raise CounterfactualShadowError("retained contributors are duplicated")
        for item in retained:
            _digest(item, "retained contribution hash")
        object.__setattr__(self, "retained_contribution_hashes", retained)
        if self.intervention is ShadowIntervention.ALTER:
            if not isinstance(self.replacement_contribution, ContributionRecord):
                raise CounterfactualShadowError("alteration requires a replacement contribution")
        elif self.replacement_contribution is not None:
            raise CounterfactualShadowError("removed shadow cannot have a replacement")
        if self.expected_outcome not in {"success", "failure"}:
            raise CounterfactualShadowError("shadow expected outcome is invalid")
        object.__setattr__(self, "success_criteria", tuple(self.success_criteria))
        if not self.success_criteria:
            raise CounterfactualShadowError("shadow success criteria are empty")
        if not isinstance(self.resources, ShadowResourceUsage):
            raise CounterfactualShadowError("shadow resources are invalid")
        if (
            self.candidate_only is not True
            or self.execution_authority
            or self.evidence_authority
            or self.settlement_authority
            or self.adaptive_credit_authority
        ):
            raise CounterfactualShadowError("shadow authority must remain zero")
        expected = _hash(self.to_dict(include_hash=False))
        if self.shadow_hash and self.shadow_hash != expected:
            raise CounterfactualShadowError("shadow hash is invalid")
        if not self.shadow_hash:
            object.__setattr__(self, "shadow_hash", expected)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            key: _jsonable(getattr(self, key))
            for key in (
                "shadow_id", "bundle_id", "primary_commitment_id",
                "primary_commitment_hash", "transaction_id", "objective_id",
                "task_state_id", "task_state_version", "request_id",
                "planned_work_hash", "cognitive_input_hash",
                "preoutcome_information_hash", "contributor_id",
                "contributor_kind", "contributor_hash", "intervention",
                "retained_contribution_hashes", "replacement_contribution",
                "transformed_input_hash",
                "expected_outcome", "success_criteria", "resources",
                "candidate_only", "execution_authority", "evidence_authority",
                "settlement_authority", "adaptive_credit_authority",
            )
        }
        if include_hash:
            value["shadow_hash"] = self.shadow_hash
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShadowPrecommitment":
        if not isinstance(value, Mapping):
            raise CounterfactualShadowError("shadow schema is invalid")
        data = dict(value)
        data["contributor_kind"] = ContributionKind(data["contributor_kind"])
        data["intervention"] = ShadowIntervention(data["intervention"])
        data["retained_contribution_hashes"] = tuple(data["retained_contribution_hashes"])
        data["replacement_contribution"] = (
            ContributionRecord.from_dict(data["replacement_contribution"])
            if data["replacement_contribution"] is not None
            else None
        )
        data["success_criteria"] = tuple(data["success_criteria"])
        data["resources"] = ShadowResourceUsage.from_dict(data["resources"])
        return cls(**data)


def _preoutcome_hash(primary: PredictionCommitment) -> str:
    return _hash(
        {
            "commitment_hash": primary.commitment_hash,
            "processing_trace": primary.processing_trace,
            "contributions": [item.to_dict() for item in primary.contributions],
            "success_criteria": list(primary.success_criteria),
        }
    )


@dataclass(frozen=True)
class ShadowBundle:
    """Primary plus bounded shadows, sealed before any outcome is attached."""

    bundle_id: str
    primary: PredictionCommitment
    primary_resources: ShadowResourceUsage
    shadows: tuple[ShadowPrecommitment, ...]
    phase: str = "pre_outcome"
    bundle_hash: str = ""

    def __post_init__(self) -> None:
        _id(self.bundle_id, "bundle_id")
        if not isinstance(self.primary, PredictionCommitment):
            raise CounterfactualShadowError("bundle primary must be a prediction commitment")
        if not isinstance(self.primary_resources, ShadowResourceUsage):
            raise CounterfactualShadowError("primary resources are invalid")
        shadows = tuple(self.shadows)
        if not 1 <= len(shadows) <= MAX_SHADOWS:
            raise CounterfactualShadowError("shadow count is empty or exceeds its bound")
        if not all(isinstance(item, ShadowPrecommitment) for item in shadows):
            raise CounterfactualShadowError("bundle shadows are invalid")
        if len({item.shadow_id for item in shadows}) != len(shadows):
            raise CounterfactualShadowError("shadow identities are duplicated")
        if len({item.contributor_id for item in shadows}) != len(shadows):
            raise CounterfactualShadowError("contributors cannot contaminate multiple shadows")
        primary_hashes = tuple(item.contribution_hash for item in self.primary.contributions)
        information_hash = _preoutcome_hash(self.primary)
        for shadow in shadows:
            target = next(
                (item for item in self.primary.contributions if item.contribution_id == shadow.contributor_id),
                None,
            )
            expected_retained = tuple(
                item.contribution_hash
                for item in self.primary.contributions
                if item.contribution_id != shadow.contributor_id
            )
            if (
                target is None
                or shadow.bundle_id != self.bundle_id
                or shadow.primary_commitment_id != self.primary.commitment_id
                or shadow.primary_commitment_hash != self.primary.commitment_hash
                or shadow.transaction_id != self.primary.transaction_id
                or shadow.objective_id != self.primary.objective_id
                or shadow.task_state_id != self.primary.task_state_id
                or shadow.task_state_version != self.primary.task_state_version
                or shadow.request_id != self.primary.request_id
                or shadow.planned_work_hash != self.primary.planned_work_hash
                or shadow.cognitive_input_hash != self.primary.cognitive_input_hash
                or shadow.preoutcome_information_hash != information_hash
                or shadow.contributor_kind is not target.kind
                or shadow.contributor_hash != target.contribution_hash
                or shadow.retained_contribution_hashes != expected_retained
                or shadow.success_criteria != self.primary.success_criteria
            ):
                raise CounterfactualShadowError("shadow crosses or forges its primary lineage")
            expected_transformed = _hash(
                {
                    "preoutcome_information_hash": information_hash,
                    "target_contributor_id": target.contribution_id,
                    "target_contributor_hash": target.contribution_hash,
                    "intervention": shadow.intervention.value,
                    "retained_contribution_hashes": expected_retained,
                    "replacement_contribution": (
                        shadow.replacement_contribution.to_dict()
                        if shadow.replacement_contribution is not None
                        else None
                    ),
                }
            )
            if shadow.transformed_input_hash != expected_transformed:
                raise CounterfactualShadowError("shadow transformed input is forged")
            if shadow.intervention is ShadowIntervention.ALTER and (
                shadow.replacement_contribution.contribution_id != target.contribution_id
                or shadow.replacement_contribution.kind is not target.kind
                or shadow.replacement_contribution == target
                or shadow.replacement_contribution.contribution_hash in primary_hashes
            ):
                raise CounterfactualShadowError("alteration does not change exactly one contributor")
        calls = self.primary_resources.model_calls + sum(
            item.resources.model_calls for item in shadows
        )
        tokens = self.primary_resources.total_tokens + sum(
            item.resources.total_tokens for item in shadows
        )
        compute = self.primary_resources.compute_units + sum(
            item.resources.compute_units for item in shadows
        )
        if (
            calls > MAX_TOTAL_BUNDLE_MODEL_CALLS
            or tokens > MAX_TOTAL_BUNDLE_TOKENS
            or compute > MAX_TOTAL_BUNDLE_COMPUTE_UNITS
        ):
            raise CounterfactualShadowError(
                "aggregate primary-and-shadow resources exceed their bundle bound"
            )
        object.__setattr__(self, "shadows", shadows)
        if self.phase != "pre_outcome":
            raise CounterfactualShadowError("shadow bundle must be sealed pre-outcome")
        expected = _hash(self.to_dict(include_hash=False))
        if self.bundle_hash and self.bundle_hash != expected:
            raise CounterfactualShadowError("shadow bundle hash is invalid")
        if not self.bundle_hash:
            object.__setattr__(self, "bundle_hash", expected)
        _record_size(self.to_dict())

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "bundle_id": self.bundle_id,
            "primary": self.primary.to_dict(),
            "primary_resources": self.primary_resources.to_dict(),
            "shadows": [item.to_dict() for item in self.shadows],
            "phase": self.phase,
        }
        if include_hash:
            value["bundle_hash"] = self.bundle_hash
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShadowBundle":
        if not isinstance(value, Mapping) or set(value) != {
            "bundle_id", "primary", "primary_resources", "shadows", "phase", "bundle_hash"
        }:
            raise CounterfactualShadowError("shadow bundle schema is invalid")
        return cls(
            value["bundle_id"],
            PredictionCommitment.from_dict(value["primary"]),
            ShadowResourceUsage.from_dict(value["primary_resources"]),
            tuple(ShadowPrecommitment.from_dict(item) for item in value["shadows"]),
            value["phase"],
            value["bundle_hash"],
        )


def _seal_shadow_bundle(
    primary: PredictionCommitment,
    specifications: Iterable[ShadowSpecification],
    *,
    bundle_id: str,
    primary_resources: ShadowResourceUsage | None = None,
) -> ShadowBundle:
    """Seal shadows from a bare pre-outcome commitment, never an episode."""

    if not isinstance(primary, PredictionCommitment):
        raise CounterfactualShadowError(
            "shadows can only be created from a pre-outcome prediction commitment"
        )
    specs = tuple(specifications)
    if not all(isinstance(item, ShadowSpecification) for item in specs):
        raise CounterfactualShadowError("shadow specifications are invalid")
    information_hash = _preoutcome_hash(primary)
    shadows: list[ShadowPrecommitment] = []
    for spec in specs:
        target = next(
            (item for item in primary.contributions if item.contribution_id == spec.contributor_id),
            None,
        )
        if target is None:
            raise CounterfactualShadowError("shadow target is not a primary contributor")
        shadows.append(
            ShadowPrecommitment(
                spec.shadow_id,
                bundle_id,
                primary.commitment_id,
                primary.commitment_hash,
                primary.transaction_id,
                primary.objective_id,
                primary.task_state_id,
                primary.task_state_version,
                primary.request_id,
                primary.planned_work_hash,
                primary.cognitive_input_hash,
                information_hash,
                target.contribution_id,
                target.kind,
                target.contribution_hash,
                spec.intervention,
                tuple(
                    item.contribution_hash
                    for item in primary.contributions
                    if item.contribution_id != target.contribution_id
                ),
                spec.replacement_contribution,
                _hash(
                    {
                        "preoutcome_information_hash": information_hash,
                        "target_contributor_id": target.contribution_id,
                        "target_contributor_hash": target.contribution_hash,
                        "intervention": spec.intervention.value,
                        "retained_contribution_hashes": tuple(
                            item.contribution_hash
                            for item in primary.contributions
                            if item.contribution_id != target.contribution_id
                        ),
                        "replacement_contribution": (
                            spec.replacement_contribution.to_dict()
                            if spec.replacement_contribution is not None
                            else None
                        ),
                    }
                ),
                spec.expected_outcome,
                primary.success_criteria,
                spec.resources,
            )
        )
    return ShadowBundle(
        bundle_id,
        primary,
        primary_resources or ShadowResourceUsage(),
        tuple(shadows),
    )


@dataclass(frozen=True)
class ShadowComparison:
    shadow_id: str
    contributor_id: str
    observed_outcome: OutcomeKind
    primary_correct: bool
    shadow_correct: bool
    effect: ShadowQualityEffect
    comparison_hash: str = ""

    def __post_init__(self) -> None:
        _id(self.shadow_id, "shadow_id")
        _id(self.contributor_id, "contributor_id")
        object.__setattr__(self, "observed_outcome", OutcomeKind(self.observed_outcome))
        object.__setattr__(self, "effect", ShadowQualityEffect(self.effect))
        if not isinstance(self.primary_correct, bool) or not isinstance(self.shadow_correct, bool):
            raise CounterfactualShadowError("comparison correctness flags are invalid")
        expected_effect = (
            ShadowQualityEffect.DEGRADED
            if self.primary_correct and not self.shadow_correct
            else ShadowQualityEffect.IMPROVED
            if self.shadow_correct and not self.primary_correct
            else ShadowQualityEffect.UNCHANGED
        )
        if self.effect is not expected_effect:
            raise CounterfactualShadowError("shadow quality effect is inconsistent")
        expected = _hash(self.to_dict(include_hash=False))
        if self.comparison_hash and self.comparison_hash != expected:
            raise CounterfactualShadowError("shadow comparison hash is invalid")
        if not self.comparison_hash:
            object.__setattr__(self, "comparison_hash", expected)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "shadow_id": self.shadow_id,
            "contributor_id": self.contributor_id,
            "observed_outcome": self.observed_outcome.value,
            "primary_correct": self.primary_correct,
            "shadow_correct": self.shadow_correct,
            "effect": self.effect.value,
        }
        if include_hash:
            value["comparison_hash"] = self.comparison_hash
        return value


@dataclass(frozen=True)
class ShadowDiagnostic:
    bundle: ShadowBundle
    episode_id: str
    episode_hash: str
    outcome_hash: str
    comparisons: tuple[ShadowComparison, ...]
    prior_eligible: bool
    eligible: bool
    disposition: ShadowDiagnosticDisposition
    reason: str
    candidate_diagnostic_only: bool = True
    diagnostic_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.bundle, ShadowBundle):
            raise CounterfactualShadowError("diagnostic bundle is invalid")
        for value, name in (
            (self.episode_id, "episode_id"),
        ):
            if not isinstance(value, str) or not value:
                raise CounterfactualShadowError(f"{name} is invalid")
        if not isinstance(self.reason, str) or not self.reason or len(self.reason) > 512:
            raise CounterfactualShadowError("diagnostic reason is invalid or unbounded")
        _digest(self.episode_hash, "episode_hash")
        _digest(self.outcome_hash, "outcome_hash")
        comparisons = tuple(self.comparisons)
        if tuple(item.shadow_id for item in comparisons) != tuple(
            item.shadow_id for item in self.bundle.shadows
        ):
            raise CounterfactualShadowError("diagnostic comparison set changed")
        object.__setattr__(self, "comparisons", comparisons)
        if self.eligible and not self.prior_eligible:
            raise CounterfactualShadowError("shadow diagnostic cannot grant eligibility")
        expected_eligible = self.prior_eligible and all(
            item.effect is ShadowQualityEffect.DEGRADED for item in comparisons
        )
        if self.eligible != expected_eligible:
            raise CounterfactualShadowError("diagnostic eligibility is inconsistent")
        object.__setattr__(
            self,
            "disposition",
            ShadowDiagnosticDisposition.PRESERVED
            if self.eligible
            else ShadowDiagnosticDisposition.WITHHELD,
        )
        if self.candidate_diagnostic_only is not True:
            raise CounterfactualShadowError("shadow result must remain diagnostic only")
        expected = _hash(self.to_dict(include_hash=False))
        if self.diagnostic_hash and self.diagnostic_hash != expected:
            raise CounterfactualShadowError("shadow diagnostic hash is invalid")
        if not self.diagnostic_hash:
            object.__setattr__(self, "diagnostic_hash", expected)
        _record_size(self.to_dict())

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "bundle": self.bundle.to_dict(),
            "episode_id": self.episode_id,
            "episode_hash": self.episode_hash,
            "outcome_hash": self.outcome_hash,
            "comparisons": [item.to_dict() for item in self.comparisons],
            "prior_eligible": self.prior_eligible,
            "eligible": self.eligible,
            "disposition": self.disposition.value,
            "reason": self.reason,
            "candidate_diagnostic_only": self.candidate_diagnostic_only,
        }
        if include_hash:
            value["diagnostic_hash"] = self.diagnostic_hash
        return value


def evaluate_shadow_diagnostic(
    bundle: ShadowBundle,
    episode: GroundedLearningEpisode,
) -> ShadowDiagnostic:
    """Compare a presealed bundle to the primary's independently grounded outcome."""

    if not isinstance(bundle, ShadowBundle) or not isinstance(episode, GroundedLearningEpisode):
        raise CounterfactualShadowError("shadow evaluation inputs are invalid")
    if (
        episode.commitment != bundle.primary
        or episode.outcome.commitment_id != bundle.primary.commitment_id
        or episode.outcome.commitment_hash != bundle.primary.commitment_hash
    ):
        raise CounterfactualShadowError("outcome is not bound to the shadow primary")
    base_episode = GroundedLearningEpisode(
        episode.commitment,
        episode.outcome,
        episode.comparison,
        episode.attribution,
    )
    observed = episode.comparison.observed_outcome
    primary_correct = bundle.primary.expected_outcome == observed.value
    comparisons = tuple(
        ShadowComparison(
            item.shadow_id,
            item.contributor_id,
            observed,
            primary_correct,
            item.expected_outcome == observed.value,
            (
                ShadowQualityEffect.DEGRADED
                if primary_correct and item.expected_outcome != observed.value
                else ShadowQualityEffect.IMPROVED
                if not primary_correct and item.expected_outcome == observed.value
                else ShadowQualityEffect.UNCHANGED
            ),
        )
        for item in bundle.shadows
    )
    eligible = episode.attribution.eligible and all(
        item.effect is ShadowQualityEffect.DEGRADED for item in comparisons
    )
    return ShadowDiagnostic(
        bundle,
        episode.commitment.episode_id,
        base_episode.episode_hash,
        episode.outcome.outcome_hash,
        comparisons,
        episode.attribution.eligible,
        eligible,
        ShadowDiagnosticDisposition.PRESERVED if eligible else ShadowDiagnosticDisposition.WITHHELD,
        (
            "every tested contributor materially improved prediction quality"
            if eligible
            else "existing credit withheld because a tested contributor was not material"
            if episode.attribution.eligible
            else "existing Round 4 eligibility was already withheld"
        ),
    )


def replay_shadow_diagnostic(
    bundle_record: ShadowBundle | Mapping[str, Any],
    episode: GroundedLearningEpisode,
    diagnostic_record: ShadowDiagnostic | None = None,
) -> ShadowDiagnostic:
    """Replay from immutable records without live calls or work."""

    bundle = (
        ShadowBundle.from_dict(bundle_record)
        if isinstance(bundle_record, Mapping)
        else bundle_record
    )
    result = evaluate_shadow_diagnostic(bundle, episode)
    if diagnostic_record is not None and result.to_dict() != diagnostic_record.to_dict():
        raise CounterfactualShadowError("shadow replay changed the diagnostic")
    return result


__all__ = [
    "CounterfactualShadowError",
    "MAX_SHADOWS",
    "ShadowBundle",
    "ShadowComparison",
    "ShadowDiagnostic",
    "ShadowDiagnosticDisposition",
    "ShadowIntervention",
    "ShadowPrecommitment",
    "ShadowQualityEffect",
    "ShadowResourceUsage",
    "ShadowSpecification",
    "evaluate_shadow_diagnostic",
    "replay_shadow_diagnostic",
]