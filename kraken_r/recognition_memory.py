"""Candidate-only recognition, compression, and reactivation.

Round 2 is an immutable in-memory projection over detailed cognition episodes.
It owns no store, clock, worker, scheduler, provider, evidence, settlement,
adaptive reducer, or runtime dispatch.  A compressed memory is a reversible
index over caller-owned episode records; recognition can only produce a
validated context projection for the existing candidate cognition kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .cognition_kernel import CognitionValidationError, SemanticJob
from .semantic_capability import CandidateCognition, SemanticCapabilityResult


MAX_EPISODES = 32
MAX_MEMORIES = 32
MAX_FEATURES = 24
MAX_SCOPE_ITEMS = 16
MAX_EXCEPTIONS = 16
MAX_SOURCE_IDS = 64
MAX_EXPANSION_REFS = 32
MAX_CUE_TOKENS = 64
MAX_CUE_BYTES = 4_096
MAX_MATCHES = 32
MAX_CONTEXT_ITEMS = 32
MAX_MEMORY_TEXT = 4096
MAX_MEMORY_SET_ITEMS = 256
MAX_MEMORY_SET_BYTES = 12_000
MAX_PROCESSING_CONTEXT_BYTES = 8_192
MIN_MATCH_SCORE = 0.30
MIN_MATCH_MARGIN = 0.10


class RecognitionMemoryError(CognitionValidationError):
    """Raised when a recognition-memory candidate contract fails closed."""


class MemoryState(str, Enum):
    ACTIVE = "active"
    DEFERRED = "deferred"
    DORMANT = "dormant"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
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
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _identifier(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(character.isspace() for character in value)
    ):
        raise RecognitionMemoryError(
            f"{field_name} must be a non-empty identifier without whitespace"
        )
    return value


def _text(value: Any, field_name: str, maximum: int = MAX_MEMORY_TEXT) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecognitionMemoryError(f"{field_name} must be non-empty text")
    if len(value) > maximum:
        raise RecognitionMemoryError(f"{field_name} exceeds its bound")
    return value


def _items(
    values: Iterable[Any],
    field_name: str,
    maximum: int,
    *,
    identifiers: bool = False,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    try:
        values = tuple(values)
    except TypeError as exc:
        raise RecognitionMemoryError(f"{field_name} must be iterable") from exc
    if not allow_empty and not values:
        raise RecognitionMemoryError(f"{field_name} cannot be empty")
    if len(values) > maximum:
        raise RecognitionMemoryError(f"{field_name} exceeds its bound")
    result = tuple(
        (_identifier(item, field_name) if identifiers else _text(item, field_name))
        for item in values
    )
    if len(set(result)) != len(result):
        raise RecognitionMemoryError(f"{field_name} must be unique")
    return result


def _normalize_terms(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    terms: list[str] = []
    for value in values:
        text = _text(value, field_name, 256).lower()
        for token in re.findall(r"[a-z0-9_]+", text):
            if token and token not in terms:
                terms.append(token)
    if len(terms) > MAX_FEATURES:
        raise RecognitionMemoryError(f"{field_name} exceeds its feature bound")
    return tuple(terms)


@dataclass(frozen=True)
class CognitionEpisode:
    """One detailed, candidate-only cognition result retained by reference."""

    episode_id: str
    problem_id: str
    branch_id: str
    concept_id: str
    concept_version: int
    semantic_job: SemanticJob
    candidate_id: str
    statement: str
    rationale: str
    detail: Mapping[str, Any]
    discriminative_features: tuple[str, ...]
    supporting_features: tuple[str, ...]
    structural_features: tuple[str, ...]
    exceptions: tuple[str, ...]
    applicability_scope: tuple[str, ...]
    source_material_ids: tuple[str, ...]
    source_lineage_ids: tuple[str, ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "episode_id",
            "problem_id",
            "branch_id",
            "concept_id",
            "candidate_id",
        ):
            _identifier(getattr(self, name), name)
        if (
            isinstance(self.concept_version, bool)
            or not isinstance(self.concept_version, int)
            or self.concept_version < 1
        ):
            raise RecognitionMemoryError("concept_version must be positive")
        object.__setattr__(self, "semantic_job", SemanticJob(self.semantic_job))
        _text(self.statement, "episode statement")
        _text(self.rationale, "episode rationale")
        object.__setattr__(
            self, "detail", _freeze(_bounded_json(self.detail, "episode detail"))
        )
        for name in (
            "discriminative_features",
            "supporting_features",
            "structural_features",
        ):
            object.__setattr__(
                self,
                name,
                _normalize_terms(getattr(self, name), name),
            )
        object.__setattr__(
            self, "exceptions", _items(self.exceptions, "exceptions", MAX_EXCEPTIONS)
        )
        object.__setattr__(
            self,
            "applicability_scope",
            _items(
                self.applicability_scope,
                "applicability_scope",
                MAX_SCOPE_ITEMS,
                identifiers=True,
            ),
        )
        object.__setattr__(
            self,
            "source_material_ids",
            _items(
                self.source_material_ids,
                "source_material_ids",
                MAX_SOURCE_IDS,
                identifiers=True,
            ),
        )
        object.__setattr__(
            self,
            "source_lineage_ids",
            _items(
                self.source_lineage_ids,
                "source_lineage_ids",
                MAX_SOURCE_IDS,
                identifiers=True,
            ),
        )
        if not isinstance(self.provenance, Mapping):
            raise RecognitionMemoryError("episode provenance must be a mapping")
        object.__setattr__(
            self, "provenance", _freeze(_bounded_json(self.provenance, "provenance"))
        )

    @classmethod
    def from_semantic_result(
        cls,
        episode_id: str,
        result: SemanticCapabilityResult,
        *,
        concept_id: str,
        concept_version: int,
        source_material_ids: Iterable[str],
        discriminative_features: Iterable[str],
        supporting_features: Iterable[str] = (),
        structural_features: Iterable[str] = (),
        exceptions: Iterable[str] = (),
        applicability_scope: Iterable[str] = (),
    ) -> "CognitionEpisode":
        if not isinstance(result, SemanticCapabilityResult):
            raise RecognitionMemoryError("episode requires a semantic result")
        if not result.candidates:
            raise RecognitionMemoryError("episode requires a candidate result")
        candidate = result.candidates[0]
        if not isinstance(candidate, CandidateCognition):
            raise RecognitionMemoryError("episode candidate is invalid")
        detail = {
            "semantic_result": result.to_dict(),
            "candidate": candidate.to_dict(),
        }
        return cls(
            episode_id=episode_id,
            problem_id=candidate.provenance.problem_id,
            branch_id=candidate.provenance.branch_id,
            concept_id=concept_id,
            concept_version=concept_version,
            semantic_job=result.job,
            candidate_id=candidate.candidate_id,
            statement=candidate.statement,
            rationale=candidate.rationale,
            detail=detail,
            discriminative_features=tuple(discriminative_features),
            supporting_features=tuple(supporting_features),
            structural_features=tuple(structural_features),
            exceptions=tuple(exceptions),
            applicability_scope=tuple(applicability_scope),
            source_material_ids=tuple(source_material_ids),
            source_lineage_ids=candidate.provenance.source_lineage_ids,
            provenance={
                "operation_request_id": candidate.provenance.operation_request_id,
                "operation_input_hash": candidate.provenance.operation_input_hash,
                "focused_context_hash": candidate.provenance.focused_context_hash,
                "topology_read_hash": candidate.provenance.topology_read_hash,
                "provider_id": candidate.provenance.provider_id,
                "model_id": candidate.provenance.model_id,
                "provider_output_hash": candidate.provenance.provider_output_hash,
            },
        )

    @property
    def detail_hash(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "problem_id": self.problem_id,
            "branch_id": self.branch_id,
            "concept_id": self.concept_id,
            "concept_version": self.concept_version,
            "semantic_job": self.semantic_job.value,
            "candidate_id": self.candidate_id,
            "statement": self.statement,
            "rationale": self.rationale,
            "detail": _jsonable(self.detail),
            "discriminative_features": list(self.discriminative_features),
            "supporting_features": list(self.supporting_features),
            "structural_features": list(self.structural_features),
            "exceptions": list(self.exceptions),
            "applicability_scope": list(self.applicability_scope),
            "source_material_ids": list(self.source_material_ids),
            "source_lineage_ids": list(self.source_lineage_ids),
            "provenance": _jsonable(self.provenance),
        }


def _bounded_json(
    value: Any,
    field_name: str,
    *,
    maximum_items: int = 64,
    maximum_bytes: int = 32_768,
    maximum_depth: int = 6,
) -> Any:
    count = 0

    def visit(item: Any, depth: int) -> Any:
        nonlocal count
        if depth > maximum_depth:
            raise RecognitionMemoryError(f"{field_name} exceeds nesting bound")
        if item is None or isinstance(item, (bool, int, float, str)):
            if isinstance(item, str) and len(item) > MAX_MEMORY_TEXT:
                raise RecognitionMemoryError(f"{field_name} contains oversized text")
            return item
        if isinstance(item, Mapping):
            count += len(item)
            if count > maximum_items:
                raise RecognitionMemoryError(f"{field_name} exceeds item bound")
            return {str(key): visit(child, depth + 1) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            count += len(item)
            if count > maximum_items:
                raise RecognitionMemoryError(f"{field_name} exceeds item bound")
            return [visit(child, depth + 1) for child in item]
        raise RecognitionMemoryError(f"{field_name} contains unsupported data")

    result = visit(value, 0)
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"))
    if len(raw.encode("utf-8")) > maximum_bytes:
        raise RecognitionMemoryError(f"{field_name} exceeds byte bound")
    return result


@dataclass(frozen=True)
class ExpansionReference:
    episode_id: str
    detail_hash: str
    source_material_ids: tuple[str, ...]
    source_lineage_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.episode_id, "expansion episode_id")
        _identifier(self.detail_hash, "expansion detail_hash")
        object.__setattr__(
            self,
            "source_material_ids",
            _items(
                self.source_material_ids,
                "expansion source_material_ids",
                MAX_SOURCE_IDS,
                identifiers=True,
            ),
        )
        object.__setattr__(
            self,
            "source_lineage_ids",
            _items(
                self.source_lineage_ids,
                "expansion source_lineage_ids",
                MAX_SOURCE_IDS,
                identifiers=True,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "detail_hash": self.detail_hash,
            "source_material_ids": list(self.source_material_ids),
            "source_lineage_ids": list(self.source_lineage_ids),
        }


@dataclass(frozen=True)
class CompressedMemory:
    memory_id: str
    concept_id: str
    concept_version: int
    summary: str
    discriminative_features: tuple[str, ...]
    supporting_features: tuple[str, ...]
    structural_features: tuple[str, ...]
    exceptions: tuple[str, ...]
    applicability_scope: tuple[str, ...]
    branch_ids: tuple[str, ...]
    capability_ids: tuple[str, ...]
    source_material_ids: tuple[str, ...]
    source_lineage_ids: tuple[str, ...]
    ancestry: tuple[ExpansionReference, ...]
    state: MemoryState = MemoryState.DEFERRED
    detailed_item_count: int = 1
    active_context_item_count: int = 1
    compression_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("memory_id", "concept_id"):
            _identifier(getattr(self, name), name)
        if (
            isinstance(self.concept_version, bool)
            or not isinstance(self.concept_version, int)
            or self.concept_version < 1
        ):
            raise RecognitionMemoryError("memory concept_version must be positive")
        _text(self.summary, "memory summary")
        for name in (
            "discriminative_features",
            "supporting_features",
            "structural_features",
        ):
            object.__setattr__(
                self, name, _normalize_terms(getattr(self, name), name)
            )
        if not self.discriminative_features and not self.structural_features:
            raise RecognitionMemoryError(
                "memory requires discriminative or structural features"
            )
        object.__setattr__(
            self, "exceptions", _items(self.exceptions, "memory exceptions", MAX_EXCEPTIONS)
        )
        object.__setattr__(
            self,
            "applicability_scope",
            _items(
                self.applicability_scope,
                "memory applicability_scope",
                MAX_SCOPE_ITEMS,
                identifiers=True,
            ),
        )
        object.__setattr__(
            self, "branch_ids", _items(self.branch_ids, "memory branch_ids", 8, identifiers=True)
        )
        object.__setattr__(
            self,
            "capability_ids",
            _items(self.capability_ids, "memory capability_ids", 16, identifiers=True),
        )
        object.__setattr__(
            self,
            "source_material_ids",
            _items(
                self.source_material_ids,
                "memory source_material_ids",
                MAX_SOURCE_IDS,
                identifiers=True,
            ),
        )
        object.__setattr__(
            self,
            "source_lineage_ids",
            _items(
                self.source_lineage_ids,
                "memory source_lineage_ids",
                MAX_SOURCE_IDS,
                identifiers=True,
            ),
        )
        ancestry = tuple(self.ancestry)
        if not 1 <= len(ancestry) <= MAX_EXPANSION_REFS or not all(
            isinstance(item, ExpansionReference) for item in ancestry
        ):
            raise RecognitionMemoryError("memory ancestry is invalid or unbounded")
        if len({item.episode_id for item in ancestry}) != len(ancestry):
            raise RecognitionMemoryError("memory ancestry episode IDs must be unique")
        object.__setattr__(self, "ancestry", ancestry)
        object.__setattr__(self, "state", MemoryState(self.state))
        for name in ("detailed_item_count", "active_context_item_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RecognitionMemoryError(f"{name} must be positive")
        if self.active_context_item_count >= self.detailed_item_count:
            raise RecognitionMemoryError(
                "compression must reduce active context below detailed context"
            )
        expected_hash = _digest(
            {
                "memory_id": self.memory_id,
                "concept_id": self.concept_id,
                "concept_version": self.concept_version,
                "summary": self.summary,
                "discriminative_features": self.discriminative_features,
                "supporting_features": self.supporting_features,
                "structural_features": self.structural_features,
                "exceptions": self.exceptions,
                "applicability_scope": self.applicability_scope,
                "branch_ids": self.branch_ids,
                "capability_ids": self.capability_ids,
                "source_material_ids": self.source_material_ids,
                "source_lineage_ids": self.source_lineage_ids,
                "ancestry": self.ancestry,
                "detailed_item_count": self.detailed_item_count,
                "active_context_item_count": self.active_context_item_count,
            }
        )
        if self.compression_hash and self.compression_hash != expected_hash:
            raise RecognitionMemoryError("memory compression hash is invalid")
        object.__setattr__(self, "compression_hash", expected_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "concept_id": self.concept_id,
            "concept_version": self.concept_version,
            "summary": self.summary,
            "discriminative_features": list(self.discriminative_features),
            "supporting_features": list(self.supporting_features),
            "structural_features": list(self.structural_features),
            "exceptions": list(self.exceptions),
            "applicability_scope": list(self.applicability_scope),
            "branch_ids": list(self.branch_ids),
            "capability_ids": list(self.capability_ids),
            "source_material_ids": list(self.source_material_ids),
            "source_lineage_ids": list(self.source_lineage_ids),
            "ancestry": [item.to_dict() for item in self.ancestry],
            "state": self.state.value,
            "detailed_item_count": self.detailed_item_count,
            "active_context_item_count": self.active_context_item_count,
            "compression_hash": self.compression_hash,
        }


def compress_episodes(
    episodes: Iterable[CognitionEpisode],
    *,
    concept_id: str,
    concept_version: int,
    summary: str,
    discriminative_features: Iterable[str],
    supporting_features: Iterable[str] = (),
    structural_features: Iterable[str] = (),
    exceptions: Iterable[str] = (),
    applicability_scope: Iterable[str] = (),
    capability_ids: Iterable[str] = (),
    state: MemoryState = MemoryState.DEFERRED,
) -> CompressedMemory:
    episodes = tuple(episodes)
    if not 1 <= len(episodes) <= MAX_EPISODES:
        raise RecognitionMemoryError("compression episode count is out of bounds")
    if not all(isinstance(item, CognitionEpisode) for item in episodes):
        raise RecognitionMemoryError("compression requires cognition episodes")
    _identifier(concept_id, "concept_id")
    if isinstance(concept_version, bool) or not isinstance(concept_version, int) or concept_version < 1:
        raise RecognitionMemoryError("concept_version must be positive")
    if any(item.concept_id != concept_id or item.concept_version != concept_version for item in episodes):
        raise RecognitionMemoryError("concept identity/version mismatch prevents merge")
    conflicts = set()
    for item in episodes:
        conflicts.update(set(item.discriminative_features) & set(item.exceptions))
    if conflicts:
        raise RecognitionMemoryError(
            "destructive merge would discard discriminative exceptions"
        )
    discriminative = _normalize_terms(discriminative_features, "discriminative_features")
    supporting = _normalize_terms(supporting_features, "supporting_features")
    structural = _normalize_terms(structural_features, "structural_features")
    if not discriminative and not structural:
        raise RecognitionMemoryError("compression needs discriminative features")
    memory_materials = tuple(
        dict.fromkeys(item for episode in episodes for item in episode.source_material_ids)
    )
    memory_lineage = tuple(
        dict.fromkeys(item for episode in episodes for item in episode.source_lineage_ids)
    )
    branch_ids = tuple(dict.fromkeys(item.branch_id for item in episodes))
    scopes = tuple(
        dict.fromkeys(item for episode in episodes for item in episode.applicability_scope)
    )
    merged_exceptions = tuple(
        dict.fromkeys(
            tuple(item for episode in episodes for item in episode.exceptions)
            + tuple(exceptions)
        )
    )
    ancestry = tuple(
        ExpansionReference(
            episode.episode_id,
            episode.detail_hash,
            episode.source_material_ids,
            episode.source_lineage_ids,
        )
        for episode in episodes
    )
    detailed_count = sum(
        1
        + len(episode.detail)
        + len(episode.discriminative_features)
        + len(episode.supporting_features)
        + len(episode.exceptions)
        for episode in episodes
    )
    active_count = (
        1
        + len(discriminative)
        + len(supporting)
        + len(structural)
        + len(merged_exceptions)
        + len(scopes)
        + len(ancestry)
    )
    if active_count >= detailed_count:
        raise RecognitionMemoryError(
            "compression does not reduce active context for supplied episodes"
        )
    memory_id = "memory-" + _digest(
        {
            "concept_id": concept_id,
            "concept_version": concept_version,
            "episode_ids": [item.episode_id for item in episodes],
            "discriminative_features": discriminative,
            "supporting_features": supporting,
            "structural_features": structural,
            "exceptions": merged_exceptions,
            "applicability_scope": scopes,
        }
    )[:20]
    return CompressedMemory(
        memory_id=memory_id,
        concept_id=concept_id,
        concept_version=concept_version,
        summary=summary,
        discriminative_features=discriminative,
        supporting_features=supporting,
        structural_features=structural,
        exceptions=merged_exceptions,
        applicability_scope=scopes,
        branch_ids=branch_ids,
        capability_ids=tuple(capability_ids),
        source_material_ids=memory_materials,
        source_lineage_ids=memory_lineage,
        ancestry=ancestry,
        state=state,
        detailed_item_count=detailed_count,
        active_context_item_count=active_count,
    )


@dataclass(frozen=True)
class RecognitionCue:
    cue_id: str
    tokens: tuple[str, ...]
    structural_features: tuple[str, ...] = ()
    concept_id: str | None = None
    concept_version: int | None = None
    branch_id: str | None = None
    scope: tuple[str, ...] = ()
    delayed_ticks: int = 0

    def __post_init__(self) -> None:
        _identifier(self.cue_id, "cue_id")
        tokens = _normalize_terms(self.tokens, "cue tokens")
        if not tokens or len(tokens) > MAX_CUE_TOKENS:
            raise RecognitionMemoryError("cue tokens must be bounded and non-empty")
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(
            self,
            "structural_features",
            _normalize_terms(self.structural_features, "cue structural_features"),
        )
        _bounded_json(
            {
                "tokens": list(self.tokens),
                "structural_features": list(self.structural_features),
            },
            "cue feature payload",
            maximum_items=MAX_CUE_TOKENS,
            maximum_bytes=MAX_CUE_BYTES,
        )
        if self.concept_id is not None:
            _identifier(self.concept_id, "cue concept_id")
        if self.concept_version is not None and (
            isinstance(self.concept_version, bool)
            or not isinstance(self.concept_version, int)
            or self.concept_version < 1
        ):
            raise RecognitionMemoryError("cue concept_version must be positive")
        if self.branch_id is not None:
            _identifier(self.branch_id, "cue branch_id")
        object.__setattr__(
            self,
            "scope",
            _items(self.scope, "cue scope", MAX_SCOPE_ITEMS, identifiers=True),
        )
        if (
            isinstance(self.delayed_ticks, bool)
            or not isinstance(self.delayed_ticks, int)
            or not 0 <= self.delayed_ticks <= 1_000_000
        ):
            raise RecognitionMemoryError("cue delayed_ticks is invalid")

    @classmethod
    def from_text(cls, cue_id: str, text: str, **kwargs: Any) -> "RecognitionCue":
        _text(text, "cue text")
        return cls(cue_id, tuple(re.findall(r"[a-z0-9_]+", text.lower())), **kwargs)

    @property
    def cue_hash(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "cue_id": self.cue_id,
            "tokens": list(self.tokens),
            "structural_features": list(self.structural_features),
            "concept_id": self.concept_id,
            "concept_version": self.concept_version,
            "branch_id": self.branch_id,
            "scope": list(self.scope),
            "delayed_ticks": self.delayed_ticks,
        }


@dataclass(frozen=True)
class RecognitionMatch:
    memory_id: str
    matched: bool
    score: float
    required_hits: tuple[str, ...]
    supporting_hits: tuple[str, ...]
    structural_hits: tuple[str, ...]
    exclusions: tuple[str, ...]
    reason: str
    cue_hash: str
    memory_hash: str

    def __post_init__(self) -> None:
        _identifier(self.memory_id, "match memory_id")
        if not 0.0 <= float(self.score) <= 1.0:
            raise RecognitionMemoryError("match score must be from 0 through 1")
        for name in (
            "required_hits",
            "supporting_hits",
            "structural_hits",
            "exclusions",
        ):
            object.__setattr__(
                self,
                name,
                _items(getattr(self, name), f"match {name}", MAX_FEATURES),
            )
        _text(self.reason, "match reason", 512)
        _identifier(self.cue_hash, "match cue_hash")
        _identifier(self.memory_hash, "match memory_hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "matched": self.matched,
            "score": self.score,
            "required_hits": list(self.required_hits),
            "supporting_hits": list(self.supporting_hits),
            "structural_hits": list(self.structural_hits),
            "exclusions": list(self.exclusions),
            "reason": self.reason,
            "cue_hash": self.cue_hash,
            "memory_hash": self.memory_hash,
        }


def _match_memory(memory: CompressedMemory, cue: RecognitionCue) -> RecognitionMatch:
    cue_terms = set(cue.tokens)
    required_hits = tuple(
        item for item in memory.discriminative_features if item in cue_terms
    )
    supporting_hits = tuple(
        item for item in memory.supporting_features if item in cue_terms
    )
    structural_hits = tuple(
        item
        for item in memory.structural_features
        if item in cue_terms or item in cue.structural_features
    )
    exclusions = tuple(
        item
        for item in memory.exceptions
        if set(re.findall(r"[a-z0-9_]+", item.lower())) & cue_terms
    )
    compatible = (
        (cue.concept_id is None or cue.concept_id == memory.concept_id)
        and (
            cue.concept_version is None
            or cue.concept_version == memory.concept_version
        )
        and (not cue.scope or not memory.applicability_scope or bool(set(cue.scope) & set(memory.applicability_scope)))
        and (not cue.branch_id or not memory.branch_ids or cue.branch_id in memory.branch_ids)
        and not exclusions
    )
    denominator = (
        2 * len(memory.discriminative_features)
        + len(memory.supporting_features)
        + len(memory.structural_features)
    )
    score = (
        (
            2 * len(required_hits)
            + len(supporting_hits)
            + len(structural_hits)
        )
        / denominator
        if denominator
        else 0.0
    )
    matched = compatible and score >= MIN_MATCH_SCORE and (
        bool(required_hits)
        or (bool(structural_hits) and bool(supporting_hits))
    )
    if not compatible:
        reason = "cue is incompatible with concept, scope, branch, or exception"
    elif not matched:
        reason = "cue lacks sufficient discriminative coverage"
    else:
        reason = "bounded discriminative recognition match"
    return RecognitionMatch(
        memory.memory_id,
        matched,
        round(score, 6),
        required_hits,
        supporting_hits,
        structural_hits,
        exclusions,
        reason,
        cue.cue_hash,
        memory.compression_hash,
    )


@dataclass(frozen=True)
class RecognitionProjection:
    projection_id: str
    cue_hash: str
    memory_id: str
    active_memory_ids: tuple[str, ...]
    reactivated_memory_ids: tuple[str, ...]
    relevant_material_ids: tuple[str, ...]
    branch_ids: tuple[str, ...]
    capability_ids: tuple[str, ...]
    semantic_context: Mapping[str, Any]
    expanded_episode_ids: tuple[str, ...]
    match_hash: str
    state_transition_hash: str
    source_context_item_count: int
    active_context_item_count: int
    proof: Mapping[str, Any]
    candidate_only: bool = True
    recognition_only: bool = True

    def __post_init__(self) -> None:
        _identifier(self.projection_id, "recognition projection_id")
        _identifier(self.cue_hash, "recognition cue_hash")
        _identifier(self.memory_id, "recognition memory_id")
        for name, maximum in (
            ("active_memory_ids", MAX_MEMORIES),
            ("reactivated_memory_ids", MAX_MEMORIES),
            ("relevant_material_ids", MAX_CONTEXT_ITEMS),
            ("branch_ids", 8),
            ("capability_ids", 16),
            ("expanded_episode_ids", MAX_EPISODES),
        ):
            object.__setattr__(
                self,
                name,
                _items(getattr(self, name), f"recognition {name}", maximum, identifiers=True),
            )
        object.__setattr__(
            self,
            "semantic_context",
            _freeze(_bounded_json(self.semantic_context, "semantic context")),
        )
        _identifier(self.match_hash, "recognition match_hash")
        _identifier(self.state_transition_hash, "recognition state_transition_hash")
        for name in ("source_context_item_count", "active_context_item_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RecognitionMemoryError(f"{name} must be positive")
        if self.active_context_item_count >= self.source_context_item_count:
            raise RecognitionMemoryError("recognition projection did not reduce context")
        if self.candidate_only is not True or self.recognition_only is not True:
            raise RecognitionMemoryError("recognition projection authority is invalid")
        object.__setattr__(
            self,
            "proof",
            _freeze(
                _bounded_json(
                    self.proof,
                    "recognition proof",
                    maximum_items=MAX_MEMORY_SET_ITEMS + MAX_CUE_TOKENS,
                    maximum_bytes=MAX_MEMORY_SET_BYTES + 4_096,
                )
            ),
        )
        expected_projection = "recognition-" + _digest(
            {
                "cue_hash": self.cue_hash,
                "memory_id": self.memory_id,
                "reactivated_memory_ids": self.reactivated_memory_ids,
                "relevant_material_ids": self.relevant_material_ids,
                "branch_ids": self.branch_ids,
                "capability_ids": self.capability_ids,
                "match_hash": self.match_hash,
                "state_transition_hash": self.state_transition_hash,
            }
        )[:20]
        if self.projection_id != expected_projection:
            raise RecognitionMemoryError("recognition projection identity is invalid")

    def to_processing_context(self) -> dict[str, Any]:
        payload = {
            "cue_hash": self.cue_hash,
            "memory_id": self.memory_id,
            "active_memory_ids": list(self.active_memory_ids),
            "reactivated_memory_ids": list(self.reactivated_memory_ids),
            "relevant_material_ids": list(self.relevant_material_ids),
            "branch_ids": list(self.branch_ids),
            "capability_ids": list(self.capability_ids),
            "semantic_context": _jsonable(self.semantic_context),
            "expanded_episode_ids": list(self.expanded_episode_ids),
            "match_hash": self.match_hash,
            "state_transition_hash": self.state_transition_hash,
            "source_context_item_count": self.source_context_item_count,
            "active_context_item_count": self.active_context_item_count,
            "recognition_only": True,
        }
        result = {
            "projection_id": self.projection_id,
            "proof_hash": _digest(self.proof),
            "payload": json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ),
            "candidate_only": True,
        }
        if len(result["payload"]) > MAX_PROCESSING_CONTEXT_BYTES:
            raise RecognitionMemoryError(
                "recognition payload exceeds cognition string bound"
            )
        _bounded_json(
            result,
            "recognition processing context",
            maximum_items=MAX_MEMORY_SET_ITEMS + MAX_CUE_TOKENS + 64,
            maximum_bytes=MAX_PROCESSING_CONTEXT_BYTES + 1_024,
            maximum_depth=8,
        )
        return result

    def to_replay_proof(self) -> dict[str, Any]:
        return _jsonable(self.proof)


@dataclass(frozen=True)
class RecognitionResult:
    cue: RecognitionCue
    matches: tuple[RecognitionMatch, ...]
    selected_memory_id: str | None
    projection: RecognitionProjection | None
    expanded_episodes: tuple[CognitionEpisode, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cue": self.cue.to_dict(),
            "matches": [item.to_dict() for item in self.matches],
            "selected_memory_id": self.selected_memory_id,
            "projection": (
                self.projection.to_processing_context()
                if self.projection is not None
                else None
            ),
            "expanded_episode_ids": [item.episode_id for item in self.expanded_episodes],
        }


@dataclass(frozen=True)
class RecognitionMemorySet:
    episodes: tuple[CognitionEpisode, ...]
    memories: tuple[CompressedMemory, ...]

    def __post_init__(self) -> None:
        if len(self.episodes) > MAX_EPISODES or len(self.memories) > MAX_MEMORIES:
            raise RecognitionMemoryError("memory set exceeds its bound")
        if not all(isinstance(item, CognitionEpisode) for item in self.episodes):
            raise RecognitionMemoryError("memory set episodes are invalid")
        if not all(isinstance(item, CompressedMemory) for item in self.memories):
            raise RecognitionMemoryError("memory set memories are invalid")
        if len({item.episode_id for item in self.episodes}) != len(self.episodes):
            raise RecognitionMemoryError("episode IDs must be unique")
        if len({item.memory_id for item in self.memories}) != len(self.memories):
            raise RecognitionMemoryError("memory IDs must be unique")
        episode_ids = {item.episode_id for item in self.episodes}
        if any(
            reference.episode_id not in episode_ids
            for memory in self.memories
            for reference in memory.ancestry
        ):
            raise RecognitionMemoryError("memory ancestry references undeclared episode")
        _bounded_json(
            {
                "episodes": [item.to_dict() for item in self.episodes],
                "memories": [item.to_dict() for item in self.memories],
            },
            "memory set replay proof",
            maximum_items=MAX_MEMORY_SET_ITEMS,
            maximum_bytes=MAX_MEMORY_SET_BYTES,
        )
        episode_by_id = {item.episode_id: item for item in self.episodes}
        active_ids = [
            item.memory_id for item in self.memories
            if item.state is MemoryState.ACTIVE
        ]
        for memory in self.memories:
            expanded = [episode_by_id[item.episode_id] for item in memory.ancestry]
            active_count = (
                1 + len(memory.discriminative_features)
                + len(memory.supporting_features)
                + len(memory.structural_features)
                + len(memory.exceptions)
                + len(memory.applicability_scope)
            )
            source_count = sum(
                1 + len(item.detail) + len(item.statement) // 64
                for item in expanded
            )
            candidate_payload = {
                "cue_hash": "0" * 64,
                "memory_id": memory.memory_id,
                "active_memory_ids": active_ids,
                "reactivated_memory_ids": (
                    [memory.memory_id]
                    if memory.state in (MemoryState.DORMANT, MemoryState.DEFERRED)
                    else []
                ),
                "relevant_material_ids": list(memory.source_material_ids),
                "branch_ids": list(memory.branch_ids),
                "capability_ids": list(memory.capability_ids),
                "semantic_context": {
                    "summary": memory.summary,
                    "concept_id": memory.concept_id,
                    "concept_version": memory.concept_version,
                    "exceptions": list(memory.exceptions),
                    "applicability_scope": list(memory.applicability_scope),
                    "discriminative_features": list(memory.discriminative_features),
                    "expanded_episode_ids": [
                        item.episode_id for item in expanded
                    ],
                    "delayed_ticks": 1_000_000,
                },
                "expanded_episode_ids": [item.episode_id for item in expanded],
                "match_hash": "0" * 64,
                "state_transition_hash": "0" * 64,
                "source_context_item_count": max(source_count, active_count + 1),
                "active_context_item_count": active_count,
                "recognition_only": True,
            }
            candidate_payload_text = json.dumps(
                candidate_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            candidate_envelope = {
                "projection_id": "recognition-" + ("0" * 20),
                "proof_hash": "0" * 64,
                "payload": candidate_payload_text,
                "candidate_only": True,
            }
            if (
                len(candidate_payload_text) > MAX_PROCESSING_CONTEXT_BYTES
                or len(json.dumps(
                    candidate_envelope,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8"))
                > MAX_PROCESSING_CONTEXT_BYTES + 1_024
            ):
                raise RecognitionMemoryError(
                    "memory set cannot fit the cognition recognition envelope"
                )
        object.__setattr__(self, "episodes", tuple(self.episodes))
        object.__setattr__(self, "memories", tuple(self.memories))

    @classmethod
    def empty(cls) -> "RecognitionMemorySet":
        return cls((), ())

    def add_episodes(self, episodes: Iterable[CognitionEpisode]) -> "RecognitionMemorySet":
        combined = self.episodes + tuple(episodes)
        return RecognitionMemorySet(combined, self.memories)

    def add_memory(self, memory: CompressedMemory) -> "RecognitionMemorySet":
        return RecognitionMemorySet(self.episodes, self.memories + (memory,))

    def transition(
        self, memory_id: str, state: MemoryState
    ) -> "RecognitionMemorySet":
        if memory_id not in {item.memory_id for item in self.memories}:
            raise RecognitionMemoryError("cannot transition an unknown memory")
        return RecognitionMemorySet(
            self.episodes,
            tuple(
                CompressedMemory(
                    **{
                        **item.to_dict(),
                        "ancestry": tuple(item.ancestry),
                        "state": state,
                        "compression_hash": item.compression_hash,
                    }
                )
                if item.memory_id == memory_id
                else item
                for item in self.memories
            ),
        )

    def expand(self, memory_id: str) -> tuple[CognitionEpisode, ...]:
        memory = next(
            (item for item in self.memories if item.memory_id == memory_id), None
        )
        if memory is None:
            raise RecognitionMemoryError("cannot expand an unknown memory")
        episodes = {item.episode_id: item for item in self.episodes}
        result: list[CognitionEpisode] = []
        for reference in memory.ancestry:
            episode = episodes.get(reference.episode_id)
            if episode is None or episode.detail_hash != reference.detail_hash:
                raise RecognitionMemoryError("memory expansion provenance is invalid")
            if (
                episode.source_material_ids != reference.source_material_ids
                or episode.source_lineage_ids != reference.source_lineage_ids
            ):
                raise RecognitionMemoryError("memory expansion ancestry was altered")
            result.append(episode)
        return tuple(result)


def recognize(
    memory_set: RecognitionMemorySet,
    cue: RecognitionCue,
) -> RecognitionResult:
    if not isinstance(memory_set, RecognitionMemorySet):
        raise RecognitionMemoryError("recognition requires a memory set")
    matches = tuple(_match_memory(memory, cue) for memory in memory_set.memories)
    if len(matches) > MAX_MATCHES:
        raise RecognitionMemoryError("recognition matches exceed their bound")
    accepted = sorted(
        (item for item in matches if item.matched),
        key=lambda item: (-item.score, item.memory_id),
    )
    if not accepted:
        return RecognitionResult(cue, matches, None, None, ())
    if len(accepted) > 1 and accepted[0].score - accepted[1].score < MIN_MATCH_MARGIN:
        return RecognitionResult(cue, matches, None, None, ())
    selected_id = accepted[0].memory_id
    memory = next(item for item in memory_set.memories if item.memory_id == selected_id)
    expanded = memory_set.expand(selected_id)
    active_memory_ids = tuple(
        item.memory_id
        for item in memory_set.memories
        if item.state is MemoryState.ACTIVE
    )
    reactivated = (
        (selected_id,)
        if memory.state in (MemoryState.DORMANT, MemoryState.DEFERRED)
        else ()
    )
    relevant_materials = tuple(
        dict.fromkeys(item for item in memory.source_material_ids)
    )
    branch_ids = memory.branch_ids
    capability_ids = memory.capability_ids
    semantic_context = {
        "summary": memory.summary,
        "concept_id": memory.concept_id,
        "concept_version": memory.concept_version,
        "exceptions": list(memory.exceptions),
        "applicability_scope": list(memory.applicability_scope),
        "discriminative_features": list(memory.discriminative_features),
        "expanded_episode_ids": [item.episode_id for item in expanded],
        "delayed_ticks": cue.delayed_ticks,
    }
    source_count = sum(
        1 + len(item.detail) + len(item.statement) // 64 for item in expanded
    )
    active_count = (
        1
        + len(memory.discriminative_features)
        + len(memory.supporting_features)
        + len(memory.structural_features)
        + len(memory.exceptions)
        + len(memory.applicability_scope)
    )
    match_hash = _digest([item.to_dict() for item in matches])
    transition_hash = _digest(
        {
            "memory_id": selected_id,
            "from": memory.state.value,
            "to": MemoryState.ACTIVE.value if reactivated else memory.state.value,
            "cue_hash": cue.cue_hash,
        }
    )
    projection_id = "recognition-" + _digest(
        {
            "cue_hash": cue.cue_hash,
            "memory_id": selected_id,
            "reactivated_memory_ids": reactivated,
            "relevant_material_ids": relevant_materials,
            "branch_ids": branch_ids,
            "capability_ids": capability_ids,
            "match_hash": match_hash,
            "state_transition_hash": transition_hash,
        }
    )[:20]
    projection = RecognitionProjection(
        projection_id,
        cue.cue_hash,
        selected_id,
        active_memory_ids,
        reactivated,
        relevant_materials,
        branch_ids,
        capability_ids,
        semantic_context,
        tuple(item.episode_id for item in expanded),
        match_hash,
        transition_hash,
        max(source_count, active_count + 1),
        active_count,
        {
            "cue": cue.to_dict(),
            "episodes": [item.to_dict() for item in memory_set.episodes],
            "memories": [item.to_dict() for item in memory_set.memories],
        },
    )
    return RecognitionResult(cue, matches, selected_id, projection, expanded)


def reactivate(
    memory_set: RecognitionMemorySet,
    cue: RecognitionCue,
) -> tuple[RecognitionMemorySet, RecognitionResult]:
    result = recognize(memory_set, cue)
    if result.selected_memory_id is None or result.projection is None:
        return memory_set, result
    return (
        memory_set.transition(result.selected_memory_id, MemoryState.ACTIVE),
        result,
    )


def validate_recognition_context(
    value: Mapping[str, Any],
    proof: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecognitionMemoryError("recognition context must be a mapping")
    envelope = dict(value)
    if set(envelope) != {"projection_id", "proof_hash", "payload", "candidate_only"}:
        raise RecognitionMemoryError("recognition context envelope is invalid")
    if envelope["candidate_only"] is not True or not isinstance(
        envelope["payload"], str
    ):
        raise RecognitionMemoryError("recognition context envelope authority is invalid")
    try:
        payload = json.loads(envelope["payload"])
    except (TypeError, ValueError) as exc:
        raise RecognitionMemoryError("recognition context payload is invalid") from exc
    value = {
        "projection_id": envelope["projection_id"],
        "proof_hash": envelope["proof_hash"],
        "candidate_only": True,
        **payload,
    }
    expected = {
        "projection_id",
        "cue_hash",
        "memory_id",
        "active_memory_ids",
        "reactivated_memory_ids",
        "relevant_material_ids",
        "branch_ids",
        "capability_ids",
        "semantic_context",
        "expanded_episode_ids",
        "match_hash",
        "state_transition_hash",
        "source_context_item_count",
        "active_context_item_count",
        "proof_hash",
        "candidate_only",
        "recognition_only",
    }
    if set(value) != expected:
        raise RecognitionMemoryError("recognition context schema is invalid")
    projection = RecognitionProjection(
        value["projection_id"],
        value["cue_hash"],
        value["memory_id"],
        tuple(value["active_memory_ids"]),
        tuple(value["reactivated_memory_ids"]),
        tuple(value["relevant_material_ids"]),
        tuple(value["branch_ids"]),
        tuple(value["capability_ids"]),
        value["semantic_context"],
        tuple(value["expanded_episode_ids"]),
        value["match_hash"],
        value["state_transition_hash"],
        value["source_context_item_count"],
        value["active_context_item_count"],
        proof,
        value["candidate_only"],
        value["recognition_only"],
    )
    if value["proof_hash"] != _digest(proof):
        raise RecognitionMemoryError("recognition proof commitment is invalid")
    if not isinstance(proof, Mapping) or set(proof) != {"cue", "episodes", "memories"}:
        raise RecognitionMemoryError("recognition proof schema is invalid")
    cue_data = proof["cue"]
    cue = RecognitionCue(
        cue_data["cue_id"],
        tuple(cue_data["tokens"]),
        tuple(cue_data["structural_features"]),
        cue_data["concept_id"],
        cue_data["concept_version"],
        cue_data["branch_id"],
        tuple(cue_data["scope"]),
        cue_data["delayed_ticks"],
    )
    episodes = tuple(
        CognitionEpisode(
            item["episode_id"], item["problem_id"], item["branch_id"],
            item["concept_id"], item["concept_version"], item["semantic_job"],
            item["candidate_id"], item["statement"], item["rationale"], item["detail"],
            tuple(item["discriminative_features"]), tuple(item["supporting_features"]),
            tuple(item["structural_features"]), tuple(item["exceptions"]),
            tuple(item["applicability_scope"]), tuple(item["source_material_ids"]),
            tuple(item["source_lineage_ids"]), item["provenance"],
        )
        for item in proof["episodes"]
    )
    memories = []
    for item in proof["memories"]:
        ancestry = tuple(
            ExpansionReference(
                ref["episode_id"], ref["detail_hash"],
                tuple(ref["source_material_ids"]), tuple(ref["source_lineage_ids"]),
            )
            for ref in item["ancestry"]
        )
        memories.append(CompressedMemory(
            item["memory_id"], item["concept_id"], item["concept_version"],
            item["summary"], tuple(item["discriminative_features"]),
            tuple(item["supporting_features"]), tuple(item["structural_features"]),
            tuple(item["exceptions"]), tuple(item["applicability_scope"]),
            tuple(item["branch_ids"]), tuple(item["capability_ids"]),
            tuple(item["source_material_ids"]), tuple(item["source_lineage_ids"]),
            ancestry, item["state"], item["detailed_item_count"],
            item["active_context_item_count"], item["compression_hash"],
        ))
    memory_set = RecognitionMemorySet(episodes, tuple(memories))
    memory = next(
        (item for item in memory_set.memories if item.memory_id == value["memory_id"]),
        None,
    )
    if memory is None:
        raise RecognitionMemoryError("recognition proof omits selected memory")
    expanded = memory_set.expand(memory.memory_id)
    expected_matches = tuple(
        _match_memory(item, cue) for item in memory_set.memories
    )
    proof_matches = [item.to_dict() for item in expected_matches]
    if _digest(proof_matches) != value["match_hash"]:
        raise RecognitionMemoryError("recognition proof match binding is invalid")
    accepted = sorted(
        (item for item in expected_matches if item.matched),
        key=lambda item: (-item.score, item.memory_id),
    )
    if (
        not accepted
        or accepted[0].memory_id != memory.memory_id
        or (
            len(accepted) > 1
            and accepted[0].score - accepted[1].score < MIN_MATCH_MARGIN
        )
    ):
        raise RecognitionMemoryError("recognition proof selection is invalid")
    reactivated = (
        (memory.memory_id,)
        if memory.state in (MemoryState.DORMANT, MemoryState.DEFERRED)
        else ()
    )
    expected_transition = _digest(
        {
            "memory_id": memory.memory_id,
            "from": memory.state.value,
            "to": MemoryState.ACTIVE.value if reactivated else memory.state.value,
            "cue_hash": cue.cue_hash,
        }
    )
    expected_active = tuple(
        item.memory_id for item in memory_set.memories
        if item.state is MemoryState.ACTIVE
    )
    expected_semantic = {
        "summary": memory.summary,
        "concept_id": memory.concept_id,
        "concept_version": memory.concept_version,
        "exceptions": list(memory.exceptions),
        "applicability_scope": list(memory.applicability_scope),
        "discriminative_features": list(memory.discriminative_features),
        "expanded_episode_ids": [item.episode_id for item in expanded],
        "delayed_ticks": cue.delayed_ticks,
    }
    expected_source_count = sum(
        1 + len(item.detail) + len(item.statement) // 64 for item in expanded
    )
    expected_active_count = (
        1 + len(memory.discriminative_features)
        + len(memory.supporting_features) + len(memory.structural_features)
        + len(memory.exceptions) + len(memory.applicability_scope)
    )
    if (
        value["cue_hash"] != cue.cue_hash
        or value["memory_id"] != memory.memory_id
        or tuple(value["active_memory_ids"]) != expected_active
        or tuple(value["reactivated_memory_ids"]) != reactivated
        or tuple(value["relevant_material_ids"]) != memory.source_material_ids
        or tuple(value["branch_ids"]) != memory.branch_ids
        or tuple(value["capability_ids"]) != memory.capability_ids
        or tuple(value["expanded_episode_ids"])
        != tuple(item.episode_id for item in expanded)
        or value["state_transition_hash"] != expected_transition
        or value["semantic_context"] != expected_semantic
        or value["source_context_item_count"]
        != max(expected_source_count, expected_active_count + 1)
        or value["active_context_item_count"] != expected_active_count
    ):
        raise RecognitionMemoryError("recognition proof projection binding is invalid")
    expanded = dict(value)
    expanded["_serialized_context"] = envelope
    return _freeze(expanded)


__all__ = [
    "CognitionEpisode",
    "CompressedMemory",
    "ExpansionReference",
    "MAX_CUE_TOKENS",
    "MAX_EPISODES",
    "MAX_FEATURES",
    "MAX_MEMORIES",
    "MemoryState",
    "MIN_MATCH_MARGIN",
    "MIN_MATCH_SCORE",
    "RecognitionCue",
    "RecognitionMatch",
    "RecognitionMemoryError",
    "RecognitionMemorySet",
    "RecognitionProjection",
    "RecognitionResult",
    "compress_episodes",
    "reactivate",
    "recognize",
    "validate_recognition_context",
]