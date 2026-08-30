"""Bounded external-reality observations for the Kraken-R candidate layer.

This module deliberately accepts caller-supplied retrieval records only.  It
does not search, fetch, cache, persist, settle, execute, or promote anything.
The internet therefore feeds the epistemic graph rather than bypassing it:
retrieval can revise a candidate claim, but it cannot manufacture evidence,
truth, requirement satisfaction, or adaptive learning.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .cognition_kernel import (
    CognitionValidationError,
    EpistemicClaim,
    EpistemicStatus,
    VerificationObligation,
    VerificationPressure,
)


MAX_EVIDENCE_NEEDS = 16
MAX_EXTERNAL_SOURCES = 24
MAX_CLAIM_ASSESSMENTS = 48
MAX_REVISIONS = 24
MAX_SOURCE_PARENTS = 8
MAX_SOURCE_CONTENT = 24_000
MAX_SOURCE_SCOPE_ITEMS = 16
MAX_EXTERNAL_PAYLOAD_ITEMS = 96
MAX_EXTERNAL_PAYLOAD_BYTES = 64_000
_TRUSTED_SOURCE_VERIFIERS = MappingProxyType(
    {
        "kraken-r-qualification-verifier": (
            "1ZdCfds/dYv7JQYaz6/jm5O1wu/4QrZZZOGcBR7C5t4="
        ),
    }
)


class ExternalRealityError(CognitionValidationError):
    """Raised when a retrieval contract is malformed or inapplicable."""


class SourceRole(str, Enum):
    DISCOVERY = "discovery"
    EVIDENTIARY = "evidentiary"


class ClaimRelation(str, Enum):
    ENTAILS = "entails"
    CONTRADICTS = "contradicts"
    INSUFFICIENT = "insufficient"


_FORBIDDEN_AUTHORITY_KEYS = frozenset(
    {
        "authority",
        "evidence",
        "evidence_id",
        "evidence_ids",
        "execution",
        "execution_truth",
        "ground_truth",
        "learning",
        "learning_update",
        "adaptive_credit",
        "adaptive_update",
        "settlement",
        "settlement_id",
        "truth",
        "verified",
    }
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def source_attestation_payload(
    *,
    artifact_id: str,
    source_id: str,
    source_kind: str,
    independence_family: str,
    root_artifact_id: str,
    parent_artifact_ids: Iterable[str],
    retrieved_on: str,
    effective_from: str | None,
    effective_until: str | None,
    jurisdiction: Iterable[str],
    scopes: Iterable[str],
    regimes: Iterable[str],
    method: str,
    content_hash: str,
) -> bytes:
    value = {
        "artifact_id": artifact_id,
        "source_id": source_id,
        "source_kind": source_kind,
        "independence_family": independence_family,
        "root_artifact_id": root_artifact_id,
        "parent_artifact_ids": list(parent_artifact_ids),
        "retrieved_on": retrieved_on,
        "effective_from": effective_from,
        "effective_until": effective_until,
        "jurisdiction": list(jurisdiction),
        "scopes": list(scopes),
        "regimes": list(regimes),
        "method": method,
        "content_hash": content_hash,
    }
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _identifier(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(character.isspace() for character in value)
    ):
        raise ExternalRealityError(
            f"{name} must be a non-empty identifier without whitespace"
        )
    return value


def _text(value: Any, name: str, maximum: int = 4_096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExternalRealityError(f"{name} must be non-empty text")
    if len(value) > maximum:
        raise ExternalRealityError(f"{name} exceeds its bounded length")
    return value


def _bounded_text(value: Any, name: str, maximum: int = 4_096) -> str:
    if not isinstance(value, str):
        raise ExternalRealityError(f"{name} must be text")
    if len(value) > maximum:
        raise ExternalRealityError(f"{name} exceeds its bounded length")
    return value


def _ids(value: Iterable[str], name: str, maximum: int, *, allow_empty: bool = True) -> tuple[str, ...]:
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ExternalRealityError(f"{name} must be iterable") from exc
    if len(values) > maximum or (not allow_empty and not values):
        raise ExternalRealityError(f"{name} exceeds its bounded limit or is empty")
    result = tuple(_identifier(item, name) for item in values)
    if len(set(result)) != len(result):
        raise ExternalRealityError(f"{name} cannot contain duplicate identities")
    return result


def _texts(value: Iterable[str], name: str, maximum: int, *, maximum_length: int = 512) -> tuple[str, ...]:
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ExternalRealityError(f"{name} must be iterable") from exc
    if len(values) > maximum:
        raise ExternalRealityError(f"{name} exceeds its bounded limit")
    result = tuple(_text(item, name, maximum_length) for item in values)
    if len(set(result)) != len(result):
        raise ExternalRealityError(f"{name} cannot contain duplicate values")
    return result


def _date(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ExternalRealityError(f"{name} must be an ISO date")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ExternalRealityError(f"{name} must be an ISO date") from exc
    return value


def _optional_date(value: Any, name: str) -> str | None:
    return None if value is None else _date(value, name)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _bounded_payload(value: Any, name: str) -> Any:
    count = 0

    def visit(item: Any, depth: int, path: str) -> None:
        nonlocal count
        if depth > 8:
            raise ExternalRealityError(f"{name} exceeds payload depth")
        if item is None or isinstance(item, (bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ExternalRealityError(f"{name} contains a non-finite number")
            return
        if isinstance(item, str):
            if len(item) > 8_192:
                raise ExternalRealityError(f"{path} contains oversized text")
            return
        if isinstance(item, Mapping):
            count += len(item)
            if len(item) > MAX_EXTERNAL_PAYLOAD_ITEMS or count > MAX_EXTERNAL_PAYLOAD_ITEMS:
                raise ExternalRealityError(f"{name} exceeds payload item limit")
            for key, child in item.items():
                if not isinstance(key, str) or not key or len(key) > 128:
                    raise ExternalRealityError(f"{path} contains an invalid key")
                if key.lower() in _FORBIDDEN_AUTHORITY_KEYS:
                    raise ExternalRealityError(f"{path} contains forbidden authority field {key}")
                visit(child, depth + 1, f"{path}.{key}")
            return
        if isinstance(item, (tuple, list)):
            count += len(item)
            if len(item) > MAX_EXTERNAL_PAYLOAD_ITEMS or count > MAX_EXTERNAL_PAYLOAD_ITEMS:
                raise ExternalRealityError(f"{name} exceeds payload item limit")
            for index, child in enumerate(item):
                visit(child, depth + 1, f"{path}[{index}]")
            return
        raise ExternalRealityError(f"{name} contains unsupported data")

    visit(value, 0, name)
    raw = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if len(raw.encode("utf-8")) > MAX_EXTERNAL_PAYLOAD_BYTES:
        raise ExternalRealityError(f"{name} exceeds payload byte limit")
    return value


@dataclass(frozen=True)
class EvidenceNeed:
    """A bounded request to test one unresolved load-bearing claim."""

    need_id: str
    problem_id: str
    claim_id: str
    reason: str
    verification_pressure: VerificationPressure
    required_precision: str = "claim-level"
    currentness_required: bool = False
    max_age_days: int | None = None
    required_jurisdictions: tuple[str, ...] = ()
    required_scopes: tuple[str, ...] = ()
    required_regimes: tuple[str, ...] = ()
    required_source_kinds: tuple[str, ...] = ()
    required_methods: tuple[str, ...] = ()
    downstream_leverage: tuple[str, ...] = ()
    satisfying_conditions: tuple[str, ...] = ()
    contradicting_conditions: tuple[str, ...] = ()
    load_bearing: bool = True
    candidate_only: bool = True

    def __post_init__(self) -> None:
        _identifier(self.need_id, "need_id")
        _identifier(self.problem_id, "problem_id")
        _identifier(self.claim_id, "claim_id")
        _text(self.reason, "need reason", 2_048)
        if not isinstance(self.verification_pressure, VerificationPressure):
            raise ExternalRealityError("verification_pressure is invalid")
        _identifier(self.required_precision, "required_precision")
        if self.max_age_days is not None:
            if isinstance(self.max_age_days, bool) or not isinstance(self.max_age_days, int) or self.max_age_days < 0 or self.max_age_days > 3_650:
                raise ExternalRealityError("max_age_days is outside its bounded range")
        if self.currentness_required and self.max_age_days is None:
            raise ExternalRealityError(
                "currentness-required evidence needs require max_age_days"
            )
        for value, name, limit in (
            (self.required_jurisdictions, "required_jurisdictions", 8),
            (self.required_scopes, "required_scopes", MAX_SOURCE_SCOPE_ITEMS),
            (self.required_regimes, "required_regimes", MAX_SOURCE_SCOPE_ITEMS),
            (self.required_source_kinds, "required_source_kinds", 8),
            (self.required_methods, "required_methods", 8),
            (self.downstream_leverage, "downstream_leverage", MAX_SOURCE_SCOPE_ITEMS),
            (self.satisfying_conditions, "satisfying_conditions", 8),
            (self.contradicting_conditions, "contradicting_conditions", 8),
        ):
            object.__setattr__(
                self,
                name,
                _texts(value, name, limit),
            )
        if not self.load_bearing:
            raise ExternalRealityError("evidence needs are restricted to load-bearing claims")
        if not self.candidate_only:
            raise ExternalRealityError("evidence needs must remain candidate-only")

    def to_dict(self) -> dict[str, Any]:
        return {
            "need_id": self.need_id,
            "problem_id": self.problem_id,
            "claim_id": self.claim_id,
            "reason": self.reason,
            "verification_pressure": self.verification_pressure.to_dict(),
            "required_precision": self.required_precision,
            "currentness_required": self.currentness_required,
            "max_age_days": self.max_age_days,
            "required_jurisdictions": list(self.required_jurisdictions),
            "required_scopes": list(self.required_scopes),
            "required_regimes": list(self.required_regimes),
            "required_source_kinds": list(self.required_source_kinds),
            "required_methods": list(self.required_methods),
            "downstream_leverage": list(self.downstream_leverage),
            "satisfying_conditions": list(self.satisfying_conditions),
            "contradicting_conditions": list(self.contradicting_conditions),
            "load_bearing": self.load_bearing,
            "candidate_only": True,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceNeed":
        expected = {
            "need_id", "problem_id", "claim_id", "reason",
            "verification_pressure", "required_precision",
            "currentness_required", "max_age_days", "required_jurisdictions",
            "required_scopes", "required_regimes", "required_source_kinds",
            "required_methods", "downstream_leverage", "satisfying_conditions",
            "contradicting_conditions", "load_bearing", "candidate_only",
        }
        if set(value) != expected:
            raise ExternalRealityError("evidence need schema is invalid")
        pressure = value["verification_pressure"]
        return cls(
            value["need_id"],
            value["problem_id"],
            value["claim_id"],
            value["reason"],
            VerificationPressure(
                pressure["stakes"],
                pressure["volatility"],
                pressure["locality"],
                pressure["precision"],
                pressure["novelty"],
                pressure["contestability"],
                pressure["actionability"],
            ),
            value["required_precision"],
            value["currentness_required"],
            value["max_age_days"],
            tuple(value["required_jurisdictions"]),
            tuple(value["required_scopes"]),
            tuple(value["required_regimes"]),
            tuple(value["required_source_kinds"]),
            tuple(value["required_methods"]),
            tuple(value["downstream_leverage"]),
            tuple(value["satisfying_conditions"]),
            tuple(value["contradicting_conditions"]),
            value["load_bearing"],
            value["candidate_only"],
        )


@dataclass(frozen=True)
class SourceArtifact:
    """A retrieved artifact with explicit role, ancestry, and applicability."""

    artifact_id: str
    source_id: str
    source_kind: str
    role: SourceRole
    independence_family: str
    root_artifact_id: str
    parent_artifact_ids: tuple[str, ...]
    retrieved_on: str
    effective_from: str | None
    effective_until: str | None
    jurisdiction: tuple[str, ...]
    scopes: tuple[str, ...]
    regimes: tuple[str, ...]
    method: str
    content: str
    content_hash: str
    verifier_id: str | None = None
    attestation_signature: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    candidate_only: bool = True

    def __post_init__(self) -> None:
        for value, name in (
            (self.artifact_id, "artifact_id"),
            (self.source_id, "source_id"),
            (self.independence_family, "independence_family"),
            (self.root_artifact_id, "root_artifact_id"),
            (self.method, "method"),
        ):
            _identifier(value, name)
        _identifier(self.source_kind, "source_kind")
        object.__setattr__(self, "role", SourceRole(self.role))
        object.__setattr__(self, "parent_artifact_ids", _ids(self.parent_artifact_ids, "parent_artifact_ids", MAX_SOURCE_PARENTS))
        if self.artifact_id in self.parent_artifact_ids:
            raise ExternalRealityError("source artifact cannot be its own parent")
        object.__setattr__(self, "retrieved_on", _date(self.retrieved_on, "retrieved_on"))
        object.__setattr__(self, "effective_from", _optional_date(self.effective_from, "effective_from"))
        object.__setattr__(self, "effective_until", _optional_date(self.effective_until, "effective_until"))
        if self.effective_from and self.effective_until and self.effective_from > self.effective_until:
            raise ExternalRealityError("source effective dates are reversed")
        for value, name, limit in (
            (self.jurisdiction, "jurisdiction", 8),
            (self.scopes, "scopes", MAX_SOURCE_SCOPE_ITEMS),
            (self.regimes, "regimes", MAX_SOURCE_SCOPE_ITEMS),
        ):
            object.__setattr__(self, name, tuple(_identifier(item, name) for item in _ids(value, name, limit)))
        _text(self.content, "source content", MAX_SOURCE_CONTENT)
        _identifier(self.content_hash, "content_hash")
        if self.content_hash != _digest(self.content):
            raise ExternalRealityError("source content hash is invalid")
        if self.source_kind.lower() in {"search_result", "search_snippet", "snippet", "index", "discovery"} and self.role is SourceRole.EVIDENTIARY:
            raise ExternalRealityError("discovery source cannot be promoted to evidentiary")
        if self.role is SourceRole.EVIDENTIARY:
            if (
                self.verifier_id not in _TRUSTED_SOURCE_VERIFIERS
                or not isinstance(self.attestation_signature, str)
            ):
                raise ExternalRealityError(
                    "evidentiary source requires a trusted verifier attestation"
                )
            payload = source_attestation_payload(
                artifact_id=self.artifact_id,
                source_id=self.source_id,
                source_kind=self.source_kind,
                independence_family=self.independence_family,
                root_artifact_id=self.root_artifact_id,
                parent_artifact_ids=self.parent_artifact_ids,
                retrieved_on=self.retrieved_on,
                effective_from=self.effective_from,
                effective_until=self.effective_until,
                jurisdiction=self.jurisdiction,
                scopes=self.scopes,
                regimes=self.regimes,
                method=self.method,
                content_hash=self.content_hash,
            )
            try:
                public_key = base64.b64decode(
                    _TRUSTED_SOURCE_VERIFIERS[self.verifier_id],
                    validate=True,
                )
                signature = base64.b64decode(
                    self.attestation_signature, validate=True
                )
                Ed25519PublicKey.from_public_bytes(public_key).verify(
                    signature, payload
                )
            except (ValueError, InvalidSignature) as exc:
                raise ExternalRealityError(
                    "evidentiary source attestation is invalid"
                ) from exc
        elif self.verifier_id is not None or self.attestation_signature is not None:
            raise ExternalRealityError(
                "discovery source cannot carry evidentiary attestation"
            )
        if not self.candidate_only:
            raise ExternalRealityError("source artifacts must remain candidate-only")
        provenance = _bounded_payload(self.provenance, "source provenance")
        if (
            not provenance
            or not isinstance(provenance.get("retrieval"), str)
            or not provenance["retrieval"].strip()
        ):
            raise ExternalRealityError(
                "source provenance requires a retrieval method record"
            )
        object.__setattr__(self, "provenance", _freeze(provenance))

    def applicable_to(self, need: EvidenceNeed, as_of: str) -> tuple[bool, str]:
        as_of = _date(as_of, "as_of")
        if need.required_source_kinds and self.source_kind not in need.required_source_kinds:
            return False, "source kind does not satisfy the evidence need"
        if need.required_methods and self.method not in need.required_methods:
            return False, "source method does not satisfy the evidence need"
        if need.required_jurisdictions and not set(need.required_jurisdictions).intersection(self.jurisdiction):
            return False, "source jurisdiction is not applicable"
        if need.required_scopes and not set(need.required_scopes).issubset(self.scopes):
            return False, "source scope is not applicable"
        if need.required_regimes and not set(need.required_regimes).issubset(self.regimes):
            return False, "source regime is not applicable"
        if self.effective_from and as_of < self.effective_from:
            return False, "source is not yet effective"
        if self.effective_until and as_of > self.effective_until:
            return False, "source is no longer effective"
        if need.currentness_required:
            age = (date.fromisoformat(as_of) - date.fromisoformat(self.retrieved_on)).days
            if age < 0 or (need.max_age_days is not None and age > need.max_age_days):
                return False, "source is stale for the required currentness"
        return True, "source is applicable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "role": self.role.value,
            "independence_family": self.independence_family,
            "root_artifact_id": self.root_artifact_id,
            "parent_artifact_ids": list(self.parent_artifact_ids),
            "retrieved_on": self.retrieved_on,
            "effective_from": self.effective_from,
            "effective_until": self.effective_until,
            "jurisdiction": list(self.jurisdiction),
            "scopes": list(self.scopes),
            "regimes": list(self.regimes),
            "method": self.method,
            "content": self.content,
            "content_hash": self.content_hash,
            "verifier_id": self.verifier_id,
            "attestation_signature": self.attestation_signature,
            "provenance": _jsonable(self.provenance),
            "candidate_only": True,
        }

    @classmethod
    def create(
        cls,
        artifact_id: str,
        source_id: str,
        source_kind: str,
        role: SourceRole | str,
        independence_family: str,
        *,
        root_artifact_id: str | None = None,
        parent_artifact_ids: Iterable[str] = (),
        retrieved_on: str,
        effective_from: str | None = None,
        effective_until: str | None = None,
        jurisdiction: Iterable[str] = (),
        scopes: Iterable[str] = (),
        regimes: Iterable[str] = (),
        method: str,
        content: str,
        verifier_id: str | None = None,
        attestation_signature: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> "SourceArtifact":
        return cls(
            artifact_id,
            source_id,
            source_kind,
            SourceRole(role),
            independence_family,
            root_artifact_id or artifact_id,
            tuple(parent_artifact_ids),
            retrieved_on,
            effective_from,
            effective_until,
            tuple(jurisdiction),
            tuple(scopes),
            tuple(regimes),
            method,
            content,
            _digest(content),
            verifier_id,
            attestation_signature,
            provenance or {},
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceArtifact":
        expected = {
            "artifact_id", "source_id", "source_kind", "role",
            "independence_family", "root_artifact_id", "parent_artifact_ids",
            "retrieved_on", "effective_from", "effective_until",
            "jurisdiction", "scopes", "regimes", "method", "content",
            "content_hash", "verifier_id", "attestation_signature",
            "provenance", "candidate_only",
        }
        if set(value) != expected:
            raise ExternalRealityError("source artifact schema is invalid")
        return cls(
            value["artifact_id"],
            value["source_id"],
            value["source_kind"],
            value["role"],
            value["independence_family"],
            value["root_artifact_id"],
            tuple(value["parent_artifact_ids"]),
            value["retrieved_on"],
            value["effective_from"],
            value["effective_until"],
            tuple(value["jurisdiction"]),
            tuple(value["scopes"]),
            tuple(value["regimes"]),
            value["method"],
            value["content"],
            value["content_hash"],
            value["verifier_id"],
            value["attestation_signature"],
            value.get("provenance", {}),
            value["candidate_only"],
        )


@dataclass(frozen=True)
class ClaimAssessment:
    """One exact claim/source relation, bound to a quote in the artifact."""

    assessment_id: str
    need_id: str
    claim_id: str
    artifact_id: str
    relation: ClaimRelation
    claim_hash: str
    quoted_text: str
    rationale: str
    candidate_only: bool = True

    def __post_init__(self) -> None:
        for value, name in (
            (self.assessment_id, "assessment_id"),
            (self.need_id, "need_id"),
            (self.claim_id, "claim_id"),
            (self.artifact_id, "artifact_id"),
            (self.claim_hash, "claim_hash"),
        ):
            _identifier(value, name)
        object.__setattr__(self, "relation", ClaimRelation(self.relation))
        if self.relation is ClaimRelation.INSUFFICIENT:
            _bounded_text(self.quoted_text, "quoted_text", 8_192)
        else:
            _text(self.quoted_text, "quoted_text", 8_192)
        _text(self.rationale, "assessment rationale", 2_048)
        if not self.candidate_only:
            raise ExternalRealityError("claim assessments must remain candidate-only")

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "need_id": self.need_id,
            "claim_id": self.claim_id,
            "artifact_id": self.artifact_id,
            "relation": self.relation.value,
            "claim_hash": self.claim_hash,
            "quoted_text": self.quoted_text,
            "rationale": self.rationale,
            "candidate_only": True,
        }

    @classmethod
    def create(
        cls,
        assessment_id: str,
        need_id: str,
        claim: EpistemicClaim,
        artifact_id: str,
        relation: ClaimRelation | str,
        quoted_text: str,
        rationale: str,
    ) -> "ClaimAssessment":
        return cls(
            assessment_id,
            need_id,
            claim.claim_id,
            artifact_id,
            ClaimRelation(relation),
            _digest(claim.statement),
            quoted_text,
            rationale,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClaimAssessment":
        expected = {
            "assessment_id", "need_id", "claim_id", "artifact_id",
            "relation", "claim_hash", "quoted_text", "rationale",
            "candidate_only",
        }
        if set(value) != expected:
            raise ExternalRealityError("claim assessment schema is invalid")
        return cls(
            value["assessment_id"],
            value["need_id"],
            value["claim_id"],
            value["artifact_id"],
            value["relation"],
            value["claim_hash"],
            value["quoted_text"],
            value["rationale"],
            value["candidate_only"],
        )


@dataclass(frozen=True)
class RetrievalObservation:
    """Immutable bounded retrieval output; it is not evidence."""

    observation_id: str
    request_id: str
    problem_id: str
    as_of: str
    need_ids: tuple[str, ...]
    sources: tuple[SourceArtifact, ...]
    assessments: tuple[ClaimAssessment, ...]
    discovery_artifact_ids: tuple[str, ...] = ()
    candidate_only: bool = True

    def __post_init__(self) -> None:
        for value, name in (
            (self.observation_id, "observation_id"),
            (self.request_id, "request_id"),
            (self.problem_id, "problem_id"),
        ):
            _identifier(value, name)
        object.__setattr__(self, "as_of", _date(self.as_of, "as_of"))
        object.__setattr__(self, "need_ids", _ids(self.need_ids, "need_ids", MAX_EVIDENCE_NEEDS, allow_empty=False))
        sources = tuple(self.sources)
        assessments = tuple(self.assessments)
        if len(sources) > MAX_EXTERNAL_SOURCES or not sources or len(assessments) > MAX_CLAIM_ASSESSMENTS:
            raise ExternalRealityError("retrieval observation exceeds its bounded record limits")
        if not all(isinstance(item, SourceArtifact) for item in sources):
            raise ExternalRealityError("retrieval sources are invalid")
        if not all(isinstance(item, ClaimAssessment) for item in assessments):
            raise ExternalRealityError("retrieval assessments are invalid")
        if len({item.artifact_id for item in sources}) != len(sources):
            raise ExternalRealityError("retrieval artifact identities must be unique")
        if len({item.assessment_id for item in assessments}) != len(assessments):
            raise ExternalRealityError("retrieval assessment identities must be unique")
        identity_bindings: dict[str, tuple[str, str]] = {}
        for source in sources:
            binding = (
                source.independence_family,
                source.root_artifact_id,
            )
            prior_binding = identity_bindings.setdefault(
                source.source_id, binding
            )
            if prior_binding != binding:
                raise ExternalRealityError(
                    "one source identity cannot claim conflicting "
                    "independence families or root lineages"
                )
        source_by_id = {item.artifact_id: item for item in sources}
        for source in sources:
            if source.root_artifact_id not in source_by_id and source.root_artifact_id != source.artifact_id:
                raise ExternalRealityError("source root lineage is undeclared")
            if set(source.parent_artifact_ids) - set(source_by_id):
                raise ExternalRealityError("source parent lineage is undeclared")
            if source.parent_artifact_ids:
                parents = [source_by_id[item] for item in source.parent_artifact_ids]
                if any(parent.independence_family != source.independence_family for parent in parents):
                    raise ExternalRealityError("source lineage changes independence family")
                if any(parent.root_artifact_id != source.root_artifact_id for parent in parents):
                    raise ExternalRealityError("source lineage changes root identity")
            if source.artifact_id == source.root_artifact_id and source.parent_artifact_ids:
                raise ExternalRealityError("root source cannot have a parent")
        def visit(source_id: str, path: frozenset[str]) -> None:
            if source_id in path:
                raise ExternalRealityError("source ancestry contains a cycle")
            for parent_id in source_by_id[source_id].parent_artifact_ids:
                visit(parent_id, path | {source_id})

        for source in sources:
            visit(source.artifact_id, frozenset())
        object.__setattr__(self, "discovery_artifact_ids", _ids(self.discovery_artifact_ids, "discovery_artifact_ids", MAX_EXTERNAL_SOURCES))
        if set(self.discovery_artifact_ids) - set(source_by_id):
            raise ExternalRealityError("discovery artifact is undeclared")
        if any(source_by_id[item].role is not SourceRole.DISCOVERY for item in self.discovery_artifact_ids):
            raise ExternalRealityError("evidentiary source cannot be labeled discovery")
        for assessment in assessments:
            source = source_by_id.get(assessment.artifact_id)
            if source is None:
                raise ExternalRealityError("assessment artifact is undeclared")
            if source.role is not SourceRole.EVIDENTIARY:
                raise ExternalRealityError("discovery source cannot support a claim assessment")
            if assessment.relation is not ClaimRelation.INSUFFICIENT and assessment.quoted_text not in source.content:
                raise ExternalRealityError("claim assessment quote is not present in source content")
        if not self.candidate_only:
            raise ExternalRealityError("retrieval observations must remain candidate-only")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "assessments", assessments)

    @property
    def retrieval_hash(self) -> str:
        return _digest(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "observation_id": self.observation_id,
            "request_id": self.request_id,
            "problem_id": self.problem_id,
            "as_of": self.as_of,
            "need_ids": list(self.need_ids),
            "sources": [item.to_dict() for item in self.sources],
            "assessments": [item.to_dict() for item in self.assessments],
            "discovery_artifact_ids": list(self.discovery_artifact_ids),
            "candidate_only": True,
        }
        if include_hash:
            result["retrieval_hash"] = self.retrieval_hash
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RetrievalObservation":
        expected = {
            "observation_id", "request_id", "problem_id", "as_of", "need_ids",
            "sources", "assessments", "discovery_artifact_ids", "candidate_only",
            "retrieval_hash",
        }
        if set(value) != expected:
            raise ExternalRealityError("retrieval observation schema is invalid")
        observation = cls(
            value["observation_id"],
            value["request_id"],
            value["problem_id"],
            value["as_of"],
            tuple(value["need_ids"]),
            tuple(SourceArtifact.from_dict(item) for item in value["sources"]),
            tuple(ClaimAssessment.from_dict(item) for item in value["assessments"]),
            tuple(value["discovery_artifact_ids"]),
            value["candidate_only"],
        )
        if value["retrieval_hash"] != observation.retrieval_hash:
            raise ExternalRealityError("retrieval observation hash mismatch")
        return observation


@dataclass(frozen=True)
class CandidateEpistemicRevision:
    revision_id: str
    observation_id: str
    claim_id: str
    prior_status: EpistemicStatus
    revised_status: EpistemicStatus
    supporting_artifact_ids: tuple[str, ...]
    contradicting_artifact_ids: tuple[str, ...]
    counted_independence_families: tuple[str, ...]
    prior_claim_hash: str
    revised_claim_hash: str
    candidate_only: bool = True

    def __post_init__(self) -> None:
        for value, name in (
            (self.revision_id, "revision_id"),
            (self.observation_id, "observation_id"),
            (self.claim_id, "claim_id"),
            (self.prior_claim_hash, "prior_claim_hash"),
            (self.revised_claim_hash, "revised_claim_hash"),
        ):
            _identifier(value, name)
        object.__setattr__(self, "prior_status", EpistemicStatus(self.prior_status))
        object.__setattr__(self, "revised_status", EpistemicStatus(self.revised_status))
        object.__setattr__(self, "supporting_artifact_ids", _ids(self.supporting_artifact_ids, "supporting_artifact_ids", MAX_EXTERNAL_SOURCES))
        object.__setattr__(self, "contradicting_artifact_ids", _ids(self.contradicting_artifact_ids, "contradicting_artifact_ids", MAX_EXTERNAL_SOURCES))
        object.__setattr__(self, "counted_independence_families", _ids(self.counted_independence_families, "counted_independence_families", MAX_EXTERNAL_SOURCES))
        if not self.candidate_only:
            raise ExternalRealityError("epistemic revisions must remain candidate-only")

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "observation_id": self.observation_id,
            "claim_id": self.claim_id,
            "prior_status": self.prior_status.value,
            "revised_status": self.revised_status.value,
            "supporting_artifact_ids": list(self.supporting_artifact_ids),
            "contradicting_artifact_ids": list(self.contradicting_artifact_ids),
            "counted_independence_families": list(self.counted_independence_families),
            "prior_claim_hash": self.prior_claim_hash,
            "revised_claim_hash": self.revised_claim_hash,
            "candidate_only": True,
        }

    @property
    def independent_source_count(self) -> int:
        return len(self.counted_independence_families)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateEpistemicRevision":
        expected = {
            "revision_id", "observation_id", "claim_id", "prior_status",
            "revised_status", "supporting_artifact_ids",
            "contradicting_artifact_ids", "counted_independence_families",
            "prior_claim_hash", "revised_claim_hash", "candidate_only",
        }
        if set(value) != expected:
            raise ExternalRealityError("candidate revision schema is invalid")
        return cls(
            value["revision_id"],
            value["observation_id"],
            value["claim_id"],
            value["prior_status"],
            value["revised_status"],
            tuple(value["supporting_artifact_ids"]),
            tuple(value["contradicting_artifact_ids"]),
            tuple(value["counted_independence_families"]),
            value["prior_claim_hash"],
            value["revised_claim_hash"],
            value["candidate_only"],
        )


def derive_evidence_needs(
    problem_id: str,
    claims: Iterable[EpistemicClaim],
    obligations: Iterable[VerificationObligation] = (),
) -> tuple[EvidenceNeed, ...]:
    """Derive retrieval pressure without treating every claim as retrieval-worthy."""

    _identifier(problem_id, "problem_id")
    claims = tuple(claims)
    if len(claims) > 48:
        raise ExternalRealityError("claims exceed the evidence-need bound")
    obligation_by_id = {item.obligation_id: item for item in obligations}
    needs: list[EvidenceNeed] = []
    for claim in claims:
        if claim.problem_id != problem_id or claim.status not in (EpistemicStatus.PROPOSAL, EpistemicStatus.UNRESOLVED):
            continue
        pressure = claim.verification_pressure
        load_bearing = (
            pressure.score >= 0.35
            or pressure.stakes >= 0.60
            or pressure.actionability >= 0.60
            or bool(claim.unresolved_ids)
            or bool(claim.falsification_conditions)
        )
        if not load_bearing:
            continue
        reason_parts = ["unresolved candidate claim"]
        if pressure.stakes >= 0.60:
            reason_parts.append("high stakes")
        if pressure.volatility >= 0.50:
            reason_parts.append("volatile")
        if pressure.locality >= 0.50:
            reason_parts.append("local applicability")
        if pressure.precision >= 0.50:
            reason_parts.append("claim-level precision")
        if claim.unresolved_ids:
            reason_parts.append("open proof obligations")
        if any(
            obligation_by_id.get(item) is not None
            and obligation_by_id[item].kind.value in {"regime_applicability", "temporal_ordering", "stale_or_correlated_source"}
            for item in claim.unresolved_ids
        ):
            reason_parts.append("currentness or regime check")
        need = EvidenceNeed(
            need_id=f"{claim.claim_id}-external-need",
            problem_id=problem_id,
            claim_id=claim.claim_id,
            reason="; ".join(reason_parts),
            verification_pressure=pressure,
            required_precision="claim-level",
            currentness_required=pressure.volatility >= 0.50,
            max_age_days=30 if pressure.volatility >= 0.50 else None,
            downstream_leverage=claim.unresolved_ids or (claim.use_site,),
            satisfying_conditions=tuple(item.statement for item in claim.falsification_conditions[:4]),
            contradicting_conditions=tuple(item.discriminator for item in claim.falsification_conditions[:4]),
        )
        needs.append(need)
    if len(needs) > MAX_EVIDENCE_NEEDS:
        raise ExternalRealityError("derived evidence needs exceed the bounded limit")
    return tuple(needs)


def validate_evidence_needs(
    problem_id: str,
    claims: Iterable[EpistemicClaim],
    needs: Iterable[EvidenceNeed],
) -> tuple[EvidenceNeed, ...]:
    """Require every caller-refined need to bind a canonically eligible claim."""

    claim_by_id = {item.claim_id: item for item in claims}
    validated = tuple(needs)
    for need in validated:
        claim = claim_by_id.get(need.claim_id)
        if (
            need.problem_id != problem_id
            or claim is None
            or claim.problem_id != problem_id
            or claim.status
            not in (EpistemicStatus.UNRESOLVED, EpistemicStatus.PROPOSAL)
            or not (
                claim.verification_pressure.score >= 0.35
                or claim.verification_pressure.stakes >= 0.60
                or claim.verification_pressure.actionability >= 0.60
                or bool(claim.unresolved_ids)
                or bool(claim.falsification_conditions)
            )
        ):
            raise ExternalRealityError(
                "evidence need does not bind an unresolved load-bearing claim"
            )
    return validated


def validate_retrieval_observation(
    observation: RetrievalObservation,
    claims: Iterable[EpistemicClaim],
    needs: Iterable[EvidenceNeed],
    *,
    as_of: str | None = None,
) -> None:
    """Validate exact claim bindings and applicability before revision."""

    if not isinstance(observation, RetrievalObservation):
        raise ExternalRealityError("retrieval observation is invalid")
    claims_by_id = {item.claim_id: item for item in claims}
    needs_by_id = {item.need_id: item for item in needs}
    if observation.problem_id not in {item.problem_id for item in claims_by_id.values()} and claims_by_id:
        raise ExternalRealityError("retrieval observation is bound to another problem")
    if set(observation.need_ids) - set(needs_by_id):
        raise ExternalRealityError("retrieval observation names an undeclared evidence need")
    source_by_id = {item.artifact_id: item for item in observation.sources}
    if as_of is not None and _date(as_of, "as_of") != observation.as_of:
        raise ExternalRealityError("retrieval observation currentness date is not replay-bound")
    for assessment in observation.assessments:
        if assessment.need_id not in observation.need_ids:
            raise ExternalRealityError(
                "retrieval assessment references a need outside "
                "the observation scope"
            )
        need = needs_by_id.get(assessment.need_id)
        claim = claims_by_id.get(assessment.claim_id)
        source = source_by_id.get(assessment.artifact_id)
        if need is None or claim is None or source is None:
            raise ExternalRealityError("retrieval assessment has an undeclared binding")
        if need.claim_id != claim.claim_id or claim.problem_id != observation.problem_id:
            raise ExternalRealityError("retrieval assessment claim binding is invalid")
        if assessment.claim_hash != _digest(claim.statement):
            raise ExternalRealityError("retrieval assessment claim hash is invalid")
        applicable, reason = source.applicable_to(need, observation.as_of)
        if not applicable and assessment.relation is not ClaimRelation.INSUFFICIENT:
            raise ExternalRealityError(f"retrieval source is inapplicable: {reason}")
    for need_id in observation.need_ids:
        if not any(item.need_id == need_id for item in observation.assessments):
            raise ExternalRealityError("each retrieval need must have a claim-level disposition")


def _revise_candidate_claims(
    claims: Iterable[EpistemicClaim],
    needs: Iterable[EvidenceNeed],
    observation: RetrievalObservation,
    operation_request: Any,
    capability: Any,
    projection: Any,
    decision: Any,
) -> tuple[tuple[EpistemicClaim, ...], tuple[CandidateEpistemicRevision, ...]]:
    """Apply one bounded epistemic projection; this is not adaptive learning."""

    from .cognition_kernel import (
        AdaptiveProcessingProjection,
        OperationRequest,
        ProcessingCapability,
        ProcessingDecision,
        ProcessingOperation,
    )

    if not (
        isinstance(operation_request, OperationRequest)
        and isinstance(capability, ProcessingCapability)
        and isinstance(projection, AdaptiveProcessingProjection)
        and isinstance(decision, ProcessingDecision)
        and operation_request.operation is ProcessingOperation.RETRIEVE
        and capability.operation is ProcessingOperation.RETRIEVE
        and operation_request.request_id == observation.request_id
        and operation_request.capability_id == capability.capability_id
        and operation_request.branch_id == capability.branch_id
        and capability.capability_id in projection.available_capability_ids
        and decision.projection_id == projection.projection_id
        and decision.selected_capability_id == capability.capability_id
        and decision.operation is ProcessingOperation.RETRIEVE
        and decision.branch_id == capability.branch_id
    ):
        raise ExternalRealityError(
            "candidate revision requires its topology-selected retrieve request"
        )
    claims = tuple(claims)
    needs = tuple(needs)
    validate_retrieval_observation(observation, claims, needs)
    source_by_id = {item.artifact_id: item for item in observation.sources}
    need_by_id = {item.need_id: item for item in needs}
    assessments_by_claim: dict[str, list[ClaimAssessment]] = {}
    for assessment in observation.assessments:
        assessments_by_claim.setdefault(assessment.claim_id, []).append(assessment)
    revised: list[EpistemicClaim] = []
    revisions: list[CandidateEpistemicRevision] = []
    for claim in claims:
        assessments = assessments_by_claim.get(claim.claim_id, [])
        if not assessments:
            revised.append(claim)
            continue
        supporting: list[ClaimAssessment] = []
        contradicting: list[ClaimAssessment] = []
        families: dict[str, ClaimAssessment] = {}
        for assessment in assessments:
            source = source_by_id[assessment.artifact_id]
            need = need_by_id[assessment.need_id]
            applicable, _ = source.applicable_to(need, observation.as_of)
            if not applicable:
                continue
            if assessment.relation is ClaimRelation.ENTAILS:
                supporting.append(assessment)
                families.setdefault(source.independence_family, assessment)
            elif assessment.relation is ClaimRelation.CONTRADICTS:
                contradicting.append(assessment)
                families.setdefault(source.independence_family, assessment)
        if not supporting and not contradicting:
            revised.append(claim)
            continue
        new_status = (
            EpistemicStatus.CONTRADICTION
            if contradicting and not supporting
            else EpistemicStatus.SUPPORT
            if supporting and not contradicting
            else EpistemicStatus.UNRESOLVED
        )
        updated = replace(
            claim,
            status=new_status,
            competence_origin=claim.competence_origin,
            evidence_ids=(),
            declared_only=True,
        )
        revision = CandidateEpistemicRevision(
            revision_id=f"{observation.observation_id}-{claim.claim_id}-revision",
            observation_id=observation.observation_id,
            claim_id=claim.claim_id,
            prior_status=claim.status,
            revised_status=new_status,
            supporting_artifact_ids=tuple(item.artifact_id for item in supporting),
            contradicting_artifact_ids=tuple(item.artifact_id for item in contradicting),
            counted_independence_families=tuple(sorted(families)),
            prior_claim_hash=_digest(claim.to_dict()),
            revised_claim_hash=_digest(updated.to_dict()),
        )
        revised.append(updated)
        revisions.append(revision)
    if len(revisions) > MAX_REVISIONS:
        raise ExternalRealityError("epistemic revisions exceed their bounded limit")
    return tuple(revised), tuple(revisions)


def revise_candidate_claims(
    trace: Any,
) -> tuple[tuple[EpistemicClaim, ...], tuple[CandidateEpistemicRevision, ...]]:
    """Expose revisions only after full processing-trace validation."""

    from .cognition_kernel import ProcessingTrace, replay_processing_trace

    if not isinstance(trace, ProcessingTrace):
        raise ExternalRealityError(
            "candidate revision requires a validated processing trace"
        )
    validated = replay_processing_trace(trace)
    if not validated.retrieval_observation:
        raise ExternalRealityError("processing trace has no retrieval revision")
    return validated.claims, validated.candidate_revisions


def replay_retrieval_observation(
    trace: Any,
) -> tuple[tuple[EpistemicClaim, ...], tuple[CandidateEpistemicRevision, ...]]:
    """Replay a complete bound trace without retrieval or network access."""

    return revise_candidate_claims(trace)


def _fixture_source(
    artifact_id: str,
    source_id: str,
    source_kind: str,
    family: str,
    content: str,
    *,
    role: SourceRole = SourceRole.EVIDENTIARY,
    retrieved_on: str = "2026-08-30",
    effective_from: str | None = None,
    effective_until: str | None = None,
    jurisdiction: tuple[str, ...] = ("global",),
    scopes: tuple[str, ...] = ("general",),
    regimes: tuple[str, ...] = ("default",),
    method: str = "primary-record",
    attestation_signature: str,
) -> SourceArtifact:
    return SourceArtifact.create(
        artifact_id,
        source_id,
        source_kind,
        role,
        family,
        retrieved_on=retrieved_on,
        effective_from=effective_from,
        effective_until=effective_until,
        jurisdiction=jurisdiction,
        scopes=scopes,
        regimes=regimes,
        method=method,
        content=content,
        verifier_id="kraken-r-qualification-verifier",
        attestation_signature=attestation_signature,
        provenance={"retrieval": "fixture"},
    )


def legal_currentness_fixture() -> tuple[EvidenceNeed, RetrievalObservation]:
    """Small fixture where an expired rule cannot satisfy a current-law need."""

    claim = EpistemicClaim(
        "legal-fixture-claim",
        "legal-fixture-problem",
        "The current rule permits the declared operation.",
        verification_pressure=VerificationPressure(stakes=0.9, volatility=0.9, precision=0.9, actionability=0.8),
    )
    need = EvidenceNeed(
        "legal-fixture-need",
        claim.problem_id,
        claim.claim_id,
        "Current legal text is load-bearing and volatile.",
        claim.verification_pressure,
        currentness_required=True,
        max_age_days=30,
        required_jurisdictions=("fictionland",),
        required_scopes=("operation-x",),
        required_source_kinds=("official_regulation",),
        required_methods=("primary-record",),
    )
    source = _fixture_source(
        "legal-fixture-source",
        "legal-fixture-source-id",
        "official_regulation",
        "legal-root",
        "The current rule permits the declared operation.",
        jurisdiction=("fictionland",),
        scopes=("operation-x",),
        effective_from="2026-01-01",
        attestation_signature=(
            "4+tOPEA+un+H8InYwh/fpb4j6xUDosNpXwPD2frXaZfRuVbwkQbHg2DWMH0zX"
            "kR9gqbKoRz1ZOKZaylrIUnzCA=="
        ),
    )
    observation = RetrievalObservation(
        "legal-fixture-observation",
        "legal-fixture-request",
        claim.problem_id,
        "2026-08-30",
        (need.need_id,),
        (source,),
        (
            ClaimAssessment(
                "legal-fixture-assessment",
                need.need_id,
                claim.claim_id,
                source.artifact_id,
                ClaimRelation.ENTAILS,
                _digest(claim.statement),
                claim.statement,
                "The effective official text states the claim.",
            ),
        ),
    )
    return need, observation


def scientific_regime_fixture() -> tuple[EvidenceNeed, RetrievalObservation]:
    """Small fixture where a result outside the declared regime is rejected."""

    claim = EpistemicClaim(
        "science-fixture-claim",
        "science-fixture-problem",
        "The intervention is effective in regime-a.",
        verification_pressure=VerificationPressure(stakes=0.7, precision=0.9, contestability=0.8),
    )
    need = EvidenceNeed(
        "science-fixture-need",
        claim.problem_id,
        claim.claim_id,
        "The claimed effect is regime-dependent.",
        claim.verification_pressure,
        required_regimes=("regime-a",),
        required_scopes=("intervention-x",),
        required_source_kinds=("peer_reviewed",),
        required_methods=("controlled-study",),
    )
    source = _fixture_source(
        "science-fixture-source",
        "science-fixture-source-id",
        "peer_reviewed",
        "study-family",
        "The intervention is effective in regime-a.",
        scopes=("intervention-x",),
        regimes=("regime-a",),
        method="controlled-study",
        attestation_signature=(
            "Jx0Jv/cOd4DlsF3Fxb2F6Nfe6S3dYzdYdEfa+U8xZlsdsz27xNWkcZxlKnu+X"
            "Bik2HuWd9HoKgK4KdlbA9YBAw=="
        ),
    )
    observation = RetrievalObservation(
        "science-fixture-observation",
        "science-fixture-request",
        claim.problem_id,
        "2026-08-30",
        (need.need_id,),
        (source,),
        (
            ClaimAssessment(
                "science-fixture-assessment",
                need.need_id,
                claim.claim_id,
                source.artifact_id,
                ClaimRelation.ENTAILS,
                _digest(claim.statement),
                claim.statement,
                "The study's declared regime matches the claim.",
            ),
        ),
    )
    return need, observation


__all__ = [
    "CandidateEpistemicRevision",
    "ClaimAssessment",
    "ClaimRelation",
    "EvidenceNeed",
    "ExternalRealityError",
    "MAX_CLAIM_ASSESSMENTS",
    "MAX_EVIDENCE_NEEDS",
    "MAX_EXTERNAL_SOURCES",
    "RetrievalObservation",
    "SourceArtifact",
    "SourceRole",
    "derive_evidence_needs",
    "legal_currentness_fixture",
    "replay_retrieval_observation",
    "revise_candidate_claims",
    "scientific_regime_fixture",
    "source_attestation_payload",
    "validate_retrieval_observation",
]