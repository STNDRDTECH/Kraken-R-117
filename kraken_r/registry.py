"""Read-only architecture registry for the candidate Kraken-R foundation."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


class ArchitectureRegistryError(ValueError):
    """Raised when a registry file cannot be interpreted."""


class RegistryStatus(str, Enum):
    ACTIVE = "active"
    PARTIAL = "partial"
    DORMANT = "dormant"
    DESIGN_ONLY = "design_only"
    ALTERNATE = "alternate"
    DUPLICATE = "duplicate"
    MISSING = "missing"
    EXPERIMENTAL = "experimental"
    INFRASTRUCTURE = "infrastructure"


class RegistryAuthority(str, Enum):
    """Descriptive labels; a live Kraken-R authority is intentionally absent."""

    LEGACY_REFERENCE = "legacy_reference"
    KRAKEN_CANDIDATE = "kraken_candidate"
    DOMAIN_ADAPTER = "domain_adapter"
    DOCUMENTATION_ONLY = "documentation_only"
    NONE = "none"


class RegistryDisposition(str, Enum):
    KEEP = "keep"
    MERGE = "merge"
    QUARANTINE = "quarantine"
    REBUILD = "rebuild"
    LEAVE_EXPERIMENTAL = "leave_experimental"


REQUIRED_CATEGORIES = (
    "core_execution",
    "task_state_reasoning",
    "evidence_settlement",
    "memory_learning",
    "emergence_routing",
    "homeostasis_health",
    "governance_safety",
    "evolution_mutation",
    "specialized_processes",
    "j4h_semantic_review",
    "domain_adapters",
    "tools_execution",
    "operator_surfaces",
    "alternate_legacy",
    "design_assets",
    "distinctive_concepts",
)

_MECHANISM_FIELDS = {
    "mechanism_id",
    "category",
    "name",
    "intended_role",
    "current_status",
    "dependencies",
    "authority",
    "evidence_grade",
    "evidence_refs",
    "duplicates",
    "disposition",
    "source_refs",
    "planned",
    "notes",
}
_REGISTRY_FIELDS = {
    "registry_kind",
    "registry_id",
    "version",
    "last_reviewed",
    "authority_boundary",
    "mechanisms",
}
_AUTHORITY_BOUNDARY_FIELDS = {
    "legacy_reference",
    "kraken_candidate",
    "preservation",
}


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArchitectureRegistryError(f"{field_name} must be a non-empty string")
    return value


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ArchitectureRegistryError(f"{field_name} must be a list of strings")
    return tuple(value)


@dataclass(frozen=True)
class MechanismRecord:
    """One descriptive architecture-registry entry."""

    mechanism_id: str
    category: str
    name: str
    intended_role: str
    current_status: RegistryStatus
    dependencies: tuple[str, ...]
    authority: RegistryAuthority
    evidence_grade: str
    evidence_refs: tuple[str, ...]
    duplicates: tuple[str, ...]
    disposition: RegistryDisposition
    source_refs: tuple[str, ...]
    planned: bool
    notes: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MechanismRecord":
        if not isinstance(raw, dict):
            raise ArchitectureRegistryError("Mechanism record must be an object")
        missing = sorted(_MECHANISM_FIELDS - set(raw))
        if missing:
            raise ArchitectureRegistryError(
                f"Mechanism record is missing required fields: {', '.join(missing)}"
            )
        unknown = sorted(set(raw) - _MECHANISM_FIELDS)
        if unknown:
            raise ArchitectureRegistryError(
                f"Mechanism record has unknown fields: {', '.join(unknown)}"
            )
        if type(raw["planned"]) is not bool:
            raise ArchitectureRegistryError("planned must be a boolean")
        try:
            return cls(
                mechanism_id=_nonempty_string(raw["mechanism_id"], "mechanism_id"),
                category=_nonempty_string(raw["category"], "category"),
                name=_nonempty_string(raw["name"], "name"),
                intended_role=_nonempty_string(raw["intended_role"], "intended_role"),
                current_status=RegistryStatus(raw["current_status"]),
                dependencies=_string_list(raw["dependencies"], "dependencies"),
                authority=RegistryAuthority(raw["authority"]),
                evidence_grade=_nonempty_string(
                    raw["evidence_grade"], "evidence_grade"
                ),
                evidence_refs=_string_list(raw["evidence_refs"], "evidence_refs"),
                duplicates=_string_list(raw["duplicates"], "duplicates"),
                disposition=RegistryDisposition(raw["disposition"]),
                source_refs=_string_list(raw["source_refs"], "source_refs"),
                planned=raw["planned"],
                notes=raw["notes"] if isinstance(raw["notes"], str) else _nonempty_string(raw["notes"], "notes"),
            )
        except (TypeError, ValueError) as exc:
            raise ArchitectureRegistryError(
                f"Invalid mechanism record "
                f"{raw.get('mechanism_id', '<unknown>')!r}: {exc}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "category": self.category,
            "name": self.name,
            "intended_role": self.intended_role,
            "current_status": self.current_status.value,
            "dependencies": list(self.dependencies),
            "authority": self.authority.value,
            "evidence_grade": self.evidence_grade,
            "evidence_refs": list(self.evidence_refs),
            "duplicates": list(self.duplicates),
            "disposition": self.disposition.value,
            "source_refs": list(self.source_refs),
            "planned": self.planned,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ValidationReport:
    """Deterministic inspection result for one registry."""

    ok: bool
    errors: tuple[str, ...] = ()
    mechanism_count: int = 0
    category_count: int = 0
    planned_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "mechanism_count": self.mechanism_count,
            "category_count": self.category_count,
            "planned_count": self.planned_count,
        }


class ArchitectureRegistry:
    """In-memory registry that never writes or mutates runtime state."""

    registry_kind = "kraken_r_architecture"
    registry_version = "1.0"

    def __init__(
        self,
        mechanisms: Iterable[MechanismRecord],
        *,
        registry_id: str = "kraken-r-foundation",
        version: str = registry_version,
        last_reviewed: str = "unreviewed",
        authority_boundary: Mapping[str, str] | None = None,
    ) -> None:
        self.registry_id = registry_id
        self.version = version
        self.last_reviewed = last_reviewed
        self.authority_boundary = dict(authority_boundary or {})
        self._mechanisms = {record.mechanism_id: record for record in mechanisms}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArchitectureRegistry":
        if not isinstance(payload, dict):
            raise ArchitectureRegistryError("Registry root must be an object")
        missing = sorted(_REGISTRY_FIELDS - set(payload))
        if missing:
            raise ArchitectureRegistryError(
                f"Registry is missing required fields: {', '.join(missing)}"
            )
        unknown = sorted(set(payload) - _REGISTRY_FIELDS)
        if unknown:
            raise ArchitectureRegistryError(
                f"Registry has unknown fields: {', '.join(unknown)}"
            )
        if payload.get("registry_kind") != cls.registry_kind:
            raise ArchitectureRegistryError(
                f"Expected registry_kind={cls.registry_kind!r}"
            )
        registry_id = _nonempty_string(payload["registry_id"], "registry_id")
        version = _nonempty_string(payload["version"], "version")
        last_reviewed = _nonempty_string(payload["last_reviewed"], "last_reviewed")
        try:
            date.fromisoformat(last_reviewed)
        except ValueError as exc:
            raise ArchitectureRegistryError(
                "last_reviewed must be an ISO-8601 date"
            ) from exc
        authority_boundary = payload["authority_boundary"]
        if not isinstance(authority_boundary, dict):
            raise ArchitectureRegistryError("authority_boundary must be an object")
        boundary_missing = sorted(_AUTHORITY_BOUNDARY_FIELDS - set(authority_boundary))
        boundary_unknown = sorted(set(authority_boundary) - _AUTHORITY_BOUNDARY_FIELDS)
        if boundary_missing or boundary_unknown:
            details = []
            if boundary_missing:
                details.append("missing " + ", ".join(boundary_missing))
            if boundary_unknown:
                details.append("unknown " + ", ".join(boundary_unknown))
            raise ArchitectureRegistryError(
                "authority_boundary fields invalid: " + "; ".join(details)
            )
        checked_boundary = {
            key: _nonempty_string(authority_boundary[key], f"authority_boundary.{key}")
            for key in _AUTHORITY_BOUNDARY_FIELDS
        }
        raw_mechanisms = payload.get("mechanisms")
        if not isinstance(raw_mechanisms, list):
            raise ArchitectureRegistryError("mechanisms must be a list")
        records = [MechanismRecord.from_dict(raw) for raw in raw_mechanisms]
        duplicate_ids = _duplicates(record.mechanism_id for record in records)
        if duplicate_ids:
            raise ArchitectureRegistryError(
                f"Duplicate mechanism IDs: {', '.join(sorted(duplicate_ids))}"
            )
        return cls(
            records,
            registry_id=registry_id,
            version=version,
            last_reviewed=last_reviewed,
            authority_boundary=checked_boundary,
        )

    def all(self) -> tuple[MechanismRecord, ...]:
        return tuple(self._mechanisms.values())

    def get(self, mechanism_id: str) -> MechanismRecord | None:
        return self._mechanisms.get(mechanism_id)

    def by_category(self, category: str) -> tuple[MechanismRecord, ...]:
        return tuple(
            record
            for record in self._mechanisms.values()
            if record.category == category
        )

    def validate(self) -> ValidationReport:
        errors: list[str] = []
        records = self._mechanisms
        categories = {record.category for record in records.values()}

        if not self.registry_id.strip():
            errors.append("registry_id must be non-empty")
        if not self.version.strip():
            errors.append("version must be non-empty")
        try:
            date.fromisoformat(self.last_reviewed)
        except (TypeError, ValueError):
            errors.append("last_reviewed must be an ISO-8601 date")
        for key in _AUTHORITY_BOUNDARY_FIELDS:
            value = self.authority_boundary.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"authority_boundary.{key} must be a non-empty string")
        if not records:
            errors.append("registry must contain at least one mechanism")

        missing_categories = sorted(set(REQUIRED_CATEGORIES) - categories)
        if missing_categories:
            errors.append(
                "Missing required census categories: " + ", ".join(missing_categories)
            )

        for record in records.values():
            prefix = f"{record.mechanism_id}:"
            if not record.mechanism_id.strip():
                errors.append(f"{prefix} mechanism_id must be non-empty")
            if record.category not in REQUIRED_CATEGORIES:
                errors.append(f"{prefix} unknown category {record.category!r}")
            if not record.name.strip():
                errors.append(f"{prefix} name must be non-empty")
            if not record.intended_role.strip():
                errors.append(f"{prefix} intended_role must be non-empty")

            for dependency in record.dependencies:
                if dependency not in records:
                    errors.append(f"{prefix} unknown dependency {dependency!r}")
                if dependency == record.mechanism_id:
                    errors.append(f"{prefix} cannot depend on itself")
            for duplicate in record.duplicates:
                if duplicate not in records:
                    errors.append(f"{prefix} unknown duplicate {duplicate!r}")
                if duplicate == record.mechanism_id:
                    errors.append(f"{prefix} cannot duplicate itself")

            if record.planned and record.evidence_grade in {"operational", "grounded"}:
                errors.append(
                    f"{prefix} planned capability cannot claim "
                    f"{record.evidence_grade} evidence"
                )
            if record.planned and record.authority == RegistryAuthority.LEGACY_REFERENCE:
                errors.append(f"{prefix} planned capability cannot have legacy authority")

        errors.extend(self._dependency_cycle_errors())
        errors.extend(self._duplicate_symmetry_errors())
        return ValidationReport(
            ok=not errors,
            errors=tuple(errors),
            mechanism_count=len(records),
            category_count=len(categories),
            planned_count=sum(record.planned for record in records.values()),
        )

    def _dependency_cycle_errors(self) -> list[str]:
        indegree = {mechanism_id: 0 for mechanism_id in self._mechanisms}
        dependents: dict[str, list[str]] = defaultdict(list)
        for record in self._mechanisms.values():
            for dependency in record.dependencies:
                if dependency in self._mechanisms:
                    indegree[record.mechanism_id] += 1
                    dependents[dependency].append(record.mechanism_id)

        queue = deque(
            mechanism_id for mechanism_id, degree in indegree.items() if degree == 0
        )
        processed = 0
        while queue:
            mechanism_id = queue.popleft()
            processed += 1
            for dependent in dependents[mechanism_id]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    queue.append(dependent)
        if processed == len(self._mechanisms):
            return []
        cycle_nodes = sorted(
            mechanism_id for mechanism_id, degree in indegree.items() if degree > 0
        )
        return ["Dependency cycle detected: " + ", ".join(cycle_nodes)]

    def _duplicate_symmetry_errors(self) -> list[str]:
        errors: list[str] = []
        for record in self._mechanisms.values():
            for duplicate_id in record.duplicates:
                counterpart = self._mechanisms.get(duplicate_id)
                if counterpart and record.mechanism_id not in counterpart.duplicates:
                    errors.append(
                        f"{record.mechanism_id}: duplicate relationship with "
                        f"{duplicate_id} is not reciprocal"
                    )
        return errors


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def load_registry(path: str | Path) -> ArchitectureRegistry:
    """Load a registry without importing or touching runtime components."""

    registry_path = Path(path)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchitectureRegistryError(
            f"Could not load registry {registry_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ArchitectureRegistryError("Registry root must be an object")
    return ArchitectureRegistry.from_dict(payload)


def load_default_registry() -> ArchitectureRegistry:
    return load_registry(Path(__file__).with_name("architecture_registry.json"))


__all__ = [
    "ArchitectureRegistry",
    "ArchitectureRegistryError",
    "MechanismRecord",
    "REQUIRED_CATEGORIES",
    "RegistryAuthority",
    "RegistryDisposition",
    "RegistryStatus",
    "ValidationReport",
    "load_default_registry",
    "load_registry",
]