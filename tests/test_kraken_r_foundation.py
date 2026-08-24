"""Focused acceptance tests for the isolated Kraken-R foundation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from kraken_r import (
    Action,
    ArchitectureRegistry,
    ArchitectureRegistryError,
    Authority,
    Capability,
    ContractValidationError,
    Decision,
    Event,
    Evidence,
    EvidenceGrade,
    ExecutionResult,
    GroundTruth,
    Hypothesis,
    LearningUpdate,
    Lineage,
    Mutation,
    Objective,
    Plan,
    Regression,
    RegistryAuthority,
    RegistryDisposition,
    RegistryStatus,
    Settlement,
    Signal,
    State,
    TaskState,
    load_default_registry,
)
from kraken_r.validate import (
    CANONICAL_CONTRACTS,
    ConstitutionValidationError,
    load_constitution_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


def test_all_canonical_contracts_are_importable_and_serializable() -> None:
    assert set(CANONICAL_CONTRACTS) == {
        "Objective", "TaskState", "Plan", "Hypothesis", "Signal", "Event",
        "Action", "ExecutionResult", "Evidence", "GroundTruth", "Decision",
        "Settlement", "LearningUpdate", "Capability", "Authority", "Mutation",
        "Lineage", "Regression",
    }
    objective = Objective("obj-1", "Improve a bounded behavior")
    contracts = [
        objective,
        TaskState("state-1", objective.objective_id, 1, "proposed"),
        Plan("plan-1", objective.objective_id),
        Hypothesis("hyp-1", objective.objective_id, "The bounded change helps"),
        Signal("sig-1", "observation", "corr-1", "test"),
        Action("act-1", objective.objective_id, "inspect", "artifact"),
        ExecutionResult("exec-1", "act-1", "completed"),
        Evidence("ev-1", "exec-1", EvidenceGrade.OPERATIONAL, "focused-test"),
        Decision("dec-1", objective.objective_id, "accept", "Observed evidence supports it"),
        Settlement("set-1", "dec-1", "success", "success"),
        LearningUpdate("learn-1", "set-1", "capability", "cap-1"),
        Capability("cap-1", "bounded inspection"),
        Mutation("mut-1", "propose"),
        Lineage("lin-1", "mut-1"),
        Regression("reg-1", "mut-1", "baseline-1", "pass"),
    ]
    assert all(isinstance(item.to_dict(), dict) for item in contracts)
    assert State is TaskState
    assert Event is Signal
    assert GroundTruth is Evidence


def test_contract_records_defensively_freeze_nested_inputs() -> None:
    objective_input = {"origin": {"label": "fixture"}}
    state_input = {"nested": {"value": 1}}
    objective = Objective("obj-immutable", "Preserve evidence", provenance=objective_input)
    state = TaskState("state-immutable", "obj-immutable", 1, "observed", state_input)
    objective_input["origin"]["label"] = "mutated outside"
    state_input["nested"]["value"] = 2
    assert objective.provenance["origin"]["label"] == "fixture"
    assert state.values["nested"]["value"] == 1
    with pytest.raises(TypeError):
        objective.provenance["origin"] = "changed"
    with pytest.raises(TypeError):
        state.values["nested"]["value"] = 3
    assert objective.to_dict()["provenance"] == {"origin": {"label": "fixture"}}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TaskState("state-invalid", "obj-1", 1, "new", authority="unapproved"),
        lambda: Signal("signal-invalid", "topic", "corr", "test", authority="unapproved"),
        lambda: Action("action-invalid", "obj-1", "inspect", "target", authority="unapproved"),
        lambda: Decision("decision-invalid", "obj-1", "accept", "reason", authority="unapproved"),
        lambda: Capability("capability-invalid", "name", authority="unapproved"),
        lambda: Mutation("mutation-invalid", "propose", authority="unapproved"),
        lambda: Signal("signal-grade-invalid", "topic", "corr", "test", evidence_grade="fabricated"),
        lambda: Evidence("evidence-grade-invalid", "subject", "fabricated", "test"),
        lambda: Capability("capability-grade-invalid", "name", evidence_grade="fabricated"),
    ],
)
def test_contracts_reject_invalid_authority_and_evidence_grade(factory) -> None:
    with pytest.raises(ContractValidationError):
        factory()


def test_default_registry_covers_all_census_categories_and_validates() -> None:
    registry = load_default_registry()
    report = registry.validate()
    assert report.ok, report.errors
    assert report.category_count == 16
    assert report.mechanism_count >= 40
    assert report.planned_count >= 10
    assert {record.category for record in registry.all()} == {
        "core_execution", "task_state_reasoning", "evidence_settlement",
        "memory_learning", "emergence_routing", "homeostasis_health",
        "governance_safety", "evolution_mutation", "specialized_processes",
        "j4h_semantic_review", "domain_adapters", "tools_execution",
        "operator_surfaces", "alternate_legacy", "design_assets",
        "distinctive_concepts",
    }


def test_registry_records_authority_evidence_dependencies_and_dispositions() -> None:
    registry = load_default_registry()
    records = registry.all()
    assert all(record.authority for record in records)
    assert all(record.evidence_grade for record in records)
    assert all(record.disposition for record in records)
    assert any(record.dependencies for record in records)
    assert any(record.duplicates for record in records)
    assert registry.get("legacy_daemon").authority is RegistryAuthority.LEGACY_REFERENCE
    assert registry.get("autonomous_cycle").authority is RegistryAuthority.LEGACY_REFERENCE
    assert registry.get("kraken_r_foundation").authority is RegistryAuthority.KRAKEN_CANDIDATE
    assert registry.get("kraken_r_foundation").disposition is RegistryDisposition.KEEP


def test_planned_mechanisms_are_non_authoritative_and_have_no_operational_evidence() -> None:
    registry = load_default_registry()
    planned = [record for record in registry.all() if record.planned]
    assert planned
    assert all(record.authority is RegistryAuthority.DOCUMENTATION_ONLY for record in planned)
    assert all(record.evidence_grade == "none" for record in planned)
    assert all(
        record.current_status in {RegistryStatus.DESIGN_ONLY, RegistryStatus.MISSING}
        for record in planned
    )
    assert all(
        record.disposition
        in {RegistryDisposition.LEAVE_EXPERIMENTAL, RegistryDisposition.REBUILD}
        for record in planned
    )


def test_registry_rejects_duplicate_ids_unknown_references_and_cycles() -> None:
    raw = json.loads((ROOT / "kraken_r/architecture_registry.json").read_text())
    duplicate = dict(raw["mechanisms"][0])
    with pytest.raises(ArchitectureRegistryError, match="Duplicate mechanism IDs"):
        ArchitectureRegistry.from_dict({**raw, "mechanisms": [duplicate, duplicate]})

    records = list(load_default_registry().all())
    records[0] = type(records[0])(
        **{**records[0].__dict__, "dependencies": ("missing-id",)}
    )
    report = ArchitectureRegistry(records).validate()
    assert any("unknown dependency" in error for error in report.errors)

    a = type(records[0])(
        **{**records[0].__dict__, "mechanism_id": "cycle-a", "dependencies": ("cycle-b",)}
    )
    b = type(records[0])(
        **{**records[0].__dict__, "mechanism_id": "cycle-b", "dependencies": ("cycle-a",)}
    )
    report = ArchitectureRegistry([a, b]).validate()
    assert any("Dependency cycle" in error for error in report.errors)


def test_registry_strictly_requires_constitutional_metadata_and_json_types() -> None:
    raw = json.loads((ROOT / "kraken_r/architecture_registry.json").read_text())
    missing_boundary = dict(raw)
    missing_boundary.pop("authority_boundary")
    with pytest.raises(ArchitectureRegistryError, match="authority_boundary"):
        ArchitectureRegistry.from_dict(missing_boundary)

    invalid_date = dict(raw)
    invalid_date["last_reviewed"] = 20260824
    with pytest.raises(ArchitectureRegistryError, match="last_reviewed"):
        ArchitectureRegistry.from_dict(invalid_date)

    invalid_dependencies = json.loads(json.dumps(raw))
    invalid_dependencies["mechanisms"][0]["dependencies"] = "not-a-list"
    with pytest.raises(ArchitectureRegistryError, match="dependencies"):
        ArchitectureRegistry.from_dict(invalid_dependencies)

    unknown_record_field = json.loads(json.dumps(raw))
    unknown_record_field["mechanisms"][0]["runtime_authority"] = "forbidden"
    with pytest.raises(ArchitectureRegistryError, match="unknown fields"):
        ArchitectureRegistry.from_dict(unknown_record_field)


def test_constitution_metadata_is_loaded_and_enforces_candidate_only_boundary(tmp_path: Path) -> None:
    metadata = load_constitution_metadata()
    assert metadata["authority_boundary"]["allowed_effects"] == [
        "documentation", "standalone_validation", "candidate_cycle_execution"
    ]
    assert set(metadata["canonical_lifecycle"]) == set(CANONICAL_CONTRACTS)

    malformed = dict(metadata)
    malformed["authority_boundary"] = dict(metadata["authority_boundary"])
    malformed["authority_boundary"]["allowed_effects"] = ["runtime_wiring"]
    path = tmp_path / "constitution.json"
    path.write_text(json.dumps(malformed))
    with pytest.raises(ConstitutionValidationError, match="allowed_effects"):
        load_constitution_metadata(path)


def test_validator_is_standalone_and_does_not_import_or_wire_legacy_runtime() -> None:
    legacy_files = [
        ROOT / "rogal_core/daemon.py",
        ROOT / "rogal_core/autonomous_cycle.py",
        ROOT / ".replit",
    ]
    legacy_files = [path for path in legacy_files if path.exists()]
    before = {path: path.read_bytes() for path in legacy_files}
    result = subprocess.run(
        [sys.executable, "-m", "kraken_r", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["contracts_ok"] is True
    assert report["constitution"]["ok"] is True
    assert report["constitution"]["allowed_effects"] == [
        "documentation", "standalone_validation", "candidate_cycle_execution"
    ]
    assert report["cycle"]["modes"] == [
        "success", "failure", "contradiction", "insufficient_evidence"
    ]
    assert report["legacy_runtime_wiring"] == "absent"
    assert "rogal_core.daemon" not in result.stdout
    assert all(path.read_bytes() == content for path, content in before.items())


def test_kraken_r_source_has_no_legacy_runtime_imports_or_runtime_authorities() -> None:
    source = "\n".join(path.read_text() for path in (ROOT / "kraken_r").glob("*.py"))
    assert "import rogal_core" not in source
    assert "from rogal_core" not in source
    assert "EventBus" not in source
    assert "SQLite" not in source
    assert "subprocess" not in source
    assert "socket" not in source


def test_legacy_and_forgotten_mechanisms_are_classified() -> None:
    preserved = [
        ROOT / "rogal_core/orzhaal_bubble.py",
        ROOT / "rogal_core/code_executor/orzhaal_bubble.py",
        ROOT / "recursive_core",
        ROOT / "core/recursive",
    ]
    if any(path.exists() for path in preserved):
        assert all(path.exists() for path in preserved)
    registry = load_default_registry()
    assert registry.get("orzhaal_bubble").disposition is RegistryDisposition.QUARANTINE
    assert registry.get("recursive_core").disposition is RegistryDisposition.QUARANTINE