"""Bounded, independently verified candidate execution observations.

This module is a narrowly scoped candidate adapter.  It is not a scheduler,
store, dispatcher, or general command runner.  It accepts only a declared
candidate test action, materializes declared files in a disposable directory,
and returns an attested observation.  A separate verifier must accept that
observation before it can enter the constitutional cycle as grounded evidence.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Any, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from bounded_execution_runner import IsolatedRunnerError, run_isolated_pytest

from .contracts import Action, Authority, TaskState


class GroundedExecutionError(ValueError):
    """Raised when an execution request or record crosses a hard boundary."""


class GroundedExecutionRejected(GroundedExecutionError):
    """Raised before a request is allowed to materialize or execute."""


class ExecutionStatus(str, Enum):
    """Transport statuses; none of these are a caller-declared success claim."""

    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SETUP_FAILED = "setup_failed"
    EXECUTION_FAILED = "execution_failed"


class ExecutionFailureCode(str, Enum):
    """Failure classifications derived from bounded execution facts."""

    TEST_FAILURE = "test_failure"
    NO_TESTS_EXECUTED = "no_tests_executed"
    TIMEOUT = "timeout"
    SETUP_FAILED = "setup_failed"
    RUNNER_FAILURE = "runner_failure"


class EpistemicOutcomeClass(str, Enum):
    """The only outcomes a verified bounded observation can represent."""

    TASK_SUCCESS = "task_success"
    TASK_FAILURE = "task_failure"
    INFRASTRUCTURE_SETUP_FAILURE = "infrastructure_setup_failure"
    TIMEOUT_RESOURCE_FAILURE = "timeout_resource_failure"
    EXECUTION_FAILURE = "execution_failure"
    CONTRADICTION = "contradiction"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


_CREDITABLE_OUTCOME_CLASSES = {
    EpistemicOutcomeClass.TASK_SUCCESS,
    EpistemicOutcomeClass.TASK_FAILURE,
}


def _expected_epistemic_class(
    status: ExecutionStatus,
    failure_code: ExecutionFailureCode | None,
    *,
    execution_completed: bool,
    tests_run: int,
    tests_passed: int,
    tests_failed: int,
) -> EpistemicOutcomeClass:
    """Derive the class from executor facts, never a caller-provided label."""

    if status is ExecutionStatus.COMPLETED:
        if (
            failure_code is None
            and execution_completed
            and tests_run > 0
            and tests_passed == tests_run
            and tests_failed == 0
        ):
            return EpistemicOutcomeClass.TASK_SUCCESS
    elif status is ExecutionStatus.FAILED:
        if (
            failure_code is ExecutionFailureCode.TEST_FAILURE
            and execution_completed
            and tests_run > 0
            and tests_failed > 0
        ):
            return EpistemicOutcomeClass.TASK_FAILURE
        if (
            failure_code is ExecutionFailureCode.NO_TESTS_EXECUTED
            and execution_completed
            and tests_run == 0
        ):
            return EpistemicOutcomeClass.INSUFFICIENT_EVIDENCE
    elif status is ExecutionStatus.TIMEOUT:
        if failure_code is ExecutionFailureCode.TIMEOUT and not execution_completed:
            return EpistemicOutcomeClass.TIMEOUT_RESOURCE_FAILURE
    elif status is ExecutionStatus.SETUP_FAILED:
        if failure_code is ExecutionFailureCode.SETUP_FAILED and not execution_completed:
            return EpistemicOutcomeClass.INFRASTRUCTURE_SETUP_FAILURE
    elif status is ExecutionStatus.EXECUTION_FAILED:
        if failure_code is ExecutionFailureCode.RUNNER_FAILURE:
            return EpistemicOutcomeClass.EXECUTION_FAILURE
    raise GroundedExecutionError(
        "execution facts do not establish one canonical epistemic outcome class"
    )


def _freeze_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GroundedExecutionError(f"{field_name} must be a mapping")
    return MappingProxyType({str(key): item for key, item in value.items()})


def _canonical_json(value: Any) -> str:
    def default(item: Any) -> Any:
        if isinstance(item, Enum):
            return item.value
        if hasattr(item, "to_dict"):
            return item.to_dict()
        raise TypeError(f"{type(item).__name__} is not canonical JSON")

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=default,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _causal_criterion_binding(request: "GroundedExecutionRequest") -> str:
    """Bind causal proof to the independently executed criterion, not subject.

    The implementation under test may change between a passing update and a
    later failing falsification.  Continuity therefore means the same action
    contract and exact executed test paths/content, which the executor signs
    and the verifier recomputes from the sealed request.
    """

    return _digest(
        {
            "action": {
                "operation": request.action.operation,
                "target": request.action.target,
                "authority": request.action.authority.value,
            },
            "test_paths": list(request.test_paths),
            "test_files": {path: request.files[path] for path in request.test_paths},
        }
    )


def _safe_path(value: str, field_name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or len(value) > 240 or "\x00" in value:
        raise GroundedExecutionRejected(f"{field_name} must be a bounded relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise GroundedExecutionRejected(f"{field_name} escapes the isolated workspace")
    return path


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GroundedExecutionError(f"{field_name} must be a mapping")
    return value


def _action_from_dict(payload: Mapping[str, Any]) -> Action:
    return Action(
        payload["action_id"],
        payload["objective_id"],
        payload["operation"],
        payload["target"],
        parameters=_mapping(payload.get("parameters", {}), "action parameters"),
        preconditions=tuple(payload.get("preconditions", ())),
        requested_by=payload.get("requested_by", "kraken_r_candidate"),
        authority=payload.get("authority", Authority.KRAKEN_CANDIDATE.value),
    )


def task_state_from_dict(payload: Mapping[str, Any]) -> TaskState:
    """Reconstruct the sealed authorized-state identity from canonical JSON."""

    return TaskState(
        payload["state_id"],
        payload["objective_id"],
        payload["version"],
        payload["phase"],
        values=_mapping(payload.get("values", {}), "state values"),
        evidence_ids=tuple(payload.get("evidence_ids", ())),
        authority=payload.get("authority", Authority.KRAKEN_CANDIDATE.value),
    )


def task_state_to_json(state: TaskState) -> str:
    if not isinstance(state, TaskState):
        raise GroundedExecutionError("authorized state must be a TaskState")
    return _canonical_json(state.to_dict())


def task_state_from_json(payload: str) -> TaskState:
    try:
        return task_state_from_dict(_mapping(json.loads(payload), "serialized state"))
    except (TypeError, ValueError, json.JSONDecodeError, KeyError) as exc:
        raise GroundedExecutionError("serialized authorized state is invalid") from exc


@dataclass(frozen=True)
class ExecutionLimits:
    """Explicit resource limits for one bounded child-process observation."""

    timeout_seconds: float = 5.0
    cpu_seconds: int = 3
    memory_bytes: int = 536_870_912
    output_bytes: int = 32_768

    def __post_init__(self) -> None:
        if not isinstance(self.timeout_seconds, (int, float)) or not (
            0.1 <= float(self.timeout_seconds) <= 30.0
        ):
            raise GroundedExecutionError("timeout_seconds must be from 0.1 through 30")
        if isinstance(self.cpu_seconds, bool) or not 1 <= self.cpu_seconds <= 30:
            raise GroundedExecutionError("cpu_seconds must be from 1 through 30")
        if (
            isinstance(self.memory_bytes, bool)
            or not 64 * 1024 * 1024 <= self.memory_bytes <= 1024 * 1024 * 1024
        ):
            raise GroundedExecutionError(
                "memory_bytes must be from 64 MiB through 1 GiB"
            )
        if (
            isinstance(self.output_bytes, bool)
            or not 1_024 <= self.output_bytes <= 1_048_576
        ):
            raise GroundedExecutionError(
                "output_bytes must be from 1024 through 1048576"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout_seconds": float(self.timeout_seconds),
            "cpu_seconds": self.cpu_seconds,
            "memory_bytes": self.memory_bytes,
            "output_bytes": self.output_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionLimits":
        try:
            return cls(
                timeout_seconds=payload["timeout_seconds"],
                cpu_seconds=payload["cpu_seconds"],
                memory_bytes=payload["memory_bytes"],
                output_bytes=payload["output_bytes"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GroundedExecutionError("serialized execution limits are invalid") from exc


@dataclass(frozen=True)
class GroundedExecutionRequest:
    """Sealed input for a single authorized candidate test action."""

    request_id: str
    transaction_id: str
    objective_id: str
    task_state_id: str
    task_state_version: int
    action: Action
    files: Mapping[str, str]
    test_paths: tuple[str, ...]
    limits: ExecutionLimits = field(default_factory=ExecutionLimits)
    causal_parent_record_id: str | None = None
    causal_target: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "transaction_id",
            "objective_id",
            "task_state_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise GroundedExecutionError(f"{name} must be a non-empty string")
        if (
            isinstance(self.task_state_version, bool)
            or not isinstance(self.task_state_version, int)
            or self.task_state_version < 1
        ):
            raise GroundedExecutionError("task_state_version must be positive")
        if not isinstance(self.action, Action):
            raise GroundedExecutionError("request requires an Action")
        if self.action.objective_id != self.objective_id:
            raise GroundedExecutionRejected("action objective does not match request")
        if self.action.authority is not Authority.KRAKEN_CANDIDATE:
            raise GroundedExecutionRejected("execution action is not candidate-authorized")
        if self.action.requested_by != "kraken_r_candidate":
            raise GroundedExecutionRejected("execution action requester is not candidate")
        if self.action.operation != "run_bounded_pytest":
            raise GroundedExecutionRejected("action operation is not an allowed test action")
        if self.action.target != "isolated_workspace":
            raise GroundedExecutionRejected("action target is not the isolated workspace")
        files = _freeze_mapping(self.files, "files")
        if not files or len(files) > 32:
            raise GroundedExecutionRejected("files must contain from 1 through 32 entries")
        total_bytes = 0
        normalized: dict[str, str] = {}
        for raw_path, content in files.items():
            path = _safe_path(raw_path, "files path")
            if not isinstance(content, str):
                raise GroundedExecutionRejected("workspace file content must be text")
            total_bytes += len(content.encode("utf-8"))
            normalized[path.as_posix()] = content
        if total_bytes > 1_048_576:
            raise GroundedExecutionRejected("workspace input exceeds 1 MiB")
        tests = tuple(self.test_paths)
        if not tests or len(tests) > 16:
            raise GroundedExecutionRejected("test_paths must contain from 1 through 16 paths")
        for raw_path in tests:
            path = _safe_path(raw_path, "test path")
            if path.suffix != ".py" or path.as_posix() not in normalized:
                raise GroundedExecutionRejected(
                    "each test path must name a declared Python workspace file"
                )
        object.__setattr__(self, "files", MappingProxyType(normalized))
        object.__setattr__(self, "test_paths", tests)
        if not isinstance(self.limits, ExecutionLimits):
            raise GroundedExecutionError("limits must be ExecutionLimits")
        if self.causal_parent_record_id is not None and (
            not isinstance(self.causal_parent_record_id, str)
            or not self.causal_parent_record_id.strip()
        ):
            raise GroundedExecutionError(
                "causal_parent_record_id must be a non-empty string or None"
            )
        if self.causal_target is not None:
            if not isinstance(self.causal_target, Mapping) or set(self.causal_target) != {
                "target_record_id",
                "target_settlement_id",
                "target_update_lineage_id",
                "target_audit_hash",
                "target_criterion_binding",
            }:
                raise GroundedExecutionError("causal_target must use the exact target schema")
            target = _freeze_mapping(self.causal_target, "causal_target")
            if any(not isinstance(value, str) or not value for value in target.values()):
                raise GroundedExecutionError("causal_target values must be non-empty strings")
            if self.causal_parent_record_id != target["target_record_id"]:
                raise GroundedExecutionError("causal target and parent record must agree")
            object.__setattr__(self, "causal_target", target)

    @property
    def input_hash(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "transaction_id": self.transaction_id,
            "objective_id": self.objective_id,
            "task_state_id": self.task_state_id,
            "task_state_version": self.task_state_version,
            "action": self.action.to_dict(),
            "files": dict(self.files),
            "test_paths": list(self.test_paths),
            "limits": self.limits.to_dict(),
            "causal_parent_record_id": self.causal_parent_record_id,
            "causal_target": (
                dict(self.causal_target) if self.causal_target is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GroundedExecutionRequest":
        try:
            return cls(
                payload["request_id"],
                payload["transaction_id"],
                payload["objective_id"],
                payload["task_state_id"],
                payload["task_state_version"],
                _action_from_dict(_mapping(payload["action"], "serialized action")),
                _mapping(payload["files"], "serialized files"),
                tuple(payload["test_paths"]),
                ExecutionLimits.from_dict(
                    _mapping(payload["limits"], "serialized limits")
                ),
                payload.get("causal_parent_record_id"),
                payload.get("causal_target"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GroundedExecutionError("serialized execution request is invalid") from exc

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "GroundedExecutionRequest":
        try:
            return cls.from_dict(_mapping(json.loads(payload), "serialized request"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise GroundedExecutionError("serialized execution request is invalid") from exc


@dataclass(frozen=True)
class ExecutionObservation:
    """Raw process facts captured by the bounded transport."""

    status: ExecutionStatus
    exit_code: int | None
    execution_completed: bool
    tests_run: int
    tests_passed: int
    tests_failed: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    failure_code: ExecutionFailureCode | None = None
    epistemic_class: EpistemicOutcomeClass | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ExecutionStatus):
            object.__setattr__(self, "status", ExecutionStatus(self.status))
        if self.exit_code is not None and not isinstance(self.exit_code, int):
            raise GroundedExecutionError("exit_code must be an integer or None")
        for field_name in ("tests_run", "tests_passed", "tests_failed"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise GroundedExecutionError(f"{field_name} must be a non-negative integer")
        if self.tests_passed + self.tests_failed > self.tests_run:
            raise GroundedExecutionError("test counts exceed collected tests")
        if not isinstance(self.execution_completed, bool):
            raise GroundedExecutionError("execution_completed must be a boolean")
        if not isinstance(self.elapsed_seconds, (int, float)) or self.elapsed_seconds < 0:
            raise GroundedExecutionError("elapsed_seconds must be non-negative")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise GroundedExecutionError("execution output must be text")
        if self.failure_code is not None and not isinstance(
            self.failure_code, ExecutionFailureCode
        ):
            object.__setattr__(
                self, "failure_code", ExecutionFailureCode(self.failure_code)
            )
        expected_class = _expected_epistemic_class(
            self.status,
            self.failure_code,
            execution_completed=self.execution_completed,
            tests_run=self.tests_run,
            tests_passed=self.tests_passed,
            tests_failed=self.tests_failed,
        )
        if self.epistemic_class is None:
            object.__setattr__(self, "epistemic_class", expected_class)
        elif not isinstance(self.epistemic_class, EpistemicOutcomeClass):
            object.__setattr__(
                self,
                "epistemic_class",
                EpistemicOutcomeClass(self.epistemic_class),
            )
        if self.epistemic_class is not expected_class:
            raise GroundedExecutionError(
                "execution epistemic class does not match attested process facts"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "exit_code": self.exit_code,
            "execution_completed": self.execution_completed,
            "tests_run": self.tests_run,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "elapsed_seconds": round(float(self.elapsed_seconds), 6),
            "failure_code": (
                self.failure_code.value if self.failure_code is not None else None
            ),
            "epistemic_class": self.epistemic_class.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionObservation":
        try:
            return cls(
                payload["status"],
                payload.get("exit_code"),
                payload["execution_completed"],
                payload["tests_run"],
                payload["tests_passed"],
                payload["tests_failed"],
                payload["stdout"],
                payload["stderr"],
                payload["elapsed_seconds"],
                payload.get("failure_code"),
                payload.get("epistemic_class"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GroundedExecutionError("serialized execution observation is invalid") from exc


@dataclass(frozen=True)
class ExecutionProvenance:
    """Provenance fixed by the executor, not supplied by test or model output."""

    executor_id: str
    isolation: str
    resource_limits_enforced: bool
    cleanup_verified: bool
    workspace_hash: str
    child_runtime_verified: bool = False
    child_runtime_fingerprint: str | None = None
    causal_target_witness: str | None = None
    causal_criterion_binding: str | None = None

    def __post_init__(self) -> None:
        if self.isolation != "user_mount_network_pid_namespace":
            raise GroundedExecutionError("unknown execution isolation boundary")
        if not isinstance(self.resource_limits_enforced, bool) or not isinstance(
            self.cleanup_verified, bool
        ) or not isinstance(
            self.child_runtime_verified, bool
        ):
            raise GroundedExecutionError("execution provenance booleans are required")
        for field_name in ("executor_id", "workspace_hash"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise GroundedExecutionError(f"{field_name} must be non-empty")
        if self.child_runtime_verified:
            if (
                not isinstance(self.child_runtime_fingerprint, str)
                or len(self.child_runtime_fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in self.child_runtime_fingerprint)
            ):
                raise GroundedExecutionError(
                    "verified child runtime requires a canonical fingerprint"
                )
        elif self.child_runtime_fingerprint is not None:
            raise GroundedExecutionError(
                "unverified child runtime cannot carry a runtime fingerprint"
            )
        if self.causal_target_witness is not None and (
            not isinstance(self.causal_target_witness, str)
            or len(self.causal_target_witness) != 64
            or any(char not in "0123456789abcdef" for char in self.causal_target_witness)
        ):
            raise GroundedExecutionError("causal target witness must be a SHA-256 digest")
        if self.causal_criterion_binding is not None and (
            not isinstance(self.causal_criterion_binding, str)
            or len(self.causal_criterion_binding) != 64
            or any(char not in "0123456789abcdef" for char in self.causal_criterion_binding)
        ):
            raise GroundedExecutionError("causal criterion binding must be a SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "executor_id": self.executor_id,
            "isolation": self.isolation,
            "resource_limits_enforced": self.resource_limits_enforced,
            "cleanup_verified": self.cleanup_verified,
            "workspace_hash": self.workspace_hash,
            "child_runtime_verified": self.child_runtime_verified,
            "child_runtime_fingerprint": self.child_runtime_fingerprint,
            "causal_target_witness": self.causal_target_witness,
            "causal_criterion_binding": self.causal_criterion_binding,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionProvenance":
        if (
            "child_runtime_verified" not in payload
            or "child_runtime_fingerprint" not in payload
            or "causal_target_witness" not in payload
            or "causal_criterion_binding" not in payload
        ):
            raise GroundedExecutionError(
                "serialized execution provenance predates the child runtime contract"
            )
        try:
            return cls(
                payload["executor_id"],
                payload["isolation"],
                payload["resource_limits_enforced"],
                payload["cleanup_verified"],
                payload["workspace_hash"],
                payload["child_runtime_verified"],
                payload["child_runtime_fingerprint"],
                payload["causal_target_witness"],
                payload["causal_criterion_binding"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GroundedExecutionError("serialized execution provenance is invalid") from exc


@dataclass(frozen=True)
class ExecutionAttestation:
    """Opaque executor attestation checked by an independent verifier."""

    attestor_id: str
    key_fingerprint: str
    signature: str
    public_key: str
    algorithm: str = "ed25519"

    def __post_init__(self) -> None:
        for field_name in (
            "attestor_id",
            "key_fingerprint",
            "signature",
            "public_key",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise GroundedExecutionError(f"{field_name} must be non-empty")
        if self.algorithm != "ed25519":
            raise GroundedExecutionError("attestation algorithm must be ed25519")

    def to_dict(self) -> dict[str, str]:
        return {
            "attestor_id": self.attestor_id,
            "key_fingerprint": self.key_fingerprint,
            "signature": self.signature,
            "public_key": self.public_key,
            "algorithm": self.algorithm,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionAttestation":
        try:
            return cls(
                payload["attestor_id"],
                payload["key_fingerprint"],
                payload["signature"],
                payload["public_key"],
                payload.get("algorithm", "ed25519"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GroundedExecutionError("serialized execution attestation is invalid") from exc


@dataclass(frozen=True)
class TrustedExecutorIdentity:
    """Externally supplied executor identity; records cannot establish trust."""

    attestor_id: str
    key_fingerprint: str
    public_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.attestor_id, str) or not self.attestor_id:
            raise GroundedExecutionError("trusted executor id is required")
        try:
            raw_key = base64.b64decode(self.public_key, validate=True)
        except ValueError as exc:
            raise GroundedExecutionError("trusted executor public key is invalid") from exc
        if len(raw_key) != 32:
            raise GroundedExecutionError("trusted executor public key has invalid length")
        if self.key_fingerprint != hashlib.sha256(raw_key).hexdigest()[:24]:
            raise GroundedExecutionError("trusted executor fingerprint does not match key")

    def to_dict(self) -> dict[str, str]:
        return {
            "attestor_id": self.attestor_id,
            "key_fingerprint": self.key_fingerprint,
            "public_key": self.public_key,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrustedExecutorIdentity":
        try:
            return cls(
                payload["attestor_id"],
                payload["key_fingerprint"],
                payload["public_key"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GroundedExecutionError("serialized trusted executor is invalid") from exc


@dataclass(frozen=True)
class GroundedExecutionRecord:
    """Immutable executor-issued record; it is not evidence until verified."""

    record_id: str
    request_id: str
    transaction_id: str
    objective_id: str
    task_state_id: str
    task_state_version: int
    action: Action
    input_hash: str
    output_hash: str
    observation: ExecutionObservation
    provenance: ExecutionProvenance
    record_hash: str
    attestation: ExecutionAttestation

    def __post_init__(self) -> None:
        for field_name in (
            "record_id",
            "request_id",
            "transaction_id",
            "objective_id",
            "task_state_id",
            "input_hash",
            "output_hash",
            "record_hash",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise GroundedExecutionError(f"{field_name} must be non-empty")
        if not isinstance(self.action, Action):
            raise GroundedExecutionError("record action must be an Action")
        if not isinstance(self.observation, ExecutionObservation):
            raise GroundedExecutionError("record observation is required")
        if not isinstance(self.provenance, ExecutionProvenance):
            raise GroundedExecutionError("record provenance is required")
        if not isinstance(self.attestation, ExecutionAttestation):
            raise GroundedExecutionError("record attestation is required")

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "request_id": self.request_id,
            "transaction_id": self.transaction_id,
            "objective_id": self.objective_id,
            "task_state_id": self.task_state_id,
            "task_state_version": self.task_state_version,
            "action": self.action.to_dict(),
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "observation": self.observation.to_dict(),
            "provenance": self.provenance.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.unsigned_payload(),
            "record_hash": self.record_hash,
            "attestation": self.attestation.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GroundedExecutionRecord":
        try:
            return cls(
                payload["record_id"],
                payload["request_id"],
                payload["transaction_id"],
                payload["objective_id"],
                payload["task_state_id"],
                payload["task_state_version"],
                _action_from_dict(_mapping(payload["action"], "serialized action")),
                payload["input_hash"],
                payload["output_hash"],
                ExecutionObservation.from_dict(
                    _mapping(payload["observation"], "serialized observation")
                ),
                ExecutionProvenance.from_dict(
                    _mapping(payload["provenance"], "serialized provenance")
                ),
                payload["record_hash"],
                ExecutionAttestation.from_dict(
                    _mapping(payload["attestation"], "serialized attestation")
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GroundedExecutionError("serialized execution record is invalid") from exc

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "GroundedExecutionRecord":
        try:
            return cls.from_dict(_mapping(json.loads(payload), "serialized record"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise GroundedExecutionError("serialized execution record is invalid") from exc


class GroundedDeliveryLedger:
    """Bounded receipt ledger with optional durable replay and expiry semantics."""

    _SCHEMA_VERSION = 2
    _LEGACY_SCHEMA_VERSION = 1
    _MAX_RECEIPTS = 256
    _MAX_IDENTITIES = 512

    def __init__(
        self,
        storage_path: str | Path | None = None,
        *,
        receipt_ttl_seconds: int = 3600,
        clock: callable | None = None,
    ) -> None:
        if (
            isinstance(receipt_ttl_seconds, bool)
            or not isinstance(receipt_ttl_seconds, int)
            or not 1 <= receipt_ttl_seconds <= 86_400
        ):
            raise GroundedExecutionError("receipt ttl must be from 1 through 86400 seconds")
        self._storage_path = Path(storage_path) if storage_path is not None else None
        self._receipt_ttl_seconds = receipt_ttl_seconds
        self._clock = clock or time.time
        self._receipts: dict[tuple[str, str], tuple[str, int]] = {}
        self._identities: dict[tuple[str, str], str] = {}
        self._needs_migration = False
        self._lock = threading.Lock()
        if self._storage_path is not None:
            self._load()
            if self._needs_migration:
                # A valid legacy file is upgraded immediately via the same
                # fsync + replace path as claims; it is never left half-migrated
                # pending a later delivery.
                self._persist()
                self._needs_migration = False

    @property
    def is_durable(self) -> bool:
        return self._storage_path is not None

    def _load(self) -> None:
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            payload = json.loads(self._storage_path.read_text(encoding="utf-8"))
            version = payload.get("schema_version")
            if version not in {self._LEGACY_SCHEMA_VERSION, self._SCHEMA_VERSION}:
                raise ValueError("unknown receipt schema")
            if (
                isinstance(payload.get("receipt_ttl_seconds"), bool)
                or not isinstance(payload.get("receipt_ttl_seconds"), int)
                or not 1 <= payload["receipt_ttl_seconds"] <= 86_400
            ):
                raise ValueError("invalid receipt ttl")
            if payload["receipt_ttl_seconds"] != self._receipt_ttl_seconds:
                raise ValueError("receipt ttl does not match durable ledger")
            receipts = payload.get("receipts", ())
            if not isinstance(receipts, list) or len(receipts) > self._MAX_RECEIPTS:
                raise ValueError("invalid receipt retention")
            loaded: dict[tuple[str, str], tuple[str, int]] = {}
            for item in receipts:
                stage = item["stage"]
                identity = item["identity"]
                binding = item["binding"]
                expires_at = item["expires_at"]
                if (
                    stage not in {"execution", "evidence", "settlement", "learning"}
                    or not isinstance(identity, str)
                    or not isinstance(binding, str)
                    or isinstance(expires_at, bool)
                    or not isinstance(expires_at, int)
                ):
                    raise ValueError("invalid receipt")
                if (
                    not identity
                    or not binding
                    or len(identity) > 256
                    or len(binding) > 256
                    or (stage, identity) in loaded
                ):
                    raise ValueError("invalid receipt")
                loaded[(stage, identity)] = (binding, expires_at)
            identities: dict[tuple[str, str], str] = {}
            if version == self._SCHEMA_VERSION:
                tombstones = payload.get("identity_tombstones")
                if not isinstance(tombstones, list) or len(tombstones) > self._MAX_IDENTITIES:
                    raise ValueError("invalid identity tombstones")
                for item in tombstones:
                    stage, identity, binding = item["stage"], item["identity"], item["binding"]
                    if (
                        stage not in {"execution", "evidence", "settlement", "learning"}
                        or not isinstance(identity, str) or not identity or len(identity) > 256
                        or not isinstance(binding, str) or not binding or len(binding) > 256
                        or (stage, identity) in identities
                    ):
                        raise ValueError("invalid identity tombstone")
                    identities[(stage, identity)] = binding
            else:
                # Valid v1 receipts become permanent identities on first v2 load.
                identities = {key: binding for key, (binding, _) in loaded.items()}
                self._needs_migration = True
            if len(identities) > self._MAX_IDENTITIES or not set(loaded).issubset(identities):
                raise ValueError("invalid identity retention")
            self._receipts = loaded
            self._identities = identities
        except (OSError, TypeError, ValueError, KeyError, AttributeError, json.JSONDecodeError) as exc:
            raise GroundedExecutionError("durable receipt ledger is invalid") from exc

    def _persist(self) -> None:
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self._SCHEMA_VERSION,
            "receipt_ttl_seconds": self._receipt_ttl_seconds,
            "receipts": [
                {
                    "stage": stage,
                    "identity": identity,
                    "binding": binding,
                    "expires_at": expires_at,
                }
                for (stage, identity), (binding, expires_at) in sorted(
                    self._receipts.items()
                )
            ],
            "identity_tombstones": [
                {"stage": stage, "identity": identity, "binding": binding}
                for (stage, identity), binding in sorted(self._identities.items())
            ],
        }
        temporary = self._storage_path.with_suffix(self._storage_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self._storage_path)
        directory_fd = os.open(self._storage_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @contextmanager
    def _locked_receipts(self):
        """Serialize durable read-modify-write claims across processes."""

        with self._lock:
            if self._storage_path is None:
                yield
                return
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self._storage_path.with_suffix(
                self._storage_path.suffix + ".lock"
            )
            with lock_path.open("a+", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    # Another process can have committed a claim since this
                    # instance started, so refresh only while holding the lock.
                    self._load()
                    yield
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _purge_expired(self, now: int) -> bool:
        expired = [
            key for key, (_, expires_at) in self._receipts.items() if expires_at <= now
        ]
        for key in expired:
            del self._receipts[key]
        return bool(expired)

    def receipt_status(self, stage: str, identity: str) -> str:
        """Return active, expired, or absent without claiming a delivery."""

        with self._locked_receipts():
            now = int(self._clock())
            existing = self._receipts.get((stage, identity))
            if existing is not None and existing[1] > now:
                return "active"
            changed = self._purge_expired(now)
            if changed:
                self._persist()
            return "expired" if existing is not None else "absent"

    def claim(self, stage: str, identity: str, binding: str) -> None:
        if stage not in {"execution", "evidence", "settlement", "learning"}:
            raise GroundedExecutionError("unknown grounded delivery stage")
        if (
            not isinstance(identity, str)
            or not isinstance(binding, str)
            or not identity
            or not binding
            or len(identity) > 256
            or len(binding) > 256
        ):
            raise GroundedExecutionError("delivery identity and binding are required")
        key = (stage, identity)
        with self._locked_receipts():
            now = int(self._clock())
            changed = self._purge_expired(now)
            previous_binding = self._identities.get(key)
            if previous_binding is None:
                if len(self._identities) >= self._MAX_IDENTITIES:
                    raise GroundedExecutionRejected("grounded delivery identity retention is exhausted")
                if len(self._receipts) >= self._MAX_RECEIPTS:
                    raise GroundedExecutionRejected("grounded delivery receipt retention is exhausted")
                self._receipts[key] = (binding, now + self._receipt_ttl_seconds)
                self._identities[key] = binding
                self._persist()
                return
            if changed:
                self._persist()
            if previous_binding == binding:
                raise GroundedExecutionRejected(
                    f"duplicate grounded {stage} delivery is already reconciled"
                )
            raise GroundedExecutionRejected(
                f"conflicting grounded {stage} delivery has the same identity"
            )


class ExecutionAttestor:
    """Ed25519 signer or public verifier; private material never serializes."""

    def __init__(
        self,
        private_key: bytes | None = None,
        public_key: bytes | None = None,
        *,
        attestor_id: str = "kraken_r_grounded_execution_v1",
    ) -> None:
        if private_key is not None and public_key is not None:
            raise GroundedExecutionError("attestor accepts private or public key, not both")
        if private_key is not None:
            try:
                self._private_key = Ed25519PrivateKey.from_private_bytes(bytes(private_key))
            except ValueError as exc:
                raise GroundedExecutionError("invalid Ed25519 private key") from exc
            self._public_key = self._private_key.public_key()
        elif public_key is not None:
            try:
                self._public_key = Ed25519PublicKey.from_public_bytes(bytes(public_key))
            except ValueError as exc:
                raise GroundedExecutionError("invalid Ed25519 public key") from exc
            self._private_key = None
        else:
            self._private_key = Ed25519PrivateKey.generate()
            self._public_key = self._private_key.public_key()
        self.attestor_id = attestor_id
        self.public_key = self._public_key.public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        self.public_key_encoded = base64.b64encode(self.public_key).decode("ascii")
        self.key_fingerprint = hashlib.sha256(self.public_key).hexdigest()[:24]

    def trusted_identity(self) -> TrustedExecutorIdentity:
        """Export the public identity that a verifier must receive out of band."""

        return TrustedExecutorIdentity(
            self.attestor_id,
            self.key_fingerprint,
            self.public_key_encoded,
        )

    def attest(self, payload: Mapping[str, Any]) -> ExecutionAttestation:
        if self._private_key is None:
            raise GroundedExecutionError("public-only attestor cannot sign records")
        signature = base64.b64encode(
            self._private_key.sign(_canonical_json(payload).encode("utf-8"))
        ).decode("ascii")
        return ExecutionAttestation(
            self.attestor_id,
            self.key_fingerprint,
            signature,
            self.public_key_encoded,
        )

    def verifies(
        self, payload: Mapping[str, Any], attestation: ExecutionAttestation
    ) -> bool:
        if (
            attestation.attestor_id != self.attestor_id
            or attestation.key_fingerprint != self.key_fingerprint
        ):
            return False
        if attestation.public_key != self.public_key_encoded:
            return False
        try:
            self._public_key.verify(
                base64.b64decode(attestation.signature, validate=True),
                _canonical_json(payload).encode("utf-8"),
            )
        except (ValueError, TypeError):
            return False
        except Exception:
            return False
        return True


def _workspace_hash(files: Mapping[str, str]) -> str:
    return _digest({"files": dict(files)})


def _observation_from_isolated_result(result: Any) -> ExecutionObservation:
    if result.timed_out:
        return ExecutionObservation(
            ExecutionStatus.TIMEOUT,
            None,
            False,
            0,
            0,
            0,
            result.stdout,
            result.stderr,
            result.elapsed_seconds,
            ExecutionFailureCode.TIMEOUT,
        )
    if result.sandbox_error:
        return ExecutionObservation(
            ExecutionStatus.SETUP_FAILED,
            result.exit_code,
            False,
            0,
            0,
            0,
            result.stdout,
            result.stderr,
            result.elapsed_seconds,
            ExecutionFailureCode.SETUP_FAILED,
        )
    if not result.runtime_verified:
        return ExecutionObservation(
            ExecutionStatus.EXECUTION_FAILED,
            result.exit_code,
            False,
            0,
            0,
            0,
            result.stdout,
            result.stderr,
            result.elapsed_seconds,
            ExecutionFailureCode.RUNNER_FAILURE,
        )
    if not result.test_outcomes_complete:
        return ExecutionObservation(
            ExecutionStatus.EXECUTION_FAILED,
            result.exit_code,
            False,
            0,
            0,
            0,
            result.stdout,
            result.stderr,
            result.elapsed_seconds,
            ExecutionFailureCode.RUNNER_FAILURE,
        )
    tests_run = result.tests_run
    tests_passed = result.tests_passed
    tests_failed = result.tests_failed
    exit_code = result.exit_code
    if tests_run == 0:
        if exit_code == 5:
            status = ExecutionStatus.FAILED
            failure = ExecutionFailureCode.NO_TESTS_EXECUTED
        else:
            status = ExecutionStatus.SETUP_FAILED
            failure = ExecutionFailureCode.SETUP_FAILED
    elif exit_code == 0 and tests_passed == tests_run and tests_failed == 0:
        status = ExecutionStatus.COMPLETED
        failure = None
    else:
        status = ExecutionStatus.FAILED
        failure = ExecutionFailureCode.TEST_FAILURE
    return ExecutionObservation(
        status,
        exit_code,
        True,
        tests_run,
        tests_passed,
        tests_failed,
        result.stdout,
        result.stderr,
        result.elapsed_seconds,
        failure,
    )


class GroundedExecutionExecutor:
    """Materialize and observe one request without turning it into evidence."""

    def __init__(
        self,
        attestor: ExecutionAttestor | None = None,
        delivery_ledger: GroundedDeliveryLedger | None = None,
        *,
        executor_id: str = "kraken_r_bounded_pytest_executor_v1",
    ) -> None:
        self._attestor = attestor or ExecutionAttestor()
        if not isinstance(delivery_ledger, GroundedDeliveryLedger) or not (
            delivery_ledger.is_durable
        ):
            raise GroundedExecutionError(
                "grounded execution requires a durable delivery ledger"
            )
        self.delivery_ledger = delivery_ledger
        self.executor_id = executor_id

    def verifier(self) -> "GroundedExecutionVerifier":
        """Return the verifier without exposing the executor signing capability."""

        return GroundedExecutionVerifier(
            self._attestor.trusted_identity(), self.delivery_ledger
        )

    def trusted_executor(self) -> TrustedExecutorIdentity:
        """Return the external public trust anchor for restart verification."""

        return self._attestor.trusted_identity()

    @staticmethod
    def _validate_authorization(
        request: GroundedExecutionRequest, authorized_state: TaskState
    ) -> None:
        if not isinstance(authorized_state, TaskState):
            raise GroundedExecutionRejected("authorized TaskState is required")
        if (
            authorized_state.objective_id != request.objective_id
            or authorized_state.state_id != request.task_state_id
            or authorized_state.version != request.task_state_version
        ):
            raise GroundedExecutionRejected("execution request has stale task-state binding")
        if authorized_state.phase != "authorized":
            raise GroundedExecutionRejected("task state does not authorize execution")
        if authorized_state.authority is not Authority.KRAKEN_CANDIDATE:
            raise GroundedExecutionRejected("task state is not candidate-authorized")
        if request.task_state_version != 5:
            raise GroundedExecutionRejected(
                "grounded execution is limited to the candidate authorized state"
            )

    def _record(
        self,
        request: GroundedExecutionRequest,
        observation: ExecutionObservation,
        workspace_hash: str,
        *,
        resource_limits_enforced: bool,
        cleanup_verified: bool,
        child_runtime_verified: bool,
        child_runtime_fingerprint: str | None,
    ) -> GroundedExecutionRecord:
        provenance = ExecutionProvenance(
            self.executor_id,
            "user_mount_network_pid_namespace",
            resource_limits_enforced,
            cleanup_verified,
            workspace_hash,
            child_runtime_verified,
            child_runtime_fingerprint,
            (
                _digest(
                    {
                        "request_input_hash": request.input_hash,
                        "causal_target": dict(request.causal_target),
                        "causal_criterion_binding": _causal_criterion_binding(request),
                    }
                )
                if request.causal_target is not None
                else None
            ),
            _causal_criterion_binding(request),
        )
        base = {
            "record_id": f"{request.request_id}-record",
            "request_id": request.request_id,
            "transaction_id": request.transaction_id,
            "objective_id": request.objective_id,
            "task_state_id": request.task_state_id,
            "task_state_version": request.task_state_version,
            "action": request.action,
            "input_hash": request.input_hash,
            "output_hash": _digest(observation.to_dict()),
            "observation": observation,
            "provenance": provenance,
        }
        unsigned = {
            **base,
            "action": request.action.to_dict(),
            "observation": observation.to_dict(),
            "provenance": provenance.to_dict(),
        }
        record_hash = _digest(unsigned)
        payload = {**unsigned, "record_hash": record_hash}
        attestation = self._attestor.attest(payload)
        return GroundedExecutionRecord(
            **base,
            record_hash=record_hash,
            attestation=attestation,
        )

    def execute(
        self,
        request: GroundedExecutionRequest,
        *,
        authorized_state: TaskState,
    ) -> GroundedExecutionRecord:
        """Run the declared test files after all authorization checks succeed."""

        if not isinstance(request, GroundedExecutionRequest):
            raise GroundedExecutionRejected("grounded execution request is required")
        self._validate_authorization(request, authorized_state)
        workspace_hash = _workspace_hash(request.files)
        self.delivery_ledger.claim("execution", request.request_id, request.input_hash)
        with TemporaryDirectory(prefix="kraken-r-grounded-") as temporary:
            workspace = Path(temporary)
            for relative_path, content in request.files.items():
                path = workspace / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            try:
                isolated_result = run_isolated_pytest(
                    workspace,
                    request.test_paths,
                    timeout_seconds=float(request.limits.timeout_seconds),
                    cpu_seconds=request.limits.cpu_seconds,
                    memory_bytes=request.limits.memory_bytes,
                    output_bytes=request.limits.output_bytes,
                )
            except IsolatedRunnerError as exc:
                raise GroundedExecutionRejected(
                    f"OS-isolated execution boundary is unavailable: {exc}"
                ) from exc
            observation = _observation_from_isolated_result(isolated_result)
        return self._record(
            request,
            observation,
            workspace_hash,
            resource_limits_enforced=isolated_result.resource_limits_enforced,
            cleanup_verified=isolated_result.cleanup_verified,
            child_runtime_verified=isolated_result.runtime_verified,
            child_runtime_fingerprint=isolated_result.runtime_fingerprint,
        )


@dataclass(frozen=True)
class VerifiedGroundedExecution:
    """Verifier-issued interpretation of an attested execution record."""

    record: GroundedExecutionRecord
    observed_outcome: str
    verified: bool = True
    epistemic_class: EpistemicOutcomeClass | None = None

    def __post_init__(self) -> None:
        if self.observed_outcome not in {"success", "failure", "not_observed"}:
            raise GroundedExecutionError("invalid grounded observed outcome")
        if self.verified is not True:
            raise GroundedExecutionError("verified execution must be verifier-issued")
        expected_outcome = (
            "success"
            if self.record.observation.epistemic_class
            is EpistemicOutcomeClass.TASK_SUCCESS
            else "failure"
            if self.record.observation.epistemic_class
            is EpistemicOutcomeClass.TASK_FAILURE
            else "not_observed"
        )
        if self.observed_outcome != expected_outcome:
            raise GroundedExecutionError(
                "verified outcome does not match the attested epistemic class"
            )
        if self.epistemic_class is None:
            object.__setattr__(
                self, "epistemic_class", self.record.observation.epistemic_class
            )
        elif not isinstance(self.epistemic_class, EpistemicOutcomeClass):
            object.__setattr__(
                self, "epistemic_class", EpistemicOutcomeClass(self.epistemic_class)
            )
        if self.epistemic_class is not self.record.observation.epistemic_class:
            raise GroundedExecutionError(
                "verified epistemic class does not match attested execution facts"
            )

    @property
    def action(self) -> Action:
        return self.record.action


class GroundedExecutionVerifier:
    """Independently recompute record integrity before constitutional use."""

    def __init__(
        self,
        trusted_executor: TrustedExecutorIdentity,
        delivery_ledger: GroundedDeliveryLedger | None = None,
    ) -> None:
        if not isinstance(trusted_executor, TrustedExecutorIdentity):
            raise GroundedExecutionError("verifier requires a trusted executor identity")
        self._trusted_executor = trusted_executor
        self._attestor = ExecutionAttestor(
            public_key=base64.b64decode(trusted_executor.public_key),
            attestor_id=trusted_executor.attestor_id,
        )
        self.delivery_ledger = delivery_ledger or GroundedDeliveryLedger()

    @classmethod
    def from_record(
        cls,
        record: GroundedExecutionRecord,
        *,
        trusted_executor: TrustedExecutorIdentity,
        delivery_ledger: GroundedDeliveryLedger | None = None,
    ) -> "GroundedExecutionVerifier":
        if not isinstance(record, GroundedExecutionRecord):
            raise GroundedExecutionError("record is required to recover verifier identity")
        return cls(trusted_executor, delivery_ledger=delivery_ledger)

    @staticmethod
    def _validate_state(
        request: GroundedExecutionRequest, authorized_state: TaskState
    ) -> None:
        GroundedExecutionExecutor._validate_authorization(request, authorized_state)

    def verify(
        self,
        record: GroundedExecutionRecord,
        *,
        request: GroundedExecutionRequest,
        authorized_state: TaskState,
    ) -> VerifiedGroundedExecution:
        """Reject forged, stale, malformed, or self-reported execution records."""

        if not isinstance(record, GroundedExecutionRecord):
            raise GroundedExecutionRejected("unattested execution cannot be verified")
        self._validate_state(request, authorized_state)
        expected_identity = (
            record.request_id == request.request_id
            and record.transaction_id == request.transaction_id
            and record.objective_id == request.objective_id
            and record.task_state_id == request.task_state_id
            and record.task_state_version == request.task_state_version
            and record.action == request.action
        )
        if not expected_identity:
            raise GroundedExecutionRejected("record identity does not match sealed request")
        if record.input_hash != request.input_hash:
            raise GroundedExecutionRejected("record input hash does not match request")
        if record.provenance.workspace_hash != _workspace_hash(request.files):
            raise GroundedExecutionRejected("record workspace provenance does not match input")
        expected_witness = (
            _digest(
                {
                    "request_input_hash": request.input_hash,
                    "causal_target": dict(request.causal_target),
                    "causal_criterion_binding": _causal_criterion_binding(request),
                }
            )
            if request.causal_target is not None
            else None
        )
        if record.provenance.causal_criterion_binding != _causal_criterion_binding(request):
            raise GroundedExecutionRejected(
                "record causal criterion binding does not match sealed request"
            )
        if record.provenance.causal_target_witness != expected_witness:
            raise GroundedExecutionRejected(
                "record causal target witness does not match sealed request"
            )
        if not (
            record.provenance.resource_limits_enforced
            and record.provenance.cleanup_verified
            and record.provenance.isolation == "user_mount_network_pid_namespace"
        ):
            raise GroundedExecutionRejected("record lacks bounded isolation provenance")
        if record.output_hash != _digest(record.observation.to_dict()):
            raise GroundedExecutionRejected("record output hash is invalid")
        unsigned = record.unsigned_payload()
        expected_hash = _digest(unsigned)
        if record.record_hash != expected_hash:
            raise GroundedExecutionRejected("record hash is invalid")
        signed_payload = {**unsigned, "record_hash": record.record_hash}
        if not self._attestor.verifies(signed_payload, record.attestation):
            raise GroundedExecutionRejected("record lacks a valid executor attestation")

        observation = record.observation
        outcome_class = observation.epistemic_class
        if outcome_class in _CREDITABLE_OUTCOME_CLASSES and not (
            record.provenance.child_runtime_verified
            and record.provenance.child_runtime_fingerprint
        ):
            raise GroundedExecutionRejected(
                "creditable record lacks verified child runtime provenance"
            )
        if observation.status is ExecutionStatus.COMPLETED:
            if not (
                observation.execution_completed
                and observation.exit_code == 0
                and observation.tests_run > 0
                and observation.tests_passed == observation.tests_run
                and observation.tests_failed == 0
            ):
                raise GroundedExecutionRejected(
                    "completed record does not satisfy independent success facts"
                )
            outcome = "success"
        elif observation.status is ExecutionStatus.FAILED:
            if not observation.execution_completed:
                raise GroundedExecutionRejected("failed record did not complete execution")
            outcome = "failure" if observation.tests_run > 0 else "not_observed"
        elif observation.status in {
            ExecutionStatus.TIMEOUT,
            ExecutionStatus.SETUP_FAILED,
            ExecutionStatus.EXECUTION_FAILED,
        }:
            outcome = "not_observed"
        else:  # pragma: no cover - enum exhaustiveness guard.
            raise GroundedExecutionRejected("unknown execution status")
        return VerifiedGroundedExecution(record, outcome, epistemic_class=outcome_class)


def replay_grounded_execution(
    record: GroundedExecutionRecord,
    *,
    request: GroundedExecutionRequest,
    authorized_state: TaskState,
    verifier: GroundedExecutionVerifier,
) -> VerifiedGroundedExecution:
    """Structurally replay an immutable record without starting another child."""

    return verifier.verify(
        record,
        request=request,
        authorized_state=authorized_state,
    )


def make_grounded_action(
    objective_id: str,
    *,
    action_id: str | None = None,
) -> Action:
    """Create the sole candidate action shape accepted by this adapter."""

    return Action(
        action_id or f"{objective_id}-action",
        objective_id,
        "run_bounded_pytest",
        "isolated_workspace",
        parameters={"runner": "pytest", "declared_files_only": True},
        preconditions=("objective_acquired",),
        authority=Authority.KRAKEN_CANDIDATE,
    )


__all__ = [
    "ExecutionAttestation",
    "ExecutionAttestor",
    "ExecutionFailureCode",
    "ExecutionLimits",
    "ExecutionObservation",
    "ExecutionProvenance",
    "ExecutionStatus",
    "GroundedExecutionError",
    "GroundedDeliveryLedger",
    "GroundedExecutionExecutor",
    "GroundedExecutionRecord",
    "GroundedExecutionRejected",
    "GroundedExecutionRequest",
    "GroundedExecutionVerifier",
    "TrustedExecutorIdentity",
    "VerifiedGroundedExecution",
    "make_grounded_action",
    "replay_grounded_execution",
    "task_state_from_dict",
    "task_state_from_json",
    "task_state_to_json",
]