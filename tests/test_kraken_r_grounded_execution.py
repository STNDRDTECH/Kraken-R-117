"""Acceptance tests for independently verified bounded candidate execution."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from kraken_r import (
    Authority,
    EvidenceGrade,
    ExecutionAttestation,
    ExecutionFailureCode,
    EpistemicOutcomeClass,
    ExecutionLimits,
    ExecutionObservation,
    ExecutionStatus,
    GroundedExecutionExecutor,
    GroundedDeliveryLedger,
    GroundedExecutionRejected,
    GroundedExecutionRequest,
    GroundedExecutionVerifier,
    Objective,
    TaskState,
    TrustedExecutorIdentity,
    VerifiedGroundedExecution,
    make_grounded_action,
    replay_grounded_execution,
    run_constitutional_cycle,
)
from kraken_r import grounded_execution as execution_module
import bounded_execution_runner as runner_module


def _request(
    name: str,
    *,
    files: dict[str, str] | None = None,
    limits: ExecutionLimits | None = None,
) -> tuple[Objective, TaskState, GroundedExecutionRequest]:
    objective = Objective(
        f"{name}-objective",
        "Run one sealed candidate test.",
        provenance={"transaction_id": f"{name}-transaction"},
    )
    action = make_grounded_action(objective.objective_id)
    state = TaskState(
        f"{objective.objective_id}-state-5",
        objective.objective_id,
        5,
        "authorized",
        values={"action_id": action.action_id},
        authority=Authority.KRAKEN_CANDIDATE,
    )
    workspace = files or {
        "subject.py": "def answer():\n    return 42\n",
        "test_subject.py": (
            "from subject import answer\n\n"
            "def test_answer():\n"
            "    assert answer() == 42\n"
        ),
    }
    request = GroundedExecutionRequest(
        f"{name}-request",
        f"{name}-transaction",
        objective.objective_id,
        state.state_id,
        state.version,
        action,
        workspace,
        tuple(path for path in workspace if path.startswith("test_")),
        limits or ExecutionLimits(timeout_seconds=8.0),
    )
    return objective, state, request


def _execute(
    name: str,
    *,
    files: dict[str, str] | None = None,
    limits: ExecutionLimits | None = None,
):
    objective, state, request = _request(name, files=files, limits=limits)
    executor = GroundedExecutionExecutor(
        delivery_ledger=GroundedDeliveryLedger(
            Path(tempfile.mkdtemp(prefix="kraken-r-test-receipts-")) / "receipts.json"
        )
    )
    verifier = executor.verifier()
    record = executor.execute(request, authorized_state=state)
    verified = verifier.verify(record, request=request, authorized_state=state)
    return objective, state, request, executor, verifier, record, verified


def test_verified_real_execution_is_the_only_grounded_cycle_path() -> None:
    objective, state, request, executor, verifier, record, verified = _execute("real-pass")

    assert verified.observed_outcome == "success"
    assert record.observation.status is ExecutionStatus.COMPLETED
    assert record.observation.tests_run == record.observation.tests_passed == 1
    replay = replay_grounded_execution(
        record,
        request=request,
        authorized_state=state,
        verifier=verifier,
    )
    assert replay == verified
    fresh_verifier = GroundedExecutionVerifier.from_record(
        record, trusted_executor=executor.trusted_executor()
    )
    assert fresh_verifier.verify(record, request=request, authorized_state=state) == verified

    trace = run_constitutional_cycle(
        objective,
        grounded_execution=verified,
        grounded_request=request,
        grounded_verifier=verifier,
    )
    assert trace.decision.outcome == "success"
    assert trace.evidence[0].grade is EvidenceGrade.GROUNDED
    assert trace.evidence[0].source == "grounded_execution_verifier"
    assert trace.learning_update is not None
    assert trace.execution.observations["record_hash"] == record.record_hash

    with pytest.raises(Exception, match="verifier-issued"):
        run_constitutional_cycle(objective, grounded_execution=record)  # type: ignore[arg-type]


def test_duplicate_execution_and_lifecycle_delivery_are_rejected() -> None:
    objective, state, request, executor, verifier, record, verified = _execute(
        "duplicate-delivery"
    )
    with pytest.raises(GroundedExecutionRejected, match="duplicate grounded execution"):
        executor.execute(request, authorized_state=state)

    run_constitutional_cycle(
        objective,
        grounded_execution=verified,
        grounded_request=request,
        grounded_verifier=verifier,
    )
    with pytest.raises(GroundedExecutionRejected, match="duplicate grounded evidence"):
        run_constitutional_cycle(
            objective,
            grounded_execution=verified,
            grounded_request=request,
            grounded_verifier=verifier,
        )
    with pytest.raises(GroundedExecutionRejected, match="duplicate grounded settlement"):
        verifier.delivery_ledger.claim(
            "settlement", f"{objective.objective_id}-settlement", record.record_hash
        )


def test_isolated_workspace_hides_host_environment_and_network() -> None:
    files = {
        "test_isolation.py": (
            "import os\n"
            "import socket\n\n"
            "def test_isolation_boundary():\n"
            "    assert not os.path.exists('/home/runner/workspace')\n"
            "    assert 'SESSION_SECRET' not in os.environ\n"
            "    try:\n"
            "        socket.create_connection(('1.1.1.1', 53), timeout=0.2)\n"
            "    except OSError:\n"
            "        return\n"
            "    raise AssertionError('network was reachable')\n"
        )
    }
    _, _, _, _, _, record, verified = _execute("isolation-boundary", files=files)
    assert record.observation.status is ExecutionStatus.COMPLETED
    assert verified.observed_outcome == "success"


def test_timeout_kills_pid_namespace_descendants() -> None:
    files = {
        "test_descendant.py": (
            "import os\n"
            "import time\n\n"
            "def test_timeout_with_child():\n"
            "    if os.fork() == 0:\n"
            "        os.setsid()\n"
            "        os.execv('/bin/sh', ['kraken-r-escaped-descendant', '-c', 'sleep 30'])\n"
            "    time.sleep(30)\n"
        )
    }
    _, _, _, _, _, record, verified = _execute(
        "descendant-timeout",
        files=files,
        limits=ExecutionLimits(timeout_seconds=0.3, cpu_seconds=1),
    )
    assert record.observation.status is ExecutionStatus.TIMEOUT
    assert verified.observed_outcome == "not_observed"
    assert record.provenance.resource_limits_enforced is True
    assert record.provenance.cleanup_verified is True
    processes = subprocess.run(
        ["ps", "-eo", "args"], check=True, text=True, capture_output=True
    ).stdout
    assert "kraken-r-escaped-descendant" not in processes


def test_durable_restart_verification_and_duplicate_receipts(tmp_path: Path) -> None:
    receipt_path = tmp_path / "grounded-receipts.json"
    objective, state, request = _request("durable-restart")
    executor = GroundedExecutionExecutor(
        delivery_ledger=execution_module.GroundedDeliveryLedger(receipt_path)
    )
    verifier = executor.verifier()
    record = executor.execute(request, authorized_state=state)
    verified = verifier.verify(record, request=request, authorized_state=state)
    trace = run_constitutional_cycle(
        objective,
        grounded_execution=verified,
        grounded_request=request,
        grounded_verifier=verifier,
    )
    assert trace.learning_update is not None

    request_path = tmp_path / "request.json"
    state_path = tmp_path / "state.json"
    record_path = tmp_path / "record.json"
    trust_path = tmp_path / "trusted-executor.json"
    request_path.write_text(request.to_json(), encoding="utf-8")
    state_path.write_text(execution_module.task_state_to_json(state), encoding="utf-8")
    record_path.write_text(record.to_json(), encoding="utf-8")
    trust_path.write_text(
        json.dumps(executor.trusted_executor().to_dict()), encoding="utf-8"
    )
    child = """
import json
import sys
from pathlib import Path
from kraken_r import (
    GroundedDeliveryLedger,
    GroundedExecutionRecord,
    GroundedExecutionRejected,
    GroundedExecutionRequest,
    GroundedExecutionVerifier,
    TrustedExecutorIdentity,
)
from kraken_r.grounded_execution import task_state_from_json
request = GroundedExecutionRequest.from_json(Path(sys.argv[1]).read_text())
state = task_state_from_json(Path(sys.argv[2]).read_text())
record = GroundedExecutionRecord.from_json(Path(sys.argv[3]).read_text())
trusted = TrustedExecutorIdentity.from_dict(json.loads(Path(sys.argv[4]).read_text()))
ledger = GroundedDeliveryLedger(Path(sys.argv[5]))
verifier = GroundedExecutionVerifier.from_record(record, trusted_executor=trusted, delivery_ledger=ledger)
assert verifier.verify(record, request=request, authorized_state=state).observed_outcome == "success"
claims = (
    ("execution", request.request_id, request.input_hash),
    ("evidence", record.record_id, record.output_hash),
    ("settlement", request.objective_id + "-settlement", record.record_hash),
    ("learning", request.objective_id + "-learning", record.record_hash),
)
for stage, identity, binding in claims:
    try:
        ledger.claim(stage, identity, binding)
    except GroundedExecutionRejected:
        continue
    raise AssertionError("restart accepted duplicate " + stage)
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            child,
            str(request_path),
            str(state_path),
            str(record_path),
            str(trust_path),
            str(receipt_path),
        ],
        check=True,
        cwd=Path(__file__).parents[1],
    )


def test_record_key_cannot_be_its_own_trust_root() -> None:
    _, state, request, executor, _, record, _ = _execute("pinned-key")
    attacker = execution_module.ExecutionAttestor(
        attestor_id=record.attestation.attestor_id
    )
    forged = replace(
        record,
        attestation=attacker.attest(
            {**record.unsigned_payload(), "record_hash": record.record_hash}
        ),
    )
    verifier = GroundedExecutionVerifier.from_record(
        forged, trusted_executor=executor.trusted_executor()
    )
    with pytest.raises(GroundedExecutionRejected, match="attestation"):
        verifier.verify(forged, request=request, authorized_state=state)


def test_durable_receipts_replay_until_expiry(tmp_path: Path) -> None:
    now = [1_000]
    path = tmp_path / "expiry-receipts.json"
    ledger = execution_module.GroundedDeliveryLedger(
        path, receipt_ttl_seconds=10, clock=lambda: now[0]
    )
    ledger.claim("execution", "expiry-request", "sealed-input")
    assert ledger.receipt_status("execution", "expiry-request") == "active"

    restarted = execution_module.GroundedDeliveryLedger(
        path, receipt_ttl_seconds=10, clock=lambda: now[0]
    )
    with pytest.raises(GroundedExecutionRejected, match="duplicate grounded execution"):
        restarted.claim("execution", "expiry-request", "sealed-input")

    now[0] += 10
    expired = execution_module.GroundedDeliveryLedger(
        path, receipt_ttl_seconds=10, clock=lambda: now[0]
    )
    assert expired.receipt_status("execution", "expiry-request") == "expired"
    expired.claim("execution", "expiry-request", "sealed-input")


def test_durable_receipt_claim_is_atomic_across_processes(tmp_path: Path) -> None:
    receipt_path = tmp_path / "shared-receipts.json"
    start_path = tmp_path / "start"
    child = """
import sys
import time
from pathlib import Path
from kraken_r import GroundedDeliveryLedger, GroundedExecutionRejected
path = Path(sys.argv[1])
start = Path(sys.argv[2])
while not start.exists():
    time.sleep(0.005)
ledger = GroundedDeliveryLedger(path)
try:
    ledger.claim("execution", "same-request", "sealed-input")
except GroundedExecutionRejected:
    print("rejected")
else:
    print("claimed")
"""
    children = [
        subprocess.Popen(
            [sys.executable, "-c", child, str(receipt_path), str(start_path)],
            cwd=Path(__file__).parents[1],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(2)
    ]
    start_path.touch()
    outputs = [child_process.communicate(timeout=10) for child_process in children]
    assert all(child_process.returncode == 0 for child_process in children)
    assert sorted(stdout.strip() for stdout, _ in outputs) == ["claimed", "rejected"]


def test_failed_isolated_run_cannot_be_promoted_to_a_success_claim() -> None:
    files = {
        "subject.py": "def answer():\n    return 0\n",
        "test_subject.py": (
            "from subject import answer\n\n"
            "def test_answer():\n"
            "    assert answer() == 42\n"
        ),
    }
    objective, state, request, _, verifier, record, verified = _execute(
        "real-failure", files=files
    )

    assert record.observation.status is ExecutionStatus.FAILED
    assert verified.observed_outcome == "failure"
    trace = run_constitutional_cycle(
        objective,
        grounded_execution=verified,
        grounded_request=request,
        grounded_verifier=verifier,
    )
    assert trace.decision.outcome == "failure"
    assert all(item.grade is EvidenceGrade.GROUNDED for item in trace.evidence)
    with pytest.raises(Exception, match="epistemic class"):
        VerifiedGroundedExecution(record, "success")


def test_candidate_conftest_cannot_forge_a_grounded_success() -> None:
    files = {
        "conftest.py": (
            "import pytest\n\n"
            "@pytest.hookimpl(hookwrapper=True, trylast=True)\n"
            "def pytest_runtest_call(item):\n"
            "    outcome = yield\n"
            "    outcome.force_result(None)\n"
        ),
        "test_forged_success.py": (
            "def test_cannot_be_forced_to_pass():\n"
            "    assert False\n"
        ),
    }
    _, _, _, _, _, record, verified = _execute("forged-success", files=files)

    assert record.observation.status is ExecutionStatus.FAILED
    assert record.observation.failure_code is ExecutionFailureCode.TEST_FAILURE
    assert record.observation.tests_failed == 1
    assert verified.observed_outcome == "failure"


def test_candidate_cannot_register_a_pytest_plugin_to_forge_success() -> None:
    files = {
        "test_dynamic_forgery.py": (
            "class Forger:\n"
            "    def pytest_sessionfinish(self, session, exitstatus):\n"
            "        session.exitstatus = 0\n\n"
            "def test_plugin_registration_is_rejected(pytestconfig):\n"
            "    pytestconfig.pluginmanager.register(Forger(), 'forger')\n"
            "    assert False\n"
        ),
    }
    _, _, _, _, _, record, verified = _execute("dynamic-forgery", files=files)

    assert record.observation.status is ExecutionStatus.FAILED
    assert record.observation.failure_code is ExecutionFailureCode.TEST_FAILURE
    assert record.observation.tests_failed == 1
    assert verified.observed_outcome == "failure"


def test_timeout_and_malformed_test_output_withhold_evidence() -> None:
    timeout_files = {
        "test_slow.py": (
            "import time\n\n"
            "def test_waits_too_long():\n"
            "    time.sleep(2)\n"
        ),
    }
    objective, state, request, _, verifier, record, verified = _execute(
        "real-timeout",
        files=timeout_files,
        limits=ExecutionLimits(timeout_seconds=0.2, cpu_seconds=1),
    )
    assert record.observation.status is ExecutionStatus.TIMEOUT
    assert verified.observed_outcome == "not_observed"
    timeout_trace = run_constitutional_cycle(
        objective,
        grounded_execution=verified,
        grounded_request=request,
        grounded_verifier=verifier,
    )
    assert timeout_trace.evidence == ()
    assert timeout_trace.learning_update is None

    malformed_files = {
        "test_broken.py": "def test_broken(:\n    pass\n",
    }
    objective, state, request, _, verifier, malformed, malformed_verified = _execute(
        "malformed-test",
        files=malformed_files,
    )
    assert malformed.observation.status is ExecutionStatus.EXECUTION_FAILED
    assert (
        malformed.observation.epistemic_class
        is EpistemicOutcomeClass.EXECUTION_FAILURE
    )
    assert malformed_verified.observed_outcome == "not_observed"
    malformed_trace = run_constitutional_cycle(
        objective,
        grounded_execution=malformed_verified,
        grounded_request=request,
        grounded_verifier=verifier,
    )
    assert malformed_trace.evidence == ()
    assert malformed_trace.learning_update is None


def test_child_runtime_contract_and_epistemic_taxonomy_fail_closed() -> None:
    _, _, _, _, _, record, verified = _execute("runtime-contract")

    assert record.provenance.child_runtime_verified is True
    assert record.provenance.child_runtime_fingerprint is not None
    assert (
        record.observation.epistemic_class
        is EpistemicOutcomeClass.TASK_SUCCESS
    )
    assert verified.epistemic_class is EpistemicOutcomeClass.TASK_SUCCESS

    with pytest.raises(Exception, match="epistemic class"):
        replace(
            record.observation,
            epistemic_class=EpistemicOutcomeClass.TASK_FAILURE,
        )

    malformed = ExecutionObservation(
        ExecutionStatus.EXECUTION_FAILED,
        2,
        False,
        0,
        0,
        0,
        "",
        "malformed child facts",
        0.01,
        ExecutionFailureCode.RUNNER_FAILURE,
    )
    assert malformed.epistemic_class is EpistemicOutcomeClass.EXECUTION_FAILURE
    with pytest.raises(Exception, match="epistemic class"):
        replace(malformed, epistemic_class=EpistemicOutcomeClass.TASK_FAILURE)


def test_child_runtime_contract_rejects_forgery_and_workspace_escape() -> None:
    expected = {
        "contract": "kraken_r_isolated_pytest_v1",
        "python_implementation": "CPython",
        "python_version": "3.11.0",
        "pytest_version": "8.4.2",
    }
    expected_fingerprint = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    marker = "KRAKEN_CHILD_RUNTIME:" + json.dumps(expected, sort_keys=True)
    assert runner_module._runtime_fingerprint(
        marker, expected, expected_fingerprint
    ) == (True, expected_fingerprint)
    assert runner_module._runtime_fingerprint(
        marker + "\n" + marker, expected, expected_fingerprint
    ) == (False, None)
    forged = dict(expected, pytest_version="0.0.0")
    assert runner_module._runtime_fingerprint(
        "KRAKEN_CHILD_RUNTIME:" + json.dumps(forged),
        expected,
        expected_fingerprint,
    ) == (False, None)
    for path in ("/etc/passwd", "../outside.py", "nested/../../outside.py"):
        with pytest.raises(runner_module.IsolatedRunnerError, match="escapes"):
            runner_module._workspace_relative_path(path, "test path")
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        (workspace / "tests").mkdir()
        (workspace / "tests" / "escape.py").symlink_to("/etc/passwd")
        with pytest.raises(runner_module.IsolatedRunnerError, match="outside"):
            runner_module._workspace_test_paths(workspace, ("tests/escape.py",))

    unverified = runner_module.IsolatedPytestResult(
        exit_code=0,
        tests_run=1,
        tests_passed=1,
        tests_failed=0,
        stdout="1 passed",
        stderr="",
        elapsed_seconds=0.01,
        timed_out=False,
        sandbox_error=None,
        resource_limits_enforced=True,
        cleanup_verified=True,
        test_outcomes_complete=True,
        runtime_verified=False,
    )
    observed = execution_module._observation_from_isolated_result(unverified)
    assert observed.status is ExecutionStatus.EXECUTION_FAILED
    assert observed.epistemic_class is EpistemicOutcomeClass.EXECUTION_FAILURE


def test_fixture_setup_failure_cannot_be_promoted_to_grounded_test_failure() -> None:
    files = {
        "conftest.py": (
            "import pytest\n\n"
            "@pytest.fixture\n"
            "def unavailable_dependency():\n"
            "    raise RuntimeError('fixture setup failed')\n"
        ),
        "test_setup.py": (
            "def test_requires_fixture(unavailable_dependency):\n"
            "    assert True\n"
        ),
    }
    _, _, _, _, _, record, verified = _execute("fixture-setup", files=files)

    assert record.observation.status is ExecutionStatus.SETUP_FAILED
    assert record.observation.failure_code is ExecutionFailureCode.SETUP_FAILED
    assert record.observation.tests_run == 0
    assert verified.observed_outcome == "not_observed"


def test_stale_state_workspace_escape_and_tampered_self_report_fail_closed() -> None:
    objective, state, request, _, verifier, record, _ = _execute("tamper")

    stale_state = TaskState(
        f"{objective.objective_id}-state-6",
        objective.objective_id,
        6,
        "authorized",
        values={"action_id": request.action.action_id},
    )
    with pytest.raises(GroundedExecutionRejected, match="stale"):
        verifier.verify(record, request=request, authorized_state=stale_state)

    escaped_files = {
        "../outside.py": "raise AssertionError('must not materialize')\n",
        "test_subject.py": "def test_noop():\n    assert True\n",
    }
    with pytest.raises(GroundedExecutionRejected, match="escapes"):
        _request("workspace-escape", files=escaped_files)

    self_report = ExecutionObservation(
        ExecutionStatus.COMPLETED,
        0,
        True,
        1,
        1,
        0,
        "model says success",
        "",
        0.01,
    )
    forged = replace(
        record,
        output_hash=execution_module._digest(self_report.to_dict()),
        observation=self_report,
    )
    forged = replace(
        forged,
        record_hash=execution_module._digest(forged.unsigned_payload()),
        attestation=ExecutionAttestation(
            "self-reported-record",
            "not-an-executor-key",
            "self-reported-success",
            "not-a-public-key",
        ),
    )
    with pytest.raises(GroundedExecutionRejected, match="attestation"):
        verifier.verify(forged, request=request, authorized_state=state)


def test_fixture_and_grounded_ablation_is_descriptive_only() -> None:
    objective = Objective(
        "ablation-objective",
        "Compare execution source labels without a performance claim.",
        provenance={"transaction_id": "ablation-transaction"},
    )
    fixture = run_constitutional_cycle(objective)
    grounded_objective, state, request, _, verifier, _, verified = _execute(
        "ablation-grounded"
    )
    grounded = run_constitutional_cycle(
        grounded_objective,
        grounded_execution=verified,
        grounded_request=request,
        grounded_verifier=verifier,
    )

    assert fixture.evidence[0].grade is EvidenceGrade.OPERATIONAL
    assert grounded.evidence[0].grade is EvidenceGrade.GROUNDED
    assert fixture.provenance["source"] == "deterministic_fixture"
    assert grounded.provenance["source"] == "grounded_execution_verifier"
    assert "performance" not in grounded.decision.rationale.lower()