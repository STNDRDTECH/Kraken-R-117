"""Minimal OS-isolated test runner used by the Kraken-R candidate adapter.

This transport module intentionally lives outside ``kraken_r`` so the candidate
package retains its strict static ban on broad runtime-authority symbols.  It
exposes no generic command API: callers can run pytest against declared paths
only.
"""

from __future__ import annotations

import os
import json
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Iterable


class IsolatedRunnerError(RuntimeError):
    """Raised when the OS isolation boundary cannot be established."""


@dataclass(frozen=True)
class IsolatedPytestResult:
    exit_code: int | None
    tests_run: int
    tests_passed: int
    tests_failed: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool
    sandbox_error: str | None = None
    resource_limits_enforced: bool = False
    cleanup_verified: bool = False
    test_outcomes_complete: bool = False


@dataclass(frozen=True)
class _PytestFacts:
    """Machine-readable pytest outcomes, conservative about ambiguity."""

    collected: int
    passed: int
    failed: int
    ambiguous: bool


class _CappedCapture:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._parts: list[bytes] = []
        self._size = 0
        self.truncated = False

    def consume(self, stream) -> None:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            remaining = self._limit - self._size
            if remaining > 0:
                self._parts.append(chunk[:remaining])
                self._size += min(len(chunk), remaining)
            if len(chunk) > remaining:
                self.truncated = True

    def text(self) -> str:
        suffix = b"\n[output truncated]\n" if self.truncated else b""
        return (b"".join(self._parts) + suffix).decode("utf-8", errors="replace")


_SUMMARY = re.compile(r"(?P<count>\d+) (?P<label>passed|failed|error|errors)\b")
_FACTS = re.compile(r"^KRAKEN_PYTEST_FACTS:(\{.*\})$", re.MULTILINE)
_LIMITS = re.compile(r"^KRAKEN_LIMITS:(\d+):(\d+):(\d+)$", re.MULTILINE)


def _parse_counts(output: str) -> tuple[int, int, int, bool]:
    facts = _FACTS.search(output)
    if facts:
        try:
            payload = json.loads(facts.group(1))
            collected = int(payload["collected"])
            passed = int(payload["passed"])
            failed = int(payload["failed"])
            ambiguous = bool(payload["ambiguous"])
            if (
                collected >= 0
                and passed >= 0
                and failed >= 0
                and passed + failed <= collected
            ):
                complete = not ambiguous and passed + failed == collected
                return passed + failed, passed, failed, complete
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    passed = 0
    failed = 0
    for match in _SUMMARY.finditer(output):
        count = int(match.group("count"))
        label = match.group("label")
        if label == "passed":
            passed += count
        else:
            failed += count
    # Summary parsing cannot distinguish test assertions from collection or
    # fixture failures.  Preserve diagnostics but never promote it as evidence.
    return passed + failed, passed, failed, False


def _limits_verified(
    output: str, *, cpu_seconds: int, memory_bytes: int, output_bytes: int
) -> bool:
    match = _LIMITS.search(output)
    if match is None:
        return False
    observed = tuple(int(value) for value in match.groups())
    expected = (
        cpu_seconds,
        max(65536, memory_bytes // 1024),
        max(2, output_bytes // 512),
    )
    return observed == expected


def _namespace_script() -> str:
    return r"""
set -eu
root=$1
workspace=$2
python_bin=$3
chroot_bin=$4
venv=$5
shift 5
mount --make-rprivate /
mount -t tmpfs -o mode=755,size=64m tmpfs "$root"
 mkdir -p "$root/work" "$root/nix" "$root/usr" "$root/bin" "$root/lib" "$root/lib64" "$root/etc" "$root/tmp" "$root/dev" "$root/venv" "$root/runner"
mount --bind "$workspace" "$root/work"
mount -o remount,bind,rw "$root/work"
mount --bind "$venv" "$root/venv"
mount -o remount,bind,ro "$root/venv"
for source in /nix /usr /bin /lib /lib64 /etc; do
  if [ -e "$source" ]; then
    mkdir -p "$root$source"
    mount --rbind "$source" "$root$source"
    mount -o remount,bind,ro "$root$source"
  fi
done
mount -t tmpfs -o mode=1777,size=32m tmpfs "$root/tmp"
 cat > "$root/runner/pytest.ini" <<'PYTEST_CONFIG'
[pytest]
addopts =
PYTEST_CONFIG
for device in null zero urandom; do
  if [ -e "/dev/$device" ]; then
    : > "$root/dev/$device"
    mount --bind "/dev/$device" "$root/dev/$device"
    mount -o remount,bind,ro "$root/dev/$device"
  fi
done
ulimit -t "$KRAKEN_CPU_SECONDS"
ulimit -d "$KRAKEN_MEMORY_KIB"
ulimit -f "$KRAKEN_OUTPUT_BLOCKS"
printf 'KRAKEN_LIMITS:%s:%s:%s\n' "$(ulimit -t)" "$(ulimit -d)" "$(ulimit -f)"
exec "$chroot_bin" "$root" /bin/sh -ceu 'cd /work; exec "$@"' sh "$python_bin" -c '
import json
import sys
import pytest

class Facts:
    def __init__(self):
        self.collected = 0
        self.passed = 0
        self.failed = 0
        self.ambiguous = False
    def pytest_collection_finish(self, session):
        self.collected = len(session.items)
        def reject_plugin_mutation(*args, **kwargs):
            raise RuntimeError("candidate pytest plugins are not permitted")
        session.config.pluginmanager.register = reject_plugin_mutation
        session.config.pluginmanager.unregister = reject_plugin_mutation
    def pytest_collectreport(self, report):
        if report.failed:
            self.ambiguous = True
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        outcome = yield
        if outcome.excinfo is None:
            self.passed += 1
        else:
            self.failed += 1

facts = Facts()
code = pytest.main([
    "-c", "/runner/pytest.ini",
    "--noconftest",
    "-q",
    "--assert=plain",
    "--disable-warnings",
    *sys.argv[1:],
], plugins=[facts])
print("KRAKEN_PYTEST_FACTS:" + json.dumps({"collected": facts.collected, "passed": facts.passed, "failed": facts.failed, "ambiguous": facts.ambiguous}, sort_keys=True))
raise SystemExit(code)
' "$@"
"""


def _readers(process: subprocess.Popen[bytes], limit: int):
    stdout = _CappedCapture(limit)
    stderr = _CappedCapture(limit)
    threads = [
        threading.Thread(target=stdout.consume, args=(process.stdout,), daemon=True),
        threading.Thread(target=stderr.consume, args=(process.stderr,), daemon=True),
    ]
    for thread in threads:
        thread.start()
    return stdout, stderr, threads


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _process_group_is_gone(process_group_id: int) -> bool:
    """Confirm no descendant still occupies the sandbox process group."""

    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _terminate_and_verify_process_group(process_group_id: int) -> bool:
    """Kill escaped descendants too, then observe that their group has vanished."""

    deadline = monotonic() + 1.0
    while True:
        if _process_group_is_gone(process_group_id):
            return True
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            return True
        if monotonic() >= deadline:
            return _process_group_is_gone(process_group_id)
        sleep(0.02)


def run_isolated_pytest(
    workspace: str | Path,
    test_paths: Iterable[str],
    *,
    timeout_seconds: float,
    cpu_seconds: int,
    memory_bytes: int,
    output_bytes: int,
) -> IsolatedPytestResult:
    """Run pytest in a new user/mount/network/PID namespace or fail closed."""

    workspace_path = Path(workspace).resolve()
    if not workspace_path.is_dir():
        raise IsolatedRunnerError("declared workspace does not exist")
    unshare = shutil.which("unshare")
    chroot = shutil.which("chroot")
    if unshare is None or chroot is None:
        raise IsolatedRunnerError("OS namespace isolation tools are unavailable")
    venv_path = Path(sys.executable).parent.parent
    if not (venv_path / "bin" / "python").exists():
        raise IsolatedRunnerError("isolated Python runtime is unavailable")
    paths = tuple(str(Path("/work") / item) for item in test_paths)
    if not paths:
        raise IsolatedRunnerError("at least one declared test path is required")
    with tempfile.TemporaryDirectory(prefix="kraken-r-sandbox-root-") as root:
        environment = {
            "HOME": "/tmp",
            "LANG": "C.UTF-8",
            "PATH": "/nix/var/nix/profiles/default/bin:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "KRAKEN_CPU_SECONDS": str(cpu_seconds),
            "KRAKEN_MEMORY_KIB": str(max(65536, memory_bytes // 1024)),
            "KRAKEN_OUTPUT_BLOCKS": str(max(2, output_bytes // 512)),
        }
        command = [
            unshare,
            "--user",
            "--map-root-user",
            "--mount",
            "--net",
            "--pid",
            "--fork",
            "--kill-child=SIGKILL",
            "/bin/sh",
            "-ceu",
            _namespace_script(),
            "kraken-r-namespace",
            root,
            str(workspace_path),
            "/venv/bin/python",
            chroot,
            str(venv_path),
            *paths,
        ]
        started = monotonic()
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            start_new_session=True,
        )
        stdout, stderr, threads = _readers(process, output_bytes)
        timed_out = False
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_process_group(process)
            process.wait(timeout=2.0)
        for thread in threads:
            thread.join(timeout=1.0)
        captured_stdout = stdout.text()
        captured_stderr = stderr.text()
        elapsed = monotonic() - started
        combined_output = f"{captured_stdout}\n{captured_stderr}"
        limits_enforced = _limits_verified(
            combined_output,
            cpu_seconds=cpu_seconds,
            memory_bytes=memory_bytes,
            output_bytes=output_bytes,
        )
        cleanup_verified = _terminate_and_verify_process_group(process.pid)
        if timed_out:
            return IsolatedPytestResult(
                None,
                0,
                0,
                0,
                captured_stdout,
                captured_stderr or "isolated execution exceeded wall-clock timeout",
                elapsed,
                True,
                None,
                limits_enforced,
                cleanup_verified,
                False,
            )
        tests_run, tests_passed, tests_failed, test_outcomes_complete = _parse_counts(
            combined_output
        )
        sandbox_error = None
        if process.returncode is None:
            sandbox_error = "namespace process did not report an exit code"
        elif process.returncode == 1 and tests_run == 0:
            sandbox_error = "isolated runner failed before pytest collected tests"
        return IsolatedPytestResult(
            process.returncode,
            tests_run,
            tests_passed,
            tests_failed,
            captured_stdout,
            captured_stderr,
            elapsed,
            False,
            sandbox_error,
            limits_enforced,
            cleanup_verified,
            test_outcomes_complete,
        )