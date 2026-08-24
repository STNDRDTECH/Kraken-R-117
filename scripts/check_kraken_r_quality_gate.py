"""Fail-closed structural checks for the Kraken-R GitHub quality gate."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs/kraken_r/ROUNDS_1_8_ACCEPTANCE_MANIFEST.json"
EXPECTED_MANIFEST_BLOB_SHA = "4f650aff4ce7af7e66035ef976df5a4eca4d441f"
ALLOWED_INFRASTRUCTURE_PATHS = {
    ".github/workflows/kraken-r-quality-gate.yml",
    "docs/kraken_r/ROUNDS_1_8_ACCEPTANCE_MANIFEST.json",
    "scripts/check_kraken_r_quality_gate.py",
}
FORBIDDEN_RUNTIME_SUFFIXES = {".db", ".sqlite", ".wal", ".shm", ".pyc"}
FORBIDDEN_RUNTIME_PARTS = {".rogal", ".pytest_cache", "__pycache__", "session_logs"}


def fail(message: str) -> "NoReturn":
    print(f"quality gate: FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        fail(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def blob_sha(path: Path) -> str:
    content = path.read_bytes()
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def is_forbidden_runtime_path(path: str) -> bool:
    candidate = Path(path)
    return bool(
        FORBIDDEN_RUNTIME_PARTS.intersection(candidate.parts)
        or candidate.suffix in FORBIDDEN_RUNTIME_SUFFIXES
        or candidate.name.endswith((".db", ".sqlite"))
    )


def main() -> int:
    if not MANIFEST_PATH.is_file():
        fail(f"missing manifest: {MANIFEST_PATH.relative_to(ROOT)}")
    if blob_sha(MANIFEST_PATH) != EXPECTED_MANIFEST_BLOB_SHA:
        fail("acceptance manifest was altered; update it only through review")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        fail("unsupported acceptance manifest schema")
    accepted_round_8 = manifest.get("accepted_round_8_commit")
    accepted_round_7 = manifest.get("accepted_round_7_commit")
    if not accepted_round_8 or not accepted_round_7:
        fail("manifest is missing accepted ancestry")
    if git("rev-parse", f"{accepted_round_8}^").strip() != accepted_round_7:
        fail("accepted Round 8 no longer has the accepted Round 7 parent")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", accepted_round_8, "HEAD"],
        cwd=ROOT,
        check=False,
    )
    if ancestry.returncode:
        fail("HEAD is not descended from accepted Round 8")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 34:
        fail("manifest must contain exactly 34 protected artifacts")
    artifact_paths = [item.get("path") for item in artifacts]
    if len(set(artifact_paths)) != len(artifact_paths):
        fail("manifest contains duplicate artifact paths")
    tracked = set(git("ls-files", "-z").split("\0"))
    for artifact in artifacts:
        path_text = artifact.get("path")
        if not isinstance(path_text, str) or not path_text:
            fail("manifest contains an invalid artifact path")
        path = Path(path_text)
        if path.is_absolute() or ".." in path.parts:
            fail(f"unsafe artifact path: {path_text}")
        if path_text not in tracked:
            fail(f"protected artifact is not tracked: {path_text}")
        file_path = ROOT / path
        if not file_path.is_file():
            fail(f"protected artifact is missing: {path_text}")
        if blob_sha(file_path) != artifact.get("blob_sha"):
            fail(f"protected artifact hash mismatch: {path_text}")

    required_tests = manifest.get("required_test_paths")
    if sorted(required_tests or []) != sorted(
        path for path in tracked if path.startswith("tests/test_kraken_r") and path.endswith(".py")
    ):
        fail("required Kraken-R test set does not match the manifest")
    for path_text in [*manifest.get("required_validator_inputs", []), *required_tests]:
        if path_text not in tracked or not (ROOT / path_text).is_file():
            fail(f"required validator/test input is missing: {path_text}")

    runtime_paths = sorted(path for path in tracked if is_forbidden_runtime_path(path))
    if runtime_paths:
        fail("forbidden runtime artifacts are tracked: " + ", ".join(runtime_paths))

    changed = []
    for line in git("diff", "--name-only", accepted_round_8, "HEAD").splitlines():
        if line and line not in ALLOWED_INFRASTRUCTURE_PATHS:
            changed.append(line)
    if changed:
        fail("protected or unrelated files changed: " + ", ".join(sorted(changed)))
    whitespace = subprocess.run(
        ["git", "diff", "--check", accepted_round_8, "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if whitespace.returncode:
        fail("whitespace errors detected:\n" + whitespace.stdout)

    print(
        "quality gate: structural checks passed "
        f"({len(artifacts)} artifacts, {len(required_tests)} Kraken-R test files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())