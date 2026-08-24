"""Verify a checkout of the isolated Round 8 Kraken-R publication."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs/kraken_r/ROUND_8_PUBLICATION_MANIFEST.json"
PUBLISHED_REPOSITORY = "https://github.com/STNDRDTECH/Kraken-R-117.git"
REQUIRED_RUNTIME_REMOVALS = {
    ".rogal/ROGAL_ACTIVE_DEVELOPMENT_LEDGER.md",
    ".rogal/session_logs/LATEST.md",
}


def git_blob_sha(path: Path) -> str:
    content = path.read_bytes()
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def direct_parent(root: Path) -> str:
    result = subprocess.run(
        ["git", "cat-file", "-p", "HEAD"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    parents = [
        line.removeprefix("parent ")
        for line in result.stdout.splitlines()
        if line.startswith("parent ")
    ]
    if len(parents) != 1:
        raise RuntimeError(f"expected one direct parent, found {parents!r}")
    return parents[0]


def tree_entries(root: Path, revision: str) -> dict[str, tuple[str, str, str]]:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "-z", revision],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    entries: dict[str, tuple[str, str, str]] = {}
    for raw_entry in result.stdout.split(b"\0"):
        if not raw_entry:
            continue
        metadata, raw_path = raw_entry.split(b"\t", 1)
        mode, entry_type, sha = metadata.decode("ascii").split(" ")
        entries[raw_path.decode("utf-8")] = (mode, entry_type, sha)
    return entries


def run(command: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def verify_checkout(root: Path, *, published_checkout: bool) -> int:
    manifest = json.loads(
        (root / "docs/kraken_r/ROUND_8_PUBLICATION_MANIFEST.json").read_text()
    )
    if published_checkout:
        expected_parent = manifest["accepted_parent_commits"]["published_round_7"]
        actual_parent = direct_parent(root)
        if actual_parent != expected_parent:
            raise RuntimeError(
                f"unexpected direct parent: {actual_parent}; "
                f"expected {expected_parent}"
            )
        parent_entries = tree_entries(root, expected_parent)
        candidate_entries = tree_entries(root, "HEAD")
        allowed_paths = {
            artifact["path"] for artifact in manifest["artifacts"]
        } | {
            "docs/kraken_r/ROUND_8_PUBLICATION_MANIFEST.json",
            "scripts/verify_round_8_publication.py",
        }
        undeclared = sorted(
            path for path in candidate_entries if path not in parent_entries and path not in allowed_paths
        )
        deleted = sorted(path for path in parent_entries if path not in candidate_entries)
        unexpected_deletions = sorted(set(deleted) - REQUIRED_RUNTIME_REMOVALS)
        missing_required_removals = sorted(
            path for path in REQUIRED_RUNTIME_REMOVALS if path in candidate_entries
        )
        modified = sorted(
            path
            for path in candidate_entries.keys() & parent_entries.keys()
            if candidate_entries[path] != parent_entries[path] and path not in allowed_paths
        )
        if (
            undeclared
            or unexpected_deletions
            or missing_required_removals
            or modified
        ):
            raise RuntimeError(
                "published tree changed outside the declared Round 8 set: "
                f"added={undeclared!r}, deleted={unexpected_deletions!r}, "
                f"required_removals_present={missing_required_removals!r}, "
                f"modified={modified!r}"
            )
        runtime_state = sorted(
            path
            for path in candidate_entries
            if (
                path.startswith(".rogal/")
                or "/.rogal/" in path
                or Path(path).suffix in {".wal", ".shm"}
                or Path(path).name.endswith((".db", ".sqlite"))
            )
        )
        if runtime_state:
            raise RuntimeError(f"runtime state in published tree: {runtime_state!r}")

    mismatches: list[str] = []
    for artifact in manifest["artifacts"]:
        relative_path = Path(artifact["path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"unsafe manifest path: {relative_path}")
        path = root / relative_path
        if not path.is_file():
            mismatches.append(f"missing {relative_path}")
        elif git_blob_sha(path) != artifact["blob_sha"]:
            mismatches.append(f"hash mismatch {relative_path}")
    if mismatches:
        raise RuntimeError("; ".join(mismatches))

    forbidden = [
        artifact["path"]
        for artifact in manifest["artifacts"]
        if ".rogal" in Path(artifact["path"]).parts
        or Path(artifact["path"]).suffix in {".wal", ".shm"}
        or Path(artifact["path"]).name.endswith((".db", ".sqlite"))
    ]
    if forbidden:
        raise RuntimeError(f"runtime state was included in the publication set: {forbidden!r}")

    test_paths = sorted(root.glob("tests/test_kraken_r*.py"))
    if not test_paths:
        raise RuntimeError("no Kraken-R tests found")
    pytest = run([sys.executable, "-m", "pytest", "-q", *map(str, test_paths)], root)
    validation = run([sys.executable, "-m", "kraken_r.validate"], root)

    print(f"artifacts: {len(manifest['artifacts'])} verified")
    if published_checkout:
        print("lineage: direct accepted Round 7 parent verified")
    print(pytest.stdout, end="")
    print(validation.stdout, end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "--published-checkout",
        action="store_true",
        help="verify a fresh clone of the published Round 8 branch",
    )
    args = parser.parse_args(argv)
    if not args.published_checkout:
        return verify_checkout(ROOT, published_checkout=False)

    manifest = json.loads(MANIFEST_PATH.read_text())
    with tempfile.TemporaryDirectory(prefix="kraken-r-published-") as directory:
        clone_root = Path(directory) / "checkout"
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "2",
                "--branch",
                manifest["published_ref"],
                PUBLISHED_REPOSITORY,
                str(clone_root),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return verify_checkout(clone_root, published_checkout=True)


if __name__ == "__main__":
    raise SystemExit(main())