"""Verify a checkout of the isolated Round 7 Kraken-R publication."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs/kraken_r/ROUND_7_PUBLICATION_MANIFEST.json"
PUBLISHED_ORIGIN = "github.com/STNDRDTECH/Kraken-R-117"
PUBLISHED_REPOSITORY = "https://github.com/STNDRDTECH/Kraken-R-117.git"


def git_blob_sha(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


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


def is_isolated_published_checkout(root: Path) -> bool:
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0 and PUBLISHED_ORIGIN in result.stdout


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
        (root / "docs/kraken_r/ROUND_7_PUBLICATION_MANIFEST.json").read_text()
    )
    if published_checkout:
        expected_parent = manifest["accepted_parent_commits"]["published_round_6"]
        actual_parent = direct_parent(root)
        if actual_parent != expected_parent:
            raise RuntimeError(
                f"unexpected direct parent: {actual_parent}; "
                f"expected {expected_parent}"
            )

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

    test_paths = sorted(root.glob("tests/test_kraken_r*.py"))
    if not test_paths:
        raise RuntimeError("no Kraken-R tests found")
    pytest = run([sys.executable, "-m", "pytest", "-q", *map(str, test_paths)], root)
    cli = run([sys.executable, "-m", "kraken_r", "--json"], root)
    cli_result = json.loads(cli.stdout)
    if cli_result.get("ok") is not True:
        raise RuntimeError("Kraken-R CLI validation did not report ok")

    print(f"artifacts: {len(manifest['artifacts'])} verified")
    if published_checkout:
        print("lineage: direct Round 6 parent verified")
    print(pytest.stdout, end="")
    print("python -m kraken_r --json: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "--published-checkout",
        action="store_true",
        help="require the direct accepted Round 6 parent for an isolated published ref",
    )
    args = parser.parse_args(argv)
    if not args.published_checkout:
        return verify_checkout(ROOT, published_checkout=False)
    if is_isolated_published_checkout(ROOT):
        return verify_checkout(ROOT, published_checkout=True)

    manifest = json.loads(MANIFEST_PATH.read_text())
    with tempfile.TemporaryDirectory(prefix="kraken-r-published-") as directory:
        clone_root = Path(directory) / "checkout"
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
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