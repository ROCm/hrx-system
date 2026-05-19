#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0
"""Vendors a pruned IREE runtime snapshot into HRX.

The normal git workflow is:

  1. import-pristine: commit an explicit, pruned IREE snapshot.
  2. apply-patches: apply each HRX-local IREE patch as its own commit.
  3. dump-patches: regenerate the patch directory from those commits.

The update/check/diff commands remain useful for local validation and compare
against the final patched tree.
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SCRIPT_VERSION = 1
DEFAULT_VENDOR_DIR = Path("third_party/iree-runtime")
DEFAULT_PATCH_DIR = Path("scripts/iree-runtime-patches")

# Paths copied from the IREE superproject archive. Submodules are handled below.
VENDOR_PATHS = (
    ".gitmodules",
    "CMakeLists.txt",
    "LICENSE",
    "README.md",
    "build_tools/cmake",
    "build_tools/embed_data",
    "build_tools/third_party/flatcc",
    "build_tools/third_party/libbacktrace",
    "build_tools/third_party/tracy",
    "build_tools/third_party/llvm-project/CMakeLists.txt",
    "build_tools/tracing",
    "compiler/bindings/c",
    "runtime",
)
STUB_CMAKE_DIRS = (
    "compiler",
    "tests",
    "tools",
)

# Submodule contents copied as source. HSA headers are deliberately external:
# HRX/ROCm provides them with find_package(hsa-runtime64).
VENDORED_SUBMODULES = (
    "third_party/benchmark",
    "third_party/flatcc",
    "third_party/tracy",
)
EXTERNAL_DEPS = (
    {
        "name": "hsa-runtime64",
        "kind": "cmake-package",
        "reason": "ROCm supplies HSA headers and libhsa-runtime64.",
        "replaces_iree_path": "third_party/hsa-runtime-headers",
    },
)

EXCLUDE_DIR_NAMES = {
    ".git",
    ".github",
    "__pycache__",
}
EXCLUDE_PATH_SUFFIXES = (
    ("bindings", "python"),
    ("runtime", "demo"),
    ("tokenizer",),
)
EXCLUDE_FILE_SUFFIXES = ()
EXCLUDE_FILE_NAMES = {
    "BUILD",
    "BUILD.bazel",
    "MODULE.bazel",
    "WORKSPACE",
}


class VendorError(RuntimeError):
    pass


@dataclass(frozen=True)
class SubmoduleInfo:
    path: str
    url: str | None
    commit: str


@dataclass(frozen=True)
class GitDependency:
    name: str
    url: str
    commit: str
    destination: str
    reason: str


@dataclass(frozen=True)
class GitDependencyInfo:
    name: str
    url: str
    commit: str
    destination: str
    reason: str


VENDORED_GIT_DEPS = (
    GitDependency(
        name="mimalloc",
        url="https://github.com/microsoft/mimalloc.git",
        commit="51c09e7b6a0ac5feeba998710f00c7dd7aa67bbf",
        destination="third_party/mimalloc",
        reason="IREE_ALLOCATOR_SYSTEM=mimalloc consumes include/ and src/static.c.",
    ),
    GitDependency(
        name="libbacktrace",
        url="https://github.com/ianlancetaylor/libbacktrace.git",
        commit="b9e40069c0b47a722286b94eb5231f7f05c08713",
        destination="third_party/libbacktrace",
        reason="IREE_ENABLE_LIBBACKTRACE builds this source with IREE's CMake wrapper.",
    ),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(
    args: Sequence[str],
    cwd: Path,
    *,
    check: bool = True,
    capture: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        env=env,
    )
    if check and result.returncode != 0:
        detail = ""
        if capture:
            detail = f"\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        raise VendorError(f"command failed in {cwd}: {' '.join(args)}{detail}")
    return result


def git(cwd: Path, *args: str) -> str:
    return run(("git", *args), cwd=cwd).stdout.strip()


def resolve_commit(iree_repo: Path, ref: str) -> str:
    return git(iree_repo, "rev-parse", "--verify", f"{ref}^{{commit}}")


def archive_ref(iree_repo: Path, ref: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    command = f"git archive --format=tar {ref} | tar -xf - -C {destination}"
    run(("bash", "-lc", command), cwd=iree_repo)


def should_exclude(path: Path) -> bool:
    parts = path.parts
    if any(part in EXCLUDE_DIR_NAMES for part in parts):
        return True
    if any(parts[-len(suffix) :] == suffix for suffix in EXCLUDE_PATH_SUFFIXES):
        return True
    name = path.name
    if name in EXCLUDE_FILE_NAMES:
        return True
    return any(name.endswith(suffix) for suffix in EXCLUDE_FILE_SUFFIXES)


def copy_filtered(src: Path, dst: Path) -> None:
    if should_exclude(src):
        return
    if src.is_symlink():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(os.readlink(src), dst)
        return
    if src.is_dir():
        for child in src.iterdir():
            copy_filtered(child, dst / child.name)
        return
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_paths(extracted_root: Path, destination: Path, paths: Iterable[str]) -> None:
    for rel in paths:
        src = extracted_root / rel
        if not src.exists() and not src.is_symlink():
            raise VendorError(f"IREE path listed for vendoring does not exist: {rel}")
        copy_filtered(src, destination / rel)


def write_stub_cmake_dirs(destination: Path) -> None:
    for rel in STUB_CMAKE_DIRS:
        stub = destination / rel / "CMakeLists.txt"
        if stub.exists():
            continue
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text(
            "# Stub generated by scripts/vendor_iree_runtime.py for HRX's "
            "pruned IREE runtime import.\n",
            encoding="utf-8",
        )


def submodule_commit(iree_repo: Path, ref: str, submodule_path: str) -> str:
    entry = git(iree_repo, "ls-tree", ref, submodule_path)
    if not entry:
        raise VendorError(f"submodule not found at {ref}: {submodule_path}")
    parts = entry.split()
    if len(parts) < 3 or parts[1] != "commit":
        raise VendorError(f"path is not a submodule at {ref}: {submodule_path}")
    return parts[2]


def submodule_url(extracted_root: Path, submodule_path: str) -> str | None:
    gitmodules = extracted_root / ".gitmodules"
    if not gitmodules.exists():
        return None
    result = run(
        (
            "git",
            "config",
            "-f",
            str(gitmodules),
            "--get",
            f"submodule.{submodule_path}.url",
        ),
        cwd=extracted_root,
        check=False,
    )
    return result.stdout.strip() or None


def assert_submodule_worktree(iree_repo: Path, submodule_path: str, commit: str) -> Path:
    submodule = iree_repo / submodule_path
    if not submodule.exists():
        raise VendorError(
            f"required submodule worktree is missing: {submodule_path}; "
            "run git submodule update for that path"
        )
    head = git(submodule, "rev-parse", "HEAD")
    if head != commit:
        raise VendorError(
            f"submodule {submodule_path} is at {head}, but IREE ref expects {commit}"
        )
    status = git(submodule, "status", "--short")
    if status:
        raise VendorError(f"submodule {submodule_path} has local changes:\n{status}")
    return submodule


def copy_submodules(
    iree_repo: Path,
    extracted_root: Path,
    destination: Path,
    ref: str,
) -> list[SubmoduleInfo]:
    infos: list[SubmoduleInfo] = []
    for submodule_path in VENDORED_SUBMODULES:
        commit = submodule_commit(iree_repo, ref, submodule_path)
        submodule = assert_submodule_worktree(iree_repo, submodule_path, commit)
        copy_filtered(submodule, destination / submodule_path)
        infos.append(
            SubmoduleInfo(
                path=submodule_path,
                url=submodule_url(extracted_root, submodule_path),
                commit=commit,
            )
        )
    return infos


def materialize_git_dependency(dep: GitDependency, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    checkout = work_dir / dep.name
    run(("git", "init", str(checkout)), cwd=work_dir)
    git(checkout, "remote", "add", "origin", dep.url)
    fetch_result = run(
        ("git", "fetch", "--depth=1", "origin", dep.commit),
        cwd=checkout,
        check=False,
    )
    if fetch_result.returncode != 0:
        # Some hosts do not allow fetching an arbitrary object by SHA. Fall back
        # to a normal fetch while keeping checkout pinned to the exact commit.
        run(("git", "fetch", "origin"), cwd=checkout)
    git(checkout, "checkout", "--detach", dep.commit)
    status = git(checkout, "status", "--short")
    if status:
        raise VendorError(f"git dependency {dep.name} has local changes:\n{status}")
    return checkout


def copy_git_dependencies(destination: Path, work_dir: Path) -> list[GitDependencyInfo]:
    infos: list[GitDependencyInfo] = []
    for dep in VENDORED_GIT_DEPS:
        checkout = materialize_git_dependency(dep, work_dir / "git-deps")
        copy_filtered(checkout, destination / dep.destination)
        infos.append(
            GitDependencyInfo(
                name=dep.name,
                url=dep.url,
                commit=dep.commit,
                destination=dep.destination,
                reason=dep.reason,
            )
        )
    return infos


def patch_files(patch_dir: Path) -> list[Path]:
    if not patch_dir.exists():
        return []
    return sorted(path for path in patch_dir.iterdir() if path.suffix == ".patch")


def apply_patches(destination: Path, patches: Sequence[Path]) -> None:
    env = os.environ.copy()
    # Keep git from discovering the enclosing HRX repository. Patches are
    # relative to the vendored IREE root, not to the HRX repo root.
    env["GIT_CEILING_DIRECTORIES"] = str(destination.parent)
    for patch in patches:
        run(
            ("git", "apply", "--unsafe-paths", str(patch.resolve())),
            cwd=destination,
            env=env,
        )


def upstream_commit_date(iree_repo: Path, commit: str) -> str:
    return git(iree_repo, "show", "-s", "--format=%cI", commit)


def write_metadata(
    destination: Path,
    *,
    iree_repo: Path,
    ref: str,
    commit: str,
    submodules: Sequence[SubmoduleInfo],
    git_dependencies: Sequence[GitDependencyInfo],
) -> None:
    metadata = {
        "schema": 1,
        "script_version": SCRIPT_VERSION,
        "upstream": {
            "repository": "https://github.com/iree-org/iree.git",
            "requested_ref": ref,
            "commit": commit,
            "commit_date": upstream_commit_date(iree_repo, commit),
        },
        "paths": list(VENDOR_PATHS),
        "stub_cmake_dirs": list(STUB_CMAKE_DIRS),
        "excluded": {
            "directory_names": sorted(EXCLUDE_DIR_NAMES),
            "file_names": sorted(EXCLUDE_FILE_NAMES),
            "file_suffixes": list(EXCLUDE_FILE_SUFFIXES),
            "path_suffixes": [list(suffix) for suffix in EXCLUDE_PATH_SUFFIXES],
        },
        "submodules": [
            {
                "path": info.path,
                "url": info.url,
                "commit": info.commit,
            }
            for info in submodules
        ],
        "git_dependencies": [
            {
                "name": info.name,
                "url": info.url,
                "commit": info.commit,
                "destination": info.destination,
                "reason": info.reason,
            }
            for info in git_dependencies
        ],
        "external_dependencies": list(EXTERNAL_DEPS),
    }
    with metadata_path(destination).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")


def metadata_path(destination: Path) -> Path:
    return destination.parent / f"{destination.name}.HRX_VENDOR.json"


def generate_vendor_tree(
    *,
    iree_repo: Path,
    ref: str,
    destination: Path,
    patch_dir: Path,
    apply_patch_queue: bool = True,
) -> None:
    iree_repo = iree_repo.resolve()
    commit = resolve_commit(iree_repo, ref)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="hrx-iree-archive-") as temp_dir:
        extracted_root = Path(temp_dir) / "iree"
        archive_ref(iree_repo, commit, extracted_root)
        copy_paths(extracted_root, destination, VENDOR_PATHS)
        write_stub_cmake_dirs(destination)
        submodules = copy_submodules(iree_repo, extracted_root, destination, commit)
        git_dependencies = copy_git_dependencies(destination, Path(temp_dir))
    if apply_patch_queue:
        apply_patches(destination, patch_files(patch_dir.resolve()))
    write_metadata(
        destination,
        iree_repo=iree_repo,
        ref=ref,
        commit=commit,
        submodules=submodules,
        git_dependencies=git_dependencies,
    )


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_fingerprint(root: Path) -> dict[str, str]:
    fingerprint: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            fingerprint[rel] = f"symlink:{os.readlink(path)}"
        elif path.is_file():
            fingerprint[rel] = f"file:{file_digest(path)}"
        elif path.is_dir():
            fingerprint[f"{rel}/"] = "dir"
    return fingerprint


def compare_trees(expected: Path, actual: Path) -> list[str]:
    expected_fp = tree_fingerprint(expected)
    actual_fp = tree_fingerprint(actual) if actual.exists() else {}
    messages: list[str] = []
    for rel in sorted(expected_fp.keys() - actual_fp.keys()):
        messages.append(f"missing: {rel}")
    for rel in sorted(actual_fp.keys() - expected_fp.keys()):
        messages.append(f"extra: {rel}")
    for rel in sorted(expected_fp.keys() & actual_fp.keys()):
        if expected_fp[rel] != actual_fp[rel]:
            messages.append(f"changed: {rel}")
    return messages


def compare_metadata(expected_vendor_dir: Path, actual_vendor_dir: Path) -> list[str]:
    expected = metadata_path(expected_vendor_dir)
    actual = metadata_path(actual_vendor_dir)
    if not expected.exists():
        return []
    if not actual.exists():
        return [f"missing metadata: {actual}"]
    if file_digest(expected) != file_digest(actual):
        return [f"changed metadata: {actual}"]
    return []


def hrx_repo() -> Path:
    root = git(repo_root(), "rev-parse", "--show-toplevel")
    return Path(root)


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(hrx_repo()).as_posix()


def ensure_clean_worktree(repo: Path) -> None:
    status = git(repo, "status", "--short")
    if status:
        raise VendorError(
            "git working tree must be clean before committing vendor changes:\n"
            f"{status}"
        )


def git_commit(repo: Path, message: str) -> str:
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def update(args: argparse.Namespace) -> int:
    generate_vendor_tree(
        iree_repo=args.iree_repo,
        ref=args.ref,
        destination=args.vendor_dir,
        patch_dir=args.patch_dir,
    )
    print(f"updated {args.vendor_dir} from {args.ref}")
    return 0


def import_pristine(args: argparse.Namespace) -> int:
    repo = hrx_repo()
    ensure_clean_worktree(repo)
    generate_vendor_tree(
        iree_repo=args.iree_repo,
        ref=args.ref,
        destination=args.vendor_dir,
        patch_dir=args.patch_dir,
        apply_patch_queue=False,
    )
    git(
        repo,
        "add",
        repo_relative(args.vendor_dir),
        repo_relative(metadata_path(args.vendor_dir)),
    )
    commit = git_commit(
        repo,
        f"Vendor pristine IREE runtime {resolve_commit(args.iree_repo, args.ref)[:12]}",
    )
    print(f"created pristine import commit {commit}")
    return 0


def patch_subject(patch: Path) -> str:
    with patch.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("Subject:"):
                subject = line.removeprefix("Subject:").strip()
                if subject.startswith("[PATCH"):
                    closing = subject.find("]")
                    if closing != -1:
                        subject = subject[closing + 1 :].strip()
                if subject:
                    return subject
            if line.startswith("diff --git "):
                break
    return f"Apply IREE patch {patch.stem}"


def apply_patch_commit(repo: Path, vendor_dir: Path, patch: Path) -> str:
    rel_vendor = repo_relative(vendor_dir)
    am_result = run(
        ("git", "am", "--3way", f"--directory={rel_vendor}", str(patch.resolve())),
        cwd=repo,
        check=False,
    )
    if am_result.returncode == 0:
        return git(repo, "rev-parse", "HEAD")
    run(("git", "am", "--abort"), cwd=repo, check=False)

    git(repo, "apply", f"--directory={rel_vendor}", str(patch.resolve()))
    git(repo, "add", rel_vendor)
    return git_commit(repo, patch_subject(patch))


def apply_patch_queue(args: argparse.Namespace) -> int:
    repo = hrx_repo()
    ensure_clean_worktree(repo)
    patches = patch_files(args.patch_dir)
    if not patches:
        print(f"no patches found in {args.patch_dir}")
        return 0
    for patch in patches:
        commit = apply_patch_commit(repo, args.vendor_dir, patch)
        print(f"applied {patch.name} as {commit}")
    return 0


def check(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="hrx-iree-check-") as temp_dir:
        generated = Path(temp_dir) / "iree-runtime"
        generate_vendor_tree(
            iree_repo=args.iree_repo,
            ref=args.ref,
            destination=generated,
            patch_dir=args.patch_dir,
        )
        differences = compare_trees(generated, args.vendor_dir)
        differences.extend(compare_metadata(generated, args.vendor_dir))
    if differences:
        print(f"{args.vendor_dir} is out of date:", file=sys.stderr)
        for line in differences[:200]:
            print(f"  {line}", file=sys.stderr)
        if len(differences) > 200:
            print(f"  ... {len(differences) - 200} more differences", file=sys.stderr)
        return 1
    print(f"{args.vendor_dir} matches {args.ref}")
    return 0


def clear_patch_dir(patch_dir: Path) -> None:
    patch_dir.mkdir(parents=True, exist_ok=True)
    for path in patch_dir.iterdir():
        if path.suffix == ".patch":
            path.unlink()


def dump_patches(args: argparse.Namespace) -> int:
    repo = hrx_repo()
    clear_patch_dir(args.patch_dir)
    rel_vendor = repo_relative(args.vendor_dir)
    run(
        (
            "git",
            "format-patch",
            f"--relative={rel_vendor}",
            "--output-directory",
            str(args.patch_dir),
            f"{args.diffbase}..HEAD",
            "--",
            rel_vendor,
        ),
        cwd=repo,
        capture=False,
    )
    print(f"dumped patches from {args.diffbase}..HEAD to {args.patch_dir}")
    return 0


def diff(args: argparse.Namespace) -> int:
    if not args.vendor_dir.exists():
        raise VendorError(f"vendor directory does not exist: {args.vendor_dir}")
    with tempfile.TemporaryDirectory(prefix="hrx-iree-diff-") as temp_dir:
        generated = Path(temp_dir) / "iree-runtime"
        generate_vendor_tree(
            iree_repo=args.iree_repo,
            ref=args.ref,
            destination=generated,
            patch_dir=args.patch_dir,
        )
        cmp = filecmp.dircmp(args.vendor_dir, generated)
        report_dircmp(cmp)
    return 0


def report_dircmp(cmp: filecmp.dircmp[str]) -> None:
    left = Path(cmp.left)
    right = Path(cmp.right)
    for name in cmp.left_only:
        print(f"only current: {left / name}")
    for name in cmp.right_only:
        print(f"only generated: {right / name}")
    for name in cmp.diff_files:
        print(f"differs: {left / name}")
    for subcmp in cmp.subdirs.values():
        report_dircmp(subcmp)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iree-repo",
        type=Path,
        default=Path("../iree"),
        help="Path to an IREE git checkout.",
    )
    parser.add_argument(
        "--ref",
        default="05110733b50c2c0faafbe7452ab77f6e8088d33b",
        help="IREE commit/ref to import.",
    )
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=repo_root() / DEFAULT_VENDOR_DIR,
        help="Destination vendor directory.",
    )
    parser.add_argument(
        "--patch-dir",
        type=Path,
        default=repo_root() / DEFAULT_PATCH_DIR,
        help="Directory containing *.patch files relative to IREE root.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("update", help="Regenerate the vendored snapshot.")
    subparsers.add_parser(
        "import-pristine",
        help="Regenerate and commit the pristine pruned import without patches.",
    )
    subparsers.add_parser(
        "apply-patches",
        help="Apply patch files to the vendored import, one commit per patch.",
    )
    dump_parser = subparsers.add_parser(
        "dump-patches",
        help="Regenerate patch files from vendor commits after the diffbase.",
    )
    dump_parser.add_argument(
        "--diffbase",
        required=True,
        help="Pristine import commit; patches are dumped from diffbase..HEAD.",
    )
    subparsers.add_parser("check", help="Verify the vendored snapshot is current.")
    subparsers.add_parser("diff", help="List paths that differ from regeneration.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    args.iree_repo = args.iree_repo.resolve()
    args.vendor_dir = args.vendor_dir.resolve()
    args.patch_dir = args.patch_dir.resolve()
    try:
        if args.command == "update":
            return update(args)
        if args.command == "import-pristine":
            return import_pristine(args)
        if args.command == "apply-patches":
            return apply_patch_queue(args)
        if args.command == "dump-patches":
            return dump_patches(args)
        if args.command == "check":
            return check(args)
        if args.command == "diff":
            return diff(args)
        raise AssertionError(args.command)
    except VendorError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
