#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap build-only dependencies for HRX scripts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from hrx_build_tools import REPO_ROOT, log, remove_tree, rocm_build_env, rocm_tool, run


CATCH2_VERSION = "3.8.1"
CATCH2_URL = (
    f"https://github.com/catchorg/Catch2/archive/refs/tags/v{CATCH2_VERSION}.tar.gz"
)
CATCH2_SHA256 = "18b3f70ac80fccc340d8c6ff0f339b2ae64944782f8d2fca2bd705cf47cadb79"


def ensure_build_deps(prefix: Path, rocm_root: Path, force: bool = False) -> None:
    ensure_catch2(prefix.resolve(), rocm_root.resolve(), force=force)


def ensure_catch2(prefix: Path, rocm_root: Path, force: bool = False) -> None:
    config_path = prefix / "lib" / "cmake" / "Catch2" / "Catch2Config.cmake"
    if config_path.exists() and not force:
        log(f"Catch2 already available: {config_path}")
        return

    work_root = prefix.parent / "build-deps-work"
    archive_path = work_root / f"catch2-v{CATCH2_VERSION}.tar.gz"
    source_dir = work_root / f"Catch2-{CATCH2_VERSION}"
    build_dir = work_root / f"Catch2-{CATCH2_VERSION}-build"

    work_root.mkdir(parents=True, exist_ok=True)
    _download(CATCH2_URL, archive_path, CATCH2_SHA256)
    if source_dir.exists():
        remove_tree(source_dir)
    _extract_tarball(archive_path, work_root)
    if build_dir.exists():
        remove_tree(build_dir)

    env = rocm_build_env(rocm_root)
    cmake_args = [
        "cmake",
        "-S",
        source_dir,
        "-B",
        build_dir,
        "-GNinja",
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        f"-DCMAKE_C_COMPILER={rocm_tool(rocm_root, 'clang')}",
        f"-DCMAKE_CXX_COMPILER={rocm_tool(rocm_root, 'clang++')}",
        f"-DCMAKE_AR={rocm_tool(rocm_root, 'llvm-ar')}",
        f"-DCMAKE_RANLIB={rocm_tool(rocm_root, 'llvm-ranlib')}",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
        "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
        "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
        "-DCATCH_INSTALL_DOCS=OFF",
        "-DCATCH_INSTALL_EXTRAS=ON",
        "-DBUILD_TESTING=OFF",
    ]
    run(cmake_args, cwd=REPO_ROOT, env=env)
    run(["cmake", "--build", build_dir, "--target", "install"], cwd=REPO_ROOT, env=env)


def _download(url: str, path: Path, expected_sha256: str) -> None:
    if path.exists() and _sha256_bytes(path.read_bytes()) == expected_sha256:
        log(f"  == Cached {path.name}")
        return
    log(f"  ++ Downloading {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as f:
        shutil.copyfileobj(response, f)
    actual = _sha256_bytes(tmp.read_bytes())
    if actual != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {url}: expected {expected_sha256}, got {actual}"
        )
    tmp.replace(path)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extract_tarball(archive_path: Path, output_dir: Path) -> None:
    output_resolved = output_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            target = output_dir / member.name
            parent = target.parent.resolve()
            if output_resolved != parent and output_resolved not in parent.parents:
                raise RuntimeError(
                    f"Archive path escapes output directory: {member.name}"
                )
        tf.extractall(output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    shared_root = REPO_ROOT.parent / "build" / "rocm-root"
    core_root = REPO_ROOT.parent / "build" / "hrx-core"
    parser.add_argument("--rocm-root", type=Path, default=shared_root)
    parser.add_argument("--prefix", type=Path, default=core_root / "build-deps")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    ensure_build_deps(args.prefix, args.rocm_root, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
