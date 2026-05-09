#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate an HRX core build and install tree."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from hrx_build_tools import REPO_ROOT, require_file, run


def _env(rocm_root: Path, install_prefix: Path) -> dict[str, str]:
    env = os.environ.copy()
    lib_paths = [install_prefix / "lib", rocm_root / "lib"]
    env["LD_LIBRARY_PATH"] = (
        ":".join(str(p) for p in lib_paths) + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    env["PATH"] = f"{install_prefix / 'bin'}:{rocm_root / 'bin'}:{env.get('PATH', '')}"
    env["CMAKE_PREFIX_PATH"] = (
        f"{install_prefix}:{rocm_root}:{env.get('CMAKE_PREFIX_PATH', '')}"
    )
    return env


def test(args: argparse.Namespace) -> None:
    rocm_root = args.rocm_root.resolve()
    build_dir = args.build_dir.resolve()
    install_prefix = args.install_prefix.resolve()
    env = _env(rocm_root, install_prefix)

    require_file(rocm_root, "ROCm build root")
    require_file(build_dir / "CTestTestfile.cmake", "build-tree CTest file")
    require_file(install_prefix / "lib" / "libhrx.so", "installed libhrx.so")
    require_file(install_prefix / "bin" / "hrx-info", "installed hrx-info")

    run(
        ["ctest", "--test-dir", build_dir, "--output-on-failure"],
        cwd=REPO_ROOT,
        env=env,
    )
    run([install_prefix / "bin" / "hrx-info", "--device=cpu:0"], cwd=REPO_ROOT, env=env)
    if args.gpu:
        run(
            [install_prefix / "bin" / "hrx-info", "--device=gpu:0"],
            cwd=REPO_ROOT,
            env=env,
        )

    cts_dir = install_prefix / "share" / "hrx-cts"
    if cts_dir.exists():
        env["HRX_LIBRARY"] = str(install_prefix / "lib" / "libhrx.so")
        run(
            ["ctest", "--test-dir", cts_dir, "--output-on-failure"],
            cwd=REPO_ROOT,
            env=env,
        )

    smoke_build = args.package_smoke_build_dir.resolve()
    run(
        [
            "cmake",
            "-S",
            REPO_ROOT / "cts" / "package_smoke",
            "-B",
            smoke_build,
            "-GNinja",
            f"-DCMAKE_PREFIX_PATH={install_prefix};{rocm_root}",
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
            "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
            "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
            "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    run(["cmake", "--build", smoke_build], cwd=REPO_ROOT, env=env)
    run([smoke_build / "hrx_package_smoke"], cwd=REPO_ROOT, env=env)
    run([smoke_build / "hrx_package_smoke_cxx"], cwd=REPO_ROOT, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    shared_root = REPO_ROOT.parent / "build" / "rocm-root"
    core_root = REPO_ROOT.parent / "build" / "hrx-core"
    parser.add_argument("--rocm-root", type=Path, default=shared_root)
    parser.add_argument("--build-dir", type=Path, default=core_root / "build")
    parser.add_argument("--install-prefix", type=Path, default=shared_root)
    parser.add_argument(
        "--package-smoke-build-dir",
        type=Path,
        default=core_root / "package-smoke",
    )
    parser.add_argument("--gpu", action="store_true", help="Run GPU smoke tests")
    args = parser.parse_args(argv)
    test(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
