#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0

"""Configure, build, and install the HRX core scope."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from hrx_build_tools import REPO_ROOT, require_file, run


def build(args: argparse.Namespace) -> None:
    rocm_root = args.rocm_root.resolve()
    build_dir = args.build_dir.resolve()
    install_prefix = args.install_prefix.resolve()

    require_file(rocm_root, "ROCm build root")

    cmake_args = [
        "cmake",
        "-S",
        REPO_ROOT,
        "-B",
        build_dir,
        "-GNinja",
        f"-DCMAKE_PREFIX_PATH={rocm_root}",
        f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        "-DCMAKE_C_COMPILER=clang",
        "-DCMAKE_CXX_COMPILER=clang++",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
        "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
        "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
        f"-DCMAKE_BUILD_TYPE={args.build_type}",
        f"-DHRX_BUILD_CTS={'ON' if args.cts else 'OFF'}",
        f"-DHRX_BUILD_PASSTHROUGH={'ON' if args.passthrough else 'OFF'}",
        f"-DHRX_ENABLE_IREE_AMDGPU={'ON' if args.amdgpu else 'OFF'}",
        # IREE's libbacktrace wrapper generates headers in the IREE source tree
        # during configure. Keep this bolt-on build from dirtying submodules.
        "-DIREE_ENABLE_LIBBACKTRACE=OFF",
    ]
    if args.iree_source_dir:
        cmake_args.append(f"-DHRX_IREE_SOURCE_DIR={args.iree_source_dir.resolve()}")
    cmake_args.extend(f"-D{option}" for option in args.cmake_option)

    env = os.environ.copy()
    env.setdefault("CMAKE_PREFIX_PATH", str(rocm_root))
    run(cmake_args, cwd=REPO_ROOT, env=env)
    run(
        ["cmake", "--build", build_dir, "--target", args.target], cwd=REPO_ROOT, env=env
    )
    run(["cmake", "--install", build_dir], cwd=REPO_ROOT, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    shared_root = REPO_ROOT.parent / "build" / "rocm-root"
    core_root = REPO_ROOT.parent / "build" / "hrx-core"
    parser.add_argument("--rocm-root", type=Path, default=shared_root)
    parser.add_argument("--build-dir", type=Path, default=core_root / "build")
    parser.add_argument("--install-prefix", type=Path, default=shared_root)
    parser.add_argument("--build-type", default="RelWithDebInfo")
    parser.add_argument("--target", default="all")
    parser.add_argument("--iree-source-dir", type=Path)
    parser.add_argument("--cts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--passthrough", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--amdgpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "-D",
        dest="cmake_option",
        action="append",
        default=[],
        help="Additional CMake definition, for example -DHRX_ENABLE_TRACY=ON",
    )
    args = parser.parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
