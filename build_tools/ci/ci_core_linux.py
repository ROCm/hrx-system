#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Linux CI driver for configuring, building, testing, and packaging HRX.

The GitHub workflow is intentionally thin: it selects a configuration and this
script owns the build details. That keeps sanitizer, packaging, and future
release variants reusable without growing large stringly shell snippets in YAML.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_tools.ci import ci_core_common as common
from build_tools.ci.ci_core_common import (
    ARTIFACT_SETS,
    CORE_CTEST_EXCLUDE_REGEXES,
    cmake_options_from_env,
    combine_ctest_exclude_regex,
    copy_matching_rocm_paths,
    copy_relative_path,
    default_ctest_parallelism,
    extract_tar_package_archive,
    log,
    path_relative_to,
    prepare_composed_install_root,
    remove_tree,
    require_path,
    rocm_llvm_bin,
    run,
    sanitizer_options,
)
from build_tools.devtools import ctest as ctest_dev

PACKAGE_NAMES = [
    "hrx-public-linux-x86_64",
    "hrx-public-deps-linux-x86_64",
    "hrx-tests-linux-x86_64",
    "hrx-rocm-buildenv-linux-x86_64",
]

PUBLIC_DEPS_REQUIRED_GLOBS = [
    "bin/rocminfo",
    "lib/libhsa-runtime64.so*",
    "lib/libhsa-amd-aqlprofile64.so*",
]

PUBLIC_DEPS_OPTIONAL_GLOBS = [
    "lib/libhsakmt.*",
    "lib/librocprofiler-register.so*",
    "lib/rocm_sysdeps/lib/*.so*",
]

PLATFORM = "linux"

# Re-export shared helpers used by existing tests and callers of this script.
ROCM_ARTIFACT_VARIANT_LOG_KEY = common.ROCM_ARTIFACT_VARIANT_LOG_KEY
ROCM_ARTIFACT_VARIANTS = common.ROCM_ARTIFACT_VARIANTS
S3Object = common.S3Object
rocm_artifact_variant_from_configure_log = (
    common.rocm_artifact_variant_from_configure_log
)
s3_cache_path = common.s3_cache_path
validate_rocm_artifact_variant = common.validate_rocm_artifact_variant


def fetch_rocm(args: argparse.Namespace) -> None:
    common.fetch_rocm(args, platform_name=PLATFORM, platform_display="Linux")


def extract_packages(args: argparse.Namespace) -> None:
    common.extract_packages(
        args,
        package_roots={
            "hrx-public-linux-x86_64": args.public_install_dir.resolve(),
            "hrx-public-deps-linux-x86_64": args.public_deps_dir.resolve(),
            "hrx-tests-linux-x86_64": args.tests_install_dir.resolve(),
        },
        extension=".tar.zst",
        extract_archive=extract_tar_package_archive,
    )


def write_package_manifest(
    manifest_path: Path,
    *,
    package_name: str,
    source_root: Path,
    tarball_path: Path,
    rocm_root: Path | None = None,
) -> None:
    common.write_package_manifest(
        manifest_path,
        package_name=package_name,
        source_root=source_root,
        archive_path=tarball_path,
        rocm_root=rocm_root,
        archive_name_key="tarball",
        archive_sha256_key="tarball_sha256",
    )


def rocm_tool(rocm_root: Path, name: str) -> Path:
    tool_path = rocm_llvm_bin(rocm_root) / name
    if not tool_path.exists():
        raise FileNotFoundError(f"Missing ROCm LLVM tool {name}: {tool_path}")
    return tool_path


def rocm_build_env(
    rocm_root: Path, base_env: dict[str, str] | None = None
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    llvm_bin = rocm_llvm_bin(rocm_root)
    env["PATH"] = f"{llvm_bin}:{rocm_root / 'bin'}:{env.get('PATH', '')}"
    env["CMAKE_PREFIX_PATH"] = f"{rocm_root}:{env.get('CMAKE_PREFIX_PATH', '')}"
    return env


def install_runtime_env(
    root: Path, base_env: dict[str, str] | None = None
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    lib_paths = [
        root / "lib",
        root / "lib" / "rocm_sysdeps" / "lib",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(path) for path in lib_paths] + [env.get("LD_LIBRARY_PATH", "")]
    )
    env["PATH"] = f"{root / 'bin'}:{env.get('PATH', '')}"
    env["CMAKE_PREFIX_PATH"] = f"{root}:{env.get('CMAKE_PREFIX_PATH', '')}"
    return env


def sanitizer_debug_options(sanitizer: str, *, assertions: bool) -> list[str]:
    if sanitizer == "none":
        return []
    flags = [
        "-O1",
        "-g",
        "-fno-omit-frame-pointer",
        "-fno-optimize-sibling-calls",
    ]
    if not assertions:
        flags.append("-DNDEBUG")
    joined_flags = " ".join(flags)
    return [
        f"CMAKE_C_FLAGS_RELWITHDEBINFO={joined_flags}",
        f"CMAKE_CXX_FLAGS_RELWITHDEBINFO={joined_flags}",
    ]


def sanitizer_flag(sanitizer: str) -> str | None:
    sanitizer_flags = {
        "asan": "-fsanitize=address",
        "tsan": "-fsanitize=thread",
        "msan": "-fsanitize=memory",
        "ubsan": "-fsanitize=undefined",
    }
    return sanitizer_flags.get(sanitizer)


def sanitizer_configure_options(sanitizer: str) -> list[str]:
    if sanitizer != "msan":
        return []
    # Google Benchmark uses try_run checks to select its regex backend. Those
    # checks run against uninstrumented system libraries under MSAN and can fail
    # even though the backend builds. Preseed the backend choice so sanitizer
    # CI still compiles benchmark binaries.
    return [
        "HAVE_STD_REGEX=OFF",
        "HAVE_GNU_POSIX_REGEX=OFF",
        "HAVE_POSIX_REGEX=ON",
    ]


def amdgpu_device_binary_source_options(rocm_root: Path) -> list[str]:
    return [
        "IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE=source",
        "IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm",
        f"IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH={rocm_root.as_posix()}",
    ]


def add_sanitizer_runtime_env(
    env: dict[str, str], *, sanitizer: str, rocm_root: Path
) -> dict[str, str]:
    if sanitizer == "none":
        return env
    env = dict(env)
    symbolizer_path = rocm_root / "lib" / "llvm" / "bin" / "llvm-symbolizer"
    if symbolizer_path.exists():
        for key in [
            "ASAN_SYMBOLIZER_PATH",
            "LSAN_SYMBOLIZER_PATH",
            "MSAN_SYMBOLIZER_PATH",
            "TSAN_SYMBOLIZER_PATH",
            "UBSAN_SYMBOLIZER_PATH",
        ]:
            env.setdefault(key, os.fspath(symbolizer_path))
    env.setdefault("ASAN_OPTIONS", "symbolize=1")
    env.setdefault("LSAN_OPTIONS", "symbolize=1")
    env.setdefault("MSAN_OPTIONS", "symbolize=1")
    env.setdefault("TSAN_OPTIONS", "symbolize=1")
    env.setdefault("UBSAN_OPTIONS", "print_stacktrace=1")
    return env


def build_core(args: argparse.Namespace) -> None:
    rocm_root = args.rocm_root.resolve()
    build_dir = args.build_dir.resolve()
    public_install_dir = args.public_install_dir.resolve()
    tests_install_dir = args.tests_install_dir.resolve()
    require_path(rocm_root, "ROCm build root")

    c_compiler = rocm_tool(rocm_root, "clang")
    cxx_compiler = rocm_tool(rocm_root, "clang++")
    ar = rocm_tool(rocm_root, "llvm-ar")
    ranlib = rocm_tool(rocm_root, "llvm-ranlib")

    cmake_prefix_path = ";".join([str(rocm_root)])
    ctest_enabled = True
    cmake_defines = [
        f"CMAKE_PREFIX_PATH={cmake_prefix_path}",
        f"CMAKE_INSTALL_PREFIX={public_install_dir}",
        "CMAKE_INSTALL_LIBDIR=lib",
        f"CMAKE_C_COMPILER={c_compiler}",
        f"CMAKE_CXX_COMPILER={cxx_compiler}",
        f"CMAKE_ASM_COMPILER={c_compiler}",
        f"CMAKE_AR={ar}",
        f"CMAKE_RANLIB={ranlib}",
        "CMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
        "CMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
        "CMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
        f"CMAKE_BUILD_TYPE={args.build_type}",
        f"IREE_BUILD_TESTS={'ON' if ctest_enabled else 'OFF'}",
        "IREE_BUILD_BENCHMARKS=ON",
        "LOOM_BUILD=ON",
        f"LIBHRX_BUILD_CTS={'ON' if ctest_enabled else 'OFF'}",
        f"HRX_INSTALL_TESTS={'ON' if ctest_enabled else 'OFF'}",
        f"LIBHRX_BUILD_PASSTHROUGH={'ON' if args.passthrough else 'OFF'}",
        f"IREE_HAL_DRIVER_AMDGPU={'ON' if args.amdgpu else 'OFF'}",
        f"IREE_ROCM_PATH={rocm_root}",
        "IREE_ENABLE_LIBBACKTRACE=OFF",
        f"IREE_ENABLE_ASSERTIONS={'ON' if args.assertions else 'OFF'}",
    ]
    cmake_defines.extend(sanitizer_options(args.sanitizer))
    cmake_defines.extend(
        sanitizer_debug_options(args.sanitizer, assertions=args.assertions)
    )
    cmake_defines.extend(sanitizer_configure_options(args.sanitizer))
    if args.amdgpu:
        cmake_defines.extend(amdgpu_device_binary_source_options(rocm_root))
    cmake_defines.extend(cmake_options_from_env())
    cmake_defines.extend(args.cmake_option)

    env = rocm_build_env(rocm_root)
    cmake_configure_cmd = [
        "cmake",
        "-S",
        REPO_ROOT,
        "-B",
        build_dir,
        "-GNinja",
        *[f"-D{option.removeprefix('-D')}" for option in cmake_defines],
    ]
    run(
        cmake_configure_cmd,
        cwd=REPO_ROOT,
        env=env,
        pretty_command=True,
    )
    if args.sanitizer != "none":
        log(
            "Sanitizer build tree configured; selected source tests own the "
            "build closure."
        )
        return

    run(
        ["cmake", "--build", build_dir, "--target", args.target], cwd=REPO_ROOT, env=env
    )

    for install_dir, component in [
        (public_install_dir, args.public_component),
        (tests_install_dir, args.tests_component),
    ]:
        if install_dir.exists():
            remove_tree(install_dir)
        run(
            [
                "cmake",
                "--install",
                build_dir,
                "--prefix",
                install_dir,
                "--component",
                component,
            ],
            cwd=REPO_ROOT,
            env=env,
        )


def ctest_arguments(args: argparse.Namespace) -> list[str]:
    ctest_parallelism = args.ctest_parallelism or default_ctest_parallelism()
    arguments = ["--parallel", str(ctest_parallelism)]
    if args.ctest_regex:
        arguments.extend(["-R", args.ctest_regex])
    ctest_exclude_regex = combine_ctest_exclude_regex(
        *CORE_CTEST_EXCLUDE_REGEXES, args.ctest_exclude_regex
    )
    if ctest_exclude_regex:
        arguments.extend(["-E", ctest_exclude_regex])
    if args.ctest_label_regex:
        arguments.extend(["-L", args.ctest_label_regex])
    if args.ctest_label_exclude_regex:
        arguments.extend(["-LE", args.ctest_label_exclude_regex])
    return arguments


def apply_test_environment(
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, str]:
    env = dict(env)
    if args.cts_device:
        env["HRX_CTS_DEVICE"] = args.cts_device
    if args.test_tmpdir is not None:
        env["HRX_TEST_TMPDIR"] = os.fspath(args.test_tmpdir.resolve())
    return env


def test_source_build(args: argparse.Namespace) -> None:
    rocm_root = args.rocm_root.resolve()
    build_dir = args.build_dir.resolve()
    require_path(rocm_root, "ROCm build root")
    require_path(build_dir / "CTestTestfile.cmake", "source CTest file")

    env = add_sanitizer_runtime_env(
        rocm_build_env(rocm_root),
        sanitizer=args.sanitizer,
        rocm_root=rocm_root,
    )
    env = apply_test_environment(args, env)
    test_step = ctest_dev.CTestBuildAndRunStep(
        cmake="cmake",
        ctest="ctest",
        build_dir=build_dir,
        arguments=[*ctest_arguments(args), "--no-tests=error"],
        cwd=REPO_ROOT,
        env=env,
    )
    result = test_step.run(verbose=True)
    if result != 0:
        raise subprocess.CalledProcessError(result, "selected CTest build and run")


def test_core(args: argparse.Namespace) -> None:
    if args.sanitizer != "none":
        test_source_build(args)
        return

    rocm_root = args.rocm_root.resolve()
    smoke_build_dir = args.package_smoke_build_dir.resolve()
    if args.prepare_public_deps:
        prepare_public_deps_root(args)
    install_root = prepare_composed_install_root(args)
    installed_tests_dir = install_root / "share" / "hrx-system" / "tests"

    require_path(installed_tests_dir / "CTestTestfile.cmake", "installed CTest file")
    require_path(install_root / "lib" / "libhrx.so", "installed libhrx.so")
    require_path(install_root / "lib" / "libloomc.so", "installed libloomc.so")
    require_path(install_root / "bin" / "hrx-info", "installed hrx-info")

    if rocm_root.exists():
        base_env = rocm_build_env(rocm_root)
    elif args.package_smoke:
        require_path(rocm_root, "ROCm build root")
        base_env = rocm_build_env(rocm_root)
    else:
        base_env = dict(os.environ)

    env = apply_test_environment(args, install_runtime_env(install_root, base_env))
    ctest_cmd = [
        "ctest",
        "--test-dir",
        installed_tests_dir,
        "--output-on-failure",
        *ctest_arguments(args),
    ]
    run(ctest_cmd, cwd=REPO_ROOT, env=env, stderr_to_stdout=True)
    run([install_root / "bin" / "hrx-info"], cwd=REPO_ROOT, env=env)
    run([install_root / "bin" / "hrx-info", "--device=cpu:0"], cwd=REPO_ROOT, env=env)
    if args.gpu:
        run(
            [install_root / "bin" / "hrx-info", "--device=gpu:0"],
            cwd=REPO_ROOT,
            env=env,
        )

    if not args.package_smoke:
        return

    sanitizer_link_flag = sanitizer_flag(args.sanitizer)
    smoke_link_flags = "-fuse-ld=lld"
    smoke_sanitizer_options = []
    if sanitizer_link_flag:
        smoke_link_flags = f"{smoke_link_flags} {sanitizer_link_flag}"
        smoke_sanitizer_options = [
            f"-DCMAKE_C_FLAGS={sanitizer_link_flag}",
            f"-DCMAKE_CXX_FLAGS={sanitizer_link_flag}",
        ]
    hrx_smoke_build_dir = smoke_build_dir / "hrx"
    loomc_smoke_build_dir = smoke_build_dir / "loomc"
    smoke_cmake_options = [
        "-GNinja",
        f"-DCMAKE_PREFIX_PATH={install_root};{rocm_root}",
        f"-DCMAKE_C_COMPILER={rocm_tool(rocm_root, 'clang')}",
        f"-DCMAKE_CXX_COMPILER={rocm_tool(rocm_root, 'clang++')}",
        f"-DCMAKE_AR={rocm_tool(rocm_root, 'llvm-ar')}",
        f"-DCMAKE_RANLIB={rocm_tool(rocm_root, 'llvm-ranlib')}",
        *smoke_sanitizer_options,
        f"-DCMAKE_EXE_LINKER_FLAGS={smoke_link_flags}",
        f"-DCMAKE_SHARED_LINKER_FLAGS={smoke_link_flags}",
        f"-DCMAKE_MODULE_LINKER_FLAGS={smoke_link_flags}",
    ]
    run(
        [
            "cmake",
            "-S",
            REPO_ROOT / "libhrx" / "cts" / "package_smoke",
            "-B",
            hrx_smoke_build_dir,
            *smoke_cmake_options,
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    run(["cmake", "--build", hrx_smoke_build_dir], cwd=REPO_ROOT, env=env)
    run([hrx_smoke_build_dir / "hrx_package_smoke"], cwd=REPO_ROOT, env=env)
    run([hrx_smoke_build_dir / "hrx_package_smoke_cxx"], cwd=REPO_ROOT, env=env)
    run(
        [
            "cmake",
            "-S",
            REPO_ROOT / "loom" / "binding" / "c" / "packaging" / "package_smoke",
            "-B",
            loomc_smoke_build_dir,
            *smoke_cmake_options,
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    run(["cmake", "--build", loomc_smoke_build_dir], cwd=REPO_ROOT, env=env)
    run([loomc_smoke_build_dir / "loomc_package_smoke"], cwd=REPO_ROOT, env=env)


ROCM_BUILDENV_EXCLUDE_PATHS = {
    "bin/hrx-info",
    "env.sh",
    "hrx-public-linux-x86_64-env.sh",
    "hrx-public-deps-linux-x86_64-env.sh",
    "hrx-tests-linux-x86_64-env.sh",
    "hrx-rocm-buildenv-linux-x86_64-env.sh",
    "include/hrx",
    "include/loomc",
    "include/passthrough",
    "lib/cmake/hrx",
    "lib/cmake/loomc",
    "lib/libhrx.so",
    "lib/libloomc.so",
    "share/hrx-cts",
    "share/hrx-system",
}

ROCM_BUILDENV_EXCLUDE_PREFIXES = (
    "include/hrx/",
    "include/loomc/",
    "include/passthrough/",
    "lib/cmake/hrx/",
    "lib/cmake/loomc/",
    "lib/libhrx.so.",
    "lib/libloomc.so.",
    "share/hrx-cts/",
    "share/hrx-system/",
)


def tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def make_env_script_content() -> str:
    return """#!/usr/bin/env bash
# Source this file to use this HRX installation overlay.
_hrx_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HRX_HOME="${_hrx_env_dir}"
export ROCM_HOME="${_hrx_env_dir}"
export PATH="${_hrx_env_dir}/bin:${PATH}"
export LD_LIBRARY_PATH="${_hrx_env_dir}/lib:${_hrx_env_dir}/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${_hrx_env_dir}:${CMAKE_PREFIX_PATH:-}"
unset _hrx_env_dir
"""


def should_exclude_tar_path(
    arcname: str,
    *,
    exclude_paths: set[str],
    exclude_prefixes: tuple[str, ...],
) -> bool:
    return arcname in exclude_paths or any(
        arcname.startswith(prefix) for prefix in exclude_prefixes
    )


def create_tar_zst(
    source_dir: Path,
    tarball_path: Path,
    *,
    exclude_paths: set[str] | None = None,
    exclude_prefixes: tuple[str, ...] = (),
) -> None:
    try:
        import zstandard
    except ModuleNotFoundError as e:
        raise RuntimeError("Install the zstandard Python package") from e
    tarball_path.parent.mkdir(parents=True, exist_ok=True)
    exclude_paths = exclude_paths or set()

    def filter_member(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if should_exclude_tar_path(
            info.name, exclude_paths=exclude_paths, exclude_prefixes=exclude_prefixes
        ):
            return None
        return tar_filter(info)

    cctx = zstandard.ZstdCompressor(level=3, threads=-1)
    with tarball_path.open("wb") as f:
        with cctx.stream_writer(f, closefd=False) as zstd_stream:
            with tarfile.open(fileobj=zstd_stream, mode="w|") as tf:
                for child in sorted(source_dir.iterdir(), key=lambda p: p.name):
                    tf.add(
                        child, arcname=child.name, recursive=True, filter=filter_member
                    )


def write_package_env_script(output_dir: Path, script_name: str) -> Path:
    script_path = output_dir / script_name
    script_path.write_text(make_env_script_content())
    script_path.chmod(0o755)
    return script_path


def ldd_rocm_dependencies(rocm_root: Path, binary: Path) -> list[Path]:
    if not binary.exists():
        return []
    result = subprocess.run(
        ["ldd", binary],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    deps: list[Path] = []
    for line in result.stdout.splitlines():
        match = re.search(r"=>\s+(\S+)\s+\(", line)
        if not match:
            match = re.match(r"\s*(/\S+)\s+\(", line)
        if not match:
            continue
        dep_path = Path(match.group(1))
        if path_relative_to(dep_path, rocm_root) is not None:
            deps.append(dep_path)
    return deps


def prepare_public_deps_root(args: argparse.Namespace) -> None:
    rocm_root = args.rocm_root.resolve()
    deps_root = args.public_deps_dir.resolve()
    require_path(rocm_root, "ROCm build root")

    if deps_root.exists():
        remove_tree(deps_root)
    deps_root.mkdir(parents=True)

    missing_required = []
    for pattern in PUBLIC_DEPS_REQUIRED_GLOBS:
        if not list(rocm_root.glob(pattern)):
            missing_required.append(pattern)
    if missing_required:
        raise RuntimeError(
            "Missing required public dependency files in ROCm root:\n  "
            + "\n  ".join(missing_required)
        )

    copied = copy_matching_rocm_paths(
        rocm_root, deps_root, PUBLIC_DEPS_REQUIRED_GLOBS + PUBLIC_DEPS_OPTIONAL_GLOBS
    )
    for seed in [
        rocm_root / "bin" / "rocminfo",
        rocm_root / "lib" / "libhsa-runtime64.so",
        rocm_root / "lib" / "libhsa-amd-aqlprofile64.so",
    ]:
        for dep_path in ldd_rocm_dependencies(rocm_root, seed):
            relpath = dep_path.resolve().relative_to(rocm_root.resolve())
            copy_relative_path(rocm_root, deps_root, relpath)
            copied.append(relpath)

    manifest = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "package": "hrx-public-deps-linux-x86_64",
        "platform": platform.platform(),
        "rocm_root": str(rocm_root),
        "copied_paths": sorted({os.fspath(path) for path in copied}),
    }
    rocm_manifest = rocm_root / ".hrx-rocm-artifacts.json"
    if rocm_manifest.exists():
        manifest["rocm_artifacts"] = json.loads(rocm_manifest.read_text())
    (deps_root / ".hrx-package.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    log(f"Public dependency root ready: {deps_root}")


def package_tree(
    *,
    package_name: str,
    source_root: Path,
    output_dir: Path,
    suffix: str,
    rocm_root: Path | None = None,
    env_script_name: str | None = None,
    exclude_paths: set[str] | None = None,
    exclude_prefixes: tuple[str, ...] = (),
) -> None:
    require_path(source_root, f"{package_name} source root")
    output_dir.mkdir(parents=True, exist_ok=True)
    package_exclude_paths = set(exclude_paths or set())
    package_exclude_paths.add("env.sh")
    if env_script_name:
        package_exclude_paths.add(env_script_name)

    package_stem = f"{package_name}-{suffix}" if suffix else package_name
    tarball_path = output_dir / f"{package_stem}.tar.zst"
    if tarball_path.exists():
        tarball_path.unlink()
    create_tar_zst(
        source_root,
        tarball_path,
        exclude_paths=package_exclude_paths,
        exclude_prefixes=exclude_prefixes,
    )
    env_script_path = None
    if env_script_name:
        env_script_path = write_package_env_script(output_dir, env_script_name)
    manifest_path = output_dir / f"{package_stem}.manifest.json"
    write_package_manifest(
        manifest_path,
        package_name=package_name,
        source_root=source_root,
        tarball_path=tarball_path,
        rocm_root=rocm_root,
    )
    log(f"Created {tarball_path}")
    if env_script_path:
        log(f"Created {env_script_path}")
    log(f"Created {manifest_path}")


def package_core(args: argparse.Namespace) -> None:
    rocm_root = args.rocm_root.resolve()
    output_dir = args.package_output_dir.resolve()
    suffix = args.package_suffix or dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")

    require_path(
        args.public_install_dir.resolve() / "lib" / "libhrx.so", "public libhrx.so"
    )
    require_path(
        args.public_install_dir.resolve() / "lib" / "libloomc.so",
        "public libloomc.so",
    )
    require_path(
        args.public_install_dir.resolve() / "bin" / "hrx-info", "public hrx-info"
    )
    require_path(
        args.public_install_dir.resolve()
        / "lib"
        / "cmake"
        / "hrx"
        / "hrx-config.cmake",
        "public hrx CMake package",
    )
    require_path(
        args.public_install_dir.resolve()
        / "lib"
        / "cmake"
        / "loomc"
        / "loomc-config.cmake",
        "public loomc CMake package",
    )
    require_path(
        args.tests_install_dir.resolve()
        / "share"
        / "hrx-system"
        / "tests"
        / "CTestTestfile.cmake",
        "installed CTest file",
    )
    prepare_public_deps_root(args)

    package_tree(
        package_name="hrx-public-linux-x86_64",
        source_root=args.public_install_dir.resolve(),
        output_dir=output_dir,
        suffix=suffix,
        rocm_root=rocm_root,
        env_script_name="hrx-public-linux-x86_64-env.sh",
    )
    package_tree(
        package_name="hrx-public-deps-linux-x86_64",
        source_root=args.public_deps_dir.resolve(),
        output_dir=output_dir,
        suffix=suffix,
        rocm_root=rocm_root,
        env_script_name="hrx-public-deps-linux-x86_64-env.sh",
    )
    package_tree(
        package_name="hrx-tests-linux-x86_64",
        source_root=args.tests_install_dir.resolve(),
        output_dir=output_dir,
        suffix=suffix,
        rocm_root=rocm_root,
    )
    package_tree(
        package_name="hrx-rocm-buildenv-linux-x86_64",
        source_root=rocm_root,
        output_dir=output_dir,
        suffix=suffix,
        rocm_root=rocm_root,
        env_script_name="hrx-rocm-buildenv-linux-x86_64-env.sh",
        exclude_paths=ROCM_BUILDENV_EXCLUDE_PATHS,
        exclude_prefixes=ROCM_BUILDENV_EXCLUDE_PREFIXES,
    )


def run_all(args: argparse.Namespace) -> None:
    common.run_all(
        args,
        fetch_rocm_fn=fetch_rocm,
        build_core_fn=build_core,
        test_core_fn=test_core,
        package_core_fn=package_core,
    )


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    common.add_shared_args(
        parser,
        default_output=REPO_ROOT / "build" / "linux",
        sanitizer_choices=["none", "asan", "tsan", "msan", "ubsan"],
        passthrough_default=True,
        include_gpu=True,
        include_amdgpu=True,
        artifact_sets=ARTIFACT_SETS,
    )


def main(argv: list[str] | None = None) -> int:
    return common.main(
        argv,
        description=__doc__,
        add_shared_args_fn=add_shared_args,
        run_all_fn=run_all,
        fetch_rocm_fn=fetch_rocm,
        build_core_fn=build_core,
        test_core_fn=test_core,
        extract_packages_fn=extract_packages,
        package_core_fn=package_core,
    )


if __name__ == "__main__":
    raise SystemExit(main())
