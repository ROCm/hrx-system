#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Windows CI driver for configuring, building, testing, and packaging HRX."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_tools.ci import ci_core_common as common
from build_tools.ci.ci_core_common import (
    CORE_CTEST_EXCLUDE_REGEXES,
    checked_dest,
    cmake_options_from_env,
    combine_ctest_exclude_regex,
    default_ctest_parallelism,
    log,
    remove_tree,
    require_path,
    rocm_llvm_bin,
    sanitizer_options,
    write_package_manifest,
)

PACKAGE_NAMES = [
    "hrx-public-windows-x86_64",
    "hrx-public-deps-windows-x86_64",
    "hrx-tests-windows-x86_64",
    "hrx-rocm-buildenv-windows-x86_64",
]

# Windows core package CI uses ROCm LLVM/toolchain artifacts but does not build
# or ship AMDGPU runtime payloads.
PUBLIC_DEPS_REQUIRED_GLOBS = []
PUBLIC_DEPS_OPTIONAL_GLOBS = []

PLATFORM = "windows"
WINDOWS_ROCM_DEPENDENCY_MODE = "auto"
WINDOWS_TOOLCHAINS = ("msvc", "clang-cl")
_MSVC_BUILD_ENV_CACHE: dict[str, str] | None = None


@dataclass(frozen=True)
class WindowsToolchain:
    name: str
    c_compiler: Path
    cxx_compiler: Path
    asm_compiler: Path
    ar: Path
    linker: Path
    rc: Path
    mt: Path


# Windows package CI fetches the compiler/build environment by default. Runtime
# ROCm payloads are intentionally not part of the public-deps package.
ARTIFACT_SETS = {
    "core": {
        "sysdeps": ["lib", "dev"],
        "base": ["lib", "run", "dev"],
        "amd-llvm": ["lib", "run"],
    },
    "core-with-llvm-dev": {
        "sysdeps": ["lib", "dev"],
        "base": ["lib", "run", "dev"],
        "amd-llvm": ["lib", "run", "dev"],
    },
    "core-with-upstream-hip": {
        "sysdeps": ["lib", "dev"],
        "base": ["lib", "run", "dev"],
        "amd-llvm": ["lib", "run", "dev"],
        "core-hip": ["lib", "run", "dev"],
        "core-hipinfo": ["run"],
        "core-kpack": ["lib", "dev"],
    },
}

# Re-export shared helpers used by existing tests and callers of this script.
ROCM_ARTIFACT_VARIANT_LOG_KEY = common.ROCM_ARTIFACT_VARIANT_LOG_KEY
ROCM_ARTIFACT_VARIANTS = common.ROCM_ARTIFACT_VARIANTS
S3Object = common.S3Object
rocm_artifact_variant_from_configure_log = (
    common.rocm_artifact_variant_from_configure_log
)
s3_cache_path = common.s3_cache_path
validate_rocm_artifact_variant = common.validate_rocm_artifact_variant


def run(
    args: Iterable[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stderr_to_stdout: bool = False,
    pretty_command: bool = False,
) -> None:
    common.run(
        args,
        cwd=cwd,
        env=env,
        stderr_to_stdout=stderr_to_stdout,
        pretty_command=pretty_command,
        line_continuation="`",
    )


def fetch_rocm(args: argparse.Namespace) -> None:
    common.fetch_rocm(
        args,
        platform_name=PLATFORM,
        platform_display="Windows",
        artifact_sets=ARTIFACT_SETS,
        materialize_links=True,
        preserve_symlinks=False,
    )


def extract_packages(args: argparse.Namespace) -> None:
    common.extract_packages(
        args,
        package_roots={
            "hrx-public-windows-x86_64": args.public_install_dir.resolve(),
            "hrx-public-deps-windows-x86_64": args.public_deps_dir.resolve(),
            "hrx-tests-windows-x86_64": args.tests_install_dir.resolve(),
        },
        extension=".zip",
        extract_archive=extract_zip_archive,
    )


def copy_matching_rocm_paths(
    rocm_root: Path, dst_root: Path, patterns: list[str]
) -> list[Path]:
    return common.copy_matching_rocm_paths(
        rocm_root, dst_root, patterns, preserve_symlinks=False
    )


def prepare_composed_install_root(args: argparse.Namespace) -> Path:
    return common.prepare_composed_install_root(args, preserve_symlinks=False)


def extract_zip_archive(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            dest_path = checked_dest(output_dir, info.filename)
            if info.is_dir():
                dest_path.mkdir(parents=True, exist_ok=True)
                continue
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if dest_path.exists() or dest_path.is_symlink():
                remove_tree(dest_path)
            with zf.open(info) as source, dest_path.open("wb") as out:
                shutil.copyfileobj(source, out)


def find_path(candidates: Iterable[Path | str], description: str) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing {description}; checked: "
        + ", ".join(os.fspath(candidate) for candidate in candidates)
    )


def env_get(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is not None:
        return value
    lower_name = name.lower()
    for key, value in env.items():
        if key.lower() == lower_name:
            return value
    return None


def find_tool_in_path(
    names: Iterable[str], env: Mapping[str, str] | None = None
) -> Path | None:
    path = env_get(env, "PATH") if env is not None else None
    for name in names:
        found = shutil.which(name, path=path)
        if found:
            return Path(found)
    return None


def require_tool_in_path(
    names: Iterable[str], description: str, env: Mapping[str, str] | None = None
) -> Path:
    if found := find_tool_in_path(names, env):
        return found
    raise FileNotFoundError(f"Missing {description} on PATH")


def windows_sdk_arch(env: Mapping[str, str]) -> str:
    arch = (env_get(env, "PROCESSOR_ARCHITECTURE") or platform.machine()).lower()
    return "arm64" if "arm64" in arch else "x64"


def windows_sdk_version_key(path: Path) -> tuple[int, ...]:
    return tuple(int(part) if part.isdigit() else -1 for part in path.name.split("."))


def windows_sdk_tool_candidates(
    name: str, env: Mapping[str, str] | None = None
) -> list[Path]:
    env = env or os.environ
    arch = windows_sdk_arch(env)
    candidates: list[Path] = []
    if found := find_tool_in_path([name], env):
        candidates.append(found)

    sdk_roots: list[Path] = []
    if windows_sdk_dir := env_get(env, "WindowsSdkDir"):
        sdk_roots.append(Path(windows_sdk_dir))
    for env_name in ("ProgramFiles(x86)", "ProgramFiles"):
        if program_files := env_get(env, env_name):
            sdk_roots.append(Path(program_files) / "Windows Kits" / "10")

    seen: set[Path] = set()
    for sdk_root in sdk_roots:
        sdk_root = sdk_root.resolve()
        if sdk_root in seen:
            continue
        seen.add(sdk_root)
        bin_root = sdk_root / "bin"
        if windows_sdk_version := env_get(env, "WindowsSDKVersion"):
            version = windows_sdk_version.rstrip("\\/")
            candidates.append(bin_root / version / arch / name)
        candidates.append(bin_root / arch / name)
        if bin_root.exists():
            version_dirs = sorted(
                [path for path in bin_root.iterdir() if path.is_dir()],
                key=windows_sdk_version_key,
                reverse=True,
            )
            candidates.extend(path / arch / name for path in version_dirs)
    return candidates


def windows_sdk_tool(name: str, env: Mapping[str, str] | None = None) -> Path:
    return find_path(windows_sdk_tool_candidates(name, env), f"Windows SDK tool {name}")


def existing_unique_paths(candidates: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate.exists():
            continue
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        paths.append(candidate)
    return paths


def vcvarsall_candidates(env: Mapping[str, str]) -> list[Path]:
    candidates: list[Path] = []
    if vs_install_dir := env_get(env, "VSINSTALLDIR"):
        candidates.append(
            Path(vs_install_dir) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        )

    vswhere_candidates: list[Path] = []
    if vswhere := find_tool_in_path(["vswhere.exe"], env):
        vswhere_candidates.append(vswhere)
    for env_name in ("ProgramFiles(x86)", "ProgramFiles"):
        if program_files := env_get(env, env_name):
            vswhere_candidates.append(
                Path(program_files)
                / "Microsoft Visual Studio"
                / "Installer"
                / "vswhere.exe"
            )
    for vswhere in vswhere_candidates:
        if not vswhere.exists():
            continue
        try:
            result = subprocess.run(
                [
                    vswhere,
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(env),
            )
            if install_path := result.stdout.strip():
                candidates.append(
                    Path(install_path) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
                )
                break
        except subprocess.CalledProcessError:
            pass
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        if program_files := env_get(env, env_name):
            candidates.extend(
                Path(program_files).glob(
                    "Microsoft Visual Studio/2022/*/VC/Auxiliary/Build/vcvarsall.bat"
                )
            )
    return existing_unique_paths(candidates)


def find_vcvarsall(env: Mapping[str, str]) -> Path:
    return find_path(vcvarsall_candidates(env), "Visual Studio vcvarsall.bat")


def parse_cmd_environment(output: str) -> dict[str, str]:
    parsed_env: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            parsed_env[key] = value
    return parsed_env


def make_vcvarsall_wrapper_content(vcvarsall: Path) -> str:
    return (
        "@echo off\r\n"
        f'call "{vcvarsall.as_posix()}" x64\r\n'
        "if errorlevel 1 exit /b %errorlevel%\r\n"
        "set\r\n"
    )


def run_vcvarsall(vcvarsall: Path, env: Mapping[str, str]) -> dict[str, str]:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", suffix=".cmd", delete=False
    ) as wrapper:
        wrapper.write(make_vcvarsall_wrapper_content(vcvarsall))
        wrapper_path = Path(wrapper.name)
    try:
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", wrapper_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(env),
        )
    finally:
        try:
            wrapper_path.unlink()
        except FileNotFoundError:
            pass
    if result.returncode != 0:
        raise RuntimeError(
            f"{vcvarsall} failed with exit code {result.returncode}:\n"
            f"{result.stdout.strip()}"
        )
    parsed_env = parse_cmd_environment(result.stdout)
    if not env_get(parsed_env, "VSCMD_ARG_TGT_ARCH"):
        raise RuntimeError(f"{vcvarsall} did not produce an MSVC build environment")
    return parsed_env


def msvc_build_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    if os.name != "nt":
        return env
    if (
        env_get(env, "VSCMD_ARG_TGT_ARCH")
        and find_tool_in_path(["rc.exe"], env)
        and find_tool_in_path(["mt.exe"], env)
    ):
        return env

    global _MSVC_BUILD_ENV_CACHE
    if _MSVC_BUILD_ENV_CACHE is None:
        failures: list[str] = []
        for vcvarsall in vcvarsall_candidates(env):
            try:
                candidate_env = run_vcvarsall(vcvarsall, env)
            except RuntimeError as e:
                failures.append(str(e))
                continue
            missing_tools = [
                name
                for name in ("rc.exe", "mt.exe")
                if not find_tool_in_path([name], candidate_env)
            ]
            if missing_tools:
                failures.append(
                    f"{vcvarsall} did not expose required Windows SDK tools: "
                    + ", ".join(missing_tools)
                )
                continue
            _MSVC_BUILD_ENV_CACHE = candidate_env
            break
        if _MSVC_BUILD_ENV_CACHE is None:
            checked = vcvarsall_candidates(env)
            if not checked:
                find_vcvarsall(env)
            raise RuntimeError(
                "Could not initialize an MSVC x64 build environment.\n\n"
                + "\n\n".join(failures)
            )
    env.update(_MSVC_BUILD_ENV_CACHE)
    return env


def rocm_tool(rocm_root: Path, names: str | Iterable[str]) -> Path:
    if isinstance(names, str):
        names = [names]
    else:
        names = list(names)
    llvm_bin = rocm_llvm_bin(rocm_root)
    candidates = [llvm_bin / name for name in names] + [
        rocm_root / "bin" / name for name in names
    ]
    return find_path(candidates, f"ROCm tool {', '.join(names)}")


def windows_executable_name(name: str) -> str:
    return name if name.lower().endswith(".exe") else f"{name}.exe"


def cmake_path(path: str | os.PathLike[str]) -> str:
    return os.fspath(path).replace("\\", "/")


def cmake_path_list(paths: Iterable[str | os.PathLike[str]]) -> str:
    return ";".join(cmake_path(path) for path in paths)


def rocm_runtime_env(
    rocm_root: Path, base_env: dict[str, str] | None = None
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    paths = [
        rocm_llvm_bin(rocm_root),
        rocm_root / "bin",
    ]
    env["PATH"] = os.pathsep.join(
        [os.fspath(path) for path in paths] + [env.get("PATH", "")]
    )
    env["CMAKE_PREFIX_PATH"] = os.pathsep.join(
        [os.fspath(rocm_root), env.get("CMAKE_PREFIX_PATH", "")]
    )
    return env


def rocm_build_env(
    rocm_root: Path, base_env: dict[str, str] | None = None
) -> dict[str, str]:
    return rocm_runtime_env(rocm_root, msvc_build_env(base_env))


def install_runtime_env(
    root: Path, base_env: dict[str, str] | None = None
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    paths = [
        root / "bin",
        root / "lib",
    ]
    env["PATH"] = os.pathsep.join(
        [os.fspath(path) for path in paths] + [env.get("PATH", "")]
    )
    env["CMAKE_PREFIX_PATH"] = os.pathsep.join(
        [os.fspath(root), env.get("CMAKE_PREFIX_PATH", "")]
    )
    return env


def windows_toolchain(
    args: argparse.Namespace,
    rocm_root: Path,
    env: Mapping[str, str] | None = None,
) -> WindowsToolchain:
    toolchain_name = args.windows_toolchain
    if toolchain_name not in WINDOWS_TOOLCHAINS:
        raise ValueError(
            f"Unsupported Windows toolchain {toolchain_name!r}; expected one of "
            + ", ".join(WINDOWS_TOOLCHAINS)
        )
    rc = windows_sdk_tool("rc.exe", env)
    mt = windows_sdk_tool("mt.exe", env)
    if toolchain_name == "msvc":
        cl = require_tool_in_path(["cl.exe"], "MSVC compiler cl.exe", env)
        return WindowsToolchain(
            name=toolchain_name,
            c_compiler=cl,
            cxx_compiler=cl,
            asm_compiler=cl,
            ar=require_tool_in_path(["lib.exe"], "MSVC archive tool lib.exe", env),
            linker=require_tool_in_path(["link.exe"], "MSVC linker link.exe", env),
            rc=rc,
            mt=mt,
        )
    llvm_bin = rocm_llvm_bin(rocm_root)
    return WindowsToolchain(
        name=toolchain_name,
        c_compiler=rocm_tool(rocm_root, ["clang-cl.exe"]),
        cxx_compiler=rocm_tool(rocm_root, ["clang-cl.exe"]),
        asm_compiler=rocm_tool(rocm_root, ["clang-cl.exe"]),
        ar=find_path(
            [
                llvm_bin / "llvm-lib.exe",
                rocm_root / "bin" / "llvm-lib.exe",
            ],
            "ROCm LLVM archive tool llvm-lib.exe",
        ),
        linker=find_path(
            [
                llvm_bin / "lld-link.exe",
                rocm_root / "bin" / "lld-link.exe",
            ],
            "ROCm LLVM linker lld-link.exe",
        ),
        rc=rc,
        mt=mt,
    )


def build_core(args: argparse.Namespace) -> None:
    rocm_root = args.rocm_root.resolve()
    build_dir = args.build_dir.resolve()
    public_install_dir = args.public_install_dir.resolve()
    tests_install_dir = args.tests_install_dir.resolve()
    require_path(rocm_root, "ROCm build root")

    env = rocm_build_env(rocm_root)
    toolchain = windows_toolchain(args, rocm_root, env)

    cmake_prefix_path = cmake_path_list([rocm_root])
    ctest_enabled = True
    cmake_defines = [
        f"CMAKE_PREFIX_PATH={cmake_prefix_path}",
        f"CMAKE_INSTALL_PREFIX={cmake_path(public_install_dir)}",
        "CMAKE_INSTALL_BINDIR=bin",
        "CMAKE_INSTALL_LIBDIR=lib",
        f"CMAKE_C_COMPILER={cmake_path(toolchain.c_compiler)}",
        f"CMAKE_CXX_COMPILER={cmake_path(toolchain.cxx_compiler)}",
        f"CMAKE_ASM_COMPILER={cmake_path(toolchain.asm_compiler)}",
        f"CMAKE_AR={cmake_path(toolchain.ar)}",
        f"CMAKE_LINKER={cmake_path(toolchain.linker)}",
        f"CMAKE_RC_COMPILER={cmake_path(toolchain.rc)}",
        f"CMAKE_MT={cmake_path(toolchain.mt)}",
        f"CMAKE_BUILD_TYPE={args.build_type}",
        f"IREE_ROCM_DEPENDENCY_MODE={WINDOWS_ROCM_DEPENDENCY_MODE}",
        f"IREE_BUILD_TESTS={'ON' if ctest_enabled else 'OFF'}",
        "IREE_BUILD_BENCHMARKS=OFF",
        "LOOM_BUILD=OFF",
        "LIBHRX_BUILD=ON",
        "LIBHRX_BUILD_CTS=OFF",
        "LIBHRX_BUILD_HIP_BINDING=OFF",
        f"HRX_INSTALL_TESTS={'ON' if ctest_enabled else 'OFF'}",
        "LIBHRX_BUILD_PASSTHROUGH=OFF",
        "IREE_HAL_DRIVER_AMDGPU=OFF",
        f"IREE_ROCM_PATH={cmake_path(rocm_root)}",
        "IREE_ENABLE_LIBBACKTRACE=OFF",
        f"IREE_ENABLE_ASSERTIONS={'ON' if args.assertions else 'OFF'}",
    ]
    cmake_defines.extend(sanitizer_options(args.sanitizer))
    cmake_defines.extend(cmake_options_from_env())
    cmake_defines.extend(args.cmake_option)

    cmake_configure_cmd = [
        "cmake",
        "-S",
        REPO_ROOT,
        "-B",
        build_dir,
        "-GNinja",
        *[f"-D{option.removeprefix('-D')}" for option in cmake_defines],
    ]
    run(cmake_configure_cmd, cwd=REPO_ROOT, env=env, pretty_command=True)
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
    manifest = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "package": "hrx-public-deps-windows-x86_64",
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


def test_core(args: argparse.Namespace) -> None:
    rocm_root = args.rocm_root.resolve()
    smoke_build_dir = args.package_smoke_build_dir.resolve()
    if args.prepare_public_deps:
        prepare_public_deps_root(args)
    install_root = prepare_composed_install_root(args)
    installed_tests_dir = install_root / "share" / "hrx-system" / "tests"

    require_path(installed_tests_dir / "CTestTestfile.cmake", "installed CTest file")
    require_path(install_root / "bin" / "hrx.dll", "installed hrx.dll")
    require_path(install_root / "bin" / "hrx-info.exe", "installed hrx-info.exe")

    if args.package_smoke:
        base_env = rocm_build_env(rocm_root) if rocm_root.exists() else msvc_build_env()
    else:
        base_env = (
            rocm_runtime_env(rocm_root) if rocm_root.exists() else dict(os.environ)
        )
    env = install_runtime_env(install_root, base_env)
    env.setdefault("HRX_TEST_PYTHON", sys.executable)
    if args.cts_device:
        env["HRX_CTS_DEVICE"] = args.cts_device
    if args.test_tmpdir is not None:
        env["HRX_TEST_TMPDIR"] = os.fspath(args.test_tmpdir.resolve())

    ctest_parallelism = args.ctest_parallelism or default_ctest_parallelism()
    ctest_cmd = [
        "ctest",
        "--test-dir",
        installed_tests_dir,
        "--output-on-failure",
        "--parallel",
        str(ctest_parallelism),
    ]
    if args.ctest_regex:
        ctest_cmd.extend(["-R", args.ctest_regex])
    ctest_exclude_regex = combine_ctest_exclude_regex(
        *CORE_CTEST_EXCLUDE_REGEXES, args.ctest_exclude_regex
    )
    if ctest_exclude_regex:
        ctest_cmd.extend(["-E", ctest_exclude_regex])
    if args.ctest_label_regex:
        ctest_cmd.extend(["-L", args.ctest_label_regex])
    if args.ctest_label_exclude_regex:
        ctest_cmd.extend(["-LE", args.ctest_label_exclude_regex])
    run(ctest_cmd, cwd=REPO_ROOT, env=env, stderr_to_stdout=True)

    hrx_info = install_root / "bin" / "hrx-info.exe"
    if hrx_info.exists():
        run([hrx_info], cwd=REPO_ROOT, env=env)
        run([hrx_info, "--device=cpu:0"], cwd=REPO_ROOT, env=env)

    if not args.package_smoke:
        return

    hrx_smoke_build_dir = smoke_build_dir / "hrx"
    loomc_smoke_build_dir = smoke_build_dir / "loomc"
    toolchain = windows_toolchain(args, rocm_root, env)
    smoke_cmake_options = [
        "-GNinja",
        f"-DCMAKE_PREFIX_PATH={cmake_path_list([install_root, rocm_root])}",
        f"-DCMAKE_C_COMPILER={cmake_path(toolchain.c_compiler)}",
        f"-DCMAKE_CXX_COMPILER={cmake_path(toolchain.cxx_compiler)}",
        f"-DCMAKE_ASM_COMPILER={cmake_path(toolchain.asm_compiler)}",
        f"-DCMAKE_AR={cmake_path(toolchain.ar)}",
        f"-DCMAKE_LINKER={cmake_path(toolchain.linker)}",
        f"-DCMAKE_RC_COMPILER={cmake_path(toolchain.rc)}",
        f"-DCMAKE_MT={cmake_path(toolchain.mt)}",
        f"-DIREE_ROCM_DEPENDENCY_MODE={WINDOWS_ROCM_DEPENDENCY_MODE}",
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
    run(
        [hrx_smoke_build_dir / windows_executable_name("hrx_package_smoke")],
        cwd=REPO_ROOT,
        env=env,
    )
    run(
        [hrx_smoke_build_dir / windows_executable_name("hrx_package_smoke_cxx")],
        cwd=REPO_ROOT,
        env=env,
    )

    if not (install_root / "bin" / "loomc.dll").exists():
        log("Skipping loomc package smoke for Windows build with LOOM_BUILD=OFF.")
        return

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
    run(
        [loomc_smoke_build_dir / windows_executable_name("loomc_package_smoke")],
        cwd=REPO_ROOT,
        env=env,
    )


ROCM_BUILDENV_EXCLUDE_PATHS = {
    "bin/hrx-info.exe",
    "env.cmd",
    "env.ps1",
    "hrx-public-windows-x86_64-env.cmd",
    "hrx-public-windows-x86_64-env.ps1",
    "hrx-public-deps-windows-x86_64-env.cmd",
    "hrx-public-deps-windows-x86_64-env.ps1",
    "hrx-tests-windows-x86_64-env.cmd",
    "hrx-tests-windows-x86_64-env.ps1",
    "hrx-rocm-buildenv-windows-x86_64-env.cmd",
    "hrx-rocm-buildenv-windows-x86_64-env.ps1",
    "include/hrx",
    "include/loomc",
    "include/passthrough",
    "lib/cmake/hrx",
    "lib/cmake/loomc",
    "bin/hrx.dll",
    "bin/loomc.dll",
    "lib/hrx.lib",
    "lib/loomc.lib",
    "share/hrx-cts",
    "share/hrx-system",
}

ROCM_BUILDENV_EXCLUDE_PREFIXES = (
    "include/hrx/",
    "include/loomc/",
    "include/passthrough/",
    "lib/cmake/hrx/",
    "lib/cmake/loomc/",
    "share/hrx-cts/",
    "share/hrx-system/",
)


def make_env_ps1_content() -> str:
    return """# Source this file to use this HRX installation overlay.
$HrxEnvDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:HRX_HOME = $HrxEnvDir
$env:ROCM_HOME = $HrxEnvDir
$env:PATH = "$HrxEnvDir\\bin;$HrxEnvDir\\lib;$env:PATH"
$env:CMAKE_PREFIX_PATH = "$HrxEnvDir;$env:CMAKE_PREFIX_PATH"
Remove-Variable HrxEnvDir
"""


def make_env_cmd_content() -> str:
    return """@echo off
set "HRX_HOME=%~dp0"
set "ROCM_HOME=%~dp0"
set "PATH=%HRX_HOME%bin;%HRX_HOME%lib;%PATH%"
set "CMAKE_PREFIX_PATH=%HRX_HOME%;%CMAKE_PREFIX_PATH%"
"""


def write_package_env_scripts(output_dir: Path, script_stem: str) -> list[Path]:
    ps1_path = output_dir / f"{script_stem}.ps1"
    cmd_path = output_dir / f"{script_stem}.cmd"
    ps1_path.write_text(make_env_ps1_content())
    cmd_path.write_text(make_env_cmd_content())
    return [ps1_path, cmd_path]


def should_exclude_archive_path(
    arcname: str,
    *,
    exclude_paths: set[str],
    exclude_prefixes: tuple[str, ...],
) -> bool:
    normalized = arcname.replace("\\", "/")
    return normalized in exclude_paths or any(
        normalized.startswith(prefix) for prefix in exclude_prefixes
    )


def create_zip(
    source_dir: Path,
    zip_path: Path,
    *,
    exclude_paths: set[str] | None = None,
    exclude_prefixes: tuple[str, ...] = (),
) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    exclude_paths = exclude_paths or set()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            source = common.materialized_source_path(path)
            if source.is_dir():
                continue
            arcname = path.relative_to(source_dir).as_posix()
            if should_exclude_archive_path(
                arcname, exclude_paths=exclude_paths, exclude_prefixes=exclude_prefixes
            ):
                continue
            zf.write(source, arcname)


def package_tree(
    *,
    package_name: str,
    source_root: Path,
    output_dir: Path,
    suffix: str,
    rocm_root: Path | None = None,
    env_script_stem: str | None = None,
    exclude_paths: set[str] | None = None,
    exclude_prefixes: tuple[str, ...] = (),
) -> None:
    require_path(source_root, f"{package_name} source root")
    output_dir.mkdir(parents=True, exist_ok=True)
    package_exclude_paths = set(exclude_paths or set())
    package_exclude_paths.update({"env.cmd", "env.ps1"})
    if env_script_stem:
        package_exclude_paths.add(f"{env_script_stem}.cmd")
        package_exclude_paths.add(f"{env_script_stem}.ps1")

    package_stem = f"{package_name}-{suffix}" if suffix else package_name
    archive_path = output_dir / f"{package_stem}.zip"
    if archive_path.exists():
        archive_path.unlink()
    create_zip(
        source_root,
        archive_path,
        exclude_paths=package_exclude_paths,
        exclude_prefixes=exclude_prefixes,
    )
    env_script_paths: list[Path] = []
    if env_script_stem:
        env_script_paths = write_package_env_scripts(output_dir, env_script_stem)
    manifest_path = output_dir / f"{package_stem}.manifest.json"
    write_package_manifest(
        manifest_path,
        package_name=package_name,
        source_root=source_root,
        archive_path=archive_path,
        rocm_root=rocm_root,
    )
    log(f"Created {archive_path}")
    for env_script_path in env_script_paths:
        log(f"Created {env_script_path}")
    log(f"Created {manifest_path}")


GENERATED_PACKAGE_FILE_GLOBS = (
    "**/CTestTestfile.cmake",
    "**/CTestCustom.cmake",
    "**/*Config.cmake",
    "**/*-config.cmake",
)


def path_spellings(path: Path) -> set[str]:
    resolved = path.resolve()
    spellings = {
        os.fspath(resolved),
        resolved.as_posix(),
        os.fspath(resolved).replace("/", "\\"),
        os.fspath(resolved).replace("\\", "/"),
    }
    return {spelling for spelling in spellings if spelling}


def scan_generated_package_files(
    roots: Iterable[Path],
    *,
    forbidden_paths: Iterable[Path],
    forbidden_interpreters: Iterable[Path],
) -> None:
    forbidden_strings: set[str] = set()
    for path in forbidden_paths:
        forbidden_strings.update(path_spellings(path))
    for path in forbidden_interpreters:
        forbidden_strings.update(path_spellings(path))

    absolute_python_pattern = re.compile(
        r"([A-Za-z]:[/\\][^\"'\s;)]*python(?:3|\d+)?(?:\.exe)?)",
        re.IGNORECASE,
    )
    violations: list[str] = []
    seen_files: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for pattern in GENERATED_PACKAGE_FILE_GLOBS:
            for cmake_file in root.glob(pattern):
                if cmake_file in seen_files or not cmake_file.is_file():
                    continue
                seen_files.add(cmake_file)
                text = cmake_file.read_text(encoding="utf-8", errors="replace")
                for forbidden in sorted(forbidden_strings, key=len, reverse=True):
                    if forbidden and forbidden in text:
                        violations.append(f"{cmake_file}: captured {forbidden}")
                for match in absolute_python_pattern.finditer(text):
                    python_path = match.group(1)
                    if "HRX_TEST_PYTHON" not in text:
                        violations.append(
                            f"{cmake_file}: captured interpreter {python_path}"
                        )
                    elif python_path in forbidden_strings:
                        violations.append(
                            f"{cmake_file}: captured interpreter {python_path}"
                        )
    if violations:
        raise RuntimeError(
            "Generated package CMake/CTest files are not relocatable:\n  "
            + "\n  ".join(violations)
        )


def package_core(args: argparse.Namespace) -> None:
    rocm_root = args.rocm_root.resolve()
    output_dir = args.package_output_dir.resolve()
    public_install_dir = args.public_install_dir.resolve()
    tests_install_dir = args.tests_install_dir.resolve()
    suffix = args.package_suffix or dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")

    require_path(public_install_dir / "bin" / "hrx.dll", "public hrx.dll")
    require_path(public_install_dir / "bin" / "hrx-info.exe", "public hrx-info.exe")
    require_path(public_install_dir / "lib" / "hrx.lib", "public hrx.lib")
    require_path(
        public_install_dir / "include" / "hrx" / "hrx_runtime.h",
        "public HRX C runtime header",
    )
    require_path(
        public_install_dir / "include" / "hrx" / "hrx_runtime_cxx.h",
        "public HRX C++ runtime header",
    )
    require_path(
        public_install_dir / "lib" / "cmake" / "hrx" / "hrx-config.cmake",
        "public hrx CMake package",
    )
    require_path(
        tests_install_dir / "share" / "hrx-system" / "tests" / "CTestTestfile.cmake",
        "installed CTest file",
    )
    scan_generated_package_files(
        [public_install_dir, tests_install_dir],
        forbidden_paths=[
            REPO_ROOT,
            args.build_dir,
            public_install_dir,
            tests_install_dir,
        ],
        forbidden_interpreters=[Path(sys.executable)],
    )
    prepare_public_deps_root(args)

    package_tree(
        package_name="hrx-public-windows-x86_64",
        source_root=public_install_dir,
        output_dir=output_dir,
        suffix=suffix,
        rocm_root=rocm_root,
        env_script_stem="hrx-public-windows-x86_64-env",
    )
    package_tree(
        package_name="hrx-public-deps-windows-x86_64",
        source_root=args.public_deps_dir.resolve(),
        output_dir=output_dir,
        suffix=suffix,
        rocm_root=rocm_root,
        env_script_stem="hrx-public-deps-windows-x86_64-env",
    )
    package_tree(
        package_name="hrx-tests-windows-x86_64",
        source_root=tests_install_dir,
        output_dir=output_dir,
        suffix=suffix,
        rocm_root=rocm_root,
    )
    package_tree(
        package_name="hrx-rocm-buildenv-windows-x86_64",
        source_root=rocm_root,
        output_dir=output_dir,
        suffix=suffix,
        rocm_root=rocm_root,
        env_script_stem="hrx-rocm-buildenv-windows-x86_64-env",
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
        default_output=REPO_ROOT / "build" / "windows",
        sanitizer_choices=["none", "asan", "tsan", "ubsan"],
        passthrough_default=False,
        artifact_sets=ARTIFACT_SETS,
    )
    parser.add_argument(
        "--windows-toolchain",
        default=common.env_default("HRX_WINDOWS_TOOLCHAIN", "msvc"),
        choices=WINDOWS_TOOLCHAINS,
        help="Coherent Windows compiler/linker tuple to use.",
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
