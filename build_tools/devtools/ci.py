#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Local runner for source-tree IREE CI jobs."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_PY_ROOT = str(REPO_ROOT)
if _REPO_PY_ROOT not in sys.path:
    sys.path.insert(0, _REPO_PY_ROOT)
_LOOM_PY_ROOT = str(REPO_ROOT / "loom" / "py")
if _LOOM_PY_ROOT not in sys.path:
    sys.path.insert(0, _LOOM_PY_ROOT)

from loom.build_tools import target_presets
from loom.build_tools.bazel import configure as loom_configure
from runtime.build_tools.bazel import configure as runtime_configure

CMAKE_CI_BUILD_ROOT = Path("build/ci")
TSAN_SUPPRESSIONS_FILE = REPO_ROOT / "build_tools/devtools/tsan.supp"
CMAKE_SANITIZER_OPTIONS = {
    "asan": ("-DIREE_ENABLE_ASAN=ON",),
    "msan": ("-DIREE_ENABLE_MSAN=ON",),
    "tsan": ("-DIREE_ENABLE_TSAN=ON",),
    "ubsan": ("-DIREE_ENABLE_UBSAN=ON",),
}
CMAKE_HAL_DRIVER_DEFINES = tuple(
    (driver, option)
    for option, driver in runtime_configure.DRIVER_DEFINES.items()
    if driver != "task"
)
CMAKE_LOOM_ARCH_DEFINES = tuple(
    (architecture, option)
    for option, architecture in target_presets.ARCH_DEFINES.items()
)
CMAKE_LOOM_EMIT_DEFINES = tuple(
    (artifact_format, option)
    for option, artifact_format in target_presets.EMIT_DEFINES.items()
)
CMAKE_LOOM_EXECUTE_DEFINES = tuple(
    (substrate, option) for option, substrate in target_presets.EXECUTE_DEFINES.items()
)
CMAKE_LOOM_IMPORTER_DEFINES = tuple(
    (importer, option) for option, importer in loom_configure.IMPORT_DEFINES.items()
)
CI_SUPPORTED_HAL_DRIVERS = frozenset(
    ("task", *(driver for driver, _ in CMAKE_HAL_DRIVER_DEFINES))
)
REPOSITORY_BUILD_HAL_DRIVERS = (
    "amd",
    *(
        driver
        for driver in runtime_configure.ALL_DRIVERS
        if driver != "amd" and driver not in runtime_configure.AMD_ENGINES
    ),
)
REPOSITORY_BUILD_LOOM_ARCHITECTURES = loom_configure.ARCHITECTURES
REPOSITORY_BUILD_LOOM_ARTIFACT_FORMATS = loom_configure.ARTIFACT_FORMATS
REPOSITORY_BUILD_LOOM_EXECUTION_SUBSTRATES = loom_configure.EXECUTION_SUBSTRATES
REPOSITORY_BUILD_LOOM_IMPORTERS = loom_configure.IMPORTERS
ROCM_PINNED_DEPENDENCY_MODE_OPTION = "-DIREE_ROCM_DEPENDENCY_MODE=pinned"
AMDGPU_DEVICE_BINARY_SOURCE_OPTIONS = (
    "-DIREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE=source",
    "-DIREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm",
)
AMDGPU_DEVICE_BINARY_PREBUILT_OPTIONS = (
    "-DIREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE=prebuilt",
    "-DIREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=none",
)
BAZEL_COMMANDS = {
    "iree-bazel-amd-client": ("amd-client", None),
    "iree-bazel-amd-client-asan": ("amd-client", "asan"),
    "iree-bazel-xdna": ("xdna", None),
    "iree-bazel-xdna-asan": ("xdna", "asan"),
    "iree-bazel-cpu": ("cpu", None),
    "iree-bazel-repository-build": ("repository-build", None),
    "iree-bazel-repository-integration": ("repository-integration", None),
    "iree-bazel-cpu-asan": ("cpu", "asan"),
    "iree-bazel-cpu-msan": ("cpu", "msan"),
    "iree-bazel-cpu-tsan": ("cpu", "tsan"),
    "iree-bazel-cpu-ubsan": ("cpu", "ubsan"),
    "iree-bazel-cpu-sanitizers": ("cpu", "all"),
    "iree-bazel-amdgpu": ("amdgpu", None),
    "iree-bazel-amdgpu-asan": ("amdgpu", "asan"),
    "iree-bazel-amdgpu-tsan": ("amdgpu", "tsan"),
    "iree-bazel-amdgpu-ubsan": ("amdgpu", "ubsan"),
    "iree-bazel-amdgpu-native": ("amdgpu-native", None),
    "iree-bazel-amdgpu-native-asan": ("amdgpu-native", "asan"),
    "iree-bazel-loom-amdgpu": ("loom-amdgpu", None),
    "iree-bazel-vulkan": ("vulkan", None),
}
CMAKE_COMMANDS = {
    "iree-cmake-xdna": ("xdna", None),
    "iree-cmake-xdna-asan": ("xdna", "asan"),
    "iree-cmake-cpu": ("cpu", None),
    "iree-cmake-repository-build": ("repository-build", None),
    "iree-cmake-cpu-asan": ("cpu", "asan"),
    "iree-cmake-cpu-msan": ("cpu", "msan"),
    "iree-cmake-cpu-tsan": ("cpu", "tsan"),
    "iree-cmake-cpu-ubsan": ("cpu", "ubsan"),
    "iree-cmake-cpu-sanitizers": ("cpu", "all"),
    "iree-cmake-amdgpu": ("amdgpu", None),
    "iree-cmake-amdgpu-asan": ("amdgpu", "asan"),
    "iree-cmake-amdgpu-msan": ("amdgpu", "msan"),
    "iree-cmake-amdgpu-tsan": ("amdgpu", "tsan"),
    "iree-cmake-amdgpu-ubsan": ("amdgpu", "ubsan"),
    "iree-cmake-amdgpu-sanitizers": ("amdgpu", "all"),
    "iree-cmake-loom-amdgpu": ("loom-amdgpu", None),
    "iree-cmake-vulkan": ("vulkan", None),
    "iree-cmake-vulkan-asan": ("vulkan", "asan"),
    "iree-cmake-vulkan-msan": ("vulkan", "msan"),
    "iree-cmake-vulkan-tsan": ("vulkan", "tsan"),
    "iree-cmake-vulkan-ubsan": ("vulkan", "ubsan"),
    "iree-cmake-vulkan-sanitizers": ("vulkan", "all"),
}
CMAKE_SANITIZER_SMOKE_COMMAND = "iree-cmake-sanitizer-smoke"
IMPORTER_COMMANDS = {
    "iree-importers-tilelang": "tilelang",
}
if __package__:
    from . import ci_config
else:
    from build_tools.devtools import ci_config

from build_tools.ci import windows_diagnostics
from build_tools.devtools import (
    bazel_profile_summary,
    bazel_test_results,
    run_requirements,
)


@dataclass(frozen=True)
class RequirementAudit:
    # Read-only command exposing tags before resource filtering.
    argv: tuple[str, ...]
    # Parser that rejects undeclared requirements in the command output.
    validate: Callable[[str], None]


@dataclass(frozen=True)
class CiStep:
    name: str
    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()
    # Metadata preflight that must succeed before executing a test command.
    requirement_audit: RequirementAudit | None = None
    # Bazel execution profile summarized after this command completes.
    bazel_profile: Path | None = None
    # Current-invocation Bazel test evidence staged after this command.
    test_results: bazel_test_results.BazelTestResultsCapture | None = None

    def command_line(self) -> str:
        env_args = tuple(f"{key}={value}" for key, value in self.env)
        return shlex.join(env_args + self.argv)


@dataclass(frozen=True)
class StepResult:
    step: CiStep
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def command_targets(explicit_targets: list[str] | None = None) -> tuple[str, ...]:
    if explicit_targets:
        return tuple(explicit_targets)
    targets = []
    for directory in ci_config.IREE_TARGET_DIRECTORIES:
        if (REPO_ROOT / directory).is_dir():
            targets.append(f"//{directory}/...")
    if not targets:
        raise RuntimeError("no IREE target directories exist in this checkout")
    return tuple(targets)


def dev_command(*args: str) -> tuple[str, ...]:
    return (os.environ.get("IREE_CI_PYTHON", "python3"), "dev.py", *args)


def sanitizer_env(config: str | None) -> tuple[tuple[str, str], ...]:
    if config not in CMAKE_SANITIZER_OPTIONS:
        return ()
    environment = []
    if config == "tsan":
        environment.append(("TSAN_OPTIONS", f"suppressions={TSAN_SUPPRESSIONS_FILE}"))

    # Bazel's test environment does not inherit the host toolchain's PATH.
    # Pass the symbolizer explicitly so sanitizer failures retain source frames.
    symbolizer_key = f"{config.upper()}_SYMBOLIZER_PATH"
    symbolizer = os.environ.get(symbolizer_key)
    if not symbolizer and (rocm_root := os.environ.get("HRX_ROCM_ROOT")):
        executable = (
            "llvm-symbolizer.exe" if sys.platform == "win32" else "llvm-symbolizer"
        )
        candidate = Path(rocm_root) / "lib" / "llvm" / "bin" / executable
        if candidate.is_file():
            symbolizer = str(candidate)
    if not symbolizer:
        symbolizer = shutil.which("llvm-symbolizer")
    if symbolizer:
        environment.append((symbolizer_key, symbolizer))
    return tuple(environment)


def platform_coverage_enabled() -> bool:
    selection = os.environ.get("IREE_CI_PLATFORM_COVERAGE", "0")
    if selection not in ("0", "1"):
        raise ValueError("IREE_CI_PLATFORM_COVERAGE must be 0 or 1")
    return selection == "1"


def platform_coverage_build_steps(step: CiStep) -> list[CiStep]:
    return [step] if platform_coverage_enabled() else []


def codec_options(build_system: str) -> tuple[str, ...]:
    enabled = platform_coverage_enabled()
    groups = (
        ("image", ci_config.OPTIONAL_IMAGE_CODECS),
        ("audio", ci_config.OPTIONAL_AUDIO_CODECS),
    )
    if build_system == "cmake":
        return tuple(
            "-DLOOM_SERVE_"
            + kind.upper()
            + "_CODECS="
            + ";".join(codecs if enabled else ())
            for kind, codecs in groups
        )
    return tuple(
        "--//loom/config/serve:"
        + kind
        + "_codecs="
        + ",".join(codecs if enabled else ())
        for kind, codecs in groups
    )


# Bazel test actions do not inherit the client environment. Preserve the
# selectors that establish the runner's GPU lease without choosing one here.
AMD_DEVICE_VISIBILITY_ENV_VARS = (
    "HAL_VISIBLE_XDNA_DEVICES",
    "HAL_VISIBLE_RDNA_DEVICES",
    "HAL_VISIBLE_CDNA_DEVICES",
    "HAL_VISIBLE_AMDGPU_DEVICES",
    "AMDF_CTS_GPU_NATIVE_IDENTITY",
)


def amd_test_env() -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, value)
        for name in AMD_DEVICE_VISIBILITY_ENV_VARS
        if (value := os.environ.get(name)) is not None
    )


def amd_gpu_cts_test_env(target_selector: str) -> tuple[tuple[str, str], ...]:
    # Exclude only HAL backend suffixes. Native AQL/PM4/SDMA tests retain their
    # own capability admission regardless of the selected GPU architecture.
    selectors = target_selector.split(",")
    family = next(
        (
            name
            for name, prefixes in ci_config.AMD_GPU_CTS_TARGET_PREFIXES.items()
            if all(selector.startswith(prefixes) for selector in selectors)
        ),
        None,
    )
    if family is None:
        return ()
    positive, _, negative = os.environ.get("GTEST_FILTER", "*").partition("-")
    exclusions = [negative] if negative else []
    for name in ci_config.AMD_GPU_CTS_TARGET_PREFIXES:
        if name != family:
            exclusions.extend((f"CTS*/*/{name}", f"CTS*/*/{name}_*"))
    return (("GTEST_FILTER", positive + "-" + ":".join(exclusions)),)


def vulkan_device_test_env() -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, value)
        for name in ("DRI_PRIME", "AMDF_CTS_GPU_NATIVE_IDENTITY")
        if (value := os.environ.get(name))
    )


def cmake_amdgpu_device_binary_options() -> tuple[str, ...]:
    rocm_root = os.environ.get("HRX_ROCM_ROOT")
    rocm_path_option = (
        (f"-DIREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH={rocm_root}",)
        if rocm_root
        else ()
    )
    return AMDGPU_DEVICE_BINARY_SOURCE_OPTIONS + rocm_path_option


def cmake_tests_enabled(sanitizer: str | None) -> bool:
    if sanitizer is None:
        return True
    if sanitizer in ci_config.SANITIZER_TEST_CONFIGS:
        return True
    if sanitizer in ci_config.SANITIZER_BUILD_CONFIGS:
        return False
    raise ValueError(f"unknown CMake sanitizer config: {sanitizer}")


def cmake_dev_command(command_name: str, *args: str) -> tuple[str, ...]:
    build_dir = CMAKE_CI_BUILD_ROOT / command_name
    return dev_command("--cmake-build-dir", str(build_dir), "cmake", *args)


def validate_enabled_drivers(enabled_drivers: tuple[str, ...]) -> frozenset[str]:
    enabled_driver_set = frozenset(enabled_drivers)
    unsupported_drivers = enabled_driver_set.difference(CI_SUPPORTED_HAL_DRIVERS)
    if unsupported_drivers:
        raise ValueError(
            "unsupported CI HAL driver(s): " + ", ".join(sorted(unsupported_drivers))
        )
    if "amd" in enabled_driver_set:
        enabled_driver_set |= {"amd-gpu", "amd-xdna"}
    return enabled_driver_set


def amdgpu_bazel_options(target_selector: str) -> tuple[str, ...]:
    return (
        runtime_configure.NATIVE_AMDGPU_TARGETS_FLAG + "=" + target_selector,
        loom_configure.NATIVE_AMDGPU_TARGETS_FLAG + "=iree_hal",
    )


def amd_gpu_device_compiler_options() -> tuple[str, ...]:
    # The native HAL consumes HIP code objects without loading ROCr. Its CTS
    # still compiles real HIP inputs with the installed ROCm device toolchain.
    options = ("--repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm",)
    if rocm_root := os.environ.get("HRX_ROCM_ROOT"):
        options += (f"--repo_env=IREE_ROCM_PATH={rocm_root}",)
    return options


def amd_gpu_windows_weights_step(
    options: tuple[str, ...], config: str | None
) -> CiStep:
    return bazel_test_step(
        "Test AMD GPU weight streaming",
        (ci_config.AMD_GPU_WEIGHTS_TARGET,),
        config=config,
        available_resources=ci_config.AMDGPU_WINDOWS_RESOURCES,
        test_env=amd_test_env(),
        bazel_options=options
        + (
            "--test_arg=--weight_device_uri=rdna",
            "--test_filter=Streaming/*",
        ),
    )


def bazel_configure_step(
    enabled_drivers: tuple[str, ...] = (),
    *,
    enabled_loom_architectures: tuple[str, ...] | None = None,
    enabled_loom_artifact_formats: tuple[str, ...] | None = None,
    enabled_loom_execution_substrates: tuple[str, ...] | None = None,
    enabled_loom_importers: tuple[str, ...] | None = None,
    extra_options: tuple[str, ...] = (),
) -> CiStep:
    enabled_driver_set = validate_enabled_drivers(enabled_drivers)
    drivers = sorted((enabled_driver_set | {"task"}) - {"amd"})
    command = [
        "bazel",
        "configure",
        runtime_configure.NATIVE_DRIVER_FLAG + "=" + ",".join(drivers),
    ]
    if "amdgpu" in enabled_driver_set:
        command.append(ROCM_PINNED_DEPENDENCY_MODE_OPTION)
        rocm_root = os.environ.get("HRX_ROCM_ROOT")
        if rocm_root:
            command.append(f"-DIREE_ROCM_PATH={rocm_root}")
    if enabled_loom_architectures is not None:
        command.append(
            loom_configure.NATIVE_ARCH_FLAG + "=" + ",".join(enabled_loom_architectures)
        )
    if enabled_loom_artifact_formats is not None:
        command.append(
            loom_configure.NATIVE_EMIT_FLAG
            + "="
            + ",".join(enabled_loom_artifact_formats)
        )
    if enabled_loom_execution_substrates is not None:
        command.append(
            loom_configure.NATIVE_EXECUTE_FLAG
            + "="
            + ",".join(enabled_loom_execution_substrates)
        )
    if enabled_loom_importers is not None:
        command.append(
            loom_configure.NATIVE_IMPORT_FLAG + "=" + ",".join(enabled_loom_importers)
        )
    return CiStep(
        "Configure Bazel",
        dev_command(*command, *codec_options("bazel"), *extra_options),
    )


def bazel_build_step(
    name: str,
    targets: tuple[str, ...],
    config: str | None = None,
    bazel_options: tuple[str, ...] = (),
) -> CiStep:
    command = [
        "bazel",
        "build",
        *ci_config.BAZEL_DEFAULT_OPTIONS,
        *codec_options("bazel"),
    ]
    if config is not None:
        command.append(f"--config={config}")
    command.extend(bazel_options)
    if any(target.startswith("-") for target in targets):
        command.append("--")
    command.extend(targets)
    return CiStep(name, dev_command(*command))


def bazel_test_step(
    name: str,
    targets: tuple[str, ...],
    config: str | None = None,
    test_tag_filters: tuple[str, ...] = (),
    test_language_filters: tuple[str, ...] = (),
    test_env: tuple[tuple[str, str], ...] = (),
    bazel_options: tuple[str, ...] = (),
    available_resources: tuple[str, ...] = (),
) -> CiStep:
    for option_name, argument_name in (
        ("--test_tag_filters", "test_tag_filters"),
        ("--test_lang_filters", "test_language_filters"),
    ):
        if any(option.split("=", 1)[0] == option_name for option in bazel_options):
            raise ValueError(f"Use {argument_name} to compose Bazel test filters")
    options = list(ci_config.BAZEL_DEFAULT_OPTIONS)
    options.extend(codec_options("bazel"))
    if config is not None:
        options.append(f"--config={config}")
    for key, value in sanitizer_env(config) + test_env:
        options.append(f"--test_env={key}={value}")
    options.extend(bazel_options)
    options.append("--build_tests_only")
    # Focus filters compose with mandatory all-participant admission.
    test_tag_filters = tuple(
        dict.fromkeys(
            test_tag_filters + run_requirements.bazel_exclusions(available_resources)
        )
    )
    if test_tag_filters:
        options.append("--test_tag_filters=" + ",".join(test_tag_filters))
    if test_language_filters:
        options.append("--test_lang_filters=" + ",".join(test_language_filters))
    command = ["bazel", "test", *options]
    if any(target.startswith("-") for target in targets):
        command.append("--")
    command.extend(targets)
    return CiStep(
        name,
        dev_command(*command),
        requirement_audit=RequirementAudit(
            dev_command(
                "bazel",
                "query",
                run_requirements.bazel_test_query(targets),
                "--output=xml",
            ),
            run_requirements.audit_bazel_xml,
        ),
    )


def bazel_run_step(
    name: str,
    target: str,
    program_args: tuple[str, ...] = (),
    bazel_options: tuple[str, ...] = (),
) -> CiStep:
    command = [
        "bazel",
        "run",
        *ci_config.BAZEL_DEFAULT_OPTIONS,
        *codec_options("bazel"),
        *bazel_options,
        target,
    ]
    if program_args:
        command.extend(("--", *program_args))
    return CiStep(name, dev_command(*command))


def cmake_configure_step(
    command_name: str,
    *,
    enabled_drivers: tuple[str, ...] = (),
    enabled_loom_architectures: tuple[str, ...] | None = None,
    enabled_loom_artifact_formats: tuple[str, ...] | None = None,
    enabled_loom_execution_substrates: tuple[str, ...] | None = None,
    enabled_loom_importers: tuple[str, ...] | None = None,
    amdgpu_target_selector: str | None = ci_config.DEFAULT_AMDGPU_TARGET_SELECTOR,
    amdgpu_device_binary_mode: str = "source",
    sanitizer: str | None = None,
    build_tests: bool | None = None,
    extra_options: tuple[str, ...] = (),
) -> CiStep:
    enabled_driver_set = validate_enabled_drivers(enabled_drivers)
    loom_capability_axes = (
        enabled_loom_architectures,
        enabled_loom_artifact_formats,
        enabled_loom_execution_substrates,
    )
    if any(axis is not None for axis in loom_capability_axes) and any(
        axis is None for axis in loom_capability_axes
    ):
        raise ValueError(
            "CMake Loom capability selection requires architecture, artifact, "
            "and execution lists together"
        )
    tests_enabled = (
        cmake_tests_enabled(sanitizer) if build_tests is None else build_tests
    )
    command = [
        "configure",
        "--fresh",
        "-GNinja",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        f"-DIREE_BUILD_TESTS={'ON' if tests_enabled else 'OFF'}",
        f"-DIREE_BUILD_BENCHMARKS={'ON' if tests_enabled else 'OFF'}",
        "-DIREE_ENABLE_LIBBACKTRACE=OFF",
        "-DLIBHRX_BUILD=OFF",
        "-DIREE_ENABLE_VULKAN=OFF",
        "-DIREE_ENABLE_D3D12=OFF",
        *codec_options("cmake"),
    ]
    for driver, define in CMAKE_HAL_DRIVER_DEFINES:
        command.append(f"-D{define}={'ON' if driver in enabled_driver_set else 'OFF'}")
    if enabled_loom_architectures is not None:
        command.append("-DLOOM_TARGET_DEFAULT=OFF")
        enabled_loom_architecture_set = frozenset(enabled_loom_architectures)
        for architecture, define in CMAKE_LOOM_ARCH_DEFINES:
            command.append(
                f"-D{define}={'ON' if architecture in enabled_loom_architecture_set else 'OFF'}"
            )
    if enabled_loom_artifact_formats is not None:
        enabled_loom_artifact_format_set = frozenset(enabled_loom_artifact_formats)
        for artifact_format, define in CMAKE_LOOM_EMIT_DEFINES:
            command.append(
                f"-D{define}={'ON' if artifact_format in enabled_loom_artifact_format_set else 'OFF'}"
            )
    if enabled_loom_execution_substrates is not None:
        enabled_loom_execution_substrate_set = frozenset(
            enabled_loom_execution_substrates
        )
        for substrate, define in CMAKE_LOOM_EXECUTE_DEFINES:
            command.append(
                f"-D{define}={'ON' if substrate in enabled_loom_execution_substrate_set else 'OFF'}"
            )
    if enabled_loom_importers is not None:
        enabled_loom_importer_set = frozenset(enabled_loom_importers)
        for importer, define in CMAKE_LOOM_IMPORTER_DEFINES:
            command.append(
                f"-D{define}={'ON' if importer in enabled_loom_importer_set else 'OFF'}"
            )
    if "amdgpu" in enabled_driver_set:
        command.append(ROCM_PINNED_DEPENDENCY_MODE_OPTION)
        if amdgpu_target_selector is not None:
            command.append(f"-DIREE_HAL_AMDGPU_TARGETS={amdgpu_target_selector}")
            command.append("-DLOOM_ARCH_AMDGPU_TARGETS=iree_hal")
        if amdgpu_device_binary_mode == "source":
            command.extend(cmake_amdgpu_device_binary_options())
        elif amdgpu_device_binary_mode == "prebuilt":
            command.extend(AMDGPU_DEVICE_BINARY_PREBUILT_OPTIONS)
        else:
            raise ValueError(
                f"unsupported AMDGPU device binary mode: {amdgpu_device_binary_mode}"
            )
    if sanitizer is not None:
        command.append("-DIREE_ENABLE_ASSERTIONS=ON")
        command.extend(CMAKE_SANITIZER_OPTIONS[sanitizer])
    return CiStep(
        "Configure CMake", cmake_dev_command(command_name, *command, *extra_options)
    )


def cmake_build_step(
    command_name: str,
    name: str,
    targets: tuple[str, ...] = (),
) -> CiStep:
    parallelism = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
    parallel_args = ("--parallel", parallelism) if parallelism else ("--parallel",)
    return CiStep(
        name,
        cmake_dev_command(command_name, "build", *targets, *parallel_args),
    )


def cmake_explicit_build_steps(
    command_name: str,
    name: str,
    targets: tuple[str, ...] = (),
    *,
    tests_enabled: bool,
) -> list[CiStep]:
    step = cmake_build_step(command_name, name, targets)
    if not tests_enabled:
        return [step]
    return platform_coverage_build_steps(step)


def combine_ctest_regex(*regexes: str) -> str:
    return "|".join(f"({regex})" for regex in regexes if regex)


def cmake_test_step(
    command_name: str,
    name: str,
    *,
    regex: str | None = None,
    label_regex: str | None = None,
    label_exclude_regex: str | None = None,
    exclude_regex: str = "",
    env: tuple[tuple[str, str], ...] = (),
    parallelism: int = 8,
    available_resources: tuple[str, ...] = (),
) -> CiStep:
    command = ["test", "--parallel", str(parallelism), "--no-tests=error"]
    if regex:
        command.extend(["-R", regex])
    if exclude_regex:
        command.extend(["-E", exclude_regex])
    if label_regex:
        command.extend(["-L", label_regex])
    if label_exclude_regex:
        command.extend(["-LE", label_exclude_regex])
    audit_command = cmake_dev_command(command_name, *command, "--show-only=json-v1")
    # CTest -LE accepts one regex; combine admission with caller exclusions.
    unavailable = run_requirements.ctest_exclusion_regex(available_resources)
    if unavailable:
        if label_exclude_regex:
            command[-1] = combine_ctest_regex(label_exclude_regex, unavailable)
        else:
            command.extend(["-LE", unavailable])
    return CiStep(
        name,
        cmake_dev_command(command_name, *command),
        env=env,
        requirement_audit=RequirementAudit(
            audit_command, run_requirements.audit_ctest_json
        ),
    )


def cpu_steps(targets: tuple[str, ...]) -> list[CiStep]:
    scoped_targets = targets + ci_config.CPU_BAZEL_TARGET_EXCLUDES
    return [
        bazel_configure_step(),
        *platform_coverage_build_steps(bazel_build_step("Build IREE", scoped_targets)),
        bazel_test_step(
            "Test IREE",
            scoped_targets + ci_config.CPU_XFAIL_TARGETS,
            test_tag_filters=ci_config.CPU_RESOURCE_TAG_EXCLUDES,
        ),
    ]


def repository_configure_step() -> CiStep:
    return bazel_configure_step(
        enabled_drivers=REPOSITORY_BUILD_HAL_DRIVERS,
        enabled_loom_architectures=REPOSITORY_BUILD_LOOM_ARCHITECTURES,
        enabled_loom_artifact_formats=REPOSITORY_BUILD_LOOM_ARTIFACT_FORMATS,
        enabled_loom_execution_substrates=REPOSITORY_BUILD_LOOM_EXECUTION_SUBSTRATES,
        enabled_loom_importers=REPOSITORY_BUILD_LOOM_IMPORTERS,
        extra_options=ci_config.REPOSITORY_BAZEL_OPTIONS,
    )


def repository_build_step() -> CiStep:
    return bazel_build_step(
        "Build repository",
        ("//...",),
        bazel_options=ci_config.REPOSITORY_BAZEL_OPTIONS,
    )


def repository_build_steps() -> list[CiStep]:
    return [
        repository_configure_step(),
        repository_build_step(),
    ]


def repository_integration_steps(amdgpu_target_selector: str) -> list[CiStep]:
    if not os.environ.get("HRX_ROCM_ROOT"):
        raise ValueError(
            "iree-bazel-repository-integration requires HRX_ROCM_ROOT so the "
            "AMDGPU device compiler is exercised"
        )
    platform_coverage = platform_coverage_enabled()
    amdgpu_options = amdgpu_bazel_options(amdgpu_target_selector)
    test_tag_filters = ci_config.CPU_RESOURCE_TAG_EXCLUDES
    test_language_filters: tuple[str, ...] = ()
    if not platform_coverage:
        # The secondary compiler lane keeps focused native tests while the
        # platform lane owns portable Python and corpus scenario coverage.
        test_tag_filters += ("-loom-corpus",)
        test_language_filters = ci_config.PYTHON_TEST_BAZEL_LANGUAGE_EXCLUSIONS
    return [
        repository_configure_step(),
        *([repository_build_step()] if platform_coverage else []),
        bazel_build_step(
            "Build AMDGPU device toolchain smoke",
            ci_config.BAZEL_REPOSITORY_INTEGRATION_DEVICE_TARGETS,
            bazel_options=ci_config.REPOSITORY_BAZEL_OPTIONS + amdgpu_options,
        ),
        bazel_test_step(
            "Test repository",
            ci_config.BAZEL_REPOSITORY_TEST_TARGETS,
            test_tag_filters=test_tag_filters,
            test_language_filters=test_language_filters,
            bazel_options=ci_config.REPOSITORY_BAZEL_OPTIONS,
        ),
        bazel_run_step(
            "Run dynamic library environment smoke",
            ci_config.BAZEL_REPOSITORY_INTEGRATION_DYNAMIC_LIBRARY_TARGET,
            bazel_options=ci_config.REPOSITORY_BAZEL_OPTIONS,
        ),
        bazel_run_step(
            "Run executable alias smoke",
            ci_config.BAZEL_REPOSITORY_INTEGRATION_ALIAS_TARGET,
            program_args=("--help",),
            bazel_options=ci_config.REPOSITORY_BAZEL_OPTIONS,
        ),
        shared_tool_bazel_step(bazel_options=ci_config.REPOSITORY_BAZEL_OPTIONS),
    ]


def shared_tool_bazel_step(
    config: str | None = None, bazel_options: tuple[str, ...] = ()
) -> CiStep:
    return bazel_test_step(
        "Test shared Loom tools",
        (
            "//loom/src/loom/tools/loom-compile:execution_test",
            "//loom/src/loom/tools/loom-opt:execution_test",
            "//loom/src/loom/tools/loom-format:execution_test",
            "//loom/src/loom/tools/loom-link:execution_test",
            "//loom/src/loom/tools/loom-run:execution_test",
            "//loom/src/loom/tooling/target/cpu:run_cxx_input_test",
            "//loom/src/loom/import/cxx/test:export_roundtrip_test",
            "//loom/src/loom/tools/loom-import-cxx:execution_test",
            "//loom/src/loom/tools/loom-serve:execution_test",
            "//loom/src/loom/tools/loom-check:execution_test",
            "//loom/src/loom/tools/loom-check/test:core_test",
            "//loom/src/loom/tools/loom-test:execution_test",
            "//loom/src/loom/tools/loom-test:scenario_execution_test",
            "//loom/src/loom/tools/loom-benchmark:execution_test",
            "//loom/src/loom/tooling/execution/hal:command_scenario_execution_test",
            "//loom/src/loom/tooling/execution/hal:command_scenario_specialization_test",
            "//loom/src/loom/tooling/target/cpu:benchmark_scenario_correctness_test",
            "//loom/src/loom/tooling/target/cpu/serve:all",
        ),
        config=config,
        bazel_options=bazel_options + ("--//loom/config:shared_tools",),
    )


def cpu_sanitizer_steps(targets: tuple[str, ...]) -> list[CiStep]:
    steps = [bazel_configure_step()]
    for config in ci_config.SANITIZER_TEST_CONFIGS:
        steps.extend(cpu_config_steps(targets, config))
    return steps


def cpu_config_steps(targets: tuple[str, ...], config: str) -> list[CiStep]:
    scoped_targets = targets + ci_config.CPU_BAZEL_TARGET_EXCLUDES
    if config in ci_config.SANITIZER_TEST_CONFIGS:
        steps = [
            *platform_coverage_build_steps(
                bazel_build_step(
                    f"Build IREE with {config.upper()}",
                    scoped_targets,
                    config=config,
                )
            ),
            bazel_test_step(
                f"Test IREE with {config.upper()}",
                scoped_targets + ci_config.CPU_SANITIZERS_XFAIL_TARGETS,
                config=config,
                test_tag_filters=ci_config.CPU_RESOURCE_TAG_EXCLUDES,
                test_language_filters=ci_config.PYTHON_TEST_BAZEL_LANGUAGE_EXCLUSIONS,
            ),
        ]
        if config == "asan":
            steps.append(
                shared_tool_bazel_step(
                    config=config,
                    bazel_options=("--//loom/config/import:enable=cxx",),
                )
            )
        return steps
    if config in ci_config.SANITIZER_BUILD_CONFIGS:
        return [
            bazel_build_step(
                f"Build IREE with {config.upper()}",
                scoped_targets,
                config=config,
            )
        ]
    raise ValueError(f"unknown Bazel sanitizer config: {config}")


def xdna_steps(targets: tuple[str, ...], config: str | None) -> list[CiStep]:
    config_name = f" / {config.upper()}" if config is not None else ""
    # Native ELF consumers and the HAL share the XDNA provider.
    options = (
        "--//libamdf/config:families=xdna",
        "--//runtime/config/hal:drivers=task,amd-xdna",
    )
    return [
        bazel_configure_step(
            enabled_loom_architectures=("aie2p", "vm"),
            enabled_loom_artifact_formats=("elf", "vm"),
            enabled_loom_execution_substrates=("iree_hal", "iree_vm"),
            extra_options=options,
        ),
        *platform_coverage_build_steps(
            bazel_build_step(
                f"Build IREE / XDNA{config_name}",
                targets,
                config=config,
                bazel_options=options,
            )
        ),
        bazel_test_step(
            f"Test IREE / XDNA{config_name}",
            targets,
            config=config,
            test_tag_filters=ci_config.XDNA_BAZEL_TEST_TAG_FILTERS,
            available_resources=ci_config.XDNA_RESOURCES,
            test_env=amd_test_env(),
            bazel_options=options,
        ),
    ]


def amd_client_steps(targets: tuple[str, ...], config: str | None) -> list[CiStep]:
    config_name = f" / {config.upper()}" if config is not None else ""
    options = ci_config.AMD_CLIENT_BAZEL_OPTIONS
    default_targets = ci_config.AMD_CLIENT_BAZEL_TARGETS
    resources = ci_config.XDNA_RESOURCES
    test_tag_filters = ci_config.XDNA_BAZEL_TEST_TAG_FILTERS
    native_options = ()
    if sys.platform == "win32":
        options = ci_config.AMD_CLIENT_WINDOWS_BAZEL_OPTIONS
        default_targets = ci_config.AMD_CLIENT_WINDOWS_BAZEL_TARGETS
        resources = ci_config.AMD_CLIENT_WINDOWS_RESOURCES
        test_tag_filters += (
            ci_config.VULKAN_BAZEL_TEST_TAG_FILTERS
            + (ci_config.AMD_GPU_RUN_REQUIREMENT_TAG,)
            + ci_config.AMD_GPU_WINDOWS_TEST_TAG_FILTERS
        )
        native_options = (
            ci_config.AMD_CLIENT_GPU_BAZEL_OPTIONS + amd_gpu_device_compiler_options()
        )
    additional_steps = []
    if sys.platform == "win32" and not targets:
        additional_steps.append(
            amd_gpu_windows_weights_step(options + native_options, config)
        )
    targets = targets or default_targets
    return [
        bazel_configure_step(extra_options=options),
        *platform_coverage_build_steps(
            bazel_build_step(
                f"Build IREE / AMD RDNA+XDNA{config_name}",
                targets,
                config=config,
                bazel_options=options + native_options,
            )
        ),
        bazel_test_step(
            f"Test IREE / AMD RDNA+XDNA{config_name}",
            targets,
            config=config,
            test_tag_filters=test_tag_filters,
            available_resources=resources,
            test_env=amd_test_env() + amd_gpu_cts_test_env("rdna"),
            bazel_options=options + native_options,
        ),
        *additional_steps,
    ]


def amdgpu_test_steps(
    targets: tuple[str, ...],
    target_selector: str,
    config: str | None = None,
    xfail_targets: tuple[str, ...] = (),
) -> list[CiStep]:
    config_name = f" / {config.upper()}" if config is not None else ""
    scoped_targets = targets + ci_config.AMDGPU_BAZEL_TARGET_EXCLUDES
    bazel_options = (
        ci_config.AMDGPU_BAZEL_OPTIONS
        + amd_gpu_device_compiler_options()
        + (
            # Compiler compatibility and hardware execution share one target
            # selector so other architectures are rejected during analysis.
            "--//loom/config/arch/amdgpu:targets=" + target_selector,
            "--//runtime/src/iree/hal/drivers/amdgpu:targets=" + target_selector,
            "--//runtime/src/iree/hal/drivers/amd/gpu:targets=" + target_selector,
            "--//runtime/config/hal:drivers=task,amd-gpu",
        )
    )
    host_sanitizer_tag_filters = (
        (f"-{ci_config.HOST_TSAN_INCOMPATIBLE_TEST_LABEL}",) if config == "tsan" else ()
    )
    return [
        *platform_coverage_build_steps(
            bazel_build_step(
                f"Build IREE / AMDGPU{config_name}",
                scoped_targets,
                config=config,
                bazel_options=bazel_options,
            )
        ),
        bazel_test_step(
            f"Test IREE / AMDGPU{config_name}",
            scoped_targets + xfail_targets,
            config=config,
            test_tag_filters=(
                ci_config.AMDGPU_BAZEL_TEST_TAG_FILTERS + host_sanitizer_tag_filters
            ),
            available_resources=ci_config.AMDGPU_RESOURCES,
            test_env=amd_test_env() + amd_gpu_cts_test_env(target_selector),
            bazel_options=bazel_options,
        ),
    ]


def amdgpu_windows_steps(
    targets: tuple[str, ...], target_selector: str, config: str | None
) -> list[CiStep]:
    options = (
        ci_config.AMDGPU_WINDOWS_BAZEL_OPTIONS
        + amd_gpu_device_compiler_options()
        + (
            "--//loom/config/arch/amdgpu:targets=" + target_selector,
            # Loom's IREE HAL representation compatibility reads the shared
            # AMDGPU selector even when native execution uses the AMD GPU HAL.
            "--//runtime/src/iree/hal/drivers/amdgpu:targets=" + target_selector,
            "--//runtime/src/iree/hal/drivers/amd/gpu:targets=" + target_selector,
        )
    )
    config_name = f" / {config.upper()}" if config is not None else ""
    # A required finite PM4 copy establishes the exact GPU and usable native
    # transport. SDMA remains capability-admitted; its skip is not a proof.
    queue_options = options + (
        f"--test_arg=--amdf_gpu_target={target_selector}",
        "--test_arg=--amdf_require_test=Pm4KernelQueueTest.ExecutesMaterializedCopyData",
        "--test_arg=--gtest_fail_fast=1",
    )
    steps = [
        bazel_configure_step(extra_options=ci_config.AMDGPU_WINDOWS_BAZEL_OPTIONS),
        bazel_test_step(
            "Query Windows GPU native queue capabilities",
            ("//libamdf/src/gpu/umd/wddm/wkmi:adapter_info_system_test",),
            config=config,
            available_resources=ci_config.AMDGPU_WINDOWS_RESOURCES,
            test_env=amd_test_env(),
            bazel_options=options
            + (
                f"--test_arg=--amdf_gpu_target={target_selector}",
                "--test_output=all",
            ),
        ),
        bazel_run_step(
            "Enumerate Windows GPU queue families",
            "//libamdf/examples:enumerate",
            bazel_options=options
            + ((f"--config={config}",) if config is not None else ()),
        ),
    ]
    for lifetime in ("process", "instance"):
        steps.append(
            bazel_test_step(
                f"Check Windows GPU finite copies / {lifetime}{config_name}",
                (ci_config.AMDGPU_WINDOWS_QUEUE_TARGET,),
                config=config,
                available_resources=ci_config.AMDGPU_WINDOWS_RESOURCES,
                test_env=amd_test_env(),
                bazel_options=queue_options
                + (
                    f"--test_arg=--amdf_native_lifetime={lifetime}",
                    "--test_filter=" + ci_config.AMDGPU_WINDOWS_COPY_FILTER,
                ),
            )
        )
    steps.append(
        bazel_test_step(
            f"Test Windows GPU coherent signals{config_name}",
            ("//libamdf/cts/gpu/pm4:pm4_dynamic",),
            config=config,
            available_resources=ci_config.AMDGPU_WINDOWS_RESOURCES,
            test_env=amd_test_env(),
            bazel_options=options
            + (
                f"--test_arg=--amdf_gpu_target={target_selector}",
                "--test_env=GTEST_FAIL_FAST=1",
                "--test_filter=*Pm4AtomicStoreTest.*:*Pm4WaitTest.*:"
                "Pm4UnconditionalWaitTest.*",
            ),
        )
    )
    steps.append(
        bazel_test_step(
            f"Test IREE / Windows AMDGPU{config_name}",
            targets or ci_config.AMDGPU_WINDOWS_BAZEL_TARGETS,
            config=config,
            test_tag_filters=ci_config.AMD_GPU_WINDOWS_TEST_TAG_FILTERS,
            available_resources=ci_config.AMDGPU_WINDOWS_RESOURCES,
            test_env=amd_test_env() + amd_gpu_cts_test_env(target_selector),
            bazel_options=options + ("--test_env=GTEST_FAIL_FAST=1",),
        )
    )
    steps.append(
        bazel_test_step(
            f"Test Windows GPU queues / instance{config_name}",
            (ci_config.AMDGPU_WINDOWS_QUEUE_TARGET,),
            config=config,
            available_resources=ci_config.AMDGPU_WINDOWS_RESOURCES,
            test_env=amd_test_env(),
            bazel_options=queue_options
            + ("--test_arg=--amdf_native_lifetime=instance",),
        )
    )
    if not targets:
        steps.append(amd_gpu_windows_weights_step(options, config))
    return steps


def amdgpu_steps(
    targets: tuple[str, ...],
    target_selector: str,
    enabled_loom_importers: tuple[str, ...] | None = None,
) -> list[CiStep]:
    return [
        bazel_configure_step(
            enabled_drivers=("amd-gpu",),
            enabled_loom_architectures=("amdgpu", "vm"),
            enabled_loom_artifact_formats=("elf", "vm"),
            enabled_loom_execution_substrates=("iree_hal", "iree_vm"),
            enabled_loom_importers=enabled_loom_importers,
        ),
        *amdgpu_test_steps(
            targets,
            target_selector,
            xfail_targets=(
                ci_config.AMDGPU_XFAIL_TARGETS
                + ci_config.amdgpu_bazel_xfail_targets(target_selector)
            ),
        ),
    ]


def loom_amdgpu_bazel_steps() -> list[CiStep]:
    return [
        bazel_configure_step(
            enabled_loom_architectures=("amdgpu", "vm"),
            enabled_loom_artifact_formats=("elf", "vm"),
            enabled_loom_execution_substrates=("iree_hal", "iree_vm"),
        ),
        bazel_test_step(
            "Test Loom AMDGPU compile coverage",
            ci_config.LOOM_AMDGPU_BAZEL_COMPILE_TEST_TARGETS,
            test_tag_filters=ci_config.CPU_RESOURCE_TAG_EXCLUDES,
        ),
    ]


def amdgpu_config_steps(
    targets: tuple[str, ...],
    target_selector: str,
    config: str,
) -> list[CiStep]:
    xfail_targets = (
        ci_config.AMDGPU_TSAN_SANITIZERS_XFAIL_TARGETS
        if config == "tsan"
        else ci_config.AMDGPU_SANITIZERS_XFAIL_TARGETS
    )
    return amdgpu_test_steps(
        targets,
        target_selector,
        config=config,
        xfail_targets=(
            xfail_targets + ci_config.amdgpu_bazel_xfail_targets(target_selector)
        ),
    )


def vulkan_steps(targets: tuple[str, ...]) -> list[CiStep]:
    scoped_targets = targets + ci_config.VULKAN_BAZEL_TARGET_EXCLUDES
    # A container that cannot create namespaces must fail instead of silently
    # downgrading to processwrapper-sandbox and hiding source-access problems.
    api_options = ("--//build_tools/vulkan/config:enabled=true",)
    bazel_options = api_options + (
        ("--spawn_strategy=linux-sandbox",) if sys.platform == "linux" else ()
    )
    return [
        bazel_configure_step(
            enabled_drivers=("vulkan",),
            enabled_loom_architectures=("spirv", "vm"),
            enabled_loom_artifact_formats=("spirv", "vm"),
            enabled_loom_execution_substrates=("iree_hal", "iree_vm"),
            extra_options=api_options,
        ),
        *platform_coverage_build_steps(
            bazel_build_step(
                "Build IREE / Vulkan",
                scoped_targets,
                bazel_options=bazel_options,
            )
        ),
        bazel_test_step(
            "Test IREE / Vulkan",
            scoped_targets + ci_config.VULKAN_XFAIL_TARGETS,
            test_tag_filters=ci_config.VULKAN_BAZEL_TEST_TAG_FILTERS,
            available_resources=ci_config.VULKAN_RESOURCES,
            test_env=vulkan_device_test_env(),
            bazel_options=bazel_options,
        ),
    ]


def cmake_cpu_steps(command_name: str, sanitizer: str | None) -> list[CiStep]:
    sanitizer_name = f" with {sanitizer.upper()}" if sanitizer is not None else ""
    tests_enabled = cmake_tests_enabled(sanitizer)
    xfail_regex = (
        ci_config.CPU_SANITIZERS_CTEST_EXCLUDE_REGEX
        if sanitizer is not None
        else ci_config.CPU_CTEST_EXCLUDE_REGEX
    )
    exclude_regex = combine_ctest_regex(
        xfail_regex,
        ci_config.NON_CPU_HAL_DRIVER_CTEST_REGEX,
    )
    steps = [
        cmake_configure_step(command_name, sanitizer=sanitizer),
        *cmake_explicit_build_steps(
            command_name,
            f"Build IREE CMake{sanitizer_name}",
            tests_enabled=tests_enabled,
        ),
    ]
    if tests_enabled:
        steps.append(
            cmake_test_step(
                command_name,
                f"Test IREE CMake{sanitizer_name}",
                exclude_regex=exclude_regex,
                env=sanitizer_env(sanitizer),
                label_exclude_regex=combine_ctest_regex(
                    ci_config.CTEST_RESOURCE_LABEL_EXCLUDE_REGEX,
                    ci_config.CTEST_MANUAL_LABEL_EXCLUDE_REGEX,
                    (
                        ci_config.PYTHON_TEST_CTEST_LABEL_REGEX
                        if sanitizer is not None
                        else ""
                    ),
                ),
            )
        )
    if sanitizer is None:
        # Reuse the static qualification tree so both final-link modes are
        # exercised without another compilation tree or worker job.
        steps.extend(
            [
                CiStep(
                    "Configure shared Loom tools",
                    cmake_dev_command(
                        command_name,
                        "configure",
                        "-DLOOM_LINK_SHARED_TOOLS=ON",
                        "-DLOOM_IMPORT_CXX=ON",
                    ),
                ),
                cmake_build_step(
                    command_name,
                    "Build shared Loom tools",
                    (
                        "loom-compile",
                        "loom-opt",
                        "loom-format",
                        "loom-link",
                        "loom-run",
                        "loom-import-cxx",
                        "loom-serve",
                        "loom-check",
                        "loom-check-test",
                        "loom-test",
                        "loom-benchmark",
                        "loom_tooling_target_cpu_serve_loom-serve-test-tool",
                    ),
                ),
                cmake_test_step(
                    command_name,
                    "Test shared Loom tools",
                    regex=(
                        r"^(loom/tools/(loom-compile|loom-opt|loom-format|loom-link|loom-import-cxx|loom-run|loom-serve|loom-check|loom-test|loom-benchmark)/(execution_test|scenario_execution_test)"
                        r"|loom/tools/loom-check/test/.*"
                        r"|loom/tooling/execution/hal/command_scenario_(execution|specialization)_test"
                        r"|loom/tooling/target/cpu/benchmark_scenario_correctness_test"
                        r"|loom/tooling/target/cpu/run_cxx_input_test"
                        r"|loom/tooling/target/cpu/serve/serve_command_test)$"
                    ),
                ),
            ]
        )
    return steps


def cmake_repository_build_steps(command_name: str) -> list[CiStep]:
    return [
        cmake_configure_step(
            command_name,
            enabled_drivers=REPOSITORY_BUILD_HAL_DRIVERS,
            enabled_loom_architectures=REPOSITORY_BUILD_LOOM_ARCHITECTURES,
            enabled_loom_artifact_formats=REPOSITORY_BUILD_LOOM_ARTIFACT_FORMATS,
            enabled_loom_execution_substrates=REPOSITORY_BUILD_LOOM_EXECUTION_SUBSTRATES,
            enabled_loom_importers=REPOSITORY_BUILD_LOOM_IMPORTERS,
            amdgpu_target_selector=None,
            amdgpu_device_binary_mode="prebuilt",
            extra_options=(
                "-DIREE_ENABLE_VULKAN=ON",
                "-DIREE_ENABLE_D3D12=ON",
            ),
        ),
        cmake_build_step(command_name, "Build repository"),
        cmake_test_step(
            command_name,
            "Test repository smoke",
            regex=combine_ctest_regex(*ci_config.CMAKE_REPOSITORY_SMOKE_CTEST_REGEXES),
            label_exclude_regex=combine_ctest_regex(
                ci_config.CTEST_RESOURCE_LABEL_EXCLUDE_REGEX,
                ci_config.CTEST_MANUAL_LABEL_EXCLUDE_REGEX,
            ),
            parallelism=2,
        ),
    ]


def cmake_xdna_steps(command_name: str, sanitizer: str | None) -> list[CiStep]:
    sanitizer_name = f" with {sanitizer.upper()}" if sanitizer is not None else ""
    tests_enabled = cmake_tests_enabled(sanitizer)
    package_regex = combine_ctest_regex(
        ci_config.AMDF_CTEST_REGEX, ci_config.XDNA_CTEST_PACKAGE_REGEX
    )
    steps = [
        cmake_configure_step(
            command_name,
            enabled_drivers=("amd-xdna",),
            enabled_loom_architectures=("aie2p", "vm"),
            enabled_loom_artifact_formats=("elf", "vm"),
            enabled_loom_execution_substrates=("iree_hal", "iree_vm"),
            sanitizer=sanitizer,
            extra_options=(
                "-DAMDF_FAMILY_RDNA=OFF",
                "-DAMDF_FAMILY_CDNA=OFF",
                "-DAMDF_FAMILY_XDNA=ON",
                "-DLOOM_BUILD=ON",
            ),
        ),
        *cmake_explicit_build_steps(
            command_name,
            f"Build IREE CMake XDNA{sanitizer_name}",
            ci_config.XDNA_CMAKE_BUILD_TARGETS,
            tests_enabled=tests_enabled,
        ),
    ]
    if not tests_enabled:
        return steps

    steps.extend(
        [
            cmake_test_step(
                command_name,
                f"Test IREE CMake XDNA package tests{sanitizer_name}",
                regex=package_regex,
                label_exclude_regex=combine_ctest_regex(
                    ci_config.CTEST_MANUAL_LABEL_EXCLUDE_REGEX,
                    ci_config.CTEST_RESOURCE_LABEL_EXCLUDE_REGEX,
                ),
                parallelism=1,
            ),
            cmake_test_step(
                command_name,
                f"Test IREE CMake XDNA resource tests{sanitizer_name}",
                label_regex=ci_config.XDNA_CTEST_RESOURCE_LABEL,
                available_resources=ci_config.XDNA_RESOURCES,
                label_exclude_regex=ci_config.CTEST_MANUAL_LABEL_EXCLUDE_REGEX,
                parallelism=1,
            ),
        ]
    )
    return steps


def cmake_amdgpu_steps(
    command_name: str, sanitizer: str | None, target_selector: str
) -> list[CiStep]:
    sanitizer_name = f" with {sanitizer.upper()}" if sanitizer is not None else ""
    tests_enabled = cmake_tests_enabled(sanitizer)
    if sanitizer == "tsan":
        xfail_regex = ci_config.AMDGPU_TSAN_SANITIZERS_CTEST_EXCLUDE_REGEX
    elif sanitizer is not None:
        xfail_regex = ci_config.AMDGPU_SANITIZERS_CTEST_EXCLUDE_REGEX
    else:
        xfail_regex = ci_config.AMDGPU_CTEST_EXCLUDE_REGEX
    build_targets = ci_config.AMDGPU_CMAKE_BUILD_TARGETS
    steps = [
        cmake_configure_step(
            command_name,
            enabled_drivers=("amd-gpu",),
            enabled_loom_architectures=("amdgpu", "vm"),
            enabled_loom_artifact_formats=("elf", "vm"),
            enabled_loom_execution_substrates=("iree_hal", "iree_vm"),
            sanitizer=sanitizer,
            extra_options=(
                "-DAMDF_BUILD=ON",
                "-DAMDF_FAMILY_RDNA=ON",
                "-DAMDF_FAMILY_CDNA=ON",
                "-DAMDF_FAMILY_XDNA=OFF",
                "-DLOOM_BUILD=ON",
                "-DIREE_HAL_AMDGPU_TARGETS=" + target_selector,
                "-DIREE_HAL_AMD_GPU_TARGETS=" + target_selector,
                "-DLOOM_ARCH_AMDGPU_TARGETS=" + target_selector,
            )
            + cmake_amdgpu_device_binary_options(),
        ),
        *cmake_explicit_build_steps(
            command_name,
            f"Build IREE CMake AMDGPU{sanitizer_name}",
            build_targets,
            tests_enabled=tests_enabled,
        ),
    ]
    if not tests_enabled:
        return steps

    steps.append(
        cmake_test_step(
            command_name,
            f"Test IREE CMake AMDGPU package tests{sanitizer_name}",
            regex=ci_config.AMDGPU_CMAKE_PACKAGE_REGEX,
            exclude_regex=xfail_regex,
            available_resources=ci_config.AMDGPU_RESOURCES,
            env=sanitizer_env(sanitizer)
            + amd_test_env()
            + amd_gpu_cts_test_env(target_selector),
            parallelism=1,
        )
    )
    resource_exclude_regex = combine_ctest_regex(
        ci_config.AMDGPU_CMAKE_PACKAGE_REGEX,
        xfail_regex,
    )
    resource_label_exclude_regex = ci_config.CTEST_MANUAL_LABEL_EXCLUDE_REGEX
    if sanitizer == "tsan":
        resource_label_exclude_regex = combine_ctest_regex(
            resource_label_exclude_regex, ci_config.HOST_TSAN_INCOMPATIBLE_TEST_LABEL
        )
    steps.append(
        cmake_test_step(
            command_name,
            f"Test IREE CMake AMDGPU resource tests{sanitizer_name}",
            label_regex=ci_config.AMDGPU_CTEST_RESOURCE_LABEL_REGEX,
            label_exclude_regex=resource_label_exclude_regex,
            exclude_regex=resource_exclude_regex,
            available_resources=ci_config.AMDGPU_RESOURCES,
            env=sanitizer_env(sanitizer)
            + amd_test_env()
            + amd_gpu_cts_test_env(target_selector),
            parallelism=1,
        )
    )
    return steps


def cmake_loom_amdgpu_steps(command_name: str) -> list[CiStep]:
    return [
        cmake_configure_step(
            command_name,
            enabled_loom_architectures=("amdgpu", "vm"),
            enabled_loom_artifact_formats=("elf", "vm"),
            enabled_loom_execution_substrates=("iree_hal", "iree_vm"),
        ),
        cmake_test_step(
            command_name,
            "Test Loom CMake AMDGPU compile coverage",
            regex=combine_ctest_regex(
                *ci_config.LOOM_AMDGPU_CMAKE_COMPILE_CTEST_REGEXES
            ),
            label_exclude_regex=ci_config.CTEST_RESOURCE_LABEL_EXCLUDE_REGEX,
        ),
    ]


def cmake_vulkan_steps(command_name: str, sanitizer: str | None) -> list[CiStep]:
    sanitizer_name = f" with {sanitizer.upper()}" if sanitizer is not None else ""
    tests_enabled = cmake_tests_enabled(sanitizer)
    build_targets = ci_config.VULKAN_CMAKE_DRIVER_TARGETS
    steps = [
        cmake_configure_step(
            command_name,
            enabled_drivers=("vulkan",),
            enabled_loom_architectures=("spirv", "vm"),
            enabled_loom_artifact_formats=("spirv", "vm"),
            enabled_loom_execution_substrates=("iree_hal", "iree_vm"),
            sanitizer=sanitizer,
            extra_options=("-DIREE_ENABLE_VULKAN=ON",),
        ),
        *cmake_explicit_build_steps(
            command_name,
            f"Build IREE CMake Vulkan{sanitizer_name}",
            build_targets,
            tests_enabled=tests_enabled,
        ),
    ]
    if tests_enabled:
        steps.append(
            cmake_test_step(
                command_name,
                f"Test IREE CMake Vulkan package tests{sanitizer_name}",
                regex=ci_config.VULKAN_CTEST_REGEX,
                available_resources=ci_config.VULKAN_RESOURCES,
                env=sanitizer_env(sanitizer) + vulkan_device_test_env(),
            )
        )
        steps.append(
            cmake_test_step(
                command_name,
                f"Test IREE CMake Vulkan resource tests{sanitizer_name}",
                label_regex=ci_config.VULKAN_CTEST_RESOURCE_LABEL_REGEX,
                label_exclude_regex=ci_config.CTEST_MANUAL_LABEL_EXCLUDE_REGEX,
                exclude_regex=ci_config.VULKAN_CTEST_REGEX,
                available_resources=ci_config.VULKAN_RESOURCES,
                env=sanitizer_env(sanitizer) + vulkan_device_test_env(),
            )
        )
    return steps


def cmake_target_steps(
    command_name: str,
    target_group: str,
    sanitizer: str | None,
    amdgpu_target_selector: str = ci_config.DEFAULT_AMDGPU_TARGET_SELECTOR,
) -> list[CiStep]:
    if target_group == "cpu":
        return cmake_cpu_steps(command_name, sanitizer)
    if target_group == "repository-build":
        if sanitizer is not None:
            raise ValueError("CMake repository builds do not support sanitizers")
        return cmake_repository_build_steps(command_name)
    if target_group == "amdgpu":
        return cmake_amdgpu_steps(command_name, sanitizer, amdgpu_target_selector)
    if target_group == "xdna":
        return cmake_xdna_steps(command_name, sanitizer)
    if target_group == "loom-amdgpu":
        if sanitizer is not None:
            raise ValueError("Loom AMDGPU CMake CI does not support sanitizers")
        return cmake_loom_amdgpu_steps(command_name)
    if target_group == "vulkan":
        return cmake_vulkan_steps(command_name, sanitizer)
    raise ValueError(f"unknown CMake CI target: {target_group}")


def cmake_sanitizer_steps(
    prefix: str,
    target_group: str,
    amdgpu_target_selector: str = ci_config.DEFAULT_AMDGPU_TARGET_SELECTOR,
) -> list[CiStep]:
    steps = []
    for config in ci_config.SANITIZER_TEST_CONFIGS:
        command_name = f"{prefix}-{config}"
        steps.extend(
            cmake_target_steps(
                command_name, target_group, config, amdgpu_target_selector
            )
        )
    return steps


def cmake_sanitizer_smoke_steps() -> list[CiStep]:
    steps = []
    test_regex = combine_ctest_regex(*ci_config.CMAKE_SANITIZER_SMOKE_CTEST_REGEXES)
    for config in ci_config.SANITIZER_TEST_CONFIGS:
        command_name = f"{CMAKE_SANITIZER_SMOKE_COMMAND}-{config}"
        steps.extend(
            [
                cmake_configure_step(
                    command_name,
                    sanitizer=config,
                    build_tests=True,
                ),
                cmake_test_step(
                    command_name,
                    f"Test IREE CMake sanitizer smoke with {config.upper()}",
                    regex=test_regex,
                    env=sanitizer_env(config),
                    parallelism=2,
                ),
            ]
        )
    return steps


def importer_steps(command_name: str, importer_name: str) -> list[CiStep]:
    if importer_name == "tilelang":
        return tilelang_importer_steps(command_name)
    raise ValueError(f"unknown importer CI target: {importer_name}")


def tilelang_importer_steps(command_name: str) -> list[CiStep]:
    ctest_regex = combine_ctest_regex(*ci_config.IMPORTER_TILELANG_CTEST_REGEXES)
    return [
        CiStep(
            "Setup TileLang importer environment",
            dev_command("importers", "setup", "tilelang"),
        ),
        CiStep(
            "Report TileLang importer environment",
            dev_command("importers", "env", "tilelang"),
        ),
        CiStep(
            "Test TileLang importer with Bazel",
            dev_command(
                "bazel",
                "test",
                "--importer-env",
                "tilelang",
                *codec_options("bazel"),
                *ci_config.IMPORTER_TILELANG_BAZEL_TEST_TARGETS,
            ),
        ),
        CiStep(
            "Configure TileLang importer CMake",
            cmake_dev_command(
                command_name,
                "configure",
                "--fresh",
                "-GNinja",
                "--importer-env",
                "tilelang",
                "-DIREE_BUILD_TESTS=ON",
                "-DLIBHRX_BUILD=OFF",
                *codec_options("cmake"),
            ),
        ),
        CiStep(
            "Build TileLang importer CMake verifier",
            cmake_dev_command(
                command_name,
                "build",
                "loom-opt",
                "--parallel",
            ),
        ),
        CiStep(
            "Test TileLang importer with CMake",
            cmake_dev_command(
                command_name,
                "test",
                "--importer-env",
                "tilelang",
                "--parallel",
                "8",
                "--no-tests=error",
                "-R",
                ctest_regex,
            ),
        ),
    ]


def _steps_from_args(args: argparse.Namespace) -> list[CiStep]:
    bazel_target_group = BAZEL_COMMANDS.get(args.command, (None, None))[0]
    cmake_target_group = CMAKE_COMMANDS.get(args.command, (None, None))[0]
    amdgpu_target_bazel_groups = (
        "amdgpu",
        "amdgpu-native",
        "repository-integration",
    )
    accepts_amdgpu_target = (
        bazel_target_group in amdgpu_target_bazel_groups
        or cmake_target_group == "amdgpu"
    )
    if args.amdgpu_target is not None and not accepts_amdgpu_target:
        raise ValueError(
            "--amdgpu-target is only supported for AMDGPU and "
            "repository-integration CI commands"
        )
    if args.loom_importer is not None and bazel_target_group != "amdgpu":
        raise ValueError(
            "--loom-importer is only supported for AMDGPU Bazel CI commands"
        )
    amdgpu_target_selector = (
        args.amdgpu_target or ci_config.DEFAULT_AMDGPU_TARGET_SELECTOR
    )
    enabled_loom_importers = (
        tuple(args.loom_importer) if args.loom_importer is not None else None
    )

    if args.command == CMAKE_SANITIZER_SMOKE_COMMAND:
        if args.target:
            raise ValueError("--target is only supported for Bazel CI commands")
        return cmake_sanitizer_smoke_steps()

    if args.command in IMPORTER_COMMANDS:
        if args.target:
            raise ValueError("--target is not supported for importer CI commands")
        return importer_steps(args.command, IMPORTER_COMMANDS[args.command])

    if args.command in CMAKE_COMMANDS:
        if args.target:
            raise ValueError("--target is only supported for Bazel CI commands")
        target_group, sanitizer = CMAKE_COMMANDS[args.command]
        if sanitizer == "all":
            prefix = args.command.removesuffix("-sanitizers")
            return cmake_sanitizer_steps(prefix, target_group, amdgpu_target_selector)
        return cmake_target_steps(
            args.command, target_group, sanitizer, amdgpu_target_selector
        )

    bazel_target, sanitizer = BAZEL_COMMANDS[args.command]
    if bazel_target == "amd-client":
        return amd_client_steps(
            tuple(args.target or ()),
            sanitizer,
        )
    if bazel_target == "amdgpu-native":
        if sys.platform != "win32":
            raise ValueError("Native kernel-queue CI requires Windows")
        if args.amdgpu_target is None:
            raise ValueError("Windows GPU CI requires an exact --amdgpu-target")
        return amdgpu_windows_steps(
            tuple(args.target or ()), amdgpu_target_selector, sanitizer
        )
    if bazel_target == "xdna":
        return xdna_steps(
            tuple(args.target) if args.target else ci_config.XDNA_BAZEL_TARGETS,
            sanitizer,
        )
    if bazel_target == "loom-amdgpu":
        if args.target:
            raise ValueError("--target is not supported by Loom AMDGPU CI")
        if sanitizer is not None:
            raise ValueError("Loom AMDGPU Bazel CI does not support sanitizers")
        return loom_amdgpu_bazel_steps()
    if bazel_target == "repository-build":
        if args.target:
            raise ValueError(
                "--target is not supported by the repository-wide build command"
            )
        return repository_build_steps()
    if bazel_target == "repository-integration":
        if args.target:
            raise ValueError(
                "--target is not supported by the repository-wide integration command"
            )
        return repository_integration_steps(amdgpu_target_selector)
    targets = command_targets(args.target)
    if bazel_target == "cpu":
        if sanitizer == "all":
            return cpu_sanitizer_steps(targets)
        if sanitizer is not None:
            return [bazel_configure_step(), *cpu_config_steps(targets, sanitizer)]
        return cpu_steps(targets)
    if bazel_target == "amdgpu":
        if not args.target:
            targets += ci_config.AMDF_BAZEL_TARGETS
        if sanitizer is not None:
            return [
                bazel_configure_step(
                    enabled_drivers=("amd-gpu",),
                    enabled_loom_architectures=("amdgpu", "vm"),
                    enabled_loom_artifact_formats=("elf", "vm"),
                    enabled_loom_execution_substrates=("iree_hal", "iree_vm"),
                    enabled_loom_importers=enabled_loom_importers,
                ),
                *amdgpu_config_steps(
                    targets,
                    amdgpu_target_selector,
                    sanitizer,
                ),
            ]
        return amdgpu_steps(
            targets,
            amdgpu_target_selector,
            enabled_loom_importers=enabled_loom_importers,
        )
    if bazel_target == "vulkan":
        return vulkan_steps(targets)
    raise ValueError(f"unknown Bazel CI target: {bazel_target}")


def add_bazel_profiles(steps: list[CiStep], profile_dir: Path) -> list[CiStep]:
    profiled_steps = []
    profile_owners = {}
    for step in steps:
        if step.argv[1:4] not in (
            ("dev.py", "bazel", "build"),
            ("dev.py", "bazel", "run"),
            ("dev.py", "bazel", "test"),
        ):
            profiled_steps.append(step)
            continue

        if any(
            arg == "--profile" or arg.startswith("--profile=") for arg in step.argv[4:]
        ):
            raise ValueError(f"{step.name} already configures a Bazel profile")
        profile_stem = "-".join(
            token.lower() for token in re.findall(r"[A-Za-z0-9]+", step.name)
        )
        if not profile_stem:
            raise ValueError(f"cannot derive a Bazel profile name from {step.name!r}")
        profile_path = profile_dir / f"{profile_stem}.profile.gz"
        if previous_owner := profile_owners.get(profile_path):
            raise ValueError(
                f"Bazel profile path collision between {previous_owner!r} and "
                f"{step.name!r}: {profile_path}"
            )
        profile_owners[profile_path] = step.name
        profiled_steps.append(
            replace(
                step,
                argv=step.argv[:4] + (f"--profile={profile_path}",) + step.argv[4:],
                bazel_profile=profile_path,
            )
        )
    return profiled_steps


def add_bazel_test_results(steps: list[CiStep], output_dir: Path) -> list[CiStep]:
    captured_steps = []
    event_owners = {}
    for step in steps:
        if step.argv[1:4] != ("dev.py", "bazel", "test"):
            captured_steps.append(step)
            continue

        if any(
            arg == "--build_event_json_file"
            or arg.startswith("--build_event_json_file=")
            for arg in step.argv[4:]
        ):
            raise ValueError(f"{step.name} already configures a Bazel build event file")
        event_stem = "-".join(
            token.lower() for token in re.findall(r"[A-Za-z0-9]+", step.name)
        )
        if not event_stem:
            raise ValueError(
                f"cannot derive a Bazel build event name from {step.name!r}"
            )
        event_path = output_dir / ".events" / f"{event_stem}.json"
        if previous_owner := event_owners.get(event_path):
            raise ValueError(
                f"Bazel build event path collision between {previous_owner!r} and "
                f"{step.name!r}: {event_path}"
            )
        event_owners[event_path] = step.name
        captured_steps.append(
            replace(
                step,
                argv=step.argv[:4]
                + (f"--build_event_json_file={event_path}",)
                + step.argv[4:],
                test_results=bazel_test_results.BazelTestResultsCapture(
                    build_event_path=event_path,
                    output_dir=output_dir / event_stem,
                ),
            )
        )
    if not event_owners:
        raise ValueError("--bazel-test-results-dir requires a Bazel test phase")
    return captured_steps


def bazel_profile_dir(args: argparse.Namespace) -> Path | None:
    if args.bazel_profile_dir is None:
        return None
    return args.bazel_profile_dir.expanduser().resolve()


def bazel_test_results_dir(args: argparse.Namespace) -> Path | None:
    if args.bazel_test_results_dir is None:
        return None
    return args.bazel_test_results_dir.expanduser().resolve()


def steps_from_args(args: argparse.Namespace) -> list[CiStep]:
    profile_dir = bazel_profile_dir(args)
    test_results_dir = bazel_test_results_dir(args)
    if profile_dir is not None and args.command not in BAZEL_COMMANDS:
        raise ValueError("--bazel-profile-dir is only supported for Bazel CI commands")
    if test_results_dir is not None and args.command not in BAZEL_COMMANDS:
        raise ValueError(
            "--bazel-test-results-dir is only supported for Bazel CI commands"
        )
    steps = _steps_from_args(args)
    if args.keep_going:
        # Continue independent actions inside each phase as well as subsequent
        # phases. This lets actions finish normal output-stream cleanup before
        # the next phase reuses Bazel's action log directory.
        for index, step in enumerate(steps):
            if step.argv[1:4] in (
                ("dev.py", "bazel", "build"),
                ("dev.py", "bazel", "test"),
                ("dev.py", "bazel", "run"),
            ):
                steps[index] = replace(
                    step, argv=step.argv[:4] + ("--keep_going",) + step.argv[4:]
                )
            elif step.argv[1:3] == ("dev.py", "--cmake-build-dir") and step.argv[
                4:6
            ] == ("cmake", "build"):
                # CI CMake configurations use Ninja. Pass native build options
                # after CMake's argument separator.
                steps[index] = replace(step, argv=step.argv + ("--", "-k", "0"))
    if profile_dir is not None:
        steps = add_bazel_profiles(steps, profile_dir)
    if test_results_dir is not None:
        steps = add_bazel_test_results(steps, test_results_dir)
    return steps


def github_actions_enabled() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def print_group_start(name: str) -> None:
    if github_actions_enabled():
        print(f"::group::{name}")


def print_group_end() -> None:
    if github_actions_enabled():
        print("::endgroup::")


def audit_requirements(
    audit: RequirementAudit, environment: dict[str, str] | None
) -> int:
    completed = subprocess.run(
        audit.argv,
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        text=True,
    )
    if completed.returncode:
        return completed.returncode
    try:
        audit.validate(completed.stdout)
    except (ValueError, ET.ParseError, KeyError, TypeError) as error:
        print(f"Invalid test run requirements: {error}", file=sys.stderr)
        return 1
    return 0


def publish_bazel_profile(step: CiStep) -> None:
    """Publishes the cost summary for a completed profiled Bazel phase."""
    if step.bazel_profile is None:
        return
    summary = bazel_profile_summary.summarize(step.bazel_profile)
    output_path = bazel_profile_summary.summary_path(step.bazel_profile)
    bazel_profile_summary.write(summary, output_path)
    print(bazel_profile_summary.render_console(summary, step.name), flush=True)
    github_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if github_summary_path:
        with Path(github_summary_path).open("a", encoding="utf-8") as stream:
            stream.write(bazel_profile_summary.render_markdown(summary, step.name))
            stream.write("\n")


def run_step(step: CiStep, verbose: bool) -> StepResult:
    print(f"[run] {step.name}", flush=True)
    if verbose or github_actions_enabled():
        print("  " + step.command_line(), flush=True)
    start_time = time.monotonic()
    if step.env:
        environment = os.environ.copy()
        for key, value in step.env:
            if key == "TSAN_OPTIONS" and environment.get(key):
                environment[key] = value + " " + environment[key]
            else:
                environment[key] = value
    else:
        environment = None
    if step.requirement_audit is not None:
        print("[check] Declared test run requirements", flush=True)
        if verbose or github_actions_enabled():
            print("  " + shlex.join(step.requirement_audit.argv), flush=True)
        audit_result = audit_requirements(step.requirement_audit, environment)
        if audit_result:
            print(f"[fail] {step.name}: run-requirement audit", flush=True)
            return StepResult(step, audit_result)
    artifact_dir = os.environ.get(windows_diagnostics.ARTIFACT_DIR_ENV)
    if artifact_dir:
        build_dir = None
        if "--cmake-build-dir" in step.argv:
            build_dir = REPO_ROOT / step.argv[step.argv.index("--cmake-build-dir") + 1]
        returncode = windows_diagnostics.run(
            step.argv,
            cwd=REPO_ROOT,
            env=environment,
            artifact_dir=Path(artifact_dir),
            label=step.name,
            build_dir=build_dir,
        )
    else:
        returncode = subprocess.run(
            step.argv, cwd=REPO_ROOT, env=environment
        ).returncode
    if step.bazel_profile is not None:
        try:
            publish_bazel_profile(step)
        except (bazel_profile_summary.BazelProfileSummaryError, OSError) as error:
            print(f"[fail] Summarize Bazel profile: {error}", file=sys.stderr)
            if returncode == 0:
                returncode = 1
    if step.test_results is not None:
        try:
            output_count = bazel_test_results.stage(
                step.test_results.build_event_path,
                step.test_results.output_dir,
            )
            print(
                f"[artifact] Staged {output_count} current Bazel test outputs",
                flush=True,
            )
        except bazel_test_results.BazelTestResultsError as error:
            print(f"[fail] Stage Bazel test results: {error}", file=sys.stderr)
            if returncode == 0:
                returncode = 1
    elapsed_seconds = time.monotonic() - start_time
    result = StepResult(step, returncode)
    if result.ok:
        print(f"[ok] {step.name} ({elapsed_seconds:.1f}s)", flush=True)
    else:
        print(
            f"[fail] {step.name} ({elapsed_seconds:.1f}s, exit {returncode})",
            flush=True,
        )
        print("  " + step.command_line(), flush=True)
    return result


def run_steps(
    steps: list[CiStep],
    *,
    dry_run: bool,
    keep_going: bool,
    verbose: bool,
) -> int:
    if dry_run:
        for step in steps:
            print(step.command_line())
        return 0

    print("== IREE CI ==", flush=True)
    failures = []
    for step in steps:
        print_group_start(step.name)
        try:
            result = run_step(step, verbose=verbose)
        finally:
            print_group_end()
        if not result.ok:
            if not keep_going:
                return result.returncode
            failures.append(result)

    if failures:
        print("", flush=True)
        print("Failed phases:", flush=True)
        for failure in failures:
            print(f"  {failure.step.name}: {failure.returncode}", flush=True)
        return failures[-1].returncode
    return 0


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source-tree IREE CI command groups locally.",
    )
    parser.add_argument(
        "command",
        choices=(
            *BAZEL_COMMANDS,
            *CMAKE_COMMANDS,
            CMAKE_SANITIZER_SMOKE_COMMAND,
            *IMPORTER_COMMANDS,
        ),
        help="CI command group to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the underlying dev.py commands without executing them.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help=(
            "Run every phase, let build tools finish independent actions after failures, "
            "and report failures at the end."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each underlying command before running it.",
    )
    parser.add_argument(
        "--amdgpu-target",
        help=(
            "Exact AMDGPU target or family selector for AMDGPU and "
            "repository-integration CI commands. "
            f"Defaults to {ci_config.DEFAULT_AMDGPU_TARGET_SELECTOR}."
        ),
    )
    parser.add_argument(
        "--loom-importer",
        action="append",
        choices=REPOSITORY_BUILD_LOOM_IMPORTERS,
        help=(
            "Loom importer to enable for AMDGPU Bazel CI commands. May be "
            "specified more than once."
        ),
    )
    parser.add_argument(
        "--bazel-profile-dir",
        type=Path,
        help=(
            "Write one compressed Bazel execution profile per build/test/run phase "
            "to this directory. Only supported for Bazel CI commands."
        ),
    )
    parser.add_argument(
        "--bazel-test-results-dir",
        type=Path,
        help=(
            "Stage test.log and test.xml outputs named by the current Bazel "
            "invocation into this fresh directory. Only supported for Bazel "
            "CI commands."
        ),
    )
    parser.add_argument(
        "--target",
        action="append",
        help=(
            "Bazel target pattern to build/test. Defaults to the IREE target "
            "directories present in the checkout."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    steps = steps_from_args(args)
    profile_dir = bazel_profile_dir(args)
    test_results_dir = bazel_test_results_dir(args)
    if profile_dir is not None and not args.dry_run:
        profile_dir.mkdir(parents=True, exist_ok=True)
    if test_results_dir is not None and not args.dry_run:
        bazel_test_results.prepare_output_directory(test_results_dir)
    return run_steps(
        steps,
        dry_run=args.dry_run,
        keep_going=args.keep_going,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
