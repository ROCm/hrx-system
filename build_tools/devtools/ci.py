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
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CMAKE_CI_BUILD_ROOT = Path("build/ci")
TSAN_SUPPRESSIONS_FILE = REPO_ROOT / "build_tools/devtools/tsan.supp"
CMAKE_SANITIZER_OPTIONS = {
    "asan": ("-DIREE_ENABLE_ASAN=ON",),
    "msan": ("-DIREE_ENABLE_MSAN=ON",),
    "tsan": ("-DIREE_ENABLE_TSAN=ON",),
    "ubsan": ("-DIREE_ENABLE_UBSAN=ON",),
}
BAZEL_HAL_DRIVER_DEFINES = (
    ("amdgpu", "IREE_HAL_DRIVER_AMDGPU"),
    ("local-task", "IREE_HAL_DRIVER_LOCAL_TASK"),
    ("vulkan", "IREE_HAL_DRIVER_VULKAN"),
    ("webgpu", "IREE_HAL_DRIVER_WEBGPU"),
)
CMAKE_HAL_DRIVER_DEFINES = (
    ("amdgpu", "IREE_HAL_DRIVER_AMDGPU"),
    ("vulkan", "IREE_HAL_DRIVER_VULKAN"),
    ("webgpu", "IREE_HAL_DRIVER_WEBGPU"),
)
CMAKE_LOOM_TARGET_DEFINES = (
    ("amdgpu", "LOOM_TARGET_AMDGPU"),
    ("iree_vm", "LOOM_TARGET_IREE_VM"),
    ("llvmir", "LOOM_TARGET_LLVMIR"),
    ("spirv", "LOOM_TARGET_SPIRV"),
    ("wasm", "LOOM_TARGET_WASM"),
    ("x86", "LOOM_TARGET_X86"),
)
CMAKE_LOOM_IMPORTER_DEFINES = (
    ("mlir", "LOOM_IMPORT_MLIR"),
    ("tilelang", "LOOM_IMPORT_TILELANG"),
)
CI_SUPPORTED_HAL_DRIVERS = frozenset(driver for driver, _ in BAZEL_HAL_DRIVER_DEFINES)
REPOSITORY_BUILD_HAL_DRIVERS = (
    "amdgpu",
    "local-task",
    "vulkan",
    "webgpu",
)
REPOSITORY_BUILD_LOOM_TARGETS = (
    "amdgpu",
    "iree_vm",
    "llvmir",
    "spirv",
    "wasm",
    "x86",
)
REPOSITORY_BUILD_LOOM_IMPORTERS = ("mlir", "tilelang")
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
    "iree-bazel-loom-amdgpu": ("loom-amdgpu", None),
    "iree-bazel-vulkan": ("vulkan", None),
}
CMAKE_COMMANDS = {
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
    sys.path.insert(0, str(REPO_ROOT))
    from build_tools.devtools import ci_config


@dataclass(frozen=True)
class CiStep:
    name: str
    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()

    def command_line(self) -> str:
        env_args = tuple(f"{key}={value}" for key, value in self.env)
        return shlex.join(env_args + self.argv)


@dataclass(frozen=True)
class StepResult:
    step: CiStep
    returncode: int
    elapsed_seconds: float

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


def python_command(path: str, *args: str) -> tuple[str, ...]:
    return (os.environ.get("IREE_CI_PYTHON", "python3"), path, *args)


def sanitizer_env(config: str | None) -> tuple[tuple[str, str], ...]:
    if config != "tsan":
        return ()
    return (("TSAN_OPTIONS", f"suppressions={TSAN_SUPPRESSIONS_FILE}"),)


def amdgpu_libhsa_test_env() -> tuple[tuple[str, str], ...]:
    libhsa_path = os.environ.get("IREE_HAL_AMDGPU_LIBHSA_PATH")
    if not libhsa_path:
        rocm_root = os.environ.get("HRX_ROCM_ROOT")
        if not rocm_root:
            return ()
        relative_path = (
            Path("bin/hsa-runtime64.dll")
            if sys.platform == "win32"
            else Path("lib/libhsa-runtime64.so.1")
        )
        libhsa_path = str(Path(rocm_root) / relative_path)
    return (("IREE_HAL_AMDGPU_LIBHSA_PATH", libhsa_path),)


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
    return enabled_driver_set


def amdgpu_bazel_options(target_selector: str) -> tuple[str, ...]:
    return (
        "--//runtime/src/iree/hal/drivers/amdgpu:targets=" + target_selector,
        "--//loom/config/target/amdgpu:targets=iree_hal",
    )


def bazel_configure_step(
    enabled_drivers: tuple[str, ...] = (),
    *,
    enabled_loom_targets: tuple[str, ...] | None = None,
    enabled_loom_importers: tuple[str, ...] | None = None,
) -> CiStep:
    enabled_driver_set = validate_enabled_drivers(enabled_drivers)
    command = ["bazel", "configure"]
    for driver, define in BAZEL_HAL_DRIVER_DEFINES:
        if driver in enabled_driver_set:
            command.append(f"-D{define}=ON")
    if "amdgpu" in enabled_driver_set:
        command.append(ROCM_PINNED_DEPENDENCY_MODE_OPTION)
        rocm_root = os.environ.get("HRX_ROCM_ROOT")
        if rocm_root:
            command.append(f"-DIREE_ROCM_PATH={rocm_root}")
    if enabled_loom_targets is not None:
        command.append(
            "--//loom/config/target:enable=" + ",".join(enabled_loom_targets)
        )
    if enabled_loom_importers is not None:
        command.append(
            "--//loom/config/import:enable=" + ",".join(enabled_loom_importers)
        )
    return CiStep("Configure Bazel", dev_command(*command))


def bazel_build_step(
    name: str,
    targets: tuple[str, ...],
    config: str | None = None,
    bazel_options: tuple[str, ...] = (),
) -> CiStep:
    command = ["bazel", "build"]
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
    test_env: tuple[tuple[str, str], ...] = (),
    bazel_options: tuple[str, ...] = (),
) -> CiStep:
    options = []
    if config is not None:
        options.append(f"--config={config}")
    if test_tag_filters:
        options.append("--test_tag_filters=" + ",".join(test_tag_filters))
    for key, value in sanitizer_env(config) + test_env:
        options.append(f"--test_env={key}={value}")
    options.extend(bazel_options)
    command = ["bazel", "test", *options]
    if any(target.startswith("-") for target in targets):
        command.append("--")
    command.extend(targets)
    return CiStep(name, dev_command(*command))


def bazel_run_step(
    name: str,
    target: str,
    program_args: tuple[str, ...] = (),
    bazel_options: tuple[str, ...] = (),
) -> CiStep:
    command = ["bazel", "run", *bazel_options, target]
    if program_args:
        command.extend(("--", *program_args))
    return CiStep(name, dev_command(*command))


def cmake_configure_step(
    command_name: str,
    *,
    enabled_drivers: tuple[str, ...] = (),
    enabled_loom_targets: tuple[str, ...] | None = None,
    enabled_loom_importers: tuple[str, ...] | None = None,
    amdgpu_target_selector: str | None = ci_config.DEFAULT_AMDGPU_TARGET_SELECTOR,
    amdgpu_device_binary_mode: str = "source",
    sanitizer: str | None = None,
    build_tests: bool | None = None,
) -> CiStep:
    enabled_driver_set = validate_enabled_drivers(enabled_drivers)
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
    ]
    for driver, define in CMAKE_HAL_DRIVER_DEFINES:
        command.append(f"-D{define}={'ON' if driver in enabled_driver_set else 'OFF'}")
    if enabled_loom_targets is not None:
        enabled_loom_target_set = frozenset(enabled_loom_targets)
        for target, define in CMAKE_LOOM_TARGET_DEFINES:
            command.append(
                f"-D{define}={'ON' if target in enabled_loom_target_set else 'OFF'}"
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
            command.append("-DLOOM_TARGET_AMDGPU_TARGETS=iree_hal")
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
    return CiStep("Configure CMake", cmake_dev_command(command_name, *command))


def cmake_build_step(
    command_name: str,
    name: str,
    targets: tuple[str, ...] = (),
) -> CiStep:
    return CiStep(
        name,
        cmake_dev_command(command_name, "build", *targets, "--parallel"),
    )


def cmake_runtime_resource_build_target(resource_label: str) -> str:
    prefix = ci_config.RUNTIME_CTEST_RESOURCE_LABEL_PREFIX
    if not resource_label.startswith(prefix):
        raise ValueError(f"expected CTest runtime resource label: {resource_label}")
    resource_name = resource_label.removeprefix(prefix)
    resource_target_suffix = "".join(
        c if c.isalnum() or c in "_.+-" else "-" for c in resource_name
    )
    return "iree-test-resource-" + resource_target_suffix


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
    return CiStep(name, cmake_dev_command(command_name, *command), env=env)


def cpu_steps(targets: tuple[str, ...]) -> list[CiStep]:
    scoped_targets = targets + ci_config.CPU_BAZEL_TARGET_EXCLUDES
    return [
        bazel_configure_step(),
        bazel_build_step("Build IREE", scoped_targets),
        bazel_test_step(
            "Test IREE",
            scoped_targets + ci_config.CPU_XFAIL_TARGETS,
            test_tag_filters=ci_config.CPU_RESOURCE_TAG_EXCLUDES,
        ),
    ]


def repository_build_steps() -> list[CiStep]:
    return [
        bazel_configure_step(
            enabled_drivers=REPOSITORY_BUILD_HAL_DRIVERS,
            enabled_loom_targets=REPOSITORY_BUILD_LOOM_TARGETS,
            enabled_loom_importers=REPOSITORY_BUILD_LOOM_IMPORTERS,
        ),
        bazel_build_step("Build repository", ("//...",)),
    ]


def repository_integration_steps(amdgpu_target_selector: str) -> list[CiStep]:
    if not os.environ.get("HRX_ROCM_ROOT"):
        raise ValueError(
            "iree-bazel-repository-integration requires HRX_ROCM_ROOT so the "
            "AMDGPU device compiler is exercised"
        )
    amdgpu_options = amdgpu_bazel_options(amdgpu_target_selector)
    return [
        *repository_build_steps(),
        bazel_build_step(
            "Build AMDGPU device toolchain smoke",
            ci_config.BAZEL_REPOSITORY_INTEGRATION_DEVICE_TARGETS,
            bazel_options=amdgpu_options,
        ),
        bazel_test_step(
            "Test repository",
            ci_config.BAZEL_REPOSITORY_TEST_TARGETS,
            test_tag_filters=ci_config.CPU_RESOURCE_TAG_EXCLUDES,
        ),
        CiStep(
            "Test lock-free Bazel launch",
            python_command("build_tools/devtools/bazel_launcher_integration_test.py"),
        ),
        bazel_run_step(
            "Run dynamic library environment smoke",
            ci_config.BAZEL_REPOSITORY_INTEGRATION_DYNAMIC_LIBRARY_TARGET,
        ),
        bazel_run_step(
            "Run executable alias smoke",
            ci_config.BAZEL_REPOSITORY_INTEGRATION_ALIAS_TARGET,
            program_args=("--help",),
        ),
    ]


def cpu_sanitizer_steps(targets: tuple[str, ...]) -> list[CiStep]:
    steps = [bazel_configure_step()]
    for config in ci_config.SANITIZER_TEST_CONFIGS:
        steps.extend(cpu_config_steps(targets, config))
    for config in ci_config.SANITIZER_BUILD_CONFIGS:
        steps.extend(cpu_config_steps(targets, config))
    return steps


def cpu_config_steps(targets: tuple[str, ...], config: str) -> list[CiStep]:
    scoped_targets = targets + ci_config.CPU_BAZEL_TARGET_EXCLUDES
    if config in ci_config.SANITIZER_TEST_CONFIGS:
        return [
            bazel_test_step(
                f"Test IREE with {config.upper()}",
                scoped_targets + ci_config.CPU_SANITIZERS_XFAIL_TARGETS,
                config=config,
                test_tag_filters=ci_config.CPU_RESOURCE_TAG_EXCLUDES,
            )
        ]
    if config in ci_config.SANITIZER_BUILD_CONFIGS:
        return [
            bazel_build_step(
                f"Build IREE with {config.upper()}",
                scoped_targets,
                config=config,
            )
        ]
    raise ValueError(f"unknown Bazel sanitizer config: {config}")


def amdgpu_build_and_test_steps(
    targets: tuple[str, ...],
    target_selector: str,
    config: str | None = None,
    xfail_targets: tuple[str, ...] = (),
) -> list[CiStep]:
    config_name = f" / {config.upper()}" if config is not None else ""
    scoped_targets = targets + ci_config.AMDGPU_BAZEL_TARGET_EXCLUDES
    bazel_options = amdgpu_bazel_options(target_selector)
    host_sanitizer_tag_filters = (
        (f"-{ci_config.HOST_TSAN_INCOMPATIBLE_TEST_LABEL}",) if config == "tsan" else ()
    )
    return [
        bazel_build_step(
            f"Build IREE / AMDGPU{config_name}",
            scoped_targets,
            config=config,
            bazel_options=bazel_options,
        ),
        bazel_test_step(
            f"Test IREE / AMDGPU{config_name}",
            scoped_targets + xfail_targets,
            config=config,
            test_tag_filters=(
                ci_config.AMDGPU_BAZEL_TEST_TAG_FILTERS + host_sanitizer_tag_filters
            ),
            test_env=amdgpu_libhsa_test_env(),
            bazel_options=bazel_options,
        ),
    ]


def amdgpu_steps(targets: tuple[str, ...], target_selector: str) -> list[CiStep]:
    return [
        bazel_configure_step(enabled_drivers=("amdgpu",)),
        *amdgpu_build_and_test_steps(
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
        bazel_configure_step(),
        bazel_test_step(
            "Test Loom AMDGPU compile coverage",
            ci_config.LOOM_AMDGPU_BAZEL_COMPILE_TEST_TARGETS,
            test_tag_filters=ci_config.CPU_RESOURCE_TAG_EXCLUDES,
        ),
    ]


def amdgpu_config_steps(
    targets: tuple[str, ...], target_selector: str, config: str
) -> list[CiStep]:
    xfail_targets = (
        ci_config.AMDGPU_TSAN_SANITIZERS_XFAIL_TARGETS
        if config == "tsan"
        else ci_config.AMDGPU_SANITIZERS_XFAIL_TARGETS
    )
    return amdgpu_build_and_test_steps(
        targets,
        target_selector,
        config=config,
        xfail_targets=(
            xfail_targets + ci_config.amdgpu_bazel_xfail_targets(target_selector)
        ),
    )


def vulkan_steps(targets: tuple[str, ...]) -> list[CiStep]:
    scoped_targets = targets + ci_config.VULKAN_BAZEL_TARGET_EXCLUDES
    return [
        bazel_configure_step(enabled_drivers=("vulkan",)),
        bazel_build_step(
            "Build IREE / Vulkan",
            scoped_targets,
        ),
        bazel_test_step(
            "Test IREE / Vulkan",
            scoped_targets + ci_config.VULKAN_XFAIL_TARGETS,
            test_tag_filters=ci_config.VULKAN_BAZEL_TEST_TAG_FILTERS,
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
        cmake_build_step(command_name, f"Build IREE CMake{sanitizer_name}"),
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
                ),
            )
        )
    return steps


def cmake_repository_build_steps(command_name: str) -> list[CiStep]:
    return [
        cmake_configure_step(
            command_name,
            enabled_drivers=REPOSITORY_BUILD_HAL_DRIVERS,
            enabled_loom_targets=REPOSITORY_BUILD_LOOM_TARGETS,
            enabled_loom_importers=REPOSITORY_BUILD_LOOM_IMPORTERS,
            amdgpu_target_selector=None,
            amdgpu_device_binary_mode="prebuilt",
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
    build_targets = ci_config.AMDGPU_CMAKE_DRIVER_TARGETS
    if tests_enabled:
        build_targets += (
            cmake_runtime_resource_build_target(
                ci_config.AMDGPU_CTEST_RESOURCE_LABEL_REGEX
            ),
        )
    steps = [
        cmake_configure_step(
            command_name,
            enabled_drivers=("amdgpu",),
            amdgpu_target_selector=target_selector,
            sanitizer=sanitizer,
        ),
        cmake_build_step(
            command_name,
            f"Build IREE CMake AMDGPU{sanitizer_name}",
            build_targets,
        ),
    ]
    if not tests_enabled:
        return steps

    steps.append(
        cmake_test_step(
            command_name,
            f"Test IREE CMake AMDGPU package tests{sanitizer_name}",
            regex="^iree/hal/drivers/amdgpu/",
            exclude_regex=xfail_regex,
            env=sanitizer_env(sanitizer) + amdgpu_libhsa_test_env(),
            parallelism=1,
        )
    )
    resource_exclude_regex = combine_ctest_regex(
        "^iree/hal/drivers/amdgpu/",
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
            env=sanitizer_env(sanitizer) + amdgpu_libhsa_test_env(),
            parallelism=1,
        )
    )
    return steps


def cmake_loom_amdgpu_steps(command_name: str) -> list[CiStep]:
    return [
        cmake_configure_step(command_name),
        cmake_build_step(
            command_name,
            "Build Loom CMake AMDGPU compile coverage",
            ci_config.LOOM_AMDGPU_CMAKE_COMPILE_TEST_BUILD_TARGETS,
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
    if tests_enabled:
        build_targets += (
            cmake_runtime_resource_build_target(
                ci_config.VULKAN_CTEST_RESOURCE_LABEL_REGEX
            ),
        )
    steps = [
        cmake_configure_step(
            command_name,
            enabled_drivers=("vulkan",),
            sanitizer=sanitizer,
        ),
        cmake_build_step(
            command_name,
            f"Build IREE CMake Vulkan{sanitizer_name}",
            build_targets,
        ),
    ]
    if tests_enabled:
        steps.append(
            cmake_test_step(
                command_name,
                f"Test IREE CMake Vulkan package tests{sanitizer_name}",
                regex=ci_config.VULKAN_CTEST_REGEX,
                env=sanitizer_env(sanitizer),
            )
        )
        steps.append(
            cmake_test_step(
                command_name,
                f"Test IREE CMake Vulkan resource tests{sanitizer_name}",
                label_regex=ci_config.VULKAN_CTEST_RESOURCE_LABEL_REGEX,
                label_exclude_regex=ci_config.CTEST_MANUAL_LABEL_EXCLUDE_REGEX,
                exclude_regex=ci_config.VULKAN_CTEST_REGEX,
                env=sanitizer_env(sanitizer),
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
    for config in ci_config.SANITIZER_BUILD_CONFIGS:
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
    for config in ci_config.SANITIZER_BUILD_CONFIGS:
        command_name = f"{CMAKE_SANITIZER_SMOKE_COMMAND}-{config}"
        steps.extend(
            [
                cmake_configure_step(
                    command_name,
                    sanitizer=config,
                    build_tests=False,
                ),
                cmake_build_step(
                    command_name,
                    f"Build IREE CMake sanitizer smoke with {config.upper()}",
                    ci_config.CMAKE_SANITIZER_SMOKE_LIBRARY_BUILD_TARGETS,
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
    amdgpu_target_selector = (
        args.amdgpu_target or ci_config.DEFAULT_AMDGPU_TARGET_SELECTOR
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
        if sanitizer is not None:
            return [
                bazel_configure_step(enabled_drivers=("amdgpu",)),
                *amdgpu_config_steps(targets, amdgpu_target_selector, sanitizer),
            ]
        return amdgpu_steps(targets, amdgpu_target_selector)
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
            CiStep(
                step.name,
                step.argv[:4] + (f"--profile={profile_path}",) + step.argv[4:],
                step.env,
            )
        )
    return profiled_steps


def bazel_profile_dir(args: argparse.Namespace) -> Path | None:
    if args.bazel_profile_dir is None:
        return None
    return args.bazel_profile_dir.expanduser().resolve()


def steps_from_args(args: argparse.Namespace) -> list[CiStep]:
    profile_dir = bazel_profile_dir(args)
    if profile_dir is not None and args.command not in BAZEL_COMMANDS:
        raise ValueError("--bazel-profile-dir is only supported for Bazel CI commands")
    steps = _steps_from_args(args)
    if profile_dir is not None:
        steps = add_bazel_profiles(steps, profile_dir)
    return steps


def github_actions_enabled() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def print_group_start(name: str) -> None:
    if github_actions_enabled():
        print(f"::group::{name}")


def print_group_end() -> None:
    if github_actions_enabled():
        print("::endgroup::")


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
    returncode = subprocess.run(step.argv, cwd=REPO_ROOT, env=environment).returncode
    elapsed_seconds = time.monotonic() - start_time
    result = StepResult(step, returncode, elapsed_seconds)
    if result.ok:
        print(f"[ok] {step.name} ({elapsed_seconds:.1f}s)", flush=True)
    else:
        print(
            f"[fail] {step.name} ({elapsed_seconds:.1f}s, exit {returncode})",
            flush=True,
        )
        print("  " + step.command_line(), flush=True)
    return result


def write_step_summary(results: list[StepResult]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = [
        "## IREE CI",
        "",
        "| Phase | Result | Time |",
        "| --- | --- | ---: |",
    ]
    for result in results:
        outcome = "pass" if result.ok else f"fail ({result.returncode})"
        lines.append(
            f"| {result.step.name} | {outcome} | {result.elapsed_seconds:.1f}s |"
        )
    with Path(summary_path).open("a", encoding="utf-8") as summary_file:
        summary_file.write("\n".join(lines) + "\n")


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
    results = []
    for step in steps:
        print_group_start(step.name)
        try:
            result = run_step(step, verbose=verbose)
        finally:
            print_group_end()
        results.append(result)
        if not result.ok and not keep_going:
            write_step_summary(results)
            return result.returncode

    write_step_summary(results)
    failures = [result for result in results if not result.ok]
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
        help="Run every phase and report failures at the end.",
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
        "--bazel-profile-dir",
        type=Path,
        help=(
            "Write one compressed Bazel execution profile per build/test/run phase "
            "to this directory. Only supported for Bazel CI commands."
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
    if profile_dir is not None and not args.dry_run:
        profile_dir.mkdir(parents=True, exist_ok=True)
    return run_steps(
        steps,
        dry_run=args.dry_run,
        keep_going=args.keep_going,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
