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
CI_HAL_DRIVER_DEFINES = (
    ("amdgpu", "IREE_HAL_DRIVER_AMDGPU"),
    ("vulkan", "IREE_HAL_DRIVER_VULKAN"),
)
ROCM_PINNED_DEPENDENCY_MODE_OPTION = "-DIREE_ROCM_DEPENDENCY_MODE=pinned"
BAZEL_COMMANDS = {
    "iree-bazel-cpu": ("cpu", None),
    "iree-bazel-cpu-asan": ("cpu", "asan"),
    "iree-bazel-cpu-msan": ("cpu", "msan"),
    "iree-bazel-cpu-tsan": ("cpu", "tsan"),
    "iree-bazel-cpu-ubsan": ("cpu", "ubsan"),
    "iree-bazel-cpu-sanitizers": ("cpu", "all"),
    "iree-bazel-amdgpu": ("amdgpu", None),
    "iree-bazel-amdgpu-asan": ("amdgpu", "asan"),
    "iree-bazel-amdgpu-msan": ("amdgpu", "msan"),
    "iree-bazel-amdgpu-tsan": ("amdgpu", "tsan"),
    "iree-bazel-amdgpu-ubsan": ("amdgpu", "ubsan"),
    "iree-bazel-amdgpu-sanitizers": ("amdgpu", "all"),
    "iree-bazel-loom-amdgpu": ("loom-amdgpu", None),
    "iree-bazel-vulkan": ("vulkan", None),
    "iree-bazel-vulkan-asan": ("vulkan", "asan"),
    "iree-bazel-vulkan-msan": ("vulkan", "msan"),
    "iree-bazel-vulkan-tsan": ("vulkan", "tsan"),
    "iree-bazel-vulkan-ubsan": ("vulkan", "ubsan"),
    "iree-bazel-vulkan-sanitizers": ("vulkan", "all"),
}
CMAKE_COMMANDS = {
    "iree-cmake-cpu": ("cpu", None),
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


def sanitizer_env(config: str | None) -> tuple[tuple[str, str], ...]:
    if config != "tsan":
        return ()
    return (("TSAN_OPTIONS", f"suppressions={TSAN_SUPPRESSIONS_FILE}"),)


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
    supported_drivers = frozenset(driver for driver, _ in CI_HAL_DRIVER_DEFINES)
    enabled_driver_set = frozenset(enabled_drivers)
    unsupported_drivers = enabled_driver_set.difference(supported_drivers)
    if unsupported_drivers:
        raise ValueError(
            "unsupported CI HAL driver(s): " + ", ".join(sorted(unsupported_drivers))
        )
    return enabled_driver_set


def bazel_configure_step(enabled_drivers: tuple[str, ...] = ()) -> CiStep:
    enabled_driver_set = validate_enabled_drivers(enabled_drivers)
    command = ["bazel", "configure"]
    for driver, define in CI_HAL_DRIVER_DEFINES:
        if driver in enabled_driver_set:
            command.append(f"-D{define}=ON")
    if "amdgpu" in enabled_driver_set:
        command.append(ROCM_PINNED_DEPENDENCY_MODE_OPTION)
    return CiStep("Configure Bazel", dev_command(*command))


def bazel_build_step(
    name: str,
    targets: tuple[str, ...],
    config: str | None = None,
) -> CiStep:
    command = ["bazel", "build"]
    if config is not None:
        command.append(f"--config={config}")
    if any(target.startswith("-") for target in targets):
        command.append("--")
    command.extend(targets)
    return CiStep(name, dev_command(*command))


def bazel_test_step(
    name: str,
    targets: tuple[str, ...],
    config: str | None = None,
    test_tag_filters: tuple[str, ...] = (),
) -> CiStep:
    options = []
    if config is not None:
        options.append(f"--config={config}")
    if test_tag_filters:
        options.append("--test_tag_filters=" + ",".join(test_tag_filters))
    command = ["bazel", "test", *options]
    if any(target.startswith("-") for target in targets):
        command.append("--")
    command.extend(targets)
    return CiStep(name, dev_command(*command), env=sanitizer_env(config))


def cmake_configure_step(
    command_name: str,
    *,
    enabled_drivers: tuple[str, ...] = (),
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
    for driver, define in CI_HAL_DRIVER_DEFINES:
        command.append(f"-D{define}={'ON' if driver in enabled_driver_set else 'OFF'}")
    if "amdgpu" in enabled_driver_set:
        command.append(ROCM_PINNED_DEPENDENCY_MODE_OPTION)
        command.append(f"-DIREE_HAL_AMDGPU_TARGETS={ci_config.AMDGPU_TARGET_SELECTOR}")
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


def amdgpu_test_steps(
    targets: tuple[str, ...],
    config: str | None = None,
    xfail_targets: tuple[str, ...] = (),
) -> list[CiStep]:
    config_name = f" and {config.upper()}" if config is not None else ""
    steps = []
    for (
        slice_name,
        target_prefix,
        default_target,
        resource_tag,
    ) in ci_config.AMDGPU_BAZEL_RESOURCE_SLICES:
        slice_targets = resource_slice_targets(
            targets + xfail_targets,
            target_prefix=target_prefix,
            default_target=default_target,
        )
        if not slice_targets:
            continue
        steps.append(
            bazel_test_step(
                f"Test IREE AMDGPU {slice_name} resources{config_name}",
                slice_targets,
                config=config,
                test_tag_filters=(resource_tag,),
            )
        )
    return steps


def resource_slice_targets(
    targets: tuple[str, ...],
    *,
    target_prefix: str,
    default_target: str,
) -> tuple[str, ...]:
    positive_targets = []
    negative_targets = []
    seen_targets = set()
    for target in targets:
        is_negative = target.startswith("-")
        raw_target = target[1:] if is_negative else target
        if raw_target in ("//...", "..."):
            selected_target = "-" + default_target if is_negative else default_target
        elif target_in_prefix(raw_target, target_prefix):
            selected_target = target
        else:
            continue
        if selected_target in seen_targets:
            continue
        if is_negative:
            negative_targets.append(selected_target)
        else:
            positive_targets.append(selected_target)
        seen_targets.add(selected_target)
    if not positive_targets:
        return ()
    return tuple(positive_targets + negative_targets)


def target_in_prefix(target: str, target_prefix: str) -> bool:
    return (
        target == target_prefix
        or target.startswith(target_prefix + "/")
        or target.startswith(target_prefix + ":")
    )


def amdgpu_steps(targets: tuple[str, ...]) -> list[CiStep]:
    return [
        bazel_configure_step(enabled_drivers=("amdgpu",)),
        bazel_build_step(
            "Build IREE with AMDGPU",
            ci_config.AMDGPU_BAZEL_DRIVER_TARGETS,
        ),
        *amdgpu_test_steps(targets, xfail_targets=ci_config.AMDGPU_XFAIL_TARGETS),
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


def amdgpu_sanitizer_steps(targets: tuple[str, ...]) -> list[CiStep]:
    steps = [bazel_configure_step(enabled_drivers=("amdgpu",))]
    for config in ci_config.SANITIZER_TEST_CONFIGS:
        steps.extend(amdgpu_config_steps(targets, config))
    for config in ci_config.SANITIZER_BUILD_CONFIGS:
        steps.extend(amdgpu_config_steps(targets, config))
    return steps


def amdgpu_config_steps(targets: tuple[str, ...], config: str) -> list[CiStep]:
    if config in ci_config.SANITIZER_TEST_CONFIGS:
        xfail_targets = (
            ci_config.AMDGPU_TSAN_SANITIZERS_XFAIL_TARGETS
            if config == "tsan"
            else ci_config.AMDGPU_SANITIZERS_XFAIL_TARGETS
        )
        return amdgpu_test_steps(
            targets,
            config=config,
            xfail_targets=xfail_targets,
        )
    if config in ci_config.SANITIZER_BUILD_CONFIGS:
        return [
            bazel_build_step(
                f"Build IREE with AMDGPU and {config.upper()}",
                ci_config.AMDGPU_BAZEL_DRIVER_TARGETS,
                config=config,
            )
        ]
    raise ValueError(f"unknown Bazel AMDGPU sanitizer config: {config}")


def vulkan_test_steps(
    targets: tuple[str, ...],
    config: str | None = None,
) -> list[CiStep]:
    config_name = f" with {config.upper()}" if config is not None else ""
    steps = [
        bazel_test_step(
            f"Test IREE Vulkan package tests{config_name}",
            ci_config.VULKAN_BAZEL_DRIVER_TARGETS + ci_config.VULKAN_XFAIL_TARGETS,
            config=config,
        ),
    ]
    for (
        slice_name,
        target_prefix,
        default_target,
        resource_tag,
    ) in ci_config.VULKAN_BAZEL_RESOURCE_SLICES:
        slice_targets = resource_slice_targets(
            targets,
            target_prefix=target_prefix,
            default_target=default_target,
        )
        if not slice_targets:
            continue
        steps.append(
            bazel_test_step(
                f"Test IREE Vulkan {slice_name} resources{config_name}",
                slice_targets,
                config=config,
                test_tag_filters=(resource_tag,),
            )
        )
    return steps


def vulkan_steps(targets: tuple[str, ...]) -> list[CiStep]:
    return [
        bazel_configure_step(enabled_drivers=("vulkan",)),
        bazel_build_step(
            "Build IREE with Vulkan",
            ci_config.VULKAN_BAZEL_DRIVER_TARGETS,
        ),
        *vulkan_test_steps(targets),
    ]


def vulkan_sanitizer_steps(targets: tuple[str, ...]) -> list[CiStep]:
    steps = [bazel_configure_step(enabled_drivers=("vulkan",))]
    for config in ci_config.SANITIZER_TEST_CONFIGS:
        steps.extend(vulkan_config_steps(targets, config))
    for config in ci_config.SANITIZER_BUILD_CONFIGS:
        steps.extend(vulkan_config_steps(targets, config))
    return steps


def vulkan_config_steps(targets: tuple[str, ...], config: str) -> list[CiStep]:
    # Vulkan hardware tests execute the system loader and ICD in-process. Keep
    # sanitizer coverage to compile-time checks here; the unsanitized Vulkan
    # lane owns execution on real devices.
    if config in ci_config.SANITIZER_TEST_CONFIGS:
        return [
            bazel_build_step(
                f"Build IREE with Vulkan and {config.upper()}",
                ci_config.VULKAN_BAZEL_DRIVER_TARGETS,
                config=config,
            )
        ]
    if config in ci_config.SANITIZER_BUILD_CONFIGS:
        return [
            bazel_build_step(
                f"Build IREE with Vulkan and {config.upper()}",
                ci_config.VULKAN_BAZEL_DRIVER_TARGETS,
                config=config,
            )
        ]
    raise ValueError(f"unknown Bazel Vulkan sanitizer config: {config}")


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


def cmake_amdgpu_steps(command_name: str, sanitizer: str | None) -> list[CiStep]:
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
            env=sanitizer_env(sanitizer),
            parallelism=1,
        )
    )
    resource_exclude_regex = combine_ctest_regex(
        "^iree/hal/drivers/amdgpu/",
        xfail_regex,
    )
    steps.append(
        cmake_test_step(
            command_name,
            f"Test IREE CMake AMDGPU resource tests{sanitizer_name}",
            label_regex=ci_config.AMDGPU_CTEST_RESOURCE_LABEL_REGEX,
            label_exclude_regex=ci_config.CTEST_MANUAL_LABEL_EXCLUDE_REGEX,
            exclude_regex=resource_exclude_regex,
            env=sanitizer_env(sanitizer),
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
    command_name: str, target_group: str, sanitizer: str | None
) -> list[CiStep]:
    if target_group == "cpu":
        return cmake_cpu_steps(command_name, sanitizer)
    if target_group == "amdgpu":
        return cmake_amdgpu_steps(command_name, sanitizer)
    if target_group == "loom-amdgpu":
        if sanitizer is not None:
            raise ValueError("Loom AMDGPU CMake CI does not support sanitizers")
        return cmake_loom_amdgpu_steps(command_name)
    if target_group == "vulkan":
        return cmake_vulkan_steps(command_name, sanitizer)
    raise ValueError(f"unknown CMake CI target: {target_group}")


def cmake_sanitizer_steps(prefix: str, target_group: str) -> list[CiStep]:
    steps = []
    for config in ci_config.SANITIZER_TEST_CONFIGS:
        command_name = f"{prefix}-{config}"
        steps.extend(cmake_target_steps(command_name, target_group, config))
    for config in ci_config.SANITIZER_BUILD_CONFIGS:
        command_name = f"{prefix}-{config}"
        steps.extend(cmake_target_steps(command_name, target_group, config))
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
                cmake_build_step(
                    command_name,
                    f"Build IREE CMake sanitizer smoke with {config.upper()}",
                    ci_config.CMAKE_SANITIZER_SMOKE_TEST_BUILD_TARGETS,
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


def steps_from_args(args: argparse.Namespace) -> list[CiStep]:
    if args.command == CMAKE_SANITIZER_SMOKE_COMMAND:
        if args.target:
            raise ValueError("--target is only supported for Bazel CI commands")
        return cmake_sanitizer_smoke_steps()

    if args.command in CMAKE_COMMANDS:
        if args.target:
            raise ValueError("--target is only supported for Bazel CI commands")
        target_group, sanitizer = CMAKE_COMMANDS[args.command]
        if sanitizer == "all":
            prefix = args.command.removesuffix("-sanitizers")
            return cmake_sanitizer_steps(prefix, target_group)
        return cmake_target_steps(args.command, target_group, sanitizer)

    bazel_target, sanitizer = BAZEL_COMMANDS[args.command]
    if bazel_target == "loom-amdgpu":
        if args.target:
            raise ValueError("--target is not supported by Loom AMDGPU CI")
        if sanitizer is not None:
            raise ValueError("Loom AMDGPU Bazel CI does not support sanitizers")
        return loom_amdgpu_bazel_steps()
    targets = command_targets(args.target)
    if bazel_target == "cpu":
        if sanitizer == "all":
            return cpu_sanitizer_steps(targets)
        if sanitizer is not None:
            return [bazel_configure_step(), *cpu_config_steps(targets, sanitizer)]
        return cpu_steps(targets)
    if bazel_target == "amdgpu":
        if sanitizer == "all":
            return amdgpu_sanitizer_steps(targets)
        if sanitizer is not None:
            return [
                bazel_configure_step(enabled_drivers=("amdgpu",)),
                *amdgpu_config_steps(targets, sanitizer),
            ]
        return amdgpu_steps(targets)
    if bazel_target == "vulkan":
        if sanitizer == "all":
            return vulkan_sanitizer_steps(targets)
        if sanitizer is not None:
            return [
                bazel_configure_step(enabled_drivers=("vulkan",)),
                *vulkan_config_steps(targets, sanitizer),
            ]
        return vulkan_steps(targets)
    raise ValueError(f"unknown Bazel CI target: {bazel_target}")


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
    return run_steps(
        steps_from_args(args),
        dry_run=args.dry_run,
        keep_going=args.keep_going,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
