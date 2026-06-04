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
}
CMAKE_COMMANDS = {
    "iree-cmake-cpu": (False, None),
    "iree-cmake-cpu-asan": (False, "asan"),
    "iree-cmake-cpu-msan": (False, "msan"),
    "iree-cmake-cpu-tsan": (False, "tsan"),
    "iree-cmake-cpu-ubsan": (False, "ubsan"),
    "iree-cmake-cpu-sanitizers": (False, "all"),
    "iree-cmake-amdgpu": (True, None),
    "iree-cmake-amdgpu-asan": (True, "asan"),
    "iree-cmake-amdgpu-msan": (True, "msan"),
    "iree-cmake-amdgpu-tsan": (True, "tsan"),
    "iree-cmake-amdgpu-ubsan": (True, "ubsan"),
    "iree-cmake-amdgpu-sanitizers": (True, "all"),
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


def sanitizer_env(config: str | None) -> tuple[tuple[str, str], ...]:
    if config != "tsan":
        return ()
    return (("TSAN_OPTIONS", f"suppressions={TSAN_SUPPRESSIONS_FILE}"),)


def cmake_dev_command(command_name: str, *args: str) -> tuple[str, ...]:
    build_dir = CMAKE_CI_BUILD_ROOT / command_name
    return dev_command("--cmake-build-dir", str(build_dir), "cmake", *args)


def bazel_configure_step(enable_amdgpu: bool = False) -> CiStep:
    command = ["bazel", "configure"]
    if enable_amdgpu:
        command.append("-DIREE_HAL_DRIVER_AMDGPU=ON")
        command.append(ROCM_PINNED_DEPENDENCY_MODE_OPTION)
    return CiStep("Configure Bazel", dev_command(*command))


def bazel_build_step(
    name: str,
    targets: tuple[str, ...],
    config: str | None = None,
) -> CiStep:
    command = ["bazel", "build", *targets]
    if config is not None:
        command.append(f"--config={config}")
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
    enable_amdgpu: bool = False,
    sanitizer: str | None = None,
) -> CiStep:
    command = [
        "configure",
        "--fresh",
        "-GNinja",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
        "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
        "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
        f"-DIREE_BUILD_TESTS={'OFF' if sanitizer == 'msan' else 'ON'}",
        f"-DIREE_BUILD_BENCHMARKS={'OFF' if sanitizer == 'msan' else 'ON'}",
        "-DIREE_ENABLE_LIBBACKTRACE=OFF",
        "-DLIBHRX_BUILD=OFF",
        f"-DIREE_HAL_DRIVER_AMDGPU={'ON' if enable_amdgpu else 'OFF'}",
    ]
    if enable_amdgpu:
        command.append(ROCM_PINNED_DEPENDENCY_MODE_OPTION)
        command.append(f"-DIREE_HAL_AMDGPU_TARGETS={ci_config.AMDGPU_TARGET_SELECTOR}")
    if sanitizer is not None:
        command.append("-DIREE_ENABLE_ASSERTIONS=ON")
        command.extend(CMAKE_SANITIZER_OPTIONS[sanitizer])
    return CiStep("Configure CMake", cmake_dev_command(command_name, *command))


def cmake_build_step(command_name: str, name: str) -> CiStep:
    return CiStep(
        name,
        cmake_dev_command(command_name, "build", "--parallel"),
    )


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
    command = ["test", "--parallel", str(parallelism)]
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
    return [
        bazel_configure_step(),
        bazel_build_step("Build IREE", targets),
        bazel_test_step("Test IREE", targets + ci_config.CPU_XFAIL_TARGETS),
    ]


def cpu_sanitizer_steps(targets: tuple[str, ...]) -> list[CiStep]:
    steps = [bazel_configure_step()]
    for config in ci_config.SANITIZER_TEST_CONFIGS:
        steps.extend(cpu_config_steps(targets, config))
    for config in ci_config.SANITIZER_BUILD_CONFIGS:
        steps.extend(cpu_config_steps(targets, config))
    return steps


def cpu_config_steps(targets: tuple[str, ...], config: str) -> list[CiStep]:
    if config in ci_config.SANITIZER_TEST_CONFIGS:
        return [
            bazel_test_step(
                f"Test IREE with {config.upper()}",
                targets + ci_config.CPU_SANITIZERS_XFAIL_TARGETS,
                config=config,
            )
        ]
    if config in ci_config.SANITIZER_BUILD_CONFIGS:
        return [
            bazel_build_step(
                f"Build IREE with {config.upper()}",
                targets,
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
    return [
        bazel_test_step(
            f"Test IREE AMDGPU resources{config_name}",
            targets + xfail_targets,
            config=config,
            test_tag_filters=(ci_config.AMDGPU_RESOURCE_TAG,),
        ),
    ]


def amdgpu_steps(targets: tuple[str, ...]) -> list[CiStep]:
    return [
        bazel_configure_step(enable_amdgpu=True),
        bazel_build_step(
            "Build IREE with AMDGPU",
            ci_config.AMDGPU_DRIVER_TARGETS,
        ),
        *amdgpu_test_steps(targets, xfail_targets=ci_config.AMDGPU_XFAIL_TARGETS),
    ]


def amdgpu_sanitizer_steps(targets: tuple[str, ...]) -> list[CiStep]:
    steps = [bazel_configure_step(enable_amdgpu=True)]
    for config in ci_config.SANITIZER_TEST_CONFIGS:
        steps.extend(amdgpu_config_steps(targets, config))
    for config in ci_config.SANITIZER_BUILD_CONFIGS:
        steps.extend(amdgpu_config_steps(targets, config))
    return steps


def amdgpu_config_steps(targets: tuple[str, ...], config: str) -> list[CiStep]:
    if config in ci_config.SANITIZER_TEST_CONFIGS:
        return amdgpu_test_steps(
            targets,
            config=config,
            xfail_targets=ci_config.AMDGPU_SANITIZERS_XFAIL_TARGETS,
        )
    if config in ci_config.SANITIZER_BUILD_CONFIGS:
        return [
            bazel_build_step(
                f"Build IREE with AMDGPU and {config.upper()}",
                ci_config.AMDGPU_DRIVER_TARGETS,
                config=config,
            )
        ]
    raise ValueError(f"unknown Bazel AMDGPU sanitizer config: {config}")


def cmake_cpu_steps(command_name: str, sanitizer: str | None) -> list[CiStep]:
    sanitizer_name = f" with {sanitizer.upper()}" if sanitizer is not None else ""
    xfail_regex = (
        ci_config.CPU_SANITIZERS_CTEST_EXCLUDE_REGEX
        if sanitizer is not None
        else ci_config.CPU_CTEST_EXCLUDE_REGEX
    )
    steps = [
        cmake_configure_step(command_name, sanitizer=sanitizer),
        cmake_build_step(command_name, f"Build IREE CMake{sanitizer_name}"),
    ]
    if sanitizer != "msan":
        steps.append(
            cmake_test_step(
                command_name,
                f"Test IREE CMake{sanitizer_name}",
                exclude_regex=xfail_regex,
                env=sanitizer_env(sanitizer),
                label_exclude_regex="runtime-resource=",
            )
        )
    return steps


def cmake_amdgpu_steps(command_name: str, sanitizer: str | None) -> list[CiStep]:
    sanitizer_name = f" with {sanitizer.upper()}" if sanitizer is not None else ""
    xfail_regex = (
        ci_config.AMDGPU_SANITIZERS_CTEST_EXCLUDE_REGEX
        if sanitizer is not None
        else ci_config.AMDGPU_CTEST_EXCLUDE_REGEX
    )
    steps = [
        cmake_configure_step(
            command_name,
            enable_amdgpu=True,
            sanitizer=sanitizer,
        ),
        cmake_build_step(command_name, f"Build IREE CMake AMDGPU{sanitizer_name}"),
    ]
    if sanitizer == "msan":
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
            label_regex=ci_config.AMDGPU_CTEST_RESOURCE_LABEL,
            exclude_regex=resource_exclude_regex,
            env=sanitizer_env(sanitizer),
            parallelism=1,
        )
    )
    return steps


def cmake_sanitizer_steps(prefix: str, enable_amdgpu: bool) -> list[CiStep]:
    steps = []
    for config in ci_config.SANITIZER_TEST_CONFIGS:
        command_name = f"{prefix}-{config}"
        if enable_amdgpu:
            steps.extend(cmake_amdgpu_steps(command_name, config))
        else:
            steps.extend(cmake_cpu_steps(command_name, config))
    for config in ci_config.SANITIZER_BUILD_CONFIGS:
        command_name = f"{prefix}-{config}"
        if enable_amdgpu:
            steps.extend(cmake_amdgpu_steps(command_name, config))
        else:
            steps.extend(cmake_cpu_steps(command_name, config))
    return steps


def steps_from_args(args: argparse.Namespace) -> list[CiStep]:
    if args.command in CMAKE_COMMANDS:
        if args.target:
            raise ValueError("--target is only supported for Bazel CI commands")
        enable_amdgpu, sanitizer = CMAKE_COMMANDS[args.command]
        if sanitizer == "all":
            prefix = args.command.removesuffix("-sanitizers")
            return cmake_sanitizer_steps(prefix, enable_amdgpu)
        if enable_amdgpu:
            return cmake_amdgpu_steps(args.command, sanitizer)
        return cmake_cpu_steps(args.command, sanitizer)

    bazel_target, sanitizer = BAZEL_COMMANDS[args.command]
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
                bazel_configure_step(enable_amdgpu=True),
                *amdgpu_config_steps(targets, sanitizer),
            ]
        return amdgpu_steps(targets)
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
