#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates local Bazel configuration for this checkout."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

LOOM_EXECUTE_SUBSTRATES = ("iree_hal", "iree_vm")
LOOM_TARGETS = ("amdgpu", "iree_vm", "llvmir", "spirv", "wasm", "x86")
LOOM_EMITTERS = ("amdgpu", "iree_vm", "llvmir", "spirv", "wasm")
HOST_DRIVERS = ("local-sync", "local-task", "null")
DEFAULT_LOOM_EXECUTE = LOOM_EXECUTE_SUBSTRATES
DEFAULT_LOOM_TARGETS = ("amdgpu", "iree_vm", "llvmir", "spirv", "x86")

SDK_DRIVER_PACKAGES = {
    "amdgpu": (
        "runtime/src/iree/hal/drivers/amdgpu",
        "runtime/src/iree/hal/drivers/amdgpu/abi",
        "runtime/src/iree/hal/drivers/amdgpu/cts",
        "runtime/src/iree/hal/drivers/amdgpu/device",
        "runtime/src/iree/hal/drivers/amdgpu/device/binaries",
        "runtime/src/iree/hal/drivers/amdgpu/device/binaries/prebuilt",
        "runtime/src/iree/hal/drivers/amdgpu/device/binaries/source",
        "runtime/src/iree/hal/drivers/amdgpu/registration",
        "runtime/src/iree/hal/drivers/amdgpu/util",
    ),
    "cuda": (
        "runtime/src/iree/hal/drivers/cuda",
        "runtime/src/iree/hal/drivers/cuda/cts",
        "runtime/src/iree/hal/drivers/cuda/registration",
    ),
    "hip": (
        "runtime/src/iree/hal/drivers/hip",
        "runtime/src/iree/hal/drivers/hip/cts",
        "runtime/src/iree/hal/drivers/hip/registration",
        "runtime/src/iree/hal/drivers/hip/util",
    ),
    "vulkan": (
        "runtime/src/iree/hal/drivers/vulkan",
        "runtime/src/iree/hal/drivers/vulkan/cts",
        "runtime/src/iree/hal/drivers/vulkan/registration",
        "runtime/src/iree/hal/drivers/vulkan/util",
    ),
    "webgpu": (
        "runtime/src/iree/hal/drivers/webgpu",
        "runtime/src/iree/hal/drivers/webgpu/cts",
        "runtime/src/iree/hal/drivers/webgpu/registration",
    ),
}
ROCM_DRIVERS = frozenset(("amdgpu", "hip"))
DEPENDENCY_MODES = frozenset(("pinned", "package", "auto"))
SUPPORTED_ENABLE_DRIVERS = frozenset(
    (*HOST_DRIVERS, "amdgpu", "hip", "vulkan", "webgpu")
)
ALL_DRIVERS = tuple(HOST_DRIVERS) + tuple(SDK_DRIVER_PACKAGES)
DRIVER_DEFINES = {
    "IREE_HAL_DRIVER_AMDGPU": "amdgpu",
    "IREE_HAL_DRIVER_HIP": "hip",
    "IREE_HAL_DRIVER_LOCAL_SYNC": "local-sync",
    "IREE_HAL_DRIVER_LOCAL_TASK": "local-task",
    "IREE_HAL_DRIVER_NULL": "null",
    "IREE_HAL_DRIVER_VULKAN": "vulkan",
    "IREE_HAL_DRIVER_WEBGPU": "webgpu",
}
LOOM_TARGET_DEFINES = {
    "LOOM_TARGET_AMDGPU": "amdgpu",
    "LOOM_TARGET_IREE_VM": "iree_vm",
    "LOOM_TARGET_LLVMIR": "llvmir",
    "LOOM_TARGET_SPIRV": "spirv",
    "LOOM_TARGET_WASM": "wasm",
    "LOOM_TARGET_X86": "x86",
}
LOOM_EMIT_DEFINES = {
    "LOOM_EMIT_LLVMIR": "llvmir",
}
LOOM_EXECUTE_DEFINES = {
    "LOOM_EXECUTE_IREE_HAL": "iree_hal",
    "LOOM_EXECUTE_IREE_VM": "iree_vm",
}
UNSUPPORTED_DRIVER_DEFINES = {
    "IREE_HAL_DRIVER_CUDA": "cuda",
    "IREE_HAL_DRIVER_METAL": "metal",
}
REMOVED_OPTIONS = frozenset(
    ("--enable-driver", "--include-driver", "--exclude-driver", "--rocm-path")
)
NATIVE_DRIVER_FLAG = "--//runtime/config/hal:drivers"
NATIVE_LOOM_TARGET_FLAG = "--//loom/config/target:enable"
NATIVE_LOOM_EMIT_FLAG = "--//loom/config/emit:enable"
NATIVE_LOOM_EXECUTE_FLAG = "--//loom/config/execute:enable"
NATIVE_REPO_ENV_PREFIX = "--repo_env="
TRUE_VALUES = frozenset(("1", "ON", "TRUE", "YES"))
FALSE_VALUES = frozenset(("0", "OFF", "FALSE", "NO"))


@dataclass
class ConfigRequest:
    output: Path
    enabled_drivers: set[str] = field(default_factory=lambda: set(HOST_DRIVERS))
    driver_source: str | None = None
    dependency_mode: str = "pinned"
    rocm_dependency_mode: str | None = None
    enabled_loom_targets: set[str] = field(
        default_factory=lambda: set(DEFAULT_LOOM_TARGETS)
    )
    enabled_loom_execute: set[str] = field(
        default_factory=lambda: set(DEFAULT_LOOM_EXECUTE)
    )
    loom_execute_source: str | None = None
    loom_emit_source: str | None = None
    loom_target_source: str | None = None
    enabled_loom_emitters: set[str] = field(default_factory=set)
    rocm_path: str | None = None

    def set_driver(self, driver: str, enabled: bool) -> None:
        if self.driver_source == "native":
            raise SystemExit(
                "Do not mix portable -DIREE_HAL_DRIVER_* options with the "
                f"native {NATIVE_DRIVER_FLAG}=... Bazel option."
            )
        self.driver_source = "portable"
        if enabled:
            self.enabled_drivers.add(driver)
        else:
            self.enabled_drivers.discard(driver)

    def set_driver_list(self, drivers: set[str]) -> None:
        if self.driver_source == "portable":
            raise SystemExit(
                "Do not mix portable -DIREE_HAL_DRIVER_* options with the "
                f"native {NATIVE_DRIVER_FLAG}=... Bazel option."
            )
        unknown_drivers = drivers.difference(ALL_DRIVERS)
        if unknown_drivers:
            raise SystemExit(
                "Unknown runtime HAL driver(s): {}".format(
                    ", ".join(sorted(unknown_drivers))
                )
            )
        self.driver_source = "native"
        self.enabled_drivers = set(drivers)

    def set_loom_target(self, target: str, enabled: bool) -> None:
        if self.loom_target_source == "native":
            raise SystemExit(
                "Do not mix portable -DLOOM_TARGET_* options with the "
                f"native {NATIVE_LOOM_TARGET_FLAG}=... Bazel option."
            )
        self.loom_target_source = "portable"
        if enabled:
            self.enabled_loom_targets.add(target)
        else:
            self.enabled_loom_targets.discard(target)

    def set_loom_target_list(self, targets: set[str]) -> None:
        if self.loom_target_source == "portable":
            raise SystemExit(
                "Do not mix portable -DLOOM_TARGET_* options with the "
                f"native {NATIVE_LOOM_TARGET_FLAG}=... Bazel option."
            )
        unknown_targets = targets.difference(LOOM_TARGETS)
        if unknown_targets:
            raise SystemExit(
                "Unknown Loom target(s): {}".format(", ".join(sorted(unknown_targets)))
            )
        self.loom_target_source = "native"
        self.enabled_loom_targets = set(targets)

    def set_loom_emitter(self, emitter: str, enabled: bool) -> None:
        if self.loom_emit_source == "native":
            raise SystemExit(
                "Do not mix portable -DLOOM_EMIT_* options with the "
                f"native {NATIVE_LOOM_EMIT_FLAG}=... Bazel option."
            )
        self.loom_emit_source = "portable"
        if enabled:
            self.enabled_loom_emitters.add(emitter)
        else:
            self.enabled_loom_emitters.discard(emitter)

    def set_loom_emitter_list(self, emitters: set[str]) -> None:
        if self.loom_emit_source == "portable":
            raise SystemExit(
                "Do not mix portable -DLOOM_EMIT_* options with the "
                f"native {NATIVE_LOOM_EMIT_FLAG}=... Bazel option."
            )
        unknown_emitters = emitters.difference(LOOM_EMITTERS)
        if unknown_emitters:
            raise SystemExit(
                "Unknown explicit Loom emitter(s): {}".format(
                    ", ".join(sorted(unknown_emitters))
                )
            )
        self.loom_emit_source = "native"
        self.enabled_loom_emitters = set(emitters)

    def set_loom_execute(self, execute: str, enabled: bool) -> None:
        if self.loom_execute_source == "native":
            raise SystemExit(
                "Do not mix portable -DLOOM_EXECUTE_* options with the "
                f"native {NATIVE_LOOM_EXECUTE_FLAG}=... Bazel option."
            )
        self.loom_execute_source = "portable"
        if enabled:
            self.enabled_loom_execute.add(execute)
        else:
            self.enabled_loom_execute.discard(execute)

    def set_loom_execute_list(self, execute_values: set[str]) -> None:
        if self.loom_execute_source == "portable":
            raise SystemExit(
                "Do not mix portable -DLOOM_EXECUTE_* options with the "
                f"native {NATIVE_LOOM_EXECUTE_FLAG}=... Bazel option."
            )
        unknown_execute_values = execute_values.difference(LOOM_EXECUTE_SUBSTRATES)
        if unknown_execute_values:
            raise SystemExit(
                "Unknown Loom execute substrate(s): {}".format(
                    ", ".join(sorted(unknown_execute_values))
                )
            )
        self.loom_execute_source = "native"
        self.enabled_loom_execute = set(execute_values)

    def set_rocm_path(self, path: str) -> None:
        rocm_path = resolve_rocm_path(path)
        if self.rocm_path is not None and self.rocm_path != rocm_path:
            raise SystemExit(
                "Conflicting ROCm roots: {} and {}".format(self.rocm_path, rocm_path)
            )
        self.rocm_path = rocm_path

    def set_dependency_mode(self, mode: str) -> None:
        dependency_mode = mode.lower()
        if dependency_mode not in DEPENDENCY_MODES:
            raise SystemExit(
                "IREE_DEPENDENCY_MODE must be one of pinned, package, or auto; "
                f"got {mode!r}."
            )
        self.dependency_mode = dependency_mode

    def set_rocm_dependency_mode(self, mode: str) -> None:
        dependency_mode = mode.lower()
        if dependency_mode not in DEPENDENCY_MODES:
            raise SystemExit(
                "IREE_ROCM_DEPENDENCY_MODE must be one of pinned, package, or auto; "
                f"got {mode!r}."
            )
        self.rocm_dependency_mode = dependency_mode

    def effective_rocm_dependency_mode(self) -> str:
        if self.rocm_dependency_mode is not None:
            return self.rocm_dependency_mode
        if self.rocm_path is not None:
            return "package"
        return self.dependency_mode


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Writes .bazelrc.configured for this checkout. Defaults to a "
            "SDK-free recursive build scope."
        ),
        epilog="""Examples:
  python build_tools/bazel/configure.py
  python build_tools/bazel/configure.py -DIREE_HAL_DRIVER_AMDGPU=ON
  python build_tools/bazel/configure.py -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
  python build_tools/bazel/configure.py -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_DEPENDENCY_MODE=pinned
  python build_tools/bazel/configure.py --//runtime/config/hal:drivers=amdgpu,local-sync,local-task,null --repo_env=IREE_ROCM_PATH=/opt/rocm
  python build_tools/bazel/configure.py -DLOOM_TARGET_SPIRV=OFF
  python build_tools/bazel/configure.py -DLOOM_TARGET_AMDGPU=ON -DLOOM_EXECUTE_IREE_HAL=ON -DIREE_HAL_DRIVER_AMDGPU=ON -DIREE_ROCM_PATH=/opt/rocm
  python build_tools/bazel/configure.py -DLOOM_TARGET_AMDGPU=ON -DLOOM_EMIT_LLVMIR=ON

Portable -D project options are documented in BUILDING.md. Other Bazel-native
overrides belong in .bazelrc.local.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".bazelrc.configured"),
        help="Path to write. Defaults to .bazelrc.configured.",
    )
    parser.add_argument(
        "-D",
        action="append",
        default=[],
        dest="defines",
        metavar="NAME=VALUE",
        help="Portable project configuration option documented in BUILDING.md.",
    )
    args, native_bazel_args = parser.parse_known_args(argv)
    args.native_bazel_args = native_bazel_args
    return args


def option_name(arg: str) -> str:
    return arg.split("=", 1)[0]


def check_removed_option(arg: str) -> None:
    removed_option = option_name(arg)
    if removed_option not in REMOVED_OPTIONS:
        return
    raise SystemExit(
        "{} was removed. Use portable -DIREE_HAL_DRIVER_* options and "
        "-DIREE_ROCM_PATH=/path/to/rocm when package-mode ROCm headers are "
        "required. The native Bazel form is "
        "{}=amdgpu,local-sync,local-task,null "
        "--repo_env=IREE_ROCM_PATH=/path/to/rocm.".format(
            removed_option, NATIVE_DRIVER_FLAG
        )
    )


def parse_bool(name: str, value: str) -> bool:
    normalized_value = value.upper()
    if normalized_value in TRUE_VALUES:
        return True
    if normalized_value in FALSE_VALUES:
        return False
    raise SystemExit(f"{name} expects ON or OFF, got {value!r}.")


def parse_define(define: str) -> tuple[str, str]:
    if "=" not in define:
        raise SystemExit(f"-D{define} must use NAME=VALUE.")
    name, value = define.split("=", 1)
    name = name.split(":", 1)[0]
    if not name:
        raise SystemExit(f"-D{define} has no option name.")
    return name, value


def resolve_rocm_path(path: str) -> str:
    if not path:
        raise SystemExit("IREE_ROCM_PATH must not be empty.")
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"IREE_ROCM_PATH does not exist: {resolved}")
    if not (resolved / "include").is_dir():
        raise SystemExit(f"IREE_ROCM_PATH has no include directory: {resolved}")
    return str(resolved)


def apply_define(request: ConfigRequest, define: str) -> None:
    name, value = parse_define(define)
    if name in DRIVER_DEFINES:
        request.set_driver(DRIVER_DEFINES[name], parse_bool(name, value))
        return
    if name in UNSUPPORTED_DRIVER_DEFINES:
        if parse_bool(name, value):
            raise SystemExit(f"{name}=ON is not configured in the Bazel graph yet.")
        return
    if name in LOOM_TARGET_DEFINES:
        request.set_loom_target(LOOM_TARGET_DEFINES[name], parse_bool(name, value))
        return
    if name in LOOM_EMIT_DEFINES:
        request.set_loom_emitter(LOOM_EMIT_DEFINES[name], parse_bool(name, value))
        return
    if name in LOOM_EXECUTE_DEFINES:
        request.set_loom_execute(LOOM_EXECUTE_DEFINES[name], parse_bool(name, value))
        return
    if name == "IREE_ROCM_PATH":
        request.set_rocm_path(value)
        return
    if name == "IREE_DEPENDENCY_MODE":
        request.set_dependency_mode(value)
        return
    if name == "IREE_ROCM_DEPENDENCY_MODE":
        request.set_rocm_dependency_mode(value)
        return
    raise SystemExit(
        f"Unsupported Bazel configure option -D{name}. Published portable "
        "configuration options are documented in BUILDING.md."
    )


def parse_driver_list(value: str) -> set[str]:
    if not value:
        return set()
    return {driver for driver in value.split(",") if driver}


def parse_string_list(value: str) -> set[str]:
    if not value:
        return set()
    return {part for part in value.split(",") if part}


def apply_native_bazel_arg(request: ConfigRequest, arg: str) -> None:
    check_removed_option(arg)
    if arg == "--":
        raise SystemExit("build_tools/bazel/configure.py does not use a -- separator.")
    if arg.startswith(NATIVE_DRIVER_FLAG + "="):
        request.set_driver_list(parse_driver_list(arg.split("=", 1)[1]))
        return
    if arg == NATIVE_DRIVER_FLAG:
        raise SystemExit(f"{NATIVE_DRIVER_FLAG} must use --flag=value syntax.")
    if arg.startswith(NATIVE_LOOM_TARGET_FLAG + "="):
        request.set_loom_target_list(parse_string_list(arg.split("=", 1)[1]))
        return
    if arg == NATIVE_LOOM_TARGET_FLAG:
        raise SystemExit(f"{NATIVE_LOOM_TARGET_FLAG} must use --flag=value syntax.")
    if arg.startswith(NATIVE_LOOM_EMIT_FLAG + "="):
        request.set_loom_emitter_list(parse_string_list(arg.split("=", 1)[1]))
        return
    if arg == NATIVE_LOOM_EMIT_FLAG:
        raise SystemExit(f"{NATIVE_LOOM_EMIT_FLAG} must use --flag=value syntax.")
    if arg.startswith(NATIVE_LOOM_EXECUTE_FLAG + "="):
        request.set_loom_execute_list(parse_string_list(arg.split("=", 1)[1]))
        return
    if arg == NATIVE_LOOM_EXECUTE_FLAG:
        raise SystemExit(f"{NATIVE_LOOM_EXECUTE_FLAG} must use --flag=value syntax.")
    if arg.startswith(NATIVE_REPO_ENV_PREFIX):
        repo_env = arg[len(NATIVE_REPO_ENV_PREFIX) :]
        if "=" not in repo_env:
            raise SystemExit("--repo_env must use --repo_env=NAME=VALUE syntax.")
        name, value = repo_env.split("=", 1)
        if name == "IREE_ROCM_PATH":
            request.set_rocm_path(value)
            return
        if name == "IREE_DEPENDENCY_MODE":
            request.set_dependency_mode(value)
            return
        if name == "IREE_ROCM_DEPENDENCY_MODE":
            request.set_rocm_dependency_mode(value)
            return
        raise SystemExit(
            "build_tools/bazel/configure.py only accepts "
            "--repo_env=IREE_ROCM_PATH=..., "
            "--repo_env=IREE_DEPENDENCY_MODE=..., and "
            "--repo_env=IREE_ROCM_DEPENDENCY_MODE=... "
            "for generated configuration. Put other Bazel-native overrides "
            "in .bazelrc.local."
        )
    raise SystemExit(
        f"Unsupported Bazel configure argument {arg!r}. Use documented -D "
        "project options, the native runtime driver build setting, or "
        ".bazelrc.local for checkout-specific Bazel overrides."
    )


def request_from_args(args: argparse.Namespace) -> ConfigRequest:
    request = ConfigRequest(output=args.output)
    for define in args.defines:
        apply_define(request, define)
    for native_bazel_arg in args.native_bazel_args:
        apply_native_bazel_arg(request, native_bazel_arg)
    return request


def ordered_driver_set(values: set[str]) -> list[str]:
    return [driver for driver in ALL_DRIVERS if driver in values]


def ordered_loom_target_set(values: set[str]) -> list[str]:
    return [target for target in LOOM_TARGETS if target in values]


def ordered_loom_emitter_set(values: set[str]) -> list[str]:
    return [emitter for emitter in LOOM_EMITTERS if emitter in values]


def ordered_loom_execute_set(values: set[str]) -> list[str]:
    return [execute for execute in LOOM_EXECUTE_SUBSTRATES if execute in values]


def bazelrc_line(command: str, option: str) -> str:
    return f"{command} {shlex.quote(option)}"


def generate_config(args: argparse.Namespace) -> str:
    request = request_from_args(args)
    unsupported_enabled_drivers = request.enabled_drivers.difference(
        SUPPORTED_ENABLE_DRIVERS
    )
    if unsupported_enabled_drivers:
        raise SystemExit(
            "Driver enablement is not yet configured for: {}. "
            "Leave these drivers disabled until their repositories have been "
            "ported.".format(", ".join(sorted(unsupported_enabled_drivers)))
        )

    rocm_enabled_drivers = request.enabled_drivers.intersection(ROCM_DRIVERS)
    if rocm_enabled_drivers and not request.rocm_path:
        rocm_path = os.environ.get("IREE_ROCM_PATH")
        if rocm_path:
            request.set_rocm_path(rocm_path)

    effective_rocm_dependency_mode = request.effective_rocm_dependency_mode()
    if (
        rocm_enabled_drivers
        and effective_rocm_dependency_mode == "package"
        and not request.rocm_path
    ):
        raise SystemExit(
            "ROCm package dependency mode requires -DIREE_ROCM_PATH=/path/to/rocm, "
            "--repo_env=IREE_ROCM_PATH=/path/to/rocm, or IREE_ROCM_PATH in the "
            "environment when enabling {}. Set IREE_ROCM_DEPENDENCY_MODE=pinned "
            "or auto to use pinned ROCm headers without a ROCm root.".format(
                ", ".join(sorted(rocm_enabled_drivers))
            )
        )

    lines = [
        "# AUTO-GENERATED by build_tools/bazel/configure.py; do not edit.",
        "# Re-run python build_tools/bazel/configure.py or use .bazelrc.local for local overrides.",
        "",
        "# Runtime driver registry scope.",
        bazelrc_line(
            "build",
            "--//runtime/config/hal:drivers="
            + ",".join(ordered_driver_set(request.enabled_drivers)),
        ),
        "",
        "# Source dependency mode.",
        bazelrc_line(
            "common",
            "--repo_env=IREE_DEPENDENCY_MODE=" + request.dependency_mode,
        ),
        "",
        "# Loom target, execute substrate, and explicit debug emitter scope.",
        bazelrc_line(
            "build",
            "--//loom/config/target:enable="
            + ",".join(ordered_loom_target_set(request.enabled_loom_targets)),
        ),
        bazelrc_line(
            "build",
            "--//loom/config/execute:enable="
            + ",".join(ordered_loom_execute_set(request.enabled_loom_execute)),
        ),
        bazelrc_line(
            "build",
            "--//loom/config/emit:enable="
            + ",".join(ordered_loom_emitter_set(request.enabled_loom_emitters)),
        ),
        "",
    ]
    if request.rocm_dependency_mode or request.rocm_path:
        lines.append(
            bazelrc_line(
                "common",
                "--repo_env=IREE_ROCM_DEPENDENCY_MODE="
                + effective_rocm_dependency_mode,
            )
        )
    lines.extend(
        [
            "",
            "# Optional AMDGPU device compiler toolchain.",
        ]
    )
    if "amdgpu" in request.enabled_drivers and request.rocm_path:
        lines.append(
            bazelrc_line(
                "common",
                "--repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm",
            )
        )
    else:
        lines.append(
            bazelrc_line(
                "common",
                "--repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=none",
            )
        )

    if request.rocm_path:
        lines.extend(
            [
                bazelrc_line(
                    "common", "--repo_env=IREE_ROCM_PATH=" + request.rocm_path
                ),
                "",
            ]
        )
    else:
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_arguments()
    config = generate_config(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(config, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
