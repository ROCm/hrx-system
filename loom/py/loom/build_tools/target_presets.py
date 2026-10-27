# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom target capability presets used by repository build configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

ARCH_DEFINES = {
    "LOOM_ARCH_AIE2P": "aie2p",
    "LOOM_ARCH_AMDGPU": "amdgpu",
    "LOOM_ARCH_SPIRV": "spirv",
    "LOOM_ARCH_VM": "vm",
    "LOOM_ARCH_WASM": "wasm",
    "LOOM_ARCH_X86": "x86",
}
EMIT_DEFINES = {
    "LOOM_EMIT_ELF": "elf",
    "LOOM_EMIT_SPIRV": "spirv",
    "LOOM_EMIT_VM": "vm",
    "LOOM_EMIT_WASM": "wasm",
}
EXECUTE_DEFINES = {
    "LOOM_EXECUTE_IREE_HAL": "iree_hal",
    "LOOM_EXECUTE_IREE_VM": "iree_vm",
    "LOOM_EXECUTE_WASM_HOST": "wasm_host",
}
ARCHITECTURES = tuple(ARCH_DEFINES.values())
ARTIFACT_FORMATS = tuple(EMIT_DEFINES.values())
EXECUTION_SUBSTRATES = tuple(EXECUTE_DEFINES.values())


@dataclass(frozen=True)
class TargetCapabilities:
    architectures: tuple[str, ...] = ()  # Compiler architecture providers.
    artifacts: tuple[str, ...] = ()  # Emitted artifact formats.
    execution: tuple[str, ...] = ()  # In-process execution substrates.


@dataclass(frozen=True)
class TargetPreset:
    key: str  # Stable internal and CMake key.
    option: str  # Portable -D option name.
    description: str  # User-facing generated documentation.
    includes: tuple[str, ...] = ()  # Transitively included preset keys.
    # Capabilities contributed directly by this preset.
    capabilities: TargetCapabilities = field(default_factory=TargetCapabilities)

    @property
    def bazel_config(self) -> str:
        return "loom-target-" + self.key.replace("_", "-")


TARGET_PRESETS = (
    TargetPreset(
        key="vm",
        option="LOOM_TARGET_ALL_VM",
        description="Build Loom VM compilation and execution support.",
        capabilities=TargetCapabilities(("vm",), ("vm",), ("iree_vm",)),
    ),
    TargetPreset(
        key="cpu",
        option="LOOM_TARGET_ALL_CPU",
        description="Build Loom CPU and supporting VM capabilities.",
        includes=("vm",),
        capabilities=TargetCapabilities(("x86",), ("elf",), ("iree_hal",)),
    ),
    TargetPreset(
        key="amd_gpu",
        option="LOOM_TARGET_ALL_AMD_GPU",
        description="Build Loom AMD GPU and supporting VM capabilities.",
        includes=("vm",),
        capabilities=TargetCapabilities(("amdgpu",), ("elf",), ("iree_hal",)),
    ),
    TargetPreset(
        key="amd_npu",
        option="LOOM_TARGET_ALL_AMD_NPU",
        description="Build Loom AMD NPU and supporting VM capabilities.",
        includes=("vm",),
        capabilities=TargetCapabilities(("aie2p",), ("elf",), ("iree_hal",)),
    ),
    TargetPreset(
        key="amd",
        option="LOOM_TARGET_ALL_AMD",
        description="Build Loom AMD GPU, NPU, and supporting VM capabilities.",
        includes=("amd_gpu", "amd_npu"),
    ),
    TargetPreset(
        key="spirv",
        option="LOOM_TARGET_ALL_SPIRV",
        description="Build Loom SPIR-V and supporting VM capabilities.",
        includes=("vm",),
        capabilities=TargetCapabilities(("spirv",), ("spirv",), ("iree_hal",)),
    ),
    TargetPreset(
        key="wasm",
        option="LOOM_TARGET_ALL_WASM",
        description="Build Loom WebAssembly and supporting VM capabilities.",
        includes=("vm",),
        capabilities=TargetCapabilities(("wasm",), ("wasm",), ("wasm_host",)),
    ),
    TargetPreset(
        key="default",
        option="LOOM_TARGET_DEFAULT",
        description="Build the ordinary non-WebAssembly Loom target set.",
        includes=("cpu", "amd", "spirv"),
    ),
    TargetPreset(
        key="all",
        option="LOOM_TARGET_ALL",
        description="Build every Loom target capability.",
        includes=("default", "wasm"),
    ),
)

PRESETS_BY_KEY = {preset.key: preset for preset in TARGET_PRESETS}
PRESETS_BY_OPTION = {preset.option: preset for preset in TARGET_PRESETS}
PRESETS_BY_BAZEL_CONFIG = {preset.bazel_config: preset for preset in TARGET_PRESETS}


def _ordered(values: set[str], order: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(value for value in order if value in values)


def resolve_presets(keys: set[str] | tuple[str, ...]) -> TargetCapabilities:
    architectures: set[str] = set()
    artifacts: set[str] = set()
    execution: set[str] = set()
    resolved: set[str] = set()
    active: set[str] = set()

    def add(key: str) -> None:
        if key in resolved:
            return
        if key in active:
            raise ValueError(f"Loom target preset cycle at {key}")
        preset = PRESETS_BY_KEY.get(key)
        if preset is None:
            raise ValueError(f"unknown Loom target preset: {key}")
        active.add(key)
        for included_key in preset.includes:
            add(included_key)
        active.remove(key)
        architectures.update(preset.capabilities.architectures)
        artifacts.update(preset.capabilities.artifacts)
        execution.update(preset.capabilities.execution)
        resolved.add(key)

    for key in keys:
        add(key)
    return TargetCapabilities(
        _ordered(architectures, ARCHITECTURES),
        _ordered(artifacts, ARTIFACT_FORMATS),
        _ordered(execution, EXECUTION_SUBSTRATES),
    )


def _validate_presets() -> None:
    if len(PRESETS_BY_KEY) != len(TARGET_PRESETS):
        raise ValueError("duplicate Loom target preset key")
    if len(PRESETS_BY_OPTION) != len(TARGET_PRESETS):
        raise ValueError("duplicate Loom target preset option")
    if len(PRESETS_BY_BAZEL_CONFIG) != len(TARGET_PRESETS):
        raise ValueError("duplicate Loom target preset Bazel config")
    for preset in TARGET_PRESETS:
        unknown_includes = set(preset.includes).difference(PRESETS_BY_KEY)
        unknown_architectures = set(preset.capabilities.architectures).difference(
            ARCHITECTURES
        )
        unknown_artifacts = set(preset.capabilities.artifacts).difference(
            ARTIFACT_FORMATS
        )
        unknown_execution = set(preset.capabilities.execution).difference(
            EXECUTION_SUBSTRATES
        )
        if unknown_includes:
            raise ValueError(
                f"Loom target preset {preset.key} includes unknown presets: "
                + ", ".join(sorted(unknown_includes))
            )
        if unknown_architectures or unknown_artifacts or unknown_execution:
            raise ValueError(
                f"Loom target preset {preset.key} has unknown capabilities"
            )
        resolve_presets((preset.key,))


_validate_presets()
