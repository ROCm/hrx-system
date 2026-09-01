# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom build and run requirements."""

load(
    "//build_tools/bazel:requirements.bzl",
    "build_requirement",
)

EMIT_AMDGPU = build_requirement(
    id = "loom.emit.amdgpu",
    label = Label("//loom/requirements:emit_amdgpu"),
    enabled_by = Label("//loom/config/emit:amdgpu"),
    cmake_condition = "LOOM_EMIT_AMDGPU",
)

EMIT_LLVMIR = build_requirement(
    id = "loom.emit.llvmir",
    label = Label("//loom/requirements:emit_llvmir"),
    enabled_by = Label("//loom/config/emit:llvmir"),
    cmake_condition = "LOOM_EMIT_LLVMIR",
)

EMIT_SPIRV = build_requirement(
    id = "loom.emit.spirv",
    label = Label("//loom/requirements:emit_spirv"),
    enabled_by = Label("//loom/config/emit:spirv"),
    cmake_condition = "LOOM_EMIT_SPIRV",
)

EMIT_WASM = build_requirement(
    id = "loom.emit.wasm",
    label = Label("//loom/requirements:emit_wasm"),
    enabled_by = Label("//loom/config/emit:wasm"),
    cmake_condition = "LOOM_EMIT_WASM",
)

EXECUTE_IREE_HAL = build_requirement(
    id = "loom.execute.iree_hal",
    label = Label("//loom/requirements:execute_iree_hal"),
    enabled_by = Label("//loom/config/execute:iree_hal"),
    cmake_condition = "LOOM_EXECUTE_IREE_HAL",
)

IMPORT_MLIR = build_requirement(
    id = "loom.import.mlir",
    label = Label("//loom/requirements:import_mlir"),
    enabled_by = Label("//loom/config/import:mlir"),
    cmake_condition = "LOOM_IMPORT_MLIR",
)

IMPORT_TILELANG = build_requirement(
    id = "loom.import.tilelang",
    label = Label("//loom/requirements:import_tilelang"),
    enabled_by = Label("//loom/config/import:tilelang"),
    cmake_condition = "LOOM_IMPORT_TILELANG",
)

TARGET_ARCH_AMDGPU = build_requirement(
    id = "loom.target.arch.amdgpu",
    label = Label("//loom/requirements:target_arch_amdgpu"),
    enabled_by = Label("//loom/config/target/arch:amdgpu"),
    cmake_condition = "LOOM_TARGET_ARCH_AMDGPU",
)

TARGET_ARCH_LLVMIR = build_requirement(
    id = "loom.target.arch.llvmir",
    label = Label("//loom/requirements:target_arch_llvmir"),
    enabled_by = Label("//loom/config/target/arch:llvmir"),
    cmake_condition = "LOOM_TARGET_ARCH_LLVMIR",
)

TARGET_ARCH_SPIRV = build_requirement(
    id = "loom.target.arch.spirv",
    label = Label("//loom/requirements:target_arch_spirv"),
    enabled_by = Label("//loom/config/target/arch:spirv"),
    cmake_condition = "LOOM_TARGET_ARCH_SPIRV",
)

TARGET_ARCH_VM = build_requirement(
    id = "loom.target.arch.vm",
    label = Label("//loom/requirements:target_arch_vm"),
    enabled_by = Label("//loom/config/target/arch:vm"),
    cmake_condition = "LOOM_TARGET_ARCH_VM",
)

TARGET_ARCH_WASM = build_requirement(
    id = "loom.target.arch.wasm",
    label = Label("//loom/requirements:target_arch_wasm"),
    enabled_by = Label("//loom/config/target/arch:wasm"),
    cmake_condition = "LOOM_TARGET_ARCH_WASM",
)

TARGET_ARCH_X86 = build_requirement(
    id = "loom.target.arch.x86",
    label = Label("//loom/requirements:target_arch_x86"),
    enabled_by = Label("//loom/config/target/arch:x86"),
    cmake_condition = "LOOM_TARGET_ARCH_X86",
)

REQUIREMENTS = [
    EMIT_AMDGPU,
    EMIT_LLVMIR,
    EMIT_SPIRV,
    EMIT_WASM,
    EXECUTE_IREE_HAL,
    IMPORT_MLIR,
    IMPORT_TILELANG,
    TARGET_ARCH_AMDGPU,
    TARGET_ARCH_LLVMIR,
    TARGET_ARCH_SPIRV,
    TARGET_ARCH_VM,
    TARGET_ARCH_WASM,
    TARGET_ARCH_X86,
]
