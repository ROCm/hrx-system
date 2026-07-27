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
    label = "//loom/requirements:emit_amdgpu",
    enabled_by = "//loom/config/emit:amdgpu",
    cmake_condition = "LOOM_EMIT_AMDGPU",
)

EMIT_IREE_VM = build_requirement(
    id = "loom.emit.iree_vm",
    label = "//loom/requirements:emit_iree_vm",
    enabled_by = "//loom/config/emit:iree_vm",
    cmake_condition = "LOOM_EMIT_IREE_VM",
)

EMIT_LLVMIR = build_requirement(
    id = "loom.emit.llvmir",
    label = "//loom/requirements:emit_llvmir",
    enabled_by = "//loom/config/emit:llvmir",
    cmake_condition = "LOOM_EMIT_LLVMIR",
)

EMIT_SPIRV = build_requirement(
    id = "loom.emit.spirv",
    label = "//loom/requirements:emit_spirv",
    enabled_by = "//loom/config/emit:spirv",
    cmake_condition = "LOOM_EMIT_SPIRV",
)

EMIT_WASM = build_requirement(
    id = "loom.emit.wasm",
    label = "//loom/requirements:emit_wasm",
    enabled_by = "//loom/config/emit:wasm",
    cmake_condition = "LOOM_EMIT_WASM",
)

EXECUTE_IREE_HAL = build_requirement(
    id = "loom.execute.iree_hal",
    label = "//loom/requirements:execute_iree_hal",
    enabled_by = "//loom/config/execute:iree_hal",
    cmake_condition = "LOOM_EXECUTE_IREE_HAL",
)

EXECUTE_IREE_VM = build_requirement(
    id = "loom.execute.iree_vm",
    label = "//loom/requirements:execute_iree_vm",
    enabled_by = "//loom/config/execute:iree_vm",
    cmake_condition = "LOOM_EXECUTE_IREE_VM",
)

IMPORT_MLIR = build_requirement(
    id = "loom.import.mlir",
    label = "//loom/requirements:import_mlir",
    enabled_by = "//loom/config/import:mlir",
    cmake_condition = "LOOM_IMPORT_MLIR",
)

IMPORT_TILELANG = build_requirement(
    id = "loom.import.tilelang",
    label = "//loom/requirements:import_tilelang",
    enabled_by = "//loom/config/import:tilelang",
    cmake_condition = "LOOM_IMPORT_TILELANG",
)

TARGET_ARCH_AMDGPU = build_requirement(
    id = "loom.target.arch.amdgpu",
    label = "//loom/requirements:target_arch_amdgpu",
    enabled_by = "//loom/config/target/arch:amdgpu",
    cmake_condition = "LOOM_TARGET_ARCH_AMDGPU",
)

TARGET_ARCH_IREE_VM = build_requirement(
    id = "loom.target.arch.iree_vm",
    label = "//loom/requirements:target_arch_iree_vm",
    enabled_by = "//loom/config/target/arch:iree_vm",
    cmake_condition = "LOOM_TARGET_ARCH_IREE_VM",
)

TARGET_ARCH_LLVMIR = build_requirement(
    id = "loom.target.arch.llvmir",
    label = "//loom/requirements:target_arch_llvmir",
    enabled_by = "//loom/config/target/arch:llvmir",
    cmake_condition = "LOOM_TARGET_ARCH_LLVMIR",
)

TARGET_ARCH_SPIRV = build_requirement(
    id = "loom.target.arch.spirv",
    label = "//loom/requirements:target_arch_spirv",
    enabled_by = "//loom/config/target/arch:spirv",
    cmake_condition = "LOOM_TARGET_ARCH_SPIRV",
)

TARGET_ARCH_WASM = build_requirement(
    id = "loom.target.arch.wasm",
    label = "//loom/requirements:target_arch_wasm",
    enabled_by = "//loom/config/target/arch:wasm",
    cmake_condition = "LOOM_TARGET_ARCH_WASM",
)

TARGET_ARCH_X86 = build_requirement(
    id = "loom.target.arch.x86",
    label = "//loom/requirements:target_arch_x86",
    enabled_by = "//loom/config/target/arch:x86",
    cmake_condition = "LOOM_TARGET_ARCH_X86",
)

REQUIREMENTS = [
    EMIT_AMDGPU,
    EMIT_IREE_VM,
    EMIT_LLVMIR,
    EMIT_SPIRV,
    EMIT_WASM,
    EXECUTE_IREE_HAL,
    EXECUTE_IREE_VM,
    IMPORT_MLIR,
    IMPORT_TILELANG,
    TARGET_ARCH_AMDGPU,
    TARGET_ARCH_IREE_VM,
    TARGET_ARCH_LLVMIR,
    TARGET_ARCH_SPIRV,
    TARGET_ARCH_WASM,
    TARGET_ARCH_X86,
]
