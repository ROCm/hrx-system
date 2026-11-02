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

ARTIFACT_AMDGPU_ELF = build_requirement(
    id = "loom.artifact.amdgpu_elf",
    label = Label("//loom/requirements:artifact_amdgpu_elf"),
    enabled_by = Label("//loom/config/emit:amdgpu_elf"),
)

ARTIFACT_AIE2P_ELF = build_requirement(
    id = "loom.artifact.aie2p_elf",
    label = Label("//loom/requirements:artifact_aie2p_elf"),
    enabled_by = Label("//loom/config/emit:aie2p_elf"),
)

ARTIFACT_SPIRV_BINARY = build_requirement(
    id = "loom.artifact.spirv_binary",
    label = Label("//loom/requirements:artifact_spirv_binary"),
    enabled_by = Label("//loom/config/emit:spirv_binary"),
)

ARTIFACT_VM_BINARY = build_requirement(
    id = "loom.artifact.vm_binary",
    label = Label("//loom/requirements:artifact_vm_binary"),
    enabled_by = Label("//loom/config/emit:vm_binary"),
)

ARTIFACT_WASM_BINARY = build_requirement(
    id = "loom.artifact.wasm_binary",
    label = Label("//loom/requirements:artifact_wasm_binary"),
    enabled_by = Label("//loom/config/emit:wasm_binary"),
)

ARTIFACT_X86_ELF = build_requirement(
    id = "loom.artifact.x86_elf",
    label = Label("//loom/requirements:artifact_x86_elf"),
    enabled_by = Label("//loom/config/emit:x86_elf"),
)

NATIVE_EMISSION = build_requirement(
    id = "loom.emit.native",
    label = Label("//loom/requirements:emit_native"),
    enabled_by = Label("//loom/config/emit:native"),
)

EXECUTE_IREE_HAL = build_requirement(
    id = "loom.execute.iree_hal",
    label = Label("//loom/requirements:execute_iree_hal"),
    enabled_by = Label("//loom/config/execute:iree_hal"),
)

EXECUTE_IREE_VM = build_requirement(
    id = "loom.execute.iree_vm",
    label = Label("//loom/requirements:execute_iree_vm"),
    enabled_by = Label("//loom/config/execute:iree_vm"),
)

EXECUTE_WASM_HOST = build_requirement(
    id = "loom.execute.wasm_host",
    label = Label("//loom/requirements:execute_wasm_host"),
    enabled_by = Label("//loom/config/execute:wasm_host"),
)

IMPORT_CXX = build_requirement(
    id = "loom.import.cxx",
    label = Label("//loom/requirements:import_cxx"),
    enabled_by = Label("//loom/config/import:cxx"),
)

IMPORT_MLIR = build_requirement(
    id = "loom.import.mlir",
    label = Label("//loom/requirements:import_mlir"),
    enabled_by = Label("//loom/config/import:mlir"),
)

IMPORT_TILELANG = build_requirement(
    id = "loom.import.tilelang",
    label = Label("//loom/requirements:import_tilelang"),
    enabled_by = Label("//loom/config/import:tilelang"),
)

ARCH_AMDGPU = build_requirement(
    id = "loom.arch.amdgpu",
    label = Label("//loom/requirements:arch_amdgpu"),
    enabled_by = Label("//loom/config/arch:amdgpu"),
)

ARCH_SPIRV = build_requirement(
    id = "loom.arch.spirv",
    label = Label("//loom/requirements:arch_spirv"),
    enabled_by = Label("//loom/config/arch:spirv"),
)

ARCH_WASM = build_requirement(
    id = "loom.arch.wasm",
    label = Label("//loom/requirements:arch_wasm"),
    enabled_by = Label("//loom/config/arch:wasm"),
)

ARCH_VM = build_requirement(
    id = "loom.arch.vm",
    label = Label("//loom/requirements:arch_vm"),
    enabled_by = Label("//loom/config/arch:vm"),
)

ARCH_AIE2P = build_requirement(
    id = "loom.arch.aie2p",
    label = Label("//loom/requirements:arch_aie2p"),
    enabled_by = Label("//loom/config/arch:aie2p"),
)

ARCH_X86 = build_requirement(
    id = "loom.arch.x86",
    label = Label("//loom/requirements:arch_x86"),
    enabled_by = Label("//loom/config/arch:x86"),
)

REQUIREMENTS = [
    ARCH_AIE2P,
    ARCH_AMDGPU,
    ARCH_SPIRV,
    ARCH_VM,
    ARCH_WASM,
    ARCH_X86,
    ARTIFACT_AIE2P_ELF,
    ARTIFACT_AMDGPU_ELF,
    ARTIFACT_SPIRV_BINARY,
    ARTIFACT_VM_BINARY,
    ARTIFACT_WASM_BINARY,
    ARTIFACT_X86_ELF,
    NATIVE_EMISSION,
    EXECUTE_IREE_HAL,
    EXECUTE_IREE_VM,
    EXECUTE_WASM_HOST,
    IMPORT_CXX,
    IMPORT_MLIR,
    IMPORT_TILELANG,
]
