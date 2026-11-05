# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom package policy."""

load(
    "//build_tools/bazel:package_policy.bzl",
    "apply_target_policy",
    "apply_test_policy",
    "collect_package_policy",
    "package_policy",
)
load(
    "//build_tools/vulkan/requirements:defs.bzl",
    "VULKAN_API",
)
load(
    "//loom/requirements:defs.bzl",
    "ARCH_AIE2P",
    "ARCH_AMDGPU",
    "ARCH_SPIRV",
    "ARCH_VM",
    "ARCH_WASM",
    "ARCH_X86",
    "ARTIFACT_AIE2P_ELF",
    "ARTIFACT_AMDGPU_ELF",
    "ARTIFACT_SPIRV_BINARY",
    "ARTIFACT_VM_BINARY",
    "ARTIFACT_WASM_BINARY",
    "ARTIFACT_X86_ELF",
    "IMPORT_CXX",
    "IMPORT_MLIR",
    "IMPORT_TILELANG",
    "NATIVE_EMISSION",
)

PACKAGE_POLICIES = [
    package_policy(
        packages = ["loom/..."],
        forbidden_deps = ["//runtime/src/iree/hal/drivers/amdgpu/..."],
    ),
    package_policy(
        packages = ["loom/src/loom/runtime/..."],
        forbidden_deps = [
            "//loom/src/loom/serve/...",
            "//loom/src/loom/service/...",
        ],
    ),
    package_policy(
        packages = ["loom/src/loom/service/..."],
        forbidden_deps = ["//loom/src/loom/serve/..."],
    ),
    package_policy(
        packages = ["loom/binding/c/target/spirv/vulkan/..."],
        build_requirements = [ARCH_SPIRV, VULKAN_API],
    ),
    package_policy(
        packages = [
            "loom/binding/c/import/cxx/...",
            "loom/src/loom/import/cxx/...",
            "loom/src/loom/tools/loom-import-cxx/...",
        ],
        build_requirements = [IMPORT_CXX],
    ),
    package_policy(
        packages = [
            "loom/src/loom/serve/...",
            "loom/src/loom/tooling/target/amd/xdna/serve/...",
            "loom/src/loom/tooling/target/amdgpu/serve/...",
            "loom/src/loom/tooling/target/cpu/serve/...",
            "loom/src/loom/tooling/target/spirv/serve/...",
            "loom/src/loom/tools/loom-serve/...",
        ],
        build_requirements = [ARCH_VM, ARTIFACT_VM_BINARY],
    ),
    package_policy(
        packages = ["loom/src/loom/target/arch/amd/xdna/..."],
        build_requirements = [ARCH_AIE2P],
    ),
    package_policy(
        packages = ["loom/src/loom/target/arch/amdgpu/..."],
        build_requirements = [ARCH_AMDGPU],
    ),
    package_policy(
        packages = ["loom/src/loom/target/arch/spirv/..."],
        build_requirements = [ARCH_SPIRV],
    ),
    package_policy(
        packages = ["loom/src/loom/target/arch/vm/..."],
        build_requirements = [ARCH_VM],
    ),
    package_policy(
        packages = ["loom/src/loom/target/arch/wasm/..."],
        build_requirements = [ARCH_WASM],
    ),
    package_policy(
        packages = ["loom/src/loom/target/arch/x86/..."],
        build_requirements = [ARCH_X86],
    ),
    package_policy(
        packages = ["loom/src/loom/target/emit/native"],
        build_requirements = [NATIVE_EMISSION],
    ),
    package_policy(
        packages = ["loom/src/loom/target/emit/native/amdgpu/..."],
        build_requirements = [ARCH_AMDGPU, ARTIFACT_AMDGPU_ELF],
    ),
    package_policy(
        packages = ["loom/src/loom/target/emit/native/aie2p/..."],
        build_requirements = [ARCH_AIE2P, ARTIFACT_AIE2P_ELF],
    ),
    package_policy(
        packages = ["loom/src/loom/target/emit/native/x86/..."],
        build_requirements = [ARCH_X86, ARTIFACT_X86_ELF],
    ),
    package_policy(
        packages = ["loom/src/loom/target/emit/spirv/..."],
        build_requirements = [ARCH_SPIRV, ARTIFACT_SPIRV_BINARY],
    ),
    package_policy(
        packages = ["loom/src/loom/target/emit/vm/..."],
        build_requirements = [ARCH_VM, ARTIFACT_VM_BINARY],
    ),
    package_policy(
        packages = ["loom/src/loom/target/emit/wasm/..."],
        build_requirements = [ARCH_WASM, ARTIFACT_WASM_BINARY],
    ),
    package_policy(
        packages = ["loom/src/loom/tooling/target/amd/xdna/..."],
        build_requirements = [ARCH_AIE2P, ARTIFACT_AIE2P_ELF],
    ),
    package_policy(
        packages = ["loom/src/loom/tooling/target/amdgpu/..."],
        build_requirements = [
            ARCH_AMDGPU,
            ARTIFACT_AMDGPU_ELF,
        ],
    ),
    package_policy(
        packages = ["loom/src/loom/tooling/target/cpu/..."],
        build_requirements = [ARCH_X86, ARTIFACT_X86_ELF],
    ),
    package_policy(
        packages = ["loom/src/loom/tooling/target/spirv/..."],
        build_requirements = [
            ARCH_SPIRV,
            ARTIFACT_SPIRV_BINARY,
        ],
    ),
    package_policy(
        packages = ["loom/src/loom/tooling/target/vm/..."],
        build_requirements = [ARCH_VM, ARTIFACT_VM_BINARY],
    ),
    package_policy(
        packages = ["loom/src/loom/tooling/target/wasm/..."],
        build_requirements = [ARCH_WASM, ARTIFACT_WASM_BINARY],
    ),
    package_policy(
        packages = ["loom/py/loom/importers/mlir/..."],
        build_requirements = [IMPORT_MLIR],
    ),
    package_policy(
        packages = ["loom/py/loom/importers/tilelang/..."],
        build_requirements = [IMPORT_TILELANG],
    ),
    package_policy(
        packages = ["loom/binding/c/benchmark/target/amdgpu/..."],
        build_requirements = [ARCH_AMDGPU, ARTIFACT_AMDGPU_ELF],
    ),
    package_policy(
        packages = ["loom/binding/c/benchmark/target/spirv/..."],
        build_requirements = [ARCH_SPIRV, ARTIFACT_SPIRV_BINARY],
    ),
    package_policy(
        packages = ["loom/binding/c/test/target/spirv/..."],
        build_requirements = [ARCH_SPIRV],
    ),
]

def _current_policy():
    return collect_package_policy(native.package_name(), PACKAGE_POLICIES)

def apply_loom_target_policy(kwargs, name = None):
    return apply_target_policy(kwargs, _current_policy(), name = name)

def apply_loom_test_policy(kwargs, name = None):
    return apply_test_policy(kwargs, _current_policy(), name = name)
