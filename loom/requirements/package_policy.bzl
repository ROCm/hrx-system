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
    "//loom/requirements:defs.bzl",
    "EMIT_AMDGPU",
    "EMIT_LLVMIR",
    "EMIT_SPIRV",
    "EMIT_WASM",
    "IMPORT_MLIR",
    "IMPORT_TILELANG",
    "TARGET_ARCH_AMDGPU",
    "TARGET_ARCH_LLVMIR",
    "TARGET_ARCH_SPIRV",
    "TARGET_ARCH_WASM",
    "TARGET_ARCH_X86",
)

PACKAGE_POLICIES = [
    package_policy(
        packages = ["loom/src/loom/target/arch/amdgpu/..."],
        build_requirements = [TARGET_ARCH_AMDGPU],
    ),
    package_policy(
        packages = ["loom/src/loom/target/arch/amdgpu/hal"],
        forbidden_deps = ["//runtime/src/iree/hal/drivers/amdgpu/..."],
    ),
    package_policy(
        packages = ["loom/src/loom/target/arch/llvmir/..."],
        build_requirements = [TARGET_ARCH_LLVMIR],
    ),
    package_policy(
        packages = ["loom/src/loom/target/arch/spirv/..."],
        build_requirements = [TARGET_ARCH_SPIRV],
    ),
    package_policy(
        packages = ["loom/src/loom/target/arch/wasm/..."],
        build_requirements = [TARGET_ARCH_WASM],
    ),
    package_policy(
        packages = ["loom/src/loom/target/arch/x86/..."],
        build_requirements = [TARGET_ARCH_X86],
    ),
    package_policy(
        packages = ["loom/src/loom/target/emit/llvmir/..."],
        build_requirements = [TARGET_ARCH_LLVMIR, EMIT_LLVMIR],
    ),
    package_policy(
        packages = ["loom/src/loom/target/emit/llvmir/amdgpu/..."],
        build_requirements = [TARGET_ARCH_AMDGPU],
    ),
    package_policy(
        packages = ["loom/src/loom/target/emit/llvmir/x86/..."],
        build_requirements = [TARGET_ARCH_X86],
    ),
    package_policy(
        packages = ["loom/src/loom/target/emit/native/amdgpu/..."],
        build_requirements = [TARGET_ARCH_AMDGPU, EMIT_AMDGPU],
        forbidden_deps = ["//runtime/src/iree/hal/drivers/amdgpu/..."],
    ),
    package_policy(
        packages = ["loom/src/loom/target/emit/native/x86/..."],
        build_requirements = [TARGET_ARCH_X86],
    ),
    package_policy(
        packages = ["loom/src/loom/target/emit/spirv/..."],
        build_requirements = [TARGET_ARCH_SPIRV, EMIT_SPIRV],
    ),
    package_policy(
        packages = ["loom/src/loom/target/emit/wasm/..."],
        build_requirements = [TARGET_ARCH_WASM, EMIT_WASM],
    ),
    package_policy(
        packages = ["loom/src/loom/tooling/target/amdgpu/..."],
        build_requirements = [
            TARGET_ARCH_AMDGPU,
            EMIT_AMDGPU,
        ],
    ),
    package_policy(
        packages = ["loom/src/loom/tooling/target/spirv/..."],
        build_requirements = [
            TARGET_ARCH_SPIRV,
            EMIT_SPIRV,
        ],
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
        packages = ["loom/binding/c/benchmark/target/spirv/..."],
        build_requirements = [TARGET_ARCH_SPIRV, EMIT_SPIRV],
    ),
    package_policy(
        packages = ["loom/binding/c/test/target/spirv/..."],
        build_requirements = [TARGET_ARCH_SPIRV, EMIT_SPIRV],
    ),
]

def _current_policy():
    return collect_package_policy(native.package_name(), PACKAGE_POLICIES)

def apply_loom_target_policy(kwargs, name = None):
    return apply_target_policy(kwargs, _current_policy(), name = name)

def apply_loom_test_policy(kwargs, name = None):
    return apply_test_policy(kwargs, _current_policy(), name = name)
