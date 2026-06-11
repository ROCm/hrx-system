# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Configuration for source-tree IREE CI command groups."""

from __future__ import annotations

import re
from dataclasses import dataclass

# This file is the CI policy map. Entries here should describe workflow
# boundaries: package roots, target patterns, requirement labels, resource
# slices, and named xfail groups. Individual test targets belong near the test;
# hardware execution joins CI through run-requirement tags and CTest labels.

IREE_TARGET_DIRECTORIES = ("runtime", "loom")

# ASAN, UBSAN, and TSAN run tests. MSAN builds stay useful, but running tests
# requires an instrumented host dependency stack that the CI images do not yet
# provide.
SANITIZER_TEST_CONFIGS = ("asan", "ubsan", "tsan")
SANITIZER_BUILD_CONFIGS = ("msan",)

CMAKE_SANITIZER_SMOKE_TEST_BUILD_TARGETS = (
    "iree::base::status_test",
    "loom::format::bytecode::varint_test",
)
CMAKE_SANITIZER_SMOKE_LIBRARY_BUILD_TARGETS = (
    "iree::base",
    "loom::format::bytecode::varint",
)
CMAKE_SANITIZER_SMOKE_CTEST_REGEXES = (
    "^iree/base/status_test$",
    "^loom/format/bytecode/varint_test$",
)


@dataclass(frozen=True)
class TestXfail:
    bazel_pattern: str | None = None
    ctest_regex: str | None = None


def bazel_xfail(pattern: str, *, ctest_regex: str | None = None) -> TestXfail:
    if pattern.startswith("-"):
        raise ValueError("bazel xfail patterns must omit the leading '-'")
    return TestXfail(
        bazel_pattern=pattern,
        ctest_regex=ctest_regex or bazel_pattern_to_ctest_regex(pattern),
    )


def ctest_xfail(regex: str) -> TestXfail:
    return TestXfail(ctest_regex=regex)


def bazel_pattern_to_ctest_regex(pattern: str) -> str:
    path = None
    for bazel_prefix, ctest_prefix in (
        ("//runtime/src/", ""),
        ("//loom/src/", ""),
        ("//loom/", "loom/"),
    ):
        if pattern.startswith(bazel_prefix):
            path = ctest_prefix + pattern.removeprefix(bazel_prefix)
            break
    if path is None:
        raise ValueError(f"cannot map Bazel pattern to CTest name: {pattern}")

    if path.endswith("/..."):
        ctest_prefix = path.removesuffix("/...")
        return "^" + re.escape(ctest_prefix) + "/"

    if ":" not in path:
        raise ValueError(f"expected exact Bazel test label or ... pattern: {pattern}")
    package_path, target_name = path.split(":", 1)
    return "^" + re.escape(f"{package_path}/{target_name}") + "$"


def bazel_loom_src_label_to_cmake_target(label: str) -> str:
    if not label.startswith("//loom/src/loom/") or ":" not in label:
        raise ValueError(f"expected exact Loom src Bazel label: {label}")
    package_path, target_name = label.removeprefix("//loom/src/loom/").split(":", 1)
    return "loom_" + package_path.replace("/", "_") + "_" + target_name


def bazel_xfail_targets(xfails: tuple[TestXfail, ...]) -> tuple[str, ...]:
    return tuple(
        f"-{xfail.bazel_pattern}" for xfail in xfails if xfail.bazel_pattern is not None
    )


def ctest_exclude_regex(xfails: tuple[TestXfail, ...]) -> str:
    return "|".join(
        f"({xfail.ctest_regex})" for xfail in xfails if xfail.ctest_regex is not None
    )


CPU_XFAILS = (bazel_xfail("//runtime/src/iree/hal/local/elf:elf_module_test"),)
CPU_SANITIZERS_XFAILS = (
    bazel_xfail("//runtime/src/iree/async/platform/io_uring/cts/..."),
    bazel_xfail("//runtime/src/iree/hal:string_util_test"),
    bazel_xfail("//runtime/src/iree/hal/local/elf/..."),
    bazel_xfail("//runtime/src/iree/hal/local:profile_test"),
    bazel_xfail("//runtime/src/iree/hal/replay:execute_test"),
    bazel_xfail("//runtime/src/iree/io/formats/gguf:gguf_parser_test"),
    bazel_xfail("//runtime/src/iree/tokenizer/..."),
    bazel_xfail("//runtime/src/iree/tooling/profile:cli_test"),
    bazel_xfail("//runtime/src/iree/vm:list_test"),
)
CPU_XFAIL_TARGETS = bazel_xfail_targets(CPU_XFAILS)
CPU_CTEST_EXCLUDE_REGEX = ctest_exclude_regex(CPU_XFAILS)
CPU_SANITIZERS_XFAIL_TARGETS = bazel_xfail_targets(CPU_SANITIZERS_XFAILS)
CPU_SANITIZERS_CTEST_EXCLUDE_REGEX = ctest_exclude_regex(CPU_SANITIZERS_XFAILS)
CPU_BAZEL_TARGET_EXCLUDES = (
    "-//runtime/src/iree/hal/drivers/amdgpu/...",
    "-//runtime/src/iree/hal/drivers/cuda/...",
    "-//runtime/src/iree/hal/drivers/hip/...",
    "-//runtime/src/iree/hal/drivers/vulkan/...",
    "-//runtime/src/iree/hal/drivers/webgpu/...",
)
CPU_RESOURCE_TAG_EXCLUDES = (
    "-iree-run-requirement=runtime.resource.amd_gpu",
    "-iree-run-requirement=runtime.resource.nvidia_gpu",
    "-iree-run-requirement=runtime.resource.vulkan_device",
    "-iree-run-requirement=runtime.resource.webgpu_device",
)
NON_CPU_HAL_DRIVER_CTEST_REGEX = (
    r"^iree/hal/drivers/(amdgpu|cuda|hip|metal|vulkan|webgpu)/"
)

AMDGPU_BAZEL_DRIVER_TARGETS = ("//runtime/src/iree/hal/drivers/amdgpu/...",)
AMDGPU_CMAKE_DRIVER_TARGETS = ("runtime/src/iree/hal/drivers/amdgpu/all",)
AMDGPU_TARGET_SELECTOR = "gfx942"
RUNTIME_AMDGPU_RESOURCE_TAG = "iree-run-requirement=runtime.resource.amd_gpu"
AMDGPU_BAZEL_RESOURCE_SLICES = (
    ("runtime", "//runtime", "//runtime/...", RUNTIME_AMDGPU_RESOURCE_TAG),
)
RUNTIME_CTEST_RESOURCE_LABEL_PREFIX = "runtime-resource="
CTEST_RESOURCE_LABEL_EXCLUDE_REGEX = RUNTIME_CTEST_RESOURCE_LABEL_PREFIX
AMDGPU_CTEST_RESOURCE_LABEL_REGEX = "runtime-resource=amd-gpu"
LOOM_AMDGPU_BAZEL_COMPILE_TEST_TARGETS = (
    "//loom/src/loom/target/arch/amdgpu:target_info_test",
    "//loom/src/loom/target/arch/amdgpu:registers_test",
    "//loom/src/loom/target/arch/amdgpu:provider_test",
    "//loom/src/loom/target/arch/amdgpu/encoding:encoding_test",
    "//loom/src/loom/target/arch/amdgpu/matrix:contract_test",
    "//loom/src/loom/target/arch/amdgpu/matrix:projection_test",
    "//loom/src/loom/target/arch/amdgpu/planning:matrix_wait_states_test",
    "//loom/src/loom/target/arch/amdgpu/planning:storage_lease_test",
    "//loom/src/loom/target/emit/native/amdgpu:descriptor_test",
    "//loom/src/loom/target/emit/native/amdgpu:hal_kernel_library_test",
    "//loom/src/loom/target/emit/native/amdgpu:hsaco_test",
    "//loom/src/loom/target/emit/native/amdgpu:kernel_hsaco_test",
    "//loom/src/loom/target/emit/native/amdgpu:metadata_test",
    "//loom/src/loom/target/emit/native/amdgpu:spill_lowering_test",
    "//loom/src/loom/target/emit/native/amdgpu:storage_layout_test",
    "//loom/src/loom/tooling/target/amdgpu:artifact_provider_test",
)
LOOM_AMDGPU_CMAKE_COMPILE_TEST_BUILD_TARGETS = tuple(
    bazel_loom_src_label_to_cmake_target(target)
    for target in LOOM_AMDGPU_BAZEL_COMPILE_TEST_TARGETS
)
LOOM_AMDGPU_CMAKE_COMPILE_CTEST_REGEXES = tuple(
    bazel_pattern_to_ctest_regex(target)
    for target in LOOM_AMDGPU_BAZEL_COMPILE_TEST_TARGETS
)
AMDGPU_XFAILS = (
    bazel_xfail("//runtime/src/iree/hal/drivers/amdgpu/cts/..."),
    bazel_xfail("//runtime/src/iree/hal/drivers/amdgpu:allocator_test"),
    bazel_xfail("//runtime/src/iree/hal/drivers/amdgpu:host_queue_command_buffer_test"),
    bazel_xfail(
        "//runtime/src/iree/hal/drivers/amdgpu:pm4_command_buffer_benchmark_test"
    ),
    ctest_xfail("^iree/hal/drivers/amdgpu/host_queue_pending_test$"),
    ctest_xfail("^iree/hal/drivers/amdgpu/host_queue_staging_test$"),
    ctest_xfail("^iree/hal/drivers/amdgpu/host_queue_submission_test$"),
    ctest_xfail("^iree/hal/drivers/amdgpu/slab_provider_test$"),
    ctest_xfail("^iree/hal/drivers/amdgpu/system_test$"),
    ctest_xfail("^iree/hal/drivers/amdgpu/util/queue_benchmark_test$"),
    ctest_xfail("^iree/hal/drivers/amdgpu/util/vmem_test$"),
)
AMDGPU_SANITIZERS_XFAILS = (
    bazel_xfail("//runtime/src/iree/hal/drivers/amdgpu/cts/..."),
    bazel_xfail("//runtime/src/iree/hal/drivers/amdgpu:allocator_test"),
    bazel_xfail("//runtime/src/iree/hal/drivers/amdgpu:host_queue_command_buffer_test"),
    bazel_xfail("//runtime/src/iree/hal/drivers/amdgpu:host_queue_pending_test"),
    bazel_xfail("//runtime/src/iree/hal/drivers/amdgpu:host_queue_staging_test"),
    bazel_xfail("//runtime/src/iree/hal/drivers/amdgpu:host_queue_submission_test"),
    bazel_xfail(
        "//runtime/src/iree/hal/drivers/amdgpu:pm4_command_buffer_benchmark_test"
    ),
    bazel_xfail("//runtime/src/iree/hal/drivers/amdgpu:slab_provider_test"),
    bazel_xfail("//runtime/src/iree/hal/drivers/amdgpu/util:blit_benchmark_test"),
    bazel_xfail("//runtime/src/iree/hal/drivers/amdgpu/util:block_pool_test"),
    bazel_xfail("//runtime/src/iree/hal/drivers/amdgpu/util:pm4_emitter_test"),
    bazel_xfail("//runtime/src/iree/hal/drivers/amdgpu/util:pm4_program_test"),
    bazel_xfail("//runtime/src/iree/hal/drivers/amdgpu/util:queue_benchmark_test"),
)
AMDGPU_TSAN_XFAILS = (
    bazel_xfail(
        "//runtime/src/iree/hal/drivers/amdgpu:host_queue_command_buffer_profiling_test"
    ),
    # This live PM4 path intentionally loads ROCR and submits real GPU work.
    # ThreadSanitizer cannot model ROCR's uninstrumented async worker threads,
    # and reports races in ROCr runtime bookkeeping during HSA object setup.
    bazel_xfail("//runtime/src/iree/hal/drivers/amdgpu/util:pm4_dispatch_live_test"),
)
AMDGPU_XFAIL_TARGETS = bazel_xfail_targets(AMDGPU_XFAILS)
AMDGPU_CTEST_EXCLUDE_REGEX = ctest_exclude_regex(AMDGPU_XFAILS)
AMDGPU_SANITIZERS_XFAIL_TARGETS = bazel_xfail_targets(AMDGPU_SANITIZERS_XFAILS)
AMDGPU_SANITIZERS_CTEST_EXCLUDE_REGEX = ctest_exclude_regex(AMDGPU_SANITIZERS_XFAILS)
AMDGPU_TSAN_XFAIL_TARGETS = bazel_xfail_targets(AMDGPU_TSAN_XFAILS)
AMDGPU_TSAN_CTEST_EXCLUDE_REGEX = ctest_exclude_regex(AMDGPU_TSAN_XFAILS)
AMDGPU_TSAN_SANITIZERS_XFAIL_TARGETS = bazel_xfail_targets(
    AMDGPU_SANITIZERS_XFAILS + AMDGPU_TSAN_XFAILS
)
AMDGPU_TSAN_SANITIZERS_CTEST_EXCLUDE_REGEX = ctest_exclude_regex(
    AMDGPU_SANITIZERS_XFAILS + AMDGPU_TSAN_XFAILS
)

VULKAN_BAZEL_DRIVER_TARGETS = ("//runtime/src/iree/hal/drivers/vulkan/...",)
RUNTIME_VULKAN_RESOURCE_TAG = "iree-run-requirement=runtime.resource.vulkan_device"
VULKAN_BAZEL_RESOURCE_SLICES = (
    ("loom", "//loom", "//loom/...", RUNTIME_VULKAN_RESOURCE_TAG),
)
VULKAN_CMAKE_DRIVER_TARGETS = ("runtime/src/iree/hal/drivers/vulkan/all",)
VULKAN_CTEST_REGEX = r"^iree/hal/drivers/vulkan/"
VULKAN_CTEST_RESOURCE_LABEL_REGEX = "runtime-resource=vulkan-device"
VULKAN_XFAILS = (
    # These generic executable CTS binaries still bind the empty compatibility
    # testdata registration instead of real SPIR-V payloads.
    bazel_xfail("//runtime/src/iree/hal/drivers/vulkan/cts:profiling_tests"),
    bazel_xfail(
        "//runtime/src/iree/hal/drivers/vulkan/cts:command_buffer_dispatch_tests"
    ),
    bazel_xfail(
        "//runtime/src/iree/hal/drivers/vulkan/cts:command_buffer_dispatch_pipeline_tests"
    ),
    bazel_xfail(
        "//runtime/src/iree/hal/drivers/vulkan/cts:command_buffer_dispatch_reuse_tests"
    ),
    bazel_xfail(
        "//runtime/src/iree/hal/drivers/vulkan/cts:command_buffer_dispatch_constants_bindings_tests"
    ),
    bazel_xfail(
        "//runtime/src/iree/hal/drivers/vulkan/cts:command_buffer_dispatch_constants_tests"
    ),
    bazel_xfail(
        "//runtime/src/iree/hal/drivers/vulkan/cts:command_buffer_dispatch_indirect_parameters_tests"
    ),
    bazel_xfail(
        "//runtime/src/iree/hal/drivers/vulkan/cts:command_buffer_dispatch_multi_entrypoint_tests"
    ),
    bazel_xfail(
        "//runtime/src/iree/hal/drivers/vulkan/cts:command_buffer_dispatch_multi_workgroup_tests"
    ),
    bazel_xfail("//runtime/src/iree/hal/drivers/vulkan/cts:executable_tests"),
    bazel_xfail(
        "//runtime/src/iree/hal/drivers/vulkan/cts:queue_dispatch_direct_tests"
    ),
    bazel_xfail(
        "//runtime/src/iree/hal/drivers/vulkan/cts:queue_dispatch_indirect_parameters_tests"
    ),
    bazel_xfail(
        "//runtime/src/iree/hal/drivers/vulkan/cts:queue_descriptor_cache_tests"
    ),
)
VULKAN_XFAIL_TARGETS = bazel_xfail_targets(VULKAN_XFAILS)
