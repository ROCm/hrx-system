// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/pass_registry.h"

#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/pipeline/pass_requirements.h"
#include "loom/target/arch/amdgpu/hal/binding_materialization_pass.h"

static const loom_pass_requirement_def_t
    kAmdgpuMaterializeHalKernelAbiRequirements[] = {
        {
            .capability_type = &loom_low_pass_capability_type,
            .key = IREE_SVL(
                LOOM_LOW_PASS_REQUIREMENT_TARGET_LOW_DESCRIPTOR_REGISTRY),
            .description =
                IREE_SVL("Requires a pass environment target-low descriptor "
                         "registry."),
        },
};

static const loom_pass_descriptor_t kAmdgpuPassDescriptors[] = {
    {
        .key = IREE_SVL("amdgpu-materialize-hal-buffer-descriptors"),
        .info = loom_amdgpu_materialize_hal_buffer_descriptors_pass_info,
        .function_run = loom_amdgpu_materialize_hal_buffer_descriptors_run,
        .requirement_defs = kAmdgpuMaterializeHalKernelAbiRequirements,
        .requirement_count =
            IREE_ARRAYSIZE(kAmdgpuMaterializeHalKernelAbiRequirements),
    },
    {
        .key = IREE_SVL("amdgpu-materialize-hal-kernel-abi"),
        .info = loom_amdgpu_materialize_hal_kernel_abi_pass_info,
        .function_run = loom_amdgpu_materialize_hal_kernel_abi_run,
        .requirement_defs = kAmdgpuMaterializeHalKernelAbiRequirements,
        .requirement_count =
            IREE_ARRAYSIZE(kAmdgpuMaterializeHalKernelAbiRequirements),
    },
};

const loom_pass_registry_t loom_amdgpu_pass_registry = {
    .descriptors = kAmdgpuPassDescriptors,
    .descriptor_count = IREE_ARRAYSIZE(kAmdgpuPassDescriptors),
};
