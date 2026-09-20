// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/pass_registry.h"

#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/pipeline/pass_requirements.h"
#include "loom/target/arch/x86/sysv_abi_materialization_pass.h"

static const loom_pass_requirement_def_t kX86MaterializeSysvAbiRequirements[] =
    {
        {
            .capability_type = &loom_low_pass_capability_type,
            .key = IREE_SVL(
                LOOM_LOW_PASS_REQUIREMENT_TARGET_LOW_DESCRIPTOR_REGISTRY),
            .description =
                IREE_SVL("Requires a pass environment target-low descriptor "
                         "registry."),
        },
};

static const loom_pass_descriptor_t kX86PassDescriptors[] = {
    {
        .key = IREE_SVL("x86-materialize-sysv-abi"),
        .info = loom_x86_materialize_sysv_abi_pass_info,
        .module_run = loom_x86_materialize_sysv_abi_run,
        .requirement_defs = kX86MaterializeSysvAbiRequirements,
        .requirement_count = IREE_ARRAYSIZE(kX86MaterializeSysvAbiRequirements),
    },
};

const loom_pass_registry_t loom_x86_pass_registry = {
    .descriptors = kX86PassDescriptors,
    .descriptor_count = IREE_ARRAYSIZE(kX86PassDescriptors),
};
