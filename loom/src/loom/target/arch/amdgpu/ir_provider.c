// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/ir_provider.h"

#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/low_asm_diagnostics.h"
#include "loom/target/arch/amdgpu/ops/registry.h"

static const loom_target_low_asm_diagnostic_provider_t*
    kLoomAmdgpuIrLowAsmDiagnosticProviders[] = {
        &loom_amdgpu_low_asm_diagnostic_provider,
};

const loom_target_provider_t loom_amdgpu_ir_target_provider = {
    .register_context = loom_amdgpu_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_amdgpu_low_descriptor_registry_initialize,
    .low_asm_diagnostic_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomAmdgpuIrLowAsmDiagnosticProviders),
            .values = kLoomAmdgpuIrLowAsmDiagnosticProviders,
        },
    .target_fact_type = &loom_amdgpu_target_fact_type,
};
