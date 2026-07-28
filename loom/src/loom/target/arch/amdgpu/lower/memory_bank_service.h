// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source-memory packet bank-service reporting.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_BANK_SERVICE_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_BANK_SERVICE_H_

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/codegen/low/source_memory_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Populates bank-service evidence for a selected source-memory packet.
//
// The output remains empty when the target or packet has no registered model.
// A selected model always produces either an exact result or an explicit
// unknown proof reason. Analysis runs only while detail report rows are
// requested.
iree_status_t loom_amdgpu_memory_report_bank_service(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_descriptor_t* descriptor,
    const loom_low_source_memory_access_plan_t* source,
    loom_low_lower_memory_bank_service_report_t* out_report);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MEMORY_BANK_SERVICE_H_
