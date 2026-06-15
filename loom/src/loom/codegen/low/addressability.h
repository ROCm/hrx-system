// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-low operand addressability validation.
//
// Descriptor operands describe both the register classes they accept and how
// the final assigned register location is addressed by the target packet. This
// layer validates final schedule/allocation tables against those descriptor
// address maps before target-specific emitters try to encode packets.

#ifndef LOOM_CODEGEN_LOW_ADDRESSABILITY_H_
#define LOOM_CODEGEN_LOW_ADDRESSABILITY_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/error/emitter.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_addressability_validation_result_t {
  // Number of error diagnostics emitted for unaddressable packet operands.
  uint32_t error_count;
} loom_low_addressability_validation_result_t;

// Validates that descriptor-backed packet operands can address their final
// allocated register assignments. User-facing violations are emitted through
// |emitter| and counted in |out_result|; infrastructure/table corruption is
// returned as status.
iree_status_t loom_low_addressability_validate_allocated_packets(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_diagnostic_emitter_t emitter,
    loom_low_addressability_validation_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ADDRESSABILITY_H_
