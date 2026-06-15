// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured diagnostic feedback for completed allocation tables.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_DIAGNOSTICS_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_DIAGNOSTICS_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation/table.h"
#include "loom/error/emitter.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_low_allocation_diagnostic_bits_e {
  // Emits BACKEND/008 warnings for spill plans predicted by allocation.
  LOOM_LOW_ALLOCATION_DIAGNOSTIC_PREDICTED_SPILLS = 1u << 0,
  // Emits BACKEND/006 remarks for low.copy coalescing decisions.
  LOOM_LOW_ALLOCATION_DIAGNOSTIC_COPY_DECISIONS = 1u << 1,
  // Emits BACKEND/043 remarks for placement-affinity decisions.
  LOOM_LOW_ALLOCATION_DIAGNOSTIC_PLACEMENT_DECISIONS = 1u << 2,
} loom_low_allocation_diagnostic_bits_t;

// Bitset of loom_low_allocation_diagnostic_bits_t values.
typedef uint32_t loom_low_allocation_diagnostic_flags_t;

// Emits the requested structured diagnostic feedback for |table|.
iree_status_t loom_low_allocation_diagnostics_emit(
    const loom_low_allocation_table_t* table,
    loom_low_allocation_diagnostic_flags_t flags,
    iree_diagnostic_emitter_t emitter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_DIAGNOSTICS_H_
