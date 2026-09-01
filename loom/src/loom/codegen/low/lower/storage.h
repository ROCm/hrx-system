// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-value storage demand for source-to-Low lowering.

#ifndef LOOM_CODEGEN_LOW_LOWER_STORAGE_H_
#define LOOM_CODEGEN_LOW_LOWER_STORAGE_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |source_op| must emit a target-Low operation.
bool loom_low_lower_source_op_requires_emission(
    const loom_low_lower_context_t* context, const loom_op_t* source_op);

// Returns the exact condition value when facts prove a cfg.cond_br direction.
bool loom_low_lower_cfg_cond_br_exact_bool(
    const loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_condition);

// Returns true when a source result requires an emitted Low SSA value.
bool loom_low_lower_result_storage_required(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id);

// Propagates source-value storage demand through structural operations and
// selected lowering plans, then marks plans whose results can be elided.
void loom_low_lower_analyze_storage_demands(loom_low_lower_context_t* context,
                                            loom_region_t* source_body);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_STORAGE_H_
