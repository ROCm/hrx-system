// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-to-Low proof and materialization of Low call predicates.

#ifndef LOOM_CODEGEN_LOW_LOWER_CALL_PREDICATES_H_
#define LOOM_CODEGEN_LOW_LOWER_CALL_PREDICATES_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/rewrite/remap.h"

#ifdef __cplusplus
extern "C" {
#endif

// Proves |callee|'s argument predicates from the source invocation operands,
// materializes them as low.assume facts, and updates |remap| so callee
// arguments resolve to the assumed Low values. |remap| must initially map each
// callee argument to its already-materialized Low call operand.
iree_status_t loom_low_materialize_call_argument_contract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, loom_func_like_t callee,
    const loom_value_id_t* callee_arguments, uint16_t callee_argument_count,
    loom_value_slice_t source_operands, loom_ir_remap_t* remap);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_CALL_PREDICATES_H_
