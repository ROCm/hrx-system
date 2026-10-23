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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_call_argument_contract_t
    loom_low_call_argument_contract_t;
typedef struct loom_low_call_result_contract_t loom_low_call_result_contract_t;

// Proves the helper's argument predicates and retains a self-contained copy in
// the function arena. Formal parameter identities remain distinct even when
// several invocation operands name the same source value. Returns NULL when
// the helper has no preconditions or a diagnostic rejects the invocation.
iree_status_t loom_low_plan_call_argument_contract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t callee_name, loom_func_like_t callee,
    const loom_value_id_t* callee_arguments, uint16_t callee_argument_count,
    loom_value_slice_t source_operands,
    const loom_low_call_argument_contract_t** out_contract);

// Retains the helper's result postconditions and their complete formal
// argument/result correspondence. Predicates involving any result are
// postconditions; argument-only predicates remain caller preconditions.
iree_status_t loom_low_plan_call_result_contract(
    loom_low_lower_context_t* context, loom_func_like_t callee,
    const loom_low_call_result_contract_t** out_contract);

// Materializes the proved predicates as low.assume and replaces |low_operands|
// with its results. The contract retains the formal-to-operand correspondence;
// emission needs neither the helper signature nor the source proof environment.
iree_status_t loom_low_materialize_call_argument_contract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_call_argument_contract_t* contract,
    loom_value_id_t* low_operands);

// Materializes retained postconditions after a call result has been projected
// into logical carriers. |low_results| is replaced with constrained aliases;
// argument carriers remain available to predicates without changing their
// established source bindings.
iree_status_t loom_low_materialize_call_result_contract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_call_result_contract_t* contract,
    const loom_value_id_t* low_arguments, loom_value_id_t* low_results);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_CALL_PREDICATES_H_
