// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_UTIL_REFERENCE_TRANSFER_H_
#define LOOM_UTIL_REFERENCE_TRANSFER_H_

#include "loom/ops/op_defs.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the reference origin, or unknown for facts without a reference.
loom_value_fact_reference_origin_t loom_value_facts_reference_origin(
    const loom_fact_context_t* context, loom_value_facts_t facts);

// Substitutes the reference origin in facts owned by |context|. Other reference
// properties and scalar facts are preserved. Facts without a reference remain
// unchanged.
iree_status_t loom_value_facts_rebind_reference_origin(
    loom_fact_context_t* context, loom_value_fact_reference_origin_t origin,
    loom_value_facts_t* inout_facts);

// Reframes a caller-owned reference as one callee entry argument. A view's
// passed address becomes the callee-local root at byte offset zero; its proven
// address alignment, footprint, memory space, and nullability survive. Caller
// SSA and alias-scope identities do not cross the function boundary. Facts
// without a storage reference remain unchanged.
iree_status_t loom_value_facts_rebase_call_argument(
    loom_fact_context_t* context, loom_value_id_t argument_id,
    loom_value_fact_reference_origin_t argument_origin,
    loom_value_facts_t* inout_facts);

// Reference substitution for one direct call. All storage arguments contribute
// once to |common_origin|; individual returned arguments use indexed lookup.
// Borrowed argument facts remain live while translating the call's results.
typedef struct loom_reference_call_t {
  // Module owning the formal SSA identities and their current argument indices.
  const loom_module_t* module;
  // Caller-owned facts for actual argument values.
  const loom_value_fact_table_t* argument_facts;
  // Actual argument IDs, indexed by the callee's formal argument ordinal.
  loom_value_slice_t arguments;
  // Origin guarantee common to all storage arguments.
  loom_value_fact_reference_origin_t common_origin;
  // Origin of storage allocated during this call, relative to the caller.
  loom_value_fact_reference_origin_t allocation_origin;
} loom_reference_call_t;

// Prepares substitution in linear time in the call's argument count. The
// known caller origin names the projected function region containing the call.
void loom_reference_call_initialize(
    const loom_module_t* module, const loom_value_fact_table_t* argument_facts,
    loom_value_slice_t arguments,
    loom_value_fact_reference_origin_t caller_origin,
    loom_reference_call_t* out_call);

// Translates a callee return origin into the caller's invocation. In
// particular, a recursive callee's entry can refer to an allocation made by its
// caller.
loom_value_fact_reference_origin_t loom_reference_call_result_origin(
    const loom_reference_call_t* call,
    loom_value_fact_reference_origin_t return_origin);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_UTIL_REFERENCE_TRANSFER_H_
