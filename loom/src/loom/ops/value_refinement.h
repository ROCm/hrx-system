// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Explicit same-value refinements of shaped carriers.

#ifndef LOOM_OPS_VALUE_REFINEMENT_H_
#define LOOM_OPS_VALUE_REFINEMENT_H_

#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Verifies that an asserted shape does not contradict known static extents.
// The operation's generated constraints own carrier, element, and rank checks.
iree_status_t loom_value_refinement_verify(const loom_module_t* module,
                                           const loom_op_t* op,
                                           iree_diagnostic_emitter_t emitter);

// Forwards value facts without creating a new value or ownership identity.
iree_status_t loom_value_refinement_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// Propagates static structure in both directions through the asserted identity.
iree_status_t loom_value_refinement_type_transfer(
    loom_type_transfer_context_t* context, const loom_module_t* module,
    loom_op_t* op);

// Erases the assertion once source and result have the same exact type.
iree_status_t loom_value_refinement_canonicalize(loom_op_t* op,
                                                 loom_rewriter_t* rewriter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_VALUE_REFINEMENT_H_
