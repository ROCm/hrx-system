// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared verification for function-owned target/ABI/export contracts.

#ifndef LOOM_OPS_FUNCTION_CONTRACT_VERIFY_H_
#define LOOM_OPS_FUNCTION_CONTRACT_VERIFY_H_

#include "iree/base/api.h"
#include "loom/error/emitter.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Verifies generic function signature and target contracts. Predicate values
// must belong to the function signature. Source values and typed target
// registers must satisfy the predicate kind's semantic value domain. Registers
// without semantic types defer domain validation to the target-bound verifier,
// which owns the canonical register representation. Dialect-specific verifiers
// should call this before checking dialect-local function rules.
iree_status_t loom_function_contract_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter);

// Verifies a function implementation provider and its target applicability
// contract. Provider target witnesses describe identity only; target-neutral
// execution choices and limits belong to typed provider requirements.
iree_status_t loom_function_provider_contract_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Verifies that an operation boundary matches the signature of |callee|.
// Unresolved symbols are valid in partial modules and are checked after link
// resolution. Dynamic dimensions and SSA encodings in the callee signature are
// compared after remapping its arguments and results to the call boundary.
iree_status_t loom_function_call_contract_verify(
    const loom_module_t* module, const loom_op_t* op, loom_symbol_ref_t callee,
    loom_value_slice_t operands, loom_value_slice_t results,
    iree_diagnostic_emitter_t emitter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_FUNCTION_CONTRACT_VERIFY_H_
