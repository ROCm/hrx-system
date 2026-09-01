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

// Verifies generic function target contract structure. Dialect-specific
// verifiers should call this before checking dialect-local function rules.
iree_status_t loom_function_contract_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter);

// Verifies generic module import policy and naming invariants on a function
// declaration. Function kinds without import fields pass unchanged.
iree_status_t loom_function_import_contract_verify(
    const loom_module_t* module, const loom_op_t* op,
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

// Verifies that |function_type| exactly describes |callee|'s argument and
// result contract. Unresolved symbols are valid in partial modules and are
// checked after link resolution.
iree_status_t loom_function_type_contract_verify(
    const loom_module_t* module, const loom_op_t* op, loom_symbol_ref_t callee,
    loom_type_t function_type, iree_diagnostic_emitter_t emitter);

// Verifies that |callee| names an optional function import. Unresolved symbols
// are valid in partial modules and are checked after link resolution.
iree_status_t loom_function_optional_import_contract_verify(
    const loom_module_t* module, const loom_op_t* op, loom_symbol_ref_t callee,
    uint16_t callee_attr_index, iree_diagnostic_emitter_t emitter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_FUNCTION_CONTRACT_VERIFY_H_
