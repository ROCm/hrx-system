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
#include "loom/ir/type_refs.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Verifies generic function signature and target contracts. Predicate values
// must belong to the function signature. Source values and typed target
// registers must satisfy the predicate kind's semantic value domain. Registers
// without semantic types expose only their target-owned carrier identity;
// boundaries that bind semantic values to those contracts must prove the
// predicates against the source values. Dialect-specific verifiers should call
// this before checking dialect-local function rules.
iree_status_t loom_function_contract_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter);

// Verifies a function implementation provider and its target applicability
// contract. Provider target witnesses describe identity only; target-neutral
// execution choices and limits belong to typed provider requirements.
iree_status_t loom_function_provider_contract_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

enum loom_function_call_argument_match_flag_bits_e {
  // Allows a tensor or view value to be materialized as a buffer ABI argument.
  LOOM_FUNCTION_CALL_ARGUMENT_MATCH_FLAG_ALLOW_BUFFER_MATERIALIZATION = 1u << 0,
  // Template applicability proves dimensions and complete encodings separately.
  // Scalar, nominal, and composite input types retain exact type equality.
  LOOM_FUNCTION_CALL_ARGUMENT_MATCH_FLAG_TEMPLATE_INPUTS = 1u << 1,
};
typedef uint16_t loom_function_call_argument_match_flags_t;

// Verifies that an operation boundary matches the signature of |callee|.
// Unresolved symbols are valid in partial modules and are checked after link
// resolution. Dynamic dimensions and SSA encodings in the callee signature are
// compared after remapping its arguments and results to the call boundary.
// Execution boundaries may select explicit argument materialization policies;
// result types always match the semantic contract exactly.
iree_status_t loom_function_call_contract_verify(
    const loom_module_t* module, const loom_op_t* op, loom_symbol_ref_t callee,
    loom_value_slice_t operands, loom_value_slice_t results,
    loom_function_call_argument_match_flags_t argument_match_flags,
    iree_diagnostic_emitter_t emitter);

// Verifies a callable exit terminator against its enclosing function-like
// result tuple. The exit operands are compared after remapping dependent result
// types to the returned values; function arguments retain their in-body IDs.
iree_status_t loom_function_exit_contract_verify(
    const loom_module_t* module, loom_func_like_t function,
    const loom_op_t* exit_op, iree_diagnostic_emitter_t emitter);

// Tests one call argument against its function contract type after applying
// |value_remap|. Execution boundaries that own storage materialization may
// allow tensor/view values to satisfy buffer ABI arguments.
iree_status_t loom_function_call_argument_type_matches(
    const loom_module_t* module, loom_type_t actual_type,
    loom_type_t expected_type, const loom_type_value_remap_t* value_remap,
    loom_function_call_argument_match_flags_t flags, bool* out_matches);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_FUNCTION_CONTRACT_VERIFY_H_
