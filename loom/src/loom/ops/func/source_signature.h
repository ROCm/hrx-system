// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_OPS_FUNC_SOURCE_SIGNATURE_H_
#define LOOM_OPS_FUNC_SOURCE_SIGNATURE_H_

#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable view of one verified #func.abi.c.signature contract.
typedef struct loom_func_c_abi_signature_t {
  // Root type index for each source argument.
  loom_i64_array_t arguments;
  // Root type index for each source result.
  loom_i64_array_t results;
  // Flat postorder #func.abi.c.type table.
  loom_parameterized_attr_array_t types;
} loom_func_c_abi_signature_t;

// Returns true and populates |out_signature| when |attr| is a C ABI signature.
// The returned arrays borrow immutable module attribute storage.
bool loom_func_c_abi_signature_cast(loom_attribute_t attr,
                                    loom_func_c_abi_signature_t* out_signature);

// Verifies a retained source signature against its callable's flattened
// signature. Unknown registered source signature families remain available for
// their owning frontend/target integration; the core C family is verified here
// completely.
iree_status_t loom_func_source_signature_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_FUNC_SOURCE_SIGNATURE_H_
