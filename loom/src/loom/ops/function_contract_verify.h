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
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Verifies generic function target contract structure. Dialect-specific
// verifiers should call this before checking dialect-local function rules.
iree_status_t loom_function_contract_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_FUNCTION_CONTRACT_VERIFY_H_
