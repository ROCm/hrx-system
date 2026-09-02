// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared effect semantics for operations that reference callable symbols.

#ifndef LOOM_OPS_CALLABLE_EFFECTS_H_
#define LOOM_OPS_CALLABLE_EFFECTS_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/rewrite/rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |function| has pure callable semantics. Definitions are
// classified from their cached body effects while declarations use their
// explicit purity contract.
bool loom_callable_effects_is_pure(loom_func_like_t function);

// Propagates a resolved callee's purity to |op| when the operation has no
// explicit purity. Unresolved symbols and impure callees leave |op| unchanged.
iree_status_t loom_callable_effects_propagate_purity(loom_op_t* op,
                                                     loom_symbol_ref_t callee,
                                                     uint8_t purity_attr_index,
                                                     loom_rewriter_t* rewriter);

// Returns the effective effect traits for an operation carrying a purity
// attribute at |purity_attr_index|.
loom_trait_flags_t loom_callable_effects_traits(const loom_op_t* op,
                                                uint8_t purity_attr_index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_CALLABLE_EFFECTS_H_
