// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test helpers for matching lowered IR structure without printing modules.

#ifndef LOOM_CODEGEN_LOW_TESTING_IR_MATCH_TEST_UTIL_H_
#define LOOM_CODEGEN_LOW_TESTING_IR_MATCH_TEST_UTIL_H_

#include "iree/base/api.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

namespace loom::testing {

// Finds a module symbol by textual name.
const loom_symbol_t* FindModuleSymbol(const loom_module_t* module,
                                      iree_string_view_t name);

// Finds the op defining a module symbol by textual name.
const loom_op_t* FindModuleSymbolDefiningOp(const loom_module_t* module,
                                            iree_string_view_t name);

// Returns true when two symbol refs name the same symbol.
bool SymbolRefsEqual(loom_symbol_ref_t lhs, loom_symbol_ref_t rhs);

// Returns true when an interned module string equals |expected|.
bool ModuleStringEquals(const loom_module_t* module, loom_string_id_t string_id,
                        iree_string_view_t expected);

// Returns the entry block of a target-low function body, or nullptr.
const loom_block_t* LowFuncEntryBlock(const loom_op_t* low_func_op);

// Finds the first op with |kind| inside a target-low function body.
const loom_op_t* FindLowFuncBodyOp(const loom_op_t* low_func_op,
                                   loom_op_kind_t kind);

}  // namespace loom::testing

#endif  // LOOM_CODEGEN_LOW_TESTING_IR_MATCH_TEST_UTIL_H_
