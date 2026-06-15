// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/testing/ir_match_test_util.h"

namespace loom::testing {

const loom_symbol_t* FindModuleSymbol(const loom_module_t* module,
                                      iree_string_view_t name) {
  loom_string_id_t name_id = loom_module_lookup_string(module, name);
  if (name_id == LOOM_STRING_ID_INVALID) {
    return nullptr;
  }
  uint16_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return nullptr;
  }
  return &module->symbols.entries[symbol_id];
}

const loom_op_t* FindModuleSymbolDefiningOp(const loom_module_t* module,
                                            iree_string_view_t name) {
  const loom_symbol_t* symbol = FindModuleSymbol(module, name);
  return symbol ? symbol->defining_op : nullptr;
}

bool SymbolRefsEqual(loom_symbol_ref_t lhs, loom_symbol_ref_t rhs) {
  return lhs.module_id == rhs.module_id && lhs.symbol_id == rhs.symbol_id;
}

bool ModuleStringEquals(const loom_module_t* module, loom_string_id_t string_id,
                        iree_string_view_t expected) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return false;
  }
  return iree_string_view_equal(module->strings.entries[string_id], expected);
}

const loom_block_t* LowFuncEntryBlock(const loom_op_t* low_func_op) {
  loom_region_t* body = loom_low_func_def_body(low_func_op);
  if (body == nullptr || body->block_count == 0) {
    return nullptr;
  }
  return loom_region_const_entry_block(body);
}

const loom_op_t* FindLowFuncBodyOp(const loom_op_t* low_func_op,
                                   loom_op_kind_t kind) {
  loom_region_t* body = loom_low_func_def_body(low_func_op);
  if (body == nullptr) {
    return nullptr;
  }
  loom_block_t* block = nullptr;
  loom_region_for_each_block(body, block) {
    loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (op->kind == kind) {
        return op;
      }
    }
  }
  return nullptr;
}

}  // namespace loom::testing
