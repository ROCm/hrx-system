// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/symbol_policy.h"

#include "loom/ops/op_defs.h"

bool loom_link_symbol_is_declaration(const loom_symbol_t* symbol) {
  return symbol && loom_symbol_definition_is_declaration(symbol->definition);
}

bool loom_link_symbol_is_concrete_definition(const loom_symbol_t* symbol) {
  return symbol->defining_op && !loom_link_symbol_is_declaration(symbol);
}

bool loom_link_symbol_has_global_identity(const loom_module_t* module,
                                          const loom_symbol_t* symbol) {
  if (!symbol) return false;
  if (iree_any_bit_set(symbol->flags, LOOM_SYMBOL_FLAG_PUBLIC)) {
    return true;
  }
  if (!symbol->defining_op) {
    return true;
  }
  if (loom_link_symbol_is_declaration(symbol)) {
    return true;
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_CONFIG)) {
    return true;
  }

  loom_func_like_t func = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(func)) {
    return false;
  }
  return loom_func_like_import_module(func) != LOOM_STRING_ID_INVALID ||
         loom_func_like_import_symbol(func) != LOOM_STRING_ID_INVALID ||
         loom_func_like_export_symbol(func) != LOOM_STRING_ID_INVALID;
}
