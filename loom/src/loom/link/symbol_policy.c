// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/symbol_policy.h"

#include "loom/ir/context.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/op_defs.h"

static bool loom_link_symbol_has_visibility_attr(const loom_module_t* module,
                                                 const loom_symbol_t* symbol) {
  if (!symbol || !symbol->defining_op) return false;
  const loom_op_vtable_t* vtable = loom_op_vtable(module, symbol->defining_op);
  if (!vtable || !vtable->attr_descriptors) return false;
  const loom_attribute_t* attrs = loom_op_const_attrs(symbol->defining_op);
  for (uint8_t i = 0; i < vtable->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    if (!iree_string_view_equal(loom_attr_descriptor_name(descriptor),
                                IREE_SV("visibility"))) {
      continue;
    }
    if (descriptor->attr_kind != LOOM_ATTR_ENUM ||
        i >= symbol->defining_op->attribute_count) {
      return false;
    }
    return loom_attr_as_enum(attrs[i]) != 0;
  }
  return false;
}

bool loom_link_symbol_is_declaration(const loom_symbol_t* symbol) {
  return symbol->kind == LOOM_SYMBOL_FUNC_DECL ||
         (symbol->defining_op && loom_config_decl_isa(symbol->defining_op));
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
  if (loom_link_symbol_has_visibility_attr(module, symbol)) {
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
