// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/symbol_policy.h"

#include "loom/ir/context.h"
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

void loom_link_symbol_internalize(loom_module_t* module, loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  const loom_symbol_definition_descriptor_t* definition = vtable->symbol_def;
  const uint8_t visibility_attr_index =
      loom_symbol_definition_visibility_attr_index(definition);
  const uint8_t retain_attr_index =
      definition->retain_attr_index_plus_one
          ? definition->retain_attr_index_plus_one - 1
          : LOOM_ATTR_INDEX_NONE;

  if (visibility_attr_index != LOOM_ATTR_INDEX_NONE) {
    loom_op_attrs(op)[visibility_attr_index] = loom_attr_absent();
  }
  if (retain_attr_index != LOOM_ATTR_INDEX_NONE) {
    loom_op_attrs(op)[retain_attr_index] = loom_attr_absent();
  }
  const loom_func_like_t function = loom_func_like_cast(module, op);
  if (loom_func_like_isa(function)) {
    const uint8_t export_attr_indices[] = {
        function.vtable->export_symbol_attr_index,
        function.vtable->export_attrs_attr_index,
        function.vtable->export_metadata_attr_index,
        function.vtable->export_linkage_attr_index,
    };
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(export_attr_indices); ++i) {
      if (export_attr_indices[i] != LOOM_ATTR_INDEX_NONE) {
        loom_op_attrs(op)[export_attr_indices[i]] = loom_attr_absent();
      }
    }
  }
  loom_module_link_symbol_defining_op(module, op, vtable);
}
