// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/roles.h"

#include "loom/ir/context.h"
#include "loom/ops/encoding/ops.h"

iree_string_view_t loom_encoding_role_description(loom_encoding_role_t role) {
  switch (role) {
    case LOOM_ENCODING_ROLE_ADDRESS_LAYOUT:
      return IREE_SV("an address layout encoding");
    case LOOM_ENCODING_ROLE_STORAGE_SCHEMA:
      return IREE_SV("a storage schema encoding");
    case LOOM_ENCODING_ROLE_PHYSICAL_STORAGE:
      return IREE_SV("a physical storage encoding");
    case LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM:
      return IREE_SV("a numeric transform encoding");
    case LOOM_ENCODING_ROLE_UNKNOWN:
    default:
      return IREE_SV("an encoding with any role");
  }
}

loom_encoding_role_t loom_encoding_static_role(
    const loom_module_t* module, const loom_encoding_t* encoding) {
  if (!module || !encoding || encoding->name_id == LOOM_STRING_ID_INVALID ||
      encoding->name_id >= module->strings.count) {
    return LOOM_ENCODING_ROLE_UNKNOWN;
  }

  iree_string_view_t name = module->strings.entries[encoding->name_id];
  const loom_encoding_vtable_t* vtable =
      loom_context_lookup_encoding_vtable(module->context, name);
  if (vtable && vtable->role != LOOM_ENCODING_ROLE_UNKNOWN) {
    return vtable->role;
  }

  return LOOM_ENCODING_ROLE_STORAGE_SCHEMA;
}

static bool loom_encoding_address_layout_op_isa(const loom_op_t* op) {
  return op && (loom_encoding_layout_dense_isa(op) ||
                loom_encoding_layout_strided_isa(op));
}

loom_encoding_role_t loom_encoding_value_role(const loom_module_t* module,
                                              loom_value_id_t value_id) {
  if (!module || value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= module->values.count) {
    return LOOM_ENCODING_ROLE_UNKNOWN;
  }
  loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_type_is_encoding(type)) {
    return LOOM_ENCODING_ROLE_UNKNOWN;
  }
  loom_encoding_role_t type_role = loom_type_encoding_role(type);
  if (type_role != LOOM_ENCODING_ROLE_UNKNOWN) return type_role;

  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return LOOM_ENCODING_ROLE_UNKNOWN;
  }

  const loom_op_t* op = loom_value_def_op(value);
  if (loom_encoding_address_layout_op_isa(op)) {
    return LOOM_ENCODING_ROLE_ADDRESS_LAYOUT;
  }
  if (!op || !loom_encoding_define_isa(op)) {
    return LOOM_ENCODING_ROLE_UNKNOWN;
  }

  const loom_encoding_t* spec =
      loom_module_encoding(module, loom_encoding_define_spec(op));
  return loom_encoding_static_role(module, spec);
}
