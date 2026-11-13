// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/reference_type.h"

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

bool loom_reference_type_declaration_key(const loom_module_t* module,
                                         loom_symbol_ref_t declaration,
                                         loom_string_id_t* out_namespace_id,
                                         loom_string_id_t* out_name_id) {
  if (declaration.module_id != 0 ||
      declaration.symbol_id >= module->symbols.count) {
    return false;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[declaration.symbol_id];
  if (!symbol->definition || !symbol->defining_op ||
      !symbol->definition->type_contract.namespace_attr_index_plus_one) {
    return false;
  }
  const uint8_t namespace_index =
      symbol->definition->type_contract.namespace_attr_index_plus_one - 1;
  const uint8_t name_index =
      symbol->definition->type_contract.name_attr_index_plus_one - 1;
  if (namespace_index >= symbol->defining_op->attribute_count ||
      name_index >= symbol->defining_op->attribute_count) {
    return false;
  }
  const loom_attribute_t* attributes = loom_op_attrs(symbol->defining_op);
  if (attributes[namespace_index].kind != LOOM_ATTR_STRING ||
      attributes[name_index].kind != LOOM_ATTR_STRING) {
    return false;
  }
  *out_namespace_id = loom_attr_as_string_id(attributes[namespace_index]);
  *out_name_id = loom_attr_as_string_id(attributes[name_index]);
  return true;
}

iree_status_t loom_reference_type_make(loom_module_t* module,
                                       loom_reference_ownership_t ownership,
                                       loom_symbol_ref_t declaration,
                                       loom_type_t* out_type) {
  loom_string_id_t namespace_id;
  loom_string_id_t name_id;
  if (!loom_reference_type_declaration_key(module, declaration, &namespace_id,
                                           &name_id)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "managed handle requires a preceding type declaration");
  }
  return loom_module_intern_type(
      module,
      loom_type_reference(ownership, namespace_id, name_id,
                          declaration.symbol_id),
      out_type);
}
