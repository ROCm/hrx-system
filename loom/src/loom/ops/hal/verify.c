// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ir/value_tree.h"
#include "loom/ops/hal/ops.h"
#include "loom/ops/hal/schema.h"

typedef struct loom_hal_verify_cursor_t {
  // Module owning the argument types and source spelling.
  const loom_module_t* module;
  // Operation whose geometry has passed generic verification.
  const loom_op_t* op;
  // External diagnostic sink; emitting a diagnostic may fail.
  iree_diagnostic_emitter_t emitter;
  // Next source tree node, consumed in preorder.
  const int64_t* shape;
  // Next ordinary SSA argument leaf.
  const loom_value_id_t* values;
  // False after the first semantic mismatch stops the schema walk.
  bool valid;
} loom_hal_verify_cursor_t;

static iree_status_t loom_hal_verify_shape_error(
    loom_hal_verify_cursor_t* cursor, uint32_t token,
    iree_string_view_t expected) {
  cursor->valid = false;
  const loom_diagnostic_param_t parameters[] = {
      loom_param_string(IREE_SV("shape")),
      loom_param_i64(token),
      loom_param_string(expected),
  };
  return iree_diagnostic_emit(
      cursor->emitter,
      &(loom_diagnostic_emission_t){.op = cursor->op,
                                    .error = LOOM_ERR_STRUCTURE_014,
                                    .params = parameters,
                                    .param_count = IREE_ARRAYSIZE(parameters)});
}

// Recursion follows the immutable method schema, never arbitrary guest depth.
// Generic OperandTree admission establishes complete shape/operand coverage;
// this walk additionally establishes field arities and exact semantic types.
static iree_status_t loom_hal_verify_node(loom_hal_verify_cursor_t* cursor,
                                          uint16_t node_ordinal) {
  const loom_hal_schema_node_t* node = &loom_hal_schema_nodes[node_ordinal];
  uint32_t token = (uint32_t)*cursor->shape;
  uint32_t kind = token & 3;
  uint32_t count = token >> 2;
  if (node->kind == LOOM_HAL_SCHEMA_OPTIONAL) {
    if (kind == LOOM_VALUE_TREE_ABSENT) {
      ++cursor->shape;
      return iree_ok_status();
    }
    return loom_hal_verify_node(cursor,
                                loom_hal_schema_children[node->child_base]);
  }
  ++cursor->shape;
  if (node->kind == LOOM_HAL_SCHEMA_VALUE) {
    if (kind != LOOM_VALUE_TREE_VALUE) {
      return loom_hal_verify_shape_error(cursor, token,
                                         IREE_SV("an SSA value"));
    }
    loom_type_t type =
        loom_module_value_type(cursor->module, *cursor->values++);
    bool matches =
        node->scalar_type != LOOM_SCALAR_TYPE_NONE
            ? loom_type_equal(type, loom_type_scalar(node->scalar_type))
            : loom_type_kind(type) == LOOM_TYPE_DIALECT &&
                  loom_type_dialect_param_count(type) == 0 &&
                  iree_string_view_equal(
                      loom_string_table_get(&cursor->module->strings,
                                            loom_type_dialect_name_id(type)),
                      node->reference_type);
    if (matches) {
      return iree_ok_status();
    }
    cursor->valid = false;
    const loom_diagnostic_param_t parameters[] = {
        loom_param_string(IREE_SV("arguments")),
        loom_param_type(type),
        loom_param_string(node->reference_type.size
                              ? node->reference_type
                              : iree_make_cstring_view(
                                    loom_scalar_type_name(node->scalar_type))),
    };
    return iree_diagnostic_emit(cursor->emitter,
                                &(loom_diagnostic_emission_t){
                                    .op = cursor->op,
                                    .error = LOOM_ERR_TYPE_003,
                                    .params = parameters,
                                    .param_count = IREE_ARRAYSIZE(parameters)});
  }
  bool record = node->kind == LOOM_HAL_SCHEMA_RECORD;
  if (kind != (record ? LOOM_VALUE_TREE_RECORD : LOOM_VALUE_TREE_RANGE) ||
      (record && count != node->child_count)) {
    return loom_hal_verify_shape_error(
        cursor, token,
        record ? IREE_SV("a record with the declared field count")
               : IREE_SV("a range of the declared element type"));
  }
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < count && cursor->valid && iree_status_is_ok(status);
       ++i) {
    status = loom_hal_verify_node(
        cursor, loom_hal_schema_children[node->child_base + (record ? i : 0)]);
  }
  return status;
}

iree_status_t loom_hal_method_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  const loom_hal_method_t* method =
      &loom_hal_methods[loom_op_dialect_index(op->kind)];
  loom_hal_verify_cursor_t cursor = {
      .module = module,
      .op = op,
      .emitter = emitter,
      .shape = loom_op_attrs(op)[0].i64_array,
      .values = loom_op_const_operands(op),
      .valid = true,
  };
  return loom_hal_verify_node(&cursor, method->arguments);
}
