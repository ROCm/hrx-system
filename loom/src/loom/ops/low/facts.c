// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the target-low dialect.
//
// low structural ops model register values, not target instructions. Their
// facts preserve payload properties such as "all register units are zero" so
// descriptor selection can choose inline/immediate forms before allocation
// sees artificial register-range construction.

#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/storage_facts.h"
#include "loom/util/fact_table.h"

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static iree_status_t loom_low_make_register_unit_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    loom_value_id_t result_id, loom_value_facts_t element_facts,
    loom_value_facts_t* out_facts) {
  loom_type_t result_type = loom_module_value_type(module, result_id);
  if (!loom_type_is_register(result_type)) {
    *out_facts = element_facts;
    return iree_ok_status();
  }
  return loom_value_facts_make_uniform_element(context, element_facts,
                                               out_facts);
}

static bool loom_low_attr_as_integer_facts(loom_attribute_t attr,
                                           int64_t* out_value) {
  if (attr.kind == LOOM_ATTR_I64) {
    *out_value = loom_attr_as_i64(attr);
    return true;
  }
  if (attr.kind == LOOM_ATTR_BOOL) {
    *out_value = loom_attr_as_bool(attr) ? 1 : 0;
    return true;
  }
  return false;
}

static bool loom_low_const_attrs_as_uniform_i64(loom_named_attr_slice_t attrs,
                                                int64_t* out_value) {
  if (attrs.count == 0) {
    return false;
  }

  int64_t first_value = 0;
  if (!loom_low_attr_as_integer_facts(attrs.entries[0].value, &first_value)) {
    return false;
  }
  if (attrs.count == 1) {
    *out_value = first_value;
    return true;
  }

  if (first_value != 0) {
    return false;
  }
  for (iree_host_size_t i = 1; i < attrs.count; ++i) {
    int64_t value = 0;
    if (!loom_low_attr_as_integer_facts(attrs.entries[i].value, &value) ||
        value != 0) {
      return false;
    }
  }
  *out_value = 0;
  return true;
}

//===----------------------------------------------------------------------===//
// Fact callbacks
//===----------------------------------------------------------------------===//

iree_status_t loom_low_const_facts(loom_fact_context_t* context,
                                   const loom_module_t* module,
                                   const loom_op_t* op,
                                   const loom_value_facts_t* operand_facts,
                                   loom_value_facts_t* result_facts) {
  int64_t value = 0;
  if (!loom_low_const_attrs_as_uniform_i64(loom_low_const_attrs(op), &value)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  return loom_low_make_register_unit_facts(
      context, module, loom_low_const_result(op),
      loom_value_facts_exact_i64(value), &result_facts[0]);
}

iree_status_t loom_low_copy_facts(loom_fact_context_t* context,
                                  const loom_module_t* module,
                                  const loom_op_t* op,
                                  const loom_value_facts_t* operand_facts,
                                  loom_value_facts_t* result_facts) {
  result_facts[0] = operand_facts[0];
  return iree_ok_status();
}

iree_status_t loom_low_slice_facts(loom_fact_context_t* context,
                                   const loom_module_t* module,
                                   const loom_op_t* op,
                                   const loom_value_facts_t* operand_facts,
                                   loom_value_facts_t* result_facts) {
  loom_value_facts_t element_facts = loom_value_facts_unknown();
  if (!loom_value_facts_query_all_equal_element(context, operand_facts[0],
                                                &element_facts)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  return loom_low_make_register_unit_facts(context, module,
                                           loom_low_slice_result(op),
                                           element_facts, &result_facts[0]);
}

iree_status_t loom_low_concat_facts(loom_fact_context_t* context,
                                    const loom_module_t* module,
                                    const loom_op_t* op,
                                    const loom_value_facts_t* operand_facts,
                                    loom_value_facts_t* result_facts) {
  if (op->operand_count == 0) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }

  loom_value_facts_t element_facts = loom_value_facts_unknown();
  if (!loom_value_facts_query_all_equal_element(context, operand_facts[0],
                                                &element_facts)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  for (uint16_t i = 1; i < op->operand_count; ++i) {
    loom_value_facts_t next_element_facts = loom_value_facts_unknown();
    if (!loom_value_facts_query_all_equal_element(context, operand_facts[i],
                                                  &next_element_facts) ||
        !loom_value_facts_equal(element_facts, next_element_facts)) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
  }
  return loom_low_make_register_unit_facts(context, module,
                                           loom_low_concat_result(op),
                                           element_facts, &result_facts[0]);
}

iree_status_t loom_low_storage_reserve_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_id_t storage_id = loom_low_storage_reserve_storage(op);
  loom_type_t storage_type = loom_module_value_type(module, storage_id);
  if (!loom_type_is_storage(storage_type)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  return loom_storage_facts_make_reserve(
      context, storage_id, loom_type_storage_space(storage_type),
      loom_low_storage_reserve_byte_length(op),
      loom_low_storage_reserve_byte_alignment(op), &result_facts[0]);
}

iree_status_t loom_low_storage_view_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  return loom_storage_facts_make_static_view(
      context, operand_facts[0], loom_low_storage_view_offset(op),
      loom_low_storage_view_byte_length(op), &result_facts[0]);
}
