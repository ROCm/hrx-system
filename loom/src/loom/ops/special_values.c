// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/special_values.h"

#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"

bool loom_op_is_poison(const loom_op_t* op) {
  if (!op) return false;
  switch (op->kind) {
    case LOOM_OP_SCALAR_POISON:
    case LOOM_OP_VECTOR_POISON:
      return true;
    default:
      return false;
  }
}

bool loom_value_is_poison(const loom_module_t* module,
                          loom_value_id_t value_id) {
  if (!module || value_id >= module->values.count) return false;
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return false;
  return loom_op_is_poison(loom_value_def_op(value));
}

iree_status_t loom_poison_build(loom_builder_t* builder,
                                loom_type_t result_type,
                                loom_location_id_t location,
                                loom_value_id_t* out_value_id) {
  if (loom_type_is_scalar(result_type)) {
    loom_op_t* op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_scalar_poison_build(builder, result_type, location, &op));
    *out_value_id = loom_scalar_poison_result(op);
    return iree_ok_status();
  }
  if (loom_type_is_vector(result_type)) {
    if (loom_type_has_static_zero_extent(result_type)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "static zero-lane vector type has no poison materializer; use "
          "vector.empty");
    }
    loom_op_t* op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_poison_build(builder, result_type, location, &op));
    *out_value_id = loom_vector_poison_result(op);
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "no poison materializer for type kind %u",
                          (unsigned)loom_type_kind(result_type));
}

bool loom_type_has_constant_materializer(loom_type_t type) {
  return loom_type_is_scalar(type);
}

static bool loom_constant_facts_match_scalar_type(loom_value_facts_t facts,
                                                  loom_scalar_type_t type) {
  if (!loom_value_facts_is_exact(facts)) return false;
  if (loom_scalar_type_is_float(type)) return loom_value_facts_is_float(facts);
  if (loom_value_facts_is_float(facts)) return false;
  if (type == LOOM_SCALAR_TYPE_I1) {
    return facts.range_lo == 0 || facts.range_lo == 1;
  }
  return type == LOOM_SCALAR_TYPE_INDEX || type == LOOM_SCALAR_TYPE_OFFSET ||
         loom_scalar_type_is_integer(type);
}

bool loom_value_facts_can_materialize_constant(loom_value_facts_t facts,
                                               loom_type_t type) {
  return loom_type_is_scalar(type) && loom_constant_facts_match_scalar_type(
                                          facts, loom_type_element_type(type));
}

static loom_attribute_t loom_constant_attr_from_facts(loom_value_facts_t facts,
                                                      loom_scalar_type_t type) {
  if (loom_scalar_type_is_float(type)) {
    return loom_attr_f64(loom_value_facts_as_f64(facts));
  }
  if (type == LOOM_SCALAR_TYPE_I1) return loom_attr_bool(facts.range_lo != 0);
  return loom_attr_i64(facts.range_lo);
}

iree_status_t loom_constant_build(loom_builder_t* builder,
                                  loom_value_facts_t facts,
                                  loom_type_t result_type,
                                  loom_location_id_t location,
                                  loom_value_id_t* out_value_id) {
  if (!out_value_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "constant materializer result output is NULL");
  }
  *out_value_id = LOOM_VALUE_ID_INVALID;
  if (!loom_type_has_constant_materializer(result_type)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "no constant materializer for type kind %u",
                            (unsigned)loom_type_kind(result_type));
  }

  loom_scalar_type_t scalar_type = loom_type_element_type(result_type);
  if (!loom_value_facts_can_materialize_constant(facts, result_type)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "exact facts are not materializable as scalar type %u",
        (unsigned)scalar_type);
  }

  loom_attribute_t attr = loom_constant_attr_from_facts(facts, scalar_type);
  if (scalar_type == LOOM_SCALAR_TYPE_INDEX ||
      scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
    loom_op_t* constant_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_constant_build(builder, attr, result_type,
                                                   location, &constant_op));
    *out_value_id = loom_index_constant_result(constant_op);
    return iree_ok_status();
  }

  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(builder, attr, result_type,
                                                  location, &constant_op));
  *out_value_id = loom_scalar_constant_result(constant_op);
  return iree_ok_status();
}

bool loom_op_is_empty(const loom_op_t* op) {
  if (!op) return false;
  switch (op->kind) {
    case LOOM_OP_VECTOR_EMPTY:
      return true;
    default:
      return false;
  }
}

bool loom_type_has_empty_materializer(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_has_static_zero_extent(type);
}

iree_status_t loom_empty_build(loom_builder_t* builder, loom_type_t result_type,
                               loom_location_id_t location,
                               loom_value_id_t* out_value_id) {
  if (loom_type_is_vector(result_type) &&
      loom_type_has_static_zero_extent(result_type)) {
    loom_op_t* op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_empty_build(builder, result_type, location, &op));
    *out_value_id = loom_vector_empty_result(op);
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "no empty materializer for type kind %u",
                          (unsigned)loom_type_kind(result_type));
}
