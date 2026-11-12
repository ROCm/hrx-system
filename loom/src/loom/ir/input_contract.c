// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/input_contract.h"

#include "loom/ir/module.h"

uint32_t loom_input_shape_requirements(const loom_input_shape_t* shape,
                                       const loom_input_shape_t* inherited,
                                       uint16_t argument,
                                       loom_input_requirement_t* output) {
  if (shape == inherited) {
    return 0;
  }
  if (inherited && (shape->kind != inherited->kind ||
                    shape->element_type != inherited->element_type ||
                    shape->rank != inherited->rank ||
                    shape->access_alignment != inherited->access_alignment)) {
    inherited = NULL;
  }
  uint32_t count = 0;
  for (uint8_t axis = 0; axis < shape->rank; ++axis) {
    const bool dynamic = (shape->dynamic_axes & (1u << axis)) != 0;
    const uint64_t expected = shape->dimensions[axis];
    if ((dynamic && !expected) ||
        (inherited && expected == inherited->dimensions[axis] &&
         ((shape->dynamic_axes ^ inherited->dynamic_axes) & (1u << axis)) ==
             0)) {
      continue;
    }
    if (output) {
      output[count] = (loom_input_requirement_t){
          .argument = argument,
          .axis = axis,
          .expected_is_argument = dynamic,
          .expected = dynamic ? expected - 1 : expected,
      };
    }
    ++count;
  }
  if (shape->kind == LOOM_TYPE_GROUP || shape->kind == LOOM_TYPE_VECTOR ||
      (inherited && shape->attachment == inherited->attachment &&
       shape->encoding_ordinal == inherited->encoding_ordinal)) {
    return count;
  }
  if (output) {
    output[count] = (loom_input_requirement_t){
        .argument = argument,
        .axis = UINT8_MAX,
        .expected_is_argument =
            shape->attachment == LOOM_INPUT_ENCODING_ARGUMENT,
        .expected = shape->encoding_ordinal,
    };
  }
  return count + 1;
}

// Verification establishes that these are input definitions in the owning
// signature. Their definition/operand index preserves source order even when
// SSA IDs are sparse or allocated in a different order.
static uint16_t loom_input_argument_ordinal(const loom_module_t* module,
                                            loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(module, value_id);
  return loom_value_is_block_arg(value)
             ? loom_value_def_index(value)
             : loom_use_operand_index(loom_value_uses(value)[0]);
}

static void loom_input_shape_project(const loom_module_t* module,
                                     loom_type_t type,
                                     loom_input_shape_t* shape) {
  *shape = (loom_input_shape_t){
      .kind = loom_type_kind(type),
      .element_type = loom_type_element_type(type),
      .rank = loom_type_rank(type),
      .access_alignment = loom_type_view_alignment_override(type),
      .attachment = loom_type_has_ssa_encoding(type)
                        ? LOOM_INPUT_ENCODING_ARGUMENT
                        : (type.encoding_id ? LOOM_INPUT_ENCODING_STATIC
                                            : LOOM_INPUT_ENCODING_NONE),
      .encoding_ordinal = loom_type_has_ssa_encoding(type)
                              ? loom_input_argument_ordinal(
                                    module, loom_type_encoding_value_id(type))
                              : type.encoding_id,
  };
  for (uint8_t axis = 0; axis < shape->rank; ++axis) {
    if (loom_type_dim_is_dynamic_at(type, axis)) {
      const loom_value_id_t value_id = loom_type_dim_value_id_at(type, axis);
      shape->dynamic_axes |= (uint16_t)(1u << axis);
      shape->dimensions[axis] =
          value_id == LOOM_VALUE_ID_INVALID
              ? 0
              : (uint64_t)loom_input_argument_ordinal(module, value_id) + 1;
    } else {
      shape->dimensions[axis] = loom_type_dim_static_size_at(type, axis);
    }
  }
}

uint32_t loom_input_contract_project(const loom_module_t* module,
                                     const loom_value_id_t* arguments,
                                     uint16_t argument_count,
                                     const loom_value_id_t* inherited_arguments,
                                     loom_input_requirement_t* output) {
  iree_alignas(8) uint8_t shape_storage[sizeof(loom_input_shape_t) +
                                        LOOM_TYPE_MAX_RANK * sizeof(uint64_t)];
  iree_alignas(8)
      uint8_t inherited_storage[sizeof(loom_input_shape_t) +
                                LOOM_TYPE_MAX_RANK * sizeof(uint64_t)];
  loom_input_shape_t* shape = (loom_input_shape_t*)shape_storage;
  loom_input_shape_t* inherited = (loom_input_shape_t*)inherited_storage;
  uint32_t count = 0;
  for (uint16_t i = 0; i < argument_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, arguments[i]);
    if (!loom_type_has_dimensions(type)) {
      continue;
    }
    loom_input_shape_project(module, type, shape);
    const loom_input_shape_t* inherited_shape = NULL;
    if (inherited_arguments) {
      const loom_type_t inherited_type =
          loom_module_value_type(module, inherited_arguments[i]);
      if (loom_type_has_dimensions(inherited_type)) {
        loom_input_shape_project(module, inherited_type, inherited);
        inherited_shape = inherited;
      }
    }
    count += loom_input_shape_requirements(shape, inherited_shape, i,
                                           output ? output + count : NULL);
  }
  return count;
}

loom_predicate_t loom_input_requirement_dimension_predicate(
    const loom_module_t* module, const loom_value_id_t* arguments,
    const loom_input_requirement_t* requirement) {
  const loom_type_t type =
      loom_module_value_type(module, arguments[requirement->argument]);
  loom_predicate_t predicate = {.kind = LOOM_PREDICATE_EQ, .arg_count = 2};
  if (loom_type_dim_is_dynamic_at(type, requirement->axis)) {
    const loom_value_id_t dimension =
        loom_type_dim_value_id_at(type, requirement->axis);
    if (dimension != LOOM_VALUE_ID_INVALID) {
      predicate.arg_tags[0] = LOOM_PRED_ARG_VALUE;
      predicate.args[0] = dimension;
    }
  } else {
    predicate.arg_tags[0] = LOOM_PRED_ARG_CONST;
    predicate.args[0] = loom_type_dim_static_size_at(type, requirement->axis);
  }
  predicate.arg_tags[1] = requirement->expected_is_argument
                              ? LOOM_PRED_ARG_VALUE
                              : LOOM_PRED_ARG_CONST;
  predicate.args[1] = requirement->expected_is_argument
                          ? arguments[requirement->expected]
                          : (int64_t)requirement->expected;
  return predicate;
}
