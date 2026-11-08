// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/ir/module.h"
#include "loom/tooling/testbench/numeric_value.h"
#include "loom/tooling/testbench/testbench.h"

enum {
  LOOM_TESTBENCH_MAX_SHAPE_RANK = 32,
};

static iree_string_view_t loom_testbench_string_from_id(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return loom_string_table_get(&module->strings, string_id);
}

static loom_type_t loom_testbench_value_type(const loom_module_t* module,
                                             loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return (loom_type_t){0};
  }
  return loom_module_value_type(module, value_id);
}

static bool loom_testbench_value_ids_are_in_range(
    const loom_module_t* module, const loom_value_id_t* value_ids,
    iree_host_size_t value_count) {
  for (iree_host_size_t value_index = 0; value_index < value_count;
       ++value_index) {
    if (value_ids[value_index] >= module->values.count) {
      return false;
    }
  }
  return true;
}

bool loom_testbench_is_value_source_op(const loom_op_t* op) {
  return loom_check_literal_isa(op) || loom_check_generate_iota_isa(op) ||
         loom_check_generate_fill_isa(op) ||
         loom_check_generate_random_uniform_isa(op) ||
         loom_check_file_read_npy_isa(op) || loom_check_tensor_view_isa(op) ||
         loom_check_entropy_fork_isa(op) || loom_check_entropy_read_isa(op);
}

bool loom_testbench_plan_value_source(
    const loom_module_t* module, const loom_op_t* op,
    loom_testbench_value_source_plan_t* out_source) {
  memset(out_source, 0, sizeof(*out_source));
  out_source->op = op;
  if (loom_check_literal_isa(op)) {
    out_source->kind = LOOM_TESTBENCH_VALUE_SOURCE_LITERAL;
    out_source->value_id = loom_check_literal_result(op);
    out_source->type = loom_testbench_value_type(module, out_source->value_id);
    out_source->literal.value = loom_check_literal_value(op);
    return out_source->value_id < module->values.count;
  }
  if (loom_check_generate_iota_isa(op)) {
    out_source->kind = LOOM_TESTBENCH_VALUE_SOURCE_IOTA;
    out_source->value_id = loom_check_generate_iota_result(op);
    out_source->type = loom_testbench_value_type(module, out_source->value_id);
    out_source->iota.offset = loom_check_generate_iota_offset(op);
    out_source->iota.step = loom_check_generate_iota_step(op);
    if (loom_check_generate_iota_has_period(op)) {
      int64_t period = loom_check_generate_iota_period(op);
      if (period <= 0 || (uint64_t)period > (uint64_t)IREE_HOST_SIZE_MAX) {
        return false;
      }
      out_source->iota.period = (iree_host_size_t)period;
    }
    return out_source->value_id < module->values.count;
  }
  if (loom_check_generate_fill_isa(op)) {
    out_source->kind = LOOM_TESTBENCH_VALUE_SOURCE_FILL;
    out_source->value_id = loom_check_generate_fill_result(op);
    out_source->type = loom_testbench_value_type(module, out_source->value_id);
    out_source->fill.value = loom_check_generate_fill_value(op);
    return out_source->value_id < module->values.count;
  }
  if (loom_check_generate_random_uniform_isa(op)) {
    out_source->kind = LOOM_TESTBENCH_VALUE_SOURCE_RANDOM_UNIFORM;
    out_source->value_id = loom_check_generate_random_uniform_result(op);
    out_source->type = loom_testbench_value_type(module, out_source->value_id);
    out_source->random_uniform.seed_value_id =
        loom_check_generate_random_uniform_seed(op);
    out_source->random_uniform.lower =
        loom_check_generate_random_uniform_lower(op);
    out_source->random_uniform.upper =
        loom_check_generate_random_uniform_upper(op);
    return out_source->value_id < module->values.count &&
           out_source->random_uniform.seed_value_id < module->values.count;
  }
  if (loom_check_file_read_npy_isa(op)) {
    out_source->kind = LOOM_TESTBENCH_VALUE_SOURCE_FILE_READ_NPY;
    out_source->value_id = loom_check_file_read_npy_result(op);
    out_source->type = loom_testbench_value_type(module, out_source->value_id);
    out_source->file.path_id = loom_check_file_read_npy_path(op);
    out_source->file.path =
        loom_testbench_string_from_id(module, out_source->file.path_id);
    return out_source->value_id < module->values.count &&
           out_source->file.path_id < module->strings.count;
  }
  if (loom_check_tensor_view_isa(op)) {
    out_source->kind = LOOM_TESTBENCH_VALUE_SOURCE_TENSOR_VIEW;
    out_source->value_id = loom_check_tensor_view_result(op);
    out_source->type = loom_testbench_value_type(module, out_source->value_id);
    out_source->tensor_view.source_value_id = loom_check_tensor_view_source(op);
    int64_t byte_offset = loom_check_tensor_view_byte_offset(op);
    if (out_source->value_id >= module->values.count || byte_offset < 0 ||
        out_source->tensor_view.source_value_id >= module->values.count) {
      return false;
    }
    const loom_value_t* source_value =
        loom_module_value(module, out_source->tensor_view.source_value_id);
    if (loom_value_is_block_arg(source_value)) {
      return false;
    }
    const loom_op_t* source_op = loom_value_def_op(source_value);
    if (!source_op || !loom_testbench_is_value_source_op(source_op)) {
      return false;
    }
    out_source->tensor_view.byte_offset = (iree_device_size_t)byte_offset;
    return true;
  }
  if (loom_check_entropy_fork_isa(op)) {
    out_source->kind = LOOM_TESTBENCH_VALUE_SOURCE_ENTROPY_FORK;
    out_source->value_id = loom_check_entropy_fork_result(op);
    out_source->type = loom_testbench_value_type(module, out_source->value_id);
    out_source->entropy_fork.entropy_value_id =
        loom_check_entropy_fork_entropy(op);
    out_source->entropy_fork.name_id = loom_check_entropy_fork_fork_name(op);
    out_source->entropy_fork.name =
        loom_testbench_string_from_id(module, out_source->entropy_fork.name_id);
    return out_source->value_id < module->values.count &&
           out_source->entropy_fork.entropy_value_id < module->values.count &&
           out_source->entropy_fork.name_id < module->strings.count &&
           !iree_string_view_is_empty(out_source->entropy_fork.name);
  }
  if (loom_check_entropy_read_isa(op)) {
    const loom_value_slice_t ordinals = loom_check_entropy_read_ordinals(op);
    const loom_i64_array_t static_ordinals =
        loom_attr_as_i64_array(loom_check_entropy_read_static_ordinals(op));
    out_source->kind = LOOM_TESTBENCH_VALUE_SOURCE_ENTROPY_READ;
    out_source->value_id = loom_check_entropy_read_result(op);
    out_source->type = loom_testbench_value_type(module, out_source->value_id);
    out_source->entropy_read.entropy_value_id =
        loom_check_entropy_read_entropy(op);
    out_source->entropy_read.ordinal_value_ids = ordinals.values;
    out_source->entropy_read.ordinal_value_count = ordinals.count;
    out_source->entropy_read.static_ordinals = static_ordinals.values;
    out_source->entropy_read.static_ordinal_count = static_ordinals.count;
    if (out_source->value_id >= module->values.count ||
        out_source->entropy_read.entropy_value_id >= module->values.count ||
        static_ordinals.count != 1) {
      return false;
    }
    for (iree_host_size_t i = 0; i < ordinals.count; ++i) {
      if (ordinals.values[i] >= module->values.count) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool loom_testbench_is_expectation_op(const loom_op_t* op) {
  return loom_check_expectation_classify(op) != LOOM_CHECK_EXPECTATION_NONE;
}

bool loom_testbench_plan_expectation(
    const loom_module_t* module, const loom_op_t* op,
    loom_testbench_expectation_plan_t* out_expectation) {
  memset(out_expectation, 0, sizeof(*out_expectation));
  out_expectation->op = op;
  out_expectation->expected_value_id = LOOM_VALUE_ID_INVALID;
  out_expectation->kind = loom_check_expectation_classify(op);
  if (loom_check_expectation_kind_is_pair(out_expectation->kind)) {
    out_expectation->actual_value_id = loom_op_const_operands(op)[0];
    out_expectation->expected_value_id = loom_op_const_operands(op)[1];
  }

  switch (out_expectation->kind) {
    case LOOM_CHECK_EXPECTATION_EQUAL:
    case LOOM_CHECK_EXPECTATION_BITWISE:
      break;
    case LOOM_CHECK_EXPECTATION_APPROXIMATE:
      out_expectation->approximate.absolute_tolerance =
          loom_check_expect_approximate_atol(op);
      out_expectation->approximate.relative_tolerance =
          loom_check_expect_approximate_rtol(op);
      out_expectation->approximate.nan_policy =
          loom_check_expect_approximate_nan(op);
      out_expectation->approximate.flush_subnormals =
          loom_check_expect_approximate_subnormal(op) ==
          LOOM_CHECK_EXPECT_APPROXIMATE_SUBNORMAL_FLUSH;
      break;
    case LOOM_CHECK_EXPECTATION_CLOSE_SIGNED:
    case LOOM_CHECK_EXPECTATION_CLOSE_UNSIGNED: {
      const int64_t maximum_distance =
          out_expectation->kind == LOOM_CHECK_EXPECTATION_CLOSE_SIGNED
              ? loom_check_expect_close_signed_max_distance(op)
              : loom_check_expect_close_unsigned_max_distance(op);
      if (maximum_distance < 0) {
        return false;
      }
      out_expectation->integer_close.maximum_distance =
          (uint64_t)maximum_distance;
      break;
    }
    case LOOM_CHECK_EXPECTATION_SHAPE: {
      out_expectation->actual_value_id = loom_check_expect_shape_value(op);
      const loom_value_slice_t dimensions = loom_check_expect_shape_dims(op);
      const loom_attribute_t static_dimensions =
          loom_check_expect_shape_static_dims(op);
      out_expectation->shape.dimension_value_ids = dimensions.values;
      out_expectation->shape.dimension_value_count = dimensions.count;
      out_expectation->shape.static_dimensions = static_dimensions.i64_array;
      out_expectation->shape.static_dimension_count = static_dimensions.count;
      break;
    }
    case LOOM_CHECK_EXPECTATION_EVENT:
      out_expectation->actual_value_id = LOOM_VALUE_ID_INVALID;
      out_expectation->event.provider_id = loom_check_expect_event_provider(op);
      out_expectation->event.provider = loom_testbench_string_from_id(
          module, out_expectation->event.provider_id);
      out_expectation->event.attrs = loom_check_expect_event_attrs(op);
      break;
    default:
      return false;
  }

  if (out_expectation->kind != LOOM_CHECK_EXPECTATION_EVENT &&
      out_expectation->actual_value_id >= module->values.count) {
    return false;
  }
  out_expectation->type =
      out_expectation->kind == LOOM_CHECK_EXPECTATION_EVENT
          ? (loom_type_t){0}
          : loom_testbench_value_type(module, out_expectation->actual_value_id);

  switch (out_expectation->kind) {
    case LOOM_CHECK_EXPECTATION_EQUAL:
    case LOOM_CHECK_EXPECTATION_BITWISE:
      return out_expectation->expected_value_id < module->values.count;
    case LOOM_CHECK_EXPECTATION_APPROXIMATE: {
      if (out_expectation->expected_value_id >= module->values.count) {
        return false;
      }
      const loom_type_t expected_type =
          loom_testbench_value_type(module, out_expectation->expected_value_id);
      const loom_testbench_numeric_type_t* actual_numeric_type =
          loom_testbench_numeric_type_from_scalar_type(
              loom_type_element_type(out_expectation->type));
      const loom_testbench_numeric_type_t* expected_numeric_type =
          loom_testbench_numeric_type_from_scalar_type(
              loom_type_element_type(expected_type));
      if (!actual_numeric_type || !expected_numeric_type ||
          !loom_testbench_numeric_type_is_float(actual_numeric_type) ||
          !loom_testbench_numeric_type_is_float(expected_numeric_type)) {
        return false;
      }
      out_expectation->approximate.actual_numeric_kind =
          (uint8_t)actual_numeric_type->kind;
      out_expectation->approximate.expected_numeric_kind =
          (uint8_t)expected_numeric_type->kind;
      return true;
    }
    case LOOM_CHECK_EXPECTATION_CLOSE_SIGNED:
    case LOOM_CHECK_EXPECTATION_CLOSE_UNSIGNED: {
      if (out_expectation->expected_value_id >= module->values.count) {
        return false;
      }
      const bool is_signed =
          out_expectation->kind == LOOM_CHECK_EXPECTATION_CLOSE_SIGNED;
      const loom_type_t expected_type =
          loom_testbench_value_type(module, out_expectation->expected_value_id);
      const loom_scalar_type_t actual_element_type =
          loom_type_element_type(out_expectation->type);
      const loom_scalar_type_t expected_element_type =
          loom_type_element_type(expected_type);
      const loom_testbench_numeric_type_t* actual_numeric_type =
          is_signed ? loom_testbench_numeric_type_from_signed_scalar_type(
                          actual_element_type)
                    : loom_testbench_numeric_type_from_unsigned_scalar_type(
                          actual_element_type);
      const loom_testbench_numeric_type_t* expected_numeric_type =
          is_signed ? loom_testbench_numeric_type_from_signed_scalar_type(
                          expected_element_type)
                    : loom_testbench_numeric_type_from_unsigned_scalar_type(
                          expected_element_type);
      if (!actual_numeric_type || !expected_numeric_type) {
        return false;
      }
      out_expectation->integer_close.actual_numeric_kind =
          (uint8_t)actual_numeric_type->kind;
      out_expectation->integer_close.expected_numeric_kind =
          (uint8_t)expected_numeric_type->kind;
      return true;
    }
    case LOOM_CHECK_EXPECTATION_SHAPE: {
      if (out_expectation->shape.static_dimension_count >
          LOOM_TESTBENCH_MAX_SHAPE_RANK) {
        return false;
      }
      iree_host_size_t dynamic_dimension_count = 0;
      for (iree_host_size_t i = 0;
           i < out_expectation->shape.static_dimension_count; ++i) {
        if (out_expectation->shape.static_dimensions[i] == INT64_MIN) {
          ++dynamic_dimension_count;
        }
      }
      return dynamic_dimension_count ==
                 out_expectation->shape.dimension_value_count &&
             loom_testbench_value_ids_are_in_range(
                 module, out_expectation->shape.dimension_value_ids,
                 out_expectation->shape.dimension_value_count);
    }
    case LOOM_CHECK_EXPECTATION_EVENT:
      return out_expectation->event.provider_id < module->strings.count &&
             !iree_string_view_is_empty(out_expectation->event.provider);
    default:
      return false;
  }
}
