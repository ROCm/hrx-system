// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test dialect op generation hooks for the test IR generator.
//
// Generates constants, simple arithmetic/cast/compare ops, counter
// ops (for canonicalize testing), and region-bearing ops (map, loop,
// branch, isolated_region) from the test dialect.

#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/gen.h"

//===----------------------------------------------------------------------===//
// test.constant
//===----------------------------------------------------------------------===//

static iree_status_t loom_test_gen_hook_test_constant(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_type_t type =
      loom_test_gen_type_palette_pick(context->gen, context->palette);
  loom_attribute_t value = loom_test_gen_constant_attr(
      type, (int64_t)loom_test_gen_next_range(context->gen, 200) - 100);

  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_test_constant_build(context->builder, value, type,
                                                LOOM_LOCATION_UNKNOWN, &op));
  loom_test_gen_values_add(context->values, loom_op_results(op)[0], type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// test.addi, test.neg, test.cast, test.cmp (simple ops)
//===----------------------------------------------------------------------===//

static iree_status_t loom_test_gen_hook_test_simple(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;

  enum {
    SIMPLE_ADDI = 0,
    SIMPLE_NEG,
    SIMPLE_CAST,
    SIMPLE_CMP,
    SIMPLE_COUNT,
  };
  uint32_t op_kind = loom_test_gen_next_range(context->gen, SIMPLE_COUNT);

  switch (op_kind) {
    case SIMPLE_ADDI: {
      // test.addi: binary, same-type operands.
      loom_value_id_t lhs, rhs;
      loom_type_t type;
      if (!loom_test_gen_values_pick_binary_integer(
              context->gen, context->values, &lhs, &rhs, &type)) {
        *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
        return iree_ok_status();
      }
      loom_op_t* op = NULL;
      IREE_RETURN_IF_ERROR(loom_test_addi_build(
          context->builder, lhs, rhs, type, LOOM_LOCATION_UNKNOWN, &op));
      loom_test_gen_values_add(context->values, loom_op_results(op)[0], type);
      *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
      return iree_ok_status();
    }
    case SIMPLE_NEG: {
      // test.neg: unary float op. Consult the vtable constraint.
      loom_type_t type = {0};
      loom_value_id_t input = loom_test_gen_values_pick_for_constraint(
          context->gen, context->values, LOOM_TYPE_CONSTRAINT_FLOAT, &type);
      if (input == LOOM_VALUE_ID_INVALID) {
        *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
        return iree_ok_status();
      }
      loom_op_t* op = NULL;
      IREE_RETURN_IF_ERROR(loom_test_neg_build(context->builder, input, type,
                                               LOOM_LOCATION_UNKNOWN, &op));
      loom_test_gen_values_add(context->values, loom_op_results(op)[0], type);
      *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
      return iree_ok_status();
    }
    case SIMPLE_CAST: {
      // test.cast: INTEGER input → FLOAT output (from vtable constraints).
      loom_type_t input_type = {0};
      loom_value_id_t input = loom_test_gen_values_pick_for_constraint(
          context->gen, context->values, LOOM_TYPE_CONSTRAINT_INTEGER,
          &input_type);
      if (input == LOOM_VALUE_ID_INVALID) {
        *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
        return iree_ok_status();
      }
      // Pick a float type for the result.
      loom_type_t result_type = {0};
      if (!loom_test_gen_type_palette_pick_constrained(
              context->gen, context->palette, LOOM_TYPE_CONSTRAINT_FLOAT,
              &result_type)) {
        *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
        return iree_ok_status();
      }
      loom_op_t* op = NULL;
      IREE_RETURN_IF_ERROR(loom_test_cast_build(context->builder, input,
                                                input_type, result_type,
                                                LOOM_LOCATION_UNKNOWN, &op));
      loom_test_gen_values_add(context->values, loom_op_results(op)[0],
                               result_type);
      *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
      return iree_ok_status();
    }
    case SIMPLE_CMP: {
      // test.cmp: integer binary comparison → i1 result.
      loom_value_id_t lhs, rhs;
      loom_type_t operand_type;
      if (!loom_test_gen_values_pick_binary_integer(
              context->gen, context->values, &lhs, &rhs, &operand_type)) {
        *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
        return iree_ok_status();
      }
      uint8_t predicate = (uint8_t)loom_test_gen_next_range(
          context->gen, LOOM_TEST_CMP_PREDICATE_COUNT_);
      loom_type_t i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
      loom_op_t* op = NULL;
      IREE_RETURN_IF_ERROR(loom_test_cmp_build(context->builder, predicate, lhs,
                                               rhs, operand_type, i1_type,
                                               LOOM_LOCATION_UNKNOWN, &op));
      loom_test_gen_values_add(context->values, loom_op_results(op)[0],
                               i1_type);
      *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
      return iree_ok_status();
    }
  }

  *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// test.counter (canonicalize testing)
//===----------------------------------------------------------------------===//

static iree_status_t loom_test_gen_hook_test_counter(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_type_t type =
      loom_test_gen_type_palette_pick(context->gen, context->palette);
  // Values 0-5 produce interesting canonicalization behavior:
  // 0 is the fixed point, positive values decrement each pass.
  int64_t value = (int64_t)loom_test_gen_next_range(context->gen, 6);
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_test_counter_build(context->builder, value, type,
                                               LOOM_LOCATION_UNKNOWN, &op));
  loom_test_gen_values_add(context->values, loom_op_results(op)[0], type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// test.reduce (variadic operand testing)
//===----------------------------------------------------------------------===//

static iree_status_t loom_test_gen_hook_test_reduce(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  // Pick a type that has values, then gather 2-4 operands of that type.
  loom_value_id_t first =
      loom_test_gen_values_pick_any(context->gen, context->values);
  if (first == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }
  loom_type_t type = loom_test_gen_values_type_of(context->values, first);
  if (!loom_type_is_scalar(type)) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }
  loom_scalar_type_t scalar = loom_type_element_type(type);

  uint32_t operand_count = 2 + loom_test_gen_next_range(context->gen, 3);
  loom_value_id_t operands[5];
  operands[0] = first;
  for (uint32_t i = 1; i < operand_count; ++i) {
    operands[i] =
        loom_test_gen_values_pick_typed(context->gen, context->values, scalar);
    if (operands[i] == LOOM_VALUE_ID_INVALID) {
      operands[i] = first;  // Reuse first if not enough values of this type.
    }
  }

  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_test_reduce_build(context->builder, operands,
                                              operand_count, type,
                                              LOOM_LOCATION_UNKNOWN, &op));
  loom_test_gen_values_add(context->values, loom_op_results(op)[0], type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// test.map
//===----------------------------------------------------------------------===//

static iree_status_t loom_test_gen_region_map(
    const loom_test_gen_hook_context_t* context,
    const loom_test_gen_body_config_t* nested_config,
    loom_test_gen_hook_result_t* out_result) {
  loom_type_t type = {0};
  loom_value_id_t input = loom_test_gen_values_pick_for_constraint(
      context->gen, context->values, LOOM_TYPE_CONSTRAINT_TILE, &type);
  if (input == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }

  loom_op_t* map_op = NULL;
  IREE_RETURN_IF_ERROR(loom_test_map_build(context->builder, &input, 1, type,
                                           NULL, 0, LOOM_LOCATION_UNKNOWN,
                                           &map_op));

  loom_region_t* body = loom_test_map_body(map_op);
  loom_builder_ip_t saved =
      loom_builder_enter_region(context->builder, map_op, body);

  loom_test_gen_values_t body_values;
  loom_test_gen_values_initialize(&body_values);
  loom_block_t* entry_block = loom_region_entry_block(body);
  for (uint16_t j = 0; j < entry_block->arg_count; ++j) {
    loom_value_id_t arg_id = loom_block_arg_id(entry_block, j);
    loom_value_t* arg_val = &context->builder->module->values.entries[arg_id];
    loom_test_gen_values_add(&body_values, arg_id, arg_val->type);
  }

  IREE_RETURN_IF_ERROR(
      loom_test_gen_body_internal(context->gen, nested_config, context->builder,
                                  &body_values, context->current_depth + 1));

  loom_value_id_t yield_val =
      loom_test_gen_values_pick_any(context->gen, &body_values);
  if (yield_val == LOOM_VALUE_ID_INVALID) {
    loom_attribute_t const_val = loom_test_gen_constant_attr(type, 0);
    loom_op_t* const_op = NULL;
    IREE_RETURN_IF_ERROR(loom_test_constant_build(
        context->builder, const_val, type, LOOM_LOCATION_UNKNOWN, &const_op));
    yield_val = loom_op_results(const_op)[0];
  }
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_test_yield_build(context->builder, &yield_val, 1,
                                             LOOM_LOCATION_UNKNOWN, &yield_op));

  loom_builder_restore(context->builder, saved);
  loom_test_gen_values_add(context->values, loom_op_results(map_op)[0], type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// test.loop
//===----------------------------------------------------------------------===//

static iree_status_t loom_test_gen_region_loop(
    const loom_test_gen_hook_context_t* context,
    const loom_test_gen_body_config_t* nested_config,
    loom_test_gen_hook_result_t* out_result) {
  loom_value_id_t lower = loom_test_gen_values_pick_typed(
      context->gen, context->values, LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t upper = loom_test_gen_values_pick_typed(
      context->gen, context->values, LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t step = loom_test_gen_values_pick_typed(
      context->gen, context->values, LOOM_SCALAR_TYPE_INDEX);

  // If no index values exist, create constants for them.
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  if (lower == LOOM_VALUE_ID_INVALID || upper == LOOM_VALUE_ID_INVALID ||
      step == LOOM_VALUE_ID_INVALID) {
    for (int k = 0; k < 3; ++k) {
      loom_attribute_t const_val = loom_test_gen_constant_attr(
          index_type, (k == 2) ? 1 : (int64_t)k * 10);
      loom_op_t* const_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_test_constant_build(context->builder, const_val, index_type,
                                   LOOM_LOCATION_UNKNOWN, &const_op));
      loom_value_id_t id = loom_op_results(const_op)[0];
      loom_test_gen_values_add(context->values, id, index_type);
      if (k == 0)
        lower = id;
      else if (k == 1)
        upper = id;
      else
        step = id;
    }
  }

  // Pick 1-2 iter_args.
  uint16_t iter_arg_count =
      (uint16_t)(1 + loom_test_gen_next_range(context->gen, 2));
  loom_value_id_t iter_args[2];
  loom_type_t result_types[2];
  for (uint16_t j = 0; j < iter_arg_count; ++j) {
    iter_args[j] = loom_test_gen_values_pick_any(context->gen, context->values);
    if (iter_args[j] == LOOM_VALUE_ID_INVALID) {
      loom_type_t type =
          loom_test_gen_type_palette_pick(context->gen, context->palette);
      loom_attribute_t const_val =
          loom_test_gen_constant_attr(type, (int64_t)j);
      loom_op_t* const_op = NULL;
      IREE_RETURN_IF_ERROR(loom_test_constant_build(
          context->builder, const_val, type, LOOM_LOCATION_UNKNOWN, &const_op));
      iter_args[j] = loom_op_results(const_op)[0];
      loom_test_gen_values_add(context->values, iter_args[j], type);
    }
    result_types[j] =
        loom_test_gen_values_type_of(context->values, iter_args[j]);
  }

  loom_op_t* loop_op = NULL;
  IREE_RETURN_IF_ERROR(loom_test_loop_build(
      context->builder, lower, upper, step, iter_args, iter_arg_count,
      result_types, iter_arg_count, NULL, 0, LOOM_LOCATION_UNKNOWN, &loop_op));

  loom_region_t* body = loom_test_loop_body(loop_op);
  loom_builder_ip_t saved =
      loom_builder_enter_region(context->builder, loop_op, body);

  loom_test_gen_values_t body_values;
  loom_test_gen_values_initialize(&body_values);
  loom_block_t* entry_block = loom_region_entry_block(body);
  for (uint16_t j = 0; j < entry_block->arg_count; ++j) {
    loom_value_id_t arg_id = loom_block_arg_id(entry_block, j);
    loom_value_t* arg_val = &context->builder->module->values.entries[arg_id];
    loom_test_gen_values_add(&body_values, arg_id, arg_val->type);
  }

  IREE_RETURN_IF_ERROR(
      loom_test_gen_body_internal(context->gen, nested_config, context->builder,
                                  &body_values, context->current_depth + 1));

  // Yield values matching iter_arg types.
  loom_value_id_t yield_vals[2];
  for (uint16_t j = 0; j < iter_arg_count; ++j) {
    loom_scalar_type_t scalar = loom_type_element_type(result_types[j]);
    yield_vals[j] =
        loom_test_gen_values_pick_typed(context->gen, &body_values, scalar);
    if (yield_vals[j] == LOOM_VALUE_ID_INVALID) {
      yield_vals[j] = loom_test_gen_values_pick_any(context->gen, &body_values);
    }
    if (yield_vals[j] == LOOM_VALUE_ID_INVALID) {
      loom_attribute_t const_val =
          loom_test_gen_constant_attr(result_types[j], (int64_t)j);
      loom_op_t* const_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_test_constant_build(context->builder, const_val, result_types[j],
                                   LOOM_LOCATION_UNKNOWN, &const_op));
      yield_vals[j] = loom_op_results(const_op)[0];
    }
  }
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_test_yield_build(context->builder, yield_vals,
                                             iter_arg_count,
                                             LOOM_LOCATION_UNKNOWN, &yield_op));

  loom_builder_restore(context->builder, saved);
  for (uint16_t j = 0; j < iter_arg_count; ++j) {
    loom_test_gen_values_add(context->values, loom_op_results(loop_op)[j],
                             result_types[j]);
  }
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// test.branch
//===----------------------------------------------------------------------===//

// Generates a region body and yields values matching |result_types|.
// Used for both then and else branches.
static iree_status_t loom_test_gen_branch_region(
    const loom_test_gen_hook_context_t* context,
    const loom_test_gen_body_config_t* nested_config,
    const loom_type_t* result_types, uint16_t result_count) {
  loom_test_gen_values_t region_values;
  loom_test_gen_values_initialize(&region_values);
  for (uint16_t j = 0; j < context->values->count; ++j) {
    loom_test_gen_values_add(&region_values, context->values->entries[j],
                             context->values->types[j]);
  }

  IREE_RETURN_IF_ERROR(
      loom_test_gen_body_internal(context->gen, nested_config, context->builder,
                                  &region_values, context->current_depth + 1));

  loom_value_id_t yields[2];
  for (uint16_t j = 0; j < result_count; ++j) {
    loom_scalar_type_t scalar = loom_type_element_type(result_types[j]);
    yields[j] =
        loom_test_gen_values_pick_typed(context->gen, &region_values, scalar);
    if (yields[j] == LOOM_VALUE_ID_INVALID) {
      yields[j] = loom_test_gen_values_pick_any(context->gen, &region_values);
    }
    if (yields[j] == LOOM_VALUE_ID_INVALID) {
      loom_attribute_t const_val =
          loom_test_gen_constant_attr(result_types[j], (int64_t)j);
      loom_op_t* const_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_test_constant_build(context->builder, const_val, result_types[j],
                                   LOOM_LOCATION_UNKNOWN, &const_op));
      yields[j] = loom_op_results(const_op)[0];
    }
  }
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_test_yield_build(context->builder, yields,
                                             result_count,
                                             LOOM_LOCATION_UNKNOWN, &yield_op));
  return iree_ok_status();
}

static iree_status_t loom_test_gen_region_branch(
    const loom_test_gen_hook_context_t* context,
    const loom_test_gen_body_config_t* nested_config,
    loom_test_gen_hook_result_t* out_result) {
  loom_value_id_t condition = loom_test_gen_values_pick_typed(
      context->gen, context->values, LOOM_SCALAR_TYPE_I1);
  if (condition == LOOM_VALUE_ID_INVALID) {
    loom_type_t i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
    loom_attribute_t const_val = loom_test_gen_constant_attr(
        i1_type, loom_test_gen_next_bool(context->gen) ? 1 : 0);
    loom_op_t* const_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_test_constant_build(context->builder, const_val, i1_type,
                                 LOOM_LOCATION_UNKNOWN, &const_op));
    condition = loom_op_results(const_op)[0];
    loom_test_gen_values_add(context->values, condition, i1_type);
  }

  uint16_t result_count =
      (uint16_t)(1 + loom_test_gen_next_range(context->gen, 2));
  loom_type_t result_types[2];
  for (uint16_t j = 0; j < result_count; ++j) {
    result_types[j] =
        loom_test_gen_type_palette_pick(context->gen, context->palette);
  }

  loom_op_t* branch_op = NULL;
  IREE_RETURN_IF_ERROR(loom_test_branch_build(
      context->builder, condition, result_types, result_count, NULL, 0,
      LOOM_LOCATION_UNKNOWN, &branch_op));

  // Generate then region.
  loom_region_t* then_region = loom_test_branch_then_region(branch_op);
  loom_builder_ip_t saved =
      loom_builder_enter_region(context->builder, branch_op, then_region);
  IREE_RETURN_IF_ERROR(loom_test_gen_branch_region(context, nested_config,
                                                   result_types, result_count));

  // Generate else region.
  loom_region_t* else_region = loom_test_branch_else_region(branch_op);
  loom_builder_enter_region(context->builder, branch_op, else_region);
  IREE_RETURN_IF_ERROR(loom_test_gen_branch_region(context, nested_config,
                                                   result_types, result_count));

  loom_builder_restore(context->builder, saved);
  for (uint16_t j = 0; j < result_count; ++j) {
    loom_test_gen_values_add(context->values, loom_op_results(branch_op)[j],
                             result_types[j]);
  }
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// test.isolated_region
//===----------------------------------------------------------------------===//

static iree_status_t loom_test_gen_region_isolated(
    const loom_test_gen_hook_context_t* context,
    const loom_test_gen_body_config_t* nested_config,
    loom_test_gen_hook_result_t* out_result) {
  uint16_t result_count =
      (uint16_t)(1 + loom_test_gen_next_range(context->gen, 2));
  loom_type_t result_types[2];
  for (uint16_t j = 0; j < result_count; ++j) {
    result_types[j] =
        loom_test_gen_type_palette_pick(context->gen, context->palette);
  }

  loom_op_t* isolated_op = NULL;
  IREE_RETURN_IF_ERROR(loom_test_isolated_region_build(
      context->builder, result_types, result_count, NULL, 0,
      LOOM_LOCATION_UNKNOWN, &isolated_op));

  loom_region_t* body = loom_test_isolated_region_body(isolated_op);
  loom_builder_ip_t saved =
      loom_builder_enter_region(context->builder, isolated_op, body);

  loom_test_gen_values_t body_values;
  loom_test_gen_values_initialize(&body_values);

  // Seed isolated region with constants so body gen has values.
  for (uint16_t j = 0; j < context->palette->count; ++j) {
    loom_type_t type = loom_type_scalar(context->palette->types[j]);
    loom_attribute_t const_val = loom_test_gen_constant_attr(type, (int64_t)j);
    loom_op_t* const_op = NULL;
    IREE_RETURN_IF_ERROR(loom_test_constant_build(
        context->builder, const_val, type, LOOM_LOCATION_UNKNOWN, &const_op));
    loom_test_gen_values_add(&body_values, loom_op_results(const_op)[0], type);
  }

  IREE_RETURN_IF_ERROR(
      loom_test_gen_body_internal(context->gen, nested_config, context->builder,
                                  &body_values, context->current_depth + 1));

  // Yield values matching result types.
  loom_value_id_t yield_vals[2];
  for (uint16_t j = 0; j < result_count; ++j) {
    loom_scalar_type_t scalar = loom_type_element_type(result_types[j]);
    yield_vals[j] =
        loom_test_gen_values_pick_typed(context->gen, &body_values, scalar);
    if (yield_vals[j] == LOOM_VALUE_ID_INVALID) {
      yield_vals[j] = loom_test_gen_values_pick_any(context->gen, &body_values);
    }
    if (yield_vals[j] == LOOM_VALUE_ID_INVALID) {
      loom_attribute_t const_val =
          loom_test_gen_constant_attr(result_types[j], (int64_t)j);
      loom_op_t* const_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_test_constant_build(context->builder, const_val, result_types[j],
                                   LOOM_LOCATION_UNKNOWN, &const_op));
      yield_vals[j] = loom_op_results(const_op)[0];
    }
  }
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_test_yield_build(context->builder, yield_vals,
                                             result_count,
                                             LOOM_LOCATION_UNKNOWN, &yield_op));

  loom_builder_restore(context->builder, saved);
  for (uint16_t j = 0; j < result_count; ++j) {
    loom_test_gen_values_add(context->values, loom_op_results(isolated_op)[j],
                             result_types[j]);
  }
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Region dispatch
//===----------------------------------------------------------------------===//

static iree_status_t loom_test_gen_hook_test_region(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;

  if (context->current_depth >= context->body_config->max_nesting_depth) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }
  if (!loom_test_gen_next_probability(
          context->gen, context->body_config->nesting_probability)) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }

  enum {
    REGION_MAP = 0,
    REGION_LOOP,
    REGION_BRANCH,
    REGION_ISOLATED,
    REGION_COUNT,
  };

  // Nested body gets a fraction of the remaining op budget.
  loom_test_gen_body_config_t nested_config = *context->body_config;
  nested_config.op_count =
      context->ops_remaining > 3 ? context->ops_remaining / 3 : 1;
  nested_config.block_arg_count = 0;

  switch (loom_test_gen_next_range(context->gen, REGION_COUNT)) {
    case REGION_MAP:
      return loom_test_gen_region_map(context, &nested_config, out_result);
    case REGION_LOOP:
      return loom_test_gen_region_loop(context, &nested_config, out_result);
    case REGION_BRANCH:
      return loom_test_gen_region_branch(context, &nested_config, out_result);
    case REGION_ISOLATED:
      return loom_test_gen_region_isolated(context, &nested_config, out_result);
  }

  *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Hook table
//===----------------------------------------------------------------------===//

static const loom_test_gen_op_hook_t loom_test_gen_test_hooks_table[] = {
    {3, loom_test_gen_hook_test_constant, NULL, NULL},
    {3, loom_test_gen_hook_test_simple, NULL, NULL},
    {1, loom_test_gen_hook_test_counter, NULL, NULL},
    {1, loom_test_gen_hook_test_reduce, NULL, NULL},
    {2, loom_test_gen_hook_test_region, NULL, NULL},
};

const loom_test_gen_op_hook_t* loom_test_gen_test_hooks(
    iree_host_size_t* out_hook_count) {
  *out_hook_count = sizeof(loom_test_gen_test_hooks_table) /
                    sizeof(loom_test_gen_test_hooks_table[0]);
  return loom_test_gen_test_hooks_table;
}
