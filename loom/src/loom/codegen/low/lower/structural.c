// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/structural.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/emission.h"
#include "loom/codegen/low/lower/function.h"
#include "loom/codegen/low/lower/storage.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"

iree_status_t loom_low_lower_map_blocks(loom_low_lower_context_t* context,
                                        loom_region_t* source_body) {
  loom_region_t* low_body = loom_low_lower_context_low_body(context);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->function_arena, source_body->block_count,
      sizeof(*context->lowering.block_map),
      (void**)&context->lowering.block_map));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->function_arena, source_body->block_count,
      sizeof(*context->lowering.successor_interpositions),
      (void**)&context->lowering.successor_interpositions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->function_arena, source_body->block_count,
      sizeof(*context->lowering.branch_plans),
      (void**)&context->lowering.branch_plans));
  memset(context->lowering.block_map, 0,
         (iree_host_size_t)source_body->block_count *
             sizeof(*context->lowering.block_map));
  memset(context->lowering.successor_interpositions, 0,
         (iree_host_size_t)source_body->block_count *
             sizeof(*context->lowering.successor_interpositions));
  for (uint16_t i = 0; i < source_body->block_count; ++i) {
    context->lowering.branch_plans[i] = loom_low_lower_plan_empty();
  }

  for (uint16_t i = 0; i < source_body->block_count; ++i) {
    loom_block_t* source_block = loom_region_block(source_body, i);
    loom_block_t* low_block = NULL;
    if (i == 0) {
      low_block = loom_region_entry_block(low_body);
    } else {
      IREE_RETURN_IF_ERROR(
          loom_region_append_block(context->module, low_body, &low_block));
    }
    low_block->label_id = source_block->label_id;
    context->lowering.block_map[i] = low_block;
  }

  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    loom_block_t* source_block = loom_region_block(source_body, block_index);
    loom_block_t* low_block = context->lowering.block_map[block_index];
    if (block_index == 0) {
      const uint16_t direct_argument_count =
          loom_low_lower_direct_argument_count(context);
      IREE_ASSERT_EQ(low_block->arg_count, direct_argument_count);
      uint16_t direct_argument_index = 0;
      for (uint16_t arg_index = 0; arg_index < source_block->arg_count;
           ++arg_index) {
        if (context->lowering.argument_map[arg_index].kind !=
            LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
            context, source_block->arg_ids[arg_index],
            low_block->arg_ids[direct_argument_index]));
        ++direct_argument_index;
      }
      continue;
    }

    for (uint16_t arg_index = 0; arg_index < source_block->arg_count;
         ++arg_index) {
      loom_type_t low_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
          context, context->source_function.op,
          source_block->arg_ids[arg_index], &low_type));
      IREE_ASSERT_NE(loom_type_kind(low_type), LOOM_TYPE_NONE);
      loom_value_id_t low_arg = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
          &context->builder, low_block, low_type, &low_arg));
      IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
          context, source_block->arg_ids[arg_index], low_arg));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_prepare_branches(loom_low_lower_context_t* context,
                                              loom_region_t* source_body) {
  if (context->policy->prepare_branch.fn == NULL) {
    return iree_ok_status();
  }

  iree_arena_allocator_t analysis_arena;
  iree_arena_initialize(context->module->arena.block_pool, &analysis_arena);
  iree_status_t status = iree_ok_status();
  for (uint16_t block_index = 0;
       block_index < source_body->block_count && iree_status_is_ok(status);
       ++block_index) {
    const loom_block_t* source_block =
        loom_region_block(source_body, block_index);
    const loom_op_t* source_terminator = loom_block_const_last_op(source_block);
    if (source_terminator == NULL || source_terminator->successor_count == 0) {
      continue;
    }
    if (loom_low_lower_cfg_cond_br_exact_bool(context, source_terminator,
                                              NULL)) {
      continue;
    }
    status = context->policy->prepare_branch.fn(
        context->policy->prepare_branch.user_data, context, source_terminator,
        &analysis_arena);
    iree_arena_reset(&analysis_arena);
    if (loom_low_lower_context_should_stop(context)) {
      break;
    }
  }
  iree_arena_deinitialize(&analysis_arena);
  return status;
}

static iree_status_t loom_low_lower_remap_values(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_id_t* source_values, iree_host_size_t value_count,
    loom_value_id_t** out_low_values) {
  *out_low_values = NULL;
  if (value_count == 0) {
    return iree_ok_status();
  }
  loom_value_id_t* low_values = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, value_count, sizeof(*low_values), (void**)&low_values));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, source_values[i], &low_values[i]));
    loom_type_t required_low_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
        context, source_op, source_values[i], &required_low_type));
    IREE_RETURN_IF_ERROR(loom_low_lower_materialize_structural_operand(
        context, source_op, i, source_values[i], required_low_type,
        &low_values[i]));
  }
  *out_low_values = low_values;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_bind_or_elide_alias(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id) {
  if (loom_low_lower_source_value_has_low_mapping(context, source_value_id)) {
    return loom_low_lower_bind_value_alias(context, source_value_id,
                                           result_value_id);
  }
  if (!loom_low_lower_result_storage_required(context, result_value_id)) {
    return loom_low_lower_elide_value(context, result_value_id);
  }
  return loom_low_lower_bind_value_alias(context, source_value_id,
                                         result_value_id);
}

static iree_status_t loom_low_lower_bind_identity_results(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  IREE_ASSERT_EQ(source_op->operand_count, source_op->result_count);
  const loom_value_id_t* source_operands = loom_op_const_operands(source_op);
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_bind_or_elide_alias(
        context, source_operands[i], source_results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_func_ref_cast(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_id_t source_id = loom_func_ref_cast_source(source_op);
  const loom_value_id_t result_id = loom_func_ref_cast_result(source_op);
  loom_value_id_t low_source_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_id, &low_source_id));
  loom_type_t low_result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op, result_id,
                                                &low_result_type));

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_func_ref_cast_build(
      &context->builder, low_source_id, low_result_type, source_op->location,
      &low_op));
  return loom_low_lower_bind_value(context, result_id,
                                   loom_low_func_ref_cast_result(low_op));
}

static iree_status_t loom_low_lower_emit_region_ops(
    loom_low_lower_context_t* context, loom_region_t* source_region,
    bool map_source_blocks);

static iree_status_t loom_low_lower_map_op_result_types(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t** out_result_types) {
  *out_result_types = NULL;
  if (source_op->result_count == 0) {
    return iree_ok_status();
  }
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, source_op->result_count, sizeof(*result_types),
      (void**)&result_types));
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_check_mapped_value(
        context, source_op, source_results[i], &result_types[i]));
    if (loom_type_kind(result_types[i]) == LOOM_TYPE_NONE) {
      return iree_ok_status();
    }
  }
  *out_result_types = result_types;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_bind_op_results(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_op_t* low_op) {
  IREE_ASSERT_EQ(source_op->result_count, low_op->result_count);
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  const loom_value_id_t* low_results = loom_op_const_results(low_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_bind_value(context, source_results[i], low_results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_func_null(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op) {
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_op_result_types(context, source_op, &result_types));
  if (result_types == NULL) return iree_ok_status();
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_func_null_build(
      &context->builder, result_types[0], source_op->location, &low_op));
  return loom_low_lower_bind_op_results(context, source_op, low_op);
}

static iree_status_t loom_low_lower_func_compare_null(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t* operands = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
      context, source_op, loom_op_const_operands(source_op),
      source_op->operand_count, &operands));
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_op_result_types(context, source_op, &result_types));
  if (result_types == NULL) return iree_ok_status();
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_func_compare_null_build(
      &context->builder, operands[0], result_types[0], source_op->location,
      &low_op));
  return loom_low_lower_bind_op_results(context, source_op, low_op);
}

static iree_status_t loom_low_lower_func_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_op_result_types(context, source_op, &result_types));
  if (result_types == NULL) return iree_ok_status();
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_func_address_build(
      &context->builder, loom_func_address_callee(source_op), result_types[0],
      source_op->location, &low_op));
  return loom_low_lower_bind_op_results(context, source_op, low_op);
}

static iree_status_t loom_low_lower_func_import_resolved(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_op_result_types(context, source_op, &result_types));
  if (result_types == NULL) return iree_ok_status();
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_func_import_resolved_build(
      &context->builder, loom_func_import_resolved_callee(source_op),
      result_types[0], source_op->location, &low_op));
  return loom_low_lower_bind_op_results(context, source_op, low_op);
}

static iree_status_t loom_low_lower_emit_scf_yield(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_slice_t values = loom_scf_yield_values(source_op);
  loom_value_id_t* low_values = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
      context, source_op, values.values, values.count, &low_values));
  loom_op_t* low_yield_op = NULL;
  return loom_low_scf_yield_build(&context->builder, low_values, values.count,
                                  source_op->location, &low_yield_op);
}

static iree_status_t loom_low_lower_emit_scf_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scf_condition_condition(source_op), &low_condition));
  const loom_value_slice_t forwarded = loom_scf_condition_forwarded(source_op);
  loom_value_id_t* low_forwarded = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
      context, source_op, forwarded.values, forwarded.count, &low_forwarded));
  loom_op_t* low_condition_op = NULL;
  return loom_low_scf_condition_build(&context->builder, low_condition,
                                      low_forwarded, forwarded.count,
                                      source_op->location, &low_condition_op);
}

static iree_status_t loom_low_lower_emit_scf_if(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scf_if_condition(source_op), &low_condition));

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_op_result_types(context, source_op, &result_types));
  if (source_op->result_count != 0 && result_types == NULL) {
    return iree_ok_status();
  }

  loom_low_scf_if_build_flags_t build_flags = 0;
  if (loom_scf_if_else_region(source_op) != NULL) {
    build_flags |= LOOM_LOW_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION;
  }
  loom_op_t* low_if_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_scf_if_build(
      &context->builder, build_flags, low_condition, result_types,
      source_op->result_count, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &low_if_op));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_bind_op_results(context, source_op, low_if_op));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, low_if_op, loom_low_scf_if_then_region(low_if_op));
  iree_status_t status = loom_low_lower_emit_region_ops(
      context, loom_scf_if_then_region(source_op),
      /*map_source_blocks=*/false);
  loom_builder_restore(&context->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  loom_region_t* source_else_region = loom_scf_if_else_region(source_op);
  if (source_else_region == NULL) {
    return iree_ok_status();
  }
  saved_ip = loom_builder_enter_region(&context->builder, low_if_op,
                                       loom_low_scf_if_else_region(low_if_op));
  status = loom_low_lower_emit_region_ops(context, source_else_region,
                                          /*map_source_blocks=*/false);
  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

static iree_status_t loom_low_lower_bind_region_entry_args(
    loom_low_lower_context_t* context, const loom_region_t* source_region,
    const loom_region_t* low_region) {
  const loom_block_t* source_entry =
      loom_region_const_entry_block(source_region);
  const loom_block_t* low_entry = loom_region_const_entry_block(low_region);
  IREE_ASSERT_EQ(source_entry->arg_count, low_entry->arg_count);
  for (uint16_t i = 0; i < source_entry->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
        context, source_entry->arg_ids[i], low_entry->arg_ids[i]));
  }
  return iree_ok_status();
}

static bool loom_low_lower_op_attr_present(const loom_op_t* op,
                                           uint8_t attr_index) {
  return !loom_attr_is_absent(loom_op_attrs(op)[attr_index]);
}

static iree_status_t loom_low_lower_emit_scf_for(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_lower_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scf_for_lower_bound(source_op), &low_lower_bound));
  loom_value_id_t low_upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scf_for_upper_bound(source_op), &low_upper_bound));
  loom_value_id_t low_step = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_scf_for_step(source_op), &low_step));

  loom_value_slice_t iter_args = loom_scf_for_iter_args(source_op);
  loom_value_id_t* low_iter_args = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
      context, source_op, iter_args.values, iter_args.count, &low_iter_args));

  loom_value_id_t low_unroll_factor = LOOM_VALUE_ID_INVALID;
  loom_low_scf_for_build_flags_t build_flags = 0;
  if (loom_scf_for_unroll_factor_is_present(source_op)) {
    build_flags |= LOOM_LOW_SCF_FOR_BUILD_FLAG_HAS_UNROLL_FACTOR;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, loom_scf_for_unroll_factor(source_op), &low_unroll_factor));
  }

  uint8_t unroll_policy = 0;
  if (loom_low_lower_op_attr_present(source_op,
                                     loom_scf_for_unroll_policy_ATTR_INDEX)) {
    build_flags |= LOOM_LOW_SCF_FOR_BUILD_FLAG_HAS_UNROLL_POLICY;
    unroll_policy = (uint8_t)loom_scf_for_unroll_policy(source_op);
  }

  loom_op_t* low_for_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_scf_for_build(
      &context->builder, build_flags, low_lower_bound, low_upper_bound,
      low_step, low_iter_args, iter_args.count, /*tied_results=*/NULL,
      /*tied_result_count=*/0, low_unroll_factor, unroll_policy,
      source_op->location, &low_for_op));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_bind_op_results(context, source_op, low_for_op));
  IREE_RETURN_IF_ERROR(loom_low_lower_bind_region_entry_args(
      context, loom_scf_for_body(source_op),
      loom_low_scf_for_body(low_for_op)));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, low_for_op, loom_low_scf_for_body(low_for_op));
  iree_status_t status = loom_low_lower_emit_region_ops(
      context, loom_scf_for_body(source_op), /*map_source_blocks=*/false);
  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

static iree_status_t loom_low_lower_emit_scf_while(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_slice_t iter_args = loom_scf_while_iter_args(source_op);
  loom_value_id_t* low_iter_args = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
      context, source_op, iter_args.values, iter_args.count, &low_iter_args));

  loom_op_t* low_while_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_scf_while_build(
      &context->builder, low_iter_args, iter_args.count,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_while_op));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_bind_op_results(context, source_op, low_while_op));
  IREE_RETURN_IF_ERROR(loom_low_lower_bind_region_entry_args(
      context, loom_scf_while_before(source_op),
      loom_low_scf_while_before(low_while_op)));
  IREE_RETURN_IF_ERROR(loom_low_lower_bind_region_entry_args(
      context, loom_scf_while_after(source_op),
      loom_low_scf_while_after(low_while_op)));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, low_while_op, loom_low_scf_while_before(low_while_op));
  iree_status_t status =
      loom_low_lower_emit_region_ops(context, loom_scf_while_before(source_op),
                                     /*map_source_blocks=*/false);
  loom_builder_restore(&context->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  saved_ip = loom_builder_enter_region(&context->builder, low_while_op,
                                       loom_low_scf_while_after(low_while_op));
  status =
      loom_low_lower_emit_region_ops(context, loom_scf_while_after(source_op),
                                     /*map_source_blocks=*/false);
  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

// Moves each call result out of its ABI transfer identity before binding it to
// the source result. This leaves the call result available for target-fixed
// placement while preserving exact reference ownership in the semantic value.
static iree_status_t loom_low_lower_bind_call_results(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_op_t* low_call_op) {
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  const loom_value_id_t* low_results = loom_op_const_results(low_call_op);
  if (!iree_any_bit_set(context->policy->flags,
                        LOOM_LOW_LOWER_POLICY_FLAG_EXPLICIT_ABI_TRANSFERS)) {
    for (uint16_t i = 0; i < source_op->result_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(context, source_results[i],
                                                     low_results[i]));
    }
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    loom_op_t* move_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_move_build(
        &context->builder, low_results[i], /*detached=*/false,
        loom_module_value_type(context->module, low_results[i]),
        source_op->location, &move_op));
    IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
        context, source_results[i], loom_low_move_result(move_op)));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_structural_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_handled) {
  *out_handled = true;
  if (loom_func_ref_cast_isa(source_op)) {
    return loom_low_lower_func_ref_cast(context, source_op);
  }
  const loom_trait_flags_t traits =
      loom_op_effective_traits(context->module, source_op);
  if (loom_traits_are_fact_identity(traits)) {
    return loom_low_lower_bind_identity_results(context, source_op);
  }
  if (loom_traits_are_value_alias(traits)) {
    IREE_ASSERT(source_op->operand_count >= 1);
    IREE_ASSERT(source_op->result_count == 1);
    return loom_low_lower_bind_or_elide_alias(
        context, loom_op_const_operands(source_op)[0],
        loom_op_const_results(source_op)[0]);
  }
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_ASSUME_SAME_ROOT: {
      return loom_low_lower_bind_or_elide_alias(
          context, loom_buffer_assume_same_root_buffer(source_op),
          loom_buffer_assume_same_root_result(source_op));
    }
    case LOOM_OP_FUNC_NULL:
      return loom_low_lower_func_null(context, source_op);
    case LOOM_OP_FUNC_COMPARE_NULL:
      return loom_low_lower_func_compare_null(context, source_op);
    case LOOM_OP_FUNC_ADDRESS:
      return loom_low_lower_func_address(context, source_op);
    case LOOM_OP_FUNC_IMPORT_RESOLVED:
      return loom_low_lower_func_import_resolved(context, source_op);
    case LOOM_OP_FUNC_RETURN: {
      loom_value_slice_t values = loom_func_return_operands(source_op);
      loom_value_id_t* low_values = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
          context, source_op, values.values, values.count, &low_values));
      loom_op_t* low_return_op = NULL;
      return loom_low_return_build(&context->builder, low_values, values.count,
                                   source_op->location, &low_return_op);
    }
    case LOOM_OP_FUNC_CALL: {
      loom_value_slice_t operands = loom_func_call_operands(source_op);
      loom_value_id_t* low_operands = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
          context, source_op, operands.values, operands.count, &low_operands));

      const loom_value_id_t* source_results = loom_op_const_results(source_op);
      loom_type_t* result_types = NULL;
      bool has_unmapped_result = false;
      if (source_op->result_count != 0) {
        IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
            context, source_op->result_count, sizeof(*result_types),
            (void**)&result_types));
        for (uint16_t i = 0; i < source_op->result_count; ++i) {
          IREE_RETURN_IF_ERROR(loom_low_lower_check_mapped_value(
              context, source_op, source_results[i], &result_types[i]));
          has_unmapped_result |=
              loom_type_kind(result_types[i]) == LOOM_TYPE_NONE;
        }
      }
      if (has_unmapped_result) {
        return iree_ok_status();
      }

      loom_low_func_call_build_flags_t build_flags = 0;
      uint8_t purity = loom_func_call_purity(source_op);
      if (purity != 0) {
        build_flags |= LOOM_LOW_FUNC_CALL_BUILD_FLAG_HAS_PURITY;
      }
      loom_op_t* low_call_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_func_call_build(
          &context->builder, build_flags, purity,
          loom_func_call_callee(source_op), low_operands, operands.count,
          result_types, source_op->result_count,
          /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
          &low_call_op));
      return loom_low_lower_bind_call_results(context, source_op, low_call_op);
    }
    case LOOM_OP_FUNC_CALL_INDIRECT: {
      loom_value_id_t* low_operands = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
          context, source_op, loom_op_const_operands(source_op),
          source_op->operand_count, &low_operands));

      const loom_value_id_t* source_results = loom_op_const_results(source_op);
      loom_type_t* result_types = NULL;
      bool has_unmapped_result = false;
      if (source_op->result_count != 0) {
        IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
            context, source_op->result_count, sizeof(*result_types),
            (void**)&result_types));
        for (uint16_t i = 0; i < source_op->result_count; ++i) {
          IREE_RETURN_IF_ERROR(loom_low_lower_check_mapped_value(
              context, source_op, source_results[i], &result_types[i]));
          has_unmapped_result |=
              loom_type_kind(result_types[i]) == LOOM_TYPE_NONE;
        }
      }
      if (has_unmapped_result) {
        return iree_ok_status();
      }

      loom_op_t* low_call_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_func_call_indirect_build(
          &context->builder, low_operands[0], low_operands + 1,
          source_op->operand_count - 1, result_types, source_op->result_count,
          /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
          &low_call_op));
      return loom_low_lower_bind_call_results(context, source_op, low_call_op);
    }
    case LOOM_OP_KERNEL_RETURN: {
      loom_op_t* low_return_op = NULL;
      return loom_low_return_build(&context->builder, NULL, 0,
                                   source_op->location, &low_return_op);
    }
    case LOOM_OP_SCF_SCHEDULE_FENCE: {
      loom_op_t* low_fence_op = NULL;
      return loom_low_schedule_fence_build(&context->builder,
                                           source_op->location, &low_fence_op);
    }
    case LOOM_OP_CFG_BR: {
      loom_block_t* low_dest = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_successor_dest(
          context, source_op, 0, &low_dest));
      loom_value_slice_t args = loom_cfg_br_args(source_op);
      loom_value_id_t* low_args = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_remap_successor_args(
          context, source_op, 0, low_dest, args.values, args.count, &low_args));
      loom_op_t* low_br_op = NULL;
      return loom_low_br_build(&context->builder, low_dest, low_args,
                               args.count, source_op->location, &low_br_op);
    }
    case LOOM_OP_CFG_COND_BR: {
      loom_block_t* low_true_dest = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_successor_dest(
          context, source_op, 0, &low_true_dest));
      loom_block_t* low_false_dest = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_successor_dest(
          context, source_op, 1, &low_false_dest));
      bool condition = false;
      if (loom_low_lower_cfg_cond_br_exact_bool(context, source_op,
                                                &condition)) {
        loom_block_t* low_dest = condition ? low_true_dest : low_false_dest;
        loom_op_t* low_br_op = NULL;
        return loom_low_br_build(&context->builder, low_dest, NULL, 0,
                                 source_op->location, &low_br_op);
      }
      loom_value_id_t low_condition = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, loom_cfg_cond_br_condition(source_op), &low_condition));
      if (context->policy->emit_cond_branch.fn != NULL) {
        return context->policy->emit_cond_branch.fn(
            context->policy->emit_cond_branch.user_data, context, source_op,
            low_condition, low_true_dest, low_false_dest);
      }
      loom_op_t* low_cond_br_op = NULL;
      return loom_low_cond_br_build(&context->builder, low_condition,
                                    low_true_dest, low_false_dest,
                                    source_op->location, &low_cond_br_op);
    }
    case LOOM_OP_CFG_SWITCH: {
      if (context->policy->switch_lowering.emit == NULL) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "cfg.switch reached a target lowering policy without direct "
            "switch emission");
      }
      loom_value_id_t low_selector = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, loom_cfg_switch_selector(source_op), &low_selector));
      loom_block_t* low_default_dest = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_successor_dest(
          context, source_op, 0, &low_default_dest));
      const loom_successor_slice_t case_dests =
          loom_cfg_switch_case_dests(source_op);
      loom_block_t** low_case_dests = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
          context, case_dests.count, sizeof(*low_case_dests),
          (void**)&low_case_dests));
      for (uint16_t i = 0; i < case_dests.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_low_lower_lookup_successor_dest(
            context, source_op, (uint16_t)(i + 1), &low_case_dests[i]));
      }
      return context->policy->switch_lowering.emit(
          context->policy->switch_lowering.user_data, context, source_op,
          low_selector, low_default_dest, low_case_dests, case_dests.count);
    }
    case LOOM_OP_SCF_YIELD:
      return loom_low_lower_emit_scf_yield(context, source_op);
    case LOOM_OP_SCF_CONDITION:
      return loom_low_lower_emit_scf_condition(context, source_op);
    case LOOM_OP_SCF_IF:
      return loom_low_lower_emit_scf_if(context, source_op);
    case LOOM_OP_SCF_FOR:
      return loom_low_lower_emit_scf_for(context, source_op);
    case LOOM_OP_SCF_WHILE:
      return loom_low_lower_emit_scf_while(context, source_op);
    default:
      *out_handled = false;
      return iree_ok_status();
  }
}

static iree_status_t loom_low_lower_emit_source_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (context->lowering.selected_plan_emit_index <
          context->lowering.selected_plan_count &&
      context->lowering
              .selected_plans[context->lowering.selected_plan_emit_index]
              .source_op == source_op) {
    return loom_low_lower_emit_selected_plan(context, source_op);
  }
  bool handled = false;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_structural_op(context, source_op, &handled));
  if (handled || loom_low_lower_op_is_source_metadata(source_op->kind)) {
    return iree_ok_status();
  }
  IREE_ASSERT_UNREACHABLE(
      "source op has neither a structural lowering nor a selected plan");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_low_lower_emit_region_ops(
    loom_low_lower_context_t* context, loom_region_t* source_region,
    bool map_source_blocks) {
  // A rejected plan leaves its low results unbound, so emission cannot resume
  // elsewhere in the region after any diagnostic error.
  iree_status_t status = iree_ok_status();
  for (uint16_t block_index = 0;
       block_index < source_region->block_count && iree_status_is_ok(status);
       ++block_index) {
    loom_block_t* source_block = loom_region_block(source_region, block_index);
    if (map_source_blocks) {
      loom_builder_set_block(&context->builder,
                             context->lowering.block_map[block_index]);
    } else if (block_index != 0) {
      IREE_ASSERT_UNREACHABLE(
          "structured source region with multiple blocks reached target-low "
          "emission");
      IREE_BUILTIN_UNREACHABLE();
    }
    loom_op_t* source_op = NULL;
    loom_block_for_each_op(source_block, source_op) {
      const uint32_t before_error_count = context->result->error_count;
      loom_low_lower_context_emission_scope_begin(context);
      status = loom_low_lower_emit_source_op(context, source_op);
      // Builders copy all caller-provided arrays and attribute payloads into
      // module-owned storage. Nested structured emission may reset this arena
      // while its parent is active because the parent builder has already
      // consumed every temporary before entering a child region.
      loom_low_lower_context_emission_scope_end(context);
      if (!iree_status_is_ok(status)) {
        break;
      }
      if (context->result->error_count != before_error_count) {
        return iree_ok_status();
      }
    }
  }
  return status;
}

iree_status_t loom_low_lower_emit_body(loom_low_lower_context_t* context,
                                       loom_region_t* source_body) {
  loom_region_t* low_body = loom_low_lower_context_low_body(context);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  iree_status_t status = loom_low_lower_emit_region_ops(
      context, source_body, /*map_source_blocks=*/true);
  loom_builder_restore(&context->builder, saved_ip);
  if (iree_status_is_ok(status) && context->result->error_count == 0) {
    IREE_ASSERT_EQ(context->lowering.selected_plan_emit_index,
                   context->lowering.selected_plan_count);
  }
  return status;
}
