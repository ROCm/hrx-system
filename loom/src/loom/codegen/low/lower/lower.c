// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/analysis/contract_vector.h"
#include "loom/analysis/kernel_async_legality.h"
#include "loom/analysis/kernel_barrier_lifetime.h"
#include "loom/analysis/vector_memory_footprint.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/contract_query.h"
#include "loom/codegen/low/lower/function_boundary.h"
#include "loom/codegen/low/lower/realization.h"
#include "loom/codegen/low/lower/report.h"
#include "loom/codegen/low/lower/rule_emit.h"
#include "loom/codegen/low/lower/rule_source_memory.h"
#include "loom/codegen/low/lower/source_call.h"
#include "loom/codegen/low/lower/source_plan.h"
#include "loom/codegen/low/lower/source_query.h"
#include "loom/codegen/low/lower/source_value_storage.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/remap.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/registers.h"

static const loom_target_bundle_t* loom_low_lower_options_bundle(
    const loom_low_lower_options_t* options) {
  return loom_target_facts_bundle(options->target_facts);
}

static void loom_low_lower_assert_options(
    const loom_module_t* module, loom_func_like_t source_function,
    const loom_low_lower_options_t* options) {
  IREE_ASSERT(module != NULL);
  IREE_ASSERT(loom_func_like_isa(source_function));
  IREE_ASSERT(options != NULL);
  IREE_ASSERT(loom_func_like_body(source_function) != NULL);
  if (loom_symbol_ref_is_valid(options->target_ref)) {
    IREE_ASSERT_EQ(options->target_ref.module_id, 0);
    IREE_ASSERT_LT(options->target_ref.symbol_id, module->symbols.count);
  }
  IREE_ASSERT(options->target_facts != NULL);
  const loom_target_bundle_t* bundle = loom_low_lower_options_bundle(options);
  IREE_ASSERT(bundle != NULL);
  IREE_ASSERT(bundle->snapshot != NULL);
  IREE_ASSERT(bundle->export_plan != NULL);
  IREE_ASSERT(bundle->config != NULL);
  IREE_ASSERT(options->fact_table != NULL);
  IREE_ASSERT(options->fact_table->context.target_facts ==
              options->target_facts);
  IREE_ASSERT(options->descriptor_registry != NULL);
  IREE_ASSERT(options->policy != NULL);
}

static iree_status_t loom_low_lowering_frame_initialize_value_ordinals(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  // Target-legalization query scopes can inspect source ops before CFG
  // conversion, so nested source-region values must be ordinal-addressable even
  // when the final source-to-low boundary expects CFG.
  return loom_local_value_domain_acquire_for_region_tree(
      context->module, source_body, &context->analysis_arena,
      &context->lowering->value_domain);
}

static loom_region_t* loom_low_lower_low_body(
    const loom_low_lower_context_t* context) {
  if (loom_low_func_def_isa(context->low_func_op)) {
    return loom_low_func_def_body(context->low_func_op);
  }
  if (loom_low_kernel_def_isa(context->low_func_op)) {
    return loom_low_kernel_def_body(context->low_func_op);
  }
  return NULL;
}

static iree_status_t loom_low_lower_map_blocks(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  loom_region_t* low_body = loom_low_lower_low_body(context);
  const iree_host_size_t block_count =
      loom_low_lower_control_block_count(context);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->emission_arena, block_count, sizeof(*context->block_map),
      (void**)&context->block_map));
  memset(context->block_map, 0, block_count * sizeof(*context->block_map));

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
    context->block_map[i] = low_block;
  }

  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    loom_block_t* source_block = loom_region_block(source_body, block_index);
    loom_block_t* low_block = context->block_map[block_index];
    if (block_index == 0) {
      IREE_RETURN_IF_ERROR(
          loom_low_lower_function_boundary_bind_entry_arguments(
              context, source_block, low_block));
      continue;
    }

    for (uint16_t arg_index = 0; arg_index < source_block->arg_count;
         ++arg_index) {
      const loom_type_t low_type =
          loom_low_lower_structural_block_argument_type(context, block_index,
                                                        arg_index);
      loom_value_id_t low_arg = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
          &context->builder, low_block, low_type, &low_arg));
      IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
          context, source_block->arg_ids[arg_index], low_arg));
    }
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_realizations_map_blocks(context));
  return loom_low_lower_control_create_blocks(context);
}

static iree_status_t loom_low_lower_emit_preamble(
    loom_low_lower_context_t* context) {
  if (context->policy->emit_preamble.fn == NULL) {
    return iree_ok_status();
  }

  loom_region_t* low_body = loom_low_lower_low_body(context);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  loom_builder_set_block(&context->builder, loom_region_entry_block(low_body));
  iree_status_t status = context->policy->emit_preamble.fn(
      context->policy->emit_preamble.user_data, context);
  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

static iree_status_t loom_low_lower_emit_entry_setup(
    loom_low_lower_context_t* context) {
  if (context->policy->entry_setup.emit == NULL) {
    return iree_ok_status();
  }

  loom_region_t* low_body = loom_low_lower_low_body(context);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  loom_builder_set_block(&context->builder, loom_region_entry_block(low_body));
  iree_status_t status = context->policy->entry_setup.emit(
      context->policy->entry_setup.user_data, context);
  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

iree_status_t loom_low_lower_remap_values(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          const loom_value_id_t* source_values,
                                          iree_host_size_t value_count,
                                          const loom_type_t* required_types,
                                          loom_value_id_t** out_low_values) {
  *out_low_values = NULL;
  if (value_count == 0) {
    return iree_ok_status();
  }
  const bool is_callable_exit =
      loom_low_lower_source_op_is_callable_exit(context, source_op);
  loom_value_id_t* low_values = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, value_count, sizeof(*low_values), (void**)&low_values));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    low_values[i] = loom_low_lower_lookup_value(context, source_values[i]);
    const loom_type_t required_low_type =
        required_types ? required_types[i]
                       : loom_module_value_type(context->module, low_values[i]);
    if (is_callable_exit) {
      IREE_RETURN_IF_ERROR(loom_low_lower_control_materialize_operand(
          context, source_op, i, required_low_type, &low_values[i]));
    }
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
  if (!loom_low_lower_source_value_storage_result_required(context,
                                                           result_value_id)) {
    loom_low_lower_elide_value(context, result_value_id);
    return iree_ok_status();
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

static iree_status_t loom_low_lower_emit_region_ops(
    loom_low_lower_context_t* context, loom_region_t* source_region,
    bool map_source_blocks);

// Rebinds tuple-local SSA references retained by a target register type from
// source identities to the Low identities that will own the type scheme.
static iree_status_t loom_low_lower_remap_type_scheme(
    loom_low_lower_context_t* context, const loom_value_id_t* source_values,
    const loom_value_id_t* low_values, uint16_t value_count,
    loom_type_t* low_types) {
  if (value_count == 0) {
    return iree_ok_status();
  }
  const loom_ir_remap_options_t options = {.allow_unmapped_values = true};
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_initialize(context->module, context->module,
                               &context->emission_arena, &options, &remap));
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_map_values(&remap, source_values, low_values, value_count));
  for (uint16_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_type(&remap, low_types[i], &low_types[i]));
  }
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

static iree_status_t loom_low_lower_emit_scf_yield(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_slice_t values = loom_scf_yield_values(source_op);
  loom_value_id_t* low_values = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
      context, source_op, values.values, values.count, /*required_types=*/NULL,
      &low_values));
  loom_op_t* low_yield_op = NULL;
  return loom_low_scf_yield_build(&context->builder, low_values, values.count,
                                  source_op->location, &low_yield_op);
}

static iree_status_t loom_low_lower_emit_scf_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_condition = loom_low_lower_lookup_value(
      context, loom_scf_condition_condition(source_op));
  const loom_value_slice_t forwarded = loom_scf_condition_forwarded(source_op);
  loom_value_id_t* low_forwarded = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
      context, source_op, forwarded.values, forwarded.count,
      /*required_types=*/NULL, &low_forwarded));
  loom_op_t* low_condition_op = NULL;
  return loom_low_scf_condition_build(&context->builder, low_condition,
                                      low_forwarded, forwarded.count,
                                      source_op->location, &low_condition_op);
}

static iree_status_t loom_low_lower_emit_scf_if(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_condition =
      loom_low_lower_lookup_value(context, loom_scf_if_condition(source_op));

  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_structural_take_types(
      context, source_op, &result_types, /*out_header_types=*/NULL));

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

static iree_status_t loom_low_lower_emit_scf_for(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_lower_bound =
      loom_low_lower_lookup_value(context, loom_scf_for_lower_bound(source_op));
  loom_value_id_t low_upper_bound =
      loom_low_lower_lookup_value(context, loom_scf_for_upper_bound(source_op));
  loom_value_id_t low_step =
      loom_low_lower_lookup_value(context, loom_scf_for_step(source_op));

  loom_value_slice_t iter_args = loom_scf_for_iter_args(source_op);
  loom_value_id_t* low_iter_args = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
      context, source_op, iter_args.values, iter_args.count,
      /*required_types=*/NULL, &low_iter_args));

  loom_value_id_t low_unroll_factor = LOOM_VALUE_ID_INVALID;
  loom_low_scf_for_build_flags_t build_flags = 0;
  if (loom_scf_for_unroll_factor_is_present(source_op)) {
    build_flags |= LOOM_LOW_SCF_FOR_BUILD_FLAG_HAS_UNROLL_FACTOR;
    low_unroll_factor = loom_low_lower_lookup_value(
        context, loom_scf_for_unroll_factor(source_op));
  }

  uint8_t unroll_policy = 0;
  if (loom_scf_for_has_unroll_policy(source_op)) {
    build_flags |= LOOM_LOW_SCF_FOR_BUILD_FLAG_HAS_UNROLL_POLICY;
    unroll_policy = (uint8_t)loom_scf_for_unroll_policy(source_op);
  }

  loom_op_t* low_for_op = NULL;
  const loom_scalar_type_t counter_type =
      loom_type_element_type(loom_module_value_type(
          context->module, loom_scf_for_lower_bound(source_op)));
  const loom_low_scf_for_signedness_t signedness =
      counter_type == LOOM_SCALAR_TYPE_OFFSET
          ? LOOM_LOW_SCF_FOR_SIGNEDNESS_UNSIGNED
          : LOOM_LOW_SCF_FOR_SIGNEDNESS_SIGNED;
  IREE_RETURN_IF_ERROR(loom_low_scf_for_build(
      &context->builder, build_flags, signedness, low_lower_bound,
      low_upper_bound, low_step, low_iter_args, iter_args.count,
      /*result_types=*/NULL,
      /*tied_results=*/NULL,
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
  const loom_block_t* source_before =
      loom_region_const_entry_block(loom_scf_while_before(source_op));

  loom_type_t* header_types = NULL;
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_structural_take_types(
      context, source_op, &result_types, &header_types));

  const iree_host_size_t identity_count =
      (iree_host_size_t)source_before->arg_count + source_op->result_count;
  loom_value_id_t* identities = NULL;
  if (identity_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, identity_count, sizeof(*identities), (void**)&identities));
    IREE_RETURN_IF_ERROR(loom_builder_reserve_values(
        &context->builder, identity_count, identities));
  }
  loom_value_id_t* result_identities =
      identities ? identities + source_before->arg_count : NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_remap_type_scheme(
      context, source_before->arg_ids, identities, source_before->arg_count,
      header_types));
  IREE_RETURN_IF_ERROR(loom_low_lower_remap_type_scheme(
      context, loom_op_const_results(source_op), result_identities,
      source_op->result_count, result_types));

  loom_value_id_t* low_iter_args = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
      context, source_op, iter_args.values, iter_args.count, header_types,
      &low_iter_args));

  loom_op_t* low_while_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_scf_while_build(
      &context->builder, low_iter_args, iter_args.count, header_types,
      result_types, source_op->result_count,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_while_op));
  const loom_block_t* low_before =
      loom_region_const_entry_block(loom_low_scf_while_before(low_while_op));
  for (uint16_t i = 0; i < source_before->arg_count; ++i) {
    IREE_ASSERT_EQ(low_before->arg_ids[i], identities[i]);
  }
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    IREE_ASSERT_EQ(loom_op_const_results(low_while_op)[i],
                   result_identities[i]);
  }
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

// Keeps the uncommon structural dispatcher out of ordinary selected-plan
// emission.
IREE_ATTRIBUTE_NOINLINE static iree_status_t loom_low_lower_structural_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_handled) {
  *out_handled = true;
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
  if (loom_low_lower_source_op_is_callable_exit(context, source_op)) {
    const loom_value_id_t* values = loom_op_const_operands(source_op);
    loom_value_id_t* low_values = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
        context, source_op, values, source_op->operand_count,
        context->lowering->boundary.result_types, &low_values));
    loom_op_t* low_return_op = NULL;
    return loom_low_return_build(&context->builder, low_values,
                                 source_op->operand_count, source_op->location,
                                 &low_return_op);
  }
  if (loom_low_lower_source_call_is_structural(context->module, source_op)) {
    return loom_low_lower_source_call(context, source_op);
  }
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_ASSUME_SAME_ROOT: {
      return loom_low_lower_bind_or_elide_alias(
          context, loom_buffer_assume_same_root_buffer(source_op),
          loom_buffer_assume_same_root_result(source_op));
    }
    case LOOM_OP_SCF_SCHEDULE_FENCE: {
      loom_op_t* low_fence_op = NULL;
      return loom_low_schedule_fence_build(&context->builder,
                                           source_op->location, &low_fence_op);
    }
    case LOOM_OP_CFG_BR: {
      loom_block_t* low_dest =
          loom_low_lower_lookup_successor_dest(context, source_op, 0);
      loom_value_slice_t args = loom_cfg_br_args(source_op);
      loom_value_slice_t low_args = {0};
      IREE_RETURN_IF_ERROR(
          loom_low_lower_realizations_emit_edge(context, source_op));
      IREE_RETURN_IF_ERROR(loom_low_lower_remap_successor_args(
          context, source_op, low_dest, args.values, args.count, &low_args));
      loom_op_t* low_br_op = NULL;
      return loom_low_br_build(&context->builder, low_dest, low_args.values,
                               low_args.count, source_op->location, &low_br_op);
    }
    case LOOM_OP_CFG_COND_BR: {
      loom_block_t* low_true_dest =
          loom_low_lower_lookup_successor_dest(context, source_op, 0);
      loom_block_t* low_false_dest =
          loom_low_lower_lookup_successor_dest(context, source_op, 1);
      bool condition = false;
      if (loom_low_lower_structural_branch_exact_bool(context, source_op,
                                                      &condition)) {
        loom_block_t* low_dest = condition ? low_true_dest : low_false_dest;
        loom_op_t* low_br_op = NULL;
        return loom_low_br_build(&context->builder, low_dest, NULL, 0,
                                 source_op->location, &low_br_op);
      }
      loom_value_id_t low_condition = loom_low_lower_lookup_value(
          context, loom_cfg_cond_br_condition(source_op));
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

static void loom_low_lower_emit_elided_selected_plan(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* selected_plan) {
  const loom_op_t* source_op = selected_plan->source_op;
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    loom_low_lower_elide_value(context, source_results[i]);
  }
  if (!iree_allocator_is_null(context->options->report_allocator)) {
    loom_low_lower_report_record_emission(context, /*emitted_low_op_count=*/0);
  }
}

static loom_value_id_t loom_low_lower_descriptor_matrix_sparse_source_value(
    const loom_contract_request_t* request) {
  loom_value_id_t source_value = LOOM_VALUE_ID_INVALID;
  const loom_contract_operand_t* operands[] = {
      &request->lhs,
      &request->rhs,
      &request->accumulator,
      &request->result,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(operands); ++i) {
    const loom_contract_value_ref_t ref =
        operands[i]->encoded.auxiliary_value_refs
            [LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SPARSE_METADATA];
    if (!loom_contract_value_ref_is_present(ref)) {
      continue;
    }
    const loom_value_id_t operand_source_value =
        loom_contract_value_ref_value_id(ref);
    if (source_value == LOOM_VALUE_ID_INVALID) {
      source_value = operand_source_value;
      continue;
    }
    IREE_ASSERT_EQ(source_value, operand_source_value,
                   "descriptor-matrix selected sparse source is ambiguous");
  }
  IREE_ASSERT_NE(source_value, LOOM_VALUE_ID_INVALID,
                 "descriptor-matrix selected sparse source is unavailable");
  return source_value;
}

static loom_value_id_t loom_low_lower_descriptor_matrix_auxiliary_source_value(
    const loom_contract_operand_t* operand,
    loom_contract_auxiliary_operand_key_t key) {
  const loom_contract_value_ref_t ref =
      operand->encoded.auxiliary_value_refs[key];
  IREE_ASSERT(loom_contract_value_ref_is_present(ref),
              "descriptor-matrix selected auxiliary operand is unavailable");
  return loom_contract_value_ref_value_id(ref);
}

static iree_status_t loom_low_lower_descriptor_matrix_packet_value(
    loom_low_lower_context_t* context,
    const loom_low_lower_descriptor_matrix_plan_t* plan,
    loom_low_operand_source_binding_t source_binding, loom_value_id_t low_lhs,
    loom_value_id_t low_rhs, loom_value_id_t low_init,
    loom_value_id_t* out_low_value) {
  switch (source_binding) {
    case LOOM_LOW_OPERAND_SOURCE_BINDING_LHS:
      *out_low_value = low_lhs;
      return iree_ok_status();
    case LOOM_LOW_OPERAND_SOURCE_BINDING_RHS:
      *out_low_value = low_rhs;
      return iree_ok_status();
    case LOOM_LOW_OPERAND_SOURCE_BINDING_ACCUMULATOR:
      *out_low_value = low_init;
      return iree_ok_status();
    case LOOM_LOW_OPERAND_SOURCE_BINDING_SPARSE_METADATA: {
      const loom_value_id_t source_value =
          loom_low_lower_descriptor_matrix_sparse_source_value(
              &plan->contract_request);
      *out_low_value = loom_low_lower_lookup_value(context, source_value);
      return iree_ok_status();
    }
    case LOOM_LOW_OPERAND_SOURCE_BINDING_LHS_SCALE: {
      const loom_value_id_t source_value =
          loom_low_lower_descriptor_matrix_auxiliary_source_value(
              &plan->contract_request.lhs,
              LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SCALE);
      *out_low_value = loom_low_lower_lookup_value(context, source_value);
      return iree_ok_status();
    }
    case LOOM_LOW_OPERAND_SOURCE_BINDING_RHS_SCALE: {
      const loom_value_id_t source_value =
          loom_low_lower_descriptor_matrix_auxiliary_source_value(
              &plan->contract_request.rhs,
              LOOM_CONTRACT_AUXILIARY_OPERAND_KEY_SCALE);
      *out_low_value = loom_low_lower_lookup_value(context, source_value);
      return iree_ok_status();
    }
    case LOOM_LOW_OPERAND_SOURCE_BINDING_NONE:
    default:
      IREE_ASSERT_UNREACHABLE(
          "descriptor-matrix selected packet operand has no source binding");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_low_lower_descriptor_matrix_packet_operands(
    loom_low_lower_context_t* context,
    const loom_low_lower_descriptor_matrix_plan_t* plan,
    loom_value_id_t low_lhs, loom_value_id_t low_rhs, loom_value_id_t low_init,
    loom_value_id_t** out_operands, iree_host_size_t* out_operand_count) {
  *out_operands = NULL;
  *out_operand_count = 0;
  const loom_low_descriptor_set_t* descriptor_set = context->descriptor_set;
  const loom_low_descriptor_t* descriptor = plan->descriptor.descriptor;

  iree_host_size_t operand_count = 0;
  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    const loom_low_operand_t* operand =
        loom_low_descriptor_operand(descriptor_set, descriptor, i);
    if (loom_low_operand_role_is_packet_operand(operand->role)) {
      ++operand_count;
    }
  }

  loom_value_id_t* operands = NULL;
  if (operand_count > 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, operand_count, sizeof(*operands), (void**)&operands));
  }

  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    const loom_low_operand_t* operand =
        loom_low_descriptor_operand(descriptor_set, descriptor, i);
    if (!loom_low_operand_role_is_packet_operand(operand->role)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_lower_descriptor_matrix_packet_value(
        context, plan, operand->source_binding, low_lhs, low_rhs, low_init,
        &operands[operand->source_value_index]));
  }

  *out_operands = operands;
  *out_operand_count = operand_count;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_descriptor_matrix_tied_results(
    loom_low_lower_context_t* context,
    const loom_low_lower_descriptor_matrix_plan_t* plan,
    const loom_tied_result_t** out_tied_results,
    iree_host_size_t* out_tied_result_count) {
  *out_tied_results = NULL;
  *out_tied_result_count = 0;
  const loom_low_descriptor_set_t* descriptor_set = context->descriptor_set;
  const loom_low_descriptor_t* descriptor = plan->descriptor.descriptor;
  iree_host_size_t tied_result_count = 0;
  IREE_ASSERT((uint64_t)descriptor->constraint_start +
                  (uint64_t)descriptor->constraint_count <=
              descriptor_set->constraint_count);
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const uint32_t row = descriptor->constraint_start + i;
    if (descriptor_set->constraints[row].kind ==
        LOOM_LOW_CONSTRAINT_KIND_TIED) {
      ++tied_result_count;
    }
  }
  if (tied_result_count == 0) {
    return iree_ok_status();
  }

  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, tied_result_count, sizeof(*tied_results),
      (void**)&tied_results));
  iree_host_size_t tied_result_index = 0;
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_TIED) {
      continue;
    }
    if (constraint->lhs_operand_index >= descriptor->result_count ||
        constraint->rhs_operand_index == LOOM_LOW_ID_NONE ||
        constraint->rhs_operand_index >= descriptor->operand_count) {
      IREE_ASSERT_UNREACHABLE(
          "descriptor-matrix selected tied result constraint is invalid");
      IREE_BUILTIN_UNREACHABLE();
    }
    const loom_low_operand_t* tied_operand = loom_low_descriptor_operand(
        descriptor_set, descriptor, constraint->rhs_operand_index);
    IREE_ASSERT_NE(tied_operand->source_value_index, LOOM_LOW_ID_NONE);
    tied_results[tied_result_index++] = (loom_tied_result_t){
        .result_index = constraint->lhs_operand_index,
        .operand_index = tied_operand->source_value_index,
        .has_type_change = false,
    };
  }

  *out_tied_results = tied_results;
  *out_tied_result_count = tied_result_count;
  return iree_ok_status();
}

static bool loom_low_lower_descriptor_matrix_destructive_operand_was_copied(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t constraint_index,
    uint16_t packet_operand_index) {
  for (uint16_t i = 0; i < constraint_index; ++i) {
    const loom_low_constraint_t* previous =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (previous->kind != LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE) {
      continue;
    }
    IREE_ASSERT(previous->rhs_operand_index != LOOM_LOW_ID_NONE);
    IREE_ASSERT(previous->rhs_operand_index < descriptor->operand_count);
    const loom_low_operand_t* previous_operand = loom_low_descriptor_operand(
        descriptor_set, descriptor, previous->rhs_operand_index);
    if (previous_operand->source_value_index == packet_operand_index) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_lower_descriptor_matrix_copy_destructive_operands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_descriptor_matrix_plan_t* plan,
    loom_value_id_t* operands) {
  const loom_low_descriptor_set_t* descriptor_set = context->descriptor_set;
  const loom_low_descriptor_t* descriptor = plan->descriptor.descriptor;
  IREE_ASSERT((uint64_t)descriptor->constraint_start +
                  (uint64_t)descriptor->constraint_count <=
              descriptor_set->constraint_count);
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE) {
      continue;
    }
    IREE_ASSERT(constraint->lhs_operand_index < descriptor->result_count);
    IREE_ASSERT(constraint->rhs_operand_index != LOOM_LOW_ID_NONE);
    IREE_ASSERT(constraint->rhs_operand_index < descriptor->operand_count);
    const loom_low_operand_t* destructive_operand = loom_low_descriptor_operand(
        descriptor_set, descriptor, constraint->rhs_operand_index);
    const uint16_t packet_operand_index =
        destructive_operand->source_value_index;
    IREE_ASSERT_NE(packet_operand_index, LOOM_LOW_ID_NONE);
    if (loom_low_lower_descriptor_matrix_destructive_operand_was_copied(
            descriptor_set, descriptor, i, packet_operand_index)) {
      continue;
    }
    const loom_value_id_t source_value = operands[packet_operand_index];
    const loom_type_t copy_type =
        loom_module_value_type(context->module, source_value);
    IREE_ASSERT(loom_low_type_is_register(copy_type));
    loom_op_t* copy_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_copy_build(
        loom_low_lower_context_builder(context), source_value, false, copy_type,
        source_op->location, &copy_op));
    operands[packet_operand_index] = loom_low_copy_result(copy_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_emit_descriptor_matrix_vector_mma(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_descriptor_matrix_plan_t* plan) {
  const loom_low_descriptor_t* descriptor = plan->descriptor.descriptor;
  if (descriptor->result_count != 1) {
    IREE_ASSERT_UNREACHABLE(
        "descriptor-matrix vector.mma descriptor result count is invalid");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_value_id_t low_lhs =
      loom_low_lower_lookup_value(context, loom_vector_mma_lhs(source_op));
  loom_value_id_t low_rhs =
      loom_low_lower_lookup_value(context, loom_vector_mma_rhs(source_op));
  if (iree_any_bit_set(
          plan->transform_flags,
          LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_TRANSFORM_TRANSPOSE_MN)) {
    const loom_value_id_t exchanged_lhs = low_lhs;
    low_lhs = low_rhs;
    low_rhs = exchanged_lhs;
  }
  loom_value_id_t low_init =
      loom_low_lower_lookup_value(context, loom_vector_mma_init(source_op));

  const loom_value_id_t result = loom_vector_mma_result(source_op);
  const loom_type_t result_low_type =
      loom_low_lower_value_binding_type(context, result);

  loom_value_id_t* operands = NULL;
  iree_host_size_t operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_lower_descriptor_matrix_packet_operands(
      context, plan, low_lhs, low_rhs, low_init, &operands, &operand_count));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_descriptor_matrix_copy_destructive_operands(
          context, source_op, plan, operands));
  const loom_tied_result_t* tied_results = NULL;
  iree_host_size_t tied_result_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_lower_descriptor_matrix_tied_results(
      context, plan, &tied_results, &tied_result_count));

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->descriptor, operands, operand_count, plan->attrs,
      &result_low_type, 1, tied_results, tied_result_count, source_op->location,
      &low_op));
  return loom_low_lower_bind_value(
      context, result, loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_low_lower_emit_descriptor_matrix_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_descriptor_matrix_plan_t* plan) {
  switch (plan->source) {
    case LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_VECTOR_MMA:
      return loom_low_lower_emit_descriptor_matrix_vector_mma(context,
                                                              source_op, plan);
    case LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_NONE:
    default:
      IREE_ASSERT_UNREACHABLE("unknown descriptor-matrix source");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static const loom_op_t* loom_low_lower_selected_plan_emission_op(
    const loom_low_lower_selected_plan_t* selected_plan) {
  if (selected_plan->emission_source_node_index == 0) {
    return selected_plan->source_op;
  }
  IREE_ASSERT(selected_plan->data.source_nodes != NULL);
  IREE_ASSERT_LT(selected_plan->emission_source_node_index,
                 selected_plan->source_node_count);
  return selected_plan->data
      .source_nodes[selected_plan->emission_source_node_index];
}

static iree_status_t loom_low_lower_emit_selected_plan(
    loom_low_lower_context_t* context, const loom_op_t* emission_op) {
  IREE_ASSERT_LT(context->lowering->source_plan.selected_plan_emit_index,
                 context->lowering->source_plan.selected_plan_count);
  const loom_low_lower_selected_plan_t selected_plan =
      context->lowering->source_plan.selected_plans
          [context->lowering->source_plan.selected_plan_emit_index++];
  IREE_ASSERT_EQ(loom_low_lower_selected_plan_emission_op(&selected_plan),
                 emission_op);
  if (iree_any_bit_set(selected_plan.flags,
                       LOOM_LOW_LOWER_SELECTED_PLAN_CLAIMED)) {
    return iree_ok_status();
  }
  if (iree_any_bit_set(selected_plan.flags,
                       LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED)) {
    loom_low_lower_emit_elided_selected_plan(context, &selected_plan);
    return iree_ok_status();
  }
  const bool report_allocator_provided =
      !iree_allocator_is_null(context->options->report_allocator);
  const uint64_t before_op_count = context->report.emitted_op_count;
  if (selected_plan.kind == LOOM_LOW_LOWER_SELECTED_PLAN_RULE) {
    IREE_ASSERT(selected_plan.rule_set != NULL);
    IREE_ASSERT(selected_plan.rule != NULL);
    IREE_RETURN_IF_ERROR(loom_low_lower_rule_set_emit_rule(
        context, selected_plan.rule_set, selected_plan.source_op,
        selected_plan.rule, selected_plan.resolved_emits,
        selected_plan.source_memory, selected_plan.data.source_nodes,
        selected_plan.source_node_count));
  } else if (selected_plan.kind ==
             LOOM_LOW_LOWER_SELECTED_PLAN_DESCRIPTOR_MATRIX) {
    IREE_ASSERT_FALSE(
        loom_low_lower_plan_is_empty(selected_plan.data.target_plan));
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_matrix_plan(
        context, selected_plan.source_op,
        (const loom_low_lower_descriptor_matrix_plan_t*)
            selected_plan.data.target_plan.target_data));
  } else if (selected_plan.kind ==
             LOOM_LOW_LOWER_SELECTED_PLAN_FUNCTION_STORAGE) {
    IREE_RETURN_IF_ERROR(loom_low_lower_function_storage_emit(
        context, selected_plan.source_op, selected_plan.data.function_storage));
  } else if (selected_plan.kind == LOOM_LOW_LOWER_SELECTED_PLAN_INVOKE) {
    IREE_RETURN_IF_ERROR(loom_low_lower_source_invoke(
        context, selected_plan.source_op, selected_plan.data.invoke));
  } else {
    IREE_ASSERT_FALSE(
        loom_low_lower_plan_is_empty(selected_plan.data.target_plan));
    IREE_ASSERT(context->policy->emit_op.fn != NULL);
    IREE_RETURN_IF_ERROR(context->policy->emit_op.fn(
        context->policy->emit_op.user_data, context, selected_plan.source_op,
        selected_plan.data.target_plan));
  }
  if (report_allocator_provided) {
    const uint64_t emitted_op_count =
        context->report.emitted_op_count - before_op_count;
    IREE_ASSERT_LE(emitted_op_count, UINT32_MAX);
    loom_low_lower_report_record_emission(context, (uint32_t)emitted_op_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_emit_source_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_low_lower_source_memory_enter_op(context, source_op);
  if (context->lowering->source_plan.selected_plan_emit_index <
          context->lowering->source_plan.selected_plan_count &&
      loom_low_lower_selected_plan_emission_op(
          &context->lowering->source_plan.selected_plans
               [context->lowering->source_plan.selected_plan_emit_index]) ==
          source_op) {
    return loom_low_lower_emit_selected_plan(context, source_op);
  }
  bool handled = false;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_structural_op(context, source_op, &handled));
  if (handled || loom_low_lower_source_plan_op_is_metadata(source_op->kind)) {
    return iree_ok_status();
  }
  IREE_ASSERT_UNREACHABLE(
      "source op has neither a structural lowering nor a selected plan");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_low_lower_emit_region_ops(
    loom_low_lower_context_t* context, loom_region_t* source_region,
    bool map_source_blocks) {
  const uint16_t* block_order =
      map_source_blocks ? context->lowering->source_plan.block_order : NULL;
  iree_status_t status = iree_ok_status();
  for (uint16_t position = 0;
       position < source_region->block_count && iree_status_is_ok(status);
       ++position) {
    const uint16_t block_index = block_order ? block_order[position] : position;
    loom_block_t* source_block = loom_region_block(source_region, block_index);
    if (map_source_blocks) {
      loom_builder_set_block(&context->builder,
                             context->block_map[block_index]);
    } else if (block_index != 0) {
      IREE_ASSERT_UNREACHABLE(
          "structured source region with multiple blocks reached target-low "
          "emission");
      IREE_BUILTIN_UNREACHABLE();
    }
    loom_low_lower_emission_scope_begin(context);
    status = loom_low_lower_realizations_emit_entry(context, source_block);
    loom_low_lower_emission_scope_end(context);
    if (!iree_status_is_ok(status)) {
      break;
    }
    loom_op_t* source_op = NULL;
    loom_block_for_each_op(source_block, source_op) {
      loom_low_lower_emission_scope_begin(context);
      status = loom_low_lower_emit_source_op(context, source_op);
      if (iree_status_is_ok(status)) {
        status = loom_low_lower_realizations_emit_after(context, source_op);
      }
      // Builders copy all caller-provided arrays and attribute payloads into
      // module-owned storage. Nested structured emission may reset this arena
      // while its parent is active because the parent builder has already
      // consumed every temporary before entering a child region.
      loom_low_lower_emission_scope_end(context);
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
  }
  return status;
}

static iree_status_t loom_low_lower_emit_body(loom_low_lower_context_t* context,
                                              loom_region_t* source_body) {
  context->lowering->source_plan.memory.cursor =
      context->lowering->source_plan.memory.first;
  loom_region_t* low_body = loom_low_lower_low_body(context);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  iree_status_t status = loom_low_lower_emit_region_ops(
      context, source_body, /*map_source_blocks=*/true);
  loom_builder_restore(&context->builder, saved_ip);
  if (iree_status_is_ok(status)) {
    IREE_ASSERT_EQ(context->lowering->source_plan.selected_plan_emit_index,
                   context->lowering->source_plan.selected_plan_count);
  }
  return status;
}

static bool loom_low_lower_cluster_size_product_fits_u32(
    loom_target_workgroup_cluster_size_t cluster_size) {
  const uint64_t cluster_size_xy =
      (uint64_t)cluster_size.x * (uint64_t)cluster_size.y;
  return cluster_size_xy <= UINT32_MAX &&
         cluster_size.z <= UINT32_MAX / cluster_size_xy;
}

static bool loom_low_lower_cluster_size_is_trivial(
    loom_target_workgroup_cluster_size_t cluster_size) {
  return cluster_size.x == 1 && cluster_size.y == 1 && cluster_size.z == 1;
}

static iree_status_t loom_low_lower_verify_static_cluster_divisibility(
    loom_low_lower_context_t* context, const loom_op_t* launch_config,
    loom_target_dispatch_workgroup_count_t workgroup_count,
    loom_target_workgroup_cluster_size_t cluster_size) {
  const uint32_t workgroup_counts[] = {
      workgroup_count.x,
      workgroup_count.y,
      workgroup_count.z,
  };
  const uint32_t cluster_sizes[] = {
      cluster_size.x,
      cluster_size.y,
      cluster_size.z,
  };
  const iree_string_view_t axes[] = {
      IREE_SV("x"),
      IREE_SV("y"),
      IREE_SV("z"),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(workgroup_counts); ++i) {
    if (workgroup_counts[i] % cluster_sizes[i] == 0) {
      continue;
    }
    const loom_diagnostic_param_t params[] = {
        loom_param_string(axes[i]),
        loom_param_u32(workgroup_counts[i]),
        loom_param_u32(cluster_sizes[i]),
    };
    return loom_low_lower_emit_target_context_error(context, launch_config,
                                                    LOOM_ERR_TARGET_064, params,
                                                    IREE_ARRAYSIZE(params));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_record_static_launch_config(
    loom_low_lower_context_t* context) {
  const loom_module_t* module = context->module;
  const loom_func_like_t source_function = context->source_function;
  const loom_value_fact_table_t* fact_table = context->options->fact_table;
  loom_low_lower_result_t* result = context->result;
  if (!loom_kernel_def_isa(source_function.op)) {
    return iree_ok_status();
  }
  if (loom_kernel_def_static_workgroup_size_from_facts(
          module, source_function.op, fact_table,
          &result->static_workgroup_size)) {
    result->static_launch_config_flags |=
        LOOM_LOW_LOWER_STATIC_LAUNCH_CONFIG_WORKGROUP_SIZE;
  }
  if (loom_kernel_def_static_workgroup_count_from_facts(
          module, source_function.op, fact_table,
          &result->static_workgroup_count)) {
    result->static_launch_config_flags |=
        LOOM_LOW_LOWER_STATIC_LAUNCH_CONFIG_WORKGROUP_COUNT;
  }

  const loom_op_t* launch_config =
      loom_kernel_def_launch_config_op(source_function.op);
  if (!loom_kernel_launch_config_has_workgroup_cluster_size(launch_config)) {
    return iree_ok_status();
  }
  if (!loom_kernel_def_static_workgroup_cluster_size_from_facts(
          module, source_function.op, fact_table,
          &result->static_workgroup_cluster_size)) {
    return loom_low_lower_emit_target_context_error(
        context, launch_config, LOOM_ERR_TARGET_062, /*extra_params=*/NULL,
        /*extra_param_count=*/0);
  }
  if (!loom_low_lower_cluster_size_product_fits_u32(
          result->static_workgroup_cluster_size)) {
    const loom_diagnostic_param_t params[] = {
        loom_param_u32(result->static_workgroup_cluster_size.x),
        loom_param_u32(result->static_workgroup_cluster_size.y),
        loom_param_u32(result->static_workgroup_cluster_size.z),
    };
    return loom_low_lower_emit_target_context_error(context, launch_config,
                                                    LOOM_ERR_TARGET_063, params,
                                                    IREE_ARRAYSIZE(params));
  }
  if (iree_any_bit_set(result->static_launch_config_flags,
                       LOOM_LOW_LOWER_STATIC_LAUNCH_CONFIG_WORKGROUP_COUNT)) {
    IREE_RETURN_IF_ERROR(loom_low_lower_verify_static_cluster_divisibility(
        context, launch_config, result->static_workgroup_count,
        result->static_workgroup_cluster_size));
    if (result->error_count != 0) {
      return iree_ok_status();
    }
  }
  if (loom_low_lower_cluster_size_is_trivial(
          result->static_workgroup_cluster_size)) {
    result->static_workgroup_cluster_size =
        (loom_target_workgroup_cluster_size_t){0};
    return iree_ok_status();
  }
  result->static_launch_config_flags |=
      LOOM_LOW_LOWER_STATIC_LAUNCH_CONFIG_WORKGROUP_CLUSTER_SIZE;
  return iree_ok_status();
}

// Plans borrow stable storage from the caller owning the lowering transaction.
// The context and every construction analysis remain phase-local stack state.
struct loom_low_lower_function_plan_t {
  // Source module; same-function source IR stays alive until this plan
  // executes.
  loom_module_t* module;
  // Source definition replaced by execution, preserving its symbol identity.
  loom_func_like_t source_function;
  // Settings still consumed after source analysis and selection have retired.
  struct {
    // Module-local target attached to the emitted function.
    loom_symbol_ref_t target_ref;
    // Immutable target facts shared by planning and emission.
    const loom_target_facts_t* target_facts;
    // Target callbacks executing the retained recipes.
    const loom_low_lower_policy_t* policy;
    // Shared module state receiving published target resources.
    loom_low_lower_module_state_t* module_state;
    // Publication observer for helpers created during target emission.
    loom_builder_callback_t on_function_created;
    // Diagnostic sink for target emission remarks.
    iree_diagnostic_emitter_t emitter;
    // Enabled target diagnostic remark categories.
    loom_target_low_legality_diagnostic_flags_t diagnostic_flags;
    // Planned control-flow form consumed by structural builders.
    loom_low_control_flow_lowering_t control_flow_lowering;
    // Assertion-reporting behavior for emitted sanitizer operations.
    loom_sanitizer_reporting_mode_t sanitizer_reporting_mode;
  } emission;
  // Selected instruction descriptors shared by planning and emission.
  const loom_low_descriptor_set_t* descriptor_set;
  // Retained selected decisions, source ordinals, and emission bindings.
  loom_low_lowering_frame_t frame;
  // Caller-owned arena backing this plan and every retained target payload.
  iree_arena_allocator_t* arena;
};

static iree_status_t loom_low_lower_function_plan_build(
    loom_low_lower_context_t* context) {
  loom_module_t* module = context->module;
  const loom_func_like_t source_function = context->source_function;
  const loom_low_lower_options_t* options = context->options;
  loom_low_lower_result_t* out_result = context->result;
  loom_region_t* source_body = loom_func_like_body(source_function);
  iree_status_t status =
      loom_low_lowering_frame_initialize_value_ordinals(context, source_body);
  if (iree_status_is_ok(status)) {
    loom_condition_query_initialize(module, &context->lowering->value_domain,
                                    &context->analysis_arena,
                                    &context->condition_query);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_record_static_launch_config(context);
  }
  if (iree_status_is_ok(status) && out_result->error_count != 0) {
    return iree_ok_status();
  }

  loom_vector_memory_footprint_result_t footprint_result = {0};
  if (iree_status_is_ok(status)) {
    status = loom_cfg_value_identity_table_initialize(
        &context->lowering->value_domain, &context->analysis_arena,
        &context->function_analysis.value_identities);
  }
  if (iree_status_is_ok(status)) {
    const loom_vector_memory_footprint_options_t footprint_options = {
        .fact_table = context->fact_table,
        .value_domain = &context->lowering->value_domain,
        .value_identities = &context->function_analysis.value_identities,
        .emitter = options->emitter,
        .max_errors = options->max_errors,
    };
    status = loom_vector_memory_footprint_verify_function(
        module, source_function, &footprint_options, &footprint_result);
  }
  if (iree_status_is_ok(status)) {
    out_result->error_count += footprint_result.error_count;
  }
  if (iree_status_is_ok(status) && out_result->error_count != 0) {
    return iree_ok_status();
  }

  loom_kernel_async_legality_result_t async_legality_result = {0};
  if (iree_status_is_ok(status)) {
    loom_kernel_async_legality_options_t async_legality_options = {
        .fact_table = context->fact_table,
        .value_domain = &context->lowering->value_domain,
        .emitter = options->emitter,
        .phase_name = IREE_SV("source-low"),
    };
    status = loom_kernel_async_legality_analyze_function(
        module, source_function, &async_legality_options,
        &context->analysis_arena, &async_legality_result);
  }
  if (iree_status_is_ok(status)) {
    out_result->error_count += async_legality_result.error_count;
  }
  if (iree_status_is_ok(status) && out_result->error_count != 0) {
    return iree_ok_status();
  }

  loom_kernel_barrier_lifetime_result_t barrier_lifetime_result = {0};
  if (iree_status_is_ok(status)) {
    const loom_kernel_barrier_lifetime_options_t barrier_lifetime_options = {
        .value_domain = &context->lowering->value_domain,
        .fact_table = context->fact_table,
        .emitter = options->emitter,
        .phase_name = IREE_SV("source-low"),
    };
    status = loom_kernel_barrier_lifetime_verify_function(
        module, source_function, &barrier_lifetime_options,
        &barrier_lifetime_result);
  }
  if (iree_status_is_ok(status)) {
    out_result->error_count += barrier_lifetime_result.error_count;
  }
  if (iree_status_is_ok(status) && out_result->error_count != 0) {
    return iree_ok_status();
  }

  loom_target_contract_query_environment_t legality_environment = {0};
  if (iree_status_is_ok(status)) {
    status = loom_target_low_descriptor_set_select_for_source_lowering(
        options->descriptor_registry,
        loom_target_facts_bundle(options->target_facts),
        &context->descriptor_set);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_source_query_environment_initialize(
        context, context->descriptor_set, &legality_environment);
  }
  loom_target_low_legality_result_t legality_result = {};
  if (iree_status_is_ok(status)) {
    loom_target_low_legality_options_t legality_options = {
        .environment = &legality_environment,
        .error_catalog = options->policy->error_catalog,
        .provider_list = options->legality_provider_list,
        .contract_query = loom_low_lower_source_query_callback(context),
        .type_supported = context->policy->source_type_supported,
        .source_vector_carrier_supported =
            context->policy->source_vector_carrier_supported,
        .source_function_vector_carrier_supported =
            context->policy->source_function_vector_carrier_supported,
        .structural_legality_flags =
            loom_low_lower_source_plan_uses_structured_control_flow(context)
                ? LOOM_TARGET_LOW_STRUCTURAL_LEGALITY_ALLOW_SOURCE_SCF
                : 0,
        .diagnostic_flags = options->legality_diagnostic_flags,
        .emitter = options->emitter,
        .max_errors = options->max_errors,
    };
    status = loom_target_low_verify_function_legality(
        module, source_function, &legality_options, &legality_result);
  }
  if (iree_status_is_ok(status)) {
    out_result->error_count = legality_result.error_count;
    out_result->remark_count = legality_result.remark_count;
    out_result->descriptor_set = legality_result.descriptor_set;
    context->descriptor_set = legality_result.descriptor_set;
  }
  if (iree_status_is_ok(status) && out_result->error_count != 0) {
    return iree_ok_status();
  }

  if (iree_status_is_ok(status) &&
      context->lowering->value_domain.value_count != 0) {
    status = iree_arena_allocate_array(
        context->function_arena, context->lowering->value_domain.value_count,
        sizeof(*context->lowering->value_bindings),
        (void**)&context->lowering->value_bindings);
  }
  if (iree_status_is_ok(status)) {
    for (loom_value_ordinal_t i = 0;
         i < context->lowering->value_domain.value_count; ++i) {
      context->lowering->value_bindings[i].type = LOOM_TYPE_ID_INVALID;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_function_boundary_validate(context);
  }
  if (iree_status_is_ok(status) &&
      !loom_low_lower_context_diagnostic_limit_reached(context)) {
    status = loom_low_lower_source_plan_build(context, source_body);
  }
  if (iree_status_is_ok(status) && context->result->error_count == 0) {
    status = loom_low_lower_control_plan_build(context);
  }
  if (iree_status_is_ok(status) && context->result->error_count == 0) {
    status = loom_low_lower_report_prepare(context);
  }
  return status;
}

iree_status_t loom_low_lower_plan_function(
    loom_module_t* module, loom_func_like_t source_function,
    const loom_low_lower_options_t* options, iree_arena_allocator_t* arena,
    loom_low_lower_result_t* out_result,
    loom_low_lower_function_plan_t** out_plan) {
  loom_low_lower_assert_options(module, source_function, options);
  *out_plan = NULL;
  *out_result = (loom_low_lower_result_t){
      .low_func_ref = loom_symbol_ref_null(),
  };
  loom_low_lower_function_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*plan), (void**)&plan));
  *plan = (loom_low_lower_function_plan_t){
      .module = module,
      .source_function = source_function,
      .emission =
          {
              .target_ref = options->target_ref,
              .target_facts = options->target_facts,
              .policy = options->policy,
              .module_state = options->module_state,
              .on_function_created = options->on_function_created,
              .emitter = options->emitter,
              .diagnostic_flags = options->legality_diagnostic_flags,
              .control_flow_lowering = options->control_flow_lowering,
              .sanitizer_reporting_mode = options->sanitizer_reporting_mode,
          },
      .arena = arena,
  };
  const loom_region_descriptor_t* source_body_descriptor =
      loom_func_like_body_region_descriptor(module, source_function);
  plan->frame.source_callable_exit_kind = source_body_descriptor->terminator;
  loom_low_lower_context_t context = {
      .module = module,
      .source_function = source_function,
      .options = options,
      .policy = options->policy,
      .result = out_result,
      .function_arena = plan->arena,
      .fact_table = options->fact_table,
      .module_state = options->module_state,
      .lowering = &plan->frame,
  };
  iree_arena_initialize(module->arena.block_pool, &context.analysis_arena);
  loom_storage_access_scope_t storage_access;
  loom_storage_access_scope_initialize(module, &context.analysis_arena,
                                       &storage_access);
  context.storage_access =
      options->storage_access ? options->storage_access : &storage_access;
  iree_status_t status = loom_low_lower_function_plan_build(&context);
  loom_local_value_domain_t* domain = &plan->frame.value_domain;
  if (iree_status_is_ok(status) && out_result->error_count == 0) {
    // Construction can grow the domain in scratch. Retain only its completed
    // ordinal sequence, preserving every assignment without another IR walk.
    status = loom_local_value_domain_relocate(domain, arena);
  }
  loom_local_value_domain_release(domain);
  iree_arena_deinitialize(&context.analysis_arena);
  if (iree_status_is_ok(status) && out_result->error_count == 0) {
    plan->descriptor_set = context.descriptor_set;
    *out_plan = plan;
  }
  return status;
}

static iree_status_t loom_low_lower_function_plan_execute(
    loom_low_lower_context_t* context) {
  loom_module_t* module = context->module;
  const loom_func_like_t source_function = context->source_function;
  loom_region_t* source_body = loom_func_like_body(source_function);
  const loom_symbol_ref_t low_func_ref = loom_func_like_callee(source_function);
  iree_arena_initialize(module->arena.block_pool, &context->emission_arena);

  loom_low_lower_emission_scope_begin(context);
  iree_status_t status = loom_low_lower_function_boundary_create(
      context, source_body, low_func_ref);
  loom_low_lower_emission_scope_end(context);
  if (iree_status_is_ok(status)) {
    loom_low_lower_report_initialize(context);
    status = loom_low_lower_map_blocks(context, source_body);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_function_boundary_remap_predicates(context);
  }
  if (iree_status_is_ok(status)) {
    loom_low_lower_emission_scope_begin(context);
    status = loom_low_lower_emit_preamble(context);
    loom_low_lower_emission_scope_end(context);
  }
  if (iree_status_is_ok(status)) {
    loom_low_lower_emission_scope_begin(context);
    status = loom_low_lower_function_boundary_emit_resource_imports(context);
    loom_low_lower_emission_scope_end(context);
  }
  if (iree_status_is_ok(status)) {
    loom_low_lower_emission_scope_begin(context);
    status = loom_low_lower_emit_entry_setup(context);
    loom_low_lower_emission_scope_end(context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_emit_body(context, source_body);
  }
  // The replacement low op carries the source symbol while the source op
  // still owns the symbol table entry. Erase clears that entry; relink it to
  // the replacement so callers keep the same symbol identity.
  if (iree_status_is_ok(status)) {
    status = loom_op_erase(module, source_function.op);
  }
  if (iree_status_is_ok(status)) {
    loom_module_link_symbol_defining_op(
        module, context->low_func_op,
        loom_op_vtable(module, context->low_func_op));
  }

  iree_arena_deinitialize(&context->emission_arena);
  return status;
}

iree_status_t loom_low_lower_emit_function(loom_low_lower_function_plan_t* plan,
                                           loom_low_lower_result_t* result) {
  const loom_low_lower_options_t options = {
      .target_ref = plan->emission.target_ref,
      .target_facts = plan->emission.target_facts,
      .policy = plan->emission.policy,
      .module_state = plan->emission.module_state,
      .on_function_created = plan->emission.on_function_created,
      .emitter = plan->emission.emitter,
      .legality_diagnostic_flags = plan->emission.diagnostic_flags,
      .control_flow_lowering = plan->emission.control_flow_lowering,
      .sanitizer_reporting_mode = plan->emission.sanitizer_reporting_mode,
      .report_allocator = result->report != NULL ? result->report->allocator
                                                 : iree_allocator_null(),
  };
  loom_low_lower_context_t context = {
      .module = plan->module,
      .source_function = plan->source_function,
      .options = &options,
      .policy = options.policy,
      .descriptor_set = plan->descriptor_set,
      .result = result,
      .function_arena = plan->arena,
      .module_state = options.module_state,
      .lowering = &plan->frame,
  };
  loom_local_value_domain_restore(&plan->frame.value_domain);
  iree_status_t status = loom_low_lower_function_plan_execute(&context);
  loom_local_value_domain_release(&plan->frame.value_domain);
  return status;
}

iree_status_t loom_low_lower_function(loom_module_t* module,
                                      loom_func_like_t source_function,
                                      const loom_low_lower_options_t* options,
                                      loom_low_lower_result_t* out_result) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(module->arena.block_pool, &arena);
  loom_low_lower_function_plan_t* plan = NULL;
  iree_status_t status = loom_low_lower_plan_function(
      module, source_function, options, &arena, out_result, &plan);
  if (iree_status_is_ok(status) && plan != NULL) {
    status = loom_low_lower_emit_function(plan, out_result);
  }
  iree_arena_deinitialize(&arena);
  return status;
}
