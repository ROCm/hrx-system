// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/boundary/projection_apply.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/rewrite/remap.h"

static iree_status_t loom_boundary_projection_name_component(
    loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_slot_t* candidate, loom_value_id_t component,
    iree_string_view_t suffix) {
  return loom_rewriter_try_set_derived_value_name(
      &plan->rewriter, candidate->value_id, component, suffix);
}

static iree_status_t loom_boundary_projection_preallocate_candidates(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  for (iree_host_size_t i = 0; i < function->candidate_count; ++i) {
    loom_boundary_projection_slot_t* candidate = &function->candidates[i];
    if (candidate->selected &&
        candidate->role == LOOM_BOUNDARY_PROJECTION_SLOT_BLOCK_ARGUMENT &&
        candidate->schema.component_count != 0) {
      candidate->component_value_ids[candidate->schema.component_count - 1] =
          candidate->value_id;
    }
  }
  for (iree_host_size_t i = 0; i < function->call_count; ++i) {
    loom_boundary_projection_call_t* call_plan = &function->calls[i];
    const loom_boundary_projection_function_t* callee =
        &plan->functions[call_plan->callee_index];
    const uint16_t result_offset =
        loom_call_like_result_offset(call_plan->call);
    const uint16_t final_result_count =
        (uint16_t)(result_offset + callee->final_result_count);
    if (final_result_count != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          plan->arena, final_result_count, sizeof(*call_plan->result_ids),
          (void**)&call_plan->result_ids));
    }
    for (uint16_t j = 0; j < final_result_count; ++j) {
      IREE_RETURN_IF_ERROR(loom_module_define_value(
          plan->module, loom_type_none(), &call_plan->result_ids[j]));
    }
    const loom_value_slice_t call_results =
        loom_call_like_results(call_plan->call);
    uint16_t next_result = result_offset;
    for (uint16_t j = 0; j < callee->result_count; ++j) {
      const loom_boundary_projection_schema_t* schema =
          &callee->result_schemas[j];
      if (!loom_boundary_projection_schema_is_projected(schema)) {
        ++next_result;
        continue;
      }
      const iree_host_size_t candidate_index =
          loom_boundary_projection_slot_index(function, call_results.values[j]);
      IREE_ASSERT_NE(candidate_index, IREE_HOST_SIZE_MAX);
      loom_boundary_projection_slot_t* candidate =
          &function->candidates[candidate_index];
      for (uint16_t component = 0; component < schema->component_count;
           ++component) {
        candidate->component_value_ids[component] =
            call_plan->result_ids[next_result++];
      }
    }
    IREE_ASSERT_EQ(next_result, final_result_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_reconstruct_block_candidate(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_t* candidate, loom_type_t logical_type) {
  IREE_ASSERT(candidate->role == LOOM_BOUNDARY_PROJECTION_SLOT_BLOCK_ARGUMENT);
  loom_block_t* block = candidate->block;
  const uint16_t original_index = loom_value_def_index(
      loom_module_value(plan->module, candidate->value_id));
  const loom_boundary_projection_schema_t* schema = &candidate->schema;
  for (uint16_t component = 0; component + 1 < schema->component_count;
       ++component) {
    IREE_RETURN_IF_ERROR(loom_module_define_value(
        plan->module, schema->component_types[component],
        &candidate->component_value_ids[component]));
    IREE_RETURN_IF_ERROR(loom_block_insert_arg(
        plan->module, block, (uint16_t)(original_index + component),
        candidate->component_value_ids[component]));
  }

  loom_builder_t* builder = &plan->rewriter.builder;
  IREE_ASSERT(candidate->reconstruction_anchor != NULL);
  loom_builder_set_before(builder, candidate->reconstruction_anchor);
  IREE_RETURN_IF_ERROR(loom_builder_reserve_results(
      builder, 1, &candidate->replacement_value_id));
  IREE_RETURN_IF_ERROR(loom_module_set_value_type(
      plan->module, candidate->replacement_value_id, logical_type));
  IREE_RETURN_IF_ERROR(loom_rewriter_move_value_name(
      &plan->rewriter, candidate->value_id, candidate->replacement_value_id));
  for (uint16_t component = 0; component < schema->component_count;
       ++component) {
    if (schema->component_name_suffixes) {
      IREE_RETURN_IF_ERROR(loom_rewriter_try_set_derived_value_name(
          &plan->rewriter, candidate->replacement_value_id,
          candidate->component_value_ids[component],
          schema->component_name_suffixes[component]));
    }
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
      &plan->rewriter, candidate->value_id, candidate->replacement_value_id));
  if (schema->component_count != 0) {
    IREE_RETURN_IF_ERROR(loom_rewriter_set_value_type(
        &plan->rewriter,
        candidate->component_value_ids[schema->component_count - 1],
        schema->component_types[schema->component_count - 1]));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_block_remove_arg(plan->module, block, original_index));
  }
  IREE_RETURN_IF_ERROR(schema->rule->transport.reconstruct(
      schema->rule, plan, function, candidate, logical_type,
      candidate->reconstruction_anchor->location,
      &candidate->replacement_value_id));
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_initialize_remap(
    loom_boundary_projection_plan_t* plan, loom_ir_remap_t* out_remap) {
  return loom_ir_remap_initialize(
      plan->module, plan->module, plan->arena,
      &(loom_ir_remap_options_t){
          .allow_unmapped_values = true,
          .remap_symbol = loom_ir_remap_symbol_callback_empty(),
      },
      out_remap);
}

static iree_status_t loom_boundary_projection_reconstruct_block_candidates(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_boundary_projection_initialize_remap(plan, &remap));
  for (iree_host_size_t i = 0; i < function->reconstruction_count; ++i) {
    loom_boundary_projection_slot_t* candidate =
        &function->candidates[function->reconstruction_order[i]];
    loom_type_t logical_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_type(&remap, candidate->logical_type, &logical_type));
    IREE_RETURN_IF_ERROR(loom_boundary_projection_reconstruct_block_candidate(
        plan, function, candidate, logical_type));
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
        &remap, candidate->value_id, candidate->replacement_value_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_copy_attributes(
    loom_ir_remap_t* remap, const loom_op_t* source, loom_op_t* target) {
  const loom_attribute_t* source_attributes = loom_op_const_attrs(source);
  loom_attribute_t* target_attributes = loom_op_attrs(target);
  for (uint8_t i = 0; i < source->attribute_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_attribute(remap, source_attributes[i],
                                                 &target_attributes[i]));
  }
  return iree_ok_status();
}

// Projects the FuncLike specialization prefix through argument expansion.
static void loom_boundary_projection_update_specialization_count(
    const loom_boundary_projection_function_t* function, loom_op_t* target) {
  const uint8_t specialization_count_attr_index =
      function->function.vtable->specialization_count_attr_index;
  if (specialization_count_attr_index == LOOM_ATTR_INDEX_NONE) {
    return;
  }

  const int64_t source_specialization_count =
      loom_func_like_specialization_count(function->function);
  IREE_ASSERT_GE(source_specialization_count, 0);
  IREE_ASSERT_LE(source_specialization_count, function->argument_count);
  const uint16_t specialization_count = (uint16_t)source_specialization_count;
  const uint16_t target_specialization_count =
      specialization_count == function->argument_count
          ? function->final_argument_count
          : function->argument_indices[specialization_count];
  loom_op_attrs(target)[specialization_count_attr_index] =
      loom_attr_i64(target_specialization_count);
}

static iree_status_t loom_boundary_projection_copy_comments(
    loom_boundary_projection_plan_t* plan, const loom_op_t* source,
    const loom_op_t* target) {
  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(plan->module, source, &comment_count);
  return comment_count == 0
             ? iree_ok_status()
             : loom_module_attach_op_comments(plan->module, target, comments,
                                              comment_count);
}

static iree_status_t loom_boundary_projection_remap_call_ties(
    loom_call_like_t call, const loom_boundary_projection_function_t* callee,
    loom_op_t* target) {
  const uint16_t operand_offset = loom_call_like_operand_offset(call);
  const uint16_t result_offset = loom_call_like_result_offset(call);
  const loom_tied_result_t* source_ties = loom_op_tied_results(call.op);
  loom_tied_result_t* target_ties = loom_op_tied_results(target);
  for (uint16_t i = 0; i < call.op->tied_result_count; ++i) {
    target_ties[i] = source_ties[i];
    if (source_ties[i].operand_index >= operand_offset) {
      const uint16_t argument_index =
          (uint16_t)(source_ties[i].operand_index - operand_offset);
      IREE_ASSERT_LT(argument_index, callee->argument_count);
      IREE_ASSERT(!loom_boundary_projection_schema_is_projected(
          &callee->argument_schemas[argument_index]));
      target_ties[i].operand_index =
          (uint16_t)(operand_offset + callee->argument_indices[argument_index]);
    }
    if (source_ties[i].result_index >= result_offset) {
      const uint16_t result_index =
          (uint16_t)(source_ties[i].result_index - result_offset);
      IREE_ASSERT_LT(result_index, callee->result_count);
      IREE_ASSERT(!loom_boundary_projection_schema_is_projected(
          &callee->result_schemas[result_index]));
      target_ties[i].result_index =
          (uint16_t)(result_offset + callee->result_indices[result_index]);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_call_segment_counts(
    loom_boundary_projection_plan_t* plan, loom_call_like_t call,
    const loom_boundary_projection_function_t* callee,
    const uint16_t** out_segment_counts, uint8_t* out_segment_count) {
  *out_segment_counts = NULL;
  const loom_op_vtable_t* vtable = loom_op_vtable(plan->module, call.op);
  *out_segment_count = loom_op_vtable_operand_segment_count(vtable);
  if (*out_segment_count == 0) {
    return iree_ok_status();
  }
  uint16_t* segment_counts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, *out_segment_count, sizeof(*segment_counts),
      (void**)&segment_counts));
  memcpy(segment_counts, loom_op_const_operand_segment_counts(call.op),
         *out_segment_count * sizeof(*segment_counts));

  uint16_t argument_index = 0;
  for (uint8_t segment = call.vtable->operand_field_index;
       segment < *out_segment_count; ++segment) {
    const uint16_t original_count = segment_counts[segment];
    for (uint16_t i = 0; i < original_count; ++i, ++argument_index) {
      IREE_ASSERT_LT(argument_index, callee->argument_count);
      const loom_boundary_projection_schema_t* schema =
          &callee->argument_schemas[argument_index];
      if (loom_boundary_projection_schema_is_projected(schema)) {
        segment_counts[segment] =
            (uint16_t)(segment_counts[segment] - 1 + schema->component_count);
      }
    }
  }
  IREE_ASSERT_EQ(argument_index, callee->argument_count);
  *out_segment_counts = segment_counts;
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_rebuild_call(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_call_t* call_plan) {
  loom_call_like_t call = call_plan->call;
  loom_op_t* source = call.op;
  loom_boundary_projection_function_t* callee =
      &plan->functions[call_plan->callee_index];
  const uint16_t operand_offset = loom_call_like_operand_offset(call);
  const uint16_t result_offset = loom_call_like_result_offset(call);
  const uint16_t final_operand_count =
      (uint16_t)(operand_offset + callee->final_argument_count);
  const uint16_t final_result_count =
      (uint16_t)(result_offset + callee->final_result_count);

  loom_value_id_t* operands = NULL;
  if (final_operand_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(plan->arena, final_operand_count,
                                  sizeof(*operands), (void**)&operands));
  }
  if (operand_offset != 0) {
    memcpy(operands, loom_op_const_operands(source),
           operand_offset * sizeof(*operands));
  }
  const loom_value_slice_t call_operands = loom_call_like_operands(call);
  uint16_t next_operand = operand_offset;
  for (uint16_t i = 0; i < callee->argument_count; ++i) {
    const loom_boundary_projection_schema_t* schema =
        &callee->argument_schemas[i];
    if (!loom_boundary_projection_schema_is_projected(schema)) {
      operands[next_operand++] = call_operands.values[i];
      continue;
    }
    const loom_boundary_projection_source_t* projection =
        &call_plan->operand_sources[i];
    IREE_ASSERT(projection->rule == schema->rule);
    loom_value_id_t* component_values =
        schema->component_count != 0 ? &operands[next_operand] : NULL;
    IREE_RETURN_IF_ERROR(schema->rule->transport.materialize_source(
        schema->rule, plan, function, projection, component_values));
    next_operand = (uint16_t)(next_operand + schema->component_count);
  }
  IREE_ASSERT_EQ(next_operand, final_operand_count);

  loom_builder_t* builder = &plan->rewriter.builder;
  loom_builder_set_before(builder, source);
  loom_value_id_t* results = call_plan->result_ids;
  loom_type_t* result_types = NULL;
  if (final_result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, final_result_count, sizeof(*result_types),
        (void**)&result_types));
  }

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_boundary_projection_initialize_remap(plan, &remap));
  const loom_value_id_t* source_results = loom_op_const_results(source);
  for (uint16_t i = 0; i < result_offset; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_map_value(&remap, source_results[i], results[i]));
  }
  uint16_t next_result = result_offset;
  const loom_value_slice_t call_results = loom_call_like_results(call);
  for (uint16_t i = 0; i < callee->result_count; ++i) {
    const loom_boundary_projection_schema_t* schema =
        &callee->result_schemas[i];
    if (!loom_boundary_projection_schema_is_projected(schema)) {
      IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
          &remap, call_results.values[i], results[next_result++]));
      continue;
    }
    next_result = (uint16_t)(next_result + schema->component_count);
  }
  IREE_ASSERT_EQ(next_result, final_result_count);

  for (uint16_t i = 0; i < result_offset; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        &remap, loom_module_value_type(plan->module, source_results[i]),
        &result_types[i]));
  }
  next_result = result_offset;
  for (uint16_t i = 0; i < callee->result_count; ++i) {
    const loom_boundary_projection_schema_t* schema =
        &callee->result_schemas[i];
    if (loom_boundary_projection_schema_is_projected(schema)) {
      for (uint16_t component = 0; component < schema->component_count;
           ++component) {
        result_types[next_result++] = schema->component_types[component];
      }
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        &remap, loom_module_value_type(plan->module, call_results.values[i]),
        &result_types[next_result++]));
  }

  const uint16_t* segment_counts = NULL;
  uint8_t segment_count = 0;
  IREE_RETURN_IF_ERROR(loom_boundary_projection_call_segment_counts(
      plan, call, callee, &segment_counts, &segment_count));
  loom_op_t* target = NULL;
  if (segment_count != 0) {
    IREE_RETURN_IF_ERROR(loom_builder_allocate_segmented_op(
        builder, source->kind, final_operand_count, segment_counts,
        segment_count, final_result_count, /*region_count=*/0,
        source->tied_result_count, source->attribute_count, source->location,
        &target));
  } else {
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
        builder, source->kind, final_operand_count, final_result_count,
        /*region_count=*/0, source->tied_result_count, source->attribute_count,
        source->location, &target));
  }
  target->instance_flags = source->instance_flags;
  target->traits = source->traits;
  target->flags |= source->flags & LOOM_OP_SOURCE_PRESENTATION_FLAG_MASK;
  if (final_operand_count != 0) {
    memcpy(loom_op_operands(target), operands,
           final_operand_count * sizeof(*operands));
  }
  for (uint16_t i = 0; i < final_result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_module_set_value_type(plan->module, results[i], result_types[i]));
  }
  if (final_result_count != 0) {
    memcpy(loom_op_results(target), results,
           final_result_count * sizeof(*results));
  }
  IREE_RETURN_IF_ERROR(
      loom_boundary_projection_remap_call_ties(call, callee, target));
  IREE_RETURN_IF_ERROR(
      loom_boundary_projection_copy_attributes(&remap, source, target));
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, target));
  IREE_RETURN_IF_ERROR(
      loom_boundary_projection_copy_comments(plan, source, target));

  for (uint16_t i = 0; i < result_offset; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_move_value_name(
        &plan->rewriter, source_results[i], results[i]));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        &plan->rewriter, source_results[i], results[i]));
  }
  next_result = result_offset;
  for (uint16_t i = 0; i < callee->result_count; ++i) {
    const loom_value_id_t source_result = call_results.values[i];
    const loom_boundary_projection_schema_t* schema =
        &callee->result_schemas[i];
    if (!loom_boundary_projection_schema_is_projected(schema)) {
      IREE_RETURN_IF_ERROR(loom_rewriter_move_value_name(
          &plan->rewriter, source_result, results[next_result]));
      IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
          &plan->rewriter, source_result, results[next_result++]));
      continue;
    }
    const iree_host_size_t candidate_index =
        loom_boundary_projection_slot_index(function, source_result);
    loom_boundary_projection_slot_t* candidate =
        &function->candidates[candidate_index];
    for (uint16_t component = 0; component < schema->component_count;
         ++component) {
      if (schema->component_name_suffixes) {
        IREE_RETURN_IF_ERROR(loom_boundary_projection_name_component(
            plan, candidate, candidate->component_value_ids[component],
            schema->component_name_suffixes[component]));
      }
    }
    loom_type_t logical_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_type(&remap, candidate->logical_type, &logical_type));
    loom_builder_set_after(builder, target);
    IREE_RETURN_IF_ERROR(schema->rule->transport.reconstruct(
        schema->rule, plan, function, candidate, logical_type, source->location,
        &candidate->replacement_value_id));
    IREE_RETURN_IF_ERROR(loom_rewriter_move_value_name(
        &plan->rewriter, source_result, candidate->replacement_value_id));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        &plan->rewriter, source_result, candidate->replacement_value_id));
    next_result = (uint16_t)(next_result + schema->component_count);
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(&plan->rewriter, source));
  ++plan->calls_rewritten;
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_rebuild_calls(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  for (iree_host_size_t i = 0; i < function->call_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_boundary_projection_rebuild_call(
        plan, function, &function->calls[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_rebuild_return(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_return_t* return_plan) {
  loom_op_t* source = return_plan->op;
  loom_value_id_t* operands = NULL;
  if (function->final_result_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(plan->arena, function->final_result_count,
                                  sizeof(*operands), (void**)&operands));
  }
  const loom_value_id_t* source_operands = loom_op_const_operands(source);
  uint16_t next_operand = 0;
  for (uint16_t i = 0; i < function->result_count; ++i) {
    const loom_boundary_projection_schema_t* schema =
        &function->result_schemas[i];
    if (!loom_boundary_projection_schema_is_projected(schema)) {
      operands[next_operand++] = source_operands[i];
      continue;
    }
    const loom_boundary_projection_source_t* projection =
        &return_plan->sources[i];
    IREE_ASSERT(projection->rule == schema->rule);
    loom_value_id_t* component_values =
        schema->component_count != 0 ? &operands[next_operand] : NULL;
    IREE_RETURN_IF_ERROR(schema->rule->transport.materialize_source(
        schema->rule, plan, function, projection, component_values));
    next_operand = (uint16_t)(next_operand + schema->component_count);
  }
  IREE_ASSERT_EQ(next_operand, function->final_result_count);

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_boundary_projection_initialize_remap(plan, &remap));
  loom_builder_t* builder = &plan->rewriter.builder;
  loom_builder_set_before(builder, source);
  loom_op_t* target = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      builder, source->kind, function->final_result_count,
      /*result_count=*/0, /*region_count=*/0, /*tied_result_count=*/0,
      source->attribute_count, source->location, &target));
  target->instance_flags = source->instance_flags;
  target->traits = source->traits;
  target->flags |= source->flags & LOOM_OP_SOURCE_PRESENTATION_FLAG_MASK;
  if (function->final_result_count != 0) {
    memcpy(loom_op_operands(target), operands,
           function->final_result_count * sizeof(*operands));
  }
  IREE_RETURN_IF_ERROR(
      loom_boundary_projection_copy_attributes(&remap, source, target));
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, target));
  IREE_RETURN_IF_ERROR(
      loom_boundary_projection_copy_comments(plan, source, target));
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(&plan->rewriter, source));
  ++plan->returns_rewritten;
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_rebuild_returns(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  for (iree_host_size_t i = 0; i < function->return_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_boundary_projection_rebuild_return(
        plan, function, &function->returns[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_rebuild_edge(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_block_t* block_plan,
    const loom_boundary_projection_edge_t* edge_plan) {
  loom_op_t* source = edge_plan->terminator;
  const loom_value_id_t* old_arguments = loom_op_const_operands(source);
  loom_value_id_t* new_arguments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, block_plan->final_argument_count, sizeof(*new_arguments),
      (void**)&new_arguments));
  uint16_t next = 0;
  for (uint16_t i = 0; i < block_plan->original_argument_count; ++i) {
    const iree_host_size_t candidate_index =
        loom_boundary_projection_slot_index(function,
                                            block_plan->original_arguments[i]);
    if (candidate_index == IREE_HOST_SIZE_MAX ||
        !function->candidates[candidate_index].selected) {
      new_arguments[next++] = old_arguments[i];
      continue;
    }
    const loom_boundary_projection_slot_t* candidate =
        &function->candidates[candidate_index];
    const loom_boundary_projection_source_t* projection =
        &edge_plan->sources[i];
    IREE_ASSERT(projection->rule == candidate->schema.rule);
    loom_value_id_t* component_values =
        candidate->schema.component_count != 0 ? &new_arguments[next] : NULL;
    IREE_RETURN_IF_ERROR(candidate->schema.rule->transport.materialize_source(
        candidate->schema.rule, plan, function, projection, component_values));
    next = (uint16_t)(next + candidate->schema.component_count);
  }
  IREE_ASSERT_EQ(next, block_plan->final_argument_count);

  loom_builder_t* builder = &plan->rewriter.builder;
  loom_builder_set_before(builder, source);
  loom_op_t* target = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op_with_successors(
      builder, source->kind, next, /*result_count=*/0,
      /*successor_count=*/1, /*region_count=*/0, /*tied_result_count=*/0,
      source->attribute_count, source->location, &target));
  target->instance_flags = source->instance_flags;
  target->traits = source->traits;
  target->flags |= source->flags & LOOM_OP_SOURCE_PRESENTATION_FLAG_MASK;
  loom_op_successors(target)[0] = block_plan->block;
  if (next != 0) {
    memcpy(loom_op_operands(target), new_arguments,
           next * sizeof(*new_arguments));
  }
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_boundary_projection_initialize_remap(plan, &remap));
  IREE_RETURN_IF_ERROR(
      loom_boundary_projection_copy_attributes(&remap, source, target));
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, target));
  IREE_RETURN_IF_ERROR(
      loom_boundary_projection_copy_comments(plan, source, target));
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(&plan->rewriter, source));
  ++plan->cfg_edges_rewritten;
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_rebuild_edges(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  for (iree_host_size_t i = 0; i < function->block_count; ++i) {
    const loom_boundary_projection_block_t* block_plan = &function->blocks[i];
    for (iree_host_size_t j = 0; j < block_plan->edge_count; ++j) {
      IREE_RETURN_IF_ERROR(loom_boundary_projection_rebuild_edge(
          plan, function, block_plan, &block_plan->edges[j]));
    }
  }
  return iree_ok_status();
}

typedef struct loom_boundary_projection_function_operand_layout_t {
  // Flat offset of declaration signature arguments.
  uint16_t argument_offset;
  // Rebuilt operation operand count.
  uint16_t operand_count;
  // Rebuilt operand segment counts, or NULL for a flat operation.
  const uint16_t* segment_counts;
  // Number of entries in segment_counts.
  uint8_t segment_count;
} loom_boundary_projection_function_operand_layout_t;

static iree_status_t loom_boundary_projection_plan_function_operand_layout(
    loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    loom_boundary_projection_function_operand_layout_t* out_layout) {
  loom_op_t* op = function->function.op;
  const loom_op_vtable_t* vtable = loom_op_vtable(plan->module, op);
  const uint8_t segment_count = loom_op_vtable_operand_segment_count(vtable);
  *out_layout = (loom_boundary_projection_function_operand_layout_t){
      .operand_count = op->operand_count,
      .segment_count = segment_count,
  };
  if (loom_func_like_body(function->function)) {
    out_layout->segment_counts =
        segment_count != 0 ? loom_op_const_operand_segment_counts(op) : NULL;
    return iree_ok_status();
  }

  const uint8_t argument_field =
      function->function.vtable->args_operand_field_index;
  if (argument_field == LOOM_OPERAND_INDEX_NONE) {
    IREE_ASSERT_EQ(function->argument_count, 0);
    IREE_ASSERT_EQ(op->operand_count, 0);
    return iree_ok_status();
  }
  out_layout->argument_offset = function->argument_operand_offset;
  IREE_ASSERT_NE(out_layout->argument_offset, UINT16_MAX);
  if (segment_count != 0) {
    const uint16_t* source_counts = loom_op_const_operand_segment_counts(op);
    uint16_t* target_counts = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan->arena, segment_count,
                                                   sizeof(*target_counts),
                                                   (void**)&target_counts));
    memcpy(target_counts, source_counts,
           segment_count * sizeof(*target_counts));
    IREE_ASSERT_EQ(source_counts[argument_field], function->argument_count);
    target_counts[argument_field] = function->final_argument_count;
    out_layout->segment_counts = target_counts;
  }
  out_layout->operand_count =
      (uint16_t)(op->operand_count + function->final_argument_count -
                 function->argument_count);
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_define_values(
    loom_boundary_projection_plan_t* plan, uint16_t count,
    loom_value_id_t** out_values) {
  *out_values = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, count, sizeof(**out_values), (void**)out_values));
  for (uint16_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_module_define_value(
        plan->module, loom_type_none(), &(*out_values)[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_map_function_results(
    const loom_boundary_projection_function_t* function,
    const loom_value_id_t* target_results, loom_ir_remap_t* remap) {
  for (uint16_t i = 0; i < function->result_count; ++i) {
    if (loom_boundary_projection_schema_is_projected(
            &function->result_schemas[i])) {
      continue;
    }
    const uint16_t target_index = function->result_indices[i];
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(remap, function->results[i],
                                                 target_results[target_index]));
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_set_function_result_types(
    loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    const loom_value_id_t* target_results, loom_ir_remap_t* remap) {
  uint16_t target_index = 0;
  for (uint16_t i = 0; i < function->result_count; ++i) {
    const loom_boundary_projection_schema_t* schema =
        &function->result_schemas[i];
    if (loom_boundary_projection_schema_is_projected(schema)) {
      for (uint16_t component = 0; component < schema->component_count;
           ++component) {
        IREE_RETURN_IF_ERROR(loom_module_set_value_type(
            plan->module, target_results[target_index++],
            schema->component_types[component]));
      }
      continue;
    }
    loom_type_t type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        remap, loom_module_value_type(plan->module, function->results[i]),
        &type));
    IREE_RETURN_IF_ERROR(loom_module_set_value_type(
        plan->module, target_results[target_index++], type));
  }
  IREE_ASSERT_EQ(target_index, function->final_result_count);
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_prepare_body_arguments(
    loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    loom_value_id_t** out_arguments, loom_ir_remap_t* remap) {
  loom_region_t* source_body = loom_func_like_body(function->function);
  loom_block_t* source_entry = loom_region_entry_block(source_body);
  IREE_ASSERT_EQ(source_entry->arg_count, function->final_argument_count);
  IREE_RETURN_IF_ERROR(loom_boundary_projection_define_values(
      plan, function->final_argument_count, out_arguments));
  IREE_RETURN_IF_ERROR(loom_ir_remap_assign_value_types(
      plan->module, source_entry->arg_ids, *out_arguments,
      function->final_argument_count));
  for (uint16_t i = 0; i < function->argument_count; ++i) {
    if (loom_boundary_projection_schema_is_projected(
            &function->argument_schemas[i])) {
      continue;
    }
    const uint16_t target_index = function->argument_indices[i];
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
        remap, function->arguments[i], (*out_arguments)[target_index]));
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_prepare_declaration_operands(
    loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    const loom_boundary_projection_function_operand_layout_t* layout,
    loom_value_id_t** out_operands, loom_ir_remap_t* remap) {
  loom_op_t* source = function->function.op;
  IREE_RETURN_IF_ERROR(loom_boundary_projection_define_values(
      plan, layout->operand_count, out_operands));
  const loom_value_id_t* source_operands = loom_op_const_operands(source);
  const uint16_t source_argument_end =
      (uint16_t)(layout->argument_offset + function->argument_count);
  const uint16_t target_argument_end =
      (uint16_t)(layout->argument_offset + function->final_argument_count);

  for (uint16_t i = 0; i < layout->argument_offset; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_map_value(remap, source_operands[i], (*out_operands)[i]));
  }
  for (uint16_t i = 0; i < function->argument_count; ++i) {
    if (loom_boundary_projection_schema_is_projected(
            &function->argument_schemas[i])) {
      continue;
    }
    const uint16_t target_index =
        (uint16_t)(layout->argument_offset + function->argument_indices[i]);
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
        remap, function->arguments[i], (*out_operands)[target_index]));
  }
  for (uint16_t source_index = source_argument_end,
                target_index = target_argument_end;
       source_index < source->operand_count; ++source_index, ++target_index) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
        remap, source_operands[source_index], (*out_operands)[target_index]));
  }

  for (uint16_t i = 0; i < layout->argument_offset; ++i) {
    loom_type_t type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        remap, loom_module_value_type(plan->module, source_operands[i]),
        &type));
    IREE_RETURN_IF_ERROR(
        loom_module_set_value_type(plan->module, (*out_operands)[i], type));
  }
  uint16_t target_index = layout->argument_offset;
  for (uint16_t i = 0; i < function->argument_count; ++i) {
    const loom_boundary_projection_schema_t* schema =
        &function->argument_schemas[i];
    if (loom_boundary_projection_schema_is_projected(schema)) {
      for (uint16_t component = 0; component < schema->component_count;
           ++component) {
        IREE_RETURN_IF_ERROR(loom_module_set_value_type(
            plan->module, (*out_operands)[target_index++],
            schema->component_types[component]));
      }
      continue;
    }
    loom_type_t type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        remap, loom_module_value_type(plan->module, function->arguments[i]),
        &type));
    IREE_RETURN_IF_ERROR(loom_module_set_value_type(
        plan->module, (*out_operands)[target_index++], type));
  }
  for (uint16_t source_index = source_argument_end;
       source_index < source->operand_count; ++source_index, ++target_index) {
    loom_type_t type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        remap,
        loom_module_value_type(plan->module, source_operands[source_index]),
        &type));
    IREE_RETURN_IF_ERROR(loom_module_set_value_type(
        plan->module, (*out_operands)[target_index], type));
  }
  IREE_ASSERT_EQ(target_index, layout->operand_count);
  return iree_ok_status();
}

static void loom_boundary_projection_remap_function_ties(
    const loom_boundary_projection_function_t* function, loom_op_t* target) {
  const loom_tied_result_t* source_ties =
      loom_op_tied_results(function->function.op);
  loom_tied_result_t* target_ties = loom_op_tied_results(target);
  for (uint16_t i = 0; i < function->function.op->tied_result_count; ++i) {
    IREE_ASSERT_LT(source_ties[i].result_index, function->result_count);
    IREE_ASSERT(!loom_boundary_projection_schema_is_projected(
        &function->result_schemas[source_ties[i].result_index]));
    target_ties[i] = source_ties[i];
    if (function->argument_operand_offset != UINT16_MAX &&
        source_ties[i].operand_index >= function->argument_operand_offset) {
      const uint16_t argument_index =
          (uint16_t)(source_ties[i].operand_index -
                     function->argument_operand_offset);
      if (argument_index < function->argument_count) {
        IREE_ASSERT(!loom_boundary_projection_schema_is_projected(
            &function->argument_schemas[argument_index]));
        target_ties[i].operand_index =
            (uint16_t)(function->argument_operand_offset +
                       function->argument_indices[argument_index]);
      } else {
        target_ties[i].operand_index =
            (uint16_t)(source_ties[i].operand_index +
                       function->final_argument_count -
                       function->argument_count);
      }
    }
    target_ties[i].result_index =
        function->result_indices[source_ties[i].result_index];
  }
}

static iree_status_t loom_boundary_projection_transfer_function_names(
    loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    const loom_boundary_projection_function_operand_layout_t* layout,
    const loom_value_id_t* target_arguments,
    const loom_value_id_t* target_operands,
    const loom_value_id_t* target_results) {
  loom_region_t* body = loom_func_like_body(function->function);
  if (body) {
    const loom_block_t* source_entry = loom_region_entry_block(body);
    for (uint16_t i = 0; i < function->final_argument_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_rewriter_move_value_name(
          &plan->rewriter, source_entry->arg_ids[i], target_arguments[i]));
    }
  } else {
    const loom_value_id_t* source_operands =
        loom_op_const_operands(function->function.op);
    for (uint16_t i = 0; i < layout->argument_offset; ++i) {
      IREE_RETURN_IF_ERROR(loom_rewriter_move_value_name(
          &plan->rewriter, source_operands[i], target_operands[i]));
    }
    uint16_t target_index = layout->argument_offset;
    for (uint16_t i = 0; i < function->argument_count; ++i) {
      const loom_boundary_projection_schema_t* schema =
          &function->argument_schemas[i];
      if (loom_boundary_projection_schema_is_projected(schema)) {
        for (uint16_t component = 0; component < schema->component_count;
             ++component) {
          if (schema->component_name_suffixes) {
            IREE_RETURN_IF_ERROR(loom_rewriter_try_set_derived_value_name(
                &plan->rewriter, function->arguments[i],
                target_operands[target_index],
                schema->component_name_suffixes[component]));
          }
          ++target_index;
        }
        loom_boundary_projection_record(plan, schema->rule, 1,
                                        schema->component_count);
      } else {
        IREE_RETURN_IF_ERROR(loom_rewriter_move_value_name(
            &plan->rewriter, function->arguments[i],
            target_operands[target_index++]));
      }
    }
    const uint16_t source_argument_end =
        (uint16_t)(layout->argument_offset + function->argument_count);
    for (uint16_t source_index = source_argument_end;
         source_index < function->function.op->operand_count;
         ++source_index, ++target_index) {
      IREE_RETURN_IF_ERROR(loom_rewriter_move_value_name(
          &plan->rewriter, source_operands[source_index],
          target_operands[target_index]));
    }
  }

  uint16_t target_result_index = 0;
  for (uint16_t i = 0; i < function->result_count; ++i) {
    const loom_boundary_projection_schema_t* schema =
        &function->result_schemas[i];
    if (loom_boundary_projection_schema_is_projected(schema)) {
      for (uint16_t component = 0; component < schema->component_count;
           ++component) {
        if (schema->component_name_suffixes) {
          IREE_RETURN_IF_ERROR(loom_rewriter_try_set_derived_value_name(
              &plan->rewriter, function->results[i],
              target_results[target_result_index],
              schema->component_name_suffixes[component]));
        }
        ++target_result_index;
      }
      loom_boundary_projection_record(plan, schema->rule, 1,
                                      schema->component_count);
    } else {
      IREE_RETURN_IF_ERROR(
          loom_rewriter_move_value_name(&plan->rewriter, function->results[i],
                                        target_results[target_result_index++]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_replace_function(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function) {
  loom_op_t* source = function->function.op;
  loom_region_t* source_body = loom_func_like_body(function->function);
  loom_boundary_projection_function_operand_layout_t layout;
  IREE_RETURN_IF_ERROR(loom_boundary_projection_plan_function_operand_layout(
      plan, function, &layout));

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_boundary_projection_initialize_remap(plan, &remap));
  loom_value_id_t* target_arguments = NULL;
  loom_value_id_t* target_operands = NULL;
  if (source_body) {
    IREE_RETURN_IF_ERROR(loom_boundary_projection_prepare_body_arguments(
        plan, function, &target_arguments, &remap));
  } else {
    IREE_RETURN_IF_ERROR(loom_boundary_projection_prepare_declaration_operands(
        plan, function, &layout, &target_operands, &remap));
    if (function->final_argument_count != 0) {
      target_arguments = target_operands + layout.argument_offset;
    }
  }

  loom_value_id_t* target_results = NULL;
  IREE_RETURN_IF_ERROR(loom_boundary_projection_define_values(
      plan, function->final_result_count, &target_results));
  IREE_RETURN_IF_ERROR(loom_boundary_projection_map_function_results(
      function, target_results, &remap));
  IREE_RETURN_IF_ERROR(loom_boundary_projection_set_function_result_types(
      plan, function, target_results, &remap));

  loom_builder_t* builder = &plan->rewriter.builder;
  loom_builder_set_before(builder, source);
  loom_op_t* target = NULL;
  if (layout.segment_count != 0) {
    IREE_RETURN_IF_ERROR(loom_builder_allocate_segmented_op(
        builder, source->kind, layout.operand_count, layout.segment_counts,
        layout.segment_count, function->final_result_count, source_body ? 1 : 0,
        source->tied_result_count, source->attribute_count, source->location,
        &target));
  } else {
    IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
        builder, source->kind, layout.operand_count,
        function->final_result_count, source_body ? 1 : 0,
        source->tied_result_count, source->attribute_count, source->location,
        &target));
  }
  target->instance_flags = source->instance_flags;
  target->traits = source->traits;
  target->flags |= source->flags & LOOM_OP_SOURCE_PRESENTATION_FLAG_MASK;
  if (layout.operand_count != 0) {
    memcpy(loom_op_operands(target),
           source_body ? loom_op_const_operands(source) : target_operands,
           layout.operand_count * sizeof(loom_value_id_t));
  }
  if (function->final_result_count != 0) {
    memcpy(loom_op_results(target), target_results,
           function->final_result_count * sizeof(*target_results));
  }
  loom_boundary_projection_remap_function_ties(function, target);
  IREE_RETURN_IF_ERROR(
      loom_boundary_projection_copy_attributes(&remap, source, target));
  loom_boundary_projection_update_specialization_count(function, target);

  loom_block_t* target_entry = NULL;
  loom_region_t* target_body = NULL;
  if (source_body) {
    IREE_RETURN_IF_ERROR(
        loom_builder_create_region(builder, target, 0, &target_entry));
    target_body = loom_op_regions(target)[0];
    for (uint16_t i = 0; i < function->final_argument_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_block_add_arg(plan->module, target_entry, target_arguments[i]));
    }
  }
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, target));
  IREE_RETURN_IF_ERROR(
      loom_boundary_projection_copy_comments(plan, source, target));
  IREE_RETURN_IF_ERROR(loom_boundary_projection_transfer_function_names(
      plan, function, &layout, target_arguments, target_operands,
      target_results));

  for (uint16_t i = 0; i < function->result_count; ++i) {
    if (loom_boundary_projection_schema_is_projected(
            &function->result_schemas[i])) {
      continue;
    }
    const uint16_t target_index = function->result_indices[i];
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        &plan->rewriter, function->results[i], target_results[target_index]));
  }

  if (source_body) {
    loom_block_t* moved_entry = NULL;
    IREE_RETURN_IF_ERROR(loom_rewriter_move_region_blocks(
        &plan->rewriter, source_body, source, target_body,
        /*target_block_index=*/1, target, &moved_entry));
    loom_builder_ip_t saved_ip =
        loom_builder_enter_region(builder, target, target_body);
    loom_op_t* branch = NULL;
    iree_status_t status = loom_cfg_br_build(
        builder, moved_entry, target_arguments, function->final_argument_count,
        source->location, &branch);
    loom_builder_restore(builder, saved_ip);
    IREE_RETURN_IF_ERROR(status);
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_erase(&plan->rewriter, source));
  loom_module_link_symbol_defining_op(plan->module, target,
                                      loom_op_vtable(plan->module, target));
  loom_function_version_update(function->version,
                               loom_func_like_cast(plan->module, target));
  function->function = loom_func_like_cast(plan->module, target);
  ++plan->functions_rewritten;
  return iree_ok_status();
}

static iree_status_t loom_boundary_projection_replace_functions(
    loom_boundary_projection_plan_t* plan) {
  for (iree_host_size_t i = 0; i < plan->function_count; ++i) {
    loom_boundary_projection_function_t* function = &plan->functions[i];
    if (!function->selected || !function->signature_changes) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_boundary_projection_replace_function(plan, function));
  }
  return iree_ok_status();
}

iree_status_t loom_boundary_projection_apply(
    loom_boundary_projection_plan_t* plan) {
  for (iree_host_size_t i = 0; i < plan->function_count; ++i) {
    if (!plan->functions[i].selected) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_boundary_projection_preallocate_candidates(
        plan, &plan->functions[i]));
  }
  for (iree_host_size_t i = 0; i < plan->function_count; ++i) {
    if (!plan->functions[i].selected) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_boundary_projection_reconstruct_block_candidates(
        plan, &plan->functions[i]));
  }
  for (iree_host_size_t i = 0; i < plan->function_count; ++i) {
    if (!plan->functions[i].selected) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_boundary_projection_rebuild_calls(plan, &plan->functions[i]));
  }
  for (iree_host_size_t i = 0; i < plan->function_count; ++i) {
    if (!plan->functions[i].selected) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_boundary_projection_rebuild_returns(plan, &plan->functions[i]));
  }
  for (iree_host_size_t i = 0; i < plan->function_count; ++i) {
    if (!plan->functions[i].selected) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_boundary_projection_rebuild_edges(plan, &plan->functions[i]));
  }
  return loom_boundary_projection_replace_functions(plan);
}
