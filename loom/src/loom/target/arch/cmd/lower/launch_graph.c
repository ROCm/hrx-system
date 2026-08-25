// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/launch_graph.h"

#include <inttypes.h>
#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/special_values.h"
#include "loom/ops/type_registry.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/cleanup/canonicalize.h"
#include "loom/transforms/cleanup/cse.h"

typedef struct loom_cmd_launch_graph_producer_frame_t {
  // Source pure operation awaiting dependency materialization.
  const loom_op_t* op;
  // Next operand to inspect before cloning |op|.
  uint16_t next_operand;
} loom_cmd_launch_graph_producer_frame_t;

typedef struct loom_cmd_launch_graph_build_t {
  // Immutable verified source module.
  const loom_module_t* source_module;
  // Source command program being factored.
  loom_func_like_t source_program;
  // Existing command schedule defining launch order.
  const loom_cmd_schedule_plan_t* schedule;
  // Borrowed source facts populated by the owning program plan.
  const loom_value_fact_table_t* source_facts;
  // Destination for user-authored contract diagnostics.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // False after a source contract diagnostic has been emitted.
  bool valid;
  // Owned aggregate host module under construction.
  loom_module_t* module;
  // Aggregate host function under construction.
  loom_func_like_t host_function;
  // Builder positioned in the aggregate host function body.
  loom_builder_t builder;
  // Source program value to aggregate host value mapping.
  loom_ir_remap_t program_remap;
  // Scratch storage discarded after extraction.
  iree_arena_allocator_t* scratch_arena;
  // Non-recursive pure-producer traversal stack.
  loom_cmd_launch_graph_producer_frame_t* producer_frames;
  // Number of active producer frames.
  iree_host_size_t producer_frame_count;
  // Allocated producer frame capacity.
  iree_host_size_t producer_frame_capacity;
  // Flattened xyz result values before tuple compaction.
  loom_value_id_t* launch_result_values;
  // Launch placement rows owned by |module|.
  loom_cmd_launch_count_t* launches;
  // Ordered wave rows owned by |module|.
  loom_cmd_schedule_wave_t* waves;
} loom_cmd_launch_graph_build_t;

static iree_string_view_t loom_cmd_launch_graph_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  IREE_ASSERT(loom_symbol_ref_is_valid(symbol_ref));
  IREE_ASSERT_EQ(symbol_ref.module_id, 0u);
  IREE_ASSERT_LT(symbol_ref.symbol_id, module->symbols.count);
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  IREE_ASSERT_LT(symbol->name_id, module->strings.count);
  return module->strings.entries[symbol->name_id];
}

static loom_op_t* loom_cmd_launch_graph_resolve_kernel(
    const loom_module_t* module, const loom_cmd_schedule_command_t* command) {
  IREE_ASSERT_EQ(command->kind, LOOM_CMD_SCHEDULE_COMMAND_KIND_KERNEL_LAUNCH);
  const loom_symbol_ref_t callee = command->callee;
  IREE_ASSERT(loom_symbol_ref_is_valid(callee));
  IREE_ASSERT_EQ(callee.module_id, 0u);
  IREE_ASSERT_LT(callee.symbol_id, module->symbols.count);
  loom_op_t* kernel_op = module->symbols.entries[callee.symbol_id].defining_op;
  IREE_ASSERT(kernel_op != NULL);
  IREE_ASSERT(loom_kernel_def_isa(kernel_op));
  return kernel_op;
}

static bool loom_cmd_launch_graph_op_is_nested_in_program(
    const loom_cmd_launch_graph_build_t* build, const loom_op_t* op) {
  for (const loom_op_t* ancestor = op ? op->parent_op : NULL; ancestor;
       ancestor = ancestor->parent_op) {
    if (ancestor == build->source_program.op) return true;
  }
  return false;
}

static iree_status_t loom_cmd_launch_graph_reserve_producer_frame(
    loom_cmd_launch_graph_build_t* build) {
  if (build->producer_frame_count < build->producer_frame_capacity) {
    return iree_ok_status();
  }
  return iree_arena_grow_array(
      build->scratch_arena, build->producer_frame_count,
      iree_max(build->producer_frame_count + 1, 16u),
      sizeof(*build->producer_frames), &build->producer_frame_capacity,
      (void**)&build->producer_frames);
}

static iree_status_t loom_cmd_launch_graph_push_producer(
    loom_cmd_launch_graph_build_t* build, loom_value_id_t source_value) {
  IREE_ASSERT_LT(source_value, build->source_module->values.count);
  const loom_value_t* value =
      loom_module_value(build->source_module, source_value);
  if (loom_value_is_block_arg(value)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "launch workload value %%%u is an unmapped block argument; "
        "buffer-sourced and residual control-flow values require a device "
        "launch slice",
        (unsigned)source_value);
  }

  const loom_op_t* producer = loom_value_def_op(value);
  IREE_ASSERT(producer != NULL);
  const loom_trait_flags_t traits =
      loom_op_effective_traits(build->source_module, producer);
  if (!loom_cmd_launch_graph_op_is_nested_in_program(build, producer) ||
      producer->region_count != 0 ||
      !iree_any_bit_set(traits, LOOM_TRAIT_PURE)) {
    const iree_string_view_t op_name =
        loom_op_name(build->source_module, producer);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "launch workload value %%%u is produced by `%.*s`, which cannot be "
        "placed in the aggregate host launch function",
        (unsigned)source_value, (int)op_name.size, op_name.data);
  }

  IREE_RETURN_IF_ERROR(loom_cmd_launch_graph_reserve_producer_frame(build));
  build->producer_frames[build->producer_frame_count++] =
      (loom_cmd_launch_graph_producer_frame_t){
          .op = producer,
          .next_operand = 0,
      };
  return iree_ok_status();
}

// Materializes the pure producer closure for one source-program value.
//
// An explicit stack keeps launch extraction bounded for large scalar graphs.
// Cloning one operation maps all of its results, so later requests for another
// result of the same producer reuse the existing clone.
static iree_status_t loom_cmd_launch_graph_resolve_program_value(
    loom_cmd_launch_graph_build_t* build, loom_value_id_t source_value,
    loom_value_id_t* out_target_value) {
  *out_target_value = LOOM_VALUE_ID_INVALID;
  if (loom_ir_remap_try_lookup_value(&build->program_remap, source_value,
                                     out_target_value)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_cmd_launch_graph_push_producer(build, source_value));
  while (build->producer_frame_count != 0) {
    loom_cmd_launch_graph_producer_frame_t* frame =
        &build->producer_frames[build->producer_frame_count - 1];

    bool pushed_dependency = false;
    const loom_value_id_t* operands = loom_op_const_operands(frame->op);
    while (frame->next_operand < frame->op->operand_count) {
      const loom_value_id_t operand = operands[frame->next_operand++];
      loom_value_id_t mapped_operand = LOOM_VALUE_ID_INVALID;
      if (loom_ir_remap_try_lookup_value(&build->program_remap, operand,
                                         &mapped_operand)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_cmd_launch_graph_push_producer(build, operand));
      pushed_dependency = true;
      break;
    }
    if (pushed_dependency) continue;

    const loom_value_id_t first_result =
        frame->op->result_count == 0 ? LOOM_VALUE_ID_INVALID
                                     : loom_op_const_results(frame->op)[0];
    loom_value_id_t mapped_first_result = LOOM_VALUE_ID_INVALID;
    if (first_result == LOOM_VALUE_ID_INVALID ||
        !loom_ir_remap_try_lookup_value(&build->program_remap, first_result,
                                        &mapped_first_result)) {
      loom_op_t* cloned_op = NULL;
      IREE_RETURN_IF_ERROR(loom_ir_clone_op(&build->builder, frame->op,
                                            &build->program_remap, &cloned_op));
    }
    --build->producer_frame_count;
  }

  return loom_ir_remap_resolve_value(&build->program_remap, source_value,
                                     out_target_value);
}

static iree_status_t loom_cmd_launch_graph_copy_value_name(
    loom_cmd_launch_graph_build_t* build, loom_ir_remap_t* remap,
    loom_value_id_t source_value, loom_value_id_t target_value) {
  const loom_string_id_t source_name =
      loom_module_value(build->source_module, source_value)->name_id;
  if (source_name == LOOM_STRING_ID_INVALID) return iree_ok_status();
  loom_string_id_t target_name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_ir_remap_string_id(
      remap, source_name, /*allow_invalid=*/false, &target_name));
  return loom_module_set_value_name(build->module, target_value, target_name);
}

static iree_status_t loom_cmd_launch_graph_attach_program_predicates(
    loom_cmd_launch_graph_build_t* build) {
  uint16_t source_predicate_count = 0;
  const loom_predicate_t* source_predicates =
      loom_func_like_predicates(build->source_program, &source_predicate_count);
  if (source_predicate_count == 0) return iree_ok_status();

  loom_predicate_t* host_source_predicates = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      build->scratch_arena, source_predicate_count,
      sizeof(*host_source_predicates), (void**)&host_source_predicates));
  uint16_t host_predicate_count = 0;
  for (uint16_t i = 0; i < source_predicate_count; ++i) {
    const loom_predicate_t predicate = source_predicates[i];
    bool all_values_mapped = true;
    for (uint8_t arg_index = 0; arg_index < predicate.arg_count; ++arg_index) {
      if (predicate.arg_tags[arg_index] != LOOM_PRED_ARG_VALUE) continue;
      loom_value_id_t target_value = LOOM_VALUE_ID_INVALID;
      if (!loom_ir_remap_try_lookup_value(
              &build->program_remap, (loom_value_id_t)predicate.args[arg_index],
              &target_value)) {
        all_values_mapped = false;
        break;
      }
    }
    if (all_values_mapped) {
      host_source_predicates[host_predicate_count++] = predicate;
    }
  }
  if (host_predicate_count == 0) return iree_ok_status();

  loom_predicate_t* host_predicates = NULL;
  IREE_RETURN_IF_ERROR(loom_ir_remap_predicate_list(
      &build->program_remap, host_source_predicates, host_predicate_count,
      &host_predicates));
  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, build->module, build->scratch_arena));
  iree_status_t status = loom_rewriter_set_attr(
      &rewriter, build->host_function.op, loom_func_def_predicates_ATTR_INDEX,
      loom_attr_predicate_list(host_predicates, host_predicate_count));
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

static iree_status_t loom_cmd_launch_graph_build_host_function(
    loom_cmd_launch_graph_build_t* build) {
  uint16_t source_argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(build->source_program, &source_argument_count);
  const int64_t specialization_count_i64 =
      loom_func_like_specialization_count(build->source_program);
  IREE_ASSERT_GE(specialization_count_i64, 0);
  IREE_ASSERT_LE(specialization_count_i64, source_argument_count);
  const uint16_t specialization_count = (uint16_t)specialization_count_i64;

  const iree_host_size_t result_count =
      build->schedule->kernel_launch_count *
      LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT;
  if (result_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "aggregate launch function requires %" PRIhsz
                            " scalar results, exceeding the %u-result IR limit",
                            result_count, (unsigned)UINT16_MAX);
  }

  loom_type_t* argument_types = NULL;
  if (specialization_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->scratch_arena, specialization_count, sizeof(*argument_types),
        (void**)&argument_types));
    // Populate the target value map before remapping types because a later
    // argument type may reference an earlier specialization argument.
    for (uint16_t i = 0; i < specialization_count; ++i) {
      argument_types[i] = loom_type_none();
    }
  }
  loom_type_t* result_types = NULL;
  if (result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->scratch_arena, result_count, sizeof(*result_types),
        (void**)&result_types));
    for (iree_host_size_t i = 0; i < result_count; ++i) {
      result_types[i] = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    }
  }

  const loom_symbol_ref_t source_callee =
      loom_func_like_callee(build->source_program);
  const iree_string_view_t source_name =
      loom_cmd_launch_graph_symbol_name(build->source_module, source_callee);
  loom_string_id_t target_name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(build->module, source_name, &target_name));
  loom_symbol_id_t target_symbol = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_add_symbol(build->module, target_name, &target_symbol));

  loom_location_id_t target_location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_ir_remap_location_id(
      &build->program_remap, build->source_program.op->location,
      &target_location));
  loom_func_def_build_flags_t build_flags = LOOM_FUNC_DEF_BUILD_FLAG_HAS_PURITY;
  uint8_t visibility = 0;
  if (loom_func_like_visibility(build->source_program) != 0) {
    build_flags |= LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY;
    visibility = LOOM_FUNC_VISIBILITY_PUBLIC;
  }

  loom_builder_initialize(build->module, &build->module->arena,
                          loom_module_block(build->module), &build->builder);
  loom_op_t* host_function_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_def_build(
      &build->builder, build_flags, visibility, /*retain=*/0, /*cc=*/0,
      LOOM_FUNC_PURITY_PURE, /*temperature=*/0, /*inline_policy=*/0,
      loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
      LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(),
      (loom_symbol_ref_t){.module_id = 0, .symbol_id = target_symbol},
      argument_types, specialization_count, result_types, result_count,
      /*tied_results=*/NULL, /*tied_result_count=*/0,
      /*predicates=*/NULL, /*predicates_count=*/0, target_location,
      &host_function_op));
  build->host_function = loom_func_like_cast(build->module, host_function_op);
  IREE_ASSERT(loom_func_like_isa(build->host_function));

  uint16_t host_argument_count = 0;
  const loom_value_id_t* host_arguments =
      loom_func_like_arg_ids(build->host_function, &host_argument_count);
  IREE_ASSERT_EQ(host_argument_count, specialization_count);
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_map_values(&build->program_remap, source_arguments,
                               host_arguments, specialization_count));
  for (uint16_t i = 0; i < specialization_count; ++i) {
    loom_type_t host_type = {0};
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        &build->program_remap,
        loom_module_value_type(build->source_module, source_arguments[i]),
        &host_type));
    IREE_RETURN_IF_ERROR(loom_module_set_value_type(
        build->module, host_arguments[i], host_type));
    IREE_RETURN_IF_ERROR(loom_cmd_launch_graph_copy_value_name(
        build, &build->program_remap, source_arguments[i], host_arguments[i]));
  }
  IREE_RETURN_IF_ERROR(loom_cmd_launch_graph_attach_program_predicates(build));

  loom_builder_enter_region(&build->builder, host_function_op,
                            loom_func_like_body(build->host_function));
  return iree_ok_status();
}

static bool loom_cmd_launch_graph_exact_u32(
    const loom_value_fact_table_t* facts, loom_value_id_t value_id,
    uint32_t* out_value, bool* out_is_exact) {
  *out_value = 0;
  *out_is_exact = false;
  int64_t value = 0;
  if (!loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(facts, value_id), &value)) {
    return true;
  }
  *out_is_exact = true;
  if (value < 0 || value > UINT32_MAX) return false;
  *out_value = (uint32_t)value;
  return true;
}

static iree_status_t loom_cmd_launch_graph_emit_direct_count_error(
    loom_cmd_launch_graph_build_t* build,
    const loom_cmd_schedule_command_t* command, uint32_t dimension,
    iree_string_view_t requirement) {
  const iree_string_view_t kernel_name =
      loom_cmd_launch_graph_symbol_name(build->source_module, command->callee);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(kernel_name),
      loom_param_with_field_ref(
          loom_param_u32(dimension),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, dimension)),
      loom_param_string(requirement),
  };
  const loom_diagnostic_emission_t emission = {
      .module = build->source_module,
      .op = command->source_op,
      .error = LOOM_ERR_LOWERING_051,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  build->valid = false;
  return iree_diagnostic_emit(build->diagnostic_emitter, &emission);
}

static iree_status_t loom_cmd_launch_graph_record_direct_dispatch(
    loom_cmd_launch_graph_build_t* build,
    const loom_cmd_schedule_command_t* command,
    loom_cmd_launch_count_t* launch) {
  IREE_ASSERT_EQ(command->kind,
                 LOOM_CMD_SCHEDULE_COMMAND_KIND_KERNEL_DISPATCH_DIRECT);
  const loom_cmd_schedule_value_slice_t counts =
      command->count_inputs.workgroup_counts;
  IREE_ASSERT_GE(counts.count, 1u);
  IREE_ASSERT_LE(counts.count, 3u);
  loom_target_dispatch_workgroup_count_t direct = {
      .x = 1,
      .y = 1,
      .z = 1,
  };
  uint32_t* direct_values[] = {&direct.x, &direct.y, &direct.z};
  for (uint16_t i = 0; i < counts.count; ++i) {
    bool is_exact = false;
    if (!loom_cmd_launch_graph_exact_u32(build->source_facts, counts.values[i],
                                         direct_values[i], &is_exact)) {
      return loom_cmd_launch_graph_emit_direct_count_error(
          build, command, i, IREE_SV("within the unsigned 32-bit range"));
    }
    if (!is_exact) {
      return loom_cmd_launch_graph_emit_direct_count_error(
          build, command, i, IREE_SV("an exact unsigned 32-bit value"));
    }
  }
  launch->kind = LOOM_CMD_LAUNCH_COUNT_KIND_DIRECT;
  launch->payload.direct = direct;
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_graph_clone_config(
    loom_cmd_launch_graph_build_t* build,
    const loom_cmd_schedule_command_t* command,
    iree_host_size_t result_offset) {
  loom_op_t* kernel_op =
      loom_cmd_launch_graph_resolve_kernel(build->source_module, command);
  loom_region_t* config_region = loom_kernel_def_config(kernel_op);
  loom_block_t* config_block = loom_region_entry_block(config_region);
  const loom_op_t* launch_config = loom_kernel_def_launch_config_op(kernel_op);

  const loom_cmd_schedule_value_slice_t source_workloads =
      command->count_inputs.workloads;
  const loom_value_slice_t config_arguments =
      loom_kernel_workload_arg_ids(build->source_module, kernel_op);
  IREE_ASSERT_EQ(source_workloads.count, config_arguments.count);

  loom_ir_remap_t config_remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_initialize(build->source_module, build->module,
                               build->scratch_arena, NULL, &config_remap));
  for (uint16_t i = 0; i < source_workloads.count; ++i) {
    loom_value_id_t target_workload = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_cmd_launch_graph_resolve_program_value(
        build, source_workloads.values[i], &target_workload));
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
        &config_remap, config_arguments.values[i], target_workload));
  }

  const loom_op_t* config_op = NULL;
  loom_block_for_each_op(config_block, config_op) {
    if (config_op == launch_config) continue;
    if (config_op->region_count != 0) {
      const iree_string_view_t op_name =
          loom_op_name(build->source_module, config_op);
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "aggregate launch extraction does not support nested regions in "
          "kernel launch configuration operation `%.*s`",
          (int)op_name.size, op_name.data);
    }
    bool materialize_results = config_op->result_count != 0;
    const loom_value_id_t* source_results = loom_op_const_results(config_op);
    for (uint16_t i = 0; i < config_op->result_count; ++i) {
      const loom_value_facts_t facts =
          loom_value_fact_table_lookup(build->source_facts, source_results[i]);
      const loom_type_t type =
          loom_module_value_type(build->source_module, source_results[i]);
      materialize_results &=
          loom_value_facts_can_materialize_constant(facts, type);
    }
    if (!materialize_results) {
      loom_op_t* cloned_op = NULL;
      IREE_RETURN_IF_ERROR(loom_ir_clone_op(&build->builder, config_op,
                                            &config_remap, &cloned_op));
      continue;
    }

    loom_location_id_t target_location = LOOM_LOCATION_UNKNOWN;
    IREE_RETURN_IF_ERROR(loom_ir_remap_location_id(
        &config_remap, config_op->location, &target_location));
    for (uint16_t i = 0; i < config_op->result_count; ++i) {
      const loom_value_id_t source_result = source_results[i];
      loom_type_t target_type = {0};
      IREE_RETURN_IF_ERROR(loom_ir_remap_type(
          &config_remap,
          loom_module_value_type(build->source_module, source_result),
          &target_type));
      loom_value_id_t target_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_constant_build(
          &build->builder,
          loom_value_fact_table_lookup(build->source_facts, source_result),
          target_type, target_location, &target_result));
      IREE_RETURN_IF_ERROR(
          loom_ir_remap_map_value(&config_remap, source_result, target_result));
      IREE_RETURN_IF_ERROR(loom_cmd_launch_graph_copy_value_name(
          build, &config_remap, source_result, target_result));
    }
  }
  for (uint8_t dimension = 0;
       dimension < LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT; ++dimension) {
    const loom_value_id_t source_count =
        loom_kernel_launch_config_workgroup_count_operand(
            launch_config, (loom_kernel_dimension_t)dimension);
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
        &config_remap, source_count,
        &build->launch_result_values[result_offset + dimension]));
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_graph_build_body(
    loom_cmd_launch_graph_build_t* build) {
  const iree_host_size_t result_count =
      build->schedule->kernel_launch_count *
      LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT;
  if (result_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(build->scratch_arena, result_count,
                                  sizeof(*build->launch_result_values),
                                  (void**)&build->launch_result_values));
  }
  if (build->schedule->command_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &build->module->arena, build->schedule->command_count,
        sizeof(*build->launches), (void**)&build->launches));
  }
  if (build->schedule->wave_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &build->module->arena, build->schedule->wave_count,
        sizeof(*build->waves), (void**)&build->waves));
    memcpy(build->waves, build->schedule->waves,
           build->schedule->wave_count * sizeof(*build->waves));
  }

  iree_host_size_t result_offset = 0;
  for (iree_host_size_t i = 0; i < build->schedule->command_count; ++i) {
    const loom_cmd_schedule_command_t* command = &build->schedule->commands[i];
    loom_cmd_launch_count_t* launch = &build->launches[i];
    launch->source_op = command->source_op;
    switch (command->kind) {
      case LOOM_CMD_SCHEDULE_COMMAND_KIND_KERNEL_LAUNCH: {
        IREE_RETURN_IF_ERROR(
            loom_cmd_launch_graph_clone_config(build, command, result_offset));
        result_offset += LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT;
        break;
      }
      case LOOM_CMD_SCHEDULE_COMMAND_KIND_KERNEL_DISPATCH_DIRECT: {
        IREE_RETURN_IF_ERROR(loom_cmd_launch_graph_record_direct_dispatch(
            build, command, launch));
        if (!build->valid) return iree_ok_status();
        break;
      }
      case LOOM_CMD_SCHEDULE_COMMAND_KIND_KERNEL_DISPATCH_INDIRECT: {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "indirect kernel dispatch requires count-origin placement");
      }
      default:
        IREE_ASSERT_UNREACHABLE("schedule command kind is valid");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
  IREE_ASSERT_EQ(result_offset, result_count);

  loom_op_t* return_op = NULL;
  return loom_func_return_build(&build->builder, build->launch_result_values,
                                result_count, build->host_function.op->location,
                                &return_op);
}

static iree_status_t loom_cmd_launch_graph_run_cse(
    loom_module_t* module, loom_func_like_t function,
    iree_arena_block_pool_t* block_pool) {
  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(block_pool, &pass_arena);
  loom_pass_t pass = {0};
  pass.info = loom_cse_pass_info();
  pass.instance_arena = &pass_arena;
  pass.arena = &pass_arena;
  const loom_pass_statistic_layout_t* statistics = pass.info->statistic_layout;
  iree_status_t status = iree_ok_status();
  if (statistics && statistics->storage_size != 0) {
    status = iree_arena_allocate(&pass_arena, statistics->storage_size,
                                 &pass.statistic_storage);
    if (iree_status_is_ok(status)) {
      memset(pass.statistic_storage, 0, statistics->storage_size);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_cse_run(&pass, module, function);
  }
  iree_arena_deinitialize(&pass_arena);
  return status;
}

static bool loom_cmd_launch_graph_tuples_equal(const loom_value_id_t* left,
                                               const loom_value_id_t* right) {
  return left[0] == right[0] && left[1] == right[1] && left[2] == right[2];
}

static iree_status_t loom_cmd_launch_graph_compact_results(
    loom_cmd_launch_graph_build_t* build, const loom_value_fact_table_t* facts,
    uint32_t* out_host_tuple_count) {
  *out_host_tuple_count = 0;
  loom_block_t* host_block =
      loom_region_entry_block(loom_func_like_body(build->host_function));
  loom_op_t* old_return_op = host_block->last_op;
  IREE_ASSERT(loom_func_return_isa(old_return_op));
  const loom_value_slice_t return_values =
      loom_func_return_operands(old_return_op);
  const iree_host_size_t result_count =
      build->schedule->kernel_launch_count *
      LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT;
  IREE_ASSERT_EQ(return_values.count, result_count);

  bool* remove_results = NULL;
  loom_value_id_t* unique_tuples = NULL;
  loom_value_id_t* kept_values = NULL;
  if (result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->scratch_arena, result_count, sizeof(*remove_results),
        (void**)&remove_results));
    memset(remove_results, 1, result_count * sizeof(*remove_results));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->scratch_arena, result_count, sizeof(*unique_tuples),
        (void**)&unique_tuples));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(build->scratch_arena, result_count,
                                  sizeof(*kept_values), (void**)&kept_values));
  }

  uint32_t host_tuple_count = 0;
  iree_host_size_t result_offset = 0;
  for (iree_host_size_t launch_index = 0;
       launch_index < build->schedule->command_count; ++launch_index) {
    if (build->schedule->commands[launch_index].kind !=
        LOOM_CMD_SCHEDULE_COMMAND_KIND_KERNEL_LAUNCH) {
      continue;
    }
    const iree_host_size_t tuple_offset = result_offset;
    result_offset += LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT;
    const loom_value_id_t* tuple = &return_values.values[tuple_offset];
    loom_target_dispatch_workgroup_count_t direct = {0};
    uint32_t* direct_values[] = {&direct.x, &direct.y, &direct.z};
    bool all_exact = true;
    for (uint8_t dimension = 0;
         dimension < LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT;
         ++dimension) {
      bool is_exact = false;
      if (!loom_cmd_launch_graph_exact_u32(
              facts, tuple[dimension], direct_values[dimension], &is_exact)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "launch %" PRIhsz
                                " workgroup-count dimension %u is outside u32",
                                launch_index, (unsigned)dimension);
      }
      all_exact &= is_exact;
    }
    if (all_exact) {
      build->launches[launch_index].kind = LOOM_CMD_LAUNCH_COUNT_KIND_DIRECT;
      build->launches[launch_index].payload.direct = direct;
      continue;
    }

    uint32_t tuple_ordinal = 0;
    for (; tuple_ordinal < host_tuple_count; ++tuple_ordinal) {
      if (loom_cmd_launch_graph_tuples_equal(
              tuple,
              &unique_tuples[(iree_host_size_t)tuple_ordinal *
                             LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT])) {
        break;
      }
    }
    if (tuple_ordinal == host_tuple_count) {
      if (host_tuple_count == UINT32_MAX) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "host launch tuple count exceeds u32");
      }
      memcpy(&unique_tuples[(iree_host_size_t)host_tuple_count *
                            LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT],
             tuple,
             LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT * sizeof(*tuple));
      for (uint8_t dimension = 0;
           dimension < LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT;
           ++dimension) {
        remove_results[tuple_offset + dimension] = false;
      }
      ++host_tuple_count;
    }
    build->launches[launch_index].kind = LOOM_CMD_LAUNCH_COUNT_KIND_HOST;
    build->launches[launch_index].payload.host_tuple_ordinal = tuple_ordinal;
  }
  IREE_ASSERT_EQ(result_offset, result_count);

  iree_host_size_t kept_count = 0;
  for (iree_host_size_t i = 0; i < result_count; ++i) {
    if (!remove_results[i]) kept_values[kept_count++] = return_values.values[i];
  }
  IREE_ASSERT_EQ(kept_count, (iree_host_size_t)host_tuple_count *
                                 LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT);

  loom_builder_t builder;
  loom_builder_initialize(build->module, &build->module->arena, host_block,
                          &builder);
  loom_builder_set_before(&builder, old_return_op);
  loom_op_t* new_return_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_return_build(&builder, kept_values, kept_count,
                                              old_return_op->location,
                                              &new_return_op));
  IREE_RETURN_IF_ERROR(loom_op_erase(build->module, old_return_op));

  uint16_t removed_count = 0;
  IREE_RETURN_IF_ERROR(loom_op_remove_results(
      build->module, build->host_function.op, remove_results,
      build->scratch_arena, &removed_count));
  IREE_ASSERT_EQ(removed_count, result_count - kept_count);
  *out_host_tuple_count = host_tuple_count;
  return iree_ok_status();
}

iree_status_t loom_cmd_launch_graph_materialize(
    const loom_module_t* source_module, loom_op_t* source_program_op,
    const loom_cmd_schedule_plan_t* schedule,
    const loom_value_fact_table_t* source_facts,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    bool* out_valid, loom_cmd_launch_graph_t* out_graph) {
  IREE_ASSERT_ARGUMENT(source_module);
  IREE_ASSERT_ARGUMENT(source_program_op);
  IREE_ASSERT_ARGUMENT(schedule);
  IREE_ASSERT_ARGUMENT(source_facts);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_valid);
  IREE_ASSERT_ARGUMENT(out_graph);
  *out_valid = false;
  memset(out_graph, 0, sizeof(*out_graph));

  loom_module_t* graph_module = NULL;
  iree_status_t status = loom_module_allocate(
      source_module->context, IREE_SV("command_launch_graph"), block_pool, NULL,
      allocator, &graph_module);
  iree_arena_allocator_t scratch_arena;
  bool scratch_arena_initialized = false;
  loom_pass_value_fact_owner_t fact_owner;
  bool fact_owner_initialized = false;
  loom_canonicalizer_t canonicalizer;
  bool canonicalizer_initialized = false;

  loom_cmd_launch_graph_build_t build = {
      .source_module = source_module,
      .source_program = loom_func_like_cast(source_module, source_program_op),
      .schedule = schedule,
      .source_facts = source_facts,
      .diagnostic_emitter = diagnostic_emitter,
      .valid = true,
      .module = graph_module,
  };
  const bool has_logical_launches = schedule->kernel_launch_count != 0;
  IREE_ASSERT(loom_func_like_isa(build.source_program));
  if (iree_status_is_ok(status)) {
    iree_arena_initialize(block_pool, &scratch_arena);
    scratch_arena_initialized = true;
    build.scratch_arena = &scratch_arena;
    status =
        loom_ir_remap_initialize(source_module, graph_module, &scratch_arena,
                                 NULL, &build.program_remap);
  }
  if (iree_status_is_ok(status)) {
    status = loom_cmd_launch_graph_build_host_function(&build);
  }
  if (iree_status_is_ok(status)) {
    status = loom_cmd_launch_graph_build_body(&build);
  }
  if (has_logical_launches && build.valid && iree_status_is_ok(status)) {
    loom_pass_value_fact_owner_initialize(block_pool, &fact_owner);
    fact_owner_initialized = true;
    status = loom_canonicalizer_initialize(graph_module, &scratch_arena,
                                           &fact_owner, &canonicalizer);
    canonicalizer_initialized = iree_status_is_ok(status);
  }
  if (has_logical_launches && build.valid && iree_status_is_ok(status)) {
    loom_canonicalizer_result_t result = {0};
    status = loom_canonicalizer_run_function(
        &canonicalizer, build.host_function, &(loom_canonicalizer_options_t){0},
        &result);
  }
  if (has_logical_launches && build.valid && iree_status_is_ok(status)) {
    status = loom_cmd_launch_graph_run_cse(graph_module, build.host_function,
                                           block_pool);
  }
  if (has_logical_launches && build.valid && iree_status_is_ok(status)) {
    loom_canonicalizer_result_t result = {0};
    status = loom_canonicalizer_run_function(
        &canonicalizer, build.host_function, &(loom_canonicalizer_options_t){0},
        &result);
  }

  uint32_t host_tuple_count = 0;
  if (has_logical_launches && build.valid && iree_status_is_ok(status)) {
    const loom_value_fact_table_t* facts =
        loom_canonicalizer_fact_table(&canonicalizer);
    IREE_ASSERT(facts != NULL);
    status =
        loom_cmd_launch_graph_compact_results(&build, facts, &host_tuple_count);
  }
  if (has_logical_launches && build.valid && iree_status_is_ok(status)) {
    loom_canonicalizer_result_t result = {0};
    status = loom_canonicalizer_run_function(
        &canonicalizer, build.host_function, &(loom_canonicalizer_options_t){0},
        &result);
  }

  if (canonicalizer_initialized) {
    loom_canonicalizer_deinitialize(&canonicalizer);
  }
  if (fact_owner_initialized) {
    loom_pass_value_fact_owner_deinitialize(&fact_owner);
  }
  if (scratch_arena_initialized) {
    iree_arena_deinitialize(&scratch_arena);
  }
  if (!iree_status_is_ok(status) || !build.valid) {
    if (graph_module) loom_module_free(graph_module);
    memset(out_graph, 0, sizeof(*out_graph));
    return status;
  }

  *out_graph = (loom_cmd_launch_graph_t){
      .module = graph_module,
      .host_function_op = build.host_function.op,
      .launches = build.launches,
      .launch_count = schedule->command_count,
      .waves = build.waves,
      .wave_count = schedule->wave_count,
      .host_tuple_count = host_tuple_count,
  };
  *out_valid = true;
  return iree_ok_status();
}

void loom_cmd_launch_graph_deinitialize(loom_cmd_launch_graph_t* graph) {
  if (!graph) return;
  if (graph->module) loom_module_free(graph->module);
  memset(graph, 0, sizeof(*graph));
}
