// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/launch_graph.h"

#include <inttypes.h>
#include <string.h>

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
#include "loom/transforms/cleanup/dce.h"
#include "loom/transforms/symbol/inline_callables.h"

typedef struct loom_cmd_launch_producer_frame_t {
  // Source pure operation awaiting dependency materialization.
  const loom_op_t* op;
  // Next operand to inspect before cloning |op|.
  uint16_t next_operand;
} loom_cmd_launch_producer_frame_t;

// Shared source and target state for one aggregate launch program.
typedef struct loom_cmd_launch_program_build_t {
  // Immutable verified source module.
  const loom_module_t* source_module;
  // Plan-wide resolved kernel definitions.
  const loom_cmd_launch_definition_table_t* definitions;
  // Borrowed facts for source roots and kernel launch configurations.
  const loom_value_fact_table_t* source_facts;
  // Owned aggregate host module under construction.
  loom_module_t* module;
  // Scratch storage discarded after program construction.
  iree_arena_allocator_t* scratch_arena;
  // Target helper callee for each plan-wide kernel definition.
  loom_symbol_ref_t* helper_callees;
  // Maximum helper workload arity, used to reuse one root call operand array.
  uint16_t maximum_workload_count;
  // Reusable call operands sized to |maximum_workload_count|.
  loom_value_id_t* call_operands;
  // Reusable non-recursive pure-producer traversal state.
  struct {
    // Active frames in dependency traversal order.
    loom_cmd_launch_producer_frame_t* frames;
    // Number of active frames in |frames|.
    iree_host_size_t count;
    // Allocated entry capacity of |frames|.
    iree_host_size_t capacity;
  } producer_stack;
} loom_cmd_launch_program_build_t;

// Root-local state used while constructing one aggregate host function.
typedef struct loom_cmd_launch_root_build_t {
  // Shared launch program state.
  loom_cmd_launch_program_build_t* program;
  // Source request defining the root, schedule, and launch resolutions.
  const loom_cmd_launch_program_source_t* source;
  // Source command program being factored.
  loom_func_like_t source_program;
  // Aggregate host function under construction.
  loom_func_like_t host_function;
  // Builder positioned in the aggregate host function body.
  loom_builder_t builder;
  // Source program value to aggregate host value mapping.
  loom_ir_remap_t program_remap;
  // Flattened xyz result values before tuple compaction.
  loom_value_id_t* launch_result_values;
  // Launch placement rows owned by the shared module.
  loom_cmd_launch_count_t* launches;
  // Ordered wave rows owned by the shared module.
  loom_cmd_schedule_wave_t* waves;
} loom_cmd_launch_root_build_t;

static iree_string_view_t loom_cmd_launch_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  IREE_ASSERT(loom_symbol_ref_is_valid(symbol_ref));
  IREE_ASSERT_EQ(symbol_ref.module_id, 0u);
  IREE_ASSERT_LT(symbol_ref.symbol_id, module->symbols.count);
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  IREE_ASSERT_LT(symbol->name_id, module->strings.count);
  return module->strings.entries[symbol->name_id];
}

static iree_status_t loom_cmd_launch_copy_value_name(
    const loom_cmd_launch_program_build_t* program, loom_ir_remap_t* remap,
    loom_value_id_t source_value, loom_value_id_t target_value) {
  const loom_string_id_t source_name =
      loom_module_value(program->source_module, source_value)->name_id;
  if (source_name == LOOM_STRING_ID_INVALID) return iree_ok_status();
  loom_string_id_t target_name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_ir_remap_string_id(
      remap, source_name, /*allow_invalid=*/false, &target_name));
  return loom_module_set_value_name(program->module, target_value, target_name);
}

static bool loom_cmd_launch_predicate_values_are_mapped(
    const loom_ir_remap_t* remap, const loom_predicate_t* predicate) {
  for (uint8_t i = 0; i < predicate->arg_count; ++i) {
    if (predicate->arg_tags[i] != LOOM_PRED_ARG_VALUE) continue;
    loom_value_id_t ignored = LOOM_VALUE_ID_INVALID;
    if (!loom_ir_remap_try_lookup_value(
            remap, (loom_value_id_t)predicate->args[i], &ignored)) {
      return false;
    }
  }
  return true;
}

// Copies the subset of a source function's predicates whose SSA references
// are represented by |remap|. Root functions therefore retain specialization
// predicates, while helpers retain workload predicates.
static iree_status_t loom_cmd_launch_copy_mapped_predicates(
    const loom_cmd_launch_program_build_t* program,
    loom_func_like_t source_function, loom_ir_remap_t* remap,
    loom_op_t* target_function_op) {
  uint16_t source_predicate_count = 0;
  const loom_predicate_t* source_predicates =
      loom_func_like_predicates(source_function, &source_predicate_count);
  if (source_predicate_count == 0) return iree_ok_status();

  loom_predicate_t* selected_predicates = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      program->scratch_arena, source_predicate_count,
      sizeof(*selected_predicates), (void**)&selected_predicates));
  uint16_t selected_predicate_count = 0;
  for (uint16_t i = 0; i < source_predicate_count; ++i) {
    if (loom_cmd_launch_predicate_values_are_mapped(remap,
                                                    &source_predicates[i])) {
      selected_predicates[selected_predicate_count++] = source_predicates[i];
    }
  }
  if (selected_predicate_count == 0) return iree_ok_status();

  loom_predicate_t* target_predicates = NULL;
  IREE_RETURN_IF_ERROR(loom_ir_remap_predicate_list(remap, selected_predicates,
                                                    selected_predicate_count,
                                                    &target_predicates));
  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(loom_rewriter_initialize(&rewriter, program->module,
                                                program->scratch_arena));
  const iree_status_t status = loom_rewriter_set_attr(
      &rewriter, target_function_op, loom_func_def_predicates_ATTR_INDEX,
      loom_attr_predicate_list(target_predicates, selected_predicate_count));
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

// Maps function arguments before remapping their possibly value-dependent
// types. Function shells use none placeholders until this mapping is complete.
static iree_status_t loom_cmd_launch_configure_arguments(
    const loom_cmd_launch_program_build_t* program,
    const loom_value_id_t* source_arguments, uint16_t source_argument_count,
    loom_func_like_t target_function, loom_ir_remap_t* remap) {
  uint16_t target_argument_count = 0;
  const loom_value_id_t* target_arguments =
      loom_func_like_arg_ids(target_function, &target_argument_count);
  IREE_ASSERT_EQ(target_argument_count, source_argument_count);
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(
      remap, source_arguments, target_arguments, source_argument_count));
  for (uint16_t i = 0; i < source_argument_count; ++i) {
    loom_type_t target_type = {0};
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        remap,
        loom_module_value_type(program->source_module, source_arguments[i]),
        &target_type));
    IREE_RETURN_IF_ERROR(loom_module_set_value_type(
        program->module, target_arguments[i], target_type));
    IREE_RETURN_IF_ERROR(loom_cmd_launch_copy_value_name(
        program, remap, source_arguments[i], target_arguments[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_initialize_pass(
    const loom_pass_info_t* info, iree_diagnostic_emitter_t emitter,
    iree_arena_allocator_t* arena, loom_pass_t* out_pass) {
  *out_pass = (loom_pass_t){
      .info = info,
      .instance_arena = arena,
      .arena = arena,
      .diagnostic_emitter = emitter,
  };
  const loom_pass_statistic_layout_t* statistics = info->statistic_layout;
  if (!statistics || statistics->storage_size == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, statistics->storage_size,
                                           &out_pass->statistic_storage));
  memset(out_pass->statistic_storage, 0, statistics->storage_size);
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_fail_internal_diagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  (void)user_data;
  (void)emission;
  return iree_make_status(
      IREE_STATUS_INTERNAL,
      "generic inliner rejected a compiler-generated launch helper call");
}

static iree_status_t loom_cmd_launch_run_cse(
    loom_module_t* module, loom_func_like_t function,
    iree_arena_block_pool_t* block_pool, bool* out_changed) {
  *out_changed = false;
  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(block_pool, &pass_arena);
  loom_pass_t pass = {0};
  iree_status_t status = loom_cmd_launch_initialize_pass(
      loom_cse_pass_info(), (iree_diagnostic_emitter_t){0}, &pass_arena, &pass);
  if (iree_status_is_ok(status)) {
    status = loom_cse_run(&pass, module, function);
    *out_changed = pass.changed;
  }
  iree_arena_deinitialize(&pass_arena);
  return status;
}

static iree_status_t loom_cmd_launch_run_dce(
    loom_module_t* module, loom_func_like_t function,
    iree_arena_block_pool_t* block_pool) {
  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(block_pool, &pass_arena);
  loom_pass_t pass = {0};
  iree_status_t status = loom_cmd_launch_initialize_pass(
      loom_dce_pass_info(), (iree_diagnostic_emitter_t){0}, &pass_arena, &pass);
  if (iree_status_is_ok(status)) {
    status = loom_dce_run(&pass, module, function);
  }
  iree_arena_deinitialize(&pass_arena);
  return status;
}

static iree_status_t loom_cmd_launch_run_inliner(
    loom_module_t* module, iree_arena_block_pool_t* block_pool) {
  iree_arena_allocator_t pass_arena;
  iree_arena_initialize(block_pool, &pass_arena);
  const iree_diagnostic_emitter_t emitter = {
      .fn = loom_cmd_launch_fail_internal_diagnostic,
  };
  loom_pass_t pass = {0};
  iree_status_t status = loom_cmd_launch_initialize_pass(
      loom_inline_callables_pass_info(), emitter, &pass_arena, &pass);
  if (iree_status_is_ok(status)) {
    status = loom_inline_callables_run(&pass, module);
  }
  iree_arena_deinitialize(&pass_arena);
  return status;
}

// Clones one verified kernel configuration body into a target function body.
static iree_status_t loom_cmd_launch_clone_config_body(
    const loom_cmd_launch_program_build_t* program, loom_op_t* kernel_op,
    loom_builder_t* builder, loom_ir_remap_t* remap,
    loom_value_id_t* out_result_values) {
  const loom_region_t* config_region = loom_kernel_def_config(kernel_op);
  const loom_block_t* config_block =
      loom_region_const_entry_block(config_region);
  const loom_op_t* launch_config = loom_kernel_def_launch_config_op(kernel_op);

  const loom_op_t* config_op = NULL;
  loom_block_for_each_op(config_block, config_op) {
    if (config_op == launch_config) continue;
    if (config_op->region_count != 0) {
      const iree_string_view_t op_name =
          loom_op_name(program->source_module, config_op);
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "aggregate launch extraction does not support nested regions in "
          "kernel launch configuration operation `%.*s`",
          (int)op_name.size, op_name.data);
    }

    bool materialize_results = config_op->result_count != 0;
    const loom_value_id_t* source_results = loom_op_const_results(config_op);
    for (uint16_t i = 0; i < config_op->result_count; ++i) {
      const loom_value_facts_t facts = loom_value_fact_table_lookup(
          program->source_facts, source_results[i]);
      const loom_type_t type =
          loom_module_value_type(program->source_module, source_results[i]);
      materialize_results &=
          loom_value_facts_can_materialize_constant(facts, type);
    }
    if (!materialize_results) {
      loom_op_t* cloned_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_ir_clone_op(builder, config_op, remap, &cloned_op));
      continue;
    }

    loom_location_id_t target_location = LOOM_LOCATION_UNKNOWN;
    IREE_RETURN_IF_ERROR(loom_ir_remap_location_id(remap, config_op->location,
                                                   &target_location));
    for (uint16_t i = 0; i < config_op->result_count; ++i) {
      const loom_value_id_t source_result = source_results[i];
      loom_type_t target_type = {0};
      IREE_RETURN_IF_ERROR(loom_ir_remap_type(
          remap, loom_module_value_type(program->source_module, source_result),
          &target_type));
      loom_value_id_t target_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_constant_build(
          builder,
          loom_value_fact_table_lookup(program->source_facts, source_result),
          target_type, target_location, &target_result));
      IREE_RETURN_IF_ERROR(
          loom_ir_remap_map_value(remap, source_result, target_result));
      IREE_RETURN_IF_ERROR(loom_cmd_launch_copy_value_name(
          program, remap, source_result, target_result));
    }
  }

  for (uint8_t dimension = 0;
       dimension < LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT; ++dimension) {
    const loom_value_id_t source_count =
        loom_kernel_launch_config_workgroup_count_operand(
            launch_config, (loom_kernel_dimension_t)dimension);
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
        remap, source_count, &out_result_values[dimension]));
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_build_helper(
    loom_cmd_launch_program_build_t* program, uint32_t definition_ordinal) {
  loom_op_t* source_kernel = program->definitions->entries[definition_ordinal];
  loom_func_like_t source_function =
      loom_func_like_cast(program->source_module, source_kernel);
  const loom_value_slice_t source_arguments =
      loom_kernel_workload_arg_ids(program->source_module, source_kernel);
  program->maximum_workload_count =
      iree_max(program->maximum_workload_count, source_arguments.count);

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_initialize(program->source_module, program->module,
                               program->scratch_arena, NULL, &remap));
  loom_type_t* argument_types = NULL;
  if (source_arguments.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        program->scratch_arena, source_arguments.count, sizeof(*argument_types),
        (void**)&argument_types));
    for (uint16_t i = 0; i < source_arguments.count; ++i) {
      argument_types[i] = loom_type_none();
    }
  }
  loom_type_t result_types[LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(result_types); ++i) {
    result_types[i] = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  }

  const loom_symbol_ref_t source_callee =
      loom_func_like_callee(source_function);
  const iree_string_view_t source_name =
      loom_cmd_launch_symbol_name(program->source_module, source_callee);
  loom_string_id_t target_name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(program->module, source_name, &target_name));
  loom_symbol_id_t target_symbol = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_add_symbol(program->module, target_name, &target_symbol));
  const loom_symbol_ref_t target_callee = {
      .module_id = 0,
      .symbol_id = target_symbol,
  };

  loom_location_id_t target_location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_ir_remap_location_id(
      &remap, source_kernel->location, &target_location));
  loom_builder_t builder;
  loom_builder_initialize(program->module, &program->module->arena,
                          loom_module_block(program->module), &builder);
  loom_op_t* helper_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_def_build(
      &builder,
      LOOM_FUNC_DEF_BUILD_FLAG_HAS_PURITY |
          LOOM_FUNC_DEF_BUILD_FLAG_HAS_INLINE_POLICY,
      /*visibility=*/0, /*retain=*/0, /*cc=*/0, LOOM_FUNC_PURITY_PURE,
      /*temperature=*/0, LOOM_INLINE_POLICY_INLINE, loom_symbol_ref_null(),
      /*abi=*/0, loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), target_callee, argument_types,
      source_arguments.count, result_types, IREE_ARRAYSIZE(result_types),
      /*tied_results=*/NULL, /*tied_result_count=*/0,
      /*predicates=*/NULL, /*predicates_count=*/0, target_location,
      &helper_op));
  const loom_func_like_t helper =
      loom_func_like_cast(program->module, helper_op);
  IREE_RETURN_IF_ERROR(loom_cmd_launch_configure_arguments(
      program, source_arguments.values, source_arguments.count, helper,
      &remap));
  IREE_RETURN_IF_ERROR(loom_cmd_launch_copy_mapped_predicates(
      program, source_function, &remap, helper_op));

  loom_builder_enter_region(&builder, helper_op, loom_func_like_body(helper));
  loom_value_id_t result_values[LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT];
  IREE_RETURN_IF_ERROR(loom_cmd_launch_clone_config_body(
      program, source_kernel, &builder, &remap, result_values));
  loom_op_t* return_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_return_build(&builder, result_values,
                                              IREE_ARRAYSIZE(result_values),
                                              target_location, &return_op));
  program->helper_callees[definition_ordinal] = target_callee;
  return iree_ok_status();
}

static bool loom_cmd_launch_op_is_nested_in_program(
    const loom_cmd_launch_root_build_t* build, const loom_op_t* op) {
  for (const loom_op_t* ancestor = op ? op->parent_op : NULL; ancestor;
       ancestor = ancestor->parent_op) {
    if (ancestor == build->source_program.op) return true;
  }
  return false;
}

static iree_status_t loom_cmd_launch_reserve_producer_frame(
    loom_cmd_launch_root_build_t* build) {
  loom_cmd_launch_program_build_t* program = build->program;
  if (program->producer_stack.count < program->producer_stack.capacity) {
    return iree_ok_status();
  }
  return iree_arena_grow_array(program->scratch_arena,
                               program->producer_stack.count,
                               iree_max(program->producer_stack.count + 1, 16u),
                               sizeof(*program->producer_stack.frames),
                               &program->producer_stack.capacity,
                               (void**)&program->producer_stack.frames);
}

static iree_status_t loom_cmd_launch_push_producer(
    loom_cmd_launch_root_build_t* build, loom_value_id_t source_value) {
  const loom_module_t* source_module = build->program->source_module;
  IREE_ASSERT_LT(source_value, source_module->values.count);
  const loom_value_t* value = loom_module_value(source_module, source_value);
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
      loom_op_effective_traits(source_module, producer);
  if (!loom_cmd_launch_op_is_nested_in_program(build, producer) ||
      producer->region_count != 0 ||
      !iree_any_bit_set(traits, LOOM_TRAIT_PURE)) {
    const iree_string_view_t op_name = loom_op_name(source_module, producer);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "launch workload value %%%u is produced by `%.*s`, which cannot be "
        "placed in the aggregate host launch function",
        (unsigned)source_value, (int)op_name.size, op_name.data);
  }

  IREE_RETURN_IF_ERROR(loom_cmd_launch_reserve_producer_frame(build));
  build->program->producer_stack
      .frames[build->program->producer_stack.count++] =
      (loom_cmd_launch_producer_frame_t){
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
static iree_status_t loom_cmd_launch_resolve_program_value(
    loom_cmd_launch_root_build_t* build, loom_value_id_t source_value,
    loom_value_id_t* out_target_value) {
  *out_target_value = LOOM_VALUE_ID_INVALID;
  if (loom_ir_remap_try_lookup_value(&build->program_remap, source_value,
                                     out_target_value)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_cmd_launch_push_producer(build, source_value));
  while (build->program->producer_stack.count != 0) {
    loom_cmd_launch_producer_frame_t* frame =
        &build->program->producer_stack
             .frames[build->program->producer_stack.count - 1];

    bool pushed_dependency = false;
    const loom_value_id_t* operands = loom_op_const_operands(frame->op);
    while (frame->next_operand < frame->op->operand_count) {
      const loom_value_id_t operand = operands[frame->next_operand++];
      loom_value_id_t mapped_operand = LOOM_VALUE_ID_INVALID;
      if (loom_ir_remap_try_lookup_value(&build->program_remap, operand,
                                         &mapped_operand)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_cmd_launch_push_producer(build, operand));
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
    --build->program->producer_stack.count;
  }

  return loom_ir_remap_resolve_value(&build->program_remap, source_value,
                                     out_target_value);
}

static iree_status_t loom_cmd_launch_build_host_function(
    loom_cmd_launch_root_build_t* build) {
  uint16_t source_argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(build->source_program, &source_argument_count);
  const int64_t specialization_count_i64 =
      loom_func_like_specialization_count(build->source_program);
  IREE_ASSERT_GE(specialization_count_i64, 0);
  IREE_ASSERT_LE(specialization_count_i64, source_argument_count);
  const uint16_t specialization_count = (uint16_t)specialization_count_i64;
  const iree_host_size_t result_count =
      build->source->schedule->command_count *
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
        build->program->scratch_arena, specialization_count,
        sizeof(*argument_types), (void**)&argument_types));
    for (uint16_t i = 0; i < specialization_count; ++i) {
      argument_types[i] = loom_type_none();
    }
  }
  loom_type_t* result_types = NULL;
  if (result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->program->scratch_arena, result_count, sizeof(*result_types),
        (void**)&result_types));
    for (iree_host_size_t i = 0; i < result_count; ++i) {
      result_types[i] = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    }
  }

  const loom_symbol_ref_t source_callee =
      loom_func_like_callee(build->source_program);
  const iree_string_view_t source_name =
      loom_cmd_launch_symbol_name(build->program->source_module, source_callee);
  loom_string_id_t target_name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(build->program->module,
                                                 source_name, &target_name));
  loom_symbol_id_t target_symbol = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(build->program->module,
                                              target_name, &target_symbol));

  loom_location_id_t target_location = LOOM_LOCATION_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_ir_remap_location_id(
      &build->program_remap, build->source_program.op->location,
      &target_location));
  loom_func_def_build_flags_t build_flags =
      LOOM_FUNC_DEF_BUILD_FLAG_HAS_RETAIN | LOOM_FUNC_DEF_BUILD_FLAG_HAS_PURITY;
  uint8_t visibility = 0;
  if (loom_func_like_visibility(build->source_program) != 0) {
    build_flags |= LOOM_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY;
    visibility = LOOM_FUNC_VISIBILITY_PUBLIC;
  }

  loom_builder_initialize(
      build->program->module, &build->program->module->arena,
      loom_module_block(build->program->module), &build->builder);
  loom_op_t* host_function_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_def_build(
      &build->builder, build_flags, visibility, LOOM_FUNC_RETAIN_RETAIN,
      /*cc=*/0, LOOM_FUNC_PURITY_PURE, /*temperature=*/0,
      /*inline_policy=*/0, loom_symbol_ref_null(), /*abi=*/0,
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(),
      (loom_symbol_ref_t){.module_id = 0, .symbol_id = target_symbol},
      argument_types, specialization_count, result_types, result_count,
      /*tied_results=*/NULL, /*tied_result_count=*/0,
      /*predicates=*/NULL, /*predicates_count=*/0, target_location,
      &host_function_op));
  build->host_function =
      loom_func_like_cast(build->program->module, host_function_op);
  IREE_RETURN_IF_ERROR(loom_cmd_launch_configure_arguments(
      build->program, source_arguments, specialization_count,
      build->host_function, &build->program_remap));
  IREE_RETURN_IF_ERROR(loom_cmd_launch_copy_mapped_predicates(
      build->program, build->source_program, &build->program_remap,
      host_function_op));
  loom_builder_enter_region(&build->builder, host_function_op,
                            loom_func_like_body(build->host_function));
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_build_root_body(
    loom_cmd_launch_root_build_t* build) {
  const loom_cmd_schedule_plan_t* schedule = build->source->schedule;
  const iree_host_size_t result_count =
      schedule->command_count * LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT;
  if (result_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(build->program->scratch_arena, result_count,
                                  sizeof(*build->launch_result_values),
                                  (void**)&build->launch_result_values));
  }
  if (schedule->command_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &build->program->module->arena, schedule->command_count,
        sizeof(*build->launches), (void**)&build->launches));
    memset(build->launches, 0,
           schedule->command_count * sizeof(*build->launches));
  }
  if (schedule->wave_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &build->program->module->arena, schedule->wave_count,
        sizeof(*build->waves), (void**)&build->waves));
    memcpy(build->waves, schedule->waves,
           schedule->wave_count * sizeof(*build->waves));
  }

  const loom_type_t result_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  for (iree_host_size_t launch_index = 0;
       launch_index < schedule->command_count; ++launch_index) {
    const loom_op_t* launch_op = schedule->commands[launch_index];
    build->launches[launch_index].source_op = launch_op;
    const uint32_t definition_ordinal =
        build->source->resolution->definition_ordinals[launch_index];
    const loom_value_slice_t source_workloads =
        loom_kernel_launch_workloads(launch_op);

    for (uint16_t i = 0; i < source_workloads.count; ++i) {
      IREE_RETURN_IF_ERROR(loom_cmd_launch_resolve_program_value(
          build, source_workloads.values[i],
          &build->program->call_operands[i]));
    }

    loom_location_id_t target_location = LOOM_LOCATION_UNKNOWN;
    IREE_RETURN_IF_ERROR(loom_ir_remap_location_id(
        &build->program_remap, launch_op->location, &target_location));
    loom_op_t* call_op = NULL;
    IREE_RETURN_IF_ERROR(loom_func_call_build(
        &build->builder, LOOM_FUNC_CALL_BUILD_FLAG_HAS_PURITY,
        LOOM_FUNC_PURITY_PURE, /*temperature=*/0, /*inline_policy=*/0,
        build->program->helper_callees[definition_ordinal],
        build->program->call_operands, source_workloads.count, result_types,
        IREE_ARRAYSIZE(result_types),
        /*tied_results=*/NULL, /*tied_result_count=*/0, target_location,
        &call_op));
    memcpy(&build->launch_result_values
                [launch_index * LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT],
           loom_op_const_results(call_op),
           IREE_ARRAYSIZE(result_types) * sizeof(*build->launch_result_values));
  }

  loom_op_t* return_op = NULL;
  return loom_func_return_build(&build->builder, build->launch_result_values,
                                result_count, build->host_function.op->location,
                                &return_op);
}

static bool loom_cmd_launch_exact_u32(const loom_value_fact_table_t* facts,
                                      loom_value_id_t value_id,
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

static bool loom_cmd_launch_tuples_equal(const loom_value_id_t* left,
                                         const loom_value_id_t* right) {
  return left[0] == right[0] && left[1] == right[1] && left[2] == right[2];
}

static iree_status_t loom_cmd_launch_compact_results(
    loom_cmd_launch_root_build_t* build, const loom_value_fact_table_t* facts,
    uint32_t* out_host_tuple_count, bool* out_removed_results) {
  *out_host_tuple_count = 0;
  *out_removed_results = false;
  loom_block_t* host_block =
      loom_region_entry_block(loom_func_like_body(build->host_function));
  loom_op_t* old_return_op = host_block->last_op;
  IREE_ASSERT(loom_func_return_isa(old_return_op));
  const loom_value_slice_t return_values =
      loom_func_return_operands(old_return_op);
  const iree_host_size_t result_count =
      build->source->schedule->command_count *
      LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT;
  IREE_ASSERT_EQ(return_values.count, result_count);

  bool* remove_results = NULL;
  loom_value_id_t* unique_tuples = NULL;
  loom_value_id_t* kept_values = build->launch_result_values;
  if (result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->program->scratch_arena, result_count, sizeof(*remove_results),
        (void**)&remove_results));
    memset(remove_results, 1, result_count * sizeof(*remove_results));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->program->scratch_arena, result_count, sizeof(*unique_tuples),
        (void**)&unique_tuples));
  }

  uint32_t host_tuple_count = 0;
  for (iree_host_size_t launch_index = 0;
       launch_index < build->source->schedule->command_count; ++launch_index) {
    const loom_value_id_t* tuple =
        &return_values.values[launch_index *
                              LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT];
    loom_target_dispatch_workgroup_count_t direct = {0};
    uint32_t* direct_values[] = {&direct.x, &direct.y, &direct.z};
    bool all_exact = true;
    for (uint8_t dimension = 0;
         dimension < LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT;
         ++dimension) {
      bool is_exact = false;
      if (!loom_cmd_launch_exact_u32(facts, tuple[dimension],
                                     direct_values[dimension], &is_exact)) {
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
      if (loom_cmd_launch_tuples_equal(
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
        remove_results[launch_index *
                           LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT +
                       dimension] = false;
      }
      ++host_tuple_count;
    }
    build->launches[launch_index].kind = LOOM_CMD_LAUNCH_COUNT_KIND_HOST;
    build->launches[launch_index].payload.host_tuple_ordinal = tuple_ordinal;
  }

  iree_host_size_t kept_count = 0;
  for (iree_host_size_t i = 0; i < result_count; ++i) {
    if (!remove_results[i]) kept_values[kept_count++] = return_values.values[i];
  }
  IREE_ASSERT_EQ(kept_count, (iree_host_size_t)host_tuple_count *
                                 LOOM_CMD_PROGRAM_LAUNCH_COUNT_DIMENSION_COUNT);

  loom_builder_t builder;
  loom_builder_initialize(build->program->module,
                          &build->program->module->arena, host_block, &builder);
  loom_builder_set_before(&builder, old_return_op);
  loom_op_t* new_return_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_return_build(&builder, kept_values, kept_count,
                                              old_return_op->location,
                                              &new_return_op));
  IREE_RETURN_IF_ERROR(loom_op_erase(build->program->module, old_return_op));

  uint16_t removed_count = 0;
  IREE_RETURN_IF_ERROR(loom_op_remove_results(
      build->program->module, build->host_function.op, remove_results,
      build->program->scratch_arena, &removed_count));
  IREE_ASSERT_EQ(removed_count, result_count - kept_count);
  *out_host_tuple_count = host_tuple_count;
  *out_removed_results = removed_count != 0;
  return iree_ok_status();
}

static iree_status_t loom_cmd_launch_optimize_roots(
    loom_cmd_launch_root_build_t* root_builds, iree_host_size_t root_count,
    iree_arena_block_pool_t* block_pool, loom_cmd_launch_graph_t* out_graphs) {
  loom_module_t* module = root_builds[0].program->module;
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_cmd_launch_run_cse(
        module, root_builds[i].host_function, block_pool, &changed));
  }
  IREE_RETURN_IF_ERROR(loom_cmd_launch_run_inliner(module, block_pool));

  iree_arena_allocator_t* scratch_arena = root_builds[0].program->scratch_arena;
  loom_pass_value_fact_owner_t fact_owner;
  loom_pass_value_fact_owner_initialize(block_pool, &fact_owner);
  loom_canonicalizer_t canonicalizer;
  iree_status_t status = loom_canonicalizer_initialize(
      module, scratch_arena, &fact_owner, &canonicalizer);
  const bool canonicalizer_initialized = iree_status_is_ok(status);
  for (iree_host_size_t i = 0; i < root_count && iree_status_is_ok(status);
       ++i) {
    loom_cmd_launch_root_build_t* build = &root_builds[i];
    loom_canonicalizer_result_t canonicalize_result = {0};
    status = loom_canonicalizer_run_function(
        &canonicalizer, build->host_function,
        &(loom_canonicalizer_options_t){0}, &canonicalize_result);
    bool cse_changed = false;
    if (iree_status_is_ok(status)) {
      status = loom_cmd_launch_run_cse(module, build->host_function, block_pool,
                                       &cse_changed);
    }
    if (iree_status_is_ok(status) && cse_changed) {
      canonicalize_result = (loom_canonicalizer_result_t){0};
      status = loom_canonicalizer_run_function(
          &canonicalizer, build->host_function,
          &(loom_canonicalizer_options_t){0}, &canonicalize_result);
    }

    uint32_t host_tuple_count = 0;
    bool removed_results = false;
    if (iree_status_is_ok(status)) {
      const loom_value_fact_table_t* facts =
          loom_canonicalizer_fact_table(&canonicalizer);
      IREE_ASSERT(facts != NULL);
      status = loom_cmd_launch_compact_results(build, facts, &host_tuple_count,
                                               &removed_results);
    }
    if (iree_status_is_ok(status) && removed_results) {
      status =
          loom_cmd_launch_run_dce(module, build->host_function, block_pool);
    }
    if (iree_status_is_ok(status)) {
      out_graphs[i] = (loom_cmd_launch_graph_t){
          .host_function_op = build->host_function.op,
          .launches = build->launches,
          .launch_count = build->source->schedule->command_count,
          .waves = build->waves,
          .wave_count = build->source->schedule->wave_count,
          .host_tuple_count = host_tuple_count,
      };
    }
  }
  if (canonicalizer_initialized) {
    loom_canonicalizer_deinitialize(&canonicalizer);
  }
  loom_pass_value_fact_owner_deinitialize(&fact_owner);
  return status;
}

iree_status_t loom_cmd_launch_program_materialize(
    const loom_module_t* source_module,
    const loom_cmd_launch_program_source_t* sources,
    iree_host_size_t source_count, const loom_value_fact_table_t* source_facts,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** out_module, loom_cmd_launch_graph_t* out_graphs) {
  *out_module = NULL;
  memset(out_graphs, 0, source_count * sizeof(*out_graphs));

  loom_module_t* module = NULL;
  iree_status_t status = loom_module_allocate(
      source_module->context, IREE_SV("command_program_launch"), block_pool,
      NULL, allocator, &module);
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(block_pool, &scratch_arena);

  loom_cmd_launch_program_build_t program = {
      .source_module = source_module,
      .definitions = sources[0].resolution->definitions,
      .source_facts = source_facts,
      .module = module,
      .scratch_arena = &scratch_arena,
  };
  if (iree_status_is_ok(status) && program.definitions->count != 0) {
    status = iree_arena_allocate_array(
        &scratch_arena, program.definitions->count,
        sizeof(*program.helper_callees), (void**)&program.helper_callees);
  }
  for (uint32_t i = 0;
       i < program.definitions->count && iree_status_is_ok(status); ++i) {
    status = loom_cmd_launch_build_helper(&program, i);
  }
  if (iree_status_is_ok(status) && program.maximum_workload_count != 0) {
    status = iree_arena_allocate_array(
        &scratch_arena, program.maximum_workload_count,
        sizeof(*program.call_operands), (void**)&program.call_operands);
  }

  loom_cmd_launch_root_build_t* root_builds = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_arena_allocate_array(&scratch_arena, source_count,
                                  sizeof(*root_builds), (void**)&root_builds);
  }
  if (iree_status_is_ok(status)) {
    memset(root_builds, 0, source_count * sizeof(*root_builds));
  }
  for (iree_host_size_t i = 0; i < source_count && iree_status_is_ok(status);
       ++i) {
    loom_cmd_launch_root_build_t* build = &root_builds[i];
    *build = (loom_cmd_launch_root_build_t){
        .program = &program,
        .source = &sources[i],
        .source_program =
            loom_func_like_cast(source_module, sources[i].program_op),
    };
    status = loom_ir_remap_initialize(source_module, module, &scratch_arena,
                                      NULL, &build->program_remap);
    if (iree_status_is_ok(status)) {
      status = loom_cmd_launch_build_host_function(build);
    }
    if (iree_status_is_ok(status)) {
      status = loom_cmd_launch_build_root_body(build);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_cmd_launch_optimize_roots(root_builds, source_count,
                                            block_pool, out_graphs);
  }

  iree_arena_deinitialize(&scratch_arena);
  if (!iree_status_is_ok(status)) {
    if (module) loom_module_free(module);
    memset(out_graphs, 0, source_count * sizeof(*out_graphs));
    return status;
  }
  *out_module = module;
  return iree_ok_status();
}
