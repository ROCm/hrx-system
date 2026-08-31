// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/program_plan.h"

#include <string.h>

#include "loom/analysis/symbol_references.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/type_registry.h"
#include "loom/pass/builder.h"
#include "loom/pass/interpreter.h"
#include "loom/pass/program.h"
#include "loom/target/arch/cmd/lower/dispatch_counts.h"
#include "loom/target/arch/cmd/lower/lower.h"
#include "loom/target/arch/cmd/lower/parameters.h"
#include "loom/target/arch/cmd/lower/program_composition.h"
#include "loom/target/arch/cmd/lower/program_plan_requests.h"
#include "loom/target/arch/cmd/lower/schedule.h"
#include "loom/target/arch/cmd/lower/transients.h"
#include "loom/transforms/kernel/resolve_launches.h"
#include "loom/transforms/symbol/template_expansion_pipeline.h"
#include "loom/util/fact_table.h"

static iree_string_view_t loom_cmd_program_plan_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  IREE_ASSERT(loom_symbol_ref_is_valid(symbol_ref));
  IREE_ASSERT_EQ(symbol_ref.module_id, 0u);
  IREE_ASSERT_LT(symbol_ref.symbol_id, module->symbols.count);
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  IREE_ASSERT_LT(symbol->name_id, module->strings.count);
  return module->strings.entries[symbol->name_id];
}

typedef struct loom_cmd_program_root_build_t {
  // Mutable root operation in the preparation module.
  loom_op_t* program_op;
  // First caller-order root using |program_op|.
  uint32_t canonical_root_ordinal;
  // Function-like view of |program_op|.
  loom_func_like_t program;
  // Prepared issue schedule borrowing the preparation module.
  loom_cmd_schedule_plan_t schedule;
  // Scratch-backed physical count placement in schedule command order.
  const loom_cmd_dispatch_count_t* dispatch_counts;
  // Scratch class-specific requirement by schedule command, or NULL.
  const uint32_t* command_requirement_indices;
  // Scratch-backed facts consumed by closed command lowering.
  loom_cmd_lower_plan_t lower_plan;
  // Owned immutable parameter requirements transferred to the final root.
  loom_cmd_parameter_requirement_table_t parameters;
  // Aggregate transient storage requirement transferred to the final root.
  loom_cmd_transient_requirement_t transient;
  // Owned atomic entry requirements transferred to the final root.
  uint32_t* entry_requirement_indices;
  // Number of entries in |entry_requirement_indices|.
  uint32_t entry_requirement_count;
} loom_cmd_program_root_build_t;

// Scratch indexes used to assign plan-wide and root-local entry requirements.
typedef struct loom_cmd_program_symbol_index_t {
  // Root or entry ordinal indexed by preparation-module symbol ID. Command
  // definitions name their first caller-order root while kernel entry
  // declarations name their plan-wide requirement; the symbol kinds are
  // disjoint.
  uint32_t* ordinal_by_symbol;
  // Root-local executable/entry slot by plan requirement index.
  uint32_t* root_slot_by_requirement;
  // Root generation in which each slot mapping is valid.
  iree_host_size_t* root_slot_generations;
  // Active nonzero root generation.
  iree_host_size_t root_generation;
} loom_cmd_program_symbol_index_t;

static iree_status_t loom_cmd_program_plan_build_cleanup_body(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  loom_op_t* run_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_pass_ir_build_run(builder, 0, IREE_SV("canonicalize"),
                             loom_named_attr_slice_empty(), &run_op));
  return loom_pass_ir_build_run(builder, 0, IREE_SV("cse"),
                                loom_named_attr_slice_empty(), &run_op);
}

static iree_status_t loom_cmd_program_plan_build_unroll_body(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  loom_op_t* run_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_pass_ir_build_run(builder, 0, IREE_SV("unroll-scf-for"),
                             loom_named_attr_slice_empty(), &run_op));
  loom_op_t* if_changed_op = NULL;
  return loom_pass_ir_build_if_changed(
      builder, loom_cmd_program_plan_build_cleanup_body, NULL, &if_changed_op);
}

static iree_status_t loom_cmd_program_plan_build_post_inline_cleanup_body(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  loom_op_t* run_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_pass_ir_build_run(builder, 0, IREE_SV("canonicalize"),
                             loom_named_attr_slice_empty(), &run_op));
  IREE_RETURN_IF_ERROR(loom_pass_ir_build_run(
      builder, 0, IREE_SV("licm"), loom_named_attr_slice_empty(), &run_op));
  return loom_pass_ir_build_run(builder, 0, IREE_SV("cse"),
                                loom_named_attr_slice_empty(), &run_op);
}

static iree_status_t loom_cmd_program_plan_build_post_inline_body(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  loom_op_t* for_op = NULL;
  return loom_pass_ir_build_for(
      builder, LOOM_PASS_ANCHOR_FUNC,
      loom_cmd_program_plan_build_post_inline_cleanup_body, NULL, &for_op);
}

static iree_status_t loom_cmd_program_plan_build_preparation_body(
    loom_builder_t* builder, void* user_data) {
  const iree_host_size_t template_demand_count =
      *(const iree_host_size_t*)user_data;
  if (template_demand_count != 0) {
    IREE_RETURN_IF_ERROR(loom_template_expansion_pipeline_build(
        builder, loom_cmd_program_plan_build_post_inline_body, NULL));
  }
  // Minimize pure calls while loops and control flow are still compact. This
  // lets CSE reuse equal configuration calls before inlining expands them.
  loom_op_t* for_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_ir_build_for(
      builder, LOOM_PASS_ANCHOR_FUNC, loom_cmd_program_plan_build_cleanup_body,
      NULL, &for_op));
  loom_op_t* run_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_pass_ir_build_run(builder, 0, IREE_SV("inline-callables"),
                             loom_named_attr_slice_empty(), &run_op));
  loom_op_t* if_changed_op = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_ir_build_if_changed(
      builder, loom_cmd_program_plan_build_post_inline_body, NULL,
      &if_changed_op));
  // Expand loops only after each remaining syntactic call has been inlined.
  // Per-function cleanup runs only when unrolling changed that function.
  IREE_RETURN_IF_ERROR(loom_pass_ir_build_for(
      builder, LOOM_PASS_ANCHOR_FUNC, loom_cmd_program_plan_build_unroll_body,
      NULL, &for_op));
  return loom_pass_ir_build_run(builder, 0, IREE_SV("symbol-dce"),
                                loom_named_attr_slice_empty(), &run_op);
}

// Resolves caller-owned configuration dataflow before schedule construction.
// A single module run normalizes all selected callables, inlines the projected
// pure configuration functions at their command CFG sites, and cleans only
// functions changed by inlining. The body-blind selective product contains no
// device implementation IR.
static iree_status_t loom_cmd_program_plan_prepare_roots(
    loom_module_t* module, const loom_pass_registry_t* pass_registry,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, iree_host_size_t template_demand_count,
    bool* out_valid) {
  *out_valid = false;

  loom_module_t* pipeline_module = NULL;
  IREE_RETURN_IF_ERROR(loom_module_allocate(
      module->context, IREE_SV("__command_program_preparation"), block_pool,
      NULL, module->allocator, &pipeline_module));

  loom_op_t* pipeline_op = NULL;
  iree_status_t status = loom_pass_ir_build_pipeline(
      pipeline_module, IREE_SV("__command_program_preparation"),
      LOOM_PASS_ANCHOR_MODULE, loom_cmd_program_plan_build_preparation_body,
      &template_demand_count, &pipeline_op);

  const loom_pass_program_compile_options_t compile_options = {
      .registry = pass_registry,
  };
  loom_pass_program_t program = {0};
  if (iree_status_is_ok(status)) {
    status = loom_pass_program_compile_pipeline(
        pipeline_module, pipeline_op, &compile_options, block_pool, &program);
  }

  const loom_pass_interpreter_options_t interpreter_options = {
      .block_pool = block_pool,
      .diagnostic_emitter = diagnostic_emitter,
  };
  bool valid = false;
  if (iree_status_is_ok(status)) {
    loom_pass_run_result_t result = {0};
    status = loom_pass_interpreter_run_module(&program, module,
                                              &interpreter_options, &result);
    valid = result.error_count == 0;
  }

  loom_pass_program_deinitialize(&program);
  loom_module_free(pipeline_module);
  if (iree_status_is_ok(status)) *out_valid = valid;
  return status;
}

static iree_status_t loom_cmd_program_plan_allocate_tables(
    iree_host_size_t root_count, iree_host_size_t entry_requirement_capacity,
    loom_cmd_program_plan_t* plan) {
  if (root_count > IREE_HOST_SIZE_MAX / sizeof(*plan->roots) ||
      entry_requirement_capacity >
          IREE_HOST_SIZE_MAX / sizeof(*plan->entry_requirements)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command program plan tables are too large");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(plan->host_allocator,
                                             root_count * sizeof(*plan->roots),
                                             (void**)&plan->roots));
  memset(plan->roots, 0, root_count * sizeof(*plan->roots));
  plan->root_count = root_count;
  if (entry_requirement_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        plan->host_allocator,
        entry_requirement_capacity * sizeof(*plan->entry_requirements),
        (void**)&plan->entry_requirements));
  }
  return iree_ok_status();
}

static uint32_t loom_cmd_program_plan_declaration_requirement(
    loom_cmd_program_plan_t* plan,
    loom_cmd_program_symbol_index_t* symbol_index, loom_symbol_ref_t callee,
    const loom_op_t* declaration_op) {
  IREE_ASSERT(loom_symbol_ref_is_valid(callee));
  IREE_ASSERT_EQ(callee.module_id, 0u);
  uint32_t requirement_index =
      symbol_index->ordinal_by_symbol[callee.symbol_id];
  if (requirement_index != UINT32_MAX) return requirement_index;

  IREE_ASSERT_LT(plan->entry_requirement_count, UINT32_MAX);
  requirement_index = (uint32_t)plan->entry_requirement_count++;
  plan->entry_requirements[requirement_index] = (loom_cmd_entry_requirement_t){
      .declaration_op = declaration_op,
  };
  symbol_index->ordinal_by_symbol[callee.symbol_id] = requirement_index;
  return requirement_index;
}

static uint32_t loom_cmd_program_plan_root_entry_slot(
    loom_cmd_program_symbol_index_t* symbol_index, uint32_t requirement_index,
    uint32_t* requirement_indices, uint32_t* requirement_count) {
  if (symbol_index->root_slot_generations[requirement_index] !=
      symbol_index->root_generation) {
    IREE_ASSERT_LT(*requirement_count, UINT32_MAX);
    const uint32_t root_slot = (*requirement_count)++;
    symbol_index->root_slot_generations[requirement_index] =
        symbol_index->root_generation;
    symbol_index->root_slot_by_requirement[requirement_index] = root_slot;
    requirement_indices[root_slot] = requirement_index;
    return root_slot;
  }
  return symbol_index->root_slot_by_requirement[requirement_index];
}

static bool loom_cmd_program_plan_exact_scalar_bits(
    loom_type_t type, loom_value_facts_t facts,
    loom_cmd_lower_dispatch_argument_kind_t* out_kind, uint64_t* out_bits) {
  IREE_ASSERT(loom_type_is_scalar(type));
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  const int32_t bit_count = loom_scalar_type_bitwidth(scalar_type);
  if (bit_count <= 0 || bit_count > 64) return false;
  if (loom_scalar_type_is_float(scalar_type)) {
    if (!loom_value_facts_as_exact_float_bits(scalar_type, facts, out_bits)) {
      return false;
    }
  } else if (!loom_value_facts_as_exact_raw_bits(facts, bit_count, out_bits)) {
    return false;
  }

  if (bit_count <= 8) {
    *out_kind = LOOM_CMD_LOWER_DISPATCH_ARGUMENT_KIND_B8;
  } else if (bit_count <= 16) {
    *out_kind = LOOM_CMD_LOWER_DISPATCH_ARGUMENT_KIND_B16;
  } else if (bit_count <= 32) {
    *out_kind = LOOM_CMD_LOWER_DISPATCH_ARGUMENT_KIND_B32;
  } else {
    *out_kind = LOOM_CMD_LOWER_DISPATCH_ARGUMENT_KIND_B64;
  }
  return true;
}

typedef enum loom_cmd_program_argument_classification_e {
  LOOM_CMD_PROGRAM_ARGUMENT_CLASSIFICATION_OK = 0,
  LOOM_CMD_PROGRAM_ARGUMENT_CLASSIFICATION_NON_EXACT_SCALAR = 1,
  LOOM_CMD_PROGRAM_ARGUMENT_CLASSIFICATION_UNSUPPORTED_TYPE = 2,
} loom_cmd_program_argument_classification_t;

static loom_cmd_program_argument_classification_t
loom_cmd_program_plan_classify_argument(
    const loom_module_t* module, const loom_value_fact_table_t* facts,
    loom_value_id_t source_value,
    loom_cmd_lower_dispatch_argument_t* out_argument) {
  const loom_type_t type = loom_module_value_type(module, source_value);
  if (loom_type_is_buffer(type) || loom_type_is_view(type)) {
    *out_argument = (loom_cmd_lower_dispatch_argument_t){
        .kind = LOOM_CMD_LOWER_DISPATCH_ARGUMENT_KIND_BUFFER,
        .source_value = source_value,
    };
    return LOOM_CMD_PROGRAM_ARGUMENT_CLASSIFICATION_OK;
  }
  if (loom_type_is_scalar(type)) {
    loom_cmd_lower_dispatch_argument_kind_t kind = 0;
    uint64_t scalar_bits = 0;
    if (loom_cmd_program_plan_exact_scalar_bits(
            type, loom_value_fact_table_lookup(facts, source_value), &kind,
            &scalar_bits)) {
      *out_argument = (loom_cmd_lower_dispatch_argument_t){
          .kind = kind,
          .source_value = source_value,
          .scalar_bits = scalar_bits,
      };
      return LOOM_CMD_PROGRAM_ARGUMENT_CLASSIFICATION_OK;
    }
    return LOOM_CMD_PROGRAM_ARGUMENT_CLASSIFICATION_NON_EXACT_SCALAR;
  }
  return LOOM_CMD_PROGRAM_ARGUMENT_CLASSIFICATION_UNSUPPORTED_TYPE;
}

static iree_status_t loom_cmd_program_plan_emit_argument_error(
    const loom_module_t* module, const loom_cmd_schedule_command_t* command,
    uint32_t argument_index, loom_type_t type,
    loom_cmd_program_argument_classification_t classification,
    iree_diagnostic_emitter_t emitter) {
  IREE_ASSERT_NE(classification, LOOM_CMD_PROGRAM_ARGUMENT_CLASSIFICATION_OK);
  iree_string_view_t requirement = IREE_SV("a scalar, buffer, or view value");
  if (classification ==
      LOOM_CMD_PROGRAM_ARGUMENT_CLASSIFICATION_NON_EXACT_SCALAR) {
    requirement = IREE_SV("an exact scalar bit pattern");
  }
  const uint32_t flat_argument_offset = command->workgroup_counts.count;
  const loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_cmd_program_plan_symbol_name(module, command->callee)),
      loom_param_with_field_ref(
          loom_param_u32(argument_index),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND,
                                    flat_argument_offset + argument_index)),
      loom_param_type(type),
      loom_param_string(requirement),
  };
  const loom_diagnostic_emission_t emission = {
      .module = module,
      .op = command->source_op,
      .error = LOOM_ERR_LOWERING_052,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_cmd_program_plan_copy_u32_table(
    const uint32_t* source, uint32_t count, iree_allocator_t allocator,
    uint32_t** out_table) {
  *out_table = NULL;
  if (count == 0) return iree_ok_status();
  iree_host_size_t byte_length = 0;
  if (!iree_host_size_checked_mul(count, sizeof(**out_table), &byte_length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command root index table is too large");
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, byte_length, (void**)out_table));
  memcpy(*out_table, source, byte_length);
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_plan_build_lower_plan(
    loom_cmd_program_plan_t* plan, loom_module_t* preparation_module,
    loom_op_t* root_program_op, const loom_cmd_schedule_plan_t* schedule,
    const loom_value_fact_table_t* source_facts,
    const loom_cmd_dispatch_count_t* dispatch_counts,
    const uint32_t* command_requirement_indices,
    loom_cmd_program_symbol_index_t* symbol_index,
    iree_arena_allocator_t* scratch_arena,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_valid,
    loom_cmd_parameter_requirement_table_t* out_parameters,
    loom_cmd_transient_requirement_t* out_transient,
    loom_cmd_lower_plan_t* out_lower_plan,
    uint32_t** out_entry_requirement_indices,
    uint32_t* out_entry_requirement_count) {
  *out_valid = false;
  *out_entry_requirement_indices = NULL;
  *out_entry_requirement_count = 0;
  if (schedule->command_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command root has too many issue commands");
  }

  loom_cmd_lower_dispatch_t* dispatches = NULL;
  loom_cmd_lower_dispatch_argument_t* arguments = NULL;
  uint32_t* entry_requirement_indices = NULL;
  if (schedule->command_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, schedule->command_count,
                                  sizeof(*dispatches), (void**)&dispatches));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, schedule->command_count,
                                  sizeof(*entry_requirement_indices),
                                  (void**)&entry_requirement_indices));
  }
  if (schedule->argument_value_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, schedule->argument_value_count,
                                  sizeof(*arguments), (void**)&arguments));
  }

  uint32_t entry_requirement_count = 0;
  iree_host_size_t argument_offset = 0;
  for (iree_host_size_t i = 0; i < schedule->command_count; ++i) {
    const loom_cmd_schedule_command_t* command = &schedule->commands[i];
    const loom_value_id_t* source_arguments = command->arguments.values;
    const uint16_t source_argument_count = command->arguments.count;
    IREE_ASSERT(loom_symbol_ref_is_valid(command->callee));
    IREE_ASSERT_EQ(command->callee.module_id, 0u);
    IREE_ASSERT_LT(command->callee.symbol_id,
                   preparation_module->symbols.count);
    const loom_op_t* declaration_op =
        preparation_module->symbols.entries[command->callee.symbol_id]
            .defining_op;
    IREE_ASSERT(loom_kernel_entry_decl_isa(declaration_op));
    uint32_t requirement_index = UINT32_MAX;
    if (command_requirement_indices != NULL) {
      requirement_index = command_requirement_indices[i];
    }
    if (requirement_index == UINT32_MAX) {
      requirement_index = loom_cmd_program_plan_declaration_requirement(
          plan, symbol_index, command->callee, declaration_op);
    }

    const uint32_t root_entry_slot = loom_cmd_program_plan_root_entry_slot(
        symbol_index, requirement_index, entry_requirement_indices,
        &entry_requirement_count);

    loom_cmd_lower_dispatch_argument_t* dispatch_arguments =
        source_argument_count == 0 ? NULL : &arguments[argument_offset];
    uint32_t operand_value_count = 0;
    for (uint16_t argument_index = 0; argument_index < source_argument_count;
         ++argument_index) {
      const loom_value_id_t source_argument = source_arguments[argument_index];
      const loom_type_t source_type =
          loom_module_value_type(preparation_module, source_argument);
      const loom_cmd_program_argument_classification_t classification =
          loom_cmd_program_plan_classify_argument(
              preparation_module, source_facts, source_argument,
              &dispatch_arguments[argument_index]);
      if (classification != LOOM_CMD_PROGRAM_ARGUMENT_CLASSIFICATION_OK) {
        return loom_cmd_program_plan_emit_argument_error(
            preparation_module, command, argument_index, source_type,
            classification, diagnostic_emitter);
      }
      operand_value_count +=
          dispatch_arguments[argument_index].kind ==
                  LOOM_CMD_LOWER_DISPATCH_ARGUMENT_KIND_BUFFER
              ? 3
              : 1;
    }
    argument_offset += source_argument_count;
    dispatches[i] = (loom_cmd_lower_dispatch_t){
        .executable_index = root_entry_slot,
        .entry_index = root_entry_slot,
        .arguments = dispatch_arguments,
        .argument_count = source_argument_count,
        .operand_value_count = operand_value_count,
    };
  }
  IREE_ASSERT_EQ(argument_offset, schedule->argument_value_count);

  const loom_func_like_t root_program =
      loom_func_like_cast(preparation_module, root_program_op);
  uint16_t argument_count = 0;
  loom_func_like_arg_ids(root_program, &argument_count);
  const int64_t specialization_count_i64 =
      loom_func_like_specialization_count(root_program);
  IREE_ASSERT_GE(specialization_count_i64, 0);
  IREE_ASSERT_LE(specialization_count_i64, argument_count);
  const uint16_t binding_count =
      argument_count - (uint16_t)specialization_count_i64;

  loom_cmd_buffer_binding_t* bindings = NULL;
  if (binding_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, binding_count, sizeof(*bindings), (void**)&bindings));
  }
  loom_cmd_parameter_layout_t parameter_layout = {0};
  IREE_RETURN_IF_ERROR(loom_cmd_parameter_layout_build(
      preparation_module, root_program, source_facts, scratch_arena,
      plan->host_allocator, bindings, binding_count, out_parameters,
      &parameter_layout));
  loom_cmd_transient_layout_t transient_layout = {0};
  IREE_RETURN_IF_ERROR(loom_cmd_transient_layout_build(
      preparation_module, root_program, source_facts, schedule,
      parameter_layout.rebindable_binding_count, scratch_arena,
      &transient_layout));
  *out_transient = transient_layout.requirement;

  iree_host_size_t buffer_range_count = 0;
  if (!iree_host_size_checked_add(parameter_layout.buffer_range_count,
                                  transient_layout.buffer_range_count,
                                  &buffer_range_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command buffer range table is too large");
  }
  loom_cmd_buffer_range_t* buffer_ranges = NULL;
  if (buffer_range_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, buffer_range_count, sizeof(*buffer_ranges),
        (void**)&buffer_ranges));
    if (parameter_layout.buffer_range_count != 0) {
      memcpy(buffer_ranges, parameter_layout.buffer_ranges,
             parameter_layout.buffer_range_count * sizeof(*buffer_ranges));
    }
    if (transient_layout.buffer_range_count != 0) {
      memcpy(buffer_ranges + parameter_layout.buffer_range_count,
             transient_layout.buffer_ranges,
             transient_layout.buffer_range_count * sizeof(*buffer_ranges));
    }
  }

  uint32_t* owned_entry_requirement_indices = NULL;
  iree_status_t index_status = loom_cmd_program_plan_copy_u32_table(
      entry_requirement_indices, entry_requirement_count, plan->host_allocator,
      &owned_entry_requirement_indices);
  if (!iree_status_is_ok(index_status)) {
    return index_status;
  }

  const bool has_transient =
      transient_layout.requirement.binding_index != UINT32_MAX;
  *out_lower_plan = (loom_cmd_lower_plan_t){
      .bindings = bindings,
      .binding_count = binding_count,
      .buffer_ranges = buffer_ranges,
      .buffer_range_count = buffer_range_count,
      .abi_layout =
          {
              .fixed_buffer_count = parameter_layout.fixed_buffer_count,
              .rebindable_binding_count =
                  parameter_layout.rebindable_binding_count +
                  (has_transient ? 1u : 0u),
              .executable_count = entry_requirement_count,
              .entry_count = entry_requirement_count,
          },
      .schedule = schedule,
      .dispatch_counts = dispatch_counts,
      .dispatches = dispatches,
  };
  *out_entry_requirement_indices = owned_entry_requirement_indices;
  *out_entry_requirement_count = entry_requirement_count;
  *out_valid = true;
  return iree_ok_status();
}

iree_status_t loom_cmd_program_plan_prepare_materialization(
    loom_link_plan_materialization_t* materialization,
    const loom_symbol_ref_t* program_refs, iree_host_size_t program_count,
    const loom_cmd_program_kernel_source_t* kernel_source,
    const loom_pass_registry_t* pass_registry,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, bool* out_valid,
    loom_cmd_program_plan_t* out_plan, iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(materialization);
  IREE_ASSERT_ARGUMENT(materialization->module);
  IREE_ASSERT_ARGUMENT(program_refs);
  IREE_ASSERT_GT(program_count, 0u);
  IREE_ASSERT_ARGUMENT(pass_registry);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_valid);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_valid = false;
  memset(out_plan, 0, sizeof(*out_plan));

  loom_module_t* preparation_module = materialization->module;
  const loom_symbol_ref_t* configuration_functions =
      materialization->target_kernel_configurations.values;
  const iree_host_size_t configuration_function_count =
      materialization->target_kernel_configurations.count;
  *materialization = (loom_link_plan_materialization_t){0};

  loom_cmd_program_plan_t plan = {
      .host_allocator = host_allocator,
  };
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(block_pool, &scratch_arena);

  loom_cmd_program_root_build_t* root_builds = NULL;
  iree_status_t status =
      iree_arena_allocate_array(&scratch_arena, program_count,
                                sizeof(*root_builds), (void**)&root_builds);
  if (iree_status_is_ok(status)) {
    memset(root_builds, 0, program_count * sizeof(*root_builds));
  }
  loom_func_like_t* root_programs = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&scratch_arena, program_count,
                                       sizeof(*root_programs),
                                       (void**)&root_programs);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < program_count; ++i) {
      IREE_ASSERT(loom_symbol_ref_is_valid(program_refs[i]));
      IREE_ASSERT_EQ(program_refs[i].module_id, 0u);
      IREE_ASSERT_LT(program_refs[i].symbol_id,
                     preparation_module->symbols.count);
      loom_cmd_program_root_build_t* root = &root_builds[i];
      root->program_op =
          preparation_module->symbols.entries[program_refs[i].symbol_id]
              .defining_op;
      IREE_ASSERT(root->program_op != NULL);
      root->program = loom_func_like_cast(preparation_module, root->program_op);
      IREE_ASSERT(loom_func_like_isa(root->program));
      root_programs[i] = root->program;
    }
  }

  loom_symbol_reference_table_t references = {0};
  loom_kernel_launch_entry_table_t kernel_entry_table = {0};
  iree_host_size_t template_demand_count = 0;
  bool valid = false;
  if (iree_status_is_ok(status)) {
    status = loom_symbol_reference_table_build(preparation_module,
                                               &scratch_arena, &references);
  }
  if (iree_status_is_ok(status)) {
    template_demand_count = references.template_demands.count;
  }
  if (iree_status_is_ok(status)) {
    status = loom_kernel_resolve_launches(
        preparation_module, &references, configuration_functions,
        configuration_function_count, diagnostic_emitter, &scratch_arena,
        &kernel_entry_table, &valid);
  }
  if (valid && iree_status_is_ok(status)) {
    status = loom_cmd_program_composition_flatten(
        preparation_module, &references, root_programs, program_count,
        diagnostic_emitter, &scratch_arena, &valid);
  }
  if (valid && iree_status_is_ok(status)) {
    status = loom_cmd_program_plan_prepare_roots(
        preparation_module, pass_registry, diagnostic_emitter, block_pool,
        template_demand_count, &valid);
  }

  iree_host_size_t entry_requirement_capacity = 0;
  for (iree_host_size_t i = 0;
       valid && i < program_count && iree_status_is_ok(status); ++i) {
    loom_cmd_program_root_build_t* root = &root_builds[i];
    status = loom_cmd_schedule_plan_build(preparation_module,
                                          loom_func_like_body(root->program),
                                          &scratch_arena, &root->schedule);
    if (iree_status_is_ok(status)) {
      if (root->schedule.command_count >
          (iree_host_size_t)UINT32_MAX - entry_requirement_capacity) {
        status =
            iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "command program roots have too many issue sites");
      } else {
        entry_requirement_capacity += root->schedule.command_count;
      }
    }
  }

  loom_value_fact_table_t source_facts = {0};
  if (valid && iree_status_is_ok(status)) {
    status = loom_value_fact_table_initialize(&source_facts, &scratch_arena,
                                              preparation_module->values.count);
  }
  if (valid && iree_status_is_ok(status)) {
    loom_type_registry_configure_fact_context(&source_facts.context);
  }
  for (iree_host_size_t i = 0;
       valid && i < program_count && iree_status_is_ok(status); ++i) {
    status = loom_value_fact_table_compute(&source_facts, preparation_module,
                                           root_builds[i].program);
  }
  for (iree_host_size_t i = 0;
       valid && i < program_count && iree_status_is_ok(status); ++i) {
    loom_cmd_program_root_build_t* root = &root_builds[i];
    status = loom_cmd_dispatch_count_table_build(
        preparation_module, &root->schedule, &source_facts, diagnostic_emitter,
        &scratch_arena, &valid, &root->dispatch_counts);
  }

  loom_cmd_program_kernel_source_t prepared_kernel_source = {0};
  const loom_cmd_program_kernel_source_t* effective_kernel_source = NULL;
  if (valid && iree_status_is_ok(status) && kernel_source != NULL) {
    iree_host_size_t* prepared_source_definitions = NULL;
    status = iree_arena_allocate_array(&scratch_arena,
                                       preparation_module->symbols.count,
                                       sizeof(*prepared_source_definitions),
                                       (void**)&prepared_source_definitions);
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < preparation_module->symbols.count; ++i) {
        prepared_source_definitions[i] = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
      }
      IREE_ASSERT_LE(kernel_source->source_definitions.count,
                     kernel_entry_table.count);
      const iree_host_size_t source_symbol_count =
          kernel_source->source_definitions.count;
      for (iree_host_size_t source_symbol_id = 0;
           source_symbol_id < source_symbol_count; ++source_symbol_id) {
        loom_op_t* entry_op = kernel_entry_table.values[source_symbol_id];
        const iree_host_size_t source_definition =
            kernel_source->source_definitions.values[source_symbol_id];
        if (entry_op == NULL ||
            iree_any_bit_set(entry_op->flags, LOOM_OP_FLAG_DEAD) ||
            source_definition == LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
          continue;
        }
        const loom_symbol_ref_t entry_ref =
            loom_kernel_entry_decl_callee(entry_op);
        IREE_ASSERT(loom_symbol_ref_is_valid(entry_ref));
        IREE_ASSERT_EQ(entry_ref.module_id, 0u);
        IREE_ASSERT_LT(entry_ref.symbol_id, preparation_module->symbols.count);
        prepared_source_definitions[entry_ref.symbol_id] = source_definition;
      }
      prepared_kernel_source = *kernel_source;
      prepared_kernel_source.source_definitions.values =
          prepared_source_definitions;
      prepared_kernel_source.source_definitions.count =
          preparation_module->symbols.count;
      effective_kernel_source = &prepared_kernel_source;
    }
  }

  if (valid && iree_status_is_ok(status)) {
    status = loom_cmd_program_plan_allocate_tables(
        program_count, entry_requirement_capacity, &plan);
  }
  if (valid && iree_status_is_ok(status) && effective_kernel_source != NULL) {
    loom_cmd_program_kernel_site_root_t* kernel_site_roots = NULL;
    status = iree_arena_allocate_array(&scratch_arena, program_count,
                                       sizeof(*kernel_site_roots),
                                       (void**)&kernel_site_roots);
    for (iree_host_size_t i = 0; i < program_count && iree_status_is_ok(status);
         ++i) {
      kernel_site_roots[i].schedule = &root_builds[i].schedule;
      kernel_site_roots[i].requirement_indices = NULL;
    }
    if (iree_status_is_ok(status)) {
      status = loom_cmd_program_plan_publish_kernel_requests(
          &plan, preparation_module, &source_facts, effective_kernel_source,
          kernel_site_roots, program_count, &scratch_arena);
    }
    for (iree_host_size_t i = 0; i < program_count && iree_status_is_ok(status);
         ++i) {
      root_builds[i].command_requirement_indices =
          kernel_site_roots[i].requirement_indices;
    }
  }
  loom_cmd_program_symbol_index_t symbol_index = {0};
  if (valid && iree_status_is_ok(status) &&
      preparation_module->symbols.count > 0) {
    status = iree_arena_allocate_array(&scratch_arena,
                                       preparation_module->symbols.count,
                                       sizeof(*symbol_index.ordinal_by_symbol),
                                       (void**)&symbol_index.ordinal_by_symbol);
    if (iree_status_is_ok(status)) {
      memset(symbol_index.ordinal_by_symbol, 0xFF,
             preparation_module->symbols.count *
                 sizeof(*symbol_index.ordinal_by_symbol));
    }
  }
  for (iree_host_size_t i = 0;
       valid && i < program_count && iree_status_is_ok(status); ++i) {
    const loom_symbol_id_t symbol_id =
        loom_func_like_callee(root_builds[i].program).symbol_id;
    uint32_t canonical_root_ordinal = symbol_index.ordinal_by_symbol[symbol_id];
    if (canonical_root_ordinal == UINT32_MAX) {
      IREE_ASSERT_LT(i, UINT32_MAX);
      canonical_root_ordinal = (uint32_t)i;
      symbol_index.ordinal_by_symbol[symbol_id] = canonical_root_ordinal;
    }
    root_builds[i].canonical_root_ordinal = canonical_root_ordinal;
  }
  if (valid && iree_status_is_ok(status) && entry_requirement_capacity > 0) {
    status = iree_arena_allocate_array(
        &scratch_arena, entry_requirement_capacity,
        sizeof(*symbol_index.root_slot_by_requirement),
        (void**)&symbol_index.root_slot_by_requirement);
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate_array(
          &scratch_arena, entry_requirement_capacity,
          sizeof(*symbol_index.root_slot_generations),
          (void**)&symbol_index.root_slot_generations);
    }
    if (iree_status_is_ok(status)) {
      memset(symbol_index.root_slot_generations, 0,
             entry_requirement_capacity *
                 sizeof(*symbol_index.root_slot_generations));
    }
  }
  for (iree_host_size_t i = 0;
       valid && i < program_count && iree_status_is_ok(status); ++i) {
    loom_cmd_program_root_build_t* root = &root_builds[i];
    symbol_index.root_generation = i + 1;
    status = loom_cmd_program_plan_build_lower_plan(
        &plan, preparation_module, root->program_op, &root->schedule,
        &source_facts, root->dispatch_counts, root->command_requirement_indices,
        &symbol_index, &scratch_arena, diagnostic_emitter, &valid,
        &root->parameters, &root->transient, &root->lower_plan,
        &root->entry_requirement_indices, &root->entry_requirement_count);
  }
  for (iree_host_size_t i = 0;
       valid && i < program_count && iree_status_is_ok(status); ++i) {
    loom_cmd_program_root_build_t* root = &root_builds[i];
    if (root->canonical_root_ordinal != i) {
      IREE_ASSERT_LT(root->canonical_root_ordinal, i);
      root->program_op = root_builds[root->canonical_root_ordinal].program_op;
      continue;
    }
    loom_op_t* preparation_root_function = NULL;
    status = loom_cmd_lower_program_to_low(preparation_module, root->program_op,
                                           &root->lower_plan,
                                           &preparation_root_function);
    if (iree_status_is_ok(status)) {
      root->program_op = preparation_root_function;
    }
  }
  if (valid && iree_status_is_ok(status)) {
    plan.root_module = preparation_module;
    preparation_module = NULL;
    for (iree_host_size_t i = 0; i < program_count; ++i) {
      loom_cmd_program_root_t* root = &plan.roots[i];
      loom_cmd_program_root_build_t* build = &root_builds[i];
      root->function_op = build->program_op;
      root->abi_layout = build->lower_plan.abi_layout;
      root->entry_requirement_indices = build->entry_requirement_indices;
      root->entry_requirement_count = build->entry_requirement_count;
      root->parameters = build->parameters;
      root->transient = build->transient;
      root->launch_counts = (loom_cmd_program_launch_count_requirement_t){
          .binding_index = UINT32_MAX,
      };
      build->entry_requirement_indices = NULL;
      build->entry_requirement_count = 0;
      memset(&build->parameters, 0, sizeof(build->parameters));
    }
  }

  if (root_builds) {
    for (iree_host_size_t i = 0; i < program_count; ++i) {
      loom_cmd_parameter_requirement_table_deinitialize(
          &root_builds[i].parameters, host_allocator);
      iree_allocator_free(host_allocator,
                          root_builds[i].entry_requirement_indices);
    }
  }
  if (preparation_module) loom_module_free(preparation_module);
  iree_arena_deinitialize(&scratch_arena);
  if (iree_status_is_ok(status) && valid) {
    *out_valid = true;
    *out_plan = plan;
  } else {
    loom_cmd_program_plan_deinitialize(&plan);
  }
  return status;
}

void loom_cmd_program_plan_deinitialize(loom_cmd_program_plan_t* plan) {
  if (!plan) return;
  for (iree_host_size_t i = 0; i < plan->root_count; ++i) {
    loom_cmd_parameter_requirement_table_deinitialize(
        &plan->roots[i].parameters, plan->host_allocator);
    iree_allocator_free(plan->host_allocator,
                        plan->roots[i].entry_requirement_indices);
  }
  iree_allocator_free(plan->host_allocator, plan->entry_requirements);
  iree_allocator_free(plan->host_allocator, plan->roots);
  if (plan->root_module) loom_module_free(plan->root_module);
  memset(plan, 0, sizeof(*plan));
}
