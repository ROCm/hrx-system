// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/program_plan.h"

#include <string.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/link/linker.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/type_registry.h"
#include "loom/pass/builder.h"
#include "loom/pass/interpreter.h"
#include "loom/pass/program.h"
#include "loom/target/arch/cmd/lower/lower.h"
#include "loom/target/arch/cmd/lower/parameters.h"
#include "loom/target/arch/cmd/lower/program_composition.h"
#include "loom/target/arch/cmd/lower/schedule.h"
#include "loom/target/arch/cmd/lower/transients.h"
#include "loom/target/function_contract.h"
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

static loom_op_t* loom_cmd_program_plan_find_symbol(
    loom_module_t* module, iree_string_view_t symbol_name) {
  const loom_string_id_t name_id =
      loom_module_lookup_string(module, symbol_name);
  IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
  const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
  IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
  loom_op_t* defining_op = module->symbols.entries[symbol_id].defining_op;
  IREE_ASSERT(defining_op != NULL);
  return defining_op;
}

typedef struct loom_cmd_program_root_build_t {
  iree_string_view_t name;
  iree_string_view_t launch_name;
  loom_op_t* program_op;
  loom_func_like_t program;
  loom_cmd_schedule_plan_t schedule;
  loom_cmd_launch_graph_t launch_graph;
  loom_cmd_lower_plan_t lower_plan;
  loom_cmd_lower_launch_t* lower_launches;
  loom_cmd_parameter_requirement_table_t parameters;
  loom_cmd_transient_requirement_t transient;
  uint32_t* entry_indices;
  uint32_t entry_count;
  uint32_t* executable_unit_indices;
  uint32_t executable_count;
  loom_cmd_program_entry_t* entries;
} loom_cmd_program_root_build_t;

static iree_status_t loom_cmd_program_plan_build_root_preparation_body(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  loom_op_t* run_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_pass_ir_build_run(builder, 0, IREE_SV("canonicalize"),
                             loom_named_attr_slice_empty(), &run_op));
  IREE_RETURN_IF_ERROR(
      loom_pass_ir_build_run(builder, 0, IREE_SV("unroll-scf-for"),
                             loom_named_attr_slice_empty(), &run_op));
  return loom_pass_ir_build_run(builder, 0, IREE_SV("canonicalize"),
                                loom_named_attr_slice_empty(), &run_op);
}

// Resolves source structure that must be static before schedule construction.
//
// The shared expanded-source pipeline intentionally preserves local loop
// policies for target kernels that may acquire facts later. Command roots have
// a stronger boundary: scheduling must see only a closed sequence of launches
// and explicit scheduling regions. Run a function-root preparation program on
// the selected command roots so dependency kernel bodies retain their ordinary
// target-compilation path.
static iree_status_t loom_cmd_program_plan_prepare_roots(
    loom_module_t* module, const loom_func_like_t* root_programs,
    iree_host_size_t root_program_count,
    const loom_pass_registry_t* pass_registry,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, bool* out_valid) {
  *out_valid = false;

  loom_module_t* pipeline_module = NULL;
  IREE_RETURN_IF_ERROR(loom_module_allocate(
      module->context, IREE_SV("__command_program_preparation"), block_pool,
      NULL, module->allocator, &pipeline_module));

  loom_op_t* pipeline_op = NULL;
  iree_status_t status = loom_pass_ir_build_pipeline(
      pipeline_module, IREE_SV("__command_program_preparation"),
      LOOM_PASS_ANCHOR_FUNC, loom_cmd_program_plan_build_root_preparation_body,
      NULL, &pipeline_op);

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
  bool valid = true;
  for (iree_host_size_t i = 0;
       i < root_program_count && iree_status_is_ok(status) && valid; ++i) {
    loom_pass_run_result_t result = {0};
    status = loom_pass_interpreter_run_function(
        &program, module, root_programs[i], &interpreter_options, &result);
    valid = result.error_count == 0;
  }

  loom_pass_program_deinitialize(&program);
  loom_module_free(pipeline_module);
  if (iree_status_is_ok(status)) *out_valid = valid;
  return status;
}

static iree_status_t loom_cmd_program_plan_resolve_function_target_facts(
    const loom_module_t* module, loom_func_like_t function,
    loom_symbol_fact_table_t* symbol_facts,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    bool* out_valid, const loom_target_facts_t** out_target_facts) {
  *out_valid = true;
  *out_target_facts = NULL;
  const loom_symbol_ref_t function_ref = loom_func_like_callee(function);
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      symbol_facts, module, function_ref, &base_facts));
  const loom_func_symbol_facts_t* function_facts =
      loom_func_symbol_facts_cast(base_facts);
  if (!loom_symbol_ref_is_valid(function_facts->target_symbol)) {
    return iree_ok_status();
  }
  return loom_target_function_contract_resolve_facts(
      module, symbol_facts, function_facts, diagnostic_emitter, arena,
      out_valid, out_target_facts);
}

static iree_status_t loom_cmd_program_plan_compute_kernel_config_facts(
    const loom_module_t* module, const loom_cmd_program_root_build_t* roots,
    iree_host_size_t root_count, iree_arena_allocator_t* scratch_arena,
    loom_symbol_fact_table_t* symbol_facts,
    iree_diagnostic_emitter_t diagnostic_emitter,
    loom_value_fact_table_t* facts, bool* out_valid) {
  *out_valid = true;
  uint8_t* computed_symbols = NULL;
  if (module->symbols.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, module->symbols.count, sizeof(*computed_symbols),
        (void**)&computed_symbols));
    for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
      computed_symbols[i] = 0;
    }
  }

  for (iree_host_size_t root_index = 0; root_index < root_count; ++root_index) {
    const loom_cmd_schedule_plan_t* schedule = &roots[root_index].schedule;
    for (iree_host_size_t launch_index = 0;
         launch_index < schedule->command_count; ++launch_index) {
      const loom_op_t* launch_op = schedule->commands[launch_index];
      IREE_ASSERT(loom_kernel_launch_isa(launch_op));
      const loom_symbol_ref_t callee = loom_kernel_launch_callee(launch_op);
      IREE_ASSERT(loom_symbol_ref_is_valid(callee));
      IREE_ASSERT_EQ(callee.module_id, 0u);
      IREE_ASSERT_LT(callee.symbol_id, module->symbols.count);
      if (computed_symbols[callee.symbol_id]) continue;
      computed_symbols[callee.symbol_id] = 1;

      loom_op_t* kernel_op =
          module->symbols.entries[callee.symbol_id].defining_op;
      IREE_ASSERT(kernel_op != NULL);
      IREE_ASSERT(loom_kernel_def_isa(kernel_op));
      loom_region_t* config_region = loom_kernel_def_config(kernel_op);
      if (!config_region) continue;
      const loom_target_facts_t* target_facts = NULL;
      IREE_RETURN_IF_ERROR(loom_cmd_program_plan_resolve_function_target_facts(
          module, loom_func_like_cast(module, kernel_op), symbol_facts,
          diagnostic_emitter, scratch_arena, out_valid, &target_facts));
      if (!*out_valid) return iree_ok_status();
      facts->context.target_facts = target_facts;
      IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region(
          facts, module, loom_func_like_cast(module, kernel_op), config_region,
          kernel_op));
    }
  }
  return iree_ok_status();
}

// Establishes the dependency-unit contract before consumers assume every
// scheduled launch has a materializable Loom kernel body.
static iree_status_t loom_cmd_program_plan_validate_kernel_definitions(
    const loom_module_t* module, const loom_cmd_program_root_build_t* roots,
    iree_host_size_t root_count, iree_diagnostic_emitter_t emitter,
    bool* out_valid) {
  *out_valid = false;
  for (iree_host_size_t root_index = 0; root_index < root_count; ++root_index) {
    const loom_cmd_schedule_plan_t* schedule = &roots[root_index].schedule;
    for (iree_host_size_t launch_index = 0;
         launch_index < schedule->command_count; ++launch_index) {
      const loom_op_t* launch_op = schedule->commands[launch_index];
      IREE_ASSERT(loom_kernel_launch_isa(launch_op));
      const loom_symbol_ref_t callee = loom_kernel_launch_callee(launch_op);
      IREE_ASSERT(loom_symbol_ref_is_valid(callee));
      IREE_ASSERT_EQ(callee.module_id, 0u);
      IREE_ASSERT_LT(callee.symbol_id, module->symbols.count);
      const loom_symbol_t* symbol = &module->symbols.entries[callee.symbol_id];
      const loom_op_t* kernel_op = symbol->defining_op;
      IREE_ASSERT(kernel_op != NULL);
      if (loom_kernel_def_isa(kernel_op)) continue;

      IREE_ASSERT(loom_kernel_decl_isa(kernel_op));
      IREE_ASSERT_LT(symbol->name_id, module->strings.count);
      const loom_diagnostic_param_t params[] = {
          loom_param_string(module->strings.entries[symbol->name_id]),
      };
      const loom_diagnostic_related_op_t related_op = {
          .label = IREE_SV("launched here"),
          .op = launch_op,
      };
      const loom_diagnostic_emission_t emission = {
          .op = kernel_op,
          .error = LOOM_ERR_LOWERING_050,
          .params = params,
          .param_count = IREE_ARRAYSIZE(params),
          .related_ops = &related_op,
          .related_op_count = 1,
      };
      return iree_diagnostic_emit(emitter, &emission);
    }
  }
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_plan_allocate_tables(
    iree_host_size_t root_count, loom_cmd_program_plan_t* plan) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &plan->arena, root_count, sizeof(*plan->roots), (void**)&plan->roots));
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    plan->roots[i] = (loom_cmd_program_root_t){0};
  }
  plan->root_count = root_count;
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_plan_build_lower_plan(
    loom_cmd_program_plan_t* plan, loom_module_t* preparation_module,
    loom_op_t* root_program_op, const loom_cmd_schedule_plan_t* schedule,
    const loom_value_fact_table_t* source_facts,
    const loom_cmd_launch_graph_t* launch_graph,
    loom_cmd_kernel_entry_t* entries, const loom_op_t** entry_launches,
    uint32_t* inout_entry_count, iree_arena_allocator_t* scratch_arena,
    iree_arena_block_pool_t* block_pool,
    loom_cmd_parameter_requirement_table_t* out_parameters,
    loom_cmd_transient_requirement_t* out_transient,
    loom_cmd_lower_plan_t* out_lower_plan,
    loom_cmd_lower_launch_t** out_lower_launches, uint32_t** out_entry_indices,
    uint32_t* out_entry_count) {
  *out_lower_launches = NULL;
  *out_entry_indices = NULL;
  *out_entry_count = 0;
  if (schedule->command_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command root has too many kernel launches");
  }

  loom_cmd_lower_launch_t* launches = NULL;
  uint32_t* entry_indices = NULL;
  if (schedule->command_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, schedule->command_count,
                                  sizeof(*launches), (void**)&launches));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, schedule->command_count, sizeof(*entry_indices),
        (void**)&entry_indices));
  }
  uint32_t root_entry_count = 0;
  for (iree_host_size_t i = 0; i < schedule->command_count; ++i) {
    const loom_op_t* source_launch = schedule->commands[i];
    uint32_t entry_index = 0;
    while (entry_index < *inout_entry_count &&
           !loom_cmd_kernel_entry_launches_equivalent(
               preparation_module, source_launch, entry_launches[entry_index],
               source_facts)) {
      ++entry_index;
    }
    if (entry_index == *inout_entry_count) {
      IREE_RETURN_IF_ERROR(loom_cmd_kernel_entry_materialize(
          preparation_module, source_launch, source_facts, entry_index,
          block_pool, plan->host_allocator, &entries[entry_index]));
      entry_launches[entry_index] = source_launch;
      ++*inout_entry_count;
    }
    const loom_cmd_kernel_entry_t* entry = &entries[entry_index];
    uint32_t root_entry_index = 0;
    while (root_entry_index < root_entry_count &&
           entry_indices[root_entry_index] != entry_index) {
      ++root_entry_index;
    }
    if (root_entry_index == root_entry_count) {
      entry_indices[root_entry_count++] = entry_index;
    }
    launches[i] = (loom_cmd_lower_launch_t){
        .entry_index = root_entry_index,
        .argument_count = entry->argument_count,
        .source_argument_ordinals = entry->source_argument_ordinals,
    };
  }

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
      &plan->arena, bindings, binding_count, out_parameters,
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
    for (iree_host_size_t i = 0; i < parameter_layout.buffer_range_count; ++i) {
      buffer_ranges[i] = parameter_layout.buffer_ranges[i];
    }
    for (iree_host_size_t i = 0; i < transient_layout.buffer_range_count; ++i) {
      buffer_ranges[parameter_layout.buffer_range_count + i] =
          transient_layout.buffer_ranges[i];
    }
  }

  const bool has_host_launch_counts = launch_graph->host_tuple_count > 0;
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
                  (has_transient ? 1u : 0u) +
                  (has_host_launch_counts ? 1u : 0u),
              .executable_count = 0,
              .entry_count = 0,
          },
      .launch_graph = launch_graph,
      .launch_count_binding =
          {
              .resource_index = parameter_layout.rebindable_binding_count +
                                (has_transient ? 1u : 0u),
              .byte_offset = 0,
          },
      .launches = launches,
  };
  *out_lower_launches = launches;
  *out_entry_indices = entry_indices;
  *out_entry_count = root_entry_count;
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_plan_allocate_root_tables(
    loom_cmd_program_root_build_t* root, iree_arena_allocator_t* arena) {
  if (root->entry_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, root->entry_count,
                                                 sizeof(*root->entries),
                                                 (void**)&root->entries));
  return iree_arena_allocate_array(arena, root->entry_count,
                                   sizeof(*root->executable_unit_indices),
                                   (void**)&root->executable_unit_indices);
}

// Packs specialized kernel entries by exact sealed target and establishes the
// independent executable and entry tables consumed by each command root.
static iree_status_t loom_cmd_program_plan_pack_kernel_units(
    loom_cmd_program_plan_t* plan, const loom_module_t* preparation_module,
    const loom_cmd_kernel_entry_t* entries, uint32_t entry_count,
    loom_cmd_program_root_build_t* roots, iree_host_size_t root_count,
    iree_arena_allocator_t* scratch_arena,
    iree_arena_block_pool_t* block_pool) {
  if (entry_count == 0) return iree_ok_status();

  iree_host_size_t target_slot_count = 0;
  if (!iree_host_size_checked_add(preparation_module->symbols.count, 1,
                                  &target_slot_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command kernel target table is too large");
  }
  uint32_t* target_unit_indices = NULL;
  uint32_t* entry_unit_indices = NULL;
  uint32_t* entry_unit_export_indices = NULL;
  uint32_t* unit_first_entries = NULL;
  uint32_t* unit_last_entries = NULL;
  uint32_t* unit_entry_counts = NULL;
  uint32_t* next_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, target_slot_count, sizeof(*target_unit_indices),
      (void**)&target_unit_indices));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, entry_count,
                                                 sizeof(*entry_unit_indices),
                                                 (void**)&entry_unit_indices));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, entry_count, sizeof(*entry_unit_export_indices),
      (void**)&entry_unit_export_indices));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, entry_count,
                                                 sizeof(*unit_first_entries),
                                                 (void**)&unit_first_entries));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, entry_count,
                                                 sizeof(*unit_last_entries),
                                                 (void**)&unit_last_entries));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, entry_count,
                                                 sizeof(*unit_entry_counts),
                                                 (void**)&unit_entry_counts));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, entry_count,
                                                 sizeof(*next_entries),
                                                 (void**)&next_entries));
  for (iree_host_size_t i = 0; i < target_slot_count; ++i) {
    target_unit_indices[i] = UINT32_MAX;
  }
  for (uint32_t i = 0; i < entry_count; ++i) {
    unit_first_entries[i] = UINT32_MAX;
    unit_last_entries[i] = UINT32_MAX;
    unit_entry_counts[i] = 0;
    next_entries[i] = UINT32_MAX;
  }

  uint32_t unit_count = 0;
  for (uint32_t entry_index = 0; entry_index < entry_count; ++entry_index) {
    const loom_symbol_id_t target_symbol_id =
        entries[entry_index].source_target_symbol_id;
    const iree_host_size_t target_slot =
        target_symbol_id == LOOM_SYMBOL_ID_INVALID
            ? preparation_module->symbols.count
            : target_symbol_id;
    uint32_t unit_index = target_unit_indices[target_slot];
    if (unit_index == UINT32_MAX) {
      unit_index = unit_count++;
      target_unit_indices[target_slot] = unit_index;
      unit_first_entries[unit_index] = entry_index;
    } else {
      next_entries[unit_last_entries[unit_index]] = entry_index;
    }
    unit_last_entries[unit_index] = entry_index;
    entry_unit_indices[entry_index] = unit_index;
    entry_unit_export_indices[entry_index] = unit_entry_counts[unit_index]++;
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &plan->arena, unit_count, sizeof(*plan->dependency_units),
      (void**)&plan->dependency_units));
  for (uint32_t i = 0; i < unit_count; ++i) {
    plan->dependency_units[i] = (loom_cmd_kernel_unit_t){0};
  }

  uint32_t* packed_entry_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, entry_count, sizeof(*packed_entry_indices),
      (void**)&packed_entry_indices));
  for (uint32_t unit_index = 0; unit_index < unit_count; ++unit_index) {
    uint32_t packed_entry_count = 0;
    for (uint32_t entry_index = unit_first_entries[unit_index];
         entry_index != UINT32_MAX; entry_index = next_entries[entry_index]) {
      packed_entry_indices[packed_entry_count++] = entry_index;
    }
    IREE_ASSERT_EQ(packed_entry_count, unit_entry_counts[unit_index]);
    IREE_RETURN_IF_ERROR(loom_cmd_kernel_unit_pack(
        entries, packed_entry_indices, packed_entry_count, block_pool,
        plan->host_allocator, &plan->dependency_units[unit_index]));
    ++plan->dependency_count;
  }

  for (iree_host_size_t root_index = 0; root_index < root_count; ++root_index) {
    loom_cmd_program_root_build_t* root = &roots[root_index];
    IREE_RETURN_IF_ERROR(
        loom_cmd_program_plan_allocate_root_tables(root, &plan->arena));
    for (uint32_t root_entry_index = 0; root_entry_index < root->entry_count;
         ++root_entry_index) {
      const uint32_t entry_index = root->entry_indices[root_entry_index];
      const uint32_t unit_index = entry_unit_indices[entry_index];
      uint32_t executable_index = 0;
      while (executable_index < root->executable_count &&
             root->executable_unit_indices[executable_index] != unit_index) {
        ++executable_index;
      }
      if (executable_index == root->executable_count) {
        root->executable_unit_indices[root->executable_count++] = unit_index;
      }
      root->entries[root_entry_index] = (loom_cmd_program_entry_t){
          .executable_index = executable_index,
          .unit_export_index = entry_unit_export_indices[entry_index],
      };
    }
    for (iree_host_size_t launch_index = 0;
         launch_index < root->schedule.command_count; ++launch_index) {
      loom_cmd_lower_launch_t* launch = &root->lower_launches[launch_index];
      launch->executable_index =
          root->entries[launch->entry_index].executable_index;
    }
    root->lower_plan.abi_layout.executable_count = root->executable_count;
    root->lower_plan.abi_layout.entry_count = root->entry_count;
  }
  return iree_ok_status();
}

iree_status_t loom_cmd_program_plan_prepare(
    const loom_module_t* source_module,
    const loom_op_t* const* source_program_ops,
    iree_host_size_t source_program_count,
    const loom_pass_registry_t* pass_registry,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, bool* out_valid,
    loom_cmd_program_plan_t* out_plan, iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(source_module);
  IREE_ASSERT_ARGUMENT(source_program_ops);
  IREE_ASSERT_GT(source_program_count, 0u);
  IREE_ASSERT_ARGUMENT(pass_registry);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_valid);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_valid = false;
  memset(out_plan, 0, sizeof(*out_plan));

  loom_cmd_program_plan_t plan = {
      .host_allocator = host_allocator,
  };
  iree_arena_initialize(block_pool, &plan.arena);
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(block_pool, &scratch_arena);

  iree_string_view_t* root_names = NULL;
  iree_string_view_t* launch_names = NULL;
  loom_cmd_program_root_build_t* root_builds = NULL;
  iree_status_t status =
      iree_arena_allocate_array(&scratch_arena, source_program_count,
                                sizeof(*root_names), (void**)&root_names);
  if (iree_status_is_ok(status)) {
    status =
        iree_arena_allocate_array(&scratch_arena, source_program_count,
                                  sizeof(*launch_names), (void**)&launch_names);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_arena_allocate_array(&scratch_arena, source_program_count,
                                  sizeof(*root_builds), (void**)&root_builds);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < source_program_count; ++i) {
      root_builds[i] = (loom_cmd_program_root_build_t){0};
      IREE_ASSERT_ARGUMENT(source_program_ops[i]);
      const loom_func_like_t source_program =
          loom_func_like_cast(source_module, (loom_op_t*)source_program_ops[i]);
      IREE_ASSERT(loom_func_like_isa(source_program));
      const loom_symbol_ref_t source_program_ref =
          loom_func_like_callee(source_program);
      root_names[i] =
          loom_cmd_program_plan_symbol_name(source_module, source_program_ref);
      root_builds[i].name = root_names[i];
    }
  }

  const loom_module_t* source_modules[] = {source_module};
  loom_module_t* preparation_module = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_link_materialized_modules(
        source_modules, IREE_ARRAYSIZE(source_modules),
        &(loom_link_options_t){
            .module_name = IREE_SV("command_program_roots"),
            .root_symbols =
                {
                    .count = source_program_count,
                    .values = root_names,
                },
        },
        block_pool, host_allocator, &preparation_module);
  }

  loom_func_like_t* root_programs = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&scratch_arena, source_program_count,
                                       sizeof(*root_programs),
                                       (void**)&root_programs);
  }
  for (iree_host_size_t i = 0;
       i < source_program_count && iree_status_is_ok(status); ++i) {
    loom_cmd_program_root_build_t* root = &root_builds[i];
    root->program_op =
        loom_cmd_program_plan_find_symbol(preparation_module, root->name);
    root->program = loom_func_like_cast(preparation_module, root->program_op);
    IREE_ASSERT(loom_func_like_isa(root->program));
    root_programs[i] = root->program;
  }
  bool valid = false;
  if (iree_status_is_ok(status)) {
    status = loom_cmd_program_composition_flatten(
        preparation_module, source_module, root_programs, source_program_count,
        diagnostic_emitter, &scratch_arena, &valid);
  }
  if (valid && iree_status_is_ok(status)) {
    status = loom_cmd_program_plan_prepare_roots(
        preparation_module, root_programs, source_program_count, pass_registry,
        diagnostic_emitter, block_pool, &valid);
  }

  iree_host_size_t dependency_capacity = 0;
  for (iree_host_size_t i = 0;
       valid && i < source_program_count && iree_status_is_ok(status); ++i) {
    loom_cmd_program_root_build_t* root = &root_builds[i];
    status = loom_cmd_schedule_plan_build(preparation_module,
                                          loom_func_like_body(root->program),
                                          &scratch_arena, &root->schedule);
    if (iree_status_is_ok(status)) {
      if (root->schedule.command_count >
          (iree_host_size_t)UINT32_MAX - dependency_capacity) {
        status = iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "command program roots have too many dependency launch sites");
      } else {
        dependency_capacity += root->schedule.command_count;
      }
    }
  }

  if (valid && iree_status_is_ok(status)) {
    status = loom_cmd_program_plan_validate_kernel_definitions(
        preparation_module, root_builds, source_program_count,
        diagnostic_emitter, &valid);
  }

  loom_value_fact_table_t source_facts = {0};
  loom_symbol_fact_table_t symbol_facts = {0};
  if (valid && iree_status_is_ok(status)) {
    status = loom_value_fact_table_initialize(&source_facts, &scratch_arena,
                                              preparation_module->values.count);
  }
  if (valid && iree_status_is_ok(status)) {
    loom_type_registry_configure_fact_context(&source_facts.context);
    loom_symbol_fact_table_initialize(&symbol_facts, &scratch_arena);
  }
  for (iree_host_size_t i = 0;
       valid && i < source_program_count && iree_status_is_ok(status); ++i) {
    const loom_target_facts_t* target_facts = NULL;
    status = loom_cmd_program_plan_resolve_function_target_facts(
        preparation_module, root_builds[i].program, &symbol_facts,
        diagnostic_emitter, &scratch_arena, &valid, &target_facts);
    if (valid && iree_status_is_ok(status)) {
      source_facts.context.target_facts = target_facts;
      status = loom_value_fact_table_compute(&source_facts, preparation_module,
                                             root_builds[i].program);
    }
  }
  if (valid && iree_status_is_ok(status)) {
    status = loom_cmd_program_plan_compute_kernel_config_facts(
        preparation_module, root_builds, source_program_count, &scratch_arena,
        &symbol_facts, diagnostic_emitter, &source_facts, &valid);
  }
  for (iree_host_size_t i = 0;
       valid && i < source_program_count && iree_status_is_ok(status); ++i) {
    loom_cmd_program_root_build_t* root = &root_builds[i];
    status = loom_cmd_launch_graph_materialize(
        preparation_module, root->program_op, &root->schedule, &source_facts,
        block_pool, host_allocator, &root->launch_graph);
    if (iree_status_is_ok(status)) {
      const loom_symbol_ref_t launch_ref =
          loom_func_like_callee(loom_func_like_cast(
              root->launch_graph.module, root->launch_graph.host_function_op));
      root->launch_name = loom_cmd_program_plan_symbol_name(
          root->launch_graph.module, launch_ref);
      launch_names[i] = root->launch_name;
    }
  }

  if (valid && iree_status_is_ok(status)) {
    status = loom_cmd_program_plan_allocate_tables(source_program_count, &plan);
  }
  loom_cmd_kernel_entry_t* kernel_entries = NULL;
  const loom_op_t** entry_launches = NULL;
  if (valid && iree_status_is_ok(status) && dependency_capacity > 0) {
    status = iree_arena_allocate_array(&scratch_arena, dependency_capacity,
                                       sizeof(*kernel_entries),
                                       (void**)&kernel_entries);
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < dependency_capacity; ++i) {
        kernel_entries[i] = (loom_cmd_kernel_entry_t){0};
      }
      status = iree_arena_allocate_array(&scratch_arena, dependency_capacity,
                                         sizeof(*entry_launches),
                                         (void**)&entry_launches);
    }
  }
  uint32_t kernel_entry_count = 0;
  for (iree_host_size_t i = 0;
       valid && i < source_program_count && iree_status_is_ok(status); ++i) {
    loom_cmd_program_root_build_t* root = &root_builds[i];
    status = loom_cmd_program_plan_build_lower_plan(
        &plan, preparation_module, root->program_op, &root->schedule,
        &source_facts, &root->launch_graph, kernel_entries, entry_launches,
        &kernel_entry_count, &scratch_arena, block_pool, &root->parameters,
        &root->transient, &root->lower_plan, &root->lower_launches,
        &root->entry_indices, &root->entry_count);
  }
  if (valid && iree_status_is_ok(status)) {
    status = loom_cmd_program_plan_pack_kernel_units(
        &plan, preparation_module, kernel_entries, kernel_entry_count,
        root_builds, source_program_count, &scratch_arena, block_pool);
  }
  for (iree_host_size_t i = 0;
       valid && i < source_program_count && iree_status_is_ok(status); ++i) {
    loom_op_t* preparation_root_function = NULL;
    status = loom_cmd_lower_program_to_low(
        preparation_module, root_builds[i].program_op,
        &root_builds[i].lower_plan, &preparation_root_function);
  }
  if (valid && iree_status_is_ok(status)) {
    const loom_module_t* root_source_modules[] = {preparation_module};
    status = loom_link_materialized_modules(
        root_source_modules, IREE_ARRAYSIZE(root_source_modules),
        &(loom_link_options_t){
            .module_name = IREE_SV("command_program_roots"),
            .root_symbols =
                {
                    .count = source_program_count,
                    .values = root_names,
                },
        },
        block_pool, host_allocator, &plan.root_module);
  }
  const loom_module_t** launch_source_modules = NULL;
  if (valid && iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&scratch_arena, source_program_count,
                                       sizeof(*launch_source_modules),
                                       (void**)&launch_source_modules);
  }
  if (valid && iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < source_program_count; ++i) {
      launch_source_modules[i] = root_builds[i].launch_graph.module;
    }
    status = loom_link_materialized_modules(
        launch_source_modules, source_program_count,
        &(loom_link_options_t){
            .module_name = IREE_SV("command_program_launch"),
            .root_symbols =
                {
                    .count = source_program_count,
                    .values = launch_names,
                },
        },
        block_pool, host_allocator, &plan.launch_module);
  }
  if (valid && iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < source_program_count; ++i) {
      loom_cmd_program_root_t* root = &plan.roots[i];
      loom_cmd_program_root_build_t* build = &root_builds[i];
      root->function_op =
          loom_cmd_program_plan_find_symbol(plan.root_module, build->name);
      root->abi_layout = build->lower_plan.abi_layout;
      root->launch_function_op = loom_cmd_program_plan_find_symbol(
          plan.launch_module, build->launch_name);
      root->launch_tuple_count = build->launch_graph.host_tuple_count;
      root->executable_unit_indices = build->executable_unit_indices;
      root->executable_count = build->executable_count;
      root->entries = build->entries;
      root->entry_count = build->entry_count;
      root->parameters = build->parameters;
      root->transient = build->transient;
      build->executable_unit_indices = NULL;
      build->executable_count = 0;
      build->entries = NULL;
      build->entry_count = 0;
      memset(&build->parameters, 0, sizeof(build->parameters));
    }
  }

  for (uint32_t i = 0; i < kernel_entry_count; ++i) {
    loom_cmd_kernel_entry_deinitialize(&kernel_entries[i]);
  }
  if (root_builds) {
    for (iree_host_size_t i = 0; i < source_program_count; ++i) {
      loom_cmd_launch_graph_deinitialize(&root_builds[i].launch_graph);
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

loom_cmd_program_requirements_t loom_cmd_program_root_requirements(
    const loom_cmd_program_root_t* root) {
  IREE_ASSERT_ARGUMENT(root);
  loom_cmd_program_requirements_t requirements = {
      .fixed_buffer_count = root->abi_layout.fixed_buffer_count,
      .rebindable_binding_count = root->abi_layout.rebindable_binding_count,
      .executable_count = root->abi_layout.executable_count,
      .entry_count = root->abi_layout.entry_count,
      .transient =
          {
              .binding_index = root->transient.binding_index,
              .required_byte_length = root->transient.required_byte_length,
              .minimum_alignment = root->transient.minimum_alignment,
          },
      .launch_counts =
          {
              .binding_index = UINT32_MAX,
          },
  };
  if (root->launch_tuple_count != 0) {
    IREE_ASSERT_GT(requirements.rebindable_binding_count, 0u);
    requirements.launch_counts = (loom_cmd_program_launch_count_requirement_t){
        .binding_index = requirements.rebindable_binding_count - 1,
        .required_byte_length = (uint64_t)root->launch_tuple_count *
                                LOOM_CMD_PROGRAM_LAUNCH_COUNT_TUPLE_BYTE_LENGTH,
        .minimum_alignment = LOOM_CMD_PROGRAM_LAUNCH_COUNT_TUPLE_ALIGNMENT,
    };
  }
  return requirements;
}

void loom_cmd_program_plan_deinitialize(loom_cmd_program_plan_t* plan) {
  if (!plan) return;
  for (iree_host_size_t i = 0; i < plan->dependency_count; ++i) {
    loom_cmd_kernel_unit_deinitialize(&plan->dependency_units[i]);
  }
  if (plan->launch_module) loom_module_free(plan->launch_module);
  if (plan->root_module) loom_module_free(plan->root_module);
  iree_arena_deinitialize(&plan->arena);
  memset(plan, 0, sizeof(*plan));
}
