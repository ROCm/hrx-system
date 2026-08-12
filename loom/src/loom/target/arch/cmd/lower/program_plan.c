// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/program_plan.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/link/linker.h"
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
  loom_op_t* program_op;
  loom_func_like_t program;
  loom_cmd_schedule_plan_t schedule;
  loom_cmd_launch_graph_t launch_graph;
  loom_cmd_lower_plan_t lower_plan;
  loom_cmd_parameter_requirement_table_t parameters;
  loom_cmd_transient_requirement_t transient;
  uint32_t* dependency_unit_indices;
  uint32_t dependency_count;
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

static iree_status_t loom_cmd_program_plan_compute_kernel_config_facts(
    const loom_module_t* module, const loom_cmd_program_root_build_t* roots,
    iree_host_size_t root_count, iree_arena_allocator_t* scratch_arena,
    loom_value_fact_table_t* facts) {
  uint8_t* computed_symbols = NULL;
  if (module->symbols.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, module->symbols.count, sizeof(*computed_symbols),
        (void**)&computed_symbols));
    memset(computed_symbols, 0,
           module->symbols.count * sizeof(*computed_symbols));
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
    iree_host_size_t root_count, iree_host_size_t dependency_capacity,
    loom_cmd_program_plan_t* plan) {
  if (root_count > IREE_HOST_SIZE_MAX / sizeof(*plan->roots) ||
      dependency_capacity >
          IREE_HOST_SIZE_MAX / sizeof(*plan->dependency_units)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command program plan tables are too large");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(plan->host_allocator,
                                             root_count * sizeof(*plan->roots),
                                             (void**)&plan->roots));
  memset(plan->roots, 0, root_count * sizeof(*plan->roots));
  plan->root_count = root_count;
  if (dependency_capacity == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      plan->host_allocator,
      dependency_capacity * sizeof(*plan->dependency_units),
      (void**)&plan->dependency_units));
  memset(plan->dependency_units, 0,
         dependency_capacity * sizeof(*plan->dependency_units));
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_plan_build_lower_plan(
    loom_cmd_program_plan_t* plan, loom_module_t* preparation_module,
    loom_op_t* root_program_op, const loom_cmd_schedule_plan_t* schedule,
    const loom_value_fact_table_t* source_facts,
    const loom_cmd_launch_graph_t* launch_graph,
    const loom_op_t** dependency_launches,
    iree_arena_allocator_t* scratch_arena, iree_arena_block_pool_t* block_pool,
    loom_cmd_parameter_requirement_table_t* out_parameters,
    loom_cmd_transient_requirement_t* out_transient,
    loom_cmd_lower_plan_t* out_lower_plan,
    uint32_t** out_dependency_unit_indices, uint32_t* out_dependency_count) {
  *out_dependency_unit_indices = NULL;
  *out_dependency_count = 0;
  if (schedule->command_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command root has too many dependency units");
  }

  loom_cmd_lower_launch_t* launches = NULL;
  uint32_t* dependency_unit_indices = NULL;
  if (schedule->command_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, schedule->command_count,
                                  sizeof(*launches), (void**)&launches));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, schedule->command_count,
        sizeof(*dependency_unit_indices), (void**)&dependency_unit_indices));
  }
  uint32_t dependency_count = 0;
  for (iree_host_size_t i = 0; i < schedule->command_count; ++i) {
    const loom_op_t* source_launch = schedule->commands[i];
    uint32_t dependency_index = 0;
    while (dependency_index < plan->dependency_count &&
           !loom_cmd_kernel_unit_launches_equivalent(
               preparation_module, source_launch,
               dependency_launches[dependency_index], source_facts)) {
      ++dependency_index;
    }
    if (dependency_index == plan->dependency_count) {
      loom_cmd_kernel_unit_t* new_unit =
          &plan->dependency_units[dependency_index];
      IREE_RETURN_IF_ERROR(loom_cmd_kernel_unit_materialize(
          preparation_module, source_launch, source_facts, block_pool,
          plan->host_allocator, new_unit));
      dependency_launches[dependency_index] = source_launch;
      ++plan->dependency_count;
    }
    const loom_cmd_kernel_unit_t* unit =
        &plan->dependency_units[dependency_index];
    uint32_t root_dependency_index = 0;
    while (root_dependency_index < dependency_count &&
           dependency_unit_indices[root_dependency_index] != dependency_index) {
      ++root_dependency_index;
    }
    if (root_dependency_index == dependency_count) {
      dependency_unit_indices[dependency_count++] = dependency_index;
    }
    launches[i] = (loom_cmd_lower_launch_t){
        .executable_index = root_dependency_index,
        .entry_index = root_dependency_index,
        .argument_count = unit->argument_count,
        .source_argument_ordinals = unit->source_argument_ordinals,
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

  uint32_t* owned_dependency_unit_indices = NULL;
  if (dependency_count > 0) {
    iree_host_size_t dependency_table_size = 0;
    if (!iree_host_size_checked_mul(dependency_count,
                                    sizeof(*owned_dependency_unit_indices),
                                    &dependency_table_size)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "command root dependency table is too large");
    }
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc(plan->host_allocator, dependency_table_size,
                              (void**)&owned_dependency_unit_indices));
    memcpy(owned_dependency_unit_indices, dependency_unit_indices,
           dependency_count * sizeof(*owned_dependency_unit_indices));
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
              .executable_count = dependency_count,
              .entry_count = dependency_count,
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
  *out_dependency_unit_indices = owned_dependency_unit_indices;
  *out_dependency_count = dependency_count;
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
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(block_pool, &scratch_arena);

  iree_string_view_t* root_names = NULL;
  loom_cmd_program_root_build_t* root_builds = NULL;
  iree_status_t status =
      iree_arena_allocate_array(&scratch_arena, source_program_count,
                                sizeof(*root_names), (void**)&root_names);
  if (iree_status_is_ok(status)) {
    status =
        iree_arena_allocate_array(&scratch_arena, source_program_count,
                                  sizeof(*root_builds), (void**)&root_builds);
  }
  if (iree_status_is_ok(status)) {
    memset(root_builds, 0, source_program_count * sizeof(*root_builds));
    for (iree_host_size_t i = 0; i < source_program_count; ++i) {
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
  if (valid && iree_status_is_ok(status)) {
    status = loom_value_fact_table_initialize(&source_facts, &scratch_arena,
                                              preparation_module->values.count);
  }
  if (valid && iree_status_is_ok(status)) {
    loom_type_registry_configure_fact_context(&source_facts.context);
  }
  for (iree_host_size_t i = 0;
       valid && i < source_program_count && iree_status_is_ok(status); ++i) {
    status = loom_value_fact_table_compute(&source_facts, preparation_module,
                                           root_builds[i].program);
  }
  if (valid && iree_status_is_ok(status)) {
    status = loom_cmd_program_plan_compute_kernel_config_facts(
        preparation_module, root_builds, source_program_count, &scratch_arena,
        &source_facts);
  }
  for (iree_host_size_t i = 0;
       valid && i < source_program_count && iree_status_is_ok(status); ++i) {
    loom_cmd_program_root_build_t* root = &root_builds[i];
    status = loom_cmd_launch_graph_materialize(
        preparation_module, root->program_op, &root->schedule, &source_facts,
        block_pool, host_allocator, &root->launch_graph);
  }

  if (valid && iree_status_is_ok(status)) {
    status = loom_cmd_program_plan_allocate_tables(source_program_count,
                                                   dependency_capacity, &plan);
  }
  const loom_op_t** dependency_launches = NULL;
  if (valid && iree_status_is_ok(status) && dependency_capacity > 0) {
    status = iree_arena_allocate_array(&scratch_arena, dependency_capacity,
                                       sizeof(*dependency_launches),
                                       (void**)&dependency_launches);
  }
  for (iree_host_size_t i = 0;
       valid && i < source_program_count && iree_status_is_ok(status); ++i) {
    loom_cmd_program_root_build_t* root = &root_builds[i];
    status = loom_cmd_program_plan_build_lower_plan(
        &plan, preparation_module, root->program_op, &root->schedule,
        &source_facts, &root->launch_graph, dependency_launches, &scratch_arena,
        block_pool, &root->parameters, &root->transient, &root->lower_plan,
        &root->dependency_unit_indices, &root->dependency_count);
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
                    .values = root_names,
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
      root->launch_function_op =
          loom_cmd_program_plan_find_symbol(plan.launch_module, build->name);
      root->launch_tuple_count = build->launch_graph.host_tuple_count;
      root->dependency_unit_indices = build->dependency_unit_indices;
      root->dependency_count = build->dependency_count;
      root->parameters = build->parameters;
      root->transient = build->transient;
      build->dependency_unit_indices = NULL;
      build->dependency_count = 0;
      memset(&build->parameters, 0, sizeof(build->parameters));
    }
  }

  if (root_builds) {
    for (iree_host_size_t i = 0; i < source_program_count; ++i) {
      loom_cmd_launch_graph_deinitialize(&root_builds[i].launch_graph);
      loom_cmd_parameter_requirement_table_deinitialize(
          &root_builds[i].parameters, host_allocator);
      iree_allocator_free(host_allocator,
                          root_builds[i].dependency_unit_indices);
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
  for (iree_host_size_t i = 0; i < plan->root_count; ++i) {
    loom_cmd_parameter_requirement_table_deinitialize(
        &plan->roots[i].parameters, plan->host_allocator);
    iree_allocator_free(plan->host_allocator,
                        plan->roots[i].dependency_unit_indices);
  }
  for (iree_host_size_t i = 0; i < plan->dependency_count; ++i) {
    loom_cmd_kernel_unit_deinitialize(&plan->dependency_units[i]);
  }
  iree_allocator_free(plan->host_allocator, plan->dependency_units);
  iree_allocator_free(plan->host_allocator, plan->roots);
  if (plan->launch_module) loom_module_free(plan->launch_module);
  if (plan->root_module) loom_module_free(plan->root_module);
  memset(plan, 0, sizeof(*plan));
}
