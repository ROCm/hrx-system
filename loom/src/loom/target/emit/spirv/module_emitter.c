// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_emitter.h"

#include "loom/target/emit/spirv/function_emitter.h"
#include "loom/target/emit/spirv/module_abi.h"
#include "loom/target/emit/spirv/module_types.h"

typedef struct loom_spirv_emit_module_state_t {
  // Module containing the prepared Low functions.
  loom_module_t* module;
  // Immutable semantic program being emitted.
  const loom_spirv_program_plan_t* program;
  // Module-emission scratch arena.
  iree_arena_allocator_t* scratch_arena;
  // Sectioned SPIR-V module builder shared by every emitted function.
  loom_spirv_module_builder_t builder;
  // SPIR-V type and constant emission cache shared by the module.
  loom_spirv_type_context_t type_context;
  // Physical function identities and signatures in program order.
  loom_spirv_function_emission_plan_t* emission_plans;
  // Physical push-constant storage shared by HAL kernel entries.
  loom_spirv_module_shared_bda_root_t shared_bda_root;
  // Shared Input variables for invocation and subgroup builtins.
  uint32_t builtin_variable_ids[LOOM_SPIRV_BUILTIN_VARIABLE_COUNT];
  // Module-arena helper IDs, indexed by typed descriptor and allocated on use.
  uint32_t* component_math_function_ids;
} loom_spirv_emit_module_state_t;

static void loom_spirv_emit_module_state_initialize(
    const loom_spirv_program_plan_t* program,
    iree_arena_allocator_t* scratch_arena, iree_allocator_t allocator,
    loom_spirv_emit_module_state_t* out_state) {
  *out_state = (loom_spirv_emit_module_state_t){
      .module = program->module,
      .program = program,
      .scratch_arena = scratch_arena,
  };
  loom_spirv_module_builder_initialize(
      loom_spirv_function_plan_target_bundle(
          &program->functions[program->first_entry_index]),
      allocator, &out_state->builder);
  loom_spirv_type_context_initialize(&out_state->builder, scratch_arena,
                                     &out_state->type_context);
}

static iree_status_t loom_spirv_emit_initialize_function_plans(
    loom_spirv_emit_module_state_t* state) {
  const loom_spirv_program_plan_t* program = state->program;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, program->function_count,
      sizeof(*state->emission_plans), (void**)&state->emission_plans));
  for (iree_host_size_t i = 0; i < program->function_count; ++i) {
    state->emission_plans[i] = (loom_spirv_function_emission_plan_t){
        .function = &program->functions[i],
    };
  }
  return iree_ok_status();
}

static void loom_spirv_emit_module_state_deinitialize(
    loom_spirv_emit_module_state_t* state) {
  loom_spirv_module_builder_deinitialize(&state->builder);
  *state = (loom_spirv_emit_module_state_t){0};
}

static iree_status_t loom_spirv_emit_function_into_module(
    loom_spirv_emit_module_state_t* state,
    loom_spirv_function_emission_plan_t* emission_plan,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* accepted) {
  loom_diagnostic_export_scope_t diagnostic_scope = {
      .export_name =
          loom_target_facts_export_name(emission_plan->function->target_facts),
      .target = diagnostic_emitter,
  };
  loom_spirv_function_emission_context_t context = {
      .module = state->module,
      .program = state->program,
      .emission_plans = state->emission_plans,
      .diagnostic_emitter =
          loom_diagnostic_export_scope_emitter(&diagnostic_scope),
      .accepted = accepted,
      .scratch_arena = state->scratch_arena,
      .builder = &state->builder,
      .type_context = &state->type_context,
      .shared_bda_root = &state->shared_bda_root,
      .builtin_variable_ids = state->builtin_variable_ids,
      .component_math_function_ids = &state->component_math_function_ids,
  };
  return loom_spirv_emit_low_function(&context, emission_plan);
}

static iree_status_t loom_spirv_emit_module_state_finalize(
    loom_spirv_emit_module_state_t* state,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_accepted,
    loom_spirv_module_binary_t* out_module) {
  loom_spirv_module_abi_context_t context = {
      .module = state->module,
      .scratch_arena = state->scratch_arena,
      .builder = &state->builder,
      .type_context = &state->type_context,
      .shared_bda_root = &state->shared_bda_root,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_emit_shared_bda_root(&context));
  return loom_spirv_module_builder_finalize(&state->builder, diagnostic_emitter,
                                            out_accepted, out_module);
}

iree_status_t loom_spirv_program_emit_binary(
    const loom_spirv_program_plan_t* program,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena, iree_allocator_t allocator,
    bool* out_accepted, loom_spirv_module_binary_t* out_module) {
  IREE_ASSERT_ARGUMENT(program);
  IREE_ASSERT_ARGUMENT(program->module);
  IREE_ASSERT_ARGUMENT(program->functions);
  IREE_ASSERT_NE(program->function_count, 0u);
  IREE_ASSERT_NE(program->entry_count, 0u);
  IREE_ASSERT_NE(program->first_entry_index,
                 LOOM_SPIRV_PROGRAM_FUNCTION_INDEX_NONE);
  IREE_ASSERT_LT(program->first_entry_index, program->function_count);
  IREE_ASSERT_ARGUMENT(scratch_arena);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_accepted = false;
  *out_module = (loom_spirv_module_binary_t){0};

  loom_spirv_emit_module_state_t state = {0};
  loom_spirv_emit_module_state_initialize(program, scratch_arena, allocator,
                                          &state);
  bool functions_accepted = true;
  iree_status_t status = loom_spirv_emit_initialize_function_plans(&state);
  for (iree_host_size_t i = 0; i < program->function_count &&
                               functions_accepted && iree_status_is_ok(status);
       ++i) {
    status = loom_spirv_emit_function_into_module(
        &state, &state.emission_plans[i], diagnostic_emitter,
        &functions_accepted);
  }
  if (iree_status_is_ok(status) && functions_accepted) {
    status = loom_spirv_emit_module_state_finalize(&state, diagnostic_emitter,
                                                   out_accepted, out_module);
  }
  loom_spirv_emit_module_state_deinitialize(&state);
  if (!iree_status_is_ok(status)) {
    loom_spirv_module_binary_deinitialize(out_module, allocator);
  }
  return status;
}
