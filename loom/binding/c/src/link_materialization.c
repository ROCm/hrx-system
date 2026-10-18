// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "link_materialization.h"

#include <string.h>

#include "config.h"
#include "context.h"
#include "diagnostic.h"
#include "link_index.h"
#include "loom/codegen/low/repr.h"
#include "loom/ir/module.h"
#include "loom/link/plan_materializer.h"
#include "loom/target/module_specialization.h"
#include "loomc/iree.h"
#include "target.h"
#include "workspace.h"

static iree_status_t loomc_link_materialization_capture_diagnostic(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loomc_link_materialization_state_t* state =
      (loomc_link_materialization_state_t*)user_data;
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic(
      state->diagnostics.result, state->diagnostics.source, diagnostic,
      /*text_print_options=*/NULL));
}

static loom_diagnostic_sink_t loomc_link_materialization_diagnostic_sink(
    void* user_data, const loom_link_module_index_provider_t* provider) {
  loomc_link_materialization_state_t* state =
      (loomc_link_materialization_state_t*)user_data;
  const loomc_source_t* source = NULL;
  if (state->composition.sources.appended.source != NULL &&
      provider->ordinal ==
          state->composition.sources.appended.provider_ordinal) {
    source = state->composition.sources.appended.source;
  } else {
    source = loomc_link_index_source_for_provider(
        state->composition.sources.link_index, provider->ordinal);
  }
  state->diagnostics.source = source;
  return (loom_diagnostic_sink_t){
      .fn = loomc_link_materialization_capture_diagnostic,
      .user_data = state,
  };
}

// One invocation assignment expressed in the immutable source index domain.
typedef struct loomc_link_target_assignment_t {
  // Index-wide source symbol selected by an invocation assignment.
  iree_host_size_t symbol_ordinal;
  // Complete profile borrowed for the synchronous invocation.
  const loom_target_profile_t* profile;
} loomc_link_target_assignment_t;

struct loomc_link_target_specializations_t {
  // Arena-owned source assignments, with function rows before declaration rows.
  loomc_link_target_assignment_t* assignments;
  // Number of function rows at the start of assignments.
  iree_host_size_t function_count;
  // Total number of source assignments, including declaration aliases.
  iree_host_size_t count;
};

static loom_symbol_ref_t loomc_link_configuration_function(
    const loom_link_plan_materialization_t* materialization,
    loom_symbol_ref_t function_ref) {
  return materialization->target_kernel_configurations.count != 0
             ? materialization->target_kernel_configurations
                   .values[function_ref.symbol_id]
             : loom_symbol_ref_null();
}

// Joins validated input rows to the linker's retained source correspondence.
// This runs once while all caller-named symbols are present. Later modules
// consume source ordinals even when private names change or symbols disappear.
static iree_status_t loomc_link_target_specializations_capture(
    const loom_link_module_index_t* index,
    const loom_link_plan_materialization_t* materialization,
    loom_target_specialization_request_list_t* inout_requests,
    loom_target_declaration_binding_list_t bindings,
    iree_arena_allocator_t* scratch_arena, iree_arena_allocator_t* arena,
    const loomc_link_target_specializations_t** out_specializations) {
  const loom_target_specialization_request_list_t requests = *inout_requests;
  const iree_host_size_t symbol_count = materialization->module->symbols.count;
  uint32_t* input_rows = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, symbol_count, sizeof(*input_rows), (void**)&input_rows));
  for (iree_host_size_t i = 0; i < symbol_count; ++i) {
    input_rows[i] = UINT32_MAX;
  }
  for (iree_host_size_t i = 0; i < requests.count; ++i) {
    input_rows[requests.values[i].function_ref.symbol_id] = i;
  }
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    input_rows[bindings.values[i].target_ref.symbol_id] = requests.count + i;
  }

  iree_host_size_t function_count = 0;
  iree_host_size_t count = 0;
  uint32_t* source_rows = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, materialization->target_symbols.count,
      sizeof(*source_rows), (void**)&source_rows));
  for (iree_host_size_t i = 0; i < materialization->target_symbols.count; ++i) {
    source_rows[i] = UINT32_MAX;
  }
  for (iree_host_size_t i = 0; i < materialization->target_symbols.count; ++i) {
    const loom_symbol_ref_t ref = materialization->target_symbols.values[i];
    if (!loom_symbol_ref_is_valid(ref) ||
        input_rows[ref.symbol_id] == UINT32_MAX ||
        source_rows[i] != UINT32_MAX) {
      continue;
    }
    const uint32_t row = input_rows[ref.symbol_id];
    if (row < requests.count) {
      source_rows[i] = row;
      ++function_count;
      ++count;
    } else {
      // Target declarations have global identity. A body-only provider can
      // introduce another declaration absent from the command projection.
      // Join the index-owned group once and retain every source identity.
      const loom_link_module_index_symbol_t* symbol =
          loom_link_module_index_symbol_at(index, i);
      for (const loom_link_module_index_symbol_t* alias =
               loom_link_module_index_lookup_global(index, symbol->name);
           alias != NULL;
           alias = loom_link_module_index_next_global_duplicate(index, alias)) {
        source_rows[alias->ordinal] = row;
        ++count;
      }
    }
  }
  loomc_link_target_specializations_t* specializations = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, sizeof(*specializations),
                                           (void**)&specializations));
  *specializations = (loomc_link_target_specializations_t){
      .function_count = function_count,
      .count = count,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, count, sizeof(*specializations->assignments),
      (void**)&specializations->assignments));
  iree_host_size_t next_function = 0;
  iree_host_size_t next_binding = function_count;
  for (iree_host_size_t i = 0; i < materialization->target_symbols.count; ++i) {
    const uint32_t row = source_rows[i];
    if (row == UINT32_MAX) {
      continue;
    }
    const bool is_function = row < requests.count;
    const iree_host_size_t output =
        is_function ? next_function++ : next_binding++;
    specializations->assignments[output] = (loomc_link_target_assignment_t){
        .symbol_ordinal = i,
        .profile = is_function
                       ? requests.values[row].target_profile
                       : bindings.values[row - requests.count].target_profile,
    };
  }

  // A direct kernel assignment owns both the logical entry and its projected
  // configuration. Declaration bindings already cover both through their
  // shared target witness. Explicitly assigning the synthetic helper as well
  // would assign the same function twice and is rejected at this input
  // boundary.
  iree_host_size_t configuration_count = 0;
  for (iree_host_size_t i = 0; i < requests.count; ++i) {
    const loom_symbol_ref_t configuration = loomc_link_configuration_function(
        materialization, requests.values[i].function_ref);
    if (!loom_symbol_ref_is_valid(configuration)) {
      continue;
    }
    if (input_rows[configuration.symbol_id] != UINT32_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "target specialization assigns both a kernel and "
                              "its projected configuration function");
    }
    ++configuration_count;
  }
  if (configuration_count != 0) {
    loom_target_specialization_request_t* expanded = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, requests.count + configuration_count, sizeof(*expanded),
        (void**)&expanded));
    memcpy(expanded, requests.values, requests.count * sizeof(*expanded));
    iree_host_size_t output = requests.count;
    for (iree_host_size_t i = 0; i < requests.count; ++i) {
      const loom_symbol_ref_t configuration = loomc_link_configuration_function(
          materialization, requests.values[i].function_ref);
      if (!loom_symbol_ref_is_valid(configuration)) {
        continue;
      }
      expanded[output++] = (loom_target_specialization_request_t){
          .function_ref = configuration,
          .target_profile = requests.values[i].target_profile,
      };
    }
    *inout_requests = (loom_target_specialization_request_list_t){
        .values = expanded,
        .count = output,
    };
  }
  *out_specializations = specializations;
  return iree_ok_status();
}

static iree_status_t loomc_link_target_specializations_project(
    const loomc_link_target_specializations_t* specializations,
    const loom_link_plan_materialization_t* materialization,
    iree_arena_allocator_t* arena,
    loom_target_specialization_request_list_t* out_requests,
    loom_target_declaration_binding_list_t* out_bindings) {
  const iree_host_size_t symbol_count = materialization->module->symbols.count;
  const iree_host_size_t word_count = iree_bitmap_calculate_words(symbol_count);
  uint64_t* words = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, word_count, sizeof(*words), (void**)&words));
  memset(words, 0, word_count * sizeof(*words));
  const iree_bitmap_t assigned = {.bit_count = symbol_count, .words = words};
  loom_target_specialization_request_t* requests = NULL;
  loom_target_declaration_binding_t* bindings = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, specializations->function_count * 2,
                                sizeof(*requests), (void**)&requests));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, specializations->count - specializations->function_count,
      sizeof(*bindings), (void**)&bindings));
  iree_host_size_t request_count = 0;
  iree_host_size_t binding_count = 0;
  for (iree_host_size_t i = 0; i < specializations->count; ++i) {
    const loomc_link_target_assignment_t* assignment =
        &specializations->assignments[i];
    const loom_symbol_ref_t ref =
        materialization->target_symbols.values[assignment->symbol_ordinal];
    if (!loom_symbol_ref_is_valid(ref) ||
        iree_bitmap_test(assigned, ref.symbol_id)) {
      continue;
    }
    iree_bitmap_set(assigned, ref.symbol_id);
    if (i < specializations->function_count) {
      requests[request_count++] = (loom_target_specialization_request_t){
          .function_ref = ref,
          .target_profile = assignment->profile,
      };
      const loom_symbol_ref_t configuration =
          loomc_link_configuration_function(materialization, ref);
      if (loom_symbol_ref_is_valid(configuration) &&
          !iree_bitmap_test(assigned, configuration.symbol_id)) {
        iree_bitmap_set(assigned, configuration.symbol_id);
        requests[request_count++] = (loom_target_specialization_request_t){
            .function_ref = configuration,
            .target_profile = assignment->profile,
        };
      }
    } else {
      bindings[binding_count++] = (loom_target_declaration_binding_t){
          .target_ref = ref,
          .target_profile = assignment->profile,
      };
    }
  }
  *out_requests = (loom_target_specialization_request_list_t){
      .values = requests, .count = request_count};
  *out_bindings = (loom_target_declaration_binding_list_t){
      .values = bindings, .count = binding_count};
  return iree_ok_status();
}

static iree_status_t loomc_link_materialization_prepare_module(
    void* user_data, const loom_link_module_index_t* index,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_link_plan_materialization_t* materialization) {
  loom_module_t** inout_module = &materialization->module;
  loomc_link_materialization_state_t* state =
      (loomc_link_materialization_state_t*)user_data;
  const loomc_config_apply_text_to_module_options_t apply_options = {
      .config = state->specialization.config,
      .module = *inout_module,
      .source_resolver = (loom_source_resolver_t){0},
      .result = state->diagnostics.result,
      .diagnostic_code = loomc_make_cstring_view("CONFIG/INVALID"),
      .block_pool = block_pool,
      .allocator = loomc_allocator_from_iree(allocator),
  };
  loomc_status_t status = loomc_config_apply_text_to_module(&apply_options);
  if (!loomc_status_is_ok(status)) {
    return iree_status_from_loomc(status);
  }
  if (!loomc_result_succeeded(state->diagnostics.result)) {
    return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
  }

  const loomc_target_specialization_options_t* target =
      state->specialization.target;
  const loomc_target_profile_t* kernel_profile =
      state->specialization.kernel_profile;
  if (kernel_profile == NULL &&
      (target == NULL || (target->specialization_count == 0 &&
                          target->target_binding_count == 0))) {
    return iree_ok_status();
  }

  uint32_t error_count = 0;
  const loom_target_environment_t* target_environment =
      loomc_target_environment_loom_target_environment(
          loomc_context_target_environment(state->composition.context));
  const loomc_target_pass_environment_t* pass_environment =
      loomc_context_target_pass_environment(state->composition.context);
  loomc_diagnostic_capture_t capture;
  loomc_diagnostic_capture_initialize(
      state->diagnostics.result, /*source=*/NULL, *inout_module,
      (loom_source_resolver_t){0}, LOOM_EMITTER_PASS,
      pass_environment ? &pass_environment->diagnostic_type_print_options
                       : NULL,
      &capture);
  const iree_diagnostic_emitter_t diagnostic_emitter = {
      .fn = loomc_diagnostic_capture_emission,
      .user_data = &capture,
  };
  if (kernel_profile != NULL) {
    status =
        loomc_status_from_iree(loom_target_specialize_module_kernel_entries(
            target_environment,
            loomc_target_profile_loom_target_profile(kernel_profile),
            diagnostic_emitter, block_pool, allocator, inout_module,
            &error_count));
  } else {
    iree_arena_allocator_t arena;
    iree_arena_initialize(block_pool, &arena);
    loom_target_specialization_request_list_t requests = {0};
    loom_target_declaration_binding_list_t bindings = {0};
    if (state->specialization.indexed == NULL) {
      status = loomc_target_specialization_options_make_lists(
          target, *inout_module, &arena, &requests, &bindings);
      if (loomc_status_is_ok(status)) {
        status =
            loomc_status_from_iree(loomc_link_target_specializations_capture(
                index, materialization, &requests, bindings, &arena,
                state->arena, &state->specialization.indexed));
      }
    } else {
      status = loomc_status_from_iree(loomc_link_target_specializations_project(
          state->specialization.indexed, materialization, &arena, &requests,
          &bindings));
    }
    if (loomc_status_is_ok(status)) {
      status = loomc_status_from_iree(loom_target_specialize_module(
          target_environment, requests, bindings, diagnostic_emitter,
          block_pool, allocator, inout_module, &error_count));
    }
    iree_arena_deinitialize(&arena);
  }
  if (!loomc_status_is_ok(status)) {
    return iree_status_from_loomc(status);
  }
  if (error_count == 0) {
    return iree_ok_status();
  }
  loomc_result_set_status_code(state->diagnostics.result,
                               LOOMC_STATUS_FAILED_PRECONDITION);
  return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
}

void loomc_link_materialization_state_initialize(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_link_index_t* link_index, const loomc_config_options_t* config,
    const loomc_target_specialization_options_t* target_specialization,
    loomc_result_t* result, iree_arena_allocator_t* arena,
    loomc_allocator_t allocator,
    loomc_link_materialization_state_t* out_state) {
  *out_state = (loomc_link_materialization_state_t){
      .composition =
          {
              .context = context,
              .workspace = workspace,
              .sources =
                  {
                      .link_index = link_index,
                      .appended =
                          {
                              .provider_ordinal = IREE_HOST_SIZE_MAX,
                          },
                  },
          },
      .specialization =
          {
              .config = config,
              .target = target_specialization,
          },
      .diagnostics =
          {
              .result = result,
          },
      .arena = arena,
      .allocator = allocator,
  };
}

void loomc_link_materialization_state_initialize_overlay(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_link_index_t* library_index,
    iree_host_size_t appended_provider_ordinal,
    const loomc_source_t* appended_source, const loomc_config_options_t* config,
    const loomc_target_specialization_options_t* target_specialization,
    loomc_result_t* result, iree_arena_allocator_t* arena,
    loomc_allocator_t allocator,
    loomc_link_materialization_state_t* out_state) {
  loomc_link_materialization_state_initialize(
      context, workspace, library_index, config, target_specialization, result,
      arena, allocator, out_state);
  out_state->composition.sources.appended.source = appended_source;
  out_state->composition.sources.appended.provider_ordinal =
      appended_provider_ordinal;
}

loom_link_plan_materialization_environment_t
loomc_link_materialization_state_environment(
    loomc_link_materialization_state_t* state) {
  loom_low_repr_environment_t low_repr_environment = {0};
  loomc_target_pass_environment_initialize_low_repr_environment(
      loomc_context_target_pass_environment(state->composition.context),
      &low_repr_environment);
  return (loom_link_plan_materialization_environment_t){
      .context = loomc_context_loom_context(state->composition.context),
      .block_pool = loomc_workspace_block_pool(state->composition.workspace),
      .low_repr_environment = low_repr_environment,
      .diagnostic_sink = loomc_link_materialization_diagnostic_sink,
      .prepare_module = loomc_link_materialization_prepare_module,
      .user_data = state,
      .allocator = iree_allocator_from_loomc(state->allocator),
  };
}
