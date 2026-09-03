// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "link_materialization.h"

#include "config.h"
#include "context.h"
#include "diagnostic.h"
#include "link_index.h"
#include "loom/codegen/low/repr.h"
#include "loom/target/module_specialization.h"
#include "loomc/iree.h"
#include "target.h"
#include "workspace.h"

static iree_status_t loomc_link_materialization_capture_diagnostic(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loomc_link_materialization_state_t* state =
      (loomc_link_materialization_state_t*)user_data;
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic(
      state->diagnostics.result, state->diagnostics.source, diagnostic));
}

static iree_status_t loomc_link_materialization_capture_emission(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loomc_link_materialization_state_t* state =
      (loomc_link_materialization_state_t*)user_data;
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic_emission(
      state->diagnostics.result, /*source=*/NULL, LOOM_EMITTER_PASS, emission));
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

static iree_status_t loomc_link_materialization_prepare_module(
    void* user_data, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_module_t** inout_module) {
  loomc_link_materialization_state_t* state =
      (loomc_link_materialization_state_t*)user_data;
  const loomc_config_apply_text_to_module_options_t apply_options = {
      .config = state->specialization.config,
      .module = *inout_module,
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
  if (target == NULL || (target->specialization_count == 0 &&
                         target->target_binding_count == 0)) {
    return iree_ok_status();
  }

  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  loom_target_specialization_request_list_t requests = {0};
  loom_target_declaration_binding_list_t bindings = {0};
  status = loomc_target_specialization_options_make_lists(
      target, LOOMC_TARGET_SPECIALIZATION_LIST_FLAG_NONE, &arena, &requests,
      &bindings);
  uint32_t error_count = 0;
  if (loomc_status_is_ok(status)) {
    status = loomc_status_from_iree(loom_target_specialize_module(
        loomc_target_environment_loom_target_environment(
            loomc_context_target_environment(state->composition.context)),
        requests, bindings,
        (iree_diagnostic_emitter_t){
            .fn = loomc_link_materialization_capture_emission,
            .user_data = state,
        },
        block_pool, allocator, inout_module, &error_count));
  }
  iree_arena_deinitialize(&arena);
  if (!loomc_status_is_ok(status)) {
    return iree_status_from_loomc(status);
  }
  if (error_count == 0) {
    return iree_ok_status();
  }
  status = loomc_result_set_state(state->diagnostics.result,
                                  LOOMC_RESULT_STATE_FAILED);
  if (!loomc_status_is_ok(status)) {
    return iree_status_from_loomc(status);
  }
  return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
}

void loomc_link_materialization_state_initialize(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_link_index_t* link_index, const loomc_config_options_t* config,
    const loomc_target_specialization_options_t* target_specialization,
    loomc_result_t* result, loomc_allocator_t allocator,
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
      .allocator = allocator,
  };
}

void loomc_link_materialization_state_initialize_overlay(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_link_index_t* library_index,
    iree_host_size_t appended_provider_ordinal,
    const loomc_source_t* appended_source, const loomc_config_options_t* config,
    const loomc_target_specialization_options_t* target_specialization,
    loomc_result_t* result, loomc_allocator_t allocator,
    loomc_link_materialization_state_t* out_state) {
  loomc_link_materialization_state_initialize(context, workspace, library_index,
                                              config, target_specialization,
                                              result, allocator, out_state);
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
