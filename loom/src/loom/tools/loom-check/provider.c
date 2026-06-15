// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/provider.h"

#include <stdio.h>

#include "loom/tooling/context/context.h"
#include "loom/tools/loom-check/main.h"

enum {
  LOOM_CHECK_PROVIDER_TARGET_PROVIDER_CAPACITY = 64,
  LOOM_CHECK_PROVIDER_EMIT_PROVIDER_CAPACITY = 64,
  LOOM_CHECK_PROVIDER_REQUIREMENT_PROVIDER_CAPACITY = 64,
};

typedef struct loom_check_provider_environment_state_t {
  // Core target provider table assembled once for the environment.
  const loom_target_provider_t*
      target_providers[LOOM_CHECK_PROVIDER_TARGET_PROVIDER_CAPACITY];
  // Number of entries in |target_providers|.
  iree_host_size_t target_provider_count;
  // Provider-set view over |target_providers|.
  loom_target_provider_set_t target_provider_set;
  // Core target environment composed from |target_providers|.
  loom_target_environment_t target_environment;
  // Emit provider table assembled once for the environment.
  const loom_check_emit_provider_t*
      emit_providers[LOOM_CHECK_PROVIDER_EMIT_PROVIDER_CAPACITY];
  // Number of entries in |emit_providers|.
  iree_host_size_t emit_provider_count;
  // Requirement provider table assembled once for the environment.
  const loom_check_requirement_provider_t*
      requirement_providers[LOOM_CHECK_PROVIDER_REQUIREMENT_PROVIDER_CAPACITY];
  // Number of entries in |requirement_providers|.
  iree_host_size_t requirement_provider_count;
} loom_check_provider_environment_state_t;

static iree_status_t loom_check_provider_append_target_provider(
    loom_check_provider_environment_state_t* state,
    const loom_check_provider_t* provider) {
  if (provider->target_provider == NULL) {
    return iree_ok_status();
  }
  if (state->target_provider_count >= IREE_ARRAYSIZE(state->target_providers)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "loom-check target provider capacity exceeded");
  }
  state->target_providers[state->target_provider_count++] =
      provider->target_provider;
  return iree_ok_status();
}

static iree_status_t loom_check_provider_append_emit_providers(
    loom_check_provider_environment_state_t* state,
    const loom_check_provider_t* provider) {
  if (state->emit_provider_count + provider->emit_provider_count >
      IREE_ARRAYSIZE(state->emit_providers)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "loom-check emit provider capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider->emit_provider_count; ++i) {
    state->emit_providers[state->emit_provider_count++] =
        provider->emit_providers[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_check_provider_append_requirement_providers(
    loom_check_provider_environment_state_t* state,
    const loom_check_provider_t* provider) {
  if (state->requirement_provider_count + provider->requirement_provider_count >
      IREE_ARRAYSIZE(state->requirement_providers)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "loom-check requirement provider capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider->requirement_provider_count; ++i) {
    state->requirement_providers[state->requirement_provider_count++] =
        provider->requirement_providers[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_check_provider_environment_state_initialize(
    const loom_check_provider_set_t* provider_set,
    loom_check_provider_environment_state_t* out_state) {
  *out_state = (loom_check_provider_environment_state_t){0};

  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_check_provider_t* provider = provider_set->providers[i];
    IREE_RETURN_IF_ERROR(
        loom_check_provider_append_target_provider(out_state, provider));
    IREE_RETURN_IF_ERROR(
        loom_check_provider_append_emit_providers(out_state, provider));
    IREE_RETURN_IF_ERROR(
        loom_check_provider_append_requirement_providers(out_state, provider));
  }
  out_state->target_provider_set = loom_target_provider_set_make(
      out_state->target_providers, out_state->target_provider_count);
  IREE_RETURN_IF_ERROR(loom_target_environment_initialize(
      &out_state->target_provider_set, &out_state->target_environment));
  return iree_ok_status();
}

static iree_status_t loom_check_provider_register_context(
    void* user_data, loom_context_t* context) {
  loom_check_provider_environment_state_t* state =
      (loom_check_provider_environment_state_t*)user_data;
  return loom_tooling_context_register_tool_dialects_with_target_environment(
      &state->target_environment, context);
}

static iree_status_t loom_check_provider_initialize_low_descriptor_registry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  loom_check_provider_environment_state_t* state =
      (loom_check_provider_environment_state_t*)user_data;
  return loom_target_environment_initialize_low_descriptor_registry(
      &state->target_environment, out_registry);
}

static iree_status_t loom_check_provider_initialize_low_lower_policy_registry(
    void* user_data, loom_low_lower_policy_registry_t* out_registry) {
  loom_check_provider_environment_state_t* state =
      (loom_check_provider_environment_state_t*)user_data;
  return loom_target_environment_initialize_low_lower_policy_registry(
      &state->target_environment, out_registry);
}

static iree_status_t loom_check_provider_initialize_math_policy_registry(
    void* user_data, loom_target_math_policy_registry_t* out_registry) {
  loom_check_provider_environment_state_t* state =
      (loom_check_provider_environment_state_t*)user_data;
  return loom_target_environment_initialize_math_policy_registry(
      &state->target_environment, out_registry);
}

int loom_check_provider_main(int argc, char** argv,
                             const loom_check_provider_set_t* provider_set) {
  loom_check_provider_environment_state_t state;
  iree_status_t status =
      loom_check_provider_environment_state_initialize(provider_set, &state);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }

  const loom_check_environment_t environment = {
      .register_context =
          {
              .fn = loom_check_provider_register_context,
              .user_data = &state,
          },
      .target_environment = &state.target_environment,
      .initialize_low_descriptor_registry =
          {
              .fn = loom_check_provider_initialize_low_descriptor_registry,
              .user_data = &state,
          },
      .initialize_low_lower_policy_registry =
          {
              .fn = loom_check_provider_initialize_low_lower_policy_registry,
              .user_data = &state,
          },
      .initialize_math_policy_registry =
          {
              .fn = loom_check_provider_initialize_math_policy_registry,
              .user_data = &state,
          },
      .pass_registry =
          loom_target_environment_pass_registry(&state.target_environment),
      .low_legality_provider_list =
          loom_target_environment_low_legality_provider_list(
              &state.target_environment),
      .legalizer_provider_list =
          loom_target_environment_legalizer_provider_list(
              &state.target_environment),
      .low_packet_diagnostic_provider_list =
          loom_target_environment_low_packet_diagnostic_provider_list(
              &state.target_environment),
      .low_asm_diagnostic_provider_list =
          loom_target_environment_low_asm_diagnostic_provider_list(
              &state.target_environment),
      .low_verify_provider_list =
          loom_target_environment_low_verify_provider_list(
              &state.target_environment),
      .emit_providers =
          {
              .providers = state.emit_providers,
              .provider_count = state.emit_provider_count,
          },
      .requirement_providers =
          {
              .providers = state.requirement_providers,
              .provider_count = state.requirement_provider_count,
          },
  };
  return loom_check_main(argc, argv, &environment);
}
