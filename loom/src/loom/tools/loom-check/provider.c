// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/provider.h"

#include <stdio.h>

#include "loom/tools/loom-check/main.h"
#include "loom/tools/loom-check/source_low.h"
#include "loom/transforms/cleanup/configured.h"
#include "loomc/interop.h"

enum {
  LOOM_CHECK_PROVIDER_EMIT_PROVIDER_CAPACITY = 64,
  LOOM_CHECK_PROVIDER_REQUIREMENT_PROVIDER_CAPACITY = 64,
};

typedef struct loom_check_provider_environment_state_t {
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
  out_state->emit_providers[out_state->emit_provider_count++] =
      &loom_check_source_low_emit_provider;

  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_check_provider_t* provider = provider_set->providers[i];
    IREE_RETURN_IF_ERROR(
        loom_check_provider_append_emit_providers(out_state, provider));
    IREE_RETURN_IF_ERROR(
        loom_check_provider_append_requirement_providers(out_state, provider));
  }
  return iree_ok_status();
}

int loom_check_provider_main(int argc, char** argv,
                             const loom_check_provider_set_t* provider_set,
                             loom_input_format_list_t input_formats) {
  loom_check_provider_environment_state_t state;
  iree_status_t status =
      loom_check_provider_environment_state_initialize(provider_set, &state);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }

  const loom_check_environment_t environment = {
      .input_formats = input_formats,
      .target_environment = loomc_target_environment_get_interop_view(
          provider_set->target_environment),
      .cleanup_pattern_provider_set =
          loom_cleanup_configured_pattern_provider_set(),
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
  return loom_check_main(argc, argv, &environment,
                         provider_set->target_environment,
                         provider_set->compile_provider);
}
