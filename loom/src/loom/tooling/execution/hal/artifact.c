// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/artifact.h"

void loom_run_hal_artifact_provider_registry_initialize_from_entries(
    const loom_run_hal_artifact_provider_t* const* providers,
    iree_host_size_t provider_count,
    loom_run_hal_artifact_provider_registry_t* out_registry) {
  IREE_ASSERT_ARGUMENT(out_registry);
  *out_registry = (loom_run_hal_artifact_provider_registry_t){
      .providers = providers,
      .provider_count = provider_count,
  };
}

const loom_run_hal_artifact_provider_t*
loom_run_hal_artifact_provider_registry_lookup(
    const loom_run_hal_artifact_provider_registry_t* registry,
    iree_string_view_t name) {
  IREE_ASSERT_ARGUMENT(registry);
  for (iree_host_size_t i = 0; i < registry->provider_count; ++i) {
    const loom_run_hal_artifact_provider_t* provider = registry->providers[i];
    if (iree_string_view_equal(provider->name, name)) {
      return provider;
    }
  }
  return NULL;
}

iree_status_t loom_run_hal_artifact_provider_registry_format_names(
    const loom_run_hal_artifact_provider_registry_t* registry,
    iree_string_builder_t* output) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(output);
  iree_host_size_t appended_count = 0;
  for (iree_host_size_t i = 0; i < registry->provider_count; ++i) {
    const loom_run_hal_artifact_provider_t* provider = registry->providers[i];
    if (appended_count > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, ", "));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(output, provider->name));
    ++appended_count;
  }
  return iree_ok_status();
}
