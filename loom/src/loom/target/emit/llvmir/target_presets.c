// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/target_presets.h"

bool loom_llvmir_target_profile_registry_lookup(
    const loom_llvmir_target_profile_registry_t* registry,
    iree_string_view_t profile_name,
    const loom_llvmir_target_profile_t** out_profile) {
  *out_profile = NULL;

  profile_name = iree_string_view_trim(profile_name);
  if (iree_string_view_is_empty(profile_name)) {
    *out_profile = registry->default_profile;
    return *out_profile != NULL;
  }

  for (iree_host_size_t provider_ordinal = 0;
       provider_ordinal < registry->provider_count; ++provider_ordinal) {
    const loom_llvmir_target_profile_provider_t* provider =
        registry->providers[provider_ordinal];
    for (iree_host_size_t profile_ordinal = 0;
         profile_ordinal < provider->profile_count; ++profile_ordinal) {
      const loom_llvmir_target_profile_t* profile =
          provider->profiles[profile_ordinal];
      if (iree_string_view_equal(profile_name, profile->name)) {
        *out_profile = profile;
        return true;
      }
    }
  }
  return false;
}

bool loom_llvmir_target_profile_registry_project_bundle(
    const loom_llvmir_target_profile_registry_t* registry,
    const loom_llvmir_target_profile_projection_request_t* request,
    const loom_llvmir_target_profile_t** out_profile) {
  *out_profile = NULL;
  if (!registry || !request || !request->bundle) {
    return false;
  }
  for (iree_host_size_t provider_ordinal = 0;
       provider_ordinal < registry->provider_count; ++provider_ordinal) {
    const loom_llvmir_target_profile_provider_t* provider =
        registry->providers[provider_ordinal];
    if (provider->project_bundle &&
        provider->project_bundle(request, out_profile)) {
      return true;
    }
  }
  return false;
}
