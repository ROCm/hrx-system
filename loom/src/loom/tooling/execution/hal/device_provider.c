// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/device_provider.h"

static iree_status_t loom_device_provider_validate_selected_target(
    const loom_device_provider_t* provider,
    const loom_device_target_t* target) {
  if (target->executable_target == NULL ||
      iree_string_view_is_empty(target->executable_target->target_key)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "device provider '%.*s' returned no executable target identity",
        (int)provider->artifact_provider->name.size,
        provider->artifact_provider->name.data);
  }
  if (loom_device_target_bundle(target) == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "device provider '%.*s' returned no complete target bundle",
        (int)provider->artifact_provider->name.size,
        provider->artifact_provider->name.data);
  }
  return iree_ok_status();
}

iree_status_t loom_device_provider_select_target(
    const loom_device_provider_t* provider,
    const struct loom_run_hal_runtime_t* runtime,
    loom_device_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = (loom_device_target_t){0};

  if (provider->artifact_provider == NULL || provider->select_target == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "device provider has no complete preferred target selection hook");
  }
  iree_status_t status = provider->select_target(provider, runtime, out_target);
  if (iree_status_is_ok(status)) {
    status =
        loom_device_provider_validate_selected_target(provider, out_target);
  }
  if (!iree_status_is_ok(status)) {
    *out_target = (loom_device_target_t){0};
  }
  return status;
}

iree_status_t loom_device_provider_select_compatible_target(
    const loom_device_provider_t* provider,
    const struct loom_run_hal_runtime_t* runtime,
    const loom_target_facts_t* target_requirement,
    loom_device_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = (loom_device_target_t){0};

  if (provider->artifact_provider == NULL ||
      provider->select_compatible_target == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "device provider has no complete compatible target selection hook");
  }

  iree_status_t status = provider->select_compatible_target(
      provider, runtime, target_requirement, out_target);
  if (iree_status_is_ok(status)) {
    status =
        loom_device_provider_validate_selected_target(provider, out_target);
  }
  if (!iree_status_is_ok(status)) {
    *out_target = (loom_device_target_t){0};
  }
  return status;
}

static iree_status_t loom_device_target_profile_project_facts(
    const loom_target_profile_t* base_profile, iree_arena_allocator_t* arena,
    loom_target_facts_t* out_facts) {
  const loom_device_target_profile_t* profile =
      (const loom_device_target_profile_t*)base_profile;
  return profile->provider->project_target_facts(
      profile->provider, profile->runtime, profile->target, arena, out_facts);
}

iree_status_t loom_device_target_profile_initialize(
    const loom_device_provider_t* provider,
    const struct loom_run_hal_runtime_t* runtime,
    const loom_device_target_t* target,
    loom_device_target_profile_t* out_profile) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(out_profile);
  *out_profile = (loom_device_target_profile_t){0};
  if (provider->artifact_provider == NULL ||
      provider->artifact_provider->target_profile_type == NULL ||
      provider->artifact_provider->target_profile_type->fact_type == NULL ||
      provider->project_target_facts == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "device provider has no complete target fact projection");
  }
  if (loom_device_target_bundle(target) == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "selected device target has no complete bundle");
  }
  if (iree_string_view_is_empty(loom_device_target_key(target))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "selected device target has no executable target identity");
  }
  const loom_target_profile_type_t* family_type =
      provider->artifact_provider->target_profile_type;
  out_profile->type = (loom_target_profile_type_t){
      .name = family_type->name,
      .fact_type = family_type->fact_type,
      .project_facts = loom_device_target_profile_project_facts,
  };
  out_profile->base = (loom_target_profile_t){
      .type = &out_profile->type,
      .target_bundle = loom_device_target_bundle(target),
  };
  out_profile->provider = provider;
  out_profile->runtime = runtime;
  out_profile->target = target;
  return iree_ok_status();
}

void loom_device_provider_registry_initialize_from_entries(
    const loom_device_provider_t* const* providers,
    iree_host_size_t provider_count,
    loom_device_provider_registry_t* out_registry) {
  IREE_ASSERT_ARGUMENT(out_registry);
  *out_registry = (loom_device_provider_registry_t){
      .providers = providers,
      .provider_count = provider_count,
  };
}

const loom_device_provider_t* loom_device_provider_registry_lookup(
    const loom_device_provider_registry_t* registry, iree_string_view_t name) {
  IREE_ASSERT_ARGUMENT(registry);
  for (iree_host_size_t i = 0; i < registry->provider_count; ++i) {
    const loom_device_provider_t* provider = registry->providers[i];
    if (iree_string_view_equal(provider->artifact_provider->name, name)) {
      return provider;
    }
  }
  return NULL;
}

iree_status_t loom_device_provider_registry_format_driver_names(
    const loom_device_provider_registry_t* registry,
    iree_string_builder_t* output) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(output);
  for (iree_host_size_t i = 0; i < registry->provider_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, ", "));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        output, registry->providers[i]->driver_name));
  }
  return iree_ok_status();
}
