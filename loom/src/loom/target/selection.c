// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/selection.h"

static bool loom_target_selection_bundle_is_complete(
    const loom_target_bundle_t* bundle) {
  return bundle != NULL && bundle->snapshot != NULL &&
         bundle->export_plan != NULL && bundle->config != NULL;
}

iree_status_t loom_target_specification_parse(
    iree_string_view_t value, loom_target_specification_t* out_specification) {
  IREE_ASSERT_ARGUMENT(out_specification);
  *out_specification = (loom_target_specification_t){0};

  const iree_string_view_t specification = iree_string_view_trim(value);
  iree_string_view_t family = iree_string_view_empty();
  iree_string_view_t selector = iree_string_view_empty();
  if (iree_string_view_split(specification, ':', &family, &selector) < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target '%.*s' must use family:selector syntax",
                            (int)specification.size, specification.data);
  }
  family = iree_string_view_trim(family);
  selector = iree_string_view_trim(selector);
  if (iree_string_view_is_empty(family) ||
      iree_string_view_is_empty(selector)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target '%.*s' must name a non-empty family and selector",
        (int)specification.size, specification.data);
  }

  *out_specification = (loom_target_specification_t){
      .family = family,
      .selector = selector,
  };
  return iree_ok_status();
}

iree_status_t loom_target_environment_select_profile(
    const loom_target_environment_t* environment,
    const loom_target_specification_t* specification,
    const loom_target_profile_t** out_profile) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(specification);
  IREE_ASSERT_ARGUMENT(out_profile);
  *out_profile = NULL;

  const loom_target_provider_t* selected_provider = NULL;
  const loom_target_provider_set_t* provider_set = environment->provider_set;
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_target_provider_t* provider = provider_set->providers[i];
    if (provider->profile_type != NULL &&
        iree_string_view_equal(provider->profile_type->name,
                               specification->family)) {
      selected_provider = provider;
      break;
    }
  }
  if (selected_provider == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target family '%.*s' is not available in this binary",
        (int)specification->family.size, specification->family.data);
  }
  if (selected_provider->select_profile == NULL) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "target family '%.*s' does not expose named profiles",
        (int)specification->family.size, specification->family.data);
  }

  const loom_target_profile_t* profile = NULL;
  IREE_RETURN_IF_ERROR(
      selected_provider->select_profile(specification->selector, &profile));
  if (profile == NULL || profile->type != selected_provider->profile_type ||
      !loom_target_selection_bundle_is_complete(profile->target_bundle)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "target family '%.*s' selected an invalid profile",
                            (int)specification->family.size,
                            specification->family.data);
  }
  *out_profile = profile;
  return iree_ok_status();
}

iree_status_t loom_target_environment_select_cpu_profile(
    const loom_target_environment_t* environment,
    const iree_cpu_data_t* cpu_data, const loom_target_facts_t* requirement,
    const loom_target_profile_t* profile, iree_allocator_t allocator,
    loom_target_profile_selection_t* out_selection) {
  *out_selection = (loom_target_profile_selection_t){0};
  const loom_target_provider_set_t* providers = environment->provider_set;
  loom_target_profile_selection_t selected = {0};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < providers->provider_count; ++i) {
    const loom_target_provider_t* provider = providers->providers[i];
    if (!provider->select_cpu_profile) {
      continue;
    }
    loom_target_profile_selection_t candidate = {0};
    status = provider->select_cpu_profile(cpu_data, requirement, profile,
                                          allocator, &candidate);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (candidate.profile == NULL) {
      continue;
    }
    if (selected.profile != NULL) {
      loom_target_profile_selection_release(&candidate, allocator);
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "multiple compiler providers accept this CPU");
      break;
    }
    selected = candidate;
  }
  if (iree_status_is_ok(status) && selected.profile == NULL) {
    const iree_string_view_t architecture =
        iree_cpu_architecture_name(cpu_data->architecture);
    const iree_string_view_t requested =
        profile       ? profile->target_bundle->name
        : requirement ? loom_target_facts_identity_name(requirement)
                      : IREE_SV("automatic");
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "no native CPU profile for '%.*s' satisfies target '%.*s'",
        (int)architecture.size, architecture.data, (int)requested.size,
        requested.data);
  }
  if (iree_status_is_ok(status)) {
    *out_selection = selected;
    selected = (loom_target_profile_selection_t){0};
  }
  loom_target_profile_selection_release(&selected, allocator);
  return status;
}
