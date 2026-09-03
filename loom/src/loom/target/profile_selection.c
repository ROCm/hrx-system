// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/profile_selection.h"

#include "loom/target/provider.h"

iree_status_t loom_target_profile_specification_parse(
    iree_string_view_t value,
    loom_target_profile_specification_t* out_specification) {
  IREE_ASSERT_ARGUMENT(out_specification);
  *out_specification = (loom_target_profile_specification_t){0};

  value = iree_string_view_trim(value);
  iree_string_view_t family = iree_string_view_empty();
  iree_string_view_t selector = iree_string_view_empty();
  if (iree_string_view_split(value, ':', &family, &selector) < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target profile '%.*s' must use family:selector syntax",
        (int)value.size, value.data);
  }
  family = iree_string_view_trim(family);
  selector = iree_string_view_trim(selector);
  if (iree_string_view_is_empty(family) ||
      iree_string_view_is_empty(selector)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target profile '%.*s' must name a non-empty family and selector",
        (int)value.size, value.data);
  }

  *out_specification = (loom_target_profile_specification_t){
      .family = family,
      .selector = selector,
  };
  return iree_ok_status();
}

void loom_target_profile_selection_deinitialize(
    loom_target_profile_selection_t* selection) {
  if (selection == NULL) {
    return;
  }
  if (selection->provider != NULL) {
    IREE_ASSERT(selection->provider->release_profile_selection != NULL);
    selection->provider->release_profile_selection(selection->provider,
                                                   selection);
  }
  *selection = (loom_target_profile_selection_t){0};
}

iree_status_t loom_target_provider_select_profile(
    const loom_target_provider_t* provider, iree_string_view_t selector,
    iree_allocator_t allocator,
    loom_target_profile_selection_t* out_selection) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(out_selection);
  *out_selection = (loom_target_profile_selection_t){0};

  selector = iree_string_view_trim(selector);
  if (iree_string_view_is_empty(selector)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target profile selector must not be empty");
  }
  if (provider->profile_type == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "target provider has no profile family");
  }
  if (provider->select_profile == NULL) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "target family '%.*s' does not support public profile selection",
        (int)provider->profile_type->name.size,
        provider->profile_type->name.data);
  }
  if (provider->release_profile_selection == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "target family '%.*s' has no profile selection release hook",
        (int)provider->profile_type->name.size,
        provider->profile_type->name.data);
  }

  loom_target_profile_selection_t selection = {
      .provider = provider,
      .allocator = allocator,
  };
  iree_status_t status =
      provider->select_profile(provider, selector, allocator, &selection);
  selection.provider = provider;
  selection.allocator = allocator;
  if (iree_status_is_ok(status) &&
      (selection.profile == NULL || selection.profile->type == NULL ||
       selection.profile->type != provider->profile_type ||
       iree_string_view_is_empty(selection.selector))) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "target family '%.*s' returned an invalid profile selection",
        (int)provider->profile_type->name.size,
        provider->profile_type->name.data);
  }
  if (iree_status_is_ok(status)) {
    *out_selection = selection;
  } else if (selection.storage != NULL) {
    selection.provider = provider;
    selection.allocator = allocator;
    loom_target_profile_selection_deinitialize(&selection);
  }
  return status;
}

iree_status_t loom_target_environment_select_profile(
    const loom_target_environment_t* environment, iree_string_view_t value,
    iree_allocator_t allocator,
    loom_target_profile_selection_t* out_selection) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(out_selection);
  *out_selection = (loom_target_profile_selection_t){0};

  loom_target_profile_specification_t specification = {0};
  IREE_RETURN_IF_ERROR(
      loom_target_profile_specification_parse(value, &specification));
  const loom_target_provider_t* provider =
      loom_target_environment_lookup_family_provider(environment,
                                                     specification.family);
  if (provider == NULL) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "target profile family '%.*s' is not linked into this compiler",
        (int)specification.family.size, specification.family.data);
  }
  return loom_target_provider_select_profile(provider, specification.selector,
                                             allocator, out_selection);
}
