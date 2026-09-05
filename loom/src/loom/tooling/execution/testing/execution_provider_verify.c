// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/testing/execution_provider_verify.h"

#include <inttypes.h>

static iree_status_t loom_run_execution_provider_verify(
    const loom_run_execution_provider_t* provider,
    iree_host_size_t provider_index) {
  if (provider == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "loom execution provider %" PRIhsz " is null",
                            provider_index);
  }
  if (iree_string_view_is_empty(iree_string_view_trim(provider->name))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "loom execution provider %" PRIhsz " has no name",
                            provider_index);
  }
  const loom_device_provider_t* device_provider = provider->device_provider;
  if (device_provider == NULL) {
    return iree_ok_status();
  }
  if (device_provider->artifact_provider == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "loom execution provider '%.*s' has a device provider without an "
        "artifact provider",
        (int)provider->name.size, provider->name.data);
  }
  if (iree_string_view_is_empty(
          iree_string_view_trim(device_provider->driver_name))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "loom execution provider '%.*s' has an unnamed HAL driver",
        (int)provider->name.size, provider->name.data);
  }
  if (device_provider->select_target == NULL ||
      device_provider->select_compatible_target == NULL ||
      device_provider->project_target_facts == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "loom execution provider '%.*s' has an incomplete device provider",
        (int)provider->name.size, provider->name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_run_execution_provider_verify_unique_name(
    const loom_run_execution_provider_set_t* provider_set,
    iree_host_size_t provider_index) {
  const loom_run_execution_provider_t* provider =
      provider_set->providers[provider_index];
  for (iree_host_size_t i = 0; i < provider_index; ++i) {
    const loom_run_execution_provider_t* existing = provider_set->providers[i];
    if (iree_string_view_equal(existing->name, provider->name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate loom execution provider '%.*s'",
                              (int)provider->name.size, provider->name.data);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_run_execution_provider_verify_unique_driver_name(
    const loom_run_execution_provider_set_t* provider_set,
    iree_host_size_t provider_index) {
  const loom_device_provider_t* device_provider =
      provider_set->providers[provider_index]->device_provider;
  if (device_provider == NULL) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < provider_index; ++i) {
    const loom_device_provider_t* existing =
        provider_set->providers[i]->device_provider;
    if (existing != NULL &&
        iree_string_view_equal(existing->driver_name,
                               device_provider->driver_name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "duplicate Loom device provider for HAL driver '%.*s'",
          (int)device_provider->driver_name.size,
          device_provider->driver_name.data);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_run_execution_provider_set_verify(
    const loom_run_execution_provider_set_t* provider_set) {
  if (provider_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "loom execution provider set is required");
  }
  if (provider_set->provider_count != 0 && provider_set->providers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "loom execution provider table is required");
  }
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_run_execution_provider_t* provider = provider_set->providers[i];
    IREE_RETURN_IF_ERROR(loom_run_execution_provider_verify(provider, i));
    IREE_RETURN_IF_ERROR(
        loom_run_execution_provider_verify_unique_name(provider_set, i));
    IREE_RETURN_IF_ERROR(
        loom_run_execution_provider_verify_unique_driver_name(provider_set, i));
  }
  return iree_ok_status();
}
