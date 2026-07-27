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
  if (provider->execution_backend_count != 0 &&
      provider->execution_backends == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "loom execution provider '%.*s' has no execution backend table",
        (int)provider->name.size, provider->name.data);
  }
  for (iree_host_size_t i = 0; i < provider->execution_backend_count; ++i) {
    const loom_run_execution_backend_t* backend =
        provider->execution_backends[i];
    if (backend == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "loom execution provider '%.*s' has null execution backend %" PRIhsz,
          (int)provider->name.size, provider->name.data, i);
    }
    if (iree_string_view_is_empty(iree_string_view_trim(backend->name))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "loom execution provider '%.*s' has unnamed "
                              "execution backend %" PRIhsz,
                              (int)provider->name.size, provider->name.data, i);
    }
    if (backend->run_one_shot == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "loom execution backend '%.*s' has no one-shot "
                              "run hook",
                              (int)backend->name.size, backend->name.data);
    }
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

static iree_status_t
loom_run_execution_provider_verify_unique_execution_backend_name(
    const loom_run_execution_provider_set_t* provider_set,
    iree_host_size_t provider_index, iree_host_size_t backend_index) {
  const loom_run_execution_backend_t* backend =
      provider_set->providers[provider_index]
          ->execution_backends[backend_index];
  for (iree_host_size_t i = 0; i <= provider_index; ++i) {
    const loom_run_execution_provider_t* provider = provider_set->providers[i];
    const iree_host_size_t end =
        i == provider_index ? backend_index : provider->execution_backend_count;
    for (iree_host_size_t j = 0; j < end; ++j) {
      const loom_run_execution_backend_t* existing =
          provider->execution_backends[j];
      if (iree_string_view_equal(existing->name, backend->name)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate loom execution backend '%.*s'",
                                (int)backend->name.size, backend->name.data);
      }
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
    for (iree_host_size_t j = 0; j < provider->execution_backend_count; ++j) {
      IREE_RETURN_IF_ERROR(
          loom_run_execution_provider_verify_unique_execution_backend_name(
              provider_set, i, j));
    }
  }
  return iree_ok_status();
}
