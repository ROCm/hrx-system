// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-serve/execution_options.h"

#include "iree/base/internal/json.h"

static iree_status_t loom_serve_execution_options_resource(
    iree_string_view_t value, iree_host_size_t device_count,
    uint32_t* out_index) {
  if (iree_string_view_equal(value, IREE_SV("host"))) {
    *out_index = 0;
    return iree_ok_status();
  }
  uint64_t index = 0;
  IREE_RETURN_IF_ERROR(iree_status_annotate(
      iree_json_parse_uint64(value, &index),
      IREE_SV("execution resource requires host or a --device ordinal")));
  if (index >= device_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "execution resource '%.*s' must be host or a --device ordinal in "
        "[0, %" PRIhsz ")",
        (int)value.size, value.data, device_count);
  }
  *out_index = (uint32_t)index + 1;
  return iree_ok_status();
}

static iree_status_t loom_serve_execution_options_limits(
    iree_host_size_t device_count, iree_string_view_list_t values,
    uint32_t* max_active) {
  uint64_t seen = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < values.count && iree_status_is_ok(status);
       ++i) {
    iree_string_view_t resource, count;
    iree_string_view_split(values.values[i], ':', &resource, &count);
    uint32_t index = 0;
    status =
        loom_serve_execution_options_resource(resource, device_count, &index);
    if (iree_status_is_ok(status)) {
      uint64_t limit = 0;
      status = iree_status_annotate(
          iree_json_parse_uint64(count, &limit),
          IREE_SV("execution limit requires RESOURCE:COUNT"));
      if (iree_status_is_ok(status)) {
        if (limit > UINT32_MAX) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "execution limit requires RESOURCE:COUNT "
                                    "with COUNT in [0, %" PRIu32 "]",
                                    UINT32_MAX);
        } else if (seen & (UINT64_C(1) << index)) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "duplicate execution limit for '%.*s'",
                                    (int)resource.size, resource.data);
        } else {
          seen |= UINT64_C(1) << index;
          max_active[index] = (uint32_t)limit;
        }
      }
    }
  }
  return status;
}

static iree_status_t loom_serve_execution_options_scope(
    iree_host_size_t device_count, iree_string_view_t value,
    loom_serve_execution_scope_t* out_scope) {
  iree_string_view_t mode, resources;
  iree_string_view_split(value, ':', &mode, &resources);
  if (iree_string_view_equal(mode, IREE_SV("exclude"))) {
    out_scope->coexistence = LOOM_SERVE_EXECUTION_COEXISTENCE_EXCLUDE;
  } else if (iree_string_view_equal(mode, IREE_SV("overlap"))) {
    out_scope->coexistence = LOOM_SERVE_EXECUTION_COEXISTENCE_OVERLAP;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "execution scope requires exclude:RESOURCES or overlap:RESOURCES");
  }
  iree_status_t status = iree_ok_status();
  bool more = true;
  while (more && iree_status_is_ok(status)) {
    iree_string_view_t resource;
    more = iree_string_view_split(resources, ',', &resource, &resources) != -1;
    uint32_t index = 0;
    status =
        loom_serve_execution_options_resource(resource, device_count, &index);
    if (iree_status_is_ok(status)) {
      const uint64_t bit = UINT64_C(1) << index;
      if (out_scope->resources & bit) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "duplicate execution scope resource '%.*s'",
                                  (int)resource.size, resource.data);
      } else {
        out_scope->resources |= bit;
      }
    }
  }
  return status;
}

iree_status_t loom_serve_execution_options_initialize(
    iree_host_size_t device_count, iree_string_view_list_t limits,
    iree_string_view_list_t scopes, iree_allocator_t host_allocator,
    loom_serve_execution_options_t* out_options) {
  *out_options =
      (loom_serve_execution_options_t){.host_allocator = host_allocator};
  if (!device_count ||
      device_count >= IREE_ARRAYSIZE(out_options->max_active)) {
    *out_options = (loom_serve_execution_options_t){0};
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "execution policy supports 1 to 63 devices");
  }
  iree_status_t status = loom_serve_execution_options_limits(
      device_count, limits, out_options->max_active);
  const iree_host_size_t scope_count = iree_max(scopes.count, 1);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, scope_count,
                                         sizeof(*out_options->scopes),
                                         (void**)&out_options->scopes);
  }
  for (iree_host_size_t i = 0; i < scopes.count && iree_status_is_ok(status);
       ++i) {
    status = loom_serve_execution_options_scope(device_count, scopes.values[i],
                                                &out_options->scopes[i]);
  }
  if (iree_status_is_ok(status)) {
    if (!scopes.count) {
      out_options->scopes[0] = (loom_serve_execution_scope_t){
          .resources = UINT64_MAX >> (63 - device_count),
          .coexistence = LOOM_SERVE_EXECUTION_COEXISTENCE_EXCLUDE};
    }
    out_options->policy =
        (loom_serve_execution_policy_t){.max_active = out_options->max_active,
                                        .scope_count = scope_count,
                                        .scopes = out_options->scopes};
  } else {
    loom_serve_execution_options_deinitialize(out_options);
  }
  return status;
}

void loom_serve_execution_options_deinitialize(
    loom_serve_execution_options_t* options) {
  iree_allocator_free(options->host_allocator, options->scopes);
  *options = (loom_serve_execution_options_t){0};
}
