// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-serve/options.h"

#include "iree/base/internal/json.h"
#include "iree/base/internal/path.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/tooling/device_flags.h"

IREE_FLAG(int32_t, workers, 8,
          "Shared CPU workers for compilation, scheduling and media jobs.");
IREE_FLAG(string, devices, "",
          "Machine JSON with named devices and explicit storage pools. "
          "Mutually exclusive "
          "with --device and single-device memory/idle options.");
IREE_FLAG(string, wait_mode, "",
          "Device completion policy for all devices: active favors latency; "
          "blocking disables optional polling. Requests remain asynchronous. "
          "Overrides wait_mode in --devices; default active.");
IREE_FLAG(string, pool_backing, "",
          "elastic shares demand-committed parameter/state/workspace slabs; "
          "fixed backs retained storage eagerly and supports device address "
          "sanitization. Single-device option; default elastic.");
IREE_FLAG(string, weight_slab_bytes, "",
          "Reusable physical slab size for elastic parameters and shared "
          "workspace (default 1 GiB). Coarse slabs reduce native allocation "
          "and mapping work; loading streams through independent bounded "
          "read windows. Zero selects "
          "the allocator recommendation. Accepts byte counts or size strings "
          "such as 1gib. Ignored for fixed backing.");
IREE_FLAG(string, state_slab_bytes, "",
          "Physical commitment/reclamation granularity for elastic mutable "
          "state (default 64 MiB). Logical tensor and KV block sizes are "
          "unchanged. Zero selects the allocator recommendation. Ignored "
          "for fixed backing. Accepts byte counts or size strings such as "
          "64mib.");
IREE_FLAG(string, memory_bytes, "",
          "Shared elastic physical-backing budget; a nonzero limit "
          "requires elastic backing. Single-device shortcut; use --devices "
          "for named shared pools. Zero selects physical source capacity; a "
          "nonzero budget can tighten it. Ordinary native buffers, host "
          "snapshots, other host allocations, and driver overhead are outside "
          "this budget. "
          "Accepts byte counts or size strings such as 64gib.");
IREE_FLAG(string, workspace_idle_ms, "",
          "Single-device workspace idle expiry in milliseconds; default zero "
          "disables automatic release.");
IREE_FLAG(string, parameter_idle_ms, "",
          "Single-device unpinned parameter idle expiry in milliseconds; "
          "default zero disables automatic deactivation. Sessions and history "
          "remain resident.");

iree_status_t loom_serve_host_create_from_flags(
    const loom_runtime_compiler_options_t* compiler_options,
    iree_allocator_t host_allocator, loom_runtime_host_t** out_host) {
  return loom_runtime_host_create((iree_host_size_t)FLAG_workers,
                                  compiler_options, host_allocator, out_host);
}

static iree_status_t loom_serve_device_configuration_initialize_options_impl(
    const loom_serve_device_configuration_options_t* options,
    iree_allocator_t host_allocator,
    loom_serve_device_configuration_t* out_configuration) {
  if (options->configuration_path.size) {
    if (options->device_uris.count || options->pool_backing.size ||
        options->weight_slab_bytes.size || options->state_slab_bytes.size ||
        options->memory_bytes.size || options->workspace_idle_ms.size ||
        options->parameter_idle_ms.size) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--devices cannot be combined with --device or "
          "single-device backing, slab, memory, or idle flags");
    }
    iree_io_file_contents_t* contents = NULL;
    IREE_RETURN_IF_ERROR(iree_io_file_contents_read(options->configuration_path,
                                                    host_allocator, &contents));
    iree_status_t status = loom_serve_device_configuration_initialize(
        iree_make_string_view((const char*)contents->const_buffer.data,
                              contents->const_buffer.data_length),
        host_allocator, out_configuration);
    iree_io_file_contents_free(contents);
    IREE_RETURN_IF_ERROR(status);
  } else {
    if (!options->device_uris.count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--device is required, or provide --devices=path.json");
    }
    if (options->device_uris.count != 1) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "multiple devices require named entries in --devices=path.json");
    }
    IREE_RETURN_IF_ERROR(loom_serve_device_configuration_initialize_uri(
        options->device_uris.values[0], host_allocator, out_configuration));
    loom_runtime_pool_options_t* pool_options =
        &out_configuration->pools[0].options;
    if (iree_string_view_equal(options->pool_backing, IREE_SV("fixed"))) {
      pool_options->elastic = false;
    } else if (options->pool_backing.size &&
               !iree_string_view_equal(options->pool_backing,
                                       IREE_SV("elastic"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pool_backing must be elastic or fixed");
    }
    const iree_string_view_t values[] = {
        options->weight_slab_bytes, options->state_slab_bytes,
        options->memory_bytes, options->workspace_idle_ms,
        options->parameter_idle_ms};
    const char* names[] = {"weight_slab_bytes", "state_slab_bytes",
                           "memory_bytes", "workspace_idle_ms",
                           "parameter_idle_ms"};
    uint64_t numbers[] = {pool_options->weight_slab_size,
                          pool_options->state_slab_size, 0, 0, 0};
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(values) && iree_status_is_ok(status); ++i) {
      if (values[i].size) {
        if (i < 3) {
          iree_device_size_t size = 0;
          status = iree_string_view_parse_device_size(values[i], &size);
          numbers[i] = size;
        } else {
          status = iree_json_parse_uint64(values[i], &numbers[i]);
        }
        if (!iree_status_is_ok(status)) {
          status = iree_status_annotate_f(status, "--%s", names[i]);
        }
      }
    }
    IREE_RETURN_IF_ERROR(status);
    if (numbers[3] > INT64_MAX / 1000000 || numbers[4] > INT64_MAX / 1000000) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "pool idle interval is out of range");
    }
    if (!pool_options->elastic && numbers[2]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "a physical budget requires elastic backing");
    }
    pool_options->weight_slab_size = (iree_device_size_t)numbers[0];
    pool_options->state_slab_size = (iree_device_size_t)numbers[1];
    pool_options->memory_limit = numbers[2];
    out_configuration->idle[0] = (loom_runtime_pool_idle_policy_t){
        .workspace_timeout = (iree_duration_t)numbers[3] * 1000000,
        .parameter_timeout = (iree_duration_t)numbers[4] * 1000000};
  }
  if (options->wait_mode.size) {
    IREE_RETURN_IF_ERROR(loom_serve_device_configuration_set_wait_mode(
                             out_configuration, options->wait_mode),
                         "--wait_mode");
  }
  return iree_ok_status();
}

iree_status_t loom_serve_device_configuration_initialize_from_options(
    const loom_serve_device_configuration_options_t* options,
    iree_allocator_t host_allocator,
    loom_serve_device_configuration_t* out_configuration) {
  *out_configuration = (loom_serve_device_configuration_t){0};
  iree_status_t status =
      loom_serve_device_configuration_initialize_options_impl(
          options, host_allocator, out_configuration);
  if (!iree_status_is_ok(status)) {
    loom_serve_device_configuration_deinitialize(out_configuration);
  }
  return status;
}

iree_status_t loom_serve_device_group_allocate_from_flags(
    loom_runtime_host_t* host, iree_hal_driver_registry_t* driver_registry,
    iree_allocator_t host_allocator,
    loom_serve_device_configuration_t* out_configuration,
    loom_runtime_device_group_t** out_group,
    iree_hal_profiling_from_flags_t** out_profiling) {
  *out_group = NULL;
  *out_profiling = NULL;
  const loom_serve_device_configuration_options_t configuration_options = {
      .configuration_path = iree_make_cstring_view(FLAG_devices),
      .device_uris = iree_hal_device_flag_list(),
      .wait_mode = iree_make_cstring_view(FLAG_wait_mode),
      .pool_backing = iree_make_cstring_view(FLAG_pool_backing),
      .weight_slab_bytes = iree_make_cstring_view(FLAG_weight_slab_bytes),
      .state_slab_bytes = iree_make_cstring_view(FLAG_state_slab_bytes),
      .memory_bytes = iree_make_cstring_view(FLAG_memory_bytes),
      .workspace_idle_ms = iree_make_cstring_view(FLAG_workspace_idle_ms),
      .parameter_idle_ms = iree_make_cstring_view(FLAG_parameter_idle_ms),
  };
  IREE_RETURN_IF_ERROR(loom_serve_device_configuration_initialize_from_options(
      &configuration_options, host_allocator, out_configuration));
  bool retain_profile_metadata = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_profiling_from_flags_requires_retained_command_buffer_metadata(
          &retain_profile_metadata));
  iree_status_t status = iree_ok_status();
  struct {
    // URI scheme borrowed from decoded configuration through this function.
    iree_string_view_t name;
    // Retained configured driver released after group creation.
    iree_hal_driver_t* handle;
  } drivers[IREE_HAL_TOPOLOGY_MAX_DEVICE_COUNT] = {0};
  iree_host_size_t driver_count = 0;
  loom_runtime_device_options_t* options = out_configuration->options;
  for (iree_host_size_t i = 0;
       i < out_configuration->count && iree_status_is_ok(status); ++i) {
    const iree_string_view_t name = iree_uri_schema(options[i].uri);
    iree_host_size_t driver_index = 0;
    while (driver_index < driver_count &&
           !iree_string_view_equal(name, drivers[driver_index].name)) {
      ++driver_index;
    }
    if (driver_index == driver_count) {
      drivers[driver_index].name = name;
      if (iree_string_view_equal(name, IREE_SV("task"))) {
        drivers[driver_index].handle = loom_runtime_host_task_driver(host);
        iree_hal_driver_retain(drivers[driver_index].handle);
      } else if (!driver_registry) {
        status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                  "no linked HAL driver for URI scheme '%.*s'",
                                  (int)name.size, name.data);
      } else {
        status = iree_hal_driver_registry_acquire_driver(
            driver_registry, name, &drivers[driver_index].handle);
      }
      if (iree_status_is_ok(status)) {
        ++driver_count;
      }
    }
    options[i].driver = drivers[driver_index].handle;
    options[i].event_sink = iree_hal_device_event_sink_stderr();
    options[i].retain_profile_metadata = retain_profile_metadata;
  }
  if (iree_status_is_ok(status)) {
    status = loom_runtime_device_group_allocate(
        host, out_configuration->count, options, host_allocator, out_group);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_begin_device_group_profiling_from_flags(
        loom_runtime_device_group_handle(*out_group), host_allocator,
        out_profiling);
  }
  for (iree_host_size_t i = 0;
       i < out_configuration->pool_count && iree_status_is_ok(status); ++i) {
    const loom_serve_pool_configuration_t* declaration =
        &out_configuration->pools[i];
    loom_runtime_pool_t* pool = NULL;
    status = loom_runtime_device_group_create_pool(
        *out_group, declaration->binding, &declaration->options, &pool);
  }
  for (iree_host_size_t i = 0; i < driver_count; ++i) {
    iree_hal_driver_release(drivers[i].handle);
  }
  for (iree_host_size_t i = 0; i < out_configuration->count; ++i) {
    options[i].driver = NULL;
  }
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status, iree_hal_end_profiling_from_flags(*out_profiling));
    *out_profiling = NULL;
    status =
        iree_status_join(status, loom_runtime_device_group_free(*out_group));
    *out_group = NULL;
  }
  return status;
}
