// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/tooling/runtime.h"

#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/platform/posix/api.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/io/parameter_provider.h"
#include "iree/io/scope_map.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/parameter_util.h"

static iree_status_t qwen_tooling_command_buffer_mode_from_flags(
    iree_hal_command_buffer_mode_t* out_mode) {
  bool retain_profile_metadata = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_profiling_from_flags_requires_retained_command_buffer_metadata(
          &retain_profile_metadata));
  *out_mode = retain_profile_metadata
                  ? IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA |
                        IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA
                  : IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT;
  return iree_ok_status();
}

static iree_status_t qwen_tooling_create_device_group_from_flags(
    iree_async_proactor_pool_t* proactor_pool,
    iree_async_frontier_tracker_t* frontier_tracker,
    iree_allocator_t host_allocator, iree_hal_device_group_t** out_group) {
  *out_group = NULL;
  const iree_string_view_list_t device_flags = iree_hal_device_flag_list();
  if (device_flags.count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen requires exactly one --device flag; received %" PRIhsz,
        device_flags.count);
  }

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;
  create_params.event_sink = iree_hal_device_event_sink_stderr();

  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_create_device_from_flags(
      iree_hal_available_driver_registry(), iree_string_view_empty(),
      &create_params, host_allocator, &device));

  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder, frontier_tracker);
  iree_status_t status =
      iree_hal_device_group_builder_add_device(&builder, device);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_builder_finalize(&builder, host_allocator,
                                                    out_group);
  } else {
    iree_hal_device_group_builder_deinitialize(&builder);
  }
  iree_hal_device_release(device);
  return status;
}

static iree_status_t qwen_tooling_create_parameter_source_from_flags(
    iree_allocator_t host_allocator,
    iree_io_parameter_index_t** out_parameter_index,
    iree_io_parameter_provider_t** out_parameter_provider) {
  *out_parameter_index = NULL;
  *out_parameter_provider = NULL;

  iree_io_scope_map_t scope_map;
  iree_io_scope_map_initialize(host_allocator, &scope_map);
  iree_status_t status =
      iree_tooling_build_parameter_indices_from_flags(&scope_map);
  if (iree_status_is_ok(status) && scope_map.count != 1) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen requires exactly one anonymous --parameters scope; received "
        "%" PRIhsz " scopes",
        scope_map.count);
  }

  iree_io_parameter_index_t* index = NULL;
  if (iree_status_is_ok(status)) {
    iree_io_scope_map_entry_t* entry = scope_map.entries[0];
    if (!iree_string_view_is_empty(entry->scope)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Qwen parameters must use the anonymous scope; remove the '%.*s=' "
          "prefix from --parameters",
          (int)entry->scope.size, entry->scope.data);
    } else if (iree_io_parameter_index_count(entry->index) == 0) {
      status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                "the anonymous parameter index is empty");
    } else {
      index = entry->index;
    }
  }

  iree_io_parameter_provider_t* provider = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_provider_create(
        iree_string_view_empty(), index,
        IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
        host_allocator, &provider);
  }
  if (iree_status_is_ok(status)) {
    iree_io_parameter_index_retain(index);
    *out_parameter_index = index;
    *out_parameter_provider = provider;
  } else {
    iree_io_parameter_provider_release(provider);
  }
  iree_io_scope_map_deinitialize(&scope_map);
  return status;
}

iree_status_t qwen_tooling_runtime_context_initialize_from_flags(
    iree_allocator_t host_allocator,
    qwen_tooling_runtime_context_t* out_context) {
  if (!out_context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_context must not be NULL");
  }
  memset(out_context, 0, sizeof(*out_context));
  out_context->host_allocator = host_allocator;

  iree_async_proactor_pool_options_t proactor_pool_options =
      iree_async_proactor_pool_options_default();
  proactor_pool_options.proactor_create = iree_async_proactor_create_posix;
  iree_status_t status = iree_async_proactor_pool_create(
      /*node_count=*/1, /*node_ids=*/NULL, proactor_pool_options,
      host_allocator, &out_context->proactor_pool);
  if (iree_status_is_ok(status)) {
    status = iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), host_allocator,
        &out_context->frontier_tracker);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_create_device_group_from_flags(
        out_context->proactor_pool, out_context->frontier_tracker,
        host_allocator, &out_context->device_group);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_command_buffer_mode_from_flags(
        &out_context->command_buffer_mode);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_create_parameter_source_from_flags(
        host_allocator, &out_context->parameter_index,
        &out_context->parameter_provider);
  }
  if (!iree_status_is_ok(status)) {
    qwen_tooling_runtime_context_deinitialize(out_context);
  }
  return status;
}

void qwen_tooling_runtime_context_deinitialize(
    qwen_tooling_runtime_context_t* context) {
  if (!context) return;
  iree_io_parameter_provider_release(context->parameter_provider);
  iree_io_parameter_index_release(context->parameter_index);
  iree_hal_device_group_release(context->device_group);
  iree_async_frontier_tracker_release(context->frontier_tracker);
  iree_async_proactor_pool_release(context->proactor_pool);
  memset(context, 0, sizeof(*context));
}

iree_hal_device_t* qwen_tooling_runtime_context_device(
    const qwen_tooling_runtime_context_t* context) {
  return context && context->device_group
             ? iree_hal_device_group_device_at(context->device_group, 0)
             : NULL;
}
