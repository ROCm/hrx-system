// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/tooling/runtime.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "experimental/id4/kernels/embedded_loom_sources.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/io/scope_map.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/parameter_util.h"

static iree_status_t id4_tooling_runtime_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_tooling_runtime_validate_context_options(
    const id4_tooling_runtime_context_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ID4 runtime context options are required");
  }
  IREE_RETURN_IF_ERROR(id4_tooling_runtime_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("ID4 runtime context")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "ID4 runtime context extension structures are not supported");
  }
  if (iree_string_view_is_empty(options->executable_cache_identifier)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "ID4 runtime context executable cache identifier is required");
  }
  return iree_ok_status();
}

static iree_status_t id4_tooling_command_buffer_mode_from_flags(
    iree_hal_command_buffer_mode_t* out_mode) {
  bool retain_profile_metadata = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_profiling_from_flags_requires_retained_command_buffer_metadata(
          &retain_profile_metadata));
  *out_mode = retain_profile_metadata
                  ? IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA
                  : IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT;
  return iree_ok_status();
}

static iree_status_t id4_tooling_create_device_group_from_flags(
    iree_async_proactor_pool_t* proactor_pool,
    iree_async_frontier_tracker_t* frontier_tracker,
    iree_allocator_t host_allocator, iree_hal_device_group_t** out_group) {
  *out_group = NULL;
  if (iree_hal_device_flag_list().count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "ID4 runtime requires at least one --device= flag; use --list_devices "
        "to inspect available devices");
  }

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;

  iree_hal_device_list_t* device_list = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_create_devices_from_flags(
      iree_hal_available_driver_registry(), iree_string_view_empty(),
      &create_params, host_allocator, &device_list));

  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder, frontier_tracker);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < device_list->count && iree_status_is_ok(status); ++i) {
    status = iree_hal_device_group_builder_add_device(
        &builder, iree_hal_device_list_at(device_list, i));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_builder_finalize(&builder, host_allocator,
                                                    out_group);
  } else {
    iree_hal_device_group_builder_deinitialize(&builder);
  }
  iree_hal_device_list_free(device_list);
  return status;
}

iree_status_t id4_tooling_runtime_context_initialize_from_flags(
    const id4_tooling_runtime_context_options_t* options,
    iree_allocator_t host_allocator,
    id4_tooling_runtime_context_t* out_context) {
  IREE_ASSERT_ARGUMENT(out_context);
  memset(out_context, 0, sizeof(*out_context));
  IREE_RETURN_IF_ERROR(id4_tooling_runtime_validate_context_options(options));
  out_context->host_allocator = host_allocator;

  iree_status_t status = iree_async_proactor_pool_create(
      /*node_count=*/1, /*node_ids=*/NULL,
      iree_async_proactor_pool_options_default(), host_allocator,
      &out_context->proactor_pool);
  if (iree_status_is_ok(status)) {
    iree_async_frontier_tracker_options_t tracker_options =
        iree_async_frontier_tracker_options_default();
    status = iree_async_frontier_tracker_create(tracker_options, host_allocator,
                                                &out_context->frontier_tracker);
  }
  if (iree_status_is_ok(status)) {
    status = id4_tooling_create_device_group_from_flags(
        out_context->proactor_pool, out_context->frontier_tracker,
        host_allocator, &out_context->device_group);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_device_t* primary_device =
        id4_tooling_runtime_context_primary_device(out_context);
    status = iree_hal_executable_cache_create(
        primary_device, options->executable_cache_identifier,
        &out_context->executable_cache);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_kernel_cache_create_options_t kernel_cache_options;
    memset(&kernel_cache_options, 0, sizeof(kernel_cache_options));
    kernel_cache_options.structure_size = sizeof(kernel_cache_options);
    kernel_cache_options.target_processor =
        id4_pipeline_kernel_cache_default_target_processor();
    status = id4_pipeline_kernel_cache_create(
        &kernel_cache_options, host_allocator, &out_context->kernel_cache);
  }
  if (iree_status_is_ok(status)) {
    status = id4_tooling_command_buffer_mode_from_flags(
        &out_context->command_buffer_mode);
  }
  if (!iree_status_is_ok(status)) {
    id4_tooling_runtime_context_deinitialize(out_context);
  }
  return status;
}

void id4_tooling_runtime_context_deinitialize(
    id4_tooling_runtime_context_t* context) {
  if (!context) return;
  id4_pipeline_kernel_cache_release(context->kernel_cache);
  iree_hal_executable_cache_release(context->executable_cache);
  iree_hal_device_group_release(context->device_group);
  iree_async_frontier_tracker_release(context->frontier_tracker);
  iree_async_proactor_pool_release(context->proactor_pool);
  memset(context, 0, sizeof(*context));
}

iree_hal_device_t* id4_tooling_runtime_context_primary_device(
    const id4_tooling_runtime_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(context->device_group);
  return iree_hal_device_group_device_at(context->device_group, 0);
}

id4_pipeline_stage_services_t id4_tooling_runtime_context_stage_services(
    const id4_tooling_runtime_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  id4_pipeline_stage_services_t services = {
      // Device group made from standard --device= flags.
      .device_group = context->device_group,
      // Executable cache shared by prepared stage bundles.
      .executable_cache = context->executable_cache,
      // Host allocator used by stage planning and preparation.
      .host_allocator = context->host_allocator,
  };
  return services;
}

iree_status_t id4_tooling_create_embedded_kernel_library(
    iree_allocator_t host_allocator,
    id4_pipeline_kernel_library_t** out_library) {
  IREE_ASSERT_ARGUMENT(out_library);
  *out_library = NULL;

  const iree_file_toc_t* toc = id4_kernel_embedded_loom_sources_create();
  const iree_host_size_t file_count = id4_kernel_embedded_loom_sources_size();
  id4_pipeline_kernel_source_file_t* source_files = NULL;
  iree_status_t status = iree_ok_status();
  if (file_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, file_count,
                                         sizeof(source_files[0]),
                                         (void**)&source_files);
  }
  for (iree_host_size_t i = 0; i < file_count && iree_status_is_ok(status);
       ++i) {
    source_files[i] = (id4_pipeline_kernel_source_file_t){
        // Source identifier formatted as <module_path>.loom.
        .source_identifier = iree_make_cstring_view(toc[i].name),
        // Embedded Loom source payload.
        .source_contents = iree_make_const_byte_span(toc[i].data, toc[i].size),
    };
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_kernel_library_create_from_source_files(
        file_count, source_files, host_allocator, out_library);
  }
  iree_allocator_free(host_allocator, source_files);
  return status;
}

iree_status_t id4_tooling_create_parameter_provider_from_flags(
    iree_string_view_t scope, iree_allocator_t host_allocator,
    iree_io_parameter_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = NULL;

  iree_io_scope_map_t scope_map;
  iree_io_scope_map_initialize(host_allocator, &scope_map);
  iree_status_t status =
      iree_tooling_build_parameter_indices_from_flags(&scope_map);

  iree_io_parameter_index_t* index = NULL;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < scope_map.count; ++i) {
      iree_io_scope_map_entry_t* entry = scope_map.entries[i];
      if (iree_string_view_equal(entry->scope, scope)) {
        index = entry->index;
        break;
      }
    }
    if (!index) {
      if (iree_string_view_is_empty(scope)) {
        status = iree_make_status(
            IREE_STATUS_NOT_FOUND,
            "required anonymous parameter scope was not loaded; pass "
            "--parameters=<file>");
      } else {
        status = iree_make_status(
            IREE_STATUS_NOT_FOUND,
            "required parameter scope `%.*s` was not loaded; pass "
            "--parameters=%.*s=<file>",
            (int)scope.size, scope.data, (int)scope.size, scope.data);
      }
    }
  }
  if (iree_status_is_ok(status) && iree_io_parameter_index_count(index) == 0) {
    if (iree_string_view_is_empty(scope)) {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "anonymous parameter scope was loaded with no parameters");
    } else {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "parameter scope `%.*s` was loaded with no parameters",
          (int)scope.size, scope.data);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_provider_create(
        scope, index,
        IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
        host_allocator, out_provider);
  }
  iree_io_scope_map_deinitialize(&scope_map);
  return status;
}
