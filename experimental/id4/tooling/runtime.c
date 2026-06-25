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

#define ID4_TOOLING_RUNTIME_QUEUE_UPDATE_CHUNK_LENGTH (4 * 1024 * 1024)

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

static iree_hal_semaphore_list_t id4_tooling_single_semaphore_list(
    iree_hal_semaphore_t** semaphore_storage, uint64_t* payload_storage,
    iree_hal_semaphore_t* semaphore, uint64_t payload_value) {
  *semaphore_storage = semaphore;
  *payload_storage = payload_value;
  iree_hal_semaphore_list_t list = {
      // One semaphore edge in this stack-backed list.
      .count = 1,
      // Stack-backed semaphore handle.
      .semaphores = semaphore_storage,
      // Stack-backed payload value.
      .payload_values = payload_storage,
  };
  return list;
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

void id4_tooling_buffer_binding_set_deinitialize(
    id4_tooling_buffer_binding_set_t* binding_set) {
  if (!binding_set) return;
  if (binding_set->buffers) {
    for (iree_host_size_t i = 0; i < binding_set->count; ++i) {
      iree_hal_buffer_release(binding_set->buffers[i]);
    }
  }
  iree_allocator_t host_allocator = binding_set->host_allocator;
  iree_allocator_free(host_allocator, binding_set->buffers);
  iree_allocator_free(host_allocator, binding_set->bindings);
  memset(binding_set, 0, sizeof(*binding_set));
}

static iree_status_t id4_tooling_allocate_binding_set(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_host_size_t count,
    const id4_pipeline_tensor_layout_t* (*layout_at)(const void* user_data,
                                                     iree_host_size_t index),
    const void* user_data, iree_allocator_t host_allocator,
    id4_tooling_buffer_binding_set_t* out_binding_set) {
  IREE_ASSERT_ARGUMENT(out_binding_set);
  memset(out_binding_set, 0, sizeof(*out_binding_set));
  if (!device) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "binding allocation requires a device");
  }
  out_binding_set->host_allocator = host_allocator;
  out_binding_set->count = count;
  if (count == 0) return iree_ok_status();

  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, count, sizeof(out_binding_set->buffers[0]),
      (void**)&out_binding_set->buffers);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, count,
                                         sizeof(out_binding_set->bindings[0]),
                                         (void**)&out_binding_set->bindings);
  }
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const id4_pipeline_tensor_layout_t* layout = layout_at(user_data, i);
    if (!layout) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "binding layout %" PRIhsz " is missing", i);
      break;
    }
    if (layout->byte_length == 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "binding layout `%.*s` has zero byte length",
                                (int)layout->name.size, layout->name.data);
      break;
    }
    iree_hal_buffer_params_t params;
    memset(&params, 0, sizeof(params));
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
                   IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE;
    params.queue_affinity = queue_affinity;
    params.min_alignment = layout->alignment ? layout->alignment : 1;
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), params, layout->byte_length,
        &out_binding_set->buffers[i]);
    if (iree_status_is_ok(status)) {
      out_binding_set->bindings[i] = (iree_hal_buffer_binding_t){
          // Tensor buffer supplied in plan order.
          .buffer = out_binding_set->buffers[i],
          // Standalone allocation starts at byte zero.
          .offset = 0,
          // Full planned tensor byte range.
          .length = layout->byte_length,
      };
    }
  }
  if (!iree_status_is_ok(status)) {
    id4_tooling_buffer_binding_set_deinitialize(out_binding_set);
  }
  return status;
}

static const id4_pipeline_tensor_layout_t* id4_tooling_boundary_layout_at(
    const void* user_data, iree_host_size_t index) {
  const id4_pipeline_plan_t* plan = (const id4_pipeline_plan_t*)user_data;
  const id4_pipeline_boundary_tensor_plan_t* boundary =
      id4_pipeline_plan_boundary_tensor_at(plan, index);
  return boundary ? &boundary->layout : NULL;
}

iree_status_t id4_tooling_allocate_boundary_bindings(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, iree_allocator_t host_allocator,
    id4_tooling_buffer_binding_set_t* out_binding_set) {
  if (!plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "boundary binding allocation requires a plan");
  }
  return id4_tooling_allocate_binding_set(
      device, queue_affinity, id4_pipeline_plan_boundary_tensor_count(plan),
      id4_tooling_boundary_layout_at, plan, host_allocator, out_binding_set);
}

static const id4_pipeline_tensor_layout_t* id4_tooling_diagnostic_tap_layout_at(
    const void* user_data, iree_host_size_t index) {
  const id4_pipeline_plan_t* plan = (const id4_pipeline_plan_t*)user_data;
  const id4_pipeline_diagnostic_tap_plan_t* tap =
      id4_pipeline_plan_diagnostic_tap_at(plan, index);
  return tap ? &tap->layout : NULL;
}

iree_status_t id4_tooling_allocate_diagnostic_tap_bindings(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, iree_allocator_t host_allocator,
    id4_tooling_buffer_binding_set_t* out_binding_set) {
  if (!plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "diagnostic tap allocation requires a plan");
  }
  return id4_tooling_allocate_binding_set(
      device, queue_affinity, id4_pipeline_plan_diagnostic_tap_count(plan),
      id4_tooling_diagnostic_tap_layout_at, plan, host_allocator,
      out_binding_set);
}

iree_status_t id4_tooling_find_boundary_binding(
    const id4_pipeline_plan_t* plan,
    const id4_tooling_buffer_binding_set_t* binding_set,
    iree_string_view_t name, iree_hal_buffer_binding_t* out_binding) {
  IREE_ASSERT_ARGUMENT(out_binding);
  if (!plan || !binding_set) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "boundary binding lookup requires plan and set");
  }
  const iree_host_size_t boundary_count =
      id4_pipeline_plan_boundary_tensor_count(plan);
  if (binding_set->count != boundary_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "boundary binding set has %" PRIhsz
                            " entries but plan has %" PRIhsz
                            " boundary tensors",
                            binding_set->count, boundary_count);
  }
  for (iree_host_size_t i = 0; i < boundary_count; ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (boundary && iree_string_view_equal(boundary->layout.name, name)) {
      *out_binding = binding_set->bindings[i];
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "boundary tensor `%.*s` not found", (int)name.size,
                          name.data);
}

iree_status_t id4_tooling_find_diagnostic_tap_binding(
    const id4_pipeline_plan_t* plan,
    const id4_tooling_buffer_binding_set_t* binding_set,
    iree_string_view_t name, iree_hal_buffer_binding_t* out_binding) {
  IREE_ASSERT_ARGUMENT(out_binding);
  if (!plan || !binding_set) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "diagnostic tap binding lookup requires plan and set");
  }
  const iree_host_size_t tap_count =
      id4_pipeline_plan_diagnostic_tap_count(plan);
  if (binding_set->count != tap_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "diagnostic tap binding set has %" PRIhsz
                            " entries but plan has %" PRIhsz " diagnostic taps",
                            binding_set->count, tap_count);
  }
  for (iree_host_size_t i = 0; i < tap_count; ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* tap =
        id4_pipeline_plan_diagnostic_tap_at(plan, i);
    if (tap && iree_string_view_equal(tap->name, name)) {
      *out_binding = binding_set->bindings[i];
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "diagnostic tap `%.*s` not found", (int)name.size,
                          name.data);
}

iree_status_t id4_tooling_queue_update_binding(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_buffer_binding_t* binding, const void* source_data,
    iree_host_size_t source_length, iree_hal_semaphore_t* semaphore,
    uint64_t* inout_payload_value) {
  if (!device || !binding || !source_data || !semaphore ||
      !inout_payload_value) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "queue update requires device, binding, source, semaphore, and value");
  }
  if (source_length != binding->length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source length %" PRIhsz
                            " does not match binding length %" PRIu64,
                            source_length, (uint64_t)binding->length);
  }
  if (source_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue update requires a non-empty binding");
  }
  iree_host_size_t source_offset = 0;
  while (source_offset < source_length) {
    const iree_host_size_t remaining_length = source_length - source_offset;
    const iree_host_size_t chunk_length =
        remaining_length > ID4_TOOLING_RUNTIME_QUEUE_UPDATE_CHUNK_LENGTH
            ? ID4_TOOLING_RUNTIME_QUEUE_UPDATE_CHUNK_LENGTH
            : remaining_length;
    iree_hal_semaphore_list_t wait_list = iree_hal_semaphore_list_empty();
    iree_hal_semaphore_t* wait_semaphore = NULL;
    uint64_t wait_payload_value = *inout_payload_value;
    if (wait_payload_value != 0) {
      wait_list = id4_tooling_single_semaphore_list(
          &wait_semaphore, &wait_payload_value, semaphore, wait_payload_value);
    }
    iree_hal_semaphore_t* signal_semaphore = NULL;
    uint64_t signal_payload_value = wait_payload_value + 1;
    iree_hal_semaphore_list_t signal_list = id4_tooling_single_semaphore_list(
        &signal_semaphore, &signal_payload_value, semaphore,
        signal_payload_value);
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_update(
        device, queue_affinity, wait_list, signal_list, source_data,
        source_offset, binding->buffer, binding->offset + source_offset,
        chunk_length, IREE_HAL_UPDATE_FLAG_NONE));
    *inout_payload_value = signal_payload_value;
    source_offset += chunk_length;
  }
  return iree_ok_status();
}
