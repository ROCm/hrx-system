// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/binding.h"

#include <string.h>

#define ID4_PIPELINE_BINDING_QUEUE_UPDATE_CHUNK_LENGTH (4 * 1024 * 1024)

static iree_hal_semaphore_list_t id4_pipeline_single_semaphore_list(
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

static iree_status_t id4_pipeline_validate_semaphore_list(
    iree_hal_semaphore_list_t semaphore_list, iree_string_view_t list_name) {
  if (semaphore_list.count == 0) return iree_ok_status();
  if (!semaphore_list.semaphores) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s semaphore array is required",
                            (int)list_name.size, list_name.data);
  }
  if (!semaphore_list.payload_values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s payload value array is required",
                            (int)list_name.size, list_name.data);
  }
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    if (!semaphore_list.semaphores[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s semaphore %" PRIhsz " is NULL",
                              (int)list_name.size, list_name.data, i);
    }
  }
  return iree_ok_status();
}

void id4_pipeline_buffer_binding_set_deinitialize(
    id4_pipeline_buffer_binding_set_t* binding_set) {
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

static iree_status_t id4_pipeline_allocate_binding_set(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_host_size_t count,
    const id4_pipeline_tensor_layout_t* (*layout_at)(const void* user_data,
                                                     iree_host_size_t index),
    const void* user_data, iree_allocator_t host_allocator,
    id4_pipeline_buffer_binding_set_t* out_binding_set) {
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
    id4_pipeline_buffer_binding_set_deinitialize(out_binding_set);
  }
  return status;
}

static const id4_pipeline_tensor_layout_t* id4_pipeline_boundary_layout_at(
    const void* user_data, iree_host_size_t index) {
  const id4_pipeline_plan_t* plan = (const id4_pipeline_plan_t*)user_data;
  const id4_pipeline_boundary_tensor_plan_t* boundary =
      id4_pipeline_plan_boundary_tensor_at(plan, index);
  return boundary ? &boundary->layout : NULL;
}

iree_status_t id4_pipeline_allocate_boundary_bindings(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, iree_allocator_t host_allocator,
    id4_pipeline_buffer_binding_set_t* out_binding_set) {
  if (!plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "boundary binding allocation requires a plan");
  }
  return id4_pipeline_allocate_binding_set(
      device, queue_affinity, id4_pipeline_plan_boundary_tensor_count(plan),
      id4_pipeline_boundary_layout_at, plan, host_allocator, out_binding_set);
}

static const id4_pipeline_tensor_layout_t*
id4_pipeline_diagnostic_tap_layout_at(const void* user_data,
                                      iree_host_size_t index) {
  const id4_pipeline_plan_t* plan = (const id4_pipeline_plan_t*)user_data;
  const id4_pipeline_diagnostic_tap_plan_t* tap =
      id4_pipeline_plan_diagnostic_tap_at(plan, index);
  return tap ? &tap->layout : NULL;
}

iree_status_t id4_pipeline_allocate_diagnostic_tap_bindings(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, iree_allocator_t host_allocator,
    id4_pipeline_buffer_binding_set_t* out_binding_set) {
  if (!plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "diagnostic tap allocation requires a plan");
  }
  return id4_pipeline_allocate_binding_set(
      device, queue_affinity, id4_pipeline_plan_diagnostic_tap_count(plan),
      id4_pipeline_diagnostic_tap_layout_at, plan, host_allocator,
      out_binding_set);
}

iree_status_t id4_pipeline_find_boundary_binding(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_buffer_binding_set_t* binding_set,
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

iree_status_t id4_pipeline_find_diagnostic_tap_binding(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_buffer_binding_set_t* binding_set,
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

static iree_status_t id4_pipeline_validate_replacement_binding(
    const id4_pipeline_tensor_layout_t* layout,
    iree_hal_buffer_binding_t replacement) {
  if (!replacement.buffer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "replacement binding for boundary tensor `%.*s` has no buffer",
        (int)layout->name.size, layout->name.data);
  }
  if (replacement.length < layout->byte_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "replacement binding for boundary tensor `%.*s` has length %" PRIu64
        " but requires %" PRIu64,
        (int)layout->name.size, layout->name.data, (uint64_t)replacement.length,
        (uint64_t)layout->byte_length);
  }
  const iree_device_size_t alignment =
      layout->alignment ? layout->alignment : 1;
  if ((replacement.offset % alignment) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "replacement binding for boundary tensor `%.*s` has offset %" PRIu64
        " that is not aligned to %" PRIu64,
        (int)layout->name.size, layout->name.data, (uint64_t)replacement.offset,
        (uint64_t)alignment);
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_replace_boundary_binding(
    const id4_pipeline_plan_t* plan,
    id4_pipeline_buffer_binding_set_t* binding_set, iree_string_view_t name,
    iree_hal_buffer_binding_t replacement) {
  if (!plan || !binding_set) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "boundary binding replacement requires plan and "
                            "set");
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
    if (!boundary || !iree_string_view_equal(boundary->layout.name, name)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(id4_pipeline_validate_replacement_binding(
        &boundary->layout, replacement));
    iree_hal_buffer_retain(replacement.buffer);
    iree_hal_buffer_release(binding_set->buffers[i]);
    binding_set->buffers[i] = replacement.buffer;
    binding_set->bindings[i] = replacement;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "boundary tensor `%.*s` not found", (int)name.size,
                          name.data);
}

iree_status_t id4_pipeline_queue_update_binding(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_buffer_binding_t* binding, const void* source_data,
    iree_host_size_t source_length,
    iree_hal_semaphore_list_t initial_wait_semaphore_list,
    iree_hal_semaphore_t* semaphore, uint64_t* inout_payload_value) {
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
  IREE_RETURN_IF_ERROR(id4_pipeline_validate_semaphore_list(
      initial_wait_semaphore_list, IREE_SV("queue update initial wait")));
  if (initial_wait_semaphore_list.count != 0 && *inout_payload_value != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "queue update initial waits require a zero starting payload");
  }
  iree_host_size_t source_offset = 0;
  while (source_offset < source_length) {
    const iree_host_size_t remaining_length = source_length - source_offset;
    const iree_host_size_t chunk_length =
        remaining_length > ID4_PIPELINE_BINDING_QUEUE_UPDATE_CHUNK_LENGTH
            ? ID4_PIPELINE_BINDING_QUEUE_UPDATE_CHUNK_LENGTH
            : remaining_length;
    iree_hal_semaphore_list_t wait_list = iree_hal_semaphore_list_empty();
    iree_hal_semaphore_t* wait_semaphore = NULL;
    uint64_t wait_payload_value = *inout_payload_value;
    if (source_offset == 0 && wait_payload_value == 0) {
      wait_list = initial_wait_semaphore_list;
    } else if (wait_payload_value != 0) {
      wait_list = id4_pipeline_single_semaphore_list(
          &wait_semaphore, &wait_payload_value, semaphore, wait_payload_value);
    }
    iree_hal_semaphore_t* signal_semaphore = NULL;
    uint64_t signal_payload_value = wait_payload_value + 1;
    iree_hal_semaphore_list_t signal_list = id4_pipeline_single_semaphore_list(
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
