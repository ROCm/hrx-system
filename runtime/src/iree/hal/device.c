// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/device.h"

#include <inttypes.h>
#include <string.h>

#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/detail.h"
#include "iree/hal/resource.h"

//===----------------------------------------------------------------------===//
// iree_hal_device_t
//===----------------------------------------------------------------------===//

#define _VTABLE_DISPATCH(device, method_name) \
  IREE_HAL_VTABLE_DISPATCH(device, iree_hal_device, method_name)

IREE_HAL_API_RETAIN_RELEASE(device);

IREE_API_EXPORT iree_status_t iree_hal_device_create_params_verify(
    const iree_hal_device_create_params_t* params) {
  IREE_ASSERT_ARGUMENT(params);
  if (IREE_UNLIKELY(!params->proactor_pool)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL device creation requires a valid proactor pool");
  }
  if (IREE_UNLIKELY(!iree_hal_device_event_sink_is_valid(params->event_sink))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL device creation requires a valid device event sink");
  }
  if (IREE_UNLIKELY(params->runtime_features &
                    ~IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAGS_KNOWN)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL device creation requested unknown runtime "
                            "feature flags 0x%" PRIx64,
                            params->runtime_features &
                                ~IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAGS_KNOWN);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_string_view_t
iree_hal_device_id(iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  return _VTABLE_DISPATCH(device, id)(device);
}

IREE_API_EXPORT iree_allocator_t
iree_hal_device_host_allocator(iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  return _VTABLE_DISPATCH(device, host_allocator)(device);
}

IREE_API_EXPORT iree_hal_allocator_t* iree_hal_device_allocator(
    iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  return _VTABLE_DISPATCH(device, device_allocator)(device);
}

IREE_API_EXPORT void iree_hal_device_replace_channel_provider(
    iree_hal_device_t* device, iree_hal_channel_provider_t* new_provider) {
  IREE_ASSERT_ARGUMENT(device);
  _VTABLE_DISPATCH(device, replace_channel_provider)(device, new_provider);
}

IREE_API_EXPORT
iree_status_t iree_hal_device_trim(iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(device, trim)(device);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT const iree_hal_device_spec_t* iree_hal_device_spec(
    iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  return _VTABLE_DISPATCH(device, device_spec)(device);
}

IREE_API_EXPORT const iree_hal_queue_family_t* iree_hal_device_queue_family(
    iree_hal_device_t* device, iree_hal_queue_family_ordinal_t family_ordinal) {
  IREE_ASSERT_ARGUMENT(device);
  return _VTABLE_DISPATCH(device, queue_family)(device, family_ordinal);
}

IREE_API_EXPORT iree_hal_queue_t* iree_hal_device_queue(
    iree_hal_device_t* device, iree_hal_queue_family_ordinal_t family_ordinal,
    iree_hal_queue_ordinal_t queue_ordinal) {
  IREE_ASSERT_ARGUMENT(device);
  return _VTABLE_DISPATCH(device, queue)(device, family_ordinal, queue_ordinal);
}

IREE_API_EXPORT void iree_hal_device_observation_initialize(
    iree_hal_device_observation_flags_t requested_flags,
    iree_hal_device_observation_t* out_observation) {
  IREE_ASSERT_ARGUMENT(out_observation);
  memset(out_observation, 0, sizeof(*out_observation));
  out_observation->requested_flags = requested_flags;
  out_observation->sample_time_ns = iree_time_now();
}

IREE_API_EXPORT void iree_hal_device_observation_set_memory_total(
    iree_device_size_t total_bytes,
    iree_hal_device_observation_t* out_observation) {
  IREE_ASSERT_ARGUMENT(out_observation);
  out_observation->provided_flags |= IREE_HAL_DEVICE_OBSERVATION_FLAG_MEMORY;
  out_observation->memory.flags |=
      IREE_HAL_DEVICE_MEMORY_OBSERVATION_FLAG_TOTAL_BYTES;
  out_observation->memory.total_bytes = total_bytes;
}

IREE_API_EXPORT void iree_hal_device_observation_set_memory_available(
    iree_device_size_t available_bytes,
    iree_hal_device_observation_t* out_observation) {
  IREE_ASSERT_ARGUMENT(out_observation);
  out_observation->provided_flags |= IREE_HAL_DEVICE_OBSERVATION_FLAG_MEMORY;
  out_observation->memory.flags |=
      IREE_HAL_DEVICE_MEMORY_OBSERVATION_FLAG_AVAILABLE_BYTES;
  out_observation->memory.available_bytes = available_bytes;
}

IREE_API_EXPORT iree_status_t
iree_hal_device_observation_populate_memory_total_from_spec(
    const iree_hal_device_spec_t* device_spec,
    iree_hal_device_observation_t* out_observation) {
  IREE_ASSERT_ARGUMENT(out_observation);
  if (!device_spec) return iree_ok_status();

  const iree_hal_device_memory_spec_t* memory =
      iree_hal_device_spec_memory(device_spec);
  iree_device_size_t total_bytes = 0;
  bool has_known_capacity = false;
  for (iree_host_size_t i = 0; i < memory->heap_count; ++i) {
    const iree_hal_memory_heap_spec_t* heap = &memory->heaps[i];
    if (iree_all_bits_set(heap->flags,
                          IREE_HAL_MEMORY_HEAP_SPEC_FLAG_CAPACITY_UNKNOWN)) {
      continue;
    }
    if (IREE_UNLIKELY(heap->capacity_bytes > IREE_DEVICE_SIZE_MAX)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "memory heap capacity %" PRIu64
                              " exceeds the representable device size",
                              heap->capacity_bytes);
    }
    if (IREE_UNLIKELY(!iree_device_size_checked_add(
            total_bytes, (iree_device_size_t)heap->capacity_bytes,
            &total_bytes))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "memory heap capacity sum overflowed");
    }
    has_known_capacity = true;
  }
  if (has_known_capacity) {
    iree_hal_device_observation_set_memory_total(total_bytes, out_observation);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_device_sample_observation(
    iree_hal_device_t* device,
    iree_hal_device_observation_flags_t requested_flags,
    iree_hal_device_observation_t* out_observation) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_observation);
  if (IREE_UNLIKELY(iree_any_bit_set(requested_flags,
                                     ~IREE_HAL_DEVICE_OBSERVATION_FLAG_ALL))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported device observation flags 0x%016" PRIx64, requested_flags);
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_device_observation_initialize(requested_flags, out_observation);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, _VTABLE_DISPATCH(device, sample_observation)(device, requested_flags,
                                                       out_observation));
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT const iree_hal_device_topology_info_t*
iree_hal_device_topology_info(iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  return _VTABLE_DISPATCH(device, topology_info)(device);
}

IREE_API_EXPORT iree_status_t iree_hal_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  IREE_ASSERT_ARGUMENT(src_device);
  IREE_ASSERT_ARGUMENT(dst_device);
  IREE_ASSERT_ARGUMENT(edge);
  return _VTABLE_DISPATCH(src_device, refine_topology_edge)(src_device,
                                                            dst_device, edge);
}

IREE_API_EXPORT iree_status_t iree_hal_device_assign_topology_info(
    iree_hal_device_t* device,
    const iree_hal_device_topology_info_t* topology_info) {
  IREE_ASSERT_ARGUMENT(device);
  return _VTABLE_DISPATCH(device, assign_topology_info)(device, topology_info);
}

IREE_API_EXPORT iree_hal_semaphore_compatibility_t
iree_hal_device_query_semaphore_compatibility(iree_hal_device_t* device,
                                              iree_hal_semaphore_t* semaphore) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(semaphore);
  return _VTABLE_DISPATCH(device, query_semaphore_compatibility)(device,
                                                                 semaphore);
}

IREE_API_EXPORT iree_status_t iree_hal_device_query_queue_pool_backend(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_pool_backend_t* out_backend) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_backend);
  memset(out_backend, 0, sizeof(*out_backend));
  const iree_hal_device_topology_info_t* topology_info =
      iree_hal_device_topology_info(device);
  if (!topology_info->topology || !topology_info->frontier.tracker) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "device queue pool backends are unavailable before the device is "
        "assigned to a device group");
  }
  return _VTABLE_DISPATCH(device, query_queue_pool_backend)(
      device, queue_affinity, out_backend);
}

IREE_API_EXPORT iree_status_t iree_hal_device_load_executable(
    iree_hal_device_t* device, const iree_hal_queue_family_t* queue_family,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* params,
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(queue_family);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(out_executable);
  const iree_hal_queue_family_ordinal_t queue_family_ordinal =
      iree_hal_queue_family_ordinal(queue_family);
  const iree_hal_device_queue_spec_t* queue_spec =
      iree_hal_device_spec_queues(iree_hal_device_spec(device));
  if (IREE_UNLIKELY(queue_family_ordinal >= queue_spec->family_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable queue family ordinal %u is invalid",
                            queue_family_ordinal);
  }
  if (IREE_UNLIKELY(iree_hal_device_queue_family(
                        device, queue_family_ordinal) != queue_family)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "executable queue family %u must be borrowed from the device",
        queue_family_ordinal);
  }
  if (IREE_UNLIKELY(!iree_any_bit_set(
          queue_spec->families[queue_family_ordinal].role_flags,
          IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue family %u cannot load dispatch executables",
                            queue_family_ordinal);
  }
  if (IREE_UNLIKELY(iree_hal_device_spec_executable_target_ordinal(
                        iree_hal_device_spec(device), target) ==
                    IREE_HOST_SIZE_MAX)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "executable target must be borrowed from the device spec");
  }
  const iree_hal_queue_family_spec_t* queue_family_spec =
      &queue_spec->families[queue_family_ordinal];
  if (IREE_UNLIKELY(
          !iree_all_bits_set(target->physical_device_affinity,
                             queue_family_spec->physical_device_affinity))) {
    return iree_make_status(
        IREE_STATUS_INCOMPATIBLE,
        "executable target `%.*s:%.*s` physical-device affinity 0x%016" PRIx64
        " does not cover queue family %u affinity 0x%016" PRIx64,
        (int)target->family.size, target->family.data,
        (int)target->target_key.size, target->target_key.data,
        target->physical_device_affinity, queue_family_ordinal,
        queue_family_spec->physical_device_affinity);
  }
  if (IREE_UNLIKELY(iree_const_byte_span_is_empty(params->executable_data))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable data must not be empty");
  }
  if (IREE_UNLIKELY(!params->executable_data.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable data pointer must not be NULL");
  }
  if (IREE_UNLIKELY(params->constant_count != 0 && !params->constants)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "executable constants are required when constant count is nonzero");
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_executable_t* executable = NULL;
  iree_status_t status = _VTABLE_DISPATCH(device, load_executable)(
      device, queue_family, target, params, &executable);
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(!executable || iree_hal_executable_queue_family(
                                       executable) != queue_family)) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "device executable loader did not return an executable for queue "
        "family %u",
        queue_family_ordinal);
  }
  if (iree_status_is_ok(status)) {
    *out_executable = executable;
  } else {
    iree_hal_executable_release(executable);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_wait_semaphores(
    iree_hal_device_t* device, iree_async_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout,
    iree_async_wait_flags_t flags) {
  IREE_ASSERT_ARGUMENT(device);
  if (semaphore_list.count == 0) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  // HAL semaphores embed async semaphores at offset 0 (toll-free bridge).
  iree_status_t status = iree_async_semaphore_multi_wait(
      wait_mode, (iree_async_semaphore_t**)semaphore_list.semaphores,
      semaphore_list.payload_values, semaphore_list.count, timeout, flags,
      iree_allocator_system());
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_device_profiling_begin(
    iree_hal_device_t* device,
    const iree_hal_device_profiling_options_t* options) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(options);

  const iree_hal_device_profiling_flags_t supported_flags =
      IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS;
  if (iree_any_bit_set(options->flags, ~supported_flags)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile option flags 0x%x",
                            options->flags & ~supported_flags);
  }
  const iree_hal_profile_capture_filter_flags_t supported_filter_flags =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN |
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_BUFFER_ID |
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_INDEX |
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_PHYSICAL_DEVICE_ORDINAL |
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_QUEUE_ORDINAL;
  if (iree_any_bit_set(options->capture_filter.flags,
                       ~supported_filter_flags)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported profile capture filter bits 0x%x",
        options->capture_filter.flags & ~supported_filter_flags);
  }
  if (options->capture_filter.reserved0 != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "profile capture filter reserved fields must be zero");
  }
  if (iree_any_bit_set(
          options->capture_filter.flags,
          IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN) &&
      iree_string_view_is_empty(
          options->capture_filter.executable_function_pattern)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "profile capture executable function filter must not be empty");
  }
  if (iree_any_bit_set(
          options->capture_filter.flags,
          IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_BUFFER_ID) &&
      options->capture_filter.command_buffer_id == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "profile capture command buffer filter must be nonzero");
  }

  if (options->counter_set_count != 0) {
    if (!options->counter_sets) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "hardware counter set selections require a counter_sets array");
    }
    if (!iree_hal_device_profiling_options_requests_counters(options)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "hardware counter set selections require a counter profiling data "
          "family");
    }
    for (iree_host_size_t i = 0; i < options->counter_set_count; ++i) {
      const iree_hal_profile_counter_set_selection_t* counter_set =
          &options->counter_sets[i];
      if (counter_set->counter_name_count == 0) {
        continue;
      }
      if (!counter_set->counter_names) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "hardware counter set %" PRIhsz
                                " names counters but has no counter_names "
                                "array",
                                i);
      }
      for (iree_host_size_t j = 0; j < counter_set->counter_name_count; ++j) {
        iree_string_view_t counter_name = counter_set->counter_names[j];
        if (iree_string_view_is_empty(counter_name) || !counter_name.data) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "hardware counter set %" PRIhsz
              " has an empty counter name at index %" PRIhsz,
              i, j);
        }
      }
    }
  }

  const bool data_requested =
      options->data_families != IREE_HAL_DEVICE_PROFILING_DATA_NONE ||
      iree_hal_device_profiling_options_requests_lightweight_statistics(
          options);
  if (data_requested && !options->sink) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL-native profiling with requested data requires a profile sink");
  }
  if (!data_requested) {
    return iree_ok_status();
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      _VTABLE_DISPATCH(device, profiling_begin)(device, options);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_device_profiling_flush(iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(device, profiling_flush)(device);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_device_profiling_end(iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(device, profiling_end)(device);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_device_list_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_device_list_allocate(
    iree_host_size_t capacity, iree_allocator_t host_allocator,
    iree_hal_device_list_t** out_list) {
  IREE_ASSERT_ARGUMENT(out_list);
  *out_list = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_device_list_t* list = NULL;
  iree_host_size_t total_size =
      sizeof(*list) + capacity * sizeof(list->devices[0]);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&list));
  list->host_allocator = host_allocator;
  list->capacity = capacity;
  list->count = 0;
  *out_list = list;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT void iree_hal_device_list_free(iree_hal_device_list_t* list) {
  if (!list) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = list->host_allocator;
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    iree_hal_device_release(list->devices[i]);
  }
  iree_allocator_free(host_allocator, list);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_hal_device_list_push_back(
    iree_hal_device_list_t* list, iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(list);
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();
  if (list->count + 1 <= list->capacity) {
    iree_hal_device_retain(device);
    list->devices[list->count++] = device;
  } else {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "list capacity %" PRIhsz
                              " reached; no more devices can be added",
                              list->capacity);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_hal_device_t* iree_hal_device_list_at(
    const iree_hal_device_list_t* list, iree_host_size_t i) {
  IREE_ASSERT_ARGUMENT(list);
  return i < list->count ? list->devices[i] : NULL;
}
