// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Direct queue operations. Each call is a complete submission: wait for
// semaphores, execute one operation, signal semaphores. No command buffer.

#include <stdlib.h>

#include "hrx_internal.h"

// Number of semaphore wrappers converted without a transient allocation.
#define HRX_INLINE_QUEUE_SEMAPHORES 16

typedef struct hrx_hal_semaphore_list_storage_t {
  // Common-case storage used for short semaphore lists.
  iree_hal_semaphore_t* inline_semaphores[HRX_INLINE_QUEUE_SEMAPHORES];

  // Overflow storage allocated when |inline_semaphores| is too small.
  iree_hal_semaphore_t** heap_semaphores;

  // HAL list borrowing the selected storage and the HRX payload values.
  iree_hal_semaphore_list_t list;
} hrx_hal_semaphore_list_storage_t;

typedef struct hrx_hal_semaphore_lists_t {
  // Converted wait semaphore list.
  hrx_hal_semaphore_list_storage_t wait;

  // Converted signal semaphore list.
  hrx_hal_semaphore_list_storage_t signal;
} hrx_hal_semaphore_lists_t;

static void hrx_hal_semaphore_list_storage_deinitialize(
    hrx_hal_semaphore_list_storage_t* storage) {
  iree_allocator_free(iree_allocator_system(), storage->heap_semaphores);
  memset(storage, 0, sizeof(*storage));
}

// Validates and wraps an HRX semaphore list using caller-provided stack
// storage with heap overflow. The HRX list and storage must remain live until
// the HAL queue operation returns.
static iree_status_t hrx_hal_semaphore_list_from_hrx(
    const char* list_name, const hrx_semaphore_list_t* list,
    hrx_hal_semaphore_list_storage_t* out_storage) {
  memset(out_storage, 0, sizeof(*out_storage));
  if (!list || list->count == 0) return iree_ok_status();
  if (!list->semaphores || !list->values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s semaphore list storage is NULL for %zu entries",
                            list_name, list->count);
  }
  for (size_t i = 0; i < list->count; ++i) {
    if (!list->semaphores[i] || !list->semaphores[i]->hal_semaphore) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%s semaphore %zu is NULL", list_name, i);
    }
  }

  iree_hal_semaphore_t** hal_semaphores = out_storage->inline_semaphores;
  if (list->count > IREE_ARRAYSIZE(out_storage->inline_semaphores)) {
    iree_host_size_t allocation_size = 0;
    if (!iree_host_size_checked_mul(list->count, sizeof(*hal_semaphores),
                                    &allocation_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "%s semaphore list allocation overflows",
                              list_name);
    }
    iree_hal_semaphore_t** heap_semaphores = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        iree_allocator_system(), allocation_size, (void**)&heap_semaphores));
    out_storage->heap_semaphores = heap_semaphores;
    hal_semaphores = heap_semaphores;
  }

  for (size_t i = 0; i < list->count; ++i) {
    hal_semaphores[i] = list->semaphores[i]->hal_semaphore;
  }
  out_storage->list.count = (iree_host_size_t)list->count;
  out_storage->list.semaphores = hal_semaphores;
  out_storage->list.payload_values = list->values;
  return iree_ok_status();
}

static void hrx_hal_semaphore_lists_deinitialize(
    hrx_hal_semaphore_lists_t* lists) {
  hrx_hal_semaphore_list_storage_deinitialize(&lists->signal);
  hrx_hal_semaphore_list_storage_deinitialize(&lists->wait);
}

static iree_status_t hrx_hal_semaphore_lists_initialize(
    const hrx_semaphore_list_t* wait_semaphores,
    const hrx_semaphore_list_t* signal_semaphores,
    hrx_hal_semaphore_lists_t* out_lists) {
  memset(out_lists, 0, sizeof(*out_lists));
  iree_status_t status = hrx_hal_semaphore_list_from_hrx(
      "wait", wait_semaphores, &out_lists->wait);
  if (iree_status_is_ok(status)) {
    status = hrx_hal_semaphore_list_from_hrx("signal", signal_semaphores,
                                             &out_lists->signal);
  }
  if (!iree_status_is_ok(status)) {
    hrx_hal_semaphore_lists_deinitialize(out_lists);
  }
  return status;
}

static iree_hal_queue_affinity_t hrx_normalize_queue_affinity(
    hrx_queue_affinity_t affinity) {
  return affinity == 0 ? IREE_HAL_QUEUE_AFFINITY_ANY
                       : (iree_hal_queue_affinity_t)affinity;
}

typedef struct hrx_host_call_thunk_t {
  hrx_host_call_fn_t callback;
  void* user_data;
} hrx_host_call_thunk_t;

static iree_status_t hrx_queue_host_call_thunk(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  (void)args;
  (void)context;
  hrx_host_call_thunk_t* thunk = (hrx_host_call_thunk_t*)user_data;
  hrx_status_t status = thunk->callback(thunk->user_data);
  free(thunk);
  return hrx_status_to_iree(status);
}

hrx_status_t hrx_queue_fill(hrx_device_t device, hrx_queue_affinity_t affinity,
                            const hrx_semaphore_list_t* wait_semaphores,
                            const hrx_semaphore_list_t* signal_semaphores,
                            hrx_buffer_t buffer, size_t offset, size_t size,
                            const void* pattern, size_t pattern_size) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_queue_fill");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, size);
  if (!device || !buffer || !pattern) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "device, buffer, or pattern is NULL"));
  }

  iree_hal_queue_t* queue = NULL;
  iree_status_t status = hrx_hal_device_select_queue(
      device->hal_device, affinity, IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER,
      &queue);
  if (!iree_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  hrx_hal_semaphore_lists_t semaphore_lists;
  status = hrx_hal_semaphore_lists_initialize(
      wait_semaphores, signal_semaphores, &semaphore_lists);
  if (!iree_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  status = iree_hal_queue_fill(
      queue, semaphore_lists.wait.list, semaphore_lists.signal.list,
      buffer->hal_buffer, (iree_device_size_t)offset, (iree_device_size_t)size,
      pattern, (iree_host_size_t)pattern_size, IREE_HAL_FILL_FLAG_NONE);
  hrx_hal_semaphore_lists_deinitialize(&semaphore_lists);
  HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
}

hrx_status_t hrx_queue_copy(hrx_device_t device, hrx_queue_affinity_t affinity,
                            const hrx_semaphore_list_t* wait_semaphores,
                            const hrx_semaphore_list_t* signal_semaphores,
                            hrx_buffer_t src, size_t src_offset,
                            hrx_buffer_t dst, size_t dst_offset, size_t size) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_queue_copy");
  HRX_TRACE_ZONE_APPEND_BYTES(z0, size);
  if (!device || !src || !dst) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                                                "device, src, or dst is NULL"));
  }

  iree_hal_queue_t* queue = NULL;
  iree_status_t status = hrx_hal_device_select_queue(
      device->hal_device, affinity, IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER,
      &queue);
  if (!iree_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  hrx_hal_semaphore_lists_t semaphore_lists;
  status = hrx_hal_semaphore_lists_initialize(
      wait_semaphores, signal_semaphores, &semaphore_lists);
  if (!iree_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  status = iree_hal_queue_copy(
      queue, semaphore_lists.wait.list, semaphore_lists.signal.list,
      src->hal_buffer, (iree_device_size_t)src_offset, dst->hal_buffer,
      (iree_device_size_t)dst_offset, (iree_device_size_t)size,
      IREE_HAL_COPY_FLAG_NONE);
  hrx_hal_semaphore_lists_deinitialize(&semaphore_lists);
  HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
}

hrx_status_t hrx_queue_barrier(hrx_device_t device,
                               hrx_queue_affinity_t affinity,
                               const hrx_semaphore_list_t* wait_semaphores,
                               const hrx_semaphore_list_t* signal_semaphores) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_queue_barrier");
  if (!device) {
    HRX_RETURN_AND_END_ZONE(
        z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "device is NULL"));
  }

  iree_hal_queue_t* queue = NULL;
  iree_status_t status = hrx_hal_device_select_queue(
      device->hal_device, affinity, /*required_roles=*/0, &queue);
  if (!iree_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  hrx_hal_semaphore_lists_t semaphore_lists;
  status = hrx_hal_semaphore_lists_initialize(
      wait_semaphores, signal_semaphores, &semaphore_lists);
  if (!iree_status_is_ok(status)) {
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  status = iree_hal_queue_barrier(queue, semaphore_lists.wait.list,
                                  semaphore_lists.signal.list,
                                  IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
  hrx_hal_semaphore_lists_deinitialize(&semaphore_lists);
  HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
}

hrx_status_t hrx_queue_dispatch(
    hrx_device_t device, hrx_queue_affinity_t affinity,
    const hrx_semaphore_list_t* wait_semaphores,
    const hrx_semaphore_list_t* signal_semaphores, hrx_executable_t executable,
    uint32_t export_ordinal, const hrx_dispatch_config_t* config,
    const void* constants, size_t constants_size,
    const hrx_buffer_ref_t* bindings, size_t binding_count, uint32_t flags) {
  HRX_TRACE_ZONE_BEGIN(z0, "hrx_queue_dispatch");
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, export_ordinal);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, binding_count);
  if (!device || !executable || !config || (binding_count > 0 && !bindings) ||
      (constants_size > 0 && !constants)) {
    HRX_RETURN_AND_END_ZONE(
        z0,
        hrx_make_status(
            HRX_STATUS_INVALID_ARGUMENT,
            "device, executable, config, constants, or bindings are invalid"));
  }

  iree_hal_dispatch_flags_t hal_flags = IREE_HAL_DISPATCH_FLAG_NONE;
  hrx_status_t flag_status =
      hrx_iree_dispatch_flags_from_hrx(flags, &hal_flags);
  if (!hrx_status_is_ok(flag_status)) HRX_RETURN_AND_END_ZONE(z0, flag_status);

  iree_hal_buffer_ref_t* hal_bindings = NULL;
  if (binding_count > 0) {
    hal_bindings = (iree_hal_buffer_ref_t*)calloc(
        binding_count, sizeof(iree_hal_buffer_ref_t));
    if (!hal_bindings) {
      HRX_RETURN_AND_END_ZONE(
          z0, hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                              "failed to allocate dispatch bindings"));
    }
    for (size_t i = 0; i < binding_count; ++i) {
      if (!bindings[i].buffer) {
        free(hal_bindings);
        HRX_RETURN_AND_END_ZONE(z0, hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                                                    "binding buffer is NULL"));
      }
      hal_bindings[i] =
          iree_hal_make_buffer_ref(bindings[i].buffer->hal_buffer,
                                   (iree_device_size_t)bindings[i].offset,
                                   (iree_device_size_t)bindings[i].length);
    }
  }

  hrx_hal_semaphore_lists_t semaphore_lists;
  iree_status_t status = hrx_hal_semaphore_lists_initialize(
      wait_semaphores, signal_semaphores, &semaphore_lists);
  if (!iree_status_is_ok(status)) {
    free(hal_bindings);
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  iree_hal_dispatch_config_t hal_config = {
      .workgroup_size =
          {
              config->workgroup_size[0],
              config->workgroup_size[1],
              config->workgroup_size[2],
          },
      .workgroup_count =
          {
              config->workgroup_count[0],
              config->workgroup_count[1],
              config->workgroup_count[2],
          },
  };
  iree_const_byte_span_t hal_constants =
      iree_make_const_byte_span((const uint8_t*)constants, constants_size);
  iree_hal_buffer_ref_list_t hal_binding_list = {
      .count = (iree_host_size_t)binding_count,
      .values = hal_bindings,
  };

  status = iree_hal_device_queue_dispatch(
      device->hal_device, hrx_normalize_queue_affinity(affinity),
      semaphore_lists.wait.list, semaphore_lists.signal.list,
      executable->hal_executable,
      iree_hal_executable_function_from_index(export_ordinal), hal_config,
      hal_constants, hal_binding_list, hal_flags);
  free(hal_bindings);
  hrx_hal_semaphore_lists_deinitialize(&semaphore_lists);
  HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
}

hrx_status_t hrx_queue_host_call(hrx_device_t device,
                                 hrx_queue_affinity_t affinity,
                                 const hrx_semaphore_list_t* wait_semaphores,
                                 const hrx_semaphore_list_t* signal_semaphores,
                                 hrx_host_call_fn_t callback, void* user_data) {
  if (!device || !callback) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "device or callback is NULL");
  }

  hrx_host_call_thunk_t* thunk =
      (hrx_host_call_thunk_t*)malloc(sizeof(hrx_host_call_thunk_t));
  if (!thunk) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "failed to allocate host call thunk");
  }
  thunk->callback = callback;
  thunk->user_data = user_data;

  hrx_hal_semaphore_lists_t semaphore_lists;
  iree_status_t status = hrx_hal_semaphore_lists_initialize(
      wait_semaphores, signal_semaphores, &semaphore_lists);
  if (!iree_status_is_ok(status)) {
    free(thunk);
    return hrx_status_from_iree(status);
  }

  const uint64_t args[4] = {0, 0, 0, 0};
  iree_hal_host_call_t call = {
      .fn = hrx_queue_host_call_thunk,
      .user_data = thunk,
  };
  status = iree_hal_device_queue_host_call(
      device->hal_device, hrx_normalize_queue_affinity(affinity),
      semaphore_lists.wait.list, semaphore_lists.signal.list, call, args,
      IREE_HAL_HOST_CALL_FLAG_NONE);
  hrx_hal_semaphore_lists_deinitialize(&semaphore_lists);
  if (!iree_status_is_ok(status)) {
    free(thunk);
  }
  return hrx_status_from_iree(status);
}
