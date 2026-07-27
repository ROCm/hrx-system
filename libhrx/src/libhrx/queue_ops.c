// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Direct queue operations. Each call is a complete submission: wait for
// semaphores, execute one operation, signal semaphores. No command buffer.

#include <stdlib.h>

#include "hrx_internal.h"

// Convert hrx semaphore lists to IREE semaphore lists.
// Caller must ensure the backing arrays outlive the returned list.
static iree_hal_semaphore_list_t hrx_to_iree_semaphore_list(
    const hrx_semaphore_list_t* list, iree_hal_semaphore_t** hal_semaphores,
    uint64_t* values) {
  iree_hal_semaphore_list_t result = iree_hal_semaphore_list_empty();
  if (!list || list->count == 0) return result;
  for (size_t i = 0; i < list->count; i++) {
    hal_semaphores[i] = list->semaphores[i]->hal_semaphore;
    values[i] = list->values[i];
  }
  result.count = (iree_host_size_t)list->count;
  result.semaphores = hal_semaphores;
  result.payload_values = values;
  return result;
}

// Max semaphores per direct queue op (stack-allocated arrays).
#define HRX_MAX_QUEUE_SEMAPHORES 16

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

  iree_hal_queue_affinity_t queue_affinity =
      hrx_normalize_queue_affinity(affinity);

  // Create a one-shot command buffer with fill.
  iree_hal_command_buffer_t* cb = NULL;
  iree_status_t status = iree_hal_command_buffer_create(
      device->hal_device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, queue_affinity,
      /*binding_capacity=*/0, &cb);
  if (!iree_status_is_ok(status))
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));

  status = iree_hal_command_buffer_begin(cb);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(cb);
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  iree_hal_buffer_ref_t target_ref = iree_hal_make_buffer_ref(
      buffer->hal_buffer, (iree_device_size_t)offset, (iree_device_size_t)size);
  status = iree_hal_command_buffer_fill_buffer(
      cb, target_ref, pattern, (iree_host_size_t)pattern_size, /*flags=*/0);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(cb);
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  status = iree_hal_command_buffer_end(cb);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(cb);
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  // Build semaphore lists.
  iree_hal_semaphore_t* wait_hal[HRX_MAX_QUEUE_SEMAPHORES];
  uint64_t wait_vals[HRX_MAX_QUEUE_SEMAPHORES];
  iree_hal_semaphore_t* sig_hal[HRX_MAX_QUEUE_SEMAPHORES];
  uint64_t sig_vals[HRX_MAX_QUEUE_SEMAPHORES];

  iree_hal_semaphore_list_t wait_list =
      hrx_to_iree_semaphore_list(wait_semaphores, wait_hal, wait_vals);
  iree_hal_semaphore_list_t sig_list =
      hrx_to_iree_semaphore_list(signal_semaphores, sig_hal, sig_vals);

  iree_hal_buffer_binding_table_t bt = iree_hal_buffer_binding_table_empty();
  status = iree_hal_device_queue_execute(device->hal_device, queue_affinity,
                                         wait_list, sig_list, cb, bt,
                                         /*flags=*/0);
  iree_hal_command_buffer_release(cb);
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

  iree_hal_queue_affinity_t queue_affinity =
      hrx_normalize_queue_affinity(affinity);

  iree_hal_command_buffer_t* cb = NULL;
  iree_status_t status = iree_hal_command_buffer_create(
      device->hal_device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, queue_affinity, 0, &cb);
  if (!iree_status_is_ok(status))
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));

  status = iree_hal_command_buffer_begin(cb);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(cb);
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  iree_hal_buffer_ref_t src_ref =
      iree_hal_make_buffer_ref(src->hal_buffer, (iree_device_size_t)src_offset,
                               (iree_device_size_t)size);
  iree_hal_buffer_ref_t dst_ref =
      iree_hal_make_buffer_ref(dst->hal_buffer, (iree_device_size_t)dst_offset,
                               (iree_device_size_t)size);
  status =
      iree_hal_command_buffer_copy_buffer(cb, src_ref, dst_ref, /*flags=*/0);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(cb);
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  status = iree_hal_command_buffer_end(cb);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(cb);
    HRX_RETURN_AND_END_ZONE(z0, hrx_status_from_iree(status));
  }

  iree_hal_semaphore_t* wait_hal[HRX_MAX_QUEUE_SEMAPHORES];
  uint64_t wait_vals[HRX_MAX_QUEUE_SEMAPHORES];
  iree_hal_semaphore_t* sig_hal[HRX_MAX_QUEUE_SEMAPHORES];
  uint64_t sig_vals[HRX_MAX_QUEUE_SEMAPHORES];

  iree_hal_semaphore_list_t wait_list =
      hrx_to_iree_semaphore_list(wait_semaphores, wait_hal, wait_vals);
  iree_hal_semaphore_list_t sig_list =
      hrx_to_iree_semaphore_list(signal_semaphores, sig_hal, sig_vals);

  iree_hal_buffer_binding_table_t bt = iree_hal_buffer_binding_table_empty();
  status = iree_hal_device_queue_execute(device->hal_device, queue_affinity,
                                         wait_list, sig_list, cb, bt,
                                         /*flags=*/0);
  iree_hal_command_buffer_release(cb);
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

  iree_hal_semaphore_t* wait_hal[HRX_MAX_QUEUE_SEMAPHORES];
  uint64_t wait_vals[HRX_MAX_QUEUE_SEMAPHORES];
  iree_hal_semaphore_t* sig_hal[HRX_MAX_QUEUE_SEMAPHORES];
  uint64_t sig_vals[HRX_MAX_QUEUE_SEMAPHORES];

  iree_hal_semaphore_list_t wait_list =
      hrx_to_iree_semaphore_list(wait_semaphores, wait_hal, wait_vals);
  iree_hal_semaphore_list_t sig_list =
      hrx_to_iree_semaphore_list(signal_semaphores, sig_hal, sig_vals);

  iree_status_t status = iree_hal_device_queue_barrier(
      device->hal_device, hrx_normalize_queue_affinity(affinity), wait_list,
      sig_list, /*flags=*/0);
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

  iree_hal_semaphore_t* wait_hal[HRX_MAX_QUEUE_SEMAPHORES];
  uint64_t wait_vals[HRX_MAX_QUEUE_SEMAPHORES];
  iree_hal_semaphore_t* sig_hal[HRX_MAX_QUEUE_SEMAPHORES];
  uint64_t sig_vals[HRX_MAX_QUEUE_SEMAPHORES];

  iree_hal_semaphore_list_t wait_list =
      hrx_to_iree_semaphore_list(wait_semaphores, wait_hal, wait_vals);
  iree_hal_semaphore_list_t sig_list =
      hrx_to_iree_semaphore_list(signal_semaphores, sig_hal, sig_vals);

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

  iree_status_t status = iree_hal_device_queue_dispatch(
      device->hal_device, hrx_normalize_queue_affinity(affinity), wait_list,
      sig_list, executable->hal_executable,
      iree_hal_executable_function_from_index(export_ordinal), hal_config,
      hal_constants, hal_binding_list, hal_flags);
  free(hal_bindings);
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

  iree_hal_semaphore_t* wait_hal[HRX_MAX_QUEUE_SEMAPHORES];
  uint64_t wait_vals[HRX_MAX_QUEUE_SEMAPHORES];
  iree_hal_semaphore_t* sig_hal[HRX_MAX_QUEUE_SEMAPHORES];
  uint64_t sig_vals[HRX_MAX_QUEUE_SEMAPHORES];

  iree_hal_semaphore_list_t wait_list =
      hrx_to_iree_semaphore_list(wait_semaphores, wait_hal, wait_vals);
  iree_hal_semaphore_list_t sig_list =
      hrx_to_iree_semaphore_list(signal_semaphores, sig_hal, sig_vals);

  const uint64_t args[4] = {0, 0, 0, 0};
  iree_hal_host_call_t call = {
      .fn = hrx_queue_host_call_thunk,
      .user_data = thunk,
  };
  iree_status_t status = iree_hal_device_queue_host_call(
      device->hal_device, hrx_normalize_queue_affinity(affinity), wait_list,
      sig_list, call, args, IREE_HAL_HOST_CALL_FLAG_NONE);
  if (!iree_status_is_ok(status)) {
    free(thunk);
  }
  return hrx_status_from_iree(status);
}
