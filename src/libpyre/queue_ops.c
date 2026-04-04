// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Direct queue operations. Each call is a complete submission: wait for
// semaphores, execute one operation, signal semaphores. No command buffer.

#include "pyre_internal.h"

#include <stdlib.h>

// Convert pyre semaphore lists to IREE semaphore lists.
// Caller must ensure the backing arrays outlive the returned list.
static iree_hal_semaphore_list_t pyre_to_iree_semaphore_list(
    const pyre_semaphore_list_t* list,
    iree_hal_semaphore_t** hal_semaphores,
    uint64_t* values) {
  iree_hal_semaphore_list_t result = {0};
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
#define PYRE_MAX_QUEUE_SEMAPHORES 16

typedef struct pyre_host_call_thunk_t {
  pyre_host_call_fn_t callback;
  void* user_data;
} pyre_host_call_thunk_t;

static iree_status_t pyre_queue_host_call_thunk(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  (void)args;
  (void)context;
  pyre_host_call_thunk_t* thunk = (pyre_host_call_thunk_t*)user_data;
  pyre_status_t status = thunk->callback(thunk->user_data);
  free(thunk);
  return pyre_status_to_iree(status);
}

pyre_status_t pyre_queue_fill(
    pyre_device_t device, pyre_queue_affinity_t affinity,
    const pyre_semaphore_list_t* wait_semaphores,
    const pyre_semaphore_list_t* signal_semaphores, pyre_buffer_t buffer,
    size_t offset, size_t size, const void* pattern, size_t pattern_size) {
  if (!device || !buffer || !pattern) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "device, buffer, or pattern is NULL");
  }

  // Create a one-shot command buffer with fill.
  iree_hal_command_buffer_t* cb = NULL;
  iree_status_t status = iree_hal_command_buffer_create(
      device->hal_device,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, (iree_hal_queue_affinity_t)affinity,
      /*binding_capacity=*/0, &cb);
  if (!iree_status_is_ok(status)) return pyre_status_from_iree(status);

  status = iree_hal_command_buffer_begin(cb);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(cb);
    return pyre_status_from_iree(status);
  }

  iree_hal_buffer_ref_t target_ref = iree_hal_make_buffer_ref(
      buffer->hal_buffer, (iree_device_size_t)offset, (iree_device_size_t)size);
  status = iree_hal_command_buffer_fill_buffer(
      cb, target_ref, pattern, (iree_host_size_t)pattern_size, /*flags=*/0);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(cb);
    return pyre_status_from_iree(status);
  }

  status = iree_hal_command_buffer_end(cb);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(cb);
    return pyre_status_from_iree(status);
  }

  // Build semaphore lists.
  iree_hal_semaphore_t* wait_hal[PYRE_MAX_QUEUE_SEMAPHORES];
  uint64_t wait_vals[PYRE_MAX_QUEUE_SEMAPHORES];
  iree_hal_semaphore_t* sig_hal[PYRE_MAX_QUEUE_SEMAPHORES];
  uint64_t sig_vals[PYRE_MAX_QUEUE_SEMAPHORES];

  iree_hal_semaphore_list_t wait_list =
      pyre_to_iree_semaphore_list(wait_semaphores, wait_hal, wait_vals);
  iree_hal_semaphore_list_t sig_list =
      pyre_to_iree_semaphore_list(signal_semaphores, sig_hal, sig_vals);

  iree_hal_buffer_binding_table_t bt = {0};
  status = iree_hal_device_queue_execute(
      device->hal_device, (iree_hal_queue_affinity_t)affinity,
      wait_list, sig_list, cb, bt, /*flags=*/0);
  iree_hal_command_buffer_release(cb);
  return pyre_status_from_iree(status);
}

pyre_status_t pyre_queue_copy(
    pyre_device_t device, pyre_queue_affinity_t affinity,
    const pyre_semaphore_list_t* wait_semaphores,
    const pyre_semaphore_list_t* signal_semaphores, pyre_buffer_t src,
    size_t src_offset, pyre_buffer_t dst, size_t dst_offset, size_t size) {
  if (!device || !src || !dst) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "device, src, or dst is NULL");
  }

  iree_hal_command_buffer_t* cb = NULL;
  iree_status_t status = iree_hal_command_buffer_create(
      device->hal_device,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, (iree_hal_queue_affinity_t)affinity,
      0, &cb);
  if (!iree_status_is_ok(status)) return pyre_status_from_iree(status);

  status = iree_hal_command_buffer_begin(cb);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(cb);
    return pyre_status_from_iree(status);
  }

  iree_hal_buffer_ref_t src_ref = iree_hal_make_buffer_ref(
      src->hal_buffer, (iree_device_size_t)src_offset, (iree_device_size_t)size);
  iree_hal_buffer_ref_t dst_ref = iree_hal_make_buffer_ref(
      dst->hal_buffer, (iree_device_size_t)dst_offset, (iree_device_size_t)size);
  status = iree_hal_command_buffer_copy_buffer(cb, src_ref, dst_ref, /*flags=*/0);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(cb);
    return pyre_status_from_iree(status);
  }

  status = iree_hal_command_buffer_end(cb);
  if (!iree_status_is_ok(status)) {
    iree_hal_command_buffer_release(cb);
    return pyre_status_from_iree(status);
  }

  iree_hal_semaphore_t* wait_hal[PYRE_MAX_QUEUE_SEMAPHORES];
  uint64_t wait_vals[PYRE_MAX_QUEUE_SEMAPHORES];
  iree_hal_semaphore_t* sig_hal[PYRE_MAX_QUEUE_SEMAPHORES];
  uint64_t sig_vals[PYRE_MAX_QUEUE_SEMAPHORES];

  iree_hal_semaphore_list_t wait_list =
      pyre_to_iree_semaphore_list(wait_semaphores, wait_hal, wait_vals);
  iree_hal_semaphore_list_t sig_list =
      pyre_to_iree_semaphore_list(signal_semaphores, sig_hal, sig_vals);

  iree_hal_buffer_binding_table_t bt = {0};
  status = iree_hal_device_queue_execute(
      device->hal_device, (iree_hal_queue_affinity_t)affinity,
      wait_list, sig_list, cb, bt, /*flags=*/0);
  iree_hal_command_buffer_release(cb);
  return pyre_status_from_iree(status);
}

pyre_status_t pyre_queue_barrier(
    pyre_device_t device, pyre_queue_affinity_t affinity,
    const pyre_semaphore_list_t* wait_semaphores,
    const pyre_semaphore_list_t* signal_semaphores) {
  if (!device) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "device is NULL");
  }

  iree_hal_semaphore_t* wait_hal[PYRE_MAX_QUEUE_SEMAPHORES];
  uint64_t wait_vals[PYRE_MAX_QUEUE_SEMAPHORES];
  iree_hal_semaphore_t* sig_hal[PYRE_MAX_QUEUE_SEMAPHORES];
  uint64_t sig_vals[PYRE_MAX_QUEUE_SEMAPHORES];

  iree_hal_semaphore_list_t wait_list =
      pyre_to_iree_semaphore_list(wait_semaphores, wait_hal, wait_vals);
  iree_hal_semaphore_list_t sig_list =
      pyre_to_iree_semaphore_list(signal_semaphores, sig_hal, sig_vals);

  iree_status_t status = iree_hal_device_queue_barrier(
      device->hal_device, (iree_hal_queue_affinity_t)affinity,
      wait_list, sig_list, /*flags=*/0);
  return pyre_status_from_iree(status);
}

pyre_status_t pyre_queue_dispatch(
    pyre_device_t device, pyre_queue_affinity_t affinity,
    const pyre_semaphore_list_t* wait_semaphores,
    const pyre_semaphore_list_t* signal_semaphores,
    pyre_executable_t executable, uint32_t export_ordinal,
    const pyre_dispatch_config_t* config,
    const void* constants, size_t constants_size,
    const pyre_buffer_ref_t* bindings, size_t binding_count,
    uint32_t flags) {
  if (!device || !executable || !config ||
      (binding_count > 0 && !bindings) ||
      (constants_size > 0 && !constants)) {
    return pyre_make_status(
        PYRE_STATUS_INVALID_ARGUMENT,
        "device, executable, config, constants, or bindings are invalid");
  }

  iree_hal_buffer_ref_t* hal_bindings = NULL;
  if (binding_count > 0) {
    hal_bindings = (iree_hal_buffer_ref_t*)calloc(
        binding_count, sizeof(iree_hal_buffer_ref_t));
    if (!hal_bindings) {
      return pyre_make_status(PYRE_STATUS_OUT_OF_MEMORY,
                              "failed to allocate dispatch bindings");
    }
    for (size_t i = 0; i < binding_count; ++i) {
      if (!bindings[i].buffer) {
        free(hal_bindings);
        return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                                "binding buffer is NULL");
      }
      hal_bindings[i] = iree_hal_make_buffer_ref(
          bindings[i].buffer->hal_buffer,
          (iree_device_size_t)bindings[i].offset,
          (iree_device_size_t)bindings[i].length);
    }
  }

  iree_hal_semaphore_t* wait_hal[PYRE_MAX_QUEUE_SEMAPHORES];
  uint64_t wait_vals[PYRE_MAX_QUEUE_SEMAPHORES];
  iree_hal_semaphore_t* sig_hal[PYRE_MAX_QUEUE_SEMAPHORES];
  uint64_t sig_vals[PYRE_MAX_QUEUE_SEMAPHORES];

  iree_hal_semaphore_list_t wait_list =
      pyre_to_iree_semaphore_list(wait_semaphores, wait_hal, wait_vals);
  iree_hal_semaphore_list_t sig_list =
      pyre_to_iree_semaphore_list(signal_semaphores, sig_hal, sig_vals);

  iree_hal_dispatch_config_t hal_config = {
      .workgroup_size = {
          config->workgroup_size[0],
          config->workgroup_size[1],
          config->workgroup_size[2],
      },
      .workgroup_count = {
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
      device->hal_device, (iree_hal_queue_affinity_t)affinity,
      wait_list, sig_list, executable->hal_executable,
      (iree_hal_executable_export_ordinal_t)export_ordinal,
      hal_config, hal_constants, hal_binding_list,
      (iree_hal_dispatch_flags_t)flags);
  free(hal_bindings);
  return pyre_status_from_iree(status);
}

pyre_status_t pyre_queue_host_call(
    pyre_device_t device, pyre_queue_affinity_t affinity,
    const pyre_semaphore_list_t* wait_semaphores,
    const pyre_semaphore_list_t* signal_semaphores,
    pyre_host_call_fn_t callback, void* user_data) {
  if (!device || !callback) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "device or callback is NULL");
  }

  pyre_host_call_thunk_t* thunk =
      (pyre_host_call_thunk_t*)malloc(sizeof(pyre_host_call_thunk_t));
  if (!thunk) {
    return pyre_make_status(PYRE_STATUS_OUT_OF_MEMORY,
                            "failed to allocate host call thunk");
  }
  thunk->callback = callback;
  thunk->user_data = user_data;

  iree_hal_semaphore_t* wait_hal[PYRE_MAX_QUEUE_SEMAPHORES];
  uint64_t wait_vals[PYRE_MAX_QUEUE_SEMAPHORES];
  iree_hal_semaphore_t* sig_hal[PYRE_MAX_QUEUE_SEMAPHORES];
  uint64_t sig_vals[PYRE_MAX_QUEUE_SEMAPHORES];

  iree_hal_semaphore_list_t wait_list =
      pyre_to_iree_semaphore_list(wait_semaphores, wait_hal, wait_vals);
  iree_hal_semaphore_list_t sig_list =
      pyre_to_iree_semaphore_list(signal_semaphores, sig_hal, sig_vals);

  const uint64_t args[4] = {0, 0, 0, 0};
  iree_hal_host_call_t call = {
      .fn = pyre_queue_host_call_thunk,
      .user_data = thunk,
  };
  iree_status_t status = iree_hal_device_queue_host_call(
      device->hal_device, (iree_hal_queue_affinity_t)affinity,
      wait_list, sig_list, call, args, IREE_HAL_HOST_CALL_FLAG_NONE);
  if (!iree_status_is_ok(status)) {
    free(thunk);
  }
  return pyre_status_from_iree(status);
}
