// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Direct queue operations. Each call is a complete submission: wait for
// semaphores, execute one operation, signal semaphores. No command buffer.

#include "pyre_internal.h"

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
