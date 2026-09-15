// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/kernel_queue.h"

#include <stddef.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/device.h"
#include "libamdf/src/structure.h"

amdf_status_t amdf_kernel_queue_initialize(
    amdf_kernel_queue_t* queue, const amdf_kernel_queue_vtable_t* vtable,
    amdf_device_t* device, const amdf_kernel_queue_info_t* info) {
  const amdf_status_t status = amdf_device_register_child(device);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  queue->host_allocator = amdf_device_host_allocator(device);
  queue->vtable = vtable;
  queue->device = device;
  queue->info = *info;
  return AMDF_STATUS_OK;
}

void amdf_kernel_queue_deinitialize(amdf_kernel_queue_t* queue) {
  amdf_device_unregister_child(queue->device);
  queue->device = NULL;
}

amdf_status_t AMDF_CALL amdf_kernel_queue_query_info(
    amdf_kernel_queue_t* queue, amdf_kernel_queue_info_t* out_info) {
  if (queue == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO,
      (uint32_t)sizeof(amdf_kernel_queue_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }

  const uint32_t structure_size = out_info->structure_size;
  void* const next = out_info->next;
  *out_info = queue->info;
  out_info->structure_size = structure_size;
  out_info->next = next;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_kernel_queue_query_status(
    amdf_kernel_queue_t* queue, amdf_kernel_queue_status_t* out_status) {
  if (queue == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_status, AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
      (uint32_t)sizeof(amdf_kernel_queue_status_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }

  amdf_kernel_queue_status_t value = {0};
  value.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS;
  value.structure_size = sizeof(value);
  const amdf_status_t query_status = queue->vtable->query_status(queue, &value);
  if (!amdf_status_is_ok(query_status)) {
    return query_status;
  }
  const uint32_t structure_size = out_status->structure_size;
  void* const next = out_status->next;
  *out_status = value;
  out_status->structure_size = structure_size;
  out_status->next = next;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_kernel_queue_wait(
    amdf_kernel_queue_t* queue, uint64_t submission,
    uint64_t timeout_nanoseconds, uint64_t poll_duration_nanoseconds) {
  if (queue == NULL || submission == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return queue->vtable->wait(queue, submission, timeout_nanoseconds,
                             poll_duration_nanoseconds);
}

amdf_status_t AMDF_CALL amdf_kernel_queue_destroy(amdf_kernel_queue_t* queue) {
  if (queue == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = queue->vtable->destroy_native(queue);
  if (amdf_status_is_ok(status)) {
    const amdf_allocator_t host_allocator = queue->host_allocator;
    amdf_kernel_queue_deinitialize(queue);
    amdf_free(host_allocator, queue);
  }
  return status;
}
