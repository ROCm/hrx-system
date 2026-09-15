// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/user_queue.h"

#include "libamdf/src/allocator.h"
#include "libamdf/src/device.h"
#include "libamdf/src/structure.h"

amdf_status_t amdf_user_queue_initialize(amdf_user_queue_t* queue,
                                         const amdf_user_queue_vtable_t* vtable,
                                         amdf_device_t* device,
                                         const amdf_user_queue_info_t* info) {
  const amdf_status_t status = amdf_device_register_child(device);
  if (!amdf_status_is_ok(status)) return status;
  queue->host_allocator = amdf_device_host_allocator(device);
  queue->vtable = vtable;
  queue->device = device;
  queue->info = *info;
  amdf_child_tracker_initialize(&queue->mappings);
  return AMDF_STATUS_OK;
}

void amdf_user_queue_deinitialize(amdf_user_queue_t* queue) {
  amdf_device_unregister_child(queue->device);
  queue->device = NULL;
}

amdf_status_t amdf_user_queue_mapping_initialize(
    amdf_user_queue_mapping_t* mapping,
    const amdf_user_queue_mapping_vtable_t* vtable, amdf_user_queue_t* queue,
    amdf_device_t* producer_device,
    const amdf_user_queue_mapping_info_t* info) {
  amdf_status_t status = amdf_child_tracker_register(&queue->mappings);
  if (!amdf_status_is_ok(status)) return status;
  if (producer_device != NULL) {
    status = amdf_device_register_child(producer_device);
    if (!amdf_status_is_ok(status)) {
      amdf_child_tracker_unregister(&queue->mappings);
      return status;
    }
  }
  mapping->host_allocator = queue->host_allocator;
  mapping->vtable = vtable;
  mapping->queue = queue;
  mapping->producer_device = producer_device;
  mapping->info = *info;
  return AMDF_STATUS_OK;
}

void amdf_user_queue_mapping_deinitialize(amdf_user_queue_mapping_t* mapping) {
  if (mapping->producer_device != NULL) {
    amdf_device_unregister_child(mapping->producer_device);
    mapping->producer_device = NULL;
  }
  amdf_child_tracker_unregister(&mapping->queue->mappings);
  mapping->queue = NULL;
}

amdf_status_t AMDF_CALL amdf_user_queue_query_info(
    amdf_user_queue_t* queue, amdf_user_queue_info_t* out_info) {
  if (queue == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO,
      (uint32_t)sizeof(amdf_user_queue_info_t));
  if (!amdf_status_is_ok(status)) return status;

  const uint32_t structure_size = out_info->structure_size;
  void* const next = out_info->next;
  *out_info = queue->info;
  out_info->structure_size = structure_size;
  out_info->next = next;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL
amdf_user_queue_map(amdf_user_queue_t* queue, amdf_device_t* producer_device,
                    amdf_user_queue_mapping_t** out_mapping) {
  if (queue == NULL || out_mapping == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_user_queue_mapping_t* mapping = NULL;
  const amdf_status_t status =
      queue->vtable->map(queue, producer_device, &mapping);
  if (amdf_status_is_ok(status)) *out_mapping = mapping;
  return status;
}

amdf_status_t AMDF_CALL
amdf_user_queue_mapping_query_info(amdf_user_queue_mapping_t* mapping,
                                   amdf_user_queue_mapping_info_t* out_info) {
  if (mapping == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO,
      (uint32_t)sizeof(amdf_user_queue_mapping_info_t));
  if (!amdf_status_is_ok(status)) return status;

  const uint32_t structure_size = out_info->structure_size;
  void* const next = out_info->next;
  *out_info = mapping->info;
  out_info->structure_size = structure_size;
  out_info->next = next;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL
amdf_user_queue_mapping_destroy(amdf_user_queue_mapping_t* mapping) {
  if (mapping == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = mapping->vtable->destroy_native(mapping);
  if (amdf_status_is_ok(status)) {
    const amdf_allocator_t host_allocator = mapping->host_allocator;
    amdf_user_queue_mapping_deinitialize(mapping);
    amdf_free(host_allocator, mapping);
  }
  return status;
}

amdf_status_t AMDF_CALL amdf_user_queue_query_status(
    amdf_user_queue_t* queue, amdf_user_queue_status_t* out_status) {
  if (queue == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_status, AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS,
      (uint32_t)sizeof(amdf_user_queue_status_t));
  if (!amdf_status_is_ok(status)) return status;

  amdf_user_queue_status_t value = {
      .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS,
      .structure_size = sizeof(value),
  };
  const amdf_status_t query_status = queue->vtable->query_status(queue, &value);
  if (!amdf_status_is_ok(query_status)) return query_status;
  const uint32_t structure_size = out_status->structure_size;
  void* const next = out_status->next;
  *out_status = value;
  out_status->structure_size = structure_size;
  out_status->next = next;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_user_queue_wait_consumed(
    amdf_user_queue_t* queue, uint64_t published_index,
    uint64_t timeout_nanoseconds, uint64_t poll_duration_nanoseconds) {
  if (queue == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return queue->vtable->wait_consumed(
      queue, published_index, timeout_nanoseconds, poll_duration_nanoseconds);
}

amdf_status_t AMDF_CALL amdf_user_queue_destroy(amdf_user_queue_t* queue) {
  if (queue == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (amdf_child_tracker_count(&queue->mappings) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  const amdf_status_t status = queue->vtable->destroy_native(queue);
  if (amdf_status_is_ok(status)) {
    const amdf_allocator_t host_allocator = queue->host_allocator;
    amdf_user_queue_deinitialize(queue);
    amdf_free(host_allocator, queue);
  }
  return status;
}
