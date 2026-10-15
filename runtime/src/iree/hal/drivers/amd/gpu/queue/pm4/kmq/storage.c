// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/kmq/storage.h"

#include "iree/hal/drivers/amd/gpu/queue/pm4/backing.h"
#include "iree/hal/drivers/amd/status.h"

iree_status_t iree_hal_amd_gpu_pm4_kmq_storage_allocate(
    const iree_hal_amd_gpu_context_t* context, uint32_t family_ordinal,
    uint64_t command_capacity, uint32_t submission_capacity,
    const iree_hal_amd_fabric_memory_options_t* memory_options,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_kmq_storage_t** out_storage,
    iree_hal_amd_gpu_pm4_ring_t* out_ring) {
  *out_storage = NULL;
  iree_hal_amd_gpu_pm4_kmq_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*storage) + submission_capacity * sizeof(uint64_t),
      (void**)&storage));
  *out_storage = storage;
  storage->context = context;
  storage->host_allocator = host_allocator;
  storage->points = (uint64_t*)(storage + 1);
  storage->capacity_mask = submission_capacity - 1;
  iree_atomic_store(&storage->wake_required, 1, iree_memory_order_relaxed);
  const amdf_gpu_kernel_queue_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .queue_family_ordinal = family_ordinal,
      .maximum_pending_submission_count = submission_capacity,
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->gpu->kernel_queue_create(context->device, &create_info,
                                        &storage->queue),
      "gpu.kernel_queue_create"));
  amdf_kernel_queue_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO,
      .structure_size = sizeof(info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->kernel_queue_query_info(storage->queue, &info),
      "kernel_queue_query_info"));
  if (!iree_any_bit_set(info.notification_types,
                        UINT64_C(1) << context->kernel_notification.type)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "PM4 KERNEL queue has no compatible native wake");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_pm4_backing_allocate(
      context, memory_options, command_capacity, 64, &storage->memory, NULL));
  iree_hal_amd_gpu_pm4_ring_initialize(
      (uint32_t*)storage->memory.contents.data,
      (uint32_t)(command_capacity / sizeof(uint32_t)), out_ring);
  return iree_ok_status();
}

amdf_status_t iree_hal_amd_gpu_pm4_kmq_storage_submit(
    iree_hal_amd_gpu_pm4_kmq_storage_t* storage, uint64_t position) {
  const amdf_gpu_kernel_queue_submission_info_t submission = {
      .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO,
      .structure_size = sizeof(submission),
      .command_count = 1,
      .commands = &storage->pending,
  };
  uint64_t point = 0;
  amdf_status_t status = storage->context->gpu->kernel_queue_submit(
      storage->queue, &submission, &point);
  if (amdf_status_is_ok(status)) {
    storage->points[position & storage->capacity_mask] = point;
    storage->last_submission = point;
    storage->pending.byte_length = 0;
  }
  return status;
}

amdf_status_t iree_hal_amd_gpu_pm4_kmq_storage_refresh(
    iree_hal_amd_gpu_pm4_kmq_storage_t* storage, uint64_t* out_retired) {
  amdf_kernel_queue_status_t snapshot = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
      .structure_size = sizeof(snapshot),
  };
  amdf_status_t status = storage->context->api->kernel_queue_refresh_status(
      storage->queue, &snapshot);
  if (amdf_status_is_ok(status)) {
    status = snapshot.terminal_status;
    *out_retired = snapshot.retired_submission;
  }
  return status;
}

amdf_status_t iree_hal_amd_gpu_pm4_kmq_storage_arm(
    iree_hal_amd_gpu_pm4_kmq_storage_t* storage, uint64_t submission) {
  if (submission == storage->armed_submission) {
    return AMDF_STATUS_OK;
  }
  amdf_status_t status =
      storage->context->api->kernel_queue_request_notification(
          storage->queue, submission, &storage->context->kernel_notification);
  if (amdf_status_is_ok(status)) {
    storage->armed_submission = submission;
  }
  return status;
}

amdf_status_t iree_hal_amd_gpu_pm4_kmq_storage_free(
    iree_hal_amd_gpu_pm4_kmq_storage_t* storage) {
  if (!storage) {
    return AMDF_STATUS_OK;
  }
  if (!amdf_status_is_ok(storage->removal_status)) {
    return storage->removal_status;
  }
  const iree_hal_amd_gpu_context_t* context = storage->context;
  if (storage->queue && storage->last_submission) {
    amdf_status_t status = context->api->kernel_queue_wait(
        storage->queue, storage->last_submission, AMDF_TIMEOUT_INFINITE, 0);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  if (storage->queue) {
    const amdf_status_t status =
        context->api->kernel_queue_destroy(storage->queue);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
      return status;
    }
    storage->queue = NULL;
    storage->removal_status = status;
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  iree_hal_amd_fabric_memory_deinitialize(context->api, context->event_sink,
                                          &storage->memory);
  iree_allocator_free(storage->host_allocator, storage);
  return AMDF_STATUS_OK;
}
