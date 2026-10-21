// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/scratch.h"

#include "iree/hal/drivers/amd/status.h"

enum {
  IREE_HAL_AMD_GPU_AQL_SCRATCH_IDLE = 0,
  IREE_HAL_AMD_GPU_AQL_SCRATCH_ACQUIRING,
  IREE_HAL_AMD_GPU_AQL_SCRATCH_READY,
  IREE_HAL_AMD_GPU_AQL_SCRATCH_RELEASING,
  IREE_HAL_AMD_GPU_AQL_SCRATCH_FAILED,
};

static void iree_hal_amd_gpu_aql_scratch_wake(
    iree_hal_amd_gpu_context_t* context) {
  const amdf_status_t status = iree_hal_amd_gpu_context_signal(context);
  if (!amdf_status_is_ok(status)) {
    iree_hal_amd_gpu_context_report(context, status,
                                    "AQL scratch service wake");
  }
}

void iree_hal_amd_gpu_aql_scratch_initialize(
    iree_hal_amd_gpu_aql_growth_context_t* growth, amdf_user_queue_t* queue,
    const iree_hal_amd_fabric_memory_options_t* options,
    iree_hal_amd_gpu_aql_scratch_t* out_scratch) {
  *out_scratch = (iree_hal_amd_gpu_aql_scratch_t){
      .entry.kind = IREE_HAL_AMD_GPU_AQL_ENTRY_SCRATCH,
      .growth = growth,
      .queue = queue,
      .options = options};
}

static void iree_hal_amd_gpu_aql_scratch_acquired(void* user_data,
                                                  iree_status_t status) {
  iree_hal_amd_gpu_aql_scratch_t* scratch = user_data;
  iree_hal_amd_gpu_aql_growth_context_t* growth = scratch->growth;
  iree_hal_amd_gpu_publication_t* publication = growth->publication;
  const uint32_t ordinal = growth->queue_ordinal;
  iree_atomic_fetch_add(&growth->pending_count, 1, iree_memory_order_relaxed);
  scratch->status = status;
  iree_atomic_store(&scratch->phase, IREE_HAL_AMD_GPU_AQL_SCRATCH_READY,
                    iree_memory_order_release);
  iree_hal_amd_gpu_aql_growth_enqueue(growth, &scratch->entry);
  // Resumption can precede enqueue's final publisher access. This last queue
  // access joins that callback even if scratch has already returned to IDLE.
  iree_atomic_fetch_sub(&growth->pending_count, 1, iree_memory_order_release);
  iree_hal_amd_gpu_publication_notify(publication, ordinal);
}

static void iree_hal_amd_gpu_aql_scratch_acquire(void* user_data) {
  iree_hal_amd_gpu_aql_scratch_t* scratch = user_data;
  scratch->allocation = (iree_hal_amd_fabric_memory_request_t){
      .options = scratch->options,
      .byte_length = scratch->request.byte_length,
      .alignment = 4096};
  iree_hal_amd_gpu_context_acquire_memory(
      scratch->growth->context, 1, &scratch->allocation, &scratch->memory,
      (iree_hal_amd_fabric_memory_callback_t){
          .fn = iree_hal_amd_gpu_aql_scratch_acquired, .user_data = scratch},
      &scratch->acquire);
}

amdf_status_t iree_hal_amd_gpu_aql_scratch_poll(
    iree_hal_amd_gpu_aql_scratch_t* scratch) {
  if (iree_atomic_load(&scratch->phase, iree_memory_order_acquire) !=
      IREE_HAL_AMD_GPU_AQL_SCRATCH_IDLE) {
    return AMDF_STATUS_OK;
  }
  amdf_gpu_aql_scratch_request_t request = {
      .type = AMDF_STRUCTURE_TYPE_GPU_AQL_SCRATCH_REQUEST,
      .structure_size = sizeof(request)};
  amdf_status_t status =
      scratch->growth->context->gpu->user_queue_query_scratch(scratch->queue,
                                                              &request);
  if (!amdf_status_is_ok(status) || !request.byte_length) {
    return status;
  }
  scratch->request = request;
  iree_atomic_store(&scratch->phase, IREE_HAL_AMD_GPU_AQL_SCRATCH_ACQUIRING,
                    iree_memory_order_release);
  if (scratch->memory.handle) {
    iree_hal_amd_gpu_context_release_memory(
        scratch->growth->context, 1, &scratch->memory,
        (iree_hal_amd_fabric_memory_release_callback_t){
            .fn = iree_hal_amd_gpu_aql_scratch_acquire, .user_data = scratch},
        &scratch->release);
  } else {
    iree_hal_amd_gpu_aql_scratch_acquire(scratch);
  }
  return AMDF_STATUS_OK;
}

iree_status_t iree_hal_amd_gpu_aql_scratch_resume(
    iree_hal_amd_gpu_aql_scratch_t* scratch) {
  iree_status_t status = scratch->status;
  scratch->status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    const amdf_gpu_queue_scratch_t binding = {
        .memory = scratch->memory.handle,
        .byte_length = scratch->request.byte_length,
        .maximum_private_segment_byte_length =
            scratch->request.private_segment_byte_length,
        .maximum_wave_count = scratch->request.wave_count};
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        scratch->growth->context->gpu->user_queue_resume_scratch(scratch->queue,
                                                                 &binding),
        "user_queue_resume_scratch");
  }
  iree_atomic_store(&scratch->phase,
                    iree_status_is_ok(status)
                        ? IREE_HAL_AMD_GPU_AQL_SCRATCH_IDLE
                        : IREE_HAL_AMD_GPU_AQL_SCRATCH_FAILED,
                    iree_memory_order_release);
  // Firmware can request a larger allocation immediately after resume, before
  // IDLE becomes visible to completion. Recheck after transferring ownership so
  // that an event observed during READY cannot lose that next request.
  iree_hal_amd_gpu_aql_scratch_wake(scratch->growth->context);
  return status;
}

bool iree_hal_amd_gpu_aql_scratch_is_pending(
    const iree_hal_amd_gpu_aql_scratch_t* scratch) {
  const int32_t phase =
      iree_atomic_load(&scratch->phase, iree_memory_order_acquire);
  return phase == IREE_HAL_AMD_GPU_AQL_SCRATCH_ACQUIRING ||
         phase == IREE_HAL_AMD_GPU_AQL_SCRATCH_READY ||
         phase == IREE_HAL_AMD_GPU_AQL_SCRATCH_RELEASING;
}

static void iree_hal_amd_gpu_aql_scratch_released(void* user_data) {
  iree_hal_amd_gpu_aql_scratch_t* scratch = user_data;
  iree_hal_amd_gpu_context_t* context = scratch->growth->context;
  // IDLE ends this callback's queue borrow. Only the device wake may follow.
  iree_atomic_store(&scratch->phase, IREE_HAL_AMD_GPU_AQL_SCRATCH_IDLE,
                    iree_memory_order_release);
  iree_hal_amd_gpu_aql_scratch_wake(context);
}

amdf_status_t iree_hal_amd_gpu_aql_scratch_trim(
    iree_hal_amd_gpu_aql_scratch_t* scratch) {
  if (!scratch->memory.handle) {
    return AMDF_STATUS_OK;
  }
  const amdf_status_t status =
      scratch->growth->context->gpu->user_queue_trim_scratch(scratch->queue);
  if (amdf_status_is_ok(status)) {
    iree_atomic_store(&scratch->phase, IREE_HAL_AMD_GPU_AQL_SCRATCH_RELEASING,
                      iree_memory_order_release);
    iree_hal_amd_gpu_context_release_memory(
        scratch->growth->context, 1, &scratch->memory,
        (iree_hal_amd_fabric_memory_release_callback_t){
            .fn = iree_hal_amd_gpu_aql_scratch_released, .user_data = scratch},
        &scratch->release);
  }
  return status;
}

void iree_hal_amd_gpu_aql_scratch_deinitialize(
    iree_hal_amd_gpu_aql_scratch_t* scratch) {
  iree_hal_amd_fabric_memory_deinitialize(scratch->growth->context->api,
                                          scratch->growth->context->event_sink,
                                          &scratch->memory);
  iree_status_free(scratch->status);
}
