// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/sdma_queue.h"

#include <string.h>

iree_status_t iree_hal_amdgpu_sdma_queue_initialize(
    const iree_hal_amdgpu_sdma_queue_params_t* params,
    iree_hal_amdgpu_sdma_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(out_queue);
  if (!params->libhsa || !params->agent.handle ||
      params->capacity_bytes < 256 ||
      (params->capacity_bytes & (params->capacity_bytes - 1))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid native SDMA queue parameters");
  }
  iree_hal_amdgpu_sdma_queue_t queue = {0};
  queue.libhsa = params->libhsa;
  queue.agent = params->agent;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_sdma_query_capabilities(
      params->gfxip_version, &queue.capabilities));
  hsa_amd_queue_create_desc_t descriptor = {
      .version = HSA_AMD_QUEUE_CREATE_DESC_VERSION,
      .flags = HSA_AMD_QUEUE_CREATE_SYSTEM_MEM,
      .engine_type = HSA_AMD_QUEUE_ENGINE_SDMA,
      .queue_size_bytes = params->capacity_bytes,
      .priority = params->priority,
      .callback = params->error_callback,
      .callback_data = params->error_callback_data,
      .engine.sdma = {.sdma_engine_id = params->engine_id},
  };
  iree_status_t status = iree_hsa_amd_queue_create(
      IREE_LIBHSA(params->libhsa), params->agent, &descriptor, 1);
  queue.handle = descriptor.queue;
  if (iree_status_is_ok(status) && !queue.handle) {
    status =
        iree_make_status(IREE_STATUS_INTERNAL, "HSA returned no SDMA queue");
  }
  uint64_t read_address = 0, write_address = 0;
  if (iree_status_is_ok(status))
    status = iree_hsa_amd_queue_get_info(
        IREE_LIBHSA(params->libhsa), queue.handle,
        HSA_AMD_QUEUE_INFO_READ_POINTER, &read_address);
  if (iree_status_is_ok(status))
    status = iree_hsa_amd_queue_get_info(
        IREE_LIBHSA(params->libhsa), queue.handle,
        HSA_AMD_QUEUE_INFO_WRITE_POINTER, &write_address);
  if (iree_status_is_ok(status))
    status = iree_hsa_amd_queue_get_info(
        IREE_LIBHSA(params->libhsa), queue.handle,
        HSA_AMD_QUEUE_INFO_SDMA_ENGINE_ID, &queue.engine_id);
  if (iree_status_is_ok(status)) {
    const amd_signal_t* signal =
        (const amd_signal_t*)queue.handle->doorbell_signal.handle;
    if (!signal || signal->kind != AMD_SIGNAL_KIND_DOORBELL ||
        !signal->hardware_doorbell_ptr) {
      status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "SDMA queue has no hardware doorbell");
    } else {
      status = iree_hal_amdgpu_sdma_ring_initialize(
          (uint32_t*)queue.handle->base_address, queue.handle->size,
          (uint64_t*)read_address, (uint64_t*)write_address,
          signal->hardware_doorbell_ptr, &queue.ring);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_queue = queue;
  } else if (queue.handle) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_queue_destroy_raw(params->libhsa, queue.handle));
  }
  return status;
}

void iree_hal_amdgpu_sdma_queue_deinitialize(
    iree_hal_amdgpu_sdma_queue_t* queue) {
  if (queue->handle)
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_queue_destroy_raw(queue->libhsa, queue->handle));
  memset(queue, 0, sizeof(*queue));
}

iree_status_t iree_hal_amdgpu_sdma_queue_elapsed_ns(
    const iree_hal_amdgpu_sdma_queue_t* queue, uint64_t start_tick,
    uint64_t end_tick, double* out_elapsed_ns) {
  IREE_ASSERT_ARGUMENT(out_elapsed_ns);
  if (!queue->handle || end_tick < start_tick)
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid SDMA timestamp range");
  uint64_t start = 0, end = 0, frequency = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_profiling_convert_tick_to_system_domain(
      IREE_LIBHSA(queue->libhsa), queue->agent, start_tick, &start));
  IREE_RETURN_IF_ERROR(iree_hsa_amd_profiling_convert_tick_to_system_domain(
      IREE_LIBHSA(queue->libhsa), queue->agent, end_tick, &end));
  IREE_RETURN_IF_ERROR(iree_hsa_system_get_info(
      IREE_LIBHSA(queue->libhsa), HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY,
      &frequency));
  if (!frequency || end < start)
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "invalid converted SDMA timestamp range");
  *out_elapsed_ns = (double)(end - start) * 1e9 / (double)frequency;
  return iree_ok_status();
}
