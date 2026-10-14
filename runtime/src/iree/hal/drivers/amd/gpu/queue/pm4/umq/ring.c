// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/umq/ring.h"

#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/drivers/amd/status.h"

static_assert(sizeof(iree_hal_amd_gpu_pm4_ring_t) <= 24,
              "ring cursors must leave room for publication operands");
static_assert(sizeof(iree_hal_amd_gpu_pm4_ring_reservation_t) <= 16,
              "one reservation is a destination and a monotonic position");

void iree_hal_amd_gpu_pm4_ring_initialize(
    uint32_t* words, uint32_t capacity, iree_hal_amd_gpu_pm4_ring_t* out_ring) {
  *out_ring = (iree_hal_amd_gpu_pm4_ring_t){
      .words = words,
      .capacity = capacity,
  };
}

bool iree_hal_amd_gpu_pm4_ring_try_reserve(
    iree_hal_amd_gpu_pm4_ring_t* ring, uint64_t consumed_words,
    uint32_t word_count,
    iree_hal_amd_gpu_pm4_ring_reservation_t* out_reservation) {
  uint32_t padding = (0u - word_count) & 7u;
  // Type-3 NOP requires a header and at least one following DWORD.
  if (padding == 1) {
    padding += 8;
  }
  const uint32_t span = word_count + padding;
  const uint32_t offset = ring->reserved_words & (ring->capacity - 1);
  const uint32_t tail =
      offset + span > ring->capacity ? ring->capacity - offset : 0;
  if (ring->reserved_words - consumed_words + tail + span >= ring->capacity) {
    return false;
  }
  if (tail) {
    iree_hal_amd_gpu_pm4_emit_nop(ring->words + offset, tail);
  }
  uint32_t* words = ring->words + (tail ? 0 : offset);
  if (padding) {
    iree_hal_amd_gpu_pm4_emit_nop(words + word_count, padding);
  }
  *out_reservation = (iree_hal_amd_gpu_pm4_ring_reservation_t){
      .words = words,
      .position = ring->reserved_words + tail,
  };
  ring->reserved_words += tail + span;
  return true;
}

iree_status_t iree_hal_amd_gpu_pm4_ring_storage_initialize(
    const iree_hal_amd_gpu_context_t* context,
    const amdf_gpu_user_queue_create_info_t* create_info,
    iree_hal_amd_gpu_pm4_ring_storage_t* out_storage,
    iree_hal_amd_gpu_pm4_ring_t* out_ring,
    amdf_user_queue_mapping_info_t* out_mapping_info) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_storage = (iree_hal_amd_gpu_pm4_ring_storage_t){.context = context};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_HAL_AMD_STATUS_FROM_AMDF(
              context->gpu->user_queue_create(context->device, create_info,
                                              &out_storage->queue),
              "gpu.user_queue_create"));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_HAL_AMD_STATUS_FROM_AMDF(
              context->api->user_queue_map(out_storage->queue, NULL,
                                           &out_storage->mapping),
              "user_queue_map"));
  amdf_user_queue_mapping_info_t mapping_info = {
      .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO,
      .structure_size = sizeof(mapping_info),
  };
  iree_status_t status =
      IREE_HAL_AMD_STATUS_FROM_AMDF(context->api->user_queue_mapping_query_info(
                                        out_storage->mapping, &mapping_info),
                                    "user_queue_mapping_query_info");
  if (iree_status_is_ok(status)) {
    iree_hal_amd_gpu_pm4_ring_initialize(
        (uint32_t*)(uintptr_t)mapping_info.ring_address,
        (uint32_t)(mapping_info.ring_byte_length / sizeof(uint32_t)), out_ring);
    *out_mapping_info = mapping_info;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

amdf_status_t iree_hal_amd_gpu_pm4_ring_storage_deinitialize(
    iree_hal_amd_gpu_pm4_ring_storage_t* storage, uint64_t published_words) {
  if (!amdf_status_is_ok(storage->removal_status)) {
    return storage->removal_status;
  }
  const amdf_api_t* api = storage->context->api;
  if (storage->queue && published_words) {
    amdf_status_t status = api->user_queue_wait_consumed(
        storage->queue, published_words, AMDF_TIMEOUT_INFINITE, 0);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  if (storage->mapping) {
    amdf_status_t status = api->user_queue_mapping_destroy(storage->mapping);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    storage->mapping = NULL;
  }
  if (storage->queue) {
    amdf_status_t status = api->user_queue_destroy(storage->queue);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
      return status;
    }
    storage->queue = NULL;
    storage->removal_status = status;
    return status;
  }
  return AMDF_STATUS_OK;
}
