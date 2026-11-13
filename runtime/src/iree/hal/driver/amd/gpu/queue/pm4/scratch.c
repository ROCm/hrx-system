// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/pm4/scratch.h"

#include "iree/hal/driver/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/driver/amd/status.h"

iree_status_t iree_hal_amd_gpu_pm4_scratch_profile_initialize(
    const amdf_gpu_endpoint_info_t* target,
    amdf_queue_format_features_t features,
    iree_hal_amd_gpu_pm4_scratch_profile_t* out_profile) {
  *out_profile = (iree_hal_amd_gpu_pm4_scratch_profile_t){0};
  if (!(features & AMDF_GPU_PM4_FORMAT_FEATURE_DISPATCH_SCRATCH)) {
    return iree_ok_status();
  }
  const uint32_t engine_count = target->topology.shader_engine_count_per_xcc;
  if (!engine_count || !target->compute.compute_unit_count ||
      !target->compute.maximum_scratch_wave_count_per_compute_unit ||
      target->topology.xcc_count != 1) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "native PM4 scratch geometry is unavailable");
  }
  const uint64_t waves_per_engine =
      (((uint64_t)target->compute.compute_unit_count + engine_count - 1) /
       engine_count) *
      target->compute.maximum_scratch_wave_count_per_compute_unit;
  if (waves_per_engine > 0xfffu) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "native PM4 scratch wave capacity exceeds the "
                            "register field");
  }
  *out_profile = (iree_hal_amd_gpu_pm4_scratch_profile_t){
      .size_register = target->gfx_ip.major == 12 && target->gfx_ip.minor == 5
                           ? 0x2e16
                           : 0x2e18,
      .maximum_wave_byte_length =
          (target->gfx_ip.major >= 12 ? 0x3ffffu : 0x7fffu) << 8,
      .waves_per_engine = (uint32_t)waves_per_engine,
      .engine_count = engine_count,
  };
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_scratch_storage_prepare(
    uint32_t wave_byte_length, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_pm4_scratch_storage_t** out_storage) {
  *out_storage = NULL;
  iree_hal_amd_gpu_pm4_scratch_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*storage),
                                             (void**)&storage));
  storage->host_allocator = host_allocator;
  storage->wave_byte_length = wave_byte_length;
  *out_storage = storage;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_scratch_storage_initialize(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_pm4_scratch_storage_t* storage) {
  return IREE_HAL_AMD_STATUS_FROM_AMDF(
      context->api->memory_query_address(storage->memory.handle, 0,
                                         AMDF_MEMORY_ADDRESS_GPU,
                                         &storage->address),
      "query scratch address");
}

static void iree_hal_amd_gpu_pm4_scratch_storage_released(void* user_data) {
  iree_hal_amd_gpu_pm4_scratch_storage_t* storage = user_data;
  iree_allocator_free(storage->host_allocator, storage);
}

void iree_hal_amd_gpu_pm4_scratch_storage_free(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_pm4_scratch_storage_t* storage) {
  while (storage) {
    iree_hal_amd_gpu_pm4_scratch_storage_t* next = storage->next;
    iree_hal_amd_gpu_context_release_memory(
        context, 1, &storage->memory,
        (iree_hal_amd_fabric_memory_release_callback_t){
            .fn = iree_hal_amd_gpu_pm4_scratch_storage_released,
            .user_data = storage},
        &storage->release);
    storage = next;
  }
}

void iree_hal_amd_gpu_pm4_scratch_install(
    iree_hal_amd_gpu_context_t* context,
    iree_hal_amd_gpu_pm4_scratch_storage_t* storage, uint64_t completed,
    iree_hal_amd_gpu_pm4_scratch_t* scratch) {
  if (scratch->current) {
    if (scratch->current->last_use > completed) {
      scratch->current->next = scratch->retired;
      scratch->retired = scratch->current;
      scratch->binding_word_count = 9;
    } else {
      iree_hal_amd_gpu_pm4_scratch_storage_free(context, scratch->current);
    }
  }
  scratch->current = storage;
  scratch->binding_word_count = iree_max(scratch->binding_word_count, 7u);
}

uint32_t* iree_hal_amd_gpu_pm4_scratch_emit(
    uint32_t* words, const iree_hal_amd_gpu_pm4_scratch_t* scratch,
    uint32_t word_count) {
  if (!word_count) {
    return words;
  }
  if (word_count == 9) {
    words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
  }
  const uint64_t address = scratch->current->address;
  const uint32_t base[] = {(uint32_t)(address >> 8), (uint32_t)(address >> 40)};
  words = iree_hal_amd_gpu_pm4_emit_registers(words, 0x2e10, 2, base);
  const uint32_t size = scratch->profile.waves_per_engine |
                        ((scratch->current->wave_byte_length >> 8) << 12);
  return iree_hal_amd_gpu_pm4_emit_registers(
      words, scratch->profile.size_register, 1, &size);
}

iree_hal_amd_gpu_pm4_scratch_storage_t* iree_hal_amd_gpu_pm4_scratch_publish(
    iree_hal_amd_gpu_pm4_scratch_t* scratch, uint64_t epoch,
    uint32_t required) {
  if (required) {
    scratch->current->last_use = epoch;
    scratch->binding_word_count = 0;
  }
  iree_hal_amd_gpu_pm4_scratch_storage_t* retired = scratch->retired;
  scratch->retired = NULL;
  return retired;
}

void iree_hal_amd_gpu_pm4_scratch_trim(
    iree_hal_amd_gpu_context_t* context, uint64_t completed,
    iree_hal_amd_gpu_pm4_scratch_t* scratch) {
  if (scratch->current && scratch->current->last_use <= completed) {
    iree_hal_amd_gpu_pm4_scratch_storage_free(context, scratch->current);
    scratch->current = NULL;
  }
  iree_hal_amd_gpu_pm4_scratch_storage_t** link = &scratch->retired;
  while (*link) {
    iree_hal_amd_gpu_pm4_scratch_storage_t* storage = *link;
    if (storage->last_use <= completed) {
      *link = storage->next;
      storage->next = NULL;
      iree_hal_amd_gpu_pm4_scratch_storage_free(context, storage);
    } else {
      link = &storage->next;
    }
  }
}
