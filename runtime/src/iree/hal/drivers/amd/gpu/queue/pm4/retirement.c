// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/retirement.h"

#include "iree/base/internal/math.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"

static_assert(sizeof(iree_hal_amd_gpu_pm4_retirement_record_t) == 16,
              "PM4 completion records must remain four per cache line");
static_assert(offsetof(iree_hal_amd_gpu_pm4_retirement_record_t, generation) ==
                  8,
              "generation commits the preceding eight-byte pointer payload");

void iree_hal_amd_gpu_pm4_retirement_ring_initialize(
    iree_hal_amd_gpu_pm4_retirement_record_t* records, uint64_t device_address,
    uint32_t capacity, iree_hal_amd_gpu_pm4_retirement_ring_t* out_ring) {
  *out_ring = (iree_hal_amd_gpu_pm4_retirement_ring_t){
      .records = records,
      .device_address = device_address,
      .capacity_mask = capacity - 1,
      .capacity_shift = iree_math_count_trailing_zeros_u32(capacity),
  };
  for (uint32_t i = 0; i < capacity; ++i) {
    records[i].action = 0;
    iree_atomic_store(&records[i].generation, 0, iree_memory_order_relaxed);
    records[i].reserved = 0;
  }
}

uint32_t* iree_hal_amd_gpu_pm4_retirement_emit(
    uint32_t* words, const iree_hal_amd_gpu_pm4_retirement_ring_t* ring,
    uint64_t position, void* action, uint32_t release_cache_control) {
  const uint64_t address = ring->device_address +
                           (position & ring->capacity_mask) *
                               sizeof(iree_hal_amd_gpu_pm4_retirement_record_t);
  const uint64_t action_bits = (uintptr_t)action;
  const uint32_t action_words[] = {(uint32_t)action_bits,
                                   (uint32_t)(action_bits >> 32)};
  words = iree_hal_amd_gpu_pm4_emit_write(words, address, 2, action_words);
  const uint32_t generation = (uint32_t)(position >> ring->capacity_shift) + 1;
  return iree_hal_amd_gpu_pm4_emit_release32(
      words, release_cache_control,
      address + offsetof(iree_hal_amd_gpu_pm4_retirement_record_t, generation),
      generation);
}

void* iree_hal_amd_gpu_pm4_retirement_poll(
    const iree_hal_amd_gpu_pm4_retirement_ring_t* ring, uint64_t position) {
  const iree_hal_amd_gpu_pm4_retirement_record_t* record =
      &ring->records[position & ring->capacity_mask];
  const uint32_t generation = (uint32_t)(position >> ring->capacity_shift) + 1;
  if ((uint32_t)iree_atomic_load(&record->generation,
                                 iree_memory_order_acquire) != generation) {
    return NULL;
  }
  return (void*)(uintptr_t)record->action;
}
