// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/umq/ring.h"

#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"

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
