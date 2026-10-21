// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Gfx12.0 SDMA packet encoding for native queues. Callers own capacity,
// publication, queue context, and memory lifetime. No packet is written on
// validation failure. Capacities and return values are in DWORDs; a zero
// return value indicates invalid arguments or insufficient storage.
#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_SDMA_EMITTER_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_SDMA_EMITTER_H_

#include <stdbool.h>
#include <stdint.h>

// Public packet definitions and cache-control sequences:
// https://github.com/ROCm/ROCR-Runtime/blob/amd-staging/runtime/hsa-runtime/core/inc/sdma_registers.h
// https://github.com/ROCm/ROCR-Runtime/blob/amd-staging/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp
// INDIRECT placement follows sdma_v7_0_ring_emit_ib:
// https://github.com/torvalds/linux/blob/v7.0/drivers/gpu/drm/amd/amdgpu/sdma_v7_0.c
enum {
  IREE_HAL_AMDGPU_SDMA_COPY_DWORDS = 7,
  IREE_HAL_AMDGPU_SDMA_POLL_DWORDS = 6,
  IREE_HAL_AMDGPU_SDMA_FENCE_DWORDS = 4,
  IREE_HAL_AMDGPU_SDMA_GCR_DWORDS = 5,
  IREE_HAL_AMDGPU_SDMA_TIMESTAMP_DWORDS = 3,
  IREE_HAL_AMDGPU_SDMA_COPY_MAX_BYTES = 1u << 22,
};

static inline uint32_t iree_hal_amdgpu_sdma_emit_copy(uint32_t capacity,
                                                      uint32_t* dwords,
                                                      uint64_t source,
                                                      uint64_t target,
                                                      uint32_t byte_count) {
  if (!dwords || capacity < IREE_HAL_AMDGPU_SDMA_COPY_DWORDS || !byte_count ||
      byte_count > IREE_HAL_AMDGPU_SDMA_COPY_MAX_BYTES ||
      source > UINT64_MAX - (byte_count - 1) ||
      target > UINT64_MAX - (byte_count - 1))
    return 0;
  dwords[0] = 1;  // COPY, LINEAR.
  dwords[1] = byte_count - 1;
  dwords[2] = 0;
  dwords[3] = (uint32_t)source;
  dwords[4] = (uint32_t)(source >> 32);
  dwords[5] = (uint32_t)target;
  dwords[6] = (uint32_t)(target >> 32);
  return IREE_HAL_AMDGPU_SDMA_COPY_DWORDS;
}

// Emits NOP padding followed by a native INDIRECT packet. |stream_dword_offset|
// is the packet insertion position in the executing ring, not the address of
// this host staging buffer. The packet must end on an eight-DWORD boundary.
// Requires an IB-enabled native queue and a caller-qualified VMID/CSA contract.
// The IB base is 32-byte aligned; length is a nonzero 20-bit DWORD count.
// A nonzero CSA points to caller-owned context-save storage kept live with the
// IB. Encoding success alone does not establish queue support or safe
// preemption. No privilege bit is emitted. Returns the total DWORD count
// including padding.
static inline uint32_t iree_hal_amdgpu_sdma_emit_indirect(
    uint32_t capacity, uint32_t* dwords, uint64_t stream_dword_offset,
    uint64_t ib_address, uint32_t ib_dword_count, uint32_t vmid,
    uint64_t csa_address) {
  const uint32_t padding = (2u - (uint32_t)stream_dword_offset) & 7u;
  const uint32_t count = padding + 6u;
  if (!dwords || capacity < count || !ib_address || (ib_address & 31u) ||
      !ib_dword_count || ib_dword_count > 0xfffffu || vmid > 15u ||
      (csa_address & 3u) ||
      ib_address > UINT64_MAX - ((uint64_t)ib_dword_count * 4u - 1u))
    return 0;
  for (uint32_t i = 0; i < padding; ++i) dwords[i] = 0;
  dwords[padding + 0] = 4u | (vmid << 16);
  dwords[padding + 1] = (uint32_t)ib_address;
  dwords[padding + 2] = (uint32_t)(ib_address >> 32);
  dwords[padding + 3] = ib_dword_count;
  dwords[padding + 4] = (uint32_t)csa_address;
  dwords[padding + 5] = (uint32_t)(csa_address >> 32);
  return count;
}

// Polls a DWORD in shared memory until equal. Use a separate ready word per
// batch to avoid skipping a transient equality milestone. 0xfff retries is
// the hardware's indefinite-wait encoding; host timeout/cancellation is
// external.
static inline uint32_t iree_hal_amdgpu_sdma_emit_poll32(uint32_t capacity,
                                                        uint32_t* dwords,
                                                        uint64_t address,
                                                        uint32_t value) {
  if (!dwords || capacity < IREE_HAL_AMDGPU_SDMA_POLL_DWORDS || (address & 3))
    return 0;
  dwords[0] = 8u | (3u << 28) | (1u << 31);
  dwords[1] = (uint32_t)address;
  dwords[2] = (uint32_t)(address >> 32);
  dwords[3] = value;
  dwords[4] = UINT32_MAX;
  dwords[5] = 4u | (0xfffu << 16);
  return IREE_HAL_AMDGPU_SDMA_POLL_DWORDS;
}

// Writes a DWORD to system memory (gfx12 mtype=3, sys=1). Payload ordering
// must be supplied by the stream, including the appropriate GCR operation.
static inline uint32_t iree_hal_amdgpu_sdma_emit_fence32(uint32_t capacity,
                                                         uint32_t* dwords,
                                                         uint64_t address,
                                                         uint32_t value) {
  if (!dwords || capacity < IREE_HAL_AMDGPU_SDMA_FENCE_DWORDS || (address & 3))
    return 0;
  dwords[0] = 5u | (3u << 16) | (1u << 20);
  dwords[1] = (uint32_t)address;
  dwords[2] = (uint32_t)(address >> 32);
  dwords[3] = value;
  return IREE_HAL_AMDGPU_SDMA_FENCE_DWORDS;
}

// Matches ROCr's gfx10-12.0 USER_GCR acquire/release sequences. Full range.
static inline uint32_t iree_hal_amdgpu_sdma_emit_gcr(uint32_t capacity,
                                                     uint32_t* dwords,
                                                     bool invalidate) {
  if (!dwords || capacity < IREE_HAL_AMDGPU_SDMA_GCR_DWORDS) return 0;
  dwords[0] = 0x11u | (1u << 8);
  dwords[1] = 0;
  dwords[2] = (1u << 31) | (1u << 22);
  if (invalidate)
    dwords[2] |= (1u << 30) | (1u << 25) | (1u << 24) | (1u << 23);
  dwords[3] = 0;
  dwords[4] = 0;
  return IREE_HAL_AMDGPU_SDMA_GCR_DWORDS;
}

// GET_GLOBAL timestamps require 32-byte-aligned output storage.
static inline uint32_t iree_hal_amdgpu_sdma_emit_timestamp(uint32_t capacity,
                                                           uint32_t* dwords,
                                                           uint64_t address) {
  if (!dwords || capacity < IREE_HAL_AMDGPU_SDMA_TIMESTAMP_DWORDS ||
      (address & 31))
    return 0;
  dwords[0] = 13u | (2u << 8);
  dwords[1] = (uint32_t)address;
  dwords[2] = (uint32_t)(address >> 32);
  return IREE_HAL_AMDGPU_SDMA_TIMESTAMP_DWORDS;
}

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_SDMA_EMITTER_H_
