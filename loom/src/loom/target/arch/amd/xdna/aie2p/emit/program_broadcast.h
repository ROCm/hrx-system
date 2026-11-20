// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_BROADCAST_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_BROADCAST_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

// One fully relocated initial program section on a worker in a physical row.
typedef struct loom_aie2p_worker_code_t {
  // Word-aligned destination address in the tile-control aperture.
  uint32_t address;
  // Native instruction bytes at that destination.
  iree_const_byte_span_t contents;
} loom_aie2p_worker_code_t;

// Tile-control writes in increasing program-address order, before completion.
typedef struct loom_aie2p_worker_code_stream_t {
  // Packet headers, commands, and data in host word order.
  uint32_t* words;
  // Number of populated words. Zero means this delivery can be elided.
  iree_host_size_t word_count;
} loom_aie2p_worker_code_stream_t;

typedef struct loom_aie2p_program_broadcast_t {
  // First delivery, received by every participating worker in the row.
  loom_aie2p_worker_code_stream_t shared;
  // Subsequent private writes indexed in the same order as the input workers.
  loom_aie2p_worker_code_stream_t* individual;
} loom_aie2p_program_broadcast_t;

// Partitions linked instructions while the receiving workers remain in reset.
// A common write can touch only bytes owned by every receiver. Differences at
// that address are repaired through the receiver's individual stream before
// activation. Equal-frequency payloads prefer the first input worker.
iree_status_t loom_aie2p_program_broadcast_build(
    const loom_aie2p_worker_code_t* workers, iree_host_size_t worker_count,
    iree_arena_allocator_t* arena, loom_aie2p_program_broadcast_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_BROADCAST_H_
