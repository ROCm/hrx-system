// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_JOIN_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_JOIN_H_

#include "iree/base/internal/arena.h"
#include "loom/target/arch/amd/xdna/array/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_aie2p_join_action_t {
  // Core-relative selector of the shared memory module's semaphore.
  uint16_t selector;
  // Acquire consumes one credit; release publishes one credit.
  bool acquire;
} loom_aie2p_join_action_t;

typedef struct loom_aie2p_join_program_t {
  // Ordered gather, parent rendezvous, and child release instructions.
  const loom_aie2p_join_action_t* actions;
  // Number of semaphore operations at each phase boundary.
  iree_host_size_t count;
} loom_aie2p_join_program_t;

typedef struct loom_aie2p_join_locks_t {
  // Column-major physical memory-module index.
  iree_host_size_t owner;
  // Arrival semaphore followed by the departure semaphore; both start at zero.
  uint16_t first;
} loom_aie2p_join_locks_t;

typedef struct loom_aie2p_join_plan_t {
  // Program per physical tile, empty for absent or singleton participants.
  loom_aie2p_join_program_t* programs;
  // Distinct semaphore pairs requiring initialization before cores start.
  loom_aie2p_join_locks_t* locks;
  // Number of edges in the selected tree.
  iree_host_size_t edge_count;
} loom_aie2p_join_plan_t;

// Estimates whether repeated expansion of this fixed lock program costs more
// code than one leaf helper and its call boundaries. Ties remain inline. The
// final packet plan, including allocation and alignment, owns exact byte size.
bool loom_aie2p_join_program_prefers_sharing(
    const loom_aie2p_join_program_t* program, iree_host_size_t use_count);

// Selects a reusable gather/release tree over common core lock apertures.
// participants and allocated_locks have one entry per column-major physical
// tile. Only compute cores participate; other tiles may own reachable locks.
// Capacity and connectivity depend on authored placement. An invalid plan
// leaves allocated_locks unchanged; a valid plan reserves its semaphore pairs.
// Returned storage belongs to arena. No source control or worker analysis
// occurs.
iree_status_t loom_aie2p_join_plan_build(const loom_xdna_array_family_t* family,
                                         const bool* participants,
                                         uint16_t* allocated_locks,
                                         iree_arena_allocator_t* arena,
                                         loom_aie2p_join_plan_t* out_plan,
                                         bool* out_valid);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_JOIN_H_
