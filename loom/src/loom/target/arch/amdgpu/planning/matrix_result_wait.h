// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pending MFMA result and accumulator-read windows over physical VGPR and AGPR
// storage.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_MATRIX_RESULT_WAIT_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_MATRIX_RESULT_WAIT_H_

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/packet.h"
#include "loom/target/arch/amdgpu/planning/matrix_wait_states.h"
#include "loom/target/arch/amdgpu/planning/structural_packet.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_matrix_result_wait_t
    loom_amdgpu_matrix_result_wait_t;

// Complete forwarding identity. All eight bytes participate in equality;
// instruction forms are normalized away on CDNA2, which forwards by range.
typedef struct loom_amdgpu_matrix_result_identity_t {
  // Result start in the component's disjoint physical-bank index space.
  uint32_t base;
  // Number of physical result registers.
  uint16_t count;
  // Canonical instruction form, or zero when forwarding ignores the form.
  uint16_t descriptor_ref;
} loom_amdgpu_matrix_result_identity_t;

// Source facts retained once in final packet order by counter-wait planning.
typedef struct loom_amdgpu_matrix_result_source_t {
  // Scheduled packet opening the result window.
  uint32_t packet_index;
  // Schedule node owning the producer for diagnostic provenance.
  uint32_t node_index;
  // Canonical instruction identity and execution pipeline.
  loom_amdgpu_matrix_instruction_t instruction;
  // Physical result range and normalized accumulator-forwarding identity.
  loom_amdgpu_matrix_result_identity_t identity;
  // Physical accumulator input start; its width matches identity.count.
  uint32_t accumulator_base;
  // Packet operand carrying the accumulator input.
  uint8_t accumulator_operand;
  // Exact-range floor: zero, two slots (1), or four slots (2).
  uint8_t floor_ordinal;
  // Ordinary VALU read, memory read and overwrite windows.
  uint8_t ordinary[3];
  // Partial accumulator windows indexed by consumer pipeline.
  uint8_t accumulator[LOOM_AMDGPU_MATRIX_PIPELINE_COUNT_];
  // Slots before an ordinary write may overwrite the accumulator input.
  uint8_t accumulator_read;
} loom_amdgpu_matrix_result_source_t;

typedef struct loom_amdgpu_matrix_result_match_t {
  // Source opening the strongest pending window, or NULL when satisfied.
  const loom_amdgpu_matrix_result_source_t* source;
  // Concrete access to that producer's physical result or accumulator input.
  loom_amdgpu_matrix_wait_result_use_t use;
  // Full producer-to-consumer separation.
  uint16_t required;
  // Remaining instruction slots before this use.
  uint16_t remaining;
} loom_amdgpu_matrix_result_match_t;

// Allocates the transient owner and exact source stream counted by the existing
// node classifier. Returns NULL without allocating for a non-MFMA profile or a
// function without MFMAs. The arena spans counter-wait collection and final
// fixed-wait planning.
iree_status_t loom_amdgpu_matrix_result_wait_allocate(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    loom_amdgpu_matrix_wait_profile_t profile, iree_host_size_t source_count,
    iree_arena_allocator_t* arena,
    loom_amdgpu_matrix_result_wait_t** out_state);

// Collects static exits during the existing counter-wait traversal. Progress
// excludes elided packets and inserted waits, and uses the minimum native path
// length of a structural packet. Ordinary writes never delete matrix windows.
void loom_amdgpu_matrix_result_wait_begin_static_block(
    loom_amdgpu_matrix_result_wait_t* state, uint16_t block_index);
void loom_amdgpu_matrix_result_wait_collect(
    loom_amdgpu_matrix_result_wait_t* state,
    const loom_low_packet_view_t* packet, bool matrix_source,
    uint8_t minimum_instruction_count, bool quiescent_boundary);
iree_status_t loom_amdgpu_matrix_result_wait_end_static_block(
    loom_amdgpu_matrix_result_wait_t* state);

// Joins retained exits through the shared CFG solver without revisiting IR.
iree_status_t loom_amdgpu_matrix_result_wait_finalize_static(
    loom_amdgpu_matrix_result_wait_t* state);

// Begins final planning with refined predecessor exits where available and
// conservative static exits for backedges.
void loom_amdgpu_matrix_result_wait_begin_block(
    loom_amdgpu_matrix_result_wait_t* state, uint16_t block_index);

// Returns the retained source for this packet, or NULL for an ordinary packet.
const loom_amdgpu_matrix_result_source_t*
loom_amdgpu_matrix_result_wait_source_at(
    const loom_amdgpu_matrix_result_wait_t* state, uint32_t packet_index);

// Matches actual physical accesses, including final sequential copies. A
// coalesced SSA edge has no native read. Structural facts are supplied by the
// owning packet traversal; memory consumers use their descriptor's read class.
loom_amdgpu_matrix_result_match_t loom_amdgpu_matrix_result_wait_inspect(
    const loom_amdgpu_matrix_result_wait_t* state,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_matrix_result_source_t* source,
    const loom_amdgpu_structural_packet_info_t* structural,
    bool memory_consumer);

// Credits guaranteed native progress, including inserted waits. Committing a
// matrix packet publishes its result after its own instruction has advanced.
void loom_amdgpu_matrix_result_wait_advance(
    loom_amdgpu_matrix_result_wait_t* state, uint64_t instruction_count);

// Returns the strongest live result window and starts a fresh physical epoch.
// Callers use this at boundaries whose callee contract includes result state.
loom_amdgpu_matrix_result_match_t loom_amdgpu_matrix_result_wait_drain(
    loom_amdgpu_matrix_result_wait_t* state);

void loom_amdgpu_matrix_result_wait_commit(
    loom_amdgpu_matrix_result_wait_t* state,
    const loom_amdgpu_matrix_result_source_t* source);

// Retains the final exit with actual inserted-wait progress for later blocks.
void loom_amdgpu_matrix_result_wait_end_block(
    loom_amdgpu_matrix_result_wait_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_MATRIX_RESULT_WAIT_H_
