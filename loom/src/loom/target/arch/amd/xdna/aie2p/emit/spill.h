// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native stack traffic for the shared allocator's retained spill decisions.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_SPILL_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_SPILL_H_

#include "loom/codegen/low/allocation_materialization.h"

#ifdef __cplusplus
extern "C" {
#endif

// One native component of a register payload. Byte positions are affine in
// the logical unit and complete slot size, allowing packed component planes.
typedef struct loom_aie2p_spill_piece_t {
  // Native descriptor ordinals, indexed by store (0=load) and far addressing.
  uint16_t descriptors[2][2];
  // Loads preserving already defined parts, indexed by far addressing.
  uint16_t continuation_descriptors[2];
  // Constant byte position within the component plane.
  uint16_t byte_offset;
  // Byte stride between logical units within this component plane.
  uint16_t unit_byte_stride;
  // Bytes per complete slot unit preceding this component plane.
  uint16_t plane_byte_stride;
  // Defined register parts transferred, or zero for a complete carrier.
  uint32_t register_parts;
} loom_aie2p_spill_piece_t;

// Native transfer for a complete carrier or one chunk of a register tuple.
// Multiple pieces construct disjoint parts through tied continuations.
typedef struct loom_aie2p_spill_recipe_t {
  // Native carrier class; a differing source class uses an ordinary Low copy.
  uint16_t reg_class_id;
  // First component in the immutable piece table.
  uint16_t piece_start;
  // Number of native components transferred in order.
  uint8_t piece_count;
  // Number of logical register units per recipe instance.
  uint8_t unit_count;
} loom_aie2p_spill_recipe_t;

// Plans native transfers and extends the append-only stack projection supplied
// in user_data as loom_aie2p_stack_layout_t. Recipe storage belongs to arena;
// the stack projection outlives all allocation repair batches.
iree_status_t loom_aie2p_spill_plan(
    void* user_data, const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* spill_plans,
    iree_host_size_t spill_plan_count,
    const loom_low_spill_materialization_plan_t* materialization_plan,
    iree_arena_allocator_t* arena, iree_diagnostic_emitter_t emitter,
    bool* out_accepted,
    loom_low_spill_materialization_traffic_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_SPILL_H_
