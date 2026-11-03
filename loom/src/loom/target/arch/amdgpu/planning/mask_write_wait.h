// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GFX11 wave64 lane-mask storage reuse over final physical packets.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_MASK_WRITE_WAIT_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_MASK_WRITE_WAIT_H_

#include "loom/target/arch/amdgpu/planning/structural_packet.h"
#include "loom/target/arch/amdgpu/planning/wait_states.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_mask_write_wait_t loom_amdgpu_mask_write_wait_t;

// Emits a residual after a physical overwrite, in scheduled packet order.
typedef iree_status_t (*loom_amdgpu_mask_write_wait_emit_fn_t)(
    void* user_data, const loom_amdgpu_wait_state_t* wait_state);

// Creates transient collection state for an affected wave64 function. The
// caller selects applicability from processor facts before calling.
iree_status_t loom_amdgpu_mask_write_wait_allocate(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_arena_allocator_t* arena, loom_amdgpu_mask_write_wait_t** out_state);

// Collects mask reads and physical overwrites from the owning final-packet
// traversal. Structural moves are already resolved into native execution order.
// |quiescent_call_boundary| selects whether a structural call restores this
// family before transferring control.
iree_status_t loom_amdgpu_mask_write_wait_collect(
    loom_amdgpu_mask_write_wait_t* state, const loom_low_packet_view_t* packet,
    const loom_amdgpu_structural_packet_info_t* structural,
    loom_amdgpu_descriptor_traits_t traits, bool quiescent_call_boundary);

// Retires every mask-read destination class at a quiescent native boundary.
iree_status_t loom_amdgpu_mask_write_wait_collect_boundary(
    loom_amdgpu_mask_write_wait_t* state, const loom_low_packet_view_t* packet,
    uint32_t instruction_offset);

// Propagates retained block summaries over the schedule CFG and emits waits.
// No IR or allocation traversal is performed during resolution.
iree_status_t loom_amdgpu_mask_write_wait_resolve(
    loom_amdgpu_mask_write_wait_t* state,
    loom_amdgpu_mask_write_wait_emit_fn_t emit, void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_MASK_WRITE_WAIT_H_
