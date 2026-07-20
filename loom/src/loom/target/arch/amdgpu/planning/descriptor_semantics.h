// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared AMDGPU descriptor semantic predicates.
//
// AMDGPU target-low descriptors carry compact target facts generated from the
// Python descriptor tables. This layer centralizes backend predicates that need
// to stay consistent across wait-state and wait-counter planning.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_DESCRIPTOR_SEMANTICS_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_DESCRIPTOR_SEMANTICS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |descriptor| issues on the vector ALU.
bool loom_amdgpu_descriptor_uses_vector_alu(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns true when |descriptor| issues on the scalar ALU.
bool loom_amdgpu_descriptor_uses_scalar_alu(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns true when |descriptor| issues on an AMDGPU vector-memory pipeline.
bool loom_amdgpu_descriptor_uses_vector_memory(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns true when |descriptor| is a transcendental VALU packet.
bool loom_amdgpu_descriptor_is_transcendental(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns true when |descriptor| is a DPP lane-crossing packet.
bool loom_amdgpu_descriptor_is_dpp(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns true when |descriptor| reads one VGPR lane into an SGPR.
bool loom_amdgpu_descriptor_is_readfirstlane(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns true when |descriptor| uses an SDWA packet encoding.
bool loom_amdgpu_descriptor_is_sdwa(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns true when |descriptor| implicitly drains gfx125x XCNT before it
// executes.
bool loom_amdgpu_descriptor_implicitly_drains_xcnt(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Builds AMDGPU schedule state-read rows for structural low materializations.
iree_status_t loom_amdgpu_descriptor_build_structural_state_reads(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena,
    loom_low_schedule_structural_state_read_list_t* out_state_reads);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_DESCRIPTOR_SEMANTICS_H_
