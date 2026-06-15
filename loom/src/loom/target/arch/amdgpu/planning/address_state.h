// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target-low address-state materialization.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_ADDRESS_STATE_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_ADDRESS_STATE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/frame.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Inserts target packets required by target-state address maps in |frame|.
//
// This is a target-owned finalization step over an already scheduled and
// allocated spill-free low function. The function is intentionally framed as
// address-state materialization instead of a gfx125x-only fix: descriptor
// operands name target-state address maps, and the architecture supplies the
// packet sequence that makes those operands encodable. For gfx125x this emits
// s_set_vgpr_msb packets before packets whose VGPR operands use a high 256-VGPR
// window and resets MODE to the architectural zero window before leaving each
// low block.
iree_status_t loom_amdgpu_materialize_address_state(
    loom_module_t* module, loom_op_t* function_op,
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_low_emission_frame_materialize_address_state_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_ADDRESS_STATE_H_
