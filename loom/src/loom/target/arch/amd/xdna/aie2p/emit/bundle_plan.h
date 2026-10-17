// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Final AIE2P VLIW bundle planning over a scheduled and allocated Low frame.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_BUNDLE_PLAN_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_BUNDLE_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/frame.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_program.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel used when a physical slot was synthesized by the target planner.
#define LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE UINT32_MAX

// Architectural core program-memory capacity in bytes.
#define LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE 16384u

// Plans physical bundles for one successful, spill-free AIE2P Low frame.
//
// Blocks retain source order. Entry and selected native branch destinations
// begin at 16-byte program addresses. Forward-only destinations share a
// preceding NOP tail when one contains an aligned entry. The tail may also
// retire predecessor work or control; entering its NOPs performs no predecessor
// work. Backward destinations have direct entries so repeated transfers pay no
// added NOP prefix. Fallthrough-only labels need no padding.
//
// Block analysis selects fallthrough, J, JZ, or JNZ from the following block,
// and emission consumes that retained choice. Contribution-relative targets
// remain fixups until final code placement. Structural branches and returns
// issue within the same block's architectural tail window when concrete
// register timing and bundle resources permit, so useful work fills their
// delay slots. Implicit NOP gaps materialize only the selected control
// position. A two-branch terminator keeps the second branch beyond the first
// window. Native instructions publish their concrete accesses in accepted
// semantic order. Shared physical admission may place an instruction earlier
// within generated bounded resource history when signed WAR/WAW timing permits
// it. An encoded-packet window admits only exact legal slot unions, then
// appends closed positions to the detached plan in chronological issue order.
// Every native expansion piece forwards its actual availability through the
// frame's retained dependency groups; semantic consumers never precede
// producers. Coalesced storage setup forwards incoming payload availability
// without an unrelated native high-water floor. Source-order boundaries close
// the window on both sides, including zero-width declarations without fake
// instructions. Gaps occupy code bytes without allocating per-cycle bundle or
// slot records. Every control-flow edge reaches a quiescent event/resource
// boundary before successor entry, including fallthrough and backedges.
// Structural control's source deadline constrains retirement; native condition
// and LR reads retain concrete register-event issue admission. Native
// branch-delay cycles contribute to this boundary. Prebound live-ins and
// resource imports anchor physical assignments without occupying an instruction
// slot. The returned plan owns its function name, resource bindings, storage
// requirements, and physical tables in |arena|. It contains no compiler frame
// or IR references.
iree_status_t loom_aie2p_bundle_plan_build(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_aie2p_leaf_program_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_BUNDLE_PLAN_H_
