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
// begin at 16-byte program addresses; fallthrough-only labels need no padding.
// Block analysis selects fallthrough, J, JZ, or JNZ from the following block,
// and emission consumes that retained choice with explicit architectural delay
// bundles. Contribution-relative targets remain fixups until final code
// placement. Structural low.return is materialized as RET and hoisted over
// useful work in the same block when physical timing and resources permit it.
// Return placement considers the architectural tail window, including implicit
// NOP gaps, and materializes only the selected return position.
// Descriptor runs use the minimum contiguous partition of exact physical
// bundle formats because AIE2P's format domain is not downward closed.
// Allocation-planned structural moves split a logical schedule cycle into
// ordered physical bundles. Shared physical issue admission forwards retained
// source dependencies from each actual producer issue instead of shifting all
// logical deadlines after an independent register stall. Structural storage
// setup forwards payload availability even when it coalesces or issues early;
// native moves retain concrete register-event admission. Native expansion can
// cover later logical gaps. Collective bundle resource occupancy remains part
// of physical admission. Gaps occupy code bytes without allocating per-cycle
// bundle or slot records. The frame retains its grouped dependency index.
// Every control-flow edge reaches a quiescent event/resource boundary before
// successor entry, including fallthrough and backedges. Structural control's
// source deadline constrains retirement; native condition and LR reads retain
// concrete register-event issue admission. Native branch-delay cycles
// contribute to this boundary. Prebound live-ins and resource imports anchor
// physical assignments without occupying an instruction slot. Empty
// non-terminator logical cycles not already covered by native expansion are
// materialized as NOP bundles. The returned plan owns its function name,
// resource bindings, storage requirements, and physical tables in |arena|. It
// contains no compiler frame or IR references.
iree_status_t loom_aie2p_bundle_plan_build(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_aie2p_leaf_program_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_BUNDLE_PLAN_H_
