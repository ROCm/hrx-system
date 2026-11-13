// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CREDITS_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CREDITS_H_

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initial reusable credits in the selected physical protocol. Configuration
// owns any initial reservation; retained batches exchange one complete turn.
// Mixed-capacity aliases establish this value at every construction entry.
uint32_t loom_aie2p_native_initial_credits(
    const loom_aie2p_native_context_t* context,
    const loom_aie2p_native_channel_t* channel);

// Selects initialization/restoration at the existing serial boundary enclosing
// each private construction. Mixed-capacity aliases initialize at entry.
// Other exact balanced endpoints require no action. Exact residuals
// retire after their writer; other protocols restore only after a complete
// child join and acknowledge writes before the completion join. Selection uses
// retained endpoint counts, ownership, and execution participants, not source
// traversal. Returns the upper bound on additional control routes so routing
// storage includes restoration before any physical paths are allocated.
iree_status_t loom_aie2p_native_select_credits(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    iree_host_size_t* out_control_count);

// Assigns initialization/restoration to an existing participant, preferring its
// already selected control path. Status carries route admission failures.
iree_status_t loom_aie2p_native_route_credits(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, uint32_t* next_source,
    uint8_t* next_controller, bool* out_valid);

// Whether this child needs an initial join before its private state can be
// restored. The ordinary after-child join then orders restoration completion.
bool loom_aie2p_native_credits_require_quiescence(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_invocation_t* child);

// Whether this boundary must establish fresh credits before participants start.
bool loom_aie2p_native_credits_require_initialization(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_invocation_t* child);

typedef enum loom_aie2p_native_credit_phase_t {
  // Seeds mixed-capacity private pairs before any new-lifetime user begins.
  LOOM_AIE2P_NATIVE_CREDIT_INITIALIZE,
  // Retires exact unread records after their writers complete.
  LOOM_AIE2P_NATIVE_CREDIT_RETIRE,
  // Restores private pairs after every old-lifetime participant retires.
  LOOM_AIE2P_NATIVE_CREDIT_RESET,
  LOOM_AIE2P_NATIVE_CREDIT_PHASE_COUNT,
} loom_aie2p_native_credit_phase_t;

// Emits selected actions for this engine. Entry initialization acknowledges its
// writes before the entry join. Residual credits retire before the first exit
// join; quiescent restorations follow it and complete before the last.
iree_status_t loom_aie2p_native_emit_credits(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    iree_host_size_t engine, const loom_pipeline_invocation_t* child,
    loom_aie2p_native_credit_phase_t phase);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CREDITS_H_
