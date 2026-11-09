// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CHANNEL_STATE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CHANNEL_STATE_H_

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

#ifdef __cplusplus
extern "C" {
#endif

// Accumulates core admission bounds and ownership for one read/write cursor.
// Imported channels preserve their position beyond the current worker call.
void loom_aie2p_native_cursor_bind(loom_aie2p_native_cursor_t* cursor,
                                   uint64_t capacity, uint32_t worker,
                                   uint32_t maximum_admissions, bool imported);

// Whether an advancing cursor requires construction-owned frame storage instead
// of worker-local SSA state. The representative worker has already been
// admitted.
bool loom_aie2p_native_cursor_persistent(
    const loom_pipeline_invocations_t* invocations, iree_host_size_t channel,
    const loom_aie2p_native_cursor_t* cursor);

// A verified read/write capability selects protocol ownership independently of
// whether a transfer reads or writes the record's payload bytes.
const loom_aie2p_native_cursor_t* loom_aie2p_native_record_cursor(
    const loom_aie2p_native_channel_t* channel, loom_type_t capability);

// Initializes only frame-backed cells constructed by owner on this engine.
// Ancestor cells and worker-local SSA cursors remain outside this boundary.
iree_status_t loom_aie2p_native_initialize_cursors(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    loom_value_id_t frame, iree_host_size_t engine,
    const loom_pipeline_invocation_t* owner);

// Binds unique native endpoint state to its channel's constructing owner.
// Existing distribution controller selection and retirement precede this
// step. It consumes retained geometry and ownership, without source traversal.
iree_status_t loom_aie2p_native_select_initializers(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, bool* out_valid);

// Initializes this engine's selected descriptors at construction entry. Task
// publication remains at the transferring child's entry, which may repeat while
// the constructing owner and its hardware iterator stay live.
iree_status_t loom_aie2p_native_initialize_endpoints(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    iree_host_size_t engine, const loom_pipeline_invocation_t* invocation);

// Forms one scalar cursor cell in an invocation-private frame. Channel ordinal
// and direction give a stable identity across child calls. Ordinary private
// storage promotion carries these cells through the resulting control graph.
iree_status_t loom_aie2p_native_cursor_view(loom_builder_t* builder,
                                            loom_value_id_t frame,
                                            iree_host_size_t channel,
                                            bool writer,
                                            loom_value_id_t* out_view);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CHANNEL_STATE_H_
