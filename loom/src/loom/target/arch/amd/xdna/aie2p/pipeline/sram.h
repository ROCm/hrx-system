// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SRAM_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SRAM_H_

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

#ifdef __cplusplus
extern "C" {
#endif

// Publishes one complete retained batch after its producer owns the full SRAM
// lifetime. This writes the selected lock values without a response packet.
iree_status_t loom_aie2p_native_sram_publish_helper(
    loom_aie2p_native_context_t* context,
    const loom_aie2p_native_channel_t* channel, uint8_t packet,
    loom_symbol_ref_t* out_symbol);

// Atomically acquires or returns credits through the memory tile's exact
// request word, retrying until its response reports success. The selected
// core reply stream is exclusive across each request and response. Payload
// bytes stay in SRAM and are accessible to the core only through DMA.
iree_status_t loom_aie2p_native_sram_lock_helper(
    loom_aie2p_native_context_t* context,
    const loom_aie2p_native_channel_t* channel, uint16_t lock, bool acquire,
    uint32_t credits, loom_symbol_ref_t* out_symbol);

// Finds the first SRAM request inside a pending DMA response interval, or NULL
// when the retained schedule keeps their shared reply stream disjoint. Both
// sequences have producer-established block/source order; the merge advances
// each frontier once and neither traverses IR nor changes authored waits.
const loom_op_t* loom_aie2p_native_sram_reply_conflict(
    const loom_pipeline_invocations_t* invocations,
    const loom_pipeline_worker_t* source,
    const loom_aie2p_native_worker_t* worker, iree_host_size_t worker_index);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SRAM_H_
