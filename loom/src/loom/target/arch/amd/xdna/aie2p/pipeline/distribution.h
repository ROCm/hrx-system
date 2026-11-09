// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_DISTRIBUTION_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_DISTRIBUTION_H_

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

#ifdef __cplusplus
extern "C" {
#endif

// Finds immutable descriptors for an ordered phase on an already admitted
// path. The caller establishes retirement between phases. Source and
// destination channel identities include storage and semaphore ownership; equal
// physical tiles alone allow route sharing, not descriptor reuse. Returns NULL
// when a new geometry is needed. Packet paths also compare their retained copy
// order.
loom_aie2p_native_distribution_t* loom_aie2p_native_reusable_distribution(
    const loom_aie2p_native_worker_t* owner,
    const loom_aie2p_native_channel_t* source,
    const loom_pipeline_transport_t* transport,
    const loom_pipeline_transport_batch_t* batch, uint32_t repetitions);

// Selects on-chip copy programs before transfer and configuration emission.
iree_status_t loom_aie2p_native_select_worker_distributions(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    iree_host_size_t worker_index, uint32_t* next_source, bool* out_valid);

// Binds controller publications to their retained ownership intervals.
iree_status_t loom_aie2p_native_bind_distribution_phases(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, bool* out_valid);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_DISTRIBUTION_H_
