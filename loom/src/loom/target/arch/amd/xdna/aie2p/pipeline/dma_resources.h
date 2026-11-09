// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_DMA_RESOURCES_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_DMA_RESOURCES_H_

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resource reuse within one physical execution owner. Common execution
// selection has proved that its distinct worker occurrences do not overlap.
// Async legality has proved that each block retires its groups before exit.
// Within one occurrence and block, the retained completion orders reuse.
// Source-operation identity alone cannot distinguish repeated invocations.
bool loom_aie2p_native_transfer_retired(
    const loom_aie2p_native_transfer_t* transfer, iree_host_size_t worker_index,
    const loom_op_t* issue);

// Forms one immutable descriptor image, including its link within the task.
// Allocation and reuse compare the same complete image; dynamic addresses and
// completion releases are installed separately by their respective owners.
void loom_aie2p_native_task_descriptor_words(
    const loom_aie2p_native_transfer_task_t* task,
    const loom_aie2p_native_dma_descriptor_t* descriptor,
    const loom_aie2p_native_transfer_t* transfer, uint32_t out_words[8]);

// Selects a retired compatible sequence from the physical owner's inventory.
// Local sequences retain their path and completion credit; shim sequences can
// cross paths within one bank. Reusing a sequence at different addresses
// promotes its retained template to per-issue address patching. Autonomous
// descriptors retain iteration state throughout the invocation and cannot be
// reused here. Returns NULL when allocation of a new sequence is necessary.
loom_aie2p_native_dma_descriptor_t* loom_aie2p_native_reusable_descriptor(
    const loom_aie2p_native_worker_t* worker,
    const loom_aie2p_native_transfer_task_t* task, bool shim);

// Selects a retired arrival credit on the given tile, independently of
// descriptor geometry. Returns NULL when no existing credit can be reused.
loom_aie2p_native_dma_completion_t* loom_aie2p_native_reusable_completion(
    const loom_aie2p_native_worker_t* worker,
    const loom_aie2p_native_tile_t* tile,
    const loom_aie2p_native_transfer_t* next);

// Admits the required task geometry before any optional address variants.
// Physical inventory tails belong to resource_owner, which may cover several
// serialized worker occurrences. Runtime addressing never changes this order.
iree_status_t loom_aie2p_native_select_task_descriptors(
    loom_aie2p_native_context_t* context, loom_aie2p_native_worker_t* worker,
    loom_aie2p_native_worker_t* resource_owner,
    loom_aie2p_native_transfer_task_t* task, bool autonomous,
    loom_aie2p_native_dma_descriptor_t*** descriptor_tail,
    loom_aie2p_native_dma_completion_t*** completion_tail, bool* out_valid);

// Refines admitted core tasks with fixed-address variants using only spare
// descriptors. Variants inherit routes and completion credits; dynamic tasks
// and exhausted banks retain their original patchable descriptors.
iree_status_t loom_aie2p_native_refine_transfer_addresses(
    loom_aie2p_native_context_t* context, iree_host_size_t worker_count);

// Groups final descriptor policies into submission bodies that fit the native
// call ABI. Fixed local variants pass a queue word; dynamic local patching has
// a separate body with a fixed patch destination. No body is emitted here.
iree_status_t loom_aie2p_native_group_submissions(
    loom_aie2p_native_context_t* context, iree_host_size_t worker_count);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_DMA_RESOURCES_H_
