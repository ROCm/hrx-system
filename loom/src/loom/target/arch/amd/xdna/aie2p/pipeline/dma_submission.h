// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_DMA_SUBMISSION_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_DMA_SUBMISSION_H_

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits the selected task's submission callable and, when amortized by its
// admitted group, the shared address-patch body. The adapter binds relocation
// bases, patch destinations, and queue words to that body's physical ABI.
// Enqueues the receiver before the sender. This consumes the
// finalized descriptor policy; it does not allocate descriptors or add waits.
iree_status_t loom_aie2p_native_emit_submission(
    loom_aie2p_native_context_t* context,
    loom_aie2p_native_transfer_task_t* task);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_DMA_SUBMISSION_H_
