// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SEMAPHORES_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SEMAPHORES_H_

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

#ifdef __cplusplus
extern "C" {
#endif

// Assigns physical FIFO semaphore pairs after batch-credit selection. The
// common lifetime layout overlays compatible serial constructions; parent and
// concurrent protocols retain distinct pairs. The first logical user of each
// pair owns startup initialization. Completion/join locks are allocated later.
iree_status_t loom_aie2p_native_assign_channel_locks(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, bool* out_valid);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SEMAPHORES_H_
