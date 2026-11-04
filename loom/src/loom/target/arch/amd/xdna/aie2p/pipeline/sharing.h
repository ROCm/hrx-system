// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SHARING_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SHARING_H_

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

#ifdef __cplusplus
extern "C" {
#endif

// A complete destination record is granted before its sender task is queued.
// Such payloads cannot block control traffic behind an unavailable receiver.
typedef struct loom_aie2p_native_delivery_t {
  // Uniform unicast source whose selected path can carry control packets.
  loom_aie2p_native_distribution_t* distribution;
  // Directed lanes and actual-use result owned by route admission.
  loom_aie2p_array_route_sharing_t sharing;
  // Absolute sender task register, used by the receiver's continuation.
  uint32_t address;
  // One complete record task with no repeat, used for every receiver grant.
  uint32_t task;
} loom_aie2p_native_delivery_t;

// Selects at most one eligible payload path per control destination from the
// retained physical distributions and exact configuration credit returns.
// No payload framing or task policy changes until a route actually borrows it.
iree_status_t loom_aie2p_native_prepare_sharing(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization);

// Commits framing and receiver-granted admission for actually borrowed paths.
// Configuration emission consumes the receiver channel's retained delivery.
void loom_aie2p_native_commit_sharing(loom_aie2p_native_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SHARING_H_
