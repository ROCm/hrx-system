// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Logical AIE2P array topology validation and transport classification.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_TOPOLOGY_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_TOPOLOGY_H_

#include "loom/target/arch/amd/xdna/aie2p/array/plan.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the unwrapped owner endpoint for |endpoint|.
const loom_aie2p_array_endpoint_t* loom_aie2p_array_topology_base_endpoint(
    const loom_aie2p_array_plan_t* plan,
    const loom_aie2p_array_endpoint_t* endpoint);

// Validates the complete logical topology and classifies channel transports.
iree_status_t loom_aie2p_array_topology_validate(
    const loom_value_fact_table_t* facts, iree_arena_allocator_t* arena,
    loom_aie2p_array_plan_t* plan,
    loom_aie2p_array_channel_t* mutable_channels);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_TOPOLOGY_H_
