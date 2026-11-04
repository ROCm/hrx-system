// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Program identity planning for physically placed AIE2P resident workers.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_RESIDENT_PLAN_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_RESIDENT_PLAN_H_

#include "iree/base/api.h"
#include "loom/target/arch/amd/xdna/aie2p/array/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Groups physically planned workers by their complete materialized program
// semantics. |worker_plans| and |resident_programs| must each have
// |plan->worker_count| records. Returns the populated resident-program count.
// Program ordinals and representatives follow first-worker order.
iree_host_size_t loom_aie2p_array_resident_plan_build(
    const loom_aie2p_array_plan_t* plan,
    loom_aie2p_array_worker_plan_t* worker_plans,
    loom_aie2p_array_resident_program_plan_t* resident_programs);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_RESIDENT_PLAN_H_
