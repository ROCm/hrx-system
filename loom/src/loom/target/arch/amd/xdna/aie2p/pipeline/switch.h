// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SWITCH_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SWITCH_H_

#include "loom/ir/ir.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/execution_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_aie2p_native_context_t;

// Admits physical route edges against packet/circuit ownership and switch
// resources, retaining the complete initialization banks in the pass arena.
// Destination sets connected by a shared output require one arbiter. Disjoint
// sets share only when fed exclusively by the same input, so arbitration adds
// no blocking dependency between independent streams. This does not prove
// progress for dependencies already present in the selected routes.
// Allocation errors return a status; route conflicts emit the pipeline's
// target-support diagnostic and leave the caller's false |out_valid| intact.
iree_status_t loom_aie2p_native_switches_build(
    struct loom_aie2p_native_context_t* context, const loom_op_t* entry,
    bool* out_valid);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_SWITCH_H_
