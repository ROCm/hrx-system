// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P resident-pipeline graph lowering to array Low.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_LOWER_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_LOWER_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Lowers one verified kernel-scoped pipeline definition to an AIE2P array
// program. Exact SSA facts select group cardinalities, buffer capacities, and
// dynamic tile dimensions. The source function is replaced only after the
// complete array Low function has been constructed successfully.
iree_status_t loom_aie2p_pipeline_lower_to_array_low(
    loom_module_t* module, loom_func_like_t pipeline,
    const loom_value_fact_table_t* facts, loom_op_t** out_low_function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_LOWER_H_
