// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Schedule-node and dependency-graph construction.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_GRAPH_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_GRAPH_H_

#include "loom/codegen/low/schedule/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Appends an ordering edge, omitting self edges and consecutive duplicates.
// Allocation and representational capacity failures are returned as status.
iree_status_t loom_low_schedule_add_dependency(
    loom_low_schedule_build_state_t* state, uint32_t producer_node,
    uint32_t consumer_node, loom_low_schedule_dependency_kind_t kind,
    uint32_t operand_index);

iree_status_t loom_low_schedule_fill_nodes(
    loom_low_schedule_build_state_t* state);

iree_status_t loom_low_schedule_build_dependencies(
    loom_low_schedule_build_state_t* state,
    const loom_liveness_analysis_t* liveness);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_GRAPH_H_
