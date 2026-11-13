// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_VERIFY_VERIFY_CONSUMPTION_H_
#define LOOM_VERIFY_VERIFY_CONSUMPTION_H_

#include "loom/ops/op_defs.h"
#include "loom/util/cfg_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_verify_state_t loom_verify_state_t;
typedef struct loom_verify_consumption_t loom_verify_consumption_t;

// Retains the verifier walk's region hierarchy and shared CFG. The returned
// index names this region for the remainder of the verification invocation.
iree_status_t loom_verify_consumption_record_region(
    loom_verify_state_t* state, const loom_region_t* region,
    const loom_op_t* owner, loom_region_execution_t execution,
    const loom_cfg_graph_t* graph, uint32_t parent_index, uint32_t* out_index);

// Publishes required nonwriting storage identities after this operation's
// structural, semantic, and operand-availability checks have succeeded.
iree_status_t loom_verify_consumption_record_aliases(loom_verify_state_t* state,
                                                     const loom_op_t* op);

// Records an operand whose ownership transfers at this operation, for the
// grouped ownership check.
iree_status_t loom_verify_consumption_record(loom_verify_state_t* state,
                                             const loom_op_t* op,
                                             uint16_t operand_index);

// Checks each consumed ownership family once using retained regions and SSA
// use lists. Structural verification must have succeeded before this call.
iree_status_t loom_verify_consumption_check(loom_verify_state_t* state);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_VERIFY_VERIFY_CONSUMPTION_H_
