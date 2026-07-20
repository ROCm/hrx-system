// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_OPS_SCF_RESIDENCY_H_
#define LOOM_OPS_SCF_RESIDENCY_H_

#include "iree/base/api.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Builds a candidate identity boundary with an exact witness of the source
// producer whose movement was proven legal. The producer must be a regionless,
// successorless, untied expression so target-low repair can reconstruct its
// finite materialization recipe.
iree_status_t loom_scf_residency_candidate_build_for_proven_source(
    loom_builder_t* builder, int64_t candidate_id, int64_t recompute_cost,
    const loom_op_t* source_op, uint16_t source_result_index,
    bool preserves_baseline, loom_location_id_t location,
    loom_op_t** out_candidate_op);

// Validates that a candidate still names the exact source producer captured by
// its legality proof.
iree_status_t loom_scf_residency_candidate_validate_proven_source(
    const loom_module_t* module, const loom_op_t* candidate_op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_SCF_RESIDENCY_H_
