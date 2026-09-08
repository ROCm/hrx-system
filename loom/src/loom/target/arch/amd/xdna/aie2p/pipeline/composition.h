// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P source-callable composition for co-located pipeline stages.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_COMPOSITION_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_COMPOSITION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/pipeline_plan.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_aie2p_pipeline_composition_t {
  // Effective source callable for each physical resident instance.
  const loom_symbol_ref_t* instance_entries;

  // Synthesized source function indexed by scheduling group, or NULL when the
  // group has one logical stage and uses its entry directly.
  loom_op_t** group_functions;

  // Number of entries in |group_functions|.
  uint32_t group_count;
} loom_aie2p_pipeline_composition_t;

// Materializes one source callable for every multi-stage scheduling group.
//
// Synthesized callables use the group's physical boundary ports as their ABI,
// inline logical stage bodies in graph order, and use private local buffers for
// internal pointwise flows. Composition happens in source IR so target entry
// buffers remain ordinary values until the composite reaches source-to-Low.
iree_status_t loom_aie2p_pipeline_composition_materialize(
    loom_module_t* module, const loom_pipeline_plan_t* plan,
    iree_arena_allocator_t* arena,
    loom_aie2p_pipeline_composition_t* out_composition);

// Erases source callables synthesized by |composition|.
iree_status_t loom_aie2p_pipeline_composition_erase(
    loom_module_t* module, loom_aie2p_pipeline_composition_t* composition);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_COMPOSITION_H_
