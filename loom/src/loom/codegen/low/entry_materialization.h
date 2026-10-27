// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Placement of immutable values imported through a function's entry ABI.

#ifndef LOOM_CODEGEN_LOW_ENTRY_MATERIALIZATION_H_
#define LOOM_CODEGEN_LOW_ENTRY_MATERIALIZATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Plans one materialization point per entry argument or resource import.
// The ABI owner guarantees that the imported contents remain accessible and
// immutable throughout the invocation. Each point dominates every use and
// never executes more often than entry: cyclic uses select an acyclic dominator
// or entry itself. Nested uses project to their containing body operation.
// Type/attribute captures retain entry placement. INVALID and unused values
// produce NULL points.
//
// |values| and |out_before_ops| have |value_count| entries. The caller owns the
// output array. Planning consumes verified Low IR and builds CFG/dominance
// facts once for the batch. The resulting ordinary-body anchors survive entry
// argument removal and resource replacement; consume the entire plan before
// any other rewrite changes its uses, blocks, or anchors. The preamble itself
// remains intact until the ABI owner replaces its imports with native loads.
iree_status_t loom_low_entry_materialization_plan(const loom_module_t* module,
                                                  const loom_region_t* body,
                                                  const loom_value_id_t* values,
                                                  iree_host_size_t value_count,
                                                  iree_arena_allocator_t* arena,
                                                  loom_op_t** out_before_ops);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_CODEGEN_LOW_ENTRY_MATERIALIZATION_H_
