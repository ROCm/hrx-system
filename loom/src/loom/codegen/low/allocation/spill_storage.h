// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared reservations for spill plans in one immutable allocation snapshot.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_SPILL_STORAGE_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_SPILL_STORAGE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Groups equally sized/aligned spill plans in the same storage space when their
// enclosing block lifetimes do not overlap. Whole-block intervals remain valid
// when materialization inserts traffic in source order and later scheduling
// reorders instructions within blocks. The intervals include sparse holes so
// expanded-execution stores cannot overwrite inactive-lane contents. Incoming
// argument stores extend the interval through their predecessor blocks.
// Callable entry/result cells keep exclusive reservations for direct ABI
// storage transport.
//
// Each result names the earliest plan sharing that reservation. Materialization
// can therefore create reservations in plan order without forward references.
// NULL denotes independent reservations, the allocation-free single-block or
// single-plan case. Results belong to |arena| and are consumed before rewriting
// this snapshot; reservations from previous repairs retain their placements.
iree_status_t loom_low_allocation_spill_storage_plan(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plans, iree_host_size_t plan_count,
    iree_arena_allocator_t* arena, const uint32_t** out_representatives);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_SPILL_STORAGE_H_
