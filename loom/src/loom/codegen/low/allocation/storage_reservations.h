// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Physical storage reservations assembled from semantic and retained reads.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_STORAGE_RESERVATIONS_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_STORAGE_RESERVATIONS_H_

#include "loom/codegen/low/allocation/unit_liveness.h"

#ifdef __cplusplus
extern "C" {
#endif

// Completes mandatory storage-component lifetimes and merges their semantic
// segments with decomposed edge reads and the retained CFG predecessor closure.
// Reservations remain sparse across blocks where no component is needed.
// Published segments use |result_arena|; per-value ranges use |decision_arena|.
// Temporary merge storage is released before returning.
iree_status_t loom_low_allocation_storage_reservations_build(
    loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement,
    iree_arena_allocator_t* result_arena,
    iree_arena_allocator_t* decision_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_STORAGE_RESERVATIONS_H_
