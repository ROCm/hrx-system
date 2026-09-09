// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Precomputed separation of scalar and aggregate register lifetimes.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_SCALAR_PACKING_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_SCALAR_PACKING_H_

#include "iree/base/api.h"
#include "iree/base/bitmap.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/interval_order.h"
#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_allocation_scalar_packing_t {
  // Scalar intervals overlapping aggregate storage, indexed by interval index.
  iree_bitmap_t overlapping_intervals;
  // Exclusive scalar packing frontier for each descriptor register class.
  const uint32_t* frontiers_by_reg_class;
} loom_low_allocation_scalar_packing_t;

// Plans scalar placement preferences before coloring. Scalars overlapping
// aggregates pack down from the class's pressure frontier, leaving aligned
// low spans available to aggregates. Physical-ID classes use their generated
// candidate order instead. Two sweeps over |order| index overlap in linear
// work, without a per-scalar scan through all intervals.
//
// |liveness| and |order| contain classes from |descriptor_set|. The returned
// bitmap is empty when no applicable aggregate exists.
iree_status_t loom_low_allocation_scalar_packing_build(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_interval_order_t* order,
    iree_arena_allocator_t* arena,
    loom_low_allocation_scalar_packing_t* out_packing);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_SCALAR_PACKING_H_
