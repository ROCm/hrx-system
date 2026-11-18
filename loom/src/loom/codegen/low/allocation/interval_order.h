// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocatable liveness interval ordering for low allocation.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_INTERVAL_ORDER_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_INTERVAL_ORDER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/placement.h"

#ifdef __cplusplus
extern "C" {
#endif

// One interval and its retained acquisition-order keys.
typedef struct loom_low_allocation_interval_order_entry_t {
  // Borrowed semantic interval whose identity and uses remain unchanged.
  const loom_liveness_interval_t* interval;
  // Earliest required acquisition point retained by unit liveness.
  uint32_t acquisition_start_point;
  // Positive source-before-user priority for mandatory storage nonorigins.
  // Origins and ordinary values use zero to retain end/identity preference.
  uint32_t topology_rank;
} loom_low_allocation_interval_order_entry_t;

// Deterministic storage-allocation order over allocatable intervals.
typedef struct loom_low_allocation_interval_order_t {
  // Intervals ordered by acquisition start, source-before-user priority,
  // semantic end, and value identity.
  const loom_low_allocation_interval_order_entry_t* intervals;
  // Number of entries in |intervals|.
  iree_host_size_t interval_count;
  // At least one multi-unit interval uses contiguous allocation units instead
  // of explicit physical-register IDs, so scalar packing may change placement.
  bool has_packable_aggregates;
} loom_low_allocation_interval_order_t;

// Builds acquisition chronology with mandatory sources before their results at
// equal starts. |unit_liveness| must retain mandatory starts from |placement|.
// Ordinary values and storage origins retain shortest-end then value-identity
// preference, including when optional storage relations are present.
iree_status_t loom_low_allocation_interval_order_build(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_placement_table_t* placement, iree_arena_allocator_t* arena,
    loom_low_allocation_interval_order_t* out_order);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_INTERVAL_ORDER_H_
