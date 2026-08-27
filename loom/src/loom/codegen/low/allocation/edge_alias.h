// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Path-sensitive storage-alias checks for control-flow edge payloads.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_EDGE_ALIAS_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_EDGE_ALIAS_H_

#include "iree/base/api.h"
#include "loom/analysis/consumption.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/placement.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef iree_status_t (*loom_low_allocation_edge_alias_consumption_query_fn_t)(
    void* user_data, const loom_region_t* region,
    loom_consumption_region_query_t** out_query);

typedef struct loom_low_allocation_edge_alias_context_t {
  // Function-local placement relations.
  const loom_low_placement_table_t* placement;
  // Callback that lazily returns a consumption query for one relation region.
  loom_low_allocation_edge_alias_consumption_query_fn_t consumption_query;
  // Opaque caller state passed to consumption_query.
  void* user_data;
} loom_low_allocation_edge_alias_context_t;

// Returns whether |interval| may share the edge destination units covered by
// |relation| with |counterpart|. The query proves that neither the destination
// nor a distinct source remains observable after the edge handoff; linear
// interval overlap alone is not sufficient for that decision.
iree_status_t loom_low_allocation_edge_alias_allows_counterpart_overlap(
    const loom_low_allocation_edge_alias_context_t* context,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* counterpart,
    uint32_t destination_unit_offset, uint32_t destination_unit_count,
    bool* out_allows_overlap);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_EDGE_ALIAS_H_
