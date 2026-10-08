// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prospective concat placement using retained allocation and storage facts.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_CONCAT_RESERVATION_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_CONCAT_RESERVATION_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/allocation/search.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects a future concat result reservation or a current source assignment
// when that avoids a lower residency tier. The source and result intervals
// share a value class; |result_range| contains the assembly's retained
// placement relations and |ignored_value_ids| contains its concat sources. The
// result may instead be the assembly's exclusive whole-value edge destination,
// when the caller proves that source storage is consumed before destination
// acquisition.
//
// A value_id of INVALID means no placement was selected. This query does not
// publish the assignment: the allocator retains ownership of active storage,
// liveness and leases. Result reservations may share leases with the ignored
// sources; a source assignment requires no ignored leases.
// Optional reservations preserve outstanding source leases. Ordinary placement
// owns the decision to release those leases for capacity or residency; result
// leases marked releasable for pressure remain available to reservations.
iree_status_t loom_low_allocation_concat_reservation_find(
    loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* source_interval,
    const loom_low_placement_relation_t* relation,
    const loom_liveness_interval_t* result_interval,
    const loom_low_placement_relation_range_t* result_range,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    loom_low_allocation_assignment_t* out_assignment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_CONCAT_RESERVATION_H_
