// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_H_
#define EXPERIMENTAL_ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_H_

#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/parameter_materialization.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opaque unique owner for one complete set of materialized parameter domains.
typedef struct id4_pipeline_parameter_materialization_group_t
    id4_pipeline_parameter_materialization_group_t;

// Opaque complete resident parameter slab set.
typedef struct id4_pipeline_parameter_slab_set_t
    id4_pipeline_parameter_slab_set_t;

// Opaque plan defining every required parameter domain.
typedef struct id4_pipeline_plan_t id4_pipeline_plan_t;

// Creates an empty materialization group for every parameter domain in |plan|.
// The group retains the exact plan until it is retired and released.
iree_status_t id4_pipeline_parameter_materialization_group_create(
    const id4_pipeline_plan_t* plan, iree_allocator_t host_allocator,
    id4_pipeline_parameter_materialization_group_t** out_group);

// Transfers one published materialization into |group|. The materialization
// must belong to the group's exact plan and its slab must not already have been
// adopted. On success |*inout_materialization| is set to NULL. On failure the
// caller retains ownership.
iree_status_t id4_pipeline_parameter_materialization_group_adopt(
    id4_pipeline_parameter_materialization_group_t* group,
    id4_pipeline_parameter_materialization_t** inout_materialization);

// Finalizes a complete group into one immutable parameter binding. Every plan
// slab must have been adopted exactly once. Failure leaves the group open for
// correction or retirement.
iree_status_t id4_pipeline_parameter_materialization_group_finalize(
    id4_pipeline_parameter_materialization_group_t* group);

// Returns the complete resident slab set after successful finalization.
id4_pipeline_parameter_slab_set_t*
id4_pipeline_parameter_materialization_group_parameter_slabs(
    const id4_pipeline_parameter_materialization_group_t* group);

// Returns the joined publication edges after successful finalization.
iree_hal_semaphore_list_t
id4_pipeline_parameter_materialization_group_readiness_semaphore_list(
    const id4_pipeline_parameter_materialization_group_t* group);

// Waits indefinitely for every finalized domain publication edge. Queue and
// semaphore failures are returned to the caller.
iree_status_t id4_pipeline_parameter_materialization_group_wait_ready(
    const id4_pipeline_parameter_materialization_group_t* group);

// Releases the complete binding and synchronously retires each adopted domain
// after |last_use_wait_list|. Each domain deallocation is routed through its
// own buffer placement and independently signaled completion edge; no ordering
// between queue submissions is assumed. When the list is empty, each domain's
// publication edge is used, which is suitable before the group has been used.
// A retry after a rejected retirement submission must provide the same or a
// stronger last-use edge.
iree_status_t id4_pipeline_parameter_materialization_group_retire(
    id4_pipeline_parameter_materialization_group_t* group,
    iree_hal_semaphore_list_t last_use_wait_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink);

// Releases a completely retired materialization group.
void id4_pipeline_parameter_materialization_group_release(
    id4_pipeline_parameter_materialization_group_t* group);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_PARAMETER_MATERIALIZATION_GROUP_H_
