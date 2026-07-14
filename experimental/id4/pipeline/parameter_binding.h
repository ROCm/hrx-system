// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PARAMETER_BINDING_H_
#define EXPERIMENTAL_ID4_PIPELINE_PARAMETER_BINDING_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opaque immutable complete parameter binding assembled from published domains.
typedef struct id4_pipeline_parameter_binding_t
    id4_pipeline_parameter_binding_t;

// Opaque published owner for one planned parameter domain.
typedef struct id4_pipeline_parameter_materialization_t
    id4_pipeline_parameter_materialization_t;

// Opaque complete resident parameter slab set.
typedef struct id4_pipeline_parameter_slab_set_t
    id4_pipeline_parameter_slab_set_t;

// Opaque plan defining every required parameter domain.
typedef struct id4_pipeline_plan_t id4_pipeline_plan_t;

// Options for assembling one complete parameter binding.
typedef struct id4_pipeline_parameter_binding_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Exact plan used to acquire every materialized domain.
  const id4_pipeline_plan_t* plan;
  // Number of published domains in |materializations|.
  iree_host_size_t materialization_count;
  // Published materializations containing exactly one entry per planned slab.
  id4_pipeline_parameter_materialization_t* const* materializations;
} id4_pipeline_parameter_binding_create_options_t;

// Assembles a complete immutable parameter binding from independently
// published domains. Every planned slab must be supplied exactly once and every
// materialization must have been acquired from |plan|. The returned binding
// retains all materializations, preventing their retirement while any execution
// bundle can still use their allocations. Callers retain their management
// references, release all bindings, and then retire each materialization after
// its last-use edge.
iree_status_t id4_pipeline_parameter_binding_create(
    const id4_pipeline_parameter_binding_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_binding_t** out_binding);

// Retains |binding| for the caller.
void id4_pipeline_parameter_binding_retain(
    id4_pipeline_parameter_binding_t* binding);

// Releases |binding| from the caller.
void id4_pipeline_parameter_binding_release(
    id4_pipeline_parameter_binding_t* binding);

// Returns the complete resident slab set retained by |binding|.
id4_pipeline_parameter_slab_set_t* id4_pipeline_parameter_binding_slabs(
    const id4_pipeline_parameter_binding_t* binding);

// Returns the joined publication edges for all domains in |binding|.
iree_hal_semaphore_list_t
id4_pipeline_parameter_binding_readiness_semaphore_list(
    const id4_pipeline_parameter_binding_t* binding);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_PARAMETER_BINDING_H_
