// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_STAGE_H_
#define EXPERIMENTAL_ID4_PIPELINE_STAGE_H_

#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/plan.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/parameter_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opaque prepared execution bundle.
typedef struct id4_pipeline_bundle_t id4_pipeline_bundle_t;

// Pipeline stage base object embedded by concrete stages.
typedef struct id4_pipeline_stage_t id4_pipeline_stage_t;

// Destroys stage-specific bundle payload storage.
typedef void(IREE_API_PTR* id4_pipeline_bundle_payload_destroy_fn_t)(
    id4_pipeline_bundle_t* bundle, void* payload);

// Shared services available to all pipeline stages.
typedef struct id4_pipeline_stage_services_t {
  // Device group inspected during planning and used during execution.
  iree_hal_device_group_t* device_group;
  // Optional executable cache shared by stages that materialize HAL
  // executables.
  iree_hal_executable_cache_t* executable_cache;
  // Host allocator used for stage-owned metadata.
  iree_allocator_t host_allocator;
} id4_pipeline_stage_services_t;

// Options for loading immutable stage assets and runtime state.
typedef struct id4_pipeline_stage_load_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Diagnostics sink for load events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_pipeline_stage_load_options_t;

// Options for creating an inspectable execution plan.
typedef struct id4_pipeline_stage_plan_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Device index used by single-device stage plans.
  iree_host_size_t device_index;
  // Queue affinity used by single-device stage plans.
  iree_hal_queue_affinity_t queue_affinity;
  // Diagnostics sink for plan events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_pipeline_stage_plan_options_t;

// Options for preparing a reusable execution bundle from a plan.
typedef struct id4_pipeline_stage_prepare_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Parameter provider used to populate planned parameter slabs.
  iree_io_parameter_provider_t* parameter_provider;
  // Semaphores that parameter loading and command-buffer preparation wait on.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphores signaled when parameter loading and preparation complete.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Diagnostics sink for prepare events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_pipeline_stage_prepare_options_t;

// Options for issuing a prepared bundle.
typedef struct id4_pipeline_stage_issue_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Semaphores that execution waits on.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphores signaled when execution completes.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Diagnostics sink for issue events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_pipeline_stage_issue_options_t;

// Options for creating a prepared execution bundle.
typedef struct id4_pipeline_bundle_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Plan retained by the bundle.
  const id4_pipeline_plan_t* plan;
  // Loaded parameter slabs retained by the bundle, if the plan uses them.
  id4_pipeline_parameter_slab_set_t* parameter_slabs;
  // Semaphores that must be reached before bundle contents are ready.
  iree_hal_semaphore_list_t readiness_semaphore_list;
  // Stage-specific payload byte length stored inline with the bundle.
  iree_host_size_t payload_size;
  // Stage-specific payload alignment. Zero selects pointer alignment.
  iree_host_size_t payload_alignment;
  // Optional destructor for initialized stage-specific payload storage.
  id4_pipeline_bundle_payload_destroy_fn_t payload_destroy;
} id4_pipeline_bundle_create_options_t;

// Vtable implemented by concrete pipeline stages.
typedef struct id4_pipeline_stage_vtable_t {
  // Destroys the concrete stage object when the base ref count reaches zero.
  void(IREE_API_PTR* destroy)(id4_pipeline_stage_t* stage);
  // Loads immutable stage assets and initializes stage runtime state.
  iree_status_t(IREE_API_PTR* load)(
      id4_pipeline_stage_t* stage,
      const id4_pipeline_stage_load_options_t* options);
  // Produces an inspectable plan for the current stage configuration.
  iree_status_t(IREE_API_PTR* plan)(
      id4_pipeline_stage_t* stage,
      const id4_pipeline_stage_plan_options_t* options,
      id4_pipeline_plan_t** out_plan);
  // Prepares reusable HAL state from a plan.
  iree_status_t(IREE_API_PTR* prepare)(
      id4_pipeline_stage_t* stage, const id4_pipeline_plan_t* plan,
      const id4_pipeline_stage_prepare_options_t* options,
      id4_pipeline_bundle_t** out_bundle);
  // Issues one execution of a prepared bundle.
  iree_status_t(IREE_API_PTR* issue)(
      id4_pipeline_stage_t* stage, id4_pipeline_bundle_t* bundle,
      const id4_pipeline_stage_issue_options_t* options);
} id4_pipeline_stage_vtable_t;

// Base class embedded as the first field in concrete pipeline stages.
struct id4_pipeline_stage_t {
  // Reference count for shared stage ownership.
  iree_atomic_ref_count_t ref_count;
  // Implementation vtable for this concrete stage.
  const id4_pipeline_stage_vtable_t* vtable;
  // Retained service handles shared with the concrete stage.
  id4_pipeline_stage_services_t services;
};

// Initializes a concrete stage base with retained service handles.
iree_status_t id4_pipeline_stage_initialize(
    const id4_pipeline_stage_vtable_t* vtable,
    const id4_pipeline_stage_services_t* services,
    id4_pipeline_stage_t* out_stage);

// Deinitializes service handles held by |stage|.
void id4_pipeline_stage_deinitialize(id4_pipeline_stage_t* stage);

// Retains |stage| for the caller.
void id4_pipeline_stage_retain(id4_pipeline_stage_t* stage);

// Releases |stage| from the caller.
void id4_pipeline_stage_release(id4_pipeline_stage_t* stage);

// Returns immutable services captured by |stage|.
const id4_pipeline_stage_services_t* id4_pipeline_stage_services(
    const id4_pipeline_stage_t* stage);

// Loads immutable stage assets and runtime state.
iree_status_t id4_pipeline_stage_load(
    id4_pipeline_stage_t* stage,
    const id4_pipeline_stage_load_options_t* options);

// Produces an inspectable plan for |stage|.
iree_status_t id4_pipeline_stage_plan(
    id4_pipeline_stage_t* stage,
    const id4_pipeline_stage_plan_options_t* options,
    id4_pipeline_plan_t** out_plan);

// Prepares a reusable execution bundle from |plan|.
iree_status_t id4_pipeline_stage_prepare(
    id4_pipeline_stage_t* stage, const id4_pipeline_plan_t* plan,
    const id4_pipeline_stage_prepare_options_t* options,
    id4_pipeline_bundle_t** out_bundle);

// Issues one execution of |bundle|.
iree_status_t id4_pipeline_stage_issue(
    id4_pipeline_stage_t* stage, id4_pipeline_bundle_t* bundle,
    const id4_pipeline_stage_issue_options_t* options);

// Creates a bundle retaining its plan, parameter slabs, readiness semaphores,
// and optional stage-specific payload storage.
iree_status_t id4_pipeline_bundle_create(
    const id4_pipeline_bundle_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_bundle_t** out_bundle);

// Retains |bundle| for the caller.
void id4_pipeline_bundle_retain(id4_pipeline_bundle_t* bundle);

// Releases |bundle| from the caller.
void id4_pipeline_bundle_release(id4_pipeline_bundle_t* bundle);

// Returns the plan retained by |bundle|.
const id4_pipeline_plan_t* id4_pipeline_bundle_plan(
    const id4_pipeline_bundle_t* bundle);

// Returns the parameter slab set retained by |bundle|, if any.
id4_pipeline_parameter_slab_set_t* id4_pipeline_bundle_parameter_slabs(
    const id4_pipeline_bundle_t* bundle);

// Returns semaphores that must be reached before |bundle| contents are ready.
iree_hal_semaphore_list_t id4_pipeline_bundle_readiness_semaphore_list(
    const id4_pipeline_bundle_t* bundle);

// Returns mutable stage-specific payload storage, if present.
void* id4_pipeline_bundle_payload(id4_pipeline_bundle_t* bundle);

// Returns immutable stage-specific payload storage, if present.
const void* id4_pipeline_bundle_const_payload(
    const id4_pipeline_bundle_t* bundle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_STAGE_H_
