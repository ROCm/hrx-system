// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PARAMETER_SLAB_H_
#define EXPERIMENTAL_ID4_PIPELINE_PARAMETER_SLAB_H_

#include <stdint.h>

#include "experimental/id4/pipeline/diagnostics.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/parameter_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Plan-local device placement identifier used by parameter slabs.
typedef uint32_t id4_pipeline_device_placement_id_t;

// Invalid device placement identifier.
#define ID4_PIPELINE_DEVICE_PLACEMENT_ID_INVALID UINT32_MAX

// Source parameter span and planned slab target span.
typedef struct id4_pipeline_parameter_request_t {
  // Parameter key in the provider scope.
  iree_string_view_t key;
  // Source and target byte range for the gather operation.
  iree_io_parameter_span_t span;
} id4_pipeline_parameter_request_t;

// Returns a source-to-target byte span for parameter gathers.
static inline iree_io_parameter_span_t id4_pipeline_parameter_span(
    uint64_t parameter_offset, iree_device_size_t buffer_offset,
    iree_device_size_t length) {
  iree_io_parameter_span_t span;
  span.parameter_offset = parameter_offset;
  span.buffer_offset = buffer_offset;
  span.length = length;
  return span;
}

// Returns a parameter request for gathering |key| into |span|.
static inline id4_pipeline_parameter_request_t id4_pipeline_parameter_request(
    iree_string_view_t key, iree_io_parameter_span_t span) {
  id4_pipeline_parameter_request_t request;
  request.key = key;
  request.span = span;
  return request;
}

// Appends a packed gather span to a parameter slab under construction.
iree_status_t id4_pipeline_parameter_slab_pack_span(
    iree_device_size_t byte_length, iree_device_size_t alignment,
    iree_device_size_t* io_slab_byte_length,
    iree_io_parameter_span_t* out_span);

// Returns HAL buffer parameters for a device-local parameter slab.
static inline iree_hal_buffer_params_t
id4_pipeline_parameter_slab_device_local_params(
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_usage_t usage,
    iree_device_size_t min_alignment) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = usage;
  params.queue_affinity = queue_affinity;
  params.min_alignment = min_alignment;
  return params;
}

// Planned parameter slab populated during prepare.
typedef struct id4_pipeline_parameter_slab_plan_t {
  // Provider scope containing all request keys in this slab.
  iree_string_view_t scope;
  // Placement where the slab is allocated.
  id4_pipeline_device_placement_id_t placement_id;
  // HAL buffer parameters used for slab allocation.
  iree_hal_buffer_params_t target_params;
  // Total slab byte length.
  iree_device_size_t byte_length;
  // Required slab base alignment in bytes.
  iree_device_size_t alignment;
  // Number of parameter requests.
  iree_host_size_t request_count;
  // Parameter requests in gather enumeration order.
  const id4_pipeline_parameter_request_t* requests;
} id4_pipeline_parameter_slab_plan_t;

// Returns a parameter slab plan value for |requests|.
static inline id4_pipeline_parameter_slab_plan_t
id4_pipeline_make_parameter_slab_plan(
    iree_string_view_t scope, id4_pipeline_device_placement_id_t placement_id,
    iree_hal_buffer_params_t target_params, iree_device_size_t byte_length,
    iree_device_size_t alignment, iree_host_size_t request_count,
    const id4_pipeline_parameter_request_t* requests) {
  id4_pipeline_parameter_slab_plan_t plan;
  plan.scope = scope;
  plan.placement_id = placement_id;
  plan.target_params = target_params;
  plan.byte_length = byte_length;
  plan.alignment = alignment;
  plan.request_count = request_count;
  plan.requests = requests;
  return plan;
}

// Returns a device-local parameter slab plan value for |requests|.
static inline id4_pipeline_parameter_slab_plan_t
id4_pipeline_make_device_local_parameter_slab_plan(
    iree_string_view_t scope, id4_pipeline_device_placement_id_t placement_id,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_usage_t usage,
    iree_device_size_t byte_length, iree_device_size_t alignment,
    iree_host_size_t request_count,
    const id4_pipeline_parameter_request_t* requests) {
  return id4_pipeline_make_parameter_slab_plan(
      scope, placement_id,
      id4_pipeline_parameter_slab_device_local_params(queue_affinity, usage,
                                                      alignment),
      byte_length, alignment, request_count, requests);
}

// State passed to the IREE parameter provider enumerator callback.
typedef struct id4_pipeline_parameter_slab_enumerator_state_t {
  // Slab plan being enumerated.
  const id4_pipeline_parameter_slab_plan_t* slab;
} id4_pipeline_parameter_slab_enumerator_state_t;

// Resolved parameter slab load work for one planned slab.
typedef struct id4_pipeline_parameter_slab_load_t {
  // Plan-local slab index.
  iree_host_size_t slab_index;
  // Planned slab metadata to load.
  const id4_pipeline_parameter_slab_plan_t* slab;
  // Device index within the plan device group.
  iree_host_size_t device_index;
  // HAL device where the slab buffer is allocated and populated.
  iree_hal_device_t* device;
  // Queue affinity used for the provider gather operation.
  iree_hal_queue_affinity_t queue_affinity;
} id4_pipeline_parameter_slab_load_t;

// Loaded parameter slab buffers owned by a prepared bundle.
typedef struct id4_pipeline_parameter_slab_set_t
    id4_pipeline_parameter_slab_set_t;

// Validates that a parameter slab references valid placements and byte ranges.
iree_status_t id4_pipeline_parameter_slab_validate(
    const id4_pipeline_parameter_slab_plan_t* slab,
    iree_host_size_t placement_count);

// Enumerates one planned parameter request in IREE provider callback form.
iree_status_t id4_pipeline_parameter_slab_enumerate(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span);

// Returns an IREE parameter enumerator for |state|.
iree_io_parameter_enumerator_t id4_pipeline_parameter_slab_enumerator(
    id4_pipeline_parameter_slab_enumerator_state_t* state);

// Allocates and asynchronously gathers all planned slabs from |provider|.
iree_status_t id4_pipeline_parameter_slab_set_load(
    iree_io_parameter_provider_t* provider,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t load_count,
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_string_view_t stage_name,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t** out_slab_set);

// Retains |slab_set| for the caller.
void id4_pipeline_parameter_slab_set_retain(
    id4_pipeline_parameter_slab_set_t* slab_set);

// Releases |slab_set| from the caller.
void id4_pipeline_parameter_slab_set_release(
    id4_pipeline_parameter_slab_set_t* slab_set);

// Returns the number of loaded slabs in |slab_set|.
iree_host_size_t id4_pipeline_parameter_slab_set_count(
    const id4_pipeline_parameter_slab_set_t* slab_set);

// Returns loaded slab buffer |index| or NULL when out of range.
iree_hal_buffer_t* id4_pipeline_parameter_slab_set_buffer_at(
    const id4_pipeline_parameter_slab_set_t* slab_set, iree_host_size_t index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_PARAMETER_SLAB_H_
