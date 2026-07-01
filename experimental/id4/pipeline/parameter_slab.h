// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PARAMETER_SLAB_H_
#define EXPERIMENTAL_ID4_PIPELINE_PARAMETER_SLAB_H_

#include <stdbool.h>
#include <stdint.h>

#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/region.h"
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
  // Issue-time binding-table slot used for this slab.
  uint32_t binding_slot;
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

// Prepare-time parameter loading step kind.
typedef uint32_t id4_pipeline_parameter_load_step_kind_t;

// Parameter loading step kind values.
typedef enum id4_pipeline_parameter_load_step_kind_e {
  // Direct IREE parameter provider gather into a final parameter slab.
  ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_GATHER = 1u,
  // Provider FP8 e4m3 weights and F32 row scales encoded into BF16 storage.
  ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16 = 2u,
  // Provider BF16 matrix weights packed into compact BF16 RHS tiles.
  ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_BF16_LINEAR_RHS_TILE = 3u,
  // Provider FP8 e4m3 weights and F32 row scales encoded into compact BF16 RHS
  // tiles.
  ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16_LINEAR_RHS_TILE =
      4u,
} id4_pipeline_parameter_load_step_kind_e;

// Prepare-time parameter loading group kind.
typedef uint32_t id4_pipeline_parameter_load_group_kind_t;

// Parameter loading group kind values.
typedef enum id4_pipeline_parameter_load_group_kind_e {
  // Single direct provider gather submitted independently.
  ID4_PIPELINE_PARAMETER_LOAD_GROUP_KIND_GATHER = 1u,
  // Contiguous encoded load steps sharing one bounded staging run.
  ID4_PIPELINE_PARAMETER_LOAD_GROUP_KIND_ENCODE = 2u,
} id4_pipeline_parameter_load_group_kind_e;

// Sentinel readiness partition for load steps without a consumer-region key.
#define ID4_PIPELINE_PARAMETER_LOAD_READINESS_GROUP_NONE IREE_HOST_SIZE_MAX

// Production staging chunk byte budget for prepare-time parameter encoders.
static const iree_device_size_t
    ID4_PIPELINE_PARAMETER_ENCODER_DEFAULT_STAGING_CHUNK_BYTE_CAPACITY =
        1024ull * 1024ull * 1024ull;

// Provider source tensor consumed by a parameter load step.
typedef struct id4_pipeline_parameter_load_source_t {
  // Provider scope containing the source tensor.
  iree_string_view_t source_scope;
  // Provider key for the source tensor.
  iree_string_view_t key;
  // Scalar element type stored by the provider.
  id4_pipeline_tensor_dtype_t dtype;
  // Logical tensor shape stored by the provider.
  id4_pipeline_tensor_shape_t shape;
  // Dense source tensor byte length.
  iree_device_size_t byte_length;
} id4_pipeline_parameter_load_source_t;

// Returns a parameter load source descriptor.
static inline id4_pipeline_parameter_load_source_t
id4_pipeline_parameter_load_source(iree_string_view_t source_scope,
                                   iree_string_view_t key,
                                   id4_pipeline_tensor_dtype_t dtype,
                                   id4_pipeline_tensor_shape_t shape,
                                   iree_device_size_t byte_length) {
  id4_pipeline_parameter_load_source_t source;
  source.source_scope = source_scope;
  source.key = key;
  source.dtype = dtype;
  source.shape = shape;
  source.byte_length = byte_length;
  return source;
}

// Prepare-time work that populates final parameter slab storage.
typedef struct id4_pipeline_parameter_load_step_t {
  // Human-readable load step name for diagnostics and plan dumps.
  iree_string_view_t name;
  // Operation performed by this loading step.
  id4_pipeline_parameter_load_step_kind_t kind;
  // Provider source scope supplying direct gather requests.
  iree_string_view_t source_scope;
  // Number of provider source descriptors used by encoded load steps.
  iree_host_size_t source_count;
  // Provider source descriptors used by encoded load steps.
  const id4_pipeline_parameter_load_source_t* sources;
  // Final parameter slab populated by this step.
  iree_host_size_t target_slab_index;
  // First request ordinal in the target slab request table.
  iree_host_size_t request_offset;
  // Number of requests consumed from the target slab request table.
  iree_host_size_t request_count;
  // Optional request ordinals for non-contiguous direct gather steps.
  const iree_host_size_t* request_indices;
  // Semantic partition key used to keep readiness edges consumer-precise.
  iree_host_size_t readiness_group_key;
} id4_pipeline_parameter_load_step_t;

// Contiguous prepare-time work submitted under one readiness edge.
typedef struct id4_pipeline_parameter_load_group_t {
  // First load-step ordinal represented by this group.
  iree_host_size_t step_offset;
  // Number of load steps represented by this group.
  iree_host_size_t step_count;
  // Submission strategy used for this group.
  id4_pipeline_parameter_load_group_kind_t kind;
  // Final parameter slab populated by all steps in this group.
  iree_host_size_t target_slab_index;
} id4_pipeline_parameter_load_group_t;

// Runtime context for submitting one planned parameter load group.
typedef struct id4_pipeline_parameter_load_group_context_t {
  // Plan-local load group ordinal.
  iree_host_size_t group_index;
  // First region that consumes the group, or IREE_HOST_SIZE_MAX when unknown.
  iree_host_size_t first_consumer_region_id;
  // Region issuing the group, or IREE_HOST_SIZE_MAX outside region issue.
  iree_host_size_t submit_region_id;
} id4_pipeline_parameter_load_group_context_t;

// Returns a direct provider-gather load step into a final parameter slab.
static inline id4_pipeline_parameter_load_step_t
id4_pipeline_parameter_gather_load_step(iree_string_view_t name,
                                        iree_string_view_t source_scope,
                                        iree_host_size_t target_slab_index,
                                        iree_host_size_t request_offset,
                                        iree_host_size_t request_count) {
  id4_pipeline_parameter_load_step_t step;
  step.name = name;
  step.kind = ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_GATHER;
  step.source_scope = source_scope;
  step.source_count = 0;
  step.sources = NULL;
  step.target_slab_index = target_slab_index;
  step.request_offset = request_offset;
  step.request_count = request_count;
  step.request_indices = NULL;
  step.readiness_group_key = ID4_PIPELINE_PARAMETER_LOAD_READINESS_GROUP_NONE;
  return step;
}

// Returns a direct provider-gather load step over explicit request ordinals.
static inline id4_pipeline_parameter_load_step_t
id4_pipeline_parameter_indexed_gather_load_step(
    iree_string_view_t name, iree_string_view_t source_scope,
    iree_host_size_t target_slab_index, iree_host_size_t request_count,
    const iree_host_size_t* request_indices) {
  id4_pipeline_parameter_load_step_t step;
  step.name = name;
  step.kind = ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_GATHER;
  step.source_scope = source_scope;
  step.source_count = 0;
  step.sources = NULL;
  step.target_slab_index = target_slab_index;
  step.request_offset = 0;
  step.request_count = request_count;
  step.request_indices = request_indices;
  step.readiness_group_key = ID4_PIPELINE_PARAMETER_LOAD_READINESS_GROUP_NONE;
  return step;
}

// Returns an FP8 e4m3 plus row-scale encoder load step into a final slab.
static inline id4_pipeline_parameter_load_step_t
id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_load_step(
    iree_string_view_t name, iree_host_size_t source_count,
    const id4_pipeline_parameter_load_source_t* sources,
    iree_host_size_t target_slab_index, iree_host_size_t request_offset) {
  id4_pipeline_parameter_load_step_t step;
  step.name = name;
  step.kind =
      ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16;
  step.source_scope = iree_string_view_empty();
  step.source_count = source_count;
  step.sources = sources;
  step.target_slab_index = target_slab_index;
  step.request_offset = request_offset;
  step.request_count = 1;
  step.request_indices = NULL;
  step.readiness_group_key = ID4_PIPELINE_PARAMETER_LOAD_READINESS_GROUP_NONE;
  return step;
}

// Returns a BF16 linear RHS tile encoder load step into a final slab.
static inline id4_pipeline_parameter_load_step_t
id4_pipeline_parameter_encode_bf16_linear_rhs_tile_load_step(
    iree_string_view_t name, iree_host_size_t source_count,
    const id4_pipeline_parameter_load_source_t* sources,
    iree_host_size_t target_slab_index, iree_host_size_t request_offset) {
  id4_pipeline_parameter_load_step_t step;
  step.name = name;
  step.kind = ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_BF16_LINEAR_RHS_TILE;
  step.source_scope = iree_string_view_empty();
  step.source_count = source_count;
  step.sources = sources;
  step.target_slab_index = target_slab_index;
  step.request_offset = request_offset;
  step.request_count = 1;
  step.request_indices = NULL;
  step.readiness_group_key = ID4_PIPELINE_PARAMETER_LOAD_READINESS_GROUP_NONE;
  return step;
}

// Returns an FP8 e4m3 plus row-scale linear RHS tile encoder load step.
static inline id4_pipeline_parameter_load_step_t
id4_pipeline_parameter_encode_fp8_e4m3_scaled_to_bf16_linear_rhs_tile_load_step(
    iree_string_view_t name, iree_host_size_t source_count,
    const id4_pipeline_parameter_load_source_t* sources,
    iree_host_size_t target_slab_index, iree_host_size_t request_offset) {
  id4_pipeline_parameter_load_step_t step;
  step.name = name;
  step.kind =
      ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_ENCODE_FP8_E4M3_SCALED_TO_BF16_LINEAR_RHS_TILE;
  step.source_scope = iree_string_view_empty();
  step.source_count = source_count;
  step.sources = sources;
  step.target_slab_index = target_slab_index;
  step.request_offset = request_offset;
  step.request_count = 1;
  step.request_indices = NULL;
  step.readiness_group_key = ID4_PIPELINE_PARAMETER_LOAD_READINESS_GROUP_NONE;
  return step;
}

// Returns a parameter slab plan value for |requests|.
static inline id4_pipeline_parameter_slab_plan_t
id4_pipeline_make_parameter_slab_plan(
    iree_string_view_t scope, id4_pipeline_device_placement_id_t placement_id,
    uint32_t binding_slot, iree_hal_buffer_params_t target_params,
    iree_device_size_t byte_length, iree_device_size_t alignment,
    iree_host_size_t request_count,
    const id4_pipeline_parameter_request_t* requests) {
  id4_pipeline_parameter_slab_plan_t plan;
  plan.scope = scope;
  plan.placement_id = placement_id;
  plan.binding_slot = binding_slot;
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
    uint32_t binding_slot, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_buffer_usage_t usage, iree_device_size_t byte_length,
    iree_device_size_t alignment, iree_host_size_t request_count,
    const id4_pipeline_parameter_request_t* requests) {
  return id4_pipeline_make_parameter_slab_plan(
      scope, placement_id, binding_slot,
      id4_pipeline_parameter_slab_device_local_params(queue_affinity, usage,
                                                      alignment),
      byte_length, alignment, request_count, requests);
}

// State passed to the IREE parameter provider enumerator callback.
typedef struct id4_pipeline_parameter_slab_enumerator_state_t {
  // Slab plan being enumerated.
  const id4_pipeline_parameter_slab_plan_t* slab;
  // First request ordinal visible to the enumerator.
  iree_host_size_t request_offset;
  // Number of requests visible to the enumerator.
  iree_host_size_t request_count;
  // Optional explicit request ordinals visible to the enumerator.
  const iree_host_size_t* request_indices;
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

// Options for allocating and populating planned parameter slab buffers.
typedef struct id4_pipeline_parameter_slab_set_load_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Provider used for direct gathers and encoded source tensor gathers.
  iree_io_parameter_provider_t* provider;
  // Kernel library used by encoded load steps.
  id4_pipeline_kernel_library_t* kernel_library;
  // Loom kernel cache used by encoded load steps.
  id4_pipeline_kernel_cache_t* kernel_cache;
  // HAL executable cache used by encoded load steps.
  iree_hal_executable_cache_t* executable_cache;
  // HAL command-buffer mode used by prepare-time encoder dispatches.
  iree_hal_command_buffer_mode_t command_buffer_mode;
  // Maximum source bytes staged in one encoder chunk.
  iree_device_size_t encoder_staging_chunk_byte_capacity;
  // Diagnostic artifact classes requested while JITing encoder kernels.
  id4_pipeline_kernel_diagnostic_artifact_flags_t diagnostic_artifact_flags;
  // Semaphores that all parameter loading waits on.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphores signaled when all parameter loading is complete.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Diagnostics sink for load, gather, encode, and failure events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_pipeline_parameter_slab_set_load_options_t;

// Loaded parameter slab buffers owned by a prepared bundle.
typedef struct id4_pipeline_parameter_slab_set_t
    id4_pipeline_parameter_slab_set_t;

// Issue-local context for deferred parameter load submissions.
typedef struct id4_pipeline_parameter_slab_issue_context_t
    id4_pipeline_parameter_slab_issue_context_t;

// Validates that a parameter slab references valid placements and byte ranges.
iree_status_t id4_pipeline_parameter_slab_validate(
    const id4_pipeline_parameter_slab_plan_t* slab,
    iree_host_size_t placement_count);

// Validates that |step| references a valid final slab request range.
iree_status_t id4_pipeline_parameter_load_step_validate(
    const id4_pipeline_parameter_load_step_t* step, iree_host_size_t slab_count,
    const id4_pipeline_parameter_slab_plan_t* slabs);

// Returns the number of readiness groups represented by |load_steps|.
iree_status_t id4_pipeline_parameter_load_group_count(
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_host_size_t* out_group_count);

// Returns readiness group |group_index| represented by |load_steps|.
iree_status_t id4_pipeline_parameter_load_group_at(
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_host_size_t group_index,
    id4_pipeline_parameter_load_group_t* out_group);

// Enumerates one planned parameter request in IREE provider callback form.
iree_status_t id4_pipeline_parameter_slab_enumerate(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span);

// Returns an IREE parameter enumerator for |state|.
iree_io_parameter_enumerator_t id4_pipeline_parameter_slab_enumerator(
    id4_pipeline_parameter_slab_enumerator_state_t* state);

// Allocates and asynchronously populates all planned slabs.
iree_status_t id4_pipeline_parameter_slab_set_load(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    iree_host_size_t load_count,
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_string_view_t stage_name, iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t** out_slab_set);

// Allocates final slabs and retained readiness edges without submitting load
// work.
iree_status_t id4_pipeline_parameter_slab_set_prepare(
    const id4_pipeline_parameter_slab_set_load_options_t* options,
    iree_host_size_t load_count,
    const id4_pipeline_parameter_slab_load_t* loads,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_string_view_t stage_name, iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t** out_slab_set);

// Creates an issue-local context for deferred parameter load submissions.
iree_status_t id4_pipeline_parameter_slab_issue_context_create(
    id4_pipeline_parameter_slab_set_t* slab_set,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_issue_context_t** out_context);

// Submits any issue-local transient cleanup and returns cleanup readiness
// edges.
iree_status_t id4_pipeline_parameter_slab_issue_context_finish(
    id4_pipeline_parameter_slab_issue_context_t* context,
    iree_hal_semaphore_list_t* out_cleanup_wait_list);

// Releases an issue-local deferred loading context.
void id4_pipeline_parameter_slab_issue_context_release(
    id4_pipeline_parameter_slab_issue_context_t* context);

// Submits planned parameter load group |group_index| through |context|,
// signaling its retained readiness edge when the final slab bytes are
// populated.
iree_status_t id4_pipeline_parameter_slab_issue_context_submit_load_group(
    id4_pipeline_parameter_slab_issue_context_t* context,
    iree_host_size_t load_step_count,
    const id4_pipeline_parameter_load_step_t* load_steps,
    id4_pipeline_parameter_load_group_context_t group_context,
    iree_string_view_t stage_name,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink);

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

// Returns the number of retained parameter load readiness groups.
iree_host_size_t id4_pipeline_parameter_slab_set_load_group_count(
    const id4_pipeline_parameter_slab_set_t* slab_set);

// Returns retained parameter load group |index| represented by |slab_set|.
iree_status_t id4_pipeline_parameter_slab_set_load_group_at(
    const id4_pipeline_parameter_slab_set_t* slab_set, iree_host_size_t index,
    id4_pipeline_parameter_load_group_t* out_group);

// Returns true when |slab_set| owns retained load context for issue-time
// materialization.
bool id4_pipeline_parameter_slab_set_has_deferred_load_context(
    const id4_pipeline_parameter_slab_set_t* slab_set);

// Returns the borrowed readiness semaphore and payload for load group |index|.
iree_status_t id4_pipeline_parameter_slab_set_load_group_ready_at(
    const id4_pipeline_parameter_slab_set_t* slab_set, iree_host_size_t index,
    iree_hal_semaphore_t** out_semaphore, uint64_t* out_payload_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_PARAMETER_SLAB_H_
