// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PLAN_H_
#define EXPERIMENTAL_ID4_PIPELINE_PLAN_H_

#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/kernel_library.h"
#include "experimental/id4/pipeline/parameter_slab.h"
#include "experimental/id4/pipeline/region.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opaque planned execution object.
typedef struct id4_pipeline_plan_t id4_pipeline_plan_t;

// Opaque semantic pipeline program retained by program-backed plans.
typedef struct id4_pipeline_program_t id4_pipeline_program_t;

// Device placement selected during planning.
typedef struct id4_pipeline_device_placement_t {
  // Human-readable role for diagnostics.
  iree_string_view_t role;
  // Device index within the retained HAL device group.
  iree_host_size_t device_index;
  // Queue affinity used for work submitted on this placement.
  iree_hal_queue_affinity_t queue_affinity;
} id4_pipeline_device_placement_t;

// Config key/value pair used to specialize a planned kernel.
typedef id4_pipeline_kernel_config_binding_t id4_pipeline_plan_config_binding_t;

// Stage boundary tensor behavior flags.
typedef uint32_t id4_pipeline_boundary_tensor_flags_t;

// Stage boundary tensor behavior flag bits.
typedef enum id4_pipeline_boundary_tensor_flag_bits_e {
  // Tensor is supplied by the stage caller in the issue-time binding table.
  ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_IMPORTED = 1u << 0,
  // Tensor is exposed as a stage output after execution completes.
  ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED = 1u << 1,
  // Tensor contents are initialized before stage execution begins.
  ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_INITIALIZED = 1u << 2,
} id4_pipeline_boundary_tensor_flag_bits_t;

// Planned memory slab outside of parameter slab storage.
typedef struct id4_pipeline_memory_slab_plan_t {
  // Human-readable slab name for diagnostics.
  iree_string_view_t name;
  // Plan-local placement identifier.
  uint32_t placement_id;
  // Issue-time binding-table slot used by this slab.
  uint32_t binding_slot;
  // HAL buffer parameters used when materializing the slab.
  iree_hal_buffer_params_t params;
  // Planned slab byte length.
  iree_device_size_t byte_length;
  // Required slab base alignment in bytes.
  iree_device_size_t alignment;
  // Peak concurrently live bytes represented in the slab.
  iree_device_size_t high_water_mark;
} id4_pipeline_memory_slab_plan_t;

// Planned constant tensor embedded into a constant slab.
typedef struct id4_pipeline_constant_request_t {
  // Constant tensor diagnostic name.
  iree_string_view_t name;
  // Target byte range inside the constant slab.
  iree_io_parameter_span_t span;
} id4_pipeline_constant_request_t;

// Planned constant slab populated during prepare from program-owned data.
typedef struct id4_pipeline_constant_slab_plan_t {
  // Human-readable slab name for diagnostics.
  iree_string_view_t name;
  // Plan-local placement identifier.
  uint32_t placement_id;
  // Issue-time binding-table slot used for this slab.
  uint32_t binding_slot;
  // HAL buffer parameters used for slab allocation.
  iree_hal_buffer_params_t target_params;
  // Total slab byte length.
  iree_device_size_t byte_length;
  // Required slab base alignment in bytes.
  iree_device_size_t alignment;
  // Number of constant requests.
  iree_host_size_t request_count;
  // Constant requests in program constant-operation order.
  const id4_pipeline_constant_request_t* requests;
} id4_pipeline_constant_slab_plan_t;

// Planned external tensor bound at a stage boundary.
typedef struct id4_pipeline_boundary_tensor_plan_t {
  // Tensor layout and stable diagnostic name.
  id4_pipeline_tensor_layout_t layout;
  // Boundary behavior flags.
  id4_pipeline_boundary_tensor_flags_t flags;
  // Region index whose binding table contains this tensor.
  uint32_t region_id;
  // Plan-local placement identifier.
  uint32_t placement_id;
  // Issue-time binding-table slot containing this tensor.
  uint32_t binding_slot;
} id4_pipeline_boundary_tensor_plan_t;

// Planned kernel specialization emitted by a stage.
typedef struct id4_pipeline_kernel_plan_t {
  // Stable kernel selection key.
  iree_string_view_t specialization_key;
  // Stable module path resolved through the kernel library.
  iree_string_view_t module_path;
  // Exported HAL function name resolved during preparation.
  iree_string_view_t function_name;
  // Plan-local placement identifier.
  uint32_t placement_id;
  // Number of config bindings copied into this kernel plan.
  iree_host_size_t config_binding_count;
  // Config bindings copied into this kernel plan.
  const id4_pipeline_plan_config_binding_t* config_bindings;
} id4_pipeline_kernel_plan_t;

// Planned reusable executable region.
typedef struct id4_pipeline_region_plan_t {
  // Human-readable region name for diagnostics.
  iree_string_view_t name;
  // Plan-local placement identifier.
  uint32_t placement_id;
  // Exact issue-time binding-table capacity.
  iree_host_size_t binding_capacity;
  // Binding-table slot reserved for the local transient slab.
  uint32_t local_binding_slot;
  // Required alignment for local tensor suballocations in the region slab.
  iree_device_size_t local_tensor_alignment;
  // Dry-run region statistics used for diagnostics and memory planning.
  id4_pipeline_region_statistics_t statistics;
  // Number of local tensor lifetime records.
  iree_host_size_t local_lifetime_count;
  // Local tensor lifetime records owned by the containing plan.
  const id4_pipeline_region_local_lifetime_t* local_lifetimes;
} id4_pipeline_region_plan_t;

// Planned diagnostics tap.
typedef struct id4_pipeline_diagnostic_tap_plan_t {
  // Human-readable tap name.
  iree_string_view_t name;
  // Region index containing the tapped value.
  uint32_t region_id;
  // Plan-local placement identifier.
  uint32_t placement_id;
  // Issue-time binding-table slot receiving the captured tensor copy.
  uint32_t binding_slot;
  // Operation ordinal after which the tap can be captured.
  iree_host_size_t after_operation_ordinal;
  // Tensor or value name exposed by the tap.
  iree_string_view_t target_name;
  // Tensor layout and stable capture record name.
  id4_pipeline_tensor_layout_t layout;
} id4_pipeline_diagnostic_tap_plan_t;

// Options for creating an inspectable plan.
typedef struct id4_pipeline_plan_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Stage name used in diagnostics.
  iree_string_view_t stage_name;
  // Optional immutable source program retained by program-backed plans.
  const id4_pipeline_program_t* source_program;
  // Device group retained by the plan.
  iree_hal_device_group_t* device_group;
  // Number of explicit placements.
  iree_host_size_t placement_count;
  // Explicit placements to copy into the plan.
  const id4_pipeline_device_placement_t* placements;
  // Number of parameter slabs.
  iree_host_size_t parameter_slab_count;
  // Parameter slabs to copy into the plan.
  const id4_pipeline_parameter_slab_plan_t* parameter_slabs;
  // Number of planned constant slabs.
  iree_host_size_t constant_slab_count;
  // Constant slabs to copy into the plan.
  const id4_pipeline_constant_slab_plan_t* constant_slabs;
  // Number of planned non-parameter memory slabs.
  iree_host_size_t memory_slab_count;
  // Planned non-parameter memory slabs to copy into the plan.
  const id4_pipeline_memory_slab_plan_t* memory_slabs;
  // Number of planned external boundary tensors.
  iree_host_size_t boundary_tensor_count;
  // Planned external boundary tensors to copy into the plan.
  const id4_pipeline_boundary_tensor_plan_t* boundary_tensors;
  // Number of planned kernel specializations.
  iree_host_size_t kernel_count;
  // Planned kernel specializations to copy into the plan.
  const id4_pipeline_kernel_plan_t* kernels;
  // Number of planned reusable regions.
  iree_host_size_t region_count;
  // Planned reusable regions to copy into the plan.
  const id4_pipeline_region_plan_t* regions;
  // Number of planned diagnostics taps.
  iree_host_size_t diagnostic_tap_count;
  // Planned diagnostics taps to copy into the plan.
  const id4_pipeline_diagnostic_tap_plan_t* diagnostic_taps;
  // Diagnostics sink for plan creation events.
  id4_pipeline_diagnostics_sink_t* diagnostics_sink;
} id4_pipeline_plan_create_options_t;

// Creates a plan by copying placement and parameter-slab metadata.
iree_status_t id4_pipeline_plan_create(
    const id4_pipeline_plan_create_options_t* options,
    iree_allocator_t host_allocator, id4_pipeline_plan_t** out_plan);

// Retains |plan| for the caller.
void id4_pipeline_plan_retain(id4_pipeline_plan_t* plan);

// Releases |plan| from the caller.
void id4_pipeline_plan_release(id4_pipeline_plan_t* plan);

// Returns the stage name copied into |plan|.
iree_string_view_t id4_pipeline_plan_stage_name(
    const id4_pipeline_plan_t* plan);

// Returns the retained device group associated with |plan|.
iree_hal_device_group_t* id4_pipeline_plan_device_group(
    const id4_pipeline_plan_t* plan);

// Returns the number of placements in |plan|.
iree_host_size_t id4_pipeline_plan_placement_count(
    const id4_pipeline_plan_t* plan);

// Returns placement |index| or NULL when out of range.
const id4_pipeline_device_placement_t* id4_pipeline_plan_placement_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index);

// Returns the retained immutable source program for program-backed plans.
const id4_pipeline_program_t* id4_pipeline_plan_source_program(
    const id4_pipeline_plan_t* plan);

// Returns the number of parameter slabs in |plan|.
iree_host_size_t id4_pipeline_plan_parameter_slab_count(
    const id4_pipeline_plan_t* plan);

// Returns parameter slab |index| or NULL when out of range.
const id4_pipeline_parameter_slab_plan_t* id4_pipeline_plan_parameter_slab_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index);

// Returns the number of constant slabs in |plan|.
iree_host_size_t id4_pipeline_plan_constant_slab_count(
    const id4_pipeline_plan_t* plan);

// Returns constant slab |index| or NULL when out of range.
const id4_pipeline_constant_slab_plan_t* id4_pipeline_plan_constant_slab_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index);

// Returns the number of memory slabs in |plan|.
iree_host_size_t id4_pipeline_plan_memory_slab_count(
    const id4_pipeline_plan_t* plan);

// Returns memory slab |index| or NULL when out of range.
const id4_pipeline_memory_slab_plan_t* id4_pipeline_plan_memory_slab_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index);

// Returns the number of boundary tensors in |plan|.
iree_host_size_t id4_pipeline_plan_boundary_tensor_count(
    const id4_pipeline_plan_t* plan);

// Returns boundary tensor |index| or NULL when out of range.
const id4_pipeline_boundary_tensor_plan_t* id4_pipeline_plan_boundary_tensor_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index);

// Returns the number of kernel specializations in |plan|.
iree_host_size_t id4_pipeline_plan_kernel_count(
    const id4_pipeline_plan_t* plan);

// Returns kernel specialization |index| or NULL when out of range.
const id4_pipeline_kernel_plan_t* id4_pipeline_plan_kernel_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index);

// Returns the number of regions in |plan|.
iree_host_size_t id4_pipeline_plan_region_count(
    const id4_pipeline_plan_t* plan);

// Returns region |index| or NULL when out of range.
const id4_pipeline_region_plan_t* id4_pipeline_plan_region_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index);

// Returns the number of diagnostic taps in |plan|.
iree_host_size_t id4_pipeline_plan_diagnostic_tap_count(
    const id4_pipeline_plan_t* plan);

// Returns diagnostic tap |index| or NULL when out of range.
const id4_pipeline_diagnostic_tap_plan_t* id4_pipeline_plan_diagnostic_tap_at(
    const id4_pipeline_plan_t* plan, iree_host_size_t index);

// Loads all planned parameter slabs through |provider|.
iree_status_t id4_pipeline_plan_load_parameter_slabs(
    const id4_pipeline_plan_t* plan, iree_io_parameter_provider_t* provider,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_slab_set_t** out_slab_set);

// Appends deterministic JSON describing the plan.
iree_status_t id4_pipeline_plan_format_json(const id4_pipeline_plan_t* plan,
                                            iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_PLAN_H_
