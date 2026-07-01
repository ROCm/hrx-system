// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_DIAGNOSTICS_H_
#define EXPERIMENTAL_ID4_PIPELINE_DIAGNOSTICS_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Pipeline diagnostic event kind.
typedef enum id4_pipeline_diagnostic_event_kind_e {
  // Lifecycle event such as load, plan, prepare, or issue.
  ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_LIFECYCLE = 0,
  // Structured plan event.
  ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_PLAN = 1,
  // Parameter slab planning or loading event.
  ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_PARAMETER_SLAB = 2,
  // Kernel compilation, emission, or HAL preparation event.
  ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_KERNEL = 3,
  // Host-observed phase timing event.
  ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_TIMING = 4,
} id4_pipeline_diagnostic_event_kind_t;

// Diagnostic artifacts requested from the Loom compiler path.
typedef uint32_t id4_pipeline_kernel_diagnostic_artifact_flags_t;

// Diagnostic artifact request bits.
typedef enum id4_pipeline_kernel_diagnostic_artifact_flag_bits_e {
  // Copy textual Loom module IR after successful compilation.
  ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_MODULE_TEXT = 1u << 0,
  // Copy binary Loom bytecode after successful compilation.
  ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_MODULE_BYTECODE = 1u << 1,
  // Copy the JSON compile report.
  ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_COMPILE_REPORT_JSON = 1u << 2,
  // Copy the JSON emit artifact manifest.
  ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_EMIT_MANIFEST_JSON = 1u << 3,
} id4_pipeline_kernel_diagnostic_artifact_flag_bits_t;

// Parameter slab payload attached to parameter-slab diagnostic events.
typedef struct id4_pipeline_parameter_slab_diagnostic_t {
  // Plan-local slab index.
  iree_host_size_t slab_index;
  // Request index, or IREE_HOST_SIZE_MAX when the event describes the slab.
  iree_host_size_t request_index;
  // Parameter scope associated with this event.
  iree_string_view_t scope;
  // Parameter key for request-level events.
  iree_string_view_t parameter_key;
  // Source parameter byte offset for request-level events.
  uint64_t parameter_offset;
  // Target slab byte offset for request-level events.
  uint64_t buffer_offset;
  // Byte length for request-level events.
  uint64_t length;
  // Plan-local placement identifier.
  uint32_t placement_id;
  // Device index within the plan device group.
  iree_host_size_t device_index;
  // Queue affinity used by loading work.
  iree_hal_queue_affinity_t queue_affinity;
  // Total slab byte length.
  iree_device_size_t slab_byte_length;
  // Required slab base alignment in bytes.
  iree_device_size_t slab_alignment;
  // Number of parameter requests in the slab.
  iree_host_size_t request_count;
} id4_pipeline_parameter_slab_diagnostic_t;

// Parameter loading payload attached to parameter-slab diagnostic events.
typedef struct id4_pipeline_parameter_load_diagnostic_t {
  // Plan-local slab index populated by this loading window.
  iree_host_size_t slab_index;
  // First load-step ordinal represented by this loading window.
  iree_host_size_t load_step_offset;
  // Number of load steps represented by this loading window.
  iree_host_size_t load_step_count;
  // Number of bounded staging slots allocated for encoded source tensors.
  iree_host_size_t staging_slot_count;
  // Byte length of one bounded staging slot.
  iree_device_size_t staging_slot_byte_length;
  // Total byte length of all bounded staging slots.
  iree_device_size_t staging_total_byte_length;
  // Number of staging chunks submitted by this loading window.
  iree_host_size_t staging_chunk_count;
  // Number of logical provider source tensors gathered into staging.
  iree_host_size_t logical_source_count;
  // Number of provider gather batches submitted by this loading window.
  iree_host_size_t source_gather_batch_count;
  // Total provider source bytes gathered by this loading window.
  iree_device_size_t source_byte_length;
  // Total final slab bytes populated by this loading window.
  iree_device_size_t target_byte_length;
  // Number of encoder dispatches recorded by this loading window.
  iree_host_size_t encoder_dispatch_count;
} id4_pipeline_parameter_load_diagnostic_t;

// Kernel payload attached to kernel diagnostic events.
typedef struct id4_pipeline_kernel_diagnostic_t {
  // Kernel-cache phase associated with the event.
  iree_string_view_t phase;
  // Source identifier passed to Loom.
  iree_string_view_t source_identifier;
  // Runtime module path passed to Loom.
  iree_string_view_t module_path;
  // Target processor selected by the kernel cache.
  iree_string_view_t target_processor;
  // Loom artifact format emitted by the target backend.
  iree_string_view_t loom_artifact_format;
  // HAL executable format inferred from artifact bytes.
  iree_string_view_t hal_executable_format;
  // Number of Loom config bindings applied to the compile invocation.
  iree_host_size_t config_binding_count;
  // Primary executable artifact byte length.
  iree_host_size_t artifact_byte_length;
  // Valid executable byte length inferred by the HAL executable cache.
  iree_host_size_t inferred_executable_byte_length;
  // Result diagnostic index, or IREE_HOST_SIZE_MAX when not
  // diagnostic-specific.
  iree_host_size_t diagnostic_index;
  // Loom diagnostic severity, or -1 when not diagnostic-specific.
  int32_t diagnostic_severity;
  // Queue affinity used to prepare the HAL executable.
  iree_hal_queue_affinity_t queue_affinity;
  // HAL executable caching mode used during preparation.
  iree_hal_executable_caching_mode_t caching_mode;
} id4_pipeline_kernel_diagnostic_t;

// Host-observed timing payload attached to timing diagnostic events.
typedef struct id4_pipeline_timing_diagnostic_t {
  // Monotonic start timestamp in nanoseconds.
  iree_time_t start_time_ns;
  // Monotonic end timestamp in nanoseconds.
  iree_time_t end_time_ns;
  // Elapsed duration in nanoseconds.
  iree_duration_t duration_ns;
} id4_pipeline_timing_diagnostic_t;

// Structured diagnostic event emitted by pipeline infrastructure.
typedef struct id4_pipeline_diagnostic_event_t {
  // Kind of event being emitted.
  id4_pipeline_diagnostic_event_kind_t kind;
  // Stage name associated with the event.
  iree_string_view_t stage_name;
  // Stable event key within the stage.
  iree_string_view_t key;
  // Human-readable event message.
  iree_string_view_t message;
  // Optional parameter slab payload valid only during the emit callback.
  const id4_pipeline_parameter_slab_diagnostic_t* parameter_slab;
  // Optional parameter loading payload valid only during the emit callback.
  const id4_pipeline_parameter_load_diagnostic_t* parameter_load;
  // Optional kernel payload valid only during the emit callback.
  const id4_pipeline_kernel_diagnostic_t* kernel;
  // Optional timing payload valid only during the emit callback.
  const id4_pipeline_timing_diagnostic_t* timing;
} id4_pipeline_diagnostic_event_t;

// Function pointer used to consume a diagnostic event.
typedef iree_status_t(IREE_API_PTR* id4_pipeline_diagnostics_emit_fn_t)(
    void* user_data, const id4_pipeline_diagnostic_event_t* event);

// Caller-owned diagnostic sink.
typedef struct id4_pipeline_diagnostics_sink_t {
  // Callback invoked for each event.
  id4_pipeline_diagnostics_emit_fn_t emit;
  // Caller-owned pointer passed to the callback.
  void* user_data;
} id4_pipeline_diagnostics_sink_t;

// Initializes |out_sink| to explicitly ignore all diagnostic events.
void id4_pipeline_diagnostics_sink_initialize_ignore(
    id4_pipeline_diagnostics_sink_t* out_sink);

// Validates that |sink| can receive diagnostic events for |usage_name|.
iree_status_t id4_pipeline_diagnostics_validate_sink(
    const id4_pipeline_diagnostics_sink_t* sink, iree_string_view_t usage_name);

// Emits a diagnostic event through a validated sink.
iree_status_t id4_pipeline_diagnostics_emit(
    id4_pipeline_diagnostics_sink_t* sink,
    const id4_pipeline_diagnostic_event_t* event);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_DIAGNOSTICS_H_
