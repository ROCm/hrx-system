// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBHRX_SRC_LIBHRX_COMMAND_RECORDING_H_
#define LIBHRX_SRC_LIBHRX_COMMAND_RECORDING_H_

#include "graph_analysis.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Producer-resolved commands stored inline in frontend graph nodes. Resources
// are borrowed under each producer's lifetime contract; argument storage is
// graph-owned. Retaining HAL command buffers capture their usual references.
// These records add no owner, allocation, or replay-time translation.
enum hrx_command_kind_e {
  HRX_COMMAND_DISPATCH = 0,
  HRX_COMMAND_COPY,
  HRX_COMMAND_FILL,
  HRX_COMMAND_ATOMIC_BATCH,
};

typedef struct hrx_command_t {
  // Resolved operation kind; frontend node classification stays separate.
  uint8_t kind;
} hrx_command_t;

typedef struct hrx_command_dispatch_t {
  // Common first member used by the recorder to select a typed payload.
  hrx_command_t command;
  // Nonzero when inline extended geometry follows this structure.
  uint8_t extended_geometry;
  // Export selected from the executable.
  uint32_t export_ordinal;
  // Already translated HAL dispatch flags; no truncation.
  iree_hal_dispatch_flags_t flags;
  // Executable retained by the existing graph/module and recorded commands.
  iree_hal_executable_t* executable;
  // Graph-owned fixed argument bytes; embedded pointers remain borrowed.
  iree_const_byte_span_t constants;
  // Graph-owned explicit binding references.
  iree_hal_buffer_ref_list_t bindings;
  // Number of workgroups in each dimension.
  uint32_t workgroup_count[3];
  // Number of workitems in each workgroup dimension.
  uint32_t workgroup_size[3];
} hrx_command_dispatch_t;

typedef struct hrx_command_dispatch_extended_t {
  // Ordinary dispatch prefix.
  hrx_command_dispatch_t dispatch;
  // Dynamic workgroup-local storage in bytes.
  uint32_t dynamic_workgroup_local_memory;
  // Exact workitem extent, with zeroes for complete workgroups.
  uint32_t workitem_count[3];
} hrx_command_dispatch_extended_t;

typedef struct hrx_command_copy_t {
  // Common first member used by the recorder to select a typed payload.
  hrx_command_t command;
  // Nonzero when inline row/slice geometry follows this structure.
  uint8_t strided;
  // Publication required for a host-visible destination before signaling.
  uint8_t release_to_host;
  // Already translated HAL copy flags; no truncation.
  iree_hal_copy_flags_t flags;
  // Resolved source HAL allocation.
  iree_hal_buffer_t* source_buffer;
  // Starting byte offset in the source allocation.
  iree_device_size_t source_offset;
  // Resolved destination HAL allocation.
  iree_hal_buffer_t* target_buffer;
  // Starting byte offset in the destination allocation.
  iree_device_size_t target_offset;
  // Contiguous copy length, or row width when strided.
  iree_device_size_t length;
} hrx_command_copy_t;

typedef struct hrx_command_copy_strided_t {
  // Ordinary copy prefix; each row is one contiguous operation.
  hrx_command_copy_t copy;
  // Number of rows in each slice.
  iree_device_size_t row_count;
  // Number of slices.
  iree_device_size_t slice_count;
  // Source byte displacement between rows.
  iree_device_size_t source_row_stride;
  // Destination byte displacement between rows.
  iree_device_size_t target_row_stride;
  // Source byte displacement between slices, established at construction.
  iree_device_size_t source_slice_stride;
  // Destination byte displacement between slices, established at construction.
  iree_device_size_t target_slice_stride;
} hrx_command_copy_strided_t;

typedef struct hrx_command_fill_t {
  // Common first member used by the recorder to select a typed payload.
  hrx_command_t command;
  // Size of the pattern in bytes, validated at construction.
  uint8_t pattern_size;
  // Repeated pattern value.
  uint32_t pattern;
  // Already translated HAL fill flags; no truncation.
  iree_hal_fill_flags_t flags;
  // Exact destination range, including the resolved byte length.
  iree_hal_buffer_ref_t target;
} hrx_command_fill_t;

// Operation represented by one atomic batch entry.
typedef enum hrx_atomic_operation_kind_e {
  HRX_ATOMIC_OPERATION_WAIT = 0,
  HRX_ATOMIC_OPERATION_STORE = 1,
  HRX_ATOMIC_OPERATION_UPDATE = 2,
} hrx_atomic_operation_kind_t;

// Parameters for one atomic batch entry.
typedef union hrx_atomic_operation_params_u {
  // Parameters used by HRX_ATOMIC_OPERATION_WAIT.
  iree_hal_atomic_wait_params_t wait;
  // Parameters used by HRX_ATOMIC_OPERATION_STORE.
  iree_hal_atomic_store_params_t store;
  // Parameters used by HRX_ATOMIC_OPERATION_UPDATE.
  iree_hal_atomic_rmw_params_t update;
} hrx_atomic_operation_params_t;

// Fully resolved operation submitted at one position in an atomic batch.
typedef struct hrx_atomic_operation_t {
  // Operation performed by this entry.
  hrx_atomic_operation_kind_t kind;
  // HAL allocation containing the naturally aligned target cell.
  iree_hal_buffer_t* target_buffer;
  // Byte offset of the target cell in |target_buffer|.
  iree_device_size_t target_offset;
  // Parameters selected by |kind|.
  hrx_atomic_operation_params_t params;
} hrx_atomic_operation_t;

// One ordered batch with the same execution-stage policy as its producer.
typedef struct hrx_command_atomic_batch_t {
  // Common first member used by the recorder to select this payload.
  hrx_command_t command;
  // Number of resolved entries in |operations|.
  iree_host_size_t operation_count;
  // Frontend-owned validated operations.
  hrx_atomic_operation_t* operations;
  // Operations preceding the first atomic in this batch.
  iree_hal_execution_stage_t source_stage;
  // Operations ordered after each atomic in this batch.
  iree_hal_execution_stage_t target_stage;
} hrx_command_atomic_batch_t;

// Constructs HAL geometry from a resolved dispatch. Only producers with inline
// extended storage set extended_geometry; ordinary native dispatches do not
// reserve it.
iree_hal_dispatch_config_t hrx_command_dispatch_config(
    const hrx_command_dispatch_t* dispatch);

// Records a validated atomic batch. The first operation uses |source_stage|;
// subsequent entries follow the preceding atomic operation. Empty batches are
// harmless. Validation and queue admission belong to the caller.
iree_status_t hrx_command_record_atomics(
    iree_hal_command_buffer_t* command_buffer, iree_host_size_t operation_count,
    const hrx_atomic_operation_t* operations,
    iree_hal_execution_stage_t source_stage,
    iree_hal_execution_stage_t target_stage);

// Records one scheduled workstream into an already open command buffer. The
// source table contains resolved commands for recordable nodes and opaque
// frontend payloads otherwise. Entry/exit synchronization and debug groups stay
// with the frontend; this walk emits only the analyzed dependency barriers and
// accumulates the publication requirements of its commands.
iree_status_t hrx_graph_record_commands(
    const hrx_graph_schedule_t* schedule, uint32_t node_start_index,
    uint32_t node_count, uint8_t stream_index,
    const iree_hal_barrier_t* dependency_barrier,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_barrier_flags_t* out_release_flags);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_LIBHRX_COMMAND_RECORDING_H_
