// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured pipeline realization reports.

#ifndef LOOM_TARGET_REPORTING_PIPELINE_PLAN_H_
#define LOOM_TARGET_REPORTING_PIPELINE_PLAN_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_compile_report_t loom_target_compile_report_t;

// Pipeline worker behavior and available physical-plan facts.
enum loom_target_compile_report_pipeline_worker_flag_bits_e {
  // The worker remains resident and repeatedly processes activations.
  LOOM_TARGET_COMPILE_REPORT_PIPELINE_WORKER_RESIDENT = 1u << 0,
  // The worker contains a finite inner loop reducing several input records.
  LOOM_TARGET_COMPILE_REPORT_PIPELINE_WORKER_FOLDED = 1u << 1,
  // The worker has an exact physical placement.
  LOOM_TARGET_COMPILE_REPORT_PIPELINE_WORKER_PLACED = 1u << 2,
};
typedef uint8_t loom_target_compile_report_pipeline_worker_flags_t;

// Entity owning one end of a realized pipeline channel.
typedef enum loom_target_compile_report_pipeline_endpoint_owner_e {
  // The endpoint is not associated with a reportable owner.
  LOOM_TARGET_COMPILE_REPORT_PIPELINE_ENDPOINT_OWNER_NONE = 0,
  // The endpoint is an external artifact binding.
  LOOM_TARGET_COMPILE_REPORT_PIPELINE_ENDPOINT_OWNER_BINDING = 1,
  // The endpoint is a resident or scheduled worker.
  LOOM_TARGET_COMPILE_REPORT_PIPELINE_ENDPOINT_OWNER_WORKER = 2,
} loom_target_compile_report_pipeline_endpoint_owner_t;

// Physical placement selected for one pipeline worker.
typedef struct loom_target_compile_report_pipeline_placement_t {
  // Number of valid leading coordinates.
  uint8_t rank;
  // First physical coordinate, such as an AIE column.
  uint16_t x;
  // Second physical coordinate, such as an AIE row.
  uint16_t y;
  // Third physical coordinate when the target has one.
  uint16_t z;
} loom_target_compile_report_pipeline_placement_t;

// One endpoint of a realized pipeline channel.
typedef struct loom_target_compile_report_pipeline_endpoint_t {
  // Kind of entity owning the endpoint.
  loom_target_compile_report_pipeline_endpoint_owner_t owner;
  // Index of the binding or worker selected by |owner|.
  uint32_t owner_index;
  // Port ordinal in the owner ABI.
  uint32_t port;
} loom_target_compile_report_pipeline_endpoint_t;

// Fixed physical storage schema consumed through one pipeline channel.
typedef struct loom_target_compile_report_pipeline_storage_t {
  // Canonical schema family name, or empty for ordinary dense records.
  iree_string_view_t schema_name;
  // Resolved address-layout name.
  iree_string_view_t address_layout;
  // Physical transform selected between external and channel storage.
  iree_string_view_t transform;
  // Logical elements represented by one schema record.
  uint32_t schema_logical_element_count;
  // Physical bytes occupied by one schema record.
  uint32_t schema_record_byte_count;
  // Required byte alignment of one schema record.
  uint32_t schema_required_alignment;
  // Schema records carried by one channel record.
  uint32_t schema_records_per_channel_record;
  // Additional bytes introduced by the selected transform.
  uint64_t transform_extra_byte_count;
} loom_target_compile_report_pipeline_storage_t;

// External transfer selected for one pipeline channel.
typedef struct loom_target_compile_report_pipeline_transfer_t {
  // External binding ordinal, or UINT32_MAX when the channel is internal.
  uint32_t binding_ordinal;
  // Selected partition lane, or zero when unpartitioned.
  uint32_t partition_lane;
  // Number of partitions, or one when unpartitioned.
  uint32_t partition_lane_count;
  // Byte offset added to the external binding base.
  uint64_t binding_byte_offset;
  // Contiguous bytes touched by one transfer task.
  uint64_t binding_span_byte_count;
  // Bytes transferred by one task execution.
  uint32_t task_byte_count;
  // Number of task executions per activation.
  uint32_t task_repeat_count;
  // Total external bytes transferred per activation.
  uint64_t activation_byte_count;
} loom_target_compile_report_pipeline_transfer_t;

// Aggregate facts for one realized pipeline root.
typedef struct loom_target_compile_report_pipeline_plan_summary_t {
  // Public pipeline or artifact root name.
  iree_string_view_t root_name;
  // Selected realization class, such as "spatial-program".
  iree_string_view_t realization;
  // Number of logical worker groups.
  uint32_t group_count;
  // Number of realized workers.
  uint32_t worker_count;
  // Number of external artifact bindings.
  uint32_t binding_count;
  // Number of realized channels.
  uint32_t channel_count;
  // Number of physical channel ring slots.
  uint32_t channel_slot_count;
  // Number of target hardware locks allocated to channels.
  uint32_t hardware_lock_count;
  // Number of target DMA channels allocated to channels.
  uint32_t dma_channel_count;
  // Number of DMA buffer descriptors allocated to channels.
  uint32_t dma_buffer_descriptor_count;
  // Number of physical route connections programmed for channels.
  uint32_t route_count;
  // Total native code bytes across realized workers.
  uint64_t worker_code_byte_count;
  // Largest native worker program.
  uint32_t maximum_worker_code_byte_count;
  // Smallest remaining program-memory headroom among workers.
  uint32_t minimum_worker_code_headroom_byte_count;
  // Total worker-private local storage allocated across physical tiles.
  uint64_t worker_storage_byte_count;
  // Total channel storage allocated across physical tiles.
  uint64_t channel_storage_byte_count;
  // Largest local-memory high-water mark on one physical tile.
  uint32_t maximum_tile_local_memory_byte_count;
  // Smallest remaining local-memory headroom among occupied tiles.
  uint32_t minimum_tile_local_memory_headroom_byte_count;
  // Largest occupied local-memory bank on one physical tile.
  uint32_t maximum_bank_storage_byte_count;
  // Storage capacity of the bank reported by
  // |maximum_bank_storage_byte_count|.
  uint32_t bank_storage_capacity_byte_count;
  // External DMA bytes moved per pipeline activation.
  uint64_t external_dma_byte_count;
  // Routed on-array DMA bytes moved per pipeline activation.
  uint64_t routed_dma_byte_count;
} loom_target_compile_report_pipeline_plan_summary_t;

// One realized pipeline worker.
typedef struct loom_target_compile_report_pipeline_worker_row_t {
  // Worker index used by channel endpoints.
  uint32_t worker_index;
  // Logical group containing the worker.
  uint32_t group_index;
  // Lane ordinal within the logical group.
  uint32_t lane;
  // Worker behavior and available-fact flags.
  loom_target_compile_report_pipeline_worker_flags_t flags;
  // Function implementing one worker firing.
  iree_string_view_t entry_name;
  // Exact target placement when PLACED is set.
  loom_target_compile_report_pipeline_placement_t placement;
  // Number of input channels carrying the worker state machine.
  uint32_t input_channel_count;
  // Number of output channels carrying the worker state machine.
  uint32_t output_channel_count;
  // Number of ring positions retained in the worker state machine.
  uint32_t ring_state_count;
  // Synchronized input record firings per port and pipeline activation.
  uint32_t input_record_count;
  // Synchronized output record firings per port and pipeline activation.
  uint32_t output_record_count;
  // Finite inner-loop trip count, or zero when no inner loop was generated.
  uint32_t finite_loop_trip_count;
  // Combining operation applied by a finite record fold, or empty otherwise.
  iree_string_view_t fold_kind;
  // Final native code bytes in the worker program.
  uint32_t code_byte_count;
  // Target program-memory capacity available to the worker.
  uint32_t code_capacity_byte_count;
  // Worker-private local storage bytes on the physical tile.
  uint32_t worker_storage_byte_count;
  // Channel storage bytes owned by the physical tile.
  uint32_t channel_storage_byte_count;
  // Local-memory high-water mark on the physical tile.
  uint32_t local_memory_byte_count;
  // Target local-memory capacity available on the physical tile.
  uint32_t local_memory_capacity_byte_count;
  // Largest occupied local-memory bank on the physical tile.
  uint32_t maximum_bank_storage_byte_count;
  // Storage capacity of one local-memory bank.
  uint32_t bank_storage_capacity_byte_count;
} loom_target_compile_report_pipeline_worker_row_t;

// One realized pipeline channel.
typedef struct loom_target_compile_report_pipeline_channel_row_t {
  // Channel index referenced by target resource plans.
  uint32_t channel_index;
  // Physical transport selected by the target planner.
  iree_string_view_t transport;
  // Sending endpoint.
  loom_target_compile_report_pipeline_endpoint_t sender;
  // Receiving endpoint.
  loom_target_compile_report_pipeline_endpoint_t receiver;
  // Number of records retained in the bounded channel ring.
  uint32_t capacity;
  // Number of ordered records transferred per pipeline activation.
  uint32_t record_count;
  // Number of bytes in one channel record.
  uint32_t record_byte_count;
  // Total logical bytes carried per pipeline activation.
  uint64_t activation_byte_count;
  // Physical local-memory bytes allocated to the channel ring.
  uint64_t local_storage_byte_count;
  // Number of target hardware locks allocated to the channel.
  uint32_t hardware_lock_count;
  // Number of target DMA channels allocated to the channel.
  uint32_t dma_channel_count;
  // Number of target DMA buffer descriptors allocated to the channel.
  uint32_t dma_buffer_descriptor_count;
  // Number of physical route connections programmed for the channel.
  uint32_t route_count;
  // Exact encoded storage carried by an external endpoint.
  loom_target_compile_report_pipeline_storage_t storage;
  // Exact external transfer, absent when binding_ordinal is UINT32_MAX.
  loom_target_compile_report_pipeline_transfer_t transfer;
} loom_target_compile_report_pipeline_channel_row_t;

// Owned pipeline realization snapshot in one compile report.
//
// Strings remain borrowed under the same lifetime contract as the enclosing
// report. Row arrays are contiguous and owned by the report allocator.
typedef struct loom_target_compile_report_pipeline_plan_t {
  // Aggregate physical realization facts.
  loom_target_compile_report_pipeline_plan_summary_t summary;
  // Owned realized worker rows, present only in detailed reports.
  const loom_target_compile_report_pipeline_worker_row_t* worker_rows;
  // Number of entries in |worker_rows|.
  iree_host_size_t worker_row_count;
  // Owned realized channel rows, present only in detailed reports.
  const loom_target_compile_report_pipeline_channel_row_t* channel_rows;
  // Number of entries in |channel_rows|.
  iree_host_size_t channel_row_count;
} loom_target_compile_report_pipeline_plan_t;

// Releases row storage owned by |plan| and resets it to zero.
void loom_target_compile_report_pipeline_plan_deinitialize(
    loom_target_compile_report_pipeline_plan_t* plan,
    iree_allocator_t host_allocator);

// Initializes |out_target| as an owned copy of |source|.
iree_status_t loom_target_compile_report_pipeline_plan_clone(
    const loom_target_compile_report_pipeline_plan_t* source,
    loom_target_compile_report_pipeline_plan_t* out_target,
    iree_allocator_t host_allocator);

// Records one pipeline realization, copying detail rows only when requested.
iree_status_t loom_target_compile_report_record_pipeline_plan(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_pipeline_plan_t* plan);

// Returns a stable report spelling for |owner|.
iree_string_view_t loom_target_compile_report_pipeline_endpoint_owner_name(
    loom_target_compile_report_pipeline_endpoint_owner_t owner);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_REPORTING_PIPELINE_PLAN_H_
