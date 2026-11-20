// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured pipeline realization reports.

#ifndef LOOM_TARGET_REPORTING_PIPELINE_PLAN_H_
#define LOOM_TARGET_REPORTING_PIPELINE_PLAN_H_

#include "iree/base/api.h"
#include "loom/target/reporting/row_list.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_compile_report_t loom_target_compile_report_t;

// Available fact families. Missing families are absent in formatted reports;
// zero is reserved for a measured empty inventory.
enum loom_target_compile_report_pipeline_fact_bits_e {
  // Logical groups, channels, record protocols, and their assigned resources.
  LOOM_TARGET_COMPILE_REPORT_PIPELINE_FACT_CHANNELS = 1u << 0,
  // Distinct compiled images, independently of their physical load count.
  LOOM_TARGET_COMPILE_REPORT_PIPELINE_FACT_PROGRAMS = 1u << 1,
  // Physical memory occupancy, including stores with no resident worker.
  LOOM_TARGET_COMPILE_REPORT_PIPELINE_FACT_MEMORY = 1u << 2,
};
typedef uint8_t loom_target_compile_report_pipeline_facts_t;

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

// Exact period and trip count per entry to a repeated native packet range.
typedef struct loom_target_compile_report_packet_loop_t {
  // First repeated packet, in bytes from the callable entry.
  uint32_t byte_offset;
  // Physical issue packets per iteration, including empty packets.
  uint32_t packet_count;
  // Iterations on each entry to this range.
  uint32_t trip_count;
  // Issue packets consisting entirely of NOPs in one iteration.
  uint32_t empty_packet_count;
} loom_target_compile_report_packet_loop_t;

// A logical callable in a final worker image. Equal group and address on
// the same worker identify shared storage even when call counts differ.
typedef struct loom_target_compile_report_placed_function_t {
  // Module-owned spelling of the compiled function.
  iree_string_view_t name;
  // Logical worker index in the owning pipeline plan.
  uint32_t worker_index;
  // Worker placement in this pipeline.
  loom_target_compile_report_pipeline_placement_t placement;
  // Loaded image group within the worker, with zero denoting the resident
  // group.
  uint32_t group;
  // Worker-visible entry address while the group is loaded.
  uint32_t address;
  // Encoded bytes in the body alone, excluding inter-function padding.
  uint32_t code_byte_count;
  // Physical packets stored in the body once.
  uint32_t stored_packet_count;
  // Packets issued on a completed call, excluding all callee bodies.
  uint64_t packets_per_call;
  // Calls executed by one invocation of the worker entry.
  uint64_t calls_per_entry;
  // First repeated region in the enclosing plan's packet loop array.
  iree_host_size_t first_loop;
  // Number of consecutive loop records belonging to this function.
  uint32_t loop_count;
  // Whether the compiler accounted for the complete local control path.
  bool has_packets_per_call;
  // Whether all caller visits and their products are proven.
  bool has_calls_per_entry;
} loom_target_compile_report_placed_function_t;

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
  // Address envelope of one transfer task, including gaps between coordinates.
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
  // Fact families retained at the reporting boundary.
  loom_target_compile_report_pipeline_facts_t available_facts;
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
  // Distinct native images, before replication onto workers.
  uint32_t program_count;
  // Code bytes in distinct images, counted once per compiled function.
  uint64_t program_code_byte_count;
  // Number of physical backing stores in the report inventory.
  uint32_t memory_count;
  // Union of explicitly reserved bytes across physical backing stores.
  uint64_t reserved_storage_byte_count;
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

// One authored asynchronous copy between a root buffer and a worker. Counts
// are local to the containing construction, even when its caller is a loop.
typedef struct loom_target_compile_report_pipeline_copy_t {
  // Name of the pipeline that owns the external bindings.
  iree_string_view_t root;
  // Source construction containing the strand that issues this copy.
  iree_string_view_t phase;
  // Construction occurrence in this root; distinct call sites can bind the
  // same source phase differently or occur on alternative paths.
  iree_host_size_t occurrence;
  // Index of the root buffer argument.
  uint32_t binding;
  // True for a read from the external binding, false for a write to it.
  bool read;
  // True when participant count times per-worker visits is known.
  bool visits_known;
  // True when the worker control-flow analysis proves its own block visits.
  bool worker_visits_known;
  // Payload bytes per copy, or a negative value when runtime-dependent.
  int64_t bytes_per_copy;
  // Bytes in the contiguous inner axes of a known strided external copy.
  // Zero when the complete copy is contiguous or its order is unknown.
  uint64_t contiguous_run_bytes;
  // Copies from this site per invocation of its phase when visits_known.
  uint64_t visits;
  // Construction channel ordinal when the destination is freshly reserved.
  iree_host_size_t fresh_fifo;
  // Records in that FIFO, or zero without a same-block reservation proof.
  uint64_t fresh_fifo_capacity;
  // Times one participant visits this site when worker_visits_known.
  uint64_t visits_per_worker;
} loom_target_compile_report_pipeline_copy_t;

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
  // Number of independently advanced channel rings in the worker state.
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

// One physical backing store shared by reservations and loaded programs.
// Occupancy counts the union of resident ranges; the high-water extent also
// includes unoccupied alignment gaps. No logical channel ownership is inferred.
typedef struct loom_target_compile_report_pipeline_memory_row_t {
  // Dense index in the reported memory inventory.
  uint32_t memory_index;
  // Physical owner of the backing store.
  loom_target_compile_report_pipeline_placement_t placement;
  // Union of explicit lifetime-long reservations.
  uint32_t reserved_byte_count;
  // Private and read-only program data placed outside the reservations.
  uint32_t program_data_byte_count;
  // Total occupied bytes, excluding alignment gaps.
  uint32_t occupied_byte_count;
  // End of the highest resident byte relative to the backing-store base.
  uint32_t high_water_byte_count;
  // Complete addressable capacity of this backing store.
  uint32_t capacity_byte_count;
  // Largest occupied byte count in any bank of this backing store.
  uint32_t maximum_bank_storage_byte_count;
  // Capacity of one bank in this backing store.
  uint32_t bank_storage_capacity_byte_count;
} loom_target_compile_report_pipeline_memory_row_t;

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
  // Owned physical memory rows, present only in detailed reports.
  const loom_target_compile_report_pipeline_memory_row_t* memory_rows;
  // Number of entries in |memory_rows|.
  iree_host_size_t memory_row_count;
  // Final callable placements available independently of source-detail rows.
  const loom_target_compile_report_placed_function_t* functions;
  // Number of logical callable placements.
  iree_host_size_t function_count;
  // Repeated native ranges indexed by the placed functions.
  const loom_target_compile_report_packet_loop_t* packet_loops;
  // Total number of repeated native ranges across all placements.
  iree_host_size_t packet_loop_count;
} loom_target_compile_report_pipeline_plan_t;

// Releases every plan owned by |list| and resets it to empty.
void loom_target_compile_report_pipeline_plan_list_deinitialize(
    loom_target_compile_report_row_list_t* list,
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
