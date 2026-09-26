// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Resident AIE2P array topology extraction and physical NPU2 planning.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_PLAN_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/function_requirements.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/ops/combining.h"
#include "loom/target/arch/amd/xdna/array/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_memory_access_map_t loom_low_memory_access_map_t;
typedef struct loom_target_facts_t loom_target_facts_t;
typedef struct loom_aie2p_array_leaf_t loom_aie2p_array_leaf_t;

// Access permitted through one external array-program binding.
typedef enum loom_aie2p_array_binding_access_e {
  LOOM_AIE2P_ARRAY_BINDING_ACCESS_READ = 1,
  LOOM_AIE2P_ARRAY_BINDING_ACCESS_WRITE = 2,
  LOOM_AIE2P_ARRAY_BINDING_ACCESS_READ_WRITE = 3,
} loom_aie2p_array_binding_access_t;

// Kind of entity owning a channel endpoint.
typedef enum loom_aie2p_array_endpoint_owner_kind_e {
  LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_BINDING = 1,
  LOOM_AIE2P_ARRAY_ENDPOINT_OWNER_WORKER = 2,
} loom_aie2p_array_endpoint_owner_kind_t;

// Data-flow direction represented by an endpoint.
typedef enum loom_aie2p_array_endpoint_direction_e {
  LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_SEND = 1,
  LOOM_AIE2P_ARRAY_ENDPOINT_DIRECTION_RECEIVE = 2,
} loom_aie2p_array_endpoint_direction_t;

// Physical transport selected for one logical channel.
typedef enum loom_aie2p_array_channel_transport_e {
  // External binding traffic routed through shim and compute DMA engines.
  LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_EXTERNAL_DMA = 1,
  // Adjacent workers communicating through a shared local-memory window.
  LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_NEIGHBOR_MEMORY = 2,
  // Non-adjacent workers communicating through compute DMA and stream routes.
  LOOM_AIE2P_ARRAY_CHANNEL_TRANSPORT_ROUTED_DMA = 3,
} loom_aie2p_array_channel_transport_t;

// DMA transfer direction relative to local memory.
typedef enum loom_aie2p_array_dma_direction_e {
  LOOM_AIE2P_ARRAY_DMA_DIRECTION_MEMORY_TO_STREAM = 1,
  LOOM_AIE2P_ARRAY_DMA_DIRECTION_STREAM_TO_MEMORY = 2,
} loom_aie2p_array_dma_direction_t;

typedef uint8_t loom_aie2p_array_dma_flags_t;
enum loom_aie2p_array_dma_flag_bits_e {
  // The DMA engine resides on a shim tile instead of a compute tile.
  LOOM_AIE2P_ARRAY_DMA_FLAG_SHIM = 1u << 0,
  // This row owns core and DMA lifecycle programming for a worker-free tile.
  LOOM_AIE2P_ARRAY_DMA_FLAG_SERVICE_TILE_LIFECYCLE = 1u << 1,
};

// Physical switch container programmed by a route connection.
typedef enum loom_aie2p_array_switch_kind_e {
  LOOM_AIE2P_ARRAY_SWITCH_KIND_STREAM_SWITCH = 1,
  LOOM_AIE2P_ARRAY_SWITCH_KIND_SHIM_MUX = 2,
} loom_aie2p_array_switch_kind_t;

// Logical group whose exact lane count is known during array planning.
typedef struct loom_aie2p_array_group_t {
  // SSA result naming the group in the array function.
  loom_value_id_t value_id;
  // Exact number of resident worker lanes in the group.
  uint32_t lane_count;
} loom_aie2p_array_group_t;

// One external array-program binding.
typedef struct loom_aie2p_array_binding_t {
  // SSA result naming the binding in the array function.
  loom_value_id_t value_id;
  // Array-program ABI binding ordinal.
  uint32_t ordinal;
  // Permitted access through the binding.
  loom_aie2p_array_binding_access_t access;
} loom_aie2p_array_binding_t;

// One resident worker instance.
typedef struct loom_aie2p_array_worker_t {
  // SSA result naming the worker in the array function.
  loom_value_id_t value_id;
  // Index of the logical group containing this worker.
  uint32_t group_index;
  // Exact lane ordinal within the logical group.
  uint32_t lane;
  // Core function executed by the worker.
  loom_symbol_ref_t entry;
  // Source leaf selected for the worker entry during topology extraction.
  const loom_aie2p_array_leaf_t* leaf;
  // Number of source records folded into one output, or zero when recordwise.
  uint32_t fold_record_count;
  // Callable output port carrying the folded result.
  uint32_t fold_output_port;
  // Number of contiguous callable output ports folded independently.
  uint32_t fold_output_count;
  // Elementwise combining operation used by the temporal fold.
  loom_combining_kind_t fold_kind;
  // Floating-point permissions applied by the temporal fold.
  uint8_t fold_fast_math_flags;
  // Number of direct endpoints participating in at least one channel.
  uint32_t active_endpoint_count;
  // Physical compute tile selected by the authored placement constraint.
  loom_xdna_tile_coordinate_t coordinate;
} loom_aie2p_array_worker_t;

// One typed sender or receiver in the resident topology.
typedef struct loom_aie2p_array_endpoint_t {
  // SSA result naming the endpoint in the array function.
  loom_value_id_t value_id;
  // Sender or receiver direction.
  loom_aie2p_array_endpoint_direction_t direction;
  // Kind of entity referenced by owner_index.
  loom_aie2p_array_endpoint_owner_kind_t owner_kind;
  // Binding or worker index selected by owner_kind.
  uint32_t owner_index;
  // Port ordinal in the owner ABI.
  uint32_t port;
  // First logical channel using this endpoint, or UINT32_MAX when unused.
  uint32_t first_channel_index;
  // Number of logical channels using this endpoint.
  uint32_t channel_use_count;
  // Matched leaf resource ordinal for a worker endpoint, or UINT32_MAX when
  // its pointer is unused by the leaf. Unused for bindings.
  uint32_t worker_resource_ordinal;
  // Typed tile value transported through this endpoint.
  loom_type_t message_type;
  // Exact byte offset from the runtime binding base.
  uint64_t binding_byte_offset;
  // Raw binding endpoint wrapped by this view, or UINT32_MAX when unwrapped.
  uint32_t binding_view_source_endpoint_index;
  // Records selected from the source by an active binding view, or zero when
  // this endpoint is direct or unused.
  uint32_t binding_view_record_count;
  // Selected partition lane, or zero for an unpartitioned binding view.
  uint32_t partition_lane;
  // Number of source partitions, or one for an unpartitioned binding view.
  uint32_t partition_lane_count;
  // Whether the binding view selects one leading-dimension lane.
  bool binding_view_partitioned;
} loom_aie2p_array_endpoint_t;

// One typed bounded channel connecting a sender to a receiver.
typedef struct loom_aie2p_array_channel_t {
  // SSA result naming the channel in the array function.
  loom_value_id_t value_id;
  // Index of the sending endpoint.
  uint32_t sender_endpoint_index;
  // Index of the receiving endpoint.
  uint32_t receiver_endpoint_index;
  // Channel owning shared producer storage, DMA, and locks. Equal to this
  // channel's index for a canonical sender or a single consumer. Worker
  // fanout shares producer storage, DMA, and locks. Binding fanout shares the
  // host-facing shim DMA and runtime binding patch.
  uint32_t source_channel_index;
  // First record in channel_slots for this channel's contiguous ring.
  uint32_t first_channel_slot;
  // DMA row serving the sending endpoint, or UINT32_MAX for memory transport.
  uint32_t sender_dma_index;
  // Number of records held by the channel ring.
  uint32_t capacity;
  // Number of ordered records transferred per activation.
  uint32_t record_count;
  // Byte length of one topology-admitted tile record.
  uint32_t record_byte_length;
  // Transfer-length field value admitted for DMA-backed transports. Neighbor
  // memory transport does not consume this value.
  uint32_t encoded_dma_record_length;
  // Physical transport selected by planning.
  loom_aie2p_array_channel_transport_t transport;
} loom_aie2p_array_channel_t;

// Declared requirements associated with a worker entry symbol.
struct loom_aie2p_array_leaf_t {
  // Module-local core entry symbol.
  loom_symbol_ref_t entry;
  // Source core function selected by entry.
  const loom_op_t* function_op;
  // Immutable invocation target facts selected at the source boundary.
  const loom_target_facts_t* function_target_facts;
  // Optional source memory proofs translated with the resident clone.
  const loom_low_memory_access_map_t* memory_accesses;
  // Source imports and storage retained before resident worker construction.
  loom_low_function_requirements_t requirements;
};

// Contiguous equal-width transfers between an output and private fold state.
typedef struct loom_aie2p_array_fold_span_t {
  // Output ordinal relative to the worker's contiguous folded output range.
  uint32_t output_index;
  // Byte offset from the current output ring record.
  uint32_t output_byte_offset;
  // Byte offset from the worker's private fold allocation.
  uint32_t state_byte_offset;
  // One scalar F32 or up to four complete 64-byte accumulator lanes.
  uint32_t byte_length;
  // Positive number of adjacent fragments of byte_length bytes.
  uint32_t repeat_count;
} loom_aie2p_array_fold_span_t;

// Private local state for folds wider than the register-backed realization.
// Source resource imports keep pointing at their ordinary channel rings.
typedef struct loom_aie2p_array_fold_state_plan_t {
  // Byte offset in the worker tile's local data memory.
  uint32_t owner_offset;
  // Worker-visible address of the private allocation.
  uint32_t load_address;
  // Allocation extent, or zero when the worker uses register-backed folding.
  uint32_t byte_length;
  // Output-ordered contiguous transfer spans retained by physical planning.
  const loom_aie2p_array_fold_span_t* spans;
  // Number of fixed-width runs and trailing fragments in the allocation.
  uint32_t span_count;
} loom_aie2p_array_fold_state_plan_t;

// Final placement of one resident worker before native compilation.
typedef struct loom_aie2p_array_worker_plan_t {
  // Index of the logical worker represented by this placement.
  uint32_t worker_index;
  // Physical compute tile executing the worker.
  loom_xdna_tile_coordinate_t coordinate;
  // First worker_ports row and worker_resource_ports entry for this worker.
  uint32_t first_port;
  // Number of contiguous ports, ordered by their first channel binding.
  uint32_t port_count;
  // Number of resident ring positions consumed by leaf or generated code.
  uint32_t resident_state_count;
  // Private ordered-fold state reserved before channel rings are placed.
  loom_aie2p_array_fold_state_plan_t fold_state;
} loom_aie2p_array_worker_plan_t;

// Final local-data placement for one compiled worker storage domain.
typedef struct loom_aie2p_array_worker_storage_plan_t {
  // Index of the logical worker owning the storage domain.
  uint32_t worker_index;
  // Structural Low storage space represented by the domain.
  loom_storage_space_t storage_space;
  // Byte offset in the worker tile's local data memory.
  uint32_t owner_offset;
  // Worker-visible load address used to relocate the leaf contribution.
  uint32_t load_address;
  // Number of bytes occupied by the storage domain.
  uint32_t byte_length;
} loom_aie2p_array_worker_storage_plan_t;

// Worker ABI port bound to one planned channel ring.
typedef struct loom_aie2p_array_worker_port_plan_t {
  // Index of the logical worker owning the port.
  uint32_t worker_index;
  // Worker ABI port index selected by the topology endpoint.
  uint32_t port;
  // Sender or receiver direction of the worker endpoint.
  loom_aie2p_array_endpoint_direction_t direction;
  // Index of the logical channel bound to the port.
  uint32_t channel_index;
  // First row of the credit/ready lock pair used by the resident worker.
  uint32_t credit_lock_index;
  // Worker-local resident state ordinal carrying the ring position, or
  // UINT32_MAX when neither the leaf nor generated code consumes its address.
  uint32_t resident_state_ordinal;
} loom_aie2p_array_worker_port_plan_t;

// One endpoint's physical view of a channel record.
typedef struct loom_aie2p_array_channel_storage_plan_t {
  // Compute tile owning the record storage, or an invalid coordinate when the
  // endpoint is an external binding.
  loom_xdna_tile_coordinate_t owner;
  // Byte offset in the owner's local data memory.
  uint32_t owner_offset;
  // Endpoint-visible load address, or UINT32_MAX for an external binding.
  uint32_t load_address;
} loom_aie2p_array_channel_storage_plan_t;

// One logical record slot and its endpoint-local physical storage.
typedef struct loom_aie2p_array_channel_slot_t {
  // Index of the logical channel owning this record slot.
  uint32_t channel_index;
  // Ring position within the logical channel.
  uint32_t slot;
  // Number of bytes occupied by each materialized endpoint record.
  uint32_t byte_length;
  // Producer-side storage view.
  loom_aie2p_array_channel_storage_plan_t sender_storage;
  // Consumer-side storage view. Neighbor-memory channels alias sender storage.
  loom_aie2p_array_channel_storage_plan_t receiver_storage;
} loom_aie2p_array_channel_slot_t;

// One hardware lock allocated to channel ring synchronization.
typedef struct loom_aie2p_array_lock_plan_t {
  // Index of the logical channel synchronized by this lock.
  uint32_t channel_index;
  // Physical tile containing the lock.
  loom_xdna_tile_coordinate_t coordinate;
  // Tile-local hardware lock ordinal.
  uint8_t lock_id;
  // Initial signed lock value.
  int8_t initial_value;
  // Logical endpoint whose local ring is synchronized by the lock.
  loom_aie2p_array_endpoint_direction_t ring_endpoint_direction;
  // Zero for the producer-credit lock and one for the consumer-ready lock.
  uint8_t consumer_ready;
} loom_aie2p_array_lock_plan_t;

// One physical DMA channel and its contiguous buffer-descriptor range.
typedef struct loom_aie2p_array_dma_plan_t {
  // Index of the logical channel transported by this DMA channel.
  uint32_t channel_index;
  // Physical tile containing the DMA engine.
  loom_xdna_tile_coordinate_t coordinate;
  // Transfer direction relative to local memory.
  loom_aie2p_array_dma_direction_t direction;
  // Direction-local DMA channel ordinal.
  uint8_t dma_channel;
  // Engine placement and lifecycle ownership flags.
  loom_aie2p_array_dma_flags_t flags;
  // First tile-local buffer descriptor allocated to the channel.
  uint16_t buffer_descriptor_start;
  // Number of buffer descriptors allocated to the ring.
  uint16_t buffer_descriptor_count;
  // First row of the credit/ready lock pair, or UINT32_MAX for a shim DMA.
  uint32_t credit_lock_index;
} loom_aie2p_array_dma_plan_t;

// One programmed source-to-destination stream-switch connection.
typedef struct loom_aie2p_array_route_plan_t {
  // First logical channel that allocated this physical connection. Sibling
  // channels with the same canonical source may reuse its route prefix.
  uint32_t channel_index;
  // Physical tile containing the switch container.
  loom_xdna_tile_coordinate_t coordinate;
  // Stream switch or shim mux container.
  loom_aie2p_array_switch_kind_t switch_kind;
  // Architectural source port class.
  loom_xdna_stream_port_t source_port;
  // Source channel ordinal within source_port.
  uint8_t source_channel;
  // Architectural destination port class.
  loom_xdna_stream_port_t destination_port;
  // Destination channel ordinal within destination_port.
  uint8_t destination_channel;
} loom_aie2p_array_route_plan_t;

// Packet resources owned by the completion endpoint on one used shim.
// Both egress DMA engines on the shim share this exact-match route. Circuit
// data routes use separate stream ports and do not consume packet resources.
typedef struct loom_aie2p_array_completion_route_t {
  // Shim owning the completion endpoint and its packet resources.
  loom_xdna_tile_coordinate_t coordinate;
  // Physical TileControl slave port ordinal.
  uint8_t source_ordinal;
  // Physical firmware-facing master port ordinal.
  uint8_t destination_ordinal;
  // Selected exact-match packet identity, also written to each DMA controller.
  uint8_t packet_id;
  // Packet arbiter assigned to the destination master port.
  uint8_t arbiter;
  // Master-select group within the selected arbiter.
  uint8_t master_select;
  // Packet matching rule within the source slave port.
  uint8_t rule_slot;
} loom_aie2p_array_completion_route_t;

enum {
  // AIE2P shim DMA buffer descriptors expose three address dimensions.
  LOOM_AIE2P_ARRAY_BINDING_DMA_DIMENSION_COUNT = 3,
};

// One innermost-first shim DMA address dimension.
typedef struct loom_aie2p_array_binding_dma_dimension_t {
  // Actual address-generation step in family-native address units.
  uint32_t step_size;
  // Actual number of addresses visited before the next dimension advances, or
  // zero for the outermost dimension.
  uint32_t wrap;
} loom_aie2p_array_binding_dma_dimension_t;

// Runtime patch record connecting an ABI binding to one shim DMA program.
typedef struct loom_aie2p_array_binding_plan_t {
  // Index of the logical binding represented by this patch.
  uint32_t binding_index;
  // Index of the logical channel using the binding.
  uint32_t channel_index;
  // Exact shim DMA row patched and queued by this binding.
  uint32_t dma_index;
  // Selected partition lane in the external tile, or zero when direct.
  uint32_t partition_lane;
  // External tile partition count, or one when direct.
  uint32_t partition_lane_count;
  // Exact completion route row, or UINT32_MAX for an ingress binding.
  uint32_t completion_route_index;
  // Byte offset added to the runtime binding base address.
  uint64_t binding_byte_offset;
  // Physical byte span reachable from binding_byte_offset.
  uint64_t binding_span_byte_length;
  // Number of bytes transferred by each shim DMA task execution.
  uint32_t transfer_byte_length;
  // Actual number of executions requested by the queued shim DMA task.
  uint16_t task_repeat_count;
  // Number of populated multidimensional address-generator rows.
  uint8_t dma_dimension_count;
  // Innermost-first multidimensional address-generator rows.
  loom_aie2p_array_binding_dma_dimension_t
      dma_dimensions[LOOM_AIE2P_ARRAY_BINDING_DMA_DIMENSION_COUNT];
} loom_aie2p_array_binding_plan_t;

// Complete arena-owned logical topology and deterministic NPU2 physical plan.
//
// The plan borrows the immutable Loom module and detached leaf contributions.
// All arrays are allocated from the caller-provided arena and contain no
// serialized compiler IR.
typedef struct loom_aie2p_array_plan_t {
  // Array-program function consumed by the plan.
  const loom_op_t* function_op;
  // Immutable physical family used by the plan.
  const loom_xdna_array_family_t* family;
  // Logical worker groups in source order.
  const loom_aie2p_array_group_t* groups;
  // Number of logical worker groups.
  iree_host_size_t group_count;
  // External bindings in source order.
  const loom_aie2p_array_binding_t* bindings;
  // Number of active external bindings retained by the topology.
  iree_host_size_t binding_count;
  // Dense external ABI cardinality, including unused binding slots.
  uint32_t binding_slot_count;
  // Resident workers in source order.
  const loom_aie2p_array_worker_t* workers;
  // Number of resident workers.
  iree_host_size_t worker_count;
  // Typed endpoints, including partition adapters, in source order.
  const loom_aie2p_array_endpoint_t* endpoints;
  // Number of typed endpoints.
  iree_host_size_t endpoint_count;
  // Logical channels in source order.
  const loom_aie2p_array_channel_t* channels;
  // Number of logical channels.
  iree_host_size_t channel_count;
  // Physical worker placements.
  const loom_aie2p_array_worker_plan_t* worker_plans;
  // Number of physical worker placements.
  iree_host_size_t worker_plan_count;
  // Function-local worker storage placements.
  const loom_aie2p_array_worker_storage_plan_t* worker_storage;
  // Number of function-local worker storage placements.
  iree_host_size_t worker_storage_count;
  // Worker ABI ports bound to planned channel rings.
  const loom_aie2p_array_worker_port_plan_t* worker_ports;
  // Number of worker ABI port bindings.
  iree_host_size_t worker_port_count;
  // Worker port row indices in each worker's leaf resource declaration order.
  // Each worker range starts at first_port and contains resource_count entries
  // from its requirements. It shares the worker_ports allocation; unused
  // topology ports require synchronization but have no resource-map entry.
  const uint32_t* worker_resource_ports;
  // Worker-local resident state ordinals in folded-output order. Each worker
  // range starts at first_port and contains fold_output_count entries.
  const uint32_t* worker_fold_output_states;
  // Logical channel slots with endpoint-local storage views.
  const loom_aie2p_array_channel_slot_t* channel_slots;
  // Number of logical channel slots.
  iree_host_size_t channel_slot_count;
  // Hardware lock allocations.
  const loom_aie2p_array_lock_plan_t* locks;
  // Number of hardware lock allocations.
  iree_host_size_t lock_count;
  // Compute and shim DMA channel allocations.
  const loom_aie2p_array_dma_plan_t* dma_channels;
  // Number of DMA channel allocations.
  iree_host_size_t dma_channel_count;
  // Programmed stream route connections.
  const loom_aie2p_array_route_plan_t* routes;
  // Number of stream route connections.
  iree_host_size_t route_count;
  // Completion packet resources, once per shim with an egress binding.
  const loom_aie2p_array_completion_route_t* completion_routes;
  // Number of completion routes.
  iree_host_size_t completion_route_count;
  // External binding patch records.
  const loom_aie2p_array_binding_plan_t* binding_plans;
  // Number of external binding patch records.
  iree_host_size_t binding_plan_count;
} loom_aie2p_array_plan_t;

// Extracts and plans one verified amd.xdna.aie2p.array Low function.
//
// Low verification establishes the empty register signature, isolated
// single-block topology vocabulary, typed tile payloads, and core worker
// contracts. Entity lookups consume those producer invariants directly.
// Extraction admits defined workers with empty register signatures before
// publishing their entities and consuming the source inventory.
//
// Exact SSA facts drive all resource cardinalities and placement coordinates.
// Every worker entry must have one matching source inventory in |leaves|. The
// planner maps external binding channels through shim DMA and compute endpoints
// whose storage and locks are visible to the worker. The worker tile is
// preferred, with adjacent compute tiles providing additional physical DMA
// channels without changing the logical port topology. Vertically adjacent
// workers communicate through neighbor-visible memory; all other worker
// channels use compute DMA and the stream network.
// Final resident code size and allocation are checked after channel
// realization; physical planning does not compile source leaves or allocate
// their registers.
// Authored target admission failures are delivered through |diagnostic_emitter|
// with |out_valid| false. Only a complete admitted plan is published in
// |out_plan|; otherwise it remains empty. Status reports infrastructure
// failures.
iree_status_t loom_aie2p_array_plan_build(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_aie2p_array_leaf_t* leaves, iree_host_size_t leaf_count,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_aie2p_array_plan_t* out_plan, bool* out_valid);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_PLAN_H_
