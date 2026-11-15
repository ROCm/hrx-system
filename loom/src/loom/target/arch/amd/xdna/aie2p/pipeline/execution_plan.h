// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_EXECUTION_PLAN_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_EXECUTION_PLAN_H_

#include "loom/analysis/pipeline/entries.h"
#include "loom/target/arch/amd/xdna/aie2p/array/route.h"

#ifdef __cplusplus
extern "C" {
#endif

// Complete endpoint ownership for one compiler-generated control stream.
// Commands, replies, and collective words retain their framing on circuits.
typedef struct loom_aie2p_native_control_source_t {
  // Source-owned physical prefixes, independent of destination route labels.
  loom_aie2p_array_route_tree_t tree;
  // Sole owned endpoint, or UINT32_MAX for multiple/external destinations
  // or any route borrowing payload lanes.
  uint32_t destination;
  // Packet identities already emitted by this physical sender.
  uint32_t packet_mask;
} loom_aie2p_native_control_source_t;

// A private downward command path through its column's SRAM tile. Hardware
// register commands retire independently of payload consumers. Late senders
// can share the retained interval with disjoint packet identities.
typedef struct loom_aie2p_native_control_path_t {
  // First edge in the original physical route inventory.
  iree_host_size_t first;
  // Number of edges; zero means that no eligible path was selected.
  iree_host_size_t count;
  // Original control source owning the path's packet namespace.
  uint32_t source;
  // Complete register-command carrier shared by admitted core senders.
  loom_aie2p_array_route_sharing_t sharing;
  // Outgoing core packet identities reserved across the column, including
  // collective branches that may traverse a shared command prefix.
  uint32_t packets;
} loom_aie2p_native_control_path_t;

// One physical route edge carrying the packet identity used during admission.
typedef struct loom_aie2p_native_route_t {
  // Corresponding edge in the canonical route builder.
  const loom_aie2p_array_route_plan_t* edge;
  // Packet ID, or UINT8_MAX for a circuit without packet filtering.
  uint8_t packet;
  // A complete compiler-owned register command terminating at tile control.
  // Its body and hardware consumption do not depend on payload credits.
  bool register_command;
  // Retained control source, or UINT32_MAX for ordinary payload transport.
  uint32_t control_source;
} loom_aie2p_native_route_t;

typedef enum loom_aie2p_native_switch_bank_e {
  LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS = 0,
  LOOM_AIE2P_NATIVE_SWITCH_BANK_SLAVES,
  LOOM_AIE2P_NATIVE_SWITCH_BANK_FILTERS,
  LOOM_AIE2P_NATIVE_SWITCH_BANK_COUNT,
} loom_aie2p_native_switch_bank_t;

// Complete valid register bank, with zero words for every disabled entry.
typedef struct loom_aie2p_native_register_bank_t {
  // Absolute address in the native transaction's 32-bit register space.
  uint32_t address;
  // Number of contiguous register words, excluding reserved address holes.
  uint32_t word_count;
  // Desired values retained by route admission in the owning execution plan.
  uint32_t* words;
} loom_aie2p_native_register_bank_t;

// A native pipeline owns route selection on each traversed switch. Its first
// invocation replaces complete banks after prior work has drained; externally
// prefixed routes are not an undeclared input to this materialization.
typedef struct loom_aie2p_native_switch_t {
  // Next routed switch in first-use order, excluding untouched tiles.
  struct loom_aie2p_native_switch_t* next;
  // Physical owner retained from switch admission.
  loom_xdna_tile_coordinate_t coordinate;
  // Master, slave, and packet-filter banks from the generated register facts.
  loom_aie2p_native_register_bank_t banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_COUNT];
} loom_aie2p_native_switch_t;

// Physical endpoint identity, independent of selected route edges. DMA channels
// name engines, including on shim tiles; other ports name stream channels.
typedef struct loom_aie2p_native_connection_endpoint_t {
  // Tile containing the endpoint.
  loom_xdna_tile_coordinate_t coordinate;
  // Endpoint's architectural port class.
  loom_xdna_stream_port_t port;
  // Direction-specific channel within the port.
  uint8_t channel;
} loom_aie2p_native_connection_endpoint_t;

// One requested control delivery, retained before physical routing. Equal
// source/packet pairs form one multicast destination set.
typedef struct loom_aie2p_native_control_connection_t {
  // Physical sender of the framed control record.
  loom_aie2p_native_connection_endpoint_t source;
  // Intended consumer of that record, independent of chosen switch paths.
  loom_aie2p_native_connection_endpoint_t destination;
  // Five-bit packet identity within the sender's namespace.
  uint8_t packet;
} loom_aie2p_native_control_connection_t;

// Complete delivery contract for one physical sender and packet class.
// All destinations consume the same framing; payload receivers remove the
// header while control receivers retain it.
typedef struct loom_aie2p_native_connection_t {
  // Physical sender, including its direction-local DMA engine when applicable.
  loom_aie2p_native_connection_endpoint_t source;
  // Five-bit packet identity, or UINT8_MAX for an unframed circuit.
  uint8_t packet;
  // Receivers consume the packet header instead of stripping it at the switch.
  bool retain_header;
  // First receiver in the owning inventory's flat destination array.
  iree_host_size_t destination_offset;
  // Number of receivers in the complete multicast destination set.
  iree_host_size_t destination_count;
} loom_aie2p_native_connection_t;

// Source-independent intended deliveries, produced once before switch auditing.
// Flat destination storage keeps independent ownership copies
// allocation-bounded.
typedef struct loom_aie2p_native_connections_t {
  // Distinct sender/packet deliveries, including both payload and control.
  const loom_aie2p_native_connection_t* groups;
  // Number of complete delivery groups.
  iree_host_size_t group_count;
  // Concatenated destination sets referenced by group offsets.
  const loom_aie2p_native_connection_endpoint_t* destinations;
  // Number of endpoints in destinations.
  iree_host_size_t destination_count;
} loom_aie2p_native_connections_t;

// Allocated physical extents after source program construction. Counts describe
// occupied prefixes; lifetime-specific descriptor reuse remains in the source
// selection and does not make those prefixes available to independent work.
typedef struct loom_aie2p_execution_tile_t {
  // Reserved local memory prefix, in bytes; zero on shim tiles.
  uint64_t memory_length;
  // Occupied hardware semaphore prefix.
  uint16_t lock_count;
  // Occupied descriptors per channel-parity bank; only SRAM uses bank one.
  uint16_t descriptor_counts[2];
  // Occupied memory-to-stream engine prefix.
  uint8_t mm2s_count;
  // Occupied stream-to-memory engine prefix.
  uint8_t s2mm_count;
} loom_aie2p_execution_tile_t;

// Physical selection shared by all native entries of one pipeline. Every
// referenced array belongs to the compilation owner. Static family facts and
// the representation descriptor are the only references outside that owner.
// Program construction completes this record before entry emission consumes it;
// independent module projection retains one complete copy per shared scope.
typedef struct loom_aie2p_execution_plan_t {
  // Common lifetime and ownership contract, embedded first for projection.
  loom_pipeline_entry_scope_t base;
  // Physical occupancy indexed by column * family row count + row.
  loom_aie2p_execution_tile_t* tiles;
  // Complete selected routes, occupied lanes, and packet-domain ownership.
  loom_aie2p_array_route_builder_t routing;
  // Packet identity paired one-for-one with routing.routes.
  loom_aie2p_native_route_t* routes;
  // Source ownership for each tile's core and tile-control outputs.
  loom_aie2p_native_control_source_t* control_sources;
  // Unique source per core/tile-control destination; UINT32_MAX means unused,
  // and UINT32_MAX - 1 means that different sources share the destination.
  uint32_t* control_destination_sources;
  // Longest private downward hardware-command suffix per physical column.
  loom_aie2p_native_control_path_t* control_paths;
  // Admitted switch register banks in first-use order.
  loom_aie2p_native_switch_t* switches;
  // Complete payload/control intent, independent of selected switch paths.
  loom_aie2p_native_connections_t connections;
} loom_aie2p_execution_plan_t;

// Concrete common-scope representation with owned physical resource copying.
extern const loom_pipeline_entry_scope_type_t loom_aie2p_execution_plan_type;

// Copies the physical portion of a completed selection into the caller's scope.
// The source may borrow the producer's transient arrays. The target retains no
// such references, and its already initialized common lifetime prefix remains
// unchanged. Extra connection capacity reserves one receiver and a complete
// physical route per added delivery, without recopying the selected inventory.
iree_status_t loom_aie2p_execution_plan_copy_resources(
    const loom_aie2p_execution_plan_t* source,
    iree_host_size_t added_connection_capacity, iree_arena_allocator_t* arena,
    loom_aie2p_execution_plan_t* target);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_EXECUTION_PLAN_H_
