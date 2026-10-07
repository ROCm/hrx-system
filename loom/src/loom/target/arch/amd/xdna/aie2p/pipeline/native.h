// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_NATIVE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_NATIVE_H_

#include "loom/target/arch/amd/xdna/aie2p/array/route.h"
#include "loom/target/arch/amd/xdna/aie2p/facts.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/worker.h"
#include "loom/transforms/pipeline/realization.h"

// One endpoint's admitted cursor behavior for this execution occurrence.
typedef struct loom_aie2p_native_cursor_t {
  // Strand owning the cursor, or UINT32_MAX until selected.
  uint32_t worker;
  // Retained invocation admission bound, independent of physical slot count.
  uint32_t maximum_admissions;
  // Multiple admissions can select different physical slots.
  bool advances;
} loom_aie2p_native_cursor_t;

// Physical FIFO state. Addresses select storage; identity selects the protocol.
typedef struct loom_aie2p_native_channel_t {
  // Source channel occurrence retaining geometry and identity.
  const loom_pipeline_resource_channel_t* source;
  // Canonical tile memory owning all records and credits.
  uint32_t pool_index;
  // First slot's byte address relative to the owner tile.
  uint32_t byte_offset;
  // Physical byte displacement between adjacent records.
  uint32_t byte_stride;
  // Whole slot envelope used for neighbor visibility admission. Interleaved
  // records may have overlapping envelopes and disjoint payload elements.
  uint32_t record_byte_length;
  // Semaphore containing reusable slot credits.
  uint16_t free_lock;
  // Semaphore containing published record credits.
  uint16_t ready_lock;
  // Single cursor owner and slot movement in each direction.
  struct {
    // Strand advancing the read cursor.
    loom_aie2p_native_cursor_t reader;
    // Strand advancing the write cursor.
    loom_aie2p_native_cursor_t writer;
  } cursor;
} loom_aie2p_native_channel_t;

// One worker's admitted access to a channel's resident memory and credits.
typedef struct loom_aie2p_native_channel_access_t {
  // Protocol instance whose storage is addressed through this access window.
  const loom_aie2p_native_channel_t* channel;
  // First record's load address as seen by this worker.
  uint32_t address;
  // Minimum alignment of every record address.
  uint64_t alignment;
  // Worker-visible semaphore selectors for the channel's owner tile.
  struct {
    // Reusable record credits.
    uint16_t free;
    // Published record credits.
    uint16_t ready;
  } locks;
} loom_aie2p_native_channel_access_t;

// Explicit caller buffer contract, shared by every borrowing DMA operation.
typedef struct loom_aie2p_native_binding_t {
  // Original entry argument identity.
  loom_value_id_t root;
  // Complete minimum accessible extent proved from retained endpoint ranges.
  uint64_t byte_length;
  // Minimum caller-provided alignment retained from construction facts.
  uint64_t byte_alignment;
  // Union of native read/write binding access bits.
  uint32_t access;
} loom_aie2p_native_binding_t;

// Per-tile resource ownership established before any executable emission.
typedef struct loom_aie2p_native_tile_t {
  // Physical coordinate in this execution occurrence.
  loom_xdna_tile_coordinate_t coordinate;
  // Generated capacity and addressing facts.
  const loom_xdna_tile_facts_t* facts;
  // Canonical memory pool, or UINT32_MAX for a shim/memory tile.
  uint32_t pool_index;
  // Next available hardware semaphore.
  uint16_t next_lock;
  // Next available DMA buffer descriptor.
  uint16_t next_descriptor;
  // Next available memory-to-stream DMA engine.
  uint8_t next_mm2s;
  // Next available stream-to-memory DMA engine.
  uint8_t next_s2mm;
  // Whether a resident strand owns this core.
  bool has_worker;
} loom_aie2p_native_tile_t;

// One independently ordered DMA stream pair owned by one issuing worker.
typedef struct loom_aie2p_native_dma_path_t {
  // Next pair used by the worker.
  struct loom_aie2p_native_dma_path_t* next;
  // On-chip endpoint.
  loom_aie2p_native_tile_t* local;
  // External endpoint.
  loom_aie2p_native_tile_t* shim;
  // Local direction-specific engine.
  uint8_t local_engine;
  // Shim direction-specific engine.
  uint8_t shim_engine;
  // Destination control packet IDs.
  struct {
    // Local endpoint's tile-control route.
    uint8_t local;
    // Shim endpoint's tile-control route.
    uint8_t shim;
  } control;
  // Shim completion controller, zero for ingress without task tokens.
  uint8_t completion_packet;
  // True for external reads, false for external writes.
  bool ingress;
} loom_aie2p_native_dma_path_t;

// One physical route edge carrying the packet identity used during admission.
typedef struct loom_aie2p_native_route_t {
  // Corresponding edge in the canonical route builder.
  const loom_aie2p_array_route_plan_t* edge;
  // Packet ID, or UINT8_MAX for an unframed circuit.
  uint8_t packet;
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
  // Desired values retained by route admission in the pass arena.
  uint32_t* words;
} loom_aie2p_native_register_bank_t;

// A native pipeline owns route selection on each traversed switch. Its first
// invocation replaces complete banks after prior work has drained; externally
// prefixed routes are not an undeclared input to this materialization.
typedef struct loom_aie2p_native_switch_t {
  // Next routed switch in first-use order, excluding untouched tiles.
  struct loom_aie2p_native_switch_t* next;
  // Master, slave, and packet-filter banks from the generated register facts.
  loom_aie2p_native_register_bank_t banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_COUNT];
} loom_aie2p_native_switch_t;

typedef struct loom_aie2p_native_transfer_t {
  // Next transfer in the worker's retained issue order.
  struct loom_aie2p_native_transfer_t* next;
  // Canonical admitted copy request and its group index.
  const loom_kernel_async_transfer_t* source;
  // Wait that retires this transfer's group.
  const loom_op_t* completion;
  // Read admission fused into this descriptor's ready-credit acquire, or NULL.
  const loom_op_t* admission;
  // Admitted engine pair and physical routes.
  loom_aie2p_native_dma_path_t* path;
  // Caller buffer ordinal supplying the external base.
  uint32_t binding;
  // Worker-local address pair for a runtime-varying external offset. Unused
  // when invocation binding patches the dedicated Shim descriptor directly.
  uint64_t base_storage_offset;
  // Worker-visible load address of the dynamic transfer's relocated base.
  uint32_t base_load_address;
  // Local descriptor allocated for this static transfer site.
  uint16_t local_descriptor;
  // Shim descriptor allocated for this static transfer site.
  uint16_t shim_descriptor;
  // On-chip DMA completion semaphore for local arrivals.
  uint16_t completion_lock;
  // Worker-visible selector of the local-arrival completion semaphore.
  uint16_t completion_selector;
  // Full local descriptor template, excluding the dynamic record base.
  uint32_t local_words[6];
  // Full shim descriptor template, excluding its relocated address pair.
  uint32_t shim_words[8];
  // Canonical external endpoint projection, retained through emission.
  const loom_view_region_t* external_view;
  // Canonical local endpoint projection, retained through emission.
  const loom_view_region_t* local_view;
  // Admitted channel geometry owning the local record's physical slots.
  const loom_aie2p_native_channel_t* local_channel;
  // Read/write capability carrying the record's byte offset after realization.
  loom_value_id_t local_record;
  // This transfer's endpoint can select more than the initial physical slot.
  bool record_dynamic;
  // Selected ordinary helper that submits both endpoint descriptors.
  loom_symbol_ref_t submit;
  // Selected ordinary helper that observes endpoint completion.
  loom_symbol_ref_t wait;
} loom_aie2p_native_transfer_t;

typedef enum loom_aie2p_native_execution_e {
  LOOM_AIE2P_NATIVE_EXECUTION_CORE,
  LOOM_AIE2P_NATIVE_EXECUTION_CONFIGURATION,
  LOOM_AIE2P_NATIVE_EXECUTION_DMA,
} loom_aie2p_native_execution_t;

typedef struct loom_aie2p_native_worker_t {
  // Authored location used for physical access and route selection.
  loom_aie2p_native_tile_t* tile;
  // Engine implementing the strand; only CORE occupies instruction memory.
  loom_aie2p_native_execution_t execution;
  // Autonomous descriptor iteration for a regular ingress worker.
  struct {
    // Actual task execution count; zero for instruction/configuration engines.
    uint32_t count;
    // First external record's byte offset from its caller binding.
    uint32_t external_byte_offset;
    // Byte increment between consecutive external records; zero reuses input.
    uint32_t external_byte_stride;
  } repetition;
  // Semaphore released at the complete worker exit.
  uint16_t completion_lock;
  // Worker-visible selector of its completion semaphore.
  uint16_t completion_selector;
  // Admitted channel access windows in private cursor-state order.
  struct {
    // Access records owned by this worker occurrence.
    loom_aie2p_native_channel_access_t* accesses;
    // Number of distinct channel accesses.
    iree_host_size_t count;
    // Access index by construction channel index, UINT32_MAX when not used.
    uint32_t* indices;
  } channels;
  // First selected borrowed DMA operation.
  loom_aie2p_native_transfer_t* transfers;
  // Ordered DMA paths owned exclusively by this worker.
  loom_aie2p_native_dma_path_t* paths;
} loom_aie2p_native_worker_t;

typedef struct loom_aie2p_native_context_t {
  // Common compiler lifecycle and diagnostic owner.
  loom_pass_t* pass;
  // Exact deployment profile selected by the entry.
  const loom_aie2p_target_facts_t* target;
  // Immutable physical array topology.
  const loom_xdna_array_family_t* family;
  // Physical tiles in column-major order.
  loom_aie2p_native_tile_t* tiles;
  // Tile owner by canonical backing-pool ordinal.
  loom_aie2p_native_tile_t** pool_tiles;
  // Common construction inventory for tile memory.
  loom_pipeline_realization_inventory_t inventory;
  // Source channels in canonical construction order.
  loom_aie2p_native_channel_t* channels;
  // Bound workers in source strand order.
  loom_aie2p_native_worker_t* workers;
  // Caller buffer contracts in source argument order.
  loom_aie2p_native_binding_t* bindings;
  // Number of caller buffer contracts.
  iree_host_size_t binding_count;
  // Physical interconnect allocation shared by payload and control routes.
  loom_aie2p_array_route_builder_t routing;
  // Packet identity for every physical route edge.
  loom_aie2p_native_route_t* routes;
  // Complete desired routing state for each traversed switch.
  loom_aie2p_native_switch_t* switches;
  // Typed ordinary helper emitter.
  loom_aie2p_worker_builder_t code;
} loom_aie2p_native_context_t;

iree_status_t loom_aie2p_native_reject(
    const loom_aie2p_native_context_t* context, const loom_op_t* op,
    iree_string_view_t requirement);

iree_status_t loom_aie2p_native_emit_worker(
    void* user_data, loom_rewriter_t* rewriter,
    const loom_pipeline_realization_t* realization,
    iree_host_size_t worker_index);
iree_status_t loom_aie2p_native_emit_configuration(
    void* user_data, loom_rewriter_t* rewriter,
    const loom_pipeline_realization_t* realization);

// Selects descriptor, engine, route, completion, and relocated-address storage
// from the canonical movement streams before either executable is rewritten.
iree_status_t loom_aie2p_native_select_transfers(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, bool* out_valid);

// Replaces retained copies/groups/waits using the selected transport helpers.
// Channel access carriers are replaced by the following channel rewrite.
iree_status_t loom_aie2p_native_emit_transfers(
    loom_aie2p_native_context_t* context, loom_rewriter_t* rewriter,
    const loom_pipeline_realization_t* realization,
    iree_host_size_t worker_index);

iree_status_t loom_aie2p_pipeline_realize(loom_pass_t* pass,
                                          loom_module_t* module,
                                          loom_func_like_t function);

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_NATIVE_H_
