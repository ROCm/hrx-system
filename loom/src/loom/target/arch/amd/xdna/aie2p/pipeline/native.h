// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_NATIVE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_NATIVE_H_

#include "loom/target/arch/amd/xdna/aie2p/array/route.h"
#include "loom/target/arch/amd/xdna/aie2p/facts.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_plan.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/switch.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/worker.h"
#include "loom/target/arch/amd/xdna/array/registers.h"
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
  // Logical payload elements occupy one contiguous byte range in order.
  bool record_dense;
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
  // Admitted owner aperture shared by payload addresses and lock selectors.
  const loom_xdna_address_window_t* window;
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

// One worker formal bound to a pipeline-owned local allocation.
typedef struct loom_aie2p_native_buffer_access_t {
  // Worker entry argument whose uses consume this address.
  loom_value_id_t argument;
  // Allocation base load address as seen by this worker.
  uint32_t address;
  // Minimum alignment guaranteed for the allocation base.
  uint64_t alignment;
} loom_aie2p_native_buffer_access_t;

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
  // Native on-chip DMA encoding, NULL for a shim.
  const loom_aie2p_native_memory_layout_t* memory_layout;
  // Canonical memory pool, or UINT32_MAX for a shim tile.
  uint32_t pool_index;
  // Next available hardware semaphore.
  uint16_t next_lock;
  // Allocated descriptor count per channel-parity bank. Compute and Shim
  // modules use only bank zero; SRAM even/odd channels own disjoint banks.
  uint16_t descriptor_counts[2];
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

// One arrival credit reused only after its latest wait consumes that credit.
// Descriptors with different immutable geometries can share this semaphore.
typedef struct loom_aie2p_native_dma_completion_t {
  // Next physical completion semaphore in the worker's allocation order.
  struct loom_aie2p_native_dma_completion_t* next;
  // Local DMA tile owning the semaphore.
  loom_aie2p_native_tile_t* tile;
  // Most recently assigned copy across every descriptor sharing this credit.
  const struct loom_aie2p_native_transfer_t* last_transfer;
  // Tile-local hardware semaphore index.
  uint16_t lock;
  // Worker-visible selector, unused for configuration-only transfers.
  uint16_t selector;
} loom_aie2p_native_dma_completion_t;

// One physical DMA endpoint reused after its prior transfer and any other
// transfer sharing its completion semaphore have retired. Local and shim
// descriptors have independent geometries and retirement identities.
typedef struct loom_aie2p_native_dma_descriptor_t {
  // Next physical endpoint in the worker's allocation order.
  struct loom_aie2p_native_dma_descriptor_t* next;
  // Admitted engine pair and physical routes.
  loom_aie2p_native_dma_path_t* path;
  // Most recently assigned copy, retaining its retirement and address policy.
  const struct loom_aie2p_native_transfer_t* last_transfer;
  // Endpoint sequence, excluding dynamic addresses and completion releases.
  loom_aie2p_native_dma_sequence_t sequence;
  // Local arrival credit; NULL for shim, egress, and autonomous descriptors.
  loom_aie2p_native_dma_completion_t* completion;
  // True for the external shim endpoint, false for the local memory endpoint.
  bool shim;
  // Each issue patches this endpoint's address. A shared shim also patches
  // fixed offsets when its copies address different caller storage.
  bool address_dynamic;
} loom_aie2p_native_dma_descriptor_t;

typedef struct loom_aie2p_native_transfer_task_t
    loom_aie2p_native_transfer_task_t;

typedef struct loom_aie2p_native_transfer_t {
  // Next transfer in the worker's retained issue order.
  struct loom_aie2p_native_transfer_t* next;
  // Next segment in the same physical task.
  struct loom_aie2p_native_transfer_t* task_next;
  // Physical task selected for this transfer's compatible group partition.
  loom_aie2p_native_transfer_task_t* task;
  // Canonical admitted copy request and its group index.
  const loom_kernel_async_transfer_t* source;
  // Wait that retires this transfer's group.
  const loom_op_t* completion;
  // Read admission fused into the local ready-credit acquire, or NULL.
  const loom_op_t* admission;
  // Local endpoint whose descriptor and completion credit are available.
  loom_aie2p_native_dma_descriptor_t* local_descriptor;
  // External endpoint whose preceding transfer has retired before this copy.
  loom_aie2p_native_dma_descriptor_t* shim_descriptor;
  // Caller buffer ordinal supplying the external base.
  uint32_t binding;
  // Worker-local address pair when the descriptor patches its external address.
  // Unused when invocation binding patches one shared fixed address directly.
  uint64_t base_storage_offset;
  // Worker-visible load address of the dynamic transfer's relocated base.
  uint32_t base_load_address;
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
  // Ordinal of this segment in both endpoint descriptor chains.
  iree_host_size_t descriptor_ordinal;
  // Immutable local descriptor words before physical chain links are assigned.
  uint32_t local_words[8];
  // Immutable shim descriptor words before physical chain links are assigned.
  uint32_t shim_words[8];
} loom_aie2p_native_transfer_t;

// One physical task for a compatible partition of an authored async group.
// Both endpoint chains preserve member order and complete as one unit.
struct loom_aie2p_native_transfer_task_t {
  // Next task in the worker's retained issue order.
  loom_aie2p_native_transfer_task_t* next;
  // First transfer in this task's descriptor order.
  loom_aie2p_native_transfer_t* first_transfer;
  // Last transfer, retained while selection appends compatible members.
  loom_aie2p_native_transfer_t* last_transfer;
  // Number of descriptors in each endpoint chain.
  iree_host_size_t transfer_count;
  // Shared engine pair and physical route for every member.
  loom_aie2p_native_dma_path_t* path;
  // Local endpoint descriptor chain and ingress completion credit.
  loom_aie2p_native_dma_descriptor_t* local_descriptor;
  // External endpoint descriptor chain.
  loom_aie2p_native_dma_descriptor_t* shim_descriptor;
  // Every local descriptor address is patched at each issue.
  bool local_address_dynamic;
  // Every shim descriptor address is patched at each issue.
  bool shim_address_dynamic;
  // Selected ordinary helper that submits both endpoint chains.
  loom_symbol_ref_t submit;
  // Selected ordinary helper that observes task completion.
  loom_symbol_ref_t wait;
};

// One on-chip DMA endpoint in an on-die channel distribution.
typedef struct loom_aie2p_native_distribution_endpoint_t {
  // FIFO whose whole records are transferred by this endpoint.
  const loom_aie2p_native_channel_t* channel;
  // On-chip memory tile owning the FIFO and DMA engine.
  loom_aie2p_native_tile_t* tile;
  // Direction-specific DMA engine.
  uint8_t engine;
  // Descriptor images, finite queued tasks, and invocation rewind state.
  loom_aie2p_native_dma_sequence_t sequence;
  // Invocation retirement must explicitly observe this DMA channel idle.
  bool wait_for_idle;
} loom_aie2p_native_distribution_endpoint_t;

// One source DMA stream multicast to independently completing destination
// rings. All endpoints execute the same bounded number of records.
typedef struct loom_aie2p_native_distribution_t {
  // Next distribution owned by the same transport strand.
  struct loom_aie2p_native_distribution_t* next;
  // Memory-to-stream endpoint reclaiming source records after transmission.
  loom_aie2p_native_distribution_endpoint_t sender;
  // Arena-owned stream-to-memory endpoints in source fanout order.
  loom_aie2p_native_distribution_endpoint_t* receivers;
  // Number of independently publishing receiver rings.
  iree_host_size_t receiver_count;
  // Exact number of records transferred by every endpoint.
  uint32_t repetitions;
} loom_aie2p_native_distribution_t;

typedef enum loom_aie2p_native_execution_e {
  LOOM_AIE2P_NATIVE_EXECUTION_CORE,
  LOOM_AIE2P_NATIVE_EXECUTION_CONFIGURATION,
  LOOM_AIE2P_NATIVE_EXECUTION_DMA,
  LOOM_AIE2P_NATIVE_EXECUTION_DISTRIBUTION,
} loom_aie2p_native_execution_t;

typedef struct loom_aie2p_native_worker_t {
  // Authored location used for physical access and route selection.
  loom_aie2p_native_tile_t* tile;
  // Engine implementing the strand; only CORE occupies instruction memory.
  loom_aie2p_native_execution_t execution;
  // Autonomous descriptor iteration for a regular ingress worker.
  struct {
    // Local record count; zero for instruction/configuration engines.
    uint32_t count;
    // First external record's byte offset from its caller binding.
    uint32_t external_byte_offset;
    // Executions of the external descriptor; one when records are coalesced.
    uint32_t shim_repetitions;
    // External descriptor geometry, excluding the relocated address pair.
    uint32_t shim_words[8];
  } repetition;
  // Semaphore released at the complete worker exit.
  uint16_t completion_lock;
  // Worker-visible selector of its completion semaphore.
  uint16_t completion_selector;
  // Pipeline-owned buffer captures selected for this worker.
  struct {
    // Access records in worker argument order.
    loom_aie2p_native_buffer_access_t* accesses;
    // Number of retained buffer accesses.
    iree_host_size_t count;
  } buffers;
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
  // Physical tasks in first-member selection order.
  loom_aie2p_native_transfer_task_t* transfer_tasks;
  // Physical local/shim endpoints in their first-use initialization order.
  loom_aie2p_native_dma_descriptor_t* descriptors;
  // Physical arrival credits shared independently of descriptor geometry.
  loom_aie2p_native_dma_completion_t* dma_completions;
  // First selected on-die channel distribution.
  loom_aie2p_native_distribution_t* distributions;
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

// Selects and retains a regular ingress stream's external descriptor geometry.
// Stable-source and exact consumer-completion admission precede this query.
bool loom_aie2p_native_select_ingress_descriptor(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_worker_t* source,
    const loom_movement_request_t* request, uint32_t byte_stride,
    uint32_t repetitions, loom_aie2p_native_worker_t* worker);

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
