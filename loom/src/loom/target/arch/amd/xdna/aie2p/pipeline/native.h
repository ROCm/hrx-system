// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_NATIVE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_NATIVE_H_

#include "loom/analysis/pipeline/execution.h"
#include "loom/target/arch/amd/xdna/aie2p/array/route.h"
#include "loom/target/arch/amd/xdna/aie2p/facts.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_plan.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/join.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/switch.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/worker.h"
#include "loom/target/arch/amd/xdna/array/registers.h"
#include "loom/transforms/pipeline/realization.h"

#ifdef __cplusplus
extern "C" {
#endif

// One endpoint's admitted cursor behavior for this execution occurrence.
typedef struct loom_aie2p_native_cursor_t {
  // Representative strand owning the cursor, or UINT32_MAX until selected.
  uint32_t worker;
  // Retained admission bound of the representative worker's invocation,
  // independent of physical slot count. Shared cursors have no aggregate bound.
  uint32_t maximum_admissions;
  // Last claiming transport batch; each phase owns this cursor at most once.
  // Unused for instruction-executing cursor owners.
  uint32_t transport_batch;
  // Several worker occurrences use this cursor. They require one physical
  // engine with disjoint lifetimes; single-owner completion proofs do not
  // apply.
  bool shared;
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
  // Nonzero only when a complete FIFO is retained across ordered replays.
  // Its producer admits the complete batch. A finite lifetime needs one
  // ownership credit; an autonomous stream can exchange credits per replay.
  struct {
    // Number of adjacent logical records in the retained batch.
    uint32_t record_count;
    // Ready/free credits exchanged at each complete batch boundary.
    uint32_t credits;
  } replay;
  // Existing compute reader whose nth admission proves this FIFO's nth record
  // has been transported. NULL when no exact one-to-one progress is retained.
  const struct loom_aie2p_native_channel_t* progress;
  // Exact singleton credit-return sites owned by an existing configuration
  // reader. The final site retires ownership without granting more work.
  struct {
    // Number of complete records reclaimed by the straight-line continuation.
    uint32_t count;
    // Last credit return, or NULL when this reader has no such continuation.
    const struct loom_aie2p_native_configuration_action_t* last;
  } configuration_returns;
  // Receiver-granted sender whose payload path actually carries control too.
  struct loom_aie2p_native_delivery_t* delivery;
} loom_aie2p_native_channel_t;

// One worker's admitted access to a channel's resident memory and credits.
typedef struct loom_aie2p_native_channel_access_t {
  // Protocol instance whose storage is addressed through this access window.
  const loom_aie2p_native_channel_t* channel;
  // Admitted owner aperture shared by payload addresses and lock selectors.
  // NULL for a finite SRAM producer whose payload is consumed only by DMA.
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
  // Completion tokens consumed per task: zero for visible-memory ingress,
  // one for shim egress, or two when both SRAM and shim must retire.
  uint8_t completion_tokens;
  // Latest selected token-producing task on this path. Other paths sharing
  // the core stream must retire it or join it in the same source wait.
  const struct loom_aie2p_native_transfer_t* last_token_transfer;
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

// Repeated external-address patch bodies for one physical local descriptor.
// The local descriptor fixes the routes, chain geometry, and local patch
// policy; relocation bases, shim commands, and shim queue words remain call
// arguments. Ordered worker occurrences on the same core share this body.
typedef struct loom_aie2p_native_dma_submit_t {
  // Number of static sites, independent of runtime loop trip counts.
  iree_host_size_t use_count;
  // One native leaf materialized on first use, or null before emission.
  loom_symbol_ref_t function;
} loom_aie2p_native_dma_submit_t;

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
  // Each issue patches this endpoint's address. Shared descriptors also patch
  // fixed offsets when their copies address different storage or record slots.
  bool address_dynamic;
  // External-patch group shared by ordered uses of this physical endpoint.
  loom_aie2p_native_dma_submit_t* submit_group;
} loom_aie2p_native_dma_descriptor_t;

typedef struct loom_aie2p_native_transfer_task_t
    loom_aie2p_native_transfer_task_t;

typedef struct loom_aie2p_native_relocation_t {
  // First dynamic copy using this physical core's caller binding. It owns the
  // invocation-time address initialization shared by all later copies.
  const struct loom_aie2p_native_transfer_t* first_transfer;
  // Immutable caller base address pair in the core's local memory module.
  uint64_t storage_offset;
  // Core-relative load address of that pair.
  uint32_t load_address;
} loom_aie2p_native_relocation_t;

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
  // Worker occurrence owning the issue and completion. Repeated invocations
  // can share source operations while having disjoint execution lifetimes.
  iree_host_size_t worker_index;
  // Read admission fused into the local ready-credit acquire, or NULL.
  const loom_op_t* admission;
  // Local endpoint whose descriptor and completion credit are available.
  loom_aie2p_native_dma_descriptor_t* local_descriptor;
  // External endpoint whose preceding transfer has retired before this copy.
  loom_aie2p_native_dma_descriptor_t* shim_descriptor;
  // Caller buffer ordinal supplying the external base.
  uint32_t binding;
  // Authored copies of adjacent records represented by this physical transfer.
  uint32_t source_count;
  // Immutable caller base shared by this core's dynamic copies of one binding.
  // NULL when invocation binding patches a fixed descriptor address directly.
  const loom_aie2p_native_relocation_t* relocation;
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
  // Retained external-patch sharing group, or NULL for inline-only submissions.
  loom_aie2p_native_dma_submit_t* submit_group;
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

// One immutable packet class selecting a set of independently published FIFOs.
typedef struct loom_aie2p_native_distribution_packet_t {
  // First visit whose consecutive fanout branches define this class.
  const loom_pipeline_transport_copy_t* destinations;
  // Number of distinct destination channels in the visit.
  iree_host_size_t destination_count;
} loom_aie2p_native_distribution_packet_t;

// One source DMA program multicast to independently completing destination
// rings. Packet classes may select different subsets for each source visit.
typedef struct loom_aie2p_native_distribution_t {
  // Next distribution owned by the same transport strand.
  struct loom_aie2p_native_distribution_t* next;
  // First geometry owning this physical engine/route path; self when unshared.
  const struct loom_aie2p_native_distribution_t* path;
  // Canonical ownership and copy schedule used to intern geometry variants.
  const loom_pipeline_transport_batch_t* source;
  // Loop table owning source; shared geometries may come from different calls.
  const loom_pipeline_transport_t* transport;
  // Memory-to-stream endpoint reclaiming source records after transmission.
  loom_aie2p_native_distribution_endpoint_t sender;
  // Arena-owned stream-to-memory endpoints in the physical path's fanout order.
  loom_aie2p_native_distribution_endpoint_t* receivers;
  // Number of independently publishing receiver rings.
  iree_host_size_t receiver_count;
  // Channel-indexed receiver ordinals owned by the first packet geometry.
  // UINT32_MAX marks channels outside this path's receiver set.
  uint32_t* receiver_ordinals;
  // Packet classes across all phases, owned by the first packet geometry.
  loom_aie2p_native_distribution_packet_t* packets;
  // Physical routing identity shared by all packet classes on this path.
  uint32_t source_route;
  // Number of packet identities in the physical path's class table.
  uint8_t packet_count;
  // Source transmissions in a uniform circuit stream; zero for packet programs.
  uint32_t repetitions;
  // Exact route-edge interval produced for this uniform source's fanout.
  struct {
    // First edge in the route builder's stable append-only table.
    iree_host_size_t first;
    // Number of edges, excluding routes owned by other distributions.
    iree_host_size_t count;
  } routes;
} loom_aie2p_native_distribution_t;

// One ordered use of an immutable distribution geometry. The controller's
// full-batch publication supplies the existing source readiness boundary.
typedef struct loom_aie2p_native_distribution_phase_t {
  // Next authored batch in this transport worker's execution order.
  struct loom_aie2p_native_distribution_phase_t* next;
  // Preceding phase whose endpoints retire before this phase is queued.
  const struct loom_aie2p_native_distribution_phase_t* previous;
  // Canonical source batch retaining its admission and replay contract.
  const loom_pipeline_transport_batch_t* source;
  // Shared immutable sender/receiver geometry for this phase.
  const loom_aie2p_native_distribution_t* distribution;
} loom_aie2p_native_distribution_phase_t;

// Repeated retirement observations of one physical DMA channel by one core.
// The controller's routes and the channel's status geometry are identical at
// every use, independently of descriptor contents or child invocation.
typedef struct loom_aie2p_native_retirement_t {
  // Next channel observed by this representative instruction worker.
  struct loom_aie2p_native_retirement_t* next;
  // First selected command supplying the immutable physical endpoint.
  const struct loom_aie2p_native_distribution_command_t* source;
  // Static child endpoint sites; runtime loop visits do not increase this.
  iree_host_size_t use_count;
  // Shared polling leaf, created during emission when several sites use it.
  loom_symbol_ref_t function;
} loom_aie2p_native_retirement_t;

// A finite child endpoint controlled by its producer or consuming core. The
// child boundary queues it before local work and observes idle after all local
// async waits, when the core's scalar reply stream is drained.
typedef struct loom_aie2p_native_distribution_command_t {
  // Next endpoint controlled by this worker occurrence.
  struct loom_aie2p_native_distribution_command_t* next;
  // Complete source child whose selected visit owns this endpoint lifetime.
  const loom_pipeline_invocation_t* invocation;
  // Immutable descriptor geometry selected for this child.
  const loom_aie2p_native_distribution_endpoint_t* endpoint;
  // Direction of the endpoint's DMA channel.
  loom_xdna_dma_direction_t direction;
  // Selected reuse group for this controller's status observation.
  loom_aie2p_native_retirement_t* retirement;
} loom_aie2p_native_distribution_command_t;

typedef enum loom_aie2p_native_execution_e {
  LOOM_AIE2P_NATIVE_EXECUTION_CORE,
  LOOM_AIE2P_NATIVE_EXECUTION_CONFIGURATION,
  LOOM_AIE2P_NATIVE_EXECUTION_DMA,
  LOOM_AIE2P_NATIVE_EXECUTION_DISTRIBUTION,
} loom_aie2p_native_execution_t;

// One configuration-owned channel turn. Whole-record ownership makes a lock
// value update safe: the opposite endpoint cannot publish another turn until
// this endpoint returns its credit.
typedef struct loom_aie2p_native_configuration_action_t {
  // Channel whose selected protocol supplies this action's semaphore.
  const loom_aie2p_native_channel_t* channel;
  // Physical semaphore within the channel's memory owner.
  uint16_t lock;
  // Value to observe before the update; -1 means no wait.
  int16_t expected;
  // Value to publish; -1 means this source action has no physical update.
  int16_t value;
  // Ordered distribution submitted before this complete-batch publication.
  const loom_aie2p_native_distribution_phase_t* distribution_phase;
} loom_aie2p_native_configuration_action_t;

typedef struct loom_aie2p_native_worker_t {
  // Authored location used for physical access and route selection.
  loom_aie2p_native_tile_t* tile;
  // Engine implementing the strand; only CORE occupies instruction memory.
  loom_aie2p_native_execution_t execution;
  // Generated private cursor-frame formal, or invalid if the worker has no
  // advancing endpoint. The physical entry initializes this state once.
  loom_value_id_t cursor_state_argument;
  // Configuration channel turns indexed by the source channel action table.
  loom_aie2p_native_configuration_action_t* configuration_actions;
  // Fixed record byte offsets indexed by the source's owned-value domain.
  const uint32_t* configuration_records;
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
  // Dynamic address slots indexed by caller binding, owned by the physical
  // core representative across child occurrences. Allocated at first use.
  loom_aie2p_native_relocation_t* relocations;
  // Physical tasks in first-member selection order.
  loom_aie2p_native_transfer_task_t* transfer_tasks;
  // Physical local/shim endpoints in first-use initialization order. Ordered
  // core occurrences retain these on their representative instruction worker.
  loom_aie2p_native_dma_descriptor_t* descriptors;
  // Physical arrival credits shared independently of descriptor geometry and
  // retained on the same representative as the descriptors they retire.
  loom_aie2p_native_dma_completion_t* dma_completions;
  // Immutable geometries owned by the representative distribution engine.
  loom_aie2p_native_distribution_t* distributions;
  // Ordered uses driven by controller publications; NULL for autonomous paths.
  loom_aie2p_native_distribution_phase_t* distribution_phases;
  // Child endpoints started and retired around this core's local worker call.
  loom_aie2p_native_distribution_command_t* distribution_commands;
  // Physical channel polling bodies shared by this core's ordered children.
  loom_aie2p_native_retirement_t* retirements;
  // Any occurrence sharing this distribution engine needs packet routing.
  bool distribution_packets;
  // DMA paths owned by this physical worker, shared by its ordered occurrences.
  loom_aie2p_native_dma_path_t* paths;
  // Physical DMA paths indexed by tile and direction. The representative
  // instruction worker owns these across its ordered child occurrences.
  loom_aie2p_native_dma_path_t** paths_by_tile;
  // Control-packet identity by destination tile, or UINT8_MAX when unassigned.
  // Routes belong to the physical core, independently of child call sites.
  uint8_t* control_packets;
  // Tile-control reply identity by source tile, or UINT8_MAX when unassigned.
  // Existing DMA-token routes can also carry explicitly requested status reads.
  uint8_t* control_replies;
  // Number of tile-control destinations selected for the physical core.
  uint8_t control_packet_count;
  // An external-write path already owns the core's completion stream.
  bool has_egress;
  // Reader-driven continuation groups sharing one window counter pair.
  struct loom_aie2p_native_refill_t* refills;
} loom_aie2p_native_worker_t;

typedef struct loom_aie2p_native_join_function_t {
  // Lazily emitted call-free helper, or null before its first boundary.
  loom_symbol_ref_t symbol;
  // Number of generated call sites sharing this exact lock program.
  iree_host_size_t use_count;
} loom_aie2p_native_join_function_t;

typedef struct loom_aie2p_native_join_t {
  // Physical lock tree shared by scopes with exactly equal participant sets.
  loom_aie2p_join_plan_t plan;
  // Lazily emitted join helper per physical engine, shared across phase visits.
  loom_aie2p_native_join_function_t* functions;
  // Next distinct tree requiring configuration-time semaphore initialization.
  struct loom_aie2p_native_join_t* next;
} loom_aie2p_native_join_t;

typedef struct loom_aie2p_native_collective_t {
  // Bound serial scope owning the collective's participant set.
  const loom_pipeline_invocation_schedule_t* scope;
  // Closed source borrow and its escaping value tuple.
  const loom_pipeline_collective_t* source;
  // Canonical FIFO whose existing reader supplies the record.
  const loom_aie2p_native_channel_t* channel;
  // Physical engine retaining the FIFO read cursor.
  iree_host_size_t reader;
  // Broadcast relay; equals reader when the reader broadcasts directly.
  iree_host_size_t relay;
  // Other participating cores receiving the reader's values.
  iree_host_size_t peer_count;
  // Reader-visible base address of the FIFO's record storage.
  uint32_t address;
  // Reader-visible readiness semaphore selector.
  uint16_t ready_lock;
  // Reader-visible reclamation semaphore selector.
  uint16_t free_lock;
  // Packet class on the reader's scalar stream.
  uint8_t reader_packet;
  // Packet class on the relay's scalar stream.
  uint8_t relay_packet;
} loom_aie2p_native_collective_t;

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
  // Admitted core sharing and transitive participants at each phase boundary.
  loom_pipeline_execution_t execution;
  // Collective selections indexed by source scope and borrow address.
  loom_aie2p_native_collective_t* collectives;
  // Number of closed immutable source borrows.
  iree_host_size_t collective_count;
  // Invocation-major core controllers of finite distribution boundaries, or
  // NULL when there are no ordered distributions. Each row retains the
  // invocation's complete transitive execution participants.
  iree_host_size_t* distribution_controllers;
  // Physical join selected for each canonical execution schedule row.
  loom_aie2p_native_join_t** joins;
  // Distinct join trees; equal participant sets share one initialized instance.
  loom_aie2p_native_join_t* first_join;
  // Admitted distribution-path sharing, separate from instruction engines.
  loom_pipeline_execution_t distribution_execution;
  // Physical core engine per worker; autonomous work has IREE_HOST_SIZE_MAX.
  const iree_host_size_t* core_engines;
  // Distribution path per worker; other execution has IREE_HOST_SIZE_MAX.
  const iree_host_size_t* distribution_engines;
  // Proposed transport-worker equalities from shared physical FIFO cursors.
  loom_pipeline_execution_sharing_t* distribution_sharing;
  // Number of retained distribution-sharing constraints.
  iree_host_size_t distribution_sharing_count;
  // Allocated capacity of the distribution-sharing constraint array.
  iree_host_size_t distribution_sharing_capacity;
  // Complete entry program per physical core, initialized and joined once.
  loom_func_like_t* programs;
  // Caller buffer contracts in source argument order.
  loom_aie2p_native_binding_t* bindings;
  // Number of caller buffer contracts.
  iree_host_size_t binding_count;
  // Physical interconnect allocation shared by payload and control routes.
  loom_aie2p_array_route_builder_t routing;
  // Packet identity for every physical route edge.
  loom_aie2p_native_route_t* routes;
  // Complete source ownership per tile's core and tile-control output.
  loom_aie2p_native_control_source_t* control_sources;
  // Unique control source per core/tile-control endpoint. UINT32_MAX is
  // unused; UINT32_MAX - 1 marks an endpoint shared by different sources.
  uint32_t* control_destination_sources;
  // Optional receiver-granted payload path per destination tile. Allocated
  // only when resident task refill needs control routes.
  struct loom_aie2p_native_delivery_t** deliveries;
  // Complete desired routing state for each traversed switch.
  loom_aie2p_native_switch_t* switches;
  // Typed ordinary helper emitter.
  loom_aie2p_worker_builder_t code;
} loom_aie2p_native_context_t;

iree_status_t loom_aie2p_native_reject(
    const loom_aie2p_native_context_t* context, const loom_op_t* op,
    iree_string_view_t requirement);

// Resolves a construction-owned protocol in the admitted execution inventory.
// The channel plan retains this identity through captures and child calls.
static inline loom_aie2p_native_channel_t* loom_aie2p_native_channel(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    const loom_channel_identity_t* identity) {
  return &context->channels[loom_pipeline_invocations_channel_index(
      &realization->invocations, identity)];
}

iree_status_t loom_aie2p_native_emit_worker(
    void* user_data, loom_rewriter_t* rewriter,
    const loom_pipeline_realization_t* realization,
    iree_host_size_t worker_index);
iree_status_t loom_aie2p_native_emit_configuration(
    void* user_data, loom_rewriter_t* rewriter,
    const loom_pipeline_realization_t* realization);

// Builds the physical entries around common retained control projection.
// Child workers return normally; each entry owns private cursor initialization
// and the final completion publication and quiescent wait.
// Unsupported caller captures emit a diagnostic and leave out_valid false.
iree_status_t loom_aie2p_native_emit_programs(
    loom_aie2p_native_context_t* context, loom_rewriter_t* rewriter,
    const loom_pipeline_realization_t* realization, bool* out_valid);

// Selects the lock trees consumed by serial child retirement, using retained
// engine participation and physical resource capacities only.
iree_status_t loom_aie2p_native_select_joins(
    loom_aie2p_native_context_t* context, bool* out_valid);

// Emits the selected collective after this core's child and DMA have retired.
iree_status_t loom_aie2p_native_join_boundary(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    iree_host_size_t engine, const loom_pipeline_invocation_schedule_t* scope);

// Forms one scalar cursor cell in an invocation-private frame. Channel ordinal
// and direction give a stable identity across child calls. Ordinary private
// storage promotion carries these cells through the resulting control graph.
iree_status_t loom_aie2p_native_cursor_view(loom_builder_t* builder,
                                            loom_value_id_t frame,
                                            iree_host_size_t channel,
                                            bool writer,
                                            loom_value_id_t* out_view);

// Advances an admitted FIFO byte offset by one record, wrapping at capacity.
// Retains the ring's range and alignment on the returned current record.
iree_status_t loom_aie2p_native_cursor_advance(
    loom_builder_t* builder, const loom_aie2p_native_channel_t* channel,
    loom_value_id_t origin, loom_value_id_t cursor, loom_location_id_t location,
    loom_value_id_t* out_record, loom_value_id_t* out_next);

// Selects descriptor, engine, route, completion, and relocated-address storage
// from the canonical movement streams before either executable is rewritten.
iree_status_t loom_aie2p_native_select_transfers(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, bool* out_valid);

// Selects on-chip copy programs and binds controller publications to their
// retained ownership intervals before transfer and configuration emission.
iree_status_t loom_aie2p_native_select_worker_distributions(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    iree_host_size_t worker_index, uint32_t* next_source, bool* out_valid);
iree_status_t loom_aie2p_native_bind_distribution_phases(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, bool* out_valid);

iree_status_t loom_aie2p_native_select_control_route(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    iree_host_size_t worker_index, const loom_aie2p_native_tile_t* tile,
    bool reply, uint32_t* next_source, uint8_t* next_controller,
    bool* out_valid);

iree_status_t loom_aie2p_native_distribution_boundary(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    iree_host_size_t worker_index, const loom_pipeline_invocation_t* invocation,
    bool before);

// Shared physical descriptor allocation for external and on-chip transport.
uint16_t* loom_aie2p_native_descriptor_bank(loom_aie2p_native_tile_t* tile,
                                            uint8_t engine, uint16_t* out_base,
                                            uint16_t* out_count);

typedef struct loom_aie2p_native_queue_status_t {
  // Absolute native register address of the selected DMA channel status.
  uint64_t address;
  // Pending queue, running task, and stalled-task bits that must all clear.
  uint32_t pending_mask;
} loom_aie2p_native_queue_status_t;

// Shares the complete retirement condition between configuration and resident
// workers. Payload arrival alone does not cover trailing descriptor updates.
loom_aie2p_native_queue_status_t loom_aie2p_native_queue_status(
    const loom_aie2p_native_context_t* context,
    const loom_aie2p_native_tile_t* tile, uint16_t engine,
    loom_xdna_dma_direction_t direction);

// Selects startup tasks and, when required, a retained reader continuation.
// progress_channel is NULL for queues without an equal-ordinal channel proof,
// including queues whose admission belongs to an ordered distribution phase.
iree_status_t loom_aie2p_native_select_sequence(
    loom_aie2p_native_context_t* context, loom_aie2p_native_tile_t* tile,
    uint8_t engine, loom_xdna_dma_direction_t direction,
    const loom_aie2p_native_channel_t* progress_channel, uint32_t repetitions,
    uint32_t capacity, uint32_t byte_offset, uint32_t byte_stride,
    uint32_t credit_interval, const uint32_t template_words[8],
    loom_aie2p_native_dma_sequence_t* out_sequence,
    iree_string_view_t* out_reason, bool* out_valid);

// Attaches packet identity and retained control ownership to route edges.
// Payload callers use UINT32_MAX for |control_source|.
void loom_aie2p_native_route_label(loom_aie2p_native_context_t* context,
                                   iree_host_size_t first, uint8_t packet,
                                   uint32_t control_source);

// Adds a packet route between admitted endpoints and labels its new edges.
bool loom_aie2p_native_packet_route(loom_aie2p_native_context_t* context,
                                    uint32_t* next_source,
                                    loom_xdna_tile_coordinate_t source,
                                    loom_xdna_stream_port_t source_port,
                                    loom_xdna_tile_coordinate_t destination,
                                    loom_xdna_stream_port_t destination_port,
                                    uint8_t packet);

// Selects and retains a regular ingress stream's external descriptor geometry.
// Stable-source and exact consumer-completion admission precede this query.
bool loom_aie2p_native_select_ingress_descriptor(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_worker_t* source,
    const loom_movement_request_t* request, uint32_t byte_stride,
    uint32_t repetitions, uint32_t period, loom_aie2p_native_worker_t* worker);

// Replaces retained copies/groups/waits using the selected transport helpers.
// Channel access carriers are replaced by the following channel rewrite.
iree_status_t loom_aie2p_native_emit_transfers(
    loom_aie2p_native_context_t* context, loom_rewriter_t* rewriter,
    const loom_pipeline_realization_t* realization,
    iree_host_size_t worker_index);

iree_status_t loom_aie2p_pipeline_realize(loom_pass_t* pass,
                                          loom_module_t* module,
                                          loom_func_like_t function);

// Selects the FIFO reader, relay, scalar carriers, and shared cursor ownership.
iree_status_t loom_aie2p_native_select_collectives(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, bool* out_valid);
// Routes the selected value tuples alongside model and phase-control packets.
iree_status_t loom_aie2p_native_route_collectives(
    loom_aie2p_native_context_t* context, uint32_t* next_source,
    bool* out_valid);
// Acquires the record on its selected reader; peers receive no payload view.
iree_status_t loom_aie2p_native_collective_begin(
    void* user_data, loom_builder_t* builder, iree_host_size_t engine,
    const loom_pipeline_invocation_schedule_t* scope,
    const loom_pipeline_collective_t* collective,
    const loom_value_id_t* state_arguments, loom_type_t payload_type,
    loom_value_id_t* out_payload);
// Broadcasts values, joins the participants, and returns the reader's credit.
iree_status_t loom_aie2p_native_collective_end(
    void* user_data, loom_builder_t* builder, iree_host_size_t engine,
    const loom_pipeline_invocation_schedule_t* scope,
    const loom_pipeline_collective_t* collective, loom_value_id_t* values);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_NATIVE_H_
