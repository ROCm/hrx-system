// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Mutable context for one target-low scheduling run.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_CONTEXT_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_CONTEXT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/memory_access.h"
#include "loom/codegen/low/schedule/dependency_index.h"
#include "loom/codegen/low/schedule/resource_calendar.h"
#include "loom/codegen/low/schedule/storage_relation_index.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/util/cfg_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_LOW_SCHEDULE_EFFECT_MEMORY_SPACE_COUNT \
  ((uint32_t)LOOM_LOW_MEMORY_SPACE_WASM_MEMORY + 1u)

#define LOOM_LOW_SCHEDULE_PAIR_AFFINITY_RECORD_NONE UINT32_MAX

typedef struct loom_low_schedule_dependency_endpoint_t {
  // Descriptor-local attachment row, or LOOM_LOW_ID_NONE.
  uint16_t attachment_index;
  // Target timing event observed at this endpoint, or NONE.
  uint16_t timing_event_id;
  // Kind of descriptor attachment named by attachment_index.
  loom_low_schedule_dependency_attachment_kind_t attachment_kind;
} loom_low_schedule_dependency_endpoint_t;

typedef struct loom_low_schedule_state_access_t {
  // Node performing the architectural-state access, or NONE.
  uint32_t node_index;
  // Descriptor attachment carrying the access timing event.
  loom_low_schedule_dependency_endpoint_t endpoint;
} loom_low_schedule_state_access_t;

typedef struct loom_low_schedule_hazard_state_t {
  // Hazard kind tracked by this state.
  loom_low_hazard_kind_t kind;
  // Interpretation of reference_id.
  loom_low_hazard_reference_kind_t reference_kind;
  // Resource, counter, or target-owned hazard identifier.
  uint16_t reference_id;
  // Producer stage published by the most recent matching scheduled hazard.
  uint16_t producer_stage;
  // Region block containing the most recent producer.
  uint32_t block_index;
  // Most recent producer node index.
  uint32_t node_index;
  // Scheduled ordinal of the most recent producer.
  uint32_t scheduled_ordinal;
  // Abstract issue cycle of the most recent producer.
  uint32_t issue_cycle;
  // Hazard row ordinal within the producer node's schedule class.
  uint16_t hazard_ordinal;
  // Required minimum distance published by the producer row.
  uint16_t distance;
  // Hazard flags published by the producer row.
  loom_low_hazard_flags_t hazard_flags;
} loom_low_schedule_hazard_state_t;

typedef struct loom_low_schedule_state_read_record_t {
  // Architectural-state read retained until a later write subsumes it.
  loom_low_schedule_state_access_t access;
  // Next outstanding read record for the same descriptor register class.
  uint32_t next_record;
} loom_low_schedule_state_read_record_t;

typedef struct loom_low_schedule_state_chain_read_record_t {
  // Architectural-state read of the value produced by the key node.
  loom_low_schedule_state_access_t access;
  // Next state-chain read record for the same producer node.
  uint32_t next_record;
} loom_low_schedule_state_chain_read_record_t;

typedef struct loom_low_schedule_pair_affinity_record_t {
  // Descriptor-set ordinal that can be the first visible packet.
  uint32_t first_descriptor_ordinal;
  // Descriptor-set ordinal that can be the second visible packet.
  uint32_t second_descriptor_ordinal;
  // Next pair-affinity record for the same first descriptor.
  uint32_t next_record;
  // Next pair-affinity record for the same second descriptor.
  uint32_t reverse_next_record;
  // Relative benefit for forming this pair.
  uint16_t priority;
  // Index + 1 into the pair-affinity placement recipe table.
  uint16_t placement_recipe_index;
} loom_low_schedule_pair_affinity_record_t;

typedef struct loom_low_schedule_preferred_pair_node_t {
  // Preferred predecessor node, or LOOM_LOW_SCHEDULE_NODE_NONE.
  uint32_t predecessor_node;
  // Preferred successor node, or LOOM_LOW_SCHEDULE_NODE_NONE.
  uint32_t successor_node;
} loom_low_schedule_preferred_pair_node_t;

typedef struct loom_low_schedule_storage_read_record_t {
  // Node that reads a value whose storage may later be consumed by a tied op.
  uint32_t reader_node;
  // First register unit read relative to the tracked value.
  uint32_t unit_offset;
  // Number of contiguous register units read.
  uint32_t unit_count;
  // Register part mask read by reader_node.
  loom_low_register_part_mask_t read_mask;
  // Descriptor-local operand row for the read, or LOOM_LOW_ID_NONE.
  uint16_t descriptor_operand_index;
  // Timing event observed by the read, or LOOM_LOW_TIMING_EVENT_NONE.
  uint16_t timing_event_id;
  // Next outstanding storage-read record for the same value ordinal.
  uint32_t next_record;
} loom_low_schedule_storage_read_record_t;

typedef struct loom_low_schedule_effect_frontier_entry_t {
  // Node carrying the outstanding descriptor effect.
  uint32_t node_index;
  // Descriptor-local effect row, or LOOM_LOW_ID_NONE for structural effects.
  uint16_t effect_ordinal;
  // Timing event observed by the effect, or LOOM_LOW_TIMING_EVENT_NONE.
  uint16_t timing_event_id;
  // Alias summary used to decide which later effects depend on this entry.
  loom_low_memory_access_summary_t summary;
} loom_low_schedule_effect_frontier_entry_t;

typedef struct loom_low_schedule_edge_source_record_t {
  // Structural source value being traced toward a packet producer.
  loom_value_ordinal_t value_ordinal;
  // First relevant unit in value_ordinal.
  uint32_t value_unit_offset;
  // Corresponding first unit in the edge destination value.
  uint32_t destination_unit_offset;
  // Number of storage units mapped by this record.
  uint32_t unit_count;
} loom_low_schedule_edge_source_record_t;

enum loom_low_schedule_value_flag_bits_e {
  // Value is live in the current simulated block schedule.
  LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE = 1u << 0,
  // Value ordinal is present in storage_read_touched_ordinals.
  LOOM_LOW_SCHEDULE_VALUE_FLAG_STORAGE_READ_TOUCHED = 1u << 1,
  // Reads must be tracked because storage can be overwritten in-place.
  LOOM_LOW_SCHEDULE_VALUE_FLAG_STORAGE_READ_TRACKED = 1u << 2,
  // Candidate alias claim scratch has been initialized for this value.
  LOOM_LOW_SCHEDULE_VALUE_FLAG_CANDIDATE_ALIAS_CLAIM = 1u << 3,
  // Value has at least one active incoming pressure alias relation.
  LOOM_LOW_SCHEDULE_VALUE_FLAG_ACTIVE_PRESSURE_ALIAS = 1u << 4,
};
typedef uint16_t loom_low_schedule_value_flags_t;

typedef struct loom_low_schedule_value_record_t {
  // Module value represented by this local record.
  loom_value_id_t value_id;
  // Same-block producer node index, or NONE for block arguments/external defs.
  uint32_t producer_node;
  // First same-class architectural-state writer after the producer.
  loom_low_schedule_state_access_t state_next_write;
  // Register units contributed to the pressure model.
  uint32_t unit_count;
  // Live units currently charged to this value in the pressure model.
  uint32_t live_unit_count;
  // Remaining operand uses in the current simulated block schedule.
  uint32_t remaining_use_count;
  // Descriptor-set-local register class, or LOOM_LOW_REG_CLASS_NONE.
  uint16_t register_class_id;
  // Mutable per-schedule flags.
  loom_low_schedule_value_flags_t flags;
} loom_low_schedule_value_record_t;

typedef struct loom_low_schedule_alias_pressure_limit_t {
  // Hard live-unit limit shared by the alias set.
  uint32_t live_unit_limit;
  // Representative descriptor register class used by diagnostics.
  uint16_t representative_reg_class_id;
} loom_low_schedule_alias_pressure_limit_t;

typedef struct loom_low_schedule_build_state_t {
  // Module containing the low function being scheduled.
  loom_module_t* module;
  // Scheduler options provided by the caller.
  const loom_low_schedule_options_t* options;
  // Direct register-class pressure cliffs from |options|, or NULL.
  const loom_target_residency_direct_resource_table_t* pressure_cliffs;
  // Derived target pressure resources from |options|, or NULL.
  const loom_target_residency_derived_resource_table_t* pressure_resources;
  // Arena owning all table storage produced by this schedule.
  iree_arena_allocator_t* arena;
  // Low function definition operation being scheduled.
  const loom_op_t* function_op;
  // Body region of function_op.
  loom_region_t* body;
  // Resolved target records and descriptor set for the low function.
  loom_low_resolved_target_t target;
  // Descriptor register-class resolver for module register types.
  loom_low_register_type_resolver_t register_type_resolver;
  // Active function-local value domain for this scheduling run.
  const loom_local_value_domain_t* value_domain;
  // Shared read-only control-flow graph for the function body.
  const loom_cfg_graph_t* cfg_graph;
  // Dense per-local-value scheduler records indexed by value ordinal.
  loom_low_schedule_value_record_t* values;
  // Schedule block records indexed by region block ordinal.
  loom_low_schedule_block_t* blocks;
  // Schedule node records indexed by scheduler node ordinal.
  loom_low_schedule_node_t* nodes;
  // Per-block liveness order records populated as nodes are scheduled.
  loom_liveness_block_order_t* liveness_block_orders;
  // Operation pointers in final scheduled order.
  const loom_op_t** scheduled_ops;
  // Function-local storage layout accumulated while populating schedule nodes.
  loom_low_storage_layout_builder_t storage_layout_builder;
  // Stable dependency graph accumulated while building the schedule DAG.
  loom_low_schedule_dependency_graph_t dependencies;
  // Compact verified storage relations grouped by owning schedule node.
  loom_low_schedule_storage_relation_index_t storage_relations;
  // Total storage relations counted while populating schedule nodes.
  iree_host_size_t storage_relation_count;
  // Compact producer/consumer groups used by list scheduling.
  loom_low_schedule_dependency_index_t dependency_index;
  // Node indices in final scheduled order.
  uint32_t* scheduled_node_indices;
  // Contiguous node groups sharing an abstract issue cycle.
  loom_low_schedule_issue_group_t* issue_groups;
  // Pressure-model steps in scheduled order for scored strategy runs.
  loom_low_schedule_pressure_step_t* pressure_steps;
  // Candidate decisions in scheduled order when requested.
  loom_low_schedule_candidate_decision_t* candidate_decisions;
  // Descriptor effects in scheduled order.
  loom_low_schedule_effect_use_t* effect_uses;
  // Descriptor hazard rows in scheduled order.
  loom_low_schedule_hazard_use_t* hazard_uses;
  // Minimum-distance hazard gaps in scheduled order.
  loom_low_schedule_hazard_gap_t* hazard_gaps;
  // Schedule-class model quality summaries in schedule-class order.
  loom_low_schedule_model_summary_t* model_summaries;
  // Capacity-aware resource occupancy for the current scheduled block.
  loom_low_schedule_resource_calendar_t resource_calendar;
  // Earliest issue cycle allowed by each node's latency-bearing dependencies.
  uint32_t* node_ready_issue_cycles;
  // Target-provided issue cost for completion waits required by each node.
  uint16_t* node_completion_wait_cycles;
  // Completion latency opened by counter-tracked memory-effect producers.
  uint16_t* node_opened_completion_latency_cycles;
  // Maximum same-block producer latency consumed by each node.
  uint16_t* node_dependency_latency_cycles;
  // Longest same-block latency path starting at each node.
  uint32_t* node_critical_path_cycles;
  // Downstream visible register demand reached through structural nodes.
  uint32_t* node_pressure_demand_units;
  // Maximum downstream register width needed to advance each node's value.
  uint32_t* node_pressure_activation_units;
  // Downstream activation footprint indexed by schedule node then
  // register-packing resource.
  uint32_t* node_register_packing_activation_units;
  // Earliest downstream packing-resource exit in source order, indexed by
  // schedule node then register-packing resource.
  uint32_t* node_register_packing_completion_sinks;
  // Most recent producer state for each minimum-distance hazard key.
  loom_low_schedule_hazard_state_t* hazard_states;
  // Descriptor register-class state read/write bits, dense by register class.
  uint8_t* reg_class_state_flags;
  // Hard register-pressure limits used while scoring candidates.
  struct {
    // Live-unit limits indexed by descriptor register-class ID. Classes in an
    // alias set use alias_sets instead.
    uint32_t* by_reg_class;
    // Shared limits indexed by one-based register alias-set ID.
    loom_low_schedule_alias_pressure_limit_t* alias_sets;
    // Highest dense one-based alias-set ID, or zero when none are present.
    uint16_t alias_set_count;
  } pressure_limits;
  // Most recent architectural-state writer, dense by register class.
  loom_low_schedule_state_access_t* state_last_writes;
  // First architectural-state writer in the current block, dense by register
  // class.
  loom_low_schedule_state_access_t* state_first_writes;
  // Most recent non-writing state-ordering access, dense by register class.
  loom_low_schedule_state_access_t* state_ordering_frontiers;
  // Outstanding architectural-state read lists, dense by register class.
  uint32_t* state_read_heads;
  // Outstanding state-read records used by state_read_heads.
  loom_low_schedule_state_read_record_t* state_read_records;
  // Source-order state readers keyed by same-block state producer node.
  uint32_t* state_chain_read_heads;
  // State-chain read records used by state_chain_read_heads.
  loom_low_schedule_state_chain_read_record_t* state_chain_read_records;
  // Pair-affinity record heads, dense by first descriptor ordinal.
  uint32_t* pair_affinity_heads;
  // Pair-affinity record heads by second descriptor ordinal when pair setup is
  // present.
  uint32_t* pair_affinity_reverse_heads;
  // Pair-affinity records linked from pair_affinity_heads.
  loom_low_schedule_pair_affinity_record_t* pair_affinity_records;
  // Concrete preferred-pair neighbors indexed by schedule node.
  loom_low_schedule_preferred_pair_node_t* preferred_pair_nodes;
  // Concrete placement-sensitive pair opportunities in scheduled order.
  loom_low_placement_pair_use_t* placement_pair_uses;
  // Reusable descriptor-row projection for one packet's operands.
  struct {
    // Descriptor-local row indices keyed by packet operand index.
    uint16_t* indices;
    // Allocated entries in |indices|.
    iree_host_size_t capacity;
  } descriptor_operands;
  // Per-block readers of values whose storage may be consumed by tied ops.
  struct {
    // Outstanding read lists, dense by local value ordinal.
    uint32_t* heads;
    // Read records used by heads.
    loom_low_schedule_storage_read_record_t* records;
    // Value ordinals whose heads were touched in the current block.
    loom_value_ordinal_t* touched_ordinals;
    // Reusable worklist for edge source ranges composed through aliases.
    loom_low_schedule_edge_source_record_t* edge_source_worklist;
    // Allocated edge-source worklist capacity.
    iree_host_size_t edge_source_worklist_capacity;
    // Reusable source-relation flags indexed by node operand.
    uint8_t* operand_relation_flags;
    // Allocated entries in |operand_relation_flags|.
    iree_host_size_t operand_relation_flag_capacity;
    // Number of populated read records.
    iree_host_size_t record_count;
    // Allocated read record capacity.
    iree_host_size_t record_capacity;
    // Number of touched value ordinals in the current block.
    iree_host_size_t touched_count;
  } storage_reads;
  // Scratch outstanding effect reads, reused for each block.
  loom_low_schedule_effect_frontier_entry_t* effect_read_entries;
  // Scratch outstanding effect writes, reused for each block.
  loom_low_schedule_effect_frontier_entry_t* effect_write_entries;
  // Optional source-derived memory access records for the function.
  const loom_low_memory_access_record_t* memory_access_records;
  // Per-resource aggregate resource pressure, dense by descriptor resource id
  // until compacted after scheduling.
  loom_low_schedule_resource_summary_t* resource_summaries;
  // Issue uses that establish matrix/vector coexecution retention state.
  iree_host_size_t matrix_coexecution_source_use_count;
  // Number of populated scheduled_node_indices entries.
  iree_host_size_t scheduled_node_count;
  // Number of populated issue-group entries.
  iree_host_size_t issue_group_count;
  // Number of error diagnostics emitted while attempting scheduling.
  uint32_t error_count;
  // Terminal hard-scheduling failure recorded during schedule construction.
  loom_low_schedule_failure_t failure;
  // Number of populated pressure_steps entries.
  iree_host_size_t pressure_step_count;
  // Number of populated candidate_decisions entries.
  iree_host_size_t candidate_decision_count;
  // Number of populated resource_uses entries.
  iree_host_size_t resource_use_count;
  // Number of populated effect_uses entries.
  iree_host_size_t effect_use_count;
  // Number of populated hazard_uses entries.
  iree_host_size_t hazard_use_count;
  // Number of populated hazard_gaps entries.
  iree_host_size_t hazard_gap_count;
  // Number of populated model_summaries entries.
  iree_host_size_t model_summary_count;
  // Number of populated hazard_states entries.
  iree_host_size_t hazard_state_count;
  // Number of populated resource_summaries entries after compaction.
  iree_host_size_t resource_summary_count;
  // Number of populated outstanding state-read records.
  iree_host_size_t state_read_record_count;
  // Allocated outstanding state-read record capacity.
  iree_host_size_t state_read_record_capacity;
  // Number of populated state-chain read records.
  iree_host_size_t state_chain_read_record_count;
  // Number of populated pair-affinity records.
  iree_host_size_t pair_affinity_record_count;
  // Number of consumers published to final-producer unlock summaries.
  uint64_t unlock_summary_publication_count;
  // Number of detached transfer nodes that can set up placement-sensitive
  // pairs.
  iree_host_size_t detached_transfer_node_count;
  // Number of populated placement-pair use records.
  iree_host_size_t placement_pair_use_count;
  // Allocated state-chain read record capacity.
  iree_host_size_t state_chain_read_record_capacity;
  // Allocated effect-frontier read scratch capacity.
  iree_host_size_t effect_read_capacity;
  // Allocated effect-frontier write scratch capacity.
  iree_host_size_t effect_write_capacity;
  // Number of rows in |memory_access_records|.
  iree_host_size_t memory_access_record_count;
  // Next memory access record to bind while walking function-order nodes.
  iree_host_size_t memory_access_record_bind_index;
  // Allocated effect-use record capacity.
  iree_host_size_t effect_use_capacity;
  // Allocated hazard-use record capacity.
  iree_host_size_t hazard_use_capacity;
  // Allocated hazard-gap record capacity.
  iree_host_size_t hazard_gap_capacity;
  // Allocated hazard-state record capacity.
  iree_host_size_t hazard_state_capacity;
  // Current block being scheduled.
  uint32_t current_block_index;
  // Current issue cycle within the block being scheduled.
  uint32_t current_issue_cycle;
  // Pending visible descriptor node that can start a target pair.
  uint32_t pending_pair_affinity_node;
} loom_low_schedule_build_state_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_CONTEXT_H_
