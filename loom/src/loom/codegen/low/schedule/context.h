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
#include "loom/codegen/low/schedule/types.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_LOW_SCHEDULE_EFFECT_MEMORY_SPACE_COUNT \
  ((uint32_t)LOOM_LOW_MEMORY_SPACE_WASM_MEMORY + 1u)

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

typedef struct loom_low_schedule_pressure_cliff_range_t {
  // First pressure cliff row for the register class.
  uint32_t start;
  // Number of pressure cliff rows for the register class.
  uint32_t count;
} loom_low_schedule_pressure_cliff_range_t;

typedef struct loom_low_schedule_state_read_record_t {
  // Node that reads an architectural state register.
  uint32_t node_index;
  // Next outstanding read record for the same descriptor register class.
  uint32_t next_record;
} loom_low_schedule_state_read_record_t;

typedef struct loom_low_schedule_state_chain_read_record_t {
  // Node that reads an architectural state value produced by the key node.
  uint32_t reader_node;
  // Next state-chain read record for the same producer node.
  uint32_t next_record;
} loom_low_schedule_state_chain_read_record_t;

typedef struct loom_low_schedule_storage_read_record_t {
  // Node that reads a value whose storage may later be consumed by a tied op.
  uint32_t reader_node;
  // Register part mask read by reader_node.
  loom_low_register_part_mask_t read_mask;
  // Next outstanding storage-read record for the same value ordinal.
  uint32_t next_record;
} loom_low_schedule_storage_read_record_t;

enum loom_low_schedule_value_flag_bits_e {
  // Value is live in the current simulated block schedule.
  LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE = 1u << 0,
  // Value ordinal is present in storage_read_touched_ordinals.
  LOOM_LOW_SCHEDULE_VALUE_FLAG_STORAGE_READ_TOUCHED = 1u << 1,
};
typedef uint16_t loom_low_schedule_value_flags_t;

typedef struct loom_low_schedule_value_record_t {
  // Module value represented by this local record.
  loom_value_id_t value_id;
  // Same-block producer node index, or NONE for block arguments/external defs.
  uint32_t producer_node;
  // Register units contributed to the pressure model.
  uint32_t unit_count;
  // Remaining operand uses in the current simulated block schedule.
  uint32_t remaining_use_count;
  // Descriptor-set-local register class, or LOOM_LOW_REG_CLASS_NONE.
  uint16_t register_class_id;
  // Mutable per-schedule flags.
  loom_low_schedule_value_flags_t flags;
} loom_low_schedule_value_record_t;

typedef struct loom_low_schedule_build_state_t {
  // Module containing the low function being scheduled.
  loom_module_t* module;
  // Scheduler options provided by the caller.
  const loom_low_schedule_options_t* options;
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
  // Dense per-local-value scheduler records indexed by value ordinal.
  loom_low_schedule_value_record_t* values;
  // Schedule block records indexed by region block ordinal.
  loom_low_schedule_block_t* blocks;
  // Schedule node records indexed by scheduler node ordinal.
  loom_low_schedule_node_t* nodes;
  // Dependency records accumulated while building the schedule DAG.
  loom_low_schedule_dependency_t* dependencies;
  // Open-addressed dependency-set entries storing one-based dependency indices.
  uint32_t* dependency_set_indices;
  // Node indices in final scheduled order.
  uint32_t* scheduled_node_indices;
  // Pressure-model steps in scheduled order for scored strategy runs.
  loom_low_schedule_pressure_step_t* pressure_steps;
  // Candidate decisions in scheduled order when requested.
  loom_low_schedule_candidate_decision_t* candidate_decisions;
  // Descriptor resource uses in scheduled order.
  loom_low_schedule_resource_use_t* resource_uses;
  // Descriptor effects in scheduled order.
  loom_low_schedule_effect_use_t* effect_uses;
  // Descriptor hazard rows in scheduled order.
  loom_low_schedule_hazard_use_t* hazard_uses;
  // Minimum-distance hazard gaps in scheduled order.
  loom_low_schedule_hazard_gap_t* hazard_gaps;
  // Schedule-class model quality summaries in schedule-class order.
  loom_low_schedule_model_summary_t* model_summaries;
  // Per-resource next issue cycle available to descriptor-resource scoring.
  uint32_t* resource_ready_issue_cycles;
  // Earliest issue cycle at which each node's SSA inputs are ready.
  uint32_t* node_ready_issue_cycles;
  // Per-node head of outgoing dependency lists during list scheduling.
  const uint32_t* outgoing_heads;
  // Per-dependency next link for outgoing dependency lists.
  const uint32_t* outgoing_next_indices;
  // Most recent producer state for each minimum-distance hazard key.
  loom_low_schedule_hazard_state_t* hazard_states;
  // Descriptor register-class state read/write bits, dense by register class.
  uint8_t* reg_class_state_flags;
  // Pressure cliff ranges indexed by descriptor register-class ID.
  loom_low_schedule_pressure_cliff_range_t* pressure_cliff_ranges;
  // Most recent architectural-state writer node, dense by register class.
  uint32_t* state_last_write_nodes;
  // Outstanding architectural-state read lists, dense by register class.
  uint32_t* state_read_heads;
  // Outstanding state-read records used by state_read_heads.
  loom_low_schedule_state_read_record_t* state_read_records;
  // Source-order state readers keyed by same-block state producer node.
  uint32_t* state_chain_read_heads;
  // State-chain read records used by state_chain_read_heads.
  loom_low_schedule_state_chain_read_record_t* state_chain_read_records;
  // Per-block readers of values whose storage may be consumed by tied ops.
  struct {
    // Outstanding read lists, dense by local value ordinal.
    uint32_t* heads;
    // Read records used by heads.
    loom_low_schedule_storage_read_record_t* records;
    // Value ordinals whose heads were touched in the current block.
    loom_value_ordinal_t* touched_ordinals;
    // Number of populated read records.
    iree_host_size_t record_count;
    // Allocated read record capacity.
    iree_host_size_t record_capacity;
    // Number of touched value ordinals in the current block.
    iree_host_size_t touched_count;
  } storage_reads;
  // Scratch effect-frontier read node indices, reused for each block.
  uint32_t* effect_read_nodes;
  // Scratch effect-frontier read summaries, parallel to effect_read_nodes.
  loom_low_memory_access_summary_t* effect_read_summaries;
  // Optional source-derived memory access records for the function.
  const loom_low_memory_access_record_t* memory_access_records;
  // Per-resource aggregate resource pressure, dense by descriptor resource id
  // until compacted after scheduling.
  loom_low_schedule_resource_summary_t* resource_summaries;
  // Number of populated dependency records.
  iree_host_size_t dependency_count;
  // Allocated dependency record capacity.
  iree_host_size_t dependency_capacity;
  // Allocated dependency-set entry capacity.
  iree_host_size_t dependency_set_capacity;
  // Number of populated scheduled_node_indices entries.
  iree_host_size_t scheduled_node_count;
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
  // Allocated state-chain read record capacity.
  iree_host_size_t state_chain_read_record_capacity;
  // Allocated effect-frontier read scratch capacity.
  iree_host_size_t effect_read_capacity;
  // Number of rows in |memory_access_records|.
  iree_host_size_t memory_access_record_count;
  // Next memory access record to bind while walking function-order nodes.
  iree_host_size_t memory_access_record_bind_index;
  // Allocated resource-use record capacity.
  iree_host_size_t resource_use_capacity;
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

static inline uint32_t loom_low_schedule_saturating_add_u32(uint32_t lhs,
                                                            uint32_t rhs) {
  if (lhs > UINT32_MAX - rhs) {
    return UINT32_MAX;
  }
  return lhs + rhs;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_CONTEXT_H_
