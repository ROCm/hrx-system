// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent scheduler table for target-low functions.
//
// This layer consumes ordinary Loom IR plus descriptor tables and produces a
// deterministic schedule table. The default scheduler is intentionally
// conservative: it builds the dependency graph and records a source-priority
// topological order without mutating IR. Optional strategies can score bounded
// windows of dependency-ready nodes for register pressure and target resources
// while still keeping target hazard insertion, allocation, and diagnostics on
// this table instead of creating a second low-level IR container.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_TYPES_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_TYPES_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/codegen/low/descriptor_cost.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/memory_access.h"
#include "loom/codegen/low/placement.h"
#include "loom/codegen/low/schedule/dependencies.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/target/residency.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/cfg_loop.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_allocation_budget_t loom_low_allocation_budget_t;

// Sentinel for absent schedule node indices.
#define LOOM_LOW_SCHEDULE_NODE_NONE UINT32_MAX

typedef enum loom_low_schedule_node_kind_e {
  // Ordinary structural low op such as low.copy, low.move, or low.reload.
  LOOM_LOW_SCHEDULE_NODE_STRUCTURAL = 0,
  // Descriptor-backed packet such as low.op or low.const.
  LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR = 1,
  // Block terminator kept fixed after all schedulable block contents.
  LOOM_LOW_SCHEDULE_NODE_TERMINATOR = 2,
} loom_low_schedule_node_kind_t;

enum loom_low_schedule_node_flag_bits_e {
  // Value ordinals are stored in overflow_value_ordinals instead of
  // inline_value_ordinals.
  LOOM_LOW_SCHEDULE_NODE_FLAG_VALUE_ORDINALS_OVERFLOW = 1u << 0,
  // Node occupies a fixed source-order position between reorderable ranges.
  LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY = 1u << 1,
  // Structural node establishes storage without consuming payload contents.
  LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP = 1u << 2,
  // Node observes completion of externally visible program-exit memory.
  LOOM_LOW_SCHEDULE_NODE_FLAG_PROGRAM_EXIT_MEMORY = 1u << 3,
  // Descriptor results overwrite storage before untied operands are consumed.
  LOOM_LOW_SCHEDULE_NODE_FLAG_EARLY_CLOBBER = 1u << 4,
  // Structural node is transparent between target packet-pair candidates.
  // Its bit immediately precedes DESCRIPTOR_SETUP so candidate scoring can
  // map direct transparency to the common storage-advance fact with one shift.
  LOOM_LOW_SCHEDULE_NODE_FLAG_PAIR_TRANSPARENT = 1u << 5,
  // Structural node advances an SSA storage path into a descriptor operand.
  LOOM_LOW_SCHEDULE_NODE_FLAG_DESCRIPTOR_SETUP =
      LOOM_LOW_SCHEDULE_NODE_FLAG_PAIR_TRANSPARENT << 1u,
};
typedef uint16_t loom_low_schedule_node_flags_t;

#define LOOM_LOW_SCHEDULE_NODE_INLINE_VALUE_ORDINAL_CAPACITY 4

#define LOOM_LOW_SCHEDULE_FAILURE_CYCLE_NODE_CAPACITY 16

typedef enum loom_low_schedule_failure_kind_e {
  // No terminal schedule failure was recorded.
  LOOM_LOW_SCHEDULE_FAILURE_NONE = 0,
  // Remaining same-block dependencies formed a cycle.
  LOOM_LOW_SCHEDULE_FAILURE_DEPENDENCY_CYCLE = 1,
} loom_low_schedule_failure_kind_t;

enum loom_low_schedule_failure_flag_bits_e {
  // The cycle path exceeded the inline diagnostic node capacity.
  LOOM_LOW_SCHEDULE_FAILURE_FLAG_CYCLE_PATH_TRUNCATED = 1u << 0,
  // The scheduler found an unresolved witness edge but not a closed cycle path.
  LOOM_LOW_SCHEDULE_FAILURE_FLAG_WITNESS_EDGE_ONLY = 1u << 1,
};
typedef uint16_t loom_low_schedule_failure_flags_t;

// Terminal schedule failure evidence. This is populated only when
// loom_low_schedule_table_t::error_count is non-zero.
typedef struct loom_low_schedule_failure_t {
  // Recorded terminal schedule failure kind.
  loom_low_schedule_failure_kind_t kind;
  // Additional evidence flags for the recorded failure.
  loom_low_schedule_failure_flags_t flags;
  // Region block containing the scheduler failure.
  uint32_t block_index;
  // Number of nodes owned by the failed block.
  uint32_t block_node_count;
  // Number of nodes scheduled in the failed block before progress stopped.
  uint32_t scheduled_node_count;
  // Number of unscheduled nodes remaining in the failed block.
  uint32_t unscheduled_node_count;
  // Producer node for the representative unresolved dependency edge.
  uint32_t producer_node;
  // Consumer node for the representative unresolved dependency edge.
  uint32_t consumer_node;
  // Dependency kind for the representative unresolved edge.
  loom_low_schedule_dependency_kind_t dependency_kind;
  // Operand index for the representative edge, or UINT32_MAX.
  uint32_t operand_index;
  // Architectural-state SSA value read across a clobber edge in the cycle, or
  // LOOM_VALUE_ID_INVALID when the cycle has no explicit state-value witness.
  loom_value_id_t state_value_id;
  // Inline same-block cycle node path. When non-empty, the last node has a
  // dependency edge back to the first node.
  uint32_t cycle_nodes[LOOM_LOW_SCHEDULE_FAILURE_CYCLE_NODE_CAPACITY];
  // Number of populated entries in |cycle_nodes|.
  uint32_t cycle_node_count;
} loom_low_schedule_failure_t;

static inline bool loom_low_schedule_failure_is_present(
    const loom_low_schedule_failure_t* failure) {
  return failure && failure->kind != LOOM_LOW_SCHEDULE_FAILURE_NONE;
}

enum loom_low_schedule_diagnostic_bits_e {
  // Emits one BACKEND/003 remark per hard-bounded register-pressure summary.
  LOOM_LOW_SCHEDULE_DIAGNOSTIC_PRESSURE_PEAKS = 1u << 0,
  // Emits BACKEND/013 remarks for resources tied at the schedule bottleneck.
  LOOM_LOW_SCHEDULE_DIAGNOSTIC_RESOURCE_BOTTLENECKS = 1u << 1,
  // Emits BACKEND/014 remarks for required delay/wait hazard gaps.
  LOOM_LOW_SCHEDULE_DIAGNOSTIC_HAZARD_GAPS = 1u << 2,
  // Emits BACKEND/015 remarks for pressure-scheduler candidate choices.
  LOOM_LOW_SCHEDULE_DIAGNOSTIC_CANDIDATE_DECISIONS = 1u << 3,
  // Emits BACKEND/016 remarks for non-exact schedule model quality.
  LOOM_LOW_SCHEDULE_DIAGNOSTIC_MODEL_QUALITY = 1u << 4,
};
typedef uint32_t loom_low_schedule_diagnostic_flags_t;

enum loom_low_schedule_flag_bits_e {
  // Retains source-order liveness analysis in the returned schedule table.
  LOOM_LOW_SCHEDULE_FLAG_RETAIN_LIVENESS = 1u << 0,
  // Retains per-node pressure-model steps for detailed schedule inspection.
  LOOM_LOW_SCHEDULE_FLAG_RETAIN_PRESSURE_STEPS = 1u << 1,
};
typedef uint32_t loom_low_schedule_flags_t;

typedef enum loom_low_schedule_strategy_e {
  // Chooses the first ready node in source order.
  LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY = 0,
  // Chooses ready nodes using a target-independent register-pressure score.
  LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE = 1,
  // Chooses ready nodes using latency hiding before register-pressure ties.
  LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING = 2,
  // Chooses ready nodes using descriptor resource/hazard stall estimates.
  LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL = 3,
} loom_low_schedule_strategy_t;

#define LOOM_LOW_SCHEDULE_MEMORY_ACCESS_RECORD_NONE UINT32_MAX
#define LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE UINT32_MAX

// One target-provided pair-affinity row.
//
// These rows are optimistic scheduling hints. The generic scheduler can place
// pair-compatible descriptors adjacent, including through transparent
// structural ops, but final target packetization remains responsible for exact
// legality such as register banks, tied operands, literal payloads, and whether
// a structural packet truly emits no native instruction after allocation.
typedef struct loom_low_schedule_pair_affinity_t {
  // Descriptor that can be the first visible packet in a pair.
  const loom_low_descriptor_t* first_descriptor;
  // Descriptor that can be the second visible packet in a pair.
  const loom_low_descriptor_t* second_descriptor;
  // Relative benefit for forming this pair. Zero disables the row.
  uint16_t priority;
  // Index + 1 into the list placement_recipes table, or
  // LOOM_LOW_PLACEMENT_PAIR_RECIPE_NONE.
  uint16_t placement_recipe_index;
} loom_low_schedule_pair_affinity_t;

// List of target-provided pair-affinity rows.
typedef struct loom_low_schedule_pair_affinity_list_t {
  // Borrowed pair-affinity rows.
  const loom_low_schedule_pair_affinity_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
  // Borrowed target-provided placement recipes referenced by values.
  const loom_low_placement_pair_recipe_t* placement_recipes;
  // Number of entries in placement_recipes.
  iree_host_size_t placement_recipe_count;
} loom_low_schedule_pair_affinity_list_t;

static inline loom_low_schedule_pair_affinity_list_t
loom_low_schedule_pair_affinity_list_empty(void) {
  return (loom_low_schedule_pair_affinity_list_t){0};
}

static inline bool loom_low_schedule_pair_affinity_list_is_empty(
    loom_low_schedule_pair_affinity_list_t list) {
  return list.count == 0;
}

// Target state dependency for structural low materializations.
typedef struct loom_low_schedule_structural_state_read_t {
  // Register class written by a structural low materialization.
  uint16_t result_reg_class_id;
  // Architectural state register class read by the materialization.
  uint16_t state_reg_class_id;
} loom_low_schedule_structural_state_read_t;

// List of target-provided structural state-read rows.
typedef struct loom_low_schedule_structural_state_read_list_t {
  // Borrowed structural state-read rows.
  const loom_low_schedule_structural_state_read_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
} loom_low_schedule_structural_state_read_list_t;

static inline loom_low_schedule_structural_state_read_list_t
loom_low_schedule_structural_state_read_list_empty(void) {
  return (loom_low_schedule_structural_state_read_list_t){0};
}

static inline bool loom_low_schedule_structural_state_read_list_is_empty(
    loom_low_schedule_structural_state_read_list_t list) {
  return list.count == 0;
}

// One scheduled operation in a low function body.
typedef struct loom_low_schedule_node_t {
  // Operation represented by this node.
  const loom_op_t* op;
  // Block containing |op|.
  const loom_block_t* block;
  // Descriptor row for descriptor-backed nodes, or NULL.
  const loom_low_descriptor_t* descriptor;
  // Schedule-class row for descriptor-backed nodes, or NULL.
  const loom_low_schedule_class_t* schedule_class;
  // Region block ordinal containing |op|.
  uint32_t block_index;
  // Source-order ordinal within the whole low function body.
  uint32_t source_ordinal;
  // Scheduled ordinal within |block| after topological scheduling.
  uint32_t scheduled_ordinal;
  // Source memory-access record attached to this node, or NONE.
  uint32_t memory_access_record_index;
  // Effective traits used for conservative structural ordering.
  loom_trait_flags_t traits;
  // Kind of schedule node.
  loom_low_schedule_node_kind_t kind;
  // Number of operand value ordinals.
  uint16_t operand_count;
  // Number of result value ordinals.
  uint16_t result_count;
  // Number of indexed storage relations owned by this node.
  uint16_t storage_relation_count;
  // Per-node storage flags.
  loom_low_schedule_node_flags_t flags;
  // Operand ordinals followed by result ordinals. Small nodes store ordinals
  // inline to avoid an extra pointer chase; large nodes store one contiguous
  // arena allocation through overflow_value_ordinals.
  union {
    // Inline operand/result ordinals for the common low-op arity.
    loom_value_ordinal_t inline_value_ordinals
        [LOOM_LOW_SCHEDULE_NODE_INLINE_VALUE_ORDINAL_CAPACITY];
    // Arena-owned overflow operand/result ordinal payload.
    loom_value_ordinal_t* overflow_value_ordinals;
  } value_ordinals;
} loom_low_schedule_node_t;

static inline loom_value_ordinal_t* loom_low_schedule_node_value_ordinals(
    loom_low_schedule_node_t* node) {
  if (iree_any_bit_set(node->flags,
                       LOOM_LOW_SCHEDULE_NODE_FLAG_VALUE_ORDINALS_OVERFLOW)) {
    return node->value_ordinals.overflow_value_ordinals;
  }
  return node->value_ordinals.inline_value_ordinals;
}

static inline const loom_value_ordinal_t*
loom_low_schedule_node_const_value_ordinals(
    const loom_low_schedule_node_t* node) {
  if (iree_any_bit_set(node->flags,
                       LOOM_LOW_SCHEDULE_NODE_FLAG_VALUE_ORDINALS_OVERFLOW)) {
    return node->value_ordinals.overflow_value_ordinals;
  }
  return node->value_ordinals.inline_value_ordinals;
}

static inline loom_value_ordinal_t* loom_low_schedule_node_operand_ordinals(
    loom_low_schedule_node_t* node) {
  return loom_low_schedule_node_value_ordinals(node);
}

static inline const loom_value_ordinal_t*
loom_low_schedule_node_const_operand_ordinals(
    const loom_low_schedule_node_t* node) {
  return loom_low_schedule_node_const_value_ordinals(node);
}

static inline loom_value_ordinal_t* loom_low_schedule_node_result_ordinals(
    loom_low_schedule_node_t* node) {
  return loom_low_schedule_node_value_ordinals(node) + node->operand_count;
}

static inline const loom_value_ordinal_t*
loom_low_schedule_node_const_result_ordinals(
    const loom_low_schedule_node_t* node) {
  return loom_low_schedule_node_const_value_ordinals(node) +
         node->operand_count;
}

// Pressure-model step recorded while scheduling one node. This is an aggregate
// target-independent register-pressure estimate across all register classes,
// not a replacement for source-order liveness or target occupancy analysis.
typedef struct loom_low_schedule_pressure_step_t {
  // Scheduled node represented by this step.
  uint32_t node_index;
  // Region block containing |node_index|.
  uint32_t block_index;
  // Scheduled ordinal within |block_index|.
  uint32_t scheduled_ordinal;
  // Aggregate register live units before scheduling the node.
  uint64_t live_units_before;
  // Register live units killed by the node.
  uint64_t killed_live_units;
  // Register live units produced by the node.
  uint64_t produced_live_units;
  // Aggregate register live units after scheduling the node.
  uint64_t live_units_after;
} loom_low_schedule_pressure_step_t;

// Scheduler candidate decision recorded when a ready set has an alternative to
// reject. This is intentionally compact: it captures the chosen candidate and
// the best rejected alternative, which is enough to explain local latency and
// pressure tradeoffs without recording every ready-set member in large blocks.
typedef struct loom_low_schedule_candidate_decision_t {
  // Region block containing the decision.
  uint32_t block_index;
  // Scheduled ordinal within |block_index|.
  uint32_t scheduled_ordinal;
  // Number of dependency-ready candidates at this ordinal.
  uint32_t ready_candidate_count;
  // Number of ready candidates selected for exact scoring.
  uint32_t scored_candidate_count;
  // Chosen schedule node.
  uint32_t chosen_node;
  // Best rejected schedule node, or LOOM_LOW_SCHEDULE_NODE_NONE.
  uint32_t rejected_node;
  // Chosen maximum same-block producer latency among SSA operands.
  uint16_t chosen_dependency_latency_cycles;
  // Chosen descriptor latency in cycles.
  uint16_t chosen_latency_cycles;
  // Chosen target pair-affinity score.
  uint16_t chosen_pair_affinity_score;
  // Best rejected maximum same-block producer latency among SSA operands.
  uint16_t rejected_dependency_latency_cycles;
  // Best rejected descriptor latency in cycles.
  uint16_t rejected_latency_cycles;
  // Best rejected target pair-affinity score.
  uint16_t rejected_pair_affinity_score;
  // Chosen aggregate live register units after scheduling the node.
  uint64_t chosen_projected_live_units;
  // Chosen live register units killed by scheduling the node.
  uint64_t chosen_killed_live_units;
  // Chosen live register units produced by scheduling the node.
  uint64_t chosen_produced_live_units;
  // Best rejected aggregate live register units after scheduling the node.
  uint64_t rejected_projected_live_units;
  // Best rejected live register units killed by scheduling the node.
  uint64_t rejected_killed_live_units;
  // Best rejected live register units produced by scheduling the node.
  uint64_t rejected_produced_live_units;
  // Chosen cycles until all latency-bearing dependencies are ready.
  uint32_t chosen_data_ready_stall_cycles;
  // Chosen cycles blocked by descriptor resource occupancy.
  uint32_t chosen_resource_stall_cycles;
  // Chosen cycles blocked by target hazard distance rows.
  uint32_t chosen_hazard_stall_cycles;
  // Chosen target-provided issue cost for a completion wait.
  uint32_t chosen_completion_wait_cycles;
  // Chosen maximum stall across dependencies, resources, hazards, and
  // completion waits.
  uint32_t chosen_effective_stall_cycles;
  // Target resource table identifier causing the chosen resource stall, or
  // LOOM_LOW_RESOURCE_NONE.
  uint16_t chosen_bottleneck_resource_id;
  // Chosen target pressure-cliff penalty.
  uint32_t chosen_pressure_cliff_penalty;
  // Chosen crossed-cliff source, or closest upcoming source when no cliff was
  // crossed.
  iree_string_view_t chosen_pressure_cliff_source;
  // Chosen crossed pressure cliff, or LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE.
  uint32_t chosen_pressure_cliff_units;
  // Chosen live units remaining before the next pressure cliff when no cliff
  // was crossed, or LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE.
  uint32_t chosen_units_until_pressure_cliff;
  // Best rejected cycles until all latency-bearing dependencies are ready.
  uint32_t rejected_data_ready_stall_cycles;
  // Best rejected cycles blocked by descriptor resource occupancy.
  uint32_t rejected_resource_stall_cycles;
  // Best rejected cycles blocked by target hazard distance rows.
  uint32_t rejected_hazard_stall_cycles;
  // Best rejected target-provided issue cost for a completion wait.
  uint32_t rejected_completion_wait_cycles;
  // Best rejected maximum stall across dependencies, resources, hazards, and
  // completion waits.
  uint32_t rejected_effective_stall_cycles;
  // Target resource table identifier causing the rejected resource stall, or
  // LOOM_LOW_RESOURCE_NONE.
  uint16_t rejected_bottleneck_resource_id;
  // Best rejected target pressure-cliff penalty.
  uint32_t rejected_pressure_cliff_penalty;
  // Best rejected crossed-cliff source, or closest upcoming source when no
  // cliff was crossed.
  iree_string_view_t rejected_pressure_cliff_source;
  // Best rejected crossed pressure cliff, or
  // LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE.
  uint32_t rejected_pressure_cliff_units;
  // Best rejected live units remaining before the next pressure cliff when no
  // cliff was crossed, or LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE.
  uint32_t rejected_units_until_pressure_cliff;
} loom_low_schedule_candidate_decision_t;

// Descriptor effect row recorded in scheduled order. Effects describe memory,
// counter, call, barrier, and control behavior used by dependency construction
// and target-owned wait and hazard planning.
typedef struct loom_low_schedule_effect_use_t {
  // Scheduled node carrying the effect row.
  uint32_t node_index;
  // Region block containing |node_index|.
  uint32_t block_index;
  // Scheduled ordinal within |block_index|.
  uint32_t scheduled_ordinal;
  // Effect row ordinal within the node's descriptor.
  uint16_t effect_ordinal;
  // Effect kind used by dependency and legality construction.
  loom_low_effect_kind_t kind;
  // Memory space or external resource touched by the effect.
  loom_low_memory_space_t memory_space;
  // Target-owned scope identifier for ordering and visibility.
  uint16_t scope_id;
  // Effect flags used by scheduling and verification.
  loom_low_effect_flags_t effect_flags;
  // Target counter identifier for counter effects, or a target-owned
  // completion-counter override for memory effects.
  uint16_t counter_id;
  // Access width in bits, or zero when not width-specific.
  uint16_t width_bits;
} loom_low_schedule_effect_use_t;

// Descriptor hazard row recorded in scheduled order. These rows are passive
// facts: target overlays consume them to insert waits, enforce distances, or
// record richer backend diagnostics.
typedef struct loom_low_schedule_hazard_use_t {
  // Scheduled node carrying the hazard row.
  uint32_t node_index;
  // Region block containing |node_index|.
  uint32_t block_index;
  // Scheduled ordinal within |block_index|.
  uint32_t scheduled_ordinal;
  // Hazard row ordinal within the node's schedule class.
  uint16_t hazard_ordinal;
  // Hazard kind used by schedule policy and verification.
  loom_low_hazard_kind_t kind;
  // Interpretation of reference_id.
  loom_low_hazard_reference_kind_t reference_kind;
  // Resource, counter, or target-owned hazard identifier.
  uint16_t reference_id;
  // Borrowed stable resource name when reference_kind is RESOURCE.
  iree_string_view_t resource_name;
  // Producer pipeline stage participating in the hazard.
  uint16_t producer_stage;
  // Consumer pipeline stage participating in the hazard.
  uint16_t consumer_stage;
  // Required distance or target-owned hazard value.
  uint16_t distance;
  // Hazard flags for target-owned refinements.
  loom_low_hazard_flags_t hazard_flags;
} loom_low_schedule_hazard_use_t;

// Minimum-distance hazard gap recorded after scheduling. These rows identify
// where the chosen order needs an abstract delay/wait before final emission.
// They are not inserted operations; target overlays decide whether a gap
// becomes a wait packet, a delay packet, or a diagnostic.
typedef struct loom_low_schedule_hazard_gap_t {
  // Producer node carrying the previous hazard row.
  uint32_t producer_node;
  // Consumer node carrying the hazard row that requires additional distance.
  uint32_t consumer_node;
  // Region block containing both nodes.
  uint32_t block_index;
  // Scheduled ordinal of the producer node within |block_index|.
  uint32_t producer_scheduled_ordinal;
  // Scheduled ordinal of the consumer node within |block_index|.
  uint32_t consumer_scheduled_ordinal;
  // Hazard row ordinal within the producer node's schedule class.
  uint16_t producer_hazard_ordinal;
  // Hazard row ordinal within the consumer node's schedule class.
  uint16_t consumer_hazard_ordinal;
  // Hazard kind that produced this gap.
  loom_low_hazard_kind_t kind;
  // Interpretation of reference_id.
  loom_low_hazard_reference_kind_t reference_kind;
  // Resource, counter, or target-owned hazard identifier.
  uint16_t reference_id;
  // Borrowed stable resource name when reference_kind is RESOURCE.
  iree_string_view_t resource_name;
  // Producer pipeline stage participating in the hazard.
  uint16_t producer_stage;
  // Consumer pipeline stage participating in the hazard.
  uint16_t consumer_stage;
  // Required minimum distance in abstract issue slots.
  uint16_t required_distance;
  // Actual scheduled distance in abstract issue slots.
  uint32_t actual_distance;
  // Additional abstract issue slots needed before the consumer.
  uint16_t required_delay;
  // Hazard flags for target-owned refinements.
  loom_low_hazard_flags_t hazard_flags;
} loom_low_schedule_hazard_gap_t;

// Schedule-class model quality summary for schedule classes used by this
// function. These summaries make model uncertainty visible without repeating
// the same schedule-class facts on every packet.
typedef struct loom_low_schedule_model_summary_t {
  // Schedule node where this class first appears, or
  // LOOM_LOW_SCHEDULE_NODE_NONE.
  uint32_t first_node;
  // Target descriptor schedule-class identifier.
  uint32_t schedule_class_id;
  // Borrowed stable schedule-class name.
  iree_string_view_t schedule_class_name;
  // Descriptor schedule latency in cycles.
  uint16_t latency_cycles;
  // Descriptor latency interpretation.
  loom_low_latency_kind_t latency_kind;
  // Descriptor schedule-model quality.
  loom_low_model_quality_t model_quality;
  // Number of issue-resource rows consumed by the schedule class.
  uint16_t issue_use_count;
  // Number of hazard rows attached to the schedule class.
  uint16_t hazard_count;
  // Number of scheduled nodes using this schedule class.
  uint32_t use_count;
} loom_low_schedule_model_summary_t;

// Schedule metadata for one low function block.
typedef struct loom_low_schedule_block_t {
  // Region block represented by this record.
  const loom_block_t* block;
  // First source-order node index owned by this block.
  uint32_t node_start;
  // Number of nodes owned by this block.
  uint32_t node_count;
  // First entry in the table scheduled-node-index array.
  uint32_t scheduled_node_start;
  // Number of scheduled-node-index entries owned by this block.
  uint32_t scheduled_node_count;
} loom_low_schedule_block_t;

// Options controlling low schedule construction.
typedef struct loom_low_schedule_options_t {
  // Optional source-derived memory summaries for the modeled function. Empty
  // uses conservative descriptor effect summaries.
  loom_low_memory_access_table_t memory_access_table;
  // Optional immutable target residency policy.
  const loom_target_residency_model_t* residency_model;
  // Optional explicit allocation budgets. These are interpreted as hard
  // pressure limits by the scheduler so resource-stall scheduling can shorten
  // live ranges before allocation reaches the final physical storage ceiling.
  const loom_low_allocation_budget_t* allocation_budgets;
  // Number of entries in |allocation_budgets|.
  iree_host_size_t allocation_budget_count;
  // Optional target-provided pair-affinity table.
  loom_low_schedule_pair_affinity_list_t pair_affinities;
  // Optional concrete pair groups preferred when rescheduling rewritten IR.
  loom_low_placement_pair_use_list_t preferred_pair_uses;
  // Optional target-provided implicit state reads for structural low
  // materializations that emit target packets without descriptor rows.
  loom_low_schedule_structural_state_read_list_t structural_state_reads;
  // Structured diagnostic emitter for user IR failures.
  iree_diagnostic_emitter_t emitter;
  // Optional backend feedback diagnostics to emit after scheduling analysis.
  loom_low_schedule_diagnostic_flags_t diagnostic_flags;
  // Schedule construction behavior flags.
  loom_low_schedule_flags_t flags;
  // Candidate selection strategy used within each dependency-ready set.
  loom_low_schedule_strategy_t strategy;
} loom_low_schedule_options_t;

// Schedule table for one target-low function body. All arrays are arena-owned
// by the caller-provided arena passed to loom_low_schedule_function.
typedef struct loom_low_schedule_table_t {
  // Module containing the scheduled low function.
  const loom_module_t* module;
  // Target-low function operation scheduled by this table.
  const loom_op_t* function_op;
  // Resolved target context selected by |function_op|.
  loom_low_resolved_target_t target;
  // Borrowed source-derived memory summaries attached to scheduled nodes.
  loom_low_memory_access_table_t memory_access_table;
  // Function-local storage reservations packed during source node collection.
  loom_low_storage_layout_t storage_layout;
  // Function-local value IDs indexed by local value ordinal.
  const loom_value_id_t* value_ids;
  // Number of entries in |value_ids|.
  loom_value_ordinal_t value_count;
  // Optional liveness analysis retained for table consumers that request it.
  loom_liveness_analysis_t liveness;
  // Per-block schedule records in region block order.
  const loom_low_schedule_block_t* blocks;
  // Number of block records.
  iree_host_size_t block_count;
  // Final top-level operation order retained for downstream liveness analysis.
  loom_liveness_order_t operation_order;
  // Read-only control-flow graph shared by target planning overlays.
  loom_cfg_graph_t cfg_graph;
  // Canonical loop intervals shared by target planning overlays.
  loom_cfg_loop_forest_t loop_forest;
  // Per-op schedule nodes in source order.
  const loom_low_schedule_node_t* nodes;
  // Number of schedule nodes.
  iree_host_size_t node_count;
  // Stable ordering dependency graph consumed by scheduling and target
  // planning.
  loom_low_schedule_dependency_graph_t dependencies;
  // Number of distinct producer-to-consumer dependency groups used by list
  // scheduling.
  uint32_t dependency_group_count;
  // Number of consumers published to their final dependency producer while
  // maintaining pressure summaries.
  uint64_t unlock_summary_publication_count;
  // Node indices in scheduled order, grouped by block.
  const uint32_t* scheduled_node_indices;
  // Number of scheduled node indices.
  iree_host_size_t scheduled_node_count;
  // Concrete placement-sensitive pair opportunities in scheduled order.
  loom_low_placement_pair_use_list_t placement_pair_uses;
  // Number of error diagnostics emitted while attempting scheduling.
  uint32_t error_count;
  // Terminal hard-scheduling failure when |error_count| is non-zero.
  loom_low_schedule_failure_t failure;
  // Pressure-model steps in scheduled order when explicitly retained for a
  // scored strategy. Empty unless RETAIN_PRESSURE_STEPS was requested.
  const loom_low_schedule_pressure_step_t* pressure_steps;
  // Number of pressure-model steps.
  iree_host_size_t pressure_step_count;
  // Candidate decisions in scheduled order when requested by diagnostic flags.
  // Empty for source-priority scheduling and scored scheduling without
  // candidate diagnostics.
  const loom_low_schedule_candidate_decision_t* candidate_decisions;
  // Number of candidate decision records.
  iree_host_size_t candidate_decision_count;
  // Number of descriptor issue-use rows referenced by scheduled nodes.
  iree_host_size_t resource_use_count;
  // Issue uses that establish matrix/vector coexecution retention state.
  iree_host_size_t matrix_coexecution_source_use_count;
  // Descriptor effects in scheduled order.
  const loom_low_schedule_effect_use_t* effect_uses;
  // Number of effect-use records.
  iree_host_size_t effect_use_count;
  // Descriptor hazards in scheduled order. Empty when scheduled nodes do not
  // reference descriptor hazard rows.
  const loom_low_schedule_hazard_use_t* hazard_uses;
  // Number of hazard-use records.
  iree_host_size_t hazard_use_count;
  // Minimum-distance hazard gaps in scheduled order.
  const loom_low_schedule_hazard_gap_t* hazard_gaps;
  // Number of minimum-distance hazard gaps.
  iree_host_size_t hazard_gap_count;
  // Schedule-class model-quality summaries in descriptor schedule-class order.
  const loom_low_schedule_model_summary_t* model_summaries;
  // Number of schedule-class model-quality summaries.
  iree_host_size_t model_summary_count;
  // Per-resource aggregate schedule pressure in resource-id order.
  const loom_low_descriptor_resource_cost_t* resource_summaries;
  // Number of resource summary records.
  iree_host_size_t resource_summary_count;
} loom_low_schedule_table_t;

// Returns the source-order schedule node for |op|, or NULL when |op| does not
// belong to |schedule|. The returned node retains its final scheduled ordinal.
static inline const loom_low_schedule_node_t* loom_low_schedule_node_for_op(
    const loom_low_schedule_table_t* schedule, const loom_op_t* op) {
  if (schedule == NULL || op == NULL || op->parent_block == NULL ||
      iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    return NULL;
  }
  const loom_block_t* block = op->parent_block;
  const uint16_t block_index = block->region_index;
  if (block_index >= schedule->block_count ||
      schedule->blocks[block_index].block != block) {
    return NULL;
  }
  const loom_low_schedule_block_t* block_record =
      &schedule->blocks[block_index];
  uint32_t begin = block_record->node_start;
  uint32_t end = begin + block_record->node_count;
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2u;
    const loom_op_t* candidate_op = schedule->nodes[middle].op;
    if (candidate_op->block_ordinal < op->block_ordinal) {
      begin = middle + 1u;
    } else {
      end = middle;
    }
  }
  const uint32_t block_end =
      block_record->node_start + block_record->node_count;
  return begin < block_end && schedule->nodes[begin].op == op
             ? &schedule->nodes[begin]
             : NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_TYPES_H_
