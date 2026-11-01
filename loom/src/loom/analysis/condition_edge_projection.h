// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compact transport of condition facts through structured control-flow edges.

#ifndef LOOM_ANALYSIS_CONDITION_EDGE_PROJECTION_H_
#define LOOM_ANALYSIS_CONDITION_EDGE_PROJECTION_H_

#include "loom/analysis/condition_facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// One condition-source value forwarded to a structured successor value.
typedef struct loom_condition_edge_mapping_t {
  // Value available at the source edge terminator.
  loom_value_id_t source;

  // Corresponding successor block argument.
  loom_value_id_t target;
} loom_condition_edge_mapping_t;

// Compact projection of one complete condition derivation through a structured
// control-flow edge. Source facts remain factorized: body arguments map back to
// their edge values at query time instead of materializing the Cartesian
// product of duplicate payload images.
typedef struct loom_condition_edge_projection_t {
  // Complete facts in the source edge's SSA domain.
  loom_condition_derivation_t source_derivation;

  // Module containing source and target values.
  const loom_module_t* module;

  // Region containing definitions local to the source edge.
  const loom_region_t* source_region;

  // Single successor block receiving every mapped target.
  const loom_block_t* target_block;

  // Source value indexed by target block argument ordinal.
  loom_value_id_t* target_sources;

  // Allocated entry count in target_sources.
  iree_host_size_t target_source_capacity;

  // Source-to-target mappings sorted by source and then target.
  loom_condition_edge_mapping_t* mappings;

  // Number of populated entries in mappings and target_sources.
  iree_host_size_t mapping_count;

  // Allocated entry count in mappings.
  iree_host_size_t mapping_capacity;

  // Number of source integer relations observable in the target scope.
  iree_host_size_t visible_integer_relation_count;

  // Number of source Boolean facts observable in the target scope.
  iree_host_size_t visible_boolean_fact_count;
} loom_condition_edge_projection_t;

// Initializes an empty edge projection whose storage grows in |arena|.
void loom_condition_edge_projection_initialize(
    iree_arena_allocator_t* arena,
    loom_condition_edge_projection_t* out_projection);

// Resets source facts and mapping state while retaining high-water storage.
void loom_condition_edge_projection_reset(
    loom_condition_edge_projection_t* projection);

// Updates the exact edge mapping after source_derivation has been populated.
// |sources| follows target block argument order and may contain duplicates.
iree_status_t loom_condition_edge_projection_update_mapping(
    loom_condition_edge_projection_t* projection, const loom_module_t* module,
    const loom_region_t* source_region, const loom_block_t* target_block,
    const loom_value_id_t* sources, iree_host_size_t source_count);

// Returns true when no retained source fact is observable in the target scope.
bool loom_condition_edge_projection_is_empty(
    const loom_condition_edge_projection_t* projection);

// Maps a target-scope operand back into the source derivation's SSA domain.
loom_condition_integer_operand_t loom_condition_edge_projection_source_operand(
    const loom_condition_edge_projection_t* projection,
    loom_condition_integer_operand_t operand);

// Maps a source operand to one canonical value available in the target scope.
// Returns false when a source-local value was not forwarded across the edge.
bool loom_condition_edge_projection_target_operand(
    const loom_condition_edge_projection_t* projection,
    loom_condition_integer_operand_t operand,
    loom_condition_integer_operand_t* out_operand);

// Queries exact Boolean truth through the edge mapping.
bool loom_condition_edge_projection_query_boolean(
    const loom_condition_edge_projection_t* projection,
    loom_value_id_t value_id, bool* out_value);

// Applies projected relations to scalar facts for one target-scope value.
bool loom_condition_edge_projection_apply_to_value_facts(
    const loom_condition_edge_projection_t* projection,
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_value_facts_t* inout_facts);

// Queries possible integer order outcomes in the source derivation after
// mapping target operands through the edge. Contradictory relations return
// zero.
loom_integer_order_outcome_bits_t
loom_condition_edge_projection_query_integer_order(
    const loom_condition_edge_projection_t* projection,
    const loom_value_fact_table_t* fact_table,
    loom_condition_integer_operand_t left,
    loom_condition_integer_operand_t right);

// Proves one target-scope relation from the retained source derivation.
bool loom_condition_edge_projection_proves_integer_relation(
    const loom_condition_edge_projection_t* projection,
    const loom_value_fact_table_t* fact_table,
    const loom_condition_integer_relation_t* queried, bool* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_CONDITION_EDGE_PROJECTION_H_
