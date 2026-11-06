// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compact publication for solved CFG condition relations.

#ifndef LOOM_ANALYSIS_CFG_CONDITION_RELATION_TABLE_H_
#define LOOM_ANALYSIS_CFG_CONDITION_RELATION_TABLE_H_

#include "loom/analysis/cfg_condition_facts.h"
#include "loom/analysis/cfg_condition_operand_domain.h"

#ifdef __cplusplus
extern "C" {
#endif

// Mutable solved facts for one table view during publication.
typedef struct loom_cfg_condition_relation_table_builder_view_t {
  // Scratch-owned integer relation matrix with construction set IDs.
  loom_condition_relation_matrix_t integer_relations;

  // Scratch-owned exact-false and exact-true construction set IDs.
  loom_condition_relation_set_id_t boolean_values[2];
} loom_cfg_condition_relation_table_builder_view_t;

// Scratch-owned inputs for compact immutable table publication.
typedef struct loom_cfg_condition_relation_table_builder_t {
  // Operand domain to copy into the retained arena.
  const loom_cfg_condition_operand_domain_t* operand_domain;

  // Construction set store owning every root in views.
  loom_condition_relation_set_builder_t* set_builder;

  // Optional semantic incidence from derived anchors to relation operands.
  loom_cfg_condition_relation_anchor_builder_t* derived_anchors;

  // Solved block and edge views rewritten with published set IDs.
  loom_cfg_condition_relation_table_builder_view_t* views;

  // View ordinal for each stable CFG edge.
  const uint32_t* edge_view_indices;

  // Number of entries in views.
  uint32_t view_count;

  // Number of leading block views in views.
  uint32_t block_count;

  // Number of entries in edge_view_indices.
  uint32_t edge_count;
} loom_cfg_condition_relation_table_builder_t;

// Publishes solved construction state into a compact immutable table. Rewrites
// construction set IDs in |builder->views| while retaining only query state in
// |arena|. All transient root and serialization storage uses |scratch_arena|.
iree_status_t loom_cfg_condition_relation_table_publish(
    loom_cfg_condition_relation_table_builder_t* builder,
    iree_arena_allocator_t* scratch_arena, iree_arena_allocator_t* arena,
    loom_cfg_condition_relation_table_t* out_table);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_CFG_CONDITION_RELATION_TABLE_H_
