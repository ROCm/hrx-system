// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Function-local storage placement relations for target-low allocation.
//
// Placement is the target-independent producer of storage-affinity facts. It
// records the IR-level relationships that make two low values profitable or
// mandatory to place in the same concrete storage without deciding where that
// storage is. Allocation consumes this table when assigning locations; move
// sequencing consumes the allocation result after unavoidable copies remain.

#ifndef LOOM_CODEGEN_LOW_PLACEMENT_H_
#define LOOM_CODEGEN_LOW_PLACEMENT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/liveness.h"
#include "loom/ir/ir.h"
#include "loom/ir/local_value_domain.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_low_placement_relation_kind_e {
  // Unknown or uninitialized placement relation kind.
  LOOM_LOW_PLACEMENT_RELATION_UNKNOWN = 0,
  // Result and source unit ranges should occupy identical storage units.
  LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE = 1,
  // Result units should occupy a subrange of the source storage units.
  LOOM_LOW_PLACEMENT_RELATION_SUBRANGE = 2,
  // Result units should occupy a contiguous packed range of source values.
  LOOM_LOW_PLACEMENT_RELATION_CONTIGUOUS_PART = 3,
} loom_low_placement_relation_kind_t;

typedef enum loom_low_placement_cause_e {
  // Unknown or uninitialized placement cause.
  LOOM_LOW_PLACEMENT_CAUSE_UNKNOWN = 0,
  // Descriptor tied result requiring source/result storage identity.
  LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT = 1,
  // low.copy source/result storage affinity.
  LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY = 2,
  // low.slice source/result subrange affinity.
  LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE = 3,
  // low.concat source/result contiguous packing affinity.
  LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT = 4,
  // low.br edge payload source/block-argument affinity.
  LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH = 5,
} loom_low_placement_cause_t;

enum loom_low_placement_relation_flag_bits_e {
  // The relation is required for the selected target operation semantics.
  LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD = 1u << 0,
  // The relation removes a move when allocation can satisfy it.
  LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED = 1u << 1,
};

// Bitset of loom_low_placement_relation_flag_bits_e values.
typedef uint16_t loom_low_placement_relation_flags_t;

// One directional placement relation keyed by result and source value ordinals.
typedef struct loom_low_placement_relation_t {
  // Operation that introduced this relation.
  const loom_op_t* op;
  // Result or destination value ordinal receiving preferred storage.
  loom_value_ordinal_t result_ordinal;
  // Source value ordinal providing preferred storage.
  loom_value_ordinal_t source_ordinal;
  // Unit offset inside the result assignment.
  uint32_t result_unit_offset;
  // Unit offset inside the source assignment.
  uint32_t source_unit_offset;
  // Number of units covered by this relation.
  uint32_t unit_count;
  // Structural relation shape.
  loom_low_placement_relation_kind_t kind;
  // IR feature that created the relation.
  loom_low_placement_cause_t cause;
  // Hard/soft relation behavior.
  loom_low_placement_relation_flags_t flags;
} loom_low_placement_relation_t;

// Contiguous relation range for one result value ordinal.
typedef struct loom_low_placement_relation_range_t {
  // First relation index for the value ordinal.
  uint32_t start;
  // Number of relation records for the value ordinal.
  uint32_t count;
} loom_low_placement_relation_range_t;

// Placement analysis table for one target-low function body.
typedef struct loom_low_placement_table_t {
  // Module containing the analyzed low function.
  const loom_module_t* module;
  // Region analyzed.
  const loom_region_t* region;
  // Local value IDs indexed by placement value ordinal.
  const loom_value_id_t* value_ids;
  // Number of local value IDs.
  loom_value_ordinal_t value_count;
  // Placement relations grouped by result value ordinal.
  const loom_low_placement_relation_t* relations;
  // Number of relation records.
  iree_host_size_t relation_count;
  // Relation ranges into |relations| indexed by result value ordinal.
  const loom_low_placement_relation_range_t* ranges_by_result_ordinal;
  // Relation indices grouped by source value ordinal. Each entry indexes
  // |relations|.
  const uint32_t* relation_indices_by_source_ordinal;
  // Relation ranges into |relation_indices_by_source_ordinal| indexed by source
  // value ordinal.
  const loom_low_placement_relation_range_t* ranges_by_source_ordinal;
} loom_low_placement_table_t;

// Returns true when relations with |cause| can justify two otherwise
// conflicting allocation assignments sharing target-visible storage.
bool loom_low_placement_cause_can_alias(loom_low_placement_cause_t cause);

// Builds a function-local placement relation table over an acquired value
// domain and its liveness analysis.
iree_status_t loom_low_placement_analyze_region(
    loom_module_t* module, const loom_region_t* region,
    const loom_local_value_domain_t* value_domain,
    const loom_liveness_analysis_t* liveness, iree_arena_allocator_t* arena,
    loom_low_placement_table_t* out_table);

// Returns the relation range for |result_ordinal|. The ordinal must belong to
// this placement table.
loom_low_placement_relation_range_t
loom_low_placement_relation_range_for_value_ordinal(
    const loom_low_placement_table_t* table,
    loom_value_ordinal_t result_ordinal);

// Returns the relation range for |source_ordinal|. The returned range indexes
// |relation_indices_by_source_ordinal|, whose records then index |relations|.
// The ordinal must belong to this placement table.
loom_low_placement_relation_range_t
loom_low_placement_relation_range_for_source_value_ordinal(
    const loom_low_placement_table_t* table,
    loom_value_ordinal_t source_ordinal);

// Returns the local value ID for |value_ordinal|.
loom_value_id_t loom_low_placement_value_id(
    const loom_low_placement_table_t* table,
    loom_value_ordinal_t value_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_PLACEMENT_H_
