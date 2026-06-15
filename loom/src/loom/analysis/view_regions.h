// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Symbolic view regions over buffer roots.
//
// This analysis summarizes typed view SSA values as half-open byte regions:
//
//   [begin_byte_offset, end_byte_offset)
//
// The summary is rooted by storage identity and keeps symbolic expressions for
// begin, byte length, and end. It consumes the existing value fact table for
// root/base/footprint/alignment facts and uses the generic symbolic expression
// substrate for exact affine byte relationships. Memory access flags are
// derived from actual op operand descriptors, not authored on the views.

#ifndef LOOM_ANALYSIS_VIEW_REGIONS_H_
#define LOOM_ANALYSIS_VIEW_REGIONS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbolic_expr.h"
#include "loom/ir/ir.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t loom_view_region_id_t;
#define LOOM_VIEW_REGION_ID_INVALID ((loom_view_region_id_t)UINT32_MAX)

// Access bits derived from memory-operand descriptors.
enum loom_view_access_flag_bits_e {
  // The view is read by at least one memory operation.
  LOOM_VIEW_ACCESS_READ = 1u << 0,

  // The view is written by at least one memory operation.
  LOOM_VIEW_ACCESS_WRITE = 1u << 1,
};
typedef uint32_t loom_view_access_flags_t;

// Precision bits describing which parts of a region are known symbolically.
enum loom_view_region_precision_flag_bits_e {
  // The storage root identity is known.
  LOOM_VIEW_REGION_PRECISION_ROOT = 1u << 0,

  // The begin byte expression is exact linear symbolic form.
  LOOM_VIEW_REGION_PRECISION_BEGIN = 1u << 1,

  // The byte length expression is exact linear symbolic form.
  LOOM_VIEW_REGION_PRECISION_LENGTH = 1u << 2,

  // The end byte expression is exact linear symbolic form.
  LOOM_VIEW_REGION_PRECISION_END = 1u << 3,
};
typedef uint32_t loom_view_region_precision_flags_t;

// A typed projection over a buffer-like root.
typedef struct loom_view_region_t {
  // Region ID assigned by the owning table.
  loom_view_region_id_t region_id;

  // SSA value whose type is a view.
  loom_value_id_t view_value_id;

  // SSA value representing the storage root identity.
  loom_value_id_t root_value_id;

  // Symbolic byte offset of the view base relative to root_value_id.
  loom_symbolic_expr_t begin_byte_offset;

  // Symbolic byte length of the conservative footprint envelope.
  loom_symbolic_expr_t byte_length;

  // Symbolic byte offset one past the conservative footprint envelope.
  loom_symbolic_expr_t end_byte_offset;

  // Minimum provable alignment of begin_byte_offset relative to root_value_id.
  uint64_t minimum_alignment;

  // Minimum provable byte alignment of the root storage base.
  uint64_t root_minimum_alignment;

  // Static addressed element byte count, or -1 for sub-byte/unknown elements.
  int64_t static_element_byte_count;

  // Target-independent memory space for the underlying storage root.
  loom_value_fact_memory_space_t memory_space;

  // Derived memory access bits for this particular SSA view value.
  loom_view_access_flags_t access_flags;

  // Bitfield of loom_view_region_precision_flag_bits_e.
  loom_view_region_precision_flags_t precision_flags;
} loom_view_region_t;

// Dense analysis table for one function-local value domain.
typedef struct loom_view_region_table_t {
  // Module whose SSA values are summarized.
  const loom_module_t* module;

  // Current fact table consumed by expression and view-reference queries.
  loom_value_fact_table_t* fact_table;

  // Fact context wrapper for APIs that query fact extensions.
  loom_fact_context_t fact_context;

  // Arena used for all table, region, and symbolic expression storage.
  iree_arena_allocator_t* arena;

  // Symbolic expression context shared by all region construction.
  loom_symbolic_expr_context_t expression_context;

  // Borrowed active local value domain used for value ID to ordinal mapping.
  const loom_local_value_domain_t* value_domain;

  // Dense map from local value ordinal to region ID, or INVALID.
  loom_view_region_id_t* region_ids_by_value_ordinal;

  // Per-local-value construction state for recursion guards.
  uint8_t* states_by_value_ordinal;

  // Compact region storage indexed by region ID.
  loom_view_region_t* regions;

  // Number of valid entries in regions.
  iree_host_size_t region_count;

  // Allocated entry count for regions.
  iree_host_size_t region_capacity;
} loom_view_region_table_t;

// Initializes a view-region table from a caller-owned active local value
// domain. The table does not compute facts; callers pass the current fact table
// they want the analysis to consume. The value domain must remain active until
// the table is dead.
iree_status_t loom_view_region_table_initialize(
    loom_value_fact_table_t* fact_table,
    const loom_local_value_domain_t* value_domain,
    iree_arena_allocator_t* arena, loom_view_region_table_t* out_table);

// Ensures a region summary exists for |value_id| when it has a view type.
// Non-view values return NULL without error.
iree_status_t loom_view_region_table_get(loom_view_region_table_t* table,
                                         loom_value_id_t value_id,
                                         const loom_view_region_t** out_region);

// Walks the table's local value domain region, constructs summaries for view
// values, and derives per-view access flags from memory-operand descriptors.
iree_status_t loom_view_region_table_analyze(loom_view_region_table_t* table);

// Returns aggregate access flags for all summarized regions with
// |root_value_id|.
loom_view_access_flags_t loom_view_region_table_root_access_flags(
    const loom_view_region_table_t* table, loom_value_id_t root_value_id);

// Attempts to prove that two same-root view regions cannot overlap. Different
// root SSA values are conservative unknown until a separate root-disjoint fact
// system exists.
iree_status_t loom_view_regions_prove_no_overlap(
    loom_view_region_table_t* table, const loom_view_region_t* left_region,
    const loom_view_region_t* right_region, bool* out_no_overlap);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_VIEW_REGIONS_H_
