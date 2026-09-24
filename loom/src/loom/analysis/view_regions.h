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
typedef uint8_t loom_view_access_flags_t;

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

  // SSA view value that materializes the root-relative base resource.
  loom_value_id_t base_view_value_id;

  // SSA value representing the storage root identity.
  loom_value_id_t root_value_id;

  // Comparable alias scope for disjointness proofs, or NONE when distinct
  // root identities are only addressing provenance.
  loom_value_fact_alias_scope_id_t alias_scope_id;

  // Known nullability of the underlying storage root.
  loom_value_fact_reference_nullability_t nullability;

  // Unsigned source address-carrier width guaranteed for valid active accesses
  // through this view, or zero when no source carrier contract is known.
  uint8_t address_bitwidth;

  // Intrinsic storage origin retained from the source reference.
  loom_value_fact_reference_origin_t origin;

  // Symbolic byte offset of the view base relative to root_value_id.
  loom_symbolic_expr_t begin_byte_offset;

  // Symbolic byte offset of base_view_value_id relative to root_value_id.
  loom_symbolic_expr_t base_begin_byte_offset;

  // Symbolic byte offset from base_view_value_id to view_value_id.
  loom_symbolic_expr_t projection_byte_offset;

  // Existing SSA value that materializes begin_byte_offset, or invalid when
  // the begin offset only exists as a symbolic expression.
  loom_value_id_t begin_value_id;

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

static_assert(sizeof(loom_view_region_t) == 440,
              "view regions must remain 440 bytes");

// Dense analysis table for one function-local value domain.
typedef struct loom_view_region_table_t {
  // Borrowed symbolic expression context shared by all region construction.
  loom_symbolic_expr_context_t* expression_context;

  // Borrowed active local value domain extended for discovered values.
  loom_local_value_domain_t* value_domain;

  // Dense map from local value ordinal to region ID, or INVALID.
  loom_view_region_id_t* region_ids_by_value_ordinal;

  // Per-local-value construction state for recursion guards.
  uint8_t* states_by_value_ordinal;

  // Aggregate root accesses and value invariance, indexed by local ordinal.
  // Populated by analyze, including raw buffers and nested execution scopes.
  uint8_t* storage_flags_by_value_ordinal;

  // Memory spaces that may change through acquisition, unknown effects, or
  // writes without comparable storage identities. Bit i names memory space i.
  uint32_t interference_memory_spaces;

  // Memory spaces written through scoped roots that can vary across executions.
  // Such writes may interfere with other varying roots despite local noalias.
  uint32_t varying_root_write_memory_spaces;

  // Compact region storage indexed by region ID.
  loom_view_region_t* regions;

  // Number of valid entries in regions.
  iree_host_size_t region_count;

  // Allocated entry count for regions.
  iree_host_size_t region_capacity;
} loom_view_region_table_t;

// Initializes a view-region table from a caller-owned active local value
// domain and its matching symbolic expression context. The value domain and
// expression context must remain active until the table is dead. Captured
// reference backing roots are registered before allocating ordinal tables.
iree_status_t loom_view_region_table_initialize(
    loom_local_value_domain_t* value_domain,
    loom_symbolic_expr_context_t* expression_context,
    loom_view_region_table_t* out_table);

// Ensures a region summary exists for |value_id| when it has a view type.
// Non-view values return NULL without error.
iree_status_t loom_view_region_table_get(loom_view_region_table_t* table,
                                         loom_value_id_t value_id,
                                         const loom_view_region_t** out_region);

// Looks up an already constructed region summary without allocating or
// recursively constructing missing producers. Returns false when |value_id| is
// outside the analyzed local domain or when the summary is not ready.
bool loom_view_region_table_try_lookup(const loom_view_region_table_t* table,
                                       loom_value_id_t value_id,
                                       const loom_view_region_t** out_region);

// Derives the exact one-element byte region addressed by |static_indices| and
// |dynamic_indices| within |view_value_id|. The returned region preserves the
// storage root, alias scope, nullability, and memory space of the source view.
// Unsupported element widths or malformed index lists return OK with
// |out_derived| false. Nonlinear but well-formed addresses remain derived with
// conservative symbolic begin/end expressions.
iree_status_t loom_view_region_table_derive_element_region(
    loom_view_region_table_t* table, loom_value_id_t view_value_id,
    loom_attribute_t static_indices, loom_value_slice_t dynamic_indices,
    loom_view_region_t* out_region, bool* out_derived);

// Walks the table's local value domain region, constructs summaries for view
// values, and derives per-view/root accesses and memory-space interference.
// Retains cross-execution value invariance when scoped writes require it.
// The result covers nested control flow and is invalidated by IR mutation.
iree_status_t loom_view_region_table_analyze(loom_view_region_table_t* table);

// Returns analyzed aggregate access flags through |root_value_id| in constant
// time, or zero when the root is outside the table's local value domain.
loom_view_access_flags_t loom_view_region_table_root_access_flags(
    const loom_view_region_table_t* table, loom_value_id_t root_value_id);

// Proves that an accessed storage root remains unchanged throughout the
// analyzed domain. Local read-only access alone is insufficient: acquisition
// may import another participant's writes, and unknown effects or incomparable
// written roots may alias it. Constant storage is immutable by contract.
// Scoped noalias separates repeated accesses only when at least one root keeps
// the same storage across executions. The analysis retains this correspondence
// alongside the aggregate accesses; the query uses constant-time indexed facts.
// |alias_scope_id| and |memory_space| are retained facts for the queried
// access. Requires analyze to have completed; this query performs no IR
// traversal.
bool loom_view_region_table_root_is_stable(
    const loom_view_region_table_t* table, loom_value_id_t root_value_id,
    loom_value_fact_alias_scope_id_t alias_scope_id,
    loom_value_fact_memory_space_t memory_space);

// Returns true when two concrete memory spaces cannot name the same storage.
// Unknown and generic spaces remain conservative, as do distinct global-like
// spaces whose target representations may overlap.
bool loom_view_memory_spaces_are_disjoint(loom_value_fact_memory_space_t left,
                                          loom_value_fact_memory_space_t right);

// Attempts to prove that two view regions cannot overlap. Same-root regions
// use symbolic byte intervals. Distinct roots are disjoint when their concrete
// memory spaces cannot alias or both carry comparable and different alias
// scopes. These identities and symbolic terms describe one execution. A caller
// comparing different executions establishes root/term correspondence before
// applying this proof; equal SSA IDs alone do not establish that
// correspondence.
iree_status_t loom_view_regions_prove_no_overlap(
    loom_view_region_table_t* table, const loom_view_region_t* left_region,
    const loom_view_region_t* right_region, bool* out_no_overlap);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_VIEW_REGIONS_H_
