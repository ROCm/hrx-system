// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target contract query ABI.
//
// Contract queries are the read-only selection layer shared by target
// legalization, target-low legality, and source-to-low emission. They answer
// whether a source op is already in a form accepted by the selected target
// bundle and descriptor set. Unsupported program forms are reported as compact
// query results; non-OK status is reserved for infrastructure failures.
//
// Target packages provide generated contract fragments and dense indices in
// rodata. Queries use direct dialect/op lookup; index construction and fragment
// composition happen at build time, independently for each target policy.

#ifndef LOOM_TARGET_CONTRACT_H_
#define LOOM_TARGET_CONTRACT_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/error/error_defs.h"
#include "loom/ir/ir.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/func/ops.h"
#include "loom/target/facts.h"
#include "loom/target/types.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_view_region_table_t loom_view_region_table_t;
typedef struct loom_matrix_fragment_layout_t loom_matrix_fragment_layout_t;
typedef struct loom_native_contraction_facts_t loom_native_contraction_facts_t;
typedef struct loom_local_value_domain_t loom_local_value_domain_t;

// Returns true when |type| has a complete source-vector carrier mapping under
// the selected target policy.
typedef bool (*loom_target_source_vector_carrier_supported_fn_t)(
    void* user_data, const loom_module_t* module, loom_type_t type);

typedef struct loom_target_source_vector_carrier_supported_callback_t {
  // Optional target query for complete source-vector carrier admission.
  loom_target_source_vector_carrier_supported_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_target_source_vector_carrier_supported_callback_t;

// Maximum number of logical lane candidates in a packet policy. Shared
// component planning represents the candidate intersection as one bit per
// policy entry.
#define LOOM_TARGET_VECTOR_PACKET_LANE_COUNT_LIMIT 64u

// Target vector packet candidates consumed by shared legalization.
//
// The target contributes representation widths only. Shared planners query
// projected operations through the target contract before selecting a packet
// width. A handled answer is authoritative; targets whose custom Low planner
// leaves structural memory candidates unhandled retain their declared packet
// decomposition. Membership here is not itself a final legality claim.
typedef struct loom_target_vector_packet_lane_limit_t {
  // Scalar element type whose structural carrier has a logical lane ceiling.
  loom_scalar_type_t element_type;
  // Maximum logical lanes carried by one structural packet.
  uint16_t maximum_lane_count;
} loom_target_vector_packet_lane_limit_t;

typedef struct loom_target_vector_packet_policy_t {
  // Native packet widths in bits used by structural memory and carrier
  // legalization. Widths are byte-aligned powers of two.
  const uint16_t* native_bit_counts;
  // Logical lane counts worth evaluating for decomposable components.
  const uint16_t* native_lane_counts;
  // Physical target width of index lanes. Zero uses the abstract source width.
  uint8_t index_bit_count;
  // Physical target width of offset lanes. Zero uses the abstract source width.
  uint8_t offset_bit_count;
  // Sparse per-element lane ceilings for structural packet widths.
  const loom_target_vector_packet_lane_limit_t* structural_lane_limits;
  // Largest payload in bits that remains owned by ordinary structural
  // lowering instead of packet legalization.
  uint16_t maximum_unpacketized_bit_count;
  // Number of entries in |native_bit_counts|.
  uint8_t native_bit_count_count;
  // Number of entries in |native_lane_counts|, at most
  // LOOM_TARGET_VECTOR_PACKET_LANE_COUNT_LIMIT.
  uint8_t native_lane_count_count;
  // Number of entries in |structural_lane_limits|.
  uint8_t structural_lane_limit_count;
} loom_target_vector_packet_policy_t;

// Scoped vector lane-count projection for one contract query. An all-zero
// value reads authored types directly. A populated projection presents static
// vectors with |source_lane_count| total lanes as rank-one vectors with
// |projected_lane_count| lanes without mutating source IR. Source-memory
// matching sees the same projected lane count while retaining every authored
// address, layout, alignment, and execution fact.
typedef struct loom_target_contract_vector_lane_projection_t {
  // Authored static vector lane count selected for projection.
  uint32_t source_lane_count;
  // Rank-one vector lane count presented to the contract query.
  uint32_t projected_lane_count;
} loom_target_contract_vector_lane_projection_t;

// Returns the vector lane count visible to a projected contract query. Facts
// for unrelated lane counts retain their authored value.
static inline uint32_t loom_target_contract_query_vector_lane_count(
    loom_target_contract_vector_lane_projection_t projection,
    uint32_t authored_lane_count) {
  if (projection.source_lane_count != 0 &&
      authored_lane_count == projection.source_lane_count) {
    return projection.projected_lane_count;
  }
  return authored_lane_count;
}

// Returns the projected type for |value_id|. Callers first test for an empty
// projection so ordinary authored queries remain on the direct type-table
// path and only projected queries perform shape projection.
loom_type_t loom_target_contract_query_projected_value_type(
    loom_target_contract_vector_lane_projection_t projection,
    const loom_module_t* module, loom_value_id_t value_id);

// Returns the scoped query type for |value_id|.
static inline loom_type_t loom_target_contract_query_value_type(
    loom_target_contract_vector_lane_projection_t projection,
    const loom_module_t* module, loom_value_id_t value_id) {
  if (projection.source_lane_count == 0) {
    return loom_module_value_type(module, value_id);
  }
  return loom_target_contract_query_projected_value_type(projection, module,
                                                         value_id);
}

typedef enum loom_target_contract_query_outcome_e {
  // No linked contract fragment or provider has an opinion about the op.
  LOOM_TARGET_CONTRACT_QUERY_UNHANDLED = 0,
  // The op is already legal for the selected target contract.
  LOOM_TARGET_CONTRACT_QUERY_LEGAL = 1,
  // The op family is recognized but unsupported by the selected target
  // contract.
  LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED = 2,
  // The op violates the source contract required before target selection.
  LOOM_TARGET_CONTRACT_QUERY_INVALID_IR = 3,
} loom_target_contract_query_outcome_t;

typedef struct loom_target_contract_rejection_t {
  // Stable structured diagnostic identity.
  loom_error_ref_t error_ref;
  // Materialized diagnostic parameters.
  const loom_diagnostic_param_t* params;
  // Number of materialized diagnostic parameters.
  iree_host_size_t param_count;
} loom_target_contract_rejection_t;

enum loom_target_contract_descriptor_matrix_transform_flag_bits_e {
  // The selected descriptor computes the transposed matrix product by
  // exchanging source LHS/RHS operands and transposing M/N request facts.
  LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_TRANSFORM_TRANSPOSE_MN = 1u << 0,
};
typedef uint32_t loom_target_contract_descriptor_matrix_transform_flags_t;

typedef struct loom_target_contract_query_result_t {
  // Query outcome.
  loom_target_contract_query_outcome_t outcome;
  // Active binding ordinal selected or rejected by the query.
  uint16_t binding_index;
  // Composed case ordinal selected or rejected by the query.
  uint16_t case_index;
  // Policy rule-set ordinal selected or rejected by the query.
  uint16_t rule_set_index;
  // Rule row ordinal selected or rejected by the query.
  uint16_t rule_index;
  // Diagnostic row ordinal retained by the rejected rule.
  uint16_t diagnostic_index;
  // Number of guards matched before the selected rejected guard.
  uint16_t matched_guard_count;
  // Low descriptor row selected by the accepted rule, or NULL.
  const loom_low_descriptor_t* selected_descriptor;
  // Matrix-fragment lane/register layout selected with the source contract, or
  // NULL when the queried op is not a matrix-fragment contract.
  const loom_matrix_fragment_layout_t* selected_matrix_fragment_layout;
  // Compact native contraction placement selected with the source contract,
  // or NULL when the selected target primitive has no modeled native layout.
  const loom_native_contraction_facts_t* selected_native_contraction_facts;
  // Compact target-independent rejection flags.
  uint32_t source_rejection_bits;
  // Optional target-independent rejection detail enum.
  uint32_t source_rejection_detail;
  // Compact target-owned rejection flags.
  uint32_t target_rejection_bits;
  // Target feature bits missing from the selected bundle.
  uint32_t missing_feature_bits;
  // Value fact categories missing for the selected rule.
  uint32_t missing_fact_bits;
  // Exact source-to-native transform selected with a descriptor-matrix row.
  loom_target_contract_descriptor_matrix_transform_flags_t
      selected_descriptor_matrix_transform_flags;
  // Optional rejection payload. Usually points into rodata or a scoped arena.
  const loom_target_contract_rejection_t* rejection;
} loom_target_contract_query_result_t;

typedef iree_status_t (*loom_target_contract_query_get_or_allocate_state_fn_t)(
    void* user_data, const void* key, iree_host_size_t data_length,
    void** out_data);

typedef struct loom_target_contract_query_state_allocator_t {
  // Callback invoked to find or allocate target-owned query state.
  loom_target_contract_query_get_or_allocate_state_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_target_contract_query_state_allocator_t;

typedef uint8_t loom_target_contract_system_t;

enum loom_target_contract_system_e {
  // No contract-system row is attached.
  LOOM_TARGET_CONTRACT_SYSTEM_NONE = 0,
  // Descriptor-rule row selected from a generated descriptor-rule pool.
  LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE = 1,
  // Value-alias row with no emitted low descriptor.
  LOOM_TARGET_CONTRACT_SYSTEM_VALUE_ALIAS = 2,
  // Value-elide lower-rule row with no emitted low descriptor.
  LOOM_TARGET_CONTRACT_SYSTEM_VALUE_ELIDE = 3,
  // Source-memory row selected from a generated source-memory pool.
  LOOM_TARGET_CONTRACT_SYSTEM_SOURCE_MEMORY = 4,
  // Environment row selected from a generated environment pool.
  LOOM_TARGET_CONTRACT_SYSTEM_ENVIRONMENT = 5,
  // Descriptor-matrix row selected from a generated matrix-contract pool.
  LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_MATRIX = 6,
  // Guard-only lower-rule row for a bounded non-descriptor recipe.
  LOOM_TARGET_CONTRACT_SYSTEM_RECIPE_RULE = 7,
  // Explicit target rejection for an operation family with no matching native
  // case. Reference legalization may rewrite the operation before lowering.
  LOOM_TARGET_CONTRACT_SYSTEM_UNSUPPORTED = 8,
};

#define LOOM_TARGET_CONTRACT_ROW_NONE ((uint16_t)UINT16_MAX)

typedef struct loom_target_contract_op_entry_t {
  // First case row for the dialect-local op index.
  uint16_t case_start;
  // Number of case rows available for the dialect-local op index.
  uint16_t case_count;
} loom_target_contract_op_entry_t;

// Returns an empty target contract op entry.
static inline loom_target_contract_op_entry_t
loom_target_contract_op_entry_empty(void) {
  return (loom_target_contract_op_entry_t){
      /*.case_start=*/LOOM_TARGET_CONTRACT_ROW_NONE,
      /*.case_count=*/0,
  };
}

// Returns true when |entry| has no case rows.
static inline bool loom_target_contract_op_entry_is_empty(
    loom_target_contract_op_entry_t entry) {
  return entry.case_count == 0;
}

typedef struct loom_target_contract_dialect_table_t {
  // Number of dialect-local op entries.
  uint16_t op_count;
  // Dense op entries indexed by loom_op_dialect_index.
  const loom_target_contract_op_entry_t* op_entries;
} loom_target_contract_dialect_table_t;

typedef struct loom_target_contract_case_t {
  // Contract system that owns the selected row.
  loom_target_contract_system_t system;
  // Active binding ordinal that owns the selected row.
  uint8_t binding_index;
  // System-specific row index, or LOOM_TARGET_CONTRACT_ROW_NONE.
  uint16_t row_index;
} loom_target_contract_case_t;

typedef struct loom_target_contract_descriptor_rule_t {
  // Descriptor-rule row in the target-owned rule interpreter table.
  uint16_t rule_index;
} loom_target_contract_descriptor_rule_t;

typedef uint8_t loom_target_contract_descriptor_matrix_source_t;

enum loom_target_contract_descriptor_matrix_source_e {
  // No descriptor-matrix source adapter is selected.
  LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_NONE = 0,
  // Source vector.mma op adapted through the generic matrix contract request.
  LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_SOURCE_VECTOR_MMA = 1,
};

typedef struct loom_target_contract_descriptor_matrix_rule_t {
  // Shared source adapter used to build a generic matrix contract request.
  loom_target_contract_descriptor_matrix_source_t source;
  // Reserved byte for future per-row behavior flags.
  uint8_t reserved;
  // Reserved halfword for future row-local descriptor ranges.
  uint16_t reserved0;
} loom_target_contract_descriptor_matrix_rule_t;

typedef uint8_t loom_target_contract_fragment_flags_t;

enum loom_target_contract_fragment_flag_bits_e {
  // Fragment cases participate in read-only target contract queries.
  LOOM_TARGET_CONTRACT_FRAGMENT_FLAG_TARGET_QUERY =
      (loom_target_contract_fragment_flags_t)(1u << 0),
};

typedef struct loom_target_contract_fragment_t {
  // Fragment behavior flags.
  loom_target_contract_fragment_flags_t flags;
  // Number of descriptor-rule rows.
  uint16_t descriptor_rule_count;
  // Descriptor-rule row pool.
  const loom_target_contract_descriptor_rule_t* descriptor_rules;
  // Number of descriptor-matrix rows.
  uint16_t descriptor_matrix_count;
  // Descriptor-matrix row pool.
  const loom_target_contract_descriptor_matrix_rule_t* descriptor_matrices;
} loom_target_contract_fragment_t;

// Returns true when |fragment| participates in read-only target contract
// queries.
static inline bool loom_target_contract_fragment_queries_target(
    const loom_target_contract_fragment_t* fragment) {
  return (fragment->flags & LOOM_TARGET_CONTRACT_FRAGMENT_FLAG_TARGET_QUERY) !=
         0;
}

typedef struct loom_target_contract_binding_t {
  // Generated fragment rodata linked into the active target package.
  const loom_target_contract_fragment_t* fragment;
  // Policy rule-set ordinal corresponding to |fragment|'s descriptor-rule rows.
  uint16_t rule_set_index;
} loom_target_contract_binding_t;

typedef struct loom_target_contract_index_t {
  // First dialect id covered by dialects.
  uint8_t dialect_base_id;
  // Number of dense dialect slots.
  uint8_t dialect_count;
  // Dense dialect slots indexed by dialect id minus dialect_base_id.
  const loom_target_contract_dialect_table_t* dialects;
  // Number of composed case rows.
  uint16_t case_count;
  // Composed case rows referenced by dense op entries.
  const loom_target_contract_case_t* cases;
  // Number of active fragment bindings.
  uint8_t binding_count;
  // Active fragment bindings referenced by composed case rows.
  const loom_target_contract_binding_t* bindings;
  // Relocation-free candidate-selection rows, buckets, and ordinal sequences.
  const uint32_t* selection_data;
} loom_target_contract_index_t;

// Looks up the compact case span for an op kind in a composed index.
static inline loom_target_contract_op_entry_t
loom_target_contract_index_lookup_kind(
    const loom_target_contract_index_t* index, loom_op_kind_t op_kind) {
  const uint8_t dialect_id = loom_op_dialect_id(op_kind);
  const uint8_t op_index = loom_op_dialect_index(op_kind);
  if (dialect_id < index->dialect_base_id) {
    return loom_target_contract_op_entry_empty();
  }
  const uint8_t dialect_index = dialect_id - index->dialect_base_id;
  if (dialect_index >= index->dialect_count) {
    return loom_target_contract_op_entry_empty();
  }
  const loom_target_contract_dialect_table_t* dialect_table =
      &index->dialects[dialect_index];
  if (op_index >= dialect_table->op_count) {
    return loom_target_contract_op_entry_empty();
  }
  return dialect_table->op_entries[op_index];
}

// Returns an empty target contract query result.
static inline loom_target_contract_query_result_t
loom_target_contract_query_result_empty(void) {
  return (loom_target_contract_query_result_t){
      /*.outcome=*/LOOM_TARGET_CONTRACT_QUERY_UNHANDLED,
      /*.binding_index=*/UINT16_MAX,
      /*.case_index=*/UINT16_MAX,
      /*.rule_set_index=*/UINT16_MAX,
      /*.rule_index=*/UINT16_MAX,
      /*.diagnostic_index=*/UINT16_MAX,
      /*.matched_guard_count=*/0,
      /*.selected_descriptor=*/NULL,
      /*.selected_matrix_fragment_layout=*/NULL,
      /*.selected_native_contraction_facts=*/NULL,
      /*.source_rejection_bits=*/0,
      /*.source_rejection_detail=*/0,
      /*.target_rejection_bits=*/0,
      /*.missing_feature_bits=*/0,
      /*.missing_fact_bits=*/0,
      /*.selected_descriptor_matrix_transform_flags=*/0,
      /*.rejection=*/NULL,
  };
}

typedef struct loom_target_contract_query_environment_t {
  // Source module being queried.
  const loom_module_t* module;
  // Source function containing the queried op.
  loom_func_like_t function;
  // Borrowed immutable target facts selected for this query.
  const loom_target_facts_t* target_facts;
  // Low descriptor set selected for this query.
  const loom_low_descriptor_set_t* descriptor_set;
  // Source value facts visible to the query.
  const loom_value_fact_table_t* fact_table;
  // Shared reference/access scope valid for this immutable source snapshot.
  struct loom_storage_access_scope_t* storage_access;
  // Optional scoped vector lane projection used for candidate planning.
  loom_target_contract_vector_lane_projection_t vector_lane_projection;
  // Optional active value domain extended by ordinal-keyed query analyses.
  loom_local_value_domain_t* value_domain;
  // Optional function-local view-region analysis visible to the query.
  const loom_view_region_table_t* view_regions;
  // Scoped arena available for rare query-side auxiliary records.
  iree_arena_allocator_t* arena;
  // Optional scoped storage allocator for target-owned query analyses.
  loom_target_contract_query_state_allocator_t target_state_allocator;
} loom_target_contract_query_environment_t;

// Returns the common target bundle selected for |environment|.
static inline const loom_target_bundle_t*
loom_target_contract_query_environment_bundle(
    const loom_target_contract_query_environment_t* environment) {
  return loom_target_facts_bundle(environment->target_facts);
}

// Returns scoped target-owned query state for |key|, or NULL when the query
// environment has no state allocator.
iree_status_t loom_target_contract_query_get_or_allocate_target_state(
    const loom_target_contract_query_environment_t* environment,
    const void* key, iree_host_size_t data_length, void** out_data);

typedef iree_status_t (*loom_target_contract_query_op_fn_t)(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* op, loom_target_contract_query_result_t* out_result);

typedef struct loom_target_contract_query_callback_t {
  // Callback invoked to query one source op.
  loom_target_contract_query_op_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_target_contract_query_callback_t;

// Returns an empty target contract query callback.
static inline loom_target_contract_query_callback_t
loom_target_contract_query_callback_empty(void) {
  return (loom_target_contract_query_callback_t){0};
}

// Returns true when |callback| has no query function.
static inline bool loom_target_contract_query_callback_is_empty(
    loom_target_contract_query_callback_t callback) {
  return callback.fn == NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_CONTRACT_H_
