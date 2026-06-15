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
// Target packages provide generated contract fragments in rodata. A lowering
// run composes the active fragment set into a small dense root index so hot
// queries use direct dialect/op lookup without linking or scanning unrelated
// targets.

#ifndef LOOM_TARGET_CONTRACT_H_
#define LOOM_TARGET_CONTRACT_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/error/error_defs.h"
#include "loom/ir/ir.h"
#include "loom/ops/func/ops.h"
#include "loom/target/types.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_view_region_table_t loom_view_region_table_t;
typedef struct loom_matrix_fragment_layout_t loom_matrix_fragment_layout_t;

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
  // Optional rejection payload. Usually points into rodata or a scoped arena.
  const loom_target_contract_rejection_t* rejection;
} loom_target_contract_query_result_t;

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

typedef struct loom_target_contract_fragment_case_t {
  // Contract system that owns the selected row.
  loom_target_contract_system_t system;
  // Reserved byte for future row flags while keeping the case 4 bytes.
  uint8_t reserved;
  // Fragment-local system-specific row index, or LOOM_TARGET_CONTRACT_ROW_NONE.
  uint16_t row_index;
} loom_target_contract_fragment_case_t;

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
      (loom_target_contract_fragment_flags_t)1u << 0,
};

typedef struct loom_target_contract_fragment_t {
  // First dialect id covered by dialects.
  uint8_t dialect_base_id;
  // Number of dense dialect slots.
  uint8_t dialect_count;
  // Fragment behavior flags.
  loom_target_contract_fragment_flags_t flags;
  // Dense dialect slots indexed by dialect id minus dialect_base_id.
  const loom_target_contract_dialect_table_t* dialects;
  // Number of generic case rows.
  uint16_t case_count;
  // Fragment-local generic case rows referenced by dense op entries.
  const loom_target_contract_fragment_case_t* cases;
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
} loom_target_contract_index_t;

// Looks up the compact case span for an op kind in a generated fragment.
static inline loom_target_contract_op_entry_t
loom_target_contract_fragment_lookup_kind(
    const loom_target_contract_fragment_t* fragment, loom_op_kind_t op_kind) {
  const uint8_t dialect_id = loom_op_dialect_id(op_kind);
  const uint8_t op_index = loom_op_dialect_index(op_kind);
  if (dialect_id < fragment->dialect_base_id) {
    return loom_target_contract_op_entry_empty();
  }
  const uint8_t dialect_index = dialect_id - fragment->dialect_base_id;
  if (dialect_index >= fragment->dialect_count) {
    return loom_target_contract_op_entry_empty();
  }
  const loom_target_contract_dialect_table_t* dialect_table =
      &fragment->dialects[dialect_index];
  if (op_index >= dialect_table->op_count) {
    return loom_target_contract_op_entry_empty();
  }
  return dialect_table->op_entries[op_index];
}

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

// Composes |bindings| into a dense root index allocated from |arena|.
iree_status_t loom_target_contract_index_compose(
    const loom_target_contract_binding_t* bindings, uint16_t binding_count,
    loom_target_contract_index_t* out_index, iree_arena_allocator_t* arena);

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
      /*.source_rejection_bits=*/0,
      /*.source_rejection_detail=*/0,
      /*.target_rejection_bits=*/0,
      /*.missing_feature_bits=*/0,
      /*.missing_fact_bits=*/0,
      /*.rejection=*/NULL,
  };
}

typedef struct loom_target_contract_query_environment_t {
  // Source module being queried.
  const loom_module_t* module;
  // Source function containing the queried op.
  loom_func_like_t function;
  // Target bundle selected for this query.
  const loom_target_bundle_t* bundle;
  // Target-owned payload associated with |bundle|, or NULL.
  const void* target_data;
  // Module-local target record symbol selected for this query.
  loom_symbol_ref_t target_ref;
  // Low descriptor set selected for this query.
  const loom_low_descriptor_set_t* descriptor_set;
  // Source value facts visible to the query.
  const loom_value_fact_table_t* fact_table;
  // Optional function-local view-region analysis visible to the query.
  const loom_view_region_table_t* view_regions;
  // Scoped arena available for rare query-side auxiliary records.
  iree_arena_allocator_t* arena;
} loom_target_contract_query_environment_t;

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
