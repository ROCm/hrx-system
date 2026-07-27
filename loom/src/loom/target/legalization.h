// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target legalization registry and callback ABI.
//
// Target legalization is the mutable sibling of target contract queries. The
// query layer answers whether an op is already accepted by the selected target
// contract; legalizers rewrite unsupported source forms into lower or more
// target-shaped IR. Target packages contribute callbacks through provider
// tables, while generic reference legalizers can be composed after the
// target-owned providers to offer a scalar path for ops the target does not
// retain for native lowering.

#ifndef LOOM_TARGET_LEGALIZATION_H_
#define LOOM_TARGET_LEGALIZATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/pass/types.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/contract.h"
#include "loom/target/types.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_view_region_table_t loom_view_region_table_t;
typedef struct loom_target_legalizer_entry_t loom_target_legalizer_entry_t;
typedef struct loom_target_legalizer_provider_t
    loom_target_legalizer_provider_t;

typedef enum loom_target_legalization_mode_e {
  // Opportunistic legalization. Unsupported ops can be left for a later run.
  LOOM_TARGET_LEGALIZATION_MODE_EAGER = 0,
  // Final legalization. Remaining unsupported ops must be diagnosed by the
  // caller after the rewrite fixed point.
  LOOM_TARGET_LEGALIZATION_MODE_FINAL = 1,
} loom_target_legalization_mode_t;

typedef enum loom_target_legalization_policy_e {
  // Prefer target-native rewrites and contracts, then use reference fallbacks.
  LOOM_TARGET_LEGALIZATION_POLICY_PREFER_NATIVE = 0,
  // Skip target-native rewrites and prefer semantic reference rewrites.
  LOOM_TARGET_LEGALIZATION_POLICY_REFERENCE_ONLY = 1,
  // Require target-native contracts or rewrites; reject reference fallbacks.
  LOOM_TARGET_LEGALIZATION_POLICY_REQUIRE_NATIVE = 2,
} loom_target_legalization_policy_t;

typedef enum loom_target_legalizer_action_e {
  // The legalizer has no opinion about this op under the current target.
  LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT = 0,
  // The legalizer rewrote the op or erased it through the active rewriter.
  LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN = 1,
  // The legalizer recognizes the op but leaves it for a later legalization,
  // target lowering, or target verification phase.
  LOOM_TARGET_LEGALIZER_ACTION_DEFER = 2,
  // The legalizer found invalid source IR and expects the final verifier to
  // emit the precise diagnostic.
  LOOM_TARGET_LEGALIZER_ACTION_REJECT_INVALID_IR = 3,
  // The legalizer recognizes the final op shape but has no lowering for the
  // selected target contract.
  LOOM_TARGET_LEGALIZER_ACTION_REJECT_UNSUPPORTED_FINAL = 4,
} loom_target_legalizer_action_t;

typedef enum loom_target_legalizer_strategy_e {
  // The provider did not classify its rewrite strategy.
  LOOM_TARGET_LEGALIZER_STRATEGY_UNKNOWN = 0,
  // Target-specific rewrite intended to reach a native target contract.
  LOOM_TARGET_LEGALIZER_STRATEGY_TARGET = 1,
  // Target-independent reference rewrite used as a portable fallback.
  LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE = 2,
} loom_target_legalizer_strategy_t;

enum loom_target_legalizer_entry_flag_bits_e {
  // Calls this legalizer even when the target contract already accepts the op.
  LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL = 1u << 0,
};
typedef uint32_t loom_target_legalizer_entry_flags_t;

typedef struct loom_target_legalizer_result_t {
  // Outcome selected by the legalizer.
  loom_target_legalizer_action_t action;
  // Target-independent rejection flags added by the legalizer.
  uint32_t source_rejection_bits;
  // Optional target-independent rejection detail enum added by the legalizer.
  uint32_t source_rejection_detail;
  // Target-owned rejection flags added by the legalizer.
  uint32_t target_rejection_bits;
  // Target feature bits required by the legalizer but absent from the bundle.
  uint32_t missing_feature_bits;
  // Value fact categories required by the legalizer but absent from the IR.
  uint32_t missing_fact_bits;
} loom_target_legalizer_result_t;

typedef struct loom_target_legalization_context_t {
  // Pass invocation owning diagnostics, statistics, and scoped arenas.
  loom_pass_t* pass;
  // Source module being legalized.
  loom_module_t* module;
  // Source function containing the current rewrite region.
  loom_func_like_t function;
  // Target bundle selected for this function.
  const loom_target_bundle_t* bundle;
  // Target-owned payload associated with |bundle|, or NULL.
  const void* target_data;
  // Module-local target record symbol selected for this function.
  loom_symbol_ref_t target_ref;
  // Low descriptor set selected by the target bundle.
  const loom_low_descriptor_set_t* descriptor_set;
  // Source value facts visible to legalizers.
  const loom_value_fact_table_t* fact_table;
  // Optional view-region analysis visible to legalizers.
  const loom_view_region_table_t* view_regions;
  // Active rewriter for mutating the function body.
  loom_rewriter_t* rewriter;
  // Current legalization phase.
  loom_target_legalization_mode_t mode;
  // Strategy policy controlling native and reference legalizer participation.
  loom_target_legalization_policy_t policy;
  // Scoped arena available for rare legalization-side records.
  iree_arena_allocator_t* arena;
  // Target contract query result for the op currently being passed to a
  // legalizer, or NULL outside a legalizer callback.
  const loom_target_contract_query_result_t* contract_query_result;
  // Read-only target contract query for already-legal checks.
  loom_target_contract_query_callback_t contract_query;
} loom_target_legalization_context_t;

typedef iree_status_t (*loom_target_legalizer_fn_t)(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

struct loom_target_legalizer_entry_t {
  // Entry-specific behavior flags.
  loom_target_legalizer_entry_flags_t flags;
  // Op kind this legalizer can rewrite.
  loom_op_kind_t root_kind;
  // Stable provider name attached while composing the dense registry.
  iree_string_view_t provider_name;
  // Rewrite strategy attached while composing the dense registry.
  loom_target_legalizer_strategy_t provider_strategy;
  // Rewriter callback for root_kind.
  loom_target_legalizer_fn_t legalize;
  // Provider-owned payload consumed by legalize.
  const void* user_data;
};

struct loom_target_legalizer_provider_t {
  // Stable provider name for cold reports and diagnostics.
  iree_string_view_t name;
  // Strategy shared by legalizer rows from this provider.
  loom_target_legalizer_strategy_t strategy;
  // Legalizer rows contributed by this provider.
  const loom_target_legalizer_entry_t* entries;
  // Number of rows in entries.
  uint16_t entry_count;
};

typedef struct loom_target_legalizer_provider_list_t {
  // Number of provider pointers in values.
  iree_host_size_t count;
  // Borrowed provider pointer table.
  const loom_target_legalizer_provider_t* const* values;
} loom_target_legalizer_provider_list_t;

// Creates a borrowed target legalizer provider list.
static inline loom_target_legalizer_provider_list_t
loom_target_legalizer_provider_list_make(
    const loom_target_legalizer_provider_t* const* values,
    iree_host_size_t count) {
  return (loom_target_legalizer_provider_list_t){
      /*.count=*/count,
      /*.values=*/values,
  };
}

// Returns an empty target legalizer provider list.
static inline loom_target_legalizer_provider_list_t
loom_target_legalizer_provider_list_empty(void) {
  return (loom_target_legalizer_provider_list_t){0};
}

// Returns true if |list| has no target legalizer providers.
static inline bool loom_target_legalizer_provider_list_is_empty(
    loom_target_legalizer_provider_list_t list) {
  return list.count == 0;
}

typedef struct loom_target_legalizer_op_entry_t {
  // First legalizer row for the dialect-local op index.
  uint16_t entry_start;
  // Number of legalizer rows available for the dialect-local op index.
  uint16_t entry_count;
} loom_target_legalizer_op_entry_t;

// Returns an empty target legalizer op entry.
static inline loom_target_legalizer_op_entry_t
loom_target_legalizer_op_entry_empty(void) {
  return (loom_target_legalizer_op_entry_t){
      /*.entry_start=*/UINT16_MAX,
      /*.entry_count=*/0,
  };
}

// Returns true when |entry| has no legalizer rows.
static inline bool loom_target_legalizer_op_entry_is_empty(
    loom_target_legalizer_op_entry_t entry) {
  return entry.entry_count == 0;
}

typedef struct loom_target_legalizer_dialect_table_t {
  // Number of dialect-local op entries.
  uint16_t op_count;
  // Dense op entries indexed by loom_op_dialect_index.
  const loom_target_legalizer_op_entry_t* op_entries;
} loom_target_legalizer_dialect_table_t;

typedef struct loom_target_legalizer_registry_t {
  // First dialect id covered by dialects.
  uint8_t dialect_base_id;
  // Number of dense dialect slots.
  uint8_t dialect_count;
  // Dense dialect slots indexed by dialect id minus dialect_base_id.
  const loom_target_legalizer_dialect_table_t* dialects;
  // Legalizer rows copied from providers in dispatch order.
  const loom_target_legalizer_entry_t* entries;
  // Number of rows in entries.
  uint16_t entry_count;
} loom_target_legalizer_registry_t;

// Looks up the compact legalizer span for an op kind in a composed registry.
static inline loom_target_legalizer_op_entry_t
loom_target_legalizer_registry_lookup_kind(
    const loom_target_legalizer_registry_t* registry, loom_op_kind_t op_kind) {
  const uint8_t dialect_id = loom_op_dialect_id(op_kind);
  const uint8_t op_index = loom_op_dialect_index(op_kind);
  if (dialect_id < registry->dialect_base_id) {
    return loom_target_legalizer_op_entry_empty();
  }
  const uint8_t dialect_index = dialect_id - registry->dialect_base_id;
  if (dialect_index >= registry->dialect_count) {
    return loom_target_legalizer_op_entry_empty();
  }
  const loom_target_legalizer_dialect_table_t* dialect_table =
      &registry->dialects[dialect_index];
  if (op_index >= dialect_table->op_count) {
    return loom_target_legalizer_op_entry_empty();
  }
  return dialect_table->op_entries[op_index];
}

// Composes |providers| into a dense op-kind dispatch registry allocated from
// |arena|. Provider order is preserved within each op span.
iree_status_t loom_target_legalizer_registry_compose(
    const loom_target_legalizer_provider_t* const* providers,
    iree_host_size_t provider_count,
    loom_target_legalizer_registry_t* out_registry,
    iree_arena_allocator_t* arena);

// Queries whether |op| is already legal for the selected target contract.
iree_status_t loom_target_legalization_query_contract(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    loom_target_contract_query_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_LEGALIZATION_H_
