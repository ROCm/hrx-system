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
typedef struct loom_local_value_domain_t loom_local_value_domain_t;
typedef struct loom_motion_read_barrier_table_t
    loom_motion_read_barrier_table_t;
typedef struct loom_target_legalizer_entry_t loom_target_legalizer_entry_t;
typedef struct loom_target_legalizer_rule_t loom_target_legalizer_rule_t;
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

typedef enum loom_target_boundary_representation_e {
  // Semantic boundaries still carry authored logical values.
  LOOM_TARGET_BOUNDARY_REPRESENTATION_LOGICAL = 0,
  // Boundary projection has materialized target physical carriers.
  LOOM_TARGET_BOUNDARY_REPRESENTATION_PROJECTED = 1,
} loom_target_boundary_representation_t;

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
  // Requires an explicit unsupported contract before applying the recipe or
  // rejecting it under require-native. An unanswered query can still have a
  // native lowering. Reference-only explicitly requests the recipe instead.
  LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION = 1u << 1,
  // Requires an unanswered contract query before applying the recipe or
  // rejecting it under require-native. An explicit target rejection remains
  // authoritative. Reference-only explicitly requests the recipe instead.
  LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_UNHANDLED = 1u << 2,
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
  // Borrowed immutable target facts selected for this function.
  const loom_target_facts_t* target_facts;
  // Low descriptor set selected by the target bundle.
  const loom_low_descriptor_set_t* descriptor_set;
  // Target packet candidates selected for |descriptor_set|, or NULL.
  const loom_target_vector_packet_policy_t* vector_packet_policy;
  // Target packet and scalar-word representation contract for shared
  // packed-bitstream legalization, or NULL when disabled.
  const loom_target_bitstream_policy_t* bitstream_policy;
  // Complete target admission query for authored source vector carriers.
  // Semantic rewrites that consume an aggregate may precede this admission;
  // scalar fallback must not.
  loom_target_vector_carrier_supported_callback_t
      source_vector_carrier_supported;
  // True when the source function signature contains a vector outside the
  // target's callable carrier domain. Reference fallback leaves the body
  // intact so final legality can diagnose the boundary without first
  // expanding it.
  bool source_function_has_unsupported_vector_carrier;
  // Source value facts visible to legalizers.
  const loom_value_fact_table_t* fact_table;
  // Analyzed view-region table visible to legalizers.
  const loom_view_region_table_t* view_regions;
  // Immutable-snapshot owner for legalizers requiring function analyses.
  loom_target_contract_function_analysis_callback_t function_analysis;
  // Lazy block-local read motion barriers visible to legalizers.
  loom_motion_read_barrier_table_t* read_motion_barriers;
  // Function-local value domain extendable during the current callback.
  loom_local_value_domain_t* value_domain;
  // Active rewriter for mutating the function body.
  loom_rewriter_t* rewriter;
  // Packet graph creation observed during the active legalizer callback.
  struct {
    // Value checkpoint immediately before the first packet materialization.
    loom_value_id_t value_checkpoint;
    // True when value_checkpoint has been recorded.
    bool recorded;
  } component_packet_materialization;
  // Current legalization phase.
  loom_target_legalization_mode_t mode;
  // Representation state of callable, CFG, and LoopLike boundaries.
  loom_target_boundary_representation_t boundary_representation;
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

// Records that the active legalizer is about to materialize a packet graph.
// The target legalization driver uses the first checkpoint to give generated
// decomposable operations to the component planner before reference fallback.
static inline void
loom_target_legalization_record_component_packet_materialization(
    loom_target_legalization_context_t* context) {
  if (!context->component_packet_materialization.recorded) {
    context->component_packet_materialization.value_checkpoint =
        loom_rewriter_value_checkpoint(context->rewriter);
    context->component_packet_materialization.recorded = true;
  }
}

// Returns the common target bundle selected for |context|.
static inline const loom_target_bundle_t*
loom_target_legalization_context_bundle(
    const loom_target_legalization_context_t* context) {
  return loom_target_facts_bundle(context->target_facts);
}

typedef iree_status_t (*loom_target_legalizer_fn_t)(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

// Returns true when a legalizer entry applies to |op|. Matchers run before the
// target contract query and must remain cheap, allocation-free, and read-only.
typedef bool (*loom_target_legalizer_match_fn_t)(
    const loom_target_legalizer_entry_t* entry,
    const loom_target_legalization_context_t* context, const loom_op_t* op);

// Provider-owned rule copied into a composed legalizer registry.
struct loom_target_legalizer_rule_t {
  // Entry-specific behavior flags.
  loom_target_legalizer_entry_flags_t flags;
  // Op kind this legalizer can rewrite.
  loom_op_kind_t root_kind;
  // Applicable scalar/element types of operand zero, or zero if unconstrained.
  // A constrained rule's root kind must have at least one operand. Dispatch
  // checks this domain before querying the target contract or applying policy.
  loom_scalar_type_set_t first_operand_element_types;
  // Optional target-owned applicability predicate evaluated before querying
  // the target contract. NULL applies to every matching root and type.
  loom_target_legalizer_match_fn_t match;
  // Rewriter callback for root_kind.
  loom_target_legalizer_fn_t legalize;
};

// Composed legalizer entry used during legalization dispatch.
struct loom_target_legalizer_entry_t {
  // Entry-specific behavior flags.
  loom_target_legalizer_entry_flags_t flags;
  // Op kind this legalizer can rewrite.
  loom_op_kind_t root_kind;
  // Operand-zero type domain copied from the provider rule; zero is
  // unrestricted.
  loom_scalar_type_set_t first_operand_element_types;
  // Stable provider name attached while composing the dense registry.
  iree_string_view_t provider_name;
  // Rewrite strategy attached while composing the dense registry.
  loom_target_legalizer_strategy_t provider_strategy;
  // Optional target-owned applicability predicate copied from the rule.
  loom_target_legalizer_match_fn_t match;
  // Rewriter callback for root_kind.
  loom_target_legalizer_fn_t legalize;
};

struct loom_target_legalizer_provider_t {
  // Stable provider name for cold reports and diagnostics.
  iree_string_view_t name;
  // Strategy shared by legalizer rows from this provider.
  loom_target_legalizer_strategy_t strategy;
  // Legalizer rules contributed by this provider.
  const loom_target_legalizer_rule_t* rules;
  // Number of provider-owned rules.
  uint16_t rule_count;
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

// Owned storage for a composed target legalizer registry.
//
// Provider-owned names and callbacks remain borrowed from their defining
// compiler packages. The allocation contains only the dense dispatch tables
// and copied entry records required to query them efficiently.
typedef struct loom_target_legalizer_registry_storage_t {
  // Allocator used to release |allocation|.
  iree_allocator_t allocator;
  // Single allocation backing every pointer in |registry|.
  iree_byte_span_t allocation;
  // Immutable registry view over |allocation|.
  loom_target_legalizer_registry_t registry;
} loom_target_legalizer_registry_storage_t;

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

// Initializes owned storage by composing one or more ordered provider lists.
//
// List order and provider order within each list are preserved within every op
// span. The initialized storage must be deinitialized before |allocator| is
// released.
iree_status_t loom_target_legalizer_registry_storage_initialize(
    const loom_target_legalizer_provider_list_t* provider_lists,
    iree_host_size_t provider_list_count, iree_allocator_t allocator,
    loom_target_legalizer_registry_storage_t* out_storage);

// Releases storage initialized by
// loom_target_legalizer_registry_storage_initialize.
void loom_target_legalizer_registry_storage_deinitialize(
    loom_target_legalizer_registry_storage_t* storage);

// Returns the immutable registry view owned by |storage|.
const loom_target_legalizer_registry_t*
loom_target_legalizer_registry_storage_registry(
    const loom_target_legalizer_registry_storage_t* storage);

// Acquires complete function analyses for a memory or component legalizer.
iree_status_t loom_target_legalization_resolve_function_analysis(
    loom_target_legalization_context_t* context);

// Queries whether |op| is already legal for the selected target contract.
// Rejected-rule identity and diagnostic payloads may be omitted; final legality
// checking requests those details only for operations that remain unsupported.
iree_status_t loom_target_legalization_query_contract(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    loom_target_contract_query_result_t* out_result);

// Queries |op| through a scoped vector lane-count projection. The projection
// applies only for this query and does not mutate authored IR or context state.
// As above, rejected-rule diagnostic details may be omitted.
iree_status_t
loom_target_legalization_query_contract_with_vector_lane_projection(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    loom_target_contract_vector_lane_projection_t vector_lane_projection,
    loom_target_contract_query_result_t* out_result);

// Returns true when |type| has a source vector carrier under the selected
// target lowering policy. Non-vector types always return true. A concrete
// source passes its value identity; hypothetical types use
// LOOM_VALUE_ID_INVALID.
bool loom_target_legalization_has_source_vector_carrier(
    const loom_target_legalization_context_t* context, loom_type_t type,
    loom_value_id_t value_id);

// Returns true when every vector operand and result of |op| has a source
// carrier under the selected target lowering policy.
bool loom_target_legalization_op_has_source_vector_carriers(
    const loom_target_legalization_context_t* context, const loom_op_t* op);

// Returns true when |type| has a packet carrier under the target's dedicated
// packed-bitstream policy. Non-vector types always return true.
bool loom_target_legalization_has_bitstream_packet_carrier(
    const loom_target_legalization_context_t* context, loom_type_t type);

// Returns true when every vector operand and result of |op| has a carrier
// under the target's dedicated packed-bitstream policy.
bool loom_target_legalization_op_has_bitstream_packet_carriers(
    const loom_target_legalization_context_t* context, const loom_op_t* op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_LEGALIZATION_H_
