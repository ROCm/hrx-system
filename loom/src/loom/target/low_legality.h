// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-low source legality verification.
//
// This verifier runs before source-to-low lowering. It checks the source
// function against the selected target bundle and linked low descriptor set,
// then gives target-family providers a hook to accept, reject, or report on
// target-owned source contracts such as memory-addressing forms, packed dot, or
// matrix ops. Providers are refinement points; table-driven lowering rules are
// still the source of truth for whether a source op can be selected.
// User IR failures are reported through structured backend diagnostics;
// infrastructure and compiler-configuration failures are returned as status.

#ifndef LOOM_TARGET_LOW_LEGALITY_H_
#define LOOM_TARGET_LOW_LEGALITY_H_

#include "iree/base/api.h"
#include "loom/error/emitter.h"
#include "loom/error/error_defs.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"
#include "loom/target/contract.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/types.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_low_legality_context_t
    loom_target_low_legality_context_t;
typedef struct loom_target_low_legality_provider_t
    loom_target_low_legality_provider_t;
typedef struct loom_local_value_domain_t loom_local_value_domain_t;
typedef struct loom_view_region_table_t loom_view_region_table_t;

typedef enum loom_target_low_legality_diagnostic_flag_bits_e {
  // Emit target source-memory decision remarks from providers that support
  // them, including access, cache-policy, and prefetch selection.
  LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS = 1u << 0,
  // All target-low legality diagnostic flags known to this header.
  LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_ALL =
      LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_MEMORY_ACCESS,
} loom_target_low_legality_diagnostic_flag_bits_t;
typedef uint32_t loom_target_low_legality_diagnostic_flags_t;

typedef enum loom_target_low_structural_legality_flag_bits_e {
  // Source SCF regions are still legal in the caller's current phase. CFG-mode
  // source-to-low leaves this unset and requires an explicit structural
  // lowering pass before executable target-low lowering.
  LOOM_TARGET_LOW_STRUCTURAL_LEGALITY_ALLOW_SOURCE_SCF = 1u << 0,
  // All structural legality flags known to this header.
  LOOM_TARGET_LOW_STRUCTURAL_LEGALITY_ALL =
      LOOM_TARGET_LOW_STRUCTURAL_LEGALITY_ALLOW_SOURCE_SCF,
} loom_target_low_structural_legality_flag_bits_t;
typedef uint32_t loom_target_low_structural_legality_flags_t;

// Bitset of built-in dialect ids a target-low legality provider can handle.
typedef uint32_t loom_target_low_legality_builtin_dialect_bits_t;

// Returns true if |bits| contains |dialect_id|.
static inline bool loom_target_low_legality_builtin_dialect_bits_contain(
    loom_target_low_legality_builtin_dialect_bits_t bits, uint8_t dialect_id) {
  return dialect_id < 32 && iree_any_bit_set(bits, 1u << dialect_id);
}

typedef iree_status_t (*loom_target_low_legality_try_op_fn_t)(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

typedef bool (*loom_target_low_legality_type_supported_fn_t)(
    void* user_data, const loom_module_t* module, loom_type_t type);

typedef struct loom_target_low_legality_type_supported_callback_t {
  // Optional callback reporting target-owned source types that can be mapped to
  // target-low values by the selected lowering policy.
  loom_target_low_legality_type_supported_fn_t fn;
  // Caller-owned payload passed to |fn|.
  void* user_data;
} loom_target_low_legality_type_supported_callback_t;

struct loom_target_low_legality_provider_t {
  // Stable provider name available to callbacks when emitting diagnostics.
  iree_string_view_t name;
  // Built-in source dialect ids this provider can answer.
  loom_target_low_legality_builtin_dialect_bits_t builtin_dialect_bits;
  // Attempts to verify one source op. Sets |out_handled| false when the op does
  // not belong to this provider.
  loom_target_low_legality_try_op_fn_t try_verify_op;
};

typedef struct loom_target_low_legality_provider_list_t {
  // Total number of values in the list.
  iree_host_size_t count;
  // Value list or NULL if no values.
  const loom_target_low_legality_provider_t* const* values;
} loom_target_low_legality_provider_list_t;

// Creates a source legality provider list from borrowed storage.
static inline loom_target_low_legality_provider_list_t
loom_target_low_legality_provider_list_make(
    const loom_target_low_legality_provider_t* const* values,
    iree_host_size_t count) {
  loom_target_low_legality_provider_list_t list = {
      /*.count=*/count,
      /*.values=*/values,
  };
  return list;
}

// Returns an empty source legality provider list.
static inline loom_target_low_legality_provider_list_t
loom_target_low_legality_provider_list_empty(void) {
  loom_target_low_legality_provider_list_t list = {0};
  return list;
}

// Returns true if |list| has no source legality providers.
static inline bool loom_target_low_legality_provider_list_is_empty(
    loom_target_low_legality_provider_list_t list) {
  return list.count == 0;
}

typedef struct loom_target_low_legality_options_t {
  // Target bundle selected for this source-to-low lowering attempt.
  const loom_target_bundle_t* bundle;
  // Target-owned payload associated with |bundle|, or NULL.
  const void* target_data;
  // Module-local target record symbol selected for this lowering attempt.
  loom_symbol_ref_t target_ref;
  // Low descriptor registry linked into the current compiler binary.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Catalog resolving compact diagnostic refs emitted by target contract
  // queries and target-owned legality providers.
  const loom_error_catalog_t* error_catalog;
  // Optional target-specific source legality providers.
  loom_target_low_legality_provider_list_t provider_list;
  // Optional contract query over the selected target lowering policy. This is
  // consulted before target-local providers so table-backed contracts stay the
  // source of truth for table-backed source-to-low emission.
  loom_target_contract_query_callback_t contract_query;
  // Optional target-owned source type mapping predicate. Missing keeps the
  // verifier limited to target-independent low-compatible types and registered
  // type semantics.
  loom_target_low_legality_type_supported_callback_t type_supported;
  // Caller-owned facts for |function|.
  loom_value_fact_table_t* fact_table;
  // Optional active value domain for |function|'s body. Providers can use this
  // to request shared function-local analyses without acquiring module scratch.
  const loom_local_value_domain_t* value_domain;
  // Structural source forms permitted by the caller's current phase.
  loom_target_low_structural_legality_flags_t structural_legality_flags;
  // Optional target-specific feedback diagnostics to emit during source
  // legality. Zero keeps legality quiet except for errors and provider-owned
  // mandatory contract remarks.
  loom_target_low_legality_diagnostic_flags_t diagnostic_flags;
  // Structured diagnostic emitter for user legality failures and remarks.
  iree_diagnostic_emitter_t emitter;
  // Maximum number of errors to emit before aborting the walk. Zero means no
  // limit.
  uint32_t max_errors;
} loom_target_low_legality_options_t;

typedef struct loom_target_low_legality_result_t {
  // Number of error diagnostics emitted.
  uint32_t error_count;
  // Number of remark diagnostics emitted.
  uint32_t remark_count;
  // Descriptor set selected by options.bundle, or NULL when selection failed
  // before verification started.
  const loom_low_descriptor_set_t* descriptor_set;
} loom_target_low_legality_result_t;

// Verifies that |function| is legal as source IR for target-low lowering under
// |options|.
//
// User IR legality failures are counted in |out_result| and emitted through
// options.emitter. The function still returns OK unless an infrastructure error
// such as malformed options, invalid provider tables, or registry lookup
// failure occurs.
iree_status_t loom_target_low_verify_function_legality(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_low_legality_options_t* options,
    loom_target_low_legality_result_t* out_result);

// Returns the source module being checked.
const loom_module_t* loom_target_low_legality_module(
    const loom_target_low_legality_context_t* context);

// Returns the function being checked.
loom_func_like_t loom_target_low_legality_function(
    const loom_target_low_legality_context_t* context);

// Returns the checked function's symbol name, or a placeholder when the
// checked subject is not a named function-like op.
iree_string_view_t loom_target_low_legality_function_name(
    const loom_target_low_legality_context_t* context);

// Returns the selected target bundle.
const loom_target_bundle_t* loom_target_low_legality_bundle(
    const loom_target_low_legality_context_t* context);

// Returns the selected low descriptor set.
const loom_low_descriptor_set_t* loom_target_low_legality_descriptor_set(
    const loom_target_low_legality_context_t* context);

// Returns source value facts available to target-specific legality providers.
const loom_value_fact_table_t* loom_target_low_legality_fact_table(
    const loom_target_low_legality_context_t* context);

// Returns a lazily analyzed view-region table when the caller supplied a value
// domain for the checked function. Standalone legality callers that do not
// provide a domain receive NULL.
iree_status_t loom_target_low_legality_view_regions(
    loom_target_low_legality_context_t* context,
    const loom_view_region_table_t** out_view_regions);

// Returns the optional feedback diagnostics requested by the caller.
loom_target_low_legality_diagnostic_flags_t
loom_target_low_legality_diagnostic_flags(
    const loom_target_low_legality_context_t* context);

// Emits a generated structured legality diagnostic.
iree_status_t loom_target_low_legality_emit_error_ref(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_error_ref_t error_ref, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count);

// Emits ERR_BACKEND_017 for a target memory-access selection decision.
iree_status_t loom_target_low_legality_record_memory_access(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t memory_space, iree_string_view_t operation_kind,
    iree_string_view_t packet_key, iree_string_view_t decision,
    uint32_t element_bytes, uint32_t vector_lanes,
    uint32_t dynamic_stride_bytes, uint32_t vector_lane_stride_bytes,
    uint32_t bank_stride_words, uint32_t bank_conflict_degree,
    iree_string_view_t bank_conflict_kind);

// Emits ERR_BACKEND_040 for a source memory cache-policy decision.
iree_status_t loom_target_low_legality_record_memory_cache_policy(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t memory_space, iree_string_view_t operation_kind,
    iree_string_view_t cache_scope, iree_string_view_t cache_temporal,
    iree_string_view_t decision_key, iree_string_view_t decision,
    iree_string_view_t encoding_key, bool scope_attr_present,
    int64_t scope_attr, bool th_attr_present, int64_t th_attr,
    bool nt_attr_present, int64_t nt_attr);

// Emits ERR_BACKEND_041 for a source memory prefetch decision.
iree_status_t loom_target_low_legality_record_memory_prefetch(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t memory_space, iree_string_view_t prefetch_intent,
    iree_string_view_t prefetch_locality, iree_string_view_t decision_key,
    iree_string_view_t decision, iree_string_view_t packet_key,
    int64_t immediate_offset, uint32_t scalar_byte_offset,
    iree_string_view_t dynamic_index_kind, uint32_t count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_LOW_LEGALITY_H_
