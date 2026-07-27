// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Func symbol facts.
//
// These facts are the dense, direct-indexed func context shared by source funcs
// and target-low funcs. Dialects expose structure through the
// FuncLike interface; passes consume this payload instead of walking attrs or
// looking for companion global records.

#ifndef LOOM_OPS_FUNC_SYMBOL_FACTS_H_
#define LOOM_OPS_FUNC_SYMBOL_FACTS_H_

#include "iree/base/api.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/ir/ir.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_func_symbol_facts_t loom_func_symbol_facts_t;

// Resolved func symbol payload.
typedef struct loom_func_symbol_facts_t {
  // Common symbol-fact header.
  loom_symbol_facts_base_t base;

  // Defining func op.
  loom_op_t* func_op;

  // Module-local symbol reference for the func definition.
  loom_symbol_ref_t symbol;

  // Borrowed func symbol name from the module string table.
  iree_string_view_t name;

  // Visibility enum value from the func interface.
  uint8_t visibility;

  // Calling convention enum value from the func interface.
  uint8_t calling_convention;

  // Purity enum value from the func interface.
  uint8_t purity;

  // Execution temperature enum value from the func interface.
  uint8_t temperature;

  // Inline policy enum value from the func interface.
  uint8_t inline_policy;

  // True when the func op owns an implementation body.
  bool has_body;

  // Implementation contract string ID for provider funcs, or invalid.
  loom_string_id_t implements_id;

  // Borrowed implementation contract key, or empty for non-providers.
  iree_string_view_t implements;

  // Provider priority for funcs with an implementation contract.
  int64_t priority;

  // Borrowed argument value IDs in signature order.
  const loom_value_id_t* argument_ids;

  // Number of argument value IDs.
  uint16_t argument_count;

  // Borrowed result value IDs in signature order.
  const loom_value_id_t* result_ids;

  // Number of result value IDs.
  uint16_t result_count;

  // Borrowed predicate list for provider or callable selection.
  const loom_predicate_t* predicates;

  // Number of predicate entries.
  uint16_t predicate_count;

  // True when the func is an external import declaration.
  bool imports;

  // Borrowed import module name for external declarations, or empty.
  iree_string_view_t import_module;

  // Borrowed import symbol name for external declarations, or empty.
  iree_string_view_t import_symbol;

  // Module-local target record symbol, or null for target-independent funcs.
  loom_symbol_ref_t target_symbol;

  // True when the func declares a callable target ABI override.
  bool has_abi;

  // Callable ABI override, or UNKNOWN when has_abi is false.
  loom_target_abi_kind_t abi_kind;

  // Borrowed target ABI attribute overrides from the func op.
  loom_named_attr_slice_t abi_attrs;

  // True when the func declares an artifact/package export.
  bool exports;

  // Borrowed export symbol name, or empty to preserve the func symbol name.
  iree_string_view_t export_symbol;

  // True when export_linkage was explicitly provided.
  bool has_export_linkage;

  // Export linkage override when has_export_linkage is true.
  loom_target_linkage_t export_linkage;

} loom_func_symbol_facts_t;

// Symbol fact domain used by generated func symbol descriptors.
extern const loom_symbol_fact_domain_t loom_func_symbol_fact_domain;

// Casts generic symbol facts to func facts when domains match.
const loom_func_symbol_facts_t* loom_func_symbol_facts_cast(
    const loom_symbol_facts_base_t* facts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_FUNC_SYMBOL_FACTS_H_
