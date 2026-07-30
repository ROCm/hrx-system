// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target dialect symbol facts.
//
// Target-like ops select generated target-family rows through typed attributes
// and project them into dense target-neutral facts. Backend-specific facts stay
// in backend packages; this layer owns only the shared target bundle shape.

#ifndef LOOM_OPS_TARGET_FACTS_H_
#define LOOM_OPS_TARGET_FACTS_H_

#include "iree/base/api.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/ir/ir.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_facts_t loom_target_facts_t;
typedef struct loom_target_symbol_facts_t loom_target_symbol_facts_t;

// Projects family-owned facts from one verified target op.
//
// The common target fact base has already been initialized from generated
// table data and authored attrs. Implementations populate only their typed
// extension. Parsing and target verification make this callback infallible.
typedef void (*loom_target_fact_project_fn_t)(const loom_module_t* module,
                                              const loom_op_t* target_op,
                                              loom_target_facts_t* facts);

// Returns whether one effective fact value satisfies a same-type requirement.
typedef bool (*loom_target_fact_satisfies_requirement_fn_t)(
    const loom_target_facts_t* effective,
    const loom_target_facts_t* requirement);

// Static type descriptor for one target-family fact representation.
struct loom_target_fact_type_t {
  // Stable fact-family name used in diagnostics.
  iree_string_view_t name;

  // Size of the family-owned facts object beginning with loom_target_facts_t.
  iree_host_size_t storage_size;

  // Optional family extension projector invoked once at the target-op boundary.
  loom_target_fact_project_fn_t project;

  // Optional satisfaction relation for distinct same-type fact values.
  loom_target_fact_satisfies_requirement_fn_t satisfies_requirement;
};

// Typed target-neutral facts projected from an authored target witness.
//
// Target-family fact structures embed this as their first field. Static type
// identity, the selected row, authored attr presence, and owned projected
// values replace downstream access to target IR.
struct loom_target_facts_t {
  // Static fact-family type and checked-dispatch identity.
  const loom_target_fact_type_t* fact_type;

  // Typed selector value that chose the generated base row.
  uint8_t selector;

  // Projected attrs explicitly present in the authored target witness.
  loom_target_authored_attr_set_t authored_attrs;

  // Owned common target projection after authored attrs are applied.
  loom_target_bundle_storage_t storage;
};

// Resolved target record payload.
typedef struct loom_target_symbol_facts_t {
  // Common symbol-fact header.
  loom_symbol_facts_base_t base;

  // Typed target facts projected from the authored target witness.
  const loom_target_facts_t* projection;

  // Borrowed typed target operation defining the record.
  loom_target_like_t target;

  // Module-local symbol reference for the target definition.
  loom_symbol_ref_t symbol;

  // Borrowed target symbol name from the module string table.
  iree_string_view_t name;

  // Typed selector value from the target-like op.
  uint8_t selector;

  // Borrowed generated target row bundle used as the base before overrides.
  const loom_target_bundle_t* row_bundle;

  // Materialized target bundle after typed attr projections are applied.
  loom_target_bundle_storage_t storage;
} loom_target_symbol_facts_t;

// Symbol fact domain used by generated target-like record descriptors.
extern const loom_symbol_fact_domain_t loom_target_symbol_fact_domain;

// Casts generic symbol facts to target facts when domains match.
const loom_target_symbol_facts_t* loom_target_symbol_facts_cast(
    const loom_symbol_facts_base_t* facts);

// Returns whether projected |attr_index| was explicitly present in the
// authored target witness.
static inline bool loom_target_facts_attr_is_authored(
    const loom_target_facts_t* facts, uint8_t attr_index) {
  return loom_target_authored_attr_set_contains(&facts->authored_attrs,
                                                attr_index);
}

// Returns whether |effective| satisfies every requirement in |requirement|.
//
// Distinct values must have the same static fact type. Family relations
// dispatch directly through that type and never inspect target IR or scan a
// provider registry.
bool loom_target_facts_satisfy_requirement(
    const loom_target_facts_t* effective,
    const loom_target_facts_t* requirement);

// Common structural relation for target families whose selector, snapshot,
// and configuration fully define compatibility. Function ABI and export facts
// do not participate.
bool loom_target_facts_structural_satisfy_requirement(
    const loom_target_facts_t* effective,
    const loom_target_facts_t* requirement);

// Returns whether |effective_snapshot| satisfies the structural requirements
// in |target_requirement|. Representation widths and address spaces must
// match, fixed subgroup sizes must agree, and effective capacity limits must
// meet or exceed nonzero required limits.
bool loom_target_snapshot_satisfies_requirement(
    const loom_target_snapshot_t* effective_snapshot,
    const loom_target_snapshot_t* target_requirement);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_TARGET_FACTS_H_
