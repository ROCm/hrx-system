// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable typed target facts.
//
// Target facts are compiler analysis state, independent of the target IR that
// may have authored them. Authored target-op projection is confined to the
// target symbol fact domain.

#ifndef LOOM_TARGET_FACTS_H_
#define LOOM_TARGET_FACTS_H_

#include "iree/base/api.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_facts_t loom_target_facts_t;

// Target-neutral fields whose explicit presence can affect specialization or
// must survive projection into durable IR.
typedef uint8_t loom_target_fact_field_t;
enum loom_target_fact_field_e {
  LOOM_TARGET_FACT_FIELD_CODEGEN_FORMAT = 0,
  LOOM_TARGET_FACT_FIELD_ARTIFACT_FORMAT,
  LOOM_TARGET_FACT_FIELD_DEFAULT_POINTER_BITWIDTH,
  LOOM_TARGET_FACT_FIELD_INDEX_BITWIDTH,
  LOOM_TARGET_FACT_FIELD_OFFSET_BITWIDTH,
  LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_SIZE_X,
  LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_SIZE_Y,
  LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_SIZE_Z,
  LOOM_TARGET_FACT_FIELD_MAX_FLAT_WORKGROUP_SIZE,
  LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_STORAGE_BYTES,
  LOOM_TARGET_FACT_FIELD_SUBGROUP_SIZE,
  LOOM_TARGET_FACT_FIELD_MAX_GRID_SIZE_X,
  LOOM_TARGET_FACT_FIELD_MAX_GRID_SIZE_Y,
  LOOM_TARGET_FACT_FIELD_MAX_GRID_SIZE_Z,
  LOOM_TARGET_FACT_FIELD_MAX_FLAT_GRID_SIZE,
  LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_COUNT_X,
  LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_COUNT_Y,
  LOOM_TARGET_FACT_FIELD_MAX_WORKGROUP_COUNT_Z,
  LOOM_TARGET_FACT_FIELD_MEMORY_SPACE_GENERIC,
  LOOM_TARGET_FACT_FIELD_MEMORY_SPACE_GLOBAL,
  LOOM_TARGET_FACT_FIELD_MEMORY_SPACE_WORKGROUP,
  LOOM_TARGET_FACT_FIELD_MEMORY_SPACE_CONSTANT,
  LOOM_TARGET_FACT_FIELD_MEMORY_SPACE_PRIVATE,
  LOOM_TARGET_FACT_FIELD_MEMORY_SPACE_HOST,
  LOOM_TARGET_FACT_FIELD_MEMORY_SPACE_DESCRIPTOR,
  LOOM_TARGET_FACT_FIELD_ABI,
  LOOM_TARGET_FACT_FIELD_EXPORT_SYMBOL,
  LOOM_TARGET_FACT_FIELD_LINKAGE,
  LOOM_TARGET_FACT_FIELD_CONTRACT_SET_KEY,
  LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS,
  LOOM_TARGET_FACT_FIELD_COUNT_,
};
static_assert(LOOM_TARGET_FACT_FIELD_COUNT_ <= 64,
              "target fact explicitness must fit in one word");

// Set of target-neutral fact fields.
typedef uint64_t loom_target_fact_field_set_t;

// Records explicit presence of |field|.
static inline void loom_target_fact_field_set_insert(
    loom_target_fact_field_set_t* set, loom_target_fact_field_t field) {
  *set |= UINT64_C(1) << field;
}

// Returns whether |field| is present in |set|.
static inline bool loom_target_fact_field_set_contains(
    loom_target_fact_field_set_t set, loom_target_fact_field_t field) {
  return iree_any_bit_set(set, UINT64_C(1) << field);
}

// Returns whether one effective fact identity satisfies a same-type identity
// requirement.
typedef bool (*loom_target_fact_satisfies_identity_requirement_fn_t)(
    const loom_target_facts_t* effective,
    const loom_target_facts_t* requirement);

// Returns whether one effective fact value satisfies a same-type
// specialization requirement.
typedef bool (*loom_target_fact_satisfies_specialization_requirement_fn_t)(
    const loom_target_facts_t* effective,
    const loom_target_facts_t* requirement);

// Rebinds family-owned views after the common fact storage changes.
typedef void (*loom_target_fact_rebind_fn_t)(loom_target_facts_t* facts);

// Returns a concise identity name derived from structured target facts.
//
// This is presentation-only. Target compatibility, specialization, and
// lowering must consume the structured facts rather than comparing this name.
typedef iree_string_view_t (*loom_target_fact_identity_name_fn_t)(
    const loom_target_facts_t* facts);

// Static type descriptor for one target-family fact representation.
struct loom_target_fact_type_t {
  // Stable target-family name used in diagnostics and pass predicates.
  iree_string_view_t name;

  // Size of the family-owned facts object beginning with loom_target_facts_t.
  iree_host_size_t storage_size;

  // Optional identity-domain relation for distinct same-type fact values.
  loom_target_fact_satisfies_identity_requirement_fn_t
      satisfies_identity_requirement;

  // Optional full specialization relation for distinct same-type fact values.
  loom_target_fact_satisfies_specialization_requirement_fn_t
      satisfies_specialization_requirement;

  // Optional family-owned rebind callback used only while constructing facts.
  loom_target_fact_rebind_fn_t rebind;

  // Optional presentation projection for diagnostics and reports.
  loom_target_fact_identity_name_fn_t identity_name;
};

// Typed target-neutral facts projected from available target information.
//
// Target-family fact structures embed this as their first field. Static type
// identity, selection provenance, and owned projected values replace
// downstream access to target IR.
struct loom_target_facts_t {
  // Static fact-family type and checked-dispatch identity.
  const loom_target_fact_type_t* fact_type;

  // Typed selector value that chose the generated base row.
  uint8_t selector;

  // Target-neutral semantic inputs explicitly supplied by IR or a profile.
  loom_target_fact_field_set_t explicit_fields;

  // Owned common target projection after explicit inputs are applied.
  loom_target_bundle_storage_t storage;
};

// Returns the immutable common target bundle projected into |facts|.
static inline const loom_target_bundle_t* loom_target_facts_bundle(
    const loom_target_facts_t* facts) {
  return facts ? &facts->storage.bundle : NULL;
}

// Returns a concise identity name derived from |facts|.
//
// Families without a structured presentation callback use the common bundle
// name as a diagnostic fallback.
static inline iree_string_view_t loom_target_facts_identity_name(
    const loom_target_facts_t* facts) {
  if (facts == NULL) {
    return iree_string_view_empty();
  }
  return facts->fact_type->identity_name != NULL
             ? facts->fact_type->identity_name(facts)
             : facts->storage.bundle.name;
}

// Returns whether |field| was supplied as an explicit semantic input.
static inline bool loom_target_facts_field_is_explicit(
    const loom_target_facts_t* facts, loom_target_fact_field_t field) {
  return loom_target_fact_field_set_contains(facts->explicit_fields, field);
}

// Returns whether the identity of |effective| satisfies |requirement|.
//
// The same fact object satisfies itself. Distinct values require the same
// static fact type and a family-defined identity relation. Common target
// projections are deliberately not treated as an identity fallback.
bool loom_target_facts_satisfy_identity_requirement(
    const loom_target_facts_t* effective,
    const loom_target_facts_t* requirement);

// Identity relation for simple target families completely identified by their
// typed selector.
bool loom_target_facts_selector_satisfies_identity_requirement(
    const loom_target_facts_t* effective,
    const loom_target_facts_t* requirement);

// Returns whether |effective| satisfies every specialization requirement in
// |requirement|.
//
// Distinct values must have the same static fact type. Family relations
// dispatch directly through that type and never inspect target IR or scan a
// provider registry.
bool loom_target_facts_satisfy_specialization_requirement(
    const loom_target_facts_t* effective,
    const loom_target_facts_t* requirement);

// Returns whether |lhs| and |rhs| carry the same durable target semantics.
//
// Equivalent facts have the same static fact type and explicit-input
// provenance, and each satisfies the other's family-defined specialization
// requirements. This is stricter than having the same effective values:
// explicitly supplying a value equal to a selector default remains observable
// so later specialization cannot reinterpret the target differently.
bool loom_target_facts_are_equivalent(const loom_target_facts_t* lhs,
                                      const loom_target_facts_t* rhs);

// Common structural relation for target families whose selector, snapshot,
// and configuration fully define compatibility. Function ABI and export facts
// do not participate.
bool loom_target_facts_structural_satisfy_specialization_requirement(
    const loom_target_facts_t* effective,
    const loom_target_facts_t* requirement);

// Returns whether |effective_snapshot| satisfies the structural requirements
// in |target_requirement|. Representation widths and address spaces must
// match, fixed subgroup sizes must agree, and effective capacity limits must
// meet or exceed nonzero required limits.
bool loom_target_snapshot_satisfies_specialization_requirement(
    const loom_target_snapshot_t* effective_snapshot,
    const loom_target_snapshot_t* target_requirement);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_FACTS_H_
