// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target dialect symbol fact projection.
//
// This is the sole bridge from authored target-like ops to immutable typed
// target facts. Facts-only compiler consumers include loom/target/facts.h and
// never depend on this target-IR boundary.

#ifndef LOOM_OPS_TARGET_FACTS_H_
#define LOOM_OPS_TARGET_FACTS_H_

#include "iree/base/api.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/ir/ir.h"
#include "loom/target/facts.h"
#include "loom/target/projection.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_symbol_facts_t loom_target_symbol_facts_t;

// Verified target-record payload independent of its source representation.
//
// Attributes retain the identity domain of |strings|. Projectors may read
// scalar attributes directly and resolve STRING attributes through
// loom_target_record_view_string().
typedef struct loom_target_record_view_t {
  // Target-family projection descriptor.
  const loom_target_like_descriptor_t* descriptor;

  // Borrowed target record name.
  iree_string_view_t name;

  // Borrowed dense attribute slots in descriptor order.
  const loom_attribute_t* attributes;

  // Number of entries in |attributes|.
  uint8_t attribute_count;

  // Concrete selector row established by source verification.
  uint8_t selector;

  // Borrowed string identity table used by STRING attributes.
  struct {
    // Borrowed string views indexed by STRING payload IDs.
    const iree_string_view_t* values;

    // Number of entries in |values|.
    iree_host_size_t count;
  } strings;
} loom_target_record_view_t;

// Returns one target-record attribute or ABSENT when |attribute_index| is not
// present in the source representation.
static inline loom_attribute_t loom_target_record_view_attribute(
    const loom_target_record_view_t* record, uint8_t attribute_index) {
  return attribute_index < record->attribute_count
             ? record->attributes[attribute_index]
             : loom_attr_absent();
}

// Resolves one present STRING attribute in |record|'s source identity domain.
static inline iree_string_view_t loom_target_record_view_string(
    const loom_target_record_view_t* record, loom_attribute_t attribute) {
  IREE_ASSERT(attribute.kind == LOOM_ATTR_STRING);
  const loom_string_id_t string_id = loom_attr_as_string_id(attribute);
  IREE_ASSERT(string_id < record->strings.count);
  return record->strings.values[string_id];
}

// Projects family-owned facts from one verified target record.
//
// The common target fact base has already been initialized from generated
// table data and authored attrs. Implementations populate only their typed
// extension. Parsing and target verification make this callback infallible.
typedef void (*loom_target_fact_project_fn_t)(
    const loom_target_record_view_t* record, loom_target_facts_t* facts);

// Authored target-op projection adapter named by a target-like descriptor.
struct loom_target_fact_projector_t {
  // Projects one verified target op into its family-owned typed fact extension.
  loom_target_fact_project_fn_t project;
};

// Resolved target record payload.
typedef struct loom_target_symbol_facts_t {
  // Common symbol-fact header.
  loom_symbol_facts_base_t base;

  // Typed target facts projected from the authored target witness.
  const loom_target_facts_t* projection;

  // Module-local symbol reference for the target definition.
  loom_symbol_ref_t symbol;

  // Borrowed target symbol name from the module string table.
  iree_string_view_t name;
} loom_target_symbol_facts_t;

// Symbol fact domain used by generated target-like record descriptors.
extern const loom_symbol_fact_domain_t loom_target_symbol_fact_domain;

// Casts generic symbol facts to target facts when domains match.
const loom_target_symbol_facts_t* loom_target_symbol_facts_cast(
    const loom_symbol_facts_base_t* facts);

// Projects one verified target record into caller-allocated typed fact
// storage. |row_bundle| is the selector-resolved row established while
// verifying |record|. |out_facts| must provide
// record->descriptor->fact_type->storage_size bytes and remains owned by the
// caller.
void loom_target_facts_project_record(const loom_target_record_view_t* record,
                                      const loom_target_bundle_t* row_bundle,
                                      loom_target_facts_t* out_facts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_TARGET_FACTS_H_
