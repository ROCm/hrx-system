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

typedef struct loom_target_symbol_facts_t loom_target_symbol_facts_t;

// Resolved target record payload.
typedef struct loom_target_symbol_facts_t {
  // Common symbol-fact header.
  loom_symbol_facts_base_t base;

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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_TARGET_FACTS_H_
