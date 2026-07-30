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

// Projects family-owned facts from one verified target op.
//
// The common target fact base has already been initialized from generated
// table data and authored attrs. Implementations populate only their typed
// extension. Parsing and target verification make this callback infallible.
typedef void (*loom_target_fact_project_fn_t)(const loom_module_t* module,
                                              const loom_op_t* target_op,
                                              loom_target_facts_t* facts);

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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_TARGET_FACTS_H_
