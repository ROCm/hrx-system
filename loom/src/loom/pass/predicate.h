// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared pass.where predicate contracts.

#ifndef LOOM_PASS_PREDICATE_H_
#define LOOM_PASS_PREDICATE_H_

#include "iree/base/api.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_pass_predicate_verify_context_t {
  // Module containing the pass.where op.
  const loom_module_t* pipeline_module;
  // pass.where op being verified.
  const loom_op_t* where_op;
  // Anchor kind active at the pass.where op.
  loom_pass_kind_t anchor_kind;
  // Predicate key resolved from the pass.where op.
  iree_string_view_t predicate;
} loom_pass_predicate_verify_context_t;

typedef struct loom_pass_predicate_evaluate_context_t {
  // Module containing the pass.where op.
  const loom_module_t* pipeline_module;
  // pass.where op being evaluated.
  const loom_op_t* where_op;
  // Anchor kind active at the pass.where op.
  loom_pass_kind_t anchor_kind;
  // Predicate key resolved from the pass.where op.
  iree_string_view_t predicate;
  // Typed execution environment capabilities active for this pass run.
  const loom_pass_environment_t* environment;
  // Target module currently being transformed.
  const loom_module_t* target_module;
  // Current symbol when anchor_kind is LOOM_PASS_FUNCTION.
  const loom_symbol_t* symbol;
  // Current function when anchor_kind is LOOM_PASS_FUNCTION.
  loom_func_like_t function;
} loom_pass_predicate_evaluate_context_t;

// Validates provider-owned pass.where predicate syntax.
typedef iree_status_t (*loom_pass_predicate_verify_callback_t)(
    void* user_data, const loom_pass_predicate_verify_context_t* context);

// Evaluates provider-owned pass.where predicates against the current anchor.
typedef iree_status_t (*loom_pass_predicate_evaluate_callback_t)(
    void* user_data, const loom_pass_predicate_evaluate_context_t* context,
    bool* out_match);

typedef struct loom_pass_predicate_provider_t {
  // Optional callback used to validate provider-owned predicate schemas.
  loom_pass_predicate_verify_callback_t verify;
  // Optional callback used to evaluate provider-owned predicates.
  loom_pass_predicate_evaluate_callback_t evaluate;
  // Opaque caller data passed to |verify| and |evaluate|.
  void* user_data;
} loom_pass_predicate_provider_t;

// Resolves a user-facing trait spelling to its trait bits.
bool loom_pass_predicate_lookup_trait(iree_string_view_t name,
                                      loom_trait_flags_t* out_trait_flags);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_PREDICATE_H_
