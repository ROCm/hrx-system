// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source boundary validation performed before target materialization.

#ifndef LOOM_CODEGEN_LOW_TRANSFORMS_PIPELINE_TARGET_LEGALIZE_PREFLIGHT_H_
#define LOOM_CODEGEN_LOW_TRANSFORMS_PIPELINE_TARGET_LEGALIZE_PREFLIGHT_H_

#include "iree/base/api.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/target/types.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_target_legalize_preflight_t {
  // Module owning the operation and values being checked.
  const loom_module_t* module;
  // Target-bound function whose entry assumptions carry topology contracts.
  loom_func_like_t function;
  // Selected target bundle used to render diagnostic context.
  const loom_target_bundle_t* target_bundle;
  // Structured sink receiving source boundary diagnostics.
  iree_diagnostic_emitter_t emitter;
  // Maximum number of diagnostics emitted, or zero for no limit.
  uint32_t max_errors;
  // Number of diagnostics emitted by this function invocation.
  uint32_t error_count;
} loom_low_target_legalize_preflight_t;

void loom_low_target_legalize_preflight_initialize(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* target_bundle,
    iree_diagnostic_emitter_t emitter, uint32_t max_errors,
    loom_low_target_legalize_preflight_t* out_preflight);

// Checks authored assumption predicates against established value facts.
// Contradictions are emitted as source diagnostics and counted in |preflight|.
iree_status_t loom_low_target_legalize_preflight_check_assume(
    loom_low_target_legalize_preflight_t* preflight, const loom_op_t* op,
    const loom_value_fact_table_t* fact_table);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TRANSFORMS_PIPELINE_TARGET_LEGALIZE_PREFLIGHT_H_
