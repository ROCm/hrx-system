// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pass pipeline program interpreter.
//
// Executes the compact instruction program produced by program.h. The
// interpreter owns only transient execution state: per-invocation pass arenas,
// deterministic symbol snapshots, current anchor context, and diagnostic
// provenance. The compiled program remains immutable and reusable.

#ifndef LOOM_PASS_INTERPRETER_H_
#define LOOM_PASS_INTERPRETER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/pass/environment.h"
#include "loom/pass/program.h"
#include "loom/pass/trace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_pass_interpreter_t loom_pass_interpreter_t;
typedef struct loom_pass_report_t loom_pass_report_t;

typedef struct loom_pass_interpreter_options_t {
  // Shared block pool used for pass instance, scratch, and snapshot arenas.
  iree_arena_block_pool_t* block_pool;
  // Optional provider for pass.where predicates outside the core built-ins.
  loom_pass_predicate_provider_t predicate_provider;
  // Optional structured diagnostic emitter copied into every pass instance.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Caller-owned execution environment capabilities.
  loom_pass_environment_t environment;
  // Optional caller-owned execution report appended as passes run.
  loom_pass_report_t* report;
  // Optional caller-owned IR trace receiving selected pass boundaries.
  loom_pass_trace_t* trace;
} loom_pass_interpreter_options_t;

// Executes a module-root compiled pass program.
//
// Status is reserved for infrastructure failure while compiling or running the
// interpreter. Pass-emitted diagnostics are counted in |out_result| and do not
// become iree_status_t failures.
iree_status_t loom_pass_interpreter_run_module(
    const loom_pass_program_t* program, loom_module_t* module,
    const loom_pass_interpreter_options_t* options,
    loom_pass_run_result_t* out_result);

// Executes a function-root compiled pass program on one function-like symbol.
//
// Status is reserved for infrastructure failure while compiling or running the
// interpreter. Pass-emitted diagnostics are counted in |out_result| and do not
// become iree_status_t failures.
iree_status_t loom_pass_interpreter_run_function(
    const loom_pass_program_t* program, loom_module_t* module,
    loom_func_like_t function, const loom_pass_interpreter_options_t* options,
    loom_pass_run_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_INTERPRETER_H_
