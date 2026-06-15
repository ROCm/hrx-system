// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Kernel async stream legality analysis.
//
// This analysis proves kernel.async group/wait streams are in the
// straight-line form required by target lowering. Local op verifiers check
// token types, memory spaces, static footprints, and cache policies; this
// analysis checks the temporal stream contract that depends on program order.

#ifndef LOOM_ANALYSIS_KERNEL_ASYNC_LEGALITY_H_
#define LOOM_ANALYSIS_KERNEL_ASYNC_LEGALITY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_kernel_async_legality_options_t {
  // Scratch arena for stream, movement, and view-region analysis storage.
  iree_arena_allocator_t* arena;
  // Active local value domain for the function being verified.
  const loom_local_value_domain_t* value_domain;
  // Borrowed function-local value facts. The analysis only reads this table.
  loom_value_fact_table_t* fact_table;
  // Structured diagnostic emitter for user legality failures.
  iree_diagnostic_emitter_t emitter;
  // Name of the phase reporting diagnostics, such as "source-low" or
  // "kernel-async-legality".
  iree_string_view_t phase_name;
} loom_kernel_async_legality_options_t;

typedef struct loom_kernel_async_legality_result_t {
  // Number of error diagnostics emitted.
  uint32_t error_count;
  // Number of blocks checked for async stream legality.
  uint64_t blocks_checked;
  // Number of kernel.async.group ops checked.
  uint64_t groups_checked;
  // Number of kernel.async.wait ops checked.
  uint64_t waits_checked;
} loom_kernel_async_legality_result_t;

// Verifies the kernel async stream contract for one function-like body.
//
// User IR failures are emitted through |options->emitter| and counted in
// |out_result|. The analysis stops after the first stream violation because the
// pending-group state is no longer meaningful after that point. The function
// returns OK for user IR failures so callers can decide whether an illegal
// stream is a pass failure, a source-to-low diagnostic, or another
// production-path gate. Infrastructure failures such as arena allocation
// failures are returned as status failures.
iree_status_t loom_kernel_async_legality_verify_function(
    const loom_module_t* module, loom_func_like_t function,
    const loom_kernel_async_legality_options_t* options,
    loom_kernel_async_legality_result_t* out_result);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_KERNEL_ASYNC_LEGALITY_H_
