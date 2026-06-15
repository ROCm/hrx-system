// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE VM archive candidates produced by Loom execution tooling.

#ifndef LOOM_TOOLING_EXECUTION_IREEVM_CANDIDATE_H_
#define LOOM_TOOLING_EXECUTION_IREEVM_CANDIDATE_H_

#include "iree/base/api.h"
#include "loom/target/emit/ireevm/module_archive.h"
#include "loom/tooling/execution/compile_options.h"
#include "loom/tooling/execution/session.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_ireevm_run_candidate_t {
  // Host allocator used for owned candidate storage.
  iree_allocator_t host_allocator;
  // True when archive emission produced |archive|.
  bool emitted;
  // Structured report for this candidate's emission.
  loom_target_compile_report_t compile_report;
  // VM bytecode archive produced by the IREE VM emitter path.
  loom_ireevm_module_archive_t archive;
} loom_ireevm_run_candidate_t;

// Emits |run_module| to an IREE VM archive candidate. The module must already
// contain prepared target-low VM functions and imports.
iree_status_t loom_ireevm_run_candidate_emit(
    loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* options,
    iree_allocator_t allocator, loom_ireevm_run_candidate_t* out_candidate);

// Releases all artifact storage owned by |candidate|.
void loom_ireevm_run_candidate_deinitialize(
    loom_ireevm_run_candidate_t* candidate);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_IREEVM_CANDIDATE_H_
