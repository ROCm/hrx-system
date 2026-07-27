// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/ireevm/candidate.h"

#include "loom/target/emit/ireevm/archive_emitter.h"

static iree_status_t loom_ireevm_run_candidate_publish_compile_report(
    const loom_run_candidate_compile_options_t* options,
    const loom_ireevm_run_candidate_t* candidate) {
  if (options->report == NULL) {
    return iree_ok_status();
  }
  loom_target_compile_report_t report = {0};
  IREE_RETURN_IF_ERROR(loom_target_compile_report_clone(
      &candidate->compile_report, options->report->allocator, &report));
  loom_target_compile_report_deinitialize(options->report);
  *options->report = report;
  return iree_ok_status();
}

iree_status_t loom_ireevm_run_candidate_emit(
    loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* options,
    iree_allocator_t allocator, loom_ireevm_run_candidate_t* out_candidate) {
  *out_candidate = (loom_ireevm_run_candidate_t){
      .host_allocator = allocator,
  };
  loom_target_compile_report_t* report =
      options->report != NULL ? &out_candidate->compile_report : NULL;
  if (report != NULL) {
    const iree_allocator_t report_allocator =
        iree_allocator_is_null(options->report->allocator)
            ? iree_allocator_null()
            : allocator;
    IREE_RETURN_IF_ERROR(loom_target_compile_report_clone(
        options->report, report_allocator, report));
    loom_target_compile_report_initialize_if_empty(report, report_allocator);
  }

  const loom_ireevm_archive_emit_options_t archive_emit_options = {
      .module_name = options->module_name,
      .diagnostic_sink = options->diagnostic_sink,
      .source_resolver = options->source_resolver,
      .max_errors = options->max_errors,
      .report = report,
  };
  iree_status_t status = loom_ireevm_emit_module_archive_from_ir(
      run_module->module, &archive_emit_options, allocator,
      &out_candidate->emitted, &out_candidate->archive);
  status = iree_status_join(
      status,
      loom_ireevm_run_candidate_publish_compile_report(options, out_candidate));
  if (!iree_status_is_ok(status)) {
    loom_ireevm_run_candidate_deinitialize(out_candidate);
  }
  return status;
}

void loom_ireevm_run_candidate_deinitialize(
    loom_ireevm_run_candidate_t* candidate) {
  if (candidate == NULL) {
    return;
  }
  loom_ireevm_module_archive_deinitialize(&candidate->archive,
                                          candidate->host_allocator);
  loom_target_compile_report_deinitialize(&candidate->compile_report);
  *candidate = (loom_ireevm_run_candidate_t){0};
}
