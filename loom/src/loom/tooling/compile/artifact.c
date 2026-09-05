// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/compile/artifact.h"

static iree_status_t loom_artifact_candidate_publish_report(
    const loom_compile_options_t* options,
    const loom_artifact_candidate_t* candidate) {
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

static iree_status_t loom_artifact_candidate_initialize(
    const loom_artifact_provider_t* provider,
    const loom_compile_options_t* options, iree_allocator_t allocator,
    loom_artifact_candidate_t* out_candidate) {
  *out_candidate = (loom_artifact_candidate_t){
      .host_allocator = allocator,
      .provider = provider,
  };
  loom_target_compile_report_t* report =
      options->report != NULL ? &out_candidate->compile_report : NULL;
  if (report == NULL) {
    return iree_ok_status();
  }
  const iree_allocator_t report_allocator =
      iree_allocator_is_null(options->report->allocator) ? iree_allocator_null()
                                                         : allocator;
  IREE_RETURN_IF_ERROR(loom_target_compile_report_clone(
      options->report, report_allocator, report));
  loom_target_compile_report_initialize_if_empty(report, report_allocator);
  report->artifact_kind = provider->artifact_kind;
  report->backend_name = provider->name;
  report->target_family_name = provider->target_profile_type->name;
  return iree_ok_status();
}

static void loom_artifact_candidate_record_report_status(
    const loom_compile_options_t* options, loom_artifact_candidate_t* candidate,
    iree_status_code_t status_code) {
  loom_target_compile_report_t* report =
      options->report != NULL ? &candidate->compile_report : NULL;
  if (report == NULL) {
    return;
  }
  report->artifact_kind = candidate->provider->artifact_kind;
  report->backend_name = candidate->provider->name;
  report->target_family_name = candidate->provider->target_profile_type->name;
  if (candidate->compiled) {
    report->target_key = candidate->artifact.target_key;
    report->artifact_format = loom_target_artifact_format_name(
        candidate->artifact.target_artifact_format);
    if (candidate->artifact.executable_data != NULL) {
      loom_target_compile_report_record_artifact_size(
          report,
          iree_byte_sequence_length(candidate->artifact.executable_data));
    }
  }
  loom_target_compile_report_record_status(report, status_code);
}

static iree_status_t loom_artifact_candidate_emit(
    const loom_artifact_provider_t* provider,
    const loom_artifact_target_t* target, loom_module_t* module,
    const loom_compile_options_t* options, iree_allocator_t allocator,
    loom_artifact_candidate_t* candidate) {
  if (provider->emit_artifact == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "artifact provider '%.*s' is missing required "
                            "emit hook",
                            (int)provider->name.size, provider->name.data);
  }

  loom_target_compile_report_t* report =
      options->report != NULL ? &candidate->compile_report : NULL;
  loom_compile_options_t provider_options = *options;
  provider_options.report = report;
  iree_status_t status = provider->emit_artifact(
      provider, module, target, &provider_options, allocator,
      &candidate->compiled, &candidate->artifact);
  if (iree_status_is_ok(status) && candidate->compiled) {
    IREE_ASSERT(candidate->artifact.target_bundle != NULL);
    IREE_ASSERT(candidate->artifact.target_artifact_data != NULL);
    IREE_ASSERT_GT(
        iree_byte_sequence_length(candidate->artifact.target_artifact_data), 0);
    IREE_ASSERT(candidate->artifact.executable_data != NULL);
    IREE_ASSERT_GT(
        iree_byte_sequence_length(candidate->artifact.executable_data), 0);
    IREE_ASSERT(candidate->artifact.sidecar_count == 0 ||
                candidate->artifact.sidecars != NULL);
  }
  return status;
}

iree_status_t loom_artifact_candidate_emit_target(
    const loom_artifact_provider_t* provider,
    const loom_artifact_target_t* target, loom_module_t* module,
    const loom_compile_options_t* options, iree_allocator_t allocator,
    loom_artifact_candidate_t* out_candidate) {
  if (target == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "artifact emission requires a selected target");
  }
  iree_status_t status = loom_artifact_candidate_initialize(
      provider, options, allocator, out_candidate);
  if (iree_status_is_ok(status)) {
    status = loom_artifact_candidate_emit(provider, target, module, options,
                                          allocator, out_candidate);
  }
  loom_artifact_candidate_record_report_status(options, out_candidate,
                                               iree_status_code(status));
  status = iree_status_join(
      status, loom_artifact_candidate_publish_report(options, out_candidate));
  if (!iree_status_is_ok(status)) {
    loom_artifact_candidate_deinitialize(out_candidate);
  }
  return status;
}

iree_status_t loom_artifact_candidate_emit_module_target(
    const loom_artifact_provider_t* provider, loom_module_t* module,
    const loom_compile_options_t* options, iree_allocator_t allocator,
    loom_artifact_candidate_t* out_candidate) {
  iree_status_t status = loom_artifact_candidate_initialize(
      provider, options, allocator, out_candidate);
  if (iree_status_is_ok(status)) {
    const loom_artifact_target_t authored_target = {0};
    status = loom_artifact_candidate_emit(provider, &authored_target, module,
                                          options, allocator, out_candidate);
  }
  loom_artifact_candidate_record_report_status(options, out_candidate,
                                               iree_status_code(status));
  status = iree_status_join(
      status, loom_artifact_candidate_publish_report(options, out_candidate));
  if (!iree_status_is_ok(status)) {
    loom_artifact_candidate_deinitialize(out_candidate);
  }
  return status;
}

void loom_artifact_candidate_deinitialize(
    loom_artifact_candidate_t* candidate) {
  if (candidate == NULL) {
    return;
  }
  if (candidate->provider != NULL &&
      candidate->provider->deinitialize_artifact != NULL) {
    candidate->provider->deinitialize_artifact(
        candidate->provider, &candidate->artifact, candidate->host_allocator);
  }
  loom_target_compile_report_deinitialize(&candidate->compile_report);
  *candidate = (loom_artifact_candidate_t){0};
}
