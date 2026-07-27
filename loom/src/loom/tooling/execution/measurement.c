// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/measurement.h"

enum {
  LOOM_RUN_MEASUREMENT_SUPPORTED_KIND_FLAGS = LOOM_RUN_MEASUREMENT_KIND_TIMING,
  LOOM_RUN_MEASUREMENT_KNOWN_KIND_FLAGS =
      LOOM_RUN_MEASUREMENT_KIND_TIMING | LOOM_RUN_MEASUREMENT_KIND_STATISTICS |
      LOOM_RUN_MEASUREMENT_KIND_DEEP_PROFILE,
};

void loom_run_measurement_options_initialize(
    loom_run_measurement_options_t* out_options) {
  *out_options = (loom_run_measurement_options_t){0};
}

void loom_run_measurement_result_initialize(
    loom_run_measurement_sample_t* samples, iree_host_size_t sample_capacity,
    loom_run_measurement_result_t* out_result) {
  *out_result = (loom_run_measurement_result_t){
      .samples = samples,
      .sample_capacity = sample_capacity,
  };
}

iree_string_view_t loom_run_measurement_boundary_name(
    loom_run_measurement_boundary_t boundary) {
  switch (boundary) {
    case LOOM_RUN_MEASUREMENT_BOUNDARY_COMPILE:
      return IREE_SV("compile");
    case LOOM_RUN_MEASUREMENT_BOUNDARY_PREPARE_CANDIDATE:
      return IREE_SV("prepare-candidate");
    case LOOM_RUN_MEASUREMENT_BOUNDARY_PREPARE_INVOCATION:
      return IREE_SV("prepare-invocation");
    case LOOM_RUN_MEASUREMENT_BOUNDARY_SUBMIT:
      return IREE_SV("submit");
    case LOOM_RUN_MEASUREMENT_BOUNDARY_INVOKE:
      return IREE_SV("invoke");
    case LOOM_RUN_MEASUREMENT_BOUNDARY_COLLECT_RESULTS:
      return IREE_SV("collect-results");
    case LOOM_RUN_MEASUREMENT_BOUNDARY_END_TO_END:
      return IREE_SV("end-to-end");
    default:
      return IREE_SV("unknown");
  }
}

static bool loom_run_measurement_boundary_is_single(
    loom_run_measurement_boundary_t boundary) {
  return boundary != 0 && (boundary & (boundary - 1)) == 0;
}

static iree_status_t loom_run_measurement_options_validate(
    const loom_run_measurement_options_t* options) {
  if (options->kind_flags & ~LOOM_RUN_MEASUREMENT_KNOWN_KIND_FLAGS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown measurement kind flags 0x%08X",
                            options->kind_flags);
  }
  if (options->boundary_flags & ~LOOM_RUN_MEASUREMENT_BOUNDARY_ALL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown measurement boundary flags 0x%08X",
                            options->boundary_flags);
  }
  const loom_run_measurement_kind_flags_t unsupported_kind_flags =
      options->kind_flags & ~LOOM_RUN_MEASUREMENT_SUPPORTED_KIND_FLAGS;
  if (unsupported_kind_flags != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "measurement kind flags 0x%08X are declared but not implemented",
        unsupported_kind_flags);
  }
  return iree_ok_status();
}

iree_status_t loom_run_measurement_scope_begin(
    const loom_run_measurement_options_t* options,
    loom_run_measurement_boundary_t boundary,
    loom_run_measurement_result_t* result,
    loom_run_measurement_scope_t* out_scope) {
  *out_scope = (loom_run_measurement_scope_t){0};
  IREE_RETURN_IF_ERROR(loom_run_measurement_options_validate(options));
  if (!loom_run_measurement_boundary_is_single(boundary)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "measurement boundary must be a single flag");
  }

  const loom_run_measurement_kind_flags_t active_kind_flags =
      options->kind_flags & LOOM_RUN_MEASUREMENT_SUPPORTED_KIND_FLAGS;
  if (active_kind_flags == 0 ||
      !iree_any_bit_set(options->boundary_flags, boundary)) {
    return iree_ok_status();
  }
  if (result == NULL || result->samples == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "measurement timing requires sample storage");
  }
  if (result->sample_count >= result->sample_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "measurement sample count %" PRIhsz
                            " exceeds capacity %" PRIhsz,
                            result->sample_count + 1, result->sample_capacity);
  }

  const iree_host_size_t sample_index = result->sample_count++;
  loom_run_measurement_sample_t* sample = &result->samples[sample_index];
  *sample = (loom_run_measurement_sample_t){
      .boundary = boundary,
      .kind_flags = active_kind_flags,
      .start_time_ns = iree_time_now(),
      .status_code = IREE_STATUS_OK,
  };
  *out_scope = (loom_run_measurement_scope_t){
      .result = result,
      .sample_index = sample_index,
      .boundary = boundary,
      .kind_flags = active_kind_flags,
      .start_time_ns = sample->start_time_ns,
      .is_recording = true,
  };
  return iree_ok_status();
}

iree_status_t loom_run_measurement_scope_end(
    loom_run_measurement_scope_t* scope,
    iree_status_code_t operation_status_code) {
  if (!scope->is_recording) {
    return iree_ok_status();
  }
  loom_run_measurement_sample_t* sample =
      &scope->result->samples[scope->sample_index];
  const iree_time_t end_time_ns = iree_time_now();
  sample->duration_ns = end_time_ns >= scope->start_time_ns
                            ? end_time_ns - scope->start_time_ns
                            : 0;
  sample->status_code = operation_status_code;
  *scope = (loom_run_measurement_scope_t){0};
  return iree_ok_status();
}

iree_status_t loom_run_measurement_run_step(
    const loom_run_measurement_options_t* options,
    loom_run_measurement_boundary_t boundary,
    loom_run_measurement_step_callback_t callback,
    loom_run_measurement_result_t* result) {
  loom_run_measurement_scope_t scope = {0};
  IREE_RETURN_IF_ERROR(
      loom_run_measurement_scope_begin(options, boundary, result, &scope));
  iree_status_t status = callback.fn(callback.user_data);
  iree_status_t scope_status =
      loom_run_measurement_scope_end(&scope, iree_status_code(status));
  return iree_status_join(status, scope_status);
}
