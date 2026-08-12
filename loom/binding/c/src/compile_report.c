// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compile_report.h"

#include <string.h>

#include "loom/util/stream.h"
#include "loomc/iree.h"
#include "loomc/status.h"

static bool loomc_compile_report_mode_is_valid(
    loomc_compile_report_mode_t mode) {
  switch (mode) {
    case LOOMC_COMPILE_REPORT_MODE_NONE:
    case LOOMC_COMPILE_REPORT_MODE_SUMMARY:
    case LOOMC_COMPILE_REPORT_MODE_DETAILS:
      return true;
    default:
      return false;
  }
}

loomc_status_t loomc_compile_report_options_validate(
    const loomc_compile_report_options_t* options) {
  if (options == NULL ||
      options->type != LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compile report options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compile report options structure_size is too small");
  }
  if (!loomc_compile_report_mode_is_valid(options->mode)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "compile report mode is invalid");
  }
  if (options->identifier.data == NULL && options->identifier.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "compile report identifier is malformed");
  }
  if (options->mode == LOOMC_COMPILE_REPORT_MODE_NONE &&
      !loomc_string_view_is_empty(options->identifier)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compile report identifier requires a non-NONE report mode");
  }
  return loomc_ok_status();
}

loom_target_compile_report_format_mode_t loomc_compile_report_format_mode(
    loomc_compile_report_mode_t mode) {
  switch (mode) {
    case LOOMC_COMPILE_REPORT_MODE_SUMMARY:
      return LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY;
    case LOOMC_COMPILE_REPORT_MODE_DETAILS:
      return LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS;
    case LOOMC_COMPILE_REPORT_MODE_NONE:
    default:
      return LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE;
  }
}

loom_target_compile_report_detail_flags_t
loomc_compile_report_requested_detail_flags(loomc_compile_report_mode_t mode) {
  if (mode != LOOMC_COMPILE_REPORT_MODE_DETAILS) {
    return LOOM_TARGET_COMPILE_REPORT_DETAIL_NONE;
  }
  return LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS |
         LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS |
         LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS |
         LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS |
         LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_INSERTION_ROWS;
}

loomc_status_t loomc_compile_report_make_identifier(
    const loomc_compile_report_options_t* options,
    loomc_string_view_t default_identifier, loomc_allocator_t allocator,
    loomc_string_view_t* out_identifier) {
  *out_identifier = loomc_string_view_empty();
  if (options != NULL && !loomc_string_view_is_empty(options->identifier)) {
    return loomc_string_view_clone(options->identifier, allocator,
                                   out_identifier);
  }
  const loomc_string_view_t suffix =
      loomc_make_cstring_view(".compile-report.json");
  if (default_identifier.size > LOOMC_HOST_SIZE_MAX - suffix.size) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "compile report identifier is too large");
  }
  const loomc_host_size_t identifier_length =
      default_identifier.size + suffix.size;
  char* identifier = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc_uninitialized(
      allocator, identifier_length, (void**)&identifier));
  memcpy(identifier, default_identifier.data, default_identifier.size);
  memcpy(identifier + default_identifier.size, suffix.data, suffix.size);
  *out_identifier = loomc_make_string_view(identifier, identifier_length);
  return loomc_ok_status();
}

loomc_status_t loomc_compile_report_add_artifact(
    loomc_result_t* result, loomc_compile_report_mode_t mode,
    loomc_string_view_t identifier,
    const loom_target_compile_report_t* report) {
  loomc_allocator_t allocator = loomc_result_allocator(result);
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_from_loomc(allocator),
                                 &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  const loom_target_compile_report_format_options_t format_options = {
      .mode = loomc_compile_report_format_mode(mode),
  };
  loomc_status_t status = loomc_status_from_iree(
      loom_target_compile_report_format_json(report, &format_options, &stream));

  char* report_storage = NULL;
  iree_host_size_t report_length = 0;
  if (loomc_status_is_ok(status)) {
    report_length = iree_string_builder_size(&builder);
    report_storage = iree_string_builder_take_storage(&builder);
    status = loomc_result_add_artifact_take_contents(
        result, LOOMC_ARTIFACT_KIND_REPORT,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON),
        identifier, loomc_make_byte_span(report_storage, report_length));
  }
  if (loomc_status_is_ok(status)) report_storage = NULL;
  loomc_allocator_free(allocator, report_storage);
  iree_string_builder_deinitialize(&builder);
  return status;
}

loomc_status_t loomc_compile_report_mode_parse(
    loomc_string_view_t value, loomc_compile_report_mode_t* out_mode) {
  if (out_mode == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_mode must not be NULL");
  }
  if (loomc_string_view_is_empty(value) ||
      loomc_string_view_equal(value, loomc_make_cstring_view("none"))) {
    *out_mode = LOOMC_COMPILE_REPORT_MODE_NONE;
    return loomc_ok_status();
  }
  if (loomc_string_view_equal(value, loomc_make_cstring_view("summary")) ||
      loomc_string_view_equal(value, loomc_make_cstring_view("json")) ||
      loomc_string_view_equal(value, loomc_make_cstring_view("json-summary"))) {
    *out_mode = LOOMC_COMPILE_REPORT_MODE_SUMMARY;
    return loomc_ok_status();
  }
  if (loomc_string_view_equal(value, loomc_make_cstring_view("details")) ||
      loomc_string_view_equal(value, loomc_make_cstring_view("json-details"))) {
    *out_mode = LOOMC_COMPILE_REPORT_MODE_DETAILS;
    return loomc_ok_status();
  }
  return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                           "unsupported compile report mode");
}

loomc_string_view_t loomc_compile_report_mode_name(
    loomc_compile_report_mode_t mode) {
  switch (mode) {
    case LOOMC_COMPILE_REPORT_MODE_SUMMARY:
      return loomc_make_cstring_view("summary");
    case LOOMC_COMPILE_REPORT_MODE_DETAILS:
      return loomc_make_cstring_view("details");
    case LOOMC_COMPILE_REPORT_MODE_NONE:
    default:
      return loomc_make_cstring_view("none");
  }
}
