// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "pass_trace.h"

#include "loom/target/function_version_projection.h"
#include "loomc/iree.h"
#include "target.h"

static loomc_status_t loomc_pass_trace_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "pass trace string has length but no data");
  }
  return loomc_ok_status();
}

loomc_status_t loomc_pass_trace_options_validate(
    const loomc_pass_trace_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_PASS_TRACE_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "pass trace options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "pass trace options structure_size is too small");
  }
  switch (options->format) {
    case LOOMC_PASS_TRACE_FORMAT_TEXT:
    case LOOMC_PASS_TRACE_FORMAT_JSONL:
      break;
    default:
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "pass trace format is invalid");
  }
  const loomc_pass_trace_flags_t known_flags =
      LOOMC_PASS_TRACE_FLAG_BEFORE_ALL | LOOMC_PASS_TRACE_FLAG_AFTER_ALL;
  if ((options->flags & ~known_flags) != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "pass trace options contain unknown flags");
  }
  if (options->before_filter_count != 0 && options->before_filters == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "pass trace before_filter_count is nonzero but before_filters is "
        "NULL");
  }
  if (options->after_filter_count != 0 && options->after_filters == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "pass trace after_filter_count is nonzero but after_filters is NULL");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_pass_trace_validate_string_view(options->tool_name));
  LOOMC_RETURN_IF_ERROR(
      loomc_pass_trace_validate_string_view(options->input_identifier));
  for (loomc_host_size_t i = 0; i < options->before_filter_count; ++i) {
    LOOMC_RETURN_IF_ERROR(
        loomc_pass_trace_validate_string_view(options->before_filters[i]));
  }
  for (loomc_host_size_t i = 0; i < options->after_filter_count; ++i) {
    LOOMC_RETURN_IF_ERROR(
        loomc_pass_trace_validate_string_view(options->after_filters[i]));
  }
  const bool has_selection = options->flags != 0 ||
                             options->before_filter_count != 0 ||
                             options->after_filter_count != 0;
  if (!has_selection) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "pass trace options select no boundaries");
  }
  if (options->sink.write == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "pass trace options require a write callback");
  }
  const bool artifact_sink_enabled = options->artifact_sink.open != NULL ||
                                     options->artifact_sink.close != NULL;
  if (artifact_sink_enabled && (options->artifact_sink.open == NULL ||
                                options->artifact_sink.close == NULL)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "pass trace artifact sink requires open and close callbacks");
  }
  return loomc_ok_status();
}

static iree_status_t loomc_pass_trace_write(void* user_data,
                                            iree_string_view_t fragment) {
  loomc_pass_trace_state_t* state = (loomc_pass_trace_state_t*)user_data;
  loomc_status_t status =
      state->public_options->sink.write(state->public_options->sink.user_data,
                                        loomc_string_view_from_iree(fragment));
  if (!loomc_status_is_ok(status)) {
    state->callback_failed = true;
  }
  return iree_status_from_loomc(status);
}

static loomc_pass_trace_point_t loomc_pass_trace_point_from_internal(
    loom_pass_trace_point_t point) {
  switch (point) {
    case LOOM_PASS_TRACE_POINT_BEFORE:
      return LOOMC_PASS_TRACE_POINT_BEFORE;
    case LOOM_PASS_TRACE_POINT_AFTER:
      return LOOMC_PASS_TRACE_POINT_AFTER;
  }
  IREE_ASSERT_UNREACHABLE("invalid pass trace point");
  IREE_BUILTIN_UNREACHABLE();
}

static loomc_status_t loomc_pass_trace_artifact_validate(
    const loomc_pass_trace_artifact_t* artifact) {
  LOOMC_RETURN_IF_ERROR(
      loomc_pass_trace_validate_string_view(artifact->reference));
  if (loomc_string_view_is_empty(artifact->reference)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "pass trace artifact callback returned an empty reference");
  }
  if (artifact->sink.write == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "pass trace artifact callback returned no write callback");
  }
  return loomc_ok_status();
}

static iree_status_t loomc_pass_trace_artifact_write(
    void* user_data, iree_string_view_t fragment) {
  loomc_pass_trace_state_t* state = (loomc_pass_trace_state_t*)user_data;
  loomc_status_t status =
      state->public_artifact.sink.write(state->public_artifact.sink.user_data,
                                        loomc_string_view_from_iree(fragment));
  if (!loomc_status_is_ok(status)) {
    state->callback_failed = true;
  }
  return iree_status_from_loomc(status);
}

static iree_status_t loomc_pass_trace_artifact_open(
    void* user_data, const loom_pass_trace_event_t* event,
    iree_host_size_t event_ordinal, loom_pass_trace_artifact_t* out_artifact) {
  loomc_pass_trace_state_t* state = (loomc_pass_trace_state_t*)user_data;
  state->public_artifact = (loomc_pass_trace_artifact_t){0};
  const loomc_pass_trace_event_t public_event = {
      .event_ordinal = event_ordinal,
      .point = loomc_pass_trace_point_from_internal(event->point),
      .pass_key =
          loomc_string_view_from_iree(loom_pass_trace_event_pass_key(event)),
  };
  loomc_status_t status = state->public_options->artifact_sink.open(
      state->public_options->artifact_sink.user_data, &public_event,
      &state->public_artifact);
  if (!loomc_status_is_ok(status)) {
    state->callback_failed = true;
    return iree_status_from_loomc(status);
  }

  status = loomc_pass_trace_artifact_validate(&state->public_artifact);
  if (!loomc_status_is_ok(status)) {
    state->callback_failed = true;
    status = loomc_status_join(
        status, state->public_options->artifact_sink.close(
                    state->public_options->artifact_sink.user_data,
                    &state->public_artifact));
    state->public_artifact = (loomc_pass_trace_artifact_t){0};
    return iree_status_from_loomc(status);
  }

  state->artifact_stream = (loom_output_stream_t){
      .write = loomc_pass_trace_artifact_write,
      .user_data = state,
  };
  *out_artifact = (loom_pass_trace_artifact_t){
      .stream = &state->artifact_stream,
      .path = iree_string_view_from_loomc(state->public_artifact.reference),
  };
  return iree_ok_status();
}

static iree_status_t loomc_pass_trace_artifact_close(
    void* user_data, loom_pass_trace_artifact_t* artifact) {
  (void)artifact;
  loomc_pass_trace_state_t* state = (loomc_pass_trace_state_t*)user_data;
  loomc_status_t status = state->public_options->artifact_sink.close(
      state->public_options->artifact_sink.user_data, &state->public_artifact);
  if (!loomc_status_is_ok(status)) {
    state->callback_failed = true;
  }
  state->public_artifact = (loomc_pass_trace_artifact_t){0};
  state->artifact_stream = (loom_output_stream_t){0};
  return iree_status_from_loomc(status);
}

static iree_status_t loomc_pass_trace_project_snapshot(
    void* user_data, const loom_module_t* source_module,
    loom_module_t** out_projected_module) {
  *out_projected_module = NULL;
  const loomc_pass_trace_state_t* state =
      (const loomc_pass_trace_state_t*)user_data;
  if (state->function_versions->count == 0) {
    return iree_ok_status();
  }
  return loom_target_function_versions_project_module(
      source_module, state->function_versions, state->block_pool,
      source_module->host_allocator, /*out_versions=*/NULL,
      out_projected_module);
}

void loomc_pass_trace_state_initialize(
    const loomc_pass_trace_options_t* options, iree_string_view_t stage,
    const loomc_target_pass_environment_t* target_environment,
    const loom_function_version_list_t* function_versions,
    iree_arena_block_pool_t* block_pool, loomc_pass_trace_state_t* out_state) {
  *out_state = (loomc_pass_trace_state_t){
      .public_options = options,
      .stream =
          {
              .write = loomc_pass_trace_write,
              .user_data = out_state,
          },
      .function_versions = function_versions,
      .block_pool = block_pool,
  };
  loom_pass_trace_options_initialize(&out_state->options);
  out_state->options.stream = &out_state->stream;
  out_state->options.format = options->format == LOOMC_PASS_TRACE_FORMAT_JSONL
                                  ? LOOM_PASS_TRACE_FORMAT_JSONL
                                  : LOOM_PASS_TRACE_FORMAT_TEXT;
  out_state->options.tool_name =
      iree_string_view_from_loomc(options->tool_name);
  out_state->options.input_path =
      iree_string_view_from_loomc(options->input_identifier);
  out_state->options.stage = stage;
  out_state->options.dump_before = (iree_string_view_list_t){
      .values = (const iree_string_view_t*)options->before_filters,
      .count = options->before_filter_count,
  };
  out_state->options.dump_after = (iree_string_view_list_t){
      .values = (const iree_string_view_t*)options->after_filters,
      .count = options->after_filter_count,
  };
  out_state->options.dump_before_all =
      iree_any_bit_set(options->flags, LOOMC_PASS_TRACE_FLAG_BEFORE_ALL);
  out_state->options.dump_after_all =
      iree_any_bit_set(options->flags, LOOMC_PASS_TRACE_FLAG_AFTER_ALL);
  if (options->artifact_sink.open != NULL) {
    out_state->options.artifact_sink = (loom_pass_trace_artifact_sink_t){
        .open = loomc_pass_trace_artifact_open,
        .close = loomc_pass_trace_artifact_close,
        .user_data = out_state,
    };
  }
  loomc_target_pass_environment_initialize_text_asm_environment(
      target_environment,
      &out_state->options.print_options.low_asm_environment);
  loom_pass_trace_initialize(&out_state->options, &out_state->trace);
  loom_pass_trace_bind_snapshot_projector(
      &out_state->trace, (loom_pass_trace_snapshot_projector_t){
                             .project = loomc_pass_trace_project_snapshot,
                             .user_data = out_state,
                         });
}
