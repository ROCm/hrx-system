// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/pass/trace_cli.h"

#include <inttypes.h>

#include "iree/base/tooling/flags.h"

static bool FLAG_dump_ir_before_all = false;
static bool FLAG_dump_ir_after_all = false;
static const char* FLAG_dump_ir_format = "text";
static const char* FLAG_dump_ir_output = "stderr";

#if IREE_FLAGS_ENABLE_CLI == 1
static iree_flag_string_list_storage_t FLAG_dump_ir_before_storage = {
    .capacity = 1,
    .count = 0,
};
static iree_flag_string_list_storage_t FLAG_dump_ir_after_storage = {
    .capacity = 1,
    .count = 0,
};

IREE_STATIC_INITIALIZER(loom_tooling_pass_trace_register_flags) {
  iree_flag_register(
      __FILE__, __LINE__, IREE_FLAG_TYPE_callback, &FLAG_dump_ir_before_storage,
      iree_flag_string_list_parse, iree_flag_string_list_print,
      IREE_SV("dump-ir-before"),
      IREE_SV("Pass key, pass.pipeline symbol, or stage to dump before. "
              "Repeat for multiple filters."));
  iree_flag_register(
      __FILE__, __LINE__, IREE_FLAG_TYPE_callback, &FLAG_dump_ir_after_storage,
      iree_flag_string_list_parse, iree_flag_string_list_print,
      IREE_SV("dump-ir-after"),
      IREE_SV("Pass key, pass.pipeline symbol, or stage to dump after. "
              "Repeat for multiple filters."));
  iree_flag_register(__FILE__, __LINE__, IREE_FLAG_TYPE_bool,
                     &FLAG_dump_ir_before_all, NULL, NULL,
                     IREE_SV("dump-ir-before-all"),
                     IREE_SV("Dump IR before every pass invocation."));
  iree_flag_register(__FILE__, __LINE__, IREE_FLAG_TYPE_bool,
                     &FLAG_dump_ir_after_all, NULL, NULL,
                     IREE_SV("dump-ir-after-all"),
                     IREE_SV("Dump IR after every pass invocation."));
  iree_flag_register(__FILE__, __LINE__, IREE_FLAG_TYPE_string,
                     &FLAG_dump_ir_format, NULL, NULL,
                     IREE_SV("dump-ir-format"),
                     IREE_SV("IR dump format. Use 'text' or 'jsonl'."));
  iree_flag_register(
      __FILE__, __LINE__, IREE_FLAG_TYPE_string, &FLAG_dump_ir_output, NULL,
      NULL, IREE_SV("dump-ir-output"),
      IREE_SV("IR dump output path. Use 'stderr', 'stdout'/'-', a file path, "
              "or a directory path for an index plus per-event IR files."));
}

static iree_flag_string_list_t loom_tooling_pass_trace_dump_ir_before_list(
    void) {
  return (iree_flag_string_list_t){
      .count = FLAG_dump_ir_before_storage.count,
      .values = FLAG_dump_ir_before_storage.count == 1
                    ? &FLAG_dump_ir_before_storage.inline_value
                    : FLAG_dump_ir_before_storage.values,
  };
}

static iree_flag_string_list_t loom_tooling_pass_trace_dump_ir_after_list(
    void) {
  return (iree_flag_string_list_t){
      .count = FLAG_dump_ir_after_storage.count,
      .values = FLAG_dump_ir_after_storage.count == 1
                    ? &FLAG_dump_ir_after_storage.inline_value
                    : FLAG_dump_ir_after_storage.values,
  };
}
#else
static iree_flag_string_list_t loom_tooling_pass_trace_dump_ir_before_list(
    void) {
  return iree_string_view_list_empty();
}

static iree_flag_string_list_t loom_tooling_pass_trace_dump_ir_after_list(
    void) {
  return iree_string_view_list_empty();
}
#endif  // IREE_FLAGS_ENABLE_CLI == 1

bool loom_tooling_pass_trace_flags_requested(void) {
  return FLAG_dump_ir_before_all || FLAG_dump_ir_after_all ||
         loom_tooling_pass_trace_dump_ir_before_list().count > 0 ||
         loom_tooling_pass_trace_dump_ir_after_list().count > 0;
}

static bool loom_tooling_pass_trace_path_is_stdout(iree_string_view_t path) {
  return loom_tooling_file_path_is_stdio(path) ||
         iree_string_view_equal(path, IREE_SV("stdout"));
}

static bool loom_tooling_pass_trace_path_is_stream(iree_string_view_t path) {
  return loom_tooling_pass_trace_path_is_stdout(path) ||
         iree_string_view_equal(path, IREE_SV("stderr"));
}

static iree_string_view_t loom_tooling_pass_trace_point_name(
    loom_pass_trace_point_t point) {
  switch (point) {
    case LOOM_PASS_TRACE_POINT_BEFORE:
      return IREE_SV("before");
    case LOOM_PASS_TRACE_POINT_AFTER:
      return IREE_SV("after");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_tooling_pass_trace_pass_key(
    const loom_pass_trace_event_t* event) {
  const loom_pass_program_instruction_t* instruction = event->instruction;
  if (!instruction ||
      instruction->kind != LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE ||
      !instruction->invoke.descriptor) {
    return IREE_SV("unknown");
  }
  return instruction->invoke.descriptor->key;
}

static bool loom_tooling_pass_trace_safe_path_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
}

static iree_status_t loom_tooling_pass_trace_append_sanitized_path_fragment(
    iree_string_builder_t* builder, iree_string_view_t value) {
  value = iree_string_view_trim(value);
  bool appended = false;
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    const char c = value.data[i];
    if (loom_tooling_pass_trace_safe_path_char(c)) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          builder, iree_make_string_view(&value.data[i], 1)));
      appended = true;
    } else if (appended) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "-"));
    }
  }
  if (!appended) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, "unknown"));
  }
  return iree_ok_status();
}

static iree_status_t loom_tooling_pass_trace_build_artifact_relative_path(
    const loom_pass_trace_event_t* event, iree_host_size_t event_ordinal,
    iree_allocator_t allocator, char** out_path) {
  *out_path = NULL;
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator, &builder);
  const iree_string_view_t point =
      loom_tooling_pass_trace_point_name(event->point);
  iree_status_t status = iree_string_builder_append_format(
      &builder, "ir/%06" PRIhsz "-%.*s-", event_ordinal, (int)point.size,
      point.data);
  if (iree_status_is_ok(status)) {
    status = loom_tooling_pass_trace_append_sanitized_path_fragment(
        &builder, loom_tooling_pass_trace_pass_key(event));
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&builder, ".loom");
  }
  if (iree_status_is_ok(status)) {
    *out_path = iree_string_builder_take_storage(&builder);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_tooling_pass_trace_open_bundle_artifact(
    void* user_data, const loom_pass_trace_event_t* event,
    iree_host_size_t event_ordinal, loom_pass_trace_artifact_t* out_artifact) {
  loom_tooling_pass_trace_t* trace = (loom_tooling_pass_trace_t*)user_data;
  *out_artifact = (loom_pass_trace_artifact_t){0};
  if (trace->bundle_artifact_path || trace->bundle_artifact_relative_path) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "pass trace bundle artifact is already open");
  }
  IREE_RETURN_IF_ERROR(loom_tooling_pass_trace_build_artifact_relative_path(
      event, event_ordinal, trace->host_allocator,
      &trace->bundle_artifact_relative_path));
  iree_status_t status = loom_tooling_file_path_join(
      iree_make_cstring_view(trace->bundle_directory),
      iree_make_cstring_view(trace->bundle_artifact_relative_path),
      trace->host_allocator, &trace->bundle_artifact_path);
  if (iree_status_is_ok(status)) {
    status = loom_tooling_output_stream_open(
        iree_make_cstring_view(trace->bundle_artifact_path),
        trace->host_allocator, &trace->bundle_artifact_output);
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(trace->host_allocator,
                        trace->bundle_artifact_relative_path);
    iree_allocator_free(trace->host_allocator, trace->bundle_artifact_path);
    trace->bundle_artifact_relative_path = NULL;
    trace->bundle_artifact_path = NULL;
    return status;
  }
  *out_artifact = (loom_pass_trace_artifact_t){
      .stream = &trace->bundle_artifact_output.stream,
      .path = iree_make_cstring_view(trace->bundle_artifact_relative_path),
  };
  return iree_ok_status();
}

static iree_status_t loom_tooling_pass_trace_close_bundle_artifact(
    void* user_data, loom_pass_trace_artifact_t* artifact) {
  (void)artifact;
  loom_tooling_pass_trace_t* trace = (loom_tooling_pass_trace_t*)user_data;
  iree_status_t status =
      loom_tooling_output_stream_close(&trace->bundle_artifact_output);
  iree_allocator_free(trace->host_allocator,
                      trace->bundle_artifact_relative_path);
  iree_allocator_free(trace->host_allocator, trace->bundle_artifact_path);
  trace->bundle_artifact_relative_path = NULL;
  trace->bundle_artifact_path = NULL;
  return status;
}

static void loom_tooling_pass_trace_release_bundle_paths(
    loom_tooling_pass_trace_t* trace) {
  iree_allocator_free(trace->host_allocator, trace->bundle_index_path);
  iree_allocator_free(trace->host_allocator, trace->bundle_ir_directory);
  iree_allocator_free(trace->host_allocator, trace->bundle_directory);
  trace->bundle_index_path = NULL;
  trace->bundle_ir_directory = NULL;
  trace->bundle_directory = NULL;
}

static iree_status_t loom_tooling_pass_trace_open_bundle(
    loom_tooling_pass_trace_t* trace, iree_string_view_t output_path,
    iree_allocator_t allocator) {
  trace->host_allocator = allocator;
  iree_status_t status =
      loom_tooling_create_directory_if_needed(output_path, allocator);
  if (iree_status_is_ok(status)) {
    status = loom_tooling_file_path_join(output_path, IREE_SV("ir"), allocator,
                                         &trace->bundle_ir_directory);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_create_directory_if_needed(
        iree_make_cstring_view(trace->bundle_ir_directory), allocator);
  }

  const iree_string_view_t index_leaf =
      trace->pass_options.format == LOOM_PASS_TRACE_FORMAT_JSONL
          ? IREE_SV("trace.jsonl")
          : IREE_SV("trace.txt");
  if (iree_status_is_ok(status)) {
    status = loom_tooling_file_path_join(output_path, index_leaf, allocator,
                                         &trace->bundle_index_path);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_file_path_join(output_path, iree_string_view_empty(),
                                         allocator, &trace->bundle_directory);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_output_stream_open(
        iree_make_cstring_view(trace->bundle_index_path), allocator,
        &trace->output);
  }
  if (iree_status_is_ok(status)) {
    trace->pass_options.artifact_sink = (loom_pass_trace_artifact_sink_t){
        .open = loom_tooling_pass_trace_open_bundle_artifact,
        .close = loom_tooling_pass_trace_close_bundle_artifact,
        .user_data = trace,
    };
  } else {
    loom_tooling_pass_trace_release_bundle_paths(trace);
  }
  return status;
}

static iree_status_t loom_tooling_pass_trace_check_stdout_conflicts(
    const loom_tooling_pass_trace_open_options_t* options,
    iree_string_view_t dump_output_path) {
  if (!loom_tooling_pass_trace_path_is_stdout(dump_output_path)) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < options->stdout_conflict_count; ++i) {
    const loom_tooling_pass_trace_stdout_conflict_t* conflict =
        &options->stdout_conflicts[i];
    if (!conflict->active ||
        !loom_tooling_pass_trace_path_is_stdout(conflict->path)) {
      continue;
    }
    iree_string_view_t path = conflict->path;
    if (iree_string_view_is_empty(path)) {
      path = IREE_SV("-");
    }
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--dump-ir-output=stdout cannot share stdout with %.*s=%.*s",
        (int)conflict->flag_name.size, conflict->flag_name.data, (int)path.size,
        path.data);
  }
  return iree_ok_status();
}

iree_status_t loom_tooling_pass_trace_open_from_flags(
    const loom_tooling_pass_trace_open_options_t* options,
    iree_allocator_t allocator, loom_tooling_pass_trace_t* out_trace) {
  *out_trace = (loom_tooling_pass_trace_t){0};
  if (!loom_tooling_pass_trace_flags_requested()) {
    return iree_ok_status();
  }

  loom_pass_trace_options_initialize(&out_trace->pass_options);
  out_trace->pass_options.dump_before =
      loom_tooling_pass_trace_dump_ir_before_list();
  out_trace->pass_options.dump_after =
      loom_tooling_pass_trace_dump_ir_after_list();
  out_trace->pass_options.dump_before_all = FLAG_dump_ir_before_all;
  out_trace->pass_options.dump_after_all = FLAG_dump_ir_after_all;
  out_trace->pass_options.tool_name = options->tool_name;
  out_trace->pass_options.input_path = options->input_path;
  out_trace->pass_options.print_options.low_asm_descriptor_set_key =
      options->low_asm_descriptor_set_key;

  IREE_RETURN_IF_ERROR(
      loom_pass_trace_parse_format(iree_make_cstring_view(FLAG_dump_ir_format),
                                   &out_trace->pass_options.format));
  const iree_string_view_t output_path =
      iree_make_cstring_view(FLAG_dump_ir_output);
  IREE_RETURN_IF_ERROR(
      loom_tooling_pass_trace_check_stdout_conflicts(options, output_path));
  bool output_is_directory = false;
  IREE_RETURN_IF_ERROR(loom_tooling_file_path_is_directory(
      output_path, allocator, &output_is_directory));
  if (!loom_tooling_pass_trace_path_is_stream(output_path) &&
      (output_is_directory ||
       loom_tooling_file_path_has_trailing_separator(output_path))) {
    IREE_RETURN_IF_ERROR(
        loom_tooling_pass_trace_open_bundle(out_trace, output_path, allocator));
  } else {
    IREE_RETURN_IF_ERROR(loom_tooling_output_stream_open(output_path, allocator,
                                                         &out_trace->output));
  }
  out_trace->pass_options.stream = &out_trace->output.stream;
  out_trace->enabled = true;
  return iree_ok_status();
}

iree_status_t loom_tooling_pass_trace_close(loom_tooling_pass_trace_t* trace) {
  if (!trace || !trace->enabled) {
    return iree_ok_status();
  }
  trace->enabled = false;
  iree_status_t status = loom_tooling_output_stream_close(&trace->output);
  loom_tooling_pass_trace_release_bundle_paths(trace);
  return status;
}

const loom_pass_trace_options_t* loom_tooling_pass_trace_options(
    const loom_tooling_pass_trace_t* trace) {
  return trace && trace->enabled ? &trace->pass_options : NULL;
}
