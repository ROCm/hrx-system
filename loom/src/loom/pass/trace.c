// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/trace.h"

#include <inttypes.h>

#include "loom/util/json.h"

void loom_pass_trace_options_initialize(
    loom_pass_trace_options_t* out_options) {
  *out_options = (loom_pass_trace_options_t){
      .format = LOOM_PASS_TRACE_FORMAT_TEXT,
      .print_options = {.flags = LOOM_TEXT_PRINT_DEFAULT},
  };
}

iree_status_t loom_pass_trace_parse_format(
    iree_string_view_t value, loom_pass_trace_format_t* out_format) {
  value = iree_string_view_trim(value);
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("text"))) {
    *out_format = LOOM_PASS_TRACE_FORMAT_TEXT;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("jsonl"))) {
    *out_format = LOOM_PASS_TRACE_FORMAT_JSONL;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported --dump-ir-format mode '%.*s'",
                          (int)value.size, value.data);
}

bool loom_pass_trace_options_is_enabled(
    const loom_pass_trace_options_t* options) {
  return options && options->stream &&
         (options->dump_before_all || options->dump_after_all ||
          options->dump_before.count > 0 || options->dump_after.count > 0);
}

void loom_pass_trace_initialize(const loom_pass_trace_options_t* options,
                                loom_pass_trace_t* out_trace) {
  *out_trace = (loom_pass_trace_t){
      .options = options,
  };
}

static iree_string_view_t loom_pass_trace_kind_name(loom_pass_kind_t kind) {
  switch (kind) {
    case LOOM_PASS_MODULE:
      return IREE_SV("module");
    case LOOM_PASS_FUNCTION:
      return IREE_SV("func");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_pass_trace_point_name(
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

static iree_string_view_t loom_pass_trace_pass_key(
    const loom_pass_trace_event_t* event) {
  const loom_pass_program_instruction_t* instruction = event->instruction;
  if (!instruction ||
      instruction->kind != LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE ||
      !instruction->invoke.descriptor) {
    return IREE_SV("<unknown>");
  }
  return instruction->invoke.descriptor->key;
}

static iree_string_view_t loom_pass_trace_pass_kind(
    const loom_pass_trace_event_t* event) {
  const loom_pass_program_instruction_t* instruction = event->instruction;
  if (!instruction ||
      instruction->kind != LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE ||
      !instruction->invoke.info) {
    return IREE_SV("unknown");
  }
  return loom_pass_trace_kind_name(instruction->invoke.info->kind);
}

static bool loom_pass_trace_name_matches(
    iree_string_view_t request, const loom_pass_trace_options_t* options,
    const loom_pass_trace_event_t* event) {
  request = iree_string_view_trim(request);
  if (iree_string_view_is_empty(request)) {
    return false;
  }
  return iree_string_view_equal(request, loom_pass_trace_pass_key(event)) ||
         iree_string_view_equal(request, options->stage) ||
         iree_string_view_equal(request, event->pipeline_symbol);
}

static bool loom_pass_trace_list_matches(
    iree_string_view_list_t list, const loom_pass_trace_options_t* options,
    const loom_pass_trace_event_t* event) {
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    if (loom_pass_trace_name_matches(list.values[i], options, event)) {
      return true;
    }
  }
  return false;
}

static bool loom_pass_trace_event_matches(
    const loom_pass_trace_options_t* options,
    const loom_pass_trace_event_t* event) {
  switch (event->point) {
    case LOOM_PASS_TRACE_POINT_BEFORE:
      return options->dump_before_all ||
             loom_pass_trace_list_matches(options->dump_before, options, event);
    case LOOM_PASS_TRACE_POINT_AFTER:
      return options->dump_after_all ||
             loom_pass_trace_list_matches(options->dump_after, options, event);
    default:
      return false;
  }
}

static iree_string_view_t loom_pass_trace_or_unknown(iree_string_view_t value) {
  return iree_string_view_is_empty(value) ? IREE_SV("<unknown>") : value;
}

static bool loom_pass_trace_artifact_sink_is_enabled(
    const loom_pass_trace_artifact_sink_t* sink) {
  return sink->open != NULL || sink->close != NULL;
}

static iree_status_t loom_pass_trace_validate_artifact_sink(
    const loom_pass_trace_artifact_sink_t* sink) {
  if (!loom_pass_trace_artifact_sink_is_enabled(sink)) {
    return iree_ok_status();
  }
  if (sink->open == NULL || sink->close == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "pass trace artifact sink requires both open and close callbacks");
  }
  return iree_ok_status();
}

static iree_status_t loom_pass_trace_write_text_event(
    const loom_pass_trace_options_t* options, loom_output_stream_t* stream,
    const loom_pass_trace_event_t* event, iree_host_size_t event_ordinal,
    iree_string_view_t artifact_path, bool include_ir) {
  const iree_string_view_t point = loom_pass_trace_point_name(event->point);
  const iree_string_view_t pass_key = loom_pass_trace_pass_key(event);
  const iree_string_view_t pass_kind = loom_pass_trace_pass_kind(event);
  const iree_string_view_t anchor_kind =
      loom_pass_trace_kind_name(event->anchor_kind);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream,
      "//"
      "===---------------------------------------------------------------------"
      "-===//\n"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(
          stream,
          "// Loom IR dump %.*s pass %.*s\n"
          "// tool: %.*s\n"
          "// input: %.*s\n"
          "// stage: %.*s\n"
          "// pipeline: %.*s\n"
          "// pass_kind: %.*s\n"
          "// anchor: %.*s\n"
          "// symbol: %.*s\n"
          "// event_ordinal: %" PRIhsz "\n"
          "// invocation_ordinal: %" PRIhsz "\n"
          "// instruction_index: %" PRIhsz "\n"
          "// status: %s\n"
          "// changed: %s\n"
          "// diagnostics: errors=%" PRIu32 " warnings=%" PRIu32
          " remarks=%" PRIu32 "\n",
          (int)point.size, point.data, (int)pass_key.size, pass_key.data,
          (int)loom_pass_trace_or_unknown(options->tool_name).size,
          loom_pass_trace_or_unknown(options->tool_name).data,
          (int)loom_pass_trace_or_unknown(options->input_path).size,
          loom_pass_trace_or_unknown(options->input_path).data,
          (int)loom_pass_trace_or_unknown(options->stage).size,
          loom_pass_trace_or_unknown(options->stage).data,
          (int)loom_pass_trace_or_unknown(event->pipeline_symbol).size,
          loom_pass_trace_or_unknown(event->pipeline_symbol).data,
          (int)pass_kind.size, pass_kind.data, (int)anchor_kind.size,
          anchor_kind.data, (int)event->symbol_name.size,
          event->symbol_name.data, event_ordinal, event->invocation_ordinal,
          event->instruction_index, iree_status_code_string(event->status_code),
          event->changed ? "true" : "false", event->error_count,
          event->warning_count, event->remark_count));
  if (!iree_string_view_is_empty(artifact_path)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "// ir_path: %.*s\n", (int)artifact_path.size,
        artifact_path.data));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream,
      "//"
      "===-----------------------------------------------------------------"
      "-----===//\n"));
  if (!include_ir) {
    return loom_output_stream_write_cstring(stream, "\n");
  }
  IREE_RETURN_IF_ERROR(loom_text_print_module_with_options(
      event->module, stream, &options->print_options));
  return loom_output_stream_write_cstring(stream, "\n");
}

static iree_status_t loom_pass_trace_write_text(
    loom_pass_trace_t* trace, const loom_pass_trace_event_t* event,
    iree_host_size_t event_ordinal,
    const loom_pass_trace_artifact_t* artifact) {
  const loom_pass_trace_options_t* options = trace->options;
  if (artifact == NULL) {
    return loom_pass_trace_write_text_event(
        options, options->stream, event, event_ordinal,
        iree_string_view_empty(), /*include_ir=*/true);
  }
  IREE_RETURN_IF_ERROR(loom_pass_trace_write_text_event(
      options, options->stream, event, event_ordinal, artifact->path,
      /*include_ir=*/false));
  return loom_pass_trace_write_text_event(
      options, artifact->stream, event, event_ordinal, iree_string_view_empty(),
      /*include_ir=*/true);
}

static iree_status_t loom_pass_trace_write_json_string_field(
    loom_output_stream_t* stream, const char* name, iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(stream, "\"%s\":", name));
  return loom_json_write_escaped_string(stream, value);
}

static iree_status_t loom_pass_trace_write_jsonl(
    loom_pass_trace_t* trace, const loom_pass_trace_event_t* event,
    iree_host_size_t event_ordinal,
    const loom_pass_trace_artifact_t* artifact) {
  const loom_pass_trace_options_t* options = trace->options;
  loom_output_stream_t* stream = options->stream;
  const iree_string_view_t point = loom_pass_trace_point_name(event->point);
  const iree_string_view_t pass_key = loom_pass_trace_pass_key(event);
  const iree_string_view_t pass_kind = loom_pass_trace_pass_kind(event);
  const iree_string_view_t anchor_kind =
      loom_pass_trace_kind_name(event->anchor_kind);

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_pass_trace_write_json_string_field(
      stream, "tool", options->tool_name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  IREE_RETURN_IF_ERROR(loom_pass_trace_write_json_string_field(
      stream, "input", options->input_path));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  IREE_RETURN_IF_ERROR(
      loom_pass_trace_write_json_string_field(stream, "stage", options->stage));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  IREE_RETURN_IF_ERROR(loom_pass_trace_write_json_string_field(
      stream, "pipeline", event->pipeline_symbol));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  IREE_RETURN_IF_ERROR(
      loom_pass_trace_write_json_string_field(stream, "pass", pass_key));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  IREE_RETURN_IF_ERROR(
      loom_pass_trace_write_json_string_field(stream, "pass_kind", pass_kind));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  IREE_RETURN_IF_ERROR(
      loom_pass_trace_write_json_string_field(stream, "anchor", anchor_kind));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  IREE_RETURN_IF_ERROR(loom_pass_trace_write_json_string_field(
      stream, "symbol", event->symbol_name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  IREE_RETURN_IF_ERROR(
      loom_pass_trace_write_json_string_field(stream, "point", point));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"event_ordinal\":%" PRIhsz ",\"invocation_ordinal\":%" PRIhsz
      ",\"instruction_index\":%" PRIhsz ",\"status\":",
      event_ordinal, event->invocation_ordinal, event->instruction_index));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, iree_status_code_string(event->status_code)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"changed\":%s,\"diagnostics\":{\"errors\":%" PRIu32
      ",\"warnings\":%" PRIu32 ",\"remarks\":%" PRIu32 "}",
      event->changed ? "true" : "false", event->error_count,
      event->warning_count, event->remark_count));
  if (artifact != NULL) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    IREE_RETURN_IF_ERROR(loom_pass_trace_write_json_string_field(
        stream, "ir_path", artifact->path));
    return loom_output_stream_write_cstring(stream, "}\n");
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"ir\":\""));
  loom_json_escape_stream_t escape_data;
  loom_output_stream_t escape_stream;
  loom_json_escape_stream_init(stream, &escape_data, &escape_stream);
  IREE_RETURN_IF_ERROR(loom_text_print_module_with_options(
      event->module, &escape_stream, &options->print_options));
  return loom_output_stream_write_cstring(stream, "\"}\n");
}

iree_status_t loom_pass_trace_emit(loom_pass_trace_t* trace,
                                   const loom_pass_trace_event_t* event) {
  if (!trace || !loom_pass_trace_options_is_enabled(trace->options)) {
    return iree_ok_status();
  }
  if (!event || !event->module || !event->instruction ||
      event->instruction->kind != LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass trace requires an invoke event");
  }
  if (!loom_pass_trace_event_matches(trace->options, event)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_pass_trace_validate_artifact_sink(&trace->options->artifact_sink));

  const iree_host_size_t event_ordinal = trace->next_event_ordinal++;
  loom_pass_trace_artifact_t artifact = {0};
  const bool artifact_enabled =
      loom_pass_trace_artifact_sink_is_enabled(&trace->options->artifact_sink);
  if (artifact_enabled) {
    IREE_RETURN_IF_ERROR(trace->options->artifact_sink.open(
        trace->options->artifact_sink.user_data, event, event_ordinal,
        &artifact));
    if (!artifact.stream || iree_string_view_is_empty(artifact.path)) {
      iree_status_t close_status = trace->options->artifact_sink.close(
          trace->options->artifact_sink.user_data, &artifact);
      return iree_status_join(
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "pass trace artifact sink returned no stream/path"),
          close_status);
    }
  }

  iree_status_t status = iree_ok_status();
  switch (trace->options->format) {
    case LOOM_PASS_TRACE_FORMAT_TEXT:
      status = loom_pass_trace_write_text(trace, event, event_ordinal,
                                          artifact_enabled ? &artifact : NULL);
      break;
    case LOOM_PASS_TRACE_FORMAT_JSONL:
      if (artifact_enabled) {
        status = loom_text_print_module_with_options(
            event->module, artifact.stream, &trace->options->print_options);
        if (iree_status_is_ok(status)) {
          status = loom_output_stream_write_cstring(artifact.stream, "\n");
        }
      }
      if (iree_status_is_ok(status)) {
        status = loom_pass_trace_write_jsonl(
            trace, event, event_ordinal, artifact_enabled ? &artifact : NULL);
      }
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported pass trace format");
      break;
  }
  if (artifact_enabled) {
    status = iree_status_join(
        status, trace->options->artifact_sink.close(
                    trace->options->artifact_sink.user_data, &artifact));
  }
  return status;
}
