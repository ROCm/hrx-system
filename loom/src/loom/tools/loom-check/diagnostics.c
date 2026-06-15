// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/diagnostics.h"

#include <inttypes.h>
#include <string.h>

#include "loom/error/renderer.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

//===----------------------------------------------------------------------===//
// Diagnostic collection
//===----------------------------------------------------------------------===//

static iree_status_t loom_check_diagnostic_collector_grow(
    loom_check_diagnostic_collector_t* collector) {
  iree_host_size_t new_capacity =
      collector->capacity == 0 ? 16 : collector->capacity * 2;
  loom_check_collected_diagnostic_t* new_diagnostics = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      collector->arena,
      new_capacity * sizeof(loom_check_collected_diagnostic_t),
      (void**)&new_diagnostics));
  if (collector->count > 0) {
    memcpy(new_diagnostics, collector->diagnostics,
           collector->count * sizeof(loom_check_collected_diagnostic_t));
  }
  collector->diagnostics = new_diagnostics;
  collector->capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_check_format_type(loom_type_t type, void* user_data,
                                            loom_output_stream_t* stream) {
  const loom_check_diagnostic_collector_t* collector =
      (const loom_check_diagnostic_collector_t*)user_data;
  if (collector->type_print_context.options.low_asm_environment.vtable) {
    return loom_text_print_type_with_options(
        type, collector->module, stream,
        &collector->type_print_context.options);
  }
  if (!collector->module) {
    return loom_type_format_minimal(type, NULL, stream);
  }
  return loom_text_print_type(type, collector->module, stream);
}

static iree_status_t loom_check_diagnostic_collector_copy_string(
    loom_check_diagnostic_collector_t* collector, iree_string_view_t source,
    iree_string_view_t* out_copy) {
  *out_copy = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) return iree_ok_status();
  char* target = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(collector->arena, source.size, (void**)&target));
  memcpy(target, source.data, source.size);
  *out_copy = iree_make_string_view(target, source.size);
  return iree_ok_status();
}

static iree_status_t loom_check_render_string_list_param(
    loom_diagnostic_string_list_t string_list, loom_output_stream_t* stream) {
  if (string_list.count > 0 && !string_list.values) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "string list param has count > 0 but values NULL");
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
  for (iree_host_size_t i = 0; i < string_list.count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    }
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write(stream, string_list.values[i]));
  }
  return loom_output_stream_write_char(stream, ']');
}

static iree_status_t loom_check_render_param_value(
    const loom_diagnostic_param_t* param, loom_type_formatter_t type_formatter,
    loom_output_stream_t* stream) {
  switch (param->kind) {
    case LOOM_PARAM_STRING:
      return loom_output_stream_write(stream, param->string);
    case LOOM_PARAM_I64:
      return loom_output_stream_write_format(stream, "%" PRId64, param->i64);
    case LOOM_PARAM_U32:
      return loom_output_stream_write_format(stream, "%" PRIu32, param->u32);
    case LOOM_PARAM_U64:
      return loom_output_stream_write_format(stream, "%" PRIu64, param->u64);
    case LOOM_PARAM_STRING_LIST:
      return loom_check_render_string_list_param(param->string_list, stream);
    case LOOM_PARAM_BOOL:
      return loom_output_stream_write_cstring(
          stream, param->boolean ? "true" : "false");
    case LOOM_PARAM_TYPE:
      if (type_formatter.fn) {
        return type_formatter.fn(param->type, type_formatter.user_data, stream);
      }
      return loom_output_stream_write_cstring(stream, "<type>");
    default:
      return loom_output_stream_write_cstring(stream, "<?>");
  }
}

static iree_status_t loom_check_diagnostic_collector_copy_param_values(
    loom_check_diagnostic_collector_t* collector,
    const loom_diagnostic_t* diagnostic, loom_type_formatter_t type_formatter,
    iree_string_view_t** out_values, iree_host_size_t* out_count) {
  *out_values = NULL;
  *out_count = 0;
  if (!diagnostic->params || !diagnostic->error ||
      !diagnostic->error->param_defs || diagnostic->param_count == 0) {
    return iree_ok_status();
  }

  const iree_host_size_t param_count =
      iree_min(diagnostic->param_count, diagnostic->error->param_count);
  if (param_count == 0) {
    return iree_ok_status();
  }
  iree_string_view_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      collector->arena, param_count, sizeof(*values), (void**)&values));

  for (iree_host_size_t i = 0; i < param_count; ++i) {
    iree_string_builder_t value_builder;
    iree_string_builder_initialize(collector->host_allocator, &value_builder);
    loom_output_stream_t value_stream;
    loom_output_stream_for_builder(&value_builder, &value_stream);
    iree_status_t status = loom_check_render_param_value(
        &diagnostic->params[i], type_formatter, &value_stream);
    if (iree_status_is_ok(status)) {
      status = loom_check_diagnostic_collector_copy_string(
          collector, iree_string_builder_view(&value_builder), &values[i]);
    }
    iree_string_builder_deinitialize(&value_builder);
    IREE_RETURN_IF_ERROR(status);
  }

  *out_values = values;
  *out_count = param_count;
  return iree_ok_status();
}

iree_status_t loom_check_diagnostic_collector_sink(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loom_check_diagnostic_collector_t* collector =
      (loom_check_diagnostic_collector_t*)user_data;
  if (collector->count >= collector->capacity) {
    IREE_RETURN_IF_ERROR(loom_check_diagnostic_collector_grow(collector));
  }

  iree_string_builder_t message_builder;
  iree_string_builder_initialize(collector->host_allocator, &message_builder);
  loom_output_stream_t message_stream;
  loom_output_stream_for_builder(&message_builder, &message_stream);
  loom_type_formatter_t type_formatter = {loom_check_format_type, collector};
  iree_status_t render_status = loom_diagnostic_render_message(
      diagnostic->error, diagnostic->params, diagnostic->param_count,
      type_formatter, &message_stream);
  if (!iree_status_is_ok(render_status)) {
    iree_string_builder_deinitialize(&message_builder);
    return render_status;
  }

  iree_string_view_t message = iree_string_view_empty();
  iree_status_t copy_status = loom_check_diagnostic_collector_copy_string(
      collector, iree_string_builder_view(&message_builder), &message);
  iree_string_builder_deinitialize(&message_builder);
  IREE_RETURN_IF_ERROR(copy_status);

  iree_string_view_t* param_values = NULL;
  iree_host_size_t param_value_count = 0;
  IREE_RETURN_IF_ERROR(loom_check_diagnostic_collector_copy_param_values(
      collector, diagnostic, type_formatter, &param_values,
      &param_value_count));

  iree_string_builder_t formatted_builder;
  iree_string_builder_initialize(collector->host_allocator, &formatted_builder);
  loom_output_stream_t formatted_stream;
  loom_output_stream_for_builder(&formatted_builder, &formatted_stream);
  const loom_diagnostic_format_options_t format_options = {
      .type_formatter = type_formatter,
  };
  iree_status_t format_status = loom_diagnostic_format_with_options(
      diagnostic, &format_options, &formatted_stream);
  iree_string_view_t formatted_diagnostic = iree_string_view_empty();
  if (iree_status_is_ok(format_status)) {
    format_status = loom_check_diagnostic_collector_copy_string(
        collector, iree_string_builder_view(&formatted_builder),
        &formatted_diagnostic);
  }
  iree_string_builder_deinitialize(&formatted_builder);
  IREE_RETURN_IF_ERROR(format_status);

  collector->diagnostics[collector->count++] =
      (loom_check_collected_diagnostic_t){
          .severity = diagnostic->severity,
          .domain = diagnostic->error->domain,
          .code = diagnostic->error->code,
          .error = diagnostic->error,
          .origin_line = diagnostic->origin.start_line,
          .message = message,
          .param_values = param_values,
          .param_value_count = param_value_count,
          .formatted_diagnostic = formatted_diagnostic,
      };
  loom_check_diagnostic_capture_t diagnostic_capture = {
      .type_formatter = type_formatter,
      .result = collector->result,
  };
  IREE_RETURN_IF_ERROR(
      loom_check_diagnostic_capture_sink(&diagnostic_capture, diagnostic));
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Diagnostic emission materialization
//===----------------------------------------------------------------------===//

iree_status_t loom_check_source_resolver_for_case(
    loom_module_t* module, iree_string_view_t filename,
    iree_string_view_t source, loom_source_entry_t* out_source_entry,
    loom_source_table_resolver_t* out_source_resolver) {
  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_register_source(module, filename, &source_id));
  *out_source_entry = (loom_source_entry_t){
      .source_id = source_id,
      .source = source,
      .filename = filename,
  };
  *out_source_resolver = (loom_source_table_resolver_t){
      .entries = out_source_entry,
      .count = 1,
  };
  return iree_ok_status();
}

static bool loom_check_diagnostic_resolve_location(
    const loom_check_diagnostic_emitter_capture_t* capture, const loom_op_t* op,
    loom_source_range_t* out_source_location) {
  if (!capture || !capture->module || !op) return false;
  if (!loom_source_resolve(capture->source_resolver, capture->module,
                           op->location, out_source_location)) {
    return false;
  }
  if (out_source_location->provenance ==
          LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE &&
      out_source_location->source.size > 0) {
    out_source_location->provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE;
  }
  return true;
}

static iree_host_size_t loom_check_diagnostic_collect_related_locations(
    const loom_check_diagnostic_emitter_capture_t* capture,
    const loom_diagnostic_related_op_t* related_ops,
    iree_host_size_t related_op_count,
    loom_diagnostic_related_location_t* out_related_locations,
    iree_host_size_t* out_omitted_count) {
  *out_omitted_count = 0;
  if (!related_ops || related_op_count == 0) return 0;
  iree_host_size_t related_location_count = 0;
  for (iree_host_size_t i = 0; i < related_op_count; ++i) {
    loom_source_range_t source_location = {
        .provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE,
    };
    if (!loom_check_diagnostic_resolve_location(capture, related_ops[i].op,
                                                &source_location)) {
      continue;
    }
    if (related_location_count >= LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS) {
      ++*out_omitted_count;
      continue;
    }
    out_related_locations[related_location_count++] =
        (loom_diagnostic_related_location_t){
            .label = related_ops[i].label,
            .source_location = source_location,
        };
  }
  return related_location_count;
}

iree_status_t loom_check_diagnostic_emitter_capture_emit(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loom_check_diagnostic_emitter_capture_t* capture =
      (loom_check_diagnostic_emitter_capture_t*)user_data;
  if (!capture || !capture->diagnostic_collector || !emission ||
      !emission->error) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "diagnostic emitter capture requires an emission");
  }

  loom_diagnostic_t diagnostic = {
      .severity = emission->error->severity,
      .error = emission->error,
      .params = emission->params,
      .param_count = emission->param_count,
      .emitter = capture->emitter,
      .origin = {.provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE},
      .source_location = {.provenance =
                              LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE},
  };

  loom_diagnostic_related_location_t
      related_locations[LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS];
  diagnostic.related_location_count =
      loom_check_diagnostic_collect_related_locations(
          capture, emission->related_ops, emission->related_op_count,
          related_locations, &diagnostic.related_location_omitted_count);
  if (diagnostic.related_location_count > 0) {
    diagnostic.related_locations = related_locations;
  }

  if (loom_check_diagnostic_resolve_location(capture, emission->op,
                                             &diagnostic.source_location)) {
    diagnostic.origin = diagnostic.source_location;
  }

  IREE_RETURN_IF_ERROR(loom_check_diagnostic_collector_sink(
      capture->diagnostic_collector, &diagnostic));
  ++capture->emission_count;
  return iree_ok_status();
}

static loom_source_range_t loom_check_case_source_range(
    iree_string_view_t filename, iree_string_view_t source) {
  iree_host_size_t line_start = 0;
  uint32_t line = 1;
  while (line_start < source.size) {
    iree_string_view_t remaining =
        iree_string_view_substr(source, line_start, IREE_HOST_SIZE_MAX);
    const iree_host_size_t newline_position =
        iree_string_view_find_char(remaining, '\n', 0);
    const iree_host_size_t line_length =
        newline_position == IREE_STRING_VIEW_NPOS ? remaining.size
                                                  : newline_position;
    iree_string_view_t line_text =
        iree_string_view_substr(source, line_start, line_length);
    iree_string_view_t trimmed = iree_string_view_trim(line_text);
    if (!iree_string_view_is_empty(trimmed) &&
        !iree_string_view_starts_with(trimmed, IREE_SV("//"))) {
      const iree_host_size_t start =
          line_start + (iree_host_size_t)(trimmed.data - line_text.data);
      const iree_host_size_t end = start + trimmed.size;
      const uint32_t start_column = (uint32_t)(start - line_start) + 1;
      const uint32_t end_column = start_column + (uint32_t)trimmed.size;
      return (loom_source_range_t){
          .provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE,
          .filename = filename,
          .source = source,
          .start = start,
          .end = end,
          .start_line = line,
          .start_column = start_column,
          .end_line = line,
          .end_column = end_column,
      };
    }
    if (newline_position == IREE_STRING_VIEW_NPOS) {
      break;
    }
    line_start += newline_position + 1;
    ++line;
  }
  if (source.size == 0) {
    return (loom_source_range_t){
        .provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE,
    };
  }
  return (loom_source_range_t){
      .provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE,
      .filename = filename,
      .source = source,
      .start = 0,
      .end = 0,
      .start_line = 1,
      .start_column = 1,
      .end_line = 1,
      .end_column = 1,
  };
}

iree_status_t loom_check_diagnostic_collector_emit_case_source(
    loom_check_diagnostic_collector_t* collector,
    const loom_check_case_t* test_case, iree_string_view_t filename,
    loom_emitter_t emitter, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count) {
  if (!collector) return iree_ok_status();
  if (!test_case || !error) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "case-source diagnostic emission is malformed");
  }
  const loom_source_range_t source_range =
      loom_check_case_source_range(filename, test_case->input);
  const loom_diagnostic_t diagnostic = {
      .severity = error->severity,
      .error = error,
      .params = params,
      .param_count = param_count,
      .emitter = emitter,
      .origin = source_range,
      .source_location = source_range,
  };
  return loom_check_diagnostic_collector_sink(collector, &diagnostic);
}

//===----------------------------------------------------------------------===//
// Annotation matching
//===----------------------------------------------------------------------===//

static int loom_check_diagnostic_find_param_index(
    const loom_check_collected_diagnostic_t* diagnostic,
    iree_string_view_t name) {
  if (!diagnostic->error || !diagnostic->error->param_defs) {
    return -1;
  }
  for (iree_host_size_t i = 0; i < diagnostic->param_value_count; ++i) {
    if (i >= diagnostic->error->param_count) {
      return -1;
    }
    if (iree_string_view_equal(
            name,
            iree_make_cstring_view(diagnostic->error->param_defs[i].name))) {
      return (int)i;
    }
  }
  return -1;
}

static bool loom_check_diagnostic_params_match_annotation(
    const loom_check_collected_diagnostic_t* diagnostic,
    const loom_check_annotation_t* annotation) {
  for (uint8_t i = 0; i < annotation->param_match_count; ++i) {
    const loom_check_annotation_param_match_t* match =
        &annotation->param_matches[i];
    const int param_index =
        loom_check_diagnostic_find_param_index(diagnostic, match->name);
    if (param_index < 0) {
      return false;
    }
    if (!iree_string_view_equal(diagnostic->param_values[param_index],
                                match->value)) {
      return false;
    }
  }
  return true;
}

static bool loom_check_diagnostic_matches_annotation(
    const loom_check_collected_diagnostic_t* diagnostic,
    const loom_check_annotation_t* annotation) {
  if (diagnostic->severity != annotation->severity) return false;

  if (annotation->domain != LOOM_ERROR_DOMAIN_COUNT_ &&
      diagnostic->domain != annotation->domain) {
    return false;
  }

  if (annotation->code != 0 && diagnostic->code != annotation->code) {
    return false;
  }

  if (diagnostic->origin_line != (uint32_t)annotation->target_line) {
    return false;
  }

  for (uint8_t i = 0; i < annotation->message_substring_count; ++i) {
    if (iree_string_view_find(diagnostic->message,
                              annotation->message_substrings[i],
                              0) == IREE_STRING_VIEW_NPOS) {
      return false;
    }
  }

  if (!loom_check_diagnostic_params_match_annotation(diagnostic, annotation)) {
    return false;
  }

  return true;
}

static iree_status_t loom_check_match_annotations(
    loom_check_collected_diagnostic_t* diagnostics,
    iree_host_size_t diagnostic_count,
    const loom_check_annotation_t* annotations,
    iree_host_size_t annotation_count, loom_check_file_report_t* report,
    iree_host_size_t case_index) {
  for (iree_host_size_t a = 0; a < annotation_count; ++a) {
    for (iree_host_size_t d = 0; d < diagnostic_count; ++d) {
      if (diagnostics[d].matched) continue;
      if (loom_check_diagnostic_matches_annotation(&diagnostics[d],
                                                   &annotations[a])) {
        diagnostics[d].matched = true;
        IREE_RETURN_IF_ERROR(loom_check_file_report_mark_annotation_matched(
            report, case_index, a));
        break;
      }
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Result assembly
//===----------------------------------------------------------------------===//

static const char* loom_check_annotation_severity_name(
    loom_diagnostic_severity_t severity) {
  switch (severity) {
    case LOOM_DIAGNOSTIC_ERROR:
      return "ERROR";
    case LOOM_DIAGNOSTIC_WARNING:
      return "WARNING";
    case LOOM_DIAGNOSTIC_REMARK:
      return "REMARK";
    case LOOM_DIAGNOSTIC_COUNT_:
      break;
  }
  return "ERROR";
}

static iree_status_t loom_check_append_diagnostic_annotation_line(
    const loom_check_collected_diagnostic_t* diagnostic,
    iree_host_size_t line_offset, iree_string_view_t indentation,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, indentation));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "// %s@+%zu: %s",
      loom_check_annotation_severity_name(diagnostic->severity), line_offset,
      loom_error_domain_name(diagnostic->domain)));
  if (diagnostic->code != 0) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(builder, "/%03u", diagnostic->code));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  return iree_ok_status();
}

static bool loom_check_find_input_line_start(
    const loom_check_case_t* test_case, iree_host_size_t target_line,
    loom_check_source_range_t* out_range, iree_string_view_t* out_indentation) {
  *out_indentation = iree_string_view_empty();

  if (target_line == 0 || !test_case->input.data) return false;

  iree_string_view_t scanner = test_case->input;
  iree_host_size_t line_number = 1;
  while (true) {
    const char* line_start =
        scanner.data ? scanner.data
                     : test_case->input.data + test_case->input.size;
    if (line_number == target_line) {
      if (line_start < test_case->input.data ||
          line_start > test_case->input.data + test_case->input.size) {
        return false;
      }
      iree_host_size_t line_offset =
          (iree_host_size_t)(line_start - test_case->input.data);
      iree_host_size_t source_offset =
          test_case->input_range.start_byte + line_offset;
      *out_range = (loom_check_source_range_t){
          .start_byte = source_offset,
          .end_byte = source_offset,
      };

      iree_host_size_t remaining_length =
          (iree_host_size_t)(test_case->input.data + test_case->input.size -
                             line_start);
      iree_string_view_t line =
          iree_make_string_view(line_start, remaining_length);
      intptr_t newline = iree_string_view_find_char(line, '\n', 0);
      if (newline >= 0) {
        line = iree_string_view_substr(line, 0, (iree_host_size_t)newline);
      }
      iree_host_size_t indentation_length = 0;
      while (indentation_length < line.size &&
             (line.data[indentation_length] == ' ' ||
              line.data[indentation_length] == '\t')) {
        ++indentation_length;
      }
      *out_indentation = iree_string_view_substr(line, 0, indentation_length);
      return true;
    }
    if (iree_string_view_is_empty(scanner)) break;

    intptr_t newline = iree_string_view_find_char(scanner, '\n', 0);
    if (newline < 0) {
      scanner = iree_string_view_empty();
    } else {
      scanner = iree_string_view_substr(scanner, (iree_host_size_t)newline + 1,
                                        IREE_HOST_SIZE_MAX);
    }
    ++line_number;
  }
  return false;
}

static loom_check_source_range_t loom_check_annotation_delete_range(
    const loom_check_case_t* test_case,
    const loom_check_annotation_t* annotation) {
  loom_check_source_range_t range = annotation->source_range;
  if (!test_case->input.data ||
      range.start_byte < test_case->input_range.start_byte ||
      range.end_byte > test_case->input_range.end_byte) {
    return range;
  }

  iree_host_size_t relative_end =
      range.end_byte - test_case->input_range.start_byte;
  if (relative_end < test_case->input.size) {
    if (test_case->input.data[relative_end] == '\r') {
      ++range.end_byte;
      ++relative_end;
    }
    if (relative_end < test_case->input.size &&
        test_case->input.data[relative_end] == '\n') {
      ++range.end_byte;
    }
  }
  return range;
}

static iree_status_t loom_check_append_annotation_edit_json(
    loom_check_result_t* result, loom_check_update_edit_kind_t kind,
    loom_check_source_range_t range, iree_host_size_t target_line,
    iree_string_view_t text) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&result->annotation_edits.json, &stream);
  if (result->annotation_edits.count > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\n"));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "\"kind\": "));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      &stream, loom_check_update_edit_kind_name(kind)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      ", \"range\": {\"start_byte\": %zu, \"end_byte\": %zu}, "
      "\"target_line\": %zu, \"text\": ",
      range.start_byte, range.end_byte, target_line));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, text));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
  ++result->annotation_edits.count;
  return iree_ok_status();
}

static iree_status_t loom_check_append_annotation_param_matches(
    const loom_check_annotation_t* annotation, iree_string_builder_t* builder) {
  if (annotation->param_match_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " {"));
  for (uint8_t i = 0; i < annotation->param_match_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ", "));
    }
    const loom_check_annotation_param_match_t* match =
        &annotation->param_matches[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "%.*s=\"%.*s\"", (int)match->name.size, match->name.data,
        (int)match->value.size, match->value.data));
  }
  return iree_string_builder_append_cstring(builder, "}");
}

static bool loom_check_previous_unmatched_diagnostic_on_line(
    const loom_check_collected_diagnostic_t* diagnostics,
    iree_host_size_t begin, uint32_t origin_line) {
  for (iree_host_size_t i = 0; i < begin; ++i) {
    if (!diagnostics[i].matched && diagnostics[i].origin_line == origin_line) {
      return true;
    }
  }
  return false;
}

static iree_host_size_t loom_check_count_unmatched_diagnostics_on_line(
    const loom_check_collected_diagnostic_t* diagnostics,
    iree_host_size_t diagnostic_count, uint32_t origin_line) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < diagnostic_count; ++i) {
    if (!diagnostics[i].matched && diagnostics[i].origin_line == origin_line) {
      ++count;
    }
  }
  return count;
}

static iree_status_t loom_check_build_annotation_edits(
    const loom_check_collected_diagnostic_t* diagnostics,
    iree_host_size_t diagnostic_count,
    const loom_check_annotation_t* annotations,
    iree_host_size_t annotation_count, const loom_check_file_report_t* report,
    iree_host_size_t case_index, const loom_check_case_t* test_case,
    iree_allocator_t allocator, loom_check_result_t* result) {
  for (iree_host_size_t a = 0; a < annotation_count; ++a) {
    bool annotation_matched = false;
    IREE_RETURN_IF_ERROR(loom_check_file_report_annotation_matched(
        report, case_index, a, &annotation_matched));
    if (annotation_matched) continue;
    const loom_check_annotation_t* annotation = &annotations[a];
    loom_check_source_range_t delete_range =
        loom_check_annotation_delete_range(test_case, annotation);
    IREE_RETURN_IF_ERROR(loom_check_append_annotation_edit_json(
        result, LOOM_CHECK_UPDATE_EDIT_DELETE_DIAGNOSTIC_ANNOTATION,
        delete_range, annotation->target_line, iree_string_view_empty()));
  }

  for (iree_host_size_t d = 0; d < diagnostic_count; ++d) {
    const loom_check_collected_diagnostic_t* diagnostic = &diagnostics[d];
    if (diagnostic->matched || diagnostic->origin_line == 0) continue;
    if (loom_check_previous_unmatched_diagnostic_on_line(
            diagnostics, d, diagnostic->origin_line)) {
      continue;
    }

    loom_check_source_range_t insert_range = loom_check_source_range_empty();
    iree_string_view_t indentation = iree_string_view_empty();
    if (!loom_check_find_input_line_start(test_case, diagnostic->origin_line,
                                          &insert_range, &indentation)) {
      continue;
    }

    iree_string_builder_t text;
    iree_string_builder_initialize(allocator, &text);
    iree_status_t status = iree_ok_status();
    iree_host_size_t line_diagnostic_count =
        loom_check_count_unmatched_diagnostics_on_line(
            diagnostics, diagnostic_count, diagnostic->origin_line);
    iree_host_size_t line_offset = line_diagnostic_count;
    for (iree_host_size_t i = d;
         iree_status_is_ok(status) && i < diagnostic_count; ++i) {
      if (diagnostics[i].matched ||
          diagnostics[i].origin_line != diagnostic->origin_line) {
        continue;
      }
      status = loom_check_append_diagnostic_annotation_line(
          &diagnostics[i], line_offset, indentation, &text);
      --line_offset;
    }
    if (iree_status_is_ok(status)) {
      status = loom_check_append_annotation_edit_json(
          result, LOOM_CHECK_UPDATE_EDIT_INSERT_DIAGNOSTIC_ANNOTATIONS,
          insert_range, diagnostic->origin_line,
          iree_string_builder_view(&text));
    }
    iree_string_builder_deinitialize(&text);
    IREE_RETURN_IF_ERROR(status);
  }

  return iree_ok_status();
}

static iree_status_t loom_check_assemble_diagnostic_match_detail(
    const loom_check_collected_diagnostic_t* diagnostics,
    iree_host_size_t diagnostic_count,
    const loom_check_annotation_t* annotations,
    iree_host_size_t annotation_count, const loom_check_file_report_t* report,
    iree_host_size_t case_index, iree_string_builder_t* detail) {
  for (iree_host_size_t a = 0; a < annotation_count; ++a) {
    bool annotation_matched = false;
    IREE_RETURN_IF_ERROR(loom_check_file_report_annotation_matched(
        report, case_index, a, &annotation_matched));
    if (annotation_matched) continue;
    const loom_check_annotation_t* annotation = &annotations[a];
    const char* severity_name =
        loom_diagnostic_severity_name(annotation->severity);
    if (annotation->domain != LOOM_ERROR_DOMAIN_COUNT_ &&
        annotation->code != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          detail, "unmatched annotation line %zu: expected %s %s/%03u",
          annotation->target_line, severity_name,
          loom_error_domain_name(annotation->domain), annotation->code));
    } else if (annotation->domain != LOOM_ERROR_DOMAIN_COUNT_) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          detail, "unmatched annotation line %zu: expected %s %s/*",
          annotation->target_line, severity_name,
          loom_error_domain_name(annotation->domain)));
    } else if (annotation->code != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          detail, "unmatched annotation line %zu: expected %s */%03u",
          annotation->target_line, severity_name, annotation->code));
    } else {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          detail, "unmatched annotation line %zu: expected %s",
          annotation->target_line, severity_name));
    }
    IREE_RETURN_IF_ERROR(
        loom_check_append_annotation_param_matches(annotation, detail));
    for (uint8_t i = 0; i < annotation->message_substring_count; ++i) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          detail, " \"%.*s\"", (int)annotation->message_substrings[i].size,
          annotation->message_substrings[i].data));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(detail, "\n"));
  }

  for (iree_host_size_t d = 0; d < diagnostic_count; ++d) {
    if (diagnostics[d].matched) continue;
    const loom_check_collected_diagnostic_t* diagnostic = &diagnostics[d];
    const char* severity_name =
        loom_diagnostic_severity_name(diagnostic->severity);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        detail, "unexpected %s at line %u: [%s/%03u] %.*s\n", severity_name,
        diagnostic->origin_line, loom_error_domain_name(diagnostic->domain),
        diagnostic->code, (int)diagnostic->message.size,
        diagnostic->message.data));
    if (!iree_string_view_is_empty(diagnostic->formatted_diagnostic)) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          detail, diagnostic->formatted_diagnostic));
    }
  }

  return iree_ok_status();
}

iree_status_t loom_check_diagnostic_collector_finish(
    loom_check_diagnostic_collector_t* collector,
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_allocator_t allocator,
    loom_check_result_t* result) {
  IREE_RETURN_IF_ERROR(loom_check_match_annotations(
      collector->diagnostics, collector->count, test_case->annotations,
      test_case->annotation_count, report, case_index));

  bool all_annotations_matched = true;
  for (iree_host_size_t i = 0; i < test_case->annotation_count; ++i) {
    bool annotation_matched = false;
    IREE_RETURN_IF_ERROR(loom_check_file_report_annotation_matched(
        report, case_index, i, &annotation_matched));
    if (!annotation_matched) {
      all_annotations_matched = false;
      break;
    }
  }

  bool all_diagnostics_matched = true;
  for (iree_host_size_t i = 0; i < collector->count; ++i) {
    if (!collector->diagnostics[i].matched) {
      all_diagnostics_matched = false;
      break;
    }
  }

  if (all_annotations_matched && all_diagnostics_matched) {
    result->raw_outcome = LOOM_CHECK_PASS;
    return iree_ok_status();
  }

  result->raw_outcome = LOOM_CHECK_FAIL;
  IREE_RETURN_IF_ERROR(loom_check_assemble_diagnostic_match_detail(
      collector->diagnostics, collector->count, test_case->annotations,
      test_case->annotation_count, report, case_index, &result->detail));
  return loom_check_build_annotation_edits(
      collector->diagnostics, collector->count, test_case->annotations,
      test_case->annotation_count, report, case_index, test_case, allocator,
      result);
}
