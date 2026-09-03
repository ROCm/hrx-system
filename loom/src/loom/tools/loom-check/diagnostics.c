// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/diagnostics.h"

#include <string.h>

#include "loom/error/renderer.h"
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

iree_status_t loom_check_diagnostic_collector_sink(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loom_check_diagnostic_collector_t* collector =
      (loom_check_diagnostic_collector_t*)user_data;
  if (collector->count >= collector->capacity) {
    IREE_RETURN_IF_ERROR(loom_check_diagnostic_collector_grow(collector));
  }

  const loom_test_diagnostic_format_options_t format_options = {
      .module = collector->module,
      .text_print_options = collector->type_print_context.options,
  };
  IREE_RETURN_IF_ERROR(loom_test_diagnostic_materialize(
      diagnostic, &format_options, collector->arena, collector->host_allocator,
      &collector->diagnostics[collector->count]));
  ++collector->count;

  loom_type_formatter_t type_formatter = {loom_check_format_type, collector};
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
    const loom_check_diagnostic_emitter_capture_t* capture,
    const loom_module_t* module, const loom_op_t* op,
    loom_source_range_t* out_source_location) {
  if (!capture || !op) return false;
  if (!module) module = capture->module;
  if (!module || !loom_source_resolve(capture->source_resolver, module,
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
    if (!loom_check_diagnostic_resolve_location(capture, related_ops[i].module,
                                                related_ops[i].op,
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
      .severity = loom_error_def_severity(emission->error),
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

  if (loom_check_diagnostic_resolve_location(capture, emission->module,
                                             emission->op,
                                             &diagnostic.source_location)) {
    diagnostic.origin = diagnostic.source_location;
  }

  const loom_module_t* diagnostic_module =
      emission->module ? emission->module : capture->module;
  const loom_module_t* previous_module = capture->diagnostic_collector->module;
  capture->diagnostic_collector->module = diagnostic_module;
  iree_status_t status = loom_check_diagnostic_collector_sink(
      capture->diagnostic_collector, &diagnostic);
  capture->diagnostic_collector->module = previous_module;
  IREE_RETURN_IF_ERROR(status);
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
    const loom_test_case_t* test_case, iree_string_view_t filename,
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
      .severity = loom_error_def_severity(error),
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

static iree_status_t loom_check_match_annotations(
    loom_check_collected_diagnostic_t* diagnostics,
    iree_host_size_t diagnostic_count,
    const loom_test_annotation_t* annotations,
    iree_host_size_t annotation_count, loom_check_file_report_t* report,
    iree_host_size_t case_index, iree_arena_allocator_t* arena) {
  iree_host_size_t* annotation_to_diagnostic = NULL;
  IREE_RETURN_IF_ERROR(loom_test_diagnostics_match_annotations(
      diagnostics, diagnostic_count, annotations, annotation_count, arena,
      &annotation_to_diagnostic));
  for (iree_host_size_t i = 0; i < annotation_count; ++i) {
    if (annotation_to_diagnostic[i] != IREE_HOST_SIZE_MAX) {
      IREE_RETURN_IF_ERROR(loom_check_file_report_mark_annotation_matched(
          report, case_index, i));
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
    const loom_test_case_t* test_case, iree_host_size_t target_line,
    loom_test_source_range_t* out_range, iree_string_view_t* out_indentation) {
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
      *out_range = (loom_test_source_range_t){
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

static loom_test_source_range_t loom_check_annotation_delete_range(
    const loom_test_case_t* test_case,
    const loom_test_annotation_t* annotation) {
  loom_test_source_range_t range = annotation->source_range;
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

static iree_status_t loom_check_append_annotation_param_matches(
    const loom_test_annotation_t* annotation, iree_string_builder_t* builder) {
  if (annotation->param_match_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, " {"));
  for (uint8_t i = 0; i < annotation->param_match_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ", "));
    }
    const loom_test_annotation_param_match_t* match =
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
    const loom_test_annotation_t* annotations,
    iree_host_size_t annotation_count, const loom_check_file_report_t* report,
    iree_host_size_t case_index, const loom_test_case_t* test_case,
    iree_allocator_t allocator, loom_check_result_t* result) {
  for (iree_host_size_t a = 0; a < annotation_count; ++a) {
    bool annotation_matched = false;
    IREE_RETURN_IF_ERROR(loom_check_file_report_annotation_matched(
        report, case_index, a, &annotation_matched));
    if (annotation_matched) continue;
    const loom_test_annotation_t* annotation = &annotations[a];
    loom_test_source_range_t delete_range =
        loom_check_annotation_delete_range(test_case, annotation);
    IREE_RETURN_IF_ERROR(loom_check_result_append_annotation_edit(
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

    loom_test_source_range_t insert_range = loom_test_source_range_empty();
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
      status = loom_check_result_append_annotation_edit(
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
    const loom_test_annotation_t* annotations,
    iree_host_size_t annotation_count, const loom_check_file_report_t* report,
    iree_host_size_t case_index, iree_string_builder_t* detail) {
  for (iree_host_size_t a = 0; a < annotation_count; ++a) {
    bool annotation_matched = false;
    IREE_RETURN_IF_ERROR(loom_check_file_report_annotation_matched(
        report, case_index, a, &annotation_matched));
    if (annotation_matched) continue;
    const loom_test_annotation_t* annotation = &annotations[a];
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
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_allocator_t allocator,
    loom_check_result_t* result) {
  IREE_RETURN_IF_ERROR(loom_check_match_annotations(
      collector->diagnostics, collector->count, test_case->annotations,
      test_case->annotation_count, report, case_index, collector->arena));

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
