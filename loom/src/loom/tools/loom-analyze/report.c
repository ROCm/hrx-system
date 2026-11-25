// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-analyze/report.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "loom/util/json.h"

iree_status_t loom_analyze_report_parse_format(
    iree_string_view_t value, loom_analyze_report_format_t* out_format) {
  if (iree_string_view_equal(value, IREE_SV("text"))) {
    *out_format = LOOM_ANALYZE_REPORT_FORMAT_TEXT;
  } else if (iree_string_view_equal(value, IREE_SV("json"))) {
    *out_format = LOOM_ANALYZE_REPORT_FORMAT_JSON;
  } else if (iree_string_view_equal(value, IREE_SV("sarif"))) {
    *out_format = LOOM_ANALYZE_REPORT_FORMAT_SARIF;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown --format '%.*s'; expected text, json, or "
                            "sarif",
                            (int)value.size, value.data);
  }
  return iree_ok_status();
}

static uint32_t loom_analyze_copy_count(const loom_analyze_finding_t* finding) {
  if (finding->rule != LOOM_ANALYZE_RULE_REIMPLEMENTED_MOTIF) {
    return 1;
  }
  uint32_t count = 0;
  for (uint32_t i = 0; i < finding->occurrence_count; ++i) {
    if (!finding->occurrences[i].is_motif_definition) {
      ++count;
    }
  }
  return count;
}

iree_host_size_t loom_analyze_report_result_count(
    const loom_analyze_result_t* result) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < result->finding_count; ++i) {
    count += loom_analyze_copy_count(&result->findings[i]);
  }
  return count;
}

static const loom_analyze_occurrence_t* loom_analyze_motif_definition(
    const loom_analyze_finding_t* finding) {
  for (uint32_t i = 0; i < finding->occurrence_count; ++i) {
    if (finding->occurrences[i].is_motif_definition) {
      return &finding->occurrences[i];
    }
  }
  return NULL;
}

static iree_string_view_t loom_analyze_file_or_unknown(
    iree_string_view_t file) {
  return iree_string_view_is_empty(file) ? IREE_SV("<unknown>") : file;
}

// Writes the primary message shared by every result of |finding|.
static iree_status_t loom_analyze_write_message(
    const loom_analyze_finding_t* finding, loom_output_stream_t* stream) {
  if (finding->rule == LOOM_ANALYZE_RULE_REIMPLEMENTED_MOTIF) {
    return loom_output_stream_write_format(
        stream,
        "%" PRIu32
        "-op sequence repeats the body of motif @%.*s%s; call the "
        "motif instead of copying it",
        finding->op_count, (int)finding->motif_symbol.size,
        finding->motif_symbol.data,
        finding->parameterized ? " (up to constant values)" : "");
  }
  return loom_output_stream_write_format(
      stream,
      "%" PRIu32 "-op sequence is duplicated %" PRIu32
      " times%s; factor it into a func.def inline or template.def motif",
      finding->op_count, finding->occurrence_count,
      finding->parameterized ? " (up to constant values)" : "");
}

// Writes the note text for one occurrence, without location or severity.
static iree_status_t loom_analyze_write_occurrence_note(
    const loom_analyze_occurrence_t* occurrence, loom_output_stream_t* stream) {
  if (occurrence->is_motif_definition) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "motif @%.*s is defined here", (int)occurrence->symbol.size,
        occurrence->symbol.data));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "copy spans lines %" PRIu32 "-%" PRIu32, occurrence->start_line,
        occurrence->end_line));
    if (!iree_string_view_is_empty(occurrence->symbol)) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream, " in @%.*s", (int)occurrence->symbol.size,
          occurrence->symbol.data));
    }
  }
  if (occurrence->in_launch_config) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        stream,
        "; this copy is in a kernel launch-config region, which cannot call "
        "functions today"));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Text
//===----------------------------------------------------------------------===//

static iree_status_t loom_analyze_write_text_location(
    const loom_analyze_occurrence_t* occurrence, loom_output_stream_t* stream) {
  const iree_string_view_t file =
      loom_analyze_file_or_unknown(occurrence->file);
  return loom_output_stream_write_format(
      stream, "%.*s:%" PRIu32 ":%" PRIu32 ": ", (int)file.size, file.data,
      occurrence->start_line, occurrence->start_column);
}

static iree_status_t loom_analyze_write_text_note(
    const loom_analyze_occurrence_t* occurrence, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_analyze_write_text_location(occurrence, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "note: "));
  IREE_RETURN_IF_ERROR(loom_analyze_write_occurrence_note(occurrence, stream));
  return loom_output_stream_write_char(stream, '\n');
}

static iree_status_t loom_analyze_write_text_warning(
    const loom_analyze_finding_t* finding,
    const loom_analyze_occurrence_t* anchor, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_analyze_write_text_location(anchor, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "warning: "));
  IREE_RETURN_IF_ERROR(loom_analyze_write_message(finding, stream));
  const iree_string_view_t rule_id = loom_analyze_rule_id(finding->rule);
  return loom_output_stream_write_format(stream, " [%.*s]\n", (int)rule_id.size,
                                         rule_id.data);
}

static iree_status_t loom_analyze_write_text(
    const loom_analyze_result_t* result, loom_output_stream_t* stream) {
  for (iree_host_size_t f = 0; f < result->finding_count; ++f) {
    const loom_analyze_finding_t* finding = &result->findings[f];
    if (finding->rule == LOOM_ANALYZE_RULE_REIMPLEMENTED_MOTIF) {
      const loom_analyze_occurrence_t* definition =
          loom_analyze_motif_definition(finding);
      for (uint32_t i = 0; i < finding->occurrence_count; ++i) {
        const loom_analyze_occurrence_t* copy = &finding->occurrences[i];
        if (copy->is_motif_definition) {
          continue;
        }
        IREE_RETURN_IF_ERROR(
            loom_analyze_write_text_warning(finding, copy, stream));
        IREE_RETURN_IF_ERROR(loom_analyze_write_text_note(copy, stream));
        if (definition) {
          IREE_RETURN_IF_ERROR(
              loom_analyze_write_text_note(definition, stream));
        }
      }
    } else {
      IREE_RETURN_IF_ERROR(loom_analyze_write_text_warning(
          finding, &finding->occurrences[0], stream));
      for (uint32_t i = 0; i < finding->occurrence_count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_analyze_write_text_note(&finding->occurrences[i], stream));
      }
    }
  }
  const loom_analyze_summary_t* summary = &result->summary;
  const double percent = summary->op_count
                             ? 100.0 * (double)summary->duplicated_op_count /
                                   (double)summary->op_count
                             : 0.0;
  const double plan_percent =
      summary->op_count ? 100.0 * (double)summary->removable_op_count /
                              (double)summary->op_count
                        : 0.0;
  return loom_output_stream_write_format(
      stream,
      "loom-analyze: %" PRIhsz " finding(s) in %" PRIu32 " module(s); %" PRIu64
      " of %" PRIu64
      " ops (%.1f%%) are in duplicated sequences\n"
      "loom-analyze: refactor plan of %" PRIu32
      " non-overlapping finding(s) removes %" PRIu64 " ops (%.1f%%)\n",
      loom_analyze_report_result_count(result), summary->module_count,
      summary->duplicated_op_count, summary->op_count, percent,
      summary->selected_count, summary->removable_op_count, plan_percent);
}

//===----------------------------------------------------------------------===//
// JSON helpers
//===----------------------------------------------------------------------===//

static iree_status_t loom_analyze_write_fingerprint_string(
    uint64_t fingerprint, loom_output_stream_t* stream) {
  char buffer[17];
  snprintf(buffer, sizeof(buffer), "%016" PRIx64, fingerprint);
  return loom_json_write_escaped_string(stream, iree_make_cstring_view(buffer));
}

static iree_status_t loom_analyze_write_message_string(
    const loom_analyze_finding_t* finding, loom_output_stream_t* stream) {
  loom_json_string_writer_t writer;
  IREE_RETURN_IF_ERROR(loom_json_string_begin(stream, &writer));
  IREE_RETURN_IF_ERROR(loom_analyze_write_message(finding, &writer.stream));
  return loom_json_string_end(&writer);
}

static iree_status_t loom_analyze_write_note_string(
    const loom_analyze_occurrence_t* occurrence, loom_output_stream_t* stream) {
  loom_json_string_writer_t writer;
  IREE_RETURN_IF_ERROR(loom_json_string_begin(stream, &writer));
  IREE_RETURN_IF_ERROR(
      loom_analyze_write_occurrence_note(occurrence, &writer.stream));
  return loom_json_string_end(&writer);
}

//===----------------------------------------------------------------------===//
// JSON
//===----------------------------------------------------------------------===//

// Writes `"field": [{"name": "%x", "type": "index"}, ...]`. Names carry the
// '%' sigil so a rewriter can splice them into source unchanged.
static iree_status_t loom_analyze_write_bindings(
    loom_json_object_writer_t* parent, iree_string_view_t field,
    const loom_analyze_binding_t* bindings, uint32_t count) {
  loom_output_stream_t* stream = parent->stream;
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(parent, field));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  for (uint32_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("name")));
    if (iree_string_view_is_empty(bindings[i].name)) {
      IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, IREE_SV("")));
    } else {
      loom_json_string_writer_t name;
      IREE_RETURN_IF_ERROR(loom_json_string_begin(stream, &name));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&name.stream, '%'));
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write(&name.stream, bindings[i].name));
      IREE_RETURN_IF_ERROR(loom_json_string_end(&name));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("type"), bindings[i].type));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  return loom_json_array_end(&array);
}

static iree_status_t loom_analyze_write_json_summary(
    const loom_analyze_result_t* result, loom_json_object_writer_t* parent) {
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(parent, IREE_SV("summary")));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(parent->stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("modules"), result->summary.module_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("ops"), result->summary.op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("duplicated_ops"), result->summary.duplicated_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &object, IREE_SV("removable_ops"), result->summary.removable_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("selected"), result->summary.selected_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("results"), loom_analyze_report_result_count(result)));
  return loom_json_object_end(&object);
}

static iree_status_t loom_analyze_write_json(
    const loom_analyze_result_t* result, const loom_analyze_options_t* options,
    iree_string_view_t tool_version, loom_output_stream_t* stream) {
  loom_json_object_writer_t root;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &root));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &root, IREE_SV("tool"), IREE_SV("loom-analyze")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &root, IREE_SV("version"), tool_version));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &root, IREE_SV("min_ops"), options->min_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &root, IREE_SV("min_occurrences"), options->min_occurrence_count));
  IREE_RETURN_IF_ERROR(loom_analyze_write_json_summary(result, &root));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&root, IREE_SV("findings")));
  loom_json_array_writer_t findings;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &findings));
  for (iree_host_size_t f = 0; f < result->finding_count; ++f) {
    const loom_analyze_finding_t* finding = &result->findings[f];
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&findings));
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("rule"), loom_analyze_rule_id(finding->rule)));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("message")));
    IREE_RETURN_IF_ERROR(loom_analyze_write_message_string(finding, stream));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("op_count"), finding->op_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("input_count"), finding->input_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("output_count"), finding->output_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
        &object, IREE_SV("removable_ops"), finding->removable_op_count));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("fingerprint")));
    IREE_RETURN_IF_ERROR(
        loom_analyze_write_fingerprint_string(finding->fingerprint, stream));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &object, IREE_SV("parameterized"), finding->parameterized));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &object, IREE_SV("selected"), finding->selected));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
        &object, IREE_SV("motif"), finding->motif_symbol));

    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("occurrences")));
    loom_json_array_writer_t occurrences;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &occurrences));
    for (uint32_t i = 0; i < finding->occurrence_count; ++i) {
      const loom_analyze_occurrence_t* occurrence = &finding->occurrences[i];
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&occurrences));
      loom_json_object_writer_t entry;
      IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &entry));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &entry, IREE_SV("file"), occurrence->file));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &entry, IREE_SV("start_line"), occurrence->start_line));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &entry, IREE_SV("start_column"), occurrence->start_column));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &entry, IREE_SV("end_line"), occurrence->end_line));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &entry, IREE_SV("end_column"), occurrence->end_column));
      IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
          &entry, IREE_SV("callable"), !occurrence->in_launch_config));
      IREE_RETURN_IF_ERROR(loom_analyze_write_bindings(
          &entry, IREE_SV("inputs"), occurrence->inputs,
          occurrence->input_count));
      IREE_RETURN_IF_ERROR(loom_analyze_write_bindings(
          &entry, IREE_SV("outputs"), occurrence->outputs,
          occurrence->output_count));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
          &entry, IREE_SV("symbol"), occurrence->symbol));
      IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
          &entry, IREE_SV("launch_config"), occurrence->in_launch_config));
      IREE_RETURN_IF_ERROR(
          loom_json_object_write_bool_field(&entry, IREE_SV("motif_definition"),
                                            occurrence->is_motif_definition));
      IREE_RETURN_IF_ERROR(loom_json_object_end(&entry));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&occurrences));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&findings));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&root));
  return loom_output_stream_write_char(stream, '\n');
}

//===----------------------------------------------------------------------===//
// SARIF 2.1.0
//===----------------------------------------------------------------------===//

static bool loom_analyze_is_uri_unreserved(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '-' || c == '.' || c == '_' ||
         c == '~' || c == '/' || c == ':';
}

// Writes |path| as a SARIF artifact URI string: relative paths stay relative
// references, absolute paths become file:// URIs, separators become '/', and
// reserved bytes are percent-encoded.
static iree_status_t loom_analyze_write_artifact_uri(
    iree_string_view_t path, loom_output_stream_t* stream) {
  loom_json_string_writer_t writer;
  IREE_RETURN_IF_ERROR(loom_json_string_begin(stream, &writer));
  loom_output_stream_t* out = &writer.stream;
  const bool windows_absolute = path.size >= 2 && path.data[1] == ':';
  const bool posix_absolute = path.size >= 1 && (path.data[0] == '/');
  if (windows_absolute) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(out, "file:///"));
  } else if (posix_absolute) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(out, "file://"));
  }
  for (iree_host_size_t i = 0; i < path.size; ++i) {
    const char c = path.data[i] == '\\' ? '/' : path.data[i];
    if (loom_analyze_is_uri_unreserved(c)) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(out, c));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_format(out, "%%%02X", (unsigned char)c));
    }
  }
  return loom_json_string_end(&writer);
}

static iree_status_t loom_analyze_write_text_object(
    loom_output_stream_t* stream, iree_string_view_t text) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_string_field(&object, IREE_SV("text"), text));
  return loom_json_object_end(&object);
}

// Writes a SARIF location object; |id| < 0 omits the id, and |note| writes a
// location message when true.
static iree_status_t loom_analyze_write_sarif_location(
    const loom_analyze_occurrence_t* occurrence, int32_t id, bool note,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t location;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &location));
  if (id >= 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_int32_field(&location, IREE_SV("id"), id));
  }
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&location, IREE_SV("physicalLocation")));
  loom_json_object_writer_t physical;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &physical));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&physical, IREE_SV("artifactLocation")));
  loom_json_object_writer_t artifact;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &artifact));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&artifact, IREE_SV("uri")));
  IREE_RETURN_IF_ERROR(loom_analyze_write_artifact_uri(
      loom_analyze_file_or_unknown(occurrence->file), stream));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&artifact));
  if (occurrence->start_line > 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&physical, IREE_SV("region")));
    loom_json_object_writer_t region;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &region));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &region, IREE_SV("startLine"), occurrence->start_line));
    if (occurrence->start_column > 0) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &region, IREE_SV("startColumn"), occurrence->start_column));
    }
    if (occurrence->end_line >= occurrence->start_line) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &region, IREE_SV("endLine"), occurrence->end_line));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&region));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_end(&physical));
  if (!iree_string_view_is_empty(occurrence->symbol)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&location, IREE_SV("logicalLocations")));
    loom_json_array_writer_t logical;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &logical));
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&logical));
    loom_json_object_writer_t entry;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &entry));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &entry, IREE_SV("name"), occurrence->symbol));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &entry, IREE_SV("kind"), IREE_SV("function")));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&entry));
    IREE_RETURN_IF_ERROR(loom_json_array_end(&logical));
  }
  if (note) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&location, IREE_SV("message")));
    loom_json_object_writer_t message;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &message));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&message, IREE_SV("text")));
    IREE_RETURN_IF_ERROR(loom_analyze_write_note_string(occurrence, stream));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&message));
  }
  return loom_json_object_end(&location);
}

static iree_status_t loom_analyze_write_sarif_rules(
    loom_json_object_writer_t* driver) {
  loom_output_stream_t* stream = driver->stream;
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(driver, IREE_SV("rules")));
  loom_json_array_writer_t rules;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &rules));
  for (int r = 0; r < LOOM_ANALYZE_RULE_COUNT_; ++r) {
    const loom_analyze_rule_t rule = (loom_analyze_rule_t)r;
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&rules));
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    const iree_string_view_t id = loom_analyze_rule_id(rule);
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_string_field(&object, IREE_SV("id"), id));
    // The rule name is the ID without the tool prefix, e.g.
    // "duplicate-sequence".
    iree_string_view_t name = id;
    iree_host_size_t slash = iree_string_view_find_char(id, '/', 0);
    if (slash != IREE_STRING_VIEW_NPOS) {
      name = iree_string_view_substr(id, slash + 1, IREE_HOST_SIZE_MAX);
    }
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_string_field(&object, IREE_SV("name"), name));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("shortDescription")));
    IREE_RETURN_IF_ERROR(loom_analyze_write_text_object(
        stream, loom_analyze_rule_description(rule)));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("defaultConfiguration")));
    loom_json_object_writer_t configuration;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &configuration));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &configuration, IREE_SV("level"), IREE_SV("warning")));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&configuration));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  return loom_json_array_end(&rules);
}

// Writes one SARIF result. |anchor| is the primary location; every other
// occurrence in |finding| except |anchor| becomes a related location.
// |ordinal| distinguishes multiple results produced by one finding.
static iree_status_t loom_analyze_write_sarif_result(
    const loom_analyze_finding_t* finding,
    const loom_analyze_occurrence_t* anchor, uint32_t ordinal,
    loom_json_array_writer_t* results) {
  loom_output_stream_t* stream = results->stream;
  IREE_RETURN_IF_ERROR(loom_json_array_begin_element(results));
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("ruleId"), loom_analyze_rule_id(finding->rule)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_int32_field(
      &object, IREE_SV("ruleIndex"), (int32_t)finding->rule));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("level"), IREE_SV("warning")));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("message")));
  loom_json_object_writer_t message;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &message));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&message, IREE_SV("text")));
  IREE_RETURN_IF_ERROR(loom_analyze_write_message_string(finding, stream));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&message));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("locations")));
  loom_json_array_writer_t locations;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &locations));
  IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&locations));
  IREE_RETURN_IF_ERROR(loom_analyze_write_sarif_location(
      anchor, /*id=*/-1, /*note=*/false, stream));
  IREE_RETURN_IF_ERROR(loom_json_array_end(&locations));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("relatedLocations")));
  loom_json_array_writer_t related;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &related));
  int32_t related_id = 0;
  for (uint32_t i = 0; i < finding->occurrence_count; ++i) {
    const loom_analyze_occurrence_t* occurrence = &finding->occurrences[i];
    if (occurrence == anchor) {
      continue;
    }
    // A motif result relates only to its definition, not to sibling copies.
    if (finding->rule == LOOM_ANALYZE_RULE_REIMPLEMENTED_MOTIF &&
        !occurrence->is_motif_definition) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&related));
    IREE_RETURN_IF_ERROR(loom_analyze_write_sarif_location(
        occurrence, related_id++, /*note=*/true, stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&related));

  // The canonical sequence hash is stable across runs and independent of
  // file names; the ordinal separates copies reported for one motif.
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("partialFingerprints")));
  loom_json_object_writer_t fingerprints;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &fingerprints));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
      &fingerprints, IREE_SV("loomSequenceHash/v1")));
  char buffer[32];
  snprintf(buffer, sizeof(buffer), "%016" PRIx64 ":%" PRIu32,
           finding->fingerprint, ordinal);
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(stream, iree_make_cstring_view(buffer)));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&fingerprints));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("properties")));
  loom_json_object_writer_t properties;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &properties));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &properties, IREE_SV("opCount"), finding->op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &properties, IREE_SV("occurrenceCount"), finding->occurrence_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &properties, IREE_SV("parameterized"), finding->parameterized));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &properties, IREE_SV("inputCount"), finding->input_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &properties, IREE_SV("outputCount"), finding->output_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &properties, IREE_SV("removableOpCount"), finding->removable_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&properties));
  return loom_json_object_end(&object);
}

static iree_status_t loom_analyze_write_sarif(
    const loom_analyze_result_t* result, iree_string_view_t tool_version,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t root;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &root));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &root, IREE_SV("$schema"),
      IREE_SV("https://json.schemastore.org/sarif-2.1.0.json")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &root, IREE_SV("version"), IREE_SV("2.1.0")));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&root, IREE_SV("runs")));
  loom_json_array_writer_t runs;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &runs));
  IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&runs));
  loom_json_object_writer_t run;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &run));

  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&run, IREE_SV("tool")));
  loom_json_object_writer_t tool;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &tool));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&tool, IREE_SV("driver")));
  loom_json_object_writer_t driver;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &driver));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &driver, IREE_SV("name"), IREE_SV("loom-analyze")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &driver, IREE_SV("semanticVersion"), tool_version));
  IREE_RETURN_IF_ERROR(loom_analyze_write_sarif_rules(&driver));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&driver));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&tool));

  // Loom source columns count Unicode code points.
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &run, IREE_SV("columnKind"), IREE_SV("unicodeCodePoints")));

  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&run, IREE_SV("results")));
  loom_json_array_writer_t results;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &results));
  for (iree_host_size_t f = 0; f < result->finding_count; ++f) {
    const loom_analyze_finding_t* finding = &result->findings[f];
    if (finding->rule == LOOM_ANALYZE_RULE_REIMPLEMENTED_MOTIF) {
      uint32_t ordinal = 0;
      for (uint32_t i = 0; i < finding->occurrence_count; ++i) {
        const loom_analyze_occurrence_t* copy = &finding->occurrences[i];
        if (copy->is_motif_definition) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_analyze_write_sarif_result(
            finding, copy, ordinal++, &results));
      }
    } else {
      IREE_RETURN_IF_ERROR(loom_analyze_write_sarif_result(
          finding, &finding->occurrences[0], 0, &results));
    }
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&results));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&run, IREE_SV("properties")));
  loom_json_object_writer_t properties;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &properties));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &properties, IREE_SV("moduleCount"), result->summary.module_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &properties, IREE_SV("opCount"), result->summary.op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
      &properties, IREE_SV("duplicatedOpCount"),
      result->summary.duplicated_op_count));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&properties));

  IREE_RETURN_IF_ERROR(loom_json_object_end(&run));
  IREE_RETURN_IF_ERROR(loom_json_array_end(&runs));
  IREE_RETURN_IF_ERROR(loom_json_object_end(&root));
  return loom_output_stream_write_char(stream, '\n');
}

iree_status_t loom_analyze_report_write(const loom_analyze_result_t* result,
                                        const loom_analyze_options_t* options,
                                        loom_analyze_report_format_t format,
                                        iree_string_view_t tool_version,
                                        loom_output_stream_t* stream) {
  switch (format) {
    case LOOM_ANALYZE_REPORT_FORMAT_TEXT:
      return loom_analyze_write_text(result, stream);
    case LOOM_ANALYZE_REPORT_FORMAT_JSON:
      return loom_analyze_write_json(result, options, tool_version, stream);
    case LOOM_ANALYZE_REPORT_FORMAT_SARIF:
      return loom_analyze_write_sarif(result, tool_version, stream);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown report format %d", (int)format);
}
