// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/json_sink.h"

#include <inttypes.h>

#include "loom/error/renderer.h"
#include "loom/util/json.h"

//===----------------------------------------------------------------------===//
// Param value rendering for JSON
//===----------------------------------------------------------------------===//

// Renders a single param value as a JSON scalar or array.
static iree_status_t loom_json_render_param_value(
    const loom_diagnostic_param_t* param, loom_type_formatter_t type_formatter,
    loom_output_stream_t* stream) {
  switch (param->kind) {
    case LOOM_PARAM_STRING:
      return loom_json_write_escaped_string(stream, param->string);
    case LOOM_PARAM_I64:
      return loom_output_stream_write_format(stream, "%" PRId64, param->i64);
    case LOOM_PARAM_U32:
      return loom_output_stream_write_format(stream, "%" PRIu32, param->u32);
    case LOOM_PARAM_U64:
      return loom_output_stream_write_format(stream, "%" PRIu64, param->u64);
    case LOOM_PARAM_STRING_LIST:
      if (param->string_list.count > 0 && !param->string_list.values) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "string list param has count > 0 but values NULL");
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
      for (iree_host_size_t i = 0; i < param->string_list.count; ++i) {
        if (i > 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
        }
        IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
            stream, param->string_list.values[i]));
      }
      return loom_output_stream_write_char(stream, ']');
    case LOOM_PARAM_BOOL:
      return loom_output_stream_write_cstring(
          stream, param->boolean ? "true" : "false");
    case LOOM_PARAM_TYPE: {
      // Render the type through the JSON-escaping adapter directly
      // into the output stream. Zero allocations.
      loom_json_escape_stream_t escape_data;
      loom_output_stream_t escape_stream;
      loom_json_escape_stream_init(stream, &escape_data, &escape_stream);
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
      if (type_formatter.fn) {
        IREE_RETURN_IF_ERROR(type_formatter.fn(
            param->type, type_formatter.user_data, &escape_stream));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(&escape_stream, "<type>"));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
      return iree_ok_status();
    }
    default:
      return loom_json_write_escaped_cstring(stream, "<?>");
  }
}

//===----------------------------------------------------------------------===//
// JSON sink implementation
//===----------------------------------------------------------------------===//

enum {
  LOOM_JSON_SOURCE_EXCERPT_CONTEXT_BYTES = 32,
  LOOM_JSON_SOURCE_EXCERPT_MAX_BYTES = 192,
};

// Returns the JSON field-kind string for a diagnostic field ref.
static const char* loom_json_diagnostic_field_kind_name(
    loom_diagnostic_field_kind_t kind) {
  switch (kind) {
    case LOOM_DIAGNOSTIC_FIELD_OPERAND:
      return "operand";
    case LOOM_DIAGNOSTIC_FIELD_RESULT:
      return "result";
    case LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE:
      return "attribute";
    case LOOM_DIAGNOSTIC_FIELD_REGION:
      return "region";
    case LOOM_DIAGNOSTIC_FIELD_SUCCESSOR:
      return "successor";
    case LOOM_DIAGNOSTIC_FIELD_NONE:
    default:
      return NULL;
  }
}

// Renders one structured field ref object.
static iree_status_t loom_json_render_field_ref(
    loom_output_stream_t* stream, loom_diagnostic_field_ref_t field_ref) {
  const char* kind_name = loom_json_diagnostic_field_kind_name(field_ref.kind);
  if (!kind_name) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "invalid diagnostic field ref kind %d",
                            (int)field_ref.kind);
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{\"kind\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(stream, kind_name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"index\":%" PRIu16 ",\"occurrence\":%" PRIu16 "}",
      field_ref.index, field_ref.occurrence));
  return iree_ok_status();
}

// Returns true if the range carries any location metadata worth serializing.
// Source bytes are serialized only as a bounded excerpt object below, not as
// the full backing buffer, so each JSONL diagnostic stays small.
static bool loom_json_source_range_has_metadata(
    const loom_source_range_t* range) {
  return range->provenance == LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE ||
         range->filename.size > 0 || range->start != 0 || range->end != 0 ||
         range->start_line != 0 || range->start_column != 0 ||
         range->end_line != 0 || range->end_column != 0;
}

static iree_host_size_t loom_json_find_line_start(iree_string_view_t source,
                                                  iree_host_size_t position) {
  position = iree_min(position, source.size);
  while (position > 0 && source.data[position - 1] != '\n') {
    --position;
  }
  return position;
}

static iree_host_size_t loom_json_find_line_end(iree_string_view_t source,
                                                iree_host_size_t position) {
  position = iree_min(position, source.size);
  while (position < source.size && source.data[position] != '\n') {
    ++position;
  }
  return position;
}

static iree_status_t loom_json_render_source_excerpt(
    loom_output_stream_t* stream, const loom_source_range_t* range) {
  iree_host_size_t position = iree_min(range->start, range->source.size);
  iree_host_size_t line_start =
      loom_json_find_line_start(range->source, position);
  iree_host_size_t line_end = loom_json_find_line_end(range->source, position);
  iree_host_size_t excerpt_start = line_start;
  iree_host_size_t excerpt_end = line_end;
  if (excerpt_end - excerpt_start > LOOM_JSON_SOURCE_EXCERPT_MAX_BYTES) {
    excerpt_start =
        position > line_start + LOOM_JSON_SOURCE_EXCERPT_CONTEXT_BYTES
            ? position - LOOM_JSON_SOURCE_EXCERPT_CONTEXT_BYTES
            : line_start;
    excerpt_end =
        iree_min(line_end, excerpt_start + LOOM_JSON_SOURCE_EXCERPT_MAX_BYTES);
    iree_host_size_t span_end = iree_min(
        range->end > range->start ? range->end : range->start, line_end);
    if (span_end > excerpt_end) {
      excerpt_end = span_end;
      if (excerpt_end - line_start > LOOM_JSON_SOURCE_EXCERPT_MAX_BYTES) {
        excerpt_start = excerpt_end - LOOM_JSON_SOURCE_EXCERPT_MAX_BYTES;
      } else {
        excerpt_start = line_start;
      }
    }
  }
  bool truncated_prefix = excerpt_start > line_start;
  bool truncated_suffix = excerpt_end < line_end;

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"excerpt\":{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      "\"start_byte\":%zu,\"end_byte\":%zu,"
      "\"truncated_prefix\":%s,\"truncated_suffix\":%s",
      (size_t)excerpt_start, (size_t)excerpt_end,
      truncated_prefix ? "true" : "false",
      truncated_suffix ? "true" : "false"));
  if (range->source.size > 0) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"text\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, iree_make_string_view(range->source.data + excerpt_start,
                                      excerpt_end - excerpt_start)));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '}'));
  return iree_ok_status();
}

// Renders one named source range object when the range has location metadata.
static iree_status_t loom_json_render_source_range(
    loom_output_stream_t* stream, const char* field_name,
    const loom_source_range_t* range) {
  if (!loom_json_source_range_has_metadata(range)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(stream, field_name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ":{"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "\"provenance\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, loom_source_provenance_name(range->provenance)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"filename\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, range->filename));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"start_line\":%" PRIu32 ",\"start_column\":%" PRIu32
      ",\"end_line\":%" PRIu32 ",\"end_column\":%" PRIu32
      ",\"start_byte\":%zu,\"end_byte\":%zu",
      range->start_line, range->start_column, range->end_line,
      range->end_column, (size_t)range->start, (size_t)range->end));
  IREE_RETURN_IF_ERROR(loom_json_render_source_excerpt(stream, range));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '}'));
  return iree_ok_status();
}

// Renders per-token highlight byte ranges when present. Primary diagnostic
// highlights include a param name when the highlight references a structured
// param; related-location highlights may omit that linkage.
static iree_status_t loom_json_render_highlights(
    loom_output_stream_t* stream, const char* field_name,
    const loom_highlight_range_t* highlights, iree_host_size_t highlight_count,
    const loom_error_def_t* error, iree_host_size_t param_count) {
  if (!highlights || highlight_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(stream, field_name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ":["));
  for (iree_host_size_t highlight_index = 0; highlight_index < highlight_count;
       ++highlight_index) {
    const loom_highlight_range_t* highlight = &highlights[highlight_index];
    if (highlight_index > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "{\"start_byte\":%zu,\"end_byte\":%zu",
        (size_t)highlight->start, (size_t)highlight->end));
    if (loom_diagnostic_field_ref_is_set(highlight->field_ref)) {
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, ",\"field\":"));
      IREE_RETURN_IF_ERROR(
          loom_json_render_field_ref(stream, highlight->field_ref));
      if (error && error->param_defs) {
        if (highlight->param_index >= param_count ||
            highlight->param_index >= error->param_count) {
          return iree_make_status(
              IREE_STATUS_INTERNAL,
              "diagnostic highlight %zu references invalid param index %zu",
              (size_t)highlight_index, (size_t)highlight->param_index);
        }
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(stream, ",\"param\":"));
        IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
            stream, error->param_defs[highlight->param_index].name));
      }
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '}'));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  return iree_ok_status();
}

static iree_status_t loom_json_render_related_locations(
    loom_output_stream_t* stream, const loom_diagnostic_t* diagnostic) {
  if (diagnostic->related_location_count == 0 &&
      diagnostic->related_location_omitted_count == 0) {
    return iree_ok_status();
  }
  if (diagnostic->related_location_count > 0 &&
      !diagnostic->related_locations) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "diagnostic related location count requires related locations");
  }

  if (diagnostic->related_location_count > 0) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"related_locations\":["));
    for (iree_host_size_t i = 0; i < diagnostic->related_location_count; ++i) {
      const loom_diagnostic_related_location_t* related =
          &diagnostic->related_locations[i];
      if (i > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
      }
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, "{\"label\":"));
      IREE_RETURN_IF_ERROR(
          loom_json_write_escaped_string(stream, related->label));
      IREE_RETURN_IF_ERROR(loom_json_render_source_range(
          stream, "source_location", &related->source_location));
      IREE_RETURN_IF_ERROR(
          loom_json_render_highlights(stream, "highlights", related->highlights,
                                      related->highlight_count, NULL, 0));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '}'));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  }
  if (diagnostic->related_location_omitted_count > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"related_location_omitted_count\":%" PRIhsz,
        diagnostic->related_location_omitted_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_json_render_highlight_omissions(
    loom_output_stream_t* stream, const loom_diagnostic_t* diagnostic) {
  if (diagnostic->highlight_omitted_count == 0) {
    return iree_ok_status();
  }
  return loom_output_stream_write_format(
      stream, ",\"highlight_omitted_count\":%" PRIhsz,
      diagnostic->highlight_omitted_count);
}

// Renders structured field refs attached to params.
static iree_status_t loom_json_render_param_fields(
    loom_output_stream_t* stream, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count) {
  if (!error || !error->param_defs || !params || param_count == 0) {
    return iree_ok_status();
  }
  bool has_field_refs = false;
  for (iree_host_size_t i = 0; i < param_count; ++i) {
    if (loom_diagnostic_field_ref_is_set(params[i].field_ref)) {
      has_field_refs = true;
      break;
    }
  }
  if (!has_field_refs) return iree_ok_status();

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"param_fields\":{"));
  bool first = true;
  for (iree_host_size_t i = 0; i < param_count; ++i) {
    if (!loom_diagnostic_field_ref_is_set(params[i].field_ref)) continue;
    if (!first) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    first = false;
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_cstring(stream, error->param_defs[i].name));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ':'));
    IREE_RETURN_IF_ERROR(
        loom_json_render_field_ref(stream, params[i].field_ref));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  return iree_ok_status();
}

iree_status_t loom_diagnostic_json_write_object(
    loom_output_stream_t* stream, const loom_diagnostic_t* diagnostic,
    loom_type_formatter_t type_formatter) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));

  // Severity (always present).
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "\"severity\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, loom_diagnostic_severity_name(diagnostic->severity)));

  const loom_error_def_t* error = diagnostic->error;

  // Stable symbolic error ID.
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"error_id\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_cstring(stream, error->error_id));

  // Domain.
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"domain\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, loom_error_domain_name(error->domain)));

  // Code.
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"code\":%" PRIu16, error->code));

  // One-line summary.
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"summary\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(stream, error->summary));

  // Emitter.
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"emitter\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, loom_emitter_name(diagnostic->emitter)));

  // Source locations and highlight byte ranges when present.
  IREE_RETURN_IF_ERROR(
      loom_json_render_source_range(stream, "origin", &diagnostic->origin));
  IREE_RETURN_IF_ERROR(loom_json_render_source_range(
      stream, "source_location", &diagnostic->source_location));
  IREE_RETURN_IF_ERROR(loom_json_render_highlights(
      stream, "highlights", diagnostic->highlights, diagnostic->highlight_count,
      diagnostic->error, diagnostic->param_count));
  IREE_RETURN_IF_ERROR(
      loom_json_render_highlight_omissions(stream, diagnostic));
  IREE_RETURN_IF_ERROR(loom_json_render_related_locations(stream, diagnostic));

  // Message: rendered from the error def's template and params, streamed
  // through the JSON-escaping adapter directly to the output. Zero allocs.
  {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"message\":\""));
    loom_json_escape_stream_t escape_data;
    loom_output_stream_t escape_stream;
    loom_json_escape_stream_init(stream, &escape_data, &escape_stream);
    IREE_RETURN_IF_ERROR(loom_diagnostic_render_message(
        error, diagnostic->params, diagnostic->param_count, type_formatter,
        &escape_stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  }

  // Fix hint (when present).
  if (error->fix_hint_template) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"fix_hint\":\""));
    loom_json_escape_stream_t escape_data;
    loom_output_stream_t escape_stream;
    loom_json_escape_stream_init(stream, &escape_data, &escape_stream);
    IREE_RETURN_IF_ERROR(loom_diagnostic_render_fix_hint(
        error, diagnostic->params, diagnostic->param_count, type_formatter,
        &escape_stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  }

  // Params object (when present). Clamp to the schema length to avoid
  // OOB reads on param_defs if a diagnostic carries more runtime params
  // than the error definition declares.
  iree_host_size_t emit_param_count =
      (diagnostic->params && error->param_defs)
          ? iree_min(diagnostic->param_count, error->param_count)
          : 0;
  if (emit_param_count > 0) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"params\":{"));
    for (iree_host_size_t i = 0; i < emit_param_count; ++i) {
      if (i > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
      }
      // Validate that the runtime param kind matches the schema.
      if (diagnostic->params[i].kind != error->param_defs[i].kind) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "diagnostic param %zu kind mismatch: runtime %d vs schema %d",
            (size_t)i, (int)diagnostic->params[i].kind,
            (int)error->param_defs[i].kind);
      }
      // Param name.
      IREE_RETURN_IF_ERROR(
          loom_json_write_escaped_cstring(stream, error->param_defs[i].name));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ":"));
      // Param value.
      IREE_RETURN_IF_ERROR(loom_json_render_param_value(
          &diagnostic->params[i], type_formatter, stream));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }
  IREE_RETURN_IF_ERROR(loom_json_render_param_fields(
      stream, error, diagnostic->params, emit_param_count));

  // Close object.
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));

  return iree_ok_status();
}

iree_status_t loom_diagnostic_json_sink(void* user_data,
                                        const loom_diagnostic_t* diagnostic) {
  loom_json_sink_options_t* options = (loom_json_sink_options_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_diagnostic_json_write_object(
      options->stream, diagnostic, options->type_formatter));
  return loom_output_stream_write_cstring(options->stream, "\n");
}
