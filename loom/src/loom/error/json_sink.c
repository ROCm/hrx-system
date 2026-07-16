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
      loom_json_array_writer_t array;
      IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
      for (iree_host_size_t i = 0; i < param->string_list.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_json_array_write_string_element(
            &array, param->string_list.values[i]));
      }
      return loom_json_array_end(&array);
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
    IREE_ASSERT_UNREACHABLE("invalid diagnostic field ref kind");
    IREE_BUILTIN_UNREACHABLE();
  }
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("kind"), iree_make_cstring_view(kind_name)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("index"), field_ref.index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("occurrence"), field_ref.occurrence));
  return loom_json_object_end(&object);
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
    const loom_source_range_t* range, loom_json_object_writer_t* object) {
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
      loom_json_object_begin_field(object, IREE_SV("excerpt")));
  loom_json_object_writer_t excerpt;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(object->stream, &excerpt));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &excerpt, IREE_SV("start_byte"), excerpt_start));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &excerpt, IREE_SV("end_byte"), excerpt_end));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &excerpt, IREE_SV("truncated_prefix"), truncated_prefix));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &excerpt, IREE_SV("truncated_suffix"), truncated_suffix));
  if (range->source.size > 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &excerpt, IREE_SV("text"),
        iree_make_string_view(range->source.data + excerpt_start,
                              excerpt_end - excerpt_start)));
  }
  return loom_json_object_end(&excerpt);
}

// Renders one named source range object when the range has location metadata.
static iree_status_t loom_json_render_source_range(
    loom_json_object_writer_t* object, iree_string_view_t field_name,
    const loom_source_range_t* range) {
  if (!loom_json_source_range_has_metadata(range)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(object, field_name));
  loom_json_object_writer_t range_object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(object->stream, &range_object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &range_object, IREE_SV("provenance"),
      iree_make_cstring_view(loom_source_provenance_name(range->provenance))));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &range_object, IREE_SV("filename"), range->filename));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &range_object, IREE_SV("start_line"), range->start_line));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &range_object, IREE_SV("start_column"), range->start_column));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &range_object, IREE_SV("end_line"), range->end_line));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &range_object, IREE_SV("end_column"), range->end_column));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &range_object, IREE_SV("start_byte"), range->start));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &range_object, IREE_SV("end_byte"), range->end));
  IREE_RETURN_IF_ERROR(loom_json_render_source_excerpt(range, &range_object));
  return loom_json_object_end(&range_object);
}

// Renders per-token highlight byte ranges when present. Primary diagnostic
// highlights include a param name when the highlight references a structured
// param; related-location highlights may omit that linkage.
static iree_status_t loom_json_render_highlights(
    loom_json_object_writer_t* object, iree_string_view_t field_name,
    const loom_highlight_range_t* highlights, iree_host_size_t highlight_count,
    const loom_error_def_t* error, iree_host_size_t param_count) {
  if (!highlights || highlight_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(object, field_name));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(object->stream, &array));
  for (iree_host_size_t highlight_index = 0; highlight_index < highlight_count;
       ++highlight_index) {
    const loom_highlight_range_t* highlight = &highlights[highlight_index];
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    loom_json_object_writer_t highlight_object;
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin(object->stream, &highlight_object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &highlight_object, IREE_SV("start_byte"), highlight->start));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &highlight_object, IREE_SV("end_byte"), highlight->end));
    if (loom_diagnostic_field_ref_is_set(highlight->field_ref)) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&highlight_object, IREE_SV("field")));
      IREE_RETURN_IF_ERROR(
          loom_json_render_field_ref(object->stream, highlight->field_ref));
      if (error && error->param_defs) {
        if (highlight->param_index >= param_count ||
            highlight->param_index >= error->param_count) {
          IREE_ASSERT_UNREACHABLE(
              "diagnostic highlight references invalid param index");
          IREE_BUILTIN_UNREACHABLE();
        }
        IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
            &highlight_object, IREE_SV("param"),
            iree_make_cstring_view(
                error->param_defs[highlight->param_index].name)));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&highlight_object));
  }
  return loom_json_array_end(&array);
}

static iree_status_t loom_json_render_related_locations(
    loom_json_object_writer_t* object, const loom_diagnostic_t* diagnostic) {
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
        loom_json_object_begin_field(object, IREE_SV("related_locations")));
    loom_json_array_writer_t related_locations;
    IREE_RETURN_IF_ERROR(
        loom_json_array_begin(object->stream, &related_locations));
    for (iree_host_size_t i = 0; i < diagnostic->related_location_count; ++i) {
      const loom_diagnostic_related_location_t* related =
          &diagnostic->related_locations[i];
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&related_locations));
      loom_json_object_writer_t related_object;
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin(object->stream, &related_object));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &related_object, IREE_SV("label"), related->label));
      IREE_RETURN_IF_ERROR(loom_json_render_source_range(
          &related_object, IREE_SV("source_location"),
          &related->source_location));
      IREE_RETURN_IF_ERROR(loom_json_render_highlights(
          &related_object, IREE_SV("highlights"), related->highlights,
          related->highlight_count, NULL, 0));
      IREE_RETURN_IF_ERROR(loom_json_object_end(&related_object));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&related_locations));
  }
  if (diagnostic->related_location_omitted_count > 0) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        object, IREE_SV("related_location_omitted_count"),
        diagnostic->related_location_omitted_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_json_render_highlight_omissions(
    loom_json_object_writer_t* object, const loom_diagnostic_t* diagnostic) {
  if (diagnostic->highlight_omitted_count == 0) {
    return iree_ok_status();
  }
  return loom_json_object_write_host_size_field(
      object, IREE_SV("highlight_omitted_count"),
      diagnostic->highlight_omitted_count);
}

// Renders structured field refs attached to params.
static iree_status_t loom_json_render_param_fields(
    loom_json_object_writer_t* object, const loom_error_def_t* error,
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
      loom_json_object_begin_field(object, IREE_SV("param_fields")));
  loom_json_object_writer_t param_fields;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(object->stream, &param_fields));
  for (iree_host_size_t i = 0; i < param_count; ++i) {
    if (!loom_diagnostic_field_ref_is_set(params[i].field_ref)) continue;
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &param_fields, iree_make_cstring_view(error->param_defs[i].name)));
    IREE_RETURN_IF_ERROR(
        loom_json_render_field_ref(object->stream, params[i].field_ref));
  }
  return loom_json_object_end(&param_fields);
}

iree_status_t loom_diagnostic_json_write_object(
    loom_output_stream_t* stream, const loom_diagnostic_t* diagnostic,
    loom_type_formatter_t type_formatter) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));

  // Severity (always present).
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("severity"),
      iree_make_cstring_view(
          loom_diagnostic_severity_name(diagnostic->severity))));

  const loom_error_def_t* error = diagnostic->error;

  // Stable symbolic error ID.
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("error_id"), iree_make_cstring_view(error->error_id)));

  // Domain.
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("domain"),
      iree_make_cstring_view(loom_error_domain_name(error->domain))));

  // Code.
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("code"), error->code));

  // One-line summary.
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("summary"), iree_make_cstring_view(error->summary)));

  // Emitter.
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("emitter"),
      iree_make_cstring_view(loom_emitter_name(diagnostic->emitter))));

  // Source locations and highlight byte ranges when present.
  IREE_RETURN_IF_ERROR(loom_json_render_source_range(&object, IREE_SV("origin"),
                                                     &diagnostic->origin));
  IREE_RETURN_IF_ERROR(loom_json_render_source_range(
      &object, IREE_SV("source_location"), &diagnostic->source_location));
  IREE_RETURN_IF_ERROR(loom_json_render_highlights(
      &object, IREE_SV("highlights"), diagnostic->highlights,
      diagnostic->highlight_count, diagnostic->error, diagnostic->param_count));
  IREE_RETURN_IF_ERROR(
      loom_json_render_highlight_omissions(&object, diagnostic));
  IREE_RETURN_IF_ERROR(loom_json_render_related_locations(&object, diagnostic));

  // Message: rendered from the error def's template and params, streamed
  // through the JSON-escaping adapter directly to the output. Zero allocs.
  {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("message")));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
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
        loom_json_object_begin_field(&object, IREE_SV("fix_hint")));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
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
        loom_json_object_begin_field(&object, IREE_SV("params")));
    loom_json_object_writer_t params;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &params));
    for (iree_host_size_t i = 0; i < emit_param_count; ++i) {
      // Validate that the runtime param kind matches the schema.
      if (diagnostic->params[i].kind != error->param_defs[i].kind) {
        IREE_ASSERT_UNREACHABLE("diagnostic param kind does not match schema");
        IREE_BUILTIN_UNREACHABLE();
      }
      // Param name.
      IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
          &params, iree_make_cstring_view(error->param_defs[i].name)));
      // Param value.
      IREE_RETURN_IF_ERROR(loom_json_render_param_value(
          &diagnostic->params[i], type_formatter, stream));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&params));
  }
  IREE_RETURN_IF_ERROR(loom_json_render_param_fields(
      &object, error, diagnostic->params, emit_param_count));

  // Close object.
  return loom_json_object_end(&object);
}

iree_status_t loom_diagnostic_json_sink(void* user_data,
                                        const loom_diagnostic_t* diagnostic) {
  loom_json_sink_options_t* options = (loom_json_sink_options_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_diagnostic_json_write_object(
      options->stream, diagnostic, options->type_formatter));
  return loom_output_stream_write_cstring(options->stream, "\n");
}
