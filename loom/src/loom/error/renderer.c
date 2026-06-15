// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/renderer.h"

#include <inttypes.h>
#include <string.h>

//===----------------------------------------------------------------------===//
// Template expansion
//===----------------------------------------------------------------------===//

// Finds the param_defs index for |name| in the error def's param schema.
// Returns -1 if not found. Linear scan is fine for diagnostic schemas.
static int loom_find_param_index(const loom_error_def_t* error,
                                 iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < error->param_count; ++i) {
    if (iree_string_view_equal(
            name, iree_make_cstring_view(error->param_defs[i].name))) {
      return (int)i;
    }
  }
  return -1;
}

static iree_status_t loom_render_string_list(
    loom_diagnostic_string_list_t string_list, loom_output_stream_t* stream) {
  if (string_list.count > 0 && !string_list.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
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

// Renders a single param value into the stream.
static iree_status_t loom_render_param(const loom_diagnostic_param_t* param,
                                       loom_type_formatter_t type_formatter,
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
      return loom_render_string_list(param->string_list, stream);
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

// Expands a message template, substituting {param_name} placeholders with
// rendered param values. Literal text between placeholders is copied verbatim.
static iree_status_t loom_expand_template(const char* message_template,
                                          const loom_error_def_t* error,
                                          const loom_diagnostic_param_t* params,
                                          iree_host_size_t param_count,
                                          loom_type_formatter_t type_formatter,
                                          loom_output_stream_t* stream) {
  const char* cursor = message_template;
  while (*cursor) {
    // Find next placeholder start.
    const char* open_brace = strchr(cursor, '{');
    if (!open_brace) {
      // No more placeholders — append remaining literal text.
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, cursor));
      break;
    }

    // Append literal text before the placeholder.
    if (open_brace > cursor) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write(
          stream, iree_make_string_view(
                      cursor, (iree_host_size_t)(open_brace - cursor))));
    }

    // Find the closing brace.
    const char* close_brace = strchr(open_brace + 1, '}');
    if (!close_brace) {
      // Malformed template — append the rest literally.
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, open_brace));
      break;
    }

    // Extract the placeholder name.
    iree_string_view_t placeholder_name = iree_make_string_view(
        open_brace + 1, (iree_host_size_t)(close_brace - open_brace - 1));

    // Look up the param by name in the error def's param schema.
    int param_index = loom_find_param_index(error, placeholder_name);
    if (param_index >= 0 && (iree_host_size_t)param_index < param_count) {
      if (params[param_index].kind != error->param_defs[param_index].kind) {
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "diagnostic param '%.*s' kind mismatch: runtime %d vs schema %d",
            (int)placeholder_name.size, placeholder_name.data,
            (int)params[param_index].kind,
            (int)error->param_defs[param_index].kind);
      }
      IREE_RETURN_IF_ERROR(
          loom_render_param(&params[param_index], type_formatter, stream));
    } else {
      // Unknown placeholder — emit it literally so the output is debuggable.
      IREE_RETURN_IF_ERROR(loom_output_stream_write(
          stream,
          iree_make_string_view(
              open_brace, (iree_host_size_t)(close_brace - open_brace + 1))));
    }

    cursor = close_brace + 1;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t loom_diagnostic_render_message(
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count, loom_type_formatter_t type_formatter,
    loom_output_stream_t* stream) {
  if (!error || !error->message_template) {
    return iree_ok_status();
  }
  if (param_count > 0 && !params) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "param_count > 0 but params is NULL");
  }
  return loom_expand_template(error->message_template, error, params,
                              param_count, type_formatter, stream);
}

iree_status_t loom_diagnostic_render_fix_hint(
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count, loom_type_formatter_t type_formatter,
    loom_output_stream_t* stream) {
  if (!error || !error->fix_hint_template) {
    return iree_ok_status();
  }
  if (param_count > 0 && !params) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "param_count > 0 but params is NULL");
  }
  return loom_expand_template(error->fix_hint_template, error, params,
                              param_count, type_formatter, stream);
}

//===----------------------------------------------------------------------===//
// Minimal type formatter
//===----------------------------------------------------------------------===//

iree_status_t loom_type_format_minimal(loom_type_t type, void* user_data,
                                       loom_output_stream_t* stream) {
  (void)user_data;
  loom_type_kind_t kind = loom_type_kind(type);
  switch (kind) {
    case LOOM_TYPE_SCALAR: {
      const char* name = loom_scalar_type_name(loom_type_element_type(type));
      if (name) {
        return loom_output_stream_write_cstring(stream, name);
      }
      return loom_output_stream_write_cstring(stream, "<scalar>");
    }
    case LOOM_TYPE_TILE:
      return loom_output_stream_write_cstring(stream, "tile<...>");
    case LOOM_TYPE_TENSOR:
      return loom_output_stream_write_cstring(stream, "tensor<...>");
    case LOOM_TYPE_VECTOR:
      return loom_output_stream_write_cstring(stream, "vector<...>");
    case LOOM_TYPE_VIEW:
      return loom_output_stream_write_cstring(stream, "view<...>");
    case LOOM_TYPE_BUFFER:
      return loom_output_stream_write_cstring(stream, "buffer");
    case LOOM_TYPE_REGISTER:
      return loom_output_stream_write_cstring(stream, "reg<...>");
    case LOOM_TYPE_STORAGE:
      return loom_output_stream_write_cstring(stream, "low.storage<...>");
    case LOOM_TYPE_GROUP:
      return loom_output_stream_write_cstring(stream, "group<...>");
    case LOOM_TYPE_FUNCTION:
      return loom_output_stream_write_cstring(stream, "(...) -> (...)");
    case LOOM_TYPE_DIALECT:
      return loom_output_stream_write_cstring(stream, "dialect<...>");
    case LOOM_TYPE_ENCODING: {
      loom_encoding_role_t role = loom_type_encoding_role(type);
      if (role == LOOM_ENCODING_ROLE_UNKNOWN) {
        return loom_output_stream_write_cstring(stream, "encoding");
      }
      const char* role_name = loom_encoding_role_name(role);
      if (role_name) {
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(stream, "encoding<"));
        IREE_RETURN_IF_ERROR(
            loom_output_stream_write_cstring(stream, role_name));
        return loom_output_stream_write_cstring(stream, ">");
      }
      return loom_output_stream_write_cstring(stream, "encoding<?>");
    }
    case LOOM_TYPE_POOL:
      return loom_output_stream_write_cstring(stream, "pool<...>");
    case LOOM_TYPE_NONE:
      return loom_output_stream_write_cstring(stream, "<none>");
    default:
      return loom_output_stream_write_format(stream, "<type:%d>", (int)kind);
  }
}
