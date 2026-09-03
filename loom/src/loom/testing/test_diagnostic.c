// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/testing/test_diagnostic.h"

#include <inttypes.h>
#include <string.h>

#include "loom/error/renderer.h"
#include "loom/util/stream.h"

static iree_status_t loom_test_diagnostic_copy_string(
    iree_arena_allocator_t* arena, iree_string_view_t source,
    iree_string_view_t* out_copy) {
  *out_copy = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) return iree_ok_status();
  char* target = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, source.size, (void**)&target));
  memcpy(target, source.data, source.size);
  *out_copy = iree_make_string_view(target, source.size);
  return iree_ok_status();
}

static iree_status_t loom_test_diagnostic_format_type(
    loom_type_t type, void* user_data, loom_output_stream_t* stream) {
  const loom_test_diagnostic_format_options_t* options =
      (const loom_test_diagnostic_format_options_t*)user_data;
  if (!options || !options->module) {
    return loom_type_format_minimal(type, NULL, stream);
  }
  return loom_text_print_type_with_options(type, options->module, stream,
                                           &options->text_print_options);
}

static iree_status_t loom_test_diagnostic_render_string_list(
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

static iree_status_t loom_test_diagnostic_render_param(
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
      return loom_test_diagnostic_render_string_list(param->string_list,
                                                     stream);
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

static iree_status_t loom_test_diagnostic_copy_param_values(
    const loom_diagnostic_t* diagnostic, loom_type_formatter_t type_formatter,
    iree_arena_allocator_t* arena, iree_allocator_t host_allocator,
    iree_string_view_t** out_values, iree_host_size_t* out_count) {
  *out_values = NULL;
  *out_count = 0;
  if (!diagnostic->params || !diagnostic->error ||
      diagnostic->error->param_count == 0 || diagnostic->param_count == 0) {
    return iree_ok_status();
  }

  const iree_host_size_t param_count =
      iree_min(diagnostic->param_count, diagnostic->error->param_count);
  iree_string_view_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, param_count, sizeof(*values), (void**)&values));

  for (iree_host_size_t i = 0; i < param_count; ++i) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(host_allocator, &builder);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&builder, &stream);
    iree_status_t status = loom_test_diagnostic_render_param(
        &diagnostic->params[i], type_formatter, &stream);
    if (iree_status_is_ok(status)) {
      status = loom_test_diagnostic_copy_string(
          arena, iree_string_builder_view(&builder), &values[i]);
    }
    iree_string_builder_deinitialize(&builder);
    IREE_RETURN_IF_ERROR(status);
  }

  *out_values = values;
  *out_count = param_count;
  return iree_ok_status();
}

iree_status_t loom_test_diagnostic_materialize(
    const loom_diagnostic_t* diagnostic,
    const loom_test_diagnostic_format_options_t* options,
    iree_arena_allocator_t* arena, iree_allocator_t host_allocator,
    loom_test_diagnostic_t* out_diagnostic) {
  *out_diagnostic = (loom_test_diagnostic_t){0};
  loom_type_formatter_t type_formatter = {
      .fn = loom_test_diagnostic_format_type,
      .user_data = (void*)options,
  };

  iree_string_builder_t message_builder;
  iree_string_builder_initialize(host_allocator, &message_builder);
  loom_output_stream_t message_stream;
  loom_output_stream_for_builder(&message_builder, &message_stream);
  iree_status_t status = loom_diagnostic_render_message(
      diagnostic->error, diagnostic->params, diagnostic->param_count,
      type_formatter, &message_stream);
  if (iree_status_is_ok(status)) {
    status = loom_test_diagnostic_copy_string(
        arena, iree_string_builder_view(&message_builder),
        &out_diagnostic->message);
  }
  iree_string_builder_deinitialize(&message_builder);

  if (iree_status_is_ok(status)) {
    status = loom_test_diagnostic_copy_param_values(
        diagnostic, type_formatter, arena, host_allocator,
        &out_diagnostic->param_values, &out_diagnostic->param_value_count);
  }

  iree_string_builder_t formatted_builder;
  iree_string_builder_initialize(host_allocator, &formatted_builder);
  if (iree_status_is_ok(status)) {
    loom_output_stream_t formatted_stream;
    loom_output_stream_for_builder(&formatted_builder, &formatted_stream);
    const loom_diagnostic_format_options_t format_options = {
        .type_formatter = type_formatter,
    };
    status = loom_diagnostic_format_with_options(diagnostic, &format_options,
                                                 &formatted_stream);
  }
  if (iree_status_is_ok(status)) {
    status = loom_test_diagnostic_copy_string(
        arena, iree_string_builder_view(&formatted_builder),
        &out_diagnostic->formatted_diagnostic);
  }
  iree_string_builder_deinitialize(&formatted_builder);
  IREE_RETURN_IF_ERROR(status);

  out_diagnostic->severity = diagnostic->severity;
  out_diagnostic->domain = loom_error_def_domain(diagnostic->error);
  out_diagnostic->code = loom_error_def_code(diagnostic->error);
  out_diagnostic->error = diagnostic->error;
  out_diagnostic->origin_line = diagnostic->origin.start_line;
  return iree_ok_status();
}

static int loom_test_diagnostic_find_param_index(
    const loom_test_diagnostic_t* diagnostic, iree_string_view_t name) {
  if (!diagnostic->error || diagnostic->error->param_count == 0) return -1;
  for (iree_host_size_t i = 0; i < diagnostic->param_value_count; ++i) {
    if (i >= diagnostic->error->param_count) return -1;
    if (iree_string_view_equal(
            name, iree_make_cstring_view(
                      loom_error_def_param_name(diagnostic->error, i)))) {
      return (int)i;
    }
  }
  return -1;
}

bool loom_test_diagnostic_matches_annotation(
    const loom_test_diagnostic_t* diagnostic,
    const loom_test_annotation_t* annotation) {
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
  for (uint8_t i = 0; i < annotation->param_match_count; ++i) {
    const loom_test_annotation_param_match_t* match =
        &annotation->param_matches[i];
    const int param_index =
        loom_test_diagnostic_find_param_index(diagnostic, match->name);
    if (param_index < 0 ||
        !iree_string_view_equal(diagnostic->param_values[param_index],
                                match->value)) {
      return false;
    }
  }
  return true;
}

typedef struct loom_test_diagnostic_match_context_t {
  loom_test_diagnostic_t* diagnostics;
  iree_host_size_t diagnostic_count;
  const loom_test_annotation_t* annotations;
  iree_host_size_t annotation_count;
  iree_host_size_t* annotation_to_diagnostic;
  bool* visited_annotations;
} loom_test_diagnostic_match_context_t;

static bool loom_test_diagnostic_match_augment(
    loom_test_diagnostic_match_context_t* context,
    iree_host_size_t diagnostic_index) {
  for (iree_host_size_t annotation_index = 0;
       annotation_index < context->annotation_count; ++annotation_index) {
    if (context->visited_annotations[annotation_index] ||
        !loom_test_diagnostic_matches_annotation(
            &context->diagnostics[diagnostic_index],
            &context->annotations[annotation_index])) {
      continue;
    }
    context->visited_annotations[annotation_index] = true;
    const iree_host_size_t previous_diagnostic =
        context->annotation_to_diagnostic[annotation_index];
    if (previous_diagnostic == IREE_HOST_SIZE_MAX ||
        loom_test_diagnostic_match_augment(context, previous_diagnostic)) {
      context->annotation_to_diagnostic[annotation_index] = diagnostic_index;
      return true;
    }
  }
  return false;
}

iree_status_t loom_test_diagnostics_match_annotations(
    loom_test_diagnostic_t* diagnostics, iree_host_size_t diagnostic_count,
    const loom_test_annotation_t* annotations,
    iree_host_size_t annotation_count, iree_arena_allocator_t* arena,
    iree_host_size_t** out_annotation_to_diagnostic) {
  *out_annotation_to_diagnostic = NULL;
  for (iree_host_size_t i = 0; i < diagnostic_count; ++i) {
    diagnostics[i].matched = false;
  }
  if (annotation_count == 0) return iree_ok_status();

  iree_host_size_t* annotation_to_diagnostic = NULL;
  bool* visited_annotations = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, annotation_count, sizeof(*annotation_to_diagnostic),
      (void**)&annotation_to_diagnostic));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, annotation_count,
                                                 sizeof(*visited_annotations),
                                                 (void**)&visited_annotations));
  for (iree_host_size_t i = 0; i < annotation_count; ++i) {
    annotation_to_diagnostic[i] = IREE_HOST_SIZE_MAX;
  }

  loom_test_diagnostic_match_context_t context = {
      .diagnostics = diagnostics,
      .diagnostic_count = diagnostic_count,
      .annotations = annotations,
      .annotation_count = annotation_count,
      .annotation_to_diagnostic = annotation_to_diagnostic,
      .visited_annotations = visited_annotations,
  };
  for (iree_host_size_t diagnostic_index = 0;
       diagnostic_index < diagnostic_count; ++diagnostic_index) {
    memset(visited_annotations, 0,
           annotation_count * sizeof(*visited_annotations));
    loom_test_diagnostic_match_augment(&context, diagnostic_index);
  }
  for (iree_host_size_t annotation_index = 0;
       annotation_index < annotation_count; ++annotation_index) {
    const iree_host_size_t diagnostic_index =
        annotation_to_diagnostic[annotation_index];
    if (diagnostic_index != IREE_HOST_SIZE_MAX) {
      diagnostics[diagnostic_index].matched = true;
    }
  }

  *out_annotation_to_diagnostic = annotation_to_diagnostic;
  return iree_ok_status();
}
