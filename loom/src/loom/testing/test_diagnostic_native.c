// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/testing/test_diagnostic_native.h"

#include <string.h>

#include "loom/error/renderer.h"
#include "loom/util/stream.h"

static iree_status_t loom_test_diagnostic_copy_string(
    iree_arena_allocator_t* arena, iree_string_view_t source,
    iree_string_view_t* out_copy) {
  *out_copy = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) {
    return iree_ok_status();
  }
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

static iree_status_t loom_test_diagnostic_copy_params(
    const loom_diagnostic_t* diagnostic, loom_type_formatter_t type_formatter,
    iree_arena_allocator_t* arena, iree_allocator_t host_allocator,
    loom_test_diagnostic_param_t** out_params, iree_host_size_t* out_count) {
  *out_params = NULL;
  *out_count = 0;
  if (!diagnostic->params || !diagnostic->error ||
      diagnostic->error->param_count == 0 || diagnostic->param_count == 0) {
    return iree_ok_status();
  }

  const iree_host_size_t param_count =
      iree_min(diagnostic->param_count, diagnostic->error->param_count);
  loom_test_diagnostic_param_t* params = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, param_count, sizeof(*params), (void**)&params));

  for (iree_host_size_t i = 0; i < param_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_test_diagnostic_copy_string(
        arena,
        iree_make_cstring_view(loom_error_def_param_name(diagnostic->error, i)),
        &params[i].name));
    iree_string_builder_t builder;
    iree_string_builder_initialize(host_allocator, &builder);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&builder, &stream);
    iree_status_t status = loom_diagnostic_render_param_value(
        &diagnostic->params[i], type_formatter, &stream);
    if (iree_status_is_ok(status)) {
      status = loom_test_diagnostic_copy_string(
          arena, iree_string_builder_view(&builder), &params[i].value);
    }
    iree_string_builder_deinitialize(&builder);
    IREE_RETURN_IF_ERROR(status);
  }

  *out_params = params;
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
    status = loom_test_diagnostic_copy_params(
        diagnostic, type_formatter, arena, host_allocator,
        &out_diagnostic->params, &out_diagnostic->param_count);
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
  IREE_RETURN_IF_ERROR(loom_test_diagnostic_copy_string(
      arena, diagnostic->origin.filename, &out_diagnostic->origin.filename));
  out_diagnostic->origin.line = diagnostic->origin.start_line;
  return iree_ok_status();
}
