// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <string.h>

#include "iree/io/file_contents.h"
#include "iree/io/stdio_stream.h"
#include "iree/io/stream.h"
#include "loom/format/bytecode/format.h"
#include "loomc/iree.h"
#include "module.h"
#include "source.h"

static bool loomc_module_source_has_bytecode_magic(loomc_byte_span_t contents) {
  return contents.data_length >= LOOM_BYTECODE_MAGIC_LENGTH &&
         memcmp(contents.data, LOOM_BYTECODE_MAGIC,
                LOOM_BYTECODE_MAGIC_LENGTH) == 0;
}

static loomc_status_t loomc_module_select_deserialize_format(
    const loomc_module_deserialize_options_t* options,
    loomc_source_format_t fallback_format, loomc_byte_span_t contents,
    loomc_source_format_t* out_format) {
  LOOMC_RETURN_IF_ERROR(loomc_module_validate_deserialize_options(options));
  loomc_source_format_t format =
      options ? options->format : LOOMC_SOURCE_FORMAT_UNKNOWN;
  if (format == LOOMC_SOURCE_FORMAT_UNKNOWN) {
    format = fallback_format;
  }
  if (format == LOOMC_SOURCE_FORMAT_UNKNOWN) {
    format = loomc_module_source_has_bytecode_magic(contents)
                 ? LOOMC_SOURCE_FORMAT_BYTECODE
                 : LOOMC_SOURCE_FORMAT_TEXT;
  }
  if (format != LOOMC_SOURCE_FORMAT_TEXT &&
      format != LOOMC_SOURCE_FORMAT_BYTECODE) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "module deserialize format must be text or bytecode");
  }
  *out_format = format;
  return loomc_ok_status();
}

static loomc_status_t loomc_module_deserialize_selected_source(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_source_t* source,
    const loomc_module_deserialize_options_t* options,
    loomc_source_format_t format, loomc_allocator_t allocator,
    loomc_module_t** out_module, loomc_result_t** out_result) {
  switch (format) {
    case LOOMC_SOURCE_FORMAT_TEXT:
      return loomc_module_deserialize_text_from_source(
          context, workspace, source, options, allocator, out_module,
          out_result);
    case LOOMC_SOURCE_FORMAT_BYTECODE:
      return loomc_module_deserialize_bytecode_from_source(
          context, workspace, source, options, allocator, out_module,
          out_result);
    default:
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "selected module format is invalid");
  }
}

static loomc_string_view_t loomc_module_source_identifier(
    const loomc_module_deserialize_options_t* options,
    loomc_string_view_t fallback_identifier) {
  if (options != NULL && !loomc_string_view_is_empty(options->identifier)) {
    return options->identifier;
  }
  return fallback_identifier;
}

static void loomc_module_source_release_file_contents(
    void* user_data, loomc_byte_span_t contents) {
  (void)contents;
  iree_io_file_contents_free((iree_io_file_contents_t*)user_data);
}

loomc_status_t loomc_module_deserialize_from_source(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_source_t* source,
    const loomc_module_deserialize_options_t* options,
    loomc_allocator_t allocator, loomc_module_t** out_module,
    loomc_result_t** out_result) {
  LOOMC_RETURN_IF_ERROR(loomc_module_validate_deserialize_source_arguments(
      context, workspace, source, out_module, out_result));
  loomc_source_format_t format = LOOMC_SOURCE_FORMAT_UNKNOWN;
  LOOMC_RETURN_IF_ERROR(loomc_module_select_deserialize_format(
      options, loomc_source_format(source), loomc_source_contents(source),
      &format));
  return loomc_module_deserialize_selected_source(context, workspace, source,
                                                  options, format, allocator,
                                                  out_module, out_result);
}

loomc_status_t loomc_module_deserialize_from_file(
    loomc_context_t* context, loomc_workspace_t* workspace, FILE* file,
    const loomc_module_deserialize_options_t* options,
    loomc_allocator_t allocator, loomc_module_t** out_module,
    loomc_result_t** out_result) {
  if (out_module != NULL) {
    *out_module = NULL;
  }
  if (out_result != NULL) {
    *out_result = NULL;
  }
  if (context == NULL || workspace == NULL || file == NULL ||
      out_module == NULL || out_result == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "context, workspace, file, out_module, and out_result must not be "
        "NULL");
  }

  uint8_t* storage = NULL;
  loomc_host_size_t storage_length = 0;
  LOOMC_RETURN_IF_ERROR(loomc_source_read_file_to_storage(
      file, allocator, &storage, &storage_length));
  const loomc_byte_span_t contents =
      loomc_make_byte_span(storage, storage_length);
  loomc_source_format_t format = LOOMC_SOURCE_FORMAT_UNKNOWN;
  loomc_status_t status = loomc_module_select_deserialize_format(
      options, LOOMC_SOURCE_FORMAT_UNKNOWN, contents, &format);
  loomc_source_t* source = NULL;
  if (loomc_status_is_ok(status)) {
    status = loomc_source_create_take_contents(
        format,
        loomc_module_source_identifier(options, loomc_string_view_empty()),
        contents, allocator, &source);
    if (loomc_status_is_ok(status)) {
      storage = NULL;
    }
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_module_deserialize_selected_source(
        context, workspace, source, options, format, allocator, out_module,
        out_result);
  }
  loomc_source_release(source);
  loomc_allocator_free(allocator, storage);
  return status;
}

loomc_status_t loomc_module_deserialize_from_path(
    loomc_context_t* context, loomc_workspace_t* workspace,
    loomc_string_view_t path, const loomc_module_deserialize_options_t* options,
    loomc_allocator_t allocator, loomc_module_t** out_module,
    loomc_result_t** out_result) {
  if (out_module != NULL) {
    *out_module = NULL;
  }
  if (out_result != NULL) {
    *out_result = NULL;
  }
  if (context == NULL || workspace == NULL ||
      loomc_string_view_is_empty(path) || out_module == NULL ||
      out_result == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "context, workspace, path, out_module, and out_result must be valid");
  }
  if (!loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "allocator.ctl must not be NULL");
  }

  iree_io_file_contents_t* file_contents = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(iree_io_file_contents_read(
      iree_string_view_from_loomc(path), iree_allocator_from_loomc(allocator),
      &file_contents)));
  const loomc_byte_span_t contents =
      loomc_byte_span_from_iree(file_contents->const_buffer);
  loomc_source_format_t format = LOOMC_SOURCE_FORMAT_UNKNOWN;
  loomc_status_t status = loomc_module_select_deserialize_format(
      options, LOOMC_SOURCE_FORMAT_UNKNOWN, contents, &format);
  loomc_source_t* source = NULL;
  if (loomc_status_is_ok(status)) {
    const loomc_source_options_t source_options = {
        .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        .structure_size = sizeof(source_options),
        .format = format,
        .identifier = loomc_module_source_identifier(options, path),
        .contents = contents,
        .storage = LOOMC_SOURCE_STORAGE_EXTERNAL,
        .release = loomc_module_source_release_file_contents,
        .release_user_data = file_contents,
    };
    status = loomc_source_create(&source_options, allocator, &source);
    if (loomc_status_is_ok(status)) {
      file_contents = NULL;
    }
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_module_deserialize_selected_source(
        context, workspace, source, options, format, allocator, out_module,
        out_result);
  }
  loomc_source_release(source);
  iree_io_file_contents_free(file_contents);
  return status;
}

static loomc_status_t loomc_module_select_serialize_format(
    const loomc_module_serialize_options_t* options,
    loomc_source_format_t* out_format) {
  LOOMC_RETURN_IF_ERROR(loomc_module_validate_serialize_options(options));
  loomc_source_format_t format =
      options ? options->format : LOOMC_SOURCE_FORMAT_UNKNOWN;
  if (format == LOOMC_SOURCE_FORMAT_UNKNOWN) {
    format = LOOMC_SOURCE_FORMAT_TEXT;
  }
  if (format != LOOMC_SOURCE_FORMAT_TEXT &&
      format != LOOMC_SOURCE_FORMAT_BYTECODE) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "module serialize format must be text or bytecode");
  }
  *out_format = format;
  return loomc_ok_status();
}

static loomc_status_t loomc_module_serialize_selected_source(
    const loomc_module_t* module,
    const loomc_module_serialize_options_t* options,
    loomc_source_format_t format, loomc_allocator_t allocator,
    loomc_source_t** out_source) {
  switch (format) {
    case LOOMC_SOURCE_FORMAT_TEXT:
      return loomc_module_serialize_text_to_source(module, options, allocator,
                                                   out_source);
    case LOOMC_SOURCE_FORMAT_BYTECODE:
      return loomc_module_serialize_bytecode_to_source(module, options,
                                                       allocator, out_source);
    default:
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "selected module format is invalid");
  }
}

loomc_status_t loomc_module_serialize_to_source(
    const loomc_module_t* module,
    const loomc_module_serialize_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source) {
  if (out_source != NULL) {
    *out_source = NULL;
  }
  loomc_source_format_t format = LOOMC_SOURCE_FORMAT_UNKNOWN;
  LOOMC_RETURN_IF_ERROR(loomc_module_select_serialize_format(options, &format));
  return loomc_module_serialize_selected_source(module, options, format,
                                                allocator, out_source);
}

loomc_status_t loomc_module_serialize_to_file(
    const loomc_module_t* module,
    const loomc_module_serialize_options_t* options, FILE* file) {
  if (module == NULL || file == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module and file must not be NULL");
  }
  loomc_source_t* source = NULL;
  loomc_status_t status = loomc_module_serialize_to_source(
      module, options, loomc_module_allocator(module), &source);
  if (loomc_status_is_ok(status)) {
    const loomc_byte_span_t contents = loomc_source_contents(source);
    if (contents.data_length != 0 &&
        fwrite(contents.data, 1, contents.data_length, file) !=
            contents.data_length) {
      status = loomc_make_status(LOOMC_STATUS_UNKNOWN,
                                 "failed to write serialized module");
    }
  }
  loomc_source_release(source);
  return status;
}

loomc_status_t loomc_module_serialize_to_path(
    const loomc_module_t* module,
    const loomc_module_serialize_options_t* options, loomc_string_view_t path,
    loomc_allocator_t allocator) {
  if (module == NULL || loomc_string_view_is_empty(path)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module must not be NULL and path must not be "
                             "empty");
  }
  if (!loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "allocator.ctl must not be NULL");
  }

  loomc_source_t* source = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_module_serialize_to_source(module, options, allocator, &source));
  iree_io_stream_t* stream = NULL;
  loomc_status_t status = loomc_status_from_iree(iree_io_stdio_stream_open(
      IREE_IO_STDIO_STREAM_MODE_WRITE | IREE_IO_STDIO_STREAM_MODE_DISCARD,
      iree_string_view_from_loomc(path), iree_allocator_from_loomc(allocator),
      &stream));
  if (loomc_status_is_ok(status)) {
    const loomc_byte_span_t contents = loomc_source_contents(source);
    status = loomc_status_from_iree(
        iree_io_stream_write(stream, contents.data_length, contents.data));
  }
  iree_io_stream_release(stream);
  loomc_source_release(source);
  return status;
}
