// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/artifact.h"

#include <stdio.h>

#include "iree/io/stdio_stream.h"
#include "iree/io/stream.h"
#include "loomc/iree.h"

static loomc_status_t loomc_artifact_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_artifact_validate_contents(
    loomc_byte_span_t contents) {
  if (contents.data == NULL && contents.data_length != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact contents have length but no data");
  }
  return loomc_ok_status();
}

static loomc_source_format_t loomc_artifact_infer_source_format(
    const loomc_artifact_t* artifact, loomc_source_format_t fallback_format) {
  if (fallback_format != LOOMC_SOURCE_FORMAT_UNKNOWN) {
    return fallback_format;
  }
  if (loomc_string_view_equal(
          artifact->format,
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_TEXT))) {
    return LOOMC_SOURCE_FORMAT_TEXT;
  }
  if (loomc_string_view_equal(
          artifact->format,
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE))) {
    return LOOMC_SOURCE_FORMAT_BYTECODE;
  }
  return LOOMC_SOURCE_FORMAT_UNKNOWN;
}

loomc_status_t loomc_artifact_create_source(const loomc_artifact_t* artifact,
                                            loomc_source_format_t format,
                                            loomc_allocator_t allocator,
                                            loomc_source_t** out_source) {
  if (out_source == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_source must not be NULL");
  }
  *out_source = NULL;
  if (artifact == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_artifact_validate_string_view(artifact->format));
  LOOMC_RETURN_IF_ERROR(
      loomc_artifact_validate_string_view(artifact->identifier));
  LOOMC_RETURN_IF_ERROR(loomc_artifact_validate_contents(artifact->contents));

  loomc_source_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(options),
      .format = loomc_artifact_infer_source_format(artifact, format),
      .identifier = artifact->identifier,
      .contents = artifact->contents,
      .storage = LOOMC_SOURCE_STORAGE_COPY,
  };
  return loomc_source_create(&options, allocator, out_source);
}

loomc_status_t loomc_artifact_write_to_file(const loomc_artifact_t* artifact,
                                            FILE* file) {
  if (artifact == NULL || file == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact and file must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_artifact_validate_contents(artifact->contents));
  if (artifact->contents.data_length == 0) {
    return loomc_ok_status();
  }
  if (fwrite(artifact->contents.data, 1, artifact->contents.data_length,
             file) != artifact->contents.data_length) {
    return loomc_make_status(LOOMC_STATUS_UNKNOWN,
                             "failed to write artifact bytes");
  }
  return loomc_ok_status();
}

loomc_status_t loomc_artifact_write_to_path(const loomc_artifact_t* artifact,
                                            loomc_string_view_t path,
                                            loomc_allocator_t allocator) {
  if (artifact == NULL || loomc_string_view_is_empty(path)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact and path must be valid");
  }
  LOOMC_RETURN_IF_ERROR(loomc_artifact_validate_string_view(path));
  LOOMC_RETURN_IF_ERROR(loomc_artifact_validate_contents(artifact->contents));

  if (!loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "allocator.ctl must not be NULL");
  }
  iree_io_stream_t* stream = NULL;
  loomc_status_t status = loomc_status_from_iree(iree_io_stdio_stream_open(
      IREE_IO_STDIO_STREAM_MODE_WRITE | IREE_IO_STDIO_STREAM_MODE_DISCARD,
      iree_string_view_from_loomc(path), iree_allocator_from_loomc(allocator),
      &stream));
  if (loomc_status_is_ok(status) && artifact->contents.data_length != 0) {
    status = loomc_status_from_iree(iree_io_stream_write(
        stream, artifact->contents.data_length, artifact->contents.data));
  }
  iree_io_stream_release(stream);
  return status;
}
