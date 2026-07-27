// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "source.h"

#include <stdio.h>
#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/io/file_contents.h"
#include "loomc/iree.h"

enum {
  LOOMC_SOURCE_FILE_READ_BLOCK_SIZE = 32 * 1024,
};

struct loomc_source_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used to release this object and copied storage.
  loomc_allocator_t allocator;
  // Source format.
  loomc_source_format_t format;
  // Copied diagnostic/cache identifier.
  loomc_string_view_t identifier;
  // Source bytes.
  loomc_byte_span_t contents;
  // Storage policy for contents.
  loomc_source_storage_t storage;
  // Callback used to release externally owned contents.
  loomc_source_release_fn_t release;
  // User data passed to release.
  void* release_user_data;
};

typedef struct loomc_source_resolved_load_options_t {
  // Loaded source format.
  loomc_source_format_t format;

  // Identifier used in diagnostics and cache keys.
  loomc_string_view_t identifier;
} loomc_source_resolved_load_options_t;

static loomc_status_t loomc_source_validate_options(
    const loomc_source_options_t* options) {
  if (options == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source options must not be NULL");
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(LOOMC_STATUS_UNIMPLEMENTED,
                             "source option extensions are not supported");
  }
  if (options->format > LOOMC_SOURCE_FORMAT_BYTECODE) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source format is invalid");
  }
  if (options->storage > LOOMC_SOURCE_STORAGE_EXTERNAL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source storage policy is invalid");
  }
  if (options->contents.data == NULL && options->contents.data_length != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source contents have length but no data");
  }
  if (options->storage == LOOMC_SOURCE_STORAGE_EXTERNAL &&
      options->release == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "external source storage requires a release");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_source_resolve_load_options(
    const loomc_source_load_options_t* options,
    loomc_string_view_t fallback_identifier,
    loomc_source_resolved_load_options_t* out_options) {
  if (options != NULL) {
    if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
        options->type != LOOMC_STRUCTURE_TYPE_SOURCE_LOAD_OPTIONS) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "source load options have an unknown structure type");
    }
    if (options->structure_size != 0 &&
        options->structure_size < sizeof(*options)) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "source load options structure_size is too small");
    }
    if (options->next != NULL) {
      return loomc_make_status(
          LOOMC_STATUS_UNIMPLEMENTED,
          "source load option extensions are not supported");
    }
    if (options->format > LOOMC_SOURCE_FORMAT_BYTECODE) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "source load format is invalid");
    }
    if (options->identifier.data == NULL && options->identifier.size != 0) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "source load identifier has length but no data");
    }
  }

  out_options->format = options ? options->format : LOOMC_SOURCE_FORMAT_UNKNOWN;
  out_options->identifier =
      options ? options->identifier : loomc_string_view_empty();
  if (loomc_string_view_is_empty(out_options->identifier)) {
    out_options->identifier = fallback_identifier;
  }
  return loomc_ok_status();
}

static void loomc_source_destroy(loomc_source_t* source) {
  loomc_allocator_t allocator = source->allocator;
  loomc_allocator_free(allocator, (void*)source->identifier.data);
  if (source->storage == LOOMC_SOURCE_STORAGE_COPY) {
    loomc_allocator_free(allocator, (void*)source->contents.data);
  } else if (source->storage == LOOMC_SOURCE_STORAGE_EXTERNAL) {
    source->release(source->release_user_data, source->contents);
  }
  loomc_allocator_free(allocator, source);
}

static loomc_status_t loomc_source_storage_reserve(
    loomc_allocator_t allocator, loomc_host_size_t required_capacity,
    loomc_host_size_t* inout_capacity, uint8_t** inout_data) {
  if (*inout_capacity >= required_capacity) {
    return loomc_ok_status();
  }
  loomc_host_size_t new_capacity = *inout_capacity ? *inout_capacity : 4096;
  while (new_capacity < required_capacity) {
    if (new_capacity > LOOMC_HOST_SIZE_MAX / 2) {
      new_capacity = required_capacity;
      break;
    }
    new_capacity *= 2;
  }
  loomc_allocator_alloc_params_t params = {
      .byte_length = new_capacity,
  };
  void* data = *inout_data;
  loomc_status_t status = allocator.ctl(
      allocator.self, LOOMC_ALLOCATOR_COMMAND_REALLOC, &params, &data);
  if (loomc_status_is_ok(status)) {
    *inout_data = (uint8_t*)data;
    *inout_capacity = new_capacity;
  }
  return status;
}

static void loomc_source_release_file_contents(void* user_data,
                                               loomc_byte_span_t contents) {
  (void)contents;
  iree_io_file_contents_free((iree_io_file_contents_t*)user_data);
}

loomc_status_t loomc_source_create(const loomc_source_options_t* options,
                                   loomc_allocator_t allocator,
                                   loomc_source_t** out_source) {
  if (out_source == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_source must not be NULL");
  }
  *out_source = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_source_validate_options(options));

  loomc_source_t* source = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*source), (void**)&source));
  iree_atomic_ref_count_init(&source->ref_count);
  source->allocator = allocator;
  source->format = options->format;
  source->storage = LOOMC_SOURCE_STORAGE_BORROWED;
  source->release = options->release;
  source->release_user_data = options->release_user_data;

  loomc_status_t status = loomc_string_view_clone(
      options->identifier, allocator, &source->identifier);
  if (loomc_status_is_ok(status)) {
    if (options->storage == LOOMC_SOURCE_STORAGE_COPY &&
        options->contents.data_length != 0) {
      uint8_t* contents = NULL;
      status = loomc_allocator_malloc_uninitialized(
          allocator, options->contents.data_length, (void**)&contents);
      if (loomc_status_is_ok(status)) {
        memcpy(contents, options->contents.data, options->contents.data_length);
        source->contents =
            loomc_make_byte_span(contents, options->contents.data_length);
        source->storage = LOOMC_SOURCE_STORAGE_COPY;
      }
    } else {
      source->contents = options->contents;
      source->storage = options->storage;
    }
  }

  if (loomc_status_is_ok(status)) {
    *out_source = source;
  } else {
    loomc_source_destroy(source);
  }
  return status;
}

loomc_status_t loomc_source_read_file_to_storage(
    FILE* file, loomc_allocator_t allocator, uint8_t** out_data,
    loomc_host_size_t* out_data_length) {
  if (out_data == NULL || out_data_length == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_data and out_data_length must not be NULL");
  }
  *out_data = NULL;
  *out_data_length = 0;
  if (file == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "file must not be NULL");
  }

  if (!loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "allocator.ctl must not be NULL");
  }
  uint8_t* data = NULL;
  loomc_host_size_t length = 0;
  loomc_host_size_t capacity = 0;
  loomc_status_t status = loomc_ok_status();
  uint8_t buffer[LOOMC_SOURCE_FILE_READ_BLOCK_SIZE];
  while (loomc_status_is_ok(status)) {
    size_t read_length = fread(buffer, 1, sizeof(buffer), file);
    if (read_length != 0) {
      if (read_length > LOOMC_HOST_SIZE_MAX - length) {
        status = loomc_make_status(LOOMC_STATUS_OUT_OF_RANGE,
                                   "source file is too large");
      }
      if (loomc_status_is_ok(status)) {
        const loomc_host_size_t new_length = length + read_length;
        status = loomc_source_storage_reserve(allocator, new_length, &capacity,
                                              &data);
        if (loomc_status_is_ok(status)) {
          memcpy(data + length, buffer, read_length);
          length = new_length;
        }
      }
    }
    if (read_length < sizeof(buffer)) {
      if (ferror(file) != 0 && loomc_status_is_ok(status)) {
        status =
            loomc_make_status(LOOMC_STATUS_UNKNOWN, "failed to read source");
      }
      break;
    }
  }

  if (loomc_status_is_ok(status)) {
    *out_data = data;
    *out_data_length = length;
    data = NULL;
  }
  loomc_allocator_free(allocator, data);
  return status;
}

loomc_status_t loomc_source_create_from_file(
    FILE* file, const loomc_source_load_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source) {
  if (out_source == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_source must not be NULL");
  }
  *out_source = NULL;
  if (file == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "file must not be NULL");
  }

  loomc_source_resolved_load_options_t resolved_options = {0};
  LOOMC_RETURN_IF_ERROR(loomc_source_resolve_load_options(
      options, loomc_string_view_empty(), &resolved_options));

  uint8_t* contents = NULL;
  loomc_host_size_t contents_length = 0;
  LOOMC_RETURN_IF_ERROR(loomc_source_read_file_to_storage(
      file, allocator, &contents, &contents_length));

  loomc_status_t status = loomc_source_create_take_contents(
      resolved_options.format, resolved_options.identifier,
      loomc_make_byte_span(contents, contents_length), allocator, out_source);
  if (loomc_status_is_ok(status)) {
    contents = NULL;
  }
  loomc_allocator_free(allocator, contents);
  return status;
}

loomc_status_t loomc_source_create_from_path(
    loomc_string_view_t path, const loomc_source_load_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source) {
  if (out_source == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_source must not be NULL");
  }
  *out_source = NULL;
  if (path.data == NULL && path.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "path has length but no data");
  }
  if (loomc_string_view_is_empty(path)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "path must not be empty");
  }

  loomc_source_resolved_load_options_t resolved_options = {0};
  LOOMC_RETURN_IF_ERROR(
      loomc_source_resolve_load_options(options, path, &resolved_options));

  if (!loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "allocator.ctl must not be NULL");
  }
  iree_io_file_contents_t* file_contents = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(iree_io_file_contents_read(
      iree_string_view_from_loomc(path), iree_allocator_from_loomc(allocator),
      &file_contents)));

  loomc_source_t* source = NULL;
  loomc_byte_span_t contents =
      loomc_byte_span_from_iree(file_contents->const_buffer);
  loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = resolved_options.format,
      .identifier = resolved_options.identifier,
      .contents = contents,
      .storage = LOOMC_SOURCE_STORAGE_EXTERNAL,
      .release = loomc_source_release_file_contents,
      .release_user_data = file_contents,
  };
  loomc_status_t status =
      loomc_source_create(&source_options, allocator, &source);
  if (loomc_status_is_ok(status)) {
    file_contents = NULL;
    *out_source = source;
    source = NULL;
  }
  loomc_source_release(source);
  iree_io_file_contents_free(file_contents);
  return status;
}

loomc_status_t loomc_source_create_take_contents(loomc_source_format_t format,
                                                 loomc_string_view_t identifier,
                                                 loomc_byte_span_t contents,
                                                 loomc_allocator_t allocator,
                                                 loomc_source_t** out_source) {
  if (out_source == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_source must not be NULL");
  }
  if (contents.data == NULL && contents.data_length != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source contents have length but no data");
  }
  *out_source = NULL;
  loomc_source_t* source = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*source), (void**)&source));
  iree_atomic_ref_count_init(&source->ref_count);
  source->allocator = allocator;
  source->format = format;
  source->contents = loomc_byte_span_empty();
  source->storage = LOOMC_SOURCE_STORAGE_BORROWED;
  source->release = NULL;
  source->release_user_data = NULL;

  loomc_status_t status =
      loomc_string_view_clone(identifier, allocator, &source->identifier);
  if (loomc_status_is_ok(status)) {
    source->contents = contents;
    source->storage = LOOMC_SOURCE_STORAGE_COPY;
    *out_source = source;
  } else {
    loomc_source_destroy(source);
  }
  return status;
}

loomc_status_t loomc_source_take_contents(loomc_source_t* source,
                                          loomc_byte_span_t* out_contents) {
  if (out_contents == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_contents must not be NULL");
  }
  *out_contents = loomc_byte_span_empty();
  if (source == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source must not be NULL");
  }
  if (source->storage != LOOMC_SOURCE_STORAGE_COPY) {
    return loomc_make_status(
        LOOMC_STATUS_FAILED_PRECONDITION,
        "source contents are not allocator-owned and cannot be transferred");
  }

  *out_contents = source->contents;
  source->contents = loomc_byte_span_empty();
  source->storage = LOOMC_SOURCE_STORAGE_BORROWED;
  return loomc_ok_status();
}

void loomc_source_retain(loomc_source_t* source) {
  if (source == NULL) {
    return;
  }
  iree_atomic_ref_count_inc(&source->ref_count);
}

void loomc_source_release(loomc_source_t* source) {
  if (source == NULL) {
    return;
  }
  if (iree_atomic_ref_count_dec(&source->ref_count) == 1) {
    loomc_source_destroy(source);
  }
}

loomc_source_format_t loomc_source_format(const loomc_source_t* source) {
  return source ? source->format : LOOMC_SOURCE_FORMAT_UNKNOWN;
}

loomc_string_view_t loomc_source_identifier(const loomc_source_t* source) {
  return source ? source->identifier : loomc_string_view_empty();
}

loomc_byte_span_t loomc_source_contents(const loomc_source_t* source) {
  return source ? source->contents : loomc_byte_span_empty();
}
