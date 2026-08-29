// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "module_bytecode.h"

#include <string.h>

#include "context.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/io/stream.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loomc/iree.h"
#include "module.h"
#include "source.h"
#include "target.h"

enum {
  LOOMC_MODULE_BYTECODE_BLOCK_SIZE = 32 * 1024,
};

typedef struct loomc_module_byte_buffer_stream_t {
  // Base stream header.
  iree_io_stream_t base;

  // Allocator used for stream and buffer storage.
  loomc_allocator_t allocator;

  // Owned contiguous byte buffer.
  uint8_t* data;

  // Number of valid bytes in |data|.
  iree_host_size_t length;

  // Allocated capacity of |data|.
  iree_host_size_t capacity;

  // Current stream cursor.
  iree_io_stream_pos_t position;
} loomc_module_byte_buffer_stream_t;

static loomc_status_t loomc_module_decode_bytecode_source(
    loomc_context_t* context, const loomc_source_t* source,
    const loomc_module_resolved_deserialize_options_t* options,
    loom_diagnostic_sink_t diagnostic_sink, loomc_allocator_t allocator,
    loomc_module_t* module, loom_module_t** out_internal_module) {
  *out_internal_module = NULL;
  const loomc_byte_span_t contents = loomc_source_contents(source);
  loom_bytecode_read_options_t read_options = {
      .diagnostic_sink = diagnostic_sink,
  };
  loomc_target_pass_environment_initialize_low_repr_environment(
      loomc_context_target_pass_environment(context),
      &read_options.low_repr_environment);
  loom_bytecode_read_result_t read_result = {0};
  loomc_status_t status = loomc_status_from_iree(loom_bytecode_read_module(
      iree_make_const_byte_span(contents.data, contents.data_length),
      iree_string_view_from_loomc(options->identifier),
      loomc_context_loom_context(context), loomc_module_block_pool(module),
      &read_options, &read_result, out_internal_module,
      iree_allocator_from_loomc(allocator)));
  if (loomc_status_is_ok(status) && read_result.error_count != 0) {
    loom_module_free(*out_internal_module);
    *out_internal_module = NULL;
  }
  return status;
}

static loomc_module_byte_buffer_stream_t* loomc_module_byte_buffer_stream_cast(
    iree_io_stream_t* base_stream) {
  return (loomc_module_byte_buffer_stream_t*)base_stream;
}

static void loomc_module_byte_buffer_stream_destroy(
    iree_io_stream_t* IREE_RESTRICT base_stream) {
  loomc_module_byte_buffer_stream_t* stream =
      loomc_module_byte_buffer_stream_cast(base_stream);
  const loomc_allocator_t allocator = stream->allocator;
  loomc_allocator_free(allocator, stream->data);
  loomc_allocator_free(allocator, stream);
}

static iree_io_stream_pos_t loomc_module_byte_buffer_stream_offset(
    iree_io_stream_t* base_stream) {
  return loomc_module_byte_buffer_stream_cast(base_stream)->position;
}

static iree_io_stream_pos_t loomc_module_byte_buffer_stream_length(
    iree_io_stream_t* base_stream) {
  return (iree_io_stream_pos_t)loomc_module_byte_buffer_stream_cast(base_stream)
      ->length;
}

static iree_status_t loomc_module_byte_buffer_stream_seek(
    iree_io_stream_t* base_stream, iree_io_stream_seek_mode_t seek_mode,
    iree_io_stream_pos_t seek_offset) {
  loomc_module_byte_buffer_stream_t* stream =
      loomc_module_byte_buffer_stream_cast(base_stream);
  iree_io_stream_pos_t position = 0;
  switch (seek_mode) {
    case IREE_IO_STREAM_SEEK_SET:
      position = seek_offset;
      break;
    case IREE_IO_STREAM_SEEK_FROM_CURRENT:
      position = stream->position + seek_offset;
      break;
    case IREE_IO_STREAM_SEEK_FROM_END:
      position = (iree_io_stream_pos_t)stream->length + seek_offset;
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stream seek mode is invalid");
  }
  if (position < 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "stream seek before start");
  }
  stream->position = position;
  return iree_ok_status();
}

static iree_status_t loomc_module_byte_buffer_stream_reserve(
    loomc_module_byte_buffer_stream_t* stream, iree_host_size_t capacity) {
  if (stream->capacity >= capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity = stream->capacity ? stream->capacity : 4096;
  while (new_capacity < capacity) {
    if (new_capacity > IREE_HOST_SIZE_MAX / 2) {
      new_capacity = capacity;
      break;
    }
    new_capacity *= 2;
  }
  const loomc_allocator_alloc_params_t params = {
      .byte_length = new_capacity,
  };
  void* data = stream->data;
  iree_status_t status = iree_status_from_loomc(stream->allocator.ctl(
      stream->allocator.self, LOOMC_ALLOCATOR_COMMAND_REALLOC, &params, &data));
  if (iree_status_is_ok(status)) {
    stream->data = (uint8_t*)data;
    stream->capacity = new_capacity;
  }
  return status;
}

static iree_status_t loomc_module_byte_buffer_stream_read(
    iree_io_stream_t* base_stream, iree_host_size_t buffer_capacity,
    void* buffer, iree_host_size_t* out_buffer_length) {
  loomc_module_byte_buffer_stream_t* stream =
      loomc_module_byte_buffer_stream_cast(base_stream);
  if (stream->position < 0 || (uint64_t)stream->position > stream->length) {
    if (out_buffer_length != NULL) {
      *out_buffer_length = 0;
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "read past stream end");
  }
  const iree_host_size_t remaining =
      stream->length - (iree_host_size_t)stream->position;
  const iree_host_size_t read_length =
      remaining < buffer_capacity ? remaining : buffer_capacity;
  if (read_length != 0) {
    memcpy(buffer, stream->data + stream->position, read_length);
  }
  stream->position += read_length;
  if (out_buffer_length != NULL) {
    *out_buffer_length = read_length;
    return iree_ok_status();
  }
  if (read_length != buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "short stream read");
  }
  return iree_ok_status();
}

static iree_status_t loomc_module_byte_buffer_stream_write(
    iree_io_stream_t* base_stream, iree_host_size_t buffer_length,
    const void* buffer) {
  loomc_module_byte_buffer_stream_t* stream =
      loomc_module_byte_buffer_stream_cast(base_stream);
  if (stream->position < 0 || (uint64_t)stream->position > IREE_HOST_SIZE_MAX ||
      buffer_length > IREE_HOST_SIZE_MAX - (iree_host_size_t)stream->position) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "stream write length is too large");
  }
  const iree_host_size_t position = (iree_host_size_t)stream->position;
  const iree_host_size_t new_position = position + buffer_length;
  IREE_RETURN_IF_ERROR(
      loomc_module_byte_buffer_stream_reserve(stream, new_position));
  if (position > stream->length) {
    memset(stream->data + stream->length, 0, position - stream->length);
  }
  if (buffer_length != 0) {
    memcpy(stream->data + position, buffer, buffer_length);
  }
  stream->position = (iree_io_stream_pos_t)new_position;
  if (new_position > stream->length) {
    stream->length = new_position;
  }
  return iree_ok_status();
}

static iree_status_t loomc_module_byte_buffer_stream_fill(
    iree_io_stream_t* base_stream, iree_io_stream_pos_t count,
    const void* pattern, iree_host_size_t pattern_length) {
  if (count < 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "stream fill count is negative");
  }
  for (iree_io_stream_pos_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loomc_module_byte_buffer_stream_write(
        base_stream, pattern_length, pattern));
  }
  return iree_ok_status();
}

static iree_status_t loomc_module_byte_buffer_stream_map_read(
    iree_io_stream_t* base_stream, iree_host_size_t length,
    iree_const_byte_span_t* out_span) {
  (void)base_stream;
  (void)length;
  (void)out_span;
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "module byte buffer streams do not support mapping");
}

static iree_status_t loomc_module_byte_buffer_stream_map_write(
    iree_io_stream_t* base_stream, iree_host_size_t length,
    iree_byte_span_t* out_span) {
  (void)base_stream;
  (void)length;
  (void)out_span;
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "module byte buffer streams do not support mapping");
}

static const iree_io_stream_vtable_t loomc_module_byte_buffer_stream_vtable = {
    .destroy = loomc_module_byte_buffer_stream_destroy,
    .offset = loomc_module_byte_buffer_stream_offset,
    .length = loomc_module_byte_buffer_stream_length,
    .seek = loomc_module_byte_buffer_stream_seek,
    .read = loomc_module_byte_buffer_stream_read,
    .write = loomc_module_byte_buffer_stream_write,
    .fill = loomc_module_byte_buffer_stream_fill,
    .map_read = loomc_module_byte_buffer_stream_map_read,
    .map_write = loomc_module_byte_buffer_stream_map_write,
};

static loomc_status_t loomc_module_byte_buffer_stream_create(
    loomc_allocator_t allocator,
    loomc_module_byte_buffer_stream_t** out_stream) {
  *out_stream = NULL;
  loomc_module_byte_buffer_stream_t* stream = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*stream), (void**)&stream));
  memset(stream, 0, sizeof(*stream));
  iree_atomic_ref_count_init(&stream->base.ref_count);
  stream->base.vtable = &loomc_module_byte_buffer_stream_vtable;
  stream->base.mode =
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE;
  stream->allocator = allocator;
  *out_stream = stream;
  return loomc_ok_status();
}

static uint8_t* loomc_module_byte_buffer_stream_take_storage(
    loomc_module_byte_buffer_stream_t* stream, iree_host_size_t* out_length) {
  uint8_t* data = stream->data;
  *out_length = stream->length;
  stream->data = NULL;
  stream->length = 0;
  stream->capacity = 0;
  stream->position = 0;
  return data;
}

static loomc_status_t loomc_module_serialize_bytecode_to_stream(
    const loomc_context_t* context, const loom_module_t* internal_module,
    const loomc_module_symbol_projection_t* projection,
    iree_io_stream_t* target_stream, loomc_allocator_t allocator) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(LOOMC_MODULE_BYTECODE_BLOCK_SIZE,
                                   iree_allocator_from_loomc(allocator),
                                   &block_pool);
  loom_bytecode_write_options_t write_options = {0};
  loomc_target_pass_environment_initialize_low_repr_environment(
      loomc_context_target_pass_environment(context),
      &write_options.low_repr_environment);
  if (projection != NULL) {
    write_options.symbol_projection = (loom_bytecode_symbol_projection_t){
        .module_symbol_ids = projection->module_symbol_ids,
        .wire_symbol_ordinals = projection->bytecode_symbol_ordinals,
        .count = projection->count,
    };
  }
  loomc_status_t status = loomc_status_from_iree(loom_bytecode_write_module(
      internal_module, target_stream, &write_options, &block_pool));
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

loomc_status_t loomc_module_serialize_internal_bytecode_to_source(
    const loomc_context_t* context, const loom_module_t* internal_module,
    loomc_string_view_t identifier,
    const loomc_module_symbol_projection_t* projection,
    loomc_allocator_t allocator, loomc_source_t** out_source) {
  *out_source = NULL;
  loomc_module_byte_buffer_stream_t* stream = NULL;
  loomc_status_t status =
      loomc_module_byte_buffer_stream_create(allocator, &stream);
  iree_io_stream_t* base_stream = NULL;
  if (loomc_status_is_ok(status)) {
    base_stream = &stream->base;
    status = loomc_module_serialize_bytecode_to_stream(
        context, internal_module, projection, base_stream, allocator);
  }

  uint8_t* storage = NULL;
  iree_host_size_t stream_length = 0;
  if (loomc_status_is_ok(status)) {
    storage =
        loomc_module_byte_buffer_stream_take_storage(stream, &stream_length);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_source_create_take_contents(
        LOOMC_SOURCE_FORMAT_BYTECODE, identifier,
        loomc_make_byte_span(storage, stream_length), allocator, out_source);
  }
  if (!loomc_status_is_ok(status)) {
    loomc_allocator_free(allocator, storage);
  }
  iree_io_stream_release(base_stream);
  return status;
}

static loomc_status_t loomc_module_encode_bytecode_source(
    const loomc_module_t* module, const loom_module_t* internal_module,
    const loomc_module_resolved_serialize_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source) {
  return loomc_module_serialize_internal_bytecode_to_source(
      loomc_module_context(module), internal_module, options->identifier,
      /*projection=*/NULL, allocator, out_source);
}

loomc_status_t loomc_module_deserialize_bytecode_from_source(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_source_t* source,
    const loomc_module_deserialize_options_t* options,
    loomc_allocator_t allocator, loomc_module_t** out_module,
    loomc_result_t** out_result) {
  return loomc_module_deserialize_explicit_source(
      context, workspace, source, options, LOOMC_SOURCE_FORMAT_BYTECODE,
      loomc_module_decode_bytecode_source, allocator, out_module, out_result);
}

loomc_status_t loomc_module_serialize_bytecode_to_source(
    const loomc_module_t* module,
    const loomc_module_serialize_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source) {
  return loomc_module_serialize_explicit_source(
      module, options, LOOMC_SOURCE_FORMAT_BYTECODE,
      loomc_module_encode_bytecode_source, allocator, out_source);
}
