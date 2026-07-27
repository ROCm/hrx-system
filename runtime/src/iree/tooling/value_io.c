// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/value_io.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "iree/io/stdio_stream.h"
#include "iree/io/stream.h"
#include "iree/tooling/numpy_io.h"

//===----------------------------------------------------------------------===//
// Scalar Values
//===----------------------------------------------------------------------===//

static bool iree_tooling_value_copy_to_cstring(iree_string_view_t value,
                                               char* buffer,
                                               iree_host_size_t capacity) {
  if (value.size >= capacity) return false;
  memcpy(buffer, value.data, value.size);
  buffer[value.size] = 0;
  return true;
}

static iree_status_t iree_tooling_value_parse_failure(
    iree_tooling_value_kind_t kind, iree_string_view_t literal) {
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "parsing value `%.*s` as kind %u", (int)literal.size,
                          literal.data, (unsigned)kind);
}

static bool iree_tooling_value_parse_int64_range(iree_string_view_t literal,
                                                 int64_t min_value,
                                                 int64_t max_value,
                                                 int64_t* out_value) {
  char buffer[64] = {0};
  if (!iree_tooling_value_copy_to_cstring(literal, buffer,
                                          IREE_ARRAYSIZE(buffer))) {
    return false;
  }
  errno = 0;
  char* end = NULL;
  long long parsed_value = strtoll(buffer, &end, /*base=*/0);
  if (buffer == end || *end != 0) return false;
  if ((parsed_value == LLONG_MIN || parsed_value == LLONG_MAX) &&
      errno == ERANGE) {
    return false;
  }
  if (parsed_value < min_value || parsed_value > max_value) return false;
  *out_value = (int64_t)parsed_value;
  return true;
}

static bool iree_tooling_value_parse_uint64_range(iree_string_view_t literal,
                                                  uint64_t max_value,
                                                  uint64_t* out_value) {
  if (iree_string_view_starts_with_char(literal, '-')) return false;
  char buffer[64] = {0};
  if (!iree_tooling_value_copy_to_cstring(literal, buffer,
                                          IREE_ARRAYSIZE(buffer))) {
    return false;
  }
  errno = 0;
  char* end = NULL;
  unsigned long long parsed_value = strtoull(buffer, &end, /*base=*/0);
  if (buffer == end || *end != 0) return false;
  if (parsed_value == ULLONG_MAX && errno == ERANGE) return false;
  if (parsed_value > max_value) return false;
  *out_value = (uint64_t)parsed_value;
  return true;
}

static bool iree_tooling_value_parse_float32(iree_string_view_t literal,
                                             float* out_value) {
  char buffer[64] = {0};
  if (!iree_tooling_value_copy_to_cstring(literal, buffer,
                                          IREE_ARRAYSIZE(buffer))) {
    return false;
  }
  errno = 0;
  char* end = NULL;
  float parsed_value = strtof(buffer, &end);
  if (buffer == end || *end != 0) return false;
  if (errno == ERANGE) return false;
  *out_value = parsed_value;
  return true;
}

static bool iree_tooling_value_parse_float64(iree_string_view_t literal,
                                             double* out_value) {
  char buffer[128] = {0};
  if (!iree_tooling_value_copy_to_cstring(literal, buffer,
                                          IREE_ARRAYSIZE(buffer))) {
    return false;
  }
  errno = 0;
  char* end = NULL;
  double parsed_value = strtod(buffer, &end);
  if (buffer == end || *end != 0) return false;
  if (errno == ERANGE) return false;
  *out_value = parsed_value;
  return true;
}

IREE_API_EXPORT iree_status_t iree_tooling_value_parse(
    iree_tooling_value_kind_t kind, iree_string_view_t literal,
    iree_tooling_value_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  memset(out_value, 0, sizeof(*out_value));
  out_value->kind = kind;
  switch (kind) {
    case IREE_TOOLING_VALUE_KIND_I32: {
      int64_t parsed_value = 0;
      if (!iree_tooling_value_parse_int64_range(literal, INT32_MIN, INT32_MAX,
                                                &parsed_value)) {
        return iree_tooling_value_parse_failure(kind, literal);
      }
      out_value->storage.i32 = (int32_t)parsed_value;
      return iree_ok_status();
    }
    case IREE_TOOLING_VALUE_KIND_U32:
    case IREE_TOOLING_VALUE_KIND_RAW_U32: {
      uint64_t parsed_value = 0;
      if (!iree_tooling_value_parse_uint64_range(literal, UINT32_MAX,
                                                 &parsed_value)) {
        return iree_tooling_value_parse_failure(kind, literal);
      }
      out_value->storage.u32 = (uint32_t)parsed_value;
      return iree_ok_status();
    }
    case IREE_TOOLING_VALUE_KIND_I64: {
      int64_t parsed_value = 0;
      if (!iree_tooling_value_parse_int64_range(literal, INT64_MIN, INT64_MAX,
                                                &parsed_value)) {
        return iree_tooling_value_parse_failure(kind, literal);
      }
      out_value->storage.i64 = parsed_value;
      return iree_ok_status();
    }
    case IREE_TOOLING_VALUE_KIND_U64: {
      uint64_t parsed_value = 0;
      if (!iree_tooling_value_parse_uint64_range(literal, UINT64_MAX,
                                                 &parsed_value)) {
        return iree_tooling_value_parse_failure(kind, literal);
      }
      out_value->storage.u64 = parsed_value;
      return iree_ok_status();
    }
    case IREE_TOOLING_VALUE_KIND_F32: {
      float parsed_value = 0.0f;
      if (!iree_tooling_value_parse_float32(literal, &parsed_value)) {
        return iree_tooling_value_parse_failure(kind, literal);
      }
      out_value->storage.f32 = parsed_value;
      return iree_ok_status();
    }
    case IREE_TOOLING_VALUE_KIND_F64: {
      double parsed_value = 0.0;
      if (!iree_tooling_value_parse_float64(literal, &parsed_value)) {
        return iree_tooling_value_parse_failure(kind, literal);
      }
      out_value->storage.f64 = parsed_value;
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported value kind %u", (unsigned)kind);
  }
}

static bool iree_tooling_value_kind_parse(iree_string_view_t type,
                                          iree_tooling_value_kind_t* out_kind) {
  if (iree_string_view_equal(type, IREE_SV("i32"))) {
    *out_kind = IREE_TOOLING_VALUE_KIND_I32;
    return true;
  } else if (iree_string_view_equal(type, IREE_SV("u32"))) {
    *out_kind = IREE_TOOLING_VALUE_KIND_U32;
    return true;
  } else if (iree_string_view_equal(type, IREE_SV("i64"))) {
    *out_kind = IREE_TOOLING_VALUE_KIND_I64;
    return true;
  } else if (iree_string_view_equal(type, IREE_SV("u64"))) {
    *out_kind = IREE_TOOLING_VALUE_KIND_U64;
    return true;
  } else if (iree_string_view_equal(type, IREE_SV("f32"))) {
    *out_kind = IREE_TOOLING_VALUE_KIND_F32;
    return true;
  } else if (iree_string_view_equal(type, IREE_SV("f64"))) {
    *out_kind = IREE_TOOLING_VALUE_KIND_F64;
    return true;
  }
  return false;
}

static bool iree_tooling_value_is_bare_raw_u32(iree_string_view_t spec) {
  return iree_string_view_starts_with(spec, IREE_SV("0x")) ||
         iree_string_view_starts_with(spec, IREE_SV("0X"));
}

IREE_API_EXPORT iree_status_t iree_tooling_value_spec_parse(
    iree_string_view_t spec, iree_tooling_value_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  if (iree_tooling_value_is_bare_raw_u32(spec)) {
    return iree_tooling_value_parse(IREE_TOOLING_VALUE_KIND_RAW_U32, spec,
                                    out_value);
  }
  iree_string_view_t type = iree_string_view_empty();
  iree_string_view_t literal = iree_string_view_empty();
  if (iree_string_view_split(spec, '=', &type, &literal) == -1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value specification `%.*s` must use type=value "
                            "or bare 0x... raw u32 form",
                            (int)spec.size, spec.data);
  }
  iree_tooling_value_kind_t kind = IREE_TOOLING_VALUE_KIND_NONE;
  if (!iree_tooling_value_kind_parse(type, &kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported value type `%.*s`", (int)type.size,
                            type.data);
  }
  return iree_tooling_value_parse(kind, literal, out_value);
}

IREE_API_EXPORT iree_host_size_t
iree_tooling_value_abi_word_count(iree_tooling_value_kind_t kind) {
  switch (kind) {
    case IREE_TOOLING_VALUE_KIND_I32:
    case IREE_TOOLING_VALUE_KIND_U32:
    case IREE_TOOLING_VALUE_KIND_F32:
    case IREE_TOOLING_VALUE_KIND_RAW_U32:
      return 1;
    case IREE_TOOLING_VALUE_KIND_I64:
    case IREE_TOOLING_VALUE_KIND_U64:
    case IREE_TOOLING_VALUE_KIND_F64:
      return 2;
    default:
      return 0;
  }
}

IREE_API_EXPORT iree_status_t iree_tooling_value_write_abi_words(
    const iree_tooling_value_t* value, iree_host_size_t word_capacity,
    uint32_t* out_words, iree_host_size_t* out_word_count) {
  IREE_ASSERT_ARGUMENT(value);
  IREE_ASSERT_ARGUMENT(out_word_count);
  const iree_host_size_t word_count =
      iree_tooling_value_abi_word_count(value->kind);
  if (word_count == 0) {
    *out_word_count = 0;
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported value kind %u", (unsigned)value->kind);
  }
  if (word_capacity < word_count) {
    *out_word_count = word_count;
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "value kind %u requires %" PRIhsz
                            " ABI words but capacity is %" PRIhsz,
                            (unsigned)value->kind, word_count, word_capacity);
  }

  switch (value->kind) {
    case IREE_TOOLING_VALUE_KIND_I32:
      memcpy(out_words, &value->storage.i32, sizeof(uint32_t));
      break;
    case IREE_TOOLING_VALUE_KIND_U32:
    case IREE_TOOLING_VALUE_KIND_RAW_U32:
      memcpy(out_words, &value->storage.u32, sizeof(uint32_t));
      break;
    case IREE_TOOLING_VALUE_KIND_I64:
      memcpy(out_words, &value->storage.i64, sizeof(uint64_t));
      break;
    case IREE_TOOLING_VALUE_KIND_U64:
      memcpy(out_words, &value->storage.u64, sizeof(uint64_t));
      break;
    case IREE_TOOLING_VALUE_KIND_F32: {
      static_assert(sizeof(float) == sizeof(uint32_t), "f32 ABI size");
      memcpy(out_words, &value->storage.f32, sizeof(uint32_t));
      break;
    }
    case IREE_TOOLING_VALUE_KIND_F64: {
      static_assert(sizeof(double) == sizeof(uint64_t), "f64 ABI size");
      memcpy(out_words, &value->storage.f64, sizeof(uint64_t));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported value kind %u",
                              (unsigned)value->kind);
  }
  *out_word_count = word_count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Stream List
//===----------------------------------------------------------------------===//

static iree_status_t iree_tooling_value_io_stream_open_path(
    iree_io_stdio_stream_mode_t mode, iree_string_view_t path,
    uint64_t file_offset, iree_allocator_t host_allocator,
    iree_io_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);

  iree_io_stream_t* stream = NULL;
  iree_status_t status =
      iree_io_stdio_stream_open(mode, path, host_allocator, &stream);
  if (iree_status_is_ok(status) && file_offset > 0) {
    status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, file_offset);
  }

  if (iree_status_is_ok(status)) {
    *out_stream = stream;
  } else {
    iree_io_stream_release(stream);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

typedef struct iree_tooling_value_io_stream_list_entry_t {
  iree_string_view_t path;
  iree_io_stream_t* stream;
} iree_tooling_value_io_stream_list_entry_t;

typedef struct iree_tooling_value_io_stream_list_t {
  iree_allocator_t host_allocator;
  iree_io_stdio_stream_mode_t mode;
  iree_host_size_t capacity;
  iree_host_size_t count;
  iree_tooling_value_io_stream_list_entry_t** entries;
} iree_tooling_value_io_stream_list_t;

static iree_status_t iree_tooling_value_io_stream_list_allocate(
    iree_io_stdio_stream_mode_t mode, iree_allocator_t host_allocator,
    iree_tooling_value_io_stream_list_t** out_list) {
  IREE_ASSERT_ARGUMENT(out_list);
  *out_list = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tooling_value_io_stream_list_t* list = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*list), (void**)&list);
  if (iree_status_is_ok(status)) {
    memset(list, 0, sizeof(*list));
    list->host_allocator = host_allocator;
    list->mode = mode;
    *out_list = list;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_tooling_value_io_stream_list_free(
    iree_tooling_value_io_stream_list_t* list) {
  if (!list) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < list->count; ++i) {
    iree_tooling_value_io_stream_list_entry_t* entry = list->entries[i];
    iree_io_stream_release(entry->stream);
    iree_allocator_free(list->host_allocator, entry);
  }
  iree_allocator_free(list->host_allocator, list->entries);
  iree_allocator_free(list->host_allocator, list);

  IREE_TRACE_ZONE_END(z0);
}

static iree_tooling_value_io_stream_list_entry_t*
iree_tooling_value_io_stream_list_find_entry(
    iree_tooling_value_io_stream_list_t* list, iree_string_view_t path) {
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    iree_tooling_value_io_stream_list_entry_t* entry = list->entries[i];
    if (iree_string_view_equal(path, entry->path)) return entry;
  }
  return NULL;
}

static iree_status_t iree_tooling_value_io_stream_list_append_entry(
    iree_tooling_value_io_stream_list_t* list, iree_string_view_t path,
    iree_io_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);

  if (list->count + 1 > list->capacity) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_allocator_grow_array(list->host_allocator,
                                  /*min_capacity=*/16, sizeof(list->entries[0]),
                                  &list->capacity, (void**)&list->entries));
  }

  iree_tooling_value_io_stream_list_entry_t* entry = NULL;
  iree_status_t status = iree_allocator_malloc_with_trailing(
      list->host_allocator, sizeof(*entry), path.size, (void**)&entry);
  if (iree_status_is_ok(status)) {
    entry->path.data = (const char*)entry + sizeof(*entry);
    entry->path.size = path.size;
    memcpy((void*)entry->path.data, path.data, path.size);
    entry->stream = stream;
    iree_io_stream_retain(entry->stream);
  }
  if (iree_status_is_ok(status)) {
    list->entries[list->count++] = entry;
  } else {
    iree_allocator_free(list->host_allocator, entry);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_tooling_value_io_stream_list_open_existing(
    iree_string_view_t path, iree_tooling_value_io_stream_list_entry_t* entry,
    bool is_append, iree_io_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);

  iree_status_t status = iree_ok_status();
  if (!is_append) {
    if (iree_all_bits_set(iree_io_stream_mode(entry->stream),
                          IREE_IO_STREAM_MODE_SEEKABLE)) {
      status = iree_io_stream_seek(entry->stream, IREE_IO_STREAM_SEEK_SET, 0);
    } else {
      status = iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "opened stream from `%.*s` is not seekable and cannot be reopened",
          (int)path.size, path.data);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_io_stream_retain(entry->stream);
    *out_stream = entry->stream;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_tooling_value_io_stream_list_open(
    iree_tooling_value_io_stream_list_t* list, iree_string_view_t path,
    bool is_append, iree_io_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);

  iree_tooling_value_io_stream_list_entry_t* entry =
      iree_tooling_value_io_stream_list_find_entry(list, path);
  if (entry) {
    iree_status_t status = iree_tooling_value_io_stream_list_open_existing(
        path, entry, is_append, out_stream);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_io_stream_t* stream = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tooling_value_io_stream_open_path(
              list->mode, path, 0, list->host_allocator, &stream));
  iree_status_t status =
      iree_tooling_value_io_stream_list_append_entry(list, path, stream);
  if (iree_status_is_ok(status)) {
    *out_stream = stream;
  } else {
    iree_io_stream_release(stream);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Buffer Specifications
//===----------------------------------------------------------------------===//

struct iree_tooling_value_io_context_t {
  iree_tooling_value_io_stream_list_t* input_streams;
};

IREE_API_EXPORT iree_status_t iree_tooling_value_io_context_allocate(
    iree_allocator_t host_allocator,
    iree_tooling_value_io_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tooling_value_io_context_t* context = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*context), (void**)&context);
  if (iree_status_is_ok(status)) {
    memset(context, 0, sizeof(*context));
    status = iree_tooling_value_io_stream_list_allocate(
        IREE_IO_STDIO_STREAM_MODE_READ, host_allocator,
        &context->input_streams);
  }
  if (iree_status_is_ok(status)) {
    *out_context = context;
  } else {
    if (context != NULL) {
      iree_tooling_value_io_stream_list_free(context->input_streams);
      iree_allocator_free(host_allocator, context);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_tooling_value_io_context_free(
    iree_tooling_value_io_context_t* context) {
  if (!context) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tooling_value_io_stream_list_t* input_streams = context->input_streams;
  iree_allocator_t host_allocator = input_streams->host_allocator;
  iree_tooling_value_io_stream_list_free(input_streams);
  iree_allocator_free(host_allocator, context);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tooling_buffer_view_file_read_callback(
    iree_hal_buffer_mapping_t* mapping, void* user_data) {
  iree_io_stream_t* stream = (iree_io_stream_t*)user_data;
  return iree_io_stream_read(stream, mapping->contents.data_length,
                             mapping->contents.data,
                             /*out_buffer_length=*/NULL);
}

static iree_status_t iree_tooling_buffer_view_parse_binary_file(
    iree_tooling_value_io_context_t* context, iree_string_view_t metadata,
    iree_string_view_t contents, iree_hal_device_t* device,
    iree_hal_allocator_t* device_allocator,
    iree_hal_buffer_view_t** out_buffer_view) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, contents.data, contents.size);

  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  iree_host_size_t shape_rank = 0;
  iree_hal_dim_t shape[128] = {0};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_parse_shape_and_element_type(
                                            metadata, IREE_ARRAYSIZE(shape),
                                            &shape_rank, shape, &element_type));
  iree_hal_encoding_type_t encoding_type =
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR;

  bool is_append = !iree_string_view_starts_with(contents, IREE_SV("@"));
  iree_string_view_t path =
      iree_string_view_substr(contents, 1, IREE_HOST_SIZE_MAX);
  iree_io_stream_t* stream = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tooling_value_io_stream_list_open(context->input_streams, path,
                                                 is_append, &stream));

  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree_status_t status = iree_hal_buffer_view_generate_buffer(
      device, device_allocator, shape_rank, shape, element_type, encoding_type,
      buffer_params, iree_tooling_buffer_view_file_read_callback, stream,
      out_buffer_view);

  iree_io_stream_release(stream);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_tooling_buffer_view_parse_numpy_file(
    iree_tooling_value_io_context_t* context, iree_string_view_t spec,
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_hal_buffer_view_t** out_buffer_view) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, spec.data, spec.size);

  bool is_append = !iree_string_view_starts_with(spec, IREE_SV("@"));
  iree_string_view_t path =
      iree_string_view_substr(spec, 1, IREE_HOST_SIZE_MAX);
  if (!iree_string_view_ends_with(path, IREE_SV(".npy"))) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                             "only numpy (.npy) files are supported for "
                             "metadata-less buffer view I/O"));
  }

  iree_io_stream_t* stream = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tooling_value_io_stream_list_open(context->input_streams, path,
                                                 is_append, &stream));

  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_READ,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree_status_t status = iree_numpy_npy_load_ndarray(
      stream, IREE_NUMPY_NPY_LOAD_OPTION_DEFAULT, buffer_params, device,
      device_allocator, out_buffer_view);

  iree_io_stream_release(stream);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_tooling_buffer_view_spec_parse(
    iree_tooling_value_io_context_t* context, iree_string_view_t spec,
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_hal_buffer_view_t** out_buffer_view) {
  IREE_ASSERT_ARGUMENT(out_buffer_view);
  *out_buffer_view = NULL;
  if (iree_string_view_starts_with_char(spec, '&')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer view specification `%.*s` must not use "
                            "storage-buffer prefix `&`",
                            (int)spec.size, spec.data);
  }
  if (iree_string_view_starts_with_char(spec, '*')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer view specification `%.*s` expands to "
                            "multiple values",
                            (int)spec.size, spec.data);
  }
  if (iree_string_view_starts_with_char(spec, '@') ||
      iree_string_view_starts_with_char(spec, '+')) {
    return iree_tooling_buffer_view_parse_numpy_file(
        context, spec, device, device_allocator, out_buffer_view);
  }

  iree_string_view_t metadata = iree_string_view_empty();
  iree_string_view_t contents = iree_string_view_empty();
  if (iree_string_view_split(spec, '=', &metadata, &contents) != -1 &&
      (iree_string_view_starts_with_char(contents, '@') ||
       iree_string_view_starts_with_char(contents, '+'))) {
    return iree_tooling_buffer_view_parse_binary_file(
        context, metadata, contents, device, device_allocator, out_buffer_view);
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, spec.data, spec.size);
  iree_status_t status = iree_hal_buffer_view_parse(
      spec, device, device_allocator, out_buffer_view);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_tooling_storage_buffer_spec_parse(
    iree_tooling_value_io_context_t* context, iree_string_view_t spec,
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  if (!iree_string_view_consume_prefix_char(&spec, '&')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "storage-buffer specification must start with `&`");
  }

  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(iree_tooling_buffer_view_spec_parse(
      context, spec, device, device_allocator, &buffer_view));
  iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_view);
  iree_hal_buffer_retain(buffer);
  iree_hal_buffer_view_release(buffer_view);
  *out_buffer = buffer;
  return iree_ok_status();
}

IREE_API_EXPORT void iree_tooling_buffer_binding_deinitialize(
    iree_tooling_buffer_binding_t* binding) {
  if (!binding) return;
  iree_hal_buffer_release(binding->buffer);
  iree_hal_buffer_view_release(binding->buffer_view);
  memset(binding, 0, sizeof(*binding));
}

IREE_API_EXPORT iree_status_t iree_tooling_buffer_binding_spec_parse(
    iree_tooling_value_io_context_t* context, iree_string_view_t spec,
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_tooling_buffer_binding_t* out_binding) {
  IREE_ASSERT_ARGUMENT(out_binding);
  memset(out_binding, 0, sizeof(*out_binding));
  if (iree_string_view_starts_with_char(spec, '&')) {
    iree_string_view_t storage_spec = spec;
    iree_string_view_consume_prefix_char(&storage_spec, '&');
    iree_hal_buffer_view_t* buffer_view = NULL;
    IREE_RETURN_IF_ERROR(iree_tooling_buffer_view_spec_parse(
        context, storage_spec, device, device_allocator, &buffer_view));
    iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_view);
    iree_hal_buffer_retain(buffer);
    out_binding->kind = IREE_TOOLING_BUFFER_BINDING_KIND_STORAGE_BUFFER;
    out_binding->buffer = buffer;
    out_binding->byte_offset = 0;
    out_binding->byte_length = iree_hal_buffer_view_byte_length(buffer_view);
    iree_hal_buffer_view_release(buffer_view);
    return iree_ok_status();
  }

  iree_hal_buffer_view_t* buffer_view = NULL;
  IREE_RETURN_IF_ERROR(iree_tooling_buffer_view_spec_parse(
      context, spec, device, device_allocator, &buffer_view));
  iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_view);
  iree_hal_buffer_retain(buffer);
  out_binding->kind = IREE_TOOLING_BUFFER_BINDING_KIND_BUFFER_VIEW;
  out_binding->buffer = buffer;
  out_binding->buffer_view = buffer_view;
  out_binding->byte_offset = 0;
  out_binding->byte_length = iree_hal_buffer_view_byte_length(buffer_view);
  return iree_ok_status();
}
