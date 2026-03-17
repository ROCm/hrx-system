// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/testing/mock_executable_cache.h"

#include <string.h>

#include "iree/hal/testing/mock_executable.h"

static const iree_hal_executable_cache_vtable_t
    iree_hal_mock_executable_cache_vtable;

typedef struct iree_hal_mock_executable_cache_t {
  // Base HAL resource header.
  iree_hal_resource_t resource;
  // Allocator used to free the executable cache.
  iree_allocator_t host_allocator;
  // Number of entries in supported_formats.
  iree_host_size_t supported_format_count;
  // Supported format strings copied into trailing storage.
  iree_string_view_t* supported_formats;
} iree_hal_mock_executable_cache_t;

static iree_hal_mock_executable_cache_t* iree_hal_mock_executable_cache_cast(
    iree_hal_executable_cache_t* base_cache) {
  IREE_HAL_ASSERT_TYPE(base_cache, &iree_hal_mock_executable_cache_vtable);
  return (iree_hal_mock_executable_cache_t*)base_cache;
}

static void iree_hal_mock_executable_cache_destroy(
    iree_hal_executable_cache_t* base_cache) {
  iree_hal_mock_executable_cache_t* cache =
      iree_hal_mock_executable_cache_cast(base_cache);
  iree_allocator_t host_allocator = cache->host_allocator;
  iree_allocator_free(host_allocator, cache);
}

static iree_status_t iree_hal_mock_executable_cache_infer_format(
    iree_hal_executable_cache_t* base_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  iree_hal_mock_executable_cache_t* cache =
      iree_hal_mock_executable_cache_cast(base_cache);
  (void)caching_mode;
  if (cache->supported_format_count == 0) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "mock cache has no supported formats");
  }

  iree_string_view_t inferred_format = cache->supported_formats[0];
  if (inferred_format.size >= executable_format_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable format buffer too small");
  }
  memcpy(executable_format, inferred_format.data, inferred_format.size);
  executable_format[inferred_format.size] = 0;
  *out_inferred_size = executable_data.data_length;
  return iree_ok_status();
}

static bool iree_hal_mock_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* base_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  iree_hal_mock_executable_cache_t* cache =
      iree_hal_mock_executable_cache_cast(base_cache);
  (void)caching_mode;
  for (iree_host_size_t i = 0; i < cache->supported_format_count; ++i) {
    if (iree_string_view_equal(cache->supported_formats[i],
                               executable_format)) {
      return true;
    }
  }
  return false;
}

static iree_status_t iree_hal_mock_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* base_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  iree_hal_mock_executable_cache_t* cache =
      iree_hal_mock_executable_cache_cast(base_cache);
  if (!iree_hal_mock_executable_cache_can_prepare_format(
          base_cache, executable_params->caching_mode,
          executable_params->executable_format)) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "mock cache does not support format '%.*s'",
                            (int)executable_params->executable_format.size,
                            executable_params->executable_format.data);
  }
  return iree_hal_mock_executable_create(executable_params,
                                         cache->host_allocator, out_executable);
}

iree_status_t iree_hal_mock_executable_cache_create(
    iree_string_view_t identifier, const iree_string_view_t* supported_formats,
    iree_host_size_t supported_format_count, iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = NULL;
  (void)identifier;

  iree_host_size_t total_format_length = 0;
  for (iree_host_size_t i = 0; i < supported_format_count; ++i) {
    if (IREE_UNLIKELY(!iree_host_size_checked_add(total_format_length,
                                                  supported_formats[i].size,
                                                  &total_format_length))) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "mock executable format strings are too large");
    }
  }

  iree_host_size_t supported_formats_offset = 0;
  iree_host_size_t format_storage_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_mock_executable_cache_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(supported_format_count, iree_string_view_t,
                                iree_alignof(iree_string_view_t),
                                &supported_formats_offset),
      IREE_STRUCT_FIELD(total_format_length, char, &format_storage_offset)));

  iree_hal_mock_executable_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&cache));
  memset(cache, 0, total_size);

  iree_hal_resource_initialize(&iree_hal_mock_executable_cache_vtable,
                               &cache->resource);
  cache->host_allocator = host_allocator;
  cache->supported_format_count = supported_format_count;
  cache->supported_formats =
      (iree_string_view_t*)((char*)cache + supported_formats_offset);

  char* format_storage = (char*)cache + format_storage_offset;
  for (iree_host_size_t i = 0; i < supported_format_count; ++i) {
    memcpy(format_storage, supported_formats[i].data,
           supported_formats[i].size);
    cache->supported_formats[i] =
        iree_make_string_view(format_storage, supported_formats[i].size);
    format_storage += supported_formats[i].size;
  }

  *out_executable_cache = (iree_hal_executable_cache_t*)cache;
  return iree_ok_status();
}

static const iree_hal_executable_cache_vtable_t
    iree_hal_mock_executable_cache_vtable = {
        .destroy = iree_hal_mock_executable_cache_destroy,
        .infer_format = iree_hal_mock_executable_cache_infer_format,
        .can_prepare_format = iree_hal_mock_executable_cache_can_prepare_format,
        .prepare_executable = iree_hal_mock_executable_cache_prepare_executable,
};
