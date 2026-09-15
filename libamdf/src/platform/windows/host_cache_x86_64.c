// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <emmintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "libamdf/src/platform/windows/host_cache.h"

#if !defined(_M_X64) && !defined(__x86_64__)
#error "The Windows host-cache implementation requires an x86-64 host"
#endif

// AMD64 cache lines are 64 bytes on every Windows host supported by this leaf.
#define AMDF_WINDOWS_HOST_CACHE_LINE_SIZE 64u

uint32_t amdf_windows_host_cache_line_size(void) {
  return AMDF_WINDOWS_HOST_CACHE_LINE_SIZE;
}

amdf_status_t amdf_windows_host_cache_control(
    amdf_host_cache_operation_t operation, void* pointer,
    uint64_t byte_length) {
  if ((operation != AMDF_HOST_CACHE_OPERATION_FLUSH &&
       operation != AMDF_HOST_CACHE_OPERATION_INVALIDATE) ||
      (pointer == NULL && byte_length != 0)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (byte_length == 0) {
    return AMDF_STATUS_OK;
  }
  const uintptr_t first_byte = (uintptr_t)pointer;
  if (byte_length > UINTPTR_MAX - first_byte) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const uintptr_t last_byte = first_byte + (uintptr_t)byte_length;
  const uintptr_t first_line =
      first_byte & ~(uintptr_t)(AMDF_WINDOWS_HOST_CACHE_LINE_SIZE - 1);
  if (last_byte > UINTPTR_MAX - (AMDF_WINDOWS_HOST_CACHE_LINE_SIZE - 1)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const uintptr_t last_line =
      (last_byte + (AMDF_WINDOWS_HOST_CACHE_LINE_SIZE - 1)) &
      ~(uintptr_t)(AMDF_WINDOWS_HOST_CACHE_LINE_SIZE - 1);
  for (uintptr_t line = first_line; line < last_line;
       line += AMDF_WINDOWS_HOST_CACHE_LINE_SIZE) {
    _mm_clflush((const void*)line);
  }
  _mm_mfence();
  return AMDF_STATUS_OK;
}
