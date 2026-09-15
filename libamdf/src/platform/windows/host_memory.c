// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/host_memory.h"

#include <emmintrin.h>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "libamdf/src/platform/windows/host_cache.h"

amdf_status_t amdf_platform_host_memory_allocate(uint64_t byte_length,
                                                 void** out_pointer) {
  void* pointer = VirtualAlloc(NULL, (SIZE_T)byte_length,
                               MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (pointer == NULL)
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  *out_pointer = pointer;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_platform_host_memory_free(void* pointer,
                                             uint64_t byte_length) {
  (void)byte_length;
  return VirtualFree(pointer, 0, MEM_RELEASE)
             ? AMDF_STATUS_OK
             : amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
}

amdf_status_t amdf_platform_host_memory_query_cache_line_size(
    uint32_t* out_line_size) {
  *out_line_size = amdf_windows_host_cache_line_size();
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_platform_host_memory_cache_control(void* pointer,
                                                      uint64_t byte_length,
                                                      uint32_t line_size) {
  (void)line_size;
  _mm_mfence();
  return amdf_windows_host_cache_control(AMDF_HOST_CACHE_OPERATION_FLUSH,
                                         pointer, byte_length);
}
