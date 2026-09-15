// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cpuid.h>
#include <emmintrin.h>
#include <stdint.h>

#include "libamdf/src/platform/linux/host_cache.h"

amdf_status_t amdf_linux_host_cache_query_line_size(uint32_t* out_line_size) {
  unsigned int eax, ebx, ecx, edx;
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx) || !(edx & (1u << 19))) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const uint32_t line_size = ((ebx >> 8) & 0xFFu) * 8;
  if (line_size == 0 || (line_size & (line_size - 1)) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  *out_line_size = line_size;
  return AMDF_STATUS_OK;
}

void amdf_linux_host_cache_transfer(void* pointer, uint64_t byte_length,
                                    uint32_t line_size) {
  if (byte_length == 0) return;
  const uintptr_t first = (uintptr_t)pointer & ~((uintptr_t)line_size - 1);
  const uintptr_t last =
      ((uintptr_t)pointer + byte_length - 1) & ~((uintptr_t)line_size - 1);
  _mm_mfence();
  uintptr_t line = first;
  while (line != last) {
    _mm_clflush((const void*)line);
    line += line_size;
  }
  _mm_clflush((const void*)last);
  _mm_mfence();
}
