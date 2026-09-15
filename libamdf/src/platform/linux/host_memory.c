// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/platform/host_memory.h"

#include <sys/mman.h>

#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/platform/linux/host_cache.h"

amdf_status_t amdf_platform_host_memory_allocate(uint64_t byte_length,
                                                 void** out_pointer) {
  void* pointer = mmap(NULL, (size_t)byte_length, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (pointer == MAP_FAILED) return amdf_linux_error(errno);
  *out_pointer = pointer;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_platform_host_memory_free(void* pointer,
                                             uint64_t byte_length) {
  return munmap(pointer, (size_t)byte_length) == 0 ? AMDF_STATUS_OK
                                                   : amdf_linux_error(errno);
}

amdf_status_t amdf_platform_host_memory_query_cache_line_size(
    uint32_t* out_line_size) {
  return amdf_linux_host_cache_query_line_size(out_line_size);
}

amdf_status_t amdf_platform_host_memory_cache_control(void* pointer,
                                                      uint64_t byte_length,
                                                      uint32_t line_size) {
  amdf_linux_host_cache_transfer(pointer, byte_length, line_size);
  return AMDF_STATUS_OK;
}
