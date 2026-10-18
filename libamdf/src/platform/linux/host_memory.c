// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/platform/host_memory.h"

#include <fcntl.h>
#include <limits.h>
#include <linux/mempolicy.h>
#include <sys/mman.h>
#include <sys/syscall.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/platform/linux/host_cache.h"

// Stable Linux UAPI value, available at runtime since Linux 5.14. Naming it
// locally lets builds use older sysroot headers without changing admission on
// kernels that do not support fallible writable population.
enum { AMDF_LINUX_MADV_POPULATE_WRITE = 23 };

amdf_status_t amdf_platform_host_backing_create(
    uint64_t byte_length, amdf_platform_host_backing_t* out_backing) {
  int descriptor = memfd_create("libamdf", MFD_CLOEXEC | MFD_ALLOW_SEALING);
  if (descriptor < 0) {
    return amdf_linux_error(errno);
  }
  amdf_status_t status = AMDF_STATUS_OK;
  void* pointer = MAP_FAILED;
  if (ftruncate(descriptor, (off_t)byte_length) != 0) {
    status = amdf_linux_error(errno);
  } else if (fcntl(descriptor, F_ADD_SEALS,
                   F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL) != 0) {
    status = amdf_linux_error(errno);
  } else {
    pointer = mmap(NULL, (size_t)byte_length, PROT_READ | PROT_WRITE,
                   MAP_SHARED, descriptor, 0);
    if (pointer == MAP_FAILED) {
      status = amdf_linux_error(errno);
    }
  }
  if (amdf_status_is_ok(status)) {
    *out_backing = (amdf_platform_host_backing_t){
        .handle = (uintptr_t)descriptor,
        .pointer = pointer,
    };
  } else {
    const amdf_status_t cleanup = amdf_linux_file_close(&descriptor);
    if (!amdf_status_is_ok(cleanup)) {
      status = cleanup;
    }
  }
  return status;
}

amdf_status_t amdf_platform_host_backing_destroy(
    amdf_platform_host_backing_t* backing, uint64_t byte_length) {
  if (munmap(backing->pointer, (size_t)byte_length) != 0) {
    return amdf_linux_error(errno);
  }
  int descriptor = (int)backing->handle;
  return amdf_linux_file_close(&descriptor);
}

amdf_status_t amdf_platform_host_memory_allocate(uint64_t byte_length,
                                                 void** out_pointer) {
  void* pointer = mmap(NULL, (size_t)byte_length, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (pointer == MAP_FAILED) {
    return amdf_linux_error(errno);
  }
  *out_pointer = pointer;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_platform_host_memory_free(void* pointer,
                                             uint64_t byte_length) {
  return munmap(pointer, (size_t)byte_length) == 0 ? AMDF_STATUS_OK
                                                   : amdf_linux_error(errno);
}

amdf_memory_placement_kinds_t amdf_platform_host_memory_placement_kinds(void) {
  return UINT32_C(1) << AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA;
}

amdf_status_t amdf_platform_host_memory_place(void* pointer,
                                              uint64_t byte_length,
                                              amdf_memory_placement_t placement,
                                              amdf_allocator_t host_allocator) {
  const uint64_t bit_count = (uint64_t)placement.node + 1;
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return amdf_linux_error(EINVAL);
  }
  // Linux get_nodes rejects bitmaps larger than one native page.
  if (bit_count > (uint64_t)page_size * CHAR_BIT) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const size_t word_bits = sizeof(unsigned long) * CHAR_BIT;
  const size_t word_count = (size_t)((bit_count + word_bits - 1) / word_bits);
  unsigned long* nodes = NULL;
  amdf_status_t status =
      amdf_calloc_array(host_allocator, word_count, sizeof(*nodes),
                        amdf_alignof(unsigned long), (void**)&nodes);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  nodes[placement.node / word_bits] = 1UL << (placement.node % word_bits);
  // Linux get_nodes subtracts one from maxnode before copying the bitmap.
  if (syscall(SYS_mbind, pointer, (size_t)byte_length,
              MPOL_BIND | MPOL_F_STATIC_NODES, nodes,
              (unsigned long)bit_count + 1, 0UL) != 0) {
    status = amdf_linux_error(errno);
  } else if (madvise(pointer, (size_t)byte_length,
                     AMDF_LINUX_MADV_POPULATE_WRITE) != 0) {
    // Population has an error channel; ordinary first-touch stores could
    // instead fault the process when the selected node cannot supply pages.
    status = amdf_linux_error(errno);
  }
  amdf_free(host_allocator, nodes);
  return status;
}

amdf_status_t amdf_platform_host_memory_query_cache_transition(
    amdf_cache_transition_t* out_transition) {
  uint32_t line_size = 0;
  const amdf_status_t status =
      amdf_linux_host_cache_query_line_size(&line_size);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  *out_transition = (amdf_cache_transition_t){
      .kind = AMDF_CACHE_TRANSITION_KIND_RANGE,
      .executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT,
      .host_operation = AMDF_HOST_CACHE_OPERATION_FLUSH,
      .host_instruction = AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH,
      .host_fence_before = AMDF_HOST_CACHE_FENCE_X86_MFENCE,
      .host_fence_after = AMDF_HOST_CACHE_FENCE_X86_MFENCE,
      .range_granularity = line_size,
  };
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_platform_host_memory_cache_control(void* pointer,
                                                      uint64_t byte_length,
                                                      uint32_t line_size) {
  amdf_linux_host_cache_transfer(pointer, byte_length, line_size);
  return AMDF_STATUS_OK;
}
