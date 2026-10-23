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
#include <stdio.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/utsname.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/host_reservation.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/platform/linux/host_cache.h"

// Stable Linux UAPI value, available at runtime since Linux 5.14. Naming it
// locally lets builds use older sysroot headers without changing admission on
// kernels that do not support fallible writable population.
enum {
  AMDF_LINUX_MADV_POPULATE_WRITE = 23,
  // Synchronous huge-folio construction, available since Linux 6.1.
  AMDF_LINUX_MADV_COLLAPSE = 25,
  // PMD geometry on the supported Linux x86-64 host platform.
  AMDF_LINUX_HOST_FOLIO_SIZE = 2u << 20,
};

static amdf_status_t amdf_linux_host_backing_map(int descriptor,
                                                 uint64_t byte_length,
                                                 void** out_pointer) {
  if (byte_length < AMDF_LINUX_HOST_FOLIO_SIZE) {
    void* pointer = mmap(NULL, (size_t)byte_length, PROT_READ | PROT_WRITE,
                         MAP_SHARED, descriptor, 0);
    if (pointer == MAP_FAILED) {
      return amdf_linux_error(errno);
    }
    *out_pointer = pointer;
    return AMDF_STATUS_OK;
  }
  void* pointer = NULL;
  amdf_status_t status = amdf_platform_host_reservation_create(
      byte_length, AMDF_LINUX_HOST_FOLIO_SIZE, &pointer);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_platform_host_reservation_replace(
      pointer, byte_length, 0, (uintptr_t)descriptor, 0,
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
  if (amdf_status_is_ok(status)) {
    *out_pointer = pointer;
  } else {
    const amdf_status_t cleanup =
        amdf_platform_host_reservation_destroy(pointer, byte_length);
    if (!amdf_status_is_ok(cleanup)) {
      status = cleanup;
    }
  }
  return status;
}

amdf_status_t amdf_platform_host_backing_prepare(
    uint64_t byte_length, amdf_platform_host_backing_t* backing) {
  *backing = (amdf_platform_host_backing_t){.handle = UINTPTR_MAX};
  const int descriptor =
      memfd_create("libamdf", MFD_CLOEXEC | MFD_ALLOW_SEALING);
  if (descriptor < 0) {
    return amdf_linux_error(errno);
  }
  backing->handle = (uintptr_t)descriptor;
  amdf_status_t status = AMDF_STATUS_OK;
  void* pointer = MAP_FAILED;
  if (ftruncate(descriptor, (off_t)byte_length) != 0) {
    status = amdf_linux_error(errno);
  } else if (fcntl(descriptor, F_ADD_SEALS,
                   F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL) != 0) {
    status = amdf_linux_error(errno);
  } else {
    status = amdf_linux_host_backing_map(descriptor, byte_length, &pointer);
  }
  if (amdf_status_is_ok(status)) {
    backing->pointer = pointer;
  }
  return status;
}

amdf_status_t amdf_platform_host_backing_destroy(
    amdf_platform_host_backing_t* backing, uint64_t byte_length) {
  if (backing->pointer != NULL) {
    if (munmap(backing->pointer, (size_t)byte_length) != 0) {
      return amdf_linux_error(errno);
    }
    backing->pointer = NULL;
  }
  if (backing->handle != UINTPTR_MAX) {
    int descriptor = (int)backing->handle;
    // Linux consumes a valid descriptor even when close reports an error.
    backing->handle = UINTPTR_MAX;
    return amdf_linux_file_close(&descriptor);
  }
  return AMDF_STATUS_OK;
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

static amdf_status_t amdf_linux_host_memory_bind(
    void* pointer, uint64_t byte_length, amdf_memory_placement_t placement,
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
  }
  amdf_free(host_allocator, nodes);
  return status;
}

amdf_status_t amdf_platform_host_memory_place(void* pointer,
                                              uint64_t byte_length,
                                              amdf_memory_placement_t placement,
                                              amdf_allocator_t host_allocator) {
  amdf_status_t status = amdf_linux_host_memory_bind(pointer, byte_length,
                                                     placement, host_allocator);
  if (amdf_status_is_ok(status) &&
      madvise(pointer, (size_t)byte_length, AMDF_LINUX_MADV_POPULATE_WRITE) !=
          0) {
    // Population has an error channel; ordinary first-touch stores could
    // instead fault the process when the selected node cannot supply pages.
    status = amdf_linux_error(errno);
  }
  return status;
}

amdf_status_t amdf_platform_host_backing_populate(
    const amdf_platform_host_backing_t* backing, uint64_t byte_length,
    amdf_memory_placement_t placement, amdf_allocator_t host_allocator) {
  // Before Linux 7.2, udmabuf constructs one scatterlist entry per base page
  // with a folio-relative offset. DMA importers count that offset as additional
  // pages, producing repeated/wrong GPU mappings. Restrict this fresh export
  // backing before population; suppressing only MADV_COLLAPSE would still
  // permit transparent promotion before export pins the pages.
  struct utsname kernel;
  if (uname(&kernel) != 0) {
    return amdf_linux_error(errno);
  }
  unsigned int major = 0, minor = 0;
  const bool large_pages =
      sscanf(kernel.release, "%u.%u", &major, &minor) == 2 &&
      (major > 7 || (major == 7 && minor >= 2));
  if (!large_pages &&
      madvise(backing->pointer, (size_t)byte_length, MADV_NOHUGEPAGE) != 0) {
    return amdf_linux_error(errno);
  }
  amdf_status_t status = AMDF_STATUS_OK;
  if (placement.kind != AMDF_MEMORY_PLACEMENT_KIND_NONE) {
    status = amdf_linux_host_memory_bind(backing->pointer, byte_length,
                                         placement, host_allocator);
  }
  const uint64_t folio_length =
      large_pages ? byte_length & ~((uint64_t)AMDF_LINUX_HOST_FOLIO_SIZE - 1)
                  : 0;
  const long page_size = sysconf(_SC_PAGESIZE);
  if (amdf_status_is_ok(status) && page_size <= 0) {
    status = amdf_linux_error(EINVAL);
  }
  // Collapse needs a resident page in each folio. Seeding only one base page
  // avoids constructing and then replacing every base-page allocation. NUMA
  // binding precedes both the seeds and the folios built from them.
  for (uint64_t offset = 0; amdf_status_is_ok(status) && offset < folio_length;
       offset += AMDF_LINUX_HOST_FOLIO_SIZE) {
    if (madvise((uint8_t*)backing->pointer + offset, (size_t)page_size,
                AMDF_LINUX_MADV_POPULATE_WRITE) != 0) {
      status = amdf_linux_error(errno);
    }
  }
  if (amdf_status_is_ok(status) && folio_length != 0) {
    // Promotion is an optimization, not an allocation requirement. A native
    // refusal may leave any subset promoted; the mandatory population below
    // establishes every writable translation regardless of page size.
    (void)madvise(backing->pointer, (size_t)folio_length,
                  AMDF_LINUX_MADV_COLLAPSE);
  }
  // Install writable CPU translations for the entire extent, including any
  // sub-folio tail, before export pins pages and cache publication reads them.
  if (amdf_status_is_ok(status) &&
      madvise(backing->pointer, (size_t)byte_length,
              AMDF_LINUX_MADV_POPULATE_WRITE) != 0) {
    status = amdf_linux_error(errno);
  }
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
