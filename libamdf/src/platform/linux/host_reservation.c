// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/platform/host_reservation.h"

#include <inttypes.h>
#include <stdio.h>
#include <sys/mman.h>

#include "libamdf/src/platform/linux/file.h"

amdf_status_t amdf_platform_host_reservation_query_capabilities(
    amdf_memory_reservation_capabilities_t* out_capabilities) {
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  *out_capabilities = (amdf_memory_reservation_capabilities_t){
      .supported_access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .required_access = AMDF_MEMORY_ACCESS_READ,
      .address_domain_ordinal = AMDF_ADDRESS_DOMAIN_ORDINAL_NONE,
      .supported_flags = AMDF_MEMORY_RESERVATION_FLAG_REQUIRE_COMMON_ADDRESS,
      .maximum_byte_length = (SIZE_MAX / 2) & ~((uint64_t)page_size - 1),
      .reservation_granularity = (uint64_t)page_size,
      .maximum_alignment = (SIZE_MAX / 2) + 1,
      .mapping_granularity = (uint64_t)page_size,
      .minimum_address = (uint64_t)page_size,
      .maximum_address = UINTPTR_MAX,
      .required_source_flags = AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE,
  };
  return AMDF_STATUS_OK;
}

// Linux mmap has no upper-bound argument. When its default placement lies
// outside a required device window, find a gap in the process address map and
// claim it without replacing any mapping that raced the snapshot.
static amdf_status_t amdf_linux_host_reservation_create_bounded(
    uint64_t byte_length, uint64_t alignment, uint64_t minimum_address,
    uint64_t maximum_address, void** out_pointer) {
  if (alignment == 0) {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    alignment = (uint64_t)page_size;
  }
  if (alignment - 1 > maximum_address - minimum_address) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  FILE* maps = fopen("/proc/self/maps", "re");
  if (maps == NULL) {
    return amdf_linux_error(errno);
  }
  uint64_t candidate = (minimum_address + alignment - 1) & ~(alignment - 1);
  uint64_t begin = 0;
  uint64_t end = 0;
  int fields = 0;
  while ((fields = fscanf(maps, "%" SCNx64 "-%" SCNx64 "%*[^\n]\n", &begin,
                          &end)) == 2) {
    if (candidate > maximum_address ||
        byte_length - 1 > maximum_address - candidate ||
        (begin > candidate && byte_length <= begin - candidate)) {
      break;
    }
    if (end > candidate) {
      if (end > maximum_address || alignment - 1 > maximum_address - end) {
        candidate = 0;
        break;
      }
      candidate = (end + alignment - 1) & ~(alignment - 1);
    }
  }
  amdf_status_t status = AMDF_STATUS_OK;
  if (ferror(maps)) {
    status = amdf_linux_error(errno != 0 ? errno : EIO);
  } else if (fields != EOF && fields != 2) {
    status = amdf_linux_error(EPROTO);
  }
  if (fclose(maps) != 0 && amdf_status_is_ok(status)) {
    status = amdf_linux_error(errno);
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (candidate == 0 || candidate > maximum_address ||
      byte_length - 1 > maximum_address - candidate) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  void* pointer =
      mmap((void*)(uintptr_t)candidate, (size_t)byte_length, PROT_NONE,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
  if (pointer == MAP_FAILED) {
    return errno == EEXIST
               ? amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED)
               : amdf_linux_error(errno);
  }
  if ((uintptr_t)pointer != candidate) {
    // Older kernels can ignore an unknown non-replacing placement flag.
    const amdf_status_t unsupported =
        amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    return munmap(pointer, (size_t)byte_length) == 0 ? unsupported
                                                     : amdf_linux_error(errno);
  }
  *out_pointer = pointer;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_platform_host_reservation_create(uint64_t byte_length,
                                                    uint64_t minimum_alignment,
                                                    uint64_t minimum_address,
                                                    uint64_t maximum_address,
                                                    void** out_pointer) {
  // The public envelope bounds both operands to half the host address range.
  const size_t length = (size_t)(byte_length + minimum_alignment);
  void* storage =
      mmap(NULL, length, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (storage == MAP_FAILED) {
    return amdf_linux_error(errno);
  }
  const uintptr_t base = (uintptr_t)storage;
  const uintptr_t aligned =
      minimum_alignment != 0
          ? (base + minimum_alignment - 1) & ~(uintptr_t)(minimum_alignment - 1)
          : base;
  if (aligned < minimum_address || aligned > maximum_address ||
      byte_length - 1 > maximum_address - aligned) {
    if (munmap(storage, length) != 0) {
      return amdf_linux_error(errno);
    }
    return amdf_linux_host_reservation_create_bounded(
        byte_length, minimum_alignment, minimum_address, maximum_address,
        out_pointer);
  }
  const size_t prefix = (size_t)(aligned - base);
  const size_t suffix = length - prefix - (size_t)byte_length;
  if (prefix != 0 && munmap(storage, prefix) != 0) {
    amdf_status_t status = amdf_linux_error(errno);
    if (munmap(storage, length) != 0) {
      status = amdf_linux_error(errno);
    }
    return status;
  }
  if (suffix != 0 && munmap((void*)(aligned + byte_length), suffix) != 0) {
    amdf_status_t status = amdf_linux_error(errno);
    if (munmap((void*)aligned, length - prefix) != 0) {
      status = amdf_linux_error(errno);
    }
    return status;
  }
  *out_pointer = (void*)aligned;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_platform_host_reservation_replace(
    void* pointer, uint64_t byte_length, uint64_t mapping_granularity,
    uintptr_t source, uint64_t source_byte_offset,
    amdf_memory_access_t access) {
  (void)mapping_granularity;
  const int protection =
      PROT_READ | ((access & AMDF_MEMORY_ACCESS_WRITE) != 0 ? PROT_WRITE : 0);
  return mmap(pointer, (size_t)byte_length, protection, MAP_SHARED | MAP_FIXED,
              (int)source, (off_t)source_byte_offset) != MAP_FAILED
             ? AMDF_STATUS_OK
             : amdf_linux_error(errno);
}

amdf_status_t amdf_platform_host_reservation_clear(void* pointer,
                                                   uint64_t byte_length) {
  return mmap(pointer, (size_t)byte_length, PROT_NONE,
              MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) != MAP_FAILED
             ? AMDF_STATUS_OK
             : amdf_linux_error(errno);
}

amdf_status_t amdf_platform_host_reservation_destroy(void* pointer,
                                                     uint64_t byte_length) {
  return munmap(pointer, (size_t)byte_length) == 0 ? AMDF_STATUS_OK
                                                   : amdf_linux_error(errno);
}
