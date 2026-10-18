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

amdf_status_t amdf_platform_host_backing_create(
    uint64_t byte_length, amdf_platform_host_backing_t* out_backing) {
  HANDLE section =
      CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
                         (DWORD)(byte_length >> 32), (DWORD)byte_length, NULL);
  if (section == NULL) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  void* pointer = MapViewOfFile(section, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0,
                                (SIZE_T)byte_length);
  if (pointer == NULL) {
    amdf_status_t status =
        amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
    if (!CloseHandle(section)) {
      status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
    }
    return status;
  }
  *out_backing = (amdf_platform_host_backing_t){
      .handle = (uintptr_t)section,
      .pointer = pointer,
  };
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_platform_host_backing_destroy(
    amdf_platform_host_backing_t* backing, uint64_t byte_length) {
  (void)byte_length;
  if (!UnmapViewOfFile(backing->pointer)) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  return CloseHandle((HANDLE)backing->handle)
             ? AMDF_STATUS_OK
             : amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
}

amdf_status_t amdf_platform_host_backing_prepare(
    const amdf_platform_host_backing_t* backing, uint64_t byte_length,
    amdf_memory_placement_t placement, amdf_allocator_t host_allocator) {
  (void)backing;
  (void)byte_length;
  (void)placement;
  (void)host_allocator;
  // Platform-owned host exports are unavailable; WDDM owns its file backing.
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_status_t amdf_platform_host_memory_allocate(uint64_t byte_length,
                                                 void** out_pointer) {
  void* pointer = VirtualAlloc(NULL, (SIZE_T)byte_length,
                               MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (pointer == NULL) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
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

amdf_memory_placement_kinds_t amdf_platform_host_memory_placement_kinds(void) {
  // VirtualAllocExNuma and MapViewOfFileExNuma express preferences and permit
  // fallback. They cannot establish a required physical placement contract.
  return 0;
}

amdf_status_t amdf_platform_host_memory_place(void* pointer,
                                              uint64_t byte_length,
                                              amdf_memory_placement_t placement,
                                              amdf_allocator_t host_allocator) {
  (void)pointer;
  (void)byte_length;
  (void)placement;
  (void)host_allocator;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_status_t amdf_platform_host_memory_query_cache_transition(
    amdf_cache_transition_t* out_transition) {
  *out_transition = (amdf_cache_transition_t){
      .kind = AMDF_CACHE_TRANSITION_KIND_RANGE,
      .executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT,
      .host_operation = AMDF_HOST_CACHE_OPERATION_FLUSH,
      .host_instruction = AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH,
      .host_fence_before = AMDF_HOST_CACHE_FENCE_X86_MFENCE,
      .host_fence_after = AMDF_HOST_CACHE_FENCE_X86_MFENCE,
      .range_granularity = amdf_windows_host_cache_line_size(),
  };
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
