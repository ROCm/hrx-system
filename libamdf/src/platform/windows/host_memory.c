// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/host_memory.h"

#include <emmintrin.h>
#define NTDDI_VERSION NTDDI_WIN10_RS4
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "libamdf/src/platform/host_reservation.h"
#include "libamdf/src/platform/windows/host_cache.h"

amdf_status_t amdf_platform_host_memory_sample_observation(
    amdf_memory_scope_observation_t* out_observation) {
  MEMORYSTATUSEX memory = {.dwLength = sizeof(memory)};
  if (!GlobalMemoryStatusEx(&memory)) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  out_observation->capacity_byte_length = memory.ullTotalPhys;
  out_observation->available_byte_length = memory.ullAvailPhys;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_platform_host_backing_prepare(
    uint64_t byte_length, uint64_t minimum_alignment,
    amdf_platform_host_backing_t* backing) {
  *backing = (amdf_platform_host_backing_t){.handle = UINTPTR_MAX};
  HANDLE section =
      CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
                         (DWORD)(byte_length >> 32), (DWORD)byte_length, NULL);
  if (section == NULL) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  backing->handle = (uintptr_t)section;
  amdf_status_t status = amdf_platform_host_reservation_create(
      byte_length, minimum_alignment, 0, UINTPTR_MAX, &backing->pointer);
  if (amdf_status_is_ok(status)) {
    status = amdf_platform_host_reservation_replace(
        backing->pointer, byte_length, byte_length, backing->handle, 0,
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
  }
  return status;
}

amdf_status_t amdf_platform_host_backing_destroy(
    amdf_platform_host_backing_t* backing, uint64_t byte_length) {
  if (backing->pointer != NULL) {
    const amdf_status_t status =
        amdf_platform_host_reservation_destroy(backing->pointer, byte_length);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    backing->pointer = NULL;
  }
  if (backing->handle != UINTPTR_MAX) {
    if (!CloseHandle((HANDLE)backing->handle)) {
      return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
    }
    backing->handle = UINTPTR_MAX;
  }
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_platform_host_backing_populate(
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
                                                 uint64_t minimum_alignment,
                                                 void** out_pointer) {
  MEM_ADDRESS_REQUIREMENTS requirements = {0};
  requirements.Alignment = (SIZE_T)minimum_alignment;
  MEM_EXTENDED_PARAMETER parameter = {0};
  parameter.Type = MemExtendedParameterAddressRequirements;
  parameter.Pointer = &requirements;
  void* pointer = VirtualAlloc2(GetCurrentProcess(), NULL, (SIZE_T)byte_length,
                                MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE,
                                minimum_alignment != 0 ? &parameter : NULL,
                                minimum_alignment != 0 ? 1 : 0);
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
