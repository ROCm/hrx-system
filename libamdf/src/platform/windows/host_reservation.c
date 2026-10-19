// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/host_reservation.h"

#define NTDDI_VERSION NTDDI_WIN10_RS4
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

amdf_status_t amdf_platform_host_reservation_query_capabilities(
    amdf_memory_reservation_capabilities_t* out_capabilities) {
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  const uint64_t maximum = (uintptr_t)info.lpMaximumApplicationAddress;
  uint64_t alignment = 1;
  while (alignment <= maximum / 2) {
    alignment *= 2;
  }
  *out_capabilities = (amdf_memory_reservation_capabilities_t){
      .supported_access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .required_access = AMDF_MEMORY_ACCESS_READ,
      .address_domain_ordinal = AMDF_ADDRESS_DOMAIN_ORDINAL_NONE,
      .maximum_byte_length =
          (maximum - (uintptr_t)info.lpMinimumApplicationAddress + 1) &
          ~((uint64_t)info.dwPageSize - 1),
      .reservation_granularity = info.dwPageSize,
      .maximum_alignment = alignment,
      .mapping_granularity = info.dwPageSize,
      .minimum_address = (uintptr_t)info.lpMinimumApplicationAddress,
      .maximum_address = maximum,
      .required_source_flags = AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE,
  };
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_platform_host_reservation_create(uint64_t byte_length,
                                                    uint64_t minimum_alignment,
                                                    void** out_pointer) {
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  MEM_ADDRESS_REQUIREMENTS requirements = {0};
  requirements.Alignment =
      (SIZE_T)(minimum_alignment > info.dwAllocationGranularity
                   ? minimum_alignment
                   : info.dwAllocationGranularity);
  MEM_EXTENDED_PARAMETER parameter = {0};
  parameter.Type = MemExtendedParameterAddressRequirements;
  parameter.Pointer = &requirements;
  void* pointer = VirtualAlloc2(GetCurrentProcess(), NULL, (SIZE_T)byte_length,
                                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                PAGE_NOACCESS, &parameter, 1);
  if (pointer == NULL) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  *out_pointer = pointer;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_platform_host_reservation_clear(void* pointer,
                                                   uint64_t byte_length) {
  const uintptr_t end = (uintptr_t)pointer + (uintptr_t)byte_length;
  uintptr_t cursor = (uintptr_t)pointer;
  amdf_status_t status = AMDF_STATUS_OK;
  while (cursor < end && amdf_status_is_ok(status)) {
    MEMORY_BASIC_INFORMATION region;
    if (VirtualQuery((void*)cursor, &region, sizeof(region)) == 0) {
      status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
    } else {
      // Each installed view is exactly one independently replaceable unit.
      // Holes may span many units; querying OS geometry avoids a range ledger
      // and avoids walking every page of a large, mostly empty reservation.
      const uintptr_t next = (uintptr_t)region.BaseAddress + region.RegionSize;
      if (region.Type == MEM_MAPPED) {
        if (!UnmapViewOfFile2(GetCurrentProcess(), region.AllocationBase,
                              MEM_PRESERVE_PLACEHOLDER)) {
          status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
        }
      }
      cursor = next;
    }
  }
  return status;
}

static amdf_status_t amdf_windows_host_reservation_replace_unit(
    void* pointer, uint64_t byte_length, uintptr_t source,
    uint64_t source_byte_offset, amdf_memory_access_t access) {
  amdf_status_t status =
      amdf_platform_host_reservation_clear(pointer, byte_length);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  MEMORY_BASIC_INFORMATION region;
  if (VirtualQuery(pointer, &region, sizeof(region)) == 0) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  const uintptr_t end = (uintptr_t)region.BaseAddress + region.RegionSize;
  // Split only the placeholder being populated; a large empty arena remains
  // one native range. Neither split releases any of its addresses.
  // VirtualQuery starts its contiguous scan at the queried page. AllocationBase
  // identifies the placeholder's beginning when the destination is interior.
  if (region.AllocationBase != pointer &&
      !VirtualFree(
          region.AllocationBase,
          (SIZE_T)((uintptr_t)pointer - (uintptr_t)region.AllocationBase),
          MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  if ((uintptr_t)pointer + byte_length != end &&
      !VirtualFree(pointer, (SIZE_T)byte_length,
                   MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  const ULONG protection =
      (access & AMDF_MEMORY_ACCESS_WRITE) != 0 ? PAGE_READWRITE : PAGE_READONLY;
  return MapViewOfFile3((HANDLE)source, GetCurrentProcess(), pointer,
                        source_byte_offset, (SIZE_T)byte_length,
                        MEM_REPLACE_PLACEHOLDER, protection, NULL, 0) != NULL
             ? AMDF_STATUS_OK
             : amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
}

amdf_status_t amdf_platform_host_reservation_replace(
    void* pointer, uint64_t byte_length, uint64_t mapping_granularity,
    uintptr_t source, uint64_t source_byte_offset,
    amdf_memory_access_t access) {
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint64_t offset = 0;
       offset < byte_length && amdf_status_is_ok(status);) {
    const uint64_t remaining = byte_length - offset;
    const uint64_t unit_length =
        remaining < mapping_granularity ? remaining : mapping_granularity;
    status = amdf_windows_host_reservation_replace_unit(
        (uint8_t*)pointer + offset, unit_length, source,
        source_byte_offset + offset, access);
    offset += unit_length;
  }
  return status;
}

amdf_status_t amdf_platform_host_reservation_destroy(void* pointer,
                                                     uint64_t byte_length) {
  const amdf_status_t status =
      amdf_platform_host_reservation_clear(pointer, byte_length);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  MEMORY_BASIC_INFORMATION region;
  if (VirtualQuery(pointer, &region, sizeof(region)) == 0) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  if (region.RegionSize != byte_length &&
      !VirtualFree(pointer, (SIZE_T)byte_length,
                   MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  return VirtualFree(pointer, 0, MEM_RELEASE)
             ? AMDF_STATUS_OK
             : amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
}
