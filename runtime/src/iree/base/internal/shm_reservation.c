// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/memory.h"
#include "iree/base/internal/shm.h"

#if defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>
#elif defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_APPLE)
#include <errno.h>
#include <sys/mman.h>
#endif

#if defined(IREE_PLATFORM_WINDOWS)

// These entry points are optional on older Windows versions. Their module is
// process-owned; resolving them never loads or retains a library.
typedef void*(WINAPI* iree_shm_virtual_alloc2_fn_t)(HANDLE, void*, SIZE_T,
                                                    ULONG, ULONG,
                                                    MEM_EXTENDED_PARAMETER*,
                                                    ULONG);
typedef void*(WINAPI* iree_shm_map_view3_fn_t)(HANDLE, HANDLE, void*, ULONG64,
                                               SIZE_T, ULONG, ULONG,
                                               MEM_EXTENDED_PARAMETER*, ULONG);
typedef BOOL(WINAPI* iree_shm_unmap_view2_fn_t)(HANDLE, void*, ULONG);

typedef struct iree_shm_reservation_api_t {
  // Reserves aligned placeholder intervals.
  iree_shm_virtual_alloc2_fn_t allocate;
  // Replaces one exact placeholder with shared backing.
  iree_shm_map_view3_fn_t map;
  // Restores a mapped view to its owned placeholder.
  iree_shm_unmap_view2_fn_t unmap;
} iree_shm_reservation_api_t;

static bool iree_shm_reservation_query_api(
    iree_shm_reservation_api_t* out_api) {
  HMODULE module = GetModuleHandleW(L"kernelbase.dll");
  if (!module) {
    return false;
  }
  *out_api = (iree_shm_reservation_api_t){
      .allocate =
          (iree_shm_virtual_alloc2_fn_t)GetProcAddress(module, "VirtualAlloc2"),
      .map = (iree_shm_map_view3_fn_t)GetProcAddress(module, "MapViewOfFile3"),
      .unmap =
          (iree_shm_unmap_view2_fn_t)GetProcAddress(module, "UnmapViewOfFile2"),
  };
  return out_api->allocate && out_api->map && out_api->unmap;
}

static iree_status_t iree_shm_reservation_resolve_api(
    iree_shm_reservation_api_t* out_api) {
  if (!iree_shm_reservation_query_api(out_api)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "Windows placeholder mappings are unavailable");
  }
  return iree_ok_status();
}

static iree_status_t iree_shm_reservation_native_error(const char* operation) {
  DWORD error = GetLastError();
  return iree_make_status(iree_status_code_from_win32_error(error),
                          "%s failed (%lu)", operation, (unsigned long)error);
}

// Every installed view is one replacement unit, with a possibly shorter final
// unit. Untouched holes remain large placeholders; split only the unit about
// to receive backing.
static iree_status_t iree_shm_reservation_prepare_unit(
    const iree_shm_reservation_api_t* api, uint8_t* address,
    iree_host_size_t granularity) {
  MEMORY_BASIC_INFORMATION region = {0};
  if (!VirtualQuery(address, &region, sizeof(region))) {
    return iree_shm_reservation_native_error("VirtualQuery");
  }
  if (region.State == MEM_COMMIT) {
    if (!api->unmap(GetCurrentProcess(), address, MEM_PRESERVE_PLACEHOLDER)) {
      return iree_shm_reservation_native_error("UnmapViewOfFile2");
    }
    return iree_ok_status();
  }
  // VirtualQuery describes the suffix starting at the queried page. The
  // placeholder itself begins at AllocationBase, which can precede that page.
  iree_host_size_t prefix = address - (uint8_t*)region.AllocationBase;
  if (prefix && !VirtualFree(region.AllocationBase, prefix,
                             MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
    return iree_shm_reservation_native_error("VirtualFree(prefix split)");
  }
  iree_host_size_t remaining_length =
      region.RegionSize - (address - (uint8_t*)region.BaseAddress);
  if (remaining_length > granularity &&
      !VirtualFree(address, granularity,
                   MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
    return iree_shm_reservation_native_error("VirtualFree(unit split)");
  }
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_WINDOWS

bool iree_shm_reservation_is_supported(void) {
#if defined(IREE_PLATFORM_WINDOWS)
  iree_shm_reservation_api_t api;
  return iree_shm_reservation_query_api(&api);
#elif defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_APPLE)
  return true;
#else
  return false;
#endif
}

iree_status_t iree_shm_reservation_initialize(
    iree_host_size_t byte_length, iree_host_size_t alignment,
    iree_host_size_t granularity, iree_shm_reservation_t* out_reservation) {
  const iree_memory_info_t info = iree_memory_query_info();
  if (!granularity) {
    granularity = info.normal_page_size;
  }
  if (!alignment) {
    alignment = granularity;
  }
  if (!byte_length || !granularity ||
      !iree_host_size_is_valid_alignment(granularity) ||
      granularity < info.normal_page_size ||
      byte_length % info.normal_page_size ||
      !iree_host_size_is_valid_alignment(alignment) ||
      alignment < info.normal_page_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid shared address reservation geometry");
  }
  alignment = iree_max(alignment, granularity);
  iree_shm_reservation_t reservation = {
      .byte_length = byte_length,
      .granularity = granularity,
  };
#if defined(IREE_PLATFORM_WINDOWS)
  iree_shm_reservation_api_t api;
  IREE_RETURN_IF_ERROR(iree_shm_reservation_resolve_api(&api));
  MEM_ADDRESS_REQUIREMENTS requirements = {
      .Alignment = iree_max(alignment, info.normal_page_granularity),
  };
  MEM_EXTENDED_PARAMETER parameter = {0};
  parameter.Type = MemExtendedParameterAddressRequirements;
  parameter.Pointer = &requirements;
  reservation.base = api.allocate(GetCurrentProcess(), NULL, byte_length,
                                  MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                  PAGE_NOACCESS, &parameter, 1);
  if (!reservation.base) {
    return iree_shm_reservation_native_error("VirtualAlloc2");
  }
  reservation.allocation_base = reservation.base;
  reservation.allocation_byte_length = byte_length;
#elif defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_APPLE)
  if (!iree_host_size_checked_add(byte_length,
                                  alignment - info.normal_page_size,
                                  &reservation.allocation_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "shared address reservation extent overflows");
  }
  reservation.allocation_base =
      mmap(NULL, reservation.allocation_byte_length, PROT_NONE,
           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (reservation.allocation_base == MAP_FAILED) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "mmap shared address reservation failed");
  }
  reservation.base = (uint8_t*)iree_host_align(
      (uintptr_t)reservation.allocation_base, alignment);
#else
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "shared address reservations are unavailable");
#endif
  *out_reservation = reservation;
  return iree_ok_status();
}

iree_status_t iree_shm_reservation_update(
    iree_shm_reservation_t* reservation, iree_host_size_t update_count,
    const iree_shm_mapping_update_t* updates) {
  const iree_host_size_t page_size = iree_memory_query_info().normal_page_size;
  iree_host_size_t previous_end = 0;
  for (iree_host_size_t i = 0; i < update_count; ++i) {
    const iree_shm_mapping_update_t* update = &updates[i];
    if (!update->byte_length || update->target_offset < previous_end ||
        update->target_offset > reservation->byte_length ||
        update->byte_length >
            reservation->byte_length - update->target_offset ||
        update->target_offset % reservation->granularity ||
        (update->byte_length % reservation->granularity &&
         update->byte_length !=
             reservation->byte_length - update->target_offset)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "invalid shared mapping destination range");
    }
    previous_end = update->target_offset + update->byte_length;
    if (update->source) {
      if (!iree_shm_handle_is_valid(update->source->handle) ||
          update->source_offset > update->source->size ||
          update->byte_length > update->source->size - update->source_offset ||
          update->source_offset % page_size || !update->access ||
          (update->access & ~(IREE_SHM_ACCESS_READ | IREE_SHM_ACCESS_WRITE))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid shared mapping source or access");
      }
#if defined(IREE_PLATFORM_WINDOWS)
      if (!(update->access & IREE_SHM_ACCESS_READ)) {
        return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "Windows shared views require read access");
      }
#endif
    } else if (update->source_offset || update->access) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "detach cannot specify source offset or access");
    }
  }
  iree_status_t status = iree_ok_status();
#if defined(IREE_PLATFORM_WINDOWS)
  iree_shm_reservation_api_t api;
  IREE_RETURN_IF_ERROR(iree_shm_reservation_resolve_api(&api));
  for (iree_host_size_t i = 0; i < update_count && iree_status_is_ok(status);
       ++i) {
    const iree_shm_mapping_update_t* update = &updates[i];
    for (iree_host_size_t offset = 0;
         offset < update->byte_length && iree_status_is_ok(status);
         offset +=
         iree_min(reservation->granularity, update->byte_length - offset)) {
      uint8_t* address = reservation->base + update->target_offset + offset;
      const iree_host_size_t unit_length =
          iree_min(reservation->granularity, update->byte_length - offset);
      status = iree_shm_reservation_prepare_unit(&api, address, unit_length);
      if (iree_status_is_ok(status) && update->source &&
          !api.map((HANDLE)(uintptr_t)update->source->handle.value,
                   GetCurrentProcess(), address, update->source_offset + offset,
                   unit_length, MEM_REPLACE_PLACEHOLDER,
                   (update->access & IREE_SHM_ACCESS_WRITE) ? PAGE_READWRITE
                                                            : PAGE_READONLY,
                   NULL, 0)) {
        status = iree_shm_reservation_native_error("MapViewOfFile3");
      }
    }
  }
#elif defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_APPLE)
  for (iree_host_size_t i = 0; i < update_count && iree_status_is_ok(status);
       ++i) {
    const iree_shm_mapping_update_t* update = &updates[i];
    const int protection =
        ((update->access & IREE_SHM_ACCESS_READ) ? PROT_READ : 0) |
        ((update->access & IREE_SHM_ACCESS_WRITE) ? PROT_WRITE : 0);
    const int flags =
        MAP_FIXED | (update->source ? MAP_SHARED : MAP_PRIVATE | MAP_ANONYMOUS);
    const int descriptor =
        update->source ? (int)update->source->handle.value : -1;
    void* address =
        mmap(reservation->base + update->target_offset, update->byte_length,
             protection, flags, descriptor, (off_t)update->source_offset);
    if (address == MAP_FAILED) {
      status = iree_make_status(iree_status_code_from_errno(errno),
                                "mmap shared address replacement failed");
    }
  }
#else
  status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "shared address reservations are unavailable");
#endif
  return status;
}

iree_status_t iree_shm_reservation_detach(iree_shm_reservation_t* reservation) {
  const iree_shm_mapping_update_t update = {
      .byte_length = reservation->byte_length,
  };
  return iree_shm_reservation_update(reservation, 1, &update);
}

iree_status_t iree_shm_reservation_deinitialize(
    iree_shm_reservation_t* reservation) {
  if (!reservation->allocation_base) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
#if defined(IREE_PLATFORM_WINDOWS)
  iree_shm_reservation_api_t api;
  status = iree_shm_reservation_resolve_api(&api);
  uint8_t* address = reservation->allocation_base;
  uint8_t* end = address + reservation->allocation_byte_length;
  while (address < end && iree_status_is_ok(status)) {
    MEMORY_BASIC_INFORMATION region = {0};
    if (!VirtualQuery(address, &region, sizeof(region))) {
      status = iree_shm_reservation_native_error("VirtualQuery");
    } else {
      if (region.State == MEM_COMMIT &&
          !api.unmap(GetCurrentProcess(), address, MEM_PRESERVE_PLACEHOLDER)) {
        status = iree_shm_reservation_native_error("UnmapViewOfFile2");
      }
      if (iree_status_is_ok(status) && !VirtualFree(address, 0, MEM_RELEASE)) {
        status = iree_shm_reservation_native_error("VirtualFree");
      }
      address += region.RegionSize;
    }
  }
#elif defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_APPLE)
  if (munmap(reservation->allocation_base,
             reservation->allocation_byte_length) != 0) {
    status = iree_make_status(iree_status_code_from_errno(errno),
                              "munmap shared address reservation failed");
  }
#else
  status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "shared address reservations are unavailable");
#endif
  *reservation = (iree_shm_reservation_t){0};
  return status;
}
