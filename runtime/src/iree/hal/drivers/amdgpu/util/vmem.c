// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/vmem.h"

#include "iree/hal/drivers/amdgpu/util/topology.h"

#if defined(IREE_PLATFORM_WINDOWS)

// clang-format off
#include <windows.h>
// clang-format on

#include <limits.h>

typedef PVOID(WINAPI* iree_hal_amdgpu_virtual_alloc2_fn_t)(
    HANDLE process, PVOID base_address, SIZE_T size, ULONG allocation_type,
    ULONG page_protection, MEM_EXTENDED_PARAMETER* extended_parameters,
    ULONG parameter_count);

typedef PVOID(WINAPI* iree_hal_amdgpu_map_view_of_file3_fn_t)(
    HANDLE file_mapping, HANDLE process, PVOID base_address, ULONG64 offset,
    SIZE_T view_size, ULONG allocation_type, ULONG page_protection,
    MEM_EXTENDED_PARAMETER* extended_parameters, ULONG parameter_count);
#endif  // IREE_PLATFORM_WINDOWS

//===----------------------------------------------------------------------===//
// Virtual Memory Utilities
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_find_global_memory_pool_state_t {
  const iree_hal_amdgpu_libhsa_t* libhsa;
  hsa_amd_memory_pool_global_flag_t match_flags;
  hsa_amd_memory_pool_t best_pool;
} iree_hal_amdgpu_find_global_memory_pool_state_t;
static hsa_status_t iree_hal_amdgpu_find_global_memory_pool_iterator(
    hsa_amd_memory_pool_t memory_pool, void* user_data) {
  iree_hal_amdgpu_find_global_memory_pool_state_t* state =
      (iree_hal_amdgpu_find_global_memory_pool_state_t*)user_data;

  // Filter to the global segment only.
  hsa_region_segment_t segment = 0;
  hsa_status_t hsa_status = iree_hsa_amd_memory_pool_get_info_raw(
      state->libhsa, memory_pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (hsa_status != HSA_STATUS_SUCCESS) return hsa_status;
  if (segment != HSA_REGION_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

  // Must be able to allocate. This should be true for any pool we query that
  // matches the other flags. Workgroup-private pools won't have this set.
  bool alloc_allowed = false;
  hsa_status = iree_hsa_amd_memory_pool_get_info_raw(
      state->libhsa, memory_pool,
      HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);
  if (hsa_status != HSA_STATUS_SUCCESS) return hsa_status;
  if (!alloc_allowed) return HSA_STATUS_SUCCESS;

  // Match if flags are present.
  uint32_t global_flags = 0;
  hsa_status = iree_hsa_amd_memory_pool_get_info_raw(
      state->libhsa, memory_pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
      &global_flags);
  if (hsa_status != HSA_STATUS_SUCCESS) return hsa_status;
  if (global_flags & state->match_flags) {
    state->best_pool = memory_pool;
    return HSA_STATUS_INFO_BREAK;
  }

  return HSA_STATUS_SUCCESS;
}

static iree_status_t iree_hal_amdgpu_query_global_memory_pool(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    hsa_amd_memory_pool_global_flag_t match_flags, bool* out_available,
    hsa_amd_memory_pool_t* out_pool) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_available);
  IREE_ASSERT_ARGUMENT(out_pool);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_available = false;
  memset(out_pool, 0, sizeof(*out_pool));

  iree_hal_amdgpu_find_global_memory_pool_state_t find_state = {
      .libhsa = libhsa,
      .match_flags = match_flags,
      .best_pool = {0},
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hsa_amd_agent_iterate_memory_pools(
              IREE_LIBHSA(libhsa), agent,
              iree_hal_amdgpu_find_global_memory_pool_iterator, &find_state));
  if (find_state.best_pool.handle) {
    *out_available = true;
    *out_pool = find_state.best_pool;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_find_global_memory_pool(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    hsa_amd_memory_pool_global_flag_t match_flags,
    hsa_amd_memory_pool_t* out_pool) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  bool available = false;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_query_global_memory_pool(libhsa, agent, match_flags,
                                                   &available, out_pool));
  if (!available) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_NOT_FOUND,
                             "no memory pool matching the required flags %u",
                             match_flags));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_find_coarse_global_memory_pool(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    hsa_amd_memory_pool_t* out_pool) {
  return iree_hal_amdgpu_find_global_memory_pool(
      libhsa, agent, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED, out_pool);
}

iree_status_t iree_hal_amdgpu_find_fine_global_memory_pool(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    hsa_amd_memory_pool_t* out_pool) {
  return iree_hal_amdgpu_find_global_memory_pool(
      libhsa, agent,
      HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
          HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED,
      out_pool);
}

iree_status_t iree_hal_amdgpu_query_fine_global_memory_pool(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    bool* out_available, hsa_amd_memory_pool_t* out_pool) {
  return iree_hal_amdgpu_query_global_memory_pool(
      libhsa, agent,
      HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
          HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED,
      out_available, out_pool);
}

iree_status_t iree_hal_amdgpu_query_extended_fine_global_memory_pool(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    bool* out_available, hsa_amd_memory_pool_t* out_pool) {
  return iree_hal_amdgpu_query_global_memory_pool(
      libhsa, agent,
      HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED,
      out_available, out_pool);
}

bool iree_hal_amdgpu_try_find_coarse_global_memory_pool(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    hsa_amd_memory_pool_t* out_pool) {
  memset(out_pool, 0, sizeof(*out_pool));
  iree_hal_amdgpu_find_global_memory_pool_state_t find_state = {
      .libhsa = libhsa,
      .match_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED,
      .best_pool = {0},
  };
  (void)iree_hsa_amd_agent_iterate_memory_pools_raw(
      libhsa, agent, iree_hal_amdgpu_find_global_memory_pool_iterator,
      &find_state);
  *out_pool = find_state.best_pool;
  return find_state.best_pool.handle != 0;
}

bool iree_hal_amdgpu_try_find_fine_global_memory_pool(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    hsa_amd_memory_pool_t* out_pool) {
  memset(out_pool, 0, sizeof(*out_pool));
  iree_hal_amdgpu_find_global_memory_pool_state_t find_state = {
      .libhsa = libhsa,
      .match_flags =
          HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
          HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED,
      .best_pool = {0},
  };
  (void)iree_hsa_amd_agent_iterate_memory_pools_raw(
      libhsa, agent, iree_hal_amdgpu_find_global_memory_pool_iterator,
      &find_state);
  *out_pool = find_state.best_pool;
  return find_state.best_pool.handle != 0;
}

iree_status_t iree_hal_amdgpu_vmem_translate_memory_type(
    iree_hal_amdgpu_vmem_memory_type_t memory_type,
    hsa_amd_memory_type_t* out_hsa_memory_type) {
  IREE_ASSERT_ARGUMENT(out_hsa_memory_type);
  switch (memory_type) {
    case IREE_HAL_AMDGPU_VMEM_MEMORY_TYPE_DEFAULT:
      *out_hsa_memory_type = MEMORY_TYPE_NONE;
      return iree_ok_status();
    case IREE_HAL_AMDGPU_VMEM_MEMORY_TYPE_PINNED_HOST:
      *out_hsa_memory_type = MEMORY_TYPE_PINNED;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported vmem memory type: %d",
                              (int)memory_type);
  }
}

iree_status_t iree_hal_amdgpu_vmem_query_alloc_granularity(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_amd_memory_pool_t memory_pool,
    iree_hal_amdgpu_vmem_granularity_t* out_granularity) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_granularity);
  *out_granularity = (iree_hal_amdgpu_vmem_granularity_t){0};

  size_t minimum = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(libhsa), memory_pool,
      HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &minimum));
  if (minimum == 0 || !iree_device_size_is_power_of_two(minimum)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "invalid minimum HSA VMM allocation granule for an AMDGPU memory pool: "
        "%" PRIhsz,
        (iree_host_size_t)minimum);
  }

  size_t recommended = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(libhsa), memory_pool,
      HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE, &recommended));
  if (recommended < minimum || !iree_device_size_is_power_of_two(recommended)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "invalid recommended HSA VMM allocation granule for an AMDGPU memory "
        "pool: "
        "%" PRIhsz,
        (iree_host_size_t)recommended);
  }

  out_granularity->minimum = (iree_device_size_t)minimum;
  out_granularity->recommended = (iree_device_size_t)recommended;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_vmem_build_access_descs_for_topology(
    const iree_hal_amdgpu_topology_t* topology, hsa_agent_t local_agent,
    iree_hal_amdgpu_vmem_access_mode_t access_mode,
    iree_host_size_t access_desc_capacity,
    hsa_amd_memory_access_desc_t* access_descs,
    iree_host_size_t* out_access_desc_count) {
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(access_descs);
  IREE_ASSERT_ARGUMENT(out_access_desc_count);
  *out_access_desc_count = 0;

  if (IREE_UNLIKELY(access_desc_capacity < topology->all_agent_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU VMM access descriptor capacity %" PRIhsz
                            " is smaller than topology agent count %" PRIhsz,
                            access_desc_capacity, topology->all_agent_count);
  }

  switch (access_mode) {
    case IREE_HAL_AMDGPU_ACCESS_MODE_SHARED: {
      // All devices get read/write access.
      for (iree_host_size_t i = 0; i < topology->all_agent_count; ++i) {
        access_descs[(*out_access_desc_count)++] =
            (hsa_amd_memory_access_desc_t){
                .agent_handle = topology->all_agents[i],
                .permissions = HSA_ACCESS_PERMISSION_RW,
            };
      }
      return iree_ok_status();
    }
    case IREE_HAL_AMDGPU_ACCESS_MODE_DEVICE_SHARED: {
      // All GPU devices get read/write access. The backing allocation remains
      // inaccessible to CPU agents.
      for (iree_host_size_t i = 0; i < topology->gpu_agent_count; ++i) {
        access_descs[(*out_access_desc_count)++] =
            (hsa_amd_memory_access_desc_t){
                .agent_handle = topology->gpu_agents[i],
                .permissions = HSA_ACCESS_PERMISSION_RW,
            };
      }
      return iree_ok_status();
    }
    case IREE_HAL_AMDGPU_ACCESS_MODE_EXCLUSIVE: {
      // Only the local agent can access the allocation.
      access_descs[(*out_access_desc_count)++] = (hsa_amd_memory_access_desc_t){
          .agent_handle = local_agent,
          .permissions = HSA_ACCESS_PERMISSION_RW,
      };
      return iree_ok_status();
    }
    case IREE_HAL_AMDGPU_ACCESS_MODE_EXCLUSIVE_CONSUMER: {
      // Local agent gets read, all other agents get write.
      for (iree_host_size_t i = 0; i < topology->all_agent_count; ++i) {
        hsa_agent_t agent = topology->all_agents[i];
        hsa_access_permission_t permissions = agent.handle == local_agent.handle
                                                  ? HSA_ACCESS_PERMISSION_RO
                                                  : HSA_ACCESS_PERMISSION_WO;
        access_descs[(*out_access_desc_count)++] =
            (hsa_amd_memory_access_desc_t){
                .agent_handle = agent,
                .permissions = permissions,
            };
      }
      return iree_ok_status();
    }
    case IREE_HAL_AMDGPU_ACCESS_MODE_EXCLUSIVE_PRODUCER: {
      // Local agent gets write, all other agents get read.
      for (iree_host_size_t i = 0; i < topology->all_agent_count; ++i) {
        hsa_agent_t agent = topology->all_agents[i];
        hsa_access_permission_t permissions = agent.handle == local_agent.handle
                                                  ? HSA_ACCESS_PERMISSION_WO
                                                  : HSA_ACCESS_PERMISSION_RO;
        access_descs[(*out_access_desc_count)++] =
            (hsa_amd_memory_access_desc_t){
                .agent_handle = agent,
                .permissions = permissions,
            };
      }
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported AMDGPU VMM access mode: %d",
                              (int)access_mode);
  }
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_vmem_ringbuffer_t
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_WINDOWS)

static void iree_hal_amdgpu_vmem_release_host_alias_views(
    void* va_base_ptr, iree_device_size_t capacity,
    iree_host_size_t mapped_view_count) {
  for (iree_host_size_t i = 0; i < mapped_view_count; ++i) {
    void* view_ptr = (uint8_t*)va_base_ptr + i * capacity;
    const BOOL unmapped = UnmapViewOfFile(view_ptr);
    IREE_ASSERT(unmapped && "failed to unmap AMDGPU host ringbuffer view");
    (void)unmapped;
  }
}

static iree_status_t iree_hal_amdgpu_vmem_ringbuffer_initialize_host_aliases(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_device_size_t min_capacity,
    const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_vmem_ringbuffer_t* out_ringbuffer) {
  if (IREE_UNLIKELY(min_capacity == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU ringbuffer capacity must be non-zero");
  }
  if (IREE_UNLIKELY(topology->gpu_agent_count > INT_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU ringbuffer GPU agent count exceeds the "
                            "HSA host registration limit");
  }

  HMODULE kernelbase_module = GetModuleHandleW(L"kernelbase.dll");
  iree_hal_amdgpu_virtual_alloc2_fn_t virtual_alloc2 =
      kernelbase_module ? (iree_hal_amdgpu_virtual_alloc2_fn_t)GetProcAddress(
                              kernelbase_module, "VirtualAlloc2")
                        : NULL;
  iree_hal_amdgpu_map_view_of_file3_fn_t map_view_of_file3 =
      kernelbase_module
          ? (iree_hal_amdgpu_map_view_of_file3_fn_t)GetProcAddress(
                kernelbase_module, "MapViewOfFile3")
          : NULL;
  if (IREE_UNLIKELY(!virtual_alloc2 || !map_view_of_file3)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "Windows host-aliased AMDGPU ringbuffers require VirtualAlloc2 and "
        "MapViewOfFile3");
  }

  SYSTEM_INFO system_info;
  GetSystemInfo(&system_info);
  iree_device_size_t capacity = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_align(
                        min_capacity,
                        (iree_device_size_t)system_info.dwAllocationGranularity,
                        &capacity) ||
                    capacity > SIZE_MAX / 3)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU host ringbuffer capacity cannot be "
                            "represented: minimum=%" PRIu64,
                            (uint64_t)min_capacity);
  }

  const SIZE_T view_size = (SIZE_T)capacity;
  void* placeholder_base =
      virtual_alloc2(/*process=*/NULL, /*base_address=*/NULL, view_size * 3,
                     MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS,
                     /*extended_parameters=*/NULL, /*parameter_count=*/0);
  if (IREE_UNLIKELY(!placeholder_base)) {
    const DWORD error = GetLastError();
    return iree_make_status(
        iree_status_code_from_win32_error(error),
        "VirtualAlloc2 failed to reserve %" PRIu64
        " bytes for AMDGPU host ringbuffer aliases; error=%lu",
        (uint64_t)capacity * 3, (unsigned long)error);
  }

  bool first_split = false;
  bool second_split = false;
  void* mapped_views[3] = {NULL};
  HANDLE section = NULL;
  iree_status_t status = iree_ok_status();

  if (!VirtualFree(placeholder_base, view_size,
                   MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
    const DWORD error = GetLastError();
    status = iree_make_status(
        iree_status_code_from_win32_error(error),
        "VirtualFree failed to split the first AMDGPU host ringbuffer "
        "placeholder; error=%lu",
        (unsigned long)error);
  } else {
    first_split = true;
  }
  if (iree_status_is_ok(status)) {
    void* second_placeholder = (uint8_t*)placeholder_base + capacity;
    if (!VirtualFree(second_placeholder, view_size,
                     MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
      const DWORD error = GetLastError();
      status = iree_make_status(
          iree_status_code_from_win32_error(error),
          "VirtualFree failed to split the second AMDGPU host ringbuffer "
          "placeholder; error=%lu",
          (unsigned long)error);
    } else {
      second_split = true;
    }
  }

  if (iree_status_is_ok(status)) {
    ULARGE_INTEGER section_size;
    section_size.QuadPart = (ULONGLONG)capacity;
    section = CreateFileMappingW(INVALID_HANDLE_VALUE,
                                 /*lpFileMappingAttributes=*/NULL,
                                 PAGE_READWRITE, section_size.HighPart,
                                 section_size.LowPart, /*lpName=*/NULL);
    if (!section) {
      const DWORD error = GetLastError();
      status = iree_make_status(iree_status_code_from_win32_error(error),
                                "CreateFileMappingW failed for a %" PRIu64
                                " byte AMDGPU host ringbuffer; error=%lu",
                                (uint64_t)capacity, (unsigned long)error);
    }
  }

  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < IREE_ARRAYSIZE(mapped_views); ++i) {
    void* placeholder = (uint8_t*)placeholder_base + i * capacity;
    mapped_views[i] =
        map_view_of_file3(section, /*process=*/NULL, placeholder, /*offset=*/0,
                          view_size, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE,
                          /*extended_parameters=*/NULL, /*parameter_count=*/0);
    if (!mapped_views[i]) {
      const DWORD error = GetLastError();
      status = iree_make_status(
          iree_status_code_from_win32_error(error),
          "MapViewOfFile3 failed for AMDGPU host ringbuffer view %" PRIhsz
          "; error=%lu",
          i, (unsigned long)error);
    }
  }

  if (section) {
    const BOOL closed = CloseHandle(section);
    section = NULL;
    if (iree_status_is_ok(status) && !closed) {
      const DWORD error = GetLastError();
      status = iree_make_status(
          iree_status_code_from_win32_error(error),
          "CloseHandle failed for AMDGPU host ringbuffer section; error=%lu",
          (unsigned long)error);
    }
  }

  if (iree_status_is_ok(status)) {
    out_ringbuffer->capacity = capacity;
    out_ringbuffer->storage =
        IREE_HAL_AMDGPU_VMEM_RINGBUFFER_STORAGE_HOST_ALIASED;
    out_ringbuffer->va_base_ptr = placeholder_base;
    out_ringbuffer->ring_base_ptr = (uint8_t*)placeholder_base + capacity;
    out_ringbuffer->mapped_view_count = IREE_ARRAYSIZE(mapped_views);

    void* device_va_base_ptr = NULL;
    status = iree_hsa_amd_memory_lock(
        IREE_LIBHSA(libhsa), out_ringbuffer->va_base_ptr, (size_t)capacity * 3,
        topology->gpu_agent_count ? (hsa_agent_t*)topology->gpu_agents : NULL,
        (int)topology->gpu_agent_count, &device_va_base_ptr);
    if (iree_status_is_ok(status)) {
      out_ringbuffer->device_base_ptr = (uint8_t*)device_va_base_ptr + capacity;
      out_ringbuffer->host_registration_active = true;
    } else {
      iree_hal_amdgpu_vmem_ringbuffer_deinitialize(libhsa, out_ringbuffer);
    }
  } else if (second_split) {
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(mapped_views); ++i) {
      void* placeholder = (uint8_t*)placeholder_base + i * capacity;
      if (mapped_views[i]) {
        const BOOL unmapped = UnmapViewOfFile(mapped_views[i]);
        IREE_ASSERT(unmapped && "failed to unmap partial host ringbuffer view");
        (void)unmapped;
      } else {
        const BOOL released = VirtualFree(placeholder, 0, MEM_RELEASE);
        IREE_ASSERT(released &&
                    "failed to release host ringbuffer placeholder");
        (void)released;
      }
    }
  } else if (first_split) {
    const BOOL first_released = VirtualFree(placeholder_base, 0, MEM_RELEASE);
    const BOOL remainder_released =
        VirtualFree((uint8_t*)placeholder_base + capacity, 0, MEM_RELEASE);
    IREE_ASSERT(first_released && remainder_released &&
                "failed to release split host ringbuffer placeholders");
    (void)first_released;
    (void)remainder_released;
  } else {
    const BOOL released = VirtualFree(placeholder_base, 0, MEM_RELEASE);
    IREE_ASSERT(released && "failed to release host ringbuffer placeholder");
    (void)released;
  }

  return status;
}

#endif  // IREE_PLATFORM_WINDOWS

iree_status_t iree_hal_amdgpu_vmem_ringbuffer_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t local_agent,
    hsa_amd_memory_pool_t memory_pool,
    iree_hal_amdgpu_vmem_memory_type_t memory_type,
    iree_device_size_t min_capacity, iree_host_size_t access_desc_count,
    const hsa_amd_memory_access_desc_t* access_descs,
    iree_hal_amdgpu_vmem_ringbuffer_t* out_ringbuffer) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_ringbuffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, min_capacity);
  memset(out_ringbuffer, 0, sizeof(*out_ringbuffer));

  hsa_amd_memory_type_t hsa_memory_type = MEMORY_TYPE_NONE;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_vmem_translate_memory_type(memory_type,
                                                     &hsa_memory_type));

  // hsa_amd_vmem_handle_create wants values aligned to this value.
  iree_hal_amdgpu_vmem_granularity_t granularity;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_vmem_query_alloc_granularity(libhsa, memory_pool,
                                                       &granularity));

  // Round up capacity and alignment to the allocation granule.
  const size_t alignment = (size_t)granularity.recommended;
  const size_t capacity =
      (size_t)iree_device_align(min_capacity, granularity.recommended);
  out_ringbuffer->capacity = capacity;

  // Reserve the virtual address space for the 3x the capacity. We'll map the
  // physical allocation into this address space.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_amd_vmem_address_reserve_align(
          IREE_LIBHSA(libhsa), &out_ringbuffer->va_base_ptr, capacity * 3,
          /*address=*/0, alignment, /*flags=*/0),
      "reserving ringbuffer capacity*3 (%" PRIdsz "*3=%" PRIdsz ")", capacity,
      capacity * 3);
  out_ringbuffer->ring_base_ptr =
      (uint8_t*)out_ringbuffer->va_base_ptr + capacity;
  out_ringbuffer->device_base_ptr = out_ringbuffer->ring_base_ptr;
  out_ringbuffer->storage = IREE_HAL_AMDGPU_VMEM_RINGBUFFER_STORAGE_HSA_VMEM;

  // Allocate the physical memory for backing the ringbuffer.
  iree_status_t status = iree_hsa_amd_vmem_handle_create(
      IREE_LIBHSA(libhsa), memory_pool, capacity, hsa_memory_type, /*flags=*/0,
      &out_ringbuffer->alloc_handle);

  void* va_offsets[3] = {
      (uint8_t*)out_ringbuffer->va_base_ptr + 0 * capacity,
      (uint8_t*)out_ringbuffer->va_base_ptr + 1 * capacity,
      (uint8_t*)out_ringbuffer->va_base_ptr + 2 * capacity,
  };

  // Map the physical allocation into the virtual address space 3 times
  // (prev, base, next).
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < 3; ++i) {
    status =
        iree_hsa_amd_vmem_map(IREE_LIBHSA(libhsa), va_offsets[i], capacity, 0,
                              out_ringbuffer->alloc_handle, /*flags=*/0);
    if (iree_status_is_ok(status)) ++out_ringbuffer->mapped_view_count;
  }

  // Enable access on requested devices (no access by default).
  // Must be done per memory handle, not the entire VA.
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < 3; ++i) {
    status =
        iree_hsa_amd_vmem_set_access(IREE_LIBHSA(libhsa), va_offsets[i],
                                     capacity, access_descs, access_desc_count);
    if (!iree_status_is_ok(status)) break;
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_vmem_ringbuffer_deinitialize(libhsa, out_ringbuffer);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amdgpu_vmem_ringbuffer_initialize_with_topology(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t local_agent,
    hsa_amd_memory_pool_t memory_pool,
    iree_hal_amdgpu_vmem_memory_type_t memory_type,
    iree_device_size_t min_capacity, const iree_hal_amdgpu_topology_t* topology,
    iree_hal_amdgpu_vmem_access_mode_t access_mode,
    iree_hal_amdgpu_vmem_ringbuffer_t* out_ringbuffer) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(topology);
  IREE_ASSERT_ARGUMENT(out_ringbuffer);
  memset(out_ringbuffer, 0, sizeof(*out_ringbuffer));
  IREE_TRACE_ZONE_BEGIN(z0);

#if defined(IREE_PLATFORM_WINDOWS)
  hsa_amd_memory_type_t hsa_memory_type = MEMORY_TYPE_NONE;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_vmem_translate_memory_type(memory_type,
                                                     &hsa_memory_type));
  (void)hsa_memory_type;
  (void)local_agent;
  (void)memory_pool;
  if (IREE_UNLIKELY(access_mode != IREE_HAL_AMDGPU_ACCESS_MODE_SHARED)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                             "Windows host-aliased AMDGPU ringbuffers require "
                             "shared access"));
  }
  iree_status_t status =
      iree_hal_amdgpu_vmem_ringbuffer_initialize_host_aliases(
          libhsa, min_capacity, topology, out_ringbuffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
#else

  // Allocate scratch for the access descriptors. Note that though we allocate
  // for all agents we don't pass agents with HSA_ACCESS_PERMISSION_NONE as that
  // actually causes HSA to allocate information about that agent.
  // HSA_ACCESS_PERMISSION_NONE should only be used to _remove_ access that was
  // previous granted.
  iree_host_size_t access_desc_count = 0;
  hsa_amd_memory_access_desc_t* access_descs =
      (hsa_amd_memory_access_desc_t*)iree_alloca(
          topology->all_agent_count * sizeof(hsa_amd_memory_access_desc_t));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_vmem_build_access_descs_for_topology(
              topology, local_agent, access_mode, topology->all_agent_count,
              access_descs, &access_desc_count));

  // Route to the explicit initializer.
  iree_status_t status = iree_hal_amdgpu_vmem_ringbuffer_initialize(
      libhsa, local_agent, memory_pool, memory_type, min_capacity,
      access_desc_count, access_descs, out_ringbuffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
#endif  // IREE_PLATFORM_WINDOWS
}

void iree_hal_amdgpu_vmem_ringbuffer_deinitialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_vmem_ringbuffer_t* ringbuffer) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(ringbuffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  switch (ringbuffer->storage) {
    case IREE_HAL_AMDGPU_VMEM_RINGBUFFER_STORAGE_NONE:
      break;
    case IREE_HAL_AMDGPU_VMEM_RINGBUFFER_STORAGE_HSA_VMEM: {
      for (iree_host_size_t i = 0; i < ringbuffer->mapped_view_count; ++i) {
        void* va_offset =
            (uint8_t*)ringbuffer->va_base_ptr + i * ringbuffer->capacity;
        iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_amd_vmem_unmap_raw(
            libhsa, va_offset, ringbuffer->capacity));
      }
      if (ringbuffer->alloc_handle.handle) {
        iree_hal_amdgpu_hsa_cleanup_assert_success(
            iree_hsa_amd_vmem_handle_release_raw(libhsa,
                                                 ringbuffer->alloc_handle));
      }
      if (ringbuffer->va_base_ptr) {
        iree_hal_amdgpu_hsa_cleanup_assert_success(
            iree_hsa_amd_vmem_address_free_raw(libhsa, ringbuffer->va_base_ptr,
                                               ringbuffer->capacity * 3));
      }
      break;
    }
    case IREE_HAL_AMDGPU_VMEM_RINGBUFFER_STORAGE_HOST_ALIASED:
#if defined(IREE_PLATFORM_WINDOWS)
      if (ringbuffer->host_registration_active) {
        iree_hal_amdgpu_hsa_cleanup_assert_success(
            iree_hsa_amd_memory_unlock_raw(libhsa, ringbuffer->va_base_ptr));
      }
      if (ringbuffer->va_base_ptr) {
        iree_hal_amdgpu_vmem_release_host_alias_views(
            ringbuffer->va_base_ptr, ringbuffer->capacity,
            ringbuffer->mapped_view_count);
      }
#else
      IREE_ASSERT(false && "host-aliased ringbuffer on unsupported platform");
#endif  // IREE_PLATFORM_WINDOWS
      break;
    default:
      IREE_ASSERT(false && "invalid AMDGPU ringbuffer storage strategy");
      break;
  }

  memset(ringbuffer, 0, sizeof(*ringbuffer));

  IREE_TRACE_ZONE_END(z0);
}
