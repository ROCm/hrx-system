// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/memory.h"

#include <stddef.h>
#include <stdint.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/windows/host_cache.h"
#include "libamdf/src/xdna/umd/mcdm/context.h"
#include "libamdf/src/xdna/umd/mcdm/device.h"
#include "libamdf/src/xdna/umd/mcdm/kernel_execution.h"
#include "libamdf/src/xdna/umd/mcdm/memory_profile.h"

#define AMDF_WINDOWS_KMT_PAGE_SIZE UINT64_C(4096)

struct amdf_xdna_umd_memory_t {
  // Device borrowed while this attachment remains live.
  amdf_xdna_umd_device_t* device;
  // Borrowed context for private instruction memory, NULL for ordinary backing.
  amdf_xdna_umd_context_t* context;
  // Context-qualified native backing owned directly by this memory resource.
  amdf_windows_xdna_private_allocation_t private_allocation;
  // KMT resource owning the allocation, or zero for an ungrouped allocation.
  D3DKMT_HANDLE resource;
  // KMT physical allocation attached to the XDNA device.
  D3DKMT_HANDLE allocation;
  // Provider-owned host base used by the standard allocation.
  void* host_pointer;
  // Physical allocation length in bytes.
  uint64_t byte_length;
  // Stable XDNA virtual base established before publication.
  uint64_t device_address;
  // Accepted paging fence that must retire before native release.
  uint64_t pending_paging_fence;
};

struct amdf_xdna_umd_host_mapping_t {
  // Host allocator copied for independent mapping teardown.
  amdf_allocator_t host_allocator;
  // First byte exposed by this mapping.
  void* pointer;
  // Exposed byte length.
  uint64_t byte_length;
};

static amdf_status_t amdf_windows_xdna_memory_release_native(
    amdf_xdna_umd_memory_t* memory) {
  if (memory->context != NULL) {
    amdf_status_t status = amdf_windows_xdna_kernel_execution_release_memory(
        memory->context->kernel_execution);
    if (amdf_status_is_ok(status)) {
      status = amdf_windows_xdna_private_allocation_destroy(
          &memory->private_allocation);
    }
    return status;
  }
  if (memory->pending_paging_fence != 0) {
    const amdf_status_t status = amdf_kmt_wait_for_paging(
        memory->device->kmt, memory->device->device,
        memory->device->paging_sync_object, memory->device->paging_fence,
        memory->pending_paging_fence);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    memory->pending_paging_fence = 0;
  }
  if (memory->resource != 0 || memory->allocation != 0) {
    D3DKMT_DESTROYALLOCATION2 destroy = {0};
    destroy.hDevice = memory->device->device;
    destroy.hResource = memory->resource;
    if (memory->resource == 0) {
      destroy.phAllocationList = &memory->allocation;
      destroy.AllocationCount = 1;
    }
    destroy.Flags.AssumeNotInUse = 1;
    const amdf_status_t status =
        amdf_kmt_make_status(memory->device->kmt->destroy_allocation(&destroy));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    memory->resource = 0;
    memory->allocation = 0;
    memory->device_address = 0;
  }
  if (memory->host_pointer != NULL) {
    if (!VirtualFree(memory->host_pointer, 0, MEM_RELEASE)) {
      return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
    }
    memory->host_pointer = NULL;
    memory->byte_length = 0;
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_windows_xdna_memory_create_allocation(
    amdf_xdna_umd_memory_t* memory) {
  D3DDDI_ALLOCATIONINFO2 allocation_info = {0};
  allocation_info.pSystemMem = memory->host_pointer;
  allocation_info.VidPnSourceId = D3DDDI_ID_UNINITIALIZED;

  D3DKMT_CREATESTANDARDALLOCATION standard_allocation = {0};
  standard_allocation.Type = D3DKMT_STANDARDALLOCATIONTYPE_EXISTINGHEAP;
  standard_allocation.ExistingHeapData.Size =
      (D3DKMT_SIZE_T)memory->byte_length;

  D3DKMT_CREATEALLOCATION create = {0};
  create.hDevice = memory->device->device;
  create.pStandardAllocation = &standard_allocation;
  create.NumAllocations = 1;
  create.pAllocationInfo2 = &allocation_info;
  create.Flags.StandardAllocation = 1;
  create.Flags.ExistingSysMem = 1;
  amdf_status_t status =
      amdf_kmt_make_status(memory->device->kmt->create_allocation(&create));
  if (amdf_status_is_ok(status)) {
    memory->resource = create.hResource;
    memory->allocation = allocation_info.hAllocation;
    if (memory->allocation == 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
  }
  return status;
}

static amdf_status_t amdf_windows_xdna_memory_map_device_address(
    amdf_xdna_umd_memory_t* memory,
    const amdf_memory_address_capabilities_t* address_capabilities,
    amdf_memory_access_t device_access) {
  D3DDDI_MAPGPUVIRTUALADDRESS map = {0};
  map.hPagingQueue = memory->device->paging_queue;
  map.hAllocation = memory->allocation;
  map.MinimumAddress = address_capabilities->minimum_address;
  // Both the native VA and its shim-DMA translation must fit the advertised
  // envelope. Reserve the target's DMA bias before requesting a native range.
  const uint64_t maximum_address = address_capabilities->maximum_address -
                                   memory->device->profile->dma.byte_offset;
  // KMT takes an exclusive, page-aligned end; zero leaves a full-width range
  // unconstrained. The public envelope uses an inclusive final byte.
  map.MaximumAddress = maximum_address == UINT64_MAX ? 0 : maximum_address + 1;
  map.SizeInPages = memory->byte_length / AMDF_WINDOWS_KMT_PAGE_SIZE;
  map.Protection.Write = (device_access & AMDF_MEMORY_ACCESS_WRITE) != 0;
  map.Protection.Execute = (device_access & AMDF_MEMORY_ACCESS_EXECUTE) != 0;
  NTSTATUS native_status = memory->device->kmt->map_gpu_virtual_address(&map);
  if (!amdf_kmt_status_is_success_or_pending(native_status)) {
    return amdf_kmt_make_status(native_status);
  }
  memory->pending_paging_fence = map.PagingFenceValue;
  amdf_status_t status = amdf_kmt_wait_for_paging(
      memory->device->kmt, memory->device->device,
      memory->device->paging_sync_object, memory->device->paging_fence,
      memory->pending_paging_fence);
  if (amdf_status_is_ok(status)) {
    memory->pending_paging_fence = 0;
    memory->device_address = map.VirtualAddress;
    if (memory->device_address < address_capabilities->minimum_address ||
        memory->device_address > maximum_address ||
        memory->byte_length - 1 > maximum_address - memory->device_address ||
        (memory->device_address & (AMDF_WINDOWS_XDNA_ADDRESS_ALIGNMENT - 1)) !=
            0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
  }
  return status;
}

static amdf_status_t amdf_windows_xdna_memory_make_resident(
    amdf_xdna_umd_memory_t* memory) {
  const D3DKMT_HANDLE allocation = memory->allocation;
  D3DDDI_MAKERESIDENT make_resident = {0};
  make_resident.hPagingQueue = memory->device->paging_queue;
  make_resident.NumAllocations = 1;
  make_resident.AllocationList = &allocation;
  const NTSTATUS native_status =
      memory->device->kmt->make_resident(&make_resident);
  if (!amdf_kmt_status_is_success_or_pending(native_status)) {
    return amdf_kmt_make_status(native_status);
  }
  memory->pending_paging_fence = make_resident.PagingFenceValue;
  amdf_status_t status = amdf_kmt_wait_for_paging(
      memory->device->kmt, memory->device->device,
      memory->device->paging_sync_object, memory->device->paging_fence,
      memory->pending_paging_fence);
  if (amdf_status_is_ok(status)) {
    memory->pending_paging_fence = 0;
    if (make_resident.NumAllocations != 1) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
  }
  return status;
}

amdf_status_t amdf_xdna_umd_device_query_memory_profile(
    amdf_xdna_umd_device_t* device, uint32_t memory_profile_ordinal,
    amdf_memory_native_profile_t* out_profile) {
  if (!amdf_kmt_api_supports_memory(device->kmt)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  return amdf_windows_xdna_query_memory_profile(
      device->profile, memory_profile_ordinal, out_profile);
}

void amdf_xdna_umd_context_query_memory_profile(
    amdf_xdna_umd_context_t* context,
    amdf_memory_native_profile_t* out_profile) {
  const uint64_t byte_length = AMDF_WINDOWS_XDNA_PRIVATE_APERTURE_SIZE -
                               AMDF_WINDOWS_XDNA_PRIVATE_BOOTSTRAP_SIZE;
  *out_profile = (amdf_memory_native_profile_t){
      .ordinal = 0,
      .memory_class = AMDF_MEMORY_CLASS_PRIVATE,
      .roles =
          AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
      .guaranteed_flags =
          AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
      .supported_flags =
          AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
      .guaranteed_device_access = AMDF_MEMORY_ACCESS_READ |
                                  AMDF_MEMORY_ACCESS_WRITE |
                                  AMDF_MEMORY_ACCESS_EXECUTE,
      .supported_device_access = AMDF_MEMORY_ACCESS_READ |
                                 AMDF_MEMORY_ACCESS_WRITE |
                                 AMDF_MEMORY_ACCESS_EXECUTE,
      .device_address =
          {
              .address_domain_ordinal = AMDF_ADDRESS_DOMAIN_ORDINAL_NONE,
              .address_bit_count =
                  context->device->profile->dma.address_bit_count,
              .maximum_address =
                  (UINT64_C(1)
                   << context->device->profile->dma.address_bit_count) -
                  1,
              .minimum_alignment = AMDF_WINDOWS_KMT_PAGE_SIZE,
          },
      .allocation =
          {
              .maximum_byte_length = byte_length,
              .byte_length_granularity = byte_length,
              .minimum_alignment = AMDF_WINDOWS_KMT_PAGE_SIZE,
              .maximum_alignment = AMDF_WINDOWS_KMT_PAGE_SIZE,
              .native_byte_length_granularity = AMDF_WINDOWS_KMT_PAGE_SIZE,
          },
      .host_mapping =
          {
              .maximum_byte_length = byte_length,
              .byte_offset_granularity = 1,
              .byte_length_granularity = 1,
              .supported_access =
                  AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
          },
      .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE,
  };
}

amdf_status_t amdf_xdna_umd_memory_prepare_private(
    amdf_xdna_umd_context_t* context,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_create_info_t* create_info,
    amdf_xdna_umd_memory_t** memory_state,
    amdf_xdna_umd_memory_result_t* out_result) {
  amdf_xdna_umd_device_t* device = context->device;
  amdf_xdna_umd_memory_t* memory = NULL;
  amdf_status_t status =
      amdf_calloc(device->host_allocator, sizeof(*memory),
                  amdf_alignof(amdf_xdna_umd_memory_t), (void**)&memory);
  if (!amdf_status_is_ok(status)) return status;
  memory->device = device;
  memory->context = context;
  *memory_state = memory;
  const amdf_windows_xdna_private_allocation_descriptor_t descriptor = {
      .requested_byte_length = AMDF_WINDOWS_XDNA_PRIVATE_APERTURE_SIZE,
      .allocation_byte_length = AMDF_WINDOWS_XDNA_PRIVATE_APERTURE_SIZE,
      .type = 0x3323,
      .policy = 2,
      .xcl_flags = ((context->command_aperture_cookie | 0x100u) << 16) | 1u,
      .selector = 1,
      .flags = AMDF_WINDOWS_XDNA_PRIVATE_ALLOCATION_FLAG_DEVICE_ADDRESS,
  };
  amdf_windows_xdna_private_allocation_initialize(device, &descriptor,
                                                  &memory->private_allocation);
  status = amdf_windows_xdna_kernel_execution_prepare_memory(
      context->kernel_execution, &memory->private_allocation);
  const uint64_t prefix = AMDF_WINDOWS_XDNA_PRIVATE_BOOTSTRAP_SIZE;
  if (amdf_status_is_ok(status)) {
    memory->host_pointer =
        (uint8_t*)memory->private_allocation.host_pointer + prefix;
    memory->byte_length = create_info->byte_length;
    memory->device_address =
        memory->private_allocation.firmware_address + prefix;
    *out_result = (amdf_xdna_umd_memory_result_t){
        .flags = profile->guaranteed_flags,
        .source_byte_offset = prefix,
        .byte_length = create_info->byte_length,
        .alignment = AMDF_WINDOWS_KMT_PAGE_SIZE,
        .native_allocation_byte_length = descriptor.allocation_byte_length,
        .native_allocation_granularity = AMDF_WINDOWS_KMT_PAGE_SIZE,
        .device_address = memory->device_address,
        .address_kinds = profile->address_kinds,
    };
  }
  return status;
}

amdf_status_t amdf_xdna_umd_memory_prepare_import(
    amdf_xdna_umd_device_t* device, const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_import_info_t* import_info,
    const amdf_external_memory_t* external_memory,
    amdf_xdna_umd_memory_t** memory_state,
    amdf_xdna_umd_memory_result_t* out_result) {
  (void)device;
  (void)profile;
  (void)import_info;
  (void)external_memory;
  (void)memory_state;
  (void)out_result;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_status_t amdf_xdna_umd_memory_export(
    amdf_xdna_umd_memory_t* memory,
    const amdf_memory_export_info_t* export_info,
    amdf_external_memory_t* out_value) {
  (void)memory;
  (void)export_info;
  (void)out_value;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

amdf_status_t amdf_xdna_umd_memory_describe_site(
    amdf_xdna_umd_memory_t* memory, const amdf_memory_site_query_t* query,
    amdf_memory_site_description_t* out_description) {
  (void)memory;
  const amdf_queue_family_info_t* family = query->queue_family_info;
  if (family->command_type != AMDF_QUEUE_COMMAND_TYPE_XDNA ||
      family->format_version != AMDF_XDNA_QUEUE_FORMAT_VERSION_1 ||
      (family->roles & AMDF_QUEUE_ROLE_COMPUTE) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_memory_site_description_t description = {0};
  if ((query->access_info->access & AMDF_MEMORY_ACCESS_READ) != 0) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_READ;
  }
  if ((query->access_info->access & AMDF_MEMORY_ACCESS_WRITE) != 0) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_WRITE;
  }
  // Shim DMA accesses resident system backing without another device cache
  // transition. The program must finish the relevant DMA before the caller's
  // ordering edge; host publication/invalidation belongs to its mapping.
  description.release.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
  description.acquire.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
  *out_description = description;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_xdna_umd_memory_prepare(
    amdf_xdna_umd_device_t* device, const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_create_info_t* create_info,
    amdf_xdna_umd_memory_t** memory_state,
    amdf_xdna_umd_memory_result_t* out_result) {
  const uint64_t byte_length = (create_info->byte_length +
                                AMDF_WINDOWS_XDNA_ALLOCATION_GRANULARITY - 1) &
                               ~(AMDF_WINDOWS_XDNA_ALLOCATION_GRANULARITY - 1);

  amdf_xdna_umd_memory_t* memory = NULL;
  amdf_status_t status =
      amdf_calloc(device->host_allocator, sizeof(*memory),
                  amdf_alignof(amdf_xdna_umd_memory_t), (void**)&memory);
  if (!amdf_status_is_ok(status)) return status;
  memory->device = device;
  *memory_state = memory;
  memory->byte_length = byte_length;
  memory->host_pointer = VirtualAlloc(NULL, (SIZE_T)byte_length,
                                      MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (memory->host_pointer == NULL) {
    status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_xdna_memory_create_allocation(memory);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_xdna_memory_map_device_address(
        memory, &profile->device_address, create_info->device_access);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_xdna_memory_make_resident(memory);
  }

  if (amdf_status_is_ok(status)) {
    amdf_xdna_umd_memory_result_t result = {0};
    result.flags = profile->guaranteed_flags;
    result.source_byte_offset = 0;
    result.byte_length = memory->byte_length;
    result.alignment = AMDF_WINDOWS_XDNA_ADDRESS_ALIGNMENT;
    result.native_allocation_byte_length = memory->byte_length;
    result.native_allocation_granularity =
        AMDF_WINDOWS_XDNA_ALLOCATION_GRANULARITY;
    result.physical_backing_id.words[0] =
        (uint64_t)(uintptr_t)memory->host_pointer;
    result.physical_backing_id.words[1] = memory->byte_length;
    result.device_address = memory->device_address;
    result.address_kinds = profile->address_kinds;
    result.dma_address =
        memory->device_address + device->profile->dma.byte_offset;
    *out_result = result;
  }
  return status;
}

amdf_status_t amdf_xdna_umd_memory_destroy(amdf_xdna_umd_memory_t* memory) {
  const amdf_status_t status = amdf_windows_xdna_memory_release_native(memory);
  if (amdf_status_is_ok(status)) {
    amdf_free(memory->device->host_allocator, memory);
  }
  return status;
}

void amdf_xdna_umd_memory_abandon(amdf_xdna_umd_memory_t* memory) {
  amdf_free(memory->device->host_allocator, memory);
}

amdf_status_t amdf_xdna_umd_memory_map(
    amdf_xdna_umd_memory_t* memory,
    const amdf_host_mapping_capabilities_t* capabilities,
    const amdf_memory_map_info_t* map_info,
    amdf_xdna_umd_host_mapping_t** out_mapping,
    amdf_xdna_umd_host_mapping_result_t* out_result) {
  amdf_xdna_umd_host_mapping_t* mapping = NULL;
  amdf_status_t status =
      amdf_calloc(memory->device->host_allocator, sizeof(*mapping),
                  amdf_alignof(amdf_xdna_umd_host_mapping_t), (void**)&mapping);
  if (!amdf_status_is_ok(status)) return status;
  mapping->host_allocator = memory->device->host_allocator;
  mapping->pointer = (uint8_t*)memory->host_pointer + map_info->byte_offset;
  mapping->byte_length = map_info->byte_length;

  amdf_xdna_umd_host_mapping_result_t result = {0};
  result.flags = capabilities->supported_access;
  result.pointer = mapping->pointer;
  result.byte_length = mapping->byte_length;
  result.cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
  result.cache_line_size = amdf_windows_host_cache_line_size();
  result.flush = (amdf_cache_transition_t){
      .kind = AMDF_CACHE_TRANSITION_KIND_RANGE,
      .executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT,
      .host_operation = AMDF_HOST_CACHE_OPERATION_FLUSH,
      .host_instruction = AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH,
      .host_fence_after = AMDF_HOST_CACHE_FENCE_X86_MFENCE,
      .range_granularity = result.cache_line_size,
  };
  result.invalidate = (amdf_cache_transition_t){
      .kind = AMDF_CACHE_TRANSITION_KIND_RANGE,
      .executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT,
      .host_operation = AMDF_HOST_CACHE_OPERATION_INVALIDATE,
      .host_instruction = AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH,
      .host_fence_after = AMDF_HOST_CACHE_FENCE_X86_MFENCE,
      .range_granularity = result.cache_line_size,
  };
  *out_result = result;
  *out_mapping = mapping;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_xdna_umd_host_mapping_cache_control(
    amdf_xdna_umd_host_mapping_t* mapping,
    amdf_host_cache_operation_t operation, uint64_t byte_offset,
    uint64_t byte_length) {
  return amdf_windows_host_cache_control(
      operation, (uint8_t*)mapping->pointer + byte_offset, byte_length);
}

amdf_status_t amdf_xdna_umd_host_mapping_destroy(
    amdf_xdna_umd_host_mapping_t* mapping) {
  amdf_free(mapping->host_allocator, mapping);
  return AMDF_STATUS_OK;
}
