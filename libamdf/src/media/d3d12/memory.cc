// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/d3d12/memory.h"

#include <cstring>

#include "libamdf/src/host_mapping.h"
#include "libamdf/src/media/d3d12/object.h"

namespace amdf::media::d3d12 {
namespace {

// The native driver switches from small allocation pages to 2 MiB pages as a
// heap grows. Explicitly provisioning full large pages gives every admitted
// logical length the same exact pre-allocation payload geometry. Image
// placement and host feedback keep their independent 64 KiB alignment.
constexpr uint64_t kSharedGranularity = 2 * 1024 * 1024;
constexpr uint64_t kHostGranularity = 65536;

uint32_t Read32(const uint8_t* bytes) {
  uint32_t value;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

uint64_t Read64(const uint8_t* bytes) {
  uint64_t value;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

// Qualifies the allocation's AMD heap record. The live memory owner retains the
// KMT import, including partial construction, until common rollback or release.
// No mapping, residency operation, or queue is needed for this cold import.
amdf_status_t QueryNativeExtent(Memory* memory) {
  Device* device = memory->device;
  D3DKMT_QUERYRESOURCEINFOFROMNTHANDLE query = {
      .hDevice = device->kmt_device,
      .hNtHandle = memory->shared_handle,
  };
  amdf_status_t status = amdf_kmt_make_status(device->query_resource(&query));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (query.NumAllocations != 1 || query.TotalPrivateDriverDataSize < 0x198 ||
      query.TotalPrivateDriverDataSize > 65536 ||
      query.PrivateRuntimeDataSize > 65536 ||
      query.ResourcePrivateDriverDataSize > 65536) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const size_t driver_length = (query.TotalPrivateDriverDataSize + 7u) & ~7u;
  const size_t runtime_length = (query.PrivateRuntimeDataSize + 7u) & ~7u;
  const size_t resource_length =
      (query.ResourcePrivateDriverDataSize + 7u) & ~7u;
  uint8_t* storage = nullptr;
  status = amdf_calloc(device->base.base.host_allocator,
                       driver_length + runtime_length + resource_length, 8,
                       reinterpret_cast<void**>(&storage));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  D3DDDI_OPENALLOCATIONINFO2 allocation = {};
  D3DKMT_OPENRESOURCEFROMNTHANDLE open = {
      .hDevice = device->kmt_device,
      .hNtHandle = memory->shared_handle,
      .NumAllocations = 1,
      .pOpenAllocationInfo2 = &allocation,
      .PrivateRuntimeDataSize = query.PrivateRuntimeDataSize,
      .pPrivateRuntimeData =
          runtime_length == 0 ? nullptr : storage + driver_length,
      .ResourcePrivateDriverDataSize = query.ResourcePrivateDriverDataSize,
      .pResourcePrivateDriverData =
          resource_length == 0 ? nullptr
                               : storage + driver_length + runtime_length,
      .TotalPrivateDriverDataBufferSize = query.TotalPrivateDriverDataSize,
      .pTotalPrivateDriverDataBuffer = storage,
  };
  status = amdf_kmt_make_status(device->open_resource(&open));
  if (amdf_status_is_ok(status)) {
    memory->kmt_resource = open.hResource;
    memory->kmt_allocation = allocation.hAllocation;
    const auto* bytes =
        static_cast<const uint8_t*>(allocation.pPrivateDriverData);
    const uintptr_t pointer = reinterpret_cast<uintptr_t>(bytes);
    const uintptr_t begin = reinterpret_cast<uintptr_t>(storage);
    const uintptr_t end = begin + query.TotalPrivateDriverDataSize;
    const size_t length = allocation.PrivateDriverDataSize;
    bool qualified = pointer >= begin && pointer <= end &&
                     length <= end - pointer && length >= 4;
    if (qualified) {
      const size_t prefix = Read32(bytes) == length ? 0 : 64;
      qualified = length >= prefix + 0x198;
      if (qualified) {
        const uint8_t* record = bytes + prefix;
        qualified = Read32(record) >= 0x198 &&
                    Read32(record) <= length - prefix &&
                    Read32(record + 0xcc) == 32 && Read32(record + 0x194) == 0;
        if (qualified) {
          memory->native_byte_length = Read64(record + 0x68);
        }
      }
    }
    if (!qualified) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
  }
  amdf_free(device->base.base.host_allocator, storage);
  return status;
}

amdf_status_t DescribeSite(const amdf_memory_site_query_t* query,
                           amdf_memory_site_description_t* out_description) {
  if (query->queue_family_info->command_type != AMDF_QUEUE_COMMAND_TYPE_MEDIA) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_memory_site_description_t description = {};
  if ((query->access & AMDF_MEMORY_ACCESS_READ) != 0) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_READ;
  }
  if ((query->access & AMDF_MEMORY_ACCESS_WRITE) != 0) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_WRITE;
  }
  // Preparation records native resource transitions and publication executes
  // them before/after media access. Their visibility is not a coherent no-op;
  // callers still supply the producer/consumer execution dependency.
  description.release.kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL;
  description.release.executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API;
  description.release.operation = AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM;
  description.acquire.kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL;
  description.acquire.executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API;
  description.acquire.operation = AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM;
  *out_description = description;
  return AMDF_STATUS_OK;
}

amdf_memory_host_description_t DescribeHost(const void*,
                                            amdf_external_memory_type_t,
                                            amdf_memory_flags_t) {
  amdf_memory_host_description_t description = {
      .cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK,
  };
  description.flush.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
  description.invalidate.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
  return description;
}

amdf_status_t DescribeMemorySite(
    amdf_memory_t* memory, uint32_t access_ordinal, uint32_t family_ordinal,
    amdf_memory_site_description_t* out_description) {
  const amdf_memory_access_state_t& access = memory->accesses[access_ordinal];
  if (family_ordinal >= access.device->queue_family_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  amdf_memory_site_query_t query = {
      .access = access.info.access,
      .flags = memory->info.flags | access.info.flags,
      .queue_family_info = &access.device->queue_families[family_ordinal],
  };
  return DescribeSite(&query, out_description);
}

void AMDF_CALL ReleaseHandle(void*, amdf_external_memory_type_t,
                             amdf_external_memory_payload_t payload) {
  const BOOL closed = CloseHandle(payload.native_handle);
  amdf_assert(closed && "owned media transport handle must remain valid");
  (void)closed;
}

amdf_status_t ExportMemory(amdf_memory_t* memory, uint32_t access_ordinal,
                           const amdf_memory_export_info_t* export_info,
                           amdf_external_memory_t* out_value) {
  const Memory* native = NativeMemory(memory, access_ordinal);
  HANDLE handle = nullptr;
  if (!DuplicateHandle(GetCurrentProcess(), native->shared_handle,
                       GetCurrentProcess(), &handle, 0, FALSE,
                       DUPLICATE_SAME_ACCESS)) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  amdf_external_memory_t value = {
      .type = AMDF_EXTERNAL_MEMORY_TYPE_D3D12_HEAP,
  };
  value.payload.native_handle = handle;
  value.source_byte_offset = export_info->byte_offset;
  value.byte_length = export_info->byte_length;
  value.release = ReleaseHandle;
  *out_value = value;
  return AMDF_STATUS_OK;
}

void DestroyMapping(amdf_host_mapping_t*) {}

const amdf_host_mapping_vtable_t kMappingOperations = {nullptr, DestroyMapping};

amdf_status_t MapMemory(amdf_memory_t* memory, uint32_t access_ordinal,
                        const amdf_host_mapping_capabilities_t* capabilities,
                        const amdf_memory_map_info_t* map_info,
                        amdf_host_mapping_t** out_mapping) {
  amdf_host_mapping_t* mapping = nullptr;
  const amdf_status_t status = Allocate(memory->host_allocator, &mapping);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_host_mapping_initialize(mapping, &kMappingOperations, memory);
  mapping->info.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
  mapping->info.structure_size = sizeof(mapping->info);
  mapping->info.flags = map_info->flags;
  mapping->info.cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
  mapping->info.pointer =
      static_cast<uint8_t*>(
          NativeMemory(memory, access_ordinal)->host_pointer) +
      map_info->byte_offset;
  mapping->info.memory_byte_offset = map_info->byte_offset;
  mapping->info.byte_length = map_info->byte_length;
  mapping->info.byte_offset_granularity = capabilities->byte_offset_granularity;
  mapping->info.byte_length_granularity = capabilities->byte_length_granularity;
  mapping->info.flush.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
  mapping->info.invalidate.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
  mapping->info.features = capabilities->features;
  *out_mapping = mapping;
  return AMDF_STATUS_OK;
}

amdf_status_t DestroyMemory(amdf_memory_t* memory, uint32_t access_ordinal) {
  Memory* native = NativeMemory(memory, access_ordinal);
  amdf_status_t status = AMDF_STATUS_OK;
  if (native->kmt_resource != 0 || native->kmt_allocation != 0) {
    D3DKMT_DESTROYALLOCATION2 destroy = {
        .hDevice = native->device->kmt_device,
        .hResource = native->kmt_resource,
    };
    if (destroy.hResource == 0) {
      destroy.AllocationCount = 1;
      destroy.phAllocationList = &native->kmt_allocation;
    }
    destroy.Flags.AssumeNotInUse = 1;
    status =
        amdf_kmt_make_status(native->device->kmt->destroy_allocation(&destroy));
    if (amdf_status_is_ok(status)) {
      native->kmt_resource = native->kmt_allocation = 0;
    }
  }
  native->resource.Reset();
  native->heap.Reset();
  if (native->shared_handle != nullptr) {
    if (CloseHandle(native->shared_handle)) {
      native->shared_handle = nullptr;
    } else if (amdf_status_is_ok(status)) {
      status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
    }
  }
  if (native->host_pointer != nullptr) {
    if (VirtualFree(native->host_pointer, 0, MEM_RELEASE)) {
      native->host_pointer = nullptr;
    } else if (amdf_status_is_ok(status)) {
      status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
    }
  }
  return status;
}

uint64_t RemainingBacking(const amdf_memory_t* memory,
                          uint32_t access_ordinal) {
  const auto* native =
      static_cast<const Memory*>(memory->accesses[access_ordinal].native);
  return native->kmt_resource != 0 || native->kmt_allocation != 0 ||
                 native->shared_handle != nullptr ||
                 native->host_pointer != nullptr || native->heap != nullptr ||
                 native->resource != nullptr
             ? native->native_byte_length
             : 0;
}

void AbandonMemory(amdf_memory_t* memory, uint32_t access_ordinal) {
  auto* native = static_cast<Memory*>(memory->accesses[access_ordinal].native);
  // Common teardown may preserve backing after a failed peer detach. Dropping
  // metadata must not implicitly Release a COM owner of still-reachable bytes.
  native->resource.Detach();
  native->heap.Detach();
  Delete(memory->host_allocator, native);
  memory->accesses[access_ordinal].native = nullptr;
}

const amdf_memory_vtable_t kMemoryOperations = {
    ExportMemory,  DescribeMemorySite, MapMemory,    nullptr,
    DestroyMemory, RemainingBacking,   AbandonMemory};

}  // namespace

amdf_status_t QueryMemoryProfile(amdf_device_t* base, uint32_t ordinal,
                                 amdf_memory_native_profile_t* out_profile) {
  if (ordinal > kFeedbackBuffer) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const auto* device = reinterpret_cast<const Device*>(base);
  amdf_memory_native_profile_t profile = {
      .ordinal = ordinal,
      .roles = AMDF_MEMORY_PROFILE_ROLE_CREATE,
      .guaranteed_device_access =
          AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
  };
  profile.supported_device_access = profile.guaranteed_device_access;
  profile.device_address.address_domain_ordinal =
      AMDF_ADDRESS_DOMAIN_ORDINAL_NONE;
  profile.allocation.maximum_byte_length = device->maximum_memory_byte_length;
  profile.allocation.byte_length_granularity = 1;
  profile.allocation.minimum_alignment = kHostGranularity;
  profile.allocation.maximum_alignment = kHostGranularity;
  profile.visibility.describe_site = DescribeSite;
  if (ordinal == kSharedHeap) {
    profile.allocation.maximum_byte_length &= ~(kSharedGranularity - 1);
    profile.memory_class = AMDF_MEMORY_CLASS_LOCAL;
    profile.roles |= AMDF_MEMORY_PROFILE_ROLE_EXPORT;
    profile.guaranteed_flags =
        AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_SHAREABLE;
    profile.allocation.native_byte_length_granularity = kSharedGranularity;
    profile.external_memory_support_count = 1;
    auto& transport = profile.external_memory_support[0];
    transport.type = AMDF_EXTERNAL_MEMORY_TYPE_D3D12_HEAP;
    transport.flags = AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT |
                      AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET |
                      AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_FOREIGN_API;
    transport.source_offset_alignment = 1;
    transport.byte_length_alignment = 1;
    transport.maximum_byte_length = device->maximum_memory_byte_length;
  } else {
    profile.memory_class = AMDF_MEMORY_CLASS_SYSTEM;
    profile.roles |= AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    profile.guaranteed_flags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_HOST_COHERENT;
    profile.allocation.native_byte_length_granularity = kHostGranularity;
    profile.host_mapping.maximum_byte_length =
        device->maximum_memory_byte_length;
    profile.host_mapping.byte_offset_granularity = 1;
    profile.host_mapping.byte_length_granularity = 1;
    profile.host_mapping.supported_access =
        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    profile.host_mapping.features = AMDF_HOST_MAPPING_FEATURE_ATOMIC_ACCESS;
    profile.visibility.describe_host = DescribeHost;
    if (!device->existing_heaps) {
      profile.availability = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
  }
  profile.supported_flags = profile.guaranteed_flags;
  *out_profile = profile;
  return AMDF_STATUS_OK;
}

amdf_status_t PrepareMemory(amdf_memory_t* memory,
                            const amdf_memory_native_group_t* group,
                            const amdf_memory_native_create_info_t* create_info,
                            amdf_memory_info_t* out_info) {
  const uint32_t access_ordinal = group->access_ordinals[0];
  auto& access = memory->accesses[access_ordinal];
  auto* device = reinterpret_cast<Device*>(access.device);
  const auto& profile = group->profiles[access_ordinal];
  Memory* native = nullptr;
  amdf_status_t status = Allocate(memory->host_allocator, &native);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  access.vtable = &kMemoryOperations;
  access.native = native;
  native->device = device;
  native->profile = static_cast<MemoryProfile>(profile.ordinal);
  const uint64_t heap_byte_length =
      (create_info->byte_length + kHostGranularity - 1) &
      ~(kHostGranularity - 1);
  const uint64_t expected_native_byte_length =
      (create_info->byte_length +
       profile.allocation.native_byte_length_granularity - 1) &
      ~(profile.allocation.native_byte_length_granularity - 1);
  // Account for acquired payload even if a later native operation fails.
  native->native_byte_length = expected_native_byte_length;
  if (native->profile == kSharedHeap) {
    D3D12_HEAP_DESC description = {
        .SizeInBytes = expected_native_byte_length,
        .Alignment = kHostGranularity,
    };
    description.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    description.Properties.CreationNodeMask = 1;
    description.Properties.VisibleNodeMask = 1;
    description.Flags = D3D12_HEAP_FLAG_SHARED;
    status = Hresult(
        device->native->CreateHeap(&description, IID_PPV_ARGS(&native->heap)));
    if (amdf_status_is_ok(status)) {
      status = Hresult(device->native->CreateSharedHandle(
          native->heap.Get(), nullptr, GENERIC_ALL, nullptr,
          &native->shared_handle));
    }
    if (amdf_status_is_ok(status)) {
      status = QueryNativeExtent(native);
    }
    if (amdf_status_is_ok(status) &&
        native->native_byte_length != expected_native_byte_length) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
  } else {
    native->host_pointer =
        VirtualAlloc(nullptr, static_cast<SIZE_T>(heap_byte_length),
                     MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    status = native->host_pointer != nullptr
                 ? AMDF_STATUS_OK
                 : amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
    if (amdf_status_is_ok(status)) {
      status = Hresult(device->native->OpenExistingHeapFromAddress(
          native->host_pointer, IID_PPV_ARGS(&native->heap)));
    }
    if (amdf_status_is_ok(status)) {
      const auto description = native->heap->GetDesc();
      if (description.SizeInBytes != heap_byte_length ||
          description.Properties.Type != D3D12_HEAP_TYPE_CUSTOM ||
          description.Properties.CPUPageProperty !=
              D3D12_CPU_PAGE_PROPERTY_WRITE_BACK ||
          description.Properties.MemoryPoolPreference != D3D12_MEMORY_POOL_L0 ||
          (description.Flags & D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER) == 0) {
        status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
      }
    }
  }
  if (amdf_status_is_ok(status)) {
    D3D12_RESOURCE_DESC description = {
        .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
        .Width = heap_byte_length,
    };
    description.Height = description.DepthOrArraySize = description.MipLevels =
        1;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    const bool feedback = native->profile == kFeedbackBuffer;
    if (feedback) {
      description.Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    }
    status = Hresult(device->native->CreatePlacedResource(
        native->heap.Get(), 0, &description,
        feedback ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COMMON,
        nullptr, IID_PPV_ARGS(&native->resource)));
  }
  if (amdf_status_is_ok(status)) {
    amdf_device_info_t device_info = {};
    device->base.base.vtable->query_info(&device->base.base, &device_info);
    access.info.access = create_info->device_access;
    access.info.device_id = device_info.id;
    access.info.reset_epoch = device_info.reset_epoch;
    access.info.flags = profile.guaranteed_flags & AMDF_MEMORY_ACCESS_FLAGS;
    access.info.address_domain_ordinal = AMDF_ADDRESS_DOMAIN_ORDINAL_NONE;
    // Core construction already owns the consumer count and backing ordinal.
    // Native preparation contributes backing facts without replacing that
    // lifetime metadata, which also drives peer detach and terminal release.
    out_info->type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    out_info->structure_size = sizeof(*out_info);
    out_info->memory_profile_ordinal = profile.ordinal;
    out_info->memory_class = profile.memory_class;
    out_info->flags = profile.guaranteed_flags & AMDF_MEMORY_BACKING_FLAGS;
    out_info->byte_length = create_info->byte_length;
    out_info->alignment = kHostGranularity;
    out_info->native_allocation_byte_length = native->native_byte_length;
    out_info->native_allocation_granularity =
        profile.allocation.native_byte_length_granularity;
  }
  return status;
}

amdf_status_t PrepareMemoryImport(amdf_memory_t*, uint32_t,
                                  const amdf_memory_native_profile_t*,
                                  const amdf_memory_native_import_info_t*,
                                  const amdf_external_memory_t*,
                                  amdf_memory_info_t*) {
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

}  // namespace amdf::media::d3d12
