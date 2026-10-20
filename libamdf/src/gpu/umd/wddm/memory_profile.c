// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/memory_profile.h"

#include "libamdf/src/gpu/umd/memory.h"
#include "libamdf/src/platform/windows/endpoint.h"

amdf_status_t amdf_windows_gpu_query_memory_capabilities(
    const amdf_platform_endpoint_t* endpoint,
    amdf_windows_gpu_memory_capabilities_t* out_capabilities) {
  D3DKMT_QUERY_GPUMMU_CAPS query = {0};
  query.PhysicalAdapterIndex = endpoint->physical_adapter_index;
  const amdf_status_t status = amdf_kmt_query_adapter_info(
      &endpoint->instance->kmt, endpoint->adapter, KMTQAITYPE_QUERY_GPUMMU_CAPS,
      &query, sizeof(query));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (query.Caps.VirtualAddressBitCount == 0 ||
      query.Caps.VirtualAddressBitCount > 64) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }

  D3DKMT_WDDM_2_0_CAPS scheduling = {0};
  const amdf_status_t scheduling_status = amdf_kmt_query_adapter_info(
      &endpoint->instance->kmt, endpoint->adapter, KMTQAITYPE_WDDM_2_0_CAPS,
      &scheduling, sizeof(scheduling));
  if (!amdf_status_is_ok(scheduling_status)) {
    return scheduling_status;
  }

  const amdf_windows_gpu_memory_capabilities_t capabilities = {
      .virtual_address_bit_count = query.Caps.VirtualAddressBitCount,
      .read_only_memory_supported = query.Caps.Flags.ReadOnlyMemorySupported,
      .no_execute_memory_supported = query.Caps.Flags.NoExecuteMemorySupported,
      .cache_coherent_memory_supported =
          query.Caps.Flags.CacheCoherentMemorySupported,
      .atomic_update_64_supported = scheduling.Support64BitAtomics,
  };
  *out_capabilities = capabilities;
  return AMDF_STATUS_OK;
}

static uint64_t amdf_windows_gpu_maximum_address(
    const amdf_windows_gpu_memory_capabilities_t* capabilities) {
  return capabilities->virtual_address_bit_count == 64
             ? UINT64_MAX
             : (UINT64_C(1) << capabilities->virtual_address_bit_count) - 1;
}

static uint64_t amdf_windows_gpu_maximum_byte_length(
    const amdf_windows_gpu_memory_capabilities_t* capabilities) {
  const uint64_t address_span_half =
      capabilities->virtual_address_bit_count == 64
          ? UINT64_MAX / 2
          : UINT64_C(1) << (capabilities->virtual_address_bit_count - 1);
  const uint64_t host_span_half = SIZE_MAX / 2;
  const uint64_t limit =
      address_span_half < host_span_half ? address_span_half : host_span_half;
  return limit & ~(AMDF_WINDOWS_GPU_RESERVATION_GRANULARITY - 1);
}

static uint64_t amdf_windows_gpu_maximum_alignment(
    const amdf_windows_gpu_memory_capabilities_t* capabilities) {
  uint64_t limit = amdf_windows_gpu_maximum_byte_length(capabilities);
  uint64_t alignment = 1;
  while (alignment <= limit / 2) {
    alignment <<= 1;
  }
  return alignment;
}

amdf_status_t amdf_gpu_wddm_query_memory_profile(
    const amdf_windows_gpu_memory_capabilities_t* capabilities,
    uint32_t memory_profile_ordinal,
    amdf_memory_native_profile_t* out_profile) {
  if (memory_profile_ordinal > 3) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  if (amdf_windows_gpu_maximum_byte_length(capabilities) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  amdf_memory_access_t guaranteed_device_access = AMDF_MEMORY_ACCESS_READ;
  if (!capabilities->read_only_memory_supported) {
    guaranteed_device_access |= AMDF_MEMORY_ACCESS_WRITE;
  }
  if (!capabilities->no_execute_memory_supported) {
    guaranteed_device_access |= AMDF_MEMORY_ACCESS_EXECUTE;
  }
  const uint64_t maximum_address =
      amdf_windows_gpu_maximum_address(capabilities);
  const uint64_t maximum_byte_length =
      amdf_windows_gpu_maximum_byte_length(capabilities);
  const uint64_t maximum_alignment =
      amdf_windows_gpu_maximum_alignment(capabilities);
  amdf_memory_native_profile_t profile = {
      .ordinal = memory_profile_ordinal,
      .visibility = {.describe_site = amdf_gpu_umd_memory_describe_site},
      .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
      .guaranteed_device_access = guaranteed_device_access,
      .supported_device_access = AMDF_MEMORY_ACCESS_READ |
                                 AMDF_MEMORY_ACCESS_WRITE |
                                 AMDF_MEMORY_ACCESS_EXECUTE,
      .device_address =
          {
              .address_domain_ordinal = 0,
              .address_bit_count = capabilities->virtual_address_bit_count,
              .minimum_address = 0,
              .maximum_address = maximum_address,
          },
  };
  const amdf_memory_construction_capabilities_t allocation = {
      .maximum_byte_length = maximum_byte_length,
      .byte_length_granularity = 1,
      .minimum_alignment = AMDF_WINDOWS_GPU_RESERVATION_GRANULARITY,
      .maximum_alignment = maximum_alignment,
      .native_byte_length_granularity = AMDF_WINDOWS_GPU_PAGE_SIZE,
  };
  const amdf_host_mapping_capabilities_t host_mapping = {
      .maximum_byte_length = maximum_byte_length,
      .byte_offset_granularity = 1,
      .byte_length_granularity = 1,
      .supported_access =
          AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
  };
  if (memory_profile_ordinal == 0) {
    profile.memory_class = AMDF_MEMORY_CLASS_SYSTEM;
    profile.roles =
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    profile.guaranteed_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE |
                               AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                               AMDF_MEMORY_FLAG_HOST_FILE_IO;
    profile.supported_flags = profile.guaranteed_flags |
                              AMDF_MEMORY_FLAG_QUEUE_STORAGE |
                              AMDF_MEMORY_FLAG_MAPPING_SOURCE;
    if (capabilities->cache_coherent_memory_supported) {
      profile.supported_flags |= AMDF_MEMORY_FLAG_HOST_COHERENT;
      // WDDM's CPU-visible updates are at least 32 bits; the adapter reports
      // whether 64-bit updates are also atomic. These ordinary coherent
      // loads/stores do not require PCIe read-modify-write transactions.
      // Queue encoding and pair reach independently qualify their use.
      profile.atomic_operations_32 =
          AMDF_ATOMIC_OPERATION_WAIT | AMDF_ATOMIC_OPERATION_STORE;
      if (capabilities->atomic_update_64_supported) {
        profile.atomic_operations_64 =
            AMDF_ATOMIC_OPERATION_WAIT | AMDF_ATOMIC_OPERATION_STORE;
      }
    }
    profile.visibility.describe_site =
        amdf_gpu_umd_memory_describe_system_store_site;
    profile.device_address.minimum_alignment =
        AMDF_WINDOWS_GPU_RESERVATION_GRANULARITY;
    profile.allocation = allocation;
    profile.host_mapping = host_mapping;
    profile.host_mapping.features = AMDF_HOST_MAPPING_FEATURE_FILE_IO;
  } else if (memory_profile_ordinal == 1) {
    profile.memory_class = AMDF_MEMORY_CLASS_LOCAL;
    profile.roles = AMDF_MEMORY_PROFILE_ROLE_CREATE;
    profile.guaranteed_flags =
        AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    profile.supported_flags = profile.guaranteed_flags |
                              AMDF_MEMORY_FLAG_QUEUE_STORAGE |
                              AMDF_MEMORY_FLAG_MAPPING_SOURCE;
    profile.device_address.minimum_alignment =
        AMDF_WINDOWS_GPU_RESERVATION_GRANULARITY;
    profile.allocation = allocation;
  } else if (memory_profile_ordinal == 2) {
    profile.memory_class = AMDF_MEMORY_CLASS_SYSTEM;
    profile.roles =
        AMDF_MEMORY_PROFILE_ROLE_REGISTER | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    profile.guaranteed_flags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    profile.supported_flags = profile.guaranteed_flags |
                              AMDF_MEMORY_FLAG_QUEUE_STORAGE |
                              AMDF_MEMORY_FLAG_MAPPING_SOURCE;
    if (capabilities->cache_coherent_memory_supported) {
      profile.supported_flags |= AMDF_MEMORY_FLAG_HOST_COHERENT;
    }
    profile.device_address.minimum_alignment = AMDF_WINDOWS_GPU_PAGE_SIZE;
    profile.registration = (amdf_memory_construction_capabilities_t){
        .maximum_byte_length = maximum_byte_length,
        .byte_length_granularity = AMDF_WINDOWS_GPU_PAGE_SIZE,
        .registered_host_pointer_alignment = AMDF_WINDOWS_GPU_PAGE_SIZE,
        .registered_host_cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK,
        .minimum_alignment = AMDF_WINDOWS_GPU_PAGE_SIZE,
        .maximum_alignment = maximum_alignment,
        .native_byte_length_granularity = AMDF_WINDOWS_GPU_PAGE_SIZE,
    };
    profile.host_mapping = host_mapping;
  } else {
    profile.memory_class = AMDF_MEMORY_CLASS_LOCAL;
    profile.roles =
        AMDF_MEMORY_PROFILE_ROLE_IMPORT | AMDF_MEMORY_PROFILE_ROLE_EXPORT;
    profile.guaranteed_flags = AMDF_MEMORY_FLAG_DEVICE_LOCAL |
                               AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                               AMDF_MEMORY_FLAG_SHAREABLE;
    profile.supported_flags =
        profile.guaranteed_flags | AMDF_MEMORY_FLAG_MAPPING_SOURCE;
    profile.device_address.minimum_alignment = 1;
    profile.import = allocation;
    profile.import.minimum_alignment = 1;
    profile.external_memory_support_count = 1;
    profile.external_memory_support[0] = (amdf_external_memory_support_t){
        .type = AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE,
        .flags = AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT |
                 AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT |
                 AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET |
                 AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_FOREIGN_API,
        .source_offset_alignment = 1,
        .byte_length_alignment = 1,
        .maximum_byte_length = maximum_byte_length,
    };
  }
  *out_profile = profile;
  return AMDF_STATUS_OK;
}
