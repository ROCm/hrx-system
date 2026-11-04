// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/memory_profile.h"

#include "libamdf/src/device_state.h"
#include "libamdf/src/gpu/umd/memory.h"

static amdf_memory_site_description_t amdf_gpu_dxg_describe_program_site(
    amdf_memory_access_t access, const amdf_device_t* device,
    amdf_memory_atomic_capabilities_t operations) {
  return amdf_gpu_umd_memory_describe_program_site(access, &device->endpoint_id,
                                                   operations);
}

static amdf_memory_host_description_t amdf_gpu_dxg_describe_host(
    const void* data, amdf_external_memory_type_t external_memory_type,
    amdf_memory_flags_t flags, amdf_host_memory_properties_t properties) {
  (void)external_memory_type;
  const amdf_gpu_dxg_memory_capabilities_t* capabilities = data;
  const bool coherent = (flags & AMDF_MEMORY_FLAG_HOST_COHERENT) != 0;
  amdf_memory_host_description_t result = {
      .properties = properties,
      .cache_line_size = capabilities->host_cache_transition.range_granularity,
      .flush = capabilities->host_cache_transition,
      .invalidate = capabilities->host_cache_transition,
  };
  result.flush.host_operation = AMDF_HOST_CACHE_OPERATION_FLUSH;
  result.invalidate.host_operation = AMDF_HOST_CACHE_OPERATION_INVALIDATE;
  if (!coherent) {
    // The host callback owns both CPU writeback and native allocation
    // publication for noncoherent mappings.
    result.flush.executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API;
    result.flush.host_instruction = AMDF_HOST_CACHE_INSTRUCTION_NONE;
    result.flush.host_fence_before = AMDF_HOST_CACHE_FENCE_NONE;
    result.flush.host_fence_after = AMDF_HOST_CACHE_FENCE_NONE;
  }
  return result;
}

static amdf_memory_atomic_capabilities_t amdf_gpu_dxg_query_program_atomics(
    const amdf_memory_native_profile_t* profile, amdf_memory_flags_t flags) {
  const amdf_gpu_dxg_memory_capabilities_t* capabilities =
      profile->construction.data;
  const uint32_t major = capabilities->program_gfx_major;
  if (major < 9 || major > 12 ||
      (profile->roles & AMDF_MEMORY_PROFILE_ROLE_IMPORT) != 0) {
    return (amdf_memory_atomic_capabilities_t){0};
  }
  const amdf_atomic_operations_t integer =
      amdf_gpu_umd_program_atomic_integer_operations();
  const bool fine = (flags & AMDF_MEMORY_FLAG_HOST_COHERENT) != 0;
  const amdf_atomic_operations_t remote =
      capabilities->program_host_is_fabric &&
              capabilities->program_host_rmw_supported
          ? integer
          : amdf_gpu_umd_program_atomic_pcie_operations(
                capabilities->program_host_rmw_supported != 0);
  const amdf_atomic_operations_t remote_64 =
      capabilities->atomic_update_64_supported
          ? remote
          : remote &
                ~(AMDF_ATOMIC_OPERATION_LOAD | AMDF_ATOMIC_OPERATION_STORE);
  // The DXG private provider's FINE_GRAIN policy selects the coherent access
  // protocol. GC12 additionally distinguishes device/system instructions on
  // cached pages.
  amdf_memory_atomic_capabilities_t result = {
      .device_scope_32 = !fine || major == 12 ? integer : remote,
      .device_scope_64 = !fine || major == 12 ? integer : remote_64,
  };
  if (fine && profile->memory_class == AMDF_MEMORY_CLASS_SYSTEM) {
    result.system_scope_32 = remote;
    result.system_scope_64 = remote_64;
  }
  return result;
}

static uint64_t amdf_gpu_dxg_maximum_address(
    const amdf_gpu_dxg_memory_capabilities_t* capabilities) {
  return capabilities->virtual_address_bit_count == 64
             ? UINT64_MAX
             : (UINT64_C(1) << capabilities->virtual_address_bit_count) - 1;
}

static uint64_t amdf_gpu_dxg_maximum_byte_length(
    const amdf_gpu_dxg_memory_capabilities_t* capabilities) {
  const uint64_t address_span_half =
      capabilities->virtual_address_bit_count == 64
          ? UINT64_MAX / 2
          : UINT64_C(1) << (capabilities->virtual_address_bit_count - 1);
  const uint64_t host_span_half = SIZE_MAX / 2;
  const uint64_t limit =
      address_span_half < host_span_half ? address_span_half : host_span_half;
  return limit & ~(AMDF_GPU_DXG_RESERVATION_GRANULARITY - 1);
}

static uint64_t amdf_gpu_dxg_maximum_alignment(
    const amdf_gpu_dxg_memory_capabilities_t* capabilities) {
  uint64_t limit = amdf_gpu_dxg_maximum_byte_length(capabilities);
  uint64_t alignment = 1;
  while (alignment <= limit / 2) {
    alignment <<= 1;
  }
  return alignment;
}

amdf_status_t amdf_gpu_dxg_query_memory_profile(
    const amdf_gpu_dxg_memory_capabilities_t* capabilities,
    uint32_t memory_profile_ordinal,
    amdf_memory_native_profile_t* out_profile) {
  const uint32_t memory_profile_count =
      capabilities->d3d12_resource_supported ? 4 : 3;
  if (memory_profile_ordinal >= memory_profile_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  if (amdf_gpu_dxg_maximum_byte_length(capabilities) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  amdf_memory_access_t guaranteed_device_access = AMDF_MEMORY_ACCESS_READ;
  if (!capabilities->read_only_memory_supported) {
    guaranteed_device_access |= AMDF_MEMORY_ACCESS_WRITE;
  }
  if (!capabilities->no_execute_memory_supported) {
    guaranteed_device_access |= AMDF_MEMORY_ACCESS_EXECUTE;
  }
  const uint64_t maximum_address = amdf_gpu_dxg_maximum_address(capabilities);
  const uint64_t maximum_byte_length =
      amdf_gpu_dxg_maximum_byte_length(capabilities);
  const uint64_t maximum_alignment =
      amdf_gpu_dxg_maximum_alignment(capabilities);
  amdf_memory_native_profile_t profile = {
      .ordinal = memory_profile_ordinal,
      .query_program_atomics = amdf_gpu_dxg_query_program_atomics,
      .construction = {.data = capabilities},
      .visibility =
          {
              .describe_site = amdf_gpu_umd_memory_describe_site,
              .describe_program = amdf_gpu_dxg_describe_program_site,
              .describe_host = amdf_gpu_dxg_describe_host,
              .data = capabilities,
          },
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
      .minimum_alignment = AMDF_GPU_DXG_RESERVATION_GRANULARITY,
      .maximum_alignment = maximum_alignment,
      .native_byte_length_granularity = AMDF_GPU_DXG_PAGE_SIZE,
  };
  const amdf_host_mapping_capabilities_t host_mapping = {
      .properties = AMDF_HOST_MEMORY_PROPERTY_READ_CACHED |
                    AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK,
      .maximum_byte_length = maximum_byte_length,
      .byte_offset_granularity = 1,
      .byte_length_granularity = 1,
      .supported_access =
          AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
      .features = AMDF_HOST_MAPPING_FEATURE_ATOMIC_ACCESS,
  };
  if (memory_profile_ordinal == 0) {
    profile.memory_class = AMDF_MEMORY_CLASS_SYSTEM;
    profile.roles =
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    profile.guaranteed_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE |
                               AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                               AMDF_MEMORY_FLAG_HOST_FILE_IO;
    profile.supported_flags =
        profile.guaranteed_flags | AMDF_MEMORY_FLAG_QUEUE_STORAGE |
        AMDF_MEMORY_FLAG_MAPPING_SOURCE | AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE |
        AMDF_MEMORY_FLAG_DEVICE_UNCACHED;
    if (capabilities->cache_coherent_memory_supported) {
      profile.supported_flags |= AMDF_MEMORY_FLAG_HOST_COHERENT;
      // DXG CPU-visible updates are at least 32 bits; the adapter reports
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
        AMDF_GPU_DXG_RESERVATION_GRANULARITY;
    profile.allocation = allocation;
    profile.host_mapping = host_mapping;
    profile.host_mapping.features |= AMDF_HOST_MAPPING_FEATURE_FILE_IO |
                                     AMDF_HOST_MAPPING_FEATURE_BUFFERED_FILE_IO;
  } else if (memory_profile_ordinal == 1) {
    profile.memory_class = AMDF_MEMORY_CLASS_LOCAL;
    profile.roles = AMDF_MEMORY_PROFILE_ROLE_CREATE;
    profile.guaranteed_flags =
        AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    profile.supported_flags =
        profile.guaranteed_flags | AMDF_MEMORY_FLAG_QUEUE_STORAGE |
        AMDF_MEMORY_FLAG_MAPPING_SOURCE | AMDF_MEMORY_FLAG_DEVICE_UNCACHED;
    profile.device_address.minimum_alignment =
        AMDF_GPU_DXG_RESERVATION_GRANULARITY;
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
    profile.device_address.minimum_alignment = AMDF_GPU_DXG_PAGE_SIZE;
    profile.registration = (amdf_memory_construction_capabilities_t){
        .maximum_byte_length = maximum_byte_length,
        .byte_length_granularity = AMDF_GPU_DXG_PAGE_SIZE,
        .registered_host_pointer_alignment = AMDF_GPU_DXG_PAGE_SIZE,
        .registered_host_properties = (AMDF_HOST_MEMORY_PROPERTY_READ_CACHED |
                                       AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK),
        .minimum_alignment = AMDF_GPU_DXG_PAGE_SIZE,
        .maximum_alignment = maximum_alignment,
        .native_byte_length_granularity = AMDF_GPU_DXG_PAGE_SIZE,
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
