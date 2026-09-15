// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/drm/memory_profile.h"

#include "libamdf/src/platform/linux/endpoint.h"
#include "libamdf/src/xdna/umd/memory_profile.h"

amdf_status_t amdf_linux_xdna_query_memory_profile(
    const amdf_xdna_endpoint_profile_t* target, size_t page_size,
    uint32_t memory_profile_ordinal,
    amdf_memory_native_profile_t* out_profile) {
  if (memory_profile_ordinal > 2 || target->dma.address_bit_count == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const uint64_t maximum_address =
      (UINT64_C(1) << target->dma.address_bit_count) - 1;
  const uint64_t maximum_byte_length =
      (maximum_address - target->dma.byte_offset + 1) &
      ~(uint64_t)(page_size - 1);
  // SHARE buffers use either caller SVA or a driver-assigned IOVA. Page
  // alignment is the strongest address guarantee common to both modes.
  amdf_memory_native_profile_t profile = {
      .ordinal = memory_profile_ordinal,
      .guaranteed_flags =
          AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
      .supported_flags =
          AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
      .guaranteed_device_access =
          AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .supported_device_access =
          AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      .device_address =
          {
              .address_domain_ordinal = 0,
              .address_bit_count = target->dma.address_bit_count,
              .minimum_address = 0,
              .maximum_address = maximum_address,
              .minimum_alignment = 1,
          },
      .host_mapping =
          {
              .maximum_byte_length = maximum_byte_length,
              .byte_offset_granularity = 1,
              .byte_length_granularity = 1,
              .supported_access =
                  AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
          },
  };
  if (memory_profile_ordinal == 0) {
    profile.memory_class = AMDF_MEMORY_CLASS_SYSTEM;
    profile.roles = AMDF_MEMORY_PROFILE_ROLE_CREATE |
                    AMDF_MEMORY_PROFILE_ROLE_EXPORT |
                    AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    profile.guaranteed_flags |= AMDF_MEMORY_FLAG_SHAREABLE;
    profile.supported_flags |= AMDF_MEMORY_FLAG_SHAREABLE;
    profile.allocation = (amdf_memory_construction_capabilities_t){
        .maximum_byte_length = maximum_byte_length,
        .byte_length_granularity = 1,
        .minimum_alignment = page_size,
        .maximum_alignment = page_size,
        .native_byte_length_granularity = page_size,
    };
    profile.external_memory_support_count = 1;
    profile.external_memory_support[0] = (amdf_external_memory_support_t){
        .type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
        .flags = AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT |
                 AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET |
                 AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_CROSS_PROCESS,
        .source_offset_alignment = 1,
        .byte_length_alignment = 1,
    };
  } else if (memory_profile_ordinal == 1) {
    profile.memory_class = AMDF_MEMORY_CLASS_SYSTEM;
    profile.roles =
        AMDF_MEMORY_PROFILE_ROLE_IMPORT | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    profile.import = (amdf_memory_construction_capabilities_t){
        .maximum_byte_length = maximum_byte_length,
        .byte_length_granularity = 1,
        .minimum_alignment = 1,
        .maximum_alignment = page_size,
        .native_byte_length_granularity = page_size,
    };
    profile.external_memory_support_count = 1;
    profile.external_memory_support[0] = (amdf_external_memory_support_t){
        .type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
        .flags = AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT |
                 AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET |
                 AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_CROSS_PROCESS |
                 AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_FOREIGN_API,
        .source_offset_alignment = 1,
        .byte_length_alignment = 1,
    };
  } else {
    profile.memory_class = AMDF_MEMORY_CLASS_SYSTEM;
    profile.roles =
        AMDF_MEMORY_PROFILE_ROLE_REGISTER | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
    profile.registration = (amdf_memory_construction_capabilities_t){
        .maximum_byte_length = maximum_byte_length,
        .byte_length_granularity = 1,
        .registered_host_pointer_alignment = 1,
        .minimum_alignment = 1,
        .maximum_alignment = page_size,
        .native_byte_length_granularity = page_size,
    };
  }
  profile.address_kinds = (UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE) |
                          (UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_DMA);
  *out_profile = profile;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_xdna_umd_query_endpoint_memory_profile(
    const amdf_platform_endpoint_t* endpoint,
    const amdf_xdna_endpoint_profile_t* target, uint32_t profile_ordinal,
    amdf_memory_native_profile_t* out_profile) {
  return amdf_linux_xdna_query_memory_profile(
      target, endpoint->instance->page_size, profile_ordinal, out_profile);
}
