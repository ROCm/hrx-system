// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/mcdm/memory_profile.h"

#include "libamdf/src/platform/windows/endpoint.h"
#include "libamdf/src/xdna/umd/memory_profile.h"

amdf_status_t amdf_windows_xdna_query_memory_profile(
    const amdf_xdna_endpoint_profile_t* target, uint32_t memory_profile_ordinal,
    amdf_memory_native_profile_t* out_profile) {
  if (memory_profile_ordinal != 0 || target->dma.address_bit_count == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const uint32_t address_bit_count = target->dma.address_bit_count;
  const uint64_t maximum_address = UINT64_MAX >> (64 - address_bit_count);
  const uint64_t address_capacity = maximum_address - target->dma.byte_offset -
                                    AMDF_WINDOWS_XDNA_ALLOCATION_GRANULARITY +
                                    1;
  const uint64_t maximum_byte_length =
      (address_capacity < SIZE_MAX ? address_capacity : SIZE_MAX) &
      ~(AMDF_WINDOWS_XDNA_ALLOCATION_GRANULARITY - 1);
  *out_profile = (amdf_memory_native_profile_t){
      .ordinal = 0,
      .address_kinds = (UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE) |
                       (UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_DMA),
      .memory_class = AMDF_MEMORY_CLASS_SYSTEM,
      .roles =
          AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
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
              .address_bit_count = address_bit_count,
              .minimum_address = AMDF_WINDOWS_XDNA_ALLOCATION_GRANULARITY,
              .maximum_address = maximum_address,
              .minimum_alignment = AMDF_WINDOWS_XDNA_ADDRESS_ALIGNMENT,
          },
      .allocation =
          {
              .maximum_byte_length = maximum_byte_length,
              .byte_length_granularity = 1,
              .minimum_alignment = AMDF_WINDOWS_XDNA_ADDRESS_ALIGNMENT,
              .maximum_alignment = AMDF_WINDOWS_XDNA_ADDRESS_ALIGNMENT,
              .native_byte_length_granularity =
                  AMDF_WINDOWS_XDNA_ALLOCATION_GRANULARITY,
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
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_xdna_umd_query_endpoint_memory_profile(
    const amdf_platform_endpoint_t* endpoint,
    const amdf_xdna_endpoint_profile_t* target, uint32_t profile_ordinal,
    amdf_memory_native_profile_t* out_profile) {
  if (!amdf_kmt_api_supports_memory(&endpoint->instance->kmt)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  return amdf_windows_xdna_query_memory_profile(target, profile_ordinal,
                                                out_profile);
}
