// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/memory_profile.h"

#include "gtest/gtest.h"

namespace {

TEST(DxgMemoryProfileTest, ReportsExactHostMappingFeatures) {
  const amdf_gpu_dxg_memory_capabilities_t capabilities = {
      .virtual_address_bit_count = 48,
  };
  amdf_memory_native_profile_t profile = {};

  ASSERT_EQ(amdf_gpu_dxg_query_memory_profile(&capabilities, 0, &profile),
            AMDF_STATUS_OK);
  EXPECT_EQ(profile.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
  EXPECT_EQ(profile.roles, AMDF_MEMORY_PROFILE_ROLE_CREATE |
                               AMDF_MEMORY_PROFILE_ROLE_HOST_MAP);
  EXPECT_EQ(profile.host_mapping.features,
            AMDF_HOST_MAPPING_FEATURE_ATOMIC_ACCESS |
                AMDF_HOST_MAPPING_FEATURE_FILE_IO |
                AMDF_HOST_MAPPING_FEATURE_BUFFERED_FILE_IO);

  ASSERT_EQ(amdf_gpu_dxg_query_memory_profile(&capabilities, 1, &profile),
            AMDF_STATUS_OK);
  EXPECT_EQ(profile.memory_class, AMDF_MEMORY_CLASS_LOCAL);
  EXPECT_EQ(profile.roles, AMDF_MEMORY_PROFILE_ROLE_CREATE);
  EXPECT_EQ(profile.host_mapping.features, 0u);

  ASSERT_EQ(amdf_gpu_dxg_query_memory_profile(&capabilities, 2, &profile),
            AMDF_STATUS_OK);
  EXPECT_EQ(profile.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
  EXPECT_EQ(profile.roles, AMDF_MEMORY_PROFILE_ROLE_REGISTER |
                               AMDF_MEMORY_PROFILE_ROLE_HOST_MAP);
  EXPECT_EQ(profile.host_mapping.features,
            AMDF_HOST_MAPPING_FEATURE_ATOMIC_ACCESS);
}

TEST(DxgMemoryProfileTest, ReportsExactAllocationFlagSupport) {
  const amdf_gpu_dxg_memory_capabilities_t capabilities = {
      .virtual_address_bit_count = 48,
      .cache_coherent_memory_supported = 1,
      .d3d12_resource_supported = 1,
  };
  amdf_memory_native_profile_t profile = {};

  ASSERT_EQ(amdf_gpu_dxg_query_memory_profile(&capabilities, 0, &profile),
            AMDF_STATUS_OK);
  EXPECT_EQ(
      profile.supported_flags,
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_HOST_COHERENT |
          AMDF_MEMORY_FLAG_DEVICE_ADDRESS | AMDF_MEMORY_FLAG_MAPPING_SOURCE |
          AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE | AMDF_MEMORY_FLAG_HOST_FILE_IO |
          AMDF_MEMORY_FLAG_QUEUE_STORAGE | AMDF_MEMORY_FLAG_DEVICE_UNCACHED);

  ASSERT_EQ(amdf_gpu_dxg_query_memory_profile(&capabilities, 1, &profile),
            AMDF_STATUS_OK);
  EXPECT_EQ(profile.supported_flags, AMDF_MEMORY_FLAG_DEVICE_LOCAL |
                                         AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                                         AMDF_MEMORY_FLAG_MAPPING_SOURCE |
                                         AMDF_MEMORY_FLAG_QUEUE_STORAGE |
                                         AMDF_MEMORY_FLAG_DEVICE_UNCACHED);

  ASSERT_EQ(amdf_gpu_dxg_query_memory_profile(&capabilities, 2, &profile),
            AMDF_STATUS_OK);
  EXPECT_EQ(profile.supported_flags, AMDF_MEMORY_FLAG_HOST_VISIBLE |
                                         AMDF_MEMORY_FLAG_HOST_COHERENT |
                                         AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                                         AMDF_MEMORY_FLAG_MAPPING_SOURCE |
                                         AMDF_MEMORY_FLAG_QUEUE_STORAGE);

  ASSERT_EQ(amdf_gpu_dxg_query_memory_profile(&capabilities, 3, &profile),
            AMDF_STATUS_OK);
  EXPECT_EQ(profile.supported_flags, AMDF_MEMORY_FLAG_DEVICE_LOCAL |
                                         AMDF_MEMORY_FLAG_SHAREABLE |
                                         AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                                         AMDF_MEMORY_FLAG_MAPPING_SOURCE);
}

TEST(DxgMemoryProfileTest, ExternalResourceProfileRequiresHostSupport) {
  amdf_gpu_dxg_memory_capabilities_t capabilities = {
      .virtual_address_bit_count = 48,
  };
  amdf_memory_native_profile_t profile = {.ordinal = 73};

  EXPECT_EQ(amdf_gpu_dxg_query_memory_profile(&capabilities, 3, &profile),
            amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
  EXPECT_EQ(profile.ordinal, 73u);

  capabilities.d3d12_resource_supported = 1;
  ASSERT_EQ(amdf_gpu_dxg_query_memory_profile(&capabilities, 3, &profile),
            AMDF_STATUS_OK);
  EXPECT_EQ(profile.roles,
            AMDF_MEMORY_PROFILE_ROLE_IMPORT | AMDF_MEMORY_PROFILE_ROLE_EXPORT);
  ASSERT_EQ(profile.external_memory_support_count, 1u);
  EXPECT_EQ(profile.external_memory_support[0].type,
            AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE);
}

}  // namespace
