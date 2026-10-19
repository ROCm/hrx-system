// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/wkmi/endpoint_properties.h"

#include <array>
#include <cstring>

#include "amdf/gpu.h"
#include "gtest/gtest.h"

namespace {

amdf_wkmi_bridge_gpu_properties_t MakeProperties() {
  amdf_wkmi_bridge_gpu_properties_t properties = {};
  properties.gfx_ip_major = 12;
  properties.gfx_ip_minor = 5;
  properties.gfx_ip_stepping = 0;
  properties.asic_revision = 0xD1;
  properties.wavefront_size = 32;
  properties.compute_unit_count = 256;
  properties.maximum_wave_count_per_compute_unit = 64;
  properties.maximum_scratch_wave_count_per_compute_unit = 32;
  properties.local_data_share_byte_length = 320u * 1024u;
  properties.xcc_count = 8;
  properties.shader_engine_count = 16;
  properties.supports_pm4_kernel_queue = 1;
  properties.supports_sdma_kernel_queue = 1;
  return properties;
}

TEST(WkmiEndpointPropertiesTest, NormalizesMultiXccTopology) {
  const amdf_wkmi_bridge_gpu_properties_t provider_properties =
      MakeProperties();
  amdf_gpu_endpoint_properties_t properties = {};

  ASSERT_TRUE(amdf_gpu_wddm_wkmi_endpoint_properties_translate(
      &provider_properties, &properties));

  EXPECT_EQ(properties.gfx_ip.major, 12u);
  EXPECT_EQ(properties.gfx_ip.minor, 5u);
  EXPECT_EQ(properties.gfx_ip.stepping, 0u);
  EXPECT_EQ(properties.asic_revision, 1u);
  EXPECT_EQ(properties.compute.wavefront_size, 32u);
  EXPECT_EQ(properties.compute.compute_unit_count, 256u);
  EXPECT_EQ(properties.compute.maximum_wave_count_per_compute_unit, 64u);
  EXPECT_EQ(properties.compute.maximum_scratch_wave_count_per_compute_unit,
            32u);
  EXPECT_EQ(properties.compute.local_data_share_byte_length, 320u * 1024u);
  EXPECT_EQ(properties.topology.xcc_count, 8u);
  EXPECT_EQ(properties.topology.shader_engine_count_per_xcc, 2u);
  ASSERT_EQ(properties.queue_family_count, 2u);
  EXPECT_EQ(properties.queue_families[0].command_type,
            AMDF_QUEUE_COMMAND_TYPE_GPU_PM4);
  EXPECT_EQ(properties.queue_families[0].publication_modes,
            AMDF_QUEUE_PUBLICATION_MODE_KERNEL);
  EXPECT_EQ(properties.queue_families[1].command_type,
            AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);
  EXPECT_EQ(properties.queue_families[1].publication_modes,
            AMDF_QUEUE_PUBLICATION_MODE_KERNEL);

  // Translated queue roles and operation capabilities must form a usable
  // public endpoint profile, without freezing a particular capability table.
  amdf_gpu_endpoint_profile_t profile = {};
  EXPECT_TRUE(amdf_gpu_endpoint_profile_initialize(&properties, &profile));
}

TEST(WkmiEndpointPropertiesTest, NormalizesMissingScratchSlots) {
  amdf_wkmi_bridge_gpu_properties_t provider_properties = MakeProperties();
  provider_properties.maximum_scratch_wave_count_per_compute_unit = 0;
  amdf_gpu_endpoint_properties_t properties = {};

  ASSERT_TRUE(amdf_gpu_wddm_wkmi_endpoint_properties_translate(
      &provider_properties, &properties));

  EXPECT_EQ(properties.compute.maximum_scratch_wave_count_per_compute_unit,
            32u);
}

TEST(WkmiEndpointPropertiesTest, MappingOrderingRequiresHardwareQueueHandle) {
  for (uint32_t hardware : {0u, 1u}) {
    auto provider = MakeProperties();
    provider.supports_sdma_hardware_queue = hardware;
    amdf_gpu_endpoint_properties_t properties = {};
    ASSERT_TRUE(amdf_gpu_wddm_wkmi_endpoint_properties_translate(&provider,
                                                                 &properties));
    ASSERT_EQ(properties.queue_family_count, 2u);
    EXPECT_NE(properties.queue_families[0].kernel_queue_capabilities &
                  AMDF_KERNEL_QUEUE_CAPABILITY_VIRTUAL_MEMORY,
              0u);
    EXPECT_EQ((properties.queue_families[1].kernel_queue_capabilities &
               AMDF_KERNEL_QUEUE_CAPABILITY_VIRTUAL_MEMORY) != 0,
              hardware != 0);
  }
}

TEST(WkmiEndpointPropertiesTest, RdnaExposesPm4AndSdmaPacketFeatures) {
  for (uint32_t target :
       {110000u, 110001u, 110002u, 110003u, 110500u, 110501u, 110502u, 110503u,
        110700u, 110701u, 110702u, 120000u, 120001u, 120500u, 120501u}) {
    SCOPED_TRACE(target);
    auto provider = MakeProperties();
    provider.gfx_ip_major = target / 10000;
    provider.gfx_ip_minor = (target / 100) % 100;
    provider.gfx_ip_stepping = target % 100;
    amdf_gpu_endpoint_properties_t properties = {};
    ASSERT_TRUE(amdf_gpu_wddm_wkmi_endpoint_properties_translate(&provider,
                                                                 &properties));
    ASSERT_EQ(properties.queue_family_count, 2u);
    EXPECT_EQ(properties.queue_families[0].command_type,
              AMDF_QUEUE_COMMAND_TYPE_GPU_PM4);
    EXPECT_EQ(properties.queue_families[0].format_features,
              AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR);
    const auto& sdma = properties.queue_families[1];
    EXPECT_EQ(sdma.command_type, AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);
    const amdf_queue_format_features_t fence =
        target >= 120000 ? AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM
                         : AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_MEMORY_TYPE;
    const amdf_queue_format_features_t scope =
        target >= 120500 ? AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE : 0;
    const amdf_queue_format_features_t rectangle =
        AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT |
        (target >= 120000
             ? AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_WIDE
             : AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_EXTENDED_Z);
    EXPECT_EQ(sdma.format_features, fence | scope | rectangle);
    EXPECT_EQ(sdma.roles, AMDF_QUEUE_ROLE_TRANSFER);
    EXPECT_EQ(sdma.cache_operations, 0u);
  }
}

TEST(WkmiEndpointPropertiesTest, SdmaFieldsFollowNativeEncoding) {
  constexpr std::array<int32_t, 5> kMajors = {9, 10, 11, 12, 12};
  constexpr std::array<int32_t, 5> kMinors = {4, 3, 5, 0, 5};
  constexpr std::array<amdf_queue_format_features_t, 5> kFeatures = {
      AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT,
      AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_MEMORY_TYPE |
          AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT |
          AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_EXTENDED_Z,
      AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_MEMORY_TYPE |
          AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT |
          AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_EXTENDED_Z,
      AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM |
          AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT |
          AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_WIDE,
      AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM |
          AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE |
          AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT |
          AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_WIDE};
  for (size_t i = 0; i < kMajors.size(); ++i) {
    SCOPED_TRACE(i);
    auto native = MakeProperties();
    native.gfx_ip_major = kMajors[i];
    native.gfx_ip_minor = kMinors[i];
    amdf_gpu_endpoint_properties_t properties = {};
    ASSERT_TRUE(
        amdf_gpu_wddm_wkmi_endpoint_properties_translate(&native, &properties));
    ASSERT_EQ(properties.queue_family_count, 2u);
    const auto& family = properties.queue_families[1];
    EXPECT_EQ(family.command_type, AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);
    EXPECT_EQ(family.format_features, kFeatures[i]);
    EXPECT_EQ(family.roles, AMDF_QUEUE_ROLE_TRANSFER);
    EXPECT_EQ(family.cache_operations, 0u);
    EXPECT_EQ(family.cache_transition_kinds, 0u);
  }
}

TEST(WkmiEndpointPropertiesTest, UnknownFamiliesDoNotInferRectangularLayout) {
  for (const auto& ip :
       {std::array<int32_t, 2>{8, 0}, {12, 1}, {12, 6}, {13, 0}}) {
    auto native = MakeProperties();
    native.gfx_ip_major = ip[0];
    native.gfx_ip_minor = ip[1];
    amdf_gpu_endpoint_properties_t properties = {};
    ASSERT_TRUE(
        amdf_gpu_wddm_wkmi_endpoint_properties_translate(&native, &properties));
    ASSERT_EQ(properties.queue_family_count, 2u);
    EXPECT_EQ(properties.queue_families[1].format_features &
                  (AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT |
                   AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_EXTENDED_Z |
                   AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_WIDE),
              0u);
  }
}

TEST(WkmiEndpointPropertiesTest, RejectsUnrepresentableProperties) {
  const amdf_wkmi_bridge_gpu_properties_t valid = MakeProperties();
  EXPECT_FALSE(
      amdf_gpu_wddm_wkmi_endpoint_properties_translate(nullptr, nullptr));
  EXPECT_FALSE(
      amdf_gpu_wddm_wkmi_endpoint_properties_translate(&valid, nullptr));

  amdf_gpu_endpoint_properties_t properties;
  std::memset(&properties, 0xA5, sizeof(properties));
  const amdf_gpu_endpoint_properties_t original = properties;
  amdf_wkmi_bridge_gpu_properties_t provider_properties = valid;
  provider_properties.gfx_ip_major = -1;
  EXPECT_FALSE(amdf_gpu_wddm_wkmi_endpoint_properties_translate(
      &provider_properties, &properties));
  EXPECT_EQ(std::memcmp(&properties, &original, sizeof(properties)), 0);

  provider_properties = valid;
  provider_properties.xcc_count = 0;
  EXPECT_FALSE(amdf_gpu_wddm_wkmi_endpoint_properties_translate(
      &provider_properties, &properties));
  provider_properties = valid;
  provider_properties.shader_engine_count = 0;
  EXPECT_FALSE(amdf_gpu_wddm_wkmi_endpoint_properties_translate(
      &provider_properties, &properties));
  provider_properties = valid;
  provider_properties.shader_engine_count = 15;
  EXPECT_FALSE(amdf_gpu_wddm_wkmi_endpoint_properties_translate(
      &provider_properties, &properties));
}

}  // namespace
