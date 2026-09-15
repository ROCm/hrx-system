// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/wddm/wkmi/endpoint_properties.h"

#include <cstring>

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
  EXPECT_EQ(properties.queue_families[0].roles,
            AMDF_QUEUE_ROLE_COMPUTE | AMDF_QUEUE_ROLE_TRANSFER |
                AMDF_QUEUE_ROLE_CACHE_CONTROL);
  EXPECT_EQ(properties.queue_families[0].cache_operations,
            AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM);
  EXPECT_EQ(properties.queue_families[0].cache_transition_kinds,
            AMDF_CACHE_TRANSITION_KINDS_GLOBAL);
  EXPECT_EQ(properties.queue_families[1].command_type,
            AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);
  EXPECT_EQ(properties.queue_families[1].publication_modes,
            AMDF_QUEUE_PUBLICATION_MODE_KERNEL);
  EXPECT_EQ(properties.queue_families[1].roles, AMDF_QUEUE_ROLE_TRANSFER);
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
