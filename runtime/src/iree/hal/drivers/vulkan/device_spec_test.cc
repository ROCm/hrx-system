// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/device_spec.h"

#include <cstring>
#include <vector>

#include "iree/hal/drivers/vulkan/device_spec_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "vulkan/vulkan_core.h"

namespace iree::hal::vulkan {
namespace {

static iree_hal_vulkan_device_spec_t MakeTestSpec() {
  return {
      /*.api_version=*/VK_MAKE_API_VERSION(0, 1, 3, 0),
      /*.driver_version=*/1234,
      /*.physical_device_type=*/VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
      /*.enabled_features=*/IREE_HAL_VULKAN_FEATURE_REQUIRED_BASELINE |
          IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16 |
          IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL,
      /*.flags=*/IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_NONE,
  };
}

TEST(DeviceSpecTest, EncodesAndDecodesPayload) {
  iree_hal_vulkan_device_spec_t source = MakeTestSpec();
  iree_hal_vulkan_cooperative_matrix_property_t source_property = {
      /*.m_size=*/16,
      /*.n_size=*/8,
      /*.k_size=*/16,
      /*.a_type=*/1,
      /*.b_type=*/2,
      /*.c_type=*/3,
      /*.result_type=*/4,
      /*.saturating_accumulation=*/1,
      /*.scope=*/5,
  };
  iree_host_size_t payload_size = 0;
  IREE_ASSERT_OK(
      iree_hal_vulkan_device_spec_calculate_payload_size(1, &payload_size));
  std::vector<uint8_t> payload_storage(payload_size);
  IREE_ASSERT_OK(iree_hal_vulkan_device_spec_encode(
      &source, 1, &source_property,
      iree_make_byte_span(payload_storage.data(), payload_storage.size())));

  iree_hal_vulkan_device_spec_t decoded = {0};
  IREE_ASSERT_OK(iree_hal_vulkan_device_spec_decode(
      iree_make_const_byte_span(payload_storage.data(), payload_storage.size()),
      &decoded));
  EXPECT_EQ(decoded.api_version, VK_MAKE_API_VERSION(0, 1, 3, 0));
  EXPECT_EQ(decoded.driver_version, 1234u);
  EXPECT_EQ(decoded.physical_device_type, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
  EXPECT_EQ(decoded.enabled_features, source.enabled_features);
  EXPECT_EQ(decoded.flags, source.flags);
  ASSERT_EQ(decoded.cooperative_matrix.count, 1);
  iree_hal_vulkan_cooperative_matrix_property_t decoded_property = {};
  ASSERT_TRUE(iree_hal_vulkan_device_spec_read_cooperative_matrix_property(
      &decoded, 0, &decoded_property));
  EXPECT_EQ(decoded_property.m_size, source_property.m_size);
  EXPECT_EQ(decoded_property.k_size, source_property.k_size);
  EXPECT_EQ(decoded_property.result_type, source_property.result_type);
  EXPECT_EQ(decoded_property.scope, source_property.scope);
  EXPECT_FALSE(iree_hal_vulkan_device_spec_read_cooperative_matrix_property(
      &decoded, 1, &decoded_property));
}

TEST(DeviceSpecTest, RejectsMalformedPayloads) {
  iree_hal_vulkan_device_spec_t source = MakeTestSpec();
  iree_host_size_t payload_size = 0;
  IREE_ASSERT_OK(
      iree_hal_vulkan_device_spec_calculate_payload_size(0, &payload_size));
  std::vector<uint8_t> canonical_payload(payload_size);
  IREE_ASSERT_OK(iree_hal_vulkan_device_spec_encode(
      &source, 0, nullptr,
      iree_make_byte_span(canonical_payload.data(), canonical_payload.size())));

  iree_hal_vulkan_device_spec_t decoded = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_vulkan_device_spec_encode(
                            &source, 0, nullptr,
                            iree_make_byte_span(canonical_payload.data(),
                                                canonical_payload.size() - 1)));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_vulkan_device_spec_decode(
          iree_make_const_byte_span(canonical_payload.data(),
                                    canonical_payload.size() - 1),
          &decoded));

  std::vector<uint8_t> malformed_payload = canonical_payload;
  malformed_payload[0] ^= 0xff;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_vulkan_device_spec_decode(
                            iree_make_const_byte_span(malformed_payload.data(),
                                                      malformed_payload.size()),
                            &decoded));

  malformed_payload = canonical_payload;
  malformed_payload[4] = 0xff;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_vulkan_device_spec_decode(
                            iree_make_const_byte_span(malformed_payload.data(),
                                                      malformed_payload.size()),
                            &decoded));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_vulkan_device_spec_encode(
          &source, 0, nullptr, iree_make_byte_span(NULL, payload_size)));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_vulkan_device_spec_decode(
          iree_make_const_byte_span(NULL, payload_size), &decoded));
}

TEST(DeviceSpecTest, AddsAndFindsCoreFacet) {
  iree_hal_vulkan_device_spec_t source = MakeTestSpec();
  iree_hal_vulkan_cooperative_matrix_property_t source_property = {
      /*.m_size=*/16,
      /*.n_size=*/16,
      /*.k_size=*/8,
      /*.a_type=*/1,
      /*.b_type=*/1,
      /*.c_type=*/2,
      /*.result_type=*/2,
      /*.saturating_accumulation=*/0,
      /*.scope=*/3,
  };

  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(iree_hal_vulkan_device_spec_builder_add_facet(
      &builder, &source, 1, &source_property));

  iree_hal_device_spec_t* device_spec = NULL;
  IREE_ASSERT_OK(iree_hal_device_spec_builder_finalize(&builder, &device_spec));
  const iree_hal_device_spec_facet_t* facet =
      iree_hal_vulkan_device_spec_find_facet(device_spec);
  ASSERT_NE(facet, nullptr);

  iree_hal_vulkan_device_spec_t decoded = {0};
  IREE_ASSERT_OK(iree_hal_vulkan_device_spec_decode_facet(facet, &decoded));
  EXPECT_EQ(decoded.enabled_features, source.enabled_features);
  EXPECT_EQ(decoded.physical_device_type, source.physical_device_type);
  ASSERT_EQ(decoded.cooperative_matrix.count, 1);

  iree_hal_device_spec_release(device_spec);
  iree_hal_device_spec_builder_deinitialize(&builder);
}

TEST(DeviceSpecTest, CreatesSpecFromParams) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(IREE_SV("test"), iree_allocator_system(),
                                     iree_allocator_system(), &allocator));

  iree_hal_vulkan_physical_device_snapshot_t physical_device = {};
  physical_device.ordinal = 3;
  std::strncpy(physical_device.properties2.properties.deviceName,
               "Vulkan test device",
               sizeof(physical_device.properties2.properties.deviceName) - 1);
  physical_device.properties2.properties.apiVersion =
      VK_MAKE_API_VERSION(0, 1, 3, 0);
  physical_device.properties2.properties.driverVersion = 1234;
  physical_device.properties2.properties.vendorID = 0x1002;
  physical_device.properties2.properties.deviceID = 0x744c;
  physical_device.properties2.properties.deviceType =
      VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  physical_device.properties2.properties.limits.maxComputeWorkGroupInvocations =
      1024;
  physical_device.properties2.properties.limits.maxComputeWorkGroupSize[0] =
      1024;
  physical_device.properties2.properties.limits.maxComputeWorkGroupSize[1] =
      1024;
  physical_device.properties2.properties.limits.maxComputeWorkGroupSize[2] = 64;
  physical_device.properties2.properties.limits.maxComputeWorkGroupCount[0] =
      65535;
  physical_device.properties2.properties.limits.maxComputeWorkGroupCount[1] =
      65535;
  physical_device.properties2.properties.limits.maxComputeWorkGroupCount[2] =
      65535;
  physical_device.properties2.properties.limits.maxComputeSharedMemorySize =
      64 * 1024;
  physical_device.properties2.properties.limits.maxStorageBufferRange = 1ull
                                                                        << 30;
  physical_device.properties2.properties.limits
      .minStorageBufferOffsetAlignment = 16;
  physical_device.properties2.properties.limits.timestampPeriod = 1.0f;
  physical_device.properties11.maxMemoryAllocationSize = 1ull << 32;
  physical_device.id_properties.deviceUUID[0] = 0x11;
  physical_device.subgroup_properties.subgroupSize = 32;
  physical_device.subgroup_size_control_properties.minSubgroupSize = 32;
  physical_device.subgroup_size_control_properties.maxSubgroupSize = 64;
  physical_device.calibrated_timestamp_time_domains =
      IREE_HAL_VULKAN_TIME_DOMAIN_DEVICE |
      IREE_HAL_VULKAN_TIME_DOMAIN_CLOCK_MONOTONIC;
  VkCooperativeMatrixPropertiesKHR cooperative_matrix_property = {};
  cooperative_matrix_property.sType =
      VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
  cooperative_matrix_property.MSize = 16;
  cooperative_matrix_property.NSize = 16;
  cooperative_matrix_property.KSize = 16;
  cooperative_matrix_property.AType = VK_COMPONENT_TYPE_FLOAT16_KHR;
  cooperative_matrix_property.BType = VK_COMPONENT_TYPE_FLOAT16_KHR;
  cooperative_matrix_property.CType = VK_COMPONENT_TYPE_FLOAT32_KHR;
  cooperative_matrix_property.ResultType = VK_COMPONENT_TYPE_FLOAT32_KHR;
  cooperative_matrix_property.scope = VK_SCOPE_SUBGROUP_KHR;
  physical_device.cooperative_matrix_property_count = 1;
  physical_device.cooperative_matrix_property_rows =
      &cooperative_matrix_property;

  iree_hal_vulkan_device_plan_t device_plan = {};
  device_plan.request_flags = IREE_HAL_VULKAN_REQUEST_FLAG_TRACING;
  device_plan.enabled_features =
      IREE_HAL_VULKAN_FEATURE_REQUIRED_BASELINE |
      IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES |
      IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL;
  device_plan.enabled_extensions =
      IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_CALIBRATED_TIMESTAMPS;
  device_plan.queue_assignment.queue_count = 2;
  device_plan.queue_assignment.compute.affinity = 1ull << 0;
  device_plan.queue_assignment.transfer.affinity = 1ull << 1;
  device_plan.queue_assignment.compute.timestamp_valid_bits = 64;
  device_plan.queue_assignment.transfer.timestamp_valid_bits = 64;

  iree_hal_vulkan_device_spec_params_t params = {
      /*.logical_device_id=*/IREE_SV("vulkan://0"),
      /*.display_name=*/IREE_SV("Vulkan test logical device"),
      /*.physical_device=*/&physical_device,
      /*.device_plan=*/&device_plan,
      /*.device_allocator=*/allocator,
  };
  iree_hal_device_spec_t* device_spec = NULL;
  IREE_ASSERT_OK(iree_hal_vulkan_device_spec_create(
      &params, iree_allocator_system(), &device_spec));

  const iree_hal_device_identity_spec_t* identity =
      iree_hal_device_spec_identity(device_spec);
  ASSERT_NE(identity, nullptr);
  EXPECT_TRUE(iree_string_view_equal(identity->driver_id, IREE_SV("vulkan")));
  EXPECT_TRUE(iree_string_view_equal(identity->backend_id, IREE_SV("vulkan")));
  EXPECT_EQ(identity->vendor_id, 0x1002);
  EXPECT_EQ(identity->device_id, 0x744c);
  ASSERT_EQ(identity->physical_device_count, 1);
  EXPECT_EQ(identity->physical_devices[0].physical_ordinal, 3);
  EXPECT_EQ(identity->physical_devices[0].identity.uuid.bytes[0], 0x11);

  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(device_spec);
  ASSERT_NE(queues, nullptr);
  ASSERT_EQ(queues->family_count, 1);
  EXPECT_EQ(queues->families[0].queue_count, 2);
  EXPECT_EQ(queues->families[0].queue_affinity, 3u);
  EXPECT_EQ(queues->families[0].timestamp_frequency_hz, 1000000000ull);

  const iree_hal_device_dispatch_spec_t* dispatch =
      iree_hal_device_spec_dispatch(device_spec);
  ASSERT_NE(dispatch, nullptr);
  EXPECT_EQ(dispatch->launch.maximum_workgroup_invocations, 1024);
  EXPECT_EQ(dispatch->launch.maximum_workgroup_size[2], 64);
  EXPECT_EQ(dispatch->subgroup.default_size, 32);
  EXPECT_EQ(dispatch->subgroup.supported_size_mask, 1ull << 32);
  EXPECT_EQ(dispatch->execution.maximum_workgroup_local_memory_size, 64 * 1024);
  EXPECT_EQ(dispatch->addressing.minimum_buffer_device_address_alignment, 16);

  const iree_hal_device_timing_spec_t* timing =
      iree_hal_device_spec_timing(device_spec);
  ASSERT_NE(timing, nullptr);
  EXPECT_EQ(timing->timestamp_valid_bits, 64);
  EXPECT_TRUE(iree_all_bits_set(
      timing->flags, IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS |
                         IREE_HAL_DEVICE_TIMING_SPEC_FLAG_HOST_CORRELATION |
                         IREE_HAL_DEVICE_TIMING_SPEC_FLAG_TRACE_CAPTURE));

  const iree_hal_device_executable_spec_t* executables =
      iree_hal_device_spec_executables(device_spec);
  ASSERT_NE(executables, nullptr);
  iree_hal_executable_target_selection_t target_selection = {
      /*.family=*/IREE_SV("spirv"),
      /*.target_key=*/IREE_SV("vulkan1.3+bda"),
      /*.kind_flags=*/IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_GENERIC,
      /*.physical_device_affinity=*/0,
  };
  const iree_hal_executable_target_selection_result_t selection_result =
      iree_hal_device_spec_select_executable_target(device_spec,
                                                    &target_selection);
  EXPECT_EQ(IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED,
            selection_result.outcome);
  ASSERT_NE(selection_result.target, nullptr);

  const iree_hal_device_spec_facet_t* facet =
      iree_hal_vulkan_device_spec_find_facet(device_spec);
  ASSERT_NE(facet, nullptr);
  iree_hal_vulkan_device_spec_t decoded = {};
  IREE_ASSERT_OK(iree_hal_vulkan_device_spec_decode_facet(facet, &decoded));
  EXPECT_EQ(decoded.api_version, VK_MAKE_API_VERSION(0, 1, 3, 0));
  EXPECT_EQ(decoded.enabled_features, device_plan.enabled_features);
  EXPECT_EQ(decoded.physical_device_type, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
  ASSERT_EQ(decoded.cooperative_matrix.count, 1);
  iree_hal_vulkan_cooperative_matrix_property_t decoded_property = {};
  ASSERT_TRUE(iree_hal_vulkan_device_spec_read_cooperative_matrix_property(
      &decoded, 0, &decoded_property));
  EXPECT_EQ(decoded_property.m_size, cooperative_matrix_property.MSize);
  EXPECT_EQ(decoded_property.a_type, cooperative_matrix_property.AType);
  EXPECT_EQ(decoded_property.scope, cooperative_matrix_property.scope);

  iree_hal_device_spec_release(device_spec);
  iree_hal_allocator_release(allocator);
}

}  // namespace
}  // namespace iree::hal::vulkan
