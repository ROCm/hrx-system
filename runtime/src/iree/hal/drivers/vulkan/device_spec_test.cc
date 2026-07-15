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
  std::vector<uint8_t> payload_storage(
      iree_hal_vulkan_device_spec_payload_size());
  IREE_ASSERT_OK(iree_hal_vulkan_device_spec_encode(
      &source,
      iree_make_byte_span(payload_storage.data(), payload_storage.size())));
  static const uint8_t kExpectedPayload[] = {
      'V', 'V', 'D', 'S', 2, 0, 0,    0,    0x00, 0x30, 0x40, 0x00, 0xd2, 0x04,
      0,   0,   2,   0,   0, 0, 0xc0, 0x27, 0,    0,    0,    0,    0,    0,
  };
  ASSERT_EQ(payload_storage.size(), sizeof(kExpectedPayload));
  EXPECT_EQ(memcmp(payload_storage.data(), kExpectedPayload,
                   sizeof(kExpectedPayload)),
            0);

  iree_hal_vulkan_device_spec_t decoded = {0};
  IREE_ASSERT_OK(iree_hal_vulkan_device_spec_decode(
      iree_make_const_byte_span(payload_storage.data(), payload_storage.size()),
      &decoded));
  EXPECT_EQ(decoded.api_version, VK_MAKE_API_VERSION(0, 1, 3, 0));
  EXPECT_EQ(decoded.driver_version, 1234u);
  EXPECT_EQ(decoded.physical_device_type, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
  EXPECT_EQ(decoded.enabled_features, source.enabled_features);
  EXPECT_EQ(decoded.flags, source.flags);
}

TEST(DeviceSpecTest, RejectsMalformedPayloads) {
  iree_hal_vulkan_device_spec_t source = MakeTestSpec();
  std::vector<uint8_t> canonical_payload(
      iree_hal_vulkan_device_spec_payload_size());
  IREE_ASSERT_OK(iree_hal_vulkan_device_spec_encode(
      &source,
      iree_make_byte_span(canonical_payload.data(), canonical_payload.size())));

  iree_hal_vulkan_device_spec_t decoded = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_vulkan_device_spec_encode(
          &source, iree_make_byte_span(canonical_payload.data(),
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
          &source, iree_make_byte_span(
                       NULL, iree_hal_vulkan_device_spec_payload_size())));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_vulkan_device_spec_decode(
          iree_make_const_byte_span(NULL,
                                    iree_hal_vulkan_device_spec_payload_size()),
          &decoded));
}

TEST(DeviceSpecTest, AddsAndFindsCoreFacet) {
  iree_hal_vulkan_device_spec_t source = MakeTestSpec();

  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      iree_hal_vulkan_device_spec_builder_add_facet(&builder, &source));

  iree_hal_device_spec_t* device_spec = NULL;
  IREE_ASSERT_OK(iree_hal_device_spec_builder_finalize(&builder, &device_spec));
  const iree_hal_device_spec_facet_t* facet =
      iree_hal_vulkan_device_spec_find_facet(device_spec);
  ASSERT_NE(facet, nullptr);

  iree_hal_vulkan_device_spec_t decoded = {0};
  IREE_ASSERT_OK(iree_hal_vulkan_device_spec_decode_facet(facet, &decoded));
  EXPECT_EQ(decoded.enabled_features, source.enabled_features);
  EXPECT_EQ(decoded.physical_device_type, source.physical_device_type);

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

  iree_hal_vulkan_device_plan_t device_plan = {};
  device_plan.request_flags = IREE_HAL_VULKAN_REQUEST_FLAG_TRACING;
  device_plan.enabled_features =
      IREE_HAL_VULKAN_FEATURE_REQUIRED_BASELINE |
      IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL;
  device_plan.enabled_extensions =
      IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_CALIBRATED_TIMESTAMPS;
  device_plan.queue_assignment.queue_count = 2;
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
  ASSERT_EQ(executables->format_count, 1);
  EXPECT_TRUE(iree_string_view_equal(executables->formats[0].format,
                                     IREE_SV("vulkan-spirv-bda")));
  iree_hal_executable_target_selection_t target_selection = {
      /*.family=*/IREE_SV("spirv"),
      /*.target_key=*/IREE_SV("vulkan1.3+bda"),
      /*.kind_flags=*/IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
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

  iree_hal_device_spec_release(device_spec);
  iree_hal_allocator_release(allocator);
}

}  // namespace
}  // namespace iree::hal::vulkan
