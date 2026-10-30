// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/device_spec.h"

#include <algorithm>
#include <cstring>
#include <tuple>
#include <vector>

#include "iree/hal/drivers/vulkan/atomic.h"
#include "iree/hal/drivers/vulkan/device_spec_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "vulkan/vulkan_core.h"

namespace iree::hal::vulkan {
namespace {

static iree_hal_vulkan_device_spec_t MakeTestSpec() {
  return {
      .api_version = VK_MAKE_API_VERSION(0, 1, 3, 0),
      .driver_version = 1234,
      .physical_device_type = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
      .enabled_features =
          {
              .general = IREE_HAL_VULKAN_FEATURE_REQUIRED_BASELINE |
                         IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16 |
                         IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL,
              .atomics =
                  IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT64_ADD,
          },
      .flags = IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_NONE,
      .subgroup_supported_operations =
          VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT,
      .maximum_workgroup_subgroup_count = 32,
  };
}

TEST(DeviceSpecTest, EncodesAndDecodesPayload) {
  iree_hal_vulkan_device_spec_t source = MakeTestSpec();
  iree_hal_vulkan_cooperative_matrix_property_t source_property = {
      .m_size = 16,
      .n_size = 8,
      .k_size = 16,
      .a_type = 1,
      .b_type = 2,
      .c_type = 3,
      .result_type = 4,
      .saturating_accumulation = 1,
      .scope = 5,
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
  EXPECT_EQ(decoded.enabled_features.general, source.enabled_features.general);
  EXPECT_EQ(decoded.enabled_features.atomics, source.enabled_features.atomics);
  EXPECT_EQ(decoded.flags, source.flags);
  EXPECT_EQ(decoded.subgroup_supported_operations,
            source.subgroup_supported_operations);
  EXPECT_EQ(decoded.maximum_workgroup_subgroup_count, 32u);
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

TEST(DeviceSpecTest, CanonicalPropertyOrderSupportsBinaryLookup) {
  for (size_t count : {0u, 1u, 2u, 3u, 64u, 1024u}) {
    SCOPED_TRACE(count);
    std::vector<iree_hal_vulkan_cooperative_matrix_property_t> properties;
    for (size_t i = 0; i < count; ++i) {
      // Include repeated keys with different saturation modes. Multiplication
      // permutes the rows so neither input order nor unique keys are assumed.
      const uint32_t ordinal = static_cast<uint32_t>((i * 37) % count);
      properties.push_back({
          .m_size = ordinal / 4 + 1,
          .n_size = 16,
          .k_size = 16,
          .a_type = VK_COMPONENT_TYPE_SINT8_KHR,
          .b_type = VK_COMPONENT_TYPE_SINT8_KHR,
          .c_type = VK_COMPONENT_TYPE_SINT32_KHR,
          .result_type = VK_COMPONENT_TYPE_SINT32_KHR,
          .saturating_accumulation = ordinal % 2,
          .scope = VK_SCOPE_SUBGROUP_KHR,
      });
    }
    iree_hal_vulkan_device_spec_t source = MakeTestSpec();
    iree_host_size_t payload_size = 0;
    IREE_ASSERT_OK(iree_hal_vulkan_device_spec_calculate_payload_size(
        properties.size(), &payload_size));
    std::vector<uint8_t> payload(payload_size);
    IREE_ASSERT_OK(iree_hal_vulkan_device_spec_encode(
        &source, properties.size(), properties.data(),
        iree_make_byte_span(payload.data(), payload.size())));
    iree_hal_vulkan_device_spec_t decoded = {};
    IREE_ASSERT_OK(iree_hal_vulkan_device_spec_decode(
        iree_make_const_byte_span(payload.data(), payload.size()), &decoded));
    for (const auto& property : properties) {
      EXPECT_TRUE(iree_hal_vulkan_device_spec_supports_cooperative_matrix(
          &decoded, &property));
    }
    const auto key = [](const auto& property) {
      return std::tie(property.m_size, property.n_size, property.k_size,
                      property.a_type, property.b_type, property.c_type,
                      property.result_type, property.scope,
                      property.saturating_accumulation);
    };
    std::sort(properties.begin(), properties.end(),
              [&](auto& lhs, auto& rhs) { return key(lhs) < key(rhs); });
    for (size_t i = 0; i < properties.size(); ++i) {
      iree_hal_vulkan_cooperative_matrix_property_t actual = {};
      ASSERT_TRUE(iree_hal_vulkan_device_spec_read_cooperative_matrix_property(
          &decoded, i, &actual));
      EXPECT_EQ(key(actual), key(properties[i]));
    }
    const iree_hal_vulkan_cooperative_matrix_property_t absent = {
        .m_size = static_cast<uint32_t>(count + 1),
    };
    EXPECT_FALSE(iree_hal_vulkan_device_spec_supports_cooperative_matrix(
        &decoded, &absent));

    if (count > 1) {
      // Keep the public envelope intact while breaking the ordering invariant.
      const size_t offset =
          decoded.cooperative_matrix.encoded_data.data - payload.data();
      const size_t row_size =
          decoded.cooperative_matrix.encoded_data.data_length / count;
      std::swap_ranges(payload.begin() + offset,
                       payload.begin() + offset + row_size,
                       payload.begin() + offset + (count - 1) * row_size);
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_INVALID_ARGUMENT,
          iree_hal_vulkan_device_spec_decode(
              iree_make_const_byte_span(payload.data(), payload.size()),
              &decoded));
      EXPECT_EQ(decoded.cooperative_matrix.count, 0u);
    }
  }
}

TEST(DeviceSpecTest, PropertyLookupChecksEveryRequirement) {
  const iree_hal_vulkan_device_spec_t source = MakeTestSpec();
  for (uint32_t saturation_mode : {0u, 1u}) {
    const iree_hal_vulkan_cooperative_matrix_property_t property = {
        .m_size = 16,
        .n_size = 8,
        .k_size = 16,
        .a_type = VK_COMPONENT_TYPE_SINT8_KHR,
        .b_type = VK_COMPONENT_TYPE_SINT8_KHR,
        .c_type = VK_COMPONENT_TYPE_SINT32_KHR,
        .result_type = VK_COMPONENT_TYPE_SINT32_KHR,
        .saturating_accumulation = saturation_mode,
        .scope = VK_SCOPE_SUBGROUP_KHR,
    };
    iree_host_size_t payload_size = 0;
    IREE_ASSERT_OK(
        iree_hal_vulkan_device_spec_calculate_payload_size(1, &payload_size));
    std::vector<uint8_t> payload(payload_size);
    IREE_ASSERT_OK(iree_hal_vulkan_device_spec_encode(
        &source, 1, &property,
        iree_make_byte_span(payload.data(), payload.size())));
    iree_hal_vulkan_device_spec_t decoded = {};
    IREE_ASSERT_OK(iree_hal_vulkan_device_spec_decode(
        iree_make_const_byte_span(payload.data(), payload.size()), &decoded));
    for (uint32_t requires_saturation : {0u, 1u}) {
      auto requirement = property;
      requirement.saturating_accumulation = requires_saturation;
      EXPECT_EQ(iree_hal_vulkan_device_spec_supports_cooperative_matrix(
                    &decoded, &requirement),
                requires_saturation == saturation_mode);
    }
    using Property = iree_hal_vulkan_cooperative_matrix_property_t;
    for (auto member : {&Property::m_size, &Property::n_size, &Property::k_size,
                        &Property::a_type, &Property::b_type, &Property::c_type,
                        &Property::result_type, &Property::scope}) {
      auto requirement = property;
      ++(requirement.*member);
      EXPECT_FALSE(iree_hal_vulkan_device_spec_supports_cooperative_matrix(
          &decoded, &requirement));
    }
  }
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
      .m_size = 16,
      .n_size = 16,
      .k_size = 8,
      .a_type = 1,
      .b_type = 1,
      .c_type = 2,
      .result_type = 2,
      .saturating_accumulation = 0,
      .scope = 3,
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
  EXPECT_EQ(decoded.enabled_features.general, source.enabled_features.general);
  EXPECT_EQ(decoded.enabled_features.atomics, source.enabled_features.atomics);
  EXPECT_EQ(decoded.physical_device_type, source.physical_device_type);
  EXPECT_EQ(decoded.subgroup_supported_operations,
            source.subgroup_supported_operations);
  ASSERT_EQ(decoded.cooperative_matrix.count, 1);

  iree_hal_device_spec_release(device_spec);
  iree_hal_device_spec_builder_deinitialize(&builder);
}

TEST(DeviceSpecTest, AtomicCapabilitiesRequireExactFeatures) {
  const iree_hal_vulkan_features_t required_features = {
      .general =
          IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES |
          IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL |
          IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE,
      .atomics = 0,
  };
  const iree_hal_vulkan_general_features_t required_feature_bits[] = {
      IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES,
      IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL,
      IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE,
  };
  for (iree_hal_vulkan_general_features_t removed_feature :
       required_feature_bits) {
    SCOPED_TRACE(removed_feature);
    iree_hal_vulkan_features_t remaining_features = required_features;
    remaining_features.general &= ~removed_feature;
    const iree_hal_atomic_capabilities_t capabilities =
        iree_hal_vulkan_atomic_capabilities(remaining_features);
    EXPECT_EQ(capabilities.operations.device_scope_32,
              IREE_HAL_ATOMIC_OPERATION_FLAG_NONE);
    EXPECT_EQ(capabilities.wait_conditions.device_scope_32,
              IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_NONE);
  }

  iree_hal_atomic_capabilities_t capabilities =
      iree_hal_vulkan_atomic_capabilities(required_features);
  EXPECT_EQ(capabilities.operations.device_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  EXPECT_EQ(capabilities.wait_conditions.device_scope_32,
            IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL);
  EXPECT_EQ(capabilities.operations.device_scope_64,
            IREE_HAL_ATOMIC_OPERATION_FLAG_NONE);

  iree_hal_vulkan_features_t int64_features = required_features;
  int64_features.atomics |= IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_INT64;
  capabilities = iree_hal_vulkan_atomic_capabilities(int64_features);
  EXPECT_EQ(capabilities.operations.device_scope_64,
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  EXPECT_EQ(capabilities.wait_conditions.device_scope_64,
            IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL);
  EXPECT_EQ(capabilities.operations.system_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAG_NONE);
  EXPECT_EQ(capabilities.operations.system_scope_64,
            IREE_HAL_ATOMIC_OPERATION_FLAG_NONE);
}

TEST(DeviceSpecTest, CommandTimestampsRequireComparableQueueClock) {
  iree_hal_allocator_t* allocator = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(iree_allocator_system(), &allocator));
  struct Case {
    // Native queue support for query capture and GPU resolve.
    VkQueueFlags queue_flags;
    // Number of implemented clock bits.
    uint32_t valid_bits;
    // Nanoseconds per native tick.
    float period;
    // Extension providing the cross-submission clock contract.
    bool calibrated;
    // Device-address support for binding-relative results.
    bool addresses;
    // Expected advertised command timestamp capabilities.
    iree_hal_queue_family_spec_flags_t expected;
  };
  constexpr auto kDirect = IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_COMMAND_TIMESTAMPS;
  constexpr auto kBoth =
      kDirect | IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_INDIRECT_COMMAND_TIMESTAMPS;
  const Case cases[] = {
      {VK_QUEUE_COMPUTE_BIT, 64, 1.0f, true, true, kBoth},
      {VK_QUEUE_COMPUTE_BIT, 32, 10.0f, true, true, kBoth},
      {VK_QUEUE_COMPUTE_BIT, 64, 1.0f, true, false, kDirect},
      {VK_QUEUE_COMPUTE_BIT, 64, 1.0f, false, true, 0},
      {VK_QUEUE_COMPUTE_BIT, 0, 1.0f, true, true, 0},
      {VK_QUEUE_COMPUTE_BIT, 64, 0.0f, true, true, 0},
      {VK_QUEUE_TRANSFER_BIT, 64, 1.0f, true, true, 0},
  };
  for (const auto& entry : cases) {
    SCOPED_TRACE(&entry - cases);
    iree_hal_vulkan_physical_device_snapshot_t physical_device = {};
    physical_device.properties2.properties.limits.timestampPeriod =
        entry.period;
    iree_hal_vulkan_queue_family_plan_t family = {
        .flags = entry.queue_flags,
        .timestamp_valid_bits = entry.valid_bits,
        .queue_count = 1,
    };
    uint32_t queue_index = 0;
    iree_hal_vulkan_device_plan_t plan = {
        .queue_inventory = {1, &family, 1, &queue_index},
    };
    if (entry.calibrated) {
      plan.enabled_extensions =
          IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_CALIBRATED_TIMESTAMPS;
    }
    if (entry.addresses) {
      plan.enabled_features.general =
          IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES;
    }
    const iree_hal_vulkan_device_spec_params_t params = {
        .logical_device_id = IREE_SV("vulkan://0"),
        .physical_device = &physical_device,
        .device_plan = &plan,
        .device_allocator = allocator,
    };
    iree_hal_device_spec_t* spec = nullptr;
    IREE_ASSERT_OK(iree_hal_vulkan_device_spec_create(
        &params, iree_allocator_system(), &spec));
    EXPECT_EQ(iree_hal_device_spec_queues(spec)->families[0].flags,
              entry.expected);
    iree_hal_device_spec_release(spec);
  }
  iree_hal_allocator_release(allocator);
}

TEST(DeviceSpecTest, SparseGeometryUsesAddressLimitsBeyondDescriptorRanges) {
  iree_hal_allocator_t* allocator = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(iree_allocator_system(), &allocator));
  iree_hal_vulkan_physical_device_snapshot_t physical_device = {};
  physical_device.properties2.properties.limits.maxStorageBufferRange = 1u
                                                                        << 27;
  physical_device.properties2.properties.limits.sparseAddressSpaceSize =
      UINT64_C(1) << 40;
  physical_device.maintenance4_properties.maxBufferSize = UINT64_C(1) << 38;
  physical_device.properties11.maxMemoryAllocationSize = UINT64_C(1) << 34;
  const iree_hal_vulkan_device_plan_t device_plan = {};
  iree_hal_vulkan_device_spec_params_t params = {
      .logical_device_id = IREE_SV("vulkan://0"),
      .physical_device = &physical_device,
      .device_plan = &device_plan,
      .device_allocator = allocator,
  };
  for (const VkDeviceSize alignment : {VkDeviceSize{0}, VkDeviceSize{65536}}) {
    params.sparse_buffer_requirements.alignment = alignment;
    iree_hal_device_spec_t* spec = nullptr;
    IREE_ASSERT_OK(iree_hal_vulkan_device_spec_create(
        &params, iree_allocator_system(), &spec));
    const auto* memory = iree_hal_device_spec_virtual_memory(spec);
    ASSERT_NE(memory, nullptr);
    if (!alignment) {
      EXPECT_EQ(0u, memory->class_count);
    } else {
      ASSERT_EQ(1u, memory->class_count);
      const auto& memory_class = memory->classes[0];
      EXPECT_EQ(alignment, memory_class.minimum_page_size);
      EXPECT_EQ(alignment, memory_class.recommended_page_size);
      EXPECT_EQ(UINT64_C(1) << 38, memory_class.maximum_reservation_size);
      EXPECT_EQ(UINT64_C(1) << 34,
                memory_class.maximum_physical_allocation_size);
      EXPECT_FALSE(
          iree_any_bit_set(memory_class.operation_flags,
                           IREE_HAL_VIRTUAL_MEMORY_OPERATION_FLAG_ADVISE));
    }
    iree_hal_device_spec_release(spec);
  }
  iree_hal_allocator_release(allocator);
}

struct FloatWidthSupport {
  // Whether the control is supported for F16.
  VkBool32 f16;
  // Whether the control is supported for F32.
  VkBool32 f32;
  // Whether the control is supported for F64.
  VkBool32 f64;
};

struct FloatControlsCase {
  // Per-width denormal-preservation support.
  FloatWidthSupport denorm_preserve;
  // Widths whose denormal modes can be controlled independently.
  VkShaderFloatControlsIndependence denorm_independence;
  // Per-width round-to-nearest-even support.
  FloatWidthSupport rounding_rte;
  // Widths whose rounding modes can be controlled independently.
  VkShaderFloatControlsIndependence rounding_independence;
  // Per-width signed-zero, infinity, and NaN preservation support.
  FloatWidthSupport signed_zero_inf_nan_preserve;
  // Float-control promises retained by the device facet.
  iree_hal_vulkan_device_spec_flags_t expected_flags;
};

class DeviceSpecFloatControlsTest
    : public ::testing::TestWithParam<FloatControlsCase> {};

TEST_P(DeviceSpecFloatControlsTest, CreatesSpecFromParams) {
  iree_hal_allocator_t* allocator = NULL;
  IREE_ASSERT_OK(
      iree_hal_allocator_create_heap(iree_allocator_system(), &allocator));

  iree_hal_vulkan_physical_device_snapshot_t physical_device = {
      .ordinal = 3,
  };
  physical_device.float_controls_properties.shaderDenormPreserveFloat16 =
      GetParam().denorm_preserve.f16;
  physical_device.float_controls_properties.shaderDenormPreserveFloat32 =
      GetParam().denorm_preserve.f32;
  physical_device.float_controls_properties.shaderDenormPreserveFloat64 =
      GetParam().denorm_preserve.f64;
  physical_device.float_controls_properties.denormBehaviorIndependence =
      GetParam().denorm_independence;
  physical_device.float_controls_properties.shaderRoundingModeRTEFloat16 =
      GetParam().rounding_rte.f16;
  physical_device.float_controls_properties.shaderRoundingModeRTEFloat32 =
      GetParam().rounding_rte.f32;
  physical_device.float_controls_properties.shaderRoundingModeRTEFloat64 =
      GetParam().rounding_rte.f64;
  physical_device.float_controls_properties.roundingModeIndependence =
      GetParam().rounding_independence;
  physical_device.float_controls_properties
      .shaderSignedZeroInfNanPreserveFloat16 =
      GetParam().signed_zero_inf_nan_preserve.f16;
  physical_device.float_controls_properties
      .shaderSignedZeroInfNanPreserveFloat32 =
      GetParam().signed_zero_inf_nan_preserve.f32;
  physical_device.float_controls_properties
      .shaderSignedZeroInfNanPreserveFloat64 =
      GetParam().signed_zero_inf_nan_preserve.f64;
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
  physical_device.subgroup_properties.supportedOperations =
      VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT;
  physical_device.subgroup_size_control_properties.minSubgroupSize = 32;
  physical_device.subgroup_size_control_properties.maxSubgroupSize = 64;
  physical_device.subgroup_size_control_properties
      .maxComputeWorkgroupSubgroups = 16;
  physical_device.calibrated_timestamp_time_domains =
      IREE_HAL_VULKAN_TIME_DOMAIN_DEVICE |
      IREE_HAL_VULKAN_TIME_DOMAIN_CLOCK_MONOTONIC;
  VkCooperativeMatrixPropertiesKHR cooperative_matrix_property = {
      .sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
      .MSize = 16,
      .NSize = 16,
      .KSize = 16,
      .AType = VK_COMPONENT_TYPE_FLOAT16_KHR,
      .BType = VK_COMPONENT_TYPE_FLOAT16_KHR,
      .CType = VK_COMPONENT_TYPE_FLOAT32_KHR,
      .ResultType = VK_COMPONENT_TYPE_FLOAT32_KHR,
      .scope = VK_SCOPE_SUBGROUP_KHR,
  };
  physical_device.cooperative_matrix_property_count = 1;
  physical_device.cooperative_matrix_property_rows =
      &cooperative_matrix_property;

  iree_hal_vulkan_device_plan_t device_plan = {
      .request_flags = IREE_HAL_VULKAN_REQUEST_FLAG_TRACING,
  };
  device_plan.enabled_features.general =
      IREE_HAL_VULKAN_FEATURE_REQUIRED_BASELINE |
      IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES |
      IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL |
      IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE |
      IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL;
  device_plan.enabled_features.atomics =
      IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_INT64;
  device_plan.enabled_extensions =
      IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_CALIBRATED_TIMESTAMPS;
  uint32_t queue_indices[] = {0, 1, 0};
  iree_hal_vulkan_queue_family_plan_t queue_families[] = {
      {
          .native_family_index = 2,
          .flags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
          .timestamp_valid_bits = 64,
          .queue_count = 2,
          .queue_offset = 0,
      },
      {
          .native_family_index = 5,
          .flags = VK_QUEUE_TRANSFER_BIT,
          .timestamp_valid_bits = 48,
          .queue_count = 1,
          .queue_offset = 2,
      },
  };
  device_plan.queue_inventory = {
      .family_count = IREE_ARRAYSIZE(queue_families),
      .families = queue_families,
      .queue_count = IREE_ARRAYSIZE(queue_indices),
      .queue_indices = queue_indices,
  };

  iree_hal_vulkan_device_spec_params_t params = {
      .logical_device_id = IREE_SV("vulkan://0"),
      .display_name = IREE_SV("Vulkan test logical device"),
      .physical_device = &physical_device,
      .device_plan = &device_plan,
      .device_allocator = allocator,
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
  ASSERT_EQ(queues->family_count, 2);
  EXPECT_TRUE(iree_string_view_equal(queues->families[0].name,
                                     IREE_SV("queue-family-2")));
  EXPECT_EQ(queues->families[0].provisioned_queue_count, 2);
  EXPECT_EQ(queues->families[0].timestamp_valid_bits, 64u);
  EXPECT_EQ(queues->families[0].timestamp_frequency_hz, 1000000000ull);
  EXPECT_EQ(queues->families[0].role_flags,
            IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH |
                IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER |
                IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_HOST_CALL |
                IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_PROFILING |
                IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_ATOMIC);
  EXPECT_EQ(queues->families[0].atomic_capabilities.operations.device_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  EXPECT_EQ(queues->families[0].atomic_capabilities.operations.device_scope_64,
            IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  EXPECT_EQ(
      queues->families[0].atomic_capabilities.wait_conditions.device_scope_32,
      IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL);
  EXPECT_EQ(queues->families[0]
                .zero_compute_atomic_capabilities.operations.device_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAG_NONE);

  EXPECT_TRUE(iree_string_view_equal(queues->families[1].name,
                                     IREE_SV("queue-family-5")));
  EXPECT_EQ(queues->families[1].provisioned_queue_count, 1);
  EXPECT_EQ(queues->families[1].timestamp_valid_bits, 48u);
  EXPECT_EQ(queues->families[1].role_flags,
            IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER |
                IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_HOST_CALL |
                IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_PROFILING);
  EXPECT_EQ(queues->families[1].atomic_capabilities.operations.device_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAG_NONE);

  const iree_hal_device_memory_spec_t* memory =
      iree_hal_device_spec_memory(device_spec);
  ASSERT_NE(memory, nullptr);
  const iree_hal_atomic_operation_capabilities_t expected_memory_operations =
      iree_hal_atomic_operation_capabilities_for_host(
          IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  bool found_atomic_memory_type = false;
  for (iree_host_size_t i = 0; i < memory->memory_type_count; ++i) {
    const iree_hal_memory_type_spec_t* memory_type = &memory->memory_types[i];
    if (!iree_all_bits_set(memory_type->memory_type,
                           IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE) ||
        !iree_all_bits_set(memory_type->allowed_buffer_usage,
                           IREE_HAL_BUFFER_USAGE_STORAGE)) {
      continue;
    }
    found_atomic_memory_type = true;
    EXPECT_EQ(memory_type->atomic_operations.device_scope_32,
              expected_memory_operations.device_scope_32);
    EXPECT_EQ(memory_type->atomic_operations.device_scope_64,
              expected_memory_operations.device_scope_64);
    EXPECT_EQ(memory_type->atomic_operations.system_scope_32,
              expected_memory_operations.system_scope_32);
    EXPECT_EQ(memory_type->atomic_operations.system_scope_64,
              expected_memory_operations.system_scope_64);
  }
  EXPECT_TRUE(found_atomic_memory_type);

  const iree_hal_device_dispatch_spec_t* dispatch =
      iree_hal_device_spec_dispatch(device_spec);
  ASSERT_NE(dispatch, nullptr);
  EXPECT_EQ(dispatch->launch.maximum_workgroup_invocations, 1024);
  EXPECT_EQ(dispatch->launch.maximum_workgroup_size[2], 64);
  EXPECT_EQ(dispatch->subgroup.default_size, 32);
  EXPECT_EQ(dispatch->subgroup.minimum_size, 32);
  EXPECT_EQ(dispatch->subgroup.maximum_size, 32);
  EXPECT_EQ(dispatch->subgroup.supported_size_mask, 1ull << 32);
  EXPECT_EQ(dispatch->execution.maximum_workgroup_local_memory_size, 64 * 1024);
  EXPECT_EQ(dispatch->addressing.minimum_buffer_device_address_alignment, 16);

  const iree_hal_device_timing_spec_t* timing =
      iree_hal_device_spec_timing(device_spec);
  ASSERT_NE(timing, nullptr);
  EXPECT_EQ(timing->timestamp_valid_bits, 48);
  EXPECT_TRUE(iree_all_bits_set(
      timing->flags, IREE_HAL_DEVICE_TIMING_SPEC_FLAG_DEVICE_TIMESTAMPS |
                         IREE_HAL_DEVICE_TIMING_SPEC_FLAG_HOST_CORRELATION |
                         IREE_HAL_DEVICE_TIMING_SPEC_FLAG_TRACE_CAPTURE));

  const iree_hal_device_executable_spec_t* executables =
      iree_hal_device_spec_executables(device_spec);
  ASSERT_NE(executables, nullptr);
  iree_hal_executable_target_selection_t target_selection = {
      .family = IREE_SV("spirv"),
      .target_key = IREE_SV("vulkan1.3+bda"),
      .kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_GENERIC,
      .physical_device_affinity = 0,
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
  EXPECT_EQ(decoded.flags, GetParam().expected_flags);
  EXPECT_EQ(decoded.enabled_features.general,
            device_plan.enabled_features.general);
  EXPECT_EQ(decoded.enabled_features.atomics,
            device_plan.enabled_features.atomics);
  EXPECT_EQ(decoded.physical_device_type, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
  EXPECT_EQ(decoded.subgroup_supported_operations,
            physical_device.subgroup_properties.supportedOperations);
  EXPECT_EQ(decoded.maximum_workgroup_subgroup_count, 16u);
  ASSERT_EQ(decoded.cooperative_matrix.count, 1);
  iree_hal_vulkan_cooperative_matrix_property_t decoded_property = {};
  ASSERT_TRUE(iree_hal_vulkan_device_spec_read_cooperative_matrix_property(
      &decoded, 0, &decoded_property));
  EXPECT_EQ(decoded_property.m_size, cooperative_matrix_property.MSize);
  EXPECT_EQ(decoded_property.a_type, cooperative_matrix_property.AType);
  EXPECT_EQ(decoded_property.scope, cooperative_matrix_property.scope);

  iree_hal_device_spec_release(device_spec);
  physical_device.subgroup_size_control_properties.requiredSubgroupSizeStages =
      VK_SHADER_STAGE_COMPUTE_BIT;
  IREE_ASSERT_OK(iree_hal_vulkan_device_spec_create(
      &params, iree_allocator_system(), &device_spec));
  dispatch = iree_hal_device_spec_dispatch(device_spec);
  ASSERT_NE(dispatch, nullptr);
  EXPECT_EQ(dispatch->subgroup.minimum_size, 32);
  EXPECT_EQ(dispatch->subgroup.maximum_size, 64);
  iree_hal_device_spec_release(device_spec);

  device_plan.enabled_features.general &=
      ~IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL;
  IREE_ASSERT_OK(iree_hal_vulkan_device_spec_create(
      &params, iree_allocator_system(), &device_spec));
  dispatch = iree_hal_device_spec_dispatch(device_spec);
  ASSERT_NE(dispatch, nullptr);
  EXPECT_EQ(dispatch->subgroup.minimum_size, 32);
  EXPECT_EQ(dispatch->subgroup.maximum_size, 32);
  iree_hal_device_spec_release(device_spec);
  iree_hal_allocator_release(allocator);
}

INSTANTIATE_TEST_SUITE_P(
    FloatControls, DeviceSpecFloatControlsTest,
    ::testing::Values(
        FloatControlsCase{{VK_FALSE, VK_FALSE, VK_FALSE},
                          VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
                          {VK_FALSE, VK_FALSE, VK_FALSE},
                          VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
                          {VK_FALSE, VK_FALSE, VK_FALSE},
                          IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_NONE},
        FloatControlsCase{{VK_TRUE, VK_TRUE, VK_TRUE},
                          VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE,
                          {VK_TRUE, VK_TRUE, VK_TRUE},
                          VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE,
                          {VK_FALSE, VK_FALSE, VK_FALSE},
                          IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_NONE},
        FloatControlsCase{
            {VK_FALSE, VK_TRUE, VK_FALSE},
            VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY,
            {VK_FALSE, VK_TRUE, VK_FALSE},
            VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY,
            {VK_FALSE, VK_TRUE, VK_FALSE},
            IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT32_DENORM_PRESERVE |
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT32_ROUNDING_RTE |
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT32_SIGNED_ZERO_INF_NAN_PRESERVE},
        FloatControlsCase{
            {VK_TRUE, VK_FALSE, VK_FALSE},
            VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY,
            {VK_TRUE, VK_FALSE, VK_FALSE},
            VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY,
            {VK_TRUE, VK_FALSE, VK_FALSE},
            IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT16_SIGNED_ZERO_INF_NAN_PRESERVE},
        FloatControlsCase{
            {VK_TRUE, VK_FALSE, VK_TRUE},
            VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY,
            {VK_TRUE, VK_FALSE, VK_TRUE},
            VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY,
            {VK_TRUE, VK_FALSE, VK_TRUE},
            IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT16_64_DENORM_PRESERVE |
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT16_64_ROUNDING_RTE |
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT16_SIGNED_ZERO_INF_NAN_PRESERVE |
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT64_SIGNED_ZERO_INF_NAN_PRESERVE},
        FloatControlsCase{
            {VK_TRUE, VK_TRUE, VK_TRUE},
            VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY,
            {VK_TRUE, VK_TRUE, VK_TRUE},
            VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE,
            {VK_TRUE, VK_TRUE, VK_TRUE},
            IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT32_DENORM_PRESERVE |
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT16_64_DENORM_PRESERVE |
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT16_SIGNED_ZERO_INF_NAN_PRESERVE |
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT32_SIGNED_ZERO_INF_NAN_PRESERVE |
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT64_SIGNED_ZERO_INF_NAN_PRESERVE},
        FloatControlsCase{
            {VK_TRUE, VK_TRUE, VK_TRUE},
            VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE,
            {VK_TRUE, VK_TRUE, VK_TRUE},
            VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY,
            {VK_FALSE, VK_FALSE, VK_FALSE},
            IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT16_64_ROUNDING_RTE |
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT32_ROUNDING_RTE},
        FloatControlsCase{
            {VK_TRUE, VK_TRUE, VK_TRUE},
            VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY,
            {VK_TRUE, VK_TRUE, VK_TRUE},
            VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY,
            {VK_FALSE, VK_FALSE, VK_FALSE},
            IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT32_DENORM_PRESERVE |
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT16_64_DENORM_PRESERVE |
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT16_64_ROUNDING_RTE |
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT32_ROUNDING_RTE},
        FloatControlsCase{
            {VK_TRUE, VK_TRUE, VK_TRUE},
            VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
            {VK_TRUE, VK_TRUE, VK_TRUE},
            VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
            {VK_TRUE, VK_TRUE, VK_TRUE},
            IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT32_DENORM_PRESERVE |
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT16_64_DENORM_PRESERVE |
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT16_64_ROUNDING_RTE |
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT32_ROUNDING_RTE |
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT16_SIGNED_ZERO_INF_NAN_PRESERVE |
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT32_SIGNED_ZERO_INF_NAN_PRESERVE |
                IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT64_SIGNED_ZERO_INF_NAN_PRESERVE}));

}  // namespace
}  // namespace iree::hal::vulkan
