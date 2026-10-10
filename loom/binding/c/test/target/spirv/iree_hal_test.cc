// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/iree_hal.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/device_spec.h"
#include "iree/hal/utils/device_spec_builder.h"
#include "iree/testing/gtest.h"
#include "loomc/diagnostic.h"
#include "loomc/result.h"
#include "loomc/target.h"
#include "loomc/target/spirv/base.h"
#include "loomc/target/spirv/iree_hal.h"
#include "loomc/target/spirv/profile.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using DeviceSpecPtr =
    HandlePtr<iree_hal_device_spec_t, iree_hal_device_spec_release>;

constexpr uint32_t kVulkanApiVersion13 =
    (1u << 22) | (3u << 12) | static_cast<uint32_t>(0);
constexpr uint32_t kVulkanComponentTypeBfloat16Khr = 1000141000;
constexpr uint32_t kVulkanSubgroupFeatureBallotBit = 0x00000008;
constexpr uint64_t kMaximumWorkgroupLocalMemorySize = 32 * 1024;

constexpr const char kF16CooperativeMatrixRow[] =
    "khr.cooperative_matrix.f16.16x16x16.f32.subgroup";
constexpr const char kBf16CooperativeMatrixRow[] =
    "khr.cooperative_matrix.bf16.16x16x16.f32.subgroup";
constexpr const char kS8CooperativeMatrixRow[] =
    "khr.cooperative_matrix.s8.16x16x32.s32.subgroup.signed_saturating";
constexpr const char kU8CooperativeMatrixRow[] =
    "khr.cooperative_matrix.u8.16x16x32.u32.subgroup";

typedef uint32_t DeviceSpecFlags;
typedef enum DeviceSpecFlagBits {
  kDeviceSpecFlagNone = 0u,
  kDeviceSpecFlagIncludeDispatch = 1u << 0,
  kDeviceSpecFlagIncludeExecutableTarget = 1u << 1,
  kDeviceSpecFlagIncludeWorkgroupStorageLimit = 1u << 2,
} DeviceSpecFlagBits;

constexpr DeviceSpecFlags kCompleteDeviceSpecFlags =
    kDeviceSpecFlagIncludeDispatch | kDeviceSpecFlagIncludeExecutableTarget |
    kDeviceSpecFlagIncludeWorkgroupStorageLimit;

typedef struct FakeHalDevice {
  // HAL resource header used by device vtable dispatch.
  iree_hal_resource_t resource;

  // Immutable device facts borrowed from the test.
  const iree_hal_device_spec_t* device_spec;
} FakeHalDevice;

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

static const iree_hal_device_spec_t* FakeHalDeviceSpec(
    iree_hal_device_t* base_device) {
  FakeHalDevice* device = reinterpret_cast<FakeHalDevice*>(base_device);
  return device->device_spec;
}

static iree_hal_device_vtable_t MakeFakeHalDeviceVtable() {
  iree_hal_device_vtable_t vtable = {};
  vtable.device_spec = FakeHalDeviceSpec;
  return vtable;
}

static const iree_hal_device_vtable_t kFakeHalDeviceVtable =
    MakeFakeHalDeviceVtable();

void InitializeFakeDevice(const iree_hal_device_spec_t* device_spec,
                          FakeHalDevice* out_device) {
  out_device->device_spec = device_spec;
  iree_hal_resource_initialize(&kFakeHalDeviceVtable, &out_device->resource);
}

iree_hal_vulkan_features_t RequiredVulkanFeatures() {
  return {
      .general = IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES |
                 IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT64,
      .atomics = 0,
  };
}

iree_status_t CreateVulkanDeviceSpec(
    iree_hal_vulkan_features_t enabled_features, DeviceSpecFlags flags,
    DeviceSpecPtr* out_device_spec,
    iree_host_size_t cooperative_matrix_property_count = 0,
    const iree_hal_vulkan_cooperative_matrix_property_t*
        cooperative_matrix_properties = nullptr,
    iree_hal_vulkan_device_spec_flags_t device_spec_flags =
        IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_NONE,
    uint32_t subgroup_supported_operations = 0) {
  out_device_spec->reset();
  iree_hal_vulkan_device_spec_t vulkan_spec = {
      .api_version = kVulkanApiVersion13,
      .driver_version = 1,
      .physical_device_type = 2,
      .enabled_features = enabled_features,
      .flags = device_spec_flags,
      .subgroup_supported_operations = subgroup_supported_operations,
  };
  iree_host_size_t vulkan_payload_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_spec_calculate_payload_size(
      cooperative_matrix_property_count, &vulkan_payload_size));
  std::vector<uint8_t> vulkan_payload_storage(vulkan_payload_size);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_spec_encode(
      &vulkan_spec, cooperative_matrix_property_count,
      cooperative_matrix_properties,
      iree_make_byte_span(vulkan_payload_storage.data(),
                          vulkan_payload_storage.size())));
  iree_hal_device_spec_facet_t vulkan_facet = {
      .schema_id =
          iree_make_cstring_view(IREE_HAL_VULKAN_DEVICE_SPEC_SCHEMA_ID),
      .schema_version = IREE_HAL_VULKAN_DEVICE_SPEC_SCHEMA_VERSION,
      .payload = iree_make_const_byte_span(vulkan_payload_storage.data(),
                                           vulkan_payload_storage.size()),
  };

  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status = iree_ok_status();
  if (iree_any_bit_set(flags, kDeviceSpecFlagIncludeDispatch)) {
    iree_hal_device_dispatch_spec_t dispatch = {
        .launch =
            {
                .maximum_workgroup_invocations = 256,
                .maximum_workgroup_size = {256, 128, 64},
                .maximum_workgroup_count = {65535, 65535, 65535},
            },
        .subgroup =
            {
                .default_size = 32,
                .minimum_size = 32,
                .maximum_size = 32,
                .supported_size_mask = 1ull << 32,
            },
        .execution =
            {
                .unit_count = 1,
                .group_count = 1,
                .maximum_resident_workgroup_count = 0,
                .maximum_resident_invocation_count = 0,
                .maximum_resident_subgroup_count = 0,
                .maximum_register_count = 0,
                .maximum_workgroup_register_count = 0,
                .maximum_local_memory_size = 0,
                .maximum_workgroup_local_memory_size =
                    iree_any_bit_set(
                        flags, kDeviceSpecFlagIncludeWorkgroupStorageLimit)
                        ? kMaximumWorkgroupLocalMemorySize
                        : 0,
                .maximum_workgroup_local_memory_size_optin = 0,
            },
        .addressing =
            {
                .pointer_size_bits = 64,
                .address_space_bits = 64,
            },
        .flags = IREE_HAL_DEVICE_DISPATCH_SPEC_FLAG_NONE,
    };
    status = iree_hal_device_spec_builder_set_dispatch(&builder, &dispatch);
  }
  const iree_hal_executable_target_t executable_target = {
      .family = IREE_SV("spirv"),
      .target_key = IREE_SV("vulkan1.3+bda"),
      .kind = IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
      .priority = 100,
      .physical_device_affinity = 1,
      .flags = IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
  };
  const bool include_target =
      iree_any_bit_set(flags, kDeviceSpecFlagIncludeExecutableTarget);
  const iree_hal_device_executable_spec_t executables = {
      .target_count = include_target ? 1u : 0u,
      .targets = include_target ? &executable_target : nullptr,
      .flags = IREE_HAL_DEVICE_EXECUTABLE_SPEC_FLAG_NONE,
  };
  if (iree_status_is_ok(status) && include_target) {
    status =
        iree_hal_device_spec_builder_set_executables(&builder, &executables);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_add_facet(&builder, &vulkan_facet);
  }
  iree_hal_device_spec_t* device_spec = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_finalize(&builder, &device_spec);
  }
  iree_hal_device_spec_builder_deinitialize(&builder);
  if (iree_status_is_ok(status)) {
    out_device_spec->reset(device_spec);
  }
  return status;
}

TargetEnvironmentPtr CreateSpirvTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_spirv(
      loomc_allocator_system(), &target_environment);
  LOOMC_EXPECT_OK(status);
  return TargetEnvironmentPtr(target_environment);
}

iree_hal_vulkan_features_t CooperativeMatrixVulkanFeatures() {
  iree_hal_vulkan_features_t features = RequiredVulkanFeatures();
  features.general |=
      IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16 |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT8 |
      IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_8BIT_ACCESS |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_TYPE |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_COOPERATIVE_MATRIX;
  return features;
}

iree_hal_vulkan_cooperative_matrix_property_t F16MatrixProperty() {
  return {
      .m_size = 16,
      .n_size = 16,
      .k_size = 16,
      .a_type = LOOMC_SPIRV_COMPONENT_TYPE_FLOAT16_NV,
      .b_type = LOOMC_SPIRV_COMPONENT_TYPE_FLOAT16_NV,
      .c_type = LOOMC_SPIRV_COMPONENT_TYPE_FLOAT32_NV,
      .result_type = LOOMC_SPIRV_COMPONENT_TYPE_FLOAT32_NV,
      .saturating_accumulation = 0,
      .scope = LOOMC_SPIRV_SCOPE_SUBGROUP,
  };
}

iree_hal_vulkan_cooperative_matrix_property_t Bf16MatrixProperty() {
  return {
      .m_size = 16,
      .n_size = 16,
      .k_size = 16,
      .a_type = kVulkanComponentTypeBfloat16Khr,
      .b_type = kVulkanComponentTypeBfloat16Khr,
      .c_type = LOOMC_SPIRV_COMPONENT_TYPE_FLOAT32_NV,
      .result_type = LOOMC_SPIRV_COMPONENT_TYPE_FLOAT32_NV,
      .saturating_accumulation = 0,
      .scope = LOOMC_SPIRV_SCOPE_SUBGROUP,
  };
}

iree_hal_vulkan_cooperative_matrix_property_t S8MatrixProperty(
    uint32_t saturating_accumulation) {
  return {
      .m_size = 16,
      .n_size = 16,
      .k_size = 32,
      .a_type = LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT8_NV,
      .b_type = LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT8_NV,
      .c_type = LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV,
      .result_type = LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV,
      .saturating_accumulation = saturating_accumulation,
      .scope = LOOMC_SPIRV_SCOPE_SUBGROUP,
  };
}

iree_hal_vulkan_cooperative_matrix_property_t U8MatrixProperty() {
  return {
      .m_size = 16,
      .n_size = 16,
      .k_size = 32,
      .a_type = LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_NV,
      .b_type = LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT8_NV,
      .c_type = LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT32_NV,
      .result_type = LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT32_NV,
      .saturating_accumulation = 0,
      .scope = LOOMC_SPIRV_SCOPE_SUBGROUP,
  };
}

void ExpectSucceededResult(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  if (!loomc_result_succeeded(result) &&
      loomc_result_diagnostic_count(result) != 0) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, 0);
    ASSERT_NE(diagnostic, nullptr);
    ADD_FAILURE() << ToString(diagnostic->message);
  }
  EXPECT_TRUE(loomc_result_succeeded(result));
}

void ExpectFailedSpirvIreeHalResult(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result));
  ASSERT_GE(loomc_result_diagnostic_count(result), 1u);
  const loomc_diagnostic_t* diagnostic = loomc_result_diagnostic_at(result, 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(diagnostic->severity, LOOMC_DIAGNOSTIC_SEVERITY_ERROR);
  EXPECT_EQ(ToString(diagnostic->code), "SPIRV/IREE_HAL");
}

bool FindCooperativeMatrixRow(const loomc_target_profile_t* profile,
                              const char* name,
                              loomc_spirv_cooperative_matrix_row_t* out_row) {
  loomc_spirv_profile_info_t info = {};
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_info(profile, &info));
  for (loomc_host_size_t i = 0; i < info.cooperative_matrix_row_count; ++i) {
    loomc_spirv_cooperative_matrix_row_t row = {};
    LOOMC_EXPECT_OK(
        loomc_spirv_target_profile_cooperative_matrix_row_at(profile, i, &row));
    if (ToString(row.name) == name) {
      *out_row = row;
      return true;
    }
  }
  return false;
}

TargetProfilePtr SelectTargetFromHal(
    loomc_target_environment_t* target_environment, FakeHalDevice* device,
    loomc_result_t** out_result,
    const iree_hal_executable_target_t** out_executable_target = nullptr) {
  iree_hal_device_t* hal_device = reinterpret_cast<iree_hal_device_t*>(device);
  loomc_spirv_iree_hal_target_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SPIRV_IREE_HAL_TARGET_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .identifier = loomc_make_cstring_view("fake-vulkan"),
      .device = hal_device,
      .physical_device_affinity = 0,
      .target_profile = nullptr,
  };
  loomc_iree_hal_target_selection_t selection = {};
  loomc_status_t status = loomc_target_select_spirv_iree_hal(
      target_environment, &options, loomc_allocator_system(), &selection,
      out_result);
  LOOMC_EXPECT_OK(status);
  if (out_executable_target != nullptr) {
    *out_executable_target = selection.executable_target;
  }
  return TargetProfilePtr(selection.target_profile);
}

TEST(LoomcSpirvIreeHalTargetTest, CreatesProfileFromHalFacts) {
  iree_hal_vulkan_features_t enabled_features = RequiredVulkanFeatures();
  enabled_features.general |=
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16 |
      IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_8BIT_ACCESS;
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateVulkanDeviceSpec(
      enabled_features, kCompleteDeviceSpecFlags, &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_result_t* result = nullptr;
  const iree_hal_executable_target_t* executable_target = nullptr;
  TargetProfilePtr profile = SelectTargetFromHal(
      target_environment.get(), &device, &result, &executable_target);
  ResultPtr result_ptr(result);

  ASSERT_NE(profile.get(), nullptr);
  ASSERT_NE(executable_target, nullptr);
  EXPECT_TRUE(iree_string_view_equal(executable_target->target_key,
                                     IREE_SV("vulkan1.3+bda")));
  ExpectSucceededResult(result_ptr.get());
  loomc_spirv_limit_value_t limit = {};
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_limit(
      profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X, &limit));
  EXPECT_EQ(limit.state, LOOMC_TARGET_FACT_STATE_TRUE);
  EXPECT_EQ(limit.value, 256u);
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_limit(
      profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES, &limit));
  EXPECT_EQ(limit.state, LOOMC_TARGET_FACT_STATE_TRUE);
  EXPECT_EQ(limit.value, kMaximumWorkgroupLocalMemorySize);
  loomc_target_fact_state_t feature_state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_feature(
      profile.get(), LOOMC_SPIRV_FEATURE_FLOAT16, &feature_state));
  EXPECT_EQ(feature_state, LOOMC_TARGET_FACT_STATE_TRUE);
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_feature(
      profile.get(), LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
      &feature_state));
  EXPECT_EQ(feature_state, LOOMC_TARGET_FACT_STATE_TRUE);
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_feature(
      profile.get(), LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM, &feature_state));
  EXPECT_EQ(feature_state, LOOMC_TARGET_FACT_STATE_TRUE);
}

TEST(LoomcSpirvIreeHalTargetTest, PreservesExtendedCompilerFeatureFamily) {
  iree_hal_vulkan_features_t enabled_features = RequiredVulkanFeatures();
  enabled_features.general |=
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16 |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT64 |
      IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL |
      IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE;
  enabled_features.atomics =
      IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_ALL_RECOGNIZED;
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateVulkanDeviceSpec(
      enabled_features, kCompleteDeviceSpecFlags, &device_spec,
      /*cooperative_matrix_property_count=*/0,
      /*cooperative_matrix_properties=*/nullptr,
      IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_FLOAT32_DENORM_PRESERVE,
      /*subgroup_supported_operations=*/kVulkanSubgroupFeatureBallotBit));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile =
      SelectTargetFromHal(target_environment.get(), &device, &result);
  ResultPtr result_ptr(result);

  ASSERT_NE(profile.get(), nullptr);
  ExpectSucceededResult(result_ptr.get());
  const loomc_spirv_feature_t expected_features[] = {
      LOOMC_SPIRV_FEATURE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE,
      LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_INT64_ATOMICS,
      LOOMC_SPIRV_FEATURE_WORKGROUP_INT64_ATOMICS,
      LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT16_ATOMICS,
      LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMICS,
      LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT16_ATOMIC_ADD,
      LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT16_ATOMIC_ADD,
      LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT32_ATOMICS,
      LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMICS,
      LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT32_ATOMIC_ADD,
      LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT32_ATOMIC_ADD,
      LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT64_ATOMICS,
      LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMICS,
      LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_FLOAT64_ATOMIC_ADD,
      LOOMC_SPIRV_FEATURE_WORKGROUP_FLOAT64_ATOMIC_ADD,
      LOOMC_SPIRV_FEATURE_FLOAT32_DENORM_PRESERVE,
      LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_BALLOT,
  };
  for (loomc_spirv_feature_t feature : expected_features) {
    loomc_target_fact_state_t state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
    LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_feature(profile.get(),
                                                             feature, &state));
    EXPECT_EQ(state, LOOMC_TARGET_FACT_STATE_TRUE) << feature;
  }
}

TEST(LoomcSpirvIreeHalTargetTest,
     FiltersCooperativeMatrixRowsToDeviceProperties) {
  const iree_hal_vulkan_cooperative_matrix_property_t properties[] = {
      F16MatrixProperty(),
      Bf16MatrixProperty(),
      S8MatrixProperty(/*saturating_accumulation=*/0),
      U8MatrixProperty(),
  };
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateVulkanDeviceSpec(
      CooperativeMatrixVulkanFeatures(), kCompleteDeviceSpecFlags, &device_spec,
      IREE_ARRAYSIZE(properties), properties));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile =
      SelectTargetFromHal(target_environment.get(), &device, &result);
  ResultPtr result_ptr(result);

  ASSERT_NE(profile.get(), nullptr);
  ExpectSucceededResult(result_ptr.get());
  loomc_spirv_cooperative_matrix_row_t row = {};
  ASSERT_TRUE(
      FindCooperativeMatrixRow(profile.get(), kF16CooperativeMatrixRow, &row));
  EXPECT_EQ(row.state, LOOMC_TARGET_FACT_STATE_TRUE);
  ASSERT_TRUE(
      FindCooperativeMatrixRow(profile.get(), kBf16CooperativeMatrixRow, &row));
  EXPECT_EQ(row.state, LOOMC_TARGET_FACT_STATE_TRUE);
  ASSERT_TRUE(
      FindCooperativeMatrixRow(profile.get(), kU8CooperativeMatrixRow, &row));
  EXPECT_EQ(row.state, LOOMC_TARGET_FACT_STATE_TRUE);
  ASSERT_TRUE(
      FindCooperativeMatrixRow(profile.get(), kS8CooperativeMatrixRow, &row));
  EXPECT_EQ(row.state, LOOMC_TARGET_FACT_STATE_FALSE);
  EXPECT_EQ(ToString(row.provenance),
            "iree-hal:vulkan.device.cooperative_matrix_properties");
}

TEST(LoomcSpirvIreeHalTargetTest, AcceptsSaturatingCooperativeMatrixProperty) {
  const iree_hal_vulkan_cooperative_matrix_property_t property =
      S8MatrixProperty(/*saturating_accumulation=*/1);
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateVulkanDeviceSpec(
      CooperativeMatrixVulkanFeatures(), kCompleteDeviceSpecFlags, &device_spec,
      /*cooperative_matrix_property_count=*/1, &property));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile =
      SelectTargetFromHal(target_environment.get(), &device, &result);
  ResultPtr result_ptr(result);

  ASSERT_NE(profile.get(), nullptr);
  ExpectSucceededResult(result_ptr.get());
  loomc_spirv_cooperative_matrix_row_t row = {};
  ASSERT_TRUE(
      FindCooperativeMatrixRow(profile.get(), kS8CooperativeMatrixRow, &row));
  EXPECT_EQ(row.state, LOOMC_TARGET_FACT_STATE_TRUE);
  ASSERT_TRUE(
      FindCooperativeMatrixRow(profile.get(), kF16CooperativeMatrixRow, &row));
  EXPECT_EQ(row.state, LOOMC_TARGET_FACT_STATE_FALSE);
}

TEST(LoomcSpirvIreeHalTargetTest,
     EmptyDeviceTableRejectsAllCooperativeMatrixRows) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateVulkanDeviceSpec(CooperativeMatrixVulkanFeatures(),
                                        kCompleteDeviceSpecFlags,
                                        &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile =
      SelectTargetFromHal(target_environment.get(), &device, &result);
  ResultPtr result_ptr(result);

  ASSERT_NE(profile.get(), nullptr);
  ExpectSucceededResult(result_ptr.get());
  loomc_spirv_profile_info_t info = {};
  LOOMC_ASSERT_OK(loomc_spirv_target_profile_query_info(profile.get(), &info));
  ASSERT_GT(info.cooperative_matrix_row_count, 0u);
  for (loomc_host_size_t i = 0; i < info.cooperative_matrix_row_count; ++i) {
    loomc_spirv_cooperative_matrix_row_t row = {};
    LOOMC_ASSERT_OK(loomc_spirv_target_profile_cooperative_matrix_row_at(
        profile.get(), i, &row));
    EXPECT_EQ(row.state, LOOMC_TARGET_FACT_STATE_FALSE) << ToString(row.name);
  }
}

TEST(LoomcSpirvIreeHalTargetTest, MissingExecutableTargetFailsResult) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(
      CreateVulkanDeviceSpec(RequiredVulkanFeatures(),
                             kDeviceSpecFlagIncludeDispatch |
                                 kDeviceSpecFlagIncludeWorkgroupStorageLimit,
                             &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile =
      SelectTargetFromHal(target_environment.get(), &device, &result);
  ResultPtr result_ptr(result);

  EXPECT_EQ(profile.get(), nullptr);
  ExpectFailedSpirvIreeHalResult(result_ptr.get());
}

TEST(LoomcSpirvIreeHalTargetTest, MissingRequiredHalFactFailsResult) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateVulkanDeviceSpec(RequiredVulkanFeatures(),
                                        kDeviceSpecFlagIncludeExecutableTarget,
                                        &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile =
      SelectTargetFromHal(target_environment.get(), &device, &result);
  ResultPtr result_ptr(result);

  EXPECT_EQ(profile.get(), nullptr);
  ExpectFailedSpirvIreeHalResult(result_ptr.get());
}

TEST(LoomcSpirvIreeHalTargetTest, MissingWorkgroupStorageLimitFailsResult) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateVulkanDeviceSpec(
      RequiredVulkanFeatures(),
      kDeviceSpecFlagIncludeDispatch | kDeviceSpecFlagIncludeExecutableTarget,
      &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile =
      SelectTargetFromHal(target_environment.get(), &device, &result);
  ResultPtr result_ptr(result);

  EXPECT_EQ(profile.get(), nullptr);
  ExpectFailedSpirvIreeHalResult(result_ptr.get());
}

TEST(LoomcSpirvIreeHalTargetTest, ProviderRoutesThroughGenericHalRouter) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateVulkanDeviceSpec(
      RequiredVulkanFeatures(), kCompleteDeviceSpecFlags, &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  iree_hal_device_t* hal_device = reinterpret_cast<iree_hal_device_t*>(&device);
  const loomc_iree_hal_target_provider_t* providers[] = {
      loomc_spirv_iree_hal_target_provider(),
  };
  loomc_iree_hal_target_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_IREE_HAL_TARGET_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .identifier = loomc_make_cstring_view("router"),
      .device = hal_device,
      .physical_device_affinity = 0,
      .target_profile = nullptr,
      .providers = providers,
      .provider_count = IREE_ARRAYSIZE(providers),
  };
  loomc_result_t* result = nullptr;
  loomc_iree_hal_target_selection_t selection = {};
  LOOMC_ASSERT_OK(loomc_target_select_iree_hal(
      target_environment.get(), &options, loomc_allocator_system(), &selection,
      &result));
  TargetProfilePtr profile_ptr(selection.target_profile);
  ResultPtr result_ptr(result);

  ASSERT_NE(profile_ptr.get(), nullptr);
  ASSERT_NE(selection.executable_target, nullptr);
  EXPECT_TRUE(iree_string_view_equal(selection.executable_target->target_key,
                                     IREE_SV("vulkan1.3+bda")));
  ExpectSucceededResult(result_ptr.get());
}

TEST(LoomcSpirvIreeHalTargetTest,
     ForcedProfilePreservesExactCompilerAndLoaderPair) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateVulkanDeviceSpec(
      RequiredVulkanFeatures(), kCompleteDeviceSpecFlags, &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_target_profile_t* requested_profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_select(
      target_environment.get(), loomc_make_cstring_view("spirv:vulkan1.3+bda"),
      loomc_allocator_system(), &requested_profile));
  TargetProfilePtr requested_profile_ptr(requested_profile);
  const loomc_iree_hal_target_provider_t* providers[] = {
      loomc_spirv_iree_hal_target_provider(),
  };
  loomc_iree_hal_target_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_IREE_HAL_TARGET_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .identifier = loomc_string_view_empty(),
      .device = reinterpret_cast<iree_hal_device_t*>(&device),
      .physical_device_affinity = 0,
      .target_profile = requested_profile_ptr.get(),
      .providers = providers,
      .provider_count = IREE_ARRAYSIZE(providers),
  };
  loomc_iree_hal_target_selection_t selection = {};
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_target_select_iree_hal(
      target_environment.get(), &options, loomc_allocator_system(), &selection,
      &result));
  TargetProfilePtr selected_profile_ptr(selection.target_profile);
  ResultPtr result_ptr(result);

  ExpectSucceededResult(result_ptr.get());
  EXPECT_EQ(selected_profile_ptr.get(), requested_profile_ptr.get());
  ASSERT_NE(selection.executable_target, nullptr);
  EXPECT_TRUE(iree_string_view_equal(selection.executable_target->target_key,
                                     IREE_SV("vulkan1.3+bda")));
}

TEST(LoomcSpirvIreeHalTargetTest, ForcedProfileMustMatchLoaderContract) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateVulkanDeviceSpec(
      RequiredVulkanFeatures(), kCompleteDeviceSpecFlags, &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_target_profile_t* requested_profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_select(
      target_environment.get(),
      loomc_make_cstring_view("spirv:vulkan1.3+bda+extended-types"),
      loomc_allocator_system(), &requested_profile));
  TargetProfilePtr requested_profile_ptr(requested_profile);
  const loomc_iree_hal_target_provider_t* providers[] = {
      loomc_spirv_iree_hal_target_provider(),
  };
  loomc_iree_hal_target_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_IREE_HAL_TARGET_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .identifier = loomc_string_view_empty(),
      .device = reinterpret_cast<iree_hal_device_t*>(&device),
      .physical_device_affinity = 0,
      .target_profile = requested_profile_ptr.get(),
      .providers = providers,
      .provider_count = IREE_ARRAYSIZE(providers),
  };
  loomc_iree_hal_target_selection_t selection = {};
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_target_select_iree_hal(
      target_environment.get(), &options, loomc_allocator_system(), &selection,
      &result));
  TargetProfilePtr selected_profile_ptr(selection.target_profile);
  ResultPtr result_ptr(result);

  EXPECT_EQ(selected_profile_ptr.get(), nullptr);
  EXPECT_EQ(selection.executable_target, nullptr);
  ExpectFailedSpirvIreeHalResult(result_ptr.get());
}

TEST(LoomcSpirvIreeHalTargetTest, ProviderMissLetsRouterReportUnsupported) {
  FakeHalDevice device = {};
  InitializeFakeDevice(nullptr, &device);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  iree_hal_device_t* hal_device = reinterpret_cast<iree_hal_device_t*>(&device);
  const loomc_iree_hal_target_provider_t* providers[] = {
      loomc_spirv_iree_hal_target_provider(),
  };
  loomc_iree_hal_target_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_IREE_HAL_TARGET_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .identifier = loomc_make_cstring_view("miss"),
      .device = hal_device,
      .physical_device_affinity = 0,
      .target_profile = nullptr,
      .providers = providers,
      .provider_count = IREE_ARRAYSIZE(providers),
  };
  loomc_result_t* result = nullptr;
  loomc_iree_hal_target_selection_t selection = {};
  LOOMC_ASSERT_OK(loomc_target_select_iree_hal(
      target_environment.get(), &options, loomc_allocator_system(), &selection,
      &result));
  TargetProfilePtr profile_ptr(selection.target_profile);
  ResultPtr result_ptr(result);

  EXPECT_EQ(profile_ptr.get(), nullptr);
  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result_ptr.get()));
}

}  // namespace
