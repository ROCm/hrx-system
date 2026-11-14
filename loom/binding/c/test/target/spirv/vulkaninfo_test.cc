// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/spirv/vulkaninfo.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/temp_file.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/status.h"
#include "loomc/target/spirv/base.h"
#include "loomc/target/spirv/profile.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;

constexpr char kRawDevicesJson[] = R"json({
  "devices": [
    {
      "properties": {
        "apiVersion": 4202496,
        "limits": {
          "maxComputeWorkGroupCount": [16, 8, 4],
          "maxComputeWorkGroupInvocations": 128,
          "maxComputeWorkGroupSize": [128, 2, 1],
          "maxComputeSharedMemorySize": 32768
        }
      },
      "features": {
        "shaderInt64": true,
        "shaderFloat64": true
      },
      "extended_features": {
        "VkPhysicalDeviceVulkan12Features": {
          "bufferDeviceAddress": true
        }
      }
    }
  ]
})json";

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

SourcePtr CreateJsonSource(const char* identifier, const char* json) {
  loomc_source_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .format = LOOMC_SOURCE_FORMAT_UNKNOWN,
      .identifier = loomc_make_cstring_view(identifier),
      .contents = loomc_make_byte_span(json, strlen(json)),
      .storage = LOOMC_SOURCE_STORAGE_COPY,
      .release = nullptr,
      .release_user_data = nullptr,
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  return SourcePtr(source);
}

TargetEnvironmentPtr CreateSpirvTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_spirv(
      loomc_allocator_system(), &target_environment);
  LOOMC_EXPECT_OK(status);
  return TargetEnvironmentPtr(target_environment);
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

void ExpectFailedResultCode(const loomc_result_t* result,
                            const char* expected_code) {
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result));
  EXPECT_EQ(loomc_result_state(result), LOOMC_RESULT_STATE_FAILED);
  ASSERT_GE(loomc_result_diagnostic_count(result), 1u);
  const loomc_diagnostic_t* diagnostic = loomc_result_diagnostic_at(result, 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(diagnostic->severity, LOOMC_DIAGNOSTIC_SEVERITY_ERROR);
  EXPECT_EQ(ToString(diagnostic->code), expected_code);
}

TargetProfilePtr ImportVulkaninfoProfile(
    loomc_target_environment_t* target_environment,
    const loomc_source_t* source,
    const loomc_spirv_vulkaninfo_profile_options_t* options) {
  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_target_profile_create_spirv_vulkaninfo(
      target_environment, source, options, loomc_allocator_system(), &profile,
      &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return TargetProfilePtr(profile);
}

void ExpectFeatureState(const loomc_target_profile_t* profile,
                        loomc_spirv_feature_t feature,
                        loomc_target_fact_state_t expected_state) {
  loomc_target_fact_state_t state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_EXPECT_OK(
      loomc_spirv_target_profile_query_feature(profile, feature, &state));
  EXPECT_EQ(state, expected_state);
}

void ExpectLimitValue(const loomc_target_profile_t* profile,
                      loomc_spirv_limit_t limit,
                      loomc_target_fact_state_t expected_state,
                      uint64_t expected_value) {
  loomc_spirv_limit_value_t value = {
      .state = LOOMC_TARGET_FACT_STATE_UNKNOWN,
      .value = 0,
  };
  LOOMC_EXPECT_OK(
      loomc_spirv_target_profile_query_limit(profile, limit, &value));
  EXPECT_EQ(value.state, expected_state);
  EXPECT_EQ(value.value, expected_value);
}

void ExpectEnvironmentValue(const loomc_target_profile_t* profile,
                            loomc_spirv_environment_t environment,
                            loomc_target_fact_state_t expected_state,
                            uint64_t expected_value) {
  loomc_spirv_environment_value_t value = {
      .state = LOOMC_TARGET_FACT_STATE_UNKNOWN,
      .value = 0,
  };
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_environment(
      profile, environment, &value));
  EXPECT_EQ(value.state, expected_state);
  EXPECT_EQ(value.value, expected_value);
}

TEST(TargetSpirvVulkaninfoTest, ImportsFusedArithmeticWidthsIndependently) {
  auto environment = CreateSpirvTargetEnvironment();
  auto source = CreateJsonSource("fma-device.json", R"json({
    "devices": [{
      "properties": { "apiVersion": 4206592 },
      "features": { "shaderFloat64": true },
      "extended_features": {
        "VkPhysicalDeviceVulkan12Features": { "shaderFloat16": true },
        "VkPhysicalDeviceShaderFmaFeaturesKHR": {
          "shaderFmaFloat16": true,
          "shaderFmaFloat32": false,
          "shaderFmaFloat64": true
        }
      }
    }]
  })json");
  auto profile =
      ImportVulkaninfoProfile(environment.get(), source.get(), nullptr);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_FMA_FLOAT16,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_FMA_FLOAT32,
                     LOOMC_TARGET_FACT_STATE_FALSE);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_FMA_FLOAT64,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  auto absent_source = CreateJsonSource("older-device.json", kRawDevicesJson);
  auto absent_profile =
      ImportVulkaninfoProfile(environment.get(), absent_source.get(), nullptr);
  for (auto feature :
       {LOOMC_SPIRV_FEATURE_FMA_FLOAT16, LOOMC_SPIRV_FEATURE_FMA_FLOAT32,
        LOOMC_SPIRV_FEATURE_FMA_FLOAT64, LOOMC_SPIRV_FEATURE_FLOAT_CONTROLS2}) {
    ExpectFeatureState(absent_profile.get(), feature,
                       LOOMC_TARGET_FACT_STATE_UNKNOWN);
  }
}

TEST(TargetSpirvVulkaninfoTest, RetainsIndependentHalfAndWideFloatControls) {
  auto environment = CreateSpirvTargetEnvironment();
  const char* names[] = {"VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL",
                         "VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY",
                         "VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE", nullptr};
  for (const char* independence : names) {
    SCOPED_TRACE(independence ? independence : "unknown");
    std::string controls = R"json({"shaderDenormPreserveFloat16":true,
      "shaderDenormPreserveFloat32":false,"shaderDenormPreserveFloat64":false,
      "shaderRoundingModeRTEFloat16":false,"shaderRoundingModeRTEFloat32":false,
      "shaderRoundingModeRTEFloat64":true)json";
    if (independence) {
      controls += std::string(",\"denormBehaviorIndependence\":\"") +
                  independence + "\",\"roundingModeIndependence\":\"" +
                  independence + "\"";
    }
    controls += "}";
    const std::string json =
        R"json({"devices":[{"properties":{"apiVersion":4206592,
          "VkPhysicalDeviceFloatControlsProperties":)json" +
        controls + "}}]}";
    auto source = CreateJsonSource("float-widths.json", json.c_str());
    auto profile =
        ImportVulkaninfoProfile(environment.get(), source.get(), nullptr);
    const auto expected = !independence ? LOOMC_TARGET_FACT_STATE_UNKNOWN
                          : independence == names[0]
                              ? LOOMC_TARGET_FACT_STATE_TRUE
                              : LOOMC_TARGET_FACT_STATE_FALSE;
    ExpectFeatureState(profile.get(),
                       LOOMC_SPIRV_FEATURE_FLOAT16_DENORM_PRESERVE, expected);
    ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_FLOAT64_ROUNDING_RTE,
                       expected);
    ExpectFeatureState(profile.get(),
                       LOOMC_SPIRV_FEATURE_FLOAT64_DENORM_PRESERVE,
                       LOOMC_TARGET_FACT_STATE_FALSE);
    ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_FLOAT16_ROUNDING_RTE,
                       LOOMC_TARGET_FACT_STATE_FALSE);
  }
}

TEST(TargetSpirvVulkaninfoTest, ImportsFloatControls2CoreAndExtensionNames) {
  auto environment = CreateSpirvTargetEnvironment();
  for (const char* name : {"VkPhysicalDeviceVulkan14Features",
                           "VkPhysicalDeviceShaderFloatControls2Features",
                           "VkPhysicalDeviceShaderFloatControls2FeaturesKHR"}) {
    for (bool enabled : {false, true}) {
      SCOPED_TRACE(::testing::Message() << name << ":" << enabled);
      const std::string json =
          std::string(R"json({"devices":[{"properties":{"apiVersion":4210688},
            "extended_features":{")json") +
          name +
          "\":{\"shaderFloatControls2\":" + (enabled ? "true" : "false") +
          "}}}]}";
      auto source = CreateJsonSource("float-controls.json", json.c_str());
      auto profile =
          ImportVulkaninfoProfile(environment.get(), source.get(), nullptr);
      ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_FLOAT_CONTROLS2,
                         enabled ? LOOMC_TARGET_FACT_STATE_TRUE
                                 : LOOMC_TARGET_FACT_STATE_FALSE);
    }
  }
}

TEST(TargetSpirvVulkaninfoTest, ImportsGpuinfoProfileWrapper) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  const std::string json = R"json({
    "capabilities": {
      "device": {
        "properties": {
          "VkPhysicalDeviceProperties": {
            "apiVersion": 4206592,
            "vendorID": 4318,
            "deviceName": "Unit Test GPU",
            "limits": {
              "maxComputeWorkGroupCount": [65535, 1024, 64],
              "maxComputeWorkGroupInvocations": 256,
              "maxComputeWorkGroupSize": [256, 128, 64],
              "maxComputeSharedMemorySize": 49152
            }
          },
          "VkPhysicalDeviceVulkan11Properties": {
            "subgroupSize": 32,
            "subgroupSupportedOperations": [
              "VK_SUBGROUP_FEATURE_BASIC_BIT",
              "VK_SUBGROUP_FEATURE_BALLOT_BIT",
              "VK_SUBGROUP_FEATURE_VOTE_BIT",
              "VK_SUBGROUP_FEATURE_ARITHMETIC_BIT",
              "VK_SUBGROUP_FEATURE_SHUFFLE_BIT",
              "VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT",
              "VK_SUBGROUP_FEATURE_CLUSTERED_BIT"
            ]
          },
          "VkPhysicalDeviceVulkan12Properties": {
            "denormBehaviorIndependence":
              "VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY",
            "roundingModeIndependence":
              "VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_32_BIT_ONLY",
            "shaderDenormPreserveFloat16": true,
            "shaderDenormPreserveFloat32": true,
            "shaderDenormPreserveFloat64": true,
            "shaderSignedZeroInfNanPreserveFloat16": true,
            "shaderSignedZeroInfNanPreserveFloat32": true,
            "shaderSignedZeroInfNanPreserveFloat64": true,
            "shaderRoundingModeRTEFloat16": true,
            "shaderRoundingModeRTEFloat32": true,
            "shaderRoundingModeRTEFloat64": true
          }
        },
        "features": {
          "VkPhysicalDeviceFeatures": {
            "shaderFloat64": true,
            "shaderInt16": true,
            "shaderInt64": true
          },
          "VkPhysicalDeviceVulkan12Features": {
            "shaderSubgroupExtendedTypes": true,
            "shaderFloat16": true,
            "shaderInt8": true,
            "storageBuffer8BitAccess": true,
            "bufferDeviceAddress": true,
            "vulkanMemoryModel": true,
            "vulkanMemoryModelDeviceScope": true,
            "shaderBufferInt64Atomics": true,
            "shaderSharedInt64Atomics": true
          },
          "VkPhysicalDeviceVulkan11Features": {
            "storageBuffer16BitAccess": true
          },
          "VkPhysicalDeviceCooperativeMatrixFeaturesKHR": {
            "cooperativeMatrix": true
          },
          "VkPhysicalDeviceShaderAtomicFloatFeaturesEXT": {
            "shaderBufferFloat32Atomics": true,
            "shaderSharedFloat32Atomics": true,
            "shaderBufferFloat32AtomicAdd": true,
            "shaderSharedFloat32AtomicAdd": true,
            "shaderBufferFloat64Atomics": true,
            "shaderSharedFloat64Atomics": true,
            "shaderBufferFloat64AtomicAdd": true,
            "shaderSharedFloat64AtomicAdd": true
          },
          "VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT": {
            "shaderBufferFloat16Atomics": true,
            "shaderSharedFloat16Atomics": true,
            "shaderBufferFloat16AtomicAdd": true,
            "shaderSharedFloat16AtomicAdd": true
          },
          "UnexpectedFutureFeatureStruct": {
            "surprise": true
          }
        }
      }
    },
    "profiles": {
      "VP_unit_test": {
        "version": 1,
        "api-version": "1.3.0",
        "capabilities": ["device"],
        "label": "Unit Test GPU"
      }
    }
  })json";
  SourcePtr source = CreateJsonSource("unit-vulkaninfo.json", json.c_str());
  loomc_spirv_vulkaninfo_profile_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SPIRV_VULKANINFO_PROFILE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .identifier = loomc_make_cstring_view("imported-wrapper"),
      .profile_name = loomc_make_cstring_view("VP_unit_test"),
      .device_index = 0,
  };
  TargetProfilePtr profile =
      ImportVulkaninfoProfile(target_environment.get(), source.get(), &options);

  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_VULKAN_SHADER,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_FLOAT16,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_FLOAT64,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(),
                     LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  const loomc_spirv_feature_t extended_features[] = {
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
      LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_VOTE,
      LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_ARITHMETIC,
      LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_CLUSTERED,
      LOOMC_SPIRV_FEATURE_SUBGROUP_EXTENDED_TYPES,
      LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_SHUFFLE,
      LOOMC_SPIRV_FEATURE_GROUP_NON_UNIFORM_SHUFFLE_RELATIVE,
      LOOMC_SPIRV_FEATURE_FLOAT16_64_DENORM_PRESERVE,
      LOOMC_SPIRV_FEATURE_FLOAT16_64_ROUNDING_RTE,
      LOOMC_SPIRV_FEATURE_FLOAT32_ROUNDING_RTE,
      LOOMC_SPIRV_FEATURE_FLOAT16_SIGNED_ZERO_INF_NAN_PRESERVE,
      LOOMC_SPIRV_FEATURE_FLOAT32_SIGNED_ZERO_INF_NAN_PRESERVE,
      LOOMC_SPIRV_FEATURE_FLOAT64_SIGNED_ZERO_INF_NAN_PRESERVE,
      LOOMC_SPIRV_FEATURE_FLOAT64_TO_FLOAT16_CONVERSION,
  };
  for (loomc_spirv_feature_t feature : extended_features) {
    ExpectFeatureState(profile.get(), feature, LOOMC_TARGET_FACT_STATE_TRUE);
  }

  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X,
                   LOOMC_TARGET_FACT_STATE_TRUE, 256);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Y,
                   LOOMC_TARGET_FACT_STATE_TRUE, 128);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Z,
                   LOOMC_TARGET_FACT_STATE_TRUE, 64);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_FLAT_WORKGROUP_SIZE,
                   LOOMC_TARGET_FACT_STATE_TRUE, 256);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES,
                   LOOMC_TARGET_FACT_STATE_TRUE, UINT64_C(49152));
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_DEFAULT_SUBGROUP_SIZE,
                   LOOMC_TARGET_FACT_STATE_TRUE, 32);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_X,
                   LOOMC_TARGET_FACT_STATE_TRUE, 65535);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_X,
                   LOOMC_TARGET_FACT_STATE_UNKNOWN, 0);
  ExpectEnvironmentValue(profile.get(),
                         LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION,
                         LOOMC_TARGET_FACT_STATE_TRUE, LOOMC_SPIRV_VERSION_1_6);

  loomc_spirv_profile_info_t info = {};
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_info(profile.get(), &info));
  EXPECT_GE(info.extension_count, 1u);
  EXPECT_GE(info.capability_count, 1u);
  EXPECT_GE(info.cooperative_matrix_row_count, 1u);
}

TEST(TargetSpirvVulkaninfoTest, ImportsRawDevicesArrayByIndex) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  const std::string json = R"json({
    "devices": [
      {
        "properties": {
          "apiVersion": 4194304,
          "limits": {
            "maxComputeWorkGroupCount": [1, 1, 1],
            "maxComputeWorkGroupInvocations": 64,
            "maxComputeWorkGroupSize": [64, 1, 1],
            "maxComputeSharedMemorySize": 8192
          }
        },
        "features": {
          "shaderInt64": false
        }
      },
      {
        "properties": {
          "apiVersion": 4202496,
          "vendorID": 4098,
          "limits": {
            "maxComputeWorkGroupCount": [16, 8, 4],
            "maxComputeWorkGroupInvocations": 128,
            "maxComputeWorkGroupSize": [128, 2, 1],
            "maxComputeSharedMemorySize": 32768
          }
        },
        "features": {
          "shaderInt64": true,
          "shaderFloat64": true
        },
        "extended_features": {
          "VkPhysicalDeviceVulkan12Features": {
            "bufferDeviceAddress": true
          },
          "VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures": {
            "shaderSubgroupExtendedTypes": true
          }
        }
      }
    ]
  })json";
  SourcePtr source = CreateJsonSource("raw-vulkaninfo.json", json.c_str());
  loomc_spirv_vulkaninfo_profile_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SPIRV_VULKANINFO_PROFILE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .identifier = loomc_make_cstring_view("selected-device"),
      .profile_name = loomc_string_view_empty(),
      .device_index = 1,
  };
  TargetProfilePtr profile =
      ImportVulkaninfoProfile(target_environment.get(), source.get(), &options);

  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_SUBGROUP_EXTENDED_TYPES,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_INT64,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_FLOAT64,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectFeatureState(profile.get(),
                     LOOMC_SPIRV_FEATURE_FLOAT64_TO_FLOAT16_CONVERSION,
                     LOOMC_TARGET_FACT_STATE_FALSE);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X,
                   LOOMC_TARGET_FACT_STATE_TRUE, 128);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_X,
                   LOOMC_TARGET_FACT_STATE_TRUE, 16);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES,
                   LOOMC_TARGET_FACT_STATE_TRUE, UINT64_C(32768));
  ExpectEnvironmentValue(profile.get(),
                         LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION,
                         LOOMC_TARGET_FACT_STATE_TRUE, LOOMC_SPIRV_VERSION_1_5);
}

TEST(TargetSpirvVulkaninfoTest, ImportsSourceLoadedFromOpenFile) {
  FILE* file = tmpfile();
  ASSERT_NE(file, nullptr);
  ASSERT_EQ(fwrite(kRawDevicesJson, 1, strlen(kRawDevicesJson), file),
            strlen(kRawDevicesJson));
  ASSERT_EQ(fseek(file, 0, SEEK_SET), 0);

  loomc_source_load_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_LOAD_OPTIONS,
      .structure_size = sizeof(source_options),
      .next = nullptr,
      .format = LOOMC_SOURCE_FORMAT_UNKNOWN,
      .identifier = loomc_make_cstring_view("open-file-vulkaninfo.json"),
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status = loomc_source_create_from_file(
      file, &source_options, loomc_allocator_system(), &source);
  LOOMC_ASSERT_OK(status);
  SourcePtr source_ptr(source);
  ASSERT_EQ(fclose(file), 0);

  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  TargetProfilePtr profile = ImportVulkaninfoProfile(
      target_environment.get(), source_ptr.get(), /*options=*/nullptr);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES,
                   LOOMC_TARGET_FACT_STATE_TRUE, UINT64_C(32768));
}

TEST(TargetSpirvVulkaninfoTest, ImportsSourceLoadedFromPath) {
  iree::testing::TempFilePath path("loomc_vulkaninfo", ".json");
  {
    std::ofstream output(path.path(), std::ios::binary);
    output << kRawDevicesJson;
    output.close();
    ASSERT_TRUE(output.good());
  }

  loomc_source_t* source = nullptr;
  loomc_status_t status = loomc_source_create_from_path(
      loomc_make_string_view(path.path().data(), path.path().size()), nullptr,
      loomc_allocator_system(), &source);
  LOOMC_ASSERT_OK(status);
  SourcePtr source_ptr(source);

  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  TargetProfilePtr profile = ImportVulkaninfoProfile(
      target_environment.get(), source_ptr.get(), /*options=*/nullptr);
  ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
                     LOOMC_TARGET_FACT_STATE_TRUE);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X,
                   LOOMC_TARGET_FACT_STATE_TRUE, 128);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES,
                   LOOMC_TARGET_FACT_STATE_TRUE, UINT64_C(32768));

  EXPECT_TRUE(path.Remove());
}

TEST(TargetSpirvVulkaninfoTest, ImportsComputeSubgroupSizeControlFamily) {
  TargetEnvironmentPtr environment = CreateSpirvTargetEnvironment();
  for (const std::string spelling :
       {"Vulkan13", "SubgroupSizeControl", "SubgroupSizeControlEXT"}) {
    for (bool control : {false, true}) {
      for (bool compute : {false, true}) {
        SCOPED_TRACE(spelling);
        SCOPED_TRACE(control);
        SCOPED_TRACE(compute);
        const std::string stem = spelling == "SubgroupSizeControlEXT"
                                     ? "SubgroupSizeControl"
                                     : spelling;
        const std::string suffix =
            spelling == "SubgroupSizeControlEXT" ? "EXT" : "";
        const std::string json =
            R"({"devices":[{"properties":{
              "VkPhysicalDeviceSubgroupProperties":{"subgroupSize":64},
              "VkPhysicalDevice)" +
            stem + "Properties" + suffix + R"(":{
                "minSubgroupSize":32,"maxSubgroupSize":64,
                "maxComputeWorkgroupSubgroups":16,"requiredSubgroupSizeStages":[")" +
            (compute ? "VK_SHADER_STAGE_COMPUTE_BIT"
                     : "VK_SHADER_STAGE_FRAGMENT_BIT") +
            R"("]}},"extended_features":{"VkPhysicalDevice)" + stem +
            "Features" + suffix + R"(":{"subgroupSizeControl":)" +
            (control ? "true" : "false") + "}}}]}";
        SourcePtr source = CreateJsonSource("subgroups.json", json.c_str());
        TargetProfilePtr profile =
            ImportVulkaninfoProfile(environment.get(), source.get(), nullptr);
        ASSERT_NE(profile.get(), nullptr);
        ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_DEFAULT_SUBGROUP_SIZE,
                         LOOMC_TARGET_FACT_STATE_TRUE, 64);
        ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MIN_SUBGROUP_SIZE,
                         LOOMC_TARGET_FACT_STATE_TRUE,
                         control && compute ? 32 : 64);
        ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_SUBGROUP_SIZE,
                         LOOMC_TARGET_FACT_STATE_TRUE, 64);
        ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE,
                         LOOMC_TARGET_FACT_STATE_UNKNOWN, 0);
        ExpectLimitValue(profile.get(),
                         LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SUBGROUP_COUNT,
                         control && compute ? LOOMC_TARGET_FACT_STATE_TRUE
                                            : LOOMC_TARGET_FACT_STATE_UNKNOWN,
                         control && compute ? 16 : 0);
      }
    }
  }
}

TEST(TargetSpirvVulkaninfoTest, ImportsStorageFeatureStructFamily) {
  TargetEnvironmentPtr environment = CreateSpirvTargetEnvironment();
  for (const std::string spelling : {"Vulkan11", "16BitStorage"}) {
    for (const std::string container : {"features", "extended_features"}) {
      for (const std::string value : {"true", "false", "null", ""}) {
        SCOPED_TRACE(spelling);
        SCOPED_TRACE(container);
        SCOPED_TRACE(value);
        const std::string field =
            value.empty() ? "" : "\"storageBuffer16BitAccess\":" + value;
        const std::string json =
            "{\"devices\":[{\"properties\":{\"apiVersion\":4206592},\"" +
            container + "\":{\"VkPhysicalDevice" + spelling + "Features\":{" +
            field + "}}}]}";
        SourcePtr source = CreateJsonSource("storage.json", json.c_str());
        TargetProfilePtr profile =
            ImportVulkaninfoProfile(environment.get(), source.get(), nullptr);
        const loomc_target_fact_state_t expected =
            value == "true"    ? LOOMC_TARGET_FACT_STATE_TRUE
            : value == "false" ? LOOMC_TARGET_FACT_STATE_FALSE
                               : LOOMC_TARGET_FACT_STATE_UNKNOWN;
        ExpectFeatureState(profile.get(),
                           LOOMC_SPIRV_FEATURE_STORAGE_BUFFER_16BIT_ACCESS,
                           expected);
      }
    }
  }
}

TEST(TargetSpirvVulkaninfoTest, IntegerDotFeatureStructFamily) {
  TargetEnvironmentPtr environment = CreateSpirvTargetEnvironment();
  for (const std::string spelling :
       {"Vulkan13Features", "ShaderIntegerDotProductFeatures",
        "ShaderIntegerDotProductFeaturesKHR"}) {
    for (const std::string container : {"features", "extended_features"}) {
      for (const std::string value : {"true", "false", "null", ""}) {
        SCOPED_TRACE(spelling);
        SCOPED_TRACE(container);
        SCOPED_TRACE(value);
        const std::string field =
            value.empty() ? "" : "\"shaderIntegerDotProduct\":" + value;
        const std::string json =
            "{\"devices\":[{\"properties\":{\"apiVersion\":4206592},\"" +
            container +
            "\":{\"VkPhysicalDeviceVulkan12Features\":{\"shaderInt8\":true},"
            "\"VkPhysicalDevice" +
            spelling + "\":{" + field + "}}}]}";
        SourcePtr source = CreateJsonSource("dot.json", json.c_str());
        TargetProfilePtr profile =
            ImportVulkaninfoProfile(environment.get(), source.get(), nullptr);
        const loomc_target_fact_state_t expected =
            value == "true"    ? LOOMC_TARGET_FACT_STATE_TRUE
            : value == "false" ? LOOMC_TARGET_FACT_STATE_FALSE
                               : LOOMC_TARGET_FACT_STATE_UNKNOWN;
        for (loomc_spirv_feature_t feature :
             {LOOMC_SPIRV_FEATURE_INTEGER_DOT_PRODUCT,
              LOOMC_SPIRV_FEATURE_INTEGER_DOT_PRODUCT_INPUT_ALL,
              LOOMC_SPIRV_FEATURE_INTEGER_DOT_PRODUCT_INPUT4X8_BIT,
              LOOMC_SPIRV_FEATURE_INTEGER_DOT_PRODUCT_INPUT4X8_BIT_PACKED}) {
          ExpectFeatureState(profile.get(), feature, expected);
        }
      }
    }
  }
}

TEST(TargetSpirvVulkaninfoTest, PackedIntegerDotDoesNotRequireInt8) {
  TargetEnvironmentPtr environment = CreateSpirvTargetEnvironment();
  for (const std::string int8 : {"false", "null"}) {
    const std::string json =
        "{\"devices\":[{\"properties\":{\"apiVersion\":4206592},\"features\":{"
        "\"VkPhysicalDeviceVulkan12Features\":{\"shaderInt8\":" +
        int8 +
        "},"
        "\"VkPhysicalDeviceVulkan13Features\":{\"shaderIntegerDotProduct\":"
        "true}}}]}";
    SourcePtr source = CreateJsonSource("packed-dot.json", json.c_str());
    TargetProfilePtr profile =
        ImportVulkaninfoProfile(environment.get(), source.get(), nullptr);
    ExpectFeatureState(profile.get(), LOOMC_SPIRV_FEATURE_INTEGER_DOT_PRODUCT,
                       LOOMC_TARGET_FACT_STATE_TRUE);
    ExpectFeatureState(
        profile.get(),
        LOOMC_SPIRV_FEATURE_INTEGER_DOT_PRODUCT_INPUT4X8_BIT_PACKED,
        LOOMC_TARGET_FACT_STATE_TRUE);
    ExpectFeatureState(profile.get(),
                       LOOMC_SPIRV_FEATURE_INTEGER_DOT_PRODUCT_INPUT4X8_BIT,
                       int8 == "false" ? LOOMC_TARGET_FACT_STATE_FALSE
                                       : LOOMC_TARGET_FACT_STATE_UNKNOWN);
  }
}

TEST(TargetSpirvVulkaninfoTest, ReportsMalformedJsonAsResultDiagnostic) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  SourcePtr source = CreateJsonSource("bad-vulkaninfo.json", "{");
  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_target_profile_create_spirv_vulkaninfo(
      target_environment.get(), source.get(), /*options=*/nullptr,
      loomc_allocator_system(), &profile, &result);
  LOOMC_EXPECT_OK(status);
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);
  EXPECT_EQ(profile_ptr.get(), nullptr);
  ExpectFailedResultCode(result_ptr.get(), "SPIRV/VULKANINFO");
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(diagnostic->range.source, source.get());
}

TEST(TargetSpirvVulkaninfoTest, ReportsMissingSelectedProfile) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  const std::string json = R"json({
    "capabilities": {"device": {"properties": {}, "features": {}}},
    "profiles": {"VP_present": {"capabilities": ["device"]}}
  })json";
  SourcePtr source = CreateJsonSource("profiles.json", json.c_str());
  loomc_spirv_vulkaninfo_profile_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SPIRV_VULKANINFO_PROFILE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .identifier = loomc_string_view_empty(),
      .profile_name = loomc_make_cstring_view("VP_missing"),
      .device_index = 0,
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_target_profile_create_spirv_vulkaninfo(
      target_environment.get(), source.get(), &options,
      loomc_allocator_system(), &profile, &result);
  LOOMC_EXPECT_OK(status);
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);
  EXPECT_EQ(profile_ptr.get(), nullptr);
  ExpectFailedResultCode(result_ptr.get(), "SPIRV/VULKANINFO");
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("VP_missing"));
}

TEST(TargetSpirvVulkaninfoTest, ReportsMalformedLimitArray) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  const std::string json = R"json({
    "devices": [
      {
        "properties": {
          "apiVersion": 4206592,
          "limits": {
            "maxComputeWorkGroupSize": [256]
          }
        },
        "features": {}
      }
    ]
  })json";
  SourcePtr source = CreateJsonSource("short-limits.json", json.c_str());
  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_target_profile_create_spirv_vulkaninfo(
      target_environment.get(), source.get(), /*options=*/nullptr,
      loomc_allocator_system(), &profile, &result);
  LOOMC_EXPECT_OK(status);
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);
  EXPECT_EQ(profile_ptr.get(), nullptr);
  ExpectFailedResultCode(result_ptr.get(), "SPIRV/VULKANINFO");
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("maxComputeWorkGroupSize"));
}

}  // namespace
