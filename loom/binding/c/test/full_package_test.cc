// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <memory>
#include <string>

#include "iree/testing/gtest.h"
#include "loomc/context.h"
#include "loomc/diagnostic.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/target.h"
#include "loomc/target/iree_hal.h"
#include "loomc/target/spirv.h"
#include "loomc/target/spirv/iree_hal.h"
#include "loomc/target/spirv/vulkan.h"
#include "loomc/target/spirv/vulkaninfo.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;

constexpr char kVulkaninfoJson[] = R"json({
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
        "shaderInt64": true
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

TEST(LoomcFullPackageTest, LinksCoreAndSpirvTargetPackages) {
  loomc_spirv_vulkan_function_table_t vulkan_functions = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_VULKAN_FUNCTION_TABLE,
      /*.structure_size=*/sizeof(vulkan_functions),
      /*.next=*/nullptr,
      /*.get_physical_device_properties2=*/nullptr,
      /*.get_physical_device_features2=*/nullptr,
      /*.enumerate_device_extension_properties=*/nullptr,
  };
  EXPECT_EQ(vulkan_functions.type,
            LOOMC_STRUCTURE_TYPE_SPIRV_VULKAN_FUNCTION_TABLE);
  loomc_iree_hal_profile_options_t hal_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(hal_options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("full-package"),
      /*.device=*/nullptr,
      /*.executable_cache=*/nullptr,
      /*.providers=*/nullptr,
      /*.provider_count=*/0,
  };
  EXPECT_EQ(hal_options.type, LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS);
  const loomc_iree_hal_profile_provider_t* provider =
      loomc_spirv_iree_hal_profile_provider();
  ASSERT_NE(provider, nullptr);

  loomc_target_environment_t* target_environment = nullptr;
  LOOMC_ASSERT_OK(loomc_target_environment_create_spirv(
      loomc_allocator_system(), &target_environment));
  TargetEnvironmentPtr target_environment_ptr(target_environment);

  loomc_context_target_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_environment=*/target_environment_ptr.get(),
  };
  loomc_context_options_t context_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      /*.structure_size=*/sizeof(context_options),
      /*.next=*/&target_options,
  };
  loomc_context_t* context = nullptr;
  LOOMC_ASSERT_OK(loomc_context_create(&context_options,
                                       loomc_allocator_system(), &context));
  ContextPtr context_ptr(context);
  ASSERT_NE(context_ptr.get(), nullptr);

  loomc_byte_span_t source_contents =
      loomc_make_byte_span(kVulkaninfoJson, strlen(kVulkaninfoJson));
  loomc_source_options_t source_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(source_options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_UNKNOWN,
      /*.identifier=*/loomc_make_cstring_view("vulkaninfo.json"),
      /*.contents=*/source_contents,
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
      /*.release=*/nullptr,
      /*.release_user_data=*/nullptr,
  };
  loomc_source_t* source = nullptr;
  LOOMC_ASSERT_OK(
      loomc_source_create(&source_options, loomc_allocator_system(), &source));
  SourcePtr source_ptr(source);

  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_create_spirv_vulkaninfo(
      target_environment_ptr.get(), source_ptr.get(), nullptr,
      loomc_allocator_system(), &profile, &result));
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());

  loomc_spirv_limit_value_t storage_limit = {
      /*.state=*/LOOMC_TARGET_FACT_STATE_UNKNOWN,
      /*.value=*/0,
  };
  LOOMC_ASSERT_OK(loomc_spirv_target_profile_query_limit(
      profile_ptr.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES,
      &storage_limit));
  EXPECT_EQ(storage_limit.state, LOOMC_TARGET_FACT_STATE_TRUE);
  EXPECT_EQ(storage_limit.value, 32768u);
}

}  // namespace
