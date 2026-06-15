// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/spirv/profile.h"

#include <cstdint>
#include <memory>
#include <string>

#include "iree/testing/gtest.h"
#include "loomc/context.h"
#include "loomc/pass.h"
#include "loomc/result.h"
#include "loomc/status.h"
#include "loomc/target.h"
#include "loomc/target/spirv/base.h"
#include "target.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using TargetSelectionPtr =
    HandlePtr<loomc_target_selection_t, loomc_target_selection_release>;

constexpr uint32_t kSpirvVersion10 = LOOMC_SPIRV_VERSION_1_0;
constexpr uint32_t kSpirvVersion13 = LOOMC_SPIRV_VERSION_1_3;
constexpr uint32_t kSpirvAddressingModelLogical = 0;
constexpr uint32_t kSpirvAddressingModelPhysicalStorageBuffer64 = 5348;
constexpr uint32_t kSpirvMemoryModelGlsl450 = 1;
constexpr uint32_t kSpirvMemoryModelVulkan = 3;
constexpr uint32_t kSpirvCapabilityShader = 1;
constexpr uint32_t kSpirvCapabilityFloat16 = 9;
constexpr uint32_t kSpirvCapabilityFloat64 = 10;
constexpr uint32_t kSpirvCapabilityInt64 = 11;
constexpr uint32_t kSpirvCapabilityVulkanMemoryModel = 5345;
constexpr uint32_t kSpirvCapabilityPhysicalStorageBufferAddresses = 5347;
constexpr uint32_t kSpirvStorageClassPhysicalStorageBuffer = 5349;
constexpr uint32_t kSpirvDecorationRestrictPointer = 5355;
constexpr uint32_t kSpirvDecorationAliasedPointer = 5356;
constexpr const char* kF16CooperativeMatrixRow =
    "khr.cooperative_matrix.f16.16x16x16.f32.subgroup";
constexpr const char* kPackedS8CooperativeVectorRow =
    "nv.cooperative_vector.u32.32x32.s8_packed";

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

void ExpectFailedResult(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result));
  EXPECT_EQ(loomc_result_state(result), LOOMC_RESULT_STATE_FAILED);
  ASSERT_GE(loomc_result_diagnostic_count(result), 1u);
  const loomc_diagnostic_t* diagnostic = loomc_result_diagnostic_at(result, 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(diagnostic->severity, LOOMC_DIAGNOSTIC_SEVERITY_ERROR);
  EXPECT_EQ(ToString(diagnostic->code), "SPIRV/PROFILE");
}

TargetEnvironmentPtr CreateSpirvTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_spirv(
      loomc_allocator_system(), &target_environment);
  LOOMC_EXPECT_OK(status);
  return TargetEnvironmentPtr(target_environment);
}

ContextPtr CreateSpirvContext(loomc_target_environment_t* target_environment) {
  loomc_context_target_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_environment=*/target_environment,
  };
  loomc_context_options_t context_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      /*.structure_size=*/sizeof(context_options),
      /*.next=*/&target_options,
  };
  loomc_context_t* context = nullptr;
  loomc_status_t status = loomc_context_create(
      &context_options, loomc_allocator_system(), &context);
  LOOMC_EXPECT_OK(status);
  return ContextPtr(context);
}

TargetProfilePtr CreateSpirvProfile(
    loomc_target_environment_t* target_environment,
    const loomc_spirv_profile_options_t* options) {
  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_target_profile_create_spirv(
      target_environment, options, loomc_allocator_system(), &profile, &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return TargetProfilePtr(profile);
}

TargetProfilePtr RefineSpirvProfile(
    const loomc_target_profile_t* base_profile,
    const loomc_spirv_profile_options_t* options) {
  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_spirv_target_profile_refine(
      base_profile, options, loomc_allocator_system(), &profile, &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return TargetProfilePtr(profile);
}

TargetSelectionPtr CreateSelectionFromProfile(loomc_target_profile_t* profile) {
  loomc_target_selection_t* selection = nullptr;
  loomc_status_t status = loomc_target_selection_create_from_profile(
      profile, loomc_allocator_system(), &selection);
  LOOMC_EXPECT_OK(status);
  return TargetSelectionPtr(selection);
}

bool ProfileHasExtension(const loomc_target_profile_t* profile,
                         const char* expected_extension) {
  loomc_spirv_profile_info_t info = {};
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_info(profile, &info));
  for (loomc_host_size_t i = 0; i < info.extension_count; ++i) {
    loomc_string_view_t extension = loomc_string_view_empty();
    LOOMC_EXPECT_OK(
        loomc_spirv_target_profile_extension_at(profile, i, &extension));
    if (ToString(extension) == expected_extension) {
      return true;
    }
  }
  return false;
}

bool ProfileHasCapability(const loomc_target_profile_t* profile,
                          uint32_t expected_capability) {
  loomc_spirv_profile_info_t info = {};
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_info(profile, &info));
  for (loomc_host_size_t i = 0; i < info.capability_count; ++i) {
    uint32_t capability = 0;
    LOOMC_EXPECT_OK(
        loomc_spirv_target_profile_capability_at(profile, i, &capability));
    if (capability == expected_capability) {
      return true;
    }
  }
  return false;
}

bool ProfileHasStorageClass(const loomc_target_profile_t* profile,
                            uint32_t expected_storage_class) {
  loomc_spirv_profile_info_t info = {};
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_info(profile, &info));
  for (loomc_host_size_t i = 0; i < info.storage_class_count; ++i) {
    uint32_t storage_class = 0;
    LOOMC_EXPECT_OK(loomc_spirv_target_profile_storage_class_at(
        profile, i, &storage_class));
    if (storage_class == expected_storage_class) {
      return true;
    }
  }
  return false;
}

bool ProfileHasDecoration(const loomc_target_profile_t* profile,
                          uint32_t expected_decoration) {
  loomc_spirv_profile_info_t info = {};
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_info(profile, &info));
  for (loomc_host_size_t i = 0; i < info.decoration_count; ++i) {
    uint32_t decoration = 0;
    LOOMC_EXPECT_OK(
        loomc_spirv_target_profile_decoration_at(profile, i, &decoration));
    if (decoration == expected_decoration) {
      return true;
    }
  }
  return false;
}

bool FindCooperativeMatrixRow(const loomc_target_profile_t* profile,
                              const char* expected_name,
                              loomc_spirv_cooperative_matrix_row_t* out_row) {
  loomc_spirv_profile_info_t info = {};
  loomc_status_t status = loomc_spirv_target_profile_query_info(profile, &info);
  const bool query_ok = loomc_status_is_ok(status);
  LOOMC_EXPECT_OK(status);
  if (!query_ok) {
    return false;
  }
  for (loomc_host_size_t i = 0; i < info.cooperative_matrix_row_count; ++i) {
    loomc_spirv_cooperative_matrix_row_t row = {};
    status =
        loomc_spirv_target_profile_cooperative_matrix_row_at(profile, i, &row);
    const bool row_ok = loomc_status_is_ok(status);
    LOOMC_EXPECT_OK(status);
    if (!row_ok) {
      return false;
    }
    if (ToString(row.name) == expected_name) {
      *out_row = row;
      return true;
    }
  }
  return false;
}

bool FindCooperativeVectorRow(const loomc_target_profile_t* profile,
                              const char* expected_name,
                              loomc_spirv_cooperative_vector_row_t* out_row) {
  loomc_spirv_profile_info_t info = {};
  loomc_status_t status = loomc_spirv_target_profile_query_info(profile, &info);
  const bool query_ok = loomc_status_is_ok(status);
  LOOMC_EXPECT_OK(status);
  if (!query_ok) {
    return false;
  }
  for (loomc_host_size_t i = 0; i < info.cooperative_vector_row_count; ++i) {
    loomc_spirv_cooperative_vector_row_t row = {};
    status =
        loomc_spirv_target_profile_cooperative_vector_row_at(profile, i, &row);
    const bool row_ok = loomc_status_is_ok(status);
    LOOMC_EXPECT_OK(status);
    if (!row_ok) {
      return false;
    }
    if (ToString(row.name) == expected_name) {
      *out_row = row;
      return true;
    }
  }
  return false;
}

loomc_spirv_cooperative_matrix_row_t MakeCustomMatrixRow(
    loomc_string_view_t name, loomc_target_fact_state_t state,
    loomc_string_view_t provenance) {
  return {
      /*.name=*/name,
      /*.state=*/state,
      /*.provenance=*/provenance,
      /*.required_features=*/
      loomc_spirv_feature_bit(LOOMC_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR) |
          loomc_spirv_feature_bit(LOOMC_SPIRV_FEATURE_FLOAT16),
      /*.m_size=*/8,
      /*.n_size=*/8,
      /*.k_size=*/16,
      /*.lhs_type=*/LOOMC_SPIRV_SCALAR_TYPE_F16,
      /*.rhs_type=*/LOOMC_SPIRV_SCALAR_TYPE_F16,
      /*.accumulator_type=*/LOOMC_SPIRV_SCALAR_TYPE_F32,
      /*.result_type=*/LOOMC_SPIRV_SCALAR_TYPE_F32,
      /*.scope=*/LOOMC_SPIRV_SCOPE_SUBGROUP,
      /*.layout_flags=*/LOOMC_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_BIT,
      /*.storage_class_flags=*/LOOMC_SPIRV_STORAGE_CLASS_BIT_STORAGE_BUFFER,
      /*.operand_flags=*/0,
  };
}

loomc_spirv_cooperative_vector_row_t MakeCustomVectorRow(
    loomc_string_view_t name, loomc_target_fact_state_t state,
    loomc_string_view_t provenance) {
  return {
      /*.name=*/name,
      /*.state=*/state,
      /*.provenance=*/provenance,
      /*.required_features=*/
      loomc_spirv_feature_bit(LOOMC_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV),
      /*.m_size=*/64,
      /*.k_size=*/32,
      /*.input_type=*/LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT32_NV,
      /*.input_interpretation=*/
      LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT8_PACKED_NV,
      /*.matrix_interpretation=*/LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT8_NV,
      /*.bias_interpretation=*/LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV,
      /*.result_type=*/LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV,
      /*.matrix_layout_flags=*/
      LOOMC_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_BIT,
      /*.storage_class_flags=*/
      LOOMC_SPIRV_STORAGE_CLASS_BIT_PHYSICAL_STORAGE_BUFFER,
      /*.flags=*/0,
  };
}

void ExpectLimitValue(const loomc_target_profile_t* profile,
                      loomc_spirv_limit_t limit,
                      loomc_target_fact_state_t expected_state,
                      uint64_t expected_value) {
  loomc_spirv_limit_value_t value = {
      /*.state=*/LOOMC_TARGET_FACT_STATE_UNKNOWN,
      /*.value=*/0,
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
      /*.state=*/LOOMC_TARGET_FACT_STATE_UNKNOWN,
      /*.value=*/0,
  };
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_environment(
      profile, environment, &value));
  EXPECT_EQ(value.state, expected_state);
  EXPECT_EQ(value.value, expected_value);
}

void ExpectPartialProfileSelection(const loomc_target_profile_t* profile) {
  const loom_target_selection_t selection =
      loomc_target_profile_loom_target_selection(profile);
  EXPECT_EQ(selection.bundle, nullptr);
  EXPECT_NE(selection.data, nullptr);
}

void ExpectVulkanBdaProfileBundle(const loomc_target_profile_t* profile) {
  const loom_target_selection_t selection =
      loomc_target_profile_loom_target_selection(profile);
  ASSERT_NE(selection.bundle, nullptr);
  ASSERT_NE(selection.bundle->snapshot, nullptr);
  ASSERT_NE(selection.bundle->config, nullptr);
  EXPECT_EQ(selection.bundle->snapshot->codegen_format,
            LOOM_TARGET_CODEGEN_FORMAT_SPIRV);
  EXPECT_EQ(selection.bundle->snapshot->artifact_format,
            LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY);
  const loomc_spirv_feature_bits_t expected_features =
      loomc_spirv_feature_bit(LOOMC_SPIRV_FEATURE_VULKAN_SHADER) |
      loomc_spirv_feature_bit(LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER) |
      loomc_spirv_feature_bit(LOOMC_SPIRV_FEATURE_INT64);
  EXPECT_EQ(selection.bundle->config->contract_feature_bits, expected_features);
  EXPECT_NE(selection.data, nullptr);
}

TEST(TargetSpirvProfileTest, CreatesEmptyPartialProfile) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  TargetProfilePtr profile = CreateSpirvProfile(target_environment.get(),
                                                /*options=*/nullptr);
  ExpectPartialProfileSelection(profile.get());

  loomc_target_fact_state_t state = LOOMC_TARGET_FACT_STATE_TRUE;
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_feature(
      profile.get(), LOOMC_SPIRV_FEATURE_VULKAN_SHADER, &state));
  EXPECT_EQ(state, LOOMC_TARGET_FACT_STATE_UNKNOWN);

  loomc_spirv_profile_info_t info = {};
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_info(profile.get(), &info));
  EXPECT_EQ(info.minimum_spirv_version, 0u);
  EXPECT_EQ(info.addressing_model, kSpirvAddressingModelLogical);
  EXPECT_EQ(info.memory_model, kSpirvMemoryModelGlsl450);
  EXPECT_EQ(info.extension_count, 0u);
  EXPECT_EQ(info.capability_count, 0u);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X,
                   LOOMC_TARGET_FACT_STATE_UNKNOWN, 0);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE,
                   LOOMC_TARGET_FACT_STATE_UNKNOWN, 0);
  ExpectEnvironmentValue(profile.get(),
                         LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION,
                         LOOMC_TARGET_FACT_STATE_UNKNOWN, 0);
}

TEST(TargetSpirvProfileTest, PreservesExplicitNumericLimitFacts) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_limit_fact_t limits[] = {
      {
          /*.limit=*/LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/1024,
          /*.provenance=*/
          loomc_make_cstring_view("vulkaninfo:maxComputeWorkGroupSize[0]"),
      },
      {
          /*.limit=*/LOOMC_SPIRV_LIMIT_MAX_FLAT_WORKGROUP_SIZE,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/1024,
          /*.provenance=*/
          loomc_make_cstring_view("vulkaninfo:maxComputeWorkGroupInvocations"),
      },
      {
          /*.limit=*/LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/UINT64_C(49152),
          /*.provenance=*/
          loomc_make_cstring_view("vulkaninfo:maxComputeSharedMemorySize"),
      },
      {
          /*.limit=*/LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/32,
          /*.provenance=*/loomc_make_cstring_view("vulkaninfo:subgroupSize"),
      },
      {
          /*.limit=*/LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Z,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/65535,
          /*.provenance=*/
          loomc_make_cstring_view("vulkaninfo:maxComputeWorkGroupCount[2]"),
      },
      {
          /*.limit=*/LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_X,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/4096,
          /*.provenance=*/loomc_make_cstring_view("profile:maxGridSize[0]"),
      },
      {
          /*.limit=*/LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_Y,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/2048,
          /*.provenance=*/loomc_make_cstring_view("profile:maxGridSize[1]"),
      },
      {
          /*.limit=*/LOOMC_SPIRV_LIMIT_MAX_FLAT_GRID_SIZE,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/UINT64_C(0x100000000),
          /*.provenance=*/loomc_make_cstring_view("profile:maxFlatGridSize"),
      },
  };
  loomc_spirv_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("offline-vulkan13-limits"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
      /*.feature_facts=*/nullptr,
      /*.feature_fact_count=*/0,
      /*.limit_facts=*/limits,
      /*.limit_fact_count=*/8,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
  };
  TargetProfilePtr profile =
      CreateSpirvProfile(target_environment.get(), &options);
  ExpectVulkanBdaProfileBundle(profile.get());
  const loom_target_selection_t selection =
      loomc_target_profile_loom_target_selection(profile.get());
  EXPECT_EQ(selection.bundle->snapshot->max_workgroup_size.x, 1024u);
  EXPECT_EQ(selection.bundle->snapshot->max_workgroup_size.y, 0u);
  EXPECT_EQ(selection.bundle->snapshot->max_flat_workgroup_size, 1024u);
  EXPECT_EQ(selection.bundle->snapshot->max_workgroup_storage_bytes,
            UINT64_C(49152));
  EXPECT_EQ(selection.bundle->snapshot->subgroup_size, 32u);
  EXPECT_EQ(selection.bundle->snapshot->max_workgroup_count.z, 65535u);
  EXPECT_EQ(selection.bundle->snapshot->max_grid_size.x, 4096u);
  EXPECT_EQ(selection.bundle->snapshot->max_grid_size.y, 2048u);
  EXPECT_EQ(selection.bundle->snapshot->max_grid_size.z, 0u);
  EXPECT_EQ(selection.bundle->snapshot->max_flat_grid_size,
            UINT64_C(0x100000000));

  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X,
                   LOOMC_TARGET_FACT_STATE_TRUE, 1024);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_Y,
                   LOOMC_TARGET_FACT_STATE_UNKNOWN, 0);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_FLAT_WORKGROUP_SIZE,
                   LOOMC_TARGET_FACT_STATE_TRUE, 1024);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES,
                   LOOMC_TARGET_FACT_STATE_TRUE, UINT64_C(49152));
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE,
                   LOOMC_TARGET_FACT_STATE_TRUE, 32);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_Z,
                   LOOMC_TARGET_FACT_STATE_TRUE, 65535);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_X,
                   LOOMC_TARGET_FACT_STATE_TRUE, 4096);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_GRID_SIZE_Z,
                   LOOMC_TARGET_FACT_STATE_UNKNOWN, 0);
  ExpectLimitValue(profile.get(), LOOMC_SPIRV_LIMIT_MAX_FLAT_GRID_SIZE,
                   LOOMC_TARGET_FACT_STATE_TRUE, UINT64_C(0x100000000));
}

TEST(TargetSpirvProfileTest, PreservesExplicitEnvironmentFacts) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_environment_fact_t environment[] = {
      {
          /*.environment=*/LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/kSpirvVersion13,
          /*.provenance=*/loomc_make_cstring_view("vulkaninfo:apiVersion"),
      },
  };
  loomc_spirv_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("offline-vulkan13-environment"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
      /*.feature_facts=*/nullptr,
      /*.feature_fact_count=*/0,
      /*.limit_facts=*/nullptr,
      /*.limit_fact_count=*/0,
      /*.environment_facts=*/environment,
      /*.environment_fact_count=*/1,
  };
  TargetProfilePtr profile =
      CreateSpirvProfile(target_environment.get(), &options);
  ExpectVulkanBdaProfileBundle(profile.get());
  ExpectEnvironmentValue(profile.get(),
                         LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION,
                         LOOMC_TARGET_FACT_STATE_TRUE, kSpirvVersion13);
}

TEST(TargetSpirvProfileTest, RefinesProfileWithAdditionalFacts) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_limit_fact_t base_limits[] = {
      {
          /*.limit=*/LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/32,
          /*.provenance=*/loomc_make_cstring_view("base:subgroup"),
      },
  };
  loomc_spirv_profile_options_t base_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(base_options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("base-profile"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_NONE,
      /*.feature_facts=*/nullptr,
      /*.feature_fact_count=*/0,
      /*.limit_facts=*/base_limits,
      /*.limit_fact_count=*/1,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
  };
  TargetProfilePtr base_profile =
      CreateSpirvProfile(target_environment.get(), &base_options);
  ExpectPartialProfileSelection(base_profile.get());
  ExpectLimitValue(base_profile.get(), LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE,
                   LOOMC_TARGET_FACT_STATE_TRUE, 32);

  loomc_spirv_environment_fact_t environment[] = {
      {
          /*.environment=*/LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/kSpirvVersion13,
          /*.provenance=*/loomc_make_cstring_view("refine:apiVersion"),
      },
  };
  loomc_spirv_profile_options_t refine_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(refine_options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("refined-profile"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
      /*.feature_facts=*/nullptr,
      /*.feature_fact_count=*/0,
      /*.limit_facts=*/nullptr,
      /*.limit_fact_count=*/0,
      /*.environment_facts=*/environment,
      /*.environment_fact_count=*/1,
  };
  TargetProfilePtr refined_profile =
      RefineSpirvProfile(base_profile.get(), &refine_options);
  ExpectVulkanBdaProfileBundle(refined_profile.get());
  const loom_target_selection_t selection =
      loomc_target_profile_loom_target_selection(refined_profile.get());
  EXPECT_EQ(selection.bundle->snapshot->subgroup_size, 32u);
  ExpectLimitValue(refined_profile.get(), LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE,
                   LOOMC_TARGET_FACT_STATE_TRUE, 32);
  ExpectEnvironmentValue(refined_profile.get(),
                         LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION,
                         LOOMC_TARGET_FACT_STATE_TRUE, kSpirvVersion13);

  ExpectPartialProfileSelection(base_profile.get());
  ExpectEnvironmentValue(base_profile.get(),
                         LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION,
                         LOOMC_TARGET_FACT_STATE_UNKNOWN, 0);
}

TEST(TargetSpirvProfileTest, RefinementCanCloneProfile) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_limit_fact_t limits[] = {
      {
          /*.limit=*/LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/256,
          /*.provenance=*/loomc_make_cstring_view("base:workgroup-x"),
      },
  };
  loomc_spirv_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("clone-source"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_NONE,
      /*.feature_facts=*/nullptr,
      /*.feature_fact_count=*/0,
      /*.limit_facts=*/limits,
      /*.limit_fact_count=*/1,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
  };
  TargetProfilePtr base_profile =
      CreateSpirvProfile(target_environment.get(), &options);
  TargetProfilePtr cloned_profile =
      RefineSpirvProfile(base_profile.get(), /*options=*/nullptr);
  ExpectPartialProfileSelection(cloned_profile.get());
  ExpectLimitValue(cloned_profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X,
                   LOOMC_TARGET_FACT_STATE_TRUE, 256);
}

TEST(TargetSpirvProfileTest, CreatesPresetProfileAndQueriesRows) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("offline-vulkan13"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
      /*.feature_facts=*/nullptr,
      /*.feature_fact_count=*/0,
      /*.limit_facts=*/nullptr,
      /*.limit_fact_count=*/0,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
  };
  TargetProfilePtr profile =
      CreateSpirvProfile(target_environment.get(), &options);
  ExpectVulkanBdaProfileBundle(profile.get());

  loomc_target_fact_state_t state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_feature(
      profile.get(), LOOMC_SPIRV_FEATURE_VULKAN_SHADER, &state));
  EXPECT_EQ(state, LOOMC_TARGET_FACT_STATE_TRUE);
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_feature(
      profile.get(), LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER, &state));
  EXPECT_EQ(state, LOOMC_TARGET_FACT_STATE_TRUE);
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_feature(
      profile.get(), LOOMC_SPIRV_FEATURE_INT64, &state));
  EXPECT_EQ(state, LOOMC_TARGET_FACT_STATE_TRUE);
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_feature(
      profile.get(), LOOMC_SPIRV_FEATURE_FLOAT16, &state));
  EXPECT_EQ(state, LOOMC_TARGET_FACT_STATE_UNKNOWN);

  loomc_spirv_profile_info_t info = {};
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_info(profile.get(), &info));
  EXPECT_EQ(info.minimum_spirv_version, kSpirvVersion13);
  EXPECT_EQ(info.addressing_model,
            kSpirvAddressingModelPhysicalStorageBuffer64);
  EXPECT_EQ(info.memory_model, kSpirvMemoryModelVulkan);
  EXPECT_TRUE(
      ProfileHasExtension(profile.get(), "SPV_KHR_vulkan_memory_model"));
  EXPECT_TRUE(
      ProfileHasExtension(profile.get(), "SPV_KHR_physical_storage_buffer"));
  EXPECT_TRUE(ProfileHasCapability(profile.get(), kSpirvCapabilityShader));
  EXPECT_TRUE(
      ProfileHasCapability(profile.get(), kSpirvCapabilityVulkanMemoryModel));
  EXPECT_TRUE(ProfileHasCapability(
      profile.get(), kSpirvCapabilityPhysicalStorageBufferAddresses));
  EXPECT_TRUE(ProfileHasCapability(profile.get(), kSpirvCapabilityInt64));
  EXPECT_TRUE(ProfileHasStorageClass(profile.get(),
                                     kSpirvStorageClassPhysicalStorageBuffer));
  EXPECT_TRUE(
      ProfileHasDecoration(profile.get(), kSpirvDecorationRestrictPointer));
  EXPECT_TRUE(
      ProfileHasDecoration(profile.get(), kSpirvDecorationAliasedPointer));

  loomc_string_view_t extension = loomc_string_view_empty();
  loomc_status_t extension_status = loomc_spirv_target_profile_extension_at(
      profile.get(), info.extension_count, &extension);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_OUT_OF_RANGE, extension_status);
}

TEST(TargetSpirvProfileTest, RefinesPresetWithExplicitTrueFact) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_feature_fact_t facts[] = {
      {
          /*.feature=*/LOOMC_SPIRV_FEATURE_FLOAT16,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.provenance=*/
          loomc_make_cstring_view("vulkaninfo:shaderFloat16"),
      },
  };
  loomc_spirv_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("offline-vulkan13-f16"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
      /*.feature_facts=*/facts,
      /*.feature_fact_count=*/1,
      /*.limit_facts=*/nullptr,
      /*.limit_fact_count=*/0,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
  };
  TargetProfilePtr profile =
      CreateSpirvProfile(target_environment.get(), &options);
  const loom_target_selection_t selection =
      loomc_target_profile_loom_target_selection(profile.get());
  ASSERT_NE(selection.bundle, nullptr);
  EXPECT_NE(selection.bundle->config->contract_feature_bits &
                loomc_spirv_feature_bit(LOOMC_SPIRV_FEATURE_FLOAT16),
            0u);

  loomc_target_fact_state_t state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_feature(
      profile.get(), LOOMC_SPIRV_FEATURE_FLOAT16, &state));
  EXPECT_EQ(state, LOOMC_TARGET_FACT_STATE_TRUE);
  EXPECT_TRUE(ProfileHasCapability(profile.get(), kSpirvCapabilityFloat16));
}

TEST(TargetSpirvProfileTest, QueriesCooperativePropertyRows) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_feature_fact_t facts[] = {
      {
          /*.feature=*/LOOMC_SPIRV_FEATURE_FLOAT16,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.provenance=*/
          loomc_make_cstring_view("vulkaninfo:shaderFloat16"),
      },
      {
          /*.feature=*/LOOMC_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.provenance=*/
          loomc_make_cstring_view("vulkaninfo:cooperativeMatrix"),
      },
      {
          /*.feature=*/LOOMC_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.provenance=*/
          loomc_make_cstring_view("vulkaninfo:cooperativeVectorNV"),
      },
  };
  loomc_spirv_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("offline-vulkan13-coop"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
      /*.feature_facts=*/facts,
      /*.feature_fact_count=*/3,
      /*.limit_facts=*/nullptr,
      /*.limit_fact_count=*/0,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
  };
  TargetProfilePtr profile =
      CreateSpirvProfile(target_environment.get(), &options);

  loomc_spirv_profile_info_t info = {};
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_info(profile.get(), &info));
  EXPECT_GE(info.cooperative_matrix_row_count, 1u);
  EXPECT_GE(info.cooperative_vector_row_count, 1u);

  loomc_spirv_cooperative_matrix_row_t matrix_row = {};
  ASSERT_TRUE(FindCooperativeMatrixRow(profile.get(), kF16CooperativeMatrixRow,
                                       &matrix_row));
  EXPECT_EQ(matrix_row.state, LOOMC_TARGET_FACT_STATE_TRUE);
  EXPECT_TRUE(loomc_string_view_is_empty(matrix_row.provenance));
  EXPECT_EQ(
      matrix_row.required_features,
      loomc_spirv_feature_bit(LOOMC_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR) |
          loomc_spirv_feature_bit(LOOMC_SPIRV_FEATURE_FLOAT16));
  EXPECT_EQ(matrix_row.m_size, 16u);
  EXPECT_EQ(matrix_row.n_size, 16u);
  EXPECT_EQ(matrix_row.k_size, 16u);
  EXPECT_EQ(matrix_row.lhs_type, LOOMC_SPIRV_SCALAR_TYPE_F16);
  EXPECT_EQ(matrix_row.rhs_type, LOOMC_SPIRV_SCALAR_TYPE_F16);
  EXPECT_EQ(matrix_row.accumulator_type, LOOMC_SPIRV_SCALAR_TYPE_F32);
  EXPECT_EQ(matrix_row.result_type, LOOMC_SPIRV_SCALAR_TYPE_F32);
  EXPECT_EQ(matrix_row.scope, LOOMC_SPIRV_SCOPE_SUBGROUP);
  EXPECT_NE(matrix_row.layout_flags &
                LOOMC_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_BIT,
            0u);
  EXPECT_NE(matrix_row.layout_flags &
                LOOMC_SPIRV_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_BIT,
            0u);
  EXPECT_NE(matrix_row.storage_class_flags &
                LOOMC_SPIRV_STORAGE_CLASS_BIT_PHYSICAL_STORAGE_BUFFER,
            0u);
  EXPECT_EQ(matrix_row.operand_flags, 0u);

  loomc_spirv_cooperative_vector_row_t vector_row = {};
  ASSERT_TRUE(FindCooperativeVectorRow(
      profile.get(), kPackedS8CooperativeVectorRow, &vector_row));
  EXPECT_EQ(vector_row.state, LOOMC_TARGET_FACT_STATE_TRUE);
  EXPECT_TRUE(loomc_string_view_is_empty(vector_row.provenance));
  EXPECT_EQ(vector_row.required_features,
            loomc_spirv_feature_bit(LOOMC_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV));
  EXPECT_EQ(vector_row.m_size, 32u);
  EXPECT_EQ(vector_row.k_size, 32u);
  EXPECT_EQ(vector_row.input_type,
            LOOMC_SPIRV_COMPONENT_TYPE_UNSIGNED_INT32_NV);
  EXPECT_EQ(vector_row.input_interpretation,
            LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT8_PACKED_NV);
  EXPECT_EQ(vector_row.matrix_interpretation,
            LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT8_NV);
  EXPECT_EQ(vector_row.bias_interpretation,
            LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV);
  EXPECT_EQ(vector_row.result_type, LOOMC_SPIRV_COMPONENT_TYPE_SIGNED_INT32_NV);
  EXPECT_NE(
      vector_row.matrix_layout_flags &
          LOOMC_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_BIT,
      0u);
  EXPECT_NE(vector_row.storage_class_flags &
                LOOMC_SPIRV_STORAGE_CLASS_BIT_PHYSICAL_STORAGE_BUFFER,
            0u);
  EXPECT_EQ(vector_row.flags, 0u);

  loomc_spirv_cooperative_matrix_row_t out_of_range_matrix_row = {};
  loomc_status_t matrix_status =
      loomc_spirv_target_profile_cooperative_matrix_row_at(
          profile.get(), info.cooperative_matrix_row_count,
          &out_of_range_matrix_row);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_OUT_OF_RANGE, matrix_status);

  loomc_spirv_cooperative_vector_row_t out_of_range_vector_row = {};
  loomc_status_t vector_status =
      loomc_spirv_target_profile_cooperative_vector_row_at(
          profile.get(), info.cooperative_vector_row_count,
          &out_of_range_vector_row);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_OUT_OF_RANGE, vector_status);
}

TEST(TargetSpirvProfileTest, ExplicitCooperativeRowsAnnotateModelRows) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_feature_fact_t facts[] = {
      {
          /*.feature=*/LOOMC_SPIRV_FEATURE_FLOAT16,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.provenance=*/loomc_make_cstring_view("probe:float16"),
      },
      {
          /*.feature=*/LOOMC_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.provenance=*/loomc_make_cstring_view("probe:matrix"),
      },
  };
  loomc_spirv_profile_options_t model_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(model_options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("cooperative-row-model"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
      /*.feature_facts=*/facts,
      /*.feature_fact_count=*/2,
      /*.limit_facts=*/nullptr,
      /*.limit_fact_count=*/0,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
      /*.cooperative_matrix_rows=*/nullptr,
      /*.cooperative_matrix_row_count=*/0,
      /*.cooperative_vector_rows=*/nullptr,
      /*.cooperative_vector_row_count=*/0,
  };
  TargetProfilePtr model_profile =
      CreateSpirvProfile(target_environment.get(), &model_options);
  loomc_spirv_profile_info_t model_info = {};
  LOOMC_EXPECT_OK(
      loomc_spirv_target_profile_query_info(model_profile.get(), &model_info));

  loomc_spirv_cooperative_matrix_row_t explicit_row = {};
  ASSERT_TRUE(FindCooperativeMatrixRow(
      model_profile.get(), kF16CooperativeMatrixRow, &explicit_row));
  explicit_row.provenance = loomc_make_cstring_view("vulkaninfo:row[0]");

  loomc_spirv_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("cooperative-row-annotated"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
      /*.feature_facts=*/facts,
      /*.feature_fact_count=*/2,
      /*.limit_facts=*/nullptr,
      /*.limit_fact_count=*/0,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
      /*.cooperative_matrix_rows=*/&explicit_row,
      /*.cooperative_matrix_row_count=*/1,
      /*.cooperative_vector_rows=*/nullptr,
      /*.cooperative_vector_row_count=*/0,
  };
  TargetProfilePtr profile =
      CreateSpirvProfile(target_environment.get(), &options);
  loomc_spirv_profile_info_t info = {};
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_info(profile.get(), &info));
  EXPECT_EQ(info.cooperative_matrix_row_count,
            model_info.cooperative_matrix_row_count);

  loomc_spirv_cooperative_matrix_row_t matrix_row = {};
  ASSERT_TRUE(FindCooperativeMatrixRow(profile.get(), kF16CooperativeMatrixRow,
                                       &matrix_row));
  EXPECT_EQ(matrix_row.state, LOOMC_TARGET_FACT_STATE_TRUE);
  EXPECT_EQ(ToString(matrix_row.provenance), "vulkaninfo:row[0]");
}

TEST(TargetSpirvProfileTest, AppliesExplicitCooperativeRowFacts) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_feature_fact_t facts[] = {
      {
          /*.feature=*/LOOMC_SPIRV_FEATURE_FLOAT16,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.provenance=*/loomc_make_cstring_view("probe:float16"),
      },
      {
          /*.feature=*/LOOMC_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.provenance=*/loomc_make_cstring_view("probe:matrix"),
      },
      {
          /*.feature=*/LOOMC_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.provenance=*/loomc_make_cstring_view("probe:vector"),
      },
  };
  loomc_spirv_cooperative_matrix_row_t matrix_rows[] = {
      MakeCustomMatrixRow(
          loomc_make_cstring_view("probe.matrix.f16.8x8x16.f32.subgroup"),
          LOOMC_TARGET_FACT_STATE_TRUE,
          loomc_make_cstring_view("vulkaninfo:matrix-row")),
  };
  loomc_spirv_cooperative_vector_row_t vector_rows[] = {
      MakeCustomVectorRow(
          loomc_make_cstring_view("probe.vector.u32.64x32.s8_packed"),
          LOOMC_TARGET_FACT_STATE_TRUE,
          loomc_make_cstring_view("vulkaninfo:vector-row")),
  };
  loomc_spirv_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("explicit-cooperative-rows"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
      /*.feature_facts=*/facts,
      /*.feature_fact_count=*/3,
      /*.limit_facts=*/nullptr,
      /*.limit_fact_count=*/0,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
      /*.cooperative_matrix_rows=*/matrix_rows,
      /*.cooperative_matrix_row_count=*/1,
      /*.cooperative_vector_rows=*/vector_rows,
      /*.cooperative_vector_row_count=*/1,
  };
  TargetProfilePtr profile =
      CreateSpirvProfile(target_environment.get(), &options);

  loomc_spirv_cooperative_matrix_row_t matrix_row = {};
  ASSERT_TRUE(FindCooperativeMatrixRow(
      profile.get(), "probe.matrix.f16.8x8x16.f32.subgroup", &matrix_row));
  EXPECT_EQ(matrix_row.state, LOOMC_TARGET_FACT_STATE_TRUE);
  EXPECT_EQ(ToString(matrix_row.provenance), "vulkaninfo:matrix-row");
  EXPECT_EQ(matrix_row.m_size, 8u);
  EXPECT_EQ(matrix_row.n_size, 8u);
  EXPECT_EQ(matrix_row.k_size, 16u);
  EXPECT_EQ(matrix_row.layout_flags,
            LOOMC_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_BIT);
  EXPECT_EQ(matrix_row.storage_class_flags,
            LOOMC_SPIRV_STORAGE_CLASS_BIT_STORAGE_BUFFER);

  loomc_spirv_cooperative_vector_row_t vector_row = {};
  ASSERT_TRUE(FindCooperativeVectorRow(
      profile.get(), "probe.vector.u32.64x32.s8_packed", &vector_row));
  EXPECT_EQ(vector_row.state, LOOMC_TARGET_FACT_STATE_TRUE);
  EXPECT_EQ(ToString(vector_row.provenance), "vulkaninfo:vector-row");
  EXPECT_EQ(vector_row.m_size, 64u);
  EXPECT_EQ(vector_row.k_size, 32u);
  EXPECT_EQ(
      vector_row.matrix_layout_flags,
      LOOMC_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_BIT);
  EXPECT_EQ(vector_row.storage_class_flags,
            LOOMC_SPIRV_STORAGE_CLASS_BIT_PHYSICAL_STORAGE_BUFFER);
}

TEST(TargetSpirvProfileTest, UnavailableCooperativeRowsSuppressModelRows) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_feature_fact_t facts[] = {
      {
          /*.feature=*/LOOMC_SPIRV_FEATURE_FLOAT16,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.provenance=*/loomc_make_cstring_view("probe:float16"),
      },
      {
          /*.feature=*/LOOMC_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.provenance=*/loomc_make_cstring_view("probe:matrix"),
      },
  };
  loomc_spirv_profile_options_t model_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(model_options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("cooperative-row-model"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
      /*.feature_facts=*/facts,
      /*.feature_fact_count=*/2,
      /*.limit_facts=*/nullptr,
      /*.limit_fact_count=*/0,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
      /*.cooperative_matrix_rows=*/nullptr,
      /*.cooperative_matrix_row_count=*/0,
      /*.cooperative_vector_rows=*/nullptr,
      /*.cooperative_vector_row_count=*/0,
  };
  TargetProfilePtr model_profile =
      CreateSpirvProfile(target_environment.get(), &model_options);
  loomc_spirv_cooperative_matrix_row_t unavailable_row = {};
  ASSERT_TRUE(FindCooperativeMatrixRow(
      model_profile.get(), kF16CooperativeMatrixRow, &unavailable_row));
  unavailable_row.state = LOOMC_TARGET_FACT_STATE_FALSE;
  unavailable_row.provenance = loomc_make_cstring_view("override:no-f16-row");

  loomc_spirv_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("cooperative-row-unavailable"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
      /*.feature_facts=*/facts,
      /*.feature_fact_count=*/2,
      /*.limit_facts=*/nullptr,
      /*.limit_fact_count=*/0,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
      /*.cooperative_matrix_rows=*/&unavailable_row,
      /*.cooperative_matrix_row_count=*/1,
      /*.cooperative_vector_rows=*/nullptr,
      /*.cooperative_vector_row_count=*/0,
  };
  TargetProfilePtr profile =
      CreateSpirvProfile(target_environment.get(), &options);

  loomc_spirv_cooperative_matrix_row_t matrix_row = {};
  ASSERT_TRUE(FindCooperativeMatrixRow(profile.get(), kF16CooperativeMatrixRow,
                                       &matrix_row));
  EXPECT_EQ(matrix_row.state, LOOMC_TARGET_FACT_STATE_FALSE);
  EXPECT_EQ(ToString(matrix_row.provenance), "override:no-f16-row");
}

TEST(TargetSpirvProfileTest,
     ReportsContradictoryCooperativeRowsWithProvenance) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_cooperative_matrix_row_t matrix_rows[] = {
      MakeCustomMatrixRow(loomc_make_cstring_view("probe.matrix.conflict"),
                          LOOMC_TARGET_FACT_STATE_TRUE,
                          loomc_make_cstring_view("probe:matrix-true")),
      MakeCustomMatrixRow(loomc_make_cstring_view("probe.matrix.conflict"),
                          LOOMC_TARGET_FACT_STATE_FALSE,
                          loomc_make_cstring_view("override:matrix-false")),
  };
  loomc_spirv_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("contradictory-matrix-rows"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_NONE,
      /*.feature_facts=*/nullptr,
      /*.feature_fact_count=*/0,
      /*.limit_facts=*/nullptr,
      /*.limit_fact_count=*/0,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
      /*.cooperative_matrix_rows=*/matrix_rows,
      /*.cooperative_matrix_row_count=*/2,
      /*.cooperative_vector_rows=*/nullptr,
      /*.cooperative_vector_row_count=*/0,
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_target_profile_create_spirv(
      target_environment.get(), &options, loomc_allocator_system(), &profile,
      &result);
  LOOMC_EXPECT_OK(status);
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);
  EXPECT_EQ(profile_ptr.get(), nullptr);
  ExpectFailedResult(result_ptr.get());
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("contradictory cooperative matrix row"));
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("probe:matrix-true"));
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("override:matrix-false"));
}

TEST(TargetSpirvProfileTest, RefinesCooperativeRowsWithOwnedProvenance) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_feature_fact_t facts[] = {
      {
          /*.feature=*/LOOMC_SPIRV_FEATURE_FLOAT16,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.provenance=*/loomc_make_cstring_view("probe:float16"),
      },
      {
          /*.feature=*/LOOMC_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.provenance=*/loomc_make_cstring_view("probe:matrix"),
      },
  };
  std::string provenance = "base:matrix-row";
  loomc_spirv_cooperative_matrix_row_t matrix_rows[] = {
      MakeCustomMatrixRow(
          loomc_make_cstring_view("probe.matrix.refine"),
          LOOMC_TARGET_FACT_STATE_TRUE,
          loomc_make_string_view(provenance.data(), provenance.size())),
  };
  loomc_spirv_profile_options_t base_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(base_options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("base-cooperative-row"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
      /*.feature_facts=*/facts,
      /*.feature_fact_count=*/2,
      /*.limit_facts=*/nullptr,
      /*.limit_fact_count=*/0,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
      /*.cooperative_matrix_rows=*/matrix_rows,
      /*.cooperative_matrix_row_count=*/1,
      /*.cooperative_vector_rows=*/nullptr,
      /*.cooperative_vector_row_count=*/0,
  };
  TargetProfilePtr base_profile =
      CreateSpirvProfile(target_environment.get(), &base_options);
  for (char& character : provenance) {
    character = 'x';
  }
  TargetProfilePtr refined_profile =
      RefineSpirvProfile(base_profile.get(), /*options=*/nullptr);

  loomc_spirv_cooperative_matrix_row_t matrix_row = {};
  ASSERT_TRUE(FindCooperativeMatrixRow(refined_profile.get(),
                                       "probe.matrix.refine", &matrix_row));
  EXPECT_EQ(matrix_row.state, LOOMC_TARGET_FACT_STATE_TRUE);
  EXPECT_EQ(ToString(matrix_row.provenance), "base:matrix-row");
}

TEST(TargetSpirvProfileTest, PreservesKnownFalseFeatureFacts) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_feature_fact_t facts[] = {
      {
          /*.feature=*/LOOMC_SPIRV_FEATURE_FLOAT64,
          /*.state=*/LOOMC_TARGET_FACT_STATE_FALSE,
          /*.provenance=*/
          loomc_make_cstring_view("vulkaninfo:shaderFloat64"),
      },
  };
  loomc_spirv_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("partial-no-f64"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_NONE,
      /*.feature_facts=*/facts,
      /*.feature_fact_count=*/1,
      /*.limit_facts=*/nullptr,
      /*.limit_fact_count=*/0,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
  };
  TargetProfilePtr profile =
      CreateSpirvProfile(target_environment.get(), &options);

  loomc_target_fact_state_t state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_feature(
      profile.get(), LOOMC_SPIRV_FEATURE_FLOAT64, &state));
  EXPECT_EQ(state, LOOMC_TARGET_FACT_STATE_FALSE);
  EXPECT_FALSE(ProfileHasCapability(profile.get(), kSpirvCapabilityFloat64));
}

TEST(TargetSpirvProfileTest, ReportsContradictoryFactsWithProvenance) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_feature_fact_t facts[] = {
      {
          /*.feature=*/LOOMC_SPIRV_FEATURE_FLOAT16,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.provenance=*/loomc_make_cstring_view("probe:a"),
      },
      {
          /*.feature=*/LOOMC_SPIRV_FEATURE_FLOAT16,
          /*.state=*/LOOMC_TARGET_FACT_STATE_FALSE,
          /*.provenance=*/loomc_make_cstring_view("override:b"),
      },
  };
  loomc_spirv_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("contradiction"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_NONE,
      /*.feature_facts=*/facts,
      /*.feature_fact_count=*/2,
      /*.limit_facts=*/nullptr,
      /*.limit_fact_count=*/0,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_target_profile_create_spirv(
      target_environment.get(), &options, loomc_allocator_system(), &profile,
      &result);
  LOOMC_EXPECT_OK(status);
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);
  EXPECT_EQ(profile_ptr.get(), nullptr);
  ExpectFailedResult(result_ptr.get());
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("contradictory facts"));
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("spirv.float16"));
  EXPECT_THAT(ToString(diagnostic->message), ::testing::HasSubstr("probe:a"));
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("override:b"));
}

TEST(TargetSpirvProfileTest, ReportsContradictoryLimitFactsWithProvenance) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_limit_fact_t limits[] = {
      {
          /*.limit=*/LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/UINT64_C(49152),
          /*.provenance=*/loomc_make_cstring_view("probe:a"),
      },
      {
          /*.limit=*/LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_STORAGE_BYTES,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/UINT64_C(32768),
          /*.provenance=*/loomc_make_cstring_view("override:b"),
      },
  };
  loomc_spirv_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("contradictory-limits"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_NONE,
      /*.feature_facts=*/nullptr,
      /*.feature_fact_count=*/0,
      /*.limit_facts=*/limits,
      /*.limit_fact_count=*/2,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_target_profile_create_spirv(
      target_environment.get(), &options, loomc_allocator_system(), &profile,
      &result);
  LOOMC_EXPECT_OK(status);
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);
  EXPECT_EQ(profile_ptr.get(), nullptr);
  ExpectFailedResult(result_ptr.get());
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("contradictory values"));
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("spirv.max_workgroup_storage_bytes"));
  EXPECT_THAT(ToString(diagnostic->message), ::testing::HasSubstr("probe:a"));
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("override:b"));
  EXPECT_THAT(ToString(diagnostic->message), ::testing::HasSubstr("49152"));
  EXPECT_THAT(ToString(diagnostic->message), ::testing::HasSubstr("32768"));
}

TEST(TargetSpirvProfileTest,
     ReportsRefinementContradictionsWithBaseProvenance) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  std::string base_provenance = "base:subgroup";
  loomc_spirv_limit_fact_t base_limits[] = {
      {
          /*.limit=*/LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/32,
          /*.provenance=*/
          loomc_make_string_view(base_provenance.data(),
                                 base_provenance.size()),
      },
  };
  loomc_spirv_profile_options_t base_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(base_options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("refine-contradiction-base"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_NONE,
      /*.feature_facts=*/nullptr,
      /*.feature_fact_count=*/0,
      /*.limit_facts=*/base_limits,
      /*.limit_fact_count=*/1,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
  };
  TargetProfilePtr base_profile =
      CreateSpirvProfile(target_environment.get(), &base_options);
  for (char& character : base_provenance) {
    character = 'x';
  }

  loomc_spirv_limit_fact_t refine_limits[] = {
      {
          /*.limit=*/LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/64,
          /*.provenance=*/loomc_make_cstring_view("refine:subgroup"),
      },
  };
  loomc_spirv_profile_options_t refine_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(refine_options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("refine-contradiction"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_NONE,
      /*.feature_facts=*/nullptr,
      /*.feature_fact_count=*/0,
      /*.limit_facts=*/refine_limits,
      /*.limit_fact_count=*/1,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
  };
  loomc_target_profile_t* refined_profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_spirv_target_profile_refine(
      base_profile.get(), &refine_options, loomc_allocator_system(),
      &refined_profile, &result);
  LOOMC_EXPECT_OK(status);
  TargetProfilePtr refined_profile_ptr(refined_profile);
  ResultPtr result_ptr(result);
  EXPECT_EQ(refined_profile_ptr.get(), nullptr);
  ExpectFailedResult(result_ptr.get());
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("contradictory values"));
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("base:subgroup"));
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("refine:subgroup"));
}

TEST(TargetSpirvProfileTest,
     ReportsContradictoryEnvironmentFactsWithProvenance) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_environment_fact_t environment[] = {
      {
          /*.environment=*/LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/kSpirvVersion13,
          /*.provenance=*/loomc_make_cstring_view("probe:a"),
      },
      {
          /*.environment=*/LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/kSpirvVersion10,
          /*.provenance=*/loomc_make_cstring_view("override:b"),
      },
  };
  loomc_spirv_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("contradictory-environment"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_NONE,
      /*.feature_facts=*/nullptr,
      /*.feature_fact_count=*/0,
      /*.limit_facts=*/nullptr,
      /*.limit_fact_count=*/0,
      /*.environment_facts=*/environment,
      /*.environment_fact_count=*/2,
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_target_profile_create_spirv(
      target_environment.get(), &options, loomc_allocator_system(), &profile,
      &result);
  LOOMC_EXPECT_OK(status);
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);
  EXPECT_EQ(profile_ptr.get(), nullptr);
  ExpectFailedResult(result_ptr.get());
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("contradictory values"));
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("spirv.max_spirv_version"));
  EXPECT_THAT(ToString(diagnostic->message), ::testing::HasSubstr("probe:a"));
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("override:b"));
}

TEST(TargetSpirvProfileTest, ReportsInvalidZeroLimitValuesAsResult) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_limit_fact_t limits[] = {
      {
          /*.limit=*/LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/0,
          /*.provenance=*/loomc_make_cstring_view("probe:zero"),
      },
  };
  loomc_spirv_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("invalid-zero-limit"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_NONE,
      /*.feature_facts=*/nullptr,
      /*.feature_fact_count=*/0,
      /*.limit_facts=*/limits,
      /*.limit_fact_count=*/1,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_target_profile_create_spirv(
      target_environment.get(), &options, loomc_allocator_system(), &profile,
      &result);
  LOOMC_EXPECT_OK(status);
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);
  EXPECT_EQ(profile_ptr.get(), nullptr);
  ExpectFailedResult(result_ptr.get());
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("invalid zero value"));
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("spirv.max_workgroup_size_x"));
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("probe:zero"));
}

TEST(TargetSpirvProfileTest, ReportsOutOfRangeLimitValuesAsResult) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_limit_fact_t limits[] = {
      {
          /*.limit=*/LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_COUNT_X,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/UINT64_C(0x100000000),
          /*.provenance=*/loomc_make_cstring_view("probe:too-large"),
      },
  };
  loomc_spirv_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("out-of-range-limit"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_NONE,
      /*.feature_facts=*/nullptr,
      /*.feature_fact_count=*/0,
      /*.limit_facts=*/limits,
      /*.limit_fact_count=*/1,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_target_profile_create_spirv(
      target_environment.get(), &options, loomc_allocator_system(), &profile,
      &result);
  LOOMC_EXPECT_OK(status);
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);
  EXPECT_EQ(profile_ptr.get(), nullptr);
  ExpectFailedResult(result_ptr.get());
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("exceeds uint32_t range"));
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("spirv.max_workgroup_count_x"));
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("probe:too-large"));
}

TEST(TargetSpirvProfileTest, ReportsEnvironmentVersionTooLowAsResult) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_environment_fact_t environment[] = {
      {
          /*.environment=*/LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.value=*/kSpirvVersion10,
          /*.provenance=*/loomc_make_cstring_view("vulkaninfo:apiVersion"),
      },
  };
  loomc_spirv_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("vulkan13-on-spirv10"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
      /*.feature_facts=*/nullptr,
      /*.feature_fact_count=*/0,
      /*.limit_facts=*/nullptr,
      /*.limit_fact_count=*/0,
      /*.environment_facts=*/environment,
      /*.environment_fact_count=*/1,
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_target_profile_create_spirv(
      target_environment.get(), &options, loomc_allocator_system(), &profile,
      &result);
  LOOMC_EXPECT_OK(status);
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);
  EXPECT_EQ(profile_ptr.get(), nullptr);
  ExpectFailedResult(result_ptr.get());
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("requires SPIR-V version"));
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("spirv.max_spirv_version"));
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("vulkaninfo:apiVersion"));
}

TEST(TargetSpirvProfileTest, ReportsMissingFeatureDependenciesAsResult) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_feature_fact_t facts[] = {
      {
          /*.feature=*/LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
          /*.state=*/LOOMC_TARGET_FACT_STATE_TRUE,
          /*.provenance=*/
          loomc_make_cstring_view("override:physical-storage-buffer"),
      },
  };
  loomc_spirv_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("missing-dependency"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_NONE,
      /*.feature_facts=*/facts,
      /*.feature_fact_count=*/1,
      /*.limit_facts=*/nullptr,
      /*.limit_fact_count=*/0,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_target_profile_create_spirv(
      target_environment.get(), &options, loomc_allocator_system(), &profile,
      &result);
  LOOMC_EXPECT_OK(status);
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);
  EXPECT_EQ(profile_ptr.get(), nullptr);
  ExpectFailedResult(result_ptr.get());
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("missing dependency"));
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("spirv.physical_storage_buffer"));
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("spirv.vulkan.shader"));
}

TEST(TargetSpirvProfileTest, RejectsNonSpirvProfileQueries) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_target_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("generic-profile"),
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_status_t create_status = loomc_target_profile_create_empty(
      target_environment.get(), &options, loomc_allocator_system(), &profile);
  LOOMC_EXPECT_OK(create_status);
  TargetProfilePtr profile_ptr(profile);

  loomc_target_fact_state_t state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  loomc_status_t query_status = loomc_spirv_target_profile_query_feature(
      profile_ptr.get(), LOOMC_SPIRV_FEATURE_FLOAT16, &state);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, query_status);

  loomc_spirv_limit_value_t value = {
      /*.state=*/LOOMC_TARGET_FACT_STATE_UNKNOWN,
      /*.value=*/0,
  };
  loomc_status_t limit_query_status = loomc_spirv_target_profile_query_limit(
      profile_ptr.get(), LOOMC_SPIRV_LIMIT_SUBGROUP_SIZE, &value);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, limit_query_status);

  loomc_spirv_environment_value_t environment_value = {
      /*.state=*/LOOMC_TARGET_FACT_STATE_UNKNOWN,
      /*.value=*/0,
  };
  loomc_status_t environment_query_status =
      loomc_spirv_target_profile_query_environment(
          profile_ptr.get(), LOOMC_SPIRV_ENVIRONMENT_MAX_SPIRV_VERSION,
          &environment_value);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         environment_query_status);
}

TEST(TargetSpirvProfileTest, PreparedProfileSelectionCreatesTargetPipeline) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_spirv_profile_options_t profile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(profile_options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("selected-vulkan13"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
      /*.feature_facts=*/nullptr,
      /*.feature_fact_count=*/0,
      /*.limit_facts=*/nullptr,
      /*.limit_fact_count=*/0,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
  };
  TargetProfilePtr profile =
      CreateSpirvProfile(target_environment.get(), &profile_options);
  TargetSelectionPtr selection = CreateSelectionFromProfile(profile.get());
  ContextPtr context = CreateSpirvContext(target_environment.get());

  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/selection.get(),
  };
  loomc_target_pipeline_options_t pipeline_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      /*.structure_size=*/sizeof(pipeline_options),
      /*.next=*/&target_options,
      /*.identifier=*/loomc_make_cstring_view("selected-spirv-pipeline"),
      /*.kind=*/LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      /*.control_flow_lowering=*/LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      /*.source_to_low_max_errors=*/20,
  };
  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_target_pipeline(
      context.get(), &pipeline_options, loomc_allocator_system(), &pass_program,
      &result);
  LOOMC_EXPECT_OK(status);
  PassProgramPtr pass_program_ptr(pass_program);
  ResultPtr result_ptr(result);
  EXPECT_NE(pass_program_ptr.get(), nullptr);
  ExpectSucceededResult(result_ptr.get());
}

}  // namespace
