// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/iree_hal.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "iree/testing/gtest.h"
#include "loomc/context.h"
#include "loomc/diagnostic.h"
#include "loomc/module.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/target.h"
#include "loomc/target/spirv/base.h"
#include "loomc/target/spirv/profile.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

typedef struct FakeProviderState {
  // Whether this provider recognizes the routed device.
  bool supports;

  // Number of times the callback was invoked.
  int call_count;
} FakeProviderState;

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

ModulePtr ParseModule(const char* source_text) {
  loomc_context_t* context = nullptr;
  LOOMC_EXPECT_OK(
      loomc_context_create(nullptr, loomc_allocator_system(), &context));
  ContextPtr context_ptr(context);

  loomc_workspace_t* workspace = nullptr;
  LOOMC_EXPECT_OK(
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
  WorkspacePtr workspace_ptr(workspace);

  const loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .next = nullptr,
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("runtime_features.loom"),
      .contents = loomc_make_byte_span(source_text, strlen(source_text)),
      .storage = LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(
      loomc_source_create(&source_options, loomc_allocator_system(), &source));
  SourcePtr source_ptr(source);

  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_deserialize_text_from_source(
      context_ptr.get(), workspace_ptr.get(), source_ptr.get(), nullptr,
      loomc_allocator_system(), &module, &result));
  ResultPtr result_ptr(result);
  if (result_ptr && !loomc_result_succeeded(result_ptr.get())) {
    for (loomc_host_size_t i = 0;
         i < loomc_result_diagnostic_count(result_ptr.get()); ++i) {
      const loomc_diagnostic_t* diagnostic =
          loomc_result_diagnostic_at(result_ptr.get(), i);
      ADD_FAILURE() << ToString(diagnostic->message);
    }
  }
  return ModulePtr(module);
}

iree_hal_device_t* FakeDevice() {
  return reinterpret_cast<iree_hal_device_t*>(static_cast<uintptr_t>(1));
}

const iree_hal_executable_target_t* FakeExecutableTarget() {
  return reinterpret_cast<const iree_hal_executable_target_t*>(
      static_cast<uintptr_t>(2));
}

TargetEnvironmentPtr CreateSpirvTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_spirv(
      loomc_allocator_system(), &target_environment);
  LOOMC_EXPECT_OK(status);
  return TargetEnvironmentPtr(target_environment);
}

void ExpectFailedTargetResult(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result));
  ASSERT_GE(loomc_result_diagnostic_count(result), 1u);
  const loomc_diagnostic_t* diagnostic = loomc_result_diagnostic_at(result, 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(diagnostic->severity, LOOMC_DIAGNOSTIC_SEVERITY_ERROR);
  EXPECT_EQ(ToString(diagnostic->code), "IREE_HAL/TARGET");
}

loomc_status_t FakeSelectTarget(
    void* user_data, loomc_target_environment_t* target_environment,
    const loomc_iree_hal_target_options_t* options, loomc_allocator_t allocator,
    bool* out_supported, loomc_iree_hal_target_selection_t* out_selection,
    loomc_result_t** out_result) {
  FakeProviderState* state = static_cast<FakeProviderState*>(user_data);
  ++state->call_count;
  *out_supported = state->supports;
  *out_selection = {};
  *out_result = nullptr;
  if (!state->supports) {
    return loomc_ok_status();
  }

  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_spirv_profile_options_t profile_options = {
      .type = LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      .structure_size = sizeof(profile_options),
      .next = nullptr,
      .identifier = options->identifier,
      .preset = LOOMC_SPIRV_PROFILE_PRESET_NONE,
      .feature_facts = nullptr,
      .feature_fact_count = 0,
      .limit_facts = nullptr,
      .limit_fact_count = 0,
      .environment_facts = nullptr,
      .environment_fact_count = 0,
  };
  loomc_status_t status = loomc_target_profile_create_spirv(
      target_environment, &profile_options, allocator, &profile, &result);
  if (loomc_status_is_ok(status)) {
    out_selection->target_profile = profile;
    out_selection->executable_target = FakeExecutableTarget();
    *out_result = result;
    profile = nullptr;
    result = nullptr;
  } else {
    loomc_target_profile_release(profile);
    loomc_result_release(result);
  }
  return status;
}

TEST(LoomcIreeHalTargetTest, RejectsInvalidArguments) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_result_t* result = nullptr;
  loomc_iree_hal_target_selection_t selection = {};

  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_target_select_iree_hal(nullptr, nullptr, loomc_allocator_system(),
                                   &selection, &result));

  loomc_iree_hal_target_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_IREE_HAL_TARGET_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .identifier = loomc_make_cstring_view("invalid"),
      .device = nullptr,
      .physical_device_affinity = 0,
      .target_profile = nullptr,
      .providers = nullptr,
      .provider_count = 0,
  };
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_target_select_iree_hal(
                             target_environment.get(), &options,
                             loomc_allocator_system(), &selection, &result));
}

TEST(LoomcIreeHalTargetTest, EmptyProviderTableReturnsFailedResult) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_iree_hal_target_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_IREE_HAL_TARGET_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .identifier = loomc_make_cstring_view("empty"),
      .device = FakeDevice(),
      .physical_device_affinity = 0,
      .target_profile = nullptr,
      .providers = nullptr,
      .provider_count = 0,
  };
  loomc_result_t* result = nullptr;
  loomc_iree_hal_target_selection_t selection = {};
  LOOMC_ASSERT_OK(loomc_target_select_iree_hal(
      target_environment.get(), &options, loomc_allocator_system(), &selection,
      &result));
  TargetProfilePtr profile_ptr(selection.target_profile);
  ResultPtr result_ptr(result);

  EXPECT_EQ(profile_ptr.get(), nullptr);
  ExpectFailedTargetResult(result_ptr.get());
}

TEST(LoomcIreeHalTargetTest, UnsupportedProvidersReturnFailedResult) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  FakeProviderState first_state = {.supports = false, .call_count = 0};
  FakeProviderState second_state = {.supports = false, .call_count = 0};
  const loomc_iree_hal_target_provider_t first_provider = {
      .name = loomc_make_cstring_view("first"),
      .user_data = &first_state,
      .select_target = FakeSelectTarget,
  };
  const loomc_iree_hal_target_provider_t second_provider = {
      .name = loomc_make_cstring_view("second"),
      .user_data = &second_state,
      .select_target = FakeSelectTarget,
  };
  const loomc_iree_hal_target_provider_t* providers[] = {
      &first_provider,
      &second_provider,
  };
  loomc_iree_hal_target_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_IREE_HAL_TARGET_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .identifier = loomc_make_cstring_view("unsupported"),
      .device = FakeDevice(),
      .physical_device_affinity = 0,
      .target_profile = nullptr,
      .providers = providers,
      .provider_count = 2,
  };
  loomc_result_t* result = nullptr;
  loomc_iree_hal_target_selection_t selection = {};
  LOOMC_ASSERT_OK(loomc_target_select_iree_hal(
      target_environment.get(), &options, loomc_allocator_system(), &selection,
      &result));
  TargetProfilePtr profile_ptr(selection.target_profile);
  ResultPtr result_ptr(result);

  EXPECT_EQ(first_state.call_count, 1);
  EXPECT_EQ(second_state.call_count, 1);
  EXPECT_EQ(profile_ptr.get(), nullptr);
  ExpectFailedTargetResult(result_ptr.get());
}

TEST(LoomcIreeHalTargetTest, OneEnabledRouteCreatesProfile) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  FakeProviderState state = {.supports = true, .call_count = 0};
  const loomc_iree_hal_target_provider_t provider = {
      .name = loomc_make_cstring_view("enabled"),
      .user_data = &state,
      .select_target = FakeSelectTarget,
  };
  const loomc_iree_hal_target_provider_t* providers[] = {&provider};
  loomc_iree_hal_target_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_IREE_HAL_TARGET_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .identifier = loomc_make_cstring_view("enabled"),
      .device = FakeDevice(),
      .physical_device_affinity = 0,
      .target_profile = nullptr,
      .providers = providers,
      .provider_count = 1,
  };
  loomc_result_t* result = nullptr;
  loomc_iree_hal_target_selection_t selection = {};
  LOOMC_ASSERT_OK(loomc_target_select_iree_hal(
      target_environment.get(), &options, loomc_allocator_system(), &selection,
      &result));
  TargetProfilePtr profile_ptr(selection.target_profile);
  ResultPtr result_ptr(result);

  EXPECT_EQ(state.call_count, 1);
  ASSERT_NE(profile_ptr.get(), nullptr);
  EXPECT_EQ(selection.executable_target, FakeExecutableTarget());
  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
}

TEST(LoomcIreeHalTargetTest, MultipleRoutesStopAtFirstSupportedProvider) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  FakeProviderState first_state = {.supports = false, .call_count = 0};
  FakeProviderState second_state = {.supports = true, .call_count = 0};
  FakeProviderState third_state = {.supports = true, .call_count = 0};
  const loomc_iree_hal_target_provider_t first_provider = {
      .name = loomc_make_cstring_view("first"),
      .user_data = &first_state,
      .select_target = FakeSelectTarget,
  };
  const loomc_iree_hal_target_provider_t second_provider = {
      .name = loomc_make_cstring_view("second"),
      .user_data = &second_state,
      .select_target = FakeSelectTarget,
  };
  const loomc_iree_hal_target_provider_t third_provider = {
      .name = loomc_make_cstring_view("third"),
      .user_data = &third_state,
      .select_target = FakeSelectTarget,
  };
  const loomc_iree_hal_target_provider_t* providers[] = {
      &first_provider,
      &second_provider,
      &third_provider,
  };
  loomc_iree_hal_target_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_IREE_HAL_TARGET_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .identifier = loomc_make_cstring_view("multi"),
      .device = FakeDevice(),
      .physical_device_affinity = 0,
      .target_profile = nullptr,
      .providers = providers,
      .provider_count = 3,
  };
  loomc_result_t* result = nullptr;
  loomc_iree_hal_target_selection_t selection = {};
  LOOMC_ASSERT_OK(loomc_target_select_iree_hal(
      target_environment.get(), &options, loomc_allocator_system(), &selection,
      &result));
  TargetProfilePtr profile_ptr(selection.target_profile);
  ResultPtr result_ptr(result);

  EXPECT_EQ(first_state.call_count, 1);
  EXPECT_EQ(second_state.call_count, 1);
  EXPECT_EQ(third_state.call_count, 0);
  ASSERT_NE(profile_ptr.get(), nullptr);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
}

TEST(LoomcIreeHalRuntimeFeaturesTest, RejectsInvalidArguments) {
  iree_hal_device_runtime_feature_flags_t runtime_features =
      IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_FEEDBACK;
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_iree_hal_module_query_runtime_features(
                             nullptr, nullptr, &runtime_features));
  EXPECT_EQ(runtime_features, IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_NONE);

  ModulePtr module = ParseModule("");
  ASSERT_NE(module, nullptr);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_iree_hal_module_query_runtime_features(
                             module.get(), nullptr, nullptr));

  const loomc_sanitizer_options_t invalid_options = {
      .type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
      .structure_size = sizeof(invalid_options),
      .next = nullptr,
      .checks = 1ull << 63,
  };
  runtime_features = IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_FEEDBACK;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_iree_hal_module_query_runtime_features(
          module.get(), &invalid_options, &runtime_features));
  EXPECT_EQ(runtime_features, IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_NONE);
}

TEST(LoomcIreeHalRuntimeFeaturesTest, MapsRequestedSanitizerServices) {
  ModulePtr module = ParseModule("");
  ASSERT_NE(module, nullptr);
  loomc_sanitizer_options_t sanitizer_options = {
      .type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
      .structure_size = sizeof(sanitizer_options),
      .next = nullptr,
      .checks = LOOMC_SANITIZER_CHECK_ACCESS | LOOMC_SANITIZER_CHECK_RACE,
      .flags = 0,
      .reporting_mode = LOOMC_SANITIZER_REPORTING_MODE_DEFAULT,
  };

  iree_hal_device_runtime_feature_flags_t runtime_features =
      IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_NONE;
  LOOMC_ASSERT_OK(loomc_iree_hal_module_query_runtime_features(
      module.get(), &sanitizer_options, &runtime_features));
  EXPECT_EQ(runtime_features, IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_FEEDBACK |
                                  IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_ASAN |
                                  IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_TSAN);

  sanitizer_options.reporting_mode = LOOMC_SANITIZER_REPORTING_MODE_TRAP;
  LOOMC_ASSERT_OK(loomc_iree_hal_module_query_runtime_features(
      module.get(), &sanitizer_options, &runtime_features));
  EXPECT_EQ(runtime_features, IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_ASAN |
                                  IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_TSAN);
}

TEST(LoomcIreeHalRuntimeFeaturesTest, FindsAuthoredSanitizerOperations) {
  ModulePtr module = ParseModule(R"(
func.def @entry(%value: index) -> (index) {
  %checked = sanitizer.assert.value %value [ne(%value, 0)] : index
  func.return %checked : index
}
)");
  ASSERT_NE(module, nullptr);

  iree_hal_device_runtime_feature_flags_t runtime_features =
      IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_NONE;
  LOOMC_ASSERT_OK(loomc_iree_hal_module_query_runtime_features(
      module.get(), nullptr, &runtime_features));
  EXPECT_EQ(runtime_features, IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_FEEDBACK);
}

}  // namespace
