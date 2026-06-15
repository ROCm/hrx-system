// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/iree_hal.h"

#include <cstdint>
#include <memory>
#include <string>

#include "iree/testing/gtest.h"
#include "loomc/diagnostic.h"
#include "loomc/result.h"
#include "loomc/target.h"
#include "loomc/target/spirv/base.h"
#include "loomc/target/spirv/profile.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;

typedef struct FakeProviderState {
  // Whether this provider recognizes the routed device.
  bool supports;

  // Number of times the callback was invoked.
  int call_count;
} FakeProviderState;

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

iree_hal_device_t* FakeDevice() {
  return reinterpret_cast<iree_hal_device_t*>(static_cast<uintptr_t>(1));
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

loomc_status_t FakeCreateProfile(
    void* user_data, loomc_target_environment_t* target_environment,
    const loomc_iree_hal_profile_options_t* options,
    loomc_allocator_t allocator, bool* out_supported,
    loomc_target_profile_t** out_profile, loomc_result_t** out_result) {
  FakeProviderState* state = static_cast<FakeProviderState*>(user_data);
  ++state->call_count;
  *out_supported = state->supports;
  *out_profile = nullptr;
  *out_result = nullptr;
  if (!state->supports) {
    return loomc_ok_status();
  }

  loomc_target_profile_t* profile = nullptr;
  loomc_result_t* result = nullptr;
  loomc_spirv_profile_options_t profile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(profile_options),
      /*.next=*/nullptr,
      /*.identifier=*/options->identifier,
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_NONE,
      /*.feature_facts=*/nullptr,
      /*.feature_fact_count=*/0,
      /*.limit_facts=*/nullptr,
      /*.limit_fact_count=*/0,
      /*.environment_facts=*/nullptr,
      /*.environment_fact_count=*/0,
  };
  loomc_status_t status = loomc_target_profile_create_spirv(
      target_environment, &profile_options, allocator, &profile, &result);
  if (loomc_status_is_ok(status)) {
    *out_profile = profile;
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
  loomc_target_profile_t* profile = nullptr;

  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_target_profile_create_iree_hal(
          nullptr, nullptr, loomc_allocator_system(), &profile, &result));

  loomc_iree_hal_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("invalid"),
      /*.device=*/nullptr,
      /*.executable_cache=*/nullptr,
      /*.providers=*/nullptr,
      /*.provider_count=*/0,
  };
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_target_profile_create_iree_hal(
                             target_environment.get(), &options,
                             loomc_allocator_system(), &profile, &result));
}

TEST(LoomcIreeHalTargetTest, EmptyProviderTableReturnsFailedResult) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_iree_hal_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("empty"),
      /*.device=*/FakeDevice(),
      /*.executable_cache=*/nullptr,
      /*.providers=*/nullptr,
      /*.provider_count=*/0,
  };
  loomc_result_t* result = nullptr;
  loomc_target_profile_t* profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_create_iree_hal(
      target_environment.get(), &options, loomc_allocator_system(), &profile,
      &result));
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);

  EXPECT_EQ(profile_ptr.get(), nullptr);
  ExpectFailedTargetResult(result_ptr.get());
}

TEST(LoomcIreeHalTargetTest, UnsupportedProvidersReturnFailedResult) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  FakeProviderState first_state = {/*.supports=*/false, /*.call_count=*/0};
  FakeProviderState second_state = {/*.supports=*/false, /*.call_count=*/0};
  const loomc_iree_hal_profile_provider_t first_provider = {
      /*.name=*/loomc_make_cstring_view("first"),
      /*.user_data=*/&first_state,
      /*.create_profile=*/FakeCreateProfile,
  };
  const loomc_iree_hal_profile_provider_t second_provider = {
      /*.name=*/loomc_make_cstring_view("second"),
      /*.user_data=*/&second_state,
      /*.create_profile=*/FakeCreateProfile,
  };
  const loomc_iree_hal_profile_provider_t* providers[] = {
      &first_provider,
      &second_provider,
  };
  loomc_iree_hal_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("unsupported"),
      /*.device=*/FakeDevice(),
      /*.executable_cache=*/nullptr,
      /*.providers=*/providers,
      /*.provider_count=*/2,
  };
  loomc_result_t* result = nullptr;
  loomc_target_profile_t* profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_create_iree_hal(
      target_environment.get(), &options, loomc_allocator_system(), &profile,
      &result));
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);

  EXPECT_EQ(first_state.call_count, 1);
  EXPECT_EQ(second_state.call_count, 1);
  EXPECT_EQ(profile_ptr.get(), nullptr);
  ExpectFailedTargetResult(result_ptr.get());
}

TEST(LoomcIreeHalTargetTest, OneEnabledRouteCreatesProfile) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  FakeProviderState state = {/*.supports=*/true, /*.call_count=*/0};
  const loomc_iree_hal_profile_provider_t provider = {
      /*.name=*/loomc_make_cstring_view("enabled"),
      /*.user_data=*/&state,
      /*.create_profile=*/FakeCreateProfile,
  };
  const loomc_iree_hal_profile_provider_t* providers[] = {&provider};
  loomc_iree_hal_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("enabled"),
      /*.device=*/FakeDevice(),
      /*.executable_cache=*/nullptr,
      /*.providers=*/providers,
      /*.provider_count=*/1,
  };
  loomc_result_t* result = nullptr;
  loomc_target_profile_t* profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_create_iree_hal(
      target_environment.get(), &options, loomc_allocator_system(), &profile,
      &result));
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);

  EXPECT_EQ(state.call_count, 1);
  ASSERT_NE(profile_ptr.get(), nullptr);
  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
}

TEST(LoomcIreeHalTargetTest, MultipleRoutesStopAtFirstSupportedProvider) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  FakeProviderState first_state = {/*.supports=*/false, /*.call_count=*/0};
  FakeProviderState second_state = {/*.supports=*/true, /*.call_count=*/0};
  FakeProviderState third_state = {/*.supports=*/true, /*.call_count=*/0};
  const loomc_iree_hal_profile_provider_t first_provider = {
      /*.name=*/loomc_make_cstring_view("first"),
      /*.user_data=*/&first_state,
      /*.create_profile=*/FakeCreateProfile,
  };
  const loomc_iree_hal_profile_provider_t second_provider = {
      /*.name=*/loomc_make_cstring_view("second"),
      /*.user_data=*/&second_state,
      /*.create_profile=*/FakeCreateProfile,
  };
  const loomc_iree_hal_profile_provider_t third_provider = {
      /*.name=*/loomc_make_cstring_view("third"),
      /*.user_data=*/&third_state,
      /*.create_profile=*/FakeCreateProfile,
  };
  const loomc_iree_hal_profile_provider_t* providers[] = {
      &first_provider,
      &second_provider,
      &third_provider,
  };
  loomc_iree_hal_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("multi"),
      /*.device=*/FakeDevice(),
      /*.executable_cache=*/nullptr,
      /*.providers=*/providers,
      /*.provider_count=*/3,
  };
  loomc_result_t* result = nullptr;
  loomc_target_profile_t* profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_create_iree_hal(
      target_environment.get(), &options, loomc_allocator_system(), &profile,
      &result));
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);

  EXPECT_EQ(first_state.call_count, 1);
  EXPECT_EQ(second_state.call_count, 1);
  EXPECT_EQ(third_state.call_count, 0);
  ASSERT_NE(profile_ptr.get(), nullptr);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
}

}  // namespace
