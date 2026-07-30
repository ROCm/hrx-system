// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/amdgpu/iree_hal.h"

#include <memory>
#include <string>

#include "iree/hal/api.h"
#include "iree/hal/executable/amdgpu/executable_target.h"
#include "iree/hal/utils/device_spec_builder.h"
#include "iree/testing/gtest.h"
#include "loomc/diagnostic.h"
#include "loomc/result.h"
#include "loomc/target.h"
#include "loomc/target/amdgpu/base.h"
#include "loomc/target/amdgpu/profile.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using DeviceSpecPtr =
    HandlePtr<iree_hal_device_spec_t, iree_hal_device_spec_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;

typedef struct FakeHalDevice {
  // HAL resource header used by device vtable dispatch.
  iree_hal_resource_t resource;

  // Immutable device facts borrowed from the test.
  const iree_hal_device_spec_t* device_spec;
} FakeHalDevice;

typedef struct TestTarget {
  // Canonical exact AMDGPU target key.
  iree_string_view_t target_key;

  // Physical-device set represented by the target.
  iree_hal_physical_device_affinity_t physical_device_affinity;
} TestTarget;

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

iree_status_t CreateAmdgpuDeviceSpec(const TestTarget* targets,
                                     iree_host_size_t target_count,
                                     DeviceSpecPtr* out_device_spec) {
  out_device_spec->reset();
  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < target_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_amdgpu_target_identity_t identity = {};
    status = iree_hal_amdgpu_target_identity_parse_artifact_key(
        targets[i].target_key, &identity);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_device_spec_builder_add_executable_targets(
          &builder, &identity, targets[i].physical_device_affinity);
    }
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

TargetEnvironmentPtr CreateAmdgpuTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_amdgpu(
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

void ExpectFailedAdapterResult(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result));
  ASSERT_GE(loomc_result_diagnostic_count(result), 1u);
  const loomc_diagnostic_t* diagnostic = loomc_result_diagnostic_at(result, 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(diagnostic->severity, LOOMC_DIAGNOSTIC_SEVERITY_ERROR);
  EXPECT_EQ(ToString(diagnostic->code), "AMDGPU/IREE_HAL");
}

TargetProfilePtr CreateProfileFromHal(
    loomc_target_environment_t* target_environment, FakeHalDevice* device,
    iree_hal_physical_device_affinity_t physical_device_affinity,
    loomc_result_t** out_result) {
  iree_hal_device_t* hal_device = reinterpret_cast<iree_hal_device_t*>(device);
  const loomc_amdgpu_iree_hal_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_IREE_HAL_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("fake-amdgpu"),
      /*.device=*/hal_device,
      /*.physical_device_affinity=*/physical_device_affinity,
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_status_t status = loomc_target_profile_create_amdgpu_iree_hal(
      target_environment, &options, loomc_allocator_system(), &profile,
      out_result);
  LOOMC_EXPECT_OK(status);
  return TargetProfilePtr(profile);
}

TEST(LoomcAmdgpuIreeHalTargetTest, CreatesExactProfileFromHalTarget) {
  const TestTarget targets[] = {
      {IREE_SV("gfx1151"), 1},
  };
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(
      CreateAmdgpuDeviceSpec(targets, IREE_ARRAYSIZE(targets), &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile =
      CreateProfileFromHal(target_environment.get(), &device,
                           /*physical_device_affinity=*/0, &result);
  ResultPtr result_ptr(result);

  ASSERT_NE(profile.get(), nullptr);
  ExpectSucceededResult(result_ptr.get());
  loomc_amdgpu_target_identity_t identity = {};
  LOOMC_EXPECT_OK(
      loomc_amdgpu_target_profile_query_identity(profile.get(), &identity));
  EXPECT_EQ(ToString(identity.target), "gfx1151");
  EXPECT_EQ(identity.amdhsa_features.sramecc,
            LOOMC_AMDGPU_TARGET_FEATURE_UNSUPPORTED);
  EXPECT_EQ(identity.amdhsa_features.xnack,
            LOOMC_AMDGPU_TARGET_FEATURE_UNSUPPORTED);
}

TEST(LoomcAmdgpuIreeHalTargetTest, PreservesStructuredAmdhsaFeatureModes) {
  const TestTarget targets[] = {
      {IREE_SV("gfx942:sramecc+:xnack-"), 1},
  };
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(
      CreateAmdgpuDeviceSpec(targets, IREE_ARRAYSIZE(targets), &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile =
      CreateProfileFromHal(target_environment.get(), &device,
                           /*physical_device_affinity=*/0, &result);
  ResultPtr result_ptr(result);

  ASSERT_NE(profile.get(), nullptr);
  ExpectSucceededResult(result_ptr.get());
  loomc_amdgpu_target_identity_t identity = {};
  LOOMC_EXPECT_OK(
      loomc_amdgpu_target_profile_query_identity(profile.get(), &identity));
  EXPECT_EQ(ToString(identity.target), "gfx942");
  EXPECT_EQ(identity.amdhsa_features.sramecc, LOOMC_AMDGPU_TARGET_FEATURE_ON);
  EXPECT_EQ(identity.amdhsa_features.xnack, LOOMC_AMDGPU_TARGET_FEATURE_OFF);
}

TEST(LoomcAmdgpuIreeHalTargetTest,
     PhysicalAffinityDisambiguatesHeterogeneousTargets) {
  const TestTarget targets[] = {
      {IREE_SV("gfx1151"), 1},
      {IREE_SV("gfx1100"), 2},
  };
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(
      CreateAmdgpuDeviceSpec(targets, IREE_ARRAYSIZE(targets), &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();

  loomc_result_t* ambiguous_result = nullptr;
  TargetProfilePtr ambiguous_profile =
      CreateProfileFromHal(target_environment.get(), &device,
                           /*physical_device_affinity=*/0, &ambiguous_result);
  ResultPtr ambiguous_result_ptr(ambiguous_result);
  EXPECT_EQ(ambiguous_profile.get(), nullptr);
  ExpectFailedAdapterResult(ambiguous_result_ptr.get());

  loomc_result_t* selected_result = nullptr;
  TargetProfilePtr selected_profile =
      CreateProfileFromHal(target_environment.get(), &device,
                           /*physical_device_affinity=*/2, &selected_result);
  ResultPtr selected_result_ptr(selected_result);
  ASSERT_NE(selected_profile.get(), nullptr);
  ExpectSucceededResult(selected_result_ptr.get());
  loomc_amdgpu_target_identity_t identity = {};
  LOOMC_EXPECT_OK(loomc_amdgpu_target_profile_query_identity(
      selected_profile.get(), &identity));
  EXPECT_EQ(ToString(identity.target), "gfx1100");
}

TEST(LoomcAmdgpuIreeHalTargetTest, ProviderRoutesThroughGenericHalRouter) {
  const TestTarget targets[] = {
      {IREE_SV("gfx1151"), 1},
  };
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(
      CreateAmdgpuDeviceSpec(targets, IREE_ARRAYSIZE(targets), &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  iree_hal_device_t* hal_device = reinterpret_cast<iree_hal_device_t*>(&device);
  const loomc_iree_hal_profile_provider_t* providers[] = {
      loomc_amdgpu_iree_hal_profile_provider(),
  };
  const loomc_iree_hal_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("router"),
      /*.device=*/hal_device,
      /*.physical_device_affinity=*/1,
      /*.providers=*/providers,
      /*.provider_count=*/IREE_ARRAYSIZE(providers),
  };
  loomc_result_t* result = nullptr;
  loomc_target_profile_t* profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_create_iree_hal(
      target_environment.get(), &options, loomc_allocator_system(), &profile,
      &result));
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);

  ASSERT_NE(profile_ptr.get(), nullptr);
  ExpectSucceededResult(result_ptr.get());
  loomc_amdgpu_target_identity_t identity = {};
  LOOMC_EXPECT_OK(
      loomc_amdgpu_target_profile_query_identity(profile_ptr.get(), &identity));
  EXPECT_EQ(ToString(identity.target), "gfx1151");
}

TEST(LoomcAmdgpuIreeHalTargetTest, ProviderMissLetsRouterReportUnsupported) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(
      CreateAmdgpuDeviceSpec(nullptr, /*target_count=*/0, &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  iree_hal_device_t* hal_device = reinterpret_cast<iree_hal_device_t*>(&device);
  const loomc_iree_hal_profile_provider_t* providers[] = {
      loomc_amdgpu_iree_hal_profile_provider(),
  };
  const loomc_iree_hal_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("miss"),
      /*.device=*/hal_device,
      /*.physical_device_affinity=*/0,
      /*.providers=*/providers,
      /*.provider_count=*/IREE_ARRAYSIZE(providers),
  };
  loomc_result_t* result = nullptr;
  loomc_target_profile_t* profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_create_iree_hal(
      target_environment.get(), &options, loomc_allocator_system(), &profile,
      &result));
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);

  EXPECT_EQ(profile_ptr.get(), nullptr);
  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result_ptr.get()));
}

}  // namespace
