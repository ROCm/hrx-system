// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/amd/xdna/iree_hal.h"

#include <string>

#include "iree/hal/api.h"
#include "iree/hal/utils/device_spec_builder.h"
#include "iree/testing/gtest.h"
#include "loomc/diagnostic.h"
#include "loomc/result.h"
#include "loomc/target/amd/xdna.h"
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
  // XDNA target key advertised by the fake device.
  iree_string_view_t target_key;
  // Physical-device set represented by the target.
  iree_hal_physical_device_affinity_t physical_device_affinity;
  // Advertised executable target kind.
  iree_hal_executable_target_kind_t kind;
} TestTarget;

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

iree_status_t CreateXdnaDeviceSpec(const TestTarget* targets,
                                   iree_host_size_t target_count,
                                   DeviceSpecPtr* out_device_spec) {
  out_device_spec->reset();
  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < target_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_executable_target_t target = {
        /*.family=*/IREE_SV("xdna"),
        /*.target_key=*/targets[i].target_key,
        /*.kind=*/targets[i].kind,
        /*.priority=*/0,
        /*.physical_device_affinity=*/targets[i].physical_device_affinity,
        /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
    };
    status =
        iree_hal_device_spec_builder_add_executable_target(&builder, &target);
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

TargetEnvironmentPtr CreateXdnaTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_xdna(
      loomc_allocator_system(), &target_environment);
  LOOMC_EXPECT_OK(status);
  return TargetEnvironmentPtr(target_environment);
}

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

void ExpectFailedAdapterResult(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result));
  ASSERT_GE(loomc_result_diagnostic_count(result), 1u);
  const loomc_diagnostic_t* diagnostic = loomc_result_diagnostic_at(result, 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(diagnostic->severity, LOOMC_DIAGNOSTIC_SEVERITY_ERROR);
  EXPECT_EQ(ToString(diagnostic->code), "XDNA/IREE_HAL");
}

TargetProfilePtr SelectTargetFromHal(
    loomc_target_environment_t* target_environment, FakeHalDevice* device,
    iree_hal_physical_device_affinity_t physical_device_affinity,
    loomc_target_profile_t* requested_profile, loomc_result_t** out_result,
    const iree_hal_executable_target_t** out_executable_target = nullptr) {
  const loomc_iree_hal_target_provider_t* providers[] = {
      loomc_xdna_iree_hal_target_provider(),
  };
  const loomc_iree_hal_target_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_IREE_HAL_TARGET_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("fake-xdna"),
      /*.device=*/reinterpret_cast<iree_hal_device_t*>(device),
      /*.physical_device_affinity=*/physical_device_affinity,
      /*.target_profile=*/requested_profile,
      /*.providers=*/providers,
      /*.provider_count=*/IREE_ARRAYSIZE(providers),
  };
  loomc_iree_hal_target_selection_t selection = {};
  loomc_status_t status = loomc_target_select_iree_hal(
      target_environment, &options, loomc_allocator_system(), &selection,
      out_result);
  LOOMC_EXPECT_OK(status);
  if (out_executable_target != nullptr) {
    *out_executable_target = selection.executable_target;
  }
  return TargetProfilePtr(selection.target_profile);
}

TEST(LoomcXdnaIreeHalTargetTest, SelectsKnownExactTarget) {
  const iree_string_view_t kTargetKey = IREE_SV("amd.xdna.strix_halo.17f0_11");
  const TestTarget targets[] = {
      {kTargetKey, 1, IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT},
  };
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(
      CreateXdnaDeviceSpec(targets, IREE_ARRAYSIZE(targets), &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateXdnaTargetEnvironment();
  loomc_result_t* result = nullptr;
  const iree_hal_executable_target_t* executable_target = nullptr;
  TargetProfilePtr profile = SelectTargetFromHal(
      target_environment.get(), &device, /*physical_device_affinity=*/0,
      /*requested_profile=*/nullptr, &result, &executable_target);
  ResultPtr result_ptr(result);

  ASSERT_NE(profile.get(), nullptr);
  ASSERT_NE(executable_target, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(executable_target->target_key, kTargetKey));
  ExpectSucceededResult(result_ptr.get());
}

TEST(LoomcXdnaIreeHalTargetTest, PhysicalAffinityDisambiguatesExactTargets) {
  const TestTarget targets[] = {
      {IREE_SV("amd.xdna.strix_halo.17f0_11"), 1,
       IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT},
      {IREE_SV("amd.xdna.strix.17f0_10"), 2,
       IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT},
  };
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(
      CreateXdnaDeviceSpec(targets, IREE_ARRAYSIZE(targets), &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateXdnaTargetEnvironment();

  loomc_result_t* ambiguous_result = nullptr;
  TargetProfilePtr ambiguous_profile = SelectTargetFromHal(
      target_environment.get(), &device, /*physical_device_affinity=*/0,
      /*requested_profile=*/nullptr, &ambiguous_result);
  ResultPtr ambiguous_result_ptr(ambiguous_result);
  EXPECT_EQ(ambiguous_profile.get(), nullptr);
  ExpectFailedAdapterResult(ambiguous_result_ptr.get());

  loomc_result_t* selected_result = nullptr;
  const iree_hal_executable_target_t* executable_target = nullptr;
  TargetProfilePtr selected_profile = SelectTargetFromHal(
      target_environment.get(), &device, /*physical_device_affinity=*/2,
      /*requested_profile=*/nullptr, &selected_result, &executable_target);
  ResultPtr selected_result_ptr(selected_result);
  ASSERT_NE(selected_profile.get(), nullptr);
  ASSERT_NE(executable_target, nullptr);
  EXPECT_EQ(executable_target->physical_device_affinity, 2u);
  EXPECT_TRUE(iree_string_view_equal(executable_target->target_key,
                                     IREE_SV("amd.xdna.strix.17f0_10")));
  ExpectSucceededResult(selected_result_ptr.get());
}

TEST(LoomcXdnaIreeHalTargetTest, RejectsUnknownExactTarget) {
  const TestTarget targets[] = {
      {IREE_SV("amd.xdna.unknown"), 1, IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT},
  };
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(
      CreateXdnaDeviceSpec(targets, IREE_ARRAYSIZE(targets), &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateXdnaTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile = SelectTargetFromHal(
      target_environment.get(), &device, /*physical_device_affinity=*/0,
      /*requested_profile=*/nullptr, &result);
  ResultPtr result_ptr(result);

  EXPECT_EQ(profile.get(), nullptr);
  ExpectFailedAdapterResult(result_ptr.get());
}

TEST(LoomcXdnaIreeHalTargetTest, ForcedProfileMustBeLoadableByDevice) {
  const TestTarget targets[] = {
      {IREE_SV("amd.xdna.unknown"), 1, IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT},
  };
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(
      CreateXdnaDeviceSpec(targets, IREE_ARRAYSIZE(targets), &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateXdnaTargetEnvironment();
  loomc_target_profile_t* requested_profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_create_xdna(
      target_environment.get(),
      loomc_make_cstring_view("amd.xdna.strix_halo.17f0_11"),
      loomc_allocator_system(), &requested_profile));
  TargetProfilePtr requested_profile_ptr(requested_profile);
  loomc_result_t* result = nullptr;
  TargetProfilePtr selected_profile = SelectTargetFromHal(
      target_environment.get(), &device, /*physical_device_affinity=*/1,
      requested_profile_ptr.get(), &result);
  ResultPtr result_ptr(result);

  EXPECT_EQ(selected_profile.get(), nullptr);
  ExpectFailedAdapterResult(result_ptr.get());
}

TEST(LoomcXdnaIreeHalTargetTest, ForcedProfilePreservesExactPair) {
  const iree_string_view_t kTargetKey = IREE_SV("amd.xdna.strix_halo.17f0_11");
  const TestTarget targets[] = {
      {kTargetKey, 1, IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT},
  };
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(
      CreateXdnaDeviceSpec(targets, IREE_ARRAYSIZE(targets), &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateXdnaTargetEnvironment();
  loomc_target_profile_t* requested_profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_create_xdna(
      target_environment.get(),
      loomc_make_string_view(kTargetKey.data, kTargetKey.size),
      loomc_allocator_system(), &requested_profile));
  TargetProfilePtr requested_profile_ptr(requested_profile);
  loomc_result_t* result = nullptr;
  const iree_hal_executable_target_t* executable_target = nullptr;
  TargetProfilePtr selected_profile = SelectTargetFromHal(
      target_environment.get(), &device, /*physical_device_affinity=*/1,
      requested_profile_ptr.get(), &result, &executable_target);
  ResultPtr result_ptr(result);

  EXPECT_EQ(selected_profile.get(), requested_profile_ptr.get());
  ASSERT_NE(executable_target, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(executable_target->target_key, kTargetKey));
  ExpectSucceededResult(result_ptr.get());
}

TEST(LoomcXdnaIreeHalTargetTest, ProviderMissLetsRouterReportUnsupported) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(
      CreateXdnaDeviceSpec(nullptr, /*target_count=*/0, &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateXdnaTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile = SelectTargetFromHal(
      target_environment.get(), &device, /*physical_device_affinity=*/0,
      /*requested_profile=*/nullptr, &result);
  ResultPtr result_ptr(result);

  EXPECT_EQ(profile.get(), nullptr);
  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result_ptr.get()));
}

}  // namespace
