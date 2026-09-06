// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/amdgpu/device_provider.h"

#include <memory>

#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/artifact_key.h"
#include "loom/target/arch/amdgpu/facts.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/tooling/execution/hal/runtime.h"

namespace loom {
namespace {

static const loom_amdgpu_target_info_t* LookupTarget(const char* name) {
  const loom_amdgpu_target_info_t* target = nullptr;
  IREE_CHECK_OK(loom_amdgpu_target_info_lookup_target(
      iree_make_cstring_view(name), &target));
  return target;
}

struct DeviceSpecDeleter {
  void operator()(iree_hal_device_spec_t* device_spec) const {
    iree_hal_device_spec_release(device_spec);
  }
};
using DeviceSpecPtr =
    std::unique_ptr<iree_hal_device_spec_t, DeviceSpecDeleter>;

typedef struct FakeHalDevice {
  // HAL resource header used by device vtable dispatch.
  iree_hal_resource_t resource;

  // Immutable device facts borrowed from the test fixture.
  const iree_hal_device_spec_t* device_spec;
} FakeHalDevice;

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

enum class AmdgpuDeviceTargetSet {
  kGenericOnly,
  kGenericAndExact,
  kGenericAndUnknownExact,
};

static iree_status_t CreateAmdgpuExecutableDeviceSpec(
    AmdgpuDeviceTargetSet target_set, DeviceSpecPtr* out_device_spec) {
  out_device_spec->reset();
  const iree_hal_executable_target_t executable_targets[] = {
      {
          /*.family=*/IREE_SV("amdgpu"),
          /*.target_key=*/IREE_SV("gfx11-generic"),
          /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
          /*.priority=*/50,
          /*.physical_device_affinity=*/1,
          /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
      },
      {
          /*.family=*/IREE_SV("amdgpu"),
          /*.target_key=*/
          target_set == AmdgpuDeviceTargetSet::kGenericAndUnknownExact
              ? IREE_SV("gfx9999")
              : IREE_SV("gfx1151"),
          /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
          /*.priority=*/100,
          /*.physical_device_affinity=*/1,
          /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
      },
  };
  const iree_hal_device_executable_spec_t executables = {
      /*.target_count=*/target_set != AmdgpuDeviceTargetSet::kGenericOnly
          ? IREE_ARRAYSIZE(executable_targets)
          : IREE_ARRAYSIZE(executable_targets) - 1,
      /*.targets=*/executable_targets,
      /*.flags=*/IREE_HAL_DEVICE_EXECUTABLE_SPEC_FLAG_NONE,
  };
  const iree_hal_queue_family_spec_t queue_family = {
      /*.name=*/IREE_SV("dispatch"),
      /*.provisioned_queue_count=*/1,
      /*.priority_count=*/1,
      /*.timestamp_valid_bits=*/0,
      /*.timestamp_frequency_hz=*/0,
      /*.physical_device_affinity=*/1,
      /*.role_flags=*/IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH,
  };
  const iree_hal_device_queue_spec_t queues = {
      /*.family_count=*/1,
      /*.families=*/&queue_family,
  };
  const iree_hal_device_spec_params_t params = {
      /*.identity=*/nullptr,
      /*.memory=*/nullptr,
      /*.virtual_memory=*/nullptr,
      /*.queues=*/&queues,
      /*.dispatch=*/nullptr,
      /*.timing=*/nullptr,
      /*.executables=*/&executables,
      /*.sanitizer=*/nullptr,
      /*.facet_count=*/0,
      /*.facets=*/nullptr,
  };
  iree_hal_device_spec_t* device_spec = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_create(
      &params, iree_allocator_system(), &device_spec));
  out_device_spec->reset(device_spec);
  return iree_ok_status();
}

static iree_status_t CreateSingleAmdgpuExecutableDeviceSpec(
    iree_string_view_t target_key, iree_hal_executable_target_kind_t kind,
    DeviceSpecPtr* out_device_spec) {
  out_device_spec->reset();
  const iree_hal_executable_target_t executable_target = {
      /*.family=*/IREE_SV("amdgpu"),
      /*.target_key=*/target_key,
      /*.kind=*/kind,
      /*.priority=*/100,
      /*.physical_device_affinity=*/1,
      /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
  };
  const iree_hal_device_executable_spec_t executables = {
      /*.target_count=*/1,
      /*.targets=*/&executable_target,
      /*.flags=*/IREE_HAL_DEVICE_EXECUTABLE_SPEC_FLAG_NONE,
  };
  const iree_hal_queue_family_spec_t queue_family = {
      /*.name=*/IREE_SV("dispatch"),
      /*.provisioned_queue_count=*/1,
      /*.priority_count=*/1,
      /*.timestamp_valid_bits=*/0,
      /*.timestamp_frequency_hz=*/0,
      /*.physical_device_affinity=*/1,
      /*.role_flags=*/IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH,
  };
  const iree_hal_device_queue_spec_t queues = {
      /*.family_count=*/1,
      /*.families=*/&queue_family,
  };
  const iree_hal_device_spec_params_t params = {
      /*.identity=*/nullptr,
      /*.memory=*/nullptr,
      /*.virtual_memory=*/nullptr,
      /*.queues=*/&queues,
      /*.dispatch=*/nullptr,
      /*.timing=*/nullptr,
      /*.executables=*/&executables,
      /*.sanitizer=*/nullptr,
      /*.facet_count=*/0,
      /*.facets=*/nullptr,
  };
  iree_hal_device_spec_t* device_spec = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_create(
      &params, iree_allocator_system(), &device_spec));
  out_device_spec->reset(device_spec);
  return iree_ok_status();
}

static loom_amdgpu_target_facts_t MakeTargetRequirement(const char* name) {
  loom_amdgpu_target_facts_t requirement = {};
  requirement.base.fact_type = &loom_amdgpu_target_fact_type;
  loom_amdgpu_target_identity_initialize(LookupTarget(name),
                                         &requirement.identity);
  return requirement;
}

static void ExpectSelectedTarget(
    const loom_device_target_t& target,
    iree_hal_executable_target_kind_t expected_kind,
    iree_string_view_t expected_target_key) {
  const loom_amdgpu_target_profile_t* target_profile =
      loom_amdgpu_target_profile_cast(target.artifact_target.target_profile);
  ASSERT_NE(target_profile, nullptr);
  ASSERT_NE(target.executable_target, nullptr);
  EXPECT_NE(loom_device_target_bundle(&target), nullptr);
  EXPECT_EQ(target.executable_target->kind, expected_kind);
  EXPECT_TRUE(iree_string_view_equal(target.artifact_target.target_key,
                                     expected_target_key));
  EXPECT_TRUE(iree_string_view_equal(target_profile->identity.target->name,
                                     expected_target_key));
}

class AmdgpuDeviceProviderTest : public ::testing::Test {
 protected:
  iree_status_t Initialize(AmdgpuDeviceTargetSet target_set) {
    IREE_RETURN_IF_ERROR(
        CreateAmdgpuExecutableDeviceSpec(target_set, &device_spec_));
    InitializeDevice();
    return iree_ok_status();
  }

  iree_status_t Initialize(iree_string_view_t target_key,
                           iree_hal_executable_target_kind_t kind) {
    IREE_RETURN_IF_ERROR(CreateSingleAmdgpuExecutableDeviceSpec(
        target_key, kind, &device_spec_));
    InitializeDevice();
    return iree_ok_status();
  }

  iree_status_t SelectCompatibleTarget(
      const loom_amdgpu_target_facts_t& requirement,
      loom_device_target_t* out_target) {
    return loom_device_provider_select_compatible_target(
        &loom_amdgpu_device_provider, &runtime_, &requirement.base,
        iree_allocator_system(), out_target);
  }

  void InitializeDevice() {
    device_.device_spec = device_spec_.get();
    iree_hal_resource_initialize(&kFakeHalDeviceVtable, &device_.resource);
    iree_hal_queue_family_initialize(/*ordinal=*/0, &dispatch_queue_family_);
    dispatch_queue_.queue_family = &dispatch_queue_family_;
    runtime_.device = reinterpret_cast<iree_hal_device_t*>(&device_);
    runtime_.dispatch_queue = &dispatch_queue_;
  }

  // Immutable device-spec storage used by |device_|.
  DeviceSpecPtr device_spec_;

  // Stack HAL device exposing |device_spec_| through the real device API.
  FakeHalDevice device_ = {};

  // Stack family identity selecting the sole dispatch family in |device_spec_|.
  iree_hal_queue_family_t dispatch_queue_family_ = {};
  // Stack queue carrying |dispatch_queue_family_| through the runtime contract.
  iree_hal_queue_t dispatch_queue_ = {};

  // Runtime view passed through the production device-provider contract.
  loom_run_hal_runtime_t runtime_ = {};
};

TEST_F(AmdgpuDeviceProviderTest, PrefersExactDeviceTarget) {
  IREE_ASSERT_OK(Initialize(AmdgpuDeviceTargetSet::kGenericAndExact));
  loom_device_target_t target = {};
  IREE_ASSERT_OK(loom_amdgpu_device_provider.select_target(
      &loom_amdgpu_device_provider, &runtime_, iree_allocator_null(), &target));
  ExpectSelectedTarget(target, IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
                       IREE_SV("gfx1151"));
}

TEST_F(AmdgpuDeviceProviderTest, FallsBackToAdvertisedGenericTarget) {
  IREE_ASSERT_OK(Initialize(AmdgpuDeviceTargetSet::kGenericOnly));
  loom_device_target_t target = {};
  IREE_ASSERT_OK(loom_amdgpu_device_provider.select_target(
      &loom_amdgpu_device_provider, &runtime_, iree_allocator_system(),
      &target));
  ExpectSelectedTarget(target, IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
                       IREE_SV("gfx11-generic"));
}

TEST_F(AmdgpuDeviceProviderTest, RefinesAuthoredGenericTarget) {
  IREE_ASSERT_OK(Initialize(AmdgpuDeviceTargetSet::kGenericAndExact));
  const loom_amdgpu_target_facts_t requirement =
      MakeTargetRequirement("gfx11-generic");
  loom_device_target_t target = {};
  IREE_ASSERT_OK(SelectCompatibleTarget(requirement, &target));
  ExpectSelectedTarget(target, IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
                       IREE_SV("gfx1151"));
}

TEST_F(AmdgpuDeviceProviderTest, FallsBackToCompatibleGenericTarget) {
  IREE_ASSERT_OK(Initialize(AmdgpuDeviceTargetSet::kGenericOnly));
  const loom_amdgpu_target_facts_t requirement =
      MakeTargetRequirement("gfx11-generic");
  loom_device_target_t target = {};
  IREE_ASSERT_OK(SelectCompatibleTarget(requirement, &target));
  ExpectSelectedTarget(target, IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
                       IREE_SV("gfx11-generic"));
}

TEST_F(AmdgpuDeviceProviderTest, PreservesAuthoredExactTarget) {
  IREE_ASSERT_OK(Initialize(AmdgpuDeviceTargetSet::kGenericAndExact));
  const loom_amdgpu_target_facts_t requirement =
      MakeTargetRequirement("gfx1151");
  loom_device_target_t target = {};
  IREE_ASSERT_OK(SelectCompatibleTarget(requirement, &target));
  ExpectSelectedTarget(target, IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
                       IREE_SV("gfx1151"));
}

TEST_F(AmdgpuDeviceProviderTest, PreservesTargetOverlay) {
  const loom_amdgpu_target_info_t* target_info = LookupTarget("gfx1250-a0");
  loom_amdgpu_target_identity_t identity = {};
  loom_amdgpu_target_identity_initialize(target_info, &identity);
  char target_key_storage[128] = {};
  iree_string_view_t target_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_amdgpu_artifact_key_format(
      &identity, sizeof(target_key_storage), target_key_storage, &target_key));
  IREE_ASSERT_OK(Initialize(target_key, IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT));

  const loom_amdgpu_target_facts_t requirement =
      MakeTargetRequirement("gfx1250-a0");
  loom_device_target_t target = {};
  IREE_ASSERT_OK(SelectCompatibleTarget(requirement, &target));
  ExpectSelectedTarget(target, IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
                       IREE_SV("gfx1250-a0"));
  const loom_amdgpu_target_profile_t* profile =
      loom_amdgpu_target_profile_cast(target.artifact_target.target_profile);
  ASSERT_NE(profile, nullptr);
  EXPECT_EQ(profile->identity.target, target_info);
}

TEST_F(AmdgpuDeviceProviderTest, RejectsTargetOverlayMismatch) {
  IREE_ASSERT_OK(
      Initialize(IREE_SV("gfx1250"), IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT));
  const loom_amdgpu_target_facts_t requirement =
      MakeTargetRequirement("gfx1250-a0");
  loom_device_target_t target = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        SelectCompatibleTarget(requirement, &target));
  EXPECT_EQ(target.artifact_target.target_profile, nullptr);
  EXPECT_EQ(target.executable_target, nullptr);
}

TEST_F(AmdgpuDeviceProviderTest, RejectsIncompatibleAuthoredExactTarget) {
  IREE_ASSERT_OK(Initialize(AmdgpuDeviceTargetSet::kGenericAndExact));
  const loom_amdgpu_target_facts_t requirement =
      MakeTargetRequirement("gfx1150");
  loom_device_target_t target = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        SelectCompatibleTarget(requirement, &target));
  EXPECT_EQ(target.artifact_target.target_profile, nullptr);
  EXPECT_EQ(target.executable_target, nullptr);
}

TEST_F(AmdgpuDeviceProviderTest, RejectsTargetKindKeyMismatch) {
  IREE_ASSERT_OK(Initialize(IREE_SV("gfx11-generic"),
                            IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT));
  loom_device_target_t target = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_amdgpu_device_provider.select_target(
                            &loom_amdgpu_device_provider, &runtime_,
                            iree_allocator_system(), &target));
  EXPECT_EQ(target.artifact_target.target_profile, nullptr);
  EXPECT_EQ(target.executable_target, nullptr);
}

TEST_F(AmdgpuDeviceProviderTest, SkipsUnknownExactTarget) {
  IREE_ASSERT_OK(Initialize(AmdgpuDeviceTargetSet::kGenericAndUnknownExact));
  loom_device_target_t target = {};
  IREE_ASSERT_OK(loom_amdgpu_device_provider.select_target(
      &loom_amdgpu_device_provider, &runtime_, iree_allocator_system(),
      &target));
  ExpectSelectedTarget(target, IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
                       IREE_SV("gfx11-generic"));
}

TEST_F(AmdgpuDeviceProviderTest, RejectsOnlyUnknownTarget) {
  IREE_ASSERT_OK(
      Initialize(IREE_SV("gfx9999"), IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT));
  loom_device_target_t target = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        loom_amdgpu_device_provider.select_target(
                            &loom_amdgpu_device_provider, &runtime_,
                            iree_allocator_system(), &target));
  EXPECT_EQ(target.artifact_target.target_profile, nullptr);
  EXPECT_EQ(target.executable_target, nullptr);
}

}  // namespace
}  // namespace loom
