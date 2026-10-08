// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/cpu/iree_hal.h"

#include <memory>
#include <string>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/drivers/task/device_spec.h"
#include "iree/hal/utils/device_spec_builder.h"
#include "iree/testing/gtest.h"
#include "loomc/diagnostic.h"
#include "loomc/interop.h"
#include "loomc/result.h"
#include "loomc/target/configured.h"
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

iree_status_t CreateCpuDeviceSpec(bool include_cpu_facet,
                                  bool include_loader_target,
                                  DeviceSpecPtr* out_device_spec,
                                  uint64_t cpu_field0_bits = 0) {
  out_device_spec->reset();
  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status = iree_ok_status();

  std::vector<uint8_t> cpu_payload;
  if (include_cpu_facet) {
    const iree_hal_cpu_device_spec_t cpu_spec = {
        /*.cpu_data=*/
        {
            /*.architecture=*/IREE_CPU_ARCHITECTURE_X86_64,
            /*.fields=*/{cpu_field0_bits},
        },
        /*.flags=*/IREE_HAL_CPU_DEVICE_SPEC_FLAG_NONE,
    };
    cpu_payload.resize(iree_hal_cpu_device_spec_payload_size());
    status = iree_hal_cpu_device_spec_encode(
        &cpu_spec, iree_make_byte_span(cpu_payload.data(), cpu_payload.size()));
    if (iree_status_is_ok(status)) {
      const iree_hal_device_spec_facet_t facet = {
          /*.schema_id=*/
          iree_make_cstring_view(IREE_HAL_CPU_DEVICE_SPEC_SCHEMA_ID),
          /*.schema_version=*/IREE_HAL_CPU_DEVICE_SPEC_SCHEMA_VERSION,
          /*.payload=*/
          iree_make_const_byte_span(cpu_payload.data(), cpu_payload.size()),
      };
      status = iree_hal_device_spec_builder_add_facet(&builder, &facet);
    }
  }

  if (iree_status_is_ok(status) && include_loader_target) {
    const iree_hal_executable_target_t target = {
        /*.family=*/IREE_SV("cpu"),
        /*.target_key=*/IREE_SV("x86_64"),
        /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
        /*.priority=*/0,
        /*.physical_device_affinity=*/1,
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

TargetEnvironmentPtr CreateConfiguredTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  LOOMC_EXPECT_OK(loomc_target_environment_create_configured(
      loomc_allocator_system(), &target_environment));
  return TargetEnvironmentPtr(target_environment);
}

TargetProfilePtr SelectCpuTarget(
    loomc_target_environment_t* target_environment, FakeHalDevice* device,
    loomc_result_t** out_result,
    const iree_hal_executable_target_t** out_executable_target = nullptr,
    loomc_target_profile_t* requested_profile = nullptr) {
  const loomc_iree_hal_target_provider_t* providers[] = {
      loomc_cpu_iree_hal_target_provider(),
  };
  const loomc_iree_hal_target_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_IREE_HAL_TARGET_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("fake-cpu"),
      /*.device=*/reinterpret_cast<iree_hal_device_t*>(device),
      /*.physical_device_affinity=*/1,
      /*.target_profile=*/requested_profile,
      /*.providers=*/providers,
      /*.provider_count=*/IREE_ARRAYSIZE(providers),
  };
  loomc_iree_hal_target_selection_t selection = {};
  LOOMC_EXPECT_OK(loomc_target_select_iree_hal(target_environment, &options,
                                               loomc_allocator_system(),
                                               &selection, out_result));
  if (out_executable_target != nullptr) {
    *out_executable_target = selection.executable_target;
  }
  return TargetProfilePtr(selection.target_profile);
}

TEST(LoomcCpuIreeHalTargetTest, PreservesExplicitNativeProfileIdentity) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateCpuDeviceSpec(/*include_cpu_facet=*/true,
                                     /*include_loader_target=*/true,
                                     &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateConfiguredTargetEnvironment();
  loomc_target_profile_t* selected_profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_select(
      target_environment.get(), loomc_make_cstring_view("x86:scalar"),
      loomc_allocator_system(), &selected_profile));
  TargetProfilePtr requested_profile(selected_profile);

  loomc_result_t* result = nullptr;
  TargetProfilePtr profile = SelectCpuTarget(
      target_environment.get(), &device, &result,
      /*out_executable_target=*/nullptr, requested_profile.get());
  ResultPtr result_ptr(result);

  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  EXPECT_EQ(profile.get(), requested_profile.get());
}

TEST(LoomcCpuIreeHalTargetTest, SelectsNativeProfileAndLoaderTogether) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateCpuDeviceSpec(/*include_cpu_facet=*/true,
                                     /*include_loader_target=*/true,
                                     &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateConfiguredTargetEnvironment();
  loomc_result_t* result = nullptr;
  const iree_hal_executable_target_t* executable_target = nullptr;
  TargetProfilePtr profile = SelectCpuTarget(target_environment.get(), &device,
                                             &result, &executable_target);
  ResultPtr result_ptr(result);

  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  ASSERT_NE(profile.get(), nullptr);
  ASSERT_NE(executable_target, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(executable_target->family, IREE_SV("cpu")));
  EXPECT_TRUE(
      iree_string_view_equal(executable_target->target_key, IREE_SV("x86_64")));
  EXPECT_EQ(executable_target->kind, IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC);

  const loom_target_profile_t* native_profile =
      loomc_target_profile_get_interop_view(profile.get());
  ASSERT_NE(native_profile, nullptr);
  ASSERT_NE(native_profile->target_bundle, nullptr);
  ASSERT_NE(native_profile->target_bundle->snapshot, nullptr);
  EXPECT_EQ(native_profile->target_bundle->snapshot->default_pointer_bitwidth,
            64u);
}

TEST(LoomcCpuIreeHalTargetTest, SelectsStrongestProfileFromDeviceFacts) {
  const uint64_t cpu_field0_bits =
      IREE_CPU_DATA0_X86_64_AVX | IREE_CPU_DATA0_X86_64_FMA |
      IREE_CPU_DATA0_X86_64_AVX2 | IREE_CPU_DATA0_X86_64_AVX512F |
      IREE_CPU_DATA0_X86_64_AVX512VL | IREE_CPU_DATA0_X86_64_AVX512DQ |
      IREE_CPU_DATA0_X86_64_AVX512BW;
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateCpuDeviceSpec(/*include_cpu_facet=*/true,
                                     /*include_loader_target=*/true,
                                     &device_spec, cpu_field0_bits));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateConfiguredTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile =
      SelectCpuTarget(target_environment.get(), &device, &result);
  ResultPtr result_ptr(result);

  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  ASSERT_NE(profile.get(), nullptr);
  const loom_target_profile_t* native_profile =
      loomc_target_profile_get_interop_view(profile.get());
  ASSERT_NE(native_profile, nullptr);
  ASSERT_NE(native_profile->target_bundle, nullptr);
  EXPECT_TRUE(iree_string_view_equal(native_profile->target_bundle->name,
                                     IREE_SV("x86-avx512")));
}

TEST(LoomcCpuIreeHalTargetTest, ReportsMissingLoaderAsTargetDiagnostic) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateCpuDeviceSpec(/*include_cpu_facet=*/true,
                                     /*include_loader_target=*/false,
                                     &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateConfiguredTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile =
      SelectCpuTarget(target_environment.get(), &device, &result);
  ResultPtr result_ptr(result);

  EXPECT_EQ(profile.get(), nullptr);
  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result_ptr.get()));
  ASSERT_GE(loomc_result_diagnostic_count(result_ptr.get()), 1u);
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(ToString(diagnostic->code), "CPU/IREE_HAL");
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("no compatible native CPU loader"));
}

TEST(LoomcCpuIreeHalTargetTest, DoesNotClaimDevicesWithoutCpuFacts) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateCpuDeviceSpec(/*include_cpu_facet=*/false,
                                     /*include_loader_target=*/true,
                                     &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateConfiguredTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile =
      SelectCpuTarget(target_environment.get(), &device, &result);
  ResultPtr result_ptr(result);

  EXPECT_EQ(profile.get(), nullptr);
  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result_ptr.get()));
  ASSERT_GE(loomc_result_diagnostic_count(result_ptr.get()), 1u);
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(ToString(diagnostic->code), "IREE_HAL/TARGET");
}

}  // namespace
