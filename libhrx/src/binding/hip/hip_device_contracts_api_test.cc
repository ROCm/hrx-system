// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <climits>
#include <thread>

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/testing/gtest.h"

namespace {

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipHalDeinitFn = hipError_t (*)(void);
using HipGetDeviceCountFn = hipError_t (*)(int* count);
using HipGetDeviceFn = hipError_t (*)(int* device);
using HipSetDeviceFn = hipError_t (*)(int device);
using HipSetValidDevicesFn = hipError_t (*)(int* devices, int count);
using HipGetDeviceFlagsFn = hipError_t (*)(unsigned int* flags);
using HipSetDeviceFlagsFn = hipError_t (*)(unsigned int flags);
using HipDeviceGetLimitFn = hipError_t (*)(size_t* value, hipLimit_t limit);
using HipDeviceSetLimitFn = hipError_t (*)(hipLimit_t limit, size_t value);
using HipDevicePrimaryCtxGetStateFn = hipError_t (*)(hipDevice_t device,
                                                     unsigned int* flags,
                                                     int* active);
using HipDeviceResetFn = hipError_t (*)(void);
using HipDeviceGetPCIBusIdFn = hipError_t (*)(char* pci_bus_id, int length,
                                              int device);
using HipDeviceGetByPCIBusIdFn = hipError_t (*)(int* device,
                                                const char* pci_bus_id);
using HipLaunchCooperativeKernelMultiDeviceFn = hipError_t (*)(
    hipLaunchParams* launch_params, int device_count, unsigned int flags);
using HipGetLastErrorFn = hipError_t (*)(void);
using HipExtGetLastErrorFn = hipError_t (*)(void);
using HipPeekAtLastErrorFn = hipError_t (*)(void);

struct HipRuntimeApi {
  // Initializes the exact runtime instance loaded by the fixture.
  HipInitFn init = nullptr;
  // Deinitializes the runtime instance before its DSO is closed.
  HipHalDeinitFn hal_deinit = nullptr;
  // Reports the number of devices visible to the runtime.
  HipGetDeviceCountFn get_device_count = nullptr;
  // Reports the calling thread's selected device.
  HipGetDeviceFn get_device = nullptr;
  // Explicitly selects the calling thread's current device.
  HipSetDeviceFn set_device = nullptr;
  // Selects a preferred device list for the calling thread.
  HipSetValidDevicesFn set_valid_devices = nullptr;
  // Reports the current device's public scheduling flags.
  HipGetDeviceFlagsFn get_device_flags = nullptr;
  // Replaces the current device's public scheduling flags.
  HipSetDeviceFlagsFn set_device_flags = nullptr;
  // Reports one runtime resource limit.
  HipDeviceGetLimitFn device_get_limit = nullptr;
  // Replaces one runtime resource limit.
  HipDeviceSetLimitFn device_set_limit = nullptr;
  // Reports a primary context's flags and activation state.
  HipDevicePrimaryCtxGetStateFn device_primary_context_get_state = nullptr;
  // Resets the calling thread's current device.
  HipDeviceResetFn device_reset = nullptr;
  // Formats a device's PCI identity.
  HipDeviceGetPCIBusIdFn device_get_pci_bus_id = nullptr;
  // Resolves a formatted PCI identity to a device ordinal.
  HipDeviceGetByPCIBusIdFn device_get_by_pci_bus_id = nullptr;
  // Launches one cooperative kernel on each listed device.
  HipLaunchCooperativeKernelMultiDeviceFn
      launch_cooperative_kernel_multi_device = nullptr;
  // Returns and clears the calling thread's last HIP error.
  HipGetLastErrorFn get_last_error = nullptr;
  // Returns and clears the calling thread's most recent HIP call result.
  HipExtGetLastErrorFn ext_get_last_error = nullptr;
  // Returns without clearing the calling thread's last HIP error.
  HipPeekAtLastErrorFn peek_at_last_error = nullptr;
};

class HipDeviceContractsApiTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!dso_.is_open()) {
      ASSERT_TRUE(dso_.Open()) << dso_.error();
      api_.init = dso_.Resolve<HipInitFn>("hipInit");
      api_.hal_deinit = dso_.Resolve<HipHalDeinitFn>("hipHALDeinit");
      api_.get_device_count =
          dso_.Resolve<HipGetDeviceCountFn>("hipGetDeviceCount");
      api_.get_device = dso_.Resolve<HipGetDeviceFn>("hipGetDevice");
      api_.set_device = dso_.Resolve<HipSetDeviceFn>("hipSetDevice");
      api_.set_valid_devices =
          dso_.Resolve<HipSetValidDevicesFn>("hipSetValidDevices");
      api_.get_device_flags =
          dso_.Resolve<HipGetDeviceFlagsFn>("hipGetDeviceFlags");
      api_.set_device_flags =
          dso_.Resolve<HipSetDeviceFlagsFn>("hipSetDeviceFlags");
      api_.device_get_limit =
          dso_.Resolve<HipDeviceGetLimitFn>("hipDeviceGetLimit");
      api_.device_set_limit =
          dso_.Resolve<HipDeviceSetLimitFn>("hipDeviceSetLimit");
      api_.device_primary_context_get_state =
          dso_.Resolve<HipDevicePrimaryCtxGetStateFn>(
              "hipDevicePrimaryCtxGetState");
      api_.device_reset = dso_.Resolve<HipDeviceResetFn>("hipDeviceReset");
      api_.device_get_pci_bus_id =
          dso_.Resolve<HipDeviceGetPCIBusIdFn>("hipDeviceGetPCIBusId");
      api_.device_get_by_pci_bus_id =
          dso_.Resolve<HipDeviceGetByPCIBusIdFn>("hipDeviceGetByPCIBusId");
      api_.launch_cooperative_kernel_multi_device =
          dso_.Resolve<HipLaunchCooperativeKernelMultiDeviceFn>(
              "hipLaunchCooperativeKernelMultiDevice");
      api_.get_last_error = dso_.Resolve<HipGetLastErrorFn>("hipGetLastError");
      api_.ext_get_last_error =
          dso_.Resolve<HipExtGetLastErrorFn>("hipExtGetLastError");
      api_.peek_at_last_error =
          dso_.Resolve<HipPeekAtLastErrorFn>("hipPeekAtLastError");
    }

    ASSERT_NE(nullptr, api_.init);
    ASSERT_NE(nullptr, api_.hal_deinit);
    ASSERT_NE(nullptr, api_.get_device_count);
    ASSERT_NE(nullptr, api_.get_device);
    ASSERT_NE(nullptr, api_.set_device);
    ASSERT_NE(nullptr, api_.set_valid_devices);
    ASSERT_NE(nullptr, api_.get_device_flags);
    ASSERT_NE(nullptr, api_.set_device_flags);
    ASSERT_NE(nullptr, api_.device_get_limit);
    ASSERT_NE(nullptr, api_.device_set_limit);
    ASSERT_NE(nullptr, api_.device_primary_context_get_state);
    ASSERT_NE(nullptr, api_.device_reset);
    ASSERT_NE(nullptr, api_.device_get_pci_bus_id);
    ASSERT_NE(nullptr, api_.device_get_by_pci_bus_id);
    ASSERT_NE(nullptr, api_.launch_cooperative_kernel_multi_device);
    ASSERT_NE(nullptr, api_.get_last_error);
    ASSERT_NE(nullptr, api_.ext_get_last_error);
    ASSERT_NE(nullptr, api_.peek_at_last_error);
    ASSERT_EQ(hipSuccess, api_.init(/*flags=*/0));
    // Test cases share the runner thread but own independent last-error state.
    (void)api_.get_last_error();
    (void)api_.ext_get_last_error();
    ASSERT_EQ(hipSuccess, api_.peek_at_last_error());
  }

  static void TearDownTestSuite() {
    if (!dso_.is_open()) return;
    ASSERT_NE(nullptr, api_.hal_deinit);
    EXPECT_EQ(hipSuccess, api_.hal_deinit());
    api_ = {};
    EXPECT_TRUE(dso_.Close()) << dso_.error();
  }

  // Exact DSO owner shared by every test in this fixture.
  static hrx::hip::testing::HipDso dso_;
  // Exact public API resolved from dso_.
  static HipRuntimeApi api_;
};

hrx::hip::testing::HipDso HipDeviceContractsApiTest::dso_;
HipRuntimeApi HipDeviceContractsApiTest::api_;

TEST_F(HipDeviceContractsApiTest, DeviceFlagsRoundTripUnderConcurrentAccess) {
  unsigned int flags = UINT_MAX;
  ASSERT_EQ(hipSuccess, api_.get_device_flags(&flags));
  EXPECT_EQ(hipDeviceScheduleAuto, flags);

  ASSERT_EQ(hipSuccess,
            api_.set_device_flags(hipDeviceScheduleYield | hipDeviceMapHost |
                                  hipDeviceLmemResizeToMax));
  ASSERT_EQ(hipSuccess, api_.get_device_flags(&flags));
  EXPECT_EQ(hipDeviceScheduleYield, flags);

  std::atomic<bool> start{false};
  std::atomic<bool> invalid_value_observed{false};
  std::thread setter([&] {
    while (!start.load(std::memory_order_acquire)) {
    }
    for (int i = 0; i < 2000; ++i) {
      const unsigned int next_flags =
          i % 2 == 0 ? hipDeviceScheduleSpin : hipDeviceScheduleYield;
      if (api_.set_device_flags(next_flags) != hipSuccess) {
        invalid_value_observed.store(true, std::memory_order_release);
      }
    }
  });
  std::thread getter([&] {
    start.store(true, std::memory_order_release);
    for (int i = 0; i < 2000; ++i) {
      unsigned int observed_flags = UINT_MAX;
      if (api_.get_device_flags(&observed_flags) != hipSuccess ||
          (observed_flags != hipDeviceScheduleSpin &&
           observed_flags != hipDeviceScheduleYield)) {
        invalid_value_observed.store(true, std::memory_order_release);
      }
    }
  });
  setter.join();
  getter.join();
  EXPECT_FALSE(invalid_value_observed.load(std::memory_order_acquire));
  EXPECT_EQ(hipSuccess, api_.set_device_flags(hipDeviceScheduleAuto));
}

TEST_F(HipDeviceContractsApiTest, DeviceResetRestoresSchedulingDefaults) {
  ASSERT_EQ(hipSuccess, api_.set_device_flags(hipDeviceScheduleBlockingSync));
  unsigned int flags = UINT_MAX;
  ASSERT_EQ(hipSuccess, api_.get_device_flags(&flags));
  ASSERT_EQ(hipDeviceScheduleBlockingSync, flags);

  ASSERT_EQ(hipSuccess, api_.device_reset());
  ASSERT_EQ(hipSuccess, api_.get_device_flags(&flags));
  EXPECT_EQ(hipDeviceScheduleAuto, flags);
}

TEST_F(HipDeviceContractsApiTest, EmptyValidDeviceListPreservesSelection) {
  int device_count = 0;
  ASSERT_EQ(hipSuccess, api_.get_device_count(&device_count));
  if (device_count < 2) {
    GTEST_SKIP() << "requires two independently selectable devices";
  }

  int selected_device = -1;
  hipError_t result = hipSuccess;
  std::thread selection_thread([&] {
    int preferred_device = device_count - 1;
    result = api_.set_valid_devices(&preferred_device, /*count=*/1);
    if (result != hipSuccess) return;
    result = api_.set_valid_devices(/*devices=*/nullptr, /*count=*/0);
    if (result != hipSuccess) return;
    result = api_.get_device(&selected_device);
  });
  selection_thread.join();
  ASSERT_EQ(hipSuccess, result);
  EXPECT_EQ(device_count - 1, selected_device);
}

TEST_F(HipDeviceContractsApiTest,
       ValidDevicePreferenceDoesNotActivatePrimaryContext) {
  int device_count = 0;
  ASSERT_EQ(hipSuccess, api_.get_device_count(&device_count));
  const int preferred_device = device_count - 1;
  ASSERT_EQ(hipSuccess, api_.set_device(preferred_device));
  ASSERT_EQ(hipSuccess, api_.device_reset());

  int active = -1;
  hipError_t result = hipSuccess;
  std::thread selection_thread([&] {
    int preferred_device_copy = preferred_device;
    result = api_.set_valid_devices(&preferred_device_copy, /*count=*/1);
    if (result != hipSuccess) return;
    result = api_.device_primary_context_get_state(preferred_device,
                                                   /*flags=*/nullptr, &active);
  });
  selection_thread.join();
  ASSERT_EQ(hipSuccess, result);
  EXPECT_EQ(0, active);
}

TEST_F(HipDeviceContractsApiTest,
       ValidDevicePreferenceResetsAcrossRuntimeGenerations) {
  int device_count = 0;
  ASSERT_EQ(hipSuccess, api_.get_device_count(&device_count));
  if (device_count < 2) {
    GTEST_SKIP() << "requires two independently selectable devices";
  }

  std::atomic<int> phase{0};
  hipError_t result = hipSuccess;
  int selected_device = -1;
  std::thread selection_thread([&] {
    result = api_.set_device(/*device=*/0);
    phase.store(1, std::memory_order_release);
    while (phase.load(std::memory_order_acquire) != 2) {
    }
    if (result != hipSuccess) return;
    int preferred_device = 1;
    result = api_.set_valid_devices(&preferred_device, /*count=*/1);
    if (result != hipSuccess) return;
    result = api_.get_device(&selected_device);
  });

  while (phase.load(std::memory_order_acquire) != 1) {
  }
  const hipError_t deinit_result = api_.hal_deinit();
  const hipError_t init_result =
      deinit_result == hipSuccess ? api_.init(/*flags=*/0) : deinit_result;
  phase.store(2, std::memory_order_release);
  selection_thread.join();

  ASSERT_EQ(hipSuccess, deinit_result);
  ASSERT_EQ(hipSuccess, init_result);
  ASSERT_EQ(hipSuccess, result);
  EXPECT_EQ(1, selected_device);
}

TEST_F(HipDeviceContractsApiTest, PciIdentifiersRoundTripForEveryDevice) {
  int device_count = 0;
  ASSERT_EQ(hipSuccess, api_.get_device_count(&device_count));
  for (int device = 0; device < device_count; ++device) {
    char pci_bus_id[32] = {};
    ASSERT_EQ(hipSuccess, api_.device_get_pci_bus_id(
                              pci_bus_id, sizeof(pci_bus_id), device));
    int resolved_device = -1;
    ASSERT_EQ(hipSuccess,
              api_.device_get_by_pci_bus_id(&resolved_device, pci_bus_id));
    EXPECT_EQ(device, resolved_device);
  }
}

TEST_F(HipDeviceContractsApiTest, LimitValidationCoversTheDeclaredDomain) {
  size_t value = SIZE_MAX;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.device_get_limit(&value, static_cast<hipLimit_t>(-1)));
  EXPECT_EQ(SIZE_MAX, value);
  EXPECT_EQ(hipErrorInvalidValue,
            api_.device_get_limit(&value,
                                  static_cast<hipLimit_t>(hipLimitRange + 1)));
  EXPECT_EQ(SIZE_MAX, value);
  EXPECT_EQ(hipErrorInvalidValue,
            api_.device_get_limit(&value, static_cast<hipLimit_t>(0x1000)));
  EXPECT_EQ(SIZE_MAX, value);
  EXPECT_EQ(hipErrorInvalidValue,
            api_.device_set_limit(static_cast<hipLimit_t>(hipLimitRange + 1),
                                  /*value=*/0));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.device_set_limit(static_cast<hipLimit_t>(0x1002),
                                  /*value=*/0));
}

TEST_F(HipDeviceContractsApiTest, UnsupportedFunctionsPublishLastError) {
  ASSERT_EQ(hipSuccess, api_.get_last_error());
  EXPECT_EQ(hipErrorNotSupported,
            api_.launch_cooperative_kernel_multi_device(
                /*launch_params=*/nullptr, /*device_count=*/0, /*flags=*/0));
  EXPECT_EQ(hipErrorNotSupported, api_.peek_at_last_error());
  EXPECT_EQ(hipErrorNotSupported, api_.get_last_error());
}

TEST_F(HipDeviceContractsApiTest, CommandAndOrdinaryErrorsRemainDistinct) {
  ASSERT_EQ(hipSuccess, api_.get_last_error());
  ASSERT_EQ(hipSuccess, api_.ext_get_last_error());

  EXPECT_EQ(hipErrorNotSupported,
            api_.launch_cooperative_kernel_multi_device(
                /*launch_params=*/nullptr, /*device_count=*/0, /*flags=*/0));
  EXPECT_EQ(hipErrorNotSupported, api_.ext_get_last_error());

  int device_count = 0;
  ASSERT_EQ(hipSuccess, api_.get_device_count(&device_count));
  EXPECT_GT(device_count, 0);
  EXPECT_EQ(hipSuccess, api_.ext_get_last_error());

  EXPECT_EQ(hipErrorNotSupported, api_.peek_at_last_error());
  EXPECT_EQ(hipErrorNotSupported, api_.get_last_error());
}

}  // namespace
