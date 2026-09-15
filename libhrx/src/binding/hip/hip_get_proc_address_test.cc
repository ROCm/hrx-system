// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <thread>

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/testing/gtest.h"

namespace {

enum : uint64_t {
  kDefaultStream = 0,
  kLegacyStream = 1,
  kPerThreadDefaultStream = 2,
};

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipHalDeinitFn = hipError_t (*)(void);
using HipGetProcAddressFn = hipError_t (*)(const char* symbol, void** function,
                                           int hip_version, uint64_t flags,
                                           void* symbol_status);
using HipGetDriverEntryPointFn = hipError_t (*)(const char* symbol,
                                                void** function, uint64_t flags,
                                                void* status);
using HipGetDevicePropertiesR0000Fn = hipError_t (*)(hipDeviceProp_tR0000* prop,
                                                     int device);
using HipGetLastErrorFn = hipError_t (*)(void);
using HipPeekAtLastErrorFn = hipError_t (*)(void);
using HipDeviceComputeCapabilityFn = hipError_t (*)(int* major, int* minor,
                                                    hipDevice_t device);

struct HipRuntimeApi {
  // Initializes the exact runtime instance loaded by the fixture.
  HipInitFn init = nullptr;
  // Deinitializes the runtime instance before its DSO is closed.
  HipHalDeinitFn hal_deinit = nullptr;
  // Resolves public runtime functions with version and stream semantics.
  HipGetProcAddressFn get_proc_address = nullptr;
  // Per-thread-default-stream variant of get_proc_address.
  HipGetProcAddressFn get_proc_address_spt = nullptr;
  // Resolves public driver entry points with stream semantics.
  HipGetDriverEntryPointFn get_driver_entry_point = nullptr;
  // Per-thread-default-stream variant of get_driver_entry_point.
  HipGetDriverEntryPointFn get_driver_entry_point_spt = nullptr;
  // Returns and clears the calling thread's last HIP error.
  HipGetLastErrorFn get_last_error = nullptr;
  // Returns without clearing the calling thread's last HIP error.
  HipPeekAtLastErrorFn peek_at_last_error = nullptr;
  // Queries the selected device's compute capability.
  HipDeviceComputeCapabilityFn device_compute_capability = nullptr;
};

class HipGetProcAddressTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!dso_.is_open()) {
      ASSERT_TRUE(dso_.Open()) << dso_.error();
      api_.init = dso_.Resolve<HipInitFn>("hipInit");
      api_.hal_deinit = dso_.Resolve<HipHalDeinitFn>("hipHALDeinit");
      api_.get_proc_address =
          dso_.Resolve<HipGetProcAddressFn>("hipGetProcAddress");
      api_.get_proc_address_spt =
          dso_.Resolve<HipGetProcAddressFn>("hipGetProcAddress_spt");
      api_.get_driver_entry_point =
          dso_.Resolve<HipGetDriverEntryPointFn>("hipGetDriverEntryPoint");
      api_.get_driver_entry_point_spt =
          dso_.Resolve<HipGetDriverEntryPointFn>("hipGetDriverEntryPoint_spt");
      api_.get_last_error = dso_.Resolve<HipGetLastErrorFn>("hipGetLastError");
      api_.peek_at_last_error =
          dso_.Resolve<HipPeekAtLastErrorFn>("hipPeekAtLastError");
      api_.device_compute_capability =
          dso_.Resolve<HipDeviceComputeCapabilityFn>(
              "hipDeviceComputeCapability");
    }

    ASSERT_NE(nullptr, api_.init);
    ASSERT_NE(nullptr, api_.hal_deinit);
    ASSERT_NE(nullptr, api_.get_proc_address);
    ASSERT_NE(nullptr, api_.get_proc_address_spt);
    ASSERT_NE(nullptr, api_.get_driver_entry_point);
    ASSERT_NE(nullptr, api_.get_driver_entry_point_spt);
    ASSERT_NE(nullptr, api_.get_last_error);
    ASSERT_NE(nullptr, api_.peek_at_last_error);
    ASSERT_NE(nullptr, api_.device_compute_capability);
    ASSERT_EQ(hipSuccess, api_.init(/*flags=*/0));
    ASSERT_EQ(hipSuccess, api_.get_last_error());
  }

  static void TearDownTestSuite() {
    if (!dso_.is_open()) return;
    ASSERT_NE(nullptr, api_.hal_deinit);
    EXPECT_EQ(hipSuccess, api_.hal_deinit());
    api_ = {};
    EXPECT_TRUE(dso_.Close()) << dso_.error();
  }

  void ExpectLookup(HipGetProcAddressFn lookup, const char* symbol,
                    int hip_version, uint64_t flags,
                    const char* expected_symbol) {
    void* function = nullptr;
    int symbol_status = -1;
    ASSERT_EQ(hipSuccess,
              lookup(symbol, &function, hip_version, flags, &symbol_status));
    EXPECT_EQ(dso_.Resolve<void*>(expected_symbol), function);
    EXPECT_EQ(0, symbol_status);
  }

  // Exact DSO owner shared by every test in this fixture.
  static hrx::hip::testing::HipDso dso_;
  // Exact public API resolved from dso_.
  static HipRuntimeApi api_;
};

hrx::hip::testing::HipDso HipGetProcAddressTest::dso_;
HipRuntimeApi HipGetProcAddressTest::api_;

TEST_F(HipGetProcAddressTest, ResolvesVersionedDevicePropertyAbis) {
  ExpectLookup(api_.get_proc_address, "hipGetDeviceProperties",
               /*hip_version=*/599, kDefaultStream,
               "hipGetDevicePropertiesR0000");
  ExpectLookup(api_.get_proc_address, "hipGetDeviceProperties",
               /*hip_version=*/600, kDefaultStream,
               "hipGetDevicePropertiesR0600");
  ExpectLookup(api_.get_proc_address_spt, "hipChooseDevice",
               /*hip_version=*/599, kDefaultStream, "hipChooseDeviceR0000");
  ExpectLookup(api_.get_proc_address_spt, "hipChooseDevice",
               /*hip_version=*/600, kDefaultStream, "hipChooseDeviceR0600");

  struct PropertyCanary {
    // Legacy ABI object passed to the unsuffixed facade.
    hipDeviceProp_tR0000 properties;
    // Detects writes beyond the legacy object boundary.
    uint64_t canary;
  } property_canary = {/*properties=*/{},
                       /*canary=*/UINT64_C(0x123456789ABCDEF0)};
  auto get_properties =
      dso_.Resolve<HipGetDevicePropertiesR0000Fn>("hipGetDeviceProperties");
  ASSERT_NE(nullptr, get_properties);
  ASSERT_EQ(hipSuccess, get_properties(&property_canary.properties, 0));
  EXPECT_EQ(UINT64_C(0x123456789ABCDEF0), property_canary.canary);
  EXPECT_GT(property_canary.properties.gcnArch, 0);
}

TEST_F(HipGetProcAddressTest, SptLookupPreservesExplicitStreamMode) {
  ExpectLookup(api_.get_proc_address_spt, "hipStreamSynchronize",
               /*hip_version=*/600, kDefaultStream, "hipStreamSynchronize_spt");
  ExpectLookup(api_.get_proc_address_spt, "hipStreamSynchronize",
               /*hip_version=*/600, kLegacyStream, "hipStreamSynchronize");
  ExpectLookup(api_.get_proc_address_spt, "hipStreamSynchronize",
               /*hip_version=*/600, kPerThreadDefaultStream,
               "hipStreamSynchronize_spt");

  void* function = reinterpret_cast<void*>(uintptr_t{1});
  int symbol_status = -1;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.get_proc_address_spt("hipStreamSynchronize", &function,
                                      /*hip_version=*/600, /*flags=*/3,
                                      &symbol_status));
  EXPECT_EQ(nullptr, function);
  EXPECT_EQ(1, symbol_status);
  EXPECT_EQ(hipErrorInvalidValue, api_.peek_at_last_error());
  EXPECT_EQ(hipErrorInvalidValue, api_.get_last_error());
}

TEST_F(HipGetProcAddressTest, MissingDriverSymbolsPublishDetailedFailure) {
  const HipGetDriverEntryPointFn lookups[] = {
      api_.get_driver_entry_point,
      api_.get_driver_entry_point_spt,
  };
  for (HipGetDriverEntryPointFn lookup : lookups) {
    ASSERT_EQ(hipSuccess, api_.get_last_error());
    void* function = reinterpret_cast<void*>(uintptr_t{1});
    int symbol_status = -1;
    EXPECT_EQ(hipErrorInvalidValue,
              lookup("hipSymbolThatDoesNotExist", &function, kDefaultStream,
                     &symbol_status));
    EXPECT_EQ(nullptr, function);
    EXPECT_EQ(1, symbol_status);
    EXPECT_EQ(hipErrorInvalidValue, api_.peek_at_last_error());
    EXPECT_EQ(hipErrorInvalidValue, api_.get_last_error());
  }
}

TEST_F(HipGetProcAddressTest, MissingProcSymbolsPublishDetailedFailure) {
  const HipGetProcAddressFn lookups[] = {
      api_.get_proc_address,
      api_.get_proc_address_spt,
  };
  for (HipGetProcAddressFn lookup : lookups) {
    ASSERT_EQ(hipSuccess, api_.get_last_error());
    void* function = reinterpret_cast<void*>(uintptr_t{1});
    int symbol_status = -1;
    EXPECT_EQ(hipErrorInvalidValue,
              lookup("hipSymbolThatDoesNotExist", &function,
                     /*hip_version=*/600, kDefaultStream, &symbol_status));
    EXPECT_EQ(nullptr, function);
    EXPECT_EQ(1, symbol_status);
    EXPECT_EQ(hipErrorInvalidValue, api_.peek_at_last_error());
    EXPECT_EQ(hipErrorInvalidValue, api_.get_last_error());
  }
}

TEST_F(HipGetProcAddressTest, NonFatalErrorsRemainThreadLocal) {
  hipError_t other_thread_result = hipSuccess;
  hipError_t other_thread_last_error = hipSuccess;
  std::thread other_thread([&] {
    int minor = 0;
    other_thread_result =
        api_.device_compute_capability(nullptr, &minor, /*device=*/0);
    other_thread_last_error = api_.peek_at_last_error();
  });
  other_thread.join();

  EXPECT_EQ(hipErrorInvalidValue, other_thread_result);
  EXPECT_EQ(hipErrorInvalidValue, other_thread_last_error);
  EXPECT_EQ(hipSuccess, api_.peek_at_last_error());
}

}  // namespace
