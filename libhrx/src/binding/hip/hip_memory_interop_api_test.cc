// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <cstring>

#include "api.h"
#include "hip_dso_test_util.h"
#include "iree/testing/gtest.h"

namespace {

class HipMemoryInteropApiTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    ASSERT_TRUE(dso_.Open()) << dso_.error();
    const auto initialize = dso_.Resolve<decltype(&hipInit)>("hipInit");
    ASSERT_NE(initialize, nullptr);
    ASSERT_EQ(hipSuccess, initialize(0));
  }

  static void TearDownTestSuite() {
    if (dso_.is_open()) {
      const auto deinitialize =
          dso_.Resolve<decltype(&hipHALDeinit)>("hipHALDeinit");
      ASSERT_NE(deinitialize, nullptr);
      EXPECT_EQ(hipSuccess, deinitialize());
      EXPECT_TRUE(dso_.Close()) << dso_.error();
    }
  }

  void SetUp() override {
    allocate_ = dso_.Resolve<decltype(&hipMalloc)>("hipMalloc");
    allocate_async_ = dso_.Resolve<decltype(&hipMallocAsync)>("hipMallocAsync");
    free_ = dso_.Resolve<decltype(&hipFree)>("hipFree");
    allocate_host_ = dso_.Resolve<decltype(&hipHostAlloc)>("hipHostAlloc");
    free_host_ = dso_.Resolve<decltype(&hipFreeHost)>("hipFreeHost");
    register_host_ =
        dso_.Resolve<decltype(&hipHostRegister)>("hipHostRegister");
    unregister_host_ =
        dso_.Resolve<decltype(&hipHostUnregister)>("hipHostUnregister");
    host_device_pointer_ = dso_.Resolve<decltype(&hipHostGetDevicePointer)>(
        "hipHostGetDevicePointer");
    address_range_ =
        dso_.Resolve<decltype(&hipMemGetAddressRange)>("hipMemGetAddressRange");
    copy_ = dso_.Resolve<decltype(&hipMemcpy)>("hipMemcpy");
    synchronize_ =
        dso_.Resolve<decltype(&hipDeviceSynchronize)>("hipDeviceSynchronize");
    ASSERT_NE(allocate_, nullptr);
    ASSERT_NE(allocate_async_, nullptr);
    ASSERT_NE(free_, nullptr);
    ASSERT_NE(allocate_host_, nullptr);
    ASSERT_NE(free_host_, nullptr);
    ASSERT_NE(register_host_, nullptr);
    ASSERT_NE(unregister_host_, nullptr);
    ASSERT_NE(host_device_pointer_, nullptr);
    ASSERT_NE(address_range_, nullptr);
    ASSERT_NE(copy_, nullptr);
    ASSERT_NE(synchronize_, nullptr);
    ready_ = true;
  }

  void TearDown() override {
    if (ready_) {
      EXPECT_EQ(hipSuccess, synchronize_());
      if (device_memory_) {
        EXPECT_EQ(hipSuccess, free_(device_memory_));
      }
      if (host_memory_) {
        EXPECT_EQ(hipSuccess, free_host_(host_memory_));
      }
      if (registered_) {
        EXPECT_EQ(hipSuccess, unregister_host_(storage_.data()));
      }
    }
  }

  void CheckHostRange(void* host_pointer) {
    constexpr size_t kOffset = 13;
    constexpr size_t kLength = 257;
    auto* host_bytes = static_cast<uint8_t*>(host_pointer);
    for (size_t i = 0; i < storage_.size(); ++i) {
      host_bytes[i] = i * 17 + 3;
    }
    std::array<uint8_t, 4096> expected;
    memcpy(expected.data(), host_bytes, expected.size());

    hipDeviceptr_t base = nullptr;
    ASSERT_EQ(hipSuccess, host_device_pointer_(&base, host_pointer, 0));
    ASSERT_NE(base, nullptr);
    hipDeviceptr_t interior = nullptr;
    ASSERT_EQ(hipSuccess,
              host_device_pointer_(&interior, host_bytes + kOffset, 0));
    EXPECT_EQ(static_cast<uint8_t*>(base) + kOffset, interior);
    EXPECT_EQ(0, memcmp(expected.data(), host_bytes, expected.size()));

    ASSERT_EQ(hipSuccess, allocate_(&device_memory_, kLength));
    ASSERT_EQ(hipSuccess, copy_(device_memory_, interior, kLength,
                                hipMemcpyDeviceToDevice));
    std::array<uint8_t, kLength> result = {};
    ASSERT_EQ(hipSuccess, copy_(result.data(), device_memory_, kLength,
                                hipMemcpyDeviceToHost));
    EXPECT_EQ(0, memcmp(result.data(), expected.data() + kOffset, kLength));

    result.fill(0xA7);
    ASSERT_EQ(hipSuccess, copy_(device_memory_, result.data(), kLength,
                                hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, copy_(interior, device_memory_, kLength,
                                hipMemcpyDeviceToDevice));
    ASSERT_EQ(hipSuccess, synchronize_());
    memset(expected.data() + kOffset, 0xA7, kLength);
    EXPECT_EQ(0, memcmp(expected.data(), host_bytes, expected.size()));
  }

  // One exact runtime incarnation shared through synchronous suite teardown.
  static inline hrx::hip::testing::HipDso dso_;
  // All per-case entry points are ready for checked resource cleanup.
  bool ready_ = false;
  // User-owned storage retained through unregister and all device accesses.
  alignas(4096) std::array<uint8_t, 4096> storage_ = {};
  // True while the runtime has registered |storage_|.
  bool registered_ = false;
  // Runtime-owned host allocation, released during teardown.
  void* host_memory_ = nullptr;
  // Runtime-owned device allocation, released after all copies complete.
  hipDeviceptr_t device_memory_ = nullptr;
  // Allocates ordinary device memory.
  decltype(&hipMalloc) allocate_ = nullptr;
  // Allocates device memory from the selected stream-ordered pool.
  decltype(&hipMallocAsync) allocate_async_ = nullptr;
  // Releases device memory.
  decltype(&hipFree) free_ = nullptr;
  // Allocates pinned host memory.
  decltype(&hipHostAlloc) allocate_host_ = nullptr;
  // Releases owned host memory.
  decltype(&hipFreeHost) free_host_ = nullptr;
  // Imports user-owned host memory.
  decltype(&hipHostRegister) register_host_ = nullptr;
  // Removes the host registration without freeing user storage.
  decltype(&hipHostUnregister) unregister_host_ = nullptr;
  // Exports a native device address from a host allocation or registration.
  decltype(&hipHostGetDevicePointer) host_device_pointer_ = nullptr;
  // Queries the logical allocation containing an interior device address.
  decltype(&hipMemGetAddressRange) address_range_ = nullptr;
  // Copies bytes through the public synchronous API.
  decltype(&hipMemcpy) copy_ = nullptr;
  // Completes device accesses before direct CPU inspection or teardown.
  decltype(&hipDeviceSynchronize) synchronize_ = nullptr;
};

TEST_F(HipMemoryInteropApiTest, RegisteredHostRangePreservesBytesAndOwnership) {
  ASSERT_EQ(hipSuccess, register_host_(storage_.data(), storage_.size(),
                                       hipHostRegisterMapped));
  registered_ = true;
  ASSERT_NO_FATAL_FAILURE(CheckHostRange(storage_.data()));
  const auto expected = storage_;
  ASSERT_EQ(hipSuccess, unregister_host_(storage_.data()));
  registered_ = false;
  EXPECT_EQ(expected, storage_);
  storage_.fill(0x3C);
  EXPECT_EQ(0x3C, storage_.back());
}

TEST_F(HipMemoryInteropApiTest, OwnedHostRangeExportsUsableInteriorAddress) {
  ASSERT_EQ(hipSuccess, allocate_host_(&host_memory_, storage_.size(),
                                       hipHostMallocMapped));
  ASSERT_NO_FATAL_FAILURE(CheckHostRange(host_memory_));
}

TEST_F(HipMemoryInteropApiTest, PoolAllocationExportsLogicalRangeAndBytes) {
  constexpr size_t kLength = 257;
  ASSERT_EQ(hipSuccess, allocate_async_(&device_memory_, kLength, nullptr));
  ASSERT_EQ(hipSuccess, synchronize_());
  hipDeviceptr_t base = nullptr;
  size_t length = 0;
  ASSERT_EQ(hipSuccess,
            address_range_(&base, &length,
                           static_cast<uint8_t*>(device_memory_) + 13));
  EXPECT_EQ(device_memory_, base);
  EXPECT_EQ(kLength, length);
  std::array<uint8_t, kLength> expected;
  expected.fill(0x5A);
  ASSERT_EQ(hipSuccess, copy_(device_memory_, expected.data(), kLength,
                              hipMemcpyHostToDevice));
  std::array<uint8_t, kLength> result = {};
  ASSERT_EQ(hipSuccess, copy_(result.data(), device_memory_, kLength,
                              hipMemcpyDeviceToHost));
  EXPECT_EQ(expected, result);
}

}  // namespace
