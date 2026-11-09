// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <memory>

#include "api.h"
#include "hip_dso_test_util.h"
#include "iree/testing/gtest.h"

namespace {

void ExerciseRegisteredMemory(hrx::hip::testing::HipDso& dso, uint8_t pattern) {
  const auto register_host =
      dso.Resolve<decltype(&hipHostRegister)>("hipHostRegister");
  const auto unregister_host =
      dso.Resolve<decltype(&hipHostUnregister)>("hipHostUnregister");
  const auto device_pointer = dso.Resolve<decltype(&hipHostGetDevicePointer)>(
      "hipHostGetDevicePointer");
  const auto allocate = dso.Resolve<decltype(&hipMalloc)>("hipMalloc");
  const auto release = dso.Resolve<decltype(&hipFree)>("hipFree");
  const auto copy = dso.Resolve<decltype(&hipMemcpy)>("hipMemcpy");
  ASSERT_NE(register_host, nullptr);
  ASSERT_NE(unregister_host, nullptr);
  ASSERT_NE(device_pointer, nullptr);
  ASSERT_NE(allocate, nullptr);
  ASSERT_NE(release, nullptr);
  ASSERT_NE(copy, nullptr);

  alignas(4096) std::array<uint8_t, 4096> source;
  source.fill(pattern);
  ASSERT_EQ(hipSuccess,
            register_host(source.data(), source.size(), hipHostRegisterMapped));
  const auto unregister = [&](void* pointer) {
    EXPECT_EQ(hipSuccess, unregister_host(pointer));
  };
  std::unique_ptr<void, decltype(unregister)> registration(source.data(),
                                                           unregister);
  void* source_device_pointer = nullptr;
  ASSERT_EQ(hipSuccess,
            device_pointer(&source_device_pointer, source.data(), 0));
  void* allocation = nullptr;
  ASSERT_EQ(hipSuccess, allocate(&allocation, source.size()));
  const auto free_allocation = [&](void* pointer) {
    EXPECT_EQ(hipSuccess, release(pointer));
  };
  std::unique_ptr<void, decltype(free_allocation)> device_memory(
      allocation, free_allocation);
  ASSERT_EQ(hipSuccess, copy(device_memory.get(), source_device_pointer,
                             source.size(), hipMemcpyDeviceToDevice));
  std::array<uint8_t, 4096> result = {};
  ASSERT_EQ(hipSuccess, copy(result.data(), device_memory.get(), result.size(),
                             hipMemcpyDeviceToHost));
  EXPECT_EQ(result, source);
}

TEST(HipLifecycleTest, ReopensAndRegistersHostMemoryAfterDeinitialization) {
  hrx::hip::testing::HipDso dso;
  for (int generation = 0; generation < 3; ++generation) {
    SCOPED_TRACE(generation);
    ASSERT_TRUE(dso.Open()) << dso.error();
    const auto initialize = dso.Resolve<decltype(&hipInit)>("hipInit");
    const auto deinitialize =
        dso.Resolve<decltype(&hipHALDeinit)>("hipHALDeinit");
    ASSERT_NE(initialize, nullptr);
    ASSERT_NE(deinitialize, nullptr);
    ASSERT_EQ(hipSuccess, initialize(0));
    ExerciseRegisteredMemory(dso, static_cast<uint8_t>(generation + 0x31));
    ASSERT_EQ(hipSuccess, deinitialize());
    ASSERT_TRUE(dso.Close()) << dso.error();
    ASSERT_FALSE(HasFatalFailure());
  }
}

}  // namespace
