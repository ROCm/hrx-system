// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <cstdint>

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/testing/gtest.h"
#include "libhrx/cts/core/amdgpu_executable_test_data.hpp"

namespace {

// Independent nonblocking streams communicate through explicitly qualified
// signal storage. Waiting on the consumer must not require another API call
// that flushes the producer. All kernels are finite single-workitem stores.
class HipStreamProgressTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    ASSERT_TRUE(library_.Open()) << library_.error();
    ASSERT_EQ(hipSuccess, Resolve<decltype(&hipInit)>("hipInit")(0));
  }

  static void TearDownTestSuite() {
    if (library_.is_open()) {
      EXPECT_EQ(hipSuccess, Resolve<decltype(&hipHALDeinit)>("hipHALDeinit")());
      EXPECT_TRUE(library_.Close()) << library_.error();
    }
  }

  template <typename T>
  static T Resolve(const char* name) {
    T function = library_.Resolve<T>(name);
    EXPECT_NE(nullptr, function) << name << ": " << library_.error();
    return function;
  }

  void SetUp() override {
    int supports_wait = 0;
    ASSERT_EQ(
        hipSuccess,
        Resolve<decltype(&hipDeviceGetAttribute)>("hipDeviceGetAttribute")(
            &supports_wait, hipDeviceAttributeCanUseStreamWaitValue, 0));
    if (!supports_wait) {
      GTEST_SKIP() << "device does not support stream memory waits";
    }
    void* signal = nullptr;
    ASSERT_EQ(
        hipSuccess,
        Resolve<decltype(&hipExtMallocWithFlags)>("hipExtMallocWithFlags")(
            &signal, sizeof(uint64_t), hipMallocSignalMemory));
    signal_ = static_cast<uint32_t*>(signal);
    std::atomic_ref<uint32_t>(*signal_).store(0, std::memory_order_release);
    const auto create_stream = Resolve<decltype(&hipStreamCreateWithFlags)>(
        "hipStreamCreateWithFlags");
    ASSERT_EQ(hipSuccess, create_stream(&producer_, hipStreamNonBlocking));
    ASSERT_EQ(hipSuccess, create_stream(&consumer_, hipStreamNonBlocking));
  }

  void TearDown() override {
    // An API rejection must not leave an accepted wait holding the fixture's
    // storage. Successful execution reaches this point through its GPU writer.
    if (signal_) {
      std::atomic_ref<uint32_t>(*signal_).store(kExpectedValue,
                                                std::memory_order_release);
    }
    const auto destroy_stream =
        Resolve<decltype(&hipStreamDestroy)>("hipStreamDestroy");
    if (consumer_) {
      EXPECT_EQ(hipSuccess, destroy_stream(consumer_));
    }
    if (producer_) {
      EXPECT_EQ(hipSuccess, destroy_stream(producer_));
    }
    if (module_) {
      EXPECT_EQ(hipSuccess, Resolve<decltype(&hipModuleUnload)>(
                                "hipModuleUnload")(module_));
    }
    if (signal_) {
      EXPECT_EQ(hipSuccess, Resolve<decltype(&hipFree)>("hipFree")(signal_));
    }
  }

  void EnqueueConsumer() {
    ASSERT_EQ(hipSuccess,
              Resolve<decltype(&hipStreamWaitValue32)>("hipStreamWaitValue32")(
                  consumer_, signal_, kExpectedValue, hipStreamWaitValueEq,
                  UINT32_MAX));
  }

  void JoinConsumer() {
    // This is deliberately the only stream touched after the producer call.
    ASSERT_EQ(hipSuccess, Resolve<decltype(&hipStreamSynchronize)>(
                              "hipStreamSynchronize")(consumer_));
    EXPECT_EQ(kExpectedValue, std::atomic_ref<uint32_t>(*signal_).load(
                                  std::memory_order_acquire));
  }

  // Byte-repeated value can also be produced by a byte fill.
  static constexpr uint32_t kExpectedValue = 0x5a5a5a5a;
  // One exact frontend and native driver span the complete suite.
  static inline hrx::hip::testing::HipDso library_;
  // Nonblocking stream that publishes the value consumed by the other stream.
  hipStream_t producer_ = nullptr;
  // Nonblocking stream with an explicitly ordered memory-value wait.
  hipStream_t consumer_ = nullptr;
  // Host-visible, qualified native atomic storage retained through both
  // streams.
  uint32_t* signal_ = nullptr;
  // Optional executable retained through every producer operation.
  hipModule_t module_ = nullptr;
};

TEST_F(HipStreamProgressTest, KernelPublishesWithoutAnotherProducerCall) {
  hipDeviceProp_t properties = {};
  ASSERT_EQ(hipSuccess, Resolve<decltype(&hipGetDevicePropertiesR0600)>(
                            "hipGetDevicePropertiesR0600")(&properties, 0));
  const auto image =
      hrx_cts::FindAmdgpuExecutableTestImage(properties.gcnArchName);
  ASSERT_NE(nullptr, image.file);
  ASSERT_EQ(hipSuccess, Resolve<decltype(&hipModuleLoadData)>(
                            "hipModuleLoadData")(&module_, image.file->data));
  hipFunction_t function = nullptr;
  ASSERT_EQ(hipSuccess,
            Resolve<decltype(&hipModuleGetFunction)>("hipModuleGetFunction")(
                &function, module_, "hrx_store_output"));
  uint32_t value = kExpectedValue;
  void* arguments[] = {&signal_, &value};
  ASSERT_NO_FATAL_FAILURE(EnqueueConsumer());
  ASSERT_EQ(hipSuccess,
            Resolve<decltype(&hipModuleLaunchKernel)>("hipModuleLaunchKernel")(
                function, 1, 1, 1, 1, 1, 1, 0, producer_, arguments, nullptr));
  ASSERT_NO_FATAL_FAILURE(JoinConsumer());
}

}  // namespace
