// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/testing/gtest.h"
#include "libhrx/cts/core/amdgpu_executable_test_data.hpp"

namespace {

enum class ProducerOperation {
  kKernel,
  kKernelBurst,
  kFill,
  kFill8,
  kFill16,
  kFill32,
  kFill2D,
  kFill3D,
  kUpload,
  kRegisteredUpload,
  kCopy,
  kUpload2D,
  kCopy2D,
};

// Independent nonblocking streams communicate through explicitly qualified
// signal storage. Waiting on the consumer must not require another API call
// that flushes the producer. All kernels are finite single-workitem stores.
class HipStreamProgressTest : public testing::TestWithParam<ProducerOperation> {
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
    const bool pitched = GetParam() == ProducerOperation::kFill2D ||
                         GetParam() == ProducerOperation::kFill3D ||
                         GetParam() == ProducerOperation::kUpload2D ||
                         GetParam() == ProducerOperation::kCopy2D;
    expected_ = pitched ? 0x005a005a : 0x5a5a5a5a;
    void* signal = nullptr;
    ASSERT_EQ(
        hipSuccess,
        Resolve<decltype(&hipExtMallocWithFlags)>("hipExtMallocWithFlags")(
            &signal, sizeof(uint64_t), hipMallocSignalMemory));
    signal_ = static_cast<uint32_t*>(signal);
    std::memset(signal_, 0, sizeof(uint64_t));
    const auto create_stream = Resolve<decltype(&hipStreamCreateWithFlags)>(
        "hipStreamCreateWithFlags");
    ASSERT_EQ(hipSuccess, create_stream(&producer_, hipStreamNonBlocking));
    ASSERT_EQ(hipSuccess, create_stream(&consumer_, hipStreamNonBlocking));
    ASSERT_EQ(
        hipSuccess,
        Resolve<decltype(&hipEventCreateWithFlags)>("hipEventCreateWithFlags")(
            &completion_, hipEventBlockingSync | hipEventDisableTiming));
  }

  void TearDown() override {
    // An API rejection must not leave an accepted wait holding the fixture's
    // storage. Successful execution reaches this point through its GPU writer.
    if (signal_) {
      std::atomic_ref<uint32_t>(*signal_).store(expected_,
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
    if (completion_) {
      EXPECT_EQ(hipSuccess, Resolve<decltype(&hipEventDestroy)>(
                                "hipEventDestroy")(completion_));
    }
    if (module_) {
      EXPECT_EQ(hipSuccess, Resolve<decltype(&hipModuleUnload)>(
                                "hipModuleUnload")(module_));
    }
    if (device_source_) {
      EXPECT_EQ(hipSuccess,
                Resolve<decltype(&hipFree)>("hipFree")(device_source_));
    }
    if (host_source_) {
      EXPECT_EQ(hipSuccess,
                Resolve<decltype(&hipFreeHost)>("hipFreeHost")(host_source_));
    }
    if (signal_) {
      EXPECT_EQ(hipSuccess, Resolve<decltype(&hipFree)>("hipFree")(signal_));
    }
  }

  void EnqueueConsumer() {
    ASSERT_EQ(
        hipSuccess,
        Resolve<decltype(&hipStreamWaitValue32)>("hipStreamWaitValue32")(
            consumer_, signal_, expected_, hipStreamWaitValueEq, UINT32_MAX));
    ASSERT_EQ(hipSuccess, Resolve<decltype(&hipEventRecord)>("hipEventRecord")(
                              completion_, consumer_));
  }

  void JoinConsumer() {
    // Event synchronization waits only on this recorded point. Stream
    // synchronization can flush other streams and mask an unsubmitted producer.
    ASSERT_EQ(hipSuccess, Resolve<decltype(&hipEventSynchronize)>(
                              "hipEventSynchronize")(completion_));
    EXPECT_EQ(expected_, std::atomic_ref<uint32_t>(*signal_).load(
                             std::memory_order_acquire));
  }

  // Exact value after either a contiguous operation or a strided byte
  // operation.
  uint32_t expected_ = 0;
  // One exact frontend and native driver span the complete suite.
  static inline hrx::hip::testing::HipDso library_;
  // Nonblocking stream that publishes the value consumed by the other stream.
  hipStream_t producer_ = nullptr;
  // Nonblocking stream with an explicitly ordered memory-value wait.
  hipStream_t consumer_ = nullptr;
  // Recorded after the consumer wait, before the producing operation is issued.
  hipEvent_t completion_ = nullptr;
  // Host-visible, qualified native atomic storage retained through both
  // streams.
  uint32_t* signal_ = nullptr;
  // Ordinary device source retained until the producing stream completes.
  void* device_source_ = nullptr;
  // Registered host source retained until the producing stream completes.
  void* host_source_ = nullptr;
  // Optional executable retained through every producer operation.
  hipModule_t module_ = nullptr;
};

TEST_P(HipStreamProgressTest, PublishesWithoutAnotherProducerCall) {
  const ProducerOperation operation = GetParam();
  hipFunction_t function = nullptr;
  if (operation == ProducerOperation::kKernel ||
      operation == ProducerOperation::kKernelBurst) {
    hipDeviceProp_t properties = {};
    ASSERT_EQ(hipSuccess, Resolve<decltype(&hipGetDevicePropertiesR0600)>(
                              "hipGetDevicePropertiesR0600")(&properties, 0));
    const auto image =
        hrx_cts::FindAmdgpuExecutableTestImage(properties.gcnArchName);
    ASSERT_NE(nullptr, image.file);
    ASSERT_EQ(hipSuccess, Resolve<decltype(&hipModuleLoadData)>(
                              "hipModuleLoadData")(&module_, image.file->data));
    ASSERT_EQ(hipSuccess,
              Resolve<decltype(&hipModuleGetFunction)>("hipModuleGetFunction")(
                  &function, module_, "hrx_store_output"));
  }

  std::array<uint8_t, sizeof(uint64_t)> source;
  source.fill(0x5a);
  if (operation == ProducerOperation::kCopy ||
      operation == ProducerOperation::kCopy2D) {
    ASSERT_EQ(hipSuccess, Resolve<decltype(&hipMalloc)>("hipMalloc")(
                              &device_source_, source.size()));
    ASSERT_EQ(hipSuccess, Resolve<decltype(&hipMemcpy)>("hipMemcpy")(
                              device_source_, source.data(), source.size(),
                              hipMemcpyHostToDevice));
  } else if (operation == ProducerOperation::kRegisteredUpload) {
    ASSERT_EQ(hipSuccess,
              Resolve<decltype(&hipHostAlloc)>("hipHostAlloc")(
                  &host_source_, source.size(), hipHostMallocDefault));
    std::memcpy(host_source_, source.data(), source.size());
  }

  // All setup that could synchronize the context precedes the unsatisfied
  // consumer. No observer API is called on the producer after its final write.
  ASSERT_NO_FATAL_FAILURE(EnqueueConsumer());
  switch (operation) {
    case ProducerOperation::kKernel:
    case ProducerOperation::kKernelBurst: {
      const uint32_t count = operation == ProducerOperation::kKernel ? 1 : 17;
      for (uint32_t i = 0; i < count; ++i) {
        // Only the final dispatch satisfies the independent consumer. Earlier
        // publication cannot hide a finite tail waiting for another API call.
        uint32_t value = i + 1 == count ? expected_ : 0;
        void* arguments[] = {&signal_, &value};
        ASSERT_EQ(
            hipSuccess,
            Resolve<decltype(&hipModuleLaunchKernel)>("hipModuleLaunchKernel")(
                function, 1, 1, 1, 1, 1, 1, 0, producer_, arguments, nullptr));
      }
      break;
    }
    case ProducerOperation::kFill:
      ASSERT_EQ(hipSuccess, Resolve<decltype(&hipMemsetAsync)>(
                                "hipMemsetAsync")(signal_, 0x5a, 4, producer_));
      break;
    case ProducerOperation::kFill8:
      ASSERT_EQ(hipSuccess,
                Resolve<decltype(&hipMemsetD8Async)>("hipMemsetD8Async")(
                    signal_, 0x5a, 4, producer_));
      break;
    case ProducerOperation::kFill16:
      ASSERT_EQ(hipSuccess,
                Resolve<decltype(&hipMemsetD16Async)>("hipMemsetD16Async")(
                    signal_, 0x5a5a, 2, producer_));
      break;
    case ProducerOperation::kFill32:
      ASSERT_EQ(hipSuccess,
                Resolve<decltype(&hipMemsetD32Async)>("hipMemsetD32Async")(
                    signal_, expected_, 1, producer_));
      break;
    case ProducerOperation::kFill2D:
      ASSERT_EQ(hipSuccess,
                Resolve<decltype(&hipMemset2DAsync)>("hipMemset2DAsync")(
                    signal_, 2, 0x5a, 1, 4, producer_));
      break;
    case ProducerOperation::kFill3D: {
      const hipPitchedPtr target = {signal_, 2, 1, 2};
      const hipExtent extent = {1, 2, 2};
      ASSERT_EQ(hipSuccess,
                Resolve<decltype(&hipMemset3DAsync)>("hipMemset3DAsync")(
                    target, 0x5a, extent, producer_));
      break;
    }
    case ProducerOperation::kUpload:
    case ProducerOperation::kRegisteredUpload:
      ASSERT_EQ(hipSuccess,
                Resolve<decltype(&hipMemcpyAsync)>("hipMemcpyAsync")(
                    signal_, host_source_ ? host_source_ : source.data(),
                    source.size(), hipMemcpyHostToDevice, producer_));
      break;
    case ProducerOperation::kCopy:
      ASSERT_EQ(hipSuccess,
                Resolve<decltype(&hipMemcpyAsync)>("hipMemcpyAsync")(
                    signal_, device_source_, source.size(),
                    hipMemcpyDeviceToDevice, producer_));
      break;
    case ProducerOperation::kUpload2D:
    case ProducerOperation::kCopy2D:
      ASSERT_EQ(
          hipSuccess,
          Resolve<decltype(&hipMemcpy2DAsync)>("hipMemcpy2DAsync")(
              signal_, 2, device_source_ ? device_source_ : source.data(), 2, 1,
              4,
              device_source_ ? hipMemcpyDeviceToDevice : hipMemcpyHostToDevice,
              producer_));
      break;
  }
  ASSERT_NO_FATAL_FAILURE(JoinConsumer());
}

INSTANTIATE_TEST_SUITE_P(
    Operations, HipStreamProgressTest,
    testing::Values(ProducerOperation::kKernel, ProducerOperation::kKernelBurst,
                    ProducerOperation::kFill, ProducerOperation::kFill8,
                    ProducerOperation::kFill16, ProducerOperation::kFill32,
                    ProducerOperation::kFill2D, ProducerOperation::kFill3D,
                    ProducerOperation::kUpload,
                    ProducerOperation::kRegisteredUpload,
                    ProducerOperation::kCopy, ProducerOperation::kUpload2D,
                    ProducerOperation::kCopy2D),
    ([](const testing::TestParamInfo<ProducerOperation>& info) {
      static constexpr const char* names[] = {
          "Kernel", "KernelBurst", "Fill",   "Fill8",  "Fill16",
          "Fill32", "Fill2D",      "Fill3D", "Upload", "RegisteredUpload",
          "Copy",   "Upload2D",    "Copy2D"};
      return std::string(names[static_cast<size_t>(info.param)]);
    }));

}  // namespace
