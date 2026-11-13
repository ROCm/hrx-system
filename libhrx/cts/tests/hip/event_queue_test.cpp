// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/base/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/driver/amd/gpu/cts/testdata.h"
#include "iree/hal/driver/amd/gpu/target/identity.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

enum class CooperativePredecessor {
  kOrdinaryWork,
  kMemoryWait,
};

// A stream event must observe cooperative work even when that work belongs to
// a different queue realization, with or without an active memory-wait lease.
class HipCooperativeEventTest
    : public testing::TestWithParam<CooperativePredecessor> {
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
    int device = 0;
    ASSERT_EQ(hipSuccess,
              Resolve<decltype(&hipGetDevice)>("hipGetDevice")(&device));
    hipDeviceProp_t properties = {};
    ASSERT_EQ(hipSuccess,
              Resolve<decltype(&hipGetDevicePropertiesR0600)>(
                  "hipGetDevicePropertiesR0600")(&properties, device));
    if (!properties.cooperativeLaunch) {
      GTEST_SKIP() << "device does not support cooperative launch";
    }
    if (GetParam() == CooperativePredecessor::kMemoryWait) {
      int supports_wait = 0;
      ASSERT_EQ(
          hipSuccess,
          Resolve<decltype(&hipDeviceGetAttribute)>("hipDeviceGetAttribute")(
              &supports_wait, hipDeviceAttributeCanUseStreamWaitValue, device));
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
    }
    iree_hal_amd_gpu_target_identity_t identity;
    IREE_ASSERT_OK(iree_hal_amd_gpu_target_identity_parse_artifact_key(
        iree_make_cstring_view(properties.gcnArchName), &identity));
    const auto images =
        iree::hal::cts::CtsRegistry::ListExecutableTargets("hrx_hip_event");
    const auto* image =
        iree::hal::amd::gpu::cts::SelectExecutableTarget(images, identity);
    ASSERT_NE(nullptr, image) << properties.gcnArchName;
    const auto data = image->data_fn(IREE_SV("event_queue.bin"));
    ASSERT_NE(nullptr, data.data);
    ASSERT_EQ(hipSuccess, Resolve<decltype(&hipModuleLoadData)>(
                              "hipModuleLoadData")(&module_, data.data));
    ASSERT_EQ(hipSuccess,
              Resolve<decltype(&hipModuleGetFunction)>("hipModuleGetFunction")(
                  &function_, module_, "event_queue"));
    const auto allocate = Resolve<decltype(&hipMalloc)>("hipMalloc");
    ASSERT_EQ(hipSuccess,
              allocate(&output_, kWorkgroupCount * sizeof(uint32_t)));
    void* readback = nullptr;
    ASSERT_EQ(hipSuccess, Resolve<decltype(&hipHostAlloc)>("hipHostAlloc")(
                              &readback, kWorkgroupCount * sizeof(uint32_t),
                              hipHostMallocDefault));
    actual_ = static_cast<uint32_t*>(readback);
    const auto create_stream = Resolve<decltype(&hipStreamCreateWithFlags)>(
        "hipStreamCreateWithFlags");
    for (auto& stream : streams_) {
      ASSERT_EQ(hipSuccess, create_stream(&stream, hipStreamNonBlocking));
    }
    const auto create_event =
        Resolve<decltype(&hipEventCreateWithFlags)>("hipEventCreateWithFlags");
    for (auto& event : events_) {
      ASSERT_EQ(hipSuccess, create_event(&event, hipEventDisableTiming));
    }
  }

  void TearDown() override {
    // Release any accepted memory wait after an API rejection before joining
    // the queues that may still reference this fixture's buffers.
    if (signal_) {
      std::atomic_ref<uint32_t>(*signal_).store(release_value_,
                                                std::memory_order_release);
    }
    const auto synchronize =
        Resolve<decltype(&hipStreamSynchronize)>("hipStreamSynchronize");
    const auto destroy =
        Resolve<decltype(&hipStreamDestroy)>("hipStreamDestroy");
    for (auto stream : streams_) {
      if (stream) {
        EXPECT_EQ(hipSuccess, synchronize(stream));
        EXPECT_EQ(hipSuccess, destroy(stream));
      }
    }
    for (auto event : events_) {
      if (event) {
        EXPECT_EQ(hipSuccess, Resolve<decltype(&hipEventDestroy)>(
                                  "hipEventDestroy")(event));
      }
    }
    if (module_) {
      EXPECT_EQ(hipSuccess, Resolve<decltype(&hipModuleUnload)>(
                                "hipModuleUnload")(module_));
    }
    const auto free_buffer = Resolve<decltype(&hipFree)>("hipFree");
    if (output_) {
      EXPECT_EQ(hipSuccess, free_buffer(output_));
    }
    if (signal_) {
      EXPECT_EQ(hipSuccess, free_buffer(signal_));
    }
    if (actual_) {
      EXPECT_EQ(hipSuccess,
                Resolve<decltype(&hipFreeHost)>("hipFreeHost")(actual_));
    }
  }

  // Each workgroup updates its own output element before event completion.
  static constexpr uint32_t kWorkgroupCount = 4;
  // Exact frontend and native driver used throughout the suite.
  static inline hrx::hip::testing::HipDso library_;
  // Source, consumer, and independent writer streams, respectively.
  std::array<hipStream_t, 3> streams_ = {};
  // Source point and consumer completion, respectively.
  std::array<hipEvent_t, 2> events_ = {};
  // Executable retained until all streams finish.
  hipModule_t module_ = nullptr;
  // Loom entry point launched through the cooperative API, borrowed from
  // module_.
  hipFunction_t function_ = nullptr;
  // Device output accumulated by the cooperative kernel.
  void* output_ = nullptr;
  // Qualified host-visible atomic storage for the optional memory wait.
  uint32_t* signal_ = nullptr;
  // Latest value that must be released on every cleanup path.
  uint32_t release_value_ = 0;
  // Pinned host destination keeps the copy asynchronous while the producer is
  // blocked. It outlives every accepted copy and completion wait.
  uint32_t* actual_ = nullptr;
};

TEST_P(HipCooperativeEventTest, RetainsTheCooperativePointAcrossRerecording) {
  const auto record = Resolve<decltype(&hipEventRecord)>("hipEventRecord");
  const auto synchronize =
      Resolve<decltype(&hipEventSynchronize)>("hipEventSynchronize");
  for (uint32_t incarnation = 1; incarnation <= 4; ++incarnation) {
    ASSERT_EQ(hipSuccess,
              Resolve<decltype(&hipMemsetAsync)>("hipMemsetAsync")(
                  output_, 0, kWorkgroupCount * sizeof(uint32_t), streams_[0]));
    if (signal_) {
      release_value_ = incarnation;
      ASSERT_EQ(hipSuccess,
                Resolve<decltype(&hipStreamWaitValue32)>(
                    "hipStreamWaitValue32")(streams_[0], signal_, incarnation,
                                            hipStreamWaitValueEq, UINT32_MAX));
    }
    void* arguments[] = {&output_, &incarnation};
    ASSERT_EQ(hipSuccess, Resolve<decltype(&hipModuleLaunchCooperativeKernel)>(
                              "hipModuleLaunchCooperativeKernel")(
                              function_, kWorkgroupCount, 1, 1, 64, 1, 1, 0,
                              streams_[0], arguments));
    ASSERT_EQ(hipSuccess, record(events_[0], streams_[0]));
    ASSERT_EQ(hipSuccess,
              Resolve<decltype(&hipStreamWaitEvent)>("hipStreamWaitEvent")(
                  streams_[1], events_[0], 0));
    std::fill_n(actual_, kWorkgroupCount, UINT32_MAX);
    ASSERT_EQ(hipSuccess,
              Resolve<decltype(&hipMemcpyAsync)>("hipMemcpyAsync")(
                  actual_, output_, kWorkgroupCount * sizeof(uint32_t),
                  hipMemcpyDeviceToHost, streams_[1]));
    ASSERT_EQ(hipSuccess, record(events_[1], streams_[1]));

    // Replacing the event's current point must not retarget the accepted wait.
    // No further source-stream call publishes or joins the cooperative work.
    ASSERT_EQ(hipSuccess, record(events_[0], streams_[2]));
    ASSERT_EQ(hipSuccess, synchronize(events_[0]));
    if (signal_) {
      EXPECT_EQ(hipErrorNotReady,
                Resolve<decltype(&hipEventQuery)>("hipEventQuery")(events_[1]));
      ASSERT_EQ(hipSuccess, Resolve<decltype(&hipStreamWriteValue32)>(
                                "hipStreamWriteValue32")(streams_[2], signal_,
                                                         incarnation, 0));
    }
    ASSERT_EQ(hipSuccess, synchronize(events_[1]));
    for (uint32_t i = 0; i < kWorkgroupCount; ++i) {
      EXPECT_EQ(incarnation + i, actual_[i]);
    }
  }

  ASSERT_EQ(hipSuccess, record(events_[0], streams_[0]));
  ASSERT_EQ(hipSuccess, Resolve<decltype(&hipStreamDestroy)>(
                            "hipStreamDestroy")(streams_[0]));
  streams_[0] = nullptr;
  EXPECT_EQ(hipSuccess, synchronize(events_[0]));
}

INSTANTIATE_TEST_SUITE_P(Predecessors, HipCooperativeEventTest,
                         testing::Values(CooperativePredecessor::kOrdinaryWork,
                                         CooperativePredecessor::kMemoryWait));

}  // namespace
