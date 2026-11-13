// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstdint>
#include <semaphore>
#include <thread>

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/base/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/driver/amd/gpu/cts/testdata.h"
#include "iree/hal/driver/amd/gpu/target/identity.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

enum class Observer {
  kStreamQuery,
  kStreamSynchronize,
  kDeviceSynchronize,
  kEventQuery,
  kEventSynchronize,
  kTimedEventQuery,
  kTimedEventSynchronize,
  kHostCallback,
};

// Completion exposes shader writes to mapped host memory without a subsequent
// transfer or queue operation supplying an additional visibility boundary.
class HipVisibilityTest : public testing::TestWithParam<Observer> {
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
    RecordProperty("gpu_target", properties.gcnArchName);
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
    void* host = nullptr;
    using HostMalloc = hipError_t (*)(void**, size_t, unsigned int);
    ASSERT_EQ(hipSuccess, Resolve<HostMalloc>("hipHostMalloc")(
                              &host, kWorkgroupCount * sizeof(uint32_t),
                              hipHostMallocMapped | hipHostMallocNonCoherent));
    output_ = static_cast<uint32_t*>(host);
    std::fill_n(output_, kWorkgroupCount, 0);
    ASSERT_EQ(hipSuccess,
              Resolve<decltype(&hipHostGetDevicePointer)>(
                  "hipHostGetDevicePointer")(&device_output_, output_, 0));
    ASSERT_EQ(hipSuccess,
              Resolve<decltype(&hipStreamCreateWithFlags)>(
                  "hipStreamCreateWithFlags")(&stream_, hipStreamNonBlocking));
    const bool timed = GetParam() == Observer::kTimedEventQuery ||
                       GetParam() == Observer::kTimedEventSynchronize;
    ASSERT_EQ(
        hipSuccess,
        Resolve<decltype(&hipEventCreateWithFlags)>("hipEventCreateWithFlags")(
            &event_, timed ? hipEventDefault : hipEventDisableTiming));
  }

  void TearDown() override {
    if (stream_) {
      EXPECT_EQ(hipSuccess, Resolve<decltype(&hipStreamSynchronize)>(
                                "hipStreamSynchronize")(stream_));
      EXPECT_EQ(hipSuccess, Resolve<decltype(&hipStreamDestroy)>(
                                "hipStreamDestroy")(stream_));
    }
    if (event_) {
      EXPECT_EQ(hipSuccess,
                Resolve<decltype(&hipEventDestroy)>("hipEventDestroy")(event_));
    }
    if (module_) {
      EXPECT_EQ(hipSuccess, Resolve<decltype(&hipModuleUnload)>(
                                "hipModuleUnload")(module_));
    }
    if (output_) {
      EXPECT_EQ(hipSuccess,
                Resolve<decltype(&hipFreeHost)>("hipFreeHost")(output_));
    }
  }

  void Launch(uint32_t incarnation) {
    void* arguments[] = {&device_output_, &incarnation};
    ASSERT_EQ(hipSuccess, Resolve<decltype(&hipModuleLaunchKernel)>(
                              "hipModuleLaunchKernel")(
                              function_, kWorkgroupCount, 1, 1, 64, 1, 1, 0,
                              stream_, arguments, nullptr));
  }

  void Observe() {
    switch (GetParam()) {
      case Observer::kStreamQuery: {
        const auto query = Resolve<decltype(&hipStreamQuery)>("hipStreamQuery");
        hipError_t status;
        while ((status = query(stream_)) == hipErrorNotReady) {
          std::this_thread::yield();
        }
        ASSERT_EQ(hipSuccess, status);
        break;
      }
      case Observer::kStreamSynchronize:
        ASSERT_EQ(hipSuccess, Resolve<decltype(&hipStreamSynchronize)>(
                                  "hipStreamSynchronize")(stream_));
        break;
      case Observer::kDeviceSynchronize:
        ASSERT_EQ(hipSuccess, Resolve<decltype(&hipDeviceSynchronize)>(
                                  "hipDeviceSynchronize")());
        break;
      case Observer::kEventQuery:
      case Observer::kTimedEventQuery: {
        ASSERT_EQ(hipSuccess, Resolve<decltype(&hipEventRecord)>(
                                  "hipEventRecord")(event_, stream_));
        const auto query = Resolve<decltype(&hipEventQuery)>("hipEventQuery");
        hipError_t status;
        while ((status = query(event_)) == hipErrorNotReady) {
          std::this_thread::yield();
        }
        ASSERT_EQ(hipSuccess, status);
        break;
      }
      case Observer::kEventSynchronize:
      case Observer::kTimedEventSynchronize:
        ASSERT_EQ(hipSuccess, Resolve<decltype(&hipEventRecord)>(
                                  "hipEventRecord")(event_, stream_));
        ASSERT_EQ(hipSuccess, Resolve<decltype(&hipEventSynchronize)>(
                                  "hipEventSynchronize")(event_));
        break;
      case Observer::kHostCallback:
        ASSERT_EQ(hipSuccess,
                  Resolve<decltype(&hipLaunchHostFunc)>("hipLaunchHostFunc")(
                      stream_,
                      [](void* user_data) {
                        auto* self = static_cast<HipVisibilityTest*>(user_data);
                        std::copy_n(self->output_, kWorkgroupCount,
                                    self->callback_output_.begin());
                        self->callback_completed_.release();
                      },
                      this));
        callback_completed_.acquire();
        break;
    }
  }

  // Four independent workgroups each accumulate into one mapped host word.
  static constexpr uint32_t kWorkgroupCount = 4;
  // Exact frontend and native driver used throughout the suite.
  static inline hrx::hip::testing::HipDso library_;
  // Executable retained until the stream finishes.
  hipModule_t module_ = nullptr;
  // Ordinary Loom entry point borrowed from module_.
  hipFunction_t function_ = nullptr;
  // Nonblocking stream owning the tested dispatch chain.
  hipStream_t stream_ = nullptr;
  // Completion point reused for each observed prefix.
  hipEvent_t event_ = nullptr;
  // Noncoherent mapped host storage, read only after the selected observer.
  uint32_t* output_ = nullptr;
  // Device address of output_, retained through the final dispatch.
  void* device_output_ = nullptr;
  // CPU snapshot taken within the stream callback before it signals the host.
  std::array<uint32_t, kWorkgroupCount> callback_output_ = {};
  // Joins the callback's snapshot without calling another HIP observer.
  std::binary_semaphore callback_completed_{0};
};

TEST_P(HipVisibilityTest, CompletionExposesMappedShaderWrites) {
  std::array<uint32_t, kWorkgroupCount> expected = {};
  for (uint32_t incarnation = 1; incarnation <= 32; ++incarnation) {
    SCOPED_TRACE(incarnation);
    ASSERT_NO_FATAL_FAILURE(Launch(incarnation));
    ASSERT_NO_FATAL_FAILURE(Observe());
    for (uint32_t group = 0; group < kWorkgroupCount; ++group) {
      expected[group] += incarnation + group;
      EXPECT_EQ(output_[group], expected[group]);
      if (GetParam() == Observer::kHostCallback) {
        EXPECT_EQ(callback_output_[group], expected[group]);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Observers, HipVisibilityTest,
    testing::Values(Observer::kStreamQuery, Observer::kStreamSynchronize,
                    Observer::kDeviceSynchronize, Observer::kEventQuery,
                    Observer::kEventSynchronize, Observer::kTimedEventQuery,
                    Observer::kTimedEventSynchronize, Observer::kHostCallback));

}  // namespace
