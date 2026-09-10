// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "binding/hip/api.h"
#include "iree/testing/gtest.h"

namespace {

const char* CandidateLibPath() {
  if (const char* env = std::getenv("HRX_TEST_LIBAMDHIP64");
      env && *env != '\0') {
    return env;
  }
#ifdef HRX_TEST_LIBAMDHIP64_PATH
  return HRX_TEST_LIBAMDHIP64_PATH;
#else
  return "libamdhip64.so";
#endif
}

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipMallocFn = hipError_t (*)(void** pointer, size_t size);
using HipFreeFn = hipError_t (*)(void* pointer);
using HipMallocHostFn = hipError_t (*)(void** pointer, size_t size);
using HipFreeHostFn = hipError_t (*)(void* pointer);
using HipHostGetDevicePointerFn = hipError_t (*)(hipDeviceptr_t* device_pointer,
                                                 void* host_pointer,
                                                 unsigned int flags);
using HipMemsetFn = hipError_t (*)(void* pointer, int value, size_t size);
using HipMemcpyFn = hipError_t (*)(void* destination, const void* source,
                                   size_t size, hipMemcpyKind kind);
using HipStreamCreateFn = hipError_t (*)(hipStream_t* stream);
using HipStreamDestroyFn = hipError_t (*)(hipStream_t stream);
using HipStreamSynchronizeFn = hipError_t (*)(hipStream_t stream);
using HipStreamWriteValue32Fn = hipError_t (*)(hipStream_t stream,
                                               void* pointer, uint32_t value,
                                               unsigned int flags);
using HipStreamWriteValue64Fn = hipError_t (*)(hipStream_t stream,
                                               void* pointer, uint64_t value,
                                               unsigned int flags);
using HipStreamWaitValue32Fn = hipError_t (*)(hipStream_t stream, void* pointer,
                                              uint32_t value,
                                              unsigned int flags,
                                              uint32_t mask);
using HipStreamWaitValue64Fn = hipError_t (*)(hipStream_t stream, void* pointer,
                                              uint64_t value,
                                              unsigned int flags,
                                              uint64_t mask);
using HipStreamBatchMemOpFn =
    hipError_t (*)(hipStream_t stream, unsigned int count,
                   hipStreamBatchMemOpParams* parameters, unsigned int flags);
using HipStreamBeginCaptureFn = hipError_t (*)(hipStream_t stream,
                                               hipStreamCaptureMode mode);
using HipStreamEndCaptureFn = hipError_t (*)(hipStream_t stream,
                                             hipGraph_t* graph);
using HipStreamIsCapturingFn = hipError_t (*)(hipStream_t stream,
                                              hipStreamCaptureStatus* status);
using HipGraphInstantiateFn = hipError_t (*)(hipGraphExec_t* executable,
                                             hipGraph_t graph,
                                             hipGraphNode_t* error_node,
                                             char* log_buffer,
                                             size_t buffer_size);
using HipGraphLaunchFn = hipError_t (*)(hipGraphExec_t executable,
                                        hipStream_t stream);
using HipGraphExecDestroyFn = hipError_t (*)(hipGraphExec_t executable);
using HipGraphDestroyFn = hipError_t (*)(hipGraph_t graph);

struct HipRuntimeApi {
  // Handle returned by dlopen for the HIP runtime instance.
  void* library = nullptr;
  // Initializes the runtime.
  HipInitFn init = nullptr;
  // Allocates device-visible memory.
  HipMallocFn malloc = nullptr;
  // Releases device-visible memory.
  HipFreeFn free = nullptr;
  // Allocates host-visible memory.
  HipMallocHostFn malloc_host = nullptr;
  // Releases host-visible memory.
  HipFreeHostFn free_host = nullptr;
  // Returns the device-visible alias of a host allocation.
  HipHostGetDevicePointerFn host_get_device_pointer = nullptr;
  // Initializes device-visible memory.
  HipMemsetFn memset = nullptr;
  // Copies between host and device memory.
  HipMemcpyFn memcpy = nullptr;
  // Creates a stream.
  HipStreamCreateFn stream_create = nullptr;
  // Destroys a stream.
  HipStreamDestroyFn stream_destroy = nullptr;
  // Waits for all work on one stream.
  HipStreamSynchronizeFn stream_synchronize = nullptr;
  // Enqueues a 32-bit stream-ordered write.
  HipStreamWriteValue32Fn write_value_32 = nullptr;
  // Enqueues a 64-bit stream-ordered write.
  HipStreamWriteValue64Fn write_value_64 = nullptr;
  // Enqueues a 32-bit stream-ordered wait.
  HipStreamWaitValue32Fn wait_value_32 = nullptr;
  // Enqueues a 64-bit stream-ordered wait.
  HipStreamWaitValue64Fn wait_value_64 = nullptr;
  // Enqueues one transaction of stream memory operations.
  HipStreamBatchMemOpFn batch_mem_op = nullptr;
  // Begins stream capture.
  HipStreamBeginCaptureFn stream_begin_capture = nullptr;
  // Ends stream capture.
  HipStreamEndCaptureFn stream_end_capture = nullptr;
  // Queries stream capture state.
  HipStreamIsCapturingFn stream_is_capturing = nullptr;
  // Instantiates a captured graph.
  HipGraphInstantiateFn graph_instantiate = nullptr;
  // Launches a graph executable.
  HipGraphLaunchFn graph_launch = nullptr;
  // Destroys a graph executable.
  HipGraphExecDestroyFn graph_exec_destroy = nullptr;
  // Destroys a graph template.
  HipGraphDestroyFn graph_destroy = nullptr;
};

template <typename T>
T ResolveHipSymbol(void* library, const char* name) {
  return reinterpret_cast<T>(dlsym(library, name));
}

class StartBarrier {
 public:
  explicit StartBarrier(int participant_count)
      : participant_count_(participant_count) {}

  void ArriveAndWait() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++arrived_count_;
    if (arrived_count_ == participant_count_) {
      released_ = true;
      condition_.notify_all();
      return;
    }
    condition_.wait(lock, [this] { return released_; });
  }

 private:
  // Number of threads that must reach the barrier.
  const int participant_count_;
  // Serializes barrier state.
  std::mutex mutex_;
  // Announces that every participant has arrived.
  std::condition_variable condition_;
  // Number of participants currently at the barrier.
  int arrived_count_ = 0;
  // True after the final participant arrives.
  bool released_ = false;
};

class HipStreamValueApiTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!api_.library) {
      api_.library = dlopen(CandidateLibPath(), RTLD_LAZY | RTLD_LOCAL);
      if (!api_.library) {
        GTEST_SKIP() << "cannot dlopen " << CandidateLibPath() << ": "
                     << dlerror();
      }

      api_.init = ResolveHipSymbol<HipInitFn>(api_.library, "hipInit");
      api_.malloc = ResolveHipSymbol<HipMallocFn>(api_.library, "hipMalloc");
      api_.free = ResolveHipSymbol<HipFreeFn>(api_.library, "hipFree");
      api_.malloc_host =
          ResolveHipSymbol<HipMallocHostFn>(api_.library, "hipMallocHost");
      api_.free_host =
          ResolveHipSymbol<HipFreeHostFn>(api_.library, "hipFreeHost");
      api_.host_get_device_pointer =
          ResolveHipSymbol<HipHostGetDevicePointerFn>(
              api_.library, "hipHostGetDevicePointer");
      api_.memset = ResolveHipSymbol<HipMemsetFn>(api_.library, "hipMemset");
      api_.memcpy = ResolveHipSymbol<HipMemcpyFn>(api_.library, "hipMemcpy");
      api_.stream_create =
          ResolveHipSymbol<HipStreamCreateFn>(api_.library, "hipStreamCreate");
      api_.stream_destroy = ResolveHipSymbol<HipStreamDestroyFn>(
          api_.library, "hipStreamDestroy");
      api_.stream_synchronize = ResolveHipSymbol<HipStreamSynchronizeFn>(
          api_.library, "hipStreamSynchronize");
      api_.write_value_32 = ResolveHipSymbol<HipStreamWriteValue32Fn>(
          api_.library, "hipStreamWriteValue32");
      api_.write_value_64 = ResolveHipSymbol<HipStreamWriteValue64Fn>(
          api_.library, "hipStreamWriteValue64");
      api_.wait_value_32 = ResolveHipSymbol<HipStreamWaitValue32Fn>(
          api_.library, "hipStreamWaitValue32");
      api_.wait_value_64 = ResolveHipSymbol<HipStreamWaitValue64Fn>(
          api_.library, "hipStreamWaitValue64");
      api_.batch_mem_op = ResolveHipSymbol<HipStreamBatchMemOpFn>(
          api_.library, "hipStreamBatchMemOp");
      api_.stream_begin_capture = ResolveHipSymbol<HipStreamBeginCaptureFn>(
          api_.library, "hipStreamBeginCapture");
      api_.stream_end_capture = ResolveHipSymbol<HipStreamEndCaptureFn>(
          api_.library, "hipStreamEndCapture");
      api_.stream_is_capturing = ResolveHipSymbol<HipStreamIsCapturingFn>(
          api_.library, "hipStreamIsCapturing");
      api_.graph_instantiate = ResolveHipSymbol<HipGraphInstantiateFn>(
          api_.library, "hipGraphInstantiate");
      api_.graph_launch =
          ResolveHipSymbol<HipGraphLaunchFn>(api_.library, "hipGraphLaunch");
      api_.graph_exec_destroy = ResolveHipSymbol<HipGraphExecDestroyFn>(
          api_.library, "hipGraphExecDestroy");
      api_.graph_destroy =
          ResolveHipSymbol<HipGraphDestroyFn>(api_.library, "hipGraphDestroy");
    }

    ASSERT_NE(nullptr, api_.init);
    ASSERT_NE(nullptr, api_.malloc);
    ASSERT_NE(nullptr, api_.free);
    ASSERT_NE(nullptr, api_.malloc_host);
    ASSERT_NE(nullptr, api_.free_host);
    ASSERT_NE(nullptr, api_.host_get_device_pointer);
    ASSERT_NE(nullptr, api_.memset);
    ASSERT_NE(nullptr, api_.memcpy);
    ASSERT_NE(nullptr, api_.stream_create);
    ASSERT_NE(nullptr, api_.stream_destroy);
    ASSERT_NE(nullptr, api_.stream_synchronize);
    ASSERT_NE(nullptr, api_.write_value_32);
    ASSERT_NE(nullptr, api_.write_value_64);
    ASSERT_NE(nullptr, api_.wait_value_32);
    ASSERT_NE(nullptr, api_.wait_value_64);
    ASSERT_NE(nullptr, api_.batch_mem_op);
    ASSERT_NE(nullptr, api_.stream_begin_capture);
    ASSERT_NE(nullptr, api_.stream_end_capture);
    ASSERT_NE(nullptr, api_.stream_is_capturing);
    ASSERT_NE(nullptr, api_.graph_instantiate);
    ASSERT_NE(nullptr, api_.graph_launch);
    ASSERT_NE(nullptr, api_.graph_exec_destroy);
    ASSERT_NE(nullptr, api_.graph_destroy);

    const hipError_t init_result = api_.init(/*flags=*/0);
    if (init_result != hipSuccess) {
      GTEST_SKIP() << "hipInit failed: " << init_result;
    }
  }

  void TearDown() override {
    for (hipGraphExec_t executable : graph_executables_) {
      EXPECT_EQ(hipSuccess, api_.graph_exec_destroy(executable));
    }
    for (hipGraph_t graph : graphs_) {
      EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
    }
    for (hipStream_t stream : streams_) {
      EXPECT_EQ(hipSuccess, api_.stream_destroy(stream));
    }
    for (void* allocation : allocations_) {
      EXPECT_EQ(hipSuccess, api_.free(allocation));
    }
    for (void* allocation : host_allocations_) {
      EXPECT_EQ(hipSuccess, api_.free_host(allocation));
    }
  }

  hipStream_t CreateStream() {
    hipStream_t stream = nullptr;
    EXPECT_EQ(hipSuccess, api_.stream_create(&stream));
    if (stream) streams_.push_back(stream);
    return stream;
  }

  void* Allocate(size_t size) {
    void* pointer = nullptr;
    EXPECT_EQ(hipSuccess, api_.malloc(&pointer, size));
    if (pointer) allocations_.push_back(pointer);
    return pointer;
  }

  void* AllocateHost(size_t size) {
    void* pointer = nullptr;
    EXPECT_EQ(hipSuccess, api_.malloc_host(&pointer, size));
    if (pointer) host_allocations_.push_back(pointer);
    return pointer;
  }

  void ForgetAllocation(void* pointer) {
    auto it = std::find(allocations_.begin(), allocations_.end(), pointer);
    ASSERT_NE(allocations_.end(), it);
    allocations_.erase(it);
  }

  void ForgetStream(hipStream_t stream) {
    auto it = std::find(streams_.begin(), streams_.end(), stream);
    ASSERT_NE(streams_.end(), it);
    streams_.erase(it);
  }

  static HipRuntimeApi api_;
  std::vector<hipStream_t> streams_;
  std::vector<void*> allocations_;
  std::vector<void*> host_allocations_;
  std::vector<hipGraph_t> graphs_;
  std::vector<hipGraphExec_t> graph_executables_;
};

HipRuntimeApi HipStreamValueApiTest::api_;

TEST_F(HipStreamValueApiTest, ExecutesScalarWritesThroughPublicDso) {
  hipStream_t stream = CreateStream();
  void* allocation = Allocate(16);
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, allocation);
  ASSERT_EQ(hipSuccess, api_.memset(allocation, 0, 16));

  void* value_32 = allocation;
  void* value_64 = static_cast<uint8_t*>(allocation) + 8;
  EXPECT_EQ(hipSuccess, api_.write_value_32(stream, value_32, 7,
                                            hipStreamWriteValueDefault));
  EXPECT_EQ(hipSuccess, api_.write_value_32(stream, value_32, 5,
                                            hipExtStreamWriteValueIncrement));
  EXPECT_EQ(hipSuccess, api_.write_value_32(stream, value_32, 2,
                                            hipExtStreamWriteValueDecrement));
  EXPECT_EQ(hipSuccess, api_.write_value_64(stream, value_64, 11,
                                            hipStreamWriteValueDefault));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream));

  uint32_t observed_32 = 0;
  uint64_t observed_64 = 0;
  EXPECT_EQ(hipSuccess, api_.memcpy(&observed_32, value_32, sizeof(observed_32),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(hipSuccess, api_.memcpy(&observed_64, value_64, sizeof(observed_64),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(10u, observed_32);
  EXPECT_EQ(11u, observed_64);
}

TEST_F(HipStreamValueApiTest, ExecutesEveryWaitPredicateAtBothWidths) {
  hipStream_t stream = CreateStream();
  void* allocation = Allocate(48);
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, allocation);

  const uint32_t initial_32[] = {7, 9, 4, 5};
  const uint64_t initial_64[] = {7, 9, 4, 5};
  ASSERT_EQ(hipSuccess, api_.memcpy(allocation, initial_32, sizeof(initial_32),
                                    hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess,
            api_.memcpy(static_cast<uint8_t*>(allocation) + 16, initial_64,
                        sizeof(initial_64), hipMemcpyHostToDevice));

  const unsigned int predicates[] = {
      hipStreamWaitValueGte,
      hipStreamWaitValueEq,
      hipStreamWaitValueAnd,
      hipStreamWaitValueNor,
  };
  const uint32_t expected_32[] = {5, 9, 4, 5};
  const uint64_t expected_64[] = {5, 9, 4, 5};
  const uint32_t masks_32[] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, 0xFu};
  const uint64_t masks_64[] = {UINT64_MAX, UINT64_MAX, UINT64_MAX, 0xFu};
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(hipSuccess, api_.wait_value_32(
                              stream, static_cast<uint8_t*>(allocation) + i * 4,
                              expected_32[i], predicates[i], masks_32[i]));
    EXPECT_EQ(hipSuccess,
              api_.wait_value_64(stream,
                                 static_cast<uint8_t*>(allocation) + 16 + i * 8,
                                 expected_64[i], predicates[i], masks_64[i]));
  }
  EXPECT_EQ(hipSuccess, api_.stream_synchronize(stream));
}

TEST_F(HipStreamValueApiTest, ExecutesSystemScopeOperationsOnHostMemory) {
  hipStream_t stream = CreateStream();
  void* host_allocation = AllocateHost(16);
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, host_allocation);

  hipDeviceptr_t device_pointer = nullptr;
  ASSERT_EQ(hipSuccess, api_.host_get_device_pointer(
                            &device_pointer, host_allocation, /*flags=*/0));
  ASSERT_NE(nullptr, device_pointer);
  std::memset(host_allocation, 0, 16);

  void* value_32 = device_pointer;
  void* value_64 = static_cast<uint8_t*>(device_pointer) + 8;
  EXPECT_EQ(hipSuccess, api_.write_value_32(stream, value_32, 7,
                                            hipStreamWriteValueDefault));
  EXPECT_EQ(hipSuccess, api_.write_value_32(stream, value_32, 5,
                                            hipExtStreamWriteValueIncrement));
  EXPECT_EQ(hipSuccess, api_.write_value_32(stream, value_32, 2,
                                            hipExtStreamWriteValueDecrement));
  EXPECT_EQ(hipSuccess, api_.write_value_64(stream, value_64, 11,
                                            hipStreamWriteValueDefault));
  EXPECT_EQ(hipSuccess, api_.write_value_64(stream, value_64, 7,
                                            hipExtStreamWriteValueIncrement));
  EXPECT_EQ(hipSuccess, api_.write_value_64(stream, value_64, 3,
                                            hipExtStreamWriteValueDecrement));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream));

  uint32_t observed_32 = 0;
  uint64_t observed_64 = 0;
  std::memcpy(&observed_32, host_allocation, sizeof(observed_32));
  std::memcpy(&observed_64, static_cast<uint8_t*>(host_allocation) + 8,
              sizeof(observed_64));
  EXPECT_EQ(10u, observed_32);
  EXPECT_EQ(15u, observed_64);

  EXPECT_EQ(hipSuccess, api_.wait_value_32(stream, value_32, observed_32,
                                           hipStreamWaitValueEq, UINT32_MAX));
  EXPECT_EQ(hipSuccess, api_.wait_value_64(stream, value_64, observed_64,
                                           hipStreamWaitValueEq, UINT64_MAX));
  EXPECT_EQ(hipSuccess, api_.stream_synchronize(stream));
}

TEST_F(HipStreamValueApiTest, ExecutesBatchAsOneOrderedTransaction) {
  hipStream_t stream = CreateStream();
  void* allocation = Allocate(16);
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, allocation);
  ASSERT_EQ(hipSuccess, api_.memset(allocation, 0, 16));

  hipStreamBatchMemOpParams parameters[4] = {};
  parameters[0].writeValue.operation = hipStreamMemOpWriteValue32;
  parameters[0].writeValue.address = (hipDeviceptr_t)(uintptr_t)allocation;
  parameters[0].writeValue.value = 7;
  parameters[0].writeValue.flags = hipStreamWriteValueDefault;
  parameters[1].writeValue.operation = hipStreamMemOpWriteValue32;
  parameters[1].writeValue.address = (hipDeviceptr_t)(uintptr_t)allocation;
  parameters[1].writeValue.value = 5;
  parameters[1].writeValue.flags = hipExtStreamWriteValueIncrement;
  parameters[2].writeValue.operation = hipStreamMemOpWriteValue64;
  parameters[2].writeValue.address =
      (hipDeviceptr_t)(uintptr_t)(static_cast<uint8_t*>(allocation) + 8);
  parameters[2].writeValue.value64 = 20;
  parameters[2].writeValue.flags = hipStreamWriteValueDefault;
  parameters[3].writeValue.operation = hipStreamMemOpWriteValue64;
  parameters[3].writeValue.address = parameters[2].writeValue.address;
  parameters[3].writeValue.value64 = 3;
  parameters[3].writeValue.flags = hipExtStreamWriteValueDecrement;

  EXPECT_EQ(hipSuccess, api_.batch_mem_op(stream, 4, parameters, /*flags=*/0));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream));

  uint32_t observed_32 = 0;
  uint64_t observed_64 = 0;
  EXPECT_EQ(hipSuccess,
            api_.memcpy(&observed_32, allocation, sizeof(observed_32),
                        hipMemcpyDeviceToHost));
  EXPECT_EQ(hipSuccess,
            api_.memcpy(&observed_64, static_cast<uint8_t*>(allocation) + 8,
                        sizeof(observed_64), hipMemcpyDeviceToHost));
  EXPECT_EQ(12u, observed_32);
  EXPECT_EQ(17u, observed_64);
}

TEST_F(HipStreamValueApiTest, ExecutesMaximumBatchCount) {
  hipStream_t stream = CreateStream();
  void* allocation = Allocate(sizeof(uint32_t));
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, allocation);
  ASSERT_EQ(hipSuccess, api_.memset(allocation, 0, sizeof(uint32_t)));

  std::vector<hipStreamBatchMemOpParams> parameters(256);
  for (hipStreamBatchMemOpParams& parameter : parameters) {
    parameter.writeValue.operation = hipStreamMemOpWriteValue32;
    parameter.writeValue.address = (hipDeviceptr_t)(uintptr_t)allocation;
    parameter.writeValue.value = 1;
    parameter.writeValue.flags = hipExtStreamWriteValueIncrement;
  }
  ASSERT_EQ(
      hipSuccess,
      api_.batch_mem_op(stream, static_cast<unsigned int>(parameters.size()),
                        parameters.data(), /*flags=*/0));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream));

  uint32_t observed = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(&observed, allocation, sizeof(observed),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(256u, observed);
}

TEST_F(HipStreamValueApiTest, BatchDoesNotInterleaveWithSameStreamCall) {
  hipStream_t stream = CreateStream();
  void* allocation = Allocate(sizeof(uint32_t));
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, allocation);

  for (int iteration = 0; iteration < 128; ++iteration) {
    ASSERT_EQ(hipSuccess, api_.memset(allocation, 0, sizeof(uint32_t)));
    hipStreamBatchMemOpParams parameters[2] = {};
    parameters[0].writeValue.operation = hipStreamMemOpWriteValue32;
    parameters[0].writeValue.address = (hipDeviceptr_t)(uintptr_t)allocation;
    parameters[0].writeValue.value = 1;
    parameters[0].writeValue.flags = hipStreamWriteValueDefault;
    parameters[1].writeValue.operation = hipStreamMemOpWriteValue32;
    parameters[1].writeValue.address = (hipDeviceptr_t)(uintptr_t)allocation;
    parameters[1].writeValue.value = 1;
    parameters[1].writeValue.flags = hipExtStreamWriteValueIncrement;

    StartBarrier barrier(/*participant_count=*/2);
    hipError_t batch_result = hipErrorUnknown;
    hipError_t scalar_result = hipErrorUnknown;
    std::thread batch_thread([&] {
      barrier.ArriveAndWait();
      batch_result = api_.batch_mem_op(stream, 2, parameters, /*flags=*/0);
    });
    std::thread scalar_thread([&] {
      barrier.ArriveAndWait();
      scalar_result = api_.write_value_32(stream, allocation, 10,
                                          hipStreamWriteValueDefault);
    });
    batch_thread.join();
    scalar_thread.join();
    ASSERT_EQ(hipSuccess, batch_result);
    ASSERT_EQ(hipSuccess, scalar_result);
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream));

    uint32_t observed = 0;
    ASSERT_EQ(hipSuccess, api_.memcpy(&observed, allocation, sizeof(observed),
                                      hipMemcpyDeviceToHost));
    EXPECT_TRUE(observed == 2 || observed == 10)
        << "same-stream call interleaved within batch at iteration "
        << iteration;
  }
}

TEST_F(HipStreamValueApiTest, IndependentStreamWaitsDoNotShareAQueueLane) {
  hipStream_t stream_a = CreateStream();
  hipStream_t stream_b = CreateStream();
  void* allocation = Allocate(24);
  ASSERT_NE(nullptr, stream_a);
  ASSERT_NE(nullptr, stream_b);
  ASSERT_NE(nullptr, allocation);
  ASSERT_EQ(hipSuccess, api_.memset(allocation, 0, 24));

  void* value_x = allocation;
  void* value_y = static_cast<uint8_t*>(allocation) + 4;
  void* value_x_64 = static_cast<uint8_t*>(allocation) + 8;
  void* value_y_64 = static_cast<uint8_t*>(allocation) + 16;
  ASSERT_EQ(hipSuccess, api_.wait_value_32(stream_a, value_x, 1,
                                           hipStreamWaitValueEq, UINT32_MAX));
  ASSERT_EQ(hipSuccess, api_.wait_value_64(stream_a, value_x_64, 1,
                                           hipStreamWaitValueEq, UINT64_MAX));
  ASSERT_EQ(hipSuccess, api_.write_value_32(stream_b, value_y, 1,
                                            hipStreamWriteValueDefault));
  ASSERT_EQ(hipSuccess, api_.write_value_64(stream_b, value_y_64, 1,
                                            hipStreamWriteValueDefault));

  hipStreamBatchMemOpParams producer[4] = {};
  producer[0].waitValue.operation = hipStreamMemOpWaitValue32;
  producer[0].waitValue.address = (hipDeviceptr_t)(uintptr_t)value_y;
  producer[0].waitValue.value = 1;
  producer[0].waitValue.flags = hipStreamWaitValueEq;
  producer[1].writeValue.operation = hipStreamMemOpWriteValue32;
  producer[1].writeValue.address = (hipDeviceptr_t)(uintptr_t)value_x;
  producer[1].writeValue.value = 1;
  producer[1].writeValue.flags = hipStreamWriteValueDefault;
  producer[2].waitValue.operation = hipStreamMemOpWaitValue64;
  producer[2].waitValue.address = (hipDeviceptr_t)(uintptr_t)value_y_64;
  producer[2].waitValue.value64 = 1;
  producer[2].waitValue.flags = hipStreamWaitValueEq;
  producer[3].writeValue.operation = hipStreamMemOpWriteValue64;
  producer[3].writeValue.address = (hipDeviceptr_t)(uintptr_t)value_x_64;
  producer[3].writeValue.value64 = 1;
  producer[3].writeValue.flags = hipStreamWriteValueDefault;
  ASSERT_EQ(hipSuccess, api_.batch_mem_op(stream_b, 4, producer, /*flags=*/0));

  EXPECT_EQ(hipSuccess, api_.stream_synchronize(stream_b));
  EXPECT_EQ(hipSuccess, api_.stream_synchronize(stream_a));
}

TEST_F(HipStreamValueApiTest, CapturedDefaultWritesReplayAtBothWidths) {
  hipStream_t stream = CreateStream();
  void* allocation = Allocate(16);
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, allocation);
  ASSERT_EQ(hipSuccess, api_.memset(allocation, 0, 16));

  ASSERT_EQ(hipSuccess,
            api_.stream_begin_capture(stream, hipStreamCaptureModeGlobal));
  EXPECT_EQ(hipSuccess, api_.write_value_32(stream, allocation, 13,
                                            hipStreamWriteValueDefault));
  EXPECT_EQ(hipSuccess,
            api_.write_value_64(stream, static_cast<uint8_t*>(allocation) + 8,
                                29, hipStreamWriteValueDefault));

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.stream_end_capture(stream, &graph));
  ASSERT_NE(nullptr, graph);
  graphs_.push_back(graph);

  hipGraphExec_t executable = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_instantiate(&executable, graph, /*error_node=*/nullptr,
                                   /*log_buffer=*/nullptr, /*buffer_size=*/0));
  ASSERT_NE(nullptr, executable);
  graph_executables_.push_back(executable);

  for (int replay = 0; replay < 2; ++replay) {
    ASSERT_EQ(hipSuccess, api_.memset(allocation, 0, 16));
    ASSERT_EQ(hipSuccess, api_.graph_launch(executable, stream));
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream));

    uint32_t observed_32 = 0;
    uint64_t observed_64 = 0;
    EXPECT_EQ(hipSuccess,
              api_.memcpy(&observed_32, allocation, sizeof(observed_32),
                          hipMemcpyDeviceToHost));
    EXPECT_EQ(hipSuccess,
              api_.memcpy(&observed_64, static_cast<uint8_t*>(allocation) + 8,
                          sizeof(observed_64), hipMemcpyDeviceToHost));
    EXPECT_EQ(13u, observed_32);
    EXPECT_EQ(29u, observed_64);
  }
}

TEST_F(HipStreamValueApiTest, UnsupportedWaitInvalidatesCapture) {
  hipStream_t stream = CreateStream();
  void* allocation = Allocate(8);
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, allocation);

  ASSERT_EQ(hipSuccess,
            api_.stream_begin_capture(stream, hipStreamCaptureModeGlobal));
  EXPECT_EQ(hipErrorStreamCaptureUnsupported,
            api_.wait_value_32(stream, allocation, 1, hipStreamWaitValueEq,
                               UINT32_MAX));
  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
  EXPECT_EQ(hipSuccess, api_.stream_is_capturing(stream, &capture_status));
  EXPECT_EQ(hipStreamCaptureStatusInvalidated, capture_status);

  hipGraph_t graph = nullptr;
  EXPECT_EQ(hipErrorStreamCaptureInvalidated,
            api_.stream_end_capture(stream, &graph));
  EXPECT_EQ(nullptr, graph);
}

TEST_F(HipStreamValueApiTest, CaptureBeginRacesValueWaitSubmission) {
  hipStream_t stream = CreateStream();
  void* allocation = Allocate(sizeof(uint32_t));
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, allocation);

  for (int iteration = 0; iteration < 64; ++iteration) {
    ASSERT_EQ(hipSuccess, api_.memset(allocation, 0, sizeof(uint32_t)));
    StartBarrier barrier(/*participant_count=*/2);
    hipError_t wait_result = hipErrorUnknown;
    hipError_t begin_result = hipErrorUnknown;
    std::thread wait_thread([&] {
      barrier.ArriveAndWait();
      wait_result = api_.wait_value_32(stream, allocation, 0,
                                       hipStreamWaitValueEq, UINT32_MAX);
    });
    std::thread capture_thread([&] {
      barrier.ArriveAndWait();
      begin_result =
          api_.stream_begin_capture(stream, hipStreamCaptureModeRelaxed);
    });
    wait_thread.join();
    capture_thread.join();
    ASSERT_EQ(hipSuccess, begin_result);

    hipGraph_t graph = nullptr;
    const hipError_t end_result = api_.stream_end_capture(stream, &graph);
    if (wait_result == hipSuccess) {
      EXPECT_EQ(hipSuccess, end_result);
      ASSERT_NE(nullptr, graph);
      graphs_.push_back(graph);
    } else {
      EXPECT_EQ(hipErrorStreamCaptureUnsupported, wait_result);
      EXPECT_EQ(hipErrorStreamCaptureInvalidated, end_result);
      EXPECT_EQ(nullptr, graph);
    }
  }
}

TEST_F(HipStreamValueApiTest, CaptureEndRacesValueWaitSubmission) {
  hipStream_t stream = CreateStream();
  void* allocation = Allocate(sizeof(uint32_t));
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, allocation);
  ASSERT_EQ(hipSuccess, api_.memset(allocation, 0, sizeof(uint32_t)));

  for (int iteration = 0; iteration < 64; ++iteration) {
    ASSERT_EQ(hipSuccess,
              api_.stream_begin_capture(stream, hipStreamCaptureModeRelaxed));
    StartBarrier barrier(/*participant_count=*/2);
    hipError_t wait_result = hipErrorUnknown;
    hipError_t end_result = hipErrorUnknown;
    hipGraph_t graph = nullptr;
    std::thread wait_thread([&] {
      barrier.ArriveAndWait();
      wait_result = api_.wait_value_32(stream, allocation, 0,
                                       hipStreamWaitValueEq, UINT32_MAX);
    });
    std::thread capture_thread([&] {
      barrier.ArriveAndWait();
      end_result = api_.stream_end_capture(stream, &graph);
    });
    wait_thread.join();
    capture_thread.join();

    if (wait_result == hipSuccess) {
      EXPECT_EQ(hipSuccess, end_result);
      ASSERT_NE(nullptr, graph);
      graphs_.push_back(graph);
    } else {
      EXPECT_EQ(hipErrorStreamCaptureUnsupported, wait_result);
      EXPECT_EQ(hipErrorStreamCaptureInvalidated, end_result);
      EXPECT_EQ(nullptr, graph);
    }
  }
}

TEST_F(HipStreamValueApiTest, InvalidBatchDoesNotPartiallyCommit) {
  hipStream_t stream = CreateStream();
  void* allocation = Allocate(8);
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, allocation);
  ASSERT_EQ(hipSuccess, api_.memset(allocation, 0, 8));

  hipStreamBatchMemOpParams parameters[2] = {};
  parameters[0].writeValue.operation = hipStreamMemOpWriteValue32;
  parameters[0].writeValue.address = (hipDeviceptr_t)(uintptr_t)allocation;
  parameters[0].writeValue.value = 17;
  parameters[0].writeValue.flags = hipStreamWriteValueDefault;
  parameters[1].waitValue.operation = hipStreamMemOpWaitValue32;
  parameters[1].waitValue.address = 0;
  parameters[1].waitValue.value = 1;
  parameters[1].waitValue.flags = hipStreamWaitValueEq;

  EXPECT_EQ(hipErrorInvalidValue,
            api_.batch_mem_op(stream, 2, parameters, /*flags=*/0));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream));
  uint32_t observed = 1;
  EXPECT_EQ(hipSuccess, api_.memcpy(&observed, allocation, sizeof(observed),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(0u, observed);
}

TEST_F(HipStreamValueApiTest, ScalarTargetLookupRacesFreeWithoutDangling) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);

  for (int iteration = 0; iteration < 64; ++iteration) {
    void* allocation = Allocate(8);
    ASSERT_NE(nullptr, allocation);
    StartBarrier barrier(/*participant_count=*/2);
    hipError_t operation_result = hipErrorUnknown;
    hipError_t free_result = hipErrorUnknown;
    std::thread operation_thread([&] {
      barrier.ArriveAndWait();
      operation_result = api_.write_value_64(stream, allocation, 7,
                                             hipStreamWriteValueDefault);
    });
    std::thread free_thread([&] {
      barrier.ArriveAndWait();
      free_result = api_.free(allocation);
    });
    operation_thread.join();
    free_thread.join();
    ForgetAllocation(allocation);

    EXPECT_EQ(hipSuccess, free_result);
    EXPECT_TRUE(operation_result == hipSuccess ||
                operation_result == hipErrorInvalidValue);
  }
}

TEST_F(HipStreamValueApiTest, BatchTargetLookupRacesFreeWithoutPartialCommit) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);

  for (int iteration = 0; iteration < 64; ++iteration) {
    void* allocation = Allocate(8);
    ASSERT_NE(nullptr, allocation);
    hipStreamBatchMemOpParams parameters[2] = {};
    for (hipStreamBatchMemOpParams& parameter : parameters) {
      parameter.writeValue.operation = hipStreamMemOpWriteValue64;
      parameter.writeValue.address = (hipDeviceptr_t)(uintptr_t)allocation;
      parameter.writeValue.value64 = 1;
      parameter.writeValue.flags = hipExtStreamWriteValueIncrement;
    }

    StartBarrier barrier(/*participant_count=*/2);
    hipError_t operation_result = hipErrorUnknown;
    hipError_t free_result = hipErrorUnknown;
    std::thread operation_thread([&] {
      barrier.ArriveAndWait();
      operation_result = api_.batch_mem_op(stream, 2, parameters, /*flags=*/0);
    });
    std::thread free_thread([&] {
      barrier.ArriveAndWait();
      free_result = api_.free(allocation);
    });
    operation_thread.join();
    free_thread.join();
    ForgetAllocation(allocation);

    EXPECT_EQ(hipSuccess, free_result);
    EXPECT_TRUE(operation_result == hipSuccess ||
                operation_result == hipErrorInvalidValue);
  }
}

TEST_F(HipStreamValueApiTest, StreamTeardownCompletesAfterIndependentProducer) {
  hipStream_t wait_stream = CreateStream();
  hipStream_t producer_stream = CreateStream();
  void* allocation = Allocate(sizeof(uint32_t));
  ASSERT_NE(nullptr, wait_stream);
  ASSERT_NE(nullptr, producer_stream);
  ASSERT_NE(nullptr, allocation);
  ASSERT_EQ(hipSuccess, api_.memset(allocation, 0, sizeof(uint32_t)));
  ASSERT_EQ(hipSuccess, api_.wait_value_32(wait_stream, allocation, 1,
                                           hipStreamWaitValueEq, UINT32_MAX));

  ForgetStream(wait_stream);
  hipError_t destroy_result = hipErrorUnknown;
  std::thread destroy_thread(
      [&] { destroy_result = api_.stream_destroy(wait_stream); });
  EXPECT_EQ(hipSuccess, api_.write_value_32(producer_stream, allocation, 1,
                                            hipStreamWriteValueDefault));
  EXPECT_EQ(hipSuccess, api_.stream_synchronize(producer_stream));
  destroy_thread.join();
  EXPECT_EQ(hipSuccess, destroy_result);
}

}  // namespace
