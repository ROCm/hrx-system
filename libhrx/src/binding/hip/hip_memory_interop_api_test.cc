// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <semaphore>
#include <utility>

#include "api.h"
#include "common/context.h"
#include "common/stream.h"
#include "hip_dso_test_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

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
    stream_destroy_ =
        dso_.Resolve<decltype(&hipStreamDestroy)>("hipStreamDestroy");
    graph_destroy_ =
        dso_.Resolve<decltype(&hipGraphDestroy)>("hipGraphDestroy");
    graph_exec_destroy_ =
        dso_.Resolve<decltype(&hipGraphExecDestroy)>("hipGraphExecDestroy");
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
    ASSERT_NE(stream_destroy_, nullptr);
    ASSERT_NE(graph_destroy_, nullptr);
    ASSERT_NE(graph_exec_destroy_, nullptr);
    ready_ = true;
  }

  void TearDown() override {
    if (ready_) {
      callback_.release.release();
      EXPECT_EQ(hipSuccess, synchronize_());
      if (graph_executable_) {
        EXPECT_EQ(hipSuccess, graph_exec_destroy_(graph_executable_));
      }
      if (graph_) {
        EXPECT_EQ(hipSuccess, graph_destroy_(graph_));
      }
      if (copy_stream_) {
        EXPECT_EQ(hipSuccess, stream_destroy_(copy_stream_));
      }
      if (callback_stream_) {
        EXPECT_EQ(hipSuccess, stream_destroy_(callback_stream_));
      }
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

  static void HoldCallback(void* user_data) {
    auto* fixture = static_cast<HipMemoryInteropApiTest*>(user_data);
    fixture->callback_.entered.release();
    fixture->callback_.release.acquire();
    fixture->callback_.returned.store(true, std::memory_order_release);
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
  // Stream carrying bounded registered-host transfers and graph replay.
  hipStream_t copy_stream_ = nullptr;
  // Distinct nonblocking stream carrying the held application callback.
  hipStream_t callback_stream_ = nullptr;
  // Captured native transfers, retained until all launches complete.
  hipGraph_t graph_ = nullptr;
  // Compiled capture released before its template and memory owners.
  hipGraphExec_t graph_executable_ = nullptr;
  // Host-only rendezvous released before teardown waits for any device work.
  struct {
    // Announces that the frontend worker is executing the callback.
    std::binary_semaphore entered{0};
    // Lets the held callback return on every exit from the test body.
    std::binary_semaphore release{0};
    // True only after the test's teardown has released the callback.
    std::atomic<bool> returned{false};
  } callback_;
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
  // Releases streams only after the device has completed their work.
  decltype(&hipStreamDestroy) stream_destroy_ = nullptr;
  // Releases the captured native-copy template.
  decltype(&hipGraphDestroy) graph_destroy_ = nullptr;
  // Releases a compiled native-copy capture.
  decltype(&hipGraphExecDestroy) graph_exec_destroy_ = nullptr;
};

// Replaces a quiescent stream's synchronization dependency and restores its
// original owner on every exit, before fixture cleanup can submit more work.
class ScopedStreamTimeline {
 public:
  ScopedStreamTimeline(iree_hal_streaming_stream_t* stream,
                       iree_hal_semaphore_t* replacement,
                       decltype(&iree_hal_resource_release) release)
      : stream_(stream),
        original_(std::exchange(stream->timeline_semaphore, replacement)),
        release_(release) {}

  ScopedStreamTimeline(const ScopedStreamTimeline&) = delete;
  ScopedStreamTimeline& operator=(const ScopedStreamTimeline&) = delete;

  ~ScopedStreamTimeline() {
    release_(std::exchange(stream_->timeline_semaphore, original_));
  }

 private:
  // Borrowed stream, quiescent for the entire replacement scope.
  iree_hal_streaming_stream_t* stream_;
  // Owned timeline reference transferred back to |stream_| on exit.
  iree_hal_semaphore_t* original_;
  // Resource release from the same DSO that created the replacement timeline.
  decltype(&iree_hal_resource_release) release_;
};

TEST_F(HipMemoryInteropApiTest,
       SynchronousCopiesPreserveProducerFailureAndDestinationBytes) {
  const auto current_context =
      dso_.ResolveLocalForTest<decltype(&iree_hal_streaming_context_current)>(
          "iree_hal_streaming_context_current");
  const auto create_semaphore =
      dso_.ResolveLocalForTest<decltype(&iree_hal_semaphore_create)>(
          "iree_hal_semaphore_create");
  const auto fail_semaphore =
      dso_.ResolveLocalForTest<decltype(&iree_hal_semaphore_fail)>(
          "iree_hal_semaphore_fail");
  const auto release_resource =
      dso_.ResolveLocalForTest<decltype(&iree_hal_resource_release)>(
          "iree_hal_resource_release");
  const auto get_last_error =
      dso_.Resolve<decltype(&hipGetLastError)>("hipGetLastError");
  ASSERT_NE(nullptr, current_context) << dso_.error();
  ASSERT_NE(nullptr, create_semaphore) << dso_.error();
  ASSERT_NE(nullptr, fail_semaphore) << dso_.error();
  ASSERT_NE(nullptr, release_resource) << dso_.error();
  ASSERT_NE(nullptr, get_last_error) << dso_.error();

  constexpr size_t kLength = 16;
  ASSERT_EQ(hipSuccess, allocate_(&device_memory_, 2 * kLength));
  auto* source = static_cast<uint8_t*>(device_memory_);
  auto* target = source + kLength;
  std::array<uint8_t, kLength> input;
  input.fill(0x37);
  std::array<uint8_t, kLength> initial_target;
  initial_target.fill(0xC7);
  std::array<uint8_t, kLength> output;
  output.fill(0xA3);
  ASSERT_EQ(hipSuccess,
            copy_(source, input.data(), kLength, hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, copy_(target, initial_target.data(), kLength,
                              hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, synchronize_());

  auto* context = current_context();
  ASSERT_NE(nullptr, context);
  auto* stream = context->default_stream;
  ASSERT_NE(nullptr, stream);
  iree_hal_semaphore_t* failed_timeline = nullptr;
  IREE_ASSERT_OK(create_semaphore(
      context->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      stream->pending_value, IREE_HAL_SEMAPHORE_FLAG_NONE, &failed_timeline));
  fail_semaphore(failed_timeline,
                 iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED));
  {
    ScopedStreamTimeline timeline(stream, failed_timeline, release_resource);
    EXPECT_EQ(hipErrorOutOfMemory,
              copy_(target, input.data(), kLength, hipMemcpyHostToDevice));
    EXPECT_EQ(hipErrorOutOfMemory,
              copy_(output.data(), source, kLength, hipMemcpyDeviceToHost));
    EXPECT_EQ(hipErrorOutOfMemory,
              copy_(target, source, kLength, hipMemcpyDeviceToDevice));
  }
  std::array<uint8_t, kLength> untouched_output;
  untouched_output.fill(0xA3);
  EXPECT_EQ(untouched_output, output);
  ASSERT_EQ(hipSuccess,
            copy_(output.data(), target, kLength, hipMemcpyDeviceToHost));
  EXPECT_EQ(initial_target, output);
  EXPECT_EQ(hipErrorOutOfMemory, get_last_error());
}

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

TEST_F(HipMemoryInteropApiTest,
       RegisteredCopiesProgressWhileAnotherStreamCallbackIsBlocked) {
  constexpr size_t kLength = 16;
  const auto copy_async =
      dso_.Resolve<decltype(&hipMemcpyAsync)>("hipMemcpyAsync");
  const auto create_stream =
      dso_.Resolve<decltype(&hipStreamCreateWithPriority)>(
          "hipStreamCreateWithPriority");
  const auto synchronize_stream =
      dso_.Resolve<decltype(&hipStreamSynchronize)>("hipStreamSynchronize");
  const auto launch_callback =
      dso_.Resolve<decltype(&hipLaunchHostFunc)>("hipLaunchHostFunc");
  const auto begin_capture =
      dso_.Resolve<decltype(&hipStreamBeginCapture)>("hipStreamBeginCapture");
  const auto end_capture =
      dso_.Resolve<decltype(&hipStreamEndCapture)>("hipStreamEndCapture");
  const auto get_nodes =
      dso_.Resolve<decltype(&hipGraphGetNodes)>("hipGraphGetNodes");
  const auto instantiate =
      dso_.Resolve<decltype(&hipGraphInstantiate)>("hipGraphInstantiate");
  const auto launch_graph =
      dso_.Resolve<decltype(&hipGraphLaunch)>("hipGraphLaunch");
  ASSERT_NE(nullptr, copy_async);
  ASSERT_NE(nullptr, create_stream);
  ASSERT_NE(nullptr, synchronize_stream);
  ASSERT_NE(nullptr, launch_callback);
  ASSERT_NE(nullptr, begin_capture);
  ASSERT_NE(nullptr, end_capture);
  ASSERT_NE(nullptr, get_nodes);
  ASSERT_NE(nullptr, instantiate);
  ASSERT_NE(nullptr, launch_graph);

  ASSERT_EQ(hipSuccess, register_host_(storage_.data(), storage_.size(),
                                       hipHostRegisterMapped));
  registered_ = true;
  ASSERT_EQ(hipSuccess,
            allocate_host_(&host_memory_, 2 * kLength, hipHostMallocMapped));
  ASSERT_EQ(hipSuccess, allocate_(&device_memory_, kLength));
  ASSERT_EQ(hipSuccess, create_stream(&copy_stream_, hipStreamNonBlocking, 0));
  // A distinct priority selects another native queue, so a queued wait for the
  // held callback cannot physically obstruct this test's transfer queue.
  ASSERT_EQ(hipSuccess,
            create_stream(&callback_stream_, hipStreamNonBlocking, -1));
  auto* input = static_cast<uint8_t*>(host_memory_);
  auto* owned_output = input + kLength;

  ASSERT_EQ(hipSuccess,
            begin_capture(copy_stream_, hipStreamCaptureModeThreadLocal));
  hipError_t result = copy_async(device_memory_, input, kLength,
                                 hipMemcpyHostToDevice, copy_stream_);
  if (result == hipSuccess) {
    result = copy_async(owned_output, device_memory_, kLength,
                        hipMemcpyDeviceToHost, copy_stream_);
  }
  if (result == hipSuccess) {
    result = copy_async(storage_.data(), device_memory_, kLength,
                        hipMemcpyDeviceToHost, copy_stream_);
  }
  const hipError_t end_result = end_capture(copy_stream_, &graph_);
  ASSERT_EQ(hipSuccess, result);
  ASSERT_EQ(hipSuccess, end_result);
  size_t node_count = 0;
  ASSERT_EQ(hipSuccess, get_nodes(graph_, nullptr, &node_count));
  EXPECT_EQ(3u, node_count);
  ASSERT_EQ(hipSuccess,
            instantiate(&graph_executable_, graph_, nullptr, nullptr, 0));

  ASSERT_EQ(hipSuccess, launch_callback(callback_stream_, HoldCallback, this));
  callback_.entered.acquire();
  for (uint8_t iteration = 0; iteration < 3; ++iteration) {
    std::memset(input, 0x12 + iteration, kLength);
    std::memset(owned_output, 0xcd, kLength);
    storage_.fill(0xcd);
    if (iteration == 0) {
      ASSERT_EQ(hipSuccess, copy_async(device_memory_, input, kLength,
                                       hipMemcpyHostToDevice, copy_stream_));
      ASSERT_EQ(hipSuccess, copy_async(owned_output, device_memory_, kLength,
                                       hipMemcpyDeviceToHost, copy_stream_));
      ASSERT_EQ(hipSuccess, copy_async(storage_.data(), device_memory_, kLength,
                                       hipMemcpyDeviceToHost, copy_stream_));
    } else {
      ASSERT_EQ(hipSuccess, launch_graph(graph_executable_, copy_stream_));
    }
    ASSERT_EQ(hipSuccess, synchronize_stream(copy_stream_));
    EXPECT_FALSE(callback_.returned.load(std::memory_order_acquire));
    EXPECT_EQ(0, std::memcmp(input, owned_output, kLength));
    EXPECT_EQ(0, std::memcmp(input, storage_.data(), kLength));
    EXPECT_EQ(0xcd, storage_.back());
  }
}

}  // namespace
