// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>

#include "api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/testing/gtest.h"

namespace {

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipMallocFn = hipError_t (*)(hipDeviceptr_t* pointer, size_t size);
using HipFreeFn = hipError_t (*)(hipDeviceptr_t pointer);
using HipHostAllocFn = hipError_t (*)(void** pointer, size_t size,
                                      unsigned int flags);
using HipFreeHostFn = hipError_t (*)(void* pointer);
using HipMallocArrayFn = hipError_t (*)(hipArray_t* array,
                                        const hipChannelFormatDesc* descriptor,
                                        size_t width, size_t height,
                                        unsigned int flags);
using HipFreeArrayFn = hipError_t (*)(hipArray_t array);
using HipMemcpyFn = hipError_t (*)(void* destination, const void* source,
                                   size_t size, hipMemcpyKind kind);
using HipMemcpy2DFromArrayAsyncSptFn = hipError_t (*)(
    void* destination, size_t destination_pitch, hipArray_const_t source,
    size_t source_x_offset, size_t source_y_offset, size_t width, size_t height,
    hipMemcpyKind kind, hipStream_t stream);
using HipMemcpy2DFromArraySptFn = hipError_t (*)(
    void* destination, size_t destination_pitch, hipArray_const_t source,
    size_t source_x_offset, size_t source_y_offset, size_t width, size_t height,
    hipMemcpyKind kind);
using HipMemcpy2DToArrayAsyncSptFn = hipError_t (*)(
    hipArray_t destination, size_t destination_x_offset,
    size_t destination_y_offset, const void* source, size_t source_pitch,
    size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream);
using HipMemcpy2DToArraySptFn = hipError_t (*)(
    hipArray_t destination, size_t destination_x_offset,
    size_t destination_y_offset, const void* source, size_t source_pitch,
    size_t width, size_t height, hipMemcpyKind kind);
using HipMemcpyFromArraySptFn = hipError_t (*)(
    void* destination, hipArray_const_t source, size_t source_x_offset,
    size_t source_y_offset, size_t count, hipMemcpyKind kind);
using HipStreamCreateFn = hipError_t (*)(hipStream_t* stream);
using HipStreamDestroyFn = hipError_t (*)(hipStream_t stream);
using HipStreamSynchronizeFn = hipError_t (*)(hipStream_t stream);
using HipLaunchHostFuncFn = hipError_t (*)(hipStream_t stream, hipHostFn_t fn,
                                           void* user_data);
using HipStreamBeginCaptureFn = hipError_t (*)(hipStream_t stream,
                                               hipStreamCaptureMode mode);
using HipStreamEndCaptureFn = hipError_t (*)(hipStream_t stream,
                                             hipGraph_t* graph);
using HipGraphGetNodesFn = hipError_t (*)(hipGraph_t graph,
                                          hipGraphNode_t* nodes,
                                          size_t* node_count);
using HipGraphDestroyFn = hipError_t (*)(hipGraph_t graph);

struct HipRuntimeApi {
  // Initializes the exact runtime under test.
  HipInitFn init = nullptr;
  // Allocates device memory used by device-to-device copies.
  HipMallocFn malloc = nullptr;
  // Releases device allocations.
  HipFreeFn free = nullptr;
  // Allocates pinned host memory used by asynchronous copies.
  HipHostAllocFn host_alloc = nullptr;
  // Releases pinned host allocations.
  HipFreeHostFn free_host = nullptr;
  // Allocates HIP arrays.
  HipMallocArrayFn malloc_array = nullptr;
  // Releases HIP arrays.
  HipFreeArrayFn free_array = nullptr;
  // Performs synchronous memory copies used for result verification.
  HipMemcpyFn memcpy = nullptr;
  // Copies pitched array contents asynchronously on an SPT stream.
  HipMemcpy2DFromArrayAsyncSptFn memcpy_2d_from_array_async_spt = nullptr;
  // Copies pitched array contents synchronously on PTDS.
  HipMemcpy2DFromArraySptFn memcpy_2d_from_array_spt = nullptr;
  // Copies pitched memory into an array asynchronously on an SPT stream.
  HipMemcpy2DToArrayAsyncSptFn memcpy_2d_to_array_async_spt = nullptr;
  // Copies pitched memory into an array synchronously on PTDS.
  HipMemcpy2DToArraySptFn memcpy_2d_to_array_spt = nullptr;
  // Copies a packed array range synchronously on PTDS.
  HipMemcpyFromArraySptFn memcpy_from_array_spt = nullptr;
  // Creates explicit streams used for stale-handle validation.
  HipStreamCreateFn stream_create = nullptr;
  // Destroys explicit streams.
  HipStreamDestroyFn stream_destroy = nullptr;
  // Waits for a selected stream timeline.
  HipStreamSynchronizeFn stream_synchronize = nullptr;
  // Enqueues gated host work used to expose ordering behavior.
  HipLaunchHostFuncFn launch_host_function = nullptr;
  // Starts capture on the selected stream.
  HipStreamBeginCaptureFn stream_begin_capture = nullptr;
  // Ends capture and returns the captured graph.
  HipStreamEndCaptureFn stream_end_capture = nullptr;
  // Queries captured graph nodes.
  HipGraphGetNodesFn graph_get_nodes = nullptr;
  // Releases captured graphs.
  HipGraphDestroyFn graph_destroy = nullptr;
};

class HipArrayCopySptApiTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    ASSERT_TRUE(dso_.Open()) << dso_.error();
#define HRX_RESOLVE_HIP_FIELD(field, symbol)               \
  api_.field = dso_.Resolve<decltype(api_.field)>(symbol); \
  ASSERT_NE(nullptr, api_.field) << dso_.error()
    HRX_RESOLVE_HIP_FIELD(init, "hipInit");
    HRX_RESOLVE_HIP_FIELD(malloc, "hipMalloc");
    HRX_RESOLVE_HIP_FIELD(free, "hipFree");
    HRX_RESOLVE_HIP_FIELD(host_alloc, "hipHostAlloc");
    HRX_RESOLVE_HIP_FIELD(free_host, "hipFreeHost");
    HRX_RESOLVE_HIP_FIELD(malloc_array, "hipMallocArray");
    HRX_RESOLVE_HIP_FIELD(free_array, "hipFreeArray");
    HRX_RESOLVE_HIP_FIELD(memcpy, "hipMemcpy");
    HRX_RESOLVE_HIP_FIELD(memcpy_2d_from_array_async_spt,
                          "hipMemcpy2DFromArrayAsync_spt");
    HRX_RESOLVE_HIP_FIELD(memcpy_2d_from_array_spt, "hipMemcpy2DFromArray_spt");
    HRX_RESOLVE_HIP_FIELD(memcpy_2d_to_array_async_spt,
                          "hipMemcpy2DToArrayAsync_spt");
    HRX_RESOLVE_HIP_FIELD(memcpy_2d_to_array_spt, "hipMemcpy2DToArray_spt");
    HRX_RESOLVE_HIP_FIELD(memcpy_from_array_spt, "hipMemcpyFromArray_spt");
    HRX_RESOLVE_HIP_FIELD(stream_create, "hipStreamCreate");
    HRX_RESOLVE_HIP_FIELD(stream_destroy, "hipStreamDestroy");
    HRX_RESOLVE_HIP_FIELD(stream_synchronize, "hipStreamSynchronize");
    HRX_RESOLVE_HIP_FIELD(launch_host_function, "hipLaunchHostFunc");
    HRX_RESOLVE_HIP_FIELD(stream_begin_capture, "hipStreamBeginCapture");
    HRX_RESOLVE_HIP_FIELD(stream_end_capture, "hipStreamEndCapture");
    HRX_RESOLVE_HIP_FIELD(graph_get_nodes, "hipGraphGetNodes");
    HRX_RESOLVE_HIP_FIELD(graph_destroy, "hipGraphDestroy");
#undef HRX_RESOLVE_HIP_FIELD
    ASSERT_EQ(hipSuccess, api_.init(/*flags=*/0));
  }

  void TearDown() override {
    if (stream_) EXPECT_EQ(hipSuccess, api_.stream_destroy(stream_));
    if (device_pointer_) EXPECT_EQ(hipSuccess, api_.free(device_pointer_));
    if (host_pointer_) EXPECT_EQ(hipSuccess, api_.free_host(host_pointer_));
    if (array_) EXPECT_EQ(hipSuccess, api_.free_array(array_));
  }

  void AllocateArray(size_t width, size_t height) {
    const hipChannelFormatDesc descriptor = {
        /*.x=*/8,
        /*.y=*/0,
        /*.z=*/0,
        /*.w=*/0,
        /*.f=*/hipChannelFormatKindUnsigned,
    };
    ASSERT_EQ(hipSuccess, api_.malloc_array(&array_, &descriptor, width, height,
                                            /*flags=*/0));
  }

  void AllocateHost(size_t size) {
    ASSERT_EQ(hipSuccess,
              api_.host_alloc(&host_pointer_, size, hipHostMallocDefault));
  }

  // Exact shared object loaded for the complete test process.
  static hrx::hip::testing::HipDso dso_;
  // Entry points resolved from |dso_|.
  static HipRuntimeApi api_;
  // Array owned by the current test case.
  hipArray_t array_ = nullptr;
  // Explicit stream owned by the current test case.
  hipStream_t stream_ = nullptr;
  // Device allocation owned by the current test case.
  hipDeviceptr_t device_pointer_ = nullptr;
  // Pinned host allocation owned by the current test case.
  void* host_pointer_ = nullptr;
};

hrx::hip::testing::HipDso HipArrayCopySptApiTest::dso_;
HipRuntimeApi HipArrayCopySptApiTest::api_;

TEST_F(HipArrayCopySptApiTest, ExportsAllArrayCopyEntryPoints) {
  EXPECT_NE(nullptr, api_.memcpy_2d_from_array_async_spt);
  EXPECT_NE(nullptr, api_.memcpy_2d_from_array_spt);
  EXPECT_NE(nullptr, api_.memcpy_2d_to_array_async_spt);
  EXPECT_NE(nullptr, api_.memcpy_2d_to_array_spt);
  EXPECT_NE(nullptr, api_.memcpy_from_array_spt);
}

TEST_F(HipArrayCopySptApiTest, ValidatesBeforeSubmittingOrAcceptingNoOps) {
  constexpr size_t kWidth = 8;
  constexpr size_t kHeight = 3;
  AllocateArray(kWidth, kHeight);
  AllocateHost(kWidth * kHeight);
  auto* host = static_cast<uint8_t*>(host_pointer_);
  const hipMemcpyKind invalid_kind = static_cast<hipMemcpyKind>(-1);

  EXPECT_EQ(hipErrorInvalidMemcpyDirection,
            api_.memcpy_2d_to_array_async_spt(nullptr, 0, 0, nullptr, 0, 0, 0,
                                              invalid_kind, nullptr));
  EXPECT_EQ(hipErrorInvalidValue, api_.memcpy_2d_to_array_async_spt(
                                      array_, 0, 0, nullptr, kWidth, kWidth, 1,
                                      hipMemcpyHostToDevice, nullptr));
  EXPECT_EQ(hipErrorInvalidPitchValue,
            api_.memcpy_2d_to_array_async_spt(array_, 0, 0, host, 0, 0, 0,
                                              hipMemcpyHostToDevice, nullptr));
  EXPECT_EQ(hipErrorInvalidHandle, api_.memcpy_2d_to_array_async_spt(
                                       nullptr, 0, 0, host, kWidth, kWidth, 1,
                                       hipMemcpyHostToDevice, nullptr));
  EXPECT_EQ(hipErrorInvalidValue, api_.memcpy_2d_to_array_async_spt(
                                      array_, kWidth, 0, host, kWidth, 1, 1,
                                      hipMemcpyHostToDevice, nullptr));
  EXPECT_EQ(hipErrorInvalidValue, api_.memcpy_2d_to_array_async_spt(
                                      array_, 0, kHeight, host, kWidth, 1, 1,
                                      hipMemcpyHostToDevice, nullptr));

  EXPECT_EQ(hipErrorInvalidMemcpyDirection,
            api_.memcpy_2d_from_array_async_spt(nullptr, 0, nullptr, 0, 0, 0, 0,
                                                invalid_kind, nullptr));
  EXPECT_EQ(hipErrorInvalidHandle, api_.memcpy_2d_from_array_async_spt(
                                       host, kWidth, nullptr, 0, 0, kWidth, 1,
                                       hipMemcpyDeviceToHost, nullptr));
  EXPECT_EQ(hipErrorInvalidValue, api_.memcpy_2d_from_array_async_spt(
                                      nullptr, kWidth, array_, 0, 0, kWidth, 1,
                                      hipMemcpyDeviceToHost, nullptr));
  EXPECT_EQ(hipErrorInvalidPitchValue,
            api_.memcpy_2d_from_array_async_spt(
                host, 0, array_, 0, 0, 0, 0, hipMemcpyDeviceToHost, nullptr));

  hipArray_t stale_array = nullptr;
  const hipChannelFormatDesc descriptor = {
      /*.x=*/8,
      /*.y=*/0,
      /*.z=*/0,
      /*.w=*/0,
      /*.f=*/hipChannelFormatKindUnsigned,
  };
  ASSERT_EQ(hipSuccess, api_.malloc_array(&stale_array, &descriptor, kWidth,
                                          kHeight, /*flags=*/0));
  ASSERT_EQ(hipSuccess, api_.free_array(stale_array));
  EXPECT_EQ(hipErrorInvalidValue, api_.memcpy_2d_from_array_async_spt(
                                      host, kWidth, stale_array, 0, 0, kWidth,
                                      1, hipMemcpyDeviceToHost, nullptr));

  ASSERT_EQ(hipSuccess, api_.stream_create(&stream_));
  const hipStream_t stale_stream = stream_;
  ASSERT_EQ(hipSuccess, api_.stream_destroy(stream_));
  stream_ = nullptr;
  EXPECT_EQ(
      hipErrorInvalidResourceHandle,
      api_.memcpy_2d_to_array_async_spt(array_, 0, 0, host, kWidth, kWidth, 1,
                                        hipMemcpyHostToDevice, stale_stream));
}

TEST_F(HipArrayCopySptApiTest, CopiesPitchedRegionsThroughAllEntryPoints) {
  constexpr size_t kArrayWidth = 11;
  constexpr size_t kArrayHeight = 5;
  constexpr size_t kPitch = 16;
  constexpr size_t kCopyWidth = 7;
  constexpr size_t kCopyHeight = 3;
  constexpr size_t kXOffset = 2;
  constexpr size_t kYOffset = 1;
  AllocateArray(kArrayWidth, kArrayHeight);
  AllocateHost(2 * kPitch * kCopyHeight);
  auto* host = static_cast<uint8_t*>(host_pointer_);
  uint8_t* source = host;
  uint8_t* destination = host + kPitch * kCopyHeight;
  std::memset(host, 0, 2 * kPitch * kCopyHeight);
  ASSERT_EQ(hipSuccess, api_.memcpy_2d_to_array_spt(
                            array_, 0, 0, host, kArrayWidth, kArrayWidth,
                            kArrayHeight, hipMemcpyHostToDevice));
  for (size_t row = 0; row < kCopyHeight; ++row) {
    for (size_t column = 0; column < kCopyWidth; ++column) {
      source[row * kPitch + column] =
          static_cast<uint8_t>(17 + row * kCopyWidth + column);
    }
  }

  ASSERT_EQ(hipSuccess,
            api_.memcpy_2d_to_array_async_spt(
                array_, kXOffset, kYOffset, source, kPitch, kCopyWidth,
                kCopyHeight, hipMemcpyHostToDevice, nullptr));
  ASSERT_EQ(hipSuccess,
            api_.memcpy_2d_from_array_async_spt(
                destination, kPitch, array_, kXOffset, kYOffset, kCopyWidth,
                kCopyHeight, hipMemcpyDeviceToHost, hipStreamLegacy));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(hipStreamPerThread));
  for (size_t row = 0; row < kCopyHeight; ++row) {
    EXPECT_EQ(0, std::memcmp(source + row * kPitch, destination + row * kPitch,
                             kCopyWidth));
  }

  ASSERT_EQ(hipSuccess, api_.stream_create(&stream_));
  for (size_t row = 0; row < kCopyHeight; ++row) {
    for (size_t column = 0; column < kCopyWidth; ++column) {
      source[row * kPitch + column] ^= 0xA5;
      destination[row * kPitch + column] = 0;
    }
  }
  ASSERT_EQ(hipSuccess,
            api_.memcpy_2d_to_array_async_spt(
                array_, kXOffset, kYOffset, source, kPitch, kCopyWidth,
                kCopyHeight, hipMemcpyHostToDevice, stream_));
  ASSERT_EQ(hipSuccess,
            api_.memcpy_2d_from_array_async_spt(
                destination, kPitch, array_, kXOffset, kYOffset, kCopyWidth,
                kCopyHeight, hipMemcpyDeviceToHost, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
  for (size_t row = 0; row < kCopyHeight; ++row) {
    EXPECT_EQ(0, std::memcmp(source + row * kPitch, destination + row * kPitch,
                             kCopyWidth));
  }

  for (size_t row = 0; row < kCopyHeight; ++row) {
    for (size_t column = 0; column < kCopyWidth; ++column) {
      source[row * kPitch + column] ^= 0x5A;
      destination[row * kPitch + column] = 0;
    }
  }
  ASSERT_EQ(hipSuccess, api_.memcpy_2d_to_array_spt(
                            array_, kXOffset, kYOffset, source, kPitch,
                            kCopyWidth, kCopyHeight, hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, api_.memcpy_2d_from_array_spt(
                            destination, kPitch, array_, kXOffset, kYOffset,
                            kCopyWidth, kCopyHeight, hipMemcpyDeviceToHost));
  for (size_t row = 0; row < kCopyHeight; ++row) {
    EXPECT_EQ(0, std::memcmp(source + row * kPitch, destination + row * kPitch,
                             kCopyWidth));
  }

  constexpr size_t kPackedCount = 2 * kArrayWidth + 3;
  std::array<uint8_t, kPackedCount> packed = {};
  ASSERT_EQ(hipSuccess, api_.memcpy_from_array_spt(
                            packed.data(), array_, kXOffset, kYOffset,
                            packed.size(), hipMemcpyDeviceToHost));
  std::array<uint8_t, kArrayWidth * kArrayHeight> array_contents = {};
  for (size_t row = 0; row < kCopyHeight; ++row) {
    std::memcpy(
        array_contents.data() + (kYOffset + row) * kArrayWidth + kXOffset,
        source + row * kPitch, kCopyWidth);
  }
  std::array<uint8_t, kPackedCount> expected_packed = {};
  std::memcpy(expected_packed.data(),
              array_contents.data() + kYOffset * kArrayWidth + kXOffset,
              expected_packed.size());
  EXPECT_EQ(expected_packed, packed);
}

TEST_F(HipArrayCopySptApiTest, DefaultKindDistinguishesPinnedHostMemory) {
  constexpr size_t kWidth = 8;
  AllocateArray(kWidth, 1);
  AllocateHost(kWidth);
  auto* host = static_cast<uint8_t*>(host_pointer_);
  for (size_t i = 0; i < kWidth; ++i) {
    host[i] = static_cast<uint8_t>(i + 1);
  }

  ASSERT_EQ(hipSuccess,
            api_.memcpy_2d_to_array_spt(array_, 0, 0, host, kWidth, kWidth, 1,
                                        hipMemcpyDefault));
  std::memset(host, 0, kWidth);
  ASSERT_EQ(hipSuccess, api_.memcpy_from_array_spt(host, array_, 0, 0, kWidth,
                                                   hipMemcpyDefault));
  for (size_t i = 0; i < kWidth; ++i) {
    EXPECT_EQ(static_cast<uint8_t>(i + 1), host[i]);
  }

  ASSERT_EQ(hipSuccess, api_.malloc(&device_pointer_, kWidth));
  EXPECT_EQ(hipErrorInvalidMemcpyDirection,
            api_.memcpy_from_array_spt(device_pointer_, array_, 0, 0, kWidth,
                                       hipMemcpyDeviceToHost));
}

TEST_F(HipArrayCopySptApiTest, LegacySentinelCapturesOnPerThreadStream) {
  constexpr size_t kWidth = 8;
  AllocateArray(kWidth, 1);
  AllocateHost(kWidth);
  auto* source = static_cast<uint8_t*>(host_pointer_);

  ASSERT_EQ(hipSuccess,
            api_.stream_begin_capture(hipStreamPerThread,
                                      hipStreamCaptureModeThreadLocal));
  const hipError_t copy_result =
      api_.memcpy_2d_to_array_async_spt(array_, 0, 0, source, kWidth, kWidth, 1,
                                        hipMemcpyHostToDevice, hipStreamLegacy);
  hipGraph_t graph = nullptr;
  const hipError_t end_result =
      api_.stream_end_capture(hipStreamPerThread, &graph);
  EXPECT_EQ(hipSuccess, copy_result);
  ASSERT_EQ(hipSuccess, end_result);
  ASSERT_NE(nullptr, graph);
  size_t node_count = 0;
  EXPECT_EQ(hipSuccess, api_.graph_get_nodes(graph, nullptr, &node_count));
  EXPECT_EQ(1u, node_count);
  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
}

struct FillGate {
  // Set after the callback begins executing.
  std::atomic<bool> entered = false;
  // Set by the test to let the callback produce its data.
  std::atomic<bool> release = false;
  // Pitched host destination populated by the callback.
  uint8_t* destination = nullptr;
  // Destination row pitch in bytes.
  size_t pitch = 0;
  // Number of bytes populated in each row.
  size_t width = 0;
  // Number of rows populated.
  size_t height = 0;
};

void FillAfterRelease(void* user_data) {
  auto* gate = static_cast<FillGate*>(user_data);
  gate->entered.store(true, std::memory_order_release);
  while (!gate->release.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  for (size_t row = 0; row < gate->height; ++row) {
    for (size_t column = 0; column < gate->width; ++column) {
      gate->destination[row * gate->pitch + column] =
          static_cast<uint8_t>(31 + row * gate->width + column);
    }
  }
}

TEST_F(HipArrayCopySptApiTest, StridedCopyWaitsForLegacyProducer) {
  constexpr size_t kWidth = 7;
  constexpr size_t kHeight = 3;
  constexpr size_t kPitch = 13;
  AllocateArray(kWidth + 2, kHeight + 1);
  AllocateHost(2 * kPitch * kHeight);
  auto* host = static_cast<uint8_t*>(host_pointer_);
  uint8_t* source = host;
  uint8_t* destination = host + kPitch * kHeight;
  FillGate gate = {/*.entered=*/false,
                   /*.release=*/false,
                   /*.destination=*/source,
                   /*.pitch=*/kPitch,
                   /*.width=*/kWidth,
                   /*.height=*/kHeight};
  ASSERT_EQ(hipSuccess, api_.launch_host_function(hipStreamLegacy,
                                                  FillAfterRelease, &gate));
  while (!gate.entered.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  const hipError_t copy_result = api_.memcpy_2d_to_array_async_spt(
      array_, /*destination_x_offset=*/1, /*destination_y_offset=*/1, source,
      kPitch, kWidth, kHeight, hipMemcpyHostToDevice, hipStreamLegacy);
  gate.release.store(true, std::memory_order_release);
  ASSERT_EQ(hipSuccess, copy_result);
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(hipStreamPerThread));
  ASSERT_EQ(hipSuccess,
            api_.memcpy_2d_from_array_spt(
                destination, kPitch, array_, /*source_x_offset=*/1,
                /*source_y_offset=*/1, kWidth, kHeight, hipMemcpyDeviceToHost));
  for (size_t row = 0; row < kHeight; ++row) {
    EXPECT_EQ(0, std::memcmp(source + row * kPitch, destination + row * kPitch,
                             kWidth));
  }
}

TEST_F(HipArrayCopySptApiTest, ContiguousCopyWaitsForLegacyProducer) {
  constexpr size_t kWidth = 9;
  constexpr size_t kHeight = 4;
  AllocateArray(kWidth, kHeight);
  AllocateHost(2 * kWidth * kHeight);
  auto* host = static_cast<uint8_t*>(host_pointer_);
  uint8_t* source = host;
  uint8_t* destination = host + kWidth * kHeight;
  FillGate gate = {/*.entered=*/false,
                   /*.release=*/false,
                   /*.destination=*/source,
                   /*.pitch=*/kWidth,
                   /*.width=*/kWidth,
                   /*.height=*/kHeight};
  ASSERT_EQ(hipSuccess, api_.launch_host_function(hipStreamLegacy,
                                                  FillAfterRelease, &gate));
  while (!gate.entered.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  const hipError_t copy_result = api_.memcpy_2d_to_array_async_spt(
      array_, 0, 0, source, kWidth, kWidth, kHeight, hipMemcpyHostToDevice,
      hipStreamLegacy);
  gate.release.store(true, std::memory_order_release);
  ASSERT_EQ(hipSuccess, copy_result);
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(hipStreamPerThread));
  ASSERT_EQ(hipSuccess, api_.memcpy_2d_from_array_spt(
                            destination, kWidth, array_, 0, 0, kWidth, kHeight,
                            hipMemcpyDeviceToHost));
  EXPECT_EQ(0, std::memcmp(source, destination, kWidth * kHeight));
}

struct WaitGate {
  // Set after the callback begins executing.
  std::atomic<bool> entered = false;
  // Set by the test to let the callback return.
  std::atomic<bool> release = false;
};

void WaitForRelease(void* user_data) {
  auto* gate = static_cast<WaitGate*>(user_data);
  gate->entered.store(true, std::memory_order_release);
  while (!gate->release.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

TEST_F(HipArrayCopySptApiTest, NoOpAndDeviceCopyDoNotDrainPerThreadStream) {
  constexpr size_t kWidth = 8;
  AllocateArray(kWidth, 1);
  AllocateHost(kWidth);
  auto* host = static_cast<uint8_t*>(host_pointer_);
  for (size_t i = 0; i < kWidth; ++i) {
    host[i] = static_cast<uint8_t>(i + 1);
  }
  ASSERT_EQ(hipSuccess,
            api_.memcpy_2d_to_array_spt(array_, 0, 0, host, kWidth, kWidth, 1,
                                        hipMemcpyHostToDevice));

  WaitGate no_op_gate;
  ASSERT_EQ(hipSuccess, api_.launch_host_function(hipStreamPerThread,
                                                  WaitForRelease, &no_op_gate));
  while (!no_op_gate.entered.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  EXPECT_EQ(hipSuccess, api_.memcpy_2d_from_array_spt(
                            host, /*destination_pitch=*/1, array_, 0, 0,
                            /*width=*/0, /*height=*/1, hipMemcpyDeviceToHost));
  no_op_gate.release.store(true, std::memory_order_release);
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(hipStreamPerThread));

  ASSERT_EQ(hipSuccess, api_.malloc(&device_pointer_, kWidth));
  WaitGate device_gate;
  ASSERT_EQ(hipSuccess, api_.launch_host_function(
                            hipStreamPerThread, WaitForRelease, &device_gate));
  while (!device_gate.entered.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  EXPECT_EQ(hipSuccess,
            api_.memcpy_2d_from_array_spt(device_pointer_, kWidth, array_, 0, 0,
                                          kWidth, 1, hipMemcpyDeviceToDevice));
  device_gate.release.store(true, std::memory_order_release);
  EXPECT_EQ(hipSuccess, api_.stream_synchronize(hipStreamPerThread));
  std::memset(host, 0, kWidth);
  ASSERT_EQ(hipSuccess,
            api_.memcpy(host, device_pointer_, kWidth, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < kWidth; ++i) {
    EXPECT_EQ(static_cast<uint8_t>(i + 1), host[i]);
  }
}

TEST_F(HipArrayCopySptApiTest, SynchronousCopyRejectsCaptureBeforeMutation) {
  constexpr size_t kWidth = 8;
  AllocateArray(kWidth, 1);
  AllocateHost(kWidth);
  auto* source = static_cast<uint8_t*>(host_pointer_);
  ASSERT_EQ(hipSuccess,
            api_.stream_begin_capture(hipStreamPerThread,
                                      hipStreamCaptureModeThreadLocal));
  EXPECT_EQ(hipErrorStreamCaptureImplicit,
            api_.memcpy_2d_to_array_spt(array_, 0, 0, source, kWidth, kWidth, 1,
                                        hipMemcpyHostToDevice));
  hipGraph_t graph = nullptr;
  EXPECT_EQ(hipErrorStreamCaptureInvalidated,
            api_.stream_end_capture(hipStreamPerThread, &graph));
  EXPECT_EQ(nullptr, graph);

  ASSERT_EQ(hipSuccess, api_.stream_create(&stream_));
  ASSERT_EQ(hipSuccess, api_.stream_begin_capture(
                            stream_, hipStreamCaptureModeThreadLocal));
  EXPECT_EQ(hipErrorStreamCaptureImplicit,
            api_.memcpy_2d_to_array_spt(array_, 0, 0, source, kWidth, kWidth, 1,
                                        hipMemcpyHostToDevice));
  EXPECT_EQ(hipErrorStreamCaptureInvalidated,
            api_.stream_end_capture(stream_, &graph));
  EXPECT_EQ(nullptr, graph);
}

TEST_F(HipArrayCopySptApiTest, ConcurrentCopyAndFreeRetainArrayStorage) {
  constexpr size_t kIterations = 64;
  constexpr size_t kWidth = 32;
  AllocateHost(kWidth);
  auto* source = static_cast<uint8_t*>(host_pointer_);
  for (size_t iteration = 0; iteration < kIterations; ++iteration) {
    hipArray_t array = nullptr;
    const hipChannelFormatDesc descriptor = {
        /*.x=*/8,
        /*.y=*/0,
        /*.z=*/0,
        /*.w=*/0,
        /*.f=*/hipChannelFormatKindUnsigned,
    };
    ASSERT_EQ(hipSuccess, api_.malloc_array(&array, &descriptor, kWidth, 1,
                                            /*flags=*/0));
    std::atomic<bool> start = false;
    hipError_t copy_result = hipErrorUnknown;
    hipError_t free_result = hipErrorUnknown;
    std::thread copy_thread([&] {
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      if (iteration % 2 == 0) {
        copy_result = api_.memcpy_2d_to_array_async_spt(
            array, 0, 0, source, kWidth, kWidth, 1, hipMemcpyHostToDevice,
            nullptr);
      } else {
        copy_result = api_.memcpy_2d_from_array_async_spt(
            source, kWidth, array, 0, 0, kWidth, 1, hipMemcpyDeviceToHost,
            nullptr);
      }
    });
    std::thread free_thread([&] {
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      free_result = api_.free_array(array);
    });
    start.store(true, std::memory_order_release);
    copy_thread.join();
    free_thread.join();
    EXPECT_EQ(hipSuccess, free_result);
    EXPECT_TRUE(copy_result == hipSuccess ||
                copy_result == hipErrorInvalidValue)
        << "unexpected copy result " << copy_result;
  }
  EXPECT_EQ(hipSuccess, api_.stream_synchronize(hipStreamPerThread));
}

}  // namespace
