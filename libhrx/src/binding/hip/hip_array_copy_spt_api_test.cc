// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/testing/gtest.h"

namespace {

#if defined(HRX_TEST_LIBAMDHIP64_PATH)
constexpr const char* kBuildHipDsoFallbackPath = HRX_TEST_LIBAMDHIP64_PATH;
#else
constexpr const char* kBuildHipDsoFallbackPath = nullptr;
#endif

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipHrxSetArrayLeaseObserverForTestingFn =
    hipError_t (*)(hipHostFn_t observer, void* user_data);
using HipGetDeviceCountFn = hipError_t (*)(int* count);
using HipSetDeviceFn = hipError_t (*)(int device_id);
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
using HipMemcpyAsyncFn = hipError_t (*)(void* destination, const void* source,
                                        size_t size, hipMemcpyKind kind,
                                        hipStream_t stream);
using HipMemcpyHtoAAsyncFn = hipError_t (*)(hipArray_t destination,
                                            size_t destination_offset,
                                            const void* source, size_t size,
                                            hipStream_t stream);
using HipMemcpyHtoAFn = hipError_t (*)(hipArray_t destination,
                                       size_t destination_offset,
                                       const void* source, size_t size);
using HipMemcpyAtoHAsyncFn = hipError_t (*)(void* destination,
                                            hipArray_t source,
                                            size_t source_offset, size_t size,
                                            hipStream_t stream);
using HipMemcpyAtoHFn = hipError_t (*)(void* destination, hipArray_t source,
                                       size_t source_offset, size_t size);
using HipMemcpyDtoAFn = hipError_t (*)(hipArray_t destination,
                                       size_t destination_offset,
                                       hipDeviceptr_t source, size_t size);
using HipMemcpyAtoDFn = hipError_t (*)(hipDeviceptr_t destination,
                                       hipArray_t source, size_t source_offset,
                                       size_t size);
using HipMemcpyAtoAFn = hipError_t (*)(hipArray_t destination,
                                       size_t destination_offset,
                                       hipArray_t source, size_t source_offset,
                                       size_t size);
using HipMemcpy3DAsyncFn = hipError_t (*)(const hipMemcpy3DParms* parameters,
                                          hipStream_t stream);
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
using HipMemcpyToArrayFn = hipError_t (*)(hipArray_t destination,
                                          size_t destination_x_offset,
                                          size_t destination_y_offset,
                                          const void* source, size_t count,
                                          hipMemcpyKind kind);
using HipStreamCreateFn = hipError_t (*)(hipStream_t* stream);
using HipStreamCreateWithPriorityFn = hipError_t (*)(hipStream_t* stream,
                                                     unsigned int flags,
                                                     int priority);
using HipStreamDestroyFn = hipError_t (*)(hipStream_t stream);
using HipStreamSynchronizeFn = hipError_t (*)(hipStream_t stream);
using HipEventCreateWithFlagsFn = hipError_t (*)(hipEvent_t* event,
                                                 unsigned int flags);
using HipEventRecordFn = hipError_t (*)(hipEvent_t event, hipStream_t stream);
using HipStreamWaitEventFn = hipError_t (*)(hipStream_t stream,
                                            hipEvent_t event,
                                            unsigned int flags);
using HipEventDestroyFn = hipError_t (*)(hipEvent_t event);
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
  // Installs the deterministic production array-lease test observer.
  HipHrxSetArrayLeaseObserverForTestingFn set_array_lease_observer = nullptr;
  // Initializes the exact runtime under test.
  HipInitFn init = nullptr;
  // Queries devices available to the exact runtime under test.
  HipGetDeviceCountFn get_device_count = nullptr;
  // Selects the current device for the calling thread.
  HipSetDeviceFn set_device = nullptr;
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
  // Enqueues ordinary memory copies used to produce stream-ordered input.
  HipMemcpyAsyncFn memcpy_async = nullptr;
  // Enqueues a legacy host-to-array copy.
  HipMemcpyHtoAAsyncFn memcpy_htoa_async = nullptr;
  // Performs a synchronous legacy host-to-array copy.
  HipMemcpyHtoAFn memcpy_htoa = nullptr;
  // Enqueues a legacy array-to-host copy.
  HipMemcpyAtoHAsyncFn memcpy_atoh_async = nullptr;
  // Performs a synchronous legacy array-to-host copy.
  HipMemcpyAtoHFn memcpy_atoh = nullptr;
  // Performs a synchronous legacy device-to-array copy.
  HipMemcpyDtoAFn memcpy_dtoa = nullptr;
  // Performs a synchronous legacy array-to-device copy.
  HipMemcpyAtoDFn memcpy_atod = nullptr;
  // Performs a synchronous legacy array-to-array copy.
  HipMemcpyAtoAFn memcpy_atoa = nullptr;
  // Performs generic asynchronous 3D memory copies.
  HipMemcpy3DAsyncFn memcpy_3d_async = nullptr;
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
  // Copies a packed range synchronously into an array.
  HipMemcpyToArrayFn memcpy_to_array = nullptr;
  // Creates explicit streams used for stale-handle validation.
  HipStreamCreateFn stream_create = nullptr;
  // Creates explicit streams on a selected hardware-priority queue.
  HipStreamCreateWithPriorityFn stream_create_with_priority = nullptr;
  // Destroys explicit streams.
  HipStreamDestroyFn stream_destroy = nullptr;
  // Waits for a selected stream timeline.
  HipStreamSynchronizeFn stream_synchronize = nullptr;
  // Creates capture-only events used to join participant streams.
  HipEventCreateWithFlagsFn event_create_with_flags = nullptr;
  // Records an event frontier on a stream.
  HipEventRecordFn event_record = nullptr;
  // Makes a stream join a captured event frontier.
  HipStreamWaitEventFn stream_wait_event = nullptr;
  // Releases an event.
  HipEventDestroyFn event_destroy = nullptr;
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
    ASSERT_TRUE(dso_.Open(kBuildHipDsoFallbackPath)) << dso_.error();
#define HRX_RESOLVE_HIP_FIELD(field, symbol)               \
  api_.field = dso_.Resolve<decltype(api_.field)>(symbol); \
  ASSERT_NE(nullptr, api_.field) << dso_.error()
    HRX_RESOLVE_HIP_FIELD(init, "hipInit");
    HRX_RESOLVE_HIP_FIELD(set_array_lease_observer,
                          "hipHRXSetArrayLeaseObserverForTesting");
    HRX_RESOLVE_HIP_FIELD(get_device_count, "hipGetDeviceCount");
    HRX_RESOLVE_HIP_FIELD(set_device, "hipSetDevice");
    HRX_RESOLVE_HIP_FIELD(malloc, "hipMalloc");
    HRX_RESOLVE_HIP_FIELD(free, "hipFree");
    HRX_RESOLVE_HIP_FIELD(host_alloc, "hipHostAlloc");
    HRX_RESOLVE_HIP_FIELD(free_host, "hipFreeHost");
    HRX_RESOLVE_HIP_FIELD(malloc_array, "hipMallocArray");
    HRX_RESOLVE_HIP_FIELD(free_array, "hipFreeArray");
    HRX_RESOLVE_HIP_FIELD(memcpy, "hipMemcpy");
    HRX_RESOLVE_HIP_FIELD(memcpy_async, "hipMemcpyAsync");
    HRX_RESOLVE_HIP_FIELD(memcpy_htoa_async, "hipMemcpyHtoAAsync");
    HRX_RESOLVE_HIP_FIELD(memcpy_htoa, "hipMemcpyHtoA");
    HRX_RESOLVE_HIP_FIELD(memcpy_atoh_async, "hipMemcpyAtoHAsync");
    HRX_RESOLVE_HIP_FIELD(memcpy_atoh, "hipMemcpyAtoH");
    HRX_RESOLVE_HIP_FIELD(memcpy_dtoa, "hipMemcpyDtoA");
    HRX_RESOLVE_HIP_FIELD(memcpy_atod, "hipMemcpyAtoD");
    HRX_RESOLVE_HIP_FIELD(memcpy_atoa, "hipMemcpyAtoA");
    HRX_RESOLVE_HIP_FIELD(memcpy_3d_async, "hipMemcpy3DAsync");
    HRX_RESOLVE_HIP_FIELD(memcpy_2d_from_array_async_spt,
                          "hipMemcpy2DFromArrayAsync_spt");
    HRX_RESOLVE_HIP_FIELD(memcpy_2d_from_array_spt, "hipMemcpy2DFromArray_spt");
    HRX_RESOLVE_HIP_FIELD(memcpy_2d_to_array_async_spt,
                          "hipMemcpy2DToArrayAsync_spt");
    HRX_RESOLVE_HIP_FIELD(memcpy_2d_to_array_spt, "hipMemcpy2DToArray_spt");
    HRX_RESOLVE_HIP_FIELD(memcpy_from_array_spt, "hipMemcpyFromArray_spt");
    HRX_RESOLVE_HIP_FIELD(memcpy_to_array, "hipMemcpyToArray");
    HRX_RESOLVE_HIP_FIELD(stream_create, "hipStreamCreate");
    HRX_RESOLVE_HIP_FIELD(stream_create_with_priority,
                          "hipStreamCreateWithPriority");
    HRX_RESOLVE_HIP_FIELD(stream_destroy, "hipStreamDestroy");
    HRX_RESOLVE_HIP_FIELD(stream_synchronize, "hipStreamSynchronize");
    HRX_RESOLVE_HIP_FIELD(event_create_with_flags, "hipEventCreateWithFlags");
    HRX_RESOLVE_HIP_FIELD(event_record, "hipEventRecord");
    HRX_RESOLVE_HIP_FIELD(stream_wait_event, "hipStreamWaitEvent");
    HRX_RESOLVE_HIP_FIELD(event_destroy, "hipEventDestroy");
    HRX_RESOLVE_HIP_FIELD(launch_host_function, "hipLaunchHostFunc");
    HRX_RESOLVE_HIP_FIELD(stream_begin_capture, "hipStreamBeginCapture");
    HRX_RESOLVE_HIP_FIELD(stream_end_capture, "hipStreamEndCapture");
    HRX_RESOLVE_HIP_FIELD(graph_get_nodes, "hipGraphGetNodes");
    HRX_RESOLVE_HIP_FIELD(graph_destroy, "hipGraphDestroy");
#undef HRX_RESOLVE_HIP_FIELD
    ASSERT_EQ(hipSuccess, api_.init(/*flags=*/0));
  }

  void TearDown() override {
    if (stream_) {
      EXPECT_EQ(hipSuccess, api_.stream_destroy(stream_));
    }
    if (device_pointer_) {
      EXPECT_EQ(hipSuccess, api_.free(device_pointer_));
    }
    if (host_pointer_) {
      EXPECT_EQ(hipSuccess, api_.free_host(host_pointer_));
    }
    if (array_) {
      EXPECT_EQ(hipSuccess, api_.free_array(array_));
    }
  }

  void AllocateArray(size_t width, size_t height) {
    const hipChannelFormatDesc descriptor = {
        .x = 8,
        .y = 0,
        .z = 0,
        .w = 0,
        .f = hipChannelFormatKindUnsigned,
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
  EXPECT_NE(nullptr, api_.memcpy_htoa_async);
  EXPECT_NE(nullptr, api_.memcpy_htoa);
  EXPECT_NE(nullptr, api_.memcpy_atoh_async);
  EXPECT_NE(nullptr, api_.memcpy_atoh);
  EXPECT_NE(nullptr, api_.memcpy_dtoa);
  EXPECT_NE(nullptr, api_.memcpy_atod);
  EXPECT_NE(nullptr, api_.memcpy_atoa);
  EXPECT_NE(nullptr, api_.memcpy_2d_from_array_async_spt);
  EXPECT_NE(nullptr, api_.memcpy_2d_from_array_spt);
  EXPECT_NE(nullptr, api_.memcpy_2d_to_array_async_spt);
  EXPECT_NE(nullptr, api_.memcpy_2d_to_array_spt);
  EXPECT_NE(nullptr, api_.memcpy_from_array_spt);
  EXPECT_NE(nullptr, api_.memcpy_to_array);
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
      .x = 8,
      .y = 0,
      .z = 0,
      .w = 0,
      .f = hipChannelFormatKindUnsigned,
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
  constexpr size_t kArrayElementCount = kArrayWidth * kArrayHeight;
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
  std::array<uint8_t, kArrayElementCount> array_contents = {};
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

TEST_F(HipArrayCopySptApiTest,
       CapturedZeroExtentPrecedesOrdinaryOperandValidation) {
  ASSERT_EQ(hipSuccess,
            api_.stream_begin_capture(hipStreamPerThread,
                                      hipStreamCaptureModeThreadLocal));
  const hipMemcpyKind invalid_kind = static_cast<hipMemcpyKind>(-1);
  EXPECT_EQ(hipSuccess,
            api_.memcpy_2d_to_array_async_spt(nullptr, 0, 0, nullptr, 0, 0, 1,
                                              invalid_kind, nullptr));
  EXPECT_EQ(hipSuccess,
            api_.memcpy_2d_from_array_async_spt(nullptr, 0, nullptr, 0, 0, 1, 0,
                                                invalid_kind, nullptr));

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.stream_end_capture(hipStreamPerThread, &graph));
  ASSERT_NE(nullptr, graph);
  size_t node_count = 0;
  ASSERT_EQ(hipSuccess, api_.graph_get_nodes(graph, nullptr, &node_count));
  EXPECT_EQ(0u, node_count);
  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
}

TEST_F(HipArrayCopySptApiTest, JoinedParticipantCapturesH2DAndD2HArrayCopies) {
  constexpr size_t kWidth = 8;
  AllocateArray(kWidth, 1);
  AllocateHost(kWidth);
  auto* host = static_cast<uint8_t*>(host_pointer_);
  ASSERT_EQ(hipSuccess, api_.stream_create(&stream_));
  hipStream_t participant = nullptr;
  ASSERT_EQ(hipSuccess, api_.stream_create(&participant));

  for (int direction = 0; direction < 2; ++direction) {
    hipEvent_t join_event = nullptr;
    hipEvent_t return_event = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.event_create_with_flags(&join_event, hipEventDisableTiming));
    ASSERT_EQ(hipSuccess, api_.event_create_with_flags(&return_event,
                                                       hipEventDisableTiming));
    ASSERT_EQ(hipSuccess,
              api_.stream_begin_capture(stream_, hipStreamCaptureModeRelaxed));
    ASSERT_EQ(hipSuccess, api_.event_record(join_event, stream_));
    ASSERT_EQ(hipSuccess,
              api_.stream_wait_event(participant, join_event, /*flags=*/0));

    const hipError_t copy_result =
        direction == 0 ? api_.memcpy_2d_to_array_async_spt(
                             array_, 0, 0, host, kWidth, kWidth, 1,
                             hipMemcpyHostToDevice, participant)
                       : api_.memcpy_2d_from_array_async_spt(
                             host, kWidth, array_, 0, 0, kWidth, 1,
                             hipMemcpyDeviceToHost, participant);
    ASSERT_EQ(hipSuccess, copy_result) << "direction " << direction;
    ASSERT_EQ(hipSuccess, api_.event_record(return_event, participant));
    ASSERT_EQ(hipSuccess,
              api_.stream_wait_event(stream_, return_event, /*flags=*/0));

    hipGraph_t graph = nullptr;
    ASSERT_EQ(hipSuccess, api_.stream_end_capture(stream_, &graph));
    ASSERT_NE(nullptr, graph);
    size_t node_count = 0;
    ASSERT_EQ(hipSuccess, api_.graph_get_nodes(graph, nullptr, &node_count));
    EXPECT_EQ(1u, node_count) << "direction " << direction;
    EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
    EXPECT_EQ(hipSuccess, api_.event_destroy(return_event));
    EXPECT_EQ(hipSuccess, api_.event_destroy(join_event));
  }

  EXPECT_EQ(hipSuccess, api_.stream_destroy(participant));
}

TEST_F(HipArrayCopySptApiTest, RelaxedCaptureEndSerializesWithAsyncCopies) {
  constexpr size_t kIterations = 64;
  constexpr size_t kWidth = 8;
  AllocateArray(kWidth, 1);
  AllocateHost(kWidth);
  auto* host = static_cast<uint8_t*>(host_pointer_);
  ASSERT_EQ(hipSuccess,
            api_.memcpy_2d_to_array_spt(array_, 0, 0, host, kWidth, kWidth, 1,
                                        hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, api_.stream_create(&stream_));

  for (size_t iteration = 0; iteration < kIterations; ++iteration) {
    ASSERT_EQ(hipSuccess,
              api_.stream_begin_capture(stream_, hipStreamCaptureModeRelaxed));
    std::atomic<bool> start = false;
    hipError_t copy_result = hipErrorUnknown;
    hipError_t end_result = hipErrorUnknown;
    hipGraph_t graph = nullptr;
    std::thread copy_thread([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      if (iteration % 2 == 0) {
        copy_result = api_.memcpy_2d_to_array_async_spt(
            array_, 0, 0, host, kWidth, kWidth, 1, hipMemcpyHostToDevice,
            stream_);
      } else {
        copy_result = api_.memcpy_2d_from_array_async_spt(
            host, kWidth, array_, 0, 0, kWidth, 1, hipMemcpyDeviceToHost,
            stream_);
      }
    });
    std::thread end_thread([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      end_result = api_.stream_end_capture(stream_, &graph);
    });
    start.store(true, std::memory_order_release);
    copy_thread.join();
    end_thread.join();

    ASSERT_EQ(hipSuccess, copy_result) << "iteration " << iteration;
    ASSERT_EQ(hipSuccess, end_result) << "iteration " << iteration;
    ASSERT_NE(nullptr, graph) << "iteration " << iteration;
    size_t node_count = 0;
    ASSERT_EQ(hipSuccess, api_.graph_get_nodes(graph, nullptr, &node_count));
    EXPECT_LE(node_count, 1u) << "iteration " << iteration;
    EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
  }
}

class TestNotification {
 public:
  void Post() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      posted_.store(true, std::memory_order_release);
    }
    condition_.notify_all();
  }

  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock,
                    [&] { return posted_.load(std::memory_order_acquire); });
  }

  bool IsPosted() const { return posted_.load(std::memory_order_acquire); }

 private:
  // True after the notification has been posted.
  std::atomic<bool> posted_ = false;
  // Serializes condition-variable waits.
  std::mutex mutex_;
  // Wakes threads waiting for the notification.
  std::condition_variable condition_;
};

struct WaitGate {
  void EnterAndWait() {
    entered.Post();
    release.Wait();
  }

  // Posted after the callback begins executing.
  TestNotification entered;
  // Posted by the test to let the callback return.
  TestNotification release;
};

struct FillGate : public WaitGate {
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
  gate->EnterAndWait();
  for (size_t row = 0; row < gate->height; ++row) {
    for (size_t column = 0; column < gate->width; ++column) {
      gate->destination[row * gate->pitch + column] =
          static_cast<uint8_t>(31 + row * gate->width + column);
    }
  }
}

void WaitForRelease(void* user_data) {
  auto* gate = static_cast<WaitGate*>(user_data);
  gate->EnterAndWait();
}

struct CountedWaitGate {
  // Observer invocation that represents the post-lookup copy boundary.
  int target_count = 0;
  // Total observer invocations reached.
  std::atomic<int> count = 0;
  // Posted when |target_count| is reached.
  TestNotification entered;
  // Posted by the test to let the target invocation return.
  TestNotification release;
};

void WaitForTargetCount(void* user_data) {
  auto* gate = static_cast<CountedWaitGate*>(user_data);
  const int count = gate->count.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (count != gate->target_count) {
    return;
  }
  gate->entered.Post();
  gate->release.Wait();
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
  FillGate gate;
  gate.destination = source;
  gate.pitch = kPitch;
  gate.width = kWidth;
  gate.height = kHeight;
  ASSERT_EQ(hipSuccess, api_.launch_host_function(hipStreamLegacy,
                                                  FillAfterRelease, &gate));
  gate.entered.Wait();

  const hipError_t copy_result = api_.memcpy_2d_to_array_async_spt(
      array_, /*destination_x_offset=*/1, /*destination_y_offset=*/1, source,
      kPitch, kWidth, kHeight, hipMemcpyHostToDevice, hipStreamLegacy);
  gate.release.Post();
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

TEST_F(HipArrayCopySptApiTest, CrossDeviceCopiesUseTheSelectedStream) {
  constexpr size_t kWidth = 32;
  int device_count = 0;
  ASSERT_EQ(hipSuccess, api_.get_device_count(&device_count));
  if (device_count < 2) {
    GTEST_SKIP() << "requires two devices";
  }

  const hipChannelFormatDesc descriptor = {
      .x = 8,
      .y = 0,
      .z = 0,
      .w = 0,
      .f = hipChannelFormatKindUnsigned,
  };
  hipArray_t destination_array = nullptr;
  ASSERT_EQ(hipSuccess, api_.set_device(/*device_id=*/0));
  ASSERT_EQ(hipSuccess,
            api_.malloc_array(&destination_array, &descriptor, kWidth, 1,
                              /*flags=*/0));

  hipDeviceptr_t source_device = nullptr;
  void* source_host = nullptr;
  ASSERT_EQ(hipSuccess, api_.set_device(/*device_id=*/1));
  ASSERT_EQ(hipSuccess, api_.malloc(&source_device, kWidth));
  ASSERT_EQ(hipSuccess,
            api_.host_alloc(&source_host, kWidth, hipHostMallocDefault));
  auto* source = static_cast<uint8_t*>(source_host);
  for (size_t i = 0; i < kWidth; ++i) {
    source[i] = static_cast<uint8_t>(i + 1);
  }

  hipStream_t explicit_stream = nullptr;
  ASSERT_EQ(hipSuccess, api_.stream_create(&explicit_stream));
  const std::array<hipStream_t, 4> stream_arguments = {
      nullptr, hipStreamLegacy, hipStreamPerThread, explicit_stream};
  for (hipStream_t stream_argument : stream_arguments) {
    const std::array<uint8_t, kWidth> zero = {};
    ASSERT_EQ(hipSuccess, api_.set_device(/*device_id=*/1));
    ASSERT_EQ(hipSuccess, api_.memcpy(source_device, zero.data(), zero.size(),
                                      hipMemcpyHostToDevice));
    WaitGate gate;
    std::atomic<bool> copy_invoked = false;
    std::atomic<bool> thread_finished = false;
    hipError_t set_device_result = hipErrorUnknown;
    hipError_t launch_result = hipErrorUnknown;
    hipError_t producer_result = hipErrorUnknown;
    hipError_t copy_result = hipErrorUnknown;
    hipError_t synchronize_result = hipErrorUnknown;
    std::thread copy_thread([&] {
      set_device_result = api_.set_device(/*device_id=*/1);
      const hipStream_t selected_stream = stream_argument == explicit_stream
                                              ? explicit_stream
                                              : hipStreamPerThread;
      if (set_device_result == hipSuccess) {
        launch_result =
            api_.launch_host_function(selected_stream, WaitForRelease, &gate);
      }
      if (launch_result == hipSuccess) {
        producer_result =
            api_.memcpy_async(source_device, source, kWidth,
                              hipMemcpyHostToDevice, selected_stream);
        if (producer_result == hipSuccess) {
          copy_invoked.store(true, std::memory_order_release);
          copy_result = api_.memcpy_2d_to_array_async_spt(
              destination_array, 0, 0, source_device, kWidth, kWidth, 1,
              hipMemcpyDeviceToDevice, stream_argument);
          synchronize_result = api_.stream_synchronize(selected_stream);
        }
      }
      thread_finished.store(true, std::memory_order_release);
    });

    while (!gate.entered.IsPosted() &&
           !thread_finished.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    while (!copy_invoked.load(std::memory_order_acquire) &&
           !thread_finished.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    EXPECT_TRUE(gate.entered.IsPosted());
    EXPECT_TRUE(copy_invoked.load(std::memory_order_acquire));
    gate.release.Post();
    copy_thread.join();

    ASSERT_EQ(hipSuccess, set_device_result);
    ASSERT_EQ(hipSuccess, launch_result);
    ASSERT_EQ(hipSuccess, producer_result);
    ASSERT_EQ(hipSuccess, copy_result);
    ASSERT_EQ(hipSuccess, synchronize_result);
    std::array<uint8_t, kWidth> destination = {};
    ASSERT_EQ(hipSuccess, api_.set_device(/*device_id=*/0));
    ASSERT_EQ(hipSuccess, api_.memcpy_2d_from_array_spt(
                              destination.data(), kWidth, destination_array, 0,
                              0, kWidth, 1, hipMemcpyDeviceToHost));
    EXPECT_EQ(0, std::memcmp(source, destination.data(), kWidth));
  }

  ASSERT_EQ(hipSuccess, api_.set_device(/*device_id=*/1));
  EXPECT_EQ(hipSuccess, api_.stream_destroy(explicit_stream));
  EXPECT_EQ(hipSuccess, api_.free(source_device));
  EXPECT_EQ(hipSuccess, api_.free_host(source_host));
  ASSERT_EQ(hipSuccess, api_.set_device(/*device_id=*/0));
  EXPECT_EQ(hipSuccess, api_.free_array(destination_array));
}

TEST_F(HipArrayCopySptApiTest, ContiguousCopyWaitsForLegacyProducer) {
  constexpr size_t kWidth = 9;
  constexpr size_t kHeight = 4;
  AllocateArray(kWidth, kHeight);
  AllocateHost(2 * kWidth * kHeight);
  auto* host = static_cast<uint8_t*>(host_pointer_);
  uint8_t* source = host;
  uint8_t* destination = host + kWidth * kHeight;
  FillGate gate;
  gate.destination = source;
  gate.pitch = kWidth;
  gate.width = kWidth;
  gate.height = kHeight;
  ASSERT_EQ(hipSuccess, api_.launch_host_function(hipStreamLegacy,
                                                  FillAfterRelease, &gate));
  gate.entered.Wait();

  const hipError_t copy_result = api_.memcpy_2d_to_array_async_spt(
      array_, 0, 0, source, kWidth, kWidth, kHeight, hipMemcpyHostToDevice,
      hipStreamLegacy);
  gate.release.Post();
  ASSERT_EQ(hipSuccess, copy_result);
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(hipStreamPerThread));
  ASSERT_EQ(hipSuccess, api_.memcpy_2d_from_array_spt(
                            destination, kWidth, array_, 0, 0, kWidth, kHeight,
                            hipMemcpyDeviceToHost));
  EXPECT_EQ(0, std::memcmp(source, destination, kWidth * kHeight));
}

TEST_F(HipArrayCopySptApiTest, PageableCopiesWaitOnlyForTheirSelectedStream) {
  constexpr size_t kWidth = 8;
  AllocateArray(kWidth, 1);
  AllocateHost(kWidth);
  auto* source = static_cast<uint8_t*>(host_pointer_);
  for (size_t i = 0; i < kWidth; ++i) {
    source[i] = static_cast<uint8_t>(i + 1);
  }
  ASSERT_EQ(hipSuccess, api_.malloc(&device_pointer_, kWidth));
  ASSERT_EQ(hipSuccess, api_.memcpy(device_pointer_, source, kWidth,
                                    hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess,
            api_.memcpy_2d_to_array_spt(array_, 0, 0, source, kWidth, kWidth, 1,
                                        hipMemcpyHostToDevice));

  // A non-default priority selects a distinct hardware queue. This keeps the
  // blocked callback from physically obstructing the default queue while
  // preserving the distinction between a selected-stream wait and a
  // context-wide wait.
  ASSERT_EQ(hipSuccess, api_.stream_create_with_priority(
                            &stream_, hipStreamNonBlocking, /*priority=*/-1));
  WaitGate gate;
  ASSERT_EQ(hipSuccess,
            api_.launch_host_function(stream_, WaitForRelease, &gate));
  gate.entered.Wait();

  std::array<uint8_t, kWidth> generic_destination = {};
  hipMemcpy3DParms parameters = {};
  parameters.srcPtr.ptr = device_pointer_;
  parameters.srcPtr.pitch = kWidth;
  parameters.srcPtr.xsize = kWidth;
  parameters.srcPtr.ysize = 1;
  parameters.dstPtr.ptr = generic_destination.data();
  parameters.dstPtr.pitch = kWidth;
  parameters.dstPtr.xsize = kWidth;
  parameters.dstPtr.ysize = 1;
  parameters.extent = {.width = kWidth, .height = 1, .depth = 1};
  parameters.kind = hipMemcpyDeviceToHost;
  EXPECT_EQ(hipSuccess, api_.memcpy_3d_async(&parameters, nullptr));

  std::array<uint8_t, kWidth> array_destination = {};
  EXPECT_EQ(hipSuccess, api_.memcpy_2d_from_array_async_spt(
                            array_destination.data(), kWidth, array_, 0, 0,
                            kWidth, 1, hipMemcpyDeviceToHost, hipStreamLegacy));

  gate.release.Post();
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
  EXPECT_EQ(0, std::memcmp(source, generic_destination.data(), kWidth));
  EXPECT_EQ(0, std::memcmp(source, array_destination.data(), kWidth));
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
  no_op_gate.entered.Wait();
  EXPECT_EQ(hipSuccess, api_.memcpy_2d_from_array_spt(
                            host, /*destination_pitch=*/1, array_, 0, 0,
                            /*width=*/0, /*height=*/1, hipMemcpyDeviceToHost));
  no_op_gate.release.Post();
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(hipStreamPerThread));

  ASSERT_EQ(hipSuccess, api_.malloc(&device_pointer_, kWidth));
  WaitGate device_gate;
  ASSERT_EQ(hipSuccess, api_.launch_host_function(
                            hipStreamPerThread, WaitForRelease, &device_gate));
  device_gate.entered.Wait();
  EXPECT_EQ(hipSuccess,
            api_.memcpy_2d_from_array_spt(device_pointer_, kWidth, array_, 0, 0,
                                          kWidth, 1, hipMemcpyDeviceToDevice));
  device_gate.release.Post();
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
  const std::array<uint8_t, kWidth> original = {1, 2, 3, 4, 5, 6, 7, 8};
  const std::array<uint8_t, kWidth> replacement = {9,  10, 11, 12,
                                                   13, 14, 15, 16};
  ASSERT_EQ(hipSuccess, api_.memcpy_to_array(array_, 0, 0, original.data(),
                                             kWidth, hipMemcpyHostToDevice));

  const auto expect_original_contents = [&] {
    std::array<uint8_t, kWidth> actual = {};
    ASSERT_EQ(hipSuccess,
              api_.memcpy_from_array_spt(actual.data(), array_, 0, 0, kWidth,
                                         hipMemcpyDeviceToHost));
    EXPECT_EQ(original, actual);
  };

  ASSERT_EQ(hipSuccess,
            api_.stream_begin_capture(hipStreamPerThread,
                                      hipStreamCaptureModeThreadLocal));
  EXPECT_EQ(
      hipErrorStreamCaptureImplicit,
      api_.memcpy_2d_to_array_spt(array_, 0, 0, replacement.data(), kWidth,
                                  kWidth, 1, hipMemcpyHostToDevice));
  hipGraph_t graph = nullptr;
  EXPECT_EQ(hipErrorStreamCaptureInvalidated,
            api_.stream_end_capture(hipStreamPerThread, &graph));
  EXPECT_EQ(nullptr, graph);
  expect_original_contents();

  ASSERT_EQ(hipSuccess, api_.stream_create(&stream_));
  ASSERT_EQ(hipSuccess, api_.stream_begin_capture(
                            stream_, hipStreamCaptureModeThreadLocal));
  EXPECT_EQ(
      hipErrorStreamCaptureImplicit,
      api_.memcpy_2d_to_array_spt(array_, 0, 0, replacement.data(), kWidth,
                                  kWidth, 1, hipMemcpyHostToDevice));
  EXPECT_EQ(hipErrorStreamCaptureInvalidated,
            api_.stream_end_capture(stream_, &graph));
  EXPECT_EQ(nullptr, graph);
  expect_original_contents();
}

TEST_F(HipArrayCopySptApiTest, CopiesPackedRowsAtRepresentativeHeights) {
  constexpr size_t kWidth = 31;
  const std::array<size_t, 3> row_counts = {1, 256, 4096};
  for (size_t row_count : row_counts) {
    AllocateArray(kWidth, row_count);
    std::vector<uint8_t> source(kWidth * row_count);
    std::vector<uint8_t> destination(source.size(), 0);
    for (size_t i = 0; i < source.size(); ++i) {
      source[i] = static_cast<uint8_t>((i * 37 + row_count) & 0xFF);
    }

    ASSERT_EQ(hipSuccess,
              api_.memcpy_to_array(array_, 0, 0, source.data(), source.size(),
                                   hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, api_.memcpy_from_array_spt(destination.data(), array_,
                                                     0, 0, destination.size(),
                                                     hipMemcpyDeviceToHost));
    EXPECT_EQ(source, destination) << "row count " << row_count;

    ASSERT_EQ(hipSuccess, api_.malloc(&device_pointer_, source.size()));
    ASSERT_EQ(hipSuccess, api_.memcpy(device_pointer_, source.data(),
                                      source.size(), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess,
              api_.memcpy_to_array(array_, 0, 0, device_pointer_, source.size(),
                                   hipMemcpyDeviceToDevice));
    std::fill(destination.begin(), destination.end(), 0);
    ASSERT_EQ(hipSuccess, api_.memcpy_from_array_spt(destination.data(), array_,
                                                     0, 0, destination.size(),
                                                     hipMemcpyDeviceToHost));
    EXPECT_EQ(source, destination) << "D2D row count " << row_count;

    std::fill(destination.begin(), destination.end(), 0);
    ASSERT_EQ(hipSuccess, api_.memcpy(device_pointer_, destination.data(),
                                      source.size(), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, api_.memcpy_from_array_spt(device_pointer_, array_, 0,
                                                     0, source.size(),
                                                     hipMemcpyDeviceToDevice));
    ASSERT_EQ(hipSuccess, api_.memcpy(destination.data(), device_pointer_,
                                      source.size(), hipMemcpyDeviceToHost));
    EXPECT_EQ(source, destination) << "array-to-D2D row count " << row_count;

    ASSERT_EQ(hipSuccess, api_.free(device_pointer_));
    device_pointer_ = nullptr;
    ASSERT_EQ(hipSuccess, api_.free_array(array_));
    array_ = nullptr;
  }
}

TEST_F(HipArrayCopySptApiTest,
       SynchronousCopyInvalidatesCapturesOwnedByAnotherThread) {
  int device_count = 0;
  ASSERT_EQ(hipSuccess, api_.get_device_count(&device_count));
  ASSERT_GT(device_count, 0);
  const int capture_device = device_count > 1 ? 1 : 0;
  const std::array<hipStreamCaptureMode, 2> capture_modes = {
      hipStreamCaptureModeThreadLocal, hipStreamCaptureModeRelaxed};
  for (size_t iteration = 0; iteration < capture_modes.size(); ++iteration) {
    std::atomic<bool> capture_started = false;
    std::atomic<bool> release_capture = false;
    hipError_t set_device_result = hipErrorUnknown;
    hipError_t stream_create_result = hipErrorUnknown;
    hipError_t begin_result = hipErrorUnknown;
    hipError_t end_result = hipErrorUnknown;
    hipError_t stream_destroy_result = hipErrorUnknown;
    hipGraph_t graph = nullptr;
    std::thread capture_thread([&] {
      hipStream_t capture_stream = nullptr;
      set_device_result = api_.set_device(capture_device);
      if (set_device_result == hipSuccess) {
        stream_create_result = api_.stream_create(&capture_stream);
      }
      if (stream_create_result == hipSuccess) {
        begin_result =
            api_.stream_begin_capture(capture_stream, capture_modes[iteration]);
      }
      capture_started.store(true, std::memory_order_release);
      while (!release_capture.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      if (begin_result == hipSuccess) {
        end_result = api_.stream_end_capture(capture_stream, &graph);
      }
      if (capture_stream) {
        stream_destroy_result = api_.stream_destroy(capture_stream);
      }
    });

    while (!capture_started.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    hipError_t main_set_device_result = hipErrorUnknown;
    hipError_t copy_result = hipErrorUnknown;
    hipError_t repeated_copy_result = hipErrorUnknown;
    if (begin_result == hipSuccess) {
      main_set_device_result = api_.set_device(/*device_id=*/0);
      if (main_set_device_result == hipSuccess) {
        const hipMemcpyKind invalid_kind = static_cast<hipMemcpyKind>(-1);
        if (iteration == 0) {
          copy_result = api_.memcpy_2d_to_array_spt(nullptr, 0, 0, nullptr, 0,
                                                    0, 0, invalid_kind);
          repeated_copy_result = api_.memcpy_2d_to_array_spt(
              nullptr, 0, 0, nullptr, 0, 0, 0, invalid_kind);
        } else {
          copy_result = api_.memcpy_from_array_spt(nullptr, nullptr, 0, 0, 0,
                                                   invalid_kind);
          repeated_copy_result = api_.memcpy_from_array_spt(nullptr, nullptr, 0,
                                                            0, 0, invalid_kind);
        }
      }
    }
    release_capture.store(true, std::memory_order_release);
    capture_thread.join();

    EXPECT_EQ(hipSuccess, set_device_result);
    EXPECT_EQ(hipSuccess, stream_create_result);
    EXPECT_EQ(hipSuccess, begin_result);
    EXPECT_EQ(hipSuccess, main_set_device_result);
    EXPECT_EQ(hipErrorStreamCaptureImplicit, copy_result);
    EXPECT_EQ(hipErrorStreamCaptureImplicit, repeated_copy_result);
    EXPECT_EQ(hipErrorStreamCaptureInvalidated, end_result);
    EXPECT_EQ(nullptr, graph);
    EXPECT_EQ(hipSuccess, stream_destroy_result);
    if (graph) {
      EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
    }
  }
  ASSERT_EQ(hipSuccess, api_.set_device(/*device_id=*/0));
}

TEST_F(HipArrayCopySptApiTest,
       ProductionCopyLeaseKeepsArrayDestructionOnFreeCaller) {
  constexpr size_t kWidth = 32;
  AllocateArray(kWidth, /*height=*/1);
  AllocateHost(2 * kWidth);
  auto* source = static_cast<uint8_t*>(host_pointer_);
  auto* destination = source + kWidth;
  for (size_t i = 0; i < kWidth; ++i) {
    source[i] = static_cast<uint8_t>(17 + i);
    destination[i] = 0;
  }
  ASSERT_EQ(hipSuccess, api_.memcpy_to_array(array_, 0, 0, source, kWidth,
                                             hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, api_.stream_create(&stream_));

  // Keep the selected stream busy so destruction on the copy thread recreates
  // the old callback/synchronize cycle. The repaired close owner instead waits
  // on the free thread while the copy API can drop its lease and return.
  WaitGate callback_gate;
  ASSERT_EQ(hipSuccess,
            api_.launch_host_function(stream_, WaitForRelease, &callback_gate));
  callback_gate.entered.Wait();

  WaitGate lease_gate;
  ASSERT_EQ(hipSuccess,
            api_.set_array_lease_observer(WaitForRelease, &lease_gate));
  TestNotification copy_returned;
  TestNotification free_started;
  TestNotification free_returned;
  hipError_t copy_result = hipErrorUnknown;
  hipError_t free_result = hipErrorUnknown;
  hipArray_t array = array_;
  std::thread copy_thread([&] {
    copy_result = api_.memcpy_2d_from_array_async_spt(
        destination, kWidth, array, 0, 0, kWidth, 1, hipMemcpyDeviceToHost,
        stream_);
    copy_returned.Post();
  });

  lease_gate.entered.Wait();

  std::thread free_thread([&] {
    free_started.Post();
    free_result = api_.free_array(array);
    free_returned.Post();
  });
  free_started.Wait();
  EXPECT_FALSE(free_returned.IsPosted());

  lease_gate.release.Post();
  copy_returned.Wait();
  EXPECT_FALSE(free_returned.IsPosted());

  callback_gate.release.Post();
  copy_thread.join();
  free_thread.join();
  EXPECT_EQ(hipSuccess, api_.set_array_lease_observer(nullptr, nullptr));
  array_ = nullptr;

  EXPECT_EQ(hipSuccess, copy_result);
  EXPECT_EQ(hipSuccess, free_result);
  EXPECT_EQ(0, std::memcmp(source, destination, kWidth));
}

TEST_F(HipArrayCopySptApiTest,
       LegacySynchronousEntryPointsHoldLeaseThroughDispatch) {
  constexpr size_t kWidth = 32;
  const hipChannelFormatDesc descriptor = {
      .x = 8,
      .y = 0,
      .z = 0,
      .w = 0,
      .f = hipChannelFormatKindUnsigned,
  };
  AllocateHost(2 * kWidth);
  auto* source = static_cast<uint8_t*>(host_pointer_);
  auto* destination = source + kWidth;
  for (size_t i = 0; i < kWidth; ++i) {
    source[i] = static_cast<uint8_t>(23 + i);
  }
  ASSERT_EQ(hipSuccess, api_.malloc(&device_pointer_, kWidth));
  ASSERT_EQ(hipSuccess, api_.memcpy(device_pointer_, source, kWidth,
                                    hipMemcpyHostToDevice));

  enum class CopyCase { kHtoA, kAtoH, kDtoA, kAtoD };
  for (CopyCase copy_case :
       {CopyCase::kHtoA, CopyCase::kAtoH, CopyCase::kDtoA, CopyCase::kAtoD}) {
    hipArray_t array = nullptr;
    ASSERT_EQ(hipSuccess, api_.malloc_array(&array, &descriptor, kWidth, 1, 0));
    if (copy_case == CopyCase::kAtoH || copy_case == CopyCase::kAtoD) {
      ASSERT_EQ(hipSuccess, api_.memcpy_htoa(array, 0, source, kWidth));
    }
    std::memset(destination, 0, kWidth);

    CountedWaitGate lease_gate;
    lease_gate.target_count = 2;
    ASSERT_EQ(hipSuccess,
              api_.set_array_lease_observer(WaitForTargetCount, &lease_gate));
    hipError_t copy_result = hipErrorUnknown;
    std::thread copy_thread([&] {
      switch (copy_case) {
        case CopyCase::kHtoA:
          copy_result = api_.memcpy_htoa(array, 0, source, kWidth);
          break;
        case CopyCase::kAtoH:
          copy_result = api_.memcpy_atoh(destination, array, 0, kWidth);
          break;
        case CopyCase::kDtoA:
          copy_result = api_.memcpy_dtoa(array, 0, device_pointer_, kWidth);
          break;
        case CopyCase::kAtoD:
          copy_result = api_.memcpy_atod(device_pointer_, array, 0, kWidth);
          break;
      }
    });

    lease_gate.entered.Wait();

    TestNotification free_started;
    TestNotification free_returned;
    hipError_t free_result = hipErrorUnknown;
    std::thread free_thread([&] {
      free_started.Post();
      free_result = api_.free_array(array);
      free_returned.Post();
    });
    free_started.Wait();
    EXPECT_FALSE(free_returned.IsPosted());

    lease_gate.release.Post();
    copy_thread.join();
    free_thread.join();
    EXPECT_EQ(hipSuccess, api_.set_array_lease_observer(nullptr, nullptr));
    EXPECT_EQ(hipSuccess, copy_result);
    EXPECT_EQ(hipSuccess, free_result);
    if (copy_case == CopyCase::kAtoH) {
      EXPECT_EQ(0, std::memcmp(source, destination, kWidth));
    } else if (copy_case == CopyCase::kAtoD) {
      std::memset(destination, 0, kWidth);
      ASSERT_EQ(hipSuccess, api_.memcpy(destination, device_pointer_, kWidth,
                                        hipMemcpyDeviceToHost));
      EXPECT_EQ(0, std::memcmp(source, destination, kWidth));
    }
  }
}

TEST_F(HipArrayCopySptApiTest,
       LegacyAsyncEntryPointsKeepAcceptedWorkAliveAfterLeaseRelease) {
  constexpr size_t kWidth = 32;
  const hipChannelFormatDesc descriptor = {
      .x = 8,
      .y = 0,
      .z = 0,
      .w = 0,
      .f = hipChannelFormatKindUnsigned,
  };
  AllocateHost(2 * kWidth);
  auto* source = static_cast<uint8_t*>(host_pointer_);
  auto* destination = source + kWidth;
  for (size_t i = 0; i < kWidth; ++i) {
    source[i] = static_cast<uint8_t>(47 + i);
  }

  for (bool array_is_source : {false, true}) {
    hipArray_t array = nullptr;
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipSuccess, api_.malloc_array(&array, &descriptor, kWidth, 1, 0));
    ASSERT_EQ(hipSuccess, api_.stream_create(&stream));
    if (array_is_source) {
      ASSERT_EQ(hipSuccess, api_.memcpy_htoa(array, 0, source, kWidth));
      std::memset(destination, 0, kWidth);
    }

    WaitGate callback_gate;

    CountedWaitGate lease_gate;
    lease_gate.target_count = 2;
    ASSERT_EQ(hipSuccess,
              api_.set_array_lease_observer(WaitForTargetCount, &lease_gate));
    TestNotification copy_returned;
    hipError_t copy_result = hipErrorUnknown;
    std::thread copy_thread([&] {
      copy_result =
          array_is_source
              ? api_.memcpy_atoh_async(destination, array, 0, kWidth, stream)
              : api_.memcpy_htoa_async(array, 0, source, kWidth, stream);
      copy_returned.Post();
    });

    lease_gate.entered.Wait();

    TestNotification free_started;
    TestNotification free_returned;
    hipError_t free_result = hipErrorUnknown;
    std::thread free_thread([&] {
      free_started.Post();
      free_result = api_.free_array(array);
      free_returned.Post();
    });
    free_started.Wait();
    EXPECT_FALSE(free_returned.IsPosted());

    ASSERT_EQ(hipSuccess, api_.launch_host_function(stream, WaitForRelease,
                                                    &callback_gate));
    callback_gate.entered.Wait();

    lease_gate.release.Post();
    copy_returned.Wait();
    EXPECT_FALSE(free_returned.IsPosted());

    callback_gate.release.Post();
    copy_thread.join();
    free_thread.join();
    EXPECT_EQ(hipSuccess, api_.set_array_lease_observer(nullptr, nullptr));
    EXPECT_EQ(hipSuccess, copy_result);
    EXPECT_EQ(hipSuccess, free_result);
    EXPECT_EQ(hipSuccess, api_.stream_destroy(stream));
    if (array_is_source) {
      EXPECT_EQ(0, std::memcmp(source, destination, kWidth));
    }
  }
}

TEST_F(HipArrayCopySptApiTest, LegacyAtoAHoldsBothArrayLeasesThroughDispatch) {
  constexpr size_t kWidth = 32;
  const hipChannelFormatDesc descriptor = {
      .x = 8,
      .y = 0,
      .z = 0,
      .w = 0,
      .f = hipChannelFormatKindUnsigned,
  };
  hipArray_t source_array = nullptr;
  hipArray_t destination_array = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.malloc_array(&source_array, &descriptor, kWidth, 1, 0));
  ASSERT_EQ(hipSuccess,
            api_.malloc_array(&destination_array, &descriptor, kWidth, 1, 0));

  CountedWaitGate lease_gate;
  lease_gate.target_count = 3;
  ASSERT_EQ(hipSuccess,
            api_.set_array_lease_observer(WaitForTargetCount, &lease_gate));
  hipError_t copy_result = hipErrorUnknown;
  std::thread copy_thread([&] {
    copy_result =
        api_.memcpy_atoa(destination_array, 0, source_array, 0, kWidth);
  });

  lease_gate.entered.Wait();

  TestNotification source_free_started;
  TestNotification destination_free_started;
  TestNotification source_free_returned;
  TestNotification destination_free_returned;
  hipError_t source_free_result = hipErrorUnknown;
  hipError_t destination_free_result = hipErrorUnknown;
  std::thread source_free_thread([&] {
    source_free_started.Post();
    source_free_result = api_.free_array(source_array);
    source_free_returned.Post();
  });
  std::thread destination_free_thread([&] {
    destination_free_started.Post();
    destination_free_result = api_.free_array(destination_array);
    destination_free_returned.Post();
  });
  source_free_started.Wait();
  destination_free_started.Wait();
  EXPECT_FALSE(source_free_returned.IsPosted());
  EXPECT_FALSE(destination_free_returned.IsPosted());

  lease_gate.release.Post();
  copy_thread.join();
  source_free_thread.join();
  destination_free_thread.join();
  EXPECT_EQ(hipSuccess, api_.set_array_lease_observer(nullptr, nullptr));
  EXPECT_EQ(hipSuccess, copy_result);
  EXPECT_EQ(hipSuccess, source_free_result);
  EXPECT_EQ(hipSuccess, destination_free_result);
}

}  // namespace
