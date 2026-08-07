// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
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
using HipGetDeviceCountFn = hipError_t (*)(int* count);
using HipSetDeviceFn = hipError_t (*)(int device);
using HipStreamCreateFn = hipError_t (*)(hipStream_t* stream);
using HipStreamDestroyFn = hipError_t (*)(hipStream_t stream);
using HipStreamSynchronizeFn = hipError_t (*)(hipStream_t stream);
using HipMallocPitchFn = hipError_t (*)(void** pointer, size_t* pitch,
                                        size_t width, size_t height);
using HipFreeFn = hipError_t (*)(hipDeviceptr_t pointer);
using HipMemcpy2DFn = hipError_t (*)(void* dst, size_t dst_pitch,
                                     const void* src, size_t src_pitch,
                                     size_t width, size_t height,
                                     hipMemcpyKind kind);
using HipMemcpy2DAsyncFn = hipError_t (*)(void* dst, size_t dst_pitch,
                                          const void* src, size_t src_pitch,
                                          size_t width, size_t height,
                                          hipMemcpyKind kind,
                                          hipStream_t stream);

struct HipRuntimeApi {
  // Handle returned by dlopen for the HIP runtime instance.
  void* library = nullptr;
  // Initializes the HIP runtime instance.
  HipInitFn init = nullptr;
  // Returns the number of addressable devices.
  HipGetDeviceCountFn get_device_count = nullptr;
  // Selects the current device for allocation.
  HipSetDeviceFn set_device = nullptr;
  // Creates a stream on the current device.
  HipStreamCreateFn stream_create = nullptr;
  // Destroys a stream.
  HipStreamDestroyFn stream_destroy = nullptr;
  // Waits for all prior work on a stream.
  HipStreamSynchronizeFn stream_synchronize = nullptr;
  // Allocates pitched device memory on the current device.
  HipMallocPitchFn malloc_pitch = nullptr;
  // Releases device memory.
  HipFreeFn free = nullptr;
  // Performs a blocking pitched copy.
  HipMemcpy2DFn memcpy_2d = nullptr;
  // Enqueues a pitched copy on a stream.
  HipMemcpy2DAsyncFn memcpy_2d_async = nullptr;
};

template <typename T>
T ResolveHipSymbol(void* library, const char* name) {
  return reinterpret_cast<T>(dlsym(library, name));
}

class HipMemoryCopyApiTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!api_.library) {
      api_.library = dlopen(CandidateLibPath(), RTLD_LAZY | RTLD_LOCAL);
      if (!api_.library) {
        GTEST_SKIP() << "cannot dlopen " << CandidateLibPath() << ": "
                     << dlerror();
      }
      api_.init = ResolveHipSymbol<HipInitFn>(api_.library, "hipInit");
      api_.get_device_count = ResolveHipSymbol<HipGetDeviceCountFn>(
          api_.library, "hipGetDeviceCount");
      api_.set_device =
          ResolveHipSymbol<HipSetDeviceFn>(api_.library, "hipSetDevice");
      api_.stream_create =
          ResolveHipSymbol<HipStreamCreateFn>(api_.library, "hipStreamCreate");
      api_.stream_destroy = ResolveHipSymbol<HipStreamDestroyFn>(
          api_.library, "hipStreamDestroy");
      api_.stream_synchronize = ResolveHipSymbol<HipStreamSynchronizeFn>(
          api_.library, "hipStreamSynchronize");
      api_.malloc_pitch =
          ResolveHipSymbol<HipMallocPitchFn>(api_.library, "hipMallocPitch");
      api_.free = ResolveHipSymbol<HipFreeFn>(api_.library, "hipFree");
      api_.memcpy_2d =
          ResolveHipSymbol<HipMemcpy2DFn>(api_.library, "hipMemcpy2D");
      api_.memcpy_2d_async = ResolveHipSymbol<HipMemcpy2DAsyncFn>(
          api_.library, "hipMemcpy2DAsync");
    }

    ASSERT_NE(nullptr, api_.init);
    ASSERT_NE(nullptr, api_.get_device_count);
    ASSERT_NE(nullptr, api_.set_device);
    ASSERT_NE(nullptr, api_.stream_create);
    ASSERT_NE(nullptr, api_.stream_destroy);
    ASSERT_NE(nullptr, api_.stream_synchronize);
    ASSERT_NE(nullptr, api_.malloc_pitch);
    ASSERT_NE(nullptr, api_.free);
    ASSERT_NE(nullptr, api_.memcpy_2d);
    ASSERT_NE(nullptr, api_.memcpy_2d_async);

    const hipError_t init_result = api_.init(/*flags=*/0);
    if (init_result != hipSuccess) {
      GTEST_SKIP() << "hipInit failed: " << init_result;
    }
    int device_count = 0;
    ASSERT_EQ(hipSuccess, api_.get_device_count(&device_count));
    if (device_count < 2) {
      GTEST_SKIP() << "cross-device stream test requires two devices";
    }

    ASSERT_EQ(hipSuccess, api_.set_device(kStreamDevice));
    ASSERT_EQ(hipSuccess, api_.stream_create(&stream_));
    ASSERT_EQ(hipSuccess, api_.set_device(kAllocationDevice));
  }

  void TearDown() override {
    if (copy_pointer_) EXPECT_EQ(hipSuccess, api_.free(copy_pointer_));
    if (device_pointer_) EXPECT_EQ(hipSuccess, api_.free(device_pointer_));
    if (stream_) {
      EXPECT_EQ(hipSuccess, api_.set_device(kStreamDevice));
      EXPECT_EQ(hipSuccess, api_.stream_destroy(stream_));
    }
  }

  // Device that owns the stream used for copy submission.
  static constexpr int kStreamDevice = 0;
  // Device that owns both pitched allocations.
  static constexpr int kAllocationDevice = 1;
  // Runtime entry points loaded once from the HIP shared object under test.
  static HipRuntimeApi api_;
  // Stream owned by kStreamDevice.
  hipStream_t stream_ = nullptr;
  // Primary pitched allocation owned by kAllocationDevice.
  void* device_pointer_ = nullptr;
  // Secondary pitched allocation owned by kAllocationDevice.
  void* copy_pointer_ = nullptr;
};

HipRuntimeApi HipMemoryCopyApiTest::api_;

TEST_F(HipMemoryCopyApiTest, PitchedCopiesAcceptStreamFromAnotherDevice) {
  constexpr size_t kWidth = 64;
  constexpr size_t kHeight = 8;
  std::vector<uint8_t> source(kWidth * kHeight);
  std::vector<uint8_t> destination(source.size());
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<uint8_t>(i);
  }

  size_t device_pitch = 0;
  ASSERT_EQ(hipSuccess, api_.malloc_pitch(&device_pointer_, &device_pitch,
                                          kWidth, kHeight));
  size_t copy_pitch = 0;
  ASSERT_EQ(hipSuccess,
            api_.malloc_pitch(&copy_pointer_, &copy_pitch, kWidth, kHeight));

  ASSERT_EQ(
      hipSuccess,
      api_.memcpy_2d_async(device_pointer_, device_pitch, source.data(), kWidth,
                           kWidth, kHeight, hipMemcpyHostToDevice, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
  ASSERT_EQ(hipSuccess, api_.memcpy_2d(destination.data(), kWidth,
                                       device_pointer_, device_pitch, kWidth,
                                       kHeight, hipMemcpyDeviceToHost));
  EXPECT_EQ(source, destination);

  std::fill(destination.begin(), destination.end(), 0);
  ASSERT_EQ(hipSuccess,
            api_.memcpy_2d_async(destination.data(), kWidth, device_pointer_,
                                 device_pitch, kWidth, kHeight,
                                 hipMemcpyDeviceToHost, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
  EXPECT_EQ(source, destination);

  ASSERT_EQ(hipSuccess,
            api_.memcpy_2d_async(copy_pointer_, copy_pitch, device_pointer_,
                                 device_pitch, kWidth, kHeight,
                                 hipMemcpyDeviceToDevice, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
  std::fill(destination.begin(), destination.end(), 0);
  ASSERT_EQ(hipSuccess,
            api_.memcpy_2d(destination.data(), kWidth, copy_pointer_,
                           copy_pitch, kWidth, kHeight, hipMemcpyDeviceToHost));
  EXPECT_EQ(source, destination);
}

}  // namespace
