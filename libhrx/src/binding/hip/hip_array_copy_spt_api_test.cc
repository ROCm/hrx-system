// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "api.h"
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
using HipMallocArrayFn = hipError_t (*)(hipArray_t* array,
                                        const hipChannelFormatDesc* descriptor,
                                        size_t width, size_t height,
                                        unsigned int flags);
using HipFreeArrayFn = hipError_t (*)(hipArray_t array);
using HipStreamSynchronizeFn = hipError_t (*)(hipStream_t stream);
using HipMemcpy2DFromArrayAsyncSptFn = hipError_t (*)(
    void* destination, size_t destination_pitch, hipArray_const_t source,
    size_t source_x_offset, size_t source_y_offset, size_t width, size_t height,
    hipMemcpyKind kind, hipStream_t stream);
using HipMemcpy2DToArrayAsyncSptFn = hipError_t (*)(
    hipArray_t destination, size_t destination_x_offset,
    size_t destination_y_offset, const void* source, size_t source_pitch,
    size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream);
using HipMemcpyFromArraySptFn = hipError_t (*)(
    void* destination, hipArray_const_t source, size_t source_x_offset,
    size_t source_y_offset, size_t count, hipMemcpyKind kind);

// Owns the process-scoped HIP runtime instance used by this test.
struct HipRuntimeApi {
  // Handle returned by dlopen for the HIP runtime instance.
  void* library = nullptr;
  // Initializes the HIP runtime instance.
  HipInitFn init = nullptr;
  // Allocates the array used by copy operations.
  HipMallocArrayFn malloc_array = nullptr;
  // Releases the array used by copy operations.
  HipFreeArrayFn free_array = nullptr;
  // Waits for work enqueued on the per-thread default stream.
  HipStreamSynchronizeFn stream_synchronize = nullptr;
  // Copies array contents to pitched memory on a selected stream.
  HipMemcpy2DFromArrayAsyncSptFn memcpy_2d_from_array_async_spt = nullptr;
  // Copies pitched memory into an array on a selected stream.
  HipMemcpy2DToArrayAsyncSptFn memcpy_2d_to_array_async_spt = nullptr;
  // Copies a packed array range through the per-thread default stream.
  HipMemcpyFromArraySptFn memcpy_from_array_spt = nullptr;
};

template <typename T>
T ResolveHipSymbol(void* library, const char* name) {
  return reinterpret_cast<T>(dlsym(library, name));
}

class HipArrayCopySptApiTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!api_.library) {
      api_.library = dlopen(CandidateLibPath(), RTLD_LAZY | RTLD_LOCAL);
      if (!api_.library) {
        GTEST_SKIP() << "cannot dlopen " << CandidateLibPath() << ": "
                     << dlerror();
      }
      api_.init = ResolveHipSymbol<HipInitFn>(api_.library, "hipInit");
      api_.malloc_array =
          ResolveHipSymbol<HipMallocArrayFn>(api_.library, "hipMallocArray");
      api_.free_array =
          ResolveHipSymbol<HipFreeArrayFn>(api_.library, "hipFreeArray");
      api_.stream_synchronize = ResolveHipSymbol<HipStreamSynchronizeFn>(
          api_.library, "hipStreamSynchronize");
      api_.memcpy_2d_from_array_async_spt =
          ResolveHipSymbol<HipMemcpy2DFromArrayAsyncSptFn>(
              api_.library, "hipMemcpy2DFromArrayAsync_spt");
      api_.memcpy_2d_to_array_async_spt =
          ResolveHipSymbol<HipMemcpy2DToArrayAsyncSptFn>(
              api_.library, "hipMemcpy2DToArrayAsync_spt");
      api_.memcpy_from_array_spt = ResolveHipSymbol<HipMemcpyFromArraySptFn>(
          api_.library, "hipMemcpyFromArray_spt");
    }

    ASSERT_NE(nullptr, api_.init);
    ASSERT_NE(nullptr, api_.malloc_array);
    ASSERT_NE(nullptr, api_.free_array);
    ASSERT_NE(nullptr, api_.stream_synchronize);
  }

  void TearDown() override {
    if (array_) {
      EXPECT_EQ(hipSuccess, api_.free_array(array_));
      array_ = nullptr;
    }
    // The runtime owns process-scoped driver services that cannot be
    // reinitialized after the final dlclose in the same process.
  }

  // Runtime entry points loaded from the HIP shared object under test.
  static HipRuntimeApi api_;
  // Array allocated by the current test case.
  hipArray_t array_ = nullptr;
};

HipRuntimeApi HipArrayCopySptApiTest::api_;

TEST_F(HipArrayCopySptApiTest, ExportsArrayCopyEntryPoints) {
  EXPECT_NE(nullptr, api_.memcpy_2d_from_array_async_spt);
  EXPECT_NE(nullptr, api_.memcpy_2d_to_array_async_spt);
  EXPECT_NE(nullptr, api_.memcpy_from_array_spt);
}

TEST_F(HipArrayCopySptApiTest, CopiesThroughPerThreadDefaultStream) {
  ASSERT_NE(nullptr, api_.memcpy_2d_from_array_async_spt);
  ASSERT_NE(nullptr, api_.memcpy_2d_to_array_async_spt);
  ASSERT_NE(nullptr, api_.memcpy_from_array_spt);

  const hipError_t init_result = api_.init(/*flags=*/0);
  if (init_result != hipSuccess) {
    GTEST_SKIP() << "hipInit failed: " << init_result;
  }

  constexpr size_t kWidth = 8;
  constexpr size_t kHeight = 3;
  constexpr size_t kElementCount = kWidth * kHeight;
  const hipChannelFormatDesc descriptor = {
      /*.x=*/8,
      /*.y=*/0,
      /*.z=*/0,
      /*.w=*/0,
      /*.f=*/hipChannelFormatKindUnsigned,
  };
  ASSERT_EQ(hipSuccess, api_.malloc_array(&array_, &descriptor, kWidth, kHeight,
                                          /*flags=*/0));

  std::array<uint8_t, kElementCount> source = {};
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<uint8_t>(i + 1);
  }
  ASSERT_EQ(hipSuccess, api_.memcpy_2d_to_array_async_spt(
                            array_, /*destination_x_offset=*/0,
                            /*destination_y_offset=*/0, source.data(),
                            /*source_pitch=*/kWidth, kWidth, kHeight,
                            hipMemcpyHostToDevice, /*stream=*/nullptr));

  std::array<uint8_t, kElementCount> async_result = {};
  ASSERT_EQ(hipSuccess,
            api_.memcpy_2d_from_array_async_spt(
                async_result.data(), /*destination_pitch=*/kWidth, array_,
                /*source_x_offset=*/0, /*source_y_offset=*/0, kWidth, kHeight,
                hipMemcpyDeviceToHost, /*stream=*/nullptr));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(hipStreamPerThread));
  EXPECT_EQ(source, async_result);

  std::array<uint8_t, kElementCount> synchronous_result = {};
  ASSERT_EQ(hipSuccess,
            api_.memcpy_from_array_spt(
                synchronous_result.data(), array_, /*source_x_offset=*/0,
                /*source_y_offset=*/0, synchronous_result.size(),
                hipMemcpyDeviceToHost));
  EXPECT_EQ(source, synchronous_result);
}

}  // namespace
