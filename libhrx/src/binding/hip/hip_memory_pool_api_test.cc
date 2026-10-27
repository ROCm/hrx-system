// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/testing/gtest.h"

namespace {

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipGetDeviceFn = hipError_t (*)(int* device);
using HipStreamCreateFn = hipError_t (*)(hipStream_t* stream);
using HipStreamDestroyFn = hipError_t (*)(hipStream_t stream);
using HipDeviceGetDefaultMemPoolFn = hipError_t (*)(hipMemPool_t* pool,
                                                    int device);
using HipDeviceSetMemPoolFn = hipError_t (*)(int device, hipMemPool_t pool);
using HipMemPoolCreateFn = hipError_t (*)(hipMemPool_t* pool,
                                          const hipMemPoolProps* properties);
using HipMemPoolDestroyFn = hipError_t (*)(hipMemPool_t pool);
using HipMallocFromPoolAsyncFn = hipError_t (*)(void** pointer, size_t size,
                                                hipMemPool_t pool,
                                                hipStream_t stream);
using HipGraphCreateFn = hipError_t (*)(hipGraph_t* graph, unsigned int flags);
using HipGraphDestroyFn = hipError_t (*)(hipGraph_t graph);
using HipGraphAddMemAllocNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, void* allocation_parameters);

// Borrowed entry points from the suite-owned HIP shared library.
struct HipRuntimeApi {
  // Initializes the HIP runtime instance.
  HipInitFn init = nullptr;
  // Returns the current device ordinal.
  HipGetDeviceFn get_device = nullptr;
  // Creates a stream for asynchronous allocation.
  HipStreamCreateFn stream_create = nullptr;
  // Destroys the stream used by asynchronous allocation.
  HipStreamDestroyFn stream_destroy = nullptr;
  // Returns a device's default memory pool.
  HipDeviceGetDefaultMemPoolFn device_get_default_mem_pool = nullptr;
  // Selects a device's current memory pool.
  HipDeviceSetMemPoolFn device_set_mem_pool = nullptr;
  // Creates a user-owned memory pool.
  HipMemPoolCreateFn mem_pool_create = nullptr;
  // Destroys a user-owned memory pool.
  HipMemPoolDestroyFn mem_pool_destroy = nullptr;
  // Allocates stream-ordered memory from an explicit pool.
  HipMallocFromPoolAsyncFn malloc_from_pool_async = nullptr;
  // Creates a graph template.
  HipGraphCreateFn graph_create = nullptr;
  // Destroys a graph template.
  HipGraphDestroyFn graph_destroy = nullptr;
  // Adds a memory-allocation node to a graph template.
  HipGraphAddMemAllocNodeFn graph_add_mem_alloc_node = nullptr;
};

template <typename T>
T ResolveHipSymbol(hrx::hip::testing::HipDso& library, const char* name) {
  return library.Resolve<T>(name);
}

class HipMemoryPoolApiTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    ASSERT_TRUE(library_.Open()) << library_.error();

    api_.init = ResolveHipSymbol<HipInitFn>(library_, "hipInit");
    api_.get_device =
        ResolveHipSymbol<HipGetDeviceFn>(library_, "hipGetDevice");
    api_.stream_create =
        ResolveHipSymbol<HipStreamCreateFn>(library_, "hipStreamCreate");
    api_.stream_destroy =
        ResolveHipSymbol<HipStreamDestroyFn>(library_, "hipStreamDestroy");
    api_.device_get_default_mem_pool =
        ResolveHipSymbol<HipDeviceGetDefaultMemPoolFn>(
            library_, "hipDeviceGetDefaultMemPool");
    api_.device_set_mem_pool = ResolveHipSymbol<HipDeviceSetMemPoolFn>(
        library_, "hipDeviceSetMemPool");
    api_.mem_pool_create =
        ResolveHipSymbol<HipMemPoolCreateFn>(library_, "hipMemPoolCreate");
    api_.mem_pool_destroy =
        ResolveHipSymbol<HipMemPoolDestroyFn>(library_, "hipMemPoolDestroy");
    api_.malloc_from_pool_async = ResolveHipSymbol<HipMallocFromPoolAsyncFn>(
        library_, "hipMallocFromPoolAsync");
    api_.graph_create =
        ResolveHipSymbol<HipGraphCreateFn>(library_, "hipGraphCreate");
    api_.graph_destroy =
        ResolveHipSymbol<HipGraphDestroyFn>(library_, "hipGraphDestroy");
    api_.graph_add_mem_alloc_node = ResolveHipSymbol<HipGraphAddMemAllocNodeFn>(
        library_, "hipGraphAddMemAllocNode");

    ASSERT_NE(nullptr, api_.init);
    ASSERT_NE(nullptr, api_.get_device);
    ASSERT_NE(nullptr, api_.stream_create);
    ASSERT_NE(nullptr, api_.stream_destroy);
    ASSERT_NE(nullptr, api_.device_get_default_mem_pool);
    ASSERT_NE(nullptr, api_.device_set_mem_pool);
    ASSERT_NE(nullptr, api_.mem_pool_create);
    ASSERT_NE(nullptr, api_.mem_pool_destroy);
    ASSERT_NE(nullptr, api_.malloc_from_pool_async);
    ASSERT_NE(nullptr, api_.graph_create);
    ASSERT_NE(nullptr, api_.graph_destroy);
    ASSERT_NE(nullptr, api_.graph_add_mem_alloc_node);

    ASSERT_EQ(hipSuccess, api_.init(/*flags=*/0));
  }

  static void TearDownTestSuite() {
    if (!library_.is_open()) {
      return;
    }
    const auto deinit =
        library_.Resolve<decltype(&hipHALDeinit)>("hipHALDeinit");
    ASSERT_NE(nullptr, deinit) << library_.error();
    EXPECT_EQ(hipSuccess, deinit());
    EXPECT_TRUE(library_.Close()) << library_.error();
  }

  void SetUp() override {
    const auto get_last_error =
        library_.Resolve<decltype(&hipGetLastError)>("hipGetLastError");
    ASSERT_NE(nullptr, get_last_error) << library_.error();
    (void)get_last_error();
    ASSERT_EQ(hipSuccess, api_.get_device(&device_));
    ASSERT_EQ(hipSuccess, api_.stream_create(&stream_));
  }

  void TearDown() override {
    if (stream_) {
      EXPECT_EQ(hipSuccess, api_.stream_destroy(stream_));
      stream_ = nullptr;
    }
  }

  // Exact HIP artifact retained through synchronous suite shutdown.
  static inline hrx::hip::testing::HipDso library_;
  // Borrowed entry points resolved once before the suite initializes.
  static HipRuntimeApi api_;
  // Stream supplied to asynchronous allocation entry points.
  hipStream_t stream_ = nullptr;
  // Device ordinal associated with the test stream and pools.
  int device_ = -1;
};

HipRuntimeApi HipMemoryPoolApiTest::api_;

TEST_F(HipMemoryPoolApiTest, ZeroByteAllocationReturnsNull) {
  hipMemPool_t pool = nullptr;
  ASSERT_EQ(hipSuccess, api_.device_get_default_mem_pool(&pool, device_));

  void* pointer = reinterpret_cast<void*>(uintptr_t{1});
  EXPECT_EQ(hipSuccess,
            api_.malloc_from_pool_async(&pointer, /*size=*/0, pool, stream_));
  EXPECT_EQ(nullptr, pointer);
}

TEST_F(HipMemoryPoolApiTest, GraphAllocationReleasesSelectedPool) {
  hipMemPool_t default_pool = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.device_get_default_mem_pool(&default_pool, device_));

  hipMemPoolProps properties = {.allocType = hipMemAllocationTypePinned};
  properties.location.type = hipMemLocationTypeDevice;
  properties.location.id = device_;

  hipMemPool_t pool = nullptr;
  ASSERT_EQ(hipSuccess, api_.mem_pool_create(&pool, &properties));

  const hipError_t set_pool_result = api_.device_set_mem_pool(device_, pool);
  EXPECT_EQ(hipSuccess, set_pool_result);
  if (set_pool_result != hipSuccess) {
    EXPECT_EQ(hipSuccess, api_.mem_pool_destroy(pool));
    return;
  }

  hipGraph_t graph = nullptr;
  const hipError_t graph_create_result = api_.graph_create(&graph, /*flags=*/0);
  EXPECT_EQ(hipSuccess, graph_create_result);
  if (graph_create_result == hipSuccess) {
    hipMemAllocNodeParams parameters = {
        .poolProps = properties,
        .bytesize = 4096,
    };
    hipGraphNode_t node = nullptr;
    EXPECT_EQ(hipSuccess, api_.graph_add_mem_alloc_node(
                              &node, graph, /*dependencies=*/nullptr,
                              /*dependency_count=*/0, &parameters));
    EXPECT_NE(nullptr, parameters.dptr);
    EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
  }

  EXPECT_EQ(hipSuccess, api_.device_set_mem_pool(device_, default_pool));
  EXPECT_EQ(hipSuccess, api_.mem_pool_destroy(pool));
}

}  // namespace
