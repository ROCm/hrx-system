// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>

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
using HipGraphAddMemFreeNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, void* device_pointer);
using HipGraphInstantiateFn = hipError_t (*)(hipGraphExec_t* executable_graph,
                                             hipGraph_t graph,
                                             hipGraphNode_t* error_node,
                                             char* log_buffer,
                                             size_t log_buffer_size);
using HipGraphExecDestroyFn = hipError_t (*)(hipGraphExec_t executable_graph);
using HipGraphLaunchFn = hipError_t (*)(hipGraphExec_t executable_graph,
                                        hipStream_t stream);
using HipStreamSynchronizeFn = hipError_t (*)(hipStream_t stream);
using HipDeviceGetGraphMemAttributeFn = hipError_t (*)(int device,
                                                       int attribute,
                                                       void* value);
using HipStreamBeginCaptureFn = hipError_t (*)(hipStream_t stream,
                                               hipStreamCaptureMode mode);
using HipStreamEndCaptureFn = hipError_t (*)(hipStream_t stream,
                                             hipGraph_t* graph);
using HipMallocAsyncFn = hipError_t (*)(void** pointer, size_t size,
                                        hipStream_t stream);
using HipFreeAsyncFn = hipError_t (*)(void* pointer, hipStream_t stream);
using HipFreeFn = hipError_t (*)(void* pointer);

// Owns an RTLD_LOCAL HIP runtime instance and the entry points exercised by
// this test. All calls use the loaded library instead of a link-time runtime.
struct HipRuntimeApi {
  // Handle returned by dlopen for the HIP runtime instance.
  void* library = nullptr;
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
  // Adds a memory-free node to a graph template.
  HipGraphAddMemFreeNodeFn graph_add_mem_free_node = nullptr;
  // Instantiates an executable graph from a graph template.
  HipGraphInstantiateFn graph_instantiate = nullptr;
  // Destroys an executable graph.
  HipGraphExecDestroyFn graph_exec_destroy = nullptr;
  // Enqueues an executable graph on a stream.
  HipGraphLaunchFn graph_launch = nullptr;
  // Waits for all previously enqueued work on a stream.
  HipStreamSynchronizeFn stream_synchronize = nullptr;
  // Queries graph-memory accounting for a device.
  HipDeviceGetGraphMemAttributeFn device_get_graph_mem_attribute = nullptr;
  // Begins stream capture into a graph template.
  HipStreamBeginCaptureFn stream_begin_capture = nullptr;
  // Ends stream capture and returns its graph template.
  HipStreamEndCaptureFn stream_end_capture = nullptr;
  // Records a stream-ordered allocation.
  HipMallocAsyncFn malloc_async = nullptr;
  // Records a stream-ordered free.
  HipFreeAsyncFn free_async = nullptr;
  // Frees a device allocation after synchronizing prior use.
  HipFreeFn free = nullptr;
};

template <typename T>
T ResolveHipSymbol(void* library, const char* name) {
  return reinterpret_cast<T>(dlsym(library, name));
}

class HipMemoryPoolApiTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!api_.library) {
      api_.library = dlopen(CandidateLibPath(), RTLD_LAZY | RTLD_LOCAL);
      if (!api_.library) {
        GTEST_SKIP() << "cannot dlopen " << CandidateLibPath() << ": "
                     << dlerror();
      }

      api_.init = ResolveHipSymbol<HipInitFn>(api_.library, "hipInit");
      api_.get_device =
          ResolveHipSymbol<HipGetDeviceFn>(api_.library, "hipGetDevice");
      api_.stream_create =
          ResolveHipSymbol<HipStreamCreateFn>(api_.library, "hipStreamCreate");
      api_.stream_destroy = ResolveHipSymbol<HipStreamDestroyFn>(
          api_.library, "hipStreamDestroy");
      api_.device_get_default_mem_pool =
          ResolveHipSymbol<HipDeviceGetDefaultMemPoolFn>(
              api_.library, "hipDeviceGetDefaultMemPool");
      api_.device_set_mem_pool = ResolveHipSymbol<HipDeviceSetMemPoolFn>(
          api_.library, "hipDeviceSetMemPool");
      api_.mem_pool_create = ResolveHipSymbol<HipMemPoolCreateFn>(
          api_.library, "hipMemPoolCreate");
      api_.mem_pool_destroy = ResolveHipSymbol<HipMemPoolDestroyFn>(
          api_.library, "hipMemPoolDestroy");
      api_.malloc_from_pool_async = ResolveHipSymbol<HipMallocFromPoolAsyncFn>(
          api_.library, "hipMallocFromPoolAsync");
      api_.graph_create =
          ResolveHipSymbol<HipGraphCreateFn>(api_.library, "hipGraphCreate");
      api_.graph_destroy =
          ResolveHipSymbol<HipGraphDestroyFn>(api_.library, "hipGraphDestroy");
      api_.graph_add_mem_alloc_node =
          ResolveHipSymbol<HipGraphAddMemAllocNodeFn>(
              api_.library, "hipGraphAddMemAllocNode");
      api_.graph_add_mem_free_node = ResolveHipSymbol<HipGraphAddMemFreeNodeFn>(
          api_.library, "hipGraphAddMemFreeNode");
      api_.graph_instantiate = ResolveHipSymbol<HipGraphInstantiateFn>(
          api_.library, "hipGraphInstantiate");
      api_.graph_exec_destroy = ResolveHipSymbol<HipGraphExecDestroyFn>(
          api_.library, "hipGraphExecDestroy");
      api_.graph_launch =
          ResolveHipSymbol<HipGraphLaunchFn>(api_.library, "hipGraphLaunch");
      api_.stream_synchronize = ResolveHipSymbol<HipStreamSynchronizeFn>(
          api_.library, "hipStreamSynchronize");
      api_.device_get_graph_mem_attribute =
          ResolveHipSymbol<HipDeviceGetGraphMemAttributeFn>(
              api_.library, "hipDeviceGetGraphMemAttribute");
      api_.stream_begin_capture = ResolveHipSymbol<HipStreamBeginCaptureFn>(
          api_.library, "hipStreamBeginCapture");
      api_.stream_end_capture = ResolveHipSymbol<HipStreamEndCaptureFn>(
          api_.library, "hipStreamEndCapture");
      api_.malloc_async =
          ResolveHipSymbol<HipMallocAsyncFn>(api_.library, "hipMallocAsync");
      api_.free_async =
          ResolveHipSymbol<HipFreeAsyncFn>(api_.library, "hipFreeAsync");
      api_.free = ResolveHipSymbol<HipFreeFn>(api_.library, "hipFree");
    }

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
    ASSERT_NE(nullptr, api_.graph_add_mem_free_node);
    ASSERT_NE(nullptr, api_.graph_instantiate);
    ASSERT_NE(nullptr, api_.graph_exec_destroy);
    ASSERT_NE(nullptr, api_.graph_launch);
    ASSERT_NE(nullptr, api_.stream_synchronize);
    ASSERT_NE(nullptr, api_.device_get_graph_mem_attribute);
    ASSERT_NE(nullptr, api_.stream_begin_capture);
    ASSERT_NE(nullptr, api_.stream_end_capture);
    ASSERT_NE(nullptr, api_.malloc_async);
    ASSERT_NE(nullptr, api_.free_async);
    ASSERT_NE(nullptr, api_.free);

    const hipError_t init_result = api_.init(/*flags=*/0);
    if (init_result != hipSuccess) {
      GTEST_SKIP() << "hipInit failed: " << init_result;
    }
    const hipError_t get_device_result = api_.get_device(&device_);
    if (get_device_result != hipSuccess) {
      GTEST_SKIP() << "hipGetDevice failed: " << get_device_result;
    }
    ASSERT_EQ(hipSuccess, api_.stream_create(&stream_));
  }

  void TearDown() override {
    if (stream_) {
      EXPECT_EQ(hipSuccess, api_.stream_destroy(stream_));
      stream_ = nullptr;
    }
    // The runtime owns process-scoped driver services that cannot be
    // reinitialized after the final dlclose in the same process.
  }

  // Runtime entry points loaded once from the HIP shared object under test.
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

  hipMemPoolProps properties = {};
  properties.allocType = hipMemAllocationTypePinned;
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
    hipMemAllocNodeParams parameters = {};
    parameters.poolProps = properties;
    parameters.bytesize = 4096;
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

TEST_F(HipMemoryPoolApiTest, GraphAllocationLaunchUsesStableVirtualAddress) {
  uint64_t used_before = 0;
  ASSERT_EQ(hipSuccess,
            api_.device_get_graph_mem_attribute(
                device_, hipGraphMemAttrUsedMemCurrent, &used_before));

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));

  hipMemAllocNodeParams parameters = {};
  parameters.poolProps.allocType = hipMemAllocationTypePinned;
  parameters.poolProps.location.type = hipMemLocationTypeDevice;
  parameters.poolProps.location.id = device_;
  parameters.bytesize = 4096;
  hipGraphNode_t allocation_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_mem_alloc_node(
                            &allocation_node, graph, /*dependencies=*/nullptr,
                            /*dependency_count=*/0, &parameters));
  ASSERT_NE(nullptr, parameters.dptr);

  hipGraphNode_t free_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_mem_free_node(
                            &free_node, graph, &allocation_node,
                            /*dependency_count=*/1, parameters.dptr));

  hipGraphExec_t executable_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_instantiate(&executable_graph, graph,
                                               /*error_node=*/nullptr,
                                               /*log_buffer=*/nullptr,
                                               /*log_buffer_size=*/0));

  hipGraphExec_t duplicate_executable_graph = nullptr;
  EXPECT_EQ(hipErrorNotSupported,
            api_.graph_instantiate(&duplicate_executable_graph, graph,
                                   /*error_node=*/nullptr,
                                   /*log_buffer=*/nullptr,
                                   /*log_buffer_size=*/0));
  EXPECT_EQ(nullptr, duplicate_executable_graph);

  ASSERT_EQ(hipSuccess, api_.graph_launch(executable_graph, stream_));
  ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(executable_graph));
  executable_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph));
  graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));

  uint64_t used_after = 0;
  EXPECT_EQ(hipSuccess,
            api_.device_get_graph_mem_attribute(
                device_, hipGraphMemAttrUsedMemCurrent, &used_after));
  EXPECT_EQ(used_before, used_after);
}

TEST_F(HipMemoryPoolApiTest, CapturedAllocationRetainsGraphOwnership) {
  ASSERT_EQ(hipSuccess,
            api_.stream_begin_capture(stream_, hipStreamCaptureModeGlobal));

  void* pointer = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc_async(&pointer, 4096, stream_));
  ASSERT_NE(nullptr, pointer);
  ASSERT_EQ(hipSuccess, api_.free_async(pointer, stream_));

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.stream_end_capture(stream_, &graph));
  ASSERT_NE(nullptr, graph);

  hipGraphExec_t executable_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_instantiate(&executable_graph, graph,
                                               /*error_node=*/nullptr,
                                               /*log_buffer=*/nullptr,
                                               /*log_buffer_size=*/0));
  ASSERT_EQ(hipSuccess, api_.graph_launch(executable_graph, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));

  EXPECT_EQ(hipSuccess, api_.graph_exec_destroy(executable_graph));
  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
}

TEST_F(HipMemoryPoolApiTest, StreamOrderedFreeReleasesGraphAllocation) {
  uint64_t used_before = 0;
  ASSERT_EQ(hipSuccess,
            api_.device_get_graph_mem_attribute(
                device_, hipGraphMemAttrUsedMemCurrent, &used_before));

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));

  hipMemAllocNodeParams parameters = {};
  parameters.poolProps.allocType = hipMemAllocationTypePinned;
  parameters.poolProps.location.type = hipMemLocationTypeDevice;
  parameters.poolProps.location.id = device_;
  parameters.bytesize = 4096;
  hipGraphNode_t allocation_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_mem_alloc_node(
                            &allocation_node, graph, /*dependencies=*/nullptr,
                            /*dependency_count=*/0, &parameters));
  ASSERT_NE(nullptr, parameters.dptr);

  hipGraphExec_t executable_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_instantiate(&executable_graph, graph,
                                               /*error_node=*/nullptr,
                                               /*log_buffer=*/nullptr,
                                               /*log_buffer_size=*/0));
  ASSERT_EQ(hipSuccess, api_.graph_launch(executable_graph, stream_));
  ASSERT_EQ(hipSuccess, api_.free_async(parameters.dptr, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));

  uint64_t used_after = 0;
  EXPECT_EQ(hipSuccess,
            api_.device_get_graph_mem_attribute(
                device_, hipGraphMemAttrUsedMemCurrent, &used_after));
  EXPECT_EQ(used_before, used_after);

  // A later launch reactivates the graph's stable virtual address and must
  // make it available to the same stream-ordered free path again.
  ASSERT_EQ(hipSuccess, api_.graph_launch(executable_graph, stream_));
  ASSERT_EQ(hipSuccess, api_.free_async(parameters.dptr, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));

  used_after = 0;
  EXPECT_EQ(hipSuccess,
            api_.device_get_graph_mem_attribute(
                device_, hipGraphMemAttrUsedMemCurrent, &used_after));
  EXPECT_EQ(used_before, used_after);

  EXPECT_EQ(hipSuccess, api_.graph_exec_destroy(executable_graph));
  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
}

TEST_F(HipMemoryPoolApiTest, GraphFreeRetiresAndRelaunchesStablePointer) {
  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));

  hipMemAllocNodeParams parameters = {};
  parameters.poolProps.allocType = hipMemAllocationTypePinned;
  parameters.poolProps.location.type = hipMemLocationTypeDevice;
  parameters.poolProps.location.id = device_;
  parameters.bytesize = 4096;
  hipGraphNode_t allocation_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_mem_alloc_node(
                            &allocation_node, graph, /*dependencies=*/nullptr,
                            /*dependency_count=*/0, &parameters));
  ASSERT_NE(nullptr, parameters.dptr);

  hipGraphNode_t free_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_mem_free_node(
                            &free_node, graph, &allocation_node,
                            /*dependency_count=*/1, parameters.dptr));

  hipGraphExec_t executable_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_instantiate(&executable_graph, graph,
                                               /*error_node=*/nullptr,
                                               /*log_buffer=*/nullptr,
                                               /*log_buffer_size=*/0));
  ASSERT_EQ(hipSuccess, api_.graph_launch(executable_graph, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
  EXPECT_EQ(hipErrorInvalidValue, api_.free(parameters.dptr));

  // The graph free retires the current pointer lifetime. A later launch
  // republishes the same stable address and frees its new lifetime again.
  ASSERT_EQ(hipSuccess, api_.graph_launch(executable_graph, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
  EXPECT_EQ(hipErrorInvalidValue, api_.free(parameters.dptr));

  EXPECT_EQ(hipSuccess, api_.graph_exec_destroy(executable_graph));
  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
}

}  // namespace
