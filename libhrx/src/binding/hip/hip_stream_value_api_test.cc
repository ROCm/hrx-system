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
#include <utility>
#include <vector>

#include "binding/hip/api.h"
#include "iree/testing/gtest.h"

namespace {

template <typename Cleanup>
class ScopeExit {
 public:
  explicit ScopeExit(Cleanup cleanup) : cleanup_(std::move(cleanup)) {}
  ~ScopeExit() { cleanup_(); }
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;

 private:
  Cleanup cleanup_;
};
template <typename Cleanup>
ScopeExit(Cleanup) -> ScopeExit<Cleanup>;

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
using HipDeviceGetAttributeFn = hipError_t (*)(int* value,
                                               hipDeviceAttribute_t attribute,
                                               int device);
using HipMallocFn = hipError_t (*)(void** pointer, size_t size);
using HipExtMallocWithFlagsFn = hipError_t (*)(void** pointer, size_t size,
                                               unsigned int flags);
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
using HipStreamQueryFn = hipError_t (*)(hipStream_t stream);
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
using HipCtxCreateFn = hipError_t (*)(hipCtx_t* context, unsigned int flags,
                                      hipDevice_t device);
using HipCtxDestroyFn = hipError_t (*)(hipCtx_t context);
using HipCtxGetCurrentFn = hipError_t (*)(hipCtx_t* context);
using HipCtxSetCurrentFn = hipError_t (*)(hipCtx_t context);
using HipCtxEnablePeerAccessFn = hipError_t (*)(hipCtx_t peer_context,
                                                unsigned int flags);
using HipGraphCreateFn = hipError_t (*)(hipGraph_t* graph, unsigned int flags);
using HipGraphGetNodesFn = hipError_t (*)(hipGraph_t graph,
                                          hipGraphNode_t* nodes,
                                          size_t* node_count);
using HipGraphNodeGetTypeFn = hipError_t (*)(hipGraphNode_t node,
                                             hipGraphNodeType* type);
using HipGraphAddBatchMemOpNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, const void* params);
using HipGraphBatchMemOpNodeGetParamsFn = hipError_t (*)(hipGraphNode_t node,
                                                         void* params);
using HipGraphBatchMemOpNodeSetParamsFn = hipError_t (*)(hipGraphNode_t node,
                                                         const void* params);
using HipGraphExecBatchMemOpNodeSetParamsFn = hipError_t (*)(
    hipGraphExec_t executable, hipGraphNode_t node, const void* params);

struct HipRuntimeApi {
  // Handle returned by dlopen for the HIP runtime instance.
  void* library = nullptr;
  // Initializes the runtime.
  HipInitFn init = nullptr;
  // Queries one device capability.
  HipDeviceGetAttributeFn device_get_attribute = nullptr;
  // Allocates device-visible memory.
  HipMallocFn malloc = nullptr;
  // Allocates device-visible memory with an explicit allocation mode.
  HipExtMallocWithFlagsFn ext_malloc_with_flags = nullptr;
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
  // Queries one stream without blocking.
  HipStreamQueryFn stream_query = nullptr;
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
  // Creates a context and makes it current.
  HipCtxCreateFn ctx_create = nullptr;
  // Destroys a context.
  HipCtxDestroyFn ctx_destroy = nullptr;
  // Returns the current context.
  HipCtxGetCurrentFn ctx_get_current = nullptr;
  // Replaces the current context.
  HipCtxSetCurrentFn ctx_set_current = nullptr;
  // Enables access to allocations owned by another context.
  HipCtxEnablePeerAccessFn ctx_enable_peer_access = nullptr;
  // Creates an empty graph template.
  HipGraphCreateFn graph_create = nullptr;
  // Enumerates public graph nodes.
  HipGraphGetNodesFn graph_get_nodes = nullptr;
  // Queries a public graph node type.
  HipGraphNodeGetTypeFn graph_node_get_type = nullptr;
  // Adds a batch memory operation node.
  HipGraphAddBatchMemOpNodeFn graph_add_batch_mem_op_node = nullptr;
  // Queries a batch memory operation node.
  HipGraphBatchMemOpNodeGetParamsFn graph_batch_mem_op_node_get_params =
      nullptr;
  // Updates a batch memory operation node.
  HipGraphBatchMemOpNodeSetParamsFn graph_batch_mem_op_node_set_params =
      nullptr;
  // Updates a batch node in an instantiated graph.
  HipGraphExecBatchMemOpNodeSetParamsFn
      graph_exec_batch_mem_op_node_set_params = nullptr;
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
      const char* library_path = CandidateLibPath();
      ASSERT_NE(nullptr, library_path)
          << "the build must provide the libamdhip64 artifact under test";
      api_.library = dlopen(library_path, RTLD_LAZY | RTLD_LOCAL);
      ASSERT_NE(nullptr, api_.library)
          << "cannot dlopen " << library_path << ": " << dlerror();

      api_.init = ResolveHipSymbol<HipInitFn>(api_.library, "hipInit");
      api_.device_get_attribute = ResolveHipSymbol<HipDeviceGetAttributeFn>(
          api_.library, "hipDeviceGetAttribute");
      api_.malloc = ResolveHipSymbol<HipMallocFn>(api_.library, "hipMalloc");
      api_.ext_malloc_with_flags = ResolveHipSymbol<HipExtMallocWithFlagsFn>(
          api_.library, "hipExtMallocWithFlags");
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
      api_.stream_query =
          ResolveHipSymbol<HipStreamQueryFn>(api_.library, "hipStreamQuery");
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
      api_.ctx_create =
          ResolveHipSymbol<HipCtxCreateFn>(api_.library, "hipCtxCreate");
      api_.ctx_destroy =
          ResolveHipSymbol<HipCtxDestroyFn>(api_.library, "hipCtxDestroy");
      api_.ctx_get_current = ResolveHipSymbol<HipCtxGetCurrentFn>(
          api_.library, "hipCtxGetCurrent");
      api_.ctx_set_current = ResolveHipSymbol<HipCtxSetCurrentFn>(
          api_.library, "hipCtxSetCurrent");
      api_.ctx_enable_peer_access = ResolveHipSymbol<HipCtxEnablePeerAccessFn>(
          api_.library, "hipCtxEnablePeerAccess");
      api_.graph_create =
          ResolveHipSymbol<HipGraphCreateFn>(api_.library, "hipGraphCreate");
      api_.graph_get_nodes = ResolveHipSymbol<HipGraphGetNodesFn>(
          api_.library, "hipGraphGetNodes");
      api_.graph_node_get_type = ResolveHipSymbol<HipGraphNodeGetTypeFn>(
          api_.library, "hipGraphNodeGetType");
      api_.graph_add_batch_mem_op_node =
          ResolveHipSymbol<HipGraphAddBatchMemOpNodeFn>(
              api_.library, "hipGraphAddBatchMemOpNode");
      api_.graph_batch_mem_op_node_get_params =
          ResolveHipSymbol<HipGraphBatchMemOpNodeGetParamsFn>(
              api_.library, "hipGraphBatchMemOpNodeGetParams");
      api_.graph_batch_mem_op_node_set_params =
          ResolveHipSymbol<HipGraphBatchMemOpNodeSetParamsFn>(
              api_.library, "hipGraphBatchMemOpNodeSetParams");
      api_.graph_exec_batch_mem_op_node_set_params =
          ResolveHipSymbol<HipGraphExecBatchMemOpNodeSetParamsFn>(
              api_.library, "hipGraphExecBatchMemOpNodeSetParams");
    }

    ASSERT_NE(nullptr, api_.init);
    ASSERT_NE(nullptr, api_.device_get_attribute);
    ASSERT_NE(nullptr, api_.malloc);
    ASSERT_NE(nullptr, api_.ext_malloc_with_flags);
    ASSERT_NE(nullptr, api_.free);
    ASSERT_NE(nullptr, api_.malloc_host);
    ASSERT_NE(nullptr, api_.free_host);
    ASSERT_NE(nullptr, api_.host_get_device_pointer);
    ASSERT_NE(nullptr, api_.memset);
    ASSERT_NE(nullptr, api_.memcpy);
    ASSERT_NE(nullptr, api_.stream_create);
    ASSERT_NE(nullptr, api_.stream_destroy);
    ASSERT_NE(nullptr, api_.stream_query);
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
    ASSERT_NE(nullptr, api_.ctx_create);
    ASSERT_NE(nullptr, api_.ctx_destroy);
    ASSERT_NE(nullptr, api_.ctx_get_current);
    ASSERT_NE(nullptr, api_.ctx_set_current);
    ASSERT_NE(nullptr, api_.ctx_enable_peer_access);
    ASSERT_NE(nullptr, api_.graph_create);
    ASSERT_NE(nullptr, api_.graph_get_nodes);
    ASSERT_NE(nullptr, api_.graph_node_get_type);
    ASSERT_NE(nullptr, api_.graph_add_batch_mem_op_node);
    ASSERT_NE(nullptr, api_.graph_batch_mem_op_node_get_params);
    ASSERT_NE(nullptr, api_.graph_batch_mem_op_node_set_params);
    ASSERT_NE(nullptr, api_.graph_exec_batch_mem_op_node_set_params);

    ASSERT_EQ(hipSuccess, api_.init(/*flags=*/0));
    int can_use_stream_wait_value = 0;
    ASSERT_EQ(hipSuccess,
              api_.device_get_attribute(&can_use_stream_wait_value,
                                        hipDeviceAttributeCanUseStreamWaitValue,
                                        /*device=*/0));
    supports_value_waits_ = can_use_stream_wait_value != 0;
    if (const char* expected_wait_support =
            std::getenv("HRX_TEST_EXPECT_STREAM_VALUE_WAITS")) {
      ASSERT_TRUE(std::strcmp(expected_wait_support, "0") == 0 ||
                  std::strcmp(expected_wait_support, "1") == 0)
          << "HRX_TEST_EXPECT_STREAM_VALUE_WAITS must be 0 or 1";
      ASSERT_EQ(std::strcmp(expected_wait_support, "1") == 0,
                supports_value_waits_)
          << "runner stream memory wait capability differs from its configured "
             "expectation";
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
    if (stream) {
      streams_.push_back(stream);
    }
    return stream;
  }

  void* Allocate(size_t size) {
    void* pointer = nullptr;
    EXPECT_EQ(hipSuccess, api_.malloc(&pointer, size));
    if (pointer) {
      allocations_.push_back(pointer);
    }
    return pointer;
  }

  void* AllocateSignal() {
    void* pointer = nullptr;
    EXPECT_EQ(hipSuccess, api_.ext_malloc_with_flags(&pointer, sizeof(uint64_t),
                                                     hipMallocSignalMemory));
    if (pointer) {
      allocations_.push_back(pointer);
    }
    return pointer;
  }

  bool CheckWaitSupport(hipStream_t stream, void* signal) {
    if (supports_value_waits_) {
      return true;
    }
    EXPECT_EQ(hipErrorNotSupported,
              api_.wait_value_32(stream, signal, 0, hipStreamWaitValueEq,
                                 UINT32_MAX));
    return false;
  }

  void* AllocateHost(size_t size) {
    void* pointer = nullptr;
    EXPECT_EQ(hipSuccess, api_.malloc_host(&pointer, size));
    if (pointer) {
      host_allocations_.push_back(pointer);
    }
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
  bool supports_value_waits_ = false;
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

TEST_F(HipStreamValueApiTest, RejectsIncompatibleImportedBatchTargets) {
  hipCtx_t original_context = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_get_current(&original_context));

  hipCtx_t owner_context = nullptr;
  hipCtx_t execution_context = nullptr;
  void* allocation = nullptr;
  hipStream_t stream = nullptr;
  auto cleanup = [&] {
    if (stream) {
      EXPECT_EQ(hipSuccess, api_.stream_destroy(stream));
      stream = nullptr;
    }
    if (allocation) {
      EXPECT_EQ(hipSuccess, api_.ctx_set_current(owner_context));
      EXPECT_EQ(hipSuccess, api_.free(allocation));
      allocation = nullptr;
    }
    EXPECT_EQ(hipSuccess, api_.ctx_set_current(original_context));
    if (execution_context) {
      EXPECT_EQ(hipSuccess, api_.ctx_destroy(execution_context));
      execution_context = nullptr;
    }
    if (owner_context) {
      EXPECT_EQ(hipSuccess, api_.ctx_destroy(owner_context));
      owner_context = nullptr;
    }
  };
  hipError_t setup_result =
      api_.ctx_create(&owner_context, /*flags=*/0, /*device=*/0);
  if (setup_result == hipSuccess) {
    setup_result = api_.malloc(&allocation, sizeof(uint64_t));
  }
  if (setup_result == hipSuccess) {
    setup_result =
        api_.ctx_create(&execution_context, /*flags=*/0, /*device=*/0);
  }
  if (setup_result == hipSuccess) {
    setup_result = api_.ctx_enable_peer_access(owner_context, /*flags=*/0);
  }
  if (setup_result == hipSuccess) {
    setup_result = api_.stream_create(&stream);
  }
  if (setup_result != hipSuccess) {
    cleanup();
    FAIL() << "cross-context target setup failed with " << setup_result;
  }

  hipStreamBatchMemOpParams parameters[2] = {};
  parameters[0].writeValue.operation = hipStreamMemOpWriteValue32;
  parameters[0].writeValue.address = (hipDeviceptr_t)(uintptr_t)allocation;
  parameters[0].writeValue.value = 1;
  parameters[0].writeValue.flags = hipStreamWriteValueDefault;
  parameters[1].writeValue.operation = hipStreamMemOpWriteValue64;
  parameters[1].writeValue.address = (hipDeviceptr_t)(uintptr_t)allocation;
  parameters[1].writeValue.value64 = 1;
  parameters[1].writeValue.flags = hipStreamWriteValueDefault;

  // Submission is deferred until the stream is flushed. Synchronization must
  // preserve the incompatible imported target as an invalid value instead of
  // misclassifying the initialized runtime as uninitialized.
  ASSERT_EQ(hipSuccess, api_.batch_mem_op(stream, 2, parameters, /*flags=*/0));
  EXPECT_EQ(hipErrorInvalidValue, api_.stream_synchronize(stream));
  cleanup();
}

TEST_F(HipStreamValueApiTest, FailedWaitLaneDoesNotPoisonOtherStreams) {
  hipCtx_t original_context = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_get_current(&original_context));

  hipCtx_t owner_context = nullptr;
  hipCtx_t execution_context = nullptr;
  void* imported_target = nullptr;
  void* ready_signal = nullptr;
  void* blocked_signal = nullptr;
  const uint32_t ready_value = 1;
  const uint32_t blocked_value = 0;
  std::vector<hipStream_t> context_streams;
  auto cleanup = [&] {
    if (execution_context) {
      EXPECT_EQ(hipSuccess, api_.ctx_set_current(execution_context));
      if (blocked_signal) {
        __atomic_store_n(static_cast<uint32_t*>(blocked_signal), ready_value,
                         __ATOMIC_RELEASE);
      }
      for (hipStream_t stream : context_streams) {
        EXPECT_EQ(hipSuccess, api_.stream_destroy(stream));
      }
      context_streams.clear();
      if (ready_signal) {
        EXPECT_EQ(hipSuccess, api_.free(ready_signal));
        ready_signal = nullptr;
      }
      if (blocked_signal) {
        EXPECT_EQ(hipSuccess, api_.free(blocked_signal));
        blocked_signal = nullptr;
      }
    }
    if (imported_target) {
      EXPECT_EQ(hipSuccess, api_.ctx_set_current(owner_context));
      EXPECT_EQ(hipSuccess, api_.free(imported_target));
      imported_target = nullptr;
    }
    EXPECT_EQ(hipSuccess, api_.ctx_set_current(original_context));
    if (execution_context) {
      EXPECT_EQ(hipSuccess, api_.ctx_destroy(execution_context));
      execution_context = nullptr;
    }
    if (owner_context) {
      EXPECT_EQ(hipSuccess, api_.ctx_destroy(owner_context));
      owner_context = nullptr;
    }
  };
  ScopeExit cleanup_guard(cleanup);

  hipError_t setup_result =
      api_.ctx_create(&owner_context, /*flags=*/0, /*device=*/0);
  if (setup_result == hipSuccess) {
    setup_result = api_.malloc(&imported_target, sizeof(uint64_t));
  }
  if (setup_result == hipSuccess) {
    setup_result = api_.memset(imported_target, 0, sizeof(uint64_t));
  }
  if (setup_result == hipSuccess) {
    setup_result =
        api_.ctx_create(&execution_context, /*flags=*/0, /*device=*/0);
  }
  if (setup_result == hipSuccess) {
    setup_result = api_.ctx_enable_peer_access(owner_context, /*flags=*/0);
  }
  if (setup_result == hipSuccess) {
    setup_result = api_.ext_malloc_with_flags(&ready_signal, sizeof(uint64_t),
                                              hipMallocSignalMemory);
  }
  if (setup_result == hipSuccess) {
    setup_result = api_.ext_malloc_with_flags(&blocked_signal, sizeof(uint64_t),
                                              hipMallocSignalMemory);
  }
  if (setup_result == hipSuccess) {
    setup_result = api_.memcpy(ready_signal, &ready_value, sizeof(ready_value),
                               hipMemcpyHostToDevice);
  }
  if (setup_result == hipSuccess) {
    setup_result = api_.memcpy(blocked_signal, &blocked_value,
                               sizeof(blocked_value), hipMemcpyHostToDevice);
  }
  if (setup_result != hipSuccess) {
    FAIL() << "cross-context wait-lane setup failed with " << setup_result;
  }

  hipStream_t support_stream = nullptr;
  ASSERT_EQ(hipSuccess, api_.stream_create(&support_stream));
  context_streams.push_back(support_stream);
  if (!CheckWaitSupport(support_stream, ready_signal)) {
    GTEST_SKIP() << "stream memory waits are unsupported on this runner";
  }
  ASSERT_EQ(hipSuccess, api_.stream_destroy(support_stream));
  context_streams.clear();

  for (bool synchronize_owner_first : {false, true}) {
    ASSERT_EQ(hipSuccess,
              api_.memcpy(blocked_signal, &blocked_value, sizeof(blocked_value),
                          hipMemcpyHostToDevice));
    hipStream_t owner_stream = nullptr;
    hipStream_t peer_stream = nullptr;
    hipStream_t fresh_stream = nullptr;
    ASSERT_EQ(hipSuccess, api_.stream_create(&owner_stream));
    context_streams.push_back(owner_stream);
    ASSERT_EQ(hipSuccess, api_.stream_create(&peer_stream));
    context_streams.push_back(peer_stream);
    ASSERT_EQ(hipSuccess, api_.stream_create(&fresh_stream));
    context_streams.push_back(fresh_stream);

    ASSERT_EQ(hipSuccess,
              api_.wait_value_32(owner_stream, blocked_signal, ready_value,
                                 hipStreamWaitValueEq, UINT32_MAX));

    hipStreamBatchMemOpParams invalid_operations[2] = {};
    invalid_operations[0].writeValue.operation = hipStreamMemOpWriteValue32;
    invalid_operations[0].writeValue.address =
        (hipDeviceptr_t)(uintptr_t)imported_target;
    invalid_operations[0].writeValue.value = 1;
    invalid_operations[0].writeValue.flags = hipStreamWriteValueDefault;
    invalid_operations[1].writeValue.operation = hipStreamMemOpWriteValue64;
    invalid_operations[1].writeValue.address =
        (hipDeviceptr_t)(uintptr_t)imported_target;
    invalid_operations[1].writeValue.value64 = 1;
    invalid_operations[1].writeValue.flags = hipStreamWriteValueDefault;

    ASSERT_EQ(hipSuccess, api_.batch_mem_op(owner_stream, 2, invalid_operations,
                                            /*flags=*/0));
    const hipError_t owner_query_result = api_.stream_query(owner_stream);
    ASSERT_TRUE(owner_query_result == hipErrorNotReady ||
                owner_query_result == hipErrorInvalidValue);
    EXPECT_EQ(blocked_value,
              __atomic_load_n(static_cast<uint32_t*>(blocked_signal),
                              __ATOMIC_ACQUIRE));

    // Command-buffer validation may fail eagerly or after the owner's earlier
    // wait has completed. In either case, unrelated streams must make progress
    // while the accepted lane remains occupied and must stay usable after the
    // deferred owner error is observed.
    EXPECT_EQ(hipSuccess,
              api_.wait_value_32(peer_stream, ready_signal, ready_value,
                                 hipStreamWaitValueEq, UINT32_MAX));
    EXPECT_EQ(hipSuccess, api_.stream_synchronize(peer_stream));
    EXPECT_EQ(hipSuccess,
              api_.wait_value_32(fresh_stream, ready_signal, ready_value,
                                 hipStreamWaitValueEq, UINT32_MAX));
    EXPECT_EQ(hipSuccess, api_.stream_synchronize(fresh_stream));
    __atomic_store_n(static_cast<uint32_t*>(blocked_signal), ready_value,
                     __ATOMIC_RELEASE);

    if (synchronize_owner_first) {
      EXPECT_EQ(hipErrorInvalidValue, api_.stream_synchronize(owner_stream));
    }
    EXPECT_EQ(hipSuccess,
              api_.wait_value_32(peer_stream, ready_signal, ready_value,
                                 hipStreamWaitValueEq, UINT32_MAX));
    EXPECT_EQ(hipSuccess, api_.stream_synchronize(peer_stream));
    if (!synchronize_owner_first) {
      EXPECT_EQ(hipErrorInvalidValue, api_.stream_synchronize(owner_stream));
    }

    EXPECT_EQ(hipErrorInvalidValue, api_.stream_destroy(owner_stream));
    EXPECT_EQ(hipSuccess, api_.stream_destroy(peer_stream));
    EXPECT_EQ(hipSuccess, api_.stream_destroy(fresh_stream));
    context_streams.clear();
  }
}

TEST_F(HipStreamValueApiTest, PublishesFinalWriteWithoutHostFlush) {
  hipStream_t stream = CreateStream();
  void* host_allocation = AllocateHost(sizeof(uint32_t));
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, host_allocation);

  hipDeviceptr_t device_pointer = nullptr;
  ASSERT_EQ(hipSuccess, api_.host_get_device_pointer(
                            &device_pointer, host_allocation, /*flags=*/0));
  ASSERT_NE(nullptr, device_pointer);
  auto* observed_value = static_cast<uint32_t*>(host_allocation);
  __atomic_store_n(observed_value, 0u, __ATOMIC_RELEASE);

  constexpr uint32_t kExpectedValue = 0x13579BDFu;
  ASSERT_EQ(hipSuccess,
            api_.write_value_32(stream, device_pointer, kExpectedValue,
                                hipStreamWriteValueDefault));

  // This deliberately performs no stream query or synchronization: the write
  // API must arrange submission of its final batch without another HIP call.
  while (__atomic_load_n(observed_value, __ATOMIC_ACQUIRE) != kExpectedValue) {
    std::this_thread::yield();
  }
  EXPECT_EQ(kExpectedValue, __atomic_load_n(observed_value, __ATOMIC_ACQUIRE));
}

TEST_F(HipStreamValueApiTest, ExecutesEveryWaitPredicateAtBothWidths) {
  hipStream_t stream = CreateStream();
  void* signal = AllocateSignal();
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, signal);
  if (!CheckWaitSupport(stream, signal)) {
    GTEST_SKIP() << "stream memory waits are unsupported on this runner";
  }

  const unsigned int predicates[] = {
      hipStreamWaitValueGte,
      hipStreamWaitValueEq,
      hipStreamWaitValueAnd,
      hipStreamWaitValueNor,
  };
  const uint32_t initial_32[] = {7, 9, 4, 5};
  const uint64_t initial_64[] = {7, 9, 4, 5};
  const uint32_t expected_32[] = {5, 9, 4, 5};
  const uint64_t expected_64[] = {5, 9, 4, 5};
  const uint32_t masks_32[] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, 0xFu};
  const uint64_t masks_64[] = {UINT64_MAX, UINT64_MAX, UINT64_MAX, 0xFu};
  for (size_t i = 0; i < 4; ++i) {
    ASSERT_EQ(hipSuccess,
              api_.memcpy(signal, &initial_32[i], sizeof(initial_32[i]),
                          hipMemcpyHostToDevice));
    EXPECT_EQ(hipSuccess, api_.wait_value_32(stream, signal, expected_32[i],
                                             predicates[i], masks_32[i]));
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream));

    ASSERT_EQ(hipSuccess,
              api_.memcpy(signal, &initial_64[i], sizeof(initial_64[i]),
                          hipMemcpyHostToDevice));
    EXPECT_EQ(hipSuccess, api_.wait_value_64(stream, signal, expected_64[i],
                                             predicates[i], masks_64[i]));
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream));
  }
}

TEST_F(HipStreamValueApiTest, ExecutesSystemScopeWritesOnHostMemory) {
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
}

TEST_F(HipStreamValueApiTest, ExecutesSystemScopeWaitsOnHostMemory) {
  hipStream_t stream = CreateStream();
  void* host_allocation = AllocateHost(16);
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, host_allocation);

  hipDeviceptr_t device_pointer = nullptr;
  ASSERT_EQ(hipSuccess, api_.host_get_device_pointer(
                            &device_pointer, host_allocation, /*flags=*/0));
  ASSERT_NE(nullptr, device_pointer);
  if (!CheckWaitSupport(stream, device_pointer)) {
    GTEST_SKIP() << "stream memory waits are unsupported on this runner";
  }

  constexpr uint32_t kValue32 = 10;
  constexpr uint64_t kValue64 = 15;
  std::memcpy(host_allocation, &kValue32, sizeof(kValue32));
  std::memcpy(static_cast<uint8_t*>(host_allocation) + 8, &kValue64,
              sizeof(kValue64));

  EXPECT_EQ(hipSuccess, api_.wait_value_32(stream, device_pointer, kValue32,
                                           hipStreamWaitValueEq, UINT32_MAX));
  EXPECT_EQ(hipSuccess, api_.wait_value_64(
                            stream, static_cast<uint8_t*>(device_pointer) + 8,
                            kValue64, hipStreamWaitValueEq, UINT64_MAX));
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

TEST_F(HipStreamValueApiTest, ExecutesMaximumBatchAcrossUniqueAllocations) {
  hipStream_t stream = CreateStream();
  ASSERT_NE(nullptr, stream);

  constexpr size_t kOperationCount = 256;
  std::vector<void*> allocations;
  allocations.reserve(kOperationCount);
  std::vector<hipStreamBatchMemOpParams> parameters(kOperationCount);
  for (size_t i = 0; i < kOperationCount; ++i) {
    void* allocation = Allocate(sizeof(uint32_t));
    ASSERT_NE(nullptr, allocation);
    allocations.push_back(allocation);
    parameters[i].writeValue.operation = hipStreamMemOpWriteValue32;
    parameters[i].writeValue.address = (hipDeviceptr_t)(uintptr_t)allocation;
    parameters[i].writeValue.value = static_cast<uint32_t>(i + 1);
    parameters[i].writeValue.flags = hipStreamWriteValueDefault;
  }

  ASSERT_EQ(
      hipSuccess,
      api_.batch_mem_op(stream, static_cast<unsigned int>(parameters.size()),
                        parameters.data(), /*flags=*/0));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream));

  for (size_t i = 0; i < kOperationCount; ++i) {
    uint32_t observed = 0;
    ASSERT_EQ(hipSuccess, api_.memcpy(&observed, allocations[i],
                                      sizeof(observed), hipMemcpyDeviceToHost));
    EXPECT_EQ(i + 1, observed);
  }
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
  void* value_x = AllocateSignal();
  void* value_y = AllocateSignal();
  void* value_x_64 = AllocateSignal();
  void* value_y_64 = AllocateSignal();
  ASSERT_NE(nullptr, stream_a);
  ASSERT_NE(nullptr, stream_b);
  ASSERT_NE(nullptr, value_x);
  ASSERT_NE(nullptr, value_y);
  ASSERT_NE(nullptr, value_x_64);
  ASSERT_NE(nullptr, value_y_64);
  ASSERT_EQ(hipSuccess, api_.memset(value_x, 0, sizeof(uint64_t)));
  ASSERT_EQ(hipSuccess, api_.memset(value_y, 0, sizeof(uint64_t)));
  ASSERT_EQ(hipSuccess, api_.memset(value_x_64, 0, sizeof(uint64_t)));
  ASSERT_EQ(hipSuccess, api_.memset(value_y_64, 0, sizeof(uint64_t)));
  if (!CheckWaitSupport(stream_a, value_x)) {
    GTEST_SKIP() << "stream memory waits are unsupported on this runner";
  }
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

TEST_F(HipStreamValueApiTest, SameStreamPendingWaitsShareAQueueLane) {
  hipStream_t wait_stream = CreateStream();
  void* signal = AllocateSignal();
  ASSERT_NE(nullptr, wait_stream);
  ASSERT_NE(nullptr, signal);
  ASSERT_EQ(hipSuccess, api_.memset(signal, 0, sizeof(uint32_t)));
  if (!CheckWaitSupport(wait_stream, signal)) {
    GTEST_SKIP() << "stream memory waits are unsupported on this runner";
  }

  // Queue more unresolved waits than the backend's dynamic queue identity
  // space. Ordered waits on one logical stream must remain on one sticky lane
  // instead of consuming a hardware queue for every call.
  constexpr int kWaitCount = 300;
  hipError_t wait_result = hipSuccess;
  int accepted_wait_count = 0;
  for (; accepted_wait_count < kWaitCount; ++accepted_wait_count) {
    wait_result = api_.wait_value_32(wait_stream, signal, 1,
                                     hipStreamWaitValueEq, UINT32_MAX);
    if (wait_result != hipSuccess) {
      break;
    }
  }

  *static_cast<volatile uint32_t*>(signal) = 1;
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(wait_stream));
  EXPECT_EQ(hipSuccess, wait_result)
      << "accepted " << accepted_wait_count << " of " << kWaitCount << " waits";
}

TEST_F(HipStreamValueApiTest, CompletedWaitLanesRecycleAcrossLiveStreams) {
  void* signal = AllocateSignal();
  ASSERT_NE(nullptr, signal);
  const uint32_t ready_value = 1;
  ASSERT_EQ(hipSuccess, api_.memcpy(signal, &ready_value, sizeof(ready_value),
                                    hipMemcpyHostToDevice));

  hipStream_t first_stream = CreateStream();
  ASSERT_NE(nullptr, first_stream);
  if (!CheckWaitSupport(first_stream, signal)) {
    GTEST_SKIP() << "stream memory waits are unsupported on this runner";
  }

  // Keep more logical streams alive than the hardware queue identity space.
  // Each satisfied wait completes before the next stream submits, so a
  // completion-aware implementation needs only one reusable wait lane.
  constexpr int kStreamCount = 300;
  for (int i = 0; i < kStreamCount; ++i) {
    hipStream_t stream = i == 0 ? first_stream : CreateStream();
    ASSERT_NE(nullptr, stream);
    ASSERT_EQ(hipSuccess, api_.wait_value_32(stream, signal, ready_value,
                                             hipStreamWaitValueEq, UINT32_MAX));
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream));
  }
}

TEST_F(HipStreamValueApiTest, CapturedWritesAreVisibleBatchNodes) {
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

  size_t node_count = 0;
  ASSERT_EQ(hipSuccess,
            api_.graph_get_nodes(graph, /*nodes=*/nullptr, &node_count));
  ASSERT_EQ(2u, node_count);
  std::vector<hipGraphNode_t> nodes(node_count);
  ASSERT_EQ(hipSuccess, api_.graph_get_nodes(graph, nodes.data(), &node_count));
  bool saw_value_32 = false;
  bool saw_value_64 = false;
  for (hipGraphNode_t node : nodes) {
    hipGraphNodeType type = hipGraphNodeTypeEmpty;
    ASSERT_EQ(hipSuccess, api_.graph_node_get_type(node, &type));
    EXPECT_EQ(hipGraphNodeTypeBatchMemOp, type);
    hipBatchMemOpNodeParams params = {};
    ASSERT_EQ(hipSuccess,
              api_.graph_batch_mem_op_node_get_params(node, &params));
    ASSERT_EQ(1u, params.count);
    ASSERT_NE(nullptr, params.paramArray);
    if (params.paramArray[0].operation == hipStreamMemOpWriteValue32) {
      EXPECT_EQ((hipDeviceptr_t)(uintptr_t)allocation,
                params.paramArray[0].writeValue.address);
      EXPECT_EQ(13u, params.paramArray[0].writeValue.value);
      saw_value_32 = true;
    } else if (params.paramArray[0].operation == hipStreamMemOpWriteValue64) {
      EXPECT_EQ(
          (hipDeviceptr_t)(uintptr_t)(static_cast<uint8_t*>(allocation) + 8),
          params.paramArray[0].writeValue.address);
      EXPECT_EQ(29u, params.paramArray[0].writeValue.value64);
      saw_value_64 = true;
    } else {
      ADD_FAILURE() << "unexpected captured operation "
                    << params.paramArray[0].operation;
    }
  }
  EXPECT_TRUE(saw_value_32);
  EXPECT_TRUE(saw_value_64);

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

TEST_F(HipStreamValueApiTest, CapturedWriteBatchReplaysAsOneNode) {
  hipStream_t stream = CreateStream();
  void* allocation = Allocate(8);
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, allocation);
  ASSERT_EQ(hipSuccess, api_.memset(allocation, 0, 8));

  hipStreamBatchMemOpParams operations[2] = {};
  operations[0].writeValue.operation = hipStreamMemOpWriteValue64;
  operations[0].writeValue.address = (hipDeviceptr_t)(uintptr_t)allocation;
  operations[0].writeValue.value64 = 5;
  operations[0].writeValue.flags = hipExtStreamWriteValueIncrement;
  operations[1].writeValue.operation = hipStreamMemOpWriteValue64;
  operations[1].writeValue.address = (hipDeviceptr_t)(uintptr_t)allocation;
  operations[1].writeValue.value64 = 2;
  operations[1].writeValue.flags = hipExtStreamWriteValueDecrement;

  ASSERT_EQ(hipSuccess,
            api_.stream_begin_capture(stream, hipStreamCaptureModeGlobal));
  ASSERT_EQ(hipSuccess, api_.batch_mem_op(stream, 2, operations, /*flags=*/0));
  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.stream_end_capture(stream, &graph));
  ASSERT_NE(nullptr, graph);
  graphs_.push_back(graph);

  size_t node_count = 1;
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_get_nodes(graph, &node, &node_count));
  ASSERT_EQ(1u, node_count);
  ASSERT_NE(nullptr, node);
  hipGraphNodeType type = hipGraphNodeTypeEmpty;
  ASSERT_EQ(hipSuccess, api_.graph_node_get_type(node, &type));
  EXPECT_EQ(hipGraphNodeTypeBatchMemOp, type);
  hipBatchMemOpNodeParams params = {};
  ASSERT_EQ(hipSuccess, api_.graph_batch_mem_op_node_get_params(node, &params));
  ASSERT_EQ(2u, params.count);
  ASSERT_NE(nullptr, params.paramArray);
  EXPECT_EQ(hipExtStreamWriteValueIncrement,
            params.paramArray[0].writeValue.flags);
  EXPECT_EQ(hipExtStreamWriteValueDecrement,
            params.paramArray[1].writeValue.flags);

  hipGraphExec_t executable = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_instantiate(&executable, graph, /*error_node=*/nullptr,
                                   /*log_buffer=*/nullptr, /*buffer_size=*/0));
  ASSERT_NE(nullptr, executable);
  graph_executables_.push_back(executable);
  for (int replay = 0; replay < 2; ++replay) {
    const uint64_t initial_value = 10;
    ASSERT_EQ(hipSuccess,
              api_.memcpy(allocation, &initial_value, sizeof(initial_value),
                          hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, api_.graph_launch(executable, stream));
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream));
    uint64_t observed = 0;
    ASSERT_EQ(hipSuccess, api_.memcpy(&observed, allocation, sizeof(observed),
                                      hipMemcpyDeviceToHost));
    EXPECT_EQ(13u, observed);
  }
}

TEST_F(HipStreamValueApiTest, DirectBatchNodeSupportsTemplateAndExecUpdates) {
  hipStream_t stream = CreateStream();
  void* allocation = Allocate(16);
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, allocation);

  hipCtx_t context = nullptr;
  ASSERT_EQ(hipSuccess, api_.ctx_get_current(&context));
  ASSERT_NE(nullptr, context);
  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
  ASSERT_NE(nullptr, graph);
  graphs_.push_back(graph);

  hipStreamBatchMemOpParams operations[2] = {};
  for (int i = 0; i < 2; ++i) {
    operations[i].writeValue.operation = hipStreamMemOpWriteValue64;
    operations[i].writeValue.address =
        (hipDeviceptr_t)(uintptr_t)(static_cast<uint8_t*>(allocation) +
                                    i * sizeof(uint64_t));
    operations[i].writeValue.value64 = 3 + i * 2;
    operations[i].writeValue.flags = hipStreamWriteValueDefault;
  }
  hipBatchMemOpNodeParams params = {
      .ctx = context,
      .count = 1,
      .paramArray = operations,
      .flags = 0,
  };
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_batch_mem_op_node(
                            &node, graph, /*dependencies=*/nullptr,
                            /*dependency_count=*/0, &params));
  ASSERT_NE(nullptr, node);

  operations[0].writeValue.value64 = 11;
  operations[1].writeValue.value64 = 13;
  params.count = 2;
  ASSERT_EQ(hipSuccess, api_.graph_batch_mem_op_node_set_params(node, &params));
  hipBatchMemOpNodeParams queried_params = {};
  ASSERT_EQ(hipSuccess,
            api_.graph_batch_mem_op_node_get_params(node, &queried_params));
  ASSERT_EQ(2u, queried_params.count);
  ASSERT_EQ(hipSuccess,
            api_.graph_batch_mem_op_node_set_params(node, &queried_params));
  hipGraphExec_t executable = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_instantiate(&executable, graph, /*error_node=*/nullptr,
                                   /*log_buffer=*/nullptr, /*buffer_size=*/0));
  ASSERT_NE(nullptr, executable);
  graph_executables_.push_back(executable);

  operations[0].writeValue.value64 = 17;
  operations[1].writeValue.value64 = 19;
  ASSERT_EQ(hipSuccess, api_.graph_exec_batch_mem_op_node_set_params(
                            executable, node, &params));

  hipBatchMemOpNodeParams observed_params = {};
  ASSERT_EQ(hipSuccess,
            api_.graph_batch_mem_op_node_get_params(node, &observed_params));
  ASSERT_EQ(2u, observed_params.count);
  ASSERT_NE(nullptr, observed_params.paramArray);
  EXPECT_EQ(11u, observed_params.paramArray[0].writeValue.value64);
  EXPECT_EQ(13u, observed_params.paramArray[1].writeValue.value64);

  ASSERT_EQ(hipSuccess, api_.memset(allocation, 0, 16));
  ASSERT_EQ(hipSuccess, api_.graph_launch(executable, stream));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream));
  uint64_t observed[2] = {};
  ASSERT_EQ(hipSuccess, api_.memcpy(observed, allocation, sizeof(observed),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(17u, observed[0]);
  EXPECT_EQ(19u, observed[1]);

  hipGraphExec_t template_executable = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_instantiate(&template_executable, graph,
                                   /*error_node=*/nullptr,
                                   /*log_buffer=*/nullptr, /*buffer_size=*/0));
  ASSERT_NE(nullptr, template_executable);
  graph_executables_.push_back(template_executable);
  ASSERT_EQ(hipSuccess, api_.memset(allocation, 0, 16));
  ASSERT_EQ(hipSuccess, api_.graph_launch(template_executable, stream));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream));
  ASSERT_EQ(hipSuccess, api_.memcpy(observed, allocation, sizeof(observed),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(11u, observed[0]);
  EXPECT_EQ(13u, observed[1]);
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

TEST_F(HipStreamValueApiTest, WaitInBatchInvalidatesCaptureWithoutWrites) {
  hipStream_t stream = CreateStream();
  void* allocation = Allocate(8);
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, allocation);
  ASSERT_EQ(hipSuccess, api_.memset(allocation, 0, 8));

  hipStreamBatchMemOpParams operations[2] = {};
  operations[0].writeValue.operation = hipStreamMemOpWriteValue32;
  operations[0].writeValue.address = (hipDeviceptr_t)(uintptr_t)allocation;
  operations[0].writeValue.value = 47;
  operations[0].writeValue.flags = hipStreamWriteValueDefault;
  operations[1].waitValue.operation = hipStreamMemOpWaitValue32;
  operations[1].waitValue.address = (hipDeviceptr_t)(uintptr_t)allocation;
  operations[1].waitValue.value = 47;
  operations[1].waitValue.flags = hipStreamWaitValueEq;

  ASSERT_EQ(hipSuccess,
            api_.stream_begin_capture(stream, hipStreamCaptureModeGlobal));
  EXPECT_EQ(hipErrorStreamCaptureUnsupported,
            api_.batch_mem_op(stream, 2, operations, /*flags=*/0));
  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
  EXPECT_EQ(hipSuccess, api_.stream_is_capturing(stream, &capture_status));
  EXPECT_EQ(hipStreamCaptureStatusInvalidated, capture_status);

  hipGraph_t graph = nullptr;
  EXPECT_EQ(hipErrorStreamCaptureInvalidated,
            api_.stream_end_capture(stream, &graph));
  EXPECT_EQ(nullptr, graph);

  uint32_t observed = 1;
  ASSERT_EQ(hipSuccess, api_.memcpy(&observed, allocation, sizeof(observed),
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(0u, observed);
}

TEST_F(HipStreamValueApiTest, CaptureBeginRacesValueWaitSubmission) {
  hipStream_t stream = CreateStream();
  void* allocation = AllocateSignal();
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, allocation);
  if (!CheckWaitSupport(stream, allocation)) {
    GTEST_SKIP() << "stream memory waits are unsupported on this runner";
  }

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
  void* allocation = AllocateSignal();
  ASSERT_NE(nullptr, stream);
  ASSERT_NE(nullptr, allocation);
  ASSERT_EQ(hipSuccess, api_.memset(allocation, 0, sizeof(uint32_t)));
  if (!CheckWaitSupport(stream, allocation)) {
    GTEST_SKIP() << "stream memory waits are unsupported on this runner";
  }

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
  void* allocation = AllocateSignal();
  ASSERT_NE(nullptr, wait_stream);
  ASSERT_NE(nullptr, producer_stream);
  ASSERT_NE(nullptr, allocation);
  ASSERT_EQ(hipSuccess, api_.memset(allocation, 0, sizeof(uint32_t)));
  if (!CheckWaitSupport(wait_stream, allocation)) {
    GTEST_SKIP() << "stream memory waits are unsupported on this runner";
  }
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
