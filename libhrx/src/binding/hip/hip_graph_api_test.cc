// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <cstdint>
#include <thread>

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/testing/gtest.h"

namespace {

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipHalDeinitFn = hipError_t (*)(void);
using HipGetDeviceFn = hipError_t (*)(int* device);
using HipDeviceSynchronizeFn = hipError_t (*)(void);
using HipStreamCreateFn = hipError_t (*)(hipStream_t* stream);
using HipStreamDestroyFn = hipError_t (*)(hipStream_t stream);
using HipStreamSynchronizeFn = hipError_t (*)(hipStream_t stream);
using HipStreamBeginCaptureToGraphFn = hipError_t (*)(
    hipStream_t stream, hipGraph_t graph, const hipGraphNode_t* dependencies,
    const void* dependency_data, size_t dependency_count,
    hipStreamCaptureMode mode);
using HipGraphCreateFn = hipError_t (*)(hipGraph_t* graph, unsigned int flags);
using HipGraphDestroyFn = hipError_t (*)(hipGraph_t graph);
using HipGraphAddEmptyNodeFn =
    hipError_t (*)(hipGraphNode_t* node, hipGraph_t graph,
                   const hipGraphNode_t* dependencies, size_t dependency_count);
using HipGraphAddHostNodeFn = hipError_t (*)(hipGraphNode_t* node,
                                             hipGraph_t graph,
                                             const hipGraphNode_t* dependencies,
                                             size_t dependency_count,
                                             const void* parameters);
using HipGraphAddMemcpyNode1DFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, void* destination, const void* source, size_t size,
    hipMemcpyKind kind);
using HipGraphAddMemsetNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, const void* parameters);
using HipGraphAddChildGraphNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, hipGraph_t child_graph);
using HipGraphAddMemAllocNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, void* allocation_parameters);
using HipGraphAddMemFreeNodeFn = hipError_t (*)(
    hipGraphNode_t* node, hipGraph_t graph, const hipGraphNode_t* dependencies,
    size_t dependency_count, void* device_pointer);
using HipGraphAddDependenciesFn = hipError_t (*)(hipGraph_t graph,
                                                 const hipGraphNode_t* from,
                                                 const hipGraphNode_t* to,
                                                 size_t dependency_count);
using HipGraphGetNodesFn = hipError_t (*)(hipGraph_t graph,
                                          hipGraphNode_t* nodes,
                                          size_t* node_count);
using HipGraphNodeGetTypeFn = hipError_t (*)(hipGraphNode_t node,
                                             hipGraphNodeType* type);
using HipGraphInstantiateFn = hipError_t (*)(hipGraphExec_t* executable_graph,
                                             hipGraph_t graph,
                                             hipGraphNode_t* error_node,
                                             char* log_buffer,
                                             size_t log_buffer_size);
using HipGraphExecDestroyFn = hipError_t (*)(hipGraphExec_t executable_graph);
using HipGraphExecChildGraphNodeSetParamsFn =
    hipError_t (*)(hipGraphExec_t executable_graph, hipGraphNode_t node,
                   hipGraph_t child_graph);
using HipGraphExecMemcpyNodeSetParams1DFn = hipError_t (*)(
    hipGraphExec_t executable_graph, hipGraphNode_t node, void* destination,
    const void* source, size_t size, hipMemcpyKind kind);
using HipGraphExecMemsetNodeSetParamsFn =
    hipError_t (*)(hipGraphExec_t executable_graph, hipGraphNode_t node,
                   const void* parameters);
using HipGraphExecUpdateFn = hipError_t (*)(
    hipGraphExec_t executable_graph, hipGraph_t graph,
    hipGraphNode_t* error_node, hipGraphExecUpdateResult* update_result);
using HipGraphLaunchFn = hipError_t (*)(hipGraphExec_t executable_graph,
                                        hipStream_t stream);
using HipGraphDestroyNodeFn = hipError_t (*)(hipGraphNode_t node);
using HipMallocFn = hipError_t (*)(hipDeviceptr_t* pointer, size_t size);
using HipFreeFn = hipError_t (*)(void* pointer);
using HipFreeAsyncFn = hipError_t (*)(void* pointer, hipStream_t stream);
using HipMemcpyFn = hipError_t (*)(void* destination, const void* source,
                                   size_t size, hipMemcpyKind kind);

struct HipRuntimeApi {
  // Initializes the runtime under test.
  HipInitFn init = nullptr;
  // Deinitializes the runtime under test.
  HipHalDeinitFn hal_deinit = nullptr;
  // Queries the current device ordinal.
  HipGetDeviceFn get_device = nullptr;
  // Waits for all work in the current device context.
  HipDeviceSynchronizeFn device_synchronize = nullptr;
  // Creates the per-test stream.
  HipStreamCreateFn stream_create = nullptr;
  // Destroys the per-test stream.
  HipStreamDestroyFn stream_destroy = nullptr;
  // Waits for per-test stream work.
  HipStreamSynchronizeFn stream_synchronize = nullptr;
  // Begins capture into a supplied graph.
  HipStreamBeginCaptureToGraphFn stream_begin_capture_to_graph = nullptr;
  // Creates a public graph.
  HipGraphCreateFn graph_create = nullptr;
  // Destroys a public graph.
  HipGraphDestroyFn graph_destroy = nullptr;
  // Adds an empty graph node.
  HipGraphAddEmptyNodeFn graph_add_empty_node = nullptr;
  // Adds a host-call graph node.
  HipGraphAddHostNodeFn graph_add_host_node = nullptr;
  // Adds a one-dimensional memory-copy node.
  HipGraphAddMemcpyNode1DFn graph_add_memcpy_node_1d = nullptr;
  // Adds a memory-set node.
  HipGraphAddMemsetNodeFn graph_add_memset_node = nullptr;
  // Adds a child-graph node.
  HipGraphAddChildGraphNodeFn graph_add_child_graph_node = nullptr;
  // Adds a graph-memory allocation node.
  HipGraphAddMemAllocNodeFn graph_add_mem_alloc_node = nullptr;
  // Adds a graph-memory free node.
  HipGraphAddMemFreeNodeFn graph_add_mem_free_node = nullptr;
  // Adds graph dependency edges.
  HipGraphAddDependenciesFn graph_add_dependencies = nullptr;
  // Enumerates graph nodes.
  HipGraphGetNodesFn graph_get_nodes = nullptr;
  // Queries a graph node's type.
  HipGraphNodeGetTypeFn graph_node_get_type = nullptr;
  // Instantiates an executable graph.
  HipGraphInstantiateFn graph_instantiate = nullptr;
  // Destroys an executable graph.
  HipGraphExecDestroyFn graph_exec_destroy = nullptr;
  // Replaces an executable child graph.
  HipGraphExecChildGraphNodeSetParamsFn graph_exec_child_graph_node_set_params =
      nullptr;
  // Updates one executable memory-copy node.
  HipGraphExecMemcpyNodeSetParams1DFn graph_exec_memcpy_node_set_params_1d =
      nullptr;
  // Updates one executable memory-set node.
  HipGraphExecMemsetNodeSetParamsFn graph_exec_memset_node_set_params = nullptr;
  // Updates an executable from a source graph.
  HipGraphExecUpdateFn graph_exec_update = nullptr;
  // Launches an executable graph.
  HipGraphLaunchFn graph_launch = nullptr;
  // Removes a node from its source graph.
  HipGraphDestroyNodeFn graph_destroy_node = nullptr;
  // Allocates device memory.
  HipMallocFn malloc = nullptr;
  // Frees a device allocation synchronously.
  HipFreeFn free = nullptr;
  // Enqueues a device-allocation free.
  HipFreeAsyncFn free_async = nullptr;
  // Copies bytes between host and device memory.
  HipMemcpyFn memcpy = nullptr;
};

class HipGraphApiTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!dso_.is_open()) {
      ASSERT_TRUE(dso_.Open()) << dso_.error();
      api_.init = dso_.Resolve<HipInitFn>("hipInit");
      api_.hal_deinit = dso_.Resolve<HipHalDeinitFn>("hipHALDeinit");
      api_.get_device = dso_.Resolve<HipGetDeviceFn>("hipGetDevice");
      api_.device_synchronize =
          dso_.Resolve<HipDeviceSynchronizeFn>("hipDeviceSynchronize");
      api_.stream_create = dso_.Resolve<HipStreamCreateFn>("hipStreamCreate");
      api_.stream_destroy =
          dso_.Resolve<HipStreamDestroyFn>("hipStreamDestroy");
      api_.stream_synchronize =
          dso_.Resolve<HipStreamSynchronizeFn>("hipStreamSynchronize");
      api_.stream_begin_capture_to_graph =
          dso_.Resolve<HipStreamBeginCaptureToGraphFn>(
              "hipStreamBeginCaptureToGraph");
      api_.graph_create = dso_.Resolve<HipGraphCreateFn>("hipGraphCreate");
      api_.graph_destroy = dso_.Resolve<HipGraphDestroyFn>("hipGraphDestroy");
      api_.graph_add_empty_node =
          dso_.Resolve<HipGraphAddEmptyNodeFn>("hipGraphAddEmptyNode");
      api_.graph_add_host_node =
          dso_.Resolve<HipGraphAddHostNodeFn>("hipGraphAddHostNode");
      api_.graph_add_memcpy_node_1d =
          dso_.Resolve<HipGraphAddMemcpyNode1DFn>("hipGraphAddMemcpyNode1D");
      api_.graph_add_memset_node =
          dso_.Resolve<HipGraphAddMemsetNodeFn>("hipGraphAddMemsetNode");
      api_.graph_add_child_graph_node =
          dso_.Resolve<HipGraphAddChildGraphNodeFn>(
              "hipGraphAddChildGraphNode");
      api_.graph_add_mem_alloc_node =
          dso_.Resolve<HipGraphAddMemAllocNodeFn>("hipGraphAddMemAllocNode");
      api_.graph_add_mem_free_node =
          dso_.Resolve<HipGraphAddMemFreeNodeFn>("hipGraphAddMemFreeNode");
      api_.graph_add_dependencies =
          dso_.Resolve<HipGraphAddDependenciesFn>("hipGraphAddDependencies");
      api_.graph_get_nodes =
          dso_.Resolve<HipGraphGetNodesFn>("hipGraphGetNodes");
      api_.graph_node_get_type =
          dso_.Resolve<HipGraphNodeGetTypeFn>("hipGraphNodeGetType");
      api_.graph_instantiate =
          dso_.Resolve<HipGraphInstantiateFn>("hipGraphInstantiate");
      api_.graph_exec_destroy =
          dso_.Resolve<HipGraphExecDestroyFn>("hipGraphExecDestroy");
      api_.graph_exec_child_graph_node_set_params =
          dso_.Resolve<HipGraphExecChildGraphNodeSetParamsFn>(
              "hipGraphExecChildGraphNodeSetParams");
      api_.graph_exec_memcpy_node_set_params_1d =
          dso_.Resolve<HipGraphExecMemcpyNodeSetParams1DFn>(
              "hipGraphExecMemcpyNodeSetParams1D");
      api_.graph_exec_memset_node_set_params =
          dso_.Resolve<HipGraphExecMemsetNodeSetParamsFn>(
              "hipGraphExecMemsetNodeSetParams");
      api_.graph_exec_update =
          dso_.Resolve<HipGraphExecUpdateFn>("hipGraphExecUpdate");
      api_.graph_launch = dso_.Resolve<HipGraphLaunchFn>("hipGraphLaunch");
      api_.graph_destroy_node =
          dso_.Resolve<HipGraphDestroyNodeFn>("hipGraphDestroyNode");
      api_.malloc = dso_.Resolve<HipMallocFn>("hipMalloc");
      api_.free = dso_.Resolve<HipFreeFn>("hipFree");
      api_.free_async = dso_.Resolve<HipFreeAsyncFn>("hipFreeAsync");
      api_.memcpy = dso_.Resolve<HipMemcpyFn>("hipMemcpy");
    }

    ASSERT_NE(nullptr, api_.init);
    ASSERT_NE(nullptr, api_.hal_deinit);
    ASSERT_NE(nullptr, api_.get_device);
    ASSERT_NE(nullptr, api_.device_synchronize);
    ASSERT_NE(nullptr, api_.stream_create);
    ASSERT_NE(nullptr, api_.stream_destroy);
    ASSERT_NE(nullptr, api_.stream_synchronize);
    ASSERT_NE(nullptr, api_.stream_begin_capture_to_graph);
    ASSERT_NE(nullptr, api_.graph_create);
    ASSERT_NE(nullptr, api_.graph_destroy);
    ASSERT_NE(nullptr, api_.graph_add_empty_node);
    ASSERT_NE(nullptr, api_.graph_add_host_node);
    ASSERT_NE(nullptr, api_.graph_add_memcpy_node_1d);
    ASSERT_NE(nullptr, api_.graph_add_memset_node);
    ASSERT_NE(nullptr, api_.graph_add_child_graph_node);
    ASSERT_NE(nullptr, api_.graph_add_mem_alloc_node);
    ASSERT_NE(nullptr, api_.graph_add_mem_free_node);
    ASSERT_NE(nullptr, api_.graph_add_dependencies);
    ASSERT_NE(nullptr, api_.graph_get_nodes);
    ASSERT_NE(nullptr, api_.graph_node_get_type);
    ASSERT_NE(nullptr, api_.graph_instantiate);
    ASSERT_NE(nullptr, api_.graph_exec_destroy);
    ASSERT_NE(nullptr, api_.graph_exec_child_graph_node_set_params);
    ASSERT_NE(nullptr, api_.graph_exec_memcpy_node_set_params_1d);
    ASSERT_NE(nullptr, api_.graph_exec_memset_node_set_params);
    ASSERT_NE(nullptr, api_.graph_exec_update);
    ASSERT_NE(nullptr, api_.graph_launch);
    ASSERT_NE(nullptr, api_.graph_destroy_node);
    ASSERT_NE(nullptr, api_.malloc);
    ASSERT_NE(nullptr, api_.free);
    ASSERT_NE(nullptr, api_.free_async);
    ASSERT_NE(nullptr, api_.memcpy);
    ASSERT_EQ(hipSuccess, api_.init(/*flags=*/0));
    ASSERT_EQ(hipSuccess, api_.get_device(&device_));
    ASSERT_EQ(hipSuccess, api_.stream_create(&stream_));
  }

  void TearDown() override {
    if (stream_) {
      EXPECT_EQ(hipSuccess, api_.stream_destroy(stream_));
      stream_ = nullptr;
    }
  }

  static void TearDownTestSuite() {
    if (!dso_.is_open()) {
      return;
    }
    ASSERT_NE(nullptr, api_.hal_deinit);
    EXPECT_EQ(hipSuccess, api_.hal_deinit());
    api_ = {};
    EXPECT_TRUE(dso_.Close()) << dso_.error();
  }

  // Exact built HIP DSO loaded by the suite.
  static hrx::hip::testing::HipDso dso_;
  // Resolved function table shared across suite cases.
  static HipRuntimeApi api_;
  // Stream created and destroyed around each case.
  hipStream_t stream_ = nullptr;
  // Device ordinal used in graph-allocation parameters.
  int device_ = -1;
};

hrx::hip::testing::HipDso HipGraphApiTest::dso_;
HipRuntimeApi HipGraphApiTest::api_;

void NoopHostFunction(void* user_data) { (void)user_data; }

TEST_F(HipGraphApiTest, RejectsFabricatedAndStaleGraphNodeHandles) {
  const hipGraph_t fabricated_graph =
      reinterpret_cast<hipGraph_t>(uintptr_t{1});
  const hipGraphNode_t fabricated_node =
      reinterpret_cast<hipGraphNode_t>(uintptr_t{1});

  size_t node_count = 0;
  EXPECT_EQ(
      hipErrorInvalidValue,
      api_.graph_get_nodes(fabricated_graph, /*nodes=*/nullptr, &node_count));
  hipGraphNode_t output_node = nullptr;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.graph_add_empty_node(&output_node, fabricated_graph,
                                      /*dependencies=*/nullptr,
                                      /*dependency_count=*/0));
  hipGraphNodeType node_type = hipGraphNodeTypeCount;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.graph_node_get_type(fabricated_node, &node_type));

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
  const hipGraphNode_t fabricated_dependencies[] = {fabricated_node};
  EXPECT_EQ(
      hipErrorInvalidValue,
      api_.graph_add_empty_node(&output_node, graph, fabricated_dependencies,
                                /*dependency_count=*/1));

  hipGraphNode_t live_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_empty_node(&live_node, graph,
                                                  /*dependencies=*/nullptr,
                                                  /*dependency_count=*/0));
  const hipGraphNode_t dependency_targets[] = {live_node};
  EXPECT_EQ(hipErrorInvalidValue,
            api_.graph_add_dependencies(graph, fabricated_dependencies,
                                        dependency_targets,
                                        /*dependency_count=*/1));

  hipGraphExec_t executable_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_instantiate(&executable_graph, graph,
                                               /*error_node=*/nullptr,
                                               /*log_buffer=*/nullptr,
                                               /*log_buffer_size=*/0));
  const hipGraph_t stale_graph = graph;
  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph));
  graph = nullptr;

  // Executable ownership keeps the backing arena allocated while graph
  // destruction invalidates the public source graph and node handles.
  EXPECT_EQ(hipErrorInvalidValue,
            api_.graph_get_nodes(stale_graph, /*nodes=*/nullptr, &node_count));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.graph_node_get_type(live_node, &node_type));
  EXPECT_EQ(hipSuccess, api_.graph_exec_destroy(executable_graph));
  EXPECT_EQ(hipSuccess, api_.stream_synchronize(stream_));
}

TEST_F(HipGraphApiTest, RejectsDuplicateCreationDependenciesWithoutMutation) {
  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
  hipGraphNode_t first = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_empty_node(&first, graph,
                                                  /*dependencies=*/nullptr,
                                                  /*dependency_count=*/0));
  const hipGraphNode_t dependencies[] = {first, first};
  hipGraphNode_t duplicate_node = nullptr;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.graph_add_empty_node(&duplicate_node, graph, dependencies,
                                      /*dependency_count=*/2));

  size_t node_count = 0;
  EXPECT_EQ(hipSuccess,
            api_.graph_get_nodes(graph, /*nodes=*/nullptr, &node_count));
  EXPECT_EQ(1u, node_count);
  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
}

TEST_F(HipGraphApiTest, FindsNodesInReachableChildrenOnlyWhileRootIsLive) {
  hipGraph_t child_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&child_graph, /*flags=*/0));
  hipGraphNode_t child_empty_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_empty_node(&child_empty_node, child_graph,
                                      /*dependencies=*/nullptr,
                                      /*dependency_count=*/0));

  hipGraph_t parent_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&parent_graph, /*flags=*/0));
  hipGraphNode_t child_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_child_graph_node(
                            &child_node, parent_graph,
                            /*dependencies=*/nullptr,
                            /*dependency_count=*/0, child_graph));

  ASSERT_EQ(hipSuccess, api_.graph_destroy(child_graph));
  child_graph = nullptr;
  hipGraphNodeType node_type = hipGraphNodeTypeCount;
  EXPECT_EQ(hipSuccess, api_.graph_node_get_type(child_empty_node, &node_type));
  EXPECT_EQ(hipGraphNodeTypeEmpty, node_type);

  hipGraphExec_t executable_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_instantiate(&executable_graph, parent_graph,
                                               /*error_node=*/nullptr,
                                               /*log_buffer=*/nullptr,
                                               /*log_buffer_size=*/0));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(parent_graph));
  parent_graph = nullptr;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.graph_node_get_type(child_empty_node, &node_type));
  EXPECT_EQ(hipSuccess, api_.graph_exec_destroy(executable_graph));
  EXPECT_EQ(hipSuccess, api_.stream_synchronize(stream_));
}

TEST_F(HipGraphApiTest,
       BeginCaptureToGraphValidatesDependencyDataAndArgumentShape) {
  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_empty_node(&node, graph,
                                                  /*dependencies=*/nullptr,
                                                  /*dependency_count=*/0));

  const hipGraphNode_t dependencies[] = {node};
  const int dependency_data = 0;
  EXPECT_EQ(hipErrorNotSupported,
            api_.stream_begin_capture_to_graph(
                stream_, graph, dependencies, &dependency_data,
                /*dependency_count=*/1, hipStreamCaptureModeGlobal));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.stream_begin_capture_to_graph(
                stream_, /*graph=*/nullptr, /*dependencies=*/nullptr,
                /*dependency_data=*/nullptr, /*dependency_count=*/0,
                hipStreamCaptureModeGlobal));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.stream_begin_capture_to_graph(
                stream_, graph, /*dependencies=*/nullptr,
                /*dependency_data=*/nullptr, /*dependency_count=*/1,
                hipStreamCaptureModeGlobal));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.stream_begin_capture_to_graph(
                stream_, graph, dependencies, /*dependency_data=*/nullptr,
                /*dependency_count=*/0, hipStreamCaptureModeGlobal));

  const hipGraphNode_t fabricated_dependencies[] = {
      reinterpret_cast<hipGraphNode_t>(uintptr_t{1})};
  EXPECT_EQ(hipErrorInvalidValue,
            api_.stream_begin_capture_to_graph(
                stream_, graph, fabricated_dependencies,
                /*dependency_data=*/nullptr, /*dependency_count=*/1,
                hipStreamCaptureModeGlobal));
  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
}

TEST_F(HipGraphApiTest, GraphFreeRacesExternalFreeWithoutLosingOwnership) {
  constexpr int kIterationsPerFreeKind = 8;
  for (bool use_async_free : {false, true}) {
    for (int iteration = 0; iteration < kIterationsPerFreeKind; ++iteration) {
      SCOPED_TRACE(testing::Message() << "use_async_free=" << use_async_free
                                      << " iteration=" << iteration);

      hipGraph_t allocation_graph = nullptr;
      ASSERT_EQ(hipSuccess, api_.graph_create(&allocation_graph, /*flags=*/0));
      hipMemAllocNodeParams parameters = {};
      parameters.poolProps.allocType = hipMemAllocationTypePinned;
      parameters.poolProps.location.type = hipMemLocationTypeDevice;
      parameters.poolProps.location.id = device_;
      parameters.bytesize = 4096;
      hipGraphNode_t allocation_node = nullptr;
      ASSERT_EQ(hipSuccess, api_.graph_add_mem_alloc_node(
                                &allocation_node, allocation_graph,
                                /*dependencies=*/nullptr,
                                /*dependency_count=*/0, &parameters));
      ASSERT_NE(nullptr, parameters.dptr);

      hipGraphExec_t executable_graph = nullptr;
      ASSERT_EQ(hipSuccess,
                api_.graph_instantiate(&executable_graph, allocation_graph,
                                       /*error_node=*/nullptr,
                                       /*log_buffer=*/nullptr,
                                       /*log_buffer_size=*/0));
      ASSERT_EQ(hipSuccess, api_.graph_launch(executable_graph, stream_));
      ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));

      hipGraph_t free_graph = nullptr;
      ASSERT_EQ(hipSuccess, api_.graph_create(&free_graph, /*flags=*/0));
      hipGraphNode_t free_node = nullptr;
      std::atomic<bool> start{false};
      std::atomic<hipError_t> graph_free_result{hipErrorUnknown};
      std::atomic<hipError_t> external_free_result{hipErrorUnknown};
      std::thread graph_free_thread([&] {
        while (!start.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        graph_free_result.store(
            api_.graph_add_mem_free_node(
                &free_node, free_graph, /*dependencies=*/nullptr,
                /*dependency_count=*/0, parameters.dptr),
            std::memory_order_release);
      });
      std::thread external_free_thread([&] {
        while (!start.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        const hipError_t result =
            use_async_free ? api_.free_async(parameters.dptr, stream_)
                           : api_.free(parameters.dptr);
        external_free_result.store(result, std::memory_order_release);
      });
      start.store(true, std::memory_order_release);
      graph_free_thread.join();
      external_free_thread.join();

      const hipError_t graph_result =
          graph_free_result.load(std::memory_order_acquire);
      const hipError_t free_result =
          external_free_result.load(std::memory_order_acquire);
      EXPECT_TRUE(
          (graph_result == hipSuccess && free_result == hipErrorInvalidValue) ||
          (graph_result == hipErrorInvalidValue && free_result == hipSuccess));

      if (use_async_free && free_result == hipSuccess) {
        EXPECT_EQ(hipSuccess, api_.stream_synchronize(stream_));
      }
      EXPECT_EQ(hipSuccess, api_.graph_destroy(free_graph));
      if (graph_result == hipSuccess) {
        // Destroying an unlaunched free-node graph returns its claim, leaving
        // the published pointer available to the ordinary free path.
        EXPECT_EQ(hipSuccess, api_.free(parameters.dptr));
      } else if (free_result != hipSuccess) {
        // Keep an unexpected arbitration result from leaking into later
        // iterations while preserving the primary assertion above.
        EXPECT_EQ(hipSuccess, api_.free(parameters.dptr));
      }

      EXPECT_EQ(hipSuccess, api_.graph_exec_destroy(executable_graph));
      EXPECT_EQ(hipSuccess, api_.graph_destroy(allocation_graph));
      EXPECT_EQ(hipSuccess, api_.stream_synchronize(stream_));
    }
  }
}

TEST_F(HipGraphApiTest, FreeNodeCreationRacesAllocationGraphDestructionSafely) {
  constexpr int kIterationCount = 32;
  for (int iteration = 0; iteration < kIterationCount; ++iteration) {
    SCOPED_TRACE(testing::Message() << "iteration=" << iteration);

    hipGraph_t allocation_graph = nullptr;
    ASSERT_EQ(hipSuccess, api_.graph_create(&allocation_graph, /*flags=*/0));
    hipMemAllocNodeParams parameters = {};
    parameters.poolProps.allocType = hipMemAllocationTypePinned;
    parameters.poolProps.location.type = hipMemLocationTypeDevice;
    parameters.poolProps.location.id = device_;
    parameters.bytesize = 4096;
    hipGraphNode_t allocation_node = nullptr;
    ASSERT_EQ(hipSuccess, api_.graph_add_mem_alloc_node(
                              &allocation_node, allocation_graph,
                              /*dependencies=*/nullptr,
                              /*dependency_count=*/0, &parameters));

    hipGraph_t free_graph = nullptr;
    ASSERT_EQ(hipSuccess, api_.graph_create(&free_graph, /*flags=*/0));
    hipGraphNode_t free_node = nullptr;
    std::atomic<bool> start{false};
    std::atomic<hipError_t> add_result{hipErrorUnknown};
    std::atomic<hipError_t> destroy_result{hipErrorUnknown};
    std::thread add_thread([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      add_result.store(api_.graph_add_mem_free_node(
                           &free_node, free_graph, /*dependencies=*/nullptr,
                           /*dependency_count=*/0, parameters.dptr),
                       std::memory_order_release);
    });
    std::thread destroy_thread([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      destroy_result.store(api_.graph_destroy(allocation_graph),
                           std::memory_order_release);
    });
    start.store(true, std::memory_order_release);
    add_thread.join();
    destroy_thread.join();
    allocation_graph = nullptr;

    EXPECT_EQ(hipSuccess, destroy_result.load(std::memory_order_acquire));
    const hipError_t observed_add_result =
        add_result.load(std::memory_order_acquire);
    EXPECT_TRUE(observed_add_result == hipSuccess ||
                observed_add_result == hipErrorInvalidValue);
    EXPECT_EQ(hipSuccess, api_.graph_destroy(free_graph));
  }
}

TEST_F(HipGraphApiTest,
       ExternalFreeRacesUnexecutedAllocationGraphDestructionSafely) {
  constexpr int kIterationsPerFreeKind = 32;
  for (bool use_async_free : {false, true}) {
    for (int iteration = 0; iteration < kIterationsPerFreeKind; ++iteration) {
      SCOPED_TRACE(testing::Message() << "use_async_free=" << use_async_free
                                      << " iteration=" << iteration);

      hipGraph_t allocation_graph = nullptr;
      ASSERT_EQ(hipSuccess, api_.graph_create(&allocation_graph, /*flags=*/0));
      hipMemAllocNodeParams parameters = {};
      parameters.poolProps.allocType = hipMemAllocationTypePinned;
      parameters.poolProps.location.type = hipMemLocationTypeDevice;
      parameters.poolProps.location.id = device_;
      parameters.bytesize = 4096;
      hipGraphNode_t allocation_node = nullptr;
      ASSERT_EQ(hipSuccess, api_.graph_add_mem_alloc_node(
                                &allocation_node, allocation_graph,
                                /*dependencies=*/nullptr,
                                /*dependency_count=*/0, &parameters));
      ASSERT_NE(nullptr, parameters.dptr);

      std::atomic<bool> start{false};
      std::atomic<hipError_t> free_result{hipErrorUnknown};
      std::atomic<hipError_t> destroy_result{hipErrorUnknown};
      std::thread free_thread([&] {
        while (!start.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        free_result.store(use_async_free
                              ? api_.free_async(parameters.dptr, stream_)
                              : api_.free(parameters.dptr),
                          std::memory_order_release);
      });
      std::thread destroy_thread([&] {
        while (!start.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        destroy_result.store(api_.graph_destroy(allocation_graph),
                             std::memory_order_release);
      });
      start.store(true, std::memory_order_release);
      free_thread.join();
      destroy_thread.join();
      allocation_graph = nullptr;

      EXPECT_EQ(hipSuccess, destroy_result.load(std::memory_order_acquire));
      const hipError_t observed_free_result =
          free_result.load(std::memory_order_acquire);
      if (use_async_free) {
        EXPECT_TRUE(observed_free_result == hipSuccess ||
                    observed_free_result == hipErrorInvalidValue);
        if (observed_free_result == hipSuccess) {
          EXPECT_EQ(hipSuccess, api_.stream_synchronize(stream_));
        }
      } else {
        // Synchronous free requires an executed allocation. It loses whether
        // it reaches the live unexecuted pointer or graph destruction first.
        EXPECT_EQ(hipErrorInvalidValue, observed_free_result);
      }
    }
  }
}

TEST_F(HipGraphApiTest,
       GraphMemoryExecRetiresAfterItsPublicLaunchStreamIsDestroyed) {
  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
  hipMemAllocNodeParams parameters = {};
  parameters.poolProps.allocType = hipMemAllocationTypePinned;
  parameters.poolProps.location.type = hipMemLocationTypeDevice;
  parameters.poolProps.location.id = device_;
  parameters.bytesize = 4096;
  hipGraphNode_t allocation_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_mem_alloc_node(&allocation_node, graph,
                                          /*dependencies=*/nullptr,
                                          /*dependency_count=*/0, &parameters));

  hipGraphExec_t executable_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_instantiate(&executable_graph, graph,
                                               /*error_node=*/nullptr,
                                               /*log_buffer=*/nullptr,
                                               /*log_buffer_size=*/0));
  hipStream_t launch_stream = nullptr;
  ASSERT_EQ(hipSuccess, api_.stream_create(&launch_stream));
  ASSERT_EQ(hipSuccess, api_.graph_launch(executable_graph, launch_stream));

  ASSERT_EQ(hipSuccess, api_.stream_destroy(launch_stream));
  launch_stream = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_exec_destroy(executable_graph));
  executable_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.device_synchronize());

  EXPECT_EQ(hipSuccess, api_.free(parameters.dptr));
  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
}

TEST_F(HipGraphApiTest,
       RejectsGraphMemoryChildTopologyAtAllMutationBoundaries) {
  // Existing graph-memory ownership cannot be embedded through the legacy
  // child API because that API has no ownership-transfer parameter.
  hipGraph_t memory_child = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&memory_child, /*flags=*/0));
  hipMemAllocNodeParams direct_parameters = {};
  direct_parameters.poolProps.allocType = hipMemAllocationTypePinned;
  direct_parameters.poolProps.location.type = hipMemLocationTypeDevice;
  direct_parameters.poolProps.location.id = device_;
  direct_parameters.bytesize = 4096;
  hipGraphNode_t direct_allocation_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_mem_alloc_node(
                            &direct_allocation_node, memory_child,
                            /*dependencies=*/nullptr,
                            /*dependency_count=*/0, &direct_parameters));
  hipGraph_t direct_parent = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&direct_parent, /*flags=*/0));
  hipGraphNode_t rejected_child_node = nullptr;
  EXPECT_EQ(hipErrorNotSupported, api_.graph_add_child_graph_node(
                                      &rejected_child_node, direct_parent,
                                      /*dependencies=*/nullptr,
                                      /*dependency_count=*/0, memory_child));
  EXPECT_EQ(nullptr, rejected_child_node);
  EXPECT_EQ(hipSuccess, api_.graph_destroy(direct_parent));
  EXPECT_EQ(hipSuccess, api_.graph_destroy(memory_child));

  // Revalidate recursively at instantiation: the leaf gains its allocation
  // only after both child relationships have been accepted.
  hipGraph_t leaf = nullptr;
  hipGraph_t middle = nullptr;
  hipGraph_t root = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&leaf, /*flags=*/0));
  ASSERT_EQ(hipSuccess, api_.graph_create(&middle, /*flags=*/0));
  ASSERT_EQ(hipSuccess, api_.graph_create(&root, /*flags=*/0));
  hipGraphNode_t leaf_child_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_child_graph_node(&leaf_child_node, middle,
                                            /*dependencies=*/nullptr,
                                            /*dependency_count=*/0, leaf));
  hipGraphNode_t middle_child_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_child_graph_node(&middle_child_node, root,
                                            /*dependencies=*/nullptr,
                                            /*dependency_count=*/0, middle));
  hipMemAllocNodeParams nested_parameters = {};
  nested_parameters.poolProps.allocType = hipMemAllocationTypePinned;
  nested_parameters.poolProps.location.type = hipMemLocationTypeDevice;
  nested_parameters.poolProps.location.id = device_;
  nested_parameters.bytesize = 4096;
  hipGraphNode_t nested_allocation_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_mem_alloc_node(
                            &nested_allocation_node, leaf,
                            /*dependencies=*/nullptr,
                            /*dependency_count=*/0, &nested_parameters));
  hipGraphExec_t rejected_exec = nullptr;
  EXPECT_EQ(hipErrorNotSupported,
            api_.graph_instantiate(&rejected_exec, root,
                                   /*error_node=*/nullptr,
                                   /*log_buffer=*/nullptr,
                                   /*log_buffer_size=*/0));
  if (rejected_exec) {
    EXPECT_EQ(hipSuccess, api_.graph_exec_destroy(rejected_exec));
  }
  EXPECT_EQ(hipSuccess, api_.graph_destroy(root));
  EXPECT_EQ(hipSuccess, api_.graph_destroy(middle));
  EXPECT_EQ(hipSuccess, api_.graph_destroy(leaf));

  // Executable child replacement applies the same rule before compatibility
  // checks and leaves the original executable launchable after rejection.
  hipGraph_t original_child = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&original_child, /*flags=*/0));
  hipGraphNode_t original_empty_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_empty_node(&original_empty_node, original_child,
                                      /*dependencies=*/nullptr,
                                      /*dependency_count=*/0));
  hipGraph_t update_parent = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&update_parent, /*flags=*/0));
  hipGraphNode_t update_child_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_child_graph_node(
                            &update_child_node, update_parent,
                            /*dependencies=*/nullptr,
                            /*dependency_count=*/0, original_child));
  hipGraphExec_t update_exec = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_instantiate(&update_exec, update_parent,
                                               /*error_node=*/nullptr,
                                               /*log_buffer=*/nullptr,
                                               /*log_buffer_size=*/0));

  hipMemAllocNodeParams late_parameters = {};
  late_parameters.poolProps.allocType = hipMemAllocationTypePinned;
  late_parameters.poolProps.location.type = hipMemLocationTypeDevice;
  late_parameters.poolProps.location.id = device_;
  late_parameters.bytesize = 4096;
  hipGraphNode_t late_allocation_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_mem_alloc_node(
                            &late_allocation_node, original_child,
                            /*dependencies=*/nullptr,
                            /*dependency_count=*/0, &late_parameters));
  hipGraphNode_t update_error_node = nullptr;
  hipGraphExecUpdateResult update_result = hipGraphExecUpdateError;
  EXPECT_EQ(hipErrorGraphExecUpdateFailure,
            api_.graph_exec_update(update_exec, update_parent,
                                   &update_error_node, &update_result));
  EXPECT_EQ(nullptr, update_error_node);
  EXPECT_EQ(hipGraphExecUpdateErrorNotSupported, update_result);

  hipGraph_t replacement_child = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&replacement_child, /*flags=*/0));
  hipMemAllocNodeParams replacement_parameters = {};
  replacement_parameters.poolProps.allocType = hipMemAllocationTypePinned;
  replacement_parameters.poolProps.location.type = hipMemLocationTypeDevice;
  replacement_parameters.poolProps.location.id = device_;
  replacement_parameters.bytesize = 4096;
  hipGraphNode_t replacement_allocation_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_mem_alloc_node(
                            &replacement_allocation_node, replacement_child,
                            /*dependencies=*/nullptr,
                            /*dependency_count=*/0, &replacement_parameters));
  EXPECT_EQ(hipErrorNotSupported,
            api_.graph_exec_child_graph_node_set_params(
                update_exec, update_child_node, replacement_child));
  EXPECT_EQ(hipSuccess, api_.graph_launch(update_exec, stream_));
  EXPECT_EQ(hipSuccess, api_.stream_synchronize(stream_));

  EXPECT_EQ(hipSuccess, api_.graph_destroy(replacement_child));
  EXPECT_EQ(hipSuccess, api_.graph_exec_destroy(update_exec));
  EXPECT_EQ(hipSuccess, api_.graph_destroy(update_parent));
  EXPECT_EQ(hipSuccess, api_.graph_destroy(original_child));
}

TEST_F(HipGraphApiTest,
       ChildUpdateUsesInstantiatedTopologyAndRejectsChangesAtomically) {
  hipGraph_t original_child = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&original_child, /*flags=*/0));
  hipGraphNode_t original_first = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_empty_node(&original_first, original_child,
                                      /*dependencies=*/nullptr,
                                      /*dependency_count=*/0));
  hipGraphNode_t original_second = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_empty_node(&original_second, original_child,
                                      /*dependencies=*/nullptr,
                                      /*dependency_count=*/0));
  const hipGraphNode_t original_dependencies[] = {original_first};
  hipGraphNode_t original_third = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_empty_node(&original_third, original_child,
                                      original_dependencies,
                                      /*dependency_count=*/1));

  hipGraph_t parent = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&parent, /*flags=*/0));
  hipGraphNode_t child_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_child_graph_node(
                            &child_node, parent,
                            /*dependencies=*/nullptr,
                            /*dependency_count=*/0, original_child));
  hipGraphExec_t executable_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_instantiate(&executable_graph, parent,
                                               /*error_node=*/nullptr,
                                               /*log_buffer=*/nullptr,
                                               /*log_buffer_size=*/0));

  // Mutating the public source after instantiation must not change the
  // executable's compatibility reference. A graph matching the instantiated
  // three-node snapshot remains a valid replacement.
  hipGraphNode_t original_fourth = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_empty_node(&original_fourth, original_child,
                                      /*dependencies=*/nullptr,
                                      /*dependency_count=*/0));
  hipGraph_t matching_snapshot = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&matching_snapshot, /*flags=*/0));
  hipGraphNode_t matching_first = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_empty_node(&matching_first, matching_snapshot,
                                      /*dependencies=*/nullptr,
                                      /*dependency_count=*/0));
  hipGraphNode_t matching_second = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_empty_node(&matching_second, matching_snapshot,
                                      /*dependencies=*/nullptr,
                                      /*dependency_count=*/0));
  const hipGraphNode_t matching_dependencies[] = {matching_first};
  hipGraphNode_t matching_third = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_empty_node(&matching_third, matching_snapshot,
                                      matching_dependencies,
                                      /*dependency_count=*/1));
  ASSERT_EQ(hipSuccess, api_.graph_exec_child_graph_node_set_params(
                            executable_graph, child_node, matching_snapshot));
  ASSERT_EQ(hipSuccess, api_.graph_launch(executable_graph, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));

  hipGraph_t different_types = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&different_types, /*flags=*/0));
  hipGraphNode_t type_first = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_empty_node(&type_first, different_types,
                                                  /*dependencies=*/nullptr,
                                                  /*dependency_count=*/0));
  hipGraphNode_t type_second = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_empty_node(&type_second, different_types,
                                                  /*dependencies=*/nullptr,
                                                  /*dependency_count=*/0));
  const hipHostNodeParams host_parameters = {
      /*.fn=*/NoopHostFunction,
      /*.userData=*/nullptr,
  };
  const hipGraphNode_t type_dependencies[] = {type_first};
  hipGraphNode_t host_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_host_node(
                            &host_node, different_types, type_dependencies,
                            /*dependency_count=*/1, &host_parameters));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.graph_exec_child_graph_node_set_params(
                executable_graph, child_node, different_types));
  ASSERT_EQ(hipSuccess, api_.graph_launch(executable_graph, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));

  hipGraph_t different_topology = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&different_topology, /*flags=*/0));
  hipGraphNode_t topology_first = nullptr;
  hipGraphNode_t topology_second = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_empty_node(&topology_first, different_topology,
                                      /*dependencies=*/nullptr,
                                      /*dependency_count=*/0));
  ASSERT_EQ(hipSuccess,
            api_.graph_add_empty_node(&topology_second, different_topology,
                                      /*dependencies=*/nullptr,
                                      /*dependency_count=*/0));
  const hipGraphNode_t topology_dependencies[] = {topology_second};
  hipGraphNode_t topology_third = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.graph_add_empty_node(&topology_third, different_topology,
                                      topology_dependencies,
                                      /*dependency_count=*/1));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.graph_exec_child_graph_node_set_params(
                executable_graph, child_node, different_topology));

  // Both rejected updates leave the compatible replacement installed and
  // launchable.
  ASSERT_EQ(hipSuccess, api_.graph_launch(executable_graph, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));

  EXPECT_EQ(hipSuccess, api_.graph_destroy(different_topology));
  EXPECT_EQ(hipSuccess, api_.graph_destroy(different_types));
  EXPECT_EQ(hipSuccess, api_.graph_destroy(matching_snapshot));
  EXPECT_EQ(hipSuccess, api_.graph_exec_destroy(executable_graph));
  EXPECT_EQ(hipSuccess, api_.graph_destroy(parent));
  EXPECT_EQ(hipSuccess, api_.graph_destroy(original_child));
  EXPECT_EQ(hipSuccess, api_.stream_synchronize(stream_));
}

TEST_F(HipGraphApiTest, ExecMemcpyUpdateMutatesOnlyPrivateTemplate) {
  constexpr size_t kSize = sizeof(uint32_t);
  const uint32_t source_a_value = 0x12345678u;
  const uint32_t source_b_value = 0x9ABCDEF0u;
  const uint32_t zero = 0;
  hipDeviceptr_t source_a = nullptr;
  hipDeviceptr_t source_b = nullptr;
  hipDeviceptr_t destination_a = nullptr;
  hipDeviceptr_t destination_b = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc(&source_a, kSize));
  ASSERT_EQ(hipSuccess, api_.malloc(&source_b, kSize));
  ASSERT_EQ(hipSuccess, api_.malloc(&destination_a, kSize));
  ASSERT_EQ(hipSuccess, api_.malloc(&destination_b, kSize));
  ASSERT_EQ(hipSuccess, api_.memcpy(source_a, &source_a_value, kSize,
                                    hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, api_.memcpy(source_b, &source_b_value, kSize,
                                    hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess,
            api_.memcpy(destination_a, &zero, kSize, hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess,
            api_.memcpy(destination_b, &zero, kSize, hipMemcpyHostToDevice));

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
  hipGraphNode_t copy_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_memcpy_node_1d(
                            &copy_node, graph, /*dependencies=*/nullptr,
                            /*dependency_count=*/0, destination_a, source_a,
                            kSize, hipMemcpyDeviceToDevice));
  hipGraphExec_t executable_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_instantiate(&executable_graph, graph,
                                               /*error_node=*/nullptr,
                                               /*log_buffer=*/nullptr,
                                               /*log_buffer_size=*/0));

  ASSERT_EQ(hipSuccess, api_.graph_exec_memcpy_node_set_params_1d(
                            executable_graph, copy_node, destination_b,
                            source_b, kSize, hipMemcpyDeviceToDevice));
  ASSERT_EQ(hipSuccess, api_.graph_launch(executable_graph, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));

  uint32_t destination_a_value = 0;
  uint32_t destination_b_value = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(&destination_a_value, destination_a, kSize,
                                    hipMemcpyDeviceToHost));
  ASSERT_EQ(hipSuccess, api_.memcpy(&destination_b_value, destination_b, kSize,
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(zero, destination_a_value);
  EXPECT_EQ(source_b_value, destination_b_value);

  hipGraphExec_t source_executable_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_instantiate(&source_executable_graph, graph,
                                               /*error_node=*/nullptr,
                                               /*log_buffer=*/nullptr,
                                               /*log_buffer_size=*/0));
  ASSERT_EQ(hipSuccess, api_.graph_launch(source_executable_graph, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
  ASSERT_EQ(hipSuccess, api_.memcpy(&destination_a_value, destination_a, kSize,
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(source_a_value, destination_a_value);

  EXPECT_EQ(hipSuccess, api_.graph_exec_destroy(source_executable_graph));
  EXPECT_EQ(hipSuccess, api_.graph_exec_destroy(executable_graph));
  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
  EXPECT_EQ(hipSuccess, api_.free(destination_b));
  EXPECT_EQ(hipSuccess, api_.free(destination_a));
  EXPECT_EQ(hipSuccess, api_.free(source_b));
  EXPECT_EQ(hipSuccess, api_.free(source_a));
}

TEST_F(HipGraphApiTest,
       ExecMemsetUpdateTracksStableIdentityAcrossGraphMutation) {
  constexpr size_t kSize = sizeof(uint32_t);
  const uint32_t zero = 0;
  hipDeviceptr_t destination_a = nullptr;
  hipDeviceptr_t destination_b = nullptr;
  hipDeviceptr_t destination_c = nullptr;
  hipDeviceptr_t destination_d = nullptr;
  ASSERT_EQ(hipSuccess, api_.malloc(&destination_a, kSize));
  ASSERT_EQ(hipSuccess, api_.malloc(&destination_b, kSize));
  ASSERT_EQ(hipSuccess, api_.malloc(&destination_c, kSize));
  ASSERT_EQ(hipSuccess, api_.malloc(&destination_d, kSize));
  ASSERT_EQ(hipSuccess,
            api_.memcpy(destination_a, &zero, kSize, hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess,
            api_.memcpy(destination_b, &zero, kSize, hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess,
            api_.memcpy(destination_c, &zero, kSize, hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess,
            api_.memcpy(destination_d, &zero, kSize, hipMemcpyHostToDevice));

  hipGraph_t graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph, /*flags=*/0));
  hipMemsetParams first_parameters = {};
  first_parameters.dst = destination_a;
  first_parameters.value = 0x11;
  first_parameters.elementSize = 1;
  first_parameters.width = kSize;
  first_parameters.height = 1;
  hipGraphNode_t first_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_memset_node(
                            &first_node, graph, /*dependencies=*/nullptr,
                            /*dependency_count=*/0, &first_parameters));
  hipMemsetParams second_parameters = first_parameters;
  second_parameters.dst = destination_b;
  second_parameters.value = 0x22;
  hipGraphNode_t second_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_memset_node(
                            &second_node, graph, &first_node,
                            /*dependency_count=*/1, &second_parameters));

  hipGraphExec_t executable_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_instantiate(&executable_graph, graph,
                                               /*error_node=*/nullptr,
                                               /*log_buffer=*/nullptr,
                                               /*log_buffer_size=*/0));
  ASSERT_EQ(hipSuccess, api_.graph_destroy_node(first_node));
  hipMemsetParams added_parameters = first_parameters;
  added_parameters.dst = destination_c;
  added_parameters.value = 0x33;
  hipGraphNode_t added_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_memset_node(
                            &added_node, graph, /*dependencies=*/nullptr,
                            /*dependency_count=*/0, &added_parameters));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.graph_exec_memset_node_set_params(executable_graph, added_node,
                                                   &added_parameters));

  hipMemsetParams updated_second_parameters = second_parameters;
  updated_second_parameters.value = 0x44;
  ASSERT_EQ(hipSuccess,
            api_.graph_exec_memset_node_set_params(
                executable_graph, second_node, &updated_second_parameters));
  ASSERT_EQ(hipSuccess, api_.graph_launch(executable_graph, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));

  uint32_t destination_a_value = 0;
  uint32_t destination_b_value = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(&destination_a_value, destination_a, kSize,
                                    hipMemcpyDeviceToHost));
  ASSERT_EQ(hipSuccess, api_.memcpy(&destination_b_value, destination_b, kSize,
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(0x11111111u, destination_a_value);
  EXPECT_EQ(0x44444444u, destination_b_value);

  hipGraph_t replacement_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&replacement_graph, /*flags=*/0));
  hipMemsetParams replacement_first_parameters = first_parameters;
  replacement_first_parameters.dst = destination_c;
  replacement_first_parameters.value = 0x55;
  hipGraphNode_t replacement_first_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_memset_node(
                            &replacement_first_node, replacement_graph,
                            /*dependencies=*/nullptr, /*dependency_count=*/0,
                            &replacement_first_parameters));
  hipMemsetParams replacement_second_parameters = first_parameters;
  replacement_second_parameters.dst = destination_d;
  replacement_second_parameters.value = 0x66;
  hipGraphNode_t replacement_second_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_add_memset_node(
                            &replacement_second_node, replacement_graph,
                            &replacement_first_node, /*dependency_count=*/1,
                            &replacement_second_parameters));
  hipGraphNode_t error_node = nullptr;
  hipGraphExecUpdateResult update_result = hipGraphExecUpdateError;
  ASSERT_EQ(hipSuccess,
            api_.graph_exec_update(executable_graph, replacement_graph,
                                   &error_node, &update_result));
  EXPECT_EQ(nullptr, error_node);
  EXPECT_EQ(hipGraphExecUpdateSuccess, update_result);
  EXPECT_EQ(hipErrorInvalidValue,
            api_.graph_exec_memset_node_set_params(
                executable_graph, second_node, &updated_second_parameters));

  hipMemsetParams updated_replacement_parameters =
      replacement_second_parameters;
  updated_replacement_parameters.value = 0x77;
  ASSERT_EQ(hipSuccess, api_.graph_exec_memset_node_set_params(
                            executable_graph, replacement_second_node,
                            &updated_replacement_parameters));
  ASSERT_EQ(hipSuccess, api_.graph_launch(executable_graph, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));

  uint32_t destination_c_value = 0;
  uint32_t destination_d_value = 0;
  ASSERT_EQ(hipSuccess, api_.memcpy(&destination_c_value, destination_c, kSize,
                                    hipMemcpyDeviceToHost));
  ASSERT_EQ(hipSuccess, api_.memcpy(&destination_d_value, destination_d, kSize,
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(0x55555555u, destination_c_value);
  EXPECT_EQ(0x77777777u, destination_d_value);

  hipGraphExec_t source_executable_graph = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_instantiate(&source_executable_graph, graph,
                                               /*error_node=*/nullptr,
                                               /*log_buffer=*/nullptr,
                                               /*log_buffer_size=*/0));
  ASSERT_EQ(hipSuccess, api_.graph_launch(source_executable_graph, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
  ASSERT_EQ(hipSuccess, api_.memcpy(&destination_b_value, destination_b, kSize,
                                    hipMemcpyDeviceToHost));
  EXPECT_EQ(0x22222222u, destination_b_value);

  EXPECT_EQ(hipSuccess, api_.graph_exec_destroy(source_executable_graph));
  EXPECT_EQ(hipSuccess, api_.graph_exec_destroy(executable_graph));
  EXPECT_EQ(hipSuccess, api_.graph_destroy(replacement_graph));
  EXPECT_EQ(hipSuccess, api_.graph_destroy(graph));
  EXPECT_EQ(hipSuccess, api_.free(destination_d));
  EXPECT_EQ(hipSuccess, api_.free(destination_c));
  EXPECT_EQ(hipSuccess, api_.free(destination_b));
  EXPECT_EQ(hipSuccess, api_.free(destination_a));
}

}  // namespace
