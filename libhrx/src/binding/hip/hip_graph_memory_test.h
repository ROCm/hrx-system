// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_BINDING_HIP_GRAPH_MEMORY_TEST_H_
#define LIBHRX_BINDING_HIP_GRAPH_MEMORY_TEST_H_

#include <array>
#include <semaphore>

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/testing/gtest.h"

namespace hrx::hip::testing {

struct HipGraphApi {
  // Initializes the suite's single frontend instance.
  decltype(&hipInit) initialize = nullptr;
  // Joins frontend services before closing the shared object.
  hipError_t (*deinitialize)() = nullptr;
  // Reports the already-selected device without creating another device.
  decltype(&hipGetDevice) get_device = nullptr;
  // Reports the current primary compatibility context.
  decltype(&hipCtxGetCurrent) get_context = nullptr;
  // Creates a compatibility context borrowing the existing native device.
  decltype(&hipCtxCreate) create_context = nullptr;
  // Selects the context owning a transfer endpoint.
  decltype(&hipCtxSetCurrent) set_context = nullptr;
  // Releases a compatibility context after its allocations and graph owners.
  decltype(&hipCtxDestroy) destroy_context = nullptr;
  // Performs checked setup/readback outside the graph under test.
  decltype(&hipMemcpy) copy = nullptr;
  // Creates a nonblocking stream for graph launches.
  decltype(&hipStreamCreateWithFlags) stream_create = nullptr;
  // Joins and releases a stream.
  decltype(&hipStreamDestroy) stream_destroy = nullptr;
  // Waits for graph completion.
  decltype(&hipStreamSynchronize) stream_synchronize = nullptr;
  // Checks pending work without waiting on its completion.
  decltype(&hipStreamQuery) stream_query = nullptr;
  // Holds a stream at an application-owned completion gate.
  decltype(&hipLaunchHostFunc) launch_host = nullptr;
  // Allocates a device buffer for graph transfers.
  decltype(&hipMalloc) allocate = nullptr;
  // Releases the allocation after graph completion.
  decltype(&hipFree) free = nullptr;
  // Records a graph allocation while a stream is being captured.
  decltype(&hipMallocAsync) allocate_async = nullptr;
  // Ends a graph allocation without releasing its replay address.
  decltype(&hipFreeAsync) free_async = nullptr;
  // Queries physical backing and logical graph allocation usage.
  decltype(&hipDeviceGetGraphMemAttribute) graph_memory_attribute = nullptr;
  // Returns unused physical graph backing while executable graphs stay live.
  decltype(&hipDeviceGraphMemTrim) graph_memory_trim = nullptr;
  // Allocates host storage with a direct device mapping.
  decltype(&hipHostAlloc) allocate_host = nullptr;
  // Releases host storage after all graph users are joined.
  decltype(&hipFreeHost) free_host = nullptr;
  // Creates an empty graph template.
  decltype(&hipGraphCreate) graph_create = nullptr;
  // Releases a graph template.
  decltype(&hipGraphDestroy) graph_destroy = nullptr;
  // Clones a template and its owned copy payloads.
  decltype(&hipGraphClone) graph_clone = nullptr;
  // Finds a clone's node before the original template is destroyed.
  decltype(&hipGraphNodeFindInClone) find_clone_node = nullptr;
  // Compiles a template snapshot.
  decltype(&hipGraphInstantiateWithFlags) instantiate = nullptr;
  // Destroys a compiled graph after joining its launches.
  decltype(&hipGraphExecDestroy) executable_destroy = nullptr;
  // Submits a compiled graph to a stream.
  decltype(&hipGraphLaunch) launch = nullptr;
  // Adds a contiguous graph copy.
  decltype(&hipGraphAddMemcpyNode1D) add_copy_1d = nullptr;
  // Allocates graph-owned device storage with a stable graph address.
  decltype(&hipGraphAddMemAllocNode) add_allocation = nullptr;
  // Ends the graph-visible lifetime of a graph allocation.
  decltype(&hipGraphAddMemFreeNode) add_free = nullptr;
  // Adds a pitched multidimensional graph copy.
  decltype(&hipGraphAddMemcpyNode) add_copy = nullptr;
  // Adds a fill that can execute independently of other graph roots.
  decltype(&hipGraphAddMemsetNode) add_fill = nullptr;
  // Reads the original public fill geometry, including an implicit zero pitch.
  decltype(&hipGraphMemsetNodeGetParams) get_fill = nullptr;
  // Replaces a template's fill geometry and pattern.
  decltype(&hipGraphMemsetNodeSetParams) set_fill = nullptr;
  // Replaces the pattern of a compiled fill with the same geometry.
  decltype(&hipGraphExecMemsetNodeSetParams) set_executable_fill = nullptr;
  // Records or submits a pitched byte fill.
  decltype(&hipMemset2DAsync) fill_2d_async = nullptr;
  // Adds an explicit join after a set of independent operations.
  decltype(&hipGraphAddEmptyNode) add_empty = nullptr;
  // Updates a template's contiguous copy parameters.
  decltype(&hipGraphMemcpyNodeSetParams1D) set_copy = nullptr;
  // Updates only one compiled graph's contiguous copy parameters.
  decltype(&hipGraphExecMemcpyNodeSetParams1D) set_executable_copy = nullptr;
  // Updates one compiled pitched copy without joining previous launches.
  decltype(&hipGraphExecMemcpyNodeSetParams) set_executable_copy_3d = nullptr;
  // Replaces the executable parameters using a topology-compatible template.
  decltype(&hipGraphExecUpdate) update_executable = nullptr;
  // Changes an executable's node-enable state without editing its template.
  decltype(&hipGraphNodeSetEnabled) set_enabled = nullptr;
  // Reports the public operation type, including host-executed copies.
  decltype(&hipGraphNodeGetType) get_node_type = nullptr;
  // Removes a template node without changing its existing executables.
  decltype(&hipGraphDestroyNode) destroy_node = nullptr;
  // Adds public dependencies after node construction.
  decltype(&hipGraphAddDependencies) add_dependencies = nullptr;
  // Removes a validated batch of public dependencies atomically.
  decltype(&hipGraphRemoveDependencies) remove_dependencies = nullptr;
  // Enumerates public predecessor/destination pairs.
  decltype(&hipGraphGetEdges) get_edges = nullptr;
  // Enumerates nodes without public predecessors.
  decltype(&hipGraphGetRootNodes) get_roots = nullptr;
  // Enumerates one node's public predecessors.
  decltype(&hipGraphNodeGetDependencies) get_dependencies = nullptr;
  // Enumerates nodes depending on one public predecessor.
  decltype(&hipGraphNodeGetDependentNodes) get_dependents = nullptr;
  // Begins recording operations issued through the streaming copy API.
  decltype(&hipStreamBeginCapture) begin_capture = nullptr;
  // Begins a capture that extends an existing graph frontier.
  decltype(&hipStreamBeginCaptureToGraph) begin_capture_to_graph = nullptr;
  // Returns the graph built from stream operations.
  decltype(&hipStreamEndCapture) end_capture = nullptr;
  // Borrows the graph under construction during stream capture.
  decltype(&hipStreamGetCaptureInfo_v2) capture_info = nullptr;
  // Records or submits a contiguous streaming copy.
  decltype(&hipMemcpyAsync) copy_async = nullptr;
  // Records or submits a pitched multidimensional streaming copy.
  decltype(&hipMemcpy3DAsync) copy_3d_async = nullptr;
  // Enumerates only public graph operations.
  decltype(&hipGraphGetNodes) get_nodes = nullptr;
  // Adds an independently described nested graph.
  decltype(&hipGraphAddChildGraphNode) add_child = nullptr;
  // Borrows the node's independently snapshotted graph template.
  decltype(&hipGraphChildGraphNodeGetGraph) get_child = nullptr;
  // Applies the generic executable update to its typed operation.
  decltype(&hipGraphExecNodeSetParams) set_executable_node = nullptr;
  // Reports a template copy's public parameters.
  decltype(&hipGraphMemcpyNodeGetParams) get_copy = nullptr;
  // Adds an application function after a copy.
  decltype(&hipGraphAddHostNode) add_host = nullptr;
  // Reports an application callback's public function and argument.
  decltype(&hipGraphHostNodeGetParams) get_host = nullptr;
  // Updates a template's application callback.
  decltype(&hipGraphHostNodeSetParams) set_host = nullptr;
  // Updates a compiled application callback.
  decltype(&hipGraphExecHostNodeSetParams) set_executable_host = nullptr;
};

class HipGraphMemoryApiTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite();

  static void TearDownTestSuite();

  void SetUp() override;

  void TearDown() override;

  void Replay(hipGraphExec_t executable = nullptr);

  void BuildAllocationGraph(unsigned long long flags, void** out_pointer);

  void HoldStreamAtGate(hipStream_t stream = nullptr);

  void ReleaseStreamGate();

  void CheckPendingStagedCopyUpdates(bool pitched);

  static void Increment(void* user_data);

  static void ObserveDestination(void* user_data);

  static void ReplaceSource(void* user_data);

  void CheckPitchedCopyClone(bool implicit_slice_height, bool registered_host,
                             bool capture = false);

  // Exact frontend shared object and API reused for the whole suite.
  static hrx::hip::testing::HipDso dso_;
  // Function table resolved only from that shared object.
  static HipGraphApi api_;
  // Launch stream joined on every test exit.
  hipStream_t stream_ = nullptr;
  // Second stream for replaying one executable across timeline boundaries.
  hipStream_t other_stream_ = nullptr;
  // A host callback gate retained through all failure cleanup paths.
  struct StreamGate {
    // Announces that the stream reached the gated application callback.
    std::binary_semaphore entered{0};
    // Allows that callback to return and the stream to progress.
    std::binary_semaphore release{0};
    // True while teardown must release the gate before joining streams.
    bool pending = false;
  } gate_;
  // Original template, or null after explicit destruction.
  hipGraph_t graph_ = nullptr;
  // Clone whose payload lifetime is independent of the original template.
  hipGraph_t clone_ = nullptr;
  // Compiled graph snapshot.
  hipGraphExec_t executable_ = nullptr;
  // Independent executable from the same source template.
  hipGraphExec_t sibling_ = nullptr;
  // Device endpoint retained through all graph launches.
  hipDeviceptr_t device_ = nullptr;
  // Borrowed primary context owning the graph and launch stream.
  hipCtx_t primary_context_ = nullptr;
  // Secondary compatibility context sharing the suite's native GPU device.
  hipCtx_t remote_context_ = nullptr;
  // Peer copy endpoints retained through graph completion and cleanup.
  std::array<hipDeviceptr_t, 2> remote_buffers_ = {};
  // Registered host endpoints retained until graph and stream cleanup.
  std::array<void*, 2> host_buffers_ = {};
  // Initial copy data retained through teardown, including assertion failures.
  std::array<uint8_t, 64> source_ = {};
  // Replacement data used by template and executable updates.
  std::array<uint8_t, 64> replacement_ = {};
  // Data observed by a host consumer after the complete public copy operation.
  std::array<uint8_t, 64> observed_ = {};
  // Copy destination retained through teardown.
  std::array<uint8_t, 64> destination_ = {};
  // Application callback outputs retained through teardown.
  std::array<int, 3> callbacks_ = {};
};

}  // namespace hrx::hip::testing

#endif  // LIBHRX_BINDING_HIP_GRAPH_MEMORY_TEST_H_
