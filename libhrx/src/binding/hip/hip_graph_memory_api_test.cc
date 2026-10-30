// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <semaphore>

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/testing/gtest.h"

namespace {

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
  // Adds an explicit join after a set of independent operations.
  decltype(&hipGraphAddEmptyNode) add_empty = nullptr;
  // Updates a template's contiguous copy parameters.
  decltype(&hipGraphMemcpyNodeSetParams1D) set_copy = nullptr;
  // Updates only one compiled graph's contiguous copy parameters.
  decltype(&hipGraphExecMemcpyNodeSetParams1D) set_executable_copy = nullptr;
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
  // Begins recording operations issued through the streaming copy API.
  decltype(&hipStreamBeginCapture) begin_capture = nullptr;
  // Returns the graph built from stream operations.
  decltype(&hipStreamEndCapture) end_capture = nullptr;
  // Records or submits a contiguous streaming copy.
  decltype(&hipMemcpyAsync) copy_async = nullptr;
  // Enumerates only public graph operations.
  decltype(&hipGraphGetNodes) get_nodes = nullptr;
  // Adds an independently described nested graph.
  decltype(&hipGraphAddChildGraphNode) add_child = nullptr;
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
  static void SetUpTestSuite() {
    ASSERT_TRUE(dso_.Open()) << dso_.error();
#define RESOLVE(field, symbol)                             \
  api_.field = dso_.Resolve<decltype(api_.field)>(symbol); \
  ASSERT_NE(api_.field, nullptr) << dso_.error()
    RESOLVE(initialize, "hipInit");
    RESOLVE(deinitialize, "hipHALDeinit");
    RESOLVE(get_device, "hipGetDevice");
    RESOLVE(get_context, "hipCtxGetCurrent");
    RESOLVE(create_context, "hipCtxCreate");
    RESOLVE(set_context, "hipCtxSetCurrent");
    RESOLVE(destroy_context, "hipCtxDestroy");
    RESOLVE(copy, "hipMemcpy");
    RESOLVE(stream_create, "hipStreamCreateWithFlags");
    RESOLVE(stream_destroy, "hipStreamDestroy");
    RESOLVE(stream_synchronize, "hipStreamSynchronize");
    RESOLVE(stream_query, "hipStreamQuery");
    RESOLVE(launch_host, "hipLaunchHostFunc");
    RESOLVE(allocate, "hipMalloc");
    RESOLVE(free, "hipFree");
    RESOLVE(allocate_host, "hipHostAlloc");
    RESOLVE(free_host, "hipFreeHost");
    RESOLVE(graph_create, "hipGraphCreate");
    RESOLVE(graph_destroy, "hipGraphDestroy");
    RESOLVE(graph_clone, "hipGraphClone");
    RESOLVE(find_clone_node, "hipGraphNodeFindInClone");
    RESOLVE(instantiate, "hipGraphInstantiateWithFlags");
    RESOLVE(executable_destroy, "hipGraphExecDestroy");
    RESOLVE(launch, "hipGraphLaunch");
    RESOLVE(add_copy_1d, "hipGraphAddMemcpyNode1D");
    RESOLVE(add_allocation, "hipGraphAddMemAllocNode");
    RESOLVE(add_free, "hipGraphAddMemFreeNode");
    RESOLVE(add_copy, "hipGraphAddMemcpyNode");
    RESOLVE(add_fill, "hipGraphAddMemsetNode");
    RESOLVE(add_empty, "hipGraphAddEmptyNode");
    RESOLVE(set_copy, "hipGraphMemcpyNodeSetParams1D");
    RESOLVE(set_executable_copy, "hipGraphExecMemcpyNodeSetParams1D");
    RESOLVE(update_executable, "hipGraphExecUpdate");
    RESOLVE(set_enabled, "hipGraphNodeSetEnabled");
    RESOLVE(get_node_type, "hipGraphNodeGetType");
    RESOLVE(destroy_node, "hipGraphDestroyNode");
    RESOLVE(add_dependencies, "hipGraphAddDependencies");
    RESOLVE(begin_capture, "hipStreamBeginCapture");
    RESOLVE(end_capture, "hipStreamEndCapture");
    RESOLVE(copy_async, "hipMemcpyAsync");
    RESOLVE(get_nodes, "hipGraphGetNodes");
    RESOLVE(add_child, "hipGraphAddChildGraphNode");
    RESOLVE(set_executable_node, "hipGraphExecNodeSetParams");
    RESOLVE(get_copy, "hipGraphMemcpyNodeGetParams");
    RESOLVE(add_host, "hipGraphAddHostNode");
    RESOLVE(get_host, "hipGraphHostNodeGetParams");
    RESOLVE(set_host, "hipGraphHostNodeSetParams");
    RESOLVE(set_executable_host, "hipGraphExecHostNodeSetParams");
#undef RESOLVE
    ASSERT_EQ(hipSuccess, api_.initialize(0));
  }

  static void TearDownTestSuite() {
    if (!dso_.is_open()) {
      return;
    }
    ASSERT_NE(api_.deinitialize, nullptr);
    EXPECT_EQ(hipSuccess, api_.deinitialize());
    api_ = {};
    EXPECT_TRUE(dso_.Close()) << dso_.error();
  }

  void SetUp() override {
    ASSERT_EQ(hipSuccess, api_.stream_create(&stream_, hipStreamNonBlocking));
    ASSERT_EQ(hipSuccess, api_.get_context(&primary_context_));
    ASSERT_EQ(hipSuccess, api_.graph_create(&graph_, 0));
    for (size_t i = 0; i < source_.size(); ++i) {
      source_[i] = static_cast<uint8_t>(i + 1);
      replacement_[i] = static_cast<uint8_t>(i + 101);
    }
  }

  void TearDown() override {
    ReleaseStreamGate();
    if (primary_context_) {
      EXPECT_EQ(hipSuccess, api_.set_context(primary_context_));
    }
    if (stream_) {
      EXPECT_EQ(hipSuccess, api_.stream_synchronize(stream_));
    }
    if (other_stream_) {
      EXPECT_EQ(hipSuccess, api_.stream_synchronize(other_stream_));
    }
    if (executable_) {
      EXPECT_EQ(hipSuccess, api_.executable_destroy(executable_));
    }
    if (sibling_) {
      EXPECT_EQ(hipSuccess, api_.executable_destroy(sibling_));
    }
    if (clone_) {
      EXPECT_EQ(hipSuccess, api_.graph_destroy(clone_));
    }
    if (graph_) {
      EXPECT_EQ(hipSuccess, api_.graph_destroy(graph_));
    }
    if (stream_) {
      EXPECT_EQ(hipSuccess, api_.stream_destroy(stream_));
    }
    if (other_stream_) {
      EXPECT_EQ(hipSuccess, api_.stream_destroy(other_stream_));
    }
    if (device_) {
      EXPECT_EQ(hipSuccess, api_.free(device_));
    }
    for (auto* buffer : host_buffers_) {
      if (buffer) {
        EXPECT_EQ(hipSuccess, api_.free_host(buffer));
      }
    }
    if (remote_context_) {
      EXPECT_EQ(hipSuccess, api_.set_context(remote_context_));
      for (auto* buffer : remote_buffers_) {
        if (buffer) {
          EXPECT_EQ(hipSuccess, api_.free(buffer));
        }
      }
      EXPECT_EQ(hipSuccess, api_.destroy_context(remote_context_));
      EXPECT_EQ(hipSuccess, api_.set_context(primary_context_));
    }
  }

  void Replay(hipGraphExec_t executable = nullptr) {
    ASSERT_EQ(hipSuccess,
              api_.launch(executable ? executable : executable_, stream_));
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
  }

  void HoldStreamAtGate() {
    gate_.pending = true;
    ASSERT_EQ(hipSuccess, api_.launch_host(
                              stream_,
                              +[](void* argument) {
                                auto* gate = static_cast<StreamGate*>(argument);
                                gate->entered.release();
                                gate->release.acquire();
                              },
                              &gate_));
    gate_.entered.acquire();
  }

  void ReleaseStreamGate() {
    if (gate_.pending) {
      gate_.release.release();
      gate_.pending = false;
    }
  }

  static void Increment(void* user_data) { ++*static_cast<int*>(user_data); }

  static void ObserveDestination(void* user_data) {
    auto* test = static_cast<HipGraphMemoryApiTest*>(user_data);
    test->observed_ = test->destination_;
  }

  static void ReplaceSource(void* user_data) {
    auto* test = static_cast<HipGraphMemoryApiTest*>(user_data);
    test->source_ = test->replacement_;
  }

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

hrx::hip::testing::HipDso HipGraphMemoryApiTest::dso_;
HipGraphApi HipGraphMemoryApiTest::api_;

TEST_F(HipGraphMemoryApiTest, IndependentFillsJoinBeforeGraphCompletion) {
  ASSERT_EQ(hipSuccess, api_.allocate(&device_, source_.size()));
  std::array<hipGraphNode_t, 64> fills = {};
  for (size_t i = 0; i < fills.size(); ++i) {
    const hipMemsetParams params = {
        .dst = static_cast<uint8_t*>(device_) + i,
        .elementSize = 1,
        .height = 1,
        .pitch = 0,
        .value = static_cast<unsigned int>(i + 1),
        .width = 1,
    };
    ASSERT_EQ(hipSuccess,
              api_.add_fill(&fills[i], graph_, nullptr, 0, &params));
  }
  for (const bool explicit_join : {false, true}) {
    SCOPED_TRACE(explicit_join);
    if (explicit_join) {
      hipGraphNode_t join = nullptr;
      ASSERT_EQ(hipSuccess,
                api_.add_empty(&join, graph_, fills.data(), fills.size()));
    }
    ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
    for (int replay = 0; replay < 2; ++replay) {
      destination_.fill(0);
      ASSERT_EQ(hipSuccess, api_.copy_async(device_, destination_.data(),
                                            destination_.size(),
                                            hipMemcpyHostToDevice, stream_));
      Replay();
      ASSERT_FALSE(HasFatalFailure());
      ASSERT_EQ(hipSuccess,
                api_.copy(destination_.data(), device_, destination_.size(),
                          hipMemcpyDeviceToHost));
      EXPECT_EQ(source_, destination_);
    }
    ASSERT_EQ(hipSuccess, api_.executable_destroy(executable_));
    executable_ = nullptr;
  }
}

TEST_F(HipGraphMemoryApiTest, HostCopyCloneOwnsItsParameters) {
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_copy_1d(&node, graph_, nullptr, 0,
                                         destination_.data(), source_.data(),
                                         source_.size(), hipMemcpyHostToHost));
  ASSERT_EQ(hipSuccess, api_.graph_clone(&clone_, graph_));
  hipGraphNode_t cloned_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.find_clone_node(&cloned_node, node, clone_));
  ASSERT_EQ(hipSuccess,
            api_.set_copy(node, destination_.data(), replacement_.data(),
                          source_.size(), hipMemcpyHostToHost));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph_));
  graph_ = nullptr;
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, clone_, 0));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(destination_, source_);

  hipMemcpy3DParms parameters = {};
  ASSERT_EQ(hipSuccess, api_.get_copy(cloned_node, &parameters));
  EXPECT_EQ(parameters.kind, hipMemcpyHostToHost);
  EXPECT_EQ(parameters.srcPtr.ptr, source_.data());
}

TEST_F(HipGraphMemoryApiTest, ExecutableUpdatesPreserveTemplatesAndSiblings) {
  hipGraphNode_t copy_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_copy_1d(&copy_node, graph_, nullptr, 0,
                                         destination_.data(), source_.data(),
                                         source_.size(), hipMemcpyHostToHost));
  hipHostNodeParams host = {.fn = Increment, .userData = &callbacks_[0]};
  hipGraphNode_t host_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.add_host(&host_node, graph_, &copy_node, 1, &host));
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
  ASSERT_EQ(hipSuccess, api_.instantiate(&sibling_, graph_, 0));

  hipGraphNodeType type = hipGraphNodeTypeEmpty;
  ASSERT_EQ(hipSuccess, api_.get_node_type(copy_node, &type));
  EXPECT_EQ(type, hipGraphNodeTypeMemcpy);
  ASSERT_EQ(hipSuccess, api_.get_node_type(host_node, &type));
  EXPECT_EQ(type, hipGraphNodeTypeHost);

  ASSERT_EQ(hipSuccess,
            api_.set_executable_copy(executable_, copy_node,
                                     destination_.data(), replacement_.data(),
                                     source_.size(), hipMemcpyHostToHost));
  host.userData = &callbacks_[1];
  ASSERT_EQ(hipSuccess,
            api_.set_executable_host(executable_, host_node, &host));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(destination_, replacement_);
  EXPECT_EQ(callbacks_, (std::array<int, 3>{0, 1, 0}));

  hipMemcpy3DParms copy_parameters = {};
  ASSERT_EQ(hipSuccess, api_.get_copy(copy_node, &copy_parameters));
  EXPECT_EQ(copy_parameters.srcPtr.ptr, source_.data());
  hipHostNodeParams host_parameters = {};
  ASSERT_EQ(hipSuccess, api_.get_host(host_node, &host_parameters));
  EXPECT_EQ(host_parameters.userData, &callbacks_[0]);

  // Template edits and an enable-state recompilation must not replace either
  // executable's captured parameters or its already committed updates.
  ASSERT_EQ(hipSuccess,
            api_.set_copy(copy_node, destination_.data(), source_.data(),
                          source_.size() / 2, hipMemcpyHostToHost));
  host.userData = &callbacks_[2];
  ASSERT_EQ(hipSuccess, api_.set_host(host_node, &host));
  destination_.fill(0);
  ASSERT_EQ(hipSuccess, api_.set_enabled(executable_, copy_node, 0));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(destination_, (std::array<uint8_t, 64>{}));
  ASSERT_EQ(hipSuccess, api_.set_enabled(executable_, copy_node, 1));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.set_executable_copy(executable_, copy_node,
                                     destination_.data(), replacement_.data(),
                                     source_.size(), hipMemcpyDeviceToDevice));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(destination_, replacement_);
  EXPECT_EQ(callbacks_, (std::array<int, 3>{0, 3, 0}));
  Replay(sibling_);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(destination_, source_);
  EXPECT_EQ(callbacks_, (std::array<int, 3>{1, 3, 0}));

  // Whole-graph replacement changes parameters while original node handles
  // continue to identify the executable's corresponding nodes.
  ASSERT_EQ(hipSuccess, api_.graph_clone(&clone_, graph_));
  hipGraphNode_t error_node = nullptr;
  hipGraphExecUpdateResult update_result = hipGraphExecUpdateError;
  ASSERT_EQ(hipSuccess, api_.update_executable(executable_, clone_, &error_node,
                                               &update_result));
  EXPECT_EQ(update_result, hipGraphExecUpdateSuccess);
  ASSERT_EQ(hipSuccess,
            api_.set_executable_copy(executable_, copy_node,
                                     destination_.data(), replacement_.data(),
                                     source_.size(), hipMemcpyHostToHost));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(destination_, replacement_);
  EXPECT_EQ(callbacks_, (std::array<int, 3>{1, 3, 1}));
  Replay(sibling_);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(destination_, source_);
  EXPECT_EQ(callbacks_, (std::array<int, 3>{2, 3, 1}));
}

TEST_F(HipGraphMemoryApiTest, NestedUpdatesSurviveTemplateChangesAndRebuilds) {
  ASSERT_EQ(hipSuccess, api_.graph_create(&clone_, 0));
  hipHostNodeParams host = {.fn = Increment, .userData = &callbacks_[0]};
  hipGraphNode_t child_host = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_host(&child_host, clone_, nullptr, 0, &host));
  hipGraphNode_t child_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.add_child(&child_node, graph_, nullptr, 0, clone_));
  hipGraphNode_t host_node = nullptr;
  host.userData = &callbacks_[1];
  ASSERT_EQ(hipSuccess,
            api_.add_host(&host_node, graph_, &child_node, 1, &host));
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
  ASSERT_EQ(hipSuccess, api_.instantiate(&sibling_, graph_, 0));

  host.userData = &callbacks_[2];
  ASSERT_EQ(hipSuccess, api_.set_host(child_host, &host));
  ASSERT_EQ(hipSuccess,
            api_.set_executable_host(executable_, host_node, &host));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(callbacks_, (std::array<int, 3>{1, 0, 1}));

  hipGraphNodeParams update = {
      .type = hipGraphNodeTypeGraph,
      .graph = {.graph = clone_},
  };
  ASSERT_EQ(hipSuccess,
            api_.set_executable_node(executable_, child_node, &update));
  host.userData = &callbacks_[1];
  ASSERT_EQ(hipSuccess, api_.set_host(child_host, &host));
  update.type = hipGraphNodeTypeHost;
  update.host = {.fn = Increment, .userData = &callbacks_[0]};
  ASSERT_EQ(hipSuccess,
            api_.set_executable_node(executable_, host_node, &update));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(callbacks_, (std::array<int, 3>{2, 0, 2}));
  Replay(sibling_);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(callbacks_, (std::array<int, 3>{3, 1, 2}));

  // A changed child topology is rejected without replacing either snapshot.
  hipGraphNode_t extra = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_host(&extra, clone_, &child_host, 1, &host));
  update.type = hipGraphNodeTypeGraph;
  update.graph.graph = clone_;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.set_executable_node(executable_, child_node, &update));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(callbacks_, (std::array<int, 3>{4, 1, 3}));
}

TEST_F(HipGraphMemoryApiTest, TemplateNodeRemovalPreservesExecutableHandles) {
  hipHostNodeParams host = {.fn = Increment, .userData = &callbacks_[0]};
  hipGraphNode_t removed = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_host(&removed, graph_, nullptr, 0, &host));
  hipGraphNode_t copy_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_copy_1d(&copy_node, graph_, &removed, 1,
                                         destination_.data(), source_.data(),
                                         source_.size(), hipMemcpyHostToHost));
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
  ASSERT_EQ(hipSuccess, api_.destroy_node(removed));
  ASSERT_EQ(hipSuccess,
            api_.set_executable_copy(executable_, copy_node,
                                     destination_.data(), replacement_.data(),
                                     source_.size(), hipMemcpyHostToHost));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(destination_, replacement_);
  EXPECT_EQ(callbacks_[0], 1);
}

TEST_F(HipGraphMemoryApiTest, AllocationNodesRemainUntilTheirGraphIsReleased) {
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  hipMemAllocNodeParams allocation = {};
  allocation.poolProps.allocType = hipMemAllocationTypePinned;
  allocation.poolProps.location.type = hipMemLocationTypeDevice;
  allocation.poolProps.location.id = device;
  allocation.bytesize = source_.size();
  hipGraphNode_t allocation_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_allocation(&allocation_node, graph_, nullptr,
                                            0, &allocation));
  ASSERT_NE(nullptr, allocation.dptr);
  hipGraphNode_t upload = nullptr;
  ASSERT_EQ(
      hipSuccess,
      api_.add_copy_1d(&upload, graph_, &allocation_node, 1, allocation.dptr,
                       source_.data(), source_.size(), hipMemcpyHostToDevice));
  hipGraphNode_t download = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.add_copy_1d(&download, graph_, &upload, 1, destination_.data(),
                             allocation.dptr, destination_.size(),
                             hipMemcpyDeviceToHost));
  hipGraphNode_t free_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.add_free(&free_node, graph_, &download, 1, allocation.dptr));

  EXPECT_EQ(hipErrorNotSupported, api_.destroy_node(allocation_node));
  EXPECT_EQ(hipErrorNotSupported, api_.destroy_node(free_node));
  size_t node_count = 0;
  ASSERT_EQ(hipSuccess, api_.get_nodes(graph_, nullptr, &node_count));
  EXPECT_EQ(4u, node_count);

  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
  Replay();
  EXPECT_EQ(source_, destination_);
  EXPECT_EQ(hipErrorNotSupported, api_.destroy_node(allocation_node));
  EXPECT_EQ(hipErrorNotSupported, api_.destroy_node(free_node));
  ASSERT_EQ(hipSuccess,
            api_.set_executable_copy(executable_, upload, allocation.dptr,
                                     replacement_.data(), replacement_.size(),
                                     hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph_));
  graph_ = nullptr;
  destination_.fill(0);
  Replay();
  EXPECT_EQ(replacement_, destination_);

  ASSERT_EQ(hipSuccess,
            api_.stream_create(&other_stream_, hipStreamNonBlocking));
  HoldStreamAtGate();
  ASSERT_FALSE(HasFatalFailure());
  // Allocation-node backing is reused through timeline dependencies. All
  // launches must return while the first stream remains held at the gate.
  ASSERT_EQ(hipSuccess, api_.launch(executable_, stream_));
  ASSERT_EQ(hipSuccess, api_.launch(executable_, stream_));
  ASSERT_EQ(hipSuccess, api_.launch(executable_, other_stream_));
  EXPECT_EQ(hipErrorNotReady, api_.stream_query(other_stream_));
  ReleaseStreamGate();
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(other_stream_));
  EXPECT_EQ(replacement_, destination_);
}

TEST_F(HipGraphMemoryApiTest,
       DisabledGraphCarriesItsPendingLaunchToAnotherStream) {
  ASSERT_EQ(hipSuccess,
            api_.stream_create(&other_stream_, hipStreamNonBlocking));
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_copy_1d(&node, graph_, nullptr, 0,
                                         destination_.data(), source_.data(),
                                         source_.size(), hipMemcpyHostToHost));
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
  HoldStreamAtGate();
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_EQ(hipSuccess, api_.launch(executable_, stream_));
  ASSERT_EQ(hipSuccess, api_.set_enabled(executable_, node, 0));
  ASSERT_EQ(hipSuccess, api_.launch(executable_, other_stream_));
  // Even an empty updated recording cannot discard the previous launch's
  // completion when replayed on a different stream.
  EXPECT_EQ(hipErrorNotReady, api_.stream_query(other_stream_));
  ReleaseStreamGate();
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(other_stream_));
  EXPECT_EQ(source_, destination_);
}

TEST_F(HipGraphMemoryApiTest, PitchedCopyCloneReplaysItsOwnStaging) {
  ASSERT_EQ(hipSuccess, api_.allocate(&device_, source_.size()));
  const hipMemcpy3DParms upload = {
      .srcArray = nullptr,
      .srcPos = {1, 0, 0},
      .srcPtr = {source_.data(), 8, 8, 3},
      .dstArray = nullptr,
      .dstPos = {1, 0, 0},
      .dstPtr = {device_, 8, 8, 3},
      .extent = {4, 2, 2},
      .kind = hipMemcpyHostToDevice,
  };
  hipGraphNode_t upload_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.add_copy(&upload_node, graph_, nullptr, 0, &upload));
  hipMemcpy3DParms download = upload;
  download.srcPtr.ptr = device_;
  download.dstPtr.ptr = destination_.data();
  download.kind = hipMemcpyDeviceToHost;
  hipGraphNode_t download_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.add_copy(&download_node, graph_, &upload_node, 1, &download));
  ASSERT_EQ(hipSuccess, api_.graph_clone(&clone_, graph_));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph_));
  graph_ = nullptr;
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, clone_, 0));
  for (int replay = 0; replay < 2; ++replay) {
    Replay();
    ASSERT_FALSE(HasFatalFailure());
    for (size_t i = 0; i < source_.size(); ++i) {
      const bool copied = (i >= 1 && i < 5) || (i >= 9 && i < 13) ||
                          (i >= 25 && i < 29) || (i >= 33 && i < 37);
      EXPECT_EQ(destination_[i], copied ? source_[i] : 0) << i;
    }
    source_ = replacement_;
  }
}

TEST_F(HipGraphMemoryApiTest, StagedCopiesFollowPublicEnableAndDeletion) {
  ASSERT_EQ(hipSuccess, api_.allocate(&device_, source_.size()));
  for (bool pitched : {false, true}) {
    SCOPED_TRACE(pitched);
    if (!graph_) {
      ASSERT_EQ(hipSuccess, api_.graph_create(&graph_, 0));
    }
    source_.fill(0x23);
    destination_.fill(0);
    hipGraphNode_t upload = nullptr;
    hipGraphNode_t download = nullptr;
    if (pitched) {
      hipMemcpy3DParms parameters = {
          .srcPtr = {source_.data(), 8, 8, 3},
          .dstPtr = {device_, 8, 8, 3},
          .extent = {4, 2, 2},
          .kind = hipMemcpyHostToDevice,
      };
      ASSERT_EQ(hipSuccess,
                api_.add_copy(&upload, graph_, nullptr, 0, &parameters));
      parameters.srcPtr.ptr = device_;
      parameters.dstPtr.ptr = destination_.data();
      parameters.kind = hipMemcpyDeviceToHost;
      ASSERT_EQ(hipSuccess,
                api_.add_copy(&download, graph_, &upload, 1, &parameters));
    } else {
      ASSERT_EQ(
          hipSuccess,
          api_.add_copy_1d(&upload, graph_, nullptr, 0, device_, source_.data(),
                           source_.size(), hipMemcpyHostToDevice));
      ASSERT_EQ(
          hipSuccess,
          api_.add_copy_1d(&download, graph_, &upload, 1, destination_.data(),
                           device_, source_.size(), hipMemcpyDeviceToHost));
    }
    ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
    Replay();
    ASSERT_FALSE(HasFatalFailure());
    const auto original = destination_;
    EXPECT_EQ(original[0], 0x23);

    // Disabled upload leaves device bytes unchanged; disabled download leaves
    // the application destination unchanged, including its host-side helper.
    ASSERT_EQ(hipSuccess, api_.set_enabled(executable_, upload, 0));
    source_.fill(0x54);
    Replay();
    ASSERT_FALSE(HasFatalFailure());
    EXPECT_EQ(destination_, original);
    ASSERT_EQ(hipSuccess, api_.set_enabled(executable_, download, 0));
    destination_.fill(0xCC);
    const auto untouched = destination_;
    Replay();
    ASSERT_FALSE(HasFatalFailure());
    EXPECT_EQ(destination_, untouched);
    ASSERT_EQ(hipSuccess, api_.set_enabled(executable_, upload, 1));
    ASSERT_EQ(hipSuccess, api_.set_enabled(executable_, download, 1));
    Replay();
    ASSERT_FALSE(HasFatalFailure());
    EXPECT_EQ(destination_[0], 0x54);

    ASSERT_EQ(hipSuccess, api_.destroy_node(download));
    ASSERT_EQ(hipSuccess, api_.instantiate(&sibling_, graph_, 0));
    destination_ = untouched;
    Replay(sibling_);
    ASSERT_FALSE(HasFatalFailure());
    EXPECT_EQ(destination_, untouched);
    // The original executable still owns its complete captured copy.
    Replay();
    ASSERT_FALSE(HasFatalFailure());
    EXPECT_EQ(destination_[0], 0x54);
    ASSERT_EQ(hipSuccess, api_.executable_destroy(sibling_));
    sibling_ = nullptr;
    ASSERT_EQ(hipSuccess, api_.executable_destroy(executable_));
    executable_ = nullptr;
    ASSERT_EQ(hipSuccess, api_.graph_destroy(graph_));
    graph_ = nullptr;
  }
}

TEST_F(HipGraphMemoryApiTest,
       AddedCopyDependenciesBracketHostPreparationAndCompletion) {
  ASSERT_EQ(hipSuccess, api_.allocate(&device_, source_.size()));
  // Create the consumer first so creation order cannot accidentally provide
  // the post-copy ordering that its later public dependency must establish.
  hipHostNodeParams consumer_parameters = {
      .fn = ObserveDestination,
      .userData = this,
  };
  hipGraphNode_t consumer = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.add_host(&consumer, graph_, nullptr, 0, &consumer_parameters));
  hipGraphNode_t upload = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_copy_1d(&upload, graph_, nullptr, 0, device_,
                                         source_.data(), source_.size(),
                                         hipMemcpyHostToDevice));
  hipHostNodeParams producer_parameters = {
      .fn = ReplaceSource,
      .userData = this,
  };
  hipGraphNode_t producer = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.add_host(&producer, graph_, nullptr, 0, &producer_parameters));
  hipGraphNode_t download = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.add_copy_1d(&download, graph_, &upload, 1, destination_.data(),
                             device_, source_.size(), hipMemcpyDeviceToHost));
  const hipGraphNode_t from[] = {producer, download};
  const hipGraphNode_t to[] = {upload, consumer};
  ASSERT_EQ(hipSuccess, api_.add_dependencies(graph_, from, to, 2));
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(observed_, replacement_);
}

TEST_F(HipGraphMemoryApiTest, CapturedCopiesSharePublicEnableAndDeletion) {
  ASSERT_EQ(hipSuccess, api_.allocate(&device_, source_.size()));
  ASSERT_EQ(hipSuccess,
            api_.begin_capture(stream_, hipStreamCaptureModeGlobal));
  ASSERT_EQ(hipSuccess, api_.copy_async(device_, source_.data(), source_.size(),
                                        hipMemcpyHostToDevice, stream_));
  ASSERT_EQ(hipSuccess,
            api_.copy_async(destination_.data(), device_, destination_.size(),
                            hipMemcpyDeviceToHost, stream_));
  ASSERT_EQ(hipSuccess, api_.end_capture(stream_, &clone_));
  std::array<hipGraphNode_t, 2> nodes = {};
  size_t count = nodes.size();
  ASSERT_EQ(hipSuccess, api_.get_nodes(clone_, nodes.data(), &count));
  ASSERT_EQ(count, nodes.size());
  hipGraphNode_t upload = nullptr;
  for (auto node : nodes) {
    hipMemcpy3DParms parameters = {};
    ASSERT_EQ(hipSuccess, api_.get_copy(node, &parameters));
    EXPECT_EQ(source_.size(), parameters.extent.width);
    if (parameters.kind == hipMemcpyHostToDevice) {
      EXPECT_EQ(device_, parameters.dstPtr.ptr);
      EXPECT_EQ(source_.data(), parameters.srcPtr.ptr);
      upload = node;
    } else {
      EXPECT_EQ(hipMemcpyDeviceToHost, parameters.kind);
      EXPECT_EQ(destination_.data(), parameters.dstPtr.ptr);
      EXPECT_EQ(device_, parameters.srcPtr.ptr);
    }
  }
  ASSERT_NE(nullptr, upload);
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, clone_, 0));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(destination_, source_);
  ASSERT_EQ(hipSuccess, api_.set_executable_copy(
                            executable_, upload, device_, replacement_.data(),
                            replacement_.size(), hipMemcpyHostToDevice));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(destination_, replacement_);
  hipMemcpy3DParms original = {};
  ASSERT_EQ(hipSuccess, api_.get_copy(upload, &original));
  EXPECT_EQ(source_.data(), original.srcPtr.ptr);
  // Disable both public operations: neither staging callback may touch the
  // destination. This also covers implementations returning nodes unordered.
  for (auto node : nodes) {
    ASSERT_EQ(hipSuccess, api_.set_enabled(executable_, node, 0));
  }
  destination_.fill(0xCC);
  const auto untouched = destination_;
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(destination_, untouched);
  for (auto node : nodes) {
    ASSERT_EQ(hipSuccess, api_.destroy_node(node));
  }
  ASSERT_EQ(hipSuccess, api_.instantiate(&sibling_, clone_, 0));
  Replay(sibling_);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(destination_, untouched);
}

TEST_F(HipGraphMemoryApiTest, RegisteredHostCopiesPreserveCaptureAndUpdates) {
  ASSERT_EQ(hipSuccess, api_.allocate(&device_, source_.size()));
  for (auto& buffer : host_buffers_) {
    ASSERT_EQ(hipSuccess, api_.allocate_host(&buffer, source_.size(),
                                             hipHostMallocDefault));
  }
  auto* source = static_cast<uint8_t*>(host_buffers_[0]);
  auto* destination = static_cast<uint8_t*>(host_buffers_[1]);
  std::memcpy(source, source_.data(), source_.size());
  std::memset(destination, 0, source_.size());
  ASSERT_EQ(hipSuccess,
            api_.begin_capture(stream_, hipStreamCaptureModeGlobal));
  ASSERT_EQ(hipSuccess, api_.copy_async(device_, source, source_.size(),
                                        hipMemcpyDefault, stream_));
  ASSERT_EQ(hipSuccess, api_.copy_async(destination, device_, source_.size(),
                                        hipMemcpyDeviceToHost, stream_));
  ASSERT_EQ(hipSuccess, api_.end_capture(stream_, &clone_));
  std::array<hipGraphNode_t, 2> nodes = {};
  size_t count = nodes.size();
  ASSERT_EQ(hipSuccess, api_.get_nodes(clone_, nodes.data(), &count));
  ASSERT_EQ(nodes.size(), count);
  hipGraphNode_t upload = nullptr;
  hipGraphNode_t download = nullptr;
  for (auto node : nodes) {
    hipMemcpy3DParms parameters = {};
    ASSERT_EQ(hipSuccess, api_.get_copy(node, &parameters));
    if (parameters.kind == hipMemcpyDefault) {
      EXPECT_EQ(source, parameters.srcPtr.ptr);
      EXPECT_EQ(device_, parameters.dstPtr.ptr);
      upload = node;
    } else {
      EXPECT_EQ(hipMemcpyDeviceToHost, parameters.kind);
      EXPECT_EQ(device_, parameters.srcPtr.ptr);
      EXPECT_EQ(destination, parameters.dstPtr.ptr);
      download = node;
    }
  }
  ASSERT_NE(nullptr, upload);
  ASSERT_NE(nullptr, download);
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, clone_, 0));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(0, std::memcmp(source, destination, source_.size()));

  constexpr size_t offset = 8;
  constexpr size_t length = 16;
  ASSERT_EQ(hipSuccess, api_.set_executable_copy(executable_, upload, device_,
                                                 source + offset, length,
                                                 hipMemcpyDefault));
  ASSERT_EQ(hipSuccess, api_.set_executable_copy(
                            executable_, download, destination + offset,
                            device_, length, hipMemcpyDeviceToHost));
  // Reject a registered range overrun without losing the accepted parameters.
  EXPECT_EQ(hipErrorInvalidValue,
            api_.set_executable_copy(executable_, upload, device_, source + 1,
                                     source_.size(), hipMemcpyDefault));
  std::memcpy(source, replacement_.data(), replacement_.size());
  std::memset(destination, 0xCC, source_.size());
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  for (size_t i = 0; i < source_.size(); ++i) {
    EXPECT_EQ(i >= offset && i < offset + length ? replacement_[i] : 0xCC,
              destination[i]);
  }
}

TEST_F(HipGraphMemoryApiTest, HostCopyCaptureDefersUntilGraphLaunch) {
  const auto untouched = destination_;
  ASSERT_EQ(hipSuccess,
            api_.begin_capture(stream_, hipStreamCaptureModeGlobal));
  ASSERT_EQ(hipSuccess,
            api_.copy_async(destination_.data(), source_.data(), source_.size(),
                            hipMemcpyHostToHost, stream_));
  ASSERT_EQ(hipSuccess, api_.end_capture(stream_, &clone_));
  EXPECT_EQ(untouched, destination_);
  hipGraphNode_t node = nullptr;
  size_t count = 1;
  ASSERT_EQ(hipSuccess, api_.get_nodes(clone_, &node, &count));
  ASSERT_EQ(1u, count);
  hipMemcpy3DParms parameters = {};
  ASSERT_EQ(hipSuccess, api_.get_copy(node, &parameters));
  EXPECT_EQ(hipMemcpyHostToHost, parameters.kind);
  EXPECT_EQ(source_.data(), parameters.srcPtr.ptr);
  EXPECT_EQ(destination_.data(), parameters.dstPtr.ptr);
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, clone_, 0));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(source_, destination_);
}

TEST_F(HipGraphMemoryApiTest, CopyDirectionUpdatesPreserveRejectedParameters) {
  ASSERT_EQ(hipSuccess, api_.allocate(&device_, source_.size() * 2));
  auto* source = static_cast<uint8_t*>(device_);
  auto* destination = source + source_.size();
  ASSERT_EQ(hipSuccess, api_.copy(source, replacement_.data(),
                                  replacement_.size(), hipMemcpyHostToDevice));
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.add_copy_1d(&node, graph_, nullptr, 0, destination, source,
                             source_.size(), hipMemcpyDeviceToDevice));
  for (auto kind : {hipMemcpyHostToDevice, hipMemcpyDeviceToHost,
                    hipMemcpyDeviceToDevice, hipMemcpyHostToDevice}) {
    SCOPED_TRACE(kind);
    void* copy_destination =
        kind == hipMemcpyDeviceToHost ? destination_.data() : destination;
    const void* copy_source =
        kind == hipMemcpyHostToDevice ? source_.data() : source;
    ASSERT_EQ(hipSuccess, api_.set_copy(node, copy_destination, copy_source,
                                        source_.size(), kind));
    // The registered device range is smaller than this request. Reject it
    // before changing the public query or either half of the staged operation.
    EXPECT_EQ(hipErrorInvalidValue,
              api_.set_copy(node, copy_destination, copy_source,
                            source_.size() * 2 + 1, kind));
    hipMemcpy3DParms parameters = {};
    ASSERT_EQ(hipSuccess, api_.get_copy(node, &parameters));
    EXPECT_EQ(copy_destination, parameters.dstPtr.ptr);
    EXPECT_EQ(copy_source, parameters.srcPtr.ptr);
    EXPECT_EQ(source_.size(), parameters.extent.width);
    EXPECT_EQ(kind, parameters.kind);
    ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
    Replay();
    ASSERT_FALSE(HasFatalFailure());
    if (kind != hipMemcpyDeviceToHost) {
      ASSERT_EQ(hipSuccess,
                api_.copy(destination_.data(), destination, destination_.size(),
                          hipMemcpyDeviceToHost));
    }
    EXPECT_EQ(kind == hipMemcpyHostToDevice ? source_ : replacement_,
              destination_);
    ASSERT_EQ(hipSuccess, api_.executable_destroy(executable_));
    executable_ = nullptr;
  }
}

TEST_F(HipGraphMemoryApiTest,
       HostParametersCannotReplaceInternalCopyOperations) {
  hipGraphNode_t copy_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_copy_1d(&copy_node, graph_, nullptr, 0,
                                         destination_.data(), source_.data(),
                                         source_.size(), hipMemcpyHostToHost));
  hipHostNodeParams parameters = {Increment, &callbacks_[0]};
  hipGraphNode_t host_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.add_host(&host_node, graph_, &copy_node, 1, &parameters));
  EXPECT_EQ(hipErrorInvalidValue, api_.set_host(copy_node, &parameters));
  hipHostNodeParams queried = {};
  EXPECT_EQ(hipErrorInvalidValue, api_.get_host(copy_node, &queried));
  ASSERT_EQ(hipSuccess, api_.get_host(host_node, &queried));
  EXPECT_EQ(queried.fn, Increment);
  EXPECT_EQ(queried.userData, &callbacks_[0]);
  parameters.userData = &callbacks_[1];
  ASSERT_EQ(hipSuccess, api_.set_host(host_node, &parameters));
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.set_executable_host(executable_, copy_node, &parameters));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(callbacks_, (std::array<int, 3>{0, 1, 0}));
  EXPECT_EQ(destination_, source_);
  parameters.userData = &callbacks_[2];
  ASSERT_EQ(hipSuccess,
            api_.set_executable_host(executable_, host_node, &parameters));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(callbacks_, (std::array<int, 3>{0, 1, 1}));
  ASSERT_EQ(hipSuccess, api_.get_host(host_node, &queried));
  EXPECT_EQ(queried.userData, &callbacks_[1]);
}

TEST_F(HipGraphMemoryApiTest, PeerCopiesCompleteAcrossCompatibilityContexts) {
  int ordinal = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&ordinal));
  ASSERT_EQ(hipSuccess, api_.allocate(&device_, source_.size()));
  ASSERT_EQ(hipSuccess, api_.create_context(&remote_context_, 0, ordinal));
  for (auto& buffer : remote_buffers_) {
    ASSERT_EQ(hipSuccess, api_.allocate(&buffer, source_.size()));
  }

  // Exercise source-local, destination-local, and neither-local graph
  // representations with one physical device and two compatibility contexts.
  for (int placement = 0; placement < 3; ++placement) {
    hipDeviceptr_t source = placement == 0 ? device_ : remote_buffers_[0];
    hipDeviceptr_t destination = placement == 1 ? device_ : remote_buffers_[1];
    ASSERT_EQ(hipSuccess, api_.set_context(placement == 0 ? primary_context_
                                                          : remote_context_));
    ASSERT_EQ(hipSuccess, api_.copy(source, source_.data(), source_.size(),
                                    hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, api_.set_context(primary_context_));
    hipGraphNode_t node = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.add_copy_1d(&node, graph_, nullptr, 0, destination, source,
                               source_.size(), hipMemcpyDeviceToDevice));
    ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
    ASSERT_EQ(hipSuccess, api_.set_context(placement == 1 ? primary_context_
                                                          : remote_context_));
    destination_.fill(0xCC);
    const auto untouched = destination_;
    ASSERT_EQ(hipSuccess,
              api_.copy(destination, destination_.data(), destination_.size(),
                        hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, api_.set_context(primary_context_));
    ASSERT_EQ(hipSuccess, api_.set_enabled(executable_, node, 0));
    Replay();
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_EQ(hipSuccess, api_.set_context(placement == 1 ? primary_context_
                                                          : remote_context_));
    ASSERT_EQ(hipSuccess,
              api_.copy(destination_.data(), destination, destination_.size(),
                        hipMemcpyDeviceToHost));
    EXPECT_EQ(destination_, untouched) << placement;
    ASSERT_EQ(hipSuccess, api_.set_context(primary_context_));
    ASSERT_EQ(hipSuccess, api_.set_enabled(executable_, node, 1));
    Replay();
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_EQ(hipSuccess, api_.set_context(placement == 1 ? primary_context_
                                                          : remote_context_));
    ASSERT_EQ(hipSuccess,
              api_.copy(destination_.data(), destination, destination_.size(),
                        hipMemcpyDeviceToHost));
    EXPECT_EQ(destination_, source_) << placement;
    ASSERT_EQ(hipSuccess, api_.set_context(primary_context_));
    ASSERT_EQ(hipSuccess,
              api_.set_executable_copy(executable_, node,
                                       static_cast<uint8_t*>(destination) + 16,
                                       static_cast<const uint8_t*>(source) + 8,
                                       32, hipMemcpyDeviceToDevice));
    ASSERT_EQ(hipSuccess, api_.set_context(placement == 1 ? primary_context_
                                                          : remote_context_));
    destination_.fill(0xA7);
    auto expected = destination_;
    for (size_t i = 0; i < 32; ++i) {
      expected[16 + i] = source_[8 + i];
    }
    ASSERT_EQ(hipSuccess,
              api_.copy(destination, destination_.data(), destination_.size(),
                        hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, api_.set_context(primary_context_));
    Replay();
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_EQ(hipSuccess, api_.set_context(placement == 1 ? primary_context_
                                                          : remote_context_));
    ASSERT_EQ(hipSuccess,
              api_.copy(destination_.data(), destination, destination_.size(),
                        hipMemcpyDeviceToHost));
    EXPECT_EQ(destination_, expected) << placement;
    ASSERT_EQ(hipSuccess, api_.set_context(primary_context_));
    ASSERT_EQ(hipSuccess, api_.executable_destroy(executable_));
    executable_ = nullptr;
    ASSERT_EQ(hipSuccess, api_.graph_destroy(graph_));
    graph_ = nullptr;
    ASSERT_EQ(hipSuccess, api_.graph_create(&graph_, 0));
    source_ = replacement_;
    destination_.fill(0);
  }
}

}  // namespace
