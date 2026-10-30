// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

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
  // Allocates a device buffer for graph transfers.
  decltype(&hipMalloc) allocate = nullptr;
  // Releases the allocation after graph completion.
  decltype(&hipFree) free = nullptr;
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
  // Adds a pitched multidimensional graph copy.
  decltype(&hipGraphAddMemcpyNode) add_copy = nullptr;
  // Updates a template's contiguous copy parameters.
  decltype(&hipGraphMemcpyNodeSetParams1D) set_copy = nullptr;
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
    RESOLVE(allocate, "hipMalloc");
    RESOLVE(free, "hipFree");
    RESOLVE(graph_create, "hipGraphCreate");
    RESOLVE(graph_destroy, "hipGraphDestroy");
    RESOLVE(graph_clone, "hipGraphClone");
    RESOLVE(find_clone_node, "hipGraphNodeFindInClone");
    RESOLVE(instantiate, "hipGraphInstantiateWithFlags");
    RESOLVE(executable_destroy, "hipGraphExecDestroy");
    RESOLVE(launch, "hipGraphLaunch");
    RESOLVE(add_copy_1d, "hipGraphAddMemcpyNode1D");
    RESOLVE(add_copy, "hipGraphAddMemcpyNode");
    RESOLVE(set_copy, "hipGraphMemcpyNodeSetParams1D");
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
    if (primary_context_) {
      EXPECT_EQ(hipSuccess, api_.set_context(primary_context_));
    }
    if (stream_) {
      EXPECT_EQ(hipSuccess, api_.stream_synchronize(stream_));
    }
    if (executable_) {
      EXPECT_EQ(hipSuccess, api_.executable_destroy(executable_));
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
    if (device_) {
      EXPECT_EQ(hipSuccess, api_.free(device_));
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

  void Replay() {
    ASSERT_EQ(hipSuccess, api_.launch(executable_, stream_));
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
  }

  static void Increment(void* user_data) { ++*static_cast<int*>(user_data); }

  // Exact frontend shared object and API reused for the whole suite.
  static hrx::hip::testing::HipDso dso_;
  // Function table resolved only from that shared object.
  static HipGraphApi api_;
  // Launch stream joined on every test exit.
  hipStream_t stream_ = nullptr;
  // Original template, or null after explicit destruction.
  hipGraph_t graph_ = nullptr;
  // Clone whose payload lifetime is independent of the original template.
  hipGraph_t clone_ = nullptr;
  // Compiled graph snapshot.
  hipGraphExec_t executable_ = nullptr;
  // Device endpoint retained through all graph launches.
  hipDeviceptr_t device_ = nullptr;
  // Borrowed primary context owning the graph and launch stream.
  hipCtx_t primary_context_ = nullptr;
  // Secondary compatibility context sharing the suite's native GPU device.
  hipCtx_t remote_context_ = nullptr;
  // Peer copy endpoints retained through graph completion and cleanup.
  std::array<hipDeviceptr_t, 2> remote_buffers_ = {};
  // Initial copy data retained through teardown, including assertion failures.
  std::array<uint8_t, 64> source_ = {};
  // Replacement data used by template and executable updates.
  std::array<uint8_t, 64> replacement_ = {};
  // Copy destination retained through teardown.
  std::array<uint8_t, 64> destination_ = {};
  // Application callback outputs retained through teardown.
  std::array<int, 3> callbacks_ = {};
};

hrx::hip::testing::HipDso HipGraphMemoryApiTest::dso_;
HipGraphApi HipGraphMemoryApiTest::api_;

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
    Replay();
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_EQ(hipSuccess, api_.set_context(placement == 1 ? primary_context_
                                                          : remote_context_));
    ASSERT_EQ(hipSuccess,
              api_.copy(destination_.data(), destination, destination_.size(),
                        hipMemcpyDeviceToHost));
    EXPECT_EQ(destination_, source_) << placement;
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
