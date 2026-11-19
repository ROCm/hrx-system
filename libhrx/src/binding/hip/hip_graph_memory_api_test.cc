// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <semaphore>
#include <vector>

#include "binding/hip/api.h"
#include "binding/hip/hip_graph_memory_test.h"
#include "iree/testing/gtest.h"

namespace {

using hrx::hip::testing::HipGraphMemoryApiTest;

TEST_F(HipGraphMemoryApiTest,
       UnlaunchedExecutableDestructionIgnoresPendingWork) {
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
  HoldStreamAtGate();
  ASSERT_FALSE(HasFatalFailure());

  ASSERT_EQ(hipSuccess, api_.executable_destroy(executable_));
  executable_ = nullptr;
  EXPECT_EQ(hipErrorNotReady, api_.stream_query(stream_));
}

TEST_F(HipGraphMemoryApiTest, ExecutableDestructionIgnoresLaterStreamWork) {
  hipHostNodeParams host = {.fn = Increment, .userData = &callbacks_[0]};
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_host(&node, graph_, nullptr, 0, &host));
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  HoldStreamAtGate();
  ASSERT_FALSE(HasFatalFailure());

  // The executable owns its launch point, not the stream's current tail.
  ASSERT_EQ(hipSuccess, api_.executable_destroy(executable_));
  executable_ = nullptr;
  EXPECT_EQ(1, callbacks_[0]);
  EXPECT_EQ(hipErrorNotReady, api_.stream_query(stream_));
}

TEST_F(HipGraphMemoryApiTest, ExecutableDestructionIgnoresIndependentContexts) {
  hipHostNodeParams host = {.fn = Increment, .userData = &callbacks_[0]};
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_host(&node, graph_, nullptr, 0, &host));
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
  Replay();
  ASSERT_FALSE(HasFatalFailure());

  // Both contexts borrow the suite's one native device. Only their timelines
  // differ; graph destruction must not join the other context's gated work.
  int ordinal = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&ordinal));
  ASSERT_EQ(hipSuccess, api_.create_context(&remote_context_, 0, ordinal));
  ASSERT_EQ(hipSuccess,
            api_.stream_create(&other_stream_, hipStreamNonBlocking));
  HoldStreamAtGate(other_stream_);
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_EQ(hipSuccess, api_.set_context(primary_context_));

  ASSERT_EQ(hipSuccess, api_.executable_destroy(executable_));
  executable_ = nullptr;
  EXPECT_EQ(1, callbacks_[0]);
  EXPECT_EQ(hipErrorNotReady, api_.stream_query(other_stream_));
}

TEST_F(HipGraphMemoryApiTest,
       ExecutableDestructionJoinsPendingUpdatedLaunches) {
  hipHostNodeParams host = {.fn = Increment, .userData = &callbacks_[0]};
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_host(&node, graph_, nullptr, 0, &host));
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
  ASSERT_EQ(hipSuccess,
            api_.stream_create(&other_stream_, hipStreamNonBlocking));
  HoldStreamAtGate();
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_EQ(hipSuccess, api_.launch(executable_, stream_));
  host.userData = &callbacks_[1];
  ASSERT_EQ(hipSuccess, api_.set_executable_host(executable_, node, &host));
  ASSERT_EQ(hipSuccess, api_.launch(executable_, other_stream_));
  EXPECT_EQ(hipErrorNotReady, api_.stream_query(other_stream_));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph_));
  graph_ = nullptr;

  ReleaseStreamGate();
  ASSERT_EQ(hipSuccess, api_.executable_destroy(executable_));
  executable_ = nullptr;
  EXPECT_EQ(callbacks_, (std::array<int, 3>{1, 1, 0}));
  EXPECT_EQ(hipSuccess, api_.stream_query(stream_));
  EXPECT_EQ(hipSuccess, api_.stream_query(other_stream_));
}

TEST_F(HipGraphMemoryApiTest, PendingContiguousCopiesOwnTheirStaging) {
  CheckPendingStagedCopyUpdates(false);
}

TEST_F(HipGraphMemoryApiTest, PendingPitchedCopiesOwnTheirStaging) {
  CheckPendingStagedCopyUpdates(true);
}

TEST_F(HipGraphMemoryApiTest, RejectedCopiesLeaveTheOriginalGraphUsable) {
  ASSERT_EQ(hipSuccess, api_.allocate(&device_, source_.size()));
  ASSERT_EQ(hipSuccess, api_.graph_create(&clone_, 0));
  hipGraphNode_t foreign = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_empty(&foreign, clone_, nullptr, 0));
  for (const auto direction : {hipMemcpyHostToDevice, hipMemcpyDeviceToHost}) {
    SCOPED_TRACE(direction);
    void* destination =
        direction == hipMemcpyHostToDevice ? device_ : destination_.data();
    const void* source =
        direction == hipMemcpyHostToDevice ? source_.data() : device_;
    hipGraphNode_t rejected = nullptr;
    EXPECT_EQ(hipErrorInvalidValue,
              api_.add_copy_1d(&rejected, graph_, &foreign, 1, destination,
                               source, source_.size(), direction));
    const hipMemcpy3DParms params = {
        .srcPtr = {const_cast<void*>(source), 16, 16, 2},
        .dstPtr = {destination, 16, 16, 2},
        .extent = {16, 2, 2},
        .kind = direction,
    };
    EXPECT_EQ(hipErrorInvalidValue,
              api_.add_copy(&rejected, graph_, &foreign, 1, &params));
    size_t count = 0;
    ASSERT_EQ(hipSuccess, api_.get_nodes(graph_, nullptr, &count));
    EXPECT_EQ(0u, count);
  }
  hipGraphNode_t upload = nullptr;
  hipGraphNode_t download = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_copy_1d(&upload, graph_, nullptr, 0, device_,
                                         source_.data(), source_.size(),
                                         hipMemcpyHostToDevice));
  ASSERT_EQ(
      hipSuccess,
      api_.add_copy_1d(&download, graph_, &upload, 1, destination_.data(),
                       device_, destination_.size(), hipMemcpyDeviceToHost));
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
  Replay();
  EXPECT_EQ(source_, destination_);
}

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

TEST_F(HipGraphMemoryApiTest, ChildInsertionAndParentCloneOwnTheirSnapshots) {
  ASSERT_EQ(hipSuccess, api_.graph_create(&clone_, 0));
  hipHostNodeParams host = {.fn = Increment, .userData = &callbacks_[0]};
  hipGraphNode_t source_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_host(&source_node, clone_, nullptr, 0, &host));
  hipGraphNode_t child_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.add_child(&child_node, graph_, nullptr, 0, clone_));
  hipGraph_t embedded = nullptr;
  ASSERT_EQ(hipSuccess, api_.get_child(child_node, &embedded));
  EXPECT_NE(embedded, clone_);

  // Insertion captures the child before any executable exists.
  host.userData = &callbacks_[1];
  ASSERT_EQ(hipSuccess, api_.set_host(source_node, &host));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(clone_));
  clone_ = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_clone(&clone_, graph_));
  hipGraphNode_t cloned_child_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.find_clone_node(&cloned_child_node, child_node, clone_));
  hipGraph_t cloned_child = nullptr;
  ASSERT_EQ(hipSuccess, api_.get_child(cloned_child_node, &cloned_child));
  EXPECT_NE(cloned_child, embedded);
  hipGraphNode_t cloned_host = nullptr;
  size_t count = 1;
  ASSERT_EQ(hipSuccess, api_.get_nodes(cloned_child, &cloned_host, &count));
  ASSERT_EQ(count, 1u);
  host.userData = &callbacks_[2];
  ASSERT_EQ(hipSuccess, api_.set_host(cloned_host, &host));
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
  ASSERT_EQ(hipSuccess, api_.instantiate(&sibling_, clone_, 0));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  Replay(sibling_);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(callbacks_, (std::array<int, 3>{1, 0, 1}));

  HoldStreamAtGate();
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_EQ(hipSuccess, api_.launch(executable_, stream_));
  hipGraphNodeParams update = {
      .type = hipGraphNodeTypeGraph,
      .graph = {.graph = cloned_child},
  };
  ASSERT_EQ(hipSuccess,
            api_.set_executable_node(executable_, child_node, &update));
  ReleaseStreamGate();
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
  EXPECT_EQ(callbacks_, (std::array<int, 3>{2, 0, 1}));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(callbacks_, (std::array<int, 3>{2, 0, 2}));
}

TEST_F(HipGraphMemoryApiTest, BorrowedChildHandleFollowsItsOwnerLifetime) {
  ASSERT_EQ(hipSuccess, api_.graph_create(&clone_, 0));
  hipHostNodeParams host = {.fn = Increment, .userData = &callbacks_[0]};
  hipGraphNode_t host_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_host(&host_node, clone_, nullptr, 0, &host));
  hipGraphNode_t child_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.add_child(&child_node, graph_, nullptr, 0, clone_));
  hipGraph_t embedded = nullptr;
  ASSERT_EQ(hipSuccess, api_.get_child(child_node, &embedded));
  hipGraph_t repeated = nullptr;
  ASSERT_EQ(hipSuccess, api_.get_child(child_node, &repeated));
  EXPECT_EQ(embedded, repeated);
  EXPECT_EQ(hipErrorInvalidValue, api_.graph_destroy(embedded));
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, embedded, 0));
  ASSERT_EQ(hipSuccess, api_.destroy_node(child_node));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(callbacks_[0], 1);
  ASSERT_EQ(hipSuccess, api_.executable_destroy(executable_));
  executable_ = nullptr;
  // The public boundary rejects the expired borrowed handle without reading
  // its released object. No native operation is issued for this request.
  EXPECT_EQ(hipErrorInvalidValue, api_.instantiate(&executable_, embedded, 0));
  EXPECT_EQ(executable_, nullptr);
}

TEST_F(HipGraphMemoryApiTest, CaptureTransfersItsExposedBorrowedGraph) {
  ASSERT_EQ(hipSuccess,
            api_.begin_capture(stream_, hipStreamCaptureModeGlobal));
  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
  hipGraph_t captured = nullptr;
  const hipError_t query = api_.capture_info(stream_, &capture_status, nullptr,
                                             &captured, nullptr, nullptr);
  EXPECT_EQ(query, hipSuccess);
  EXPECT_EQ(capture_status, hipStreamCaptureStatusActive);
  if (query == hipSuccess && captured) {
    EXPECT_EQ(hipErrorInvalidValue, api_.graph_destroy(captured));
    hipHostNodeParams host = {.fn = Increment, .userData = &callbacks_[0]};
    hipGraphNode_t host_node = nullptr;
    EXPECT_EQ(hipSuccess,
              api_.add_host(&host_node, captured, nullptr, 0, &host));
  }
  ASSERT_EQ(hipSuccess, api_.end_capture(stream_, &clone_));
  ASSERT_EQ(captured, clone_);
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, clone_, 0));
  // EndCapture makes the exposed handle caller-owned and destroyable.
  ASSERT_EQ(hipSuccess, api_.graph_destroy(clone_));
  clone_ = nullptr;
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(callbacks_[0], 1);
}

TEST_F(HipGraphMemoryApiTest, CaptureRejectsUnsupportedDependencyData) {
  hipGraphNode_t root = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_empty(&root, graph_, nullptr, 0));
  const uint64_t dependency_data = 1;
  EXPECT_EQ(hipErrorNotSupported, api_.begin_capture_to_graph(
                                      stream_, graph_, &root, &dependency_data,
                                      1, hipStreamCaptureModeGlobal));
  hipStreamCaptureStatus status = hipStreamCaptureStatusActive;
  ASSERT_EQ(hipSuccess, api_.capture_info(stream_, &status, nullptr, nullptr,
                                          nullptr, nullptr));
  EXPECT_EQ(hipStreamCaptureStatusNone, status);
  ASSERT_EQ(hipSuccess,
            api_.begin_capture_to_graph(stream_, graph_, &root, nullptr, 1,
                                        hipStreamCaptureModeGlobal));
  hipGraph_t captured = nullptr;
  ASSERT_EQ(hipSuccess, api_.end_capture(stream_, &captured));
  EXPECT_EQ(graph_, captured);
}

TEST_F(HipGraphMemoryApiTest, NestedAllocationsKeepTheirOriginalAdmission) {
  int device = 0;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  hipMemAllocNodeParams allocation = {};
  allocation.poolProps.allocType = hipMemAllocationTypePinned;
  allocation.poolProps.location.type = hipMemLocationTypeDevice;
  allocation.poolProps.location.id = device;
  allocation.bytesize = source_.size();
  hipGraphNode_t allocation_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_allocation(&allocation_node, graph_, nullptr,
                                            0, &allocation));
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

  ASSERT_EQ(hipSuccess, api_.graph_create(&clone_, 0));
  hipGraphNode_t middle_child = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.add_child(&middle_child, clone_, nullptr, 0, graph_));
  // A retained allocation template must not become its own descendant owner.
  hipGraphNode_t rejected = nullptr;
  EXPECT_EQ(hipErrorInvalidValue,
            api_.add_child(&rejected, graph_, nullptr, 0, clone_));
  EXPECT_EQ(rejected, nullptr);
  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph_));
  graph_ = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph_, 0));
  hipGraphNode_t outer_child = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.add_child(&outer_child, graph_, nullptr, 0, clone_));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(clone_));
  clone_ = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_clone(&clone_, graph_));

  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
  EXPECT_EQ(hipErrorNotSupported, api_.instantiate(&sibling_, graph_, 0));
  EXPECT_EQ(sibling_, nullptr);
  EXPECT_EQ(hipErrorNotSupported, api_.instantiate(&sibling_, clone_, 0));
  EXPECT_EQ(sibling_, nullptr);
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(destination_, source_);

  // Recompilation holds old and candidate children simultaneously under one
  // logical executable identity, without granting another executable access.
  hipGraphNode_t error_node = nullptr;
  hipGraphExecUpdateResult result;
  ASSERT_EQ(hipSuccess,
            api_.update_executable(executable_, clone_, &error_node, &result));
  EXPECT_EQ(result, hipGraphExecUpdateSuccess);
  destination_.fill(0);
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(destination_, source_);
  ASSERT_EQ(hipSuccess, api_.executable_destroy(executable_));
  executable_ = nullptr;
  ASSERT_EQ(hipSuccess, api_.instantiate(&sibling_, clone_, 0));
  source_ = replacement_;
  destination_.fill(0);
  Replay(sibling_);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(destination_, replacement_);
}

TEST_F(HipGraphMemoryApiTest, FailedChildCompilationReleasesParentAdmission) {
  ASSERT_EQ(hipSuccess, api_.graph_create(&clone_, 0));
  int device = 0;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  hipMemAllocNodeParams allocation = {};
  allocation.poolProps.allocType = hipMemAllocationTypePinned;
  allocation.poolProps.location.type = hipMemLocationTypeDevice;
  allocation.poolProps.location.id = device;
  allocation.bytesize = source_.size();
  for (auto graph : {graph_, clone_}) {
    hipGraphNode_t allocation_node = nullptr;
    ASSERT_EQ(hipSuccess, api_.add_allocation(&allocation_node, graph, nullptr,
                                              0, &allocation));
    hipGraphNode_t free_node = nullptr;
    ASSERT_EQ(hipSuccess, api_.add_free(&free_node, graph, &allocation_node, 1,
                                        allocation.dptr));
  }
  ASSERT_EQ(hipSuccess, api_.instantiate(&sibling_, clone_, 0));
  hipGraphNode_t child_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.add_child(&child_node, graph_, nullptr, 0, clone_));
  // The parent acquires its own allocation before compilation discovers that
  // the child belongs to an independent executable.
  EXPECT_EQ(hipErrorNotSupported, api_.instantiate(&executable_, graph_, 0));
  EXPECT_EQ(executable_, nullptr);
  ASSERT_EQ(hipSuccess, api_.executable_destroy(sibling_));
  sibling_ = nullptr;
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
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
  ASSERT_EQ(hipSuccess, api_.graph_memory_trim(device));
  uint64_t used_before = 0;
  uint64_t reserved_before = 0;
  ASSERT_EQ(hipSuccess,
            api_.graph_memory_attribute(device, hipGraphMemAttrUsedMemCurrent,
                                        &used_before));
  ASSERT_EQ(hipSuccess,
            api_.graph_memory_attribute(
                device, hipGraphMemAttrReservedMemCurrent, &reserved_before));
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
  uint64_t used = 0;
  uint64_t reserved = 0;
  ASSERT_EQ(hipSuccess, api_.graph_memory_attribute(
                            device, hipGraphMemAttrUsedMemCurrent, &used));
  ASSERT_EQ(hipSuccess,
            api_.graph_memory_attribute(
                device, hipGraphMemAttrReservedMemCurrent, &reserved));
  EXPECT_EQ(used_before, used);
  EXPECT_EQ(reserved_before, reserved);
  Replay();
  EXPECT_EQ(source_, destination_);
  ASSERT_EQ(hipSuccess, api_.graph_memory_attribute(
                            device, hipGraphMemAttrUsedMemCurrent, &used));
  ASSERT_EQ(hipSuccess,
            api_.graph_memory_attribute(
                device, hipGraphMemAttrReservedMemCurrent, &reserved));
  EXPECT_EQ(used_before, used);
  EXPECT_GE(reserved, reserved_before + allocation.bytesize);
  ASSERT_EQ(hipSuccess, api_.graph_memory_trim(device));
  ASSERT_EQ(hipSuccess,
            api_.graph_memory_attribute(
                device, hipGraphMemAttrReservedMemCurrent, &reserved));
  EXPECT_EQ(reserved_before, reserved);
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

TEST_F(HipGraphMemoryApiTest, ExternalFreeAllowsAllocationGraphReplay) {
  void* pointer = nullptr;
  BuildAllocationGraph(0, &pointer);
  ASSERT_FALSE(HasFatalFailure());
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  device_ = pointer;
  EXPECT_EQ(source_, destination_);
  EXPECT_EQ(hipErrorInvalidValue, api_.launch(executable_, stream_));
  EXPECT_EQ(hipErrorInvalidValue,
            api_.free_async(static_cast<uint8_t*>(pointer) + 1, stream_));
  ASSERT_EQ(hipSuccess, api_.free_async(pointer, stream_));
  device_ = nullptr;
  source_ = replacement_;
  destination_.fill(0);
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  device_ = pointer;
  EXPECT_EQ(source_, destination_);
}

TEST_F(HipGraphMemoryApiTest, LiveAllocationSurvivesGraphDestruction) {
  void* pointer = nullptr;
  BuildAllocationGraph(0, &pointer);
  ASSERT_FALSE(HasFatalFailure());
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  device_ = pointer;
  ASSERT_EQ(hipSuccess, api_.executable_destroy(executable_));
  executable_ = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph_));
  graph_ = nullptr;
  destination_.fill(0);
  ASSERT_EQ(hipSuccess, api_.copy(destination_.data(), pointer,
                                  destination_.size(), hipMemcpyDeviceToHost));
  EXPECT_EQ(source_, destination_);
  ASSERT_EQ(hipSuccess, api_.free(pointer));
  device_ = nullptr;
}

TEST_F(HipGraphMemoryApiTest, GraphAllocationCanBeFreedFromAnotherContext) {
  void* pointer = nullptr;
  BuildAllocationGraph(0, &pointer);
  ASSERT_FALSE(HasFatalFailure());
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  device_ = pointer;
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  ASSERT_EQ(hipSuccess, api_.create_context(&remote_context_, 0, device));
  ASSERT_EQ(hipSuccess, api_.free(pointer));
  device_ = nullptr;
  ASSERT_EQ(hipSuccess, api_.set_context(primary_context_));
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  device_ = pointer;
  EXPECT_EQ(source_, destination_);
}

TEST_F(HipGraphMemoryApiTest, CompletedBackingIsSharedAndTrimmed) {
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  ASSERT_EQ(hipSuccess, api_.graph_memory_trim(device));
  uint64_t baseline = 0;
  ASSERT_EQ(hipSuccess,
            api_.graph_memory_attribute(
                device, hipGraphMemAttrReservedMemCurrent, &baseline));
  void* first = nullptr;
  BuildAllocationGraph(0, &first);
  ASSERT_FALSE(HasFatalFailure());
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  device_ = first;
  uint64_t first_reserved = 0;
  ASSERT_EQ(hipSuccess,
            api_.graph_memory_attribute(
                device, hipGraphMemAttrReservedMemCurrent, &first_reserved));
  EXPECT_GT(first_reserved, baseline);
  ASSERT_EQ(hipSuccess, api_.free(first));
  device_ = nullptr;
  clone_ = graph_;
  graph_ = nullptr;
  sibling_ = executable_;
  executable_ = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_create(&graph_, 0));
  void* second = nullptr;
  BuildAllocationGraph(0, &second);
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_NE(first, second);
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  device_ = second;
  EXPECT_EQ(source_, destination_);
  ASSERT_EQ(hipSuccess, api_.graph_memory_trim(device));
  uint64_t second_reserved = 0;
  ASSERT_EQ(hipSuccess,
            api_.graph_memory_attribute(
                device, hipGraphMemAttrReservedMemCurrent, &second_reserved));
  EXPECT_EQ(first_reserved, second_reserved);
  ASSERT_EQ(hipSuccess, api_.free(second));
  device_ = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_memory_trim(device));
  ASSERT_EQ(hipSuccess,
            api_.graph_memory_attribute(
                device, hipGraphMemAttrReservedMemCurrent, &second_reserved));
  EXPECT_EQ(baseline, second_reserved);
}

TEST_F(HipGraphMemoryApiTest, PendingFreeDoesNotBlockIndependentAllocation) {
  void* first = nullptr;
  BuildAllocationGraph(0, &first);
  ASSERT_FALSE(HasFatalFailure());
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  device_ = first;
  HoldStreamAtGate();
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_EQ(hipSuccess, api_.free_async(first, stream_));
  device_ = nullptr;

  ASSERT_EQ(hipSuccess,
            api_.stream_create(&other_stream_, hipStreamNonBlocking));
  ASSERT_EQ(hipSuccess, api_.graph_create(&clone_, 0));
  int device = -1;
  ASSERT_EQ(hipSuccess, api_.get_device(&device));
  hipMemAllocNodeParams allocation = {};
  allocation.poolProps.allocType = hipMemAllocationTypePinned;
  allocation.poolProps.location.type = hipMemLocationTypeDevice;
  allocation.poolProps.location.id = device;
  allocation.bytesize = source_.size();
  hipGraphNode_t allocation_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_allocation(&allocation_node, clone_, nullptr,
                                            0, &allocation));
  const hipMemsetParams fill = {
      .dst = allocation.dptr,
      .elementSize = 1,
      .height = 1,
      .value = 0x35,
      .width = source_.size(),
  };
  hipGraphNode_t fill_node = nullptr;
  ASSERT_EQ(hipSuccess,
            api_.add_fill(&fill_node, clone_, &allocation_node, 1, &fill));
  ASSERT_EQ(hipSuccess, api_.instantiate(&sibling_, clone_, 0));
  ASSERT_EQ(hipSuccess, api_.launch(sibling_, other_stream_));
  device_ = allocation.dptr;
  // No host callback is part of this graph. It can finish while the first
  // allocation's unmap is still waiting for an application callback.
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(other_stream_));
  EXPECT_EQ(hipErrorNotReady, api_.stream_query(stream_));
  ReleaseStreamGate();
  ASSERT_EQ(hipSuccess, api_.copy(destination_.data(), device_,
                                  destination_.size(), hipMemcpyDeviceToHost));
  for (uint8_t value : destination_) {
    EXPECT_EQ(0x35, value);
  }
}

TEST_F(HipGraphMemoryApiTest, AutoFreeOrdersUnmapBeforeReallocation) {
  void* pointer = nullptr;
  BuildAllocationGraph(hipGraphInstantiateFlagAutoFreeOnLaunch, &pointer);
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_EQ(hipSuccess, api_.launch(executable_, stream_));
  device_ = pointer;
  ASSERT_EQ(hipSuccess, api_.launch(executable_, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
  EXPECT_EQ(source_, destination_);
}

TEST_F(HipGraphMemoryApiTest, AnotherGraphCanFreeAnAllocation) {
  void* pointer = nullptr;
  BuildAllocationGraph(0, &pointer);
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_EQ(hipSuccess, api_.graph_create(&clone_, 0));
  hipGraphNode_t free_node = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_free(&free_node, clone_, nullptr, 0, pointer));
  ASSERT_EQ(hipSuccess, api_.instantiate(&sibling_, clone_, 0));
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(hipSuccess, api_.launch(executable_, stream_));
    device_ = pointer;
    ASSERT_EQ(hipSuccess, api_.launch(sibling_, stream_));
    device_ = nullptr;
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
    EXPECT_EQ(source_, destination_);
  }
}

TEST_F(HipGraphMemoryApiTest, CapturedAllocationsUseGraphLifetimes) {
  ASSERT_EQ(hipSuccess,
            api_.begin_capture(stream_, hipStreamCaptureModeGlobal));
  void* pointer = nullptr;
  ASSERT_EQ(hipSuccess, api_.allocate_async(&pointer, source_.size(), stream_));
  ASSERT_NE(nullptr, pointer);
  ASSERT_EQ(hipSuccess, api_.copy_async(pointer, source_.data(), source_.size(),
                                        hipMemcpyHostToDevice, stream_));
  ASSERT_EQ(hipSuccess,
            api_.copy_async(destination_.data(), pointer, destination_.size(),
                            hipMemcpyDeviceToHost, stream_));
  ASSERT_EQ(hipSuccess, api_.free_async(pointer, stream_));
  ASSERT_EQ(hipSuccess, api_.end_capture(stream_, &clone_));
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, clone_, 0));
  for (int i = 0; i < 2; ++i) {
    Replay();
    ASSERT_FALSE(HasFatalFailure());
    EXPECT_EQ(source_, destination_);
    source_ = replacement_;
    destination_.fill(0);
  }
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
  CheckPitchedCopyClone(false, false);
}

TEST_F(HipGraphMemoryApiTest, ImplicitSliceHeightIncludesStagedCopyOrigins) {
  CheckPitchedCopyClone(true, false);
}

TEST_F(HipGraphMemoryApiTest,
       ImplicitSliceHeightIncludesRegisteredCopyOrigins) {
  CheckPitchedCopyClone(true, true);
}

TEST_F(HipGraphMemoryApiTest, CapturedPitchedCopiesPreserveStagedOrigins) {
  CheckPitchedCopyClone(true, false, true);
}

TEST_F(HipGraphMemoryApiTest, CapturedPitchedCopiesPreserveRegisteredOrigins) {
  CheckPitchedCopyClone(true, true, true);
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
       WideJoinReplaysAfterMutationCloneAndSourceRelease) {
  destination_.fill(0);
  std::vector<hipGraphNode_t> predecessors(destination_.size() - 1);
  for (size_t i = 0; i < predecessors.size(); ++i) {
    hipHostNodeParams host = {
        .fn = +[](void* value) { ++*static_cast<uint8_t*>(value); },
        .userData = &destination_[i],
    };
    ASSERT_EQ(hipSuccess,
              api_.add_host(&predecessors[i], graph_, nullptr, 0, &host));
  }
  hipHostNodeParams host = {
      .fn =
          +[](void* value) {
            auto& values = *static_cast<decltype(destination_)*>(value);
            uint8_t sum = 0;
            for (size_t i = 0; i + 1 < values.size(); ++i) {
              sum += values[i];
            }
            values.back() = sum;
          },
      .userData = &destination_,
  };
  hipGraphNode_t join = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_host(&join, graph_, nullptr, 0, &host));
  for (size_t i = predecessors.size(); i > 0; --i) {
    ASSERT_EQ(hipSuccess,
              api_.add_dependencies(graph_, &predecessors[i - 1], &join, 1));
  }
  for (size_t i = 0; i < predecessors.size(); i += 3) {
    ASSERT_EQ(hipSuccess,
              api_.remove_dependencies(graph_, &predecessors[i], &join, 1));
    ASSERT_EQ(hipSuccess,
              api_.add_dependencies(graph_, &predecessors[i], &join, 1));
  }
  std::vector<hipGraphNode_t> observed(predecessors.size());
  size_t count = observed.size();
  ASSERT_EQ(hipSuccess, api_.get_dependencies(join, observed.data(), &count));
  ASSERT_EQ(predecessors.size(), count);
  EXPECT_TRUE(std::is_permutation(predecessors.begin(), predecessors.end(),
                                  observed.begin(), observed.end()));
  std::vector<hipGraphNode_t> destinations(predecessors.size(), join);
  ASSERT_EQ(hipSuccess,
            api_.remove_dependencies(graph_, predecessors.data(),
                                     destinations.data(), predecessors.size()));
  count = 0;
  ASSERT_EQ(hipSuccess, api_.get_roots(graph_, nullptr, &count));
  EXPECT_EQ(predecessors.size() + 1, count);
  for (auto predecessor : predecessors) {
    ASSERT_EQ(hipSuccess,
              api_.add_dependencies(graph_, &predecessor, &join, 1));
  }
  ASSERT_EQ(hipSuccess, api_.graph_clone(&clone_, graph_));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph_));
  graph_ = nullptr;
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, clone_, 0));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(clone_));
  clone_ = nullptr;
  for (size_t replay = 1; replay <= 2; ++replay) {
    Replay();
    ASSERT_FALSE(HasFatalFailure());
    for (size_t i = 0; i < predecessors.size(); ++i) {
      EXPECT_EQ(replay, destination_[i]);
    }
    EXPECT_EQ(static_cast<uint8_t>(replay * predecessors.size()),
              destination_.back());
  }
}

TEST_F(HipGraphMemoryApiTest, DependencyQueriesFollowMutationAndCloning) {
  std::array<hipGraphNode_t, 5> nodes = {};
  for (size_t i = 0; i < nodes.size(); ++i) {
    ASSERT_EQ(hipSuccess,
              api_.add_empty(&nodes[i], graph_, i == 2 ? &nodes[0] : nullptr,
                             i == 2 ? 1 : 0));
  }
  const hipGraphNode_t from[] = {nodes[1], nodes[2], nodes[0]};
  const hipGraphNode_t to[] = {nodes[2], nodes[3], nodes[4]};
  ASSERT_EQ(hipSuccess, api_.add_dependencies(graph_, from, to, 3));
  ASSERT_EQ(hipSuccess, api_.graph_clone(&clone_, graph_));

  const auto check = [&](hipGraph_t graph,
                         const std::array<hipGraphNode_t, 5>& handles) {
    size_t count = 0;
    ASSERT_EQ(hipSuccess, api_.get_edges(graph, nullptr, nullptr, &count));
    ASSERT_EQ(4u, count);
    std::array<hipGraphNode_t, 6> sources;
    std::array<hipGraphNode_t, 6> targets;
    sources.fill(handles[0]);
    targets.fill(handles[0]);
    count = sources.size();
    ASSERT_EQ(hipSuccess,
              api_.get_edges(graph, sources.data(), targets.data(), &count));
    EXPECT_EQ(4u, count);
    const std::vector<std::pair<hipGraphNode_t, hipGraphNode_t>> expected = {
        {handles[0], handles[2]},
        {handles[1], handles[2]},
        {handles[2], handles[3]},
        {handles[0], handles[4]}};
    std::vector<std::pair<hipGraphNode_t, hipGraphNode_t>> actual;
    for (size_t i = 0; i < count; ++i) {
      actual.emplace_back(sources[i], targets[i]);
    }
    EXPECT_TRUE(std::is_permutation(actual.begin(), actual.end(),
                                    expected.begin(), expected.end()));
    for (size_t i = count; i < sources.size(); ++i) {
      EXPECT_EQ(nullptr, sources[i]);
      EXPECT_EQ(nullptr, targets[i]);
    }
    // A short output reports/fills only its capacity, without overwriting the
    // adjacent sentinel. Query ordering is not part of the API contract.
    sources.fill(handles[0]);
    targets.fill(handles[0]);
    count = 1;
    ASSERT_EQ(hipSuccess,
              api_.get_edges(graph, sources.data(), targets.data(), &count));
    EXPECT_EQ(1u, count);
    EXPECT_NE(expected.end(),
              std::find(expected.begin(), expected.end(),
                        std::make_pair(sources[0], targets[0])));
    EXPECT_EQ(handles[0], sources[1]);
    EXPECT_EQ(handles[0], targets[1]);

    count = sources.size();
    ASSERT_EQ(hipSuccess, api_.get_roots(graph, sources.data(), &count));
    ASSERT_EQ(2u, count);
    EXPECT_TRUE((sources[0] == handles[0] && sources[1] == handles[1]) ||
                (sources[0] == handles[1] && sources[1] == handles[0]));
    for (size_t i = count; i < sources.size(); ++i) {
      EXPECT_EQ(nullptr, sources[i]);
    }
    sources.fill(handles[4]);
    count = 1;
    ASSERT_EQ(hipSuccess, api_.get_roots(graph, sources.data(), &count));
    EXPECT_EQ(1u, count);
    EXPECT_TRUE(sources[0] == handles[0] || sources[0] == handles[1]);
    EXPECT_EQ(handles[4], sources[1]);
    count = 0;
    ASSERT_EQ(hipSuccess, api_.get_roots(graph, sources.data() + 1, &count));
    EXPECT_EQ(0u, count);
    EXPECT_EQ(handles[4], sources[1]);
    count = sources.size();
    ASSERT_EQ(hipSuccess,
              api_.get_dependencies(handles[2], sources.data(), &count));
    ASSERT_EQ(2u, count);
    EXPECT_TRUE((sources[0] == handles[0] && sources[1] == handles[1]) ||
                (sources[0] == handles[1] && sources[1] == handles[0]));
    EXPECT_EQ(nullptr, sources[count]);
    count = sources.size();
    ASSERT_EQ(hipSuccess,
              api_.get_dependents(handles[0], sources.data(), &count));
    ASSERT_EQ(2u, count);
    EXPECT_TRUE((sources[0] == handles[2] && sources[1] == handles[4]) ||
                (sources[0] == handles[4] && sources[1] == handles[2]));
    EXPECT_EQ(nullptr, sources[count]);
  };
  check(graph_, nodes);
  ASSERT_FALSE(HasFatalFailure());
  std::array<hipGraphNode_t, 5> clones;
  for (size_t i = 0; i < nodes.size(); ++i) {
    ASSERT_EQ(hipSuccess, api_.find_clone_node(&clones[i], nodes[i], clone_));
  }
  // The second pair is absent: the first must remain untouched on rejection.
  const hipGraphNode_t invalid_from[] = {nodes[1], nodes[3]};
  const hipGraphNode_t invalid_to[] = {nodes[2], nodes[4]};
  EXPECT_EQ(hipErrorInvalidValue,
            api_.remove_dependencies(graph_, invalid_from, invalid_to, 2));
  check(graph_, nodes);
  ASSERT_FALSE(HasFatalFailure());
  // Remove both embedded and added edges, then delete a node with incoming and
  // outgoing added edges. Neither operation can change the clone's topology.
  const hipGraphNode_t removed_from[] = {nodes[0], nodes[1]};
  const hipGraphNode_t removed_to[] = {nodes[2], nodes[2]};
  ASSERT_EQ(hipSuccess,
            api_.remove_dependencies(graph_, removed_from, removed_to, 2));
  ASSERT_EQ(hipSuccess, api_.add_dependencies(graph_, &nodes[1], &nodes[2], 1));
  ASSERT_EQ(hipSuccess, api_.destroy_node(nodes[2]));
  size_t count = 0;
  ASSERT_EQ(hipSuccess, api_.get_edges(graph_, nullptr, nullptr, &count));
  EXPECT_EQ(1u, count);
  ASSERT_EQ(hipSuccess, api_.get_roots(graph_, nullptr, &count));
  EXPECT_EQ(3u, count);
  check(clone_, clones);
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
  hipGraphNode_t root = nullptr;
  size_t root_count = 1;
  ASSERT_EQ(hipSuccess, api_.get_roots(graph_, &root, &root_count));
  EXPECT_EQ(producer, root);
  size_t edge_count = 0;
  ASSERT_EQ(hipSuccess, api_.get_edges(graph_, nullptr, nullptr, &edge_count));
  EXPECT_EQ(3u, edge_count);
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
  ASSERT_EQ(hipSuccess, api_.graph_clone(&clone_, graph_));

  // Both snapshots keep their edges and copy helpers when source mutation
  // removes nodes and renumbers the remaining handles.
  ASSERT_EQ(hipSuccess, api_.destroy_node(producer));
  ASSERT_EQ(hipSuccess, api_.destroy_node(upload));
  ASSERT_EQ(hipSuccess, api_.instantiate(&sibling_, clone_, 0));
  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph_));
  graph_ = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_destroy(clone_));
  clone_ = nullptr;
  for (auto executable : {executable_, sibling_}) {
    source_.fill(0);
    destination_.fill(0);
    observed_.fill(0);
    Replay(executable);
    ASSERT_FALSE(HasFatalFailure());
    EXPECT_EQ(observed_, replacement_);
  }
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

TEST_F(HipGraphMemoryApiTest, PendingUpdatesChangeStagingAndDependencyOrder) {
  ASSERT_EQ(hipSuccess, api_.allocate(&device_, source_.size()));
  for (auto& buffer : host_buffers_) {
    ASSERT_EQ(hipSuccess, api_.allocate_host(&buffer, source_.size(),
                                             hipHostMallocDefault));
  }
  std::memcpy(host_buffers_[0], replacement_.data(), replacement_.size());
  std::memset(host_buffers_[1], 0, source_.size());
  ASSERT_EQ(hipSuccess, api_.graph_create(&clone_, 0));
  std::array<hipGraph_t, 2> graphs = {graph_, clone_};
  std::array<std::array<hipGraphNode_t, 96>, 2> roots;
  std::array<hipGraphNode_t, 2> uploads;
  for (size_t i = 0; i < graphs.size(); ++i) {
    for (auto& root : roots[i]) {
      ASSERT_EQ(hipSuccess, api_.add_empty(&root, graphs[i], nullptr, 0));
    }
    // The first template needs hidden staging callbacks, while registered
    // storage in the replacement is directly accessible to the device. Both
    // describe the same public operations, with distinct physical node indices.
    ASSERT_EQ(hipSuccess,
              api_.add_copy_1d(&uploads[i], graphs[i],
                               i == 0 ? roots[i].data() : nullptr,
                               i == 0 ? roots[i].size() : 0, device_,
                               i == 0 ? source_.data() : host_buffers_[0],
                               source_.size(), hipMemcpyHostToDevice));
    if (i == 1) {
      std::array<hipGraphNode_t, 96> destinations;
      destinations.fill(uploads[i]);
      ASSERT_EQ(hipSuccess,
                api_.add_dependencies(graphs[i], roots[i].data(),
                                      destinations.data(), roots[i].size()));
    }
    hipGraphNode_t download = nullptr;
    ASSERT_EQ(hipSuccess,
              api_.add_copy_1d(&download, graphs[i], &uploads[i], 1,
                               i == 0 ? destination_.data() : host_buffers_[1],
                               device_, source_.size(), hipMemcpyDeviceToHost));
  }
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
  for (size_t i = 0; i < graphs.size(); ++i) {
    const size_t next = i ^ 1;
    HoldStreamAtGate();
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_EQ(hipSuccess, api_.launch(executable_, stream_));
    hipGraphNode_t error_node = nullptr;
    hipGraphExecUpdateResult result;
    ASSERT_EQ(hipSuccess, api_.update_executable(executable_, graphs[next],
                                                 &error_node, &result));
    EXPECT_EQ(result, hipGraphExecUpdateSuccess);
    EXPECT_EQ(error_node, nullptr);
    ASSERT_EQ(hipSuccess,
              api_.remove_dependencies(graphs[next], &roots[next].back(),
                                       &uploads[next], 1));
    EXPECT_EQ(hipErrorGraphExecUpdateFailure,
              api_.update_executable(executable_, graphs[next], &error_node,
                                     &result));
    EXPECT_EQ(result, hipGraphExecUpdateErrorTopologyChanged);
    EXPECT_EQ(error_node, uploads[next]);
    ASSERT_EQ(hipSuccess,
              api_.add_dependencies(graphs[next], &roots[next].back(),
                                    &uploads[next], 1));
    ReleaseStreamGate();
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
    EXPECT_EQ(destination_, source_);
    if (i == 0) {
      EXPECT_EQ(
          0, std::memcmp(host_buffers_[1], observed_.data(), source_.size()));
    } else {
      EXPECT_EQ(0, std::memcmp(host_buffers_[1], replacement_.data(),
                               source_.size()));
    }
  }
  ASSERT_EQ(hipSuccess, api_.graph_destroy(clone_));
  clone_ = nullptr;
  ASSERT_EQ(hipSuccess, api_.graph_destroy(graph_));
  graph_ = nullptr;
  destination_.fill(0);
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(destination_, source_);
}

TEST_F(HipGraphMemoryApiTest, ZeroLengthCopiesKeepDirectForms) {
  ASSERT_EQ(hipSuccess, api_.allocate(&device_, source_.size()));
  ASSERT_EQ(hipSuccess, api_.allocate_host(&host_buffers_[0], source_.size(),
                                           hipHostMallocDefault));
  hipGraphNode_t node = nullptr;
  EXPECT_EQ(hipSuccess,
            api_.add_copy_1d(&node, graph_, nullptr, 0, destination_.data(),
                             source_.data(), 0, hipMemcpyHostToHost));
  EXPECT_EQ(hipSuccess, api_.add_copy_1d(&node, graph_, nullptr, 0, device_,
                                         device_, 0, hipMemcpyDeviceToDevice));
  for (bool registered : {false, true}) {
    void* host = registered ? host_buffers_[0] : source_.data();
    const hipError_t expected = registered ? hipSuccess : hipErrorInvalidValue;
    EXPECT_EQ(expected, api_.add_copy_1d(&node, graph_, nullptr, 0, device_,
                                         host, 0, hipMemcpyHostToDevice));
    EXPECT_EQ(expected, api_.add_copy_1d(&node, graph_, nullptr, 0, host,
                                         device_, 0, hipMemcpyDeviceToHost));
  }
  hipMemcpy3DParms empty = {
      .srcPtr = {device_, 1, 1, 1},
      .dstPtr = {device_, 1, 1, 1},
      .extent = {0, 1, 1},
      .kind = hipMemcpyDeviceToDevice,
  };
  EXPECT_EQ(hipErrorInvalidValue,
            api_.add_copy(&node, graph_, nullptr, 0, &empty));
}

TEST_F(HipGraphMemoryApiTest, CopyPreparationRespectsLogicalPoolBounds) {
  // Small HIP pool allocations have a larger physical extent. Direct and graph
  // copies use the requested length even when the backing contains more bytes.
  constexpr size_t length = 3;
  ASSERT_EQ(hipSuccess, api_.allocate_async(&device_, length, stream_));
  ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
  ASSERT_EQ(hipSuccess,
            api_.copy(device_, source_.data(), length, hipMemcpyHostToDevice));
  auto* pointer = static_cast<uint8_t*>(device_);
  hipGraphNode_t node = nullptr;
  ASSERT_EQ(hipSuccess, api_.add_copy_1d(&node, graph_, nullptr, 0, pointer + 2,
                                         pointer, 1, hipMemcpyDeviceToDevice));
  ASSERT_EQ(hipSuccess, api_.instantiate(&executable_, graph_, 0));
  for (hipMemcpyKind kind : {hipMemcpyDeviceToDevice, hipMemcpyDefault,
                             hipMemcpyDeviceToDeviceNoCU}) {
    for (bool source_overrun : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "kind=" << kind << " source_overrun=" << source_overrun);
      hipMemcpy3DParms parameters = {
          .srcPtr = {pointer + (source_overrun ? length : 0), 1, 1, 1},
          .dstPtr = {pointer + (source_overrun ? 0 : length), 1, 1, 1},
          .extent = {1, 1, 1},
          .kind = kind,
      };
      hipGraphNode_t rejected = nullptr;
      EXPECT_EQ(
          hipErrorInvalidValue,
          api_.copy(parameters.dstPtr.ptr, parameters.srcPtr.ptr, 1, kind));
      EXPECT_EQ(hipErrorInvalidValue,
                api_.copy_async(parameters.dstPtr.ptr, parameters.srcPtr.ptr, 1,
                                kind, stream_));
      EXPECT_EQ(hipErrorInvalidValue,
                api_.add_copy(&rejected, graph_, nullptr, 0, &parameters));
      EXPECT_EQ(
          hipErrorInvalidValue,
          api_.add_copy_1d(&rejected, graph_, nullptr, 0, parameters.dstPtr.ptr,
                           parameters.srcPtr.ptr, 1, kind));
      EXPECT_EQ(hipErrorInvalidValue,
                api_.set_copy(node, parameters.dstPtr.ptr,
                              parameters.srcPtr.ptr, 1, kind));
      EXPECT_EQ(
          hipErrorInvalidValue,
          api_.set_executable_copy(executable_, node, parameters.dstPtr.ptr,
                                   parameters.srcPtr.ptr, 1, kind));
    }
  }
  size_t node_count = 0;
  ASSERT_EQ(hipSuccess, api_.get_nodes(graph_, nullptr, &node_count));
  EXPECT_EQ(1u, node_count);
  hipMemcpy3DParms observed = {};
  ASSERT_EQ(hipSuccess, api_.get_copy(node, &observed));
  EXPECT_EQ(pointer, observed.srcPtr.ptr);
  EXPECT_EQ(pointer + 2, observed.dstPtr.ptr);
  EXPECT_EQ(hipMemcpyDeviceToDevice, observed.kind);
  Replay();
  ASSERT_FALSE(HasFatalFailure());
  std::array<uint8_t, length> output = {};
  ASSERT_EQ(hipSuccess,
            api_.copy(output.data(), pointer, length, hipMemcpyDeviceToHost));
  EXPECT_EQ(output,
            (std::array<uint8_t, length>{source_[0], source_[1], source_[0]}));
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

TEST_F(HipGraphMemoryApiTest,
       DirectCopiesPreserveSnapshotsAndRegisteredStreamOrder) {
  const size_t length = source_.size();
  ASSERT_EQ(hipSuccess, api_.allocate(&device_, 2 * length));
  for (auto& buffer : host_buffers_) {
    ASSERT_EQ(hipSuccess,
              api_.allocate_host(&buffer, length, hipHostMallocDefault));
  }
  auto* second = static_cast<uint8_t*>(device_) + length;
  std::memcpy(host_buffers_[0], replacement_.data(), length);
  for (bool inferred : {false, true}) {
    SCOPED_TRACE(inferred);
    auto pageable = source_;
    std::memset(host_buffers_[1], 0, length);
    HoldStreamAtGate();
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_EQ(hipSuccess, api_.copy_async(device_, pageable.data(), length,
                                          inferred ? hipMemcpyDefault
                                                   : hipMemcpyHostToDevice,
                                          stream_));
    // The small pageable upload owns a snapshot when the API returns, even
    // though the GPU cannot consume it until the preceding callback finishes.
    pageable.fill(0xCC);
    ASSERT_EQ(hipSuccess, api_.copy_async(second, host_buffers_[0], length,
                                          inferred ? hipMemcpyDefault
                                                   : hipMemcpyHostToDevice,
                                          stream_));
    ASSERT_EQ(hipSuccess, api_.copy_async(host_buffers_[1], device_, length,
                                          inferred ? hipMemcpyDefault
                                                   : hipMemcpyDeviceToHost,
                                          stream_));
    EXPECT_EQ(hipErrorNotReady, api_.stream_query(stream_));
    // Both registered operations must return while their predecessor remains
    // blocked. Their caller-owned storage stays live through completion.
    ReleaseStreamGate();
    ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
    EXPECT_EQ(0, std::memcmp(source_.data(), host_buffers_[1], length));
    ASSERT_EQ(hipSuccess, api_.copy(destination_.data(), second, length,
                                    hipMemcpyDeviceToHost));
    EXPECT_EQ(replacement_, destination_);
  }
}

TEST_F(HipGraphMemoryApiTest, ManagedCopiesPreserveCheckedInteriorRanges) {
  const auto allocate_managed =
      dso_.Resolve<decltype(&hipMallocManaged)>("hipMallocManaged");
  ASSERT_NE(nullptr, allocate_managed);
  const size_t length = source_.size();
  ASSERT_EQ(hipSuccess,
            allocate_managed(&device_, 2 * length, hipMemAttachGlobal));
  auto* bytes = static_cast<uint8_t*>(device_);
  for (auto kind : {hipMemcpyDeviceToDevice, hipMemcpyDefault,
                    hipMemcpyDeviceToDeviceNoCU}) {
    for (bool asynchronous : {false, true}) {
      std::memcpy(bytes, source_.data(), length);
      std::memset(bytes + length, 0xCC, length);
      const hipError_t result =
          asynchronous
              ? api_.copy_async(bytes + length + 2, bytes + 1, length - 3, kind,
                                stream_)
              : api_.copy(bytes + length + 2, bytes + 1, length - 3, kind);
      ASSERT_EQ(hipSuccess, result);
      ASSERT_EQ(hipSuccess, api_.stream_synchronize(stream_));
      EXPECT_EQ(0, std::memcmp(bytes, source_.data(), length));
      for (size_t i = 0; i < length; ++i) {
        EXPECT_EQ(i >= 2 && i < length - 1 ? source_[i - 1] : 0xCC,
                  bytes[length + i]);
      }
    }
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
  for (hipMemcpyKind kind : {hipMemcpyDeviceToDevice, hipMemcpyDefault,
                             hipMemcpyDeviceToDeviceNoCU}) {
    for (int placement = 0; placement < 3; ++placement) {
      hipDeviceptr_t source = placement == 0 ? device_ : remote_buffers_[0];
      hipDeviceptr_t destination =
          placement == 1 ? device_ : remote_buffers_[1];
      ASSERT_EQ(hipSuccess, api_.set_context(placement == 0 ? primary_context_
                                                            : remote_context_));
      ASSERT_EQ(hipSuccess, api_.copy(source, source_.data(), source_.size(),
                                      hipMemcpyHostToDevice));
      ASSERT_EQ(hipSuccess, api_.set_context(primary_context_));
      hipGraphNode_t node = nullptr;
      SCOPED_TRACE(::testing::Message()
                   << "kind=" << kind << " placement=" << placement);
      if (kind == hipMemcpyDeviceToDevice) {
        ASSERT_EQ(hipSuccess,
                  api_.add_copy_1d(&node, graph_, nullptr, 0, destination,
                                   source, source_.size(), kind));
      } else {
        hipMemcpy3DParms parameters = {
            .srcPtr = {source, 8, 8, 4},
            .dstPtr = {destination, 8, 8, 4},
            .extent = {4, 4, 2},
            .kind = kind,
        };
        ASSERT_EQ(hipSuccess,
                  api_.add_copy(&node, graph_, nullptr, 0, &parameters));
      }
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
      auto initial_expected = untouched;
      for (size_t i = 0; i < source_.size(); ++i) {
        if (kind == hipMemcpyDeviceToDevice || i % 8 < 4) {
          initial_expected[i] = source_[i];
        }
      }
      EXPECT_EQ(destination_, initial_expected) << placement;
      ASSERT_EQ(hipSuccess, api_.set_context(primary_context_));
      EXPECT_EQ(hipErrorInvalidValue,
                api_.set_executable_copy(
                    executable_, node, destination,
                    static_cast<const uint8_t*>(source) + source_.size() - 1, 2,
                    kind));
      ASSERT_EQ(hipSuccess,
                api_.set_executable_copy(
                    executable_, node, static_cast<uint8_t*>(destination) + 16,
                    static_cast<const uint8_t*>(source) + 8, 32, kind));
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
}

}  // namespace
