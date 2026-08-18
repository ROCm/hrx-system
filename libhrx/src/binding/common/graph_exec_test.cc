// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <cstring>
#include <utility>

#include "common/internal.h"
#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Marks the flag |user_data| names as reached. Used as the body of host calls
// whose only purpose is to say whether the stream got that far.
void SetFlag(void* user_data) {
  static_cast<std::atomic<bool>*>(user_data)->store(true,
                                                    std::memory_order_release);
}

// Runs |cleanup| when it leaves scope. A test body builds its handles across a
// run of fatal assertions and a fatal assertion returns from the body, so the
// releases have to sit somewhere that return cannot skip.
template <typename Cleanup>
class ScopeExit {
 public:
  explicit ScopeExit(Cleanup cleanup) : cleanup_(std::move(cleanup)) {}
  ~ScopeExit() { cleanup_(); }
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;

 private:
  // Called once, when this object is destroyed.
  Cleanup cleanup_;
};
// Required despite matching the implicit guide: clang builds this file with
// -Wctad-maybe-unsupported under -Werror, and that warning fires wherever a
// template's arguments are deduced and the template declares no guide of its
// own. A guard deduces because its cleanup is a lambda, whose type no
// declaration can spell.
template <typename Cleanup>
ScopeExit(Cleanup) -> ScopeExit<Cleanup>;

// Runs streaming graph launches against the host CPU device. Launches take the
// same block submit path they take on an accelerator; the event records they
// enqueue resolve to queue barriers because the device advertises no timestamp
// domain for the records to write into.
class GraphExecTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_initialize(/*flags=*/0)));
    hrx_device_t hrx_device = nullptr;
    IREE_ASSERT_OK(HRX_CALL(hrx_cpu_device_get(/*index=*/0, &hrx_device)));

    // Stands in for the registry entry global initialization builds around an
    // enumerated accelerator: contexts take their HAL device from the entry and
    // graphs carve their node storage out of its block pool.
    memset(&device_entry_, 0, sizeof(device_entry_));
    device_entry_.hrx_device = hrx_device;
    device_entry_.hal_device = hrx_device_hal(hrx_device);
    iree_slim_mutex_initialize(&device_entry_.primary_context_mutex);
    iree_slim_mutex_initialize(&device_entry_.graph_memory_mutex);
    iree_arena_block_pool_initialize(/*block_size=*/64 * 1024,
                                     iree_allocator_system(),
                                     &device_entry_.block_pool);

    iree_hal_streaming_context_flags_t context_flags = {};
    context_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
    IREE_ASSERT_OK(iree_hal_streaming_context_create(
        &device_entry_, context_flags, iree_allocator_system(), &context_));
    IREE_ASSERT_OK(iree_hal_streaming_stream_create(
        context_, IREE_HAL_STREAMING_STREAM_FLAG_NONE, /*priority=*/0,
        iree_allocator_system(), &stream_));
  }

  void TearDown() override {
    iree_hal_streaming_stream_release(stream_);
    iree_hal_streaming_context_release(context_);
    iree_arena_block_pool_deinitialize(&device_entry_.block_pool);
    iree_slim_mutex_deinitialize(&device_entry_.graph_memory_mutex);
    iree_slim_mutex_deinitialize(&device_entry_.primary_context_mutex);
    IREE_EXPECT_OK(HRX_CALL(hrx_cpu_shutdown()));
  }

  // Adds |count| event record nodes recording |events| to |graph|, each
  // depending on the node |tail| names. Chaining them pins the schedule's node
  // order and puts each node in a partition, and so a block, of its own.
  // Leaves |tail| naming the last node added.
  void AppendEventRecordChain(iree_hal_streaming_graph_t* graph,
                              iree_hal_streaming_event_t* const* events,
                              iree_host_size_t count,
                              iree_hal_streaming_graph_node_t** tail) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      iree_hal_streaming_graph_node_t* node = nullptr;
      IREE_ASSERT_OK(iree_hal_streaming_graph_add_event_node(
          graph, *tail ? tail : nullptr, *tail ? 1 : 0,
          IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD, events[i], &node));
      ASSERT_NE(node, nullptr);
      *tail = node;
    }
  }

  // Registry entry backing |context_|; outlives every context created from it.
  iree_hal_streaming_device_t device_entry_ = {};
  // Context owning the graphs, events, and streams each test builds.
  iree_hal_streaming_context_t* context_ = nullptr;
  // Stream every launch submits on.
  iree_hal_streaming_stream_t* stream_ = nullptr;
  // Set by the host-call node a graph carries. A fixture member rather than a
  // local because no semaphore edge a test can name joins a block an aborted
  // launch left in flight; hrx_cpu_shutdown() above is what drains the workers,
  // so the flag has to outlive the test body it is read in.
  std::atomic<bool> graph_host_node_ran_{false};
  // Set by the host call a test enqueues behind an aborted launch to give the
  // blocks that launch may have left in flight their chance to run. A fixture
  // member for the same reason, and the case is not hypothetical here: the
  // path that reads this flag false is the path where the callback is still
  // pending as the test body returns.
  std::atomic<bool> stream_marker_ran_{false};
};

// A launch answers the cross-context record rule once for the whole executable,
// and a record buried in a child graph is one of the records it answers for:
// instantiating a child graph node folds the child's answer into the parent's,
// so the launch below is refused before any of the graph is submitted.
//
// Nothing else in this executable would report the refusal in time. The
// parent's own walk holds no record node, so without the fold the launch would
// submit the host call ahead of the child and only reach the rule at the
// child's record block, leaving that host call in flight on a stream whose
// timeline the failed launch never advanced.
//
// The host-node read is what carries the fold. Drop the fold and the launch is
// still refused with the same code and the same message, raised by the child's
// own record block from inside the walk, so the status read below passes
// either way: it pins that the refusal reaches the caller, not where it was
// decided. Weaken the host-node read and nothing here tells the fold from its
// absence.
TEST_F(GraphExecTest, ChildGraphRecordRefusesALaunchOnAnotherContextsStream) {
  iree_hal_streaming_context_t* other_context = nullptr;
  iree_hal_streaming_stream_t* other_stream = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  iree_hal_streaming_graph_t* child_graph = nullptr;
  iree_hal_streaming_graph_t* parent_graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  // Hands back whatever was built, on the assertion failure paths as much as
  // on the last line. Releasing the stream and the context is what drains and
  // unregisters them, and the fixture shuts the device down either way.
  ScopeExit release_handles([&] {
    iree_hal_streaming_graph_exec_release(exec);
    iree_hal_streaming_graph_release(parent_graph);
    iree_hal_streaming_graph_release(child_graph);
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_stream_release(other_stream);
    iree_hal_streaming_context_release(other_context);
  });

  iree_hal_streaming_context_flags_t context_flags = {};
  context_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
  IREE_ASSERT_OK(iree_hal_streaming_context_create(
      &device_entry_, context_flags, iree_allocator_system(), &other_context));
  IREE_ASSERT_OK(iree_hal_streaming_stream_create(
      other_context, IREE_HAL_STREAMING_STREAM_FLAG_NONE, /*priority=*/0,
      iree_allocator_system(), &other_stream));

  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_NONE, iree_allocator_system(),
      &event));

  // The executable's only record node sits in the child graph.
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &child_graph));
  iree_hal_streaming_graph_node_t* child_tail = nullptr;
  AppendEventRecordChain(child_graph, &event, /*count=*/1, &child_tail);
  ASSERT_FALSE(HasFatalFailure());

  // The host node depends on nothing and the child graph node depends on it, so
  // the host call is the block a launch submits first and the child's record
  // the block that would refuse it. Whether the host node ran is how the test
  // sees what a refused launch had already submitted.
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &parent_graph));
  iree_hal_streaming_graph_node_t* host_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      parent_graph, /*dependencies=*/nullptr, /*dependency_count=*/0, &SetFlag,
      &graph_host_node_ran_, &host_node));
  iree_hal_streaming_graph_node_t* child_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_child_graph_node(
      parent_graph, &host_node, /*dependency_count=*/1, child_graph,
      &child_node));

  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      parent_graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

  iree_status_t status =
      iree_hal_streaming_graph_exec_launch(exec, other_stream);
  EXPECT_EQ(IREE_STATUS_INCOMPATIBLE, iree_status_code(status))
      << "a launch on another context's stream was accepted for an executable "
         "whose only record sits in a child graph";
  iree_status_free(status);

  // A refused launch leaves the stream tail where it was, so a block it had
  // enqueued would have waited on nothing and been runnable the moment the
  // queue took it. Draining the stream behind a callback enqueued after the
  // refusal is what gives such a block its chance to run before the read below;
  // synchronizing alone would not, because the launch advanced no timeline
  // value for the synchronize to wait on. This is a window and not an ordering
  // proof: user-visible order comes from semaphore edges, and a block an
  // aborted launch left behind signals a semaphore internal to the executable
  // that shares none with the callback below.
  IREE_ASSERT_OK(iree_hal_streaming_launch_host_function(other_stream, &SetFlag,
                                                         &stream_marker_ran_));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(other_stream));
  ASSERT_TRUE(stream_marker_ran_.load(std::memory_order_acquire))
      << "the stream was synchronized without running the callback behind it, "
         "so nothing here says when the launch's own blocks would have run";
  EXPECT_FALSE(graph_host_node_ran_.load(std::memory_order_acquire))
      << "the node ahead of the refused record ran, so the launch submitted "
         "part of the graph and then failed";
}

}  // namespace
