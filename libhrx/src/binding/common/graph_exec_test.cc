// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <cstring>
#include <thread>
#include <utility>

#include "common/graph.h"
#include "common/internal.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Marks the flag |user_data| names as reached. Used as the body of host calls
// whose only purpose is to say whether the stream got that far.
void SetFlag(void* user_data) {
  static_cast<std::atomic<bool>*>(user_data)->store(true,
                                                    std::memory_order_release);
}

void DestroyGraphExecHandle(iree_hal_streaming_graph_exec_t* executable) {
  if (executable) {
    IREE_EXPECT_OK(iree_hal_streaming_graph_exec_destroy_handle(executable));
  }
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
struct RejectSecondHostCallQueue {
  iree_hal_queue_t base;
  iree_hal_queue_t* target = nullptr;
  iree_hal_semaphore_t* first_call_gate = nullptr;
  uint64_t first_call_gate_value = 0;
  std::atomic<int> host_call_count = 0;
  std::atomic<int> accepted_host_call_count = 0;
  std::atomic<bool> second_call_rejected = false;
};

RejectSecondHostCallQueue* CastRejectSecondHostCallQueue(
    iree_hal_queue_t* base_queue) {
  return reinterpret_cast<RejectSecondHostCallQueue*>(base_queue);
}

void DestroyRejectSecondHostCallQueue(iree_hal_queue_t* base_queue) {
  auto* queue = CastRejectSecondHostCallQueue(base_queue);
  iree_hal_queue_release(queue->target);
  queue->target = nullptr;
}

iree_status_t RejectSecondHostCallQueueBarrier(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_queue_barrier_flags_t flags) {
  return iree_hal_queue_barrier(
      CastRejectSecondHostCallQueue(base_queue)->target, wait_semaphore_list,
      signal_semaphore_list, flags);
}

iree_status_t RejectSecondHostCallQueueHostCall(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  auto* queue = CastRejectSecondHostCallQueue(base_queue);
  const int call_index =
      queue->host_call_count.fetch_add(1, std::memory_order_acq_rel);
  if (call_index == 1) {
    queue->second_call_rejected.store(true, std::memory_order_release);
    return iree_make_status(IREE_STATUS_ABORTED,
                            "injected later graph block rejection");
  }
  if (call_index != 0 || !queue->first_call_gate) {
    return iree_hal_queue_host_call(queue->target, wait_semaphore_list,
                                    signal_semaphore_list, call, args, flags);
  }
  if (wait_semaphore_list.count >= 4) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "injected queue wait capacity exceeded");
  }

  std::array<iree_hal_semaphore_t*, 4> wait_semaphores = {};
  std::array<uint64_t, 4> wait_values = {};
  for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
    wait_semaphores[i] = wait_semaphore_list.semaphores[i];
    wait_values[i] = wait_semaphore_list.payload_values[i];
  }
  wait_semaphores[wait_semaphore_list.count] = queue->first_call_gate;
  wait_values[wait_semaphore_list.count] = queue->first_call_gate_value;
  const iree_hal_semaphore_list_t gated_waits = {
      /*.count=*/wait_semaphore_list.count + 1,
      /*.semaphores=*/wait_semaphores.data(),
      /*.payload_values=*/wait_values.data(),
  };
  iree_status_t status = iree_hal_queue_host_call(
      queue->target, gated_waits, signal_semaphore_list, call, args, flags);
  if (iree_status_is_ok(status)) {
    queue->accepted_host_call_count.fetch_add(1, std::memory_order_acq_rel);
  }
  return status;
}

iree_status_t RejectSecondHostCallQueueFlush(iree_hal_queue_t* base_queue) {
  return iree_hal_queue_flush(
      CastRejectSecondHostCallQueue(base_queue)->target);
}

const iree_hal_queue_vtable_t kRejectSecondHostCallQueueVtable = {
    /*.destroy=*/DestroyRejectSecondHostCallQueue,
    /*.barrier=*/RejectSecondHostCallQueueBarrier,
    /*.execute=*/nullptr,
    /*.host_call=*/RejectSecondHostCallQueueHostCall,
    /*.query_dispatch_concurrency=*/nullptr,
    /*.dispatch=*/nullptr,
    /*.atomic_wait=*/nullptr,
    /*.atomic_store=*/nullptr,
    /*.atomic_rmw=*/nullptr,
    /*.timestamp=*/nullptr,
    /*.flush=*/RejectSecondHostCallQueueFlush,
    /*.alloca=*/nullptr,
    /*.dealloca=*/nullptr,
    /*.transfer=*/nullptr,
    /*.read=*/nullptr,
    /*.write=*/nullptr,
};

void InitializeRejectSecondHostCallQueue(iree_hal_queue_t* target,
                                         RejectSecondHostCallQueue* out_queue) {
  out_queue->target = target;
  iree_hal_queue_retain(target);
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  params.priority = iree_hal_queue_priority(target);
  params.features = iree_hal_queue_features(target);
  params.execution_resources = iree_hal_queue_execution_resources(target);
  iree_hal_queue_initialize(iree_hal_queue_family(target), &params,
                            &kRejectSecondHostCallQueueVtable,
                            &out_queue->base);
}

iree_hal_queue_t* ReplaceGraphExecTestQueue(iree_hal_streaming_stream_t* stream,
                                            iree_hal_queue_t* replacement) {
  iree_slim_mutex_lock(&stream->mutex);
  iree_hal_queue_t* previous = stream->queue;
  stream->queue = replacement;
  iree_slim_mutex_unlock(&stream->mutex);
  return previous;
}

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
        context_, context_->queue, IREE_HAL_STREAMING_STREAM_FLAG_NONE,
        /*priority=*/0, iree_allocator_system(), &stream_));
  }

  void TearDown() override {
    // Graph-exec destruction is nonblocking. Drain its stream-ordered
    // retirement calls before releasing the fixture-owned device entry and
    // the arena block pool embedded in it.
    IREE_EXPECT_OK(iree_hal_streaming_context_synchronize(context_));
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

TEST_F(GraphExecTest, BatchMemoryNodePreservesResolvedWriteSemantics) {
  iree_hal_buffer_t* buffer = nullptr;
  iree_hal_buffer_mapping_t mapping = {};
  iree_hal_streaming_retained_buffer_ref_t target = {};
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_t* clone_graph = nullptr;
  iree_hal_streaming_graph_exec_t* executable = nullptr;
  ScopeExit release_handles([&] {
    DestroyGraphExecHandle(executable);
    iree_hal_streaming_graph_release(clone_graph);
    iree_hal_streaming_graph_release(graph);
    iree_hal_streaming_retained_buffer_ref_deinitialize(&target);
    if (mapping.contents.data) {
      IREE_EXPECT_OK(iree_hal_buffer_unmap_range(&mapping));
    }
    iree_hal_buffer_release(buffer);
  });

  const iree_hal_buffer_params_t buffer_params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_DEFAULT,
      /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
      /*.type=*/IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
  };
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      context_->device_allocator, buffer_params, sizeof(uint64_t), &buffer));
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_PERSISTENT, IREE_HAL_MEMORY_ACCESS_ALL,
      /*local_byte_offset=*/0, sizeof(uint64_t), &mapping));
  ASSERT_NE(mapping.contents.data, nullptr);
  *reinterpret_cast<uint64_t*>(mapping.contents.data) = 0;
  IREE_ASSERT_OK(hrx_buffer_create_from_hal(
      buffer, device_entry_.hrx_device,
      HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
      sizeof(uint64_t), mapping.contents.data, &target.owner));
  target.buffer = buffer;
  iree_hal_buffer_retain(target.buffer);
  target.memory_type = buffer_params.type;
  target.allocation_size = sizeof(uint64_t);
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &graph));

  const iree_hal_atomic_store_params_t store_params = {
      /*.value=*/47,
      /*.flags=*/IREE_HAL_ATOMIC_FLAG_RELEASE |
          IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_64,
      /*.reserved=*/{},
  };
  iree_hal_streaming_value_operation_t operation = {
      /*.kind=*/IREE_HAL_STREAMING_VALUE_OPERATION_STORE,
      /*.target_buffer=*/target.buffer,
      /*.target_offset=*/target.offset,
  };
  operation.params.store = store_params;
  const hrx_buffer_t owner = target.owner;
  const uint32_t public_params = 0x12345678u;
  const uint64_t public_operation = 0x87654321u;
  iree_hal_streaming_graph_node_t* node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_batch_mem_op_node(
      graph, /*dependencies=*/nullptr, /*dependency_count=*/0, &public_params,
      sizeof(public_params), &public_operation, sizeof(public_operation),
      &operation, &owner, /*operation_count=*/1, &node));
  ASSERT_NE(node, nullptr);
  ASSERT_EQ(1u, node->attrs.batch_mem_op.operation_count);
  EXPECT_EQ(store_params.flags,
            node->attrs.batch_mem_op.operations[0].params.store.flags);
  EXPECT_EQ(store_params.width,
            node->attrs.batch_mem_op.operations[0].params.store.width);
  EXPECT_EQ(store_params.value,
            node->attrs.batch_mem_op.operations[0].params.store.value);
  EXPECT_EQ(target.buffer,
            node->attrs.batch_mem_op.operations[0].target_buffer);
  EXPECT_EQ(target.offset,
            node->attrs.batch_mem_op.operations[0].target_offset);

  IREE_ASSERT_OK(iree_hal_streaming_graph_clone(graph, context_, &clone_graph));
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      clone_graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE,
      &executable));
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(executable, stream_));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
  EXPECT_EQ(47u, *reinterpret_cast<uint64_t*>(mapping.contents.data));
}

// A replayed event record ends its event's association with the graph a
// capture-time record left on it, and the launch releases every reference it
// takes over exactly once.
//
// The records are split across a parent graph and a child graph so the child's
// walk claims room in the same storage the parent's does, and there are more of
// them than the sixteen cells one chunk of that storage holds, so the walk has
// to grow it and carry across the growth what it already collected. The test
// keeps no reference to the capture graph, so the launch drops the last one and
// the release of the context the graph retained counts the destruction.
TEST_F(GraphExecTest, ReplayedEventRecordsDropEveryCapturedGraphReference) {
  static constexpr iree_host_size_t kChildEventCount = 7;
  static constexpr iree_host_size_t kParentEventCount = 10;
  std::array<iree_hal_streaming_event_t*, kChildEventCount + kParentEventCount>
      events = {};
  iree_hal_streaming_graph_t* child_graph = nullptr;
  iree_hal_streaming_graph_t* parent_graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  // Hands these handles back, on the assertion failure paths as much as on the
  // last line. They start null and a release takes null, so a body cut short
  // hands back only what it reached. The capture graph is not among them: the
  // test drops its reference mid-body on purpose, to leave the events holding
  // the last ones.
  ScopeExit release_handles([&] {
    DestroyGraphExecHandle(exec);
    iree_hal_streaming_graph_release(parent_graph);
    iree_hal_streaming_graph_release(child_graph);
    for (iree_hal_streaming_event_t* event : events) {
      iree_hal_streaming_event_release(event);
    }
  });

  for (iree_hal_streaming_event_t*& event : events) {
    IREE_ASSERT_OK(iree_hal_streaming_event_create(
        context_, IREE_HAL_STREAMING_EVENT_FLAG_NONE, iree_allocator_system(),
        &event));
  }

  // Associates every event with one capture graph.
  IREE_ASSERT_OK(iree_hal_streaming_begin_capture(
      stream_, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  for (iree_hal_streaming_event_t* event : events) {
    IREE_ASSERT_OK(iree_hal_streaming_event_record(event, stream_));
  }
  iree_hal_streaming_graph_t* captured_graph = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_end_capture(stream_, &captured_graph));
  ASSERT_NE(captured_graph, nullptr);
  for (iree_hal_streaming_event_t* event : events) {
    ASSERT_EQ(event->capture_graph, captured_graph);
  }

  // Leaves the events holding the only references to the capture graph.
  iree_hal_streaming_graph_release(captured_graph);

  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &child_graph));
  iree_hal_streaming_graph_node_t* child_tail = nullptr;
  AppendEventRecordChain(child_graph, events.data(), kChildEventCount,
                         &child_tail);
  ASSERT_FALSE(HasFatalFailure());

  // The child graph node leads, so the parent's own record nodes are walked
  // after the child's and on top of the room the child claimed.
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &parent_graph));
  iree_hal_streaming_graph_node_t* parent_tail = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_child_graph_node(
      parent_graph, /*dependencies=*/nullptr, /*dependency_count=*/0,
      child_graph, &parent_tail));
  AppendEventRecordChain(parent_graph, events.data() + kChildEventCount,
                         kParentEventCount, &parent_tail);
  ASSERT_FALSE(HasFatalFailure());

  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      parent_graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

  // A graph independently owns its construction context and execution-context
  // hint. Both name |context_| here, so destroying the one capture graph drops
  // two context references and nothing else.
  const int32_t context_references_before =
      iree_atomic_ref_count_load(&context_->ref_count);
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(exec, stream_));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));

  for (iree_hal_streaming_event_t* event : events) {
    EXPECT_EQ(event->capture_graph, nullptr);
  }
  EXPECT_EQ(context_references_before - 2,
            iree_atomic_ref_count_load(&context_->ref_count))
      << "the launch destroyed the capture graph a number of times other than "
         "once";
}

// An instantiated executable takes an event node's event only from the context
// that created the graph, which is the rule the template setters hold every
// other way of naming an event node's event to. A launch relies on it for a
// record node: it answers for every record it holds by comparing the launching
// stream's context with its own, and a record block naming an event of some
// third context would make those two questions different ones. A wait node is
// held to the same rule because the template setter holds both node types to
// it, so retargeting cannot seat an event the template would have refused.
TEST_F(GraphExecTest, ExecEventNodeTakesOnlyItsOwnContextsEvent) {
  iree_hal_streaming_context_t* other_context = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  iree_hal_streaming_event_t* replacement = nullptr;
  iree_hal_streaming_event_t* other_context_event = nullptr;
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  // Hands back whatever was built, on the assertion failure paths as much as
  // on the last line.
  ScopeExit release_handles([&] {
    DestroyGraphExecHandle(exec);
    iree_hal_streaming_graph_release(graph);
    iree_hal_streaming_event_release(other_context_event);
    iree_hal_streaming_event_release(replacement);
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_context_release(other_context);
  });

  iree_hal_streaming_context_flags_t context_flags = {};
  context_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
  IREE_ASSERT_OK(iree_hal_streaming_context_create(
      &device_entry_, context_flags, iree_allocator_system(), &other_context));

  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_NONE, iree_allocator_system(),
      &event));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_NONE, iree_allocator_system(),
      &replacement));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      other_context, IREE_HAL_STREAMING_EVENT_FLAG_NONE,
      iree_allocator_system(), &other_context_event));

  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &graph));
  iree_hal_streaming_graph_node_t* record_node = nullptr;
  AppendEventRecordChain(graph, &event, /*count=*/1, &record_node);
  ASSERT_FALSE(HasFatalFailure());
  iree_hal_streaming_graph_node_t* wait_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_event_node(
      graph, &record_node, /*dependency_count=*/1,
      IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT, event, &wait_node));

  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

  iree_status_t status = iree_hal_streaming_graph_exec_set_event_node_event(
      exec, record_node, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD,
      other_context_event);
  EXPECT_EQ(IREE_STATUS_INVALID_ARGUMENT, iree_status_code(status))
      << "an executable took a record node event from another context, which a "
         "launch would only find out at the record itself, with the blocks "
         "ahead of it already submitted";
  iree_status_free(status);

  status = iree_hal_streaming_graph_exec_set_event_node_event(
      exec, wait_node, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT,
      other_context_event);
  EXPECT_EQ(IREE_STATUS_INVALID_ARGUMENT, iree_status_code(status))
      << "an executable took a wait node event from another context, which the "
         "template setter refuses for a wait node just as it does for a record "
         "one";
  iree_status_free(status);

  // The refusals are the context rule and not a blanket one.
  IREE_EXPECT_OK(iree_hal_streaming_graph_exec_set_event_node_event(
      exec, record_node, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD,
      replacement));
  IREE_EXPECT_OK(iree_hal_streaming_graph_exec_set_event_node_event(
      exec, wait_node, IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_WAIT,
      replacement));
}

TEST_F(GraphExecTest, RebuiltHostCallRetainsExecutableUntilCompletion) {
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  ScopeExit release_handles([&] {
    DestroyGraphExecHandle(exec);
    iree_hal_streaming_graph_release(graph);
  });

  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &graph));
  iree_hal_streaming_graph_node_t* host_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      graph, /*dependencies=*/nullptr, /*dependency_count=*/0, &SetFlag,
      &graph_host_node_ran_, &host_node));
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

  // Rebuilds compile through a stack-local candidate whose state is moved into
  // |exec|. Host-call resources must name |exec| rather than that candidate so
  // queued work can keep the executable alive after this function returns.
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_rebuild_from_template(exec));
  auto* exec_resource = reinterpret_cast<iree_hal_resource_t*>(exec);
  EXPECT_EQ(1, iree_atomic_ref_count_load(&exec_resource->ref_count))
      << "rebuilt callback state retained its owning executable";
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(exec, stream_));
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_destroy_handle(exec));
  exec = nullptr;

  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
  EXPECT_TRUE(graph_host_node_ran_.load(std::memory_order_acquire));
}

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
    DestroyGraphExecHandle(exec);
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
      other_context, other_context->queue, IREE_HAL_STREAMING_STREAM_FLAG_NONE,
      /*priority=*/0, iree_allocator_system(), &other_stream));

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

// A later rejection in a recursive child launch must not let an accepted child
// block outlive graph-exec state. The rejected final block cannot signal the
// reserved stream value, so the launch drains the actual accepted signal while
// keeping pending_value unchanged.
TEST_F(GraphExecTest,
       LaterChildBlockRejectionDrainsPrefixWithoutPublishingStreamTail) {
  iree_hal_streaming_graph_t* child_graph = nullptr;
  iree_hal_streaming_graph_t* parent_graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  iree_hal_semaphore_t* gate = nullptr;
  RejectSecondHostCallQueue fault_queue = {};
  iree_hal_queue_t* original_queue = nullptr;
  bool wrapper_installed = false;
  ScopeExit release_handles([&] {
    if (wrapper_installed) {
      iree_hal_queue_retain(original_queue);
      iree_hal_queue_t* installed_queue =
          ReplaceGraphExecTestQueue(stream_, original_queue);
      iree_hal_queue_release(installed_queue);
    }
    iree_hal_semaphore_release(gate);
    DestroyGraphExecHandle(exec);
    iree_hal_streaming_graph_release(parent_graph);
    iree_hal_streaming_graph_release(child_graph);
  });

  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context_->device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &gate));
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &child_graph));
  iree_hal_streaming_graph_node_t* first_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      child_graph, /*dependencies=*/nullptr, /*dependency_count=*/0, &SetFlag,
      &graph_host_node_ran_, &first_node));
  iree_hal_streaming_graph_node_t* second_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      child_graph, &first_node, /*dependency_count=*/1, &SetFlag,
      &stream_marker_ran_, &second_node));

  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &parent_graph));
  iree_hal_streaming_graph_node_t* child_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_child_graph_node(
      parent_graph, /*dependencies=*/nullptr, /*dependency_count=*/0,
      child_graph, &child_node));
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      parent_graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

  original_queue = stream_->queue;
  InitializeRejectSecondHostCallQueue(original_queue, &fault_queue);
  EXPECT_EQ(original_queue,
            ReplaceGraphExecTestQueue(stream_, &fault_queue.base));
  // Transfer the stream's old queue reference to the wrapper target.
  iree_hal_queue_release(original_queue);
  wrapper_installed = true;
  fault_queue.first_call_gate = gate;
  fault_queue.first_call_gate_value = 1;

  iree_slim_mutex_lock(&stream_->mutex);
  const uint64_t initial_pending_value = stream_->pending_value;
  iree_slim_mutex_unlock(&stream_->mutex);

  std::atomic<bool> launch_returned = false;
  std::atomic<iree_status_code_t> launch_status_code = IREE_STATUS_UNKNOWN;
  std::thread launch_thread([&] {
    iree_status_t status = iree_hal_streaming_graph_exec_launch(exec, stream_);
    launch_status_code.store(iree_status_code(status),
                             std::memory_order_release);
    iree_status_ignore(status);
    launch_returned.store(true, std::memory_order_release);
  });

  bool observed_later_rejection = false;
  for (int i = 0; i < 1000000; ++i) {
    if (fault_queue.second_call_rejected.load(std::memory_order_acquire)) {
      observed_later_rejection = true;
      break;
    }
    if (launch_returned.load(std::memory_order_acquire)) {
      break;
    }
    std::this_thread::yield();
  }
  if (!observed_later_rejection) {
    IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, 1, /*frontier=*/nullptr));
    launch_thread.join();
    FAIL() << "recursive launch did not reach the injected later rejection";
    return;
  }
  EXPECT_EQ(
      1, fault_queue.accepted_host_call_count.load(std::memory_order_acquire));
  for (int i = 0;
       i < 100000 && !launch_returned.load(std::memory_order_acquire); ++i) {
    std::this_thread::yield();
  }
  EXPECT_FALSE(launch_returned.load(std::memory_order_acquire))
      << "launch returned while its accepted prefix was still gated";

  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, 1, /*frontier=*/nullptr));
  launch_thread.join();
  EXPECT_EQ(IREE_STATUS_ABORTED,
            launch_status_code.load(std::memory_order_acquire));
  bool accepted_prefix_completed = false;
  for (int i = 0; i < 1000000; ++i) {
    if (graph_host_node_ran_.load(std::memory_order_acquire)) {
      accepted_prefix_completed = true;
      break;
    }
    std::this_thread::yield();
  }
  EXPECT_TRUE(accepted_prefix_completed);
  EXPECT_FALSE(stream_marker_ran_.load(std::memory_order_acquire));

  iree_slim_mutex_lock(&stream_->mutex);
  const uint64_t final_pending_value = stream_->pending_value;
  iree_slim_mutex_unlock(&stream_->mutex);
  EXPECT_EQ(initial_pending_value, final_pending_value)
      << "a rejected final block installed an unreachable reserved tail";
  if (final_pending_value > initial_pending_value) {
    // Keep perturbation failures bounded: a phantom point has no queue signal.
    uint64_t timeline_value = 0;
    IREE_EXPECT_OK(
        iree_hal_semaphore_query(stream_->timeline_semaphore, &timeline_value));
    if (timeline_value < final_pending_value) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(stream_->timeline_semaphore,
                                               final_pending_value,
                                               /*frontier=*/nullptr));
    }
  }

  // Accepted work is terminal at return, so immediate executable teardown is
  // safe even though no stream-tail point was published for the failed launch.
  DestroyGraphExecHandle(exec);
  exec = nullptr;
}

}  // namespace
