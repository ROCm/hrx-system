// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <semaphore>
#include <thread>
#include <utility>
#include <vector>

#include "common/context.h"
#include "common/device.h"
#include "common/event.h"
#include "common/graph.h"
#include "common/host_callback.h"
#include "common/hrx_bridge.h"
#include "common/module.h"
#include "common/stream.h"
#include "common/stream_value.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libhrx/src/libhrx/buffer.h"
#include "libhrx/src/libhrx/device.h"
#include "libhrx/src/libhrx/mem_pool_backing.h"

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

struct GraphLaunchObservation {
  // Guards every phase field below.
  std::mutex mutex;
  // Notifies the test whenever one of the observed phases changes.
  std::condition_variable notification;
  // True after the queue wrapper injects the later submission rejection.
  bool later_call_rejected = false;
  // True after the accepted prefix host call runs.
  bool prefix_call_ran = false;
  // True after the graph launch returns to its calling thread.
  bool launch_returned = false;
  // Records whether the graph launch returned before its accepted prefix ran.
  bool launch_returned_before_prefix = false;
};

void ObserveGraphLaunchPrefix(void* user_data) {
  auto* observation = static_cast<GraphLaunchObservation*>(user_data);
  {
    std::lock_guard<std::mutex> lock(observation->mutex);
    observation->launch_returned_before_prefix = observation->launch_returned;
    observation->prefix_call_ran = true;
  }
  observation->notification.notify_all();
}

struct RejectSecondHostCallQueue {
  // Base queue interface exposed to the graph executor.
  iree_hal_queue_t base;
  // Retained queue receiving submissions accepted by this wrapper.
  iree_hal_queue_t* target = nullptr;
  // Gate appended to the first accepted host call.
  iree_hal_semaphore_t* first_call_gate = nullptr;
  // Required value of |first_call_gate|.
  uint64_t first_call_gate_value = 0;
  // Shared test phase observation.
  GraphLaunchObservation* observation = nullptr;
  // Number of host calls presented to this wrapper.
  std::atomic<int> host_call_count = 0;
  // Number of host calls accepted by the target queue.
  std::atomic<int> accepted_host_call_count = 0;
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
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_queue_barrier_flags_t flags) {
  return iree_hal_queue_barrier(
      CastRejectSecondHostCallQueue(base_queue)->target, wait_semaphore_list,
      signal_semaphore_list, barriers, flags);
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
    {
      std::lock_guard<std::mutex> lock(queue->observation->mutex);
      queue->observation->later_call_rejected = true;
    }
    queue->observation->notification.notify_all();
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
      .count = wait_semaphore_list.count + 1,
      .semaphores = wait_semaphores.data(),
      .payload_values = wait_values.data(),
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
    .destroy = DestroyRejectSecondHostCallQueue,
    .barrier = RejectSecondHostCallQueueBarrier,
    .execute = nullptr,
    .host_call = RejectSecondHostCallQueueHostCall,
    .query_dispatch_concurrency = nullptr,
    .dispatch = nullptr,
    .atomic_wait = nullptr,
    .atomic_store = nullptr,
    .atomic_rmw = nullptr,
    .timestamp = nullptr,
    .flush = RejectSecondHostCallQueueFlush,
    .alloca = nullptr,
    .dealloca = nullptr,
    .transfer = nullptr,
    .read = nullptr,
    .write = nullptr,
};

void InitializeRejectSecondHostCallQueue(iree_hal_queue_t* target,
                                         GraphLaunchObservation* observation,
                                         RejectSecondHostCallQueue* out_queue) {
  out_queue->target = target;
  out_queue->observation = observation;
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

// Metadata-only executable retained by graph dispatch blocks. The controlled
// queue below answers occupancy without consulting executable metadata and no
// dispatch is allowed to reach the executable.
struct AtomicPreflightExecutable {
  iree_hal_executable_t base;
};

void DestroyAtomicPreflightExecutable(iree_hal_executable_t* executable) {
  (void)executable;
}

const iree_hal_executable_vtable_t kAtomicPreflightExecutableVtable = {
    .destroy = DestroyAtomicPreflightExecutable,
    .function_count = nullptr,
    .function_info = nullptr,
    .function_parameters = nullptr,
    .lookup_function_by_name = nullptr,
    .try_lookup_global_by_name = nullptr,
    .global_info = nullptr,
    .global_buffer = nullptr,
};

// Queue wrapper that gives one cooperative workgroup concurrent residency and
// rejects every submission. A correct launch only asks it the former question;
// submission counters make any prefix that escapes preflight observable.
struct AtomicPreflightQueue {
  iree_hal_queue_t base;
  iree_hal_queue_t* target = nullptr;
  std::atomic<int> concurrency_query_count = 0;
  std::atomic<int> submission_count = 0;
  uint32_t queried_workgroup_size[3] = {};
};

AtomicPreflightQueue* CastAtomicPreflightQueue(iree_hal_queue_t* base_queue) {
  return reinterpret_cast<AtomicPreflightQueue*>(base_queue);
}

void DestroyAtomicPreflightQueue(iree_hal_queue_t* base_queue) {
  auto* queue = CastAtomicPreflightQueue(base_queue);
  iree_hal_queue_release(queue->target);
  queue->target = nullptr;
}

iree_status_t AtomicPreflightQueueBarrier(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_queue_barrier_flags_t flags) {
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)flags;
  CastAtomicPreflightQueue(base_queue)
      ->submission_count.fetch_add(1, std::memory_order_acq_rel);
  return iree_make_status(IREE_STATUS_ABORTED,
                          "unexpected graph submission before preflight");
}

iree_status_t AtomicPreflightQueueHostCall(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)call;
  (void)args;
  (void)flags;
  CastAtomicPreflightQueue(base_queue)
      ->submission_count.fetch_add(1, std::memory_order_acq_rel);
  return iree_make_status(IREE_STATUS_ABORTED,
                          "unexpected graph submission before preflight");
}

iree_status_t AtomicPreflightQueueQueryDispatchConcurrency(
    iree_hal_queue_t* base_queue, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function,
    iree_hal_queue_dispatch_concurrency_params_t params,
    iree_hal_queue_dispatch_concurrency_flags_t flags,
    iree_hal_queue_dispatch_concurrency_t* out_concurrency) {
  (void)executable;
  (void)function;
  (void)flags;
  auto* queue = CastAtomicPreflightQueue(base_queue);
  memcpy(queue->queried_workgroup_size, params.workgroup_size,
         sizeof(queue->queried_workgroup_size));
  queue->concurrency_query_count.fetch_add(1, std::memory_order_acq_rel);
  *out_concurrency = {
      .scheduling_domain_count = 1,
      .maximum_concurrent_workgroup_count_per_domain = 1,
  };
  return iree_ok_status();
}

iree_status_t AtomicPreflightQueueDispatch(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_dispatch_flags_t flags) {
  (void)wait_semaphore_list;
  (void)signal_semaphore_list;
  (void)executable;
  (void)function;
  (void)config;
  (void)constants;
  (void)bindings;
  (void)flags;
  CastAtomicPreflightQueue(base_queue)
      ->submission_count.fetch_add(1, std::memory_order_acq_rel);
  return iree_make_status(IREE_STATUS_ABORTED,
                          "unexpected graph submission before preflight");
}

iree_status_t AtomicPreflightQueueFlush(iree_hal_queue_t* base_queue) {
  return iree_hal_queue_flush(CastAtomicPreflightQueue(base_queue)->target);
}

const iree_hal_queue_vtable_t kAtomicPreflightQueueVtable = {
    .destroy = DestroyAtomicPreflightQueue,
    .barrier = AtomicPreflightQueueBarrier,
    .execute = nullptr,
    .host_call = AtomicPreflightQueueHostCall,
    .query_dispatch_concurrency = AtomicPreflightQueueQueryDispatchConcurrency,
    .dispatch = AtomicPreflightQueueDispatch,
    .atomic_wait = nullptr,
    .atomic_store = nullptr,
    .atomic_rmw = nullptr,
    .timestamp = nullptr,
    .flush = AtomicPreflightQueueFlush,
    .alloca = nullptr,
    .dealloca = nullptr,
    .transfer = nullptr,
    .read = nullptr,
    .write = nullptr,
};

void InitializeAtomicPreflightQueue(iree_hal_queue_t* target,
                                    AtomicPreflightQueue* out_queue) {
  out_queue->target = target;
  iree_hal_queue_retain(target);
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  params.priority = iree_hal_queue_priority(target);
  params.features = iree_hal_queue_features(target) |
                    IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH;
  params.execution_resources = iree_hal_queue_execution_resources(target);
  iree_hal_queue_initialize(iree_hal_queue_family(target), &params,
                            &kAtomicPreflightQueueVtable, &out_queue->base);
}

iree_hal_queue_t* ReplaceGraphExecTestQueue(iree_hal_streaming_stream_t* stream,
                                            iree_hal_queue_t* replacement) {
  iree_slim_mutex_lock(&stream->mutex);
  iree_hal_queue_t* previous = stream->queue;
  stream->queue = replacement;
  iree_slim_mutex_unlock(&stream->mutex);
  return previous;
}

// Observes submitted graph boundaries while the real task queue executes them.
struct GraphBoundaryQueue {
  // Queue interface borrowed by the launch stream during observation.
  iree_hal_queue_t base;
  // Retained execution target.
  iree_hal_queue_t* target = nullptr;
  // Stream predecessor that every initial workstream must wait for.
  std::pair<iree_hal_semaphore_t*, uint64_t> predecessor;
  // Actual completion points supplied to the independent workstreams.
  std::vector<std::pair<iree_hal_semaphore_t*, uint64_t>> completions;
  // True after a barrier joins all observed workstream completions.
  bool joined = false;
};

GraphBoundaryQueue* CastGraphBoundaryQueue(iree_hal_queue_t* queue) {
  return reinterpret_cast<GraphBoundaryQueue*>(queue);
}

void DestroyGraphBoundaryQueue(iree_hal_queue_t* base_queue) {
  iree_hal_queue_release(CastGraphBoundaryQueue(base_queue)->target);
}

bool ContainsPoint(iree_hal_semaphore_list_t points,
                   std::pair<iree_hal_semaphore_t*, uint64_t> expected) {
  for (iree_host_size_t i = 0; i < points.count; ++i) {
    if (points.semaphores[i] == expected.first &&
        points.payload_values[i] == expected.second) {
      return true;
    }
  }
  return false;
}

iree_status_t GraphBoundaryQueueExecute(
    iree_hal_queue_t* base_queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_queue_execute_flags_t flags) {
  auto* queue = CastGraphBoundaryQueue(base_queue);
  EXPECT_TRUE(ContainsPoint(waits, queue->predecessor));
  EXPECT_GT(signals.count, 0u);
  if (signals.count) {
    queue->completions.emplace_back(signals.semaphores[0],
                                    signals.payload_values[0]);
  }
  return iree_hal_queue_execute(queue->target, waits, signals, command_buffer,
                                binding_table, flags);
}

iree_status_t GraphBoundaryQueueBarrier(
    iree_hal_queue_t* base_queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_queue_barrier_flags_t flags) {
  auto* queue = CastGraphBoundaryQueue(base_queue);
  EXPECT_FALSE(queue->completions.empty());
  EXPECT_EQ(queue->completions.size(), waits.count);
  for (const auto& completion : queue->completions) {
    EXPECT_TRUE(ContainsPoint(waits, completion));
  }
  EXPECT_TRUE(ContainsPoint(
      signals, {queue->predecessor.first, queue->predecessor.second + 1}));
  queue->joined = true;
  return iree_hal_queue_barrier(queue->target, waits, signals, barriers, flags);
}

iree_status_t GraphBoundaryQueueFlush(iree_hal_queue_t* base_queue) {
  return iree_hal_queue_flush(CastGraphBoundaryQueue(base_queue)->target);
}

const iree_hal_queue_vtable_t kGraphBoundaryQueueVtable = {
    .destroy = DestroyGraphBoundaryQueue,
    .barrier = GraphBoundaryQueueBarrier,
    .execute = GraphBoundaryQueueExecute,
    .host_call = nullptr,
    .query_dispatch_concurrency = nullptr,
    .dispatch = nullptr,
    .atomic_wait = nullptr,
    .atomic_store = nullptr,
    .atomic_rmw = nullptr,
    .timestamp = nullptr,
    .flush = GraphBoundaryQueueFlush,
    .alloca = nullptr,
    .dealloca = nullptr,
    .transfer = nullptr,
    .read = nullptr,
    .write = nullptr,
};

void InitializeGraphBoundaryQueue(iree_hal_queue_t* target,
                                  GraphBoundaryQueue* queue) {
  queue->target = target;
  iree_hal_queue_retain(target);
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  params.priority = iree_hal_queue_priority(target);
  params.features = iree_hal_queue_features(target);
  params.execution_resources = iree_hal_queue_execution_resources(target);
  iree_hal_queue_initialize(iree_hal_queue_family(target), &params,
                            &kGraphBoundaryQueueVtable, &queue->base);
}

// Runs streaming graph launches against the host CPU device. Launches take the
// same block submit path they take on an accelerator; the event records they
// enqueue resolve to queue barriers because the device advertises no timestamp
// domain for the records to write into.
class GraphExecTest : public ::testing::Test {
 protected:
  // Owns the task device and its runtime through fixture cleanup.
  hrx_device_group_t group_ = nullptr;

  void SetUp() override {
    iree_hal_streaming_host_callback_queue_initialize(iree_allocator_system(),
                                                      &host_callbacks_);
    hrx_runtime_t runtime = nullptr;
    IREE_ASSERT_OK(HRX_CALL(
        hrx_runtime_create(nullptr, hrx_host_allocator_system(), &runtime)));
    const hrx_device_create_params_t params = {.uri = "task://"};
    hrx_status_t status = hrx_device_group_create(runtime, 1, &params, &group_);
    hrx_runtime_release(runtime);
    IREE_ASSERT_OK(HRX_CALL(status));
    hrx_device_t hrx_device = hrx_device_group_get_device(group_, 0);

    // Stands in for the registry entry global initialization builds around an
    // enumerated accelerator: contexts take their HAL device from the entry and
    // graphs carve their node storage out of its block pool.
    memset(&device_entry_, 0, sizeof(device_entry_));
    device_entry_.hrx_device = hrx_device;
    device_entry_.hal_device = hrx_device_hal(hrx_device);
    device_entry_.host_callbacks = &host_callbacks_;
    iree_slim_mutex_initialize(&device_entry_.primary_context_mutex);
    iree_slim_mutex_initialize(&device_entry_.graph_memory_mutex);
    iree_arena_block_pool_initialize(/*block_size=*/64 * 1024,
                                     iree_allocator_system(),
                                     &device_entry_.block_pool);

    iree_hal_streaming_context_flags_t context_flags = {
        .scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO,
    };
    IREE_ASSERT_OK(iree_hal_streaming_context_create(
        &device_entry_, context_flags, iree_allocator_system(), &context_));
    IREE_ASSERT_OK(iree_hal_streaming_stream_create(
        context_, context_->queue, IREE_HAL_STREAMING_STREAM_FLAG_NONE,
        /*priority=*/0, iree_allocator_system(), &stream_));
  }

  void TearDown() override {
    if (context_) {
      IREE_EXPECT_STATUS_IS(expected_synchronization_status_,
                            iree_hal_streaming_context_synchronize(context_));
    }
    iree_hal_streaming_host_callback_queue_stop(&host_callbacks_);
    iree_hal_streaming_stream_release(stream_);
    iree_hal_streaming_context_release(context_);
    iree_arena_block_pool_deinitialize(&device_entry_.block_pool);
    iree_slim_mutex_deinitialize(&device_entry_.graph_memory_mutex);
    iree_slim_mutex_deinitialize(&device_entry_.primary_context_mutex);
    hrx_device_group_release(group_);
    group_ = nullptr;
    iree_hal_streaming_host_callback_queue_deinitialize(&host_callbacks_);
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

  // Explicit frontend-owned callout service borrowed by the test device entry.
  iree_hal_streaming_host_callback_queue_t host_callbacks_ = {};
  // Registry entry backing |context_|; outlives every context created from it.
  iree_hal_streaming_device_t device_entry_ = {};
  // Context owning the graphs, events, and streams each test builds.
  iree_hal_streaming_context_t* context_ = nullptr;
  // Stream every launch submits on.
  iree_hal_streaming_stream_t* stream_ = nullptr;
  // A failed submission remains observable at the context's completion join.
  iree_status_code_t expected_synchronization_status_ = IREE_STATUS_OK;
  // Set by the host-call node a graph carries. A fixture member rather than a
  // local because no semaphore edge a test can name joins a block an aborted
  // launch left in flight; releasing the group above is what drains the
  // workers, so the flag has to outlive the test body it is read in.
  std::atomic<bool> graph_host_node_ran_{false};
  // Set by the host call a test enqueues behind an aborted launch to give the
  // blocks that launch may have left in flight their chance to run. A fixture
  // member for the same reason, and the case is not hypothetical here: the
  // path that reads this flag false is the path where the callback is still
  // pending as the test body returns.
  std::atomic<bool> stream_marker_ran_{false};
};

TEST_F(GraphExecTest, RejectedCopyConstructionPreservesTopologyAndBacking) {
  for (const size_t initial_nodes : {size_t{0}, size_t{15}, size_t{64}}) {
    SCOPED_TRACE(initial_nodes);
    std::array<int, 2> calls = {};
    iree_hal_streaming_buffer_t* destination = nullptr;
    iree_hal_streaming_graph_t* graph = nullptr;
    iree_hal_streaming_graph_exec_t* executable = nullptr;
    ScopeExit release_handles([&] {
      IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
      iree_hal_streaming_graph_exec_release(executable);
      iree_hal_streaming_graph_release(graph);
      iree_hal_streaming_memory_release_wrapped_buffer(destination);
    });
    IREE_ASSERT_OK(iree_hal_streaming_graph_create(
        context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
        &graph));
    for (size_t i = 0; i < initial_nodes; ++i) {
      IREE_ASSERT_OK(
          iree_hal_streaming_graph_add_empty_node(graph, nullptr, 0, nullptr));
    }
    IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_host_staging(
        context_, sizeof(uint32_t), &destination));
    const auto checkpoint = iree_hal_streaming_graph_copy_checkpoint(graph);
    const auto increment = +[](void* data) { ++*static_cast<int*>(data); };
    struct NodeAllocator {
      // Real arena used by the graph's node and block constructors.
      iree_allocator_t arena;
      // Construction allocation to reject in this attempt.
      size_t fail_index;
      // Number of allocations observed in this attempt.
      size_t count = 0;
    } allocator = {graph->arena_allocator, 0};
    const uint32_t value = 37;
    struct CopyParameters {
      // Executable-owned staging pointer filled during instantiation.
      void* destination;
      // Caller-owned source, live through completion.
      const uint32_t* source;
      // Counts preparation invocations independently of the copy value.
      int* calls;
    } parameters = {nullptr, &value, &calls[0]};
    const auto prepare_copy = +[](void* user_data) {
      const auto* parameters = static_cast<const CopyParameters*>(user_data);
      memcpy(parameters->destination, parameters->source, sizeof(uint32_t));
      ++*parameters->calls;
    };
    size_t rejected = 0;
    for (;; ++allocator.fail_index) {
      allocator.count = 0;
      graph->arena_allocator = {
          .self = &allocator,
          .ctl =
              +[](void* self, iree_allocator_command_t command,
                  const void* parameters, void** pointer) {
                auto* state = static_cast<NodeAllocator*>(self);
                if (command != IREE_ALLOCATOR_COMMAND_FREE &&
                    state->count++ == state->fail_index) {
                  return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
                }
                return state->arena.ctl(state->arena.self, command, parameters,
                                        pointer);
              },
      };
      iree_hal_streaming_graph_node_t* prepare = nullptr;
      iree_hal_streaming_graph_node_t* copy = nullptr;
      iree_hal_streaming_graph_node_t* complete = nullptr;
      iree_status_t status = iree_hal_streaming_graph_add_host_call_node(
          graph, nullptr, 0, prepare_copy, &parameters, &prepare);
      if (iree_status_is_ok(status)) {
        prepare->attrs.host.user_data_size = sizeof(parameters);
        status = iree_hal_streaming_graph_add_staged_copy_node(
            graph, nullptr, 0, prepare, IREE_HAL_STREAMING_GRAPH_STAGING_UPLOAD,
            destination->device_ptr, sizeof(value), sizeof(value),
            offsetof(CopyParameters, destination), &copy);
      }
      if (iree_status_is_ok(status)) {
        iree_hal_streaming_graph_attach_copy_callback(copy, prepare, false);
        status = iree_hal_streaming_graph_add_host_call_node(
            graph, &copy, 1, increment, &calls[1], &complete);
      }
      graph->arena_allocator = allocator.arena;
      if (iree_status_is_ok(status)) {
        iree_hal_streaming_graph_attach_copy_callback(copy, complete, true);
        break;
      }
      IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
      ++rejected;
      iree_hal_streaming_graph_rollback_copy(graph, checkpoint);
      EXPECT_EQ(graph->node_count, checkpoint.node_count);
      EXPECT_EQ(graph->root_count, checkpoint.root_count);
      EXPECT_EQ(graph->copy_helper_count, checkpoint.helper_count);
    }
    EXPECT_GE(rejected, 3u);
    IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
        graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &executable));
    iree_hal_streaming_graph_exec_launch_result_t result;
    IREE_ASSERT_OK(
        iree_hal_streaming_graph_exec_launch(executable, stream_, &result));
    IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
    uint32_t actual = 0;
    memcpy(&actual, destination->host_ptr, sizeof(actual));
    EXPECT_EQ(value, actual);
    EXPECT_EQ(calls, (std::array<int, 2>{1, 1}));
  }
}

TEST_F(GraphExecTest, ParallelWorkstreamsCarryBothLaunchBoundaries) {
  for (const size_t fill_count : {size_t{1}, size_t{64}}) {
    for (const bool append_barrier : {false, true}) {
      SCOPED_TRACE(fill_count);
      SCOPED_TRACE(append_barrier);
      iree_hal_streaming_buffer_t* storage = nullptr;
      iree_hal_streaming_graph_t* graph = nullptr;
      iree_hal_streaming_graph_exec_t* executable = nullptr;
      GraphBoundaryQueue queue;
      InitializeGraphBoundaryQueue(stream_->queue, &queue);
      auto* original_queue = ReplaceGraphExecTestQueue(stream_, &queue.base);
      ScopeExit release_handles([&] {
        IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
        ReplaceGraphExecTestQueue(stream_, original_queue);
        iree_hal_queue_release(&queue.base);
        iree_hal_streaming_graph_exec_release(executable);
        iree_hal_streaming_graph_release(graph);
        iree_hal_streaming_memory_release_wrapped_buffer(storage);
      });
      IREE_ASSERT_OK(iree_hal_streaming_graph_create(
          context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
          &graph));
      IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_host_staging(
          context_, fill_count, &storage));
      std::vector<iree_hal_streaming_graph_node_t*> fills(fill_count);
      for (size_t i = 0; i < fill_count; ++i) {
        IREE_ASSERT_OK(iree_hal_streaming_graph_add_fill_ptr_node(
            graph, nullptr, 0, storage->device_ptr + i,
            static_cast<uint32_t>(i + 1), 1, 1, &fills[i]));
      }
      if (append_barrier) {
        IREE_ASSERT_OK(iree_hal_streaming_graph_add_empty_node(
            graph, fills.data(), fills.size(), nullptr));
      }
      IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
          graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &executable));
      for (int replay = 0; replay < 2; ++replay) {
        memset(storage->host_ptr, 0, fill_count);
        queue.predecessor = {stream_->timeline_semaphore,
                             stream_->pending_value};
        queue.completions.clear();
        queue.joined = false;
        // The first graph also has a real predecessor; no queue-ordering
        // assumption is allowed even when its timeline has already completed.
        if (queue.predecessor.second == 0) {
          ReplaceGraphExecTestQueue(stream_, original_queue);
          IREE_ASSERT_OK(iree_hal_streaming_launch_host_function(
              stream_, SetFlag, &stream_marker_ran_));
          ReplaceGraphExecTestQueue(stream_, &queue.base);
          queue.predecessor.second = stream_->pending_value;
        }
        iree_hal_streaming_graph_exec_launch_result_t result;
        IREE_ASSERT_OK(
            iree_hal_streaming_graph_exec_launch(executable, stream_, &result));
        EXPECT_EQ(IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS, result);
        IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
        EXPECT_EQ(fill_count > 1 || append_barrier, queue.joined);
        if (fill_count > 1) {
          EXPECT_GT(queue.completions.size(), 1u);
        } else {
          EXPECT_EQ(1u, queue.completions.size());
        }
        const auto* bytes = static_cast<const uint8_t*>(storage->host_ptr);
        for (size_t i = 0; i < fill_count; ++i) {
          EXPECT_EQ(i + 1, bytes[i]);
        }
      }
    }
  }
}

TEST_F(GraphExecTest,
       HostOperationFailureStopsDependentsAndPreservesOtherStreams) {
  struct Operation {
    // Allows the whole graph to be submitted before its dependency fails.
    std::binary_semaphore release{0};
    // Proves the operation reached its allocation dependency.
    std::atomic<bool> invoked{false};
  } operation;
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* executable = nullptr;
  iree_hal_streaming_stream_t* failed_stream = nullptr;
  bool operation_released = false;
  ScopeExit release_handles([&] {
    if (!operation_released) {
      operation.release.release();
    }
    iree_hal_streaming_host_callback_queue_stop(&host_callbacks_);
    iree_hal_streaming_graph_exec_release(executable);
    iree_hal_streaming_graph_release(graph);
    iree_hal_streaming_stream_release(failed_stream);
  });
  IREE_ASSERT_OK(iree_hal_streaming_stream_create(
      context_, context_->queue, IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING,
      /*priority=*/0, iree_allocator_system(), &failed_stream));
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &graph));
  iree_hal_streaming_graph_node_t* operation_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_operation_node(
      graph, nullptr, 0,
      +[](void* user_data) {
        auto* operation = static_cast<Operation*>(user_data);
        operation->release.acquire();
        operation->invoked.store(true, std::memory_order_release);
        const iree_allocator_t allocator = {
            .self = nullptr,
            .ctl =
                +[](void*, iree_allocator_command_t command, const void*,
                    void**) {
                  return command == IREE_ALLOCATOR_COMMAND_FREE
                             ? iree_ok_status()
                             : iree_status_from_code(
                                   IREE_STATUS_RESOURCE_EXHAUSTED);
                },
        };
        void* staging = nullptr;
        return iree_allocator_malloc(allocator, 16, &staging);
      },
      &operation, &operation_node));
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      graph, &operation_node, 1, SetFlag, &graph_host_node_ran_, nullptr));
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &executable));
  iree_hal_streaming_graph_exec_launch_result_t launch_result;
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(executable, failed_stream,
                                                      &launch_result));
  EXPECT_EQ(launch_result, IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS);
  expected_synchronization_status_ = IREE_STATUS_RESOURCE_EXHAUSTED;
  operation.release.release();
  operation_released = true;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_semaphore_wait(failed_stream->timeline_semaphore,
                              failed_stream->pending_value,
                              iree_infinite_timeout(), 0));
  EXPECT_TRUE(operation.invoked.load(std::memory_order_acquire));
  EXPECT_FALSE(graph_host_node_ran_.load(std::memory_order_acquire));
  IREE_ASSERT_OK(iree_hal_streaming_launch_host_function(stream_, SetFlag,
                                                         &stream_marker_ran_));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(stream_->timeline_semaphore,
                                         stream_->pending_value,
                                         iree_infinite_timeout(), 0));
  EXPECT_TRUE(stream_marker_ran_.load(std::memory_order_acquire));
}

TEST_F(GraphExecTest, CopyHelpersShareEnableStateAndTemplateLifetime) {
  std::array<int, 2> calls = {};
  iree_hal_streaming_buffer_t* source = nullptr;
  iree_hal_streaming_buffer_t* destination = nullptr;
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_t* clone = nullptr;
  iree_hal_streaming_graph_exec_t* executable = nullptr;
  ScopeExit release_handles([&] {
    IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
    iree_hal_streaming_graph_exec_release(executable);
    iree_hal_streaming_graph_release(clone);
    iree_hal_streaming_graph_release(graph);
    iree_hal_streaming_memory_release_wrapped_buffer(source);
    iree_hal_streaming_memory_release_wrapped_buffer(destination);
  });
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &graph));
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_host_staging(
      context_, sizeof(uint32_t), &source));
  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_host_staging(
      context_, sizeof(uint32_t), &destination));
  const uint32_t value = 37;
  memcpy(source->host_ptr, &value, sizeof(value));
  memset(destination->host_ptr, 0, sizeof(value));
  const auto increment = +[](void* pointer) { ++*static_cast<int*>(pointer); };
  iree_hal_streaming_graph_node_t* pre_callback = nullptr;
  iree_hal_streaming_graph_node_t* copy = nullptr;
  iree_hal_streaming_graph_node_t* post_callback = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      graph, nullptr, 0, increment, &calls[0], &pre_callback));
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_copy_ptr_node(
      graph, &pre_callback, 1, destination->device_ptr, source->device_ptr,
      sizeof(value), &copy));
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      graph, &copy, 1, increment, &calls[1], &post_callback));
  iree_hal_streaming_graph_attach_copy_callback(copy, pre_callback, false);
  iree_hal_streaming_graph_attach_copy_callback(copy, post_callback, true);
  IREE_ASSERT_OK(iree_hal_streaming_graph_clone(
      graph, IREE_HAL_STREAMING_GRAPH_CLONE_MODE_TEMPLATE, &clone));
  auto* cloned_copy = clone->node_blocks->nodes[1];
  EXPECT_EQ(clone->copy_helper_count, 2u);
  EXPECT_EQ(cloned_copy->attrs.memcpy.pre_callback->attrs.host.copy_node,
            cloned_copy);
  EXPECT_EQ(cloned_copy->attrs.memcpy.post_callback->attrs.host.copy_node,
            cloned_copy);
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      clone, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &executable));
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_set_node_enabled(
      executable, cloned_copy, false));
  iree_hal_streaming_graph_exec_launch_result_t result;
  IREE_ASSERT_OK(
      iree_hal_streaming_graph_exec_launch(executable, stream_, &result));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
  EXPECT_EQ(calls, (std::array<int, 2>{0, 0}));
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_set_node_enabled(
      executable, cloned_copy, true));
  IREE_ASSERT_OK(
      iree_hal_streaming_graph_exec_launch(executable, stream_, &result));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
  EXPECT_EQ(calls, (std::array<int, 2>{1, 1}));
  IREE_ASSERT_OK(iree_hal_streaming_graph_destroy_node(cloned_copy));
  EXPECT_EQ(clone->node_count, 0u);
  EXPECT_EQ(clone->copy_helper_count, 0u);
  // Removing the template operation does not alter the compiled snapshot.
  IREE_ASSERT_OK(
      iree_hal_streaming_graph_exec_launch(executable, stream_, &result));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
  EXPECT_EQ(calls, (std::array<int, 2>{2, 2}));
}

TEST_F(GraphExecTest, ClonedHostOperationOwnsItsPayloadAndExecutableSnapshot) {
  struct Copy {
    // Value captured by the graph and snapshotted at instantiation.
    int value;
    // Borrowed application output, live through stream completion.
    int* output;
  };
  int output = 0;
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_t* clone = nullptr;
  iree_hal_streaming_graph_exec_t* executable = nullptr;
  ScopeExit release_handles([&] {
    IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
    iree_hal_streaming_graph_exec_release(executable);
    iree_hal_streaming_graph_release(clone);
    iree_hal_streaming_graph_release(graph);
  });
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &graph));
  Copy* copy = nullptr;
  IREE_ASSERT_OK(iree_arena_allocate(&graph->arena, sizeof(*copy),
                                     reinterpret_cast<void**>(&copy)));
  *copy = {17, &output};
  iree_hal_streaming_graph_node_t* node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_operation_node(
      graph, nullptr, 0,
      +[](void* user_data) {
        auto* copy = static_cast<Copy*>(user_data);
        *copy->output = copy->value;
        return iree_ok_status();
      },
      copy, &node));
  node->attrs.host.user_data_size = sizeof(*copy);
  IREE_ASSERT_OK(iree_hal_streaming_graph_clone(
      graph, IREE_HAL_STREAMING_GRAPH_CLONE_MODE_TEMPLATE, &clone));
  copy->value = 23;
  iree_hal_streaming_graph_release(graph);
  graph = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      clone, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &executable));
  iree_hal_streaming_graph_node_t* cloned_node = clone->node_blocks->nodes[0];
  auto* cloned_copy = static_cast<Copy*>(cloned_node->attrs.host.user_data);
  EXPECT_EQ(cloned_copy->value, 17);
  cloned_copy->value = 31;
  iree_hal_streaming_graph_exec_launch_result_t launch_result;
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(executable, stream_,
                                                      &launch_result));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
  EXPECT_EQ(output, 17);
}

TEST_F(GraphExecTest, UpdatePreservesParametersOfPendingHostOperations) {
  struct Gate {
    // Publishes that the first callback holds the frontend worker.
    std::binary_semaphore entered{0};
    // Allows each launch to continue after its update/inspection boundary.
    std::binary_semaphore release{0};
  } gate;
  struct Value {
    // Shared by graph snapshots and their pending host operations.
    iree_hal_resource_t resource;
    // Immutable value read by the operation.
    int value;
    // Records reclamation independently of graph metadata lifetime.
    std::atomic<int>* destroyed;
  };
  static const iree_hal_resource_vtable_t value_vtable = {
      .destroy =
          +[](iree_hal_resource_t* resource) {
            auto* value = reinterpret_cast<Value*>(resource);
            value->destroyed->fetch_add(1, std::memory_order_release);
            delete value;
          },
  };
  std::atomic<int> destroyed{0};
  struct Copy {
    // Immutable resource referenced by the snapshotted parameters.
    Value* value;
    // Application output retained until the launch completes.
    int* output;
  };
  int output = 0;
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* executable = nullptr;
  bool gate_released = false;
  ScopeExit release_handles([&] {
    if (!gate_released) {
      gate.release.release();
    }
    IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
    iree_hal_streaming_graph_exec_release(executable);
    iree_hal_streaming_graph_release(graph);
  });
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &graph));
  iree_hal_streaming_graph_node_t* gate_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      graph, nullptr, 0,
      +[](void* user_data) {
        auto* gate = static_cast<Gate*>(user_data);
        gate->entered.release();
        gate->release.acquire();
      },
      &gate, &gate_node));
  Copy* copy = nullptr;
  IREE_ASSERT_OK(iree_arena_allocate(&graph->arena, sizeof(*copy),
                                     reinterpret_cast<void**>(&copy)));
  auto* initial_value = new Value{{}, 17, &destroyed};
  iree_hal_resource_initialize(&value_vtable, &initial_value->resource);
  *copy = {initial_value, &output};
  iree_hal_streaming_graph_node_t* node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_operation_node(
      graph, &gate_node, 1,
      +[](void* user_data) {
        const auto* copy = static_cast<const Copy*>(user_data);
        *copy->output = copy->value->value;
        return iree_ok_status();
      },
      copy, &node));
  node->attrs.host.user_data_size = sizeof(*copy);
  node->attrs.host.resource = &initial_value->resource;
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &executable));
  iree_hal_streaming_graph_exec_launch_result_t launch_result;
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(executable, stream_,
                                                      &launch_result));
  gate.entered.acquire();

  // Rebuilding releases the compiled arena while the old launch still owns a
  // pending operation. Its captured parameters must survive that release.
  auto* updated_value = new Value{{}, 31, &destroyed};
  iree_hal_resource_initialize(&value_vtable, &updated_value->resource);
  iree_hal_resource_release(node->attrs.host.resource);
  node->attrs.host.resource = &updated_value->resource;
  copy->value = updated_value;
  iree_hal_streaming_graph_node_t* error_node = nullptr;
  iree_hal_streaming_graph_exec_update_result_t update_result;
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_update(
      executable, graph, &error_node, &update_result));
  EXPECT_EQ(update_result, IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_SUCCESS);
  EXPECT_EQ(destroyed.load(std::memory_order_acquire), 0);
  gate.release.release();
  gate_released = true;
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
  EXPECT_EQ(output, 17);
  EXPECT_EQ(destroyed.load(std::memory_order_acquire), 1);

  gate.release.release();
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(executable, stream_,
                                                      &launch_result));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
  EXPECT_EQ(output, 31);
}

TEST_F(GraphExecTest, FailedUpdatePreservesThePreviousExecutable) {
  struct AllocationControl {
    // Number of graph-owned host allocations attempted during this update.
    size_t allocation_count = 0;
    // Allocation to reject, or SIZE_MAX for ordinary allocation behavior.
    size_t fail_index = SIZE_MAX;
  } allocation_control;
  const iree_allocator_t allocator = {
      .self = &allocation_control,
      .ctl =
          +[](void* self, iree_allocator_command_t command,
              const void* parameters, void** pointer) {
            auto* control = static_cast<AllocationControl*>(self);
            if (command != IREE_ALLOCATOR_COMMAND_FREE &&
                control->allocation_count++ == control->fail_index) {
              return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
            }
            const auto system = iree_allocator_system();
            return system.ctl(system.self, command, parameters, pointer);
          },
  };
  std::array<int, 3> calls = {};
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_t* replacement = nullptr;
  iree_hal_streaming_graph_exec_t* executable = nullptr;
  ScopeExit release_handles([&] {
    IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
    iree_hal_streaming_graph_exec_release(executable);
    iree_hal_streaming_graph_release(replacement);
    iree_hal_streaming_graph_release(graph);
  });
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, allocator, &graph));
  const auto increment = +[](void* pointer) { ++*static_cast<int*>(pointer); };
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      graph, nullptr, 0, increment, &calls[0], nullptr));
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &executable));
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, allocator, &replacement));
  iree_hal_streaming_graph_node_t* replacement_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      replacement, nullptr, 0, increment, &calls[1], &replacement_node));

  allocation_control.allocation_count = 0;
  iree_hal_streaming_graph_node_t* error_node = nullptr;
  iree_hal_streaming_graph_exec_update_result_t update_result;
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_update(
      executable, replacement, &error_node, &update_result));
  const size_t allocation_count = allocation_control.allocation_count;
  ASSERT_GT(allocation_count, 0u);
  replacement_node->attrs.host.user_data = &calls[2];
  for (size_t failure_index = 0; failure_index < allocation_count;
       ++failure_index) {
    SCOPED_TRACE(failure_index);
    allocation_control.allocation_count = 0;
    allocation_control.fail_index = failure_index;
    iree_status_t status = iree_hal_streaming_graph_exec_update(
        executable, replacement, &error_node, &update_result);
    allocation_control.fail_index = SIZE_MAX;
    IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    EXPECT_EQ(update_result, IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_ERROR);
    iree_hal_streaming_graph_exec_launch_result_t launch_result;
    IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(executable, stream_,
                                                        &launch_result));
    IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
    EXPECT_EQ(calls[0], 0);
    EXPECT_EQ(calls[1], static_cast<int>(failure_index + 1));
    EXPECT_EQ(calls[2], 0);
  }
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_update(
      executable, replacement, &error_node, &update_result));
  iree_hal_streaming_graph_exec_launch_result_t launch_result;
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(executable, stream_,
                                                      &launch_result));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
  EXPECT_EQ(calls[2], 1);
}

TEST_F(GraphExecTest, BatchMemoryNodePreservesResolvedWriteSemantics) {
  iree_hal_pool_t* memory_pool = nullptr;
  iree_hal_buffer_t* buffer = nullptr;
  iree_hal_buffer_mapping_t mapping = {};
  iree_hal_streaming_retained_buffer_ref_t target = {};
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_t* clone_graph = nullptr;
  iree_hal_streaming_graph_exec_t* executable = nullptr;
  ScopeExit release_handles([&] {
    iree_hal_streaming_graph_exec_release(executable);
    iree_hal_streaming_graph_release(clone_graph);
    iree_hal_streaming_graph_release(graph);
    iree_hal_streaming_retained_buffer_ref_deinitialize(&target);
    if (mapping.contents.data) {
      IREE_EXPECT_OK(iree_hal_buffer_unmap_range(&mapping));
    }
    iree_hal_buffer_release(buffer);
    iree_hal_pool_release(memory_pool);
  });

  const hrx_memory_type_t memory_type =
      HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE;
  IREE_ASSERT_OK(hrx_mem_pool_backing_retain_allocation_source(
      device_entry_.hrx_device, memory_type, HRX_BUFFER_USAGE_DEFAULT,
      &memory_pool));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      memory_pool,
      iree_hal_make_pool_reservation_request(sizeof(uint64_t),
                                             /*min_alignment=*/0),
      iree_infinite_timeout(), &buffer));
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_PERSISTENT, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_MAP_FLAG_NONE,
      /*local_byte_offset=*/0, sizeof(uint64_t), &mapping));
  ASSERT_NE(mapping.contents.data, nullptr);
  *reinterpret_cast<uint64_t*>(mapping.contents.data) = 0;
  IREE_ASSERT_OK(hrx_buffer_create_from_hal(
      buffer, device_entry_.hrx_device->group, memory_type, sizeof(uint64_t),
      mapping.contents.data, &target.owner));
  target.buffer = buffer;
  iree_hal_buffer_retain(target.buffer);
  target.memory_type = memory_type;
  target.allocation_size = sizeof(uint64_t);
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &graph));

  const iree_hal_atomic_store_params_t store_params = {
      .value = 47,
      .flags = IREE_HAL_ATOMIC_FLAG_RELEASE | IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
      .width = IREE_HAL_ATOMIC_WIDTH_64,
      .reserved = {},
  };
  iree_hal_streaming_value_operation_t operation = {
      .kind = IREE_HAL_STREAMING_VALUE_OPERATION_STORE,
      .target_buffer = target.buffer,
      .target_offset = target.offset,
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

  IREE_ASSERT_OK(iree_hal_streaming_graph_clone(
      graph, IREE_HAL_STREAMING_GRAPH_CLONE_MODE_TEMPLATE, &clone_graph));
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      clone_graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE,
      &executable));
  iree_hal_streaming_graph_exec_launch_result_t launch_result =
      IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_ERROR;
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_launch(executable, stream_,
                                                      &launch_result));
  EXPECT_EQ(IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS, launch_result);
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
  EXPECT_EQ(47u, *reinterpret_cast<uint64_t*>(mapping.contents.data));
}

TEST_F(GraphExecTest,
       ExternalCaptureKeepsTheIncomingFrontierAndReplaysTheRecord) {
  iree_hal_streaming_event_t* first = nullptr;
  iree_hal_streaming_event_t* second = nullptr;
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* executable = nullptr;
  ScopeExit release_handles([&] {
    iree_hal_streaming_graph_exec_release(executable);
    if (first) {
      iree_hal_streaming_event_clear_capture(first);
    }
    if (second) {
      iree_hal_streaming_event_clear_capture(second);
    }
    iree_hal_streaming_graph_release(graph);
    iree_hal_streaming_event_release(first);
    iree_hal_streaming_event_release(second);
  });
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &first));
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING,
      iree_allocator_system(), &second));
  IREE_ASSERT_OK(iree_hal_streaming_begin_capture(
      stream_, IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL));
  IREE_ASSERT_OK(iree_hal_streaming_event_record(
      first, stream_, /*capture_external_record=*/true));
  IREE_ASSERT_OK(iree_hal_streaming_event_record(
      second, stream_, /*capture_external_record=*/true));
  IREE_ASSERT_OK(iree_hal_streaming_end_capture(stream_, &graph));
  ASSERT_EQ(2u, graph->node_count);
  auto* first_node = graph->node_blocks->nodes[0];
  auto* second_node = graph->node_blocks->nodes[1];
  ASSERT_EQ(IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD, first_node->type);
  ASSERT_EQ(IREE_HAL_STREAMING_GRAPH_NODE_TYPE_EVENT_RECORD, second_node->type);
  EXPECT_EQ(first, first_node->attrs.event.event);
  EXPECT_EQ(second, second_node->attrs.event.event);
  ASSERT_EQ(1u, second_node->dependency_count);
  EXPECT_EQ(first_node, second_node->dependencies[0]);
  EXPECT_EQ(0u, first->capture_dependency_count);
  ASSERT_EQ(1u, second->capture_dependency_count);
  EXPECT_EQ(first_node, second->capture_dependencies[0]);

  // Retiring the public handles detaches metadata, while graph-owned event
  // references continue to serve each later launch.
  iree_hal_streaming_event_clear_capture(first);
  iree_hal_streaming_event_clear_capture(second);
  EXPECT_EQ(nullptr, first->capture_graph);
  EXPECT_EQ(nullptr, second->capture_graph);
  // Borrowed from the graph node after the caller releases its references.
  iree_hal_streaming_event_t* recorded_second = second;
  iree_hal_streaming_event_release(first);
  first = nullptr;
  iree_hal_streaming_event_release(second);
  second = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &executable));
  for (int launch = 0; launch < 2; ++launch) {
    iree_hal_streaming_graph_exec_launch_result_t result;
    IREE_ASSERT_OK(
        iree_hal_streaming_graph_exec_launch(executable, stream_, &result));
    EXPECT_EQ(IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS, result);
    IREE_ASSERT_OK(iree_hal_streaming_event_synchronize(recorded_second));
    EXPECT_EQ(nullptr, recorded_second->capture_graph);
  }
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
    iree_hal_streaming_graph_exec_release(exec);
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
    IREE_ASSERT_OK(iree_hal_streaming_event_record(
        event, stream_, /*capture_external_record=*/false));
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

  // A graph holds a reference to the context that created it and drops it when
  // it is destroyed, so one release across the launch is the capture graph and
  // nothing else.
  const int32_t context_references_before =
      iree_atomic_ref_count_load(&context_->ref_count);
  iree_hal_streaming_graph_exec_launch_result_t launch_result =
      IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_ERROR;
  IREE_ASSERT_OK(
      iree_hal_streaming_graph_exec_launch(exec, stream_, &launch_result));
  EXPECT_EQ(IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS, launch_result);
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));

  for (iree_hal_streaming_event_t* event : events) {
    EXPECT_EQ(event->capture_graph, nullptr);
  }
  EXPECT_EQ(context_references_before - 1,
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
  std::atomic<bool> original_called = false;
  std::atomic<bool> replacement_called = false;
  iree_hal_streaming_context_t* other_context = nullptr;
  iree_hal_streaming_event_t* event = nullptr;
  iree_hal_streaming_event_t* replacement = nullptr;
  iree_hal_streaming_event_t* other_context_event = nullptr;
  iree_hal_streaming_graph_t* graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  // Hands back whatever was built, on the assertion failure paths as much as
  // on the last line.
  ScopeExit release_handles([&] {
    IREE_EXPECT_OK(iree_hal_streaming_stream_synchronize(stream_));
    iree_hal_streaming_graph_exec_release(exec);
    iree_hal_streaming_graph_release(graph);
    iree_hal_streaming_event_release(other_context_event);
    iree_hal_streaming_event_release(replacement);
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_context_release(other_context);
  });

  iree_hal_streaming_context_flags_t context_flags = {
      .scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO,
  };
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
  iree_hal_streaming_graph_node_t* host_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      graph, &wait_node, 1, SetFlag, &original_called, &host_node));

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

  // A later parameter update recompiles the graph and must preserve both
  // event replacements while leaving their template nodes untouched.
  auto* private_node = iree_hal_streaming_graph_exec_lock_node(exec, host_node);
  private_node->attrs.host.user_data = &replacement_called;
  status = iree_hal_streaming_graph_exec_rebuild_locked(exec);
  iree_hal_streaming_graph_exec_unlock(exec);
  IREE_ASSERT_OK(status);
  iree_hal_streaming_graph_exec_launch_result_t launch_result;
  IREE_ASSERT_OK(
      iree_hal_streaming_graph_exec_launch(exec, stream_, &launch_result));
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
  EXPECT_FALSE(original_called.load());
  EXPECT_TRUE(replacement_called.load());
  EXPECT_EQ(record_node->attrs.event.event, event);
  EXPECT_EQ(wait_node->attrs.event.event, event);
  iree_hal_streaming_recorded_point_t point;
  iree_hal_streaming_event_acquire_recorded_point(event, &point);
  EXPECT_EQ(point.semaphore, nullptr);
  iree_hal_streaming_event_release_recorded_point(&point);
  iree_hal_streaming_event_acquire_recorded_point(replacement, &point);
  EXPECT_NE(point.semaphore, nullptr);
  iree_hal_streaming_event_release_recorded_point(&point);
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
    iree_hal_streaming_graph_exec_release(exec);
    iree_hal_streaming_graph_release(parent_graph);
    iree_hal_streaming_graph_release(child_graph);
    iree_hal_streaming_event_release(event);
    iree_hal_streaming_stream_release(other_stream);
    iree_hal_streaming_context_release(other_context);
  });

  iree_hal_streaming_context_flags_t context_flags = {
      .scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO,
  };
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

  iree_hal_streaming_graph_exec_launch_result_t launch_result =
      IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS;
  iree_status_t status =
      iree_hal_streaming_graph_exec_launch(exec, other_stream, &launch_result);
  EXPECT_EQ(IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_ERROR, launch_result);
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

// An updated executable carries the cooperative requirement of a nested graph
// to the final launch. The launch must query the actual cooperative queue and
// reject the whole executable before reserving a stream point or submitting
// the host-call block that precedes the child.
TEST_F(GraphExecTest,
       UpdatedChildCooperativeDispatchPreflightsBeforePrefixAndReservation) {
  AtomicPreflightExecutable test_executable = {};
  iree_hal_streaming_module_t* module = nullptr;
  iree_hal_streaming_graph_t* original_child_graph = nullptr;
  iree_hal_streaming_graph_t* original_parent_graph = nullptr;
  iree_hal_streaming_graph_t* updated_child_graph = nullptr;
  iree_hal_streaming_graph_t* updated_parent_graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  AtomicPreflightQueue test_queue = {};
  iree_hal_queue_t* original_queue = nullptr;
  bool wrapper_installed = false;

  void* module_storage = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(), sizeof(*module),
                                       &module_storage));
  module = static_cast<iree_hal_streaming_module_t*>(module_storage);
  memset(module, 0, sizeof(*module));
  iree_atomic_ref_count_init(&module->ref_count);
  iree_slim_mutex_initialize(&module->global_mutex);
  module->host_allocator = iree_allocator_system();
  iree_hal_executable_initialize(iree_hal_queue_family(stream_->queue),
                                 &kAtomicPreflightExecutableVtable,
                                 &test_executable.base);
  // The module owns the executable's initial reference, just as a loaded
  // module does. Its final release below tears both down.
  module->executable = &test_executable.base;

  ScopeExit release_handles([&] {
    if (wrapper_installed) {
      iree_hal_queue_retain(original_queue);
      iree_hal_queue_t* installed_queue =
          ReplaceGraphExecTestQueue(stream_, original_queue);
      iree_hal_queue_release(installed_queue);
    }
    iree_hal_streaming_graph_exec_release(exec);
    iree_hal_streaming_graph_release(updated_parent_graph);
    iree_hal_streaming_graph_release(updated_child_graph);
    iree_hal_streaming_graph_release(original_parent_graph);
    iree_hal_streaming_graph_release(original_child_graph);
    iree_hal_streaming_module_release(module);
  });

  iree_hal_streaming_symbol_t symbol = {
      .module = module,
      .type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION,
      .executable = module->executable,
      .export_ordinal = 0,
  };

  const iree_hal_streaming_dispatch_params_t original_dispatch = {
      .grid_dim = {1, 1, 1},
      .block_dim = {1, 1, 1},
      .shared_memory_bytes = 0,
      .buffer = nullptr,
      .buffer_size = 0,
      .flags = IREE_HAL_STREAMING_DISPATCH_FLAG_COOPERATIVE,
      .workitem_count = {},
      .binding_function = nullptr,
  };
  iree_hal_streaming_dispatch_params_t updated_dispatch = original_dispatch;
  updated_dispatch.grid_dim[0] = 2;

  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &original_child_graph));
  iree_hal_streaming_graph_node_t* original_kernel_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_kernel_node(
      original_child_graph, /*dependencies=*/nullptr, /*dependency_count=*/0,
      &symbol, &original_dispatch, &original_kernel_node));
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &original_parent_graph));
  iree_hal_streaming_graph_node_t* original_host_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      original_parent_graph, /*dependencies=*/nullptr, /*dependency_count=*/0,
      &SetFlag, &graph_host_node_ran_, &original_host_node));
  iree_hal_streaming_graph_node_t* original_child_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_child_graph_node(
      original_parent_graph, &original_host_node, /*dependency_count=*/1,
      original_child_graph, &original_child_node));

  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &updated_child_graph));
  iree_hal_streaming_graph_node_t* updated_kernel_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_kernel_node(
      updated_child_graph, /*dependencies=*/nullptr, /*dependency_count=*/0,
      &symbol, &updated_dispatch, &updated_kernel_node));
  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &updated_parent_graph));
  iree_hal_streaming_graph_node_t* updated_host_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      updated_parent_graph, /*dependencies=*/nullptr, /*dependency_count=*/0,
      &SetFlag, &graph_host_node_ran_, &updated_host_node));
  iree_hal_streaming_graph_node_t* updated_child_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_child_graph_node(
      updated_parent_graph, &updated_host_node, /*dependency_count=*/1,
      updated_child_graph, &updated_child_node));

  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      original_parent_graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE,
      &exec));
  iree_hal_streaming_graph_node_t* update_error_node = nullptr;
  iree_hal_streaming_graph_exec_update_result_t update_result =
      IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_ERROR;
  IREE_ASSERT_OK(iree_hal_streaming_graph_exec_update(
      exec, updated_parent_graph, &update_error_node, &update_result));
  EXPECT_EQ(IREE_HAL_STREAMING_GRAPH_EXEC_UPDATE_SUCCESS, update_result);
  EXPECT_EQ(nullptr, update_error_node);

  original_queue = stream_->queue;
  InitializeAtomicPreflightQueue(original_queue, &test_queue);
  EXPECT_EQ(original_queue,
            ReplaceGraphExecTestQueue(stream_, &test_queue.base));
  // Transfer the stream's old queue reference to the wrapper target.
  iree_hal_queue_release(original_queue);
  wrapper_installed = true;

  iree_slim_mutex_lock(&stream_->mutex);
  const uint64_t initial_pending_value = stream_->pending_value;
  iree_slim_mutex_unlock(&stream_->mutex);
  uint64_t initial_timeline_value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(stream_->timeline_semaphore,
                                          &initial_timeline_value));

  iree_hal_streaming_graph_exec_launch_result_t launch_result =
      IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS;
  iree_status_t status =
      iree_hal_streaming_graph_exec_launch(exec, stream_, &launch_result);
  EXPECT_EQ(IREE_STATUS_OUT_OF_RANGE, iree_status_code(status));
  EXPECT_EQ(IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_COOPERATIVE_TOO_LARGE,
            launch_result);
  iree_status_free(status);
  EXPECT_EQ(1,
            test_queue.concurrency_query_count.load(std::memory_order_acquire));
  EXPECT_EQ(0, test_queue.submission_count.load(std::memory_order_acquire));
  EXPECT_FALSE(graph_host_node_ran_.load(std::memory_order_acquire));

  iree_slim_mutex_lock(&stream_->mutex);
  const uint64_t pending_after_prefix_check = stream_->pending_value;
  // A timeline at its maximum makes any reservation fail with overflow.
  // Cooperative preflight must therefore win error precedence and keep the
  // otherwise side-effect-free reservation path unobserved.
  stream_->pending_value = IREE_HAL_SEMAPHORE_MAX_VALUE;
  iree_slim_mutex_unlock(&stream_->mutex);
  EXPECT_EQ(initial_pending_value, pending_after_prefix_check);

  launch_result = IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS;
  status = iree_hal_streaming_graph_exec_launch(exec, stream_, &launch_result);
  EXPECT_EQ(IREE_STATUS_OUT_OF_RANGE, iree_status_code(status));
  EXPECT_EQ(IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_COOPERATIVE_TOO_LARGE,
            launch_result);
  iree_status_free(status);

  iree_slim_mutex_lock(&stream_->mutex);
  const uint64_t pending_after_reservation_check = stream_->pending_value;
  stream_->pending_value = initial_pending_value;
  iree_slim_mutex_unlock(&stream_->mutex);
  EXPECT_EQ(IREE_HAL_SEMAPHORE_MAX_VALUE, pending_after_reservation_check);
  EXPECT_EQ(2,
            test_queue.concurrency_query_count.load(std::memory_order_acquire));
  EXPECT_EQ(1u, test_queue.queried_workgroup_size[0]);
  EXPECT_EQ(1u, test_queue.queried_workgroup_size[1]);
  EXPECT_EQ(1u, test_queue.queried_workgroup_size[2]);
  EXPECT_EQ(0, test_queue.submission_count.load(std::memory_order_acquire));

  uint64_t final_timeline_value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(stream_->timeline_semaphore,
                                          &final_timeline_value));
  EXPECT_EQ(initial_timeline_value, final_timeline_value);
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
  GraphLaunchObservation launch_observation;
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
    iree_hal_streaming_graph_exec_release(exec);
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
      child_graph, /*dependencies=*/nullptr, /*dependency_count=*/0,
      &ObserveGraphLaunchPrefix, &launch_observation, &first_node));
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
  InitializeRejectSecondHostCallQueue(original_queue, &launch_observation,
                                      &fault_queue);
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

  std::atomic<iree_status_code_t> launch_status_code = IREE_STATUS_UNKNOWN;
  std::thread launch_thread([&] {
    iree_hal_streaming_graph_exec_launch_result_t launch_result =
        IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS;
    iree_status_t status =
        iree_hal_streaming_graph_exec_launch(exec, stream_, &launch_result);
    EXPECT_EQ(IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_ERROR, launch_result);
    launch_status_code.store(iree_status_code(status),
                             std::memory_order_release);
    iree_status_ignore(status);
    {
      std::lock_guard<std::mutex> lock(launch_observation.mutex);
      launch_observation.launch_returned = true;
    }
    launch_observation.notification.notify_all();
  });

  bool observed_later_rejection = false;
  {
    std::unique_lock<std::mutex> lock(launch_observation.mutex);
    launch_observation.notification.wait(lock, [&] {
      return launch_observation.later_call_rejected ||
             launch_observation.launch_returned;
    });
    observed_later_rejection = launch_observation.later_call_rejected;
  }
  if (!observed_later_rejection) {
    IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, 1, /*frontier=*/nullptr));
    launch_thread.join();
    if (fault_queue.accepted_host_call_count.load(std::memory_order_acquire) >
        0) {
      std::unique_lock<std::mutex> lock(launch_observation.mutex);
      launch_observation.notification.wait(
          lock, [&] { return launch_observation.prefix_call_ran; });
    }
    FAIL() << "recursive launch did not reach the injected later rejection";
    return;
  }
  EXPECT_EQ(
      1, fault_queue.accepted_host_call_count.load(std::memory_order_acquire));

  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, 1, /*frontier=*/nullptr));
  launch_thread.join();
  EXPECT_EQ(IREE_STATUS_ABORTED,
            launch_status_code.load(std::memory_order_acquire));
  {
    std::unique_lock<std::mutex> lock(launch_observation.mutex);
    launch_observation.notification.wait(
        lock, [&] { return launch_observation.prefix_call_ran; });
    EXPECT_FALSE(launch_observation.launch_returned_before_prefix)
        << "launch returned before its accepted prefix completed";
  }
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
  iree_hal_streaming_graph_exec_release(exec);
  exec = nullptr;
}

}  // namespace
