// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>

#include "common/context.h"
#include "common/device.h"
#include "common/graph.h"
#include "common/host_callback.h"
#include "common/hrx_bridge.h"
#include "common/stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libhrx/src/libhrx/device.h"

namespace {

void SetFlag(void* data) {
  static_cast<std::atomic<bool>*>(data)->store(true, std::memory_order_release);
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
  // Number of accepted callbacks that finished.
  int completed_prefix_count = 0;
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
    ++observation->completed_prefix_count;
  }
  observation->notification.notify_all();
}

struct RejectGraphQueue {
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
  // Zero-based call to reject once; later retry calls reach the real queue.
  int rejected_call_index = 1;
  // Zero-based command-buffer execution to reject once, or -1 for none.
  int rejected_execute_index = -1;
  // Number of command buffers presented to this wrapper.
  std::atomic<int> execute_count = 0;
  // Number of host calls presented to this wrapper.
  std::atomic<int> host_call_count = 0;
  // Number of host calls accepted by the target queue.
  std::atomic<int> accepted_host_call_count = 0;
};

RejectGraphQueue* CastRejectGraphQueue(iree_hal_queue_t* base_queue) {
  return reinterpret_cast<RejectGraphQueue*>(base_queue);
}

void DestroyRejectGraphQueue(iree_hal_queue_t* base_queue) {
  auto* queue = CastRejectGraphQueue(base_queue);
  iree_hal_queue_release(queue->target);
  queue->target = nullptr;
}

iree_status_t RejectGraphQueueBarrier(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hal_queue_barriers_t* barriers,
    iree_hal_queue_barrier_flags_t flags) {
  return iree_hal_queue_barrier(CastRejectGraphQueue(base_queue)->target,
                                wait_semaphore_list, signal_semaphore_list,
                                barriers, flags);
}

iree_status_t RejectGraphSubmission(RejectGraphQueue* queue) {
  {
    std::lock_guard<std::mutex> lock(queue->observation->mutex);
    queue->observation->later_call_rejected = true;
  }
  queue->observation->notification.notify_all();
  return iree_make_status(IREE_STATUS_ABORTED,
                          "injected later graph block rejection");
}

iree_status_t RejectGraphQueueExecute(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_queue_execute_flags_t flags) {
  auto* queue = CastRejectGraphQueue(base_queue);
  const int execute_index = queue->execute_count.fetch_add(1);
  if (execute_index == queue->rejected_execute_index) {
    return RejectGraphSubmission(queue);
  }
  return iree_hal_queue_execute(queue->target, wait_semaphore_list,
                                signal_semaphore_list, command_buffer,
                                binding_table, flags);
}

iree_status_t RejectGraphQueueHostCall(
    iree_hal_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  auto* queue = CastRejectGraphQueue(base_queue);
  const int call_index =
      queue->host_call_count.fetch_add(1, std::memory_order_acq_rel);
  if (call_index == queue->rejected_call_index) {
    return RejectGraphSubmission(queue);
  }
  if (call_index != 0 || !queue->first_call_gate) {
    iree_status_t status =
        iree_hal_queue_host_call(queue->target, wait_semaphore_list,
                                 signal_semaphore_list, call, args, flags);
    if (iree_status_is_ok(status)) {
      queue->accepted_host_call_count.fetch_add(1, std::memory_order_acq_rel);
    }
    return status;
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

iree_status_t RejectGraphQueueFlush(iree_hal_queue_t* base_queue) {
  return iree_hal_queue_flush(CastRejectGraphQueue(base_queue)->target);
}

const iree_hal_queue_vtable_t kRejectGraphQueueVtable = {
    .destroy = DestroyRejectGraphQueue,
    .barrier = RejectGraphQueueBarrier,
    .execute = RejectGraphQueueExecute,
    .host_call = RejectGraphQueueHostCall,
    .query_dispatch_concurrency = nullptr,
    .dispatch = nullptr,
    .atomic_wait = nullptr,
    .atomic_store = nullptr,
    .atomic_rmw = nullptr,
    .timestamp = nullptr,
    .flush = RejectGraphQueueFlush,
    .alloca = nullptr,
    .dealloca = nullptr,
    .transfer = nullptr,
    .read = nullptr,
    .write = nullptr,
};

void InitializeRejectGraphQueue(iree_hal_queue_t* target,
                                GraphLaunchObservation* observation,
                                RejectGraphQueue* out_queue) {
  out_queue->target = target;
  out_queue->observation = observation;
  iree_hal_queue_retain(target);
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  params.priority = iree_hal_queue_priority(target);
  params.features = iree_hal_queue_features(target);
  params.execution_resources = iree_hal_queue_execution_resources(target);
  iree_hal_queue_initialize(iree_hal_queue_family(target), &params,
                            &kRejectGraphQueueVtable, &out_queue->base);
}

iree_hal_queue_t* ReplaceGraphExecTestQueue(iree_hal_streaming_stream_t* stream,
                                            iree_hal_queue_t* replacement) {
  iree_slim_mutex_lock(&stream->mutex);
  iree_hal_queue_t* previous = stream->queue;
  stream->queue = replacement;
  iree_slim_mutex_unlock(&stream->mutex);
  return previous;
}

class GraphSubmissionTest
    : public ::testing::TestWithParam<std::tuple<int, bool, bool, bool>> {
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
      IREE_EXPECT_OK(iree_hal_streaming_context_synchronize(context_));
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

  // Explicit frontend-owned callout service borrowed by the test device entry.
  iree_hal_streaming_host_callback_queue_t host_callbacks_ = {};
  // Registry entry backing |context_|; outlives every context created from it.
  iree_hal_streaming_device_t device_entry_ = {};
  // Context owning the graphs, events, and streams each test builds.
  iree_hal_streaming_context_t* context_ = nullptr;
  // Stream every launch submits on.
  iree_hal_streaming_stream_t* stream_ = nullptr;
  // Rejected suffix marker. Lives through fixture teardown so accepted
  // callbacks cannot outlive it if an earlier assertion fails.
  std::atomic<bool> stream_marker_ran_{false};
};

// A rejection inside or after a child must not let accepted blocks outlive
// graph-exec state. The rejected final block cannot signal the reserved stream
// value, so the launch drains the accepted prefix with pending_value unchanged.
TEST_P(GraphSubmissionTest,
       RejectedSuffixDrainsAcceptedChildrenAndPreservesStreamTail) {
  const int child_prefix_count = std::get<0>(GetParam());
  const bool reject_parent_suffix = std::get<1>(GetParam());
  const bool parallel_prefix = std::get<2>(GetParam());
  const bool reject_recorded_block = std::get<3>(GetParam());
  const int accepted_count = child_prefix_count + 1;
  iree_hal_streaming_graph_t* child_graph = nullptr;
  iree_hal_streaming_graph_t* parent_graph = nullptr;
  iree_hal_streaming_graph_exec_t* exec = nullptr;
  iree_hal_semaphore_t* gate = nullptr;
  iree_hal_streaming_buffer_t* storage = nullptr;
  GraphLaunchObservation launch_observation;
  RejectGraphQueue fault_queue = {};
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
    iree_hal_streaming_memory_release_wrapped_buffer(storage);
  });

  IREE_ASSERT_OK(iree_hal_streaming_memory_allocate_host_staging(
      context_, sizeof(uint32_t), &storage));
  *static_cast<uint32_t*>(storage->host_ptr) = 0;
  const auto append_rejected_suffix =
      [&](iree_hal_streaming_graph_t* graph,
          iree_hal_streaming_graph_node_t* tail) {
        if (reject_recorded_block) {
          return iree_hal_streaming_graph_add_fill_ptr_node(
              graph, &tail, 1, storage->device_ptr, 34, sizeof(uint32_t), 1,
              nullptr);
        }
        return iree_hal_streaming_graph_add_host_call_node(
            graph, &tail, 1, &SetFlag, &stream_marker_ran_, nullptr);
      };
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
  for (int i = 1; i < child_prefix_count; ++i) {
    IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
        child_graph, parallel_prefix ? nullptr : &first_node,
        parallel_prefix ? 0 : 1, &ObserveGraphLaunchPrefix, &launch_observation,
        &first_node));
  }
  if (!reject_parent_suffix) {
    IREE_ASSERT_OK(append_rejected_suffix(child_graph, first_node));
  }

  IREE_ASSERT_OK(iree_hal_streaming_graph_create(
      context_, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, iree_allocator_system(),
      &parent_graph));
  iree_hal_streaming_graph_node_t* parent_prefix = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_host_call_node(
      parent_graph, nullptr, 0, &ObserveGraphLaunchPrefix, &launch_observation,
      &parent_prefix));
  // A real recorded transfer separates the parent's frontend operation from
  // the child. Its output and completion must survive a later rejected block.
  iree_hal_streaming_graph_node_t* recorded_prefix = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_fill_ptr_node(
      parent_graph, &parent_prefix, 1, storage->device_ptr, 17,
      sizeof(uint32_t), 1, &recorded_prefix));
  iree_hal_streaming_graph_node_t* child_node = nullptr;
  IREE_ASSERT_OK(iree_hal_streaming_graph_add_child_graph_node(
      parent_graph, &recorded_prefix, 1, child_graph, &child_node));
  if (reject_parent_suffix) {
    IREE_ASSERT_OK(append_rejected_suffix(parent_graph, child_node));
  }

  IREE_ASSERT_OK(iree_hal_streaming_graph_instantiate(
      parent_graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec));

  original_queue = stream_->queue;
  InitializeRejectGraphQueue(original_queue, &launch_observation, &fault_queue);
  EXPECT_EQ(original_queue,
            ReplaceGraphExecTestQueue(stream_, &fault_queue.base));
  // Transfer the stream's old queue reference to the wrapper target.
  iree_hal_queue_release(original_queue);
  wrapper_installed = true;
  fault_queue.rejected_call_index = reject_recorded_block ? -1 : accepted_count;
  fault_queue.rejected_execute_index = reject_recorded_block ? 1 : -1;
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
  EXPECT_EQ(accepted_count, fault_queue.accepted_host_call_count.load(
                                std::memory_order_acquire));

  IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, 1, /*frontier=*/nullptr));
  launch_thread.join();
  EXPECT_EQ(IREE_STATUS_ABORTED,
            launch_status_code.load(std::memory_order_acquire));
  {
    std::unique_lock<std::mutex> lock(launch_observation.mutex);
    launch_observation.notification.wait(lock, [&] {
      return launch_observation.completed_prefix_count == accepted_count;
    });
    EXPECT_FALSE(launch_observation.launch_returned_before_prefix)
        << "launch returned before its accepted prefix completed";
  }
  EXPECT_FALSE(stream_marker_ran_.load(std::memory_order_acquire));
  EXPECT_EQ(17u, *static_cast<uint32_t*>(storage->host_ptr));

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

  // The rejected suffix did not run. A retry must advance past the accepted
  // child's internal point without leaving any wait on the rejected tail.
  iree_hal_streaming_graph_exec_launch_result_t retry_result;
  IREE_ASSERT_OK(
      iree_hal_streaming_graph_exec_launch(exec, stream_, &retry_result));
  EXPECT_EQ(IREE_HAL_STREAMING_GRAPH_EXEC_LAUNCH_SUCCESS, retry_result);
  IREE_ASSERT_OK(iree_hal_streaming_stream_synchronize(stream_));
  if (reject_recorded_block) {
    EXPECT_EQ(34u, *static_cast<uint32_t*>(storage->host_ptr));
  } else {
    EXPECT_TRUE(stream_marker_ran_.load(std::memory_order_acquire));
  }

  iree_hal_streaming_graph_exec_release(exec);
  exec = nullptr;
}

INSTANTIATE_TEST_SUITE_P(Prefixes, GraphSubmissionTest,
                         ::testing::Combine(::testing::Values(1, 17, 64),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool()));

}  // namespace
