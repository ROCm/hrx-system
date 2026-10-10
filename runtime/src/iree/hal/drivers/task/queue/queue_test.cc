// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/task/queue/queue.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <thread>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/device_group.h"
#include "iree/hal/drivers/task/device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TaskQueueTest : public ::testing::TestWithParam<iree_host_size_t> {
 protected:
  void SetUp() override {
    iree_task_topology_t topology;
    iree_task_topology_initialize_from_group_count(GetParam(), &topology);
    iree_task_executor_options_t executor_options;
    iree_task_executor_options_initialize(&executor_options);
    iree_status_t executor_status = iree_task_executor_create(
        executor_options, &topology, iree_allocator_system(), &executor_);
    iree_task_topology_deinitialize(&topology);
    IREE_ASSERT_OK(executor_status);

    IREE_ASSERT_OK(iree_hal_allocator_create_heap(
        IREE_SV("task_queue_test"), iree_allocator_system(),
        iree_allocator_system(), &device_allocator_));
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        /*node_count=*/1, /*node_ids=*/nullptr,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool_));
  }

  void TearDown() override {
    iree_async_proactor_pool_release(proactor_pool_);
    iree_hal_allocator_release(device_allocator_);
    iree_task_executor_release(executor_);
  }

  iree_status_t CreateDeviceGroup(iree_hal_device_group_t** out_device_group) {
    *out_device_group = nullptr;
    iree_hal_task_device_params_t device_params;
    iree_hal_task_device_params_initialize(&device_params);
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool_;
    iree_hal_device_t* device = nullptr;
    iree_status_t status = iree_hal_task_device_create(
        IREE_SV("task_queue_test"), &device_params, /*queue_count=*/1,
        &executor_, /*loader_count=*/0, /*loaders=*/nullptr, device_allocator_,
        &create_params, iree_allocator_system(), &device);
    iree_async_frontier_tracker_t* frontier_tracker = nullptr;
    if (iree_status_is_ok(status)) {
      status = iree_async_frontier_tracker_create(
          iree_async_frontier_tracker_options_default(),
          iree_allocator_system(), &frontier_tracker);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_create_from_device(
          device, frontier_tracker, iree_allocator_system(), out_device_group);
    }
    iree_async_frontier_tracker_release(frontier_tracker);
    iree_hal_device_release(device);
    return status;
  }

  // Executor shared by each short-lived device in the test.
  iree_task_executor_t* executor_ = nullptr;

  // Heap allocator shared by each short-lived device in the test.
  iree_hal_allocator_t* device_allocator_ = nullptr;

  // Proactor pool shared by each short-lived device in the test.
  iree_async_proactor_pool_t* proactor_pool_ = nullptr;
};

TEST_P(TaskQueueTest, ExecutesBindingsWithDeclaredPermissions) {
  iree_hal_device_group_t* device_group = nullptr;
  IREE_ASSERT_OK(CreateDeviceGroup(&device_group));
  iree_hal_device_t* device = iree_hal_device_group_device_at(device_group, 0);
  iree_hal_queue_t* queue = iree_hal_device_queue(device, 0, 0);
  alignas(64) std::array<uint8_t, 18> source = {};
  alignas(64) std::array<uint8_t, 18> target = {};
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<uint8_t>(i);
  }
  target.fill(0xA5);

  iree_hal_buffer_t* source_buffer = nullptr;
  iree_hal_buffer_t* target_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING |
          IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
      16, iree_make_byte_span(source.data() + 1, 16),
      iree_hal_buffer_release_callback_null(), iree_allocator_system(),
      &source_buffer));
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING |
          IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
      16, iree_make_byte_span(target.data() + 1, 16),
      iree_hal_buffer_release_callback_null(), iree_allocator_system(),
      &target_buffer));
  iree_hal_semaphore_t* completion = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &completion));

  const iree_hal_memory_transition_recipe_info_t release_operation = {
      .kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE,
      .executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE,
      .operation = IREE_HAL_MEMORY_TRANSITION_OPERATION_RELEASE_TO_SYSTEM,
      .range_granularity = 64,
  };
  const iree_hal_memory_transition_recipe_t release_recipe = {
      .effects = {IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM},
      .operation_count = 1,
      .operations = &release_operation,
  };

  for (iree_host_size_t binding_count : {0, 2}) {
    SCOPED_TRACE(binding_count);
    target.fill(0xA5);
    iree_hal_command_buffer_t* command_buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        iree_hal_queue_family(queue), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER, binding_count, &command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
        command_buffer,
        binding_count ? iree_hal_make_indirect_buffer_ref(0, 0, 16)
                      : iree_hal_make_buffer_ref(source_buffer, 0, 16),
        binding_count ? iree_hal_make_indirect_buffer_ref(1, 0, 16)
                      : iree_hal_make_buffer_ref(target_buffer, 0, 16),
        IREE_HAL_COPY_FLAG_NONE));
    const iree_hal_buffer_ref_t barrier_ref =
        binding_count ? iree_hal_make_indirect_buffer_ref(1, 0, 16)
                      : iree_hal_make_buffer_ref(target_buffer, 0, 16);
    const iree_hal_buffer_barrier_t buffer_barrier = {
        .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
        .target_scope = 0,
        .buffer_ref = barrier_ref,
        .recipe = &release_recipe,
    };
    const iree_hal_barrier_t barrier = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        .flags = IREE_HAL_BARRIER_FLAG_NONE,
        .effects = release_recipe.effects,
        .memory_barrier_count = 0,
        .memory_barriers = nullptr,
        .buffer_barrier_count = 1,
        .buffer_barriers = &buffer_barrier,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(command_buffer, &barrier));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
    const iree_hal_buffer_binding_t binding_values[] = {
        {source_buffer, 0, 16},
        {target_buffer, 0, 16},
    };
    const iree_hal_buffer_binding_table_t bindings = {
        binding_count,
        binding_values,
    };
    uint64_t completion_value = binding_count + 1;
    const iree_hal_semaphore_list_t signal_list = {
        1,
        &completion,
        &completion_value,
    };
    IREE_ASSERT_OK(iree_hal_queue_execute(
        queue, iree_hal_semaphore_list_empty(), signal_list, command_buffer,
        bindings, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    EXPECT_EQ(0xA5, target.front());
    EXPECT_EQ(0xA5, target.back());
    for (size_t i = 1; i <= 16; ++i) {
      EXPECT_EQ(source[i], target[i]);
    }
    iree_hal_command_buffer_release(command_buffer);
  }
  iree_hal_semaphore_release(completion);
  iree_hal_buffer_release(target_buffer);
  iree_hal_buffer_release(source_buffer);
  iree_hal_device_group_release(device_group);
}

TEST_P(TaskQueueTest, TransfersReleaseBuffersBeforeTerminalSignal) {
  for (bool fail_dependency : {false, true}) {
    for (size_t length : {16u, 512u * 1024u}) {
      SCOPED_TRACE(fail_dependency);
      SCOPED_TRACE(length);
      iree_hal_device_group_t* device_group = nullptr;
      IREE_ASSERT_OK(CreateDeviceGroup(&device_group));
      auto* device = iree_hal_device_group_device_at(device_group, 0);
      auto* queue = iree_hal_device_queue(device, 0, 0);
      iree_hal_semaphore_t* ready = nullptr;
      iree_hal_semaphore_t* completion = nullptr;
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &ready));
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &completion));
      struct ReleaseState {
        // Borrowed signal queried while the final buffer reference retires.
        iree_hal_semaphore_t* completion;
        // Number of completed release callbacks.
        std::atomic<uint32_t> count{0};
      } release_state = {completion};
      iree_hal_buffer_release_callback_t release_callback = {
          [](void* user_data, iree_hal_buffer_t* buffer) {
            auto* state = static_cast<ReleaseState*>(user_data);
            uint64_t value = UINT64_MAX;
            IREE_EXPECT_OK(iree_hal_semaphore_query(state->completion, &value));
            EXPECT_EQ(value, 0u);
            state->count.fetch_add(1, std::memory_order_release);
          },
          &release_state,
      };
      std::vector<uint8_t> source(length, 0xA5);
      std::vector<uint8_t> target(length, 0x00);
      iree_hal_buffer_t* buffer = nullptr;
      IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
          iree_hal_buffer_placement_undefined(),
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
          IREE_HAL_MEMORY_ACCESS_WRITE, IREE_HAL_BUFFER_USAGE_TRANSFER, length,
          iree_make_byte_span(target.data(), target.size()), release_callback,
          iree_allocator_system(), &buffer));
      uint64_t value = 1;
      const iree_hal_semaphore_list_t waits = {1, &ready, &value};
      const iree_hal_semaphore_list_t signals = {1, &completion, &value};
      {
        iree_hal_buffer_barrier_t range = {
            .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
            .target_scope = IREE_HAL_ACCESS_SCOPE_HOST_READ,
            .buffer_ref = iree_hal_make_buffer_ref(buffer, 0, length),
        };
        iree_hal_barrier_t after = {
            .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
            .target_stage_mask = IREE_HAL_EXECUTION_STAGE_HOST,
            .buffer_barrier_count = 1,
            .buffer_barriers = &range,
        };
        const iree_hal_barrier_list_t list = {1, &after};
        const iree_hal_queue_barriers_t barriers = {nullptr, &list};
        IREE_ASSERT_OK(iree_hal_queue_upload(queue, waits, signals,
                                             source.data(), buffer, 0, length,
                                             &barriers));
      }
      // Descriptor storage has expired while the wait is unresolved. The
      // backend owns captured actions, not references to the caller's arrays.
      // The accepted operation owns the final buffer reference. Its explicit
      // dependency keeps that ownership alive until the caller drops its copy.
      iree_hal_buffer_release(buffer);
      if (fail_dependency) {
        iree_hal_semaphore_fail(ready,
                                iree_status_from_code(IREE_STATUS_CANCELLED));
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_CANCELLED,
            iree_hal_semaphore_list_wait(signals, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
        EXPECT_EQ(target, std::vector<uint8_t>(length, 0x00));
      } else {
        IREE_ASSERT_OK(iree_hal_semaphore_signal(ready, value, nullptr));
        IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
            signals, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
        EXPECT_EQ(target, source);
      }
      EXPECT_EQ(release_state.count.load(std::memory_order_acquire), 1u);
      iree_hal_semaphore_release(completion);
      iree_hal_semaphore_release(ready);
      iree_hal_device_group_release(device_group);
    }
  }
}

TEST_P(TaskQueueTest, ReleasesDeviceGroupWithAcceptedExecuteInFlight) {
  // Repeated immediate destruction drives shutdown across the control-to-
  // compute ownership handoff without externally waiting for completion.
  // Device group release must either finish or cancel every accepted
  // submission before it tears down the queue's compute item pool.
  static constexpr int kIterationCount = 256;
  for (int iteration = 0; iteration < kIterationCount; ++iteration) {
    iree_hal_device_group_t* device_group = nullptr;
    IREE_ASSERT_OK(CreateDeviceGroup(&device_group));
    iree_hal_device_t* device =
        iree_hal_device_group_device_at(device_group, 0);
    iree_hal_queue_t* queue = iree_hal_device_queue(
        device, /*family_ordinal=*/0, /*queue_ordinal=*/0);
    ASSERT_NE(queue, nullptr);

    iree_hal_command_buffer_t* command_buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        iree_hal_queue_family(queue), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER,
        /*binding_capacity=*/0, &command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

    iree_hal_semaphore_t* signal_semaphore = nullptr;
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
    iree_hal_semaphore_t* signal_semaphores[] = {signal_semaphore};
    uint64_t signal_values[] = {1};
    const iree_hal_semaphore_list_t signal_list = {
        IREE_ARRAYSIZE(signal_semaphores),
        signal_semaphores,
        signal_values,
    };

    IREE_ASSERT_OK(iree_hal_queue_execute(
        queue, iree_hal_semaphore_list_empty(), signal_list, command_buffer,
        iree_hal_buffer_binding_table_empty(),
        IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));

    iree_hal_semaphore_release(signal_semaphore);
    iree_hal_command_buffer_release(command_buffer);
    iree_hal_device_group_release(device_group);
  }
}

TEST_P(TaskQueueTest, ReusedDynamicQueueSlotAdvancesIncarnation) {
  iree_hal_device_group_t* device_group = nullptr;
  IREE_ASSERT_OK(CreateDeviceGroup(&device_group));
  iree_hal_device_t* device = iree_hal_device_group_device_at(device_group, 0);
  const iree_hal_queue_family_t* queue_family =
      iree_hal_device_queue_family(device, /*family_ordinal=*/0);
  ASSERT_NE(nullptr, queue_family);

  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  iree_hal_queue_t* queue_a = nullptr;
  IREE_ASSERT_OK(iree_hal_queue_acquire(queue_family, &params, &queue_a));
  iree_hal_queue_t* queue_b = nullptr;
  IREE_ASSERT_OK(iree_hal_queue_acquire(queue_family, &params, &queue_b));
  const iree_async_axis_t axis_a = ((iree_hal_task_queue_t*)queue_a)->axis;
  const iree_async_axis_t axis_b = ((iree_hal_task_queue_t*)queue_b)->axis;
  EXPECT_NE(iree_async_axis_queue_index(axis_a),
            iree_async_axis_queue_index(axis_b));
  EXPECT_NE(axis_a, axis_b);

  iree_hal_queue_release(queue_a);
  queue_a = nullptr;

  iree_hal_queue_t* queue_c = nullptr;
  IREE_ASSERT_OK(iree_hal_queue_acquire(queue_family, &params, &queue_c));
  const iree_async_axis_t axis_c = ((iree_hal_task_queue_t*)queue_c)->axis;
  EXPECT_EQ(iree_async_axis_queue_index(axis_a),
            iree_async_axis_queue_index(axis_c));
  EXPECT_EQ(iree_async_axis_queue_incarnation(axis_a) + 1,
            iree_async_axis_queue_incarnation(axis_c));
  EXPECT_NE(axis_a, axis_c);

  iree_hal_queue_release(queue_c);
  iree_hal_queue_release(queue_b);
  iree_hal_device_group_release(device_group);
}

TEST_P(TaskQueueTest, ConcurrentQueueAcquisitionWithQueueProgress) {
  constexpr int kThreadCount = 4;
  constexpr int kQueuesPerThread = 16;
  iree_hal_device_group_t* device_group = nullptr;
  IREE_ASSERT_OK(CreateDeviceGroup(&device_group));
  iree_hal_device_t* device = iree_hal_device_group_device_at(device_group, 0);
  iree_hal_queue_t* provisioned_queue =
      iree_hal_device_queue(device, /*family_ordinal=*/0, /*queue_ordinal=*/0);
  const iree_hal_queue_family_t* queue_family =
      iree_hal_queue_family(provisioned_queue);
  iree_async_frontier_tracker_t* frontier_tracker =
      ((iree_hal_task_queue_t*)provisioned_queue)->frontier_tracker;

  std::atomic<bool> start{false};
  std::atomic<int> remaining{kThreadCount};
  std::array<iree_async_axis_t, kThreadCount * kQueuesPerThread> axes = {};
  std::vector<std::thread> threads;
  for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([&, thread_index]() {
      iree_hal_semaphore_t* completion = nullptr;
      iree_status_t status = iree_hal_semaphore_create(
          device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &completion);
      iree_hal_queue_params_t params;
      iree_hal_queue_params_initialize(&params);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int i = 0; i < kQueuesPerThread && iree_status_is_ok(status); ++i) {
        iree_hal_queue_t* queue = nullptr;
        status = iree_hal_queue_acquire(queue_family, &params, &queue);
        uint64_t completion_value = i + 1;
        const iree_hal_semaphore_list_t signal_list = {1, &completion,
                                                       &completion_value};
        if (iree_status_is_ok(status)) {
          axes[thread_index * kQueuesPerThread + i] =
              ((iree_hal_task_queue_t*)queue)->axis;
          status = iree_hal_queue_barrier(
              queue, iree_hal_semaphore_list_empty(), signal_list,
              /*barriers=*/NULL, IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
        }
        if (iree_status_is_ok(status)) {
          status = iree_hal_semaphore_list_wait(
              signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
        }
        iree_hal_queue_release(queue);
      }
      iree_hal_semaphore_release(completion);
      IREE_EXPECT_OK(status);
      remaining.fetch_sub(1, std::memory_order_release);
    });
  }

  // Keep an existing queue completing work while other threads acquire, use
  // and release queues in this same sealed device group.
  iree_hal_semaphore_t* completion = nullptr;
  iree_status_t status = iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &completion);
  start.store(true, std::memory_order_release);
  uint64_t completion_value = 0;
  do {
    const iree_hal_semaphore_list_t signal_list = {1, &completion,
                                                   &completion_value};
    ++completion_value;
    if (iree_status_is_ok(status)) {
      status = iree_hal_queue_barrier(
          provisioned_queue, iree_hal_semaphore_list_empty(), signal_list,
          /*barriers=*/NULL, IREE_HAL_QUEUE_BARRIER_FLAG_NONE);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_list_wait(
          signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
    }
  } while (iree_status_is_ok(status) &&
           remaining.load(std::memory_order_acquire) != 0);

  for (auto& thread : threads) {
    thread.join();
  }
  IREE_EXPECT_OK(status);
  for (iree_async_axis_t axis : axes) {
    EXPECT_TRUE(
        iree_async_frontier_tracker_query_epoch(frontier_tracker, axis, 1));
  }
  std::sort(axes.begin(), axes.end());
  EXPECT_EQ(std::adjacent_find(axes.begin(), axes.end()), axes.end());
  iree_hal_semaphore_release(completion);
  iree_hal_device_group_release(device_group);
}

INSTANTIATE_TEST_SUITE_P(WorkerCounts, TaskQueueTest, ::testing::Values(1, 4));

}  // namespace
