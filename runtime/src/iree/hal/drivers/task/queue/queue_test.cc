// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/device_group.h"
#include "iree/hal/drivers/task/device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TaskQueueShutdownTest
    : public ::testing::TestWithParam<iree_host_size_t> {
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

TEST_P(TaskQueueShutdownTest, ReleasesDeviceGroupWithAcceptedExecuteInFlight) {
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

    iree_hal_command_buffer_t* command_buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/0, &command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

    iree_hal_semaphore_t* signal_semaphore = nullptr;
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
    iree_hal_semaphore_t* signal_semaphores[] = {signal_semaphore};
    uint64_t signal_values[] = {1};
    const iree_hal_semaphore_list_t signal_list = {
        IREE_ARRAYSIZE(signal_semaphores),
        signal_semaphores,
        signal_values,
    };

    IREE_ASSERT_OK(iree_hal_device_queue_execute(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        signal_list, command_buffer, iree_hal_buffer_binding_table_empty(),
        IREE_HAL_EXECUTE_FLAG_NONE));

    iree_hal_semaphore_release(signal_semaphore);
    iree_hal_command_buffer_release(command_buffer);
    iree_hal_device_group_release(device_group);
  }
}

INSTANTIATE_TEST_SUITE_P(WorkerCounts, TaskQueueShutdownTest,
                         ::testing::Values(1, 4));

}  // namespace
