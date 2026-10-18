// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/task/device.h"

#include <atomic>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/task/queue/queue.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct DeviceAllocationLifetime {
  // Device-owned allocations not yet returned to the underlying allocator.
  std::atomic<int64_t> outstanding{0};

  iree_allocator_t allocator() { return {this, Control}; }

  static iree_status_t Control(void* user_data,
                               iree_allocator_command_t command,
                               const void* params, void** pointer) {
    auto* lifetime = static_cast<DeviceAllocationLifetime*>(user_data);
    const bool had_pointer = (command == IREE_ALLOCATOR_COMMAND_FREE ||
                              command == IREE_ALLOCATOR_COMMAND_REALLOC) &&
                             *pointer != nullptr;
    const auto allocator = iree_allocator_system();
    iree_status_t status =
        allocator.ctl(allocator.self, command, params, pointer);
    if (iree_status_is_ok(status)) {
      if (command == IREE_ALLOCATOR_COMMAND_FREE) {
        if (had_pointer) {
          --lifetime->outstanding;
        }
      } else if (command != IREE_ALLOCATOR_COMMAND_REALLOC || !had_pointer) {
        ++lifetime->outstanding;
      }
    }
    return status;
  }
};

class TaskDeviceLocalityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    node_id_ = iree_numa_node_for_current_thread();
    ASSERT_NE(node_id_, IREE_NUMA_NODE_ANY);
    iree_thread_affinity_t affinity;
    iree_thread_affinity_set_group_any(node_id_, &affinity);
    iree_task_topology_t topology;
    IREE_ASSERT_OK(iree_task_topology_initialize_from_thread_affinities(
        1, &affinity, &topology));
    if (iree_task_topology_query_numa_node(&topology) == IREE_NUMA_NODE_ANY) {
      iree_task_topology_deinitialize(&topology);
      GTEST_SKIP() << "The task topology cannot resolve a physical NUMA node";
    }
    iree_task_executor_options_t options;
    iree_task_executor_options_initialize(&options);
    iree_status_t status = iree_task_executor_create(
        options, &topology, iree_allocator_system(), &executor_);
    iree_task_topology_deinitialize(&topology);
    IREE_ASSERT_OK(status);
    ASSERT_EQ(iree_task_executor_numa_node(executor_), node_id_);
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(iree_allocator_system(),
                                                  &device_allocator_));
  }

  void TearDown() override {
    iree_hal_device_release(device_);
    iree_async_proactor_pool_release(pool_);
    iree_hal_allocator_release(device_allocator_);
    iree_task_executor_release(executor_);
    iree_task_executor_release(unplaced_executor_);
  }

  iree_status_t CreateDevice(
      iree_host_size_t count, const uint32_t* nodes,
      iree_host_size_t queue_count = 1,
      iree_task_executor_t* const* queue_executors = nullptr,
      iree_allocator_t host_allocator = iree_allocator_system()) {
    IREE_RETURN_IF_ERROR(iree_async_proactor_pool_create(
        count, nodes, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &pool_));
    iree_hal_task_device_params_t params;
    iree_hal_task_device_params_initialize(&params);
    auto create_params = iree_hal_device_create_params_default();
    create_params.proactor_pool = pool_;
    return iree_hal_task_device_create(
        IREE_SV("device_test"), &params, queue_count,
        queue_executors ? queue_executors : &executor_,
        /*loader_count=*/0, /*loaders=*/nullptr, device_allocator_,
        &create_params, host_allocator, &device_);
  }

  iree_status_t CreateUnplacedExecutor() {
    iree_task_topology_t topology;
    iree_task_topology_initialize_from_group_count(1, &topology);
    iree_task_executor_options_t options;
    iree_task_executor_options_initialize(&options);
    iree_status_t status = iree_task_executor_create(
        options, &topology, iree_allocator_system(), &unplaced_executor_);
    iree_task_topology_deinitialize(&topology);
    return status;
  }

  void CheckNumaIdentity(bool expected) {
    const auto* identity =
        iree_hal_device_spec_identity(iree_hal_device_spec(device_));
    ASSERT_EQ(identity->physical_device_count, 1);
    const auto& physical_identity = identity->physical_devices[0].identity;
    EXPECT_EQ(
        iree_any_bit_set(physical_identity.flags,
                         IREE_HAL_PHYSICAL_DEVICE_IDENTITY_FLAG_NUMA_NODE),
        expected);
    if (expected) {
      EXPECT_EQ(physical_identity.numa.node_id, node_id_);
    }

    iree_async_frontier_tracker_t* tracker = nullptr;
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker));
    iree_hal_device_group_builder_t builder;
    iree_hal_device_group_builder_initialize(&builder, tracker);
    iree_async_frontier_tracker_release(tracker);
    iree_status_t status =
        iree_hal_device_group_builder_add_device(&builder, device_);
    iree_hal_device_group_t* group = nullptr;
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_builder_finalize(
          &builder, iree_allocator_system(), &group);
    }
    iree_hal_device_group_builder_deinitialize(&builder);
    IREE_ASSERT_OK(status);
    const auto* topology = iree_hal_device_group_topology(group);
    uint32_t host_node_count = 0;
    for (uint32_t i = 0; i < topology->node_count; ++i) {
      const auto* node = iree_hal_topology_node_at(topology, i);
      if (node->kind == IREE_HAL_TOPOLOGY_NODE_KIND_HOST_NUMA) {
        ++host_node_count;
        EXPECT_EQ(node->local_ordinal, node_id_);
      }
    }
    EXPECT_EQ(host_node_count, expected ? 1u : 0u);
    iree_hal_device_group_release(group);
  }

  // Physical node used to constrain the real executor's worker.
  uint32_t node_id_ = IREE_NUMA_NODE_ANY;
  // Single-node executor retained by the test and created device.
  iree_task_executor_t* executor_ = nullptr;
  // Unconstrained executor used to exercise devices without exact affinity.
  iree_task_executor_t* unplaced_executor_ = nullptr;
  // Heap allocator retained by the test and created device.
  iree_hal_allocator_t* device_allocator_ = nullptr;
  // Progress services available to device construction.
  iree_async_proactor_pool_t* pool_ = nullptr;
  // Device under test, released before its construction inputs.
  iree_hal_device_t* device_ = nullptr;
  // Outlives teardown, including failure exits from allocator lifetime tests.
  DeviceAllocationLifetime allocation_lifetime_;
};

TEST_F(TaskDeviceLocalityTest, DestructionJoinsNativeAllocationOwnership) {
  const uint32_t nodes[] = {UINT32_MAX};
  IREE_ASSERT_OK(CreateDevice(IREE_ARRAYSIZE(nodes), nodes, 1, nullptr,
                              allocation_lifetime_.allocator()));
  iree_async_frontier_tracker_t* tracker = nullptr;
  IREE_ASSERT_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &tracker));
  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder, tracker);
  iree_async_frontier_tracker_release(tracker);
  IREE_ASSERT_OK(iree_hal_device_group_builder_add_device(&builder, device_));
  iree_hal_device_group_t* group = nullptr;
  IREE_ASSERT_OK(iree_hal_device_group_builder_finalize(
      &builder, iree_allocator_system(), &group));
  iree_hal_device_group_builder_deinitialize(&builder);
  const iree_hal_pool_family_access_t family = {
      .family = iree_hal_device_queue_family(device_, 0),
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(group, {1, &family}, &options,
                                           iree_allocator_system(), &pool));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(4096, 64),
      iree_infinite_timeout(), &buffer));
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
  iree_hal_device_group_release(group);
  // Native free may still be running. Device destruction joins that work even
  // though the executor and its workers remain alive in the fixture.
  iree_hal_device_release(device_);
  device_ = nullptr;
  EXPECT_EQ(allocation_lifetime_.outstanding.load(), 0);
}

TEST_F(TaskDeviceLocalityTest, AcceptsExplicitUnplacedService) {
  const uint32_t nodes[] = {UINT32_MAX};
  IREE_ASSERT_OK(CreateDevice(IREE_ARRAYSIZE(nodes), nodes));
  ASSERT_NE(device_, nullptr);
  auto* queue = reinterpret_cast<iree_hal_task_queue_t*>(
      iree_hal_device_queue(device_, 0, 0));
  ASSERT_NE(queue, nullptr);
  iree_async_proactor_t* service = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_get(pool_, 0, &service));
  EXPECT_EQ(queue->proactor, service);
  CheckNumaIdentity(true);
}

TEST_F(TaskDeviceLocalityTest, SelectsExactNodeBeforeUnplacedService) {
  const uint32_t nodes[] = {UINT32_MAX, node_id_};
  IREE_ASSERT_OK(CreateDevice(IREE_ARRAYSIZE(nodes), nodes));
  ASSERT_NE(device_, nullptr);
  auto* queue = reinterpret_cast<iree_hal_task_queue_t*>(
      iree_hal_device_queue(device_, 0, 0));
  ASSERT_NE(queue, nullptr);
  iree_async_proactor_t* service = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_pool_get(pool_, 1, &service));
  EXPECT_EQ(queue->proactor, service);
  CheckNumaIdentity(true);
}

TEST_F(TaskDeviceLocalityTest, CapturesCommonQueueNode) {
  const uint32_t nodes[] = {UINT32_MAX};
  iree_task_executor_t* executors[] = {executor_, executor_};
  IREE_ASSERT_OK(CreateDevice(IREE_ARRAYSIZE(nodes), nodes,
                              IREE_ARRAYSIZE(executors), executors));
  CheckNumaIdentity(true);
}

TEST_F(TaskDeviceLocalityTest, UnplacedExecutorHasNoExactNode) {
  IREE_ASSERT_OK(CreateUnplacedExecutor());
  ASSERT_EQ(iree_task_executor_numa_node(unplaced_executor_),
            IREE_NUMA_NODE_ANY);
  const uint32_t nodes[] = {UINT32_MAX};
  IREE_ASSERT_OK(
      CreateDevice(IREE_ARRAYSIZE(nodes), nodes, 1, &unplaced_executor_));
  CheckNumaIdentity(false);
}

TEST_F(TaskDeviceLocalityTest, MixedQueueAffinityHasNoExactNode) {
  IREE_ASSERT_OK(CreateUnplacedExecutor());
  const uint32_t nodes[] = {UINT32_MAX};
  iree_task_executor_t* executors[] = {executor_, unplaced_executor_};
  IREE_ASSERT_OK(CreateDevice(IREE_ARRAYSIZE(nodes), nodes,
                              IREE_ARRAYSIZE(executors), executors));
  CheckNumaIdentity(false);
}

TEST_F(TaskDeviceLocalityTest, UnplacedDefaultQueueHasNoExactNode) {
  IREE_ASSERT_OK(CreateUnplacedExecutor());
  const uint32_t nodes[] = {UINT32_MAX};
  iree_task_executor_t* executors[] = {unplaced_executor_, executor_};
  IREE_ASSERT_OK(CreateDevice(IREE_ARRAYSIZE(nodes), nodes,
                              IREE_ARRAYSIZE(executors), executors));
  CheckNumaIdentity(false);
}

TEST_F(TaskDeviceLocalityTest, MissingBoundServiceFailsConstruction) {
  const uint32_t nodes[] = {UINT32_MAX - 1};
  // Reject the absent entry before attempting native affinity for this invalid
  // node. This must report selection failure, not failure to start a runner.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        CreateDevice(IREE_ARRAYSIZE(nodes), nodes));
  EXPECT_EQ(device_, nullptr);
}
}  // namespace
