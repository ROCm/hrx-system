// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/task/device.h"
#include "iree/hal/slab_pool.h"
#include "iree/module/hal/types.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class HALModuleOwnershipTest : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    iree_task_topology_t topology;
    iree_task_topology_initialize_from_group_count(1, &topology);
    iree_task_executor_options_t options;
    iree_task_executor_options_initialize(&options);
    iree_status_t status = iree_task_executor_create(
        options, &topology, iree_allocator_system(), &executor_);
    iree_task_topology_deinitialize(&topology);
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(
        iree_hal_allocator_create_heap(iree_allocator_system(), &allocator_));
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        1, nullptr, iree_async_proactor_pool_options_default(),
        iree_allocator_system(), &progress_));
    iree_hal_task_device_params_t params;
    iree_hal_task_device_params_initialize(&params);
    auto create_params = iree_hal_device_create_params_default();
    create_params.proactor_pool = progress_;
    iree_hal_device_t* device = nullptr;
    IREE_ASSERT_OK(iree_hal_task_device_create(
        IREE_SV("ownership"), &params, 1, &executor_, 0, nullptr, allocator_,
        &create_params, iree_allocator_system(), &device));
    iree_async_frontier_tracker_t* tracker = nullptr;
    status = iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker);
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_create_from_device(
          device, tracker, {this, CountGroupFree}, &group_);
    }
    iree_async_frontier_tracker_release(tracker);
    iree_hal_device_release(device);
    IREE_ASSERT_OK(status);
    group_address_ = group_;
  }

  void TearDown() override {
    iree_vm_ref_reset(&ref_);
    iree_hal_device_group_release(group_);
    iree_async_proactor_pool_release(progress_);
    iree_hal_allocator_release(allocator_);
    iree_task_executor_release(executor_);
    EXPECT_EQ(group_frees_, 1);
  }

  static iree_status_t CountGroupFree(void* self,
                                      iree_allocator_command_t command,
                                      const void* parameters, void** pointer) {
    auto* test = static_cast<HALModuleOwnershipTest*>(self);
    if (command == IREE_ALLOCATOR_COMMAND_FREE &&
        *pointer == test->group_address_) {
      ++test->group_frees_;
    }
    auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  iree_status_t Wrap(int ordinal, iree_allocator_t allocator) {
    auto* device = iree_hal_device_group_device_at(group_, 0);
    auto* queue = iree_hal_device_queue(device, 0, 0);
    switch (ordinal) {
      case IREE_HAL_MODULE_TYPE_DEVICE_GROUP:
        return iree_hal_vm_device_group_wrap(group_, allocator, &ref_);
      case IREE_HAL_MODULE_TYPE_QUEUE:
        return iree_hal_vm_queue_wrap(group_, queue, allocator, &ref_);
      case IREE_HAL_MODULE_TYPE_COMMAND_BUFFER: {
        iree_hal_command_buffer_t* command_buffer = nullptr;
        IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
            iree_hal_queue_family(queue), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
            IREE_HAL_COMMAND_CATEGORY_TRANSFER, 0, &command_buffer));
        iree_status_t status = iree_hal_vm_command_buffer_wrap(
            group_, command_buffer, allocator, &ref_);
        iree_hal_command_buffer_release(command_buffer);
        return status;
      }
      case IREE_HAL_MODULE_TYPE_SEMAPHORE: {
        iree_hal_semaphore_t* semaphore = nullptr;
        IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
            device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
            IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
        iree_status_t status =
            iree_hal_vm_semaphore_wrap(group_, semaphore, allocator, &ref_);
        iree_hal_semaphore_release(semaphore);
        return status;
      }
      case IREE_HAL_MODULE_TYPE_POOL: {
        const iree_hal_pool_family_access_t family = {
            .family = iree_hal_queue_family(queue),
            .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
        };
        const iree_hal_pool_scope_t scope = {
            .family_count = 1,
            .families = &family,
        };
        iree_hal_slab_pool_options_t options;
        iree_hal_slab_pool_options_initialize(&options);
        iree_hal_pool_t* pool = nullptr;
        IREE_RETURN_IF_ERROR(iree_hal_slab_pool_create(
            group_, scope, &options, iree_allocator_system(), &pool));
        iree_status_t status =
            iree_hal_vm_pool_wrap(group_, pool, allocator, &ref_);
        iree_hal_pool_release(pool);
        return status;
      }
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "not a wrapped type");
    }
  }

  // Native setup owners; the device retains them independently.
  iree_task_executor_t* executor_ = nullptr;
  // Host storage provider retained by the native device.
  iree_hal_allocator_t* allocator_ = nullptr;
  // Explicit owner of the native progress services.
  iree_async_proactor_pool_t* progress_ = nullptr;
  // Embedding owner, released before testing escaped wrapper ownership.
  iree_hal_device_group_t* group_ = nullptr;
  // Allocation identity observed only by the counting allocator.
  void* group_address_ = nullptr;
  // Number of final group storage releases.
  int group_frees_ = 0;
  // The tested wrapper's only owning carrier.
  iree_vm_ref_t ref_ = {};
};

TEST_P(HALModuleOwnershipTest, LastWrapperAnchorsNativeGroupUntilReset) {
  IREE_ASSERT_OK(Wrap(GetParam(), iree_allocator_system()));
  auto* group = group_;
  iree_hal_device_group_release(group_);
  group_ = nullptr;
  EXPECT_EQ(group_frees_, 0);
  EXPECT_EQ(iree_hal_device_group_device_count(group), 1);
  auto* queue =
      iree_hal_device_queue(iree_hal_device_group_device_at(group, 0), 0, 0);
  EXPECT_NE(iree_hal_queue_family(queue), nullptr);
  iree_vm_ref_t escaped = iree_vm_ref_retain(ref_);
  iree_vm_ref_reset(&ref_);
  EXPECT_EQ(group_frees_, 0);
  iree_vm_ref_reset(&escaped);
  EXPECT_EQ(group_frees_, 1);
}

TEST_P(HALModuleOwnershipTest, AllocationFailureDoesNotPublishOrRetain) {
  const iree_allocator_t exhausted = {
      nullptr, [](void*, iree_allocator_command_t, const void*, void**) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "injected allocation failure");
      }};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        Wrap(GetParam(), exhausted));
  EXPECT_TRUE(iree_vm_ref_is_null(ref_));
  iree_hal_device_group_release(group_);
  group_ = nullptr;
  EXPECT_EQ(group_frees_, 1);
}

INSTANTIATE_TEST_SUITE_P(Handles, HALModuleOwnershipTest,
                         ::testing::Values(IREE_HAL_MODULE_TYPE_DEVICE_GROUP,
                                           IREE_HAL_MODULE_TYPE_QUEUE,
                                           IREE_HAL_MODULE_TYPE_COMMAND_BUFFER,
                                           IREE_HAL_MODULE_TYPE_SEMAPHORE,
                                           IREE_HAL_MODULE_TYPE_POOL));

}  // namespace
