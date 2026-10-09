// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <future>
#include <utility>

#include "iree/hal/cts/util/atomic_test_util.h"
#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using iree::testing::status::StatusIs;

static void NotifyForeignBufferReleased(void* user_data,
                                        iree_hal_buffer_t* /*buffer*/) {
  static_cast<std::promise<void>*>(user_data)->set_value();
}

class VulkanAtomicSubmissionTest : public CtsTestBase<> {};

TEST_P(VulkanAtomicSubmissionTest,
       MissingTargetDeviceAddressFollowsModeAndReleasesSubmissions) {
  const AtomicTestRequirements requirements = {
      .operation_flags = IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
                         IREE_HAL_ATOMIC_OPERATION_FLAG_STORE |
                         IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD,
      .wait_condition_flags = IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_EQUAL,
      .memory_type = IREE_HAL_MEMORY_TYPE_NONE,
      .buffer_usage = IREE_HAL_BUFFER_USAGE_STORAGE,
      .memory_access =
          IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      .width = IREE_HAL_ATOMIC_WIDTH_32,
      .atomic_flags =
          IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE,
  };
  AtomicTestConfiguration configuration;
  if (!SelectAtomicTestConfiguration(iree_hal_device_spec(device_),
                                     requirements, &configuration)) {
    GTEST_SKIP() << "Device does not advertise the tested atomic operation set";
  }
  iree_hal_queue_t* atomic_queue = iree_hal_device_queue(
      device_, configuration.queue_family_ordinal, /*queue_ordinal=*/0);
  ASSERT_NE(atomic_queue, nullptr);

  iree_hal_allocator_t* heap_allocator = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_create_heap(
      IREE_SV("foreign"), iree_allocator_system(), iree_allocator_system(),
      &heap_allocator));

  alignas(IREE_HAL_HEAP_BUFFER_ALIGNMENT) std::array<uint8_t, sizeof(uint32_t)>
      storage = {};
  iree_hal_external_buffer_t external_buffer = {};
  external_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION;
  external_buffer.size = storage.size();
  external_buffer.handle.host_allocation.ptr = storage.data();
  std::promise<void> released;
  std::future<void> released_future = released.get_future();
  const iree_hal_buffer_release_callback_t release_callback = {
      .fn = NotifyForeignBufferReleased,
      .user_data = &released,
  };
  const iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
      .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      .min_alignment = alignof(uint32_t),
  };
  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_import_buffer(
      heap_allocator, buffer_params, &external_buffer, release_callback,
      target_buffer.out()));

  enum class AtomicKind { kWait, kStore, kRmw };
  const AtomicKind kinds[] = {AtomicKind::kWait, AtomicKind::kStore,
                              AtomicKind::kRmw};
  const iree_hal_atomic_target_error_mode_t modes[] = {
      IREE_HAL_ATOMIC_TARGET_ERROR_MODE_DEFAULT,
      IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE,
  };
  const iree_hal_buffer_binding_t binding = {
      .buffer = target_buffer.get(),
      .offset = 0,
      .length = IREE_HAL_WHOLE_BUFFER,
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      .count = 1,
      .bindings = &binding,
  };
  for (iree_hal_atomic_target_error_mode_t mode : modes) {
    const StatusCode expected_status =
        mode == IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE
            ? StatusCode::kIncompatible
            : StatusCode::kFailedPrecondition;
    for (AtomicKind kind : kinds) {
      SemaphoreList signal(device_, {0}, {1});
      iree_status_t status = iree_ok_status();
      switch (kind) {
        case AtomicKind::kWait:
          status = iree_hal_queue_atomic_wait(
              atomic_queue, iree_hal_semaphore_list_empty(), signal,
              target_buffer, /*target_offset=*/0,
              (iree_hal_atomic_wait_params_t){
                  .value = 0,
                  .mask = UINT32_MAX,
                  .flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE,
                  .width = IREE_HAL_ATOMIC_WIDTH_32,
                  .condition = IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
                  .target_error_mode = mode,
              },
              /*barriers=*/NULL);
          break;
        case AtomicKind::kStore:
          status = iree_hal_queue_atomic_store(
              atomic_queue, iree_hal_semaphore_list_empty(), signal,
              target_buffer, /*target_offset=*/0,
              (iree_hal_atomic_store_params_t){
                  .value = 1,
                  .flags = IREE_HAL_ATOMIC_FLAG_RELEASE,
                  .width = IREE_HAL_ATOMIC_WIDTH_32,
                  .target_error_mode = mode,
              },
              /*barriers=*/NULL);
          break;
        case AtomicKind::kRmw:
          status = iree_hal_queue_atomic_rmw(
              atomic_queue, iree_hal_semaphore_list_empty(), signal,
              target_buffer, /*target_offset=*/0,
              (iree_hal_atomic_rmw_params_t){
                  .operand = 1,
                  .flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE |
                           IREE_HAL_ATOMIC_FLAG_RELEASE,
                  .width = IREE_HAL_ATOMIC_WIDTH_32,
                  .operation = IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
                  .target_error_mode = mode,
              },
              /*barriers=*/NULL);
          break;
      }
      EXPECT_THAT(Status(std::move(status)), StatusIs(expected_status));
      EXPECT_THAT(
          Status(iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE)),
          StatusIs(expected_status));

      Ref<iree_hal_command_buffer_t> command_buffer;
      IREE_ASSERT_OK(CreateCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
                                         IREE_HAL_COMMAND_CATEGORY_ATOMIC,
                                         /*binding_capacity=*/1,
                                         command_buffer.out()));
      IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
      const iree_hal_buffer_ref_t target_ref =
          iree_hal_make_indirect_buffer_ref(
              /*binding=*/0, /*offset=*/0, /*length=*/sizeof(uint32_t));
      switch (kind) {
        case AtomicKind::kWait:
          IREE_ASSERT_OK(iree_hal_command_buffer_atomic_wait(
              command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
              IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE, target_ref,
              (iree_hal_atomic_wait_params_t){
                  .value = 0,
                  .mask = UINT32_MAX,
                  .flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE,
                  .width = IREE_HAL_ATOMIC_WIDTH_32,
                  .condition = IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
                  .target_error_mode = mode,
              }));
          break;
        case AtomicKind::kStore:
          IREE_ASSERT_OK(iree_hal_command_buffer_atomic_store(
              command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
              IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE, target_ref,
              (iree_hal_atomic_store_params_t){
                  .value = 1,
                  .flags = IREE_HAL_ATOMIC_FLAG_RELEASE,
                  .width = IREE_HAL_ATOMIC_WIDTH_32,
                  .target_error_mode = mode,
              }));
          break;
        case AtomicKind::kRmw:
          IREE_ASSERT_OK(iree_hal_command_buffer_atomic_rmw(
              command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
              IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE, target_ref,
              (iree_hal_atomic_rmw_params_t){
                  .operand = 1,
                  .flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE |
                           IREE_HAL_ATOMIC_FLAG_RELEASE,
                  .width = IREE_HAL_ATOMIC_WIDTH_32,
                  .operation = IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
                  .target_error_mode = mode,
              }));
          break;
      }
      IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
      SemaphoreList execute_signal(device_, {0}, {1});
      iree_hal_queue_t* queue = QueueForCommandBuffer(command_buffer);
      ASSERT_NE(nullptr, queue);
      EXPECT_THAT(
          Status(iree_hal_queue_execute(
              queue, iree_hal_semaphore_list_empty(), execute_signal,
              command_buffer, binding_table, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE)),
          StatusIs(expected_status));
      EXPECT_THAT(Status(iree_hal_semaphore_list_wait(
                      execute_signal, iree_infinite_timeout(),
                      IREE_ASYNC_WAIT_FLAG_NONE)),
                  StatusIs(expected_status));
      EXPECT_TRUE(std::all_of(storage.begin(), storage.end(),
                              [](uint8_t byte) { return byte == 0; }));
    }
  }

  target_buffer.reset();
  released_future.wait();
  iree_hal_allocator_release(heap_allocator);
}

CTS_REGISTER_TEST_SUITE(VulkanAtomicSubmissionTest);

}  // namespace iree::hal::cts
