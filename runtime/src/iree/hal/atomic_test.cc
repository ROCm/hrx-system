// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/atomic.h"

#include "iree/base/internal/atomics.h"
#include "iree/hal/api.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static constexpr iree_hal_queue_priority_t kQueuePriority =
    IREE_HAL_QUEUE_PRIORITY_NORMAL;
static const iree_hal_queue_family_spec_t kQueueFamilySpec = [] {
  iree_hal_queue_family_spec_t spec = {};
  spec.name = IREE_SV("test");
  spec.priority_count = 1;
  spec.priorities = &kQueuePriority;
  spec.physical_device_affinity = 1;
  spec.role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH |
                    IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER |
                    IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_ATOMIC;
  return spec;
}();

static void NoopCommandBufferDestroy(
    iree_hal_command_buffer_t* command_buffer) {}

static iree_status_t NoopCommandBufferBegin(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_ok_status();
}

static iree_status_t NoopCommandBufferEnd(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_ok_status();
}

static iree_status_t NoopCommandBufferAtomicWait(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_wait_params_t params) {
  return iree_ok_status();
}

static iree_status_t NoopCommandBufferAtomicStore(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_store_params_t params) {
  return iree_ok_status();
}

static iree_status_t NoopCommandBufferAtomicRmw(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_rmw_params_t params) {
  return iree_ok_status();
}

static void NoopQueueDestroy(iree_hal_queue_t* queue) {}

static iree_status_t NoopQueueAtomicWait(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_wait_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_ok_status();
}

static iree_status_t NoopQueueAtomicStore(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_store_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_ok_status();
}

static iree_status_t NoopQueueAtomicRmw(
    iree_hal_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_rmw_params_t params,
    const iree_hal_queue_barriers_t* barriers) {
  return iree_ok_status();
}

TEST(AtomicTest, HostCapabilitiesFollowLockFreeWidths) {
  const iree_hal_atomic_operation_flags_t allowed_operations =
      IREE_HAL_ATOMIC_OPERATION_FLAG_STORE |
      IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD;
  const iree_hal_atomic_operation_capabilities_t capabilities =
      iree_hal_atomic_operation_capabilities_for_host(allowed_operations);

  const iree_hal_atomic_operation_flags_t expected_32 =
      iree_atomic_int32_is_lock_free() ? allowed_operations : 0;
  EXPECT_EQ(capabilities.device_scope_32, expected_32);
  EXPECT_EQ(capabilities.system_scope_32, expected_32);

  const iree_hal_atomic_operation_flags_t expected_64 =
      iree_atomic_int64_is_lock_free() ? allowed_operations : 0;
  EXPECT_EQ(capabilities.device_scope_64, expected_64);
  EXPECT_EQ(capabilities.system_scope_64, expected_64);
}

TEST(AtomicTest, ValidatesWaitParameters) {
  iree_hal_atomic_wait_params_t params = {
      /*.value=*/1,
      /*.mask=*/UINT32_MAX,
      /*.flags=*/IREE_HAL_ATOMIC_FLAG_ACQUIRE |
          IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
      /*.condition=*/IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL,
  };
  IREE_EXPECT_OK(iree_hal_atomic_wait_params_validate(params));

  params.flags |= IREE_HAL_ATOMIC_FLAG_RELEASE;
  IREE_EXPECT_OK(iree_hal_atomic_wait_params_validate(params));

  params.flags |= 1u << 31;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_atomic_wait_params_validate(params));
}

TEST(AtomicTest, RejectsNonCanonical32BitValues) {
  iree_hal_atomic_store_params_t params = {
      /*.value=*/UINT64_C(1) << 32,
      /*.flags=*/IREE_HAL_ATOMIC_FLAG_NONE,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_atomic_store_params_validate(params));
}

TEST(AtomicTest, AcceptsInapplicableStoreOrderingFlags) {
  const iree_hal_atomic_store_params_t params = {
      .value = 1,
      .flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE,
      .width = IREE_HAL_ATOMIC_WIDTH_32,
  };
  IREE_EXPECT_OK(iree_hal_atomic_store_params_validate(params));
}

TEST(AtomicTest, ValidatesReadModifyWriteParameters) {
  iree_hal_atomic_rmw_params_t params = {
      .operand = 1,
      .flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE |
               IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
      .width = IREE_HAL_ATOMIC_WIDTH_64,
      .operation = IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
  };
  IREE_EXPECT_OK(iree_hal_atomic_rmw_params_validate(params));

  params.operation = 0xFF;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_atomic_rmw_params_validate(params));
}

TEST(AtomicTest, RejectsUnknownTargetErrorModes) {
  iree_hal_atomic_wait_params_t wait_params = {
      /*.value=*/1,
      /*.mask=*/UINT32_MAX,
      /*.flags=*/IREE_HAL_ATOMIC_FLAG_NONE,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
      /*.condition=*/IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
      /*.target_error_mode=*/2,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_atomic_wait_params_validate(wait_params));

  iree_hal_atomic_store_params_t store_params = {
      .value = 1,
      .flags = IREE_HAL_ATOMIC_FLAG_NONE,
      .width = IREE_HAL_ATOMIC_WIDTH_32,
      .target_error_mode = 2,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_atomic_store_params_validate(store_params));

  iree_hal_atomic_rmw_params_t rmw_params = {
      .operand = 1,
      .flags = IREE_HAL_ATOMIC_FLAG_NONE,
      .width = IREE_HAL_ATOMIC_WIDTH_32,
      .operation = IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
      .target_error_mode = 2,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_atomic_rmw_params_validate(rmw_params));
}

class AtomicTargetValidationTest : public ::testing::Test {
 protected:
  static iree_hal_atomic_wait_params_t WaitParams(
      iree_hal_atomic_target_error_mode_t target_error_mode,
      iree_hal_atomic_width_t width = IREE_HAL_ATOMIC_WIDTH_32) {
    return (iree_hal_atomic_wait_params_t){
        .value = 1,
        .mask = width == IREE_HAL_ATOMIC_WIDTH_32 ? UINT32_MAX : UINT64_MAX,
        .flags = IREE_HAL_ATOMIC_FLAG_NONE,
        .width = width,
        .condition = IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
        .target_error_mode = target_error_mode,
    };
  }

  static iree_hal_atomic_store_params_t StoreParams(
      iree_hal_atomic_target_error_mode_t target_error_mode,
      iree_hal_atomic_width_t width = IREE_HAL_ATOMIC_WIDTH_32) {
    return (iree_hal_atomic_store_params_t){
        .value = 1,
        .flags = IREE_HAL_ATOMIC_FLAG_NONE,
        .width = width,
        .target_error_mode = target_error_mode,
    };
  }

  static iree_hal_atomic_rmw_params_t RmwParams(
      iree_hal_atomic_target_error_mode_t target_error_mode,
      iree_hal_atomic_width_t width = IREE_HAL_ATOMIC_WIDTH_32) {
    return (iree_hal_atomic_rmw_params_t){
        .operand = 1,
        .flags = IREE_HAL_ATOMIC_FLAG_NONE,
        .width = width,
        .operation = IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
        .target_error_mode = target_error_mode,
    };
  }

  void ExpectQueueAlignmentStatus(
      iree_hal_atomic_target_error_mode_t target_error_mode,
      iree_status_code_t expected_status) {
    const iree_hal_semaphore_list_t empty = iree_hal_semaphore_list_empty();
    IREE_EXPECT_STATUS_IS(
        expected_status, iree_hal_queue_atomic_wait(
                             &queue_, empty, empty, unaligned_buffer_, 0,
                             WaitParams(target_error_mode), /*barriers=*/NULL));
    IREE_EXPECT_STATUS_IS(
        expected_status,
        iree_hal_queue_atomic_store(&queue_, empty, empty, unaligned_buffer_, 0,
                                    StoreParams(target_error_mode),
                                    /*barriers=*/NULL));
    IREE_EXPECT_STATUS_IS(expected_status,
                          iree_hal_queue_atomic_rmw(
                              &queue_, empty, empty, unaligned_buffer_, 0,
                              RmwParams(target_error_mode), /*barriers=*/NULL));
  }

  void ExpectDirectReferenceAlignmentStatus(
      iree_hal_atomic_target_error_mode_t target_error_mode,
      iree_status_code_t expected_status) {
    const iree_hal_buffer_ref_t target_ref =
        iree_hal_make_buffer_ref(unaligned_buffer_, /*offset=*/0, /*length=*/4);
    IREE_EXPECT_STATUS_IS(
        expected_status,
        iree_hal_command_buffer_atomic_wait(
            &command_buffer_, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
            IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE, target_ref,
            WaitParams(target_error_mode)));
    IREE_EXPECT_STATUS_IS(
        expected_status,
        iree_hal_command_buffer_atomic_store(
            &command_buffer_, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
            IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE, target_ref,
            StoreParams(target_error_mode)));
    IREE_EXPECT_STATUS_IS(
        expected_status,
        iree_hal_command_buffer_atomic_rmw(
            &command_buffer_, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
            IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE, target_ref,
            RmwParams(target_error_mode)));
  }

  void RecordIndirectOperations(
      iree_hal_atomic_target_error_mode_t target_error_mode,
      iree_hal_atomic_width_t width = IREE_HAL_ATOMIC_WIDTH_32) {
    const iree_device_size_t length = iree_hal_atomic_width_byte_count(width);
    const iree_hal_buffer_ref_t target_ref = iree_hal_make_indirect_buffer_ref(
        /*buffer_slot=*/0, /*offset=*/0, length);
    IREE_ASSERT_OK(iree_hal_command_buffer_atomic_wait(
        &command_buffer_, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE, target_ref,
        WaitParams(target_error_mode, width)));
    IREE_ASSERT_OK(iree_hal_command_buffer_atomic_store(
        &command_buffer_, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE, target_ref,
        StoreParams(target_error_mode, width)));
    IREE_ASSERT_OK(iree_hal_command_buffer_atomic_rmw(
        &command_buffer_, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE, target_ref,
        RmwParams(target_error_mode, width)));
  }

  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(
        IREE_SV("atomic-test"), iree_allocator_system(),
        iree_allocator_system(), &allocator_));
    const iree_hal_buffer_params_t buffer_params = {
        .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
    };
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(allocator_, buffer_params,
                                                      16, &root_buffer_));
    IREE_ASSERT_OK(iree_hal_buffer_subspan(
        root_buffer_, 1, 8, iree_allocator_system(), &unaligned_buffer_));
    ASSERT_NE(0u, iree_hal_buffer_byte_offset(unaligned_buffer_) % 4);

    const iree_host_size_t validation_state_size =
        iree_hal_command_buffer_validation_state_size(/*mode=*/0,
                                                      /*binding_capacity=*/1);
    IREE_ASSERT_OK(iree_allocator_malloc(
        iree_allocator_system(), validation_state_size, &validation_state_));
    memset(validation_state_, 0, validation_state_size);
    command_buffer_vtable_.destroy = NoopCommandBufferDestroy;
    command_buffer_vtable_.begin = NoopCommandBufferBegin;
    command_buffer_vtable_.end = NoopCommandBufferEnd;
    command_buffer_vtable_.atomic_wait = NoopCommandBufferAtomicWait;
    command_buffer_vtable_.atomic_store = NoopCommandBufferAtomicStore;
    command_buffer_vtable_.atomic_rmw = NoopCommandBufferAtomicRmw;
    iree_hal_device_queue_spec_t queues = {.family_count = 1,
                                           .families = &kQueueFamilySpec};
    iree_hal_device_spec_params_t spec_params = {.queues = &queues};
    iree_hal_device_spec_t* device_spec = nullptr;
    IREE_ASSERT_OK(iree_hal_device_spec_create(
        &spec_params, iree_allocator_system(), &device_spec));
    iree_hal_mock_device_options_t device_options;
    iree_hal_mock_device_options_initialize(&device_options);
    device_options.device_spec = device_spec;
    iree_status_t status = iree_hal_mock_device_create(
        &device_options, iree_allocator_system(), &device_);
    iree_hal_device_spec_release(device_spec);
    IREE_ASSERT_OK(status);
    queue_family_ = iree_hal_device_queue_family(device_, 0);
    iree_hal_queue_params_t queue_params;
    iree_hal_queue_params_initialize(&queue_params);
    queue_vtable_.destroy = NoopQueueDestroy;
    queue_vtable_.atomic_wait = NoopQueueAtomicWait;
    queue_vtable_.atomic_store = NoopQueueAtomicStore;
    queue_vtable_.atomic_rmw = NoopQueueAtomicRmw;
    iree_hal_queue_initialize(queue_family_, &queue_params, &queue_vtable_,
                              &queue_);
    iree_hal_command_buffer_initialize(
        allocator_, queue_family_, /*mode=*/0, IREE_HAL_COMMAND_CATEGORY_ATOMIC,
        /*binding_capacity=*/1, validation_state_, &command_buffer_vtable_,
        &command_buffer_);
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(&command_buffer_));
  }

  void TearDown() override {
    iree_hal_command_buffer_release(&command_buffer_);
    iree_hal_queue_release(&queue_);
    iree_allocator_free(iree_allocator_system(), validation_state_);
    iree_hal_buffer_release(unaligned_buffer_);
    iree_hal_buffer_release(root_buffer_);
    iree_hal_allocator_release(allocator_);
    iree_hal_device_release(device_);
  }

  iree_hal_allocator_t* allocator_ = nullptr;
  iree_hal_buffer_t* root_buffer_ = nullptr;
  iree_hal_buffer_t* unaligned_buffer_ = nullptr;
  void* validation_state_ = nullptr;
  // Canonical family borrowed from device_.
  const iree_hal_queue_family_t* queue_family_ = nullptr;
  // Device owning the canonical family used by this fixture.
  iree_hal_device_t* device_ = nullptr;
  iree_hal_queue_vtable_t queue_vtable_ = {};
  iree_hal_queue_t queue_ = {};
  iree_hal_command_buffer_vtable_t command_buffer_vtable_ = {};
  iree_hal_command_buffer_t command_buffer_ = {};
};

TEST_F(AtomicTargetValidationTest,
       DirectQueueTargetAlignmentFollowsModeForAllOperations) {
  ExpectQueueAlignmentStatus(IREE_HAL_ATOMIC_TARGET_ERROR_MODE_DEFAULT,
                             IREE_STATUS_INVALID_ARGUMENT);
  ExpectQueueAlignmentStatus(IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE,
                             IREE_STATUS_INCOMPATIBLE);
}

TEST_F(AtomicTargetValidationTest,
       DirectReferenceTargetAlignmentFollowsModeForAllOperations) {
  ExpectDirectReferenceAlignmentStatus(
      IREE_HAL_ATOMIC_TARGET_ERROR_MODE_DEFAULT, IREE_STATUS_INVALID_ARGUMENT);
  ExpectDirectReferenceAlignmentStatus(
      IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE, IREE_STATUS_INCOMPATIBLE);
}

TEST_F(AtomicTargetValidationTest,
       DefaultIndirectTargetAlignmentPrecedesForAllOperations) {
  RecordIndirectOperations(IREE_HAL_ATOMIC_TARGET_ERROR_MODE_DEFAULT);
  IREE_ASSERT_OK(iree_hal_command_buffer_end(&command_buffer_));

  const iree_hal_buffer_binding_t binding = {
      .buffer = unaligned_buffer_,
      .offset = 0,
      .length = 4,
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      .count = 1,
      .bindings = &binding,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_command_buffer_validate_submission(
                            &command_buffer_, binding_table));
}

TEST_F(AtomicTargetValidationTest,
       IncompatibleIndirectTargetAlignmentAppliesForAllOperations) {
  RecordIndirectOperations(IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE);
  IREE_ASSERT_OK(iree_hal_command_buffer_end(&command_buffer_));

  const iree_hal_buffer_binding_t binding = {
      .buffer = unaligned_buffer_,
      .offset = 0,
      .length = 4,
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      .count = 1,
      .bindings = &binding,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INCOMPATIBLE,
                        iree_hal_command_buffer_validate_submission(
                            &command_buffer_, binding_table));
}

TEST_F(AtomicTargetValidationTest,
       MixedIndirectTargetAlignmentPreservesDefaultPrecedence) {
  RecordIndirectOperations(IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE,
                           IREE_HAL_ATOMIC_WIDTH_64);
  RecordIndirectOperations(IREE_HAL_ATOMIC_TARGET_ERROR_MODE_DEFAULT,
                           IREE_HAL_ATOMIC_WIDTH_32);
  IREE_ASSERT_OK(iree_hal_command_buffer_end(&command_buffer_));

  const iree_hal_buffer_binding_t binding = {
      .buffer = unaligned_buffer_,
      .offset = 0,
      .length = 8,
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      .count = 1,
      .bindings = &binding,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_command_buffer_validate_submission(
                            &command_buffer_, binding_table));
}

TEST_F(AtomicTargetValidationTest,
       MixedIndirectTargetAlignmentKeepsOptInRequirementSeparate) {
  RecordIndirectOperations(IREE_HAL_ATOMIC_TARGET_ERROR_MODE_DEFAULT,
                           IREE_HAL_ATOMIC_WIDTH_32);
  RecordIndirectOperations(IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE,
                           IREE_HAL_ATOMIC_WIDTH_64);
  IREE_ASSERT_OK(iree_hal_command_buffer_end(&command_buffer_));

  const iree_hal_buffer_binding_t binding = {
      .buffer = root_buffer_,
      .offset = 4,
      .length = 8,
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      .count = 1,
      .bindings = &binding,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INCOMPATIBLE,
                        iree_hal_command_buffer_validate_submission(
                            &command_buffer_, binding_table));
}

TEST_F(AtomicTargetValidationTest, RejectsOutOfRangeDirectTarget) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_command_buffer_atomic_store(
          &command_buffer_, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
          IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
          iree_hal_make_buffer_ref(root_buffer_, /*offset=*/16, /*length=*/4),
          StoreParams(IREE_HAL_ATOMIC_TARGET_ERROR_MODE_DEFAULT)));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_command_buffer_atomic_store(
          &command_buffer_, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
          IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
          iree_hal_make_buffer_ref(root_buffer_, /*offset=*/16, /*length=*/4),
          StoreParams(IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE)));
}

TEST_F(AtomicTargetValidationTest,
       OptInModeDoesNotReclassifyMalformedUsageOrAccessErrors) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_command_buffer_atomic_store(
          &command_buffer_, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
          IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
          iree_hal_make_buffer_ref(root_buffer_, /*offset=*/0, /*length=*/3),
          StoreParams(IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE)));

  const iree_hal_buffer_params_t bad_usage_params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
  };
  iree_hal_buffer_t* bad_usage_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator_, bad_usage_params, /*allocation_size=*/8, &bad_usage_buffer));
  const iree_hal_semaphore_list_t empty = iree_hal_semaphore_list_empty();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_atomic_store(
          &queue_, empty, empty, bad_usage_buffer, /*target_offset=*/0,
          StoreParams(IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE),
          /*barriers=*/NULL));
  iree_hal_buffer_release(bad_usage_buffer);

  const iree_hal_buffer_params_t bad_access_params = {
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE,
      .access = IREE_HAL_MEMORY_ACCESS_READ,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
  };
  iree_hal_buffer_t* bad_access_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator_, bad_access_params, /*allocation_size=*/8,
      &bad_access_buffer));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_atomic_store(
          &queue_, empty, empty, bad_access_buffer, /*target_offset=*/0,
          StoreParams(IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE),
          /*barriers=*/NULL));
  iree_hal_buffer_release(bad_access_buffer);
}

}  // namespace
