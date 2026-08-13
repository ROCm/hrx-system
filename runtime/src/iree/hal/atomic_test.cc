// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/atomic.h"

#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

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

static iree_status_t NoopCommandBufferAtomicStore(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_store_params_t params) {
  return iree_ok_status();
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
      /*.value=*/1,
      /*.flags=*/IREE_HAL_ATOMIC_FLAG_ACQUIRE,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
  };
  IREE_EXPECT_OK(iree_hal_atomic_store_params_validate(params));
}

TEST(AtomicTest, ValidatesReadModifyWriteParameters) {
  iree_hal_atomic_rmw_params_t params = {
      /*.operand=*/1,
      /*.flags=*/IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE |
          IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_64,
      /*.operation=*/IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
  };
  IREE_EXPECT_OK(iree_hal_atomic_rmw_params_validate(params));

  params.operation = 0xFF;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_atomic_rmw_params_validate(params));
}

class AtomicTargetValidationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(
        IREE_SV("atomic-test"), iree_allocator_system(),
        iree_allocator_system(), &allocator_));
    const iree_hal_buffer_params_t buffer_params = {
        /*.usage=*/IREE_HAL_BUFFER_USAGE_STORAGE,
        /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
        /*.type=*/IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
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
    command_buffer_vtable_.atomic_store = NoopCommandBufferAtomicStore;
    iree_hal_command_buffer_initialize(
        allocator_, /*mode=*/0, IREE_HAL_COMMAND_CATEGORY_ATOMIC,
        /*queue_affinity=*/1, /*binding_capacity=*/1, validation_state_,
        &command_buffer_vtable_, &command_buffer_);
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(&command_buffer_));
  }

  void TearDown() override {
    iree_hal_command_buffer_release(&command_buffer_);
    iree_allocator_free(iree_allocator_system(), validation_state_);
    iree_hal_buffer_release(unaligned_buffer_);
    iree_hal_buffer_release(root_buffer_);
    iree_hal_allocator_release(allocator_);
  }

  iree_hal_allocator_t* allocator_ = nullptr;
  iree_hal_buffer_t* root_buffer_ = nullptr;
  iree_hal_buffer_t* unaligned_buffer_ = nullptr;
  void* validation_state_ = nullptr;
  iree_hal_command_buffer_vtable_t command_buffer_vtable_ = {};
  iree_hal_command_buffer_t command_buffer_ = {};
};

TEST_F(AtomicTargetValidationTest, ValidatesResolvedIndirectTargetAlignment) {
  const iree_hal_atomic_store_params_t store_params = {
      /*.value=*/1,
      /*.flags=*/IREE_HAL_ATOMIC_FLAG_NONE,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_atomic_store(
      &command_buffer_, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/0,
                                        /*length=*/4),
      store_params));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(&command_buffer_));

  const iree_hal_buffer_binding_t binding = {
      /*.buffer=*/unaligned_buffer_,
      /*.offset=*/0,
      /*.length=*/4,
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      /*.count=*/1,
      /*.bindings=*/&binding,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_command_buffer_validate_submission(
                            &command_buffer_, binding_table));
}

TEST_F(AtomicTargetValidationTest, RejectsOutOfRangeDirectTarget) {
  const iree_hal_atomic_store_params_t store_params = {
      /*.value=*/1,
      /*.flags=*/IREE_HAL_ATOMIC_FLAG_NONE,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_command_buffer_atomic_store(
          &command_buffer_, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
          IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
          iree_hal_make_buffer_ref(root_buffer_, /*offset=*/16, /*length=*/4),
          store_params));
}

}  // namespace
