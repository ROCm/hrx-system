// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/deferred_command_buffer.h"

#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

enum class CapturedAtomicCommandType {
  kWait,
  kStore,
  kReadModifyWrite,
};

struct CapturedAtomicCommand {
  // Kind of atomic command captured.
  CapturedAtomicCommandType type;
  // Stages preceding the atomic operation.
  iree_hal_execution_stage_t source_stage_mask;
  // Stages following the atomic operation.
  iree_hal_execution_stage_t target_stage_mask;
  // Resolved target buffer and range.
  iree_hal_buffer_ref_t target_ref;
  // Buffer length observed while the callback is executing.
  iree_device_size_t target_buffer_length;
  // Parameters captured for a wait command.
  iree_hal_atomic_wait_params_t wait_params;
  // Parameters captured for a store command.
  iree_hal_atomic_store_params_t store_params;
  // Parameters captured for a read-modify-write command.
  iree_hal_atomic_rmw_params_t rmw_params;
};

struct CapturingCommandBuffer {
  // HAL command buffer interface.
  iree_hal_command_buffer_t base;
  // Number of begin calls received.
  iree_host_size_t begin_count;
  // Number of end calls received.
  iree_host_size_t end_count;
  // Number of atomic commands captured.
  iree_host_size_t command_count;
  // Atomic commands in callback order.
  CapturedAtomicCommand commands[3];
};

static CapturingCommandBuffer* CastCapturingCommandBuffer(
    iree_hal_command_buffer_t* base_command_buffer) {
  return reinterpret_cast<CapturingCommandBuffer*>(base_command_buffer);
}

static void CapturingCommandBufferDestroy(
    iree_hal_command_buffer_t* base_command_buffer) {}

static iree_status_t CapturingCommandBufferBegin(
    iree_hal_command_buffer_t* base_command_buffer) {
  ++CastCapturingCommandBuffer(base_command_buffer)->begin_count;
  return iree_ok_status();
}

static iree_status_t CapturingCommandBufferEnd(
    iree_hal_command_buffer_t* base_command_buffer) {
  ++CastCapturingCommandBuffer(base_command_buffer)->end_count;
  return iree_ok_status();
}

static iree_status_t CapturingCommandBufferAppend(
    iree_hal_command_buffer_t* base_command_buffer,
    CapturedAtomicCommandType type,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, CapturedAtomicCommand** out_command) {
  CapturingCommandBuffer* command_buffer =
      CastCapturingCommandBuffer(base_command_buffer);
  if (command_buffer->command_count >=
      IREE_ARRAYSIZE(command_buffer->commands)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "capture command capacity exceeded");
  }
  CapturedAtomicCommand* command =
      &command_buffer->commands[command_buffer->command_count++];
  command->type = type;
  command->source_stage_mask = source_stage_mask;
  command->target_stage_mask = target_stage_mask;
  command->target_ref = target_ref;
  command->target_buffer_length =
      iree_hal_buffer_byte_length(target_ref.buffer);
  *out_command = command;
  return iree_ok_status();
}

static iree_status_t CapturingCommandBufferAtomicWait(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_wait_params_t params) {
  CapturedAtomicCommand* command = nullptr;
  IREE_RETURN_IF_ERROR(CapturingCommandBufferAppend(
      base_command_buffer, CapturedAtomicCommandType::kWait, source_stage_mask,
      target_stage_mask, target_ref, &command));
  command->wait_params = params;
  return iree_ok_status();
}

static iree_status_t CapturingCommandBufferAtomicStore(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_store_params_t params) {
  CapturedAtomicCommand* command = nullptr;
  IREE_RETURN_IF_ERROR(CapturingCommandBufferAppend(
      base_command_buffer, CapturedAtomicCommandType::kStore, source_stage_mask,
      target_stage_mask, target_ref, &command));
  command->store_params = params;
  return iree_ok_status();
}

static iree_status_t CapturingCommandBufferAtomicRmw(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_rmw_params_t params) {
  CapturedAtomicCommand* command = nullptr;
  IREE_RETURN_IF_ERROR(CapturingCommandBufferAppend(
      base_command_buffer, CapturedAtomicCommandType::kReadModifyWrite,
      source_stage_mask, target_stage_mask, target_ref, &command));
  command->rmw_params = params;
  return iree_ok_status();
}

class DeferredCommandBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(/*block_size=*/4096,
                                     iree_allocator_system(), &block_pool_);
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(
        IREE_SV("deferred-command-buffer-test"), iree_allocator_system(),
        iree_allocator_system(), &allocator_));
    const iree_hal_buffer_params_t buffer_params = {
        /*.usage=*/IREE_HAL_BUFFER_USAGE_STORAGE,
        /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
        /*.type=*/IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
    };
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        allocator_, buffer_params, /*allocation_size=*/64, &direct_buffer_));
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        allocator_, buffer_params, /*allocation_size=*/64, &binding_buffer_));
    IREE_ASSERT_OK(iree_hal_deferred_command_buffer_create(
        allocator_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_ATOMIC, /*queue_affinity=*/1,
        /*binding_capacity=*/1, &block_pool_, iree_allocator_system(),
        &deferred_command_buffer_));

    target_vtable_.destroy = CapturingCommandBufferDestroy;
    target_vtable_.begin = CapturingCommandBufferBegin;
    target_vtable_.end = CapturingCommandBufferEnd;
    target_vtable_.atomic_wait = CapturingCommandBufferAtomicWait;
    target_vtable_.atomic_store = CapturingCommandBufferAtomicStore;
    target_vtable_.atomic_rmw = CapturingCommandBufferAtomicRmw;
    iree_hal_command_buffer_initialize(
        allocator_, IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
        IREE_HAL_COMMAND_CATEGORY_ATOMIC, /*queue_affinity=*/1,
        /*binding_capacity=*/0, /*validation_state=*/nullptr, &target_vtable_,
        &target_command_buffer_.base);
    target_command_buffer_initialized_ = true;
  }

  void TearDown() override {
    if (target_command_buffer_initialized_) {
      iree_hal_command_buffer_release(&target_command_buffer_.base);
    }
    iree_hal_command_buffer_release(deferred_command_buffer_);
    iree_hal_buffer_release(binding_buffer_);
    iree_hal_buffer_release(direct_buffer_);
    iree_hal_allocator_release(allocator_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Arena backing deferred command storage and resource retention.
  iree_arena_block_pool_t block_pool_;
  // Heap allocator backing test buffers.
  iree_hal_allocator_t* allocator_ = nullptr;
  // Direct target retained by the deferred command buffer.
  iree_hal_buffer_t* direct_buffer_ = nullptr;
  // Buffer provided through the binding table at replay time.
  iree_hal_buffer_t* binding_buffer_ = nullptr;
  // Deferred command buffer under test.
  iree_hal_command_buffer_t* deferred_command_buffer_ = nullptr;
  // Callback table for the capturing target command buffer.
  iree_hal_command_buffer_vtable_t target_vtable_ = {};
  // Target receiving replayed atomic operations.
  CapturingCommandBuffer target_command_buffer_ = {};
  // Indicates that the target command buffer owns a resource reference.
  bool target_command_buffer_initialized_ = false;
};

TEST_F(DeferredCommandBufferTest, ReplaysAtomicOperations) {
  const iree_hal_atomic_wait_params_t wait_params = {
      /*.value=*/UINT64_C(0x1020304050607080),
      /*.mask=*/UINT64_MAX,
      /*.flags=*/IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE |
          IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_64,
      /*.condition=*/IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL,
  };
  const iree_hal_atomic_store_params_t store_params = {
      /*.value=*/UINT64_C(0xAABBCCDD),
      /*.flags=*/IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
  };
  const iree_hal_atomic_rmw_params_t rmw_params = {
      /*.operand=*/7,
      /*.flags=*/IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE |
          IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
      /*.operation=*/IREE_HAL_ATOMIC_RMW_OPERATION_XOR,
  };

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(deferred_command_buffer_));
  IREE_ASSERT_OK(iree_hal_command_buffer_atomic_wait(
      deferred_command_buffer_, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_ATOMIC,
      iree_hal_make_buffer_ref(direct_buffer_, /*offset=*/8, /*length=*/8),
      wait_params));
  IREE_ASSERT_OK(iree_hal_command_buffer_atomic_store(
      deferred_command_buffer_, IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/8,
                                        /*length=*/4),
      store_params));
  IREE_ASSERT_OK(iree_hal_command_buffer_atomic_rmw(
      deferred_command_buffer_, IREE_HAL_EXECUTION_STAGE_HOST,
      IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS,
      iree_hal_make_buffer_ref(direct_buffer_, /*offset=*/32, /*length=*/4),
      rmw_params));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(deferred_command_buffer_));

  iree_hal_buffer_t* retained_direct_buffer = direct_buffer_;
  iree_hal_buffer_release(direct_buffer_);
  direct_buffer_ = nullptr;

  const iree_hal_buffer_binding_t binding = {
      /*.buffer=*/binding_buffer_,
      /*.offset=*/16,
      /*.length=*/32,
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      /*.count=*/1,
      /*.bindings=*/&binding,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_validate_submission(
      deferred_command_buffer_, binding_table));
  IREE_ASSERT_OK(iree_hal_deferred_command_buffer_apply(
      deferred_command_buffer_, &target_command_buffer_.base, binding_table));

  EXPECT_EQ(1u, target_command_buffer_.begin_count);
  EXPECT_EQ(1u, target_command_buffer_.end_count);
  ASSERT_EQ(3u, target_command_buffer_.command_count);

  const CapturedAtomicCommand& wait_command =
      target_command_buffer_.commands[0];
  EXPECT_EQ(CapturedAtomicCommandType::kWait, wait_command.type);
  EXPECT_EQ(IREE_HAL_EXECUTION_STAGE_DISPATCH, wait_command.source_stage_mask);
  EXPECT_EQ(IREE_HAL_EXECUTION_STAGE_ATOMIC, wait_command.target_stage_mask);
  EXPECT_EQ(retained_direct_buffer, wait_command.target_ref.buffer);
  EXPECT_EQ(8u, wait_command.target_ref.offset);
  EXPECT_EQ(8u, wait_command.target_ref.length);
  EXPECT_EQ(64u, wait_command.target_buffer_length);
  EXPECT_EQ(wait_params.value, wait_command.wait_params.value);
  EXPECT_EQ(wait_params.mask, wait_command.wait_params.mask);
  EXPECT_EQ(wait_params.flags, wait_command.wait_params.flags);
  EXPECT_EQ(wait_params.width, wait_command.wait_params.width);
  EXPECT_EQ(wait_params.condition, wait_command.wait_params.condition);
  EXPECT_EQ(wait_params.reserved, wait_command.wait_params.reserved);

  const CapturedAtomicCommand& store_command =
      target_command_buffer_.commands[1];
  EXPECT_EQ(CapturedAtomicCommandType::kStore, store_command.type);
  EXPECT_EQ(IREE_HAL_EXECUTION_STAGE_TRANSFER, store_command.source_stage_mask);
  EXPECT_EQ(IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
            store_command.target_stage_mask);
  EXPECT_EQ(binding_buffer_, store_command.target_ref.buffer);
  EXPECT_EQ(24u, store_command.target_ref.offset);
  EXPECT_EQ(4u, store_command.target_ref.length);
  EXPECT_EQ(64u, store_command.target_buffer_length);
  EXPECT_EQ(store_params.value, store_command.store_params.value);
  EXPECT_EQ(store_params.flags, store_command.store_params.flags);
  EXPECT_EQ(store_params.width, store_command.store_params.width);
  EXPECT_EQ(store_params.reserved[0], store_command.store_params.reserved[0]);
  EXPECT_EQ(store_params.reserved[1], store_command.store_params.reserved[1]);
  EXPECT_EQ(store_params.reserved[2], store_command.store_params.reserved[2]);

  const CapturedAtomicCommand& rmw_command = target_command_buffer_.commands[2];
  EXPECT_EQ(CapturedAtomicCommandType::kReadModifyWrite, rmw_command.type);
  EXPECT_EQ(IREE_HAL_EXECUTION_STAGE_HOST, rmw_command.source_stage_mask);
  EXPECT_EQ(IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS,
            rmw_command.target_stage_mask);
  EXPECT_EQ(retained_direct_buffer, rmw_command.target_ref.buffer);
  EXPECT_EQ(32u, rmw_command.target_ref.offset);
  EXPECT_EQ(4u, rmw_command.target_ref.length);
  EXPECT_EQ(64u, rmw_command.target_buffer_length);
  EXPECT_EQ(rmw_params.operand, rmw_command.rmw_params.operand);
  EXPECT_EQ(rmw_params.flags, rmw_command.rmw_params.flags);
  EXPECT_EQ(rmw_params.width, rmw_command.rmw_params.width);
  EXPECT_EQ(rmw_params.operation, rmw_command.rmw_params.operation);
  EXPECT_EQ(rmw_params.reserved, rmw_command.rmw_params.reserved);
}

}  // namespace
