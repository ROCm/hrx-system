// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/command_buffer.h"

#include <cstdint>
#include <cstring>

#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdxdna/allocator.h"
#include "iree/hal/drivers/amdxdna/device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct FakeTargetCommandBuffer {
  iree_hal_command_buffer_t base;
  int begin_count = 0;
  int end_count = 0;
  int barrier_count = 0;
  iree_hal_buffer_ref_t barrier_ref = {};
  int fill_count = 0;
  iree_hal_buffer_ref_t fill_target_ref = {};
  uint64_t fill_pattern = 0;
  iree_host_size_t fill_pattern_length = 0;
};

static FakeTargetCommandBuffer* FakeTarget(
    iree_hal_command_buffer_t* command_buffer) {
  return reinterpret_cast<FakeTargetCommandBuffer*>(command_buffer);
}

static void FakeDestroy(iree_hal_command_buffer_t* command_buffer) {}

static iree_status_t FakeBegin(iree_hal_command_buffer_t* command_buffer) {
  ++FakeTarget(command_buffer)->begin_count;
  return iree_ok_status();
}

static iree_status_t FakeEnd(iree_hal_command_buffer_t* command_buffer) {
  ++FakeTarget(command_buffer)->end_count;
  return iree_ok_status();
}

static iree_status_t FakeBeginDebugGroup(
    iree_hal_command_buffer_t* command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  return iree_ok_status();
}

static iree_status_t FakeEndDebugGroup(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_ok_status();
}

static iree_status_t FakeExecutionBarrier(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  FakeTargetCommandBuffer* target = FakeTarget(command_buffer);
  ++target->barrier_count;
  if (buffer_barrier_count > 0) {
    target->barrier_ref = buffer_barriers[0].buffer_ref;
  }
  return iree_ok_status();
}

static iree_status_t FakeAtomicWait(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_wait_params_t params) {
  (void)command_buffer;
  (void)source_stage_mask;
  (void)target_stage_mask;
  (void)target_ref;
  (void)params;
  return iree_ok_status();
}

static iree_status_t FakeAtomicStore(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_buffer_ref_t target_ref, iree_hal_atomic_store_params_t params) {
  (void)command_buffer;
  (void)source_stage_mask;
  (void)target_stage_mask;
  (void)target_ref;
  (void)params;
  return iree_ok_status();
}

static iree_status_t FakeAtomicRmw(iree_hal_command_buffer_t* command_buffer,
                                   iree_hal_execution_stage_t source_stage_mask,
                                   iree_hal_execution_stage_t target_stage_mask,
                                   iree_hal_buffer_ref_t target_ref,
                                   iree_hal_atomic_rmw_params_t params) {
  (void)command_buffer;
  (void)source_stage_mask;
  (void)target_stage_mask;
  (void)target_ref;
  (void)params;
  return iree_ok_status();
}

static iree_status_t FakeAdviseBuffer(iree_hal_command_buffer_t* command_buffer,
                                      iree_hal_buffer_ref_t buffer_ref,
                                      iree_hal_memory_advise_flags_t flags,
                                      uint64_t arg0, uint64_t arg1) {
  return iree_ok_status();
}

static iree_status_t FakeFillBuffer(iree_hal_command_buffer_t* command_buffer,
                                    iree_hal_buffer_ref_t target_ref,
                                    const void* pattern,
                                    iree_host_size_t pattern_length,
                                    iree_hal_fill_flags_t flags) {
  FakeTargetCommandBuffer* target = FakeTarget(command_buffer);
  ++target->fill_count;
  target->fill_target_ref = target_ref;
  target->fill_pattern = 0;
  memcpy(&target->fill_pattern, pattern, pattern_length);
  target->fill_pattern_length = pattern_length;
  return iree_ok_status();
}

static iree_status_t FakeUpdateBuffer(iree_hal_command_buffer_t* command_buffer,
                                      const void* source_buffer,
                                      iree_host_size_t source_offset,
                                      iree_hal_buffer_ref_t target_ref,
                                      iree_hal_update_flags_t flags) {
  return iree_ok_status();
}

static iree_status_t FakeCopyBuffer(iree_hal_command_buffer_t* command_buffer,
                                    iree_hal_buffer_ref_t source_ref,
                                    iree_hal_buffer_ref_t target_ref,
                                    iree_hal_copy_flags_t flags) {
  return iree_ok_status();
}

static iree_status_t FakeCollective(iree_hal_command_buffer_t* command_buffer,
                                    iree_hal_channel_t* channel,
                                    iree_hal_collective_op_t op, uint32_t param,
                                    iree_hal_buffer_ref_t send_ref,
                                    iree_hal_buffer_ref_t recv_ref,
                                    iree_device_size_t element_count) {
  return iree_ok_status();
}

static iree_status_t FakeDispatch(iree_hal_command_buffer_t* command_buffer,
                                  iree_hal_executable_t* executable,
                                  iree_hal_executable_function_t function,
                                  const iree_hal_dispatch_config_t config,
                                  iree_const_byte_span_t constants,
                                  iree_hal_buffer_ref_list_t bindings,
                                  iree_hal_dispatch_flags_t flags) {
  return iree_ok_status();
}

static const iree_hal_command_buffer_vtable_t kFakeTargetVtable = {
    /*.destroy=*/FakeDestroy,
    /*.begin=*/FakeBegin,
    /*.end=*/FakeEnd,
    /*.begin_debug_group=*/FakeBeginDebugGroup,
    /*.end_debug_group=*/FakeEndDebugGroup,
    /*.execution_barrier=*/FakeExecutionBarrier,
    /*.atomic_wait=*/FakeAtomicWait,
    /*.atomic_store=*/FakeAtomicStore,
    /*.atomic_rmw=*/FakeAtomicRmw,
    /*.advise_buffer=*/FakeAdviseBuffer,
    /*.fill_buffer=*/FakeFillBuffer,
    /*.update_buffer=*/FakeUpdateBuffer,
    /*.copy_buffer=*/FakeCopyBuffer,
    /*.collective=*/FakeCollective,
    /*.dispatch=*/FakeDispatch,
};

class CommandBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(/*total_block_size=*/8 * 1024,
                                     iree_allocator_system(), &block_pool_);
    IREE_ASSERT_OK(iree_hal_amdxdna_allocator_create(iree_allocator_system(),
                                                     /*native_device=*/nullptr,
                                                     &device_allocator_));
  }

  void TearDown() override {
    iree_hal_allocator_release(device_allocator_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_hal_allocator_t* device_allocator_ = nullptr;
  iree_hal_amdxdna_native_c_device_caps_t native_caps_ = {};
  iree_arena_block_pool_t block_pool_;
};

static void ExpectUnimplemented(iree_status_t status) {
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  iree_status_free(status);
}

TEST_F(CommandBufferTest, CreateReturnsAmdxdnaCommandBuffer) {
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_command_buffer_create(
      device_allocator_, &native_caps_,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/1, &block_pool_, iree_allocator_system(),
      &command_buffer));

  EXPECT_TRUE(iree_hal_amdxdna_command_buffer_isa(command_buffer));
  iree_hal_command_buffer_release(command_buffer);
}

TEST_F(CommandBufferTest, TransferCommandsRequireNativeBlitsAtRecordTime) {
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_command_buffer_create(
      device_allocator_, &native_caps_,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &block_pool_, iree_allocator_system(),
      &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  auto* fake_buffer = reinterpret_cast<iree_hal_buffer_t*>(uintptr_t{0x1});
  const iree_hal_buffer_ref_t buffer_ref =
      iree_hal_make_buffer_ref(fake_buffer, 0, 4);
  const iree_hal_buffer_ref_t empty_ref =
      iree_hal_make_buffer_ref(fake_buffer, 0, 0);
  uint32_t pattern = 0;

  IREE_EXPECT_OK(iree_hal_command_buffer_update_buffer(
      command_buffer, &pattern, /*source_offset=*/0, empty_ref,
      IREE_HAL_UPDATE_FLAG_NONE));
  IREE_EXPECT_OK(iree_hal_command_buffer_fill_buffer(command_buffer, empty_ref,
                                                     &pattern, sizeof(pattern),
                                                     IREE_HAL_FILL_FLAG_NONE));
  IREE_EXPECT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer, empty_ref, empty_ref, IREE_HAL_COPY_FLAG_NONE));

  ExpectUnimplemented(iree_hal_command_buffer_update_buffer(
      command_buffer, &pattern, /*source_offset=*/0, buffer_ref,
      IREE_HAL_UPDATE_FLAG_NONE));
  ExpectUnimplemented(iree_hal_command_buffer_fill_buffer(
      command_buffer, buffer_ref, &pattern, sizeof(pattern),
      IREE_HAL_FILL_FLAG_NONE));
  ExpectUnimplemented(iree_hal_command_buffer_copy_buffer(
      command_buffer, buffer_ref, buffer_ref, IREE_HAL_COPY_FLAG_NONE));

  iree_hal_command_buffer_release(command_buffer);
}

TEST_F(CommandBufferTest, ApplyResolvesIndirectBindingRefsAndResetsOneShot) {
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_command_buffer_create(
      device_allocator_, &native_caps_,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/1, &block_pool_, iree_allocator_system(),
      &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  iree_hal_buffer_barrier_t buffer_barrier = {};
  buffer_barrier.source_scope = IREE_HAL_ACCESS_SCOPE_HOST_WRITE;
  buffer_barrier.target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ;
  buffer_barrier.buffer_ref = iree_hal_make_indirect_buffer_ref(
      /*buffer_slot=*/0, /*offset=*/8, /*length=*/16);
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 0, nullptr, 1, &buffer_barrier));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  FakeTargetCommandBuffer target = {};
  iree_hal_command_buffer_initialize(
      /*device_allocator=*/nullptr,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, /*validation_state=*/nullptr, &kFakeTargetVtable,
      &target.base);

  iree_hal_buffer_t* fake_buffer =
      reinterpret_cast<iree_hal_buffer_t*>(uintptr_t{0x1234});
  iree_hal_buffer_binding_t binding = {/*buffer=*/fake_buffer, /*offset=*/32,
                                       /*length=*/128};
  iree_hal_buffer_binding_table_t binding_table = {/*count=*/1, &binding};
  IREE_ASSERT_OK(iree_hal_amdxdna_command_buffer_apply(
      command_buffer, &target.base, binding_table));

  EXPECT_EQ(target.begin_count, 1);
  EXPECT_EQ(target.end_count, 1);
  EXPECT_EQ(target.barrier_count, 1);
  EXPECT_EQ(target.barrier_ref.buffer, fake_buffer);
  EXPECT_EQ(target.barrier_ref.offset, 40);
  EXPECT_EQ(target.barrier_ref.length, 16);

  IREE_ASSERT_OK(iree_hal_amdxdna_command_buffer_apply(
      command_buffer, &target.base, binding_table));
  EXPECT_EQ(target.begin_count, 2);
  EXPECT_EQ(target.end_count, 2);
  EXPECT_EQ(target.barrier_count, 1);

  iree_hal_command_buffer_release(command_buffer);
}

TEST_F(CommandBufferTest, ApplySkipsMemoryOnlyBarriers) {
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_command_buffer_create(
      device_allocator_, &native_caps_,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/1, &block_pool_, iree_allocator_system(),
      &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  iree_hal_memory_barrier_t memory_barrier = {};
  memory_barrier.source_scope = IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE;
  memory_barrier.target_scope = IREE_HAL_ACCESS_SCOPE_MEMORY_READ;
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 1, &memory_barrier, 0, nullptr));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  FakeTargetCommandBuffer target = {};
  iree_hal_command_buffer_initialize(
      /*device_allocator=*/nullptr,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, /*validation_state=*/nullptr, &kFakeTargetVtable,
      &target.base);

  IREE_ASSERT_OK(iree_hal_amdxdna_command_buffer_apply(
      command_buffer, &target.base, iree_hal_buffer_binding_table_empty()));

  EXPECT_EQ(target.begin_count, 1);
  EXPECT_EQ(target.end_count, 1);
  EXPECT_EQ(target.barrier_count, 0);

  iree_hal_command_buffer_release(command_buffer);
}

TEST_F(CommandBufferTest, ApplyRetainsBufferBarriers) {
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_amdxdna_command_buffer_create(
      device_allocator_, &native_caps_,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/1, &block_pool_, iree_allocator_system(),
      &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  iree_hal_buffer_barrier_t buffer_barrier = {};
  buffer_barrier.source_scope = IREE_HAL_ACCESS_SCOPE_HOST_WRITE;
  buffer_barrier.target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ;
  buffer_barrier.buffer_ref = iree_hal_make_indirect_buffer_ref(
      /*buffer_slot=*/0, /*offset=*/8, /*length=*/16);
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 0, nullptr, 1, &buffer_barrier));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  FakeTargetCommandBuffer target = {};
  iree_hal_command_buffer_initialize(
      /*device_allocator=*/nullptr,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, /*validation_state=*/nullptr, &kFakeTargetVtable,
      &target.base);

  iree_hal_buffer_t* fake_buffer =
      reinterpret_cast<iree_hal_buffer_t*>(uintptr_t{0x1234});
  iree_hal_buffer_binding_t binding = {/*buffer=*/fake_buffer, /*offset=*/32,
                                       /*length=*/128};
  iree_hal_buffer_binding_table_t binding_table = {/*count=*/1, &binding};
  IREE_ASSERT_OK(iree_hal_amdxdna_command_buffer_apply(
      command_buffer, &target.base, binding_table));

  EXPECT_EQ(target.begin_count, 1);
  EXPECT_EQ(target.end_count, 1);
  EXPECT_EQ(target.barrier_count, 1);
  EXPECT_EQ(target.barrier_ref.buffer, fake_buffer);
  EXPECT_EQ(target.barrier_ref.offset, 40);
  EXPECT_EQ(target.barrier_ref.length, 16);

  iree_hal_command_buffer_release(command_buffer);
}

}  // namespace
