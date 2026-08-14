// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/command_buffer.h"

#include <cstring>
#include <string>
#include <vector>

#include "iree/async/buffer_pool.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/slab.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/client/api.h"
#include "iree/hal/remote/protocol/commands.h"
#include "iree/hal/remote/util/recv_pool.h"
#include "iree/net/carrier/loopback/factory.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class RemoteCommandBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));

    iree_async_slab_options_t slab_options = {};
    slab_options.buffer_size = 128 * 1024;
    slab_options.buffer_count = 1;
    IREE_ASSERT_OK(
        iree_async_slab_create(slab_options, iree_allocator_system(), &slab_));
    IREE_ASSERT_OK(iree_async_proactor_register_slab(
        proactor_, slab_, IREE_ASYNC_BUFFER_ACCESS_FLAG_NONE, &region_));

    iree_async_buffer_pool_t* buffer_pool = nullptr;
    IREE_ASSERT_OK(iree_async_buffer_pool_create(
        region_, iree_allocator_system(), &buffer_pool));
    iree_status_t status =
        iree_hal_remote_recv_pool_wrap(proactor_, slab_, region_, buffer_pool,
                                       iree_allocator_system(), &recv_pool_);
    if (!iree_status_is_ok(status)) {
      iree_async_buffer_pool_release(buffer_pool);
    }
    IREE_ASSERT_OK(status);

    iree_net_loopback_factory_options_t factory_options =
        iree_net_loopback_factory_options_default();
    IREE_ASSERT_OK(iree_net_loopback_factory_create(
        factory_options, iree_allocator_system(), &factory_));

    iree_hal_remote_client_device_options_t device_options;
    iree_hal_remote_client_device_options_initialize(&device_options);
    device_options.transport_factory = factory_;
    device_options.server_address = IREE_SV("unused");
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    IREE_ASSERT_OK(iree_hal_remote_client_device_create(
        IREE_SV("remote"), &device_options, &create_params, recv_pool_,
        iree_allocator_system(), &device_));
  }

  void TearDown() override {
    iree_hal_device_release(device_);
    iree_net_transport_factory_release(factory_);
    iree_hal_remote_recv_pool_release(recv_pool_);
    iree_async_region_release(region_);
    iree_async_slab_release(slab_);
    iree_async_proactor_release(proactor_);
  }

  iree_status_t CreateCommandBuffer(
      iree_hal_command_buffer_t** out_command_buffer) {
    return iree_hal_command_buffer_create(
        device_,
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
            IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
        IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/1, out_command_buffer);
  }

  iree_async_proactor_t* proactor_ = nullptr;
  iree_async_slab_t* slab_ = nullptr;
  iree_async_region_t* region_ = nullptr;
  iree_hal_remote_recv_pool_t* recv_pool_ = nullptr;
  iree_net_transport_factory_t* factory_ = nullptr;
  iree_hal_device_t* device_ = nullptr;
};

TEST_F(RemoteCommandBufferTest, BindingCapacityUses16BitWireWidth) {
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/UINT16_MAX, &command_buffer));
  iree_hal_command_buffer_release(command_buffer);

  command_buffer = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_command_buffer_create(
          device_, IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
          IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
          /*binding_capacity=*/(iree_host_size_t)UINT16_MAX + 1,
          &command_buffer));
  EXPECT_EQ(command_buffer, nullptr);
}

TEST_F(RemoteCommandBufferTest, MaximumUpdateUses32BitLength) {
  constexpr iree_host_size_t kUpdateLength =
      IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE;
  std::vector<uint8_t> source_data(kUpdateLength);
  for (iree_host_size_t i = 0; i < source_data.size(); ++i) {
    source_data[i] = static_cast<uint8_t>(i * 31u + 7u);
  }

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(CreateCommandBuffer(&command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      command_buffer, source_data.data(), /*source_offset=*/0,
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/0,
                                        kUpdateLength),
      IREE_HAL_UPDATE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_const_byte_span_t stream =
      iree_hal_remote_client_command_buffer_stream(command_buffer);
  iree_hal_remote_command_view_t command;
  IREE_ASSERT_OK(iree_hal_remote_command_parse(stream, &command));
  EXPECT_EQ(command.header.type, IREE_HAL_REMOTE_CMD_BUFFER_UPDATE);
  EXPECT_EQ(command.header.length, 65576u);
  EXPECT_EQ(command.bytes.data_length, stream.data_length);

  iree_hal_command_buffer_release(command_buffer);
}

TEST_F(RemoteCommandBufferTest, RejectsUpdateAboveMaximumWithoutMutation) {
  constexpr iree_host_size_t kUpdateLength =
      IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE + 1;
  std::vector<uint8_t> source_data(kUpdateLength);

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(CreateCommandBuffer(&command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  EXPECT_EQ(
      iree_hal_remote_client_command_buffer_stream(command_buffer).data_length,
      0u);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_command_buffer_update_buffer(
          command_buffer, source_data.data(), /*source_offset=*/0,
          iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/0,
                                            kUpdateLength),
          IREE_HAL_UPDATE_FLAG_NONE));
  EXPECT_EQ(
      iree_hal_remote_client_command_buffer_stream(command_buffer).data_length,
      0u);

  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
  iree_hal_command_buffer_release(command_buffer);
}

TEST_F(RemoteCommandBufferTest, ExecutionBarrierPreservesFlags) {
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(CreateCommandBuffer(&command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  const iree_hal_execution_barrier_flags_t flags =
      IREE_HAL_EXECUTION_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE |
      IREE_HAL_EXECUTION_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE;
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer, IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_STAGE_DISPATCH, flags,
      /*memory_barrier_count=*/0, /*memory_barriers=*/nullptr,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_const_byte_span_t stream =
      iree_hal_remote_client_command_buffer_stream(command_buffer);
  iree_hal_remote_command_view_t command;
  IREE_ASSERT_OK(iree_hal_remote_command_parse(stream, &command));
  ASSERT_EQ(command.header.type, IREE_HAL_REMOTE_CMD_EXECUTION_BARRIER);
  ASSERT_EQ(command.bytes.data_length,
            sizeof(iree_hal_remote_execution_barrier_cmd_t));
  iree_hal_remote_execution_barrier_cmd_t barrier;
  memcpy(&barrier, command.bytes.data, sizeof(barrier));
  EXPECT_EQ(barrier.barrier_flags, flags);
  EXPECT_EQ(barrier.source_stage_mask, IREE_HAL_EXECUTION_STAGE_TRANSFER);
  EXPECT_EQ(barrier.target_stage_mask, IREE_HAL_EXECUTION_STAGE_DISPATCH);
  EXPECT_EQ(barrier.reserved, 0u);

  iree_hal_command_buffer_release(command_buffer);
}

TEST_F(RemoteCommandBufferTest, AtomicCommandsPreserveParameters) {
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(CreateCommandBuffer(&command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  const iree_hal_atomic_wait_params_t wait_params = {
      /*.value=*/UINT64_C(0x0123456789ABCDEF),
      /*.mask=*/UINT64_C(0xFEDCBA9876543210),
      /*.flags=*/IREE_HAL_ATOMIC_FLAG_ACQUIRE |
          IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_64,
      /*.condition=*/IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_atomic_wait(
      command_buffer, IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_STAGE_DISPATCH,
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/24,
                                        /*length=*/8),
      wait_params));

  const iree_hal_atomic_store_params_t store_params = {
      /*.value=*/UINT32_C(0x89ABCDEF),
      /*.flags=*/IREE_HAL_ATOMIC_FLAG_RELEASE |
          IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_atomic_store(
      command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/40,
                                        /*length=*/4),
      store_params));

  const iree_hal_atomic_rmw_params_t rmw_params = {
      /*.operand=*/UINT64_C(0x1020304050607080),
      /*.flags=*/IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_64,
      /*.operation=*/IREE_HAL_ATOMIC_RMW_OPERATION_XOR,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_atomic_rmw(
      command_buffer, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_TRANSFER,
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/56,
                                        /*length=*/8),
      rmw_params));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_const_byte_span_t stream =
      iree_hal_remote_client_command_buffer_stream(command_buffer);
  iree_host_size_t offset = 0;
  iree_hal_remote_command_view_t command;

  IREE_ASSERT_OK(iree_hal_remote_command_parse(
      iree_make_const_byte_span(stream.data + offset,
                                stream.data_length - offset),
      &command));
  ASSERT_EQ(command.header.type, IREE_HAL_REMOTE_CMD_ATOMIC_WAIT);
  iree_hal_remote_atomic_wait_cmd_t wait_command;
  memcpy(&wait_command, command.bytes.data, sizeof(wait_command));
  EXPECT_EQ(wait_command.target.buffer_id, 0u);
  EXPECT_EQ(wait_command.target.buffer_slot, 0u);
  EXPECT_EQ(wait_command.target.offset, 24u);
  EXPECT_EQ(wait_command.target.length, 8u);
  EXPECT_EQ(wait_command.source_stage_mask, IREE_HAL_EXECUTION_STAGE_TRANSFER);
  EXPECT_EQ(wait_command.target_stage_mask, IREE_HAL_EXECUTION_STAGE_DISPATCH);
  EXPECT_EQ(wait_command.params.value, wait_params.value);
  EXPECT_EQ(wait_command.params.mask, wait_params.mask);
  EXPECT_EQ(wait_command.params.flags, wait_params.flags);
  EXPECT_EQ(wait_command.params.width, wait_params.width);
  EXPECT_EQ(wait_command.params.condition, wait_params.condition);
  offset += command.bytes.data_length;

  IREE_ASSERT_OK(iree_hal_remote_command_parse(
      iree_make_const_byte_span(stream.data + offset,
                                stream.data_length - offset),
      &command));
  ASSERT_EQ(command.header.type, IREE_HAL_REMOTE_CMD_ATOMIC_STORE);
  iree_hal_remote_atomic_store_cmd_t store_command;
  memcpy(&store_command, command.bytes.data, sizeof(store_command));
  EXPECT_EQ(store_command.target.offset, 40u);
  EXPECT_EQ(store_command.target.length, 4u);
  EXPECT_EQ(store_command.source_stage_mask,
            IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE);
  EXPECT_EQ(store_command.target_stage_mask,
            IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE);
  EXPECT_EQ(store_command.params.value, store_params.value);
  EXPECT_EQ(store_command.params.flags, store_params.flags);
  EXPECT_EQ(store_command.params.width, store_params.width);
  offset += command.bytes.data_length;

  IREE_ASSERT_OK(iree_hal_remote_command_parse(
      iree_make_const_byte_span(stream.data + offset,
                                stream.data_length - offset),
      &command));
  ASSERT_EQ(command.header.type, IREE_HAL_REMOTE_CMD_ATOMIC_RMW);
  iree_hal_remote_atomic_rmw_cmd_t rmw_command;
  memcpy(&rmw_command, command.bytes.data, sizeof(rmw_command));
  EXPECT_EQ(rmw_command.target.offset, 56u);
  EXPECT_EQ(rmw_command.target.length, 8u);
  EXPECT_EQ(rmw_command.source_stage_mask, IREE_HAL_EXECUTION_STAGE_DISPATCH);
  EXPECT_EQ(rmw_command.target_stage_mask, IREE_HAL_EXECUTION_STAGE_TRANSFER);
  EXPECT_EQ(rmw_command.params.operand, rmw_params.operand);
  EXPECT_EQ(rmw_command.params.flags, rmw_params.flags);
  EXPECT_EQ(rmw_command.params.width, rmw_params.width);
  EXPECT_EQ(rmw_command.params.operation, rmw_params.operation);
  offset += command.bytes.data_length;
  EXPECT_EQ(offset, stream.data_length);

  iree_hal_command_buffer_release(command_buffer);
}

TEST_F(RemoteCommandBufferTest, LargeVariableRecordsUse32BitLengths) {
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(CreateCommandBuffer(&command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  std::string maximum_label(UINT16_MAX, 'L');
  IREE_ASSERT_OK(iree_hal_command_buffer_begin_debug_group(
      command_buffer,
      iree_make_string_view(maximum_label.data(), maximum_label.size()),
      iree_hal_label_color_unspecified(), /*location=*/nullptr));
  IREE_ASSERT_OK(iree_hal_command_buffer_end_debug_group(command_buffer));

  iree_const_byte_span_t stream =
      iree_hal_remote_client_command_buffer_stream(command_buffer);
  iree_hal_remote_command_view_t command;
  IREE_ASSERT_OK(iree_hal_remote_command_parse(stream, &command));
  EXPECT_EQ(command.header.type, IREE_HAL_REMOTE_CMD_DEBUG_GROUP_BEGIN);
  EXPECT_EQ(command.header.length, 65552u);

  const iree_host_size_t stream_length_before_error = stream.data_length;
  std::string oversized_label(UINT16_MAX + 1u, 'X');
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_command_buffer_begin_debug_group(
          command_buffer,
          iree_make_string_view(oversized_label.data(), oversized_label.size()),
          iree_hal_label_color_unspecified(), /*location=*/nullptr));
  stream = iree_hal_remote_client_command_buffer_stream(command_buffer);
  EXPECT_EQ(stream.data_length, stream_length_before_error);

  constexpr iree_host_size_t kMemoryBarrierCount = 8190;
  std::vector<iree_hal_memory_barrier_t> memory_barriers(kMemoryBarrierCount);
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, memory_barriers.size(),
      memory_barriers.data(), /*buffer_barrier_count=*/0,
      /*buffer_barriers=*/nullptr));

  stream = iree_hal_remote_client_command_buffer_stream(command_buffer);
  iree_host_size_t barrier_offset = 0;
  while (barrier_offset < stream.data_length) {
    IREE_ASSERT_OK(iree_hal_remote_command_parse(
        iree_make_const_byte_span(stream.data + barrier_offset,
                                  stream.data_length - barrier_offset),
        &command));
    if (command.header.type == IREE_HAL_REMOTE_CMD_EXECUTION_BARRIER) break;
    barrier_offset += command.bytes.data_length;
  }
  ASSERT_LT(barrier_offset, stream.data_length);
  EXPECT_EQ(command.header.length, 65552u);

  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
  iree_hal_command_buffer_release(command_buffer);
}

}  // namespace
