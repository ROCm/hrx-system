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
  EXPECT_EQ(command.header.length, 65544u);

  IREE_ASSERT_OK(iree_hal_command_buffer_wait_events(
      command_buffer, /*event_count=*/0, /*events=*/nullptr,
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE, memory_barriers.size(),
      memory_barriers.data(), /*buffer_barrier_count=*/0,
      /*buffer_barriers=*/nullptr));

  stream = iree_hal_remote_client_command_buffer_stream(command_buffer);
  const iree_host_size_t wait_offset =
      barrier_offset + command.bytes.data_length;
  IREE_ASSERT_OK(iree_hal_remote_command_parse(
      iree_make_const_byte_span(stream.data + wait_offset,
                                stream.data_length - wait_offset),
      &command));
  EXPECT_EQ(command.header.type, IREE_HAL_REMOTE_CMD_EVENT_WAIT);
  EXPECT_EQ(command.header.length, 65544u);

  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
  iree_hal_command_buffer_release(command_buffer);
}

}  // namespace
