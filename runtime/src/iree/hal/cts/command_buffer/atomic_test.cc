// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>

#include "iree/hal/cts/util/atomic_test_util.h"
#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

class CommandBufferAtomicTest : public CtsTestBase<> {
 protected:
  static constexpr iree_device_size_t kBufferSize = 32;
  static constexpr iree_device_size_t kTargetOffset = 8;

  bool SelectConfiguration(iree_hal_atomic_width_t width,
                           AtomicTestConfiguration* out_configuration) {
    const AtomicTestRequirements requirements = {
        /*.operation_flags=*/IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
            IREE_HAL_ATOMIC_OPERATION_FLAG_STORE |
            IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD |
            IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_XOR,
        /*.wait_condition_flags=*/IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_EQUAL,
        /*.memory_type=*/IREE_HAL_MEMORY_TYPE_NONE,
        /*.buffer_usage=*/IREE_HAL_BUFFER_USAGE_STORAGE |
            IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*.memory_access=*/IREE_HAL_MEMORY_ACCESS_READ |
            IREE_HAL_MEMORY_ACCESS_WRITE,
        /*.width=*/width,
        /*.atomic_flags=*/IREE_HAL_ATOMIC_FLAG_ACQUIRE |
            IREE_HAL_ATOMIC_FLAG_RELEASE,
    };
    return SelectAtomicTestConfiguration(iree_hal_device_spec(device_),
                                         requirements, out_configuration);
  }

  iree_status_t AllocateAtomicBuffer(
      const AtomicTestConfiguration& configuration,
      iree_hal_buffer_t** out_buffer) {
    return iree_hal_allocator_allocate_buffer(device_allocator_,
                                              configuration.buffer_params,
                                              kBufferSize, out_buffer);
  }

  template <typename ValueType>
  ValueType ReadAtomicValue(iree_hal_buffer_t* buffer) {
    const std::vector<uint8_t> bytes =
        ReadBufferBytes(buffer, kTargetOffset, sizeof(ValueType));
    ValueType value = 0;
    EXPECT_EQ(bytes.size(), sizeof(value));
    if (bytes.size() == sizeof(value)) {
      memcpy(&value, bytes.data(), sizeof(value));
    }
    return value;
  }

  iree_status_t SubmitAtomicCommandBuffer(
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_buffer_t* indirect_buffer) {
    if (!indirect_buffer) {
      return SubmitCommandBufferAndWait(command_buffer);
    }
    const iree_hal_buffer_binding_t bindings[] = {
        {
            /*.buffer=*/indirect_buffer,
            /*.offset=*/0,
            /*.length=*/IREE_HAL_WHOLE_BUFFER,
        },
    };
    return SubmitCommandBufferAndWait(
        command_buffer,
        iree_hal_buffer_binding_table_t{IREE_ARRAYSIZE(bindings), bindings});
  }

  template <typename ValueType>
  void RunReusableCommandBufferTest(iree_hal_atomic_width_t width) {
    AtomicTestConfiguration configuration;
    if (!SelectConfiguration(width, &configuration)) {
      GTEST_SKIP() << "Device does not advertise the tested "
                      "queue/memory atomic command set";
    }

    Ref<iree_hal_buffer_t> first_buffer;
    IREE_ASSERT_OK(AllocateAtomicBuffer(configuration, first_buffer.out()));
    Ref<iree_hal_buffer_t> second_buffer;
    IREE_ASSERT_OK(AllocateAtomicBuffer(configuration, second_buffer.out()));

    const bool indirect = recording_mode() == RecordingMode::kIndirect;
    const iree_host_size_t binding_capacity = indirect ? 1 : 0;
    Ref<iree_hal_command_buffer_t> command_buffer;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_ATOMIC, configuration.queue_affinity,
        binding_capacity, command_buffer.out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

    const iree_device_size_t target_length =
        iree_hal_atomic_width_byte_count(width);
    const iree_hal_buffer_ref_t target_ref =
        indirect ? iree_hal_make_indirect_buffer_ref(
                       /*binding=*/0, kTargetOffset, target_length)
                 : iree_hal_make_buffer_ref(first_buffer, kTargetOffset,
                                            target_length);
    const iree_hal_atomic_flags_t atomic_flags =
        IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE;
    const iree_hal_execution_stage_t source_stage_mask =
        IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE;
    const iree_hal_execution_stage_t target_stage_mask =
        IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE;

    const iree_hal_atomic_store_params_t store_params = {
        /*.value=*/10,
        /*.flags=*/atomic_flags,
        /*.width=*/width,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_atomic_store(
        command_buffer, source_stage_mask, target_stage_mask, target_ref,
        store_params));

    iree_hal_atomic_rmw_params_t rmw_params = {
        /*.operand=*/5,
        /*.flags=*/atomic_flags,
        /*.width=*/width,
        /*.operation=*/IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_atomic_rmw(
        command_buffer, source_stage_mask, target_stage_mask, target_ref,
        rmw_params));

    const iree_hal_atomic_wait_params_t wait_params = {
        /*.value=*/7,
        /*.mask=*/7,
        /*.flags=*/atomic_flags,
        /*.width=*/width,
        /*.condition=*/IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_atomic_wait(
        command_buffer, source_stage_mask, target_stage_mask, target_ref,
        wait_params));

    rmw_params.operand = 3;
    rmw_params.operation = IREE_HAL_ATOMIC_RMW_OPERATION_XOR;
    IREE_ASSERT_OK(iree_hal_command_buffer_atomic_rmw(
        command_buffer, source_stage_mask, target_stage_mask, target_ref,
        rmw_params));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

    IREE_ASSERT_OK(SubmitAtomicCommandBuffer(
        command_buffer, indirect ? first_buffer.get() : nullptr));
    EXPECT_EQ(ReadAtomicValue<ValueType>(first_buffer),
              static_cast<ValueType>(12));

    IREE_ASSERT_OK(SubmitAtomicCommandBuffer(
        command_buffer, indirect ? second_buffer.get() : nullptr));
    if (indirect) {
      EXPECT_EQ(ReadAtomicValue<ValueType>(second_buffer),
                static_cast<ValueType>(12));
    } else {
      EXPECT_EQ(ReadAtomicValue<ValueType>(first_buffer),
                static_cast<ValueType>(12));
    }
  }
};

TEST_P(CommandBufferAtomicTest, Reusable32) {
  RunReusableCommandBufferTest<uint32_t>(IREE_HAL_ATOMIC_WIDTH_32);
}

TEST_P(CommandBufferAtomicTest, Reusable64) {
  RunReusableCommandBufferTest<uint64_t>(IREE_HAL_ATOMIC_WIDTH_64);
}

CTS_REGISTER_COMMAND_BUFFER_TEST_SUITE(CommandBufferAtomicTest);

}  // namespace iree::hal::cts
