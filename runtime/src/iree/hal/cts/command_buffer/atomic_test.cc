// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <future>
#include <utility>

#include "iree/hal/cts/util/atomic_test_util.h"
#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

using iree::testing::status::StatusIs;

static void NotifyBufferReleased(
    void* user_data, iree_hal_external_storage_class_t storage_class,
    iree_hal_external_storage_payload_t payload) {
  (void)storage_class;
  (void)payload;
  static_cast<std::promise<void>*>(user_data)->set_value();
}

class CommandBufferAtomicTest : public CtsTestBase<> {
 protected:
  static constexpr iree_device_size_t kBufferSize = 32;
  static constexpr iree_device_size_t kTargetOffset = 8;

  bool SelectConfiguration(
      iree_hal_atomic_width_t width,
      iree_hal_atomic_operation_flags_t operation_flags,
      iree_hal_atomic_wait_condition_flags_t wait_condition_flags,
      AtomicTestConfiguration* out_configuration,
      iree_hal_atomic_flags_t scope_flags = IREE_HAL_ATOMIC_FLAG_NONE) {
    const AtomicTestRequirements requirements = {
        .operation_flags = operation_flags,
        .wait_condition_flags = wait_condition_flags,
        .buffer_usage =
            IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        .width = width,
        .atomic_flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE |
                        IREE_HAL_ATOMIC_FLAG_RELEASE | scope_flags,
    };
    if (!SelectAtomicTestConfiguration(device_, requirements,
                                       out_configuration)) {
      return false;
    }
    atomic_queue_ =
        iree_hal_device_queue(device_, out_configuration->queue_family_ordinal,
                              /*queue_ordinal=*/0);
    if (!atomic_queue_) {
      return false;
    }
    Ref<iree_hal_pool_t> pool;
    Status pool_status(
        CreateAtomicTestPool(device_group_, *out_configuration, pool.out()));
    if (!pool_status.ok()) {
      if (pool_status.code() != StatusCode::kUnavailable) {
        ADD_FAILURE() << pool_status.ToString();
      }
      return false;
    }
    atomic_pool_ = std::move(pool);
    return true;
  }

  iree_status_t AllocateAtomicBuffer(
      const AtomicTestConfiguration& configuration,
      iree_hal_buffer_t** out_buffer) {
    return iree_hal_pool_allocate_buffer(
        atomic_pool_,
        iree_hal_make_pool_reservation_request(kBufferSize,
                                               configuration.min_alignment),
        iree_infinite_timeout(), out_buffer);
  }

  iree_status_t CreateAtomicCommandBuffer(
      iree_hal_command_buffer_mode_t mode, iree_host_size_t binding_capacity,
      iree_hal_command_buffer_t** out_command_buffer) {
    return iree_hal_command_buffer_create(
        iree_hal_queue_family(atomic_queue_), mode,
        IREE_HAL_COMMAND_CATEGORY_ATOMIC, binding_capacity, out_command_buffer);
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

  template <typename ValueType>
  iree_status_t QueueStoreAtomicValue(iree_hal_buffer_t* buffer,
                                      iree_hal_atomic_width_t width,
                                      ValueType value) {
    SemaphoreList empty_wait;
    SemaphoreList signal(device_, {0}, {1});
    const iree_hal_atomic_store_params_t params = {
        .value = static_cast<uint64_t>(value),
        .flags = IREE_HAL_ATOMIC_FLAG_RELEASE,
        .width = width,
    };
    iree_status_t status =
        iree_hal_queue_atomic_store(atomic_queue_, empty_wait, signal, buffer,
                                    kTargetOffset, params, /*barriers=*/NULL);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                            IREE_ASYNC_WAIT_FLAG_NONE);
    }
    return status;
  }

  iree_status_t SubmitAtomicCommandBuffer(
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_buffer_t* indirect_buffer) {
    iree_hal_buffer_binding_t binding = {
        .buffer = indirect_buffer,
        .offset = 0,
        .length = IREE_HAL_WHOLE_BUFFER,
    };
    const iree_hal_buffer_binding_table_t binding_table = {
        .count = indirect_buffer ? 1u : 0u,
        .bindings = indirect_buffer ? &binding : nullptr,
    };

    SemaphoreList empty_wait;
    SemaphoreList signal(device_, {0}, {1});
    iree_status_t status = iree_hal_queue_execute(
        atomic_queue_, empty_wait, signal, command_buffer, binding_table,
        IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                            IREE_ASYNC_WAIT_FLAG_NONE);
    }
    return status;
  }

  void RunResolvedMisalignmentFailureTest(iree_hal_atomic_width_t width) {
    AtomicTestConfiguration configuration;
    const iree_hal_atomic_operation_flags_t operation_flags =
        IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
        IREE_HAL_ATOMIC_OPERATION_FLAG_STORE |
        IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD;
    if (!SelectConfiguration(width, operation_flags,
                             IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_EQUAL,
                             &configuration)) {
      GTEST_SKIP() << "Device does not advertise the tested "
                      "queue/memory atomic operation set";
    }

    alignas(uint64_t) std::array<uint8_t, kBufferSize + 1> storage = {};
    void* misaligned_ptr = storage.data() + 1;
    const iree_device_size_t byte_count =
        iree_hal_atomic_width_byte_count(width);
    ASSERT_NE(0u, reinterpret_cast<uintptr_t>(misaligned_ptr) % byte_count);

    std::promise<void> released;
    std::future<void> released_future = released.get_future();
    Ref<iree_hal_pool_t> import_pool;
    Status import_pool_status(CreateAtomicTestHostImportPool(
        device_group_, configuration, import_pool.out()));
    if (import_pool_status.code() == StatusCode::kUnavailable) {
      GTEST_SKIP() << "Pool cannot import host storage for target validation";
    }
    IREE_ASSERT_OK(import_pool_status.release());
    iree_hal_external_storage_t external_buffer = {
        .storage_class = iree_hal_make_external_storage_class(
            IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
        .byte_length = kBufferSize,
    };
    external_buffer.payload.host_pointer = misaligned_ptr;
    external_buffer.host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK;
    external_buffer.maintenance_range =
        iree_hal_make_buffer_range(0, kBufferSize);
    external_buffer.release = NotifyBufferReleased;
    external_buffer.release_user_data = &released;

    Ref<iree_hal_buffer_t> buffer;
    Status import_status(iree_hal_pool_import_buffer(
        import_pool, &external_buffer,
        iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER), buffer.out()));
    if (!import_status.ok()) {
      iree_hal_external_storage_release(&external_buffer);
      GTEST_SKIP() << "Pool cannot import an explicitly unaligned host "
                      "allocation for target validation: "
                   << FormatSkipStatus(import_status);
    }

    enum class AtomicKind { kWait, kStore, kRmw };
    const AtomicKind kinds[] = {AtomicKind::kWait, AtomicKind::kStore,
                                AtomicKind::kRmw};
    const iree_hal_atomic_target_error_mode_t modes[] = {
        IREE_HAL_ATOMIC_TARGET_ERROR_MODE_DEFAULT,
        IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE,
    };
    for (iree_hal_atomic_target_error_mode_t mode : modes) {
      for (AtomicKind kind : kinds) {
        const bool indirect = recording_mode() == RecordingMode::kIndirect;
        const iree_host_size_t binding_capacity = indirect ? 1 : 0;
        Ref<iree_hal_command_buffer_t> command_buffer;
        IREE_ASSERT_OK(
            CreateAtomicCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
                                      binding_capacity, command_buffer.out()));
        IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
        const iree_hal_buffer_ref_t target_ref =
            indirect
                ? iree_hal_make_indirect_buffer_ref(
                      /*binding=*/0, /*offset=*/0, byte_count)
                : iree_hal_make_buffer_ref(buffer, /*offset=*/0, byte_count);
        switch (kind) {
          case AtomicKind::kWait:
            IREE_ASSERT_OK(iree_hal_command_buffer_atomic_wait(
                command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
                IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE, target_ref,
                iree_hal_atomic_wait_params_t{
                    .value = 0,
                    .mask = width == IREE_HAL_ATOMIC_WIDTH_32 ? UINT32_MAX
                                                              : UINT64_MAX,
                    .flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE,
                    .width = width,
                    .condition = IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
                    .target_error_mode = mode,
                }));
            break;
          case AtomicKind::kStore:
            IREE_ASSERT_OK(iree_hal_command_buffer_atomic_store(
                command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
                IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE, target_ref,
                iree_hal_atomic_store_params_t{
                    .value = 1,
                    .flags = IREE_HAL_ATOMIC_FLAG_RELEASE,
                    .width = width,
                    .target_error_mode = mode,
                }));
            break;
          case AtomicKind::kRmw:
            IREE_ASSERT_OK(iree_hal_command_buffer_atomic_rmw(
                command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
                IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE, target_ref,
                iree_hal_atomic_rmw_params_t{
                    .operand = 1,
                    .flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE |
                             IREE_HAL_ATOMIC_FLAG_RELEASE,
                    .width = width,
                    .operation = IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
                    .target_error_mode = mode,
                }));
            break;
        }
        IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

        const iree_hal_buffer_binding_t binding = {
            .buffer = indirect ? buffer.get() : nullptr,
            .offset = 0,
            .length = indirect ? IREE_HAL_WHOLE_BUFFER : 0,
        };
        const iree_hal_buffer_binding_table_t binding_table = {
            .count = indirect ? 1u : 0u,
            .bindings = indirect ? &binding : nullptr,
        };
        SemaphoreList empty_wait;
        SemaphoreList signal(device_, {0}, {1});
        iree_status_t status = iree_hal_queue_execute(
            atomic_queue_, empty_wait, signal, command_buffer, binding_table,
            IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
        if (iree_status_is_ok(status)) {
          status = iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE);
        }
        const StatusCode expected_status =
            mode == IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE
                ? StatusCode::kIncompatible
                : static_cast<StatusCode>(
                      this->GetParam().default_atomic_target_error_code);
        EXPECT_THAT(Status(std::move(status)), StatusIs(expected_status));
        EXPECT_TRUE(std::all_of(storage.begin(), storage.end(),
                                [](uint8_t byte) { return byte == 0; }));
      }
    }

    buffer.reset();
    released_future.wait();
  }

  template <typename ValueType>
  void RunReusableStoreAndRmwTest(
      iree_hal_atomic_width_t width, bool include_bitwise = true,
      iree_hal_atomic_flags_t scope_flags = IREE_HAL_ATOMIC_FLAG_NONE) {
    iree_hal_atomic_operation_flags_t operation_flags =
        IREE_HAL_ATOMIC_OPERATION_FLAG_STORE |
        IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD |
        IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_SUBTRACT;
    if (include_bitwise) {
      operation_flags |= IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_AND |
                         IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_OR |
                         IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_XOR;
    }
    AtomicTestConfiguration configuration;
    if (!SelectConfiguration(width, operation_flags,
                             IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_NONE,
                             &configuration, scope_flags)) {
      GTEST_SKIP() << "Device does not advertise the tested "
                      "queue/memory atomic operation set";
    }
    Ref<iree_hal_buffer_t> first_buffer;
    IREE_ASSERT_OK(AllocateAtomicBuffer(configuration, first_buffer.out()));
    Ref<iree_hal_buffer_t> second_buffer;
    IREE_ASSERT_OK(AllocateAtomicBuffer(configuration, second_buffer.out()));

    const bool indirect = recording_mode() == RecordingMode::kIndirect;
    const iree_host_size_t binding_capacity = indirect ? 1 : 0;
    Ref<iree_hal_command_buffer_t> command_buffer;
    IREE_ASSERT_OK(
        CreateAtomicCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
                                  binding_capacity, command_buffer.out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

    const iree_device_size_t target_length =
        iree_hal_atomic_width_byte_count(width);
    const iree_hal_buffer_ref_t target_ref =
        indirect ? iree_hal_make_indirect_buffer_ref(
                       /*binding=*/0, kTargetOffset, target_length)
                 : iree_hal_make_buffer_ref(first_buffer, kTargetOffset,
                                            target_length);
    const iree_hal_atomic_flags_t atomic_flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE |
                                                 IREE_HAL_ATOMIC_FLAG_RELEASE |
                                                 scope_flags;

    // Command recording order does not establish atomic execution dependencies.
    const iree_hal_atomic_store_params_t store_params = {
        .value = 10,
        .flags = atomic_flags,
        .width = width,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_atomic_store(
        command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
        IREE_HAL_EXECUTION_STAGE_ATOMIC, target_ref, store_params));

    iree_hal_atomic_rmw_params_t rmw_params = {
        .operand = 5,
        .flags = atomic_flags,
        .width = width,
        .operation = IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_atomic_rmw(
        command_buffer, IREE_HAL_EXECUTION_STAGE_ATOMIC,
        IREE_HAL_EXECUTION_STAGE_ATOMIC, target_ref, rmw_params));

    rmw_params.operand = 3;
    rmw_params.operation = IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT;
    IREE_ASSERT_OK(iree_hal_command_buffer_atomic_rmw(
        command_buffer, IREE_HAL_EXECUTION_STAGE_ATOMIC,
        include_bitwise ? IREE_HAL_EXECUTION_STAGE_ATOMIC
                        : IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
        target_ref, rmw_params));

    if (include_bitwise) {
      rmw_params.operand = 10;
      rmw_params.operation = IREE_HAL_ATOMIC_RMW_OPERATION_AND;
      IREE_ASSERT_OK(iree_hal_command_buffer_atomic_rmw(
          command_buffer, IREE_HAL_EXECUTION_STAGE_ATOMIC,
          IREE_HAL_EXECUTION_STAGE_ATOMIC, target_ref, rmw_params));

      rmw_params.operand = 3;
      rmw_params.operation = IREE_HAL_ATOMIC_RMW_OPERATION_OR;
      IREE_ASSERT_OK(iree_hal_command_buffer_atomic_rmw(
          command_buffer, IREE_HAL_EXECUTION_STAGE_ATOMIC,
          IREE_HAL_EXECUTION_STAGE_ATOMIC, target_ref, rmw_params));

      rmw_params.operand = 15;
      rmw_params.operation = IREE_HAL_ATOMIC_RMW_OPERATION_XOR;
      IREE_ASSERT_OK(iree_hal_command_buffer_atomic_rmw(
          command_buffer, IREE_HAL_EXECUTION_STAGE_ATOMIC,
          IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE, target_ref, rmw_params));
    }
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

    const ValueType expected = include_bitwise ? 4 : 12;
    IREE_ASSERT_OK(SubmitAtomicCommandBuffer(
        command_buffer, indirect ? first_buffer.get() : nullptr));
    EXPECT_EQ(ReadAtomicValue<ValueType>(first_buffer), expected);

    iree_hal_buffer_t* second_target =
        indirect ? second_buffer.get() : first_buffer.get();
    IREE_ASSERT_OK(QueueStoreAtomicValue(second_target, width,
                                         static_cast<ValueType>(99)));
    IREE_ASSERT_OK(SubmitAtomicCommandBuffer(
        command_buffer, indirect ? second_buffer.get() : nullptr));
    if (indirect) {
      EXPECT_EQ(ReadAtomicValue<ValueType>(second_buffer), expected);
    } else {
      EXPECT_EQ(ReadAtomicValue<ValueType>(first_buffer), expected);
    }
  }

  template <typename ValueType>
  void RunReusableWaitTest(
      iree_hal_atomic_width_t width,
      iree_hal_atomic_flags_t scope_flags = IREE_HAL_ATOMIC_FLAG_NONE) {
    const iree_hal_atomic_operation_flags_t operation_flags =
        IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
        IREE_HAL_ATOMIC_OPERATION_FLAG_STORE;
    AtomicTestConfiguration configuration;
    if (!SelectConfiguration(width, operation_flags,
                             IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL,
                             &configuration, scope_flags)) {
      GTEST_SKIP() << "Device does not advertise the tested "
                      "queue/memory atomic wait set";
    }

    Ref<iree_hal_buffer_t> first_buffer;
    IREE_ASSERT_OK(AllocateAtomicBuffer(configuration, first_buffer.out()));
    Ref<iree_hal_buffer_t> second_buffer;
    IREE_ASSERT_OK(AllocateAtomicBuffer(configuration, second_buffer.out()));

    const bool indirect = recording_mode() == RecordingMode::kIndirect;
    const iree_host_size_t binding_capacity = indirect ? 1 : 0;
    Ref<iree_hal_command_buffer_t> command_buffer;
    IREE_ASSERT_OK(
        CreateAtomicCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
                                  binding_capacity, command_buffer.out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

    const iree_device_size_t target_length =
        iree_hal_atomic_width_byte_count(width);
    const iree_hal_buffer_ref_t target_ref =
        indirect ? iree_hal_make_indirect_buffer_ref(
                       /*binding=*/0, kTargetOffset, target_length)
                 : iree_hal_make_buffer_ref(first_buffer, kTargetOffset,
                                            target_length);
    const iree_hal_atomic_flags_t atomic_flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE |
                                                 IREE_HAL_ATOMIC_FLAG_RELEASE |
                                                 scope_flags;

    // Command recording order does not establish atomic execution dependencies.
    const iree_hal_atomic_store_params_t store_params = {
        .value = 0x12,
        .flags = atomic_flags,
        .width = width,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_atomic_store(
        command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
        IREE_HAL_EXECUTION_STAGE_ATOMIC, target_ref, store_params));

    iree_hal_atomic_wait_params_t wait_params = {
        .value = 0x2,
        .mask = 0xF,
        .flags = atomic_flags,
        .width = width,
        .condition = IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_atomic_wait(
        command_buffer, IREE_HAL_EXECUTION_STAGE_ATOMIC,
        IREE_HAL_EXECUTION_STAGE_ATOMIC, target_ref, wait_params));

    wait_params.value = 0x3;
    wait_params.condition = IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL;
    IREE_ASSERT_OK(iree_hal_command_buffer_atomic_wait(
        command_buffer, IREE_HAL_EXECUTION_STAGE_ATOMIC,
        IREE_HAL_EXECUTION_STAGE_ATOMIC, target_ref, wait_params));

    wait_params.value = 0x10;
    wait_params.mask =
        width == IREE_HAL_ATOMIC_WIDTH_32 ? UINT32_MAX : UINT64_MAX;
    wait_params.condition =
        IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL;
    IREE_ASSERT_OK(iree_hal_command_buffer_atomic_wait(
        command_buffer, IREE_HAL_EXECUTION_STAGE_ATOMIC,
        IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE, target_ref, wait_params));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

    IREE_ASSERT_OK(SubmitAtomicCommandBuffer(
        command_buffer, indirect ? first_buffer.get() : nullptr));
    EXPECT_EQ(ReadAtomicValue<ValueType>(first_buffer),
              static_cast<ValueType>(0x12));

    iree_hal_buffer_t* second_target =
        indirect ? second_buffer.get() : first_buffer.get();
    IREE_ASSERT_OK(QueueStoreAtomicValue(second_target, width,
                                         static_cast<ValueType>(0x22)));
    IREE_ASSERT_OK(SubmitAtomicCommandBuffer(
        command_buffer, indirect ? second_buffer.get() : nullptr));
    if (indirect) {
      EXPECT_EQ(ReadAtomicValue<ValueType>(second_buffer),
                static_cast<ValueType>(0x12));
    } else {
      EXPECT_EQ(ReadAtomicValue<ValueType>(first_buffer),
                static_cast<ValueType>(0x12));
    }
  }

  Ref<iree_hal_pool_t> atomic_pool_;
  iree_hal_queue_t* atomic_queue_ = nullptr;
};

TEST_P(CommandBufferAtomicTest, ReusableStoreAndRmw32) {
  RunReusableStoreAndRmwTest<uint32_t>(IREE_HAL_ATOMIC_WIDTH_32);
}

TEST_P(CommandBufferAtomicTest, ReusableStoreAndRmw64) {
  RunReusableStoreAndRmwTest<uint64_t>(IREE_HAL_ATOMIC_WIDTH_64);
}

// PCIe control storage may support fetch-add without bitwise RMW operations.
TEST_P(CommandBufferAtomicTest, ReusableStoreAndAddSubtract32) {
  RunReusableStoreAndRmwTest<uint32_t>(IREE_HAL_ATOMIC_WIDTH_32, false);
}

TEST_P(CommandBufferAtomicTest, ReusableStoreAndAddSubtract64) {
  RunReusableStoreAndRmwTest<uint64_t>(IREE_HAL_ATOMIC_WIDTH_64, false);
}

TEST_P(CommandBufferAtomicTest, ReusableWait32) {
  RunReusableWaitTest<uint32_t>(IREE_HAL_ATOMIC_WIDTH_32);
}

TEST_P(CommandBufferAtomicTest, ReusableWait64) {
  RunReusableWaitTest<uint64_t>(IREE_HAL_ATOMIC_WIDTH_64);
}

TEST_P(CommandBufferAtomicTest, SystemReusableStoreAndAddSubtract32) {
  RunReusableStoreAndRmwTest<uint32_t>(IREE_HAL_ATOMIC_WIDTH_32, false,
                                       IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE);
}

TEST_P(CommandBufferAtomicTest, SystemReusableStoreAndAddSubtract64) {
  RunReusableStoreAndRmwTest<uint64_t>(IREE_HAL_ATOMIC_WIDTH_64, false,
                                       IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE);
}

TEST_P(CommandBufferAtomicTest, SystemReusableWait32) {
  RunReusableWaitTest<uint32_t>(IREE_HAL_ATOMIC_WIDTH_32,
                                IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE);
}

TEST_P(CommandBufferAtomicTest, SystemReusableWait64) {
  RunReusableWaitTest<uint64_t>(IREE_HAL_ATOMIC_WIDTH_64,
                                IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE);
}

TEST_P(CommandBufferAtomicTest, ResolvedMisalignmentFails64) {
  RunResolvedMisalignmentFailureTest(IREE_HAL_ATOMIC_WIDTH_64);
}

CTS_REGISTER_COMMAND_BUFFER_TEST_SUITE(CommandBufferAtomicTest);

}  // namespace iree::hal::cts
