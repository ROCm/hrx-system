// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <cstdint>
#include <future>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/atomic_memory.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_test_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::hal::cts::Ref;
using namespace test;

class ReleaseLatch {
 public:
  explicit ReleaseLatch(int release_count)
      : future_(promise_.get_future()), remaining_(release_count) {}

  static void Notify(void* user_data, iree_hal_buffer_t* buffer) {
    (void)buffer;
    ReleaseLatch* latch = static_cast<ReleaseLatch*>(user_data);
    const int previous = latch->remaining_.fetch_sub(1);
    if (previous == 1) latch->promise_.set_value();
  }

  iree_hal_buffer_release_callback_t callback() {
    return {/*.fn=*/Notify, /*.user_data=*/this};
  }

  int remaining() const { return remaining_.load(); }

  void Wait() { future_.wait(); }

 private:
  // Notification fulfilled after the final imported buffer is released.
  std::promise<void> promise_;
  // Wait handle paired with |promise_|.
  std::future<void> future_;
  // Number of imported buffers still alive.
  std::atomic<int> remaining_;
};

class HostQueueAtomicTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_with_defaults(
        &libhsa_, &topology_));
    if (topology_.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  static iree_status_t ImportHostAtomicBuffer(
      TestLogicalDevice* test_device, iree_hal_queue_affinity_t queue_affinity,
      void* host_pointer, iree_device_size_t byte_length,
      iree_hal_memory_access_t extra_access,
      iree_device_size_t minimum_alignment,
      iree_hal_buffer_release_callback_t release_callback,
      iree_hal_buffer_t** out_buffer) {
    iree_hal_external_buffer_t external_buffer = {};
    external_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION;
    external_buffer.size = byte_length;
    external_buffer.handle.host_allocation.ptr = host_pointer;

    iree_hal_buffer_params_t params = {};
    params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL | extra_access;
    params.usage =
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
    params.queue_affinity = queue_affinity;
    params.min_alignment = minimum_alignment;
    return iree_hal_allocator_import_buffer(test_device->allocator(), params,
                                            &external_buffer, release_callback,
                                            out_buffer);
  }

  static iree_status_t CreateReusableAtomicProgram(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      iree_hal_buffer_t* static_buffer,
      iree_hal_command_buffer_t** out_command_buffer) {
    Ref<iree_hal_command_buffer_t> command_buffer;
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
        device, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_ATOMIC, queue_affinity,
        /*binding_capacity=*/1, command_buffer.out()));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer));

    const iree_hal_atomic_flags_t atomic_flags =
        IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE |
        IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE;
    const iree_hal_buffer_ref_t static_target =
        iree_hal_make_buffer_ref(static_buffer, /*offset=*/sizeof(uint32_t),
                                 /*length=*/sizeof(uint32_t));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_atomic_store(
        command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
        IREE_HAL_EXECUTION_STAGE_ATOMIC, static_target,
        (iree_hal_atomic_store_params_t){
            /*.value=*/10,
            /*.flags=*/atomic_flags,
            /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
        }));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_atomic_rmw(
        command_buffer, IREE_HAL_EXECUTION_STAGE_ATOMIC,
        IREE_HAL_EXECUTION_STAGE_ATOMIC, static_target,
        (iree_hal_atomic_rmw_params_t){
            /*.operand=*/5,
            /*.flags=*/atomic_flags,
            /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
            /*.operation=*/IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
        }));

    const iree_hal_buffer_ref_t dynamic_wait_target =
        iree_hal_make_indirect_buffer_ref(
            /*binding=*/0, /*offset=*/0, /*length=*/sizeof(uint32_t));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_atomic_wait(
        command_buffer, IREE_HAL_EXECUTION_STAGE_ATOMIC,
        IREE_HAL_EXECUTION_STAGE_ATOMIC, dynamic_wait_target,
        (iree_hal_atomic_wait_params_t){
            /*.value=*/5,
            /*.mask=*/UINT32_MAX,
            /*.flags=*/atomic_flags,
            /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
            /*.condition=*/IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
        }));
    const iree_hal_buffer_ref_t dynamic_target =
        iree_hal_make_indirect_buffer_ref(
            /*binding=*/0, /*offset=*/sizeof(uint32_t),
            /*length=*/sizeof(uint32_t));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_atomic_store(
        command_buffer, IREE_HAL_EXECUTION_STAGE_ATOMIC,
        IREE_HAL_EXECUTION_STAGE_ATOMIC, dynamic_target,
        (iree_hal_atomic_store_params_t){
            /*.value=*/20,
            /*.flags=*/atomic_flags,
            /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
        }));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_atomic_rmw(
        command_buffer, IREE_HAL_EXECUTION_STAGE_ATOMIC,
        IREE_HAL_EXECUTION_STAGE_HOST, dynamic_target,
        (iree_hal_atomic_rmw_params_t){
            /*.operand=*/3,
            /*.flags=*/atomic_flags,
            /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
            /*.operation=*/IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
        }));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_end(command_buffer));
    *out_command_buffer = command_buffer.release();
    return iree_ok_status();
  }

  static iree_status_t CreateReusableDynamicStoreProgram(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      iree_hal_command_buffer_t** out_command_buffer) {
    Ref<iree_hal_command_buffer_t> command_buffer;
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
        device, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_ATOMIC, queue_affinity,
        /*binding_capacity=*/1, command_buffer.out()));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_atomic_store(
        command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
        IREE_HAL_EXECUTION_STAGE_HOST,
        iree_hal_make_indirect_buffer_ref(
            /*binding=*/0, /*offset=*/0, /*length=*/sizeof(uint32_t)),
        (iree_hal_atomic_store_params_t){
            /*.value=*/77,
            /*.flags=*/IREE_HAL_ATOMIC_FLAG_RELEASE |
                IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
            /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
        }));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_end(command_buffer));
    *out_command_buffer = command_buffer.release();
    return iree_ok_status();
  }

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t HostQueueAtomicTest::host_allocator_;
iree_hal_amdgpu_libhsa_t HostQueueAtomicTest::libhsa_;
iree_hal_amdgpu_topology_t HostQueueAtomicTest::topology_;

TEST_F(HostQueueAtomicTest, ReusableProgramRetainsAndRebindsResources) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AQL;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  iree_hal_queue_affinity_t queue_affinity = 0;
  IREE_ASSERT_OK(QueueAffinityForPhysicalDevice(
      test_device, /*physical_device_ordinal=*/0, &queue_affinity));

  alignas(64) std::array<uint32_t, 16> static_storage = {};
  alignas(64) std::array<uint32_t, 16> first_storage = {};
  alignas(64) std::array<uint32_t, 16> second_storage = {};
  first_storage[0] = 5;
  second_storage[0] = 5;
  ReleaseLatch release_latch(/*release_count=*/3);

  Ref<iree_hal_buffer_t> static_buffer;
  IREE_ASSERT_OK(ImportHostAtomicBuffer(
      &test_device, queue_affinity, static_storage.data(),
      sizeof(static_storage), IREE_HAL_MEMORY_ACCESS_NONE,
      /*minimum_alignment=*/64, release_latch.callback(), static_buffer.out()));
  Ref<iree_hal_buffer_t> first_buffer;
  IREE_ASSERT_OK(ImportHostAtomicBuffer(
      &test_device, queue_affinity, first_storage.data(), sizeof(first_storage),
      IREE_HAL_MEMORY_ACCESS_NONE, /*minimum_alignment=*/64,
      release_latch.callback(), first_buffer.out()));
  Ref<iree_hal_buffer_t> second_buffer;
  IREE_ASSERT_OK(ImportHostAtomicBuffer(
      &test_device, queue_affinity, second_storage.data(),
      sizeof(second_storage), IREE_HAL_MEMORY_ACCESS_NONE,
      /*minimum_alignment=*/64, release_latch.callback(), second_buffer.out()));
  EXPECT_EQ(iree_hal_amdgpu_buffer_atomic_memory_cells(static_buffer),
            IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAGS_ALL);
  EXPECT_EQ(iree_hal_amdgpu_buffer_atomic_memory_cells(first_buffer),
            IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAGS_ALL);
  EXPECT_EQ(iree_hal_amdgpu_buffer_atomic_memory_cells(second_buffer),
            IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAGS_ALL);

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(CreateReusableAtomicProgram(test_device.base_device(),
                                             queue_affinity, static_buffer,
                                             command_buffer.out()));
  Ref<iree_hal_semaphore_t> gate;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), gate.out()));
  Ref<iree_hal_semaphore_t> completion;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), completion.out()));

  iree_hal_semaphore_t* gate_semaphore = gate.get();
  uint64_t gate_value = 1;
  const iree_hal_semaphore_list_t first_wait_list = {
      /*.count=*/1,
      /*.semaphores=*/&gate_semaphore,
      /*.payload_values=*/&gate_value,
  };
  iree_hal_semaphore_t* completion_semaphore = completion.get();
  uint64_t first_completion_value = 1;
  const iree_hal_semaphore_list_t first_signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&completion_semaphore,
      /*.payload_values=*/&first_completion_value,
  };
  const iree_hal_buffer_binding_t first_binding = {
      /*.buffer=*/first_buffer.get(),
      /*.offset=*/0,
      /*.length=*/IREE_HAL_WHOLE_BUFFER,
  };
  const iree_hal_buffer_binding_table_t first_binding_table = {
      /*.count=*/1,
      /*.bindings=*/&first_binding,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), queue_affinity, first_wait_list,
      first_signal_list, command_buffer, first_binding_table,
      IREE_HAL_EXECUTE_FLAG_NONE));

  const iree_hal_semaphore_list_t second_wait_list = first_signal_list;
  uint64_t second_completion_value = 2;
  const iree_hal_semaphore_list_t second_signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&completion_semaphore,
      /*.payload_values=*/&second_completion_value,
  };
  const iree_hal_buffer_binding_t second_binding = {
      /*.buffer=*/second_buffer.get(),
      /*.offset=*/0,
      /*.length=*/IREE_HAL_WHOLE_BUFFER,
  };
  const iree_hal_buffer_binding_table_t second_binding_table = {
      /*.count=*/1,
      /*.bindings=*/&second_binding,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), queue_affinity, second_wait_list,
      second_signal_list, command_buffer, second_binding_table,
      IREE_HAL_EXECUTE_FLAG_NONE));

  command_buffer.reset();
  static_buffer.reset();
  first_buffer.reset();
  second_buffer.reset();
  EXPECT_EQ(release_latch.remaining(), 3);

  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, gate_value,
                                           /*frontier=*/nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(completion, second_completion_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  release_latch.Wait();

  EXPECT_EQ(static_storage[1], 15u);
  EXPECT_EQ(first_storage[0], 5u);
  EXPECT_EQ(first_storage[1], 23u);
  EXPECT_EQ(second_storage[0], 5u);
  EXPECT_EQ(second_storage[1], 23u);
}

TEST_F(HostQueueAtomicTest, DeferredResolvedMisalignmentFailsAndQueueRecovers) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AQL;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  iree_hal_queue_affinity_t queue_affinity = 0;
  IREE_ASSERT_OK(QueueAffinityForPhysicalDevice(
      test_device, /*physical_device_ordinal=*/0, &queue_affinity));
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(CreateReusableDynamicStoreProgram(
      test_device.base_device(), queue_affinity, command_buffer.out()));

  alignas(64) std::array<uint8_t, 65> misaligned_storage = {};
  ReleaseLatch misaligned_release_latch(/*release_count=*/1);
  Ref<iree_hal_buffer_t> misaligned_buffer;
  IREE_ASSERT_OK(ImportHostAtomicBuffer(
      &test_device, queue_affinity, misaligned_storage.data() + 1,
      misaligned_storage.size() - 1, IREE_HAL_MEMORY_ACCESS_UNALIGNED,
      /*minimum_alignment=*/1, misaligned_release_latch.callback(),
      misaligned_buffer.out()));
  ASSERT_NE(reinterpret_cast<uintptr_t>(misaligned_storage.data() + 1) %
                alignof(uint32_t),
            0u);

  Ref<iree_hal_semaphore_t> gate;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), gate.out()));
  iree_hal_semaphore_t* gate_semaphore = gate.get();
  uint64_t gate_value = 1;
  const iree_hal_semaphore_list_t wait_list = {
      /*.count=*/1,
      /*.semaphores=*/&gate_semaphore,
      /*.payload_values=*/&gate_value,
  };
  Ref<iree_hal_semaphore_t> failed_completion;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), failed_completion.out()));
  iree_hal_semaphore_t* failed_completion_semaphore = failed_completion.get();
  uint64_t failed_completion_value = 1;
  const iree_hal_semaphore_list_t failed_signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&failed_completion_semaphore,
      /*.payload_values=*/&failed_completion_value,
  };
  const iree_hal_buffer_binding_t misaligned_binding = {
      /*.buffer=*/misaligned_buffer.get(),
      /*.offset=*/0,
      /*.length=*/IREE_HAL_WHOLE_BUFFER,
  };
  const iree_hal_buffer_binding_table_t misaligned_binding_table = {
      /*.count=*/1,
      /*.bindings=*/&misaligned_binding,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), queue_affinity, wait_list, failed_signal_list,
      command_buffer, misaligned_binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
  misaligned_buffer.reset();
  EXPECT_EQ(misaligned_release_latch.remaining(), 1);
  IREE_ASSERT_OK(
      iree_hal_semaphore_signal(gate, gate_value, /*frontier=*/nullptr));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_semaphore_wait(failed_completion, failed_completion_value,
                              iree_infinite_timeout(),
                              IREE_ASYNC_WAIT_FLAG_NONE));
  misaligned_release_latch.Wait();

  alignas(64) std::array<uint32_t, 16> valid_storage = {};
  ReleaseLatch valid_release_latch(/*release_count=*/1);
  Ref<iree_hal_buffer_t> valid_buffer;
  IREE_ASSERT_OK(ImportHostAtomicBuffer(
      &test_device, queue_affinity, valid_storage.data(), sizeof(valid_storage),
      IREE_HAL_MEMORY_ACCESS_NONE,
      /*minimum_alignment=*/64, valid_release_latch.callback(),
      valid_buffer.out()));
  Ref<iree_hal_semaphore_t> valid_completion;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), valid_completion.out()));
  iree_hal_semaphore_t* valid_completion_semaphore = valid_completion.get();
  uint64_t valid_completion_value = 1;
  const iree_hal_semaphore_list_t valid_signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&valid_completion_semaphore,
      /*.payload_values=*/&valid_completion_value,
  };
  const iree_hal_buffer_binding_t valid_binding = {
      /*.buffer=*/valid_buffer.get(),
      /*.offset=*/0,
      /*.length=*/IREE_HAL_WHOLE_BUFFER,
  };
  const iree_hal_buffer_binding_table_t valid_binding_table = {
      /*.count=*/1,
      /*.bindings=*/&valid_binding,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), queue_affinity,
      iree_hal_semaphore_list_empty(), valid_signal_list, command_buffer,
      valid_binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      valid_completion, valid_completion_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(valid_storage[0], 77u);

  command_buffer.reset();
  valid_buffer.reset();
  valid_release_latch.Wait();
}

}  // namespace
}  // namespace iree::hal::amdgpu
