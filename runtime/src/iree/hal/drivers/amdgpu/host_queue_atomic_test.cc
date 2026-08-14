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
#include "iree/hal/drivers/amdgpu/pm4_command_buffer.h"
#include "iree/hal/drivers/amdgpu/queue_affinity.h"
#include "iree/hal/drivers/amdgpu/util/pm4_atomic.h"
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

class HostQueueAtomicTest
    : public ::testing::TestWithParam<iree_hal_amdgpu_command_buffer_mode_t> {
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

  static iree_status_t QueueAffinityForPhysicalQueue(
      const TestLogicalDevice& test_device,
      iree_host_size_t physical_device_ordinal,
      iree_host_size_t physical_queue_ordinal,
      iree_hal_queue_affinity_t* out_queue_affinity) {
    iree_hal_amdgpu_logical_device_t* logical_device =
        test_device.logical_device();
    const iree_hal_amdgpu_queue_affinity_domain_t domain = {
        /*.supported_affinity=*/logical_device->queue_affinity_mask,
        /*.physical_device_count=*/logical_device->physical_device_count,
        /*.queue_count_per_physical_device=*/
        logical_device->system->topology.gpu_agent_queue_count,
    };
    return iree_hal_amdgpu_queue_affinity_for_physical_queue(
        domain, physical_device_ordinal, physical_queue_ordinal,
        out_queue_affinity);
  }

  static iree_status_t CreateReusableAtomicProgram(
      iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
      iree_hal_buffer_t* static_buffer,
      iree_hal_command_buffer_t** out_command_buffer) {
    Ref<iree_hal_command_buffer_t> command_buffer;
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
        device,
        IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT |
            IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA,
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

TEST_F(HostQueueAtomicTest, AutoCommandBufferModeSelectsPm4) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AUTO;
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  if (!iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_dispatch_command_buffers(
          test_device.logical_device()
              ->physical_devices[0]
              ->vendor_packet_capabilities)) {
    GTEST_SKIP() << "PM4 command buffers are not supported on this device";
  }

  iree_hal_queue_affinity_t queue_affinity = 0;
  IREE_ASSERT_OK(QueueAffinityForPhysicalDevice(
      test_device, /*physical_device_ordinal=*/0, &queue_affinity));
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_ATOMIC, queue_affinity,
      /*binding_capacity=*/0, command_buffer.out()));
  EXPECT_TRUE(iree_hal_amdgpu_pm4_command_buffer_isa(command_buffer));
}

TEST_F(HostQueueAtomicTest,
       DirectQueueSupportsWidthsConditionsAndRmwOperations) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  iree_hal_queue_affinity_t queue_affinity = 0;
  IREE_ASSERT_OK(QueueAffinityForPhysicalQueue(
      test_device, /*physical_device_ordinal=*/0,
      /*physical_queue_ordinal=*/0, &queue_affinity));
  alignas(64) std::array<uint64_t, 8> storage = {};
  ReleaseLatch release_latch(/*release_count=*/1);
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(ImportHostAtomicBuffer(
      &test_device, queue_affinity, storage.data(), sizeof(storage),
      IREE_HAL_MEMORY_ACCESS_NONE, /*minimum_alignment=*/64,
      release_latch.callback(), buffer.out()));

  Ref<iree_hal_semaphore_t> completion;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), completion.out()));
  iree_hal_semaphore_t* completion_semaphore = completion.get();
  uint64_t completion_value = 0;
  auto signal_and_wait = [&](auto submit) -> iree_status_t {
    ++completion_value;
    const iree_hal_semaphore_list_t signal_list = {
        /*.count=*/1,
        /*.semaphores=*/&completion_semaphore,
        /*.payload_values=*/&completion_value,
    };
    IREE_RETURN_IF_ERROR(submit(signal_list));
    return iree_hal_semaphore_wait(completion, completion_value,
                                   iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE);
  };
  const iree_hal_atomic_flags_t atomic_flags =
      IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE |
      IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE;
  auto run_sequence = [&](iree_device_size_t target_offset,
                          iree_hal_atomic_width_t width) -> iree_status_t {
    const uint64_t wait_mask =
        width == IREE_HAL_ATOMIC_WIDTH_32 ? UINT32_MAX : UINT64_MAX;
    IREE_RETURN_IF_ERROR(
        signal_and_wait([&](iree_hal_semaphore_list_t signal_list) {
          return iree_hal_device_queue_atomic_store(
              test_device.base_device(), queue_affinity,
              iree_hal_semaphore_list_empty(), signal_list, buffer,
              target_offset,
              (iree_hal_atomic_store_params_t){
                  /*.value=*/10,
                  /*.flags=*/atomic_flags,
                  /*.width=*/width,
              });
        }));
    IREE_RETURN_IF_ERROR(
        signal_and_wait([&](iree_hal_semaphore_list_t signal_list) {
          return iree_hal_device_queue_atomic_rmw(
              test_device.base_device(), queue_affinity,
              iree_hal_semaphore_list_empty(), signal_list, buffer,
              target_offset,
              (iree_hal_atomic_rmw_params_t){
                  /*.operand=*/5,
                  /*.flags=*/atomic_flags,
                  /*.width=*/width,
                  /*.operation=*/IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
              });
        }));

    const iree_hal_atomic_wait_condition_t wait_conditions[] = {
        IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
        IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL,
        IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL,
    };
    const uint64_t wait_values[] = {15, 14, 15};
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(wait_conditions); ++i) {
      IREE_RETURN_IF_ERROR(
          signal_and_wait([&](iree_hal_semaphore_list_t signal_list) {
            return iree_hal_device_queue_atomic_wait(
                test_device.base_device(), queue_affinity,
                iree_hal_semaphore_list_empty(), signal_list, buffer,
                target_offset,
                (iree_hal_atomic_wait_params_t){
                    /*.value=*/wait_values[i],
                    /*.mask=*/wait_mask,
                    /*.flags=*/atomic_flags,
                    /*.width=*/width,
                    /*.condition=*/wait_conditions[i],
                });
          }));
    }

    const iree_hal_atomic_rmw_operation_t rmw_operations[] = {
        IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT,
        IREE_HAL_ATOMIC_RMW_OPERATION_AND,
        IREE_HAL_ATOMIC_RMW_OPERATION_OR,
        IREE_HAL_ATOMIC_RMW_OPERATION_XOR,
    };
    const uint64_t rmw_operands[] = {2, 0xFu, 0x10u, 0x1u};
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(rmw_operations); ++i) {
      IREE_RETURN_IF_ERROR(
          signal_and_wait([&](iree_hal_semaphore_list_t signal_list) {
            return iree_hal_device_queue_atomic_rmw(
                test_device.base_device(), queue_affinity,
                iree_hal_semaphore_list_empty(), signal_list, buffer,
                target_offset,
                (iree_hal_atomic_rmw_params_t){
                    /*.operand=*/rmw_operands[i],
                    /*.flags=*/atomic_flags,
                    /*.width=*/width,
                    /*.operation=*/rmw_operations[i],
                });
          }));
    }
    return iree_ok_status();
  };

  IREE_ASSERT_OK(run_sequence(/*target_offset=*/0, IREE_HAL_ATOMIC_WIDTH_32));
  IREE_ASSERT_OK(run_sequence(/*target_offset=*/sizeof(uint64_t),
                              IREE_HAL_ATOMIC_WIDTH_64));
  EXPECT_EQ(static_cast<uint32_t>(storage[0]), 28u);
  EXPECT_EQ(storage[1], 28u);

  buffer.reset();
  release_latch.Wait();
}

TEST_F(HostQueueAtomicTest, DirectWaitBeforeStoreOnIndependentQueue) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  if (test_device.logical_device()->system->topology.gpu_agent_queue_count <
      2) {
    GTEST_SKIP() << "test requires two physical queues on one GPU";
  }

  iree_hal_queue_affinity_t wait_queue_affinity = 0;
  IREE_ASSERT_OK(QueueAffinityForPhysicalQueue(
      test_device, /*physical_device_ordinal=*/0,
      /*physical_queue_ordinal=*/0, &wait_queue_affinity));
  iree_hal_queue_affinity_t store_queue_affinity = 0;
  IREE_ASSERT_OK(QueueAffinityForPhysicalQueue(
      test_device, /*physical_device_ordinal=*/0,
      /*physical_queue_ordinal=*/1, &store_queue_affinity));
  ASSERT_NE(wait_queue_affinity, store_queue_affinity);

  alignas(64) std::array<uint32_t, 16> storage = {};
  ReleaseLatch release_latch(/*release_count=*/1);
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(ImportHostAtomicBuffer(
      &test_device, wait_queue_affinity | store_queue_affinity, storage.data(),
      sizeof(storage), IREE_HAL_MEMORY_ACCESS_NONE,
      /*minimum_alignment=*/64, release_latch.callback(), buffer.out()));

  Ref<iree_hal_semaphore_t> wait_completion;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), wait_completion.out()));
  iree_hal_semaphore_t* wait_completion_semaphore = wait_completion.get();
  uint64_t wait_completion_value = 1;
  const iree_hal_semaphore_list_t wait_signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&wait_completion_semaphore,
      /*.payload_values=*/&wait_completion_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_atomic_wait(
      test_device.base_device(), wait_queue_affinity,
      iree_hal_semaphore_list_empty(), wait_signal_list, buffer,
      /*target_offset=*/0,
      (iree_hal_atomic_wait_params_t){
          /*.value=*/1,
          /*.mask=*/UINT32_MAX,
          /*.flags=*/IREE_HAL_ATOMIC_FLAG_ACQUIRE |
              IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
          /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
          /*.condition=*/IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
      }));

  Ref<iree_hal_semaphore_t> store_completion;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), store_completion.out()));
  iree_hal_semaphore_t* store_completion_semaphore = store_completion.get();
  uint64_t store_completion_value = 1;
  const iree_hal_semaphore_list_t store_signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&store_completion_semaphore,
      /*.payload_values=*/&store_completion_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_atomic_store(
      test_device.base_device(), store_queue_affinity,
      iree_hal_semaphore_list_empty(), store_signal_list, buffer,
      /*target_offset=*/0,
      (iree_hal_atomic_store_params_t){
          /*.value=*/1,
          /*.flags=*/IREE_HAL_ATOMIC_FLAG_RELEASE |
              IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
          /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
      }));
  buffer.reset();

  IREE_ASSERT_OK(iree_hal_semaphore_wait(wait_completion, wait_completion_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      store_completion, store_completion_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));
  release_latch.Wait();
  EXPECT_EQ(storage[0], 1u);
}

TEST_F(HostQueueAtomicTest, DeferredDirectOperationsRetainTarget) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  iree_hal_queue_affinity_t queue_affinity = 0;
  IREE_ASSERT_OK(QueueAffinityForPhysicalQueue(
      test_device, /*physical_device_ordinal=*/0,
      /*physical_queue_ordinal=*/0, &queue_affinity));
  alignas(64) std::array<uint32_t, 16> storage = {};
  storage[0] = 1;
  storage[2] = 2;
  ReleaseLatch release_latch(/*release_count=*/1);
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(ImportHostAtomicBuffer(
      &test_device, queue_affinity, storage.data(), sizeof(storage),
      IREE_HAL_MEMORY_ACCESS_NONE, /*minimum_alignment=*/64,
      release_latch.callback(), buffer.out()));

  Ref<iree_hal_semaphore_t> gate;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), gate.out()));
  iree_hal_semaphore_t* gate_semaphore = gate.get();
  uint64_t gate_value = 1;
  const iree_hal_semaphore_list_t gate_wait_list = {
      /*.count=*/1,
      /*.semaphores=*/&gate_semaphore,
      /*.payload_values=*/&gate_value,
  };
  std::array<Ref<iree_hal_semaphore_t>, 3> completions;
  std::array<iree_hal_semaphore_t*, 3> completion_semaphores = {};
  std::array<uint64_t, 3> completion_values = {1, 1, 1};
  for (iree_host_size_t i = 0; i < completions.size(); ++i) {
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), completions[i].out()));
    completion_semaphores[i] = completions[i].get();
  }
  const iree_hal_atomic_flags_t atomic_flags =
      IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE |
      IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE;

  const iree_hal_semaphore_list_t wait_signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&completion_semaphores[0],
      /*.payload_values=*/&completion_values[0],
  };
  IREE_ASSERT_OK(iree_hal_device_queue_atomic_wait(
      test_device.base_device(), queue_affinity, gate_wait_list,
      wait_signal_list, buffer, /*target_offset=*/0,
      (iree_hal_atomic_wait_params_t){
          /*.value=*/1,
          /*.mask=*/UINT32_MAX,
          /*.flags=*/atomic_flags,
          /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
          /*.condition=*/IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
      }));
  const iree_hal_semaphore_list_t store_signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&completion_semaphores[1],
      /*.payload_values=*/&completion_values[1],
  };
  IREE_ASSERT_OK(iree_hal_device_queue_atomic_store(
      test_device.base_device(), queue_affinity, gate_wait_list,
      store_signal_list, buffer, /*target_offset=*/sizeof(uint32_t),
      (iree_hal_atomic_store_params_t){
          /*.value=*/7,
          /*.flags=*/atomic_flags,
          /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
      }));
  const iree_hal_semaphore_list_t rmw_signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&completion_semaphores[2],
      /*.payload_values=*/&completion_values[2],
  };
  IREE_ASSERT_OK(iree_hal_device_queue_atomic_rmw(
      test_device.base_device(), queue_affinity, gate_wait_list,
      rmw_signal_list, buffer, /*target_offset=*/2 * sizeof(uint32_t),
      (iree_hal_atomic_rmw_params_t){
          /*.operand=*/3,
          /*.flags=*/atomic_flags,
          /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
          /*.operation=*/IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
      }));

  buffer.reset();
  EXPECT_EQ(release_latch.remaining(), 1);
  IREE_ASSERT_OK(
      iree_hal_semaphore_signal(gate, gate_value, /*frontier=*/nullptr));
  for (iree_host_size_t i = 0; i < completions.size(); ++i) {
    IREE_ASSERT_OK(iree_hal_semaphore_wait(completions[i], completion_values[i],
                                           iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
  }
  release_latch.Wait();
  EXPECT_EQ(storage[0], 1u);
  EXPECT_EQ(storage[1], 7u);
  EXPECT_EQ(storage[2], 5u);
}

TEST_F(HostQueueAtomicTest, DeferredDirectMisalignmentFailsAndQueueRecovers) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  iree_hal_queue_affinity_t queue_affinity = 0;
  IREE_ASSERT_OK(QueueAffinityForPhysicalQueue(
      test_device, /*physical_device_ordinal=*/0,
      /*physical_queue_ordinal=*/0, &queue_affinity));
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
  IREE_ASSERT_OK(iree_hal_device_queue_atomic_store(
      test_device.base_device(), queue_affinity, wait_list, failed_signal_list,
      misaligned_buffer, /*target_offset=*/0,
      (iree_hal_atomic_store_params_t){
          /*.value=*/1,
          /*.flags=*/IREE_HAL_ATOMIC_FLAG_RELEASE |
              IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
          /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
      }));
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
      IREE_HAL_MEMORY_ACCESS_NONE, /*minimum_alignment=*/64,
      valid_release_latch.callback(), valid_buffer.out()));
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
  IREE_ASSERT_OK(iree_hal_device_queue_atomic_store(
      test_device.base_device(), queue_affinity,
      iree_hal_semaphore_list_empty(), valid_signal_list, valid_buffer,
      /*target_offset=*/0,
      (iree_hal_atomic_store_params_t){
          /*.value=*/77,
          /*.flags=*/IREE_HAL_ATOMIC_FLAG_RELEASE |
              IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
          /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
      }));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      valid_completion, valid_completion_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(valid_storage[0], 77u);

  valid_buffer.reset();
  valid_release_latch.Wait();
}

TEST_P(HostQueueAtomicTest, ReusableProgramRetainsAndRebindsResources) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = GetParam();
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  if (GetParam() == IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4 &&
      !iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_dispatch_command_buffers(
          test_device.logical_device()
              ->physical_devices[0]
              ->vendor_packet_capabilities)) {
    GTEST_SKIP() << "PM4 command buffers are not supported on this device";
  }

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
  if (GetParam() == IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4) {
    uint32_t operation_count = 0;
    const iree_hal_profile_command_operation_record_t* operations =
        iree_hal_amdgpu_pm4_command_buffer_profile_operations(command_buffer,
                                                              &operation_count);
    ASSERT_NE(operations, nullptr);
    ASSERT_EQ(operation_count, 5u);
    EXPECT_EQ(operations[0].type,
              IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_ATOMIC_STORE);
    EXPECT_EQ(operations[0].target_offset, sizeof(uint32_t));
    EXPECT_EQ(operations[0].length, sizeof(uint32_t));
    EXPECT_TRUE(iree_any_bit_set(
        operations[0].flags,
        IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_STATIC_BINDINGS));
    EXPECT_EQ(operations[1].type,
              IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_ATOMIC_RMW);
    EXPECT_EQ(operations[2].type,
              IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_ATOMIC_WAIT);
    EXPECT_EQ(operations[2].target_offset, 0u);
    EXPECT_EQ(operations[2].target_ordinal, 0u);
    EXPECT_TRUE(iree_any_bit_set(
        operations[2].flags,
        IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_DYNAMIC_BINDINGS));
    EXPECT_EQ(operations[3].type,
              IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_ATOMIC_STORE);
    EXPECT_EQ(operations[4].type,
              IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_ATOMIC_RMW);
    const iree_hal_amdgpu_pm4_command_buffer_publish_stats_t* publish_stats =
        iree_hal_amdgpu_pm4_command_buffer_publish_stats(command_buffer);
    ASSERT_NE(publish_stats, nullptr);
    const uint64_t native_packet_dword_count =
        2u * IREE_HAL_AMDGPU_PM4_ATOMIC_MEM_DWORD_COUNT +
        IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT32_DWORD_COUNT;
    EXPECT_GE(publish_stats->atomic_dwords, native_packet_dword_count);
    EXPECT_LE(publish_stats->atomic_dwords,
              native_packet_dword_count + 2u * 3u);
    EXPECT_GT(publish_stats->dispatch_dwords, 0u);
  }
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

TEST_P(HostQueueAtomicTest, DeferredResolvedMisalignmentFailsAndQueueRecovers) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = GetParam();
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  if (GetParam() == IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4 &&
      !iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_dispatch_command_buffers(
          test_device.logical_device()
              ->physical_devices[0]
              ->vendor_packet_capabilities)) {
    GTEST_SKIP() << "PM4 command buffers are not supported on this device";
  }

  iree_hal_queue_affinity_t queue_affinity = 0;
  IREE_ASSERT_OK(QueueAffinityForPhysicalDevice(
      test_device, /*physical_device_ordinal=*/0, &queue_affinity));
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(CreateReusableDynamicStoreProgram(
      test_device.base_device(), queue_affinity, command_buffer.out()));
  if (GetParam() == IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4) {
    const iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t* fixup_plan =
        iree_hal_amdgpu_pm4_command_buffer_fixup_plan(command_buffer);
    ASSERT_NE(fixup_plan, nullptr);
    EXPECT_EQ(fixup_plan->entry_count, 1u);
    EXPECT_EQ(fixup_plan->template_base, nullptr);
    EXPECT_EQ(fixup_plan->template_byte_length, 0u);
    const iree_hal_amdgpu_pm4_command_buffer_publish_stats_t* publish_stats =
        iree_hal_amdgpu_pm4_command_buffer_publish_stats(command_buffer);
    ASSERT_NE(publish_stats, nullptr);
    EXPECT_EQ(publish_stats->template_bytes, 0u);
    EXPECT_EQ(publish_stats->fixup_entry_bytes,
              sizeof(iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t));
  }

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

TEST_P(HostQueueAtomicTest, SupportsWidthsConditionsAndRmwOperations) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = GetParam();
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  if (GetParam() == IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4 &&
      !iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_dispatch_command_buffers(
          test_device.logical_device()
              ->physical_devices[0]
              ->vendor_packet_capabilities)) {
    GTEST_SKIP() << "PM4 command buffers are not supported on this device";
  }

  iree_hal_queue_affinity_t queue_affinity = 0;
  IREE_ASSERT_OK(QueueAffinityForPhysicalDevice(
      test_device, /*physical_device_ordinal=*/0, &queue_affinity));
  alignas(64) std::array<uint64_t, 8> storage = {};
  ReleaseLatch release_latch(/*release_count=*/1);
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(ImportHostAtomicBuffer(
      &test_device, queue_affinity, storage.data(), sizeof(storage),
      IREE_HAL_MEMORY_ACCESS_NONE, /*minimum_alignment=*/64,
      release_latch.callback(), buffer.out()));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_ATOMIC, queue_affinity,
      /*binding_capacity=*/1, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  const iree_hal_atomic_flags_t atomic_flags =
      IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE |
      IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE;
  auto append_sequence = [&](iree_hal_buffer_ref_t target,
                             iree_hal_atomic_width_t width) -> iree_status_t {
    const uint64_t wait_mask =
        width == IREE_HAL_ATOMIC_WIDTH_32 ? UINT32_MAX : UINT64_MAX;
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_atomic_store(
        command_buffer, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
        IREE_HAL_EXECUTION_STAGE_ATOMIC, target,
        (iree_hal_atomic_store_params_t){
            /*.value=*/10,
            /*.flags=*/atomic_flags,
            /*.width=*/width,
        }));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_atomic_rmw(
        command_buffer, IREE_HAL_EXECUTION_STAGE_ATOMIC,
        IREE_HAL_EXECUTION_STAGE_ATOMIC, target,
        (iree_hal_atomic_rmw_params_t){
            /*.operand=*/5,
            /*.flags=*/atomic_flags,
            /*.width=*/width,
            /*.operation=*/IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
        }));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_atomic_wait(
        command_buffer, IREE_HAL_EXECUTION_STAGE_ATOMIC,
        IREE_HAL_EXECUTION_STAGE_ATOMIC, target,
        (iree_hal_atomic_wait_params_t){
            /*.value=*/15,
            /*.mask=*/wait_mask,
            /*.flags=*/atomic_flags,
            /*.width=*/width,
            /*.condition=*/IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
        }));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_atomic_wait(
        command_buffer, IREE_HAL_EXECUTION_STAGE_ATOMIC,
        IREE_HAL_EXECUTION_STAGE_ATOMIC, target,
        (iree_hal_atomic_wait_params_t){
            /*.value=*/14,
            /*.mask=*/wait_mask,
            /*.flags=*/atomic_flags,
            /*.width=*/width,
            /*.condition=*/IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL,
        }));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_atomic_wait(
        command_buffer, IREE_HAL_EXECUTION_STAGE_ATOMIC,
        IREE_HAL_EXECUTION_STAGE_ATOMIC, target,
        (iree_hal_atomic_wait_params_t){
            /*.value=*/15,
            /*.mask=*/wait_mask,
            /*.flags=*/atomic_flags,
            /*.width=*/width,
            /*.condition=*/
            IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL,
        }));
    const iree_hal_atomic_rmw_operation_t operations[] = {
        IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT,
        IREE_HAL_ATOMIC_RMW_OPERATION_AND,
        IREE_HAL_ATOMIC_RMW_OPERATION_OR,
        IREE_HAL_ATOMIC_RMW_OPERATION_XOR,
    };
    const uint64_t operands[] = {2, 0xFu, 0x10u, 0x1u};
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(operations); ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_command_buffer_atomic_rmw(
          command_buffer, IREE_HAL_EXECUTION_STAGE_ATOMIC,
          i + 1 == IREE_ARRAYSIZE(operations) ? IREE_HAL_EXECUTION_STAGE_HOST
                                              : IREE_HAL_EXECUTION_STAGE_ATOMIC,
          target,
          (iree_hal_atomic_rmw_params_t){
              /*.operand=*/operands[i],
              /*.flags=*/atomic_flags,
              /*.width=*/width,
              /*.operation=*/operations[i],
          }));
    }
    return iree_ok_status();
  };
  IREE_ASSERT_OK(append_sequence(
      iree_hal_make_buffer_ref(buffer, /*offset=*/0, sizeof(uint32_t)),
      IREE_HAL_ATOMIC_WIDTH_32));
  IREE_ASSERT_OK(append_sequence(iree_hal_make_indirect_buffer_ref(
                                     /*binding=*/0,
                                     /*offset=*/sizeof(uint64_t),
                                     /*length=*/sizeof(uint64_t)),
                                 IREE_HAL_ATOMIC_WIDTH_64));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
  if (GetParam() == IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4) {
    const iree_hal_amdgpu_pm4_command_buffer_publish_stats_t* publish_stats =
        iree_hal_amdgpu_pm4_command_buffer_publish_stats(command_buffer);
    ASSERT_NE(publish_stats, nullptr);
    const uint64_t native_packet_dword_count =
        2u * IREE_HAL_AMDGPU_PM4_ATOMIC_MEM_DWORD_COUNT +
        3u * IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT32_DWORD_COUNT +
        3u * IREE_HAL_AMDGPU_PM4_ATOMIC_WAIT64_DWORD_COUNT;
    EXPECT_GE(publish_stats->atomic_dwords, native_packet_dword_count);
    EXPECT_LE(publish_stats->atomic_dwords,
              native_packet_dword_count + 4u * 3u);
    EXPECT_GT(publish_stats->dispatch_dwords, 0u);
  }

  Ref<iree_hal_semaphore_t> completion;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), completion.out()));
  iree_hal_semaphore_t* completion_semaphore = completion.get();
  uint64_t completion_value = 1;
  const iree_hal_semaphore_list_t signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&completion_semaphore,
      /*.payload_values=*/&completion_value,
  };
  const iree_hal_buffer_binding_t binding = {
      /*.buffer=*/buffer.get(),
      /*.offset=*/0,
      /*.length=*/IREE_HAL_WHOLE_BUFFER,
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      /*.count=*/1,
      /*.bindings=*/&binding,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), queue_affinity,
      iree_hal_semaphore_list_empty(), signal_list, command_buffer,
      binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
  command_buffer.reset();
  buffer.reset();
  IREE_ASSERT_OK(iree_hal_semaphore_wait(completion, completion_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  release_latch.Wait();
  EXPECT_EQ(static_cast<uint32_t>(storage[0]), 28u);
  EXPECT_EQ(storage[1], 28u);
}

static const char* CommandBufferModeName(
    const ::testing::TestParamInfo<iree_hal_amdgpu_command_buffer_mode_t>&
        info) {
  switch (info.param) {
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AQL:
      return "Aql";
    case IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4:
      return "Pm4";
    default:
      return "Unknown";
  }
}

INSTANTIATE_TEST_SUITE_P(
    CommandBufferModes, HostQueueAtomicTest,
    ::testing::Values(IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AQL,
                      IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4),
    CommandBufferModeName);

}  // namespace
}  // namespace iree::hal::amdgpu
