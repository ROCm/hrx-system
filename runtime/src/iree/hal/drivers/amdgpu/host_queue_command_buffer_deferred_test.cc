// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>

#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_test_util.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/pm4_command_buffer.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::hal::cts::Ref;
using namespace test;

class HostQueueCommandBufferDeferredTest : public ::testing::Test {
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

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t HostQueueCommandBufferDeferredTest::host_allocator_;
iree_hal_amdgpu_libhsa_t HostQueueCommandBufferDeferredTest::libhsa_;
iree_hal_amdgpu_topology_t HostQueueCommandBufferDeferredTest::topology_;

static iree_status_t QueueDeviceLocalDispatchTransientBuffer(
    iree_hal_device_t* device,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_device_size_t buffer_size, iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  return iree_hal_device_queue_alloca(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphore_list,
      signal_semaphore_list, /*pool=*/NULL, params, buffer_size,
      IREE_HAL_ALLOCA_FLAG_NONE, out_buffer);
}

static iree_status_t QueueHostVisibleDispatchTransientBuffer(
    iree_hal_device_t* device, const iree_hal_semaphore_list_t signal_list,
    iree_device_size_t buffer_size, iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                 IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  return iree_hal_device_queue_alloca(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      signal_list, /*pool=*/NULL, params, buffer_size,
      IREE_HAL_ALLOCA_FLAG_NONE, out_buffer);
}

TEST_F(HostQueueCommandBufferDeferredTest,
       PendingExecuteRetainsBindingResources) {
  static constexpr iree_device_size_t kBufferByteLength = 4 * sizeof(uint32_t);

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  iree_hal_executable_cache_t* executable_cache = NULL;
  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable_cache, &executable));

  Ref<iree_hal_buffer_t> source_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), kBufferByteLength, source_buffer.out()));
  const uint32_t source_values[4] = {1, 2, 3, 4};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(source_buffer, /*target_offset=*/0,
                                           source_values,
                                           sizeof(source_values)));

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), kBufferByteLength, output_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  Ref<iree_hal_semaphore_t> alloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), alloca_signal.out()));
  uint64_t alloca_signal_value = 1;
  iree_hal_semaphore_t* alloca_signal_ptr = alloca_signal.get();
  const iree_hal_semaphore_list_t alloca_signal_list = {
      /*count=*/1,
      /*semaphores=*/&alloca_signal_ptr,
      /*payload_values=*/&alloca_signal_value,
  };
  iree_hal_buffer_t* transient_raw = NULL;
  IREE_ASSERT_OK(QueueDeviceLocalDispatchTransientBuffer(
      test_device.base_device(), iree_hal_semaphore_list_empty(),
      alloca_signal_list, kBufferByteLength, &transient_raw));
  Ref<iree_hal_buffer_t> transient_buffer(transient_raw);

  Ref<iree_hal_semaphore_t> copy_signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), copy_signal.out()));
  uint64_t copy_signal_value = 1;
  iree_hal_semaphore_t* copy_signal_ptr = copy_signal.get();
  const iree_hal_semaphore_list_t copy_signal_list = {
      /*count=*/1,
      /*semaphores=*/&copy_signal_ptr,
      /*payload_values=*/&copy_signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_copy(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      alloca_signal_list, copy_signal_list, source_buffer, /*source_offset=*/0,
      transient_buffer, /*target_offset=*/0, kBufferByteLength,
      IREE_HAL_COPY_FLAG_NONE));
  alloca_signal.reset();
  IREE_ASSERT_OK(iree_hal_semaphore_wait(copy_signal, copy_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/0, /*offset=*/0, kBufferByteLength),
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/3, /*offset=*/0, kBufferByteLength),
  };
  const iree_hal_buffer_ref_list_t dispatch_bindings = {
      /*count=*/IREE_ARRAYSIZE(binding_refs),
      /*values=*/binding_refs,
  };
  const uint32_t constant_values[2] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_values, sizeof(constant_values));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/4, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants,
      dispatch_bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  Ref<iree_hal_semaphore_t> gate_signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), gate_signal.out()));
  uint64_t gate_signal_value = 1;
  iree_hal_semaphore_t* gate_signal_ptr = gate_signal.get();
  const iree_hal_semaphore_list_t gate_signal_list = {
      /*count=*/1,
      /*semaphores=*/&gate_signal_ptr,
      /*payload_values=*/&gate_signal_value,
  };

  Ref<iree_hal_semaphore_t> command_buffer_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), command_buffer_signal.out()));
  uint64_t command_buffer_signal_value = 1;
  iree_hal_semaphore_t* command_buffer_signal_ptr = command_buffer_signal.get();
  const iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1,
      /*semaphores=*/&command_buffer_signal_ptr,
      /*payload_values=*/&command_buffer_signal_value,
  };
  iree_hal_buffer_binding_t bindings[4] = {
      {
          /*buffer=*/transient_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
      {0},
      {0},
      {
          /*buffer=*/output_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      /*count=*/IREE_ARRAYSIZE(bindings),
      /*bindings=*/bindings,
  };

  Ref<iree_hal_semaphore_t> dealloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), dealloca_signal.out()));
  uint64_t dealloca_signal_value = 1;
  iree_hal_semaphore_t* dealloca_signal_ptr = dealloca_signal.get();
  const iree_hal_semaphore_list_t dealloca_signal_list = {
      /*count=*/1,
      /*semaphores=*/&dealloca_signal_ptr,
      /*payload_values=*/&dealloca_signal_value,
  };

  iree_status_t status = iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY, gate_signal_list,
      command_buffer_signal_list, command_buffer, binding_table,
      IREE_HAL_EXECUTE_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_dealloca(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        command_buffer_signal_list, dealloca_signal_list, transient_buffer,
        IREE_HAL_DEALLOCA_FLAG_NONE);
  }
  transient_buffer.reset();

  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_barrier(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_semaphore_list_empty(), gate_signal_list,
        IREE_HAL_EXECUTE_FLAG_NONE);
  }
  gate_signal.reset();

  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(dealloca_signal, dealloca_signal_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  IREE_ASSERT_OK(status);

  const uint32_t expected_values[4] = {13, 16, 19, 22};
  uint32_t actual_values[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, actual_values, sizeof(actual_values)));
  EXPECT_EQ(0, memcmp(actual_values, expected_values, sizeof(expected_values)));

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}

TEST_F(HostQueueCommandBufferDeferredTest,
       IndirectTransientBindingReusedAfterQueuedDealloca) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  Ref<iree_hal_buffer_t> output_buffers[2];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(output_buffers); ++i) {
    IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
        test_device.allocator(), sizeof(uint32_t), output_buffers[i].out()));
    IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffers[i], /*offset=*/0,
                                            IREE_HAL_WHOLE_BUFFER));
  }

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/2, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  const uint32_t expected = 0xBD3A0005u;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_indirect_buffer_ref(/*binding=*/0, /*offset=*/0,
                                        sizeof(expected)),
      &expected, sizeof(expected), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer,
      iree_hal_make_indirect_buffer_ref(/*binding=*/0, /*offset=*/0,
                                        sizeof(expected)),
      iree_hal_make_indirect_buffer_ref(/*binding=*/1, /*offset=*/0,
                                        sizeof(expected)),
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_buffer_params_t transient_params = {0};
  transient_params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  transient_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  transient_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;

  Ref<iree_hal_semaphore_t> first_alloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), first_alloca_signal.out()));
  uint64_t first_alloca_signal_value = 1;
  iree_hal_semaphore_t* first_alloca_signal_ptr = first_alloca_signal.get();
  iree_hal_semaphore_list_t first_alloca_signal_list = {
      /*count=*/1,
      /*semaphores=*/&first_alloca_signal_ptr,
      /*payload_values=*/&first_alloca_signal_value,
  };
  iree_hal_buffer_t* first_transient_raw = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), first_alloca_signal_list,
      /*pool=*/NULL, transient_params, sizeof(expected),
      IREE_HAL_ALLOCA_FLAG_NONE, &first_transient_raw));
  Ref<iree_hal_buffer_t> first_transient_buffer(first_transient_raw);

  Ref<iree_hal_semaphore_t> first_execute_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), first_execute_signal.out()));
  uint64_t first_execute_signal_value = 1;
  iree_hal_semaphore_t* first_execute_signal_ptr = first_execute_signal.get();
  iree_hal_semaphore_list_t first_execute_signal_list = {
      /*count=*/1,
      /*semaphores=*/&first_execute_signal_ptr,
      /*payload_values=*/&first_execute_signal_value,
  };
  iree_hal_buffer_binding_t first_bindings[2] = {
      {
          /*buffer=*/first_transient_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
      {
          /*buffer=*/output_buffers[0].get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
  };
  const iree_hal_buffer_binding_table_t first_binding_table = {
      /*count=*/IREE_ARRAYSIZE(first_bindings),
      /*bindings=*/first_bindings,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      first_alloca_signal_list, first_execute_signal_list, command_buffer,
      first_binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
  first_alloca_signal.reset();

  Ref<iree_hal_semaphore_t> first_dealloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), first_dealloca_signal.out()));
  uint64_t first_dealloca_signal_value = 1;
  iree_hal_semaphore_t* first_dealloca_signal_ptr = first_dealloca_signal.get();
  iree_hal_semaphore_list_t first_dealloca_signal_list = {
      /*count=*/1,
      /*semaphores=*/&first_dealloca_signal_ptr,
      /*payload_values=*/&first_dealloca_signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      first_execute_signal_list, first_dealloca_signal_list,
      first_transient_buffer.get(), IREE_HAL_DEALLOCA_FLAG_NONE));
  first_execute_signal.reset();
  first_transient_buffer.reset();

  Ref<iree_hal_semaphore_t> second_alloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), second_alloca_signal.out()));
  uint64_t second_alloca_signal_value = 1;
  iree_hal_semaphore_t* second_alloca_signal_ptr = second_alloca_signal.get();
  iree_hal_semaphore_list_t second_alloca_signal_list = {
      /*count=*/1,
      /*semaphores=*/&second_alloca_signal_ptr,
      /*payload_values=*/&second_alloca_signal_value,
  };
  iree_hal_buffer_t* second_transient_raw = NULL;
  IREE_ASSERT_OK(iree_hal_device_queue_alloca(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      first_dealloca_signal_list, second_alloca_signal_list,
      /*pool=*/NULL, transient_params, sizeof(expected),
      IREE_HAL_ALLOCA_FLAG_NONE, &second_transient_raw));
  first_dealloca_signal.reset();
  Ref<iree_hal_buffer_t> second_transient_buffer(second_transient_raw);

  Ref<iree_hal_semaphore_t> second_execute_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), second_execute_signal.out()));
  uint64_t second_execute_signal_value = 1;
  iree_hal_semaphore_t* second_execute_signal_ptr = second_execute_signal.get();
  iree_hal_semaphore_list_t second_execute_signal_list = {
      /*count=*/1,
      /*semaphores=*/&second_execute_signal_ptr,
      /*payload_values=*/&second_execute_signal_value,
  };
  iree_hal_buffer_binding_t second_bindings[2] = {
      {
          /*buffer=*/second_transient_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
      {
          /*buffer=*/output_buffers[1].get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
  };
  const iree_hal_buffer_binding_table_t second_binding_table = {
      /*count=*/IREE_ARRAYSIZE(second_bindings),
      /*bindings=*/second_bindings,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      second_alloca_signal_list, second_execute_signal_list, command_buffer,
      second_binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
  second_alloca_signal.reset();

  Ref<iree_hal_semaphore_t> second_dealloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), second_dealloca_signal.out()));
  uint64_t second_dealloca_signal_value = 1;
  iree_hal_semaphore_t* second_dealloca_signal_ptr =
      second_dealloca_signal.get();
  iree_hal_semaphore_list_t second_dealloca_signal_list = {
      /*count=*/1,
      /*semaphores=*/&second_dealloca_signal_ptr,
      /*payload_values=*/&second_dealloca_signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      second_execute_signal_list, second_dealloca_signal_list,
      second_transient_buffer.get(), IREE_HAL_DEALLOCA_FLAG_NONE));
  second_execute_signal.reset();
  second_transient_buffer.reset();

  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      second_dealloca_signal, second_dealloca_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(output_buffers); ++i) {
    uint32_t actual = 0;
    IREE_ASSERT_OK(iree_hal_buffer_map_read(output_buffers[i], /*offset=*/0,
                                            &actual, sizeof(actual)));
    EXPECT_EQ(actual, expected) << "output " << i;
  }
}

TEST_F(HostQueueCommandBufferDeferredTest,
       Pm4IndirectTransientBindingReusedAfterQueuedDealloca) {
  static constexpr iree_host_size_t kIterationCount = 8;
  static constexpr iree_device_size_t kBufferByteLength = 4 * sizeof(uint32_t);

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4;
  options.host_queues.upload_capacity = 64 * 1024;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  iree_hal_amdgpu_physical_device_t* physical_device =
      test_device.logical_device()->physical_devices[0];
  if (!iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_dispatch_command_buffers(
          physical_device->vendor_packet_capabilities)) {
    GTEST_SKIP() << "PM4 dispatch command buffers are not supported on this "
                    "physical device";
  }

  iree_hal_executable_cache_t* executable_cache = NULL;
  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable_cache, &executable));

  Ref<iree_hal_buffer_t> input_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), kBufferByteLength, input_buffer.out()));
  const uint32_t input_values[4] = {1, 2, 3, 4};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(input_buffer, /*target_offset=*/0,
                                           input_values, sizeof(input_values)));

  Ref<iree_hal_buffer_t> output_buffers[kIterationCount];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(output_buffers); ++i) {
    IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
        test_device.allocator(), kBufferByteLength, output_buffers[i].out()));
    IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffers[i], /*offset=*/0,
                                            IREE_HAL_WHOLE_BUFFER));
  }

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/0, /*offset=*/0, kBufferByteLength),
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/3, /*offset=*/0, kBufferByteLength),
  };
  const iree_hal_buffer_ref_list_t dispatch_bindings = {
      /*count=*/IREE_ARRAYSIZE(binding_refs),
      /*values=*/binding_refs,
  };
  const uint32_t constant_values[2] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_values, sizeof(constant_values));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/4, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants,
      dispatch_bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
  ASSERT_TRUE(iree_hal_amdgpu_pm4_command_buffer_isa(command_buffer));

  Ref<iree_hal_semaphore_t> previous_dealloca_signal;
  uint64_t previous_dealloca_signal_value = 0;
  for (iree_host_size_t i = 0; i < kIterationCount; ++i) {
    Ref<iree_hal_semaphore_t> alloca_signal;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), alloca_signal.out()));
    uint64_t alloca_signal_value = 1;
    iree_hal_semaphore_t* alloca_signal_ptr = alloca_signal.get();
    const iree_hal_semaphore_list_t alloca_signal_list = {
        /*count=*/1,
        /*semaphores=*/&alloca_signal_ptr,
        /*payload_values=*/&alloca_signal_value,
    };

    iree_hal_semaphore_t* alloca_wait_ptr = previous_dealloca_signal.get();
    const iree_hal_semaphore_list_t alloca_wait_list =
        alloca_wait_ptr
            ? iree_hal_semaphore_list_t{/*count=*/1,
                                        /*semaphores=*/&alloca_wait_ptr,
                                        /*payload_values=*/
                                        &previous_dealloca_signal_value}
            : iree_hal_semaphore_list_empty();
    iree_hal_buffer_t* transient_raw = NULL;
    IREE_ASSERT_OK(QueueDeviceLocalDispatchTransientBuffer(
        test_device.base_device(), alloca_wait_list, alloca_signal_list,
        kBufferByteLength, &transient_raw));
    previous_dealloca_signal.reset();
    Ref<iree_hal_buffer_t> transient_buffer(transient_raw);

    Ref<iree_hal_semaphore_t> execute_signal;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), execute_signal.out()));
    uint64_t execute_signal_value = 1;
    iree_hal_semaphore_t* execute_signal_ptr = execute_signal.get();
    const iree_hal_semaphore_list_t execute_signal_list = {
        /*count=*/1,
        /*semaphores=*/&execute_signal_ptr,
        /*payload_values=*/&execute_signal_value,
    };
    iree_hal_buffer_binding_t bindings[4] = {
        {
            /*buffer=*/input_buffer.get(),
            /*offset=*/0,
            /*length=*/IREE_HAL_WHOLE_BUFFER,
        },
        {0},
        {0},
        {
            /*buffer=*/transient_buffer.get(),
            /*offset=*/0,
            /*length=*/IREE_HAL_WHOLE_BUFFER,
        },
    };
    const iree_hal_buffer_binding_table_t binding_table = {
        /*count=*/IREE_ARRAYSIZE(bindings),
        /*bindings=*/bindings,
    };
    IREE_ASSERT_OK(iree_hal_device_queue_execute(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        alloca_signal_list, execute_signal_list, command_buffer, binding_table,
        IREE_HAL_EXECUTE_FLAG_NONE));
    alloca_signal.reset();

    Ref<iree_hal_semaphore_t> copy_signal;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), copy_signal.out()));
    uint64_t copy_signal_value = 1;
    iree_hal_semaphore_t* copy_signal_ptr = copy_signal.get();
    const iree_hal_semaphore_list_t copy_signal_list = {
        /*count=*/1,
        /*semaphores=*/&copy_signal_ptr,
        /*payload_values=*/&copy_signal_value,
    };
    IREE_ASSERT_OK(iree_hal_device_queue_copy(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        execute_signal_list, copy_signal_list, transient_buffer,
        /*source_offset=*/0, output_buffers[i],
        /*target_offset=*/0, kBufferByteLength, IREE_HAL_COPY_FLAG_NONE));
    execute_signal.reset();

    Ref<iree_hal_semaphore_t> dealloca_signal;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), dealloca_signal.out()));
    previous_dealloca_signal_value = 1;
    iree_hal_semaphore_t* dealloca_signal_ptr = dealloca_signal.get();
    const iree_hal_semaphore_list_t dealloca_signal_list = {
        /*count=*/1,
        /*semaphores=*/&dealloca_signal_ptr,
        /*payload_values=*/&previous_dealloca_signal_value,
    };
    IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        copy_signal_list, dealloca_signal_list, transient_buffer,
        IREE_HAL_DEALLOCA_FLAG_NONE));
    copy_signal.reset();
    transient_buffer.reset();
    previous_dealloca_signal.reset(dealloca_signal.release());
  }

  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      previous_dealloca_signal, previous_dealloca_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  const uint32_t expected_values[4] = {13, 16, 19, 22};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(output_buffers); ++i) {
    uint32_t actual_values[4] = {0, 0, 0, 0};
    IREE_ASSERT_OK(iree_hal_buffer_map_read(
        output_buffers[i], /*offset=*/0, actual_values, sizeof(actual_values)));
    EXPECT_EQ(0,
              memcmp(actual_values, expected_values, sizeof(expected_values)))
        << "output " << i;
  }

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}

TEST_F(HostQueueCommandBufferDeferredTest,
       IndirectPersistentAllocaSurvivesLocalAllocaChurn) {
  static constexpr iree_host_size_t kIterationCount = 36;
  static constexpr iree_device_size_t kBufferByteLength = 4 * sizeof(uint32_t);

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  iree_hal_executable_cache_t* executable_cache = NULL;
  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable_cache, &executable));

  Ref<iree_hal_buffer_t> input_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), kBufferByteLength, input_buffer.out()));
  const uint32_t input_values[4] = {1, 2, 3, 4};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(input_buffer, /*target_offset=*/0,
                                           input_values, sizeof(input_values)));

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), kBufferByteLength, output_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/0, /*offset=*/0, kBufferByteLength),
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/3, /*offset=*/0, kBufferByteLength),
  };
  const iree_hal_buffer_ref_list_t dispatch_bindings = {
      /*count=*/IREE_ARRAYSIZE(binding_refs),
      /*values=*/binding_refs,
  };
  const uint32_t constant_values[2] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_values, sizeof(constant_values));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/4, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants,
      dispatch_bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  Ref<iree_hal_semaphore_t> shared_alloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), shared_alloca_signal.out()));
  uint64_t shared_alloca_signal_value = 1;
  iree_hal_semaphore_t* shared_alloca_signal_ptr = shared_alloca_signal.get();
  const iree_hal_semaphore_list_t shared_alloca_signal_list = {
      /*count=*/1,
      /*semaphores=*/&shared_alloca_signal_ptr,
      /*payload_values=*/&shared_alloca_signal_value,
  };
  iree_hal_buffer_t* shared_raw = NULL;
  IREE_ASSERT_OK(QueueDeviceLocalDispatchTransientBuffer(
      test_device.base_device(), iree_hal_semaphore_list_empty(),
      shared_alloca_signal_list, kBufferByteLength, &shared_raw));
  Ref<iree_hal_buffer_t> shared_buffer(shared_raw);

  Ref<iree_hal_semaphore_t> previous_region_signal;
  uint64_t previous_region_signal_value = 1;
  {
    Ref<iree_hal_semaphore_t> write_signal;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), write_signal.out()));
    uint64_t write_signal_value = 1;
    iree_hal_semaphore_t* write_signal_ptr = write_signal.get();
    const iree_hal_semaphore_list_t write_signal_list = {
        /*count=*/1,
        /*semaphores=*/&write_signal_ptr,
        /*payload_values=*/&write_signal_value,
    };
    iree_hal_buffer_binding_t bindings[4] = {
        {
            /*buffer=*/input_buffer.get(),
            /*offset=*/0,
            /*length=*/IREE_HAL_WHOLE_BUFFER,
        },
        {0},
        {0},
        {
            /*buffer=*/shared_buffer.get(),
            /*offset=*/0,
            /*length=*/IREE_HAL_WHOLE_BUFFER,
        },
    };
    const iree_hal_buffer_binding_table_t binding_table = {
        /*count=*/IREE_ARRAYSIZE(bindings),
        /*bindings=*/bindings,
    };
    IREE_ASSERT_OK(iree_hal_device_queue_execute(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        shared_alloca_signal_list, write_signal_list, command_buffer,
        binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
    shared_alloca_signal.reset();
    previous_region_signal.reset(write_signal.release());
  }

  for (iree_host_size_t i = 0; i < kIterationCount; ++i) {
    Ref<iree_hal_semaphore_t> alloca_signal;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), alloca_signal.out()));
    uint64_t alloca_signal_value = 1;
    iree_hal_semaphore_t* alloca_signal_ptr = alloca_signal.get();
    const iree_hal_semaphore_list_t alloca_signal_list = {
        /*count=*/1,
        /*semaphores=*/&alloca_signal_ptr,
        /*payload_values=*/&alloca_signal_value,
    };
    iree_hal_semaphore_t* previous_region_signal_ptr =
        previous_region_signal.get();
    const iree_hal_semaphore_list_t alloca_wait_list = {
        /*count=*/1,
        /*semaphores=*/&previous_region_signal_ptr,
        /*payload_values=*/&previous_region_signal_value,
    };
    iree_hal_buffer_t* local_raw = NULL;
    IREE_ASSERT_OK(QueueDeviceLocalDispatchTransientBuffer(
        test_device.base_device(), alloca_wait_list, alloca_signal_list,
        kBufferByteLength, &local_raw));
    previous_region_signal.reset();
    Ref<iree_hal_buffer_t> local_buffer(local_raw);

    Ref<iree_hal_semaphore_t> execute_signal;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), execute_signal.out()));
    uint64_t execute_signal_value = 1;
    iree_hal_semaphore_t* execute_signal_ptr = execute_signal.get();
    const iree_hal_semaphore_list_t execute_signal_list = {
        /*count=*/1,
        /*semaphores=*/&execute_signal_ptr,
        /*payload_values=*/&execute_signal_value,
    };
    iree_hal_buffer_binding_t bindings[4] = {
        {
            /*buffer=*/input_buffer.get(),
            /*offset=*/0,
            /*length=*/IREE_HAL_WHOLE_BUFFER,
        },
        {0},
        {0},
        {
            /*buffer=*/local_buffer.get(),
            /*offset=*/0,
            /*length=*/IREE_HAL_WHOLE_BUFFER,
        },
    };
    const iree_hal_buffer_binding_table_t binding_table = {
        /*count=*/IREE_ARRAYSIZE(bindings),
        /*bindings=*/bindings,
    };
    IREE_ASSERT_OK(iree_hal_device_queue_execute(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        alloca_signal_list, execute_signal_list, command_buffer, binding_table,
        IREE_HAL_EXECUTE_FLAG_NONE));
    alloca_signal.reset();

    Ref<iree_hal_semaphore_t> dealloca_signal;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), dealloca_signal.out()));
    previous_region_signal_value = 1;
    iree_hal_semaphore_t* dealloca_signal_ptr = dealloca_signal.get();
    const iree_hal_semaphore_list_t dealloca_signal_list = {
        /*count=*/1,
        /*semaphores=*/&dealloca_signal_ptr,
        /*payload_values=*/&previous_region_signal_value,
    };
    IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        execute_signal_list, dealloca_signal_list, local_buffer,
        IREE_HAL_DEALLOCA_FLAG_NONE));
    execute_signal.reset();
    local_buffer.reset();
    previous_region_signal.reset(dealloca_signal.release());
  }

  Ref<iree_hal_semaphore_t> copy_signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), copy_signal.out()));
  uint64_t copy_signal_value = 1;
  iree_hal_semaphore_t* copy_signal_ptr = copy_signal.get();
  const iree_hal_semaphore_list_t copy_signal_list = {
      /*count=*/1,
      /*semaphores=*/&copy_signal_ptr,
      /*payload_values=*/&copy_signal_value,
  };
  iree_hal_semaphore_t* copy_wait_signal_ptr = previous_region_signal.get();
  const iree_hal_semaphore_list_t copy_wait_list = {
      /*count=*/1,
      /*semaphores=*/&copy_wait_signal_ptr,
      /*payload_values=*/&previous_region_signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_copy(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY, copy_wait_list,
      copy_signal_list, shared_buffer, /*source_offset=*/0, output_buffer,
      /*target_offset=*/0, kBufferByteLength, IREE_HAL_COPY_FLAG_NONE));
  previous_region_signal.reset();

  Ref<iree_hal_semaphore_t> shared_dealloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), shared_dealloca_signal.out()));
  uint64_t shared_dealloca_signal_value = 1;
  iree_hal_semaphore_t* shared_dealloca_signal_ptr =
      shared_dealloca_signal.get();
  const iree_hal_semaphore_list_t shared_dealloca_signal_list = {
      /*count=*/1,
      /*semaphores=*/&shared_dealloca_signal_ptr,
      /*payload_values=*/&shared_dealloca_signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY, copy_signal_list,
      shared_dealloca_signal_list, shared_buffer, IREE_HAL_DEALLOCA_FLAG_NONE));
  copy_signal.reset();
  shared_buffer.reset();

  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      shared_dealloca_signal, shared_dealloca_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  const uint32_t expected_values[4] = {13, 16, 19, 22};
  uint32_t actual_values[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, actual_values, sizeof(actual_values)));
  EXPECT_EQ(0, memcmp(actual_values, expected_values, sizeof(expected_values)));

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}

TEST_F(HostQueueCommandBufferDeferredTest,
       IndirectTransientInputBindingRetainedUntilQueuedDeallocaCompletes) {
  static constexpr iree_host_size_t kIterationCount = 36;
  static constexpr iree_device_size_t kBufferByteLength = 4 * sizeof(uint32_t);

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  iree_hal_executable_cache_t* executable_cache = NULL;
  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable_cache, &executable));

  Ref<iree_hal_buffer_t> source_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), kBufferByteLength, source_buffer.out()));
  const uint32_t source_values[4] = {1, 2, 3, 4};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(source_buffer, /*target_offset=*/0,
                                           source_values,
                                           sizeof(source_values)));

  Ref<iree_hal_buffer_t> output_buffers[kIterationCount];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(output_buffers); ++i) {
    IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
        test_device.allocator(), kBufferByteLength, output_buffers[i].out()));
    IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffers[i], /*offset=*/0,
                                            IREE_HAL_WHOLE_BUFFER));
  }

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/0, /*offset=*/0, kBufferByteLength),
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/3, /*offset=*/0, kBufferByteLength),
  };
  const iree_hal_buffer_ref_list_t dispatch_bindings = {
      /*count=*/IREE_ARRAYSIZE(binding_refs),
      /*values=*/binding_refs,
  };
  const uint32_t constant_values[2] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_values, sizeof(constant_values));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/4, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants,
      dispatch_bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  Ref<iree_hal_semaphore_t> previous_dealloca_signal;
  uint64_t previous_dealloca_signal_value = 0;
  for (iree_host_size_t i = 0; i < kIterationCount; ++i) {
    Ref<iree_hal_semaphore_t> alloca_signal;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), alloca_signal.out()));
    uint64_t alloca_signal_value = 1;
    iree_hal_semaphore_t* alloca_signal_ptr = alloca_signal.get();
    const iree_hal_semaphore_list_t alloca_signal_list = {
        /*count=*/1,
        /*semaphores=*/&alloca_signal_ptr,
        /*payload_values=*/&alloca_signal_value,
    };
    iree_hal_semaphore_t* alloca_wait_ptr = previous_dealloca_signal.get();
    const iree_hal_semaphore_list_t alloca_wait_list =
        alloca_wait_ptr
            ? iree_hal_semaphore_list_t{/*count=*/1,
                                        /*semaphores=*/&alloca_wait_ptr,
                                        /*payload_values=*/
                                        &previous_dealloca_signal_value}
            : iree_hal_semaphore_list_empty();
    iree_hal_buffer_t* transient_raw = NULL;
    IREE_ASSERT_OK(QueueDeviceLocalDispatchTransientBuffer(
        test_device.base_device(), alloca_wait_list, alloca_signal_list,
        kBufferByteLength, &transient_raw));
    previous_dealloca_signal.reset();
    Ref<iree_hal_buffer_t> transient_buffer(transient_raw);

    Ref<iree_hal_semaphore_t> copy_signal;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), copy_signal.out()));
    uint64_t copy_signal_value = 1;
    iree_hal_semaphore_t* copy_signal_ptr = copy_signal.get();
    const iree_hal_semaphore_list_t copy_signal_list = {
        /*count=*/1,
        /*semaphores=*/&copy_signal_ptr,
        /*payload_values=*/&copy_signal_value,
    };
    IREE_ASSERT_OK(iree_hal_device_queue_copy(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        alloca_signal_list, copy_signal_list, source_buffer,
        /*source_offset=*/0, transient_buffer,
        /*target_offset=*/0, kBufferByteLength, IREE_HAL_COPY_FLAG_NONE));
    alloca_signal.reset();

    Ref<iree_hal_semaphore_t> execute_signal;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), execute_signal.out()));
    uint64_t execute_signal_value = 1;
    iree_hal_semaphore_t* execute_signal_ptr = execute_signal.get();
    const iree_hal_semaphore_list_t execute_signal_list = {
        /*count=*/1,
        /*semaphores=*/&execute_signal_ptr,
        /*payload_values=*/&execute_signal_value,
    };
    iree_hal_buffer_binding_t bindings[4] = {
        {
            /*buffer=*/transient_buffer.get(),
            /*offset=*/0,
            /*length=*/IREE_HAL_WHOLE_BUFFER,
        },
        {0},
        {0},
        {
            /*buffer=*/output_buffers[i].get(),
            /*offset=*/0,
            /*length=*/IREE_HAL_WHOLE_BUFFER,
        },
    };
    const iree_hal_buffer_binding_table_t binding_table = {
        /*count=*/IREE_ARRAYSIZE(bindings),
        /*bindings=*/bindings,
    };
    IREE_ASSERT_OK(iree_hal_device_queue_execute(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        copy_signal_list, execute_signal_list, command_buffer, binding_table,
        IREE_HAL_EXECUTE_FLAG_NONE));
    copy_signal.reset();

    Ref<iree_hal_semaphore_t> dealloca_signal;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), dealloca_signal.out()));
    previous_dealloca_signal_value = 1;
    iree_hal_semaphore_t* dealloca_signal_ptr = dealloca_signal.get();
    const iree_hal_semaphore_list_t dealloca_signal_list = {
        /*count=*/1,
        /*semaphores=*/&dealloca_signal_ptr,
        /*payload_values=*/&previous_dealloca_signal_value,
    };
    IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        execute_signal_list, dealloca_signal_list, transient_buffer,
        IREE_HAL_DEALLOCA_FLAG_NONE));
    execute_signal.reset();
    transient_buffer.reset();
    previous_dealloca_signal.reset(dealloca_signal.release());
  }

  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      previous_dealloca_signal, previous_dealloca_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  const uint32_t expected_values[4] = {13, 16, 19, 22};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(output_buffers); ++i) {
    uint32_t actual_values[4] = {0, 0, 0, 0};
    IREE_ASSERT_OK(iree_hal_buffer_map_read(
        output_buffers[i], /*offset=*/0, actual_values, sizeof(actual_values)));
    EXPECT_EQ(0,
              memcmp(actual_values, expected_values, sizeof(expected_values)))
        << "output " << i;
  }

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}

TEST_F(HostQueueCommandBufferDeferredTest,
       LongDeferredTransientChainPreservesSharedSideRange) {
  static constexpr iree_host_size_t kIterationCount = 36;
  static constexpr iree_host_size_t kSelectedIteration = 31;
  static constexpr iree_device_size_t kBufferByteLength = 4 * sizeof(uint32_t);
  static constexpr iree_device_size_t kSharedByteLength = 2 * kBufferByteLength;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  iree_hal_executable_cache_t* executable_cache = NULL;
  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable_cache, &executable));

  Ref<iree_hal_buffer_t> source_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), kBufferByteLength, source_buffer.out()));
  const uint32_t source_values[4] = {1, 2, 3, 4};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(source_buffer, /*target_offset=*/0,
                                           source_values,
                                           sizeof(source_values)));

  Ref<iree_hal_buffer_t> final_output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), kBufferByteLength, final_output_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(final_output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/0, /*offset=*/0, kBufferByteLength),
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/3, /*offset=*/0, kBufferByteLength),
  };
  const iree_hal_buffer_ref_list_t dispatch_bindings = {
      /*count=*/IREE_ARRAYSIZE(binding_refs),
      /*values=*/binding_refs,
  };
  const uint32_t constant_values[2] = {3, 10};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_values, sizeof(constant_values));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/4, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants,
      dispatch_bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  Ref<iree_hal_semaphore_t> shared_alloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), shared_alloca_signal.out()));
  uint64_t shared_alloca_signal_value = 1;
  iree_hal_semaphore_t* shared_alloca_signal_ptr = shared_alloca_signal.get();
  const iree_hal_semaphore_list_t shared_alloca_signal_list = {
      /*count=*/1,
      /*semaphores=*/&shared_alloca_signal_ptr,
      /*payload_values=*/&shared_alloca_signal_value,
  };
  iree_hal_buffer_t* shared_buffer_raw = NULL;
  IREE_ASSERT_OK(QueueHostVisibleDispatchTransientBuffer(
      test_device.base_device(), shared_alloca_signal_list, kSharedByteLength,
      &shared_buffer_raw));
  Ref<iree_hal_buffer_t> shared_buffer(shared_buffer_raw);

  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(&libhsa_), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/NULL, /*attributes=*/0, &blocker_signal));
  IREE_ASSERT_OK(EnqueueRawBlockingBarrier(queue, blocker_signal));

  Ref<iree_hal_semaphore_t> previous_dealloca_signal;
  uint64_t previous_dealloca_signal_value = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < kIterationCount && iree_status_is_ok(status);
       ++i) {
    Ref<iree_hal_semaphore_t> alloca_signal;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), alloca_signal.out()));
    uint64_t alloca_signal_value = 1;
    iree_hal_semaphore_t* alloca_signal_ptr = alloca_signal.get();
    const iree_hal_semaphore_list_t alloca_signal_list = {
        /*count=*/1,
        /*semaphores=*/&alloca_signal_ptr,
        /*payload_values=*/&alloca_signal_value,
    };
    iree_hal_semaphore_t* alloca_wait_ptr = previous_dealloca_signal
                                                ? previous_dealloca_signal.get()
                                                : shared_alloca_signal.get();
    uint64_t alloca_wait_value = previous_dealloca_signal
                                     ? previous_dealloca_signal_value
                                     : shared_alloca_signal_value;
    const iree_hal_semaphore_list_t alloca_wait_list = {
        /*count=*/1,
        /*semaphores=*/&alloca_wait_ptr,
        /*payload_values=*/&alloca_wait_value,
    };

    Ref<iree_hal_semaphore_t> copy_signal;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), copy_signal.out()));
    uint64_t copy_signal_value = 1;
    iree_hal_semaphore_t* copy_signal_ptr = copy_signal.get();
    const iree_hal_semaphore_list_t copy_signal_list = {
        /*count=*/1,
        /*semaphores=*/&copy_signal_ptr,
        /*payload_values=*/&copy_signal_value,
    };

    Ref<iree_hal_semaphore_t> execute_signal;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), execute_signal.out()));
    uint64_t execute_signal_value = 1;
    iree_hal_semaphore_t* execute_signal_ptr = execute_signal.get();
    const iree_hal_semaphore_list_t execute_signal_list = {
        /*count=*/1,
        /*semaphores=*/&execute_signal_ptr,
        /*payload_values=*/&execute_signal_value,
    };

    Ref<iree_hal_semaphore_t> dealloca_signal;
    IREE_ASSERT_OK(
        CreateSemaphore(test_device.base_device(), dealloca_signal.out()));
    previous_dealloca_signal_value = 1;
    iree_hal_semaphore_t* dealloca_signal_ptr = dealloca_signal.get();
    const iree_hal_semaphore_list_t dealloca_signal_list = {
        /*count=*/1,
        /*semaphores=*/&dealloca_signal_ptr,
        /*payload_values=*/&previous_dealloca_signal_value,
    };

    iree_hal_buffer_t* transient_raw = NULL;
    status = QueueDeviceLocalDispatchTransientBuffer(
        test_device.base_device(), alloca_wait_list, alloca_signal_list,
        kBufferByteLength, &transient_raw);
    previous_dealloca_signal.reset();
    Ref<iree_hal_buffer_t> transient_buffer(transient_raw);

    if (iree_status_is_ok(status)) {
      status = iree_hal_device_queue_copy(
          test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
          alloca_signal_list, copy_signal_list, source_buffer,
          /*source_offset=*/0, transient_buffer,
          /*target_offset=*/0, kBufferByteLength, IREE_HAL_COPY_FLAG_NONE);
    }
    alloca_signal.reset();

    const iree_device_size_t shared_output_offset =
        i == kSelectedIteration ? kBufferByteLength : 0;
    iree_hal_buffer_binding_t bindings[4] = {
        {
            /*buffer=*/transient_buffer.get(),
            /*offset=*/0,
            /*length=*/IREE_HAL_WHOLE_BUFFER,
        },
        {0},
        {0},
        {
            /*buffer=*/shared_buffer.get(),
            /*offset=*/shared_output_offset,
            /*length=*/kBufferByteLength,
        },
    };
    const iree_hal_buffer_binding_table_t binding_table = {
        /*count=*/IREE_ARRAYSIZE(bindings),
        /*bindings=*/bindings,
    };
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_queue_execute(
          test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
          copy_signal_list, execute_signal_list, command_buffer, binding_table,
          IREE_HAL_EXECUTE_FLAG_NONE);
    }
    copy_signal.reset();

    if (iree_status_is_ok(status)) {
      status = iree_hal_device_queue_dealloca(
          test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
          execute_signal_list, dealloca_signal_list, transient_buffer,
          IREE_HAL_DEALLOCA_FLAG_NONE);
    }
    execute_signal.reset();
    transient_buffer.reset();
    previous_dealloca_signal.reset(dealloca_signal.release());
  }
  shared_alloca_signal.reset();

  Ref<iree_hal_semaphore_t> final_execute_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), final_execute_signal.out()));
  uint64_t final_execute_signal_value = 1;
  iree_hal_semaphore_t* final_execute_signal_ptr = final_execute_signal.get();
  const iree_hal_semaphore_list_t final_execute_signal_list = {
      /*count=*/1,
      /*semaphores=*/&final_execute_signal_ptr,
      /*payload_values=*/&final_execute_signal_value,
  };
  iree_hal_semaphore_t* final_wait_ptr = previous_dealloca_signal.get();
  const iree_hal_semaphore_list_t final_wait_list = {
      /*count=*/1,
      /*semaphores=*/&final_wait_ptr,
      /*payload_values=*/&previous_dealloca_signal_value,
  };
  iree_hal_buffer_binding_t final_bindings[4] = {
      {
          /*buffer=*/shared_buffer.get(),
          /*offset=*/kBufferByteLength,
          /*length=*/kBufferByteLength,
      },
      {0},
      {0},
      {
          /*buffer=*/final_output_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
  };
  const iree_hal_buffer_binding_table_t final_binding_table = {
      /*count=*/IREE_ARRAYSIZE(final_bindings),
      /*bindings=*/final_bindings,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_execute(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY, final_wait_list,
        final_execute_signal_list, command_buffer, final_binding_table,
        IREE_HAL_EXECUTE_FLAG_NONE);
  }
  previous_dealloca_signal.reset();

  Ref<iree_hal_semaphore_t> shared_dealloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), shared_dealloca_signal.out()));
  uint64_t shared_dealloca_signal_value = 1;
  iree_hal_semaphore_t* shared_dealloca_signal_ptr =
      shared_dealloca_signal.get();
  const iree_hal_semaphore_list_t shared_dealloca_signal_list = {
      /*count=*/1,
      /*semaphores=*/&shared_dealloca_signal_ptr,
      /*payload_values=*/&shared_dealloca_signal_value,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_dealloca(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        final_execute_signal_list, shared_dealloca_signal_list, shared_buffer,
        IREE_HAL_DEALLOCA_FLAG_NONE);
  }
  final_execute_signal.reset();
  shared_buffer.reset();

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);

  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(
        shared_dealloca_signal, shared_dealloca_signal_value,
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
  }
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));

  IREE_ASSERT_OK(status);

  const uint32_t expected_values[4] = {49, 58, 67, 76};
  uint32_t actual_values[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      final_output_buffer, /*offset=*/0, actual_values, sizeof(actual_values)));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(expected_values); ++i) {
    EXPECT_EQ(actual_values[i], expected_values[i]) << "value " << i;
  }

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}
}  // namespace
}  // namespace iree::hal::amdgpu
