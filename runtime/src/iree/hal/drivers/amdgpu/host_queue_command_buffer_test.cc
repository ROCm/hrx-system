// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_command_buffer.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_packet.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_test_util.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/pm4_command_buffer.h"
#include "iree/hal/drivers/amdgpu/queue_affinity.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::hal::cts::Ref;
using namespace test;

class HostQueueCommandBufferTest : public ::testing::Test {
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

iree_allocator_t HostQueueCommandBufferTest::host_allocator_;
iree_hal_amdgpu_libhsa_t HostQueueCommandBufferTest::libhsa_;
iree_hal_amdgpu_topology_t HostQueueCommandBufferTest::topology_;

TEST_F(HostQueueCommandBufferTest, DispatchSummariesRetainPacketOrdinals) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  TwoDispatchCommandBuffer fixture;
  IREE_ASSERT_OK(CreateTwoDispatchCommandBuffer(
      &test_device, &fixture,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA));
  EXPECT_EQ(
      iree_hal_amdgpu_aql_command_buffer_profile_id(fixture.command_buffer),
      0u);

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(fixture.command_buffer);
  ASSERT_NE(program, nullptr);
  ASSERT_NE(program->first_block, nullptr);
  EXPECT_EQ(program->first_block->dispatch_count, 2u);
  EXPECT_EQ(program->first_block->aql_packet_count, 2u);

  uint32_t summary_count = 0;
  const iree_hal_amdgpu_aql_command_buffer_dispatch_summary_t* summary =
      iree_hal_amdgpu_aql_command_buffer_dispatch_summaries(
          fixture.command_buffer, program->first_block, &summary_count);
  ASSERT_NE(summary, nullptr);
  EXPECT_EQ(summary_count, 2u);

  EXPECT_EQ(summary->packets.first_ordinal, 0u);
  EXPECT_EQ(summary->packets.dispatch_ordinal, 0u);
  EXPECT_EQ(summary->metadata.command_index, 0u);
  EXPECT_EQ(summary->metadata.function_ordinal, 0u);
  EXPECT_EQ(summary->metadata.dispatch_flags,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_NONE);

  ASSERT_NE(summary->next, nullptr);
  summary = summary->next;
  EXPECT_EQ(summary->packets.first_ordinal, 1u);
  EXPECT_EQ(summary->packets.dispatch_ordinal, 1u);
  EXPECT_EQ(summary->metadata.command_index, 1u);
  EXPECT_EQ(summary->metadata.function_ordinal, 0u);
  EXPECT_EQ(summary->metadata.dispatch_flags,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_DISPATCH_FLAG_NONE);
  EXPECT_EQ(summary->next, nullptr);
}

TEST_F(HostQueueCommandBufferTest,
       PacketControlBarriersFirstPayloadPacketForInlineWait) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  iree_hal_amdgpu_wait_resolution_t resolution = {0};
  resolution.inline_acquire_scope = IREE_HSA_FENCE_SCOPE_AGENT;
  iree_hal_amdgpu_aql_packet_control_t control =
      iree_hal_amdgpu_host_queue_command_buffer_packet_control(
          queue, &resolution, iree_hal_semaphore_list_empty(),
          /*packet_index=*/0, IREE_HSA_FENCE_SCOPE_NONE,
          IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_NONE);
  EXPECT_TRUE(control.has_barrier);
  EXPECT_EQ(control.acquire_fence_scope, IREE_HSA_FENCE_SCOPE_AGENT);
  EXPECT_EQ(control.release_fence_scope, IREE_HSA_FENCE_SCOPE_NONE);

  control = iree_hal_amdgpu_host_queue_command_buffer_packet_control(
      queue, &resolution, iree_hal_semaphore_list_empty(), /*packet_index=*/1,
      IREE_HSA_FENCE_SCOPE_NONE,
      IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_NONE);
  EXPECT_FALSE(control.has_barrier);
  EXPECT_EQ(control.acquire_fence_scope, IREE_HSA_FENCE_SCOPE_NONE);
  EXPECT_EQ(control.release_fence_scope, IREE_HSA_FENCE_SCOPE_NONE);
}

TEST_F(HostQueueCommandBufferTest,
       KernargRingUsesRecordedCpuVisibleCoarseCapability) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  ASSERT_GT(logical_device->physical_device_count, 0u);
  iree_hal_amdgpu_physical_device_t* physical_device =
      logical_device->physical_devices[0];
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  const iree_hal_amdgpu_cpu_visible_device_coarse_memory_t* capability =
      &physical_device->cpu_visible_device_coarse_memory;
  const bool uses_cpu_visible_device_coarse = iree_any_bit_set(
      capability->flags,
      IREE_HAL_AMDGPU_CPU_VISIBLE_DEVICE_COARSE_MEMORY_FLAG_AVAILABLE);
  if (uses_cpu_visible_device_coarse) {
    EXPECT_EQ(queue->kernarg_ring.publication.mode,
              capability->host_write_publication.mode);
    EXPECT_EQ(queue->kernarg_ring.publication.hdp_mem_flush_control,
              capability->host_write_publication.hdp_mem_flush_control);
  } else {
    EXPECT_EQ(queue->kernarg_ring.publication.mode,
              IREE_HAL_AMDGPU_KERNARG_RING_PUBLICATION_MODE_NONE);
  }
}

TEST_F(HostQueueCommandBufferTest,
       PrepublishedKernargsUseRecordedDeviceFineStorage) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  ASSERT_GT(logical_device->physical_device_count, 0u);
  const iree_hal_amdgpu_aql_prepublished_kernarg_storage_t* storage =
      &logical_device->physical_devices[0]->prepublished_kernarg_storage;
  if (storage->mode ==
      IREE_HAL_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_MODE_DISABLED) {
    GTEST_SKIP() << "fine-grained GPU memory pool is not available";
  }

  EXPECT_EQ(
      storage->mode,
      IREE_HAL_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_MODE_DEVICE_FINE_HOST_COHERENT);
  EXPECT_TRUE(iree_all_bits_set(storage->buffer_params.type,
                                IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                                    IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                                    IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
  EXPECT_TRUE(iree_all_bits_set(storage->buffer_params.access,
                                IREE_HAL_MEMORY_ACCESS_ALL));
  EXPECT_TRUE(iree_all_bits_set(storage->buffer_params.usage,
                                IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ |
                                    IREE_HAL_BUFFER_USAGE_MAPPING));
}

TEST_F(HostQueueCommandBufferTest, DirectDispatchUsesPrepublishedKernargs) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  ASSERT_GT(logical_device->physical_device_count, 0u);
  const iree_hal_amdgpu_aql_prepublished_kernarg_storage_t* storage =
      &logical_device->physical_devices[0]->prepublished_kernarg_storage;
  if (storage->mode ==
      IREE_HAL_AMDGPU_AQL_PREPUBLISHED_KERNARG_STORAGE_MODE_DISABLED) {
    GTEST_SKIP() << "fine-grained GPU memory pool is not available";
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
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      input_buffer.out()));
  const uint32_t input_values[4] = {1, 2, 3, 4};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(input_buffer, /*target_offset=*/0,
                                           input_values, sizeof(input_values)));

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      output_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(output_buffer)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
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
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  ASSERT_NE(program->first_block, nullptr);
  EXPECT_EQ(program->max_block_kernarg_length, 0u);
  const iree_hal_amdgpu_command_buffer_command_header_t* command =
      iree_hal_amdgpu_command_buffer_block_commands_const(program->first_block);
  ASSERT_EQ(command->opcode, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH);
  const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command =
      (const iree_hal_amdgpu_command_buffer_dispatch_command_t*)command;
  EXPECT_EQ(dispatch_command->kernarg_storage_mode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STORAGE_MODE_PREPUBLISHED);
  const uint32_t kernarg_length =
      (uint32_t)dispatch_command->kernarg_length_qwords * 8u;
  EXPECT_EQ(dispatch_command->payload_reference, 0u);
  EXPECT_NE(
      iree_hal_amdgpu_aql_command_buffer_prepublished_kernarg(
          command_buffer, dispatch_command->payload_reference, kernarg_length),
      nullptr);
  const iree_hal_amdgpu_command_buffer_command_header_t* second_command =
      (const iree_hal_amdgpu_command_buffer_command_header_t*)((const uint8_t*)
                                                                   command +
                                                               command->length_qwords *
                                                                   8u);
  ASSERT_EQ(second_command->opcode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH);
  const iree_hal_amdgpu_command_buffer_dispatch_command_t*
      second_dispatch_command =
          (const iree_hal_amdgpu_command_buffer_dispatch_command_t*)
              second_command;
  EXPECT_EQ(second_dispatch_command->kernarg_storage_mode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STORAGE_MODE_PREPUBLISHED);
  EXPECT_GT(second_dispatch_command->payload_reference, 1u);
  EXPECT_NE(iree_hal_amdgpu_aql_command_buffer_prepublished_kernarg(
                command_buffer, second_dispatch_command->payload_reference,
                (uint32_t)second_dispatch_command->kernarg_length_qwords * 8u),
            nullptr);

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
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), command_buffer_signal_list,
      command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint32_t output_values[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  const uint32_t expected_values[4] = {13, 16, 19, 22};
  EXPECT_EQ(0, memcmp(output_values, expected_values, sizeof(expected_values)));

  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  Ref<iree_hal_command_buffer_t> one_shot_command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, one_shot_command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(one_shot_command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      one_shot_command_buffer, executable,
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(one_shot_command_buffer));

  const iree_hal_amdgpu_aql_program_t* one_shot_program =
      iree_hal_amdgpu_aql_command_buffer_program(one_shot_command_buffer);
  ASSERT_NE(one_shot_program->first_block, nullptr);
  EXPECT_GT(one_shot_program->max_block_kernarg_length, 0u);
  const iree_hal_amdgpu_command_buffer_command_header_t* one_shot_command =
      iree_hal_amdgpu_command_buffer_block_commands_const(
          one_shot_program->first_block);
  ASSERT_EQ(one_shot_command->opcode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH);
  const iree_hal_amdgpu_command_buffer_dispatch_command_t*
      one_shot_dispatch_command =
          (const iree_hal_amdgpu_command_buffer_dispatch_command_t*)
              one_shot_command;
  EXPECT_EQ(one_shot_dispatch_command->kernarg_storage_mode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STORAGE_MODE_NATIVE_INLINE);
  EXPECT_EQ(
      iree_hal_amdgpu_aql_command_buffer_prepublished_kernarg(
          one_shot_command_buffer, one_shot_dispatch_command->payload_reference,
          (uint32_t)one_shot_dispatch_command->kernarg_length_qwords * 8u),
      nullptr);

  command_buffer_signal_value = 2;
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), command_buffer_signal_list,
      one_shot_command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  memset(output_values, 0, sizeof(output_values));
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  EXPECT_EQ(0, memcmp(output_values, expected_values, sizeof(expected_values)));

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}

TEST_F(HostQueueCommandBufferTest,
       Pm4DispatchUsesResidentPrepublishedCommandBuffer) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4;
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

  TwoDispatchCommandBuffer fixture;
  IREE_ASSERT_OK(CreateTwoDispatchCommandBuffer(
      &test_device, &fixture, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT));
  ASSERT_TRUE(iree_hal_amdgpu_pm4_command_buffer_isa(fixture.command_buffer));
  const iree_hal_amdgpu_pm4_program_t* pm4_program =
      iree_hal_amdgpu_pm4_command_buffer_program(fixture.command_buffer);
  ASSERT_NE(pm4_program->dwords, nullptr);
  EXPECT_GT(pm4_program->dword_count, 0u);
  const iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t* fixup_plan =
      iree_hal_amdgpu_pm4_command_buffer_fixup_plan(fixture.command_buffer);
  EXPECT_EQ(fixup_plan->entries, nullptr);
  EXPECT_EQ(fixup_plan->entry_count, 0u);
  ASSERT_NE(fixup_plan->target_base, nullptr);
  EXPECT_GT(fixup_plan->target_byte_length, 0u);

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
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), command_buffer_signal_list,
      fixture.command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectTwoDispatchOutputs(fixture);

  iree_hal_command_buffer_t* one_shot_command_buffer = nullptr;
  iree_status_t one_shot_status = iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &one_shot_command_buffer);
  EXPECT_EQ(iree_status_code(one_shot_status), IREE_STATUS_UNIMPLEMENTED);
  iree_status_free(one_shot_status);
  iree_hal_command_buffer_release(one_shot_command_buffer);
}

TEST_F(HostQueueCommandBufferTest,
       AutoCommandBufferModeUsesAqlWithoutUploadRing) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AUTO;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  EXPECT_TRUE(iree_hal_amdgpu_aql_command_buffer_isa(command_buffer));
  EXPECT_FALSE(iree_hal_amdgpu_pm4_command_buffer_isa(command_buffer));

  Ref<iree_hal_command_buffer_t> transfer_command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, transfer_command_buffer.out()));
  EXPECT_TRUE(iree_hal_amdgpu_aql_command_buffer_isa(transfer_command_buffer));
  EXPECT_FALSE(iree_hal_amdgpu_pm4_command_buffer_isa(transfer_command_buffer));
}

TEST_F(HostQueueCommandBufferTest,
       AutoCommandBufferModeUsesPm4WhenFullySupported) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AUTO;
  options.host_queues.upload_capacity = 64 * 1024;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));

  iree_hal_amdgpu_physical_device_t* physical_device =
      test_device.logical_device()->physical_devices[0];
  const bool pm4_supported =
      iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_dispatch_command_buffers(
          physical_device->vendor_packet_capabilities);
  EXPECT_EQ(pm4_supported,
            iree_hal_amdgpu_pm4_command_buffer_isa(command_buffer));
  EXPECT_NE(pm4_supported,
            iree_hal_amdgpu_aql_command_buffer_isa(command_buffer));
}

TEST_F(HostQueueCommandBufferTest,
       AutoCommandBufferModeFallsBackToAqlForUnsupportedRequest) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AUTO;
  options.host_queues.upload_capacity = 64 * 1024;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  EXPECT_TRUE(iree_hal_amdgpu_aql_command_buffer_isa(command_buffer));
  EXPECT_FALSE(iree_hal_amdgpu_pm4_command_buffer_isa(command_buffer));
}

TEST_F(HostQueueCommandBufferTest,
       ExplicitPm4CommandBufferModeRejectsUnsupportedCategories) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4;
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

  iree_hal_command_buffer_t* command_buffer = nullptr;
  iree_status_t status = iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  iree_status_free(status);
  iree_hal_command_buffer_release(command_buffer);
}

TEST_F(HostQueueCommandBufferTest, Pm4MixedDynamicDispatchUsesGpuFixup) {
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
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      input_buffer.out()));
  const uint32_t input_values[4] = {1, 2, 3, 4};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(input_buffer, /*target_offset=*/0,
                                           input_values, sizeof(input_values)));

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      output_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/3, /*offset=*/0,
          iree_hal_buffer_byte_length(output_buffer)),
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
  const iree_hal_amdgpu_pm4_program_t* pm4_program =
      iree_hal_amdgpu_pm4_command_buffer_program(command_buffer);
  ASSERT_NE(pm4_program->dwords, nullptr);
  EXPECT_GT(pm4_program->dword_count, 0u);
  const iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t* fixup_plan =
      iree_hal_amdgpu_pm4_command_buffer_fixup_plan(command_buffer);
  ASSERT_NE(fixup_plan->entries, nullptr);
  ASSERT_NE(fixup_plan->target_base, nullptr);
  EXPECT_GE(fixup_plan->entry_count, 1u);
  EXPECT_LE(fixup_plan->entry_count, 2u);
  EXPECT_GT(fixup_plan->target_byte_length, 0u);

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
          /*buffer=*/input_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
      {
          /*buffer=*/input_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
      {
          /*buffer=*/input_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
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
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), command_buffer_signal_list,
      command_buffer, binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint32_t output_values[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  const uint32_t expected_values[4] = {13, 16, 19, 22};
  EXPECT_EQ(0, memcmp(output_values, expected_values, sizeof(expected_values)));

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}

TEST_F(HostQueueCommandBufferTest, Pm4DynamicDispatchUsesBindingTableSlots) {
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
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      input_buffer.out()));
  const uint32_t input_values[4] = {1, 2, 3, 4};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(input_buffer, /*target_offset=*/0,
                                           input_values, sizeof(input_values)));

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      output_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/1, /*offset=*/0,
          iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/3, /*offset=*/0,
          iree_hal_buffer_byte_length(output_buffer)),
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
  const iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t* fixup_plan =
      iree_hal_amdgpu_pm4_command_buffer_fixup_plan(command_buffer);
  ASSERT_NE(fixup_plan->entries, nullptr);
  ASSERT_NE(fixup_plan->target_base, nullptr);
  EXPECT_GE(fixup_plan->entry_count, 2u);
  EXPECT_LE(fixup_plan->entry_count, 4u);
  EXPECT_GT(fixup_plan->target_byte_length, 0u);

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
          /*buffer=*/input_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
      {
          /*buffer=*/input_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
      {
          /*buffer=*/input_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
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
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), command_buffer_signal_list,
      command_buffer, binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint32_t output_values[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  const uint32_t expected_values[4] = {13, 16, 19, 22};
  EXPECT_EQ(0, memcmp(output_values, expected_values, sizeof(expected_values)));

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}

TEST_F(HostQueueCommandBufferTest,
       MixedDynamicDispatchUsesPatchedKernargTemplate) {
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
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      input_buffer.out()));
  const uint32_t input_values[4] = {1, 2, 3, 4};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(input_buffer, /*target_offset=*/0,
                                           input_values, sizeof(input_values)));

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      output_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/3, /*offset=*/0,
          iree_hal_buffer_byte_length(output_buffer)),
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
      test_device.base_device(),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT |
          IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/4, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants,
      dispatch_bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  ASSERT_NE(program->first_block, nullptr);
  EXPECT_GT(program->max_block_kernarg_length, 0u);
  ASSERT_EQ(program->first_block->binding_source_count, 1u);
  const iree_hal_amdgpu_command_buffer_binding_source_t* binding_source =
      iree_hal_amdgpu_command_buffer_block_binding_sources_const(
          program->first_block);
  EXPECT_EQ(binding_source->flags,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_DYNAMIC);
  EXPECT_EQ(binding_source->slot, 3u);
  EXPECT_EQ(binding_source->target_qword_index, 1u);

  const iree_hal_amdgpu_command_buffer_command_header_t* command =
      iree_hal_amdgpu_command_buffer_block_commands_const(program->first_block);
  ASSERT_EQ(command->opcode, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH);
  const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command =
      (const iree_hal_amdgpu_command_buffer_dispatch_command_t*)command;
  EXPECT_EQ(dispatch_command->kernarg_storage_mode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STORAGE_MODE_NATIVE_INLINE);
  EXPECT_EQ(dispatch_command->payload.binding_source_count, 1u);
  const iree_hal_amdgpu_profile_metadata_registry_t& profile_metadata =
      test_device.logical_device()->profile_metadata;
  ASSERT_EQ(profile_metadata.command_operation_record_count, 2u);
  const iree_hal_profile_command_operation_record_t* dispatch_operation =
      nullptr;
  for (iree_host_size_t i = 0;
       i < profile_metadata.command_operation_record_count; ++i) {
    const iree_hal_profile_command_operation_record_t& operation =
        profile_metadata.command_operation_records[i];
    if (operation.type == IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH) {
      dispatch_operation = &operation;
      break;
    }
  }
  ASSERT_NE(dispatch_operation, nullptr);
  EXPECT_EQ(dispatch_operation->binding_count, 2u);
  EXPECT_NE(dispatch_operation->flags &
                IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_STATIC_BINDINGS,
            0u);
  EXPECT_NE(dispatch_operation->flags &
                IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_DYNAMIC_BINDINGS,
            0u);
  const uint32_t kernarg_length =
      (uint32_t)dispatch_command->kernarg_length_qwords * 8u;
  EXPECT_EQ(
      iree_hal_amdgpu_aql_command_buffer_rodata(
          command_buffer, dispatch_command->payload_reference, kernarg_length),
      nullptr);

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
          /*buffer=*/input_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
      {
          /*buffer=*/input_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
      {
          /*buffer=*/input_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
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
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), command_buffer_signal_list,
      command_buffer, binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint32_t output_values[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  const uint32_t expected_values[4] = {13, 16, 19, 22};
  EXPECT_EQ(0, memcmp(output_values, expected_values, sizeof(expected_values)));

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}

TEST_F(HostQueueCommandBufferTest, DynamicDispatchUsesBindingTableSlots) {
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
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      input_buffer.out()));
  const uint32_t input_values[4] = {1, 2, 3, 4};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(input_buffer, /*target_offset=*/0,
                                           input_values, sizeof(input_values)));

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      output_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/1, /*offset=*/0,
          iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/3, /*offset=*/0,
          iree_hal_buffer_byte_length(output_buffer)),
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

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  ASSERT_NE(program->first_block, nullptr);
  ASSERT_EQ(program->first_block->binding_source_count, 2u);
  const iree_hal_amdgpu_command_buffer_binding_source_t* binding_sources =
      iree_hal_amdgpu_command_buffer_block_binding_sources_const(
          program->first_block);
  EXPECT_EQ(binding_sources[0].flags,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_DYNAMIC);
  EXPECT_EQ(binding_sources[0].slot, 1u);
  EXPECT_EQ(binding_sources[0].target_qword_index, 0u);
  EXPECT_EQ(binding_sources[1].flags,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_DYNAMIC);
  EXPECT_EQ(binding_sources[1].slot, 3u);
  EXPECT_EQ(binding_sources[1].target_qword_index, 1u);

  const iree_hal_amdgpu_command_buffer_command_header_t* command =
      iree_hal_amdgpu_command_buffer_block_commands_const(program->first_block);
  ASSERT_EQ(command->opcode, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH);
  const iree_hal_amdgpu_command_buffer_dispatch_command_t* dispatch_command =
      (const iree_hal_amdgpu_command_buffer_dispatch_command_t*)command;
  EXPECT_EQ(dispatch_command->kernarg_storage_mode,
            IREE_HAL_AMDGPU_COMMAND_BUFFER_KERNARG_STORAGE_MODE_NATIVE_INLINE);
  EXPECT_EQ(dispatch_command->payload.binding_source_count, 2u);

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
          /*buffer=*/input_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
      {
          /*buffer=*/input_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
      {
          /*buffer=*/input_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
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
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), command_buffer_signal_list,
      command_buffer, binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint32_t output_values[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  const uint32_t expected_values[4] = {13, 16, 19, 22};
  EXPECT_EQ(0, memcmp(output_values, expected_values, sizeof(expected_values)));

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}

TEST_F(HostQueueCommandBufferTest,
       CommandBufferRejectsCrossPhysicalDeviceQueue) {
  if (topology_.gpu_agent_count < 2) {
    GTEST_SKIP() << "fewer than two compatible GPU agents";
    return;
  }

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  ASSERT_GE(test_device.logical_device()->physical_device_count, 2u);

  iree_hal_queue_affinity_t device0_affinity = 0;
  IREE_ASSERT_OK(
      QueueAffinityForPhysicalDevice(test_device, 0, &device0_affinity));
  iree_hal_queue_affinity_t device1_affinity = 0;
  IREE_ASSERT_OK(
      QueueAffinityForPhysicalDevice(test_device, 1, &device1_affinity));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, device0_affinity,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_device_queue_execute(
          test_device.base_device(), device1_affinity,
          iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(),
          command_buffer, iree_hal_buffer_binding_table_empty(),
          IREE_HAL_EXECUTE_FLAG_NONE));
}

TEST_F(HostQueueCommandBufferTest,
       SingleBlockCommandBufferParksAndResumesUnderNotificationPressure) {
  static constexpr uint32_t kAqlCapacity = 64;
  static constexpr uint32_t kNotificationCapacity = 1;
  static constexpr uint32_t kKernargCapacity = 2 * kAqlCapacity;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.host_block_pools.command_buffer.usable_block_size =
      IREE_HAL_AMDGPU_AQL_PROGRAM_MIN_BLOCK_SIZE;
  options.host_queues.aql_capacity = kAqlCapacity;
  options.host_queues.notification_capacity = kNotificationCapacity;
  options.host_queues.kernarg_capacity = kKernargCapacity;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_buffer_t> pressure_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), pressure_buffer.out()));

  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), target_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(target_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/1, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  const uint32_t expected = 0xBD3A0001u;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_indirect_buffer_ref(/*binding=*/0, /*offset=*/0,
                                        sizeof(expected)),
      &expected, sizeof(expected), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  ASSERT_EQ(program->block_count, 1u);
  ASSERT_GT(program->max_block_aql_packet_count, 0u);

  Ref<iree_hal_semaphore_t> pressure_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), pressure_signal.out()));
  Ref<iree_hal_semaphore_t> command_buffer_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), command_buffer_signal.out()));

  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(&libhsa_), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/NULL, /*attributes=*/0, &blocker_signal));
  IREE_ASSERT_OK(EnqueueRawBlockingBarrier(queue, blocker_signal));

  uint64_t pressure_signal_value = 1;
  iree_hal_semaphore_t* pressure_signal_ptr = pressure_signal.get();
  iree_hal_semaphore_list_t pressure_signal_list = {
      /*count=*/1,
      /*semaphores=*/&pressure_signal_ptr,
      /*payload_values=*/&pressure_signal_value,
  };
  const uint32_t pressure_pattern = 0xABCD1234u;
  iree_status_t status = iree_hal_device_queue_fill(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), pressure_signal_list, pressure_buffer,
      /*target_offset=*/0, sizeof(pressure_pattern), &pressure_pattern,
      sizeof(pressure_pattern), IREE_HAL_FILL_FLAG_NONE);

  uint64_t command_buffer_signal_value = 1;
  iree_hal_semaphore_t* command_buffer_signal_ptr = command_buffer_signal.get();
  iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1,
      /*semaphores=*/&command_buffer_signal_ptr,
      /*payload_values=*/&command_buffer_signal_value,
  };
  iree_hal_buffer_binding_t binding = {
      /*buffer=*/target_buffer.get(),
      /*offset=*/0,
      /*length=*/IREE_HAL_WHOLE_BUFFER,
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      /*count=*/1,
      /*bindings=*/&binding,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_execute(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_semaphore_list_empty(), command_buffer_signal_list,
        command_buffer, binding_table, IREE_HAL_EXECUTE_FLAG_NONE);
  }
  const bool replay_parked =
      iree_status_is_ok(status) && HostQueueHasPostDrainAction(queue);

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);

  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(
        command_buffer_signal, command_buffer_signal_value,
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
  }
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));

  IREE_ASSERT_OK(status);
  EXPECT_TRUE(replay_parked);

  uint32_t actual = 0;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(target_buffer, /*offset=*/0, &actual,
                                          sizeof(actual)));
  EXPECT_EQ(actual, expected);
}

TEST_F(HostQueueCommandBufferTest,
       DeferredTransientBindingSurvivesQueuedDealloca) {
  static constexpr uint32_t kAqlCapacity = 64;
  static constexpr uint32_t kNotificationCapacity = 1;
  static constexpr uint32_t kKernargCapacity = 2 * kAqlCapacity;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.host_block_pools.command_buffer.usable_block_size =
      IREE_HAL_AMDGPU_AQL_PROGRAM_MIN_BLOCK_SIZE;
  options.host_queues.aql_capacity = kAqlCapacity;
  options.host_queues.notification_capacity = kNotificationCapacity;
  options.host_queues.kernarg_capacity = kKernargCapacity;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  Ref<iree_hal_buffer_t> pressure_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), pressure_buffer.out()));

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), output_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  Ref<iree_hal_semaphore_t> alloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), alloca_signal.out()));
  uint64_t alloca_signal_value = 1;
  iree_hal_semaphore_t* alloca_signal_ptr = alloca_signal.get();
  iree_hal_semaphore_list_t alloca_signal_list = {
      /*count=*/1,
      /*semaphores=*/&alloca_signal_ptr,
      /*payload_values=*/&alloca_signal_value,
  };
  iree_hal_buffer_t* transient_raw = NULL;
  IREE_ASSERT_OK(QueueTransientTransferBuffer(
      test_device.base_device(), alloca_signal_list, sizeof(uint32_t),
      &transient_raw));
  Ref<iree_hal_buffer_t> transient_buffer(transient_raw);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(alloca_signal, alloca_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/2, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  const uint32_t expected = 0xBD3A0002u;
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

  Ref<iree_hal_semaphore_t> pressure_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), pressure_signal.out()));
  Ref<iree_hal_semaphore_t> command_buffer_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), command_buffer_signal.out()));
  Ref<iree_hal_semaphore_t> dealloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), dealloca_signal.out()));

  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(&libhsa_), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/NULL, /*attributes=*/0, &blocker_signal));
  IREE_ASSERT_OK(EnqueueRawBlockingBarrier(queue, blocker_signal));

  uint64_t pressure_signal_value = 1;
  iree_hal_semaphore_t* pressure_signal_ptr = pressure_signal.get();
  iree_hal_semaphore_list_t pressure_signal_list = {
      /*count=*/1,
      /*semaphores=*/&pressure_signal_ptr,
      /*payload_values=*/&pressure_signal_value,
  };
  const uint32_t pressure_pattern = 0xABCD1234u;
  iree_status_t status = iree_hal_device_queue_fill(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), pressure_signal_list, pressure_buffer,
      /*target_offset=*/0, sizeof(pressure_pattern), &pressure_pattern,
      sizeof(pressure_pattern), IREE_HAL_FILL_FLAG_NONE);

  uint64_t command_buffer_signal_value = 1;
  iree_hal_semaphore_t* command_buffer_signal_ptr = command_buffer_signal.get();
  iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1,
      /*semaphores=*/&command_buffer_signal_ptr,
      /*payload_values=*/&command_buffer_signal_value,
  };
  iree_hal_buffer_binding_t bindings[2] = {
      {
          /*buffer=*/transient_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
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
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_execute(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_semaphore_list_empty(), command_buffer_signal_list,
        command_buffer, binding_table, IREE_HAL_EXECUTE_FLAG_NONE);
  }
  const bool replay_parked =
      iree_status_is_ok(status) && HostQueueHasPostDrainAction(queue);

  uint64_t dealloca_signal_value = 1;
  iree_hal_semaphore_t* dealloca_signal_ptr = dealloca_signal.get();
  iree_hal_semaphore_list_t dealloca_wait_list = {
      /*count=*/1,
      /*semaphores=*/&command_buffer_signal_ptr,
      /*payload_values=*/&command_buffer_signal_value,
  };
  iree_hal_semaphore_list_t dealloca_signal_list = {
      /*count=*/1,
      /*semaphores=*/&dealloca_signal_ptr,
      /*payload_values=*/&dealloca_signal_value,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_dealloca(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        dealloca_wait_list, dealloca_signal_list, transient_buffer,
        IREE_HAL_DEALLOCA_FLAG_NONE);
  }

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);

  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(dealloca_signal, dealloca_signal_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));

  IREE_ASSERT_OK(status);
  EXPECT_TRUE(replay_parked);

  uint32_t actual = 0;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(output_buffer, /*offset=*/0, &actual,
                                          sizeof(actual)));
  EXPECT_EQ(actual, expected);
}

TEST_F(HostQueueCommandBufferTest,
       OneShotStaticTransientBindingRecordsBeforeAllocaCommit) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), output_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  Ref<iree_hal_semaphore_t> alloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), alloca_signal.out()));
  uint64_t alloca_signal_value = 1;
  iree_hal_semaphore_t* alloca_signal_ptr = alloca_signal.get();
  iree_hal_semaphore_list_t alloca_signal_list = {
      /*count=*/1,
      /*semaphores=*/&alloca_signal_ptr,
      /*payload_values=*/&alloca_signal_value,
  };
  iree_hal_buffer_t* transient_raw = NULL;
  IREE_ASSERT_OK(QueueTransientTransferBuffer(
      test_device.base_device(), alloca_signal_list, sizeof(uint32_t),
      &transient_raw));
  Ref<iree_hal_buffer_t> transient_buffer(transient_raw);

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  const uint32_t expected = 0xBD3A0003u;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_buffer_ref(transient_buffer.get(), /*offset=*/0,
                               sizeof(expected)),
      &expected, sizeof(expected), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer,
      iree_hal_make_buffer_ref(transient_buffer.get(), /*offset=*/0,
                               sizeof(expected)),
      iree_hal_make_buffer_ref(output_buffer.get(), /*offset=*/0,
                               sizeof(expected)),
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  Ref<iree_hal_semaphore_t> command_buffer_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), command_buffer_signal.out()));
  uint64_t command_buffer_signal_value = 1;
  iree_hal_semaphore_t* command_buffer_signal_ptr = command_buffer_signal.get();
  iree_hal_semaphore_list_t command_buffer_signal_list = {
      /*count=*/1,
      /*semaphores=*/&command_buffer_signal_ptr,
      /*payload_values=*/&command_buffer_signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      alloca_signal_list, command_buffer_signal_list, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));

  Ref<iree_hal_semaphore_t> dealloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), dealloca_signal.out()));
  uint64_t dealloca_signal_value = 1;
  iree_hal_semaphore_t* dealloca_signal_ptr = dealloca_signal.get();
  iree_hal_semaphore_list_t dealloca_signal_list = {
      /*count=*/1,
      /*semaphores=*/&dealloca_signal_ptr,
      /*payload_values=*/&dealloca_signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      command_buffer_signal_list, dealloca_signal_list, transient_buffer,
      IREE_HAL_DEALLOCA_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(dealloca_signal, dealloca_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  uint32_t actual = 0;
  IREE_ASSERT_OK(iree_hal_buffer_map_read(output_buffer, /*offset=*/0, &actual,
                                          sizeof(actual)));
  EXPECT_EQ(actual, expected);
}

}  // namespace
}  // namespace iree::hal::amdgpu
