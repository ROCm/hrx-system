// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_command_buffer.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_packet.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_test_util.h"
#include "iree/hal/drivers/amdgpu/hostcall_provider.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/pm4_command_buffer.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

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

// Installs only the stable address consumed by dispatch recording. Provider
// resource ownership and listener lifetime are covered by
// hostcall_provider_test.
class ScopedHostcallBufferAddress {
 public:
  ScopedHostcallBufferAddress(
      iree_hal_amdgpu_physical_device_t* physical_device,
      uint64_t device_address)
      : physical_device_(physical_device) {
    EXPECT_EQ(physical_device_->hostcall_provider_state, nullptr);
    state_.device_address = device_address;
    physical_device_->hostcall_provider_state = &state_;
    for (iree_host_size_t i = 0; i < physical_device_->host_queue_count; ++i) {
      EXPECT_EQ(physical_device_->host_queues[i].hostcall_buffer, nullptr);
      physical_device_->host_queues[i].hostcall_buffer =
          reinterpret_cast<void*>(device_address);
    }
  }

  ~ScopedHostcallBufferAddress() {
    for (iree_host_size_t i = 0; i < physical_device_->host_queue_count; ++i) {
      physical_device_->host_queues[i].hostcall_buffer = nullptr;
    }
    physical_device_->hostcall_provider_state = nullptr;
  }

 private:
  iree_hal_amdgpu_physical_device_t* physical_device_;
  iree_hal_amdgpu_hostcall_provider_state_t state_ = {};
};

static iree_status_t LoadHostcallBufferExecutable(
    iree_hal_device_t* device, const iree_hal_queue_family_t* queue_family,
    iree_hal_executable_t** out_executable) {
  return LoadCtsExecutable(device, queue_family,
                           IREE_SV("hostcall_buffer_test.bin"), out_executable);
}

static iree_hal_buffer_ref_list_t MakeHostcallBufferBindingList(
    iree_hal_buffer_t* output_buffer, iree_hal_buffer_ref_t* out_binding) {
  *out_binding =
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0, sizeof(uint64_t));
  return {
      /*.count=*/1,
      /*.values=*/out_binding,
  };
}

static iree_status_t DispatchHostcallBufferDirect(
    iree_hal_device_t* device, iree_hal_queue_t* queue,
    iree_hal_executable_t* executable, iree_hal_buffer_t* output_buffer,
    uint64_t* out_device_address) {
  Ref<iree_hal_semaphore_t> signal;
  IREE_RETURN_IF_ERROR(CreateSemaphore(device, signal.out()));
  iree_hal_semaphore_t* signal_ptr = signal.get();
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&signal_ptr,
      /*.payload_values=*/&signal_value,
  };
  iree_hal_buffer_ref_t binding;
  IREE_RETURN_IF_ERROR(iree_hal_queue_dispatch(
      queue, iree_hal_semaphore_list_empty(), signal_list, executable,
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(),
      MakeHostcallBufferBindingList(output_buffer, &binding),
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_wait(signal, signal_value,
                                               iree_infinite_timeout(),
                                               IREE_ASYNC_WAIT_FLAG_NONE));
  return iree_hal_buffer_map_read(output_buffer, /*offset=*/0,
                                  out_device_address,
                                  sizeof(*out_device_address));
}

static iree_status_t RecordHostcallBufferCommandBuffer(
    iree_hal_device_t* device, iree_hal_queue_t* queue,
    iree_hal_executable_t* executable, iree_hal_buffer_t* output_buffer,
    iree_hal_command_buffer_t** out_command_buffer) {
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
      device, iree_hal_queue_family(queue),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer));
  iree_hal_buffer_ref_t binding;
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_dispatch(
      command_buffer, executable, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(),
      MakeHostcallBufferBindingList(output_buffer, &binding),
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_end(command_buffer));
  *out_command_buffer = command_buffer.release();
  return iree_ok_status();
}

static iree_status_t ExecuteHostcallBufferCommandBuffer(
    iree_hal_queue_t* queue, iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_t* output_buffer, iree_hal_semaphore_t* signal,
    uint64_t signal_value, uint64_t* out_device_address) {
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_map_zero(output_buffer, /*offset=*/0, sizeof(uint64_t)));
  iree_hal_semaphore_t* signal_ptr = signal;
  iree_hal_semaphore_list_t signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&signal_ptr,
      /*.payload_values=*/&signal_value,
  };
  IREE_RETURN_IF_ERROR(iree_hal_queue_execute(
      queue, iree_hal_semaphore_list_empty(), signal_list, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_wait(signal, signal_value,
                                               iree_infinite_timeout(),
                                               IREE_ASYNC_WAIT_FLAG_NONE));
  return iree_hal_buffer_map_read(output_buffer, /*offset=*/0,
                                  out_device_address,
                                  sizeof(*out_device_address));
}

static const uint32_t* FindPm4DispatchDirectPacket(
    const iree_hal_amdgpu_pm4_program_t* pm4_program,
    uint32_t dispatch_direct_ordinal) {
  const uint32_t dispatch_direct_header =
      iree_hal_amdgpu_pm4_make_compute_header(
          IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_DISPATCH_DIRECT,
          IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT);
  for (uint32_t i = 0; i + IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT <=
                       pm4_program->dword_count;
       ++i) {
    if (pm4_program->dwords[i] != dispatch_direct_header) continue;
    if (dispatch_direct_ordinal == 0) return &pm4_program->dwords[i];
    --dispatch_direct_ordinal;
  }
  return nullptr;
}

#if IREE_FILE_IO_ENABLE
static iree_status_t CreateTempFileWithContents(
    const std::vector<uint8_t>& data, iree::testing::TempFilePath* out_path) {
  *out_path =
      iree::testing::TempFilePath("iree_hal_amdgpu_host_queue_command_buffer");
  return iree_io_file_contents_write(
      out_path->path_view(),
      iree_make_const_byte_span(data.data(), data.size()),
      iree_allocator_system());
}

static iree_status_t ImportNativeFile(iree_hal_device_t* device,
                                      const iree::testing::TempFilePath& path,
                                      iree_hal_memory_access_t access,
                                      iree_hal_file_t** out_file) {
  iree_io_file_mode_t mode = IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_ASYNC;
  if (iree_all_bits_set(access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
    mode |= IREE_IO_FILE_MODE_WRITE;
  }
  iree_io_file_handle_t* handle = NULL;
  IREE_RETURN_IF_ERROR(iree_io_file_handle_open(
      mode, path.path_view(), iree_allocator_system(), &handle));
  iree_status_t status =
      iree_hal_file_import(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, access,
                           handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, out_file);
  iree_io_file_handle_release(handle);
  return status;
}
#endif  // IREE_FILE_IO_ENABLE

static iree_status_t QueueHostVisibleDispatchTransientBuffer(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_list, iree_device_size_t buffer_size,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                 IREE_HAL_BUFFER_USAGE_STORAGE |
                 IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.queue_family_affinity = iree_hal_make_queue_family_affinity(
      iree_hal_queue_family_ordinal(iree_hal_queue_family(&queue->base)));
  iree_hal_pool_t* pool =
      iree_hal_pool_set_select(queue->default_pool_set, params, buffer_size);
  if (!pool) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no queue pool supports the transient request");
  }
  const iree_hal_pool_reservation_request_t request = {
      .params = params,
      .allocation_size = buffer_size,
  };
  return iree_hal_queue_alloca(&queue->base, iree_hal_semaphore_list_empty(),
                               signal_list, pool,
                               /*request_count=*/1, &request, out_buffer);
}

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
       TsanAssignmentPlanForcesBarrierWhenShadowSlotsWouldOverlap) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;
  options.tsan.shadow_slot_count = 1;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.InitializeWithRuntimeFeatures(
      &options, &libhsa_, &topology_, IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_TSAN,
      host_allocator_));

  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      iree_make_cstring_view("tsan_executable_test.bin"), &executable));

  Ref<iree_hal_buffer_t> input_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint64_t),
      input_buffer.out()));
  Ref<iree_hal_buffer_t> output_buffer0;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/12 * sizeof(uint64_t),
      output_buffer0.out()));
  Ref<iree_hal_buffer_t> output_buffer1;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/12 * sizeof(uint64_t),
      output_buffer1.out()));

  const uint32_t constant_values[2] = {0x5453414Eu, 0x43464721u};
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(constant_values, sizeof(constant_values));
  iree_hal_buffer_ref_t binding_refs0[2] = {
      iree_hal_make_buffer_ref(output_buffer0, /*offset=*/0,
                               iree_hal_buffer_byte_length(output_buffer0)),
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
  };
  iree_hal_buffer_ref_t binding_refs1[2] = {
      iree_hal_make_buffer_ref(output_buffer1, /*offset=*/0,
                               iree_hal_buffer_byte_length(output_buffer1)),
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
  };
  const iree_hal_buffer_ref_list_t bindings0 = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs0),
      /*.values=*/binding_refs0,
  };
  const iree_hal_buffer_ref_list_t bindings1 = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs1),
      /*.values=*/binding_refs1,
  };

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings0,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants, bindings1,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  ASSERT_NE(program, nullptr);
  ASSERT_NE(program->first_block, nullptr);
  ASSERT_EQ(program->block_count, 1u);
  const iree_hal_amdgpu_command_buffer_block_header_t* block =
      program->first_block;
  EXPECT_EQ(block->dispatch_count, 2u);
  EXPECT_EQ(block->aql_packet_count, 2u);

  const iree_hal_amdgpu_command_buffer_command_header_t* command =
      iree_hal_amdgpu_command_buffer_block_commands_const(block);
  ASSERT_EQ(command->opcode, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH);
  EXPECT_FALSE(iree_any_bit_set(
      command->flags, IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER));
  command = iree_hal_amdgpu_command_buffer_command_next_const(command);
  ASSERT_EQ(command->opcode, IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH);
  EXPECT_TRUE(iree_any_bit_set(
      command->flags, IREE_HAL_AMDGPU_COMMAND_BUFFER_COMMAND_FLAG_HAS_BARRIER));

  iree_hal_executable_release(executable);
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

  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable));

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
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
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
  IREE_ASSERT_OK(iree_hal_queue_execute(
      test_device.queue(), iree_hal_semaphore_list_empty(),
      command_buffer_signal_list, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint32_t output_values[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  const uint32_t expected_values[4] = {49, 58, 67, 76};
  EXPECT_EQ(0, memcmp(output_values, expected_values, sizeof(expected_values)));

  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  Ref<iree_hal_command_buffer_t> one_shot_command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
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
  IREE_ASSERT_OK(iree_hal_queue_execute(
      test_device.queue(), iree_hal_semaphore_list_empty(),
      command_buffer_signal_list, one_shot_command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  memset(output_values, 0, sizeof(output_values));
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  EXPECT_EQ(0, memcmp(output_values, expected_values, sizeof(expected_values)));

  iree_hal_executable_release(executable);
}

TEST_F(HostQueueCommandBufferTest,
       DirectDispatchPreservesNestedPointersInNativeKernargs) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable));

  const iree_hal_executable_function_t normal_function =
      iree_hal_executable_function_from_index(0);
  iree_hal_executable_function_info_t normal_function_info = {};
  IREE_ASSERT_OK(iree_hal_executable_function_info(executable, normal_function,
                                                   &normal_function_info));
  ASSERT_EQ(4u, normal_function_info.parameter_count);
  std::vector<iree_hal_executable_function_parameter_t> normal_parameters(
      normal_function_info.parameter_count);
  IREE_ASSERT_OK(iree_hal_executable_function_parameters(
      executable, normal_function, normal_parameters.size(),
      normal_parameters.data()));
  ASSERT_EQ(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING,
            normal_parameters[0].type);
  ASSERT_EQ(0u, normal_parameters[0].offset);
  ASSERT_TRUE(iree_all_bits_set(
      normal_parameters[0].flags,
      IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NATIVE_ABI_OFFSET));
  EXPECT_EQ(0u, normal_parameters[0].native_abi_offset);
  ASSERT_EQ(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING,
            normal_parameters[1].type);
  ASSERT_EQ(1u, normal_parameters[1].offset);
  ASSERT_TRUE(iree_all_bits_set(
      normal_parameters[1].flags,
      IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NATIVE_ABI_OFFSET));
  EXPECT_EQ(sizeof(void*), normal_parameters[1].native_abi_offset);

  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable, IREE_SV("command_buffer_dispatch_nested_pointers_test"),
      &function));
  iree_hal_executable_function_info_t function_info = {};
  IREE_ASSERT_OK(
      iree_hal_executable_function_info(executable, function, &function_info));
  ASSERT_EQ(3u, function_info.parameter_count);
  std::vector<iree_hal_executable_function_parameter_t> parameters(
      function_info.parameter_count);
  IREE_ASSERT_OK(iree_hal_executable_function_parameters(
      executable, function, parameters.size(), parameters.data()));

  // The compiler leaves by-value parameter names empty. Their source ABI has
  // one pointer-pair record followed by two uint32_t values, so classify the
  // reflected records from that shape and their native layout instead.
  const iree_hal_executable_function_parameter_t* pointers_parameter = nullptr;
  std::vector<const iree_hal_executable_function_parameter_t*>
      scalar_parameters;
  for (const iree_hal_executable_function_parameter_t& parameter : parameters) {
    ASSERT_EQ(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
              parameter.type);
    ASSERT_TRUE(iree_all_bits_set(
        parameter.flags,
        IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NATIVE_ABI_OFFSET));
    if (parameter.size == 2 * sizeof(void*)) {
      ASSERT_EQ(nullptr, pointers_parameter);
      pointers_parameter = &parameter;
    } else if (parameter.size == sizeof(uint32_t)) {
      scalar_parameters.push_back(&parameter);
    } else {
      ADD_FAILURE() << "unexpected native parameter size: " << parameter.size;
    }
  }
  ASSERT_EQ(2u, scalar_parameters.size());
  std::sort(scalar_parameters.begin(), scalar_parameters.end(),
            [](const iree_hal_executable_function_parameter_t* lhs,
               const iree_hal_executable_function_parameter_t* rhs) {
              return lhs->native_abi_offset < rhs->native_abi_offset;
            });
  const iree_hal_executable_function_parameter_t* scale_parameter =
      scalar_parameters[0];
  const iree_hal_executable_function_parameter_t* offset_parameter =
      scalar_parameters[1];
  ASSERT_NE(nullptr, pointers_parameter);
  ASSERT_EQ(2 * sizeof(void*), pointers_parameter->size);
  ASSERT_EQ(sizeof(uint32_t), scale_parameter->size);
  ASSERT_EQ(sizeof(uint32_t), offset_parameter->size);

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

  const auto device_pointer = [](iree_hal_buffer_t* buffer) -> void* {
    iree_hal_buffer_t* allocated_buffer =
        iree_hal_buffer_allocated_buffer(buffer);
    void* base_pointer =
        iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
    if (!base_pointer) return nullptr;
    return (void*)((uintptr_t)base_pointer +
                   iree_hal_buffer_byte_offset(buffer));
  };
  struct nested_pointer_args_t {
    // Input data read by the kernel.
    uint32_t* input;
    // Output data written by the kernel.
    uint32_t* output;
  };
  const nested_pointer_args_t pointers = {
      /*.input=*/static_cast<uint32_t*>(device_pointer(input_buffer)),
      /*.output=*/static_cast<uint32_t*>(device_pointer(output_buffer)),
  };
  ASSERT_NE(nullptr, pointers.input);
  ASSERT_NE(nullptr, pointers.output);

  const size_t explicit_argument_size = std::max(
      (size_t)pointers_parameter->native_abi_offset + sizeof(pointers),
      std::max((size_t)scale_parameter->native_abi_offset + sizeof(uint32_t),
               (size_t)offset_parameter->native_abi_offset + sizeof(uint32_t)));
  ASSERT_GT(explicit_argument_size, 0u);

  Ref<iree_hal_semaphore_t> dispatch_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), dispatch_signal.out()));
  iree_hal_semaphore_t* dispatch_signal_ptr = dispatch_signal.get();

  // Normal HAL dispatch consumes the same executable through ordinary binding
  // ordinals and dense constants. Native ABI offsets remain metadata only.
  iree_hal_buffer_ref_t normal_binding_refs[2] = {
      iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer)),
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(output_buffer)),
  };
  const iree_hal_buffer_ref_list_t normal_bindings = {
      /*.count=*/IREE_ARRAYSIZE(normal_binding_refs),
      /*.values=*/normal_binding_refs,
  };
  const uint32_t normal_constants[2] = {3, 10};
  uint64_t normal_signal_value = 1;
  const iree_hal_semaphore_list_t normal_signal_semaphores = {
      /*.count=*/1,
      /*.semaphores=*/&dispatch_signal_ptr,
      /*.payload_values=*/&normal_signal_value,
  };
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      test_device.queue(), iree_hal_semaphore_list_empty(),
      normal_signal_semaphores, executable, normal_function,
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_make_const_byte_span(normal_constants, sizeof(normal_constants)),
      normal_bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(dispatch_signal, normal_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  uint32_t output_values[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  const uint32_t normal_expected_values[4] = {13, 16, 19, 22};
  EXPECT_EQ(0, memcmp(output_values, normal_expected_values,
                      sizeof(normal_expected_values)));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  const auto dispatch_with_temporary_arguments =
      [&](uint32_t scale, uint32_t offset, size_t padding,
          uint64_t signal_value) -> iree_status_t {
    std::vector<uint8_t> native_arguments(explicit_argument_size + padding,
                                          0xCD);
    memcpy(native_arguments.data() + pointers_parameter->native_abi_offset,
           &pointers, sizeof(pointers));
    memcpy(native_arguments.data() + scale_parameter->native_abi_offset, &scale,
           sizeof(scale));
    memcpy(native_arguments.data() + offset_parameter->native_abi_offset,
           &offset, sizeof(offset));
    const iree_hal_semaphore_list_t signal_semaphores = {
        /*.count=*/1,
        /*.semaphores=*/&dispatch_signal_ptr,
        /*.payload_values=*/&signal_value,
    };
    return iree_hal_queue_dispatch(
        test_device.queue(), iree_hal_semaphore_list_empty(), signal_semaphores,
        executable, function, iree_hal_make_static_dispatch_config(1, 1, 1),
        iree_make_const_byte_span(native_arguments.data(),
                                  native_arguments.size()),
        iree_hal_buffer_ref_list_empty(),
        IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS);
  };

  // The queue must retain a private copy before this temporary byte image is
  // destroyed. The padded tail is deliberately ignored by the native layout.
  IREE_ASSERT_OK(dispatch_with_temporary_arguments(/*scale=*/3, /*offset=*/10,
                                                   /*padding=*/0,
                                                   /*signal_value=*/2));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(dispatch_signal, /*value=*/2,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  const uint32_t first_expected_values[4] = {13, 16, 19, 22};
  EXPECT_EQ(0, memcmp(output_values, first_expected_values,
                      sizeof(first_expected_values)));

  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));
  IREE_ASSERT_OK(dispatch_with_temporary_arguments(/*scale=*/4, /*offset=*/1,
                                                   /*padding=*/16,
                                                   /*signal_value=*/3));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(dispatch_signal, /*value=*/3,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  const uint32_t second_expected_values[4] = {5, 9, 13, 17};
  EXPECT_EQ(0, memcmp(output_values, second_expected_values,
                      sizeof(second_expected_values)));

  iree_hal_executable_release(executable);
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
  IREE_ASSERT_OK(iree_hal_queue_execute(
      test_device.queue(), iree_hal_semaphore_list_empty(),
      command_buffer_signal_list, fixture.command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectTwoDispatchOutputs(fixture);

  iree_hal_command_buffer_t* one_shot_command_buffer = nullptr;
  iree_status_t one_shot_status = iree_hal_command_buffer_create(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      /*binding_capacity=*/0, &one_shot_command_buffer);
  EXPECT_EQ(iree_status_code(one_shot_status), IREE_STATUS_UNIMPLEMENTED);
  iree_status_free(one_shot_status);
  iree_hal_command_buffer_release(one_shot_command_buffer);
}

TEST_F(HostQueueCommandBufferTest,
       HostcallAddressIsBakedIntoDirectAndAqlDispatches) {
  constexpr uint64_t kHostcallBufferAddress = 0x123456789ABC0000ull;
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AQL;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(LoadHostcallBufferExecutable(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      executable.out()));
  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), sizeof(uint64_t), output_buffer.out()));

  uint64_t null_direct_address = UINT64_MAX;
  IREE_ASSERT_OK(DispatchHostcallBufferDirect(
      test_device.base_device(), test_device.queue(), executable, output_buffer,
      &null_direct_address));
  EXPECT_EQ(null_direct_address, 0u);

  Ref<iree_hal_command_buffer_t> null_command_buffer;
  IREE_ASSERT_OK(RecordHostcallBufferCommandBuffer(
      test_device.base_device(), test_device.queue(), executable, output_buffer,
      null_command_buffer.out()));
  ASSERT_TRUE(iree_hal_amdgpu_aql_command_buffer_isa(null_command_buffer));
  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  uint64_t null_replay_address = UINT64_MAX;
  IREE_ASSERT_OK(ExecuteHostcallBufferCommandBuffer(
      test_device.queue(), null_command_buffer, output_buffer, signal,
      /*signal_value=*/1, &null_replay_address));
  EXPECT_EQ(null_replay_address, 0u);

  ScopedHostcallBufferAddress hostcall_buffer(
      test_device.logical_device()->physical_devices[0],
      kHostcallBufferAddress);
  uint64_t direct_address = 0;
  IREE_ASSERT_OK(DispatchHostcallBufferDirect(test_device.base_device(),
                                              test_device.queue(), executable,
                                              output_buffer, &direct_address));
  EXPECT_EQ(direct_address, kHostcallBufferAddress);

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(RecordHostcallBufferCommandBuffer(
      test_device.base_device(), test_device.queue(), executable, output_buffer,
      command_buffer.out()));
  ASSERT_TRUE(iree_hal_amdgpu_aql_command_buffer_isa(command_buffer));
  for (uint64_t replay = 1; replay <= 2; ++replay) {
    uint64_t replay_address = 0;
    IREE_ASSERT_OK(ExecuteHostcallBufferCommandBuffer(
        test_device.queue(), command_buffer, output_buffer, signal, replay + 1,
        &replay_address));
    EXPECT_EQ(replay_address, kHostcallBufferAddress);
  }
}

TEST_F(HostQueueCommandBufferTest, HostcallAddressIsBakedIntoPm4Dispatches) {
  constexpr uint64_t kHostcallBufferAddress = 0x123456789ABC0000ull;
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
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(LoadHostcallBufferExecutable(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      executable.out()));
  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), sizeof(uint64_t), output_buffer.out()));
  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));

  Ref<iree_hal_command_buffer_t> null_command_buffer;
  IREE_ASSERT_OK(RecordHostcallBufferCommandBuffer(
      test_device.base_device(), test_device.queue(), executable, output_buffer,
      null_command_buffer.out()));
  ASSERT_TRUE(iree_hal_amdgpu_pm4_command_buffer_isa(null_command_buffer));
  uint64_t null_replay_address = UINT64_MAX;
  IREE_ASSERT_OK(ExecuteHostcallBufferCommandBuffer(
      test_device.queue(), null_command_buffer, output_buffer, signal,
      /*signal_value=*/1, &null_replay_address));
  EXPECT_EQ(null_replay_address, 0u);

  ScopedHostcallBufferAddress hostcall_buffer(physical_device,
                                              kHostcallBufferAddress);
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(RecordHostcallBufferCommandBuffer(
      test_device.base_device(), test_device.queue(), executable, output_buffer,
      command_buffer.out()));
  ASSERT_TRUE(iree_hal_amdgpu_pm4_command_buffer_isa(command_buffer));

  for (uint64_t replay = 1; replay <= 2; ++replay) {
    uint64_t replay_address = 0;
    IREE_ASSERT_OK(ExecuteHostcallBufferCommandBuffer(
        test_device.queue(), command_buffer, output_buffer, signal, replay + 1,
        &replay_address));
    EXPECT_EQ(replay_address, kHostcallBufferAddress);
  }
}

TEST_F(HostQueueCommandBufferTest,
       AutoCommandBufferModeSelectsPm4WhenAvailable) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AUTO;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
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

  Ref<iree_hal_command_buffer_t> transfer_command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, IREE_HAL_COMMAND_CATEGORY_TRANSFER,
      /*binding_capacity=*/0, transfer_command_buffer.out()));
  EXPECT_EQ(pm4_supported,
            iree_hal_amdgpu_pm4_command_buffer_isa(transfer_command_buffer));
  EXPECT_NE(pm4_supported,
            iree_hal_amdgpu_aql_command_buffer_isa(transfer_command_buffer));
}

TEST_F(HostQueueCommandBufferTest,
       AutoCommandBufferModeFallsBackToAqlForUnsupportedRequest) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AUTO;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      /*binding_capacity=*/0, command_buffer.out()));
  EXPECT_TRUE(iree_hal_amdgpu_aql_command_buffer_isa(command_buffer));
  EXPECT_FALSE(iree_hal_amdgpu_pm4_command_buffer_isa(command_buffer));
}

TEST_F(HostQueueCommandBufferTest,
       Pm4DispatchDirectUsesThreadDimensionsForMultiWorkgroupDispatch) {
  constexpr uint32_t kWorkgroupCount = 32u;

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

  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      iree_make_cstring_view("command_buffer_dispatch_multi_workgroup_test."
                             "bin"),
      &executable));

  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor = nullptr;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_queue_ordinal(
          executable, iree_hal_executable_function_from_index(0),
          /*queue_ordinal=*/0, &descriptor));
  ASSERT_NE(descriptor, nullptr);

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), kWorkgroupCount * sizeof(uint32_t),
      output_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  iree_hal_buffer_ref_t binding_refs[1] = {
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                               iree_hal_buffer_byte_length(output_buffer)),
  };
  const iree_hal_buffer_ref_list_t dispatch_bindings = {
      /*count=*/IREE_ARRAYSIZE(binding_refs),
      /*values=*/binding_refs,
  };

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(kWorkgroupCount, 1, 1),
      iree_const_byte_span_empty(), dispatch_bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  ASSERT_TRUE(iree_hal_amdgpu_pm4_command_buffer_isa(command_buffer));
  const iree_hal_amdgpu_pm4_program_t* pm4_program =
      iree_hal_amdgpu_pm4_command_buffer_program(command_buffer);
  ASSERT_NE(pm4_program, nullptr);
  const uint32_t* dispatch_direct =
      FindPm4DispatchDirectPacket(pm4_program, /*dispatch_direct_ordinal=*/0);
  ASSERT_NE(dispatch_direct, nullptr);
  EXPECT_EQ(dispatch_direct[1],
            kWorkgroupCount * descriptor->kernel_args.workgroup_size[0]);
  EXPECT_EQ(dispatch_direct[2], descriptor->kernel_args.workgroup_size[1]);
  EXPECT_EQ(dispatch_direct[3], descriptor->kernel_args.workgroup_size[2]);
  EXPECT_TRUE(iree_all_bits_set(
      dispatch_direct[4],
      IREE_HAL_AMDGPU_PM4_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS));

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
  IREE_ASSERT_OK(iree_hal_queue_execute(
      test_device.queue(), iree_hal_semaphore_list_empty(),
      command_buffer_signal_list, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint32_t output_values[kWorkgroupCount] = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  for (uint32_t i = 0; i < kWorkgroupCount; ++i) {
    EXPECT_EQ(output_values[i], i);
  }

  iree_hal_executable_release(executable);
}

TEST_F(HostQueueCommandBufferTest,
       ExplicitPm4CommandBufferModeRejectsEmptyCategories) {
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
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, /*command_categories=*/0,
      /*binding_capacity=*/0, &command_buffer);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  iree_status_free(status);
  iree_hal_command_buffer_release(command_buffer);
}

TEST_F(HostQueueCommandBufferTest,
       Pm4TransferCommandsExecuteStaticAndDynamicBindings) {
  // Dynamic references carry alignment 1, selecting the unaligned block-copy
  // kernel across a transfer large enough to exercise its grid-stride loop.
  constexpr iree_device_size_t kBufferLength = 2 * 1024 * 1024 + 36;
  constexpr iree_device_size_t kDynamicFillOffset = 5;
  constexpr iree_device_size_t kDynamicFillLength = 123;
  constexpr iree_device_size_t kDynamicHalfwordFillOffset = 202;
  constexpr iree_device_size_t kDynamicHalfwordFillLength = 26;
  constexpr iree_device_size_t kStaticUpdateOffset = 31;
  constexpr iree_device_size_t kDynamicUpdateOffset = 47;
  constexpr iree_host_size_t kUpdateSourceOffset = 7;
  constexpr iree_device_size_t kUpdateLength = 37;

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

  Ref<iree_hal_buffer_t> static_source;
  Ref<iree_hal_buffer_t> static_target;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), kBufferLength, static_source.out()));
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), kBufferLength, static_target.out()));
  Ref<iree_hal_buffer_t> dynamic_sources[2];
  Ref<iree_hal_buffer_t> dynamic_targets[2];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(dynamic_sources); ++i) {
    IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
        test_device.allocator(), kBufferLength, dynamic_sources[i].out()));
    IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
        test_device.allocator(), kBufferLength, dynamic_targets[i].out()));
  }
  Ref<iree_hal_buffer_t> maximum_update_target;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE,
      maximum_update_target.out()));

  const uint32_t static_fill_pattern = 0x44332211u;
  const uint8_t dynamic_fill_pattern = 0xA5u;
  const uint16_t dynamic_halfword_fill_pattern = 0x5AA5u;
  uint8_t update_source[64];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(update_source); ++i) {
    update_source[i] = (uint8_t)(0x80u + i);
  }
  std::vector<uint8_t> expected_update(
      update_source + kUpdateSourceOffset,
      update_source + kUpdateSourceOffset + kUpdateLength);
  std::vector<uint8_t> maximum_update_source(
      IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE);
  for (iree_host_size_t i = 0; i < maximum_update_source.size(); ++i) {
    maximum_update_source[i] = (uint8_t)(i * 29u + 3u);
  }
  const std::vector<uint8_t> expected_maximum_update = maximum_update_source;

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, IREE_HAL_COMMAND_CATEGORY_TRANSFER,
      /*binding_capacity=*/2, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/0, kDynamicFillOffset, kDynamicFillLength),
      &dynamic_fill_pattern, sizeof(dynamic_fill_pattern),
      IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0,
                                        kDynamicHalfwordFillOffset,
                                        kDynamicHalfwordFillLength),
      &dynamic_halfword_fill_pattern, sizeof(dynamic_halfword_fill_pattern),
      IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_buffer_ref(static_source, /*offset=*/0, kBufferLength),
      &static_fill_pattern, sizeof(static_fill_pattern),
      IREE_HAL_FILL_FLAG_NONE));

  const iree_hal_memory_barrier_t transfer_barrier = {
      /*.source_scope=*/IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
      /*.target_scope=*/IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
          IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer, IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_STAGE_TRANSFER, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/1, &transfer_barrier,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer,
      iree_hal_make_buffer_ref(static_source, /*offset=*/0, kBufferLength),
      iree_hal_make_buffer_ref(static_target, /*offset=*/0, kBufferLength),
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer,
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/0,
                                        kBufferLength),
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/1, /*offset=*/0,
                                        kBufferLength),
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer, IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_STAGE_TRANSFER, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/1, &transfer_barrier,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      command_buffer, update_source, kUpdateSourceOffset,
      iree_hal_make_buffer_ref(static_target, kStaticUpdateOffset,
                               kUpdateLength),
      IREE_HAL_UPDATE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      command_buffer, update_source, kUpdateSourceOffset,
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/1, kDynamicUpdateOffset, kUpdateLength),
      IREE_HAL_UPDATE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      command_buffer, maximum_update_source.data(), /*source_offset=*/0,
      iree_hal_make_buffer_ref(maximum_update_target, /*offset=*/0,
                               IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE),
      IREE_HAL_UPDATE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  memset(update_source, 0, sizeof(update_source));
  std::fill(maximum_update_source.begin(), maximum_update_source.end(), 0);
  ASSERT_TRUE(iree_hal_amdgpu_pm4_command_buffer_isa(command_buffer));
  const iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t* fixup_plan =
      iree_hal_amdgpu_pm4_command_buffer_fixup_plan(command_buffer);
  ASSERT_NE(fixup_plan->entries, nullptr);
  EXPECT_GE(fixup_plan->entry_count, 4u);

  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  iree_hal_semaphore_t* signal_ptr = signal.get();
  for (iree_host_size_t replay = 0; replay < IREE_ARRAYSIZE(dynamic_sources);
       ++replay) {
    std::vector<uint8_t> source_values(kBufferLength);
    for (iree_host_size_t i = 0; i < source_values.size(); ++i) {
      source_values[i] = (uint8_t)(i * 13u + replay * 17u);
    }
    IREE_ASSERT_OK(
        iree_hal_buffer_map_write(dynamic_sources[replay], /*target_offset=*/0,
                                  source_values.data(), source_values.size()));
    IREE_ASSERT_OK(iree_hal_buffer_map_zero(
        dynamic_targets[replay], /*offset=*/0, IREE_HAL_WHOLE_BUFFER));
    IREE_ASSERT_OK(iree_hal_buffer_map_zero(static_target, /*offset=*/0,
                                            IREE_HAL_WHOLE_BUFFER));
    IREE_ASSERT_OK(iree_hal_buffer_map_zero(maximum_update_target, /*offset=*/0,
                                            IREE_HAL_WHOLE_BUFFER));

    iree_hal_buffer_binding_t bindings[2] = {
        {
            /*.buffer=*/dynamic_sources[replay].get(),
            /*.offset=*/0,
            /*.length=*/IREE_HAL_WHOLE_BUFFER,
        },
        {
            /*.buffer=*/dynamic_targets[replay].get(),
            /*.offset=*/0,
            /*.length=*/IREE_HAL_WHOLE_BUFFER,
        },
    };
    const iree_hal_buffer_binding_table_t binding_table = {
        /*.count=*/IREE_ARRAYSIZE(bindings),
        /*.bindings=*/bindings,
    };
    uint64_t signal_value = replay + 1;
    const iree_hal_semaphore_list_t signal_list = {
        /*.count=*/1,
        /*.semaphores=*/&signal_ptr,
        /*.payload_values=*/&signal_value,
    };
    IREE_ASSERT_OK(iree_hal_queue_execute(
        test_device.queue(), iree_hal_semaphore_list_empty(), signal_list,
        command_buffer, binding_table, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(signal, signal_value,
                                           iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));

    std::fill(source_values.begin() + kDynamicFillOffset,
              source_values.begin() + kDynamicFillOffset + kDynamicFillLength,
              dynamic_fill_pattern);
    const uint8_t* dynamic_halfword_fill_pattern_bytes =
        reinterpret_cast<const uint8_t*>(&dynamic_halfword_fill_pattern);
    for (iree_device_size_t i = 0; i < kDynamicHalfwordFillLength; ++i) {
      source_values[kDynamicHalfwordFillOffset + i] =
          dynamic_halfword_fill_pattern_bytes
              [i % sizeof(dynamic_halfword_fill_pattern)];
    }
    std::vector<uint8_t> actual_source_values(kBufferLength);
    IREE_ASSERT_OK(iree_hal_buffer_map_read(
        dynamic_sources[replay], /*source_offset=*/0,
        actual_source_values.data(), actual_source_values.size()));
    EXPECT_EQ(actual_source_values, source_values)
        << "dynamic source replay " << replay;

    std::copy(expected_update.begin(), expected_update.end(),
              source_values.begin() + kDynamicUpdateOffset);
    std::vector<uint8_t> actual_values(kBufferLength);
    IREE_ASSERT_OK(
        iree_hal_buffer_map_read(dynamic_targets[replay], /*source_offset=*/0,
                                 actual_values.data(), actual_values.size()));
    EXPECT_EQ(actual_values, source_values)
        << "dynamic target replay " << replay;

    std::vector<uint8_t> expected_static(kBufferLength);
    const uint8_t* static_pattern_bytes =
        reinterpret_cast<const uint8_t*>(&static_fill_pattern);
    for (iree_host_size_t i = 0; i < expected_static.size(); ++i) {
      expected_static[i] =
          static_pattern_bytes[i % sizeof(static_fill_pattern)];
    }
    std::copy(expected_update.begin(), expected_update.end(),
              expected_static.begin() + kStaticUpdateOffset);
    IREE_ASSERT_OK(iree_hal_buffer_map_read(static_target, /*source_offset=*/0,
                                            actual_values.data(),
                                            actual_values.size()));
    EXPECT_EQ(actual_values, expected_static);

    std::vector<uint8_t> actual_maximum_update(
        IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE);
    IREE_ASSERT_OK(iree_hal_buffer_map_read(
        maximum_update_target, /*source_offset=*/0,
        actual_maximum_update.data(), actual_maximum_update.size()));
    EXPECT_EQ(actual_maximum_update, expected_maximum_update);
  }
}

TEST_F(HostQueueCommandBufferTest,
       AutoMixedDynamicDispatchUsesDefaultUploadRing) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AUTO;
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

  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable));

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
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH | IREE_HAL_COMMAND_CATEGORY_TRANSFER,
      /*binding_capacity=*/4, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1), constants,
      dispatch_bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  const iree_hal_memory_barrier_t dispatch_to_transfer_barrier = {
      /*.source_scope=*/IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      /*.target_scope=*/IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_TRANSFER, IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/1, &dispatch_to_transfer_barrier,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));
  const uint32_t fill_pattern = 0xA1B2C3D4u;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/3, /*offset=*/0,
                                        sizeof(fill_pattern)),
      &fill_pattern, sizeof(fill_pattern), IREE_HAL_FILL_FLAG_NONE));
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
      {0},
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
  IREE_ASSERT_OK(iree_hal_queue_execute(
      test_device.queue(), iree_hal_semaphore_list_empty(),
      command_buffer_signal_list, command_buffer, binding_table,
      IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint32_t output_values[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  const uint32_t expected_values[4] = {fill_pattern, 16, 19, 22};
  EXPECT_EQ(0, memcmp(output_values, expected_values, sizeof(expected_values)));

  iree_hal_executable_release(executable);
}

TEST_F(HostQueueCommandBufferTest, Pm4DynamicDispatchUsesDefaultUploadRing) {
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

  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable));

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
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
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
      {0},
      {
          /*buffer=*/input_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
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
  IREE_ASSERT_OK(iree_hal_queue_execute(
      test_device.queue(), iree_hal_semaphore_list_empty(),
      command_buffer_signal_list, command_buffer, binding_table,
      IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint32_t output_values[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  const uint32_t expected_values[4] = {13, 16, 19, 22};
  EXPECT_EQ(0, memcmp(output_values, expected_values, sizeof(expected_values)));

  iree_hal_executable_release(executable);
}

TEST_F(HostQueueCommandBufferTest,
       MixedDynamicDispatchUsesPatchedKernargTemplate) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable));

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
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT |
          IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH,
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
      {0},
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
  IREE_ASSERT_OK(iree_hal_queue_execute(
      test_device.queue(), iree_hal_semaphore_list_empty(),
      command_buffer_signal_list, command_buffer, binding_table,
      IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint32_t output_values[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  const uint32_t expected_values[4] = {13, 16, 19, 22};
  EXPECT_EQ(0, memcmp(output_values, expected_values, sizeof(expected_values)));

  iree_hal_executable_release(executable);
}

TEST_F(HostQueueCommandBufferTest, DynamicDispatchUsesBindingTableSlots) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable));

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
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
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
      {0},
      {
          /*buffer=*/input_buffer.get(),
          /*offset=*/0,
          /*length=*/IREE_HAL_WHOLE_BUFFER,
      },
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
  IREE_ASSERT_OK(iree_hal_queue_execute(
      test_device.queue(), iree_hal_semaphore_list_empty(),
      command_buffer_signal_list, command_buffer, binding_table,
      IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      command_buffer_signal, command_buffer_signal_value,
      iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  uint32_t output_values[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  const uint32_t expected_values[4] = {13, 16, 19, 22};
  EXPECT_EQ(0, memcmp(output_values, expected_values, sizeof(expected_values)));

  iree_hal_executable_release(executable);
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

  iree_hal_queue_t* device0_queue = test_device.queue(/*family_ordinal=*/0);
  ASSERT_NE(device0_queue, nullptr);
  iree_hal_queue_t* device1_queue = test_device.queue(/*family_ordinal=*/1);
  ASSERT_NE(device1_queue, nullptr);

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), iree_hal_queue_family(device0_queue),
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_execute(device1_queue, iree_hal_semaphore_list_empty(),
                             iree_hal_semaphore_list_empty(), command_buffer,
                             iree_hal_buffer_binding_table_empty(),
                             IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
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
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, IREE_HAL_COMMAND_CATEGORY_TRANSFER,
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
  iree_status_t status = iree_hal_queue_fill(
      test_device.queue(), iree_hal_semaphore_list_empty(),
      pressure_signal_list, pressure_buffer,
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
    status =
        iree_hal_queue_execute(&queue->base, iree_hal_semaphore_list_empty(),
                               command_buffer_signal_list, command_buffer,
                               binding_table, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
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
      queue, alloca_signal_list, sizeof(uint32_t), &transient_raw));
  Ref<iree_hal_buffer_t> transient_buffer(transient_raw);
  IREE_ASSERT_OK(iree_hal_semaphore_wait(alloca_signal, alloca_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, IREE_HAL_COMMAND_CATEGORY_TRANSFER,
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
  iree_status_t status = iree_hal_queue_fill(
      test_device.queue(), iree_hal_semaphore_list_empty(),
      pressure_signal_list, pressure_buffer,
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
    status =
        iree_hal_queue_execute(&queue->base, iree_hal_semaphore_list_empty(),
                               command_buffer_signal_list, command_buffer,
                               binding_table, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE);
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
    iree_hal_buffer_t* dealloca_buffer = transient_buffer;
    status = iree_hal_queue_dealloca(&queue->base, dealloca_wait_list,
                                     dealloca_signal_list,
                                     /*buffer_count=*/1, &dealloca_buffer);
  }
  transient_buffer.reset();
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

#if IREE_FILE_IO_ENABLE

TEST_F(HostQueueCommandBufferTest,
       HostVisibleTransientReadDispatchCompletesBeforeQueuedDealloca) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* host_queue = test_device.first_host_queue();
  ASSERT_NE(host_queue, nullptr);
  iree_hal_queue_t* queue = &host_queue->base;
  ASSERT_NE(queue, nullptr);

  const uint32_t input_values[4] = {1, 2, 3, 4};
  const uint8_t* input_bytes = reinterpret_cast<const uint8_t*>(input_values);
  std::vector<uint8_t> file_data(input_bytes,
                                 input_bytes + sizeof(input_values));
  iree::testing::TempFilePath input_file;
  IREE_ASSERT_OK(CreateTempFileWithContents(file_data, &input_file));

  Ref<iree_hal_file_t> source_file;
  IREE_ASSERT_OK(ImportNativeFile(test_device.base_device(), input_file,
                                  IREE_HAL_MEMORY_ACCESS_READ,
                                  source_file.out()));

  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(), iree_hal_queue_family(test_device.queue()),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable));

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), sizeof(input_values), output_buffer.out()));
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
  static constexpr iree_device_size_t kTransientByteLength = 201326592;
  static constexpr iree_device_size_t kDispatchInputOffset = 100663296;
  iree_hal_buffer_t* transient_raw = NULL;
  IREE_ASSERT_OK(QueueHostVisibleDispatchTransientBuffer(
      host_queue, alloca_signal_list, kTransientByteLength, &transient_raw));
  Ref<iree_hal_buffer_t> transient_buffer(transient_raw);

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), iree_hal_queue_family(queue),
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  iree_hal_buffer_ref_t binding_refs[2] = {
      iree_hal_make_buffer_ref(transient_buffer, kDispatchInputOffset,
                               sizeof(input_values)),
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                               sizeof(input_values)),
  };
  const iree_hal_buffer_ref_list_t bindings = {
      /*count=*/IREE_ARRAYSIZE(binding_refs),
      /*values=*/binding_refs,
  };
  IREE_ASSERT_OK(
      AppendConstantsBindingsDispatch(command_buffer, executable, bindings));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  Ref<iree_hal_semaphore_t> gate_signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), gate_signal.out()));
  uint64_t gate_signal_value = 1;
  iree_hal_semaphore_t* read_wait_semaphores[] = {alloca_signal.get(),
                                                  gate_signal.get()};
  uint64_t read_wait_values[] = {alloca_signal_value, gate_signal_value};
  iree_hal_semaphore_list_t read_wait_list = {
      /*count=*/IREE_ARRAYSIZE(read_wait_semaphores),
      /*semaphores=*/read_wait_semaphores,
      /*payload_values=*/read_wait_values,
  };

  Ref<iree_hal_semaphore_t> read_signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), read_signal.out()));
  uint64_t read_signal_value = 1;
  iree_hal_semaphore_t* read_signal_ptr = read_signal.get();
  iree_hal_semaphore_list_t read_signal_list = {
      /*count=*/1,
      /*semaphores=*/&read_signal_ptr,
      /*payload_values=*/&read_signal_value,
  };
  IREE_ASSERT_OK(iree_hal_queue_read(
      queue, read_wait_list, read_signal_list, source_file, /*source_offset=*/0,
      transient_buffer, kDispatchInputOffset, sizeof(input_values),
      IREE_HAL_READ_FLAG_NONE));

  Ref<iree_hal_semaphore_t> dispatch_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), dispatch_signal.out()));
  uint64_t dispatch_signal_value = 1;
  iree_hal_semaphore_t* dispatch_signal_ptr = dispatch_signal.get();
  iree_hal_semaphore_list_t dispatch_wait_list = {
      /*count=*/1,
      /*semaphores=*/&read_signal_ptr,
      /*payload_values=*/&read_signal_value,
  };
  iree_hal_semaphore_list_t dispatch_signal_list = {
      /*count=*/1,
      /*semaphores=*/&dispatch_signal_ptr,
      /*payload_values=*/&dispatch_signal_value,
  };
  IREE_ASSERT_OK(iree_hal_queue_execute(
      queue, dispatch_wait_list, dispatch_signal_list, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));

  Ref<iree_hal_semaphore_t> dealloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), dealloca_signal.out()));
  uint64_t dealloca_signal_value = 1;
  iree_hal_semaphore_t* dealloca_signal_ptr = dealloca_signal.get();
  iree_hal_semaphore_list_t dealloca_wait_list = {
      /*count=*/1,
      /*semaphores=*/&dispatch_signal_ptr,
      /*payload_values=*/&dispatch_signal_value,
  };
  iree_hal_semaphore_list_t dealloca_signal_list = {
      /*count=*/1,
      /*semaphores=*/&dealloca_signal_ptr,
      /*payload_values=*/&dealloca_signal_value,
  };
  iree_hal_buffer_t* dealloca_buffer = transient_buffer;
  IREE_ASSERT_OK(iree_hal_queue_dealloca(queue, dealloca_wait_list,
                                         dealloca_signal_list,
                                         /*buffer_count=*/1, &dealloca_buffer));

  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_signal, gate_signal_value,
                                           /*frontier=*/NULL));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(dealloca_signal, dealloca_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  const uint32_t expected_values[4] = {13, 16, 19, 22};
  uint32_t output_values[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  EXPECT_EQ(0, memcmp(output_values, expected_values, sizeof(expected_values)));

  iree_hal_executable_release(executable);
}

#endif  // IREE_FILE_IO_ENABLE

TEST_F(HostQueueCommandBufferTest,
       OneShotStaticTransientBindingRecordsBeforeAllocaCommit) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

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
      queue, alloca_signal_list, sizeof(uint32_t), &transient_raw));
  Ref<iree_hal_buffer_t> transient_buffer(transient_raw);

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(), iree_hal_queue_family(&queue->base),
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT, IREE_HAL_COMMAND_CATEGORY_TRANSFER,
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
  IREE_ASSERT_OK(iree_hal_queue_execute(
      &queue->base, alloca_signal_list, command_buffer_signal_list,
      command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));

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
  iree_hal_buffer_t* dealloca_buffer = transient_buffer;
  IREE_ASSERT_OK(iree_hal_queue_dealloca(
      &queue->base, command_buffer_signal_list, dealloca_signal_list,
      /*buffer_count=*/1, &dealloca_buffer));
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
