// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>

#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_profiling_test_util.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_test_util.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/pm4_command_buffer.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::hal::cts::Ref;
using namespace test;

class PM4CommandDispatchTest : public ::testing::Test {
 protected:
  static constexpr iree_host_size_t kOutputElementCount = 32;
  static constexpr iree_device_size_t kOutputByteLength =
      kOutputElementCount * sizeof(uint32_t);
  static constexpr iree_device_size_t kParameterOffset = sizeof(uint32_t);
  static constexpr iree_device_size_t kParameterByteLength =
      sizeof(uint32_t[3]);
  static constexpr iree_device_size_t kParameterBufferByteLength =
      kParameterOffset + kParameterByteLength;
  static constexpr uint32_t kSentinelValue = 0xCDCDCDCDu;

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

  void SetUp() override {
    iree_hal_amdgpu_logical_device_options_t options;
    iree_hal_amdgpu_logical_device_options_initialize(&options);
    options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_PM4;
    options.host_queues.upload_capacity = 64 * 1024;
    options.preallocate_pools = 0;
    IREE_ASSERT_OK(test_device_.Initialize(&options, &libhsa_, &topology_,
                                           host_allocator_));

    iree_hal_amdgpu_physical_device_t* physical_device =
        test_device_.logical_device()->physical_devices[0];
    if (!iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_dispatch_command_buffers(
            physical_device->vendor_packet_capabilities)) {
      GTEST_SKIP() << "PM4 dispatch command buffers are not supported on this "
                      "physical device";
    }

    IREE_ASSERT_OK(LoadCtsExecutable(
        test_device_.base_device(),
        IREE_SV("command_buffer_dispatch_multi_workgroup_test.bin"),
        workgroup_id_executable_.out()));
    const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor = NULL;
    IREE_ASSERT_OK(
        iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_device(
            workgroup_id_executable_,
            iree_hal_executable_function_from_index(0),
            /*device_ordinal=*/0, &descriptor));
    ASSERT_FALSE(iree_any_bit_set(
        descriptor->kernarg_layout->flags,
        IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_USES_IMPLICIT_BLOCK_COUNT));
  }

  iree_status_t CreateOutputBuffer(iree_hal_buffer_t** out_buffer) {
    IREE_RETURN_IF_ERROR(CreateHostVisibleDispatchBuffer(
        test_device_.allocator(), kOutputByteLength, out_buffer));
    return ResetOutput(*out_buffer);
  }

  iree_status_t CreateParameterBuffer(uint32_t workgroup_count_x,
                                      iree_hal_buffer_t** out_buffer) {
    IREE_RETURN_IF_ERROR(CreateHostVisibleIndirectParameterBuffer(
        test_device_.allocator(), kParameterBufferByteLength, out_buffer));
    return WriteParameters(*out_buffer, workgroup_count_x);
  }

  static iree_status_t ResetOutput(iree_hal_buffer_t* output_buffer) {
    std::array<uint32_t, kOutputElementCount> values;
    values.fill(kSentinelValue);
    return iree_hal_buffer_map_write(output_buffer, /*target_offset=*/0,
                                     values.data(), sizeof(values));
  }

  static iree_status_t WriteParameters(iree_hal_buffer_t* parameter_buffer,
                                       uint32_t workgroup_count_x) {
    const uint32_t values[3] = {workgroup_count_x, 1, 1};
    return iree_hal_buffer_map_write(parameter_buffer, kParameterOffset, values,
                                     sizeof(values));
  }

  iree_status_t CreateIndirectCommandBuffer(
      iree_hal_buffer_t* output_buffer, iree_hal_buffer_ref_t parameter_ref,
      iree_hal_dispatch_flags_t dispatch_flags, uint32_t binding_capacity,
      iree_hal_command_buffer_t** out_command_buffer,
      iree_hal_command_buffer_mode_t mode =
          IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT) {
    Ref<iree_hal_command_buffer_t> command_buffer;
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
        test_device_.base_device(), mode, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
        IREE_HAL_QUEUE_AFFINITY_ANY, binding_capacity, command_buffer.out()));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_begin(command_buffer));
    iree_hal_buffer_ref_t output_ref = iree_hal_make_buffer_ref(
        output_buffer, /*offset=*/0, kOutputByteLength);
    const iree_hal_buffer_ref_list_t bindings = {
        /*.count=*/1,
        /*.values=*/&output_ref,
    };
    iree_hal_dispatch_config_t config = {};
    config.workgroup_count_ref = parameter_ref;
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_dispatch(
        command_buffer, workgroup_id_executable_,
        iree_hal_executable_function_from_index(0), config,
        iree_const_byte_span_empty(), bindings, dispatch_flags));
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_end(command_buffer));
    *out_command_buffer = command_buffer.release();
    return iree_ok_status();
  }

  iree_status_t Execute(iree_hal_command_buffer_t* command_buffer,
                        iree_hal_buffer_binding_table_t binding_table =
                            iree_hal_buffer_binding_table_empty()) {
    Ref<iree_hal_semaphore_t> signal;
    IREE_RETURN_IF_ERROR(
        CreateSemaphore(test_device_.base_device(), signal.out()));
    iree_hal_semaphore_t* signal_ptr = signal.get();
    uint64_t signal_value = 1;
    const iree_hal_semaphore_list_t signal_list = {
        /*.count=*/1,
        /*.semaphores=*/&signal_ptr,
        /*.payload_values=*/&signal_value,
    };
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_execute(
        test_device_.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_semaphore_list_empty(), signal_list, command_buffer,
        binding_table, IREE_HAL_EXECUTE_FLAG_NONE));
    return iree_hal_semaphore_wait(signal, signal_value,
                                   iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE);
  }

  static void ExpectOutput(iree_hal_buffer_t* output_buffer,
                           uint32_t dispatched_workgroup_count) {
    std::array<uint32_t, kOutputElementCount> values = {};
    IREE_ASSERT_OK(iree_hal_buffer_map_read(output_buffer, /*offset=*/0,
                                            values.data(), sizeof(values)));
    for (iree_host_size_t i = 0; i < values.size(); ++i) {
      SCOPED_TRACE(i);
      const uint32_t expected =
          i < dispatched_workgroup_count ? (uint32_t)i : kSentinelValue;
      EXPECT_EQ(expected, values[i]);
    }
  }

  static const uint32_t* FindIndirectPacket(
      const iree_hal_amdgpu_pm4_program_t* program) {
    const uint32_t header = iree_hal_amdgpu_pm4_make_compute_header(
        IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_DISPATCH_INDIRECT,
        IREE_HAL_AMDGPU_PM4_DISPATCH_INDIRECT_MEC_DWORD_COUNT);
    for (uint32_t i = 0;
         i + IREE_HAL_AMDGPU_PM4_DISPATCH_INDIRECT_MEC_DWORD_COUNT <=
         program->dword_count;
         ++i) {
      if (program->dwords[i] == header) return &program->dwords[i];
    }
    return NULL;
  }

  static const uint32_t* FindAcquireMemPacket(
      const iree_hal_amdgpu_pm4_program_t* program) {
    const uint32_t header = iree_hal_amdgpu_pm4_make_header(
        IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_ACQUIRE_MEM,
        IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GFX10_DWORD_COUNT);
    for (uint32_t i = 0;
         i + IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GFX10_DWORD_COUNT <=
         program->dword_count;
         ++i) {
      if (program->dwords[i] == header) return &program->dwords[i];
    }
    return NULL;
  }

  static uint64_t PacketAddress(const uint32_t* packet) {
    return (uint64_t)packet[1] | (uint64_t)packet[2] << 32;
  }

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;

  TestLogicalDevice test_device_;
  Ref<iree_hal_executable_t> workgroup_id_executable_;
};

iree_allocator_t PM4CommandDispatchTest::host_allocator_;
iree_hal_amdgpu_libhsa_t PM4CommandDispatchTest::libhsa_;
iree_hal_amdgpu_topology_t PM4CommandDispatchTest::topology_;

TEST_F(PM4CommandDispatchTest, StaticParametersUseNativePacketAndRetainBuffer) {
  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateOutputBuffer(output_buffer.out()));
  Ref<iree_hal_buffer_t> parameter_buffer;
  IREE_ASSERT_OK(
      CreateParameterBuffer(/*workgroup_count_x=*/4, parameter_buffer.out()));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(CreateIndirectCommandBuffer(
      output_buffer,
      iree_hal_make_buffer_ref(parameter_buffer, kParameterOffset,
                               kParameterByteLength),
      IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS,
      /*binding_capacity=*/0, command_buffer.out()));
  ASSERT_TRUE(iree_hal_amdgpu_pm4_command_buffer_isa(command_buffer));
  const iree_hal_amdgpu_pm4_program_t* program =
      iree_hal_amdgpu_pm4_command_buffer_program(command_buffer);
  const uint32_t* packet = FindIndirectPacket(program);
  ASSERT_NE(nullptr, packet);
  EXPECT_FALSE(iree_any_bit_set(
      packet[3], IREE_HAL_AMDGPU_PM4_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS));
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(parameter_buffer);
  const uintptr_t allocation_address =
      (uintptr_t)iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
  ASSERT_NE(0u, allocation_address);
  EXPECT_EQ(allocation_address + iree_hal_buffer_byte_offset(parameter_buffer) +
                kParameterOffset,
            PacketAddress(packet));
  const iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t* fixup_plan =
      iree_hal_amdgpu_pm4_command_buffer_fixup_plan(command_buffer);
  EXPECT_EQ(0u, fixup_plan->entry_count);

  iree_hal_buffer_release(parameter_buffer.release());
  IREE_ASSERT_OK(Execute(command_buffer));
  ExpectOutput(output_buffer, /*dispatched_workgroup_count=*/4);
}

TEST_F(PM4CommandDispatchTest, DynamicBindingFixupSupportsReusableExecution) {
  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateOutputBuffer(output_buffer.out()));
  Ref<iree_hal_buffer_t> parameter_buffer;
  IREE_ASSERT_OK(
      CreateParameterBuffer(/*workgroup_count_x=*/4, parameter_buffer.out()));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(CreateIndirectCommandBuffer(
      output_buffer,
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, kParameterOffset,
                                        kParameterByteLength),
      IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS,
      /*binding_capacity=*/1, command_buffer.out()));
  ASSERT_TRUE(iree_hal_amdgpu_pm4_command_buffer_isa(command_buffer));
  const iree_hal_amdgpu_pm4_program_t* program =
      iree_hal_amdgpu_pm4_command_buffer_program(command_buffer);
  const uint32_t* packet = FindIndirectPacket(program);
  ASSERT_NE(nullptr, packet);
  EXPECT_EQ(kParameterOffset, PacketAddress(packet));
  EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(&packet[1]) % alignof(uint64_t));
  const iree_hal_amdgpu_pm4_command_buffer_fixup_plan_t* fixup_plan =
      iree_hal_amdgpu_pm4_command_buffer_fixup_plan(command_buffer);
  ASSERT_EQ(1u, fixup_plan->entry_count);
  EXPECT_EQ(0u, fixup_plan->entries[0].binding_slot);
  EXPECT_EQ(kParameterOffset, fixup_plan->entries[0].binding_offset);
  EXPECT_EQ(reinterpret_cast<const uint8_t*>(&packet[1]),
            reinterpret_cast<const uint8_t*>(fixup_plan->target_base) +
                fixup_plan->entries[0].target_offset);

  iree_hal_buffer_binding_t parameter_binding = {
      /*.buffer=*/parameter_buffer,
      /*.offset=*/0,
      /*.length=*/kParameterBufferByteLength,
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      /*.count=*/1,
      /*.bindings=*/&parameter_binding,
  };
  IREE_ASSERT_OK(Execute(command_buffer, binding_table));
  ExpectOutput(output_buffer, /*dispatched_workgroup_count=*/4);

  IREE_ASSERT_OK(ResetOutput(output_buffer));
  IREE_ASSERT_OK(WriteParameters(parameter_buffer, /*workgroup_count_x=*/2));
  IREE_ASSERT_OK(Execute(command_buffer, binding_table));
  ExpectOutput(output_buffer, /*dispatched_workgroup_count=*/2);
}

TEST_F(PM4CommandDispatchTest, DynamicParametersObservePriorDispatch) {
  Ref<iree_hal_executable_t> parameter_producer_executable;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device_.base_device(),
      IREE_SV("command_buffer_dispatch_indirect_parameters_test.bin"),
      parameter_producer_executable.out()));
  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateOutputBuffer(output_buffer.out()));
  Ref<iree_hal_buffer_t> parameter_buffer;
  IREE_ASSERT_OK(
      CreateParameterBuffer(/*workgroup_count_x=*/0, parameter_buffer.out()));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device_.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  iree_hal_buffer_ref_t parameter_ref = iree_hal_make_buffer_ref(
      parameter_buffer, kParameterOffset, kParameterByteLength);
  const iree_hal_buffer_ref_list_t producer_bindings = {
      /*.count=*/1,
      /*.values=*/&parameter_ref,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, parameter_producer_executable,
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), producer_bindings,
      IREE_HAL_DISPATCH_FLAG_NONE));
  const iree_hal_memory_barrier_t memory_barrier = {
      /*.source_scope=*/IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
          IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE,
      /*.target_scope=*/IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ |
          IREE_HAL_ACCESS_SCOPE_MEMORY_READ,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer,
      /*source_stage_mask=*/IREE_HAL_EXECUTION_STAGE_DISPATCH |
          IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      /*target_stage_mask=*/IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS |
          IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, /*memory_barrier_count=*/1,
      &memory_barrier, /*buffer_barrier_count=*/0,
      /*buffer_barriers=*/NULL));
  iree_hal_buffer_ref_t output_ref =
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0, kOutputByteLength);
  const iree_hal_buffer_ref_list_t consumer_bindings = {
      /*.count=*/1,
      /*.values=*/&output_ref,
  };
  iree_hal_dispatch_config_t indirect_config = {};
  indirect_config.workgroup_count_ref = parameter_ref;
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, workgroup_id_executable_,
      iree_hal_executable_function_from_index(0), indirect_config,
      iree_const_byte_span_empty(), consumer_bindings,
      IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
  ASSERT_TRUE(iree_hal_amdgpu_pm4_command_buffer_isa(command_buffer));

  const iree_hal_amdgpu_pm4_program_t* program =
      iree_hal_amdgpu_pm4_command_buffer_program(command_buffer);
  const uint32_t* acquire_mem_packet = FindAcquireMemPacket(program);
  ASSERT_NE(nullptr, acquire_mem_packet);
  iree_hal_amdgpu_physical_device_t* physical_device =
      test_device_.logical_device()->physical_devices[0];
  const bool cp_memory_bypasses_gl2 = iree_any_bit_set(
      physical_device->vendor_packet_capabilities,
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_CP_MEMORY_BYPASSES_GL2);
  EXPECT_EQ(cp_memory_bypasses_gl2,
            iree_any_bit_set(acquire_mem_packet[7],
                             IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GCR_GL2_WB));
  EXPECT_FALSE(iree_any_bit_set(acquire_mem_packet[7],
                                IREE_HAL_AMDGPU_PM4_ACQUIRE_MEM_GCR_GL2_INV));

  IREE_ASSERT_OK(Execute(command_buffer));
  ExpectOutput(output_buffer, /*dispatched_workgroup_count=*/4);
}

TEST_F(PM4CommandDispatchTest, ZeroParametersRemainAnExecutionTimeNoOp) {
  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateOutputBuffer(output_buffer.out()));
  Ref<iree_hal_buffer_t> parameter_buffer;
  IREE_ASSERT_OK(
      CreateParameterBuffer(/*workgroup_count_x=*/0, parameter_buffer.out()));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(CreateIndirectCommandBuffer(
      output_buffer,
      iree_hal_make_buffer_ref(parameter_buffer, kParameterOffset,
                               kParameterByteLength),
      IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS,
      /*binding_capacity=*/0, command_buffer.out()));
  const iree_hal_amdgpu_pm4_program_t* program =
      iree_hal_amdgpu_pm4_command_buffer_program(command_buffer);
  ASSERT_NE(nullptr, FindIndirectPacket(program));

  IREE_ASSERT_OK(Execute(command_buffer));
  ExpectOutput(output_buffer, /*dispatched_workgroup_count=*/0);
}

TEST_F(PM4CommandDispatchTest,
       ProfileProgramExecutesIndirectPacketAndMarksUnknownCounts) {
  iree_hal_amdgpu_physical_device_t* physical_device =
      test_device_.logical_device()->physical_devices[0];
  if (!iree_hal_amdgpu_pm4_timestamp_strategy_supports_ranges(
          physical_device->pm4_timestamp_strategy)) {
    GTEST_SKIP() << "PM4 dispatch timestamp packets are not supported on this "
                    "physical device";
  }

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device_.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateOutputBuffer(output_buffer.out()));
  Ref<iree_hal_buffer_t> parameter_buffer;
  IREE_ASSERT_OK(
      CreateParameterBuffer(/*workgroup_count_x=*/4, parameter_buffer.out()));
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(CreateIndirectCommandBuffer(
      output_buffer,
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, kParameterOffset,
                                        kParameterByteLength),
      IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS,
      /*binding_capacity=*/1, command_buffer.out(),
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA));
  const iree_hal_amdgpu_pm4_command_buffer_profile_plan_t* profile_plan =
      iree_hal_amdgpu_pm4_command_buffer_profile_plan(
          command_buffer, /*physical_queue_ordinal=*/0);
  ASSERT_NE(nullptr, profile_plan);
  const uint32_t* packet = FindIndirectPacket(&profile_plan->program);
  ASSERT_NE(nullptr, packet);
  EXPECT_EQ(kParameterOffset, PacketAddress(packet));
  EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(&packet[1]) % alignof(uint64_t));

  iree_hal_buffer_binding_t parameter_binding = {
      /*.buffer=*/parameter_buffer,
      /*.offset=*/0,
      /*.length=*/kParameterBufferByteLength,
  };
  const iree_hal_buffer_binding_table_t binding_table = {
      /*.count=*/1,
      /*.bindings=*/&parameter_binding,
  };
  IREE_ASSERT_OK(Execute(command_buffer, binding_table));
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device_.base_device()));
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device_.base_device()));
  IREE_ASSERT_OK(profiling.End());
  ExpectOutput(output_buffer, /*dispatched_workgroup_count=*/4);

  EXPECT_EQ(1, sink.command_buffer_metadata_count);
  EXPECT_EQ(1, sink.command_operation_metadata_count);
  ASSERT_EQ(1u, sink.command_operations.size());
  const iree_hal_profile_command_operation_record_t& operation =
      sink.command_operations[0];
  EXPECT_EQ(IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH, operation.type);
  EXPECT_TRUE(iree_any_bit_set(
      operation.flags,
      IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_INDIRECT_PARAMETERS));
  EXPECT_EQ(0u, operation.workgroup_count[0]);
  EXPECT_EQ(0u, operation.workgroup_count[1]);
  EXPECT_EQ(0u, operation.workgroup_count[2]);

  ASSERT_EQ(1u, sink.dispatch_events.size());
  const iree_hal_profile_dispatch_event_t& event = sink.dispatch_events[0];
  EXPECT_TRUE(iree_all_bits_set(
      event.flags,
      IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER |
          IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_INDIRECT_PARAMETERS));
  EXPECT_EQ(operation.command_buffer_id, event.command_buffer_id);
  EXPECT_EQ(operation.command_index, event.command_index);
  EXPECT_EQ(0u, event.workgroup_count[0]);
  EXPECT_EQ(0u, event.workgroup_count[1]);
  EXPECT_EQ(0u, event.workgroup_count[2]);
  ExpectDispatchEventsWithinClockCorrelationRange(sink);
}

TEST_F(PM4CommandDispatchTest, RejectsImplicitBlockCountKernargs) {
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(LoadCtsExecutable(test_device_.base_device(),
                                   IREE_SV("hostcall_buffer_test.bin"),
                                   executable.out()));
  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor = NULL;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_device(
          executable, iree_hal_executable_function_from_index(0),
          /*device_ordinal=*/0, &descriptor));
  ASSERT_TRUE(iree_any_bit_set(
      descriptor->kernarg_layout->flags,
      IREE_HAL_AMDGPU_KERNARG_LAYOUT_FLAG_USES_IMPLICIT_BLOCK_COUNT));

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device_.allocator(), sizeof(uint64_t), output_buffer.out()));
  Ref<iree_hal_buffer_t> parameter_buffer;
  IREE_ASSERT_OK(
      CreateParameterBuffer(/*workgroup_count_x=*/1, parameter_buffer.out()));
  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device_.base_device(), IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  iree_hal_dispatch_config_t config = {};
  config.workgroup_count_ref = iree_hal_make_buffer_ref(
      parameter_buffer, kParameterOffset, kParameterByteLength);
  iree_hal_buffer_ref_t output_ref =
      iree_hal_make_buffer_ref(output_buffer, /*offset=*/0, sizeof(uint64_t));
  const iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/1,
      /*.values=*/&output_ref,
  };
  iree_status_t status = iree_hal_command_buffer_dispatch(
      command_buffer, executable, iree_hal_executable_function_from_index(0),
      config, iree_const_byte_span_empty(), bindings,
      IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS);
  EXPECT_EQ(IREE_STATUS_UNIMPLEMENTED, iree_status_code(status));
  iree_status_free(status);
}

}  // namespace
}  // namespace iree::hal::amdgpu
