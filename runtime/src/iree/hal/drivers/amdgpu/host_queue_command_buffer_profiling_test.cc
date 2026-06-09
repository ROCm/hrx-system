// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_packet.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_profiling_test_util.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile_events.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/pm4_command_buffer.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::hal::cts::Ref;
using namespace test;

class HostQueueCommandBufferProfilingTest : public ::testing::Test {
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

iree_allocator_t HostQueueCommandBufferProfilingTest::host_allocator_;
iree_hal_amdgpu_libhsa_t HostQueueCommandBufferProfilingTest::libhsa_;
iree_hal_amdgpu_topology_t HostQueueCommandBufferProfilingTest::topology_;

TEST_F(HostQueueCommandBufferProfilingTest,
       ExplicitHardwareCounterSelectionEmitsMetadataWhenAvailable) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status = BeginSqWavesProfiling(&profiling, &sink);
  if (IsHardwareCounterProfilingUnavailable(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "AMDGPU hardware counter profiling unavailable";
  }
  IREE_ASSERT_OK(profiling_status);
  IREE_ASSERT_OK(profiling.End());

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.counter_set_metadata_count);
  EXPECT_EQ(1, sink.counter_metadata_count);
}

TEST_F(HostQueueCommandBufferProfilingTest,
       MultipleHardwareCounterSelectionEmitsLayoutWhenAvailable) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status = BeginSqWaveWidthProfiling(&profiling, &sink);
  if (IsHardwareCounterProfilingUnavailable(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "AMDGPU hardware counter profiling unavailable";
  }
  IREE_ASSERT_OK(profiling_status);
  IREE_ASSERT_OK(profiling.End());

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.counter_set_metadata_count);
  EXPECT_EQ(1, sink.counter_metadata_count);
  ASSERT_FALSE(sink.counter_set_records.empty());
  ASSERT_EQ(sink.counter_set_records.size() * 4u, sink.counter_records.size());
  const iree_hal_profile_counter_unit_t expected_units[] = {
      IREE_HAL_PROFILE_COUNTER_UNIT_COUNT,
      IREE_HAL_PROFILE_COUNTER_UNIT_COUNT,
      IREE_HAL_PROFILE_COUNTER_UNIT_COUNT,
      IREE_HAL_PROFILE_COUNTER_UNIT_CYCLES,
  };
  iree_host_size_t counter_record_index = 0;
  for (const auto& counter_set_record : sink.counter_set_records) {
    ASSERT_EQ(4u, counter_set_record.counter_count);
    uint32_t sample_value_count = 0;
    for (uint32_t i = 0; i < counter_set_record.counter_count; ++i) {
      const auto& counter_record = sink.counter_records[counter_record_index++];
      EXPECT_EQ(counter_set_record.counter_set_id,
                counter_record.counter_set_id);
      EXPECT_EQ(sample_value_count, counter_record.sample_value_offset);
      EXPECT_GT(counter_record.sample_value_count, 0u);
      EXPECT_EQ(expected_units[i], counter_record.unit);
      sample_value_count += counter_record.sample_value_count;
    }
    EXPECT_EQ(sample_value_count, counter_set_record.sample_value_count);
  }
}

TEST_F(HostQueueCommandBufferProfilingTest,
       Pm4CommandBufferExecuteEmitsDispatchProfileEvents) {
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
  if (!iree_hal_amdgpu_pm4_timestamp_strategy_supports_ranges(
          physical_device->pm4_timestamp_strategy)) {
    GTEST_SKIP() << "PM4 dispatch timestamp packets are not supported on this "
                    "physical device";
  }

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  TwoDispatchCommandBuffer fixture;
  IREE_ASSERT_OK(CreateTwoDispatchCommandBuffer(
      &test_device, &fixture,
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA));
  ASSERT_TRUE(iree_hal_amdgpu_pm4_command_buffer_isa(fixture.command_buffer));

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

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());

  ExpectTwoDispatchOutputs(fixture);

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.command_buffer_metadata_count);
  EXPECT_EQ(1, sink.command_operation_metadata_count);
  EXPECT_FALSE(sink.write_after_end);
  ASSERT_EQ(2u, sink.command_operations.size());
  for (iree_host_size_t i = 0; i < sink.command_operations.size(); ++i) {
    const iree_hal_profile_command_operation_record_t& operation =
        sink.command_operations[i];
    EXPECT_EQ(IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH, operation.type);
    EXPECT_EQ(sink.command_buffer_ids[0], operation.command_buffer_id);
    EXPECT_EQ((uint32_t)i, operation.command_index);
    EXPECT_NE(0u, operation.executable_id);
    EXPECT_EQ(0u, operation.function_ordinal);
    EXPECT_EQ(2u, operation.binding_count);
    EXPECT_EQ(1u, operation.workgroup_count[0]);
    EXPECT_EQ(1u, operation.workgroup_count[1]);
    EXPECT_EQ(1u, operation.workgroup_count[2]);
    EXPECT_NE(0u, operation.workgroup_size[0]);
  }
  ASSERT_EQ(2u, sink.dispatch_events.size());
  for (iree_host_size_t i = 0; i < sink.dispatch_events.size(); ++i) {
    const iree_hal_profile_dispatch_event_t& event = sink.dispatch_events[i];
    EXPECT_EQ(sizeof(iree_hal_profile_dispatch_event_t), event.record_length);
    EXPECT_EQ(IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER, event.flags);
    EXPECT_NE(0u, event.event_id);
    EXPECT_NE(0u, event.submission_id);
    EXPECT_EQ(sink.command_buffer_ids[0], event.command_buffer_id);
    EXPECT_NE(0u, event.executable_id);
    EXPECT_EQ((uint32_t)i, event.command_index);
    EXPECT_EQ(0u, event.function_ordinal);
    EXPECT_EQ(1u, event.workgroup_count[0]);
    EXPECT_EQ(1u, event.workgroup_count[1]);
    EXPECT_EQ(1u, event.workgroup_count[2]);
    EXPECT_NE(0u, event.workgroup_size[0]);
  }
  ExpectDispatchEventsHaveClockCorrelations(sink);
}

TEST_F(HostQueueCommandBufferProfilingTest,
       Pm4CommandBufferExecuteEmitsQueueDeviceProfileEnvelope) {
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

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status =
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
                          IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS,
                      CommandBufferProfileSinkAsBase(&sink));
  if (IsQueueDeviceProfilingUnavailable(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "queue-device profiling data family unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

  TwoDispatchCommandBuffer fixture;
  IREE_ASSERT_OK(CreateTwoDispatchCommandBuffer(
      &test_device, &fixture,
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA));
  ASSERT_TRUE(iree_hal_amdgpu_pm4_command_buffer_isa(fixture.command_buffer));

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

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());

  ExpectTwoDispatchOutputs(fixture);

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.command_buffer_metadata_count);
  EXPECT_EQ(1, sink.command_operation_metadata_count);
  EXPECT_EQ(1, sink.queue_event_count);
  EXPECT_EQ(1, sink.queue_device_event_count);
  EXPECT_FALSE(sink.write_after_end);
  ASSERT_EQ(1u, sink.command_buffer_ids.size());
  ASSERT_EQ(2u, sink.command_operations.size());
  for (uint32_t i = 0; i < sink.command_operations.size(); ++i) {
    const iree_hal_profile_command_operation_record_t& operation =
        sink.command_operations[i];
    EXPECT_EQ(IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH, operation.type);
    EXPECT_FALSE(
        iree_hal_profile_command_operation_has_block_structure(&operation));
    EXPECT_EQ(i, operation.command_index);
    EXPECT_EQ(sink.command_buffer_ids[0], operation.command_buffer_id);
    EXPECT_NE(0u, operation.executable_id);
    EXPECT_EQ(0u, operation.function_ordinal);
    EXPECT_EQ(2u, operation.binding_count);
    EXPECT_EQ(1u, operation.workgroup_count[0]);
    EXPECT_EQ(1u, operation.workgroup_count[1]);
    EXPECT_EQ(1u, operation.workgroup_count[2]);
    EXPECT_NE(0u, operation.workgroup_size[0]);
    EXPECT_NE(0u, operation.flags &
                      IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_STATIC_BINDINGS);
    EXPECT_NE(
        0u, operation.flags &
                IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_PREPUBLISHED_ARGUMENTS);
  }

  ASSERT_EQ(1u, sink.queue_events.size());
  ASSERT_EQ(1u, sink.queue_device_events.size());
  const iree_hal_profile_queue_event_t& queue_event = sink.queue_events[0];
  EXPECT_EQ(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE, queue_event.type);
  EXPECT_NE(0u, queue_event.submission_id);
  EXPECT_EQ(sink.command_buffer_ids[0], queue_event.command_buffer_id);
  EXPECT_EQ(0u, queue_event.wait_count);
  EXPECT_EQ(1u, queue_event.signal_count);
  EXPECT_EQ(2u, queue_event.operation_count);
  EXPECT_NE(0u, queue_event.payload_length);

  const iree_hal_profile_queue_device_event_t& device_event =
      sink.queue_device_events[0];
  EXPECT_EQ(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE, device_event.type);
  EXPECT_EQ(queue_event.submission_id, device_event.submission_id);
  EXPECT_EQ(queue_event.command_buffer_id, device_event.command_buffer_id);
  EXPECT_EQ(queue_event.stream_id, device_event.stream_id);
  EXPECT_EQ(queue_event.physical_device_ordinal,
            device_event.physical_device_ordinal);
  EXPECT_EQ(queue_event.queue_ordinal, device_event.queue_ordinal);
  EXPECT_EQ(queue_event.operation_count, device_event.operation_count);
  EXPECT_EQ(queue_event.payload_length, device_event.payload_length);
  EXPECT_NE(0u, device_event.start_tick);
  EXPECT_NE(0u, device_event.end_tick);
  EXPECT_GE(device_event.end_tick, device_event.start_tick);
}

TEST_F(HostQueueCommandBufferProfilingTest,
       AutoCommandBufferModeUsesPm4WhileQueueProfiling) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AUTO;
  options.host_queues.upload_capacity = 64 * 1024;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status =
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                      CommandBufferProfileSinkAsBase(&sink));
  if (IsProfilingUnsupported(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "device profiling data family unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

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

  IREE_ASSERT_OK(profiling.End());
}

TEST_F(HostQueueCommandBufferProfilingTest,
       AutoCommandBufferModeUsesPm4WhileDispatchProfiling) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.command_buffer_mode = IREE_HAL_AMDGPU_COMMAND_BUFFER_MODE_AUTO;
  options.host_queues.upload_capacity = 64 * 1024;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  TwoDispatchCommandBuffer fixture;
  IREE_ASSERT_OK(CreateTwoDispatchCommandBuffer(
      &test_device, &fixture,
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA));

  iree_hal_amdgpu_physical_device_t* physical_device =
      test_device.logical_device()->physical_devices[0];
  const bool pm4_supported =
      iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_dispatch_command_buffers(
          physical_device->vendor_packet_capabilities) &&
      iree_hal_amdgpu_pm4_timestamp_strategy_supports_ranges(
          physical_device->pm4_timestamp_strategy);
  EXPECT_EQ(pm4_supported,
            iree_hal_amdgpu_pm4_command_buffer_isa(fixture.command_buffer));
  EXPECT_NE(pm4_supported,
            iree_hal_amdgpu_aql_command_buffer_isa(fixture.command_buffer));

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

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());

  ExpectTwoDispatchOutputs(fixture);
  ASSERT_EQ(2u, sink.dispatch_events.size());
  for (iree_host_size_t i = 0; i < sink.dispatch_events.size(); ++i) {
    const iree_hal_profile_dispatch_event_t& event = sink.dispatch_events[i];
    EXPECT_EQ(IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER, event.flags);
    EXPECT_EQ((uint32_t)i, event.command_index);
  }
}

TEST_F(HostQueueCommandBufferProfilingTest,
       ProfilingMetadataCoversMultiplePhysicalDevices) {
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
  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  ASSERT_GE(logical_device->physical_device_count, 2u);
  ASSERT_GT(logical_device->system->topology.gpu_agent_queue_count, 0u);

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));
  IREE_ASSERT_OK(profiling.End());

  const iree_host_size_t physical_device_count =
      logical_device->physical_device_count;
  const iree_host_size_t queue_count_per_physical_device =
      logical_device->system->topology.gpu_agent_queue_count;
  EXPECT_EQ(1, sink.device_metadata_count);
  EXPECT_EQ(1, sink.queue_metadata_count);
  ASSERT_EQ(physical_device_count, sink.device_records.size());
  ASSERT_EQ(physical_device_count * queue_count_per_physical_device,
            sink.queue_records.size());
  EXPECT_TRUE(sink.clock_correlations.empty());

  for (const auto& device_record : sink.device_records) {
    EXPECT_LT(device_record.physical_device_ordinal, physical_device_count);
    EXPECT_EQ(queue_count_per_physical_device, device_record.queue_count);
  }
  for (const auto& queue_record : sink.queue_records) {
    EXPECT_LT(queue_record.physical_device_ordinal, physical_device_count);
    EXPECT_LT(queue_record.queue_ordinal, queue_count_per_physical_device);
  }
}

TEST_F(HostQueueCommandBufferProfilingTest, SinklessProfilingBeginFails) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  iree_hal_device_profiling_options_t profiling_options = {0};
  profiling_options.data_families =
      IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_device_profiling_begin(
                            test_device.base_device(), &profiling_options));
  IREE_EXPECT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_EXPECT_OK(iree_hal_device_profiling_end(test_device.base_device()));
}

TEST_F(HostQueueCommandBufferProfilingTest,
       SuppressedDeviceFineMemoryAllowsDispatchProfiling) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;
  options.suppress_device_fine_memory = 1;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  ExpectQueueEventProfilingCanBeginAndEnd(&test_device);

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));
  IREE_ASSERT_OK(profiling.End());
  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
}

TEST_F(HostQueueCommandBufferProfilingTest,
       SuppressedDeviceFineMemoryAllowsQueueDeviceProfiling) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;
  options.suppress_device_fine_memory = 1;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t status =
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS,
                      CommandBufferProfileSinkAsBase(&sink));
  if (IsQueueDeviceProfilingUnavailable(status)) {
    iree_status_free(status);
    GTEST_SKIP() << "queue-device profiling data family unsupported by backend";
  }
  IREE_ASSERT_OK(status);
  IREE_ASSERT_OK(profiling.End());
  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
}

TEST_F(HostQueueCommandBufferProfilingTest, NestedProfilingBeginFails) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink first_sink = {};
  CommandBufferProfileSinkInitialize(&first_sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 CommandBufferProfileSinkAsBase(&first_sink)));

  CommandBufferProfileSink nested_sink = {};
  CommandBufferProfileSinkInitialize(&nested_sink);
  iree_hal_device_profiling_options_t nested_options = {0};
  nested_options.data_families = IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS;
  nested_options.sink = CommandBufferProfileSinkAsBase(&nested_sink);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_device_profiling_begin(
                            test_device.base_device(), &nested_options));
  EXPECT_EQ(0, nested_sink.begin_count);
  EXPECT_EQ(0, nested_sink.end_count);

  IREE_ASSERT_OK(profiling.End());
  EXPECT_EQ(1, first_sink.begin_count);
  EXPECT_EQ(1, first_sink.end_count);
  ExpectQueueEventProfilingCanBeginAndEnd(&test_device);
}

TEST_F(HostQueueCommandBufferProfilingTest,
       ProfilingBeginSinkBeginFailureAllowsRetry) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  sink.fail_begin_session_status_code = IREE_STATUS_RESOURCE_EXHAUSTED;
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                      CommandBufferProfileSinkAsBase(&sink)));
  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(0, sink.end_count);

  ExpectQueueEventProfilingCanBeginAndEnd(&test_device);
}

TEST_F(HostQueueCommandBufferProfilingTest,
       ProfilingBeginMetadataWriteFailureEndsSessionAndAllowsRetry) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  sink.fail_write_content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES;
  sink.fail_write_remaining = 1;
  sink.fail_write_status_code = IREE_STATUS_DATA_LOSS;
  sink.expected_end_session_status_code = IREE_STATUS_DATA_LOSS;
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                      CommandBufferProfileSinkAsBase(&sink)));
  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(IREE_STATUS_DATA_LOSS, sink.observed_end_session_status_code);
  EXPECT_EQ(0, sink.device_metadata_count);

  ExpectQueueEventProfilingCanBeginAndEnd(&test_device);
}

TEST_F(HostQueueCommandBufferProfilingTest,
       ProfilingFlushWriteFailurePreservesQueueEventsForRetry) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  IREE_ASSERT_OK(SubmitProfiledQueueFill(&test_device));
  sink.fail_write_content_type = IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS;
  sink.fail_write_remaining = 1;
  sink.fail_write_status_code = IREE_STATUS_UNAVAILABLE;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_device_profiling_flush(test_device.base_device()));
  EXPECT_EQ(0, sink.queue_event_count);
  EXPECT_TRUE(sink.queue_events.empty());

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());
  EXPECT_EQ(1, sink.queue_event_count);
  ASSERT_EQ(1u, sink.queue_events.size());
  EXPECT_EQ(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL, sink.queue_events[0].type);
}

TEST_F(HostQueueCommandBufferProfilingTest,
       ProfiledQueueEventsReportDroppedRecordsWhenRingFull) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  const iree_host_size_t event_capacity =
      test_device.logical_device()
          ->profiling.event_streams.queue.stream.ring.capacity;
  ASSERT_GT(event_capacity, 0u);
  for (iree_host_size_t i = 0; i <= event_capacity; ++i) {
    iree_hal_profile_queue_event_t event =
        iree_hal_profile_queue_event_default();
    event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL;
    event.physical_device_ordinal = 0;
    event.queue_ordinal = 0;
    event.operation_count = 1;
    iree_hal_amdgpu_logical_device_record_profile_queue_event(
        test_device.base_device(), &event);
  }

  IREE_ASSERT_OK(profiling.End());
  EXPECT_EQ(1, sink.queue_event_count);
  EXPECT_EQ(event_capacity, sink.queue_events.size());
  EXPECT_EQ(1u, sink.queue_event_dropped_record_count);
  EXPECT_EQ(1u, sink.dropped_record_count);
  EXPECT_EQ(1, sink.truncated_chunk_count);
}

TEST_F(HostQueueCommandBufferProfilingTest,
       ProfiledMemoryEventsReportDroppedRecordsWhenRingFull) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  const iree_host_size_t event_capacity =
      test_device.logical_device()
          ->profiling.event_streams.memory.stream.ring.capacity;
  ASSERT_GT(event_capacity, 0u);
  iree_host_size_t recorded_count = 0;
  for (iree_host_size_t i = 0; i <= event_capacity; ++i) {
    iree_hal_profile_memory_event_t event =
        iree_hal_profile_memory_event_default();
    event.type = IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_ALLOCATE;
    event.allocation_id = i + 1;
    event.physical_device_ordinal = 0;
    event.queue_ordinal = 0;
    event.length = sizeof(uint32_t);
    if (iree_hal_amdgpu_logical_device_record_profile_memory_event(
            test_device.base_device(), &event)) {
      ++recorded_count;
    }
  }

  IREE_ASSERT_OK(profiling.End());
  EXPECT_EQ(event_capacity, recorded_count);
  EXPECT_EQ(1, sink.memory_event_count);
  EXPECT_EQ(event_capacity, sink.memory_events.size());
  EXPECT_EQ(1u, sink.memory_event_dropped_record_count);
  EXPECT_EQ(1u, sink.dropped_record_count);
  EXPECT_EQ(1, sink.truncated_chunk_count);
}

TEST_F(HostQueueCommandBufferProfilingTest,
       ProfilingFlushMetadataWriteFailurePreservesCursorForRetry) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(),
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  Ref<iree_hal_semaphore_t> signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), signal.out()));
  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_ptr = signal.get();
  const iree_hal_semaphore_list_t signal_list = {
      /*count=*/1,
      /*semaphores=*/&signal_ptr,
      /*payload_values=*/&signal_value,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, command_buffer,
      iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(signal, signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  sink.fail_write_content_type = IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS;
  sink.fail_write_remaining = 1;
  sink.fail_write_status_code = IREE_STATUS_UNAVAILABLE;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_device_profiling_flush(test_device.base_device()));
  EXPECT_EQ(0, sink.command_buffer_metadata_count);
  EXPECT_TRUE(sink.command_buffer_ids.empty());

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());
  EXPECT_EQ(1, sink.command_buffer_metadata_count);
  ASSERT_EQ(1u, sink.command_buffer_ids.size());
}

TEST_F(HostQueueCommandBufferProfilingTest,
       ProfilingEndWriteFailureReportsSessionStatusAndAllowsRetry) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  IREE_ASSERT_OK(SubmitProfiledQueueFill(&test_device));
  sink.fail_write_content_type = IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS;
  sink.fail_write_remaining = 1;
  sink.fail_write_status_code = IREE_STATUS_DATA_LOSS;
  sink.expected_end_session_status_code = IREE_STATUS_DATA_LOSS;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, profiling.End());
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(IREE_STATUS_DATA_LOSS, sink.observed_end_session_status_code);
  EXPECT_EQ(0, sink.queue_event_count);

  ExpectQueueEventProfilingCanBeginAndEnd(&test_device);
}

TEST_F(HostQueueCommandBufferProfilingTest,
       ProfilingEndSessionFailureClearsStateAndAllowsRetry) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  sink.fail_end_session_status_code = IREE_STATUS_ABORTED;
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, profiling.End());
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(IREE_STATUS_OK, sink.observed_end_session_status_code);

  ExpectQueueEventProfilingCanBeginAndEnd(&test_device);
}

TEST_F(HostQueueCommandBufferProfilingTest,
       CommandBufferDispatchesEmitProfileEvents) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  TwoDispatchCommandBuffer fixture;
  IREE_ASSERT_OK(CreateTwoDispatchCommandBuffer(&test_device, &fixture));

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

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());

  ExpectTwoDispatchOutputs(fixture);

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.device_metadata_count);
  EXPECT_EQ(1, sink.queue_metadata_count);
  EXPECT_EQ(1, sink.executable_metadata_count);
  EXPECT_EQ(1, sink.executable_function_metadata_count);
  EXPECT_EQ(1, sink.command_buffer_metadata_count);
  EXPECT_EQ(1, sink.command_operation_metadata_count);
  EXPECT_GE(sink.clock_correlation_count, 2);
  EXPECT_FALSE(sink.write_after_end);
  ASSERT_EQ(3u, sink.command_operations.size());
  uint32_t dispatch_operation_count = 0;
  uint32_t return_operation_count = 0;
  for (const auto& operation : sink.command_operations) {
    EXPECT_EQ(sink.command_buffer_ids[0], operation.command_buffer_id);
    switch (operation.type) {
      case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH:
        EXPECT_TRUE(
            iree_hal_profile_command_operation_has_block_structure(&operation));
        EXPECT_EQ(dispatch_operation_count, operation.command_index);
        EXPECT_NE(0u, operation.executable_id);
        EXPECT_NE(
            sink.executable_ids.end(),
            std::find(sink.executable_ids.begin(), sink.executable_ids.end(),
                      operation.executable_id));
        EXPECT_EQ(0u, operation.function_ordinal);
        EXPECT_EQ(2u, operation.binding_count);
        EXPECT_EQ(1u, operation.workgroup_count[0]);
        EXPECT_EQ(1u, operation.workgroup_count[1]);
        EXPECT_EQ(1u, operation.workgroup_count[2]);
        EXPECT_NE(0u, operation.workgroup_size[0]);
        EXPECT_NE(0u,
                  operation.flags &
                      IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_STATIC_BINDINGS);
        ++dispatch_operation_count;
        break;
      case IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_RETURN:
        EXPECT_TRUE(
            iree_hal_profile_command_operation_has_block_structure(&operation));
        EXPECT_EQ(2u, operation.command_index);
        EXPECT_NE(0u, operation.flags &
                          IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_CONTROL_FLOW);
        ++return_operation_count;
        break;
      default:
        FAIL() << "unexpected command operation type " << operation.type;
    }
  }
  EXPECT_EQ(2u, dispatch_operation_count);
  EXPECT_EQ(1u, return_operation_count);
  ASSERT_EQ(2u, sink.dispatch_events.size());
  for (iree_host_size_t i = 0; i < sink.dispatch_events.size(); ++i) {
    const iree_hal_profile_dispatch_event_t& event = sink.dispatch_events[i];
    EXPECT_EQ(sizeof(iree_hal_profile_dispatch_event_t), event.record_length);
    EXPECT_EQ(IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER, event.flags);
    EXPECT_NE(0u, event.event_id);
    EXPECT_NE(0u, event.submission_id);
    EXPECT_NE(0u, event.command_buffer_id);
    EXPECT_NE(
        sink.command_buffer_ids.end(),
        std::find(sink.command_buffer_ids.begin(),
                  sink.command_buffer_ids.end(), event.command_buffer_id));
    EXPECT_NE(0u, event.executable_id);
    EXPECT_NE(sink.executable_ids.end(),
              std::find(sink.executable_ids.begin(), sink.executable_ids.end(),
                        event.executable_id));
    EXPECT_NE(sink.executable_export_ids.end(),
              std::find(sink.executable_export_ids.begin(),
                        sink.executable_export_ids.end(), event.executable_id));
    EXPECT_EQ((uint32_t)i, event.command_index);
    const iree_hal_profile_command_operation_record_t* operation =
        FindCommandOperation(sink, event.command_buffer_id,
                             event.command_index);
    ASSERT_NE(nullptr, operation);
    EXPECT_EQ(IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH,
              operation->type);
    EXPECT_EQ(event.executable_id, operation->executable_id);
    EXPECT_EQ(event.function_ordinal, operation->function_ordinal);
    EXPECT_EQ(0u, event.function_ordinal);
    EXPECT_EQ(1u, event.workgroup_count[0]);
    EXPECT_EQ(1u, event.workgroup_count[1]);
    EXPECT_EQ(1u, event.workgroup_count[2]);
    EXPECT_NE(0u, event.workgroup_size[0]);
  }
  ExpectDispatchEventsHaveClockCorrelations(sink);
}

TEST_F(HostQueueCommandBufferProfilingTest,
       CommandBufferExecuteEmitsQueueDeviceSpansAndRelationships) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status =
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
                          IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS |
                          IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                      CommandBufferProfileSinkAsBase(&sink));
  if (IsQueueDeviceProfilingUnavailable(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "queue-device profiling data family unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

  TwoDispatchCommandBuffer fixture;
  IREE_ASSERT_OK(CreateTwoDispatchCommandBuffer(&test_device, &fixture));

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

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());

  ExpectTwoDispatchOutputs(fixture);

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_FALSE(sink.write_after_end);
  EXPECT_EQ(1, sink.queue_event_count);
  EXPECT_EQ(1, sink.queue_device_event_count);
  EXPECT_EQ(1, sink.relationship_count);
  ASSERT_EQ(1u, sink.queue_events.size());
  ASSERT_EQ(1u, sink.queue_device_events.size());
  ASSERT_EQ(2u, sink.dispatch_events.size());
  ASSERT_EQ(3u, sink.event_relationships.size());

  const iree_hal_profile_queue_event_t& queue_event = sink.queue_events[0];
  EXPECT_EQ(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE, queue_event.type);
  EXPECT_EQ(IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_NONE,
            queue_event.dependency_strategy);
  EXPECT_NE(0u, queue_event.submission_id);
  EXPECT_NE(0u, queue_event.command_buffer_id);
  EXPECT_EQ(0u, queue_event.wait_count);
  EXPECT_EQ(1u, queue_event.signal_count);
  EXPECT_EQ(0u, queue_event.barrier_count);
  EXPECT_EQ(3u, queue_event.operation_count);
  EXPECT_NE(
      sink.command_buffer_ids.end(),
      std::find(sink.command_buffer_ids.begin(), sink.command_buffer_ids.end(),
                queue_event.command_buffer_id));

  const iree_hal_profile_queue_device_event_t& device_event =
      sink.queue_device_events[0];
  EXPECT_EQ(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE, device_event.type);
  EXPECT_EQ(queue_event.submission_id, device_event.submission_id);
  EXPECT_EQ(queue_event.command_buffer_id, device_event.command_buffer_id);
  EXPECT_EQ(queue_event.stream_id, device_event.stream_id);
  EXPECT_EQ(queue_event.physical_device_ordinal,
            device_event.physical_device_ordinal);
  EXPECT_EQ(queue_event.queue_ordinal, device_event.queue_ordinal);
  EXPECT_EQ(queue_event.operation_count, device_event.operation_count);

  for (const auto& dispatch_event : sink.dispatch_events) {
    EXPECT_EQ(queue_event.submission_id, dispatch_event.submission_id);
    EXPECT_EQ(queue_event.command_buffer_id, dispatch_event.command_buffer_id);
    EXPECT_NE(
        nullptr,
        FindEventRelationship(
            sink,
            IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_SUBMISSION_DISPATCH,
            IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_SUBMISSION,
            dispatch_event.submission_id,
            IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_DISPATCH_EVENT,
            dispatch_event.event_id));
  }
  EXPECT_NE(
      nullptr,
      FindEventRelationship(
          sink,
          IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_SUBMISSION_QUEUE_DEVICE_EVENT,
          IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_SUBMISSION,
          device_event.submission_id,
          IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_DEVICE_EVENT,
          device_event.event_id));
  ExpectDispatchEventsHaveClockCorrelations(sink);
}

TEST_F(HostQueueCommandBufferProfilingTest,
       CommandBufferDispatchesEmitHardwareCounterSamples) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status = BeginSqWavesProfiling(&profiling, &sink);
  if (IsHardwareCounterProfilingUnavailable(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "AMDGPU hardware counter profiling unavailable";
  }
  IREE_ASSERT_OK(profiling_status);

  TwoDispatchCommandBuffer fixture;
  IREE_ASSERT_OK(CreateTwoDispatchCommandBuffer(&test_device, &fixture));

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

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());

  ExpectTwoDispatchOutputs(fixture);

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.counter_set_metadata_count);
  EXPECT_EQ(1, sink.counter_metadata_count);
  EXPECT_GE(sink.counter_sample_count, 1);
  ASSERT_EQ(2u, sink.dispatch_events.size());
  ASSERT_EQ(sink.dispatch_events.size(), sink.counter_samples.size());
  iree_host_size_t sample_value_count = 0;
  for (iree_host_size_t i = 0; i < sink.counter_samples.size(); ++i) {
    const iree_hal_profile_dispatch_event_t& event = sink.dispatch_events[i];
    const iree_hal_profile_counter_sample_record_t& sample =
        sink.counter_samples[i];
    EXPECT_TRUE(iree_all_bits_set(
        sample.flags,
        IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DISPATCH_EVENT |
            IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_COMMAND_OPERATION |
            IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DEVICE_TICK_RANGE));
    EXPECT_EQ(IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_DISPATCH, sample.scope);
    EXPECT_EQ(sample.dispatch_event_id, event.event_id);
    EXPECT_EQ(sample.submission_id, event.submission_id);
    EXPECT_EQ(sample.command_buffer_id, event.command_buffer_id);
    EXPECT_EQ(sample.executable_id, event.executable_id);
    EXPECT_EQ(sample.start_tick, event.start_tick);
    EXPECT_EQ(sample.end_tick, event.end_tick);
    EXPECT_EQ(sample.command_index, event.command_index);
    EXPECT_EQ(sample.function_ordinal, event.function_ordinal);
    sample_value_count += sample.sample_value_count;
  }
  ASSERT_EQ(sample_value_count, sink.counter_sample_values.size());
  EXPECT_NE(sink.counter_sample_values.end(),
            std::find_if(sink.counter_sample_values.begin(),
                         sink.counter_sample_values.end(),
                         [](uint64_t value) { return value != 0; }));
}

TEST_F(HostQueueCommandBufferProfilingTest,
       CounterRangeSamplesUseProfileQueueWhenAvailable) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_logical_device_t* logical_device =
      test_device.logical_device();
  ASSERT_GT(logical_device->physical_device_count, 0u);
  for (iree_host_size_t i = 0; i < logical_device->physical_device_count; ++i) {
    ASSERT_GT(logical_device->physical_devices[i]->host_queue_count, 0u);
  }

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status =
      BeginSqWavesCounterRangeProfiling(&profiling, &sink);
  if (IsHardwareCounterProfilingUnavailable(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "AMDGPU hardware counter range profiling unavailable";
  }
  IREE_ASSERT_OK(profiling_status);

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.counter_set_metadata_count);
  EXPECT_EQ(1, sink.counter_metadata_count);
  EXPECT_GE(sink.counter_sample_count, 1);
  ASSERT_FALSE(sink.counter_samples.empty());
  iree_host_size_t sample_value_count = 0;
  for (const iree_hal_profile_counter_sample_record_t& sample :
       sink.counter_samples) {
    EXPECT_TRUE(iree_all_bits_set(
        sample.flags, IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DEVICE_TICK_RANGE));
    EXPECT_EQ(IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_DEVICE_TIME_RANGE,
              sample.scope);
    EXPECT_EQ(0u, sample.dispatch_event_id);
    EXPECT_EQ(0u, sample.submission_id);
    EXPECT_EQ(0u, sample.command_buffer_id);
    EXPECT_EQ(0u, sample.executable_id);
    EXPECT_EQ(UINT32_MAX, sample.command_index);
    EXPECT_EQ(UINT32_MAX, sample.function_ordinal);
    ASSERT_LT(sample.physical_device_ordinal,
              logical_device->physical_device_count);
    const iree_hal_amdgpu_physical_device_t* physical_device =
        logical_device->physical_devices[sample.physical_device_ordinal];
    const uint32_t expected_queue_ordinal =
        (uint32_t)(physical_device->host_queue_count - 1);
    EXPECT_EQ(sample.physical_device_ordinal, physical_device->device_ordinal);
    EXPECT_EQ(expected_queue_ordinal, sample.queue_ordinal);
    EXPECT_LT(sample.start_tick, sample.end_tick);
    sample_value_count += sample.sample_value_count;
  }
  ASSERT_EQ(sample_value_count, sink.counter_sample_values.size());
}

TEST_F(HostQueueCommandBufferProfilingTest,
       ProfilingFlushCounterSampleWriteFailurePreservesSamplesForRetry) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status = BeginSqWavesProfiling(&profiling, &sink);
  if (IsHardwareCounterProfilingUnavailable(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "AMDGPU hardware counter profiling unavailable";
  }
  IREE_ASSERT_OK(profiling_status);

  TwoDispatchCommandBuffer fixture;
  IREE_ASSERT_OK(CreateTwoDispatchCommandBuffer(&test_device, &fixture));

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

  sink.fail_write_content_type = IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES;
  sink.fail_write_remaining = 1;
  sink.fail_write_status_code = IREE_STATUS_UNAVAILABLE;
  iree_status_t flush_status =
      iree_hal_device_profiling_flush(test_device.base_device());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, flush_status);
  EXPECT_EQ(0, sink.counter_sample_count);
  EXPECT_TRUE(sink.counter_samples.empty());

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());

  ExpectTwoDispatchOutputs(fixture);
  EXPECT_GE(sink.counter_sample_count, 1);
  ASSERT_EQ(2u, sink.counter_samples.size());
  EXPECT_GE(sink.dispatch_events.size(), sink.counter_samples.size());
}

TEST_F(HostQueueCommandBufferProfilingTest,
       CommandBufferDispatchProfileFilterSelectsCommandIndex) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  iree_hal_device_profiling_options_t profiling_options = {0};
  profiling_options.data_families =
      IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  profiling_options.sink = CommandBufferProfileSinkAsBase(&sink);
  profiling_options.capture_filter.flags =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_INDEX;
  profiling_options.capture_filter.command_index = 1;
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(&profiling_options));

  TwoDispatchCommandBuffer fixture;
  IREE_ASSERT_OK(CreateTwoDispatchCommandBuffer(&test_device, &fixture));

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

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());

  ExpectTwoDispatchOutputs(fixture);

  ASSERT_EQ(1u, sink.dispatch_events.size());
  const iree_hal_profile_dispatch_event_t& event = sink.dispatch_events[0];
  EXPECT_EQ(IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER, event.flags);
  EXPECT_NE(0u, event.event_id);
  EXPECT_NE(0u, event.submission_id);
  EXPECT_NE(0u, event.command_buffer_id);
  EXPECT_EQ(1u, event.command_index);
}

TEST_F(HostQueueCommandBufferProfilingTest,
       DispatchProfileFilterCopiesExecutableExportPattern) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  std::string export_pattern = "command_buffer_dispatch_*";
  iree_hal_device_profiling_options_t profiling_options = {0};
  profiling_options.data_families =
      IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  profiling_options.sink = CommandBufferProfileSinkAsBase(&sink);
  profiling_options.capture_filter.flags =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN;
  profiling_options.capture_filter.executable_function_pattern =
      iree_make_string_view(export_pattern.data(), export_pattern.size());
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(&profiling_options));

  export_pattern.assign("nomatch");

  iree_hal_executable_cache_t* executable_cache = NULL;
  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable_cache, &executable));

  const uint64_t executable_id =
      iree_hal_amdgpu_executable_profile_id(executable);
  EXPECT_TRUE(iree_hal_amdgpu_logical_device_should_profile_dispatch(
      test_device.logical_device(), executable_id, /*function_ordinal=*/0,
      /*command_buffer_id=*/0, /*command_index=*/0,
      /*physical_device_ordinal=*/0, /*queue_ordinal=*/0));

  IREE_ASSERT_OK(profiling.End());
  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}

TEST_F(HostQueueCommandBufferProfilingTest,
       ProfiledDispatchReservationFailsWhenRingFull) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(nullptr, queue);

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation = {0};
  iree_hal_amdgpu_profile_dispatch_event_reservation_t exhausted_reservation = {
      0};
  const uint32_t dispatch_event_capacity =
      iree_hal_amdgpu_host_queue_profile_dispatch_event_capacity(queue);
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  iree_status_t status =
      iree_hal_amdgpu_host_queue_reserve_profile_dispatch_events(
          queue, dispatch_event_capacity, &reservation);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_reserve_profile_dispatch_events(
        queue, 1, &exhausted_reservation);
  }
  iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue, reservation);
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  EXPECT_EQ(dispatch_event_capacity, reservation.event_count);
  EXPECT_EQ(0u, exhausted_reservation.event_count);

  IREE_ASSERT_OK(profiling.End());
}

TEST_F(HostQueueCommandBufferProfilingTest,
       ProfiledQueueDeviceReservationFailsWhenRingFull) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(nullptr, queue);

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  iree_status_t profiling_status =
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS,
                      CommandBufferProfileSinkAsBase(&sink));
  if (IsQueueDeviceProfilingUnavailable(profiling_status)) {
    iree_status_free(profiling_status);
    GTEST_SKIP() << "queue-device profiling data family unsupported by backend";
  }
  IREE_ASSERT_OK(profiling_status);

  iree_hal_amdgpu_profile_queue_device_event_reservation_t reservation = {0};
  iree_hal_amdgpu_profile_queue_device_event_reservation_t
      exhausted_reservation = {0};
  const uint32_t queue_device_event_capacity =
      queue->profiling.queue_device_events.capacity;
  ASSERT_GT(queue_device_event_capacity, 0u);
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  iree_status_t status =
      iree_hal_amdgpu_host_queue_reserve_profile_queue_device_events(
          queue, queue_device_event_capacity, &reservation);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_reserve_profile_queue_device_events(
        queue, 1, &exhausted_reservation);
  }
  iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(queue,
                                                                reservation);
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  EXPECT_EQ(queue_device_event_capacity, reservation.event_count);
  EXPECT_EQ(0u, exhausted_reservation.event_count);

  IREE_ASSERT_OK(profiling.End());
}

TEST_F(HostQueueCommandBufferProfilingTest,
       ProfiledCommandBufferDispatchSignalsSurviveAqlSlotReuse) {
  static constexpr uint32_t kAqlCapacity = 64;
  static constexpr uint32_t kDispatchCount = kAqlCapacity + 32;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.host_block_pools.command_buffer.usable_block_size =
      IREE_HAL_AMDGPU_AQL_PROGRAM_MIN_BLOCK_SIZE;
  options.host_queues.aql_capacity = kAqlCapacity;
  options.host_queues.kernarg_capacity = 2 * kAqlCapacity;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  iree_hal_executable_cache_t* executable_cache = NULL;
  iree_hal_executable_t* executable = NULL;
  IREE_ASSERT_OK(LoadCtsExecutable(
      test_device.base_device(),
      iree_make_cstring_view("command_buffer_dispatch_constants_bindings_test."
                             "bin"),
      &executable_cache, &executable));

  Ref<iree_hal_buffer_t> input_buffer;
  const uint32_t input_values[4] = {1, 2, 3, 4};
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      input_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(input_buffer, /*target_offset=*/0,
                                           input_values, sizeof(input_values)));

  Ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(CreateHostVisibleDispatchBuffer(
      test_device.allocator(), /*buffer_size=*/4 * sizeof(uint32_t),
      output_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(output_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(),
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
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
  for (uint32_t i = 0; i < kDispatchCount; ++i) {
    IREE_ASSERT_OK(
        AppendConstantsBindingsDispatch(command_buffer, executable, bindings));
  }
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

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

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());

  const uint32_t expected_values[4] = {13, 16, 19, 22};
  uint32_t output_values[4] = {0, 0, 0, 0};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      output_buffer, /*offset=*/0, output_values, sizeof(output_values)));
  EXPECT_EQ(0, memcmp(output_values, expected_values, sizeof(expected_values)));

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_FALSE(sink.write_after_end);
  ASSERT_EQ(kDispatchCount, sink.dispatch_events.size());
  for (iree_host_size_t i = 0; i < sink.dispatch_events.size(); ++i) {
    const iree_hal_profile_dispatch_event_t& event = sink.dispatch_events[i];
    EXPECT_EQ(sizeof(iree_hal_profile_dispatch_event_t), event.record_length);
    EXPECT_EQ(IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER, event.flags);
    EXPECT_NE(0u, event.event_id);
    EXPECT_NE(0u, event.submission_id);
  }

  iree_hal_executable_release(executable);
  iree_hal_executable_cache_release(executable_cache);
}

TEST_F(HostQueueCommandBufferProfilingTest,
       MetadataOnlyCommandBufferParksAndResumesUnderNotificationPressure) {
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

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  Ref<iree_hal_buffer_t> pressure_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), pressure_buffer.out()));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(),
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  ASSERT_EQ(program->max_block_aql_packet_count, 0u);

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
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_execute(
        test_device.base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
        iree_hal_semaphore_list_empty(), command_buffer_signal_list,
        command_buffer, iree_hal_buffer_binding_table_empty(),
        IREE_HAL_EXECUTE_FLAG_NONE);
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
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());
  EXPECT_TRUE(replay_parked);

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.command_buffer_metadata_count);
  EXPECT_EQ(1, sink.queue_event_count);
  ASSERT_EQ(1u, sink.command_buffer_ids.size());
  EXPECT_EQ(1u, CountQueueEvents(sink, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL));
  EXPECT_EQ(1u,
            CountQueueEvents(sink, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE));
  const iree_hal_profile_queue_event_t* execute_event =
      FindUniqueQueueEvent(sink, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE);
  ASSERT_NE(nullptr, execute_event);
  EXPECT_EQ(sink.command_buffer_ids[0], execute_event->command_buffer_id);
  EXPECT_EQ(0u, execute_event->operation_count);
  EXPECT_EQ(1u, execute_event->signal_count);
  EXPECT_LE(sink.command_operations.size(), 1u);
  for (const auto& operation : sink.command_operations) {
    EXPECT_EQ(sink.command_buffer_ids[0], operation.command_buffer_id);
    EXPECT_EQ(IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_RETURN, operation.type);
  }
}

TEST_F(HostQueueCommandBufferProfilingTest,
       LargeCommandBufferParksAndResumesUnderNotificationPressure) {
  static constexpr uint32_t kFillCount = 2048;
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

  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device.base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));

  Ref<iree_hal_buffer_t> pressure_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), sizeof(uint32_t), pressure_buffer.out()));

  const iree_device_size_t target_buffer_size = kFillCount * sizeof(uint32_t);
  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(CreateHostVisibleTransferBuffer(
      test_device.allocator(), target_buffer_size, target_buffer.out()));
  IREE_ASSERT_OK(iree_hal_buffer_map_zero(target_buffer, /*offset=*/0,
                                          IREE_HAL_WHOLE_BUFFER));

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      test_device.base_device(),
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/1, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  std::vector<uint32_t> expected(kFillCount);
  for (uint32_t i = 0; i < kFillCount; ++i) {
    expected[i] = 0xBD3A0000u | i;
    IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
        command_buffer,
        iree_hal_make_indirect_buffer_ref(/*binding=*/0, i * sizeof(uint32_t),
                                          sizeof(uint32_t)),
        &expected[i], sizeof(expected[i]), IREE_HAL_FILL_FLAG_NONE));
  }
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  ASSERT_GT(program->block_count, 1u);
  ASSERT_GT(kFillCount, kAqlCapacity);
  ASSERT_LE(program->max_block_aql_packet_count, kAqlCapacity);

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
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(test_device.base_device()));
  IREE_ASSERT_OK(profiling.End());
  EXPECT_TRUE(replay_parked);

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.command_buffer_metadata_count);
  EXPECT_EQ(1, sink.queue_event_count);
  ASSERT_EQ(1u, sink.command_buffer_ids.size());
  EXPECT_EQ(1u, CountQueueEvents(sink, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL));
  EXPECT_EQ(program->block_count,
            CountQueueEvents(sink, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE));
  EXPECT_EQ(program->command_count,
            SumQueueEventOperationCounts(
                sink, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE));
  uint32_t execute_signal_count = 0;
  for (const auto& event : sink.queue_events) {
    if (event.type != IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE) continue;
    EXPECT_EQ(sink.command_buffer_ids[0], event.command_buffer_id);
    execute_signal_count += event.signal_count;
  }
  EXPECT_EQ(1u, execute_signal_count);

  std::vector<uint32_t> actual(kFillCount);
  IREE_ASSERT_OK(iree_hal_buffer_map_read(target_buffer, /*offset=*/0,
                                          actual.data(), target_buffer_size));
  EXPECT_EQ(actual, expected);
}

}  // namespace
}  // namespace iree::hal::amdgpu
