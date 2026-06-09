// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PROFILING_TEST_UTIL_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PROFILING_TEST_UTIL_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_test_util.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile_events.h"

namespace iree::hal::amdgpu::test {

static bool IsProfilingUnsupported(iree_status_t status) {
  return iree_status_is_unimplemented(status) ||
         iree_status_is_invalid_argument(status);
}

static iree_status_t SubmitProfiledQueueFill(TestLogicalDevice* test_device) {
  Ref<iree_hal_buffer_t> target_buffer;
  IREE_RETURN_IF_ERROR(CreateHostVisibleTransferBuffer(
      test_device->allocator(), sizeof(uint32_t), target_buffer.out()));

  Ref<iree_hal_semaphore_t> signal;
  IREE_RETURN_IF_ERROR(
      CreateSemaphore(test_device->base_device(), signal.out()));
  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_ptr = signal.get();
  const iree_hal_semaphore_list_t signal_list = {
      /*count=*/1,
      /*semaphores=*/&signal_ptr,
      /*payload_values=*/&signal_value,
  };
  const uint32_t pattern = 0xA11CA7E5u;
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_fill(
      test_device->base_device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_list, target_buffer,
      /*target_offset=*/0, sizeof(pattern), &pattern, sizeof(pattern),
      IREE_HAL_FILL_FLAG_NONE));
  return iree_hal_semaphore_wait(signal, signal_value, iree_infinite_timeout(),
                                 IREE_ASYNC_WAIT_FLAG_NONE);
}

class DeviceProfilingScope {
 public:
  explicit DeviceProfilingScope(iree_hal_device_t* device) : device_(device) {}

  ~DeviceProfilingScope() {
    if (is_active_) {
      IREE_EXPECT_OK(iree_hal_device_profiling_end(device_));
    }
  }

  iree_status_t Begin(iree_hal_device_profiling_data_families_t data_families,
                      iree_hal_profile_sink_t* sink) {
    iree_hal_device_profiling_options_t options = {0};
    options.data_families = data_families;
    options.sink = sink;
    return Begin(&options);
  }

  iree_status_t Begin(const iree_hal_device_profiling_options_t* options) {
    iree_status_t status = iree_hal_device_profiling_begin(device_, options);
    if (iree_status_is_ok(status)) {
      is_active_ = true;
    }
    return status;
  }

  iree_status_t End() {
    if (!is_active_) return iree_ok_status();
    is_active_ = false;
    return iree_hal_device_profiling_end(device_);
  }

 private:
  // Device whose profiling session is active.
  iree_hal_device_t* device_ = nullptr;

  // True when |device_| has an active profiling session owned by this scope.
  bool is_active_ = false;
};

struct CommandBufferProfileSink {
  // HAL resource header for the profile sink.
  iree_hal_resource_t resource;

  // Number of session begin notifications observed.
  int begin_count = 0;

  // Number of session end notifications observed.
  int end_count = 0;

  // Number of device metadata chunks observed.
  int device_metadata_count = 0;

  // Number of queue metadata chunks observed.
  int queue_metadata_count = 0;

  // Number of executable metadata chunks observed.
  int executable_metadata_count = 0;

  // Number of executable function metadata chunks observed.
  int executable_function_metadata_count = 0;

  // Number of command-buffer metadata chunks observed.
  int command_buffer_metadata_count = 0;

  // Number of command-operation metadata chunks observed.
  int command_operation_metadata_count = 0;

  // Number of clock correlation chunks observed.
  int clock_correlation_count = 0;

  // Number of host queue event chunks observed.
  int queue_event_count = 0;

  // Number of device queue event chunks observed.
  int queue_device_event_count = 0;

  // Number of memory event chunks observed.
  int memory_event_count = 0;

  // Number of event relationship chunks observed.
  int relationship_count = 0;

  // Number of counter set metadata chunks observed.
  int counter_set_metadata_count = 0;

  // Number of counter metadata chunks observed.
  int counter_metadata_count = 0;

  // Number of counter sample chunks observed.
  int counter_sample_count = 0;

  // Number of chunks marked truncated by the producer.
  int truncated_chunk_count = 0;

  // Total dropped records reported by truncated chunks.
  uint64_t dropped_record_count = 0;

  // Dropped queue event records reported by QUEUE_EVENTS chunks.
  uint64_t queue_event_dropped_record_count = 0;

  // Dropped memory event records reported by MEMORY_EVENTS chunks.
  uint64_t memory_event_dropped_record_count = 0;

  // Device metadata records copied from DEVICES chunks.
  std::vector<iree_hal_profile_device_record_t> device_records;

  // Queue metadata records copied from QUEUES chunks.
  std::vector<iree_hal_profile_queue_record_t> queue_records;

  // Executable identifiers copied from EXECUTABLES chunks.
  std::vector<uint64_t> executable_ids;

  // Executable identifiers copied from EXECUTABLE_EXPORTS chunks.
  std::vector<uint64_t> executable_export_ids;

  // Command-buffer identifiers copied from COMMAND_BUFFERS chunks.
  std::vector<uint64_t> command_buffer_ids;

  // Command operations copied from COMMAND_OPERATIONS chunks.
  std::vector<iree_hal_profile_command_operation_record_t> command_operations;

  // Clock correlation records copied from CLOCK_CORRELATIONS chunks.
  std::vector<iree_hal_profile_clock_correlation_record_t> clock_correlations;

  // Host queue events copied from QUEUE_EVENTS chunks.
  std::vector<iree_hal_profile_queue_event_t> queue_events;

  // Device queue events copied from QUEUE_DEVICE_EVENTS chunks.
  std::vector<iree_hal_profile_queue_device_event_t> queue_device_events;

  // Memory events copied from MEMORY_EVENTS chunks.
  std::vector<iree_hal_profile_memory_event_t> memory_events;

  // Event relationships copied from EVENT_RELATIONSHIPS chunks.
  std::vector<iree_hal_profile_event_relationship_record_t> event_relationships;

  // Dispatch events copied from DISPATCH_EVENTS chunks.
  std::vector<iree_hal_profile_dispatch_event_t> dispatch_events;

  // Counter sample records copied from COUNTER_SAMPLES chunks.
  std::vector<iree_hal_profile_counter_sample_record_t> counter_samples;

  // Counter sample values copied from COUNTER_SAMPLES chunks.
  std::vector<uint64_t> counter_sample_values;

  // Counter set metadata records copied from COUNTER_SETS chunks.
  std::vector<iree_hal_profile_counter_set_record_t> counter_set_records;

  // Counter metadata records copied from COUNTERS chunks.
  std::vector<iree_hal_profile_counter_record_t> counter_records;

  // Physical device ordinals for entries in |dispatch_events|.
  std::vector<uint32_t> dispatch_event_physical_device_ordinals;

  // Session identifier observed at begin and expected on later callbacks.
  uint64_t session_id = 0;

  // True if the backend writes after ending the profiling session.
  bool write_after_end = false;

  // Status code returned from begin_session, or OK for success.
  iree_status_code_t fail_begin_session_status_code = IREE_STATUS_OK;

  // Content type whose write callback should fail, or empty when disabled.
  iree_string_view_t fail_write_content_type = {nullptr, 0};

  // Number of matching write callbacks that should fail.
  int fail_write_remaining = 0;

  // Status code returned from matching write callbacks.
  iree_status_code_t fail_write_status_code = IREE_STATUS_OK;

  // Expected session status code passed to end_session.
  iree_status_code_t expected_end_session_status_code = IREE_STATUS_OK;

  // Status code observed by the most recent end_session callback.
  iree_status_code_t observed_end_session_status_code = IREE_STATUS_OK;

  // Status code returned from end_session, or OK for success.
  iree_status_code_t fail_end_session_status_code = IREE_STATUS_OK;
};

static CommandBufferProfileSink* CommandBufferProfileSinkCast(
    iree_hal_profile_sink_t* sink) {
  return reinterpret_cast<CommandBufferProfileSink*>(sink);
}

static void CommandBufferProfileSinkDestroy(iree_hal_profile_sink_t* sink) {
  (void)sink;
}

static iree_status_t CommandBufferProfileSinkBeginSession(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  CommandBufferProfileSink* test_sink = CommandBufferProfileSinkCast(sink);
  EXPECT_EQ(0, test_sink->begin_count);
  EXPECT_EQ(0, test_sink->end_count);
  EXPECT_TRUE(iree_string_view_equal(metadata->content_type,
                                     IREE_HAL_PROFILE_CONTENT_TYPE_SESSION));
  ++test_sink->begin_count;
  if (test_sink->fail_begin_session_status_code != IREE_STATUS_OK) {
    return iree_make_status(test_sink->fail_begin_session_status_code,
                            "injected profile sink begin_session failure");
  }
  test_sink->session_id = metadata->session_id;
  return iree_ok_status();
}

static iree_status_t CommandBufferProfileSinkWrite(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  CommandBufferProfileSink* test_sink = CommandBufferProfileSinkCast(sink);
  EXPECT_EQ(1, test_sink->begin_count);
  EXPECT_EQ(0, test_sink->end_count);
  if (test_sink->end_count != 0) test_sink->write_after_end = true;
  EXPECT_EQ(test_sink->session_id, metadata->session_id);
  if (test_sink->fail_write_remaining != 0 &&
      iree_string_view_equal(metadata->content_type,
                             test_sink->fail_write_content_type)) {
    --test_sink->fail_write_remaining;
    return iree_make_status(test_sink->fail_write_status_code,
                            "injected profile sink write failure");
  }
  const bool is_truncated =
      iree_any_bit_set(metadata->flags, IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED);
  if (is_truncated) {
    ++test_sink->truncated_chunk_count;
    test_sink->dropped_record_count += metadata->dropped_record_count;
  }
  if (iovec_count == 0) {
    if (iree_string_view_equal(metadata->content_type,
                               IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
      test_sink->queue_event_dropped_record_count +=
          metadata->dropped_record_count;
      ++test_sink->queue_event_count;
    } else if (iree_string_view_equal(
                   metadata->content_type,
                   IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS)) {
      test_sink->memory_event_dropped_record_count +=
          metadata->dropped_record_count;
      ++test_sink->memory_event_count;
    }
    return iree_ok_status();
  }
  if (iovec_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected exactly one profile chunk iovec");
  }

  if (iree_string_view_equal(metadata->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES)) {
    EXPECT_EQ(0u,
              iovecs[0].data_length % sizeof(iree_hal_profile_device_record_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_device_record_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length / sizeof(iree_hal_profile_device_record_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_device_record_t),
                records[i].record_length);
      EXPECT_NE(UINT32_MAX, records[i].physical_device_ordinal);
      EXPECT_GT(records[i].queue_count, 0u);
    }
    test_sink->device_records.insert(test_sink->device_records.end(), records,
                                     records + record_count);
    ++test_sink->device_metadata_count;
  } else if (iree_string_view_equal(metadata->content_type,
                                    IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES)) {
    EXPECT_EQ(0u,
              iovecs[0].data_length % sizeof(iree_hal_profile_queue_record_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_queue_record_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length / sizeof(iree_hal_profile_queue_record_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_queue_record_t),
                records[i].record_length);
      EXPECT_NE(UINT32_MAX, records[i].physical_device_ordinal);
      EXPECT_NE(UINT32_MAX, records[i].queue_ordinal);
    }
    test_sink->queue_records.insert(test_sink->queue_records.end(), records,
                                    records + record_count);
    ++test_sink->queue_metadata_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES)) {
    EXPECT_EQ(0u, iovecs[0].data_length %
                      sizeof(iree_hal_profile_executable_record_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_executable_record_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length / sizeof(iree_hal_profile_executable_record_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_executable_record_t),
                records[i].record_length);
      EXPECT_NE(0u, records[i].executable_id);
      EXPECT_GT(records[i].function_count, 0u);
      EXPECT_NE(0u, records[i].flags &
                        IREE_HAL_PROFILE_EXECUTABLE_FLAG_CODE_OBJECT_HASH);
      EXPECT_NE(
          0u, records[i].code_object_hash[0] | records[i].code_object_hash[1]);
      test_sink->executable_ids.push_back(records[i].executable_id);
    }
    ++test_sink->executable_metadata_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_FUNCTIONS)) {
    iree_host_size_t payload_offset = 0;
    while (payload_offset < iovecs[0].data_length) {
      if (iovecs[0].data_length - payload_offset <
          sizeof(iree_hal_profile_executable_function_record_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "truncated executable function profile record");
      }
      iree_hal_profile_executable_function_record_t record;
      memcpy(&record, iovecs[0].data + payload_offset, sizeof(record));
      if (record.record_length < sizeof(record) ||
          record.record_length > iovecs[0].data_length - payload_offset) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "invalid executable function profile record");
      }
      EXPECT_NE(0u, record.executable_id);
      EXPECT_NE(UINT32_MAX, record.function_ordinal);
      EXPECT_NE(0u,
                record.flags &
                    IREE_HAL_PROFILE_EXECUTABLE_FUNCTION_FLAG_FUNCTION_HASH);
      EXPECT_NE(0u, record.function_hash[0] | record.function_hash[1]);
      EXPECT_EQ(record.name_length,
                record.record_length - (uint32_t)sizeof(record));
      test_sink->executable_export_ids.push_back(record.executable_id);
      payload_offset += record.record_length;
    }
    ++test_sink->executable_function_metadata_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS)) {
    EXPECT_EQ(0u, iovecs[0].data_length %
                      sizeof(iree_hal_profile_command_buffer_record_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_command_buffer_record_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length /
        sizeof(iree_hal_profile_command_buffer_record_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_command_buffer_record_t),
                records[i].record_length);
      EXPECT_NE(0u, records[i].command_buffer_id);
      EXPECT_NE(UINT32_MAX, records[i].physical_device_ordinal);
      test_sink->command_buffer_ids.push_back(records[i].command_buffer_id);
    }
    ++test_sink->command_buffer_metadata_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS)) {
    EXPECT_EQ(0u, iovecs[0].data_length %
                      sizeof(iree_hal_profile_command_operation_record_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_command_operation_record_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length /
        sizeof(iree_hal_profile_command_operation_record_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_command_operation_record_t),
                records[i].record_length);
      EXPECT_NE(IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_NONE, records[i].type);
      EXPECT_NE(UINT32_MAX, records[i].command_index);
      EXPECT_NE(0u, records[i].command_buffer_id);
      if (iree_hal_profile_command_operation_has_block_structure(&records[i])) {
        EXPECT_NE(UINT32_MAX, records[i].block_ordinal);
        EXPECT_NE(UINT32_MAX, records[i].block_command_ordinal);
      } else {
        EXPECT_EQ(UINT32_MAX, records[i].block_ordinal);
        EXPECT_EQ(UINT32_MAX, records[i].block_command_ordinal);
      }
      if (records[i].type == IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH) {
        EXPECT_NE(0u, records[i].executable_id);
        EXPECT_NE(UINT32_MAX, records[i].function_ordinal);
        EXPECT_NE(0u, records[i].binding_count);
        EXPECT_NE(0u, records[i].workgroup_size[0]);
      }
    }
    test_sink->command_operations.insert(test_sink->command_operations.end(),
                                         records, records + record_count);
    ++test_sink->command_operation_metadata_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_CLOCK_CORRELATIONS)) {
    EXPECT_EQ(0u, iovecs[0].data_length %
                      sizeof(iree_hal_profile_clock_correlation_record_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_clock_correlation_record_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length /
        sizeof(iree_hal_profile_clock_correlation_record_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_clock_correlation_record_t),
                records[i].record_length);
      EXPECT_NE(UINT32_MAX, records[i].physical_device_ordinal);
      EXPECT_NE(0u, records[i].sample_id);
      EXPECT_TRUE(iree_all_bits_set(
          records[i].flags,
          IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP |
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_SYSTEM_TIMESTAMP |
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET));
      EXPECT_NE(0u, records[i].device_tick);
      EXPECT_NE(0u, records[i].host_cpu_timestamp_ns);
      EXPECT_NE(0u, records[i].host_system_timestamp);
      EXPECT_NE(0u, records[i].host_system_frequency_hz);
      EXPECT_LE(records[i].host_time_begin_ns, records[i].host_time_end_ns);
    }
    test_sink->clock_correlations.insert(test_sink->clock_correlations.end(),
                                         records, records + record_count);
    ++test_sink->clock_correlation_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SETS)) {
    iree_host_size_t payload_offset = 0;
    while (payload_offset < iovecs[0].data_length) {
      if (iovecs[0].data_length - payload_offset <
          sizeof(iree_hal_profile_counter_set_record_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "truncated counter set profile record");
      }
      iree_hal_profile_counter_set_record_t record;
      memcpy(&record, iovecs[0].data + payload_offset, sizeof(record));
      if (record.record_length < sizeof(record) ||
          record.record_length > iovecs[0].data_length - payload_offset) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "invalid counter set profile record");
      }
      EXPECT_NE(0u, record.counter_set_id);
      EXPECT_GT(record.counter_count, 0u);
      EXPECT_GT(record.sample_value_count, 0u);
      EXPECT_EQ(record.name_length,
                record.record_length - (uint32_t)sizeof(record));
      test_sink->counter_set_records.push_back(record);
      payload_offset += record.record_length;
    }
    ++test_sink->counter_set_metadata_count;
  } else if (iree_string_view_equal(metadata->content_type,
                                    IREE_HAL_PROFILE_CONTENT_TYPE_COUNTERS)) {
    iree_host_size_t payload_offset = 0;
    while (payload_offset < iovecs[0].data_length) {
      if (iovecs[0].data_length - payload_offset <
          sizeof(iree_hal_profile_counter_record_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "truncated counter profile record");
      }
      iree_hal_profile_counter_record_t record;
      memcpy(&record, iovecs[0].data + payload_offset, sizeof(record));
      if (record.record_length < sizeof(record) ||
          record.record_length > iovecs[0].data_length - payload_offset) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "invalid counter profile record");
      }
      EXPECT_NE(0u, record.counter_set_id);
      EXPECT_GT(record.sample_value_count, 0u);
      const uint32_t string_length = record.block_name_length +
                                     record.name_length +
                                     record.description_length;
      EXPECT_EQ(string_length, record.record_length - (uint32_t)sizeof(record));
      test_sink->counter_records.push_back(record);
      payload_offset += record.record_length;
    }
    ++test_sink->counter_metadata_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS)) {
    EXPECT_NE(UINT32_MAX, metadata->physical_device_ordinal);
    EXPECT_NE(UINT32_MAX, metadata->queue_ordinal);
    EXPECT_EQ(
        0u, iovecs[0].data_length % sizeof(iree_hal_profile_dispatch_event_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_dispatch_event_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length / sizeof(iree_hal_profile_dispatch_event_t);
    EXPECT_GT(record_count, 0u);
    test_sink->dispatch_events.insert(test_sink->dispatch_events.end(), records,
                                      records + record_count);
    test_sink->dispatch_event_physical_device_ordinals.insert(
        test_sink->dispatch_event_physical_device_ordinals.end(), record_count,
        metadata->physical_device_ordinal);
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
    EXPECT_EQ(0u,
              iovecs[0].data_length % sizeof(iree_hal_profile_queue_event_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_queue_event_t*>(iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length / sizeof(iree_hal_profile_queue_event_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_queue_event_t),
                records[i].record_length);
      EXPECT_NE(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE, records[i].type);
      EXPECT_NE(0u, records[i].event_id);
      EXPECT_NE(0, records[i].host_time_ns);
      EXPECT_NE(UINT32_MAX, records[i].physical_device_ordinal);
      EXPECT_NE(UINT32_MAX, records[i].queue_ordinal);
    }
    test_sink->queue_events.insert(test_sink->queue_events.end(), records,
                                   records + record_count);
    test_sink->queue_event_dropped_record_count +=
        metadata->dropped_record_count;
    ++test_sink->queue_event_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_DEVICE_EVENTS)) {
    EXPECT_NE(UINT32_MAX, metadata->physical_device_ordinal);
    EXPECT_NE(UINT32_MAX, metadata->queue_ordinal);
    EXPECT_EQ(0u, iovecs[0].data_length %
                      sizeof(iree_hal_profile_queue_device_event_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_queue_device_event_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length / sizeof(iree_hal_profile_queue_device_event_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_queue_device_event_t),
                records[i].record_length);
      EXPECT_NE(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE, records[i].type);
      EXPECT_NE(0u, records[i].event_id);
      EXPECT_NE(0u, records[i].submission_id);
      EXPECT_NE(UINT32_MAX, records[i].physical_device_ordinal);
      EXPECT_NE(UINT32_MAX, records[i].queue_ordinal);
      EXPECT_NE(0u, records[i].start_tick);
      EXPECT_NE(0u, records[i].end_tick);
      EXPECT_GE(records[i].end_tick, records[i].start_tick);
    }
    test_sink->queue_device_events.insert(test_sink->queue_device_events.end(),
                                          records, records + record_count);
    ++test_sink->queue_device_event_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS)) {
    EXPECT_EQ(0u,
              iovecs[0].data_length % sizeof(iree_hal_profile_memory_event_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_memory_event_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length / sizeof(iree_hal_profile_memory_event_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_memory_event_t),
                records[i].record_length);
      EXPECT_NE(IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_NONE, records[i].type);
      EXPECT_NE(0u, records[i].event_id);
      EXPECT_NE(0, records[i].host_time_ns);
      EXPECT_NE(0u, records[i].allocation_id);
    }
    test_sink->memory_events.insert(test_sink->memory_events.end(), records,
                                    records + record_count);
    test_sink->memory_event_dropped_record_count +=
        metadata->dropped_record_count;
    ++test_sink->memory_event_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_EVENT_RELATIONSHIPS)) {
    EXPECT_NE(UINT32_MAX, metadata->physical_device_ordinal);
    EXPECT_NE(UINT32_MAX, metadata->queue_ordinal);
    EXPECT_EQ(0u, iovecs[0].data_length %
                      sizeof(iree_hal_profile_event_relationship_record_t));
    const auto* records =
        reinterpret_cast<const iree_hal_profile_event_relationship_record_t*>(
            iovecs[0].data);
    const iree_host_size_t record_count =
        iovecs[0].data_length /
        sizeof(iree_hal_profile_event_relationship_record_t);
    EXPECT_GT(record_count, 0u);
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      EXPECT_EQ(sizeof(iree_hal_profile_event_relationship_record_t),
                records[i].record_length);
      EXPECT_NE(IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_NONE, records[i].type);
      EXPECT_NE(0u, records[i].relationship_id);
      EXPECT_NE(IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_NONE,
                records[i].source_type);
      EXPECT_NE(IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_NONE,
                records[i].target_type);
      EXPECT_NE(UINT32_MAX, records[i].physical_device_ordinal);
      EXPECT_NE(UINT32_MAX, records[i].queue_ordinal);
      EXPECT_NE(0u, records[i].source_id);
      EXPECT_NE(0u, records[i].target_id);
    }
    test_sink->event_relationships.insert(test_sink->event_relationships.end(),
                                          records, records + record_count);
    ++test_sink->relationship_count;
  } else if (iree_string_view_equal(
                 metadata->content_type,
                 IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES)) {
    iree_host_size_t payload_offset = 0;
    while (payload_offset < iovecs[0].data_length) {
      if (iovecs[0].data_length - payload_offset <
          sizeof(iree_hal_profile_counter_sample_record_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "truncated counter sample profile record");
      }
      iree_hal_profile_counter_sample_record_t record;
      memcpy(&record, iovecs[0].data + payload_offset, sizeof(record));
      if (record.record_length < sizeof(record) ||
          record.record_length > iovecs[0].data_length - payload_offset) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "invalid counter sample profile record");
      }
      EXPECT_NE(0u, record.sample_id);
      EXPECT_NE(0u, record.counter_set_id);
      switch (record.scope) {
        case IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_DISPATCH:
          EXPECT_NE(0u, record.dispatch_event_id);
          break;
        case IREE_HAL_PROFILE_COUNTER_SAMPLE_SCOPE_DEVICE_TIME_RANGE:
          EXPECT_EQ(0u, record.dispatch_event_id);
          EXPECT_TRUE(iree_any_bit_set(
              record.flags,
              IREE_HAL_PROFILE_COUNTER_SAMPLE_FLAG_DEVICE_TICK_RANGE));
          break;
        default:
          ADD_FAILURE() << "unexpected counter sample scope " << record.scope;
          break;
      }
      EXPECT_GT(record.sample_value_count, 0u);
      EXPECT_EQ(record.record_length,
                sizeof(record) +
                    record.sample_value_count * (uint32_t)sizeof(uint64_t));
      const auto* values = reinterpret_cast<const uint64_t*>(
          iovecs[0].data + payload_offset + sizeof(record));
      test_sink->counter_sample_values.insert(
          test_sink->counter_sample_values.end(), values,
          values + record.sample_value_count);
      test_sink->counter_samples.push_back(record);
      payload_offset += record.record_length;
    }
    ++test_sink->counter_sample_count;
  }

  return iree_ok_status();
}

static iree_status_t CommandBufferProfileSinkEndSession(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  CommandBufferProfileSink* test_sink = CommandBufferProfileSinkCast(sink);
  EXPECT_EQ(1, test_sink->begin_count);
  EXPECT_EQ(0, test_sink->end_count);
  EXPECT_TRUE(iree_string_view_equal(metadata->content_type,
                                     IREE_HAL_PROFILE_CONTENT_TYPE_SESSION));
  EXPECT_EQ(test_sink->session_id, metadata->session_id);
  EXPECT_EQ(test_sink->expected_end_session_status_code, session_status_code);
  test_sink->observed_end_session_status_code = session_status_code;
  test_sink->end_count = 1;
  if (test_sink->fail_end_session_status_code != IREE_STATUS_OK) {
    return iree_make_status(test_sink->fail_end_session_status_code,
                            "injected profile sink end_session failure");
  }
  return iree_ok_status();
}

static const iree_hal_profile_sink_vtable_t kCommandBufferProfileSinkVTable = {
    /*.destroy=*/CommandBufferProfileSinkDestroy,
    /*.begin_session=*/CommandBufferProfileSinkBeginSession,
    /*.write=*/CommandBufferProfileSinkWrite,
    /*.end_session=*/CommandBufferProfileSinkEndSession,
};

static void CommandBufferProfileSinkInitialize(CommandBufferProfileSink* sink) {
  iree_hal_resource_initialize(&kCommandBufferProfileSinkVTable,
                               &sink->resource);
}

static iree_hal_profile_sink_t* CommandBufferProfileSinkAsBase(
    CommandBufferProfileSink* sink) {
  return reinterpret_cast<iree_hal_profile_sink_t*>(sink);
}

static void ExpectQueueEventProfilingCanBeginAndEnd(
    TestLogicalDevice* test_device) {
  CommandBufferProfileSink sink = {};
  CommandBufferProfileSinkInitialize(&sink);
  DeviceProfilingScope profiling(test_device->base_device());
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 CommandBufferProfileSinkAsBase(&sink)));
  IREE_ASSERT_OK(profiling.End());
  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
}

static void ExpectDispatchEventsHaveClockCorrelations(
    const CommandBufferProfileSink& sink) {
  ASSERT_GE(sink.clock_correlations.size(), 2u);
  ASSERT_EQ(sink.dispatch_events.size(),
            sink.dispatch_event_physical_device_ordinals.size());
  for (iree_host_size_t event_index = 0;
       event_index < sink.dispatch_events.size(); ++event_index) {
    const uint32_t physical_device_ordinal =
        sink.dispatch_event_physical_device_ordinals[event_index];
    uint64_t previous_sample_id = 0;
    uint64_t previous_device_tick = 0;
    iree_host_size_t correlation_count = 0;
    for (const iree_hal_profile_clock_correlation_record_t& correlation :
         sink.clock_correlations) {
      if (correlation.physical_device_ordinal != physical_device_ordinal ||
          !iree_any_bit_set(
              correlation.flags,
              IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK)) {
        continue;
      }
      if (previous_sample_id != 0) {
        EXPECT_GT(correlation.sample_id, previous_sample_id);
        EXPECT_GE(correlation.device_tick, previous_device_tick);
      }
      previous_sample_id = correlation.sample_id;
      previous_device_tick = correlation.device_tick;
      ++correlation_count;
    }
    EXPECT_GE(correlation_count, 2u);
  }
}

static const iree_hal_profile_command_operation_record_t* FindCommandOperation(
    const CommandBufferProfileSink& sink, uint64_t command_buffer_id,
    uint32_t command_index) {
  for (const auto& operation : sink.command_operations) {
    if (operation.command_buffer_id == command_buffer_id &&
        operation.command_index == command_index) {
      return &operation;
    }
  }
  return nullptr;
}

static const iree_hal_profile_event_relationship_record_t*
FindEventRelationship(const CommandBufferProfileSink& sink,
                      iree_hal_profile_event_relationship_type_t type,
                      iree_hal_profile_event_endpoint_type_t source_type,
                      uint64_t source_id,
                      iree_hal_profile_event_endpoint_type_t target_type,
                      uint64_t target_id) {
  for (const auto& relationship : sink.event_relationships) {
    if (relationship.type == type && relationship.source_type == source_type &&
        relationship.source_id == source_id &&
        relationship.target_type == target_type &&
        relationship.target_id == target_id) {
      return &relationship;
    }
  }
  return nullptr;
}

static const iree_hal_profile_queue_event_t* FindUniqueQueueEvent(
    const CommandBufferProfileSink& sink,
    iree_hal_profile_queue_event_type_t type) {
  const iree_hal_profile_queue_event_t* result = nullptr;
  for (const auto& event : sink.queue_events) {
    if (event.type != type) continue;
    EXPECT_EQ(nullptr, result);
    result = &event;
  }
  return result;
}

static iree_host_size_t CountQueueEvents(
    const CommandBufferProfileSink& sink,
    iree_hal_profile_queue_event_type_t type) {
  iree_host_size_t count = 0;
  for (const auto& event : sink.queue_events) {
    if (event.type == type) ++count;
  }
  return count;
}

static uint32_t SumQueueEventOperationCounts(
    const CommandBufferProfileSink& sink,
    iree_hal_profile_queue_event_type_t type) {
  uint32_t operation_count = 0;
  for (const auto& event : sink.queue_events) {
    if (event.type == type) {
      operation_count += event.operation_count;
    }
  }
  return operation_count;
}

static bool IsHardwareCounterProfilingUnavailable(iree_status_t status) {
  return IsProfilingUnsupported(status) || iree_status_is_not_found(status) ||
         iree_status_is_failed_precondition(status);
}

static bool IsQueueDeviceProfilingUnavailable(iree_status_t status) {
  return IsProfilingUnsupported(status) ||
         iree_status_is_failed_precondition(status);
}

static iree_status_t BeginHardwareCounterProfiling(
    DeviceProfilingScope* profiling, CommandBufferProfileSink* sink,
    iree_host_size_t counter_name_count, iree_string_view_t* counter_names) {
  iree_hal_profile_counter_set_selection_t counter_set = {
      /*.flags=*/IREE_HAL_PROFILE_COUNTER_SET_SELECTION_FLAG_NONE,
      /*.name=*/IREE_SV("smoke"),
      /*.counter_name_count=*/counter_name_count,
      /*.counter_names=*/counter_names,
  };
  iree_hal_device_profiling_options_t profiling_options = {0};
  profiling_options.data_families =
      IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
      IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES;
  profiling_options.sink = CommandBufferProfileSinkAsBase(sink);
  profiling_options.counter_set_count = 1;
  profiling_options.counter_sets = &counter_set;
  return profiling->Begin(&profiling_options);
}

static iree_status_t BeginSqWavesProfiling(DeviceProfilingScope* profiling,
                                           CommandBufferProfileSink* sink) {
  iree_string_view_t counter_names[] = {
      IREE_SV("SQ_WAVES"),
  };
  return BeginHardwareCounterProfiling(
      profiling, sink, IREE_ARRAYSIZE(counter_names), counter_names);
}

static iree_status_t BeginSqWavesCounterRangeProfiling(
    DeviceProfilingScope* profiling, CommandBufferProfileSink* sink) {
  iree_string_view_t counter_names[] = {
      IREE_SV("SQ_WAVES"),
  };
  iree_hal_profile_counter_set_selection_t counter_set = {
      /*.flags=*/IREE_HAL_PROFILE_COUNTER_SET_SELECTION_FLAG_NONE,
      /*.name=*/IREE_SV("smoke"),
      /*.counter_name_count=*/IREE_ARRAYSIZE(counter_names),
      /*.counter_names=*/counter_names,
  };
  iree_hal_device_profiling_options_t profiling_options = {0};
  profiling_options.data_families =
      IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES;
  profiling_options.sink = CommandBufferProfileSinkAsBase(sink);
  profiling_options.counter_set_count = 1;
  profiling_options.counter_sets = &counter_set;
  return profiling->Begin(&profiling_options);
}

static iree_status_t BeginSqWaveWidthProfiling(DeviceProfilingScope* profiling,
                                               CommandBufferProfileSink* sink) {
  iree_string_view_t counter_names[] = {
      IREE_SV("SQ_WAVES"),
      IREE_SV("SQ_WAVES_32"),
      IREE_SV("SQ_WAVES_64"),
      IREE_SV("SQ_BUSY_CYCLES"),
  };
  return BeginHardwareCounterProfiling(
      profiling, sink, IREE_ARRAYSIZE(counter_names), counter_names);
}

}  // namespace iree::hal::amdgpu::test

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PROFILING_TEST_UTIL_H_
