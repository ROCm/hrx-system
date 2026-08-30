// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/task/registration/driver_module.h"
#include "iree/hal/replay/execute.h"
#include "iree/hal/replay/file_reader.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/io/file_contents.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

static iree_hal_device_t* CreateMockDevice(const char* identifier) {
  iree_hal_mock_device_options_t options;
  iree_hal_mock_device_options_initialize(&options);
  options.identifier = iree_make_cstring_view(identifier);
  iree_hal_device_t* device = nullptr;
  IREE_CHECK_OK(
      iree_hal_mock_device_create(&options, iree_allocator_system(), &device));
  return device;
}

static iree_hal_device_group_t* CreateDeviceGroup(iree_host_size_t device_count,
                                                  iree_hal_device_t** devices) {
  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  IREE_CHECK_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &frontier_tracker));

  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder, frontier_tracker);
  iree_async_frontier_tracker_release(frontier_tracker);
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    IREE_CHECK_OK(
        iree_hal_device_group_builder_add_device(&builder, devices[i]));
  }

  iree_hal_device_group_t* group = nullptr;
  IREE_CHECK_OK(iree_hal_device_group_builder_finalize(
      &builder, iree_allocator_system(), &group));
  return group;
}

static iree_hal_device_group_t* CreateMockDeviceGroup() {
  iree_hal_device_t* device_a = CreateMockDevice("mock");
  iree_hal_device_t* device_b = CreateMockDevice("mock");
  iree_hal_device_t* devices[] = {device_a, device_b};

  iree_hal_device_group_t* group = CreateDeviceGroup(2, devices);
  iree_hal_device_release(device_a);
  iree_hal_device_release(device_b);
  return group;
}

static iree_hal_device_t* CreateLocalTaskDevice() {
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_CHECK_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, iree_async_proactor_pool_options_default(),
      iree_allocator_system(), &proactor_pool));

  iree_hal_driver_registry_t* registry = nullptr;
  IREE_CHECK_OK(
      iree_hal_driver_registry_allocate(iree_allocator_system(), &registry));
  IREE_CHECK_OK(iree_hal_task_driver_module_register(registry));

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;

  iree_hal_device_t* device = nullptr;
  iree_status_t status =
      iree_hal_create_device(registry, IREE_SV("local-task"), &create_params,
                             iree_allocator_system(), &device);
  iree_hal_driver_registry_free(registry);
  iree_async_proactor_pool_release(proactor_pool);
  IREE_CHECK_OK(status);
  return device;
}

static iree_hal_device_group_t* CreateLocalTaskDeviceGroup() {
  iree_hal_device_t* device = CreateLocalTaskDevice();
  iree_hal_device_t* devices[] = {device};
  iree_hal_device_group_t* group = CreateDeviceGroup(1, devices);
  iree_hal_device_release(device);
  return group;
}

static iree_hal_replay_recorder_t* CreateHostAllocationRecorder(
    std::vector<uint8_t>* storage,
    const iree_hal_replay_recorder_options_t* options = nullptr) {
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage->data(), storage->size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_recorder_t* recorder = nullptr;
  IREE_CHECK_OK(iree_hal_replay_recorder_create(
      file_handle, options, iree_allocator_system(), &recorder));
  iree_io_file_handle_release(file_handle);
  return recorder;
}

static iree_const_byte_span_t GetCapturedFileContents(
    const std::vector<uint8_t>& storage) {
  iree_hal_replay_file_header_t file_header;
  iree_host_size_t offset = 0;
  IREE_CHECK_OK(iree_hal_replay_file_parse_header(
      iree_make_const_byte_span(storage.data(), storage.size()), &file_header,
      &offset));
  EXPECT_LE(file_header.file_length, storage.size());
  return iree_make_const_byte_span(storage.data(),
                                   (iree_host_size_t)file_header.file_length);
}

#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
iree::testing::TempFilePath WriteTempFile(iree_const_byte_span_t contents) {
  iree::testing::TempFilePath path("iree_hal_replay_recorder_file");
  IREE_EXPECT_OK(iree_io_file_contents_write(path.path_view(), contents,
                                             iree_allocator_system()));
  return path;
}
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)

struct ReplayRecordSummary {
  // Capture-time thread identifier on the session record.
  uint64_t session_thread_id = 0;
  iree_host_size_t session_record_count = 0;
  iree_host_size_t device_object_record_count = 0;
  iree_host_size_t allocator_object_record_count = 0;
  iree_host_size_t buffer_object_record_count = 0;
  iree_host_size_t command_buffer_object_record_count = 0;
  iree_host_size_t semaphore_object_record_count = 0;
  iree_host_size_t file_object_record_count = 0;
  iree_host_size_t external_file_object_count = 0;
  iree_host_size_t inline_file_object_count = 0;
  uint64_t inline_file_reference_length = 0;
  iree_host_size_t assign_topology_record_count = 0;
  iree_host_size_t refine_topology_record_count = 0;
  iree_host_size_t allocate_buffer_record_count = 0;
  iree_host_size_t import_buffer_record_count = 0;
  iree_host_size_t buffer_map_range_record_count = 0;
  iree_host_size_t buffer_flush_range_record_count = 0;
  iree_host_size_t buffer_unmap_range_record_count = 0;
  iree_host_size_t queue_execute_record_count = 0;
  // Replay scope marker records.
  struct {
    // Number of scope begin records.
    iree_host_size_t begin_record_count = 0;
    // Number of scope end records.
    iree_host_size_t end_record_count = 0;
    // Name from the last parsed scope record.
    std::string last_name;
  } scope;
  iree_host_size_t unsupported_import_buffer_record_count = 0;
  iree_host_size_t unsupported_export_buffer_record_count = 0;
  iree_host_size_t unsupported_host_call_record_count = 0;
  iree_host_size_t buffer_range_data_payload_count = 0;
  iree_host_size_t import_buffer_payload_count = 0;
  uint64_t import_buffer_captured_data_length = 0;
  iree_host_size_t device_queue_execute_payload_count = 0;
  iree_host_size_t semaphore_object_payload_count = 0;
  // Memory type from the last parsed buffer object payload.
  iree_hal_memory_type_t last_buffer_memory_type = 0;
};

static ReplayRecordSummary ParseReplayRecordSummary(
    const std::vector<uint8_t>& storage) {
  ReplayRecordSummary summary;

  iree_hal_replay_file_header_t file_header;
  iree_host_size_t offset = 0;
  IREE_CHECK_OK(iree_hal_replay_file_parse_header(
      iree_make_const_byte_span(storage.data(), storage.size()), &file_header,
      &offset));
  EXPECT_LE(file_header.file_length, storage.size());
  iree_const_byte_span_t file_contents = iree_make_const_byte_span(
      storage.data(), (iree_host_size_t)file_header.file_length);

  uint64_t expected_sequence_ordinal = 0;
  while (offset < file_header.file_length) {
    iree_hal_replay_file_record_t record;
    IREE_CHECK_OK(iree_hal_replay_file_parse_record(file_contents, offset,
                                                    &record, &offset));
    EXPECT_EQ(expected_sequence_ordinal++, record.header.sequence_ordinal);
    switch (record.header.record_type) {
      case IREE_HAL_REPLAY_FILE_RECORD_TYPE_SESSION:
        summary.session_thread_id = record.header.thread_id;
        ++summary.session_record_count;
        break;
      case IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT:
        if (record.header.object_type == IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE) {
          ++summary.device_object_record_count;
        } else if (record.header.object_type ==
                   IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR) {
          ++summary.allocator_object_record_count;
        } else if (record.header.object_type ==
                   IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER) {
          ++summary.buffer_object_record_count;
          EXPECT_EQ(IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT,
                    record.header.payload_type);
          iree_hal_replay_buffer_object_payload_t buffer_payload;
          if (record.payload.data_length != sizeof(buffer_payload)) {
            ADD_FAILURE() << "buffer object payload size mismatch";
            return summary;
          }
          memcpy(&buffer_payload, record.payload.data, sizeof(buffer_payload));
          summary.last_buffer_memory_type = buffer_payload.memory_type;
        } else if (record.header.object_type ==
                   IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER) {
          ++summary.command_buffer_object_record_count;
        } else if (record.header.object_type ==
                   IREE_HAL_REPLAY_OBJECT_TYPE_SEMAPHORE) {
          ++summary.semaphore_object_record_count;
        } else if (record.header.object_type ==
                   IREE_HAL_REPLAY_OBJECT_TYPE_FILE) {
          ++summary.file_object_record_count;
          EXPECT_EQ(IREE_HAL_REPLAY_PAYLOAD_TYPE_FILE_OBJECT,
                    record.header.payload_type);
          iree_hal_replay_file_object_payload_t file_payload;
          if (record.payload.data_length < sizeof(file_payload)) {
            ADD_FAILURE() << "file object payload is short";
            return summary;
          }
          memcpy(&file_payload, record.payload.data, sizeof(file_payload));
          if (file_payload.reference_length > IREE_HOST_SIZE_MAX ||
              sizeof(file_payload) +
                      (iree_host_size_t)file_payload.reference_length !=
                  record.payload.data_length) {
            ADD_FAILURE() << "file object reference length mismatch";
            return summary;
          }
          if (file_payload.reference_type ==
              IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_EXTERNAL_PATH) {
            ++summary.external_file_object_count;
          } else if (file_payload.reference_type ==
                     IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_INLINE_BYTES) {
            ++summary.inline_file_object_count;
            summary.inline_file_reference_length +=
                file_payload.reference_length;
          }
        }
        break;
      case IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION:
        if (record.header.operation_code ==
            IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_ASSIGN_TOPOLOGY_INFO) {
          ++summary.assign_topology_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_REFINE_TOPOLOGY_EDGE) {
          ++summary.refine_topology_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_ALLOCATE_BUFFER) {
          ++summary.allocate_buffer_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_IMPORT_BUFFER) {
          ++summary.import_buffer_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_MAP_RANGE) {
          ++summary.buffer_map_range_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_FLUSH_RANGE) {
          ++summary.buffer_flush_range_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_UNMAP_RANGE) {
          ++summary.buffer_unmap_range_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_EXECUTE) {
          ++summary.queue_execute_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_REPLAY_SCOPE_BEGIN) {
          ++summary.scope.begin_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_REPLAY_SCOPE_END) {
          ++summary.scope.end_record_count;
        }
        if (record.header.payload_type ==
            IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE_DATA) {
          ++summary.buffer_range_data_payload_count;
        } else if (record.header.payload_type ==
                   IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_IMPORT_BUFFER) {
          ++summary.import_buffer_payload_count;
          iree_hal_replay_allocator_import_buffer_payload_t import_payload;
          if (record.payload.data_length < sizeof(import_payload)) {
            ADD_FAILURE() << "import buffer payload is short";
            return summary;
          }
          memcpy(&import_payload, record.payload.data, sizeof(import_payload));
          summary.import_buffer_captured_data_length +=
              import_payload.data_length;
        } else if (record.header.payload_type ==
                   IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_EXECUTE) {
          ++summary.device_queue_execute_payload_count;
        } else if (record.header.payload_type ==
                   IREE_HAL_REPLAY_PAYLOAD_TYPE_SEMAPHORE_OBJECT) {
          ++summary.semaphore_object_payload_count;
        } else if (record.header.payload_type ==
                   IREE_HAL_REPLAY_PAYLOAD_TYPE_REPLAY_SCOPE) {
          iree_hal_replay_scope_payload_t scope_payload;
          if (record.payload.data_length < sizeof(scope_payload)) {
            ADD_FAILURE() << "scope payload is short";
            return summary;
          }
          memcpy(&scope_payload, record.payload.data, sizeof(scope_payload));
          if (scope_payload.name_length > IREE_HOST_SIZE_MAX ||
              sizeof(scope_payload) +
                      (iree_host_size_t)scope_payload.name_length !=
                  record.payload.data_length) {
            ADD_FAILURE() << "scope payload name length mismatch";
            return summary;
          }
          summary.scope.last_name.assign(
              (const char*)record.payload.data + sizeof(scope_payload),
              (iree_host_size_t)scope_payload.name_length);
        }
        EXPECT_EQ((uint32_t)IREE_STATUS_OK, record.header.status_code);
        break;
      case IREE_HAL_REPLAY_FILE_RECORD_TYPE_UNSUPPORTED:
        if (record.header.operation_code ==
            IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_IMPORT_BUFFER) {
          ++summary.unsupported_import_buffer_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_EXPORT_BUFFER) {
          ++summary.unsupported_export_buffer_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_HOST_CALL) {
          ++summary.unsupported_host_call_record_count;
        }
        EXPECT_EQ((uint32_t)IREE_STATUS_OK, record.header.status_code);
        break;
      default:
        break;
    }
  }

  return summary;
}

static std::vector<iree_hal_replay_file_record_t> ParseOperationRecords(
    const std::vector<uint8_t>& storage) {
  iree_hal_replay_file_header_t file_header;
  iree_host_size_t offset = 0;
  IREE_CHECK_OK(iree_hal_replay_file_parse_header(
      iree_make_const_byte_span(storage.data(), storage.size()), &file_header,
      &offset));
  iree_const_byte_span_t file_contents = iree_make_const_byte_span(
      storage.data(), (iree_host_size_t)file_header.file_length);

  std::vector<iree_hal_replay_file_record_t> records;
  while (offset < file_header.file_length) {
    iree_hal_replay_file_record_t record;
    IREE_CHECK_OK(iree_hal_replay_file_parse_record(file_contents, offset,
                                                    &record, &offset));
    if (record.header.record_type ==
        IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION) {
      records.push_back(record);
    }
  }
  return records;
}

static const iree_hal_replay_file_record_t* FindOperationRecord(
    const std::vector<iree_hal_replay_file_record_t>& records,
    iree_hal_replay_operation_code_t operation_code) {
  for (const auto& record : records) {
    if (record.header.operation_code == operation_code) return &record;
  }
  return nullptr;
}

static void ReadQueueSemaphoreTail(
    const iree_hal_replay_file_record_t& record,
    iree_host_size_t fixed_payload_size,
    iree_hal_replay_semaphore_timepoint_payload_t* out_wait_payload,
    iree_hal_replay_semaphore_timepoint_payload_t* out_signal_payload) {
  ASSERT_EQ(fixed_payload_size +
                2 * sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
            record.payload.data_length);
  memcpy(out_wait_payload, record.payload.data + fixed_payload_size,
         sizeof(*out_wait_payload));
  memcpy(out_signal_payload,
         record.payload.data + fixed_payload_size + sizeof(*out_wait_payload),
         sizeof(*out_signal_payload));
}

static iree_status_t CountHostCall(void* user_data, const uint64_t args[4],
                                   iree_hal_host_call_context_t* context) {
  (void)args;
  (void)context;
  ++*(int*)user_data;
  return iree_ok_status();
}

TEST(ReplayRecorderTest, RecordsAvailableThreadIdentifier) {
  std::vector<uint8_t> storage(4096, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);

  ReplayRecordSummary summary = ParseReplayRecordSummary(storage);
  EXPECT_EQ(1u, summary.session_record_count);
#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE
  EXPECT_EQ(0u, summary.session_thread_id);
#elif defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_APPLE) || \
    defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_WINDOWS)
  EXPECT_NE(0u, summary.session_thread_id);
#else
  EXPECT_EQ(0u, summary.session_thread_id);
#endif  // native thread identifiers available
}

TEST(ReplayRecorderTest, RecordsNamedScopes) {
  std::vector<uint8_t> storage(4096, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  IREE_ASSERT_OK(iree_hal_replay_recorder_scope_begin(
      recorder, iree_make_cstring_view("execute")));
  IREE_ASSERT_OK(iree_hal_replay_recorder_scope_end(
      recorder, iree_make_cstring_view("execute")));
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);

  ReplayRecordSummary summary = ParseReplayRecordSummary(storage);
  EXPECT_EQ(1u, summary.session_record_count);
  EXPECT_EQ(1u, summary.scope.begin_record_count);
  EXPECT_EQ(1u, summary.scope.end_record_count);
  EXPECT_EQ("execute", summary.scope.last_name);
}

TEST(ReplayRecorderTest, WrapDeviceGroupRecordsOrderedDeviceOperations) {
  std::vector<uint8_t> storage(16384, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateMockDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_topology_edge_t edge = {};
  IREE_ASSERT_OK(iree_hal_device_refine_topology_edge(
      wrapped_device, iree_hal_device_group_device_at(wrapped_group, 1),
      &edge));

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  ReplayRecordSummary summary = ParseReplayRecordSummary(storage);
  EXPECT_EQ(1u, summary.session_record_count);
  EXPECT_EQ(2u, summary.device_object_record_count);
  EXPECT_EQ(2u, summary.assign_topology_record_count);
  EXPECT_EQ(1u, summary.refine_topology_record_count);
}

TEST(ReplayRecorderTest, WrappedDeviceRecordsHostCallAsUnsupported) {
  std::vector<uint8_t> storage(16384, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateLocalTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  int call_count = 0;
  iree_hal_host_call_t call =
      iree_hal_make_host_call(CountHostCall, &call_count);
  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_semaphore_list = {
      /*.count=*/1,
      /*.semaphores=*/&signal_semaphore,
      /*.payload_values=*/&signal_value,
  };
  const uint64_t args[4] = {0, 1, 2, 3};
  IREE_ASSERT_OK(iree_hal_device_queue_host_call(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal_semaphore_list, call, args,
      IREE_HAL_HOST_CALL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(signal_semaphore, signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  EXPECT_EQ(1, call_count);
  iree_hal_semaphore_release(signal_semaphore);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  ReplayRecordSummary summary = ParseReplayRecordSummary(storage);
  EXPECT_EQ(1u, summary.unsupported_host_call_record_count);
}

TEST(ReplayRecorderTest,
     WrappedAllocatorSnapshotsHostAllocationImportsAndMarksExportsUnsupported) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateLocalTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_MAPPING | IREE_HAL_BUFFER_USAGE_SHARING_EXPORT;

  alignas(64) uint8_t imported_storage[16] = {0};
  iree_hal_external_buffer_t external_buffer = {};
  external_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION;
  external_buffer.size = sizeof(imported_storage);
  external_buffer.handle.host_allocation.ptr = imported_storage;
  iree_hal_buffer_t* imported_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_import_buffer(
      allocator, params, &external_buffer,
      iree_hal_buffer_release_callback_null(), &imported_buffer));

  iree_hal_buffer_t* allocated_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, params, sizeof(imported_storage), &allocated_buffer));
  iree_hal_external_buffer_t exported_buffer = {};
  IREE_ASSERT_OK(iree_hal_allocator_export_buffer(
      allocator, allocated_buffer,
      IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &exported_buffer));
  EXPECT_EQ(IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
            exported_buffer.type);

  iree_hal_buffer_release(allocated_buffer);
  iree_hal_buffer_release(imported_buffer);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  ReplayRecordSummary summary = ParseReplayRecordSummary(storage);
  EXPECT_EQ(1u, summary.import_buffer_record_count);
  EXPECT_EQ(1u, summary.import_buffer_payload_count);
  EXPECT_EQ(sizeof(imported_storage),
            summary.import_buffer_captured_data_length);
  EXPECT_EQ(0u, summary.unsupported_import_buffer_record_count);
  EXPECT_EQ(1u, summary.unsupported_export_buffer_record_count);
  EXPECT_EQ(2u, summary.buffer_object_record_count);
}

TEST(ReplayRecorderTest, WrappedAllocatorRecordsBuffersAndMapping) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateLocalTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, 16, &buffer));

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                           0, 16, &mapping));
  iree_byte_span_t span;
  IREE_ASSERT_OK(iree_hal_buffer_mapping_subspan(
      &mapping, IREE_HAL_MEMORY_ACCESS_WRITE, 0, 16, &span));
  const uint8_t contents[16] = {
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
      0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
  };
  std::memcpy(span.data, contents, sizeof(contents));
  IREE_ASSERT_OK(iree_hal_buffer_mapping_flush_range(&mapping, 0, 16));
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  iree_hal_buffer_release(buffer);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  ReplayRecordSummary summary = ParseReplayRecordSummary(storage);
  EXPECT_EQ(1u, summary.session_record_count);
  EXPECT_EQ(1u, summary.device_object_record_count);
  EXPECT_EQ(1u, summary.allocator_object_record_count);
  EXPECT_EQ(1u, summary.buffer_object_record_count);
  EXPECT_EQ(1u, summary.allocate_buffer_record_count);
  EXPECT_EQ(1u, summary.buffer_map_range_record_count);
  EXPECT_EQ(1u, summary.buffer_flush_range_record_count);
  EXPECT_EQ(1u, summary.buffer_unmap_range_record_count);
  EXPECT_EQ(2u, summary.buffer_range_data_payload_count);
}

TEST(ReplayRecorderTest, PersistentWriteMapsFailLoud) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateLocalTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  ASSERT_NE(nullptr, allocator);

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_MAPPING | IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, 16, &buffer));

  iree_hal_buffer_mapping_t mapping;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
                                IREE_HAL_MEMORY_ACCESS_WRITE, 0, 16, &mapping));
  iree_hal_buffer_release(buffer);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);
}

TEST(ReplayRecorderTest, ExternalFileFailPolicyRejectsFdBackedFiles) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  const uint8_t file_contents[4] = {0x00, 0x01, 0x02, 0x03};
  iree::testing::TempFilePath source_file = WriteTempFile(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));

  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_options_t options =
      iree_hal_replay_recorder_options_default();
  options.external_file_policy =
      IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_POLICY_FAIL;
  iree_hal_replay_recorder_t* recorder =
      CreateHostAllocationRecorder(&storage, &options);

  iree_hal_device_group_t* source_group = CreateLocalTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));
  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);

  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS,
      source_file.path_view(), iree_allocator_system(), &file_handle));
  iree_hal_file_t* file = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_file_import(wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY,
                           IREE_HAL_MEMORY_ACCESS_READ, file_handle,
                           IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
  EXPECT_EQ(nullptr, file);
  iree_io_file_handle_release(file_handle);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);
#else
  GTEST_SKIP() << "FD-backed replay requires POSIX file IO.";
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
}

TEST(ReplayRecorderTest, ExternalFileCaptureAllPolicyEmbedsFdBackedFiles) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  const uint8_t file_contents[4] = {0x00, 0x01, 0x02, 0x03};
  iree::testing::TempFilePath source_file = WriteTempFile(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));

  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_options_t options =
      iree_hal_replay_recorder_options_default();
  options.external_file_policy =
      IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_POLICY_CAPTURE_ALL;
  iree_hal_replay_recorder_t* recorder =
      CreateHostAllocationRecorder(&storage, &options);

  iree_hal_device_group_t* source_group = CreateLocalTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));
  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);

  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS,
      source_file.path_view(), iree_allocator_system(), &file_handle));
  iree_hal_file_t* file = nullptr;
  IREE_ASSERT_OK(iree_hal_file_import(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, IREE_HAL_MEMORY_ACCESS_READ,
      file_handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
  iree_io_file_handle_release(file_handle);
  iree_hal_file_release(file);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  ReplayRecordSummary summary = ParseReplayRecordSummary(storage);
  EXPECT_EQ(1u, summary.file_object_record_count);
  EXPECT_EQ(0u, summary.external_file_object_count);
  EXPECT_EQ(1u, summary.inline_file_object_count);
  EXPECT_EQ(sizeof(file_contents), summary.inline_file_reference_length);
#else
  GTEST_SKIP() << "FD-backed replay requires POSIX file IO.";
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
}

TEST(ReplayRecorderTest, WrappedDeviceRecordsQueueExecuteSemaphores) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateLocalTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      wrapped_device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_semaphore_t* wait_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &wait_semaphore));
  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));

  iree_hal_semaphore_t* wait_semaphores[] = {wait_semaphore};
  uint64_t wait_values[] = {0};
  iree_hal_semaphore_list_t wait_list = {
      IREE_ARRAYSIZE(wait_semaphores),
      wait_semaphores,
      wait_values,
  };
  iree_hal_semaphore_t* signal_semaphores[] = {signal_semaphore};
  uint64_t signal_values[] = {1};
  iree_hal_semaphore_list_t signal_list = {
      IREE_ARRAYSIZE(signal_semaphores),
      signal_semaphores,
      signal_values,
  };
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list,
      command_buffer, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(
      iree_hal_device_queue_flush(wrapped_device, IREE_HAL_QUEUE_AFFINITY_ANY));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(signal_semaphore);
  iree_hal_semaphore_release(wait_semaphore);
  iree_hal_command_buffer_release(command_buffer);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  ReplayRecordSummary summary = ParseReplayRecordSummary(storage);
  EXPECT_EQ(1u, summary.command_buffer_object_record_count);
  EXPECT_EQ(2u, summary.semaphore_object_record_count);
  EXPECT_EQ(2u, summary.semaphore_object_payload_count);
  EXPECT_EQ(1u, summary.queue_execute_record_count);
  EXPECT_EQ(1u, summary.device_queue_execute_payload_count);
}

TEST(ReplayRecorderTest, RecordsAndReplaysCommandBufferAtomicOperations) {
  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateLocalTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));
  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);

  const iree_hal_buffer_params_t buffer_params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_STORAGE,
      /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
      /*.type=*/IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
          IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(wrapped_device), buffer_params, 64, &buffer));

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      wrapped_device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_ATOMIC, /*queue_affinity=*/1,
      /*binding_capacity=*/1, &command_buffer));
  const iree_hal_atomic_wait_params_t wait_params = {
      /*.value=*/UINT64_C(0x1020304050607080),
      /*.mask=*/UINT64_C(0xFFEEDDCCBBAA9988),
      /*.flags=*/IREE_HAL_ATOMIC_FLAGS_KNOWN,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_64,
      /*.condition=*/IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL,
  };
  const iree_hal_atomic_store_params_t store_params = {
      /*.value=*/UINT64_C(0xAABBCCDD),
      /*.flags=*/IREE_HAL_ATOMIC_FLAGS_KNOWN,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
  };
  const iree_hal_atomic_rmw_params_t rmw_params = {
      /*.operand=*/UINT64_C(0x11223344),
      /*.flags=*/IREE_HAL_ATOMIC_FLAGS_KNOWN,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
      /*.operation=*/IREE_HAL_ATOMIC_RMW_OPERATION_XOR,
  };

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_atomic_wait(
      command_buffer, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_STAGE_ATOMIC,
      iree_hal_make_buffer_ref(buffer, /*offset=*/8, /*length=*/8),
      wait_params));
  IREE_ASSERT_OK(iree_hal_command_buffer_atomic_store(
      command_buffer, IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/0, /*offset=*/12,
                                        /*length=*/4),
      store_params));
  IREE_ASSERT_OK(iree_hal_command_buffer_atomic_rmw(
      command_buffer, IREE_HAL_EXECUTION_STAGE_HOST,
      IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS,
      iree_hal_make_buffer_ref(buffer, /*offset=*/32, /*length=*/4),
      rmw_params));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(buffer);
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  const ReplayRecordSummary summary = ParseReplayRecordSummary(storage);
  EXPECT_EQ(
      summary.last_buffer_memory_type,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
          IREE_HAL_MEMORY_TYPE_HOST_CACHED | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);

  const auto records = ParseOperationRecords(storage);
  const auto* wait_record = FindOperationRecord(
      records, IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_WAIT);
  ASSERT_NE(nullptr, wait_record);
  EXPECT_EQ(IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_WAIT,
            wait_record->header.payload_type);
  EXPECT_EQ(IREE_STATUS_OK, wait_record->header.status_code);
  ASSERT_EQ(sizeof(iree_hal_replay_command_buffer_atomic_wait_payload_t),
            wait_record->payload.data_length);
  iree_hal_replay_command_buffer_atomic_wait_payload_t wait_payload;
  memcpy(&wait_payload, wait_record->payload.data, sizeof(wait_payload));
  EXPECT_NE(IREE_HAL_REPLAY_OBJECT_ID_NONE, wait_payload.target_ref.buffer_id);
  EXPECT_EQ(8u, wait_payload.target_ref.offset);
  EXPECT_EQ(8u, wait_payload.target_ref.length);
  EXPECT_EQ(IREE_HAL_EXECUTION_STAGE_DISPATCH, wait_payload.source_stage_mask);
  EXPECT_EQ(IREE_HAL_EXECUTION_STAGE_ATOMIC, wait_payload.target_stage_mask);
  EXPECT_EQ(wait_params.value, wait_payload.params.value);
  EXPECT_EQ(wait_params.mask, wait_payload.params.mask);
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAGS_KNOWN, wait_payload.params.flags);
  EXPECT_EQ(wait_params.width, wait_payload.params.width);
  EXPECT_EQ(wait_params.condition, wait_payload.params.condition);
  EXPECT_EQ(0u, wait_payload.params.reserved0);

  const auto* store_record = FindOperationRecord(
      records, IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_STORE);
  ASSERT_NE(nullptr, store_record);
  EXPECT_EQ(IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_STORE,
            store_record->header.payload_type);
  EXPECT_EQ(IREE_STATUS_OK, store_record->header.status_code);
  ASSERT_EQ(sizeof(iree_hal_replay_command_buffer_atomic_store_payload_t),
            store_record->payload.data_length);
  iree_hal_replay_command_buffer_atomic_store_payload_t store_payload;
  memcpy(&store_payload, store_record->payload.data, sizeof(store_payload));
  EXPECT_EQ(IREE_HAL_REPLAY_OBJECT_ID_NONE, store_payload.target_ref.buffer_id);
  EXPECT_EQ(0u, store_payload.target_ref.buffer_slot);
  EXPECT_EQ(12u, store_payload.target_ref.offset);
  EXPECT_EQ(4u, store_payload.target_ref.length);
  EXPECT_EQ(IREE_HAL_EXECUTION_STAGE_TRANSFER, store_payload.source_stage_mask);
  EXPECT_EQ(IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
            store_payload.target_stage_mask);
  EXPECT_EQ(store_params.value, store_payload.params.value);
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAGS_KNOWN, store_payload.params.flags);
  EXPECT_EQ(store_params.width, store_payload.params.width);
  EXPECT_EQ(0, memcmp(store_payload.params.reserved0, store_params.reserved,
                      sizeof(store_params.reserved)));

  const auto* rmw_record = FindOperationRecord(
      records, IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_RMW);
  ASSERT_NE(nullptr, rmw_record);
  EXPECT_EQ(IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_RMW,
            rmw_record->header.payload_type);
  EXPECT_EQ(IREE_STATUS_OK, rmw_record->header.status_code);
  ASSERT_EQ(sizeof(iree_hal_replay_command_buffer_atomic_rmw_payload_t),
            rmw_record->payload.data_length);
  iree_hal_replay_command_buffer_atomic_rmw_payload_t rmw_payload;
  memcpy(&rmw_payload, rmw_record->payload.data, sizeof(rmw_payload));
  EXPECT_EQ(wait_payload.target_ref.buffer_id,
            rmw_payload.target_ref.buffer_id);
  EXPECT_EQ(32u, rmw_payload.target_ref.offset);
  EXPECT_EQ(4u, rmw_payload.target_ref.length);
  EXPECT_EQ(IREE_HAL_EXECUTION_STAGE_HOST, rmw_payload.source_stage_mask);
  EXPECT_EQ(IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS,
            rmw_payload.target_stage_mask);
  EXPECT_EQ(rmw_params.operand, rmw_payload.params.operand);
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAGS_KNOWN, rmw_payload.params.flags);
  EXPECT_EQ(rmw_params.width, rmw_payload.params.width);
  EXPECT_EQ(rmw_params.operation, rmw_payload.params.operation);
  EXPECT_EQ(0u, rmw_payload.params.reserved0);

  iree_hal_device_group_t* replay_group = CreateLocalTaskDeviceGroup();
  IREE_ASSERT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, /*options=*/nullptr,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
}

TEST(ReplayRecorderTest, RecordsDeviceQueueAtomicOperations) {
  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateLocalTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));
  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);

  const iree_hal_buffer_params_t buffer_params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_STORAGE |
          IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED,
      /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
      /*.type=*/IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
  };
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(wrapped_device), buffer_params, 64, &buffer));
  const uint64_t initial_wait_value = UINT64_MAX;
  IREE_ASSERT_OK(iree_hal_buffer_map_write(buffer, /*target_offset=*/8,
                                           &initial_wait_value,
                                           sizeof(initial_wait_value)));

  iree_hal_semaphore_t* wait_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, /*queue_affinity=*/1, /*initial_value=*/7,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &wait_semaphore));
  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, /*queue_affinity=*/1, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
  iree_hal_semaphore_t* wait_semaphores[] = {wait_semaphore};
  uint64_t wait_values[] = {7};
  const iree_hal_semaphore_list_t wait_list = {
      IREE_ARRAYSIZE(wait_semaphores),
      wait_semaphores,
      wait_values,
  };
  iree_hal_semaphore_t* signal_semaphores[] = {signal_semaphore};
  uint64_t wait_signal_values[] = {9};
  const iree_hal_semaphore_list_t wait_signal_list = {
      IREE_ARRAYSIZE(signal_semaphores),
      signal_semaphores,
      wait_signal_values,
  };
  uint64_t store_signal_values[] = {10};
  const iree_hal_semaphore_list_t store_signal_list = {
      IREE_ARRAYSIZE(signal_semaphores),
      signal_semaphores,
      store_signal_values,
  };
  uint64_t rmw_signal_values[] = {11};
  const iree_hal_semaphore_list_t rmw_signal_list = {
      IREE_ARRAYSIZE(signal_semaphores),
      signal_semaphores,
      rmw_signal_values,
  };

  const iree_hal_atomic_wait_params_t wait_params = {
      /*.value=*/UINT64_C(0x1020304050607080),
      /*.mask=*/UINT64_C(0xFFEEDDCCBBAA9988),
      /*.flags=*/IREE_HAL_ATOMIC_FLAGS_KNOWN,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_64,
      /*.condition=*/IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL,
  };
  const iree_hal_atomic_store_params_t store_params = {
      /*.value=*/UINT64_C(0xAABBCCDD),
      /*.flags=*/IREE_HAL_ATOMIC_FLAGS_KNOWN,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
  };
  const iree_hal_atomic_rmw_params_t rmw_params = {
      /*.operand=*/UINT64_C(0x1122334455667788),
      /*.flags=*/IREE_HAL_ATOMIC_FLAGS_KNOWN,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_64,
      /*.operation=*/IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
  };

  IREE_ASSERT_OK(iree_hal_device_queue_atomic_wait(
      wrapped_device, /*queue_affinity=*/1, wait_list, wait_signal_list, buffer,
      /*target_offset=*/8, wait_params));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      wait_signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_device_queue_atomic_store(
      wrapped_device, /*queue_affinity=*/1, wait_list, store_signal_list,
      buffer, /*target_offset=*/16, store_params));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      store_signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_device_queue_atomic_rmw(
      wrapped_device, /*queue_affinity=*/1, wait_list, rmw_signal_list, buffer,
      /*target_offset=*/24, rmw_params));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      rmw_signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(signal_semaphore);
  iree_hal_semaphore_release(wait_semaphore);
  iree_hal_buffer_release(buffer);
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  const auto records = ParseOperationRecords(storage);
  const auto* wait_record = FindOperationRecord(
      records, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_WAIT);
  ASSERT_NE(nullptr, wait_record);
  EXPECT_EQ(IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_WAIT,
            wait_record->header.payload_type);
  EXPECT_EQ((uint32_t)IREE_STATUS_OK, wait_record->header.status_code);
  iree_hal_replay_device_queue_atomic_wait_payload_t wait_payload;
  ASSERT_GE(wait_record->payload.data_length, sizeof(wait_payload));
  memcpy(&wait_payload, wait_record->payload.data, sizeof(wait_payload));
  EXPECT_NE(IREE_HAL_REPLAY_OBJECT_ID_NONE, wait_payload.target_ref.buffer_id);
  EXPECT_EQ(wait_payload.target_ref.buffer_id,
            wait_record->header.related_object_id);
  EXPECT_EQ(8u, wait_payload.target_ref.offset);
  EXPECT_EQ(8u, wait_payload.target_ref.length);
  EXPECT_EQ(1u, wait_payload.queue_affinity);
  EXPECT_EQ(1u, wait_payload.wait_semaphore_count);
  EXPECT_EQ(1u, wait_payload.signal_semaphore_count);
  EXPECT_EQ(wait_params.value, wait_payload.params.value);
  EXPECT_EQ(wait_params.mask, wait_payload.params.mask);
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAGS_KNOWN, wait_payload.params.flags);
  EXPECT_EQ(wait_params.width, wait_payload.params.width);
  EXPECT_EQ(wait_params.condition, wait_payload.params.condition);
  EXPECT_EQ(0u, wait_payload.params.reserved0);
  iree_hal_replay_semaphore_timepoint_payload_t wait_timepoint = {};
  iree_hal_replay_semaphore_timepoint_payload_t signal_timepoint = {};
  ReadQueueSemaphoreTail(*wait_record, sizeof(wait_payload), &wait_timepoint,
                         &signal_timepoint);
  EXPECT_NE(IREE_HAL_REPLAY_OBJECT_ID_NONE, wait_timepoint.semaphore_id);
  EXPECT_NE(IREE_HAL_REPLAY_OBJECT_ID_NONE, signal_timepoint.semaphore_id);
  EXPECT_NE(wait_timepoint.semaphore_id, signal_timepoint.semaphore_id);
  EXPECT_EQ(7u, wait_timepoint.value);
  EXPECT_EQ(9u, signal_timepoint.value);

  const auto* store_record = FindOperationRecord(
      records, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_STORE);
  ASSERT_NE(nullptr, store_record);
  EXPECT_EQ(IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_STORE,
            store_record->header.payload_type);
  EXPECT_EQ((uint32_t)IREE_STATUS_OK, store_record->header.status_code);
  iree_hal_replay_device_queue_atomic_store_payload_t store_payload;
  ASSERT_GE(store_record->payload.data_length, sizeof(store_payload));
  memcpy(&store_payload, store_record->payload.data, sizeof(store_payload));
  EXPECT_EQ(wait_payload.target_ref.buffer_id,
            store_payload.target_ref.buffer_id);
  EXPECT_EQ(store_payload.target_ref.buffer_id,
            store_record->header.related_object_id);
  EXPECT_EQ(16u, store_payload.target_ref.offset);
  EXPECT_EQ(4u, store_payload.target_ref.length);
  EXPECT_EQ(1u, store_payload.queue_affinity);
  EXPECT_EQ(1u, store_payload.wait_semaphore_count);
  EXPECT_EQ(1u, store_payload.signal_semaphore_count);
  EXPECT_EQ(store_params.value, store_payload.params.value);
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAGS_KNOWN, store_payload.params.flags);
  EXPECT_EQ(store_params.width, store_payload.params.width);
  EXPECT_EQ(0, memcmp(store_payload.params.reserved0, store_params.reserved,
                      sizeof(store_params.reserved)));
  iree_hal_replay_semaphore_timepoint_payload_t store_wait_timepoint = {};
  iree_hal_replay_semaphore_timepoint_payload_t store_signal_timepoint = {};
  ReadQueueSemaphoreTail(*store_record, sizeof(store_payload),
                         &store_wait_timepoint, &store_signal_timepoint);
  EXPECT_EQ(wait_timepoint.semaphore_id, store_wait_timepoint.semaphore_id);
  EXPECT_EQ(signal_timepoint.semaphore_id, store_signal_timepoint.semaphore_id);
  EXPECT_EQ(7u, store_wait_timepoint.value);
  EXPECT_EQ(10u, store_signal_timepoint.value);

  const auto* rmw_record = FindOperationRecord(
      records, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_RMW);
  ASSERT_NE(nullptr, rmw_record);
  EXPECT_EQ(IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_RMW,
            rmw_record->header.payload_type);
  EXPECT_EQ((uint32_t)IREE_STATUS_OK, rmw_record->header.status_code);
  iree_hal_replay_device_queue_atomic_rmw_payload_t rmw_payload;
  ASSERT_GE(rmw_record->payload.data_length, sizeof(rmw_payload));
  memcpy(&rmw_payload, rmw_record->payload.data, sizeof(rmw_payload));
  EXPECT_EQ(wait_payload.target_ref.buffer_id,
            rmw_payload.target_ref.buffer_id);
  EXPECT_EQ(rmw_payload.target_ref.buffer_id,
            rmw_record->header.related_object_id);
  EXPECT_EQ(24u, rmw_payload.target_ref.offset);
  EXPECT_EQ(8u, rmw_payload.target_ref.length);
  EXPECT_EQ(1u, rmw_payload.queue_affinity);
  EXPECT_EQ(1u, rmw_payload.wait_semaphore_count);
  EXPECT_EQ(1u, rmw_payload.signal_semaphore_count);
  EXPECT_EQ(rmw_params.operand, rmw_payload.params.operand);
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAGS_KNOWN, rmw_payload.params.flags);
  EXPECT_EQ(rmw_params.width, rmw_payload.params.width);
  EXPECT_EQ(rmw_params.operation, rmw_payload.params.operation);
  EXPECT_EQ(0u, rmw_payload.params.reserved0);
  iree_hal_replay_semaphore_timepoint_payload_t rmw_wait_timepoint = {};
  iree_hal_replay_semaphore_timepoint_payload_t rmw_signal_timepoint = {};
  ReadQueueSemaphoreTail(*rmw_record, sizeof(rmw_payload), &rmw_wait_timepoint,
                         &rmw_signal_timepoint);
  EXPECT_EQ(wait_timepoint.semaphore_id, rmw_wait_timepoint.semaphore_id);
  EXPECT_EQ(signal_timepoint.semaphore_id, rmw_signal_timepoint.semaphore_id);
  EXPECT_EQ(7u, rmw_wait_timepoint.value);
  EXPECT_EQ(11u, rmw_signal_timepoint.value);
}

}  // namespace
