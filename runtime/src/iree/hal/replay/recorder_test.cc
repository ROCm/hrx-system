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
#include "iree/hal/replay/recorder_allocator.h"
#include "iree/hal/replay/recorder_record.h"
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

static iree_hal_device_t* CreateTaskDevice() {
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
      iree_hal_create_device(registry, IREE_SV("task"), &create_params,
                             iree_allocator_system(), &device);
  iree_hal_driver_registry_free(registry);
  iree_async_proactor_pool_release(proactor_pool);
  IREE_CHECK_OK(status);
  return device;
}

static iree_hal_device_group_t* CreateTaskDeviceGroup() {
  iree_hal_device_t* device = CreateTaskDevice();
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
  // Queue objects and exact-queue operation records.
  struct {
    // Session-local object identifiers assigned to provisioned queues.
    std::vector<iree_hal_replay_object_id_t> provisioned_object_ids;
    // Number of successful exact-queue execute operation records.
    iree_host_size_t execute_record_count = 0;
    // Number of exact-queue execute records carrying replayable payloads.
    iree_host_size_t execute_payload_count = 0;
    // Queue object identifier targeted by the last exact-queue execute.
    iree_hal_replay_object_id_t last_execute_object_id = 0;
    // Number of successful exact-transfer operation records.
    iree_host_size_t transfer_record_count = 0;
    // Number of exact-transfer records carrying replayable payloads.
    iree_host_size_t transfer_payload_count = 0;
    // Queue object identifier targeted by the last exact transfer.
    iree_hal_replay_object_id_t last_transfer_object_id = 0;
    // Number of sibling operations in the last exact transfer.
    uint64_t last_transfer_operation_count = 0;
    // Number of successful exact transfers marked unsupported.
    iree_host_size_t unsupported_transfer_record_count = 0;
  } queue;
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
        } else if (record.header.object_type ==
                   IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE) {
          if (record.header.payload_type ==
              IREE_HAL_REPLAY_PAYLOAD_TYPE_PROVISIONED_QUEUE_OBJECT) {
            summary.queue.provisioned_object_ids.push_back(
                record.header.object_id);
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
                   IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_EXECUTE) {
          ++summary.queue.execute_record_count;
          summary.queue.last_execute_object_id = record.header.object_id;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_TRANSFER) {
          ++summary.queue.transfer_record_count;
          summary.queue.last_transfer_object_id = record.header.object_id;
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
                   IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_EXECUTE) {
          ++summary.queue.execute_payload_count;
        } else if (record.header.payload_type ==
                   IREE_HAL_REPLAY_PAYLOAD_TYPE_SEMAPHORE_OBJECT) {
          ++summary.semaphore_object_payload_count;
        } else if (record.header.payload_type ==
                   IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_TRANSFER) {
          ++summary.queue.transfer_payload_count;
          iree_hal_replay_queue_transfer_payload_t transfer_payload;
          if (record.payload.data_length < sizeof(transfer_payload)) {
            ADD_FAILURE() << "queue transfer payload is short";
            return summary;
          }
          memcpy(&transfer_payload, record.payload.data,
                 sizeof(transfer_payload));
          summary.queue.last_transfer_operation_count =
              transfer_payload.operation_count;
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
                   IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_EXPORT) {
          ++summary.unsupported_export_buffer_record_count;
          EXPECT_EQ(IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
                    record.header.object_type);
          EXPECT_NE(IREE_HAL_REPLAY_OBJECT_ID_NONE, record.header.object_id);
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_HOST_CALL) {
          ++summary.unsupported_host_call_record_count;
        } else if (record.header.operation_code ==
                   IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_TRANSFER) {
          ++summary.queue.unsupported_transfer_record_count;
          EXPECT_EQ(IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE,
                    record.header.payload_type);
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
    if (record.header.operation_code == operation_code) {
      return &record;
    }
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

TEST(ReplayRecorderTest, RecordingFailurePreservesOperationStatus) {
  std::vector<uint8_t> storage(1024, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_replay_pending_record_t pending_record = {};
  IREE_ASSERT_OK(iree_hal_replay_recorder_begin_operation(
      recorder, IREE_HAL_REPLAY_OBJECT_ID_NONE, IREE_HAL_REPLAY_OBJECT_ID_NONE,
      IREE_HAL_REPLAY_OBJECT_ID_NONE, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
      IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_BARRIER,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_BARRIER, &pending_record));
  std::vector<uint8_t> oversized_payload(storage.size() * 2, 0);
  const iree_const_byte_span_t payload = iree_make_const_byte_span(
      oversized_payload.data(), oversized_payload.size());
  IREE_EXPECT_OK(iree_hal_replay_recorder_end_operation_with_payload(
      &pending_record, iree_ok_status(), /*iovec_count=*/1, &payload));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
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

TEST(ReplayRecorderTest, RecordsExactQueueTransferTransaction) {
  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_queue_t* queue = iree_hal_device_queue(
      wrapped_device, /*family_ordinal=*/0, /*queue_ordinal=*/0);
  ASSERT_NE(nullptr, queue);

  const iree_hal_buffer_params_t buffer_params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER |
          IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED,
      /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
      /*.type=*/IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
  };
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(wrapped_device);
  iree_hal_buffer_t* source_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, buffer_params, /*allocation_size=*/8, &source_buffer));
  iree_hal_buffer_t* target_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, buffer_params, /*allocation_size=*/16, &target_buffer));

  const uint8_t source_data[] = {0x10, 0x11, 0x12, 0x13,
                                 0x14, 0x15, 0x16, 0x17};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(source_buffer, /*target_offset=*/0,
                                           source_data, sizeof(source_data)));

  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, iree_hal_make_queue_family_affinity(0),
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&signal_semaphore,
      /*.payload_values=*/&signal_value,
  };

  const uint8_t fill_pattern = 0xA5;
  const uint8_t update_data[] = {0xE0, 0x21, 0x22, 0x23, 0x24, 0xE5};
  const uint8_t upload_data[] = {0x31, 0x32, 0x33, 0x34};
  uint8_t download_data[4] = {};
  iree_hal_transfer_operation_t operations[5] = {};
  operations[0].type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
  operations[0].fill.target_buffer = target_buffer;
  operations[0].fill.target_offset = 0;
  operations[0].fill.length = 4;
  operations[0].fill.pattern = &fill_pattern;
  operations[0].fill.pattern_length = sizeof(fill_pattern);
  operations[1].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE;
  operations[1].update.source_buffer = update_data;
  operations[1].update.source_offset = 1;
  operations[1].update.target_buffer = target_buffer;
  operations[1].update.target_offset = 4;
  operations[1].update.length = 4;
  operations[2].type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY;
  operations[2].copy.source_buffer = source_buffer;
  operations[2].copy.source_offset = 0;
  operations[2].copy.target_buffer = target_buffer;
  operations[2].copy.target_offset = 8;
  operations[2].copy.length = 4;
  operations[3].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
  operations[3].upload.source = upload_data;
  operations[3].upload.target_buffer = target_buffer;
  operations[3].upload.target_offset = 12;
  operations[3].upload.length = 4;
  operations[4].type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
  operations[4].download.source_buffer = source_buffer;
  operations[4].download.source_offset = 4;
  operations[4].download.target = download_data;
  operations[4].download.length = sizeof(download_data);
  IREE_ASSERT_OK(iree_hal_queue_transfer(
      queue, iree_hal_semaphore_list_empty(), signal_list,
      IREE_ARRAYSIZE(operations), operations));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  const uint8_t expected_target[] = {
      0xA5, 0xA5, 0xA5, 0xA5, 0x21, 0x22, 0x23, 0x24,
      0x10, 0x11, 0x12, 0x13, 0x31, 0x32, 0x33, 0x34,
  };
  uint8_t actual_target[sizeof(expected_target)] = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(target_buffer, /*source_offset=*/0,
                                          actual_target,
                                          sizeof(actual_target)));
  EXPECT_EQ(0, memcmp(expected_target, actual_target, sizeof(expected_target)));
  EXPECT_EQ(0, memcmp(source_data + 4, download_data, sizeof(download_data)));

  iree_hal_semaphore_release(signal_semaphore);
  iree_hal_buffer_release(target_buffer);
  iree_hal_buffer_release(source_buffer);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  const ReplayRecordSummary summary = ParseReplayRecordSummary(storage);
  EXPECT_FALSE(summary.queue.provisioned_object_ids.empty());
  EXPECT_EQ(1u, summary.queue.transfer_record_count);
  EXPECT_EQ(1u, summary.queue.transfer_payload_count);
  EXPECT_EQ(IREE_ARRAYSIZE(operations),
            summary.queue.last_transfer_operation_count);
  bool transfer_targets_provisioned_queue = false;
  for (iree_hal_replay_object_id_t queue_id :
       summary.queue.provisioned_object_ids) {
    transfer_targets_provisioned_queue |=
        queue_id == summary.queue.last_transfer_object_id;
  }
  EXPECT_TRUE(transfer_targets_provisioned_queue);
}

TEST(ReplayRecorderTest, WaitfulUploadRemainsLiveAndIsMarkedUnsupported) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));
  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_queue_t* queue = iree_hal_device_queue(
      wrapped_device, /*family_ordinal=*/0, /*queue_ordinal=*/0);
  ASSERT_NE(nullptr, queue);

  const iree_hal_buffer_params_t buffer_params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER |
          IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED,
      /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
      /*.type=*/IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
  };
  iree_hal_buffer_t* target_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(wrapped_device), buffer_params,
      /*allocation_size=*/4, &target_buffer));

  iree_hal_semaphore_t* wait_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, iree_hal_make_queue_family_affinity(0),
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &wait_semaphore));
  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, iree_hal_make_queue_family_affinity(0),
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
  uint64_t wait_value = 1;
  const iree_hal_semaphore_list_t wait_list = {
      /*.count=*/1,
      /*.semaphores=*/&wait_semaphore,
      /*.payload_values=*/&wait_value,
  };
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_list = {
      /*.count=*/1,
      /*.semaphores=*/&signal_semaphore,
      /*.payload_values=*/&signal_value,
  };

  uint8_t upload_data[] = {0x10, 0x11, 0x12, 0x13};
  IREE_ASSERT_OK(iree_hal_queue_upload(
      queue, wait_list, signal_list, upload_data, target_buffer,
      /*target_offset=*/0, sizeof(upload_data)));
  const uint8_t expected_data[] = {0x20, 0x21, 0x22, 0x23};
  memcpy(upload_data, expected_data, sizeof(upload_data));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(wait_semaphore, wait_value,
                                           /*frontier=*/nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  uint8_t actual_data[sizeof(expected_data)] = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(target_buffer, /*source_offset=*/0,
                                          actual_data, sizeof(actual_data)));
  EXPECT_EQ(0, memcmp(expected_data, actual_data, sizeof(expected_data)));

  iree_hal_semaphore_release(signal_semaphore);
  iree_hal_semaphore_release(wait_semaphore);
  iree_hal_buffer_release(target_buffer);
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  const ReplayRecordSummary summary = ParseReplayRecordSummary(storage);
  EXPECT_EQ(0u, summary.queue.transfer_record_count);
  EXPECT_EQ(1u, summary.queue.unsupported_transfer_record_count);

  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_hal_replay_execute_file(
                            GetCapturedFileContents(storage), replay_group,
                            /*options=*/nullptr, iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayRecorderTest, WrappedDeviceRecordsHostCallAsUnsupported) {
  std::vector<uint8_t> storage(16384, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_queue_t* queue = iree_hal_device_queue(
      wrapped_device, /*family_ordinal=*/0, /*queue_ordinal=*/0);
  ASSERT_NE(nullptr, queue);
  int call_count = 0;
  iree_hal_host_call_t call =
      iree_hal_make_host_call(CountHostCall, &call_count);
  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signal_semaphore_list = {
      /*.count=*/1,
      /*.semaphores=*/&signal_semaphore,
      /*.payload_values=*/&signal_value,
  };
  const uint64_t args[4] = {0, 1, 2, 3};
  IREE_ASSERT_OK(iree_hal_queue_host_call(
      queue, iree_hal_semaphore_list_empty(), signal_semaphore_list, call, args,
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

TEST(ReplayRecorderTest, WrappedImportsAndExportsPreserveNativeBufferViews) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
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

  alignas(64) uint8_t imported_storage[16] = {0, 1, 2,  3,  4,  5,  6,  7,
                                              8, 9, 10, 11, 12, 13, 14, 15};
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
  IREE_ASSERT_OK(iree_hal_buffer_export(
      allocated_buffer, IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &exported_buffer));
  EXPECT_EQ(IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
            exported_buffer.type);

  iree_hal_buffer_t* subspan = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(imported_buffer, 4, 8,
                                         iree_allocator_system(), &subspan));
  iree_hal_buffer_t* nested_subspan = nullptr;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(subspan, 2, 4, iree_allocator_system(),
                                         &nested_subspan));
  iree_hal_buffer_release(subspan);
  iree_hal_buffer_release(imported_buffer);
  IREE_ASSERT_OK(iree_hal_buffer_export(
      nested_subspan, IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &exported_buffer));
  EXPECT_EQ(imported_storage + 6, exported_buffer.handle.host_allocation.ptr);
  EXPECT_EQ(4u, exported_buffer.size);
  const uint8_t expected[] = {6, 7, 8, 9};
  EXPECT_EQ(0, memcmp(expected, exported_buffer.handle.host_allocation.ptr,
                      sizeof(expected)));
  iree_hal_buffer_release(nested_subspan);
  iree_hal_buffer_release(allocated_buffer);

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
  EXPECT_EQ(2u, summary.unsupported_export_buffer_record_count);
  EXPECT_EQ(2u, summary.buffer_object_record_count);
}

TEST(ReplayRecorderTest, WrappedAllocatorRecordsBuffersAndMapping) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
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

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
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

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
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
      iree_hal_file_import(wrapped_device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
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

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
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
  IREE_ASSERT_OK(iree_hal_file_import(wrapped_device,
                                      IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                                      IREE_HAL_MEMORY_ACCESS_READ, file_handle,
                                      IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
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

TEST(ReplayRecorderTest, WrappedDeviceRecordsQueueExecuteWithSparseBindings) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_queue_t* wrapped_queue = iree_hal_device_queue(
      wrapped_device, /*family_ordinal=*/0, /*queue_ordinal=*/0);
  ASSERT_NE(nullptr, wrapped_queue);

  const iree_hal_buffer_params_t buffer_params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
      /*.access=*/IREE_HAL_MEMORY_ACCESS_WRITE,
      /*.type=*/IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
          IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(wrapped_device), buffer_params, 64, &buffer));

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(wrapped_queue),
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT, IREE_HAL_COMMAND_CATEGORY_TRANSFER,
      /*binding_capacity=*/2, &command_buffer));
  const uint32_t fill_pattern = 0xA5A5A5A5u;
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer,
      iree_hal_make_indirect_buffer_ref(/*buffer_slot=*/1, /*offset=*/0,
                                        /*length=*/64),
      &fill_pattern, sizeof(fill_pattern), IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_semaphore_t* wait_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &wait_semaphore));
  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
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
  const iree_hal_buffer_binding_t bindings[] = {
      {/*.buffer=*/nullptr, /*.offset=*/0, /*.length=*/0},
      {buffer, /*offset=*/0, IREE_HAL_WHOLE_BUFFER},
  };
  IREE_ASSERT_OK(iree_hal_queue_execute(
      wrapped_queue, wait_list, signal_list, command_buffer,
      {IREE_ARRAYSIZE(bindings), bindings}, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_queue_flush(wrapped_queue));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(signal_semaphore);
  iree_hal_semaphore_release(wait_semaphore);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(buffer);

  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  ReplayRecordSummary summary = ParseReplayRecordSummary(storage);
  EXPECT_EQ(1u, summary.command_buffer_object_record_count);
  EXPECT_EQ(2u, summary.semaphore_object_record_count);
  EXPECT_EQ(2u, summary.semaphore_object_payload_count);
  EXPECT_EQ(1u, summary.queue.execute_record_count);
  EXPECT_EQ(1u, summary.queue.execute_payload_count);
  ASSERT_FALSE(summary.queue.provisioned_object_ids.empty());
  EXPECT_EQ(summary.queue.provisioned_object_ids.front(),
            summary.queue.last_execute_object_id);
}

TEST(ReplayRecorderTest, RecordsAndReplaysCommandBufferAtomicOperations) {
  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));
  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_queue_t* wrapped_queue = iree_hal_device_queue(
      wrapped_device, /*family_ordinal=*/0, /*queue_ordinal=*/0);
  ASSERT_NE(nullptr, wrapped_queue);

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
      iree_hal_queue_family(wrapped_queue),
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT, IREE_HAL_COMMAND_CATEGORY_ATOMIC,
      /*binding_capacity=*/1, &command_buffer));
  const iree_hal_atomic_wait_params_t wait_params = {
      /*.value=*/UINT64_C(0x1020304050607080),
      /*.mask=*/UINT64_C(0xFFEEDDCCBBAA9988),
      /*.flags=*/IREE_HAL_ATOMIC_FLAGS_KNOWN,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_64,
      /*.condition=*/IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL,
      /*.target_error_mode=*/
      IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE,
  };
  const iree_hal_atomic_store_params_t store_params = {
      /*.value=*/UINT64_C(0xAABBCCDD),
      /*.flags=*/IREE_HAL_ATOMIC_FLAGS_KNOWN,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
      /*.target_error_mode=*/
      IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE,
  };
  const iree_hal_atomic_rmw_params_t rmw_params = {
      /*.operand=*/UINT64_C(0x11223344),
      /*.flags=*/IREE_HAL_ATOMIC_FLAGS_KNOWN,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
      /*.operation=*/IREE_HAL_ATOMIC_RMW_OPERATION_XOR,
      /*.target_error_mode=*/
      IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE,
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
  EXPECT_EQ(wait_params.target_error_mode,
            wait_payload.params.target_error_mode);
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
  EXPECT_EQ(store_params.target_error_mode,
            store_payload.params.target_error_mode);
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
  EXPECT_EQ(rmw_params.target_error_mode, rmw_payload.params.target_error_mode);
  EXPECT_EQ(0u, rmw_payload.params.reserved0);

  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  IREE_ASSERT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, /*options=*/nullptr,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
}

TEST(ReplayRecorderTest, RecordsAndReplaysVersion2ExactQueueAtomicOperations) {
  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);

  iree_hal_device_group_t* source_group = CreateTaskDeviceGroup();
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));
  iree_hal_device_t* wrapped_device =
      iree_hal_device_group_device_at(wrapped_group, 0);
  iree_hal_queue_t* queue = iree_hal_device_queue(
      wrapped_device, /*family_ordinal=*/0, /*queue_ordinal=*/0);
  ASSERT_NE(nullptr, queue);

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
      wrapped_device,
      /*queue_family_affinity=*/iree_hal_make_queue_family_affinity(0),
      /*initial_value=*/7, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &wait_semaphore));
  iree_hal_semaphore_t* signal_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      wrapped_device,
      /*queue_family_affinity=*/iree_hal_make_queue_family_affinity(0),
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &signal_semaphore));
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
      /*.target_error_mode=*/
      IREE_HAL_ATOMIC_TARGET_ERROR_MODE_DEFAULT,
  };
  const iree_hal_atomic_store_params_t store_params = {
      /*.value=*/UINT64_C(0xAABBCCDD),
      /*.flags=*/IREE_HAL_ATOMIC_FLAGS_KNOWN,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_32,
      /*.target_error_mode=*/
      IREE_HAL_ATOMIC_TARGET_ERROR_MODE_DEFAULT,
  };
  const iree_hal_atomic_rmw_params_t rmw_params = {
      /*.operand=*/UINT64_C(0x1122334455667788),
      /*.flags=*/IREE_HAL_ATOMIC_FLAGS_KNOWN,
      /*.width=*/IREE_HAL_ATOMIC_WIDTH_64,
      /*.operation=*/IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
      /*.target_error_mode=*/
      IREE_HAL_ATOMIC_TARGET_ERROR_MODE_DEFAULT,
  };

  IREE_ASSERT_OK(iree_hal_queue_atomic_wait(queue, wait_list, wait_signal_list,
                                            buffer, /*target_offset=*/8,
                                            wait_params));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      wait_signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(
      iree_hal_queue_atomic_store(queue, wait_list, store_signal_list, buffer,
                                  /*target_offset=*/16, store_params));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      store_signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_queue_atomic_rmw(queue, wait_list, rmw_signal_list,
                                           buffer, /*target_offset=*/24,
                                           rmw_params));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      rmw_signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_semaphore_release(signal_semaphore);
  iree_hal_semaphore_release(wait_semaphore);
  iree_hal_buffer_release(buffer);
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_replay_recorder_release(recorder);
  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);

  // The three target-error bytes occupied required-zero reserved storage in
  // 7.2. This capture uses their default value, so changing only the file minor
  // produces the exact legacy wire interpretation that current replay must
  // continue to decode and execute.
  auto* file_header =
      reinterpret_cast<iree_hal_replay_file_header_t*>(storage.data());
  file_header->version_minor = 2;

  const auto records = ParseOperationRecords(storage);
  const auto* wait_record = FindOperationRecord(
      records, IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_ATOMIC_WAIT);
  ASSERT_NE(nullptr, wait_record);
  EXPECT_EQ(IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_WAIT,
            wait_record->header.payload_type);
  EXPECT_EQ(IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE, wait_record->header.object_type);
  EXPECT_EQ((uint32_t)IREE_STATUS_OK, wait_record->header.status_code);
  iree_hal_replay_queue_atomic_wait_payload_t wait_payload;
  ASSERT_GE(wait_record->payload.data_length, sizeof(wait_payload));
  memcpy(&wait_payload, wait_record->payload.data, sizeof(wait_payload));
  EXPECT_NE(IREE_HAL_REPLAY_OBJECT_ID_NONE, wait_payload.target_ref.buffer_id);
  EXPECT_EQ(wait_payload.target_ref.buffer_id,
            wait_record->header.related_object_id);
  EXPECT_EQ(8u, wait_payload.target_ref.offset);
  EXPECT_EQ(8u, wait_payload.target_ref.length);
  EXPECT_EQ(1u, wait_payload.wait_semaphore_count);
  EXPECT_EQ(1u, wait_payload.signal_semaphore_count);
  EXPECT_EQ(wait_params.value, wait_payload.params.value);
  EXPECT_EQ(wait_params.mask, wait_payload.params.mask);
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAGS_KNOWN, wait_payload.params.flags);
  EXPECT_EQ(wait_params.width, wait_payload.params.width);
  EXPECT_EQ(wait_params.condition, wait_payload.params.condition);
  EXPECT_EQ(wait_params.target_error_mode,
            wait_payload.params.target_error_mode);
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
      records, IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_ATOMIC_STORE);
  ASSERT_NE(nullptr, store_record);
  EXPECT_EQ(IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_STORE,
            store_record->header.payload_type);
  EXPECT_EQ(IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
            store_record->header.object_type);
  EXPECT_EQ((uint32_t)IREE_STATUS_OK, store_record->header.status_code);
  iree_hal_replay_queue_atomic_store_payload_t store_payload;
  ASSERT_GE(store_record->payload.data_length, sizeof(store_payload));
  memcpy(&store_payload, store_record->payload.data, sizeof(store_payload));
  EXPECT_EQ(wait_payload.target_ref.buffer_id,
            store_payload.target_ref.buffer_id);
  EXPECT_EQ(store_payload.target_ref.buffer_id,
            store_record->header.related_object_id);
  EXPECT_EQ(16u, store_payload.target_ref.offset);
  EXPECT_EQ(4u, store_payload.target_ref.length);
  EXPECT_EQ(1u, store_payload.wait_semaphore_count);
  EXPECT_EQ(1u, store_payload.signal_semaphore_count);
  EXPECT_EQ(store_params.value, store_payload.params.value);
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAGS_KNOWN, store_payload.params.flags);
  EXPECT_EQ(store_params.width, store_payload.params.width);
  EXPECT_EQ(store_params.target_error_mode,
            store_payload.params.target_error_mode);
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
      records, IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_ATOMIC_RMW);
  ASSERT_NE(nullptr, rmw_record);
  EXPECT_EQ(IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_RMW,
            rmw_record->header.payload_type);
  EXPECT_EQ(IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE, rmw_record->header.object_type);
  EXPECT_EQ((uint32_t)IREE_STATUS_OK, rmw_record->header.status_code);
  iree_hal_replay_queue_atomic_rmw_payload_t rmw_payload;
  ASSERT_GE(rmw_record->payload.data_length, sizeof(rmw_payload));
  memcpy(&rmw_payload, rmw_record->payload.data, sizeof(rmw_payload));
  EXPECT_EQ(wait_payload.target_ref.buffer_id,
            rmw_payload.target_ref.buffer_id);
  EXPECT_EQ(rmw_payload.target_ref.buffer_id,
            rmw_record->header.related_object_id);
  EXPECT_EQ(24u, rmw_payload.target_ref.offset);
  EXPECT_EQ(8u, rmw_payload.target_ref.length);
  EXPECT_EQ(1u, rmw_payload.wait_semaphore_count);
  EXPECT_EQ(1u, rmw_payload.signal_semaphore_count);
  EXPECT_EQ(rmw_params.operand, rmw_payload.params.operand);
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAGS_KNOWN, rmw_payload.params.flags);
  EXPECT_EQ(rmw_params.width, rmw_payload.params.width);
  EXPECT_EQ(rmw_params.operation, rmw_payload.params.operation);
  EXPECT_EQ(rmw_params.target_error_mode, rmw_payload.params.target_error_mode);
  EXPECT_EQ(0u, rmw_payload.params.reserved0);
  iree_hal_replay_semaphore_timepoint_payload_t rmw_wait_timepoint = {};
  iree_hal_replay_semaphore_timepoint_payload_t rmw_signal_timepoint = {};
  ReadQueueSemaphoreTail(*rmw_record, sizeof(rmw_payload), &rmw_wait_timepoint,
                         &rmw_signal_timepoint);
  EXPECT_EQ(wait_timepoint.semaphore_id, rmw_wait_timepoint.semaphore_id);
  EXPECT_EQ(signal_timepoint.semaphore_id, rmw_signal_timepoint.semaphore_id);
  EXPECT_EQ(7u, rmw_wait_timepoint.value);
  EXPECT_EQ(11u, rmw_signal_timepoint.value);

  iree_hal_device_group_t* replay_group = CreateTaskDeviceGroup();
  IREE_ASSERT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, /*options=*/nullptr,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);
}

// Minimal VMM allocator used to verify recorder identity and forwarding rules.
typedef struct RecorderVmmAllocatorState {
  iree_host_size_t reserve_count = 0;
  iree_host_size_t release_attempt_count = 0;
  iree_host_size_t release_count = 0;
  iree_host_size_t physical_allocate_count = 0;
  iree_host_size_t physical_free_attempt_count = 0;
  iree_host_size_t physical_free_count = 0;
  iree_host_size_t map_count = 0;
  iree_host_size_t unmap_count = 0;
  iree_host_size_t protect_count = 0;
  iree_host_size_t advise_count = 0;
  iree_host_size_t release_failures_remaining = 0;
  iree_host_size_t physical_free_failures_remaining = 0;
  iree_hal_buffer_t* virtual_buffer = nullptr;
  iree_hal_physical_memory_t* physical_memory = nullptr;
  bool is_mapped = false;
  iree_hal_queue_family_affinity_t reserve_queue_family_affinity = 0;
  iree_device_size_t reserve_size = 0;
  iree_hal_buffer_params_t physical_params = {};
  iree_device_size_t physical_size = 0;
  iree_device_size_t map_virtual_offset = 0;
  iree_device_size_t map_physical_offset = 0;
  iree_device_size_t map_size = 0;
  iree_hal_queue_family_affinity_t protect_queue_family_affinity = 0;
  iree_hal_virtual_memory_access_scope_t protect_access_scope = 0;
  iree_hal_memory_protection_t protection = 0;
  iree_hal_queue_family_affinity_t advise_queue_family_affinity = 0;
  iree_hal_memory_advice_t advice = 0;
} RecorderVmmAllocatorState;

typedef struct RecorderVmmPhysicalMemory {
  iree_allocator_t host_allocator;
} RecorderVmmPhysicalMemory;

typedef struct RecorderVmmAllocator {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_allocator_t* heap_allocator;
  RecorderVmmAllocatorState* state;
} RecorderVmmAllocator;

extern const iree_hal_allocator_vtable_t recorder_vmm_allocator_vtable;

static RecorderVmmAllocator* RecorderVmmAllocatorCast(
    iree_hal_allocator_t* base_allocator) {
  IREE_HAL_ASSERT_TYPE(base_allocator, &recorder_vmm_allocator_vtable);
  return reinterpret_cast<RecorderVmmAllocator*>(base_allocator);
}

static void RecorderVmmAllocatorDestroy(iree_hal_allocator_t* base_allocator) {
  RecorderVmmAllocator* allocator = RecorderVmmAllocatorCast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  iree_hal_allocator_release(allocator->heap_allocator);
  iree_allocator_free(host_allocator, allocator);
}

static iree_allocator_t RecorderVmmAllocatorHostAllocator(
    const iree_hal_allocator_t* base_allocator) {
  const RecorderVmmAllocator* allocator =
      reinterpret_cast<const RecorderVmmAllocator*>(base_allocator);
  return allocator->host_allocator;
}

static iree_status_t RecorderVmmAllocatorAllocateBuffer(
    iree_hal_allocator_t* base_allocator,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  RecorderVmmAllocator* allocator = RecorderVmmAllocatorCast(base_allocator);
  return iree_hal_allocator_allocate_buffer(allocator->heap_allocator, *params,
                                            allocation_size, out_buffer);
}

static bool RecorderVmmAllocatorSupportsVirtualMemory(
    iree_hal_allocator_t* base_allocator) {
  return true;
}

static iree_status_t RecorderVmmAllocatorQueryGranularity(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_params_t params,
    iree_device_size_t* out_minimum_page_size,
    iree_device_size_t* out_recommended_page_size) {
  *out_minimum_page_size = 4096;
  *out_recommended_page_size = 65536;
  return iree_ok_status();
}

static iree_status_t RecorderVmmAllocatorReserve(
    iree_hal_allocator_t* base_allocator,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_device_size_t size, iree_hal_buffer_t** out_virtual_buffer) {
  RecorderVmmAllocator* allocator = RecorderVmmAllocatorCast(base_allocator);
  RecorderVmmAllocatorState* state = allocator->state;
  if (state->virtual_buffer) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test allocator already has a reservation");
  }
  iree_hal_buffer_params_t params = {};
  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE;
  params.access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.queue_family_affinity = queue_family_affinity;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      allocator->heap_allocator, params, size, out_virtual_buffer));
  state->virtual_buffer = *out_virtual_buffer;
  state->reserve_queue_family_affinity = queue_family_affinity;
  state->reserve_size = size;
  ++state->reserve_count;
  return iree_ok_status();
}

static iree_status_t RecorderVmmAllocatorRelease(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer) {
  RecorderVmmAllocator* allocator = RecorderVmmAllocatorCast(base_allocator);
  RecorderVmmAllocatorState* state = allocator->state;
  ++state->release_attempt_count;
  if (virtual_buffer != state->virtual_buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "test reservation handle mismatch");
  }
  if (state->release_failures_remaining) {
    --state->release_failures_remaining;
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "injected reservation release failure");
  }
  if (state->is_mapped) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test reservation is still mapped");
  }
  state->virtual_buffer = nullptr;
  ++state->release_count;
  iree_hal_buffer_release(virtual_buffer);
  return iree_ok_status();
}

static iree_status_t RecorderVmmAllocatorAllocatePhysicalMemory(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_params_t params,
    iree_device_size_t size, iree_allocator_t host_allocator,
    iree_hal_physical_memory_t** out_physical_memory) {
  RecorderVmmAllocator* allocator = RecorderVmmAllocatorCast(base_allocator);
  RecorderVmmAllocatorState* state = allocator->state;
  if (state->physical_memory) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test allocator already has physical memory");
  }
  RecorderVmmPhysicalMemory* physical_memory = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*physical_memory),
                            reinterpret_cast<void**>(&physical_memory)));
  physical_memory->host_allocator = host_allocator;
  state->physical_memory =
      reinterpret_cast<iree_hal_physical_memory_t*>(physical_memory);
  state->physical_params = params;
  state->physical_size = size;
  ++state->physical_allocate_count;
  *out_physical_memory = state->physical_memory;
  return iree_ok_status();
}

static iree_status_t RecorderVmmAllocatorFreePhysicalMemory(
    iree_hal_allocator_t* base_allocator,
    iree_hal_physical_memory_t* physical_memory) {
  RecorderVmmAllocator* allocator = RecorderVmmAllocatorCast(base_allocator);
  RecorderVmmAllocatorState* state = allocator->state;
  ++state->physical_free_attempt_count;
  if (physical_memory != state->physical_memory) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "test physical memory handle mismatch");
  }
  if (state->physical_free_failures_remaining) {
    --state->physical_free_failures_remaining;
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "injected physical memory free failure");
  }
  if (state->is_mapped) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test physical memory is still mapped");
  }
  RecorderVmmPhysicalMemory* test_physical_memory =
      reinterpret_cast<RecorderVmmPhysicalMemory*>(physical_memory);
  iree_allocator_t host_allocator = test_physical_memory->host_allocator;
  state->physical_memory = nullptr;
  ++state->physical_free_count;
  iree_allocator_free(host_allocator, test_physical_memory);
  return iree_ok_status();
}

static iree_status_t RecorderVmmAllocatorMap(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size) {
  RecorderVmmAllocator* allocator = RecorderVmmAllocatorCast(base_allocator);
  RecorderVmmAllocatorState* state = allocator->state;
  if (virtual_buffer != state->virtual_buffer ||
      physical_memory != state->physical_memory || state->is_mapped) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test mapping state mismatch");
  }
  state->is_mapped = true;
  state->map_virtual_offset = virtual_offset;
  state->map_physical_offset = physical_offset;
  state->map_size = size;
  ++state->map_count;
  return iree_ok_status();
}

static iree_status_t RecorderVmmAllocatorUnmap(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  RecorderVmmAllocator* allocator = RecorderVmmAllocatorCast(base_allocator);
  RecorderVmmAllocatorState* state = allocator->state;
  if (virtual_buffer != state->virtual_buffer || !state->is_mapped ||
      virtual_offset != state->map_virtual_offset || size != state->map_size) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test unmapping state mismatch");
  }
  state->is_mapped = false;
  ++state->unmap_count;
  return iree_ok_status();
}

static iree_status_t RecorderVmmAllocatorProtect(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_virtual_memory_access_scope_t access_scope,
    iree_hal_memory_protection_t protection) {
  RecorderVmmAllocator* allocator = RecorderVmmAllocatorCast(base_allocator);
  RecorderVmmAllocatorState* state = allocator->state;
  if (virtual_buffer != state->virtual_buffer || !state->is_mapped ||
      virtual_offset != state->map_virtual_offset || size != state->map_size) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test protection state mismatch");
  }
  state->protect_queue_family_affinity = queue_family_affinity;
  state->protect_access_scope = access_scope;
  state->protection = protection;
  ++state->protect_count;
  return iree_ok_status();
}

static iree_status_t RecorderVmmAllocatorAdvise(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_memory_advice_t advice) {
  RecorderVmmAllocator* allocator = RecorderVmmAllocatorCast(base_allocator);
  RecorderVmmAllocatorState* state = allocator->state;
  if (virtual_buffer != state->virtual_buffer || !state->is_mapped ||
      virtual_offset != state->map_virtual_offset || size != state->map_size) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test advice state mismatch");
  }
  state->advise_queue_family_affinity = queue_family_affinity;
  state->advice = advice;
  ++state->advise_count;
  return iree_ok_status();
}

const iree_hal_allocator_vtable_t recorder_vmm_allocator_vtable = {
    /*.destroy=*/RecorderVmmAllocatorDestroy,
    /*.host_allocator=*/RecorderVmmAllocatorHostAllocator,
    /*.trim=*/nullptr,
    /*.query_statistics=*/nullptr,
    /*.query_memory_heaps=*/nullptr,
    /*.query_buffer_compatibility=*/nullptr,
    /*.allocate_buffer=*/RecorderVmmAllocatorAllocateBuffer,
    /*.deallocate_buffer=*/nullptr,
    /*.import_buffer=*/nullptr,
    /*.supports_virtual_memory=*/RecorderVmmAllocatorSupportsVirtualMemory,
    /*.virtual_memory_query_granularity=*/
    RecorderVmmAllocatorQueryGranularity,
    /*.virtual_memory_reserve=*/RecorderVmmAllocatorReserve,
    /*.virtual_memory_release=*/RecorderVmmAllocatorRelease,
    /*.physical_memory_allocate=*/RecorderVmmAllocatorAllocatePhysicalMemory,
    /*.physical_memory_free=*/RecorderVmmAllocatorFreePhysicalMemory,
    /*.virtual_memory_map=*/RecorderVmmAllocatorMap,
    /*.virtual_memory_unmap=*/RecorderVmmAllocatorUnmap,
    /*.virtual_memory_protect=*/RecorderVmmAllocatorProtect,
    /*.virtual_memory_advise=*/RecorderVmmAllocatorAdvise,
};

static iree_hal_allocator_t* CreateRecorderVmmAllocator(
    RecorderVmmAllocatorState* state) {
  iree_hal_allocator_t* heap_allocator = nullptr;
  IREE_CHECK_OK(iree_hal_allocator_create_heap(
      IREE_SV("recorder-vmm-test"), iree_allocator_system(),
      iree_allocator_system(), &heap_allocator));
  RecorderVmmAllocator* allocator = nullptr;
  IREE_CHECK_OK(iree_allocator_malloc(iree_allocator_system(),
                                      sizeof(*allocator),
                                      reinterpret_cast<void**>(&allocator)));
  iree_hal_resource_initialize(&recorder_vmm_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = iree_allocator_system();
  allocator->heap_allocator = heap_allocator;
  allocator->state = state;
  return reinterpret_cast<iree_hal_allocator_t*>(allocator);
}

static iree_hal_allocator_t* WrapRecorderVmmAllocator(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_allocator_t* base_allocator) {
  iree_hal_allocator_t* wrapped_allocator = nullptr;
  IREE_CHECK_OK(iree_hal_replay_recorder_wrap_allocator(
      recorder, device_id, /*placement_device=*/nullptr, base_allocator,
      iree_allocator_system(), &wrapped_allocator));
  return wrapped_allocator;
}

constexpr iree_hal_queue_family_affinity_t kRecorderVmmQueueFamilyAffinity = 1;
constexpr iree_device_size_t kRecorderVmmReservationSize = 16384;
constexpr iree_device_size_t kRecorderVmmVirtualOffset = 4096;
constexpr iree_device_size_t kRecorderVmmPhysicalOffset = 8192;
constexpr iree_device_size_t kRecorderVmmMappingSize = 4096;

static iree_hal_buffer_params_t RecorderVmmPhysicalParams() {
  iree_hal_buffer_params_t params = {};
  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE;
  params.access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.queue_family_affinity = kRecorderVmmQueueFamilyAffinity;
  params.min_alignment = 4096;
  return params;
}

static bool HasCapturedObject(const std::vector<uint8_t>& storage,
                              iree_hal_replay_object_type_t object_type,
                              iree_hal_replay_object_id_t object_id) {
  iree_hal_replay_file_header_t file_header;
  iree_host_size_t offset = 0;
  IREE_CHECK_OK(iree_hal_replay_file_parse_header(
      iree_make_const_byte_span(storage.data(), storage.size()), &file_header,
      &offset));
  const iree_const_byte_span_t file_contents = iree_make_const_byte_span(
      storage.data(), static_cast<iree_host_size_t>(file_header.file_length));
  while (offset < file_contents.data_length) {
    iree_hal_replay_file_record_t record;
    IREE_CHECK_OK(iree_hal_replay_file_parse_record(file_contents, offset,
                                                    &record, &offset));
    if (record.header.record_type == IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT &&
        record.header.object_type == object_type &&
        record.header.object_id == object_id) {
      return true;
    }
  }
  return false;
}

static iree_host_size_t FindCapturedOperationOffset(
    const std::vector<uint8_t>& storage,
    iree_hal_replay_operation_code_t operation_code) {
  iree_hal_replay_file_header_t file_header;
  iree_host_size_t offset = 0;
  IREE_CHECK_OK(iree_hal_replay_file_parse_header(
      iree_make_const_byte_span(storage.data(), storage.size()), &file_header,
      &offset));
  const iree_const_byte_span_t file_contents = iree_make_const_byte_span(
      storage.data(), static_cast<iree_host_size_t>(file_header.file_length));
  while (offset < file_contents.data_length) {
    const iree_host_size_t record_offset = offset;
    iree_hal_replay_file_record_t record;
    IREE_CHECK_OK(iree_hal_replay_file_parse_record(file_contents, offset,
                                                    &record, &offset));
    if (record.header.record_type ==
            IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION &&
        record.header.operation_code == operation_code) {
      return record_offset;
    }
  }
  return 0;
}

static std::vector<uint8_t> CaptureMinimalRecorderVmmLifecycle() {
  std::vector<uint8_t> storage(16384, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);
  RecorderVmmAllocatorState state;
  iree_hal_allocator_t* base_allocator = CreateRecorderVmmAllocator(&state);
  iree_hal_allocator_t* allocator =
      WrapRecorderVmmAllocator(recorder, /*device_id=*/1, base_allocator);
  iree_hal_buffer_t* virtual_buffer = nullptr;
  IREE_CHECK_OK(iree_hal_allocator_virtual_memory_reserve(
      allocator, kRecorderVmmQueueFamilyAffinity, kRecorderVmmReservationSize,
      &virtual_buffer));
  iree_hal_physical_memory_t* physical_memory = nullptr;
  IREE_CHECK_OK(iree_hal_allocator_physical_memory_allocate(
      allocator, RecorderVmmPhysicalParams(), kRecorderVmmReservationSize,
      iree_allocator_system(), &physical_memory));
  IREE_CHECK_OK(
      iree_hal_allocator_physical_memory_free(allocator, physical_memory));
  IREE_CHECK_OK(
      iree_hal_allocator_virtual_memory_release(allocator, virtual_buffer));
  IREE_CHECK_OK(iree_hal_replay_recorder_close(recorder));
  iree_hal_allocator_release(allocator);
  iree_hal_allocator_release(base_allocator);
  iree_hal_replay_recorder_release(recorder);
  return storage;
}

TEST(ReplayRecorderVmmTest, RecordsStableIdsAndCompletePayloads) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);
  RecorderVmmAllocatorState state;
  iree_hal_allocator_t* base_allocator = CreateRecorderVmmAllocator(&state);
  iree_hal_allocator_t* allocator =
      WrapRecorderVmmAllocator(recorder, /*device_id=*/1, base_allocator);

  const iree_hal_buffer_params_t physical_params = RecorderVmmPhysicalParams();
  iree_hal_buffer_t* virtual_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_reserve(
      allocator, kRecorderVmmQueueFamilyAffinity, kRecorderVmmReservationSize,
      &virtual_buffer));
  const uintptr_t virtual_buffer_address =
      reinterpret_cast<uintptr_t>(virtual_buffer);
  iree_hal_physical_memory_t* physical_memory = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_allocate(
      allocator, physical_params, kRecorderVmmReservationSize,
      iree_allocator_system(), &physical_memory));
  const uintptr_t physical_memory_address =
      reinterpret_cast<uintptr_t>(physical_memory);
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_map(
      allocator, virtual_buffer, kRecorderVmmVirtualOffset, physical_memory,
      kRecorderVmmPhysicalOffset, kRecorderVmmMappingSize));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_protect(
      allocator, virtual_buffer, kRecorderVmmVirtualOffset,
      kRecorderVmmMappingSize, kRecorderVmmQueueFamilyAffinity,
      IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE,
      IREE_HAL_MEMORY_PROTECTION_READ_WRITE));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_advise(
      allocator, virtual_buffer, kRecorderVmmVirtualOffset,
      kRecorderVmmMappingSize, kRecorderVmmQueueFamilyAffinity,
      IREE_HAL_MEMORY_ADVICE_WILL_NEED));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_unmap(
      allocator, virtual_buffer, kRecorderVmmVirtualOffset,
      kRecorderVmmMappingSize));
  IREE_ASSERT_OK(
      iree_hal_allocator_physical_memory_free(allocator, physical_memory));
  IREE_ASSERT_OK(
      iree_hal_allocator_virtual_memory_release(allocator, virtual_buffer));
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));

  const auto records = ParseOperationRecords(storage);
  const auto* reserve_record = FindOperationRecord(
      records, IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RESERVE);
  ASSERT_NE(nullptr, reserve_record);
  ASSERT_EQ(sizeof(iree_hal_replay_allocator_virtual_memory_reserve_payload_t),
            reserve_record->payload.data_length);
  iree_hal_replay_allocator_virtual_memory_reserve_payload_t reserve_payload;
  memcpy(&reserve_payload, reserve_record->payload.data,
         sizeof(reserve_payload));
  const iree_hal_replay_object_id_t virtual_buffer_id =
      reserve_record->header.related_object_id;
  EXPECT_NE(IREE_HAL_REPLAY_OBJECT_ID_NONE, virtual_buffer_id);
  EXPECT_NE(virtual_buffer_address, static_cast<uintptr_t>(virtual_buffer_id));
  EXPECT_EQ(kRecorderVmmQueueFamilyAffinity,
            reserve_payload.queue_family_affinity);
  EXPECT_EQ(kRecorderVmmReservationSize, reserve_payload.size);
  EXPECT_TRUE(HasCapturedObject(storage, IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
                                virtual_buffer_id));

  const auto* allocate_record = FindOperationRecord(
      records,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE);
  ASSERT_NE(nullptr, allocate_record);
  ASSERT_EQ(
      sizeof(iree_hal_replay_allocator_physical_memory_allocate_payload_t),
      allocate_record->payload.data_length);
  iree_hal_replay_allocator_physical_memory_allocate_payload_t allocate_payload;
  memcpy(&allocate_payload, allocate_record->payload.data,
         sizeof(allocate_payload));
  const iree_hal_replay_object_id_t physical_memory_id =
      allocate_record->header.related_object_id;
  EXPECT_NE(IREE_HAL_REPLAY_OBJECT_ID_NONE, physical_memory_id);
  EXPECT_NE(virtual_buffer_id, physical_memory_id);
  EXPECT_NE(physical_memory_address,
            static_cast<uintptr_t>(physical_memory_id));
  EXPECT_EQ(kRecorderVmmReservationSize,
            allocate_payload.allocation.allocation_size);
  EXPECT_EQ(physical_params.queue_family_affinity,
            allocate_payload.allocation.queue_family_affinity);
  EXPECT_EQ(physical_params.min_alignment,
            allocate_payload.allocation.min_alignment);
  EXPECT_EQ(physical_params.usage, allocate_payload.allocation.usage);
  EXPECT_EQ(physical_params.type, allocate_payload.allocation.type);
  EXPECT_EQ(physical_params.access, allocate_payload.allocation.access);
  EXPECT_EQ(0u, allocate_payload.allocation.reserved0);
  EXPECT_EQ(0u, allocate_payload.allocation.reserved1);
  EXPECT_TRUE(HasCapturedObject(storage,
                                IREE_HAL_REPLAY_OBJECT_TYPE_PHYSICAL_MEMORY,
                                physical_memory_id));

  const auto* map_record = FindOperationRecord(
      records, IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_MAP);
  ASSERT_NE(nullptr, map_record);
  ASSERT_EQ(sizeof(iree_hal_replay_allocator_virtual_memory_map_payload_t),
            map_record->payload.data_length);
  iree_hal_replay_allocator_virtual_memory_map_payload_t map_payload;
  memcpy(&map_payload, map_record->payload.data, sizeof(map_payload));
  EXPECT_EQ(virtual_buffer_id, map_payload.virtual_buffer_id);
  EXPECT_EQ(physical_memory_id, map_payload.physical_memory_id);
  EXPECT_EQ(kRecorderVmmVirtualOffset, map_payload.virtual_offset);
  EXPECT_EQ(kRecorderVmmPhysicalOffset, map_payload.physical_offset);
  EXPECT_EQ(kRecorderVmmMappingSize, map_payload.size);

  const auto* protect_record = FindOperationRecord(
      records, IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT);
  ASSERT_NE(nullptr, protect_record);
  ASSERT_EQ(sizeof(iree_hal_replay_allocator_virtual_memory_protect_payload_t),
            protect_record->payload.data_length);
  iree_hal_replay_allocator_virtual_memory_protect_payload_t protect_payload;
  memcpy(&protect_payload, protect_record->payload.data,
         sizeof(protect_payload));
  EXPECT_EQ(virtual_buffer_id, protect_payload.virtual_buffer_id);
  EXPECT_EQ(kRecorderVmmVirtualOffset, protect_payload.virtual_offset);
  EXPECT_EQ(kRecorderVmmMappingSize, protect_payload.size);
  EXPECT_EQ(kRecorderVmmQueueFamilyAffinity,
            protect_payload.queue_family_affinity);
  EXPECT_EQ(IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE,
            protect_payload.access_scope);
  EXPECT_EQ(0u, protect_payload.reserved0);
  EXPECT_EQ(IREE_HAL_MEMORY_PROTECTION_READ_WRITE, protect_payload.protection);

  const auto* advise_record = FindOperationRecord(
      records, IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_ADVISE);
  ASSERT_NE(nullptr, advise_record);
  iree_hal_replay_allocator_virtual_memory_advise_payload_t advise_payload;
  ASSERT_EQ(sizeof(advise_payload), advise_record->payload.data_length);
  memcpy(&advise_payload, advise_record->payload.data, sizeof(advise_payload));
  EXPECT_EQ(virtual_buffer_id, advise_payload.virtual_buffer_id);
  EXPECT_EQ(kRecorderVmmQueueFamilyAffinity,
            advise_payload.queue_family_affinity);
  EXPECT_EQ(IREE_HAL_MEMORY_ADVICE_WILL_NEED, advise_payload.advice);

  const auto* unmap_record = FindOperationRecord(
      records, IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP);
  ASSERT_NE(nullptr, unmap_record);
  iree_hal_replay_allocator_virtual_memory_unmap_payload_t unmap_payload;
  ASSERT_EQ(sizeof(unmap_payload), unmap_record->payload.data_length);
  memcpy(&unmap_payload, unmap_record->payload.data, sizeof(unmap_payload));
  EXPECT_EQ(virtual_buffer_id, unmap_payload.virtual_buffer_id);
  EXPECT_EQ(kRecorderVmmVirtualOffset, unmap_payload.virtual_offset);
  EXPECT_EQ(kRecorderVmmMappingSize, unmap_payload.size);

  const auto* free_record = FindOperationRecord(
      records, IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_FREE);
  ASSERT_NE(nullptr, free_record);
  iree_hal_replay_allocator_physical_memory_free_payload_t free_payload;
  ASSERT_EQ(sizeof(free_payload), free_record->payload.data_length);
  memcpy(&free_payload, free_record->payload.data, sizeof(free_payload));
  EXPECT_EQ(physical_memory_id, free_payload.physical_memory_id);

  const auto* release_record = FindOperationRecord(
      records, IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE);
  ASSERT_NE(nullptr, release_record);
  iree_hal_replay_allocator_virtual_memory_release_payload_t release_payload;
  ASSERT_EQ(sizeof(release_payload), release_record->payload.data_length);
  memcpy(&release_payload, release_record->payload.data,
         sizeof(release_payload));
  EXPECT_EQ(virtual_buffer_id, release_payload.virtual_buffer_id);

  EXPECT_EQ(1u, state.reserve_count);
  EXPECT_EQ(1u, state.physical_allocate_count);
  EXPECT_EQ(1u, state.map_count);
  EXPECT_EQ(1u, state.unmap_count);
  EXPECT_EQ(1u, state.protect_count);
  EXPECT_EQ(1u, state.advise_count);
  EXPECT_EQ(kRecorderVmmQueueFamilyAffinity,
            state.protect_queue_family_affinity);
  EXPECT_EQ(IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE,
            state.protect_access_scope);

  iree_hal_allocator_release(allocator);
  iree_hal_allocator_release(base_allocator);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayRecorderVmmTest, RejectsForeignRawAndConsumedHandles) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);
  RecorderVmmAllocatorState state_a;
  RecorderVmmAllocatorState state_b;
  iree_hal_allocator_t* base_allocator_a = CreateRecorderVmmAllocator(&state_a);
  iree_hal_allocator_t* base_allocator_b = CreateRecorderVmmAllocator(&state_b);
  iree_hal_allocator_t* allocator_a =
      WrapRecorderVmmAllocator(recorder, /*device_id=*/1, base_allocator_a);
  iree_hal_allocator_t* allocator_b =
      WrapRecorderVmmAllocator(recorder, /*device_id=*/1, base_allocator_b);

  iree_hal_buffer_t* virtual_buffer_a = nullptr;
  iree_hal_buffer_t* virtual_buffer_b = nullptr;
  iree_hal_physical_memory_t* physical_memory_a = nullptr;
  iree_hal_physical_memory_t* physical_memory_b = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_reserve(
      allocator_a, kRecorderVmmQueueFamilyAffinity, kRecorderVmmReservationSize,
      &virtual_buffer_a));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_reserve(
      allocator_b, kRecorderVmmQueueFamilyAffinity, kRecorderVmmReservationSize,
      &virtual_buffer_b));
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_allocate(
      allocator_a, RecorderVmmPhysicalParams(), kRecorderVmmReservationSize,
      iree_allocator_system(), &physical_memory_a));
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_allocate(
      allocator_b, RecorderVmmPhysicalParams(), kRecorderVmmReservationSize,
      iree_allocator_system(), &physical_memory_b));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_allocator_virtual_memory_map(
          allocator_b, virtual_buffer_b, kRecorderVmmVirtualOffset,
          physical_memory_a, kRecorderVmmPhysicalOffset,
          kRecorderVmmMappingSize));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_allocator_virtual_memory_map(
          allocator_a, virtual_buffer_b, kRecorderVmmVirtualOffset,
          physical_memory_a, kRecorderVmmPhysicalOffset,
          kRecorderVmmMappingSize));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_allocator_physical_memory_free(allocator_b, physical_memory_a));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_allocator_virtual_memory_release(allocator_b, virtual_buffer_a));
  auto* raw_physical_memory = reinterpret_cast<iree_hal_physical_memory_t*>(
      static_cast<uintptr_t>(0x1234));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_allocator_virtual_memory_map(
          allocator_a, virtual_buffer_a, kRecorderVmmVirtualOffset,
          raw_physical_memory, kRecorderVmmPhysicalOffset,
          kRecorderVmmMappingSize));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_allocator_physical_memory_free(
                            allocator_a, raw_physical_memory));

  iree_hal_buffer_t* ordinary_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator_a, RecorderVmmPhysicalParams(), /*allocation_size=*/64,
      &ordinary_buffer));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_allocator_virtual_memory_release(allocator_a, ordinary_buffer));
  iree_hal_buffer_release(ordinary_buffer);

  EXPECT_EQ(0u, state_a.map_count);
  EXPECT_EQ(0u, state_b.map_count);
  EXPECT_EQ(0u, state_a.release_attempt_count);
  EXPECT_EQ(0u, state_b.release_attempt_count);
  EXPECT_EQ(0u, state_a.physical_free_attempt_count);
  EXPECT_EQ(0u, state_b.physical_free_attempt_count);

  IREE_ASSERT_OK(
      iree_hal_allocator_physical_memory_free(allocator_a, physical_memory_a));
  IREE_ASSERT_OK(
      iree_hal_allocator_physical_memory_free(allocator_b, physical_memory_b));
  IREE_ASSERT_OK(
      iree_hal_allocator_virtual_memory_release(allocator_a, virtual_buffer_a));
  IREE_ASSERT_OK(
      iree_hal_allocator_virtual_memory_release(allocator_b, virtual_buffer_b));
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));

  iree_hal_allocator_release(allocator_b);
  iree_hal_allocator_release(allocator_a);
  iree_hal_allocator_release(base_allocator_b);
  iree_hal_allocator_release(base_allocator_a);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayRecorderVmmTest, FailedConsumptionIsRetryableAndSuccessIsFinal) {
  std::vector<uint8_t> storage(32768, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);
  RecorderVmmAllocatorState state;
  state.release_failures_remaining = 1;
  state.physical_free_failures_remaining = 1;
  iree_hal_allocator_t* base_allocator = CreateRecorderVmmAllocator(&state);
  iree_hal_allocator_t* allocator =
      WrapRecorderVmmAllocator(recorder, /*device_id=*/1, base_allocator);

  iree_hal_buffer_t* virtual_buffer = nullptr;
  iree_hal_physical_memory_t* physical_memory = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_reserve(
      allocator, kRecorderVmmQueueFamilyAffinity, kRecorderVmmReservationSize,
      &virtual_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_allocate(
      allocator, RecorderVmmPhysicalParams(), kRecorderVmmReservationSize,
      iree_allocator_system(), &physical_memory));
  iree_hal_buffer_retain(virtual_buffer);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_allocator_virtual_memory_release(allocator, virtual_buffer));
  EXPECT_NE(nullptr, state.virtual_buffer);
  EXPECT_EQ(1u, state.release_attempt_count);
  IREE_ASSERT_OK(
      iree_hal_allocator_virtual_memory_release(allocator, virtual_buffer));
  EXPECT_EQ(nullptr, state.virtual_buffer);
  EXPECT_EQ(2u, state.release_attempt_count);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_allocator_virtual_memory_release(allocator, virtual_buffer));
  EXPECT_EQ(2u, state.release_attempt_count);
  iree_hal_buffer_release(virtual_buffer);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_hal_allocator_physical_memory_free(allocator, physical_memory));
  EXPECT_NE(nullptr, state.physical_memory);
  EXPECT_EQ(1u, state.physical_free_attempt_count);
  IREE_ASSERT_OK(
      iree_hal_allocator_physical_memory_free(allocator, physical_memory));
  EXPECT_EQ(nullptr, state.physical_memory);
  EXPECT_EQ(2u, state.physical_free_attempt_count);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_allocator_physical_memory_free(allocator, physical_memory));
  EXPECT_EQ(2u, state.physical_free_attempt_count);
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));

  const auto records = ParseOperationRecords(storage);
  iree_hal_replay_object_id_t release_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  iree_hal_replay_object_id_t free_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  iree_host_size_t release_record_count = 0;
  iree_host_size_t free_record_count = 0;
  for (const auto& record : records) {
    if (record.header.operation_code ==
        IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE) {
      iree_hal_replay_allocator_virtual_memory_release_payload_t payload;
      ASSERT_EQ(sizeof(payload), record.payload.data_length);
      memcpy(&payload, record.payload.data, sizeof(payload));
      if (release_id == IREE_HAL_REPLAY_OBJECT_ID_NONE) {
        release_id = payload.virtual_buffer_id;
      }
      EXPECT_EQ(release_id, payload.virtual_buffer_id);
      ++release_record_count;
    } else if (record.header.operation_code ==
               IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_FREE) {
      iree_hal_replay_allocator_physical_memory_free_payload_t payload;
      ASSERT_EQ(sizeof(payload), record.payload.data_length);
      memcpy(&payload, record.payload.data, sizeof(payload));
      if (free_id == IREE_HAL_REPLAY_OBJECT_ID_NONE) {
        free_id = payload.physical_memory_id;
      }
      EXPECT_EQ(free_id, payload.physical_memory_id);
      ++free_record_count;
    }
  }
  EXPECT_EQ(2u, release_record_count);
  EXPECT_EQ(2u, free_record_count);

  iree_hal_allocator_release(allocator);
  iree_hal_allocator_release(base_allocator);
  iree_hal_replay_recorder_release(recorder);
}

TEST(ReplayRecorderVmmTest, AppendFailureDoesNotBlockVmmTeardown) {
  const std::vector<uint8_t> complete_storage =
      CaptureMinimalRecorderVmmLifecycle();
  const iree_host_size_t free_offset = FindCapturedOperationOffset(
      complete_storage,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_FREE);
  ASSERT_NE(0u, free_offset);
  std::vector<uint8_t> storage(free_offset, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);
  RecorderVmmAllocatorState state;
  iree_hal_allocator_t* base_allocator = CreateRecorderVmmAllocator(&state);
  iree_hal_allocator_t* allocator =
      WrapRecorderVmmAllocator(recorder, /*device_id=*/1, base_allocator);

  iree_hal_buffer_t* virtual_buffer = nullptr;
  iree_hal_physical_memory_t* physical_memory = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_reserve(
      allocator, kRecorderVmmQueueFamilyAffinity, kRecorderVmmReservationSize,
      &virtual_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_allocate(
      allocator, RecorderVmmPhysicalParams(), kRecorderVmmReservationSize,
      iree_allocator_system(), &physical_memory));

  IREE_EXPECT_OK(
      iree_hal_allocator_physical_memory_free(allocator, physical_memory));
  EXPECT_EQ(1u, state.physical_free_count);
  EXPECT_EQ(nullptr, state.physical_memory);
  IREE_EXPECT_OK(
      iree_hal_allocator_virtual_memory_release(allocator, virtual_buffer));
  EXPECT_EQ(1u, state.release_count);
  EXPECT_EQ(nullptr, state.virtual_buffer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_replay_recorder_close(recorder));

  iree_hal_allocator_release(allocator);
  iree_hal_allocator_release(base_allocator);
  iree_hal_replay_recorder_release(recorder);
}

}  // namespace
