// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/dump.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/replay/file_reader.h"
#include "iree/hal/replay/file_writer.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using ::testing::HasSubstr;

static iree_status_t AppendToString(void* user_data, iree_string_view_t text) {
  auto* output = static_cast<std::string*>(user_data);
  output->append(text.data, text.size);
  return iree_ok_status();
}

static iree_status_t DumpReplayToString(
    iree_const_byte_span_t file_contents,
    const iree_hal_replay_dump_options_t* options, std::string* output) {
  iree_hal_replay_dump_write_callback_t write_callback = {
      .fn = AppendToString,
      .user_data = output,
  };
  return iree_hal_replay_dump_file(file_contents, options, write_callback,
                                   iree_allocator_system());
}

static std::vector<uint8_t> MakeReplayFileStorage() {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_CHECK_OK(iree_hal_replay_file_writer_allocate(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  iree_hal_replay_file_record_metadata_t session_metadata = {
      .sequence_ordinal = 0,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_SESSION};
  IREE_CHECK_OK(iree_hal_replay_file_writer_append_record(
      writer, &session_metadata, 0, nullptr, nullptr));

  iree_hal_replay_buffer_object_payload_t buffer_payload = {
      .allocation_size = 256, .byte_length = 64, .allowed_usage = 0x11};
  iree_const_byte_span_t buffer_payload_span =
      iree_make_const_byte_span(&buffer_payload, sizeof(buffer_payload));
  iree_hal_replay_file_record_metadata_t object_metadata = {
      .sequence_ordinal = 1,
      .object_id = 7,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER};
  IREE_CHECK_OK(iree_hal_replay_file_writer_append_record(
      writer, &object_metadata, 1, &buffer_payload_span, nullptr));

  IREE_CHECK_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);
  return storage;
}

static iree_const_byte_span_t MakeReplayFileContents(
    const std::vector<uint8_t>& storage) {
  auto* file_header =
      reinterpret_cast<const iree_hal_replay_file_header_t*>(storage.data());
  return iree_make_const_byte_span(
      storage.data(), static_cast<iree_host_size_t>(file_header->file_length));
}

class ReplayFileBuilder {
 public:
  explicit ReplayFileBuilder(iree_host_size_t capacity)
      : storage_(capacity, 0) {
    iree_io_file_handle_t* file_handle = nullptr;
    IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(storage_.data(), storage_.size()),
        iree_io_file_handle_release_callback_null(), iree_allocator_system(),
        &file_handle));
    IREE_CHECK_OK(iree_hal_replay_file_writer_allocate(
        file_handle, iree_allocator_system(), &writer_));
    iree_io_file_handle_release(file_handle);
  }

  ~ReplayFileBuilder() {
    if (writer_) {
      iree_hal_replay_file_writer_free(writer_);
    }
  }

  void Append(const iree_hal_replay_file_record_metadata_t& metadata,
              iree_host_size_t payload_count,
              const iree_const_byte_span_t* payloads) {
    IREE_CHECK_OK(iree_hal_replay_file_writer_append_record(
        writer_, &metadata, payload_count, payloads, nullptr));
  }

  template <typename T>
  void Append(const iree_hal_replay_file_record_metadata_t& metadata,
              const T& payload) {
    iree_const_byte_span_t payload_span =
        iree_make_const_byte_span(&payload, sizeof(payload));
    Append(metadata, 1, &payload_span);
  }

  std::vector<uint8_t> Finish() {
    IREE_CHECK_OK(iree_hal_replay_file_writer_close(writer_));
    iree_hal_replay_file_writer_free(writer_);
    writer_ = nullptr;
    return std::move(storage_);
  }

 private:
  // Mutable file storage retained through writer closure.
  std::vector<uint8_t> storage_;
  // Replay writer targeting storage_.
  iree_hal_replay_file_writer_t* writer_ = nullptr;
};

static iree_hal_replay_file_record_metadata_t MakeAtomicRecordMetadata(
    uint64_t sequence_ordinal, iree_hal_replay_object_type_t object_type,
    iree_hal_replay_payload_type_t payload_type,
    iree_hal_replay_operation_code_t operation_code) {
  iree_hal_replay_file_record_metadata_t metadata = {};
  metadata.sequence_ordinal = sequence_ordinal;
  metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  metadata.object_type = object_type;
  metadata.object_id =
      object_type == IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE ? 200 : 100;
  metadata.payload_type = payload_type;
  metadata.operation_code = operation_code;
  return metadata;
}

template <typename T>
static void AppendQueueAtomicRecord(
    ReplayFileBuilder* builder,
    const iree_hal_replay_file_record_metadata_t& metadata, const T& payload,
    const iree_hal_replay_semaphore_timepoint_payload_t& wait_timepoint,
    const iree_hal_replay_semaphore_timepoint_payload_t& signal_timepoint) {
  iree_const_byte_span_t payloads[] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(&wait_timepoint, sizeof(wait_timepoint)),
      iree_make_const_byte_span(&signal_timepoint, sizeof(signal_timepoint)),
  };
  builder->Append(metadata, IREE_ARRAYSIZE(payloads), payloads);
}

static std::vector<uint8_t> MakeAtomicReplayFileStorage() {
  ReplayFileBuilder builder(/*capacity=*/16384);

  iree_hal_replay_command_buffer_atomic_wait_payload_t command_wait = {};
  command_wait.target_ref.buffer_id = 7;
  command_wait.target_ref.offset = 8;
  command_wait.target_ref.length = 8;
  command_wait.source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH;
  command_wait.target_stage_mask = IREE_HAL_EXECUTION_STAGE_ATOMIC;
  command_wait.params.value = 17;
  command_wait.params.mask = 255;
  command_wait.params.flags = IREE_HAL_ATOMIC_FLAGS_KNOWN;
  command_wait.params.width = IREE_HAL_ATOMIC_WIDTH_64;
  command_wait.params.condition = IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL;
  command_wait.params.target_error_mode =
      IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE;
  builder.Append(MakeAtomicRecordMetadata(
                     0, IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER,
                     IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_WAIT,
                     IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_WAIT),
                 command_wait);

  iree_hal_replay_command_buffer_atomic_store_payload_t command_store = {};
  command_store.target_ref.offset = 16;
  command_store.target_ref.length = 4;
  command_store.target_ref.buffer_slot = 3;
  command_store.source_stage_mask = IREE_HAL_EXECUTION_STAGE_ATOMIC;
  command_store.target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH;
  command_store.params.value = 34;
  command_store.params.flags = IREE_HAL_ATOMIC_FLAGS_KNOWN;
  command_store.params.width = IREE_HAL_ATOMIC_WIDTH_32;
  command_store.params.target_error_mode =
      IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE;
  builder.Append(
      MakeAtomicRecordMetadata(
          1, IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER,
          IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_STORE,
          IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_STORE),
      command_store);

  iree_hal_replay_command_buffer_atomic_rmw_payload_t command_rmw = {};
  command_rmw.target_ref.buffer_id = 8;
  command_rmw.target_ref.offset = 24;
  command_rmw.target_ref.length = 8;
  command_rmw.source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH;
  command_rmw.target_stage_mask = IREE_HAL_EXECUTION_STAGE_ATOMIC;
  command_rmw.params.operand = 51;
  command_rmw.params.flags = IREE_HAL_ATOMIC_FLAGS_KNOWN;
  command_rmw.params.width = IREE_HAL_ATOMIC_WIDTH_64;
  command_rmw.params.operation = IREE_HAL_ATOMIC_RMW_OPERATION_XOR;
  command_rmw.params.target_error_mode =
      IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE;
  builder.Append(MakeAtomicRecordMetadata(
                     2, IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER,
                     IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_RMW,
                     IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_RMW),
                 command_rmw);

  iree_hal_replay_queue_atomic_wait_payload_t queue_wait = {};
  queue_wait.target_ref.buffer_id = 10;
  queue_wait.target_ref.offset = 4;
  queue_wait.target_ref.length = 4;
  queue_wait.wait_semaphore_count = 1;
  queue_wait.signal_semaphore_count = 1;
  queue_wait.params.value = 68;
  queue_wait.params.mask = 255;
  queue_wait.params.flags = IREE_HAL_ATOMIC_FLAGS_KNOWN;
  queue_wait.params.width = IREE_HAL_ATOMIC_WIDTH_32;
  queue_wait.params.condition =
      IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL;
  queue_wait.params.target_error_mode =
      IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE;
  iree_hal_replay_semaphore_timepoint_payload_t queue_wait_wait = {
      .semaphore_id = 41, .value = 5};
  iree_hal_replay_semaphore_timepoint_payload_t queue_wait_signal = {
      .semaphore_id = 51, .value = 6};
  AppendQueueAtomicRecord(&builder,
                          MakeAtomicRecordMetadata(
                              3, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
                              IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_WAIT,
                              IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_ATOMIC_WAIT),
                          queue_wait, queue_wait_wait, queue_wait_signal);

  iree_hal_replay_queue_atomic_store_payload_t queue_store = {};
  queue_store.target_ref.buffer_id = 11;
  queue_store.target_ref.offset = 8;
  queue_store.target_ref.length = 8;
  queue_store.wait_semaphore_count = 1;
  queue_store.signal_semaphore_count = 1;
  queue_store.params.value = 85;
  queue_store.params.flags = IREE_HAL_ATOMIC_FLAGS_KNOWN;
  queue_store.params.width = IREE_HAL_ATOMIC_WIDTH_64;
  queue_store.params.target_error_mode =
      IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE;
  iree_hal_replay_semaphore_timepoint_payload_t queue_store_wait = {
      .semaphore_id = 42, .value = 7};
  iree_hal_replay_semaphore_timepoint_payload_t queue_store_signal = {
      .semaphore_id = 52, .value = 8};
  AppendQueueAtomicRecord(
      &builder,
      MakeAtomicRecordMetadata(
          4, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
          IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_STORE,
          IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_ATOMIC_STORE),
      queue_store, queue_store_wait, queue_store_signal);

  iree_hal_replay_queue_atomic_rmw_payload_t queue_rmw = {};
  queue_rmw.target_ref.buffer_id = 12;
  queue_rmw.target_ref.offset = 16;
  queue_rmw.target_ref.length = 8;
  queue_rmw.wait_semaphore_count = 1;
  queue_rmw.signal_semaphore_count = 1;
  queue_rmw.params.operand = 102;
  queue_rmw.params.flags = IREE_HAL_ATOMIC_FLAGS_KNOWN;
  queue_rmw.params.width = IREE_HAL_ATOMIC_WIDTH_64;
  queue_rmw.params.operation = IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT;
  queue_rmw.params.target_error_mode =
      IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE;
  iree_hal_replay_semaphore_timepoint_payload_t queue_rmw_wait = {
      .semaphore_id = 43, .value = 9};
  iree_hal_replay_semaphore_timepoint_payload_t queue_rmw_signal = {
      .semaphore_id = 53, .value = 10};
  AppendQueueAtomicRecord(
      &builder,
      MakeAtomicRecordMetadata(5, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
                               IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_RMW,
                               IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_ATOMIC_RMW),
      queue_rmw, queue_rmw_wait, queue_rmw_signal);

  return builder.Finish();
}

static std::vector<size_t> FindAtomicTargetErrorModeOffsets(
    const std::vector<uint8_t>& storage) {
  const iree_const_byte_span_t file_contents = MakeReplayFileContents(storage);
  iree_hal_replay_file_header_t header;
  iree_host_size_t offset = 0;
  IREE_CHECK_OK(
      iree_hal_replay_file_parse_header(file_contents, &header, &offset));
  std::vector<size_t> offsets;
  while (offset < file_contents.data_length) {
    iree_hal_replay_file_record_t record;
    IREE_CHECK_OK(iree_hal_replay_file_parse_record(file_contents, offset,
                                                    &record, &offset));
    const size_t payload_offset =
        static_cast<size_t>(record.payload.data - file_contents.data);
    switch (record.header.payload_type) {
      case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_WAIT:
        offsets.push_back(
            payload_offset +
            offsetof(iree_hal_replay_command_buffer_atomic_wait_payload_t,
                     params) +
            offsetof(iree_hal_replay_atomic_wait_params_payload_t,
                     target_error_mode));
        break;
      case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_STORE:
        offsets.push_back(
            payload_offset +
            offsetof(iree_hal_replay_command_buffer_atomic_store_payload_t,
                     params) +
            offsetof(iree_hal_replay_atomic_store_params_payload_t,
                     target_error_mode));
        break;
      case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_RMW:
        offsets.push_back(
            payload_offset +
            offsetof(iree_hal_replay_command_buffer_atomic_rmw_payload_t,
                     params) +
            offsetof(iree_hal_replay_atomic_rmw_params_payload_t,
                     target_error_mode));
        break;
      case IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_WAIT:
        offsets.push_back(
            payload_offset +
            offsetof(iree_hal_replay_queue_atomic_wait_payload_t, params) +
            offsetof(iree_hal_replay_atomic_wait_params_payload_t,
                     target_error_mode));
        break;
      case IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_STORE:
        offsets.push_back(
            payload_offset +
            offsetof(iree_hal_replay_queue_atomic_store_payload_t, params) +
            offsetof(iree_hal_replay_atomic_store_params_payload_t,
                     target_error_mode));
        break;
      case IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_RMW:
        offsets.push_back(
            payload_offset +
            offsetof(iree_hal_replay_queue_atomic_rmw_payload_t, params) +
            offsetof(iree_hal_replay_atomic_rmw_params_payload_t,
                     target_error_mode));
        break;
    }
  }
  return offsets;
}

static size_t CountOccurrences(const std::string& text,
                               const std::string& needle) {
  size_t count = 0;
  size_t offset = 0;
  while ((offset = text.find(needle, offset)) != std::string::npos) {
    ++count;
    offset += needle.size();
  }
  return count;
}

static std::vector<uint8_t> MakeScopeReplayFileStorage() {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_CHECK_OK(iree_hal_replay_file_writer_allocate(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  iree_hal_replay_file_record_metadata_t session_metadata = {
      .sequence_ordinal = 0,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_SESSION};
  IREE_CHECK_OK(iree_hal_replay_file_writer_append_record(
      writer, &session_metadata, 0, nullptr, nullptr));

  const char scope_name[] = "execute";
  iree_hal_replay_scope_payload_t payload = {.name_length =
                                                 sizeof(scope_name) - 1};
  iree_const_byte_span_t iovecs[] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(scope_name, sizeof(scope_name) - 1),
  };
  iree_hal_replay_file_record_metadata_t begin_metadata = {
      .sequence_ordinal = 1,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_REPLAY_SCOPE,
      .operation_code = IREE_HAL_REPLAY_OPERATION_CODE_REPLAY_SCOPE_BEGIN};
  IREE_CHECK_OK(iree_hal_replay_file_writer_append_record(
      writer, &begin_metadata, IREE_ARRAYSIZE(iovecs), iovecs, nullptr));

  iree_hal_replay_file_record_metadata_t end_metadata = begin_metadata;
  end_metadata.sequence_ordinal = 2;
  end_metadata.operation_code = IREE_HAL_REPLAY_OPERATION_CODE_REPLAY_SCOPE_END;
  IREE_CHECK_OK(iree_hal_replay_file_writer_append_record(
      writer, &end_metadata, IREE_ARRAYSIZE(iovecs), iovecs, nullptr));

  IREE_CHECK_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);
  return storage;
}

static std::vector<uint8_t> MakeExecutableLoadReplayFileStorage() {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_CHECK_OK(iree_hal_replay_file_writer_allocate(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  const char target_family[] = "mock";
  const char target_key[] = "metadata";
  const uint8_t executable_data[] = {0x00, 0x01, 0x02, 0x03};
  const uint32_t constants[] = {0xABCD1234u};
  iree_hal_replay_executable_metadata_header_t metadata_header = {};
  metadata_header.function_count = 1;
  const char function_name[] = "main";
  metadata_header.function_name_storage_length = sizeof(function_name) - 1;
  iree_hal_replay_executable_function_metadata_t function_metadata = {};
  function_metadata.binding_count = 2;
  function_metadata.workgroup_size[0] = 3;
  function_metadata.workgroup_size[1] = 1;
  function_metadata.workgroup_size[2] = 1;
  function_metadata.name_length = sizeof(function_name) - 1;
  iree_hal_replay_executable_load_payload_t payload = {
      .queue_family_ordinal = 0,
      .target_physical_device_affinity = 1,
      .executable_data_length = sizeof(executable_data),
      .constant_count = IREE_ARRAYSIZE(constants),
      .load_flags = IREE_HAL_EXECUTABLE_LOAD_FLAG_ENABLE_DEBUGGING,
      .target_kind = IREE_HAL_EXECUTABLE_TARGET_KIND_VIRTUAL,
      .target_family_length = sizeof(target_family) - 1,
      .target_key_length = sizeof(target_key) - 1,
      .executable_metadata_length = sizeof(metadata_header) +
                                    sizeof(function_metadata) +
                                    sizeof(function_name) - 1};
  iree_const_byte_span_t iovecs[] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(target_family, sizeof(target_family) - 1),
      iree_make_const_byte_span(target_key, sizeof(target_key) - 1),
      iree_make_const_byte_span(executable_data, sizeof(executable_data)),
      iree_make_const_byte_span(constants, sizeof(constants)),
      iree_make_const_byte_span(&metadata_header, sizeof(metadata_header)),
      iree_make_const_byte_span(&function_metadata, sizeof(function_metadata)),
      iree_make_const_byte_span(function_name, sizeof(function_name) - 1),
  };
  iree_hal_replay_file_record_metadata_t metadata = {};
  metadata.sequence_ordinal = 0;
  metadata.device_id = 1;
  metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_EXECUTABLE_LOAD;
  metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE;
  metadata.object_id = 1;
  metadata.related_object_id = 2;
  metadata.operation_code =
      IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_LOAD_EXECUTABLE;
  IREE_CHECK_OK(iree_hal_replay_file_writer_append_record(
      writer, &metadata, IREE_ARRAYSIZE(iovecs), iovecs, nullptr));

  IREE_CHECK_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);
  return storage;
}

static iree_hal_replay_file_record_metadata_t MakeAllocatorOperationMetadata(
    uint64_t sequence_ordinal, iree_hal_replay_payload_type_t payload_type,
    iree_hal_replay_operation_code_t operation_code,
    iree_hal_replay_object_id_t related_object_id) {
  iree_hal_replay_file_record_metadata_t metadata = {
      .sequence_ordinal = sequence_ordinal,
      .device_id = 1,
      .object_id = 2,
      .related_object_id = related_object_id,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .payload_type = payload_type,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR,
      .operation_code = operation_code};
  return metadata;
}

static std::vector<uint8_t> MakeVmmReplayFileStorage() {
  ReplayFileBuilder builder(/*capacity=*/8192);

  iree_hal_replay_file_record_metadata_t physical_object_metadata = {
      .sequence_ordinal = 0,
      .object_id = 23,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_PHYSICAL_MEMORY};
  builder.Append(physical_object_metadata, 0, nullptr);

  iree_hal_replay_allocator_virtual_memory_reserve_payload_t reserve = {
      .queue_family_affinity = 3, .size = 4096};
  builder.Append(
      MakeAllocatorOperationMetadata(
          1, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_RESERVE,
          IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RESERVE, 17),
      reserve);

  iree_hal_replay_allocator_virtual_memory_release_payload_t release = {
      .virtual_buffer_id = 17};
  builder.Append(
      MakeAllocatorOperationMetadata(
          2, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE,
          IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE, 17),
      release);

  iree_hal_replay_allocator_physical_memory_allocate_payload_t allocate = {};
  allocate.allocation.allocation_size = 8192;
  allocate.allocation.queue_family_affinity = 5;
  allocate.allocation.min_alignment = 4096;
  allocate.allocation.usage = 0x10;
  allocate.allocation.type = 0x20;
  allocate.allocation.access = 0x3;
  builder.Append(
      MakeAllocatorOperationMetadata(
          3, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE,
          IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE,
          23),
      allocate);

  iree_hal_replay_allocator_physical_memory_free_payload_t free = {
      .physical_memory_id = 23};
  builder.Append(
      MakeAllocatorOperationMetadata(
          4, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_PHYSICAL_MEMORY_FREE,
          IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_FREE, 23),
      free);

  iree_hal_replay_allocator_virtual_memory_map_payload_t map = {
      .virtual_buffer_id = 17,
      .physical_memory_id = 23,
      .virtual_offset = 128,
      .physical_offset = 256,
      .size = 1024};
  builder.Append(
      MakeAllocatorOperationMetadata(
          5, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_MAP,
          IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_MAP, 17),
      map);

  iree_hal_replay_allocator_virtual_memory_unmap_payload_t unmap = {
      .virtual_buffer_id = 17, .virtual_offset = 128, .size = 1024};
  builder.Append(
      MakeAllocatorOperationMetadata(
          6, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP,
          IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP, 17),
      unmap);

  iree_hal_replay_allocator_virtual_memory_protect_payload_t protect = {
      .virtual_buffer_id = 17,
      .virtual_offset = 512,
      .size = 2048,
      .queue_family_affinity = 9,
      .access_scope = 3,
      .protection = 5};
  builder.Append(
      MakeAllocatorOperationMetadata(
          7, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT,
          IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT, 17),
      protect);

  iree_hal_replay_allocator_virtual_memory_advise_payload_t advise = {
      .virtual_buffer_id = 17,
      .virtual_offset = 768,
      .size = 512,
      .queue_family_affinity = 11,
      .advice = 6};
  builder.Append(
      MakeAllocatorOperationMetadata(
          8, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_ADVISE,
          IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_ADVISE, 17),
      advise);

  return builder.Finish();
}

TEST(ReplayDumpTest, EmitsTextSummary) {
  std::vector<uint8_t> storage = MakeReplayFileStorage();
  iree_hal_replay_dump_options_t options =
      iree_hal_replay_dump_options_default();

  std::string output;
  IREE_ASSERT_OK(
      DumpReplayToString(MakeReplayFileContents(storage), &options, &output));

  EXPECT_THAT(output, HasSubstr("summary:"));
  EXPECT_THAT(output, HasSubstr("hermetic: yes"));
  EXPECT_THAT(output, HasSubstr("strict_replay_supported: yes"));
  EXPECT_THAT(output, HasSubstr("files: total=0 external=0 inline=0"));
  EXPECT_THAT(output, HasSubstr("#0 session"));
  EXPECT_THAT(output, HasSubstr("#1 object"));
  EXPECT_THAT(output, HasSubstr("object=buffer"));
  EXPECT_THAT(output, HasSubstr("payload=buffer_object"));
  EXPECT_THAT(output, HasSubstr("allocation_size=256"));
}

TEST(ReplayDumpTest, EmitsJsonlWithPayloadRanges) {
  std::vector<uint8_t> storage = MakeReplayFileStorage();
  iree_hal_replay_dump_options_t options =
      iree_hal_replay_dump_options_default();
  options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;

  std::string output;
  IREE_ASSERT_OK(
      DumpReplayToString(MakeReplayFileContents(storage), &options, &output));

  EXPECT_THAT(output, HasSubstr("\"kind\":\"file\""));
  EXPECT_THAT(output, HasSubstr("\"kind\":\"summary\""));
  EXPECT_THAT(output, HasSubstr("\"hermetic\":true"));
  EXPECT_THAT(output, HasSubstr("\"environment_referenced\":false"));
  EXPECT_THAT(output, HasSubstr("\"kind\":\"session\""));
  EXPECT_THAT(output, HasSubstr("\"kind\":\"object\""));
  EXPECT_THAT(output, HasSubstr("\"payload_type\":\"buffer_object\""));
  EXPECT_THAT(output, HasSubstr("\"payload_range\""));
  EXPECT_THAT(output, HasSubstr("\"allocation_size\":256"));
}

TEST(ReplayDumpTest, DefinesV71VmmWireSchema) {
  EXPECT_EQ(9u, IREE_HAL_REPLAY_OBJECT_TYPE_PHYSICAL_MEMORY);
  EXPECT_STREQ("physical_memory",
               iree_hal_replay_object_type_string(
                   IREE_HAL_REPLAY_OBJECT_TYPE_PHYSICAL_MEMORY));

  struct NamedPayloadType {
    iree_hal_replay_payload_type_t type;
    uint32_t value;
    const char* name;
  };
  const NamedPayloadType vmm_payload_types[] = {
      {IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_RESERVE, 46,
       "allocator_virtual_memory_reserve"},
      {IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE, 47,
       "allocator_virtual_memory_release"},
      {IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE, 48,
       "allocator_physical_memory_allocate"},
      {IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_PHYSICAL_MEMORY_FREE, 49,
       "allocator_physical_memory_free"},
      {IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_MAP, 50,
       "allocator_virtual_memory_map"},
      {IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP, 51,
       "allocator_virtual_memory_unmap"},
      {IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT, 52,
       "allocator_virtual_memory_protect"},
      {IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_ADVISE, 53,
       "allocator_virtual_memory_advise"},
  };
  for (const NamedPayloadType& payload_type : vmm_payload_types) {
    EXPECT_EQ(payload_type.value, payload_type.type);
    EXPECT_STREQ(payload_type.name,
                 iree_hal_replay_payload_type_string(payload_type.type));
  }

  const iree_hal_replay_payload_type_t exact_queue_payload_types[] = {
      IREE_HAL_REPLAY_PAYLOAD_TYPE_PROVISIONED_QUEUE_OBJECT,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_TRANSFER,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_READ,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_WRITE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ALLOCA,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_DEALLOCA,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_BARRIER,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_EXECUTE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_FAMILY_COMMAND_BUFFER_OBJECT,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_WAIT,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_STORE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_RMW,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_TIMESTAMP,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(exact_queue_payload_types);
       ++i) {
    EXPECT_EQ(33u + i, exact_queue_payload_types[i]);
  }

  const iree_hal_replay_operation_code_t vmm_operation_codes[] = {
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RESERVE,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_FREE,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_MAP,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_ADVISE,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(vmm_operation_codes); ++i) {
    EXPECT_EQ(106u + i, vmm_operation_codes[i]);
  }

  EXPECT_EQ(16u,
            sizeof(iree_hal_replay_allocator_virtual_memory_reserve_payload_t));
  EXPECT_EQ(8u,
            sizeof(iree_hal_replay_allocator_virtual_memory_release_payload_t));
  EXPECT_EQ(
      40u,
      sizeof(iree_hal_replay_allocator_physical_memory_allocate_payload_t));
  EXPECT_EQ(8u,
            sizeof(iree_hal_replay_allocator_physical_memory_free_payload_t));
  EXPECT_EQ(40u,
            sizeof(iree_hal_replay_allocator_virtual_memory_map_payload_t));
  EXPECT_EQ(24u,
            sizeof(iree_hal_replay_allocator_virtual_memory_unmap_payload_t));
  EXPECT_EQ(48u,
            sizeof(iree_hal_replay_allocator_virtual_memory_protect_payload_t));
  EXPECT_EQ(40u,
            sizeof(iree_hal_replay_allocator_virtual_memory_advise_payload_t));
  EXPECT_EQ(32u,
            offsetof(iree_hal_replay_allocator_virtual_memory_protect_payload_t,
                     access_scope));
  EXPECT_EQ(36u,
            offsetof(iree_hal_replay_allocator_virtual_memory_protect_payload_t,
                     reserved0));
  EXPECT_EQ(40u,
            offsetof(iree_hal_replay_allocator_virtual_memory_protect_payload_t,
                     protection));
}

TEST(ReplayDumpTest, EmitsVmmPayloads) {
  std::vector<uint8_t> storage = MakeVmmReplayFileStorage();
  iree_hal_replay_dump_options_t options =
      iree_hal_replay_dump_options_default();

  std::string text_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage), &options,
                                    &text_output));
  EXPECT_THAT(text_output, HasSubstr("object=physical_memory(9)"));
  const char* payload_names[] = {
      "allocator_virtual_memory_reserve",   "allocator_virtual_memory_release",
      "allocator_physical_memory_allocate", "allocator_physical_memory_free",
      "allocator_virtual_memory_map",       "allocator_virtual_memory_unmap",
      "allocator_virtual_memory_protect",   "allocator_virtual_memory_advise",
  };
  for (const char* payload_name : payload_names) {
    EXPECT_THAT(text_output, HasSubstr(std::string("payload=") + payload_name));
  }
  EXPECT_THAT(text_output, HasSubstr("queue_family_affinity=3 size=4096"));
  EXPECT_THAT(text_output, HasSubstr("physical_memory_id=23"));
  EXPECT_THAT(text_output, HasSubstr("virtual_offset=128 physical_offset=256"));
  EXPECT_THAT(text_output,
              HasSubstr("queue_family_affinity=9 access_scope=0x00000003 "
                        "protection=0x0000000000000005"));

  options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string json_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage), &options,
                                    &json_output));
  EXPECT_THAT(json_output, HasSubstr("\"object_type\":\"physical_memory\""));
  for (const char* payload_name : payload_names) {
    EXPECT_THAT(json_output, HasSubstr(std::string("\"payload_type\":\"") +
                                       payload_name + "\""));
  }
  EXPECT_THAT(json_output,
              HasSubstr("\"queue_family_affinity\":9,\"access_scope\":3,"
                        "\"protection\":5"));
}

TEST(ReplayDumpTest, DecodesUnalignedVmmPayloads) {
  std::vector<uint8_t> storage = MakeVmmReplayFileStorage();
  iree_const_byte_span_t contents = MakeReplayFileContents(storage);
  std::vector<uint8_t> unaligned_storage(contents.data_length + 1, 0);
  std::memcpy(unaligned_storage.data() + 1, contents.data,
              contents.data_length);
  iree_const_byte_span_t unaligned_contents = iree_make_const_byte_span(
      unaligned_storage.data() + 1, contents.data_length);

  iree_hal_replay_dump_options_t options =
      iree_hal_replay_dump_options_default();
  std::string text_output;
  IREE_ASSERT_OK(
      DumpReplayToString(unaligned_contents, &options, &text_output));
  EXPECT_THAT(text_output,
              HasSubstr("virtual_offset=512 size=2048 "
                        "queue_family_affinity=9 access_scope=0x00000003"));

  options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string json_output;
  IREE_ASSERT_OK(
      DumpReplayToString(unaligned_contents, &options, &json_output));
  EXPECT_THAT(json_output,
              HasSubstr("\"virtual_offset\":512,\"size\":2048,"
                        "\"queue_family_affinity\":9,\"access_scope\":3"));
}

TEST(ReplayDumpTest, EmitsScopes) {
  std::vector<uint8_t> storage = MakeScopeReplayFileStorage();
  iree_hal_replay_dump_options_t options =
      iree_hal_replay_dump_options_default();

  std::string text_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage), &options,
                                    &text_output));

  EXPECT_THAT(text_output, HasSubstr("scopes: begin=1 end=1"));
  EXPECT_THAT(text_output, HasSubstr("op=replay.scope_begin"));
  EXPECT_THAT(text_output, HasSubstr("payload=replay_scope"));
  EXPECT_THAT(text_output, HasSubstr("name=\"execute\""));

  options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string jsonl_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage), &options,
                                    &jsonl_output));

  EXPECT_THAT(jsonl_output, HasSubstr("\"scope_begin_count\":1"));
  EXPECT_THAT(jsonl_output, HasSubstr("\"scope_end_count\":1"));
  EXPECT_THAT(jsonl_output, HasSubstr("\"operation\":\"replay.scope_begin\""));
  EXPECT_THAT(jsonl_output, HasSubstr("\"payload_type\":\"replay_scope\""));
  EXPECT_THAT(jsonl_output, HasSubstr("\"name\":\"execute\""));
}

TEST(ReplayDumpTest, EmitsExecutableMetadataRanges) {
  std::vector<uint8_t> storage = MakeExecutableLoadReplayFileStorage();
  iree_hal_replay_dump_options_t options =
      iree_hal_replay_dump_options_default();

  std::string text_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage), &options,
                                    &text_output));

  EXPECT_THAT(text_output, HasSubstr("payload=executable_load"));
  EXPECT_THAT(text_output, HasSubstr("family_range=["));
  EXPECT_THAT(text_output, HasSubstr("key_range=["));
  EXPECT_THAT(text_output, HasSubstr("metadata_range=["));
  EXPECT_THAT(text_output, HasSubstr("metadata_functions=1"));
  EXPECT_THAT(text_output, HasSubstr("metadata_parameters=0"));

  options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string jsonl_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage), &options,
                                    &jsonl_output));

  EXPECT_THAT(jsonl_output, HasSubstr("\"executable_metadata_length\":"));
  EXPECT_THAT(jsonl_output, HasSubstr("\"metadata_range\""));
  EXPECT_THAT(jsonl_output, HasSubstr("\"metadata_function_count\":1"));
  EXPECT_THAT(jsonl_output, HasSubstr("\"metadata_parameter_count\":0"));
}

TEST(ReplayDumpTest, EmitsBufferRangeDataRanges) {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_allocate(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  iree_hal_replay_buffer_range_data_payload_t payload = {
      .byte_offset = 64,
      .byte_length = 4,
      .data_length = 4,
      .memory_access = IREE_HAL_MEMORY_ACCESS_WRITE,
      .map_flags = IREE_HAL_BUFFER_MAP_FLAG_DISCARD};
  const uint8_t data[] = {0x01, 0x02, 0x03, 0x04};
  iree_const_byte_span_t iovecs[2] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(data, sizeof(data)),
  };
  iree_hal_replay_file_record_metadata_t metadata = {
      .sequence_ordinal = 0,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE_DATA,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
      .operation_code = IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_FLUSH_RANGE};
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(writer, &metadata, 2,
                                                           iovecs, nullptr));
  IREE_ASSERT_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);

  iree_hal_replay_dump_options_t options =
      iree_hal_replay_dump_options_default();
  options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string output;
  IREE_ASSERT_OK(
      DumpReplayToString(MakeReplayFileContents(storage), &options, &output));

  EXPECT_THAT(output, HasSubstr("\"payload_type\":\"buffer_range_data\""));
  EXPECT_THAT(output, HasSubstr("\"data_range\""));
  EXPECT_THAT(output, HasSubstr("\"length\":4"));
  EXPECT_THAT(output, HasSubstr("\"memory_access\":2"));
  EXPECT_THAT(output, HasSubstr("\"map_flags\":1"));
  options.format = IREE_HAL_REPLAY_DUMP_FORMAT_TEXT;
  output.clear();
  IREE_ASSERT_OK(
      DumpReplayToString(MakeReplayFileContents(storage), &options, &output));
  EXPECT_THAT(output, HasSubstr("memory_access=0x0002 map_flags=0x0001"));
}

TEST(ReplayDumpTest, EmitsQueueAllocaSemaphoreRanges) {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_allocate(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  iree_hal_replay_queue_alloca_payload_t payload = {.signal_semaphore_count = 1,
                                                    .request_count = 1};
  iree_hal_replay_semaphore_timepoint_payload_t signal = {.semaphore_id = 42,
                                                          .value = 7};
  iree_hal_replay_queue_alloca_request_payload_t request = {.buffer_id = 9};
  request.allocation.allocation_size = 4096;
  request.allocation.queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY;
  request.allocation.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  request.allocation.type = IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  request.allocation.access = IREE_HAL_MEMORY_ACCESS_ALL;
  iree_const_byte_span_t iovecs[3] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(&signal, sizeof(signal)),
      iree_make_const_byte_span(&request, sizeof(request)),
  };
  iree_hal_replay_file_record_metadata_t metadata = {
      .sequence_ordinal = 0,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ALLOCA,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
      .operation_code = IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_ALLOCA};
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &metadata, IREE_ARRAYSIZE(iovecs), iovecs, nullptr));
  IREE_ASSERT_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);

  iree_hal_replay_dump_options_t text_options =
      iree_hal_replay_dump_options_default();
  std::string text_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage),
                                    &text_options, &text_output));
  EXPECT_THAT(text_output, HasSubstr("payload=queue_alloca"));
  EXPECT_THAT(text_output, HasSubstr("request_count=1"));
  EXPECT_THAT(text_output, HasSubstr("wait_range="));
  EXPECT_THAT(text_output, HasSubstr("signal_range="));
  EXPECT_THAT(text_output, HasSubstr("request_range="));

  iree_hal_replay_dump_options_t json_options =
      iree_hal_replay_dump_options_default();
  json_options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string json_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage),
                                    &json_options, &json_output));
  EXPECT_THAT(json_output, HasSubstr("\"payload_type\":\"queue_alloca\""));
  EXPECT_THAT(json_output, HasSubstr("\"wait_semaphores_range\""));
  EXPECT_THAT(json_output, HasSubstr("\"signal_semaphores_range\""));
  EXPECT_THAT(json_output, HasSubstr("\"requests_range\""));
}

TEST(ReplayDumpTest, EmitsQueueSubmissionPayloads) {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_allocate(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  iree_hal_replay_queue_execute_payload_t payload = {
      .command_buffer_id = 9,
      .wait_semaphore_count = 1,
      .signal_semaphore_count = 1,
      .binding_count = 1};
  iree_hal_replay_semaphore_timepoint_payload_t wait = {.semaphore_id = 42,
                                                        .value = 1};
  iree_hal_replay_semaphore_timepoint_payload_t signal = {.semaphore_id = 43,
                                                          .value = 2};
  iree_hal_replay_buffer_ref_payload_t binding = {
      .buffer_id = 7, .offset = 64, .length = 128};
  iree_const_byte_span_t iovecs[4] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(&wait, sizeof(wait)),
      iree_make_const_byte_span(&signal, sizeof(signal)),
      iree_make_const_byte_span(&binding, sizeof(binding)),
  };
  iree_hal_replay_file_record_metadata_t metadata = {};
  metadata.sequence_ordinal = 0;
  metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_BARRIER;
  metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE;
  metadata.operation_code = IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_BARRIER;
  iree_hal_replay_queue_barrier_payload_t barrier_payload = {
      .wait_semaphore_count = 1, .signal_semaphore_count = 1};
  iree_const_byte_span_t barrier_iovecs[] = {
      iree_make_const_byte_span(&barrier_payload, sizeof(barrier_payload)),
      iree_make_const_byte_span(&wait, sizeof(wait)),
      iree_make_const_byte_span(&signal, sizeof(signal)),
  };
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &metadata, IREE_ARRAYSIZE(barrier_iovecs), barrier_iovecs,
      nullptr));

  metadata = {};
  metadata.sequence_ordinal = 1;
  metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_EXECUTE;
  metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE;
  metadata.operation_code = IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_EXECUTE;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &metadata, IREE_ARRAYSIZE(iovecs), iovecs, nullptr));
  IREE_ASSERT_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);

  iree_hal_replay_dump_options_t text_options =
      iree_hal_replay_dump_options_default();
  std::string text_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage),
                                    &text_options, &text_output));
  EXPECT_THAT(text_output, HasSubstr("payload=queue_barrier"));
  EXPECT_THAT(text_output, HasSubstr("payload=queue_execute"));
  EXPECT_THAT(text_output, HasSubstr("wait_semaphores=[{semaphore_id=42"));
  EXPECT_THAT(text_output, HasSubstr("signal_semaphores=[{semaphore_id=43"));
  EXPECT_THAT(text_output, HasSubstr("bindings=[{buffer_id=7"));

  iree_hal_replay_dump_options_t json_options =
      iree_hal_replay_dump_options_default();
  json_options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string json_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage),
                                    &json_options, &json_output));
  EXPECT_THAT(json_output, HasSubstr("\"payload_type\":\"queue_barrier\""));
  EXPECT_THAT(json_output, HasSubstr("\"payload_type\":\"queue_execute\""));
  EXPECT_THAT(json_output,
              HasSubstr("\"wait_semaphores\":[{\"semaphore_id\":42"));
  EXPECT_THAT(json_output,
              HasSubstr("\"signal_semaphores\":[{\"semaphore_id\":43"));
  EXPECT_THAT(json_output,
              HasSubstr("\"bindings\":[{\"buffer_id\":7,\"offset\":64"));
}

TEST(ReplayDumpTest, EmitsExecutionBarrierRanges) {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_allocate(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  iree_hal_replay_command_buffer_execution_barrier_payload_t payload = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .memory_barrier_count = 1};
  iree_hal_replay_memory_barrier_payload_t memory_barrier = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_READ};
  iree_const_byte_span_t iovecs[2] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(&memory_barrier, sizeof(memory_barrier)),
  };
  iree_hal_replay_file_record_metadata_t metadata = {
      .sequence_ordinal = 0,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .payload_type =
          IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_EXECUTION_BARRIER,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER,
      .operation_code =
          IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_EXECUTION_BARRIER};
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(writer, &metadata, 2,
                                                           iovecs, nullptr));
  IREE_ASSERT_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);

  iree_hal_replay_dump_options_t text_options =
      iree_hal_replay_dump_options_default();
  std::string text_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage),
                                    &text_options, &text_output));
  EXPECT_THAT(text_output,
              HasSubstr("payload=command_buffer_execution_barrier"));
  EXPECT_THAT(text_output, HasSubstr("memory_barriers_range="));
  EXPECT_THAT(text_output, HasSubstr("buffer_barriers_range="));

  iree_hal_replay_dump_options_t json_options =
      iree_hal_replay_dump_options_default();
  json_options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string json_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage),
                                    &json_options, &json_output));
  EXPECT_THAT(
      json_output,
      HasSubstr("\"payload_type\":\"command_buffer_execution_barrier\""));
  EXPECT_THAT(json_output, HasSubstr("\"memory_barriers_range\""));
  EXPECT_THAT(json_output, HasSubstr("\"buffer_barriers_range\""));
}

TEST(ReplayDumpTest, EmitsFilePayloads) {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_allocate(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  iree_hal_replay_file_object_payload_t file_payload = {};
  file_payload.queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY;
  file_payload.file_length = 4096;
  file_payload.file_device = 10;
  file_payload.file_inode = 20;
  file_payload.file_mtime_ns = 30;
  file_payload.access = IREE_HAL_MEMORY_ACCESS_READ;
  file_payload.handle_type = IREE_IO_FILE_HANDLE_TYPE_FD;
  file_payload.reference_type =
      IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_EXTERNAL_PATH;
  file_payload.validation_type = IREE_HAL_REPLAY_FILE_VALIDATION_TYPE_IDENTITY;
  file_payload.digest_type = IREE_HAL_REPLAY_DIGEST_TYPE_NONE;
  const char file_reference[] = "/tmp/model.irpa";
  file_payload.reference_length = sizeof(file_reference) - 1;
  iree_const_byte_span_t file_iovecs[2] = {
      iree_make_const_byte_span(&file_payload, sizeof(file_payload)),
      iree_make_const_byte_span(file_reference, sizeof(file_reference) - 1),
  };
  iree_hal_replay_file_record_metadata_t file_metadata = {
      .sequence_ordinal = 0,
      .object_id = 7,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_FILE_OBJECT,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_FILE};
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &file_metadata, IREE_ARRAYSIZE(file_iovecs), file_iovecs,
      nullptr));

  IREE_ASSERT_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);

  iree_hal_replay_dump_options_t text_options =
      iree_hal_replay_dump_options_default();
  std::string text_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage),
                                    &text_options, &text_output));
  EXPECT_THAT(text_output, HasSubstr("payload=file_object"));
  EXPECT_THAT(text_output, HasSubstr("hermetic: no"));
  EXPECT_THAT(text_output, HasSubstr("environment_referenced: yes"));
  EXPECT_THAT(text_output,
              HasSubstr("files: total=1 external=1 inline=0 ranges=0"));
  EXPECT_THAT(
      text_output,
      HasSubstr(
          "file_bytes: external=4096 inline=0 ranges=0 captured_reads=0"));
  EXPECT_THAT(text_output, HasSubstr("file_validation: identity=1"));
  EXPECT_THAT(text_output, HasSubstr("reference_type=external_path(1)"));
  EXPECT_THAT(text_output, HasSubstr("validation_type=identity(1)"));
  EXPECT_THAT(text_output, HasSubstr("reference_range="));

  iree_hal_replay_dump_options_t json_options =
      iree_hal_replay_dump_options_default();
  json_options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string json_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage),
                                    &json_options, &json_output));
  EXPECT_THAT(json_output, HasSubstr("\"payload_type\":\"file_object\""));
  EXPECT_THAT(json_output, HasSubstr("\"kind\":\"summary\""));
  EXPECT_THAT(json_output, HasSubstr("\"hermetic\":false"));
  EXPECT_THAT(json_output, HasSubstr("\"environment_referenced\":true"));
  EXPECT_THAT(json_output, HasSubstr("\"external_file_count\":1"));
  EXPECT_THAT(json_output, HasSubstr("\"range_file_count\":0"));
  EXPECT_THAT(json_output, HasSubstr("\"captured_read_total_length\":0"));
  EXPECT_THAT(json_output, HasSubstr("\"identity\":1"));
  EXPECT_THAT(json_output,
              HasSubstr("\"reference_type_name\":\"external_path\""));
  EXPECT_THAT(json_output, HasSubstr("\"validation_type_name\":\"identity\""));
  EXPECT_THAT(json_output, HasSubstr("\"reference_range\""));
}

TEST(ReplayDumpTest, EmitsQueueObjectsAndTransferRanges) {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_allocate(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  iree_hal_replay_provisioned_queue_object_payload_t queue_payload = {
      .family_ordinal = 2, .queue_ordinal = 1};
  iree_const_byte_span_t queue_iovec =
      iree_make_const_byte_span(&queue_payload, sizeof(queue_payload));
  iree_hal_replay_file_record_metadata_t metadata = {};
  metadata.sequence_ordinal = 0;
  metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT;
  metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_PROVISIONED_QUEUE_OBJECT;
  metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE;
  metadata.device_id = 4;
  metadata.object_id = 9;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &metadata, 1, &queue_iovec, nullptr));

  iree_hal_replay_dynamic_queue_object_payload_t dynamic_queue_payload = {
      .family_ordinal = 3,
      .priority = -2,
      .features = IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH,
      .execution_resource_count = 2};
  const uint32_t execution_resources[] = {1, 4};
  const iree_const_byte_span_t dynamic_queue_iovecs[] = {
      iree_make_const_byte_span(&dynamic_queue_payload,
                                sizeof(dynamic_queue_payload)),
      iree_make_const_byte_span(execution_resources,
                                sizeof(execution_resources)),
  };
  metadata.sequence_ordinal = 1;
  metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_DYNAMIC_QUEUE_OBJECT;
  metadata.object_id = 10;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &metadata, IREE_ARRAYSIZE(dynamic_queue_iovecs),
      dynamic_queue_iovecs, nullptr));

  iree_hal_replay_queue_transfer_payload_t payload = {
      .wait_semaphore_count = 1,
      .signal_semaphore_count = 1,
      .operation_count = 1,
      .data_length = 4};
  iree_hal_replay_semaphore_timepoint_payload_t wait = {.semaphore_id = 42,
                                                        .value = 1};
  iree_hal_replay_semaphore_timepoint_payload_t signal = {.semaphore_id = 43,
                                                          .value = 2};
  iree_hal_replay_queue_transfer_operation_payload_t operation = {};
  operation.type = IREE_HAL_REPLAY_QUEUE_TRANSFER_OPERATION_TYPE_UPDATE;
  operation.target_ref.buffer_id = 7;
  operation.target_ref.length = 4;
  operation.data_length = 4;
  const uint8_t data[] = {0x10, 0x11, 0x12, 0x13};
  iree_const_byte_span_t iovecs[5] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(&wait, sizeof(wait)),
      iree_make_const_byte_span(&signal, sizeof(signal)),
      iree_make_const_byte_span(&operation, sizeof(operation)),
      iree_make_const_byte_span(data, sizeof(data)),
  };
  metadata = {};
  metadata.sequence_ordinal = 2;
  metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_TRANSFER;
  metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE;
  metadata.device_id = 4;
  metadata.object_id = 9;
  metadata.operation_code = IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_TRANSFER;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &metadata, IREE_ARRAYSIZE(iovecs), iovecs, nullptr));

  iree_hal_replay_queue_read_payload_t read_payload = {};
  read_payload.source_file_id = 8;
  read_payload.source_offset = 64;
  read_payload.target_ref.buffer_id = 7;
  read_payload.target_ref.length = sizeof(data);
  read_payload.captured_data_length = sizeof(data);
  read_payload.wait_semaphore_count = 1;
  read_payload.signal_semaphore_count = 1;
  iree_const_byte_span_t read_iovecs[] = {
      iree_make_const_byte_span(&read_payload, sizeof(read_payload)),
      iree_make_const_byte_span(&wait, sizeof(wait)),
      iree_make_const_byte_span(&signal, sizeof(signal)),
      iree_make_const_byte_span(data, sizeof(data)),
  };
  metadata.sequence_ordinal = 3;
  metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_READ;
  metadata.operation_code = IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_READ;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &metadata, IREE_ARRAYSIZE(read_iovecs), read_iovecs, nullptr));

  iree_hal_replay_queue_write_payload_t write_payload = {};
  write_payload.source_ref.buffer_id = 7;
  write_payload.source_ref.length = sizeof(data);
  write_payload.target_file_id = 8;
  write_payload.target_offset = 128;
  write_payload.wait_semaphore_count = 1;
  write_payload.signal_semaphore_count = 1;
  iree_const_byte_span_t write_iovecs[] = {
      iree_make_const_byte_span(&write_payload, sizeof(write_payload)),
      iree_make_const_byte_span(&wait, sizeof(wait)),
      iree_make_const_byte_span(&signal, sizeof(signal)),
  };
  metadata.sequence_ordinal = 4;
  metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_WRITE;
  metadata.operation_code = IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_WRITE;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &metadata, IREE_ARRAYSIZE(write_iovecs), write_iovecs, nullptr));
  IREE_ASSERT_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);

  iree_hal_replay_dump_options_t text_options =
      iree_hal_replay_dump_options_default();
  std::string text_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage),
                                    &text_options, &text_output));
  EXPECT_THAT(text_output, HasSubstr("payload=provisioned_queue_object"));
  EXPECT_THAT(text_output, HasSubstr("family_ordinal=2 queue_ordinal=1"));
  EXPECT_THAT(text_output, HasSubstr("payload=dynamic_queue_object"));
  EXPECT_THAT(text_output, HasSubstr("family_ordinal=3 priority=-2"));
  EXPECT_THAT(text_output, HasSubstr("execution_resources=[1,4]"));
  EXPECT_THAT(text_output, HasSubstr("payload=queue_transfer"));
  EXPECT_THAT(text_output, HasSubstr("wait_range="));
  EXPECT_THAT(text_output, HasSubstr("signal_range="));
  EXPECT_THAT(text_output, HasSubstr("operations_range="));
  EXPECT_THAT(text_output, HasSubstr("data_range="));
  EXPECT_THAT(text_output, HasSubstr("type=update(2)"));
  EXPECT_THAT(text_output, HasSubstr("payload=queue_read"));
  EXPECT_THAT(text_output, HasSubstr("captured_data_range="));
  EXPECT_THAT(text_output, HasSubstr("payload=queue_write"));
  EXPECT_THAT(text_output, HasSubstr("target_file_id=8"));

  iree_hal_replay_dump_options_t json_options =
      iree_hal_replay_dump_options_default();
  json_options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string json_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage),
                                    &json_options, &json_output));
  EXPECT_THAT(json_output,
              HasSubstr("\"payload_type\":\"provisioned_queue_object\""));
  EXPECT_THAT(json_output, HasSubstr("\"family_ordinal\":2"));
  EXPECT_THAT(json_output, HasSubstr("\"queue_ordinal\":1"));
  EXPECT_THAT(json_output,
              HasSubstr("\"payload_type\":\"dynamic_queue_object\""));
  EXPECT_THAT(json_output, HasSubstr("\"family_ordinal\":3"));
  EXPECT_THAT(json_output, HasSubstr("\"priority\":-2"));
  EXPECT_THAT(json_output, HasSubstr("\"execution_resources\":[1,4]"));
  EXPECT_THAT(json_output, HasSubstr("\"payload_type\":\"queue_transfer\""));
  EXPECT_THAT(json_output, HasSubstr("\"wait_semaphores_range\""));
  EXPECT_THAT(json_output, HasSubstr("\"signal_semaphores_range\""));
  EXPECT_THAT(json_output, HasSubstr("\"operations_range\""));
  EXPECT_THAT(json_output, HasSubstr("\"data_range\""));
  EXPECT_THAT(json_output, HasSubstr("\"type_name\":\"update\""));
  EXPECT_THAT(json_output, HasSubstr("\"payload_type\":\"queue_read\""));
  EXPECT_THAT(json_output, HasSubstr("\"captured_data_range\""));
  EXPECT_THAT(json_output, HasSubstr("\"payload_type\":\"queue_write\""));
}

TEST(ReplayDumpTest, EmitsCommandBufferTransferRanges) {
  std::vector<uint8_t> storage(4096, 0);
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage.data(), storage.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));

  iree_hal_replay_file_writer_t* writer = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_file_writer_allocate(
      file_handle, iree_allocator_system(), &writer));
  iree_io_file_handle_release(file_handle);

  iree_hal_replay_command_buffer_fill_buffer_payload_t fill_payload = {};
  fill_payload.target_ref.buffer_id = 7;
  fill_payload.target_ref.length = 4;
  fill_payload.pattern_length = 4;
  const uint32_t pattern = 0xA5A5A5A5u;
  iree_const_byte_span_t fill_iovecs[2] = {
      iree_make_const_byte_span(&fill_payload, sizeof(fill_payload)),
      iree_make_const_byte_span(&pattern, sizeof(pattern)),
  };
  iree_hal_replay_file_record_metadata_t fill_metadata = {
      .sequence_ordinal = 0,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_FILL_BUFFER,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER,
      .operation_code =
          IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_FILL_BUFFER};
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &fill_metadata, IREE_ARRAYSIZE(fill_iovecs), fill_iovecs,
      nullptr));

  iree_hal_replay_command_buffer_update_buffer_payload_t update_payload = {};
  update_payload.target_ref.buffer_id = 7;
  update_payload.target_ref.offset = 4;
  update_payload.target_ref.length = 4;
  update_payload.source_offset = 2;
  update_payload.data_length = 4;
  const uint8_t data[] = {0x20, 0x21, 0x22, 0x23};
  iree_const_byte_span_t update_iovecs[2] = {
      iree_make_const_byte_span(&update_payload, sizeof(update_payload)),
      iree_make_const_byte_span(data, sizeof(data)),
  };
  iree_hal_replay_file_record_metadata_t update_metadata = {
      .sequence_ordinal = 1,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_UPDATE_BUFFER,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER,
      .operation_code =
          IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_UPDATE_BUFFER};
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &update_metadata, IREE_ARRAYSIZE(update_iovecs), update_iovecs,
      nullptr));

  IREE_ASSERT_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);

  iree_hal_replay_dump_options_t text_options =
      iree_hal_replay_dump_options_default();
  std::string text_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage),
                                    &text_options, &text_output));
  EXPECT_THAT(text_output, HasSubstr("payload=command_buffer_fill_buffer"));
  EXPECT_THAT(text_output, HasSubstr("payload=command_buffer_update_buffer"));
  EXPECT_THAT(text_output, HasSubstr("pattern_range="));
  EXPECT_THAT(text_output, HasSubstr("data_range="));

  iree_hal_replay_dump_options_t json_options =
      iree_hal_replay_dump_options_default();
  json_options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string json_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage),
                                    &json_options, &json_output));
  EXPECT_THAT(json_output,
              HasSubstr("\"payload_type\":\"command_buffer_fill_buffer\""));
  EXPECT_THAT(json_output,
              HasSubstr("\"payload_type\":\"command_buffer_update_buffer\""));
  EXPECT_THAT(json_output, HasSubstr("\"pattern_range\""));
  EXPECT_THAT(json_output, HasSubstr("\"data_range\""));
}

TEST(ReplayDumpTest, EmitsAtomicOperations) {
  std::vector<uint8_t> storage = MakeAtomicReplayFileStorage();

  iree_hal_replay_dump_options_t text_options =
      iree_hal_replay_dump_options_default();
  std::string text_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage),
                                    &text_options, &text_output));
  EXPECT_THAT(text_output, HasSubstr("payload=command_buffer_atomic_wait"));
  EXPECT_THAT(text_output,
              HasSubstr("value=0x0000000000000011 mask=0x00000000000000ff "
                        "flags=0x00000007 width=64 condition=not_equal(1)"));
  EXPECT_THAT(text_output, HasSubstr("target_error_mode=1"));
  EXPECT_THAT(text_output, HasSubstr("payload=command_buffer_atomic_store"));
  EXPECT_THAT(text_output,
              HasSubstr("value=0x0000000000000022 flags=0x00000007 width=32"));
  EXPECT_THAT(text_output,
              HasSubstr("target_ref={buffer_id=0 offset=16 length=4 slot=3}"));
  EXPECT_THAT(text_output, HasSubstr("payload=command_buffer_atomic_rmw"));
  EXPECT_THAT(text_output,
              HasSubstr("operand=0x0000000000000033 flags=0x00000007 width=64 "
                        "operation=xor(4)"));
  EXPECT_THAT(text_output, HasSubstr("payload=queue_atomic_wait"));
  EXPECT_THAT(text_output,
              HasSubstr("value=0x0000000000000044 mask=0x00000000000000ff "
                        "flags=0x00000007 width=32 "
                        "condition=unsigned_greater_equal(2)"));
  EXPECT_THAT(text_output,
              HasSubstr("wait_semaphores=[{semaphore_id=41 value=5}]"));
  EXPECT_THAT(text_output,
              HasSubstr("signal_semaphores=[{semaphore_id=51 value=6}]"));
  EXPECT_THAT(text_output, HasSubstr("payload=queue_atomic_store"));
  EXPECT_THAT(text_output,
              HasSubstr("value=0x0000000000000055 flags=0x00000007 width=64"));
  EXPECT_THAT(text_output, HasSubstr("payload=queue_atomic_rmw"));
  EXPECT_THAT(text_output,
              HasSubstr("operand=0x0000000000000066 flags=0x00000007 width=64 "
                        "operation=subtract(1)"));
  EXPECT_THAT(text_output,
              HasSubstr("wait_semaphores=[{semaphore_id=43 value=9}]"));
  EXPECT_THAT(text_output,
              HasSubstr("signal_semaphores=[{semaphore_id=53 value=10}]"));

  iree_hal_replay_dump_options_t json_options =
      iree_hal_replay_dump_options_default();
  json_options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  std::string json_output;
  IREE_ASSERT_OK(DumpReplayToString(MakeReplayFileContents(storage),
                                    &json_options, &json_output));
  EXPECT_THAT(json_output,
              HasSubstr("\"payload_type\":\"command_buffer_atomic_wait\""));
  EXPECT_THAT(json_output,
              HasSubstr("\"value\":17,\"mask\":255,\"flags\":7,\"width\":64,"));
  EXPECT_THAT(json_output,
              HasSubstr("\"condition\":1,\"condition_name\":\"not_equal\","
                        "\"target_error_mode\":1"));
  EXPECT_THAT(json_output,
              HasSubstr("\"payload_type\":\"command_buffer_atomic_store\""));
  EXPECT_THAT(json_output,
              HasSubstr("\"target_ref\":{\"buffer_id\":0,\"offset\":16,"
                        "\"length\":4,\"buffer_slot\":3}"));
  EXPECT_THAT(json_output, HasSubstr("\"target_error_mode\":1"));
  EXPECT_THAT(json_output,
              HasSubstr("\"payload_type\":\"command_buffer_atomic_rmw\""));
  EXPECT_THAT(json_output,
              HasSubstr("\"operand\":51,\"flags\":7,\"width\":64,"
                        "\"operation\":4,\"operation_name\":\"xor\""));
  EXPECT_THAT(json_output, HasSubstr("\"payload_type\":\"queue_atomic_wait\""));
  EXPECT_THAT(json_output, HasSubstr("\"wait_semaphore_count\":1,"
                                     "\"signal_semaphore_count\":1"));
  EXPECT_THAT(json_output, HasSubstr("\"condition\":2,\"condition_name\":"
                                     "\"unsigned_greater_equal\""));
  EXPECT_THAT(json_output,
              HasSubstr("\"wait_semaphores\":[{\"semaphore_id\":41,"
                        "\"value\":5}]"));
  EXPECT_THAT(json_output,
              HasSubstr("\"signal_semaphores\":[{\"semaphore_id\":51,"
                        "\"value\":6}]"));
  EXPECT_THAT(json_output,
              HasSubstr("\"payload_type\":\"queue_atomic_store\""));
  EXPECT_THAT(json_output, HasSubstr("\"value\":85,\"flags\":7,\"width\":64"));
  EXPECT_THAT(json_output,
              HasSubstr("\"target_ref\":{\"buffer_id\":11,\"offset\":8,"
                        "\"length\":8,\"buffer_slot\":0}"));
  EXPECT_THAT(json_output, HasSubstr("\"payload_type\":\"queue_atomic_rmw\""));
  EXPECT_THAT(json_output,
              HasSubstr("\"operand\":102,\"flags\":7,\"width\":64,"
                        "\"operation\":1,\"operation_name\":\"subtract\""));
  EXPECT_THAT(json_output, HasSubstr("\"wait_semaphores_range\""));
  EXPECT_THAT(json_output, HasSubstr("\"signal_semaphores_range\""));
}

TEST(ReplayDumpTest, EmitsLegacyAtomicDefaultModes) {
  std::vector<uint8_t> storage = MakeAtomicReplayFileStorage();
  auto* header =
      reinterpret_cast<iree_hal_replay_file_header_t*>(storage.data());
  header->version_minor = 2;
  const std::vector<size_t> mode_offsets =
      FindAtomicTargetErrorModeOffsets(storage);
  ASSERT_EQ(6u, mode_offsets.size());
  for (size_t mode_offset : mode_offsets) {
    storage[mode_offset] = 0;
  }

  iree_hal_replay_dump_options_t options =
      iree_hal_replay_dump_options_default();
  std::string output;
  IREE_ASSERT_OK(
      DumpReplayToString(MakeReplayFileContents(storage), &options, &output));
  EXPECT_EQ(6u, CountOccurrences(output, "target_error_mode=0"));

  options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  output.clear();
  IREE_ASSERT_OK(
      DumpReplayToString(MakeReplayFileContents(storage), &options, &output));
  EXPECT_EQ(6u, CountOccurrences(output, "\"target_error_mode\":0"));
}

TEST(ReplayDumpTest, RejectsVersionedAtomicTargetErrorModes) {
  std::vector<uint8_t> base_storage = MakeAtomicReplayFileStorage();
  const std::vector<size_t> mode_offsets =
      FindAtomicTargetErrorModeOffsets(base_storage);
  ASSERT_EQ(6u, mode_offsets.size());
  for (size_t mode_offset : mode_offsets) {
    base_storage[mode_offset] = 0;
  }

  struct InvalidModeCase {
    uint16_t version_minor;
    uint8_t encoded_mode;
    iree_status_code_t expected_status;
  };
  const InvalidModeCase cases[] = {
      {/*version_minor=*/2, /*encoded_mode=*/1, IREE_STATUS_DATA_LOSS},
      {/*version_minor=*/3, /*encoded_mode=*/2, IREE_STATUS_INVALID_ARGUMENT},
  };
  for (const InvalidModeCase& test_case : cases) {
    for (size_t mode_offset : mode_offsets) {
      std::vector<uint8_t> storage = base_storage;
      auto* header =
          reinterpret_cast<iree_hal_replay_file_header_t*>(storage.data());
      header->version_minor = test_case.version_minor;
      storage[mode_offset] = test_case.encoded_mode;

      iree_hal_replay_dump_options_t options =
          iree_hal_replay_dump_options_default();
      std::string output;
      IREE_EXPECT_STATUS_IS(test_case.expected_status,
                            DumpReplayToString(MakeReplayFileContents(storage),
                                               &options, &output));
      options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
      output.clear();
      IREE_EXPECT_STATUS_IS(test_case.expected_status,
                            DumpReplayToString(MakeReplayFileContents(storage),
                                               &options, &output));
    }
  }
}

TEST(ReplayDumpTest, RejectsMalformedVmmPayloadSizes) {
  struct FixedPayloadSchema {
    iree_hal_replay_payload_type_t payload_type;
    iree_hal_replay_operation_code_t operation_code;
    iree_host_size_t payload_size;
  };
  const FixedPayloadSchema schemas[] = {
      {IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_RESERVE,
       IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RESERVE,
       sizeof(iree_hal_replay_allocator_virtual_memory_reserve_payload_t)},
      {IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE,
       IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE,
       sizeof(iree_hal_replay_allocator_virtual_memory_release_payload_t)},
      {IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE,
       IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE,
       sizeof(iree_hal_replay_allocator_physical_memory_allocate_payload_t)},
      {IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_PHYSICAL_MEMORY_FREE,
       IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_FREE,
       sizeof(iree_hal_replay_allocator_physical_memory_free_payload_t)},
      {IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_MAP,
       IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_MAP,
       sizeof(iree_hal_replay_allocator_virtual_memory_map_payload_t)},
      {IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP,
       IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP,
       sizeof(iree_hal_replay_allocator_virtual_memory_unmap_payload_t)},
      {IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT,
       IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT,
       sizeof(iree_hal_replay_allocator_virtual_memory_protect_payload_t)},
      {IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_ADVISE,
       IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_ADVISE,
       sizeof(iree_hal_replay_allocator_virtual_memory_advise_payload_t)},
  };

  for (const FixedPayloadSchema& schema : schemas) {
    const iree_host_size_t malformed_sizes[] = {schema.payload_size - 1,
                                                schema.payload_size + 1};
    for (iree_host_size_t malformed_size : malformed_sizes) {
      SCOPED_TRACE(::testing::Message()
                   << "payload_type=" << schema.payload_type
                   << " payload_size=" << malformed_size);
      ReplayFileBuilder builder(/*capacity=*/4096);
      std::vector<uint8_t> payload(malformed_size, 0);
      iree_const_byte_span_t payload_span =
          iree_make_const_byte_span(payload.data(), payload.size());
      builder.Append(MakeAllocatorOperationMetadata(0, schema.payload_type,
                                                    schema.operation_code, 17),
                     1, &payload_span);
      std::vector<uint8_t> storage = builder.Finish();

      iree_hal_replay_dump_options_t options =
          iree_hal_replay_dump_options_default();
      std::string output;
      IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                            DumpReplayToString(MakeReplayFileContents(storage),
                                               &options, &output));
      options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
      output.clear();
      IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                            DumpReplayToString(MakeReplayFileContents(storage),
                                               &options, &output));
    }
  }
}

TEST(ReplayDumpTest, RejectsReservedVmmPayloadFields) {
  auto expect_rejected = [](iree_hal_replay_payload_type_t payload_type,
                            iree_hal_replay_operation_code_t operation_code,
                            const auto& payload) {
    ReplayFileBuilder builder(/*capacity=*/4096);
    builder.Append(
        MakeAllocatorOperationMetadata(0, payload_type, operation_code, 17),
        payload);
    std::vector<uint8_t> storage = builder.Finish();

    iree_hal_replay_dump_options_t options =
        iree_hal_replay_dump_options_default();
    std::string output;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_DATA_LOSS,
        DumpReplayToString(MakeReplayFileContents(storage), &options, &output));
    options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
    output.clear();
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_DATA_LOSS,
        DumpReplayToString(MakeReplayFileContents(storage), &options, &output));
  };

  iree_hal_replay_allocator_physical_memory_allocate_payload_t allocate = {};
  allocate.allocation.reserved0 = 1;
  expect_rejected(
      IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE,
      allocate);

  allocate.allocation.reserved0 = 0;
  allocate.allocation.reserved1 = 1;
  expect_rejected(
      IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE,
      allocate);

  iree_hal_replay_allocator_virtual_memory_protect_payload_t protect = {
      .reserved0 = 1};
  expect_rejected(
      IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT, protect);
}

TEST(ReplayDumpTest, RejectsMalformedAtomicPayloadLayouts) {
  ReplayFileBuilder command_builder(/*capacity=*/4096);
  iree_hal_replay_command_buffer_atomic_store_payload_t command_payload = {};
  const uint8_t extra_byte = 0;
  iree_const_byte_span_t command_payloads[] = {
      iree_make_const_byte_span(&command_payload, sizeof(command_payload)),
      iree_make_const_byte_span(&extra_byte, sizeof(extra_byte)),
  };
  command_builder.Append(
      MakeAtomicRecordMetadata(
          0, IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER,
          IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_STORE,
          IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_STORE),
      IREE_ARRAYSIZE(command_payloads), command_payloads);
  std::vector<uint8_t> malformed_command_storage = command_builder.Finish();

  iree_hal_replay_dump_options_t options =
      iree_hal_replay_dump_options_default();
  std::string output;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      DumpReplayToString(MakeReplayFileContents(malformed_command_storage),
                         &options, &output));
  options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  output.clear();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      DumpReplayToString(MakeReplayFileContents(malformed_command_storage),
                         &options, &output));

  ReplayFileBuilder queue_builder(/*capacity=*/4096);
  iree_hal_replay_queue_atomic_wait_payload_t queue_payload = {
      .wait_semaphore_count = 1};
  queue_builder.Append(MakeAtomicRecordMetadata(
                           0, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
                           IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_WAIT,
                           IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_ATOMIC_WAIT),
                       queue_payload);
  std::vector<uint8_t> malformed_queue_storage = queue_builder.Finish();

  options.format = IREE_HAL_REPLAY_DUMP_FORMAT_TEXT;
  output.clear();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      DumpReplayToString(MakeReplayFileContents(malformed_queue_storage),
                         &options, &output));
  options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  output.clear();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      DumpReplayToString(MakeReplayFileContents(malformed_queue_storage),
                         &options, &output));
}

TEST(ReplayDumpTest, QueueBarriersDistinguishDefaultAndExplicitEmpty) {
  ReplayFileBuilder builder(4096);
  iree_hal_replay_file_record_metadata_t metadata = {};
  metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE;
  metadata.operation_code = IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_BARRIER;
  metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_BARRIER;
  metadata.record_flags = IREE_HAL_REPLAY_FILE_RECORD_FLAG_QUEUE_BARRIERS;
  const iree_hal_replay_queue_barrier_payload_t operation = {};
  iree_hal_replay_queue_barriers_footer_t footer = {0, UINT64_MAX, 0};
  iree_const_byte_span_t spans[] = {
      iree_make_const_byte_span(&operation, sizeof(operation)),
      iree_make_const_byte_span(&footer, sizeof(footer)),
  };
  builder.Append(metadata, IREE_ARRAYSIZE(spans), spans);
  ++metadata.sequence_ordinal;
  metadata.record_flags |=
      IREE_HAL_REPLAY_FILE_RECORD_FLAG_MEMORY_TRANSITION_RECIPES;
  iree_hal_replay_command_buffer_execution_barrier_payload_t action = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
      .buffer_barrier_count = 1};
  iree_hal_replay_buffer_barrier_payload_t range = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_HOST_READ};
  range.buffer_ref.buffer_id = 7;
  range.buffer_ref.offset = 6;
  range.buffer_ref.length = 4;
  iree_hal_replay_memory_transition_recipe_payload_t recipe = {
      .effects = IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM,
      .operation_count = 1};
  iree_hal_replay_memory_transition_operation_payload_t transition = {};
  transition.kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE;
  transition.executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE;
  transition.operation = IREE_HAL_MEMORY_TRANSITION_OPERATION_RELEASE_TO_SYSTEM;
  transition.range_granularity = 64;
  footer = {
      sizeof(action) + sizeof(range) + sizeof(recipe) + sizeof(transition), 1,
      UINT64_MAX};
  iree_const_byte_span_t explicit_spans[] = {
      iree_make_const_byte_span(&operation, sizeof(operation)),
      iree_make_const_byte_span(&action, sizeof(action)),
      iree_make_const_byte_span(&range, sizeof(range)),
      iree_make_const_byte_span(&recipe, sizeof(recipe)),
      iree_make_const_byte_span(&transition, sizeof(transition)),
      iree_make_const_byte_span(&footer, sizeof(footer)),
  };
  builder.Append(metadata, IREE_ARRAYSIZE(explicit_spans), explicit_spans);
  auto storage = builder.Finish();
  iree_hal_replay_dump_options_t options =
      iree_hal_replay_dump_options_default();
  std::string output;
  IREE_ASSERT_OK(
      DumpReplayToString(MakeReplayFileContents(storage), &options, &output));
  EXPECT_THAT(output, HasSubstr("before=default after=[]"));
  EXPECT_THAT(output, HasSubstr("buffer_ref={buffer_id=7 offset=6 length=4"));
  EXPECT_THAT(output,
              HasSubstr("recipe={effects=0x00000004 operations=[{kind=2"));
  EXPECT_THAT(output, HasSubstr("host_instruction=0 host_fence_before=0 "
                                "host_fence_after=0"));
  options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
  output.clear();
  IREE_ASSERT_OK(
      DumpReplayToString(MakeReplayFileContents(storage), &options, &output));
  EXPECT_THAT(output, HasSubstr("\"before\":null,\"after\":[]"));
  EXPECT_THAT(output, HasSubstr("\"buffer_ref\":{\"buffer_id\":7,"));
  EXPECT_THAT(output, HasSubstr("\"offset\":6,\"length\":4"));
  EXPECT_THAT(output, HasSubstr("\"recipe\":{\"effects\":4,"));
  EXPECT_THAT(output, HasSubstr("\"range_granularity\":64"));
  EXPECT_THAT(output, HasSubstr("\"host_instruction\":0,"));
  EXPECT_THAT(output, HasSubstr("\"host_fence_before\":0,"));
  EXPECT_THAT(output, HasSubstr("\"host_fence_after\":0"));
}

}  // namespace
