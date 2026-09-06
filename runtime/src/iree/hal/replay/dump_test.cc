// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/dump.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "iree/hal/api.h"
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
      /*.fn=*/AppendToString,
      /*.user_data=*/output,
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

  iree_hal_replay_file_record_metadata_t session_metadata = {};
  session_metadata.sequence_ordinal = 0;
  session_metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_SESSION;
  IREE_CHECK_OK(iree_hal_replay_file_writer_append_record(
      writer, &session_metadata, 0, nullptr, nullptr));

  iree_hal_replay_buffer_object_payload_t buffer_payload = {};
  buffer_payload.allocation_size = 256;
  buffer_payload.byte_length = 64;
  buffer_payload.allowed_usage = 0x11;
  iree_const_byte_span_t buffer_payload_span =
      iree_make_const_byte_span(&buffer_payload, sizeof(buffer_payload));
  iree_hal_replay_file_record_metadata_t object_metadata = {};
  object_metadata.sequence_ordinal = 1;
  object_metadata.object_id = 7;
  object_metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT;
  object_metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT;
  object_metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER;
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
  iree_hal_replay_semaphore_timepoint_payload_t queue_wait_wait = {};
  queue_wait_wait.semaphore_id = 41;
  queue_wait_wait.value = 5;
  iree_hal_replay_semaphore_timepoint_payload_t queue_wait_signal = {};
  queue_wait_signal.semaphore_id = 51;
  queue_wait_signal.value = 6;
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
  iree_hal_replay_semaphore_timepoint_payload_t queue_store_wait = {};
  queue_store_wait.semaphore_id = 42;
  queue_store_wait.value = 7;
  iree_hal_replay_semaphore_timepoint_payload_t queue_store_signal = {};
  queue_store_signal.semaphore_id = 52;
  queue_store_signal.value = 8;
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
  iree_hal_replay_semaphore_timepoint_payload_t queue_rmw_wait = {};
  queue_rmw_wait.semaphore_id = 43;
  queue_rmw_wait.value = 9;
  iree_hal_replay_semaphore_timepoint_payload_t queue_rmw_signal = {};
  queue_rmw_signal.semaphore_id = 53;
  queue_rmw_signal.value = 10;
  AppendQueueAtomicRecord(
      &builder,
      MakeAtomicRecordMetadata(5, IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
                               IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_RMW,
                               IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_ATOMIC_RMW),
      queue_rmw, queue_rmw_wait, queue_rmw_signal);

  return builder.Finish();
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

  iree_hal_replay_file_record_metadata_t session_metadata = {};
  session_metadata.sequence_ordinal = 0;
  session_metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_SESSION;
  IREE_CHECK_OK(iree_hal_replay_file_writer_append_record(
      writer, &session_metadata, 0, nullptr, nullptr));

  const char scope_name[] = "execute";
  iree_hal_replay_scope_payload_t payload = {};
  payload.name_length = sizeof(scope_name) - 1;
  iree_const_byte_span_t iovecs[] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(scope_name, sizeof(scope_name) - 1),
  };
  iree_hal_replay_file_record_metadata_t begin_metadata = {};
  begin_metadata.sequence_ordinal = 1;
  begin_metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  begin_metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_REPLAY_SCOPE;
  begin_metadata.operation_code =
      IREE_HAL_REPLAY_OPERATION_CODE_REPLAY_SCOPE_BEGIN;
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
  iree_hal_replay_executable_load_payload_t payload = {};
  payload.queue_family_ordinal = 0;
  payload.target_physical_device_affinity = 1;
  payload.executable_data_length = sizeof(executable_data);
  payload.constant_count = IREE_ARRAYSIZE(constants);
  payload.load_flags = IREE_HAL_EXECUTABLE_LOAD_FLAG_ENABLE_DEBUGGING;
  payload.target_kind = IREE_HAL_EXECUTABLE_TARGET_KIND_VIRTUAL;
  payload.target_family_length = sizeof(target_family) - 1;
  payload.target_key_length = sizeof(target_key) - 1;
  payload.executable_metadata_length = sizeof(metadata_header) +
                                       sizeof(function_metadata) +
                                       sizeof(function_name) - 1;
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

  iree_hal_replay_buffer_range_data_payload_t payload = {};
  payload.byte_offset = 64;
  payload.byte_length = 4;
  payload.data_length = 4;
  payload.memory_access = IREE_HAL_MEMORY_ACCESS_WRITE;
  const uint8_t data[] = {0x01, 0x02, 0x03, 0x04};
  iree_const_byte_span_t iovecs[2] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(data, sizeof(data)),
  };
  iree_hal_replay_file_record_metadata_t metadata = {};
  metadata.sequence_ordinal = 0;
  metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE_DATA;
  metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER;
  metadata.operation_code = IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_FLUSH_RANGE;
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

  iree_hal_replay_queue_alloca_payload_t payload = {};
  payload.signal_semaphore_count = 1;
  payload.request_count = 1;
  iree_hal_replay_semaphore_timepoint_payload_t signal = {};
  signal.semaphore_id = 42;
  signal.value = 7;
  iree_hal_replay_queue_alloca_request_payload_t request = {};
  request.buffer_id = 9;
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
  iree_hal_replay_file_record_metadata_t metadata = {};
  metadata.sequence_ordinal = 0;
  metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ALLOCA;
  metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE;
  metadata.operation_code = IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_ALLOCA;
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

  iree_hal_replay_queue_execute_payload_t payload = {};
  payload.command_buffer_id = 9;
  payload.wait_semaphore_count = 1;
  payload.signal_semaphore_count = 1;
  payload.binding_count = 1;
  iree_hal_replay_semaphore_timepoint_payload_t wait = {};
  wait.semaphore_id = 42;
  wait.value = 1;
  iree_hal_replay_semaphore_timepoint_payload_t signal = {};
  signal.semaphore_id = 43;
  signal.value = 2;
  iree_hal_replay_buffer_ref_payload_t binding = {};
  binding.buffer_id = 7;
  binding.offset = 64;
  binding.length = 128;
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
  iree_hal_replay_queue_barrier_payload_t barrier_payload = {};
  barrier_payload.wait_semaphore_count = 1;
  barrier_payload.signal_semaphore_count = 1;
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

  iree_hal_replay_command_buffer_execution_barrier_payload_t payload = {};
  payload.source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH;
  payload.target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER;
  payload.memory_barrier_count = 1;
  iree_hal_replay_memory_barrier_payload_t memory_barrier = {};
  memory_barrier.source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE;
  memory_barrier.target_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_READ;
  iree_const_byte_span_t iovecs[2] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(&memory_barrier, sizeof(memory_barrier)),
  };
  iree_hal_replay_file_record_metadata_t metadata = {};
  metadata.sequence_ordinal = 0;
  metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  metadata.payload_type =
      IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_EXECUTION_BARRIER;
  metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER;
  metadata.operation_code =
      IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_EXECUTION_BARRIER;
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
  iree_hal_replay_file_record_metadata_t file_metadata = {};
  file_metadata.sequence_ordinal = 0;
  file_metadata.object_id = 7;
  file_metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT;
  file_metadata.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_FILE_OBJECT;
  file_metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_FILE;
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

TEST(ReplayDumpTest, EmitsQueueTransferRanges) {
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

  iree_hal_replay_provisioned_queue_object_payload_t queue_payload = {};
  queue_payload.family_ordinal = 2;
  queue_payload.queue_ordinal = 1;
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

  iree_hal_replay_queue_transfer_payload_t payload = {};
  payload.wait_semaphore_count = 1;
  payload.signal_semaphore_count = 1;
  payload.operation_count = 1;
  payload.data_length = 4;
  iree_hal_replay_semaphore_timepoint_payload_t wait = {};
  wait.semaphore_id = 42;
  wait.value = 1;
  iree_hal_replay_semaphore_timepoint_payload_t signal = {};
  signal.semaphore_id = 43;
  signal.value = 2;
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
  metadata.sequence_ordinal = 1;
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
  metadata.sequence_ordinal = 2;
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
  metadata.sequence_ordinal = 3;
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
  iree_hal_replay_file_record_metadata_t fill_metadata = {};
  fill_metadata.sequence_ordinal = 0;
  fill_metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  fill_metadata.payload_type =
      IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_FILL_BUFFER;
  fill_metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER;
  fill_metadata.operation_code =
      IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_FILL_BUFFER;
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
  iree_hal_replay_file_record_metadata_t update_metadata = {};
  update_metadata.sequence_ordinal = 1;
  update_metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  update_metadata.payload_type =
      IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_UPDATE_BUFFER;
  update_metadata.object_type = IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER;
  update_metadata.operation_code =
      IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_UPDATE_BUFFER;
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
              HasSubstr("\"condition\":1,\"condition_name\":\"not_equal\""));
  EXPECT_THAT(json_output,
              HasSubstr("\"payload_type\":\"command_buffer_atomic_store\""));
  EXPECT_THAT(json_output,
              HasSubstr("\"target_ref\":{\"buffer_id\":0,\"offset\":16,"
                        "\"length\":4,\"buffer_slot\":3}"));
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
  iree_hal_replay_queue_atomic_wait_payload_t queue_payload = {};
  queue_payload.wait_semaphore_count = 1;
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

}  // namespace
