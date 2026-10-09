// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/file.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

void InitializeValidReplayFileHeader(
    iree_hal_replay_file_header_t* file_header) {
  memset(file_header, 0, sizeof(*file_header));
  file_header->magic = IREE_HAL_REPLAY_FILE_MAGIC;
  file_header->version_major = IREE_HAL_REPLAY_FILE_VERSION_MAJOR;
  file_header->version_minor = IREE_HAL_REPLAY_FILE_VERSION_MINOR;
  file_header->header_length = sizeof(*file_header);
}

std::vector<uint8_t> MakeReplayFileHeaderStorage() {
  std::vector<uint8_t> storage(sizeof(iree_hal_replay_file_header_t), 0);
  InitializeValidReplayFileHeader(
      reinterpret_cast<iree_hal_replay_file_header_t*>(storage.data()));
  return storage;
}

TEST(ReplayFileWriterTest, WritesReplayRecordsAndRanges) {
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

  const uint32_t first_payload = 0x12345678u;
  const uint16_t second_payload = 0xABCDu;
  iree_const_byte_span_t iovecs[2] = {
      iree_make_const_byte_span(&first_payload, sizeof(first_payload)),
      iree_make_const_byte_span(&second_payload, sizeof(second_payload)),
  };
  iree_hal_replay_file_record_metadata_t metadata = {
      .sequence_ordinal = 42,
      .thread_id = 7,
      .device_id = 1,
      .object_id = 2,
      .related_object_id = 3,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .payload_type = 10,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
      .operation_code = 11,
      .status_code = (uint32_t)IREE_STATUS_CANCELLED};
  iree_hal_replay_file_range_t payload_range =
      iree_hal_replay_file_range_empty();
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &metadata, 2, iovecs, &payload_range));

  iree_hal_replay_file_record_metadata_t empty_metadata = {
      .sequence_ordinal = 43,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_BLOB};
  iree_hal_replay_file_range_t empty_payload_range =
      iree_hal_replay_file_range_empty();
  IREE_ASSERT_OK(iree_hal_replay_file_writer_append_record(
      writer, &empty_metadata, 0, nullptr, &empty_payload_range));

  IREE_ASSERT_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);

  iree_hal_replay_file_header_t file_header;
  iree_host_size_t offset = 0;
  IREE_ASSERT_OK(iree_hal_replay_file_parse_header(
      iree_make_const_byte_span(storage.data(), storage.size()), &file_header,
      &offset));
  EXPECT_EQ(IREE_HAL_REPLAY_FILE_MAGIC, file_header.magic);
  EXPECT_EQ(IREE_HAL_REPLAY_FILE_VERSION_MAJOR, file_header.version_major);
  EXPECT_EQ(IREE_HAL_REPLAY_FILE_VERSION_MINOR, file_header.version_minor);
  EXPECT_EQ(sizeof(file_header), file_header.header_length);
  EXPECT_EQ(0u, file_header.flags);
  ASSERT_NE(0u, file_header.file_length);
  ASSERT_LE(file_header.file_length, storage.size());
  iree_const_byte_span_t file_contents = iree_make_const_byte_span(
      storage.data(), (iree_host_size_t)file_header.file_length);

  iree_hal_replay_file_record_t record;
  IREE_ASSERT_OK(iree_hal_replay_file_parse_record(file_contents, offset,
                                                   &record, &offset));
  EXPECT_EQ(IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
            record.header.record_type);
  EXPECT_EQ(42u, record.header.sequence_ordinal);
  EXPECT_EQ(7u, record.header.thread_id);
  EXPECT_EQ(1u, record.header.device_id);
  EXPECT_EQ(2u, record.header.object_id);
  EXPECT_EQ(3u, record.header.related_object_id);
  EXPECT_EQ(10u, record.header.payload_type);
  EXPECT_EQ(IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER, record.header.object_type);
  EXPECT_EQ(11u, record.header.operation_code);
  EXPECT_EQ((uint32_t)IREE_STATUS_CANCELLED, record.header.status_code);
  ASSERT_EQ(sizeof(first_payload) + sizeof(second_payload),
            record.payload.data_length);
  EXPECT_EQ(0,
            memcmp(record.payload.data, &first_payload, sizeof(first_payload)));
  EXPECT_EQ(0, memcmp(record.payload.data + sizeof(first_payload),
                      &second_payload, sizeof(second_payload)));

  EXPECT_EQ(record.payload.data - storage.data(), payload_range.offset);
  EXPECT_EQ(record.payload.data_length, payload_range.length);
  EXPECT_EQ(record.payload.data_length, payload_range.uncompressed_length);
  EXPECT_EQ(IREE_HAL_REPLAY_COMPRESSION_TYPE_NONE,
            payload_range.compression_type);
  EXPECT_EQ(IREE_HAL_REPLAY_DIGEST_TYPE_FNV1A_64, payload_range.digest_type);
  IREE_EXPECT_OK(
      iree_hal_replay_file_range_validate(file_contents, &payload_range));

  iree_hal_replay_file_record_t empty_record;
  IREE_ASSERT_OK(iree_hal_replay_file_parse_record(file_contents, offset,
                                                   &empty_record, &offset));
  EXPECT_EQ(IREE_HAL_REPLAY_FILE_RECORD_TYPE_BLOB,
            empty_record.header.record_type);
  EXPECT_EQ(43u, empty_record.header.sequence_ordinal);
  EXPECT_EQ(0u, empty_record.payload.data_length);
  EXPECT_EQ(empty_record.payload.data - storage.data(),
            empty_payload_range.offset);
  EXPECT_EQ(0u, empty_payload_range.length);
  IREE_EXPECT_OK(
      iree_hal_replay_file_range_validate(file_contents, &empty_payload_range));

  EXPECT_EQ(file_header.file_length, offset);
}

TEST(ReplayFileWriterTest, RejectsAppendAfterClose) {
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
  IREE_ASSERT_OK(iree_hal_replay_file_writer_close(writer));

  iree_hal_replay_file_record_metadata_t metadata = {
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_SESSION};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_replay_file_writer_append_record(
                            writer, &metadata, 0, nullptr, nullptr));
  iree_hal_replay_file_writer_free(writer);
}

TEST(ReplayFileWriterTest, RejectsInvalidMetadata) {
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

  iree_hal_replay_file_record_metadata_t metadata = {};
  metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_NONE;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_replay_file_writer_append_record(
                            writer, &metadata, 0, nullptr, nullptr));

  metadata.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_SESSION;
  metadata.record_flags = IREE_HAL_REPLAY_FILE_RECORD_FLAG_QUEUE_BARRIERS;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_replay_file_writer_append_record(
                            writer, &metadata, 0, nullptr, nullptr));
  metadata.record_flags =
      IREE_HAL_REPLAY_FILE_RECORD_FLAG_MEMORY_TRANSITION_RECIPES;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_replay_file_writer_append_record(
                            writer, &metadata, 0, nullptr, nullptr));

  IREE_ASSERT_OK(iree_hal_replay_file_writer_close(writer));
  iree_hal_replay_file_writer_free(writer);
}

TEST(ReplayFileParseTest, RejectsBadMagic) {
  std::vector<uint8_t> storage(sizeof(iree_hal_replay_file_header_t), 0);
  iree_hal_replay_file_header_t file_header;
  iree_host_size_t offset = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_replay_file_parse_header(
          iree_make_const_byte_span(storage.data(), storage.size()),
          &file_header, &offset));
}

TEST(ReplayFileParseTest, AcceptsSupportedMinorVersions) {
  for (uint16_t version_minor = 0;
       version_minor <= IREE_HAL_REPLAY_FILE_VERSION_MINOR; ++version_minor) {
    std::vector<uint8_t> storage = MakeReplayFileHeaderStorage();
    auto* file_header =
        reinterpret_cast<iree_hal_replay_file_header_t*>(storage.data());
    file_header->version_minor = version_minor;

    iree_hal_replay_file_header_t parsed_header;
    iree_host_size_t offset = 0;
    IREE_ASSERT_OK(iree_hal_replay_file_parse_header(
        iree_make_const_byte_span(storage.data(), storage.size()),
        &parsed_header, &offset));
    EXPECT_EQ(version_minor, parsed_header.version_minor);
  }
}

TEST(ReplayFileParseTest, RejectsFutureMinorVersion) {
  std::vector<uint8_t> storage = MakeReplayFileHeaderStorage();
  auto* file_header =
      reinterpret_cast<iree_hal_replay_file_header_t*>(storage.data());
  file_header->version_minor = IREE_HAL_REPLAY_FILE_VERSION_MINOR + 1;

  iree_hal_replay_file_header_t parsed_header;
  iree_host_size_t offset = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_replay_file_parse_header(
          iree_make_const_byte_span(storage.data(), storage.size()),
          &parsed_header, &offset));
}

TEST(ReplayFileParseTest, RejectsNonZeroFileFlags) {
  std::vector<uint8_t> storage = MakeReplayFileHeaderStorage();
  auto* file_header =
      reinterpret_cast<iree_hal_replay_file_header_t*>(storage.data());
  file_header->flags = 1;

  iree_hal_replay_file_header_t parsed_header;
  iree_host_size_t offset = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_replay_file_parse_header(
          iree_make_const_byte_span(storage.data(), storage.size()),
          &parsed_header, &offset));
}

TEST(ReplayFileParseTest, RejectsFileLengthBeforeFirstRecord) {
  std::vector<uint8_t> storage = MakeReplayFileHeaderStorage();
  auto* file_header =
      reinterpret_cast<iree_hal_replay_file_header_t*>(storage.data());
  file_header->file_length = file_header->header_length - 1;

  iree_hal_replay_file_header_t parsed_header;
  iree_host_size_t offset = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_replay_file_parse_header(
          iree_make_const_byte_span(storage.data(), storage.size()),
          &parsed_header, &offset));
}

TEST(ReplayFileParseTest, RejectsFileLengthPastContents) {
  std::vector<uint8_t> storage = MakeReplayFileHeaderStorage();
  auto* file_header =
      reinterpret_cast<iree_hal_replay_file_header_t*>(storage.data());
  file_header->file_length = storage.size() + 1;

  iree_hal_replay_file_header_t parsed_header;
  iree_host_size_t offset = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_replay_file_parse_header(
          iree_make_const_byte_span(storage.data(), storage.size()),
          &parsed_header, &offset));
}

TEST(ReplayFileParseTest, RejectsRequiredUnknownRecordType) {
  std::vector<uint8_t> storage = MakeReplayFileHeaderStorage();
  auto* file_header =
      reinterpret_cast<iree_hal_replay_file_header_t*>(storage.data());
  const size_t file_length = sizeof(iree_hal_replay_file_header_t) +
                             sizeof(iree_hal_replay_file_record_header_t);
  file_header->file_length = file_length;
  storage.resize(file_length, 0);
  auto* record_header = reinterpret_cast<iree_hal_replay_file_record_header_t*>(
      storage.data() + sizeof(iree_hal_replay_file_header_t));
  record_header->record_length = sizeof(*record_header);
  record_header->header_length = sizeof(*record_header);
  record_header->record_type = 999;

  iree_hal_replay_file_record_t parsed_record;
  iree_host_size_t next_offset = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_replay_file_parse_record(
          iree_make_const_byte_span(storage.data(), storage.size()),
          sizeof(iree_hal_replay_file_header_t), &parsed_record, &next_offset));
}

TEST(ReplayFileParseTest, AcceptsOptionalUnknownRecordType) {
  std::vector<uint8_t> storage = MakeReplayFileHeaderStorage();
  auto* file_header =
      reinterpret_cast<iree_hal_replay_file_header_t*>(storage.data());
  const size_t file_length = sizeof(iree_hal_replay_file_header_t) +
                             sizeof(iree_hal_replay_file_record_header_t);
  file_header->file_length = file_length;
  storage.resize(file_length, 0);
  auto* record_header = reinterpret_cast<iree_hal_replay_file_record_header_t*>(
      storage.data() + sizeof(iree_hal_replay_file_header_t));
  record_header->record_length = sizeof(*record_header);
  record_header->header_length = sizeof(*record_header);
  record_header->record_type = 999;
  record_header->record_flags = IREE_HAL_REPLAY_FILE_RECORD_FLAG_OPTIONAL;

  iree_hal_replay_file_record_t parsed_record;
  iree_host_size_t next_offset = 0;
  IREE_ASSERT_OK(iree_hal_replay_file_parse_record(
      iree_make_const_byte_span(storage.data(), storage.size()),
      sizeof(iree_hal_replay_file_header_t), &parsed_record, &next_offset));
  EXPECT_EQ(999u, parsed_record.header.record_type);
  EXPECT_EQ(file_length, next_offset);
}

TEST(ReplayFileRangeTest, RejectsRangePastContents) {
  std::vector<uint8_t> storage(16, 0);
  iree_hal_replay_file_range_t range = iree_hal_replay_file_range_empty();
  range.offset = 12;
  range.length = 8;
  range.uncompressed_length = 8;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_replay_file_range_validate(
          iree_make_const_byte_span(storage.data(), storage.size()), &range));
}

TEST(ReplayFileRangeTest, RejectsDigestMismatch) {
  std::vector<uint8_t> storage = {1, 2, 3, 4};
  iree_hal_replay_file_range_t range = iree_hal_replay_file_range_empty();
  range.offset = 0;
  range.length = storage.size();
  range.uncompressed_length = storage.size();
  range.digest_type = IREE_HAL_REPLAY_DIGEST_TYPE_FNV1A_64;
  memset(range.digest, 0xAA, sizeof(range.digest));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_replay_file_range_validate(
          iree_make_const_byte_span(storage.data(), storage.size()), &range));
}

TEST(ReplayFileRangeTest, RejectsDigestBytesWithoutDigestType) {
  std::vector<uint8_t> storage = {1, 2, 3, 4};
  iree_hal_replay_file_range_t range = iree_hal_replay_file_range_empty();
  range.offset = 0;
  range.length = storage.size();
  range.uncompressed_length = storage.size();
  range.digest_type = IREE_HAL_REPLAY_DIGEST_TYPE_NONE;
  range.digest[0] = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_replay_file_range_validate(
          iree_make_const_byte_span(storage.data(), storage.size()), &range));
}

TEST(ReplayFileReaderTest, QueueBarriersPreserveDefaultAndEmptyLists) {
  const iree_hal_replay_queue_barrier_payload_t operation = {};
  const iree_hal_replay_queue_barriers_footer_t footer = {
      /*.payload_length=*/0, /*.before_count=*/UINT64_MAX, /*.after_count=*/0};
  iree_hal_replay_file_record_header_t header = {};
  header.header_length = sizeof(header);
  header.payload_length = sizeof(operation) + sizeof(footer);
  header.record_length = header.header_length + header.payload_length;
  header.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  header.operation_code = IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_BARRIER;
  header.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_BARRIER;
  header.record_flags = IREE_HAL_REPLAY_FILE_RECORD_FLAG_QUEUE_BARRIERS;
  std::vector<uint8_t> bytes(header.record_length);
  memcpy(bytes.data(), &header, sizeof(header));
  memcpy(bytes.data() + sizeof(header), &operation, sizeof(operation));
  memcpy(bytes.data() + sizeof(header) + sizeof(operation), &footer,
         sizeof(footer));
  iree_hal_replay_file_record_t record;
  iree_host_size_t offset = 0;
  IREE_ASSERT_OK(iree_hal_replay_file_parse_record(
      iree_make_const_byte_span(bytes.data(), bytes.size()), 0, &record,
      &offset));
  EXPECT_EQ(record.payload.data_length, sizeof(operation));
  EXPECT_EQ(record.barriers.before.count, UINT64_MAX);
  EXPECT_EQ(record.barriers.after.count, 0u);
  EXPECT_EQ(offset, bytes.size());

  // A count cannot consume bytes belonging to the operation, and a footer
  // cannot claim an extent outside this record.
  for (auto invalid :
       {iree_hal_replay_queue_barriers_footer_t{0, 1, 0},
        iree_hal_replay_queue_barriers_footer_t{UINT64_MAX, 0, 0},
        iree_hal_replay_queue_barriers_footer_t{sizeof(operation), 0, 0}}) {
    memcpy(bytes.data() + sizeof(header) + sizeof(operation), &invalid,
           sizeof(invalid));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_DATA_LOSS,
        iree_hal_replay_file_parse_record(
            iree_make_const_byte_span(bytes.data(), bytes.size()), 0, &record,
            &offset));
  }
  memcpy(bytes.data() + sizeof(header) + sizeof(operation), &footer,
         sizeof(footer));
  header.operation_code = IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_EXECUTE;
  memcpy(bytes.data(), &header, sizeof(header));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_replay_file_parse_record(
          iree_make_const_byte_span(bytes.data(), bytes.size()), 0, &record,
          &offset));
}

TEST(ReplayFileReaderTest, RejectsRecipeEffectsWithoutOperations) {
  const iree_hal_replay_queue_barrier_payload_t operation = {};
  const iree_hal_replay_command_buffer_execution_barrier_payload_t barrier = {
      .source_stage_mask = 0,
      .target_stage_mask = 0,
      .flags = 0,
      .memory_barrier_count = 0,
      .buffer_barrier_count = 1,
  };
  const iree_hal_replay_buffer_barrier_payload_t buffer = {};
  const iree_hal_replay_memory_transition_recipe_payload_t recipe = {
      .effects = 1,
      .operation_count = 0,
  };
  const iree_hal_replay_queue_barriers_footer_t footer = {
      .payload_length = sizeof(barrier) + sizeof(buffer) + sizeof(recipe),
      .before_count = 1,
      .after_count = 0,
  };
  iree_hal_replay_file_record_header_t header = {};
  header.header_length = sizeof(header);
  header.payload_length =
      sizeof(operation) + footer.payload_length + sizeof(footer);
  header.record_length = header.header_length + header.payload_length;
  header.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
  header.operation_code = IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_BARRIER;
  header.payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_BARRIER;
  header.record_flags =
      IREE_HAL_REPLAY_FILE_RECORD_FLAG_QUEUE_BARRIERS |
      IREE_HAL_REPLAY_FILE_RECORD_FLAG_MEMORY_TRANSITION_RECIPES;

  std::vector<uint8_t> bytes(header.record_length);
  uint8_t* p = bytes.data();
  memcpy(p, &header, sizeof(header));
  p += sizeof(header);
  memcpy(p, &operation, sizeof(operation));
  p += sizeof(operation);
  memcpy(p, &barrier, sizeof(barrier));
  p += sizeof(barrier);
  memcpy(p, &buffer, sizeof(buffer));
  p += sizeof(buffer);
  memcpy(p, &recipe, sizeof(recipe));
  p += sizeof(recipe);
  memcpy(p, &footer, sizeof(footer));

  iree_hal_replay_file_record_t record;
  iree_host_size_t offset = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_replay_file_parse_record(
          iree_make_const_byte_span(bytes.data(), bytes.size()), 0, &record,
          &offset));
}

}  // namespace
