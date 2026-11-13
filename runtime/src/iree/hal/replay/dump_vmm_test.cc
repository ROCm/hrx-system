// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/replay/dump.h"
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

TEST(ReplayDumpTest, EmitsReservationAndMappingBatchPayloads) {
  ReplayFileBuilder builder(/*capacity=*/4096);
  const iree_hal_replay_virtual_buffer_create_payload_t reservation = {
      .byte_length = 16384,
      .min_alignment = 8192,
      .mapping_granularity = 4096,
      .family_count = 1,
      .flags = IREE_HAL_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS,
      .source_pool_count = 2,
  };
  const iree_hal_replay_pool_family_access_payload_t family = {
      .device_ordinal = 0,
      .family_ordinal = 1,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
  };
  const iree_hal_replay_object_id_t source_pool_ids[] = {11, 13};
  const iree_const_byte_span_t reservation_payloads[] = {
      iree_make_const_byte_span(&reservation, sizeof(reservation)),
      iree_make_const_byte_span(&family, sizeof(family)),
      iree_make_const_byte_span(source_pool_ids, sizeof(source_pool_ids)),
  };
  const iree_hal_replay_file_record_metadata_t reservation_metadata = {
      .sequence_ordinal = 0,
      .related_object_id = 17,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_VIRTUAL_BUFFER_CREATE,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
      .operation_code = IREE_HAL_REPLAY_OPERATION_CODE_VIRTUAL_BUFFER_CREATE,
  };
  builder.Append(reservation_metadata, IREE_ARRAYSIZE(reservation_payloads),
                 reservation_payloads);

  const iree_hal_replay_queue_map_payload_t batch = {
      .wait_semaphore_count = 1,
      .signal_semaphore_count = 1,
      .mapping_count = 2,
  };
  const iree_hal_replay_semaphore_timepoint_payload_t timepoints[] = {
      {.semaphore_id = 20, .value = 3},
      {.semaphore_id = 20, .value = 4},
  };
  const iree_hal_replay_memory_mapping_payload_t mappings[] = {
      {
          .target_buffer_id = 17,
          .scope_id = 2,
          .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
          .target_offset = 4096,
          .byte_length = 4096,
          .source_ref = {.buffer_id = 19, .offset = 8192, .length = 4096},
          .previous_source_ref = {.length = 4096},
          .access = IREE_HAL_MEMORY_ACCESS_READ,
      },
      {
          .target_buffer_id = 17,
          .scope_id = 2,
          .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND,
          .target_offset = 12288,
          .byte_length = 4096,
          .source_ref = {.length = 4096},
          .previous_source_ref =
              {
                  .buffer_id = 19,
                  .offset = 12288,
                  .length = 4096,
              },
      },
  };
  const iree_const_byte_span_t batch_payloads[] = {
      iree_make_const_byte_span(&batch, sizeof(batch)),
      iree_make_const_byte_span(timepoints, sizeof(timepoints)),
      iree_make_const_byte_span(mappings, sizeof(mappings)),
  };
  const iree_hal_replay_file_record_metadata_t batch_metadata = {
      .sequence_ordinal = 1,
      .object_id = 21,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_MAP,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_QUEUE,
      .operation_code = IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_MAP,
  };
  builder.Append(batch_metadata, IREE_ARRAYSIZE(batch_payloads),
                 batch_payloads);
  const auto storage = builder.Finish();
  const auto contents = MakeReplayFileContents(storage);
  // Readers accept file bytes at arbitrary addresses; payloads need no native
  // struct alignment even when they contain variable-length arrays.
  std::vector<uint8_t> unaligned_storage(contents.data_length + 1);
  std::memcpy(unaligned_storage.data() + 1, contents.data,
              contents.data_length);
  for (const auto input :
       {contents, iree_make_const_byte_span(unaligned_storage.data() + 1,
                                            contents.data_length)}) {
    auto options = iree_hal_replay_dump_options_default();
    std::string output;
    IREE_ASSERT_OK(DumpReplayToString(input, &options, &output));
    EXPECT_THAT(output, HasSubstr("buffer_id=17 byte_length=16384 "
                                  "min_alignment=8192 mapping_granularity=4096 "
                                  "flags=0x00000001 family_count=1"));
    EXPECT_THAT(output, HasSubstr("source_pool_ids=[11,13]"));
    EXPECT_THAT(output,
                HasSubstr("wait_semaphores=[{semaphore_id=20 value=3}]"));
    EXPECT_THAT(output,
                HasSubstr("signal_semaphores=[{semaphore_id=20 value=4}]"));
    EXPECT_THAT(
        output,
        HasSubstr("mappings=[{target_buffer_id=17 scope_id=2 operation=1 "
                  "target=[4096, +4096] access=0x00000001 "
                  "source={buffer_id=19 offset=8192 length=4096 slot=0} "
                  "previous_source={buffer_id=0 offset=0 length=4096 slot=0}},"
                  "{target_buffer_id=17 scope_id=2 operation=2 "
                  "target=[12288, +4096] access=0x00000000 "
                  "source={buffer_id=0 offset=0 length=4096 slot=0} "
                  "previous_source={buffer_id=19 offset=12288 length=4096 "
                  "slot=0}}]"));
    options.format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
    output.clear();
    IREE_ASSERT_OK(DumpReplayToString(input, &options, &output));
    EXPECT_THAT(
        output,
        HasSubstr(
            "\"buffer_id\":17,\"byte_length\":16384,"
            "\"min_alignment\":8192,"
            "\"mapping_granularity\":4096,\"flags\":1,\"family_count\":1"));
    EXPECT_THAT(output, HasSubstr("\"source_pool_ids\":[11,13]"));
    EXPECT_THAT(
        output,
        HasSubstr(
            "\"mappings\":[{\"target_buffer_id\":17,\"scope_id\":2,"
            "\"operation\":1,"
            "\"target_offset\":4096,\"byte_length\":4096,\"access\":1,"
            "\"source\":{\"buffer_id\":19,\"offset\":8192,\"length\":"
            "4096,\"buffer_slot\":0},\"previous_source\":{\"buffer_id\":0,"
            "\"offset\":0,\"length\":4096,\"buffer_slot\":0}},"
            "{\"target_buffer_id\":17,\"scope_id\":2,\"operation\":2,"
            "\"target_offset\":12288,\"byte_length\":4096,\"access\":0,"
            "\"source\":{\"buffer_id\":0,\"offset\":0,\"length\":4096,"
            "\"buffer_slot\":0},\"previous_source\":{\"buffer_id\":19,"
            "\"offset\":12288,\"length\":4096,\"buffer_slot\":0}}]"));
  }
}

TEST(ReplayDumpTest, RejectsTruncatedReservationAndMappingArrays) {
  const iree_hal_replay_virtual_buffer_create_payload_t reservation = {
      .byte_length = 4096,
      .family_count = 1,
  };
  const iree_hal_replay_queue_map_payload_t batch = {
      .wait_semaphore_count = 1,
      .signal_semaphore_count = 1,
      .mapping_count = 1,
  };
  for (bool is_mapping : {false, true}) {
    ReplayFileBuilder builder(/*capacity=*/4096);
    const iree_hal_replay_file_record_metadata_t metadata = {
        .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
        .payload_type = static_cast<iree_hal_replay_payload_type_t>(
            is_mapping ? IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_MAP
                       : IREE_HAL_REPLAY_PAYLOAD_TYPE_VIRTUAL_BUFFER_CREATE),
        .operation_code = static_cast<iree_hal_replay_operation_code_t>(
            is_mapping ? IREE_HAL_REPLAY_OPERATION_CODE_QUEUE_MAP
                       : IREE_HAL_REPLAY_OPERATION_CODE_VIRTUAL_BUFFER_CREATE),
    };
    const auto payload =
        is_mapping
            ? iree_make_const_byte_span(&batch, sizeof(batch))
            : iree_make_const_byte_span(&reservation, sizeof(reservation));
    builder.Append(metadata, 1, &payload);
    const auto storage = builder.Finish();
    for (auto format : {IREE_HAL_REPLAY_DUMP_FORMAT_TEXT,
                        IREE_HAL_REPLAY_DUMP_FORMAT_JSONL}) {
      const iree_hal_replay_dump_options_t options = {.format = format};
      std::string output;
      IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                            DumpReplayToString(MakeReplayFileContents(storage),
                                               &options, &output));
    }
  }
}

TEST(ReplayDumpTest, ValidatesAddressConstraintsAgainstTheFileVersion) {
  for (const uint16_t minor : {uint16_t{8}, uint16_t{9}, uint16_t{10}}) {
    for (const uint32_t flags :
         {uint32_t{0}, uint32_t{1}, UINT32_C(0x80000000)}) {
      ReplayFileBuilder builder(/*capacity=*/4096);
      const iree_hal_replay_virtual_buffer_create_payload_t reservation = {
          .byte_length = 4096,
          .flags = flags,
          .source_pool_count = 1,
      };
      const iree_hal_replay_object_id_t source_pool_id = 11;
      const iree_const_byte_span_t payloads[] = {
          iree_make_const_byte_span(&reservation, sizeof(reservation)),
          iree_make_const_byte_span(&source_pool_id, sizeof(source_pool_id)),
      };
      const iree_hal_replay_file_record_metadata_t metadata = {
          .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
          .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_VIRTUAL_BUFFER_CREATE,
          .operation_code =
              IREE_HAL_REPLAY_OPERATION_CODE_VIRTUAL_BUFFER_CREATE,
      };
      builder.Append(metadata, IREE_ARRAYSIZE(payloads), payloads);
      auto storage = builder.Finish();
      auto* header =
          reinterpret_cast<iree_hal_replay_file_header_t*>(storage.data());
      header->version_minor = minor;
      for (auto format : {IREE_HAL_REPLAY_DUMP_FORMAT_TEXT,
                          IREE_HAL_REPLAY_DUMP_FORMAT_JSONL}) {
        const iree_hal_replay_dump_options_t options = {.format = format};
        std::string output;
        auto status = DumpReplayToString(MakeReplayFileContents(storage),
                                         &options, &output);
        if (flags == 0 ||
            (flags == 1 &&
             minor >= IREE_HAL_REPLAY_VIRTUAL_BUFFER_FLAGS_VERSION_MINOR)) {
          IREE_EXPECT_OK(std::move(status));
        } else {
          IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, std::move(status));
        }
      }
    }
  }
}

}  // namespace
