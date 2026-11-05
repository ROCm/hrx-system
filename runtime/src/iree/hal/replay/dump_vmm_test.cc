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
      .operation_code = operation_code,
  };
  return metadata;
}

static std::vector<uint8_t> MakeVmmReplayFileStorage() {
  ReplayFileBuilder builder(/*capacity=*/8192);

  iree_hal_replay_file_record_metadata_t physical_object_metadata = {
      .sequence_ordinal = 0,
      .object_id = 23,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT,
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_PHYSICAL_MEMORY,
  };
  builder.Append(physical_object_metadata, 0, nullptr);

  iree_hal_replay_allocator_virtual_memory_reserve_payload_t reserve = {
      .queue_family_affinity = 3,
      .size = 4096,
  };
  builder.Append(
      MakeAllocatorOperationMetadata(
          1, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_RESERVE,
          IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RESERVE, 17),
      reserve);

  iree_hal_replay_allocator_virtual_memory_release_payload_t release = {
      .virtual_buffer_id = 17,
  };
  builder.Append(
      MakeAllocatorOperationMetadata(
          2, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE,
          IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE, 17),
      release);

  iree_hal_replay_allocator_physical_memory_allocate_payload_t allocate = {
      .allocation_size = 8192,
      .queue_family_affinity = 5,
      .usage = 0x10,
      .type = 0x20,
  };
  builder.Append(
      MakeAllocatorOperationMetadata(
          3, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE,
          IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE,
          23),
      allocate);

  iree_hal_replay_allocator_physical_memory_free_payload_t free = {
      .physical_memory_id = 23,
  };
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
      .size = 1024,
  };
  builder.Append(
      MakeAllocatorOperationMetadata(
          5, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_MAP,
          IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_MAP, 17),
      map);

  iree_hal_replay_allocator_virtual_memory_unmap_payload_t unmap = {
      .virtual_buffer_id = 17,
      .virtual_offset = 128,
      .size = 1024,
  };
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
      .protection = 5,
  };
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
      .advice = 6,
  };
  builder.Append(
      MakeAllocatorOperationMetadata(
          8, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_ADVISE,
          IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_ADVISE, 17),
      advise);

  return builder.Finish();
}

TEST(ReplayDumpTest, DefinesVmmWireSchema) {
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
      32u,
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

  iree_hal_replay_allocator_physical_memory_allocate_payload_t allocate = {
      .reserved0 = 1,
  };
  expect_rejected(
      IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE,
      allocate);

  iree_hal_replay_allocator_virtual_memory_protect_payload_t protect = {
      .reserved0 = 1,
  };
  expect_rejected(
      IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT,
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT, protect);
}

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
