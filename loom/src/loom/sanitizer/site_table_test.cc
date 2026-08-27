// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/sanitizer/site_table.h"

#include <cstring>

#include "iree/base/alignment.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/sanitizer/ops.h"

namespace loom {
namespace {

static uint16_t ReadU16(iree_const_byte_span_t data, iree_host_size_t offset) {
  return iree_unaligned_load_le_u16(data.data + offset);
}

static uint32_t ReadU32(iree_const_byte_span_t data, iree_host_size_t offset) {
  return iree_unaligned_load_le_u32(data.data + offset);
}

static iree_const_byte_span_t Record(iree_const_byte_span_t table,
                                     iree_host_size_t index) {
  return iree_make_const_byte_span(
      table.data + LOOM_SANITIZER_SITE_TABLE_HEADER_LENGTH +
          index * LOOM_SANITIZER_SITE_TABLE_RECORD_LENGTH,
      LOOM_SANITIZER_SITE_TABLE_RECORD_LENGTH);
}

static loom_sanitizer_site_payload_t MakePayload(
    iree_const_byte_span_t extension_data = iree_const_byte_span_empty()) {
  loom_sanitizer_site_payload_t payload = {};
  payload.site_kind = LOOM_SANITIZER_SITE_KIND_ACCESS;
  payload.check_kind = LOOM_SANITIZER_CHECK_KIND_ACCESS_RANGE;
  payload.provenance_kind = LOOM_SANITIZER_PROVENANCE_KIND_ANALYSIS;
  payload.lane_policy = LOOM_SANITIZER_LANE_POLICY_ANY_LANE;
  payload.lineage_role = LOOM_SANITIZER_LINEAGE_ROLE_ORIGINAL;
  payload.flags = 0x1234u;
  payload.extension_data = extension_data;
  return payload;
}

class SiteTableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_location_id_t AddFileLocation(iree_string_view_t source_name,
                                     uint16_t start_line, uint16_t start_column,
                                     uint16_t end_line, uint16_t end_column) {
    loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
    IREE_CHECK_OK(
        loom_module_register_source(module_, source_name, &source_id));
    loom_location_id_t location_id = LOOM_LOCATION_UNKNOWN;
    IREE_CHECK_OK(loom_module_add_location(
        module_,
        loom_location_file_range(source_id, start_line, start_column, end_line,
                                 end_column),
        &location_id));
    return location_id;
  }

  loom_location_id_t AddTaggedLocation(loom_location_tag_t tag,
                                       loom_location_id_t child) {
    const uint8_t payload[] = {0x7a};
    uint8_t* stored_payload = NULL;
    IREE_CHECK_OK(iree_arena_allocate(&module_->arena, sizeof(payload),
                                      (void**)&stored_payload));
    memcpy(stored_payload, payload, sizeof(payload));
    loom_location_id_t location_id = LOOM_LOCATION_UNKNOWN;
    IREE_CHECK_OK(loom_module_add_location(
        module_,
        loom_location_tagged(tag, child, stored_payload, sizeof(payload)),
        &location_id));
    return location_id;
  }

  loom_location_id_t AddFusedLocation(loom_location_id_t first_child,
                                      loom_location_id_t second_child) {
    loom_location_id_t* children = NULL;
    IREE_CHECK_OK(iree_arena_allocate_array(
        &module_->arena, 2, sizeof(*children), (void**)&children));
    children[0] = first_child;
    children[1] = second_child;
    loom_location_entry_t entry = {};
    entry.kind = LOOM_LOCATION_FUSED;
    entry.fused.count = 2;
    entry.fused.children = children;
    loom_location_id_t location_id = LOOM_LOCATION_UNKNOWN;
    IREE_CHECK_OK(loom_module_add_location(module_, entry, &location_id));
    return location_id;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  iree_arena_allocator_t arena_;
};

TEST_F(SiteTableTest, EncodesRecordsPayloadsSourcesAndUnknownLocations) {
  constexpr iree_host_size_t kModelSourceLength = sizeof("model.loom") - 1;
  const uint8_t extension_bytes[] = {0xde, 0xad, 0xbe};
  const loom_sanitizer_site_payload_t payload = MakePayload(
      iree_make_const_byte_span(extension_bytes, sizeof(extension_bytes)));
  loom_location_id_t source_location =
      AddFileLocation(IREE_SV("model.loom"), 12, 3, 12, 19);

  loom_sanitizer_site_row_t rows[3] = {};
  rows[0] = (loom_sanitizer_site_row_t){
      /*.site_id=*/0,
      /*.op=*/nullptr,
      /*.op_kind=*/LOOM_OP_SANITIZER_ASSERT_ACCESS,
      /*.location=*/source_location,
      /*.payload_location=*/source_location,
      /*.source_location=*/source_location,
      /*.flags=*/LOOM_SANITIZER_SITE_ROW_HAS_PAYLOAD,
      /*.payload=*/payload,
  };
  rows[1] = (loom_sanitizer_site_row_t){
      /*.site_id=*/1,
      /*.op=*/nullptr,
      /*.op_kind=*/LOOM_OP_SANITIZER_ASSERT_VALUE,
      /*.location=*/source_location,
      /*.payload_location=*/LOOM_LOCATION_UNKNOWN,
      /*.source_location=*/source_location,
      /*.flags=*/0,
      /*.payload=*/{},
  };
  rows[2] = (loom_sanitizer_site_row_t){
      /*.site_id=*/2,
      /*.op=*/nullptr,
      /*.op_kind=*/LOOM_OP_SANITIZER_ASSERT_OP,
      /*.location=*/LOOM_LOCATION_UNKNOWN,
      /*.payload_location=*/LOOM_LOCATION_UNKNOWN,
      /*.source_location=*/LOOM_LOCATION_UNKNOWN,
      /*.flags=*/0,
      /*.payload=*/{},
  };
  loom_sanitizer_site_collection_t collection = {
      /*.rows=*/rows,
      /*.row_count=*/IREE_ARRAYSIZE(rows),
  };

  iree_const_byte_span_t table = iree_const_byte_span_empty();
  IREE_ASSERT_OK(
      loom_sanitizer_site_table_encode(module_, &collection, &arena_, &table));

  const iree_host_size_t expected_payload_length =
      LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH + sizeof(extension_bytes);
  const iree_host_size_t expected_string_table_offset =
      LOOM_SANITIZER_SITE_TABLE_HEADER_LENGTH +
      IREE_ARRAYSIZE(rows) * LOOM_SANITIZER_SITE_TABLE_RECORD_LENGTH;
  const iree_host_size_t expected_string_table_length = kModelSourceLength + 1;
  const iree_host_size_t expected_payload_offset =
      expected_string_table_offset + expected_string_table_length;
  ASSERT_EQ(table.data_length,
            expected_payload_offset + expected_payload_length);

  EXPECT_EQ(ReadU32(table, LOOM_SANITIZER_SITE_TABLE_HEADER_MAGIC_OFFSET),
            LOOM_SANITIZER_SITE_TABLE_MAGIC);
  EXPECT_EQ(table.data[LOOM_SANITIZER_SITE_TABLE_HEADER_VERSION_OFFSET],
            LOOM_SANITIZER_SITE_TABLE_VERSION);
  EXPECT_EQ(table.data[LOOM_SANITIZER_SITE_TABLE_HEADER_HEADER_LENGTH_OFFSET],
            LOOM_SANITIZER_SITE_TABLE_HEADER_LENGTH);
  EXPECT_EQ(
      ReadU16(table, LOOM_SANITIZER_SITE_TABLE_HEADER_RECORD_LENGTH_OFFSET),
      LOOM_SANITIZER_SITE_TABLE_RECORD_LENGTH);
  EXPECT_EQ(ReadU32(table, LOOM_SANITIZER_SITE_TABLE_HEADER_ROW_COUNT_OFFSET),
            IREE_ARRAYSIZE(rows));
  EXPECT_EQ(
      ReadU32(table,
              LOOM_SANITIZER_SITE_TABLE_HEADER_STRING_TABLE_OFFSET_OFFSET),
      expected_string_table_offset);
  EXPECT_EQ(
      ReadU32(table,
              LOOM_SANITIZER_SITE_TABLE_HEADER_STRING_TABLE_LENGTH_OFFSET),
      expected_string_table_length);
  EXPECT_EQ(
      ReadU32(table,
              LOOM_SANITIZER_SITE_TABLE_HEADER_PAYLOAD_DATA_OFFSET_OFFSET),
      expected_payload_offset);
  EXPECT_EQ(
      ReadU32(table,
              LOOM_SANITIZER_SITE_TABLE_HEADER_PAYLOAD_DATA_LENGTH_OFFSET),
      expected_payload_length);

  ASSERT_EQ(memcmp(table.data + expected_string_table_offset, "model.loom",
                   kModelSourceLength),
            0);
  EXPECT_EQ(table.data[expected_string_table_offset + kModelSourceLength], 0);

  iree_const_byte_span_t first_record = Record(table, 0);
  EXPECT_EQ(
      ReadU32(first_record, LOOM_SANITIZER_SITE_TABLE_RECORD_SITE_ID_OFFSET),
      0u);
  EXPECT_EQ(
      ReadU32(first_record, LOOM_SANITIZER_SITE_TABLE_RECORD_OP_KIND_OFFSET),
      LOOM_OP_SANITIZER_ASSERT_ACCESS);
  EXPECT_EQ(
      ReadU32(first_record, LOOM_SANITIZER_SITE_TABLE_RECORD_FLAGS_OFFSET),
      LOOM_SANITIZER_SITE_TABLE_RECORD_HAS_PAYLOAD |
          LOOM_SANITIZER_SITE_TABLE_RECORD_HAS_SOURCE_LOCATION);
  EXPECT_EQ(ReadU32(first_record,
                    LOOM_SANITIZER_SITE_TABLE_RECORD_PAYLOAD_OFFSET_OFFSET),
            0u);
  EXPECT_EQ(ReadU32(first_record,
                    LOOM_SANITIZER_SITE_TABLE_RECORD_PAYLOAD_LENGTH_OFFSET),
            expected_payload_length);
  EXPECT_EQ(ReadU32(first_record,
                    LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_NAME_OFFSET_OFFSET),
            0u);
  EXPECT_EQ(ReadU32(first_record,
                    LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_NAME_LENGTH_OFFSET),
            kModelSourceLength);
  EXPECT_EQ(
      ReadU32(first_record, LOOM_SANITIZER_SITE_TABLE_RECORD_START_LINE_OFFSET),
      12u);
  EXPECT_EQ(ReadU32(first_record,
                    LOOM_SANITIZER_SITE_TABLE_RECORD_START_COLUMN_OFFSET),
            3u);
  EXPECT_EQ(
      ReadU32(first_record, LOOM_SANITIZER_SITE_TABLE_RECORD_END_LINE_OFFSET),
      12u);
  EXPECT_EQ(
      ReadU32(first_record, LOOM_SANITIZER_SITE_TABLE_RECORD_END_COLUMN_OFFSET),
      19u);
  EXPECT_EQ(ReadU16(first_record,
                    LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_KIND_OFFSET),
            LOOM_SANITIZER_SITE_TABLE_SOURCE_KIND_FILE);

  uint8_t encoded_payload[LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH +
                          sizeof(extension_bytes)] = {};
  iree_host_size_t encoded_length = 0;
  IREE_ASSERT_OK(loom_sanitizer_site_payload_encode(
      &payload, iree_make_byte_span(encoded_payload, sizeof(encoded_payload)),
      &encoded_length));
  ASSERT_EQ(encoded_length, expected_payload_length);
  EXPECT_EQ(memcmp(table.data + expected_payload_offset, encoded_payload,
                   expected_payload_length),
            0);

  iree_const_byte_span_t second_record = Record(table, 1);
  EXPECT_EQ(
      ReadU32(second_record, LOOM_SANITIZER_SITE_TABLE_RECORD_SITE_ID_OFFSET),
      1u);
  EXPECT_EQ(
      ReadU32(second_record, LOOM_SANITIZER_SITE_TABLE_RECORD_FLAGS_OFFSET),
      LOOM_SANITIZER_SITE_TABLE_RECORD_HAS_SOURCE_LOCATION);
  EXPECT_EQ(ReadU32(second_record,
                    LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_NAME_OFFSET_OFFSET),
            0u);
  EXPECT_EQ(ReadU16(second_record,
                    LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_KIND_OFFSET),
            LOOM_SANITIZER_SITE_TABLE_SOURCE_KIND_FILE);

  iree_const_byte_span_t third_record = Record(table, 2);
  EXPECT_EQ(
      ReadU32(third_record, LOOM_SANITIZER_SITE_TABLE_RECORD_SITE_ID_OFFSET),
      2u);
  EXPECT_EQ(
      ReadU32(third_record, LOOM_SANITIZER_SITE_TABLE_RECORD_FLAGS_OFFSET), 0u);
  EXPECT_EQ(ReadU32(third_record,
                    LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_NAME_LENGTH_OFFSET),
            0u);
  EXPECT_EQ(ReadU16(third_record,
                    LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_KIND_OFFSET),
            LOOM_SANITIZER_SITE_TABLE_SOURCE_KIND_NONE);
}

TEST_F(SiteTableTest, ResolvesSourcesThroughTaggedAndFusedLocations) {
  constexpr iree_host_size_t kKernelSourceLength = sizeof("kernel.loom") - 1;
  loom_location_id_t file_location =
      AddFileLocation(IREE_SV("kernel.loom"), 21, 5, 23, 7);
  loom_location_id_t tagged_location = AddTaggedLocation(
      LOOM_LOCATION_TAG_TEMPLATE_INSTANTIATION, file_location);
  loom_location_id_t fused_location =
      AddFusedLocation(LOOM_LOCATION_UNKNOWN, tagged_location);

  loom_sanitizer_site_row_t row = {
      /*.site_id=*/0,
      /*.op=*/nullptr,
      /*.op_kind=*/LOOM_OP_SANITIZER_ASSERT_LAYOUT,
      /*.location=*/fused_location,
      /*.payload_location=*/LOOM_LOCATION_UNKNOWN,
      /*.source_location=*/LOOM_LOCATION_UNKNOWN,
      /*.flags=*/0,
      /*.payload=*/{},
  };
  loom_sanitizer_site_collection_t collection = {
      /*.rows=*/&row,
      /*.row_count=*/1,
  };

  iree_const_byte_span_t table = iree_const_byte_span_empty();
  IREE_ASSERT_OK(
      loom_sanitizer_site_table_encode(module_, &collection, &arena_, &table));

  iree_const_byte_span_t record = Record(table, 0);
  const uint32_t string_table_offset = ReadU32(
      table, LOOM_SANITIZER_SITE_TABLE_HEADER_STRING_TABLE_OFFSET_OFFSET);
  EXPECT_EQ(ReadU32(record, LOOM_SANITIZER_SITE_TABLE_RECORD_FLAGS_OFFSET),
            LOOM_SANITIZER_SITE_TABLE_RECORD_HAS_SOURCE_LOCATION);
  EXPECT_EQ(ReadU32(record,
                    LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_NAME_OFFSET_OFFSET),
            0u);
  EXPECT_EQ(ReadU32(record,
                    LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_NAME_LENGTH_OFFSET),
            kKernelSourceLength);
  EXPECT_EQ(memcmp(table.data + string_table_offset, "kernel.loom",
                   kKernelSourceLength),
            0);
  EXPECT_EQ(ReadU32(record, LOOM_SANITIZER_SITE_TABLE_RECORD_START_LINE_OFFSET),
            21u);
  EXPECT_EQ(
      ReadU32(record, LOOM_SANITIZER_SITE_TABLE_RECORD_START_COLUMN_OFFSET),
      5u);
  EXPECT_EQ(ReadU32(record, LOOM_SANITIZER_SITE_TABLE_RECORD_END_LINE_OFFSET),
            23u);
  EXPECT_EQ(ReadU32(record, LOOM_SANITIZER_SITE_TABLE_RECORD_END_COLUMN_OFFSET),
            7u);
}

}  // namespace
}  // namespace loom
