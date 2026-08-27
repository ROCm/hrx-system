// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/source_context.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::testing::status::StatusIs;

enum TestSiteTableHeaderOffset {
  kHeaderMagicOffset = 0,
  kHeaderVersionOffset = 4,
  kHeaderHeaderLengthOffset = 5,
  kHeaderRecordLengthOffset = 6,
  kHeaderRowCountOffset = 8,
  kHeaderStringTableOffsetOffset = 12,
  kHeaderStringTableLengthOffset = 16,
  kHeaderPayloadDataOffsetOffset = 20,
  kHeaderPayloadDataLengthOffset = 24,
};

enum TestSiteTableRecordOffset {
  kRecordSiteIdOffset = 0,
  kRecordOpKindOffset = 4,
  kRecordFlagsOffset = 8,
  kRecordPayloadOffsetOffset = 12,
  kRecordPayloadLengthOffset = 16,
  kRecordSourceNameOffsetOffset = 20,
  kRecordSourceNameLengthOffset = 24,
  kRecordStartLineOffset = 28,
  kRecordStartColumnOffset = 32,
  kRecordEndLineOffset = 36,
  kRecordEndColumnOffset = 40,
  kRecordSourceKindOffset = 44,
};

constexpr uint32_t kSiteTableMagic = 0x5449534Cu;
constexpr uint8_t kSiteTableVersion = 1u;
constexpr uint8_t kSiteTableHeaderLength = 32u;
constexpr uint16_t kSiteTableRecordLength = 48u;
constexpr uint32_t kRecordHasPayload = 1u << 0;
constexpr uint32_t kRecordHasSourceLocation = 1u << 1;
constexpr uint16_t kSourceKindFile = 1u;
constexpr uint32_t kLoomOpSanitizerAssertAccess = (0x1Du << 8) | 0u;
constexpr uint32_t kLoomOpSanitizerRaceAccess = (0x1Du << 8) | 4u;

static void StoreU8(std::vector<uint8_t>* data, size_t offset, uint8_t value) {
  (*data)[offset] = value;
}

static void StoreU16(std::vector<uint8_t>* data, size_t offset,
                     uint16_t value) {
  iree_unaligned_store_le_u16(
      reinterpret_cast<uint16_t*>(data->data() + offset), value);
}

static void StoreU32(std::vector<uint8_t>* data, size_t offset,
                     uint32_t value) {
  iree_unaligned_store_le_u32(
      reinterpret_cast<uint32_t*>(data->data() + offset), value);
}

static std::vector<uint8_t> MakeSingleSiteTable(
    uint32_t op_kind = kLoomOpSanitizerAssertAccess) {
  constexpr char kSourceFile[] = "model/layer.loom";
  const std::vector<uint8_t> payload = {0xA5, 0x5A};
  const uint32_t string_table_offset =
      kSiteTableHeaderLength + kSiteTableRecordLength;
  const uint32_t string_table_length = sizeof(kSourceFile);
  const uint32_t payload_data_offset =
      string_table_offset + string_table_length;
  const uint32_t payload_data_length = static_cast<uint32_t>(payload.size());

  std::vector<uint8_t> table(payload_data_offset + payload_data_length, 0);
  StoreU32(&table, kHeaderMagicOffset, kSiteTableMagic);
  StoreU8(&table, kHeaderVersionOffset, kSiteTableVersion);
  StoreU8(&table, kHeaderHeaderLengthOffset, kSiteTableHeaderLength);
  StoreU16(&table, kHeaderRecordLengthOffset, kSiteTableRecordLength);
  StoreU32(&table, kHeaderRowCountOffset, 1);
  StoreU32(&table, kHeaderStringTableOffsetOffset, string_table_offset);
  StoreU32(&table, kHeaderStringTableLengthOffset, string_table_length);
  StoreU32(&table, kHeaderPayloadDataOffsetOffset, payload_data_offset);
  StoreU32(&table, kHeaderPayloadDataLengthOffset, payload_data_length);

  const uint32_t record_offset = kSiteTableHeaderLength;
  StoreU32(&table, record_offset + kRecordSiteIdOffset, 0);
  StoreU32(&table, record_offset + kRecordOpKindOffset, op_kind);
  StoreU32(&table, record_offset + kRecordFlagsOffset,
           kRecordHasPayload | kRecordHasSourceLocation);
  StoreU32(&table, record_offset + kRecordPayloadOffsetOffset, 0);
  StoreU32(&table, record_offset + kRecordPayloadLengthOffset,
           payload_data_length);
  StoreU32(&table, record_offset + kRecordSourceNameOffsetOffset, 0);
  StoreU32(&table, record_offset + kRecordSourceNameLengthOffset,
           sizeof(kSourceFile) - 1);
  StoreU32(&table, record_offset + kRecordStartLineOffset, 7);
  StoreU32(&table, record_offset + kRecordStartColumnOffset, 3);
  StoreU32(&table, record_offset + kRecordEndLineOffset, 7);
  StoreU32(&table, record_offset + kRecordEndColumnOffset, 9);
  StoreU16(&table, record_offset + kRecordSourceKindOffset, kSourceKindFile);

  std::memcpy(table.data() + string_table_offset, kSourceFile,
              sizeof(kSourceFile));
  std::memcpy(table.data() + payload_data_offset, payload.data(),
              payload.size());
  return table;
}

static void InitializeContext(
    iree_hal_amdgpu_loaded_code_object_range_t* ranges,
    iree_host_size_t range_count,
    iree_hal_amdgpu_source_context_t* out_context) {
  const uint64_t code_object_hash[2] = {0x1234u, 0x5678u};
  iree_hal_amdgpu_source_context_initialize(
      /*executable_id=*/0x123u, code_object_hash, range_count,
      /*loaded_physical_device_mask=*/range_count ? 1u : 0u, ranges,
      out_context);
}

TEST(SourceContextTest, TranslatesLoadedCodeObjectDeviceSpans) {
  uint8_t host_storage[16] = {};
  iree_hal_amdgpu_loaded_code_object_range_t ranges[2] = {};
  iree_hal_amdgpu_source_context_t context;
  InitializeContext(ranges, 2, &context);

  iree_hal_amdgpu_loaded_code_object_range_t range = {};
  range.host_pointer = host_storage;
  range.device_pointer = 0x1000u;
  range.byte_length = sizeof(host_storage);
  IREE_ASSERT_OK(iree_hal_amdgpu_source_context_set_loaded_code_object_range(
      &context, /*physical_device_ordinal=*/1, range));

  iree_const_byte_span_t host_span = iree_const_byte_span_empty();
  EXPECT_TRUE(iree_hal_amdgpu_source_context_try_translate_device_span(
      &context, /*physical_device_ordinal=*/1, /*device_pointer=*/0x1004u,
      /*byte_length=*/3, &host_span));
  EXPECT_EQ(host_span.data, host_storage + 4);
  EXPECT_EQ(host_span.data_length, 3u);

  EXPECT_FALSE(iree_hal_amdgpu_source_context_try_translate_device_span(
      &context, /*physical_device_ordinal=*/1, /*device_pointer=*/0x100fu,
      /*byte_length=*/2, &host_span));
}

TEST(SourceContextTest, ResolvesSanitizerSiteTableRecord) {
  std::vector<uint8_t> table = MakeSingleSiteTable();
  iree_hal_amdgpu_source_context_t context;
  InitializeContext(nullptr, 0, &context);

  iree_hal_device_event_site_t site = iree_hal_device_event_site_default();
  EXPECT_FALSE(iree_hal_amdgpu_source_context_try_resolve_sanitizer_site(
      &context, /*site_id=*/0, &site));

  IREE_ASSERT_OK(iree_hal_amdgpu_source_context_set_sanitizer_site_table(
      &context, iree_make_const_byte_span(table.data(), table.size())));

  ASSERT_TRUE(iree_hal_amdgpu_source_context_try_resolve_sanitizer_site(
      &context, /*site_id=*/0, &site));
  EXPECT_EQ(site.record_length, sizeof(site));
  EXPECT_EQ(site.abi_version, IREE_HAL_DEVICE_EVENT_SITE_ABI_VERSION_0);
  EXPECT_EQ(site.site_id, 0u);
  EXPECT_TRUE(
      iree_string_view_equal(site.source_file, IREE_SV("model/layer.loom")));
  EXPECT_EQ(site.start_line, 7u);
  EXPECT_EQ(site.start_column, 3u);
  EXPECT_EQ(site.end_line, 7u);
  EXPECT_EQ(site.end_column, 9u);
  EXPECT_TRUE(iree_string_view_equal(site.operation_name,
                                     IREE_SV("sanitizer.assert.access")));
  ASSERT_EQ(site.producer_payload.data_length, 2u);
  EXPECT_EQ(site.producer_payload.data[0], 0xA5u);
  EXPECT_EQ(site.producer_payload.data[1], 0x5Au);

  EXPECT_FALSE(iree_hal_amdgpu_source_context_try_resolve_sanitizer_site(
      &context, /*site_id=*/1, &site));
}

TEST(SourceContextTest, ResolvesRaceSanitizerSiteName) {
  std::vector<uint8_t> table = MakeSingleSiteTable(kLoomOpSanitizerRaceAccess);
  iree_hal_amdgpu_source_context_t context;
  InitializeContext(nullptr, 0, &context);
  IREE_ASSERT_OK(iree_hal_amdgpu_source_context_set_sanitizer_site_table(
      &context, iree_make_const_byte_span(table.data(), table.size())));

  iree_hal_device_event_site_t site = iree_hal_device_event_site_default();
  ASSERT_TRUE(iree_hal_amdgpu_source_context_try_resolve_sanitizer_site(
      &context, /*site_id=*/0, &site));
  EXPECT_TRUE(iree_string_view_equal(site.operation_name,
                                     IREE_SV("sanitizer.race.access")));
}

TEST(SourceContextTest, RejectsMalformedSanitizerSiteTable) {
  std::vector<uint8_t> table = MakeSingleSiteTable();
  StoreU32(&table, kHeaderMagicOffset, 0xFFFFFFFFu);

  iree_hal_amdgpu_source_context_t context;
  InitializeContext(nullptr, 0, &context);
  EXPECT_THAT(
      Status(iree_hal_amdgpu_source_context_set_sanitizer_site_table(
          &context, iree_make_const_byte_span(table.data(), table.size()))),
      StatusIs(StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace iree::hal::amdgpu
