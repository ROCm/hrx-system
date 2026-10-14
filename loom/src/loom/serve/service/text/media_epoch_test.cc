// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/media_epoch.h"

#include <array>
#include <cstdint>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

constexpr iree_host_size_t kPlanRecordBytes =
    LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_COUNT * sizeof(uint32_t);

uint32_t LoadPlanField(iree_const_byte_span_t bytes, iree_host_size_t record,
                       loom_serve_text_media_epoch_field_t field) {
  return iree_unaligned_load_le_u32(bytes.data + record * kPlanRecordBytes +
                                    field * sizeof(uint32_t));
}

TEST(TextMediaEpochTest, TextOnlyEpochAllocatesNothing) {
  const loom_serve_text_media_epoch_span_t spans[] = {{
      .plan_index = 3,
      .input_offset = 8,
      .input_count = 4,
  }};
  loom_serve_text_media_epoch_t epoch = {};
  IREE_ASSERT_OK(loom_serve_text_media_epoch_initialize(
      IREE_ARRAYSIZE(spans), spans, 16, 0, nullptr, 0, iree_allocator_null(),
      &epoch));

  EXPECT_EQ(epoch.plan, nullptr);
  EXPECT_EQ(epoch.packet_records, nullptr);
  EXPECT_EQ(epoch.features, nullptr);
  EXPECT_EQ(epoch.device_length, 0u);
  loom_serve_text_media_epoch_deinitialize(&epoch);
}

TEST(TextMediaEpochTest, BuildsExactCrossChunkPlanAndPacketLayout) {
  const std::array<uint8_t, 4> row_0 = {0, 1, 2, 3};
  const std::array<uint8_t, 8> row_1 = {4, 5, 6, 7, 8, 9, 10, 11};
  const std::array<uint8_t, 4> row_2 = {12, 13, 14, 15};
  const iree_const_byte_span_t rows[] = {
      iree_make_const_byte_span(row_0.data(), row_0.size()),
      iree_make_const_byte_span(row_1.data(), row_1.size()),
      iree_make_const_byte_span(row_2.data(), row_2.size()),
  };
  const std::array<uint8_t, 6> metadata = {90, 91, 92, 93, 94, 95};
  const loom_serve_text_media_input_span_t input_spans[] = {{
      .input_offset = 3,
      .input_count = 3,
      .row_offset = 0,
      .metadata_offset = 1,
      .metadata_length = 4,
  }};
  const loom_serve_text_media_input_t media = {
      .spans = input_spans,
      .span_count = IREE_ARRAYSIZE(input_spans),
      .rows = rows,
      .row_count = IREE_ARRAYSIZE(rows),
      .metadata = iree_make_const_byte_span(metadata.data(), metadata.size()),
  };
  const loom_serve_text_media_epoch_span_t first_spans[] = {{
      .plan_index = 2,
      .input_offset = 0,
      .input_count = 4,
      .media = &media,
  }};
  const loom_serve_text_media_epoch_span_t next_spans[] = {{
      .plan_index = 16,
      .input_offset = 4,
      .input_count = 2,
      .media = &media,
  }};

  loom_serve_text_media_epoch_t epoch = {};
  IREE_ASSERT_OK(loom_serve_text_media_epoch_initialize(
      IREE_ARRAYSIZE(first_spans), first_spans, 8, IREE_ARRAYSIZE(next_spans),
      next_spans, 4, iree_allocator_system(), &epoch));

  EXPECT_EQ(epoch.first_feature_count, 1u);
  EXPECT_EQ(epoch.next_feature_count, 2u);
  EXPECT_EQ(epoch.first_packet_capacity, 8u);
  EXPECT_EQ(epoch.next_packet_capacity, 4u);
  EXPECT_EQ(epoch.packet_record_bytes.data_length,
            12u * LOOM_SERVE_TEXT_MEDIA_PACKET_RECORD_BYTES);
  EXPECT_EQ(epoch.device_length, 420u);
  ASSERT_NE(epoch.features, nullptr);
  EXPECT_EQ(epoch.features[0].source.data, row_0.data());
  EXPECT_EQ(epoch.features[0].target_offset, 384u);
  EXPECT_EQ(epoch.features[1].source.data, row_1.data());
  EXPECT_EQ(epoch.features[1].target_offset, 400u);
  EXPECT_EQ(epoch.features[2].source.data, row_2.data());
  EXPECT_EQ(epoch.features[2].target_offset, 416u);

  iree_const_byte_span_t plan = iree_const_byte_span_empty();
  IREE_ASSERT_OK(iree_vm_buffer_map_read(
      epoch.plan, 0, iree_vm_buffer_length(epoch.plan), &plan));
  ASSERT_EQ(plan.data_length, 3u * kPlanRecordBytes + 8u);
  EXPECT_EQ(
      LoadPlanField(plan, 0, LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_PLAN_INDEX), 2u);
  EXPECT_EQ(LoadPlanField(plan, 0,
                          LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_SPAN_INPUT_OFFSET),
            3u);
  EXPECT_EQ(
      LoadPlanField(plan, 0, LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_FEATURE_OFFSET),
      384u);
  EXPECT_EQ(
      LoadPlanField(plan, 0, LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_FEATURE_LENGTH),
      row_0.size());
  EXPECT_EQ(
      LoadPlanField(plan, 0, LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_METADATA_OFFSET),
      3u * kPlanRecordBytes);
  EXPECT_EQ(
      LoadPlanField(plan, 0, LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_METADATA_LENGTH),
      4u);

  for (iree_host_size_t record = 1; record < 3; ++record) {
    EXPECT_EQ(LoadPlanField(plan, record,
                            LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_PLAN_INDEX),
              16u);
    EXPECT_EQ(
        LoadPlanField(plan, record,
                      LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_SPAN_INPUT_OFFSET),
        record - 1);
    EXPECT_EQ(LoadPlanField(plan, record,
                            LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_METADATA_OFFSET),
              3u * kPlanRecordBytes + 4u);
    EXPECT_EQ(LoadPlanField(plan, record,
                            LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_METADATA_LENGTH),
              4u);
  }
  EXPECT_EQ(
      LoadPlanField(plan, 1, LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_FEATURE_OFFSET),
      400u);
  EXPECT_EQ(
      LoadPlanField(plan, 1, LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_FEATURE_LENGTH),
      row_1.size());
  EXPECT_EQ(
      LoadPlanField(plan, 2, LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_FEATURE_OFFSET),
      416u);
  EXPECT_EQ(
      LoadPlanField(plan, 2, LOOM_SERVE_TEXT_MEDIA_EPOCH_FIELD_FEATURE_LENGTH),
      row_2.size());
  EXPECT_EQ(
      std::memcmp(plan.data + 3 * kPlanRecordBytes, metadata.data() + 1, 4), 0);
  EXPECT_EQ(
      std::memcmp(plan.data + 3 * kPlanRecordBytes + 4, metadata.data() + 1, 4),
      0);

  loom_serve_text_media_epoch_deinitialize(&epoch);
}

}  // namespace
