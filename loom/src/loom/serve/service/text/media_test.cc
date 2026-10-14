// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/service/text/chat.h"

namespace {

constexpr iree_host_size_t kMarkerRecordBytes =
    LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_COUNT * sizeof(uint32_t);
constexpr iree_host_size_t kSpanRecordBytes =
    LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_COUNT * sizeof(uint32_t);

void StoreField(std::vector<uint8_t>& bytes, iree_host_size_t record,
                iree_host_size_t record_bytes, iree_host_size_t field,
                uint32_t value) {
  iree_unaligned_store_le_u32(
      bytes.data() + record * record_bytes + field * sizeof(uint32_t), value);
}

uint32_t LoadField(iree_const_byte_span_t bytes, iree_host_size_t record,
                   iree_host_size_t record_bytes, iree_host_size_t field) {
  return iree_unaligned_load_le_u32(bytes.data + record * record_bytes +
                                    field * sizeof(uint32_t));
}

class TextMediaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    chat_.host_allocator = iree_allocator_system();
    chat_.images.count = images_.size();
    chat_.images.capacity = images_.size();
    chat_.images.values = images_.data();
    iree_string_builder_initialize(iree_allocator_system(), &chat_.prompt);
    iree_string_builder_initialize(iree_allocator_system(),
                                   &chat_.images.metadata);
  }

  void TearDown() override {
    iree_string_builder_deinitialize(&chat_.images.metadata);
    iree_string_builder_deinitialize(&chat_.prompt);
  }

  std::vector<uint8_t> MessagePlan() {
    std::vector<uint8_t> plan(2 * kMarkerRecordBytes + 3);
    StoreField(plan, 0, kMarkerRecordBytes,
               LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_IMAGE_INDEX, 0);
    StoreField(plan, 0, kMarkerRecordBytes,
               LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_BYTE_OFFSET, 1);
    StoreField(plan, 0, kMarkerRecordBytes,
               LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_BYTE_LENGTH, 2);
    StoreField(plan, 0, kMarkerRecordBytes,
               LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_FEATURE_COUNT, 3);
    StoreField(plan, 0, kMarkerRecordBytes,
               LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_METADATA_OFFSET,
               2 * kMarkerRecordBytes);
    StoreField(plan, 0, kMarkerRecordBytes,
               LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_METADATA_LENGTH, 2);
    StoreField(plan, 1, kMarkerRecordBytes,
               LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_IMAGE_INDEX, 1);
    StoreField(plan, 1, kMarkerRecordBytes,
               LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_BYTE_OFFSET, 5);
    StoreField(plan, 1, kMarkerRecordBytes,
               LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_BYTE_LENGTH, 1);
    StoreField(plan, 1, kMarkerRecordBytes,
               LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_FEATURE_COUNT, 4);
    StoreField(plan, 1, kMarkerRecordBytes,
               LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_METADATA_OFFSET,
               2 * kMarkerRecordBytes + 2);
    StoreField(plan, 1, kMarkerRecordBytes,
               LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_METADATA_LENGTH, 1);
    plan[2 * kMarkerRecordBytes] = 'a';
    plan[2 * kMarkerRecordBytes + 1] = 'b';
    plan[2 * kMarkerRecordBytes + 2] = 'c';
    return plan;
  }

  void ApplyMessagePlan() {
    const std::vector<uint8_t> plan = MessagePlan();
    IREE_ASSERT_OK(loom_serve_text_media_apply_message_plan(
        &chat_, 0, 10, 6, iree_make_const_byte_span(plan.data(), plan.size())));
  }

  std::array<loom_serve_text_chat_image_t, 2> images_ = {};
  loom_serve_text_chat_t chat_ = {};
};

TEST_F(TextMediaTest, CommitsExactSourceSemanticsToMediaIdentity) {
  ApplyMessagePlan();

  EXPECT_EQ(images_[0].marker_offset, 11u);
  EXPECT_EQ(images_[0].marker_length, 2u);
  EXPECT_EQ(images_[0].feature_count, 3u);
  EXPECT_EQ(images_[0].metadata_offset, 0u);
  EXPECT_EQ(images_[0].metadata_length, 2u);
  EXPECT_EQ(images_[1].marker_offset, 15u);
  EXPECT_EQ(images_[1].feature_count, 4u);
  EXPECT_EQ(images_[1].metadata_offset, 2u);
  EXPECT_TRUE(iree_string_view_equal(
      iree_string_builder_view(&chat_.images.metadata), IREE_SV("abc")));

  loom_serve_text_chat_media_identity_t identity = {};
  IREE_ASSERT_OK(loom_serve_text_chat_media_identity_initialize(
      &chat_, iree_allocator_system(), &identity));
  EXPECT_TRUE(
      loom_serve_text_chat_media_identity_equal_chat(&identity, &chat_));
  ++images_[0].feature_count;
  EXPECT_FALSE(
      loom_serve_text_chat_media_identity_equal_chat(&identity, &chat_));
  --images_[0].feature_count;
  images_[1].metadata_digest.bytes[0] ^= 1;
  EXPECT_FALSE(
      loom_serve_text_chat_media_identity_equal_chat(&identity, &chat_));
  loom_serve_text_chat_media_identity_deinitialize(&identity);
}

TEST_F(TextMediaTest, RejectsMalformedMessagePlanBeforeMutation) {
  std::vector<uint8_t> plan = MessagePlan();
  StoreField(plan, 0, kMarkerRecordBytes,
             LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_BYTE_LENGTH, 0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_media_apply_message_plan(
          &chat_, 0, 10, 6,
          iree_make_const_byte_span(plan.data(), plan.size())));

  plan = MessagePlan();
  StoreField(plan, 1, kMarkerRecordBytes,
             LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_BYTE_OFFSET, 2);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_media_apply_message_plan(
          &chat_, 0, 10, 6,
          iree_make_const_byte_span(plan.data(), plan.size())));

  plan = MessagePlan();
  StoreField(plan, 1, kMarkerRecordBytes,
             LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_METADATA_OFFSET, 0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_media_apply_message_plan(
          &chat_, 0, 10, 6,
          iree_make_const_byte_span(plan.data(), plan.size())));

  EXPECT_EQ(iree_string_builder_size(&chat_.images.metadata), 0u);
  EXPECT_EQ(images_[0].marker_length, 0u);
  EXPECT_EQ(images_[1].feature_count, 0u);
}

TEST_F(TextMediaTest, BuildsSuffixPlanOnlyAtExactMarkerBoundaries) {
  IREE_ASSERT_OK(iree_string_builder_append_string(
      &chat_.prompt, IREE_SV("0123456789abcdefghijklmnop")));
  ApplyMessagePlan();

  iree_vm_buffer_t* plan = nullptr;
  iree_host_size_t marker_count = 0;
  IREE_ASSERT_OK(loom_serve_text_media_build_input_plan(
      &chat_, 13, iree_allocator_system(), &plan, &marker_count));
  EXPECT_EQ(marker_count, 1u);
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  IREE_ASSERT_OK(
      iree_vm_buffer_map_read(plan, 0, iree_vm_buffer_length(plan), &bytes));
  EXPECT_EQ(bytes.data_length, kMarkerRecordBytes + 1);
  EXPECT_EQ(LoadField(bytes, 0, kMarkerRecordBytes,
                      LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_IMAGE_INDEX),
            1u);
  EXPECT_EQ(LoadField(bytes, 0, kMarkerRecordBytes,
                      LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_BYTE_OFFSET),
            2u);
  EXPECT_EQ(LoadField(bytes, 0, kMarkerRecordBytes,
                      LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_METADATA_OFFSET),
            kMarkerRecordBytes);
  EXPECT_EQ(bytes.data[kMarkerRecordBytes], 'c');
  iree_vm_buffer_release(plan);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_media_build_input_plan(
          &chat_, 12, iree_allocator_system(), &plan, &marker_count));

  IREE_ASSERT_OK(loom_serve_text_media_build_input_plan(
      &chat_, 16, iree_allocator_system(), &plan, &marker_count));
  EXPECT_EQ(marker_count, 0u);
  EXPECT_EQ(iree_vm_buffer_length(plan), 0u);
  iree_vm_buffer_release(plan);
}

TEST(TextMediaSpanTest, ValidatesAndCompactsSourceRanges) {
  EXPECT_EQ(sizeof(loom_serve_text_media_span_t), 3 * sizeof(uint32_t));
  std::vector<uint8_t> markers(2 * kMarkerRecordBytes);
  StoreField(markers, 0, kMarkerRecordBytes,
             LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_IMAGE_INDEX, 2);
  StoreField(markers, 0, kMarkerRecordBytes,
             LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_FEATURE_COUNT, 3);
  StoreField(markers, 1, kMarkerRecordBytes,
             LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_IMAGE_INDEX, 3);
  StoreField(markers, 1, kMarkerRecordBytes,
             LOOM_SERVE_TEXT_MEDIA_MARKER_FIELD_FEATURE_COUNT, 4);
  std::vector<uint8_t> source_spans(2 * kSpanRecordBytes);
  StoreField(source_spans, 0, kSpanRecordBytes,
             LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_IMAGE_INDEX, 2);
  StoreField(source_spans, 0, kSpanRecordBytes,
             LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_INPUT_OFFSET, 1);
  StoreField(source_spans, 0, kSpanRecordBytes,
             LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_INPUT_COUNT, 3);
  StoreField(source_spans, 1, kSpanRecordBytes,
             LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_IMAGE_INDEX, 3);
  StoreField(source_spans, 1, kSpanRecordBytes,
             LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_INPUT_OFFSET, 4);
  StoreField(source_spans, 1, kSpanRecordBytes,
             LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_INPUT_COUNT, 4);

  loom_serve_text_media_span_t* spans = nullptr;
  iree_host_size_t span_count = 0;
  IREE_ASSERT_OK(loom_serve_text_media_parse_spans(
      iree_make_const_byte_span(markers.data(), markers.size()), 2,
      iree_make_const_byte_span(source_spans.data(), source_spans.size()), 8, 1,
      iree_allocator_system(), &spans, &span_count));
  ASSERT_EQ(span_count, 2u);
  EXPECT_EQ(spans[0].image_index, 2u);
  EXPECT_EQ(spans[0].input_offset, 2u);
  EXPECT_EQ(spans[0].input_count, 3u);
  EXPECT_EQ(spans[1].image_index, 3u);
  EXPECT_EQ(spans[1].input_offset, 5u);
  EXPECT_EQ(spans[1].input_count, 4u);
  iree_allocator_free(iree_allocator_system(), spans);

  StoreField(source_spans, 1, kSpanRecordBytes,
             LOOM_SERVE_TEXT_MEDIA_SPAN_FIELD_INPUT_OFFSET, 3);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_text_media_parse_spans(
          iree_make_const_byte_span(markers.data(), markers.size()), 2,
          iree_make_const_byte_span(source_spans.data(), source_spans.size()),
          8, 1, iree_allocator_system(), &spans, &span_count));
}

}  // namespace
