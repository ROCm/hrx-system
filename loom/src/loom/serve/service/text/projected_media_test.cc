// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/projected_media.h"

#include <array>
#include <initializer_list>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

loom_serve_finite_result_t MakeResult(std::initializer_list<uint32_t> rows) {
  const iree_allocator_t allocator = iree_allocator_system();
  const iree_host_size_t row_count = rows.size();
  loom_serve_finite_result_t result = {
      .host_allocator = allocator,
      .request_count = 1,
  };
  const iree_host_size_t plan_length = 32 + 24 + row_count * 16;
  IREE_CHECK_OK(iree_vm_buffer_create(plan_length, 8, allocator, &result.plan));
  iree_byte_span_t plan;
  IREE_CHECK_OK(iree_vm_buffer_map_write(result.plan, 0, plan_length, &plan));
  result.records = iree_make_const_byte_span(plan.data, plan.data_length);
  iree_unaligned_store_le_u64(plan.data + 16, 1);
  iree_unaligned_store_le_u64(plan.data + 24, row_count);
  iree_unaligned_store_le_u64(plan.data + 32, 0);
  iree_unaligned_store_le_u64(plan.data + 40, row_count);
  IREE_CHECK_OK(iree_allocator_malloc_array(
      allocator, row_count, sizeof(uint32_t), (void**)&result.feedback.data));
  result.feedback.data_length = row_count * sizeof(uint32_t);
  iree_host_size_t i = 0;
  for (uint32_t value : rows) {
    uint8_t* record = plan.data + 56 + i * 16;
    iree_unaligned_store_le_u64(record, i * sizeof(uint32_t));
    iree_unaligned_store_le_u64(record + 8, 1);
    iree_unaligned_store_le_u32(result.feedback.data + i * sizeof(uint32_t),
                                value);
    ++i;
  }
  return result;
}

TEST(ProjectedMediaTest, OwnsAndIncrementallyReleasesExactImageResults) {
  std::array<loom_serve_text_chat_image_t, 3> images = {{
      {.metadata_offset = 0, .metadata_length = 1},
      {.metadata_offset = 1, .metadata_length = 2},
      {.metadata_offset = 3, .metadata_length = 1},
  }};
  loom_serve_text_chat_t chat = {};
  chat.images.values = images.data();
  chat.images.count = images.size();
  iree_string_builder_initialize(iree_allocator_system(),
                                 &chat.images.metadata);
  IREE_ASSERT_OK(
      iree_string_builder_append_cstring(&chat.images.metadata, "abcd"));
  loom_serve_text_media_span_t spans[] = {
      {.image_index = 0, .input_offset = 0, .input_count = 1},
      {.image_index = 1, .input_offset = 2, .input_count = 2},
      {.image_index = 2, .input_offset = 8, .input_count = 1},
  };
  loom_serve_text_input_t input = {
      .media_spans = spans,
      .media_span_count = IREE_ARRAYSIZE(spans),
  };
  loom_serve_text_projected_media_t media = {};
  IREE_ASSERT_OK(loom_serve_text_projected_media_initialize(
      &chat, &input, 2, iree_allocator_system(), &media));
  ASSERT_EQ(media.result_count, 2u);
  EXPECT_EQ(media.image_indices[0], 1u);
  EXPECT_EQ(media.image_indices[1], 2u);
  EXPECT_EQ(media.spans[0].row_offset, 0u);
  EXPECT_EQ(media.spans[1].row_offset, 2u);
  EXPECT_EQ(media.input.metadata.data_length, 4u);

  auto first = MakeResult({17, 18});
  auto second = MakeResult({29});
  loom_serve_text_projected_media_set_result(&media, 0, &first);
  loom_serve_text_projected_media_set_result(&media, 1, &second);
  EXPECT_EQ(first.plan, nullptr);
  EXPECT_EQ(second.plan, nullptr);
  EXPECT_EQ(iree_unaligned_load_le_u32(media.rows[0].data), 17u);
  EXPECT_EQ(iree_unaligned_load_le_u32(media.rows[1].data), 18u);
  EXPECT_EQ(iree_unaligned_load_le_u32(media.rows[2].data), 29u);

  loom_serve_text_projected_media_release_through(&media, 4);
  EXPECT_EQ(media.released_count, 1u);
  EXPECT_EQ(media.results[0].plan, nullptr);
  ASSERT_NE(loom_serve_text_projected_media_input(&media), nullptr);
  EXPECT_EQ(media.input.span_count, 1u);
  EXPECT_EQ(media.input.spans[0].input_offset, 8u);
  loom_serve_text_projected_media_release_through(&media, 9);
  EXPECT_EQ(loom_serve_text_projected_media_input(&media), nullptr);

  loom_serve_text_projected_media_deinitialize(&media);
  iree_string_builder_deinitialize(&chat.images.metadata);
}

}  // namespace
