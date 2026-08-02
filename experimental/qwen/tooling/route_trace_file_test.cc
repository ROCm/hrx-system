// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/tooling/route_trace_file.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

constexpr iree_host_size_t kContextCapacity = 64;
constexpr iree_host_size_t kPromptTokenCount = 11;
constexpr iree_host_size_t kGeneratedTokenCount = 7;
constexpr iree_host_size_t kCapturedTokenCount =
    kPromptTokenCount + kGeneratedTokenCount - 1;
constexpr iree_host_size_t kPlaneElementCount =
    kContextCapacity * QWEN_MODEL_LAYER_COUNT * QWEN_MODEL_ROUTE_COUNT;

qwen_route_trace_file_metadata_t MakeMetadata() {
  return qwen_route_trace_file_metadata_t{
      .model = QWEN_ROUTE_TRACE_MODEL_QWEN3_30B_A3B,
      .context_capacity = kContextCapacity,
      .captured_token_count = kCapturedTokenCount,
      .prompt_token_count = kPromptTokenCount,
      .generated_token_count = kGeneratedTokenCount,
      .parameter_count = 579,
      .parameter_layout_fingerprint = UINT64_C(0x123456789abcdef0),
      .encoded_parameter_bytes = UINT64_C(18000000000),
  };
}

std::vector<std::uint8_t> MakePayload() {
  std::vector<std::uint8_t> payload(kPlaneElementCount * sizeof(std::uint32_t) *
                                    2);
  for (std::size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<std::uint8_t>(i * 37u);
  }
  return payload;
}

TEST(QwenRouteTraceFileTest, RoundTripsMetadataAndExactPlanes) {
  auto metadata = MakeMetadata();
  auto payload = MakePayload();
  iree_byte_span_t file_data = iree_byte_span_empty();
  IREE_ASSERT_OK(qwen_route_trace_file_build(
      &metadata, iree_make_const_byte_span(payload.data(), payload.size()),
      iree_allocator_system(), &file_data));

  qwen_route_trace_file_view_t view;
  IREE_ASSERT_OK(
      qwen_route_trace_file_parse(iree_const_cast_byte_span(file_data), &view));
  EXPECT_EQ(view.metadata.model, QWEN_ROUTE_TRACE_MODEL_QWEN3_30B_A3B);
  EXPECT_EQ(view.metadata.context_capacity, kContextCapacity);
  EXPECT_EQ(view.metadata.captured_token_count, kCapturedTokenCount);
  EXPECT_EQ(view.metadata.prompt_token_count, kPromptTokenCount);
  EXPECT_EQ(view.metadata.generated_token_count, kGeneratedTokenCount);
  EXPECT_EQ(view.metadata.parameter_count, 579u);
  EXPECT_EQ(view.metadata.parameter_layout_fingerprint,
            UINT64_C(0x123456789abcdef0));
  EXPECT_EQ(view.metadata.encoded_parameter_bytes, UINT64_C(18000000000));
  ASSERT_EQ(view.route_ids.data_length, payload.size() / 2);
  ASSERT_EQ(view.route_weights.data_length, payload.size() / 2);
  EXPECT_EQ(std::memcmp(view.route_ids.data, payload.data(),
                        view.route_ids.data_length),
            0);
  EXPECT_EQ(std::memcmp(view.route_weights.data,
                        payload.data() + view.route_ids.data_length,
                        view.route_weights.data_length),
            0);

  iree_allocator_free(iree_allocator_system(), file_data.data);
}

TEST(QwenRouteTraceFileTest, RejectsInconsistentCapturedExtent) {
  auto metadata = MakeMetadata();
  ++metadata.captured_token_count;
  auto payload = MakePayload();
  iree_byte_span_t file_data = iree_byte_span_empty();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      qwen_route_trace_file_build(
          &metadata, iree_make_const_byte_span(payload.data(), payload.size()),
          iree_allocator_system(), &file_data));
  EXPECT_TRUE(iree_byte_span_is_empty(file_data));
}

TEST(QwenRouteTraceFileTest, RejectsTruncatedAndMutatedFiles) {
  auto metadata = MakeMetadata();
  auto payload = MakePayload();
  iree_byte_span_t file_data = iree_byte_span_empty();
  IREE_ASSERT_OK(qwen_route_trace_file_build(
      &metadata, iree_make_const_byte_span(payload.data(), payload.size()),
      iree_allocator_system(), &file_data));

  qwen_route_trace_file_view_t view;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      qwen_route_trace_file_parse(
          iree_make_const_byte_span(file_data.data, file_data.data_length - 1),
          &view));
  file_data.data[24] ^= 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      qwen_route_trace_file_parse(iree_const_cast_byte_span(file_data), &view));

  iree_allocator_free(iree_allocator_system(), file_data.data);
}

}  // namespace
