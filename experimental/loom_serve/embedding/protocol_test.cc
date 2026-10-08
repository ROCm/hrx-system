// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/embedding/protocol.h"

#include <limits>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(EmbeddingProtocolTest, OwnsOrderedDecodedSequences) {
  std::string json =
      R"({"model":"fixture","input":["a\u0000b","\u03bb"],"dimensions":2})";
  loom_serve_embedding_request_t request;
  IREE_ASSERT_OK(loom_serve_embedding_request_initialize(
      iree_make_cstring_view(json.c_str()), &request, iree_allocator_system()));
  json.assign(json.size(), 'x');
  ASSERT_EQ(request.base.count, 2u);
  EXPECT_EQ(request.dimensions, 2u);
  iree_const_byte_span_t bytes;
  IREE_ASSERT_OK(iree_vm_buffer_map_read(
      request.base.input, 0, iree_vm_buffer_length(request.base.input),
      &bytes));
  EXPECT_EQ(iree_unaligned_load_le_u64(bytes.data), 2u);
  EXPECT_EQ(iree_unaligned_load_le_u64(bytes.data + 8), 2u);
  EXPECT_EQ(iree_unaligned_load_le_u64(bytes.data + 16), 0u);
  const uint64_t first = iree_unaligned_load_le_u64(bytes.data + 24);
  const uint64_t second = iree_unaligned_load_le_u64(bytes.data + 48);
  EXPECT_EQ(first % 8, 0u);
  EXPECT_EQ(second % 8, 0u);
  EXPECT_EQ(iree_unaligned_load_le_u64(bytes.data + 32), 3u);
  EXPECT_EQ(iree_unaligned_load_le_u64(bytes.data + 56), 2u);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(bytes.data + first), 3),
            std::string("a\0b", 3));
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(bytes.data + second), 2),
            "λ");
  EXPECT_EQ(iree_vm_buffer_access(request.base.input),
            IREE_VM_BUFFER_ACCESS_FLAG_READ);
  loom_serve_embedding_request_deinitialize(&request);
}

TEST(EmbeddingProtocolTest, DistinguishesSingleTokenSequenceAndBatches) {
  for (const char* json : {R"({"model":"m","input":[0,2,1]})",
                           R"({"model":"m","input":[[0,2,1]]})"}) {
    loom_serve_embedding_request_t request;
    IREE_ASSERT_OK(loom_serve_embedding_request_initialize(
        iree_make_cstring_view(json), &request, iree_allocator_system()));
    EXPECT_EQ(request.base.count, 1u);
    iree_const_byte_span_t bytes;
    IREE_ASSERT_OK(iree_vm_buffer_map_read(
        request.base.input, 0, iree_vm_buffer_length(request.base.input),
        &bytes));
    EXPECT_EQ(iree_unaligned_load_le_u64(bytes.data + 16), 1u);
    EXPECT_EQ(iree_unaligned_load_le_u64(bytes.data + 32), 12u);
    const uint64_t offset = iree_unaligned_load_le_u64(bytes.data + 24);
    EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data + offset), 0u);
    EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data + offset + 4), 2u);
    EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data + offset + 8), 1u);
    loom_serve_embedding_request_deinitialize(&request);
  }
}

TEST(EmbeddingProtocolTest, RejectsAmbiguousRequestsAndReleasesPartialInput) {
  for (const char* fields :
       {R"("input":[])", R"("input":[1,"a"])", R"("input":["a",[1]])",
        R"("input":[[1],["x"]])", R"("input":[-1])", R"("input":[1.5])",
        R"("input":[2147483648])", R"("input":false)", R"("input":{})",
        R"("input":"a","input":"b")", R"("input":"a","dimensions":0)",
        R"("input":"a","dimensions":-2)", R"("input":"a","dimensions":1.5)",
        R"("input":"a","encoding_format":"int8")",
        R"("input":"a","encoding_format":"unrecognized")",
        R"("input":"a","surprise":true)", R"("input":"\ud800")",
        R"("input":"a","user":42)"}) {
    const std::string json = std::string("{\"model\":\"m\",") + fields + "}";
    SCOPED_TRACE(json);
    loom_serve_embedding_request_t request;
    iree::Status status(loom_serve_embedding_request_initialize(
        iree_make_cstring_view(json.c_str()), &request,
        iree_allocator_system()));
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(request.base.input, nullptr);
    EXPECT_EQ(request.base.count, 0u);
  }
}

TEST(EmbeddingProtocolTest, DecodesEncodingEnumAndDefersEmptyItemsToSource) {
  loom_serve_embedding_request_t request;
  IREE_ASSERT_OK(loom_serve_embedding_request_initialize(
      IREE_SV(
          R"({"model":"m","input":[[],[0]],"encoding_format":"b\u0061se64"})"),
      &request, iree_allocator_system()));
  EXPECT_EQ(request.base.count, 2u);
  EXPECT_EQ(request.encoding, LOOM_SERVE_EMBEDDING_ENCODING_BASE64);
  iree_const_byte_span_t bytes;
  IREE_ASSERT_OK(iree_vm_buffer_map_read(
      request.base.input, 0, iree_vm_buffer_length(request.base.input),
      &bytes));
  EXPECT_EQ(iree_unaligned_load_le_u64(bytes.data + 32), 0u);
  EXPECT_EQ(iree_unaligned_load_le_u64(bytes.data + 56), 4u);
  loom_serve_embedding_request_deinitialize(&request);
}

TEST(EmbeddingProtocolTest, EncodesExactFloatAndLittleEndianBase64) {
  uint8_t bytes[8];
  iree_unaligned_store_le_f32(bytes, 1.0f);
  iree_unaligned_store_le_f32(bytes + 4, -2.5f);
  const iree_const_byte_span_t row =
      iree_make_const_byte_span(bytes, sizeof(bytes));
  loom_serve_embedding_request_t request = {.base = {.count = 1},
                                            .dimensions = 2};
  iree_string_builder_t output;
  iree_string_builder_initialize(iree_allocator_system(), &output);
  IREE_ASSERT_OK(
      loom_serve_embedding_response(&request, IREE_SV("m"), &row, 3, &output));
  auto text = iree_string_builder_view(&output);
  EXPECT_EQ(
      std::string(text.data, text.size),
      R"({"object":"list","model":"m","data":[{"object":"embedding","index":0,"embedding":[1,-2.5]}],"usage":{"prompt_tokens":3,"total_tokens":3}})");
  iree_string_builder_reset(&output);
  request.encoding = LOOM_SERVE_EMBEDDING_ENCODING_BASE64;
  IREE_ASSERT_OK(
      loom_serve_embedding_response(&request, IREE_SV("m"), &row, 3, &output));
  text = iree_string_builder_view(&output);
  EXPECT_NE(
      std::string(text.data, text.size).find("\"embedding\":\"AACAPwAAIMA=\""),
      std::string::npos);
  iree_unaligned_store_le_f32(bytes, std::numeric_limits<float>::infinity());
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      loom_serve_embedding_response(&request, IREE_SV("m"), &row, 3, &output));
  iree_unaligned_store_le_f32(bytes, 1.0f);
  request.dimensions = 4;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      loom_serve_embedding_response(&request, IREE_SV("m"), &row, 3, &output));
  iree_string_builder_deinitialize(&output);
}

}  // namespace
