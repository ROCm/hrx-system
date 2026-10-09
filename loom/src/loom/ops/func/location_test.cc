// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/location.h"

#include <array>
#include <vector>

#include "iree/base/alignment.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/ops/func/location.h"
#include "loom/ops/func/location_test_data.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/testing/module_ptr.h"
#include "loom/verify/verify.h"

namespace {

class FuncLocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    const iree_file_toc_t* source = loom_location_test_data_create();
    loom_module_t* module = nullptr;
    IREE_ASSERT_OK(loom_text_parse(
        iree_make_string_view(reinterpret_cast<const char*>(source[0].data),
                              source[0].size),
        IREE_SV("captured.loom"), &context_, &pool_, nullptr, &module));
    module_.reset(module);
    loom_verify_result_t result;
    IREE_ASSERT_OK(loom_verify_module(module_.get(), nullptr, &result));
    ASSERT_EQ(result.error_count, 0u);
  }

  void TearDown() override {
    module_.reset();
    iree_arena_deinitialize(&arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  std::vector<uint8_t> Encode(iree_string_view_t function_name) {
    const auto name = loom_module_lookup_string(module_.get(), function_name);
    const auto symbol = loom_module_find_symbol(module_.get(), name);
    const auto function = loom_func_like_cast(
        module_.get(), module_->symbols.entries[symbol].defining_op);
    const auto* capture =
        loom_region_entry_block(loom_func_like_body(function))->first_op;
    EXPECT_TRUE(loom_func_location_isa(capture));
    iree_const_byte_span_t bytes;
    IREE_CHECK_OK(loom_func_location_encode(
        module_.get(), loom_func_location_nodes(capture), &arena_, &bytes));
    return {bytes.data, bytes.data + bytes.data_length};
  }

  // Backing blocks shared by the source module and independent encoding arena.
  iree_arena_block_pool_t pool_;
  // Owns encoded values independently of source module lifetime.
  iree_arena_allocator_t arena_;
  // Production dialect registry for text and bytecode parsing.
  loom_context_t context_;
  // Current source module, released explicitly at lifetime boundaries.
  loom::testing::ModulePtr module_;
};

TEST_F(FuncLocationTest, DebugStrippingPreservesSemanticCaptures) {
  const std::array<iree_string_view_t, 3> names = {
      IREE_SV("file"), IREE_SV("provenance"), IREE_SV("unknown")};
  std::vector<std::vector<uint8_t>> expected;
  for (auto name : names) {
    expected.push_back(Encode(name));
  }
  iree_io_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_READABLE |
          IREE_IO_STREAM_MODE_SEEKABLE,
      4096, iree_allocator_system(), &stream));
  loom_bytecode_write_options_t options = {
      .location_mode = LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS};
  IREE_ASSERT_OK(
      loom_bytecode_write_module(module_.get(), stream, &options, &pool_));
  std::vector<uint8_t> bytecode(iree_io_stream_length(stream));
  IREE_ASSERT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(
      iree_io_stream_read(stream, bytecode.size(), bytecode.data(), nullptr));
  iree_io_stream_release(stream);
  module_.reset();

  loom_bytecode_read_result_t result;
  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(loom_bytecode_read_module(
      iree_make_const_byte_span(bytecode.data(), bytecode.size()),
      IREE_SV("stripped.loombc"), &context_, &pool_, nullptr, &result, &module,
      iree_allocator_system()));
  ASSERT_EQ(result.error_count, 0u);
  module_.reset(module);
  ASSERT_NE(module, nullptr);
  EXPECT_LE(module->locations.count, 1u);
  for (size_t i = 0; i < names.size(); ++i) {
    EXPECT_EQ(Encode(names[i]), expected[i]);
  }
}

TEST_F(FuncLocationTest, DecodesSharedProvenanceAfterSourceTeardown) {
  const auto bytes = Encode(IREE_SV("provenance"));
  module_.reset();
  loom_location_value_t value;
  IREE_ASSERT_OK(loom_location_value_parse(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &value));
  ASSERT_EQ(value.node_count, 5u);
  EXPECT_EQ(loom_location_value_kind(value, 4), LOOM_LOCATION_VALUE_FUSED);
  ASSERT_EQ(loom_location_value_child_count(value, 4), 3u);
  EXPECT_EQ(loom_location_value_child(value, 4, 0), 2u);
  EXPECT_EQ(loom_location_value_child(value, 4, 2), 1u);
  const auto tag = loom_location_value_tagged(value, 2);
  EXPECT_EQ(tag.child, 1u);
  EXPECT_EQ(tag.tag, 2u);
  const auto file = loom_location_value_file(value, tag.child);
  EXPECT_TRUE(iree_string_view_equal(file.source, IREE_SV("helper.cc")));
  EXPECT_FALSE(file.has_text);
  const auto opaque = loom_location_value_opaque(value, 3);
  EXPECT_TRUE(iree_string_view_equal(opaque.source, IREE_SV("frontend")));
  ASSERT_EQ(opaque.data.data_length, 2u);
  EXPECT_EQ(opaque.data.data[0], 255u);
}

TEST_F(FuncLocationTest, RejectsTruncatedAndInvalidExternalValues) {
  const auto valid = Encode(IREE_SV("file"));
  for (size_t length = 0; length < valid.size(); ++length) {
    SCOPED_TRACE(length);
    loom_location_value_t value;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_location_value_parse(
            iree_make_const_byte_span(valid.data(), length), &value));
  }
  // A valid outer buffer does not establish nested name, field, or text bounds.
  const uint32_t payload = iree_unaligned_load_le_u32(valid.data() + 20);
  const std::array<size_t, 5> offsets = {12, 20, payload, payload + 24,
                                         payload + 32};
  for (auto offset : offsets) {
    SCOPED_TRACE(offset);
    auto bytes = valid;
    iree_unaligned_store_le_u32(bytes.data() + offset, UINT32_MAX);
    loom_location_value_t value;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_location_value_parse(
            iree_make_const_byte_span(bytes.data(), bytes.size()), &value));
  }
  auto graph = Encode(IREE_SV("provenance"));
  const uint32_t children = iree_unaligned_load_le_u32(
      graph.data() + LOOM_LOCATION_VALUE_HEADER_LENGTH +
      4 * LOOM_LOCATION_VALUE_NODE_LENGTH + 4);
  iree_unaligned_store_le_u32(graph.data() + children, 4);
  loom_location_value_t value;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_location_value_parse(
          iree_make_const_byte_span(graph.data(), graph.size()), &value));
}

}  // namespace
