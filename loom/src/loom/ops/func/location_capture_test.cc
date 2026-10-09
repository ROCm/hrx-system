// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/func/location_capture.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/location.h"
#include "loom/ops/func/location.h"
#include "loom/ops/func/location_capture_test_data.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/op_registry.h"
#include "loom/tooling/input/input.h"

namespace {

std::string String(iree_string_view_t value) {
  return {value.data ? value.data : "", value.size};
}

class LocationCaptureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &scratch_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    const auto* data = loom_location_capture_test_data_create();
    std::string text(reinterpret_cast<const char*>(data[0].data), data[0].size);
    loom_input_request_t request = {
        .source = iree_make_string_view(text.data(), text.size()),
        .path = IREE_SV("source.loom")};
    IREE_ASSERT_OK(loom_input_module_load(&loom_input_text_provider, &request,
                                          &context_, &pool_,
                                          iree_allocator_system(), &input_));
    ASSERT_NE(input_.module, nullptr);
  }

  void TearDown() override {
    loom_input_module_deinitialize(&input_);
    iree_arena_deinitialize(&scratch_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_op_t* Original() {
    auto* module = input_.module;
    auto name = loom_module_lookup_string(module, IREE_SV("original"));
    return module->symbols.entries[loom_module_find_symbol(module, name)]
        .defining_op;
  }

  std::vector<uint8_t> Capture(loom_location_id_t location) {
    loom_parameterized_attr_array_t nodes;
    IREE_CHECK_OK(loom_func_location_capture(
        input_.module, location, loom_input_module_source_resolver(&input_),
        &scratch_, &nodes));
    // Captured payloads are independent of both scratch and admitted sources.
    iree_arena_reset(&scratch_);
    loom_source_storage_deinitialize(&input_.sources);
    iree_const_byte_span_t bytes;
    IREE_CHECK_OK(
        loom_func_location_encode(input_.module, nodes, &scratch_, &bytes));
    return {bytes.data, bytes.data + bytes.data_length};
  }

  // Backing allocation pool, released after all compiler owners.
  iree_arena_block_pool_t pool_;
  // Short-lived capture and encoding scratch.
  iree_arena_allocator_t scratch_;
  // Production dialect descriptors shared by admitted modules.
  loom_context_t context_;
  // Admitted module and independently releasable source snapshots.
  loom_input_module_t input_ = {};
};

TEST_F(LocationCaptureTest, OriginalSpellingAndFieldsOutliveAdmission) {
  auto* add = loom_region_entry_block(loom_func_like_body(loom_func_like_cast(
                                          input_.module, Original())))
                  ->first_op;
  const auto bytes = Capture(add->location);
  loom_input_module_deinitialize(&input_);
  loom_location_value_t value;
  IREE_ASSERT_OK(
      loom_location_value_parse({bytes.data(), bytes.size()}, &value));
  ASSERT_EQ(value.node_count, 1u);
  const auto file = loom_location_value_file(value, 0);
  EXPECT_EQ(String(file.source), "source.loom");
  EXPECT_TRUE(file.has_text);
  EXPECT_EQ(String(iree_make_string_view(
                reinterpret_cast<const char*>(file.text.data),
                file.text.data_length)),
            "  %sum = scalar.addi %left, %right : i32\n");
  EXPECT_EQ(file.range.start_line, 5u);
  EXPECT_EQ(file.range.start_column, 3u);
  bool found_left = false;
  bool found_right = false;
  for (uint32_t i = 0; i < file.field_count; ++i) {
    const auto field = loom_location_value_field(value, 0, i);
    if (field.kind != LOOM_LOCATION_FIELD_OPERAND) {
      continue;
    }
    auto spelling =
        std::string(reinterpret_cast<const char*>(file.text.data) +
                        field.range.start_column - 1,
                    field.range.end_column - field.range.start_column);
    if (field.index == 0) {
      EXPECT_EQ(spelling, "%left");
      found_left = true;
    } else if (field.index == 1) {
      EXPECT_EQ(spelling, "%right");
      found_right = true;
    }
  }
  EXPECT_TRUE(found_left);
  EXPECT_TRUE(found_right);
}

TEST_F(LocationCaptureTest, ColumnsPreserveUnicodeCodePoints) {
  const auto bytes = Capture(Original()->location);
  loom_location_value_t value;
  IREE_ASSERT_OK(
      loom_location_value_parse({bytes.data(), bytes.size()}, &value));
  const auto file = loom_location_value_file(value, 0);
  EXPECT_EQ(file.range.start_line, 4u);
  const auto text = iree_make_string_view(
      reinterpret_cast<const char*>(file.text.data), file.text.data_length);
  bool found_symbol = false;
  for (uint32_t i = 0; i < file.field_count; ++i) {
    const auto field = loom_location_value_field(value, 0, i);
    if (field.range.start_line != file.range.start_line) {
      continue;
    }
    auto start = loom_source_byte_offset(text, 1, field.range.start_column);
    auto end = loom_source_byte_offset(
        text, field.range.end_line - file.range.start_line + 1,
        field.range.end_column);
    if (String(iree_make_string_view(text.data + start, end - start)) ==
        "@original") {
      EXPECT_EQ(field.range.start_column, 27u);
      EXPECT_EQ(start, 28u);
      found_symbol = true;
    }
  }
  EXPECT_TRUE(found_symbol);
}

TEST_F(LocationCaptureTest, ExclusiveEndAtNextLineDoesNotCaptureThatLine) {
  const auto source_id = loom_location_table_const_entry(
                             &input_.module->locations, Original()->location)
                             ->file.source_id;
  loom_location_id_t location;
  IREE_ASSERT_OK(loom_module_add_location(
      input_.module, loom_location_file_range(source_id, 5, 3, 6, 1),
      &location));
  const auto bytes = Capture(location);
  loom_location_value_t value;
  IREE_ASSERT_OK(
      loom_location_value_parse({bytes.data(), bytes.size()}, &value));
  const auto file = loom_location_value_file(value, 0);
  EXPECT_TRUE(file.has_text);
  EXPECT_EQ(String(iree_make_string_view(
                reinterpret_cast<const char*>(file.text.data),
                file.text.data_length)),
            "  %sum = scalar.addi %left, %right : i32\n");
}

TEST_F(LocationCaptureTest, SharedGraphRetainsOpaqueTaggedAndUnavailableData) {
  auto* module = input_.module;
  loom_source_id_t source;
  IREE_ASSERT_OK(
      loom_module_register_source(module, IREE_SV("absent.h"), &source));
  loom_location_id_t file;
  IREE_ASSERT_OK(loom_module_add_location(
      module, loom_location_file_range(source, 7, 2, 8, 1), &file));
  uint8_t payload[] = {0, 255, 3};
  loom_location_id_t tagged;
  auto tag = loom_location_tagged(LOOM_LOCATION_TAG_TEMPLATE_INSTANTIATION,
                                  file, payload, sizeof(payload));
  tag.flags = LOOM_LOCATION_FLAG_SYNTHETIC;
  IREE_ASSERT_OK(loom_module_add_location(module, tag, &tagged));
  loom_location_entry_t opaque = {.kind = LOOM_LOCATION_OPAQUE};
  opaque.opaque.source_id = source;
  opaque.opaque.data = payload;
  opaque.opaque.data_length = sizeof(payload);
  loom_location_id_t external;
  IREE_ASSERT_OK(loom_module_add_location(module, opaque, &external));
  loom_location_id_t children[] = {tagged, external, file, tagged,
                                   LOOM_LOCATION_UNKNOWN};
  loom_location_entry_t fused = {.kind = LOOM_LOCATION_FUSED};
  fused.fused.count = IREE_ARRAYSIZE(children);
  fused.fused.children = children;
  loom_location_id_t root;
  IREE_ASSERT_OK(loom_module_add_location(module, fused, &root));
  const auto bytes = Capture(root);
  loom_input_module_deinitialize(&input_);
  loom_location_value_t value;
  IREE_ASSERT_OK(
      loom_location_value_parse({bytes.data(), bytes.size()}, &value));
  ASSERT_EQ(value.node_count, 5u);
  EXPECT_EQ(loom_location_value_child(value, 4, 0), 1u);
  EXPECT_EQ(loom_location_value_child(value, 4, 2), 0u);
  EXPECT_EQ(loom_location_value_child(value, 4, 3), 1u);
  const auto captured_file = loom_location_value_file(value, 0);
  EXPECT_EQ(String(captured_file.source), "absent.h");
  EXPECT_FALSE(captured_file.has_text);
  EXPECT_EQ(captured_file.range.start_line, 7u);
  EXPECT_EQ(loom_location_value_flags(value, 1),
            LOOM_LOCATION_VALUE_FLAG_SYNTHETIC);
  const auto captured_tag = loom_location_value_tagged(value, 1);
  EXPECT_EQ(captured_tag.tag, LOOM_LOCATION_TAG_TEMPLATE_INSTANTIATION);
  EXPECT_EQ(captured_tag.child, 0u);
  ASSERT_EQ(captured_tag.data.data_length, sizeof(payload));
  EXPECT_EQ(captured_tag.data.data[1], 255u);
  const auto captured_external = loom_location_value_opaque(value, 2);
  EXPECT_EQ(String(captured_external.source), "absent.h");
  ASSERT_EQ(captured_external.data.data_length, sizeof(payload));
  EXPECT_EQ(captured_external.data.data[2], 3u);
  EXPECT_EQ(loom_location_value_kind(value, 3), LOOM_LOCATION_VALUE_UNKNOWN);
}

TEST_F(LocationCaptureTest, DeepGraphUsesNoCallStack) {
  loom_location_id_t root = Original()->location;
  constexpr uint32_t kDepth = 4096;
  for (uint32_t i = 0; i < kDepth; ++i) {
    IREE_ASSERT_OK(loom_module_add_location(
        input_.module,
        loom_location_tagged(LOOM_LOCATION_TAG_TEMPLATE_INSTANTIATION, root,
                             nullptr, 0),
        &root));
  }
  const auto bytes = Capture(root);
  loom_location_value_t value;
  IREE_ASSERT_OK(
      loom_location_value_parse({bytes.data(), bytes.size()}, &value));
  ASSERT_EQ(value.node_count, kDepth + 1);
  for (uint32_t i = 1; i <= kDepth; ++i) {
    EXPECT_EQ(loom_location_value_tagged(value, i).child, i - 1);
  }
}

TEST_F(LocationCaptureTest, UnknownLocationProducesAnExplicitUnknownNode) {
  const auto bytes = Capture(LOOM_LOCATION_UNKNOWN);
  loom_location_value_t value;
  IREE_ASSERT_OK(
      loom_location_value_parse({bytes.data(), bytes.size()}, &value));
  EXPECT_EQ(value.node_count, 1u);
  EXPECT_EQ(loom_location_value_kind(value, 0), LOOM_LOCATION_VALUE_UNKNOWN);
}

TEST_F(LocationCaptureTest, UnavailableCoordinatesKeepTheRangeWithoutText) {
  const auto source_id = loom_location_table_const_entry(
                             &input_.module->locations, Original()->location)
                             ->file.source_id;
  loom_location_id_t location;
  IREE_ASSERT_OK(loom_module_add_location(
      input_.module, loom_location_file_range(source_id, 60000, 1, 60000, 20),
      &location));
  const auto bytes = Capture(location);
  loom_location_value_t value;
  IREE_ASSERT_OK(
      loom_location_value_parse({bytes.data(), bytes.size()}, &value));
  const auto file = loom_location_value_file(value, 0);
  EXPECT_EQ(String(file.source), "source.loom");
  EXPECT_EQ(file.range.start_line, 60000u);
  EXPECT_EQ(file.range.end_column, 20u);
  EXPECT_FALSE(file.has_text);
}

TEST_F(LocationCaptureTest, PresentEmptySourceIsDistinctFromUnavailableText) {
  loom_source_id_t source_id;
  IREE_ASSERT_OK(loom_module_register_source(
      input_.module, IREE_SV("empty.loom"), &source_id));
  IREE_ASSERT_OK(loom_source_storage_insert(
      &input_.sources, source_id, IREE_SV("empty.loom"), IREE_SV("")));
  loom_location_id_t location;
  IREE_ASSERT_OK(loom_module_add_location(
      input_.module, loom_location_file_range(source_id, 1, 1, 1, 1),
      &location));
  const auto bytes = Capture(location);
  loom_location_value_t value;
  IREE_ASSERT_OK(
      loom_location_value_parse({bytes.data(), bytes.size()}, &value));
  const auto file = loom_location_value_file(value, 0);
  EXPECT_TRUE(file.has_text);
  EXPECT_EQ(file.text.data_length, 0u);
}

}  // namespace
