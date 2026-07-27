// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/value_io.h"

#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/io/file_contents.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace iree {
namespace {

using ::iree::testing::status::StatusIs;
using ::testing::ElementsAreArray;

class ValueIOTest : public ::testing::Test {
 protected:
  void SetUp() override {
    host_allocator_ = iree_allocator_system();
    IREE_ASSERT_OK(
        iree_tooling_value_io_context_allocate(host_allocator_, &context_));
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(
        IREE_SV("test"), host_allocator_, host_allocator_, &device_allocator_));
  }

  void TearDown() override {
    for (const auto& path : temp_paths_) {
      path.Remove();
    }
    iree_hal_allocator_release(device_allocator_);
    iree_tooling_value_io_context_free(context_);
  }

  std::string CreateTempFileWithContents(iree_const_byte_span_t contents) {
    iree::testing::TempFilePath path("iree_value_io_test");
    IREE_EXPECT_OK(iree_io_file_contents_write(path.path_view(), contents,
                                               host_allocator_));
    std::string result = path.path();
    temp_paths_.push_back(std::move(path));
    return result;
  }

  template <typename T>
  void AssertBufferContents(iree_hal_buffer_t* buffer,
                            std::vector<T> expected_contents) {
    ASSERT_EQ(iree_hal_buffer_byte_length(buffer),
              expected_contents.size() * sizeof(T));
    std::vector<T> actual_contents(expected_contents.size());
    IREE_ASSERT_OK(iree_hal_buffer_map_read(
        buffer, 0, actual_contents.data(), actual_contents.size() * sizeof(T)));
    EXPECT_THAT(actual_contents, ElementsAreArray(expected_contents));
  }

  template <typename T>
  void AssertBufferViewContents(iree_hal_buffer_view_t* buffer_view,
                                std::vector<iree_hal_dim_t> expected_shape,
                                iree_hal_element_type_t expected_element_type,
                                std::vector<T> expected_contents) {
    ASSERT_EQ(iree_hal_buffer_view_shape_rank(buffer_view),
              expected_shape.size());
    for (iree_host_size_t i = 0; i < expected_shape.size(); ++i) {
      EXPECT_EQ(iree_hal_buffer_view_shape_dim(buffer_view, i),
                expected_shape[i]);
    }
    EXPECT_EQ(iree_hal_buffer_view_element_type(buffer_view),
              expected_element_type);
    AssertBufferContents(iree_hal_buffer_view_buffer(buffer_view),
                         expected_contents);
  }

  iree_allocator_t host_allocator_;
  iree_tooling_value_io_context_t* context_ = nullptr;
  iree_hal_allocator_t* device_allocator_ = nullptr;
  std::vector<iree::testing::TempFilePath> temp_paths_;
};

TEST_F(ValueIOTest, ParsesScalarSpecs) {
  iree_tooling_value_t value = {};

  IREE_ASSERT_OK(iree_tooling_value_spec_parse(IREE_SV("i32=-42"), &value));
  EXPECT_EQ(value.kind, IREE_TOOLING_VALUE_KIND_I32);
  EXPECT_EQ(value.storage.i32, -42);

  IREE_ASSERT_OK(iree_tooling_value_spec_parse(IREE_SV("u32=0x2A"), &value));
  EXPECT_EQ(value.kind, IREE_TOOLING_VALUE_KIND_U32);
  EXPECT_EQ(value.storage.u32, 42u);

  IREE_ASSERT_OK(
      iree_tooling_value_spec_parse(IREE_SV("i64=-1234567890123"), &value));
  EXPECT_EQ(value.kind, IREE_TOOLING_VALUE_KIND_I64);
  EXPECT_EQ(value.storage.i64, -1234567890123ll);

  IREE_ASSERT_OK(
      iree_tooling_value_spec_parse(IREE_SV("u64=0x1122334455667788"), &value));
  EXPECT_EQ(value.kind, IREE_TOOLING_VALUE_KIND_U64);
  EXPECT_EQ(value.storage.u64, 0x1122334455667788ull);

  IREE_ASSERT_OK(iree_tooling_value_spec_parse(IREE_SV("f32=3.5"), &value));
  EXPECT_EQ(value.kind, IREE_TOOLING_VALUE_KIND_F32);
  EXPECT_EQ(value.storage.f32, 3.5f);

  IREE_ASSERT_OK(iree_tooling_value_spec_parse(IREE_SV("f64=-6.25"), &value));
  EXPECT_EQ(value.kind, IREE_TOOLING_VALUE_KIND_F64);
  EXPECT_EQ(value.storage.f64, -6.25);

  IREE_ASSERT_OK(iree_tooling_value_spec_parse(IREE_SV("0xDEADBEEF"), &value));
  EXPECT_EQ(value.kind, IREE_TOOLING_VALUE_KIND_RAW_U32);
  EXPECT_EQ(value.storage.u32, 0xDEADBEEFu);
}

TEST_F(ValueIOTest, RejectsAmbiguousOrMalformedScalarSpecs) {
  iree_tooling_value_t value = {};
  EXPECT_THAT(Status(iree_tooling_value_spec_parse(IREE_SV("42"), &value)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(
      Status(iree_tooling_value_spec_parse(IREE_SV("u32=12wat"), &value)),
      StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(Status(iree_tooling_value_spec_parse(IREE_SV("u32=-1"), &value)),
              StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(
      Status(iree_tooling_value_spec_parse(IREE_SV("0x100000000"), &value)),
      StatusIs(StatusCode::kInvalidArgument));
  EXPECT_THAT(Status(iree_tooling_value_spec_parse(IREE_SV("i8=1"), &value)),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(ValueIOTest, WritesAbiWords) {
  iree_tooling_value_t value = {};
  uint32_t words[2] = {};
  iree_host_size_t word_count = 0;

  IREE_ASSERT_OK(iree_tooling_value_spec_parse(IREE_SV("0xDEADBEEF"), &value));
  IREE_ASSERT_OK(iree_tooling_value_write_abi_words(
      &value, IREE_ARRAYSIZE(words), words, &word_count));
  EXPECT_EQ(word_count, 1u);
  EXPECT_EQ(words[0], 0xDEADBEEFu);

  IREE_ASSERT_OK(
      iree_tooling_value_spec_parse(IREE_SV("u64=0x1122334455667788"), &value));
  IREE_ASSERT_OK(iree_tooling_value_write_abi_words(
      &value, IREE_ARRAYSIZE(words), words, &word_count));
  EXPECT_EQ(word_count, 2u);
  uint64_t round_trip_u64 = 0;
  memcpy(&round_trip_u64, words, sizeof(round_trip_u64));
  EXPECT_EQ(round_trip_u64, 0x1122334455667788ull);

  IREE_ASSERT_OK(iree_tooling_value_spec_parse(IREE_SV("f32=3.5"), &value));
  IREE_ASSERT_OK(iree_tooling_value_write_abi_words(
      &value, IREE_ARRAYSIZE(words), words, &word_count));
  EXPECT_EQ(word_count, 1u);
  float round_trip_f32 = 0.0f;
  memcpy(&round_trip_f32, words, sizeof(round_trip_f32));
  EXPECT_EQ(round_trip_f32, 3.5f);
}

TEST_F(ValueIOTest, ParsesInlineBufferView) {
  iree_hal_buffer_view_t* buffer_view = nullptr;
  IREE_ASSERT_OK(iree_tooling_buffer_view_spec_parse(
      context_, IREE_SV("2x2xi32=[42 43][44 45]"),
      /*device=*/NULL, device_allocator_, &buffer_view));
  AssertBufferViewContents<int32_t>(
      buffer_view, {2, 2}, IREE_HAL_ELEMENT_TYPE_INT_32, {42, 43, 44, 45});
  iree_hal_buffer_view_release(buffer_view);
}

TEST_F(ValueIOTest, ParsesStorageBuffer) {
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_tooling_storage_buffer_spec_parse(
      context_, IREE_SV("&2xi32=[7 8]"), /*device=*/NULL, device_allocator_,
      &buffer));
  AssertBufferContents<int32_t>(buffer, {7, 8});
  iree_hal_buffer_release(buffer);
}

TEST_F(ValueIOTest, ParsesBinaryFileBufferViewsSequentially) {
  const int32_t file_contents[] = {1, 2, 3, 4};
  std::string path = CreateTempFileWithContents(
      iree_make_const_byte_span(file_contents, sizeof(file_contents)));
  std::string first_spec = "2xi32=@" + path;
  std::string next_spec = "2xi32=+" + path;

  iree_hal_buffer_view_t* first_view = nullptr;
  IREE_ASSERT_OK(iree_tooling_buffer_view_spec_parse(
      context_, iree_make_string_view(first_spec.data(), first_spec.size()),
      /*device=*/NULL, device_allocator_, &first_view));
  AssertBufferViewContents<int32_t>(first_view, {2},
                                    IREE_HAL_ELEMENT_TYPE_INT_32, {1, 2});

  iree_hal_buffer_view_t* next_view = nullptr;
  IREE_ASSERT_OK(iree_tooling_buffer_view_spec_parse(
      context_, iree_make_string_view(next_spec.data(), next_spec.size()),
      /*device=*/NULL, device_allocator_, &next_view));
  AssertBufferViewContents<int32_t>(next_view, {2},
                                    IREE_HAL_ELEMENT_TYPE_INT_32, {3, 4});

  iree_hal_buffer_view_t* reset_view = nullptr;
  IREE_ASSERT_OK(iree_tooling_buffer_view_spec_parse(
      context_, iree_make_string_view(first_spec.data(), first_spec.size()),
      /*device=*/NULL, device_allocator_, &reset_view));
  AssertBufferViewContents<int32_t>(reset_view, {2},
                                    IREE_HAL_ELEMENT_TYPE_INT_32, {1, 2});

  iree_hal_buffer_view_release(reset_view);
  iree_hal_buffer_view_release(next_view);
  iree_hal_buffer_view_release(first_view);
}

TEST_F(ValueIOTest, ParsesBindingKinds) {
  iree_tooling_buffer_binding_t view_binding = {};
  IREE_ASSERT_OK(iree_tooling_buffer_binding_spec_parse(
      context_, IREE_SV("2xi32=[10 11]"), /*device=*/NULL, device_allocator_,
      &view_binding));
  EXPECT_EQ(view_binding.kind, IREE_TOOLING_BUFFER_BINDING_KIND_BUFFER_VIEW);
  EXPECT_NE(view_binding.buffer, nullptr);
  EXPECT_NE(view_binding.buffer_view, nullptr);
  EXPECT_EQ(view_binding.byte_offset, 0u);
  EXPECT_EQ(view_binding.byte_length, 8u);
  AssertBufferViewContents<int32_t>(view_binding.buffer_view, {2},
                                    IREE_HAL_ELEMENT_TYPE_INT_32, {10, 11});
  iree_tooling_buffer_binding_deinitialize(&view_binding);

  iree_tooling_buffer_binding_t storage_binding = {};
  IREE_ASSERT_OK(iree_tooling_buffer_binding_spec_parse(
      context_, IREE_SV("&2xi32=[12 13]"), /*device=*/NULL, device_allocator_,
      &storage_binding));
  EXPECT_EQ(storage_binding.kind,
            IREE_TOOLING_BUFFER_BINDING_KIND_STORAGE_BUFFER);
  EXPECT_NE(storage_binding.buffer, nullptr);
  EXPECT_EQ(storage_binding.buffer_view, nullptr);
  EXPECT_EQ(storage_binding.byte_offset, 0u);
  EXPECT_EQ(storage_binding.byte_length, 8u);
  AssertBufferContents<int32_t>(storage_binding.buffer, {12, 13});
  iree_tooling_buffer_binding_deinitialize(&storage_binding);
}

}  // namespace
}  // namespace iree
