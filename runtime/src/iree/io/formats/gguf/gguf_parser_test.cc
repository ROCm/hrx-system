// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/formats/gguf/gguf_parser.h"

#include <algorithm>
#include <vector>

#include "iree/io/formats/gguf/testdata/gguf_files.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

static iree_io_file_handle_t* OpenTestFile(const char* name) {
  const struct iree_file_toc_t* file_toc = iree_io_gguf_files_create();
  for (size_t i = 0; i < iree_io_gguf_files_size(); ++i) {
    if (strcmp(file_toc[i].name, name) == 0) {
      iree_io_file_handle_t* file_handle = NULL;
      IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
          IREE_IO_FILE_ACCESS_READ,
          iree_make_byte_span((void*)file_toc[i].data, file_toc[i].size),
          iree_io_file_handle_release_callback_null(), iree_allocator_system(),
          &file_handle));
      return file_handle;
    }
  }
  IREE_CHECK_OK(iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "test file `%s` not found embedded into test binary", name));
  return NULL;
}

static iree_io_file_handle_t* OpenTestFileWithFirstTensorType(
    const char* name, uint32_t tensor_type,
    std::vector<uint8_t>* out_file_contents) {
  const struct iree_file_toc_t* files = iree_io_gguf_files_create();
  const struct iree_file_toc_t* file = NULL;
  for (size_t i = 0; i < iree_io_gguf_files_size(); ++i) {
    if (strcmp(files[i].name, name) == 0) {
      file = &files[i];
      break;
    }
  }
  IREE_ASSERT(file != NULL);
  out_file_contents->assign(file->data, file->data + file->size);

  static constexpr char kTensorName[] = "tensor0";
  auto tensor_name =
      std::search(out_file_contents->begin(), out_file_contents->end(),
                  kTensorName, kTensorName + sizeof(kTensorName) - 1);
  IREE_ASSERT(tensor_name != out_file_contents->end());
  const size_t tensor_name_offset =
      (size_t)(tensor_name - out_file_contents->begin());
  const size_t dimension_count_offset =
      tensor_name_offset + sizeof(kTensorName) - 1;
  IREE_ASSERT(dimension_count_offset + sizeof(uint32_t) <=
              out_file_contents->size());
  uint32_t dimension_count = 0;
  memcpy(&dimension_count, out_file_contents->data() + dimension_count_offset,
         sizeof(dimension_count));
  const size_t tensor_type_offset = dimension_count_offset +
                                    sizeof(dimension_count) +
                                    dimension_count * sizeof(uint64_t);
  IREE_ASSERT(tensor_type_offset + sizeof(tensor_type) <=
              out_file_contents->size());
  memcpy(out_file_contents->data() + tensor_type_offset, &tensor_type,
         sizeof(tensor_type));

  iree_io_file_handle_t* file_handle = NULL;
  IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ,
      iree_make_byte_span(out_file_contents->data(), out_file_contents->size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));
  return file_handle;
}

TEST(GgufFormatTest, Empty) {
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  iree_io_file_handle_t* file_handle = OpenTestFile("empty.gguf");
  IREE_ASSERT_OK(
      iree_io_parse_gguf_index(file_handle, index, iree_allocator_system()));
  iree_io_file_handle_release(file_handle);

  iree_io_parameter_index_release(index);
}

TEST(GgufFormatTest, SingleTensor) {
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  iree_io_file_handle_t* file_handle = OpenTestFile("single.gguf");
  IREE_ASSERT_OK(
      iree_io_parse_gguf_index(file_handle, index, iree_allocator_system()));
  iree_io_file_handle_release(file_handle);

  const iree_io_parameter_index_entry_t* entry0 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("tensor0"), &entry0));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("tensor0"), entry0->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry0->metadata));
  EXPECT_EQ(entry0->storage.file.offset, 384);
  EXPECT_EQ(entry0->length, 16);

  iree_io_parameter_index_release(index);
}

// Tests that GGUF version 2 parses. Other tests use version 3.
TEST(GgufFormatTest, SingleTensorV2) {
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  iree_io_file_handle_t* file_handle = OpenTestFile("single_v2.gguf");
  IREE_ASSERT_OK(
      iree_io_parse_gguf_index(file_handle, index, iree_allocator_system()));
  iree_io_file_handle_release(file_handle);

  const iree_io_parameter_index_entry_t* entry0 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("tensor0"), &entry0));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("tensor0"), entry0->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry0->metadata));
  EXPECT_EQ(entry0->storage.file.offset, 384);
  EXPECT_EQ(entry0->length, 16);

  iree_io_parameter_index_release(index);
}

TEST(GgufFormatTest, MultipleTensors) {
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  iree_io_file_handle_t* file_handle = OpenTestFile("multiple.gguf");
  IREE_ASSERT_OK(
      iree_io_parse_gguf_index(file_handle, index, iree_allocator_system()));
  iree_io_file_handle_release(file_handle);

  const iree_io_parameter_index_entry_t* entry0 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("tensor0"), &entry0));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("tensor0"), entry0->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry0->metadata));
  EXPECT_EQ(entry0->storage.file.offset, 448);
  EXPECT_EQ(entry0->length, 16);

  const iree_io_parameter_index_entry_t* entry1 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("tensor1"), &entry1));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("tensor1"), entry1->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry1->metadata));
  EXPECT_EQ(entry1->storage.file.offset, 512);
  EXPECT_EQ(entry1->length, 8);

  const iree_io_parameter_index_entry_t* entry2 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("tensor2"), &entry2));
  EXPECT_TRUE(iree_string_view_equal(IREE_SV("tensor2"), entry2->key));
  EXPECT_TRUE(iree_const_byte_span_is_empty(entry2->metadata));
  EXPECT_EQ(entry2->storage.file.offset, 576);
  EXPECT_EQ(entry2->length, 48);

  iree_io_parameter_index_release(index);
}

TEST(GgufFormatTest, Bfloat16Tensor) {
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  std::vector<uint8_t> file_contents;
  iree_io_file_handle_t* file_handle = OpenTestFileWithFirstTensorType(
      "single.gguf", /*GGML_TYPE_BF16=*/30, &file_contents);
  IREE_ASSERT_OK(
      iree_io_parse_gguf_index(file_handle, index, iree_allocator_system()));
  iree_io_file_handle_release(file_handle);

  const iree_io_parameter_index_entry_t* entry0 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("tensor0"), &entry0));
  EXPECT_EQ(entry0->storage.file.offset, 384);
  EXPECT_EQ(entry0->length, 8);

  iree_io_parameter_index_release(index);
}

TEST(GgufFormatTest, UnsupportedModernQuantizedTensor) {
  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));

  std::vector<uint8_t> file_contents;
  iree_io_file_handle_t* file_handle = OpenTestFileWithFirstTensorType(
      "single.gguf", /*GGML_TYPE_IQ2_XXS=*/16, &file_contents);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_io_parse_gguf_index(file_handle, index, iree_allocator_system()));
  iree_io_file_handle_release(file_handle);

  iree_io_parameter_index_release(index);
}

}  // namespace
}  // namespace iree
