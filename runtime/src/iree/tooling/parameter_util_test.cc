// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/parameter_util.h"

#include <cstdint>
#include <string>

#include "iree/base/internal/path.h"
#include "iree/io/file_contents.h"
#include "iree/io/parameter_index.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace iree {
namespace {

static std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

static std::string MakeSafetensorsContents(const char* tensor_name,
                                           uint64_t tensor_byte_length) {
  std::string header = "{\"";
  header += tensor_name;
  header += "\":{\"dtype\":\"F32\",\"shape\":[";
  header += std::to_string(tensor_byte_length / sizeof(float));
  header += "],\"data_offsets\":[0,";
  header += std::to_string(tensor_byte_length);
  header += "]}}";

  std::string contents;
  uint64_t header_length = header.size();
  for (int i = 0; i < 8; ++i) {
    contents.push_back(static_cast<char>((header_length >> (8 * i)) & 0xFF));
  }
  contents += header;
  contents.resize(contents.size() + tensor_byte_length, '\0');
  return contents;
}

static void WriteFile(iree_string_view_t path, const std::string& contents) {
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path, iree_make_const_byte_span(contents.data(), contents.size()),
      iree_allocator_system()));
}

TEST(ParameterUtilTest, AppendsSafetensorsIndexManifest) {
  iree::testing::TempFilePath manifest_path("parameter_util_test",
                                            ".safetensors.index.json");
  iree::testing::TempFilePath shard0_path("parameter_util_test_shard0",
                                          ".safetensors");
  iree::testing::TempFilePath shard1_path("parameter_util_test_shard1",
                                          ".safetensors");

  WriteFile(shard0_path.path_view(), MakeSafetensorsContents("tensor0", 4));
  WriteFile(shard1_path.path_view(), MakeSafetensorsContents("tensor1", 8));

  std::string manifest = "{\"metadata\":{\"total_size\":12},\"weight_map\":{";
  manifest += "\"tensor0\":\"";
  manifest += ToString(iree_file_path_basename(shard0_path.path_view()));
  manifest += "\",\"tensor1\":\"";
  manifest += ToString(iree_file_path_basename(shard1_path.path_view()));
  manifest += "\",\"tensor1_duplicate\":\"";
  manifest += ToString(iree_file_path_basename(shard1_path.path_view()));
  manifest += "\"}}";
  WriteFile(manifest_path.path_view(), manifest);

  iree_io_parameter_index_t* index = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), &index));
  IREE_ASSERT_OK(iree_tooling_append_parameter_file_to_index(
      manifest_path.path_view(), index, iree_allocator_system()));

  EXPECT_EQ(2u, iree_io_parameter_index_count(index));

  const iree_io_parameter_index_entry_t* tensor0 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("tensor0"), &tensor0));
  EXPECT_EQ(4ull, tensor0->length);

  const iree_io_parameter_index_entry_t* tensor1 = NULL;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index, IREE_SV("tensor1"), &tensor1));
  EXPECT_EQ(8ull, tensor1->length);

  iree_io_parameter_index_release(index);
}

}  // namespace
}  // namespace iree
