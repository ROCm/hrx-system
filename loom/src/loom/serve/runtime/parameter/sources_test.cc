// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/sources.h"

#include <fstream>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

class ParameterSourcesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Write(first_.path(), "same", 16);
    Write(second_.path(), "same", 32);
    Write(shard_.path(), "other", 8);
  }

  void TearDown() override {
    loom_serve_parameter_sources_release(catalog_);
    loom_serve_parameter_sources_release(prefix_);
    for (auto* file : {&first_, &second_, &shard_}) {
      if (file->Exists()) {
        EXPECT_TRUE(file->Remove());
      }
    }
  }

  void Write(const std::string& path, const char* key, uint32_t length) {
    const std::string header =
        std::string("{\"") + key + "\":{\"dtype\":\"U8\",\"shape\":[" +
        std::to_string(length) + "],\"data_offsets\":[0," +
        std::to_string(length) + "]}}";
    std::ofstream file(path, std::ios::binary);
    for (unsigned i = 0; i < 8; ++i) {
      file.put(
          static_cast<char>(static_cast<uint64_t>(header.size()) >> (8 * i)));
    }
    file.write(header.data(), header.size());
    for (uint32_t i = 0; i < length; ++i) {
      file.put(static_cast<char>(i + 1));
    }
    file.close();
    ASSERT_TRUE(file.good());
  }

  void ExpectEntry(loom_serve_parameter_sources_t* sources, uint32_t ordinal,
                   const char* key, uint64_t length) {
    const iree_io_parameter_index_entry_t* entry = nullptr;
    IREE_ASSERT_OK(iree_io_parameter_index_lookup(
        loom_serve_parameter_sources_index(sources, ordinal),
        iree_make_cstring_view(key), &entry));
    EXPECT_EQ(entry->length, length);
    EXPECT_EQ(entry->type, IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE);
  }

  // Cold filesystem namespace, independent of any source-code package.
  const loom_serve_sources_t namespace_ = {.base_directory = IREE_SV(".")};
  // Allocator owning real parsed indexes and providers.
  const iree_allocator_t allocator_ = iree_allocator_system();
  // Original immutable model source.
  iree::testing::TempFilePath first_{"parameter_source_first", ".safetensors"};
  // Request source deliberately reusing the first source's key.
  iree::testing::TempFilePath second_{"parameter_source_second", ".data"};
  // Additional shard in the original source's namespace.
  iree::testing::TempFilePath shard_{"parameter_source_shard", ".safetensors"};
  // Absent path used to exercise partially accepted cold indexing.
  iree::testing::TempFilePath missing_{"parameter_source_missing",
                                       ".safetensors"};
  // Original source owner, releasable independently of the request owner.
  loom_serve_parameter_sources_t* prefix_ = nullptr;
  // Extended source owner, including partial construction outputs.
  loom_serve_parameter_sources_t* catalog_ = nullptr;
};

TEST_F(ParameterSourcesTest, OrderedShardsAndIndependentKeysRetainTheirPrefix) {
  const loom_model_file_t paths[] = {
      {iree_make_cstring_view(first_.path().c_str()), {}},
      {iree_make_cstring_view(shard_.path().c_str()), {}},
      {iree_make_cstring_view(second_.path().c_str()), IREE_SV("safetensors")}};
  const loom_model_file_list_t base = {2, paths};
  const loom_model_file_list_t extension = {1, paths + 2};
  IREE_ASSERT_OK(loom_serve_parameter_sources_create(
      nullptr, &namespace_, 1, &base, allocator_, &prefix_));
  auto* provider = loom_serve_parameter_sources_provider(prefix_, 0);
  auto* index = loom_serve_parameter_sources_index(prefix_, 0);
  IREE_ASSERT_OK(loom_serve_parameter_sources_create(
      prefix_, &namespace_, 1, &extension, allocator_, &catalog_));
  loom_serve_parameter_sources_release(prefix_);
  prefix_ = nullptr;
  EXPECT_EQ(loom_serve_parameter_sources_count(catalog_), 2u);
  EXPECT_EQ(loom_serve_parameter_sources_provider(catalog_, 0), provider);
  EXPECT_EQ(loom_serve_parameter_sources_index(catalog_, 0), index);
  EXPECT_NE(loom_serve_parameter_sources_provider(catalog_, 1), provider);
  ExpectEntry(catalog_, 0, "same", 16);
  ExpectEntry(catalog_, 0, "other", 8);
  ExpectEntry(catalog_, 1, "same", 32);
}

TEST_F(ParameterSourcesTest, EmptyExtensionRetainsWithoutAllocatingOrOpening) {
  IREE_ASSERT_OK(loom_serve_parameter_sources_create(
      nullptr, &namespace_, 0, nullptr, iree_allocator_null(), &catalog_));
  EXPECT_EQ(catalog_, nullptr);
  EXPECT_EQ(loom_serve_parameter_sources_count(catalog_), 0u);
  const loom_model_file_t path = {iree_make_cstring_view(first_.path().c_str()),
                                  {}};
  const loom_model_file_list_t files = {1, &path};
  IREE_ASSERT_OK(loom_serve_parameter_sources_create(
      nullptr, &namespace_, 1, &files, allocator_, &prefix_));
  IREE_ASSERT_OK(loom_serve_parameter_sources_create(
      prefix_, &namespace_, 0, nullptr, iree_allocator_null(), &catalog_));
  EXPECT_EQ(catalog_, prefix_);
  loom_serve_parameter_sources_release(prefix_);
  prefix_ = nullptr;
  ExpectEntry(catalog_, 0, "same", 16);
}

TEST_F(ParameterSourcesTest, FailedExtensionDoesNotChangeThePrefix) {
  const loom_model_file_t paths[] = {
      {iree_make_cstring_view(first_.path().c_str()), {}},
      {iree_make_cstring_view(second_.path().c_str()), IREE_SV("safetensors")},
      {iree_make_cstring_view(missing_.path().c_str()), {}}};
  const loom_model_file_list_t base = {1, paths};
  const loom_model_file_list_t extension[] = {{1, paths + 1}, {1, paths + 2}};
  IREE_ASSERT_OK(loom_serve_parameter_sources_create(
      nullptr, &namespace_, 1, &base, allocator_, &prefix_));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_serve_parameter_sources_create(prefix_, &namespace_, 2, extension,
                                          allocator_, &catalog_));
  EXPECT_EQ(catalog_, nullptr);
  EXPECT_EQ(loom_serve_parameter_sources_count(prefix_), 1u);
  ExpectEntry(prefix_, 0, "same", 16);
}

}  // namespace
