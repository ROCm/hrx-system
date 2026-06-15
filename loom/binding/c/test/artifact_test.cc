// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/artifact.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/temp_file.h"
#include "loomc/status.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

std::string ToString(loomc_byte_span_t value) {
  return value.data ? std::string(reinterpret_cast<const char*>(value.data),
                                  value.data_length)
                    : std::string();
}

std::string ReadPathToString(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  EXPECT_TRUE(input.good());
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::vector<uint8_t> ReadOpenFileBytes(FILE* file) {
  EXPECT_EQ(fflush(file), 0);
  EXPECT_EQ(fseek(file, 0, SEEK_END), 0);
  long file_length = ftell(file);
  EXPECT_GE(file_length, 0);
  if (file_length < 0) {
    return {};
  }
  EXPECT_EQ(fseek(file, 0, SEEK_SET), 0);
  std::vector<uint8_t> bytes((size_t)file_length);
  if (!bytes.empty()) {
    EXPECT_EQ(fread(bytes.data(), 1, bytes.size(), file), bytes.size());
  }
  return bytes;
}

loomc_artifact_t MakeTextModuleArtifact() {
  static const char kContents[] =
      "func.def public @entry(%x: i32) -> (i32) {\n"
      "  func.return %x : i32\n"
      "}\n";
  loomc_artifact_t artifact = {
      /*.kind=*/LOOMC_ARTIFACT_KIND_MODULE,
      /*.format=*/loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_TEXT),
      /*.identifier=*/loomc_make_cstring_view("entry.loom"),
      /*.contents=*/loomc_make_byte_span(kContents, strlen(kContents)),
  };
  return artifact;
}

TEST(ArtifactTest, CreateSourceCopiesBytesAndInfersFormat) {
  loomc_artifact_t artifact = MakeTextModuleArtifact();

  loomc_source_t* source = nullptr;
  loomc_status_t status =
      loomc_artifact_create_source(&artifact, LOOMC_SOURCE_FORMAT_UNKNOWN,
                                   loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  SourcePtr source_ptr(source);

  ASSERT_NE(source_ptr.get(), nullptr);
  EXPECT_EQ(loomc_source_format(source_ptr.get()), LOOMC_SOURCE_FORMAT_TEXT);
  EXPECT_EQ(ToString(loomc_source_identifier(source_ptr.get())), "entry.loom");
  EXPECT_EQ(ToString(loomc_source_contents(source_ptr.get())),
            ToString(artifact.contents));
  EXPECT_NE(loomc_source_contents(source_ptr.get()).data,
            artifact.contents.data);
}

TEST(ArtifactTest, WriteToOpenFile) {
  loomc_artifact_t artifact = MakeTextModuleArtifact();
  FILE* file = tmpfile();
  ASSERT_NE(file, nullptr);

  loomc_status_t status = loomc_artifact_write_to_file(&artifact, file);
  LOOMC_EXPECT_OK(status);
  EXPECT_EQ(ReadOpenFileBytes(file),
            std::vector<uint8_t>(
                artifact.contents.data,
                artifact.contents.data + artifact.contents.data_length));

  EXPECT_EQ(fclose(file), 0);
}

TEST(ArtifactTest, WriteToPath) {
  loomc_artifact_t artifact = MakeTextModuleArtifact();
  iree::testing::TempFilePath path("loomc_artifact", ".loom");

  loomc_status_t status = loomc_artifact_write_to_path(
      &artifact, loomc_make_string_view(path.path().data(), path.path().size()),
      loomc_allocator_system());
  LOOMC_EXPECT_OK(status);
  EXPECT_EQ(ReadPathToString(path.path()), ToString(artifact.contents));

  EXPECT_TRUE(path.Remove());
}

}  // namespace
