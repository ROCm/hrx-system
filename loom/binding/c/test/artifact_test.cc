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

#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/temp_file.h"
#include "loomc/iree.h"
#include "loomc/status.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ByteSequencePtr =
    HandlePtr<loomc_byte_sequence_t, loomc_byte_sequence_release>;
using StreamPtr = HandlePtr<iree_io_stream_t, iree_io_stream_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

std::string ToString(loomc_byte_span_t value) {
  return value.data ? std::string(reinterpret_cast<const char*>(value.data),
                                  value.data_length)
                    : std::string();
}

std::string ToString(const loomc_byte_sequence_t* value) {
  loomc_byte_span_t contents = loomc_byte_span_empty();
  LOOMC_EXPECT_OK(
      loomc_byte_sequence_clone(value, loomc_allocator_system(), &contents));
  std::string result =
      contents.data ? std::string(reinterpret_cast<const char*>(contents.data),
                                  contents.data_length)
                    : std::string();
  loomc_allocator_free(loomc_allocator_system(), (void*)contents.data);
  return result;
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

loomc_artifact_t MakeTextModuleArtifact(loomc_byte_sequence_t** out_contents) {
  static const char kContents[] =
      "func.def public @entry(%x: i32) -> (i32) {\n"
      "  func.return %x : i32\n"
      "}\n";
  LOOMC_EXPECT_OK(loomc_byte_sequence_create_copy(
      loomc_make_byte_span(kContents, strlen(kContents)),
      loomc_allocator_system(), out_contents));
  return {
      /*.kind=*/LOOMC_ARTIFACT_KIND_MODULE,
      /*.format=*/loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_TEXT),
      /*.identifier=*/loomc_make_cstring_view("entry.loom"),
      /*.contents=*/*out_contents,
  };
}

ByteSequencePtr MakeSegmentedSequence(const std::string& contents) {
  iree_io_stream_t* stream = nullptr;
  IREE_EXPECT_OK(iree_io_vec_stream_create(IREE_IO_STREAM_MODE_WRITABLE,
                                           /*block_size=*/1024,
                                           iree_allocator_system(), &stream));
  StreamPtr stream_owner(stream);
  IREE_EXPECT_OK(
      iree_io_stream_write(stream, contents.size(), contents.data()));
  iree_byte_sequence_t* sequence = nullptr;
  IREE_EXPECT_OK(iree_io_vec_stream_move_contents(stream, &sequence));
  return ByteSequencePtr(loomc_byte_sequence_from_iree(sequence));
}

TEST(ArtifactTest, CreateSourceRetainsContiguousBytesAndInfersFormat) {
  loomc_byte_sequence_t* contents = nullptr;
  loomc_artifact_t artifact = MakeTextModuleArtifact(&contents);
  HandlePtr<loomc_byte_sequence_t, loomc_byte_sequence_release> contents_owner(
      contents);

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
  loomc_byte_span_t artifact_span = loomc_byte_span_empty();
  ASSERT_TRUE(loomc_byte_sequence_try_get_contiguous_span(artifact.contents,
                                                          &artifact_span));
  EXPECT_EQ(loomc_source_contents(source_ptr.get()).data, artifact_span.data);
}

TEST(ArtifactTest, WriteToOpenFile) {
  loomc_byte_sequence_t* contents = nullptr;
  loomc_artifact_t artifact = MakeTextModuleArtifact(&contents);
  HandlePtr<loomc_byte_sequence_t, loomc_byte_sequence_release> contents_owner(
      contents);
  FILE* file = tmpfile();
  ASSERT_NE(file, nullptr);

  loomc_status_t status = loomc_artifact_write_to_file(&artifact, file);
  LOOMC_EXPECT_OK(status);
  const std::string expected = ToString(artifact.contents);
  EXPECT_EQ(ReadOpenFileBytes(file),
            std::vector<uint8_t>(expected.begin(), expected.end()));

  EXPECT_EQ(fclose(file), 0);
}

TEST(ArtifactTest, WriteToPath) {
  loomc_byte_sequence_t* contents = nullptr;
  loomc_artifact_t artifact = MakeTextModuleArtifact(&contents);
  HandlePtr<loomc_byte_sequence_t, loomc_byte_sequence_release> contents_owner(
      contents);
  iree::testing::TempFilePath path("loomc_artifact", ".loom");

  loomc_status_t status = loomc_artifact_write_to_path(
      &artifact, loomc_make_string_view(path.path().data(), path.path().size()),
      loomc_allocator_system());
  LOOMC_EXPECT_OK(status);
  EXPECT_EQ(ReadPathToString(path.path()), ToString(artifact.contents));

  EXPECT_TRUE(path.Remove());
}

TEST(ArtifactTest, SegmentedContentsRemainComposable) {
  const std::string expected(2048, 'x');
  ByteSequencePtr contents = MakeSegmentedSequence(expected);
  loomc_byte_span_t contiguous_span = loomc_byte_span_empty();
  ASSERT_FALSE(loomc_byte_sequence_try_get_contiguous_span(contents.get(),
                                                           &contiguous_span));
  loomc_artifact_t artifact = {
      /*.kind=*/LOOMC_ARTIFACT_KIND_TEXT,
      /*.format=*/loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_TEXT),
      /*.identifier=*/loomc_make_cstring_view("segmented.loom"),
      /*.contents=*/contents.get(),
  };

  loomc_source_t* source = nullptr;
  LOOMC_ASSERT_OK(
      loomc_artifact_create_source(&artifact, LOOMC_SOURCE_FORMAT_UNKNOWN,
                                   loomc_allocator_system(), &source));
  SourcePtr source_owner(source);
  contents.reset();
  EXPECT_EQ(ToString(loomc_source_contents(source)), expected);

  ByteSequencePtr writable_contents = MakeSegmentedSequence(expected);
  artifact.contents = writable_contents.get();
  FILE* file = tmpfile();
  ASSERT_NE(file, nullptr);
  LOOMC_EXPECT_OK(loomc_artifact_write_to_file(&artifact, file));
  EXPECT_EQ(ReadOpenFileBytes(file),
            std::vector<uint8_t>(expected.begin(), expected.end()));
  EXPECT_EQ(fclose(file), 0);
}

}  // namespace
