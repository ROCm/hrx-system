// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/source.h"

#include <cstdio>
#include <fstream>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/temp_file.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;

std::string ToString(loomc_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string ToString(loomc_byte_span_t value) {
  return std::string(reinterpret_cast<const char*>(value.data),
                     value.data_length);
}

TEST(SourceTest, CopiesIdentifierAndRequestedContents) {
  char contents[] = "func.def @entry";
  loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view("kernel.loom"),
      /*.contents=*/loomc_make_byte_span(contents, sizeof(contents) - 1),
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  LOOMC_ASSERT_OK(status);

  contents[0] = 'X';
  EXPECT_EQ(loomc_source_format(source), LOOMC_SOURCE_FORMAT_TEXT);
  EXPECT_EQ(ToString(loomc_source_identifier(source)), "kernel.loom");
  EXPECT_EQ(ToString(loomc_source_contents(source)), "func.def @entry");

  loomc_source_release(source);
}

TEST(SourceTest, BorrowsContents) {
  char contents[] = "borrowed";
  loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_NONE,
      /*.structure_size=*/0,
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view("borrowed.loom"),
      /*.contents=*/loomc_make_byte_span(contents, sizeof(contents) - 1),
      /*.storage=*/LOOMC_SOURCE_STORAGE_BORROWED,
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  LOOMC_ASSERT_OK(status);

  contents[0] = 'B';
  EXPECT_EQ(ToString(loomc_source_contents(source)), "Borrowed");

  loomc_source_release(source);
}

typedef struct ExternalReleaseState {
  // Number of release callback invocations.
  int release_count;
  // Contents observed by the release callback.
  loomc_byte_span_t contents;
} ExternalReleaseState;

void ExternalRelease(void* user_data, loomc_byte_span_t contents) {
  auto* state = static_cast<ExternalReleaseState*>(user_data);
  ++state->release_count;
  state->contents = contents;
}

TEST(SourceTest, ReleasesExternalContentsAfterTakingOwnership) {
  char contents[] = "external";
  ExternalReleaseState state = {};
  loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_NONE,
      /*.structure_size=*/0,
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view("external.loom"),
      /*.contents=*/loomc_make_byte_span(contents, sizeof(contents) - 1),
      /*.storage=*/LOOMC_SOURCE_STORAGE_EXTERNAL,
      /*.release=*/ExternalRelease,
      /*.release_user_data=*/&state,
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  LOOMC_ASSERT_OK(status);
  EXPECT_EQ(state.release_count, 0);

  loomc_source_release(source);

  EXPECT_EQ(state.release_count, 1);
  EXPECT_EQ(ToString(state.contents), "external");
}

TEST(SourceTest, CreatesFromOpenFileAndOwnsContents) {
  FILE* file = tmpfile();
  ASSERT_NE(file, nullptr);
  static const char kContents[] = "loaded from FILE";
  ASSERT_EQ(fwrite(kContents, 1, sizeof(kContents) - 1, file),
            sizeof(kContents) - 1);
  ASSERT_EQ(fseek(file, 0, SEEK_SET), 0);

  loomc_source_load_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_LOAD_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view("open-file.loom"),
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status = loomc_source_create_from_file(
      file, &options, loomc_allocator_system(), &source);
  LOOMC_ASSERT_OK(status);
  SourcePtr source_ptr(source);
  ASSERT_EQ(fclose(file), 0);

  EXPECT_EQ(loomc_source_format(source_ptr.get()), LOOMC_SOURCE_FORMAT_TEXT);
  EXPECT_EQ(ToString(loomc_source_identifier(source_ptr.get())),
            "open-file.loom");
  EXPECT_EQ(ToString(loomc_source_contents(source_ptr.get())),
            "loaded from FILE");
}

TEST(SourceTest, CreatesFromPathWithPathIdentifier) {
  iree::testing::TempFilePath path("loomc_source", ".json");
  static const char kContents[] = "{\"answer\":42}";
  {
    std::ofstream output(path.path(), std::ios::binary);
    output << kContents;
    output.close();
    ASSERT_TRUE(output.good());
  }

  loomc_source_t* source = nullptr;
  loomc_status_t status = loomc_source_create_from_path(
      loomc_make_string_view(path.path().data(), path.path().size()), nullptr,
      loomc_allocator_system(), &source);
  LOOMC_ASSERT_OK(status);
  SourcePtr source_ptr(source);

  EXPECT_EQ(loomc_source_format(source_ptr.get()), LOOMC_SOURCE_FORMAT_UNKNOWN);
  EXPECT_EQ(ToString(loomc_source_identifier(source_ptr.get())), path.path());
  EXPECT_EQ(ToString(loomc_source_contents(source_ptr.get())),
            "{\"answer\":42}");

  EXPECT_TRUE(path.Remove());
}

TEST(SourceTest, RejectsInvalidOptions) {
  loomc_source_t* source = reinterpret_cast<loomc_source_t*>(0x1);
  loomc_status_t status =
      loomc_source_create(nullptr, loomc_allocator_system(), &source);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(source, nullptr);
}

TEST(SourceTest, RejectsInvalidLoadRequests) {
  loomc_source_t* source = reinterpret_cast<loomc_source_t*>(0x1);
  loomc_status_t status = loomc_source_create_from_file(
      nullptr, nullptr, loomc_allocator_system(), &source);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(source, nullptr);

  source = reinterpret_cast<loomc_source_t*>(0x1);
  status = loomc_source_create_from_path(loomc_string_view_empty(), nullptr,
                                         loomc_allocator_system(), &source);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(source, nullptr);
}

}  // namespace
