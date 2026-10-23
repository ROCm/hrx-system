// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/file.h"

#include <string>

#include "iree/io/file_contents.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/target/provider.h"
#include "loomc/interop.h"
#include "loomc/iree.h"

namespace loom {
namespace {

using ::testing::HasSubstr;

iree_string_view_t StringView(const std::string& value) {
  return iree_make_string_view(value.data(), value.size());
}

struct CaseCounts {
  iree_host_size_t pass_count = 0;
  iree_host_size_t fail_count = 0;
  iree_host_size_t skip_count = 0;
};

class FileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    target_provider_set_ = loom_target_provider_set_make(nullptr, 0);
    compile_session_.host_allocator = iree_allocator_system();
    IREE_ASSERT_OK(iree_status_from_loomc(
        loomc_target_environment_create_from_provider_set(
            &target_provider_set_, loomc_allocator_system(),
            &compile_session_.target_environment)));
    environment_ = {};
    environment_.target_environment = loomc_target_environment_get_interop_view(
        compile_session_.target_environment);
    environment_.compile_session = &compile_session_;
  }

  void TearDown() override {
    loom_check_compile_session_deinitialize(&compile_session_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t WriteFile(const std::string& path,
                          const std::string& contents) {
    return iree_io_file_contents_write(
        StringView(path),
        iree_make_const_byte_span(contents.data(), contents.size()),
        iree_allocator_system());
  }

  std::string ReadFile(const std::string& path) {
    iree_io_file_contents_t* contents = nullptr;
    iree_status_t status = iree_io_file_contents_read(
        StringView(path), iree_allocator_system(), &contents);
    if (!iree_status_is_ok(status)) {
      IREE_EXPECT_OK(status);
      return {};
    }
    const iree_string_view_t contents_view = iree_make_string_view(
        reinterpret_cast<const char*>(contents->const_buffer.data),
        contents->const_buffer.data_length);
    std::string result(contents_view.data, contents_view.size);
    iree_io_file_contents_free(contents);
    return result;
  }

  iree_status_t Process(const std::string& path, loom_check_process_mode_t mode,
                        CaseCounts* out_counts) {
    loom_check_process_options_t options = {.mode = mode};
    return loom_check_read_and_process(
        StringView(path), &options, &environment_, &block_pool_,
        iree_allocator_system(), &out_counts->pass_count,
        &out_counts->fail_count, &out_counts->skip_count);
  }

  iree_arena_block_pool_t block_pool_;
  loom_target_provider_set_t target_provider_set_;
  loom_check_compile_session_t compile_session_ = {};
  loom_check_environment_t environment_;
};

TEST_F(FileTest, UpdatePreservesPassingAndFailingChecks) {
  iree::testing::TempFilePath path("loom_check_patterns", ".loom-test");
  const std::string source =
      "// RUN: with-checks roundtrip\nfunc.def @alpha() {}\n"
      "// ----\nCHECK: func.def @alpha() {\n"
      "// ====\nfunc.def @beta() {}\n"
      "// ----\nCHECK: func.def @wrong() {\n";
  IREE_ASSERT_OK(WriteFile(path.path(), source));
  CaseCounts counts;
  IREE_ASSERT_OK(Process(path.path(), LOOM_CHECK_PROCESS_UPDATE, &counts));
  EXPECT_EQ(counts.pass_count, 1u);
  EXPECT_EQ(counts.fail_count, 1u);
  EXPECT_EQ(ReadFile(path.path()), source);
}

TEST_F(FileTest, JsonDoesNotSuggestReplacingChecksWithGoldens) {
  iree::testing::TempFilePath path("loom_check_patterns_json", ".loom-test");
  const std::string source =
      "// RUN: with-checks roundtrip\nfunc.def @alpha() {}\n"
      "// ----\nCHECK: func.def @wrong() {\n";
  IREE_ASSERT_OK(WriteFile(path.path(), source));
  loom_check_process_options_t options = {
      .json_enabled = true,
      .json_output_mode = LOOM_CHECK_JSON_OUTPUT_ALL,
  };
  CaseCounts counts;
  ::testing::internal::CaptureStdout();
  iree_status_t status = loom_check_read_and_process(
      StringView(path.path()), &options, &environment_, &block_pool_,
      iree_allocator_system(), &counts.pass_count, &counts.fail_count,
      &counts.skip_count);
  const std::string output = ::testing::internal::GetCapturedStdout();
  IREE_ASSERT_OK(status);
  EXPECT_EQ(counts.fail_count, 1u);
  EXPECT_THAT(output, HasSubstr("did not match"));
  EXPECT_THAT(output, HasSubstr("\"update_edit\":null"));
  EXPECT_EQ(ReadFile(path.path()), source);
}

}  // namespace
}  // namespace loom
