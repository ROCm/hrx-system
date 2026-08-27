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
#include "loom/testing/context.h"

namespace loom {
namespace {

using ::iree::testing::status::StatusIs;
using ::testing::HasSubstr;

iree_string_view_t StringView(const std::string& value) {
  return iree_make_string_view(value.data(), value.size());
}

std::string RootRelativeName(const std::string& path) {
  const std::string::size_type separator_position = path.find_last_of("/\\");
  return separator_position == std::string::npos
             ? path
             : path.substr(separator_position + 1);
}

iree_status_t RegisterFileTestContext(void* user_data,
                                      loom_context_t* context) {
  (void)user_data;
  return loom_testing_context_register_all_dialects(context);
}

iree_status_t InitializeFileTestLowDescriptorRegistry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  (void)user_data;
  *out_registry = {};
  return iree_ok_status();
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
    loom_context_initialize(iree_allocator_system(), &context_);
    environment_ = {};
    environment_.register_context.fn = RegisterFileTestContext;
    environment_.initialize_low_descriptor_registry.fn =
        InitializeFileTestLowDescriptorRegistry;
    IREE_ASSERT_OK(
        loom_check_context_register_and_finalize(&environment_, &context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
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

  iree_status_t Process(const std::string& path,
                        const std::string& template_root, bool update,
                        CaseCounts* out_counts) {
    loom_check_process_options_t options = {};
    options.update = update;
    options.template_root = StringView(template_root);
    return loom_check_read_and_process(
        StringView(path), &options, &environment_, &context_, &block_pool_,
        iree_allocator_system(), &out_counts->pass_count,
        &out_counts->fail_count, &out_counts->skip_count);
  }

  const std::string template_root_ = ::testing::TempDir();
  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_check_environment_t environment_;
};

TEST_F(FileTest, FreshTemplatePassesWithoutMutation) {
  iree::testing::TempFilePath template_path("loom_check_template",
                                            ".loom-test");
  iree::testing::TempFilePath target_path("loom_check_target", ".loom-test");
  const std::string template_source =
      "// RUN: roundtrip\n"
      "\n"
      "func.def @alpha() {\n"
      "}\n";
  const std::string target_source =
      "// TEMPLATE: " + RootRelativeName(template_path.path()) + "\n" +
      "// RUN: roundtrip\n"
      "\n"
      "func.def @alpha() {\n"
      "}\n";
  IREE_ASSERT_OK(WriteFile(template_path.path(), template_source));
  IREE_ASSERT_OK(WriteFile(target_path.path(), target_source));

  CaseCounts counts;
  IREE_ASSERT_OK(
      Process(target_path.path(), template_root_, /*update=*/false, &counts));

  EXPECT_EQ(counts.pass_count, 1u);
  EXPECT_EQ(counts.fail_count, 0u);
  EXPECT_EQ(counts.skip_count, 0u);
  EXPECT_EQ(ReadFile(target_path.path()), target_source);
}

TEST_F(FileTest, StaleTemplateFailsBeforeXfailCaseExecution) {
  iree::testing::TempFilePath template_path("loom_check_template",
                                            ".loom-test");
  iree::testing::TempFilePath target_path("loom_check_target", ".loom-test");
  const std::string template_source =
      "// RUN: roundtrip\n"
      "\n"
      "func.def @alpha() {\n"
      "}\n"
      "\n"
      "// ====\n"
      "\n"
      "func.def @beta() {\n"
      "}\n";
  const std::string target_source =
      "// TEMPLATE: " + RootRelativeName(template_path.path()) + "\n" +
      "// RUN: roundtrip\n"
      "\n"
      "// XFAIL: would be an unexpected pass if this stale case ran\n"
      "func.def @alpha() {\n"
      "}\n";
  IREE_ASSERT_OK(WriteFile(template_path.path(), template_source));
  IREE_ASSERT_OK(WriteFile(target_path.path(), target_source));

  CaseCounts counts;
  iree::Status status(
      Process(target_path.path(), template_root_, /*update=*/false, &counts));

  EXPECT_THAT(status, StatusIs(iree::StatusCode::kFailedPrecondition));
  EXPECT_THAT(status.ToString(), HasSubstr("is stale relative to"));
  EXPECT_THAT(status.ToString(),
              HasSubstr(RootRelativeName(template_path.path())));
  EXPECT_THAT(status.ToString(), HasSubstr("--template-root="));
  EXPECT_THAT(status.ToString(), HasSubstr("--update"));
  EXPECT_EQ(counts.pass_count, 0u);
  EXPECT_EQ(counts.fail_count, 0u);
  EXPECT_EQ(counts.skip_count, 0u);
  EXPECT_EQ(ReadFile(target_path.path()), target_source);
}

TEST_F(FileTest, MissingTemplateRootFailsWithoutCheckoutFallback) {
  iree::testing::TempFilePath template_path("loom_check_template",
                                            ".loom-test");
  iree::testing::TempFilePath target_path("loom_check_target", ".loom-test");
  const std::string template_source =
      "// RUN: roundtrip\n"
      "\n"
      "func.def @alpha() {\n"
      "}\n";
  const std::string target_source =
      "// TEMPLATE: " + RootRelativeName(template_path.path()) + "\n" +
      "// RUN: roundtrip\n"
      "\n"
      "func.def @alpha() {\n"
      "}\n";
  IREE_ASSERT_OK(WriteFile(template_path.path(), template_source));
  IREE_ASSERT_OK(WriteFile(target_path.path(), target_source));

  const std::string missing_root = template_root_ + "missing-template-root";
  CaseCounts counts;
  iree::Status status(
      Process(target_path.path(), missing_root, /*update=*/false, &counts));

  EXPECT_THAT(status, StatusIs(iree::StatusCode::kNotFound));
  EXPECT_THAT(status.ToString(), HasSubstr("reading TEMPLATE"));
  EXPECT_THAT(status.ToString(),
              HasSubstr(RootRelativeName(template_path.path())));
  EXPECT_THAT(status.ToString(), HasSubstr(target_path.path()));
  EXPECT_EQ(counts.pass_count, 0u);
  EXPECT_EQ(counts.fail_count, 0u);
  EXPECT_EQ(counts.skip_count, 0u);
  EXPECT_EQ(ReadFile(target_path.path()), target_source);
}

TEST_F(FileTest, UpdateSynchronizesOnceAndThenIsByteStable) {
  iree::testing::TempFilePath template_path("loom_check_template",
                                            ".loom-test");
  iree::testing::TempFilePath target_path("loom_check_target", ".loom-test");
  const std::string template_source =
      "// RUN: roundtrip\n"
      "\n"
      "func.def @alpha() {\n"
      "}\n"
      "\n"
      "// ====\n"
      "\n"
      "func.def @beta() {\n"
      "}\n";
  const std::string target_source =
      "// TEMPLATE: " + RootRelativeName(template_path.path()) + "\n" +
      "// RUN: roundtrip\n"
      "\n"
      "func.def @alpha() {\n"
      "}\n";
  IREE_ASSERT_OK(WriteFile(template_path.path(), template_source));
  IREE_ASSERT_OK(WriteFile(target_path.path(), target_source));

  CaseCounts first_counts;
  IREE_ASSERT_OK(Process(target_path.path(), template_root_, /*update=*/true,
                         &first_counts));
  const std::string synchronized_source = ReadFile(target_path.path());
  EXPECT_NE(synchronized_source, target_source);
  EXPECT_THAT(synchronized_source, HasSubstr("func.def @beta()"));
  EXPECT_EQ(first_counts.pass_count, 2u);
  EXPECT_EQ(first_counts.fail_count, 0u);
  EXPECT_EQ(first_counts.skip_count, 0u);

  CaseCounts second_counts;
  IREE_ASSERT_OK(Process(target_path.path(), template_root_, /*update=*/true,
                         &second_counts));
  EXPECT_EQ(ReadFile(target_path.path()), synchronized_source);
  EXPECT_EQ(second_counts.pass_count, 2u);
  EXPECT_EQ(second_counts.fail_count, 0u);
  EXPECT_EQ(second_counts.skip_count, 0u);

  CaseCounts read_only_counts;
  IREE_ASSERT_OK(Process(target_path.path(), template_root_, /*update=*/false,
                         &read_only_counts));
  EXPECT_EQ(ReadFile(target_path.path()), synchronized_source);
  EXPECT_EQ(read_only_counts.pass_count, 2u);
  EXPECT_EQ(read_only_counts.fail_count, 0u);
  EXPECT_EQ(read_only_counts.skip_count, 0u);
}

TEST_F(FileTest, FreshTemplatePreservesXfailAndXpassSemantics) {
  iree::testing::TempFilePath template_path("loom_check_template",
                                            ".loom-test");
  iree::testing::TempFilePath target_path("loom_check_target", ".loom-test");
  const std::string template_source =
      "// RUN: roundtrip\n"
      "\n"
      "func.def @expected_failure() {\n"
      "}\n"
      "\n"
      "// ====\n"
      "\n"
      "func.def @unexpected_pass() {\n"
      "}\n";
  const std::string target_source =
      "// TEMPLATE: " + RootRelativeName(template_path.path()) + "\n" +
      "// RUN: roundtrip\n"
      "\n"
      "// XFAIL: expected mismatch\n"
      "func.def @expected_failure() {\n"
      "}\n"
      "\n"
      "// ----\n"
      "func.def @different() {\n"
      "}\n"
      "\n"
      "// ====\n"
      "\n"
      "// XFAIL: obsolete expectation\n"
      "func.def @unexpected_pass() {\n"
      "}\n";
  IREE_ASSERT_OK(WriteFile(template_path.path(), template_source));
  IREE_ASSERT_OK(WriteFile(target_path.path(), target_source));

  CaseCounts counts;
  IREE_ASSERT_OK(
      Process(target_path.path(), template_root_, /*update=*/false, &counts));

  EXPECT_EQ(counts.pass_count, 1u);
  EXPECT_EQ(counts.fail_count, 1u);
  EXPECT_EQ(counts.skip_count, 0u);
  EXPECT_EQ(ReadFile(target_path.path()), target_source);
}

}  // namespace
}  // namespace loom
