// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/dependency_report.h"

#include <array>
#include <memory>
#include <string>
#include <string_view>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/link/module_index.h"
#include "loom/link/testdata/dependency_analysis_testdata.h"
#include "loom/ops/op_registry.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

struct IndexDeleter {
  void operator()(loom_link_module_index_t* index) const {
    loom_link_module_index_free(index);
  }
};
using IndexPtr = std::unique_ptr<loom_link_module_index_t, IndexDeleter>;

class LinkDependencyReportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_string_view_t FindSource(std::string_view name) {
    const iree_file_toc_t* files =
        loom_link_dependency_analysis_testdata_create();
    for (iree_host_size_t i = 0;
         i < loom_link_dependency_analysis_testdata_size(); ++i) {
      if (name == files[i].name) {
        return iree_make_string_view(files[i].data, files[i].size);
      }
    }
    return iree_string_view_empty();
  }

  iree_host_size_t AddText(loom_link_module_index_t* index,
                           std::string_view filename,
                           std::string_view provider_name,
                           loom_link_provider_role_t role) {
    const iree_string_view_t source = FindSource(filename);
    EXPECT_FALSE(iree_string_view_is_empty(source));
    const loom_link_module_index_add_options_t options = {
        /*.provider_name=*/
        iree_make_string_view(provider_name.data(), provider_name.size()),
        /*.role=*/role,
    };
    iree_host_size_t provider_ordinal = 0;
    IREE_EXPECT_OK(loom_link_module_index_add_text(
        index, source, iree_make_string_view(filename.data(), filename.size()),
        /*parse_options=*/nullptr, &options, &provider_ordinal));
    return provider_ordinal;
  }

  IndexPtr CreateIndex(std::array<iree_host_size_t, 2>* direct_providers) {
    loom_link_module_index_t* raw_index = nullptr;
    IREE_CHECK_OK(loom_link_module_index_allocate(
        &context_, &block_pool_, iree_allocator_system(), &raw_index));
    IndexPtr index(raw_index);
    AddText(index.get(), "input.loom", "//model:a",
            LOOM_LINK_PROVIDER_ROLE_INPUT);
    AddText(index.get(), "input_local.loom", "//model:a#local",
            LOOM_LINK_PROVIDER_ROLE_INPUT);
    (*direct_providers)[0] =
        AddText(index.get(), "direct.loom", "//kernel:direct",
                LOOM_LINK_PROVIDER_ROLE_LIBRARY);
    AddText(index.get(), "direct_duplicate.loom", "//kernel:duplicate",
            LOOM_LINK_PROVIDER_ROLE_LIBRARY);
    AddText(index.get(), "transitive.loom", "//kernel:transitive",
            LOOM_LINK_PROVIDER_ROLE_LIBRARY);
    (*direct_providers)[1] =
        AddText(index.get(), "unused.loom", "//kernel:unused",
                LOOM_LINK_PROVIDER_ROLE_LIBRARY);
    return index;
  }

  loom_link_dependency_analysis_t Analyze(
      const loom_link_module_index_t* index,
      const std::array<iree_host_size_t, 2>& direct_providers) {
    const loom_link_dependency_analysis_options_t options = {
        /*.direct_provider_ordinals=*/direct_providers.data(),
        /*.direct_provider_count=*/direct_providers.size(),
    };
    loom_link_dependency_analysis_t analysis = {};
    IREE_CHECK_OK(loom_link_dependency_analyze(
        index, &options, &block_pool_, &analysis_arena_,
        iree_allocator_system(), &analysis));
    return analysis;
  }

  std::string FormatJson(const loom_link_dependency_analysis_t& analysis) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&builder, &stream);
    IREE_EXPECT_OK(loom_link_dependency_format_json(
        &analysis, IREE_SV("//model:a"), &stream));
    const iree_string_view_t value = iree_string_builder_view(&builder);
    std::string result(value.data, value.size);
    iree_string_builder_deinitialize(&builder);
    return result;
  }

  std::string FormatDiagnostic(
      const loom_link_dependency_analysis_t& analysis,
      const loom_link_dependency_requirement_t& requirement) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&builder, &stream);
    IREE_EXPECT_OK(loom_link_dependency_format_diagnostic(
        &analysis, &requirement, IREE_SV("//model:a"), &stream));
    const iree_string_view_t value = iree_string_builder_view(&builder);
    std::string result(value.data, value.size);
    iree_string_builder_deinitialize(&builder);
    return result;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  iree_arena_allocator_t analysis_arena_;
};

TEST_F(LinkDependencyReportTest, FormatsDeterministicStructuredJson) {
  std::array<iree_host_size_t, 2> direct_providers = {};
  IndexPtr index = CreateIndex(&direct_providers);
  const loom_link_dependency_analysis_t analysis =
      Analyze(index.get(), direct_providers);

  EXPECT_FALSE(loom_link_dependency_analysis_succeeded(&analysis));
  const std::string first = FormatJson(analysis);
  const std::string second = FormatJson(analysis);
  EXPECT_EQ(first, second);
  EXPECT_THAT(first, ::testing::StartsWith(
                         R"({"schema_version":1,"component":"//model:a",)"));
  EXPECT_THAT(first, ::testing::HasSubstr(R"("dependency":"missing_direct")"));
  EXPECT_THAT(first, ::testing::HasSubstr(R"("resolution":"ambiguous")"));
  EXPECT_THAT(first, ::testing::HasSubstr(
                         R"("usage":["exact","interface","template"])"));
  EXPECT_THAT(first, ::testing::HasSubstr(
                         R"({"provider":"//kernel:unused","usage":[]})"));
}

TEST_F(LinkDependencyReportTest, FormatsStableFailureDiagnostics) {
  std::array<iree_host_size_t, 2> direct_providers = {};
  IndexPtr index = CreateIndex(&direct_providers);
  const loom_link_dependency_analysis_t analysis =
      Analyze(index.get(), direct_providers);

  bool found_ambiguous = false;
  bool found_missing_direct = false;
  bool found_unsatisfied = false;
  bool found_inaccessible = false;
  bool found_incompatible = false;
  for (iree_host_size_t i = 0; i < analysis.requirements.count; ++i) {
    const loom_link_dependency_requirement_t& requirement =
        analysis.requirements.values[i];
    if (loom_link_dependency_requirement_satisfied(&requirement)) {
      continue;
    }
    const iree_string_view_t code =
        loom_link_dependency_diagnostic_code(&requirement);
    const std::string code_string(code.data, code.size);
    const std::string message = FormatDiagnostic(analysis, requirement);
    if (code_string == "LINK/DEPENDENCY/AMBIGUOUS") {
      found_ambiguous = true;
      EXPECT_THAT(message,
                  ::testing::HasSubstr("multiple compatible definitions"));
    } else if (code_string == "LINK/DEPENDENCY/MISSING_DIRECT") {
      found_missing_direct = true;
      EXPECT_THAT(message,
                  ::testing::HasSubstr("no direct dependency of //model:a"));
    } else if (code_string == "LINK/DEPENDENCY/UNSATISFIED") {
      found_unsatisfied = true;
      EXPECT_THAT(message, ::testing::HasSubstr("no supplied library exports"));
    } else if (code_string == "LINK/DEPENDENCY/INACCESSIBLE") {
      found_inaccessible = true;
      EXPECT_THAT(message, ::testing::HasSubstr("definition is private"));
    } else if (code_string == "LINK/DEPENDENCY/INCOMPATIBLE") {
      found_incompatible = true;
      EXPECT_THAT(message, ::testing::HasSubstr("contract is incompatible"));
    }
  }
  EXPECT_TRUE(found_ambiguous);
  EXPECT_TRUE(found_missing_direct);
  EXPECT_TRUE(found_unsatisfied);
  EXPECT_TRUE(found_inaccessible);
  EXPECT_TRUE(found_incompatible);
}

}  // namespace
}  // namespace loom
