// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/link_dependency.h"

#include <array>
#include <memory>
#include <string>
#include <string_view>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "loom/binding/c/test/testdata/link_dependency_testdata.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using BuilderPtr =
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using LinkIndexPtr = HandlePtr<loomc_link_index_t, loomc_link_index_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

std::string ToString(loomc_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string ToString(loomc_byte_span_t value) {
  return std::string(reinterpret_cast<const char*>(value.data),
                     value.data_length);
}

class LinkDependencyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loomc_context_t* context = nullptr;
    LOOMC_ASSERT_OK(
        loomc_context_create(nullptr, loomc_allocator_system(), &context));
    context_.reset(context);

    loomc_workspace_t* workspace = nullptr;
    LOOMC_ASSERT_OK(
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
    workspace_.reset(workspace);

    loomc_link_index_builder_t* builder = nullptr;
    LOOMC_ASSERT_OK(loomc_link_index_builder_create(
        context_.get(), nullptr, loomc_allocator_system(), &builder));
    BuilderPtr builder_ptr(builder);

    AddSource(builder_ptr.get(), "input.loom", "//model:input",
              LOOMC_LINK_PROVIDER_ROLE_INPUT, &input_provider_ordinal_);
    AddSource(builder_ptr.get(), "direct.loom", "//kernel:direct",
              LOOMC_LINK_PROVIDER_ROLE_LIBRARY, &direct_provider_ordinal_);
    AddSource(builder_ptr.get(), "transitive.loom", "//kernel:transitive",
              LOOMC_LINK_PROVIDER_ROLE_LIBRARY, &transitive_provider_ordinal_);
    AddSource(builder_ptr.get(), "unused.loom", "//kernel:unused",
              LOOMC_LINK_PROVIDER_ROLE_LIBRARY, &unused_provider_ordinal_);
    ASSERT_FALSE(HasFatalFailure());

    loomc_link_index_t* link_index = nullptr;
    loomc_result_t* index_result = nullptr;
    LOOMC_ASSERT_OK(loomc_link_index_builder_finish(
        builder_ptr.get(), &link_index, &index_result));
    ResultPtr index_result_ptr(index_result);
    ASSERT_TRUE(loomc_result_succeeded(index_result_ptr.get()));
    ASSERT_NE(link_index, nullptr);
    link_index_.reset(link_index);
  }

  iree_string_view_t FindSource(std::string_view name) {
    const iree_file_toc_t* files = loomc_link_dependency_testdata_create();
    for (iree_host_size_t i = 0; i < loomc_link_dependency_testdata_size();
         ++i) {
      if (name == files[i].name) {
        return iree_make_string_view(files[i].data, files[i].size);
      }
    }
    return iree_string_view_empty();
  }

  void AddSource(loomc_link_index_builder_t* builder, std::string_view filename,
                 std::string_view provider_name,
                 loomc_link_provider_role_t role,
                 loomc_host_size_t* out_provider_ordinal) {
    const iree_string_view_t contents = FindSource(filename);
    ASSERT_FALSE(iree_string_view_is_empty(contents));
    const loomc_source_options_t source_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        /*.structure_size=*/sizeof(source_options),
        /*.next=*/nullptr,
        /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
        /*.identifier=*/
        loomc_make_string_view(filename.data(), filename.size()),
        /*.contents=*/loomc_make_byte_span(contents.data, contents.size),
        /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
    };
    loomc_source_t* source = nullptr;
    LOOMC_ASSERT_OK(loomc_source_create(&source_options,
                                        loomc_allocator_system(), &source));
    SourcePtr source_ptr(source);

    const loomc_link_index_source_options_t index_options = {
        /*.provider_name=*/
        loomc_make_string_view(provider_name.data(), provider_name.size()),
        /*.role=*/role,
    };
    loomc_link_index_source_slot_t slot = {};
    LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(
        builder, source_ptr.get(), &index_options, &slot));
    *out_provider_ordinal = slot.ordinal;
  }

  ResultPtr Analyze(const loomc_host_size_t* direct_provider_ordinals,
                    loomc_host_size_t direct_provider_count,
                    loomc_link_dependency_artifact_flags_t artifact_flags) {
    const loomc_link_dependency_analysis_options_t options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_DEPENDENCY_ANALYSIS_OPTIONS,
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.direct_provider_ordinals=*/direct_provider_ordinals,
        /*.direct_provider_count=*/direct_provider_count,
        /*.component_name=*/loomc_make_cstring_view("//model:layers"),
        /*.artifact_flags=*/artifact_flags,
        /*.report_identifier=*/
        loomc_make_cstring_view("layers.dependencies.json"),
    };
    loomc_result_t* result = nullptr;
    loomc_status_t status = loomc_link_analyze_dependencies(
        link_index_.get(), workspace_.get(), &options, &result);
    LOOMC_EXPECT_OK(status);
    return ResultPtr(result);
  }

  ContextPtr context_;
  WorkspacePtr workspace_;
  LinkIndexPtr link_index_;
  loomc_host_size_t input_provider_ordinal_ = 0;
  loomc_host_size_t direct_provider_ordinal_ = 0;
  loomc_host_size_t transitive_provider_ordinal_ = 0;
  loomc_host_size_t unused_provider_ordinal_ = 0;
};

TEST_F(LinkDependencyTest, ReportsMissingDirectDependencyAndJsonArtifact) {
  const std::array<loomc_host_size_t, 2> direct_providers = {
      direct_provider_ordinal_, unused_provider_ordinal_};
  ResultPtr result = Analyze(direct_providers.data(), direct_providers.size(),
                             LOOMC_LINK_DEPENDENCY_ARTIFACT_FLAG_REPORT_JSON);
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result.get()));

  ASSERT_EQ(loomc_result_diagnostic_count(result.get()), 1u);
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(ToString(diagnostic->code), "LINK/DEPENDENCY/MISSING_DIRECT");
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("no direct dependency of //model:layers"));
  ASSERT_NE(diagnostic->range.source, nullptr);

  // The result retains source provenance independently of the analyzed index.
  link_index_.reset();
  EXPECT_EQ(ToString(loomc_source_identifier(diagnostic->range.source)),
            "input.loom");

  ASSERT_EQ(loomc_result_artifact_count(result.get()), 1u);
  const loomc_artifact_t* artifact = loomc_result_artifact_at(result.get(), 0);
  ASSERT_NE(artifact, nullptr);
  EXPECT_EQ(ToString(artifact->format),
            LOOMC_ARTIFACT_FORMAT_LINK_DEPENDENCY_REPORT_JSON);
  EXPECT_EQ(ToString(artifact->identifier), "layers.dependencies.json");
  const std::string json = ToString(artifact->contents);
  EXPECT_THAT(json,
              ::testing::StartsWith(
                  R"({"schema_version":1,"component":"//model:layers",)"));
  EXPECT_THAT(json, ::testing::HasSubstr(R"("succeeded":false)"));
  EXPECT_THAT(json, ::testing::HasSubstr(R"("dependency":"missing_direct")"));
  EXPECT_THAT(json, ::testing::HasSubstr(
                        R"({"provider":"//kernel:unused","usage":[]})"));
}

TEST_F(LinkDependencyTest, SucceedsWhenEveryExactDependencyIsDirect) {
  const std::array<loomc_host_size_t, 3> direct_providers = {
      direct_provider_ordinal_, transitive_provider_ordinal_,
      unused_provider_ordinal_};
  ResultPtr result = Analyze(direct_providers.data(), direct_providers.size(),
                             LOOMC_LINK_DEPENDENCY_ARTIFACT_FLAG_REPORT_JSON);
  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(loomc_result_succeeded(result.get()));
  EXPECT_EQ(loomc_result_diagnostic_count(result.get()), 0u);
  ASSERT_EQ(loomc_result_artifact_count(result.get()), 1u);
  const std::string json =
      ToString(loomc_result_artifact_at(result.get(), 0)->contents);
  EXPECT_THAT(json, ::testing::HasSubstr(R"("succeeded":true)"));
  EXPECT_THAT(json, ::testing::HasSubstr(
                        R"({"provider":"//kernel:direct","usage":["exact"]})"));
  EXPECT_THAT(json,
              ::testing::HasSubstr(
                  R"({"provider":"//kernel:transitive","usage":["exact"]})"));
}

TEST_F(LinkDependencyTest, OmitsReportWhenItIsNotRequested) {
  const std::array<loomc_host_size_t, 2> direct_providers = {
      direct_provider_ordinal_, unused_provider_ordinal_};
  const loomc_link_dependency_analysis_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_DEPENDENCY_ANALYSIS_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.direct_provider_ordinals=*/direct_providers.data(),
      /*.direct_provider_count=*/direct_providers.size(),
  };
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_link_analyze_dependencies(
      link_index_.get(), workspace_.get(), &options, &result));
  ResultPtr result_ptr(result);
  EXPECT_FALSE(loomc_result_succeeded(result_ptr.get()));
  EXPECT_EQ(loomc_result_diagnostic_count(result_ptr.get()), 1u);
  EXPECT_EQ(loomc_result_artifact_count(result_ptr.get()), 0u);
}

TEST_F(LinkDependencyTest, RejectsInvalidDirectProviderSelection) {
  const loomc_link_dependency_analysis_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_DEPENDENCY_ANALYSIS_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.direct_provider_ordinals=*/&input_provider_ordinal_,
      /*.direct_provider_count=*/1,
  };
  loomc_result_t* result = reinterpret_cast<loomc_result_t*>(0x1);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_link_analyze_dependencies(link_index_.get(), workspace_.get(),
                                      &options, &result));
  EXPECT_EQ(result, nullptr);
}

TEST_F(LinkDependencyTest, RejectsReportIdentifierWithoutReportArtifact) {
  const loomc_link_dependency_analysis_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_DEPENDENCY_ANALYSIS_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.direct_provider_ordinals=*/nullptr,
      /*.direct_provider_count=*/0,
      /*.component_name=*/loomc_string_view_empty(),
      /*.artifact_flags=*/0,
      /*.report_identifier=*/loomc_make_cstring_view("orphan.json"),
  };
  loomc_result_t* result = reinterpret_cast<loomc_result_t*>(0x1);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_link_analyze_dependencies(link_index_.get(), workspace_.get(),
                                      &options, &result));
  EXPECT_EQ(result, nullptr);
}

}  // namespace
