// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>

#include "iree/testing/gtest.h"
#include "loomc/link.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using BuilderPtr =
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_free>;
using IndexPtr = HandlePtr<loomc_link_index_t, loomc_link_index_release>;
using LinkerPtr = HandlePtr<loomc_linker_t, loomc_linker_release>;

std::string Text(loomc_module_t* module) {
  const loomc_module_serialize_options_t options = {
      .format = LOOMC_SOURCE_FORMAT_TEXT,
  };
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(loomc_module_serialize_to_source(
      module, &options, loomc_allocator_system(), &source));
  SourcePtr owner(source);
  const auto contents = loomc_source_contents(source);
  return std::string(reinterpret_cast<const char*>(contents.data),
                     contents.data_length);
}

class LinkOptionsTest : public ::testing::Test {
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
    loomc_linker_t* linker = nullptr;
    LOOMC_ASSERT_OK(loomc_linker_create(context, nullptr,
                                        loomc_allocator_system(), &linker));
    linker_.reset(linker);
    loomc_link_index_builder_t* builder = nullptr;
    LOOMC_ASSERT_OK(loomc_link_index_builder_allocate(
        context, nullptr, loomc_allocator_system(), &builder));
    BuilderPtr builder_owner(builder);
    constexpr char source_text[] = R"(func.def @subject(%x: i32) -> (i32) {
  func.return %x : i32
}
func.def public @unused() {
  func.return
}
check.case @subject_case {
  %input = check.literal value(1) : i32
  %actual = func.call @subject(%input) : (i32) -> (i32)
  check.expect.equal actual(%actual) expected(%input) : i32
  check.return
}
check.benchmark<@subject_case> @subject_benchmark
)";
    const loomc_source_options_t source_options = {
        .format = LOOMC_SOURCE_FORMAT_TEXT,
        .identifier = loomc_make_cstring_view("input.loom"),
        .contents = loomc_make_byte_span(source_text, sizeof(source_text) - 1),
        .storage = LOOMC_SOURCE_STORAGE_COPY,
    };
    loomc_source_t* source = nullptr;
    LOOMC_ASSERT_OK(loomc_source_create(&source_options,
                                        loomc_allocator_system(), &source));
    SourcePtr source_owner(source);
    const loomc_link_index_source_options_t provider_options = {
        .provider_name = loomc_make_cstring_view("input"),
    };
    LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(
        builder, source, &provider_options, nullptr));
    loomc_link_index_t* index = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_ASSERT_OK(loomc_link_index_builder_finish(builder, &index, &result));
    ResultPtr result_owner(result);
    ASSERT_TRUE(loomc_result_succeeded(result));
    index_.reset(index);
  }

  ModulePtr Link(const loomc_link_options_t* options, ResultPtr* result_owner) {
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_OK(loomc_link_module(linker_.get(), workspace_.get(), options,
                                      &module, &result));
    result_owner->reset(result);
    EXPECT_TRUE(loomc_result_succeeded(result));
    return ModulePtr(module);
  }

  // Shared dialect metadata retained by the linker and index.
  ContextPtr context_;
  // Invocation-local scratch and returned module storage.
  WorkspacePtr workspace_;
  // Reusable linker configuration.
  LinkerPtr linker_;
  // Frozen provider metadata and sources.
  IndexPtr index_;
};

TEST_F(LinkOptionsTest, SelectsInputTestsAndStripsTheirWrappers) {
  loomc_link_options_t options = {
      .structure_size = sizeof(options),
      .link_index = index_.get(),
      .mode = LOOMC_LINK_MODE_LINK,
      .flags = LOOMC_LINK_FLAG_INCLUDE_INPUT_TESTS,
  };
  ResultPtr result;
  ModulePtr module = Link(&options, &result);
  const std::string text = Text(module.get());
  EXPECT_NE(text.find("@subject_case"), std::string::npos);
  EXPECT_NE(text.find("@subject_benchmark"), std::string::npos);
  EXPECT_EQ(text.find("@unused"), std::string::npos);
  EXPECT_EQ(loomc_result_artifact_count(result.get()), 0u);

  options.flags |= LOOMC_LINK_FLAG_STRIP_TEST_SYMBOLS;
  module = Link(&options, &result);
  EXPECT_EQ(Text(module.get()),
            "func.def retain @subject(%x: i32) -> (i32) {\n"
            "  func.return %x : i32\n}\n");
}

TEST_F(LinkOptionsTest, PlanReportOwnsItsBytesAfterLinkInputsAreReleased) {
  const loomc_string_view_t root = loomc_make_cstring_view("@subject");
  const loomc_link_options_t options = {
      .structure_size = sizeof(options),
      .link_index = index_.get(),
      .mode = LOOMC_LINK_MODE_LINK,
      .root_symbols = &root,
      .root_symbol_count = 1,
      .artifact_flags = LOOMC_LINK_ARTIFACT_FLAG_PLAN_TEXT,
  };
  ResultPtr result;
  ModulePtr module = Link(&options, &result);
  ASSERT_EQ(loomc_result_artifact_count(result.get()), 1u);
  module.reset();
  index_.reset();
  linker_.reset();
  workspace_.reset();
  context_.reset();
  const loomc_artifact_t* artifact = loomc_result_artifact_at(result.get(), 0);
  EXPECT_EQ(artifact->kind, LOOMC_ARTIFACT_KIND_REPORT);
  EXPECT_EQ(std::string(artifact->format.data, artifact->format.size),
            LOOMC_ARTIFACT_FORMAT_LINK_PLAN_TEXT);
  EXPECT_EQ(std::string(artifact->identifier.data, artifact->identifier.size),
            "link-plan.txt");
  loomc_source_t* source = nullptr;
  LOOMC_ASSERT_OK(loomc_artifact_create_source(
      artifact, LOOMC_SOURCE_FORMAT_TEXT, loomc_allocator_system(), &source));
  SourcePtr source_owner(source);
  const auto contents = loomc_source_contents(source);
  EXPECT_EQ(
      std::string(reinterpret_cast<const char*>(contents.data),
                  contents.data_length),
      "plan mode=link symbols=1\n"
      "[0] root @subject provider=input module= cause=- flags=definition\n");
}

TEST_F(LinkOptionsTest, OlderDescriptorsNeverReadAppendedStorage) {
  constexpr size_t prefix_size = offsetof(loomc_link_options_t, target_profile);
  const size_t declared_sizes[] = {
      0,
      prefix_size,
      prefix_size + sizeof(loomc_link_options_t::target_profile) - 1,
      offsetof(loomc_link_options_t, artifact_flags),
      offsetof(loomc_link_options_t, artifact_flags) +
          sizeof(loomc_link_artifact_flags_t) - 1,
      offsetof(loomc_link_options_t, test_provider_ordinals),
      offsetof(loomc_link_options_t, test_provider_count),
      offsetof(loomc_link_options_t, test_provider_count) +
          sizeof(loomc_host_size_t) - 1};
  for (const size_t declared_size : declared_sizes) {
    SCOPED_TRACE(declared_size);
    loomc_link_options_t options = {
        .structure_size = declared_size,
        .link_index = index_.get(),
        .artifact_flags = declared_size < offsetof(loomc_link_options_t,
                                                   test_provider_ordinals)
                              ? UINT32_MAX
                              : 0,
        .test_provider_ordinals =
            reinterpret_cast<const loomc_host_size_t*>(0x1),
        .test_provider_count = 1,
    };
    const size_t allocated_size = declared_size ? declared_size : prefix_size;
    auto storage = std::make_unique<uint8_t[]>(allocated_size);
    memcpy(storage.get(), &options, allocated_size);
    ResultPtr result;
    ModulePtr module = Link(
        reinterpret_cast<const loomc_link_options_t*>(storage.get()), &result);
    EXPECT_NE(module, nullptr);
    EXPECT_EQ(loomc_result_artifact_count(result.get()), 0u);
  }
}

TEST_F(LinkOptionsTest, RejectsInvalidPoliciesAtInvocationBoundary) {
  loomc_link_options_t options = {
      .structure_size = sizeof(options),
      .link_index = index_.get(),
      .artifact_flags = UINT32_MAX,
  };
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_link_module(linker_.get(), workspace_.get(),
                                           &options, &module, &result));
  EXPECT_EQ(module, nullptr);
  EXPECT_EQ(result, nullptr);
  options.artifact_flags = 0;
  options.flags = LOOMC_LINK_FLAG_INCLUDE_INPUT_TESTS;
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_link_module(linker_.get(), workspace_.get(),
                                           &options, &module, &result));
  EXPECT_EQ(module, nullptr);
  EXPECT_EQ(result, nullptr);
  options.flags = 0;
  options.mode = LOOMC_LINK_MODE_LINK;
  options.test_provider_count = 1;
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_link_module(linker_.get(), workspace_.get(),
                                           &options, &module, &result));
  const loomc_host_size_t invalid_provider =
      loomc_link_index_provider_count(index_.get());
  options.test_provider_ordinals = &invalid_provider;
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_OUT_OF_RANGE,
                         loomc_link_module(linker_.get(), workspace_.get(),
                                           &options, &module, &result));
}

}  // namespace
