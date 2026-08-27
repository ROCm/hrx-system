// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <string>
#include <string_view>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "loom/src/loom/link/testdata/provider_roots_testdata.h"
#include "loomc/link.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using BuilderPtr =
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using LinkerPtr = HandlePtr<loomc_linker_t, loomc_linker_release>;
using LinkIndexPtr = HandlePtr<loomc_link_index_t, loomc_link_index_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

class LinkProviderRootsTest : public ::testing::Test {
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
    LOOMC_ASSERT_OK(loomc_linker_create(context_.get(), nullptr,
                                        loomc_allocator_system(), &linker));
    linker_.reset(linker);

    loomc_link_index_builder_t* builder = nullptr;
    LOOMC_ASSERT_OK(loomc_link_index_builder_create(
        context_.get(), nullptr, loomc_allocator_system(), &builder));
    BuilderPtr builder_ptr(builder);

    AddSource(builder_ptr.get(), "direct.loom", "//app:direct",
              &direct_provider_ordinal_);
    AddSource(builder_ptr.get(), "transitive.loom", "//motif:transitive",
              &transitive_provider_ordinal_);
    AddSource(builder_ptr.get(), "unused.loom", "//motif:unused",
              &unused_provider_ordinal_);
    ASSERT_FALSE(HasFatalFailure());

    loomc_link_index_t* link_index = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_ASSERT_OK(loomc_link_index_builder_finish(builder_ptr.get(),
                                                    &link_index, &result));
    ResultPtr result_ptr(result);
    ASSERT_TRUE(loomc_result_succeeded(result_ptr.get()));
    ASSERT_NE(link_index, nullptr);
    link_index_.reset(link_index);
  }

  iree_string_view_t FindSource(std::string_view name) {
    const iree_file_toc_t* files = loom_link_provider_roots_testdata_create();
    for (iree_host_size_t i = 0; i < loom_link_provider_roots_testdata_size();
         ++i) {
      if (name == files[i].name) {
        return iree_make_string_view(files[i].data, files[i].size);
      }
    }
    return iree_string_view_empty();
  }

  void AddSource(loomc_link_index_builder_t* builder, std::string_view filename,
                 std::string_view provider_name,
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
        /*.role=*/LOOMC_LINK_PROVIDER_ROLE_LIBRARY,
    };
    loomc_link_index_source_slot_t slot = {};
    LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(
        builder, source_ptr.get(), &index_options, &slot));
    *out_provider_ordinal = slot.ordinal;
  }

  ModulePtr Link(const loomc_host_size_t* root_provider_ordinals,
                 loomc_host_size_t root_provider_count, ResultPtr* out_result) {
    loomc_link_options_t options = {};
    options.type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
    options.structure_size = sizeof(options);
    options.link_index = link_index_.get();
    options.mode = LOOMC_LINK_MODE_LINK;
    options.root_provider_ordinals = root_provider_ordinals;
    options.root_provider_count = root_provider_count;

    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    loomc_status_t status = loomc_link_module(linker_.get(), workspace_.get(),
                                              &options, &module, &result);
    LOOMC_EXPECT_OK(status);
    out_result->reset(result);
    return ModulePtr(module);
  }

  std::string SerializeText(const loomc_module_t* module) {
    loomc_module_serialize_options_t options = {};
    options.type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS;
    options.structure_size = sizeof(options);
    options.format = LOOMC_SOURCE_FORMAT_TEXT;
    loomc_source_t* source = nullptr;
    LOOMC_EXPECT_OK(loomc_module_serialize_to_source(
        module, &options, loomc_allocator_system(), &source));
    SourcePtr source_ptr(source);
    const loomc_byte_span_t contents = loomc_source_contents(source_ptr.get());
    return std::string(reinterpret_cast<const char*>(contents.data),
                       contents.data_length);
  }

  ContextPtr context_;
  WorkspacePtr workspace_;
  LinkerPtr linker_;
  LinkIndexPtr link_index_;
  loomc_host_size_t direct_provider_ordinal_ = 0;
  loomc_host_size_t transitive_provider_ordinal_ = 0;
  loomc_host_size_t unused_provider_ordinal_ = 0;
};

TEST_F(LinkProviderRootsTest, SelectsDirectExportsAndReachableClosure) {
  ResultPtr result;
  ModulePtr module = Link(&direct_provider_ordinal_, 1, &result);
  ASSERT_NE(result, nullptr);
  ASSERT_TRUE(loomc_result_succeeded(result.get()));
  ASSERT_NE(module, nullptr);

  const std::string text = SerializeText(module.get());
  EXPECT_THAT(text, ::testing::HasSubstr("func.def public retain @entry"));
  EXPECT_THAT(text,
              ::testing::HasSubstr("func.def public retain pure @secondary"));
  EXPECT_THAT(text, ::testing::HasSubstr("func.def @direct_helper"));
  EXPECT_THAT(text, ::testing::HasSubstr("func.def pure @transitive"));
  EXPECT_THAT(text, ::testing::Not(::testing::HasSubstr("transitive_unused")));
  EXPECT_THAT(text,
              ::testing::Not(::testing::HasSubstr("unreachable_private")));
  EXPECT_THAT(text, ::testing::Not(::testing::HasSubstr("unused_export")));
}

TEST_F(LinkProviderRootsTest, ReportsOutOfRangeProviderOrdinal) {
  const loomc_host_size_t invalid_provider_ordinal =
      loomc_link_index_provider_count(link_index_.get());
  ResultPtr result;
  ModulePtr module = Link(&invalid_provider_ordinal, 1, &result);
  EXPECT_EQ(module, nullptr);
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result.get()));
  ASSERT_EQ(loomc_result_diagnostic_count(result.get()), 1u);
  EXPECT_THAT(
      std::string(loomc_result_diagnostic_at(result.get(), 0)->message.data,
                  loomc_result_diagnostic_at(result.get(), 0)->message.size),
      ::testing::HasSubstr("outside the 3-provider index"));
}

TEST_F(LinkProviderRootsTest, RejectsMissingProviderOrdinalArray) {
  loomc_link_options_t options = {};
  options.type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
  options.structure_size = sizeof(options);
  options.link_index = link_index_.get();
  options.mode = LOOMC_LINK_MODE_LINK;
  options.root_provider_count = 1;

  loomc_module_t* module = reinterpret_cast<loomc_module_t*>(0x1);
  loomc_result_t* result = reinterpret_cast<loomc_result_t*>(0x1);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_link_module(linker_.get(), workspace_.get(),
                                           &options, &module, &result));
  EXPECT_EQ(module, nullptr);
  EXPECT_EQ(result, nullptr);
}

}  // namespace
