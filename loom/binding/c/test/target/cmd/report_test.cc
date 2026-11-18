// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "loom/binding/c/test/target/cmd/report_testdata.h"
#include "loomc/compile_report.h"
#include "loomc/target/cmd.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;
using ProductPtr = HandlePtr<loomc_product_t, loomc_product_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;

enum class Construction { kIndex, kModule, kRequest };

std::string Contents(const loomc_byte_sequence_t* sequence) {
  std::string text;
  LOOMC_EXPECT_OK(loomc_byte_sequence_enumerate(
      sequence, {
                    .fn =
                        [](void* user_data, loomc_byte_span_t segment) {
                          static_cast<std::string*>(user_data)->append(
                              reinterpret_cast<const char*>(segment.data),
                              segment.data_length);
                          return loomc_ok_status();
                        },
                    .user_data = &text,
                }));
  return text;
}

class CommandReportTest : public ::testing::TestWithParam<Construction> {
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
    const auto& file = loomc_cmd_report_testdata_create()[0];
    const loomc_source_options_t source_options = {
        .format = LOOMC_SOURCE_FORMAT_TEXT,
        .identifier = loomc_make_cstring_view(file.name),
        .contents = loomc_make_byte_span(file.data, file.size),
        .storage = LOOMC_SOURCE_STORAGE_COPY,
    };
    loomc_source_t* source = nullptr;
    LOOMC_ASSERT_OK(loomc_source_create(&source_options,
                                        loomc_allocator_system(), &source));
    HandlePtr<loomc_source_t, loomc_source_release> text(source);
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_ASSERT_OK(loomc_module_deserialize_from_source(
        context, workspace, source, nullptr, loomc_allocator_system(), &module,
        &result));
    ResultPtr parsed(result);
    ASSERT_TRUE(loomc_result_succeeded(result));
    module_.reset(module);
    const loomc_module_serialize_options_t serialize_options = {
        .format = LOOMC_SOURCE_FORMAT_BYTECODE,
    };
    LOOMC_ASSERT_OK(loomc_module_serialize_to_source(
        module, &serialize_options, loomc_allocator_system(), &source));
    source_.reset(source);
    loomc_link_index_builder_t* builder = nullptr;
    LOOMC_ASSERT_OK(loomc_link_index_builder_allocate(
        context, nullptr, loomc_allocator_system(), &builder));
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_free>
        builder_owner(builder);
    const loomc_link_index_source_options_t input_options = {
        .provider_name = loomc_make_cstring_view("report"),
        .role = LOOMC_LINK_PROVIDER_ROLE_INPUT,
    };
    LOOMC_ASSERT_OK(loomc_link_index_builder_add_source(
        builder, source, &input_options, nullptr));
    loomc_link_index_t* index = nullptr;
    LOOMC_ASSERT_OK(loomc_link_index_builder_finish(builder, &index, &result));
    ResultPtr indexed(result);
    ASSERT_TRUE(loomc_result_succeeded(result));
    index_.reset(index);
    ASSERT_TRUE(loomc_link_index_lookup_global(
        index, loomc_make_cstring_view("dispatch"), &root_));
    loomc_cmd_compiler_t* compiler = nullptr;
    LOOMC_ASSERT_OK(loomc_cmd_compiler_create(
        index, nullptr, loomc_allocator_system(), &compiler));
    compiler_.reset(compiler);
    const loomc_request_root_t request_root = {
        .module_ordinal = static_cast<uint32_t>(root_.provider_module_ordinal),
        .symbol_ordinal = static_cast<uint32_t>(root_.module_symbol_ordinal),
    };
    loomc_request_t* request = nullptr;
    LOOMC_ASSERT_OK(loomc_request_create(loomc_cmd_program_product_descriptor(),
                                         source, &request_root, 1, nullptr, 0,
                                         loomc_allocator_system(), &request));
    request_.reset(request);
  }

  loomc_status_t Build(const void* next, loomc_product_t** product,
                       loomc_result_t** result,
                       loomc_emit_artifact_flags_t flags = 0) {
    switch (GetParam()) {
      case Construction::kIndex: {
        const loomc_cmd_program_product_options_t options = {
            .next = next,
            .compiler = compiler_.get(),
            .root_symbol_ordinals = &root_.ordinal,
            .root_symbol_count = 1,
            .artifact_flags = flags,
        };
        return loomc_cmd_program_product_build(workspace_.get(), &options,
                                               loomc_allocator_system(),
                                               product, result);
      }
      case Construction::kModule: {
        const loomc_string_view_t root = loomc_make_cstring_view("dispatch");
        const loomc_cmd_program_module_options_t options = {
            .next = next,
            .provider_name = loomc_make_cstring_view("report"),
            .root_symbols = &root,
            .root_symbol_count = 1,
            .artifact_flags = flags,
        };
        return loomc_cmd_program_product_build_module(
            workspace_.get(), module_.get(), &options, loomc_allocator_system(),
            product, result);
      }
      case Construction::kRequest: {
        const loomc_cmd_program_request_options_t options = {
            .next = next,
            .artifact_flags = flags,
        };
        return loomc_cmd_program_product_build_request(
            context_.get(), workspace_.get(), request_.get(), &options,
            loomc_allocator_system(), product, result);
      }
    }
    std::abort();
  }

  void ReleaseCompilerOwners() {
    request_.reset();
    compiler_.reset();
    index_.reset();
    module_.reset();
    source_.reset();
    workspace_.reset();
    context_.reset();
  }

  // Original compiler owners, all released before reading the final artifacts.
  HandlePtr<loomc_context_t, loomc_context_release> context_;
  // Recycled invocation scratch, independent from the report result.
  HandlePtr<loomc_workspace_t, loomc_workspace_release> workspace_;
  // Materialized input for the direct module entry point.
  HandlePtr<loomc_module_t, loomc_module_release> module_;
  // Bytecode backing the indexed and request entry points.
  HandlePtr<loomc_source_t, loomc_source_release> source_;
  // Frozen link index retaining source identity.
  HandlePtr<loomc_link_index_t, loomc_link_index_release> index_;
  // Reusable command compiler for indexed construction.
  HandlePtr<loomc_cmd_compiler_t, loomc_cmd_compiler_release> compiler_;
  // Independently owned source request for request construction.
  HandlePtr<loomc_request_t, loomc_request_release> request_;
  // Exact source root selected by indexed construction.
  loomc_link_index_symbol_t root_ = {};
};

TEST_P(CommandReportTest, OwnsReportsWithoutChangingExecutableBytes) {
  loomc_product_t* raw_product = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(Build(nullptr, &raw_product, &raw_result));
  ProductPtr baseline(raw_product);
  ResultPtr baseline_result(raw_result);
  ASSERT_TRUE(loomc_result_succeeded(raw_result));
  ASSERT_EQ(loomc_result_artifact_count(raw_result), 0u);
  ASSERT_EQ(loomc_product_artifact_count(raw_product), 1u);
  const std::string executable =
      Contents(loomc_product_artifact_at(raw_product, 0)->contents);

  std::vector<ResultPtr> results;
  std::vector<std::string> expected;
  for (bool text : {false, true}) {
    for (auto mode :
         {LOOMC_COMPILE_REPORT_MODE_NONE, LOOMC_COMPILE_REPORT_MODE_SUMMARY,
          LOOMC_COMPILE_REPORT_MODE_DETAILS}) {
      std::string identifier = "custom.command.report";
      const loomc_compile_report_options_t report = {
          .type = LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
          .mode = mode,
          .format = text ? LOOMC_COMPILE_REPORT_FORMAT_TEXT
                         : LOOMC_COMPILE_REPORT_FORMAT_JSON,
          .identifier = mode == LOOMC_COMPILE_REPORT_MODE_DETAILS
                            ? loomc_make_cstring_view(identifier.c_str())
                            : loomc_string_view_empty(),
      };
      // Listing and report requests must compose, not select one another.
      const auto flags = mode == LOOMC_COMPILE_REPORT_MODE_DETAILS
                             ? LOOMC_EMIT_ARTIFACT_FLAG_TARGET_LISTING
                             : 0;
      LOOMC_ASSERT_OK(Build(&report, &raw_product, &raw_result, flags));
      ProductPtr product(raw_product);
      ResultPtr result(raw_result);
      ASSERT_TRUE(loomc_result_succeeded(raw_result));
      ASSERT_EQ(loomc_product_artifact_count(raw_product), 1u);
      EXPECT_EQ(Contents(loomc_product_artifact_at(raw_product, 0)->contents),
                executable);
      if (mode == LOOMC_COMPILE_REPORT_MODE_NONE) {
        EXPECT_EQ(loomc_result_artifact_count(raw_result), 0u);
        continue;
      }
      const size_t report_index = flags ? 1 : 0;
      ASSERT_EQ(loomc_result_artifact_count(raw_result), report_index + 1);
      const auto* artifact = loomc_result_artifact_at(raw_result, report_index);
      EXPECT_EQ(artifact->kind, LOOMC_ARTIFACT_KIND_REPORT);
      EXPECT_TRUE(loomc_string_view_equal(
          artifact->format,
          loomc_make_cstring_view(
              text ? LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_TEXT
                   : LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON)));
      const std::string bytes = Contents(artifact->contents);
      EXPECT_NE(bytes.find("dispatch"), std::string::npos);
      EXPECT_EQ(bytes.find("block.0.weights") != std::string::npos,
                mode == LOOMC_COMPILE_REPORT_MODE_DETAILS);
      identifier.assign(identifier.size(), '!');
      EXPECT_TRUE(loomc_string_view_equal(
          artifact->identifier,
          loomc_make_cstring_view(
              mode == LOOMC_COMPILE_REPORT_MODE_DETAILS
                  ? "custom.command.report"
                  : (text ? "command-program.compile-report.txt"
                          : "command-program.compile-report.json"))));
      expected.push_back(bytes);
      results.push_back(std::move(result));
    }
  }
  baseline.reset();
  baseline_result.reset();
  ReleaseCompilerOwners();
  for (size_t i = 0; i < results.size(); ++i) {
    const auto* artifact = loomc_result_artifact_at(
        results[i].get(), loomc_result_artifact_count(results[i].get()) - 1);
    EXPECT_EQ(Contents(artifact->contents), expected[i]);
  }
}

TEST_P(CommandReportTest, ValidatesCommonOptionsBeforeCompilation) {
  for (int invalid = 0; invalid < 5; ++invalid) {
    SCOPED_TRACE(invalid);
    loomc_compile_report_options_t report = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
        .mode = LOOMC_COMPILE_REPORT_MODE_DETAILS,
    };
    const auto duplicate = report;
    switch (invalid) {
      case 0:
        report.structure_size = 1;
        break;
      case 1:
        report.mode = static_cast<loomc_compile_report_mode_t>(99);
        break;
      case 2:
        report.format = static_cast<loomc_compile_report_format_t>(99);
        break;
      case 3:
        report.next = &duplicate;
        break;
      case 4:
        report.identifier = {nullptr, 1};
        break;
    }
    loomc_product_t* product = nullptr;
    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                           Build(&report, &product, &result));
    EXPECT_EQ(product, nullptr);
    EXPECT_EQ(result, nullptr);
    loomc_product_release(product);
    loomc_result_release(result);
  }
}

INSTANTIATE_TEST_SUITE_P(ConstructionPaths, CommandReportTest,
                         ::testing::Values(Construction::kIndex,
                                           Construction::kModule,
                                           Construction::kRequest));

}  // namespace
