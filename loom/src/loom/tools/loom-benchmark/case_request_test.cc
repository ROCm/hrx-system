// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-benchmark/case_request.h"

#include <cstring>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loomc/context.h"
#include "loomc/iree.h"
#include "loomc/source.h"
#include "loomc/target/cmd/program.h"

namespace loom {
namespace {

class CaseRequestTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_context_create(
        /*options=*/nullptr, loomc_allocator_system(), &context_)));
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_workspace_create(
        /*options=*/nullptr, loomc_allocator_system(), &workspace_)));
  }

  void TearDown() override {
    loomc_workspace_release(workspace_);
    loomc_context_release(context_);
  }

  loomc_module_t* Parse(const char* source_text) {
    const loomc_source_options_t source_options = {
        .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        .structure_size = sizeof(source_options),
        .format = LOOMC_SOURCE_FORMAT_TEXT,
        .identifier = loomc_make_cstring_view("case-request-test.loom"),
        .contents = loomc_make_byte_span(source_text, std::strlen(source_text)),
        .storage = LOOMC_SOURCE_STORAGE_COPY,
    };
    loomc_source_t* source = nullptr;
    IREE_EXPECT_OK(iree_status_from_loomc(loomc_source_create(
        &source_options, loomc_allocator_system(), &source)));
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    IREE_EXPECT_OK(iree_status_from_loomc(loomc_module_deserialize_from_source(
        context_, workspace_, source, /*options=*/nullptr,
        loomc_allocator_system(), &module, &result)));
    loomc_source_release(source);
    ExpectSucceeded(result);
    loomc_result_release(result);
    return module;
  }

  void ExpectSucceeded(const loomc_result_t* result) {
    if (result != nullptr && !loomc_result_succeeded(result)) {
      for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
           ++i) {
        const loomc_diagnostic_t* diagnostic =
            loomc_result_diagnostic_at(result, i);
        ADD_FAILURE() << std::string(diagnostic->code.data,
                                     diagnostic->code.size)
                      << ": "
                      << std::string(diagnostic->message.data,
                                     diagnostic->message.size);
      }
    }
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(loomc_result_succeeded(result));
  }

  std::vector<std::string> RequestRootNames(const loomc_request_t* request) {
    const loomc_host_size_t root_count = loomc_request_root_count(request);
    std::vector<loomc_string_view_t> roots(root_count);
    loomc_module_t* module = nullptr;
    loomc_result_t* result = nullptr;
    IREE_EXPECT_OK(iree_status_from_loomc(loomc_module_deserialize_request(
        context_, workspace_, request, loomc_allocator_system(), &module,
        roots.data(), &result)));
    ExpectSucceeded(result);
    std::vector<std::string> names;
    for (loomc_string_view_t root : roots) {
      names.emplace_back(root.data, root.size);
    }
    loomc_result_release(result);
    loomc_module_release(module);
    return names;
  }

  loomc_context_t* context_ = nullptr;
  loomc_workspace_t* workspace_ = nullptr;
};

TEST_F(CaseRequestTest, PublishesWholeCommandProgramRequests) {
  static const char kSource[] = R"(
kernel.def @first() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%width: index, %tail: buffer, %other: buffer) {
  kernel.return
}

kernel.def @second() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%width: index, %other: buffer, %storage: buffer) {
  kernel.return
}

check.case @sampled {
  %width = check.param.choice values([4, 8]) name("width") : index
  %storage = check.generate.fill value(1) : tensor<8xi32>
  %tail = check.tensor.view %storage offset(16) : tensor<8xi32> -> tensor<4xi32>
  %other = check.generate.fill value(2) : tensor<4xi32>
  kernel.launch.serial {
    kernel.launch @first(%width, %tail, %other) : (index, tensor<4xi32>, tensor<4xi32>)
    kernel.launch @second(%width, %other, %storage) : (index, tensor<4xi32>, tensor<8xi32>)
  }
  check.return
}

check.benchmark<@sampled> @wide {width = 8}
)";

  loomc_module_t* module = Parse(kSource);
  ASSERT_NE(module, nullptr);
  const iree_host_size_t assignments[] = {1};
  const loom_benchmark_case_request_options_t options = {
      .workspace = workspace_,
      .module = module,
      .case_index = 0,
      .case_sample = {.assignments = assignments},
  };
  loom_benchmark_case_request_capture_t capture = {};
  loomc_result_t* result = nullptr;
  IREE_ASSERT_OK(loom_benchmark_case_request_capture(
      &options, iree_allocator_system(), &capture, &result));
  ExpectSucceeded(result);

  EXPECT_EQ(capture.case_index, 0u);
  EXPECT_FALSE(iree_string_view_is_empty(capture.root_symbol));
  ASSERT_EQ(capture.binding_count, 2u);
  EXPECT_NE(capture.binding_value_ids[0], capture.binding_value_ids[1]);
  ASSERT_EQ(capture.request_count, 2u);
  ASSERT_EQ(loomc_cmd_program_product_program_count(capture.command_product),
            1u);
  ASSERT_EQ(loomc_cmd_program_product_entry_requirement_count(
                capture.command_product),
            2u);

  loomc_cmd_program_t program = {};
  ASSERT_TRUE(loomc_cmd_program_product_program_at(capture.command_product, 0,
                                                   &program));
  EXPECT_TRUE(iree_string_view_equal(
      capture.root_symbol,
      iree_make_string_view(program.symbol.data, program.symbol.size)));
  ASSERT_EQ(program.entry_requirement_count, 2u);

  std::vector<std::string> root_names;
  for (iree_host_size_t request_index = 0;
       request_index < capture.request_count; ++request_index) {
    const loomc_request_t* request = capture.requests[request_index];
    ASSERT_EQ(loomc_request_root_count(request), 1u);
    ASSERT_EQ(loomc_request_binding_count(request), 1u);
    loomc_request_binding_t binding = {};
    ASSERT_TRUE(loomc_request_binding_at(request, 0, &binding));
    EXPECT_EQ(binding.root_ordinal, 0u);
    EXPECT_EQ(binding.requirement_ordinal, request_index);
    std::vector<std::string> request_roots = RequestRootNames(request);
    ASSERT_EQ(request_roots.size(), 1u);
    root_names.push_back(request_roots[0]);
  }
  EXPECT_THAT(root_names, ::testing::ElementsAre("first", "second"));

  loomc_result_release(result);
  loom_benchmark_case_request_capture_deinitialize(&capture);
  loomc_module_release(module);
}

}  // namespace
}  // namespace loom
