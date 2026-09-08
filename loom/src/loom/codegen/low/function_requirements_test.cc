// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/function_requirements.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/test/low_registry.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

class LowFunctionRequirementsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_low_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_LOW,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&registry_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  testing::ModulePtr Parse(const char* source) {
    loom_text_parse_options_t options = {};
    options.diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr};
    loom_low_descriptor_text_asm_environment_initialize(
        &registry_.registry, &options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("requirements.loom"), &context_,
                                  &block_pool_, &options, &module));
    return testing::ModulePtr(module);
  }

  loom_low_function_requirements_t Analyze(loom_module_t* module) {
    loom_op_t* function = loom_block_op(loom_module_block(module), 0);
    loom_low_function_requirements_t requirements = {};
    IREE_CHECK_OK(loom_low_function_requirements_build(
        module, loom_low_func_def_body(function), &arena_, &requirements));
    return requirements;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t registry_;
};

TEST_F(LowFunctionRequirementsTest, ImportsPreserveBodyOrder) {
  auto module = Parse(R"(
low.func.def target<test.low.core> @imports() asm {
  %input = resource<native_pointer> {index = 7, source_type = buffer, extent = 64} : reg<test.ptr>
  %output = resource<native_pointer> {index = 2, source_type = buffer} : reg<test.ptr>
  return
}
)");
  const auto requirements = Analyze(module.get());
  ASSERT_EQ(requirements.resource_count, 2u);
  EXPECT_EQ(loom_low_resource_index(requirements.resources[0]), 7);
  EXPECT_EQ(loom_low_resource_extent(requirements.resources[0]), 64);
  EXPECT_EQ(loom_low_resource_index(requirements.resources[1]), 2);
}

TEST_F(LowFunctionRequirementsTest, CollectsStorageAcrossBlocks) {
  auto module = Parse(R"(
low.func.def target<test.low.core> @storage(%condition: reg<test.i32>) asm {
  %stack = storage {byte_length = 8, byte_alignment = 4} : low.storage<stack>
  %private = storage {byte_length = 12, byte_alignment = 8} : low.storage<private>
  low.cond_br %condition, ^left, ^right : reg<test.i32>
^left:
  %later = storage {byte_length = 16, byte_alignment = 16} : low.storage<stack>
  return
^right:
  return
}
)");
  const auto requirements = Analyze(module.get());
  EXPECT_EQ(requirements.node_count, 6u);
  EXPECT_EQ(requirements.return_count, 2u);
  const auto& layout = requirements.storage_layout;
  EXPECT_EQ(layout.space_sizes.stack_bytes, 32u);
  EXPECT_EQ(layout.space_sizes.private_bytes, 12u);
  ASSERT_EQ(layout.record_count, 3u);
  EXPECT_EQ(layout.records[2].reservation.byte_offset, 16u);
}

TEST_F(LowFunctionRequirementsTest, EmptyInterface) {
  auto module = Parse(R"(
low.func.def target<test.low.core> @empty() asm {
  return
}
)");
  const auto requirements = Analyze(module.get());
  EXPECT_EQ(requirements.resource_count, 0u);
  EXPECT_EQ(requirements.resources, nullptr);
  EXPECT_EQ(requirements.storage_layout.record_count, 0u);
  EXPECT_EQ(requirements.node_count, 1u);
  EXPECT_EQ(requirements.return_count, 1u);
}

TEST_F(LowFunctionRequirementsTest, ResidentControlFlowHasNoReturn) {
  auto module = Parse(R"(
low.func.def target<test.low.core> @resident() asm {
  low.br ^wait
^wait:
  low.br ^wait
}
)");
  const auto requirements = Analyze(module.get());
  EXPECT_EQ(requirements.node_count, 2u);
  EXPECT_EQ(requirements.return_count, 0u);
}

}  // namespace
}  // namespace loom
