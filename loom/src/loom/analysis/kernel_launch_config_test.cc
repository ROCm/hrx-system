// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/kernel_launch_config.h"

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/testing/context.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class KernelLaunchConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_testing_context_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr Parse(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   IREE_SV("kernel_launch_config_test.loom"),
                                   &context_, &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return ModulePtr(module);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(KernelLaunchConfigTest, EvaluatesCompiledLaunchFunction) {
  ModulePtr module = Parse(R"(
func.def public pure @entry(%row_count: i32) -> (index, index, index, index, index, index, index, index, index, index, index) where [range(%row_count, 1, 64)] {
  %row_count_index = index.cast %row_count : i32 to index
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %thirty_two = index.constant 32 : index
  %sixty_four = index.constant 64 : index
  %storage = index.constant 1024 : index
  func.return %row_count_index, %two, %one, %sixty_four, %two, %one, %one, %two, %one, %thirty_two, %storage : index, index, index, index, index, index, index, index, index, index, index
}
)");

  const loom_string_id_t name_id =
      loom_module_lookup_string(module.get(), IREE_SV("entry"));
  const loom_symbol_id_t symbol_id =
      loom_module_find_symbol(module.get(), name_id);
  ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
  const loom_kernel_launch_config_function_t function =
      loom_kernel_launch_config_function_bind(
          module.get(),
          loom_func_like_cast(module.get(),
                              module->symbols.entries[symbol_id].defining_op));
  loom_pass_value_fact_owner_t fact_owner = {};
  loom_pass_value_fact_owner_initialize(&block_pool_, &fact_owner);

  const uint64_t arguments[] = {17};
  loom_kernel_launch_config_t config = {};
  IREE_ASSERT_OK(loom_kernel_launch_config_function_evaluate(
      module.get(), &function, arguments, IREE_ARRAYSIZE(arguments),
      &fact_owner, &config));
  EXPECT_EQ(config.fields,
            LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
                LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE |
                LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_CLUSTER_SIZE |
                LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE |
                LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_STORAGE_BYTES);
  EXPECT_EQ(config.workgroup_count.x, 17u);
  EXPECT_EQ(config.workgroup_count.y, 2u);
  EXPECT_EQ(config.workgroup_count.z, 1u);
  EXPECT_EQ(config.workgroup_size.x, 64u);
  EXPECT_EQ(config.workgroup_size.y, 2u);
  EXPECT_EQ(config.workgroup_size.z, 1u);
  EXPECT_EQ(config.workgroup_cluster_size.x, 1u);
  EXPECT_EQ(config.workgroup_cluster_size.y, 2u);
  EXPECT_EQ(config.workgroup_cluster_size.z, 1u);
  EXPECT_EQ(config.subgroup_size, 32u);
  EXPECT_EQ(config.workgroup_storage_bytes, 1024u);

  const uint64_t invalid_arguments[] = {0};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_kernel_launch_config_function_evaluate(
          module.get(), &function, invalid_arguments,
          IREE_ARRAYSIZE(invalid_arguments), &fact_owner, &config));
  loom_pass_value_fact_owner_deinitialize(&fact_owner);
}

}  // namespace
}  // namespace loom
