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

TEST_F(KernelLaunchConfigTest, DirectlyEvaluatesTargetIndependentConstants) {
  ModulePtr module = Parse(R"(
kernel.def @entry() {
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %three = index.constant 3 : index
  %four = index.constant 4 : index
  %five = index.constant 5 : index
  %six = index.constant 6 : index
  kernel.launch.config workgroups(%two, %three, %four) workgroup_size(%five, %six, %one) : index
} launch() {
  kernel.return
}
)");

  const loom_kernel_launch_config_options_t options = {
      /*.function_symbol=*/IREE_SV("@entry"),
      /*.workload_arguments=*/nullptr,
      /*.workload_argument_count=*/0,
      /*.required_fields=*/
      LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
          LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE,
      /*.diagnostic_emitter=*/{},
  };
  loom_kernel_launch_config_t config = {};
  bool evaluated = false;
  IREE_ASSERT_OK(loom_kernel_launch_config_try_evaluate_direct(
      module.get(), &block_pool_, &options, &config, &evaluated));

  EXPECT_TRUE(evaluated);
  EXPECT_EQ(config.failure, LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_NONE);
  EXPECT_TRUE(config.fields &
              LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT);
  EXPECT_EQ(config.workgroup_count.x, 2u);
  EXPECT_EQ(config.workgroup_count.y, 3u);
  EXPECT_EQ(config.workgroup_count.z, 4u);
  EXPECT_TRUE(config.fields &
              LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE);
  EXPECT_EQ(config.workgroup_size.x, 5u);
  EXPECT_EQ(config.workgroup_size.y, 6u);
  EXPECT_EQ(config.workgroup_size.z, 1u);
}

TEST_F(KernelLaunchConfigTest, DirectPathSkipsTargetBoundKernels) {
  ModulePtr module = Parse(R"(
target.generic<reference> @gpu {
  subgroup_size = 32
}

kernel.def target(@gpu) @entry() {
  %one = index.constant 1 : index
  %threads = index.constant 64 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%threads, %one, %one) : index
} launch() {
  kernel.return
}
)");

  const loom_kernel_launch_config_options_t options = {
      /*.function_symbol=*/IREE_SV("entry"),
      /*.workload_arguments=*/nullptr,
      /*.workload_argument_count=*/0,
      /*.required_fields=*/
      LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
          LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE,
      /*.diagnostic_emitter=*/{},
  };
  loom_kernel_launch_config_t config = {};
  bool evaluated = true;
  IREE_ASSERT_OK(loom_kernel_launch_config_try_evaluate_direct(
      module.get(), &block_pool_, &options, &config, &evaluated));

  EXPECT_FALSE(evaluated);
  EXPECT_EQ(config.fields, 0u);
  EXPECT_EQ(config.failure, LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_NONE);
}

TEST_F(KernelLaunchConfigTest, EvaluatesTargetAndWorkloadBackedFields) {
  ModulePtr module = Parse(R"(
target.generic<reference> @gpu {
  subgroup_size = 32
}

kernel.def target(@gpu) @entry(%rows: index) {
  %one = index.constant 1 : index
  %sixty_three = index.constant 63 : index
  %sixty_four = index.constant 64 : index
  %rounded_rows = index.add %rows, %sixty_three : index
  %row_groups = index.div %rounded_rows, %sixty_four : index
  kernel.launch.config workgroups(%row_groups, %one, %one) workgroup_size(%sixty_four, %one, %one) : index
} launch() {
  kernel.return
}
)");

  const int64_t workload_arguments[] = {129};
  const loom_kernel_launch_config_options_t options = {
      /*.function_symbol=*/IREE_SV("entry"),
      /*.workload_arguments=*/workload_arguments,
      /*.workload_argument_count=*/IREE_ARRAYSIZE(workload_arguments),
      /*.required_fields=*/
      LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT |
          LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE |
          LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE,
      /*.diagnostic_emitter=*/{},
  };
  loom_kernel_launch_config_t config = {};
  IREE_ASSERT_OK(loom_kernel_launch_config_evaluate(module.get(), &block_pool_,
                                                    &options, &config));

  EXPECT_EQ(config.failure, LOOM_KERNEL_LAUNCH_CONFIG_FAILURE_NONE);
  EXPECT_TRUE(config.fields &
              LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT);
  EXPECT_EQ(config.workgroup_count.x, 3u);
  EXPECT_EQ(config.workgroup_count.y, 1u);
  EXPECT_EQ(config.workgroup_count.z, 1u);
  EXPECT_TRUE(config.fields &
              LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_SIZE);
  EXPECT_EQ(config.workgroup_size.x, 64u);
  EXPECT_EQ(config.workgroup_size.y, 1u);
  EXPECT_EQ(config.workgroup_size.z, 1u);
  EXPECT_TRUE(config.fields &
              LOOM_KERNEL_LAUNCH_CONFIG_FIELD_FLAG_SUBGROUP_SIZE);
  EXPECT_EQ(config.subgroup_size, 32u);
}

}  // namespace
}  // namespace loom
