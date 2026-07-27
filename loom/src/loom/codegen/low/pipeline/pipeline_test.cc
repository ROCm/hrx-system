// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/pipeline/pipeline.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

static iree_status_t BuildLowPreparationPipeline(loom_builder_t* builder,
                                                 void* user_data) {
  return loom_low_pipeline_build_packetization_preparation(builder);
}

class LowPipelineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t AllocateModule(iree_string_view_t name, ModulePtr* out_module) {
    *out_module = nullptr;
    loom_module_t* module = nullptr;
    IREE_RETURN_IF_ERROR(loom_module_allocate(&context_, name, &block_pool_,
                                              nullptr, iree_allocator_system(),
                                              &module));
    *out_module = ModulePtr(module);
    return iree_ok_status();
  }

  iree_string_view_t RunKey(loom_module_t* module, const loom_op_t* op) {
    return module->strings.entries[loom_pass_run_key(op)];
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(LowPipelineTest, BuildsPacketizationPreparationFragment) {
  ModulePtr module;
  IREE_ASSERT_OK(AllocateModule(IREE_SV("pipeline"), &module));

  loom_op_t* pipeline_op = nullptr;
  IREE_ASSERT_OK(loom_pass_ir_build_pipeline(
      module.get(), IREE_SV("prepare_low"), LOOM_PASS_ANCHOR_FUNC,
      BuildLowPreparationPipeline, nullptr, &pipeline_op));

  loom_block_t* pipeline_body =
      loom_region_entry_block(loom_pass_pipeline_body(pipeline_op));
  ASSERT_NE(pipeline_body, nullptr);
  ASSERT_EQ(pipeline_body->op_count, 4u);

  loom_op_t* cse_run = pipeline_body->first_op;
  ASSERT_TRUE(loom_pass_run_isa(cse_run));
  EXPECT_TRUE(
      iree_string_view_equal(RunKey(module.get(), cse_run), IREE_SV("cse")));

  loom_op_t* operand_forms_run = cse_run->next_op;
  ASSERT_TRUE(loom_pass_run_isa(operand_forms_run));
  EXPECT_TRUE(iree_string_view_equal(RunKey(module.get(), operand_forms_run),
                                     IREE_SV("low-select-operand-forms")));

  loom_op_t* dce_run = operand_forms_run->next_op;
  ASSERT_TRUE(loom_pass_run_isa(dce_run));
  EXPECT_TRUE(iree_string_view_equal(RunKey(module.get(), dce_run),
                                     IREE_SV("low-dce")));
  EXPECT_TRUE(loom_pass_yield_isa(pipeline_body->last_op));
}

}  // namespace
}  // namespace loom
