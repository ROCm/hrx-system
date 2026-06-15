// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/llvmir/provider.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ops/llvmir/ops.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

TEST(LlvmirProviderTest, RegistersLlvmirDialect) {
  loom_target_environment_t target_environment = {0};
  IREE_ASSERT_OK(loom_target_environment_initialize(
      &loom_llvmir_target_provider_set, &target_environment));

  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(
      loom_target_environment_register_context(&target_environment, &context));
  IREE_ASSERT_OK(loom_context_finalize(&context));

  loom_op_kind_t kind = LOOM_OP_KIND_UNKNOWN;
  const loom_op_vtable_t* vtable =
      loom_context_lookup_op_by_name(&context, IREE_SV("llvmir.target"), &kind);
  ASSERT_NE(vtable, nullptr);
  EXPECT_EQ(kind, LOOM_OP_LLVMIR_TARGET);

  const loom_op_semantics_t semantics =
      loom_context_resolve_op_semantics(&context, LOOM_OP_LLVMIR_TARGET);
  EXPECT_NE(semantics.phase, LOOM_OP_PHASE_UNSPECIFIED);

  loom_context_deinitialize(&context);
  loom_target_environment_deinitialize(&target_environment);
}

TEST(LlvmirProviderTest, ParsesTargetRecordWithProviderRegisteredDialect) {
  loom_target_environment_t target_environment = {0};
  IREE_ASSERT_OK(loom_target_environment_initialize(
      &loom_llvmir_target_provider_set, &target_environment));

  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(
      loom_target_environment_register_context(&target_environment, &context));
  IREE_ASSERT_OK(loom_context_finalize(&context));

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  loom_module_t* raw_module = nullptr;
  IREE_ASSERT_OK(loom_text_parse(
      IREE_SV("llvmir.target<object> @llvm_generic {\n"
              "  triple = \"loom-direct64-unknown-none\",\n"
              "  data_layout = \"e-p:64:64-i64:64-n8:16:32:64-S128\"\n"
              "}\n"),
      IREE_SV("llvmir_provider_test.loom"), &context, &block_pool, nullptr,
      &raw_module));
  ModulePtr module(raw_module);
  ASSERT_NE(module, nullptr);

  module.reset();
  iree_arena_block_pool_deinitialize(&block_pool);
  loom_context_deinitialize(&context);
  loom_target_environment_deinitialize(&target_environment);
}

TEST(LlvmirProviderTest, ProviderSetContainsOnlyLlvmir) {
  ASSERT_EQ(loom_llvmir_target_provider_set.provider_count, 1u);
  ASSERT_NE(loom_llvmir_target_provider_set.providers, nullptr);
  EXPECT_EQ(loom_llvmir_target_provider_set.providers[0],
            &loom_llvmir_target_provider);
}

}  // namespace
}  // namespace loom
