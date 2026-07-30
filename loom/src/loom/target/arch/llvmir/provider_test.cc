// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/llvmir/provider.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/llvmir/ops.h"
#include "loom/ops/target/facts.h"
#include "loom/target/arch/llvmir/facts.h"
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

TEST(LlvmirProviderTest, ProjectsTargetFactsWithProviderRegisteredDialect) {
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
              "  data_layout = \"e-p:64:64-i64:64-n8:16:32:64-S128\",\n"
              "  cpu = \"generic\",\n"
              "  features = \"+sse2\"\n"
              "}\n"),
      IREE_SV("llvmir_provider_test.loom"), &context, &block_pool, nullptr,
      &raw_module));
  ModulePtr module(raw_module);
  ASSERT_NE(module, nullptr);

  iree_arena_allocator_t fact_arena;
  iree_arena_initialize(&block_pool, &fact_arena);
  loom_symbol_fact_table_t fact_table = {0};
  loom_symbol_fact_table_initialize(&fact_table, &fact_arena);
  const loom_string_id_t symbol_name_id =
      loom_module_lookup_string(module.get(), IREE_SV("llvm_generic"));
  ASSERT_NE(symbol_name_id, LOOM_STRING_ID_INVALID);
  const loom_symbol_id_t symbol_id =
      loom_module_find_symbol(module.get(), symbol_name_id);
  ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
  const loom_symbol_facts_base_t* base_facts = nullptr;
  IREE_ASSERT_OK(loom_symbol_fact_table_lookup(&fact_table, module.get(),
                                               symbol_id, &base_facts));
  const loom_target_symbol_facts_t* target_facts =
      loom_target_symbol_facts_cast(base_facts);
  ASSERT_NE(target_facts, nullptr);
  const loom_llvmir_target_facts_t* llvmir_facts =
      loom_llvmir_target_facts_cast(target_facts->projection);
  ASSERT_NE(llvmir_facts, nullptr);
  EXPECT_TRUE(iree_string_view_equal(llvmir_facts->target_triple,
                                     IREE_SV("loom-direct64-unknown-none")));
  EXPECT_TRUE(iree_string_view_equal(
      llvmir_facts->data_layout, IREE_SV("e-p:64:64-i64:64-n8:16:32:64-S128")));
  EXPECT_TRUE(
      iree_string_view_equal(llvmir_facts->target_cpu, IREE_SV("generic")));
  EXPECT_TRUE(
      iree_string_view_equal(llvmir_facts->target_features, IREE_SV("+sse2")));
  EXPECT_TRUE(llvmir_facts->authored.target_triple);
  EXPECT_TRUE(llvmir_facts->authored.data_layout);
  EXPECT_TRUE(llvmir_facts->authored.target_cpu);
  EXPECT_TRUE(llvmir_facts->authored.target_features);

  iree_arena_deinitialize(&fact_arena);
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
