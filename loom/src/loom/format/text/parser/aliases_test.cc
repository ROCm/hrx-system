// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/parser/aliases.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/families.h"

namespace loom {
namespace {

class AliasesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_register_builtin_encoding_vtables(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &pool_);
    iree_arena_initialize(&pool_, &arena_);
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("aliases"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_encoding_t encoding = {.alias_id = LOOM_STRING_ID_INVALID};
    IREE_ASSERT_OK(loom_module_intern_string(
        module_, IREE_SV("encoding.layout.dense"), &encoding.name_id));
    IREE_ASSERT_OK(loom_module_add_encoding(module_, &encoding, &encoding_id_));
    ASSERT_TRUE(loom_encoding_static_is_valid(
        loom_module_encoding(module_, encoding_id_)));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    loom_module_free(module_);
    iree_arena_block_pool_deinitialize(&pool_);
    loom_context_deinitialize(&context_);
  }

  // Registered encoding families shared by the module and source aliases.
  loom_context_t context_ = {};
  // Shared block ownership matching the parser's module and scratch arenas.
  iree_arena_block_pool_t pool_ = {};
  // Temporary source-name index storage, released after parsing.
  iree_arena_allocator_t arena_ = {};
  // Owner of canonical strings and the encoding referenced by all aliases.
  loom_module_t* module_ = nullptr;
  // Valid canonical encoding shared by distinct source names.
  uint16_t encoding_id_ = 0;
};

TEST_F(AliasesTest, MultipleSourceNamesSurviveGrowthWithoutRetiredStorage) {
  loom_alias_table_t aliases = {};
  EXPECT_EQ(loom_alias_table_lookup(&aliases, module_, IREE_SV("unknown")), 0);
  EXPECT_EQ(arena_.used_allocation_size, 0u);
  constexpr uint32_t kNameCount = 128;
  for (uint32_t i = 0; i < kNameCount; ++i) {
    const std::string spelling = "layout_" + std::to_string(i);
    const auto name = iree_make_string_view(spelling.data(), spelling.size());
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(module_, name, &name_id));
    IREE_ASSERT_OK(
        loom_alias_table_add(&aliases, &arena_, name_id, encoding_id_));
  }
  EXPECT_EQ(module_->encodings.count, 1u);
  EXPECT_EQ(arena_.used_allocation_size, 2 * sizeof(loom_intern_segment_t));
  for (uint32_t i = 0; i < kNameCount; ++i) {
    const std::string spelling = "layout_" + std::to_string(i);
    const auto name = iree_make_string_view(spelling.data(), spelling.size());
    EXPECT_EQ(loom_alias_table_lookup(&aliases, module_, name), encoding_id_);
  }

  // Neither an uninterned spelling nor an interned non-alias gains a binding.
  const auto string_count = module_->strings.count;
  const auto module_bytes = module_->arena.used_allocation_size;
  const auto alias_bytes = arena_.used_allocation_size;
  EXPECT_EQ(loom_alias_table_lookup(&aliases, module_, IREE_SV("unknown")), 0);
  EXPECT_EQ(loom_alias_table_lookup(&aliases, module_,
                                    IREE_SV("encoding.layout.dense")),
            0);
  EXPECT_EQ(module_->strings.count, string_count);
  EXPECT_EQ(module_->arena.used_allocation_size, module_bytes);
  EXPECT_EQ(arena_.used_allocation_size, alias_bytes);
}

}  // namespace
}  // namespace loom
