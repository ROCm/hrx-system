// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/ops.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class EncodingFactsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t encoding_vtable_count = 0;
    const loom_op_vtable_t* const* encoding_vtables =
        loom_encoding_dialect_vtables(&encoding_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_ENCODING, encoding_vtables,
        (uint16_t)encoding_vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(EncodingFactsTest, StridedLayoutMalformedDynamicCountDegrades) {
  int64_t static_strides[] = {INT64_MIN, 1};
  loom_op_t* layout = nullptr;
  IREE_ASSERT_OK(loom_encoding_layout_strided_build(
      &builder_, /*strides=*/nullptr, /*strides_count=*/0, static_strides,
      IREE_ARRAYSIZE(static_strides),
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT),
      LOOM_LOCATION_UNKNOWN, &layout));

  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &module_->arena, 0));
  loom_value_facts_t result_facts[1] = {loom_value_facts_exact_i64(42)};
  IREE_ASSERT_OK(loom_encoding_layout_strided_facts(
      &table.context, module_, layout, /*operand_facts=*/nullptr,
      result_facts));

  loom_value_fact_encoding_summary_t summary = {};
  ASSERT_TRUE(loom_value_facts_query_encoding_summary(
      &table.context, result_facts[0], &summary));
  EXPECT_EQ(summary.role, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
  EXPECT_EQ(summary.address_layout.kind,
            LOOM_VALUE_FACT_ADDRESS_LAYOUT_UNKNOWN);
}

}  // namespace
}  // namespace loom
