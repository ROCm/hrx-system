// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/type.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/format.h"
#include "loom/format/bytecode/reader/module_view.h"
#include "loom/format/bytecode/reader/type_validator.h"

namespace loom {
namespace {

static iree_status_t AcceptDiagnostic(void* user_data,
                                      const loom_diagnostic_t* diagnostic) {
  (void)user_data;
  (void)diagnostic;
  return iree_ok_status();
}

class BytecodeTypeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("type_test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_bytecode_reader_decoder_initialize(
        loom_diagnostic_sink_t{AcceptDiagnostic, nullptr},
        IREE_SV("type_test.loombc"), &error_count_, &decoder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t BuildPlan(const uint8_t* data, iree_host_size_t length) {
    return loom_bytecode_type_plan_build(
        &decoder_, &context_, &module_view_, &scratch_arena_,
        iree_make_const_byte_span(data, length),
        /*section_absolute_offset=*/0);
  }

  loom_bytecode_type_materializer_t MakeMaterializer(const uint8_t* data,
                                                     iree_host_size_t length) {
    return loom_bytecode_type_materializer_t{
        /*.decoder=*/&decoder_,
        /*.bytecode=*/iree_make_const_byte_span(data, length),
        /*.context=*/&context_,
        /*.module_view=*/&module_view_,
        /*.scratch_arena=*/&scratch_arena_,
        /*.output_module=*/module_,
    };
  }

  // Minimal validated module facts receiving each test's type plan.
  loom_bytecode_reader_module_view_t module_view_ = {};
  // Bounded wire decoder sharing this fixture's diagnostic count.
  loom_bytecode_reader_decoder_t decoder_ = {};
  // Number of accepted malformed-input diagnostics.
  uint32_t error_count_ = 0;
  // Block source shared by scratch and output-module arenas.
  iree_arena_block_pool_t block_pool_;
  // Storage owning the immutable plan and temporary type payloads.
  iree_arena_allocator_t scratch_arena_;
  // Finalized empty registry sufficient for built-in types.
  loom_context_t context_;
  // Output module receiving canonical materialized types.
  loom_module_t* module_ = nullptr;
};

TEST_F(BytecodeTypeTest, BuildsAndMaterializesTopologicalPlan) {
  const uint8_t data[] = {
      0x03,
      LOOM_BYTECODE_TYPE_NONE,
      LOOM_BYTECODE_TYPE_SCALAR,
      LOOM_SCALAR_TYPE_I32,
      LOOM_BYTECODE_TYPE_FUNCTION,
      0x01,
      0x01,
      0x01,
      0x00,
  };
  IREE_ASSERT_OK(BuildPlan(data, sizeof(data)));

  ASSERT_EQ(module_view_.types.count, 3u);
  EXPECT_EQ(module_view_.types.entries[0].bytecode_offset, 1u);
  EXPECT_EQ(module_view_.types.entries[1].bytecode_offset, 2u);
  EXPECT_EQ(module_view_.types.entries[2].bytecode_offset, 4u);
  ASSERT_NE(module_view_.types.facts, nullptr);
  EXPECT_EQ(module_view_.types.facts->type_id, 2u);
  EXPECT_EQ(module_view_.types.facts->next, nullptr);

  loom_bytecode_type_materializer_t materializer =
      MakeMaterializer(data, sizeof(data));
  IREE_ASSERT_OK(loom_bytecode_type_materialize(&materializer));
  ASSERT_EQ(module_->types.count, 3u);
  EXPECT_EQ(loom_type_kind(module_->types.entries[0]), LOOM_TYPE_NONE);
  EXPECT_EQ(loom_type_kind(module_->types.entries[1]), LOOM_TYPE_SCALAR);
  EXPECT_EQ(loom_type_kind(module_->types.entries[2]), LOOM_TYPE_FUNCTION);
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeTypeTest, RejectsNonTopologicalTypeReference) {
  const uint8_t data[] = {
      0x01, LOOM_BYTECODE_TYPE_FUNCTION, 0x01, 0x00, 0x00,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED, BuildPlan(data, sizeof(data)));
  EXPECT_EQ(error_count_, 1u);
}

}  // namespace
}  // namespace loom
