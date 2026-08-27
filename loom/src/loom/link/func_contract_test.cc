// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/func_contract.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/testdata/func_contract_testdata.h"
#include "loom/ops/op_registry.h"

namespace loom {
namespace {

class LinkFuncContractTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &scratch_arena_);

    const iree_file_toc_t* files = loom_link_func_contract_testdata_create();
    ASSERT_EQ(loom_link_func_contract_testdata_size(), 1u);
    const iree_string_view_t source =
        iree_make_string_view(files[0].data, files[0].size);
    const iree_string_view_t file_name = iree_make_cstring_view(files[0].name);
    loom_text_parse_options_t options = {};
    IREE_ASSERT_OK(loom_text_parse(source, file_name, &context_, &block_pool_,
                                   &options, &module_));
    ASSERT_NE(module_, nullptr);
  }

  void TearDown() override {
    loom_module_free(module_);
    iree_arena_deinitialize(&scratch_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_op_t* FindOp(iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module_, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id =
        loom_module_find_symbol(module_, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    loom_op_t* op = module_->symbols.entries[symbol_id].defining_op;
    IREE_ASSERT(op != nullptr);
    return op;
  }

  loom_link_func_contract_mismatch_t Check(iree_string_view_t source_name,
                                           iree_string_view_t selected_name) {
    loom_ir_remap_t remap = {};
    IREE_CHECK_OK(loom_ir_remap_initialize(module_, module_, &scratch_arena_,
                                           /*options=*/nullptr, &remap));
    loom_link_func_contract_t source =
        loom_link_func_contract_from_op(module_, FindOp(source_name));
    loom_link_func_contract_t selected =
        loom_link_func_contract_from_op(module_, FindOp(selected_name));
    loom_link_func_contract_mismatch_t mismatch = {};
    IREE_CHECK_OK(
        loom_link_func_contract_check(&source, &selected, &remap, &mismatch));
    return mismatch;
  }

  bool AttrPresent(loom_op_t* op, uint8_t attr_index) {
    return attr_index != LOOM_ATTR_INDEX_NONE &&
           !loom_attr_is_absent(loom_op_attrs(op)[attr_index]);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  iree_arena_allocator_t scratch_arena_;
  loom_module_t* module_ = nullptr;
};

TEST_F(LinkFuncContractTest, CheckDoesNotFillCompatibleSelectedContract) {
  loom_op_t* selected_op = FindOp(IREE_SV("selected"));
  const loom_func_like_vtable_t* selected_function =
      loom_op_vtable(module_, selected_op)->func_like;
  ASSERT_NE(selected_function, nullptr);
  ASSERT_FALSE(AttrPresent(selected_op, selected_function->purity_attr_index));
  ASSERT_FALSE(
      AttrPresent(selected_op, selected_function->temperature_attr_index));
  ASSERT_FALSE(
      AttrPresent(selected_op, selected_function->inline_policy_attr_index));

  const loom_link_func_contract_mismatch_t mismatch =
      Check(IREE_SV("source"), IREE_SV("selected"));

  EXPECT_EQ(mismatch.kind, LOOM_LINK_FUNC_CONTRACT_MISMATCH_NONE);
  EXPECT_FALSE(AttrPresent(selected_op, selected_function->purity_attr_index));
  EXPECT_FALSE(
      AttrPresent(selected_op, selected_function->temperature_attr_index));
  EXPECT_FALSE(
      AttrPresent(selected_op, selected_function->inline_policy_attr_index));
}

TEST_F(LinkFuncContractTest, MergeFillsCompatibleSelectedContract) {
  loom_op_t* source_op = FindOp(IREE_SV("source"));
  loom_op_t* selected_op = FindOp(IREE_SV("selected"));
  loom_link_func_contract_t source =
      loom_link_func_contract_from_op(module_, source_op);
  loom_link_func_contract_t selected =
      loom_link_func_contract_from_op(module_, selected_op);
  loom_ir_remap_t remap = {};
  IREE_ASSERT_OK(loom_ir_remap_initialize(module_, module_, &scratch_arena_,
                                          /*options=*/nullptr, &remap));

  loom_link_func_contract_mismatch_t mismatch = {};
  IREE_ASSERT_OK(loom_link_func_contract_merge(&source, &selected, &remap,
                                               /*flags=*/0, &mismatch));

  EXPECT_EQ(mismatch.kind, LOOM_LINK_FUNC_CONTRACT_MISMATCH_NONE);
  const uint8_t source_purity_index = source.function->purity_attr_index;
  const uint8_t selected_purity_index = selected.function->purity_attr_index;
  ASSERT_TRUE(AttrPresent(source_op, source_purity_index));
  ASSERT_TRUE(AttrPresent(selected_op, selected_purity_index));
  EXPECT_TRUE(
      loom_attribute_equal(&source.attributes[source_purity_index],
                           &selected.attributes[selected_purity_index]));
}

TEST_F(LinkFuncContractTest, ReportsFieldMismatch) {
  const loom_link_func_contract_mismatch_t mismatch =
      Check(IREE_SV("source"), IREE_SV("field_mismatch"));

  EXPECT_EQ(mismatch.kind, LOOM_LINK_FUNC_CONTRACT_MISMATCH_FIELD);
  EXPECT_TRUE(
      iree_string_view_equal(mismatch.field_name, IREE_SV("temperature")));
}

TEST_F(LinkFuncContractTest, ReportsArgumentCountMismatch) {
  const loom_link_func_contract_mismatch_t mismatch =
      Check(IREE_SV("source"), IREE_SV("argument_count_mismatch"));

  EXPECT_EQ(mismatch.kind, LOOM_LINK_FUNC_CONTRACT_MISMATCH_COUNT);
  EXPECT_TRUE(iree_string_view_equal(mismatch.field_name, IREE_SV("args")));
  EXPECT_EQ(mismatch.detail.counts.source, 1u);
  EXPECT_EQ(mismatch.detail.counts.selected, 2u);
}

TEST_F(LinkFuncContractTest, ReportsArgumentTypeMismatch) {
  const loom_link_func_contract_mismatch_t mismatch =
      Check(IREE_SV("source"), IREE_SV("argument_type_mismatch"));

  EXPECT_EQ(mismatch.kind, LOOM_LINK_FUNC_CONTRACT_MISMATCH_TYPE);
  EXPECT_TRUE(iree_string_view_equal(mismatch.field_name, IREE_SV("arg")));
  EXPECT_EQ(mismatch.detail.type_ordinal, 0u);
}

TEST_F(LinkFuncContractTest, ReportsResultTypeMismatch) {
  const loom_link_func_contract_mismatch_t mismatch =
      Check(IREE_SV("source"), IREE_SV("result_type_mismatch"));

  EXPECT_EQ(mismatch.kind, LOOM_LINK_FUNC_CONTRACT_MISMATCH_TYPE);
  EXPECT_TRUE(iree_string_view_equal(mismatch.field_name, IREE_SV("result")));
  EXPECT_EQ(mismatch.detail.type_ordinal, 0u);
}

TEST_F(LinkFuncContractTest, ReportsKernelWorkloadCountMismatch) {
  const loom_link_func_contract_mismatch_t mismatch =
      Check(IREE_SV("kernel_source"), IREE_SV("kernel_count_mismatch"));

  EXPECT_EQ(mismatch.kind, LOOM_LINK_FUNC_CONTRACT_MISMATCH_COUNT);
  EXPECT_TRUE(iree_string_view_equal(mismatch.field_name,
                                     IREE_SV("kernel workload args")));
  EXPECT_EQ(mismatch.detail.counts.source, 1u);
  EXPECT_EQ(mismatch.detail.counts.selected, 2u);
}

TEST_F(LinkFuncContractTest, ReportsKernelWorkloadTypeMismatch) {
  const loom_link_func_contract_mismatch_t mismatch =
      Check(IREE_SV("kernel_source"), IREE_SV("kernel_type_mismatch"));

  EXPECT_EQ(mismatch.kind, LOOM_LINK_FUNC_CONTRACT_MISMATCH_TYPE);
  EXPECT_TRUE(iree_string_view_equal(mismatch.field_name,
                                     IREE_SV("kernel workload arg")));
  EXPECT_EQ(mismatch.detail.type_ordinal, 0u);
}

}  // namespace
}  // namespace loom
