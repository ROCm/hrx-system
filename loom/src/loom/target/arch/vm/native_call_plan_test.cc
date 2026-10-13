// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/native_call_plan.h"

#include <array>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace {

class VMNativeCallPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("calls"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    const char* names[] = {"signature", "version"};
    for (size_t i = 0; i < attrs_.size(); ++i) {
      IREE_ASSERT_OK(loom_module_intern_string(
          module_, iree_make_cstring_view(names[i]), &attrs_[i].name_id));
    }
    loom_string_id_t record_name, range_name, out_name;
    IREE_ASSERT_OK(
        loom_module_intern_string(module_, IREE_SV("vm.record"), &record_name));
    IREE_ASSERT_OK(
        loom_module_intern_string(module_, IREE_SV("vm.range"), &range_name));
    IREE_ASSERT_OK(
        loom_module_intern_string(module_, IREE_SV("vm.out"), &out_name));
    const loom_type_t scalar = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
    const loom_type_t fields[] = {scalar,
                                  loom_type_dialect(out_name, 1, &scalar)};
    const loom_type_t range = loom_type_dialect(range_name, 2, fields);
    loom_type_id_t signature;
    IREE_ASSERT_OK(loom_module_intern_type_id(
        module_, loom_type_dialect(record_name, 1, &range), &signature));
    attrs_[0].value = loom_attr_type(signature);
  }

  void TearDown() override {
    loom_module_free(module_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
    loom_context_deinitialize(&context_);
  }

  iree_status_t Build(uint16_t arguments = 1, uint16_t results = 1) {
    attrs_[1].value = loom_attr_i64(version_);
    loom_vm_native_import_t* import = nullptr;
    IREE_RETURN_IF_ERROR(loom_vm_native_import_build(
        module_, {attrs_.data(), attrs_.size()}, &arena_, &import));
    plan_ = {.arena = &arena_};
    std::vector<loom_vm_native_layout_plan_t*> by_type(module_->types.count);
    loom_vm_reference_plan_t references = {};
    IREE_RETURN_IF_ERROR(loom_vm_native_call_layout_bind(
        &plan_, module_, import, by_type.data(), &references, &arena_));
    std::vector<const loom_vm_native_layout_plan_t*> layouts(
        plan_.layout_count);
    for (auto* layout = plan_.layouts; layout; layout = layout->next) {
      layouts[layout->ordinal] = layout;
    }
    std::vector<iree_vm_bytecode_v0_signature_descriptor_row_t> operands(
        arguments + results, {.kind_u16 = IREE_VM_BYTECODE_SIGNATURE_KIND_I64});
    return loom_vm_native_call_template_build(
        import, loom_attr_i64_array(shape_.data(), shape_.size()),
        layouts.data(), operands.data(), arguments, results, &arena_,
        &call_template_);
  }

  // Public ABI dictionary; no binary offsets or selector placeholders.
  std::array<loom_named_attr_t, 2> attrs_ = {};
  // Semantic tree: ([(input, out)]).
  std::vector<int64_t> shape_ = {6, 7, 10, 0, 4};
  // Exact required export version.
  int64_t version_ = 7;
  // Owned compiler environment.
  loom_context_t context_ = {};
  // Shared module and plan block pool.
  iree_arena_block_pool_t pool_ = {};
  // Plan storage independent of module allocations.
  iree_arena_allocator_t arena_ = {};
  // Module supplying dictionary key spellings.
  loom_module_t* module_ = nullptr;
  // Retained declaration and physical callsite plan.
  loom_vm_native_call_plan_t plan_ = {};
  // Last successfully constructed call projection.
  loom_vm_native_call_template_t call_template_ = {};
};

TEST_F(VMNativeCallPlanTest, DerivesIndexedLayoutAndPartitionsCallsites) {
  IREE_ASSERT_OK(Build());
  EXPECT_EQ(call_template_.word_count, 4);
  EXPECT_EQ(call_template_.words[0], 1);
  EXPECT_EQ(call_template_.words[1], 2);
  EXPECT_EQ(call_template_.operand_offsets[0], 2);
  EXPECT_EQ(call_template_.operand_offsets[1], 3);
  call_template_.import_ordinal = 3;
  for (uint32_t i = 0; i < 3; ++i) {
    uint32_t ordinal = UINT32_MAX;
    uint32_t* words = nullptr;
    IREE_ASSERT_OK(loom_vm_native_call_plan_append(&plan_, &call_template_, i,
                                                   &ordinal, &words));
    EXPECT_EQ(ordinal, i);
    words[call_template_.operand_offsets[0]] = 10 + i;
    words[call_template_.operand_offsets[1]] = 20 + i;
    EXPECT_EQ(plan_.tail->row.function_ordinal_u16, i);
    EXPECT_EQ(plan_.tail->row.import_ordinal_u16, 3);
    EXPECT_EQ(plan_.tail->row.word_offset_u32, i * 4);
  }
  EXPECT_EQ(plan_.word_count, 12);
  EXPECT_EQ(plan_.head->words[2], 10);
  EXPECT_EQ(plan_.head->next->words[2], 11);
  EXPECT_EQ(call_template_.words[2], 0);
}

TEST_F(VMNativeCallPlanTest, EmptyRangeRetainsElementContract) {
  shape_ = {6, 3};
  IREE_ASSERT_OK(Build(0, 0));
  EXPECT_EQ(call_template_.word_count, 2);
  EXPECT_EQ(call_template_.words[0], 0);
  EXPECT_EQ(call_template_.words[1], 2);
}

TEST_F(VMNativeCallPlanTest, AcceptsEmptyRecord) {
  const auto signature =
      loom_type_table_get(&module_->types, attrs_[0].value.type_id);
  loom_type_id_t empty_signature;
  IREE_ASSERT_OK(loom_module_intern_type_id(
      module_,
      loom_type_dialect(loom_type_dialect_name_id(signature), 0, nullptr),
      &empty_signature));
  attrs_[0].value = loom_attr_type(empty_signature);
  shape_ = {2};
  IREE_ASSERT_OK(Build(0, 0));
  EXPECT_EQ(call_template_.word_count, 0);
}

TEST_F(VMNativeCallPlanTest, RejectsIncorrectAbiGroupingAndDirection) {
  // All cases are well-formed operand trees; only the ABI correspondence fails.
  for (const auto& shape : std::vector<std::vector<int64_t>>{
           {10, 0, 4}, {6, 7, 10, 4, 0}, {6, 6, 10, 0, 4}}) {
    shape_ = shape;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Build());
  }
  shape_ = {6, 7, 10, 0, 0};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Build(2, 0));
}

TEST_F(VMNativeCallPlanTest, RejectsNonU32Version) {
  for (int64_t version : {-1LL, 0x100000000LL}) {
    version_ = version;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Build());
  }
}

TEST_F(VMNativeCallPlanTest, RejectsUnknownLayoutField) {
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("record"),
                                           &attrs_[0].name_id));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Build());
}

}  // namespace
