// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/condition.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"

namespace loom {
namespace {

class TargetConditionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  static loom_attribute_t MakeClause(loom_parameterized_attr_kind_t family_kind,
                                     int64_t value,
                                     loom_attribute_t* out_slot) {
    *out_slot = loom_attr_i64(value);
    return loom_make_parameterized_attr(family_kind, out_slot, 1);
  }

  static void InitializeFacts(const loom_target_fact_type_t* fact_type,
                              uint32_t subgroup_size,
                              loom_target_facts_t* out_facts) {
    *out_facts = {};
    out_facts->fact_type = fact_type;
    out_facts->storage.snapshot.subgroup_size = subgroup_size;
    loom_target_bundle_storage_rebind(&out_facts->storage);
  }

  iree_arena_block_pool_t block_pool_ = {};
  loom_context_t context_ = {};
};

TEST_F(TargetConditionTest, SubgroupSizeUsesCompactPublicSpelling) {
  static constexpr const char* kSource =
      "test.func @conditions() {\n"
      "  test.parameterized_attr_array [#target.subgroup.size<64>]\n"
      "  test.yield\n"
      "}\n";
  loom_text_parse_options_t options = {};
  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(loom_text_parse(iree_make_cstring_view(kSource),
                                 IREE_SV("target_condition_test.loom"),
                                 &context_, &block_pool_, &options, &module));
  ASSERT_NE(module, nullptr);

  loom_op_t* function = loom_block_op(loom_module_block(module), 0);
  ASSERT_TRUE(loom_test_func_isa(function));
  loom_op_t* op =
      loom_block_op(loom_region_entry_block(loom_test_func_body(function)), 0);
  ASSERT_TRUE(loom_test_parameterized_attr_array_isa(op));
  const loom_parameterized_attr_array_t conditions =
      loom_test_parameterized_attr_array_values(op);
  ASSERT_EQ(conditions.count, 1u);
  ASSERT_TRUE(loom_target_subgroup_size_attr_isa(conditions.values[0]));
  EXPECT_EQ(loom_target_subgroup_size_attr_size(conditions.values[0]), 64);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_text_print_module_to_builder(module, &builder,
                                                   LOOM_TEXT_PRINT_DEFAULT));
  const std::string text(iree_string_builder_buffer(&builder),
                         iree_string_builder_size(&builder));
  EXPECT_NE(text.find("#target.subgroup.size<64>"), std::string::npos);
  iree_string_builder_deinitialize(&builder);
  loom_module_free(module);
}

TEST_F(TargetConditionTest, ResolvesRegisteredConditionDescriptor) {
  loom_attribute_t slot = {};
  const loom_attribute_t condition =
      MakeClause(LOOM_PARAMETERIZED_ATTR_TARGET_SUBGROUP_SIZE, 64, &slot);

  const loom_target_condition_descriptor_t* descriptor = nullptr;
  IREE_ASSERT_OK(
      loom_target_condition_resolve(&context_, condition, &descriptor));
  EXPECT_EQ(descriptor, &loom_target_subgroup_size_condition);
}

TEST_F(TargetConditionTest, RejectsNonConditionFamily) {
  loom_attribute_t slot = {};
  const loom_attribute_t value =
      MakeClause(LOOM_PARAMETERIZED_ATTR_TEST_TILE, 16, &slot);

  const loom_target_condition_descriptor_t* descriptor = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_condition_resolve(&context_, value, &descriptor));
  EXPECT_EQ(descriptor, nullptr);
}

TEST_F(TargetConditionTest, RejectsInvalidOrUnregisteredFamily) {
  const loom_target_condition_descriptor_t* descriptor = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_condition_resolve(&context_, loom_attr_i64(64), &descriptor));
  EXPECT_EQ(descriptor, nullptr);

  loom_attribute_t slot = {};
  const loom_attribute_t unregistered = MakeClause(
      LOOM_PARAMETERIZED_ATTR_KIND(LOOM_DIALECT_TARGET, 1), 64, &slot);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_condition_resolve(&context_, unregistered, &descriptor));
  EXPECT_EQ(descriptor, nullptr);
}

TEST_F(TargetConditionTest, ValidatesSubgroupSizeDomain) {
  const int64_t invalid_sizes[] = {
      -1,
      0,
      static_cast<int64_t>(UINT32_MAX) + 1,
  };
  for (int64_t invalid_size : invalid_sizes) {
    loom_attribute_t slot = {};
    const loom_attribute_t condition = MakeClause(
        LOOM_PARAMETERIZED_ATTR_TARGET_SUBGROUP_SIZE, invalid_size, &slot);
    const loom_target_condition_descriptor_t* descriptor = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_target_condition_resolve(&context_, condition, &descriptor));
    EXPECT_EQ(descriptor, nullptr);
  }
}

TEST_F(TargetConditionTest, EvaluatesNormalizedSubgroupSize) {
  static const loom_target_fact_type_t kFactType = {
      /*.name=*/IREE_SVL("normalized-test"),
      /*.storage_size=*/sizeof(loom_target_facts_t),
  };
  loom_attribute_t slot = {};
  const loom_attribute_t condition =
      MakeClause(LOOM_PARAMETERIZED_ATTR_TARGET_SUBGROUP_SIZE, 64, &slot);
  const loom_target_condition_descriptor_t* descriptor = nullptr;
  IREE_ASSERT_OK(
      loom_target_condition_resolve(&context_, condition, &descriptor));

  EXPECT_EQ(loom_target_condition_evaluate(descriptor, condition, nullptr),
            LOOM_TARGET_CONDITION_UNBOUND);

  loom_target_facts_t facts = {};
  InitializeFacts(&kFactType, 0, &facts);
  EXPECT_EQ(loom_target_condition_evaluate(descriptor, condition, &facts),
            LOOM_TARGET_CONDITION_UNKNOWN);

  InitializeFacts(&kFactType, 64, &facts);
  EXPECT_EQ(loom_target_condition_evaluate(descriptor, condition, &facts),
            LOOM_TARGET_CONDITION_MATCH);

  InitializeFacts(&kFactType, 32, &facts);
  EXPECT_EQ(loom_target_condition_evaluate(descriptor, condition, &facts),
            LOOM_TARGET_CONDITION_REJECT);
}

TEST_F(TargetConditionTest, RejectsIncompatibleStaticFactFamily) {
  static const loom_target_fact_type_t kRequiredFactType = {
      /*.name=*/IREE_SVL("required-test"),
      /*.storage_size=*/sizeof(loom_target_facts_t),
  };
  static const loom_target_fact_type_t kOtherFactType = {
      /*.name=*/IREE_SVL("other-test"),
      /*.storage_size=*/sizeof(loom_target_facts_t),
  };
  loom_target_condition_descriptor_t typed_descriptor =
      loom_target_subgroup_size_condition;
  typed_descriptor.required_fact_type = &kRequiredFactType;

  loom_attribute_t slot = {};
  const loom_attribute_t condition =
      MakeClause(LOOM_PARAMETERIZED_ATTR_TARGET_SUBGROUP_SIZE, 64, &slot);
  loom_target_facts_t facts = {};
  InitializeFacts(&kRequiredFactType, 64, &facts);
  EXPECT_EQ(
      loom_target_condition_evaluate(&typed_descriptor, condition, &facts),
      LOOM_TARGET_CONDITION_MATCH);

  InitializeFacts(&kOtherFactType, 64, &facts);
  EXPECT_EQ(
      loom_target_condition_evaluate(&typed_descriptor, condition, &facts),
      LOOM_TARGET_CONDITION_REJECT);
}

}  // namespace
}  // namespace loom
