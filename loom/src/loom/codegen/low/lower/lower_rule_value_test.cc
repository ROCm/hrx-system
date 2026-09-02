// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/lower_rule_value.h"

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/testing/source_workload.h"
#include "loom/ir/context.h"
#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/scalar/ops.h"

namespace loom {
namespace {

class LowLowerRuleValueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &analysis_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_low_source_workload_register_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(
        &context_, IREE_SV("lower_rule_value_test"), &block_pool_, nullptr,
        iree_allocator_system(), &module_));
    BuildFunction();
    IREE_ASSERT_OK(loom_value_fact_table_initialize(
        &fact_table_, &analysis_arena_, module_->values.count));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&analysis_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void BuildFunction() {
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(
        &module_builder, IREE_SV("project_values"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    const loom_symbol_ref_t symbol = {
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
    const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    const loom_type_t argument_types[] = {i32_type, i32_type};
    loom_op_t* function_op = nullptr;
    IREE_ASSERT_OK(loom_func_def_build(
        &module_builder, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0,
        /*cc=*/0, /*purity=*/0, /*temperature=*/0, /*inline_policy=*/0,
        loom_symbol_ref_null(), /*abi=*/0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol,
        argument_types, IREE_ARRAYSIZE(argument_types), &i32_type, 1, nullptr,
        0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &function_op));
    function_ = loom_func_like_cast(module_, function_op);

    uint16_t argument_count = 0;
    arguments_ = loom_func_like_arg_ids(function_, &argument_count);
    ASSERT_EQ(argument_count, 2u);
    loom_builder_t body_builder;
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_func_like_body(function_)), &body_builder);
    body_builder.ip.parent_op = function_op;
    IREE_ASSERT_OK(loom_scalar_addi_build(
        &body_builder, /*overflow_flags=*/0, arguments_[0], arguments_[1],
        i32_type, LOOM_LOCATION_UNKNOWN, &addi_op_));
    IREE_ASSERT_OK(loom_scalar_constant_build(
        &body_builder, loom_attr_i64(7), loom_type_scalar(LOOM_SCALAR_TYPE_I64),
        LOOM_LOCATION_UNKNOWN, &integer_constant_op_));
    IREE_ASSERT_OK(
        loom_scalar_constant_build(&body_builder, loom_attr_f64(1.5),
                                   loom_type_scalar(LOOM_SCALAR_TYPE_F32),
                                   LOOM_LOCATION_UNKNOWN, &float_constant_op_));
    const loom_value_id_t result = loom_scalar_addi_result(addi_op_);
    loom_op_t* return_op = nullptr;
    IREE_ASSERT_OK(loom_func_return_build(&body_builder, &result, 1,
                                          LOOM_LOCATION_UNKNOWN, &return_op));
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t analysis_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t function_ = {};
  const loom_value_id_t* arguments_ = nullptr;
  loom_op_t* addi_op_ = nullptr;
  loom_op_t* integer_constant_op_ = nullptr;
  loom_op_t* float_constant_op_ = nullptr;
  loom_value_fact_table_t fact_table_ = {};
};

TEST_F(LowLowerRuleValueTest, ResolvesSourceValueReferencesAndFields) {
  loom_low_lower_value_ref_t value_refs[3] = {};
  value_refs[0].kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND;
  value_refs[0].index = 0;
  value_refs[1].kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND;
  value_refs[1].index = 1;
  value_refs[2].kind = LOOM_LOW_LOWER_VALUE_REF_RESULT;
  value_refs[2].index = 0;
  loom_low_lower_rule_set_t rule_set = {};
  rule_set.value_refs = value_refs;
  rule_set.value_ref_count = IREE_ARRAYSIZE(value_refs);

  EXPECT_EQ(loom_low_lower_rule_source_value(module_, &rule_set, addi_op_, 0),
            arguments_[0]);
  EXPECT_EQ(loom_low_lower_rule_source_value(module_, &rule_set, addi_op_, 1),
            arguments_[1]);
  EXPECT_EQ(loom_low_lower_rule_source_value(module_, &rule_set, addi_op_, 2),
            loom_scalar_addi_result(addi_op_));

  const loom_value_slice_t lhs_field =
      loom_low_lower_rule_value_ref_field_span(module_, &rule_set, addi_op_, 0);
  ASSERT_EQ(lhs_field.count, 1u);
  EXPECT_EQ(lhs_field.values[0], arguments_[0]);
  const loom_value_slice_t result_field =
      loom_low_lower_rule_value_ref_field_span(module_, &rule_set, addi_op_, 2);
  ASSERT_EQ(result_field.count, 1u);
  EXPECT_EQ(result_field.values[0], loom_scalar_addi_result(addi_op_));
}

TEST_F(LowLowerRuleValueTest, ProjectsExactScalarFacts) {
  const loom_value_id_t integer_value =
      loom_scalar_constant_result(integer_constant_op_);
  IREE_ASSERT_OK(loom_value_fact_table_define(&fact_table_, integer_value,
                                              loom_value_facts_exact_i64(7)));
  loom_value_facts_t integer_facts = loom_value_facts_unknown();
  ASSERT_TRUE(loom_low_lower_rule_integer_immediate_facts(
      module_, &fact_table_, integer_value, &integer_facts));
  int64_t exact_integer = 0;
  ASSERT_TRUE(loom_value_facts_as_exact_i64(integer_facts, &exact_integer));
  EXPECT_EQ(exact_integer, 7);

  const loom_value_id_t float_value =
      loom_scalar_constant_result(float_constant_op_);
  IREE_ASSERT_OK(loom_value_fact_table_define(
      &fact_table_, float_value,
      loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 1.5)));
  loom_value_facts_t float_facts = loom_value_facts_unknown();
  ASSERT_TRUE(loom_low_lower_rule_float_immediate_facts(
      module_, &fact_table_, float_value, &float_facts));
  double exact_float = 0.0;
  ASSERT_TRUE(loom_value_facts_as_exact_float(LOOM_SCALAR_TYPE_F32, float_facts,
                                              &exact_float));
  EXPECT_DOUBLE_EQ(exact_float, 1.5);
}

TEST_F(LowLowerRuleValueTest, DerivesExactUnsignedDivisorRecipes) {
  const loom_value_id_t value_id =
      loom_scalar_constant_result(integer_constant_op_);
  const uint32_t divisors[] = {
      2u, 3u, 5u, 7u, 10u, 31u, UINT32_C(0x80000000), UINT32_MAX};
  const uint32_t numerators[] = {
      0u, 1u, 2u, 6u, 7u, 8u, UINT32_C(0x7fffffff), UINT32_MAX};
  for (uint32_t divisor : divisors) {
    IREE_ASSERT_OK(loom_value_fact_table_define(
        &fact_table_, value_id, loom_value_facts_exact_i64(divisor)));
    loom_low_lower_u32_divisor_magic_info_t info = {};
    ASSERT_TRUE(loom_low_lower_rule_value_facts_u32_divisor_magic_info(
        module_, &fact_table_, value_id, &info));
    for (uint32_t numerator : numerators) {
      uint32_t quotient =
          static_cast<uint32_t>((static_cast<uint64_t>(numerator) *
                                 static_cast<uint64_t>(info.multiplier)) >>
                                32);
      if (info.is_add) {
        quotient = ((numerator - quotient) >> 1) + quotient;
      }
      quotient >>= info.post_shift;
      EXPECT_EQ(quotient, numerator / divisor)
          << "numerator=" << numerator << " divisor=" << divisor;
    }
  }
}

}  // namespace
}  // namespace loom
