// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/rule_value.h"

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/math.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/testing/source_workload.h"
#include "loom/ir/context.h"
#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"

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
        .module_id = 0,
        .symbol_id = symbol_id,
    };
    const loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    const loom_type_t bf16_vector_type = loom_type_shaped_1d(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_BF16, loom_dim_pack_static(16), 0);
    const loom_type_t f32_vector_type = loom_type_shaped_1d(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 0);
    const loom_type_t argument_types[] = {
        i32_type,
        i32_type,
        bf16_vector_type,
        f32_vector_type,
    };
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
    ASSERT_EQ(argument_count, 4u);
    loom_builder_t body_builder;
    loom_builder_initialize(
        module_, &module_->arena,
        loom_region_entry_block(loom_func_like_body(function_)), &body_builder);
    body_builder.ip.parent_op = function_op;
    IREE_ASSERT_OK(loom_scalar_addi_build(
        &body_builder, /*overflow_flags=*/0, arguments_[0], arguments_[1],
        i32_type, LOOM_LOCATION_UNKNOWN, &addi_op_));
    IREE_ASSERT_OK(loom_scalar_muli_build(
        &body_builder, /*overflow_flags=*/0, loom_scalar_addi_result(addi_op_),
        arguments_[1], i32_type, LOOM_LOCATION_UNKNOWN, &muli_op_));
    IREE_ASSERT_OK(loom_vector_extf_build(
        &body_builder, /*instance_flags=*/0, arguments_[2], bf16_vector_type,
        f32_vector_type, LOOM_LOCATION_UNKNOWN, &direct_extf_op_));
    IREE_ASSERT_OK(loom_vector_fptrunc_build(
        &body_builder, /*instance_flags=*/0, arguments_[3], f32_vector_type,
        bf16_vector_type, LOOM_LOCATION_UNKNOWN, &fptrunc_op_));
    IREE_ASSERT_OK(loom_vector_extf_build(
        &body_builder, /*instance_flags=*/0,
        loom_vector_fptrunc_result(fptrunc_op_), bf16_vector_type,
        f32_vector_type, LOOM_LOCATION_UNKNOWN, &roundtrip_extf_op_));
    IREE_ASSERT_OK(loom_vector_negf_build(
        &body_builder, /*instance_flags=*/0,
        loom_vector_extf_result(roundtrip_extf_op_), f32_vector_type,
        LOOM_LOCATION_UNKNOWN, &vector_consumer_op_));
    const loom_type_t assume_result_types[] = {i32_type, i32_type};
    IREE_ASSERT_OK(loom_scalar_assume_build(
        &body_builder, arguments_, 2, /*predicates=*/nullptr,
        /*predicates_count=*/0, assume_result_types,
        IREE_ARRAYSIZE(assume_result_types), LOOM_LOCATION_UNKNOWN,
        &variadic_result_op_));
    IREE_ASSERT_OK(loom_scalar_constant_build(
        &body_builder, loom_attr_i64(7), loom_type_scalar(LOOM_SCALAR_TYPE_I64),
        LOOM_LOCATION_UNKNOWN, &integer_constant_op_));
    IREE_ASSERT_OK(
        loom_scalar_constant_build(&body_builder, loom_attr_f64(1.5),
                                   loom_type_scalar(LOOM_SCALAR_TYPE_F32),
                                   LOOM_LOCATION_UNKNOWN, &float_constant_op_));
    const loom_value_id_t result = loom_scalar_muli_result(muli_op_);
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
  loom_op_t* muli_op_ = nullptr;
  loom_op_t* variadic_result_op_ = nullptr;
  loom_op_t* integer_constant_op_ = nullptr;
  loom_op_t* float_constant_op_ = nullptr;
  loom_op_t* direct_extf_op_ = nullptr;
  loom_op_t* fptrunc_op_ = nullptr;
  loom_op_t* roundtrip_extf_op_ = nullptr;
  loom_op_t* vector_consumer_op_ = nullptr;
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

TEST_F(LowLowerRuleValueTest, ResolvesVariadicResultElements) {
  loom_low_lower_value_ref_t value_refs[2] = {};
  value_refs[0].kind = LOOM_LOW_LOWER_VALUE_REF_RESULT;
  value_refs[0].index = 0;
  value_refs[0].element_index = 0;
  value_refs[1].kind = LOOM_LOW_LOWER_VALUE_REF_RESULT;
  value_refs[1].index = 0;
  value_refs[1].element_index = 1;
  loom_low_lower_rule_set_t rule_set = {};
  rule_set.value_refs = value_refs;
  rule_set.value_ref_count = IREE_ARRAYSIZE(value_refs);

  const loom_value_slice_t results =
      loom_scalar_assume_results(variadic_result_op_);
  ASSERT_EQ(results.count, 2u);
  EXPECT_EQ(loom_low_lower_rule_source_value(module_, &rule_set,
                                             variadic_result_op_, 0),
            results.values[0]);
  EXPECT_EQ(loom_low_lower_rule_source_value(module_, &rule_set,
                                             variadic_result_op_, 1),
            results.values[1]);

  const loom_value_slice_t result_field =
      loom_low_lower_rule_value_ref_field_span(module_, &rule_set,
                                               variadic_result_op_, 1);
  ASSERT_EQ(result_field.count, 2u);
  EXPECT_EQ(result_field.values[0], results.values[0]);
  EXPECT_EQ(result_field.values[1], results.values[1]);
}

TEST_F(LowLowerRuleValueTest, ResolvesValuesAcrossSourceGraphNodes) {
  const loom_low_lower_value_ref_t value_refs[] = {
      {
          .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
          .source_node_index = 0,
          .index = 0,
      },
      {
          .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
          .source_node_index = 1,
          .index = 1,
      },
      {
          .kind = LOOM_LOW_LOWER_VALUE_REF_RESULT,
          .source_node_index = 1,
          .index = 0,
      },
  };
  loom_low_lower_rule_set_t rule_set = {};
  rule_set.value_refs = value_refs;
  rule_set.value_ref_count = IREE_ARRAYSIZE(value_refs);
  const loom_op_t* source_nodes[] = {addi_op_, muli_op_};

  EXPECT_EQ(loom_low_lower_rule_source_value_from_nodes(
                module_, &rule_set, addi_op_, source_nodes,
                IREE_ARRAYSIZE(source_nodes), 0),
            arguments_[0]);
  EXPECT_EQ(loom_low_lower_rule_source_value_from_nodes(
                module_, &rule_set, addi_op_, source_nodes,
                IREE_ARRAYSIZE(source_nodes), 1),
            arguments_[1]);
  EXPECT_EQ(loom_low_lower_rule_source_value_from_nodes(
                module_, &rule_set, addi_op_, source_nodes,
                IREE_ARRAYSIZE(source_nodes), 2),
            loom_scalar_muli_result(muli_op_));

  const loom_value_slice_t result_field =
      loom_low_lower_rule_value_ref_field_span_from_nodes(
          module_, &rule_set, addi_op_, source_nodes,
          IREE_ARRAYSIZE(source_nodes), 2);
  ASSERT_EQ(result_field.count, 1u);
  EXPECT_EQ(result_field.values[0], loom_scalar_muli_result(muli_op_));
}

TEST_F(LowLowerRuleValueTest,
       ExactLaneOriginsStopAtLossyConversionsAndResolveWholeOperands) {
  IREE_ASSERT_OK(
      loom_value_fact_table_compute_op(&fact_table_, module_, direct_extf_op_));
  loom_value_fact_exact_lane_origin_t direct_origin = {};
  ASSERT_TRUE(loom_value_fact_table_query_exact_lane_origin(
      &fact_table_, module_, loom_vector_extf_result(direct_extf_op_),
      &direct_origin));
  EXPECT_EQ(direct_origin.source_value_id, arguments_[2]);

  IREE_ASSERT_OK(
      loom_value_fact_table_compute_op(&fact_table_, module_, fptrunc_op_));
  IREE_ASSERT_OK(loom_value_fact_table_compute_op(&fact_table_, module_,
                                                  roundtrip_extf_op_));
  const loom_value_id_t truncated = loom_vector_fptrunc_result(fptrunc_op_);
  const loom_value_id_t extended = loom_vector_extf_result(roundtrip_extf_op_);
  loom_value_fact_exact_lane_origin_t exact_origin = {};
  EXPECT_FALSE(loom_value_fact_table_query_exact_lane_origin(
      &fact_table_, module_, truncated, &exact_origin));
  ASSERT_TRUE(loom_value_fact_table_query_exact_lane_origin(
      &fact_table_, module_, extended, &exact_origin));
  EXPECT_EQ(exact_origin.source_value_id, truncated);

  loom_value_fact_static_lane_origin_t provenance_origin = {};
  ASSERT_TRUE(loom_value_fact_table_query_static_lane_origin(
      &fact_table_, module_, extended, &provenance_origin));
  EXPECT_EQ(provenance_origin.source_value_id, arguments_[3]);

  const loom_low_lower_value_ref_t value_ref = {
      .kind = LOOM_LOW_LOWER_VALUE_REF_EXACT_LANE_ORIGIN_OPERAND,
      .source_node_index = 0,
      .index = 0,
  };
  loom_low_lower_rule_set_t rule_set = {.value_refs = &value_ref,
                                        .value_ref_count = 1};
  loom_value_id_t resolved = LOOM_VALUE_ID_INVALID;
  ASSERT_TRUE(loom_low_lower_rule_resolve_source_value_from_nodes(
      module_, &fact_table_, (loom_target_contract_vector_lane_projection_t){0},
      &rule_set, vector_consumer_op_,
      /*source_nodes=*/nullptr, /*source_node_count=*/1,
      /*value_ref_index=*/0, &resolved));
  EXPECT_EQ(resolved, truncated);
}

TEST_F(LowLowerRuleValueTest, ResolvesUniformElementOrigins) {
  const loom_value_id_t aggregate = loom_vector_extf_result(roundtrip_extf_op_);
  const loom_value_id_t scalar =
      loom_scalar_constant_result(float_constant_op_);
  IREE_ASSERT_OK(loom_value_fact_table_define_uniform_element_origin(
      &fact_table_, aggregate, scalar, scalar));

  for (auto kind :
       {LOOM_LOW_LOWER_VALUE_REF_UNIFORM_ELEMENT_ORIGIN_OPERAND,
        LOOM_LOW_LOWER_VALUE_REF_EXACT_UNIFORM_ELEMENT_ORIGIN_OPERAND}) {
    const loom_low_lower_value_ref_t value_ref = {
        .kind = static_cast<uint8_t>(kind),
        .source_node_index = 0,
        .index = 0,
    };
    loom_low_lower_rule_set_t rule_set = {.value_refs = &value_ref,
                                          .value_ref_count = 1};
    loom_value_id_t resolved = LOOM_VALUE_ID_INVALID;
    ASSERT_TRUE(loom_low_lower_rule_resolve_source_value_from_nodes(
        module_, &fact_table_,
        (loom_target_contract_vector_lane_projection_t){0}, &rule_set,
        vector_consumer_op_,
        /*source_nodes=*/nullptr, /*source_node_count=*/1,
        /*value_ref_index=*/0, &resolved));
    EXPECT_EQ(resolved, scalar);
  }
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

TEST_F(LowLowerRuleValueTest, ProjectsUnsignedNumeratorBounds) {
  const loom_value_id_t numerator = arguments_[0];
  for (uint32_t width : {32u, 64u}) {
    const uint64_t mask = UINT64_MAX >> (64 - width);
    EXPECT_EQ(
        loom_low_lower_unsigned_numerator_maximum(nullptr, numerator, width),
        mask);
    const struct {
      // Inclusive signed source range minimum.
      int64_t lower;
      // Inclusive signed source range maximum.
      int64_t upper;
      // Expected maximum after interpreting the source at the native width.
      uint64_t maximum;
    } cases[] = {
        {INT64_MIN, INT64_MAX, mask},
        {-1, 255, mask},
        {0, INT64_MAX, width == 32 ? mask : uint64_t{INT64_MAX}},
        {0, 0, 0},
        {0, 255, 255},
        {17, 65535, 65535},
        {-128, -2, mask - 1},
    };
    for (const auto& test : cases) {
      IREE_ASSERT_OK(loom_value_fact_table_define(
          &fact_table_, numerator,
          loom_value_facts_make(test.lower, test.upper, 1)));
      EXPECT_EQ(loom_low_lower_unsigned_numerator_maximum(&fact_table_,
                                                          numerator, width),
                test.maximum);
    }
  }
}

TEST_F(LowLowerRuleValueTest, DerivesBoundedUnsignedDivisorRecipe) {
  const loom_value_id_t divisor =
      loom_scalar_constant_result(integer_constant_op_);
  IREE_ASSERT_OK(loom_value_fact_table_define(&fact_table_, divisor,
                                              loom_value_facts_exact_i64(7)));
  for (int64_t maximum :
       {int64_t{0}, int64_t{6}, int64_t{255}, int64_t{INT32_MAX}}) {
    IREE_ASSERT_OK(loom_value_fact_table_define(
        &fact_table_, arguments_[0], loom_value_facts_make(0, maximum, 1)));
    loom_low_lower_unsigned_divisor_magic_info_t info = {};
    ASSERT_TRUE(loom_low_lower_rule_value_facts_u32_divisor_magic_info(
        module_, &fact_table_, arguments_[0], divisor, &info));
    const auto expected =
        loom_low_lower_unsigned_divisor_magic_info(7, 32, maximum);
    EXPECT_EQ(info.multiplier, expected.multiplier);
    EXPECT_EQ(info.post_shift, expected.post_shift);
    EXPECT_FALSE(info.is_add);
  }
}

TEST_F(LowLowerRuleValueTest, DerivesExactUnsignedDivisorRecipes) {
  const loom_value_id_t value_id =
      loom_scalar_constant_result(integer_constant_op_);
  const uint32_t divisors[] = {2u,
                               3u,
                               5u,
                               7u,
                               10u,
                               31u,
                               UINT32_C(0x7fffffff),
                               UINT32_C(0x80000000),
                               UINT32_C(0x80000001),
                               UINT32_MAX};
  const uint32_t numerators[] = {
      0u, 1u, 2u, 6u, 7u, 8u, UINT32_C(0x7fffffff), UINT32_MAX};
  for (uint32_t divisor : divisors) {
    IREE_ASSERT_OK(loom_value_fact_table_define(
        &fact_table_, value_id, loom_value_facts_exact_i64(divisor)));
    loom_low_lower_unsigned_divisor_magic_info_t info = {};
    ASSERT_TRUE(loom_low_lower_rule_value_facts_u32_divisor_magic_info(
        module_, &fact_table_, arguments_[0], value_id, &info));
    const uint64_t high_multiplier =
        loom_low_lower_u32_divisor_reciprocal(divisor);
    auto check_quotient = [&](uint32_t numerator) {
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
      uint64_t high = 0;
      uint64_t low = 0;
      iree_math_mul_u64_to_u128(numerator, high_multiplier, &high, &low);
      EXPECT_EQ(high, numerator / divisor)
          << "numerator=" << numerator << " divisor=" << divisor;
      iree_math_mul_u64_to_u128(low, divisor, &high, &low);
      EXPECT_EQ(high, numerator % divisor)
          << "numerator=" << numerator << " divisor=" << divisor;
    };
    for (uint32_t numerator : numerators) {
      check_quotient(numerator);
    }
    for (uint32_t quotient : {1u, UINT32_MAX / divisor}) {
      const uint32_t multiple = quotient * divisor;
      check_quotient(multiple - 1);
      check_quotient(multiple);
      if (multiple != UINT32_MAX) {
        check_quotient(multiple + 1);
      }
    }
    uint32_t random = 0x6c52a591u;
    for (uint32_t i = 0; i < 1024; ++i) {
      random = random * 1664525u + 1013904223u;
      check_quotient(random);
    }
  }
}

TEST(U32DivisorReciprocalTest, PreservesQuotientAndRemainder) {
  uint32_t random = 0x6c52a591u;
  for (uint32_t sample = 2; sample < 131072; ++sample) {
    random = random * 1664525u + 1013904223u;
    const uint32_t divisor = sample < 65536 ? sample : (random | 2u);
    const uint64_t reciprocal = loom_low_lower_u32_divisor_reciprocal(divisor);
    const uint32_t multiple = (UINT32_MAX / divisor) * divisor;
    const uint32_t numerators[] = {0,        1,           divisor - 1,
                                   divisor,  divisor + 1, multiple - 1,
                                   multiple, UINT32_MAX,  random};
    for (uint32_t numerator : numerators) {
      uint64_t high = 0, low = 0;
      iree_math_mul_u64_to_u128(numerator, reciprocal, &high, &low);
      ASSERT_EQ(high, numerator / divisor)
          << "numerator=" << numerator << " divisor=" << divisor;
      iree_math_mul_u64_to_u128(low, divisor, &high, &low);
      ASSERT_EQ(high, numerator % divisor)
          << "numerator=" << numerator << " divisor=" << divisor;
    }
  }
}

static uint64_t ApplyUnsignedDivisionRecipe(
    uint64_t numerator, uint32_t bit_width,
    loom_low_lower_unsigned_divisor_magic_info_t info) {
  uint64_t high = 0, low = 0;
  iree_math_mul_u64_to_u128(numerator, info.multiplier, &high, &low);
  uint64_t quotient =
      bit_width == 64 ? high : (low >> bit_width) | (high << (64 - bit_width));
  if (info.is_add) {
    quotient = ((numerator - quotient) >> 1) + quotient;
  }
  return quotient >> info.post_shift;
}

TEST(UnsignedDivisorMagicTest, ExactAcrossWidthsAndUnsignedBoundaries) {
  uint64_t random = UINT64_C(0x362b48f137159da3);
  for (uint32_t bit_width = 2; bit_width <= 64; ++bit_width) {
    const uint64_t mask = UINT64_MAX >> (64 - bit_width);
    const uint64_t half = UINT64_C(1) << (bit_width - 1);
    auto check_divisor = [&](uint64_t divisor) {
      const auto info =
          loom_low_lower_unsigned_divisor_magic_info(divisor, bit_width, mask);
      auto check_numerator = [&](uint64_t numerator) {
        const uint64_t quotient =
            ApplyUnsignedDivisionRecipe(numerator, bit_width, info);
        ASSERT_EQ(quotient, numerator / divisor)
            << "width=" << bit_width << " numerator=" << numerator
            << " divisor=" << divisor;
        ASSERT_EQ(numerator - quotient * divisor, numerator % divisor)
            << "width=" << bit_width << " numerator=" << numerator
            << " divisor=" << divisor;
      };
      if (bit_width <= 8) {
        for (uint64_t numerator = 0; numerator <= mask; ++numerator) {
          check_numerator(numerator);
        }
      } else {
        const uint64_t multiple = (mask / divisor) * divisor;
        for (uint64_t numerator :
             {UINT64_C(0), UINT64_C(1), divisor - 1, divisor,
              (divisor + 1) & mask, multiple - 1, multiple,
              (multiple + 1) & mask, half - 1, half, half + 1, mask}) {
          check_numerator(numerator);
        }
        for (uint32_t sample = 0; sample < 32; ++sample) {
          random ^= random << 13;
          random ^= random >> 7;
          random ^= random << 17;
          check_numerator(random & mask);
        }
      }
    };
    const uint64_t exhaustive_limit = bit_width <= 8 ? mask : 256;
    for (uint64_t divisor = 2; divisor <= exhaustive_limit; ++divisor) {
      check_divisor(divisor);
    }
    for (uint32_t sample = 0; sample < 256; ++sample) {
      random ^= random << 13;
      random ^= random >> 7;
      random ^= random << 17;
      check_divisor((random & mask) | 2);
    }
    for (uint64_t divisor : {half, half + 1, mask}) {
      check_divisor(divisor);
    }
  }
}

TEST(UnsignedDivisorMagicTest, ExactForEverySmallNumeratorBound) {
  for (uint32_t bit_width = 2; bit_width <= 7; ++bit_width) {
    const uint64_t mask = UINT64_MAX >> (64 - bit_width);
    for (uint64_t divisor = 2; divisor <= mask; ++divisor) {
      for (uint64_t maximum = 0; maximum <= mask; ++maximum) {
        const auto info = loom_low_lower_unsigned_divisor_magic_info(
            divisor, bit_width, maximum);
        for (uint64_t numerator = 0; numerator <= maximum; ++numerator) {
          const uint64_t quotient =
              ApplyUnsignedDivisionRecipe(numerator, bit_width, info);
          ASSERT_EQ(quotient, numerator / divisor)
              << "width=" << bit_width << " divisor=" << divisor
              << " maximum=" << maximum << " numerator=" << numerator;
          ASSERT_EQ(numerator - quotient * divisor, numerator % divisor);
        }
      }
    }
  }
}

TEST(UnsignedDivisorMagicTest, ExactForWideNumeratorBounds) {
  uint64_t random = UINT64_C(0x6351a39b72ce814d);
  for (uint32_t bit_width = 8; bit_width <= 64; ++bit_width) {
    const uint64_t mask = UINT64_MAX >> (64 - bit_width);
    const uint64_t half = UINT64_C(1) << (bit_width - 1);
    for (uint32_t sample = 0; sample < 128; ++sample) {
      random ^= random << 13;
      random ^= random >> 7;
      random ^= random << 17;
      const uint64_t divisor = sample < 64 ? sample + 2 : (random & mask) | 2;
      for (uint64_t maximum : {UINT64_C(0), UINT64_C(1), divisor - 1, divisor,
                               half - 1, half, half + 1, mask, random & mask}) {
        const auto info = loom_low_lower_unsigned_divisor_magic_info(
            divisor, bit_width, maximum);
        const uint64_t multiple = (maximum / divisor) * divisor;
        auto check_numerator = [&](uint64_t numerator) {
          if (numerator > maximum) {
            return;
          }
          const uint64_t quotient =
              ApplyUnsignedDivisionRecipe(numerator, bit_width, info);
          ASSERT_EQ(quotient, numerator / divisor)
              << "width=" << bit_width << " divisor=" << divisor
              << " maximum=" << maximum << " numerator=" << numerator;
          ASSERT_EQ(numerator - quotient * divisor, numerator % divisor);
        };
        for (uint64_t numerator :
             {UINT64_C(0), UINT64_C(1), divisor - 1, divisor,
              (divisor + 1) & mask, (multiple - 1) & mask, multiple,
              (multiple + 1) & mask, maximum}) {
          check_numerator(numerator);
        }
        for (uint32_t i = 0; i < 16; ++i) {
          random ^= random << 13;
          random ^= random >> 7;
          random ^= random << 17;
          check_numerator(random & maximum);
        }
      }
    }
  }
}

TEST(UnsignedDivisorMagicTest, NumeratorBoundRemovesAddAdjustment) {
  for (uint32_t bit_width : {32u, 64u}) {
    const uint64_t mask = UINT64_MAX >> (64 - bit_width);
    const auto unbounded =
        loom_low_lower_unsigned_divisor_magic_info(7, bit_width, mask);
    const auto bounded =
        loom_low_lower_unsigned_divisor_magic_info(7, bit_width, mask >> 1);
    EXPECT_TRUE(unbounded.is_add);
    EXPECT_FALSE(bounded.is_add);
  }
}

}  // namespace
}  // namespace loom
