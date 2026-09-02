// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/attribute.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(AttributeTest, PredicateValueTypeContracts) {
  static constexpr loom_predicate_kind_t kIntegerPredicateKinds[] = {
      LOOM_PREDICATE_EQ,   LOOM_PREDICATE_NE,    LOOM_PREDICATE_LT,
      LOOM_PREDICATE_LE,   LOOM_PREDICATE_GT,    LOOM_PREDICATE_GE,
      LOOM_PREDICATE_MUL,  LOOM_PREDICATE_MIN,   LOOM_PREDICATE_MAX,
      LOOM_PREDICATE_POW2, LOOM_PREDICATE_RANGE,
  };
  static constexpr loom_predicate_kind_t kFloatPredicateKinds[] = {
      LOOM_PREDICATE_NOT_NAN,
      LOOM_PREDICATE_NOT_INF,
      LOOM_PREDICATE_FINITE,
  };
  const loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const loom_type_t offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  const loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  const loom_type_t buffer = loom_type_buffer();

  for (loom_predicate_kind_t kind : kIntegerPredicateKinds) {
    EXPECT_TRUE(loom_predicate_kind_accepts_value_type(kind, i32));
    EXPECT_TRUE(loom_predicate_kind_accepts_value_type(kind, index));
    EXPECT_TRUE(loom_predicate_kind_accepts_value_type(kind, offset));
    EXPECT_FALSE(loom_predicate_kind_accepts_value_type(kind, f32));
    EXPECT_FALSE(loom_predicate_kind_accepts_value_type(kind, buffer));
  }
  for (loom_predicate_kind_t kind : kFloatPredicateKinds) {
    EXPECT_FALSE(loom_predicate_kind_accepts_value_type(kind, i32));
    EXPECT_TRUE(loom_predicate_kind_accepts_value_type(kind, f32));
    EXPECT_FALSE(loom_predicate_kind_accepts_value_type(kind, buffer));
  }
  EXPECT_FALSE(
      loom_predicate_kind_accepts_value_type(LOOM_PREDICATE_COUNT_, i32));
  EXPECT_FALSE(loom_predicate_kind_accepts_value_type(UINT8_MAX, i32));
}

TEST(AttributeTest, PredicateValueTypeContractsUseTypedRegisterSemantics) {
  const loom_register_type_data_t integer_data = {
      /*.carrier_payload0=*/0,
      /*.carrier_payload1=*/0,
      /*.value_type=*/loom_type_scalar(LOOM_SCALAR_TYPE_I32),
  };
  const loom_register_type_data_t float_data = {
      /*.carrier_payload0=*/0,
      /*.carrier_payload1=*/0,
      /*.value_type=*/loom_type_scalar(LOOM_SCALAR_TYPE_F32),
  };
  const loom_type_t integer_register =
      loom_type_register_payload_with_value_type(&integer_data);
  const loom_type_t float_register =
      loom_type_register_payload_with_value_type(&float_data);
  const loom_type_t untyped_register = loom_type_register_payload(0, 0);

  EXPECT_TRUE(loom_predicate_kind_accepts_value_type(LOOM_PREDICATE_EQ,
                                                     integer_register));
  EXPECT_FALSE(loom_predicate_kind_accepts_value_type(LOOM_PREDICATE_FINITE,
                                                      integer_register));
  EXPECT_FALSE(loom_predicate_kind_accepts_value_type(LOOM_PREDICATE_EQ,
                                                      float_register));
  EXPECT_TRUE(loom_predicate_kind_accepts_value_type(LOOM_PREDICATE_FINITE,
                                                     float_register));
  EXPECT_FALSE(loom_predicate_kind_accepts_value_type(LOOM_PREDICATE_EQ,
                                                      untyped_register));
}

}  // namespace
}  // namespace loom
