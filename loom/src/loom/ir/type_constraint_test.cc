// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/type_constraint.h"

#include "iree/testing/gtest.h"
#include "loom/ir/scalar_type.h"

namespace {

TEST(TypeConstraintTest, PayloadScalar) {
  const loom_scalar_type_t accepted_types[] = {
      LOOM_SCALAR_TYPE_I1,     LOOM_SCALAR_TYPE_I8,  LOOM_SCALAR_TYPE_I16,
      LOOM_SCALAR_TYPE_I32,    LOOM_SCALAR_TYPE_I64, LOOM_SCALAR_TYPE_F8E4M3,
      LOOM_SCALAR_TYPE_F8E5M2, LOOM_SCALAR_TYPE_F16, LOOM_SCALAR_TYPE_BF16,
      LOOM_SCALAR_TYPE_F32,    LOOM_SCALAR_TYPE_F64,
  };
  for (loom_scalar_type_t type : accepted_types) {
    EXPECT_TRUE(loom_type_satisfies_constraint(
        loom_type_scalar(type), LOOM_TYPE_CONSTRAINT_PAYLOAD_SCALAR));
  }

  const loom_scalar_type_t rejected_types[] = {
      LOOM_SCALAR_TYPE_INDEX,
      LOOM_SCALAR_TYPE_OFFSET,
  };
  for (loom_scalar_type_t type : rejected_types) {
    EXPECT_FALSE(loom_type_satisfies_constraint(
        loom_type_scalar(type), LOOM_TYPE_CONSTRAINT_PAYLOAD_SCALAR));
  }

  loom_type_t vector_i32 = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(1), 0);
  EXPECT_FALSE(loom_type_satisfies_constraint(
      vector_i32, LOOM_TYPE_CONSTRAINT_PAYLOAD_SCALAR));
}

TEST(TypeConstraintTest, BytePatternScalar) {
  const loom_scalar_type_t accepted_types[] = {
      LOOM_SCALAR_TYPE_I8,  LOOM_SCALAR_TYPE_I16,    LOOM_SCALAR_TYPE_I32,
      LOOM_SCALAR_TYPE_I64, LOOM_SCALAR_TYPE_F8E4M3, LOOM_SCALAR_TYPE_F8E5M2,
      LOOM_SCALAR_TYPE_F16, LOOM_SCALAR_TYPE_BF16,   LOOM_SCALAR_TYPE_F32,
      LOOM_SCALAR_TYPE_F64,
  };
  for (loom_scalar_type_t type : accepted_types) {
    EXPECT_TRUE(loom_type_satisfies_constraint(
        loom_type_scalar(type), LOOM_TYPE_CONSTRAINT_BYTE_PATTERN_SCALAR));
  }

  const loom_scalar_type_t rejected_types[] = {
      LOOM_SCALAR_TYPE_INDEX,
      LOOM_SCALAR_TYPE_OFFSET,
      LOOM_SCALAR_TYPE_I1,
  };
  for (loom_scalar_type_t type : rejected_types) {
    EXPECT_FALSE(loom_type_satisfies_constraint(
        loom_type_scalar(type), LOOM_TYPE_CONSTRAINT_BYTE_PATTERN_SCALAR));
  }
}

TEST(TypeConstraintTest, ExactI32) {
  EXPECT_TRUE(loom_type_satisfies_constraint(
      loom_type_scalar(LOOM_SCALAR_TYPE_I32), LOOM_TYPE_CONSTRAINT_I32));
  EXPECT_FALSE(loom_type_satisfies_constraint(
      loom_type_scalar(LOOM_SCALAR_TYPE_I64), LOOM_TYPE_CONSTRAINT_I32));
}

}  // namespace
