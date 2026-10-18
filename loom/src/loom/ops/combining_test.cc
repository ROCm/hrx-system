// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/combining.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "iree/testing/gtest.h"

namespace {

struct IntegerIdentityCase {
  // Combining operation under test.
  loom_combining_kind_t kind;
  // Expected i1 identity bit pattern.
  int64_t i1;
  // Expected i8 identity bit pattern.
  int64_t i8;
  // Expected i16 identity bit pattern.
  int64_t i16;
  // Expected i32 identity bit pattern.
  int64_t i32;
  // Expected i64 identity bit pattern.
  int64_t i64;
};

TEST(CombiningTest, IntegerIdentitiesCoverEveryKindAndWidth) {
  constexpr IntegerIdentityCase kCases[] = {
      {LOOM_COMBINING_KIND_ADDI, 0, 0, 0, 0, 0},
      {LOOM_COMBINING_KIND_MULI, 1, 1, 1, 1, 1},
      {LOOM_COMBINING_KIND_MINSI, 0, INT8_MAX, INT16_MAX, INT32_MAX, INT64_MAX},
      {LOOM_COMBINING_KIND_MAXSI, 1, INT8_MIN, INT16_MIN, INT32_MIN, INT64_MIN},
      {LOOM_COMBINING_KIND_MINUI, 1, -1, -1, -1, -1},
      {LOOM_COMBINING_KIND_MAXUI, 0, 0, 0, 0, 0},
      {LOOM_COMBINING_KIND_ANDI, 1, -1, -1, -1, -1},
      {LOOM_COMBINING_KIND_ORI, 0, 0, 0, 0, 0},
      {LOOM_COMBINING_KIND_XORI, 0, 0, 0, 0, 0},
  };
  constexpr loom_scalar_type_t kTypes[] = {
      LOOM_SCALAR_TYPE_I1,  LOOM_SCALAR_TYPE_I8,  LOOM_SCALAR_TYPE_I16,
      LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I64,
  };

  for (const auto& test_case : kCases) {
    const int64_t expected[] = {test_case.i1, test_case.i8, test_case.i16,
                                test_case.i32, test_case.i64};
    for (size_t i = 0; i < IREE_ARRAYSIZE(kTypes); ++i) {
      int64_t identity = 0;
      ASSERT_TRUE(loom_combining_kind_integer_identity(test_case.kind,
                                                       kTypes[i], &identity));
      EXPECT_EQ(identity, expected[i]);
    }
  }
}

TEST(CombiningTest, IntegerAlgebraicPropertiesCoverEveryKind) {
  for (uint32_t ordinal = 0; ordinal < LOOM_COMBINING_KIND_COUNT_; ++ordinal) {
    const auto kind = static_cast<loom_combining_kind_t>(ordinal);
    EXPECT_NE(loom_combining_kind_accepts_integer(kind),
              loom_combining_kind_accepts_float(kind));
    int64_t identity = 0;
    EXPECT_EQ(loom_combining_kind_integer_identity(kind, LOOM_SCALAR_TYPE_I32,
                                                   &identity),
              loom_combining_kind_accepts_integer(kind));
  }

  EXPECT_FALSE(loom_combining_kind_is_idempotent(LOOM_COMBINING_KIND_ADDI));
  EXPECT_FALSE(loom_combining_kind_is_idempotent(LOOM_COMBINING_KIND_MULI));
  EXPECT_TRUE(loom_combining_kind_is_idempotent(LOOM_COMBINING_KIND_MINSI));
  EXPECT_TRUE(loom_combining_kind_is_idempotent(LOOM_COMBINING_KIND_MAXSI));
  EXPECT_TRUE(loom_combining_kind_is_idempotent(LOOM_COMBINING_KIND_MINUI));
  EXPECT_TRUE(loom_combining_kind_is_idempotent(LOOM_COMBINING_KIND_MAXUI));
  EXPECT_TRUE(loom_combining_kind_is_idempotent(LOOM_COMBINING_KIND_ANDI));
  EXPECT_TRUE(loom_combining_kind_is_idempotent(LOOM_COMBINING_KIND_ORI));
  EXPECT_FALSE(loom_combining_kind_is_idempotent(LOOM_COMBINING_KIND_XORI));
  EXPECT_FALSE(loom_combining_kind_is_idempotent(LOOM_COMBINING_KIND_ADDF));
  EXPECT_FALSE(loom_combining_kind_is_idempotent(LOOM_COMBINING_KIND_MULF));
  EXPECT_TRUE(loom_combining_kind_is_idempotent(LOOM_COMBINING_KIND_MINIMUMF));
  EXPECT_TRUE(loom_combining_kind_is_idempotent(LOOM_COMBINING_KIND_MAXIMUMF));
  EXPECT_TRUE(loom_combining_kind_is_idempotent(LOOM_COMBINING_KIND_MINNUMF));
  EXPECT_TRUE(loom_combining_kind_is_idempotent(LOOM_COMBINING_KIND_MAXNUMF));
}

TEST(CombiningTest, I1EquivalentsCoverEveryIntegerKind) {
  constexpr struct {
    loom_combining_kind_t kind;
    loom_combining_kind_t equivalent;
  } kCases[] = {
      {LOOM_COMBINING_KIND_ADDI, LOOM_COMBINING_KIND_XORI},
      {LOOM_COMBINING_KIND_MULI, LOOM_COMBINING_KIND_ANDI},
      {LOOM_COMBINING_KIND_MINSI, LOOM_COMBINING_KIND_ORI},
      {LOOM_COMBINING_KIND_MAXSI, LOOM_COMBINING_KIND_ANDI},
      {LOOM_COMBINING_KIND_MINUI, LOOM_COMBINING_KIND_ANDI},
      {LOOM_COMBINING_KIND_MAXUI, LOOM_COMBINING_KIND_ORI},
      {LOOM_COMBINING_KIND_ANDI, LOOM_COMBINING_KIND_ANDI},
      {LOOM_COMBINING_KIND_ORI, LOOM_COMBINING_KIND_ORI},
      {LOOM_COMBINING_KIND_XORI, LOOM_COMBINING_KIND_XORI},
  };
  for (const auto& test_case : kCases) {
    loom_combining_kind_t equivalent = LOOM_COMBINING_KIND_COUNT_;
    ASSERT_TRUE(loom_combining_kind_i1_equivalent(test_case.kind, &equivalent));
    EXPECT_EQ(equivalent, test_case.equivalent);
  }
  for (const loom_combining_kind_t kind :
       {LOOM_COMBINING_KIND_ADDF, LOOM_COMBINING_KIND_MULF,
        LOOM_COMBINING_KIND_MINIMUMF, LOOM_COMBINING_KIND_MAXIMUMF,
        LOOM_COMBINING_KIND_MINNUMF, LOOM_COMBINING_KIND_MAXNUMF}) {
    loom_combining_kind_t equivalent = LOOM_COMBINING_KIND_COUNT_;
    EXPECT_FALSE(loom_combining_kind_i1_equivalent(kind, &equivalent));
  }
}

TEST(CombiningTest, FloatingIdentitiesCoverArithmeticKinds) {
  double identity = 0.0;
  ASSERT_TRUE(
      loom_combining_kind_float_identity(LOOM_COMBINING_KIND_ADDF, &identity));
  EXPECT_EQ(identity, 0.0);
  EXPECT_TRUE(std::signbit(identity));

  ASSERT_TRUE(
      loom_combining_kind_float_identity(LOOM_COMBINING_KIND_MULF, &identity));
  EXPECT_EQ(identity, 1.0);

  for (const loom_combining_kind_t kind :
       {LOOM_COMBINING_KIND_MINIMUMF, LOOM_COMBINING_KIND_MAXIMUMF,
        LOOM_COMBINING_KIND_MINNUMF, LOOM_COMBINING_KIND_MAXNUMF}) {
    EXPECT_FALSE(loom_combining_kind_float_identity(kind, &identity));
  }
}

TEST(CombiningTest, FloatingScanSeedsCoverEveryKind) {
  double seed = 0.0;
  ASSERT_TRUE(
      loom_combining_kind_float_scan_seed(LOOM_COMBINING_KIND_ADDF, &seed));
  EXPECT_EQ(seed, 0.0);
  EXPECT_TRUE(std::signbit(seed));

  ASSERT_TRUE(
      loom_combining_kind_float_scan_seed(LOOM_COMBINING_KIND_MULF, &seed));
  EXPECT_EQ(seed, 1.0);

  for (const loom_combining_kind_t kind :
       {LOOM_COMBINING_KIND_MINIMUMF, LOOM_COMBINING_KIND_MINNUMF}) {
    ASSERT_TRUE(loom_combining_kind_float_scan_seed(kind, &seed));
    EXPECT_TRUE(std::isinf(seed));
    EXPECT_FALSE(std::signbit(seed));
  }
  for (const loom_combining_kind_t kind :
       {LOOM_COMBINING_KIND_MAXIMUMF, LOOM_COMBINING_KIND_MAXNUMF}) {
    ASSERT_TRUE(loom_combining_kind_float_scan_seed(kind, &seed));
    EXPECT_TRUE(std::isinf(seed));
    EXPECT_TRUE(std::signbit(seed));
  }

  for (const loom_combining_kind_t kind :
       {LOOM_COMBINING_KIND_ADDI, LOOM_COMBINING_KIND_MULI,
        LOOM_COMBINING_KIND_MINSI, LOOM_COMBINING_KIND_MAXSI,
        LOOM_COMBINING_KIND_MINUI, LOOM_COMBINING_KIND_MAXUI,
        LOOM_COMBINING_KIND_ANDI, LOOM_COMBINING_KIND_ORI,
        LOOM_COMBINING_KIND_XORI}) {
    EXPECT_FALSE(loom_combining_kind_float_scan_seed(kind, &seed));
  }
}

}  // namespace
