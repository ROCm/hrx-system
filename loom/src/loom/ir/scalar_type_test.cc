// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/scalar_type.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(ScalarTypeTest, RoundTripNames) {
  for (int i = 0; i < LOOM_SCALAR_TYPE_COUNT_; ++i) {
    loom_scalar_type_t scalar_type = (loom_scalar_type_t)i;
    const char* name = loom_scalar_type_name(scalar_type);
    ASSERT_NE(name, nullptr) << i;

    loom_scalar_type_t parsed = LOOM_SCALAR_TYPE_COUNT_;
    ASSERT_TRUE(loom_scalar_type_parse(iree_make_cstring_view(name), &parsed))
        << name;
    EXPECT_EQ(parsed, scalar_type) << name;
    EXPECT_GT(loom_scalar_type_bitwidth(scalar_type), 0) << name;
  }
}

TEST(ScalarTypeTest, InvalidValues) {
  EXPECT_EQ(loom_scalar_type_name((loom_scalar_type_t)-1), nullptr);
  EXPECT_EQ(loom_scalar_type_name(LOOM_SCALAR_TYPE_COUNT_), nullptr);
  EXPECT_EQ(loom_scalar_type_bitwidth((loom_scalar_type_t)-1), 0);
  EXPECT_EQ(loom_scalar_type_bitwidth(LOOM_SCALAR_TYPE_COUNT_), 0);

  loom_scalar_type_t parsed = LOOM_SCALAR_TYPE_I32;
  EXPECT_FALSE(loom_scalar_type_parse(IREE_SV("not_a_scalar"), &parsed));
  EXPECT_EQ(parsed, LOOM_SCALAR_TYPE_I32);
}

TEST(ScalarTypeTest, AddressTypesAreTargetWidth) {
  EXPECT_EQ(loom_scalar_type_bitwidth(LOOM_SCALAR_TYPE_INDEX), 64);
  EXPECT_EQ(loom_scalar_type_bitwidth(LOOM_SCALAR_TYPE_OFFSET), 64);
}

}  // namespace
}  // namespace loom
