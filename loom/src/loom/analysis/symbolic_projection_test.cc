// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_projection.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

int64_t Evaluate(const loom_symbolic_projection_t& projection, int64_t value) {
  const int64_t quotient =
      (projection.scale * value + projection.offset) / projection.divisor;
  return projection.modulus ? quotient % projection.modulus : quotient;
}

TEST(SymbolicProjectionTest, CompositionPreservesConcreteValues) {
  for (int64_t scale : {1, 3}) {
    for (int64_t offset : {0, 1, 7, 17}) {
      for (int64_t divisor : {1, 2, 3, 8}) {
        for (int64_t modulus : {0, 4, 6, 16}) {
          const loom_symbolic_projection_t input = {0, scale, offset, divisor,
                                                    modulus};
          for (int64_t outer : {1, 2, 3, 4, 8}) {
            auto divided = input;
            auto remainder = input;
            const bool has_division =
                loom_symbolic_projection_divide(&divided, outer, &divided);
            const bool has_remainder = loom_symbolic_projection_remainder(
                &remainder, outer, &remainder);
            if (modulus == 0 || (modulus == 16 && outer == 4)) {
              ASSERT_TRUE(has_division);
              ASSERT_TRUE(has_remainder);
            }
            for (int64_t value = 0; value < 128; ++value) {
              const int64_t original = Evaluate(input, value);
              if (has_division) {
                ASSERT_EQ(Evaluate(divided, value), original / outer);
              }
              if (has_remainder) {
                ASSERT_EQ(Evaluate(remainder, value), original % outer);
              }
            }
          }
        }
      }
    }
  }
}

TEST(SymbolicProjectionTest, RejectsNonDivisibleAndOverflowingComposition) {
  // At x=6, (x%6)%4 differs from x%4. At x=4, (x%6)/4 differs
  // from (x/4)%1. Neither composition can discard the inner wrap.
  const loom_symbolic_projection_t input = {0, 1, 0, 1, 6};
  loom_symbolic_projection_t output = {};
  EXPECT_FALSE(loom_symbolic_projection_divide(&input, 4, &output));
  EXPECT_FALSE(loom_symbolic_projection_remainder(&input, 4, &output));
  const loom_symbolic_projection_t large = {0, 1, 0, INT64_MAX, 0};
  EXPECT_FALSE(loom_symbolic_projection_divide(&large, 2, &output));
}

TEST(SymbolicProjectionTest, DivisibilityRefinementPreservesConcreteValues) {
  for (int64_t multiple : {1, 2, 3, 8, 12}) {
    for (int64_t scale : {1, 2, 3, 8}) {
      for (int64_t offset : {0, 1, 7, 11, 17}) {
        for (int64_t divisor : {1, 2, 3, 8, 12}) {
          for (int64_t modulus : {0, 3, 8}) {
            const loom_symbolic_projection_t original = {0, scale, offset,
                                                         divisor, modulus};
            auto refined = original;
            loom_symbolic_projection_refine(multiple, &refined);
            for (int64_t value = 0; value < 256; value += multiple) {
              ASSERT_EQ(Evaluate(refined, value), Evaluate(original, value));
            }
          }
        }
      }
    }
  }
  loom_symbolic_projection_t full_tiles = {0, 1, 7, 8, 0};
  loom_symbolic_projection_refine(8, &full_tiles);
  EXPECT_EQ(full_tiles.offset, 0);
  loom_symbolic_projection_t partial_tiles = {0, 1, 7, 8, 0};
  loom_symbolic_projection_refine(4, &partial_tiles);
  EXPECT_EQ(partial_tiles.offset, 4);

  // Computing scale * known_divisor is unnecessary and could overflow.
  loom_symbolic_projection_t large = {0, INT64_MAX, 7, 8, 0};
  loom_symbolic_projection_refine(8, &large);
  EXPECT_EQ(large.offset, 0);
}

}  // namespace
}  // namespace loom
