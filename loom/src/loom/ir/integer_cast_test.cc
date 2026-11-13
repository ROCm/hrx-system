// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/integer_cast.h"

#include <cstdint>

#include "iree/testing/gtest.h"

namespace loom {
namespace {

// Evaluate the payload bits independently of the normalized cast algebra and
// the compiler's interval transfer functions.
static uint64_t Interpret(uint64_t value, uint8_t width, bool is_unsigned) {
  const uint64_t mask = width == 64 ? UINT64_MAX : (UINT64_C(1) << width) - 1;
  value &= mask;
  return is_unsigned || !(value & (UINT64_C(1) << (width - 1))) ? value
                                                                : value | ~mask;
}

static uint64_t Evaluate(const loom_integer_cast_t& cast, uint64_t value) {
  value =
      Interpret(value, cast.bit_count, cast.kind == LOOM_INTEGER_CAST_UNSIGNED);
  return cast.kind == LOOM_INTEGER_CAST_SIGNED_THEN_UNSIGNED
             ? Interpret(value, cast.unsigned_bit_count, true)
             : value;
}

static loom_integer_cast_t Cast(uint8_t width, bool is_unsigned) {
  return {
      .source_value_id = 7,
      .bit_count = width,
      .kind = static_cast<uint8_t>(is_unsigned),
      .unsigned_bit_count = 0,
  };
}

TEST(IntegerCastTest, ThreeCastCompositionMatchesBitwiseOracle) {
  const uint8_t widths[] = {1, 8, 16, 32, 64};
  const int64_t boundaries[] = {
      INT64_MIN, INT64_MIN + 1, INT32_MIN - INT64_C(1), INT32_MIN,
      INT32_MAX, UINT32_MAX,    INT64_MAX - 1,          INT64_MAX};
  for (uint8_t first : widths) {
    for (uint8_t second : widths) {
      for (uint8_t third : widths) {
        for (uint8_t kinds = 0; kinds < 8; ++kinds) {
          SCOPED_TRACE(::testing::Message() << +first << "/" << +second << "/"
                                            << +third << " kinds=" << +kinds);
          auto cast = loom_integer_cast_compose(
              Cast(first, kinds & 1),
              static_cast<loom_integer_cast_kind_t>((kinds >> 1) & 1), second);
          cast = loom_integer_cast_compose(
              cast, static_cast<loom_integer_cast_kind_t>((kinds >> 2) & 1),
              third);
          EXPECT_EQ(cast.source_value_id, 7u);
          for (int sample = -256; sample < 264; ++sample) {
            const int64_t source =
                sample < 256 ? sample : boundaries[sample - 256];
            const uint64_t expected =
                Interpret(Interpret(Interpret(source, first, kinds & 1), second,
                                    kinds & 2),
                          third, kinds & 4);
            ASSERT_EQ(Evaluate(cast, source), expected) << source;
            // The fact domain has signed 64-bit endpoints. Unsigned results
            // above INT64_MAX need not have an exact numeric representation.
            if (cast.kind != LOOM_INTEGER_CAST_SIGNED && expected > INT64_MAX) {
              continue;
            }
            const auto facts = loom_integer_cast_apply(
                &cast, loom_value_facts_exact_i64(source));
            ASSERT_EQ(facts.range_lo, static_cast<int64_t>(expected)) << source;
            ASSERT_EQ(facts.range_hi, static_cast<int64_t>(expected)) << source;
          }
        }
      }
    }
  }
}

TEST(IntegerCastTest, IntervalTransferContainsEveryConcreteResult) {
  for (uint8_t first : {1, 8, 16, 32, 64}) {
    for (uint8_t second : {1, 8, 16, 32, 64}) {
      for (uint8_t kinds = 0; kinds < 4; ++kinds) {
        const auto cast = loom_integer_cast_compose(
            Cast(first, kinds & 1),
            static_cast<loom_integer_cast_kind_t>(kinds >> 1), second);
        for (int64_t lower = -272; lower <= 256; lower += 16) {
          const int64_t upper = lower + 31;
          const auto facts = loom_integer_cast_apply(
              &cast, loom_value_facts_make(lower, upper, 1));
          const bool identity = loom_integer_cast_is_identity(
              &cast, loom_value_facts_make(lower, upper, 1));
          for (int64_t source = lower; source <= upper; ++source) {
            const uint64_t bits = Interpret(Interpret(source, first, kinds & 1),
                                            second, kinds & 2);
            if (cast.kind != LOOM_INTEGER_CAST_SIGNED && bits > INT64_MAX) {
              EXPECT_FALSE(identity);
              continue;
            }
            const int64_t expected = static_cast<int64_t>(bits);
            ASSERT_LE(facts.range_lo, expected);
            ASSERT_GE(facts.range_hi, expected);
            if (identity) {
              ASSERT_EQ(source, expected);
            }
          }
        }
      }
    }
  }
}

TEST(IntegerCastTest, SignedPayloadNeedsItsUnsignedExtensionWidth) {
  const auto byte = Cast(8, false);
  const auto halfword =
      loom_integer_cast_compose(byte, LOOM_INTEGER_CAST_UNSIGNED, 16);
  const auto word =
      loom_integer_cast_compose(byte, LOOM_INTEGER_CAST_UNSIGNED, 32);
  EXPECT_EQ(halfword.kind, LOOM_INTEGER_CAST_SIGNED_THEN_UNSIGNED);
  EXPECT_EQ(halfword.bit_count, 8);
  EXPECT_EQ(halfword.unsigned_bit_count, 16);
  EXPECT_EQ(Evaluate(halfword, -1), 65535u);
  EXPECT_EQ(Evaluate(word, -1), UINT32_MAX);
  EXPECT_FALSE(loom_integer_cast_same_interpretation(&halfword, &word));
  const auto widened =
      loom_integer_cast_compose(halfword, LOOM_INTEGER_CAST_SIGNED, 32);
  EXPECT_TRUE(loom_integer_cast_same_interpretation(&halfword, &widened));
  const auto narrowed =
      loom_integer_cast_compose(halfword, LOOM_INTEGER_CAST_SIGNED, 16);
  EXPECT_TRUE(loom_integer_cast_same_interpretation(&byte, &narrowed));
}

TEST(IntegerCastTest, IdentityRequiresTheCompleteSourceIntervalToFit) {
  for (uint8_t width : {1, 8, 16, 32}) {
    const int64_t sign = INT64_C(1) << (width - 1);
    const auto signed_cast = Cast(width, false);
    const auto unsigned_cast = Cast(width, true);
    const auto mixed_cast =
        loom_integer_cast_compose(signed_cast, LOOM_INTEGER_CAST_UNSIGNED, 64);
    EXPECT_TRUE(loom_integer_cast_is_identity(
        &signed_cast, loom_value_facts_make(-sign, sign - 1, 1)));
    EXPECT_FALSE(loom_integer_cast_is_identity(
        &signed_cast, loom_value_facts_make(-sign, sign, 1)));
    EXPECT_FALSE(loom_integer_cast_is_identity(
        &signed_cast, loom_value_facts_make(-sign - 1, sign - 1, 1)));
    EXPECT_TRUE(loom_integer_cast_is_identity(
        &unsigned_cast, loom_value_facts_make(0, 2 * sign - 1, 1)));
    EXPECT_FALSE(loom_integer_cast_is_identity(
        &unsigned_cast, loom_value_facts_make(-1, 2 * sign - 1, 1)));
    EXPECT_FALSE(loom_integer_cast_is_identity(
        &unsigned_cast, loom_value_facts_make(0, 2 * sign, 1)));
    EXPECT_TRUE(loom_integer_cast_is_identity(
        &mixed_cast, loom_value_facts_make(0, sign - 1, 1)));
    EXPECT_FALSE(loom_integer_cast_is_identity(
        &mixed_cast, loom_value_facts_make(-1, sign - 1, 1)));
  }
}

}  // namespace
}  // namespace loom
