// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/conversion/chain.h"

#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"

namespace {

uint64_t BitMask(int width) {
  return width == 64 ? UINT64_MAX : (UINT64_C(1) << width) - 1;
}

uint64_t ConvertInteger(loom_conversion_kind_t kind, uint64_t value,
                        int input_width, int result_width) {
  if (kind == LOOM_CONVERSION_EXTSI &&
      (value & (UINT64_C(1) << (input_width - 1)))) {
    value |= ~BitMask(input_width);
  }
  return value & BitMask(result_width);
}

bool IsVerifiedIntegerConversion(loom_conversion_kind_t kind, int input_width,
                                 int result_width) {
  return kind == LOOM_CONVERSION_TRUNCI ? result_width < input_width
                                        : result_width > input_width;
}

TEST(ConversionChainTest, IntegerCompositionsPreserveBits) {
  const loom_scalar_type_t types[] = {
      LOOM_SCALAR_TYPE_I1,  LOOM_SCALAR_TYPE_I8,  LOOM_SCALAR_TYPE_I16,
      LOOM_SCALAR_TYPE_I32, LOOM_SCALAR_TYPE_I64,
  };
  const loom_conversion_kind_t kinds[] = {
      LOOM_CONVERSION_EXTSI,
      LOOM_CONVERSION_EXTUI,
      LOOM_CONVERSION_TRUNCI,
  };
  for (auto input_type : types) {
    const int input_width = loom_scalar_type_bitwidth(input_type);
    std::vector<uint64_t> inputs;
    if (input_width <= 8) {
      for (uint64_t value = 0; value <= BitMask(input_width); ++value) {
        inputs.push_back(value);
      }
    } else {
      inputs = {0, 1, BitMask(input_width)};
      for (int bit = 0; bit < input_width; ++bit) {
        const uint64_t value = UINT64_C(1) << bit;
        inputs.push_back(value);
        inputs.push_back(value - 1);
        inputs.push_back(BitMask(input_width) ^ value);
      }
    }
    for (auto middle_type : types) {
      const int middle_width = loom_scalar_type_bitwidth(middle_type);
      for (auto result_type : types) {
        const int result_width = loom_scalar_type_bitwidth(result_type);
        for (auto inner : kinds) {
          if (!IsVerifiedIntegerConversion(inner, input_width, middle_width)) {
            continue;
          }
          for (auto outer : kinds) {
            if (!IsVerifiedIntegerConversion(outer, middle_width,
                                             result_width)) {
              continue;
            }
            const auto match = loom_conversion_chain_match(outer, inner);
            if (match.candidate == LOOM_CONVERSION_NONE) {
              continue;
            }
            const auto composed = loom_conversion_chain_resolve(
                match.candidate, loom_type_scalar(input_type),
                loom_type_scalar(result_type));
            ASSERT_NE(composed, LOOM_CONVERSION_NONE);
            SCOPED_TRACE(::testing::Message()
                         << "widths " << input_width << " -> " << middle_width
                         << " -> " << result_width << ", conversions "
                         << static_cast<int>(inner) << ", "
                         << static_cast<int>(outer));
            for (uint64_t input : inputs) {
              const uint64_t middle =
                  ConvertInteger(inner, input, input_width, middle_width);
              if (iree_any_bit_set(match.flags,
                                   LOOM_CONVERSION_CHAIN_FLAG_NON_NEGATIVE) &&
                  (middle & (UINT64_C(1) << (middle_width - 1)))) {
                continue;
              }
              const uint64_t expected =
                  ConvertInteger(outer, middle, middle_width, result_width);
              const uint64_t actual =
                  ConvertInteger(composed, input, input_width, result_width);
              EXPECT_EQ(actual, expected) << "input " << input;
            }
          }
        }
      }
    }
  }
}

TEST(ConversionChainTest, UnsignedAfterSignedRequiresIntermediateProof) {
  const auto match =
      loom_conversion_chain_match(LOOM_CONVERSION_EXTUI, LOOM_CONVERSION_EXTSI);
  EXPECT_EQ(match.candidate, LOOM_CONVERSION_EXTUI);
  EXPECT_TRUE(
      iree_any_bit_set(match.flags, LOOM_CONVERSION_CHAIN_FLAG_NON_NEGATIVE));
  // A Boolean true is non-negative in i1 facts, but its signed extension is -1.
  const uint64_t middle = ConvertInteger(LOOM_CONVERSION_EXTSI, 1, 1, 16);
  EXPECT_EQ(middle, UINT64_C(0xffff));
  EXPECT_NE(ConvertInteger(LOOM_CONVERSION_EXTUI, middle, 16, 32),
            ConvertInteger(match.candidate, 1, 1, 32));
}

TEST(ConversionChainTest, IntermediateRoundingAndLostBitsRemainObservable) {
  EXPECT_EQ(loom_conversion_chain_match(LOOM_CONVERSION_FPTRUNC,
                                        LOOM_CONVERSION_FPTRUNC)
                .candidate,
            LOOM_CONVERSION_NONE);
  EXPECT_EQ(
      loom_conversion_chain_match(LOOM_CONVERSION_EXTF, LOOM_CONVERSION_FPTRUNC)
          .candidate,
      LOOM_CONVERSION_NONE);
  EXPECT_EQ(
      loom_conversion_chain_match(LOOM_CONVERSION_EXTSI, LOOM_CONVERSION_TRUNCI)
          .candidate,
      LOOM_CONVERSION_NONE);
  EXPECT_EQ(
      loom_conversion_chain_match(LOOM_CONVERSION_EXTUI, LOOM_CONVERSION_TRUNCI)
          .candidate,
      LOOM_CONVERSION_NONE);
}

TEST(ConversionChainTest, FloatCompositionUsesFormatsAndEndpointWidths) {
  const auto match = loom_conversion_chain_match(LOOM_CONVERSION_FPTRUNC,
                                                 LOOM_CONVERSION_EXTF);
  EXPECT_EQ(match.candidate, LOOM_CONVERSION_EXTF);
  const loom_type_t f8 = loom_type_scalar(LOOM_SCALAR_TYPE_F8E4M3);
  const loom_type_t f16 = loom_type_scalar(LOOM_SCALAR_TYPE_F16);
  const loom_type_t bf16 = loom_type_scalar(LOOM_SCALAR_TYPE_BF16);
  EXPECT_EQ(loom_conversion_chain_resolve(match.candidate, f8, f16),
            LOOM_CONVERSION_EXTF);
  EXPECT_EQ(loom_conversion_chain_resolve(match.candidate, f16, f8),
            LOOM_CONVERSION_FPTRUNC);
  EXPECT_EQ(loom_conversion_chain_resolve(match.candidate, bf16, bf16),
            LOOM_CONVERSION_IDENTITY);
  EXPECT_EQ(loom_conversion_chain_resolve(match.candidate, bf16, f16),
            LOOM_CONVERSION_NONE);
  EXPECT_EQ(loom_conversion_chain_resolve(
                match.candidate, f8, loom_type_scalar(LOOM_SCALAR_TYPE_F8E5M2)),
            LOOM_CONVERSION_NONE);
}

}  // namespace
