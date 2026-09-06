// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/trigonometry.h"

#include "iree/math/elementary_test_util.h"

namespace iree::math::testing {
namespace {

TEST_F(ElementaryTest, TurnsF32ApproxExceptionalValues) {
  for (uint32_t input_bits : {UINT32_C(0x7F800000), UINT32_C(0xFF800000),
                              UINT32_C(0x7F800001), UINT32_C(0xFFC01234)}) {
    ExpectF32Mapping(iree_math_sin_turns_f32_approx, input_bits,
                     UINT32_C(0x7FC00000));
    ExpectF32Mapping(iree_math_cos_turns_f32_approx, input_bits,
                     UINT32_C(0x7FC00000));
  }
}

TEST_F(ElementaryTest, TurnsF32ApproxCardinalValues) {
  // Signed zero and full turns.
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x00000000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x80000000),
                   UINT32_C(0x80000000));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x3F800000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0xBF800000),
                   UINT32_C(0x80000000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0x00000000),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0x80000000),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0x3F800000),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0xBF800000),
                   UINT32_C(0x3F800000));

  // Quarter and half turns.
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x3E800000),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0xBE800000),
                   UINT32_C(0xBF800000));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x3F000000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0xBF000000),
                   UINT32_C(0x80000000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0x3E800000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0xBE800000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0x3F000000),
                   UINT32_C(0xBF800000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0xBF000000),
                   UINT32_C(0xBF800000));

  // Every f32 value with magnitude at least 2^23 is an integral turn.
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x4B000000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0xCB000000),
                   UINT32_C(0x80000000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0x4B000000),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0xCB000000),
                   UINT32_C(0x3F800000));
}

TEST_F(ElementaryTest, TurnsF32ApproxFrozenFiniteMapping) {
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0x00000006));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x80000001),
                   UINT32_C(0x80000006));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x3E000000),
                   UINT32_C(0x3F3504F3));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0x3E000000),
                   UINT32_C(0x3F3504F3));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x465BFDF6),
                   UINT32_C(0x3D7B2B74));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0xC5CD59EC),
                   UINT32_C(0x3D7B2B74));
}

}  // namespace
}  // namespace iree::math::testing
