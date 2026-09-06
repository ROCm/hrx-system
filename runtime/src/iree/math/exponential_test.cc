// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/exponential.h"

#include "iree/math/elementary_test_util.h"

namespace iree::math::testing {
namespace {

TEST_F(ElementaryTest, Exp2F32ApproxExceptionalAndFlushBoundaries) {
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x00000000),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x80000000),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x80000001),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x7F800000),
                   UINT32_C(0x7F800000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0xFF800000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x7F800001),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0xFFC01234),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0xC2FC0000),
                   UINT32_C(0x00800000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0xC2FC0001),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x43000000),
                   UINT32_C(0x7F800000));
}

TEST_F(ElementaryTest, Exp2F32ApproxFrozenFiniteMapping) {
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x3F000000),
                   UINT32_C(0x3FB504F3));
  // These inputs distinguish the selected mapping from correct rounding.
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x3F800B8B),
                   UINT32_C(0x40000801));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0xC0004654),
                   UINT32_C(0x3E7F3D4D));
}

TEST_F(ElementaryTest, Log2F32ApproxExceptionalAndFlushBoundaries) {
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x00000000),
                   UINT32_C(0xFF800000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x80000000),
                   UINT32_C(0xFF800000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0xFF800000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x807FFFFF),
                   UINT32_C(0xFF800000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0xBF800000),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0xFF800000),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x7F800000),
                   UINT32_C(0x7F800000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x7F800001),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0xFFC01234),
                   UINT32_C(0x7FC00000));
}

TEST_F(ElementaryTest, Log2F32ApproxFrozenFiniteMapping) {
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x00800000),
                   UINT32_C(0xC2FC0000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x3F000000),
                   UINT32_C(0xBF800000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x3F800000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x40000000),
                   UINT32_C(0x3F800000));
  // These inputs distinguish the selected mapping from correct rounding and
  // direct AMDGPU primitives, respectively.
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x70006067),
                   UINT32_C(0x42C2022B));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x3F7FFFFF),
                   UINT32_C(0xB3B8AA3B));
}

TEST_F(ElementaryTest, ApproxMappingsIgnoreHostDenormalFlush) {
  EnableFlushDenormalsToZero();
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0xC2FC0000),
                   UINT32_C(0x00800000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0xFF800000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x70006067),
                   UINT32_C(0x42C2022B));
}

}  // namespace
}  // namespace iree::math::testing
