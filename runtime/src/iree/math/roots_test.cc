// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/roots.h"

#include "iree/math/elementary_test_util.h"

namespace iree::math::testing {
namespace {

TEST_F(ElementaryTest, ReciprocalF32ApproxExceptionalAndFlushBoundaries) {
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x00000000),
                   UINT32_C(0x7F800000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x80000000),
                   UINT32_C(0xFF800000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0x7F800000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x80000001),
                   UINT32_C(0xFF800000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x7F800000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0xFF800000),
                   UINT32_C(0x80000000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x7F800001),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0xFFC01234),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x7F7FFFFF),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0xFF7FFFFF),
                   UINT32_C(0x80000000));
}

TEST_F(ElementaryTest, ReciprocalF32ApproxFrozenFiniteMapping) {
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x00800000),
                   UINT32_C(0x7E800000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x40800000),
                   UINT32_C(0x3E800000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x0135F782),
                   UINT32_C(0x7DB413A8));
}

TEST_F(ElementaryTest, RsqrtF32ApproxExceptionalAndFlushBoundaries) {
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x00000000),
                   UINT32_C(0x7F800000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x80000000),
                   UINT32_C(0xFF800000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0x7F800000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x80000001),
                   UINT32_C(0xFF800000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0xBF800000),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x7F800000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0xFF800000),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x7F800001),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0xFFC01234),
                   UINT32_C(0x7FC00000));
}

TEST_F(ElementaryTest, RsqrtF32ApproxFrozenFiniteMapping) {
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x40800000),
                   UINT32_C(0x3F000000));
  // This input distinguishes the staged-f32 mapping from correct rounding.
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x70000002),
                   UINT32_C(0x273504F1));
}

TEST_F(ElementaryTest, SqrtF32ApproxExceptionalAndFlushBoundaries) {
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0x00000000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0x80000000),
                   UINT32_C(0x80000000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0x80000001),
                   UINT32_C(0x80000000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0xBF800000),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0x7F800000),
                   UINT32_C(0x7F800000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0xFF800000),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0x7F800001),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0xFFC01234),
                   UINT32_C(0x7FC00000));
}

TEST_F(ElementaryTest, SqrtF32ApproxFrozenFiniteMapping) {
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0x40800000),
                   UINT32_C(0x40000000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0x6144BCA4),
                   UINT32_C(0x50606BB2));
}

TEST_F(ElementaryTest, ApproxMappingsIgnoreHostDenormalFlush) {
  EnableFlushDenormalsToZero();
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x7F7FFFFF),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0x7F800000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x70000002),
                   UINT32_C(0x273504F1));
}

}  // namespace
}  // namespace iree::math::testing
