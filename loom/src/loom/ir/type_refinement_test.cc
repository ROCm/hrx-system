// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/type_refinement.h"

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class TypeRefinementTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_type_t MakeRank3Vector(uint64_t dim0, uint64_t dim1, uint64_t dim2) {
    loom_overflow_dim_t* dimensions = nullptr;
    IREE_CHECK_OK(iree_arena_allocate_array(&arena_, 3, sizeof(*dimensions),
                                            (void**)&dimensions));
    dimensions[0] = dim0;
    dimensions[1] = dim1;
    dimensions[2] = dim2;

    loom_type_t type = {};
    uint8_t flags = 0;
    if (!loom_dim_is_dynamic(dim0) && !loom_dim_is_dynamic(dim1) &&
        !loom_dim_is_dynamic(dim2)) {
      flags |= LOOM_TYPE_FLAG_ALL_STATIC;
    }
    type.header =
        loom_type_make_header(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, 3, flags);
    type.dims[0] = (uint64_t)(uintptr_t)dimensions;
    return type;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(TypeRefinementTest, DynamicDimensionNarrowsToStaticDimension) {
  loom_type_t current =
      loom_type_shaped_2d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(1), loom_dim_pack_static(4), 0);
  loom_type_t candidate =
      loom_type_shaped_2d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(16), loom_dim_pack_static(4), 0);

  loom_type_t refined = {};
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_CONFLICT;
  IREE_ASSERT_OK(loom_type_refine_with_candidate(current, candidate, &arena_,
                                                 &refined, &result));

  EXPECT_EQ(result, LOOM_TYPE_REFINEMENT_NARROWED);
  EXPECT_EQ(loom_type_dim_static_size_at(refined, 0), 16);
  EXPECT_EQ(loom_type_dim_static_size_at(refined, 1), 4);
  EXPECT_TRUE(loom_type_is_all_static(refined));
}

TEST_F(TypeRefinementTest, StaticDimensionDoesNotWidenToDynamicDimension) {
  loom_type_t current = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 0);
  loom_type_t candidate = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(2), 0);

  loom_type_t refined = {};
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_CONFLICT;
  IREE_ASSERT_OK(loom_type_refine_shape_with_candidate(
      current, candidate, &arena_, &refined, &result));

  EXPECT_EQ(result, LOOM_TYPE_REFINEMENT_UNCHANGED);
  EXPECT_TRUE(loom_type_equal(refined, current));
}

TEST_F(TypeRefinementTest, ConflictingStaticDimensionsAreRejected) {
  loom_type_t current = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 0);
  loom_type_t candidate = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(32), 0);

  loom_type_t refined = {};
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_ASSERT_OK(loom_type_refine_with_candidate(current, candidate, &arena_,
                                                 &refined, &result));

  EXPECT_EQ(result, LOOM_TYPE_REFINEMENT_CONFLICT);
  EXPECT_TRUE(loom_type_equal(refined, current));
}

TEST_F(TypeRefinementTest, OverflowDimensionsAreRebuiltWhenNarrowed) {
  loom_type_t current =
      MakeRank3Vector(loom_dim_pack_dynamic(1), loom_dim_pack_static(4),
                      loom_dim_pack_dynamic(2));
  loom_type_t candidate =
      MakeRank3Vector(loom_dim_pack_static(16), loom_dim_pack_static(4),
                      loom_dim_pack_static(8));

  loom_type_t refined = {};
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_CONFLICT;
  IREE_ASSERT_OK(loom_type_refine_shape_with_candidate(
      current, candidate, &arena_, &refined, &result));

  EXPECT_EQ(result, LOOM_TYPE_REFINEMENT_NARROWED);
  EXPECT_FALSE(loom_type_has_inline_dims(refined));
  EXPECT_TRUE(loom_type_is_all_static(refined));
  EXPECT_EQ(loom_type_dim_static_size_at(refined, 0), 16);
  EXPECT_EQ(loom_type_dim_static_size_at(refined, 1), 4);
  EXPECT_EQ(loom_type_dim_static_size_at(refined, 2), 8);
}

TEST_F(TypeRefinementTest, PoolDimensionNarrowsLikeOtherDimensionedTypes) {
  loom_type_t current = loom_type_pool(loom_dim_pack_dynamic(1));
  loom_type_t candidate = loom_type_pool(loom_dim_pack_static(4096));

  loom_type_t refined = {};
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_CONFLICT;
  IREE_ASSERT_OK(loom_type_refine_with_candidate(current, candidate, &arena_,
                                                 &refined, &result));

  EXPECT_EQ(result, LOOM_TYPE_REFINEMENT_NARROWED);
  EXPECT_EQ(loom_type_dim_static_size_at(refined, 0), 4096);
  EXPECT_TRUE(loom_type_is_all_static(refined));
}

TEST_F(TypeRefinementTest, SsaEncodingNarrowsToStaticEncoding) {
  loom_type_t current = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 0);
  current.encoding_id = 3;
  current.encoding_flags = LOOM_ENCODING_FLAG_SSA;

  loom_type_t candidate = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 7);

  loom_type_t refined = {};
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_CONFLICT;
  IREE_ASSERT_OK(loom_type_refine_encoding_with_candidate(
      current, candidate, &arena_, &refined, &result));

  EXPECT_EQ(result, LOOM_TYPE_REFINEMENT_NARROWED);
  EXPECT_TRUE(loom_type_has_static_encoding(refined));
  EXPECT_EQ(refined.encoding_id, 7);
  EXPECT_EQ(refined.encoding_flags, 0);
  EXPECT_EQ(loom_type_dim_static_size_at(refined, 0), 16);
}

TEST_F(TypeRefinementTest, StaticEncodingDoesNotWidenToSsaEncoding) {
  loom_type_t current = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 7);
  loom_type_t candidate = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 0);
  candidate.encoding_id = 3;
  candidate.encoding_flags = LOOM_ENCODING_FLAG_SSA;

  loom_type_t refined = {};
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_CONFLICT;
  IREE_ASSERT_OK(loom_type_refine_encoding_with_candidate(
      current, candidate, &arena_, &refined, &result));

  EXPECT_EQ(result, LOOM_TYPE_REFINEMENT_UNCHANGED);
  EXPECT_TRUE(loom_type_equal(refined, current));
}

TEST_F(TypeRefinementTest, VectorEncodingCandidateConflicts) {
  loom_type_t current = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 0);
  loom_type_t candidate = current;
  candidate.encoding_id = 7;

  loom_type_t refined = {};
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_ASSERT_OK(loom_type_refine_encoding_with_candidate(
      current, candidate, &arena_, &refined, &result));

  EXPECT_EQ(result, LOOM_TYPE_REFINEMENT_CONFLICT);
  EXPECT_TRUE(loom_type_equal(refined, current));
}

TEST_F(TypeRefinementTest, UnknownEncodingRoleNarrowsToConcreteRole) {
  loom_type_t current = loom_type_encoding();
  loom_type_t candidate =
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);

  loom_type_t refined = {};
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_CONFLICT;
  IREE_ASSERT_OK(loom_type_refine_with_candidate(current, candidate, &arena_,
                                                 &refined, &result));

  EXPECT_EQ(result, LOOM_TYPE_REFINEMENT_NARROWED);
  EXPECT_EQ(loom_type_encoding_role(refined),
            LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
}

TEST_F(TypeRefinementTest, ElementMismatchConflicts) {
  loom_type_t current = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 0);
  loom_type_t candidate = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(16), 0);

  loom_type_t refined = {};
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_ASSERT_OK(loom_type_refine_with_candidate(current, candidate, &arena_,
                                                 &refined, &result));

  EXPECT_EQ(result, LOOM_TYPE_REFINEMENT_CONFLICT);
  EXPECT_TRUE(loom_type_equal(refined, current));
}

}  // namespace
}  // namespace loom
