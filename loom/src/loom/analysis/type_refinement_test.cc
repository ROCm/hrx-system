// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/type_refinement.h"

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/families.h"
#include "loom/ops/encoding/storage.h"

namespace loom {
namespace {

class AnalysisTypeRefinementTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&fact_table_, &arena_, 16));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void DefineFacts(loom_value_id_t value_id, loom_value_facts_t facts) {
    IREE_ASSERT_OK(loom_value_fact_table_define(&fact_table_, value_id, facts));
  }

  // Storage backing the analysis arena and optional module.
  iree_arena_block_pool_t block_pool_;
  // Lifetime of candidate types and fact extensions.
  iree_arena_allocator_t arena_;
  // Proven facts indexed by the identities referenced by each type.
  loom_value_fact_table_t fact_table_;
};

TEST_F(AnalysisTypeRefinementTest, ExactDimensionFactsNarrowDynamicDimensions) {
  DefineFacts(1, loom_value_facts_exact_i64(32));
  DefineFacts(2, loom_value_facts_make(1, 64, 1));

  loom_type_t current = loom_type_shaped_2d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_dynamic(1),
      loom_dim_pack_dynamic(2), 0);

  loom_type_t refined = {};
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_CONFLICT;
  IREE_ASSERT_OK(loom_type_refine_with_value_facts(current, &fact_table_,
                                                   &arena_, &refined, &result));

  EXPECT_EQ(result, LOOM_TYPE_REFINEMENT_NARROWED);
  EXPECT_EQ(loom_type_dim_static_size_at(refined, 0), 32);
  EXPECT_TRUE(loom_type_dim_is_dynamic_at(refined, 1));
  EXPECT_EQ(loom_type_dim_value_id_at(refined, 1), 2);
}

TEST_F(AnalysisTypeRefinementTest, NegativeDimensionFactConflicts) {
  DefineFacts(1, loom_value_facts_exact_i64(-1));
  loom_type_t current = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_dynamic(1), 0);

  loom_type_t refined = {};
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_ASSERT_OK(loom_type_refine_with_value_facts(current, &fact_table_,
                                                   &arena_, &refined, &result));

  EXPECT_EQ(result, LOOM_TYPE_REFINEMENT_CONFLICT);
  EXPECT_TRUE(loom_type_equal(refined, current));
}

TEST_F(AnalysisTypeRefinementTest, FloatDimensionFactConflicts) {
  DefineFacts(1, loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, 4.0));
  loom_type_t current = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_dynamic(1), 0);

  loom_type_t refined = {};
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_ASSERT_OK(loom_type_refine_with_value_facts(current, &fact_table_,
                                                   &arena_, &refined, &result));

  EXPECT_EQ(result, LOOM_TYPE_REFINEMENT_CONFLICT);
  EXPECT_TRUE(loom_type_equal(refined, current));
}

TEST_F(AnalysisTypeRefinementTest, EncodingSummaryNarrowsSsaEncoding) {
  loom_value_fact_encoding_summary_t summary = {
      .role = LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
      .static_spec_encoding_id = 11,
      .address_layout = {},
  };
  loom_value_facts_t encoding_facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_encoding_summary(
      &fact_table_.context, summary, &encoding_facts));
  DefineFacts(3, encoding_facts);

  loom_type_t current = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 0);
  current.encoding_id = 3;
  current.encoding_flags = LOOM_ENCODING_FLAG_SSA;

  loom_type_t refined = {};
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_CONFLICT;
  IREE_ASSERT_OK(loom_type_refine_with_value_facts(current, &fact_table_,
                                                   &arena_, &refined, &result));

  EXPECT_EQ(result, LOOM_TYPE_REFINEMENT_NARROWED);
  EXPECT_TRUE(loom_type_has_static_encoding(refined));
  EXPECT_EQ(refined.encoding_id, 11);
  EXPECT_EQ(refined.encoding_flags, 0);
}

TEST_F(AnalysisTypeRefinementTest, UnknownFactsLeaveTypeUnchanged) {
  loom_type_t current = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(1), 0);
  current.encoding_id = 3;
  current.encoding_flags = LOOM_ENCODING_FLAG_SSA;

  loom_type_t refined = {};
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_CONFLICT;
  IREE_ASSERT_OK(loom_type_refine_with_value_facts(current, &fact_table_,
                                                   &arena_, &refined, &result));

  EXPECT_EQ(result, LOOM_TYPE_REFINEMENT_UNCHANGED);
  EXPECT_TRUE(loom_type_equal(refined, current));
}

TEST_F(AnalysisTypeRefinementTest, MalformedWideDynamicDimensionIsIgnored) {
  uint64_t malformed_dimension =
      LOOM_DIM_DYNAMIC_FLAG | ((uint64_t)UINT32_MAX + 1);
  loom_type_t current = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, malformed_dimension, 0);

  loom_type_t refined = {};
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_CONFLICT;
  IREE_ASSERT_OK(loom_type_refine_with_value_facts(current, &fact_table_,
                                                   &arena_, &refined, &result));

  EXPECT_EQ(result, LOOM_TYPE_REFINEMENT_UNCHANGED);
  EXPECT_TRUE(loom_type_equal(refined, current));
}

class AnalysisTypeSpecializationTest : public AnalysisTypeRefinementTest {
 protected:
  void SetUp() override {
    AnalysisTypeRefinementTest::SetUp();
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_register_builtin_encoding_vtables(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("specialization"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    AnalysisTypeRefinementTest::TearDown();
  }

  // Encoding-family registration used by the production type interner.
  loom_context_t context_;
  // Owner of canonical types and static encoding definitions.
  loom_module_t* module_ = nullptr;
};

TEST_F(AnalysisTypeSpecializationTest, CanonicalizesExactEncodingCandidates) {
  DefineFacts(1, loom_value_facts_exact_i64(16));
  const loom_value_facts_t stride = loom_value_facts_exact_i64(4);
  for (bool dense : {false, true}) {
    loom_value_fact_encoding_summary_t summary = {
        .role = LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
        .static_spec_encoding_id = 0,
        .address_layout =
            {
                .kind = static_cast<loom_value_fact_address_layout_kind_t>(
                    dense ? LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE
                          : LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED),
                .rank = static_cast<uint8_t>(dense ? 0 : 1),
                .strides = dense ? nullptr : &stride,
            },
    };
    uint16_t encoding_id = 0;
    IREE_ASSERT_OK(
        loom_encoding_intern_exact_summary(module_, &summary, &encoding_id));
    ASSERT_NE(encoding_id, 0);
    for (bool named_spec : {false, true}) {
      summary.static_spec_encoding_id = named_spec ? encoding_id : 0;
      loom_value_facts_t facts = loom_value_facts_unknown();
      IREE_ASSERT_OK(loom_value_facts_make_encoding_summary(
          &fact_table_.context, summary, &facts));
      DefineFacts(3, facts);
      for (loom_type_kind_t kind :
           {LOOM_TYPE_TILE, LOOM_TYPE_TENSOR, LOOM_TYPE_VIEW}) {
        for (bool encoding_only : {false, true}) {
          SCOPED_TRACE(::testing::Message()
                       << "dense=" << dense << " named=" << named_spec
                       << " kind=" << kind
                       << " encoding_only=" << encoding_only);
          loom_type_t current = loom_type_shaped_1d(
              kind, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(1), 0);
          current.encoding_id = 3;
          current.encoding_flags = LOOM_ENCODING_FLAG_SSA;
          loom_type_t refined = {};
          loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
          auto specialize = encoding_only
                                ? loom_type_specialize_encoding_with_value_facts
                                : loom_type_specialize_with_value_facts;
          IREE_ASSERT_OK(specialize(module_, current, &fact_table_, &arena_,
                                    &refined, &result));
          EXPECT_EQ(result, LOOM_TYPE_REFINEMENT_NARROWED);
          EXPECT_EQ(refined.encoding_id, dense ? 0 : encoding_id);
          EXPECT_EQ(refined.encoding_flags, 0);
          EXPECT_EQ(loom_type_dim(refined, 0), encoding_only
                                                   ? loom_dim_pack_dynamic(1)
                                                   : loom_dim_pack_static(16));
          EXPECT_NE(loom_module_lookup_type_id(module_, refined),
                    LOOM_TYPE_ID_INVALID);
        }
      }
    }
  }
}

}  // namespace
}  // namespace loom
