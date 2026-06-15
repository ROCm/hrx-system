// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/vector/fragment.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace loom {
namespace {

class VectorFragmentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_string_id_t Intern(iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module_, name, &name_id));
    return name_id;
  }

  static loom_value_slice_t ValueSlice(loom_value_id_t* values,
                                       uint16_t count) {
    return loom_value_slice_t{
        /*.values=*/values,
        /*.count=*/count,
    };
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
};

TEST(VectorFragmentFactsTest, UnknownIsAllZeroAndByteComparable) {
  loom_vector_fragment_fact_t fact;
  loom_vector_fragment_fact_initialize(&fact);
  loom_vector_fragment_fact_t unknown;
  loom_vector_fragment_fact_initialize(&unknown);

  EXPECT_TRUE(loom_vector_fragment_fact_is_unknown(fact));
  EXPECT_TRUE(loom_vector_fragment_fact_equal(fact, unknown));

  fact.role_flags = loom_vector_fragment_role_flag(LOOM_VECTOR_ROLE_LHS);
  fact.shape_rank = 2;
  fact.shape_value_ids[0] = 12;
  fact.shape_value_ids[1] = 34;

  EXPECT_FALSE(loom_vector_fragment_fact_is_unknown(fact));
  EXPECT_FALSE(loom_vector_fragment_fact_equal(fact, unknown));
}

TEST(VectorFragmentFactsTest, PayloadRoundTripsThroughValueFacts) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_value_fact_table_t table = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena, 0));

  loom_vector_fragment_fact_t fact;
  loom_vector_fragment_fact_initialize(&fact);
  fact.flags = LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_SCHEMA |
               LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_STATIC_SCHEMA;
  fact.role_flags = loom_vector_fragment_role_flag(LOOM_VECTOR_ROLE_RHS);
  fact.schema_value_id = 56;
  fact.static_schema_encoding_id = 3;
  fact.shape_rank = 2;
  fact.shape_value_ids[0] = 78;
  fact.shape_value_ids[1] = 90;
  fact.auxiliary.present_keys = LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE;
  fact.auxiliary.values[LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE] = 123;
  fact.encoded_operand.scale_operand_count = 1;

  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(
      loom_vector_fragment_fact_make_value_facts(&table.context, fact, &facts));

  loom_vector_fragment_fact_t queried;
  ASSERT_TRUE(loom_vector_fragment_fact_query_value_facts(&table.context, facts,
                                                          &queried));
  EXPECT_TRUE(loom_vector_fragment_fact_equal(fact, queried));

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST_F(VectorFragmentTest, ParameterViewResolveMapsSchemaAndAuxiliaryKeys) {
  loom_value_id_t parameter_values[] = {12, 34, 56};
  loom_named_attr_t parameter_names[] = {
      {
          /*.name_id=*/Intern(IREE_SV("schema")),
          /*.reserved=*/{},
          /*.value=*/loom_attr_i64(0),
      },
      {
          /*.name_id=*/Intern(IREE_SV("scale")),
          /*.reserved=*/{},
          /*.value=*/loom_attr_i64(1),
      },
      {
          /*.name_id=*/Intern(IREE_SV("amax")),
          /*.reserved=*/{},
          /*.value=*/loom_attr_i64(2),
      },
  };

  loom_vector_fragment_parameter_view_t view;
  iree_string_view_t unknown_key = iree_string_view_empty();
  ASSERT_TRUE(loom_vector_fragment_parameter_view_resolve(
      module_, ValueSlice(parameter_values, 3),
      loom_make_named_attr_slice(parameter_names,
                                 IREE_ARRAYSIZE(parameter_names)),
      &view, &unknown_key));

  EXPECT_TRUE(view.has_schema);
  EXPECT_EQ(view.schema_value_id, 12);
  EXPECT_EQ(view.schema_parameter_ordinal, 0);
  EXPECT_TRUE(
      iree_all_bits_set(view.auxiliary.present_keys,
                        LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE |
                            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_AMAX));
  EXPECT_EQ(view.auxiliary.values[LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE],
            34);
  EXPECT_EQ(view.auxiliary.values[LOOM_VECTOR_ENCODING_AUXILIARY_KEY_AMAX], 56);
}

TEST_F(VectorFragmentTest, ParameterViewResolveReportsUnknownKeys) {
  loom_value_id_t parameter_values[] = {12};
  loom_named_attr_t parameter_names[] = {
      {
          /*.name_id=*/Intern(IREE_SV("mystery")),
          /*.reserved=*/{},
          /*.value=*/loom_attr_i64(0),
      },
  };

  loom_vector_fragment_parameter_view_t view;
  iree_string_view_t unknown_key = iree_string_view_empty();
  EXPECT_FALSE(loom_vector_fragment_parameter_view_resolve(
      module_, ValueSlice(parameter_values, 1),
      loom_make_named_attr_slice(parameter_names,
                                 IREE_ARRAYSIZE(parameter_names)),
      &view, &unknown_key));
  EXPECT_TRUE(iree_string_view_equal(unknown_key, IREE_SV("mystery")));
}

}  // namespace
}  // namespace loom
