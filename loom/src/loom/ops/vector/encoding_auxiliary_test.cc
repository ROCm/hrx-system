// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/vector/encoding_auxiliary.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace loom {
namespace {

class VectorEncodingAuxiliaryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
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

TEST(VectorEncodingAuxiliaryKeysTest, ScaleKeysUseStableDenseSlots) {
  loom_vector_encoding_auxiliary_key_t key =
      LOOM_VECTOR_ENCODING_AUXILIARY_KEY_COUNT_;

  ASSERT_TRUE(loom_vector_encoding_auxiliary_scale_key(0, &key));
  EXPECT_EQ(key, LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE);
  EXPECT_TRUE(iree_string_view_equal(
      loom_vector_encoding_auxiliary_key_name(key), IREE_SV("scale")));

  ASSERT_TRUE(loom_vector_encoding_auxiliary_scale_key(1, &key));
  EXPECT_EQ(key, LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SECONDARY_SCALE);
  EXPECT_TRUE(
      iree_string_view_equal(loom_vector_encoding_auxiliary_key_name(key),
                             IREE_SV("secondary_scale")));

  ASSERT_TRUE(loom_vector_encoding_auxiliary_scale_key(7, &key));
  EXPECT_EQ(key, LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE7);
  EXPECT_EQ(loom_vector_encoding_auxiliary_key_flag(key),
            LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE7);

  EXPECT_FALSE(loom_vector_encoding_auxiliary_scale_key(8, &key));
}

TEST_F(VectorEncodingAuxiliaryTest, ViewResolveMapsNamesToDenseValueSlots) {
  loom_value_id_t auxiliary_values[] = {12, 34};
  loom_named_attr_t auxiliary_names[] = {
      {
          /*.name_id=*/Intern(IREE_SV("scale")),
          /*.reserved=*/{},
          /*.value=*/loom_attr_i64(0),
      },
      {
          /*.name_id=*/Intern(IREE_SV("amax")),
          /*.reserved=*/{},
          /*.value=*/loom_attr_i64(1),
      },
  };

  loom_vector_encoding_auxiliary_view_t view;
  iree_string_view_t unknown_key = iree_string_view_empty();
  ASSERT_TRUE(loom_vector_encoding_auxiliary_view_resolve(
      module_, ValueSlice(auxiliary_values, 2),
      loom_make_named_attr_slice(auxiliary_names,
                                 IREE_ARRAYSIZE(auxiliary_names)),
      &view, &unknown_key));

  EXPECT_TRUE(iree_all_bits_set(
      view.present_keys, LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE |
                             LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_AMAX));
  EXPECT_EQ(view.values[LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE], 12);
  EXPECT_EQ(view.values[LOOM_VECTOR_ENCODING_AUXILIARY_KEY_AMAX], 34);
  EXPECT_EQ(view.values[LOOM_VECTOR_ENCODING_AUXILIARY_KEY_CODEBOOK],
            LOOM_VALUE_ID_INVALID);
}

TEST_F(VectorEncodingAuxiliaryTest, ViewResolveReportsUnknownKeys) {
  loom_value_id_t auxiliary_values[] = {12};
  loom_named_attr_t auxiliary_names[] = {
      {
          /*.name_id=*/Intern(IREE_SV("mystery")),
          /*.reserved=*/{},
          /*.value=*/loom_attr_i64(0),
      },
  };

  loom_vector_encoding_auxiliary_view_t view;
  iree_string_view_t unknown_key = iree_string_view_empty();
  EXPECT_FALSE(loom_vector_encoding_auxiliary_view_resolve(
      module_, ValueSlice(auxiliary_values, 1),
      loom_make_named_attr_slice(auxiliary_names,
                                 IREE_ARRAYSIZE(auxiliary_names)),
      &view, &unknown_key));
  EXPECT_TRUE(iree_string_view_equal(unknown_key, IREE_SV("mystery")));
}

TEST(VectorEncodingAuxiliaryKeysTest, RequiredKeysComeFromSchemaFacts) {
  loom_value_fact_encoded_operand_schema_t schema = {
      /*.element_format=*/{},
      /*.scale_format=*/{},
      /*.secondary_scale_format=*/{},
      /*.payload_packing=*/{},
      /*.scale_topology=*/LOOM_VALUE_FACT_SCALE_TOPOLOGY_RUNTIME_AMAX_DERIVED,
      /*.affine_policy=*/LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_ZERO_POINT |
          LOOM_VALUE_FACT_AFFINE_POLICY_SUM_CORRECTION,
      /*.rounding_policy=*/{},
      /*.codebook_policy=*/
      LOOM_VALUE_FACT_CODEBOOK_POLICY_DYNAMIC_TABLE_OPERAND,
      /*.sparsity_policy=*/LOOM_VALUE_FACT_SPARSITY_POLICY_MASK,
      /*.flags=*/{},
      /*.reserved=*/{},
      /*.payload_register_count=*/{},
      /*.payload_element_count=*/{},
      /*.scale_group_element_count=*/{},
      /*.scale_operand_count=*/2,
  };

  loom_vector_encoding_auxiliary_key_flags_t required_keys = 0;
  ASSERT_TRUE(loom_vector_encoding_auxiliary_required_keys_from_schema(
      schema, &required_keys, nullptr));
  EXPECT_TRUE(iree_all_bits_set(
      required_keys,
      LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE |
          LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SECONDARY_SCALE |
          LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_AMAX |
          LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_ZERO_POINT |
          LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SUM_CORRECTION |
          LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_CODEBOOK |
          LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SPARSITY));
}

TEST(VectorEncodingAuxiliaryKeysTest, RequiredKeysRejectUnsupportedScaleCount) {
  loom_value_fact_encoded_operand_schema_t schema = {
      /*.element_format=*/{},
      /*.scale_format=*/{},
      /*.secondary_scale_format=*/{},
      /*.payload_packing=*/{},
      /*.scale_topology=*/{},
      /*.affine_policy=*/{},
      /*.rounding_policy=*/{},
      /*.codebook_policy=*/{},
      /*.sparsity_policy=*/{},
      /*.flags=*/{},
      /*.reserved=*/{},
      /*.payload_register_count=*/{},
      /*.payload_element_count=*/{},
      /*.scale_group_element_count=*/{},
      /*.scale_operand_count=*/9,
  };

  uint16_t unsupported_scale_index = UINT16_MAX;
  loom_vector_encoding_auxiliary_key_flags_t required_keys = 0;
  EXPECT_FALSE(loom_vector_encoding_auxiliary_required_keys_from_schema(
      schema, &required_keys, &unsupported_scale_index));
  EXPECT_EQ(unsupported_scale_index, 8);
  EXPECT_TRUE(iree_all_bits_set(
      required_keys,
      LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE |
          LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SECONDARY_SCALE |
          LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE2 |
          LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE3 |
          LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE4 |
          LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE5 |
          LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE6 |
          LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SCALE7));
}

}  // namespace
}  // namespace loom
