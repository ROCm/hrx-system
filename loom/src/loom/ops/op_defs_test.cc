// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/op_defs.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(DialectTableHelpers, ReturnVtableArraysAndCounts) {
  const loom_op_vtable_t vtable = {};
  const loom_op_vtable_t* const vtables[] = {
      &vtable,
  };

  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* result =
      loom_dialect_vtable_array(vtables, IREE_ARRAYSIZE(vtables), &count);
  EXPECT_EQ(result, vtables);
  EXPECT_EQ(count, 1u);

  EXPECT_EQ(loom_dialect_vtable_array(vtables, IREE_ARRAYSIZE(vtables),
                                      /*out_count=*/nullptr),
            vtables);
}

TEST(DialectTableHelpers, ReturnSemanticArraysAndCounts) {
  loom_op_semantics_t semantics[] = {
      loom_op_semantics_empty(),
      loom_op_semantics_empty(),
  };
  semantics[1].phase = LOOM_OP_PHASE_EXECUTABLE;
  semantics[1].contract_families = LOOM_CONTRACT_VECTOR_COORDINATE;

  iree_host_size_t count = 0;
  const loom_op_semantics_t* result = loom_dialect_semantics_array(
      semantics, IREE_ARRAYSIZE(semantics), &count);
  EXPECT_EQ(result, semantics);
  EXPECT_EQ(count, 2u);

  EXPECT_EQ(loom_dialect_semantics_array(semantics, IREE_ARRAYSIZE(semantics),
                                         /*out_count=*/nullptr),
            semantics);
}

TEST(DialectTableHelpers, LookupSemanticsByDialectAndIndex) {
  loom_op_semantics_t semantics[] = {
      loom_op_semantics_empty(),
      loom_op_semantics_empty(),
  };
  semantics[1].phase = LOOM_OP_PHASE_EXECUTABLE;
  semantics[1].contract_families = LOOM_CONTRACT_VECTOR_COORDINATE;

  loom_op_semantics_t found = loom_dialect_semantics_lookup(
      LOOM_OP_KIND(LOOM_DIALECT_TEST, 1), LOOM_DIALECT_TEST, semantics,
      IREE_ARRAYSIZE(semantics));
  EXPECT_EQ(found.phase, LOOM_OP_PHASE_EXECUTABLE);
  EXPECT_EQ(found.contract_families, LOOM_CONTRACT_VECTOR_COORDINATE);

  loom_op_semantics_t wrong_dialect = loom_dialect_semantics_lookup(
      LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 1), LOOM_DIALECT_TEST, semantics,
      IREE_ARRAYSIZE(semantics));
  EXPECT_EQ(wrong_dialect.phase, LOOM_OP_PHASE_UNSPECIFIED);
  EXPECT_EQ(wrong_dialect.contract_families, 0u);

  loom_op_semantics_t out_of_range = loom_dialect_semantics_lookup(
      LOOM_OP_KIND(LOOM_DIALECT_TEST, 2), LOOM_DIALECT_TEST, semantics,
      IREE_ARRAYSIZE(semantics));
  EXPECT_EQ(out_of_range.phase, LOOM_OP_PHASE_UNSPECIFIED);
  EXPECT_EQ(out_of_range.contract_families, 0u);
}

TEST(AttributeHelpers, EnumArrayPreservesContentAndPresentEmpty) {
  const uint8_t values[] = {1, 255, 1};
  loom_attribute_t attr = loom_attr_enum_array(values, IREE_ARRAYSIZE(values));
  loom_enum_array_t array = loom_attr_as_enum_array(attr);

  EXPECT_EQ(array.values, values);
  EXPECT_EQ(array.count, 3u);
  EXPECT_TRUE(loom_attribute_equal(&attr, &attr));

  loom_attribute_t same = loom_attr_enum_array(values, IREE_ARRAYSIZE(values));
  EXPECT_TRUE(loom_attribute_equal(&attr, &same));
  EXPECT_EQ(loom_attribute_hash(&attr), loom_attribute_hash(&same));

  loom_attribute_t empty = loom_attr_enum_array(values, 0);
  EXPECT_EQ(empty.kind, LOOM_ATTR_ENUM_ARRAY);
  EXPECT_EQ(empty.enum_array, nullptr);
  EXPECT_FALSE(loom_attr_is_absent(empty));
  EXPECT_TRUE(loom_attr_is_absent(loom_attr_absent()));
}

TEST(AttributeHelpers, SignedEnumSetPreservesCanonicalPolarities) {
  const uint64_t words[] = {
      UINT64_C(1) << 1, 0, 0, UINT64_C(1) << 63, UINT64_C(1) << 7, 0, 0, 0,
  };
  loom_attribute_t attr =
      loom_attr_signed_enum_set(words, IREE_ARRAYSIZE(words) / 2);
  loom_signed_enum_set_t set = loom_attr_as_signed_enum_set(attr);

  EXPECT_EQ(set.words, words);
  EXPECT_EQ(set.word_count, 4u);
  EXPECT_TRUE(loom_signed_enum_set_contains_positive(set, 1));
  EXPECT_TRUE(loom_signed_enum_set_contains_positive(set, 255));
  EXPECT_FALSE(loom_signed_enum_set_contains_positive(set, 7));
  EXPECT_TRUE(loom_signed_enum_set_contains_negative(set, 7));
  EXPECT_FALSE(loom_signed_enum_set_contains_negative(set, 1));

  const uint64_t same_words[] = {
      UINT64_C(1) << 1, 0, 0, UINT64_C(1) << 63, UINT64_C(1) << 7, 0, 0, 0,
  };
  loom_attribute_t same =
      loom_attr_signed_enum_set(same_words, IREE_ARRAYSIZE(same_words) / 2);
  EXPECT_TRUE(loom_attribute_equal(&attr, &same));
  EXPECT_EQ(loom_attribute_hash(&attr), loom_attribute_hash(&same));

  loom_attribute_t empty = loom_attr_signed_enum_set(words, 0);
  EXPECT_EQ(empty.kind, LOOM_ATTR_SIGNED_ENUM_SET);
  EXPECT_EQ(empty.signed_enum_set_words, nullptr);
  EXPECT_FALSE(loom_attr_is_absent(empty));
}

TEST(AttributeHelpers, SignedEnumSetValidatesAndTrimsRepresentation) {
  const uint64_t trailing_words[] = {
      UINT64_C(1) << 1, 0, 0, 0, UINT64_C(1) << 7, 0, 0, 0,
  };
  iree_host_size_t canonical_word_count = 0;
  IREE_ASSERT_OK(loom_signed_enum_set_canonical_word_count(
      loom_make_signed_enum_set(trailing_words,
                                IREE_ARRAYSIZE(trailing_words) / 2),
      &canonical_word_count));
  EXPECT_EQ(canonical_word_count, 1u);

  const uint64_t contradictory_words[] = {
      UINT64_C(1) << 1,
      UINT64_C(1) << 1,
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_signed_enum_set_canonical_word_count(
                            loom_make_signed_enum_set(contradictory_words, 1),
                            &canonical_word_count));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_signed_enum_set_canonical_word_count(
          loom_make_signed_enum_set(nullptr, 1), &canonical_word_count));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_signed_enum_set_canonical_word_count(
          loom_make_signed_enum_set(trailing_words,
                                    LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT + 1),
          &canonical_word_count));
}

TEST(MemoryAccessHelpers, OperandIndexIsPayload) {
  loom_op_t op = {};
  op.operand_count = 5;

  loom_memory_access_vtable_t memory_access = {};
  memory_access.operation_kind = LOOM_MEMORY_ACCESS_OPERATION_STORE;
  memory_access.value_operand_index = 3;
  memory_access.expected_operand_index = LOOM_OPERAND_INDEX_NONE;
  memory_access.replacement_operand_index = LOOM_OPERAND_INDEX_NONE;

  loom_op_vtable_t op_vtable = {};
  op_vtable.fixed_operand_count = op.operand_count;
  op_vtable.memory_access = &memory_access;

  loom_memory_access_t access = {};
  access.op = &op;
  access.op_vtable = &op_vtable;

  EXPECT_FALSE(loom_memory_access_operand_index_is_payload(access, 0));
  EXPECT_TRUE(loom_memory_access_operand_index_is_payload(access, 3));
  EXPECT_FALSE(loom_memory_access_operand_index_is_payload(access, 5));

  memory_access.operation_kind = LOOM_MEMORY_ACCESS_OPERATION_LOAD;
  EXPECT_FALSE(loom_memory_access_operand_index_is_payload(access, 3));

  memory_access.operation_kind = LOOM_MEMORY_ACCESS_OPERATION_ATOMIC_CMPXCHG;
  memory_access.value_operand_index = LOOM_OPERAND_INDEX_NONE;
  memory_access.expected_operand_index = 1;
  memory_access.replacement_operand_index = 2;
  EXPECT_FALSE(loom_memory_access_operand_index_is_payload(access, 0));
  EXPECT_TRUE(loom_memory_access_operand_index_is_payload(access, 1));
  EXPECT_TRUE(loom_memory_access_operand_index_is_payload(access, 2));
}

}  // namespace
}  // namespace loom
