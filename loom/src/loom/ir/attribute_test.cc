// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/attribute.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(AttributeTest, Size) { static_assert(sizeof(loom_attribute_t) == 16); }

TEST(AttributeTest, PredicateMetadataCoversEveryKind) {
  static constexpr const char* kNames[] = {
      "eq",          "ne",           "lt",    "le",      "gt",      "ge",
      "multiple_of", "power_of_two", "range", "not_nan", "not_inf", "finite",
      "ult",         "ule",          "ugt",   "uge",
  };
  static_assert(IREE_ARRAYSIZE(kNames) == LOOM_PREDICATE_COUNT_);

  for (loom_predicate_kind_t kind = 0; kind < LOOM_PREDICATE_COUNT_; ++kind) {
    EXPECT_STREQ(loom_predicate_kind_name(kind), kNames[kind]);
    loom_predicate_kind_t parsed_kind = LOOM_PREDICATE_COUNT_;
    EXPECT_TRUE(loom_predicate_kind_parse(iree_make_cstring_view(kNames[kind]),
                                          &parsed_kind));
    EXPECT_EQ(parsed_kind, kind);
    EXPECT_NE(loom_predicate_kind_argument_count(kind), UINT8_MAX);
  }

  loom_predicate_kind_t parsed_kind = LOOM_PREDICATE_COUNT_;
  EXPECT_FALSE(loom_predicate_kind_parse(IREE_SV("unknown"), &parsed_kind));
  EXPECT_EQ(loom_predicate_kind_name(LOOM_PREDICATE_COUNT_), nullptr);
  EXPECT_EQ(loom_predicate_kind_argument_count(LOOM_PREDICATE_COUNT_),
            UINT8_MAX);
}

TEST(AttributeTest, PredicateValueTypeContracts) {
  static constexpr loom_predicate_kind_t kIntegerPredicateKinds[] = {
      LOOM_PREDICATE_LT,          LOOM_PREDICATE_LE,
      LOOM_PREDICATE_GT,          LOOM_PREDICATE_GE,
      LOOM_PREDICATE_MULTIPLE_OF, LOOM_PREDICATE_POWER_OF_TWO,
      LOOM_PREDICATE_RANGE,
  };
  static constexpr loom_predicate_kind_t kEqualityPredicateKinds[] = {
      LOOM_PREDICATE_EQ,
      LOOM_PREDICATE_NE,
  };
  static constexpr loom_predicate_kind_t kFloatPredicateKinds[] = {
      LOOM_PREDICATE_NOT_NAN,
      LOOM_PREDICATE_NOT_INF,
      LOOM_PREDICATE_FINITE,
  };
  static constexpr loom_predicate_kind_t kUnsignedPredicateKinds[] = {
      LOOM_PREDICATE_ULT,
      LOOM_PREDICATE_ULE,
      LOOM_PREDICATE_UGT,
      LOOM_PREDICATE_UGE,
  };
  const loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const loom_type_t offset = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  const loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  const loom_type_t buffer = loom_type_buffer();

  for (loom_predicate_kind_t kind : kIntegerPredicateKinds) {
    EXPECT_TRUE(loom_predicate_kind_accepts_value_type(kind, i32));
    EXPECT_TRUE(loom_predicate_kind_accepts_value_type(kind, index));
    EXPECT_TRUE(loom_predicate_kind_accepts_value_type(kind, offset));
    EXPECT_FALSE(loom_predicate_kind_accepts_value_type(kind, f32));
    EXPECT_FALSE(loom_predicate_kind_accepts_value_type(kind, buffer));
  }
  for (loom_predicate_kind_t kind : kEqualityPredicateKinds) {
    EXPECT_TRUE(loom_predicate_kind_accepts_value_type(kind, i32));
    EXPECT_TRUE(loom_predicate_kind_accepts_value_type(kind, index));
    EXPECT_TRUE(loom_predicate_kind_accepts_value_type(kind, offset));
    EXPECT_TRUE(loom_predicate_kind_accepts_value_type(kind, f32));
    EXPECT_FALSE(loom_predicate_kind_accepts_value_type(kind, buffer));
  }
  for (loom_predicate_kind_t kind : kFloatPredicateKinds) {
    EXPECT_FALSE(loom_predicate_kind_accepts_value_type(kind, i32));
    EXPECT_TRUE(loom_predicate_kind_accepts_value_type(kind, f32));
    EXPECT_FALSE(loom_predicate_kind_accepts_value_type(kind, buffer));
  }
  for (loom_predicate_kind_t kind : kUnsignedPredicateKinds) {
    EXPECT_TRUE(loom_predicate_kind_accepts_value_type(kind, i32));
    EXPECT_FALSE(loom_predicate_kind_accepts_value_type(kind, index));
    EXPECT_FALSE(loom_predicate_kind_accepts_value_type(kind, offset));
    EXPECT_FALSE(loom_predicate_kind_accepts_value_type(kind, f32));
    EXPECT_FALSE(loom_predicate_kind_accepts_value_type(kind, buffer));
  }
  EXPECT_FALSE(
      loom_predicate_kind_accepts_value_type(LOOM_PREDICATE_COUNT_, i32));
  EXPECT_FALSE(loom_predicate_kind_accepts_value_type(UINT8_MAX, i32));
}

TEST(AttributeTest, PredicateValueTypeContractsUseTypedRegisterSemantics) {
  const loom_register_type_data_t integer_data = {
      .carrier_payload0 = 0,
      .carrier_payload1 = 0,
      .value_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32),
  };
  const loom_register_type_data_t float_data = {
      .carrier_payload0 = 0,
      .carrier_payload1 = 0,
      .value_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32),
  };
  const loom_type_t integer_register =
      loom_type_register_payload_with_value_type(&integer_data);
  const loom_type_t float_register =
      loom_type_register_payload_with_value_type(&float_data);
  const loom_type_t untyped_register = loom_type_register_payload(0, 0);

  EXPECT_TRUE(loom_predicate_kind_accepts_value_type(LOOM_PREDICATE_EQ,
                                                     integer_register));
  EXPECT_TRUE(loom_predicate_kind_accepts_value_type(LOOM_PREDICATE_ULE,
                                                     integer_register));
  EXPECT_FALSE(loom_predicate_kind_accepts_value_type(LOOM_PREDICATE_FINITE,
                                                      integer_register));
  EXPECT_TRUE(loom_predicate_kind_accepts_value_type(LOOM_PREDICATE_EQ,
                                                     float_register));
  EXPECT_FALSE(loom_predicate_kind_accepts_value_type(LOOM_PREDICATE_ULE,
                                                      float_register));
  EXPECT_TRUE(loom_predicate_kind_accepts_value_type(LOOM_PREDICATE_FINITE,
                                                     float_register));
  EXPECT_FALSE(loom_predicate_kind_accepts_value_type(LOOM_PREDICATE_EQ,
                                                      untyped_register));
  EXPECT_FALSE(loom_predicate_kind_accepts_value_type(LOOM_PREDICATE_ULE,
                                                      untyped_register));
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

//===----------------------------------------------------------------------===//
// Attribute equality and hashing
//===----------------------------------------------------------------------===//

TEST(AttributeEqual, InlineI64) {
  loom_attribute_t a = loom_attr_i64(42);
  loom_attribute_t b = loom_attr_i64(42);
  loom_attribute_t c = loom_attr_i64(99);
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
  EXPECT_FALSE(loom_attribute_equal(&a, &c));
}

TEST(AttributeEqual, InlineF64) {
  loom_attribute_t a = loom_attr_f64(3.14);
  loom_attribute_t b = loom_attr_f64(3.14);
  loom_attribute_t c = loom_attr_f64(2.71);
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
  EXPECT_FALSE(loom_attribute_equal(&a, &c));
}

TEST(AttributeEqual, InlineBool) {
  loom_attribute_t t = loom_attr_bool(true);
  loom_attribute_t t2 = loom_attr_bool(true);
  loom_attribute_t f = loom_attr_bool(false);
  EXPECT_TRUE(loom_attribute_equal(&t, &t2));
  EXPECT_FALSE(loom_attribute_equal(&t, &f));
}

TEST(AttributeEqual, InlineEnum) {
  loom_attribute_t a = loom_attr_enum(3);
  loom_attribute_t b = loom_attr_enum(3);
  loom_attribute_t c = loom_attr_enum(5);
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
  EXPECT_FALSE(loom_attribute_equal(&a, &c));
}

TEST(AttributeEqual, DifferentKinds) {
  loom_attribute_t i = loom_attr_i64(1);
  loom_attribute_t f = loom_attr_f64(1.0);
  EXPECT_FALSE(loom_attribute_equal(&i, &f));
}

TEST(AttributeEqual, I64ArraySameContent) {
  int64_t values_a[] = {1, 2, 3, 4};
  int64_t values_b[] = {1, 2, 3, 4};
  loom_attribute_t a = loom_attr_i64_array(values_a, 4);
  loom_attribute_t b = loom_attr_i64_array(values_b, 4);
  // Different pointers, same content — must be equal.
  EXPECT_NE(a.i64_array, b.i64_array);
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
}

TEST(AttributeEqual, I64ArrayDifferentContent) {
  int64_t values_a[] = {1, 2, 3};
  int64_t values_b[] = {1, 2, 4};
  loom_attribute_t a = loom_attr_i64_array(values_a, 3);
  loom_attribute_t b = loom_attr_i64_array(values_b, 3);
  EXPECT_FALSE(loom_attribute_equal(&a, &b));
}

TEST(AttributeEqual, I64ArrayDifferentCount) {
  int64_t values_a[] = {1, 2, 3};
  int64_t values_b[] = {1, 2};
  loom_attribute_t a = loom_attr_i64_array(values_a, 3);
  loom_attribute_t b = loom_attr_i64_array(values_b, 2);
  EXPECT_FALSE(loom_attribute_equal(&a, &b));
}

TEST(AttributeEqual, I64ArraySamePointer) {
  int64_t values[] = {10, 20};
  loom_attribute_t a = loom_attr_i64_array(values, 2);
  loom_attribute_t b = loom_attr_i64_array(values, 2);
  // Same pointer — fast path.
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
}

TEST(AttributeEqual, PredicateListSameContent) {
  loom_predicate_t preds_a[2];
  memset(preds_a, 0, sizeof(preds_a));
  preds_a[0].kind = LOOM_PREDICATE_MULTIPLE_OF;
  preds_a[0].arg_count = 2;
  preds_a[0].args[0] = 100;
  preds_a[0].args[1] = 16;
  preds_a[1].kind = LOOM_PREDICATE_GE;
  preds_a[1].arg_count = 2;
  preds_a[1].args[0] = 100;
  preds_a[1].args[1] = 1;

  loom_predicate_t preds_b[2];
  memcpy(preds_b, preds_a, sizeof(preds_a));

  loom_attribute_t a = loom_attr_predicate_list(preds_a, 2);
  loom_attribute_t b = loom_attr_predicate_list(preds_b, 2);
  EXPECT_NE(a.predicate_list, b.predicate_list);
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
}

TEST(AttributeEqual, PredicateListDifferentContent) {
  loom_predicate_t preds_a[1];
  memset(preds_a, 0, sizeof(preds_a));
  preds_a[0].kind = LOOM_PREDICATE_MULTIPLE_OF;
  preds_a[0].arg_count = 2;
  preds_a[0].args[0] = 100;
  preds_a[0].args[1] = 16;

  loom_predicate_t preds_b[1];
  memcpy(preds_b, preds_a, sizeof(preds_a));
  preds_b[0].args[1] = 32;  // Different multiplier.

  loom_attribute_t a = loom_attr_predicate_list(preds_a, 1);
  loom_attribute_t b = loom_attr_predicate_list(preds_b, 1);
  EXPECT_FALSE(loom_attribute_equal(&a, &b));
}

TEST(AttributeEqual, DictSameContent) {
  loom_named_attr_t entries_a[2];
  memset(entries_a, 0, sizeof(entries_a));
  entries_a[0].name_id = 1;
  entries_a[0].value = loom_attr_i64(42);
  entries_a[1].name_id = 2;
  entries_a[1].value = loom_attr_bool(true);

  loom_named_attr_t entries_b[2];
  memcpy(entries_b, entries_a, sizeof(entries_a));

  loom_attribute_t a = loom_make_canonical_attr_dict(entries_a, 2);
  loom_attribute_t b = loom_make_canonical_attr_dict(entries_b, 2);
  EXPECT_NE(a.dict_entries, b.dict_entries);
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
}

TEST(AttributeEqual, DictDifferentName) {
  loom_named_attr_t entries_a[1];
  memset(entries_a, 0, sizeof(entries_a));
  entries_a[0].name_id = 1;
  entries_a[0].value = loom_attr_i64(42);

  loom_named_attr_t entries_b[1];
  memcpy(entries_b, entries_a, sizeof(entries_a));
  entries_b[0].name_id = 99;

  loom_attribute_t a = loom_make_canonical_attr_dict(entries_a, 1);
  loom_attribute_t b = loom_make_canonical_attr_dict(entries_b, 1);
  EXPECT_FALSE(loom_attribute_equal(&a, &b));
}

TEST(AttributeEqual, DictWithNestedArray) {
  int64_t arr_a[] = {5, 10, 15};
  int64_t arr_b[] = {5, 10, 15};

  loom_named_attr_t entries_a[1];
  memset(entries_a, 0, sizeof(entries_a));
  entries_a[0].name_id = 1;
  entries_a[0].value = loom_attr_i64_array(arr_a, 3);

  loom_named_attr_t entries_b[1];
  memset(entries_b, 0, sizeof(entries_b));
  entries_b[0].name_id = 1;
  entries_b[0].value = loom_attr_i64_array(arr_b, 3);

  loom_attribute_t a = loom_make_canonical_attr_dict(entries_a, 1);
  loom_attribute_t b = loom_make_canonical_attr_dict(entries_b, 1);
  // Nested array attribute with different pointers, same content.
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
}

TEST(AttributeHash, EqualAttributesEqualHash) {
  // Inline.
  loom_attribute_t i1 = loom_attr_i64(42);
  loom_attribute_t i2 = loom_attr_i64(42);
  EXPECT_EQ(loom_attribute_hash(&i1), loom_attribute_hash(&i2));

  // I64 array with different pointers.
  int64_t values_a[] = {1, 2, 3};
  int64_t values_b[] = {1, 2, 3};
  loom_attribute_t a = loom_attr_i64_array(values_a, 3);
  loom_attribute_t b = loom_attr_i64_array(values_b, 3);
  EXPECT_EQ(loom_attribute_hash(&a), loom_attribute_hash(&b));
}

TEST(AttributeHash, DifferentAttributesDifferentHash) {
  loom_attribute_t a = loom_attr_i64(42);
  loom_attribute_t b = loom_attr_i64(43);
  // Not guaranteed by spec, but extremely likely with FNV-1a.
  EXPECT_NE(loom_attribute_hash(&a), loom_attribute_hash(&b));

  int64_t values_a[] = {1, 2, 3};
  int64_t values_b[] = {1, 2, 4};
  loom_attribute_t arr_a = loom_attr_i64_array(values_a, 3);
  loom_attribute_t arr_b = loom_attr_i64_array(values_b, 3);
  EXPECT_NE(loom_attribute_hash(&arr_a), loom_attribute_hash(&arr_b));
}

TEST(AttributeHash, DifferentKindsDifferentHash) {
  loom_attribute_t i = loom_attr_i64(0);
  loom_attribute_t b = loom_attr_bool(false);
  // Both have zero payload but different kinds.
  EXPECT_NE(loom_attribute_hash(&i), loom_attribute_hash(&b));
}

//===----------------------------------------------------------------------===//
// Attribute DICT depth limiting
//===----------------------------------------------------------------------===//

// Builds a chain of DICT attributes nested to |depth| levels.
// The innermost value is an i64(42). Each level wraps the previous
// in a single-entry DICT. Returns the outermost attribute.
// All loom_named_attr_t storage is allocated from |buffer|, which
// must have room for |depth| entries.
static loom_attribute_t make_nested_dict(loom_named_attr_t* buffer, int depth) {
  loom_attribute_t inner = loom_attr_i64(42);
  for (int i = 0; i < depth; ++i) {
    memset(&buffer[i], 0, sizeof(buffer[i]));
    buffer[i].name_id = (loom_string_id_t)(i + 1);
    buffer[i].value = inner;
    inner = loom_make_canonical_attr_dict(&buffer[i], 1);
  }
  return inner;
}

TEST(AttributeEqual, DeeplyNestedDictDoesNotCrash) {
  // Build two identical DICT chains deeper than the recursion limit.
  // The function must not stack-overflow; it should return false
  // (conservative) for the portion beyond the depth limit.
  static const int kDepth = 200;
  loom_named_attr_t buffer_a[kDepth];
  loom_named_attr_t buffer_b[kDepth];
  loom_attribute_t a = make_nested_dict(buffer_a, kDepth);
  loom_attribute_t b = make_nested_dict(buffer_b, kDepth);

  // Should not crash. The result is false because the comparison
  // bails at the depth limit rather than confirming equality.
  EXPECT_FALSE(loom_attribute_equal(&a, &b));
}

TEST(AttributeEqual, ShallowNestedDictStillWorks) {
  // Nesting within the depth limit should compare correctly.
  loom_named_attr_t buffer_a[3];
  loom_named_attr_t buffer_b[3];
  loom_attribute_t a = make_nested_dict(buffer_a, 3);
  loom_attribute_t b = make_nested_dict(buffer_b, 3);
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
}

TEST(AttributeHash, DeeplyNestedDictDoesNotCrash) {
  static const int kDepth = 200;
  loom_named_attr_t buffer[kDepth];
  loom_attribute_t attr = make_nested_dict(buffer, kDepth);

  // Should not crash. Returns some deterministic hash.
  uint32_t hash = loom_attribute_hash(&attr);
  EXPECT_EQ(hash, loom_attribute_hash(&attr));
}

TEST(AttributeHash, ShallowNestedDictConsistent) {
  loom_named_attr_t buffer_a[3];
  loom_named_attr_t buffer_b[3];
  loom_attribute_t a = make_nested_dict(buffer_a, 3);
  loom_attribute_t b = make_nested_dict(buffer_b, 3);

  // Within depth limit: equal attributes must have equal hashes.
  EXPECT_TRUE(loom_attribute_equal(&a, &b));
  EXPECT_EQ(loom_attribute_hash(&a), loom_attribute_hash(&b));
}

}  // namespace
}  // namespace loom
