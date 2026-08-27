// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/attribute.h"

#include <string.h>

#include "loom/ir/structural_hash.h"

bool loom_attr_matches_scalar_type(loom_attribute_t attr,
                                   loom_scalar_type_t scalar_type,
                                   loom_attr_kind_t* out_expected_kind) {
  loom_attr_kind_t expected_kind = LOOM_ATTR_ANY;
  bool matches = false;
  if (scalar_type == LOOM_SCALAR_TYPE_I1) {
    expected_kind = LOOM_ATTR_BOOL;
    matches = attr.kind == LOOM_ATTR_BOOL ||
              (attr.kind == LOOM_ATTR_I64 && (attr.i64 == 0 || attr.i64 == 1));
  } else if (scalar_type == LOOM_SCALAR_TYPE_INDEX ||
             scalar_type == LOOM_SCALAR_TYPE_OFFSET ||
             loom_scalar_type_is_integer(scalar_type)) {
    expected_kind = LOOM_ATTR_I64;
    matches = attr.kind == LOOM_ATTR_I64;
  } else if (loom_scalar_type_is_float(scalar_type)) {
    expected_kind = LOOM_ATTR_F64;
    matches = attr.kind == LOOM_ATTR_F64;
  }
  if (out_expected_kind) *out_expected_kind = expected_kind;
  return matches;
}

const char* loom_predicate_kind_name(uint8_t kind) {
  switch ((loom_predicate_kind_t)kind) {
    case LOOM_PREDICATE_EQ:
      return "eq";
    case LOOM_PREDICATE_NE:
      return "ne";
    case LOOM_PREDICATE_LT:
      return "lt";
    case LOOM_PREDICATE_LE:
      return "le";
    case LOOM_PREDICATE_GT:
      return "gt";
    case LOOM_PREDICATE_GE:
      return "ge";
    case LOOM_PREDICATE_MUL:
      return "mul";
    case LOOM_PREDICATE_MIN:
      return "min";
    case LOOM_PREDICATE_MAX:
      return "max";
    case LOOM_PREDICATE_POW2:
      return "pow2";
    case LOOM_PREDICATE_RANGE:
      return "range";
    case LOOM_PREDICATE_NOT_NAN:
      return "not_nan";
    case LOOM_PREDICATE_NOT_INF:
      return "not_inf";
    case LOOM_PREDICATE_FINITE:
      return "finite";
    case LOOM_PREDICATE_COUNT_:
      return NULL;
  }
  return NULL;
}

uint8_t loom_predicate_kind_argument_count(uint8_t kind) {
  switch ((loom_predicate_kind_t)kind) {
    case LOOM_PREDICATE_EQ:
    case LOOM_PREDICATE_NE:
    case LOOM_PREDICATE_LT:
    case LOOM_PREDICATE_LE:
    case LOOM_PREDICATE_GT:
    case LOOM_PREDICATE_GE:
    case LOOM_PREDICATE_MUL:
    case LOOM_PREDICATE_MIN:
    case LOOM_PREDICATE_MAX:
      return 2;
    case LOOM_PREDICATE_POW2:
    case LOOM_PREDICATE_NOT_NAN:
    case LOOM_PREDICATE_NOT_INF:
    case LOOM_PREDICATE_FINITE:
      return 1;
    case LOOM_PREDICATE_RANGE:
      return 3;
    case LOOM_PREDICATE_COUNT_:
      return UINT8_MAX;
  }
  return UINT8_MAX;
}

iree_status_t loom_signed_enum_set_canonical_word_count(
    loom_signed_enum_set_t set, iree_host_size_t* out_word_count) {
  if (out_word_count == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "canonical word-count output is NULL");
  }
  *out_word_count = 0;
  if (set.word_count > LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "signed enum set has %" PRIhsz " words per polarity, max %u",
        set.word_count, (unsigned)LOOM_SIGNED_ENUM_SET_MAX_WORD_COUNT);
  }
  if (set.word_count == 0) return iree_ok_status();
  if (set.words == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty signed enum set has a NULL word pointer");
  }

  const uint64_t* negative_words = set.words + set.word_count;
  for (iree_host_size_t i = 0; i < set.word_count; ++i) {
    if ((set.words[i] & negative_words[i]) != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "signed enum set word %" PRIhsz
          " contains contradictory positive and negative assertions",
          i);
    }
  }

  iree_host_size_t canonical_word_count = set.word_count;
  while (canonical_word_count > 0 && set.words[canonical_word_count - 1] == 0 &&
         negative_words[canonical_word_count - 1] == 0) {
    --canonical_word_count;
  }
  *out_word_count = canonical_word_count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Attribute equality and hashing
//===----------------------------------------------------------------------===//

static bool loom_attribute_equal_impl(const loom_attribute_t* a,
                                      const loom_attribute_t* b,
                                      iree_host_size_t depth) {
  if (a->kind != b->kind) return false;
  switch ((loom_attr_kind_t)a->kind) {
    case LOOM_ATTR_I64_ARRAY:
      if (a->count != b->count) return false;
      if (a->i64_array == b->i64_array) return true;
      return memcmp(a->i64_array, b->i64_array,
                    (iree_host_size_t)a->count * sizeof(int64_t)) == 0;
    case LOOM_ATTR_ENUM_ARRAY:
      if (a->count != b->count) return false;
      if (a->count == 0) return true;
      if (a->enum_array == NULL || b->enum_array == NULL) return false;
      if (a->enum_array == b->enum_array) return true;
      return memcmp(a->enum_array, b->enum_array, a->count) == 0;
    case LOOM_ATTR_SIGNED_ENUM_SET:
      if (a->count != b->count) return false;
      if (a->count == 0) return true;
      if (a->signed_enum_set_words == NULL ||
          b->signed_enum_set_words == NULL) {
        return false;
      }
      if (a->signed_enum_set_words == b->signed_enum_set_words) return true;
      return memcmp(a->signed_enum_set_words, b->signed_enum_set_words,
                    (iree_host_size_t)a->count * 2 * sizeof(uint64_t)) == 0;
    case LOOM_ATTR_SYMBOL_ARRAY:
    case LOOM_ATTR_SYMBOL_SET: {
      if (a->count != b->count) return false;
      if (a->count == 0) return true;
      if (a->symbol_refs == b->symbol_refs) return true;
      return memcmp(a->symbol_refs, b->symbol_refs,
                    (iree_host_size_t)a->count * sizeof(loom_symbol_ref_t)) ==
             0;
    }
    case LOOM_ATTR_PREDICATE_LIST:
      if (a->count != b->count) return false;
      if (a->predicate_list == b->predicate_list) return true;
      return memcmp(a->predicate_list, b->predicate_list,
                    (iree_host_size_t)a->count * sizeof(loom_predicate_t)) == 0;
    case LOOM_ATTR_BYTES:
      if (a->reserved_1 != b->reserved_1) return false;
      if (a->reserved_1 == 0) return true;
      if (a->bytes == NULL || b->bytes == NULL) return false;
      if (a->bytes == b->bytes) return true;
      return memcmp(a->bytes, b->bytes, a->reserved_1) == 0;
    case LOOM_ATTR_DICT:
      if (a->count != b->count) return false;
      if (a->dict_entries == b->dict_entries) return true;
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) return false;
      for (uint16_t i = 0; i < a->count; ++i) {
        if (a->dict_entries[i].name_id != b->dict_entries[i].name_id) {
          return false;
        }
        if (!loom_attribute_equal_impl(&a->dict_entries[i].value,
                                       &b->dict_entries[i].value, depth + 1)) {
          return false;
        }
      }
      return true;
    case LOOM_ATTR_PARAMETERIZED:
      if (a->reserved_1 != b->reserved_1 || a->count != b->count) return false;
      if (a->parameterized_slots == b->parameterized_slots) return true;
      if (a->parameterized_slots == NULL || b->parameterized_slots == NULL ||
          depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return false;
      }
      for (uint16_t i = 0; i < a->count; ++i) {
        if (!loom_attribute_equal_impl(&a->parameterized_slots[i],
                                       &b->parameterized_slots[i], depth + 1)) {
          return false;
        }
      }
      return true;
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      if (a->count != b->count) return false;
      if (a->parameterized_array == b->parameterized_array) return true;
      if (a->parameterized_array == NULL || b->parameterized_array == NULL ||
          depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return false;
      }
      for (uint16_t i = 0; i < a->count; ++i) {
        if (!loom_attribute_equal_impl(&a->parameterized_array[i],
                                       &b->parameterized_array[i], depth + 1)) {
          return false;
        }
      }
      return true;
    default:
      return memcmp(a, b, sizeof(loom_attribute_t)) == 0;
  }
}

bool loom_attribute_equal(const loom_attribute_t* a,
                          const loom_attribute_t* b) {
  return loom_attribute_equal_impl(a, b, 0);
}

static uint32_t loom_attribute_hash_impl(const loom_attribute_t* attr,
                                         iree_host_size_t depth) {
  uint32_t hash = loom_structural_hash_initialize();
  hash = loom_structural_hash_mix_u8(hash, attr->kind);
  switch ((loom_attr_kind_t)attr->kind) {
    case LOOM_ATTR_I64_ARRAY:
      hash = loom_structural_hash_mix_u16(hash, attr->count);
      hash = loom_structural_hash_mix_bytes(
          hash, attr->i64_array,
          (iree_host_size_t)attr->count * sizeof(int64_t));
      break;
    case LOOM_ATTR_ENUM_ARRAY:
      hash = loom_structural_hash_mix_u16(hash, attr->count);
      if (attr->count != 0 && attr->enum_array != NULL) {
        hash =
            loom_structural_hash_mix_bytes(hash, attr->enum_array, attr->count);
      }
      break;
    case LOOM_ATTR_SIGNED_ENUM_SET:
      hash = loom_structural_hash_mix_u16(hash, attr->count);
      if (attr->count != 0 && attr->signed_enum_set_words != NULL) {
        hash = loom_structural_hash_mix_bytes(
            hash, attr->signed_enum_set_words,
            (iree_host_size_t)attr->count * 2 * sizeof(uint64_t));
      }
      break;
    case LOOM_ATTR_SYMBOL_ARRAY:
    case LOOM_ATTR_SYMBOL_SET: {
      hash = loom_structural_hash_mix_u16(hash, attr->count);
      if (attr->count != 0) {
        hash = loom_structural_hash_mix_bytes(
            hash, attr->symbol_refs,
            (iree_host_size_t)attr->count * sizeof(loom_symbol_ref_t));
      }
      break;
    }
    case LOOM_ATTR_PREDICATE_LIST:
      hash = loom_structural_hash_mix_u16(hash, attr->count);
      hash = loom_structural_hash_mix_bytes(
          hash, attr->predicate_list,
          (iree_host_size_t)attr->count * sizeof(loom_predicate_t));
      break;
    case LOOM_ATTR_BYTES:
      hash = loom_structural_hash_mix_u32(hash, attr->reserved_1);
      if (attr->reserved_1 != 0 && attr->bytes != NULL) {
        hash =
            loom_structural_hash_mix_bytes(hash, attr->bytes, attr->reserved_1);
      }
      break;
    case LOOM_ATTR_DICT:
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        hash = loom_structural_hash_mix_bytes(hash, attr,
                                              sizeof(loom_attribute_t));
        break;
      }
      hash = loom_structural_hash_mix_u16(hash, attr->count);
      for (uint16_t i = 0; i < attr->count; ++i) {
        hash =
            loom_structural_hash_mix_u32(hash, attr->dict_entries[i].name_id);
        uint32_t value_hash =
            loom_attribute_hash_impl(&attr->dict_entries[i].value, depth + 1);
        hash = loom_structural_hash_mix_u32(hash, value_hash);
      }
      break;
    case LOOM_ATTR_PARAMETERIZED:
      hash = loom_structural_hash_mix_u32(hash, attr->reserved_1);
      hash = loom_structural_hash_mix_u16(hash, attr->count);
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH ||
          (attr->count > 0 && attr->parameterized_slots == NULL)) {
        hash = loom_structural_hash_mix_u64(
            hash, (uint64_t)(uintptr_t)attr->parameterized_slots);
        break;
      }
      for (uint16_t i = 0; i < attr->count; ++i) {
        uint32_t slot_hash =
            loom_attribute_hash_impl(&attr->parameterized_slots[i], depth + 1);
        hash = loom_structural_hash_mix_u32(hash, slot_hash);
      }
      break;
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      hash = loom_structural_hash_mix_u16(hash, attr->count);
      if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH ||
          (attr->count > 0 && attr->parameterized_array == NULL)) {
        hash = loom_structural_hash_mix_u64(
            hash, (uint64_t)(uintptr_t)attr->parameterized_array);
        break;
      }
      for (uint16_t i = 0; i < attr->count; ++i) {
        uint32_t element_hash =
            loom_attribute_hash_impl(&attr->parameterized_array[i], depth + 1);
        hash = loom_structural_hash_mix_u32(hash, element_hash);
      }
      break;
    default:
      hash =
          loom_structural_hash_mix_bytes(hash, attr, sizeof(loom_attribute_t));
      break;
  }
  return loom_structural_hash_finalize(hash);
}

uint32_t loom_attribute_hash(const loom_attribute_t* attr) {
  return loom_attribute_hash_impl(attr, 0);
}
