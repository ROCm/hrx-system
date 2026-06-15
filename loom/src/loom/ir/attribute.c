// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/attribute.h"

#include <string.h>

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
      return 1;
    case LOOM_PREDICATE_RANGE:
      return 3;
    case LOOM_PREDICATE_COUNT_:
      return UINT8_MAX;
  }
  return UINT8_MAX;
}

//===----------------------------------------------------------------------===//
// Attribute equality and hashing
//===----------------------------------------------------------------------===//

// FNV-1a hash over a byte range, folded into a running hash.
static uint32_t loom_hash_bytes(const void* data, iree_host_size_t length,
                                uint32_t hash) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < length; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

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
    case LOOM_ATTR_PREDICATE_LIST:
      if (a->count != b->count) return false;
      if (a->predicate_list == b->predicate_list) return true;
      return memcmp(a->predicate_list, b->predicate_list,
                    (iree_host_size_t)a->count * sizeof(loom_predicate_t)) == 0;
    case LOOM_ATTR_DICT:
      if (a->count != b->count) return false;
      if (a->dict_entries == b->dict_entries) return true;
      if (depth >= LOOM_ATTR_DICT_MAX_NESTING_DEPTH) return false;
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
  uint32_t hash = 2166136261u;
  hash = loom_hash_bytes(&attr->kind, 1, hash);
  switch ((loom_attr_kind_t)attr->kind) {
    case LOOM_ATTR_I64_ARRAY:
      hash = loom_hash_bytes(&attr->count, sizeof(attr->count), hash);
      hash = loom_hash_bytes(attr->i64_array,
                             (iree_host_size_t)attr->count * sizeof(int64_t),
                             hash);
      break;
    case LOOM_ATTR_PREDICATE_LIST:
      hash = loom_hash_bytes(&attr->count, sizeof(attr->count), hash);
      hash = loom_hash_bytes(
          attr->predicate_list,
          (iree_host_size_t)attr->count * sizeof(loom_predicate_t), hash);
      break;
    case LOOM_ATTR_DICT:
      if (depth >= LOOM_ATTR_DICT_MAX_NESTING_DEPTH) {
        hash = loom_hash_bytes(attr, sizeof(loom_attribute_t), hash);
        break;
      }
      hash = loom_hash_bytes(&attr->count, sizeof(attr->count), hash);
      for (uint16_t i = 0; i < attr->count; ++i) {
        hash = loom_hash_bytes(&attr->dict_entries[i].name_id,
                               sizeof(loom_string_id_t), hash);
        uint32_t value_hash =
            loom_attribute_hash_impl(&attr->dict_entries[i].value, depth + 1);
        hash = loom_hash_bytes(&value_hash, sizeof(value_hash), hash);
      }
      break;
    default:
      hash = loom_hash_bytes(attr, sizeof(loom_attribute_t), hash);
      break;
  }
  return hash;
}

uint32_t loom_attribute_hash(const loom_attribute_t* attr) {
  return loom_attribute_hash_impl(attr, 0);
}
