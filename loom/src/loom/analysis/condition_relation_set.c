// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/condition_relation_set.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/structural_hash.h"

// Every subset fits in the existing ID when the domain fits its bit width.
enum { LOOM_CONDITION_RELATION_SET_IMMEDIATE_CAPACITY = 32 };

static bool loom_condition_relation_set_is_immediate(uint32_t value_count) {
  return value_count <= LOOM_CONDITION_RELATION_SET_IMMEDIATE_CAPACITY;
}

typedef struct loom_condition_relation_dense_set_t {
  // Immutable dense words for this interned set.
  const uint64_t* words;

  // Structural hash of words, retained to make table growth word-independent.
  uint32_t hash;

  // Next set ID in the content-hash collision chain.
  loom_condition_relation_set_id_t next_id;
} loom_condition_relation_dense_set_t;

union loom_condition_relation_set_index_node_t {
  struct {
    // Set ID for the lower half of this tree level.
    loom_condition_relation_set_id_t left;

    // Set ID for the upper half of this tree level.
    loom_condition_relation_set_id_t right;
  } branch;

  // Members within one 64-value leaf chunk.
  uint64_t bits;
};

static_assert(sizeof(loom_condition_relation_set_index_node_t) == 8,
              "published condition-relation set nodes must remain 8 bytes");

struct loom_condition_relation_set_builder_t {
  // Arena owning every field referenced by this builder.
  iree_arena_allocator_t* arena;

  // Interned dense sets, excluding empty and inline singleton sets.
  loom_condition_relation_dense_set_t* nodes;

  // Number of initialized entries in nodes.
  iree_host_size_t node_count;

  // Allocated entry capacity of nodes.
  iree_host_size_t node_capacity;

  // Set-ID heads for dense content-hash collision chains.
  loom_condition_relation_set_id_t* buckets;

  // Power-of-two entry count in buckets.
  iree_host_size_t bucket_count;

  // Reusable bitmap for interning and set algebra.
  uint64_t* work_words;

  // Number of values in the finite set domain.
  uint32_t value_count;

  // Number of 64-bit words in each dense bitmap.
  uint32_t word_count;

  // Root level required by the published query tree.
  uint8_t root_level;
};

static bool loom_condition_relation_set_is_singleton(
    uint32_t value_count, loom_condition_relation_set_id_t set) {
  return set != LOOM_CONDITION_RELATION_SET_EMPTY && set <= value_count;
}

static loom_condition_relation_set_id_t loom_condition_relation_set_singleton(
    uint32_t value_count, uint32_t value) {
  IREE_ASSERT_LT(value, value_count);
  return value + 1;
}

static uint32_t loom_condition_relation_set_singleton_value(
    uint32_t value_count, loom_condition_relation_set_id_t set) {
  IREE_ASSERT_TRUE(loom_condition_relation_set_is_singleton(value_count, set));
  return set - 1;
}

static iree_host_size_t loom_condition_relation_set_node_index(
    uint32_t value_count, loom_condition_relation_set_id_t set) {
  IREE_ASSERT_GT(set, value_count);
  return (iree_host_size_t)(set - value_count - 1);
}

static const loom_condition_relation_dense_set_t*
loom_condition_relation_set_builder_node(
    const loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_id_t set) {
  const iree_host_size_t index =
      loom_condition_relation_set_node_index(builder->value_count, set);
  IREE_ASSERT_LT(index, builder->node_count);
  return &builder->nodes[index];
}

static const loom_condition_relation_set_index_node_t*
loom_condition_relation_set_index_node(
    const loom_condition_relation_set_index_t* index,
    loom_condition_relation_set_id_t set) {
  const iree_host_size_t node_index =
      loom_condition_relation_set_node_index(index->value_count, set);
  IREE_ASSERT_LT(node_index, index->node_count);
  return &index->nodes[node_index];
}

static uint32_t loom_condition_relation_set_hash_words(
    const loom_condition_relation_set_builder_t* builder,
    const uint64_t* words) {
  uint32_t hash = loom_structural_hash_initialize();
  for (uint32_t i = 0; i < builder->word_count; ++i) {
    hash = loom_structural_hash_mix_u64(hash, words[i]);
  }
  return loom_structural_hash_finalize(hash);
}

#if IREE_HAVE_ATTRIBUTE(minsize)
__attribute__((minsize))
#endif
IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_condition_relation_set_rehash_dense(
    loom_condition_relation_set_builder_t* builder,
    iree_host_size_t bucket_count) {
  loom_condition_relation_set_id_t* buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, bucket_count, sizeof(*buckets), (void**)&buckets));
  memset(buckets, 0, bucket_count * sizeof(*buckets));
  for (iree_host_size_t i = 0; i < builder->node_count; ++i) {
    loom_condition_relation_dense_set_t* node = &builder->nodes[i];
    const loom_condition_relation_set_id_t set =
        builder->value_count + (loom_condition_relation_set_id_t)i + 1;
    const iree_host_size_t bucket = node->hash & (bucket_count - 1);
    node->next_id = buckets[bucket];
    buckets[bucket] = set;
  }
  builder->buckets = buckets;
  builder->bucket_count = bucket_count;
  return iree_ok_status();
}

static iree_status_t loom_condition_relation_set_prepare_dense_insert(
    loom_condition_relation_set_builder_t* builder) {
  if (builder->node_count >= UINT32_MAX - builder->value_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "condition-relation set ID capacity exceeded");
  }
  if (builder->node_count == builder->node_capacity) {
    const iree_host_size_t minimum_capacity =
        iree_max((iree_host_size_t)64, builder->node_count + 1);
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        builder->arena, builder->node_count, minimum_capacity,
        sizeof(*builder->nodes), &builder->node_capacity,
        (void**)&builder->nodes));
  }
  if (builder->node_count + 1 <=
      builder->bucket_count - builder->bucket_count / 4) {
    return iree_ok_status();
  }
  iree_host_size_t bucket_count = 0;
  if (!iree_host_size_checked_mul(builder->bucket_count, 2, &bucket_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "condition-relation set hash capacity exceeded");
  }
  return loom_condition_relation_set_rehash_dense(builder, bucket_count);
}

static uint32_t loom_condition_relation_set_count_words(
    const loom_condition_relation_set_builder_t* builder,
    const uint64_t* words) {
  uint32_t count = 0;
  for (uint32_t i = 0; i < builder->word_count; ++i) {
    count += (uint32_t)iree_math_count_ones_u64(words[i]);
  }
  return count;
}

static uint32_t loom_condition_relation_set_first_value(
    const loom_condition_relation_set_builder_t* builder,
    const uint64_t* words) {
  for (uint32_t i = 0; i < builder->word_count; ++i) {
    if (words[i] != 0) {
      return i * 64 + (uint32_t)iree_math_count_trailing_zeros_u64(words[i]);
    }
  }
  IREE_ASSERT_FALSE(true);
  return 0;
}

static iree_status_t loom_condition_relation_set_intern_words(
    loom_condition_relation_set_builder_t* builder, const uint64_t* words,
    loom_condition_relation_set_id_t* out_set) {
  const uint32_t member_count =
      loom_condition_relation_set_count_words(builder, words);
  if (member_count == 0) {
    *out_set = LOOM_CONDITION_RELATION_SET_EMPTY;
    return iree_ok_status();
  }
  if (member_count == 1) {
    *out_set = loom_condition_relation_set_singleton(
        builder->value_count,
        loom_condition_relation_set_first_value(builder, words));
    return iree_ok_status();
  }

  if (builder->bucket_count == 0) {
    IREE_RETURN_IF_ERROR(loom_condition_relation_set_rehash_dense(builder, 64));
  }
  const uint32_t hash = loom_condition_relation_set_hash_words(builder, words);
  iree_host_size_t bucket = hash & (builder->bucket_count - 1);
  for (loom_condition_relation_set_id_t set = builder->buckets[bucket];
       set != LOOM_CONDITION_RELATION_SET_EMPTY;
       set = loom_condition_relation_set_builder_node(builder, set)->next_id) {
    const loom_condition_relation_dense_set_t* node =
        loom_condition_relation_set_builder_node(builder, set);
    if (node->hash == hash &&
        memcmp(node->words, words, builder->word_count * sizeof(*words)) == 0) {
      *out_set = set;
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(
      loom_condition_relation_set_prepare_dense_insert(builder));
  bucket = hash & (builder->bucket_count - 1);
  uint64_t* copied_words = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(builder->arena, builder->word_count,
                                sizeof(*copied_words), (void**)&copied_words));
  memcpy(copied_words, words, builder->word_count * sizeof(*copied_words));
  const loom_condition_relation_set_id_t set =
      builder->value_count +
      (loom_condition_relation_set_id_t)builder->node_count + 1;
  builder->nodes[builder->node_count++] = (loom_condition_relation_dense_set_t){
      .words = copied_words,
      .hash = hash,
      .next_id = builder->buckets[bucket],
  };
  builder->buckets[bucket] = set;
  *out_set = set;
  return iree_ok_status();
}

static bool loom_condition_relation_set_builder_contains(
    const loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_id_t set, uint32_t value) {
  if (loom_condition_relation_set_is_singleton(builder->value_count, set)) {
    return loom_condition_relation_set_singleton_value(builder->value_count,
                                                       set) == value;
  }
  const loom_condition_relation_dense_set_t* node =
      loom_condition_relation_set_builder_node(builder, set);
  return (node->words[value / 64] & (UINT64_C(1) << (value % 64))) != 0;
}

static void loom_condition_relation_set_builder_copy_words(
    const loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_id_t set, uint64_t* out_words) {
  if (loom_condition_relation_set_is_singleton(builder->value_count, set)) {
    memset(out_words, 0, builder->word_count * sizeof(*out_words));
    const uint32_t value =
        loom_condition_relation_set_singleton_value(builder->value_count, set);
    out_words[value / 64] = UINT64_C(1) << (value % 64);
    return;
  }
  memcpy(out_words,
         loom_condition_relation_set_builder_node(builder, set)->words,
         builder->word_count * sizeof(*out_words));
}

IREE_ATTRIBUTE_NOINLINE
iree_status_t loom_condition_relation_set_builder_allocate(
    uint32_t value_count, iree_arena_allocator_t* scratch_arena,
    loom_condition_relation_set_builder_t** out_builder) {
  *out_builder = NULL;
  if (value_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "condition-relation value ID capacity exceeded");
  }
  loom_condition_relation_set_builder_t* builder = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(scratch_arena, sizeof(*builder), (void**)&builder));
  memset(builder, 0, sizeof(*builder));
  builder->arena = scratch_arena;
  builder->value_count = value_count;
  builder->word_count = value_count / 64 + (value_count % 64 != 0);
  uint32_t leaf_capacity = iree_max(1u, builder->word_count);
  while ((UINT32_C(1) << builder->root_level) < leaf_capacity) {
    ++builder->root_level;
  }
  if (!loom_condition_relation_set_is_immediate(value_count)) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, builder->word_count, sizeof(*builder->work_words),
        (void**)&builder->work_words));
  }
  *out_builder = builder;
  return iree_ok_status();
}

loom_condition_relation_set_id_t loom_condition_relation_set_builder_singleton(
    const loom_condition_relation_set_builder_t* builder, uint32_t value) {
  IREE_ASSERT_LT(value, builder->value_count);
  return loom_condition_relation_set_is_immediate(builder->value_count)
             ? UINT32_C(1) << value
             : value + 1;
}

IREE_ATTRIBUTE_NOINLINE
iree_status_t loom_condition_relation_set_builder_intern(
    loom_condition_relation_set_builder_t* builder, const uint32_t* values,
    iree_host_size_t value_count, loom_condition_relation_set_id_t* out_set) {
  if (loom_condition_relation_set_is_immediate(builder->value_count)) {
    loom_condition_relation_set_id_t bits = 0;
    for (iree_host_size_t i = 0; i < value_count; ++i) {
      IREE_ASSERT_LT(values[i], builder->value_count);
      bits |= UINT32_C(1) << values[i];
    }
    *out_set = bits;
    return iree_ok_status();
  }
  if (value_count == 0) {
    *out_set = LOOM_CONDITION_RELATION_SET_EMPTY;
    return iree_ok_status();
  }
  memset(builder->work_words, 0,
         builder->word_count * sizeof(*builder->work_words));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    IREE_ASSERT_LT(values[i], builder->value_count);
    builder->work_words[values[i] / 64] |= UINT64_C(1) << (values[i] % 64);
  }
  return loom_condition_relation_set_intern_words(builder, builder->work_words,
                                                  out_set);
}

IREE_ATTRIBUTE_NOINLINE
iree_status_t loom_condition_relation_set_builder_union(
    loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_id_t left,
    loom_condition_relation_set_id_t right,
    loom_condition_relation_set_id_t* out_set) {
  if (loom_condition_relation_set_is_immediate(builder->value_count)) {
    *out_set = left | right;
    return iree_ok_status();
  }
  if (left == LOOM_CONDITION_RELATION_SET_EMPTY) {
    *out_set = right;
    return iree_ok_status();
  }
  if (right == LOOM_CONDITION_RELATION_SET_EMPTY || left == right) {
    *out_set = left;
    return iree_ok_status();
  }
  loom_condition_relation_set_builder_copy_words(builder, left,
                                                 builder->work_words);
  if (loom_condition_relation_set_is_singleton(builder->value_count, right)) {
    const uint32_t value = loom_condition_relation_set_singleton_value(
        builder->value_count, right);
    builder->work_words[value / 64] |= UINT64_C(1) << (value % 64);
  } else {
    const uint64_t* right_words =
        loom_condition_relation_set_builder_node(builder, right)->words;
    for (uint32_t i = 0; i < builder->word_count; ++i) {
      builder->work_words[i] |= right_words[i];
    }
  }
  return loom_condition_relation_set_intern_words(builder, builder->work_words,
                                                  out_set);
}

IREE_ATTRIBUTE_NOINLINE
iree_status_t loom_condition_relation_set_builder_intersection(
    loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_id_t left,
    loom_condition_relation_set_id_t right,
    loom_condition_relation_set_id_t* out_set) {
  if (loom_condition_relation_set_is_immediate(builder->value_count)) {
    *out_set = left & right;
    return iree_ok_status();
  }
  if (left == LOOM_CONDITION_RELATION_SET_EMPTY ||
      right == LOOM_CONDITION_RELATION_SET_EMPTY) {
    *out_set = LOOM_CONDITION_RELATION_SET_EMPTY;
    return iree_ok_status();
  }
  if (left == right) {
    *out_set = left;
    return iree_ok_status();
  }
  if (loom_condition_relation_set_is_singleton(builder->value_count, left)) {
    const uint32_t value =
        loom_condition_relation_set_singleton_value(builder->value_count, left);
    *out_set =
        loom_condition_relation_set_builder_contains(builder, right, value)
            ? left
            : LOOM_CONDITION_RELATION_SET_EMPTY;
    return iree_ok_status();
  }
  if (loom_condition_relation_set_is_singleton(builder->value_count, right)) {
    const uint32_t value = loom_condition_relation_set_singleton_value(
        builder->value_count, right);
    *out_set =
        loom_condition_relation_set_builder_contains(builder, left, value)
            ? right
            : LOOM_CONDITION_RELATION_SET_EMPTY;
    return iree_ok_status();
  }
  const uint64_t* left_words =
      loom_condition_relation_set_builder_node(builder, left)->words;
  const uint64_t* right_words =
      loom_condition_relation_set_builder_node(builder, right)->words;
  for (uint32_t i = 0; i < builder->word_count; ++i) {
    builder->work_words[i] = left_words[i] & right_words[i];
  }
  return loom_condition_relation_set_intern_words(builder, builder->work_words,
                                                  out_set);
}

IREE_ATTRIBUTE_NOINLINE
iree_status_t loom_condition_relation_set_builder_difference(
    loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_id_t left,
    loom_condition_relation_set_id_t right,
    loom_condition_relation_set_id_t* out_set) {
  if (loom_condition_relation_set_is_immediate(builder->value_count)) {
    *out_set = left & ~right;
    return iree_ok_status();
  }
  if (left == LOOM_CONDITION_RELATION_SET_EMPTY || left == right) {
    *out_set = LOOM_CONDITION_RELATION_SET_EMPTY;
    return iree_ok_status();
  }
  if (right == LOOM_CONDITION_RELATION_SET_EMPTY) {
    *out_set = left;
    return iree_ok_status();
  }
  if (loom_condition_relation_set_is_singleton(builder->value_count, left)) {
    const uint32_t value =
        loom_condition_relation_set_singleton_value(builder->value_count, left);
    *out_set =
        loom_condition_relation_set_builder_contains(builder, right, value)
            ? LOOM_CONDITION_RELATION_SET_EMPTY
            : left;
    return iree_ok_status();
  }
  loom_condition_relation_set_builder_copy_words(builder, left,
                                                 builder->work_words);
  if (loom_condition_relation_set_is_singleton(builder->value_count, right)) {
    const uint32_t value = loom_condition_relation_set_singleton_value(
        builder->value_count, right);
    builder->work_words[value / 64] &= ~(UINT64_C(1) << (value % 64));
  } else {
    const uint64_t* right_words =
        loom_condition_relation_set_builder_node(builder, right)->words;
    for (uint32_t i = 0; i < builder->word_count; ++i) {
      builder->work_words[i] &= ~right_words[i];
    }
  }
  return loom_condition_relation_set_intern_words(builder, builder->work_words,
                                                  out_set);
}

static bool loom_condition_relation_set_visit_immediate(
    loom_condition_relation_set_id_t bits,
    loom_condition_relation_set_visit_fn_t visit, void* user_data) {
  while (bits) {
    const uint32_t value = (uint32_t)iree_math_count_trailing_zeros_u32(bits);
    if (!visit(user_data, value)) {
      return false;
    }
    bits &= bits - 1;
  }
  return true;
}

IREE_ATTRIBUTE_NOINLINE
bool loom_condition_relation_set_builder_for_each_while(
    const loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_id_t set,
    loom_condition_relation_set_visit_fn_t visit, void* user_data) {
  if (loom_condition_relation_set_is_immediate(builder->value_count)) {
    return loom_condition_relation_set_visit_immediate(set, visit, user_data);
  }
  if (set == LOOM_CONDITION_RELATION_SET_EMPTY) {
    return true;
  }
  if (loom_condition_relation_set_is_singleton(builder->value_count, set)) {
    return visit(user_data, loom_condition_relation_set_singleton_value(
                                builder->value_count, set));
  }
  const loom_condition_relation_dense_set_t* node =
      loom_condition_relation_set_builder_node(builder, set);
  for (uint32_t i = 0; i < builder->word_count; ++i) {
    uint64_t bits = node->words[i];
    while (bits != 0) {
      const uint32_t bit = (uint32_t)iree_math_count_trailing_zeros_u64(bits);
      if (!visit(user_data, i * 64 + bit)) {
        return false;
      }
      bits &= bits - 1;
    }
  }
  return true;
}

typedef struct loom_condition_relation_publish_node_t {
  // Query payload copied to the immutable index.
  loom_condition_relation_set_index_node_t query;

  // Next node ID in the structural-hash collision chain.
  loom_condition_relation_set_id_t next_id;

  // Structural hash in the low 27 bits and tree level in the high 5 bits.
  uint32_t key;
} loom_condition_relation_publish_node_t;

static_assert(sizeof(loom_condition_relation_publish_node_t) == 16,
              "publication hash nodes must remain 16 bytes");

typedef struct loom_condition_relation_publish_level_entry_t {
  // Chunk position at the current tree level.
  uint32_t position;

  // Set ID representing this position.
  loom_condition_relation_set_id_t set;
} loom_condition_relation_publish_level_entry_t;

typedef struct loom_condition_relation_publish_builder_t {
  // Dense source builder and publication scratch owner.
  loom_condition_relation_set_builder_t* source;

  // Hash-consed nodes accumulated for the immutable index.
  loom_condition_relation_publish_node_t* nodes;

  // Number of initialized entries in nodes.
  iree_host_size_t node_count;

  // Allocated entry capacity of nodes.
  iree_host_size_t node_capacity;

  // Set-ID heads for structural-hash collision chains.
  loom_condition_relation_set_id_t* buckets;

  // Power-of-two entry count in buckets.
  iree_host_size_t bucket_count;

  // Alternating sparse levels used to reduce one bitmap to its root.
  loom_condition_relation_publish_level_entry_t* levels[2];
} loom_condition_relation_publish_builder_t;

static uint32_t loom_condition_relation_publish_node_key(
    const loom_condition_relation_set_index_node_t* query, uint8_t level) {
  uint32_t hash = loom_structural_hash_initialize();
  if (level == 0) {
    hash = loom_structural_hash_mix_u64(hash, query->bits);
  } else {
    hash = loom_structural_hash_mix_u32(hash, query->branch.left);
    hash = loom_structural_hash_mix_u32(hash, query->branch.right);
  }
  return (loom_structural_hash_finalize(hash) & UINT32_C(0x07FFFFFF)) |
         ((uint32_t)level << 27);
}

static bool loom_condition_relation_publish_nodes_equal(
    const loom_condition_relation_publish_node_t* left,
    const loom_condition_relation_publish_node_t* right) {
  if (left->key != right->key) {
    return false;
  }
  return (left->key >> 27) == 0
             ? left->query.bits == right->query.bits
             : left->query.branch.left == right->query.branch.left &&
                   left->query.branch.right == right->query.branch.right;
}

#if IREE_HAVE_ATTRIBUTE(minsize)
__attribute__((minsize))
#endif
IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_condition_relation_publish_rehash(
    loom_condition_relation_publish_builder_t* builder,
    iree_host_size_t bucket_count) {
  loom_condition_relation_set_id_t* buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(builder->source->arena,
                                                 bucket_count, sizeof(*buckets),
                                                 (void**)&buckets));
  memset(buckets, 0, bucket_count * sizeof(*buckets));
  for (iree_host_size_t i = 0; i < builder->node_count; ++i) {
    loom_condition_relation_publish_node_t* node = &builder->nodes[i];
    const loom_condition_relation_set_id_t set =
        builder->source->value_count + (loom_condition_relation_set_id_t)i + 1;
    const iree_host_size_t bucket =
        loom_structural_hash_finalize(node->key) & (bucket_count - 1);
    node->next_id = buckets[bucket];
    buckets[bucket] = set;
  }
  builder->buckets = buckets;
  builder->bucket_count = bucket_count;
  return iree_ok_status();
}

static iree_status_t loom_condition_relation_publish_prepare_insert(
    loom_condition_relation_publish_builder_t* builder) {
  if (builder->node_count >= UINT32_MAX - builder->source->value_count) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "published condition-relation set ID capacity exceeded");
  }
  if (builder->node_count == builder->node_capacity) {
    const iree_host_size_t minimum_capacity =
        iree_max((iree_host_size_t)64, builder->node_count + 1);
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        builder->source->arena, builder->node_count, minimum_capacity,
        sizeof(*builder->nodes), &builder->node_capacity,
        (void**)&builder->nodes));
  }
  if (builder->node_count + 1 <=
      builder->bucket_count - builder->bucket_count / 4) {
    return iree_ok_status();
  }
  iree_host_size_t bucket_count = 0;
  if (!iree_host_size_checked_mul(builder->bucket_count, 2, &bucket_count)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "published condition-relation set hash capacity exceeded");
  }
  return loom_condition_relation_publish_rehash(builder, bucket_count);
}

static iree_status_t loom_condition_relation_publish_intern_node(
    loom_condition_relation_publish_builder_t* builder,
    loom_condition_relation_set_index_node_t query, uint8_t level,
    loom_condition_relation_set_id_t* out_set) {
  loom_condition_relation_publish_node_t candidate = {
      .query = query,
      .key = loom_condition_relation_publish_node_key(&query, level),
  };
  if (builder->bucket_count == 0) {
    IREE_RETURN_IF_ERROR(loom_condition_relation_publish_rehash(builder, 64));
  }
  const uint32_t hash = loom_structural_hash_finalize(candidate.key);
  iree_host_size_t bucket = hash & (builder->bucket_count - 1);
  for (loom_condition_relation_set_id_t set = builder->buckets[bucket];
       set != LOOM_CONDITION_RELATION_SET_EMPTY;) {
    const iree_host_size_t index = loom_condition_relation_set_node_index(
        builder->source->value_count, set);
    const loom_condition_relation_publish_node_t* node = &builder->nodes[index];
    if (loom_condition_relation_publish_nodes_equal(node, &candidate)) {
      *out_set = set;
      return iree_ok_status();
    }
    set = node->next_id;
  }

  IREE_RETURN_IF_ERROR(loom_condition_relation_publish_prepare_insert(builder));
  bucket = hash & (builder->bucket_count - 1);
  const loom_condition_relation_set_id_t set =
      builder->source->value_count +
      (loom_condition_relation_set_id_t)builder->node_count + 1;
  candidate.next_id = builder->buckets[bucket];
  builder->nodes[builder->node_count++] = candidate;
  builder->buckets[bucket] = set;
  *out_set = set;
  return iree_ok_status();
}

static iree_status_t loom_condition_relation_publish_leaf(
    loom_condition_relation_publish_builder_t* builder, uint32_t position,
    uint64_t bits, loom_condition_relation_set_id_t* out_set) {
  if ((bits & (bits - 1)) == 0) {
    const uint32_t value =
        position * 64 + (uint32_t)iree_math_count_trailing_zeros_u64(bits);
    IREE_ASSERT_LT(value, builder->source->value_count);
    *out_set = loom_condition_relation_set_singleton(
        builder->source->value_count, value);
    return iree_ok_status();
  }
  return loom_condition_relation_publish_intern_node(
      builder, (loom_condition_relation_set_index_node_t){.bits = bits}, 0,
      out_set);
}

static iree_status_t loom_condition_relation_publish_branch(
    loom_condition_relation_publish_builder_t* builder, uint8_t level,
    loom_condition_relation_set_id_t left,
    loom_condition_relation_set_id_t right,
    loom_condition_relation_set_id_t* out_set) {
  if (left == LOOM_CONDITION_RELATION_SET_EMPTY &&
      loom_condition_relation_set_is_singleton(builder->source->value_count,
                                               right)) {
    *out_set = right;
    return iree_ok_status();
  }
  if (right == LOOM_CONDITION_RELATION_SET_EMPTY &&
      loom_condition_relation_set_is_singleton(builder->source->value_count,
                                               left)) {
    *out_set = left;
    return iree_ok_status();
  }
  return loom_condition_relation_publish_intern_node(
      builder,
      (loom_condition_relation_set_index_node_t){
          .branch = {.left = left, .right = right},
      },
      level, out_set);
}

static iree_status_t loom_condition_relation_publish_dense_set(
    loom_condition_relation_publish_builder_t* builder,
    const loom_condition_relation_dense_set_t* dense,
    loom_condition_relation_set_id_t* out_root) {
  iree_host_size_t level_count = 0;
  for (uint32_t position = 0; position < builder->source->word_count;
       ++position) {
    if (dense->words[position] == 0) {
      continue;
    }
    loom_condition_relation_set_id_t leaf = LOOM_CONDITION_RELATION_SET_EMPTY;
    IREE_RETURN_IF_ERROR(loom_condition_relation_publish_leaf(
        builder, position, dense->words[position], &leaf));
    builder->levels[0][level_count++] =
        (loom_condition_relation_publish_level_entry_t){
            .position = position,
            .set = leaf,
        };
  }
  IREE_ASSERT_GT(level_count, 0u);

  uint32_t source_level = 0;
  for (uint8_t level = 1; level <= builder->source->root_level; ++level) {
    loom_condition_relation_publish_level_entry_t* source_entries =
        builder->levels[source_level];
    loom_condition_relation_publish_level_entry_t* target_entries =
        builder->levels[source_level ^ 1u];
    iree_host_size_t target_count = 0;
    for (iree_host_size_t position = 0; position < level_count;) {
      const uint32_t parent = source_entries[position].position / 2;
      loom_condition_relation_set_id_t left = LOOM_CONDITION_RELATION_SET_EMPTY;
      loom_condition_relation_set_id_t right =
          LOOM_CONDITION_RELATION_SET_EMPTY;
      while (position < level_count &&
             source_entries[position].position / 2 == parent) {
        if ((source_entries[position].position & 1u) == 0) {
          left = source_entries[position].set;
        } else {
          right = source_entries[position].set;
        }
        ++position;
      }
      loom_condition_relation_set_id_t branch =
          LOOM_CONDITION_RELATION_SET_EMPTY;
      IREE_RETURN_IF_ERROR(loom_condition_relation_publish_branch(
          builder, level, left, right, &branch));
      target_entries[target_count++] =
          (loom_condition_relation_publish_level_entry_t){
              .position = parent,
              .set = branch,
          };
    }
    level_count = target_count;
    source_level ^= 1u;
  }
  IREE_ASSERT_EQ(level_count, 1u);
  *out_root = builder->levels[source_level][0].set;
  return iree_ok_status();
}

#if IREE_HAVE_ATTRIBUTE(minsize)
__attribute__((minsize))
#endif
IREE_ATTRIBUTE_NOINLINE iree_status_t
loom_condition_relation_set_builder_publish(
    loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_id_t* roots, iree_host_size_t root_count,
    iree_arena_allocator_t* arena,
    loom_condition_relation_set_index_t* out_index) {
  *out_index = (loom_condition_relation_set_index_t){0};
  IREE_ASSERT_NE(builder->arena, arena);
  if (loom_condition_relation_set_is_immediate(builder->value_count)) {
    out_index->value_count = builder->value_count;
    return iree_ok_status();
  }

  loom_condition_relation_set_id_t* remap = NULL;
  if (builder->node_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->node_count, sizeof(*remap), (void**)&remap));
    memset(remap, 0, builder->node_count * sizeof(*remap));
  }
  loom_condition_relation_publish_builder_t publisher = {
      .source = builder,
  };
  if (builder->word_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->word_count, sizeof(*publisher.levels[0]),
        (void**)&publisher.levels[0]));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->word_count, sizeof(*publisher.levels[1]),
        (void**)&publisher.levels[1]));
  }

  for (iree_host_size_t i = 0; i < root_count; ++i) {
    if (roots[i] <= builder->value_count) {
      continue;
    }
    const iree_host_size_t source_index =
        loom_condition_relation_set_node_index(builder->value_count, roots[i]);
    IREE_ASSERT_LT(source_index, builder->node_count);
    if (remap[source_index] == LOOM_CONDITION_RELATION_SET_EMPTY) {
      IREE_RETURN_IF_ERROR(loom_condition_relation_publish_dense_set(
          &publisher, &builder->nodes[source_index], &remap[source_index]));
    }
  }

  loom_condition_relation_set_index_node_t* nodes = NULL;
  if (publisher.node_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, publisher.node_count, sizeof(*nodes), (void**)&nodes));
    for (iree_host_size_t i = 0; i < publisher.node_count; ++i) {
      nodes[i] = publisher.nodes[i].query;
    }
  }
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    if (roots[i] > builder->value_count) {
      roots[i] = remap[loom_condition_relation_set_node_index(
          builder->value_count, roots[i])];
    }
  }
  *out_index = (loom_condition_relation_set_index_t){
      .nodes = nodes,
      .value_count = builder->value_count,
      .node_count = (uint32_t)publisher.node_count,
      .root_level = builder->root_level,
  };
  return iree_ok_status();
}

IREE_ATTRIBUTE_NOINLINE
bool loom_condition_relation_set_index_contains(
    const loom_condition_relation_set_index_t* index,
    loom_condition_relation_set_id_t set, uint32_t value) {
  if (set == LOOM_CONDITION_RELATION_SET_EMPTY || value >= index->value_count) {
    return false;
  }
  if (loom_condition_relation_set_is_immediate(index->value_count)) {
    return (set & (UINT32_C(1) << value)) != 0;
  }
  uint8_t level = index->root_level;
  const uint32_t chunk = value / 64;
  while (true) {
    if (loom_condition_relation_set_is_singleton(index->value_count, set)) {
      return loom_condition_relation_set_singleton_value(index->value_count,
                                                         set) == value;
    }
    const loom_condition_relation_set_index_node_t* node =
        loom_condition_relation_set_index_node(index, set);
    if (level == 0) {
      return (node->bits & (UINT64_C(1) << (value % 64))) != 0;
    }
    set = (chunk & (UINT32_C(1) << (level - 1))) != 0 ? node->branch.right
                                                      : node->branch.left;
    if (set == LOOM_CONDITION_RELATION_SET_EMPTY) {
      return false;
    }
    --level;
  }
}

static bool loom_condition_relation_set_index_visit(
    const loom_condition_relation_set_index_t* index,
    loom_condition_relation_set_id_t set, uint8_t level, uint32_t base_chunk,
    loom_condition_relation_set_visit_fn_t visit, void* user_data) {
  if (set == LOOM_CONDITION_RELATION_SET_EMPTY) {
    return true;
  }
  if (loom_condition_relation_set_is_singleton(index->value_count, set)) {
    return visit(user_data, loom_condition_relation_set_singleton_value(
                                index->value_count, set));
  }
  const loom_condition_relation_set_index_node_t* node =
      loom_condition_relation_set_index_node(index, set);
  if (level == 0) {
    uint64_t bits = node->bits;
    while (bits != 0) {
      const uint32_t bit = (uint32_t)iree_math_count_trailing_zeros_u64(bits);
      if (!visit(user_data, base_chunk * 64 + bit)) {
        return false;
      }
      bits &= bits - 1;
    }
    return true;
  }
  return loom_condition_relation_set_index_visit(index, node->branch.left,
                                                 level - 1, base_chunk, visit,
                                                 user_data) &&
         loom_condition_relation_set_index_visit(
             index, node->branch.right, level - 1,
             base_chunk + (UINT32_C(1) << (level - 1)), visit, user_data);
}

IREE_ATTRIBUTE_NOINLINE
bool loom_condition_relation_set_index_for_each_while(
    const loom_condition_relation_set_index_t* index,
    loom_condition_relation_set_id_t set,
    loom_condition_relation_set_visit_fn_t visit, void* user_data) {
  if (loom_condition_relation_set_is_immediate(index->value_count)) {
    return loom_condition_relation_set_visit_immediate(set, visit, user_data);
  }
  return loom_condition_relation_set_index_visit(index, set, index->root_level,
                                                 0, visit, user_data);
}
