// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/condition_relation_set.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/structural_hash.h"
#include "loom/util/adaptive_sort.h"
#include "loom/util/segmented_storage.h"

enum {
  LOOM_CONDITION_RELATION_SET_SEGMENT_SHIFT = 6,
  LOOM_CONDITION_RELATION_SET_SEGMENT_CAPACITY =
      1u << LOOM_CONDITION_RELATION_SET_SEGMENT_SHIFT,
  LOOM_CONDITION_RELATION_SET_SEGMENT_MASK =
      LOOM_CONDITION_RELATION_SET_SEGMENT_CAPACITY - 1u,
};

static_assert((uint64_t)LOOM_CONDITION_RELATION_SET_SEGMENT_CAPACITY *
                      LOOM_SEGMENTED_STORAGE_MAX_SEGMENT_COUNT >=
                  (UINT64_C(1) << 32),
              "condition set storage must cover every compound node ID");

// Every subset fits in the existing ID when the domain fits its bit width.
enum { LOOM_CONDITION_RELATION_SET_IMMEDIATE_CAPACITY = 32 };

static bool loom_condition_relation_set_is_immediate(uint32_t value_count) {
  return value_count <= LOOM_CONDITION_RELATION_SET_IMMEDIATE_CAPACITY;
}

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

typedef struct loom_condition_relation_set_build_node_t {
  // Immutable query payload. Compound children always precede their parent.
  loom_condition_relation_set_index_node_t query;
  // Next node ID in the structural-hash collision chain.
  loom_condition_relation_set_id_t next_id;
  // Structural hash in the low 27 bits and tree level in the high 5 bits.
  uint32_t key;
} loom_condition_relation_set_build_node_t;

static_assert(sizeof(loom_condition_relation_set_build_node_t) == 16,
              "construction nodes must remain 16 bytes");

struct loom_condition_relation_set_builder_t {
  // Arena owning every field referenced by this builder.
  iree_arena_allocator_t* arena;
  // Lazy stable node directory, absent for empty and singleton-only builders.
  loom_segmented_storage_t* nodes;
  // Number of initialized entries in nodes.
  iree_host_size_t node_count;
  // Set-ID heads for structural-hash collision chains.
  loom_condition_relation_set_id_t* buckets;
  // Power-of-two entry count in buckets.
  iree_host_size_t bucket_count;
  // Reusable copy for inputs whose 64-value chunks are not ordered.
  uint32_t* unordered_values;
  // Allocated entry capacity of unordered_values.
  iree_host_size_t unordered_value_capacity;
  // Number of values in the finite set domain.
  uint32_t value_count;
  // Root level shared by every non-inline set ID.
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

static loom_condition_relation_set_build_node_t*
loom_condition_relation_set_builder_node_at(
    const loom_condition_relation_set_builder_t* builder,
    iree_host_size_t index) {
  loom_condition_relation_set_build_node_t* segment =
      loom_segmented_storage_segment(
          builder->nodes,
          (uint32_t)(index >> LOOM_CONDITION_RELATION_SET_SEGMENT_SHIFT));
  return &segment[index & LOOM_CONDITION_RELATION_SET_SEGMENT_MASK];
}

static const loom_condition_relation_set_build_node_t*
loom_condition_relation_set_builder_node(
    const loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_id_t set) {
  return loom_condition_relation_set_builder_node_at(
      builder,
      loom_condition_relation_set_node_index(builder->value_count, set));
}

static const loom_condition_relation_set_index_node_t*
loom_condition_relation_set_index_node(
    const loom_condition_relation_set_index_t* index,
    loom_condition_relation_set_id_t set) {
  return &index->nodes[loom_condition_relation_set_node_index(
      index->value_count, set)];
}

static uint32_t loom_condition_relation_set_node_key(
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

static bool loom_condition_relation_set_nodes_equal(
    const loom_condition_relation_set_build_node_t* left,
    const loom_condition_relation_set_build_node_t* right) {
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
IREE_ATTRIBUTE_NOINLINE static iree_status_t loom_condition_relation_set_rehash(
    loom_condition_relation_set_builder_t* builder,
    iree_host_size_t bucket_count) {
  loom_condition_relation_set_id_t* buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, bucket_count, sizeof(*buckets), (void**)&buckets));
  memset(buckets, 0, bucket_count * sizeof(*buckets));
  for (iree_host_size_t i = 0; i < builder->node_count; ++i) {
    loom_condition_relation_set_build_node_t* node =
        loom_condition_relation_set_builder_node_at(builder, i);
    const loom_condition_relation_set_id_t set =
        builder->value_count + (loom_condition_relation_set_id_t)i + 1;
    const iree_host_size_t bucket =
        loom_structural_hash_finalize(node->key) & (bucket_count - 1);
    node->next_id = buckets[bucket];
    buckets[bucket] = set;
  }
  builder->buckets = buckets;
  builder->bucket_count = bucket_count;
  return iree_ok_status();
}

static iree_status_t loom_condition_relation_set_prepare_insert(
    loom_condition_relation_set_builder_t* builder) {
  if (builder->node_count >= UINT32_MAX - builder->value_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "condition-relation set ID capacity exceeded");
  }
  if (!builder->nodes) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        builder->arena, sizeof(*builder->nodes), (void**)&builder->nodes));
    loom_segmented_storage_initialize(
        sizeof(loom_condition_relation_set_build_node_t) *
            LOOM_CONDITION_RELATION_SET_SEGMENT_CAPACITY,
        iree_alignof(loom_condition_relation_set_build_node_t), builder->nodes);
  }
  if (builder->node_count >> LOOM_CONDITION_RELATION_SET_SEGMENT_SHIFT ==
      builder->nodes->segment_count) {
    void* segment = NULL;
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
        builder->nodes, builder->arena, &segment));
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
  return loom_condition_relation_set_rehash(builder, bucket_count);
}

static iree_status_t loom_condition_relation_set_intern_node(
    loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_index_node_t query, uint8_t level,
    loom_condition_relation_set_id_t* out_set) {
  loom_condition_relation_set_build_node_t candidate = {
      .query = query,
      .key = loom_condition_relation_set_node_key(&query, level),
  };
  if (builder->bucket_count == 0) {
    IREE_RETURN_IF_ERROR(loom_condition_relation_set_rehash(builder, 64));
  }
  const uint32_t hash = loom_structural_hash_finalize(candidate.key);
  iree_host_size_t bucket = hash & (builder->bucket_count - 1);
  for (loom_condition_relation_set_id_t set = builder->buckets[bucket];
       set != LOOM_CONDITION_RELATION_SET_EMPTY;) {
    const loom_condition_relation_set_build_node_t* node =
        loom_condition_relation_set_builder_node(builder, set);
    if (loom_condition_relation_set_nodes_equal(node, &candidate)) {
      *out_set = set;
      return iree_ok_status();
    }
    set = node->next_id;
  }
  IREE_RETURN_IF_ERROR(loom_condition_relation_set_prepare_insert(builder));
  bucket = hash & (builder->bucket_count - 1);
  const loom_condition_relation_set_id_t set =
      builder->value_count +
      (loom_condition_relation_set_id_t)builder->node_count + 1;
  candidate.next_id = builder->buckets[bucket];
  *loom_condition_relation_set_builder_node_at(builder, builder->node_count++) =
      candidate;
  builder->buckets[bucket] = set;
  *out_set = set;
  return iree_ok_status();
}

static iree_status_t loom_condition_relation_set_intern_leaf(
    loom_condition_relation_set_builder_t* builder, uint32_t base_chunk,
    uint64_t bits, loom_condition_relation_set_id_t* out_set) {
  if (bits == 0) {
    *out_set = LOOM_CONDITION_RELATION_SET_EMPTY;
    return iree_ok_status();
  }
  if ((bits & (bits - 1)) == 0) {
    *out_set = loom_condition_relation_set_singleton(
        builder->value_count,
        base_chunk * 64 + (uint32_t)iree_math_count_trailing_zeros_u64(bits));
    return iree_ok_status();
  }
  return loom_condition_relation_set_intern_node(
      builder, (loom_condition_relation_set_index_node_t){.bits = bits}, 0,
      out_set);
}

static iree_status_t loom_condition_relation_set_intern_branch(
    loom_condition_relation_set_builder_t* builder, uint8_t level,
    loom_condition_relation_set_id_t left,
    loom_condition_relation_set_id_t right,
    loom_condition_relation_set_id_t* out_set) {
  if (left == LOOM_CONDITION_RELATION_SET_EMPTY &&
      right <= builder->value_count) {
    *out_set = right;
    return iree_ok_status();
  }
  if (right == LOOM_CONDITION_RELATION_SET_EMPTY &&
      left <= builder->value_count) {
    *out_set = left;
    return iree_ok_status();
  }
  return loom_condition_relation_set_intern_node(
      builder,
      (loom_condition_relation_set_index_node_t){
          .branch = {.left = left, .right = right},
      },
      level, out_set);
}

static loom_condition_relation_set_id_t loom_condition_relation_set_child(
    const loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_id_t set, uint8_t level, uint32_t side) {
  if (set == LOOM_CONDITION_RELATION_SET_EMPTY) {
    return set;
  }
  if (loom_condition_relation_set_is_singleton(builder->value_count, set)) {
    const uint32_t chunk = (set - 1) / 64;
    return ((chunk >> (level - 1)) & 1) == side ? set : 0;
  }
  const loom_condition_relation_set_index_node_t node =
      loom_condition_relation_set_builder_node(builder, set)->query;
  return side == 0 ? node.branch.left : node.branch.right;
}

static uint64_t loom_condition_relation_set_leaf_bits(
    const loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_id_t set) {
  if (set == LOOM_CONDITION_RELATION_SET_EMPTY) {
    return 0;
  }
  if (loom_condition_relation_set_is_singleton(builder->value_count, set)) {
    return UINT64_C(1) << ((set - 1) % 64);
  }
  return loom_condition_relation_set_builder_node(builder, set)->query.bits;
}

typedef enum loom_condition_relation_set_operation_e {
  LOOM_CONDITION_RELATION_SET_UNION,
  LOOM_CONDITION_RELATION_SET_INTERSECTION,
  LOOM_CONDITION_RELATION_SET_DIFFERENCE,
} loom_condition_relation_set_operation_t;

// Unchanged subtrees return their existing IDs. Recursion is bounded by the
// finite uint32_t domain, independently of the number or shape of input sets.
static iree_status_t loom_condition_relation_set_combine(
    loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_operation_t operation,
    loom_condition_relation_set_id_t left,
    loom_condition_relation_set_id_t right, uint8_t level, uint32_t base_chunk,
    loom_condition_relation_set_id_t* out_set) {
  if (left == right) {
    *out_set = operation == LOOM_CONDITION_RELATION_SET_DIFFERENCE ? 0 : left;
    return iree_ok_status();
  }
  if (left == LOOM_CONDITION_RELATION_SET_EMPTY) {
    *out_set = operation == LOOM_CONDITION_RELATION_SET_UNION ? right : 0;
    return iree_ok_status();
  }
  if (right == LOOM_CONDITION_RELATION_SET_EMPTY) {
    *out_set = operation == LOOM_CONDITION_RELATION_SET_INTERSECTION ? 0 : left;
    return iree_ok_status();
  }
  if (level == 0) {
    const uint64_t left_bits =
        loom_condition_relation_set_leaf_bits(builder, left);
    const uint64_t right_bits =
        loom_condition_relation_set_leaf_bits(builder, right);
    const uint64_t bits =
        operation == LOOM_CONDITION_RELATION_SET_UNION ? left_bits | right_bits
        : operation == LOOM_CONDITION_RELATION_SET_INTERSECTION
            ? left_bits & right_bits
            : left_bits & ~right_bits;
    return loom_condition_relation_set_intern_leaf(builder, base_chunk, bits,
                                                   out_set);
  }
  // Child IDs remain valid as recursive calls append new immutable nodes.
  const loom_condition_relation_set_id_t left_lower =
      loom_condition_relation_set_child(builder, left, level, 0);
  const loom_condition_relation_set_id_t left_upper =
      loom_condition_relation_set_child(builder, left, level, 1);
  const loom_condition_relation_set_id_t right_lower =
      loom_condition_relation_set_child(builder, right, level, 0);
  const loom_condition_relation_set_id_t right_upper =
      loom_condition_relation_set_child(builder, right, level, 1);
  loom_condition_relation_set_id_t lower = 0;
  IREE_RETURN_IF_ERROR(loom_condition_relation_set_combine(
      builder, operation, left_lower, right_lower, level - 1, base_chunk,
      &lower));
  loom_condition_relation_set_id_t upper = 0;
  IREE_RETURN_IF_ERROR(loom_condition_relation_set_combine(
      builder, operation, left_upper, right_upper, level - 1,
      base_chunk + (UINT32_C(1) << (level - 1)), &upper));
  return loom_condition_relation_set_intern_branch(builder, level, lower, upper,
                                                   out_set);
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
  *builder = (loom_condition_relation_set_builder_t){
      .arena = scratch_arena,
      .value_count = value_count,
  };
  const uint32_t chunk_count = value_count / 64 + (value_count % 64 != 0);
  while ((UINT32_C(1) << builder->root_level) < chunk_count) {
    ++builder->root_level;
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
  return loom_condition_relation_set_combine(
      builder, LOOM_CONDITION_RELATION_SET_UNION, left, right,
      builder->root_level, 0, out_set);
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
  return loom_condition_relation_set_combine(
      builder, LOOM_CONDITION_RELATION_SET_INTERSECTION, left, right,
      builder->root_level, 0, out_set);
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
  return loom_condition_relation_set_combine(
      builder, LOOM_CONDITION_RELATION_SET_DIFFERENCE, left, right,
      builder->root_level, 0, out_set);
}

static iree_status_t loom_condition_relation_set_intern_ordered(
    loom_condition_relation_set_builder_t* builder, const uint32_t* values,
    iree_host_size_t value_count, uint8_t level, uint32_t base_chunk,
    loom_condition_relation_set_id_t* out_set) {
  if (value_count == 0) {
    *out_set = LOOM_CONDITION_RELATION_SET_EMPTY;
    return iree_ok_status();
  }
  if (value_count == 1) {
    *out_set =
        loom_condition_relation_set_singleton(builder->value_count, values[0]);
    return iree_ok_status();
  }
  if (level == 0) {
    uint64_t bits = 0;
    for (iree_host_size_t i = 0; i < value_count; ++i) {
      IREE_ASSERT_LT(values[i], builder->value_count);
      bits |= UINT64_C(1) << (values[i] % 64);
    }
    return loom_condition_relation_set_intern_leaf(builder, base_chunk, bits,
                                                   out_set);
  }
  const uint32_t upper_base = base_chunk + (UINT32_C(1) << (level - 1));
  iree_host_size_t lower = 0;
  iree_host_size_t upper = value_count;
  while (lower < upper) {
    const iree_host_size_t middle = lower + (upper - lower) / 2;
    if (values[middle] / 64 < upper_base) {
      lower = middle + 1;
    } else {
      upper = middle;
    }
  }
  loom_condition_relation_set_id_t left = 0;
  IREE_RETURN_IF_ERROR(loom_condition_relation_set_intern_ordered(
      builder, values, lower, level - 1, base_chunk, &left));
  loom_condition_relation_set_id_t right = 0;
  IREE_RETURN_IF_ERROR(loom_condition_relation_set_intern_ordered(
      builder, values + lower, value_count - lower, level - 1, upper_base,
      &right));
  return loom_condition_relation_set_intern_branch(builder, level, left, right,
                                                   out_set);
}

static bool loom_condition_relation_set_value_less(const uint32_t* left,
                                                   const uint32_t* right) {
  return *left < *right;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_condition_relation_set_sort_values, uint32_t,
                          loom_condition_relation_set_value_less)

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
  // Chunk-ordered inputs build only the final tree. Arbitrary inputs use one
  // reusable copy rather than retaining the prefixes of repeated insertion.
  if (builder->root_level != 0) {
    for (iree_host_size_t i = 1; i < value_count; ++i) {
      if (values[i - 1] / 64 <= values[i] / 64) {
        continue;
      }
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          builder->arena, 0, value_count, sizeof(*builder->unordered_values),
          &builder->unordered_value_capacity,
          (void**)&builder->unordered_values));
      memcpy(builder->unordered_values, values, value_count * sizeof(*values));
      loom_condition_relation_set_sort_values(builder->unordered_values,
                                              value_count);
      values = builder->unordered_values;
      break;
    }
  }
  return loom_condition_relation_set_intern_ordered(
      builder, values, value_count, builder->root_level, 0, out_set);
}

static bool loom_condition_relation_set_builder_visit(
    const loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_id_t set, uint8_t level, uint32_t base_chunk,
    loom_condition_relation_set_visit_fn_t visit, void* user_data) {
  if (set == LOOM_CONDITION_RELATION_SET_EMPTY) {
    return true;
  }
  if (loom_condition_relation_set_is_singleton(builder->value_count, set)) {
    return visit(user_data, set - 1);
  }
  const loom_condition_relation_set_index_node_t node =
      loom_condition_relation_set_builder_node(builder, set)->query;
  if (level == 0) {
    uint64_t bits = node.bits;
    while (bits != 0) {
      const uint32_t bit = (uint32_t)iree_math_count_trailing_zeros_u64(bits);
      if (!visit(user_data, base_chunk * 64 + bit)) {
        return false;
      }
      bits &= bits - 1;
    }
    return true;
  }
  return loom_condition_relation_set_builder_visit(builder, node.branch.left,
                                                   level - 1, base_chunk, visit,
                                                   user_data) &&
         loom_condition_relation_set_builder_visit(
             builder, node.branch.right, level - 1,
             base_chunk + (UINT32_C(1) << (level - 1)), visit, user_data);
}

IREE_ATTRIBUTE_NOINLINE
bool loom_condition_relation_set_builder_for_each_while(
    const loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_id_t set,
    loom_condition_relation_set_visit_fn_t visit, void* user_data) {
  if (loom_condition_relation_set_is_immediate(builder->value_count)) {
    return loom_condition_relation_set_visit_immediate(set, visit, user_data);
  }
  return loom_condition_relation_set_builder_visit(
      builder, set, builder->root_level, 0, visit, user_data);
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

  // Compound children precede their parents. The last selected root bounds
  // the entire reachable prefix; construction after it cannot contribute.
  iree_host_size_t prefix_count = 0;
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    if (roots[i] > builder->value_count) {
      prefix_count = iree_max(
          prefix_count, (iree_host_size_t)(roots[i] - builder->value_count));
    }
  }
  loom_condition_relation_set_id_t* remap = NULL;
  if (prefix_count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, prefix_count, sizeof(*remap), (void**)&remap));
    memset(remap, 0, prefix_count * sizeof(*remap));
  }
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    if (roots[i] > builder->value_count) {
      remap[loom_condition_relation_set_node_index(builder->value_count,
                                                   roots[i])] = 1;
    }
  }
  uint32_t node_count = 0;
  for (iree_host_size_t i = prefix_count; i > 0; --i) {
    if (!remap[i - 1]) {
      continue;
    }
    ++node_count;
    const loom_condition_relation_set_build_node_t* node =
        loom_condition_relation_set_builder_node_at(builder, i - 1);
    if ((node->key >> 27) == 0) {
      continue;
    }
    const loom_condition_relation_set_id_t children[] = {
        node->query.branch.left, node->query.branch.right};
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(children); ++j) {
      if (children[j] > builder->value_count) {
        remap[loom_condition_relation_set_node_index(builder->value_count,
                                                     children[j])] = 1;
      }
    }
  }
  loom_condition_relation_set_index_node_t* nodes = NULL;
  if (node_count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, node_count, sizeof(*nodes), (void**)&nodes));
  }
  uint32_t target_index = 0;
  for (iree_host_size_t i = 0; i < prefix_count; ++i) {
    if (!remap[i]) {
      continue;
    }
    const loom_condition_relation_set_build_node_t* source =
        loom_condition_relation_set_builder_node_at(builder, i);
    loom_condition_relation_set_index_node_t node = source->query;
    if ((source->key >> 27) != 0) {
      if (node.branch.left > builder->value_count) {
        node.branch.left = remap[loom_condition_relation_set_node_index(
            builder->value_count, node.branch.left)];
      }
      if (node.branch.right > builder->value_count) {
        node.branch.right = remap[loom_condition_relation_set_node_index(
            builder->value_count, node.branch.right)];
      }
    }
    nodes[target_index] = node;
    remap[i] = builder->value_count + target_index++ + 1;
  }
  // No fallible work follows root rewriting, so failure preserves caller IDs.
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    if (roots[i] > builder->value_count) {
      roots[i] = remap[loom_condition_relation_set_node_index(
          builder->value_count, roots[i])];
    }
  }
  *out_index = (loom_condition_relation_set_index_t){
      .nodes = nodes,
      .value_count = builder->value_count,
      .node_count = node_count,
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
