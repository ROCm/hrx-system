// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/type_dependencies.h"

#include <string.h>

#include "loom/ir/parameterized_type.h"
#include "loom/util/segmented_storage.h"

//===----------------------------------------------------------------------===//
// Canonical value sets
//===----------------------------------------------------------------------===//

enum {
  LOOM_VALUE_SET_SEGMENT_SHIFT = 7,
  LOOM_VALUE_SET_SEGMENT_CAPACITY = 128,
  LOOM_VALUE_SET_SEGMENT_MASK = 127,
  LOOM_VALUE_SET_BITMAP_SHIFT = 6,
  LOOM_VALUE_SET_BITMAP_MASK = 63,
};

// Immutable membership node. Branch children are disjoint; a leaf has bit -1.
typedef struct loom_value_set_node_t {
  // Lower and upper membership subtrees; both zero for a singleton.
  loom_value_set_id_t children[2];
  // Smallest member, also the compressed-prefix representative.
  loom_value_id_t value;
  // Highest differing value-ID bit, or -1 for a singleton.
  int32_t bit;
  // Members in one aligned 64-ID block when bit < 6; zero otherwise.
  uint64_t members;
} loom_value_set_node_t;

// One-based records. Radix-map edges tag the low bit of an ID, leaving 31
// bits for record identities. Record addresses never move.
typedef struct loom_value_set_records_t {
  // Published records; prepared capacity is not included.
  uint32_t count;
  // Fixed-capacity pages allocated from the index arena.
  loom_segmented_storage_t segments;
} loom_value_set_records_t;

typedef struct loom_value_set_union_t {
  // Ordered pair of input set IDs; each pair is evaluated once.
  uint64_t key;
  // Canonical result of the union.
  loom_value_set_id_t result;
} loom_value_set_union_t;

// A compressed sixteen-way exact map over 64-bit keys has at most 16 levels.
// Leaves name canonical records; branches have the low bit set. Empty is zero.
typedef struct loom_value_set_radix_t {
  // Representative key supplying the compressed low-bit prefix.
  uint64_t key;
  // Tagged edges selected by the key nibble at shift.
  uint32_t children[16];
  // Nibble-aligned bit offset, increasing along a branch path.
  uint32_t shift;
} loom_value_set_radix_t;

struct loom_value_set_index_t {
  // Arena owning the index and all stable record pages.
  iree_arena_allocator_t* arena;
  // Immutable canonical membership nodes.
  loom_value_set_records_t nodes;
  // Memoized composite unions; singleton insertion needs no memo entry.
  loom_value_set_records_t unions;
  // Shared branch storage for both exact maps.
  loom_value_set_records_t branches;
  // Canonical membership map keyed by child pair or singleton value.
  uint32_t node_root;
  // Composite-union map keyed by ordered input pair.
  uint32_t union_root;
  // Owner bytes stored parallel to each membership node.
  iree_host_size_t node_payload_size;
};

static void loom_value_set_records_initialize(
    iree_host_size_t record_size, loom_value_set_records_t* records) {
  loom_segmented_storage_initialize(
      record_size * LOOM_VALUE_SET_SEGMENT_CAPACITY, iree_alignof(uint64_t),
      &records->segments);
}

static void* loom_value_set_record(const loom_value_set_records_t* records,
                                   uint32_t id, iree_host_size_t record_size) {
  uint8_t* segment = (uint8_t*)loom_segmented_storage_const_segment(
      &records->segments, (id - 1) >> LOOM_VALUE_SET_SEGMENT_SHIFT);
  return segment + ((id - 1) & LOOM_VALUE_SET_SEGMENT_MASK) * record_size;
}

// Reserves a stable slot without publishing it. A failed later reserve leaves
// reusable page capacity, never a half-published record.
static iree_status_t loom_value_set_records_prepare(
    loom_value_set_index_t* index, loom_value_set_records_t* records,
    iree_host_size_t record_size, void** out_record) {
  if (records->count == INT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "value set index exhausted its 31-bit IDs");
  }
  const uint32_t segment_index = records->count >> LOOM_VALUE_SET_SEGMENT_SHIFT;
  if (segment_index == records->segments.segment_count) {
    void* segment = NULL;
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(&records->segments,
                                                       index->arena, &segment));
  }
  *out_record = loom_value_set_record(records, records->count + 1, record_size);
  return iree_ok_status();
}

static loom_value_set_union_t* loom_value_set_union_record(
    const loom_value_set_index_t* index, uint32_t id) {
  return (loom_value_set_union_t*)loom_value_set_record(
      &index->unions, id, sizeof(loom_value_set_union_t));
}

static loom_value_set_radix_t* loom_value_set_radix(
    const loom_value_set_index_t* index, uint32_t id) {
  return (loom_value_set_radix_t*)loom_value_set_record(
      &index->branches, id, sizeof(loom_value_set_radix_t));
}

static iree_status_t loom_value_set_index_allocate_with_payload(
    iree_arena_allocator_t* arena, iree_host_size_t node_payload_size,
    loom_value_set_index_t** out_index) {
  *out_index = NULL;
  iree_host_size_t node_segment_size = 0;
  if (!iree_host_size_checked_add(sizeof(loom_value_set_node_t),
                                  node_payload_size, &node_segment_size) ||
      !iree_host_size_checked_mul(node_segment_size,
                                  LOOM_VALUE_SET_SEGMENT_CAPACITY,
                                  &node_segment_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "value set node payload is too large");
  }
  loom_value_set_index_t* index = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*index), (void**)&index));
  memset(index, 0, sizeof(*index));
  index->arena = arena;
  index->node_payload_size = node_payload_size;
  loom_segmented_storage_initialize(node_segment_size, iree_alignof(uint64_t),
                                    &index->nodes.segments);
  loom_value_set_records_initialize(sizeof(loom_value_set_union_t),
                                    &index->unions);
  loom_value_set_records_initialize(sizeof(loom_value_set_radix_t),
                                    &index->branches);
  *out_index = index;
  return iree_ok_status();
}

iree_status_t loom_value_set_index_allocate(
    iree_arena_allocator_t* arena, loom_value_set_index_t** out_index) {
  return loom_value_set_index_allocate_with_payload(arena, 0, out_index);
}

static const loom_value_set_node_t* loom_value_set_index_node(
    const loom_value_set_index_t* index, loom_value_set_id_t set) {
  IREE_ASSERT(set > 0 && set <= index->nodes.count);
  return (const loom_value_set_node_t*)loom_value_set_record(
      &index->nodes, set, sizeof(loom_value_set_node_t));
}

loom_value_id_t loom_value_set_index_singleton(
    const loom_value_set_index_t* index, loom_value_set_id_t set) {
  if (!set) {
    return LOOM_VALUE_ID_INVALID;
  }
  const loom_value_set_node_t* node = loom_value_set_index_node(index, set);
  return node->bit < 0 ? node->value : LOOM_VALUE_ID_INVALID;
}

static void* loom_value_set_index_node_payload(
    const loom_value_set_index_t* index, loom_value_set_id_t set) {
  if (index->node_payload_size == 0) {
    return NULL;
  }
  IREE_ASSERT(set > 0 && set <= index->nodes.count);
  uint8_t* segment = (uint8_t*)loom_segmented_storage_const_segment(
      &index->nodes.segments, (set - 1) >> LOOM_VALUE_SET_SEGMENT_SHIFT);
  return segment +
         sizeof(loom_value_set_node_t) * LOOM_VALUE_SET_SEGMENT_CAPACITY +
         ((set - 1) & LOOM_VALUE_SET_SEGMENT_MASK) * index->node_payload_size;
}

bool loom_value_set_index_contains(const loom_value_set_index_t* index,
                                   loom_value_set_id_t set,
                                   loom_value_id_t value) {
  if (!set) {
    return false;
  }
  const loom_value_set_node_t* node = loom_value_set_index_node(index, set);
  while (node->bit >= 0) {
    node = loom_value_set_index_node(index,
                                     node->children[(value >> node->bit) & 1]);
  }
  return node->value == value;
}

typedef enum loom_value_set_key_kind_e {
  LOOM_VALUE_SET_KEY_NODE,
  LOOM_VALUE_SET_KEY_UNION,
} loom_value_set_key_kind_t;

static uint64_t loom_value_set_key(const loom_value_set_index_t* index,
                                   loom_value_set_key_kind_t kind,
                                   uint32_t id) {
  if (kind == LOOM_VALUE_SET_KEY_UNION) {
    return loom_value_set_union_record(index, id)->key;
  }
  const loom_value_set_node_t* node = loom_value_set_index_node(index, id);
  return node->bit < 0
             ? (uint64_t)node->value << 32
             : ((uint64_t)node->children[0] << 32) | node->children[1];
}

// A probe result is valid until the next insertion in this map. Stable page
// storage allows intervening capacity growth without invalidating its edge.
typedef struct loom_value_set_position_t {
  // Existing leaf or subtree edge to replace on insertion.
  uint32_t* edge;
  // Matching record, or zero on a miss.
  uint32_t existing;
  // Branch nibble required when replacing a nonempty edge.
  uint32_t shift;
  // New key.
  uint64_t key;
  // Representative key of the replaced edge.
  uint64_t old_key;
} loom_value_set_position_t;

static loom_value_set_position_t loom_value_set_probe(
    loom_value_set_index_t* index, loom_value_set_key_kind_t kind,
    uint32_t* root, uint64_t key) {
  uint32_t* edge = root;
  while (*edge & 1) {
    loom_value_set_radix_t* branch = loom_value_set_radix(index, *edge >> 1);
    const uint64_t difference =
        (key ^ branch->key) & ((UINT64_C(1) << branch->shift) - 1);
    if (difference) {
      return (loom_value_set_position_t){
          .edge = edge,
          .shift =
              (uint32_t)iree_math_count_trailing_zeros_u64(difference) & ~3u,
          .key = key,
          .old_key = branch->key,
      };
    }
    edge = &branch->children[(key >> branch->shift) & 15];
  }
  if (!*edge) {
    return (loom_value_set_position_t){.edge = edge, .key = key};
  }
  const uint64_t old_key = loom_value_set_key(index, kind, *edge >> 1);
  if (old_key == key) {
    return (loom_value_set_position_t){.edge = edge, .existing = *edge >> 1};
  }
  return (loom_value_set_position_t){
      .edge = edge,
      .shift =
          (uint32_t)iree_math_count_trailing_zeros_u64(key ^ old_key) & ~3u,
      .key = key,
      .old_key = old_key,
  };
}

static iree_status_t loom_value_set_insert(loom_value_set_index_t* index,
                                           loom_value_set_position_t position,
                                           uint32_t record_id) {
  if (!*position.edge) {
    *position.edge = record_id << 1;
    return iree_ok_status();
  }
  loom_value_set_radix_t* branch = NULL;
  IREE_RETURN_IF_ERROR(loom_value_set_records_prepare(
      index, &index->branches, sizeof(*branch), (void**)&branch));
  *branch = (loom_value_set_radix_t){
      .key = position.old_key,
      .shift = position.shift,
  };
  branch->children[(position.old_key >> position.shift) & 15] = *position.edge;
  branch->children[(position.key >> position.shift) & 15] = record_id << 1;
  *position.edge = (++index->branches.count << 1) | 1;
  return iree_ok_status();
}

static iree_status_t loom_value_set_intern_node(
    loom_value_set_index_t* index, uint64_t key,
    const loom_value_set_node_t* candidate, loom_value_set_id_t* out_set) {
  loom_value_set_position_t position = loom_value_set_probe(
      index, LOOM_VALUE_SET_KEY_NODE, &index->node_root, key);
  if (position.existing) {
    *out_set = position.existing;
    return iree_ok_status();
  }
  loom_value_set_node_t* node = NULL;
  IREE_RETURN_IF_ERROR(loom_value_set_records_prepare(
      index, &index->nodes, sizeof(*node), (void**)&node));
  IREE_RETURN_IF_ERROR(
      loom_value_set_insert(index, position, index->nodes.count + 1));
  *node = *candidate;
  *out_set = ++index->nodes.count;
  if (index->node_payload_size != 0) {
    memset(loom_value_set_index_node_payload(index, *out_set), 0,
           index->node_payload_size);
  }
  return iree_ok_status();
}

static int32_t loom_value_set_highest_bit(uint32_t value) {
  return 31 - iree_math_count_leading_zeros_u32(value);
}

static iree_status_t loom_value_set_join(loom_value_set_index_t* index,
                                         loom_value_set_id_t left,
                                         loom_value_set_id_t right,
                                         loom_value_set_id_t* out_set) {
  if (!left || left == right) {
    *out_set = right;
    return iree_ok_status();
  }
  if (!right) {
    *out_set = left;
    return iree_ok_status();
  }
  const loom_value_id_t left_value =
      loom_value_set_index_node(index, left)->value;
  const loom_value_id_t right_value =
      loom_value_set_index_node(index, right)->value;
  const int32_t bit = loom_value_set_highest_bit(left_value ^ right_value);
  IREE_ASSERT(loom_value_set_index_node(index, left)->bit < bit &&
              loom_value_set_index_node(index, right)->bit < bit);
  if ((left_value >> bit) & 1) {
    loom_value_set_id_t temporary = left;
    left = right;
    right = temporary;
  }
  const loom_value_set_node_t candidate = {
      .children = {left, right},
      .value = loom_value_set_index_node(index, left)->value,
      .bit = bit,
      .members = bit < LOOM_VALUE_SET_BITMAP_SHIFT
                     ? loom_value_set_index_node(index, left)->members |
                           loom_value_set_index_node(index, right)->members
                     : 0,
  };
  return loom_value_set_intern_node(index, ((uint64_t)left << 32) | right,
                                    &candidate, out_set);
}

iree_status_t loom_value_set_index_union_nonempty(
    loom_value_set_index_t* index, loom_value_set_id_t first,
    loom_value_set_id_t second, loom_value_set_id_t* out_set) {
  if (first > second) {
    loom_value_set_id_t temporary = first;
    first = second;
    second = temporary;
  }
  const loom_value_set_node_t* a = loom_value_set_index_node(index, first);
  const loom_value_set_node_t* b = loom_value_set_index_node(index, second);
  const uint64_t key = ((uint64_t)first << 32) | second;
  const bool memoize = a->bit >= 0 && b->bit >= 0;
  if (memoize) {
    loom_value_set_position_t position = loom_value_set_probe(
        index, LOOM_VALUE_SET_KEY_UNION, &index->union_root, key);
    if (position.existing) {
      *out_set = loom_value_set_union_record(index, position.existing)->result;
      return iree_ok_status();
    }
  }
  loom_value_set_id_t result = 0;
  if (loom_value_set_highest_bit(a->value ^ b->value) >
      iree_max(a->bit, b->bit)) {
    IREE_RETURN_IF_ERROR(loom_value_set_join(index, first, second, &result));
  } else if (a->bit == b->bit) {
    loom_value_set_id_t left = 0;
    loom_value_set_id_t right = 0;
    IREE_RETURN_IF_ERROR(loom_value_set_index_union(index, a->children[0],
                                                    b->children[0], &left));
    IREE_RETURN_IF_ERROR(loom_value_set_index_union(index, a->children[1],
                                                    b->children[1], &right));
    IREE_RETURN_IF_ERROR(loom_value_set_join(index, left, right, &result));
  } else {
    const loom_value_set_node_t* higher = a->bit > b->bit ? a : b;
    const loom_value_set_node_t* lower = a->bit > b->bit ? b : a;
    const uint32_t side = (lower->value >> higher->bit) & 1;
    loom_value_set_id_t changed = 0;
    IREE_RETURN_IF_ERROR(
        loom_value_set_index_union(index, higher->children[side],
                                   a->bit > b->bit ? second : first, &changed));
    if (changed == higher->children[side]) {
      result = a->bit > b->bit ? first : second;
    } else {
      IREE_RETURN_IF_ERROR(
          loom_value_set_join(index, side ? higher->children[0] : changed,
                              side ? changed : higher->children[1], &result));
    }
  }
  if (memoize) {
    const loom_value_set_position_t position = loom_value_set_probe(
        index, LOOM_VALUE_SET_KEY_UNION, &index->union_root, key);
    loom_value_set_union_t* entry = NULL;
    IREE_RETURN_IF_ERROR(loom_value_set_records_prepare(
        index, &index->unions, sizeof(*entry), (void**)&entry));
    IREE_RETURN_IF_ERROR(
        loom_value_set_insert(index, position, index->unions.count + 1));
    *entry = (loom_value_set_union_t){.key = key, .result = result};
    ++index->unions.count;
  }
  *out_set = result;
  return iree_ok_status();
}

iree_status_t loom_value_set_index_intersection(loom_value_set_index_t* index,
                                                loom_value_set_id_t first,
                                                loom_value_set_id_t second,
                                                loom_value_set_id_t* out_set) {
  if (!first || !second || first == second) {
    *out_set = first && second ? first : 0;
    return iree_ok_status();
  }
  const loom_value_set_node_t* a = loom_value_set_index_node(index, first);
  const loom_value_set_node_t* b = loom_value_set_index_node(index, second);
  if (loom_value_set_highest_bit(a->value ^ b->value) >
      iree_max(a->bit, b->bit)) {
    *out_set = 0;
    return iree_ok_status();
  }
  if (a->bit == b->bit) {
    loom_value_set_id_t left = 0;
    loom_value_set_id_t right = 0;
    IREE_RETURN_IF_ERROR(loom_value_set_index_intersection(
        index, a->children[0], b->children[0], &left));
    IREE_RETURN_IF_ERROR(loom_value_set_index_intersection(
        index, a->children[1], b->children[1], &right));
    return loom_value_set_join(index, left, right, out_set);
  }
  const loom_value_set_node_t* higher = a->bit > b->bit ? a : b;
  const uint32_t side = ((a->bit > b->bit ? b : a)->value >> higher->bit) & 1;
  return loom_value_set_index_intersection(
      index, higher->children[side], a->bit > b->bit ? second : first, out_set);
}

iree_status_t loom_value_set_index_add(loom_value_set_index_t* index,
                                       loom_value_set_id_t set,
                                       loom_value_id_t value,
                                       loom_value_set_id_t* out_set) {
  const loom_value_set_node_t candidate = {
      .value = value,
      .bit = -1,
      .members = UINT64_C(1) << (value & LOOM_VALUE_SET_BITMAP_MASK),
  };
  loom_value_set_id_t singleton = 0;
  IREE_RETURN_IF_ERROR(loom_value_set_intern_node(index, (uint64_t)value << 32,
                                                  &candidate, &singleton));
  return loom_value_set_index_union(index, set, singleton, out_set);
}

iree_status_t loom_value_set_index_prefix(loom_value_set_index_t* index,
                                          loom_value_set_id_t set,
                                          uint64_t limit,
                                          loom_value_set_id_t* out_set) {
  if (!set) {
    *out_set = 0;
    return iree_ok_status();
  }
  const loom_value_set_node_t* node = loom_value_set_index_node(index, set);
  if (node->value >= limit) {
    *out_set = 0;
    return iree_ok_status();
  }
  if (node->bit < 0) {
    *out_set = set;
    return iree_ok_status();
  }
  const uint64_t mask = (UINT64_C(1) << (node->bit + 1)) - 1;
  if (((uint64_t)node->value | mask) < limit) {
    *out_set = set;
    return iree_ok_status();
  }
  const uint64_t boundary =
      ((uint64_t)node->value & ~mask) | (UINT64_C(1) << node->bit);
  if (limit <= boundary) {
    return loom_value_set_index_prefix(index, node->children[0], limit,
                                       out_set);
  }
  loom_value_set_id_t right = 0;
  IREE_RETURN_IF_ERROR(
      loom_value_set_index_prefix(index, node->children[1], limit, &right));
  return loom_value_set_join(index, node->children[0], right, out_set);
}

void loom_value_set_cursor_begin(const loom_value_set_index_t* index,
                                 loom_value_set_id_t set,
                                 loom_value_set_cursor_t* out_cursor) {
  *out_cursor = (loom_value_set_cursor_t){.index = index};
  if (set) {
    out_cursor->pending[out_cursor->pending_count++] = set;
  }
}

bool loom_value_set_cursor_advance(loom_value_set_cursor_t* cursor) {
  if (!cursor->pending_count) {
    return false;
  }
  const loom_value_set_node_t* node = loom_value_set_index_node(
      cursor->index, cursor->pending[--cursor->pending_count]);
  while (node->bit >= LOOM_VALUE_SET_BITMAP_SHIFT) {
    cursor->pending[cursor->pending_count++] = node->children[1];
    node = loom_value_set_index_node(cursor->index, node->children[0]);
  }
  cursor->members = node->members;
  cursor->base = node->value & ~LOOM_VALUE_SET_BITMAP_MASK;
  return true;
}

//===----------------------------------------------------------------------===//
// Active dependency ownership
//===----------------------------------------------------------------------===//

enum {
  LOOM_TYPE_DEPENDENCY_SEGMENT_SHIFT = 7,
  LOOM_TYPE_DEPENDENCY_SEGMENT_CAPACITY = 128,
  LOOM_TYPE_DEPENDENCY_SEGMENT_MASK = 127,
  LOOM_VALUE_DEPENDENCY_BITMAP_SHIFT = 6,
  LOOM_VALUE_DEPENDENCY_BITMAP_MASK = 63,
};

// One-based carrier records. Record addresses never move.
typedef struct loom_dependency_records_t {
  // Published records; prepared capacity is not included.
  uint32_t count;
  // Fixed-capacity pages allocated from the dependency arena.
  loom_segmented_storage_t segments;
} loom_dependency_records_t;

typedef enum loom_dependency_owner_kind_e {
  LOOM_DEPENDENCY_OWNER_VALUE = 0,
  LOOM_DEPENDENCY_OWNER_ATTRIBUTE = 1,
} loom_dependency_owner_kind_t;

typedef struct loom_dependency_ownership_t {
  // First active parent edge, encoded as parent ID shifted left plus side.
  uint32_t first_parent;
  // First carrier attached directly to this set.
  uint32_t first_carrier;
  // Number of active parents and directly attached carriers.
  uint32_t active_owners;
  // Links for this node's two edges in its children's active-parent lists.
  struct {
    // Previous active parent edge, or zero at the list head.
    uint32_t previous;
    // Next active parent edge, or zero at the list tail.
    uint32_t next;
  } edges[2];
} loom_dependency_ownership_t;

typedef struct loom_dependency_carrier_t {
  // Full declared membership, including providers not yet defined.
  uint32_t declared;
  // Currently active membership, or zero while dropped.
  uint32_t active;
  union {
    // Prepared membership used by failure-atomic value-owner bulk refresh.
    uint32_t prepared;
    // Retained dimension subset for attribute owners.
    uint32_t dimension_dependencies;
  };
  // Previous carrier on the active root, or zero at the list head.
  uint32_t previous;
  // Next carrier on the active root, or next recycled carrier while unused.
  uint32_t next;
  // Ownership channel selecting the identity and active reverse links.
  uint8_t kind;
  // Attribute ordinal for an operation owner; unused for value carriers.
  uint8_t attribute_index;
  // Stable owner identity, independent of type and attribute payload addresses.
  union {
    // Value whose type carries these dependencies.
    loom_value_id_t value;
    // Operation whose attribute carries these dependencies.
    loom_op_t* op;
  } owner;
} loom_dependency_carrier_t;

struct loom_type_dependency_index_t {
  // Canonical immutable value-set membership.
  loom_value_set_index_t* membership;
  // Value and attribute owners, recycled when declared membership is empty.
  loom_dependency_records_t carriers;
  // First reusable carrier, or zero when none is available.
  uint32_t free_carrier;
};

static void loom_dependency_records_initialize(
    iree_host_size_t record_size, loom_dependency_records_t* records) {
  loom_segmented_storage_initialize(
      record_size * LOOM_TYPE_DEPENDENCY_SEGMENT_CAPACITY,
      iree_alignof(uint64_t), &records->segments);
}

static void* loom_dependency_record(const loom_dependency_records_t* records,
                                    uint32_t id, iree_host_size_t record_size) {
  uint8_t* segment = (uint8_t*)loom_segmented_storage_const_segment(
      &records->segments, (id - 1) >> LOOM_TYPE_DEPENDENCY_SEGMENT_SHIFT);
  return segment + ((id - 1) & LOOM_TYPE_DEPENDENCY_SEGMENT_MASK) * record_size;
}

// Reserves a stable slot without publishing a record. A failed later reserve
// leaves only reusable page capacity, never a half-published carrier.
static iree_status_t loom_dependency_records_prepare(
    iree_arena_allocator_t* arena, loom_dependency_records_t* records,
    iree_host_size_t record_size, void** out_record) {
  if (records->count == INT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "type dependency carrier index exhausted");
  }
  const uint32_t segment_index =
      records->count >> LOOM_TYPE_DEPENDENCY_SEGMENT_SHIFT;
  if (segment_index == records->segments.segment_count) {
    void* segment = NULL;
    IREE_RETURN_IF_ERROR(
        loom_segmented_storage_append(&records->segments, arena, &segment));
  }
  *out_record =
      loom_dependency_record(records, records->count + 1, record_size);
  return iree_ok_status();
}

static const loom_value_set_node_t* loom_dependency_node(
    const loom_type_dependency_index_t* index, uint32_t id) {
  return loom_value_set_index_node(index->membership, id);
}

static loom_dependency_ownership_t* loom_dependency_ownership(
    const loom_type_dependency_index_t* index, uint32_t id,
    loom_dependency_owner_kind_t kind) {
  loom_dependency_ownership_t* ownership =
      loom_value_set_index_node_payload(index->membership, id);
  return &ownership[kind];
}

static loom_dependency_carrier_t* loom_dependency_carrier(
    const loom_type_dependency_index_t* index, uint32_t id) {
  return (loom_dependency_carrier_t*)loom_dependency_record(
      &index->carriers, id, sizeof(loom_dependency_carrier_t));
}

static iree_status_t loom_type_dependencies_allocate_index(
    loom_type_use_table_t* table) {
  if (table->index) {
    return iree_ok_status();
  }
  loom_type_dependency_index_t* index = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&table->arena, sizeof(*index), (void**)&index));
  memset(index, 0, sizeof(*index));
  IREE_RETURN_IF_ERROR(loom_value_set_index_allocate_with_payload(
      &table->arena, 2 * sizeof(loom_dependency_ownership_t),
      &index->membership));
  loom_dependency_records_initialize(sizeof(loom_dependency_carrier_t),
                                     &index->carriers);
  table->index = index;
  return iree_ok_status();
}

bool loom_type_dependencies_contains(const loom_type_use_table_t* table,
                                     loom_type_dependency_id_t root,
                                     loom_value_id_t provider) {
  return root && loom_value_set_index_contains(table->index->membership, root,
                                               provider);
}

iree_status_t loom_type_dependencies_union_nonempty(
    loom_type_use_table_t* table, loom_type_dependency_id_t first,
    loom_type_dependency_id_t second, loom_type_dependency_id_t* out_root) {
  return loom_value_set_index_union_nonempty(table->index->membership, first,
                                             second, out_root);
}

iree_status_t loom_type_dependencies_add(loom_type_use_table_t* table,
                                         loom_type_dependency_id_t root,
                                         loom_value_id_t provider,
                                         loom_type_dependency_id_t* out_root) {
  IREE_RETURN_IF_ERROR(loom_type_dependencies_allocate_index(table));
  return loom_value_set_index_add(table->index->membership, root, provider,
                                  out_root);
}

static iree_status_t loom_dependency_prefix(loom_type_use_table_t* table,
                                            uint32_t root, uint64_t limit,
                                            uint32_t* out_root) {
  return loom_value_set_index_prefix(table->index->membership, root, limit,
                                     out_root);
}

//===----------------------------------------------------------------------===//
// Construction-time dependency facts
//===----------------------------------------------------------------------===//

// Attribute payloads and TYPE IDs have already passed canonical construction.
// Aggregate recursion visits immediate payload ownership, never a child type.
static iree_status_t loom_dependency_collect_attribute(
    loom_module_t* module, const loom_attribute_t* attribute,
    loom_type_dependency_sets_t* sets) {
  switch ((loom_attr_kind_t)attribute->kind) {
    case LOOM_ATTR_TYPE: {
      IREE_RETURN_IF_ERROR(loom_type_dependencies_union(
          &module->type_uses, sets->all,
          loom_type_table_dependencies(&module->types, attribute->type_id),
          &sets->all));
      return loom_type_dependencies_union(
          &module->type_uses, sets->dimensions,
          loom_type_table_dimension_dependencies(&module->types,
                                                 attribute->type_id),
          &sets->dimensions);
    }
    case LOOM_ATTR_PREDICATE_LIST:
      for (uint16_t i = 0; i < attribute->count; ++i) {
        const loom_predicate_t* predicate = &attribute->predicate_list[i];
        for (uint8_t j = 0; j < predicate->arg_count; ++j) {
          if (predicate->arg_tags[j] == LOOM_PRED_ARG_VALUE) {
            IREE_RETURN_IF_ERROR(loom_type_dependencies_add(
                &module->type_uses, sets->all,
                (loom_value_id_t)predicate->args[j], &sets->all));
          }
        }
      }
      break;
    case LOOM_ATTR_DICT:
      for (uint16_t i = 0; i < attribute->count; ++i) {
        IREE_RETURN_IF_ERROR(loom_dependency_collect_attribute(
            module, &attribute->dict_entries[i].value, sets));
      }
      break;
    case LOOM_ATTR_PARAMETERIZED:
      for (uint16_t i = 0; i < attribute->count; ++i) {
        IREE_RETURN_IF_ERROR(loom_dependency_collect_attribute(
            module, &attribute->parameterized_slots[i], sets));
      }
      break;
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      for (uint16_t i = 0; i < attribute->count; ++i) {
        IREE_RETURN_IF_ERROR(loom_dependency_collect_attribute(
            module, &attribute->parameterized_array[i], sets));
      }
      break;
    default:
      break;
  }
  return iree_ok_status();
}

iree_status_t loom_type_dependencies_collect_immediate(
    loom_module_t* module, loom_type_t type,
    loom_type_dependency_sets_t* inout_sets) {
  if (!loom_type_kind_is_valid(loom_type_kind(type)) ||
      !loom_type_may_reference_values(type)) {
    return iree_ok_status();
  }
  switch (loom_type_kind(type)) {
    case LOOM_TYPE_PARAMETERIZED: {
      const loom_attribute_t* parameters =
          loom_type_parameterized_parameters(type);
      const uint8_t count = loom_type_parameterized_parameter_count(type);
      for (uint8_t i = 0; i < count; ++i) {
        IREE_RETURN_IF_ERROR(loom_dependency_collect_attribute(
            module, &parameters[i], inout_sets));
      }
      return iree_ok_status();
    }
    default:
      break;
  }
  if (loom_type_has_dimensions(type)) {
    for (uint8_t i = 0; i < loom_type_rank(type); ++i) {
      if (loom_type_dim_is_dynamic_at(type, i)) {
        const loom_value_id_t provider = loom_type_dim_value_id_at(type, i);
        IREE_RETURN_IF_ERROR(loom_type_dependencies_add(
            &module->type_uses, inout_sets->all, provider, &inout_sets->all));
        IREE_RETURN_IF_ERROR(loom_type_dependencies_add(
            &module->type_uses, inout_sets->dimensions, provider,
            &inout_sets->dimensions));
      }
    }
  }
  if (loom_type_has_ssa_encoding(type)) {
    IREE_RETURN_IF_ERROR(loom_type_dependencies_add(
        &module->type_uses, inout_sets->all, loom_type_encoding_value_id(type),
        &inout_sets->all));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Active reverse ownership
//===----------------------------------------------------------------------===//

static void loom_dependency_acquire(loom_type_use_table_t* table, uint32_t id,
                                    loom_dependency_owner_kind_t kind) {
  const loom_value_set_node_t* node = loom_dependency_node(table->index, id);
  loom_dependency_ownership_t* ownership =
      loom_dependency_ownership(table->index, id, kind);
  if (ownership->active_owners++) {
    return;
  }
  if (node->bit < 0) {
    loom_value_table_type_use_heads(table->value_table, node->value)->provider =
        id;
    if (kind == LOOM_DEPENDENCY_OWNER_ATTRIBUTE) {
      loom_value_table_value(table->value_table, node->value)->flags |=
          LOOM_VALUE_FLAG_ATTRIBUTE_USES;
    }
    return;
  }
  for (uint32_t side = 0; side < 2; ++side) {
    loom_dependency_ownership_t* child =
        loom_dependency_ownership(table->index, node->children[side], kind);
    const uint32_t edge = (id << 1) | side;
    ownership->edges[side].previous = 0;
    ownership->edges[side].next = child->first_parent;
    if (child->first_parent) {
      loom_dependency_ownership(table->index, child->first_parent >> 1, kind)
          ->edges[child->first_parent & 1]
          .previous = edge;
    }
    child->first_parent = edge;
    loom_dependency_acquire(table, node->children[side], kind);
  }
}

static void loom_dependency_release(loom_type_use_table_t* table, uint32_t id,
                                    loom_dependency_owner_kind_t kind) {
  const loom_value_set_node_t* node = loom_dependency_node(table->index, id);
  loom_dependency_ownership_t* ownership =
      loom_dependency_ownership(table->index, id, kind);
  if (--ownership->active_owners) {
    return;
  }
  if (node->bit < 0) {
    if (kind == LOOM_DEPENDENCY_OWNER_ATTRIBUTE) {
      loom_value_table_value(table->value_table, node->value)->flags &=
          ~LOOM_VALUE_FLAG_ATTRIBUTE_USES;
    }
    return;
  }
  for (uint32_t side = 0; side < 2; ++side) {
    const uint32_t previous = ownership->edges[side].previous;
    const uint32_t next = ownership->edges[side].next;
    if (previous) {
      loom_dependency_ownership(table->index, previous >> 1, kind)
          ->edges[previous & 1]
          .next = next;
    } else {
      loom_dependency_ownership(table->index, node->children[side], kind)
          ->first_parent = next;
    }
    if (next) {
      loom_dependency_ownership(table->index, next >> 1, kind)
          ->edges[next & 1]
          .previous = previous;
    }
    loom_dependency_release(table, node->children[side], kind);
  }
}

static void loom_dependency_attach(loom_type_use_table_t* table,
                                   uint32_t carrier_id, uint32_t root) {
  loom_dependency_carrier_t* carrier =
      loom_dependency_carrier(table->index, carrier_id);
  const loom_dependency_owner_kind_t kind = carrier->kind;
  if (carrier->active == root) {
    return;
  }
  // Acquiring first keeps shared subtrees active throughout replacement.
  if (root) {
    loom_dependency_acquire(table, root, kind);
    if (kind == LOOM_DEPENDENCY_OWNER_VALUE) {
      ++table->active_carrier_count;
    }
  }
  if (carrier->active) {
    if (carrier->previous) {
      loom_dependency_carrier(table->index, carrier->previous)->next =
          carrier->next;
    } else {
      loom_dependency_ownership(table->index, carrier->active, kind)
          ->first_carrier = carrier->next;
    }
    if (carrier->next) {
      loom_dependency_carrier(table->index, carrier->next)->previous =
          carrier->previous;
    }
    loom_dependency_release(table, carrier->active, kind);
    if (kind == LOOM_DEPENDENCY_OWNER_VALUE) {
      --table->active_carrier_count;
    }
  }
  carrier->active = root;
  carrier->previous = 0;
  carrier->next =
      root ? loom_dependency_ownership(table->index, root, kind)->first_carrier
           : 0;
  if (carrier->next) {
    loom_dependency_carrier(table->index, carrier->next)->previous = carrier_id;
  }
  if (root) {
    loom_dependency_ownership(table->index, root, kind)->first_carrier =
        carrier_id;
  }
}

static iree_status_t loom_dependency_prepare_carrier(
    loom_type_use_table_t* table, uint32_t existing_carrier,
    loom_type_dependency_id_t root, iree_host_size_t available_value_count,
    loom_type_dependency_assignment_t* out_assignment) {
  *out_assignment = (loom_type_dependency_assignment_t){.declared = root};
  if (!root) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_dependency_prefix(
      table, root, available_value_count, &out_assignment->active));
  if (existing_carrier) {
    out_assignment->carrier = existing_carrier;
    return iree_ok_status();
  }
  loom_dependency_carrier_t* carrier = NULL;
  if (table->index->free_carrier) {
    out_assignment->carrier = table->index->free_carrier;
    carrier = loom_dependency_carrier(table->index, out_assignment->carrier);
    table->index->free_carrier = carrier->next;
  } else {
    IREE_RETURN_IF_ERROR(
        loom_dependency_records_prepare(&table->arena, &table->index->carriers,
                                        sizeof(*carrier), (void**)&carrier));
    out_assignment->carrier = ++table->index->carriers.count;
  }
  *carrier = (loom_dependency_carrier_t){0};
  return iree_ok_status();
}

static void loom_dependency_commit_carrier(
    loom_type_use_table_t* table, uint32_t* slot,
    const loom_type_dependency_assignment_t* assignment) {
  if (assignment->carrier) {
    loom_dependency_attach(table, assignment->carrier, assignment->active);
    loom_dependency_carrier(table->index, assignment->carrier)->declared =
        assignment->declared;
    *slot = assignment->carrier;
  } else if (*slot) {
    loom_dependency_attach(table, *slot, 0);
    loom_dependency_carrier_t* carrier =
        loom_dependency_carrier(table->index, *slot);
    carrier->declared = 0;
    carrier->next = table->index->free_carrier;
    table->index->free_carrier = *slot;
    *slot = 0;
  }
}

iree_status_t loom_type_dependencies_prepare(
    loom_type_use_table_t* table, loom_value_id_t value_id,
    loom_type_dependency_id_t root, iree_host_size_t available_value_count,
    loom_type_dependency_assignment_t* out_assignment) {
  if (!root) {
    *out_assignment = (loom_type_dependency_assignment_t){0};
    return iree_ok_status();
  }
  const uint32_t carrier_id =
      value_id < table->value_table->count
          ? loom_value_table_type_use_heads(table->value_table, value_id)
                ->carrier
          : 0;
  IREE_RETURN_IF_ERROR(loom_dependency_prepare_carrier(
      table, carrier_id, root, available_value_count, out_assignment));
  if (out_assignment->carrier) {
    loom_dependency_carrier_t* carrier =
        loom_dependency_carrier(table->index, out_assignment->carrier);
    carrier->kind = LOOM_DEPENDENCY_OWNER_VALUE;
    carrier->owner.value = value_id;
  }
  return iree_ok_status();
}

void loom_type_dependencies_commit(
    loom_type_use_table_t* table, loom_value_id_t value_id,
    const loom_type_dependency_assignment_t* assignment) {
  loom_value_type_use_heads_t* heads =
      loom_value_table_type_use_heads(table->value_table, value_id);
  loom_dependency_commit_carrier(table, &heads->carrier, assignment);
}

iree_status_t loom_attribute_dependencies_prepare(
    loom_type_use_table_t* table, loom_op_t* op, uint8_t attribute_index,
    loom_type_dependency_sets_t dependencies,
    loom_type_dependency_assignment_t* out_assignment) {
  IREE_RETURN_IF_ERROR(loom_dependency_prepare_carrier(
      table, loom_op_attribute_owners(op)[attribute_index], dependencies.all,
      table->value_table->count, out_assignment));
  out_assignment->dimensions = dependencies.dimensions;
  if (out_assignment->carrier) {
    loom_dependency_carrier_t* carrier =
        loom_dependency_carrier(table->index, out_assignment->carrier);
    carrier->kind = LOOM_DEPENDENCY_OWNER_ATTRIBUTE;
    carrier->owner.op = op;
    carrier->attribute_index = attribute_index;
  }
  return iree_ok_status();
}

void loom_attribute_dependencies_commit(
    loom_type_use_table_t* table, loom_op_t* op, uint8_t attribute_index,
    const loom_type_dependency_assignment_t* assignment) {
  loom_dependency_commit_carrier(
      table, &loom_op_attribute_owners(op)[attribute_index], assignment);
  const uint32_t carrier = loom_op_attribute_owners(op)[attribute_index];
  if (carrier) {
    loom_dependency_carrier(table->index, carrier)->dimension_dependencies =
        assignment->dimensions;
  }
}

void loom_attribute_dependencies_drop(loom_type_use_table_t* table,
                                      loom_op_t* op, uint8_t attribute_index) {
  const loom_type_dependency_assignment_t empty = {0};
  loom_attribute_dependencies_commit(table, op, attribute_index, &empty);
}

void loom_attribute_dependencies_reset(loom_type_use_table_t* table) {
  if (!table->index) {
    return;
  }
  for (uint32_t id = 1; id <= table->index->carriers.count; ++id) {
    const loom_dependency_carrier_t* carrier =
        loom_dependency_carrier(table->index, id);
    if (carrier->declared && carrier->kind == LOOM_DEPENDENCY_OWNER_ATTRIBUTE) {
      loom_attribute_dependencies_drop(table, carrier->owner.op,
                                       carrier->attribute_index);
    }
  }
}

iree_status_t loom_type_dependencies_refresh(loom_type_use_table_t* table,
                                             loom_value_id_t value_id) {
  const uint32_t carrier_id =
      loom_value_table_type_use_heads(table->value_table, value_id)->carrier;
  if (!carrier_id) {
    return iree_ok_status();
  }
  loom_dependency_carrier_t* carrier =
      loom_dependency_carrier(table->index, carrier_id);
  uint32_t root = 0;
  IREE_RETURN_IF_ERROR(loom_dependency_prefix(
      table, carrier->declared, table->value_table->count, &root));
  loom_dependency_attach(table, carrier_id, root);
  return iree_ok_status();
}

static bool loom_dependency_carrier_is_live(const loom_value_t* value) {
  if (loom_value_is_block_arg(value)) {
    return true;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op) {
    return !iree_any_bit_set(defining_op->flags, LOOM_OP_FLAG_DEAD);
  }
  return value->use_count != 0;
}

iree_status_t loom_type_dependencies_recompute(loom_type_use_table_t* table) {
  if (!table->index) {
    return iree_ok_status();
  }
  for (uint32_t id = 1; id <= table->index->carriers.count; ++id) {
    loom_dependency_carrier_t* carrier =
        loom_dependency_carrier(table->index, id);
    if (carrier->kind != LOOM_DEPENDENCY_OWNER_VALUE) {
      continue;
    }
    carrier->prepared = 0;
    if (carrier->declared &&
        loom_dependency_carrier_is_live(loom_value_table_const_value(
            table->value_table, carrier->owner.value))) {
      IREE_RETURN_IF_ERROR(loom_dependency_prefix(table, carrier->declared,
                                                  table->value_table->count,
                                                  &carrier->prepared));
    }
  }
  for (uint32_t id = 1; id <= table->index->carriers.count; ++id) {
    loom_dependency_carrier_t* carrier =
        loom_dependency_carrier(table->index, id);
    if (carrier->declared && carrier->kind == LOOM_DEPENDENCY_OWNER_VALUE) {
      loom_dependency_attach(table, id, carrier->prepared);
    }
  }
  return iree_ok_status();
}

void loom_type_dependencies_drop(loom_type_use_table_t* table,
                                 loom_value_id_t value_id) {
  if (value_id >= table->value_table->count) {
    return;
  }
  const uint32_t carrier =
      loom_value_table_type_use_heads(table->value_table, value_id)->carrier;
  if (carrier) {
    loom_dependency_attach(table, carrier, 0);
  }
}

bool loom_type_dependencies_has_users(const loom_type_use_table_t* table,
                                      loom_value_id_t value_id) {
  if (value_id >= table->value_table->count) {
    return false;
  }
  const uint32_t provider =
      loom_value_table_const_type_use_heads(table->value_table, value_id)
          ->provider;
  return provider && loom_dependency_ownership(table->index, provider,
                                               LOOM_DEPENDENCY_OWNER_VALUE)
                             ->active_owners != 0;
}

//===----------------------------------------------------------------------===//
// Bounded membership and ownership cursors
//===----------------------------------------------------------------------===//

void loom_type_dependencies_begin(const loom_type_use_table_t* table,
                                  loom_value_id_t value_id,
                                  loom_type_use_iterator_t* out_iterator) {
  uint32_t root = 0;
  if (value_id < table->value_table->count) {
    const uint32_t carrier =
        loom_value_table_const_type_use_heads(table->value_table, value_id)
            ->carrier;
    root = carrier ? loom_dependency_carrier(table->index, carrier)->active : 0;
  }
  loom_type_dependency_set_begin(table, root, out_iterator);
}

void loom_type_dependency_set_begin(const loom_type_use_table_t* table,
                                    loom_type_dependency_id_t root,
                                    loom_type_use_iterator_t* out_iterator) {
  *out_iterator = (loom_type_use_iterator_t){.index = table->index};
  if (root) {
    out_iterator->pending[out_iterator->pending_count++] = root;
  }
}

bool loom_type_dependencies_advance(loom_type_use_iterator_t* iterator) {
  if (!iterator->pending_count) {
    return false;
  }
  const loom_value_set_node_t* node = loom_dependency_node(
      iterator->index, iterator->pending[--iterator->pending_count]);
  while (node->bit >= LOOM_VALUE_DEPENDENCY_BITMAP_SHIFT) {
    // Only later siblings need continuations. Small subtrees already retain
    // their complete membership in one bitmap.
    iterator->pending[iterator->pending_count++] = node->children[1];
    node = loom_dependency_node(iterator->index, node->children[0]);
  }
  iterator->members = node->members;
  iterator->base = node->value & ~LOOM_VALUE_DEPENDENCY_BITMAP_MASK;
  return true;
}

static void loom_dependency_users_begin(
    const loom_type_use_table_t* table, loom_value_id_t value_id,
    loom_dependency_owner_kind_t kind, loom_type_use_iterator_t* out_iterator) {
  out_iterator->index = table->index;
  out_iterator->pending_count = 0;
  out_iterator->carrier = 0;
  out_iterator->owner_kind = kind;
  if (value_id < table->value_table->count) {
    const uint32_t provider =
        loom_value_table_const_type_use_heads(table->value_table, value_id)
            ->provider;
    if (!provider) {
      return;
    }
    const loom_dependency_ownership_t* ownership =
        loom_dependency_ownership(table->index, provider, kind);
    out_iterator->carrier = ownership->first_carrier;
    if (ownership->first_parent) {
      out_iterator->pending[out_iterator->pending_count++] =
          ownership->first_parent;
    }
  }
}

static const loom_dependency_carrier_t* loom_dependency_users_next(
    loom_type_use_iterator_t* iterator) {
  while (!iterator->carrier && iterator->pending_count) {
    const uint32_t edge = iterator->pending[iterator->pending_count - 1];
    const loom_dependency_ownership_t* ownership = loom_dependency_ownership(
        iterator->index, edge >> 1, iterator->owner_kind);
    const uint32_t next = ownership->edges[edge & 1].next;
    if (next) {
      iterator->pending[iterator->pending_count - 1] = next;
    } else {
      --iterator->pending_count;
    }
    iterator->carrier = ownership->first_carrier;
    if (ownership->first_parent) {
      iterator->pending[iterator->pending_count++] = ownership->first_parent;
    }
  }
  if (!iterator->carrier) {
    return NULL;
  }
  const loom_dependency_carrier_t* carrier =
      loom_dependency_carrier(iterator->index, iterator->carrier);
  iterator->carrier = carrier->next;
  return carrier;
}

void loom_type_users_begin(const loom_type_use_table_t* table,
                           loom_value_id_t value_id,
                           loom_type_use_iterator_t* out_iterator) {
  loom_dependency_users_begin(table, value_id, LOOM_DEPENDENCY_OWNER_VALUE,
                              out_iterator);
}

loom_value_id_t loom_type_users_next(loom_type_use_iterator_t* iterator) {
  const loom_dependency_carrier_t* carrier =
      loom_dependency_users_next(iterator);
  return carrier ? carrier->owner.value : LOOM_VALUE_ID_INVALID;
}

void loom_attribute_dependencies_begin(const loom_type_use_table_t* table,
                                       const loom_op_t* op,
                                       uint8_t attribute_index,
                                       loom_type_use_iterator_t* out_iterator) {
  const uint32_t carrier = loom_op_attribute_owners(op)[attribute_index];
  const uint32_t root =
      carrier ? loom_dependency_carrier(table->index, carrier)->active : 0;
  loom_type_dependency_set_begin(table, root, out_iterator);
}

void loom_attribute_dimension_dependencies_begin(
    const loom_type_use_table_t* table, const loom_op_t* op,
    uint8_t attribute_index, loom_type_use_iterator_t* out_iterator) {
  const uint32_t carrier = loom_op_attribute_owners(op)[attribute_index];
  const uint32_t root = carrier ? loom_dependency_carrier(table->index, carrier)
                                      ->dimension_dependencies
                                : 0;
  loom_type_dependency_set_begin(table, root, out_iterator);
}

void loom_attribute_users_begin(const loom_type_use_table_t* table,
                                loom_value_id_t value_id,
                                loom_type_use_iterator_t* out_iterator) {
  loom_dependency_users_begin(table, value_id, LOOM_DEPENDENCY_OWNER_ATTRIBUTE,
                              out_iterator);
}

loom_attribute_user_t loom_attribute_users_next(
    loom_type_use_iterator_t* iterator) {
  const loom_dependency_carrier_t* carrier =
      loom_dependency_users_next(iterator);
  return carrier ? (loom_attribute_user_t){carrier->owner.op,
                                           carrier->attribute_index}
                 : (loom_attribute_user_t){0};
}
