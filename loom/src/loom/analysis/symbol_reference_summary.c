// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbol_reference_summary.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/parameterized_type.h"
#include "loom/ir/structural_hash.h"

typedef enum loom_symbol_reference_summary_transform_e {
  LOOM_SYMBOL_REFERENCE_SUMMARY_IDENTITY = 0,
  LOOM_SYMBOL_REFERENCE_SUMMARY_DICTIONARY,
  LOOM_SYMBOL_REFERENCE_SUMMARY_TYPE,
  LOOM_SYMBOL_REFERENCE_SUMMARY_ENCODING,
  LOOM_SYMBOL_REFERENCE_SUMMARY_NESTED,
} loom_symbol_reference_summary_transform_t;

// Summaries retain ordered concatenations, not expanded reference vectors.
// Empty children disappear and unary nodes compose their kind transitions.
// Every retained branch has at least two nonempty children, bounding expansion
// by the number of emitted references even through deeply shared type graphs.
typedef struct loom_symbol_reference_summary_ref_t {
  // Null for an empty summary; otherwise an arena-owned completed node.
  const struct loom_symbol_reference_summary_node_t* node;
  // Kind transformation applied before entering node.
  loom_symbol_reference_summary_transform_t transform;
} loom_symbol_reference_summary_ref_t;

typedef struct loom_symbol_reference_summary_node_t {
  // Number of targets for a leaf or nonempty children for a branch.
  iree_host_size_t count;
  // Maximum expansion stack depth, including this node.
  iree_host_size_t height;
  // Interface constraints for a leaf's targets.
  loom_symbol_interface_flags_t interfaces;
  // Graph role for a leaf's targets.
  loom_symbol_reference_role_t role;
  // Whether values contains targets instead of child summaries.
  bool is_leaf;
  // Immutable ordered payload. Singleton leaves retain their target inline.
  union {
    // Inline target when a leaf has count one.
    loom_symbol_ref_t singleton;
    // Borrowed or arena-filtered targets when a leaf has count above one.
    const loom_symbol_ref_t* targets;
    // Trailing arena-owned references when this is a branch.
    const loom_symbol_reference_summary_ref_t* children;
  } values;
} loom_symbol_reference_summary_node_t;

typedef enum loom_symbol_reference_summary_domain_e {
  LOOM_SYMBOL_REFERENCE_SUMMARY_EMPTY = 0,
  LOOM_SYMBOL_REFERENCE_SUMMARY_TYPE_STORAGE,
  LOOM_SYMBOL_REFERENCE_SUMMARY_ATTR_STORAGE,
  LOOM_SYMBOL_REFERENCE_SUMMARY_ENCODING_STORAGE,
  LOOM_SYMBOL_REFERENCE_SUMMARY_ARRAY_STORAGE,
} loom_symbol_reference_summary_domain_t;

// Fixed-width storage identity, never a recursive hash or equality query.
typedef struct loom_symbol_reference_summary_key_t {
  // Type/attribute bits, normalized leaf semantics, and domain/depth tag.
  uint64_t words[4];
} loom_symbol_reference_summary_key_t;

// A compressed binary radix has at most 256 bit decisions per lookup and one
// fewer branches than entries. There is no input-sized collision probe loop.
typedef struct loom_symbol_reference_summary_branch_t {
  // Tagged branches or untagged entry pointers for the two bit values.
  uintptr_t children[2];
  // Discriminating bit, in increasing order from the root.
  uint16_t bit;
} loom_symbol_reference_summary_branch_t;

typedef enum loom_symbol_reference_summary_children_e {
  LOOM_SYMBOL_REFERENCE_SUMMARY_NO_CHILDREN = 0,
  LOOM_SYMBOL_REFERENCE_SUMMARY_TYPES,
  LOOM_SYMBOL_REFERENCE_SUMMARY_ATTRIBUTES,
  LOOM_SYMBOL_REFERENCE_SUMMARY_DICTIONARY_ENTRIES,
  LOOM_SYMBOL_REFERENCE_SUMMARY_ENCODING_ID,
} loom_symbol_reference_summary_children_t;

typedef struct loom_symbol_reference_summary_frame_t {
  // Index entry completed by this frame.
  struct loom_symbol_reference_summary_entry_t* entry;
  // Parent frame awaiting this completed structural result.
  struct loom_symbol_reference_summary_frame_t* parent;
  // Beginning of this frame's resolved children in the shared scratch stack.
  iree_host_size_t begin;
  // Next immediate child to discover, including an optional static encoding.
  iree_host_size_t next;
  // Number of immediate children after the optional static encoding.
  iree_host_size_t count;
  // Maximum height of resolved nonempty children.
  iree_host_size_t height;
  // Borrowed immediate payload interpreted by children_kind.
  const void* values;
  // Intrinsic parameter descriptors, or null for descriptor-free children.
  const loom_attr_descriptor_t* descriptors;
  // How to obtain immediate children from values or the entry key.
  loom_symbol_reference_summary_children_t children_kind;
  // Kind transition on every payload child.
  loom_symbol_reference_summary_transform_t transform;
  // Transition from the parent into this frame's result.
  loom_symbol_reference_summary_transform_t incoming_transform;
  // Optional static encoding visited before structural type children.
  uint16_t encoding_id;
  // Aggregate depth passed to immediate attribute children.
  uint8_t depth;
} loom_symbol_reference_summary_frame_t;

typedef struct loom_symbol_reference_summary_entry_t {
  // Exact immutable input representation and bounded interpretation context.
  loom_symbol_reference_summary_key_t key;
  // Next retained entry for bucket-array resizing, independent of graph edges.
  struct loom_symbol_reference_summary_entry_t* next;
  // Construction state becomes the retained result when postorder completes.
  union {
    // Completed result, including empty and bypassed unary summaries.
    loom_symbol_reference_summary_ref_t result;
    // Active frame owned by this stable arena allocation.
    loom_symbol_reference_summary_frame_t frame;
  } state;
  // Shallow exact-key hash retained for bucket-array resizing.
  uint32_t hash;
  // Postorder completion distinguishes empty results from active construction.
  bool complete;
} loom_symbol_reference_summary_entry_t;

typedef struct loom_symbol_reference_summary_cursor_t {
  // Completed node currently expanded.
  const loom_symbol_reference_summary_node_t* node;
  // Next child for a branch; leaves are consumed in one step.
  iree_host_size_t next;
  // Kind after the incoming edge transformation.
  loom_symbol_reference_occurrence_kind_t kind;
} loom_symbol_reference_summary_cursor_t;

static loom_symbol_reference_summary_transform_t
loom_symbol_reference_summary_compose(
    loom_symbol_reference_summary_transform_t outer,
    loom_symbol_reference_summary_transform_t inner) {
  if (inner >= LOOM_SYMBOL_REFERENCE_SUMMARY_TYPE ||
      outer == LOOM_SYMBOL_REFERENCE_SUMMARY_IDENTITY) {
    return inner;
  }
  return outer;
}

static loom_symbol_reference_occurrence_kind_t
loom_symbol_reference_summary_apply(
    loom_symbol_reference_summary_transform_t transform,
    loom_symbol_reference_occurrence_kind_t kind) {
  switch (transform) {
    case LOOM_SYMBOL_REFERENCE_SUMMARY_DICTIONARY:
      return kind >= LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR &&
                     kind <= LOOM_SYMBOL_REFERENCE_OCCURRENCE_GLOBAL_ACCESS
                 ? LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR
                 : kind;
    case LOOM_SYMBOL_REFERENCE_SUMMARY_TYPE:
      return LOOM_SYMBOL_REFERENCE_OCCURRENCE_TYPE_ATTR;
    case LOOM_SYMBOL_REFERENCE_SUMMARY_ENCODING:
      return LOOM_SYMBOL_REFERENCE_OCCURRENCE_ENCODING_ATTR;
    case LOOM_SYMBOL_REFERENCE_SUMMARY_NESTED:
      return LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR;
    default:
      return kind;
  }
}

static uint32_t loom_symbol_reference_summary_bit(
    const loom_symbol_reference_summary_key_t* key, uint16_t bit) {
  return (uint32_t)((key->words[bit / 64] >> (bit % 64)) & 1);
}

static loom_symbol_reference_summary_entry_t*
loom_symbol_reference_summary_find(
    uintptr_t current, const loom_symbol_reference_summary_key_t* key,
    uint16_t* out_bit) {
  *out_bit = 256;
  if (current) {
    while ((current & 1) != 0) {
      const loom_symbol_reference_summary_branch_t* branch =
          (const loom_symbol_reference_summary_branch_t*)(current &
                                                          ~(uintptr_t)1);
      current =
          branch->children[loom_symbol_reference_summary_bit(key, branch->bit)];
    }
    loom_symbol_reference_summary_entry_t* candidate =
        (loom_symbol_reference_summary_entry_t*)current;
    for (uint16_t word = 0; word < 4; ++word) {
      uint64_t difference = key->words[word] ^ candidate->key.words[word];
      if (difference) {
        *out_bit = (uint16_t)(word * 64 +
                              iree_math_count_trailing_zeros_u64(difference));
        break;
      }
    }
    return *out_bit == 256 ? candidate : NULL;
  }
  return NULL;
}

// The caller has established absence and retained the first differing bit.
static iree_status_t loom_symbol_reference_summary_attach(
    iree_arena_allocator_t* arena, uintptr_t* edge,
    loom_symbol_reference_summary_entry_t* entry, uint16_t bit) {
  if (!*edge) {
    *edge = (uintptr_t)entry;
  } else {
    loom_symbol_reference_summary_branch_t* branch = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, sizeof(*branch), (void**)&branch));
    while ((*edge & 1) != 0) {
      loom_symbol_reference_summary_branch_t* parent =
          (loom_symbol_reference_summary_branch_t*)(*edge & ~(uintptr_t)1);
      if (parent->bit >= bit) {
        break;
      }
      edge = &parent->children[loom_symbol_reference_summary_bit(&entry->key,
                                                                 parent->bit)];
    }
    const uint32_t direction =
        loom_symbol_reference_summary_bit(&entry->key, bit);
    branch->bit = bit;
    branch->children[direction] = (uintptr_t)entry;
    branch->children[direction ^ 1] = *edge;
    *edge = (uintptr_t)branch | 1;
  }
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_summary_grow(
    loom_symbol_reference_summary_t* summary) {
  const iree_host_size_t capacity = iree_max(16, summary->index.capacity * 2);
  uintptr_t* buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      summary->arena, capacity, sizeof(*buckets), (void**)&buckets));
  memset(buckets, 0, capacity * sizeof(*buckets));
  iree_status_t status = iree_ok_status();
  for (loom_symbol_reference_summary_entry_t* entry = summary->index.entries;
       entry && iree_status_is_ok(status); entry = entry->next) {
    uintptr_t* edge = &buckets[entry->hash & (capacity - 1)];
    uint16_t bit;
    loom_symbol_reference_summary_find(*edge, &entry->key, &bit);
    status =
        loom_symbol_reference_summary_attach(summary->arena, edge, entry, bit);
  }
  if (iree_status_is_ok(status)) {
    summary->index.values = buckets;
    summary->index.capacity = capacity;
  }
  return status;
}

static iree_status_t loom_symbol_reference_summary_intern(
    loom_symbol_reference_summary_t* summary,
    loom_symbol_reference_summary_key_t key,
    loom_symbol_reference_summary_entry_t** out_entry) {
  uint32_t hash = loom_structural_hash_initialize();
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(key.words); ++i) {
    hash = loom_structural_hash_mix_u64(hash, key.words[i]);
  }
  hash = loom_structural_hash_finalize(hash);
  uintptr_t* edge =
      &summary->index.values[hash & (summary->index.capacity - 1)];
  uint16_t bit;
  loom_symbol_reference_summary_entry_t* entry =
      loom_symbol_reference_summary_find(*edge, &key, &bit);
  if (entry) {
    // Constructor-owned acyclicity makes every reused entry complete.
    IREE_ASSERT(entry->complete);
    *out_entry = entry;
    return iree_ok_status();
  }
  if (summary->index.count + 1 > summary->index.capacity * 3 / 4) {
    IREE_RETURN_IF_ERROR(loom_symbol_reference_summary_grow(summary));
    edge = &summary->index.values[hash & (summary->index.capacity - 1)];
    loom_symbol_reference_summary_find(*edge, &key, &bit);
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(summary->arena, sizeof(*entry), (void**)&entry));
  *entry = (loom_symbol_reference_summary_entry_t){
      .key = key,
      .hash = hash,
      .next = summary->index.entries,
  };
  IREE_RETURN_IF_ERROR(
      loom_symbol_reference_summary_attach(summary->arena, edge, entry, bit));
  summary->index.entries = entry;
  ++summary->index.count;
  *out_entry = entry;
  return iree_ok_status();
}

static loom_symbol_reference_summary_key_t
loom_symbol_reference_summary_type_key(loom_type_t type) {
  loom_symbol_reference_summary_key_t key = {0};
  if (!loom_symbol_reference_type_may_contain_ref(type)) {
    return key;
  }
  memcpy(key.words, &type, sizeof(type));
  key.words[3] = LOOM_SYMBOL_REFERENCE_SUMMARY_TYPE_STORAGE;
  return key;
}

static loom_symbol_reference_summary_key_t
loom_symbol_reference_summary_encoding_key(uint16_t encoding_id) {
  loom_symbol_reference_summary_key_t key = {0};
  if (encoding_id) {
    key.words[0] = encoding_id;
    key.words[3] = LOOM_SYMBOL_REFERENCE_SUMMARY_ENCODING_STORAGE;
  }
  return key;
}

static loom_symbol_reference_summary_key_t
loom_symbol_reference_summary_attr_key(loom_attribute_t attr,
                                       const loom_attr_descriptor_t* descriptor,
                                       uint8_t depth) {
  loom_symbol_reference_summary_key_t key = {0};
  switch ((loom_attr_kind_t)attr.kind) {
    case LOOM_ATTR_SYMBOL:
    case LOOM_ATTR_SYMBOL_ARRAY:
    case LOOM_ATTR_SYMBOL_SET:
      if (descriptor && descriptor->attr_kind == attr.kind &&
          descriptor->reference.symbol_ref) {
        key.words[2] = descriptor->reference.symbol_ref->interfaces |
                       ((uint64_t)descriptor->reference.symbol_ref->role << 32);
      } else {
        key.words[2] = (uint64_t)LOOM_SYMBOL_REFERENCE_ROLE_DEPENDENCY << 32;
      }
      depth = 0;
      break;
    case LOOM_ATTR_TYPE:
    case LOOM_ATTR_ENCODING:
      depth = 0;
      break;
    case LOOM_ATTR_DICT:
    case LOOM_ATTR_PARAMETERIZED:
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      if (attr.count == 0 && depth < LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return key;
      }
      break;
    default:
      return key;
  }
  memcpy(key.words, &attr, sizeof(attr));
  key.words[3] =
      LOOM_SYMBOL_REFERENCE_SUMMARY_ATTR_STORAGE | ((uint64_t)depth << 8);
  return key;
}

static iree_status_t loom_symbol_reference_summary_make_leaf(
    loom_symbol_reference_summary_t* summary, const loom_symbol_ref_t* targets,
    iree_host_size_t count, uint64_t semantics,
    loom_symbol_reference_summary_ref_t* out_result) {
  *out_result = (loom_symbol_reference_summary_ref_t){0};
  if (count == 0) {
    return iree_ok_status();
  }
  loom_symbol_reference_summary_node_t* node = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(summary->arena, sizeof(*node), (void**)&node));
  *node = (loom_symbol_reference_summary_node_t){
      .count = count,
      .height = 1,
      .interfaces = (uint32_t)semantics,
      .role = (loom_symbol_reference_role_t)(semantics >> 32),
      .is_leaf = true,
  };
  if (count == 1) {
    node->values.singleton = targets[0];
  } else {
    node->values.targets = targets;
  }
  out_result->node = node;
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_summary_check_target(
    loom_symbol_reference_summary_t* summary, loom_symbol_ref_t target) {
  if (loom_symbol_ref_is_valid(target) && target.module_id == 0 &&
      target.symbol_id >= summary->module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target symbol id %u is out of range for %" PRIhsz " symbols",
        (unsigned)target.symbol_id, summary->module->symbols.count);
  }
  return iree_ok_status();
}

// Filter shared array storage once, independently of a field's role/interfaces.
// Valid arrays remain borrowed. A gap only allocates when a later local target
// requires compaction, so ignored suffixes also retain the original prefix.
static iree_status_t loom_symbol_reference_summary_array(
    loom_symbol_reference_summary_t* summary,
    loom_symbol_reference_summary_entry_t* entry, loom_attribute_t attr) {
  loom_symbol_reference_summary_key_t key = entry->key;
  key.words[2] = 0;
  key.words[3] = LOOM_SYMBOL_REFERENCE_SUMMARY_ARRAY_STORAGE;
  loom_symbol_reference_summary_entry_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      loom_symbol_reference_summary_intern(summary, key, &storage));
  if (!storage->complete) {
    const loom_symbol_ref_array_t refs = attr.kind == LOOM_ATTR_SYMBOL_SET
                                             ? loom_attr_as_symbol_set(attr)
                                             : loom_attr_as_symbol_array(attr);
    loom_symbol_ref_t* filtered = NULL;
    iree_host_size_t count = 0;
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0; i < refs.count && iree_status_is_ok(status);
         ++i) {
      const loom_symbol_ref_t target = refs.values[i];
      status = loom_symbol_reference_summary_check_target(summary, target);
      if (!iree_status_is_ok(status) || !loom_symbol_ref_is_valid(target) ||
          target.module_id != 0) {
        continue;
      }
      if (count != i && !filtered) {
        status = iree_arena_allocate_array(
            summary->arena, refs.count, sizeof(*filtered), (void**)&filtered);
        if (iree_status_is_ok(status)) {
          memcpy(filtered, refs.values, count * sizeof(*filtered));
        }
      }
      if (iree_status_is_ok(status)) {
        if (filtered) {
          filtered[count] = target;
        }
        ++count;
      }
    }
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(loom_symbol_reference_summary_make_leaf(
        summary, filtered ? filtered : refs.values, count,
        (uint64_t)LOOM_SYMBOL_REFERENCE_ROLE_DEPENDENCY << 32,
        &storage->state.result));
    storage->complete = true;
  }
  entry->state.result = storage->state.result;
  if (storage->state.result.node &&
      entry->key.words[2] !=
          ((uint64_t)LOOM_SYMBOL_REFERENCE_ROLE_DEPENDENCY << 32)) {
    const loom_symbol_reference_summary_node_t* node =
        storage->state.result.node;
    IREE_RETURN_IF_ERROR(loom_symbol_reference_summary_make_leaf(
        summary,
        node->count == 1 ? &node->values.singleton : node->values.targets,
        node->count, entry->key.words[2], &entry->state.result));
  }
  entry->complete = true;
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_summary_enter(
    loom_symbol_reference_summary_t* summary,
    loom_symbol_reference_summary_key_t key,
    loom_symbol_reference_summary_transform_t incoming_transform,
    loom_symbol_reference_summary_ref_t* out_result) {
  *out_result = (loom_symbol_reference_summary_ref_t){0};
  const uint8_t domain = (uint8_t)key.words[3];
  if (domain == LOOM_SYMBOL_REFERENCE_SUMMARY_EMPTY) {
    return iree_ok_status();
  }
  loom_symbol_reference_summary_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_symbol_reference_summary_intern(summary, key, &entry));
  if (entry->complete) {
    *out_result = entry->state.result;
    return iree_ok_status();
  }
  loom_symbol_reference_summary_frame_t frame = {
      .entry = entry,
      .begin = summary->children.count,
      .incoming_transform = incoming_transform,
  };
  if (domain == LOOM_SYMBOL_REFERENCE_SUMMARY_TYPE_STORAGE) {
    loom_type_t type;
    memcpy(&type, key.words, sizeof(type));
    if (loom_type_has_static_encoding(type)) {
      frame.encoding_id = type.encoding_id;
    }
    switch (loom_type_kind(type)) {
      case LOOM_TYPE_REFERENCE: {
        const loom_symbol_ref_t declaration = {
            .module_id = 0,
            .symbol_id = loom_type_reference_declaration_id(type),
        };
        IREE_RETURN_IF_ERROR(
            loom_symbol_reference_summary_check_target(summary, declaration));
        IREE_RETURN_IF_ERROR(loom_symbol_reference_summary_make_leaf(
            summary, &declaration, 1,
            LOOM_SYMBOL_INTERFACE_RECORD |
                ((uint64_t)LOOM_SYMBOL_REFERENCE_ROLE_DEPENDENCY << 32),
            &entry->state.result));
        entry->complete = true;
        break;
      }
      case LOOM_TYPE_FUNCTION: {
        const loom_func_type_data_t* data = loom_type_func_data(type);
        if (data) {
          frame.children_kind = LOOM_SYMBOL_REFERENCE_SUMMARY_TYPES;
          frame.values = data->types;
          frame.count = (iree_host_size_t)data->arg_count + data->result_count;
        }
        break;
      }
      case LOOM_TYPE_DIALECT:
        frame.children_kind = LOOM_SYMBOL_REFERENCE_SUMMARY_TYPES;
        frame.values = loom_type_dialect_params(type);
        frame.count = frame.values ? loom_type_dialect_param_count(type) : 0;
        break;
      case LOOM_TYPE_PARAMETERIZED:
        frame.children_kind = LOOM_SYMBOL_REFERENCE_SUMMARY_ATTRIBUTES;
        frame.values = loom_type_parameterized_parameters(type);
        frame.descriptors =
            loom_type_parameterized_descriptor(type)->parameter_descriptors;
        frame.count = loom_type_parameterized_parameter_count(type);
        frame.depth = 1;
        break;
      case LOOM_TYPE_REGISTER:
        frame.children_kind = LOOM_SYMBOL_REFERENCE_SUMMARY_TYPES;
        frame.values = loom_type_register_value_type(type);
        frame.count = frame.values ? 1 : 0;
        break;
      default:
        break;
    }
  } else if (domain == LOOM_SYMBOL_REFERENCE_SUMMARY_ENCODING_STORAGE) {
    const loom_encoding_t* encoding =
        loom_module_encoding(summary->module, (uint16_t)key.words[0]);
    if (!encoding) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "static encoding id %u is out of range for module with %" PRIhsz
          " encodings",
          (unsigned)key.words[0], summary->module->encodings.count);
    }
    if (encoding->attribute_count && !encoding->attributes) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "non-empty encoding attribute list has a NULL entry pointer");
    }
    frame.children_kind = LOOM_SYMBOL_REFERENCE_SUMMARY_DICTIONARY_ENTRIES;
    frame.values = encoding->attributes;
    frame.count = encoding->attribute_count;
  } else {
    loom_attribute_t attr;
    memcpy(&attr, key.words, sizeof(attr));
    const uint8_t depth = (uint8_t)(key.words[3] >> 8);
    switch ((loom_attr_kind_t)attr.kind) {
      case LOOM_ATTR_SYMBOL: {
        loom_symbol_ref_t target = loom_attr_as_symbol(attr);
        IREE_RETURN_IF_ERROR(
            loom_symbol_reference_summary_check_target(summary, target));
        IREE_RETURN_IF_ERROR(loom_symbol_reference_summary_make_leaf(
            summary, &target,
            loom_symbol_ref_is_valid(target) && target.module_id == 0 ? 1 : 0,
            key.words[2], &entry->state.result));
        entry->complete = true;
        break;
      }
      case LOOM_ATTR_SYMBOL_ARRAY:
      case LOOM_ATTR_SYMBOL_SET: {
        IREE_RETURN_IF_ERROR(
            loom_symbol_reference_summary_array(summary, entry, attr));
        break;
      }
      case LOOM_ATTR_TYPE:
        if (attr.type_id == LOOM_TYPE_ID_INVALID) {
          break;
        }
        if (attr.type_id >= summary->module->types.count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "type attribute id %u is out of range for module with %" PRIhsz
              " types",
              (unsigned)attr.type_id, summary->module->types.count);
        }
        frame.children_kind = LOOM_SYMBOL_REFERENCE_SUMMARY_TYPES;
        frame.values =
            loom_type_table_entry(&summary->module->types, attr.type_id);
        frame.count = 1;
        frame.transform = LOOM_SYMBOL_REFERENCE_SUMMARY_TYPE;
        break;
      case LOOM_ATTR_ENCODING:
        frame.children_kind = LOOM_SYMBOL_REFERENCE_SUMMARY_ENCODING_ID;
        frame.count = 1;
        frame.transform = LOOM_SYMBOL_REFERENCE_SUMMARY_ENCODING;
        break;
      case LOOM_ATTR_DICT:
      case LOOM_ATTR_PARAMETERIZED:
      case LOOM_ATTR_PARAMETERIZED_ARRAY:
        if (depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "%s attribute nesting exceeds max depth %u",
              attr.kind == LOOM_ATTR_DICT ? "dict" : "aggregate",
              (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
        }
        frame.depth = depth + 1;
        frame.count = attr.count;
        frame.transform = LOOM_SYMBOL_REFERENCE_SUMMARY_NESTED;
        frame.children_kind = LOOM_SYMBOL_REFERENCE_SUMMARY_ATTRIBUTES;
        if (attr.kind == LOOM_ATTR_DICT) {
          if (attr.count && !attr.dict_entries) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "non-empty dict attribute has a NULL entry pointer");
          }
          frame.children_kind =
              LOOM_SYMBOL_REFERENCE_SUMMARY_DICTIONARY_ENTRIES;
          frame.values = attr.dict_entries;
          frame.transform = LOOM_SYMBOL_REFERENCE_SUMMARY_DICTIONARY;
        } else if (attr.kind == LOOM_ATTR_PARAMETERIZED) {
          const loom_parameterized_attr_descriptor_t* descriptor =
              loom_context_resolve_parameterized_attr(
                  summary->module->context,
                  loom_attr_as_parameterized_kind(attr));
          frame.values = attr.parameterized_slots;
          frame.descriptors = descriptor->parameter_descriptors;
        } else {
          frame.values = attr.parameterized_array;
        }
        break;
      default:
        break;
    }
  }
  if (frame.count == 0 && frame.encoding_id == 0) {
    entry->complete = true;
    *out_result = entry->state.result;
    return iree_ok_status();
  }
  frame.parent = summary->construction;
  entry->state.frame = frame;
  summary->construction = &entry->state.frame;
  return iree_ok_status();
}

static loom_symbol_reference_summary_key_t loom_symbol_reference_summary_child(
    loom_symbol_reference_summary_frame_t* frame,
    loom_symbol_reference_summary_transform_t* out_transform) {
  iree_host_size_t index = frame->next++;
  if (frame->encoding_id) {
    if (index == 0) {
      *out_transform = LOOM_SYMBOL_REFERENCE_SUMMARY_IDENTITY;
      return loom_symbol_reference_summary_encoding_key(frame->encoding_id);
    }
    --index;
  }
  *out_transform = frame->transform;
  switch (frame->children_kind) {
    case LOOM_SYMBOL_REFERENCE_SUMMARY_TYPES:
      return loom_symbol_reference_summary_type_key(
          ((const loom_type_t*)frame->values)[index]);
    case LOOM_SYMBOL_REFERENCE_SUMMARY_DICTIONARY_ENTRIES:
      return loom_symbol_reference_summary_attr_key(
          ((const loom_named_attr_t*)frame->values)[index].value, NULL,
          frame->depth);
    case LOOM_SYMBOL_REFERENCE_SUMMARY_ENCODING_ID: {
      loom_attribute_t attr;
      memcpy(&attr, frame->entry->key.words, sizeof(attr));
      return loom_symbol_reference_summary_encoding_key(attr.encoding_id);
    }
    default:
      return loom_symbol_reference_summary_attr_key(
          ((const loom_attribute_t*)frame->values)[index],
          frame->descriptors ? &frame->descriptors[index] : NULL, frame->depth);
  }
}

static iree_status_t loom_symbol_reference_summary_append_child(
    loom_symbol_reference_summary_t* summary,
    loom_symbol_reference_summary_ref_t child,
    loom_symbol_reference_summary_transform_t transform) {
  if (!child.node) {
    return iree_ok_status();
  }
  if (summary->children.count == summary->children.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        summary->arena, summary->children.count, 16,
        sizeof(*summary->children.values), &summary->children.capacity,
        (void**)&summary->children.values));
  }
  child.transform =
      loom_symbol_reference_summary_compose(transform, child.transform);
  summary->children.values[summary->children.count++] = child;
  loom_symbol_reference_summary_frame_t* frame = summary->construction;
  frame->height = iree_max(frame->height, child.node->height);
  return iree_ok_status();
}

static iree_status_t loom_symbol_reference_summary_resolve(
    loom_symbol_reference_summary_t* summary,
    loom_symbol_reference_summary_key_t key,
    loom_symbol_reference_summary_ref_t* out_result) {
  IREE_RETURN_IF_ERROR(loom_symbol_reference_summary_enter(
      summary, key, LOOM_SYMBOL_REFERENCE_SUMMARY_IDENTITY, out_result));
  iree_status_t status = iree_ok_status();
  while (summary->construction && iree_status_is_ok(status)) {
    loom_symbol_reference_summary_frame_t* frame = summary->construction;
    if (frame->next < frame->count + (frame->encoding_id != 0)) {
      loom_symbol_reference_summary_transform_t transform;
      loom_symbol_reference_summary_key_t child_key =
          loom_symbol_reference_summary_child(frame, &transform);
      const loom_symbol_reference_summary_frame_t* previous_frame =
          summary->construction;
      loom_symbol_reference_summary_ref_t child = {0};
      status = loom_symbol_reference_summary_enter(summary, child_key,
                                                   transform, &child);
      if (iree_status_is_ok(status) &&
          summary->construction == previous_frame) {
        status = loom_symbol_reference_summary_append_child(summary, child,
                                                            transform);
      }
      continue;
    }
    const iree_host_size_t count = summary->children.count - frame->begin;
    loom_symbol_reference_summary_ref_t result = {0};
    if (count == 1) {
      result = summary->children.values[frame->begin];
    } else if (count > 1) {
      loom_symbol_reference_summary_node_t* node = NULL;
      status = iree_arena_allocate(
          summary->arena,
          sizeof(*node) + count * sizeof(*summary->children.values),
          (void**)&node);
      if (iree_status_is_ok(status)) {
        loom_symbol_reference_summary_ref_t* children =
            (loom_symbol_reference_summary_ref_t*)(node + 1);
        memcpy(children, summary->children.values + frame->begin,
               count * sizeof(*children));
        *node = (loom_symbol_reference_summary_node_t){
            .count = count,
            .height = frame->height + 1,
            .values.children = children,
        };
        result.node = node;
      }
    }
    if (iree_status_is_ok(status)) {
      loom_symbol_reference_summary_entry_t* entry = frame->entry;
      summary->children.count = frame->begin;
      const loom_symbol_reference_summary_transform_t transform =
          frame->incoming_transform;
      summary->construction = frame->parent;
      entry->state.result = result;
      entry->complete = true;
      if (summary->construction) {
        status = loom_symbol_reference_summary_append_child(summary, result,
                                                            transform);
      } else {
        *out_result = result;
      }
    }
  }
  return status;
}

static iree_status_t loom_symbol_reference_summary_query(
    loom_symbol_reference_summary_t* summary,
    loom_symbol_reference_summary_key_t key,
    loom_symbol_reference_occurrence_kind_t kind) {
  summary->cursor.count = 0;
  loom_symbol_reference_summary_ref_t result = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbol_reference_summary_resolve(summary, key, &result));
  if (!result.node) {
    return iree_ok_status();
  }
  if (result.node->height > summary->cursor.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        summary->arena, 0, result.node->height, sizeof(*summary->cursor.values),
        &summary->cursor.capacity, (void**)&summary->cursor.values));
  }
  summary->cursor.values[0] = (loom_symbol_reference_summary_cursor_t){
      .node = result.node,
      .kind = loom_symbol_reference_summary_apply(result.transform, kind),
  };
  summary->cursor.count = 1;
  return iree_ok_status();
}

void loom_symbol_reference_summary_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_symbol_reference_summary_t* summary) {
  *summary = (loom_symbol_reference_summary_t){
      .module = module,
      .arena = arena,
      .index = {.values = summary->index.initial, .capacity = 4},
  };
}

iree_status_t loom_symbol_reference_summary_query_type(
    loom_symbol_reference_summary_t* summary, loom_type_t type,
    loom_symbol_reference_occurrence_kind_t kind) {
  return loom_symbol_reference_summary_query(
      summary, loom_symbol_reference_summary_type_key(type), kind);
}

iree_status_t loom_symbol_reference_summary_query_attr(
    loom_symbol_reference_summary_t* summary, loom_attribute_t attr,
    const loom_attr_descriptor_t* descriptor,
    loom_symbol_reference_occurrence_kind_t kind) {
  return loom_symbol_reference_summary_query(
      summary, loom_symbol_reference_summary_attr_key(attr, descriptor, 0),
      kind);
}

iree_status_t loom_symbol_reference_summary_query_encoding(
    loom_symbol_reference_summary_t* summary, uint16_t encoding_id,
    loom_symbol_reference_occurrence_kind_t kind) {
  return loom_symbol_reference_summary_query(
      summary, loom_symbol_reference_summary_encoding_key(encoding_id), kind);
}

bool loom_symbol_reference_summary_next(
    loom_symbol_reference_summary_t* summary,
    loom_symbol_reference_summary_span_t* out_span) {
  while (summary->cursor.count) {
    loom_symbol_reference_summary_cursor_t* frame =
        &summary->cursor.values[summary->cursor.count - 1];
    const loom_symbol_reference_summary_node_t* node = frame->node;
    if (node->is_leaf) {
      *out_span = (loom_symbol_reference_summary_span_t){
          .targets =
              node->count == 1 ? &node->values.singleton : node->values.targets,
          .count = node->count,
          .interfaces = node->interfaces,
          .kind = frame->kind,
          .role = node->role,
      };
      --summary->cursor.count;
      return true;
    }
    if (frame->next == node->count) {
      --summary->cursor.count;
      continue;
    }
    const loom_symbol_reference_summary_ref_t child =
        node->values.children[frame->next++];
    summary->cursor.values[summary->cursor.count++] =
        (loom_symbol_reference_summary_cursor_t){
            .node = child.node,
            .kind = loom_symbol_reference_summary_apply(child.transform,
                                                        frame->kind),
        };
  }
  return false;
}
