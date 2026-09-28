// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/module.h"
#include "loom/rewrite/remap.h"

// Immediate payloads retain the allocation-free path for ordinary scalar and
// inline shaped types. Recursive payloads use the invocation-local walk below.

static iree_status_t loom_ir_remap_leaf_type(loom_ir_remap_t* remap,
                                             loom_type_t source_type,
                                             uint16_t target_encoding_id,
                                             loom_type_t* out_target_type) {
  loom_type_t target_type = source_type;
  bool needs_interned_payload = false;
  loom_overflow_dim_t target_overflow_dims[LOOM_TYPE_MAX_RANK] = {0};
  if (loom_type_is_shaped(source_type) || loom_type_is_pool(source_type)) {
    uint8_t rank = loom_type_rank(source_type);
    if (loom_type_has_inline_dims(source_type)) {
      for (uint8_t i = 0; i < rank; ++i) {
        uint64_t dim = target_type.dims[i];
        if (!loom_dim_is_dynamic(dim)) {
          continue;
        }
        loom_value_id_t target_value = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
            remap, loom_dim_value_id(dim), &target_value));
        target_type.dims[i] = loom_dim_pack_dynamic(target_value);
      }
    } else if (rank > 0) {
      const loom_overflow_dim_t* source_dims =
          (const loom_overflow_dim_t*)(uintptr_t)source_type.dims[0];
      if (!source_dims) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "rank-%u type has a NULL overflow dim payload",
                                (unsigned)rank);
      }
      for (uint8_t i = 0; i < rank; ++i) {
        target_overflow_dims[i] = source_dims[i];
        if (!loom_dim_is_dynamic(target_overflow_dims[i])) {
          continue;
        }
        loom_value_id_t target_value = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
            remap, loom_dim_value_id(target_overflow_dims[i]), &target_value));
        target_overflow_dims[i] = loom_dim_pack_dynamic(target_value);
      }
      target_type.dims[0] = (uint64_t)(uintptr_t)target_overflow_dims;
      target_type.dims[1] = 0;
      needs_interned_payload = true;
    }
  }

  if (loom_type_has_ssa_encoding(source_type)) {
    loom_value_id_t target_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
        remap, loom_type_encoding_value_id(source_type), &target_value));
    if (target_value > UINT16_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "cannot store target value %%%u in a 16-bit SSA encoding reference",
          (unsigned)target_value);
    }
    target_type.encoding_id = (uint16_t)target_value;
  } else if (loom_type_has_static_encoding(source_type)) {
    target_type.encoding_id = target_encoding_id;
  }

  if (needs_interned_payload) {
    IREE_RETURN_IF_ERROR(loom_module_intern_type(remap->target_module,
                                                 target_type, &target_type));
  }
  *out_target_type = target_type;
  return iree_ok_status();
}

static bool loom_ir_remap_is_initialized(const loom_ir_remap_t* remap) {
  return remap && remap->source_module && remap->target_module && remap->arena;
}

static iree_status_t loom_ir_remap_predicate_list_into(
    loom_ir_remap_t* remap, const loom_predicate_t* source_predicates,
    iree_host_size_t predicate_count, iree_arena_allocator_t* payload_arena,
    loom_predicate_t** out_target_predicates) {
  *out_target_predicates = NULL;
  if (predicate_count == 0) {
    return iree_ok_status();
  }
  if (!source_predicates) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-empty predicate list has a NULL source pointer");
  }
  if (predicate_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "predicate list has %" PRIhsz " entries, max %u",
                            predicate_count, (unsigned)UINT16_MAX);
  }

  loom_predicate_t* target_predicates = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(payload_arena, predicate_count,
                                                 sizeof(loom_predicate_t),
                                                 (void**)&target_predicates));
  for (iree_host_size_t i = 0; i < predicate_count; ++i) {
    target_predicates[i] = source_predicates[i];
    if (target_predicates[i].arg_count >
        IREE_ARRAYSIZE(target_predicates[i].args)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "predicate %" PRIhsz " has %u args, max %" PRIhsz,
                              i, (unsigned)target_predicates[i].arg_count,
                              IREE_ARRAYSIZE(target_predicates[i].args));
    }
    for (uint8_t arg_index = 0; arg_index < target_predicates[i].arg_count;
         ++arg_index) {
      switch (
          (loom_predicate_arg_tag_t)target_predicates[i].arg_tags[arg_index]) {
        case LOOM_PRED_ARG_NONE:
        case LOOM_PRED_ARG_CONST:
          continue;
        case LOOM_PRED_ARG_VALUE:
          break;
        default:
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "predicate %" PRIhsz " arg %u has unknown tag %u", i,
              (unsigned)arg_index,
              (unsigned)target_predicates[i].arg_tags[arg_index]);
      }
      loom_value_id_t target_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
          remap, (loom_value_id_t)target_predicates[i].args[arg_index],
          &target_value));
      target_predicates[i].args[arg_index] = (int64_t)target_value;
    }
  }
  *out_target_predicates = target_predicates;
  return iree_ok_status();
}

iree_status_t loom_ir_remap_predicate_list(
    loom_ir_remap_t* remap, const loom_predicate_t* source_predicates,
    iree_host_size_t predicate_count,
    loom_predicate_t** out_target_predicates) {
  iree_arena_allocator_t* payload_arena =
      loom_ir_remap_is_initialized(remap) ? &remap->target_module->arena : NULL;
  return loom_ir_remap_predicate_list_into(remap, source_predicates,
                                           predicate_count, payload_arena,
                                           out_target_predicates);
}

static iree_status_t loom_ir_remap_leaf_attribute(
    loom_ir_remap_t* remap, loom_attribute_t source_attr,
    iree_arena_allocator_t* payload_arena, loom_attribute_t* out_target_attr) {
  *out_target_attr = source_attr;

  switch ((loom_attr_kind_t)source_attr.kind) {
    case LOOM_ATTR_ABSENT:
    case LOOM_ATTR_I64:
    case LOOM_ATTR_F64:
    case LOOM_ATTR_BOOL:
    case LOOM_ATTR_ENUM:
    case LOOM_ATTR_SCOPED_ENUM:
      return iree_ok_status();

    case LOOM_ATTR_STRING:
      return loom_ir_remap_string_id(remap, source_attr.string_id,
                                     /*allow_invalid=*/false,
                                     &out_target_attr->string_id);

    case LOOM_ATTR_SYMBOL:
      return loom_ir_remap_symbol_ref(remap, source_attr.symbol,
                                      &out_target_attr->symbol);

    case LOOM_ATTR_SYMBOL_ARRAY:
    case LOOM_ATTR_SYMBOL_SET: {
      if (source_attr.count == 0) {
        *out_target_attr = source_attr.kind == LOOM_ATTR_SYMBOL_SET
                               ? loom_attr_symbol_set(NULL, 0)
                               : loom_attr_symbol_array(NULL, 0);
        return iree_ok_status();
      }
      loom_symbol_ref_t* target_refs = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          payload_arena, source_attr.count, sizeof(*target_refs),
          (void**)&target_refs));
      for (uint16_t i = 0; i < source_attr.count; ++i) {
        IREE_RETURN_IF_ERROR(loom_ir_remap_symbol_ref(
            remap, source_attr.symbol_refs[i], &target_refs[i]));
      }
      if (source_attr.kind == LOOM_ATTR_SYMBOL_SET) {
        loom_symbol_ref_t duplicate_ref = loom_module_canonicalize_symbol_set(
            remap->target_module, target_refs, source_attr.count);
        if (loom_symbol_ref_is_valid(duplicate_ref)) {
          const loom_symbol_t* duplicate_symbol =
              &remap->target_module->symbols.entries[duplicate_ref.symbol_id];
          iree_string_view_t duplicate_name = loom_string_table_get(
              &remap->target_module->strings, duplicate_symbol->name_id);
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "remapped symbol set contains duplicate '@%.*s'",
              (int)duplicate_name.size, duplicate_name.data);
        }
        *out_target_attr = loom_attr_symbol_set(target_refs, source_attr.count);
      } else {
        *out_target_attr =
            loom_attr_symbol_array(target_refs, source_attr.count);
      }
      return iree_ok_status();
    }

    case LOOM_ATTR_I64_ARRAY: {
      if (source_attr.count == 0) {
        out_target_attr->i64_array = NULL;
        return iree_ok_status();
      }
      if (!source_attr.i64_array) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty i64 array attribute has a NULL payload");
      }
      int64_t* target_values = NULL;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(payload_arena, source_attr.count,
                                    sizeof(int64_t), (void**)&target_values));
      memcpy(target_values, source_attr.i64_array,
             (iree_host_size_t)source_attr.count * sizeof(int64_t));
      out_target_attr->i64_array = target_values;
      return iree_ok_status();
    }

    case LOOM_ATTR_ENUM_ARRAY: {
      if (source_attr.count == 0) {
        out_target_attr->enum_array = NULL;
        return iree_ok_status();
      }
      if (!source_attr.enum_array) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty enum array attribute has a NULL payload");
      }
      uint8_t* target_values = NULL;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(payload_arena, source_attr.count,
                                    sizeof(uint8_t), (void**)&target_values));
      memcpy(target_values, source_attr.enum_array, source_attr.count);
      out_target_attr->enum_array = target_values;
      return iree_ok_status();
    }

    case LOOM_ATTR_SIGNED_ENUM_SET: {
      if (source_attr.count == 0) {
        out_target_attr->signed_enum_set_words = NULL;
        return iree_ok_status();
      }
      if (!source_attr.signed_enum_set_words) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "non-empty signed enum-set attribute has a NULL payload");
      }
      uint64_t* target_words = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          payload_arena, (iree_host_size_t)source_attr.count * 2,
          sizeof(*target_words), (void**)&target_words));
      memcpy(target_words, source_attr.signed_enum_set_words,
             (iree_host_size_t)source_attr.count * 2 * sizeof(*target_words));
      out_target_attr->signed_enum_set_words = target_words;
      return iree_ok_status();
    }

    case LOOM_ATTR_BYTES: {
      const uint32_t byte_length = source_attr.reserved_1;
      if (byte_length == 0) {
        out_target_attr->bytes = NULL;
        return iree_ok_status();
      }
      if (!source_attr.bytes) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "non-empty bytes attribute has a NULL payload");
      }
      uint8_t* target_bytes = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(payload_arena, byte_length,
                                               (void**)&target_bytes));
      memcpy(target_bytes, source_attr.bytes, byte_length);
      out_target_attr->bytes = target_bytes;
      return iree_ok_status();
    }

    case LOOM_ATTR_PREDICATE_LIST: {
      loom_predicate_t* target_predicates = NULL;
      IREE_RETURN_IF_ERROR(loom_ir_remap_predicate_list_into(
          remap, source_attr.predicate_list, source_attr.count, payload_arena,
          &target_predicates));
      out_target_attr->predicate_list = target_predicates;
      return iree_ok_status();
    }

    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown attribute kind %u",
                              (unsigned)source_attr.kind);
  }
}

//===----------------------------------------------------------------------===//
// Invocation-local payload graph
//===----------------------------------------------------------------------===//

typedef enum loom_payload_kind_e {
  LOOM_PAYLOAD_TYPE,
  LOOM_PAYLOAD_ATTRIBUTE,
  LOOM_PAYLOAD_ENCODING,
} loom_payload_kind_t;

// A suspended payload becomes a memo leaf when its type or encoding completes.
// Attribute continuations are recycled instead: aggregate shape has bounded
// nesting, while TYPE-valued attributes can lead into arbitrarily deep types.
typedef struct loom_payload_frame_t {
  // Suspended owner, or next available attribute frame after completion.
  struct loom_payload_frame_t* parent;
  // Stable output slot in the parent or invoking call.
  void* destination;
  // Representation and completion policy for this frame.
  loom_payload_kind_t kind;
  // Immutable input, borrowed until the enclosing public remap call returns.
  union {
    // By-value type identity, including the immediate payload address.
    loom_type_t type;
    // Aggregate attribute or reference to another graph node.
    loom_attribute_t attribute;
    // Source-module static encoding identity.
    uint16_t encoding;
  } source;
  // Reconstructed child storage, owned by scratch or attribute payload arena.
  union {
    // Canonical target IDs for structural type children.
    loom_type_id_t* types;
    // Slots of parameterized types or aggregate attributes.
    loom_attribute_t* attributes;
    // Named children of dictionaries and static encodings.
    loom_named_attr_t* entries;
  } children;
  // Borrowed structural type children; NULL for other representations.
  const loom_type_t* source_types;
  // Number of immediate children.
  uint32_t count;
  // Next child to request; its output storage remains stable while suspended.
  uint32_t next;
  // Completed target ID, or the target of a single TYPE/ENCODING reference.
  uint32_t result;
  // Consecutive aggregate attribute nesting; type boundaries begin a new shape.
  uint32_t aggregate_depth;
  // Target module storage for public attributes, scratch for canonical inputs.
  iree_arena_allocator_t* payload_arena;
  // Memo links: tagged completed leaves or untagged intrusive radix branches.
  uintptr_t edges[2];
  // Differing identity bit, decreasing along radix branches.
  uint32_t bit;
} loom_payload_frame_t;

typedef struct loom_payload_walk_t {
  // Caller-owned correspondence, immutable during this invocation.
  loom_ir_remap_t* remap;
  // Frames, completed results and temporary canonical-construction payloads.
  iree_arena_allocator_t scratch;
  // Separate sparse identity maps for types and static encodings.
  uintptr_t memo[2];
  // Most recently suspended frame, or NULL when the walk is complete.
  loom_payload_frame_t* top;
  // Reusable attribute frames; completed type/encoding frames belong to memo.
  loom_payload_frame_t* free_frames;
  // Inline root keeps a short signature from acquiring a scratch block.
  loom_payload_frame_t root;
  // Inline children for ordinary short signatures.
  loom_type_id_t root_children[16];
} loom_payload_walk_t;

static uint64_t loom_payload_word(const loom_payload_frame_t* frame,
                                  uint32_t word) {
  if (frame->kind == LOOM_PAYLOAD_ENCODING) {
    return word == 0 ? frame->source.encoding : 0;
  }
  const loom_type_t type = frame->source.type;
  return word > 0 ? type.dims[word - 1]
                  : (uint64_t)type.header | ((uint64_t)type.encoding_id << 32) |
                        ((uint64_t)type.encoding_flags << 48);
}

static uint32_t loom_payload_side(const loom_payload_frame_t* frame,
                                  uint32_t bit) {
  return (uint32_t)((loom_payload_word(frame, bit / 64) >> (bit % 64)) & 1);
}

static loom_payload_frame_t* loom_payload_frame(uintptr_t edge) {
  return (loom_payload_frame_t*)(edge & ~(uintptr_t)1);
}

static bool loom_payload_find(uintptr_t edge, const loom_payload_frame_t* key,
                              uint32_t* out_result) {
  if (!edge) {
    return false;
  }
  while (!(edge & 1)) {
    const loom_payload_frame_t* branch = loom_payload_frame(edge);
    edge = branch->edges[loom_payload_side(key, branch->bit)];
  }
  const loom_payload_frame_t* leaf = loom_payload_frame(edge);
  for (uint32_t word = 0; word < 3; ++word) {
    if (loom_payload_word(key, word) != loom_payload_word(leaf, word)) {
      return false;
    }
  }
  *out_result = leaf->result;
  return true;
}

static void loom_payload_insert(uintptr_t* root, loom_payload_frame_t* frame) {
  const uintptr_t leaf = (uintptr_t)frame | 1;
  if (!*root) {
    *root = leaf;
    return;
  }
  uintptr_t existing = *root;
  while (!(existing & 1)) {
    const loom_payload_frame_t* branch = loom_payload_frame(existing);
    existing = branch->edges[loom_payload_side(frame, branch->bit)];
  }
  const loom_payload_frame_t* other = loom_payload_frame(existing);
  uint32_t word = 2;
  uint64_t difference =
      loom_payload_word(frame, word) ^ loom_payload_word(other, word);
  while (!difference && word > 0) {
    --word;
    difference =
        loom_payload_word(frame, word) ^ loom_payload_word(other, word);
  }
  // Canonical graphs are acyclic; a child completes before a later occurrence
  // can request it. Only distinct input identities create completed frames.
  IREE_ASSERT(difference != 0);
  frame->bit = word * 64 + 63 - iree_math_count_leading_zeros_u64(difference);
  uintptr_t* edge = root;
  while (!(*edge & 1)) {
    loom_payload_frame_t* branch = loom_payload_frame(*edge);
    if (branch->bit < frame->bit) {
      break;
    }
    edge = &branch->edges[loom_payload_side(frame, branch->bit)];
  }
  const uint32_t side = loom_payload_side(frame, frame->bit);
  frame->edges[side] = leaf;
  frame->edges[side ^ 1] = *edge;
  *edge = (uintptr_t)frame;
}

static iree_status_t loom_payload_push(loom_payload_walk_t* walk,
                                       loom_payload_frame_t** out_frame) {
  loom_payload_frame_t* frame = &walk->root;
  if (walk->top) {
    frame = walk->free_frames;
    if (frame) {
      walk->free_frames = frame->parent;
    } else {
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate(&walk->scratch, sizeof(*frame), (void**)&frame));
    }
  }
  *frame = (loom_payload_frame_t){.parent = walk->top};
  walk->top = frame;
  *out_frame = frame;
  return iree_ok_status();
}

static void loom_payload_pop(loom_payload_walk_t* walk) {
  loom_payload_frame_t* frame = walk->top;
  walk->top = frame->parent;
  if (frame->kind != LOOM_PAYLOAD_ATTRIBUTE) {
    loom_payload_insert(&walk->memo[frame->kind == LOOM_PAYLOAD_ENCODING],
                        frame);
  } else if (frame != &walk->root) {
    frame->parent = walk->free_frames;
    walk->free_frames = frame;
  }
}

static bool loom_payload_type_requires_walk(loom_type_t type) {
  switch (loom_type_kind(type)) {
    case LOOM_TYPE_FUNCTION:
    case LOOM_TYPE_DIALECT:
    case LOOM_TYPE_PARAMETERIZED:
      return true;
    case LOOM_TYPE_REGISTER:
      return loom_type_register_has_value_type(type);
    default:
      return loom_type_has_static_encoding(type);
  }
}

static iree_status_t loom_payload_request_type(loom_payload_walk_t* walk,
                                               loom_type_t type,
                                               loom_type_id_t* destination) {
  if (!loom_type_kind_is_valid(loom_type_kind(type))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown type kind %u",
                            (unsigned)loom_type_kind(type));
  }
  if (!loom_payload_type_requires_walk(type)) {
    loom_type_t target_type;
    IREE_RETURN_IF_ERROR(loom_ir_remap_leaf_type(
        walk->remap, type, type.encoding_id, &target_type));
    return loom_module_intern_type_id(walk->remap->target_module, target_type,
                                      destination);
  }
  const loom_payload_frame_t key = {.kind = LOOM_PAYLOAD_TYPE,
                                    .source.type = type};
  if (loom_payload_find(walk->memo[0], &key, destination)) {
    return iree_ok_status();
  }
  loom_payload_frame_t* frame = NULL;
  IREE_RETURN_IF_ERROR(loom_payload_push(walk, &frame));
  frame->kind = LOOM_PAYLOAD_TYPE;
  frame->source.type = type;
  frame->destination = destination;
  switch (loom_type_kind(type)) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      if (!data) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "function type has a NULL argument/result payload");
      }
      frame->source_types = data->types;
      frame->count = (uint32_t)data->arg_count + data->result_count;
      break;
    }
    case LOOM_TYPE_DIALECT:
      frame->source_types = loom_type_dialect_params(type);
      frame->count = loom_type_dialect_param_count(type);
      break;
    case LOOM_TYPE_REGISTER:
      frame->source_types = loom_type_register_value_type(type);
      frame->count = 1;
      break;
    case LOOM_TYPE_PARAMETERIZED:
      frame->count = loom_type_parameterized_parameter_count(type);
      return iree_arena_allocate_array(&walk->scratch, frame->count,
                                       sizeof(loom_attribute_t),
                                       (void**)&frame->children.attributes);
    default:
      // Inline shaped types reach the walk only for a static encoding edge.
      frame->count = 1;
      return iree_ok_status();
  }
  if (frame->count && !frame->source_types) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "non-empty source type list has a NULL payload");
  }
  if (frame == &walk->root &&
      frame->count <= IREE_ARRAYSIZE(walk->root_children)) {
    frame->children.types = walk->root_children;
    return iree_ok_status();
  }
  return iree_arena_allocate_array(&walk->scratch, frame->count,
                                   sizeof(loom_type_id_t),
                                   (void**)&frame->children.types);
}

static iree_status_t loom_payload_request_encoding(loom_payload_walk_t* walk,
                                                   uint16_t source_id,
                                                   uint32_t* destination) {
  const loom_encoding_t* source =
      loom_module_encoding(walk->remap->source_module, source_id);
  if (!source) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source encoding id %u out of range (source module has %" PRIhsz
        " encodings)",
        (unsigned)source_id, walk->remap->source_module->encodings.count);
  }
  if (walk->remap->source_module == walk->remap->target_module) {
    *destination = source_id;
    return iree_ok_status();
  }
  const loom_payload_frame_t key = {.kind = LOOM_PAYLOAD_ENCODING,
                                    .source.encoding = source_id};
  if (loom_payload_find(walk->memo[1], &key, destination)) {
    return iree_ok_status();
  }
  loom_payload_frame_t* frame = NULL;
  IREE_RETURN_IF_ERROR(loom_payload_push(walk, &frame));
  frame->kind = LOOM_PAYLOAD_ENCODING;
  frame->source.encoding = source_id;
  frame->destination = destination;
  frame->count = source->attribute_count;
  return iree_arena_allocate_array(&walk->scratch, frame->count,
                                   sizeof(loom_named_attr_t),
                                   (void**)&frame->children.entries);
}

static iree_status_t loom_payload_request_attribute(
    loom_payload_walk_t* walk, loom_attribute_t attribute,
    uint32_t aggregate_depth, iree_arena_allocator_t* payload_arena,
    loom_attribute_t* destination) {
  switch ((loom_attr_kind_t)attribute.kind) {
    case LOOM_ATTR_TYPE:
    case LOOM_ATTR_ENCODING:
      break;
    case LOOM_ATTR_DICT:
    case LOOM_ATTR_PARAMETERIZED:
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      if (aggregate_depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "aggregate attribute nesting exceeds max depth %u",
            (unsigned)LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH);
      }
      break;
    default:
      return loom_ir_remap_leaf_attribute(walk->remap, attribute, payload_arena,
                                          destination);
  }
  loom_payload_frame_t* frame = NULL;
  IREE_RETURN_IF_ERROR(loom_payload_push(walk, &frame));
  frame->kind = LOOM_PAYLOAD_ATTRIBUTE;
  frame->source.attribute = attribute;
  frame->destination = destination;
  frame->payload_arena = payload_arena;
  frame->aggregate_depth = aggregate_depth;
  if (attribute.kind == LOOM_ATTR_TYPE ||
      attribute.kind == LOOM_ATTR_ENCODING) {
    frame->count = 1;
    return iree_ok_status();
  }
  frame->count = attribute.count;
  if (attribute.kind == LOOM_ATTR_DICT) {
    if (attribute.count && !attribute.dict_entries) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "non-empty dict attribute has a NULL entry pointer");
    }
    // Canonical dictionary construction publishes its own durable payload.
    frame->payload_arena = &walk->scratch;
    return iree_arena_allocate_array(&walk->scratch, frame->count,
                                     sizeof(loom_named_attr_t),
                                     (void**)&frame->children.entries);
  }
  const loom_attribute_t* source = attribute.kind == LOOM_ATTR_PARAMETERIZED
                                       ? attribute.parameterized_slots
                                       : attribute.parameterized_array;
  if (attribute.count && !source) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "non-empty aggregate attribute has a NULL payload");
  }
  return iree_arena_allocate_array(payload_arena, frame->count,
                                   sizeof(loom_attribute_t),
                                   (void**)&frame->children.attributes);
}

static iree_status_t loom_payload_finish_type(loom_payload_walk_t* walk) {
  loom_payload_frame_t* frame = walk->top;
  loom_module_t* target = walk->remap->target_module;
  loom_type_t type = frame->source.type;
  switch (loom_type_kind(type)) {
    case LOOM_TYPE_PARAMETERIZED: {
      IREE_RETURN_IF_ERROR(loom_module_make_parameterized_type(
          target, loom_type_parameterized_descriptor(type),
          frame->children.attributes, frame->count, &type, &frame->result));
      break;
    }
    case LOOM_TYPE_DIALECT:
    case LOOM_TYPE_FUNCTION:
    case LOOM_TYPE_REGISTER: {
      if (loom_type_kind(type) == LOOM_TYPE_DIALECT) {
        loom_string_id_t name;
        IREE_RETURN_IF_ERROR(loom_ir_remap_string_id(
            walk->remap, loom_type_dialect_name_id(type),
            /*allow_invalid=*/false, &name));
        type =
            loom_type_dialect(name, loom_type_dialect_param_count(type), NULL);
      }
      IREE_RETURN_IF_ERROR(loom_module_intern_topological_type_id(
          target, type, frame->children.types, frame->count, &frame->result));
      break;
    }
    default: {
      IREE_RETURN_IF_ERROR(loom_ir_remap_leaf_type(
          walk->remap, type, (uint16_t)frame->result, &type));
      IREE_RETURN_IF_ERROR(
          loom_module_intern_type_id(target, type, &frame->result));
      break;
    }
  }
  *(loom_type_id_t*)frame->destination = frame->result;
  loom_payload_pop(walk);
  return iree_ok_status();
}

static iree_status_t loom_payload_finish_attribute(loom_payload_walk_t* walk) {
  loom_payload_frame_t* frame = walk->top;
  loom_attribute_t result = frame->source.attribute;
  switch ((loom_attr_kind_t)result.kind) {
    case LOOM_ATTR_TYPE:
      result.type_id = frame->result;
      break;
    case LOOM_ATTR_ENCODING:
      result.encoding_id = frame->result;
      break;
    case LOOM_ATTR_DICT: {
      IREE_RETURN_IF_ERROR(loom_module_make_canonical_attr_dict(
          walk->remap->target_module,
          loom_make_named_attr_slice(frame->children.entries, frame->count),
          &result));
      break;
    }
    case LOOM_ATTR_PARAMETERIZED:
      result.parameterized_slots = frame->children.attributes;
      break;
    default:
      result.parameterized_array = frame->children.attributes;
      break;
  }
  *(loom_attribute_t*)frame->destination = result;
  loom_payload_pop(walk);
  return iree_ok_status();
}

static iree_status_t loom_payload_finish_encoding(loom_payload_walk_t* walk) {
  loom_payload_frame_t* frame = walk->top;
  const loom_encoding_t* source =
      loom_module_encoding(walk->remap->source_module, frame->source.encoding);
  loom_encoding_t target = {.attribute_count = source->attribute_count,
                            .attributes = frame->children.entries};
  IREE_RETURN_IF_ERROR(loom_ir_remap_string_id(walk->remap, source->name_id,
                                               false, &target.name_id));
  IREE_RETURN_IF_ERROR(loom_ir_remap_string_id(walk->remap, source->alias_id,
                                               true, &target.alias_id));
  uint16_t result;
  IREE_RETURN_IF_ERROR(
      loom_module_add_encoding(walk->remap->target_module, &target, &result));
  frame->result = result;
  *(uint32_t*)frame->destination = result;
  loom_payload_pop(walk);
  return iree_ok_status();
}

static iree_status_t loom_payload_run(loom_payload_walk_t* walk) {
  iree_status_t status = iree_ok_status();
  while (walk->top && iree_status_is_ok(status)) {
    loom_payload_frame_t* frame = walk->top;
    if (frame->next == frame->count) {
      switch (frame->kind) {
        case LOOM_PAYLOAD_TYPE:
          status = loom_payload_finish_type(walk);
          break;
        case LOOM_PAYLOAD_ATTRIBUTE:
          status = loom_payload_finish_attribute(walk);
          break;
        case LOOM_PAYLOAD_ENCODING:
          status = loom_payload_finish_encoding(walk);
          break;
      }
      continue;
    }
    const uint32_t index = frame->next++;
    if (frame->kind == LOOM_PAYLOAD_TYPE) {
      if (loom_type_kind(frame->source.type) == LOOM_TYPE_PARAMETERIZED) {
        status = loom_payload_request_attribute(
            walk, loom_type_parameterized_parameters(frame->source.type)[index],
            1, &walk->scratch, &frame->children.attributes[index]);
      } else if (frame->source_types) {
        status = loom_payload_request_type(walk, frame->source_types[index],
                                           &frame->children.types[index]);
      } else {
        status = loom_payload_request_encoding(
            walk, frame->source.type.encoding_id, &frame->result);
      }
      continue;
    }
    const loom_attribute_t source = frame->source.attribute;
    if (frame->kind == LOOM_PAYLOAD_ATTRIBUTE &&
        source.kind == LOOM_ATTR_TYPE) {
      if (source.type_id >= walk->remap->source_module->types.count) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "source type id %u out of range (source module has %" PRIhsz
            " types)",
            (unsigned)source.type_id, walk->remap->source_module->types.count);
      } else {
        status = loom_payload_request_type(
            walk,
            loom_type_table_get(&walk->remap->source_module->types,
                                source.type_id),
            &frame->result);
      }
    } else if (frame->kind == LOOM_PAYLOAD_ATTRIBUTE &&
               source.kind == LOOM_ATTR_ENCODING) {
      if (source.encoding_id > UINT16_MAX) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "source encoding id %" PRIu32
                                  " exceeds uint16_t range",
                                  source.encoding_id);
      } else {
        status = loom_payload_request_encoding(
            walk, (uint16_t)source.encoding_id, &frame->result);
      }
    } else if (frame->kind == LOOM_PAYLOAD_ENCODING ||
               source.kind == LOOM_ATTR_DICT) {
      const loom_named_attr_t* entries =
          frame->kind == LOOM_PAYLOAD_ENCODING
              ? loom_module_encoding(walk->remap->source_module,
                                     frame->source.encoding)
                    ->attributes
              : source.dict_entries;
      loom_named_attr_t* destination = &frame->children.entries[index];
      destination->reserved = 0;
      status = loom_ir_remap_string_id(walk->remap, entries[index].name_id,
                                       false, &destination->name_id);
      if (iree_status_is_ok(status)) {
        status = loom_payload_request_attribute(
            walk, entries[index].value,
            frame->kind == LOOM_PAYLOAD_ENCODING ? 0
                                                 : frame->aggregate_depth + 1,
            &walk->scratch, &destination->value);
      }
    } else {
      const loom_attribute_t* attributes =
          source.kind == LOOM_ATTR_PARAMETERIZED ? source.parameterized_slots
                                                 : source.parameterized_array;
      status = loom_payload_request_attribute(
          walk, attributes[index], frame->aggregate_depth + 1,
          frame->payload_arena, &frame->children.attributes[index]);
    }
  }
  return status;
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t loom_ir_remap_compound_type(
    loom_ir_remap_t* remap, loom_type_t source, loom_type_t* out_target) {
  loom_payload_walk_t walk = {.remap = remap};
  iree_arena_initialize(remap->arena->block_pool, &walk.scratch);
  loom_type_id_t result = LOOM_TYPE_ID_INVALID;
  iree_status_t status = loom_payload_request_type(&walk, source, &result);
  if (iree_status_is_ok(status)) {
    status = loom_payload_run(&walk);
  }
  if (iree_status_is_ok(status)) {
    *out_target = loom_type_table_get(&remap->target_module->types, result);
  }
  iree_arena_deinitialize(&walk.scratch);
  return status;
}

iree_status_t loom_ir_remap_type(loom_ir_remap_t* remap,
                                 loom_type_t source_type,
                                 loom_type_t* out_target_type) {
  *out_target_type = source_type;
  if (!loom_type_kind_is_valid(loom_type_kind(source_type))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown type kind %u",
                            (unsigned)loom_type_kind(source_type));
  }
  if (!loom_payload_type_requires_walk(source_type)) {
    return loom_ir_remap_leaf_type(remap, source_type, source_type.encoding_id,
                                   out_target_type);
  }
  return loom_ir_remap_compound_type(remap, source_type, out_target_type);
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t loom_ir_remap_compound_attribute(
    loom_ir_remap_t* remap, loom_attribute_t source,
    loom_attribute_t* out_target) {
  loom_payload_walk_t walk = {.remap = remap};
  iree_arena_initialize(remap->arena->block_pool, &walk.scratch);
  loom_attribute_t result = source;
  iree_status_t status = loom_payload_request_attribute(
      &walk, source, 0, &remap->target_module->arena, &result);
  if (iree_status_is_ok(status)) {
    status = loom_payload_run(&walk);
  }
  if (iree_status_is_ok(status)) {
    *out_target = result;
  }
  iree_arena_deinitialize(&walk.scratch);
  return status;
}

iree_status_t loom_ir_remap_attribute(loom_ir_remap_t* remap,
                                      loom_attribute_t source_attr,
                                      loom_attribute_t* out_target_attr) {
  *out_target_attr = source_attr;
  switch ((loom_attr_kind_t)source_attr.kind) {
    case LOOM_ATTR_TYPE:
    case LOOM_ATTR_ENCODING:
    case LOOM_ATTR_DICT:
    case LOOM_ATTR_PARAMETERIZED:
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      return loom_ir_remap_compound_attribute(remap, source_attr,
                                              out_target_attr);
    default:
      return loom_ir_remap_leaf_attribute(remap, source_attr,
                                          loom_ir_remap_is_initialized(remap)
                                              ? &remap->target_module->arena
                                              : NULL,
                                          out_target_attr);
  }
}

iree_status_t loom_ir_remap_encoding_id(loom_ir_remap_t* remap,
                                        uint16_t source_encoding_id,
                                        uint16_t* out_target_encoding_id) {
  *out_target_encoding_id = 0;
  loom_payload_walk_t walk = {.remap = remap};
  iree_arena_initialize(remap->arena->block_pool, &walk.scratch);
  uint32_t result = 0;
  iree_status_t status =
      loom_payload_request_encoding(&walk, source_encoding_id, &result);
  if (iree_status_is_ok(status)) {
    status = loom_payload_run(&walk);
  }
  if (iree_status_is_ok(status)) {
    *out_target_encoding_id = (uint16_t)result;
  }
  iree_arena_deinitialize(&walk.scratch);
  return status;
}
