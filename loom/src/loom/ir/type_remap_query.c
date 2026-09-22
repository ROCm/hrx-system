// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/type_remap_query.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/attribute.h"
#include "loom/ir/parameterized_type.h"
#include "loom/ir/type_identity.h"

//===----------------------------------------------------------------------===//
// Value remapping and inline types
//===----------------------------------------------------------------------===//

loom_value_id_t loom_type_value_remap_apply(
    const loom_module_t* module, const loom_type_value_remap_t* remap,
    loom_value_id_t value_id) {
  for (const loom_type_value_remap_t* span = remap; span; span = span->next) {
    if (iree_any_bit_set(span->flags,
                         LOOM_TYPE_VALUE_REMAP_FLAG_SOURCE_DEFINITION_SLICE)) {
      if (span->count == 0 || value_id >= module->values.count) {
        continue;
      }
      IREE_ASSERT(span->source_values[0] < module->values.count);
      const loom_value_t* first_value =
          loom_module_value(module, span->source_values[0]);
      const loom_value_t* value = loom_module_value(module, value_id);
      if (loom_value_is_block_arg(first_value) !=
          loom_value_is_block_arg(value)) {
        continue;
      }
      uint16_t first_index = loom_value_def_index(first_value);
      uint16_t value_index = loom_value_def_index(value);
      if (loom_value_is_block_arg(first_value)) {
        if (loom_value_def_block(first_value) != loom_value_def_block(value)) {
          continue;
        }
      } else {
        const loom_op_t* owner_op = loom_value_def_op(first_value);
        if (owner_op) {
          if (owner_op != loom_value_def_op(value)) {
            continue;
          }
        } else {
          // Declaration operands and results have independent index domains.
          // The declaration's ordinary operand use owns its argument index.
          IREE_ASSERT_EQ(first_value->use_count, 1);
          if (loom_value_def_op(value) || value->use_count != 1) {
            continue;
          }
          const loom_use_t first_use = loom_value_uses(first_value)[0];
          const loom_use_t value_use = loom_value_uses(value)[0];
          if (loom_use_user_op(first_use) != loom_use_user_op(value_use)) {
            continue;
          }
          first_index = loom_use_operand_index(first_use);
          value_index = loom_use_operand_index(value_use);
        }
      }
      if (value_index < first_index) {
        continue;
      }
      const uint16_t span_index = (uint16_t)(value_index - first_index);
      if (span_index >= span->count) {
        continue;
      }
      IREE_ASSERT(span->source_values[span_index] == value_id);
      return span->target_values[span_index];
    }
    for (uint16_t i = 0; i < span->count; ++i) {
      if (span->source_values[i] == value_id) {
        return span->target_values[i];
      }
    }
  }
  return value_id;
}

static iree_status_t loom_type_remap_query_allocate(
    loom_type_remap_query_t* query, iree_host_size_t size, void** out_ptr) {
  if (!query->scratch_initialized) {
    iree_arena_initialize(query->module->arena.block_pool,
                          &query->scratch_arena);
    query->scratch_initialized = true;
  }
  return iree_arena_allocate(&query->scratch_arena, size, out_ptr);
}

//===----------------------------------------------------------------------===//
// Completed function proofs
//===----------------------------------------------------------------------===//

typedef struct loom_type_remap_proof_record_t {
  // Critical-bit child edges, with leaves tagged in the low bit.
  uintptr_t children[2];
  // Union-find parent for completed semantic equivalence classes.
  struct loom_type_remap_proof_record_t* parent;
  // Exact module-local canonical type identity.
  loom_type_id_t type_id;
  // Union-find rank.
  uint8_t rank;
  // Most significant differing type ID bit for a branch record.
  uint8_t bit;
} loom_type_remap_proof_record_t;

static_assert(sizeof(loom_type_remap_proof_record_t) == 32,
              "proof records must remain one half cache line");

typedef struct loom_type_remap_proof_frame_t {
  // Enclosing continuation or next reusable frame.
  struct loom_type_remap_proof_frame_t* parent;
  // Source compound whose children are being compared.
  loom_type_remap_proof_record_t* source;
  // Target compound whose children are being compared.
  loom_type_remap_proof_record_t* target;
  // Next aligned function child to compare.
  uint32_t next_child;
} loom_type_remap_proof_frame_t;

struct loom_type_remap_proof_state_t {
  // Critical-bit roots independently indexing source and target views.
  uintptr_t views[2];
  // First source/target records without another arena allocation.
  loom_type_remap_proof_record_t inline_records[2];
  // Number of initialized inline records.
  uint8_t inline_record_count;
  // First continuation without another arena allocation.
  loom_type_remap_proof_frame_t inline_frame;
  // Reusable continuation frames.
  loom_type_remap_proof_frame_t* free_frames;
};

static loom_type_remap_proof_record_t* loom_type_remap_proof_record(
    uintptr_t edge) {
  return (loom_type_remap_proof_record_t*)(edge & ~(uintptr_t)1);
}

static iree_status_t loom_type_remap_proof_find_or_create(
    loom_type_remap_query_t* query, uint32_t view, loom_type_t type,
    loom_type_remap_proof_record_t** out_record) {
  loom_type_remap_proof_state_t* state = query->proof_state;
  const loom_type_id_t type_id = loom_type_identity_find(query->module, type);
  IREE_ASSERT(type_id != LOOM_TYPE_ID_INVALID,
              "compound boundary types must be canonical module types");
  uintptr_t existing = state->views[view];
  while (existing && !(existing & 1)) {
    loom_type_remap_proof_record_t* branch =
        loom_type_remap_proof_record(existing);
    existing = branch->children[(type_id >> branch->bit) & 1];
  }
  loom_type_remap_proof_record_t* prior =
      existing ? loom_type_remap_proof_record(existing) : NULL;
  if (prior && prior->type_id == type_id) {
    *out_record = prior;
    return iree_ok_status();
  }

  loom_type_remap_proof_record_t* record = NULL;
  if (state->inline_record_count < IREE_ARRAYSIZE(state->inline_records)) {
    record = &state->inline_records[state->inline_record_count++];
  } else {
    IREE_RETURN_IF_ERROR(loom_type_remap_query_allocate(query, sizeof(*record),
                                                        (void**)&record));
  }
  *record = (loom_type_remap_proof_record_t){
      .parent = record,
      .type_id = type_id,
  };
  const uintptr_t leaf = (uintptr_t)record | 1;
  if (!state->views[view]) {
    state->views[view] = leaf;
    *out_record = record;
    return iree_ok_status();
  }

  const uint32_t difference = type_id ^ prior->type_id;
  IREE_ASSERT(difference != 0);
  record->bit = (uint8_t)(31 - iree_math_count_leading_zeros_u32(difference));
  uintptr_t* position = &state->views[view];
  while (!(*position & 1) &&
         loom_type_remap_proof_record(*position)->bit > record->bit) {
    loom_type_remap_proof_record_t* branch =
        loom_type_remap_proof_record(*position);
    position = &branch->children[(type_id >> branch->bit) & 1];
  }
  const uint32_t side = (type_id >> record->bit) & 1;
  record->children[side] = leaf;
  record->children[side ^ 1] = *position;
  *position = (uintptr_t)record;
  *out_record = record;
  return iree_ok_status();
}

static loom_type_remap_proof_record_t* loom_type_remap_proof_root(
    loom_type_remap_proof_record_t* record) {
  while (record->parent != record) {
    record->parent = record->parent->parent;
    record = record->parent;
  }
  return record;
}

static void loom_type_remap_proof_join(loom_type_remap_proof_record_t* source,
                                       loom_type_remap_proof_record_t* target) {
  source = loom_type_remap_proof_root(source);
  target = loom_type_remap_proof_root(target);
  if (source == target) {
    return;
  }
  if (source->rank < target->rank) {
    loom_type_remap_proof_record_t* temporary = source;
    source = target;
    target = temporary;
  }
  target->parent = source;
  if (source->rank == target->rank) {
    ++source->rank;
  }
}

static iree_status_t loom_type_remap_proof_push(
    loom_type_remap_query_t* query, loom_type_remap_proof_frame_t* parent,
    loom_type_remap_proof_record_t* source,
    loom_type_remap_proof_record_t* target,
    loom_type_remap_proof_frame_t** out_frame) {
  loom_type_remap_proof_state_t* state = query->proof_state;
  loom_type_remap_proof_frame_t* frame = state->free_frames;
  if (frame) {
    state->free_frames = frame->parent;
  } else {
    IREE_RETURN_IF_ERROR(
        loom_type_remap_query_allocate(query, sizeof(*frame), (void**)&frame));
  }
  *frame = (loom_type_remap_proof_frame_t){
      .parent = parent,
      .source = source,
      .target = target,
      .next_child = 1,
  };
  *out_frame = frame;
  return iree_ok_status();
}

static void loom_type_remap_proof_recycle(
    loom_type_remap_proof_state_t* state,
    loom_type_remap_proof_frame_t* frame) {
  frame->parent = state->free_frames;
  state->free_frames = frame;
}

typedef enum loom_type_remap_proof_result_e {
  LOOM_TYPE_REMAP_PROOF_EQUAL = 0,
  LOOM_TYPE_REMAP_PROOF_UNEQUAL = 1,
  LOOM_TYPE_REMAP_PROOF_UNSUPPORTED = 2,
} loom_type_remap_proof_result_t;

static iree_status_t loom_type_remap_query_prove_function(
    loom_type_remap_query_t* query, loom_type_t source, loom_type_t target,
    loom_type_remap_proof_result_t* out_result) {
  loom_type_remap_proof_state_t* state = query->proof_state;
  loom_type_remap_proof_frame_t* frame = NULL;
  loom_type_remap_proof_result_t result = LOOM_TYPE_REMAP_PROOF_EQUAL;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && result == LOOM_TYPE_REMAP_PROOF_EQUAL) {
    const loom_type_kind_t source_kind = loom_type_kind(source);
    if (source_kind != loom_type_kind(target)) {
      result = LOOM_TYPE_REMAP_PROOF_UNEQUAL;
    } else if (source_kind != LOOM_TYPE_FUNCTION) {
      bool equal = false;
      if (!loom_type_remap_query_try_inline(query, source, target, &equal)) {
        result = LOOM_TYPE_REMAP_PROOF_UNSUPPORTED;
      } else if (!equal) {
        result = LOOM_TYPE_REMAP_PROOF_UNEQUAL;
      }
    } else {
      const loom_func_type_data_t* source_data = loom_type_func_data(source);
      const loom_func_type_data_t* target_data = loom_type_func_data(target);
      IREE_ASSERT(source_data && target_data);
      if (source.header != target.header ||
          source.encoding_id != target.encoding_id ||
          source.encoding_flags != target.encoding_flags ||
          source_data->arg_count != target_data->arg_count ||
          source_data->result_count != target_data->result_count) {
        result = LOOM_TYPE_REMAP_PROOF_UNEQUAL;
      } else {
        loom_type_remap_proof_record_t* source_record = NULL;
        loom_type_remap_proof_record_t* target_record = NULL;
        status = loom_type_remap_proof_find_or_create(query, 0, source,
                                                      &source_record);
        if (iree_status_is_ok(status)) {
          status = loom_type_remap_proof_find_or_create(query, 1, target,
                                                        &target_record);
        }
        if (!iree_status_is_ok(status)) {
          break;
        }
        if (loom_type_remap_proof_root(source_record) !=
            loom_type_remap_proof_root(target_record)) {
          const uint32_t child_count =
              (uint32_t)source_data->arg_count + source_data->result_count;
          if (!child_count) {
            loom_type_remap_proof_join(source_record, target_record);
          } else {
            status = loom_type_remap_proof_push(query, frame, source_record,
                                                target_record, &frame);
            if (!iree_status_is_ok(status)) {
              break;
            }
            source = source_data->types[0];
            target = target_data->types[0];
            continue;
          }
        }
      }
    }

    while (result == LOOM_TYPE_REMAP_PROOF_EQUAL && frame) {
      const loom_func_type_data_t* source_data = loom_type_func_data(
          loom_type_table_get(&query->module->types, frame->source->type_id));
      const loom_func_type_data_t* target_data = loom_type_func_data(
          loom_type_table_get(&query->module->types, frame->target->type_id));
      const uint32_t child_count =
          (uint32_t)source_data->arg_count + source_data->result_count;
      if (frame->next_child < child_count) {
        source = source_data->types[frame->next_child];
        target = target_data->types[frame->next_child];
        ++frame->next_child;
        break;
      }
      loom_type_remap_proof_join(frame->source, frame->target);
      loom_type_remap_proof_frame_t* completed = frame;
      frame = frame->parent;
      loom_type_remap_proof_recycle(state, completed);
    }
    if (!frame) {
      break;
    }
  }
  while (frame) {
    loom_type_remap_proof_frame_t* abandoned = frame;
    frame = frame->parent;
    loom_type_remap_proof_recycle(state, abandoned);
  }
  if (iree_status_is_ok(status)) {
    *out_result = result;
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Constructor-complete semantic cursor
//===----------------------------------------------------------------------===//

typedef enum loom_type_remap_node_domain_e {
  LOOM_TYPE_REMAP_NODE_TYPE = 1,
  LOOM_TYPE_REMAP_NODE_ATTRIBUTE = 2,
  LOOM_TYPE_REMAP_NODE_BLOB_BASE = 16,
} loom_type_remap_node_domain_t;

typedef struct loom_type_remap_node_t {
  // Semantic node domain.
  uint32_t domain;
  // Domain-specific immutable payload.
  union {
    // By-value canonical or inline type.
    loom_type_t type;
    // Module-owned attribute slot.
    const loom_attribute_t* attribute;
    // Pointer-backed scalar payload.
    struct {
      // First payload byte or element.
      const void* data;
      // Number of logical payload elements or bytes.
      uint32_t count;
    } blob;
  } value;
} loom_type_remap_node_t;

typedef enum loom_type_remap_component_kind_e {
  LOOM_TYPE_REMAP_COMPONENT_END = 0,
  LOOM_TYPE_REMAP_COMPONENT_WORD = 1,
  LOOM_TYPE_REMAP_COMPONENT_VALUE = 2,
  LOOM_TYPE_REMAP_COMPONENT_CHILD = 3,
} loom_type_remap_component_kind_t;

typedef struct loom_type_remap_component_t {
  // Component interpretation.
  loom_type_remap_component_kind_t kind;
  // Exact word or SSA value ID for immediate components.
  uint64_t word;
  // Nested semantic node for child components.
  loom_type_remap_node_t child;
} loom_type_remap_component_t;

typedef struct loom_type_remap_cursor_t {
  // Node whose immediate semantic sequence is being enumerated.
  loom_type_remap_node_t node;
  // Next immediate field or structural slot.
  uint32_t index;
  // Element within a variable-length payload.
  uint32_t element_index;
} loom_type_remap_cursor_t;

static loom_type_remap_node_t loom_type_remap_node_type(loom_type_t type) {
  return (loom_type_remap_node_t){
      .domain = LOOM_TYPE_REMAP_NODE_TYPE,
      .value.type = type,
  };
}

static loom_type_remap_node_t loom_type_remap_node_attribute(
    const loom_attribute_t* attribute) {
  return (loom_type_remap_node_t){
      .domain = LOOM_TYPE_REMAP_NODE_ATTRIBUTE,
      .value.attribute = attribute,
  };
}

static loom_type_remap_node_t loom_type_remap_node_blob(loom_attr_kind_t kind,
                                                        const void* data,
                                                        uint32_t count) {
  return (loom_type_remap_node_t){
      .domain = LOOM_TYPE_REMAP_NODE_BLOB_BASE + kind,
      .value.blob = {.data = data, .count = count},
  };
}

static loom_type_remap_component_t loom_type_remap_component_word(
    uint64_t word) {
  return (loom_type_remap_component_t){
      .kind = LOOM_TYPE_REMAP_COMPONENT_WORD,
      .word = word,
  };
}

static loom_type_remap_component_t loom_type_remap_component_value(
    loom_value_id_t value_id) {
  return (loom_type_remap_component_t){
      .kind = LOOM_TYPE_REMAP_COMPONENT_VALUE,
      .word = value_id,
  };
}

static loom_type_remap_component_t loom_type_remap_component_child(
    loom_type_remap_node_t child) {
  return (loom_type_remap_component_t){
      .kind = LOOM_TYPE_REMAP_COMPONENT_CHILD,
      .child = child,
  };
}

static uint64_t loom_type_remap_read_bytes(const uint8_t* data,
                                           uint32_t byte_count,
                                           uint32_t word_index) {
  const uint32_t offset = word_index * 8;
  const uint32_t remaining = byte_count - offset;
  uint64_t word = 0;
  memcpy(&word, data + offset, iree_min(remaining, 8u));
  return word;
}

static loom_type_remap_component_t loom_type_remap_cursor_next_type(
    loom_type_remap_cursor_t* cursor) {
  const loom_type_t type = cursor->node.value.type;
  const loom_type_kind_t kind = loom_type_kind(type);
  IREE_ASSERT(loom_type_kind_is_valid(kind));
  switch (kind) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      IREE_ASSERT(data);
      const uint32_t child_count =
          (uint32_t)data->arg_count + data->result_count;
      switch (cursor->index++) {
        case 0:
          return loom_type_remap_component_word(type.header);
        case 1:
          return loom_type_remap_component_word(type.encoding_id);
        case 2:
          return loom_type_remap_component_word(type.encoding_flags);
        case 3:
          return loom_type_remap_component_word(data->arg_count);
        case 4:
          return loom_type_remap_component_word(data->result_count);
        default: {
          const uint32_t child_index = cursor->index - 6;
          return child_index < child_count
                     ? loom_type_remap_component_child(
                           loom_type_remap_node_type(data->types[child_index]))
                     : (loom_type_remap_component_t){0};
        }
      }
    }
    case LOOM_TYPE_DIALECT: {
      const uint32_t child_count = loom_type_dialect_param_count(type);
      switch (cursor->index++) {
        case 0:
          return loom_type_remap_component_word(type.header);
        case 1:
          return loom_type_remap_component_word(type.encoding_id);
        case 2:
          return loom_type_remap_component_word(
              loom_type_dialect_name_id(type));
        case 3:
          return loom_type_remap_component_word(child_count);
        default: {
          const uint32_t child_index = cursor->index - 5;
          return child_index < child_count
                     ? loom_type_remap_component_child(
                           loom_type_remap_node_type(
                               loom_type_dialect_params(type)[child_index]))
                     : (loom_type_remap_component_t){0};
        }
      }
    }
    case LOOM_TYPE_PARAMETERIZED: {
      const uint32_t child_count =
          loom_type_parameterized_parameter_count(type);
      switch (cursor->index++) {
        case 0:
          return loom_type_remap_component_word(type.header);
        case 1:
          return loom_type_remap_component_word(type.encoding_flags);
        case 2:
          return loom_type_remap_component_word(
              (uintptr_t)loom_type_parameterized_descriptor(type));
        case 3:
          return loom_type_remap_component_word(child_count);
        default: {
          const uint32_t child_index = cursor->index - 5;
          return child_index < child_count
                     ? loom_type_remap_component_child(
                           loom_type_remap_node_attribute(
                               &loom_type_parameterized_parameters(
                                   type)[child_index]))
                     : (loom_type_remap_component_t){0};
        }
      }
    }
    case LOOM_TYPE_REGISTER: {
      const bool typed = loom_type_register_has_value_type(type);
      switch (cursor->index++) {
        case 0:
          return loom_type_remap_component_word(type.header);
        case 1:
          return loom_type_remap_component_word(type.encoding_id);
        case 2:
          return loom_type_remap_component_word(type.encoding_flags);
        case 3:
          return loom_type_remap_component_word(
              typed ? loom_type_register_payload0(type) : type.dims[0]);
        case 4:
          return loom_type_remap_component_word(
              typed ? loom_type_register_payload1(type) : type.dims[1]);
        case 5:
          return typed ? loom_type_remap_component_child(
                             loom_type_remap_node_type(
                                 *loom_type_register_value_type(type)))
                       : (loom_type_remap_component_t){0};
        default:
          return (loom_type_remap_component_t){0};
      }
    }
    default:
      break;
  }

  if (loom_type_is_shaped(type) || loom_type_is_pool(type)) {
    const uint32_t rank = loom_type_rank(type);
    switch (cursor->index++) {
      case 0:
        return loom_type_remap_component_word(type.header);
      case 1:
        return loom_type_remap_component_word(type.encoding_flags);
      case 2:
        return loom_type_has_ssa_encoding(type)
                   ? loom_type_remap_component_value(
                         loom_type_encoding_value_id(type))
                   : loom_type_remap_component_word(type.encoding_id);
      default: {
        const uint32_t dimension_index = cursor->index - 4;
        if (dimension_index >= rank) {
          return (loom_type_remap_component_t){0};
        }
        const uint64_t dimension = loom_type_dim(type, dimension_index);
        return loom_dim_is_dynamic(dimension)
                   ? loom_type_remap_component_value(
                         loom_dim_value_id(dimension))
                   : loom_type_remap_component_word(dimension);
      }
    }
  }

  switch (cursor->index++) {
    case 0:
      return loom_type_remap_component_word(type.header);
    case 1:
      return loom_type_remap_component_word(type.encoding_id);
    case 2:
      return loom_type_remap_component_word(type.encoding_flags);
    case 3:
      return loom_type_remap_component_word(type.dims[0]);
    case 4:
      return loom_type_remap_component_word(type.dims[1]);
    default:
      return (loom_type_remap_component_t){0};
  }
}

static loom_type_remap_component_t loom_type_remap_cursor_next_attribute(
    const loom_module_t* module, loom_type_remap_cursor_t* cursor) {
  const loom_attribute_t* attribute = cursor->node.value.attribute;
  const loom_attr_kind_t kind = (loom_attr_kind_t)attribute->kind;
  switch (kind) {
    case LOOM_ATTR_TYPE:
      switch (cursor->index++) {
        case 0:
          return loom_type_remap_component_word(kind);
        case 1:
          IREE_ASSERT(attribute->type_id < module->types.count);
          return loom_type_remap_component_child(loom_type_remap_node_type(
              loom_type_table_get(&module->types, attribute->type_id)));
        default:
          return (loom_type_remap_component_t){0};
      }
    case LOOM_ATTR_PREDICATE_LIST:
    case LOOM_ATTR_I64_ARRAY:
    case LOOM_ATTR_ENUM_ARRAY:
    case LOOM_ATTR_SIGNED_ENUM_SET:
    case LOOM_ATTR_SYMBOL_ARRAY:
    case LOOM_ATTR_SYMBOL_SET: {
      const void* data = NULL;
      switch (kind) {
        case LOOM_ATTR_PREDICATE_LIST:
          data = attribute->predicate_list;
          break;
        case LOOM_ATTR_I64_ARRAY:
          data = attribute->i64_array;
          break;
        case LOOM_ATTR_ENUM_ARRAY:
          data = attribute->enum_array;
          break;
        case LOOM_ATTR_SIGNED_ENUM_SET:
          data = attribute->signed_enum_set_words;
          break;
        default:
          data = attribute->symbol_refs;
          break;
      }
      switch (cursor->index++) {
        case 0:
          return loom_type_remap_component_word(kind);
        case 1:
          return loom_type_remap_component_word(attribute->count);
        case 2:
          return attribute->count ? loom_type_remap_component_child(
                                        loom_type_remap_node_blob(
                                            kind, data, attribute->count))
                                  : (loom_type_remap_component_t){0};
        default:
          return (loom_type_remap_component_t){0};
      }
    }
    case LOOM_ATTR_BYTES:
      switch (cursor->index++) {
        case 0:
          return loom_type_remap_component_word(kind);
        case 1:
          return loom_type_remap_component_word(attribute->reserved_1);
        case 2:
          return attribute->reserved_1
                     ? loom_type_remap_component_child(
                           loom_type_remap_node_blob(kind, attribute->bytes,
                                                     attribute->reserved_1))
                     : (loom_type_remap_component_t){0};
        default:
          return (loom_type_remap_component_t){0};
      }
    case LOOM_ATTR_DICT:
      if (cursor->index == 0) {
        ++cursor->index;
        return loom_type_remap_component_word(kind);
      }
      if (cursor->index == 1) {
        ++cursor->index;
        return loom_type_remap_component_word(attribute->count);
      }
      if (cursor->element_index >= attribute->count) {
        return (loom_type_remap_component_t){0};
      }
      if ((cursor->index++ & 1) == 0) {
        return loom_type_remap_component_word(
            attribute->dict_entries[cursor->element_index].name_id);
      }
      return loom_type_remap_component_child(loom_type_remap_node_attribute(
          &attribute->dict_entries[cursor->element_index++].value));
    case LOOM_ATTR_PARAMETERIZED:
      if (cursor->index == 0) {
        ++cursor->index;
        return loom_type_remap_component_word(kind);
      }
      if (cursor->index == 1) {
        ++cursor->index;
        return loom_type_remap_component_word(attribute->reserved_1);
      }
      if (cursor->index == 2) {
        ++cursor->index;
        return loom_type_remap_component_word(attribute->count);
      }
      return cursor->element_index < attribute->count
                 ? loom_type_remap_component_child(
                       loom_type_remap_node_attribute(
                           &attribute
                                ->parameterized_slots[cursor->element_index++]))
                 : (loom_type_remap_component_t){0};
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      if (cursor->index == 0) {
        ++cursor->index;
        return loom_type_remap_component_word(kind);
      }
      if (cursor->index == 1) {
        ++cursor->index;
        return loom_type_remap_component_word(attribute->count);
      }
      return cursor->element_index < attribute->count
                 ? loom_type_remap_component_child(
                       loom_type_remap_node_attribute(
                           &attribute
                                ->parameterized_array[cursor->element_index++]))
                 : (loom_type_remap_component_t){0};
    default: {
      uint64_t words[2];
      static_assert(sizeof(words) == sizeof(*attribute),
                    "generic attributes must fit two words");
      memcpy(words, attribute, sizeof(words));
      return cursor->index < IREE_ARRAYSIZE(words)
                 ? loom_type_remap_component_word(words[cursor->index++])
                 : (loom_type_remap_component_t){0};
    }
  }
}

static loom_type_remap_component_t loom_type_remap_cursor_next_blob(
    loom_type_remap_cursor_t* cursor) {
  const loom_attr_kind_t kind =
      (loom_attr_kind_t)(cursor->node.domain - LOOM_TYPE_REMAP_NODE_BLOB_BASE);
  if (cursor->index == 0) {
    ++cursor->index;
    return loom_type_remap_component_word(kind);
  }
  if (cursor->index == 1) {
    ++cursor->index;
    return loom_type_remap_component_word(cursor->node.value.blob.count);
  }
  if (kind == LOOM_ATTR_PREDICATE_LIST) {
    while (cursor->element_index < cursor->node.value.blob.count) {
      const loom_predicate_t* predicate =
          &((const loom_predicate_t*)
                cursor->node.value.blob.data)[cursor->element_index];
      const uint32_t field = cursor->index++ - 2;
      if (field == 0) {
        return loom_type_remap_component_word(predicate->kind);
      }
      if (field == 1) {
        return loom_type_remap_component_word(predicate->arg_count);
      }
      if (field == 2) {
        uint32_t tags = 0;
        memcpy(&tags, predicate->arg_tags, sizeof(predicate->arg_tags));
        return loom_type_remap_component_word(tags);
      }
      const uint32_t argument_index = field - 3;
      if (argument_index < predicate->arg_count) {
        const int64_t argument = predicate->args[argument_index];
        return predicate->arg_tags[argument_index] == LOOM_PRED_ARG_VALUE
                   ? loom_type_remap_component_value((loom_value_id_t)argument)
                   : loom_type_remap_component_word((uint64_t)argument);
      }
      ++cursor->element_index;
      cursor->index = 2;
    }
    return (loom_type_remap_component_t){0};
  }

  uint32_t byte_count = 0;
  switch (kind) {
    case LOOM_ATTR_I64_ARRAY:
      byte_count = cursor->node.value.blob.count * sizeof(int64_t);
      break;
    case LOOM_ATTR_ENUM_ARRAY:
      byte_count = cursor->node.value.blob.count;
      break;
    case LOOM_ATTR_SIGNED_ENUM_SET:
      byte_count = cursor->node.value.blob.count * 2 * sizeof(uint64_t);
      break;
    case LOOM_ATTR_SYMBOL_ARRAY:
    case LOOM_ATTR_SYMBOL_SET:
      byte_count = cursor->node.value.blob.count * sizeof(loom_symbol_ref_t);
      break;
    case LOOM_ATTR_BYTES:
      byte_count = cursor->node.value.blob.count;
      break;
    default:
      IREE_ASSERT_UNREACHABLE("blob nodes have a pointer-backed attribute");
  }
  const uint32_t word_index = cursor->index++ - 2;
  return word_index * 8 < byte_count
             ? loom_type_remap_component_word(loom_type_remap_read_bytes(
                   (const uint8_t*)cursor->node.value.blob.data, byte_count,
                   word_index))
             : (loom_type_remap_component_t){0};
}

static loom_type_remap_component_t loom_type_remap_cursor_next(
    const loom_module_t* module, loom_type_remap_cursor_t* cursor) {
  if (cursor->node.domain == LOOM_TYPE_REMAP_NODE_TYPE) {
    return loom_type_remap_cursor_next_type(cursor);
  }
  if (cursor->node.domain == LOOM_TYPE_REMAP_NODE_ATTRIBUTE) {
    return loom_type_remap_cursor_next_attribute(module, cursor);
  }
  return loom_type_remap_cursor_next_blob(cursor);
}

//===----------------------------------------------------------------------===//
// Exact semantic normalization
//===----------------------------------------------------------------------===//

typedef struct loom_type_remap_node_key_t {
  // Canonical type ID plus one or immutable payload address.
  uint64_t address;
  // Node domain and payload cardinality disambiguator.
  uint64_t metadata;
} loom_type_remap_node_key_t;

typedef struct loom_type_remap_exact_record_t {
  // Critical-bit child edges, with leaves tagged in the low bit.
  uintptr_t children[2];
  // Exact physical node identity within one mapping view.
  loom_type_remap_node_key_t key;
  // Completed structural identity, or zero before normalization.
  uint32_t normalized_identity;
  // Most significant differing key bit for a branch record.
  uint8_t bit;
} loom_type_remap_exact_record_t;

static_assert(sizeof(loom_type_remap_exact_record_t) == 40,
              "exact records must remain compact");

typedef struct loom_type_remap_prefix_t {
  // Critical-bit child edges, with leaves tagged in the low bit.
  uintptr_t children[2];
  // Exact (prefix identity, next word) transition key.
  uint64_t key;
  // Canonical result identity for this transition.
  uint32_t identity;
  // Most significant differing key bit for a branch record.
  uint8_t bit;
} loom_type_remap_prefix_t;

static_assert(sizeof(loom_type_remap_prefix_t) == 32,
              "prefix transitions must remain one half cache line");

typedef struct loom_type_remap_normalize_frame_t {
  // Enclosing continuation or next reusable frame.
  struct loom_type_remap_normalize_frame_t* parent;
  // Suspended immediate semantic sequence.
  loom_type_remap_cursor_t cursor;
  // Pointer-backed node receiving the completed identity, or NULL.
  loom_type_remap_exact_record_t* record;
  // Structural identity accumulated before the pending child.
  uint32_t identity;
  // Mapping view used for VALUE components.
  uint8_t view;
} loom_type_remap_normalize_frame_t;

struct loom_type_remap_exact_state_t {
  // Critical-bit roots independently indexing source and target nodes.
  uintptr_t records[2];
  // Critical-bit root indexing exact prefix transitions.
  uintptr_t prefixes;
  // First source/target node records without another arena allocation.
  loom_type_remap_exact_record_t inline_records[2];
  // Number of initialized inline records.
  uint8_t inline_record_count;
  // First traversal continuation without another arena allocation.
  loom_type_remap_normalize_frame_t inline_frame;
  // Reusable traversal continuations.
  loom_type_remap_normalize_frame_t* free_frames;
  // Last assigned nonzero structural identity.
  uint32_t prefix_count;
};

static loom_type_remap_exact_record_t* loom_type_remap_exact_record(
    uintptr_t edge) {
  return (loom_type_remap_exact_record_t*)(edge & ~(uintptr_t)1);
}

static loom_type_remap_prefix_t* loom_type_remap_prefix(uintptr_t edge) {
  return (loom_type_remap_prefix_t*)(edge & ~(uintptr_t)1);
}

static bool loom_type_remap_node_key(const loom_type_remap_query_t* query,
                                     loom_type_remap_node_t node,
                                     loom_type_remap_node_key_t* out_key) {
  if (node.domain == LOOM_TYPE_REMAP_NODE_TYPE) {
    const loom_type_id_t type_id =
        loom_type_identity_find(query->module, node.value.type);
    if (type_id == LOOM_TYPE_ID_INVALID) {
      return false;
    }
    *out_key = (loom_type_remap_node_key_t){
        .address = (uint64_t)type_id + 1,
        .metadata = LOOM_TYPE_REMAP_NODE_TYPE,
    };
    return true;
  }
  if (node.domain == LOOM_TYPE_REMAP_NODE_ATTRIBUTE) {
    *out_key = (loom_type_remap_node_key_t){
        .address = (uintptr_t)node.value.attribute,
        .metadata = LOOM_TYPE_REMAP_NODE_ATTRIBUTE,
    };
    return true;
  }
  *out_key = (loom_type_remap_node_key_t){
      .address = (uintptr_t)node.value.blob.data,
      .metadata = ((uint64_t)node.domain << 32) | node.value.blob.count,
  };
  return true;
}

static bool loom_type_remap_node_key_equal(loom_type_remap_node_key_t left,
                                           loom_type_remap_node_key_t right) {
  return left.address == right.address && left.metadata == right.metadata;
}

static uint32_t loom_type_remap_node_key_side(loom_type_remap_node_key_t key,
                                              uint32_t bit) {
  const uint64_t word = bit < 64 ? key.address : key.metadata;
  return (uint32_t)((word >> (bit & 63)) & 1);
}

static uint32_t loom_type_remap_node_key_differing_bit(
    loom_type_remap_node_key_t left, loom_type_remap_node_key_t right) {
  uint64_t difference = left.metadata ^ right.metadata;
  if (difference) {
    return 64 + 63 - iree_math_count_leading_zeros_u64(difference);
  }
  difference = left.address ^ right.address;
  IREE_ASSERT(difference != 0);
  return 63 - iree_math_count_leading_zeros_u64(difference);
}

static iree_status_t loom_type_remap_exact_find_or_create(
    loom_type_remap_query_t* query, uint32_t view, loom_type_remap_node_t node,
    loom_type_remap_exact_record_t** out_record) {
  loom_type_remap_node_key_t key;
  if (!loom_type_remap_node_key(query, node, &key)) {
    *out_record = NULL;
    return iree_ok_status();
  }
  loom_type_remap_exact_state_t* state = query->exact_state;
  uintptr_t existing = state->records[view];
  while (existing && !(existing & 1)) {
    loom_type_remap_exact_record_t* branch =
        loom_type_remap_exact_record(existing);
    existing =
        branch->children[loom_type_remap_node_key_side(key, branch->bit)];
  }
  loom_type_remap_exact_record_t* prior =
      existing ? loom_type_remap_exact_record(existing) : NULL;
  if (prior && loom_type_remap_node_key_equal(prior->key, key)) {
    *out_record = prior;
    return iree_ok_status();
  }

  loom_type_remap_exact_record_t* record = NULL;
  if (state->inline_record_count < IREE_ARRAYSIZE(state->inline_records)) {
    record = &state->inline_records[state->inline_record_count++];
  } else {
    IREE_RETURN_IF_ERROR(loom_type_remap_query_allocate(query, sizeof(*record),
                                                        (void**)&record));
  }
  *record = (loom_type_remap_exact_record_t){.key = key};
  const uintptr_t leaf = (uintptr_t)record | 1;
  if (!state->records[view]) {
    state->records[view] = leaf;
    *out_record = record;
    return iree_ok_status();
  }

  record->bit =
      (uint8_t)loom_type_remap_node_key_differing_bit(key, prior->key);
  uintptr_t* position = &state->records[view];
  while (!(*position & 1) &&
         loom_type_remap_exact_record(*position)->bit > record->bit) {
    loom_type_remap_exact_record_t* branch =
        loom_type_remap_exact_record(*position);
    position =
        &branch->children[loom_type_remap_node_key_side(key, branch->bit)];
  }
  const uint32_t side = loom_type_remap_node_key_side(key, record->bit);
  record->children[side] = leaf;
  record->children[side ^ 1] = *position;
  *position = (uintptr_t)record;
  *out_record = record;
  return iree_ok_status();
}

static loom_type_remap_prefix_t* loom_type_remap_prefix_find(uintptr_t root,
                                                             uint64_t key) {
  uintptr_t edge = root;
  if (!edge) {
    return NULL;
  }
  while (!(edge & 1)) {
    loom_type_remap_prefix_t* branch = loom_type_remap_prefix(edge);
    edge = branch->children[(key >> branch->bit) & 1];
  }
  loom_type_remap_prefix_t* leaf = loom_type_remap_prefix(edge);
  return leaf->key == key ? leaf : NULL;
}

static iree_status_t loom_type_remap_prefix_insert(
    loom_type_remap_query_t* query, uint64_t key, uint32_t identity) {
  loom_type_remap_exact_state_t* state = query->exact_state;
  loom_type_remap_prefix_t* record = NULL;
  IREE_RETURN_IF_ERROR(
      loom_type_remap_query_allocate(query, sizeof(*record), (void**)&record));
  *record = (loom_type_remap_prefix_t){
      .key = key,
      .identity = identity,
  };
  const uintptr_t leaf = (uintptr_t)record | 1;
  if (!state->prefixes) {
    state->prefixes = leaf;
    return iree_ok_status();
  }

  uintptr_t existing = state->prefixes;
  while (!(existing & 1)) {
    loom_type_remap_prefix_t* branch = loom_type_remap_prefix(existing);
    existing = branch->children[(key >> branch->bit) & 1];
  }
  const uint64_t difference = key ^ loom_type_remap_prefix(existing)->key;
  IREE_ASSERT(difference != 0);
  record->bit = (uint8_t)(63 - iree_math_count_leading_zeros_u64(difference));
  uintptr_t* position = &state->prefixes;
  while (!(*position & 1) &&
         loom_type_remap_prefix(*position)->bit > record->bit) {
    loom_type_remap_prefix_t* branch = loom_type_remap_prefix(*position);
    position = &branch->children[(key >> branch->bit) & 1];
  }
  const uint32_t side = (uint32_t)((key >> record->bit) & 1);
  record->children[side] = leaf;
  record->children[side ^ 1] = *position;
  *position = (uintptr_t)record;
  return iree_ok_status();
}

static iree_status_t loom_type_remap_prefix_word(loom_type_remap_query_t* query,
                                                 uint32_t prefix, uint32_t word,
                                                 uint32_t* out_identity) {
  loom_type_remap_exact_state_t* state = query->exact_state;
  const uint64_t key = ((uint64_t)prefix << 32) | word;
  loom_type_remap_prefix_t* prior =
      loom_type_remap_prefix_find(state->prefixes, key);
  if (prior) {
    *out_identity = prior->identity;
    return iree_ok_status();
  }
  if (state->prefix_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "mapped type identity space exhausted");
  }
  const uint32_t identity = ++state->prefix_count;
  IREE_RETURN_IF_ERROR(loom_type_remap_prefix_insert(query, key, identity));
  *out_identity = identity;
  return iree_ok_status();
}

static iree_status_t loom_type_remap_prefix_component(
    loom_type_remap_query_t* query, uint32_t identity,
    loom_type_remap_component_t component, uint32_t view,
    uint32_t* out_identity) {
  IREE_RETURN_IF_ERROR(
      loom_type_remap_prefix_word(query, identity, component.kind, &identity));
  if (component.kind == LOOM_TYPE_REMAP_COMPONENT_WORD) {
    IREE_RETURN_IF_ERROR(loom_type_remap_prefix_word(
        query, identity, (uint32_t)component.word, &identity));
    IREE_RETURN_IF_ERROR(loom_type_remap_prefix_word(
        query, identity, (uint32_t)(component.word >> 32), &identity));
  } else if (component.kind == LOOM_TYPE_REMAP_COMPONENT_VALUE) {
    const loom_value_id_t value_id =
        view == 0 ? loom_type_value_remap_apply(query->module, query->remap,
                                                (loom_value_id_t)component.word)
                  : (loom_value_id_t)component.word;
    IREE_RETURN_IF_ERROR(
        loom_type_remap_prefix_word(query, identity, value_id, &identity));
  }
  *out_identity = identity;
  return iree_ok_status();
}

static iree_status_t loom_type_remap_normalize_push(
    loom_type_remap_query_t* query, loom_type_remap_normalize_frame_t* parent,
    loom_type_remap_cursor_t cursor, loom_type_remap_exact_record_t* record,
    uint32_t view, loom_type_remap_normalize_frame_t** out_frame) {
  loom_type_remap_exact_state_t* state = query->exact_state;
  loom_type_remap_normalize_frame_t* frame = state->free_frames;
  if (frame) {
    state->free_frames = frame->parent;
  } else {
    IREE_RETURN_IF_ERROR(
        loom_type_remap_query_allocate(query, sizeof(*frame), (void**)&frame));
  }
  *frame = (loom_type_remap_normalize_frame_t){
      .parent = parent,
      .cursor = cursor,
      .record = record,
      .view = (uint8_t)view,
  };
  *out_frame = frame;
  return iree_ok_status();
}

static void loom_type_remap_normalize_recycle(
    loom_type_remap_exact_state_t* state,
    loom_type_remap_normalize_frame_t* frame) {
  frame->parent = state->free_frames;
  state->free_frames = frame;
}

static iree_status_t loom_type_remap_normalize(loom_type_remap_query_t* query,
                                               loom_type_remap_node_t node,
                                               uint32_t view,
                                               uint32_t* out_identity) {
  loom_type_remap_exact_state_t* state = query->exact_state;
  loom_type_remap_normalize_frame_t* frame = NULL;
  uint32_t identity = 0;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    loom_type_remap_exact_record_t* record = NULL;
    status = loom_type_remap_exact_find_or_create(query, view, node, &record);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (record && record->normalized_identity) {
      identity = record->normalized_identity;
    } else {
      loom_type_remap_cursor_t cursor = {.node = node};
      loom_type_remap_component_t component =
          loom_type_remap_cursor_next(query->module, &cursor);
      uint32_t node_identity = 0;
      status = loom_type_remap_prefix_word(query, node_identity, node.domain,
                                           &node_identity);
      while (iree_status_is_ok(status) &&
             component.kind != LOOM_TYPE_REMAP_COMPONENT_CHILD) {
        status = loom_type_remap_prefix_component(
            query, node_identity, component, view, &node_identity);
        if (!iree_status_is_ok(status) ||
            component.kind == LOOM_TYPE_REMAP_COMPONENT_END) {
          break;
        }
        component = loom_type_remap_cursor_next(query->module, &cursor);
      }
      if (!iree_status_is_ok(status)) {
        break;
      }
      if (component.kind == LOOM_TYPE_REMAP_COMPONENT_CHILD) {
        status = loom_type_remap_normalize_push(query, frame, cursor, record,
                                                view, &frame);
        if (!iree_status_is_ok(status)) {
          break;
        }
        frame->identity = node_identity;
        node = component.child;
        continue;
      }
      identity = node_identity;
      if (record) {
        record->normalized_identity = identity;
      }
    }

    while (iree_status_is_ok(status) && frame) {
      status = loom_type_remap_prefix_word(query, frame->identity,
                                           LOOM_TYPE_REMAP_COMPONENT_CHILD,
                                           &frame->identity);
      if (iree_status_is_ok(status)) {
        status = loom_type_remap_prefix_word(query, frame->identity, identity,
                                             &frame->identity);
      }
      if (!iree_status_is_ok(status)) {
        break;
      }
      loom_type_remap_component_t component =
          loom_type_remap_cursor_next(query->module, &frame->cursor);
      while (component.kind != LOOM_TYPE_REMAP_COMPONENT_CHILD) {
        status = loom_type_remap_prefix_component(
            query, frame->identity, component, frame->view, &frame->identity);
        if (!iree_status_is_ok(status) ||
            component.kind == LOOM_TYPE_REMAP_COMPONENT_END) {
          break;
        }
        component = loom_type_remap_cursor_next(query->module, &frame->cursor);
      }
      if (!iree_status_is_ok(status)) {
        break;
      }
      if (component.kind == LOOM_TYPE_REMAP_COMPONENT_CHILD) {
        node = component.child;
        view = frame->view;
        break;
      }
      identity = frame->identity;
      if (frame->record) {
        frame->record->normalized_identity = identity;
      }
      loom_type_remap_normalize_frame_t* completed = frame;
      frame = frame->parent;
      loom_type_remap_normalize_recycle(state, completed);
    }
    if (!frame) {
      break;
    }
  }
  while (frame) {
    loom_type_remap_normalize_frame_t* abandoned = frame;
    frame = frame->parent;
    loom_type_remap_normalize_recycle(state, abandoned);
  }
  if (iree_status_is_ok(status)) {
    *out_identity = identity;
  }
  return status;
}

static iree_status_t loom_type_remap_query_exact_equal(
    loom_type_remap_query_t* query, loom_type_t source, loom_type_t target,
    bool* out_equal) {
  if (!query->exact_state) {
    loom_type_remap_exact_state_t* state = NULL;
    IREE_RETURN_IF_ERROR(
        loom_type_remap_query_allocate(query, sizeof(*state), (void**)&state));
    *state = (loom_type_remap_exact_state_t){
        .free_frames = &state->inline_frame,
    };
    query->exact_state = state;
  }
  uint32_t source_identity = 0;
  uint32_t target_identity = 0;
  IREE_RETURN_IF_ERROR(loom_type_remap_normalize(
      query, loom_type_remap_node_type(source), 0, &source_identity));
  IREE_RETURN_IF_ERROR(loom_type_remap_normalize(
      query, loom_type_remap_node_type(target), 1, &target_identity));
  *out_equal = source_identity == target_identity;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public query lifecycle
//===----------------------------------------------------------------------===//

void loom_type_remap_query_initialize(const loom_module_t* module,
                                      const loom_type_value_remap_t* remap,
                                      loom_type_remap_query_t* out_query) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(out_query);
  out_query->module = module;
  out_query->remap = remap;
  out_query->proof_state = NULL;
  out_query->exact_state = NULL;
  out_query->scratch_initialized = false;
  out_query->exact_mode = false;
}

void loom_type_remap_query_deinitialize(loom_type_remap_query_t* query) {
  if (!query) {
    return;
  }
  if (query->scratch_initialized) {
    iree_arena_deinitialize(&query->scratch_arena);
  }
}

IREE_ATTRIBUTE_NOINLINE iree_status_t loom_type_remap_query_equal_compound(
    loom_type_remap_query_t* query, loom_type_t source_type,
    loom_type_t target_type, bool* out_equal) {
  const loom_type_kind_t source_kind = loom_type_kind(source_type);
  if (query->exact_mode || source_kind != LOOM_TYPE_FUNCTION) {
    query->exact_mode = true;
    return loom_type_remap_query_exact_equal(query, source_type, target_type,
                                             out_equal);
  }

  if (!query->proof_state) {
    loom_type_remap_proof_state_t* state = NULL;
    IREE_RETURN_IF_ERROR(
        loom_type_remap_query_allocate(query, sizeof(*state), (void**)&state));
    *state = (loom_type_remap_proof_state_t){
        .free_frames = &state->inline_frame,
    };
    query->proof_state = state;
  }
  loom_type_remap_proof_result_t result = LOOM_TYPE_REMAP_PROOF_EQUAL;
  IREE_RETURN_IF_ERROR(loom_type_remap_query_prove_function(
      query, source_type, target_type, &result));
  if (result == LOOM_TYPE_REMAP_PROOF_UNSUPPORTED) {
    query->exact_mode = true;
    return loom_type_remap_query_exact_equal(query, source_type, target_type,
                                             out_equal);
  }
  *out_equal = result == LOOM_TYPE_REMAP_PROOF_EQUAL;
  query->exact_mode |= !*out_equal;
  return iree_ok_status();
}
