// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/kernel_specialization_key.h"

#include "loom/ir/module.h"
#include "loom/ir/structural_hash.h"
#include "loom/util/ordinal_map.h"
#include "loom/util/segmented_storage.h"

typedef enum loom_kernel_literal_task_kind_e {
  LOOM_KERNEL_LITERAL_WORD,
  LOOM_KERNEL_LITERAL_BYTES,
  LOOM_KERNEL_LITERAL_TYPE,
  LOOM_KERNEL_LITERAL_ATTRIBUTE,
  LOOM_KERNEL_LITERAL_ENCODING,
  LOOM_KERNEL_LITERAL_VALUES,
  LOOM_KERNEL_LITERAL_ATTRIBUTES,
  LOOM_KERNEL_LITERAL_NAMED_ATTRIBUTES,
  LOOM_KERNEL_LITERAL_TYPES,
  LOOM_KERNEL_LITERAL_STRINGS,
  LOOM_KERNEL_LITERAL_SYMBOLS,
  LOOM_KERNEL_LITERAL_PREDICATES,
} loom_kernel_literal_task_kind_t;

typedef struct loom_kernel_literal_task_t {
  // Interpretation of the payload.
  loom_kernel_literal_task_kind_t kind;
  // Byte or sequence length.
  uint32_t count;
  union {
    // Immediate semantic word or encoding identity.
    uint64_t word;
    // Borrowed immutable sequence or bytes.
    const void* data;
    // Type carried by value, including a fact-refined root shape.
    loom_type_t type;
    // Attribute carried by value.
    loom_attribute_t attribute;
  };
} loom_kernel_literal_task_t;

typedef struct loom_kernel_literal_frame_t {
  // Previous live frame, or NULL at the embedded root.
  struct loom_kernel_literal_frame_t* previous;
  // Reusable empty continuation frame, or NULL.
  struct loom_kernel_literal_frame_t* next;
  // Occupied entries in this frame.
  uint32_t count;
  // Fixed-size task block; no abandoned growing-array copies.
  loom_kernel_literal_task_t values[8];
} loom_kernel_literal_frame_t;

typedef struct loom_kernel_literal_cursor_t {
  // Immutable owner of every referenced module-local identity.
  const loom_module_t* module;
  // Invocation-owned scratch, never the retained source allocator.
  iree_arena_allocator_t* arena;
  // Inline task storage covers scalar and shallow payloads.
  loom_kernel_literal_frame_t initial;
  // Current nonempty frame, or the empty embedded root.
  loom_kernel_literal_frame_t* top;
  // Reached low-24-bit type/encoding identities, in separate domains.
  loom_ordinal_map_t low;
  // Lazily allocated high-byte type scopes for the complete 32-bit ID space.
  loom_ordinal_map_t** high;
  // Next deterministic first-use ordinal within this semantic stream.
  uint32_t reached_count;
} loom_kernel_literal_cursor_t;

static iree_status_t loom_kernel_literal_push(
    loom_kernel_literal_cursor_t* cursor, loom_kernel_literal_task_t task) {
  if (cursor->top->count == IREE_ARRAYSIZE(cursor->top->values)) {
    if (!cursor->top->next) {
      loom_kernel_literal_frame_t* frame = NULL;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate(cursor->arena, sizeof(*frame), (void**)&frame));
      *frame = (loom_kernel_literal_frame_t){.previous = cursor->top};
      cursor->top->next = frame;
    }
    cursor->top = cursor->top->next;
  }
  cursor->top->values[cursor->top->count++] = task;
  return iree_ok_status();
}

static iree_status_t loom_kernel_literal_word(
    loom_kernel_literal_cursor_t* cursor, uint64_t word) {
  return loom_kernel_literal_push(
      cursor, (loom_kernel_literal_task_t){.kind = LOOM_KERNEL_LITERAL_WORD,
                                           .word = word});
}

static iree_status_t loom_kernel_literal_sequence(
    loom_kernel_literal_cursor_t* cursor, loom_kernel_literal_task_kind_t kind,
    const void* data, uint32_t count) {
  if (!count) {
    return iree_ok_status();
  }
  return loom_kernel_literal_push(
      cursor,
      (loom_kernel_literal_task_t){.kind = kind, .count = count, .data = data});
}

static iree_status_t loom_kernel_literal_bytes(
    loom_kernel_literal_cursor_t* cursor, const void* data, uint32_t count) {
  return loom_kernel_literal_push(
      cursor,
      (loom_kernel_literal_task_t){
          .kind = LOOM_KERNEL_LITERAL_BYTES, .count = count, .data = data});
}

static iree_status_t loom_kernel_literal_string(
    loom_kernel_literal_cursor_t* cursor, loom_string_id_t id) {
  if (id == LOOM_STRING_ID_INVALID) {
    return loom_kernel_literal_word(cursor, 0);
  }
  const iree_string_view_t string =
      loom_string_table_get(&cursor->module->strings, id);
  IREE_RETURN_IF_ERROR(
      loom_kernel_literal_bytes(cursor, string.data, string.size));
  return loom_kernel_literal_word(cursor, 1);
}

static iree_status_t loom_kernel_literal_reach(
    loom_kernel_literal_cursor_t* cursor, uint8_t domain, uint32_t identity,
    bool* out_seen, uint32_t* out_ordinal) {
  const uint8_t high = (uint8_t)(identity >> 24);
  loom_ordinal_map_t* map = &cursor->low;
  if (high) {
    if (!cursor->high) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          cursor->arena, 256, sizeof(*cursor->high), (void**)&cursor->high));
      memset(cursor->high, 0, 256 * sizeof(*cursor->high));
    }
    if (!cursor->high[high]) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate(cursor->arena, sizeof(*map),
                                               (void**)&cursor->high[high]));
      loom_ordinal_map_initialize(cursor->arena, cursor->high[high]);
    }
    map = cursor->high[high];
  }
  *out_seen =
      loom_ordinal_map_lookup(map, domain, identity & 0xFFFFFF, out_ordinal);
  if (*out_seen) {
    return iree_ok_status();
  }
  *out_ordinal = cursor->reached_count++;
  return loom_ordinal_map_insert(map, domain, identity & 0xFFFFFF,
                                 *out_ordinal);
}

static iree_status_t loom_kernel_literal_encoding(
    loom_kernel_literal_cursor_t* cursor, uint32_t id) {
  if (!id) {
    return loom_kernel_literal_word(cursor, 0);
  }
  bool seen = false;
  uint32_t ordinal = 0;
  IREE_RETURN_IF_ERROR(
      loom_kernel_literal_reach(cursor, 1, id, &seen, &ordinal));
  if (seen) {
    IREE_RETURN_IF_ERROR(loom_kernel_literal_word(cursor, ordinal));
    return loom_kernel_literal_word(cursor, 2);
  }
  const loom_encoding_t* encoding = loom_module_encoding(cursor->module, id);
  IREE_RETURN_IF_ERROR(loom_kernel_literal_sequence(
      cursor, LOOM_KERNEL_LITERAL_NAMED_ATTRIBUTES, encoding->attributes,
      encoding->attribute_count));
  IREE_RETURN_IF_ERROR(
      loom_kernel_literal_word(cursor, encoding->attribute_count));
  IREE_RETURN_IF_ERROR(loom_kernel_literal_string(cursor, encoding->name_id));
  return loom_kernel_literal_word(cursor, 1);
}

static iree_status_t loom_kernel_literal_type(
    loom_kernel_literal_cursor_t* cursor, loom_type_t type) {
  const loom_type_kind_t kind = loom_type_kind(type);
  const bool composite =
      kind == LOOM_TYPE_FUNCTION || kind == LOOM_TYPE_DIALECT ||
      kind == LOOM_TYPE_PARAMETERIZED ||
      (kind == LOOM_TYPE_REGISTER && loom_type_register_has_value_type(type));
  if (composite) {
    // Captured root refinement changes only shaped dimensions. Composite
    // descendants belong to the owner's canonical type table.
    const loom_type_id_t id = loom_module_lookup_type_id(cursor->module, type);
    IREE_ASSERT_NE(id, LOOM_TYPE_ID_INVALID);
    bool seen = false;
    uint32_t ordinal = 0;
    IREE_RETURN_IF_ERROR(
        loom_kernel_literal_reach(cursor, 0, id, &seen, &ordinal));
    if (seen) {
      IREE_RETURN_IF_ERROR(loom_kernel_literal_word(cursor, ordinal));
      return loom_kernel_literal_word(cursor, 1);
    }
  }
  switch (kind) {
    case LOOM_TYPE_REFERENCE: {
      IREE_RETURN_IF_ERROR(loom_kernel_literal_string(
          cursor, loom_type_reference_name_id(type)));
      IREE_RETURN_IF_ERROR(loom_kernel_literal_string(
          cursor, loom_type_reference_namespace_id(type)));
      break;
    }
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* data = loom_type_func_data(type);
      IREE_RETURN_IF_ERROR(loom_kernel_literal_sequence(
          cursor, LOOM_KERNEL_LITERAL_TYPES, data->types,
          (uint32_t)data->arg_count + data->result_count));
      IREE_RETURN_IF_ERROR(
          loom_kernel_literal_word(cursor, data->result_count));
      IREE_RETURN_IF_ERROR(loom_kernel_literal_word(cursor, data->arg_count));
      break;
    }
    case LOOM_TYPE_DIALECT: {
      IREE_RETURN_IF_ERROR(loom_kernel_literal_sequence(
          cursor, LOOM_KERNEL_LITERAL_TYPES, loom_type_dialect_params(type),
          loom_type_dialect_param_count(type)));
      IREE_RETURN_IF_ERROR(loom_kernel_literal_word(
          cursor, loom_type_dialect_param_count(type)));
      IREE_RETURN_IF_ERROR(
          loom_kernel_literal_string(cursor, loom_type_dialect_name_id(type)));
      break;
    }
    case LOOM_TYPE_PARAMETERIZED: {
      IREE_RETURN_IF_ERROR(loom_kernel_literal_sequence(
          cursor, LOOM_KERNEL_LITERAL_ATTRIBUTES,
          loom_type_parameterized_parameters(type),
          loom_type_parameterized_parameter_count(type)));
      IREE_RETURN_IF_ERROR(loom_kernel_literal_word(
          cursor, loom_type_parameterized_parameter_count(type)));
      // The retained preparation belongs to one fixed compiler context.
      IREE_RETURN_IF_ERROR(loom_kernel_literal_word(
          cursor, (uintptr_t)loom_type_parameterized_descriptor(type)));
      break;
    }
    case LOOM_TYPE_REGISTER:
      if (loom_type_register_has_value_type(type)) {
        const loom_register_type_data_t* data = loom_type_register_data(type);
        IREE_RETURN_IF_ERROR(loom_kernel_literal_push(
            cursor,
            (loom_kernel_literal_task_t){.kind = LOOM_KERNEL_LITERAL_TYPE,
                                         .type = data->value_type}));
        IREE_RETURN_IF_ERROR(
            loom_kernel_literal_word(cursor, data->carrier_payload1));
        IREE_RETURN_IF_ERROR(
            loom_kernel_literal_word(cursor, data->carrier_payload0));
      } else {
        IREE_RETURN_IF_ERROR(loom_kernel_literal_word(cursor, type.dims[1]));
        IREE_RETURN_IF_ERROR(loom_kernel_literal_word(cursor, type.dims[0]));
      }
      break;
    default:
      if (loom_type_has_inline_dims(type)) {
        IREE_RETURN_IF_ERROR(loom_kernel_literal_word(cursor, type.dims[1]));
        IREE_RETURN_IF_ERROR(loom_kernel_literal_word(cursor, type.dims[0]));
      } else {
        iree_status_t status = iree_ok_status();
        for (uint8_t i = loom_type_rank(type); i && iree_status_is_ok(status);
             --i) {
          status = loom_kernel_literal_word(cursor, loom_type_dim(type, i - 1));
        }
        IREE_RETURN_IF_ERROR(status);
      }
      break;
  }
  if (kind == LOOM_TYPE_PARAMETERIZED) {
    IREE_RETURN_IF_ERROR(
        loom_kernel_literal_word(cursor, type.parameterized_kind));
  } else {
    IREE_RETURN_IF_ERROR(loom_kernel_literal_push(
        cursor,
        (loom_kernel_literal_task_t){.kind = LOOM_KERNEL_LITERAL_ENCODING,
                                     .word = type.encoding_id}));
  }
  IREE_RETURN_IF_ERROR(loom_kernel_literal_word(cursor, type.encoding_flags));
  IREE_RETURN_IF_ERROR(loom_kernel_literal_word(cursor, type.header));
  return loom_kernel_literal_word(cursor, 0);
}

static iree_status_t loom_kernel_literal_attribute(
    loom_kernel_literal_cursor_t* cursor, loom_attribute_t attribute) {
  switch ((loom_attr_kind_t)attribute.kind) {
    case LOOM_ATTR_ABSENT:
      break;
    case LOOM_ATTR_I64:
    case LOOM_ATTR_F64:
    case LOOM_ATTR_BOOL:
    case LOOM_ATTR_ENUM: {
      IREE_RETURN_IF_ERROR(loom_kernel_literal_word(cursor, attribute.raw));
      break;
    }
    case LOOM_ATTR_STRING: {
      IREE_RETURN_IF_ERROR(
          loom_kernel_literal_string(cursor, attribute.string_id));
      break;
    }
    case LOOM_ATTR_STRING_ARRAY: {
      IREE_RETURN_IF_ERROR(
          loom_kernel_literal_sequence(cursor, LOOM_KERNEL_LITERAL_STRINGS,
                                       attribute.string_ids, attribute.count));
      IREE_RETURN_IF_ERROR(loom_kernel_literal_word(cursor, attribute.count));
      break;
    }
    case LOOM_ATTR_I64_ARRAY: {
      IREE_RETURN_IF_ERROR(loom_kernel_literal_bytes(
          cursor, attribute.i64_array,
          (uint32_t)attribute.count * sizeof(int64_t)));
      break;
    }
    case LOOM_ATTR_F64_ARRAY: {
      IREE_RETURN_IF_ERROR(loom_kernel_literal_bytes(
          cursor, attribute.f64_array,
          (uint32_t)attribute.count * sizeof(double)));
      break;
    }
    case LOOM_ATTR_ENUM_ARRAY: {
      IREE_RETURN_IF_ERROR(loom_kernel_literal_bytes(
          cursor, attribute.enum_array, attribute.count));
      break;
    }
    case LOOM_ATTR_SIGNED_ENUM_SET: {
      IREE_RETURN_IF_ERROR(loom_kernel_literal_bytes(
          cursor, attribute.signed_enum_set_words,
          (uint32_t)attribute.count * 2 * sizeof(uint64_t)));
      break;
    }
    case LOOM_ATTR_BYTES: {
      IREE_RETURN_IF_ERROR(loom_kernel_literal_bytes(cursor, attribute.bytes,
                                                     attribute.reserved_1));
      IREE_RETURN_IF_ERROR(loom_kernel_literal_word(cursor, attribute.flags));
      break;
    }
    case LOOM_ATTR_TYPE: {
      IREE_RETURN_IF_ERROR(loom_kernel_literal_push(
          cursor, (loom_kernel_literal_task_t){
                      .kind = LOOM_KERNEL_LITERAL_TYPE,
                      .type = loom_type_table_get(&cursor->module->types,
                                                  attribute.type_id)}));
      break;
    }
    case LOOM_ATTR_ENCODING: {
      IREE_RETURN_IF_ERROR(loom_kernel_literal_push(
          cursor,
          (loom_kernel_literal_task_t){.kind = LOOM_KERNEL_LITERAL_ENCODING,
                                       .word = attribute.encoding_id}));
      break;
    }
    case LOOM_ATTR_DICT: {
      IREE_RETURN_IF_ERROR(loom_kernel_literal_sequence(
          cursor, LOOM_KERNEL_LITERAL_NAMED_ATTRIBUTES, attribute.dict_entries,
          attribute.count));
      IREE_RETURN_IF_ERROR(loom_kernel_literal_word(cursor, attribute.count));
      break;
    }
    case LOOM_ATTR_PARAMETERIZED: {
      IREE_RETURN_IF_ERROR(loom_kernel_literal_sequence(
          cursor, LOOM_KERNEL_LITERAL_ATTRIBUTES, attribute.parameterized_slots,
          attribute.count));
      IREE_RETURN_IF_ERROR(loom_kernel_literal_word(cursor, attribute.count));
      IREE_RETURN_IF_ERROR(
          loom_kernel_literal_word(cursor, attribute.reserved_1));
      break;
    }
    case LOOM_ATTR_PARAMETERIZED_ARRAY: {
      IREE_RETURN_IF_ERROR(loom_kernel_literal_sequence(
          cursor, LOOM_KERNEL_LITERAL_ATTRIBUTES, attribute.parameterized_array,
          attribute.count));
      IREE_RETURN_IF_ERROR(loom_kernel_literal_word(cursor, attribute.count));
      break;
    }
    case LOOM_ATTR_PREDICATE_LIST: {
      IREE_RETURN_IF_ERROR(loom_kernel_literal_sequence(
          cursor, LOOM_KERNEL_LITERAL_PREDICATES, attribute.predicate_list,
          attribute.count));
      IREE_RETURN_IF_ERROR(loom_kernel_literal_word(cursor, attribute.count));
      break;
    }
    case LOOM_ATTR_SYMBOL:
      if (loom_symbol_ref_is_valid(attribute.symbol)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "staged literal requires a closed symbol payload");
      }
      break;
    case LOOM_ATTR_SYMBOL_ARRAY:
    case LOOM_ATTR_SYMBOL_SET: {
      IREE_RETURN_IF_ERROR(
          loom_kernel_literal_sequence(cursor, LOOM_KERNEL_LITERAL_SYMBOLS,
                                       attribute.symbol_refs, attribute.count));
      IREE_RETURN_IF_ERROR(loom_kernel_literal_word(cursor, attribute.count));
      break;
    }
    case LOOM_ATTR_SCOPED_ENUM:
      // Representation-scoped ordinals occur only on Low instructions, which
      // are not ConstantLike; context-free literal payloads cannot carry them.
      IREE_CHECK_UNREACHABLE(
          "closed literals have no representation-scoped enums");
    case LOOM_ATTR_ANY:
    case LOOM_ATTR_COUNT_:
      IREE_CHECK_UNREACHABLE("verified literals have concrete attribute kinds");
  }
  return loom_kernel_literal_word(cursor, attribute.kind);
}

static iree_status_t loom_kernel_literal_next(
    loom_kernel_literal_cursor_t* cursor, loom_kernel_literal_task_t* out_part,
    bool* out_present) {
  *out_present = false;
  iree_status_t status = iree_ok_status();
  while (cursor->top->count && iree_status_is_ok(status)) {
    const loom_kernel_literal_task_t task =
        cursor->top->values[--cursor->top->count];
    if (!cursor->top->count && cursor->top->previous) {
      cursor->top = cursor->top->previous;
    }
    switch (task.kind) {
      case LOOM_KERNEL_LITERAL_WORD:
      case LOOM_KERNEL_LITERAL_BYTES:
        *out_part = task;
        *out_present = true;
        return status;
      case LOOM_KERNEL_LITERAL_TYPE:
        status = loom_kernel_literal_type(cursor, task.type);
        break;
      case LOOM_KERNEL_LITERAL_ATTRIBUTE:
        status = loom_kernel_literal_attribute(cursor, task.attribute);
        break;
      case LOOM_KERNEL_LITERAL_ENCODING:
        status = loom_kernel_literal_encoding(cursor, task.word);
        break;
      case LOOM_KERNEL_LITERAL_VALUES: {
        const loom_kernel_specialization_value_t* value = task.data;
        IREE_RETURN_IF_ERROR(loom_kernel_literal_sequence(
            cursor, LOOM_KERNEL_LITERAL_VALUES, value + 1, task.count - 1));
        if (value->kind) {
          IREE_RETURN_IF_ERROR(loom_kernel_literal_sequence(
              cursor, LOOM_KERNEL_LITERAL_ATTRIBUTES, value->attributes,
              value->attribute_count));
          IREE_RETURN_IF_ERROR(
              loom_kernel_literal_word(cursor, value->attribute_count));
          IREE_RETURN_IF_ERROR(loom_kernel_literal_word(
              cursor, value->kind | ((uint64_t)value->instance_flags << 16)));
        } else {
          IREE_RETURN_IF_ERROR(
              loom_kernel_literal_word(cursor, value->facts.range_lo));
        }
        IREE_RETURN_IF_ERROR(
            loom_kernel_literal_word(cursor, value->kind != 0));
        status = loom_kernel_literal_push(
            cursor, (loom_kernel_literal_task_t){
                        .kind = LOOM_KERNEL_LITERAL_TYPE, .type = value->type});
        break;
      }
      case LOOM_KERNEL_LITERAL_ATTRIBUTES: {
        const loom_attribute_t* attributes = task.data;
        IREE_RETURN_IF_ERROR(
            loom_kernel_literal_sequence(cursor, LOOM_KERNEL_LITERAL_ATTRIBUTES,
                                         attributes + 1, task.count - 1));
        status = loom_kernel_literal_push(
            cursor,
            (loom_kernel_literal_task_t){.kind = LOOM_KERNEL_LITERAL_ATTRIBUTE,
                                         .attribute = *attributes});
        break;
      }
      case LOOM_KERNEL_LITERAL_NAMED_ATTRIBUTES: {
        const loom_named_attr_t* attributes = task.data;
        IREE_RETURN_IF_ERROR(loom_kernel_literal_sequence(
            cursor, LOOM_KERNEL_LITERAL_NAMED_ATTRIBUTES, attributes + 1,
            task.count - 1));
        IREE_RETURN_IF_ERROR(loom_kernel_literal_push(
            cursor,
            (loom_kernel_literal_task_t){.kind = LOOM_KERNEL_LITERAL_ATTRIBUTE,
                                         .attribute = attributes->value}));
        status = loom_kernel_literal_string(cursor, attributes->name_id);
        break;
      }
      case LOOM_KERNEL_LITERAL_TYPES: {
        const loom_type_t* types = task.data;
        IREE_RETURN_IF_ERROR(loom_kernel_literal_sequence(
            cursor, LOOM_KERNEL_LITERAL_TYPES, types + 1, task.count - 1));
        status = loom_kernel_literal_push(
            cursor, (loom_kernel_literal_task_t){
                        .kind = LOOM_KERNEL_LITERAL_TYPE, .type = *types});
        break;
      }
      case LOOM_KERNEL_LITERAL_STRINGS: {
        const loom_string_id_t* strings = task.data;
        IREE_RETURN_IF_ERROR(loom_kernel_literal_sequence(
            cursor, LOOM_KERNEL_LITERAL_STRINGS, strings + 1, task.count - 1));
        status = loom_kernel_literal_string(cursor, *strings);
        break;
      }
      case LOOM_KERNEL_LITERAL_SYMBOLS: {
        const loom_symbol_ref_t* symbols = task.data;
        if (loom_symbol_ref_is_valid(*symbols)) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "staged literal requires a closed symbol payload");
        }
        status = loom_kernel_literal_sequence(
            cursor, LOOM_KERNEL_LITERAL_SYMBOLS, symbols + 1, task.count - 1);
        break;
      }
      case LOOM_KERNEL_LITERAL_PREDICATES: {
        const loom_predicate_t* predicate = task.data;
        IREE_RETURN_IF_ERROR(
            loom_kernel_literal_sequence(cursor, LOOM_KERNEL_LITERAL_PREDICATES,
                                         predicate + 1, task.count - 1));
        for (uint8_t i = predicate->arg_count; i && iree_status_is_ok(status);
             --i) {
          if (predicate->arg_tags[i - 1] != LOOM_PRED_ARG_CONST) {
            status = iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "staged literal predicate refers to a caller SSA value");
          } else {
            status = loom_kernel_literal_word(cursor, predicate->args[i - 1]);
          }
        }
        IREE_RETURN_IF_ERROR(status);
        IREE_RETURN_IF_ERROR(
            loom_kernel_literal_word(cursor, predicate->arg_count));
        status = loom_kernel_literal_word(cursor, predicate->kind);
        break;
      }
    }
  }
  return status;
}

static iree_status_t loom_kernel_literal_initialize(
    loom_kernel_literal_cursor_t* cursor, const loom_module_t* module,
    const loom_kernel_specialization_value_t* values, uint16_t count,
    iree_arena_allocator_t* arena) {
  *cursor = (loom_kernel_literal_cursor_t){.module = module, .arena = arena};
  cursor->top = &cursor->initial;
  loom_ordinal_map_initialize(arena, &cursor->low);
  IREE_RETURN_IF_ERROR(loom_kernel_literal_sequence(
      cursor, LOOM_KERNEL_LITERAL_VALUES, values, count));
  return loom_kernel_literal_word(cursor, count);
}

// Fixed chunks retain each cursor output once without growing-array copies.
typedef struct loom_kernel_specialization_part_t {
  // Distinguishes semantic words from borrowed byte spans.
  bool bytes;
  // Byte span size; zero for a word.
  uint32_t count;
  union {
    // Exact semantic word.
    uint64_t word;
    // Borrowed caller bytes until the publication returns.
    const void* data;
  };
} loom_kernel_specialization_part_t;

struct loom_kernel_specialization_query_t {
  // Count of projected semantic parts.
  uint32_t count;
  // Complete packed key storage needed if retention is admitted.
  iree_host_size_t length;
  // Portable bucket accelerator; exact comparison resolves collisions.
  uint32_t hash;
  // Fixed-size part chunks owned by invocation scratch.
  loom_segmented_storage_t storage;
  // Shallow literals require no part chunk allocation.
  loom_kernel_specialization_part_t initial[16];
};

static const loom_kernel_specialization_part_t* loom_kernel_specialization_part(
    const loom_kernel_specialization_query_t* query, uint32_t i) {
  if (i < IREE_ARRAYSIZE(query->initial)) {
    return &query->initial[i];
  }
  i -= IREE_ARRAYSIZE(query->initial);
  return (const loom_kernel_specialization_part_t*)
             loom_segmented_storage_const_segment(&query->storage, i / 32) +
         i % 32;
}

iree_status_t loom_kernel_specialization_query_allocate(
    const loom_module_t* module,
    const loom_kernel_specialization_value_t* values, uint16_t count,
    iree_arena_allocator_t* arena,
    loom_kernel_specialization_query_t** out_query) {
  *out_query = NULL;
  loom_kernel_specialization_query_t* query = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*query), (void**)&query));
  *query = (loom_kernel_specialization_query_t){0};
  loom_segmented_storage_initialize(
      32 * sizeof(loom_kernel_specialization_part_t),
      iree_alignof(loom_kernel_specialization_part_t), &query->storage);
  loom_kernel_literal_cursor_t cursor;
  IREE_RETURN_IF_ERROR(
      loom_kernel_literal_initialize(&cursor, module, values, count, arena));
  uint32_t hash = loom_structural_hash_initialize();
  iree_status_t status = iree_ok_status();
  bool present = true;
  while (present && iree_status_is_ok(status)) {
    loom_kernel_literal_task_t part;
    status = loom_kernel_literal_next(&cursor, &part, &present);
    if (!iree_status_is_ok(status) || !present) {
      break;
    }
    const bool bytes = part.kind == LOOM_KERNEL_LITERAL_BYTES;
    const iree_host_size_t length =
        1 + (bytes ? sizeof(uint32_t) + (iree_host_size_t)part.count
                   : sizeof(uint64_t));
    if (!iree_host_size_checked_add(query->length, length, &query->length) ||
        query->count == UINT32_MAX) {
      status = iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
      break;
    }
    loom_kernel_specialization_part_t* output = NULL;
    if (query->count < IREE_ARRAYSIZE(query->initial)) {
      output = &query->initial[query->count];
    } else {
      const uint32_t ordinal = query->count - IREE_ARRAYSIZE(query->initial);
      if ((ordinal % 32) == 0) {
        status = loom_segmented_storage_append(&query->storage, arena,
                                               (void**)&output);
      } else {
        output =
            (loom_kernel_specialization_part_t*)loom_segmented_storage_segment(
                &query->storage, ordinal / 32) +
            ordinal % 32;
      }
    }
    if (iree_status_is_ok(status)) {
      output->bytes = bytes;
      output->count = bytes ? part.count : 0;
      if (bytes) {
        output->data = part.data;
      } else {
        output->word = part.word;
      }
      ++query->count;
      hash = loom_structural_hash_mix_u8(hash, bytes);
      hash = bytes ? loom_structural_hash_mix_bytes(hash, part.data, part.count)
                   : loom_structural_hash_mix_u64(hash, part.word);
    }
  }
  if (iree_status_is_ok(status)) {
    query->hash = loom_structural_hash_finalize(hash);
    *out_query = query;
  }
  return status;
}

uint32_t loom_kernel_specialization_query_hash(
    const loom_kernel_specialization_query_t* query) {
  return query->hash;
}
iree_host_size_t loom_kernel_specialization_query_size(
    const loom_kernel_specialization_query_t* query) {
  return query->length;
}

iree_status_t loom_kernel_specialization_key_capture(
    const loom_kernel_specialization_query_t* query,
    iree_host_size_t byte_limit, iree_arena_allocator_t* arena,
    loom_kernel_specialization_key_t* out_key) {
  *out_key = (loom_kernel_specialization_key_t){0};
  if (query->length > byte_limit) {
    return iree_ok_status();
  }
  uint8_t* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, query->length, (void**)&data));
  uint8_t* cursor = data;
  for (uint32_t i = 0; i < query->count; ++i) {
    const loom_kernel_specialization_part_t* part =
        loom_kernel_specialization_part(query, i);
    *cursor++ = part->bytes;
    if (part->bytes) {
      memcpy(cursor, &part->count, sizeof(part->count));
      cursor += sizeof(part->count);
      if (part->count) {
        memcpy(cursor, part->data, part->count);
      }
      cursor += part->count;
    } else {
      memcpy(cursor, &part->word, sizeof(part->word));
      cursor += sizeof(part->word);
    }
  }
  *out_key =
      (loom_kernel_specialization_key_t){.data = data, .length = query->length};
  return iree_ok_status();
}

bool loom_kernel_specialization_key_matches(
    const loom_kernel_specialization_query_t* query,
    loom_kernel_specialization_key_t key) {
  if (key.length != query->length) {
    return false;
  }
  const uint8_t* cursor = key.data;
  for (uint32_t i = 0; i < query->count; ++i) {
    const loom_kernel_specialization_part_t* part =
        loom_kernel_specialization_part(query, i);
    if (*cursor++ != part->bytes) {
      return false;
    }
    if (part->bytes) {
      uint32_t count = 0;
      memcpy(&count, cursor, sizeof(count));
      cursor += sizeof(count);
      if (count != part->count ||
          (count && memcmp(cursor, part->data, count))) {
        return false;
      }
      cursor += count;
    } else {
      uint64_t word = 0;
      memcpy(&word, cursor, sizeof(word));
      cursor += sizeof(word);
      if (word != part->word) {
        return false;
      }
    }
  }
  return true;
}
