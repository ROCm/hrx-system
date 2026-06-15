// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/parser/scope.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/format/text/parser/context.h"
#include "loom/format/text/parser/diagnostics.h"

//===----------------------------------------------------------------------===//
// Name scope
//===----------------------------------------------------------------------===//

// Fibonacci hashing for interned string IDs. Multiplying by the golden ratio
// constant mixes sequential IDs well for power-of-two table capacities.
static uint32_t loom_parser_scope_hash(loom_string_id_t name_id) {
  return (uint32_t)name_id * 2654435769u;
}

static void loom_parser_scope_initialize_entries(
    loom_parser_scope_entry_t* entries, iree_host_size_t capacity) {
  for (iree_host_size_t i = 0; i < capacity; ++i) {
    entries[i].name_id = LOOM_STRING_ID_INVALID;
    entries[i].value_id = LOOM_VALUE_ID_INVALID;
  }
}

// Resets a parser-owned scope frame for reuse under a new lexical parent while
// preserving the retained hash-table allocation.
static void loom_parser_scope_reset(loom_parser_scope_t* scope,
                                    loom_parser_scope_t* parent) {
  bool had_entries = scope->count > 0;
  scope->next_free = NULL;
  scope->parent = parent;
  scope->count = 0;
  if (had_entries && scope->capacity > 0) {
    loom_parser_scope_initialize_entries(scope->entries, scope->capacity);
  }
}

// Ensures the scope's hash table is allocated and has room for at
// least one more entry. Called lazily before the first define().
static iree_status_t loom_parser_scope_ensure_capacity(
    loom_parser_scope_t* scope, iree_arena_allocator_t* arena) {
  // Grow if load factor exceeds ~70%.
  iree_host_size_t needed = scope->count + 1;
  if (scope->capacity > 0 && needed * 10 < scope->capacity * 7) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity =
      scope->capacity == 0 ? 16 : scope->capacity * 2;
  loom_parser_scope_entry_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, new_capacity, sizeof(loom_parser_scope_entry_t),
      (void**)&new_entries));
  loom_parser_scope_initialize_entries(new_entries, new_capacity);
  // Rehash existing entries.
  for (iree_host_size_t i = 0; i < scope->capacity; ++i) {
    if (scope->entries[i].name_id == LOOM_STRING_ID_INVALID) {
      continue;
    }
    uint32_t hash = loom_parser_scope_hash(scope->entries[i].name_id);
    iree_host_size_t slot = hash & (new_capacity - 1);
    while (new_entries[slot].name_id != LOOM_STRING_ID_INVALID) {
      slot = (slot + 1) & (new_capacity - 1);
    }
    new_entries[slot] = scope->entries[i];
  }
  // Old entries are arena-allocated; no free is needed.
  scope->entries = new_entries;
  scope->capacity = new_capacity;
  return iree_ok_status();
}

iree_status_t loom_parser_scope_define(loom_parser_scope_t* scope,
                                       iree_arena_allocator_t* arena,
                                       loom_string_id_t name_id,
                                       loom_value_id_t value_id,
                                       bool* out_duplicate) {
  if (out_duplicate) {
    *out_duplicate = false;
  }
  IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
  IREE_RETURN_IF_ERROR(loom_parser_scope_ensure_capacity(scope, arena));
  uint32_t hash = loom_parser_scope_hash(name_id);
  iree_host_size_t slot = hash & (scope->capacity - 1);
  while (scope->entries[slot].name_id != LOOM_STRING_ID_INVALID) {
    if (scope->entries[slot].name_id == name_id) {
      if (out_duplicate) {
        *out_duplicate = true;
      }
      return iree_ok_status();
    }
    slot = (slot + 1) & (scope->capacity - 1);
  }
  scope->entries[slot].name_id = name_id;
  scope->entries[slot].value_id = value_id;
  ++scope->count;
  return iree_ok_status();
}

loom_value_id_t loom_parser_scope_lookup_local(const loom_parser_scope_t* scope,
                                               loom_string_id_t name_id) {
  if (name_id == LOOM_STRING_ID_INVALID) {
    return LOOM_VALUE_ID_INVALID;
  }
  if (!scope || scope->capacity == 0) {
    return LOOM_VALUE_ID_INVALID;
  }
  uint32_t hash = loom_parser_scope_hash(name_id);
  iree_host_size_t slot = hash & (scope->capacity - 1);
  while (scope->entries[slot].name_id != LOOM_STRING_ID_INVALID) {
    if (scope->entries[slot].name_id == name_id) {
      return scope->entries[slot].value_id;
    }
    slot = (slot + 1) & (scope->capacity - 1);
  }
  return LOOM_VALUE_ID_INVALID;
}

loom_value_id_t loom_parser_scope_lookup(const loom_parser_scope_t* scope,
                                         loom_string_id_t name_id) {
  if (name_id == LOOM_STRING_ID_INVALID) {
    return LOOM_VALUE_ID_INVALID;
  }
  const loom_parser_scope_t* current = scope;
  while (current) {
    loom_value_id_t value_id = loom_parser_scope_lookup_local(current, name_id);
    if (value_id != LOOM_VALUE_ID_INVALID) {
      return value_id;
    }
    current = current->parent;
  }
  return LOOM_VALUE_ID_INVALID;
}

iree_status_t loom_parser_scope_push(loom_parser_t* parser,
                                     loom_parser_scope_t* parent,
                                     loom_parser_scope_t** out_scope) {
  loom_parser_scope_t* scope = parser->scope_free_list;
  if (scope) {
    parser->scope_free_list = scope->next_free;
    loom_parser_scope_reset(scope, parent);
    *out_scope = scope;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &parser->parser_arena, sizeof(loom_parser_scope_t), (void**)&scope));
  memset(scope, 0, sizeof(*scope));
  scope->parent = parent;
  *out_scope = scope;
  return iree_ok_status();
}

void loom_parser_scope_pop(loom_parser_t* parser) {
  loom_parser_scope_t* scope = parser->scope;
  parser->scope = scope->parent;
  scope->parent = NULL;
  scope->next_free = parser->scope_free_list;
  parser->scope_free_list = scope;
}

//===----------------------------------------------------------------------===//
// Definition scope
//===----------------------------------------------------------------------===//

iree_status_t loom_parser_add_unresolved_placeholder(loom_parser_t* parser,
                                                     loom_value_id_t value_id,
                                                     loom_token_t name_token) {
  loom_parser_unresolved_placeholders_t* placeholders =
      &parser->unresolved_placeholders;
  if (placeholders->count >= placeholders->capacity) {
    iree_host_size_t capacity = placeholders->capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &parser->parser_arena, placeholders->count, placeholders->count + 1,
        sizeof(loom_parser_unresolved_placeholder_t), &capacity,
        (void**)&placeholders->entries));
    placeholders->capacity = capacity;
  }
  placeholders->entries[placeholders->count++] =
      (loom_parser_unresolved_placeholder_t){
          .value_id = value_id,
          .name_token = name_token,
          .resolved = false,
      };
  return iree_ok_status();
}

static loom_parser_unresolved_placeholder_t*
loom_parser_lookup_unresolved_placeholder(loom_parser_t* parser,
                                          loom_value_id_t value_id) {
  if (!loom_parser_in_definition_scope(parser)) {
    return NULL;
  }
  for (iree_host_size_t i = parser->definition_scope.placeholder_start;
       i < parser->unresolved_placeholders.count; ++i) {
    loom_parser_unresolved_placeholder_t* placeholder =
        &parser->unresolved_placeholders.entries[i];
    if (placeholder->value_id == value_id) {
      return placeholder;
    }
  }
  return NULL;
}

iree_status_t loom_parser_definition_scope_push(
    loom_parser_t* parser, uint16_t pop_at,
    loom_parser_definition_scope_flags_t flags) {
  if (loom_parser_in_definition_scope(parser)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "nested Scope(...) format elements are not supported");
  }
  IREE_RETURN_IF_ERROR(
      loom_parser_scope_push(parser, parser->scope, &parser->scope));
  parser->definition_scope.placeholder_start =
      parser->unresolved_placeholders.count;
  parser->definition_scope.pop_at = pop_at;
  parser->definition_scope.flags = flags;
  return iree_ok_status();
}

void loom_parser_definition_scope_discard(loom_parser_t* parser) {
  if (!loom_parser_in_definition_scope(parser)) {
    return;
  }
  loom_parser_scope_pop(parser);
  parser->unresolved_placeholders.count =
      parser->definition_scope.placeholder_start;
  parser->definition_scope = (loom_parser_definition_scope_t){
      .placeholder_start = 0,
      .pop_at = UINT16_MAX,
      .flags = 0,
  };
}

iree_status_t loom_parser_definition_scope_pop_if_needed(
    loom_parser_t* parser, uint16_t format_index) {
  if (parser->definition_scope.pop_at != format_index) {
    return iree_ok_status();
  }
  const iree_host_size_t placeholder_start =
      parser->definition_scope.placeholder_start;
  for (iree_host_size_t i = placeholder_start;
       i < parser->unresolved_placeholders.count; ++i) {
    const loom_parser_unresolved_placeholder_t placeholder =
        parser->unresolved_placeholders.entries[i];
    if (placeholder.value_id == LOOM_VALUE_ID_INVALID ||
        placeholder.value_id >= parser->module->values.count) {
      continue;
    }
    if (placeholder.resolved) {
      continue;
    }
    loom_diagnostic_param_t params[] = {
        loom_param_string(placeholder.name_token.text),
    };
    iree_status_t status =
        loom_parser_emit(parser, LOOM_ERR_PARSE_022, params,
                         IREE_ARRAYSIZE(params), placeholder.name_token);
    loom_parser_definition_scope_discard(parser);
    return status;
  }
  loom_parser_definition_scope_discard(parser);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Result scope
//===----------------------------------------------------------------------===//

static iree_host_size_t loom_parser_result_scope_hash_capacity_for_count(
    iree_host_size_t entry_count) {
  iree_host_size_t hash_capacity = 32;
  while (entry_count * 10 >= hash_capacity * 7) {
    hash_capacity *= 2;
  }
  return hash_capacity;
}

iree_status_t loom_parser_result_scope_prepare(
    loom_parser_result_scope_t* scope, iree_host_size_t entry_count,
    iree_arena_allocator_t* arena) {
  scope->count = 0;
  scope->hash_capacity = 0;
  if (entry_count <= LOOM_PARSER_RESULT_SCOPE_INLINE_ENTRIES) {
    return iree_ok_status();
  }

  iree_host_size_t hash_capacity =
      loom_parser_result_scope_hash_capacity_for_count(entry_count);
  if (hash_capacity > scope->hash_storage_capacity) {
    loom_parser_scope_entry_t* hash_entries = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, hash_capacity, sizeof(*hash_entries), (void**)&hash_entries));
    scope->hash_entries = hash_entries;
    scope->hash_storage_capacity = hash_capacity;
  }
  loom_parser_scope_initialize_entries(scope->hash_entries, hash_capacity);
  scope->hash_capacity = hash_capacity;
  return iree_ok_status();
}

void loom_parser_result_scope_reset(loom_parser_result_scope_t* scope) {
  scope->count = 0;
  scope->hash_capacity = 0;
}

iree_status_t loom_parser_result_scope_define(loom_parser_result_scope_t* scope,
                                              loom_string_id_t name_id,
                                              loom_value_id_t value_id,
                                              bool* out_duplicate) {
  if (out_duplicate) {
    *out_duplicate = false;
  }
  IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
  if (scope->hash_capacity == 0) {
    for (uint16_t entry_index = 0; entry_index < scope->count; ++entry_index) {
      if (scope->inline_entries[entry_index].name_id == name_id) {
        if (out_duplicate) {
          *out_duplicate = true;
        }
        return iree_ok_status();
      }
    }
    IREE_ASSERT(scope->count < LOOM_PARSER_RESULT_SCOPE_INLINE_ENTRIES);
    scope->inline_entries[scope->count].name_id = name_id;
    scope->inline_entries[scope->count].value_id = value_id;
    ++scope->count;
    return iree_ok_status();
  }

  iree_host_size_t slot =
      loom_parser_scope_hash(name_id) & (scope->hash_capacity - 1);
  while (scope->hash_entries[slot].name_id != LOOM_STRING_ID_INVALID) {
    if (scope->hash_entries[slot].name_id == name_id) {
      if (out_duplicate) {
        *out_duplicate = true;
      }
      return iree_ok_status();
    }
    slot = (slot + 1) & (scope->hash_capacity - 1);
  }
  scope->hash_entries[slot].name_id = name_id;
  scope->hash_entries[slot].value_id = value_id;
  ++scope->count;
  return iree_ok_status();
}

static loom_value_id_t loom_parser_result_scope_lookup(
    const loom_parser_result_scope_t* scope, loom_string_id_t name_id) {
  if (scope->count == 0) {
    return LOOM_VALUE_ID_INVALID;
  }
  if (scope->hash_capacity == 0) {
    for (uint16_t entry_index = 0; entry_index < scope->count; ++entry_index) {
      if (scope->inline_entries[entry_index].name_id == name_id) {
        return scope->inline_entries[entry_index].value_id;
      }
    }
    return LOOM_VALUE_ID_INVALID;
  }

  iree_host_size_t slot =
      loom_parser_scope_hash(name_id) & (scope->hash_capacity - 1);
  while (scope->hash_entries[slot].name_id != LOOM_STRING_ID_INVALID) {
    if (scope->hash_entries[slot].name_id == name_id) {
      return scope->hash_entries[slot].value_id;
    }
    slot = (slot + 1) & (scope->hash_capacity - 1);
  }
  return LOOM_VALUE_ID_INVALID;
}

loom_value_id_t loom_parser_lookup_value(const loom_parser_t* parser,
                                         loom_string_id_t name_id) {
  if (name_id == LOOM_STRING_ID_INVALID) {
    return LOOM_VALUE_ID_INVALID;
  }
  loom_value_id_t value_id =
      loom_parser_result_scope_lookup(&parser->result_scope, name_id);
  if (value_id != LOOM_VALUE_ID_INVALID) {
    return value_id;
  }
  return loom_parser_scope_lookup(parser->scope, name_id);
}

iree_status_t loom_parser_emit_duplicate_value_name(loom_parser_t* parser,
                                                    loom_token_t name_token) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(name_token.text),
  };
  return loom_parser_emit(parser, LOOM_ERR_PARSE_002, params,
                          IREE_ARRAYSIZE(params), name_token);
}

iree_status_t loom_parser_define_value_name(loom_parser_t* parser,
                                            loom_token_t name_token,
                                            loom_value_id_t value_id) {
  if (iree_string_view_is_empty(name_token.text)) {
    return iree_ok_status();
  }

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(parser->module, name_token.text, &name_id));
  IREE_RETURN_IF_ERROR(
      loom_module_set_value_name(parser->module, value_id, name_id));
  bool duplicate = false;
  IREE_RETURN_IF_ERROR(loom_parser_scope_define(
      parser->scope, &parser->parser_arena, name_id, value_id, &duplicate));
  if (!duplicate) {
    return iree_ok_status();
  }
  return loom_parser_emit_duplicate_value_name(parser, name_token);
}

iree_status_t loom_parser_define_value(loom_parser_t* parser,
                                       loom_token_t name_token,
                                       loom_type_t type,
                                       loom_value_id_t* out_value_id) {
  if (iree_string_view_is_empty(name_token.text)) {
    return loom_module_define_value(parser->module, type, out_value_id);
  }

  loom_string_id_t name_id =
      loom_module_lookup_string(parser->module, name_token.text);
  if (name_id != LOOM_STRING_ID_INVALID) {
    loom_value_id_t value_id =
        loom_parser_scope_lookup_local(parser->scope, name_id);
    if (value_id != LOOM_VALUE_ID_INVALID) {
      loom_parser_unresolved_placeholder_t* placeholder =
          loom_parser_lookup_unresolved_placeholder(parser, value_id);
      if (placeholder && !placeholder->resolved) {
        loom_type_t placeholder_type =
            parser->module->values.entries[value_id].type;
        if (loom_type_kind(placeholder_type) != LOOM_TYPE_NONE &&
            !loom_type_equal(placeholder_type, type)) {
          return loom_parser_emit_duplicate_value_name(parser, name_token);
        }
        IREE_RETURN_IF_ERROR(
            loom_module_set_value_type(parser->module, value_id, type));
        placeholder->resolved = true;
        *out_value_id = value_id;
        return iree_ok_status();
      }
      return loom_parser_emit_duplicate_value_name(parser, name_token);
    }
  }

  IREE_RETURN_IF_ERROR(
      loom_module_define_value(parser->module, type, out_value_id));
  return loom_parser_define_value_name(parser, name_token, *out_value_id);
}

iree_status_t loom_parser_resolve_value(loom_parser_t* parser,
                                        loom_token_t name_token,
                                        loom_value_id_t* out_value_id) {
  loom_string_id_t name_id =
      loom_module_lookup_string(parser->module, name_token.text);
  *out_value_id = loom_parser_lookup_value(parser, name_id);
  if (*out_value_id != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }

  loom_diagnostic_param_t params[] = {
      loom_param_string(name_token.text),
  };
  return loom_parser_emit(parser, LOOM_ERR_PARSE_001, params,
                          IREE_ARRAYSIZE(params), name_token);
}

iree_status_t loom_parser_emit_result_count_mismatch(
    loom_parser_t* parser, const loom_op_vtable_t* vtable,
    loom_token_t op_name_token, uint16_t expected_count,
    uint16_t actual_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_vtable_name(vtable)),
      loom_param_u32(expected_count),
      loom_param_u32(actual_count),
  };
  return loom_parser_emit(parser, LOOM_ERR_PARSE_009, params,
                          IREE_ARRAYSIZE(params), op_name_token);
}
