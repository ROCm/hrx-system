// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/printer/name_plan.h"

#include <inttypes.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"

// Stack buffer size for formatting generated value-name suffixes.
#define LOOM_PRINT_NAME_SUFFIX_BUFFER_SIZE 32

typedef enum loom_print_name_resolution_kind_e {
  LOOM_PRINT_NAME_RESOLUTION_GENERATED = 0,
  LOOM_PRINT_NAME_RESOLUTION_EXPLICIT = 1,
  LOOM_PRINT_NAME_RESOLUTION_EXPLICIT_SUFFIX = 2,
} loom_print_name_resolution_kind_t;

struct loom_print_name_resolution_t {
  // Resolution mode selected for this value.
  uint8_t kind;
  // Collision-avoidance attempt selected for generated or suffixed names.
  uint8_t attempt;
};

static_assert(sizeof(loom_print_name_resolution_t) == 2,
              "name resolutions must remain compact");

typedef struct loom_print_name_index_entry_t {
  // Parser scope containing the explicit name.
  const void* scope;
  // Interned name ID plus one; zero marks an empty table entry.
  uint32_t name_key;
  // True when multiple printable values have this name in the same scope.
  bool duplicated;
} loom_print_name_index_entry_t;

static const loom_op_vtable_t* loom_print_name_defining_op_vtable(
    const loom_module_t* module, const loom_op_t* op) {
  if (!module || !module->context || !op) return NULL;
  return loom_context_resolve_op(module->context, op->kind);
}

// Returns the parser scope that receives |value_id|'s printed definition.
static const void* loom_print_name_parse_scope(const loom_module_t* module,
                                               loom_value_id_t value_id) {
  if (!module || value_id >= module->values.count) return NULL;
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    loom_block_t* block = loom_value_def_block(value);
    return block ? (const void*)block->parent_region : NULL;
  }

  loom_op_t* def_op = loom_value_def_op(value);
  if (def_op) {
    const loom_op_vtable_t* vtable =
        loom_print_name_defining_op_vtable(module, def_op);
    if (vtable && iree_any_bit_set(vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) {
      return (const void*)def_op;
    }
    loom_block_t* block = def_op->parent_block;
    return block ? (const void*)block->parent_region : NULL;
  }

  const loom_use_t* uses = loom_value_uses(value);
  if (!uses || value->use_count == 0) return NULL;
  loom_op_t* user_op = loom_use_user_op(uses[0]);
  const loom_op_vtable_t* user_vtable =
      loom_print_name_defining_op_vtable(module, user_op);
  if (user_vtable &&
      iree_any_bit_set(user_vtable->traits, LOOM_TRAIT_SYMBOL_DEFINE)) {
    return (const void*)user_op;
  }
  loom_block_t* block = user_op ? user_op->parent_block : NULL;
  return block ? (const void*)block->parent_region : NULL;
}

static bool loom_print_name_value_is_printable(const loom_module_t* module,
                                               loom_value_id_t value_id) {
  if (!module || value_id >= module->values.count) return false;
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return loom_value_def_block(value) != NULL;
  }
  loom_op_t* def_op = loom_value_def_op(value);
  if (def_op) return !iree_any_bit_set(def_op->flags, LOOM_OP_FLAG_DEAD);
  return value->use_count > 0;
}

static bool loom_print_name_value_has_name(const loom_module_t* module,
                                           loom_value_id_t value_id,
                                           loom_string_id_t* out_name_id) {
  *out_name_id = LOOM_STRING_ID_INVALID;
  if (!module || value_id >= module->values.count) return false;
  loom_string_id_t name_id = loom_module_value(module, value_id)->name_id;
  if (name_id == LOOM_STRING_ID_INVALID || name_id >= module->strings.count) {
    return false;
  }
  *out_name_id = name_id;
  return true;
}

static uint64_t loom_print_name_hash(const void* scope,
                                     loom_string_id_t name_id) {
  uint64_t value = (uint64_t)(uintptr_t)scope;
  value ^= (uint64_t)name_id + UINT64_C(0x9e3779b97f4a7c15) + (value << 6) +
           (value >> 2);
  value ^= value >> 30;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27;
  value *= UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

static loom_print_name_index_entry_t* loom_print_name_index_find(
    loom_print_name_index_entry_t* entries, iree_host_size_t capacity,
    const void* scope, loom_string_id_t name_id) {
  if (!entries || capacity == 0) return NULL;
  const uint32_t name_key = name_id + 1;
  const iree_host_size_t mask = capacity - 1;
  iree_host_size_t slot =
      (iree_host_size_t)loom_print_name_hash(scope, name_id) & mask;
  while (entries[slot].name_key != 0) {
    if (entries[slot].name_key == name_key && entries[slot].scope == scope) {
      return &entries[slot];
    }
    slot = (slot + 1) & mask;
  }
  return NULL;
}

static void loom_print_name_index_insert(loom_print_name_index_entry_t* entries,
                                         iree_host_size_t capacity,
                                         const void* scope,
                                         loom_string_id_t name_id) {
  const uint32_t name_key = name_id + 1;
  const iree_host_size_t mask = capacity - 1;
  iree_host_size_t slot =
      (iree_host_size_t)loom_print_name_hash(scope, name_id) & mask;
  while (entries[slot].name_key != 0) {
    if (entries[slot].name_key == name_key && entries[slot].scope == scope) {
      entries[slot].duplicated = true;
      return;
    }
    slot = (slot + 1) & mask;
  }
  entries[slot].scope = scope;
  entries[slot].name_key = name_key;
}

static bool loom_print_name_index_contains(
    const loom_module_t* module, loom_print_name_index_entry_t* entries,
    iree_host_size_t capacity, const void* scope, iree_string_view_t name) {
  loom_string_id_t name_id = loom_module_lookup_string(module, name);
  return name_id != LOOM_STRING_ID_INVALID &&
         loom_print_name_index_find(entries, capacity, scope, name_id) != NULL;
}

static iree_string_view_t loom_print_name_format_generated(
    loom_value_id_t value_id, uint8_t attempt, char* buffer,
    iree_host_size_t buffer_capacity) {
  int length = attempt == 0
                   ? iree_snprintf(buffer, buffer_capacity, "%" PRIu32,
                                   (uint32_t)value_id)
                   : iree_snprintf(buffer, buffer_capacity, "%" PRIu32 "$%u",
                                   (uint32_t)value_id, (unsigned)attempt);
  IREE_ASSERT(length > 0 && (iree_host_size_t)length < buffer_capacity);
  return iree_make_string_view(buffer, (iree_host_size_t)length);
}

static iree_string_view_t loom_print_name_format_explicit_candidate(
    iree_string_view_t base_name, loom_value_id_t value_id, uint8_t attempt,
    char* buffer, iree_host_size_t buffer_capacity) {
  memcpy(buffer, base_name.data, base_name.size);
  int suffix_length =
      attempt == 0
          ? iree_snprintf(buffer + base_name.size,
                          buffer_capacity - base_name.size, "$%" PRIu32,
                          (uint32_t)value_id)
          : iree_snprintf(buffer + base_name.size,
                          buffer_capacity - base_name.size, "$%" PRIu32 "$%u",
                          (uint32_t)value_id, (unsigned)attempt);
  IREE_ASSERT(suffix_length > 0 && (iree_host_size_t)suffix_length <
                                       buffer_capacity - base_name.size);
  return iree_make_string_view(
      buffer, base_name.size + (iree_host_size_t)suffix_length);
}

static loom_print_name_resolution_t loom_print_name_resolve_generated(
    const loom_module_t* module, loom_print_name_index_entry_t* entries,
    iree_host_size_t capacity, const void* scope, loom_value_id_t value_id) {
  char buffer[LOOM_PRINT_NAME_SUFFIX_BUFFER_SIZE];
  for (uint8_t attempt = 0; attempt < 8; ++attempt) {
    iree_string_view_t candidate = loom_print_name_format_generated(
        value_id, attempt, buffer, sizeof(buffer));
    if (!loom_print_name_index_contains(module, entries, capacity, scope,
                                        candidate)) {
      return (loom_print_name_resolution_t){
          .kind = LOOM_PRINT_NAME_RESOLUTION_GENERATED,
          .attempt = attempt,
      };
    }
  }
  return (loom_print_name_resolution_t){
      .kind = LOOM_PRINT_NAME_RESOLUTION_GENERATED,
  };
}

static loom_print_name_resolution_t loom_print_name_resolve_explicit_duplicate(
    const loom_module_t* module, loom_print_name_index_entry_t* entries,
    iree_host_size_t capacity, const void* scope, loom_value_id_t value_id,
    iree_string_view_t base_name, char* candidate_buffer,
    iree_host_size_t candidate_buffer_capacity) {
  for (uint8_t attempt = 0; attempt < 8; ++attempt) {
    iree_string_view_t candidate = loom_print_name_format_explicit_candidate(
        base_name, value_id, attempt, candidate_buffer,
        candidate_buffer_capacity);
    if (!loom_print_name_index_contains(module, entries, capacity, scope,
                                        candidate)) {
      return (loom_print_name_resolution_t){
          .kind = LOOM_PRINT_NAME_RESOLUTION_EXPLICIT_SUFFIX,
          .attempt = attempt,
      };
    }
  }
  return loom_print_name_resolve_generated(module, entries, capacity, scope,
                                           value_id);
}

static bool loom_print_name_direct_contains(const loom_module_t* module,
                                            const void* scope,
                                            iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < module->values.count; ++i) {
    const loom_value_id_t value_id = (loom_value_id_t)i;
    if (!loom_print_name_value_is_printable(module, value_id) ||
        loom_print_name_parse_scope(module, value_id) != scope) {
      continue;
    }
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    if (loom_print_name_value_has_name(module, value_id, &name_id) &&
        iree_string_view_equal(module->strings.entries[name_id], name)) {
      return true;
    }
  }
  return false;
}

static bool loom_print_name_direct_suffix_exists(const loom_module_t* module,
                                                 const void* scope,
                                                 iree_string_view_t base_name,
                                                 iree_string_view_t suffix) {
  for (iree_host_size_t i = 0; i < module->values.count; ++i) {
    const loom_value_id_t value_id = (loom_value_id_t)i;
    if (!loom_print_name_value_is_printable(module, value_id) ||
        loom_print_name_parse_scope(module, value_id) != scope) {
      continue;
    }
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    if (!loom_print_name_value_has_name(module, value_id, &name_id)) continue;
    iree_string_view_t name = module->strings.entries[name_id];
    if (name.size == base_name.size + suffix.size &&
        iree_string_view_starts_with(name, base_name) &&
        iree_string_view_equal(
            iree_string_view_substr(name, base_name.size, suffix.size),
            suffix)) {
      return true;
    }
  }
  return false;
}

static loom_print_name_resolution_t loom_print_name_resolve_generated_direct(
    const loom_module_t* module, const void* scope, loom_value_id_t value_id) {
  char buffer[LOOM_PRINT_NAME_SUFFIX_BUFFER_SIZE];
  for (uint8_t attempt = 0; attempt < 8; ++attempt) {
    iree_string_view_t candidate = loom_print_name_format_generated(
        value_id, attempt, buffer, sizeof(buffer));
    if (!loom_print_name_direct_contains(module, scope, candidate)) {
      return (loom_print_name_resolution_t){
          .kind = LOOM_PRINT_NAME_RESOLUTION_GENERATED,
          .attempt = attempt,
      };
    }
  }
  return (loom_print_name_resolution_t){
      .kind = LOOM_PRINT_NAME_RESOLUTION_GENERATED,
  };
}

static loom_print_name_resolution_t loom_print_name_resolve_direct(
    const loom_module_t* module, loom_value_id_t value_id) {
  const void* scope = loom_print_name_parse_scope(module, value_id);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  if (!loom_print_name_value_has_name(module, value_id, &name_id)) {
    return loom_print_name_resolve_generated_direct(module, scope, value_id);
  }

  bool duplicated = false;
  for (iree_host_size_t i = 0; i < module->values.count; ++i) {
    if (i == value_id) continue;
    const loom_value_id_t other_value_id = (loom_value_id_t)i;
    if (loom_print_name_value_is_printable(module, other_value_id) &&
        loom_print_name_parse_scope(module, other_value_id) == scope &&
        loom_module_value(module, other_value_id)->name_id == name_id) {
      duplicated = true;
      break;
    }
  }
  if (!duplicated) {
    return (loom_print_name_resolution_t){
        .kind = LOOM_PRINT_NAME_RESOLUTION_EXPLICIT,
    };
  }

  iree_string_view_t base_name = module->strings.entries[name_id];
  char suffix_buffer[LOOM_PRINT_NAME_SUFFIX_BUFFER_SIZE];
  for (uint8_t attempt = 0; attempt < 8; ++attempt) {
    int suffix_length =
        attempt == 0 ? iree_snprintf(suffix_buffer, sizeof(suffix_buffer),
                                     "$%" PRIu32, (uint32_t)value_id)
                     : iree_snprintf(suffix_buffer, sizeof(suffix_buffer),
                                     "$%" PRIu32 "$%u", (uint32_t)value_id,
                                     (unsigned)attempt);
    IREE_ASSERT(suffix_length > 0 &&
                (iree_host_size_t)suffix_length < sizeof(suffix_buffer));
    iree_string_view_t suffix =
        iree_make_string_view(suffix_buffer, (iree_host_size_t)suffix_length);
    if (!loom_print_name_direct_suffix_exists(module, scope, base_name,
                                              suffix)) {
      return (loom_print_name_resolution_t){
          .kind = LOOM_PRINT_NAME_RESOLUTION_EXPLICIT_SUFFIX,
          .attempt = attempt,
      };
    }
  }
  return loom_print_name_resolve_generated_direct(module, scope, value_id);
}

iree_status_t loom_print_name_plan_initialize(
    const loom_module_t* module, loom_print_name_plan_t* out_plan) {
  memset(out_plan, 0, sizeof(*out_plan));
  iree_arena_initialize(module->arena.block_pool, &out_plan->arena);
  if (module->values.count == 0) return iree_ok_status();

  iree_host_size_t named_value_count = 0;
  iree_host_size_t indexed_name_count = 0;
  iree_host_size_t maximum_name_length = 0;
  for (iree_host_size_t i = 0; i < module->values.count; ++i) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    if (!loom_print_name_value_has_name(module, (loom_value_id_t)i, &name_id)) {
      continue;
    }
    ++named_value_count;
    if (module->strings.entries[name_id].size > maximum_name_length) {
      maximum_name_length = module->strings.entries[name_id].size;
    }
    if (loom_print_name_value_is_printable(module, (loom_value_id_t)i)) {
      ++indexed_name_count;
    }
  }
  if (named_value_count == 0) return iree_ok_status();

  iree_status_t status = iree_arena_allocate_array(
      &out_plan->arena, module->values.count, sizeof(*out_plan->resolutions),
      (void**)&out_plan->resolutions);
  if (!iree_status_is_ok(status)) {
    loom_print_name_plan_deinitialize(out_plan);
    return status;
  }
  memset(out_plan->resolutions, 0,
         module->values.count * sizeof(*out_plan->resolutions));

  if (indexed_name_count == 0) {
    for (iree_host_size_t i = 0; i < module->values.count; ++i) {
      loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
      if (loom_print_name_value_has_name(module, (loom_value_id_t)i,
                                         &name_id)) {
        out_plan->resolutions[i].kind = LOOM_PRINT_NAME_RESOLUTION_EXPLICIT;
      }
    }
    return iree_ok_status();
  }

  iree_host_size_t index_capacity =
      iree_host_size_next_power_of_two((indexed_name_count * 4 + 2) / 3);
  if (index_capacity < 16) index_capacity = 16;
  if (index_capacity == 0) {
    loom_print_name_plan_deinitialize(out_plan);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SSA name index capacity exceeds storage limit");
  }

  iree_host_size_t index_offset = 0;
  iree_host_size_t candidate_buffer_offset = 0;
  iree_host_size_t temporary_size = 0;
  iree_arena_checkpoint_t temporary_checkpoint =
      iree_arena_checkpoint_save(&out_plan->arena);
  status = IREE_STRUCT_LAYOUT(
      0, &temporary_size,
      IREE_STRUCT_FIELD_ALIGNED(index_capacity, loom_print_name_index_entry_t,
                                iree_alignof(loom_print_name_index_entry_t),
                                &index_offset),
      IREE_STRUCT_FIELD(maximum_name_length, char, &candidate_buffer_offset),
      IREE_STRUCT_FIELD(LOOM_PRINT_NAME_SUFFIX_BUFFER_SIZE, char, NULL));
  void* temporary_storage = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate(&out_plan->arena, temporary_size,
                                 &temporary_storage);
  }
  if (!iree_status_is_ok(status)) {
    iree_arena_checkpoint_restore(&temporary_checkpoint);
    loom_print_name_plan_deinitialize(out_plan);
    return status;
  }

  loom_print_name_index_entry_t* index_entries =
      (loom_print_name_index_entry_t*)((uint8_t*)temporary_storage +
                                       index_offset);
  memset(index_entries, 0, index_capacity * sizeof(*index_entries));
  char* candidate_buffer = (char*)temporary_storage + candidate_buffer_offset;
  const iree_host_size_t candidate_buffer_capacity =
      temporary_size - candidate_buffer_offset;

  for (iree_host_size_t i = 0; i < module->values.count; ++i) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    if (loom_print_name_value_is_printable(module, (loom_value_id_t)i) &&
        loom_print_name_value_has_name(module, (loom_value_id_t)i, &name_id)) {
      loom_print_name_index_insert(
          index_entries, index_capacity,
          loom_print_name_parse_scope(module, (loom_value_id_t)i), name_id);
    }
  }

  for (iree_host_size_t i = 0; i < module->values.count; ++i) {
    const loom_value_id_t value_id = (loom_value_id_t)i;
    const void* scope = loom_print_name_parse_scope(module, value_id);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    if (!loom_print_name_value_has_name(module, value_id, &name_id)) {
      out_plan->resolutions[i] = loom_print_name_resolve_generated(
          module, index_entries, index_capacity, scope, value_id);
      continue;
    }

    loom_print_name_index_entry_t* entry = loom_print_name_index_find(
        index_entries, index_capacity, scope, name_id);
    const bool duplicated =
        entry && (entry->duplicated ||
                  !loom_print_name_value_is_printable(module, value_id));
    if (!duplicated) {
      out_plan->resolutions[i].kind = LOOM_PRINT_NAME_RESOLUTION_EXPLICIT;
      continue;
    }
    out_plan->resolutions[i] = loom_print_name_resolve_explicit_duplicate(
        module, index_entries, index_capacity, scope, value_id,
        module->strings.entries[name_id], candidate_buffer,
        candidate_buffer_capacity);
  }

  iree_arena_checkpoint_restore(&temporary_checkpoint);
  return iree_ok_status();
}

void loom_print_name_plan_deinitialize(loom_print_name_plan_t* plan) {
  iree_arena_deinitialize(&plan->arena);
  memset(plan, 0, sizeof(*plan));
}

static iree_status_t loom_print_name_write_resolution(
    loom_output_stream_t* stream, const loom_module_t* module,
    loom_value_id_t value_id, loom_print_name_resolution_t resolution) {
  if (resolution.kind == LOOM_PRINT_NAME_RESOLUTION_EXPLICIT) {
    loom_string_id_t name_id = loom_module_value(module, value_id)->name_id;
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '%'));
    return loom_output_stream_write(stream, module->strings.entries[name_id]);
  }

  if (resolution.kind == LOOM_PRINT_NAME_RESOLUTION_EXPLICIT_SUFFIX) {
    loom_string_id_t name_id = loom_module_value(module, value_id)->name_id;
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '%'));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write(stream, module->strings.entries[name_id]));
    if (resolution.attempt == 0) {
      return loom_output_stream_write_format(stream, "$%" PRIu32,
                                             (uint32_t)value_id);
    }
    return loom_output_stream_write_format(stream, "$%" PRIu32 "$%u",
                                           (uint32_t)value_id,
                                           (unsigned)resolution.attempt);
  }

  if (resolution.attempt == 0) {
    return loom_output_stream_write_format(stream, "%%%" PRIu32,
                                           (uint32_t)value_id);
  }
  return loom_output_stream_write_format(stream, "%%%" PRIu32 "$%u",
                                         (uint32_t)value_id,
                                         (unsigned)resolution.attempt);
}

iree_status_t loom_print_name_plan_write_value_ref(
    const loom_print_name_plan_t* plan, loom_output_stream_t* stream,
    const loom_module_t* module, loom_value_id_t value_id) {
  if (!module || value_id >= module->values.count) {
    return loom_output_stream_write_cstring(stream, "%?");
  }
  if (plan) {
    loom_print_name_resolution_t resolution = {
        .kind = LOOM_PRINT_NAME_RESOLUTION_GENERATED,
    };
    if (plan->resolutions) resolution = plan->resolutions[value_id];
    return loom_print_name_write_resolution(stream, module, value_id,
                                            resolution);
  }

  return loom_print_name_write_resolution(
      stream, module, value_id,
      loom_print_name_resolve_direct(module, value_id));
}
