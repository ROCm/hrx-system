// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/context.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// FNV-1a hash over a byte array.
static uint32_t loom_hash_bytes(const void* data, iree_host_size_t length) {
  uint32_t hash = 2166136261u;
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < length; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

static uint32_t loom_hash_string(iree_string_view_t string) {
  return loom_hash_bytes(string.data, string.size);
}

//===----------------------------------------------------------------------===//
// loom_context_t
//===----------------------------------------------------------------------===//

void loom_context_initialize(iree_allocator_t allocator,
                             loom_context_t* out_context) {
  memset(out_context, 0, sizeof(*out_context));
  out_context->allocator = allocator;
}

void loom_context_deinitialize(loom_context_t* context) {
  iree_allocator_free(context->allocator, context->encoding_vtables.entries);
  iree_allocator_free(context->allocator, context->op_name_table.entries);
  memset(context, 0, sizeof(*context));
}

iree_status_t loom_context_register_dialect(
    loom_context_t* context, uint8_t dialect_id,
    const loom_op_vtable_t* const* vtables, uint16_t op_count) {
  if (dialect_id >= LOOM_DIALECT_BUILTIN_COUNT_) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dialect ID %u exceeds maximum %u", dialect_id,
                            LOOM_DIALECT_BUILTIN_COUNT_ - 1);
  }
  if (op_count > 0 && vtables == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dialect ID %u registration requires vtables",
                            dialect_id);
  }
  if (context->op_vtables.dialects[dialect_id].entries != NULL) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "dialect ID %u is already registered", dialect_id);
  }
  context->op_vtables.dialects[dialect_id].op_count = op_count;
  context->op_vtables.dialects[dialect_id].entries = vtables;
  return iree_ok_status();
}

iree_status_t loom_context_register_dialect_semantics(
    loom_context_t* context, uint8_t dialect_id,
    const loom_op_semantics_t* semantics, uint16_t op_count) {
  if (dialect_id >= LOOM_DIALECT_BUILTIN_COUNT_) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dialect ID %u exceeds maximum %u", dialect_id,
                            LOOM_DIALECT_BUILTIN_COUNT_ - 1);
  }
  if (op_count > 0 && semantics == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dialect ID %u semantic registration requires metadata", dialect_id);
  }
  loom_dialect_vtables_t* dialect = &context->op_vtables.dialects[dialect_id];
  if (dialect->entries == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "dialect ID %u vtables must be registered before "
                            "semantic metadata",
                            dialect_id);
  }
  if (dialect->op_count != op_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "dialect ID %u semantic count %u does not match "
                            "vtable count %u",
                            dialect_id, op_count, dialect->op_count);
  }
  if (dialect->semantics != NULL) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "dialect ID %u semantic metadata is already registered", dialect_id);
  }
  dialect->semantics = semantics;
  return iree_ok_status();
}

iree_status_t loom_context_register_encoding_vtable(
    loom_context_t* context, const loom_encoding_vtable_t* vtable) {
  if (!vtable || iree_string_view_is_empty(vtable->name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "encoding vtable registration requires a non-empty family name");
  }

  if (loom_context_lookup_encoding_vtable(context, vtable->name)) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "encoding family '%.*s' is already registered",
                            (int)vtable->name.size, vtable->name.data);
  }

  IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
      context->allocator, context->encoding_vtables.count + 1,
      sizeof(const loom_encoding_vtable_t*),
      &context->encoding_vtables.capacity,
      (void**)&context->encoding_vtables.entries));
  context->encoding_vtables.entries[context->encoding_vtables.count++] = vtable;
  return iree_ok_status();
}

// Builds the op name hash table from all registered dialects.
static iree_status_t loom_context_build_op_name_table(loom_context_t* context) {
  uint32_t total_ops = 0;
  for (uint8_t d = 0; d < LOOM_DIALECT_BUILTIN_COUNT_; ++d) {
    total_ops += context->op_vtables.dialects[d].op_count;
  }
  if (total_ops == 0) return iree_ok_status();

  // Size to next power of 2 at ~0.75 load factor.
  uint32_t capacity = iree_host_size_next_power_of_two((total_ops * 4 + 2) / 3);
  if (capacity < 32) capacity = 32;

  loom_op_name_entry_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(context->allocator, capacity,
                                                   sizeof(loom_op_name_entry_t),
                                                   (void**)&entries));
  memset(entries, 0, (iree_host_size_t)capacity * sizeof(*entries));

  uint32_t mask = capacity - 1;
  uint32_t count = 0;
  for (uint8_t d = 0; d < LOOM_DIALECT_BUILTIN_COUNT_; ++d) {
    const loom_dialect_vtables_t* dialect = &context->op_vtables.dialects[d];
    for (uint16_t i = 0; i < dialect->op_count; ++i) {
      const loom_op_vtable_t* vtable = dialect->entries[i];
      if (!vtable) continue;
      iree_string_view_t name = loom_op_vtable_name(vtable);
      uint32_t hash = loom_hash_string(name);
      uint32_t slot = hash & mask;
      while (entries[slot].vtable != NULL) {
        slot = (slot + 1) & mask;
      }
      entries[slot].name = name;
      entries[slot].kind = LOOM_OP_KIND(d, i);
      entries[slot].vtable = vtable;
      ++count;
    }
  }

  context->op_name_table.entries = entries;
  context->op_name_table.capacity = capacity;
  context->op_name_table.count = count;
  return iree_ok_status();
}

iree_status_t loom_context_finalize(loom_context_t* context) {
  return loom_context_build_op_name_table(context);
}

//===----------------------------------------------------------------------===//
// Op lookup
//===----------------------------------------------------------------------===//

const loom_op_vtable_t* loom_context_resolve_op(const loom_context_t* context,
                                                loom_op_kind_t kind) {
  uint8_t dialect_id = loom_op_dialect_id(kind);
  if (dialect_id >= LOOM_DIALECT_BUILTIN_COUNT_) return NULL;
  const loom_dialect_vtables_t* dialect =
      &context->op_vtables.dialects[dialect_id];
  uint8_t op_index = loom_op_dialect_index(kind);
  if (op_index >= dialect->op_count) return NULL;
  return dialect->entries[op_index];
}

loom_op_semantics_t loom_context_resolve_op_semantics(
    const loom_context_t* context, loom_op_kind_t kind) {
  uint8_t dialect_id = loom_op_dialect_id(kind);
  if (dialect_id >= LOOM_DIALECT_BUILTIN_COUNT_) {
    return loom_op_semantics_empty();
  }
  const loom_dialect_vtables_t* dialect =
      &context->op_vtables.dialects[dialect_id];
  uint8_t op_index = loom_op_dialect_index(kind);
  if (op_index >= dialect->op_count || dialect->semantics == NULL) {
    return loom_op_semantics_empty();
  }
  return dialect->semantics[op_index];
}

const loom_op_vtable_t* loom_context_lookup_op_by_name(
    const loom_context_t* context, iree_string_view_t name,
    loom_op_kind_t* out_kind) {
  const loom_op_name_table_t* table = &context->op_name_table;
  if (table->capacity == 0) return NULL;
  uint32_t mask = table->capacity - 1;
  uint32_t hash = loom_hash_string(name);
  uint32_t slot = hash & mask;
  while (table->entries[slot].vtable != NULL) {
    if (iree_string_view_equal(table->entries[slot].name, name)) {
      *out_kind = table->entries[slot].kind;
      return table->entries[slot].vtable;
    }
    slot = (slot + 1) & mask;
  }
  return NULL;
}

const loom_encoding_vtable_t* loom_context_lookup_encoding_vtable(
    const loom_context_t* context, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < context->encoding_vtables.count; ++i) {
    const loom_encoding_vtable_t* vtable = context->encoding_vtables.entries[i];
    if (vtable && iree_string_view_equal(vtable->name, name)) {
      return vtable;
    }
  }
  return NULL;
}

//===----------------------------------------------------------------------===//
// Op convenience accessors
//===----------------------------------------------------------------------===//

const loom_op_vtable_t* loom_op_vtable(const loom_module_t* module,
                                       const loom_op_t* op) {
  return loom_context_resolve_op(module->context, op->kind);
}

iree_string_view_t loom_op_name(const loom_module_t* module,
                                const loom_op_t* op) {
  const loom_op_vtable_t* vtable =
      loom_context_resolve_op(module->context, op->kind);
  if (vtable) return loom_op_vtable_name(vtable);
  return IREE_SV("unknown");
}

loom_op_semantics_t loom_op_semantics(const loom_module_t* module,
                                      const loom_op_t* op) {
  return loom_context_resolve_op_semantics(module->context, op->kind);
}

bool loom_op_has_trait(const loom_module_t* module, const loom_op_t* op,
                       loom_trait_flags_t trait) {
  const loom_op_vtable_t* vtable =
      loom_context_resolve_op(module->context, op->kind);
  if (vtable) return (vtable->traits & trait) != 0;
  return false;
}
