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
  iree_allocator_free(context->allocator, context->encodings.vtables.entries);
  iree_allocator_free(context->allocator, context->encodings.names.entries);
  iree_allocator_free(context->allocator, context->op_name_table.entries);
  iree_allocator_free(context->allocator,
                      context->parameterized_attr_name_table.entries);
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

iree_status_t loom_context_register_parameterized_attrs(
    loom_context_t* context, uint8_t dialect_id,
    const loom_parameterized_attr_descriptor_t* descriptors,
    iree_host_size_t descriptor_count) {
  if (dialect_id >= LOOM_DIALECT_BUILTIN_COUNT_) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dialect ID %u exceeds maximum %u", dialect_id,
                            LOOM_DIALECT_BUILTIN_COUNT_ - 1);
  }
  if (descriptor_count == 0 || descriptors == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dialect ID %u parameterized attribute registration requires at "
        "least one descriptor",
        dialect_id);
  }
  if (descriptor_count > UINT8_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "dialect ID %u has %" PRIhsz
        " parameterized attribute families, exceeding the uint8_t limit",
        dialect_id, descriptor_count);
  }
  loom_dialect_parameterized_attrs_t* dialect =
      &context->parameterized_attrs.dialects[dialect_id];
  if (dialect->entries != NULL) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "dialect ID %u parameterized attributes are already registered",
        dialect_id);
  }
  for (iree_host_size_t i = 0; i < descriptor_count; ++i) {
    const loom_parameterized_attr_descriptor_t* descriptor = &descriptors[i];
    loom_parameterized_attr_kind_t expected_kind =
        LOOM_PARAMETERIZED_ATTR_KIND(dialect_id, i);
    if (descriptor->kind != expected_kind) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dialect ID %u parameterized attribute descriptor %" PRIhsz
          " has kind 0x%04X instead of 0x%04X",
          dialect_id, i, descriptor->kind, expected_kind);
    }
    if (descriptor->name == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dialect ID %u parameterized attribute descriptor %" PRIhsz
          " has no name",
          dialect_id, i);
    }
    if (iree_string_view_is_empty(loom_bstring_view(descriptor->name))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dialect ID %u parameterized attribute descriptor %" PRIhsz
          " has an empty name",
          dialect_id, i);
    }
    if (descriptor->parameter_count > 0 &&
        descriptor->parameter_descriptors == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameterized attribute '%.*s' has %u parameters but no "
          "descriptors",
          (int)loom_bstring_view(descriptor->name).size,
          loom_bstring_view(descriptor->name).data,
          descriptor->parameter_count);
    }
  }
  dialect->count = (uint8_t)descriptor_count;
  dialect->entries = descriptors;
  return iree_ok_status();
}

iree_status_t loom_context_register_encoding_vtable(
    loom_context_t* context, const loom_encoding_vtable_t* vtable) {
  if (!vtable || !vtable->descriptor || !vtable->descriptor->name ||
      iree_string_view_is_empty(loom_bstring_view(vtable->descriptor->name))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "encoding vtable registration requires a descriptor with a non-empty "
        "family name");
  }
  if (vtable->descriptor->parameter_count > 0 &&
      !vtable->descriptor->parameter_descriptors) {
    iree_string_view_t name = loom_bstring_view(vtable->descriptor->name);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "encoding family '%.*s' has %u parameters but no descriptors",
        (int)name.size, name.data, vtable->descriptor->parameter_count);
  }
  if ((vtable->is_static_valid == NULL) != (vtable->diagnose_static == NULL)) {
    iree_string_view_t name = loom_bstring_view(vtable->descriptor->name);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "encoding family '%.*s' must provide both static validity and "
        "diagnostic callbacks",
        (int)name.size, name.data);
  }

  iree_string_view_t name = loom_bstring_view(vtable->descriptor->name);
  for (iree_host_size_t i = 0; i < context->encodings.vtables.count; ++i) {
    const loom_encoding_vtable_t* registered_vtable =
        context->encodings.vtables.entries[i];
    iree_string_view_t registered_name =
        loom_bstring_view(registered_vtable->descriptor->name);
    if (!iree_string_view_equal(registered_name, name)) continue;
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "encoding family '%.*s' is already registered",
                            (int)name.size, name.data);
  }

  if (context->encodings.vtables.count >= UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "encoding family registry has reached its uint16_t limit");
  }

  IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
      context->allocator, context->encodings.vtables.count + 1,
      sizeof(const loom_encoding_vtable_t*),
      &context->encodings.vtables.capacity,
      (void**)&context->encodings.vtables.entries));
  context->encodings.vtables.entries[context->encodings.vtables.count++] =
      vtable;
  return iree_ok_status();
}

static iree_status_t loom_context_build_encoding_family_name_table(
    loom_context_t* context) {
  iree_host_size_t family_count = context->encodings.vtables.count;
  if (family_count == 0) return iree_ok_status();

  uint32_t capacity = iree_host_size_next_power_of_two(
      (iree_host_size_t)(family_count * 4 + 2) / 3);
  if (capacity < 8) capacity = 8;
  loom_encoding_family_name_entry_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      context->allocator, capacity, sizeof(*entries), (void**)&entries));
  memset(entries, 0, (iree_host_size_t)capacity * sizeof(*entries));

  uint32_t mask = capacity - 1;
  for (iree_host_size_t i = 0; i < family_count; ++i) {
    const loom_encoding_vtable_t* vtable =
        context->encodings.vtables.entries[i];
    iree_string_view_t name = loom_bstring_view(vtable->descriptor->name);
    uint32_t slot = loom_hash_string(name) & mask;
    while (entries[slot].family_id != LOOM_ENCODING_FAMILY_ID_INVALID) {
      slot = (slot + 1) & mask;
    }
    entries[slot].name = name;
    entries[slot].family_id = (loom_encoding_family_id_t)(i + 1);
  }

  context->encodings.names.entries = entries;
  context->encodings.names.capacity = capacity;
  context->encodings.names.count = (uint32_t)family_count;
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

static iree_status_t loom_context_build_parameterized_attr_name_table(
    loom_context_t* context) {
  uint32_t total_families = 0;
  for (uint8_t dialect_id = 0; dialect_id < LOOM_DIALECT_BUILTIN_COUNT_;
       ++dialect_id) {
    total_families += context->parameterized_attrs.dialects[dialect_id].count;
  }
  if (total_families == 0) return iree_ok_status();

  uint32_t capacity =
      iree_host_size_next_power_of_two((total_families * 4 + 2) / 3);
  if (capacity < 16) capacity = 16;
  loom_parameterized_attr_name_entry_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      context->allocator, capacity, sizeof(*entries), (void**)&entries));
  memset(entries, 0, (iree_host_size_t)capacity * sizeof(*entries));

  uint32_t mask = capacity - 1;
  uint32_t count = 0;
  for (uint8_t dialect_id = 0; dialect_id < LOOM_DIALECT_BUILTIN_COUNT_;
       ++dialect_id) {
    const loom_dialect_parameterized_attrs_t* dialect =
        &context->parameterized_attrs.dialects[dialect_id];
    for (uint8_t family_index = 0; family_index < dialect->count;
         ++family_index) {
      const loom_parameterized_attr_descriptor_t* descriptor =
          &dialect->entries[family_index];
      iree_string_view_t name = loom_bstring_view(descriptor->name);
      if (loom_context_lookup_encoding_family_by_name(context, name) !=
          LOOM_ENCODING_FAMILY_ID_INVALID) {
        iree_allocator_free(context->allocator, entries);
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "parameterized attribute family '%.*s' conflicts with a "
            "registered encoding family",
            (int)name.size, name.data);
      }
      uint32_t slot = loom_hash_string(name) & mask;
      while (entries[slot].descriptor != NULL) {
        if (iree_string_view_equal(entries[slot].name, name)) {
          iree_allocator_free(context->allocator, entries);
          return iree_make_status(
              IREE_STATUS_ALREADY_EXISTS,
              "parameterized attribute family '%.*s' is registered twice",
              (int)name.size, name.data);
        }
        slot = (slot + 1) & mask;
      }
      entries[slot].name = name;
      entries[slot].descriptor = descriptor;
      ++count;
    }
  }

  context->parameterized_attr_name_table.entries = entries;
  context->parameterized_attr_name_table.capacity = capacity;
  context->parameterized_attr_name_table.count = count;
  return iree_ok_status();
}

iree_status_t loom_context_finalize(loom_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_context_build_encoding_family_name_table(context));
  IREE_RETURN_IF_ERROR(
      loom_context_build_parameterized_attr_name_table(context));
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

const loom_parameterized_attr_descriptor_t*
loom_context_resolve_parameterized_attr(const loom_context_t* context,
                                        loom_parameterized_attr_kind_t kind) {
  uint8_t dialect_id = loom_parameterized_attr_dialect_id(kind);
  if (dialect_id >= LOOM_DIALECT_BUILTIN_COUNT_) return NULL;
  const loom_dialect_parameterized_attrs_t* dialect =
      &context->parameterized_attrs.dialects[dialect_id];
  uint8_t family_index = loom_parameterized_attr_dialect_index(kind);
  if (family_index >= dialect->count) return NULL;
  return &dialect->entries[family_index];
}

const loom_parameterized_attr_descriptor_t*
loom_context_lookup_parameterized_attr_by_name(const loom_context_t* context,
                                               iree_string_view_t name) {
  const loom_parameterized_attr_name_table_t* table =
      &context->parameterized_attr_name_table;
  if (table->capacity == 0) return NULL;
  uint32_t mask = table->capacity - 1;
  uint32_t slot = loom_hash_string(name) & mask;
  while (table->entries[slot].descriptor != NULL) {
    if (iree_string_view_equal(table->entries[slot].name, name)) {
      return table->entries[slot].descriptor;
    }
    slot = (slot + 1) & mask;
  }
  return NULL;
}

loom_encoding_family_id_t loom_context_lookup_encoding_family_by_name(
    const loom_context_t* context, iree_string_view_t name) {
  const loom_encoding_family_name_table_t* table = &context->encodings.names;
  if (table->capacity == 0) return LOOM_ENCODING_FAMILY_ID_INVALID;
  uint32_t mask = table->capacity - 1;
  uint32_t slot = loom_hash_string(name) & mask;
  while (table->entries[slot].family_id != LOOM_ENCODING_FAMILY_ID_INVALID) {
    if (iree_string_view_equal(table->entries[slot].name, name)) {
      return table->entries[slot].family_id;
    }
    slot = (slot + 1) & mask;
  }
  return LOOM_ENCODING_FAMILY_ID_INVALID;
}

const loom_encoding_vtable_t* loom_context_resolve_encoding_vtable(
    const loom_context_t* context, loom_encoding_family_id_t family_id) {
  if (family_id == LOOM_ENCODING_FAMILY_ID_INVALID) return NULL;
  return context->encodings.vtables.entries[family_id - 1];
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
