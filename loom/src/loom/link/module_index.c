// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/module_index.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/link/symbol_policy.h"
#include "loom/ops/op_defs.h"

typedef struct loom_link_module_index_name_map_entry_t {
  // Borrowed global symbol name.
  iree_string_view_t name;
  // Selected global symbol ordinal, or INVALID_ORDINAL for empty slots.
  iree_host_size_t symbol_ordinal;
} loom_link_module_index_name_map_entry_t;

struct loom_link_module_index_t {
  // Context shared by all providers.
  loom_context_t* context;
  // Block pool used by text parsing and bytecode validation.
  iree_arena_block_pool_t* block_pool;
  // Host allocator for growable index arrays.
  iree_allocator_t allocator;
  // Arena for provider labels and bytecode index metadata.
  iree_arena_allocator_t arena;
  // Growable provider record array.
  loom_link_module_index_provider_t* providers;
  // Number of live provider records.
  iree_host_size_t provider_count;
  // Allocated provider record capacity.
  iree_host_size_t provider_capacity;
  // Growable module record array.
  loom_link_module_index_module_t* modules;
  // Number of live module records.
  iree_host_size_t module_count;
  // Allocated module record capacity.
  iree_host_size_t module_capacity;
  // Growable symbol record array.
  loom_link_module_index_symbol_t* symbols;
  // Number of live symbol records.
  iree_host_size_t symbol_count;
  // Allocated symbol record capacity.
  iree_host_size_t symbol_capacity;
  // Open-addressed map from global symbol name to selected symbol ordinal.
  loom_link_module_index_name_map_entry_t* global_symbols;
  // Number of occupied global_symbols entries.
  iree_host_size_t global_symbol_count;
  // Allocated global_symbols entry capacity.
  iree_host_size_t global_symbol_capacity;
};

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static uint32_t loom_link_hash_bytes(const void* data,
                                     iree_host_size_t length) {
  uint32_t hash = 2166136261u;
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < length; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

static uint32_t loom_link_hash_string(iree_string_view_t string) {
  return loom_link_hash_bytes(string.data, string.size);
}

static iree_string_view_t loom_link_normalize_symbol_name(
    iree_string_view_t name) {
  if (iree_string_view_starts_with_char(name, '@')) {
    return iree_string_view_remove_prefix(name, 1);
  }
  return name;
}

static iree_status_t loom_link_index_copy_string(
    loom_link_module_index_t* index, iree_string_view_t source,
    iree_string_view_t* out_string) {
  *out_string = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) {
    return iree_ok_status();
  }
  char* target = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&index->arena, source.size, (void**)&target));
  memcpy(target, source.data, source.size);
  *out_string = iree_make_string_view(target, source.size);
  return iree_ok_status();
}

static iree_string_view_t loom_link_materialized_module_name(
    const loom_module_t* module) {
  if (!module || module->name_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[module->name_id];
}

static iree_status_t loom_link_index_validate_materialized_module(
    const loom_module_t* module) {
  if (module->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module name id %u is out of range",
                            (unsigned)module->name_id);
  }
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (symbol->name_id >= module->strings.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source symbol %" PRIhsz
                              " has out-of-range name id %u",
                              i, (unsigned)symbol->name_id);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_link_index_reserve_providers(
    loom_link_module_index_t* index, iree_host_size_t count) {
  if (count <= index->provider_capacity) {
    return iree_ok_status();
  }
  return iree_allocator_grow_array(
      index->allocator, count, sizeof(*index->providers),
      &index->provider_capacity, (void**)&index->providers);
}

static iree_status_t loom_link_index_reserve_modules(
    loom_link_module_index_t* index, iree_host_size_t count) {
  if (count <= index->module_capacity) {
    return iree_ok_status();
  }
  return iree_allocator_grow_array(
      index->allocator, count, sizeof(*index->modules), &index->module_capacity,
      (void**)&index->modules);
}

static iree_status_t loom_link_index_reserve_symbols(
    loom_link_module_index_t* index, iree_host_size_t count) {
  if (count <= index->symbol_capacity) {
    return iree_ok_status();
  }
  return iree_allocator_grow_array(
      index->allocator, count, sizeof(*index->symbols), &index->symbol_capacity,
      (void**)&index->symbols);
}

static void loom_link_index_name_map_initialize(
    loom_link_module_index_name_map_entry_t* entries,
    iree_host_size_t capacity) {
  for (iree_host_size_t i = 0; i < capacity; ++i) {
    entries[i].name = iree_string_view_empty();
    entries[i].symbol_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  }
}

static iree_host_size_t loom_link_index_name_map_slot(
    const loom_link_module_index_name_map_entry_t* entries,
    iree_host_size_t capacity, iree_string_view_t name) {
  iree_host_size_t mask = capacity - 1;
  iree_host_size_t slot = loom_link_hash_string(name) & mask;
  while (entries[slot].symbol_ordinal !=
         LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    if (iree_string_view_equal(entries[slot].name, name)) {
      return slot;
    }
    slot = (slot + 1) & mask;
  }
  return slot;
}

static void loom_link_index_name_map_insert_selected(
    loom_link_module_index_name_map_entry_t* entries, iree_host_size_t capacity,
    iree_string_view_t name, iree_host_size_t symbol_ordinal) {
  iree_host_size_t slot =
      loom_link_index_name_map_slot(entries, capacity, name);
  entries[slot].name = name;
  entries[slot].symbol_ordinal = symbol_ordinal;
}

static iree_status_t loom_link_index_reserve_global_symbols(
    loom_link_module_index_t* index, iree_host_size_t count) {
  if (index->global_symbol_capacity > 0 &&
      count * 4 < index->global_symbol_capacity * 3) {
    return iree_ok_status();
  }

  iree_host_size_t old_capacity = index->global_symbol_capacity;
  loom_link_module_index_name_map_entry_t* old_entries = index->global_symbols;

  iree_host_size_t new_capacity = old_capacity == 0 ? 16 : old_capacity * 2;
  while (count * 4 >= new_capacity * 3) {
    new_capacity *= 2;
  }
  loom_link_module_index_name_map_entry_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(index->allocator, new_capacity,
                                  sizeof(*new_entries), (void**)&new_entries));
  loom_link_index_name_map_initialize(new_entries, new_capacity);

  for (iree_host_size_t i = 0; i < old_capacity; ++i) {
    if (old_entries[i].symbol_ordinal ==
        LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
      continue;
    }
    loom_link_index_name_map_insert_selected(new_entries, new_capacity,
                                             old_entries[i].name,
                                             old_entries[i].symbol_ordinal);
  }

  index->global_symbols = new_entries;
  index->global_symbol_capacity = new_capacity;
  iree_allocator_free(index->allocator, old_entries);
  return iree_ok_status();
}

static bool loom_link_index_symbol_precedes(
    const loom_link_module_index_t* index, iree_host_size_t lhs_ordinal,
    iree_host_size_t rhs_ordinal) {
  const loom_link_module_index_symbol_t* lhs = &index->symbols[lhs_ordinal];
  const loom_link_module_index_symbol_t* rhs = &index->symbols[rhs_ordinal];
  const loom_link_module_index_provider_t* lhs_provider =
      &index->providers[lhs->provider_ordinal];
  const loom_link_module_index_provider_t* rhs_provider =
      &index->providers[rhs->provider_ordinal];
  if (lhs_provider->role != rhs_provider->role) {
    return lhs_provider->role < rhs_provider->role;
  }
  if (lhs->provider_ordinal != rhs->provider_ordinal) {
    return lhs->provider_ordinal < rhs->provider_ordinal;
  }
  if (lhs->provider_module_ordinal != rhs->provider_module_ordinal) {
    return lhs->provider_module_ordinal < rhs->provider_module_ordinal;
  }
  return lhs->module_symbol_ordinal < rhs->module_symbol_ordinal;
}

static void loom_link_index_insert_global_symbol(
    loom_link_module_index_t* index, iree_host_size_t symbol_ordinal) {
  loom_link_module_index_symbol_t* symbol = &index->symbols[symbol_ordinal];
  iree_host_size_t slot = loom_link_index_name_map_slot(
      index->global_symbols, index->global_symbol_capacity, symbol->name);
  if (index->global_symbols[slot].symbol_ordinal ==
      LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    index->global_symbols[slot].name = symbol->name;
    index->global_symbols[slot].symbol_ordinal = symbol_ordinal;
    ++index->global_symbol_count;
    return;
  }

  iree_host_size_t selected_ordinal =
      index->global_symbols[slot].symbol_ordinal;
  if (loom_link_index_symbol_precedes(index, symbol_ordinal,
                                      selected_ordinal)) {
    symbol->next_global_duplicate_ordinal = selected_ordinal;
    index->global_symbols[slot].symbol_ordinal = symbol_ordinal;
    return;
  }

  loom_link_module_index_symbol_t* selected = &index->symbols[selected_ordinal];
  symbol->next_global_duplicate_ordinal =
      selected->next_global_duplicate_ordinal;
  selected->next_global_duplicate_ordinal = symbol_ordinal;
}

static iree_status_t loom_link_index_append_provider(
    loom_link_module_index_t* index, loom_link_provider_kind_t kind,
    iree_string_view_t default_name,
    const loom_link_module_index_add_options_t* options,
    iree_host_size_t* out_provider_ordinal) {
  iree_string_view_t provider_name = default_name;
  loom_link_provider_role_t role = LOOM_LINK_PROVIDER_ROLE_INPUT;
  if (options) {
    if (!iree_string_view_is_empty(options->provider_name)) {
      provider_name = options->provider_name;
    }
    role = options->role;
  }
  if (role > LOOM_LINK_PROVIDER_ROLE_LIBRARY) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown link provider role %u", (unsigned)role);
  }

  iree_host_size_t provider_ordinal = index->provider_count;
  IREE_RETURN_IF_ERROR(
      loom_link_index_reserve_providers(index, provider_ordinal + 1));

  loom_link_module_index_provider_t* provider =
      &index->providers[provider_ordinal];
  *provider = (loom_link_module_index_provider_t){
      .ordinal = provider_ordinal,
      .kind = kind,
      .role = role,
      .module_start_ordinal = index->module_count,
      .module_count = 0,
  };
  IREE_RETURN_IF_ERROR(
      loom_link_index_copy_string(index, provider_name, &provider->name));
  index->provider_count = provider_ordinal + 1;
  if (out_provider_ordinal) {
    *out_provider_ordinal = provider_ordinal;
  }
  return iree_ok_status();
}

static iree_status_t loom_link_index_append_module(
    loom_link_module_index_t* index, iree_host_size_t provider_ordinal,
    iree_host_size_t provider_module_ordinal, iree_string_view_t name,
    const loom_module_t* materialized_module, bool owns_materialized_module,
    loom_link_module_index_module_t** out_module) {
  if (provider_ordinal >= index->provider_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "provider ordinal %" PRIhsz " is out of range",
                            provider_ordinal);
  }
  iree_host_size_t module_ordinal = index->module_count;
  IREE_RETURN_IF_ERROR(
      loom_link_index_reserve_modules(index, module_ordinal + 1));

  loom_link_module_index_module_t* module = &index->modules[module_ordinal];
  *module = (loom_link_module_index_module_t){
      .ordinal = module_ordinal,
      .provider_ordinal = provider_ordinal,
      .provider_module_ordinal = provider_module_ordinal,
      .name = name,
      .materialized_module = materialized_module,
      .owns_materialized_module = owns_materialized_module,
      .symbol_start_ordinal = index->symbol_count,
      .symbol_count = 0,
  };
  ++index->providers[provider_ordinal].module_count;
  index->module_count = module_ordinal + 1;
  *out_module = module;
  return iree_ok_status();
}

static iree_status_t loom_link_index_append_symbol(
    loom_link_module_index_t* index, loom_link_module_index_module_t* module,
    iree_string_view_t name, loom_symbol_kind_t kind, loom_symbol_flags_t flags,
    loom_link_symbol_identity_t identity, loom_link_symbol_flags_t link_flags,
    iree_string_view_t provider_contract) {
  iree_host_size_t symbol_ordinal = index->symbol_count;
  IREE_RETURN_IF_ERROR(
      loom_link_index_reserve_symbols(index, symbol_ordinal + 1));
  if (identity == LOOM_LINK_SYMBOL_IDENTITY_GLOBAL) {
    IREE_RETURN_IF_ERROR(loom_link_index_reserve_global_symbols(
        index, index->global_symbol_count + 1));
  }

  loom_link_module_index_symbol_t* symbol = &index->symbols[symbol_ordinal];
  *symbol = (loom_link_module_index_symbol_t){
      .ordinal = symbol_ordinal,
      .provider_ordinal = module->provider_ordinal,
      .module_ordinal = module->ordinal,
      .provider_module_ordinal = module->provider_module_ordinal,
      .module_symbol_ordinal = module->symbol_count,
      .name = name,
      .kind = kind,
      .ir_flags = flags,
      .identity = identity,
      .flags = link_flags,
      .provider_contract = provider_contract,
      .next_global_duplicate_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
  };

  ++module->symbol_count;
  index->symbol_count = symbol_ordinal + 1;
  if (identity == LOOM_LINK_SYMBOL_IDENTITY_GLOBAL) {
    loom_link_index_insert_global_symbol(index, symbol_ordinal);
  }
  return iree_ok_status();
}

static iree_string_view_t loom_link_materialized_symbol_provider_contract(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  if (symbol->kind != LOOM_SYMBOL_FUNC_TEMPLATE &&
      symbol->kind != LOOM_SYMBOL_FUNC_UKERNEL) {
    return iree_string_view_empty();
  }
  loom_func_like_t func = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(func)) {
    return iree_string_view_empty();
  }
  loom_string_id_t contract_id = loom_func_like_implements(func);
  if (contract_id == LOOM_STRING_ID_INVALID ||
      contract_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[contract_id];
}

//===----------------------------------------------------------------------===//
// Symbol classification
//===----------------------------------------------------------------------===//

static loom_link_symbol_flags_t loom_link_check_symbol_flags(
    iree_string_view_t defining_op_name) {
  if (iree_string_view_equal(defining_op_name, IREE_SV("check.case"))) {
    return LOOM_LINK_SYMBOL_FLAG_CHECK_CASE;
  }
  if (iree_string_view_equal(defining_op_name, IREE_SV("check.benchmark"))) {
    return LOOM_LINK_SYMBOL_FLAG_CHECK_BENCHMARK;
  }
  return 0;
}

static bool loom_link_materialized_symbol_has_visibility_attr(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  if (!symbol->defining_op) return false;
  const loom_op_vtable_t* vtable = loom_op_vtable(module, symbol->defining_op);
  if (!vtable || !vtable->attr_descriptors) return false;
  const loom_attribute_t* attrs = loom_op_const_attrs(symbol->defining_op);
  for (uint8_t i = 0; i < vtable->attribute_count; ++i) {
    const loom_attr_descriptor_t* descriptor = &vtable->attr_descriptors[i];
    if (!iree_string_view_equal(loom_attr_descriptor_name(descriptor),
                                IREE_SV("visibility"))) {
      continue;
    }
    if (descriptor->attr_kind != LOOM_ATTR_ENUM ||
        i >= symbol->defining_op->attribute_count) {
      return false;
    }
    return loom_attr_as_enum(attrs[i]) != 0;
  }
  return false;
}

static loom_link_symbol_flags_t loom_link_materialized_symbol_flags(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  loom_link_symbol_flags_t flags = 0;
  if (iree_any_bit_set(symbol->flags, LOOM_SYMBOL_FLAG_PUBLIC)) {
    flags |= LOOM_LINK_SYMBOL_FLAG_PUBLIC;
  }
  if (loom_link_materialized_symbol_has_visibility_attr(module, symbol)) {
    flags |= LOOM_LINK_SYMBOL_FLAG_PUBLIC;
  }
  if (loom_link_symbol_is_declaration(symbol)) {
    flags |= LOOM_LINK_SYMBOL_FLAG_DECLARATION;
  }
  if (loom_link_symbol_is_concrete_definition(symbol)) {
    flags |= LOOM_LINK_SYMBOL_FLAG_HAS_BODY;
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_CONFIG)) {
    flags |= LOOM_LINK_SYMBOL_FLAG_CONFIG;
  }
  if (symbol->defining_op) {
    flags |= loom_link_check_symbol_flags(
        loom_op_vtable_name(loom_op_vtable(module, symbol->defining_op)));
  }

  loom_func_like_t func = loom_func_like_cast(module, symbol->defining_op);
  const bool has_import =
      loom_func_like_isa(func) &&
      (loom_func_like_import_module(func) != LOOM_STRING_ID_INVALID ||
       loom_func_like_import_symbol(func) != LOOM_STRING_ID_INVALID);
  if (has_import) {
    flags |= LOOM_LINK_SYMBOL_FLAG_IMPORT;
  }
  if (iree_any_bit_set(flags, LOOM_LINK_SYMBOL_FLAG_PUBLIC) && !has_import) {
    flags |= LOOM_LINK_SYMBOL_FLAG_EXPORT;
  }
  if (loom_func_like_isa(func) &&
      loom_func_like_export_symbol(func) != LOOM_STRING_ID_INVALID) {
    flags |= LOOM_LINK_SYMBOL_FLAG_EXPORT;
  }
  return flags;
}

static loom_symbol_kind_t loom_link_bytecode_symbol_kind(
    loom_bytecode_symbol_kind_t kind) {
  switch (kind) {
    case LOOM_BYTECODE_SYMBOL_FUNC_DEF:
      return LOOM_SYMBOL_FUNC_DEF;
    case LOOM_BYTECODE_SYMBOL_FUNC_DECL:
      return LOOM_SYMBOL_FUNC_DECL;
    case LOOM_BYTECODE_SYMBOL_FUNC_TEMPLATE:
      return LOOM_SYMBOL_FUNC_TEMPLATE;
    case LOOM_BYTECODE_SYMBOL_FUNC_UKERNEL:
      return LOOM_SYMBOL_FUNC_UKERNEL;
    case LOOM_BYTECODE_SYMBOL_GLOBAL:
      return LOOM_SYMBOL_GLOBAL;
    case LOOM_BYTECODE_SYMBOL_EXECUTABLE:
      return LOOM_SYMBOL_EXECUTABLE;
    case LOOM_BYTECODE_SYMBOL_RECORD:
      return LOOM_SYMBOL_RECORD;
    default:
      return LOOM_SYMBOL_NONE;
  }
}

static bool loom_link_bytecode_symbol_is_config(
    const loom_bytecode_symbol_metadata_t* symbol) {
  return iree_string_view_equal(symbol->defining_op_name,
                                IREE_SV("config.decl")) ||
         iree_string_view_equal(symbol->defining_op_name,
                                IREE_SV("config.def"));
}

static loom_link_symbol_flags_t loom_link_bytecode_symbol_flags(
    const loom_bytecode_symbol_metadata_t* symbol) {
  loom_link_symbol_flags_t flags = 0;
  const bool is_public =
      symbol->visibility == LOOM_BYTECODE_SYMBOL_VISIBILITY_PUBLIC ||
      iree_any_bit_set(symbol->flags, LOOM_BYTECODE_SYMBOL_FLAG_PUBLIC);
  const bool is_import =
      iree_any_bit_set(symbol->flags, LOOM_BYTECODE_SYMBOL_FLAG_IMPORT);
  if (is_public) {
    flags |= LOOM_LINK_SYMBOL_FLAG_PUBLIC;
  }
  if (is_import) {
    flags |= LOOM_LINK_SYMBOL_FLAG_IMPORT;
  }
  if (is_public && !is_import) {
    flags |= LOOM_LINK_SYMBOL_FLAG_EXPORT;
  }
  if (symbol->kind == LOOM_BYTECODE_SYMBOL_FUNC_DECL || is_import ||
      iree_string_view_equal(symbol->defining_op_name,
                             IREE_SV("config.decl"))) {
    flags |= LOOM_LINK_SYMBOL_FLAG_DECLARATION;
  }
  if (symbol->has_body) {
    flags |= LOOM_LINK_SYMBOL_FLAG_HAS_BODY;
  }
  if (loom_link_bytecode_symbol_is_config(symbol)) {
    flags |= LOOM_LINK_SYMBOL_FLAG_CONFIG;
  }
  flags |= loom_link_check_symbol_flags(symbol->defining_op_name);
  return flags;
}

static loom_link_symbol_identity_t loom_link_bytecode_symbol_identity(
    const loom_bytecode_symbol_metadata_t* symbol,
    loom_link_symbol_flags_t flags) {
  if (iree_any_bit_set(flags, LOOM_LINK_SYMBOL_FLAG_PUBLIC |
                                  LOOM_LINK_SYMBOL_FLAG_IMPORT |
                                  LOOM_LINK_SYMBOL_FLAG_DECLARATION |
                                  LOOM_LINK_SYMBOL_FLAG_CONFIG)) {
    return LOOM_LINK_SYMBOL_IDENTITY_GLOBAL;
  }
  return LOOM_LINK_SYMBOL_IDENTITY_PRIVATE;
}

//===----------------------------------------------------------------------===//
// Provider indexing
//===----------------------------------------------------------------------===//

static iree_status_t loom_link_index_module_materialized_symbols(
    loom_link_module_index_t* index, loom_link_module_index_module_t* module,
    const loom_module_t* source_module) {
  for (iree_host_size_t i = 0; i < source_module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &source_module->symbols.entries[i];
    if (symbol->name_id >= source_module->strings.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source symbol %" PRIhsz
                              " has out-of-range name id %u",
                              i, (unsigned)symbol->name_id);
    }
    iree_string_view_t name = source_module->strings.entries[symbol->name_id];
    loom_link_symbol_identity_t identity =
        loom_link_symbol_has_global_identity(source_module, symbol)
            ? LOOM_LINK_SYMBOL_IDENTITY_GLOBAL
            : LOOM_LINK_SYMBOL_IDENTITY_PRIVATE;
    iree_string_view_t provider_contract =
        loom_link_materialized_symbol_provider_contract(source_module, symbol);
    IREE_RETURN_IF_ERROR(loom_link_index_append_symbol(
        index, module, name, symbol->kind, symbol->flags, identity,
        loom_link_materialized_symbol_flags(source_module, symbol),
        provider_contract));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_index_module_bytecode_symbols(
    loom_link_module_index_t* index, loom_link_module_index_module_t* module,
    const loom_bytecode_module_metadata_t* bytecode_module) {
  for (iree_host_size_t i = 0; i < bytecode_module->symbol_count; ++i) {
    const loom_bytecode_symbol_metadata_t* symbol =
        &bytecode_module->symbols[i];
    loom_link_symbol_flags_t flags = loom_link_bytecode_symbol_flags(symbol);
    IREE_RETURN_IF_ERROR(loom_link_index_append_symbol(
        index, module, symbol->name,
        loom_link_bytecode_symbol_kind(symbol->kind),
        /*flags=*/0, loom_link_bytecode_symbol_identity(symbol, flags), flags,
        symbol->implements_op_name));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t loom_link_module_index_create(
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_link_module_index_t** out_index) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_index);
  *out_index = NULL;

  loom_link_module_index_t* index = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*index), (void**)&index));
  memset(index, 0, sizeof(*index));
  index->context = context;
  index->block_pool = block_pool;
  index->allocator = allocator;
  iree_arena_initialize(block_pool, &index->arena);
  *out_index = index;
  return iree_ok_status();
}

void loom_link_module_index_free(loom_link_module_index_t* index) {
  if (!index) return;
  for (iree_host_size_t i = 0; i < index->module_count; ++i) {
    if (index->modules[i].owns_materialized_module) {
      loom_module_free((loom_module_t*)index->modules[i].materialized_module);
    }
  }
  iree_allocator_free(index->allocator, index->global_symbols);
  iree_allocator_free(index->allocator, index->symbols);
  iree_allocator_free(index->allocator, index->modules);
  iree_allocator_free(index->allocator, index->providers);
  iree_arena_deinitialize(&index->arena);
  iree_allocator_free(index->allocator, index);
}

iree_status_t loom_link_module_index_add_materialized(
    loom_link_module_index_t* index, const loom_module_t* module,
    const loom_link_module_index_add_options_t* options,
    iree_host_size_t* out_provider_ordinal) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(module);
  if (module->context != index->context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "materialized provider context mismatch");
  }
  IREE_RETURN_IF_ERROR(loom_link_index_validate_materialized_module(module));

  iree_host_size_t provider_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  IREE_RETURN_IF_ERROR(loom_link_index_append_provider(
      index, LOOM_LINK_PROVIDER_MATERIALIZED,
      loom_link_materialized_module_name(module), options, &provider_ordinal));

  loom_link_module_index_module_t* indexed_module = NULL;
  IREE_RETURN_IF_ERROR(loom_link_index_append_module(
      index, provider_ordinal, /*provider_module_ordinal=*/0,
      loom_link_materialized_module_name(module), module,
      /*owns_materialized_module=*/false, &indexed_module));
  IREE_RETURN_IF_ERROR(loom_link_index_module_materialized_symbols(
      index, indexed_module, module));
  if (out_provider_ordinal) {
    *out_provider_ordinal = provider_ordinal;
  }
  return iree_ok_status();
}

iree_status_t loom_link_module_index_add_bytecode(
    loom_link_module_index_t* index, iree_const_byte_span_t bytecode,
    iree_string_view_t filename,
    const loom_bytecode_read_options_t* read_options,
    const loom_link_module_index_add_options_t* options,
    iree_host_size_t* out_provider_ordinal) {
  IREE_ASSERT_ARGUMENT(index);

  loom_bytecode_read_result_t read_result = {0};
  loom_bytecode_file_metadata_t metadata = {0};
  IREE_RETURN_IF_ERROR(loom_bytecode_read_index(
      bytecode, filename, index->context, index->block_pool, &index->arena,
      read_options, &read_result, &metadata));
  if (read_result.error_count > 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bytecode provider '%.*s' has %u validation errors",
                            (int)filename.size, filename.data,
                            read_result.error_count);
  }

  iree_host_size_t provider_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  IREE_RETURN_IF_ERROR(
      loom_link_index_append_provider(index, LOOM_LINK_PROVIDER_BYTECODE,
                                      filename, options, &provider_ordinal));

  for (iree_host_size_t i = 0; i < metadata.module_count; ++i) {
    loom_link_module_index_module_t* indexed_module = NULL;
    IREE_RETURN_IF_ERROR(loom_link_index_append_module(
        index, provider_ordinal, i, metadata.modules[i].name,
        /*materialized_module=*/NULL, /*owns_materialized_module=*/false,
        &indexed_module));
    IREE_RETURN_IF_ERROR(loom_link_index_module_bytecode_symbols(
        index, indexed_module, &metadata.modules[i]));
  }

  if (out_provider_ordinal) {
    *out_provider_ordinal = provider_ordinal;
  }
  return iree_ok_status();
}

iree_status_t loom_link_module_index_add_text(
    loom_link_module_index_t* index, iree_string_view_t source,
    iree_string_view_t filename, const loom_text_parse_options_t* parse_options,
    const loom_link_module_index_add_options_t* options,
    iree_host_size_t* out_provider_ordinal) {
  IREE_ASSERT_ARGUMENT(index);

  loom_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(loom_text_parse(source, filename, index->context,
                                       index->block_pool, parse_options,
                                       &module));
  if (!module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "text provider '%.*s' did not parse into a module",
                            (int)filename.size, filename.data);
  }
  iree_status_t status = loom_link_index_validate_materialized_module(module);
  if (!iree_status_is_ok(status)) {
    loom_module_free(module);
    return status;
  }

  iree_host_size_t provider_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  status = loom_link_index_append_provider(
      index, LOOM_LINK_PROVIDER_TEXT, filename, options, &provider_ordinal);
  bool module_owned_by_index = false;
  if (iree_status_is_ok(status)) {
    loom_link_module_index_module_t* indexed_module = NULL;
    status = loom_link_index_append_module(
        index, provider_ordinal, /*provider_module_ordinal=*/0,
        loom_link_materialized_module_name(module), module,
        /*owns_materialized_module=*/true, &indexed_module);
    module_owned_by_index = iree_status_is_ok(status);
    if (iree_status_is_ok(status)) {
      status = loom_link_index_module_materialized_symbols(
          index, indexed_module, module);
    }
  }
  if (!iree_status_is_ok(status)) {
    if (!module_owned_by_index) {
      loom_module_free(module);
    }
    return status;
  }

  if (out_provider_ordinal) {
    *out_provider_ordinal = provider_ordinal;
  }
  return iree_ok_status();
}

iree_host_size_t loom_link_module_index_provider_count(
    const loom_link_module_index_t* index) {
  return index ? index->provider_count : 0;
}

const loom_link_module_index_provider_t* loom_link_module_index_provider_at(
    const loom_link_module_index_t* index, iree_host_size_t ordinal) {
  if (!index || ordinal >= index->provider_count) return NULL;
  return &index->providers[ordinal];
}

iree_host_size_t loom_link_module_index_module_count(
    const loom_link_module_index_t* index) {
  return index ? index->module_count : 0;
}

const loom_link_module_index_module_t* loom_link_module_index_module_at(
    const loom_link_module_index_t* index, iree_host_size_t ordinal) {
  if (!index || ordinal >= index->module_count) return NULL;
  return &index->modules[ordinal];
}

iree_host_size_t loom_link_module_index_symbol_count(
    const loom_link_module_index_t* index) {
  return index ? index->symbol_count : 0;
}

const loom_link_module_index_symbol_t* loom_link_module_index_symbol_at(
    const loom_link_module_index_t* index, iree_host_size_t ordinal) {
  if (!index || ordinal >= index->symbol_count) return NULL;
  return &index->symbols[ordinal];
}

const loom_link_module_index_provider_t* loom_link_module_index_symbol_provider(
    const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* symbol) {
  if (!index || !symbol || symbol->provider_ordinal >= index->provider_count) {
    return NULL;
  }
  return &index->providers[symbol->provider_ordinal];
}

const loom_link_module_index_module_t* loom_link_module_index_symbol_module(
    const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* symbol) {
  if (!index || !symbol || symbol->module_ordinal >= index->module_count) {
    return NULL;
  }
  return &index->modules[symbol->module_ordinal];
}

const loom_link_module_index_symbol_t* loom_link_module_index_lookup_global(
    const loom_link_module_index_t* index, iree_string_view_t name) {
  if (!index || index->global_symbol_capacity == 0) return NULL;
  name = loom_link_normalize_symbol_name(name);
  iree_host_size_t slot = loom_link_index_name_map_slot(
      index->global_symbols, index->global_symbol_capacity, name);
  iree_host_size_t symbol_ordinal = index->global_symbols[slot].symbol_ordinal;
  if (symbol_ordinal == LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    return NULL;
  }
  return &index->symbols[symbol_ordinal];
}

const loom_link_module_index_symbol_t*
loom_link_module_index_next_global_duplicate(
    const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* symbol) {
  if (!index || !symbol ||
      symbol->next_global_duplicate_ordinal ==
          LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL ||
      symbol->next_global_duplicate_ordinal >= index->symbol_count) {
    return NULL;
  }
  return &index->symbols[symbol->next_global_duplicate_ordinal];
}

const loom_link_module_index_symbol_t* loom_link_module_index_lookup_private(
    const loom_link_module_index_t* index,
    const loom_link_module_index_module_t* module, iree_string_view_t name) {
  if (!index || !module) return NULL;
  name = loom_link_normalize_symbol_name(name);
  for (iree_host_size_t i = 0; i < module->symbol_count; ++i) {
    const loom_link_module_index_symbol_t* symbol =
        &index->symbols[module->symbol_start_ordinal + i];
    if (symbol->identity != LOOM_LINK_SYMBOL_IDENTITY_PRIVATE) {
      continue;
    }
    if (iree_string_view_equal(symbol->name, name)) {
      return symbol;
    }
  }
  return NULL;
}

iree_status_t loom_link_module_index_duplicate_global_status(
    const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* selected,
    const loom_link_module_index_symbol_t* duplicate) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(selected);
  IREE_ASSERT_ARGUMENT(duplicate);
  const loom_link_module_index_provider_t* selected_provider =
      loom_link_module_index_symbol_provider(index, selected);
  const loom_link_module_index_provider_t* duplicate_provider =
      loom_link_module_index_symbol_provider(index, duplicate);
  const loom_link_module_index_module_t* selected_module =
      loom_link_module_index_symbol_module(index, selected);
  const loom_link_module_index_module_t* duplicate_module =
      loom_link_module_index_symbol_module(index, duplicate);
  if (!selected_provider || !duplicate_provider || !selected_module ||
      !duplicate_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate symbol records are stale");
  }
  return iree_make_status(
      IREE_STATUS_ALREADY_EXISTS,
      "global symbol '@%.*s' selected from provider '%.*s' module '%.*s' "
      "conflicts with provider '%.*s' module '%.*s'",
      (int)selected->name.size, selected->name.data,
      (int)selected_provider->name.size, selected_provider->name.data,
      (int)selected_module->name.size, selected_module->name.data,
      (int)duplicate_provider->name.size, duplicate_provider->name.data,
      (int)duplicate_module->name.size, duplicate_module->name.data);
}
