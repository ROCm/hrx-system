// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/module_index.h"

#include <string.h>

#include "loom/analysis/symbol_references.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/symbol_policy.h"
#include "loom/ops/module/ops.h"
#include "loom/ops/op_defs.h"

typedef struct loom_link_module_index_name_map_entry_t {
  // Borrowed symbol name.
  iree_string_view_t name;
  // First same-name symbol ordinal, or INVALID_ORDINAL for empty slots.
  iree_host_size_t first_symbol_ordinal;
  // Last same-name symbol ordinal, or INVALID_ORDINAL for empty slots.
  iree_host_size_t last_symbol_ordinal;
  // Selected global symbol ordinal, or INVALID_ORDINAL when none exists.
  iree_host_size_t selected_global_ordinal;
  // Dense template family ordinal for this global name, or invalid when this
  // name has not been used as a family identity.
  loom_link_template_family_ordinal_t template_family_ordinal;
} loom_link_module_index_name_map_entry_t;

struct loom_link_module_index_t {
  // Context shared by all providers.
  loom_context_t* context;
  // Block pool used by text parsing and bytecode validation.
  iree_arena_block_pool_t* block_pool;
  // Host allocator for growable index arrays.
  iree_allocator_t allocator;
  // Arena for provider labels and projected reference metadata.
  iree_arena_allocator_t arena;
  // Provider records in stable insertion order.
  struct {
    // Growable record storage.
    loom_link_module_index_provider_t* values;
    // Number of live records.
    iree_host_size_t count;
    // Allocated record capacity.
    iree_host_size_t capacity;
  } providers;
  // Module records in stable provider order.
  struct {
    // Growable record storage.
    loom_link_module_index_module_t* values;
    // Number of live records.
    iree_host_size_t count;
    // Allocated record capacity.
    iree_host_size_t capacity;
  } modules;
  // Symbol records in stable module order.
  struct {
    // Growable record storage.
    loom_link_module_index_symbol_t* values;
    // Number of live records.
    iree_host_size_t count;
    // Allocated record capacity.
    iree_host_size_t capacity;
  } symbols;
  // Total semantic facets across indexed symbols.
  iree_host_size_t facet_count;
  // Exported INPUT-provider symbols in stable provider order.
  struct {
    // Growable index-wide symbol ordinal storage.
    iree_host_size_t* values;
    // Number of indexed INPUT exports.
    iree_host_size_t count;
    // Allocated ordinal capacity.
    iree_host_size_t capacity;
  } input_exports;
  // Open-addressed map from every symbol name to its same-name group.
  struct {
    // Hash table storage.
    loom_link_module_index_name_map_entry_t* values;
    // Number of occupied entries.
    iree_host_size_t count;
    // Allocated entry capacity.
    iree_host_size_t capacity;
  } names;
  // Dense template-family identity records.
  struct {
    // Growable record storage.
    loom_link_module_index_template_family_t* values;
    // Number of live records.
    iree_host_size_t count;
    // Allocated record capacity.
    iree_host_size_t capacity;
  } template_families;
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
  if (count <= index->providers.capacity) {
    return iree_ok_status();
  }
  return iree_allocator_grow_array(
      index->allocator, count, sizeof(*index->providers.values),
      &index->providers.capacity, (void**)&index->providers.values);
}

static iree_status_t loom_link_index_reserve_modules(
    loom_link_module_index_t* index, iree_host_size_t count) {
  if (count <= index->modules.capacity) {
    return iree_ok_status();
  }
  return iree_allocator_grow_array(
      index->allocator, count, sizeof(*index->modules.values),
      &index->modules.capacity, (void**)&index->modules.values);
}

static iree_status_t loom_link_index_reserve_symbols(
    loom_link_module_index_t* index, iree_host_size_t count) {
  if (count <= index->symbols.capacity) {
    return iree_ok_status();
  }
  return iree_allocator_grow_array(
      index->allocator, count, sizeof(*index->symbols.values),
      &index->symbols.capacity, (void**)&index->symbols.values);
}

static iree_status_t loom_link_index_reserve_input_exports(
    loom_link_module_index_t* index, iree_host_size_t count) {
  if (count <= index->input_exports.capacity) {
    return iree_ok_status();
  }
  return iree_allocator_grow_array(
      index->allocator, count, sizeof(*index->input_exports.values),
      &index->input_exports.capacity, (void**)&index->input_exports.values);
}

static iree_status_t loom_link_index_reserve_template_families(
    loom_link_module_index_t* index, iree_host_size_t count) {
  if (count <= index->template_families.capacity) {
    return iree_ok_status();
  }
  return iree_allocator_grow_array(index->allocator, count,
                                   sizeof(*index->template_families.values),
                                   &index->template_families.capacity,
                                   (void**)&index->template_families.values);
}

static void loom_link_index_name_map_initialize(
    loom_link_module_index_name_map_entry_t* entries,
    iree_host_size_t capacity) {
  for (iree_host_size_t i = 0; i < capacity; ++i) {
    entries[i] = (loom_link_module_index_name_map_entry_t){
        .first_symbol_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
        .last_symbol_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
        .selected_global_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
        .template_family_ordinal = LOOM_LINK_TEMPLATE_FAMILY_ORDINAL_INVALID,
    };
  }
}

static iree_host_size_t loom_link_index_name_map_slot(
    const loom_link_module_index_name_map_entry_t* entries,
    iree_host_size_t capacity, iree_string_view_t name) {
  iree_host_size_t mask = capacity - 1;
  iree_host_size_t slot = loom_link_hash_string(name) & mask;
  while (entries[slot].first_symbol_ordinal !=
         LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    if (iree_string_view_equal(entries[slot].name, name)) {
      return slot;
    }
    slot = (slot + 1) & mask;
  }
  return slot;
}

static void loom_link_index_name_map_insert_entry(
    loom_link_module_index_name_map_entry_t* entries, iree_host_size_t capacity,
    loom_link_module_index_name_map_entry_t entry) {
  iree_host_size_t slot =
      loom_link_index_name_map_slot(entries, capacity, entry.name);
  entries[slot] = entry;
}

static bool loom_link_index_hash_map_has_capacity(iree_host_size_t capacity,
                                                  iree_host_size_t count) {
  return capacity != 0 && count < capacity - capacity / 4;
}

static iree_status_t loom_link_index_hash_map_grow_capacity(
    iree_host_size_t old_capacity, iree_host_size_t count,
    iree_host_size_t* out_capacity) {
  iree_host_size_t capacity = 16;
  if (old_capacity != 0 &&
      !iree_host_size_checked_mul(old_capacity, 2, &capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "link index hash capacity overflow");
  }
  while (!loom_link_index_hash_map_has_capacity(capacity, count)) {
    if (!iree_host_size_checked_mul(capacity, 2, &capacity)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "link index hash capacity overflow");
    }
  }
  *out_capacity = capacity;
  return iree_ok_status();
}

static iree_status_t loom_link_index_reserve_names(
    loom_link_module_index_t* index, iree_host_size_t count) {
  if (loom_link_index_hash_map_has_capacity(index->names.capacity, count)) {
    return iree_ok_status();
  }

  const iree_host_size_t old_capacity = index->names.capacity;
  loom_link_module_index_name_map_entry_t* old_entries = index->names.values;
  iree_host_size_t new_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_link_index_hash_map_grow_capacity(
      old_capacity, count, &new_capacity));
  loom_link_module_index_name_map_entry_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(index->allocator, new_capacity,
                                  sizeof(*new_entries), (void**)&new_entries));
  loom_link_index_name_map_initialize(new_entries, new_capacity);

  for (iree_host_size_t i = 0; i < old_capacity; ++i) {
    if (old_entries[i].first_symbol_ordinal ==
        LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
      continue;
    }
    loom_link_index_name_map_insert_entry(new_entries, new_capacity,
                                          old_entries[i]);
  }

  index->names.values = new_entries;
  index->names.capacity = new_capacity;
  iree_allocator_free(index->allocator, old_entries);
  return iree_ok_status();
}

static bool loom_link_index_symbol_precedes(
    const loom_link_module_index_t* index, iree_host_size_t lhs_ordinal,
    iree_host_size_t rhs_ordinal) {
  const loom_link_module_index_symbol_t* lhs =
      &index->symbols.values[lhs_ordinal];
  const loom_link_module_index_symbol_t* rhs =
      &index->symbols.values[rhs_ordinal];
  const loom_link_module_index_module_t* lhs_module =
      &index->modules.values[lhs->module_ordinal];
  const loom_link_module_index_module_t* rhs_module =
      &index->modules.values[rhs->module_ordinal];
  const loom_link_module_index_provider_t* lhs_provider =
      &index->providers.values[lhs_module->provider_ordinal];
  const loom_link_module_index_provider_t* rhs_provider =
      &index->providers.values[rhs_module->provider_ordinal];
  if (lhs_provider->role != rhs_provider->role) {
    return lhs_provider->role < rhs_provider->role;
  }
  if (lhs_module->provider_ordinal != rhs_module->provider_ordinal) {
    return lhs_module->provider_ordinal < rhs_module->provider_ordinal;
  }
  if (lhs_module->provider_module_ordinal !=
      rhs_module->provider_module_ordinal) {
    return lhs_module->provider_module_ordinal <
           rhs_module->provider_module_ordinal;
  }
  return lhs->module_symbol_ordinal < rhs->module_symbol_ordinal;
}

static iree_status_t loom_link_index_insert_symbol_name(
    loom_link_module_index_t* index, iree_host_size_t symbol_ordinal) {
  loom_link_module_index_symbol_t* symbol =
      &index->symbols.values[symbol_ordinal];
  iree_host_size_t slot = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  if (index->names.capacity != 0) {
    slot = loom_link_index_name_map_slot(index->names.values,
                                         index->names.capacity, symbol->name);
  }
  if (slot == LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL ||
      index->names.values[slot].first_symbol_ordinal ==
          LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    IREE_RETURN_IF_ERROR(
        loom_link_index_reserve_names(index, index->names.count + 1));
    slot = loom_link_index_name_map_slot(index->names.values,
                                         index->names.capacity, symbol->name);
  }

  loom_link_module_index_name_map_entry_t* entry = &index->names.values[slot];
  if (entry->first_symbol_ordinal == LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    *entry = (loom_link_module_index_name_map_entry_t){
        .name = symbol->name,
        .first_symbol_ordinal = symbol_ordinal,
        .last_symbol_ordinal = symbol_ordinal,
        .selected_global_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
        .template_family_ordinal = LOOM_LINK_TEMPLATE_FAMILY_ORDINAL_INVALID,
    };
    ++index->names.count;
  } else {
    index->symbols.values[entry->last_symbol_ordinal].next.same_name_ordinal =
        symbol_ordinal;
    entry->last_symbol_ordinal = symbol_ordinal;
  }

  if (symbol->identity == LOOM_LINK_SYMBOL_IDENTITY_GLOBAL &&
      (entry->selected_global_ordinal ==
           LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL ||
       loom_link_index_symbol_precedes(index, symbol_ordinal,
                                       entry->selected_global_ordinal))) {
    entry->selected_global_ordinal = symbol_ordinal;
  }
  return iree_ok_status();
}

static iree_status_t loom_link_index_resolve_template_family(
    loom_link_module_index_t* index, iree_host_size_t family_symbol_ordinal,
    loom_link_template_family_ordinal_t* out_ordinal) {
  if (family_symbol_ordinal >= index->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "template family symbol ordinal is out of range");
  }

  loom_link_module_index_symbol_t* family_symbol =
      &index->symbols.values[family_symbol_ordinal];
  loom_link_template_family_ordinal_t* identity_ordinal =
      &family_symbol->template_family_ordinal;
  iree_host_size_t identity_symbol_ordinal = family_symbol_ordinal;
  if (family_symbol->identity == LOOM_LINK_SYMBOL_IDENTITY_GLOBAL) {
    IREE_ASSERT(index->names.capacity != 0);
    const iree_host_size_t slot = loom_link_index_name_map_slot(
        index->names.values, index->names.capacity, family_symbol->name);
    loom_link_module_index_name_map_entry_t* name_entry =
        &index->names.values[slot];
    IREE_ASSERT(name_entry->first_symbol_ordinal !=
                LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL);
    identity_ordinal = &name_entry->template_family_ordinal;
    identity_symbol_ordinal = name_entry->selected_global_ordinal;
    IREE_ASSERT(identity_symbol_ordinal !=
                LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL);
  }

  if (*identity_ordinal != LOOM_LINK_TEMPLATE_FAMILY_ORDINAL_INVALID) {
    family_symbol->template_family_ordinal = *identity_ordinal;
    *out_ordinal = *identity_ordinal;
    return iree_ok_status();
  }
  if (index->template_families.count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "link index exceeds %u template families",
                            (unsigned)(UINT32_MAX - 1));
  }

  const iree_host_size_t count = index->template_families.count + 1;
  IREE_RETURN_IF_ERROR(loom_link_index_reserve_template_families(index, count));
  const loom_link_template_family_ordinal_t ordinal =
      (loom_link_template_family_ordinal_t)index->template_families.count;
  index->template_families.values[ordinal] =
      (loom_link_module_index_template_family_t){
          .ordinal = ordinal,
          .identity_symbol_ordinal = identity_symbol_ordinal,
          .name = family_symbol->name,
          .providers =
              {
                  .first_symbol_ordinal =
                      LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
                  .last_symbol_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
              },
      };
  index->template_families.count = count;
  *identity_ordinal = ordinal;
  family_symbol->template_family_ordinal = ordinal;
  *out_ordinal = ordinal;
  return iree_ok_status();
}

static void loom_link_index_append_template_provider(
    loom_link_module_index_t* index,
    loom_link_template_family_ordinal_t ordinal,
    iree_host_size_t symbol_ordinal) {
  if (ordinal == LOOM_LINK_TEMPLATE_FAMILY_ORDINAL_INVALID) {
    return;
  }
  loom_link_module_index_template_family_t* family =
      &index->template_families.values[ordinal];
  if (family->providers.first_symbol_ordinal ==
      LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    family->providers.first_symbol_ordinal = symbol_ordinal;
  } else {
    index->symbols.values[family->providers.last_symbol_ordinal]
        .next.template_provider_ordinal = symbol_ordinal;
  }
  family->providers.last_symbol_ordinal = symbol_ordinal;
  ++family->providers.count;
}

static iree_status_t loom_link_index_append_provider(
    loom_link_module_index_t* index, loom_link_provider_kind_t kind,
    iree_string_view_t default_name,
    const loom_link_module_index_add_options_t* options,
    loom_link_module_index_provider_t** out_provider) {
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

  const iree_host_size_t provider_ordinal = index->providers.count;
  iree_host_size_t provider_count = 0;
  if (!iree_host_size_checked_add(provider_ordinal, 1, &provider_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "link provider count overflow");
  }
  IREE_RETURN_IF_ERROR(
      loom_link_index_reserve_providers(index, provider_count));

  loom_link_module_index_provider_t* provider =
      &index->providers.values[provider_ordinal];
  *provider = (loom_link_module_index_provider_t){
      .ordinal = provider_ordinal,
      .kind = kind,
      .role = role,
      .module_start_ordinal = index->modules.count,
      .module_count = 0,
  };
  IREE_RETURN_IF_ERROR(
      loom_link_index_copy_string(index, provider_name, &provider->name));
  index->providers.count = provider_count;
  *out_provider = provider;
  return iree_ok_status();
}

static iree_status_t loom_link_index_append_module(
    loom_link_module_index_t* index,
    loom_link_module_index_provider_t* provider,
    iree_host_size_t provider_module_ordinal, iree_string_view_t name,
    const loom_module_t* materialized_module, bool owns_materialized_module,
    loom_link_module_index_module_t** out_module) {
  const iree_host_size_t module_ordinal = index->modules.count;
  iree_host_size_t module_count = 0;
  if (!iree_host_size_checked_add(module_ordinal, 1, &module_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "link module count overflow");
  }
  IREE_RETURN_IF_ERROR(loom_link_index_reserve_modules(index, module_count));

  loom_link_module_index_module_t* module =
      &index->modules.values[module_ordinal];
  *module = (loom_link_module_index_module_t){
      .ordinal = module_ordinal,
      .provider_ordinal = provider->ordinal,
      .provider_module_ordinal = provider_module_ordinal,
      .name = name,
      .materialized_module = materialized_module,
      .owns_materialized_module = owns_materialized_module,
      .symbol_start_ordinal = index->symbols.count,
      .symbol_count = 0,
  };
  ++provider->module_count;
  index->modules.count = module_count;
  *out_module = module;
  return iree_ok_status();
}

static iree_status_t loom_link_index_append_symbol(
    loom_link_module_index_t* index, loom_link_module_index_module_t* module,
    iree_string_view_t name, loom_symbol_kind_t kind,
    loom_link_symbol_identity_t identity, loom_link_symbol_flags_t flags,
    loom_link_symbol_facet_schema_t facet_schema) {
  const iree_host_size_t symbol_ordinal = index->symbols.count;
  iree_host_size_t symbol_count = 0;
  if (!iree_host_size_checked_add(symbol_ordinal, 1, &symbol_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "link symbol count overflow");
  }
  IREE_RETURN_IF_ERROR(loom_link_index_reserve_symbols(index, symbol_count));
  const loom_link_module_index_provider_t* provider =
      &index->providers.values[module->provider_ordinal];
  const bool is_input_export =
      provider->role == LOOM_LINK_PROVIDER_ROLE_INPUT &&
      iree_any_bit_set(flags, LOOM_LINK_SYMBOL_FLAG_EXPORT);
  iree_host_size_t input_export_count = index->input_exports.count;
  if (is_input_export) {
    if (!iree_host_size_checked_add(input_export_count, 1,
                                    &input_export_count)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "link input export count overflow");
    }
    IREE_RETURN_IF_ERROR(
        loom_link_index_reserve_input_exports(index, input_export_count));
  }
  iree_host_size_t facet_count = 0;
  if (!iree_host_size_checked_add(index->facet_count, facet_schema.facet_count,
                                  &facet_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "link symbol facet count overflow");
  }
  loom_link_module_index_symbol_t* symbol =
      &index->symbols.values[symbol_ordinal];
  *symbol = (loom_link_module_index_symbol_t){
      .ordinal = symbol_ordinal,
      .module_ordinal = module->ordinal,
      .module_symbol_ordinal = module->symbol_count,
      .name = name,
      .kind = kind,
      .template_family_ordinal = LOOM_LINK_TEMPLATE_FAMILY_ORDINAL_INVALID,
      .identity = identity,
      .flags = flags,
      .facets =
          {
              .schema = facet_schema,
              .start_ordinal = index->facet_count,
          },
      .next =
          {
              .same_name_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
              .template_provider_ordinal =
                  LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
          },
  };

  ++module->symbol_count;
  index->symbols.count = symbol_count;
  index->facet_count = facet_count;
  IREE_RETURN_IF_ERROR(
      loom_link_index_insert_symbol_name(index, symbol_ordinal));
  if (is_input_export) {
    index->input_exports.values[index->input_exports.count++] = symbol_ordinal;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Symbol classification
//===----------------------------------------------------------------------===//

// Extracts materialized/text structural roles before applying the shared
// linker facet classifier. No linker metadata is stored on operation
// definitions or reconstructed by walking symbol bodies.
static loom_link_symbol_facet_schema_t
loom_link_classify_materialized_symbol_facets(const loom_op_t* defining_op,
                                              const loom_op_vtable_t* vtable) {
  if (!defining_op || !vtable || !vtable->symbol_def) {
    return loom_link_symbol_facet_schema_classify(
        /*interfaces=*/0, /*root_region_count=*/0,
        /*body_region_index_plus_one=*/0,
        /*kernel_configuration_region_index_plus_one=*/0);
  }
  uint8_t body_region_index_plus_one = 0;
  if (vtable->func_like &&
      vtable->func_like->body_region_index != LOOM_REGION_INDEX_NONE) {
    body_region_index_plus_one =
        (uint8_t)(vtable->func_like->body_region_index + 1);
  }
  return loom_link_symbol_facet_schema_classify(
      vtable->symbol_def->interfaces, defining_op->region_count,
      body_region_index_plus_one,
      vtable->symbol_def->kernel_workload_region_index_plus_one);
}

// Extracts the equivalent roles from validated bytecode metadata. Keeping the
// representation adapters adjacent makes materialized, text, and bytecode
// providers share one semantic policy.
static loom_link_symbol_facet_schema_t
loom_link_classify_bytecode_symbol_facets(
    const loom_bytecode_symbol_metadata_t* symbol) {
  return loom_link_symbol_facet_schema_classify(
      symbol->interfaces, symbol->root_region_count,
      symbol->body_region_index_plus_one,
      symbol->kernel_workload_region_index_plus_one);
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
  if (!symbol->defining_op || loom_link_symbol_is_declaration(symbol)) {
    flags |= LOOM_LINK_SYMBOL_FLAG_DECLARATION;
  }
  if (loom_link_symbol_is_concrete_definition(symbol)) {
    flags |= LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION;
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_CONFIG)) {
    flags |= LOOM_LINK_SYMBOL_FLAG_CONFIG;
  }
  if (loom_symbol_definition_is_test_only(symbol->definition)) {
    flags |= LOOM_LINK_SYMBOL_FLAG_TEST_ONLY;
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
    case LOOM_BYTECODE_SYMBOL_TEMPLATE_DECL:
      return LOOM_SYMBOL_TEMPLATE_DECL;
    case LOOM_BYTECODE_SYMBOL_TEMPLATE_DEF:
      return LOOM_SYMBOL_TEMPLATE_DEF;
    case LOOM_BYTECODE_SYMBOL_TEMPLATE_UKERNEL:
      return LOOM_SYMBOL_TEMPLATE_UKERNEL;
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
  if (iree_any_bit_set(symbol->flags, LOOM_BYTECODE_SYMBOL_FLAG_EXPORT)) {
    flags |= LOOM_LINK_SYMBOL_FLAG_EXPORT;
  }
  const bool is_declaration =
      symbol->kind == LOOM_BYTECODE_SYMBOL_ANCHOR ||
      iree_any_bit_set(symbol->flags, LOOM_BYTECODE_SYMBOL_FLAG_DECLARATION);
  if (is_declaration || is_import) {
    flags |= LOOM_LINK_SYMBOL_FLAG_DECLARATION;
  }
  if (!is_declaration) {
    flags |= LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION;
  }
  if (iree_any_bit_set(symbol->interfaces, LOOM_SYMBOL_INTERFACE_CONFIG)) {
    flags |= LOOM_LINK_SYMBOL_FLAG_CONFIG;
  }
  if (iree_any_bit_set(symbol->flags, LOOM_BYTECODE_SYMBOL_FLAG_TEST_ONLY)) {
    flags |= LOOM_LINK_SYMBOL_FLAG_TEST_ONLY;
  }
  return flags;
}

static loom_link_symbol_identity_t loom_link_bytecode_symbol_identity(
    const loom_bytecode_symbol_metadata_t* symbol,
    loom_link_symbol_flags_t flags) {
  if (symbol->kind == LOOM_BYTECODE_SYMBOL_ANCHOR ||
      iree_any_bit_set(flags, LOOM_LINK_SYMBOL_FLAG_PUBLIC |
                                  LOOM_LINK_SYMBOL_FLAG_EXPORT |
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
    const loom_op_t* defining_op = symbol->defining_op;
    const loom_op_vtable_t* vtable =
        defining_op ? loom_op_vtable(source_module, defining_op) : NULL;
    const loom_link_symbol_facet_schema_t facet_schema =
        loom_link_classify_materialized_symbol_facets(defining_op, vtable);
    IREE_RETURN_IF_ERROR(loom_link_index_append_symbol(
        index, module, name, symbol->kind, identity,
        loom_link_materialized_symbol_flags(source_module, symbol),
        facet_schema));
  }
  return iree_ok_status();
}

static bool loom_link_symbol_is_template_provider(loom_symbol_kind_t kind) {
  return kind == LOOM_SYMBOL_TEMPLATE_DEF ||
         kind == LOOM_SYMBOL_TEMPLATE_UKERNEL;
}

static iree_status_t loom_link_index_project_materialized_template_families(
    loom_link_module_index_t* index, loom_link_module_index_module_t* module,
    const loom_module_t* source_module) {
  for (iree_host_size_t i = 0; i < source_module->symbols.count; ++i) {
    const loom_symbol_t* source_symbol = &source_module->symbols.entries[i];
    loom_link_module_index_symbol_t* symbol =
        &index->symbols.values[module->symbol_start_ordinal + i];
    if (source_symbol->kind == LOOM_SYMBOL_TEMPLATE_DECL) {
      IREE_RETURN_IF_ERROR(loom_link_index_resolve_template_family(
          index, symbol->ordinal, &symbol->template_family_ordinal));
      continue;
    }
    if (!loom_link_symbol_is_template_provider(source_symbol->kind)) {
      continue;
    }

    loom_func_like_t function =
        loom_func_like_cast(source_module, source_symbol->defining_op);
    const loom_symbol_ref_t family = loom_func_like_template_family(function);
    if (!loom_func_like_isa(function) || !loom_symbol_ref_is_valid(family) ||
        family.module_id != 0 ||
        family.symbol_id >= source_module->symbols.count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "template provider '%.*s' has an invalid family symbol",
          (int)symbol->name.size, symbol->name.data);
    }
    IREE_RETURN_IF_ERROR(loom_link_index_resolve_template_family(
        index, module->symbol_start_ordinal + family.symbol_id,
        &symbol->template_family_ordinal));
    loom_link_index_append_template_provider(
        index, symbol->template_family_ordinal, symbol->ordinal);
  }
  return iree_ok_status();
}

static iree_status_t loom_link_index_project_bytecode_template_families(
    loom_link_module_index_t* index, loom_link_module_index_module_t* module,
    const loom_bytecode_module_metadata_t* bytecode_module) {
  for (iree_host_size_t i = 0; i < bytecode_module->symbol_count; ++i) {
    const loom_bytecode_symbol_metadata_t* source_symbol =
        &bytecode_module->symbols[i];
    loom_link_module_index_symbol_t* symbol =
        &index->symbols.values[module->symbol_start_ordinal + i];
    if (source_symbol->kind == LOOM_BYTECODE_SYMBOL_TEMPLATE_DECL) {
      IREE_RETURN_IF_ERROR(loom_link_index_resolve_template_family(
          index, symbol->ordinal, &symbol->template_family_ordinal));
      continue;
    }
    if (!loom_link_symbol_is_template_provider(symbol->kind)) {
      continue;
    }
    if (source_symbol->template_family_symbol_ordinal >=
        bytecode_module->symbol_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "template provider '%.*s' has an invalid family symbol ordinal",
          (int)symbol->name.size, symbol->name.data);
    }
    IREE_RETURN_IF_ERROR(loom_link_index_resolve_template_family(
        index,
        module->symbol_start_ordinal +
            source_symbol->template_family_symbol_ordinal,
        &symbol->template_family_ordinal));
    loom_link_index_append_template_provider(
        index, symbol->template_family_ordinal, symbol->ordinal);
  }
  return iree_ok_status();
}

static uint32_t loom_link_index_count_dependency_occurrences(
    const loom_symbol_reference_table_t* table,
    loom_symbol_reference_occurrence_id_t first_occurrence_id) {
  uint32_t dependency_count = 0;
  loom_symbol_reference_occurrence_id_t occurrence_id = first_occurrence_id;
  while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
    const loom_symbol_reference_occurrence_t* occurrence =
        &table->occurrences[occurrence_id];
    if (loom_symbol_reference_occurrence_is_dependency(occurrence)) {
      ++dependency_count;
    }
    occurrence_id = occurrence->next_outgoing_occurrence_id;
  }
  return dependency_count;
}

static void loom_link_index_copy_dependency_occurrences(
    const loom_symbol_reference_table_t* table,
    loom_symbol_reference_occurrence_id_t first_occurrence_id, uint32_t* values,
    loom_symbol_interface_flags_t* target_interfaces,
    uint8_t* source_root_region_indices_plus_one, iree_host_size_t* position) {
  loom_symbol_reference_occurrence_id_t occurrence_id = first_occurrence_id;
  while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
    const loom_symbol_reference_occurrence_t* occurrence =
        &table->occurrences[occurrence_id];
    if (loom_symbol_reference_occurrence_is_dependency(occurrence)) {
      values[*position] = occurrence->target_symbol_id;
      target_interfaces[*position] = occurrence->target_interfaces;
      source_root_region_indices_plus_one[*position] =
          occurrence->source_root_region_index_plus_one;
      ++*position;
    }
    occurrence_id = occurrence->next_outgoing_occurrence_id;
  }
}

static iree_status_t loom_link_index_project_symbol_references(
    loom_link_module_index_t* index, loom_link_module_index_module_t* module,
    const loom_symbol_reference_table_t* table) {
  uint32_t* dependency_values = NULL;
  loom_symbol_interface_flags_t* dependency_target_interfaces = NULL;
  uint8_t* dependency_source_root_region_indices_plus_one = NULL;
  loom_link_template_family_ordinal_t* template_demand_values = NULL;
  uint8_t* template_demand_source_root_region_indices_plus_one = NULL;
  iree_status_t status = iree_ok_status();
  iree_host_size_t dependency_count = 0;
  for (iree_host_size_t i = 0; i < table->occurrence_count; ++i) {
    if (loom_symbol_reference_occurrence_is_dependency(
            &table->occurrences[i])) {
      ++dependency_count;
    }
  }
  module->dependencies.root_count =
      loom_link_index_count_dependency_occurrences(
          table, table->first_module_occurrence_id);
  module->dependencies.count = dependency_count;
  if (dependency_count > 0) {
    status = iree_arena_allocate_array(&index->arena, dependency_count,
                                       sizeof(*dependency_values),
                                       (void**)&dependency_values);
    module->dependencies.values = dependency_values;
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate_array(&index->arena, dependency_count,
                                         sizeof(*dependency_target_interfaces),
                                         (void**)&dependency_target_interfaces);
      module->dependencies.target_interfaces = dependency_target_interfaces;
    }
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate_array(
          &index->arena, dependency_count,
          sizeof(*dependency_source_root_region_indices_plus_one),
          (void**)&dependency_source_root_region_indices_plus_one);
      module->dependencies.source_root_region_indices_plus_one =
          dependency_source_root_region_indices_plus_one;
    }
  }

  if (iree_status_is_ok(status)) {
    module->template_demands.count = table->template_demands.count;
    if (table->template_demands.count > 0) {
      status = iree_arena_allocate_array(
          &index->arena, table->template_demands.count,
          sizeof(*template_demand_values), (void**)&template_demand_values);
      module->template_demands.values = template_demand_values;
      if (iree_status_is_ok(status)) {
        status = iree_arena_allocate_array(
            &index->arena, table->template_demands.count,
            sizeof(*template_demand_source_root_region_indices_plus_one),
            (void**)&template_demand_source_root_region_indices_plus_one);
        module->template_demands.source_root_region_indices_plus_one =
            template_demand_source_root_region_indices_plus_one;
      }
    }
  }

  iree_host_size_t dependency_position = 0;
  iree_host_size_t template_demand_position = 0;
  if (iree_status_is_ok(status)) {
    loom_link_index_copy_dependency_occurrences(
        table, table->first_module_occurrence_id, dependency_values,
        dependency_target_interfaces,
        dependency_source_root_region_indices_plus_one, &dependency_position);
    for (iree_host_size_t symbol_index = 0;
         symbol_index < table->symbol_count && iree_status_is_ok(status);
         ++symbol_index) {
      loom_link_module_index_symbol_t* symbol =
          &index->symbols.values[module->symbol_start_ordinal + symbol_index];
      const loom_symbol_reference_symbol_occurrences_t* source =
          &table->symbols[symbol_index];
      symbol->dependencies.first = (uint32_t)dependency_position;
      symbol->dependencies.count = loom_link_index_count_dependency_occurrences(
          table, source->first_outgoing_occurrence_id);
      loom_link_index_copy_dependency_occurrences(
          table, source->first_outgoing_occurrence_id, dependency_values,
          dependency_target_interfaces,
          dependency_source_root_region_indices_plus_one, &dependency_position);

      symbol->template_demands.first = (uint32_t)template_demand_position;
      symbol->template_demands.count = source->template_demand_count;
      loom_template_demand_id_t demand_id = source->first_template_demand_id;
      while (demand_id != LOOM_TEMPLATE_DEMAND_ID_INVALID &&
             iree_status_is_ok(status)) {
        const loom_template_demand_t* demand =
            &table->template_demands.values[demand_id];
        loom_link_template_family_ordinal_t family_ordinal =
            LOOM_LINK_TEMPLATE_FAMILY_ORDINAL_INVALID;
        status = loom_link_index_resolve_template_family(
            index, module->symbol_start_ordinal + demand->family_symbol_id,
            &family_ordinal);
        if (iree_status_is_ok(status)) {
          template_demand_values[template_demand_position] = family_ordinal;
          template_demand_source_root_region_indices_plus_one
              [template_demand_position] =
                  demand->source_root_region_index_plus_one;
          ++template_demand_position;
        }
        demand_id = demand->next_source_demand_id;
      }
    }
  }

  return status;
}

static iree_status_t loom_link_index_project_materialized_references(
    loom_link_module_index_t* index, loom_link_module_index_module_t* module,
    const loom_module_t* source_module) {
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(index->block_pool, &scratch_arena);

  loom_symbol_reference_table_t table = {0};
  iree_status_t status =
      loom_symbol_reference_table_build(source_module, &scratch_arena, &table);
  if (iree_status_is_ok(status)) {
    status = loom_link_index_project_symbol_references(index, module, &table);
  }

  iree_arena_deinitialize(&scratch_arena);
  return status;
}

static iree_status_t loom_link_index_module_bytecode_symbols(
    loom_link_module_index_t* index, loom_link_module_index_module_t* module,
    const loom_bytecode_module_metadata_t* bytecode_module) {
  for (iree_host_size_t i = 0; i < bytecode_module->symbol_count; ++i) {
    const loom_bytecode_symbol_metadata_t* symbol =
        &bytecode_module->symbols[i];
    loom_link_symbol_flags_t flags = loom_link_bytecode_symbol_flags(symbol);
    const loom_link_symbol_facet_schema_t facet_schema =
        loom_link_classify_bytecode_symbol_facets(symbol);
    IREE_RETURN_IF_ERROR(loom_link_index_append_symbol(
        index, module, symbol->name,
        loom_link_bytecode_symbol_kind(symbol->kind),
        loom_link_bytecode_symbol_identity(symbol, flags), flags,
        facet_schema));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_index_project_bytecode_references(
    loom_link_module_index_t* index, loom_link_module_index_module_t* module,
    const loom_bytecode_module_metadata_t* bytecode_module) {
  module->dependencies.root_count = bytecode_module->module_dependency_count;
  module->dependencies.count = bytecode_module->dependency_count;
  module->dependencies.values = bytecode_module->dependency_symbol_indices;
  module->dependencies.target_interfaces =
      bytecode_module->dependency_target_interfaces;
  module->dependencies.source_root_region_indices_plus_one =
      bytecode_module->dependency_source_root_region_indices_plus_one;
  module->template_demands.count = bytecode_module->template_demand_count;
  module->template_demands.source_root_region_indices_plus_one =
      bytecode_module->template_demand_source_root_region_indices_plus_one;
  loom_link_template_family_ordinal_t* template_demand_values = NULL;
  if (bytecode_module->template_demand_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &index->arena, bytecode_module->template_demand_count,
        sizeof(*template_demand_values), (void**)&template_demand_values));
    module->template_demands.values = template_demand_values;
  }

  for (iree_host_size_t symbol_index = 0;
       symbol_index < bytecode_module->symbol_count; ++symbol_index) {
    loom_link_module_index_symbol_t* symbol =
        &index->symbols.values[module->symbol_start_ordinal + symbol_index];
    const loom_bytecode_symbol_reference_metadata_t* source =
        &bytecode_module->symbol_references[symbol_index];
    symbol->dependencies.first = source->first_dependency_index;
    symbol->dependencies.count = source->dependency_count;
    symbol->template_demands.first = source->first_template_demand_index;
    symbol->template_demands.count = source->template_demand_count;
  }
  for (iree_host_size_t i = 0; i < bytecode_module->template_demand_count;
       ++i) {
    const uint32_t family_symbol_ordinal =
        bytecode_module->template_demand_family_symbol_ordinals[i];
    loom_link_template_family_ordinal_t family_ordinal =
        LOOM_LINK_TEMPLATE_FAMILY_ORDINAL_INVALID;
    IREE_RETURN_IF_ERROR(loom_link_index_resolve_template_family(
        index, module->symbol_start_ordinal + family_symbol_ordinal,
        &family_ordinal));
    template_demand_values[i] = family_ordinal;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Provider import projection
//===----------------------------------------------------------------------===//

static iree_status_t loom_link_index_allocate_provider_import_projection(
    loom_link_module_index_t* index, loom_link_module_index_module_t* module,
    iree_host_size_t import_count, iree_host_size_t anchor_count,
    uint32_t** out_symbol_offsets, uint32_t** out_symbol_import_ordinals) {
  *out_symbol_offsets = NULL;
  *out_symbol_import_ordinals = NULL;
  if (import_count > UINT32_MAX || anchor_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "module provider import projection exceeds uint32 capacity");
  }

  module->provider_imports.count = (uint32_t)import_count;
  module->provider_imports.anchor_count = (uint32_t)anchor_count;
  if (anchor_count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t symbol_offset_count = 0;
  if (!iree_host_size_checked_add(module->symbol_count, 1,
                                  &symbol_offset_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "module provider import symbol count overflow");
  }
  uint32_t* symbol_offsets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &index->arena, symbol_offset_count, sizeof(*symbol_offsets),
      (void**)&symbol_offsets));
  for (iree_host_size_t i = 0; i < symbol_offset_count; ++i) {
    symbol_offsets[i] = 0;
  }

  uint32_t* symbol_import_ordinals = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &index->arena, anchor_count, sizeof(*symbol_import_ordinals),
      (void**)&symbol_import_ordinals));

  module->provider_imports.symbol_offsets = symbol_offsets;
  module->provider_imports.symbol_import_ordinals = symbol_import_ordinals;
  *out_symbol_offsets = symbol_offsets;
  *out_symbol_import_ordinals = symbol_import_ordinals;
  return iree_ok_status();
}

static void loom_link_index_prefix_provider_import_counts(
    iree_host_size_t symbol_count, uint32_t* symbol_offsets) {
  for (iree_host_size_t i = 1; i <= symbol_count; ++i) {
    symbol_offsets[i] += symbol_offsets[i - 1];
  }
}

// Reverse filling decrements each prefix end into its corresponding start.
// Shift those starts into conventional CSR form and restore the final end.
static void loom_link_index_finish_provider_import_projection(
    iree_host_size_t symbol_count, iree_host_size_t anchor_count,
    uint32_t* symbol_offsets) {
  for (iree_host_size_t i = 0; i < symbol_count; ++i) {
    symbol_offsets[i] = symbol_offsets[i + 1];
  }
  symbol_offsets[symbol_count] = (uint32_t)anchor_count;
}

static iree_status_t loom_link_index_project_materialized_provider_imports(
    loom_link_module_index_t* index, loom_link_module_index_module_t* module,
    const loom_module_t* source_module) {
  iree_host_size_t import_count = 0;
  iree_host_size_t anchor_count = 0;
  for (uint16_t block_index = 0; block_index < source_module->body->block_count;
       ++block_index) {
    const loom_block_t* block =
        loom_region_const_block(source_module->body, block_index);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_module_import_isa(op)) {
        continue;
      }
      if (!iree_host_size_checked_add(import_count, 1, &import_count) ||
          !iree_host_size_checked_add(anchor_count,
                                      loom_module_import_symbols(op).count,
                                      &anchor_count)) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "module provider import count overflow");
      }
    }
  }

  uint32_t* symbol_offsets = NULL;
  uint32_t* symbol_import_ordinals = NULL;
  IREE_RETURN_IF_ERROR(loom_link_index_allocate_provider_import_projection(
      index, module, import_count, anchor_count, &symbol_offsets,
      &symbol_import_ordinals));
  const loom_op_t** import_ops = NULL;
  if (import_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &index->arena, import_count, sizeof(*import_ops), (void**)&import_ops));
    module->provider_imports.materialized_ops = import_ops;
  }

  iree_host_size_t import_ordinal = 0;
  for (uint16_t block_index = 0; block_index < source_module->body->block_count;
       ++block_index) {
    const loom_block_t* block =
        loom_region_const_block(source_module->body, block_index);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_module_import_isa(op)) {
        continue;
      }
      const loom_string_id_t provider_id = loom_module_import_provider(op);
      if (provider_id >= source_module->strings.count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "module.import provider string is invalid");
      }
      import_ops[import_ordinal++] = op;
      const loom_symbol_ref_array_t anchors = loom_module_import_symbols(op);
      for (uint16_t i = 0; i < anchors.count; ++i) {
        const loom_symbol_ref_t anchor = anchors.values[i];
        if (anchor.module_id != 0 || anchor.symbol_id >= module->symbol_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "module.import anchor %u:%u is outside its source module",
              (unsigned)anchor.module_id, (unsigned)anchor.symbol_id);
        }
        ++symbol_offsets[anchor.symbol_id + 1];
      }
    }
  }
  if (anchor_count == 0) {
    return iree_ok_status();
  }

  loom_link_index_prefix_provider_import_counts(module->symbol_count,
                                                symbol_offsets);
  for (iree_host_size_t i = import_count; i-- > 0;) {
    const loom_symbol_ref_array_t anchors =
        loom_module_import_symbols(import_ops[i]);
    for (uint16_t j = anchors.count; j-- > 0;) {
      const uint16_t symbol_id = anchors.values[j].symbol_id;
      const uint32_t occurrence_ordinal = --symbol_offsets[symbol_id + 1];
      symbol_import_ordinals[occurrence_ordinal] = (uint32_t)i;
    }
  }
  loom_link_index_finish_provider_import_projection(
      module->symbol_count, anchor_count, symbol_offsets);
  return iree_ok_status();
}

static iree_status_t loom_link_index_project_bytecode_provider_imports(
    loom_link_module_index_t* index, loom_link_module_index_module_t* module,
    const loom_bytecode_module_metadata_t* bytecode_module) {
  uint32_t* symbol_offsets = NULL;
  uint32_t* symbol_import_ordinals = NULL;
  IREE_RETURN_IF_ERROR(loom_link_index_allocate_provider_import_projection(
      index, module, bytecode_module->provider_import_count,
      bytecode_module->provider_import_anchor_count, &symbol_offsets,
      &symbol_import_ordinals));
  if (bytecode_module->provider_import_anchor_count == 0) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < bytecode_module->provider_import_count;
       ++i) {
    const loom_bytecode_provider_import_metadata_t* provider_import =
        &bytecode_module->provider_imports[i];
    const uint32_t* anchors =
        provider_import->anchor_count > 0
            ? bytecode_module->provider_import_anchor_symbol_indices +
                  provider_import->first_anchor_index
            : NULL;
    for (uint32_t j = 0; j < provider_import->anchor_count; ++j) {
      ++symbol_offsets[anchors[j] + 1];
    }
  }

  loom_link_index_prefix_provider_import_counts(module->symbol_count,
                                                symbol_offsets);
  for (iree_host_size_t i = bytecode_module->provider_import_count; i-- > 0;) {
    const loom_bytecode_provider_import_metadata_t* provider_import =
        &bytecode_module->provider_imports[i];
    const uint32_t* anchors =
        provider_import->anchor_count > 0
            ? bytecode_module->provider_import_anchor_symbol_indices +
                  provider_import->first_anchor_index
            : NULL;
    for (uint32_t j = provider_import->anchor_count; j-- > 0;) {
      const uint32_t occurrence_ordinal = --symbol_offsets[anchors[j] + 1];
      symbol_import_ordinals[occurrence_ordinal] = (uint32_t)i;
    }
  }
  loom_link_index_finish_provider_import_projection(
      module->symbol_count, bytecode_module->provider_import_anchor_count,
      symbol_offsets);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t loom_link_module_index_allocate(
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
  for (iree_host_size_t i = 0; i < index->modules.count; ++i) {
    if (index->modules.values[i].owns_materialized_module) {
      loom_module_free(
          (loom_module_t*)index->modules.values[i].materialized_module);
    }
  }
  iree_allocator_free(index->allocator, index->template_families.values);
  iree_allocator_free(index->allocator, index->names.values);
  iree_allocator_free(index->allocator, index->input_exports.values);
  iree_allocator_free(index->allocator, index->symbols.values);
  iree_allocator_free(index->allocator, index->modules.values);
  iree_allocator_free(index->allocator, index->providers.values);
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

  loom_link_module_index_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(loom_link_index_append_provider(
      index, LOOM_LINK_PROVIDER_MATERIALIZED,
      loom_link_materialized_module_name(module), options, &provider));

  loom_link_module_index_module_t* indexed_module = NULL;
  IREE_RETURN_IF_ERROR(loom_link_index_append_module(
      index, provider, /*provider_module_ordinal=*/0,
      loom_link_materialized_module_name(module), module,
      /*owns_materialized_module=*/false, &indexed_module));
  IREE_RETURN_IF_ERROR(loom_link_index_module_materialized_symbols(
      index, indexed_module, module));
  IREE_RETURN_IF_ERROR(loom_link_index_project_materialized_template_families(
      index, indexed_module, module));
  IREE_RETURN_IF_ERROR(loom_link_index_project_materialized_references(
      index, indexed_module, module));
  IREE_RETURN_IF_ERROR(loom_link_index_project_materialized_provider_imports(
      index, indexed_module, module));
  if (out_provider_ordinal) {
    *out_provider_ordinal = provider->ordinal;
  }
  return iree_ok_status();
}

iree_status_t loom_link_module_index_add_bytecode(
    loom_link_module_index_t* index, iree_const_byte_span_t bytecode,
    iree_string_view_t filename,
    const loom_bytecode_index_options_t* index_options,
    const loom_link_module_index_add_options_t* options,
    iree_host_size_t* out_provider_ordinal) {
  IREE_ASSERT_ARGUMENT(index);

  loom_bytecode_read_result_t read_result = {0};
  loom_bytecode_file_metadata_t metadata = {0};
  IREE_RETURN_IF_ERROR(loom_bytecode_read_index(
      bytecode, filename, index->context, index->block_pool, &index->arena,
      index_options, &read_result, &metadata));
  if (read_result.error_count > 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bytecode provider '%.*s' has %u validation errors",
                            (int)filename.size, filename.data,
                            read_result.error_count);
  }

  const bool provider_name_is_filename =
      !options || iree_string_view_is_empty(options->provider_name) ||
      iree_string_view_equal(options->provider_name, filename);
  iree_string_view_t retained_filename = iree_string_view_empty();
  if (!provider_name_is_filename) {
    IREE_RETURN_IF_ERROR(
        loom_link_index_copy_string(index, filename, &retained_filename));
  }

  loom_link_module_index_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(loom_link_index_append_provider(
      index, LOOM_LINK_PROVIDER_BYTECODE, filename, options, &provider));
  provider->bytecode.contents = bytecode;
  provider->bytecode.filename =
      provider_name_is_filename ? provider->name : retained_filename;
  provider->bytecode.metadata = metadata;

  for (iree_host_size_t i = 0; i < metadata.module_count; ++i) {
    loom_link_module_index_module_t* indexed_module = NULL;
    IREE_RETURN_IF_ERROR(loom_link_index_append_module(
        index, provider, i, metadata.modules[i].name,
        /*materialized_module=*/NULL, /*owns_materialized_module=*/false,
        &indexed_module));
    IREE_RETURN_IF_ERROR(loom_link_index_module_bytecode_symbols(
        index, indexed_module, &metadata.modules[i]));
    IREE_RETURN_IF_ERROR(loom_link_index_project_bytecode_template_families(
        index, indexed_module, &metadata.modules[i]));
    IREE_RETURN_IF_ERROR(loom_link_index_project_bytecode_references(
        index, indexed_module, &metadata.modules[i]));
    IREE_RETURN_IF_ERROR(loom_link_index_project_bytecode_provider_imports(
        index, indexed_module, &metadata.modules[i]));
  }

  if (out_provider_ordinal) {
    *out_provider_ordinal = provider->ordinal;
  }
  return iree_ok_status();
}

iree_status_t loom_link_module_index_add_text(
    loom_link_module_index_t* index, iree_string_view_t source,
    iree_string_view_t filename, const loom_text_parse_options_t* parse_options,
    const loom_link_module_index_add_options_t* options,
    iree_host_size_t* out_provider_ordinal) {
  IREE_ASSERT_ARGUMENT(index);

  iree_arena_allocator_t symbol_reference_arena;
  iree_arena_initialize(index->block_pool, &symbol_reference_arena);

  loom_symbol_reference_table_t symbol_references = {0};
  loom_module_t* module = NULL;
  iree_status_t status = loom_text_parse_with_symbol_references(
      source, filename, index->context, index->block_pool, parse_options,
      &symbol_reference_arena, &symbol_references, &module);
  if (iree_status_is_ok(status) && !module) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "text provider '%.*s' did not parse into a module",
                         (int)filename.size, filename.data);
  }
  if (iree_status_is_ok(status)) {
    status = loom_link_index_validate_materialized_module(module);
  }

  loom_link_module_index_provider_t* provider = NULL;
  bool module_owned_by_index = false;
  if (iree_status_is_ok(status)) {
    status = loom_link_index_append_provider(index, LOOM_LINK_PROVIDER_TEXT,
                                             filename, options, &provider);
  }
  if (iree_status_is_ok(status)) {
    loom_link_module_index_module_t* indexed_module = NULL;
    status = loom_link_index_append_module(
        index, provider, /*provider_module_ordinal=*/0,
        loom_link_materialized_module_name(module), module,
        /*owns_materialized_module=*/true, &indexed_module);
    module_owned_by_index = iree_status_is_ok(status);
    if (iree_status_is_ok(status)) {
      status = loom_link_index_module_materialized_symbols(
          index, indexed_module, module);
    }
    if (iree_status_is_ok(status)) {
      status = loom_link_index_project_materialized_template_families(
          index, indexed_module, module);
    }
    if (iree_status_is_ok(status)) {
      status = loom_link_index_project_symbol_references(index, indexed_module,
                                                         &symbol_references);
    }
    if (iree_status_is_ok(status)) {
      status = loom_link_index_project_materialized_provider_imports(
          index, indexed_module, module);
    }
  }

  iree_arena_deinitialize(&symbol_reference_arena);
  if (!iree_status_is_ok(status)) {
    if (module && !module_owned_by_index) {
      loom_module_free(module);
    }
    return status;
  }

  if (out_provider_ordinal) {
    *out_provider_ordinal = provider->ordinal;
  }
  return iree_ok_status();
}

iree_host_size_t loom_link_module_index_provider_count(
    const loom_link_module_index_t* index) {
  return index ? index->providers.count : 0;
}

const loom_link_module_index_provider_t* loom_link_module_index_provider_at(
    const loom_link_module_index_t* index, iree_host_size_t ordinal) {
  if (!index || ordinal >= index->providers.count) return NULL;
  return &index->providers.values[ordinal];
}

iree_host_size_t loom_link_module_index_module_count(
    const loom_link_module_index_t* index) {
  return index ? index->modules.count : 0;
}

const loom_link_module_index_module_t* loom_link_module_index_module_at(
    const loom_link_module_index_t* index, iree_host_size_t ordinal) {
  if (!index || ordinal >= index->modules.count) return NULL;
  return &index->modules.values[ordinal];
}

iree_host_size_t loom_link_module_index_symbol_count(
    const loom_link_module_index_t* index) {
  return index ? index->symbols.count : 0;
}

iree_host_size_t loom_link_module_index_facet_count(
    const loom_link_module_index_t* index) {
  return index ? index->facet_count : 0;
}

const loom_link_module_index_symbol_t* loom_link_module_index_symbol_at(
    const loom_link_module_index_t* index, iree_host_size_t ordinal) {
  if (!index || ordinal >= index->symbols.count) {
    return NULL;
  }
  return &index->symbols.values[ordinal];
}

loom_link_symbol_facet_kind_t loom_link_module_index_symbol_facet_kind_at(
    const loom_link_module_index_symbol_t* symbol, uint8_t ordinal) {
  return symbol ? loom_link_symbol_facet_schema_kind_at(&symbol->facets.schema,
                                                        ordinal)
                : LOOM_LINK_SYMBOL_FACET_INVALID;
}

loom_link_symbol_facet_kind_t
loom_link_module_index_symbol_source_root_facet_kind(
    const loom_link_module_index_symbol_t* symbol,
    uint8_t source_root_region_index_plus_one) {
  return symbol ? loom_link_symbol_facet_schema_source_root_kind(
                      &symbol->facets.schema, source_root_region_index_plus_one)
                : LOOM_LINK_SYMBOL_FACET_INVALID;
}

iree_host_size_t loom_link_module_index_symbol_facet_ordinal(
    const loom_link_module_index_symbol_t* symbol,
    loom_link_symbol_facet_kind_t kind) {
  if (!symbol || kind == LOOM_LINK_SYMBOL_FACET_INVALID) {
    return LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  }
  for (uint8_t i = 0; i < symbol->facets.schema.facet_count; ++i) {
    if (loom_link_module_index_symbol_facet_kind_at(symbol, i) == kind) {
      return symbol->facets.start_ordinal + i;
    }
  }
  return LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
}

loom_link_module_index_symbol_ordinal_list_t
loom_link_module_index_input_exports(const loom_link_module_index_t* index) {
  return index ? (loom_link_module_index_symbol_ordinal_list_t){
                     .values = index->input_exports.values,
                     .count = index->input_exports.count,
                 }
               : (loom_link_module_index_symbol_ordinal_list_t){0};
}

const loom_link_module_index_provider_t* loom_link_module_index_symbol_provider(
    const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* symbol) {
  const loom_link_module_index_module_t* module =
      loom_link_module_index_symbol_module(index, symbol);
  if (!module || module->provider_ordinal >= index->providers.count) {
    return NULL;
  }
  return &index->providers.values[module->provider_ordinal];
}

const loom_link_module_index_module_t* loom_link_module_index_symbol_module(
    const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* symbol) {
  if (!index || !symbol || symbol->module_ordinal >= index->modules.count) {
    return NULL;
  }
  return &index->modules.values[symbol->module_ordinal];
}

loom_link_module_index_provider_import_t
loom_link_module_index_provider_import_at(
    const loom_link_module_index_t* index,
    const loom_link_module_index_module_t* module, iree_host_size_t ordinal) {
  const loom_link_module_index_provider_t* provider =
      &index->providers.values[module->provider_ordinal];
  if (provider->kind == LOOM_LINK_PROVIDER_BYTECODE) {
    const loom_bytecode_module_metadata_t* bytecode_module =
        &provider->bytecode.metadata.modules[module->provider_module_ordinal];
    const loom_bytecode_provider_import_metadata_t* provider_import =
        &bytecode_module->provider_imports[ordinal];
    return (loom_link_module_index_provider_import_t){
        .provider = provider_import->provider,
        .anchor_count = provider_import->anchor_count,
        .comments =
            {
                .values = provider_import->comments,
                .count = provider_import->comment_count,
            },
        .leading_blank_line = provider_import->leading_blank_line,
    };
  }

  const loom_module_t* source_module = module->materialized_module;
  const loom_op_t* op = module->provider_imports.materialized_ops[ordinal];
  const loom_string_id_t provider_id = loom_module_import_provider(op);
  iree_host_size_t comment_count = 0;
  const iree_string_view_t* comments =
      loom_module_op_comments(source_module, op, &comment_count);
  return (loom_link_module_index_provider_import_t){
      .provider = source_module->strings.entries[provider_id],
      .anchor_count = loom_module_import_symbols(op).count,
      .comments =
          {
              .values = comments,
              .count = comment_count,
          },
      .leading_blank_line =
          iree_any_bit_set(op->flags, LOOM_OP_FLAG_LEADING_BLANK_LINE),
  };
}

iree_string_view_t loom_link_module_index_provider_import_key_at(
    const loom_link_module_index_t* index,
    const loom_link_module_index_module_t* module, iree_host_size_t ordinal) {
  const loom_link_module_index_provider_t* provider =
      &index->providers.values[module->provider_ordinal];
  if (provider->kind == LOOM_LINK_PROVIDER_BYTECODE) {
    const loom_bytecode_module_metadata_t* bytecode_module =
        &provider->bytecode.metadata.modules[module->provider_module_ordinal];
    return bytecode_module->provider_imports[ordinal].provider;
  }
  const loom_module_t* source_module = module->materialized_module;
  const loom_op_t* op = module->provider_imports.materialized_ops[ordinal];
  return source_module->strings.entries[loom_module_import_provider(op)];
}

uint32_t loom_link_module_index_provider_import_anchor_at(
    const loom_link_module_index_t* index,
    const loom_link_module_index_module_t* module,
    iree_host_size_t import_ordinal, iree_host_size_t anchor_ordinal) {
  const loom_link_module_index_provider_t* provider =
      &index->providers.values[module->provider_ordinal];
  if (provider->kind == LOOM_LINK_PROVIDER_BYTECODE) {
    const loom_bytecode_module_metadata_t* bytecode_module =
        &provider->bytecode.metadata.modules[module->provider_module_ordinal];
    const loom_bytecode_provider_import_metadata_t* provider_import =
        &bytecode_module->provider_imports[import_ordinal];
    return bytecode_module->provider_import_anchor_symbol_indices
        [provider_import->first_anchor_index + anchor_ordinal];
  }

  const loom_op_t* op =
      module->provider_imports.materialized_ops[import_ordinal];
  return loom_module_import_symbols(op).values[anchor_ordinal].symbol_id;
}

loom_link_module_index_provider_import_list_t
loom_link_module_index_symbol_provider_imports(
    const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* symbol) {
  const loom_link_module_index_module_t* module =
      &index->modules.values[symbol->module_ordinal];
  if (module->provider_imports.anchor_count == 0) {
    return (loom_link_module_index_provider_import_list_t){0};
  }
  const uint32_t first =
      module->provider_imports.symbol_offsets[symbol->module_symbol_ordinal];
  const uint32_t end = module->provider_imports
                           .symbol_offsets[symbol->module_symbol_ordinal + 1];
  return (loom_link_module_index_provider_import_list_t){
      .values = first < end
                    ? module->provider_imports.symbol_import_ordinals + first
                    : NULL,
      .count = end - first,
  };
}

const loom_link_module_index_symbol_t* loom_link_module_index_lookup_global(
    const loom_link_module_index_t* index, iree_string_view_t name) {
  if (!index || index->names.capacity == 0) return NULL;
  name = loom_link_normalize_symbol_name(name);
  const iree_host_size_t slot = loom_link_index_name_map_slot(
      index->names.values, index->names.capacity, name);
  const iree_host_size_t symbol_ordinal =
      index->names.values[slot].selected_global_ordinal;
  if (symbol_ordinal == LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    return NULL;
  }
  return &index->symbols.values[symbol_ordinal];
}

const loom_link_module_index_symbol_t* loom_link_module_index_lookup_name(
    const loom_link_module_index_t* index, iree_string_view_t name) {
  if (!index || index->names.capacity == 0) return NULL;
  name = loom_link_normalize_symbol_name(name);
  const iree_host_size_t slot = loom_link_index_name_map_slot(
      index->names.values, index->names.capacity, name);
  const iree_host_size_t symbol_ordinal =
      index->names.values[slot].first_symbol_ordinal;
  if (symbol_ordinal == LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) return NULL;
  return &index->symbols.values[symbol_ordinal];
}

const loom_link_module_index_symbol_t* loom_link_module_index_next_same_name(
    const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* symbol) {
  if (!index || !symbol ||
      symbol->next.same_name_ordinal ==
          LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL ||
      symbol->next.same_name_ordinal >= index->symbols.count) {
    return NULL;
  }
  return &index->symbols.values[symbol->next.same_name_ordinal];
}

const loom_link_module_index_symbol_t*
loom_link_module_index_next_global_duplicate(
    const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* symbol) {
  if (!index || !symbol ||
      symbol->identity != LOOM_LINK_SYMBOL_IDENTITY_GLOBAL) {
    return NULL;
  }
  const loom_link_module_index_symbol_t* selected =
      loom_link_module_index_lookup_global(index, symbol->name);
  if (!selected) return NULL;

  // Walk forward from the current symbol. Reaching the selected symbol after
  // wrapping completes the selected-first cycle.
  const loom_link_module_index_symbol_t* candidate =
      loom_link_module_index_next_same_name(index, symbol);
  while (candidate) {
    if (candidate == selected) return NULL;
    if (candidate->identity == LOOM_LINK_SYMBOL_IDENTITY_GLOBAL) {
      return candidate;
    }
    candidate = loom_link_module_index_next_same_name(index, candidate);
  }

  // Wrap once to the start of the insertion-ordered same-name chain.
  candidate = loom_link_module_index_lookup_name(index, symbol->name);
  while (candidate && candidate != selected) {
    if (candidate->identity == LOOM_LINK_SYMBOL_IDENTITY_GLOBAL) {
      return candidate;
    }
    candidate = loom_link_module_index_next_same_name(index, candidate);
  }
  return NULL;
}

const loom_link_module_index_symbol_t* loom_link_module_index_lookup_private(
    const loom_link_module_index_t* index,
    const loom_link_module_index_module_t* module, iree_string_view_t name) {
  if (!index || !module) return NULL;
  name = loom_link_normalize_symbol_name(name);
  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_lookup_name(index, name);
  while (symbol) {
    if (symbol->module_ordinal == module->ordinal &&
        symbol->identity == LOOM_LINK_SYMBOL_IDENTITY_PRIVATE) {
      return symbol;
    }
    symbol = loom_link_module_index_next_same_name(index, symbol);
  }
  return NULL;
}

iree_host_size_t loom_link_module_index_template_family_count(
    const loom_link_module_index_t* index) {
  return index ? index->template_families.count : 0;
}

const loom_link_module_index_template_family_t*
loom_link_module_index_template_family_at(
    const loom_link_module_index_t* index,
    loom_link_template_family_ordinal_t ordinal) {
  if (!index || ordinal >= index->template_families.count) {
    return NULL;
  }
  return &index->template_families.values[ordinal];
}

iree_status_t loom_link_module_index_annotate_global_collision(
    iree_status_t status, const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* selected,
    const loom_link_module_index_symbol_t* duplicate) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(selected);
  IREE_ASSERT_ARGUMENT(duplicate);
  const loom_link_module_index_module_t* selected_module =
      &index->modules.values[selected->module_ordinal];
  const loom_link_module_index_module_t* duplicate_module =
      &index->modules.values[duplicate->module_ordinal];
  const loom_link_module_index_provider_t* selected_provider =
      &index->providers.values[selected_module->provider_ordinal];
  const loom_link_module_index_provider_t* duplicate_provider =
      &index->providers.values[duplicate_module->provider_ordinal];
  return iree_status_annotate_f(
      status,
      "global symbol '@%.*s' selected from provider '%.*s' module '%.*s' "
      "conflicts with provider '%.*s' module '%.*s'",
      (int)selected->name.size, selected->name.data,
      (int)selected_provider->name.size, selected_provider->name.data,
      (int)selected_module->name.size, selected_module->name.data,
      (int)duplicate_provider->name.size, duplicate_provider->name.data,
      (int)duplicate_module->name.size, duplicate_module->name.data);
}
