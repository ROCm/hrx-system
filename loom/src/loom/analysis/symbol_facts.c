// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbol_facts.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

typedef enum loom_symbol_fact_state_e {
  LOOM_SYMBOL_FACT_STATE_UNCOMPUTED = 0,
  LOOM_SYMBOL_FACT_STATE_COMPUTING = 1,
  LOOM_SYMBOL_FACT_STATE_COMPUTED = 2,
} loom_symbol_fact_state_t;

struct loom_symbol_fact_context_t {
  // Fact table owning the in-flight computation.
  loom_symbol_fact_table_t* table;

  // Module whose symbols are being queried.
  const loom_module_t* module;
};

void loom_symbol_fact_table_initialize(loom_symbol_fact_table_t* table,
                                       iree_arena_allocator_t* arena) {
  loom_symbol_fact_table_initialize_with_options(table, NULL, arena);
}

void loom_symbol_fact_table_initialize_with_options(
    loom_symbol_fact_table_t* table,
    const loom_symbol_fact_table_options_t* options,
    iree_arena_allocator_t* arena) {
  memset(table, 0, sizeof(*table));
  table->resources =
      options ? options->resources : loom_symbol_fact_resource_list_empty();
  table->arena = arena;
}

void loom_symbol_fact_table_reset(loom_symbol_fact_table_t* table) {
  table->module = NULL;
  if (table->count == 0) return;
  memset(table->entries, 0, table->count * sizeof(*table->entries));
  memset(table->states, 0, table->count * sizeof(*table->states));
}

static iree_status_t loom_symbol_fact_table_ensure_capacity(
    loom_symbol_fact_table_t* table, iree_host_size_t minimum_count) {
  if (minimum_count <= table->capacity) {
    if (minimum_count > table->count) table->count = minimum_count;
    return iree_ok_status();
  }

  iree_host_size_t new_capacity = table->capacity > 0 ? table->capacity : 128;
  while (new_capacity < minimum_count) {
    if (new_capacity > SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "symbol fact table capacity overflow");
    }
    new_capacity *= 2;
  }

  const loom_symbol_facts_base_t** new_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      table->arena, new_capacity, sizeof(*new_entries), (void**)&new_entries));
  uint8_t* new_states = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      table->arena, new_capacity, sizeof(*new_states), (void**)&new_states));

  memset(new_entries, 0, new_capacity * sizeof(*new_entries));
  memset(new_states, 0, new_capacity * sizeof(*new_states));
  if (table->count > 0) {
    memcpy(new_entries, table->entries, table->count * sizeof(*new_entries));
    memcpy(new_states, table->states, table->count * sizeof(*new_states));
  }

  table->entries = new_entries;
  table->states = new_states;
  table->count = minimum_count;
  table->capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_symbol_fact_validate_computed(
    const loom_symbol_fact_domain_t* domain, const loom_symbol_t* symbol,
    const loom_symbol_facts_base_t* facts) {
  if (!facts) return iree_ok_status();
  if (facts->domain != domain) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "symbol fact domain produced mismatched payload "
                            "domain");
  }
  if (facts->symbol_kind != symbol->kind) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "symbol fact domain produced mismatched symbol "
                            "kind %u, expected %u",
                            (uint32_t)facts->symbol_kind,
                            (uint32_t)symbol->kind);
  }
  return iree_ok_status();
}

iree_status_t loom_symbol_fact_table_lookup(
    loom_symbol_fact_table_t* table, const loom_module_t* module,
    loom_symbol_id_t symbol_id, const loom_symbol_facts_base_t** out_facts) {
  *out_facts = NULL;

  if (symbol_id == LOOM_SYMBOL_ID_INVALID ||
      symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol id %u is outside the module symbol table",
                            (uint32_t)symbol_id);
  }
  if (table->module && table->module != module) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "symbol fact table is already attached to another "
                            "module");
  }
  table->module = module;
  IREE_RETURN_IF_ERROR(
      loom_symbol_fact_table_ensure_capacity(table, module->symbols.count));

  if (table->states[symbol_id] == LOOM_SYMBOL_FACT_STATE_COMPUTED) {
    *out_facts = table->entries[symbol_id];
    return iree_ok_status();
  }
  if (table->states[symbol_id] == LOOM_SYMBOL_FACT_STATE_COMPUTING) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cycle while computing symbol facts for symbol id "
                            "%u",
                            (uint32_t)symbol_id);
  }

  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  const loom_symbol_fact_domain_t* domain =
      symbol->definition ? symbol->definition->fact_domain : NULL;
  if (!domain) {
    table->entries[symbol_id] = NULL;
    table->states[symbol_id] = LOOM_SYMBOL_FACT_STATE_COMPUTED;
    return iree_ok_status();
  }
  if (!domain->compute) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "symbol fact domain has no compute callback");
  }

  table->states[symbol_id] = LOOM_SYMBOL_FACT_STATE_COMPUTING;
  loom_symbol_fact_context_t context = {
      .table = table,
      .module = module,
  };
  const loom_symbol_facts_base_t* facts = NULL;
  iree_status_t status =
      domain->compute(domain, &context, module, symbol_id, symbol, &facts);
  if (iree_status_is_ok(status)) {
    status = loom_symbol_fact_validate_computed(domain, symbol, facts);
  }
  if (iree_status_is_ok(status)) {
    table->entries[symbol_id] = facts;
    table->states[symbol_id] = LOOM_SYMBOL_FACT_STATE_COMPUTED;
    *out_facts = facts;
  } else {
    table->entries[symbol_id] = NULL;
    table->states[symbol_id] = LOOM_SYMBOL_FACT_STATE_UNCOMPUTED;
  }
  return status;
}

iree_status_t loom_symbol_fact_table_lookup_ref(
    loom_symbol_fact_table_t* table, const loom_module_t* module,
    loom_symbol_ref_t symbol_ref, const loom_symbol_facts_base_t** out_facts) {
  *out_facts = NULL;
  if (!loom_symbol_ref_is_valid(symbol_ref)) return iree_ok_status();
  if (symbol_ref.module_id != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "cross-module symbol fact lookup is not available");
  }
  return loom_symbol_fact_table_lookup(table, module, symbol_ref.symbol_id,
                                       out_facts);
}

iree_status_t loom_symbol_fact_context_allocate(
    loom_symbol_fact_context_t* context, iree_host_size_t byte_length,
    void** out_ptr) {
  return iree_arena_allocate(context->table->arena, byte_length, out_ptr);
}

iree_status_t loom_symbol_fact_context_lookup_resource(
    loom_symbol_fact_context_t* context, const void* key,
    const void** out_value) {
  *out_value = NULL;

  loom_symbol_fact_resource_list_t resources = context->table->resources;
  if (resources.count > 0 && !resources.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol fact resource list has null values");
  }
  for (iree_host_size_t i = 0; i < resources.count; ++i) {
    if (resources.values[i].key != key) continue;
    if (!resources.values[i].value) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "symbol fact resource has null value");
    }
    *out_value = resources.values[i].value;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "symbol fact resource is not available");
}

iree_status_t loom_symbol_fact_context_lookup(
    loom_symbol_fact_context_t* context, loom_symbol_id_t symbol_id,
    const loom_symbol_facts_base_t** out_facts) {
  return loom_symbol_fact_table_lookup(context->table, context->module,
                                       symbol_id, out_facts);
}

iree_status_t loom_symbol_fact_context_lookup_ref(
    loom_symbol_fact_context_t* context, loom_symbol_ref_t symbol_ref,
    const loom_symbol_facts_base_t** out_facts) {
  return loom_symbol_fact_table_lookup_ref(context->table, context->module,
                                           symbol_ref, out_facts);
}
