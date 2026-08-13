// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/planner.h"

#include <string.h>

typedef struct loom_link_plan_bitmap_t {
  // Dense bitmap words.
  uint64_t* values;
  // Number of addressable bits.
  iree_host_size_t bit_count;
} loom_link_plan_bitmap_t;

struct loom_link_plan_t {
  // Provider index this plan selects from.
  const loom_link_module_index_t* index;
  // Host allocator for plan storage.
  iree_allocator_t allocator;
  // Planning mode determining module-level archive ownership.
  loom_link_plan_mode_t mode;
  // Live symbol selections in stable worklist order.
  struct {
    // Growable selection storage.
    loom_link_plan_symbol_t* values;
    // Number of live selections.
    iree_host_size_t count;
    // Allocated selection capacity.
    iree_host_size_t capacity;
  } symbols;
  // Dense reachability state owned by one contiguous allocation.
  struct {
    // Allocation backing all bitmap views.
    uint64_t* storage;
    // Selected module-index symbols.
    loom_link_plan_bitmap_t symbols;
    // Declarations whose compile-time provider imports were satisfied.
    loom_link_plan_bitmap_t resolved_declarations;
    // Modules whose root dependency row has been expanded.
    loom_link_plan_bitmap_t modules;
    // Implementation contracts whose provider chain has been expanded.
    loom_link_plan_bitmap_t contracts;
  } reachability;
  // Reused while filtering one declaration's candidates by bound provider.
  loom_link_plan_bitmap_t candidate_providers;
};

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static iree_string_view_t loom_link_plan_normalize_symbol_name(
    iree_string_view_t name) {
  if (iree_string_view_starts_with_char(name, '@')) {
    return iree_string_view_remove_prefix(name, 1);
  }
  return name;
}

static iree_status_t loom_link_plan_reserve_symbols(loom_link_plan_t* plan,
                                                    iree_host_size_t count) {
  if (count <= plan->symbols.capacity) {
    return iree_ok_status();
  }
  return iree_allocator_grow_array(
      plan->allocator, count, sizeof(*plan->symbols.values),
      &plan->symbols.capacity, (void**)&plan->symbols.values);
}

static iree_status_t loom_link_plan_bitmap_word_count(
    iree_host_size_t bit_count, iree_host_size_t* out_word_count) {
  *out_word_count = 0;
  if (bit_count == 0) {
    return iree_ok_status();
  }
  iree_host_size_t rounded_bit_count = 0;
  if (!iree_host_size_checked_add(bit_count, 63, &rounded_bit_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "link plan bitmap size overflow");
  }
  *out_word_count = rounded_bit_count / 64;
  return iree_ok_status();
}

static iree_status_t loom_link_plan_initialize_reachability(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options) {
  plan->reachability.symbols.bit_count =
      loom_link_module_index_symbol_count(plan->index);
  if (options && options->provider_resolver &&
      options->provider_resolver->binding_count != 0) {
    plan->reachability.resolved_declarations.bit_count =
        plan->reachability.symbols.bit_count;
  }
  plan->reachability.modules.bit_count =
      loom_link_module_index_module_count(plan->index);
  plan->reachability.contracts.bit_count =
      loom_link_module_index_contract_count(plan->index);

  iree_host_size_t symbol_word_count = 0;
  iree_host_size_t resolved_declaration_word_count = 0;
  iree_host_size_t module_word_count = 0;
  iree_host_size_t contract_word_count = 0;
  IREE_RETURN_IF_ERROR(loom_link_plan_bitmap_word_count(
      plan->reachability.symbols.bit_count, &symbol_word_count));
  IREE_RETURN_IF_ERROR(loom_link_plan_bitmap_word_count(
      plan->reachability.resolved_declarations.bit_count,
      &resolved_declaration_word_count));
  IREE_RETURN_IF_ERROR(loom_link_plan_bitmap_word_count(
      plan->reachability.modules.bit_count, &module_word_count));
  IREE_RETURN_IF_ERROR(loom_link_plan_bitmap_word_count(
      plan->reachability.contracts.bit_count, &contract_word_count));

  iree_host_size_t total_word_count = 0;
  if (!iree_host_size_checked_add(symbol_word_count,
                                  resolved_declaration_word_count,
                                  &total_word_count) ||
      !iree_host_size_checked_add(total_word_count, module_word_count,
                                  &total_word_count) ||
      !iree_host_size_checked_add(total_word_count, contract_word_count,
                                  &total_word_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "link plan bitmap size overflow");
  }
  if (total_word_count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t storage_size = 0;
  if (!iree_host_size_checked_mul(total_word_count,
                                  sizeof(*plan->reachability.storage),
                                  &storage_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "link plan bitmap size overflow");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      plan->allocator, storage_size, (void**)&plan->reachability.storage));
  memset(plan->reachability.storage, 0, storage_size);
  plan->reachability.symbols.values = plan->reachability.storage;
  plan->reachability.resolved_declarations.values =
      plan->reachability.symbols.values + symbol_word_count;
  plan->reachability.modules.values =
      plan->reachability.resolved_declarations.values +
      resolved_declaration_word_count;
  plan->reachability.contracts.values =
      plan->reachability.modules.values + module_word_count;
  return iree_ok_status();
}

static iree_status_t loom_link_plan_initialize_candidate_providers(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options) {
  if (!options || !options->provider_resolver ||
      options->provider_resolver->binding_count == 0) {
    return iree_ok_status();
  }
  plan->candidate_providers.bit_count =
      loom_link_module_index_provider_count(plan->index);
  iree_host_size_t word_count = 0;
  IREE_RETURN_IF_ERROR(loom_link_plan_bitmap_word_count(
      plan->candidate_providers.bit_count, &word_count));
  if (word_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      plan->allocator, word_count, sizeof(*plan->candidate_providers.values),
      (void**)&plan->candidate_providers.values));
  memset(plan->candidate_providers.values, 0,
         word_count * sizeof(*plan->candidate_providers.values));
  return iree_ok_status();
}

static bool loom_link_plan_bitmap_contains(
    const loom_link_plan_bitmap_t* bitmap, iree_host_size_t ordinal) {
  return ordinal < bitmap->bit_count &&
         (bitmap->values[ordinal / 64] & (UINT64_C(1) << (ordinal % 64))) != 0;
}

// Returns true when |ordinal| was already present and otherwise inserts it.
static bool loom_link_plan_bitmap_test_and_set(loom_link_plan_bitmap_t* bitmap,
                                               iree_host_size_t ordinal) {
  IREE_ASSERT_LT(ordinal, bitmap->bit_count);
  uint64_t* word = &bitmap->values[ordinal / 64];
  const uint64_t mask = UINT64_C(1) << (ordinal % 64);
  const bool was_set = (*word & mask) != 0;
  *word |= mask;
  return was_set;
}

static void loom_link_plan_bitmap_clear(loom_link_plan_bitmap_t* bitmap,
                                        iree_host_size_t ordinal) {
  bitmap->values[ordinal / 64] &= ~(UINT64_C(1) << (ordinal % 64));
}

static bool loom_link_plan_symbol_is_stripped(
    const loom_link_plan_options_t* options, const loom_link_plan_t* plan,
    const loom_link_module_index_symbol_t* symbol) {
  if (options &&
      options->test_symbol_policy == LOOM_LINK_PLAN_TEST_SYMBOL_STRIP &&
      iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_TEST_ONLY)) {
    return true;
  }
  return options && options->strip_symbol &&
         options->strip_symbol(options->strip_symbol_user_data, plan->index,
                               symbol);
}

static bool loom_link_plan_symbol_is_concrete_global(
    const loom_link_module_index_symbol_t* symbol) {
  if (symbol->identity != LOOM_LINK_SYMBOL_IDENTITY_GLOBAL) {
    return false;
  }
  if (iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_DECLARATION |
                                          LOOM_LINK_SYMBOL_FLAG_IMPORT |
                                          LOOM_LINK_SYMBOL_FLAG_CONFIG)) {
    return false;
  }
  return iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_HAS_BODY);
}

static bool loom_link_plan_symbol_is_declaration_like(
    const loom_link_module_index_symbol_t* symbol) {
  return iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_DECLARATION |
                                             LOOM_LINK_SYMBOL_FLAG_IMPORT);
}

static const loom_link_module_index_symbol_t*
loom_link_plan_find_legacy_concrete_duplicate_for_declaration(
    const loom_link_plan_t* plan,
    const loom_link_module_index_symbol_t* declaration) {
  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_lookup_name(plan->index, declaration->name);
  while (symbol) {
    if (symbol != declaration &&
        !loom_link_plan_symbol_is_declaration_like(symbol) &&
        iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_HAS_BODY)) {
      return symbol;
    }
    symbol = loom_link_module_index_next_same_name(plan->index, symbol);
  }
  return NULL;
}

static iree_status_t loom_link_plan_ambiguous_import_status(
    const loom_link_plan_t* plan,
    const loom_link_module_index_symbol_t* declaration,
    const loom_link_module_index_symbol_t* first,
    const loom_link_module_index_symbol_t* second) {
  const loom_link_module_index_module_t* first_module =
      loom_link_module_index_symbol_module(plan->index, first);
  const loom_link_module_index_module_t* second_module =
      loom_link_module_index_symbol_module(plan->index, second);
  const loom_link_module_index_provider_t* first_provider =
      loom_link_module_index_symbol_provider(plan->index, first);
  const loom_link_module_index_provider_t* second_provider =
      loom_link_module_index_symbol_provider(plan->index, second);
  return iree_make_status(
      IREE_STATUS_ALREADY_EXISTS,
      "provider imports for declaration '@%.*s' resolve to both provider "
      "'%.*s' module '%.*s' and provider '%.*s' module '%.*s'",
      (int)declaration->name.size, declaration->name.data,
      (int)first_provider->name.size, first_provider->name.data,
      (int)first_module->name.size, first_module->name.data,
      (int)second_provider->name.size, second_provider->name.data,
      (int)second_module->name.size, second_module->name.data);
}

static iree_host_size_t loom_link_plan_resolve_provider_import(
    const loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_module_t* module,
    uint32_t provider_import_ordinal) {
  if (!options || !options->provider_resolver) {
    return LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  }
  const iree_string_view_t key = loom_link_module_index_provider_import_key_at(
      plan->index, module, provider_import_ordinal);
  return loom_link_provider_resolver_lookup(options->provider_resolver, key);
}

static iree_status_t
loom_link_plan_find_imported_concrete_definition_for_declaration(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_symbol_t* declaration,
    loom_link_module_index_provider_import_list_t imports,
    const loom_link_module_index_symbol_t** out_concrete_symbol) {
  *out_concrete_symbol = NULL;
  const loom_link_module_index_module_t* declaration_module =
      loom_link_module_index_symbol_module(plan->index, declaration);

  bool has_bound_provider = false;
  for (iree_host_size_t i = 0; i < imports.count; ++i) {
    const iree_host_size_t provider_ordinal =
        loom_link_plan_resolve_provider_import(
            plan, options, declaration_module, imports.values[i]);
    if (provider_ordinal == LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
      continue;
    }
    loom_link_plan_bitmap_test_and_set(&plan->candidate_providers,
                                       provider_ordinal);
    has_bound_provider = true;
  }

  const loom_link_module_index_symbol_t* concrete_symbol = NULL;
  const loom_link_module_index_symbol_t* ambiguous_symbol = NULL;
  if (has_bound_provider) {
    const loom_link_module_index_symbol_t* candidate =
        loom_link_module_index_lookup_name(plan->index, declaration->name);
    while (candidate) {
      const loom_link_module_index_module_t* candidate_module =
          loom_link_module_index_symbol_module(plan->index, candidate);
      const bool provider_is_bound = loom_link_plan_bitmap_contains(
          &plan->candidate_providers, candidate_module->provider_ordinal);
      const bool candidate_is_concrete =
          candidate != declaration &&
          !loom_link_plan_symbol_is_declaration_like(candidate) &&
          iree_any_bit_set(candidate->flags, LOOM_LINK_SYMBOL_FLAG_HAS_BODY) &&
          !loom_link_plan_symbol_is_stripped(options, plan, candidate);
      if (provider_is_bound && candidate_is_concrete) {
        if (concrete_symbol) {
          ambiguous_symbol = candidate;
          break;
        } else {
          concrete_symbol = candidate;
        }
      }
      candidate = loom_link_module_index_next_same_name(plan->index, candidate);
    }
  }

  for (iree_host_size_t i = 0; i < imports.count; ++i) {
    const iree_host_size_t provider_ordinal =
        loom_link_plan_resolve_provider_import(
            plan, options, declaration_module, imports.values[i]);
    if (provider_ordinal != LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
      loom_link_plan_bitmap_clear(&plan->candidate_providers, provider_ordinal);
    }
  }
  if (ambiguous_symbol) {
    return loom_link_plan_ambiguous_import_status(
        plan, declaration, concrete_symbol, ambiguous_symbol);
  }

  if (!concrete_symbol && (!options || options->unresolved_policy ==
                                           LOOM_LINK_PLAN_UNRESOLVED_ERROR)) {
    const iree_string_view_t first_key =
        loom_link_module_index_provider_import_key_at(
            plan->index, declaration_module, imports.values[0]);
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "provider import '%.*s' for declaration '@%.*s' has no available "
        "concrete definition across %" PRIhsz " source alternatives",
        (int)first_key.size, first_key.data, (int)declaration->name.size,
        declaration->name.data, imports.count);
  }

  *out_concrete_symbol = concrete_symbol;
  return iree_ok_status();
}

static iree_status_t loom_link_plan_check_concrete_global_collision(
    const loom_link_plan_t* plan,
    const loom_link_module_index_symbol_t* symbol) {
  if (!loom_link_plan_symbol_is_concrete_global(symbol)) {
    return iree_ok_status();
  }
  const loom_link_module_index_symbol_t* candidate =
      loom_link_module_index_lookup_name(plan->index, symbol->name);
  while (candidate) {
    if (candidate != symbol &&
        loom_link_plan_symbol_is_concrete_global(candidate) &&
        loom_link_plan_bitmap_contains(&plan->reachability.symbols,
                                       candidate->ordinal)) {
      return loom_link_module_index_duplicate_global_status(plan->index,
                                                            candidate, symbol);
    }
    candidate = loom_link_module_index_next_same_name(plan->index, candidate);
  }
  return iree_ok_status();
}

// Appends |symbol| if it has not already been selected. |out_plan_ordinal| is
// set only for a new selection so callers never need a reverse ordinal map.
static iree_status_t loom_link_plan_append_symbol(
    loom_link_plan_t* plan, const loom_link_module_index_symbol_t* symbol,
    loom_link_plan_live_reason_t reason, iree_host_size_t cause_ordinal,
    iree_string_view_t root_name, iree_host_size_t* out_plan_ordinal) {
  if (out_plan_ordinal) {
    *out_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  }
  if (loom_link_plan_bitmap_contains(&plan->reachability.symbols,
                                     symbol->ordinal)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_link_plan_check_concrete_global_collision(plan, symbol));

  const iree_host_size_t plan_ordinal = plan->symbols.count;
  iree_host_size_t symbol_count = 0;
  if (!iree_host_size_checked_add(plan_ordinal, 1, &symbol_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "link plan symbol count overflow");
  }
  IREE_RETURN_IF_ERROR(loom_link_plan_reserve_symbols(plan, symbol_count));
  plan->symbols.values[plan_ordinal] = (loom_link_plan_symbol_t){
      .ordinal = plan_ordinal,
      .symbol_ordinal = symbol->ordinal,
      .reason = reason,
      .cause_ordinal = cause_ordinal,
      .root_name = root_name,
  };
  loom_link_plan_bitmap_test_and_set(&plan->reachability.symbols,
                                     symbol->ordinal);
  plan->symbols.count = symbol_count;
  if (out_plan_ordinal) {
    *out_plan_ordinal = plan_ordinal;
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_select_required_symbol(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_symbol_t* symbol,
    loom_link_plan_live_reason_t reason, iree_host_size_t cause_ordinal,
    iree_string_view_t root_name, iree_host_size_t* out_plan_ordinal) {
  if (out_plan_ordinal) {
    *out_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  }
  if (loom_link_plan_symbol_is_stripped(options, plan, symbol)) {
    if (options &&
        options->unresolved_policy == LOOM_LINK_PLAN_UNRESOLVED_ALLOW) {
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "required symbol '@%.*s' was stripped",
                            (int)symbol->name.size, symbol->name.data);
  }
  return loom_link_plan_append_symbol(plan, symbol, reason, cause_ordinal,
                                      root_name, out_plan_ordinal);
}

static iree_status_t loom_link_plan_select_declaration_definition(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_symbol_t* declaration,
    iree_host_size_t declaration_plan_ordinal) {
  if (!loom_link_plan_symbol_is_declaration_like(declaration)) {
    return iree_ok_status();
  }
  const loom_link_module_index_provider_import_list_t imports =
      loom_link_module_index_symbol_provider_imports(plan->index, declaration);
  const loom_link_module_index_symbol_t* concrete_symbol = NULL;
  if (imports.count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_link_plan_find_imported_concrete_definition_for_declaration(
            plan, options, declaration, imports, &concrete_symbol));
  } else {
    concrete_symbol =
        loom_link_plan_find_legacy_concrete_duplicate_for_declaration(
            plan, declaration);
  }
  if (!concrete_symbol) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_link_plan_select_required_symbol(
      plan, options, concrete_symbol, LOOM_LINK_PLAN_LIVE_DEPENDENCY,
      declaration_plan_ordinal, iree_string_view_empty(),
      /*out_plan_ordinal=*/NULL));
  if (imports.count != 0) {
    loom_link_plan_bitmap_test_and_set(
        &plan->reachability.resolved_declarations, declaration->ordinal);
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_select_global_reference(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_symbol_t* referenced_symbol,
    loom_link_plan_live_reason_t reason, iree_host_size_t cause_ordinal,
    iree_string_view_t root_name) {
  const loom_link_module_index_provider_import_list_t imports =
      loom_link_module_index_symbol_provider_imports(plan->index,
                                                     referenced_symbol);
  const bool has_exact_provider_imports =
      imports.count != 0 &&
      loom_link_plan_symbol_is_declaration_like(referenced_symbol);
  const loom_link_module_index_symbol_t* selected_symbol =
      has_exact_provider_imports ? referenced_symbol
                                 : loom_link_module_index_lookup_global(
                                       plan->index, referenced_symbol->name);
  if (!selected_symbol) {
    if (options &&
        options->unresolved_policy == LOOM_LINK_PLAN_UNRESOLVED_ALLOW) {
      return iree_ok_status();
    }
    return iree_make_status(
        IREE_STATUS_NOT_FOUND, "unresolved global dependency '@%.*s'",
        (int)referenced_symbol->name.size, referenced_symbol->name.data);
  }

  iree_host_size_t selected_plan_ordinal =
      LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  IREE_RETURN_IF_ERROR(loom_link_plan_select_required_symbol(
      plan, options, selected_symbol, reason, cause_ordinal, root_name,
      &selected_plan_ordinal));
  if (selected_plan_ordinal == LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    return iree_ok_status();
  }
  return loom_link_plan_select_declaration_definition(
      plan, options, selected_symbol, selected_plan_ordinal);
}

//===----------------------------------------------------------------------===//
// Closure expansion
//===----------------------------------------------------------------------===//

static iree_status_t loom_link_plan_select_dependency_target(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_module_t* module, uint32_t target_symbol_id,
    iree_host_size_t cause_ordinal) {
  IREE_ASSERT_LT(target_symbol_id, module->symbol_count);
  const loom_link_module_index_symbol_t* target =
      loom_link_module_index_symbol_at(
          plan->index, module->symbol_start_ordinal + target_symbol_id);
  IREE_ASSERT(target);
  if (target->identity == LOOM_LINK_SYMBOL_IDENTITY_GLOBAL) {
    return loom_link_plan_select_global_reference(
        plan, options, target, LOOM_LINK_PLAN_LIVE_DEPENDENCY, cause_ordinal,
        iree_string_view_empty());
  }
  return loom_link_plan_select_required_symbol(
      plan, options, target, LOOM_LINK_PLAN_LIVE_DEPENDENCY, cause_ordinal,
      iree_string_view_empty(), /*out_plan_ordinal=*/NULL);
}

static iree_status_t loom_link_plan_expand_dependency_slice(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_module_t* module, uint32_t first,
    uint32_t count, iree_host_size_t cause_ordinal) {
  if (count == 0) {
    return iree_ok_status();
  }
  const uint32_t* dependencies = module->dependencies.values + first;
  for (uint32_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_link_plan_select_dependency_target(
        plan, options, module, dependencies[i], cause_ordinal));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_expand_contract(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    loom_link_contract_ordinal_t contract_ordinal,
    iree_host_size_t cause_ordinal) {
  if (loom_link_plan_bitmap_test_and_set(&plan->reachability.contracts,
                                         contract_ordinal)) {
    return iree_ok_status();
  }
  const loom_link_module_index_contract_t* contract =
      loom_link_module_index_contract_at(plan->index, contract_ordinal);
  IREE_ASSERT(contract);
  iree_host_size_t provider_ordinal = contract->providers.first_symbol_ordinal;
  while (provider_ordinal != LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    const loom_link_module_index_symbol_t* provider =
        loom_link_module_index_symbol_at(plan->index, provider_ordinal);
    IREE_ASSERT(provider);
    IREE_RETURN_IF_ERROR(loom_link_plan_select_required_symbol(
        plan, options, provider, LOOM_LINK_PLAN_LIVE_PROVIDER, cause_ordinal,
        iree_string_view_empty(), /*out_plan_ordinal=*/NULL));
    provider_ordinal = provider->next.contract_provider_ordinal;
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_expand_contract_slice(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_module_t* module, uint32_t first,
    uint32_t count, iree_host_size_t cause_ordinal) {
  if (count == 0) {
    return iree_ok_status();
  }
  const loom_link_contract_ordinal_t* contracts =
      module->contract_demands.values + first;
  for (uint32_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loom_link_plan_expand_contract(
        plan, options, contracts[i], cause_ordinal));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_expand_symbol(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    iree_host_size_t plan_ordinal) {
  const iree_host_size_t symbol_ordinal =
      plan->symbols.values[plan_ordinal].symbol_ordinal;
  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_symbol_at(plan->index, symbol_ordinal);
  const loom_link_module_index_module_t* module =
      loom_link_module_index_symbol_module(plan->index, symbol);
  IREE_ASSERT(symbol);
  IREE_ASSERT(module);

  if (!loom_link_plan_bitmap_test_and_set(&plan->reachability.modules,
                                          module->ordinal)) {
    IREE_RETURN_IF_ERROR(loom_link_plan_expand_dependency_slice(
        plan, options, module, /*first=*/0, module->dependencies.root_count,
        plan_ordinal));
  }
  IREE_RETURN_IF_ERROR(loom_link_plan_expand_dependency_slice(
      plan, options, module, symbol->dependencies.first,
      symbol->dependencies.count, plan_ordinal));
  return loom_link_plan_expand_contract_slice(
      plan, options, module, symbol->contract_demands.first,
      symbol->contract_demands.count, plan_ordinal);
}

//===----------------------------------------------------------------------===//
// Selection modes
//===----------------------------------------------------------------------===//

static iree_status_t loom_link_plan_select_archive(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options) {
  const iree_host_size_t symbol_count =
      loom_link_module_index_symbol_count(plan->index);
  for (iree_host_size_t i = 0; i < symbol_count; ++i) {
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(plan->index, i);
    if (loom_link_plan_symbol_is_stripped(options, plan, symbol)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_link_plan_append_symbol(
        plan, symbol, LOOM_LINK_PLAN_LIVE_ARCHIVE,
        LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL, iree_string_view_empty(),
        /*out_plan_ordinal=*/NULL));
  }
  for (iree_host_size_t i = 0; i < plan->symbols.count; ++i) {
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(
            plan->index, plan->symbols.values[i].symbol_ordinal);
    IREE_RETURN_IF_ERROR(
        loom_link_plan_select_declaration_definition(plan, options, symbol, i));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_select_root(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    iree_string_view_t root_name) {
  const iree_string_view_t normalized_name =
      loom_link_plan_normalize_symbol_name(root_name);
  if (iree_string_view_is_empty(normalized_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "root symbol name must not be empty");
  }
  const loom_link_module_index_symbol_t* root =
      loom_link_module_index_lookup_global(plan->index, normalized_name);
  if (root) {
    return loom_link_plan_select_global_reference(
        plan, options, root, LOOM_LINK_PLAN_LIVE_ROOT,
        LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL, root_name);
  }

  const loom_link_module_index_symbol_t* private_root = NULL;
  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_lookup_name(plan->index, normalized_name);
  while (symbol) {
    if (symbol->identity == LOOM_LINK_SYMBOL_IDENTITY_PRIVATE) {
      if (private_root) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "root symbol '@%.*s' is private in multiple modules",
            (int)normalized_name.size, normalized_name.data);
      }
      private_root = symbol;
    }
    symbol = loom_link_module_index_next_same_name(plan->index, symbol);
  }
  if (!private_root) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "root symbol '@%.*s' was not found",
                            (int)normalized_name.size, normalized_name.data);
  }
  return loom_link_plan_select_required_symbol(
      plan, options, private_root, LOOM_LINK_PLAN_LIVE_ROOT,
      LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL, root_name,
      /*out_plan_ordinal=*/NULL);
}

static iree_status_t loom_link_plan_select_exported_roots(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options) {
  const iree_host_size_t symbol_count =
      loom_link_module_index_symbol_count(plan->index);
  for (iree_host_size_t i = 0; i < symbol_count; ++i) {
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(plan->index, i);
    if (!iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_EXPORT)) {
      continue;
    }
    iree_host_size_t root_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    IREE_RETURN_IF_ERROR(loom_link_plan_select_required_symbol(
        plan, options, symbol, LOOM_LINK_PLAN_LIVE_ROOT,
        LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL, symbol->name,
        &root_plan_ordinal));
    if (root_plan_ordinal != LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
      IREE_RETURN_IF_ERROR(loom_link_plan_select_declaration_definition(
          plan, options, symbol, root_plan_ordinal));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_select_roots(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options) {
  if (options && options->include_exported_roots) {
    IREE_RETURN_IF_ERROR(loom_link_plan_select_exported_roots(plan, options));
  }
  const iree_host_size_t root_count = options ? options->root_symbols.count : 0;
  if (root_count > 0 && (!options || !options->root_symbols.values)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "root_symbols count is non-zero but values is NULL");
  }
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_link_plan_select_root(
        plan, options, options->root_symbols.values[i]));
  }
  if (plan->symbols.count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selective link planning requires at least one root");
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_select_selective(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options) {
  IREE_RETURN_IF_ERROR(loom_link_plan_select_roots(plan, options));
  for (iree_host_size_t i = 0; i < plan->symbols.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_link_plan_expand_symbol(plan, options, i));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t loom_link_plan_build(const loom_link_module_index_t* index,
                                   const loom_link_plan_options_t* options,
                                   iree_allocator_t allocator,
                                   loom_link_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = NULL;

  loom_link_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*plan), (void**)&plan));
  memset(plan, 0, sizeof(*plan));
  plan->index = index;
  plan->allocator = allocator;
  plan->mode = options ? options->mode : LOOM_LINK_PLAN_ARCHIVE;

  iree_status_t status = loom_link_plan_initialize_reachability(plan, options);
  if (iree_status_is_ok(status)) {
    status = loom_link_plan_initialize_candidate_providers(plan, options);
  }
  if (iree_status_is_ok(status)) {
    switch (plan->mode) {
      case LOOM_LINK_PLAN_ARCHIVE:
        status = loom_link_plan_select_archive(plan, options);
        break;
      case LOOM_LINK_PLAN_SELECTIVE:
        status = loom_link_plan_select_selective(plan, options);
        break;
      default:
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "unknown link plan mode %u", (unsigned)plan->mode);
        break;
    }
  }

  iree_allocator_free(plan->allocator, plan->candidate_providers.values);
  plan->candidate_providers = (loom_link_plan_bitmap_t){0};

  if (!iree_status_is_ok(status)) {
    loom_link_plan_free(plan);
    return status;
  }
  *out_plan = plan;
  return iree_ok_status();
}

void loom_link_plan_free(loom_link_plan_t* plan) {
  if (!plan) {
    return;
  }
  iree_allocator_free(plan->allocator, plan->candidate_providers.values);
  iree_allocator_free(plan->allocator, plan->reachability.storage);
  iree_allocator_free(plan->allocator, plan->symbols.values);
  iree_allocator_free(plan->allocator, plan);
}

const loom_link_module_index_t* loom_link_plan_index(
    const loom_link_plan_t* plan) {
  return plan ? plan->index : NULL;
}

loom_link_plan_mode_t loom_link_plan_mode(const loom_link_plan_t* plan) {
  return plan ? plan->mode : LOOM_LINK_PLAN_ARCHIVE;
}

iree_host_size_t loom_link_plan_symbol_count(const loom_link_plan_t* plan) {
  return plan ? plan->symbols.count : 0;
}

const loom_link_plan_symbol_t* loom_link_plan_symbol_at(
    const loom_link_plan_t* plan, iree_host_size_t ordinal) {
  if (!plan || ordinal >= plan->symbols.count) {
    return NULL;
  }
  return &plan->symbols.values[ordinal];
}

bool loom_link_plan_contains_symbol(const loom_link_plan_t* plan,
                                    iree_host_size_t symbol_ordinal) {
  return plan && loom_link_plan_bitmap_contains(&plan->reachability.symbols,
                                                symbol_ordinal);
}

bool loom_link_plan_symbol_imports_resolved(const loom_link_plan_t* plan,
                                            iree_host_size_t symbol_ordinal) {
  return plan && loom_link_plan_bitmap_contains(
                     &plan->reachability.resolved_declarations, symbol_ordinal);
}
