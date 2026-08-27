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

typedef struct loom_link_plan_symbol_ordinal_map_entry_t {
  // Index-wide symbol ordinal, or INVALID_ORDINAL for an empty slot.
  iree_host_size_t symbol_ordinal;
  // Plan-local symbol ordinal associated with symbol_ordinal.
  iree_host_size_t plan_ordinal;
} loom_link_plan_symbol_ordinal_map_entry_t;

typedef struct loom_link_plan_live_cause_t {
  // Why the selected symbol or facet is live.
  loom_link_plan_live_reason_t reason;
  // Plan-local symbol ordinal causing this selection, or INVALID_ORDINAL.
  iree_host_size_t symbol_plan_ordinal;
  // Plan-local facet ordinal causing this selection, or INVALID_ORDINAL.
  iree_host_size_t facet_plan_ordinal;
  // Authored root name when reason is ROOT.
  iree_string_view_t root_name;
} loom_link_plan_live_cause_t;

typedef enum loom_link_plan_facet_request_mode_e {
  // Select the symbol's primary contract/definition facet only.
  LOOM_LINK_PLAN_FACET_REQUEST_PRIMARY = 0,
  // Select the primary facet and one exact semantic facet.
  LOOM_LINK_PLAN_FACET_REQUEST_EXACT = 1,
  // Select every semantic facet.
  LOOM_LINK_PLAN_FACET_REQUEST_COMPLETE = 2,
} loom_link_plan_facet_request_mode_t;

typedef struct loom_link_plan_facet_request_t {
  // Selection mode defining the requested structural projection.
  loom_link_plan_facet_request_mode_t mode;
  // Semantic facet for EXACT mode.
  loom_link_symbol_facet_kind_t kind;
} loom_link_plan_facet_request_t;

// Definition resolution has not been attempted for this selected symbol.
#define LOOM_LINK_PLAN_DEFINITION_UNCHECKED_ORDINAL \
  LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL
// Definition resolution completed without selecting a concrete definition.
#define LOOM_LINK_PLAN_DEFINITION_ABSENT_ORDINAL \
  (LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL - 1)

typedef struct loom_link_plan_symbol_work_item_t {
  // Public selected-symbol record.
  loom_link_plan_symbol_t selection;
  // Last selected facet for O(1) append to this symbol's facet chain.
  iree_host_size_t last_facet_plan_ordinal;
  // Index-wide concrete definition symbol ordinal, UNCHECKED, or ABSENT.
  iree_host_size_t resolved_definition_symbol_ordinal;
} loom_link_plan_symbol_work_item_t;

typedef enum loom_link_plan_facet_expansion_mode_e {
  // This selected facet is accounted for by another complete-symbol item.
  LOOM_LINK_PLAN_FACET_EXPANSION_NONE = 0,
  // Expand only references originating in this selected facet.
  LOOM_LINK_PLAN_FACET_EXPANSION_EXACT = 1,
  // Expand every selected facet of the symbol in one dependency pass.
  LOOM_LINK_PLAN_FACET_EXPANSION_COMPLETE = 2,
} loom_link_plan_facet_expansion_mode_t;

typedef struct loom_link_plan_facet_work_item_t {
  // Public selected-facet record.
  loom_link_plan_facet_t selection;
  // Closure expansion performed when this work item is reached.
  loom_link_plan_facet_expansion_mode_t expansion_mode;
  // Next selected facet of the same symbol, or INVALID_ORDINAL.
  iree_host_size_t next_symbol_facet_plan_ordinal;
} loom_link_plan_facet_work_item_t;

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
    loom_link_plan_symbol_work_item_t* values;
    // Number of live selections.
    iree_host_size_t count;
    // Allocated selection capacity.
    iree_host_size_t capacity;
  } symbols;
  // Live facet selections in stable worklist order.
  struct {
    // Growable facet selection storage.
    loom_link_plan_facet_work_item_t* values;
    // Number of live facet selections.
    iree_host_size_t count;
    // Allocated facet selection capacity.
    iree_host_size_t capacity;
  } facets;
  // Reachable template families in stable first-demand order.
  struct {
    // Growable dense family ordinal storage.
    loom_link_template_family_ordinal_t* values;
    // Number of reachable demand occurrences, including repeated families.
    iree_host_size_t occurrence_count;
    // Number of unique demanded families.
    iree_host_size_t count;
    // Allocated family ordinal capacity.
    iree_host_size_t capacity;
  } demanded_template_families;
  // Dense reachability state owned by one contiguous allocation.
  struct {
    // Allocation backing all bitmap views.
    uint64_t* storage;
    // Selected module-index symbols.
    loom_link_plan_bitmap_t symbols;
    // Modules whose root dependency row has been expanded.
    loom_link_plan_bitmap_t modules;
    // Template families already recorded as demanded.
    loom_link_plan_bitmap_t template_families;
  } reachability;
  // Reverse map allocated only when the plan contains a partial symbol.
  struct {
    // Open-addressed symbol-to-plan entries.
    loom_link_plan_symbol_ordinal_map_entry_t* values;
    // Power-of-two slot capacity.
    iree_host_size_t capacity;
  } symbol_ordinals;
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

static iree_status_t loom_link_plan_reserve_facets(loom_link_plan_t* plan,
                                                   iree_host_size_t count) {
  if (count <= plan->facets.capacity) {
    return iree_ok_status();
  }
  return iree_allocator_grow_array(
      plan->allocator, count, sizeof(*plan->facets.values),
      &plan->facets.capacity, (void**)&plan->facets.values);
}

static iree_status_t loom_link_plan_reserve_demanded_template_families(
    loom_link_plan_t* plan, iree_host_size_t count) {
  if (count <= plan->demanded_template_families.capacity) {
    return iree_ok_status();
  }
  return iree_allocator_grow_array(
      plan->allocator, count, sizeof(*plan->demanded_template_families.values),
      &plan->demanded_template_families.capacity,
      (void**)&plan->demanded_template_families.values);
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
    loom_link_plan_t* plan) {
  plan->reachability.symbols.bit_count =
      loom_link_module_index_symbol_count(plan->index);
  plan->reachability.modules.bit_count =
      loom_link_module_index_module_count(plan->index);
  plan->reachability.template_families.bit_count =
      loom_link_module_index_template_family_count(plan->index);

  iree_host_size_t symbol_word_count = 0;
  iree_host_size_t module_word_count = 0;
  iree_host_size_t family_word_count = 0;
  IREE_RETURN_IF_ERROR(loom_link_plan_bitmap_word_count(
      plan->reachability.symbols.bit_count, &symbol_word_count));
  IREE_RETURN_IF_ERROR(loom_link_plan_bitmap_word_count(
      plan->reachability.modules.bit_count, &module_word_count));
  IREE_RETURN_IF_ERROR(loom_link_plan_bitmap_word_count(
      plan->reachability.template_families.bit_count, &family_word_count));

  iree_host_size_t total_word_count = 0;
  if (!iree_host_size_checked_add(symbol_word_count, module_word_count,
                                  &total_word_count) ||
      !iree_host_size_checked_add(total_word_count, family_word_count,
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
  plan->reachability.modules.values =
      plan->reachability.symbols.values + symbol_word_count;
  plan->reachability.template_families.values =
      plan->reachability.modules.values + module_word_count;
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

static iree_host_size_t loom_link_plan_hash_symbol_ordinal(
    iree_host_size_t symbol_ordinal) {
  uint64_t value = (uint64_t)symbol_ordinal;
  value ^= value >> 30;
  value *= UINT64_C(0xBF58476D1CE4E5B9);
  value ^= value >> 27;
  value *= UINT64_C(0x94D049BB133111EB);
  value ^= value >> 31;
  return (iree_host_size_t)value;
}

static iree_host_size_t loom_link_plan_symbol_ordinal_map_slot(
    const loom_link_plan_symbol_ordinal_map_entry_t* values,
    iree_host_size_t capacity, iree_host_size_t symbol_ordinal) {
  iree_host_size_t slot =
      loom_link_plan_hash_symbol_ordinal(symbol_ordinal) & (capacity - 1);
  while (values[slot].symbol_ordinal !=
             LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL &&
         values[slot].symbol_ordinal != symbol_ordinal) {
    slot = (slot + 1) & (capacity - 1);
  }
  return slot;
}

static void loom_link_plan_insert_symbol_ordinal(
    loom_link_plan_symbol_ordinal_map_entry_t* values,
    iree_host_size_t capacity, iree_host_size_t symbol_ordinal,
    iree_host_size_t plan_ordinal) {
  const iree_host_size_t slot =
      loom_link_plan_symbol_ordinal_map_slot(values, capacity, symbol_ordinal);
  IREE_ASSERT_EQ(values[slot].symbol_ordinal,
                 LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL);
  values[slot] = (loom_link_plan_symbol_ordinal_map_entry_t){
      .symbol_ordinal = symbol_ordinal,
      .plan_ordinal = plan_ordinal,
  };
}

static iree_status_t loom_link_plan_reserve_symbol_ordinals(
    loom_link_plan_t* plan, iree_host_size_t required_count) {
  if (plan->symbol_ordinals.capacity / 2 >= required_count) {
    return iree_ok_status();
  }
  iree_host_size_t capacity = plan->symbol_ordinals.capacity;
  if (capacity == 0) {
    capacity = 8;
  }
  while (capacity / 2 < required_count) {
    if (capacity > IREE_HOST_SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "link plan symbol ordinal map overflow");
    }
    capacity *= 2;
  }

  loom_link_plan_symbol_ordinal_map_entry_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      plan->allocator, capacity, sizeof(*values), (void**)&values));
  for (iree_host_size_t i = 0; i < capacity; ++i) {
    values[i].symbol_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  }
  for (iree_host_size_t i = 0; i < plan->symbols.count; ++i) {
    loom_link_plan_insert_symbol_ordinal(
        values, capacity, plan->symbols.values[i].selection.symbol_ordinal, i);
  }
  iree_allocator_free(plan->allocator, plan->symbol_ordinals.values);
  plan->symbol_ordinals.values = values;
  plan->symbol_ordinals.capacity = capacity;
  return iree_ok_status();
}

static iree_status_t loom_link_plan_lookup_symbol_ordinal(
    loom_link_plan_t* plan, iree_host_size_t symbol_ordinal,
    iree_host_size_t* out_plan_ordinal) {
  IREE_RETURN_IF_ERROR(
      loom_link_plan_reserve_symbol_ordinals(plan, plan->symbols.count));
  const iree_host_size_t slot = loom_link_plan_symbol_ordinal_map_slot(
      plan->symbol_ordinals.values, plan->symbol_ordinals.capacity,
      symbol_ordinal);
  IREE_ASSERT_EQ(plan->symbol_ordinals.values[slot].symbol_ordinal,
                 symbol_ordinal);
  *out_plan_ordinal = plan->symbol_ordinals.values[slot].plan_ordinal;
  return iree_ok_status();
}

static bool loom_link_plan_try_lookup_symbol_ordinal(
    const loom_link_plan_t* plan, iree_host_size_t symbol_ordinal,
    iree_host_size_t* out_plan_ordinal) {
  if (plan->symbol_ordinals.values == NULL) {
    return false;
  }
  const iree_host_size_t slot = loom_link_plan_symbol_ordinal_map_slot(
      plan->symbol_ordinals.values, plan->symbol_ordinals.capacity,
      symbol_ordinal);
  if (plan->symbol_ordinals.values[slot].symbol_ordinal != symbol_ordinal) {
    return false;
  }
  *out_plan_ordinal = plan->symbol_ordinals.values[slot].plan_ordinal;
  return true;
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

// Returns true when the selected product may retain an unresolved contract.
// Archive plans form relocatable modules and therefore always preserve
// declarations. Selective plans are closed unless their caller explicitly
// requests another relocatable output.
static bool loom_link_plan_allows_unresolved(
    const loom_link_plan_t* plan, const loom_link_plan_options_t* options) {
  return plan->mode == LOOM_LINK_PLAN_ARCHIVE ||
         (options &&
          options->unresolved_policy == LOOM_LINK_PLAN_UNRESOLVED_ALLOW);
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
  return iree_any_bit_set(symbol->flags,
                          LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION);
}

static iree_status_t loom_link_plan_global_collision_status(
    const loom_link_plan_t* plan,
    const loom_link_module_index_symbol_t* selected,
    const loom_link_module_index_symbol_t* duplicate) {
  return loom_link_module_index_annotate_global_collision(
      iree_status_from_code(IREE_STATUS_ALREADY_EXISTS), plan->index, selected,
      duplicate);
}

static bool loom_link_plan_symbol_is_declaration_like(
    const loom_link_module_index_symbol_t* symbol) {
  return iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_DECLARATION |
                                             LOOM_LINK_SYMBOL_FLAG_IMPORT);
}

// Returns true when |declaration| is an ordinary compile-time symbol contract.
// Other declaration families have dedicated owners: runtime imports remain
// external, templates use provider selection, and config/target/artifact
// requirements are supplied by later compilation and emission stages.
static bool loom_link_plan_symbol_is_exact_declaration(
    const loom_link_module_index_symbol_t* declaration) {
  return loom_link_plan_symbol_is_declaration_like(declaration) &&
         !iree_any_bit_set(declaration->flags, LOOM_LINK_SYMBOL_FLAG_IMPORT) &&
         iree_any_bit_set(declaration->facets.schema.interfaces,
                          LOOM_SYMBOL_INTERFACE_FUNC_LIKE) &&
         declaration->template_family_ordinal ==
             LOOM_LINK_TEMPLATE_FAMILY_ORDINAL_INVALID;
}

// Returns true when |candidate| can structurally provide |declaration|. This
// broad index-level filter avoids materializing unrelated same-name symbol
// families; full contract compatibility is checked while merging the selected
// definition.
static bool loom_link_plan_symbol_satisfies_declaration_interface(
    const loom_link_module_index_symbol_t* declaration,
    const loom_link_module_index_symbol_t* candidate) {
  return candidate != declaration &&
         !loom_link_plan_symbol_is_declaration_like(candidate) &&
         iree_any_bit_set(candidate->flags,
                          LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION) &&
         iree_all_bits_set(candidate->facets.schema.interfaces,
                           declaration->facets.schema.interfaces);
}

// Returns true when |declaration| names a target context that may be supplied
// either by an explicit target record in the link environment or by a later
// target-profile binding. Target requirements are not ordinary library edges.
static bool loom_link_plan_symbol_is_target_declaration(
    const loom_link_module_index_symbol_t* declaration) {
  return loom_link_plan_symbol_is_declaration_like(declaration) &&
         iree_any_bit_set(declaration->facets.schema.interfaces,
                          LOOM_SYMBOL_INTERFACE_TARGET);
}

// Returns true when |candidate| is owned by the same source merge as
// |declaration|. Each provider is one ownership unit and all INPUT providers
// jointly represent the direct sources of the module being assembled.
static bool loom_link_plan_symbols_share_merge_ownership(
    const loom_link_plan_t* plan,
    const loom_link_module_index_symbol_t* declaration,
    const loom_link_module_index_symbol_t* candidate) {
  const loom_link_module_index_provider_t* declaration_provider =
      loom_link_module_index_symbol_provider(plan->index, declaration);
  const loom_link_module_index_provider_t* candidate_provider =
      loom_link_module_index_symbol_provider(plan->index, candidate);
  IREE_ASSERT(declaration_provider);
  IREE_ASSERT(candidate_provider);
  return declaration_provider == candidate_provider ||
         (declaration_provider->role == LOOM_LINK_PROVIDER_ROLE_INPUT &&
          candidate_provider->role == LOOM_LINK_PROVIDER_ROLE_INPUT);
}

// Returns true when |request| consumes an implementation-bearing facet. A
// kernel or command contract can remain as a declaration in a closed portable
// product; configuration, implementation, and ordinary definition facets
// require a concrete provider.
static bool loom_link_plan_facet_request_requires_definition(
    const loom_link_module_index_symbol_t* declaration,
    loom_link_plan_facet_request_t request) {
  if (request.mode == LOOM_LINK_PLAN_FACET_REQUEST_COMPLETE) {
    return true;
  }
  if (request.mode == LOOM_LINK_PLAN_FACET_REQUEST_PRIMARY) {
    return false;
  }
  return request.kind != LOOM_LINK_SYMBOL_FACET_KERNEL_CONTRACT &&
         request.kind != LOOM_LINK_SYMBOL_FACET_COMMAND_CONTRACT;
}

static iree_status_t loom_link_plan_unresolved_exact_declaration_status(
    const loom_link_module_index_symbol_t* declaration) {
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "unresolved exact declaration '@%.*s'",
                          (int)declaration->name.size, declaration->name.data);
}

static iree_status_t loom_link_plan_find_exact_definition_for_declaration(
    const loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_symbol_t* declaration,
    loom_link_plan_facet_request_t definition_request,
    const loom_link_module_index_symbol_t** out_selected_symbol) {
  *out_selected_symbol = NULL;
  const loom_link_module_index_symbol_t* owner_definition = NULL;
  const loom_link_module_index_symbol_t* duplicate_owner_definition = NULL;
  const loom_link_module_index_symbol_t* external_definition = NULL;
  const loom_link_module_index_symbol_t* duplicate_external_definition = NULL;
  const loom_link_module_index_symbol_t* candidate =
      loom_link_module_index_lookup_name(plan->index, declaration->name);
  while (candidate) {
    const bool is_concrete_definition =
        loom_link_plan_symbol_satisfies_declaration_interface(declaration,
                                                              candidate);
    if (is_concrete_definition &&
        !loom_link_plan_symbol_is_stripped(options, plan, candidate)) {
      if (loom_link_plan_symbols_share_merge_ownership(plan, declaration,
                                                       candidate)) {
        if (owner_definition) {
          duplicate_owner_definition = candidate;
        } else {
          owner_definition = candidate;
        }
      } else if (iree_any_bit_set(candidate->flags,
                                  LOOM_LINK_SYMBOL_FLAG_EXPORT)) {
        if (external_definition) {
          duplicate_external_definition = candidate;
        } else {
          external_definition = candidate;
        }
      }
    }
    candidate = loom_link_module_index_next_same_name(plan->index, candidate);
  }

  if (duplicate_owner_definition) {
    return loom_link_plan_global_collision_status(plan, owner_definition,
                                                  duplicate_owner_definition);
  }
  if (owner_definition) {
    *out_selected_symbol = owner_definition;
  } else if (duplicate_external_definition) {
    return loom_link_plan_global_collision_status(
        plan, external_definition, duplicate_external_definition);
  } else {
    *out_selected_symbol = external_definition;
  }
  if (*out_selected_symbol ||
      !loom_link_plan_facet_request_requires_definition(declaration,
                                                        definition_request) ||
      loom_link_plan_allows_unresolved(plan, options)) {
    return iree_ok_status();
  }
  return loom_link_plan_unresolved_exact_declaration_status(declaration);
}

static iree_status_t loom_link_plan_find_target_definition_for_declaration(
    const loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_symbol_t* declaration,
    const loom_link_module_index_symbol_t** out_selected_symbol) {
  *out_selected_symbol = NULL;
  const loom_link_module_index_symbol_t* candidate =
      loom_link_module_index_lookup_name(plan->index, declaration->name);
  while (candidate) {
    if (loom_link_plan_symbol_satisfies_declaration_interface(declaration,
                                                              candidate) &&
        !loom_link_plan_symbol_is_stripped(options, plan, candidate)) {
      if (*out_selected_symbol) {
        return loom_link_plan_global_collision_status(
            plan, *out_selected_symbol, candidate);
      }
      *out_selected_symbol = candidate;
    }
    candidate = loom_link_module_index_next_same_name(plan->index, candidate);
  }
  return iree_ok_status();
}

// Rejects an explicit global root whose candidate class is not unique. INPUT
// definitions own the module being assembled and do not extract same-name
// library alternatives. Without an INPUT definition, the explicit library
// universe must expose exactly one concrete root definition.
static iree_status_t loom_link_plan_check_root_definition_ambiguity(
    const loom_link_plan_t* plan, const loom_link_module_index_symbol_t* root) {
  if (!loom_link_plan_symbol_is_concrete_global(root)) {
    return iree_ok_status();
  }
  const loom_link_module_index_provider_t* root_provider =
      loom_link_module_index_symbol_provider(plan->index, root);
  IREE_ASSERT(root_provider);
  const loom_link_module_index_symbol_t* candidate =
      loom_link_module_index_lookup_name(plan->index, root->name);
  while (candidate) {
    if (candidate != root &&
        loom_link_plan_symbol_is_concrete_global(candidate)) {
      const loom_link_module_index_provider_t* candidate_provider =
          loom_link_module_index_symbol_provider(plan->index, candidate);
      IREE_ASSERT(candidate_provider);
      const bool competing_input =
          root_provider->role == LOOM_LINK_PROVIDER_ROLE_INPUT &&
          candidate_provider->role == LOOM_LINK_PROVIDER_ROLE_INPUT;
      const bool competing_library =
          root_provider->role == LOOM_LINK_PROVIDER_ROLE_LIBRARY &&
          candidate_provider->role == LOOM_LINK_PROVIDER_ROLE_LIBRARY &&
          iree_any_bit_set(candidate->flags, LOOM_LINK_SYMBOL_FLAG_EXPORT);
      if (competing_input || competing_library) {
        return loom_link_plan_global_collision_status(plan, root, candidate);
      }
    }
    candidate = loom_link_module_index_next_same_name(plan->index, candidate);
  }
  return iree_ok_status();
}

// Appends one newly selected symbol.
static iree_status_t loom_link_plan_append_symbol(
    loom_link_plan_t* plan, const loom_link_module_index_symbol_t* symbol,
    loom_link_plan_live_cause_t cause, iree_host_size_t* out_plan_ordinal) {
  IREE_ASSERT(!loom_link_plan_bitmap_contains(&plan->reachability.symbols,
                                              symbol->ordinal));
  const iree_host_size_t plan_ordinal = plan->symbols.count;
  iree_host_size_t symbol_count = 0;
  if (!iree_host_size_checked_add(plan_ordinal, 1, &symbol_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "link plan symbol count overflow");
  }
  IREE_RETURN_IF_ERROR(loom_link_plan_reserve_symbols(plan, symbol_count));
  if (plan->symbol_ordinals.values) {
    IREE_RETURN_IF_ERROR(
        loom_link_plan_reserve_symbol_ordinals(plan, symbol_count));
  }
  plan->symbols.values[plan_ordinal] = (loom_link_plan_symbol_work_item_t){
      .selection =
          {
              .ordinal = plan_ordinal,
              .symbol_ordinal = symbol->ordinal,
              .reason = cause.reason,
              .cause_ordinal = cause.symbol_plan_ordinal,
              .primary_facet_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
              .selected_facet_count = 0,
              .root_name = cause.root_name,
          },
      .last_facet_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
      .resolved_definition_symbol_ordinal =
          LOOM_LINK_PLAN_DEFINITION_UNCHECKED_ORDINAL,
  };
  loom_link_plan_bitmap_test_and_set(&plan->reachability.symbols,
                                     symbol->ordinal);
  plan->symbols.count = symbol_count;
  if (plan->symbol_ordinals.values) {
    loom_link_plan_insert_symbol_ordinal(plan->symbol_ordinals.values,
                                         plan->symbol_ordinals.capacity,
                                         symbol->ordinal, plan_ordinal);
  }
  *out_plan_ordinal = plan_ordinal;
  return iree_ok_status();
}

static iree_status_t loom_link_plan_append_facet(
    loom_link_plan_t* plan, const loom_link_module_index_symbol_t* symbol,
    loom_link_symbol_facet_kind_t kind, iree_host_size_t symbol_plan_ordinal,
    loom_link_plan_facet_expansion_mode_t expansion_mode,
    loom_link_plan_live_cause_t cause,
    iree_host_size_t* out_facet_plan_ordinal) {
  const iree_host_size_t facet_plan_ordinal = plan->facets.count;
  iree_host_size_t facet_count = 0;
  if (!iree_host_size_checked_add(facet_plan_ordinal, 1, &facet_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "link plan facet count overflow");
  }
  IREE_RETURN_IF_ERROR(loom_link_plan_reserve_facets(plan, facet_count));
  plan->facets.values[facet_plan_ordinal] = (loom_link_plan_facet_work_item_t){
      .selection =
          {
              .ordinal = facet_plan_ordinal,
              .symbol_plan_ordinal = symbol_plan_ordinal,
              .symbol_ordinal = symbol->ordinal,
              .kind = kind,
              .reason = cause.reason,
              .cause_ordinal = cause.facet_plan_ordinal,
          },
      .expansion_mode = expansion_mode,
      .next_symbol_facet_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
  };
  plan->facets.count = facet_count;
  loom_link_plan_symbol_work_item_t* symbol_work_item =
      &plan->symbols.values[symbol_plan_ordinal];
  if (symbol_work_item->last_facet_plan_ordinal !=
      LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    plan->facets.values[symbol_work_item->last_facet_plan_ordinal]
        .next_symbol_facet_plan_ordinal = facet_plan_ordinal;
  }
  symbol_work_item->last_facet_plan_ordinal = facet_plan_ordinal;
  ++symbol_work_item->selection.selected_facet_count;
  if (kind == loom_link_module_index_symbol_facet_kind_at(symbol, 0)) {
    symbol_work_item->selection.primary_facet_ordinal = facet_plan_ordinal;
  }
  if (out_facet_plan_ordinal) {
    *out_facet_plan_ordinal = facet_plan_ordinal;
  }
  return iree_ok_status();
}

static bool loom_link_plan_facet_is_selected(
    const loom_link_plan_t* plan, iree_host_size_t symbol_plan_ordinal,
    loom_link_symbol_facet_kind_t kind) {
  const loom_link_plan_symbol_t* symbol_selection =
      &plan->symbols.values[symbol_plan_ordinal].selection;
  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_symbol_at(plan->index,
                                       symbol_selection->symbol_ordinal);
  const uint16_t complete_facet_count = symbol->facets.schema.facet_count;
  if (symbol_selection->selected_facet_count == complete_facet_count) {
    return true;
  }
  iree_host_size_t facet_plan_ordinal = symbol_selection->primary_facet_ordinal;
  while (facet_plan_ordinal != LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    const loom_link_plan_facet_work_item_t* facet_work_item =
        &plan->facets.values[facet_plan_ordinal];
    if (facet_work_item->selection.kind == kind) {
      return true;
    }
    facet_plan_ordinal = facet_work_item->next_symbol_facet_plan_ordinal;
  }
  return false;
}

static bool loom_link_plan_facet_request_is_selected(
    const loom_link_plan_t* plan, const loom_link_module_index_symbol_t* symbol,
    loom_link_plan_facet_request_t request) {
  if (!loom_link_plan_bitmap_contains(&plan->reachability.symbols,
                                      symbol->ordinal)) {
    return false;
  }
  iree_host_size_t symbol_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  if (!loom_link_plan_try_lookup_symbol_ordinal(plan, symbol->ordinal,
                                                &symbol_plan_ordinal)) {
    return true;
  }
  const loom_link_plan_symbol_t* symbol_selection =
      &plan->symbols.values[symbol_plan_ordinal].selection;
  const uint16_t complete_facet_count = symbol->facets.schema.facet_count;
  if (symbol_selection->selected_facet_count == complete_facet_count) {
    return true;
  }
  if (request.mode == LOOM_LINK_PLAN_FACET_REQUEST_EXACT) {
    return loom_link_plan_facet_is_selected(plan, symbol_plan_ordinal,
                                            request.kind);
  }
  return request.mode == LOOM_LINK_PLAN_FACET_REQUEST_PRIMARY;
}

static iree_status_t loom_link_plan_select_required_symbol_facets(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_symbol_t* symbol,
    loom_link_plan_facet_request_t request, loom_link_plan_live_cause_t cause,
    iree_host_size_t* out_new_symbol_plan_ordinal) {
  if (out_new_symbol_plan_ordinal) {
    *out_new_symbol_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  }
  if (request.mode > LOOM_LINK_PLAN_FACET_REQUEST_COMPLETE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown link facet request mode %u",
                            (unsigned)request.mode);
  }
  if (request.mode == LOOM_LINK_PLAN_FACET_REQUEST_EXACT &&
      loom_link_module_index_symbol_facet_ordinal(symbol, request.kind) ==
          LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "semantic facet 0x%04X is not exposed by symbol '@%.*s'",
        (unsigned)request.kind, (int)symbol->name.size, symbol->name.data);
  }
  if (loom_link_plan_facet_request_is_selected(plan, symbol, request)) {
    return iree_ok_status();
  }
  if (loom_link_plan_symbol_is_stripped(options, plan, symbol)) {
    if (loom_link_plan_allows_unresolved(plan, options)) {
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "required symbol '@%.*s' was stripped",
                            (int)symbol->name.size, symbol->name.data);
  }

  iree_host_size_t symbol_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  const bool is_new_symbol = !loom_link_plan_bitmap_contains(
      &plan->reachability.symbols, symbol->ordinal);
  if (is_new_symbol) {
    IREE_RETURN_IF_ERROR(loom_link_plan_append_symbol(plan, symbol, cause,
                                                      &symbol_plan_ordinal));
    if (out_new_symbol_plan_ordinal) {
      *out_new_symbol_plan_ordinal = symbol_plan_ordinal;
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_link_plan_lookup_symbol_ordinal(
        plan, symbol->ordinal, &symbol_plan_ordinal));
  }

  const bool batch_complete_expansion =
      is_new_symbol && request.mode == LOOM_LINK_PLAN_FACET_REQUEST_COMPLETE;
  bool needs_complete_expansion = batch_complete_expansion;
  if (plan->symbols.values[symbol_plan_ordinal]
          .selection.selected_facet_count == 0) {
    const loom_link_plan_facet_expansion_mode_t expansion_mode =
        needs_complete_expansion ? LOOM_LINK_PLAN_FACET_EXPANSION_COMPLETE
                                 : LOOM_LINK_PLAN_FACET_EXPANSION_EXACT;
    const loom_link_symbol_facet_kind_t primary_kind =
        loom_link_module_index_symbol_facet_kind_at(symbol, 0);
    IREE_RETURN_IF_ERROR(loom_link_plan_append_facet(
        plan, symbol, primary_kind, symbol_plan_ordinal, expansion_mode, cause,
        /*out_facet_plan_ordinal=*/NULL));
    needs_complete_expansion = false;
  }
  if (request.mode == LOOM_LINK_PLAN_FACET_REQUEST_EXACT) {
    if (!loom_link_plan_facet_is_selected(plan, symbol_plan_ordinal,
                                          request.kind)) {
      IREE_RETURN_IF_ERROR(loom_link_plan_append_facet(
          plan, symbol, request.kind, symbol_plan_ordinal,
          LOOM_LINK_PLAN_FACET_EXPANSION_EXACT, cause,
          /*out_facet_plan_ordinal=*/NULL));
    }
  } else if (request.mode == LOOM_LINK_PLAN_FACET_REQUEST_COMPLETE) {
    for (uint8_t facet_ordinal = 1;
         facet_ordinal < symbol->facets.schema.facet_count; ++facet_ordinal) {
      const loom_link_symbol_facet_kind_t kind =
          loom_link_module_index_symbol_facet_kind_at(symbol, facet_ordinal);
      if (!loom_link_plan_facet_is_selected(plan, symbol_plan_ordinal, kind)) {
        loom_link_plan_facet_expansion_mode_t expansion_mode =
            LOOM_LINK_PLAN_FACET_EXPANSION_EXACT;
        if (batch_complete_expansion) {
          expansion_mode = needs_complete_expansion
                               ? LOOM_LINK_PLAN_FACET_EXPANSION_COMPLETE
                               : LOOM_LINK_PLAN_FACET_EXPANSION_NONE;
        }
        IREE_RETURN_IF_ERROR(loom_link_plan_append_facet(
            plan, symbol, kind, symbol_plan_ordinal, expansion_mode, cause,
            /*out_facet_plan_ordinal=*/NULL));
        needs_complete_expansion = false;
      }
    }
  }
  const uint16_t complete_facet_count = symbol->facets.schema.facet_count;
  if (plan->symbols.values[symbol_plan_ordinal]
              .selection.selected_facet_count != complete_facet_count &&
      plan->symbol_ordinals.values == NULL) {
    IREE_RETURN_IF_ERROR(
        loom_link_plan_reserve_symbol_ordinals(plan, plan->symbols.count));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_select_required_symbol(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_symbol_t* symbol,
    loom_link_plan_live_cause_t cause,
    iree_host_size_t* out_new_symbol_plan_ordinal) {
  const loom_link_plan_facet_request_t request = {
      .mode = LOOM_LINK_PLAN_FACET_REQUEST_COMPLETE,
  };
  return loom_link_plan_select_required_symbol_facets(
      plan, options, symbol, request, cause, out_new_symbol_plan_ordinal);
}

static iree_status_t loom_link_plan_resolve_declaration(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_symbol_t* declaration,
    iree_host_size_t declaration_plan_ordinal,
    loom_link_plan_facet_request_t definition_request) {
  if (!loom_link_plan_symbol_is_declaration_like(declaration)) {
    return iree_ok_status();
  }
  // Template families are authored contracts carried by every using module.
  // The active library universe contributes template providers independently.
  if (declaration->template_family_ordinal !=
      LOOM_LINK_TEMPLATE_FAMILY_ORDINAL_INVALID) {
    return iree_ok_status();
  }
  const bool exact_declaration =
      loom_link_plan_symbol_is_exact_declaration(declaration);
  const bool target_declaration =
      loom_link_plan_symbol_is_target_declaration(declaration);
  if (!exact_declaration && !target_declaration) {
    return iree_ok_status();
  }
  loom_link_plan_symbol_work_item_t* declaration_work_item =
      &plan->symbols.values[declaration_plan_ordinal];
  const loom_link_module_index_symbol_t* selected_symbol = NULL;
  if (declaration_work_item->resolved_definition_symbol_ordinal ==
      LOOM_LINK_PLAN_DEFINITION_ABSENT_ORDINAL) {
    if (exact_declaration &&
        loom_link_plan_facet_request_requires_definition(declaration,
                                                         definition_request) &&
        !loom_link_plan_allows_unresolved(plan, options)) {
      return loom_link_plan_unresolved_exact_declaration_status(declaration);
    }
    return iree_ok_status();
  } else if (declaration_work_item->resolved_definition_symbol_ordinal !=
             LOOM_LINK_PLAN_DEFINITION_UNCHECKED_ORDINAL) {
    selected_symbol = loom_link_module_index_symbol_at(
        plan->index, declaration_work_item->resolved_definition_symbol_ordinal);
    IREE_ASSERT(selected_symbol);
  } else {
    if (exact_declaration) {
      IREE_RETURN_IF_ERROR(loom_link_plan_find_exact_definition_for_declaration(
          plan, options, declaration, definition_request, &selected_symbol));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_link_plan_find_target_definition_for_declaration(
              plan, options, declaration, &selected_symbol));
    }
    declaration_work_item->resolved_definition_symbol_ordinal =
        selected_symbol ? selected_symbol->ordinal
                        : LOOM_LINK_PLAN_DEFINITION_ABSENT_ORDINAL;
  }
  if (!selected_symbol) {
    return iree_ok_status();
  }
  const loom_link_plan_symbol_t* declaration_selection =
      &declaration_work_item->selection;
  const loom_link_plan_live_cause_t cause = {
      .reason = LOOM_LINK_PLAN_LIVE_DEPENDENCY,
      .symbol_plan_ordinal = declaration_plan_ordinal,
      .facet_plan_ordinal = declaration_selection->primary_facet_ordinal,
      .root_name = iree_string_view_empty(),
  };
  IREE_RETURN_IF_ERROR(loom_link_plan_select_required_symbol_facets(
      plan, options, selected_symbol, definition_request, cause,
      /*out_new_symbol_plan_ordinal=*/NULL));
  return iree_ok_status();
}

static iree_status_t loom_link_plan_select_global_reference(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_symbol_t* referenced_symbol,
    loom_link_plan_facet_request_t request, loom_link_plan_live_cause_t cause) {
  const bool references_local_declaration =
      loom_link_plan_symbol_is_declaration_like(referenced_symbol);
  const loom_link_module_index_symbol_t* selected_symbol =
      references_local_declaration ? referenced_symbol
                                   : loom_link_module_index_lookup_global(
                                         plan->index, referenced_symbol->name);
  if (!selected_symbol) {
    if (loom_link_plan_allows_unresolved(plan, options)) {
      return iree_ok_status();
    }
    return iree_make_status(
        IREE_STATUS_NOT_FOUND, "unresolved global dependency '@%.*s'",
        (int)referenced_symbol->name.size, referenced_symbol->name.data);
  }

  iree_host_size_t selected_plan_ordinal =
      LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  const bool selected_is_declaration =
      loom_link_plan_symbol_is_declaration_like(selected_symbol);
  const loom_link_plan_facet_request_t selection_request =
      selected_is_declaration
          ? (loom_link_plan_facet_request_t){
                .mode = LOOM_LINK_PLAN_FACET_REQUEST_PRIMARY,
            }
          : request;
  IREE_RETURN_IF_ERROR(loom_link_plan_select_required_symbol_facets(
      plan, options, selected_symbol, selection_request, cause,
      &selected_plan_ordinal));
  if (!selected_is_declaration) {
    return iree_ok_status();
  }
  if (selected_plan_ordinal == LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL &&
      !loom_link_plan_try_lookup_symbol_ordinal(plan, selected_symbol->ordinal,
                                                &selected_plan_ordinal)) {
    // A previous complete selection needs no projection upgrade. An unresolved
    // provider also remains unresolved against the immutable index/options.
    return iree_ok_status();
  }
  return loom_link_plan_resolve_declaration(plan, options, selected_symbol,
                                            selected_plan_ordinal, request);
}

//===----------------------------------------------------------------------===//
// Closure expansion
//===----------------------------------------------------------------------===//

// Maps one authored symbol-reference contract to the semantic provider
// projection needed by that use. Logical launches consume configuration but
// not implementation, while physical dispatches consume only the entry
// contract. Every other reference retains the complete-symbol behavior.
static loom_link_plan_facet_request_t loom_link_plan_dependency_facet_request(
    const loom_link_plan_options_t* options,
    loom_symbol_interface_flags_t target_interfaces) {
  if (!options ||
      options->dependency_policy == LOOM_LINK_PLAN_DEPENDENCY_COMPLETE) {
    return (loom_link_plan_facet_request_t){
        .mode = LOOM_LINK_PLAN_FACET_REQUEST_COMPLETE,
    };
  }
  IREE_ASSERT(options->dependency_policy ==
              LOOM_LINK_PLAN_DEPENDENCY_REQUESTED_FACETS);
  const bool requests_kernel =
      iree_any_bit_set(target_interfaces, LOOM_SYMBOL_INTERFACE_KERNEL);
  const bool requests_kernel_entry =
      iree_any_bit_set(target_interfaces, LOOM_SYMBOL_INTERFACE_KERNEL_ENTRY);
  IREE_ASSERT(!(requests_kernel && requests_kernel_entry));
  if (requests_kernel) {
    return (loom_link_plan_facet_request_t){
        .mode = LOOM_LINK_PLAN_FACET_REQUEST_EXACT,
        .kind = LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION,
    };
  }
  return (loom_link_plan_facet_request_t){
      .mode = requests_kernel_entry ? LOOM_LINK_PLAN_FACET_REQUEST_PRIMARY
                                    : LOOM_LINK_PLAN_FACET_REQUEST_COMPLETE,
  };
}

static iree_status_t loom_link_plan_select_dependency_target(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_module_t* module, uint32_t target_symbol_id,
    loom_symbol_interface_flags_t target_interfaces,
    loom_link_plan_live_cause_t cause) {
  IREE_ASSERT_LT(target_symbol_id, module->symbol_count);
  const loom_link_module_index_symbol_t* target =
      loom_link_module_index_symbol_at(
          plan->index, module->symbol_start_ordinal + target_symbol_id);
  IREE_ASSERT(target);
  const loom_link_plan_facet_request_t request =
      loom_link_plan_dependency_facet_request(options, target_interfaces);
  if (target->identity == LOOM_LINK_SYMBOL_IDENTITY_GLOBAL) {
    return loom_link_plan_select_global_reference(plan, options, target,
                                                  request, cause);
  }
  return loom_link_plan_select_required_symbol_facets(
      plan, options, target, request, cause,
      /*out_new_symbol_plan_ordinal=*/NULL);
}

static iree_status_t loom_link_plan_expand_module_dependencies(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_module_t* module,
    loom_link_plan_live_cause_t cause) {
  const uint32_t count = module->dependencies.root_count;
  const uint32_t* dependencies = module->dependencies.values;
  const loom_symbol_interface_flags_t* target_interfaces =
      module->dependencies.target_interfaces;
  for (uint32_t i = 0; i < count; ++i) {
    IREE_ASSERT_EQ(module->dependencies.source_root_region_indices_plus_one[i],
                   0);
    IREE_RETURN_IF_ERROR(loom_link_plan_select_dependency_target(
        plan, options, module, dependencies[i], target_interfaces[i], cause));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_expand_symbol_facet_dependencies(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_symbol_t* symbol,
    const loom_link_module_index_module_t* module,
    loom_link_symbol_facet_kind_t kind, loom_link_plan_live_cause_t cause) {
  if (symbol->dependencies.count == 0) {
    return iree_ok_status();
  }
  const uint32_t first = symbol->dependencies.first;
  const uint32_t* dependencies = module->dependencies.values + first;
  const loom_symbol_interface_flags_t* target_interfaces =
      module->dependencies.target_interfaces + first;
  for (uint32_t i = 0; i < symbol->dependencies.count; ++i) {
    const uint8_t source_root_region_index_plus_one =
        module->dependencies.source_root_region_indices_plus_one[first + i];
    const loom_link_symbol_facet_kind_t source_kind =
        loom_link_module_index_symbol_source_root_facet_kind(
            symbol, source_root_region_index_plus_one);
    IREE_ASSERT_NE(source_kind, LOOM_LINK_SYMBOL_FACET_INVALID);
    if (source_kind != kind) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_link_plan_select_dependency_target(
        plan, options, module, dependencies[i], target_interfaces[i], cause));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_record_template_family_demand(
    loom_link_plan_t* plan,
    loom_link_template_family_ordinal_t family_ordinal) {
  ++plan->demanded_template_families.occurrence_count;
  if (loom_link_plan_bitmap_test_and_set(&plan->reachability.template_families,
                                         family_ordinal)) {
    return iree_ok_status();
  }
  iree_host_size_t new_count = 0;
  if (!iree_host_size_checked_add(plan->demanded_template_families.count, 1,
                                  &new_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "link plan template-family demand overflow");
  }
  IREE_RETURN_IF_ERROR(
      loom_link_plan_reserve_demanded_template_families(plan, new_count));
  plan->demanded_template_families
      .values[plan->demanded_template_families.count] = family_ordinal;
  plan->demanded_template_families.count = new_count;
  return iree_ok_status();
}

static iree_status_t loom_link_plan_record_symbol_facet_template_demands(
    loom_link_plan_t* plan, const loom_link_module_index_symbol_t* symbol,
    const loom_link_module_index_module_t* module,
    loom_link_symbol_facet_kind_t kind) {
  if (symbol->template_demands.count == 0) {
    return iree_ok_status();
  }
  const uint32_t first = symbol->template_demands.first;
  const loom_link_template_family_ordinal_t* template_families =
      module->template_demands.values + first;
  for (uint32_t i = 0; i < symbol->template_demands.count; ++i) {
    const uint8_t source_root_region_index_plus_one =
        module->template_demands.source_root_region_indices_plus_one[first + i];
    const loom_link_symbol_facet_kind_t source_kind =
        loom_link_module_index_symbol_source_root_facet_kind(
            symbol, source_root_region_index_plus_one);
    IREE_ASSERT_NE(source_kind, LOOM_LINK_SYMBOL_FACET_INVALID);
    if (source_kind != kind) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_link_plan_record_template_family_demand(
        plan, template_families[i]));
  }
  return iree_ok_status();
}

static iree_host_size_t loom_link_plan_find_facet_plan_ordinal(
    const loom_link_plan_t* plan, iree_host_size_t symbol_plan_ordinal,
    loom_link_symbol_facet_kind_t kind) {
  const loom_link_plan_symbol_t* symbol_selection =
      &plan->symbols.values[symbol_plan_ordinal].selection;
  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_symbol_at(plan->index,
                                       symbol_selection->symbol_ordinal);
  IREE_ASSERT_NE(symbol_selection->primary_facet_ordinal,
                 LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL);
  IREE_ASSERT_NE(loom_link_module_index_symbol_facet_ordinal(symbol, kind),
                 LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL);
  iree_host_size_t facet_plan_ordinal = symbol_selection->primary_facet_ordinal;
  while (facet_plan_ordinal != LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    const loom_link_plan_facet_work_item_t* facet_work_item =
        &plan->facets.values[facet_plan_ordinal];
    if (facet_work_item->selection.kind == kind) {
      return facet_plan_ordinal;
    }
    facet_plan_ordinal = facet_work_item->next_symbol_facet_plan_ordinal;
  }
  IREE_ASSERT_UNREACHABLE("selected symbol facet is absent from its worklist");
  return LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
}

static loom_link_plan_live_cause_t loom_link_plan_facet_dependency_cause(
    const loom_link_plan_t* plan, iree_host_size_t symbol_plan_ordinal,
    loom_link_symbol_facet_kind_t kind) {
  return (loom_link_plan_live_cause_t){
      .reason = LOOM_LINK_PLAN_LIVE_DEPENDENCY,
      .symbol_plan_ordinal = symbol_plan_ordinal,
      .facet_plan_ordinal = loom_link_plan_find_facet_plan_ordinal(
          plan, symbol_plan_ordinal, kind),
      .root_name = iree_string_view_empty(),
  };
}

static iree_status_t loom_link_plan_expand_complete_symbol(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    const loom_link_module_index_symbol_t* symbol,
    const loom_link_module_index_module_t* module,
    iree_host_size_t symbol_plan_ordinal) {
  if (symbol->dependencies.count != 0) {
    const uint32_t dependency_first = symbol->dependencies.first;
    const uint32_t* dependencies =
        module->dependencies.values + dependency_first;
    const loom_symbol_interface_flags_t* target_interfaces =
        module->dependencies.target_interfaces + dependency_first;
    for (uint32_t i = 0; i < symbol->dependencies.count; ++i) {
      const uint8_t source_root_region_index_plus_one =
          module->dependencies
              .source_root_region_indices_plus_one[dependency_first + i];
      const loom_link_symbol_facet_kind_t source_kind =
          loom_link_module_index_symbol_source_root_facet_kind(
              symbol, source_root_region_index_plus_one);
      IREE_ASSERT_NE(source_kind, LOOM_LINK_SYMBOL_FACET_INVALID);
      const loom_link_plan_live_cause_t cause =
          loom_link_plan_facet_dependency_cause(plan, symbol_plan_ordinal,
                                                source_kind);
      IREE_RETURN_IF_ERROR(loom_link_plan_select_dependency_target(
          plan, options, module, dependencies[i], target_interfaces[i], cause));
    }
  }

  if (symbol->template_demands.count != 0) {
    const uint32_t demand_first = symbol->template_demands.first;
    const loom_link_template_family_ordinal_t* template_families =
        module->template_demands.values + demand_first;
    for (uint32_t i = 0; i < symbol->template_demands.count; ++i) {
      IREE_RETURN_IF_ERROR(loom_link_plan_record_template_family_demand(
          plan, template_families[i]));
    }
  }

  return iree_ok_status();
}

static iree_status_t loom_link_plan_expand_facet(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options,
    iree_host_size_t facet_plan_ordinal) {
  const loom_link_plan_facet_work_item_t work_item =
      plan->facets.values[facet_plan_ordinal];
  const loom_link_plan_facet_t facet = work_item.selection;
  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_symbol_at(plan->index, facet.symbol_ordinal);
  const loom_link_module_index_module_t* module =
      loom_link_module_index_symbol_module(plan->index, symbol);
  IREE_ASSERT(symbol);
  IREE_ASSERT(module);
  if (work_item.expansion_mode == LOOM_LINK_PLAN_FACET_EXPANSION_NONE) {
    return iree_ok_status();
  }
  if (work_item.expansion_mode == LOOM_LINK_PLAN_FACET_EXPANSION_COMPLETE) {
    const iree_host_size_t primary_facet_ordinal =
        plan->symbols.values[facet.symbol_plan_ordinal]
            .selection.primary_facet_ordinal;
    const loom_link_plan_live_cause_t module_cause = {
        .reason = LOOM_LINK_PLAN_LIVE_DEPENDENCY,
        .symbol_plan_ordinal = facet.symbol_plan_ordinal,
        .facet_plan_ordinal = primary_facet_ordinal,
        .root_name = iree_string_view_empty(),
    };
    if (!loom_link_plan_bitmap_test_and_set(&plan->reachability.modules,
                                            module->ordinal)) {
      IREE_RETURN_IF_ERROR(loom_link_plan_expand_module_dependencies(
          plan, options, module, module_cause));
    }
    return loom_link_plan_expand_complete_symbol(plan, options, symbol, module,
                                                 facet.symbol_plan_ordinal);
  }
  const loom_link_plan_live_cause_t cause = {
      .reason = LOOM_LINK_PLAN_LIVE_DEPENDENCY,
      .symbol_plan_ordinal = facet.symbol_plan_ordinal,
      .facet_plan_ordinal = facet_plan_ordinal,
      .root_name = iree_string_view_empty(),
  };

  if (!loom_link_plan_bitmap_test_and_set(&plan->reachability.modules,
                                          module->ordinal)) {
    IREE_RETURN_IF_ERROR(loom_link_plan_expand_module_dependencies(
        plan, options, module, cause));
  }
  IREE_RETURN_IF_ERROR(loom_link_plan_expand_symbol_facet_dependencies(
      plan, options, symbol, module, facet.kind, cause));
  return loom_link_plan_record_symbol_facet_template_demands(
      plan, symbol, module, facet.kind);
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
    const loom_link_plan_live_cause_t cause = {
        .reason = LOOM_LINK_PLAN_LIVE_ARCHIVE,
        .symbol_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
        .facet_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
        .root_name = iree_string_view_empty(),
    };
    iree_host_size_t symbol_plan_ordinal =
        LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    IREE_RETURN_IF_ERROR(loom_link_plan_append_symbol(plan, symbol, cause,
                                                      &symbol_plan_ordinal));
  }
  for (iree_host_size_t i = 0; i < plan->symbols.count; ++i) {
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(
            plan->index, plan->symbols.values[i].selection.symbol_ordinal);
    const loom_link_plan_facet_request_t definition_request = {
        .mode = LOOM_LINK_PLAN_FACET_REQUEST_COMPLETE,
    };
    IREE_RETURN_IF_ERROR(loom_link_plan_resolve_declaration(
        plan, options, symbol, i, definition_request));
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
  const loom_link_plan_live_cause_t cause = {
      .reason = LOOM_LINK_PLAN_LIVE_ROOT,
      .symbol_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
      .facet_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
      .root_name = root_name,
  };
  if (root) {
    IREE_RETURN_IF_ERROR(
        loom_link_plan_check_root_definition_ambiguity(plan, root));
    const loom_link_plan_facet_request_t request = {
        .mode = LOOM_LINK_PLAN_FACET_REQUEST_COMPLETE,
    };
    return loom_link_plan_select_global_reference(plan, options, root, request,
                                                  cause);
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
      plan, options, private_root, cause,
      /*out_new_symbol_plan_ordinal=*/NULL);
}

static iree_status_t loom_link_plan_select_input_exports(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options) {
  const loom_link_module_index_symbol_ordinal_list_t input_exports =
      loom_link_module_index_input_exports(plan->index);
  for (iree_host_size_t i = 0; i < input_exports.count; ++i) {
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(plan->index, input_exports.values[i]);
    iree_host_size_t root_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    const loom_link_plan_live_cause_t cause = {
        .reason = LOOM_LINK_PLAN_LIVE_ROOT,
        .symbol_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
        .facet_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
        .root_name = symbol->name,
    };
    IREE_RETURN_IF_ERROR(loom_link_plan_select_required_symbol(
        plan, options, symbol, cause, &root_plan_ordinal));
    if (root_plan_ordinal != LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
      const loom_link_plan_facet_request_t definition_request = {
          .mode = LOOM_LINK_PLAN_FACET_REQUEST_COMPLETE,
      };
      IREE_RETURN_IF_ERROR(loom_link_plan_resolve_declaration(
          plan, options, symbol, root_plan_ordinal, definition_request));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_select_root_facets(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options) {
  const iree_host_size_t root_count = options ? options->root_facets.count : 0;
  const loom_link_plan_root_facet_t* roots =
      options ? options->root_facets.values : NULL;
  if (root_count != 0 && roots == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "root_facets count is non-zero but values is NULL");
  }
  const iree_host_size_t symbol_count =
      loom_link_module_index_symbol_count(plan->index);
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    const loom_link_plan_root_facet_t root = roots[i];
    if (root.symbol_ordinal >= symbol_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "root facet symbol ordinal %" PRIhsz
                              " is outside the %" PRIhsz "-symbol index",
                              root.symbol_ordinal, symbol_count);
    }
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(plan->index, root.symbol_ordinal);
    const loom_link_plan_facet_request_t request = {
        .mode = LOOM_LINK_PLAN_FACET_REQUEST_EXACT,
        .kind = root.kind,
    };
    const loom_link_plan_live_cause_t cause = {
        .reason = LOOM_LINK_PLAN_LIVE_ROOT,
        .symbol_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
        .facet_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
        .root_name = symbol->name,
    };
    iree_host_size_t root_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    IREE_RETURN_IF_ERROR(loom_link_plan_select_required_symbol_facets(
        plan, options, symbol, request, cause, &root_plan_ordinal));
    if (root_plan_ordinal != LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
      IREE_RETURN_IF_ERROR(loom_link_plan_resolve_declaration(
          plan, options, symbol, root_plan_ordinal, request));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_select_roots(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options) {
  if (options && options->include_input_exports) {
    IREE_RETURN_IF_ERROR(loom_link_plan_select_input_exports(plan, options));
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
  IREE_RETURN_IF_ERROR(loom_link_plan_select_root_facets(plan, options));
  if (plan->symbols.count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selective link planning requires at least one root");
  }
  return iree_ok_status();
}

static bool loom_link_plan_symbol_is_template_provider(
    const loom_link_module_index_symbol_t* symbol) {
  return symbol->kind == LOOM_SYMBOL_TEMPLATE_DEF ||
         symbol->kind == LOOM_SYMBOL_TEMPLATE_UKERNEL;
}

static iree_status_t loom_link_plan_select_template_providers(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options) {
  const iree_host_size_t provider_count =
      options ? options->selected_template_providers.count : 0;
  const iree_host_size_t* provider_ordinals =
      options ? options->selected_template_providers.values : NULL;
  if (provider_count != 0 && provider_ordinals == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected template provider count is non-zero but values is NULL");
  }
  const iree_host_size_t symbol_count =
      loom_link_module_index_symbol_count(plan->index);
  for (iree_host_size_t i = 0; i < provider_count; ++i) {
    const iree_host_size_t symbol_ordinal = provider_ordinals[i];
    if (symbol_ordinal >= symbol_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "selected template provider ordinal %" PRIhsz
                              " is outside the %" PRIhsz "-symbol index",
                              symbol_ordinal, symbol_count);
    }
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(plan->index, symbol_ordinal);
    if (!loom_link_plan_symbol_is_template_provider(symbol)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "selected template provider ordinal %" PRIhsz
                              " names non-provider symbol '@%.*s'",
                              symbol_ordinal, (int)symbol->name.size,
                              symbol->name.data);
    }
    const loom_link_plan_live_cause_t cause = {
        .reason = LOOM_LINK_PLAN_LIVE_PROVIDER,
        .symbol_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
        .facet_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
        .root_name = iree_string_view_empty(),
    };
    IREE_RETURN_IF_ERROR(loom_link_plan_select_required_symbol(
        plan, options, symbol, cause,
        /*out_new_symbol_plan_ordinal=*/NULL));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_plan_select_selective(
    loom_link_plan_t* plan, const loom_link_plan_options_t* options) {
  IREE_RETURN_IF_ERROR(loom_link_plan_select_roots(plan, options));
  IREE_RETURN_IF_ERROR(loom_link_plan_select_template_providers(plan, options));
  for (iree_host_size_t i = 0; i < plan->facets.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_link_plan_expand_facet(plan, options, i));
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

  iree_status_t status = loom_link_plan_initialize_reachability(plan);
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
  iree_allocator_free(plan->allocator, plan->symbol_ordinals.values);
  iree_allocator_free(plan->allocator, plan->reachability.storage);
  iree_allocator_free(plan->allocator, plan->demanded_template_families.values);
  iree_allocator_free(plan->allocator, plan->facets.values);
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
  return &plan->symbols.values[ordinal].selection;
}

iree_host_size_t loom_link_plan_facet_count(const loom_link_plan_t* plan) {
  return plan ? plan->facets.count : 0;
}

const loom_link_plan_facet_t* loom_link_plan_facet_at(
    const loom_link_plan_t* plan, iree_host_size_t ordinal) {
  if (!plan || ordinal >= plan->facets.count) {
    return NULL;
  }
  return &plan->facets.values[ordinal].selection;
}

iree_host_size_t loom_link_plan_demanded_template_family_count(
    const loom_link_plan_t* plan) {
  return plan ? plan->demanded_template_families.count : 0;
}

iree_host_size_t loom_link_plan_template_demand_occurrence_count(
    const loom_link_plan_t* plan) {
  return plan ? plan->demanded_template_families.occurrence_count : 0;
}

loom_link_template_family_ordinal_t loom_link_plan_demanded_template_family_at(
    const loom_link_plan_t* plan, iree_host_size_t ordinal) {
  if (!plan || ordinal >= plan->demanded_template_families.count) {
    return LOOM_LINK_TEMPLATE_FAMILY_ORDINAL_INVALID;
  }
  return plan->demanded_template_families.values[ordinal];
}

bool loom_link_plan_contains_symbol(const loom_link_plan_t* plan,
                                    iree_host_size_t symbol_ordinal) {
  return plan && loom_link_plan_bitmap_contains(&plan->reachability.symbols,
                                                symbol_ordinal);
}

bool loom_link_plan_contains_facet(const loom_link_plan_t* plan,
                                   iree_host_size_t symbol_ordinal,
                                   loom_link_symbol_facet_kind_t kind) {
  if (!plan) {
    return false;
  }
  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_symbol_at(plan->index, symbol_ordinal);
  if (!symbol || loom_link_module_index_symbol_facet_ordinal(symbol, kind) ==
                     LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    return false;
  }
  if (!loom_link_plan_bitmap_contains(&plan->reachability.symbols,
                                      symbol_ordinal)) {
    return false;
  }
  if (plan->mode == LOOM_LINK_PLAN_ARCHIVE ||
      plan->symbol_ordinals.values == NULL) {
    return true;
  }
  iree_host_size_t symbol_plan_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  const bool found = loom_link_plan_try_lookup_symbol_ordinal(
      plan, symbol_ordinal, &symbol_plan_ordinal);
  IREE_ASSERT(found);
  return found &&
         loom_link_plan_facet_is_selected(plan, symbol_plan_ordinal, kind);
}
