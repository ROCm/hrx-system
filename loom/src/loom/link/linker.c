// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/linker.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "loom/analysis/symbol_references.h"
#include "loom/analysis/symbol_value_constraints.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/symbol_map.h"
#include "loom/link/func_contract.h"
#include "loom/link/symbol_policy.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/template/ops.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/util/adaptive_sort.h"
#include "loom/util/walk.h"

typedef uint32_t loom_linker_symbol_set_attr_id_t;

typedef uint32_t loom_linker_symbol_set_use_id_t;
#define LOOM_LINKER_SYMBOL_SET_USE_ID_INVALID \
  ((loom_linker_symbol_set_use_id_t)UINT32_MAX)

// One incoming symbol-set use of a target symbol.
typedef struct loom_linker_symbol_set_use_t {
  // Symbol-set operation attribute containing the symbol.
  loom_linker_symbol_set_attr_id_t attr_id;
  // Next use containing the same target symbol.
  loom_linker_symbol_set_use_id_t next_use_id;
} loom_linker_symbol_set_use_t;

// Construction-time incoming-use index for symbol-set attributes.
typedef struct loom_linker_symbol_set_index_t {
  // Per-target-symbol heads into |uses|.
  loom_linker_symbol_set_use_id_t* symbol_use_heads;
  // Allocated target-symbol slots in |symbol_use_heads|.
  iree_host_size_t symbol_use_head_capacity;
  // Stable target operation attribute slots carrying symbol sets.
  loom_attribute_t** attrs;
  // Number of live entries in |attrs|.
  iree_host_size_t attr_count;
  // Allocated entries in |attrs|.
  iree_host_size_t attr_capacity;
  // Per-symbol incoming uses of symbol-set operation attributes.
  loom_linker_symbol_set_use_t* uses;
  // Number of live entries in |uses|.
  iree_host_size_t use_count;
  // Allocated entries in |uses|.
  iree_host_size_t use_capacity;
} loom_linker_symbol_set_index_t;

// Transient sparse-link state for one target symbol.
typedef struct loom_linker_planned_symbol_t {
  // Requested outward disposition accumulated across selected sources.
  loom_linker_symbol_output_t output;
  // True when the construction-time identity is global across providers.
  bool global_identity;
} loom_linker_planned_symbol_t;

struct loom_linker_t {
  // Context shared by all source modules and the target module.
  loom_context_t* context;
  // Block pool used for the output module and per-input temporary arenas.
  iree_arena_block_pool_t* block_pool;
  // Host allocator used for the linker object and temporary string builders.
  iree_allocator_t allocator;
  // Persistent linker scratch for target name maps.
  iree_arena_allocator_t scratch_arena;
  // Linked output module being constructed.
  loom_module_t* target_module;
  // Hash map from target-module string IDs to target symbol IDs.
  loom_symbol_map_t target_symbol_lookup;
  // Monotonic ordinal used when assigning deterministic private conflict names.
  iree_host_size_t private_name_ordinal;
  // Symbol-set attributes indexed by referenced target symbol.
  loom_linker_symbol_set_index_t symbol_set_index;
  // Sparse-plan state indexed by target symbol.
  struct {
    // Per-target-symbol construction identity and outward disposition.
    loom_linker_planned_symbol_t* symbols;
    // Allocated entries in |symbols|.
    iree_host_size_t capacity;
  } planned;
  // True once finish has transferred the output module to the caller.
  bool finished;
};

typedef struct loom_linker_exact_selection_t {
  // Strictly increasing module-local source ordinals, or NULL when dense.
  const iree_host_size_t* ordinals;
  // Omitted source symbols already projected into the target module.
  loom_linker_source_symbol_binding_list_t bindings;
  // Output dispositions in source-selection order, or NULL when authored.
  const loom_linker_symbol_output_t* outputs;
  // Number of selected source symbols.
  iree_host_size_t count;
  // True when the selection is the complete dense source-symbol domain.
  bool dense;
} loom_linker_exact_selection_t;

// One selected source symbol definition ordered for materialization.
typedef struct loom_linker_exact_symbol_op_t {
  // Source operation defining the selected symbol.
  const loom_op_t* source_op;
  // Module-local source symbol ordinal.
  uint16_t source_symbol_id;
  // Entry in the exact selection and target-symbol projection.
  iree_host_size_t selection_ordinal;
} loom_linker_exact_symbol_op_t;

static bool loom_linker_exact_symbol_op_less(
    const loom_linker_exact_symbol_op_t* lhs,
    const loom_linker_exact_symbol_op_t* rhs) {
  return lhs->source_op->block_ordinal < rhs->source_op->block_ordinal;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_linker_sort_exact_symbol_ops,
                          loom_linker_exact_symbol_op_t,
                          loom_linker_exact_symbol_op_less)

typedef struct loom_linker_source_t {
  // Owning linker.
  loom_linker_t* linker;
  // Source module currently being cloned.
  const loom_module_t* module;
  // Per-input temporary arena.
  iree_arena_allocator_t* arena;
  // Dense source-symbol map, or entries parallel to exact.ordinals.
  loom_symbol_ref_t* target_symbols;
  // Number of symbols in the source module.
  iree_host_size_t source_symbol_count;
  // Source symbols selected by this add operation.
  uint8_t* live_symbols;
  // Source symbols whose outgoing dependency edges have been scanned.
  uint8_t* scanned_symbols;
  // Source-module dependency graph. Built only for root-filtered adds.
  loom_symbol_reference_table_t reference_table;
  // Lazily initialized source-to-target remap table.
  loom_ir_remap_t remap;
  // Symbol remap policy used by source IR cloning and contract merging.
  loom_ir_remap_symbol_callback_t symbol_remap;
  // Exact source-symbol selection state.
  loom_linker_exact_selection_t exact;
  // True when root filtering is active for this add operation.
  bool root_filtered;
} loom_linker_source_t;

static iree_string_view_t loom_link_target_symbol_name(
    const loom_module_t* target_module, loom_symbol_ref_t target_ref) {
  const loom_symbol_t* symbol =
      &target_module->symbols.entries[target_ref.symbol_id];
  return target_module->strings.entries[symbol->name_id];
}

static iree_status_t loom_link_source_symbol_name(
    const loom_module_t* source_module, uint16_t source_symbol_id,
    iree_string_view_t* out_name) {
  const loom_symbol_t* source_symbol =
      &source_module->symbols.entries[source_symbol_id];
  if (source_symbol->name_id >= source_module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source symbol %u name id %u is out of range",
                            (unsigned)source_symbol_id,
                            (unsigned)source_symbol->name_id);
  }
  *out_name = source_module->strings.entries[source_symbol->name_id];
  return iree_ok_status();
}

static bool loom_link_target_symbol_is_private_concrete(
    const loom_linker_t* linker, uint16_t target_symbol_id) {
  if (target_symbol_id >= linker->target_module->symbols.count) return false;
  const loom_symbol_t* symbol =
      &linker->target_module->symbols.entries[target_symbol_id];
  return loom_link_symbol_is_concrete_definition(symbol) &&
         !loom_link_symbol_has_global_identity(linker->target_module, symbol);
}

static iree_status_t loom_linker_allocate_fresh_private_name(
    loom_linker_t* linker, iree_string_view_t base_name,
    loom_string_id_t* out_name_id) {
  *out_name_id = LOOM_STRING_ID_INVALID;

  if (base_name.size > IREE_HOST_SIZE_MAX - 32) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "private symbol name is too long");
  }
  const iree_host_size_t candidate_capacity = base_name.size + 32;
  char* candidate = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      linker->allocator, candidate_capacity, (void**)&candidate));
  memcpy(candidate, base_name.data, base_name.size);

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    int suffix_length =
        snprintf(candidate + base_name.size, 32, "$link%" PRIhsz,
                 linker->private_name_ordinal++);
    if (suffix_length < 0 || suffix_length >= 32) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "private symbol suffix exceeded scratch "
                                "buffer capacity");
      break;
    }

    loom_string_id_t candidate_name_id = LOOM_STRING_ID_INVALID;
    status = loom_module_intern_string(
        linker->target_module,
        iree_make_string_view(candidate, base_name.size + suffix_length),
        &candidate_name_id);
    if (!iree_status_is_ok(status)) break;
    if (loom_symbol_map_find(&linker->target_symbol_lookup,
                             candidate_name_id) == LOOM_SYMBOL_ID_INVALID) {
      *out_name_id = candidate_name_id;
      break;
    }
  }
  iree_allocator_free(linker->allocator, candidate);
  return status;
}

static iree_status_t loom_linker_ensure_symbol_set_capacity(
    loom_linker_t* linker, iree_host_size_t required_capacity) {
  loom_linker_symbol_set_index_t* index = &linker->symbol_set_index;
  if (required_capacity <= index->symbol_use_head_capacity) {
    return iree_ok_status();
  }
  const iree_host_size_t old_capacity = index->symbol_use_head_capacity;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      &linker->scratch_arena, old_capacity, required_capacity,
      sizeof(*index->symbol_use_heads), &index->symbol_use_head_capacity,
      (void**)&index->symbol_use_heads));
  for (iree_host_size_t i = old_capacity; i < index->symbol_use_head_capacity;
       ++i) {
    index->symbol_use_heads[i] = LOOM_LINKER_SYMBOL_SET_USE_ID_INVALID;
  }
  return iree_ok_status();
}

static iree_status_t loom_linker_ensure_planned_symbol_capacity(
    loom_linker_t* linker, iree_host_size_t required_capacity) {
  if (required_capacity <= linker->planned.capacity) {
    return iree_ok_status();
  }
  const iree_host_size_t old_capacity = linker->planned.capacity;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      &linker->scratch_arena, old_capacity, required_capacity,
      sizeof(*linker->planned.symbols), &linker->planned.capacity,
      (void**)&linker->planned.symbols));
  memset(linker->planned.symbols + old_capacity, 0,
         (linker->planned.capacity - old_capacity) *
             sizeof(*linker->planned.symbols));
  return iree_ok_status();
}

static iree_status_t loom_linker_register_symbol_set(loom_linker_t* linker,
                                                     loom_attribute_t* attr) {
  if (attr->count == 0) return iree_ok_status();
  loom_linker_symbol_set_index_t* index = &linker->symbol_set_index;
  if (index->attr_count >= UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "linker symbol-set index cannot exceed %u attributes",
        (unsigned)UINT32_MAX);
  }
  if (index->use_count > UINT32_MAX - attr->count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "linker symbol-set index cannot exceed "
                            "%u uses",
                            (unsigned)UINT32_MAX);
  }
  IREE_RETURN_IF_ERROR(loom_linker_ensure_symbol_set_capacity(
      linker, linker->target_module->symbols.count));
  if (index->attr_count >= index->attr_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &linker->scratch_arena, index->attr_count, index->attr_count + 1,
        sizeof(*index->attrs), &index->attr_capacity, (void**)&index->attrs));
  }
  const iree_host_size_t required_use_count = index->use_count + attr->count;
  if (required_use_count > index->use_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        &linker->scratch_arena, index->use_count, required_use_count,
        sizeof(*index->uses), &index->use_capacity, (void**)&index->uses));
  }

  const loom_linker_symbol_set_attr_id_t attr_id =
      (loom_linker_symbol_set_attr_id_t)index->attr_count++;
  index->attrs[attr_id] = attr;
  for (uint16_t i = 0; i < attr->count; ++i) {
    const uint16_t symbol_id = attr->symbol_refs[i].symbol_id;
    const loom_linker_symbol_set_use_id_t use_id =
        (loom_linker_symbol_set_use_id_t)index->use_count++;
    index->uses[use_id] = (loom_linker_symbol_set_use_t){
        .attr_id = attr_id,
        .next_use_id = index->symbol_use_heads[symbol_id],
    };
    index->symbol_use_heads[symbol_id] = use_id;
  }
  return iree_ok_status();
}

static iree_status_t loom_linker_register_op_symbol_sets(void* user_data,
                                                         loom_op_t* op) {
  loom_linker_t* linker = (loom_linker_t*)user_data;
  loom_attribute_t* attrs = loom_op_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    if (attrs[i].kind != LOOM_ATTR_SYMBOL_SET) continue;
    IREE_RETURN_IF_ERROR(loom_linker_register_symbol_set(linker, &attrs[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_linker_recanonicalize_symbol_uses(
    loom_linker_t* linker, uint16_t symbol_id) {
  loom_linker_symbol_set_index_t* index = &linker->symbol_set_index;
  loom_linker_symbol_set_use_id_t use_id = index->symbol_use_heads[symbol_id];
  while (use_id != LOOM_LINKER_SYMBOL_SET_USE_ID_INVALID) {
    const loom_linker_symbol_set_use_t* use = &index->uses[use_id];
    loom_attribute_t* attr = index->attrs[use->attr_id];
    const uint16_t ref_count = attr->count;
    loom_symbol_ref_t* replacement_refs = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &linker->target_module->arena, ref_count, sizeof(*replacement_refs),
        (void**)&replacement_refs));
    memcpy(replacement_refs, attr->symbol_refs,
           (iree_host_size_t)ref_count * sizeof(*replacement_refs));
    loom_symbol_ref_t duplicate_ref = loom_module_canonicalize_symbol_set(
        linker->target_module, replacement_refs, ref_count);
    IREE_ASSERT(!loom_symbol_ref_is_valid(duplicate_ref));
    *attr = loom_attr_symbol_set(replacement_refs, ref_count);
    use_id = use->next_use_id;
  }
  return iree_ok_status();
}

static iree_status_t loom_linker_add_target_symbol(
    loom_linker_t* linker, loom_string_id_t target_name_id,
    uint16_t* out_target_symbol_id) {
  IREE_RETURN_IF_ERROR(loom_linker_ensure_symbol_set_capacity(
      linker, linker->target_module->symbols.count + 1));
  uint16_t target_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(
      linker->target_module, target_name_id, &target_symbol_id));
  IREE_RETURN_IF_ERROR(loom_symbol_map_insert(
      &linker->target_symbol_lookup, &linker->scratch_arena, target_name_id,
      target_symbol_id));
  *out_target_symbol_id = target_symbol_id;
  return iree_ok_status();
}

static iree_status_t loom_linker_rename_private_target_symbol(
    loom_linker_t* linker, uint16_t target_symbol_id) {
  if (target_symbol_id >= linker->target_module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target symbol %u is out of range",
                            (unsigned)target_symbol_id);
  }

  loom_symbol_t* symbol =
      &linker->target_module->symbols.entries[target_symbol_id];
  loom_string_id_t old_name_id = symbol->name_id;
  iree_string_view_t old_name =
      linker->target_module->strings.entries[old_name_id];

  loom_string_id_t new_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_linker_allocate_fresh_private_name(linker, old_name, &new_name_id));
  if (!loom_symbol_map_erase(&linker->target_symbol_lookup, old_name_id)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "target symbol map did not contain '@%.*s'",
                            (int)old_name.size, old_name.data);
  }
  symbol->name_id = new_name_id;
  IREE_RETURN_IF_ERROR(loom_symbol_map_insert(&linker->target_symbol_lookup,
                                              &linker->scratch_arena,
                                              new_name_id, target_symbol_id));
  return loom_linker_recanonicalize_symbol_uses(linker, target_symbol_id);
}

static iree_status_t loom_linker_resolve_source_symbol(
    loom_linker_source_t* source, uint16_t source_symbol_id,
    loom_symbol_ref_t* target_ref_slot, loom_symbol_ref_t* out_target_ref) {
  loom_symbol_ref_t cached_ref = *target_ref_slot;
  if (loom_symbol_ref_is_valid(cached_ref)) {
    *out_target_ref = cached_ref;
    return iree_ok_status();
  }

  loom_linker_t* linker = source->linker;
  const loom_symbol_t* source_symbol =
      &source->module->symbols.entries[source_symbol_id];
  iree_string_view_t source_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_link_source_symbol_name(
      source->module, source_symbol_id, &source_name));

  loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(linker->target_module,
                                                 source_name, &target_name_id));
  uint16_t target_symbol_id =
      loom_symbol_map_find(&linker->target_symbol_lookup, target_name_id);

  const bool source_global =
      loom_link_symbol_has_global_identity(source->module, source_symbol);
  const bool source_concrete =
      loom_link_symbol_is_concrete_definition(source_symbol);

  const bool target_planned_global =
      target_symbol_id != LOOM_SYMBOL_ID_INVALID &&
      target_symbol_id < linker->planned.capacity &&
      linker->planned.symbols[target_symbol_id].global_identity;
  if (source_global && target_symbol_id != LOOM_SYMBOL_ID_INVALID &&
      !target_planned_global &&
      loom_link_target_symbol_is_private_concrete(linker, target_symbol_id)) {
    IREE_RETURN_IF_ERROR(
        loom_linker_rename_private_target_symbol(linker, target_symbol_id));
    target_symbol_id = LOOM_SYMBOL_ID_INVALID;
  }

  if (source_concrete && !source_global &&
      target_symbol_id != LOOM_SYMBOL_ID_INVALID) {
    const loom_symbol_t* target_symbol =
        &linker->target_module->symbols.entries[target_symbol_id];
    if (target_symbol->defining_op &&
        !loom_link_symbol_is_declaration(target_symbol)) {
      IREE_RETURN_IF_ERROR(loom_linker_allocate_fresh_private_name(
          linker, source_name, &target_name_id));
      target_symbol_id = LOOM_SYMBOL_ID_INVALID;
    }
  }

  if (target_symbol_id == LOOM_SYMBOL_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_linker_add_target_symbol(linker, target_name_id,
                                                       &target_symbol_id));
  }
  if (source_global && target_symbol_id < linker->planned.capacity) {
    linker->planned.symbols[target_symbol_id].global_identity = true;
  }

  loom_symbol_ref_t target_ref = {
      .module_id = 0,
      .symbol_id = target_symbol_id,
  };
  *target_ref_slot = target_ref;
  *out_target_ref = target_ref;
  return iree_ok_status();
}

static iree_status_t loom_linker_map_source_symbol(
    loom_linker_source_t* source, uint16_t source_symbol_id,
    loom_symbol_ref_t* out_target_ref) {
  *out_target_ref = loom_symbol_ref_null();
  if (source_symbol_id >= source->source_symbol_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source symbol ref {module=0, symbol=%u} is out of range",
        (unsigned)source_symbol_id);
  }
  return loom_linker_resolve_source_symbol(
      source, source_symbol_id, &source->target_symbols[source_symbol_id],
      out_target_ref);
}

static iree_host_size_t loom_linker_find_exact_source_symbol(
    const loom_linker_source_t* source, uint16_t source_symbol_id) {
  iree_host_size_t low = 0;
  iree_host_size_t high = source->exact.count;
  while (low < high) {
    const iree_host_size_t middle = low + (high - low) / 2;
    const iree_host_size_t ordinal = source->exact.ordinals[middle];
    if (ordinal < source_symbol_id) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low < source->exact.count &&
                 source->exact.ordinals[low] == source_symbol_id
             ? low
             : IREE_HOST_SIZE_MAX;
}

static iree_status_t loom_linker_map_dense_source_symbol(
    loom_linker_source_t* source, uint16_t source_symbol_id,
    loom_symbol_ref_t* out_target_ref) {
  *out_target_ref = loom_symbol_ref_null();
  return loom_linker_resolve_source_symbol(
      source, source_symbol_id, &source->target_symbols[source_symbol_id],
      out_target_ref);
}

static iree_status_t loom_linker_map_exact_source_symbol(
    loom_linker_source_t* source, uint16_t source_symbol_id,
    loom_symbol_ref_t* out_target_ref) {
  *out_target_ref = loom_symbol_ref_null();
  const iree_host_size_t selection_ordinal =
      loom_linker_find_exact_source_symbol(source, source_symbol_id);
  if (selection_ordinal == IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "exact link selection missed reachable source symbol ref "
        "{module=0, symbol=%u}",
        (unsigned)source_symbol_id);
  }
  return loom_linker_resolve_source_symbol(
      source, source_symbol_id, &source->target_symbols[selection_ordinal],
      out_target_ref);
}

static iree_host_size_t loom_linker_find_source_binding(
    loom_linker_source_symbol_binding_list_t bindings,
    uint16_t source_symbol_id) {
  iree_host_size_t low = 0;
  iree_host_size_t high = bindings.count;
  while (low < high) {
    const iree_host_size_t middle = low + (high - low) / 2;
    const uint32_t candidate = bindings.values[middle].source_ordinal;
    if (candidate < source_symbol_id) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low < bindings.count &&
                 bindings.values[low].source_ordinal == source_symbol_id
             ? low
             : IREE_HOST_SIZE_MAX;
}

static iree_status_t loom_linker_map_bound_exact_source_symbol(
    loom_linker_source_t* source, uint16_t source_symbol_id,
    loom_symbol_ref_t* out_target_ref) {
  const iree_host_size_t selection_ordinal =
      loom_linker_find_exact_source_symbol(source, source_symbol_id);
  if (selection_ordinal != IREE_HOST_SIZE_MAX) {
    return loom_linker_resolve_source_symbol(
        source, source_symbol_id, &source->target_symbols[selection_ordinal],
        out_target_ref);
  }
  const iree_host_size_t binding_ordinal =
      loom_linker_find_source_binding(source->exact.bindings, source_symbol_id);
  if (binding_ordinal == IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "exact link selection missed reachable source symbol ref "
        "{module=0, symbol=%u}",
        (unsigned)source_symbol_id);
  }
  *out_target_ref = source->exact.bindings.values[binding_ordinal].target;
  return iree_ok_status();
}

static iree_status_t loom_linker_validate_symbol_remap(
    const loom_linker_source_t* source, const loom_module_t* source_module,
    loom_module_t* target_module, loom_symbol_ref_t source_ref) {
  if (source_module != source->module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "link symbol remap source module mismatch");
  }
  if (target_module != source->linker->target_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "link symbol remap target module mismatch");
  }
  if (source_ref.module_id != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source symbol ref {module=%u, symbol=%u} is not "
                            "module-local",
                            (unsigned)source_ref.module_id,
                            (unsigned)source_ref.symbol_id);
  }
  if (source_ref.symbol_id >= source->source_symbol_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source symbol ref {module=0, symbol=%u} is out of range",
        (unsigned)source_ref.symbol_id);
  }
  return iree_ok_status();
}

static iree_status_t loom_linker_remap_symbol(
    void* user_data, const loom_module_t* source_module,
    loom_module_t* target_module, loom_symbol_ref_t source_ref,
    loom_symbol_ref_t* out_target_ref) {
  loom_linker_source_t* source = (loom_linker_source_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_linker_validate_symbol_remap(
      source, source_module, target_module, source_ref));
  if (source->root_filtered && !source->live_symbols[source_ref.symbol_id]) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "root-filtered link missed reachable source symbol ref {module=0, "
        "symbol=%u}",
        (unsigned)source_ref.symbol_id);
  }
  return loom_linker_map_source_symbol(source, source_ref.symbol_id,
                                       out_target_ref);
}

static iree_status_t loom_linker_remap_exact_symbol(
    void* user_data, const loom_module_t* source_module,
    loom_module_t* target_module, loom_symbol_ref_t source_ref,
    loom_symbol_ref_t* out_target_ref) {
  loom_linker_source_t* source = (loom_linker_source_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_linker_validate_symbol_remap(
      source, source_module, target_module, source_ref));
  return loom_linker_map_exact_source_symbol(source, source_ref.symbol_id,
                                             out_target_ref);
}

static iree_status_t loom_linker_remap_bound_exact_symbol(
    void* user_data, const loom_module_t* source_module,
    loom_module_t* target_module, loom_symbol_ref_t source_ref,
    loom_symbol_ref_t* out_target_ref) {
  loom_linker_source_t* source = (loom_linker_source_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_linker_validate_symbol_remap(
      source, source_module, target_module, source_ref));
  return loom_linker_map_bound_exact_source_symbol(source, source_ref.symbol_id,
                                                   out_target_ref);
}

static iree_status_t loom_linker_remap_dense_symbol(
    void* user_data, const loom_module_t* source_module,
    loom_module_t* target_module, loom_symbol_ref_t source_ref,
    loom_symbol_ref_t* out_target_ref) {
  loom_linker_source_t* source = (loom_linker_source_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_linker_validate_symbol_remap(
      source, source_module, target_module, source_ref));
  return loom_linker_map_dense_source_symbol(source, source_ref.symbol_id,
                                             out_target_ref);
}

static iree_status_t loom_linker_get_source_remap(loom_linker_source_t* source,
                                                  loom_ir_remap_t** out_remap) {
  if (!source->remap.source_module) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
        source->module, source->linker->target_module, source->arena,
        &(loom_ir_remap_options_t){
            .remap_symbol = source->symbol_remap,
        },
        &source->remap));
  }
  *out_remap = &source->remap;
  return iree_ok_status();
}

static bool loom_link_op_symbol_ref(const loom_module_t* module,
                                    const loom_op_t* op,
                                    loom_symbol_ref_t* out_ref) {
  *out_ref = loom_symbol_ref_null();
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !vtable->symbol_def || !vtable->attr_descriptors) {
    return false;
  }
  uint8_t symbol_attr_index = vtable->symbol_def->name_attr_index;
  if (symbol_attr_index >= vtable->attribute_count ||
      symbol_attr_index >= op->attribute_count) {
    return false;
  }
  const loom_attr_descriptor_t* descriptor =
      &vtable->attr_descriptors[symbol_attr_index];
  if (descriptor->attr_kind != LOOM_ATTR_SYMBOL) {
    return false;
  }
  *out_ref = loom_attr_as_symbol(loom_op_const_attrs(op)[symbol_attr_index]);
  return loom_symbol_ref_is_valid(*out_ref) && out_ref->module_id == 0 &&
         out_ref->symbol_id < module->symbols.count;
}

static const loom_symbol_definition_descriptor_t*
loom_link_op_symbol_definition(const loom_module_t* module,
                               const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  return vtable ? vtable->symbol_def : NULL;
}

static iree_status_t loom_link_duplicate_value_definition_status(
    const loom_linker_t* linker, loom_symbol_ref_t target_ref,
    iree_string_view_t field_name) {
  iree_string_view_t name =
      loom_link_target_symbol_name(linker->target_module, target_ref);
  return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                          "duplicate value definition '@%.*s' has "
                          "incompatible contract field '%.*s'",
                          (int)name.size, name.data, (int)field_name.size,
                          field_name.data);
}

static iree_status_t loom_link_remap_value_definition(
    loom_linker_t* linker, const loom_module_t* source_module,
    const loom_op_t* source_op, iree_arena_allocator_t* arena,
    loom_type_t* out_type, loom_attribute_t* out_value) {
  const loom_symbol_definition_descriptor_t* definition =
      loom_link_op_symbol_definition(source_module, source_op);
  const uint8_t result_index =
      loom_symbol_definition_value_contract_result_index(definition);
  const uint8_t value_attr_index =
      loom_symbol_definition_value_contract_value_attr_index(definition);
  if (result_index == LOOM_RESULT_INDEX_NONE ||
      value_attr_index == LOOM_ATTR_INDEX_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol has no exact value contract");
  }
  loom_value_id_t source_value_id =
      loom_op_const_results(source_op)[result_index];
  if (source_value_id == LOOM_VALUE_ID_INVALID ||
      source_value_id >= source_module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value definition has no result value");
  }

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      source_module, linker->target_module, arena, /*options=*/NULL, &remap));
  IREE_RETURN_IF_ERROR(loom_ir_remap_type(
      &remap, loom_module_value_type(source_module, source_value_id),
      out_type));
  return loom_ir_remap_attribute(
      &remap, loom_op_const_attrs(source_op)[value_attr_index], out_value);
}

static iree_status_t loom_link_check_duplicate_value_definition(
    loom_linker_t* linker, loom_op_t* existing_op,
    const loom_module_t* source_module, loom_op_t* source_op,
    iree_arena_allocator_t* arena, loom_symbol_ref_t target_ref,
    bool* out_merge) {
  *out_merge = false;
  const loom_symbol_definition_descriptor_t* existing_definition =
      loom_link_op_symbol_definition(linker->target_module, existing_op);
  const loom_symbol_definition_descriptor_t* source_definition =
      loom_link_op_symbol_definition(source_module, source_op);
  if (!loom_symbol_definition_has_value_contract(existing_definition) ||
      !loom_symbol_definition_has_value_contract(source_definition) ||
      loom_symbol_definition_value_contract_value_attr_index(
          existing_definition) == LOOM_ATTR_INDEX_NONE ||
      loom_symbol_definition_value_contract_value_attr_index(
          source_definition) == LOOM_ATTR_INDEX_NONE ||
      existing_definition->interfaces != source_definition->interfaces) {
    return iree_ok_status();
  }

  loom_type_t existing_type = {0};
  loom_attribute_t existing_value = {0};
  IREE_RETURN_IF_ERROR(loom_link_remap_value_definition(
      linker, linker->target_module, existing_op, arena, &existing_type,
      &existing_value));

  loom_type_t new_type = {0};
  loom_attribute_t new_value = {0};
  IREE_RETURN_IF_ERROR(loom_link_remap_value_definition(
      linker, source_module, source_op, arena, &new_type, &new_value));

  if (!loom_type_equal(existing_type, new_type)) {
    return loom_link_duplicate_value_definition_status(linker, target_ref,
                                                       IREE_SV("type"));
  }
  if (!loom_attribute_equal(&existing_value, &new_value)) {
    return loom_link_duplicate_value_definition_status(linker, target_ref,
                                                       IREE_SV("value"));
  }

  *out_merge = true;
  return iree_ok_status();
}

static iree_status_t loom_link_incompatible_contract_status(
    const loom_linker_t* linker, loom_symbol_ref_t target_ref,
    iree_string_view_t field_name) {
  iree_string_view_t symbol_name =
      loom_link_target_symbol_name(linker->target_module, target_ref);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "linked declaration for '@%.*s' has incompatible "
                          "contract field '%.*s'",
                          (int)symbol_name.size, symbol_name.data,
                          (int)field_name.size, field_name.data);
}

static iree_status_t loom_link_signature_count_status(
    const loom_linker_t* linker, loom_symbol_ref_t target_ref,
    iree_string_view_t field_name, iree_host_size_t source_count,
    iree_host_size_t target_count) {
  iree_string_view_t symbol_name =
      loom_link_target_symbol_name(linker->target_module, target_ref);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "linked declaration for '@%.*s' has %" PRIhsz
                          " %.*s but selected "
                          "symbol has %" PRIhsz,
                          (int)symbol_name.size, symbol_name.data, source_count,
                          (int)field_name.size, field_name.data, target_count);
}

static iree_status_t loom_link_signature_type_status(
    const loom_linker_t* linker, loom_symbol_ref_t target_ref,
    iree_string_view_t field_name, iree_host_size_t ordinal) {
  iree_string_view_t symbol_name =
      loom_link_target_symbol_name(linker->target_module, target_ref);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "linked declaration for '@%.*s' has incompatible "
                          "%.*s type at index %" PRIhsz,
                          (int)symbol_name.size, symbol_name.data,
                          (int)field_name.size, field_name.data, ordinal);
}

static iree_status_t loom_link_func_contract_status(
    const loom_linker_t* linker, loom_symbol_ref_t target_ref,
    const loom_link_func_contract_mismatch_t* mismatch) {
  switch (mismatch->kind) {
    case LOOM_LINK_FUNC_CONTRACT_MISMATCH_NONE:
      return iree_ok_status();
    case LOOM_LINK_FUNC_CONTRACT_MISMATCH_FIELD:
      return loom_link_incompatible_contract_status(linker, target_ref,
                                                    mismatch->field_name);
    case LOOM_LINK_FUNC_CONTRACT_MISMATCH_COUNT:
      return loom_link_signature_count_status(
          linker, target_ref, mismatch->field_name,
          mismatch->detail.counts.source, mismatch->detail.counts.selected);
    case LOOM_LINK_FUNC_CONTRACT_MISMATCH_TYPE:
      return loom_link_signature_type_status(linker, target_ref,
                                             mismatch->field_name,
                                             mismatch->detail.type_ordinal);
    default:
      IREE_ASSERT_UNREACHABLE("unknown function contract mismatch");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_link_merge_func_contract(
    loom_linker_t* linker, loom_linker_source_t* link_source,
    const loom_module_t* source_module, loom_op_t* source_op,
    iree_arena_allocator_t* arena, loom_symbol_ref_t target_ref,
    loom_op_t* target_op, bool merge_output_contract) {
  if (source_op == target_op) {
    return iree_ok_status();
  }

  loom_ir_remap_symbol_callback_t symbol_callback =
      loom_ir_remap_symbol_callback_empty();
  if (source_module != linker->target_module) {
    if (!link_source) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "cross-module contract merge requires source "
                              "remap state");
    }
    symbol_callback = link_source->symbol_remap;
  }

  loom_ir_remap_t contract_remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_initialize(source_module, linker->target_module, arena,
                               &(loom_ir_remap_options_t){
                                   .remap_symbol = symbol_callback,
                               },
                               &contract_remap));

  loom_link_func_contract_t source_contract =
      loom_link_func_contract_from_op(source_module, source_op);
  loom_link_func_contract_t selected_contract =
      loom_link_func_contract_from_op(linker->target_module, target_op);
  loom_link_func_contract_mismatch_t mismatch = {0};
  const loom_link_func_contract_merge_flags_t flags =
      merge_output_contract ? LOOM_LINK_FUNC_CONTRACT_MERGE_FLAG_OUTPUT : 0;
  IREE_RETURN_IF_ERROR(loom_link_func_contract_merge(
      &source_contract, &selected_contract, &contract_remap, flags, &mismatch));
  return loom_link_func_contract_status(linker, target_ref, &mismatch);
}

static iree_status_t loom_link_incompatible_value_contract_status(
    const loom_linker_t* linker, loom_symbol_ref_t target_ref,
    iree_string_view_t field_name) {
  iree_string_view_t symbol_name =
      loom_link_target_symbol_name(linker->target_module, target_ref);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "linked value declaration for '@%.*s' has "
                          "incompatible contract field '%.*s'",
                          (int)symbol_name.size, symbol_name.data,
                          (int)field_name.size, field_name.data);
}

static iree_status_t loom_link_append_value_contract_predicates(
    loom_linker_t* linker, loom_symbol_ref_t target_ref, loom_op_t* target_op,
    const loom_symbol_definition_descriptor_t* target_definition,
    loom_attribute_t predicates) {
  if (predicates.kind == LOOM_ATTR_ABSENT || predicates.count == 0) {
    return iree_ok_status();
  }
  const uint8_t target_predicates_attr_index =
      loom_symbol_definition_value_contract_predicates_attr_index(
          target_definition);
  if (predicates.kind != LOOM_ATTR_PREDICATE_LIST ||
      !predicates.predicate_list ||
      target_predicates_attr_index == LOOM_ATTR_INDEX_NONE ||
      target_predicates_attr_index >= target_op->attribute_count) {
    return loom_link_incompatible_value_contract_status(linker, target_ref,
                                                        IREE_SV("predicates"));
  }

  loom_attribute_t old_predicates =
      loom_op_attrs(target_op)[target_predicates_attr_index];
  if (old_predicates.kind != LOOM_ATTR_ABSENT &&
      old_predicates.kind != LOOM_ATTR_PREDICATE_LIST) {
    return loom_link_incompatible_value_contract_status(linker, target_ref,
                                                        IREE_SV("predicates"));
  }
  if (old_predicates.count > 0 && !old_predicates.predicate_list) {
    return loom_link_incompatible_value_contract_status(linker, target_ref,
                                                        IREE_SV("predicates"));
  }

  iree_host_size_t total_count =
      (iree_host_size_t)old_predicates.count + predicates.count;
  if (total_count > UINT16_MAX) {
    iree_string_view_t symbol_name =
        loom_link_target_symbol_name(linker->target_module, target_ref);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "linked value declaration for '@%.*s' has %" PRIhsz
                            " merged predicates, max %u",
                            (int)symbol_name.size, symbol_name.data,
                            total_count, (unsigned)UINT16_MAX);
  }

  loom_predicate_t* merged_predicates = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &linker->target_module->arena, total_count, sizeof(*merged_predicates),
      (void**)&merged_predicates));
  if (old_predicates.count > 0) {
    memcpy(merged_predicates, old_predicates.predicate_list,
           (iree_host_size_t)old_predicates.count * sizeof(*merged_predicates));
  }
  memcpy(merged_predicates + old_predicates.count, predicates.predicate_list,
         (iree_host_size_t)predicates.count * sizeof(*merged_predicates));
  loom_op_attrs(target_op)[target_predicates_attr_index] =
      loom_attr_predicate_list(merged_predicates, (uint16_t)total_count);
  return iree_ok_status();
}

static iree_status_t loom_link_merge_value_contract(
    loom_linker_t* linker, loom_linker_source_t* link_source,
    const loom_module_t* source_module, loom_op_t* source_op,
    iree_arena_allocator_t* arena, loom_symbol_ref_t target_ref,
    loom_op_t* target_op) {
  if (source_op == target_op) return iree_ok_status();
  const loom_symbol_definition_descriptor_t* source_definition =
      loom_link_op_symbol_definition(source_module, source_op);
  const loom_symbol_definition_descriptor_t* target_definition =
      loom_link_op_symbol_definition(linker->target_module, target_op);
  if (!loom_symbol_definition_has_value_contract(source_definition) ||
      !loom_symbol_definition_has_value_contract(target_definition)) {
    return loom_link_incompatible_value_contract_status(
        linker, target_ref, IREE_SV("value contract"));
  }

  const uint8_t source_result_index =
      loom_symbol_definition_value_contract_result_index(source_definition);
  const uint8_t target_result_index =
      loom_symbol_definition_value_contract_result_index(target_definition);
  if (source_result_index == LOOM_RESULT_INDEX_NONE ||
      target_result_index == LOOM_RESULT_INDEX_NONE ||
      source_result_index >= source_op->result_count ||
      target_result_index >= target_op->result_count) {
    return loom_link_incompatible_value_contract_status(linker, target_ref,
                                                        IREE_SV("result"));
  }
  loom_value_id_t source_value_id =
      loom_op_const_results(source_op)[source_result_index];
  loom_value_id_t target_value_id =
      loom_op_const_results(target_op)[target_result_index];
  loom_type_t target_type =
      loom_module_value_type(linker->target_module, target_value_id);

  loom_ir_remap_symbol_callback_t symbol_callback =
      loom_ir_remap_symbol_callback_empty();
  if (source_module != linker->target_module) {
    if (!link_source) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "cross-module contract merge requires source "
                              "remap state");
    }
    symbol_callback = link_source->symbol_remap;
  }

  loom_ir_remap_t contract_remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_initialize(source_module, linker->target_module, arena,
                               &(loom_ir_remap_options_t){
                                   .remap_symbol = symbol_callback,
                               },
                               &contract_remap));
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(&contract_remap, source_value_id,
                                               target_value_id));
  loom_type_t remapped_source_type = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_type(
      &contract_remap, loom_module_value_type(source_module, source_value_id),
      &remapped_source_type));
  if (!loom_type_equal(remapped_source_type, target_type)) {
    return loom_link_incompatible_value_contract_status(linker, target_ref,
                                                        IREE_SV("type"));
  }

  const uint8_t source_value_attr_index =
      loom_symbol_definition_value_contract_value_attr_index(source_definition);
  const uint8_t target_value_attr_index =
      loom_symbol_definition_value_contract_value_attr_index(target_definition);
  loom_attribute_t remapped_source_value = loom_attr_absent();
  if (source_value_attr_index != LOOM_ATTR_INDEX_NONE) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_attribute(
        &contract_remap,
        loom_op_const_attrs(source_op)[source_value_attr_index],
        &remapped_source_value));
  }
  loom_attribute_t* target_value = NULL;
  if (target_value_attr_index != LOOM_ATTR_INDEX_NONE) {
    target_value = &loom_op_attrs(target_op)[target_value_attr_index];
  }
  if (!loom_attr_is_absent(remapped_source_value)) {
    if (!target_value) {
      return loom_link_incompatible_value_contract_status(linker, target_ref,
                                                          IREE_SV("value"));
    }
    if (loom_attr_is_absent(*target_value)) {
      *target_value = remapped_source_value;
    } else if (!loom_attribute_equal(target_value, &remapped_source_value)) {
      return loom_link_incompatible_value_contract_status(linker, target_ref,
                                                          IREE_SV("value"));
    }
  }

  const uint8_t source_predicates_attr_index =
      loom_symbol_definition_value_contract_predicates_attr_index(
          source_definition);
  if (source_predicates_attr_index == LOOM_ATTR_INDEX_NONE) {
    return iree_ok_status();
  }
  loom_attribute_t source_predicates =
      loom_op_const_attrs(source_op)[source_predicates_attr_index];
  if (loom_attr_is_absent(source_predicates) || source_predicates.count == 0) {
    return iree_ok_status();
  }
  if (source_predicates.kind != LOOM_ATTR_PREDICATE_LIST ||
      !source_predicates.predicate_list) {
    return loom_link_incompatible_value_contract_status(linker, target_ref,
                                                        IREE_SV("predicates"));
  }
  loom_predicate_t* remapped_predicates = NULL;
  IREE_RETURN_IF_ERROR(loom_ir_remap_predicate_list(
      &contract_remap, source_predicates.predicate_list,
      source_predicates.count, &remapped_predicates));
  loom_attribute_t predicate_attr =
      loom_attr_predicate_list(remapped_predicates, source_predicates.count);

  if (target_value && !loom_attr_is_absent(*target_value)) {
    return loom_symbol_value_constraints_check_exact(
        loom_link_target_symbol_name(linker->target_module, target_ref),
        target_type, target_value_id, *target_value, predicate_attr);
  }
  return loom_link_append_value_contract_predicates(
      linker, target_ref, target_op, target_definition, predicate_attr);
}

static iree_status_t loom_link_merge_symbol_contract(
    loom_linker_t* linker, loom_linker_source_t* link_source,
    const loom_module_t* source_module, loom_op_t* source_op,
    iree_arena_allocator_t* arena, loom_symbol_ref_t target_ref,
    loom_op_t* target_op, bool merge_output_contract) {
  if (source_op == target_op) return iree_ok_status();
  const loom_symbol_definition_descriptor_t* source_definition =
      loom_link_op_symbol_definition(source_module, source_op);
  const loom_symbol_definition_descriptor_t* target_definition =
      loom_link_op_symbol_definition(linker->target_module, target_op);
  if (!source_definition ||
      !loom_symbol_definition_satisfies(target_definition,
                                        source_definition->interfaces)) {
    return loom_link_incompatible_contract_status(linker, target_ref,
                                                  IREE_SV("symbol interfaces"));
  }

  if (loom_symbol_definition_implements(source_definition,
                                        LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
    IREE_RETURN_IF_ERROR(loom_link_merge_func_contract(
        linker, link_source, source_module, source_op, arena, target_ref,
        target_op, merge_output_contract));
  }
  if (loom_symbol_definition_has_value_contract(source_definition)) {
    IREE_RETURN_IF_ERROR(loom_link_merge_value_contract(
        linker, link_source, source_module, source_op, arena, target_ref,
        target_op));
  }
  return iree_ok_status();
}

static iree_status_t loom_linker_clone_source_op(loom_linker_source_t* source,
                                                 const loom_op_t* source_op,
                                                 loom_op_t* before_op,
                                                 loom_op_t** out_cloned_op) {
  loom_builder_t builder;
  loom_builder_initialize(
      source->linker->target_module, &source->linker->target_module->arena,
      loom_module_block(source->linker->target_module), &builder);
  builder.on_op_finalized.fn = loom_linker_register_op_symbol_sets;
  builder.on_op_finalized.user_data = source->linker;
  if (before_op) {
    loom_builder_set_before(&builder, before_op);
  }

  loom_ir_remap_t* remap = NULL;
  IREE_RETURN_IF_ERROR(loom_linker_get_source_remap(source, &remap));
  return loom_ir_clone_op(&builder, source_op, remap, out_cloned_op);
}

static loom_linker_symbol_output_t loom_linker_combine_symbol_output(
    loom_linker_symbol_output_t lhs, loom_linker_symbol_output_t rhs) {
  if (lhs == LOOM_LINKER_SYMBOL_OUTPUT_ROOT ||
      rhs == LOOM_LINKER_SYMBOL_OUTPUT_ROOT) {
    return LOOM_LINKER_SYMBOL_OUTPUT_ROOT;
  }
  if (lhs == LOOM_LINKER_SYMBOL_OUTPUT_DEPENDENCY ||
      rhs == LOOM_LINKER_SYMBOL_OUTPUT_DEPENDENCY) {
    return LOOM_LINKER_SYMBOL_OUTPUT_DEPENDENCY;
  }
  return LOOM_LINKER_SYMBOL_OUTPUT_AUTHORED;
}

typedef struct loom_link_func_output_attr_t {
  // Root operation attribute index.
  uint8_t source_attr_index;
  // Selected operation attribute index.
  uint8_t target_attr_index;
  // Stable field name used in diagnostics.
  iree_string_view_t field_name;
} loom_link_func_output_attr_t;

static iree_status_t loom_linker_replace_output_attr(
    loom_linker_t* linker, loom_symbol_ref_t target_ref, loom_ir_remap_t* remap,
    const loom_op_t* source_op, uint8_t source_attr_index, loom_op_t* target_op,
    uint8_t target_attr_index, iree_string_view_t field_name) {
  const bool source_present =
      source_attr_index != LOOM_ATTR_INDEX_NONE &&
      !loom_attr_is_absent(loom_op_const_attrs(source_op)[source_attr_index]);
  if (target_attr_index == LOOM_ATTR_INDEX_NONE) {
    return source_present ? loom_link_incompatible_contract_status(
                                linker, target_ref, field_name)
                          : iree_ok_status();
  }
  if (!source_present) {
    loom_op_attrs(target_op)[target_attr_index] = loom_attr_absent();
    return iree_ok_status();
  }
  loom_attribute_t source_attr =
      loom_op_const_attrs(source_op)[source_attr_index];
  if (remap) {
    return loom_ir_remap_attribute(
        remap, source_attr, &loom_op_attrs(target_op)[target_attr_index]);
  }
  loom_op_attrs(target_op)[target_attr_index] = source_attr;
  return iree_ok_status();
}

static iree_status_t loom_linker_apply_root_symbol_output(
    loom_linker_source_t* source, const loom_module_t* root_module,
    loom_op_t* root_op, loom_symbol_ref_t target_ref, loom_op_t* target_op) {
  loom_linker_t* linker = source->linker;
  const loom_op_vtable_t* target_vtable =
      loom_op_vtable(linker->target_module, target_op);
  const loom_symbol_definition_descriptor_t* target_definition =
      target_vtable->symbol_def;
  const uint8_t target_visibility_attr_index =
      loom_symbol_definition_visibility_attr_index(target_definition);
  const uint8_t target_retain_attr_index =
      target_definition->retain_attr_index_plus_one
          ? target_definition->retain_attr_index_plus_one - 1
          : LOOM_ATTR_INDEX_NONE;
  const loom_op_vtable_t* root_vtable = loom_op_vtable(root_module, root_op);
  const loom_symbol_definition_descriptor_t* root_definition =
      root_vtable->symbol_def;
  loom_ir_remap_t* remap = NULL;
  if (root_module != linker->target_module) {
    IREE_RETURN_IF_ERROR(loom_linker_get_source_remap(source, &remap));
  }

  IREE_RETURN_IF_ERROR(loom_linker_replace_output_attr(
      linker, target_ref, remap, root_op,
      loom_symbol_definition_visibility_attr_index(root_definition), target_op,
      target_visibility_attr_index, IREE_SV("visibility")));

  loom_func_like_t root_func = loom_func_like_cast(root_module, root_op);
  loom_func_like_t target_func =
      loom_func_like_cast(linker->target_module, target_op);
  if (loom_func_like_isa(root_func) || loom_func_like_isa(target_func)) {
    if (!loom_func_like_isa(root_func) || !loom_func_like_isa(target_func)) {
      return loom_link_incompatible_contract_status(linker, target_ref,
                                                    IREE_SV("func_like"));
    }
    const bool target_declaration =
        loom_symbol_definition_is_declaration(target_definition);
    const loom_link_func_output_attr_t export_attrs[] = {
        {
            .source_attr_index = root_func.vtable->export_symbol_attr_index,
            .target_attr_index = target_func.vtable->export_symbol_attr_index,
            .field_name = IREE_SV("export_symbol"),
        },
        {
            .source_attr_index = root_func.vtable->export_attrs_attr_index,
            .target_attr_index = target_func.vtable->export_attrs_attr_index,
            .field_name = IREE_SV("export_attrs"),
        },
        {
            .source_attr_index = root_func.vtable->export_linkage_attr_index,
            .target_attr_index = target_func.vtable->export_linkage_attr_index,
            .field_name = IREE_SV("export_linkage"),
        },
    };
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(export_attrs); ++i) {
      IREE_RETURN_IF_ERROR(loom_linker_replace_output_attr(
          linker, target_ref, remap, root_op, export_attrs[i].source_attr_index,
          target_op, export_attrs[i].target_attr_index,
          export_attrs[i].field_name));
    }
    const loom_link_func_output_attr_t import_attrs[] = {
        {
            .source_attr_index = root_func.vtable->import_module_attr_index,
            .target_attr_index = target_func.vtable->import_module_attr_index,
            .field_name = IREE_SV("import_module"),
        },
        {
            .source_attr_index = root_func.vtable->import_symbol_attr_index,
            .target_attr_index = target_func.vtable->import_symbol_attr_index,
            .field_name = IREE_SV("import_symbol"),
        },
    };
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(import_attrs); ++i) {
      IREE_RETURN_IF_ERROR(loom_linker_replace_output_attr(
          linker, target_ref, target_declaration ? remap : NULL,
          target_declaration ? root_op : target_op,
          target_declaration ? import_attrs[i].source_attr_index
                             : LOOM_ATTR_INDEX_NONE,
          target_op, import_attrs[i].target_attr_index,
          import_attrs[i].field_name));
    }
  }
  if (target_retain_attr_index != LOOM_ATTR_INDEX_NONE) {
    loom_op_attrs(target_op)[target_retain_attr_index] = loom_attr_enum(1);
  }
  loom_module_link_symbol_defining_op(linker->target_module, target_op,
                                      target_vtable);
  return iree_ok_status();
}

static iree_status_t loom_linker_apply_symbol_output(
    loom_linker_source_t* source, const loom_module_t* root_module,
    loom_op_t* root_op, loom_symbol_ref_t target_ref, loom_op_t* target_op,
    loom_linker_symbol_output_t output) {
  if (output == LOOM_LINKER_SYMBOL_OUTPUT_AUTHORED) {
    return iree_ok_status();
  }
  if (output == LOOM_LINKER_SYMBOL_OUTPUT_DEPENDENCY) {
    loom_link_symbol_internalize(source->linker->target_module, target_op);
    return iree_ok_status();
  }
  return loom_linker_apply_root_symbol_output(source, root_module, root_op,
                                              target_ref, target_op);
}

static iree_status_t loom_linker_commit_symbol_output(
    loom_linker_source_t* source, const loom_module_t* root_module,
    loom_op_t* root_op, loom_symbol_ref_t target_ref, loom_op_t* target_op,
    loom_linker_symbol_output_t output) {
  IREE_RETURN_IF_ERROR(loom_linker_apply_symbol_output(
      source, root_module, root_op, target_ref, target_op, output));
  if (output == LOOM_LINKER_SYMBOL_OUTPUT_AUTHORED) {
    return iree_ok_status();
  }
  loom_linker_t* linker = source->linker;
  linker->planned.symbols[target_ref.symbol_id].output = output;
  return iree_ok_status();
}

static iree_status_t loom_linker_duplicate_concrete_status(
    const loom_linker_t* linker, loom_symbol_ref_t target_ref) {
  iree_string_view_t name =
      loom_link_target_symbol_name(linker->target_module, target_ref);
  return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                          "duplicate concrete symbol definition '@%.*s'",
                          (int)name.size, name.data);
}

static iree_status_t loom_linker_clone_or_merge_symbol_op(
    loom_linker_source_t* source, uint16_t source_symbol_id,
    loom_symbol_ref_t target_ref, loom_linker_symbol_output_t source_output) {
  loom_linker_t* linker = source->linker;
  const loom_symbol_t* source_symbol =
      &source->module->symbols.entries[source_symbol_id];
  loom_op_t* source_op = source_symbol->defining_op;
  if (!source_op) return iree_ok_status();

  loom_symbol_t* target_symbol =
      &linker->target_module->symbols.entries[target_ref.symbol_id];
  loom_op_t* target_op = target_symbol->defining_op;
  const bool source_declaration =
      loom_link_symbol_is_declaration(source_symbol);
  const loom_linker_symbol_output_t target_output =
      target_ref.symbol_id < linker->planned.capacity
          ? linker->planned.symbols[target_ref.symbol_id].output
          : LOOM_LINKER_SYMBOL_OUTPUT_AUTHORED;
  const loom_linker_symbol_output_t output =
      loom_linker_combine_symbol_output(source_output, target_output);
  const loom_module_t* root_module = NULL;
  loom_op_t* root_op = NULL;
  if (target_output == LOOM_LINKER_SYMBOL_OUTPUT_ROOT) {
    root_module = linker->target_module;
    root_op = target_op;
  } else if (source_output == LOOM_LINKER_SYMBOL_OUTPUT_ROOT) {
    root_module = source->module;
    root_op = source_op;
  }

  if (!target_op) {
    loom_op_t* cloned_op = NULL;
    IREE_RETURN_IF_ERROR(loom_linker_clone_source_op(
        source, source_op, /*before_op=*/NULL, &cloned_op));
    return loom_linker_commit_symbol_output(source, root_module, root_op,
                                            target_ref, cloned_op, output);
  }

  const bool target_declaration =
      loom_link_symbol_is_declaration(target_symbol);
  if (target_declaration && !source_declaration) {
    loom_op_t* cloned_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_linker_clone_source_op(source, source_op, target_op, &cloned_op));
    IREE_RETURN_IF_ERROR(loom_link_merge_symbol_contract(
        linker, source, linker->target_module, target_op, source->arena,
        target_ref, cloned_op, output == LOOM_LINKER_SYMBOL_OUTPUT_AUTHORED));
    IREE_RETURN_IF_ERROR(loom_linker_apply_symbol_output(
        source, root_module, root_op, target_ref, cloned_op, output));
    IREE_RETURN_IF_ERROR(loom_op_erase(linker->target_module, target_op));
    loom_module_link_symbol_defining_op(
        linker->target_module, cloned_op,
        loom_op_vtable(linker->target_module, cloned_op));
    if (output != LOOM_LINKER_SYMBOL_OUTPUT_AUTHORED) {
      linker->planned.symbols[target_ref.symbol_id].output = output;
    }
    return iree_ok_status();
  }

  if (source_declaration) {
    IREE_RETURN_IF_ERROR(loom_link_merge_symbol_contract(
        linker, source, source->module, source_op, source->arena, target_ref,
        target_op, output == LOOM_LINKER_SYMBOL_OUTPUT_AUTHORED));
    return loom_linker_commit_symbol_output(source, root_module, root_op,
                                            target_ref, target_op, output);
  }

  bool merge_duplicate_value_definition = false;
  IREE_RETURN_IF_ERROR(loom_link_check_duplicate_value_definition(
      linker, target_op, source->module, source_op, source->arena, target_ref,
      &merge_duplicate_value_definition));
  if (merge_duplicate_value_definition) {
    return loom_linker_commit_symbol_output(source, root_module, root_op,
                                            target_ref, target_op, output);
  }
  return loom_linker_duplicate_concrete_status(linker, target_ref);
}

static iree_status_t loom_linker_mark_source_symbol_live(
    loom_linker_source_t* source, uint16_t source_symbol_id) {
  if (source_symbol_id >= source->source_symbol_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source symbol %u is out of range",
                            (unsigned)source_symbol_id);
  }
  if (source->live_symbols[source_symbol_id]) {
    return iree_ok_status();
  }
  loom_symbol_ref_t target_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(
      loom_linker_map_source_symbol(source, source_symbol_id, &target_ref));
  source->live_symbols[source_symbol_id] = 1;
  return iree_ok_status();
}

static bool loom_linker_source_symbol_is_template_provider(
    const loom_symbol_t* symbol) {
  return symbol && (symbol->kind == LOOM_SYMBOL_TEMPLATE_DEF ||
                    symbol->kind == LOOM_SYMBOL_TEMPLATE_UKERNEL);
}

static iree_status_t loom_linker_mark_template_providers_live(
    loom_linker_source_t* source, iree_string_view_t family_name) {
  if (iree_string_view_is_empty(family_name)) {
    return iree_ok_status();
  }
  for (uint16_t symbol_id = 0; symbol_id < source->source_symbol_count;
       ++symbol_id) {
    const loom_symbol_t* symbol = &source->module->symbols.entries[symbol_id];
    if (!loom_linker_source_symbol_is_template_provider(symbol)) {
      continue;
    }
    loom_func_like_t provider =
        loom_func_like_cast(source->module, symbol->defining_op);
    if (!loom_func_like_isa(provider)) continue;

    const loom_symbol_ref_t family = loom_func_like_template_family(provider);
    if (!loom_symbol_ref_is_valid(family) || family.module_id != 0 ||
        family.symbol_id >= source->module->symbols.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "template provider has an invalid family symbol");
    }
    const loom_symbol_t* family_symbol =
        &source->module->symbols.entries[family.symbol_id];
    if (family_symbol->name_id >= source->module->strings.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "template family symbol has an invalid name");
    }
    if (!iree_string_view_equal(
            source->module->strings.entries[family_symbol->name_id],
            family_name)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_linker_mark_source_symbol_live(source, symbol_id));
  }
  return iree_ok_status();
}

typedef struct loom_linker_apply_dependency_walk_t {
  // Source module currently being linked through a root filter.
  loom_linker_source_t* source;

  // Module containing the template.apply operations being scanned.
  const loom_module_t* apply_module;
} loom_linker_apply_dependency_walk_t;

static iree_status_t loom_linker_visit_apply_dependency(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_template_apply_isa(op)) {
    return iree_ok_status();
  }

  loom_linker_apply_dependency_walk_t* walk =
      (loom_linker_apply_dependency_walk_t*)user_data;
  loom_linker_source_t* source = walk->source;
  const loom_symbol_ref_t family = loom_template_apply_family(op);
  if (!loom_symbol_ref_is_valid(family) || family.module_id != 0 ||
      family.symbol_id >= walk->apply_module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "template.apply has an invalid family symbol");
  }
  const loom_symbol_t* family_symbol =
      &walk->apply_module->symbols.entries[family.symbol_id];
  if (family_symbol->name_id >= walk->apply_module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "template family symbol has an invalid name");
  }
  return loom_linker_mark_template_providers_live(
      source, walk->apply_module->strings.entries[family_symbol->name_id]);
}

static iree_status_t loom_linker_mark_function_apply_dependencies_live(
    loom_linker_source_t* source, const loom_module_t* apply_module,
    loom_func_like_t function) {
  if (!loom_func_like_isa(function)) return iree_ok_status();
  loom_linker_apply_dependency_walk_t walk = {
      .source = source,
      .apply_module = apply_module,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  return loom_walk_function(apply_module, function, LOOM_WALK_PRE_ORDER,
                            (loom_walk_callback_t){
                                .fn = loom_linker_visit_apply_dependency,
                                .user_data = &walk,
                            },
                            source->arena, &walk_result);
}

static iree_status_t loom_linker_mark_apply_dependencies_live(
    loom_linker_source_t* source, const loom_symbol_t* symbol) {
  if (!symbol || !symbol->defining_op) return iree_ok_status();
  loom_func_like_t function =
      loom_func_like_cast(source->module, symbol->defining_op);
  return loom_linker_mark_function_apply_dependencies_live(
      source, source->module, function);
}

static iree_string_view_t loom_link_normalize_root_name(
    iree_string_view_t root_name) {
  if (iree_string_view_starts_with_char(root_name, '@')) {
    root_name = iree_string_view_remove_prefix(root_name, 1);
  }
  return root_name;
}

static iree_status_t loom_linker_mark_root_symbols_live(
    loom_linker_source_t* source, const loom_linker_add_options_t* options) {
  for (iree_host_size_t root_index = 0;
       root_index < options->root_symbols.count; ++root_index) {
    iree_string_view_t root_name =
        loom_link_normalize_root_name(options->root_symbols.values[root_index]);
    if (iree_string_view_is_empty(root_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "root symbol name must not be empty");
    }
    for (iree_host_size_t symbol_index = 0;
         symbol_index < source->source_symbol_count; ++symbol_index) {
      iree_string_view_t source_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_link_source_symbol_name(
          source->module, (uint16_t)symbol_index, &source_name));
      if (!iree_string_view_equal(source_name, root_name)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_linker_mark_source_symbol_live(source, (uint16_t)symbol_index));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_linker_mark_existing_target_anchors_live(
    loom_linker_source_t* source) {
  loom_linker_t* linker = source->linker;
  for (iree_host_size_t symbol_index = 0;
       symbol_index < source->source_symbol_count; ++symbol_index) {
    iree_string_view_t source_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_link_source_symbol_name(
        source->module, (uint16_t)symbol_index, &source_name));
    loom_string_id_t target_name_id =
        loom_module_lookup_string(linker->target_module, source_name);
    if (target_name_id == LOOM_STRING_ID_INVALID) {
      continue;
    }
    uint16_t target_symbol_id =
        loom_symbol_map_find(&linker->target_symbol_lookup, target_name_id);
    if (target_symbol_id == LOOM_SYMBOL_ID_INVALID) {
      continue;
    }

    const loom_symbol_t* source_symbol =
        &source->module->symbols.entries[symbol_index];
    const loom_symbol_t* target_symbol =
        &linker->target_module->symbols.entries[target_symbol_id];
    const bool target_needs_materialization =
        !target_symbol->defining_op ||
        loom_link_symbol_is_declaration(target_symbol);
    if (target_needs_materialization ||
        loom_link_symbol_has_global_identity(source->module, source_symbol)) {
      IREE_RETURN_IF_ERROR(
          loom_linker_mark_source_symbol_live(source, (uint16_t)symbol_index));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_linker_mark_existing_target_apply_dependencies_live(
    loom_linker_source_t* source) {
  loom_linker_t* linker = source->linker;
  const loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(linker->target_module, symbol) {
    if (!symbol->defining_op) continue;
    loom_func_like_t function =
        loom_func_like_cast(linker->target_module, symbol->defining_op);
    IREE_RETURN_IF_ERROR(loom_linker_mark_function_apply_dependencies_live(
        source, linker->target_module, function));
  }
  return iree_ok_status();
}

static iree_status_t loom_linker_resolve_live_symbols(
    loom_linker_source_t* source) {
  IREE_RETURN_IF_ERROR(loom_symbol_reference_table_build(
      source->module, source->arena, &source->reference_table));

  bool changed = true;
  while (changed) {
    changed = false;
    for (uint16_t symbol_id = 0; symbol_id < source->source_symbol_count;
         ++symbol_id) {
      if (!source->live_symbols[symbol_id] ||
          source->scanned_symbols[symbol_id]) {
        continue;
      }
      source->scanned_symbols[symbol_id] = 1;
      changed = true;
      if (symbol_id >= source->reference_table.symbol_count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "source symbol %u is out of range for "
                                "reference table with %" PRIhsz " symbols",
                                (unsigned)symbol_id,
                                source->reference_table.symbol_count);
      }

      loom_symbol_reference_occurrence_id_t edge_id =
          source->reference_table.symbols[symbol_id]
              .first_outgoing_occurrence_id;
      while (edge_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
        const loom_symbol_reference_occurrence_t* edge =
            &source->reference_table.occurrences[edge_id];
        if (!loom_symbol_reference_occurrence_is_dependency(edge)) {
          edge_id = edge->next_outgoing_occurrence_id;
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_linker_mark_source_symbol_live(
            source, edge->target_symbol_id));
        edge_id = edge->next_outgoing_occurrence_id;
      }

      const loom_symbol_t* symbol = &source->module->symbols.entries[symbol_id];
      IREE_RETURN_IF_ERROR(
          loom_linker_mark_apply_dependencies_live(source, symbol));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_linker_clone_module_body(
    loom_linker_source_t* source) {
  const loom_block_t* source_block =
      loom_region_const_entry_block(source->module->body);
  for (const loom_op_t* source_op = source_block->first_op; source_op;
       source_op = source_op->next_op) {
    loom_symbol_ref_t source_ref = loom_symbol_ref_null();
    const bool has_symbol_ref =
        loom_link_op_symbol_ref(source->module, source_op, &source_ref);
    if (!has_symbol_ref) {
      if (source->root_filtered) continue;
      loom_op_t* cloned_op = NULL;
      IREE_RETURN_IF_ERROR(loom_linker_clone_source_op(
          source, source_op, /*before_op=*/NULL, &cloned_op));
      continue;
    }

    if (source->root_filtered && !source->live_symbols[source_ref.symbol_id]) {
      continue;
    }
    loom_symbol_ref_t target_ref = loom_symbol_ref_null();
    IREE_RETURN_IF_ERROR(loom_linker_map_source_symbol(
        source, source_ref.symbol_id, &target_ref));
    IREE_RETURN_IF_ERROR(loom_linker_clone_or_merge_symbol_op(
        source, source_ref.symbol_id, target_ref,
        LOOM_LINKER_SYMBOL_OUTPUT_AUTHORED));
  }
  return iree_ok_status();
}

static iree_status_t loom_link_validate_add_options(
    const loom_linker_add_options_t* options) {
  if (!options || options->root_symbols.count == 0) {
    return iree_ok_status();
  }
  if (!options->root_symbols.values) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "root_symbols count is non-zero but values is NULL");
  }
  return iree_ok_status();
}

static iree_status_t loom_linker_validate_source_module(
    const loom_linker_t* linker, const loom_module_t* source_module) {
  if (!linker || !source_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "linker and source module must not be NULL");
  }
  if (linker->finished || !linker->target_module) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot add a module after linker finish");
  }
  if (source_module->context != linker->context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source module context does not match linker");
  }
  return iree_ok_status();
}

static iree_status_t loom_linker_validate_source_symbols(
    const loom_module_t* source_module,
    loom_linker_source_symbol_list_t source_symbols) {
  if (source_symbols.count != 0 && !source_symbols.ordinals) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source symbol count is non-zero but ordinals is NULL");
  }
  for (iree_host_size_t i = 0; i < source_symbols.count; ++i) {
    const iree_host_size_t ordinal = source_symbols.ordinals[i];
    if (ordinal >= source_module->symbols.count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source symbol ordinal %" PRIhsz
                              " is out of range for module with %" PRIhsz
                              " symbols",
                              ordinal, source_module->symbols.count);
    }
    if (i != 0 && ordinal <= source_symbols.ordinals[i - 1]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source symbol ordinals must be strictly increasing");
    }
  }
  return iree_ok_status();
}

iree_status_t loom_linker_allocate(loom_context_t* context,
                                   const loom_linker_options_t* options,
                                   iree_arena_block_pool_t* block_pool,
                                   iree_allocator_t allocator,
                                   loom_linker_t** out_linker) {
  *out_linker = NULL;
  if (!context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "context must not be NULL");
  }
  if (!block_pool) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "block pool must not be NULL");
  }

  loom_linker_t* linker = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*linker), (void**)&linker));
  memset(linker, 0, sizeof(*linker));
  linker->context = context;
  linker->block_pool = block_pool;
  linker->allocator = allocator;
  iree_arena_initialize(block_pool, &linker->scratch_arena);

  iree_string_view_t module_name =
      options && !iree_string_view_is_empty(options->module_name)
          ? options->module_name
          : IREE_SV("linked");
  iree_status_t status =
      loom_module_allocate(context, module_name, block_pool, /*hints=*/NULL,
                           allocator, &linker->target_module);
  if (iree_status_is_ok(status) && options &&
      options->planned_symbol_capacity != 0) {
    status = loom_linker_ensure_planned_symbol_capacity(
        linker, options->planned_symbol_capacity);
  }
  if (!iree_status_is_ok(status)) {
    loom_module_free(linker->target_module);
    iree_arena_deinitialize(&linker->scratch_arena);
    iree_allocator_free(allocator, linker);
    return status;
  }

  *out_linker = linker;
  return iree_ok_status();
}

void loom_linker_free(loom_linker_t* linker) {
  if (!linker) return;
  if (linker->target_module) {
    loom_module_free(linker->target_module);
  }
  iree_arena_deinitialize(&linker->scratch_arena);
  iree_allocator_free(linker->allocator, linker);
}

static void loom_linker_retain_function_root(loom_linker_t* linker,
                                             loom_symbol_t* symbol) {
  loom_op_t* op = symbol->defining_op;
  loom_func_like_t function = loom_func_like_cast(linker->target_module, op);
  if (!loom_func_like_isa(function)) return;
  loom_func_like_set_retained(linker->target_module, function, true);
}

iree_status_t loom_linker_add_module(loom_linker_t* linker,
                                     const loom_module_t* source_module,
                                     const loom_linker_add_options_t* options) {
  IREE_RETURN_IF_ERROR(
      loom_linker_validate_source_module(linker, source_module));
  IREE_RETURN_IF_ERROR(loom_link_validate_add_options(options));

  iree_arena_allocator_t source_arena;
  iree_arena_initialize(linker->block_pool, &source_arena);

  loom_linker_source_t source = {
      .linker = linker,
      .module = source_module,
      .arena = &source_arena,
      .source_symbol_count = source_module->symbols.count,
      .root_filtered = options && options->root_symbols.count != 0,
  };
  source.symbol_remap =
      loom_ir_remap_symbol_callback_make(loom_linker_remap_symbol, &source);

  iree_status_t status = iree_ok_status();
  if (source.source_symbol_count > 0) {
    status = iree_arena_allocate_array(
        &source_arena, source.source_symbol_count,
        sizeof(*source.target_symbols), (void**)&source.target_symbols);
  }
  if (iree_status_is_ok(status) && source.source_symbol_count > 0) {
    for (iree_host_size_t i = 0; i < source.source_symbol_count; ++i) {
      source.target_symbols[i] = loom_symbol_ref_null();
    }
  }
  if (iree_status_is_ok(status) && source.root_filtered) {
    status = iree_arena_allocate_array(
        &source_arena, source.source_symbol_count, sizeof(*source.live_symbols),
        (void**)&source.live_symbols);
  }
  if (iree_status_is_ok(status) && source.root_filtered) {
    status = iree_arena_allocate_array(
        &source_arena, source.source_symbol_count,
        sizeof(*source.scanned_symbols), (void**)&source.scanned_symbols);
  }
  if (iree_status_is_ok(status) && source.root_filtered) {
    memset(source.live_symbols, 0,
           source.source_symbol_count * sizeof(*source.live_symbols));
    memset(source.scanned_symbols, 0,
           source.source_symbol_count * sizeof(*source.scanned_symbols));
    status = loom_linker_mark_root_symbols_live(&source, options);
  }
  if (iree_status_is_ok(status) && source.root_filtered) {
    status = loom_linker_mark_existing_target_anchors_live(&source);
  }
  if (iree_status_is_ok(status) && source.root_filtered) {
    status = loom_linker_mark_existing_target_apply_dependencies_live(&source);
  }
  if (iree_status_is_ok(status) && source.root_filtered) {
    status = loom_linker_resolve_live_symbols(&source);
  }
  if (iree_status_is_ok(status)) {
    status = loom_linker_clone_module_body(&source);
  }
  iree_arena_deinitialize(&source_arena);
  return status;
}

// Clones exact definitions in authored source operation order. Exact
// selections are sorted by symbol ordinal, which is also source order for
// ordinary modules. Authored operation moves can make those orders diverge;
// only that uncommon case needs a sorted projection.
static iree_status_t loom_linker_clone_exact_symbol_ops(
    loom_linker_source_t* source) {
  bool source_ordered = true;
  bool has_definition = false;
  uint64_t previous_block_ordinal = 0;
  for (iree_host_size_t i = 0; i < source->exact.count; ++i) {
    const uint16_t source_symbol_id = (uint16_t)source->exact.ordinals[i];
    const loom_symbol_t* source_symbol =
        &source->module->symbols.entries[source_symbol_id];
    if (source_symbol->defining_op == NULL) {
      continue;
    }
    const uint64_t block_ordinal = source_symbol->defining_op->block_ordinal;
    if (has_definition && block_ordinal < previous_block_ordinal) {
      source_ordered = false;
      break;
    }
    has_definition = true;
    previous_block_ordinal = block_ordinal;
  }
  if (source_ordered) {
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0;
         i < source->exact.count && iree_status_is_ok(status); ++i) {
      const uint16_t source_symbol_id = (uint16_t)source->exact.ordinals[i];
      const loom_symbol_t* source_symbol =
          &source->module->symbols.entries[source_symbol_id];
      if (source_symbol->defining_op == NULL) {
        continue;
      }
      status = loom_linker_clone_or_merge_symbol_op(
          source, source_symbol_id, source->target_symbols[i],
          source->exact.outputs ? source->exact.outputs[i]
                                : LOOM_LINKER_SYMBOL_OUTPUT_AUTHORED);
    }
    return status;
  }

  loom_linker_exact_symbol_op_t* selected_ops = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(source->arena, source->exact.count,
                                sizeof(*selected_ops), (void**)&selected_ops));

  iree_host_size_t selected_op_count = 0;
  for (iree_host_size_t i = 0; i < source->exact.count; ++i) {
    const uint16_t source_symbol_id = (uint16_t)source->exact.ordinals[i];
    const loom_symbol_t* source_symbol =
        &source->module->symbols.entries[source_symbol_id];
    if (source_symbol->defining_op == NULL) {
      continue;
    }
    selected_ops[selected_op_count++] = (loom_linker_exact_symbol_op_t){
        .source_op = source_symbol->defining_op,
        .source_symbol_id = source_symbol_id,
        .selection_ordinal = i,
    };
  }
  loom_linker_sort_exact_symbol_ops(selected_ops, selected_op_count);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < selected_op_count && iree_status_is_ok(status); ++i) {
    const loom_linker_exact_symbol_op_t* selected_op = &selected_ops[i];
    status = loom_linker_clone_or_merge_symbol_op(
        source, selected_op->source_symbol_id,
        source->target_symbols[selected_op->selection_ordinal],
        source->exact.outputs
            ? source->exact.outputs[selected_op->selection_ordinal]
            : LOOM_LINKER_SYMBOL_OUTPUT_AUTHORED);
  }
  return status;
}

static iree_status_t loom_linker_add_exact_selection(
    loom_linker_t* linker, const loom_module_t* source_module,
    loom_linker_exact_selection_t selection,
    loom_linker_source_symbol_output_list_t source_outputs,
    loom_linker_target_symbol_list_t out_target_symbols) {
  if (source_outputs.count != 0 && (source_outputs.count != selection.count ||
                                    source_outputs.values == NULL)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source output has %zu entries but selection has %zu",
        source_outputs.count, selection.count);
  }
  if (out_target_symbols.count != 0 &&
      (out_target_symbols.count != selection.count ||
       out_target_symbols.values == NULL)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target symbol output has %zu entries but selection has %zu",
        out_target_symbols.count, selection.count);
  }
  if (source_outputs.count != 0) {
    iree_host_size_t required_capacity = 0;
    if (!iree_host_size_checked_add(linker->target_module->symbols.count,
                                    selection.count, &required_capacity)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "planned symbol capacity overflow");
    }
    IREE_RETURN_IF_ERROR(
        loom_linker_ensure_planned_symbol_capacity(linker, required_capacity));
  }
  iree_arena_allocator_t source_arena;
  iree_arena_initialize(linker->block_pool, &source_arena);
  loom_linker_source_t source = {
      .linker = linker,
      .module = source_module,
      .arena = &source_arena,
      .source_symbol_count = source_module->symbols.count,
      .exact = selection,
  };
  source.exact.outputs = source_outputs.values;
  loom_ir_remap_symbol_fn_t remap_symbol = loom_linker_remap_exact_symbol;
  if (selection.dense) {
    remap_symbol = loom_linker_remap_dense_symbol;
  } else if (selection.bindings.count != 0) {
    remap_symbol = loom_linker_remap_bound_exact_symbol;
  }
  source.symbol_remap =
      loom_ir_remap_symbol_callback_make(remap_symbol, &source);

  iree_status_t status = iree_ok_status();
  if (out_target_symbols.count > 0) {
    source.target_symbols = out_target_symbols.values;
  } else if (source.exact.count > 0) {
    status = iree_arena_allocate_array(&source_arena, source.exact.count,
                                       sizeof(*source.target_symbols),
                                       (void**)&source.target_symbols);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < source.exact.count; ++i) {
      source.target_symbols[i] = loom_symbol_ref_null();
    }
  }
  for (iree_host_size_t i = 0;
       i < source.exact.count && iree_status_is_ok(status); ++i) {
    const uint16_t source_symbol_id =
        (uint16_t)(selection.dense ? i : selection.ordinals[i]);
    loom_symbol_ref_t target_ref = loom_symbol_ref_null();
    status = loom_linker_resolve_source_symbol(
        &source, source_symbol_id, &source.target_symbols[i], &target_ref);
  }

  if (iree_status_is_ok(status) && selection.dense) {
    const loom_block_t* source_block =
        loom_region_const_entry_block(source_module->body);
    for (const loom_op_t* source_op = source_block->first_op;
         source_op && iree_status_is_ok(status);
         source_op = source_op->next_op) {
      loom_symbol_ref_t source_ref = loom_symbol_ref_null();
      if (loom_link_op_symbol_ref(source_module, source_op, &source_ref)) {
        status = loom_linker_clone_or_merge_symbol_op(
            &source, source_ref.symbol_id,
            source.target_symbols[source_ref.symbol_id],
            source.exact.outputs ? source.exact.outputs[source_ref.symbol_id]
                                 : LOOM_LINKER_SYMBOL_OUTPUT_AUTHORED);
      } else {
        loom_op_t* cloned_op = NULL;
        status = loom_linker_clone_source_op(&source, source_op,
                                             /*before_op=*/NULL, &cloned_op);
      }
    }
  } else if (iree_status_is_ok(status)) {
    status = loom_linker_clone_exact_symbol_ops(&source);
  }

  iree_arena_deinitialize(&source_arena);
  return status;
}

iree_status_t loom_linker_add_module_symbols(
    loom_linker_t* linker, const loom_module_t* source_module,
    loom_linker_source_symbol_list_t source_symbols,
    loom_linker_source_symbol_binding_list_t source_bindings,
    loom_linker_source_symbol_output_list_t source_outputs,
    loom_linker_target_symbol_list_t out_target_symbols) {
  IREE_RETURN_IF_ERROR(
      loom_linker_validate_source_module(linker, source_module));
  IREE_RETURN_IF_ERROR(
      loom_linker_validate_source_symbols(source_module, source_symbols));
  return loom_linker_add_exact_selection(
      linker, source_module,
      (loom_linker_exact_selection_t){
          .ordinals = source_symbols.ordinals,
          .bindings = source_bindings,
          .count = source_symbols.count,
      },
      source_outputs, out_target_symbols);
}

iree_status_t loom_linker_add_exact_module(
    loom_linker_t* linker, const loom_module_t* source_module,
    loom_linker_source_symbol_output_list_t source_outputs,
    loom_linker_target_symbol_list_t out_target_symbols) {
  IREE_RETURN_IF_ERROR(
      loom_linker_validate_source_module(linker, source_module));
  return loom_linker_add_exact_selection(
      linker, source_module,
      (loom_linker_exact_selection_t){
          .count = source_module->symbols.count,
          .dense = true,
      },
      source_outputs, out_target_symbols);
}

iree_status_t loom_linker_finalize_roots(loom_linker_t* linker,
                                         iree_string_view_list_t root_symbols) {
  if (!linker) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "linker must not be NULL");
  }
  if (linker->finished || !linker->target_module) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot finalize roots after linker finish");
  }
  if (root_symbols.count != 0 && !root_symbols.values) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "root_symbols count is non-zero but values is NULL");
  }

  for (iree_host_size_t i = 0; i < root_symbols.count; ++i) {
    iree_string_view_t root_name =
        loom_link_normalize_root_name(root_symbols.values[i]);
    if (iree_string_view_is_empty(root_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "root symbol name must not be empty");
    }
    loom_string_id_t target_name_id =
        loom_module_lookup_string(linker->target_module, root_name);
    if (target_name_id == LOOM_STRING_ID_INVALID) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "root symbol '@%.*s' was not found",
                              (int)root_name.size, root_name.data);
    }
    uint16_t target_symbol_id =
        loom_symbol_map_find(&linker->target_symbol_lookup, target_name_id);
    if (target_symbol_id == LOOM_SYMBOL_ID_INVALID) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "root symbol '@%.*s' was not found",
                              (int)root_name.size, root_name.data);
    }
    loom_symbol_t* target_symbol =
        &linker->target_module->symbols.entries[target_symbol_id];
    if (!target_symbol->defining_op) {
      return iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "root symbol '@%.*s' has no materialized definition or declaration",
          (int)root_name.size, root_name.data);
    }
    loom_linker_retain_function_root(linker, target_symbol);
  }
  return iree_ok_status();
}

iree_status_t loom_linker_finish(loom_linker_t* linker,
                                 loom_module_t** out_module) {
  *out_module = NULL;
  if (!linker) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "linker must not be NULL");
  }
  if (linker->finished || !linker->target_module) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "linker has already been finished");
  }

  IREE_RETURN_IF_ERROR(loom_module_compute_uses(linker->target_module));
  *out_module = linker->target_module;
  linker->target_module = NULL;
  linker->finished = true;
  return iree_ok_status();
}

static iree_status_t loom_link_validate_inputs(
    const loom_module_t* const* source_modules,
    iree_host_size_t source_module_count) {
  if (!source_modules || source_module_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at least one source module is required");
  }
  if (!source_modules[0]) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source module 0 is NULL");
  }
  loom_context_t* context = source_modules[0]->context;
  for (iree_host_size_t i = 0; i < source_module_count; ++i) {
    if (!source_modules[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source module %" PRIhsz " is NULL", i);
    }
    if (source_modules[i]->context != context) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source modules must share one context");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_link_validate_options(
    const loom_link_options_t* options) {
  if (!options || options->root_symbols.count == 0) {
    return iree_ok_status();
  }
  if (!options->root_symbols.values) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "root_symbols count is non-zero but values is NULL");
  }
  return iree_ok_status();
}

static iree_status_t loom_link_module_contains_root(
    const loom_module_t* module, const loom_link_options_t* options,
    bool* out_contains_root) {
  *out_contains_root = false;
  for (iree_host_size_t root_index = 0;
       root_index < options->root_symbols.count; ++root_index) {
    iree_string_view_t root_name =
        loom_link_normalize_root_name(options->root_symbols.values[root_index]);
    if (iree_string_view_is_empty(root_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "root symbol name must not be empty");
    }
    for (iree_host_size_t symbol_index = 0;
         symbol_index < module->symbols.count; ++symbol_index) {
      iree_string_view_t source_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_link_source_symbol_name(
          module, (uint16_t)symbol_index, &source_name));
      if (iree_string_view_equal(source_name, root_name)) {
        *out_contains_root = true;
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_link_materialized_modules(
    const loom_module_t* const* source_modules,
    iree_host_size_t source_module_count, const loom_link_options_t* options,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** out_module) {
  *out_module = NULL;
  IREE_RETURN_IF_ERROR(
      loom_link_validate_inputs(source_modules, source_module_count));
  IREE_RETURN_IF_ERROR(loom_link_validate_options(options));

  loom_linker_t* linker = NULL;
  iree_status_t status = loom_linker_allocate(
      source_modules[0]->context,
      &(loom_linker_options_t){
          .module_name =
              options ? options->module_name : iree_string_view_empty(),
      },
      block_pool, allocator, &linker);

  const iree_host_size_t root_symbol_count =
      options ? options->root_symbols.count : 0;
  uint8_t* root_module_flags = NULL;
  if (iree_status_is_ok(status) && root_symbol_count > 0) {
    status = iree_allocator_malloc(
        allocator, source_module_count * sizeof(*root_module_flags),
        (void**)&root_module_flags);
  }
  if (iree_status_is_ok(status) && root_module_flags) {
    memset(root_module_flags, 0,
           source_module_count * sizeof(*root_module_flags));
    for (iree_host_size_t i = 0;
         i < source_module_count && iree_status_is_ok(status); ++i) {
      bool contains_root = false;
      status = loom_link_module_contains_root(source_modules[i], options,
                                              &contains_root);
      root_module_flags[i] = contains_root ? 1 : 0;
    }
  }

  loom_linker_add_options_t add_options = {
      .root_symbols =
          {
              .count = root_symbol_count,
              .values = root_symbol_count ? options->root_symbols.values : NULL,
          },
  };
  if (iree_status_is_ok(status) && root_module_flags) {
    for (iree_host_size_t i = 0;
         i < source_module_count && iree_status_is_ok(status); ++i) {
      if (!root_module_flags[i]) continue;
      status = loom_linker_add_module(linker, source_modules[i], &add_options);
    }
  }
  for (iree_host_size_t i = 0;
       i < source_module_count && iree_status_is_ok(status); ++i) {
    if (root_module_flags && root_module_flags[i]) continue;
    status = loom_linker_add_module(linker, source_modules[i], &add_options);
  }
  if (iree_status_is_ok(status) && root_symbol_count > 0) {
    status = loom_linker_finalize_roots(linker, options->root_symbols);
  }
  if (iree_status_is_ok(status)) {
    status = loom_linker_finish(linker, out_module);
  }
  iree_allocator_free(allocator, root_module_flags);
  loom_linker_free(linker);
  return status;
}
