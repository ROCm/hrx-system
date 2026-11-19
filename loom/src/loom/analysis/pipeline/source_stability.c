// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/source_stability.h"

#include "iree/base/internal/math.h"
#include "loom/analysis/view_regions.h"
#include "loom/util/adaptive_sort.h"
#include "loom/util/fact_extensions.h"

typedef struct loom_pipeline_storage_reference_t {
  // Canonical backing identity in the pipeline construction scope.
  loom_value_id_t root_value_id;
  // Comparable noalias partition, or NONE when roots may alias.
  loom_value_fact_alias_scope_id_t alias_scope_id;
  // Target-independent storage space of the backing allocation.
  loom_value_fact_memory_space_t memory_space;
} loom_pipeline_storage_reference_t;

static loom_pipeline_storage_reference_t
loom_pipeline_storage_reference_from_facts(const loom_value_fact_table_t* facts,
                                           loom_value_id_t value_id) {
  loom_pipeline_storage_reference_t reference = {
      .root_value_id = value_id,
      .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
  };
  const loom_value_facts_t value_facts =
      loom_value_fact_table_lookup(facts, value_id);
  loom_value_fact_buffer_reference_t buffer;
  if (loom_value_facts_query_buffer_reference(&facts->context, value_facts,
                                              &buffer)) {
    reference.root_value_id =
        loom_value_fact_buffer_reference_resolve_root_value(buffer, value_id);
    reference.alias_scope_id = buffer.alias_scope_id;
    reference.memory_space = buffer.memory_space;
    return reference;
  }
  loom_value_fact_view_reference_t view;
  if (loom_value_facts_query_view_reference(&facts->context, value_facts,
                                            &view)) {
    reference.root_value_id =
        loom_value_fact_view_reference_resolve_root_value(view, value_id);
    reference.alias_scope_id = view.alias_scope_id;
    reference.memory_space = view.memory_space;
  }
  return reference;
}

static bool loom_pipeline_storage_reference_may_modify(
    loom_storage_access_effects_t effects,
    const loom_pipeline_storage_reference_t* reference) {
  // Constant memory remains immutable across opaque escapes. Keep explicit
  // writes conservative here so their eventual legality does not affect this
  // proof.
  if (iree_any_bit_set(effects, LOOM_STORAGE_ACCESS_WRITE)) {
    return true;
  }
  return iree_any_bit_set(effects, LOOM_STORAGE_ACCESS_ESCAPE) &&
         reference->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT;
}

// Roots always conflict with themselves. A noalias partition is compared only
// in memory spaces that its modification can reach. The first 2^32 keys are
// root value IDs; each source space has its own partition key range.
static uint64_t loom_pipeline_partition_key(
    loom_value_fact_alias_scope_id_t identity,
    loom_value_fact_memory_space_t source_space) {
  return ((uint64_t)(1 + source_space) << 32) | identity;
}

typedef struct loom_pipeline_modifier_t {
  // Root or space-qualified partition that this worker may modify.
  uint64_t key;
  // Forward order of the modifying worker's construction.
  iree_host_size_t preorder;
  // Largest concurrent preorder from this key's beginning through this row.
  iree_host_size_t prefix_maximum;
  // Smallest concurrent preorder from this row through this key's end.
  iree_host_size_t suffix_minimum;
} loom_pipeline_modifier_t;

static bool loom_pipeline_modifier_less(const loom_pipeline_modifier_t* left,
                                        const loom_pipeline_modifier_t* right) {
  return left->key != right->key ? left->key < right->key
                                 : left->preorder < right->preorder;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_pipeline_sort_modifiers,
                          loom_pipeline_modifier_t, loom_pipeline_modifier_less)

typedef struct loom_pipeline_modifiers_t {
  // Sorted by (storage key, construction preorder) after indexing.
  loom_pipeline_modifier_t* rows;
  // Number of occupied rows.
  iree_host_size_t count;
} loom_pipeline_modifiers_t;

static loom_pipeline_lifetime_ancestry_t loom_pipeline_worker_ancestry(
    const loom_pipeline_invocations_t* invocations, iree_host_size_t worker) {
  if (!invocations || !invocations->lifetimes.has_serial) {
    return (loom_pipeline_lifetime_ancestry_t){0};
  }
  const iree_host_size_t owner =
      invocations->workers[worker].invocation->ordinal;
  const loom_pipeline_lifetimes_t* lifetimes = &invocations->lifetimes;
  return lifetimes->ancestry[lifetimes->owners[owner]];
}

static void loom_pipeline_add_modifier(
    loom_pipeline_modifiers_t* modifiers, uint64_t key,
    loom_pipeline_lifetime_ancestry_t ancestry) {
  modifiers->rows[modifiers->count++] = (loom_pipeline_modifier_t){
      .key = key,
      .preorder = ancestry.preorder,
      .prefix_maximum = ancestry.concurrent_preorder,
      .suffix_minimum = ancestry.concurrent_preorder,
  };
}

static void loom_pipeline_add_partition_modifiers(
    loom_pipeline_modifiers_t* modifiers,
    loom_value_fact_alias_scope_id_t identity, loom_memory_space_mask_t spaces,
    loom_pipeline_lifetime_ancestry_t ancestry) {
  while (spaces) {
    const uint32_t space = iree_math_count_trailing_zeros_u32(spaces);
    loom_pipeline_add_modifier(
        modifiers, loom_pipeline_partition_key(identity, space), ancestry);
    spaces &= spaces - 1;
  }
}

static void loom_pipeline_index_modifiers(
    loom_pipeline_modifiers_t* modifiers) {
  loom_pipeline_modifier_t* rows = modifiers->rows;
  loom_pipeline_sort_modifiers(rows, modifiers->count);
  for (iree_host_size_t i = 1; i < modifiers->count; ++i) {
    if (rows[i].key == rows[i - 1].key) {
      rows[i].prefix_maximum =
          iree_max(rows[i].prefix_maximum, rows[i - 1].prefix_maximum);
    }
  }
  for (iree_host_size_t end = modifiers->count; end > 1; --end) {
    if (rows[end - 2].key == rows[end - 1].key) {
      rows[end - 2].suffix_minimum =
          iree_min(rows[end - 2].suffix_minimum, rows[end - 1].suffix_minimum);
    }
  }
}

// Serial siblings are in the same order in both traversals. Concurrent
// siblings are in opposite orders. A writer on either side of the reader
// can overlap it exactly when its second ordinal crosses the reader's, or
// when they share an owner. Two extrema suffice for an entire storage key.
static bool loom_pipeline_has_modifier(
    const loom_pipeline_modifiers_t* modifiers, uint64_t key,
    loom_pipeline_lifetime_ancestry_t reader) {
  const loom_pipeline_modifier_t* rows = modifiers->rows;
  iree_host_size_t begin = 0, end = modifiers->count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (rows[middle].key < key ||
        (rows[middle].key == key && rows[middle].preorder < reader.preorder)) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return (begin && rows[begin - 1].key == key &&
          rows[begin - 1].prefix_maximum >= reader.concurrent_preorder) ||
         (begin < modifiers->count && rows[begin].key == key &&
          rows[begin].suffix_minimum <= reader.concurrent_preorder);
}

static loom_memory_space_mask_t loom_pipeline_transport_source_spaces(
    const loom_pipeline_worker_t* workers, iree_host_size_t worker_count) {
  loom_memory_space_mask_t spaces = 0;
  for (iree_host_size_t i = 0; i < worker_count; ++i) {
    const loom_pipeline_worker_t* worker = &workers[i];
    if (!worker->graph->has_cycles) {
      continue;
    }
    for (iree_host_size_t j = 0; j < worker->transport.stream_count; ++j) {
      const loom_kernel_async_stream_t* stream = worker->transport.streams[j];
      for (iree_host_size_t k = 0; k < stream->transfer_count; ++k) {
        spaces |= 1u << stream->transfers[k].request.source.memory_space;
      }
    }
  }
  return spaces;
}

static bool loom_pipeline_worker_sources_stable(
    const loom_pipeline_worker_t* worker,
    const loom_pipeline_modifiers_t* modifiers,
    loom_pipeline_lifetime_ancestry_t reader) {
  for (iree_host_size_t j = 0; j < worker->transport.stream_count; ++j) {
    const loom_kernel_async_stream_t* stream = worker->transport.streams[j];
    for (iree_host_size_t k = 0; k < stream->transfer_count; ++k) {
      const loom_movement_endpoint_t* endpoint =
          &stream->transfers[k].request.source;
      const loom_view_region_t* source = NULL;
      if (!loom_view_region_table_try_lookup(
              &worker->asynchronous.movement.view_regions, endpoint->value_id,
              &source) ||
          source->alias_scope_id == LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE ||
          loom_local_value_domain_ordinal(&worker->value_domain,
                                          source->root_value_id) <
              worker->value_domain.definition_count) {
        return false;
      }
      if (loom_pipeline_has_modifier(modifiers, source->root_value_id,
                                     reader) ||
          loom_pipeline_has_modifier(
              modifiers,
              loom_pipeline_partition_key(source->alias_scope_id,
                                          source->memory_space),
              reader) ||
          loom_pipeline_has_modifier(
              modifiers,
              loom_pipeline_partition_key(LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
                                          source->memory_space),
              reader)) {
        return false;
      }
    }
  }
  return true;
}

iree_status_t loom_pipeline_prove_source_stability(
    loom_storage_access_scope_t* storage_access,
    const loom_pipeline_invocations_t* invocations,
    loom_pipeline_worker_t* workers, iree_host_size_t worker_count) {
  const loom_memory_space_mask_t source_spaces =
      loom_pipeline_transport_source_spaces(workers, worker_count);
  if (!source_spaces) {
    return iree_ok_status();
  }
  iree_arena_allocator_t* arena = storage_access->arena;
  const loom_storage_access_function_t** accesses = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, worker_count, sizeof(*accesses), (void**)&accesses));
  const iree_host_size_t space_count = iree_math_count_ones_u32(source_spaces);
  iree_host_size_t capacity = 0;
  for (iree_host_size_t i = 0; i < worker_count; ++i) {
    loom_pipeline_worker_t* worker = &workers[i];
    loom_local_value_domain_restore(&worker->value_domain);
    iree_status_t status = loom_storage_access_require_function(
        storage_access, worker->function, &worker->value_domain, &accesses[i]);
    loom_local_value_domain_release(&worker->value_domain);
    IREE_RETURN_IF_ERROR(status);
    if (accesses[i]->unbounded_write_spaces) {
      capacity += space_count;
    }
    for (const loom_storage_access_use_t* use = accesses[i]->uses; use;
         use = use->next_in_function) {
      if (iree_any_bit_set(
              loom_storage_access_use_effects(use),
              LOOM_STORAGE_ACCESS_WRITE | LOOM_STORAGE_ACCESS_ESCAPE)) {
        capacity += 1 + space_count;
      }
    }
  }
  loom_pipeline_modifiers_t modifiers = {0};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity, sizeof(*modifiers.rows), (void**)&modifiers.rows));
  loom_memory_space_mask_t conflicts[LOOM_VALUE_FACT_MEMORY_SPACE_COUNT_] = {0};
  for (uint32_t i = 0; i < LOOM_VALUE_FACT_MEMORY_SPACE_COUNT_; ++i) {
    for (uint32_t j = 0; j < LOOM_VALUE_FACT_MEMORY_SPACE_COUNT_; ++j) {
      if ((source_spaces & (1u << j)) &&
          !loom_view_memory_spaces_are_disjoint(i, j)) {
        conflicts[i] |= 1u << j;
      }
    }
  }
  for (iree_host_size_t i = 0; i < worker_count; ++i) {
    const loom_pipeline_lifetime_ancestry_t writer =
        loom_pipeline_worker_ancestry(invocations, i);
    const loom_storage_access_function_t* access = accesses[i];
    loom_memory_space_mask_t hidden_conflicts = 0;
    for (loom_memory_space_mask_t spaces = access->unbounded_write_spaces;
         spaces; spaces &= spaces - 1) {
      hidden_conflicts |= conflicts[iree_math_count_trailing_zeros_u32(spaces)];
    }
    // An opaque call can reach a hidden mutable reference in these spaces.
    // Explicit writes remain conservative even when the root is constant.
    loom_pipeline_add_partition_modifiers(
        &modifiers, LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
        hidden_conflicts & ~(1u << LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT),
        writer);
    for (const loom_storage_access_use_t* use = access->uses; use;
         use = use->next_in_function) {
      const loom_storage_access_effects_t effects =
          loom_storage_access_use_effects(use);
      if (!iree_any_bit_set(effects, LOOM_STORAGE_ACCESS_WRITE |
                                         LOOM_STORAGE_ACCESS_ESCAPE)) {
        continue;
      }
      const loom_pipeline_storage_reference_t reference =
          loom_pipeline_storage_reference_from_facts(&workers[i].facts,
                                                     use->value_id);
      if (loom_pipeline_storage_reference_may_modify(effects, &reference)) {
        loom_pipeline_add_modifier(&modifiers, reference.root_value_id, writer);
        loom_pipeline_add_partition_modifiers(
            &modifiers, reference.alias_scope_id,
            conflicts[reference.memory_space], writer);
      }
    }
  }
  loom_pipeline_index_modifiers(&modifiers);
  for (iree_host_size_t i = 0; i < worker_count; ++i) {
    loom_pipeline_worker_t* worker = &workers[i];
    if (!worker->transport.stream_count || !worker->graph->has_cycles) {
      continue;
    }
    loom_local_value_domain_restore(&worker->value_domain);
    worker->transport.sources_stable = loom_pipeline_worker_sources_stable(
        worker, &modifiers, loom_pipeline_worker_ancestry(invocations, i));
    loom_local_value_domain_release(&worker->value_domain);
  }
  return iree_ok_status();
}
