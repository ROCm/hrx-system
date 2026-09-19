// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_table.h"

#include <stdint.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/util/fact_cfg.h"

//===----------------------------------------------------------------------===//
// Capacity management
//===----------------------------------------------------------------------===//

struct loom_value_fact_region_entry_t {
  // Region whose execution context and optional CFG structure are retained.
  const loom_region_t* region;
  // Distribution inherited from enclosing CFG cycles, excluding this region.
  uint32_t temporal_distribution;
  // CFG and forwarding components retained for the populated fact scope.
  const loom_value_fact_cfg_region_t* structure;
  // Next entry in the region-address hash collision chain.
  loom_value_fact_region_entry_t* next_bucket;
  // Next entry in the complete cache entry list.
  loom_value_fact_region_entry_t* next_entry;
};

static iree_status_t loom_value_fact_table_ensure_capacity(
    loom_value_fact_table_t* table, iree_host_size_t capacity) {
  if (capacity <= table->capacity) {
    return iree_ok_status();
  }
  const iree_host_size_t old_capacity = table->capacity;
  iree_host_size_t new_capacity = old_capacity;
  loom_value_facts_t* entries = table->entries;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      table->arena, old_capacity, capacity, sizeof(loom_value_facts_t),
      &new_capacity, (void**)&entries));
  memset(entries + old_capacity, 0,
         (new_capacity - old_capacity) * sizeof(loom_value_facts_t));
  iree_host_size_t old_word_count = (old_capacity + 63) / 64;
  iree_host_size_t word_count = (new_capacity + 63) / 64;
  uint64_t* touched_bits = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      table->arena, word_count, sizeof(*touched_bits), (void**)&touched_bits));
  if (old_word_count) {
    memcpy(touched_bits, table->touched_bits,
           old_word_count * sizeof(*touched_bits));
  }
  memset(touched_bits + old_word_count, 0,
         (word_count - old_word_count) * sizeof(*touched_bits));
  table->entries = entries;
  table->capacity = new_capacity;
  table->touched_bits = touched_bits;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_ensure_uniform_origin_capacity(
    loom_value_fact_table_t* table, iree_host_size_t capacity) {
  if (capacity <= table->uniform_element_origins.capacity) {
    return iree_ok_status();
  }
  const iree_host_size_t old_capacity = table->uniform_element_origins.capacity;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      table->arena, old_capacity, capacity, sizeof(loom_value_id_t),
      &table->uniform_element_origins.capacity,
      (void**)&table->uniform_element_origins.entries));
  for (iree_host_size_t i = old_capacity;
       i < table->uniform_element_origins.capacity; ++i) {
    table->uniform_element_origins.entries[i] = LOOM_VALUE_ID_INVALID;
  }
  return iree_ok_status();
}

static loom_value_fact_static_lane_origin_t
loom_value_fact_static_lane_origin_invalid(void) {
  return (loom_value_fact_static_lane_origin_t){
      .source_value_id = LOOM_VALUE_ID_INVALID,
  };
}

static loom_value_fact_uniform_scale_origin_t
loom_value_fact_uniform_scale_origin_invalid(void) {
  return (loom_value_fact_uniform_scale_origin_t){
      .source_value_id = LOOM_VALUE_ID_INVALID,
      .scale_value_id = LOOM_VALUE_ID_INVALID,
  };
}

static iree_status_t loom_value_fact_table_ensure_static_lane_origin_capacity(
    loom_value_fact_table_t* table, iree_host_size_t capacity) {
  if (capacity <= table->static_lane_origins.capacity) {
    return iree_ok_status();
  }
  const iree_host_size_t old_capacity = table->static_lane_origins.capacity;
  IREE_RETURN_IF_ERROR(
      iree_arena_grow_array(table->arena, old_capacity, capacity,
                            sizeof(loom_value_fact_static_lane_origin_t),
                            &table->static_lane_origins.capacity,
                            (void**)&table->static_lane_origins.entries));
  for (iree_host_size_t i = old_capacity;
       i < table->static_lane_origins.capacity; ++i) {
    table->static_lane_origins.entries[i] =
        loom_value_fact_static_lane_origin_invalid();
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_ensure_uniform_scale_origin_capacity(
    loom_value_fact_table_t* table, iree_host_size_t capacity) {
  if (capacity <= table->uniform_scale_origins.capacity) {
    return iree_ok_status();
  }
  const iree_host_size_t old_capacity = table->uniform_scale_origins.capacity;
  IREE_RETURN_IF_ERROR(
      iree_arena_grow_array(table->arena, old_capacity, capacity,
                            sizeof(loom_value_fact_uniform_scale_origin_t),
                            &table->uniform_scale_origins.capacity,
                            (void**)&table->uniform_scale_origins.entries));
  for (iree_host_size_t i = old_capacity;
       i < table->uniform_scale_origins.capacity; ++i) {
    table->uniform_scale_origins.entries[i] =
        loom_value_fact_uniform_scale_origin_invalid();
  }
  return iree_ok_status();
}

static iree_status_t
loom_value_fact_table_ensure_contextual_query_origin_capacity(
    loom_value_fact_table_t* table, iree_host_size_t capacity) {
  if (capacity <= table->contextual_query_origins.capacity) {
    return iree_ok_status();
  }
  const iree_host_size_t old_capacity =
      table->contextual_query_origins.capacity;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      table->arena, old_capacity, capacity, sizeof(uint32_t),
      &table->contextual_query_origins.capacity,
      (void**)&table->contextual_query_origins.entries));
  memset(table->contextual_query_origins.entries + old_capacity, 0,
         (table->contextual_query_origins.capacity - old_capacity) *
             sizeof(uint32_t));
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_allocate_initial_capacity(
    loom_value_fact_table_t* table, iree_host_size_t capacity) {
  return loom_value_fact_table_ensure_capacity(table, capacity);
}

static iree_status_t loom_value_fact_table_append_touched_value(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  if (table->touched_count >= table->touched_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        table->arena, table->touched_count, table->touched_count + 1,
        sizeof(*table->touched_values), &table->touched_capacity,
        (void**)&table->touched_values));
  }
  table->touched_values[table->touched_count++] = value_id;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_append_touched_uniform_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  if (table->uniform_element_origins.touched_count >=
      table->uniform_element_origins.touched_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        table->arena, table->uniform_element_origins.touched_count,
        table->uniform_element_origins.touched_count + 1,
        sizeof(*table->uniform_element_origins.touched_values),
        &table->uniform_element_origins.touched_capacity,
        (void**)&table->uniform_element_origins.touched_values));
  }
  table->uniform_element_origins
      .touched_values[table->uniform_element_origins.touched_count++] =
      value_id;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_append_touched_static_lane_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  if (table->static_lane_origins.touched_count >=
      table->static_lane_origins.touched_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        table->arena, table->static_lane_origins.touched_count,
        table->static_lane_origins.touched_count + 1,
        sizeof(*table->static_lane_origins.touched_values),
        &table->static_lane_origins.touched_capacity,
        (void**)&table->static_lane_origins.touched_values));
  }
  table->static_lane_origins
      .touched_values[table->static_lane_origins.touched_count++] = value_id;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_append_touched_uniform_scale_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  if (table->uniform_scale_origins.touched_count >=
      table->uniform_scale_origins.touched_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        table->arena, table->uniform_scale_origins.touched_count,
        table->uniform_scale_origins.touched_count + 1,
        sizeof(*table->uniform_scale_origins.touched_values),
        &table->uniform_scale_origins.touched_capacity,
        (void**)&table->uniform_scale_origins.touched_values));
  }
  table->uniform_scale_origins
      .touched_values[table->uniform_scale_origins.touched_count++] = value_id;
  return iree_ok_status();
}

static iree_status_t
loom_value_fact_table_append_touched_contextual_query_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  if (table->contextual_query_origins.touched_count >=
      table->contextual_query_origins.touched_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        table->arena, table->contextual_query_origins.touched_count,
        table->contextual_query_origins.touched_count + 1,
        sizeof(*table->contextual_query_origins.touched_values),
        &table->contextual_query_origins.touched_capacity,
        (void**)&table->contextual_query_origins.touched_values));
  }
  table->contextual_query_origins
      .touched_values[table->contextual_query_origins.touched_count++] =
      value_id;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Scratch buffers
//===----------------------------------------------------------------------===//

iree_status_t loom_value_fact_table_facts_scratch(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_facts_t** out) {
  if (count <= table->scratch.facts.capacity) {
    *out = table->scratch.facts.values;
    return iree_ok_status();
  }
  loom_value_facts_t* new_scratch = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(table->transient_arena, count,
                                                 sizeof(loom_value_facts_t),
                                                 (void**)&new_scratch));
  table->scratch.facts.values = new_scratch;
  table->scratch.facts.capacity = count;
  *out = new_scratch;
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_value_id_scratch(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_id_t** out) {
  if (count <= table->scratch.value_ids.capacity) {
    *out = table->scratch.value_ids.values;
    return iree_ok_status();
  }
  loom_value_id_t* new_scratch = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(table->transient_arena, count,
                                                 sizeof(loom_value_id_t),
                                                 (void**)&new_scratch));
  table->scratch.value_ids.values = new_scratch;
  table->scratch.value_ids.capacity = count;
  *out = new_scratch;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t loom_value_fact_table_initialize(
    loom_value_fact_table_t* table, iree_arena_allocator_t* arena,
    iree_host_size_t initial_capacity) {
  return loom_value_fact_table_initialize_with_arenas(table, arena, arena,
                                                      initial_capacity);
}

iree_status_t loom_value_fact_table_initialize_with_arenas(
    loom_value_fact_table_t* table, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* transient_arena,
    iree_host_size_t initial_capacity) {
  memset(table, 0, sizeof(*table));
  table->arena = arena;
  table->transient_arena = transient_arena;
  table->context.table = table;
  return loom_value_fact_table_allocate_initial_capacity(table,
                                                         initial_capacity);
}

void loom_value_fact_table_clear_scope(loom_value_fact_table_t* table) {
  if (table->identities.capacity) {
    for (iree_host_size_t i = 0; i < table->touched_count; ++i) {
      const loom_value_id_t value_id = table->touched_values[i];
      if (value_id < table->identities.capacity) {
        table->identities.entries[value_id] = LOOM_VALUE_ID_INVALID;
      }
    }
  }
  for (iree_host_size_t i = 0; i < table->touched_count; ++i) {
    loom_value_id_t value_id = table->touched_values[i];
    table->entries[value_id] = (loom_value_facts_t){0};
    table->touched_bits[value_id / 64] &= ~(UINT64_C(1) << (value_id % 64));
  }
  for (iree_host_size_t i = 0; i < table->uniform_element_origins.touched_count;
       ++i) {
    table->uniform_element_origins
        .entries[table->uniform_element_origins.touched_values[i]] =
        LOOM_VALUE_ID_INVALID;
  }
  for (iree_host_size_t i = 0; i < table->static_lane_origins.touched_count;
       ++i) {
    table->static_lane_origins
        .entries[table->static_lane_origins.touched_values[i]] =
        loom_value_fact_static_lane_origin_invalid();
  }
  for (iree_host_size_t i = 0; i < table->uniform_scale_origins.touched_count;
       ++i) {
    table->uniform_scale_origins
        .entries[table->uniform_scale_origins.touched_values[i]] =
        loom_value_fact_uniform_scale_origin_invalid();
  }
  for (iree_host_size_t i = 0;
       i < table->contextual_query_origins.touched_count; ++i) {
    table->contextual_query_origins
        .entries[table->contextual_query_origins.touched_values[i]] = 0;
  }
  table->touched_count = 0;
  table->count = 0;
  table->extensions.entries = NULL;
  table->extensions.capacity = 0;
  table->extensions.count = 0;
  table->extensions.buckets = NULL;
  table->extensions.bucket_count = 0;
  table->regions.buckets = NULL;
  table->regions.bucket_count = 0;
  table->regions.count = 0;
  table->regions.cfg_count = 0;
  table->regions.entries = NULL;
  table->uniform_element_origins.touched_count = 0;
  table->static_lane_origins.touched_count = 0;
  table->uniform_scale_origins.touched_count = 0;
  table->contextual_query_origins.touched_count = 0;
  table->contextual_query_origins.origin_count = 0;
  table->select_dependencies.index = NULL;
  table->select_dependencies.roots = NULL;
  table->select_dependencies.capacity = 0;
  table->scratch.facts.values = NULL;
  table->scratch.facts.capacity = 0;
  table->scratch.value_ids.values = NULL;
  table->scratch.value_ids.capacity = 0;
  table->scratch.alias_ordinals.values = NULL;
  table->scratch.alias_ordinals.capacity = 0;
  table->context.table = table;
  table->context.function = (loom_func_like_t){0};
  table->context.reference_origin = (loom_value_fact_reference_origin_t){0};
  table->context.target_facts = NULL;
}

static iree_host_size_t loom_value_fact_table_region_hash(
    const loom_region_t* region) {
  uintptr_t bits = (uintptr_t)region;
  bits ^= bits >> 17;
  bits *= (uintptr_t)0xed5ad4bbU;
  bits ^= bits >> 11;
  return (iree_host_size_t)bits;
}

static iree_status_t loom_value_fact_table_rehash_regions(
    loom_value_fact_table_t* table, iree_host_size_t new_bucket_count) {
  loom_value_fact_region_entry_t** new_buckets = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(table->transient_arena, new_bucket_count,
                                sizeof(*new_buckets), (void**)&new_buckets));
  memset(new_buckets, 0, new_bucket_count * sizeof(*new_buckets));
  for (loom_value_fact_region_entry_t* entry = table->regions.entries; entry;
       entry = entry->next_entry) {
    const iree_host_size_t bucket_index =
        loom_value_fact_table_region_hash(entry->region) &
        (new_bucket_count - 1);
    entry->next_bucket = new_buckets[bucket_index];
    new_buckets[bucket_index] = entry;
  }
  table->regions.buckets = new_buckets;
  table->regions.bucket_count = new_bucket_count;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_ensure_region_buckets(
    loom_value_fact_table_t* table, iree_host_size_t minimum_count) {
  iree_host_size_t bucket_count = table->regions.bucket_count;
  if (bucket_count == 0) {
    bucket_count = 8;
  }
  while (minimum_count > bucket_count - bucket_count / 4) {
    if (bucket_count > SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "fact region cache capacity overflow");
    }
    bucket_count *= 2;
  }
  if (bucket_count == table->regions.bucket_count) {
    return iree_ok_status();
  }
  return loom_value_fact_table_rehash_regions(table, bucket_count);
}

static loom_value_fact_region_entry_t*
loom_value_fact_table_lookup_region_entry(const loom_value_fact_table_t* table,
                                          const loom_region_t* region) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(region);
  if (table->regions.bucket_count == 0) {
    return NULL;
  }
  const iree_host_size_t bucket_index =
      loom_value_fact_table_region_hash(region) &
      (table->regions.bucket_count - 1);
  for (loom_value_fact_region_entry_t* entry =
           table->regions.buckets[bucket_index];
       entry; entry = entry->next_bucket) {
    if (entry->region == region) {
      return entry;
    }
  }
  return NULL;
}

const loom_value_fact_cfg_region_t* loom_value_fact_table_lookup_cfg_region(
    const loom_value_fact_table_t* table, const loom_region_t* region) {
  const loom_value_fact_region_entry_t* entry =
      loom_value_fact_table_lookup_region_entry(table, region);
  return entry ? entry->structure : NULL;
}

const loom_cfg_graph_t* loom_value_fact_table_lookup_cfg_graph(
    const loom_value_fact_table_t* table, const loom_region_t* region) {
  const loom_value_fact_region_entry_t* entry =
      loom_value_fact_table_lookup_region_entry(table, region);
  return entry && entry->structure ? &entry->structure->graph : NULL;
}

iree_status_t loom_value_fact_table_enumerate_cfg_graphs(
    const loom_value_fact_table_t* table,
    loom_value_fact_cfg_graph_callback_t callback) {
  iree_status_t status = iree_ok_status();
  for (const loom_value_fact_region_entry_t* entry = table->regions.entries;
       entry && iree_status_is_ok(status); entry = entry->next_entry) {
    if (entry->structure) {
      status = callback.fn(callback.user_data, &entry->structure->graph);
    }
  }
  return status;
}

static iree_status_t loom_value_fact_table_ensure_region_entry(
    loom_value_fact_table_t* table, const loom_region_t* region,
    loom_value_fact_region_entry_t** out_entry) {
  loom_value_fact_region_entry_t* entry =
      loom_value_fact_table_lookup_region_entry(table, region);
  if (!entry) {
    const iree_host_size_t new_count = table->regions.count + 1;
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_ensure_region_buckets(table, new_count));
    IREE_RETURN_IF_ERROR(iree_arena_allocate(table->transient_arena,
                                             sizeof(*entry), (void**)&entry));
    memset(entry, 0, sizeof(*entry));
    entry->region = region;
    const iree_host_size_t bucket_index =
        loom_value_fact_table_region_hash(region) &
        (table->regions.bucket_count - 1);
    entry->next_bucket = table->regions.buckets[bucket_index];
    table->regions.buckets[bucket_index] = entry;
    entry->next_entry = table->regions.entries;
    table->regions.entries = entry;
    table->regions.count = new_count;
  }
  *out_entry = entry;
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_set_region_temporal_scope(
    loom_value_fact_table_t* table, const loom_region_t* region,
    loom_value_facts_t scope) {
  loom_value_fact_region_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_region_entry(table, region, &entry));
  entry->temporal_distribution =
      scope.flags & LOOM_VALUE_FACT_DISTRIBUTION_MASK;
  return iree_ok_status();
}

loom_value_facts_t loom_value_fact_table_block_temporal_scope(
    const loom_value_fact_table_t* table, const loom_block_t* block) {
  loom_value_facts_t scope = loom_value_facts_unknown();
  const loom_value_fact_region_entry_t* entry =
      block && block->parent_region ? loom_value_fact_table_lookup_region_entry(
                                          table, block->parent_region)
                                    : NULL;
  if (!entry) {
    return scope;
  }
  scope.flags |= entry->temporal_distribution;
  const loom_value_fact_cfg_region_t* structure = entry->structure;
  if (iree_any_bit_set(block->parent_region->flags,
                       LOOM_REGION_INSTANCE_FLAG_CFG)) {
    // Builders can infer a newly inserted op before the CFG edit publishes its
    // replacement snapshot. That block has no temporal proof yet.
    const iree_host_size_t block_index =
        structure ? loom_cfg_graph_block_index(&structure->graph, block)
                  : IREE_HOST_SIZE_MAX;
    loom_value_facts_t execution = loom_value_facts_unknown();
    if (block_index == IREE_HOST_SIZE_MAX) {
      loom_value_facts_propagate_binary_distribution(scope, execution, &scope);
    } else if (structure->graph.blocks[block_index].component_is_cyclic) {
      execution = loom_value_fact_control_execution(structure->control,
                                                    (uint16_t)block_index);
      loom_value_facts_propagate_binary_distribution(scope, execution, &scope);
    }
  }
  return scope;
}

iree_status_t loom_value_fact_table_set_cfg_region(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_region_t* region,
    const loom_value_fact_cfg_region_t* structure) {
  loom_value_fact_region_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_region_entry(table, region, &entry));
  if (!entry->structure) {
    ++table->regions.cfg_count;
  }
  entry->structure = structure;
  for (iree_host_size_t i = 0; i < structure->loops.loop_count; ++i) {
    loom_value_fact_cfg_update_induction(
        table, module, structure, structure->loops.loops[i].header_index);
  }
  loom_value_fact_cfg_seed_control(table, structure, NULL);
  return iree_ok_status();
}

void loom_value_fact_table_forget_cfg_region(loom_value_fact_table_t* table,
                                             const loom_region_t* region) {
  loom_value_fact_region_entry_t* entry =
      loom_value_fact_table_lookup_region_entry(table, region);
  if (entry && entry->structure) {
    entry->structure = NULL;
    --table->regions.cfg_count;
  }
}

iree_status_t loom_value_fact_table_get_or_build_cfg_region(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_region_t* region,
    const loom_value_fact_cfg_region_t** out_region) {
  loom_value_fact_region_entry_t* entry =
      loom_value_fact_table_lookup_region_entry(table, region);
  *out_region = entry ? entry->structure : NULL;
  if (*out_region) {
    return iree_ok_status();
  }

  loom_value_fact_cfg_region_t* structure = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      table->transient_arena, sizeof(*structure), (void**)&structure));
  IREE_RETURN_IF_ERROR(loom_value_fact_cfg_region_initialize(
      module, region, table->transient_arena, structure));
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_set_cfg_region(table, module, region, structure));
  *out_region = structure;
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_define(loom_value_fact_table_t* table,
                                           loom_value_id_t value_id,
                                           loom_value_facts_t facts) {
  IREE_ASSERT_NE(facts.known_divisor, 0);
  IREE_RETURN_IF_ERROR(loom_value_fact_table_ensure_capacity(
      table, (iree_host_size_t)value_id + 1));
  uint64_t touched_bit = UINT64_C(1) << (value_id % 64);
  if (table->entries[value_id].known_divisor == 0 &&
      !(table->touched_bits[value_id / 64] & touched_bit)) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_append_touched_value(table, value_id));
    table->touched_bits[value_id / 64] |= touched_bit;
  }
  table->entries[value_id] = facts;
  if ((iree_host_size_t)value_id + 1 > table->count) {
    table->count = (iree_host_size_t)value_id + 1;
  }
  return iree_ok_status();
}

void loom_value_fact_table_undefine(loom_value_fact_table_t* table,
                                    loom_value_id_t value_id) {
  if (value_id < table->capacity) {
    table->entries[value_id] = (loom_value_facts_t){0};
  }
  if (value_id < table->identities.capacity) {
    table->identities.entries[value_id] = LOOM_VALUE_ID_INVALID;
  }
}

bool loom_value_fact_table_values_equal(const loom_value_fact_table_t* table,
                                        loom_value_id_t lhs,
                                        loom_value_id_t rhs) {
  if (lhs == LOOM_VALUE_ID_INVALID || rhs == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  if (lhs == rhs) {
    return true;
  }
  if (!table) {
    return false;
  }
  const loom_value_facts_t lhs_facts = loom_value_fact_table_lookup(table, lhs);
  const loom_value_facts_t rhs_facts = loom_value_fact_table_lookup(table, rhs);
  return loom_value_facts_is_exact(lhs_facts) &&
         loom_value_facts_is_exact(rhs_facts) &&
         !loom_value_facts_is_float(lhs_facts) &&
         !loom_value_facts_is_float(rhs_facts) &&
         lhs_facts.range_lo == rhs_facts.range_lo;
}

loom_value_id_t loom_value_fact_table_query_identity(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  if (!table || value_id >= table->identities.capacity) {
    return value_id;
  }
  const loom_value_id_t identity = table->identities.entries[value_id];
  return identity != LOOM_VALUE_ID_INVALID ? identity : value_id;
}

loom_value_set_id_t loom_value_fact_table_select_dependencies_begin(
    const loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_set_cursor_t* out_cursor) {
  loom_value_set_id_t root = 0;
  if (table && value_id < table->select_dependencies.capacity) {
    root = table->select_dependencies.roots[value_id];
  }
  loom_value_set_cursor_begin(table ? table->select_dependencies.index : NULL,
                              root, out_cursor);
  return root;
}

static loom_value_set_id_t loom_value_fact_table_select_dependency_root(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  return value_id < table->select_dependencies.capacity
             ? table->select_dependencies.roots[value_id]
             : 0;
}

static iree_status_t loom_value_fact_table_ensure_select_dependency_index(
    loom_value_fact_table_t* table) {
  if (table->select_dependencies.index) {
    return iree_ok_status();
  }
  return loom_value_set_index_allocate(table->transient_arena,
                                       &table->select_dependencies.index);
}

static iree_status_t loom_value_fact_table_ensure_select_dependency_capacity(
    loom_value_fact_table_t* table, iree_host_size_t capacity) {
  if (capacity <= table->select_dependencies.capacity) {
    return iree_ok_status();
  }
  const iree_host_size_t old_capacity = table->select_dependencies.capacity;
  IREE_RETURN_IF_ERROR(
      iree_arena_grow_array(table->transient_arena, old_capacity, capacity,
                            sizeof(*table->select_dependencies.roots),
                            &table->select_dependencies.capacity,
                            (void**)&table->select_dependencies.roots));
  memset(table->select_dependencies.roots + old_capacity, 0,
         (table->select_dependencies.capacity - old_capacity) *
             sizeof(*table->select_dependencies.roots));
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_union_select_dependency(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_set_id_t* inout_root) {
  const loom_value_set_id_t dependency =
      loom_value_fact_table_select_dependency_root(table, value_id);
  if (!dependency) {
    return iree_ok_status();
  }
  return loom_value_set_index_union(table->select_dependencies.index,
                                    *inout_root, dependency, inout_root);
}

static iree_status_t loom_value_fact_table_add_select_condition(
    loom_value_fact_table_t* table, loom_value_id_t condition,
    loom_value_set_id_t* inout_root) {
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_select_dependency_index(table));
  return loom_value_set_index_add(table->select_dependencies.index, *inout_root,
                                  condition, inout_root);
}

static iree_status_t loom_value_fact_table_set_select_dependency_root(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_set_id_t root, bool* inout_changed) {
  if (loom_value_fact_table_select_dependency_root(table, value_id) == root) {
    return iree_ok_status();
  }
  if (root) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_ensure_select_dependency_capacity(
            table, (iree_host_size_t)value_id + 1));
  }
  table->select_dependencies.roots[value_id] = root;
  if (inout_changed) {
    *inout_changed = true;
  }
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_propagate_select_dependencies(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, const loom_op_vtable_t* vtable, bool* inout_changed) {
  if (op->result_count == 0) {
    return iree_ok_status();
  }

  const bool may_add_select_condition =
      vtable && iree_any_bit_set(vtable->operand_role_mask,
                                 LOOM_OPERAND_ROLE_MASK_SELECT_CONDITION);
  if (!table->select_dependencies.index && !may_add_select_condition) {
    return iree_ok_status();
  }

  loom_value_set_id_t root = 0;
  const loom_value_id_t* operands = loom_op_const_operands(op);
  if (table->select_dependencies.index) {
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_union_select_dependency(
          table, operands[i], &root));
    }
  }

  if (may_add_select_condition) {
    for (uint16_t i = 0; i < op->operand_count; ++i) {
      if (loom_op_operand_role_at(vtable, op, i) !=
          LOOM_OPERAND_ROLE_SELECT_CONDITION) {
        continue;
      }
      const loom_value_id_t condition = operands[i];
      if (condition == LOOM_VALUE_ID_INVALID ||
          condition >= module->values.count) {
        continue;
      }
      const loom_type_t type = loom_module_value_type(module, condition);
      if (loom_type_is_scalar(type) &&
          loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1) {
        IREE_RETURN_IF_ERROR(loom_value_fact_table_add_select_condition(
            table, condition, &root));
      }
    }
  }

  if (op->attribute_count > 0 && loom_traits_are_fact_identity(op->traits)) {
    const loom_attribute_t* attributes = loom_op_const_attrs(op);
    for (uint8_t i = 0; i < op->attribute_count; ++i) {
      if (attributes[i].kind != LOOM_ATTR_PREDICATE_LIST) {
        continue;
      }
      for (uint16_t j = 0; j < attributes[i].count; ++j) {
        const loom_predicate_t* predicate = &attributes[i].predicate_list[j];
        for (uint8_t k = 0; k < predicate->arg_count; ++k) {
          if (predicate->arg_tags[k] == LOOM_PRED_ARG_VALUE) {
            IREE_RETURN_IF_ERROR(loom_value_fact_table_union_select_dependency(
                table, (loom_value_id_t)predicate->args[k], &root));
          }
        }
      }
    }
  }

  if (!root && !table->select_dependencies.roots) {
    return iree_ok_status();
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_value_id_t result = results[i];
    if (result == LOOM_VALUE_ID_INVALID || result >= module->values.count) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_set_select_dependency_root(
        table, result, root, inout_changed));
  }
  return iree_ok_status();
}

// The caller has already defined the result's numeric facts, which also owns
// its touched-value membership for scope cleanup.
static iree_status_t loom_value_fact_table_set_identity(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_id_t identity) {
  if (identity == value_id) {
    if (value_id < table->identities.capacity) {
      table->identities.entries[value_id] = LOOM_VALUE_ID_INVALID;
    }
    return iree_ok_status();
  }
  const iree_host_size_t old_capacity = table->identities.capacity;
  if (value_id >= old_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        table->arena, old_capacity, (iree_host_size_t)value_id + 1,
        sizeof(*table->identities.entries), &table->identities.capacity,
        (void**)&table->identities.entries));
    for (iree_host_size_t i = old_capacity; i < table->identities.capacity;
         ++i) {
      table->identities.entries[i] = LOOM_VALUE_ID_INVALID;
    }
  }
  table->identities.entries[value_id] = identity;
  return iree_ok_status();
}

static bool loom_value_fact_table_lookup_uniform_element_origin(
    const loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_id_t* out_scalar_value_id) {
  if (out_scalar_value_id) {
    *out_scalar_value_id = LOOM_VALUE_ID_INVALID;
  }
  if (value_id >= table->uniform_element_origins.capacity ||
      table->uniform_element_origins.entries == NULL) {
    return false;
  }
  const loom_value_id_t scalar_value_id =
      table->uniform_element_origins.entries[value_id];
  if (scalar_value_id == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  if (out_scalar_value_id) {
    *out_scalar_value_id = scalar_value_id;
  }
  return true;
}

iree_status_t loom_value_fact_table_define_uniform_element_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_id_t scalar_value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      scalar_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_value_fact_table_ensure_uniform_origin_capacity(
      table, (iree_host_size_t)value_id + 1));
  if (table->uniform_element_origins.entries[value_id] ==
      LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_append_touched_uniform_origin(table, value_id));
  }
  table->uniform_element_origins.entries[value_id] = scalar_value_id;
  return iree_ok_status();
}

bool loom_value_fact_table_query_uniform_element_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id, loom_value_id_t* out_scalar_value_id) {
  if (out_scalar_value_id) {
    *out_scalar_value_id = LOOM_VALUE_ID_INVALID;
  }
  if (table == NULL || module == NULL || value_id >= module->values.count) {
    return false;
  }
  const loom_type_t value_type = loom_module_value_type(module, value_id);
  if (!loom_type_is_shaped(value_type)) {
    return false;
  }

  loom_value_id_t scalar_value_id = LOOM_VALUE_ID_INVALID;
  if (!loom_value_fact_table_lookup_uniform_element_origin(table, value_id,
                                                           &scalar_value_id) ||
      scalar_value_id >= module->values.count) {
    return false;
  }
  const loom_type_t scalar_type =
      loom_module_value_type(module, scalar_value_id);
  if (!loom_type_is_scalar(scalar_type) ||
      loom_type_element_type(scalar_type) !=
          loom_type_element_type(value_type)) {
    return false;
  }
  if (out_scalar_value_id) {
    *out_scalar_value_id = scalar_value_id;
  }
  return true;
}

static bool loom_value_fact_table_lookup_static_lane_origin(
    const loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_static_lane_origin_t* out_origin) {
  if (out_origin) {
    *out_origin = loom_value_fact_static_lane_origin_invalid();
  }
  if (value_id >= table->static_lane_origins.capacity ||
      table->static_lane_origins.entries == NULL) {
    return false;
  }
  const loom_value_fact_static_lane_origin_t origin =
      table->static_lane_origins.entries[value_id];
  if (origin.source_value_id == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  if (out_origin) {
    *out_origin = origin;
  }
  return true;
}

iree_status_t loom_value_fact_table_define_static_lane_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_static_lane_origin_t origin) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      origin.source_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_ASSERT_NE(origin.source_lane_stride, 0u);
  IREE_RETURN_IF_ERROR(loom_value_fact_table_ensure_static_lane_origin_capacity(
      table, (iree_host_size_t)value_id + 1));
  if (table->static_lane_origins.entries[value_id].source_value_id ==
      LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_append_touched_static_lane_origin(table,
                                                                value_id));
  }
  table->static_lane_origins.entries[value_id] = origin;
  return iree_ok_status();
}

bool loom_value_fact_table_query_static_lane_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id,
    loom_value_fact_static_lane_origin_t* out_origin) {
  if (out_origin) {
    *out_origin = loom_value_fact_static_lane_origin_invalid();
  }
  if (table == NULL || module == NULL || value_id >= module->values.count) {
    return false;
  }
  loom_value_fact_static_lane_origin_t origin =
      loom_value_fact_static_lane_origin_invalid();
  if (!loom_value_fact_table_lookup_static_lane_origin(table, value_id,
                                                       &origin) ||
      origin.source_value_id >= module->values.count ||
      origin.source_lane_stride == 0) {
    return false;
  }

  const loom_type_t value_type = loom_module_value_type(module, value_id);
  const loom_type_t source_type =
      loom_module_value_type(module, origin.source_value_id);
  if (!loom_type_is_vector(value_type) || !loom_type_is_vector(source_type)) {
    return false;
  }

  uint64_t value_lane_count = 0;
  uint64_t source_lane_count = 0;
  if (!loom_type_static_element_count(value_type, &value_lane_count) ||
      !loom_type_static_element_count(source_type, &source_lane_count)) {
    return false;
  }
  uint64_t max_source_lane = origin.source_lane_offset;
  if (value_lane_count > 0) {
    const uint64_t lane_delta_count = value_lane_count - 1u;
    const uint64_t stride = origin.source_lane_stride;
    if (lane_delta_count > (UINT64_MAX - max_source_lane) / stride) {
      return false;
    }
    max_source_lane += lane_delta_count * stride;
  }
  if (max_source_lane >= source_lane_count) {
    return false;
  }

  if (out_origin) {
    *out_origin = origin;
  }
  return true;
}

static bool loom_value_fact_table_lookup_uniform_scale_origin(
    const loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_uniform_scale_origin_t* out_origin) {
  if (out_origin) {
    *out_origin = loom_value_fact_uniform_scale_origin_invalid();
  }
  if (value_id >= table->uniform_scale_origins.capacity ||
      table->uniform_scale_origins.entries == NULL) {
    return false;
  }
  const loom_value_fact_uniform_scale_origin_t origin =
      table->uniform_scale_origins.entries[value_id];
  if (origin.source_value_id == LOOM_VALUE_ID_INVALID ||
      origin.scale_value_id == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  if (out_origin) {
    *out_origin = origin;
  }
  return true;
}

iree_status_t loom_value_fact_table_define_uniform_scale_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_uniform_scale_origin_t origin) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      origin.source_value_id == LOOM_VALUE_ID_INVALID ||
      origin.scale_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_uniform_scale_origin_capacity(
          table, (iree_host_size_t)value_id + 1));
  if (table->uniform_scale_origins.entries[value_id].source_value_id ==
      LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_append_touched_uniform_scale_origin(table,
                                                                  value_id));
  }
  table->uniform_scale_origins.entries[value_id] = origin;
  return iree_ok_status();
}

bool loom_value_fact_table_query_uniform_scale_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id,
    loom_value_fact_uniform_scale_origin_t* out_origin) {
  if (out_origin) {
    *out_origin = loom_value_fact_uniform_scale_origin_invalid();
  }
  if (table == NULL || module == NULL || value_id >= module->values.count) {
    return false;
  }
  loom_value_fact_uniform_scale_origin_t origin =
      loom_value_fact_uniform_scale_origin_invalid();
  if (!loom_value_fact_table_lookup_uniform_scale_origin(table, value_id,
                                                         &origin) ||
      origin.source_value_id >= module->values.count ||
      origin.scale_value_id >= module->values.count) {
    return false;
  }

  const loom_type_t value_type = loom_module_value_type(module, value_id);
  const loom_type_t source_type =
      loom_module_value_type(module, origin.source_value_id);
  const loom_type_t scale_type =
      loom_module_value_type(module, origin.scale_value_id);
  if (!loom_type_is_vector(value_type) || !loom_type_is_vector(source_type) ||
      !loom_type_is_scalar(scale_type) ||
      loom_type_element_type(value_type) !=
          loom_type_element_type(source_type) ||
      loom_type_element_type(value_type) !=
          loom_type_element_type(scale_type)) {
    return false;
  }

  uint64_t value_lane_count = 0;
  uint64_t source_lane_count = 0;
  if (!loom_type_static_element_count(value_type, &value_lane_count) ||
      !loom_type_static_element_count(source_type, &source_lane_count) ||
      value_lane_count != source_lane_count) {
    return false;
  }

  if (out_origin) {
    *out_origin = origin;
  }
  return true;
}

static bool loom_value_fact_contextual_query_origins_equal(
    loom_value_fact_contextual_query_origin_t left,
    loom_value_fact_contextual_query_origin_t right) {
  return left.family_kind == right.family_kind &&
         loom_attribute_equal(&left.key, &right.key);
}

static bool loom_value_fact_table_lookup_contextual_query_origin(
    const loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_contextual_query_origin_t* out_origin) {
  if (out_origin) {
    *out_origin = (loom_value_fact_contextual_query_origin_t){0};
  }
  if (value_id >= table->contextual_query_origins.capacity ||
      table->contextual_query_origins.entries == NULL) {
    return false;
  }
  const uint32_t origin_id = table->contextual_query_origins.entries[value_id];
  if (origin_id == 0 ||
      origin_id > table->contextual_query_origins.origin_count) {
    return false;
  }
  if (out_origin) {
    *out_origin = table->contextual_query_origins.origins[origin_id - 1];
  }
  return true;
}

iree_status_t loom_value_fact_table_define_contextual_query_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_contextual_query_origin_t origin) {
  if (value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_ASSERT_NE(origin.family_kind, LOOM_PARAMETERIZED_ATTR_KIND_ANY);
  IREE_ASSERT_EQ(origin.reserved, 0u);
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_contextual_query_origin_capacity(
          table, (iree_host_size_t)value_id + 1));

  loom_value_fact_contextual_query_origin_t existing_origin = {0};
  if (loom_value_fact_table_lookup_contextual_query_origin(table, value_id,
                                                           &existing_origin)) {
    IREE_ASSERT(loom_value_fact_contextual_query_origins_equal(existing_origin,
                                                               origin));
    return iree_ok_status();
  }

  if (table->contextual_query_origins.origin_count >= UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "contextual query origin count exceeds uint32_t range");
  }
  if (table->contextual_query_origins.origin_count >=
      table->contextual_query_origins.origin_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        table->arena, table->contextual_query_origins.origin_count,
        table->contextual_query_origins.origin_count + 1,
        sizeof(*table->contextual_query_origins.origins),
        &table->contextual_query_origins.origin_capacity,
        (void**)&table->contextual_query_origins.origins));
  }
  const uint32_t origin_id =
      (uint32_t)++table->contextual_query_origins.origin_count;
  table->contextual_query_origins.origins[origin_id - 1] = origin;
  table->contextual_query_origins.entries[value_id] = origin_id;
  return loom_value_fact_table_append_touched_contextual_query_origin(table,
                                                                      value_id);
}

bool loom_value_fact_table_query_contextual_query_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id,
    loom_value_fact_contextual_query_origin_t* out_origin) {
  if (out_origin) {
    *out_origin = (loom_value_fact_contextual_query_origin_t){0};
  }
  if (table == NULL || module == NULL || value_id >= module->values.count ||
      !loom_type_is_scalar(loom_module_value_type(module, value_id))) {
    return false;
  }
  return loom_value_fact_table_lookup_contextual_query_origin(table, value_id,
                                                              out_origin);
}

void loom_value_fact_table_contextual_query_values(
    const loom_value_fact_table_t* table, const loom_value_id_t** out_value_ids,
    iree_host_size_t* out_value_count) {
  if (table == NULL) {
    *out_value_ids = NULL;
    *out_value_count = 0;
    return;
  }
  *out_value_ids = table->contextual_query_origins.touched_values;
  *out_value_count = table->contextual_query_origins.touched_count;
}

static iree_status_t loom_value_fact_table_clone_select_dependencies(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    loom_value_id_t value_id) {
  loom_value_set_cursor_t cursor;
  loom_value_fact_table_select_dependencies_begin(source, value_id, &cursor);
  loom_value_set_id_t root = 0;
  for (loom_value_id_t condition = loom_value_set_cursor_next(&cursor);
       condition != LOOM_VALUE_ID_INVALID;
       condition = loom_value_set_cursor_next(&cursor)) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_add_select_condition(target, condition, &root));
  }
  return loom_value_fact_table_set_select_dependency_root(
      target, value_id, root, /*inout_changed=*/NULL);
}

iree_status_t loom_value_fact_table_clone_values(
    loom_value_fact_table_t* target, loom_value_fact_table_view_t source_view,
    const loom_module_t* module) {
  const loom_value_fact_table_t* source = source_view.table;
  for (iree_host_size_t i = 0; i < source_view.value_count; ++i) {
    const loom_value_id_t value_id = source_view.value_ids[i];
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_select_dependencies(
        target, source, value_id));
    if (!loom_value_fact_table_has_entry(source, value_id)) {
      continue;
    }
    loom_value_facts_t cloned_facts = loom_value_facts_unknown();
    if (module && value_id < module->values.count) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_for_type(
          target, source, module, loom_module_value_type(module, value_id),
          source->entries[value_id], &cloned_facts));
    } else {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source, source->entries[value_id], &cloned_facts));
    }
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_define(target, value_id, cloned_facts));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_set_identity(
        target, value_id,
        loom_value_fact_table_query_identity(source, value_id)));
    loom_value_id_t scalar_origin = LOOM_VALUE_ID_INVALID;
    if (loom_value_fact_table_lookup_uniform_element_origin(source, value_id,
                                                            &scalar_origin)) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_define_uniform_element_origin(
          target, value_id, scalar_origin));
    }
    loom_value_fact_static_lane_origin_t lane_origin =
        loom_value_fact_static_lane_origin_invalid();
    if (loom_value_fact_table_lookup_static_lane_origin(source, value_id,
                                                        &lane_origin)) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_define_static_lane_origin(
          target, value_id, lane_origin));
    }
    loom_value_fact_uniform_scale_origin_t scale_origin =
        loom_value_fact_uniform_scale_origin_invalid();
    if (loom_value_fact_table_lookup_uniform_scale_origin(source, value_id,
                                                          &scale_origin)) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_define_uniform_scale_origin(
          target, value_id, scale_origin));
    }
    loom_value_fact_contextual_query_origin_t query_origin = {0};
    if (loom_value_fact_table_lookup_contextual_query_origin(source, value_id,
                                                             &query_origin)) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_define_contextual_query_origin(
          target, value_id, query_origin));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_forward_identity(
    loom_value_fact_table_t* table, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id, bool* inout_changed) {
  const loom_value_id_t identity =
      loom_value_fact_table_query_identity(table, source_value_id);
  if (identity ==
      loom_value_fact_table_query_identity(table, result_value_id)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_set_identity(table, result_value_id, identity));
  if (inout_changed) {
    *inout_changed = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_forward_uniform_origin(
    loom_value_fact_table_t* table, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id) {
  loom_value_id_t existing_origin = LOOM_VALUE_ID_INVALID;
  if (loom_value_fact_table_lookup_uniform_element_origin(
          table, result_value_id, &existing_origin)) {
    return iree_ok_status();
  }
  loom_value_id_t scalar_origin = LOOM_VALUE_ID_INVALID;
  if (!loom_value_fact_table_lookup_uniform_element_origin(
          table, source_value_id, &scalar_origin)) {
    return iree_ok_status();
  }
  return loom_value_fact_table_define_uniform_element_origin(
      table, result_value_id, scalar_origin);
}

static iree_status_t loom_value_fact_table_forward_static_lane_origin(
    loom_value_fact_table_t* table, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id) {
  loom_value_fact_static_lane_origin_t existing_origin =
      loom_value_fact_static_lane_origin_invalid();
  if (loom_value_fact_table_lookup_static_lane_origin(table, result_value_id,
                                                      &existing_origin)) {
    return iree_ok_status();
  }
  loom_value_fact_static_lane_origin_t lane_origin =
      loom_value_fact_static_lane_origin_invalid();
  if (!loom_value_fact_table_lookup_static_lane_origin(table, source_value_id,
                                                       &lane_origin)) {
    return iree_ok_status();
  }
  return loom_value_fact_table_define_static_lane_origin(table, result_value_id,
                                                         lane_origin);
}

static iree_status_t loom_value_fact_table_forward_uniform_scale_origin(
    loom_value_fact_table_t* table, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id) {
  loom_value_fact_uniform_scale_origin_t existing_origin =
      loom_value_fact_uniform_scale_origin_invalid();
  if (loom_value_fact_table_lookup_uniform_scale_origin(table, result_value_id,
                                                        &existing_origin)) {
    return iree_ok_status();
  }
  loom_value_fact_uniform_scale_origin_t scale_origin =
      loom_value_fact_uniform_scale_origin_invalid();
  if (!loom_value_fact_table_lookup_uniform_scale_origin(table, source_value_id,
                                                         &scale_origin)) {
    return iree_ok_status();
  }
  return loom_value_fact_table_define_uniform_scale_origin(
      table, result_value_id, scale_origin);
}

static iree_status_t loom_value_fact_table_forward_contextual_query_origin(
    loom_value_fact_table_t* table, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id) {
  if (result_value_id == LOOM_VALUE_ID_INVALID ||
      source_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  if (result_value_id < table->contextual_query_origins.capacity &&
      table->contextual_query_origins.entries != NULL &&
      table->contextual_query_origins.entries[result_value_id] != 0) {
    return iree_ok_status();
  }
  if (source_value_id >= table->contextual_query_origins.capacity ||
      table->contextual_query_origins.entries == NULL) {
    return iree_ok_status();
  }
  const uint32_t origin_id =
      table->contextual_query_origins.entries[source_value_id];
  if (origin_id == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_contextual_query_origin_capacity(
          table, (iree_host_size_t)result_value_id + 1));
  table->contextual_query_origins.entries[result_value_id] = origin_id;
  return loom_value_fact_table_append_touched_contextual_query_origin(
      table, result_value_id);
}

iree_status_t loom_value_fact_table_propagate_origins(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, bool* inout_changed) {
  const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_value_id_t* results = loom_op_const_results(op);
  if (loom_traits_are_value_alias(traits) && op->operand_count >= 1 &&
      op->result_count >= 1) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_forward_identity(
        table, operands[0], results[0], inout_changed));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_forward_uniform_origin(
        table, operands[0], results[0]));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_forward_static_lane_origin(
        table, operands[0], results[0]));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_forward_uniform_scale_origin(
        table, operands[0], results[0]));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_forward_contextual_query_origin(
        table, operands[0], results[0]));
  }
  if (loom_traits_are_fact_identity(traits)) {
    const uint16_t pair_count = op->operand_count < op->result_count
                                    ? op->operand_count
                                    : op->result_count;
    for (uint16_t i = 0; i < pair_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_forward_identity(
          table, operands[i], results[i], inout_changed));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_forward_uniform_origin(
          table, operands[i], results[i]));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_forward_static_lane_origin(
          table, operands[i], results[i]));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_forward_uniform_scale_origin(
          table, operands[i], results[i]));
      IREE_RETURN_IF_ERROR(
          loom_value_fact_table_forward_contextual_query_origin(
              table, operands[i], results[i]));
    }
  }
  return iree_ok_status();
}
