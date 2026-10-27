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
#include "loom/target/facts_builder.h"
#include "loom/util/fact_cfg.h"
#include "loom/util/fact_loop.h"
#include "loom/util/fact_loop_replay.h"
#include "loom/util/fact_relation.h"
#include "loom/util/fact_vector_partition.h"

//===----------------------------------------------------------------------===//
// Capacity management
//===----------------------------------------------------------------------===//

typedef struct loom_value_fact_counted_loop_domain_t {
  // Body argument whose legal uses lie in this counted domain.
  loom_value_id_t induction_value;
  // Current SSA bounds supplied by the counted-loop fact owner.
  loom_loop_domain_t domain;
} loom_value_fact_counted_loop_domain_t;

struct loom_value_fact_region_entry_t {
  // Region whose execution context and optional CFG structure are retained.
  const loom_region_t* region;
  // Owning operation retained by region construction and inference.
  loom_op_t* parent_op;
  // Distribution inherited from enclosing CFG cycles, excluding this region.
  uint32_t temporal_distribution;
  // Boolean selector truth on region entry, stored as
  // loom_region_branch_truth_t.
  uint8_t branch_truth;
  // Whether enclosing control may execute this region repeatedly.
  bool may_repeat;
  // Immutable target selected for operations executing in this region.
  const loom_target_facts_t* target_facts;
  // CFG and forwarding components retained for the populated fact scope.
  const loom_value_fact_cfg_region_t* structure;
  // Condition-loop equation retained for the populated fact scope, when
  // present.
  loom_value_fact_induction_t* induction;
  // Counted induction domain retained for this body, when present.
  loom_value_fact_counted_loop_domain_t* counted_domain;
  // Condition facts projected onto this region's entry arguments, when any.
  loom_condition_edge_projection_t* condition_projection;
  // Complete loop solve borrowed only while its outer solve is active.
  loom_value_fact_loop_replay_record_t* loop_replay;
  // Next entry in the region-address hash collision chain.
  loom_value_fact_region_entry_t* next_bucket;
};

static_assert(sizeof(loom_value_fact_region_entry_t) <= 80,
              "region execution facts must retain a compact cache footprint");

static iree_status_t loom_value_fact_table_ensure_range(
    loom_value_fact_table_t* table, loom_value_id_t first_value_id,
    iree_host_size_t end_value_id) {
  if (first_value_id >= table->first_value_id &&
      end_value_id <= table->first_value_id + table->capacity) {
    return iree_ok_status();
  }
  const iree_host_size_t old_capacity = table->capacity;
  first_value_id &= ~UINT32_C(63);
  if (old_capacity) {
    first_value_id = iree_min(first_value_id, table->first_value_id);
    end_value_id = iree_max(end_value_id, table->first_value_id + old_capacity);
  }
  iree_host_size_t new_capacity = old_capacity;
  loom_value_facts_t* entries = table->entries;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      table->arena, 0, end_value_id - first_value_id,
      sizeof(loom_value_facts_t), &new_capacity, (void**)&entries));
  // Prepending reserves headroom before the old range. Round upward to retain
  // the old suffix while moving touched membership by whole words.
  if (old_capacity && first_value_id < table->first_value_id) {
    first_value_id =
        end_value_id > new_capacity
            ? (loom_value_id_t)iree_host_align(end_value_id - new_capacity, 64)
            : 0;
  }
  const iree_host_size_t prefix_count =
      old_capacity ? table->first_value_id - first_value_id : 0;
  memset(entries, 0, prefix_count * sizeof(*entries));
  if (old_capacity) {
    memcpy(entries + prefix_count, table->entries,
           old_capacity * sizeof(*entries));
  }
  memset(entries + prefix_count + old_capacity, 0,
         (new_capacity - prefix_count - old_capacity) * sizeof(*entries));
  iree_host_size_t old_word_count = (old_capacity + 63) / 64;
  iree_host_size_t word_count = (new_capacity + 63) / 64;
  uint64_t* touched_bits = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      table->arena, word_count, sizeof(*touched_bits), (void**)&touched_bits));
  const iree_host_size_t prefix_word_count = prefix_count / 64;
  memset(touched_bits, 0, prefix_word_count * sizeof(*touched_bits));
  if (old_word_count) {
    memcpy(touched_bits + prefix_word_count, table->touched_bits,
           old_word_count * sizeof(*touched_bits));
  }
  memset(touched_bits + prefix_word_count + old_word_count, 0,
         (word_count - prefix_word_count - old_word_count) *
             sizeof(*touched_bits));
  if (table->replay.generation) {
    uint32_t* generations = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(table->transient_arena, new_capacity,
                                  sizeof(*generations), (void**)&generations));
    memset(generations, 0, prefix_count * sizeof(*generations));
    if (old_capacity) {
      memcpy(generations + prefix_count, table->replay.generations,
             old_capacity * sizeof(*generations));
    }
    memset(generations + prefix_count + old_capacity, 0,
           (new_capacity - prefix_count - old_capacity) * sizeof(*generations));
    table->replay.generations = generations;
  }
  table->entries = entries;
  table->first_value_id = first_value_id;
  table->capacity = new_capacity;
  table->touched_bits = touched_bits;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_allocate_initial_capacity(
    loom_value_fact_table_t* table, iree_host_size_t capacity) {
  return loom_value_fact_table_ensure_range(table, 0, capacity);
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

iree_status_t loom_value_fact_table_reserve_range(
    loom_value_fact_table_t* table, loom_value_fact_range_t range) {
  if (range.capacity == 0) {
    return iree_ok_status();
  }
  if (range.capacity >
      (iree_host_size_t)LOOM_VALUE_ID_INVALID - range.first_value_id) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "fact range capacity overflow");
  }
  return loom_value_fact_table_ensure_range(
      table, range.first_value_id, range.first_value_id + range.capacity);
}

iree_status_t loom_value_fact_table_reserve(loom_value_fact_table_t* table,
                                            iree_host_size_t minimum_capacity) {
  if (minimum_capacity == 0 ||
      (table->first_value_id == 0 && minimum_capacity <= table->capacity)) {
    return iree_ok_status();
  }
  iree_host_size_t capacity =
      table->capacity ? table->capacity : minimum_capacity;
  while (capacity < minimum_capacity) {
    if (!iree_host_size_checked_mul(capacity, 2, &capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "capacity overflow");
    }
  }
  return loom_value_fact_table_ensure_range(table, 0, capacity);
}

void loom_value_fact_table_clear_scope(loom_value_fact_table_t* table) {
#ifndef NDEBUG
  ++table->observation.generation;
  table->observation.pending = false;
#endif
  table->seeds.table = NULL;
  table->seeds.selected_bits = NULL;
  table->seeds.word_count = 0;
  table->replaying = false;
  table->has_conditioned_results = false;
  table->has_boolean_branch_regions = false;
  table->has_counted_loop_domains = false;
  table->condition_integer_projection_count = 0;
  table->result_type_kind_bits = 0;
  loom_value_fact_table_clear_identities(table);
  for (iree_host_size_t i = 0; i < table->touched_count; ++i) {
    loom_value_id_t value_id = table->touched_values[i];
    const iree_host_size_t index = value_id - table->first_value_id;
    table->entries[index] = (loom_value_facts_t){0};
    table->touched_bits[index / 64] &= ~(UINT64_C(1) << (index % 64));
  }
  loom_value_fact_table_clear_origin_scope(table);
  table->touched_count = 0;
  table->replay.generation = 0;
  table->replay.generations = NULL;
  table->replay.loop = NULL;
  table->replay.reads = NULL;
  table->count = 0;
  if (!table->has_persistent_extension_entries) {
    table->extensions.entries = NULL;
    table->extensions.capacity = 0;
  }
  table->extensions.count = 0;
  table->extensions.buckets = NULL;
  table->extensions.bucket_count = 0;
  table->regions.buckets = NULL;
  table->regions.bucket_count = 0;
  table->regions.count = 0;
  table->regions.cfg_count = 0;
  table->regions.has_independent_targets = false;
  table->layout_origins = NULL;
  table->vector_partitions = NULL;
  table->select_dependencies.index = NULL;
  table->select_dependencies.pages = NULL;
  table->select_dependencies.page_capacity = 0;
  loom_value_fact_integer_relations_clear_scope(table);
  table->exact_relations.ops = NULL;
  table->exact_relations.count = 0;
  table->exact_relations.capacity = 0;
  memset(&table->scratch, 0, sizeof(table->scratch));
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
  for (iree_host_size_t i = 0; i < table->regions.bucket_count; ++i) {
    loom_value_fact_region_entry_t* entry = table->regions.buckets[i];
    while (entry) {
      loom_value_fact_region_entry_t* next = entry->next_bucket;
      const iree_host_size_t bucket_index =
          loom_value_fact_table_region_hash(entry->region) &
          (new_bucket_count - 1);
      entry->next_bucket = new_buckets[bucket_index];
      new_buckets[bucket_index] = entry;
      entry = next;
    }
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

loom_op_t* loom_value_fact_table_region_parent(
    const loom_value_fact_table_t* table, const loom_region_t* region) {
  const loom_value_fact_region_entry_t* entry =
      loom_value_fact_table_lookup_region_entry(table, region);
  return entry ? entry->parent_op : NULL;
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
  for (iree_host_size_t i = 0;
       i < table->regions.bucket_count && iree_status_is_ok(status); ++i) {
    for (const loom_value_fact_region_entry_t* entry =
             table->regions.buckets[i];
         entry && iree_status_is_ok(status); entry = entry->next_bucket) {
      if (entry->structure) {
        status = callback.fn(callback.user_data, &entry->structure->graph);
      }
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
    entry->may_repeat = true;
    const iree_host_size_t bucket_index =
        loom_value_fact_table_region_hash(region) &
        (table->regions.bucket_count - 1);
    entry->next_bucket = table->regions.buckets[bucket_index];
    table->regions.buckets[bucket_index] = entry;
    table->regions.count = new_count;
  }
  *out_entry = entry;
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_set_loop_replay(
    loom_value_fact_table_t* table, const loom_region_t* body,
    loom_value_fact_loop_replay_record_t* record) {
  loom_value_fact_region_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_region_entry(table, body, &entry));
  entry->loop_replay = record;
  return iree_ok_status();
}

loom_value_fact_loop_replay_record_t* loom_value_fact_table_lookup_loop_replay(
    const loom_value_fact_table_t* table, const loom_region_t* body) {
  const loom_value_fact_region_entry_t* entry =
      loom_value_fact_table_lookup_region_entry(table, body);
  return entry ? entry->loop_replay : NULL;
}

void loom_value_fact_table_clear_loop_replay(loom_value_fact_table_t* table,
                                             const loom_region_t* body) {
  loom_value_fact_region_entry_t* entry =
      loom_value_fact_table_lookup_region_entry(table, body);
  if (entry) {
    entry->loop_replay = NULL;
  }
}

iree_status_t loom_value_fact_table_set_region_temporal_scope(
    loom_value_fact_table_t* table, const loom_region_t* region,
    loom_op_t* parent_op, loom_value_facts_t scope, bool may_repeat) {
  loom_value_fact_region_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_region_entry(table, region, &entry));
  entry->parent_op = parent_op;
  entry->temporal_distribution =
      scope.flags & LOOM_VALUE_FACT_DISTRIBUTION_MASK;
  entry->may_repeat = may_repeat;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_set_region_target_scope(
    loom_value_fact_table_t* table, const loom_region_t* region,
    const loom_target_facts_t* target_facts) {
  loom_value_fact_region_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_region_entry(table, region, &entry));
  entry->target_facts = target_facts;
  return iree_ok_status();
}

const loom_target_facts_t* loom_value_fact_table_block_target_facts(
    const loom_value_fact_table_t* table, const loom_block_t* block) {
  if (!table->regions.has_independent_targets) {
    return table->context.target_facts;
  }
  const loom_value_fact_region_entry_t* entry =
      block && block->parent_region ? loom_value_fact_table_lookup_region_entry(
                                          table, block->parent_region)
                                    : NULL;
  return entry ? entry->target_facts : NULL;
}

static iree_status_t loom_value_fact_table_seed_region_target(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, uint8_t region_index,
    const loom_target_facts_t* inherited) {
  const loom_region_descriptor_t* descriptor = loom_op_vtable_region_descriptor(
      loom_op_vtable(module, op), region_index);
  const loom_target_facts_t* target_facts = inherited;
  if (descriptor && descriptor->execution_target_attr_index_plus_one) {
    const loom_attribute_t target = loom_op_const_attrs(
        op)[descriptor->execution_target_attr_index_plus_one - 1];
    table->regions.has_independent_targets = true;
    if (!loom_attr_is_absent(target)) {
      target_facts = NULL;
      if (table->context.resolve_region_target.fn) {
        IREE_RETURN_IF_ERROR(table->context.resolve_region_target.fn(
            table->context.resolve_region_target.user_data, module,
            loom_attr_as_symbol(target), &target_facts));
      }
    }
    IREE_RETURN_IF_ERROR(loom_target_facts_builder_project_worker(
        target_facts, table->transient_arena, &target_facts));
  }
  return loom_value_fact_table_set_region_target_scope(
      table, loom_op_regions(op)[region_index], target_facts);
}

iree_status_t loom_value_fact_table_seed_root_target_scope(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_region_t* region, const loom_op_t* parent_op) {
  if (parent_op) {
    for (uint8_t i = 0; i < parent_op->region_count; ++i) {
      if (loom_op_regions(parent_op)[i] == region) {
        return loom_value_fact_table_seed_region_target(
            table, module, parent_op, i, table->context.target_facts);
      }
    }
  }
  return loom_value_fact_table_set_region_target_scope(
      table, region, table->context.target_facts);
}

iree_status_t loom_value_fact_table_seed_nested_target_scopes(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op) {
  if (!op->region_count) {
    return iree_ok_status();
  }
  const loom_target_facts_t* inherited =
      loom_value_fact_table_block_target_facts(table, op->parent_block);
  loom_region_t* const* regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (!regions[i]) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_region_target(
        table, module, op, i, inherited));
  }
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

bool loom_value_fact_table_block_may_repeat(
    const loom_value_fact_table_t* table, const loom_block_t* block) {
  const loom_value_fact_region_entry_t* entry =
      block && block->parent_region ? loom_value_fact_table_lookup_region_entry(
                                          table, block->parent_region)
                                    : NULL;
  if (!entry || entry->may_repeat) {
    return true;
  }
  if (!iree_any_bit_set(block->parent_region->flags,
                        LOOM_REGION_INSTANCE_FLAG_CFG)) {
    return false;
  }
  const loom_value_fact_cfg_region_t* structure = entry->structure;
  const iree_host_size_t block_index =
      structure ? loom_cfg_graph_block_index(&structure->graph, block)
                : IREE_HOST_SIZE_MAX;
  return block_index == IREE_HOST_SIZE_MAX ||
         structure->graph.blocks[block_index].component_is_cyclic;
}

iree_status_t loom_value_fact_table_set_region_condition_projection(
    loom_value_fact_table_t* table, const loom_region_t* region,
    loom_condition_edge_projection_t* projection) {
  loom_value_fact_region_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_region_entry(table, region, &entry));
  entry->condition_projection = projection;
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_set_region_branch_truth(
    loom_value_fact_table_t* table, const loom_region_t* region,
    loom_region_branch_truth_t truth) {
  loom_value_fact_region_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_region_entry(table, region, &entry));
  entry->branch_truth = (uint8_t)truth;
  table->has_boolean_branch_regions |=
      truth != LOOM_REGION_BRANCH_TRUTH_UNKNOWN;
  return iree_ok_status();
}

loom_region_branch_truth_t loom_value_fact_table_lookup_region_branch_truth(
    const loom_value_fact_table_t* table, const loom_region_t* region) {
  const loom_value_fact_region_entry_t* entry =
      loom_value_fact_table_lookup_region_entry(table, region);
  return entry ? (loom_region_branch_truth_t)entry->branch_truth
               : LOOM_REGION_BRANCH_TRUTH_UNKNOWN;
}

const loom_condition_edge_projection_t*
loom_value_fact_table_lookup_region_condition_projection(
    const loom_value_fact_table_t* table, const loom_region_t* region) {
  const loom_value_fact_region_entry_t* entry =
      loom_value_fact_table_lookup_region_entry(table, region);
  return entry ? entry->condition_projection : NULL;
}

loom_condition_edge_projection_t*
loom_value_fact_table_lookup_mutable_region_condition_projection(
    loom_value_fact_table_t* table, const loom_region_t* region) {
  loom_value_fact_region_entry_t* entry =
      loom_value_fact_table_lookup_region_entry(table, region);
  return entry ? entry->condition_projection : NULL;
}

iree_status_t loom_value_fact_table_set_condition_induction(
    loom_value_fact_table_t* table, const loom_region_t* condition_region,
    loom_value_fact_induction_t induction) {
  loom_value_fact_region_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_ensure_region_entry(
      table, condition_region, &entry));
  if (!entry->induction) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(table->transient_arena,
                                             sizeof(*entry->induction),
                                             (void**)&entry->induction));
  }
  *entry->induction = induction;
  return iree_ok_status();
}

const loom_value_fact_induction_t*
loom_value_fact_table_lookup_condition_induction(
    const loom_value_fact_table_t* table,
    const loom_region_t* condition_region) {
  const loom_value_fact_region_entry_t* entry =
      loom_value_fact_table_lookup_region_entry(table, condition_region);
  return entry ? entry->induction : NULL;
}

iree_status_t loom_value_fact_table_set_counted_loop_domain(
    loom_value_fact_table_t* table, const loom_region_t* body,
    loom_value_id_t induction_value, loom_loop_domain_t domain) {
  loom_value_fact_region_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_ensure_region_entry(table, body, &entry));
  if (!entry->counted_domain) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(table->transient_arena,
                                             sizeof(*entry->counted_domain),
                                             (void**)&entry->counted_domain));
  }
  *entry->counted_domain = (loom_value_fact_counted_loop_domain_t){
      .induction_value = induction_value,
      .domain = domain,
  };
  table->has_counted_loop_domains = true;
  return iree_ok_status();
}

const loom_loop_domain_t* loom_value_fact_table_lookup_counted_loop_domain(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (!loom_value_is_block_arg(value)) {
    return NULL;
  }
  const loom_value_fact_region_entry_t* entry =
      loom_value_fact_table_lookup_region_entry(
          table, loom_value_def_block(value)->parent_region);
  return entry && entry->counted_domain &&
                 entry->counted_domain->induction_value == value_id
             ? &entry->counted_domain->domain
             : NULL;
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

iree_status_t loom_value_fact_table_touch(loom_value_fact_table_t* table,
                                          loom_value_id_t value_id) {
  IREE_RETURN_IF_ERROR(loom_value_fact_table_ensure_range(
      table, value_id, (iree_host_size_t)value_id + 1));
  const iree_host_size_t index = value_id - table->first_value_id;
  const uint64_t touched_bit = UINT64_C(1) << (index % 64);
  if (!(table->touched_bits[index / 64] & touched_bit)) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_append_touched_value(table, value_id));
    table->touched_bits[index / 64] |= touched_bit;
  }
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_enable_replay_generations(
    loom_value_fact_table_t* table) {
  if (table->replay.generation) {
    return iree_ok_status();
  }
  if (table->capacity) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(table->transient_arena, table->capacity,
                                  sizeof(*table->replay.generations),
                                  (void**)&table->replay.generations));
    memset(table->replay.generations, 0,
           table->capacity * sizeof(*table->replay.generations));
  }
  table->replay.generation = 2;
  return iree_ok_status();
}

static void loom_value_fact_table_advance_generation(
    loom_value_fact_table_t* table) {
  table->replay.generation += 2;
  if (IREE_UNLIKELY(!table->replay.generation)) {
    // Zero invalidates every evaluation watermark before the counter restarts.
    memset(table->replay.generations, 0,
           table->capacity * sizeof(*table->replay.generations));
    table->replay.generation = 2;
  }
}

iree_status_t loom_value_fact_table_define(loom_value_fact_table_t* table,
                                           loom_value_id_t value_id,
                                           loom_value_facts_t facts) {
  IREE_ASSERT_NE(facts.known_divisor, 0);
  IREE_RETURN_IF_ERROR(loom_value_fact_table_touch(table, value_id));
  const iree_host_size_t index = value_id - table->first_value_id;
  if (table->replay.generation &&
      !loom_value_facts_equal(table->entries[index], facts)) {
    loom_value_fact_table_advance_generation(table);
    // Loop arguments, seeds, and resets publish without evaluating an ordinary
    // equation. Its defining operation must establish this result before reuse.
    table->replay.generations[index] = table->replay.generation | UINT32_C(1);
  }
  table->entries[index] = facts;
  if ((iree_host_size_t)value_id + 1 > table->count) {
    table->count = (iree_host_size_t)value_id + 1;
  }
  return iree_ok_status();
}

void loom_value_fact_table_undefine(loom_value_fact_table_t* table,
                                    loom_value_id_t value_id) {
  loom_value_fact_loop_replay_invalidate(table);
  loom_value_fact_integer_relations_undefine(table, value_id);
  loom_value_fact_table_clear_origins(table, value_id);
  if (table->vector_partitions) {
    loom_value_fact_table_clear_vector_partition(table, value_id);
  }
  const iree_host_size_t index =
      (iree_host_size_t)value_id - table->first_value_id;
  if (index < table->capacity) {
    table->entries[index] = (loom_value_facts_t){0};
    if (table->replay.generation) {
      loom_value_fact_table_advance_generation(table);
      table->replay.generations[index] = table->replay.generation | UINT32_C(1);
    }
  }
  loom_value_fact_table_undefine_identity(table, value_id);
}

uint64_t loom_value_fact_table_maximum_element_count(
    const loom_value_fact_table_t* table, loom_type_t type) {
  uint64_t count = 1;
  for (uint8_t axis = 0; axis < loom_type_rank(type); ++axis) {
    int64_t extent = 0;
    if (loom_type_dim_is_dynamic_at(type, axis)) {
      const loom_value_facts_t facts = loom_value_fact_table_lookup(
          table, loom_type_dim_value_id_at(type, axis));
      if (!loom_value_facts_as_non_negative_i64_maximum(facts, &extent)) {
        return UINT64_MAX;
      }
    } else {
      extent = loom_type_dim_static_size_at(type, axis);
    }
    if (!iree_checked_mul_u64(count, (uint64_t)extent, &count)) {
      return UINT64_MAX;
    }
  }
  return count;
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
  if (loom_value_fact_table_query_identity(table, lhs) ==
      loom_value_fact_table_query_identity(table, rhs)) {
    return true;
  }
  const loom_value_facts_t lhs_facts = loom_value_fact_table_lookup(table, lhs);
  const loom_value_facts_t rhs_facts = loom_value_fact_table_lookup(table, rhs);
  return loom_value_facts_is_exact(lhs_facts) &&
         loom_value_facts_is_exact(rhs_facts) &&
         !loom_value_facts_is_float(lhs_facts) &&
         !loom_value_facts_is_float(rhs_facts) &&
         lhs_facts.range_lo == rhs_facts.range_lo;
}

void loom_value_fact_table_pending_exact_relations(
    const loom_value_fact_table_t* table, loom_op_t* const** out_ops,
    iree_host_size_t* out_op_count) {
  *out_ops = table ? table->exact_relations.ops : NULL;
  *out_op_count = table ? table->exact_relations.count : 0;
}

void loom_value_fact_table_clear_pending_exact_relations(
    loom_value_fact_table_t* table) {
  table->exact_relations.count = 0;
}

#define LOOM_VALUE_FACT_EXACT_RELATION_INITIAL_CAPACITY 8

static bool loom_value_fact_table_op_has_dynamic_predicate(
    const loom_value_fact_table_t* table, const loom_op_t* op) {
  const loom_attribute_t* attributes = loom_op_const_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    if (attributes[i].kind != LOOM_ATTR_PREDICATE_LIST) {
      continue;
    }
    for (uint16_t j = 0; j < attributes[i].count; ++j) {
      const loom_predicate_t* predicate = &attributes[i].predicate_list[j];
      for (uint8_t k = 0; k < predicate->arg_count; ++k) {
        if (predicate->arg_tags[k] != LOOM_PRED_ARG_VALUE ||
            predicate->args[k] < 0) {
          continue;
        }
        const loom_value_facts_t facts = loom_value_fact_table_lookup(
            table, (loom_value_id_t)predicate->args[k]);
        if (!loom_value_facts_is_exact(facts)) {
          return true;
        }
      }
    }
  }
  return false;
}

static iree_status_t loom_value_fact_table_record_exact_relation(
    loom_value_fact_table_t* table, const loom_op_t* op,
    loom_trait_flags_t traits) {
  if (op->result_count == 0 || loom_traits_has_side_effects(traits)) {
    return iree_ok_status();
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID ||
        !loom_value_facts_is_exact(
            loom_value_fact_table_lookup(table, results[i]))) {
      return iree_ok_status();
    }
  }
  if (!loom_value_fact_table_op_has_dynamic_predicate(table, op)) {
    return iree_ok_status();
  }
  if (table->exact_relations.count == table->exact_relations.capacity) {
    const iree_host_size_t minimum_capacity =
        table->exact_relations.capacity
            ? table->exact_relations.capacity * 2
            : LOOM_VALUE_FACT_EXACT_RELATION_INITIAL_CAPACITY;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        table->transient_arena, table->exact_relations.capacity,
        minimum_capacity, sizeof(*table->exact_relations.ops),
        &table->exact_relations.capacity, (void**)&table->exact_relations.ops));
  }
  table->exact_relations.ops[table->exact_relations.count++] = (loom_op_t*)op;
  return iree_ok_status();
}

void loom_value_fact_table_scope_values(const loom_value_fact_table_t* table,
                                        const loom_value_id_t** out_value_ids,
                                        iree_host_size_t* out_value_count) {
  if (table == NULL) {
    *out_value_ids = NULL;
    *out_value_count = 0;
    return;
  }
  *out_value_ids = table->touched_values;
  *out_value_count = table->touched_count;
}

iree_status_t loom_value_fact_table_clone_values(
    loom_value_fact_table_t* target, loom_value_fact_table_view_t source_view,
    const loom_module_t* module) {
  const loom_value_fact_table_t* source = source_view.table;
  // Copied results retain their source's whole-scope invalidation requirement.
  target->has_conditioned_results |= source->has_conditioned_results;
  for (iree_host_size_t i = 0; i < source_view.value_count; ++i) {
    const loom_value_id_t value_id = source_view.value_ids[i];
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_select_dependencies(
        target, source, value_id));
    loom_value_facts_t source_facts;
    if (!loom_value_fact_table_try_lookup(source, value_id, &source_facts)) {
      continue;
    }
    loom_value_facts_t cloned_facts = loom_value_facts_unknown();
    if (module && value_id < module->values.count) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_for_type(
          target, source, module, loom_module_value_type(module, value_id),
          source_facts, &cloned_facts));
    } else {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
          target, source, source_facts, &cloned_facts));
    }
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_define(target, value_id, cloned_facts));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_identity(
        target, value_id,
        loom_value_fact_table_query_identity(source, value_id), NULL));
    IREE_RETURN_IF_ERROR(loom_value_fact_integer_relations_clone_value(
        target, source, value_id));
    if (source->vector_partitions || target->vector_partitions) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_define_vector_partition(
          target, value_id,
          loom_value_fact_table_query_vector_partition(source, value_id),
          /*inout_changed=*/NULL));
    }
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_clone_origins(target, source, value_id));
  }
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_seed_values(
    loom_value_fact_table_t* target, loom_value_fact_table_view_t source,
    const loom_module_t* module) {
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_clone_values(target, source, module));
  iree_host_size_t word_count = 0;
  for (iree_host_size_t i = 0; i < source.value_count; ++i) {
    word_count = iree_max(word_count, source.value_ids[i] / 64 + 1);
  }
  uint64_t* selected_bits = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      target->transient_arena, word_count, sizeof(*selected_bits),
      (void**)&selected_bits));
  if (word_count != 0) {
    memset(selected_bits, 0, word_count * sizeof(*selected_bits));
  }
  for (iree_host_size_t i = 0; i < source.value_count; ++i) {
    const loom_value_id_t value = source.value_ids[i];
    if (loom_value_fact_table_has_entry(source.table, value)) {
      selected_bits[value / 64] |= UINT64_C(1) << (value % 64);
    }
  }
  target->seeds.table = source.table;
  target->seeds.selected_bits = selected_bits;
  target->seeds.word_count = word_count;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_forward_identity(
    loom_value_fact_table_t* table, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id, bool* inout_changed) {
  return loom_value_fact_table_define_identity(table, result_value_id,
                                               source_value_id, inout_changed);
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
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_forward_origins(table, operands[0], results[0]));
    if (table->vector_partitions) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_forward_vector_partition(
          table, operands[0], results[0], inout_changed));
    }
  }
  if (loom_traits_are_fact_identity(traits)) {
    const uint16_t pair_count = op->operand_count < op->result_count
                                    ? op->operand_count
                                    : op->result_count;
    for (uint16_t i = 0; i < pair_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_forward_identity(
          table, operands[i], results[i], inout_changed));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_forward_origins(
          table, operands[i], results[i]));
      if (table->vector_partitions) {
        IREE_RETURN_IF_ERROR(loom_value_fact_table_forward_vector_partition(
            table, operands[i], results[i], inout_changed));
      }
    }
  }
  IREE_RETURN_IF_ERROR(loom_value_fact_integer_relations_propagate_origins(
      table, module, op, traits, inout_changed));
  if (loom_traits_are_fact_identity(traits)) {
    // Region traversal also refreshes populated scopes. Identity inference
    // retains change detection so stable visits do not republish relations
    // that a consumer has already preserved.
    if (!inout_changed || *inout_changed) {
      IREE_RETURN_IF_ERROR(
          loom_value_fact_table_record_exact_relation(table, op, traits));
    }
  }
  return iree_ok_status();
}
