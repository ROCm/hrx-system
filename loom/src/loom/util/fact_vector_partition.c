// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_vector_partition.h"

#include <string.h>

#include "loom/ir/structural_hash.h"
#include "loom/util/fact_table.h"

enum {
  LOOM_VECTOR_PARTITION_EXACT = 1u << 0,
  LOOM_VECTOR_PARTITION_SNAPSHOT = 1u << 1,
};

typedef struct loom_value_fact_vector_partition_storage_t {
  // Component arrays follow this record in the same allocation.
  loom_value_fact_vector_partition_t partition;
  // Views whose components still have materializable SSA definitions.
  uint32_t available_views;
} loom_value_fact_vector_partition_storage_t;

typedef struct loom_value_fact_vector_partition_use_t {
  // Next retained component referencing the same SSA value.
  struct loom_value_fact_vector_partition_use_t* next;
  // Record owning the referenced component, possibly shared by value aliases.
  loom_value_fact_vector_partition_storage_t* storage;
  // Component index in the record's contiguous exact and snapshot arrays.
  iree_host_size_t component_index;
} loom_value_fact_vector_partition_use_t;

typedef struct loom_value_fact_vector_partition_entry_t {
  // SSA value indexing this slot, or INVALID for an unoccupied slot.
  loom_value_id_t value_id;
  // Retained partition for this vector value, or NULL.
  loom_value_fact_vector_partition_storage_t* storage;
  // Retained component references to this SSA value.
  loom_value_fact_vector_partition_use_t* uses;
} loom_value_fact_vector_partition_entry_t;

struct loom_value_fact_vector_partitions_t {
  // Open-addressed partition and component-reference entries.
  loom_value_fact_vector_partition_entry_t* entries;
  // Power-of-two slot count, or zero before the first publication.
  iree_host_size_t capacity;
  // Occupied keys, including values whose facts have been cleared.
  iree_host_size_t count;
};

static loom_value_fact_vector_partition_entry_t*
loom_value_fact_vector_partitions_find_slot(
    loom_value_fact_vector_partition_entry_t* entries,
    iree_host_size_t capacity, loom_value_id_t value_id) {
  const iree_host_size_t mask = capacity - 1;
  iree_host_size_t slot = loom_structural_hash_finalize(value_id) & mask;
  while (entries[slot].value_id != LOOM_VALUE_ID_INVALID &&
         entries[slot].value_id != value_id) {
    slot = (slot + 1) & mask;
  }
  return &entries[slot];
}

static loom_value_fact_vector_partition_entry_t*
loom_value_fact_vector_partitions_lookup(
    const loom_value_fact_vector_partitions_t* partitions,
    loom_value_id_t value_id) {
  if (!partitions || !partitions->capacity ||
      value_id == LOOM_VALUE_ID_INVALID) {
    return NULL;
  }
  loom_value_fact_vector_partition_entry_t* entry =
      loom_value_fact_vector_partitions_find_slot(
          partitions->entries, partitions->capacity, value_id);
  return entry->value_id == value_id ? entry : NULL;
}

static iree_status_t loom_value_fact_vector_partitions_ensure_entry(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_vector_partition_entry_t** out_entry) {
  *out_entry = loom_value_fact_vector_partitions_lookup(
      table->vector_partitions, value_id);
  if (*out_entry) {
    return iree_ok_status();
  }
  if (!table->vector_partitions) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        table->transient_arena, sizeof(*table->vector_partitions),
        (void**)&table->vector_partitions));
    *table->vector_partitions = (loom_value_fact_vector_partitions_t){0};
  }
  loom_value_fact_vector_partitions_t* partitions = table->vector_partitions;
  if (partitions->count >= partitions->capacity - partitions->capacity / 4) {
    iree_host_size_t capacity = 16;
    if (partitions->capacity &&
        !iree_host_size_checked_mul(partitions->capacity, 2, &capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "vector partition index capacity overflow");
    }
    loom_value_fact_vector_partition_entry_t* entries = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        table->transient_arena, capacity, sizeof(*entries), (void**)&entries));
    for (iree_host_size_t i = 0; i < capacity; ++i) {
      entries[i] = (loom_value_fact_vector_partition_entry_t){
          .value_id = LOOM_VALUE_ID_INVALID,
      };
    }
    for (iree_host_size_t i = 0; i < partitions->capacity; ++i) {
      const loom_value_fact_vector_partition_entry_t entry =
          partitions->entries[i];
      if (entry.value_id != LOOM_VALUE_ID_INVALID) {
        *loom_value_fact_vector_partitions_find_slot(entries, capacity,
                                                     entry.value_id) = entry;
      }
    }
    partitions->entries = entries;
    partitions->capacity = capacity;
  }
  loom_value_fact_vector_partition_entry_t* entry =
      loom_value_fact_vector_partitions_find_slot(
          partitions->entries, partitions->capacity, value_id);
  entry->value_id = value_id;
  ++partitions->count;
  *out_entry = entry;
  return iree_ok_status();
}

loom_value_fact_vector_partition_t loom_value_fact_table_query_vector_partition(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  const loom_value_fact_vector_partitions_t* partitions =
      table ? table->vector_partitions : NULL;
  const loom_value_fact_vector_partition_entry_t* entry =
      loom_value_fact_vector_partitions_lookup(partitions, value_id);
  if (!entry || !entry->storage) {
    return (loom_value_fact_vector_partition_t){0};
  }
  const loom_value_fact_vector_partition_storage_t* storage = entry->storage;
  if (storage->available_views == 0) {
    return (loom_value_fact_vector_partition_t){0};
  }
  loom_value_fact_vector_partition_t partition = storage->partition;
  if (!(storage->available_views & LOOM_VECTOR_PARTITION_EXACT)) {
    partition.components = NULL;
    partition.count = 0;
    partition.regular_lane_count = 0;
  }
  if (!(storage->available_views & LOOM_VECTOR_PARTITION_SNAPSHOT)) {
    partition.snapshot_components = NULL;
    partition.snapshot_count = 0;
    partition.snapshot_lane_count = 0;
  }
  return partition;
}

static uint32_t loom_value_fact_vector_partition_use_views(
    const loom_value_fact_vector_partition_use_t* use) {
  const loom_value_fact_vector_partition_t* partition =
      &use->storage->partition;
  if (use->component_index >= partition->count) {
    return LOOM_VECTOR_PARTITION_SNAPSHOT;
  }
  return LOOM_VECTOR_PARTITION_EXACT |
         (partition->snapshot_components == partition->components
              ? LOOM_VECTOR_PARTITION_SNAPSHOT
              : 0);
}

iree_status_t loom_value_fact_table_replace_vector_partition_value(
    loom_value_fact_table_t* table, loom_value_id_t old_value,
    loom_value_id_t new_value) {
  loom_value_fact_vector_partitions_t* partitions = table->vector_partitions;
  loom_value_fact_vector_partition_entry_t* old_entry =
      loom_value_fact_vector_partitions_lookup(partitions, old_value);
  if (!old_entry || !old_entry->uses || old_value == new_value) {
    return iree_ok_status();
  }
  loom_value_fact_vector_partition_entry_t* new_entry = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_vector_partitions_ensure_entry(
      table, new_value, &new_entry));
  // Inserting the replacement may rehash the private entry records.
  old_entry = loom_value_fact_vector_partitions_lookup(partitions, old_value);
  loom_value_fact_vector_partition_use_t* use = old_entry->uses;
  old_entry->uses = NULL;
  while (use) {
    loom_value_fact_vector_partition_use_t* next = use->next;
    if (use->storage->available_views &
        loom_value_fact_vector_partition_use_views(use)) {
      loom_value_fact_vector_partition_component_t* components =
          (loom_value_fact_vector_partition_component_t*)(use->storage + 1);
      components[use->component_index].value = new_value;
      use->next = new_entry->uses;
      new_entry->uses = use;
    }
    use = next;
  }
  return iree_ok_status();
}

void loom_value_fact_table_invalidate_vector_partition_value(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  loom_value_fact_vector_partitions_t* partitions = table->vector_partitions;
  loom_value_fact_vector_partition_entry_t* entry =
      loom_value_fact_vector_partitions_lookup(partitions, value_id);
  if (!entry) {
    return;
  }
  for (loom_value_fact_vector_partition_use_t* use = entry->uses; use;
       use = use->next) {
    use->storage->available_views &=
        ~loom_value_fact_vector_partition_use_views(use);
  }
  // Retain the occupied key so collision chains need no deletion markers.
  entry->storage = NULL;
  entry->uses = NULL;
}

iree_status_t loom_value_fact_table_vector_partition_component_scratch(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_fact_vector_partition_component_t** out_components) {
  if (count <= table->scratch.vector_partition_components.capacity) {
    *out_components = table->scratch.vector_partition_components.values;
    return iree_ok_status();
  }
  loom_value_fact_vector_partition_component_t* new_scratch = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(table->transient_arena, count,
                                                 sizeof(*new_scratch),
                                                 (void**)&new_scratch));
  table->scratch.vector_partition_components.values = new_scratch;
  table->scratch.vector_partition_components.capacity = count;
  *out_components = new_scratch;
  return iree_ok_status();
}

bool loom_value_fact_table_clear_vector_partition(
    loom_value_fact_table_t* table, loom_value_id_t value_id) {
  loom_value_fact_vector_partitions_t* partitions = table->vector_partitions;
  loom_value_fact_vector_partition_entry_t* entry =
      loom_value_fact_vector_partitions_lookup(partitions, value_id);
  if (entry && entry->storage) {
    entry->storage = NULL;
    return true;
  }
  return false;
}

static bool loom_value_fact_vector_partition_components_equal(
    const loom_value_fact_vector_partition_component_t* lhs,
    const loom_value_fact_vector_partition_component_t* rhs, uint32_t count) {
  return count == 0 || memcmp(lhs, rhs, count * sizeof(*lhs)) == 0;
}

iree_status_t loom_value_fact_table_define_vector_partition(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_vector_partition_t partition, bool* inout_changed) {
  if (partition.count == 0 && partition.snapshot_count == 0) {
    if (loom_value_fact_table_clear_vector_partition(table, value_id) &&
        inout_changed) {
      *inout_changed = true;
    }
    return iree_ok_status();
  }
  IREE_ASSERT(partition.count == 0 || partition.components != NULL);
  IREE_ASSERT_NE(partition.lane_count, 0u);
  uint64_t lane_count = 0;
  for (uint32_t i = 0; i < partition.count; ++i) {
    IREE_ASSERT_NE(partition.components[i].lane_count, 0u);
    lane_count += partition.components[i].lane_count;
    if (partition.regular_lane_count != 0) {
      IREE_ASSERT_LE(partition.components[i].lane_count,
                     partition.regular_lane_count);
      if (i + 1u < partition.count) {
        IREE_ASSERT_EQ(partition.components[i].lane_count,
                       partition.regular_lane_count);
      }
    }
  }
  IREE_ASSERT(partition.count == 0 || lane_count == partition.lane_count);
  uint64_t snapshot_lane_count = 0;
  if (partition.snapshot_count == 0) {
    IREE_ASSERT(partition.snapshot_components == NULL);
    IREE_ASSERT_EQ(partition.snapshot_lane_count, 0u);
  } else {
    IREE_ASSERT(partition.snapshot_components != NULL);
    IREE_ASSERT_NE(partition.snapshot_lane_count, 0u);
    for (uint32_t i = 0; i < partition.snapshot_count; ++i) {
      IREE_ASSERT_NE(partition.snapshot_components[i].lane_count, 0u);
      IREE_ASSERT_LE(partition.snapshot_components[i].lane_count,
                     partition.snapshot_lane_count);
      if (i + 1u < partition.snapshot_count) {
        IREE_ASSERT_EQ(partition.snapshot_components[i].lane_count,
                       partition.snapshot_lane_count);
      }
      snapshot_lane_count += partition.snapshot_components[i].lane_count;
    }
    IREE_ASSERT_EQ(snapshot_lane_count, partition.lane_count);
  }
  const loom_value_fact_vector_partition_t previous =
      loom_value_fact_table_query_vector_partition(table, value_id);
  const iree_host_size_t component_bytes =
      partition.count * sizeof(*partition.components);
  const bool snapshot_reuses_components =
      partition.snapshot_count == partition.count &&
      loom_value_fact_vector_partition_components_equal(
          partition.snapshot_components, partition.components, partition.count);
  const iree_host_size_t snapshot_component_bytes =
      snapshot_reuses_components
          ? 0
          : partition.snapshot_count * sizeof(*partition.snapshot_components);
  if (previous.count == partition.count &&
      previous.lane_count == partition.lane_count &&
      previous.regular_lane_count == partition.regular_lane_count &&
      previous.snapshot_count == partition.snapshot_count &&
      previous.snapshot_lane_count == partition.snapshot_lane_count &&
      previous.flags == partition.flags &&
      loom_value_fact_vector_partition_components_equal(
          previous.components, partition.components, partition.count) &&
      loom_value_fact_vector_partition_components_equal(
          previous.snapshot_components, partition.snapshot_components,
          partition.snapshot_count)) {
    return iree_ok_status();
  }

  const iree_host_size_t component_count =
      (iree_host_size_t)partition.count +
      (snapshot_reuses_components ? 0 : partition.snapshot_count);
  loom_value_fact_vector_partition_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      table->transient_arena,
      sizeof(*storage) + component_bytes + snapshot_component_bytes +
          component_count * sizeof(loom_value_fact_vector_partition_use_t),
      (void**)&storage));
  loom_value_fact_vector_partition_component_t* components =
      (loom_value_fact_vector_partition_component_t*)(storage + 1);
  if (component_bytes) {
    memcpy(components, partition.components, component_bytes);
  }
  loom_value_fact_vector_partition_component_t* snapshot_components = NULL;
  if (partition.snapshot_count != 0) {
    if (snapshot_reuses_components) {
      snapshot_components = components;
    } else {
      snapshot_components = components + partition.count;
      memcpy(snapshot_components, partition.snapshot_components,
             snapshot_component_bytes);
    }
  }
  storage->partition = (loom_value_fact_vector_partition_t){
      .components = partition.count ? components : NULL,
      .snapshot_components = snapshot_components,
      .count = partition.count,
      .lane_count = partition.lane_count,
      .regular_lane_count = partition.regular_lane_count,
      .snapshot_count = partition.snapshot_count,
      .snapshot_lane_count = partition.snapshot_lane_count,
      .flags = partition.flags,
  };
  storage->available_views =
      (partition.count ? LOOM_VECTOR_PARTITION_EXACT : 0) |
      (partition.snapshot_count ? LOOM_VECTOR_PARTITION_SNAPSHOT : 0);
  loom_value_fact_vector_partition_use_t* uses =
      (loom_value_fact_vector_partition_use_t*)(components + component_count);
  loom_value_fact_vector_partition_entry_t* entry = NULL;
  iree_status_t status =
      loom_value_fact_vector_partitions_ensure_entry(table, value_id, &entry);
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < component_count;
       ++i) {
    status = loom_value_fact_vector_partitions_ensure_entry(
        table, components[i].value, &entry);
  }
  IREE_RETURN_IF_ERROR(status);
  for (iree_host_size_t i = 0; i < component_count; ++i) {
    entry = loom_value_fact_vector_partitions_lookup(table->vector_partitions,
                                                     components[i].value);
    uses[i] = (loom_value_fact_vector_partition_use_t){
        .next = entry->uses,
        .storage = storage,
        .component_index = i,
    };
    entry->uses = &uses[i];
  }
  loom_value_fact_vector_partitions_lookup(table->vector_partitions, value_id)
      ->storage = storage;
  if (inout_changed) {
    *inout_changed = true;
  }
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_forward_vector_partition(
    loom_value_fact_table_t* table, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id, bool* inout_changed) {
  loom_value_fact_vector_partitions_t* partitions = table->vector_partitions;
  const loom_value_fact_vector_partition_entry_t* source_entry =
      loom_value_fact_vector_partitions_lookup(partitions, source_value_id);
  if (!source_entry || !source_entry->storage ||
      !source_entry->storage->available_views) {
    if (loom_value_fact_table_clear_vector_partition(table, result_value_id) &&
        inout_changed) {
      *inout_changed = true;
    }
    return iree_ok_status();
  }
  loom_value_fact_vector_partition_storage_t* storage = source_entry->storage;
  loom_value_fact_vector_partition_entry_t* result_entry =
      loom_value_fact_vector_partitions_lookup(partitions, result_value_id);
  if (result_entry && result_entry->storage == storage) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_value_fact_vector_partitions_ensure_entry(
      table, result_value_id, &result_entry));
  result_entry->storage = storage;
  if (inout_changed) {
    *inout_changed = true;
  }
  return iree_ok_status();
}
