// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_ordinal_map.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/structural_hash.h"

// Returns either the matching bucket or the first empty bucket in its probe
// chain. At most half the buckets are occupied, so every lookup terminates.
static uint32_t loom_value_fact_ordinal_map_find_bucket(
    const loom_value_fact_ordinal_map_t* map, loom_value_id_t value_id) {
  const uint32_t mask = map->bucket_count - 1;
  const uint32_t hash =
      loom_structural_hash_finalize(loom_structural_hash_mix_u32(
          loom_structural_hash_initialize(), value_id));
  uint32_t bucket = hash & mask;
  while (map->buckets[bucket] != 0 &&
         map->values[map->buckets[bucket] - 1] != value_id) {
    bucket = (bucket + 1) & mask;
  }
  return bucket;
}

iree_status_t loom_value_fact_ordinal_map_initialize(
    loom_value_fact_table_t* table, const loom_value_id_t* values,
    uint16_t value_count, bool use_index,
    loom_value_fact_ordinal_map_t* out_map) {
  *out_map = (loom_value_fact_ordinal_map_t){
      .values = values,
      .value_count = value_count,
  };
  if (!use_index || value_count == 0) {
    return iree_ok_status();
  }

  const uint32_t bucket_count =
      iree_math_round_up_to_pow2_u32((uint32_t)value_count * 2);
  if (bucket_count > table->scratch.alias_ordinals.capacity) {
    uint16_t* buckets = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(table->transient_arena, bucket_count,
                                  sizeof(*buckets), (void**)&buckets));
    memset(buckets, 0, bucket_count * sizeof(*buckets));
    table->scratch.alias_ordinals.buckets = buckets;
    table->scratch.alias_ordinals.capacity = bucket_count;
  }

  out_map->buckets = table->scratch.alias_ordinals.buckets;
  out_map->bucket_count = bucket_count;
  for (uint16_t i = 0; i < value_count; ++i) {
    const uint32_t bucket =
        loom_value_fact_ordinal_map_find_bucket(out_map, values[i]);
    if (out_map->buckets[bucket] == 0) {
      out_map->buckets[bucket] = (uint16_t)(i + 1);
    }
  }
  return iree_ok_status();
}

void loom_value_fact_ordinal_map_reset(loom_value_fact_ordinal_map_t* map) {
  if (map->buckets) {
    memset(map->buckets, 0, map->bucket_count * sizeof(*map->buckets));
  }
  *map = (loom_value_fact_ordinal_map_t){0};
}

bool loom_value_fact_ordinal_map_find(const loom_value_fact_ordinal_map_t* map,
                                      loom_value_id_t value_id,
                                      uint16_t* out_ordinal) {
  if (map->buckets) {
    const uint16_t ordinal_plus_one =
        map->buckets[loom_value_fact_ordinal_map_find_bucket(map, value_id)];
    if (ordinal_plus_one == 0) {
      return false;
    }
    *out_ordinal = ordinal_plus_one - 1;
    return true;
  }

  for (uint16_t i = 0; i < map->value_count; ++i) {
    if (map->values[i] == value_id) {
      *out_ordinal = i;
      return true;
    }
  }
  return false;
}
