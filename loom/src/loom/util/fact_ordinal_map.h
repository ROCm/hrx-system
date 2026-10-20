// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_UTIL_FACT_ORDINAL_MAP_H_
#define LOOM_UTIL_FACT_ORDINAL_MAP_H_

#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Scope-local mapping from SSA value IDs to their first ordinal in a borrowed
// sequence. Small sequences may use bounded linear lookup without storage.
// Indexed maps lease the fact table's reusable ordinal scratch until reset; at
// most one indexed map may be live on a table at a time. Storage is
// proportional to the sequence length, independent of the magnitude or span of
// its IDs.
typedef struct loom_value_fact_ordinal_map_t {
  // Borrowed value IDs in ordinal order.
  const loom_value_id_t* values;

  // Number of value IDs in values.
  uint16_t value_count;

  // Optional hash buckets containing one-based ordinals into values; zero is
  // empty. The borrowed sequence supplies each occupied bucket's key.
  uint16_t* buckets;

  // Active power-of-two bucket count, at least twice value_count when indexed.
  uint32_t bucket_count;
} loom_value_fact_ordinal_map_t;

// Initializes a map over |values|. Indexed lookup grows and populates the
// table-owned scratch map; linear lookup borrows the sequence alone. Duplicate
// value IDs resolve to their first ordinal.
iree_status_t loom_value_fact_ordinal_map_initialize(
    loom_value_fact_table_t* table, const loom_value_id_t* values,
    uint16_t value_count, bool use_index,
    loom_value_fact_ordinal_map_t* out_map);

// Releases an indexed scratch lease by clearing only its active bucket prefix.
// Linear maps require no work. The borrowed sequence must remain valid until
// reset completes.
void loom_value_fact_ordinal_map_reset(loom_value_fact_ordinal_map_t* map);

// Returns the first ordinal associated with |value_id|.
bool loom_value_fact_ordinal_map_find(const loom_value_fact_ordinal_map_t* map,
                                      loom_value_id_t value_id,
                                      uint16_t* out_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_UTIL_FACT_ORDINAL_MAP_H_
