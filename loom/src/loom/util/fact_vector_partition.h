// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scope-local SSA materialization of exact static vector partitions.
//
// A partition records the authored scalar and vector values that form one
// logical vector. Numeric value facts remain independent of this structural
// materialization.

#ifndef LOOM_UTIL_FACT_VECTOR_PARTITION_H_
#define LOOM_UTIL_FACT_VECTOR_PARTITION_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_value_fact_table_t loom_value_fact_table_t;

// Maximum number of nested components expanded into one exact partition by
// structural fact producers. Direct regular vector partitions retain arbitrary
// component counts, preserving packet snapshots without making scalar insert
// chains quadratic in vector width.
#define LOOM_VALUE_FACT_VECTOR_PARTITION_EXPANSION_LIMIT 64u

// One authored scalar or vector component in logical lane order.
typedef struct loom_value_fact_vector_partition_component_t {
  // Scalar or rank-one vector SSA value.
  loom_value_id_t value;
  // Number of logical lanes contributed by the value. Scalars contribute one.
  uint32_t lane_count;
} loom_value_fact_vector_partition_component_t;

typedef enum loom_value_fact_vector_partition_flag_bits_e {
  LOOM_VALUE_FACT_VECTOR_PARTITION_FLAG_NONE = 0u,
  // Consuming the exact vector components preserves packetized computation
  // instead of only regrouping structural or memory values.
  LOOM_VALUE_FACT_VECTOR_PARTITION_FLAG_COMPUTED_PACKETS = 1u << 0,
} loom_value_fact_vector_partition_flag_bits_t;
typedef uint8_t loom_value_fact_vector_partition_flags_t;

// Exact component tuple and regular packet snapshot retained by the
// fact table. The two views may use different component widths: reductions
// consume exact leaves while packet materialization consumes the coarsest
// regular authored tuple available at the producer.
typedef struct loom_value_fact_vector_partition_t {
  // Scalar and vector components in logical lane order.
  const loom_value_fact_vector_partition_component_t* components;
  // Regular vector snapshot components in logical lane order, or NULL.
  const loom_value_fact_vector_partition_component_t* snapshot_components;
  // Number of components in the partition.
  uint32_t count;
  // Total logical lane count contributed by all components.
  uint32_t lane_count;
  // Logical lane count of every vector component except an optionally shorter
  // last component. Zero means the exact components contain scalars or
  // irregular vector widths.
  uint32_t regular_lane_count;
  // Number of components in the regular vector snapshot.
  uint32_t snapshot_count;
  // Logical lane count of every snapshot component except an optionally
  // shorter last component. Zero means no regular snapshot is available.
  uint32_t snapshot_lane_count;
  // Semantic properties established by the partition producer.
  loom_value_fact_vector_partition_flags_t flags;
} loom_value_fact_vector_partition_t;

// Returns true when every exact component is a vector with the same lane
// count. Regular partitions with a shorter terminal vector return false.
static inline bool loom_value_fact_vector_partition_has_uniform_components(
    loom_value_fact_vector_partition_t partition) {
  return partition.count != 0 && partition.regular_lane_count != 0 &&
         partition.lane_count ==
             (uint64_t)partition.count * partition.regular_lane_count;
}

// Returns reusable component scratch with room for |count| entries. The
// contents remain valid only until the next scratch request.
iree_status_t loom_value_fact_table_vector_partition_component_scratch(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_fact_vector_partition_component_t** out_components);

// Retains a copy of the exact vector partition and regular snapshot for
// |value_id|. An empty partition clears the fact without allocating.
// Recomputing a producer replaces its partition; unchanged partitions reuse
// their storage. When non-NULL, |inout_changed| is set when the retained
// partition changes.
iree_status_t loom_value_fact_table_define_vector_partition(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_vector_partition_t partition, bool* inout_changed);

// Returns the retained partition, or an empty record when none is available.
// Both component views borrow fact-table storage until the populated scope is
// cleared. Full SSA replacement updates their value IDs in place. A query must
// be repeated after erasure, which can withdraw either view independently.
// This is an indexed query and does not inspect source IR.
loom_value_fact_vector_partition_t loom_value_fact_table_query_vector_partition(
    const loom_value_fact_table_t* table, loom_value_id_t value_id);

// Shares the source partition with a value alias in the same SSA scope. An
// unavailable source clears the result partition. When non-NULL,
// |inout_changed| is set when the retained partition changes.
iree_status_t loom_value_fact_table_forward_vector_partition(
    loom_value_fact_table_t* table, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id, bool* inout_changed);

// Withdraws a partition when its producer is invalidated, without allocating,
// and returns whether a partition was present.
bool loom_value_fact_table_clear_vector_partition(
    loom_value_fact_table_t* table, loom_value_id_t value_id);

// Retargets retained component references after full SSA replacement. Only
// references indexed under |old_value| are visited; tuples shared by aliases
// are updated once. Both values have the same type. Allocation is needed only
// to grow the value index.
iree_status_t loom_value_fact_table_replace_vector_partition_value(
    loom_value_fact_table_t* table, loom_value_id_t old_value,
    loom_value_id_t new_value);

// Withdraws exact or snapshot views referencing a value whose definition or
// type is no longer available, and clears that value's own partition. Other
// views and aliases whose components remain live retain their facts.
void loom_value_fact_table_invalidate_vector_partition_value(
    loom_value_fact_table_t* table, loom_value_id_t value_id);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_FACT_VECTOR_PARTITION_H_
