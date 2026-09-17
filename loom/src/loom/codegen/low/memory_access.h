// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent low memory access summaries.
//
// Low scheduling consumes these summaries instead of performing alias analysis
// itself. Producers may build them conservatively from descriptor effect rows
// or precisely from source/kernel facts preserved through lowering.

#ifndef LOOM_CODEGEN_LOW_MEMORY_ACCESS_H_
#define LOOM_CODEGEN_LOW_MEMORY_ACCESS_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_op_t loom_op_t;

// Sentinel for absent analysis-owned symbolic memory expressions.
#define LOOM_LOW_MEMORY_EXPR_ID_NONE UINT32_MAX

// Sentinel for absent alias root/group identifiers.
#define LOOM_LOW_MEMORY_ALIAS_ID_NONE UINT32_MAX

typedef uint32_t loom_low_memory_expr_id_t;

typedef enum loom_low_byte_interval_precision_bits_e {
  // begin_facts carries a bounded range for the byte interval begin.
  LOOM_LOW_BYTE_INTERVAL_PRECISION_BEGIN_RANGE = 1u << 0,
  // end_facts carries a bounded range for the exclusive byte interval end.
  LOOM_LOW_BYTE_INTERVAL_PRECISION_END_RANGE = 1u << 1,
  // begin_expr_id names an exact symbolic expression for the interval begin.
  LOOM_LOW_BYTE_INTERVAL_PRECISION_BEGIN_EXPR = 1u << 2,
  // end_expr_id names an exact symbolic expression for the exclusive end.
  LOOM_LOW_BYTE_INTERVAL_PRECISION_END_EXPR = 1u << 3,
  // The interval length is exact even when begin is dynamic.
  LOOM_LOW_BYTE_INTERVAL_PRECISION_EXACT_LENGTH = 1u << 4,
} loom_low_byte_interval_precision_bits_t;
typedef uint32_t loom_low_byte_interval_precision_flags_t;

typedef struct loom_low_byte_interval_t {
  // Conservative facts for the byte interval begin relative to alias root.
  loom_value_facts_t begin_facts;
  // Conservative facts for the exclusive byte interval end relative to alias
  // root.
  loom_value_facts_t end_facts;
  // Analysis-owned exact expression ID for begin, or NONE.
  loom_low_memory_expr_id_t begin_expr_id;
  // Analysis-owned exact expression ID for end, or NONE.
  loom_low_memory_expr_id_t end_expr_id;
  // Bitset of loom_low_byte_interval_precision_bits_t values.
  loom_low_byte_interval_precision_flags_t precision_flags;
} loom_low_byte_interval_t;

typedef enum loom_low_memory_access_precision_bits_e {
  // memory_space names a non-generic low memory space.
  LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE = 1u << 0,
  // alias_root_id is known and comparable with other access summaries.
  LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT = 1u << 1,
  // alias_group_id is known and comparable with other access summaries.
  LOOM_LOW_MEMORY_ACCESS_PRECISION_GROUP = 1u << 2,
  // byte_interval carries usable range or expression precision.
  LOOM_LOW_MEMORY_ACCESS_PRECISION_INTERVAL = 1u << 3,
  // A later lane-set summary exactly describes the accessed lanes.
  LOOM_LOW_MEMORY_ACCESS_PRECISION_EXACT_LANES = 1u << 4,
  // strided_interval exactly describes one repeated byte interval per stride.
  LOOM_LOW_MEMORY_ACCESS_PRECISION_STRIDED_INTERVAL = 1u << 5,
} loom_low_memory_access_precision_bits_t;
typedef uint32_t loom_low_memory_access_precision_flags_t;

// One non-wrapping byte interval repeated at a fixed positive stride relative
// to an alias root. The interval is half-open within [0, stride_bytes).
typedef struct loom_low_strided_byte_interval_t {
  // Positive byte stride between repeated interval instances.
  uint64_t stride_bytes;
  // Inclusive byte residue where each interval begins.
  uint64_t begin_bytes;
  // Exclusive byte residue where each interval ends.
  uint64_t end_bytes;
} loom_low_strided_byte_interval_t;

typedef struct loom_low_memory_access_summary_t {
  // Normalized target-low memory space touched by this summary.
  loom_low_memory_space_t memory_space;
  // Comparable alias root identifier, or NONE when unknown.
  uint32_t alias_root_id;
  // Comparable disjoint alias group identifier, or NONE when unknown.
  uint32_t alias_group_id;
  // Bitset of loom_low_memory_access_precision_bits_t values.
  loom_low_memory_access_precision_flags_t precision_flags;
  // Optional exact repeated byte interval relative to alias_root_id.
  loom_low_strided_byte_interval_t strided_interval;
  // Optional conservative byte interval touched by this access.
  const loom_low_byte_interval_t* byte_interval;
} loom_low_memory_access_summary_t;

typedef struct loom_low_memory_access_position_t {
  // Region index of the low block when this row was recorded.
  uint16_t block_index;
  // Block-local ordinal of the low op when this row was recorded.
  uint64_t block_ordinal;
} loom_low_memory_access_position_t;

// Compares two low op positions by function order. Returns <0 when |left|
// comes before |right|, >0 when it comes after, and 0 for the same block-index
// and block-ordinal key.
static inline int loom_low_memory_access_position_compare_order(
    const loom_low_memory_access_position_t* left,
    const loom_low_memory_access_position_t* right) {
  if (left->block_index < right->block_index) {
    return -1;
  }
  if (left->block_index > right->block_index) {
    return 1;
  }
  if (left->block_ordinal < right->block_ordinal) {
    return -1;
  }
  if (left->block_ordinal > right->block_ordinal) {
    return 1;
  }
  return 0;
}

typedef struct loom_low_memory_access_record_t {
  // Low function position whose descriptor memory effect is refined by
  // |summary|.
  loom_low_memory_access_position_t position;
  // Low op whose descriptor memory effect is refined by |summary|.
  const loom_op_t* op;
  // Source-derived memory access summary for the recorded low op position.
  // Optional payloads borrow the table's retained arena independently of this
  // record, so copying records does not invalidate their summaries.
  loom_low_memory_access_summary_t summary;
} loom_low_memory_access_record_t;

typedef struct loom_low_memory_access_table_t {
  // Low function that owns the recorded low operations.
  const loom_op_t* function_op;
  // Function-order memory access records, or NULL when empty.
  const loom_low_memory_access_record_t* values;
  // Number of rows in |values|.
  iree_host_size_t count;
} loom_low_memory_access_table_t;

// Returns an empty low memory access table.
static inline loom_low_memory_access_table_t loom_low_memory_access_table_empty(
    void) {
  return (loom_low_memory_access_table_t){0};
}

// Returns true when |table| carries no memory access records.
static inline bool loom_low_memory_access_table_is_empty(
    loom_low_memory_access_table_t table) {
  return table.count == 0;
}

// Returns the canonical dependency memory-space bucket for |memory_space|.
loom_low_memory_space_t loom_low_memory_access_normalize_space(
    loom_low_memory_space_t memory_space);

// Returns true when the two memory spaces must be conservatively treated as
// possibly aliasing.
bool loom_low_memory_access_spaces_may_alias(loom_low_memory_space_t left,
                                             loom_low_memory_space_t right);

// Returns an immutable, process-lifetime summary with normalized memory-space
// precision only. Descriptor and structural effects without a more precise
// access summary share these records without allocating or copying payloads.
const loom_low_memory_access_summary_t*
loom_low_memory_access_summary_for_space(loom_low_memory_space_t memory_space);

// Returns true when two summaries must be conservatively treated as possibly
// touching the same memory.
bool loom_low_memory_access_summaries_may_alias(
    const loom_low_memory_access_summary_t* left,
    const loom_low_memory_access_summary_t* right);

// Returns true when the summaries have identical conservative alias facts.
// This proves equivalent may-alias queries, not identical runtime addresses or
// full overwrite. Retiring an access additionally requires an established
// completion dependency through an intervening opposite-kind access.
bool loom_low_memory_access_summaries_equal(
    const loom_low_memory_access_summary_t* left,
    const loom_low_memory_access_summary_t* right);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_MEMORY_ACCESS_H_
