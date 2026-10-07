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
#include "loom/analysis/symbolic_expr.h"
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

typedef enum loom_low_memory_access_source_flag_bits_e {
  // Source atomic load/store lowered through an ordinary target load/store.
  LOOM_LOW_MEMORY_ACCESS_SOURCE_FLAG_ATOMIC_OBSERVATION = 1u << 0,
} loom_low_memory_access_source_flag_bits_t;
typedef uint8_t loom_low_memory_access_source_flags_t;

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

// Conservative byte envelope [origin+lower, origin+upper) relative to one
// storage value. The captured namespace owns symbolic variable identities;
// they are not live IR value handles. Unequal storage values may still alias.
typedef struct loom_low_memory_relative_interval_t {
  // Identity of the producer's captured evaluation namespace.
  const void* scope;
  // Storage equality identity within scope; inequality proves nothing.
  uint32_t storage_id;
  // One-based disjoint-storage identity within scope, or zero when unknown.
  // Unequal nonzero identities require a comparison mode that establishes one
  // captured dynamic invocation without CFG re-entry.
  uint32_t disjoint_storage_ordinal;
  // Participant-uniform symbolic origin, with optional periodic guarantees.
  loom_symbolic_expr_t origin;
  // Inclusive lower displacement over all participants and packet elements.
  int64_t lower;
  // Exclusive upper displacement over all participants and packet elements.
  int64_t upper;
} loom_low_memory_relative_interval_t;

typedef enum loom_low_memory_comparison_e {
  // Values may come from different dynamic executions, including backedges.
  LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT,
  // Both effects execute in the same block invocation and participant domain.
  LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION,
  // Both effects execute in the same source-function invocation and participant
  // domain, and control flow between them crosses no CFG re-entry/backedge.
  LOOM_LOW_MEMORY_COMPARISON_SAME_ACYCLIC_INVOCATION,
} loom_low_memory_comparison_t;

typedef struct loom_low_memory_access_summary_t {
  // Normalized target-low memory space touched by this summary.
  uint8_t memory_space;
  // Source operation semantics retained across target-low lowering.
  loom_low_memory_access_source_flags_t source_flags;
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
  // Optional captured relative footprint, independent of numeric alias labels.
  const loom_low_memory_relative_interval_t* relative_interval;
} loom_low_memory_access_summary_t;

static_assert(LOOM_LOW_MEMORY_SPACE_WASM_MEMORY <= UINT8_MAX,
              "memory spaces must fit compact access summaries");
static_assert(sizeof(loom_low_memory_access_summary_t) ==
                  40 + 2 * sizeof(void*),
              "memory access summaries must remain compact");

// Compiler-owned packet/effect bindings. The arena and operations outlive the
// map; it contains no IR attributes and never infers an address from Low IR.
typedef struct loom_low_memory_access_map_t loom_low_memory_access_map_t;

// Creates an initially empty map in |arena|.
iree_status_t loom_low_memory_access_map_create(
    iree_arena_allocator_t* arena, loom_low_memory_access_map_t** out_map);

// Retains one concrete descriptor effect and deep-copies its proof payload.
// The producer supplies the descriptor-local ordinal, never a memory-space
// guess. Returned effect summaries have stable addresses for the map lifetime.
iree_status_t loom_low_memory_access_map_insert(
    loom_low_memory_access_map_t* map, const loom_op_t* op,
    uint16_t effect_ordinal, const loom_low_memory_access_summary_t* summary);

// Returns the binding for this exact effect, or NULL for an unrefined effect.
const loom_low_memory_access_summary_t* loom_low_memory_access_map_lookup(
    const loom_low_memory_access_map_t* map, const loom_op_t* op,
    uint16_t effect_ordinal);

// Transfers preserved effects at an instruction replacement boundary. The
// replacement must preserve the meaning and ordinals of descriptor effects.
iree_status_t loom_low_memory_access_map_replace(
    loom_low_memory_access_map_t* map, const loom_op_t* old_op,
    const loom_op_t* new_op);

// Transfers bindings for operations moved within the same module. Both maps
// use the module arena; immutable proof payloads retain their captured scopes.
iree_status_t loom_low_memory_access_map_transfer(
    const loom_low_memory_access_map_t* source,
    loom_low_memory_access_map_t* target);

// One invocation of a clone, with freshly mapped captured evaluation scopes.
// The source map is immutable; target bindings own all copied payloads.
typedef struct loom_low_memory_access_clone_t loom_low_memory_access_clone_t;

iree_status_t loom_low_memory_access_clone_create(
    const loom_low_memory_access_map_t* source,
    loom_low_memory_access_map_t* target, iree_arena_allocator_t* scratch_arena,
    loom_low_memory_access_clone_t** out_clone);

// Matches the generic operation-clone observer signature.
iree_status_t loom_low_memory_access_clone_op(void* clone,
                                              const loom_op_t* source_op,
                                              loom_op_t* target_op);

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
    const loom_low_memory_access_summary_t* right,
    loom_low_memory_comparison_t comparison);

// Returns true when the summaries have identical conservative alias facts.
// Source flags do not participate. This proves equivalent may-alias queries,
// not identical runtime addresses or full overwrite. Retiring an access
// additionally requires an established completion dependency through an
// intervening opposite-kind access.
bool loom_low_memory_access_summaries_equal(
    const loom_low_memory_access_summary_t* left,
    const loom_low_memory_access_summary_t* right);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_MEMORY_ACCESS_H_
