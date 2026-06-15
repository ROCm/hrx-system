// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-low source memory access planning.
//
// This layer owns the target-independent half of vector memory lowering:
// decomposing a typed view/vector access plus value facts into a compact source
// plan with arena-compatible lifetime. Targets wrap this plan with their own
// addressing modes, descriptor choices, register classes, and machine-specific
// fallback decisions.

#ifndef LOOM_CODEGEN_LOW_SOURCE_MEMORY_PLAN_H_
#define LOOM_CODEGEN_LOW_SOURCE_MEMORY_PLAN_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/codegen/low/memory_access.h"
#include "loom/ir/attribute.h"
#include "loom/ir/ir.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE UINT32_MAX

typedef enum loom_low_source_memory_operation_kind_e {
  LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD = 0,
  LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE = 1,
  LOOM_LOW_SOURCE_MEMORY_OPERATION_PREFETCH = 2,
  LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_REDUCE = 3,
  LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW = 4,
  LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_CMPXCHG = 5,
} loom_low_source_memory_operation_kind_t;

typedef enum loom_low_source_memory_dynamic_index_source_e {
  // The access has no dynamic index.
  LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_NONE = 0,
  // The dynamic index is an arbitrary SSA value.
  LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE = 1,
  // The dynamic index is produced by kernel.workitem.id.
  LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID = 2,
  // The dynamic index is produced by kernel.workgroup.id.
  LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID = 3,
} loom_low_source_memory_dynamic_index_source_t;

typedef uint32_t loom_low_source_memory_access_rejection_flags_t;

#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_UNSUPPORTED_OP \
  ((uint32_t)1u << 0)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DESCRIBE_FAILED \
  ((uint32_t)1u << 1)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_LAYOUT ((uint32_t)1u << 2)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_ELEMENT_WIDTH \
  ((uint32_t)1u << 3)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_RANK ((uint32_t)1u << 4)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_LANE_COUNT \
  ((uint32_t)1u << 5)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE \
  ((uint32_t)1u << 6)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_STATIC_OFFSET \
  ((uint32_t)1u << 7)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_COUNT \
  ((uint32_t)1u << 8)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_AXIS ((uint32_t)1u << 9)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE \
  ((uint32_t)1u << 10)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_SOURCE ((uint32_t)1u << 11)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_BASE ((uint32_t)1u << 12)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_BASE_OVERFLOW \
  ((uint32_t)1u << 13)
#define LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_CACHE_POLICY \
  ((uint32_t)1u << 14)

typedef struct loom_low_source_memory_access_diagnostic_t {
  // Bitset of source-access rejection reasons observed while planning.
  loom_low_source_memory_access_rejection_flags_t rejection_bits;
} loom_low_source_memory_access_diagnostic_t;

#define LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_AXIS_NONE UINT8_MAX
#define LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY (LOOM_TYPE_MAX_RANK + 1)

typedef struct loom_low_source_memory_dynamic_term_t {
  // Dynamic source SSA value multiplied into this address term.
  loom_value_id_t index;
  // Dynamic dimension SSA values multiplied into this term's byte stride.
  loom_value_id_t stride_values[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY];
  // Number of populated entries in stride_values.
  uint8_t stride_value_count;
  // Source provenance for the dynamic index value.
  loom_low_source_memory_dynamic_index_source_t source;
  // Workitem/workgroup dimension when |source| is a coordinate.
  loom_kernel_dimension_t dimension;
  // View axis represented by |index|, or AXIS_NONE for a dynamic view base or
  // a recovered term from an already-linearized index expression.
  uint8_t axis;
  // Static byte stride multiplied by |index| and each stride value before
  // adding the term to the address.
  int64_t byte_stride;
  // Facts for |index| multiplied by all stride factors and |byte_stride|, in
  // bytes.
  loom_value_facts_t byte_facts;
  // Power-of-two shift used to compute the static byte stride, or
  // LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE when multiplication or
  // target-specific handling is required. Dynamic stride values are still
  // multiplied separately.
  uint32_t byte_shift;
} loom_low_source_memory_dynamic_term_t;

typedef struct loom_low_source_memory_access_plan_t {
  // Source operation category being planned.
  loom_low_source_memory_operation_kind_t operation_kind;
  // Source view SSA value consumed by the memory operation.
  loom_value_id_t view_value_id;
  // Target-independent memory space selected from source view facts.
  loom_value_fact_memory_space_t memory_space;
  // Source SSA value that represents the storage root.
  loom_value_id_t root_value_id;
  // Minimum provable byte alignment of the storage root base address.
  uint32_t root_minimum_alignment;
  // Comparable alias scope for disjointness proofs, or NONE.
  loom_value_fact_alias_scope_id_t alias_scope_id;
  // Byte count of one addressed view element.
  uint32_t element_byte_count;
  // Static number of vector lanes addressed by the operation.
  uint32_t vector_lane_count;
  // Byte stride between adjacent vector lanes along the vector axis.
  int64_t vector_lane_byte_stride;
  // Total static byte offset selected from the source view access.
  int64_t static_byte_offset;
  // Static byte offset contributed by the source view base.
  int64_t static_view_base_byte_offset;
  // Minimum provable byte alignment of the final accessed address.
  uint32_t minimum_alignment;
  // Dynamic address terms. The first |dynamic_view_base_term_count| entries
  // belong to the source view base; remaining entries describe indexed access.
  loom_low_source_memory_dynamic_term_t
      dynamic_terms[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY];
  // Number of populated dynamic address terms.
  uint8_t dynamic_term_count;
  // Number of leading dynamic address terms contributed by the source view
  // base.
  uint8_t dynamic_view_base_term_count;
  // Optional cache policy copied from the source memory op.
  loom_vector_memory_cache_policy_t cache_policy;
} loom_low_source_memory_access_plan_t;

static inline bool loom_low_source_memory_access_is_dynamic(
    const loom_low_source_memory_access_plan_t* plan) {
  return plan->dynamic_term_count != 0;
}

static inline const loom_low_source_memory_dynamic_term_t*
loom_low_source_memory_access_single_dynamic_term(
    const loom_low_source_memory_access_plan_t* plan) {
  return plan->dynamic_term_count == 1 ? &plan->dynamic_terms[0] : NULL;
}

// Returns true when |value_id| names a block argument in |module|. Targets use
// this to recognize source roots already materialized by ABI lowering.
static inline bool loom_low_source_memory_value_is_block_argument(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return false;
  }
  return loom_value_is_block_arg(loom_module_value(module, value_id));
}

// Returns true when |term|'s dynamic byte offset is proven to fit in an
// unsigned integer with |bit_count| bits.
static inline bool loom_low_source_memory_dynamic_term_fits_unsigned_bit_count(
    const loom_low_source_memory_dynamic_term_t* term, uint8_t bit_count) {
  return loom_value_facts_fit_unsigned_bit_count(term->byte_facts, bit_count);
}

// Returns true when the sum of all dynamic byte terms plus
// |static_byte_offset| is proven to fit in an unsigned integer with
// |bit_count| bits.
bool loom_low_source_memory_dynamic_offset_fits_unsigned_bit_count(
    const loom_low_source_memory_access_plan_t* plan,
    int64_t static_byte_offset, uint8_t bit_count);

// Builds a dependency/scheduling summary from an already selected source
// memory access plan. |out_interval| is caller-owned and may be borrowed by the
// returned summary when interval precision is available.
void loom_low_source_memory_access_plan_make_summary(
    const loom_low_source_memory_access_plan_t* plan,
    loom_low_byte_interval_t* out_interval,
    loom_low_memory_access_summary_t* out_summary);

// Builds a target-independent source memory plan for indexed source memory ops.
//
// Returns false when the source op cannot be decomposed into a complete source
// plan. Targets are responsible for checking their own memory spaces, address
// forms, descriptor availability, immediate ranges, and register footprints.
bool loom_low_source_memory_access_plan_build(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* source_op, loom_low_source_memory_access_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* out_diagnostic);

// Builds a target-independent source memory plan for a whole typed view.
//
// This is the view-payload sibling of vector.load/store planning. It treats the
// full static footprint of |view_value_id| as the transferred vector payload
// and preserves a single materializable dynamic subview offset when the view is
// immediately produced by view.subview. Targets use this for memory-to-memory
// movement ops such as async global-to-workgroup gathers, where the source IR
// names a view projection rather than an indexed vector load.
bool loom_low_source_memory_access_plan_build_view(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_value_id_t view_value_id,
    loom_vector_memory_cache_policy_t cache_policy,
    loom_low_source_memory_access_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* out_diagnostic);

// Returns a stable diagnostic constraint key for source memory access rejection
// flags.
iree_string_view_t loom_low_source_memory_access_rejection_key(
    loom_low_source_memory_access_rejection_flags_t rejection_bits);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SOURCE_MEMORY_PLAN_H_
