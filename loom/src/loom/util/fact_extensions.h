// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Optional fact payloads and type-owned domains. Extension IDs are local to
// one fact table and its current analysis scope; cross-table transfer clones
// payload ownership, while scalar fields remain in the dense value record.

#ifndef LOOM_UTIL_FACT_EXTENSIONS_H_
#define LOOM_UTIL_FACT_EXTENSIONS_H_

#include "iree/base/api.h"
#include "loom/ir/facts.h"
#include "loom/ir/ir.h"
#include "loom/util/numeric_format.h"
#include "loom/util/reference_facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_value_fact_table_t loom_value_fact_table_t;
typedef struct loom_value_fact_extension_entry_t
    loom_value_fact_extension_entry_t;
typedef struct loom_value_fact_domain_t loom_value_fact_domain_t;
typedef const loom_value_fact_domain_t* (
    *loom_value_fact_type_domain_resolver_fn_t)(
    void* user_data, const loom_fact_context_t* context,
    const loom_module_t* module, loom_type_t type);

typedef struct loom_value_fact_type_domain_resolver_callback_t {
  // Function that maps |type| to its type-owned fact domain.
  loom_value_fact_type_domain_resolver_fn_t fn;

  // Opaque callback payload passed to |fn|.
  void* user_data;
} loom_value_fact_type_domain_resolver_callback_t;

static inline loom_value_fact_type_domain_resolver_callback_t
loom_value_fact_type_domain_resolver_callback_empty(void) {
  return (loom_value_fact_type_domain_resolver_callback_t){
      /*.fn=*/NULL,
      /*.user_data=*/NULL,
  };
}

static inline loom_value_fact_type_domain_resolver_callback_t
loom_value_fact_type_domain_resolver_callback_make(
    loom_value_fact_type_domain_resolver_fn_t fn, void* user_data) {
  return (loom_value_fact_type_domain_resolver_callback_t){
      /*.fn=*/fn,
      /*.user_data=*/user_data,
  };
}

// Maximum lane facts stored in a small static vector extension. Larger static
// vectors degrade to unknown facts instead of allocating per-lane analysis
// payloads.
#define LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT 16

// Maximum raw payload bytes a type-owned fact domain may intern. Larger domain
// payloads degrade to no extension, which is the domain top/unknown value.
#define LOOM_VALUE_FACT_RAW_PAYLOAD_LENGTH_LIMIT 256

// Every logical element of an aggregate value shares the same facts. Vector
// values interpret the element as a lane; target-low register-range values
// interpret it as one register unit.
typedef struct loom_value_fact_uniform_element_t {
  // Scalar facts that apply to every lane.
  loom_value_facts_t element;
} loom_value_fact_uniform_element_t;

// Per-lane facts for a small all-static vector, in logical lane order.
typedef struct loom_value_fact_small_static_lanes_t {
  // Borrowed lane facts. Query results point into the owning fact table.
  const loom_value_facts_t* lanes;
  // Number of lane facts in the slice.
  iree_host_size_t count;
} loom_value_fact_small_static_lanes_t;

// Vector value is a lane-coordinate sequence: base + lane_ordinal * step.
typedef struct loom_value_fact_vector_iota_t {
  // Facts for the first produced coordinate.
  loom_value_facts_t base;
  // Facts for the logical lane-ordinal delta.
  loom_value_facts_t step;
  // Fixed-width signed coordinate carrier. Zero retains the mathematical
  // source-index domain instead of applying modular payload arithmetic.
  uint8_t bit_count;
} loom_value_fact_vector_iota_t;

// Vector value is a range mask produced by vector.mask.range. These operand
// facts do not establish a prefix: fixed-width coordinate arithmetic may wrap,
// and a negative step can produce a suffix instead.
typedef struct loom_value_fact_vector_prefix_mask_t {
  // Facts for the first tested coordinate.
  loom_value_facts_t lower_bound;
  // Facts for the exclusive coordinate bound.
  loom_value_facts_t upper_bound;
  // Facts for the coordinate delta between adjacent logical lanes.
  loom_value_facts_t step;
} loom_value_fact_vector_prefix_mask_t;

// Address-layout summary kind carried by encoding facts.
typedef enum loom_value_fact_address_layout_kind_e {
  // No usable address-layout summary is known.
  LOOM_VALUE_FACT_ADDRESS_LAYOUT_UNKNOWN = 0,
  // Dense row-major layout. The consuming shaped type supplies rank/extents.
  LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE = 1,
  // Explicit per-axis element strides.
  LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED = 2,
} loom_value_fact_address_layout_kind_t;

// Summary of an address-layout encoding's logical-to-physical map.
typedef struct loom_value_fact_address_layout_t {
  // Known address-layout category.
  loom_value_fact_address_layout_kind_t kind;

  // Number of stride facts in strides, at most LOOM_TYPE_MAX_RANK. Dense
  // layouts leave this zero because rank comes from the consuming shaped type.
  uint8_t rank;

  // Borrowed per-axis element-stride facts for strided layouts. Query results
  // point into the owning fact table.
  const loom_value_facts_t* strides;
} loom_value_fact_address_layout_t;

// Physical payload bit/field layout used by an encoded operand schema.
typedef enum loom_value_fact_payload_packing_bits_e {
  // No usable payload-packing fact is known.
  LOOM_VALUE_FACT_PAYLOAD_PACKING_UNKNOWN = 0,
  LOOM_VALUE_FACT_PAYLOAD_PACKING_DENSE_LANES = 1u << 0,
  LOOM_VALUE_FACT_PAYLOAD_PACKING_LITTLE_ENDIAN_NIBBLES = 1u << 1,
  LOOM_VALUE_FACT_PAYLOAD_PACKING_BIG_ENDIAN_NIBBLES = 1u << 2,
  LOOM_VALUE_FACT_PAYLOAD_PACKING_BITFIELD_STREAM = 1u << 3,
  LOOM_VALUE_FACT_PAYLOAD_PACKING_BITPLANE_STREAM = 1u << 4,
  LOOM_VALUE_FACT_PAYLOAD_PACKING_MULTI_STREAM = 1u << 5,
  LOOM_VALUE_FACT_PAYLOAD_PACKING_BASE_N_PACKED = 1u << 6,
  LOOM_VALUE_FACT_PAYLOAD_PACKING_CODEBOOK_INDICES = 1u << 7,
  LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT = 1u << 8,
  LOOM_VALUE_FACT_PAYLOAD_PACKING_INTERLEAVED_SCALE_PAYLOAD = 1u << 9,
  LOOM_VALUE_FACT_PAYLOAD_PACKING_SEPARATE_SCALE_PAYLOAD = 1u << 10,
  LOOM_VALUE_FACT_PAYLOAD_PACKING_ALL =
      (LOOM_VALUE_FACT_PAYLOAD_PACKING_DENSE_LANES |
       LOOM_VALUE_FACT_PAYLOAD_PACKING_LITTLE_ENDIAN_NIBBLES |
       LOOM_VALUE_FACT_PAYLOAD_PACKING_BIG_ENDIAN_NIBBLES |
       LOOM_VALUE_FACT_PAYLOAD_PACKING_BITFIELD_STREAM |
       LOOM_VALUE_FACT_PAYLOAD_PACKING_BITPLANE_STREAM |
       LOOM_VALUE_FACT_PAYLOAD_PACKING_MULTI_STREAM |
       LOOM_VALUE_FACT_PAYLOAD_PACKING_BASE_N_PACKED |
       LOOM_VALUE_FACT_PAYLOAD_PACKING_CODEBOOK_INDICES |
       LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT |
       LOOM_VALUE_FACT_PAYLOAD_PACKING_INTERLEAVED_SCALE_PAYLOAD |
       LOOM_VALUE_FACT_PAYLOAD_PACKING_SEPARATE_SCALE_PAYLOAD),
} loom_value_fact_payload_packing_bits_t;

typedef uint32_t loom_value_fact_payload_packing_flags_t;

// How scale-like data maps to logical elements.
typedef enum loom_value_fact_scale_topology_bits_e {
  // Zero: unknown when isolated, none when the whole scale descriptor is zero.
  LOOM_VALUE_FACT_SCALE_TOPOLOGY_UNKNOWN = 0,
  LOOM_VALUE_FACT_SCALE_TOPOLOGY_NONE = 0,
  LOOM_VALUE_FACT_SCALE_TOPOLOGY_TENSOR_GLOBAL = 1u << 0,
  LOOM_VALUE_FACT_SCALE_TOPOLOGY_ROW = 1u << 1,
  LOOM_VALUE_FACT_SCALE_TOPOLOGY_COLUMN = 1u << 2,
  LOOM_VALUE_FACT_SCALE_TOPOLOGY_CHANNEL = 1u << 3,
  LOOM_VALUE_FACT_SCALE_TOPOLOGY_GROUP_1D = 1u << 4,
  LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D = 1u << 5,
  LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_2D = 1u << 6,
  LOOM_VALUE_FACT_SCALE_TOPOLOGY_SUBBLOCK_IN_SUPERBLOCK = 1u << 7,
  LOOM_VALUE_FACT_SCALE_TOPOLOGY_HIERARCHICAL = 1u << 8,
  LOOM_VALUE_FACT_SCALE_TOPOLOGY_PER_TOKEN = 1u << 9,
  LOOM_VALUE_FACT_SCALE_TOPOLOGY_PER_HEAD = 1u << 10,
  LOOM_VALUE_FACT_SCALE_TOPOLOGY_PER_PAGE = 1u << 11,
  LOOM_VALUE_FACT_SCALE_TOPOLOGY_RUNTIME_AMAX_DERIVED = 1u << 12,
  LOOM_VALUE_FACT_SCALE_TOPOLOGY_ALL =
      (LOOM_VALUE_FACT_SCALE_TOPOLOGY_TENSOR_GLOBAL |
       LOOM_VALUE_FACT_SCALE_TOPOLOGY_ROW |
       LOOM_VALUE_FACT_SCALE_TOPOLOGY_COLUMN |
       LOOM_VALUE_FACT_SCALE_TOPOLOGY_CHANNEL |
       LOOM_VALUE_FACT_SCALE_TOPOLOGY_GROUP_1D |
       LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D |
       LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_2D |
       LOOM_VALUE_FACT_SCALE_TOPOLOGY_SUBBLOCK_IN_SUPERBLOCK |
       LOOM_VALUE_FACT_SCALE_TOPOLOGY_HIERARCHICAL |
       LOOM_VALUE_FACT_SCALE_TOPOLOGY_PER_TOKEN |
       LOOM_VALUE_FACT_SCALE_TOPOLOGY_PER_HEAD |
       LOOM_VALUE_FACT_SCALE_TOPOLOGY_PER_PAGE |
       LOOM_VALUE_FACT_SCALE_TOPOLOGY_RUNTIME_AMAX_DERIVED),
} loom_value_fact_scale_topology_bits_t;

typedef uint32_t loom_value_fact_scale_topology_flags_t;

// Affine or correction policy applied after payload decode.
typedef enum loom_value_fact_affine_policy_bits_e {
  // Zero: no affine policy.
  LOOM_VALUE_FACT_AFFINE_POLICY_UNKNOWN = 0,
  LOOM_VALUE_FACT_AFFINE_POLICY_NONE = 0,
  LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY = 1u << 0,
  LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_MIN = 1u << 1,
  LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_ZERO_POINT = 1u << 2,
  LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_BIAS = 1u << 3,
  LOOM_VALUE_FACT_AFFINE_POLICY_SUPER_SCALE_TIMES_SUBSCALE = 1u << 4,
  LOOM_VALUE_FACT_AFFINE_POLICY_SUM_CORRECTION = 1u << 5,
  LOOM_VALUE_FACT_AFFINE_POLICY_ALL =
      (LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY |
       LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_MIN |
       LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_ZERO_POINT |
       LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_BIAS |
       LOOM_VALUE_FACT_AFFINE_POLICY_SUPER_SCALE_TIMES_SUBSCALE |
       LOOM_VALUE_FACT_AFFINE_POLICY_SUM_CORRECTION),
} loom_value_fact_affine_policy_bits_t;

typedef uint32_t loom_value_fact_affine_policy_flags_t;

// Rounding/conversion policy for reference encode/decode or reencoding.
typedef enum loom_value_fact_rounding_policy_bits_e {
  // Zero: no rounding policy.
  LOOM_VALUE_FACT_ROUNDING_POLICY_UNKNOWN = 0,
  LOOM_VALUE_FACT_ROUNDING_POLICY_NONE = 0,
  LOOM_VALUE_FACT_ROUNDING_POLICY_NEAREST_EVEN = 1u << 0,
  LOOM_VALUE_FACT_ROUNDING_POLICY_NEAREST_AWAY = 1u << 1,
  LOOM_VALUE_FACT_ROUNDING_POLICY_TOWARD_ZERO = 1u << 2,
  LOOM_VALUE_FACT_ROUNDING_POLICY_DOWN = 1u << 3,
  LOOM_VALUE_FACT_ROUNDING_POLICY_UP = 1u << 4,
  LOOM_VALUE_FACT_ROUNDING_POLICY_STOCHASTIC = 1u << 5,
  LOOM_VALUE_FACT_ROUNDING_POLICY_SATFINITE = 1u << 6,
  LOOM_VALUE_FACT_ROUNDING_POLICY_OVERFLOW_TO_INF = 1u << 7,
  LOOM_VALUE_FACT_ROUNDING_POLICY_OVERFLOW_TO_NAN = 1u << 8,
  LOOM_VALUE_FACT_ROUNDING_POLICY_FLUSH_SUBNORMAL = 1u << 9,
  LOOM_VALUE_FACT_ROUNDING_POLICY_PRESERVE_SUBNORMAL = 1u << 10,
  LOOM_VALUE_FACT_ROUNDING_POLICY_RELU_CLAMP = 1u << 11,
  LOOM_VALUE_FACT_ROUNDING_POLICY_FINITE_ONLY = 1u << 12,
  LOOM_VALUE_FACT_ROUNDING_POLICY_ALL =
      (LOOM_VALUE_FACT_ROUNDING_POLICY_NEAREST_EVEN |
       LOOM_VALUE_FACT_ROUNDING_POLICY_NEAREST_AWAY |
       LOOM_VALUE_FACT_ROUNDING_POLICY_TOWARD_ZERO |
       LOOM_VALUE_FACT_ROUNDING_POLICY_DOWN |
       LOOM_VALUE_FACT_ROUNDING_POLICY_UP |
       LOOM_VALUE_FACT_ROUNDING_POLICY_STOCHASTIC |
       LOOM_VALUE_FACT_ROUNDING_POLICY_SATFINITE |
       LOOM_VALUE_FACT_ROUNDING_POLICY_OVERFLOW_TO_INF |
       LOOM_VALUE_FACT_ROUNDING_POLICY_OVERFLOW_TO_NAN |
       LOOM_VALUE_FACT_ROUNDING_POLICY_FLUSH_SUBNORMAL |
       LOOM_VALUE_FACT_ROUNDING_POLICY_PRESERVE_SUBNORMAL |
       LOOM_VALUE_FACT_ROUNDING_POLICY_RELU_CLAMP |
       LOOM_VALUE_FACT_ROUNDING_POLICY_FINITE_ONLY),
} loom_value_fact_rounding_policy_bits_t;

typedef uint32_t loom_value_fact_rounding_policy_flags_t;

// Table/codebook source policy for nonlinear encoded operands.
typedef enum loom_value_fact_codebook_policy_bits_e {
  // Zero: no codebook policy.
  LOOM_VALUE_FACT_CODEBOOK_POLICY_UNKNOWN = 0,
  LOOM_VALUE_FACT_CODEBOOK_POLICY_NONE = 0,
  LOOM_VALUE_FACT_CODEBOOK_POLICY_STATIC_BUILTIN_TABLE = 1u << 0,
  LOOM_VALUE_FACT_CODEBOOK_POLICY_STATIC_SYMBOL_TABLE = 1u << 1,
  LOOM_VALUE_FACT_CODEBOOK_POLICY_GLOBAL_DATA_TABLE = 1u << 2,
  LOOM_VALUE_FACT_CODEBOOK_POLICY_DYNAMIC_TABLE_OPERAND = 1u << 3,
  LOOM_VALUE_FACT_CODEBOOK_POLICY_PER_SUPERBLOCK_TABLE = 1u << 4,
  LOOM_VALUE_FACT_CODEBOOK_POLICY_ALL =
      (LOOM_VALUE_FACT_CODEBOOK_POLICY_STATIC_BUILTIN_TABLE |
       LOOM_VALUE_FACT_CODEBOOK_POLICY_STATIC_SYMBOL_TABLE |
       LOOM_VALUE_FACT_CODEBOOK_POLICY_GLOBAL_DATA_TABLE |
       LOOM_VALUE_FACT_CODEBOOK_POLICY_DYNAMIC_TABLE_OPERAND |
       LOOM_VALUE_FACT_CODEBOOK_POLICY_PER_SUPERBLOCK_TABLE),
} loom_value_fact_codebook_policy_bits_t;

typedef uint32_t loom_value_fact_codebook_policy_flags_t;

// Sparse metadata contract attached to an encoded operand.
typedef enum loom_value_fact_sparsity_policy_bits_e {
  // Zero: no sparsity policy.
  LOOM_VALUE_FACT_SPARSITY_POLICY_UNKNOWN = 0,
  LOOM_VALUE_FACT_SPARSITY_POLICY_NONE = 0,
  LOOM_VALUE_FACT_SPARSITY_POLICY_MASK = 1u << 0,
  LOOM_VALUE_FACT_SPARSITY_POLICY_N_M_STRUCTURED = 1u << 1,
  LOOM_VALUE_FACT_SPARSITY_POLICY_BLOCK_SPARSE = 1u << 2,
  LOOM_VALUE_FACT_SPARSITY_POLICY_BSR = 1u << 3,
  LOOM_VALUE_FACT_SPARSITY_POLICY_CSR = 1u << 4,
  LOOM_VALUE_FACT_SPARSITY_POLICY_COO = 1u << 5,
  LOOM_VALUE_FACT_SPARSITY_POLICY_PAGE_TABLE = 1u << 6,
  LOOM_VALUE_FACT_SPARSITY_POLICY_MOE_ROUTING = 1u << 7,
  LOOM_VALUE_FACT_SPARSITY_POLICY_OUTLIER_SIDE_STREAM = 1u << 8,
  LOOM_VALUE_FACT_SPARSITY_POLICY_ALL =
      (LOOM_VALUE_FACT_SPARSITY_POLICY_MASK |
       LOOM_VALUE_FACT_SPARSITY_POLICY_N_M_STRUCTURED |
       LOOM_VALUE_FACT_SPARSITY_POLICY_BLOCK_SPARSE |
       LOOM_VALUE_FACT_SPARSITY_POLICY_BSR |
       LOOM_VALUE_FACT_SPARSITY_POLICY_CSR |
       LOOM_VALUE_FACT_SPARSITY_POLICY_COO |
       LOOM_VALUE_FACT_SPARSITY_POLICY_PAGE_TABLE |
       LOOM_VALUE_FACT_SPARSITY_POLICY_MOE_ROUTING |
       LOOM_VALUE_FACT_SPARSITY_POLICY_OUTLIER_SIDE_STREAM),
} loom_value_fact_sparsity_policy_bits_t;

typedef uint32_t loom_value_fact_sparsity_policy_flags_t;

typedef enum loom_value_fact_encoded_operand_flag_bits_e {
  // A zero scale can refine to an unscaled target contract.
  LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ZERO_SCALE_FALLBACK = 1u << 0,
  // The logical element format is known to be semantic none rather than
  // unknown.
  LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ELEMENT_FORMAT_NONE = 1u << 1,
  // The affine policy is known to be semantic none rather than unknown.
  LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_AFFINE_NONE = 1u << 2,
} loom_value_fact_encoded_operand_flag_bits_t;

typedef uint32_t loom_value_fact_encoded_operand_flags_t;

// Encoding facts use the generic encoding-owned summary representation. This
// alias keeps fact-domain terminology at analysis callsites without defining a
// second storage contract.
#define LOOM_VALUE_FACT_SCALE_GROUP_MAX_RANK LOOM_ENCODING_SCALE_GROUP_MAX_RANK
typedef loom_encoding_operand_summary_t
    loom_value_fact_encoded_operand_schema_t;
static_assert(sizeof(loom_value_fact_encoded_operand_schema_t) == 72,
              "encoded operand schema must be padding-free for raw equality");

// Returns the number of exact logical scale-group dimensions. The fixed-width
// shape is canonical: positive extents precede zero-filled trailing entries.
static inline uint8_t loom_value_fact_encoded_operand_scale_group_rank(
    const loom_value_fact_encoded_operand_schema_t* schema) {
  uint8_t rank = 0;
  while (rank < LOOM_VALUE_FACT_SCALE_GROUP_MAX_RANK &&
         schema->scale_group.shape[rank] != 0) {
    ++rank;
  }
  return rank;
}

// Summary of a storage-schema encoding.
typedef struct loom_value_fact_storage_schema_t {
  // One-based static schema encoding ID when a complete static schema is known.
  // Zero means no exact nested storage schema replacement is known.
  uint16_t static_spec_encoding_id;

  // Target-independent encoded operand facts.
  loom_value_fact_encoded_operand_schema_t encoded_operand;
} loom_value_fact_storage_schema_t;

// Returns true when encoded operand schemas are byte-identical. Encoded schema
// records are fixed-width descriptors and must be zero-initialized before
// population so padding never participates in equality.
bool loom_value_fact_encoded_operand_schema_equal(
    loom_value_fact_encoded_operand_schema_t lhs,
    loom_value_fact_encoded_operand_schema_t rhs);

// Returns true when no encoded operand facts are known.
bool loom_value_fact_encoded_operand_schema_is_unknown(
    loom_value_fact_encoded_operand_schema_t schema);

// Returns true when the scale subdescriptor is nonzero. The scale
// subdescriptor includes scale formats, topology, scale counts, and scale
// flags; payload and non-scale interpretation fields are ignored.
bool loom_value_fact_encoded_operand_schema_has_scale(
    loom_value_fact_encoded_operand_schema_t schema);

// Returns true when the scale subdescriptor is either all-zero or complete
// enough to identify explicit scale topology and operands.
bool loom_value_fact_encoded_operand_schema_scale_is_complete(
    loom_value_fact_encoded_operand_schema_t schema);

// Returns true when structured sparsity density is either absent or complete
// and valid for the selected sparsity policy.
bool loom_value_fact_encoded_operand_schema_sparsity_is_complete(
    loom_value_fact_encoded_operand_schema_t schema);

// Summary of an SSA encoding value.
typedef struct loom_value_fact_encoding_summary_t {
  // Known semantic role from the value type or encoding family.
  loom_encoding_role_t role;

  // One-based static encoding spec ID when a complete static encoding is known.
  // Zero means no exact static replacement is known.
  uint16_t static_spec_encoding_id;

  // Address-layout facts when this encoding directly is a layout or composes a
  // physical storage encoding with a known layout.
  loom_value_fact_address_layout_t address_layout;

  // Storage-schema facts when this encoding directly is a schema or composes a
  // physical storage encoding with a known schema.
  loom_value_fact_storage_schema_t storage_schema;
} loom_value_fact_encoding_summary_t;

// Type-owned operations for the optional extension slot in
// loom_value_facts_t.
//
// Scalar facts stay in the dense value record and use the generic transfer
// functions in loom/ir/facts.h. A fact domain only owns interpretation of the
// cold extension payload for values whose type declares the domain. Domains are
// immutable .rodata objects referenced from type descriptors; there is no
// process-global schema registry or schema ID stored in extension entries.
//
// Returning no extension is the domain top/unknown value. Domain callbacks must
// keep payloads bounded and degrade to no extension instead of allocating
// unbounded analysis state.
struct loom_value_fact_domain_t {
  // Returns true when |lhs| and |rhs| carry semantically equal extension facts
  // for |type|. Scalar range/divisor fields are compared by the generic caller.
  bool (*extensions_equal)(const loom_value_fact_domain_t* domain,
                           const loom_module_t* module, loom_type_t type,
                           const loom_value_fact_table_t* lhs_table,
                           loom_value_facts_t lhs,
                           const loom_value_fact_table_t* rhs_table,
                           loom_value_facts_t rhs);

  // Clones the extension payload carried by |facts| from |source| into
  // |target|. The generic caller copies scalar fields before invoking this.
  iree_status_t (*clone_extension)(const loom_value_fact_domain_t* domain,
                                   const loom_module_t* module,
                                   loom_type_t type,
                                   loom_value_fact_table_t* target,
                                   const loom_value_fact_table_t* source,
                                   loom_value_facts_t facts,
                                   loom_value_facts_t* inout_facts);

  // Computes the extension component of a join for two facts of |type|. The
  // generic caller has already joined scalar fields in |inout_facts|.
  iree_status_t (*meet_extension)(const loom_value_fact_domain_t* domain,
                                  const loom_module_t* module, loom_type_t type,
                                  loom_value_fact_table_t* target,
                                  const loom_value_fact_table_t* lhs_table,
                                  loom_value_facts_t lhs,
                                  const loom_value_fact_table_t* rhs_table,
                                  loom_value_facts_t rhs,
                                  loom_value_facts_t* inout_facts);

  // Widens a loop-carried extension fact. The default domain-free behavior is
  // conservative: stable extensions are preserved, changing extensions degrade
  // to no extension. |iteration| is zero-based for the loop summary solver.
  iree_status_t (*widen_extension)(
      const loom_value_fact_domain_t* domain, const loom_module_t* module,
      loom_type_t type, loom_value_fact_table_t* target,
      const loom_value_fact_table_t* previous_table,
      loom_value_facts_t previous, const loom_value_fact_table_t* next_table,
      loom_value_facts_t next, uint32_t iteration,
      loom_value_facts_t* inout_facts);
};

//===----------------------------------------------------------------------===//
// Cross-table transfers
//===----------------------------------------------------------------------===//

// Clones |facts| from |source| into |target|, re-interning any context-local
// extension payloads in the target table. The returned facts are valid for
// |target|'s fact context and preserve scalar range/divisibility fields.
// Cloning within one table retains its immutable payloads without allocation.
iree_status_t loom_value_fact_table_clone_fact(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    loom_value_facts_t facts, loom_value_facts_t* out_facts);

// Clones |facts| using the fact domain implied by |type|. Prefer this typed
// form when the value/type is available; extension payloads are type-owned.
// Cloning within one table retains its immutable payloads without allocation.
iree_status_t loom_value_fact_table_clone_fact_for_type(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    const loom_module_t* module, loom_type_t type, loom_value_facts_t facts,
    loom_value_facts_t* out_facts);

// Returns true when two fact values are semantically equal, including any
// extension payloads, even when the extension IDs were interned in different
// fact tables.
bool loom_value_fact_table_facts_equal(const loom_value_fact_table_t* lhs_table,
                                       loom_value_facts_t lhs,
                                       const loom_value_fact_table_t* rhs_table,
                                       loom_value_facts_t rhs);

// Typed semantic equality for facts whose values have |type|.
bool loom_value_fact_table_facts_equal_for_type(
    const loom_module_t* module, loom_type_t type,
    const loom_value_fact_table_t* lhs_table, loom_value_facts_t lhs,
    const loom_value_fact_table_t* rhs_table, loom_value_facts_t rhs);

// Returns true when both facts carry semantically equal extension payloads.
// Facts without extensions compare equal only when neither side has an
// extension.
bool loom_value_fact_table_extensions_equal(
    const loom_value_fact_table_t* lhs_table, loom_value_facts_t lhs,
    const loom_value_fact_table_t* rhs_table, loom_value_facts_t rhs);

// Typed semantic equality for extension payloads whose values have |type|.
bool loom_value_fact_table_extensions_equal_for_type(
    const loom_module_t* module, loom_type_t type,
    const loom_value_fact_table_t* lhs_table, loom_value_facts_t lhs,
    const loom_value_fact_table_t* rhs_table, loom_value_facts_t rhs);

// Computes a conservative join for two facts of |type| and stores the result in
// |target|. Extension payloads are preserved only when the type's fact domain
// proves they remain valid across the join.
iree_status_t loom_value_fact_table_meet_for_type(
    loom_value_fact_table_t* target, const loom_module_t* module,
    loom_type_t type, const loom_value_fact_table_t* lhs_table,
    loom_value_facts_t lhs, const loom_value_fact_table_t* rhs_table,
    loom_value_facts_t rhs, loom_value_facts_t* out_facts);

// Widens a loop-carried fact of |type| from |previous| to |next|. This is a
// convergence accelerator for summary solvers, not a scalar transfer function.
// Expanding integer endpoints stop at the signed or nonnegative 32-bit carrier
// boundary before reaching their scalar type domain. Stable endpoints remain
// bounded independently of other changing facts.
iree_status_t loom_value_fact_table_widen_for_type(
    loom_value_fact_table_t* target, const loom_module_t* module,
    loom_type_t type, const loom_value_fact_table_t* previous_table,
    loom_value_facts_t previous, const loom_value_fact_table_t* next_table,
    loom_value_facts_t next, uint32_t iteration, loom_value_facts_t* out_facts);

//===----------------------------------------------------------------------===//
// Fact extensions
//===----------------------------------------------------------------------===//

// Creates facts for a vector whose every lane has |element| facts. Float
// predicates and exact destination formats common to every lane are also
// available directly on the vector. Execution distribution remains separate.
iree_status_t loom_value_facts_make_uniform_element(
    loom_fact_context_t* context, loom_value_facts_t element,
    loom_value_facts_t* out);

// Returns true and populates |out| when |facts| is a uniform-element vector
// extension in |context|.
bool loom_value_facts_query_uniform_element(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_uniform_element_t* out);

// Creates facts for a small all-static vector with explicit per-lane facts.
// Float predicates and exact destination formats common to every lane are also
// available directly on the vector. Vectors with more than
// LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT lanes degrade to unknown facts.
iree_status_t loom_value_facts_make_small_static_lanes(
    loom_fact_context_t* context, loom_value_fact_small_static_lanes_t lanes,
    loom_value_facts_t* out);

// Returns true and populates |out| when |facts| is a small-static-lanes vector
// extension in |context|.
bool loom_value_facts_query_small_static_lanes(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_small_static_lanes_t* out);

// Returns true when |facts| represents a scalar value or aggregate whose every
// logical element has the same facts. Unknown scalar facts are not reported
// because carrying a uniform-unknown extension adds no useful information.
// |out_element| is required and receives the shared element facts.
bool loom_value_facts_query_all_equal_element(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_facts_t* out_element);

// Creates facts for a vector.iota-style lane-coordinate sequence.
iree_status_t loom_value_facts_make_vector_iota(
    loom_fact_context_t* context, loom_value_fact_vector_iota_t iota,
    loom_value_facts_t* out);

// Returns true and populates |out| when |facts| is a vector.iota extension in
// |context|.
bool loom_value_facts_query_vector_iota(const loom_fact_context_t* context,
                                        loom_value_facts_t facts,
                                        loom_value_fact_vector_iota_t* out);

// Evaluates one coordinate using the sequence's arithmetic domain. Unknown
// operands or an unrepresentable mathematical index produce unknown facts.
// Fixed-width coordinates wrap modulo their declared width, including i64.
loom_value_facts_t loom_value_fact_vector_iota_element(
    loom_value_fact_vector_iota_t iota, uint64_t lane_ordinal);

// Bounds all coordinates without signed wrap. The lane count is an exact count
// or an upper bound; UINT64_MAX is unknown. The empty interval is [0, -1].
// Returns false when the mathematical endpoints cannot be represented in the
// coordinate domain. This is not a bound on a wrapping sequence.
bool loom_value_fact_vector_iota_bounds(loom_value_fact_vector_iota_t iota,
                                        uint64_t maximum_lane_count,
                                        int64_t* out_lower, int64_t* out_upper);

// Returns inclusive integer bounds across every logical lane represented by a
// uniform-element, small-static-lanes, or vector-iota extension. Iota bounds
// use |maximum_lane_count| as either the exact static lane count or a proven
// upper bound; UINT64_MAX means the lane count is unavailable. Empty explicit
// lane lists and zero-count iotas report the canonical empty interval [0, -1].
// Returns false when any lane is floating point or unbounded, an iota may wrap,
// or |facts| has another extension kind.
bool loom_value_facts_query_vector_integer_bounds(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    uint64_t maximum_lane_count, int64_t* out_lower, int64_t* out_upper);

// Creates facts for a vector.mask.range-style prefix mask.
iree_status_t loom_value_facts_make_vector_prefix_mask(
    loom_fact_context_t* context, loom_value_fact_vector_prefix_mask_t mask,
    loom_value_facts_t* out);

// Returns true and populates |out| when |facts| is a vector.mask.range
// extension in |context|.
bool loom_value_facts_query_vector_prefix_mask(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_vector_prefix_mask_t* out);

// Creates facts for an SSA encoding summary.
iree_status_t loom_value_facts_make_encoding_summary(
    loom_fact_context_t* context, loom_value_fact_encoding_summary_t summary,
    loom_value_facts_t* out);

// Returns true and populates |out| when |facts| is an encoding-summary
// extension in |context|.
bool loom_value_facts_query_encoding_summary(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_encoding_summary_t* out);

// Creates facts for a buffer storage root.
iree_status_t loom_value_facts_make_buffer_reference(
    loom_fact_context_t* context, loom_value_fact_buffer_reference_t reference,
    loom_value_facts_t* out);

// Returns true and populates |out| when |facts| is a buffer-reference
// extension in |context|.
bool loom_value_facts_query_buffer_reference(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_buffer_reference_t* out);

// Creates facts for a typed view projection.
iree_status_t loom_value_facts_make_view_reference(
    loom_fact_context_t* context, loom_value_fact_view_reference_t reference,
    loom_value_facts_t* out);

// Returns true and populates |out| when |facts| is a view-reference extension
// in |context|.
bool loom_value_facts_query_view_reference(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_view_reference_t* out);

// Creates a bounded raw extension payload for type-owned fact domains. The
// |payload_tag| namespace belongs to the value type domain; it is not a global
// schema ID. Payloads larger than LOOM_VALUE_FACT_RAW_PAYLOAD_LENGTH_LIMIT
// degrade to unknown facts.
iree_status_t loom_value_facts_make_extension_payload(
    loom_fact_context_t* context, uint8_t payload_tag, const void* payload,
    iree_host_size_t payload_length, loom_value_facts_t* out);

// Returns true and populates |out_payload|/|out_payload_length| when |facts|
// carries a raw payload with |payload_tag|. Returned bytes are borrowed from
// the fact table that owns |context| and remain valid for the table lifetime.
bool loom_value_facts_query_extension_payload(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    uint8_t payload_tag, const void** out_payload,
    iree_host_size_t* out_payload_length);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_UTIL_FACT_EXTENSIONS_H_
