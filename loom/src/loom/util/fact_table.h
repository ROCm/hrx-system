// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Per-value fact table: dense array of loom_value_facts_t keyed by
// loom_value_id_t. Arena-allocated. A zero-initialized table is valid (empty).
//
// Lookup always succeeds: returns unknown facts for undefined entries
// or out-of-range value IDs. Undefined entries are detected by
// known_divisor == 0 (valid facts always have known_divisor >= 1),
// allowing O(1) initialization via memset(0).
//
// Define stores facts for a value ID, growing the dense value-entry array as
// needed. Compute runs a forward pass over an explicit region tree, calling
// each op's fact inference function to seed initial facts from constants and op
// semantics.
//
// The table is a reusable component: borrowed by the rewriter for
// canonicalization, owned by pass-scoped storage, and usable standalone for IPO
// or analysis tools.
//
// Typical lifecycle:
//
//   loom_value_fact_table_t table = {0};
//   IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(
//       &table, arena, value_count));
//   IREE_RETURN_IF_ERROR(loom_value_fact_table_compute(
//       &table, module, function));
//   loom_value_facts_t facts = loom_value_fact_table_lookup(&table, id);

#ifndef LOOM_UTIL_FACT_TABLE_H_
#define LOOM_UTIL_FACT_TABLE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/facts.h"
#include "loom/ir/ir.h"
#include "loom/util/numeric_format.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Fact table
//===----------------------------------------------------------------------===//

typedef struct loom_value_fact_table_t loom_value_fact_table_t;
typedef struct loom_value_fact_extension_entry_t
    loom_value_fact_extension_entry_t;
typedef struct loom_value_fact_domain_t loom_value_fact_domain_t;
typedef struct loom_target_bundle_t loom_target_bundle_t;

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
} loom_value_fact_vector_iota_t;

// Vector value is a prefix mask produced by vector.mask.range.
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

  // Number of stride facts in strides for strided layouts. Dense layouts leave
  // this zero because rank comes from the consuming shaped type.
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
} loom_value_fact_encoded_operand_flag_bits_t;

typedef uint32_t loom_value_fact_encoded_operand_flags_t;

// Target-independent encoded operand facts for storage schemas or prepared
// matrix/vector fragments. Bulk payload, scale, table, and sparse metadata data
// remain SSA values; this summary only records compact interpretation facts.
typedef struct loom_value_fact_encoded_operand_schema_t {
  // Logical element format after payload interpretation.
  loom_value_fact_numeric_format_flags_t element_format;

  // Primary/local scale format.
  loom_value_fact_numeric_format_flags_t scale_format;

  // Secondary/global/super scale format for hierarchical schemes.
  loom_value_fact_numeric_format_flags_t secondary_scale_format;

  // Physical payload packing or target-fragment arrangement.
  loom_value_fact_payload_packing_flags_t payload_packing;

  // How scale values are broadcast over logical elements.
  loom_value_fact_scale_topology_flags_t scale_topology;

  // Affine, offset, or correction contract.
  loom_value_fact_affine_policy_flags_t affine_policy;

  // Rounding or finite-policy contract.
  loom_value_fact_rounding_policy_flags_t rounding_policy;

  // Codebook/table ownership contract.
  loom_value_fact_codebook_policy_flags_t codebook_policy;

  // Sparse metadata contract.
  loom_value_fact_sparsity_policy_flags_t sparsity_policy;

  // Bitset of loom_value_fact_encoded_operand_flag_bits_t values.
  loom_value_fact_encoded_operand_flags_t flags;

  // Reserved zero bits keep the descriptor padding-free for raw equality.
  uint32_t reserved;

  // Number of 32-bit payload registers in a prepared fragment, or zero when
  // not target-fragment-shaped.
  uint16_t payload_register_count;

  // Number of logical elements represented by the payload.
  uint16_t payload_element_count;

  // Number of logical elements covered by one primary/local scale value.
  uint16_t scale_group_element_count;

  // Number of explicit scale-like SSA operands required by this schema.
  uint16_t scale_operand_count;
} loom_value_fact_encoded_operand_schema_t;
static_assert(sizeof(loom_value_fact_encoded_operand_schema_t) == 64,
              "encoded operand schema must be padding-free for raw equality");

// Summary of a storage-schema encoding.
typedef struct loom_value_fact_storage_schema_t {
  // One-based static schema encoding ID when known. Zero means no exact nested
  // storage schema is known.
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

// Summary of an SSA encoding value.
typedef struct loom_value_fact_encoding_summary_t {
  // Known semantic role from the value type or encoding family.
  loom_encoding_role_t role;

  // One-based static encoding spec ID when the value came from
  // encoding.define. Zero means no exact static spec is known.
  uint16_t static_spec_encoding_id;

  // Address-layout facts when this encoding directly is a layout or composes a
  // physical storage encoding with a known layout.
  loom_value_fact_address_layout_t address_layout;

  // Storage-schema facts when this encoding directly is a schema or composes a
  // physical storage encoding with a known schema.
  loom_value_fact_storage_schema_t storage_schema;
} loom_value_fact_encoding_summary_t;

// Known reference nullability for storage-like values.
typedef uint32_t loom_value_fact_reference_nullability_t;
#define LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN \
  ((loom_value_fact_reference_nullability_t)0)
#define LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NULL \
  ((loom_value_fact_reference_nullability_t)1)
#define LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NON_NULL \
  ((loom_value_fact_reference_nullability_t)2)

// Comparable alias scope for storage-like values. NONE means root_value_id is
// only provenance for addressing and same-root propagation; consumers must not
// use it to prove disjointness against another root.
typedef loom_value_id_t loom_value_fact_alias_scope_id_t;
#define LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE \
  ((loom_value_fact_alias_scope_id_t)LOOM_VALUE_ID_INVALID)

// Buffer value is an opaque storage root.
typedef struct loom_value_fact_buffer_reference_t {
  // Conservative byte extent facts for the root storage allocation.
  loom_value_facts_t maximum_byte_extent;

  // Minimum provable byte alignment of the root storage base. One means
  // unknown beyond byte alignment.
  uint64_t minimum_alignment;

  // Target-independent memory space for the storage root.
  loom_value_fact_memory_space_t memory_space;

  // SSA value that represents the root storage identity.
  loom_value_id_t root_value_id;

  // Comparable alias scope for disjointness proofs, or NONE.
  loom_value_fact_alias_scope_id_t alias_scope_id;

  // Known nullability for the storage root.
  loom_value_fact_reference_nullability_t nullability;
} loom_value_fact_buffer_reference_t;

// View value is a typed projection over a storage root.
typedef struct loom_value_fact_view_reference_t {
  // Byte offset facts for the view base relative to root_value_id.
  loom_value_facts_t base_byte_offset;

  // Conservative byte length facts for the whole-view footprint envelope.
  loom_value_facts_t footprint_byte_length;

  // Minimum provable alignment of base_byte_offset relative to root_value_id.
  // The root's own absolute pointer alignment is tracked separately.
  uint64_t minimum_alignment;

  // Minimum provable byte alignment of the root storage base. One means
  // unknown beyond byte alignment.
  uint64_t root_minimum_alignment;

  // Static addressed element byte count, or -1 for sub-byte/unknown elements.
  int64_t static_element_byte_count;

  // Target-independent memory space for the underlying storage root.
  loom_value_fact_memory_space_t memory_space;

  // SSA value that represents the root storage identity.
  loom_value_id_t root_value_id;

  // Comparable alias scope for disjointness proofs, or NONE.
  loom_value_fact_alias_scope_id_t alias_scope_id;

  // Known nullability for the underlying storage root.
  loom_value_fact_reference_nullability_t nullability;
} loom_value_fact_view_reference_t;

// Per-analysis context passed to op fact inference callbacks.
struct loom_fact_context_t {
  // Table that owns the dense facts and any extension payloads allocated by
  // inference helpers.
  loom_value_fact_table_t* table;

  // Function-like op whose projected region is currently being analyzed. Empty
  // when facts are computed for individual detached ops instead of a full
  // function-like projection.
  loom_func_like_t function;

  // Optional selected target bundle for target-sensitive fact inference.
  // Generic analyses leave this NULL and receive source-level facts.
  const loom_target_bundle_t* target_bundle;

  // Optional type-domain resolver installed by layers that own registered type
  // descriptors. The fact table itself intentionally does not depend on the
  // generated type registry; callers that can map |type| to a descriptor can
  // return descriptor->fact_domain here.
  loom_value_fact_type_domain_resolver_callback_t resolve_type_domain;
};

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

struct loom_value_fact_table_t {
  // Arena for direct-address entries and touched-value storage.
  iree_arena_allocator_t* arena;
  // Arena for scope-local extension payloads and inference scratch buffers.
  iree_arena_allocator_t* transient_arena;

  // Dense fact entries indexed by value ID.
  loom_value_facts_t* entries;
  // Highest defined value ID plus one.
  iree_host_size_t count;
  // Allocated entry count.
  iree_host_size_t capacity;
  // Value IDs defined in the current populated scope.
  loom_value_id_t* touched_values;
  // Number of populated entries in touched_values.
  iree_host_size_t touched_count;
  // Allocated touched_values entry count.
  iree_host_size_t touched_capacity;
  // Context object passed to op-specific fact inference callbacks.
  loom_fact_context_t context;

  // Interned fact extension payloads. Extension IDs stored in
  // loom_value_facts_t are one-based indexes into entries and are only valid
  // for this table/context.
  struct {
    // Extension entries indexed by one-based extension ID minus one.
    loom_value_fact_extension_entry_t* entries;
    // Allocated extension entry count.
    iree_host_size_t capacity;
    // Defined extension entry count.
    iree_host_size_t count;
    // Hash buckets storing one-based extension IDs, or zero for empty buckets.
    loom_value_fact_extension_id_t* buckets;
    // Allocated hash bucket count.
    iree_host_size_t bucket_count;
  } extensions;

  // Reusable scratch buffers for fact inference calls. Allocated on first use,
  // grown only when an op needs more slots. Never shrinks. Old buffers are
  // abandoned in the arena and freed in bulk with the arena.
  struct {
    // Fact scratch buffer state.
    struct {
      // Scratch fact values used for operand and result fact arrays.
      loom_value_facts_t* values;
      // Allocated fact scratch entry count.
      iree_host_size_t capacity;
    } facts;
    // Value ID scratch buffer state.
    struct {
      // Scratch value IDs used by transforms that materialize replacements.
      loom_value_id_t* values;
      // Allocated value ID scratch entry count.
      iree_host_size_t capacity;
    } value_ids;
  } scratch;
};

// Initializes the table with the given arena and pre-allocates
// |initial_capacity| value entries. All entries are zero-initialized
// (known_divisor == 0 means undefined). The arena is stored and used for
// subsequent value-entry growth, extension payloads, and scratch buffers.
iree_status_t loom_value_fact_table_initialize(
    loom_value_fact_table_t* table, iree_arena_allocator_t* arena,
    iree_host_size_t initial_capacity);

// Initializes the table using separate arenas for persistent direct-address
// entries and transient scope-local payloads. Use this when reusing one table
// across multiple populated scopes: clear touched entries, reset the transient
// arena, and keep the direct entry array live.
iree_status_t loom_value_fact_table_initialize_with_arenas(
    loom_value_fact_table_t* table, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* transient_arena, iree_host_size_t initial_capacity);

// Clears facts populated in the current scope and forgets transient extension
// and scratch state. Callers that provided a separate transient arena should
// reset that arena after this call. Direct-address entry storage and touched
// storage remain allocated for reuse.
void loom_value_fact_table_clear_scope(loom_value_fact_table_t* table);

// Returns true when |value_id| has explicitly defined facts in |table|.
static inline bool loom_value_fact_table_has_entry(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  return value_id < table->capacity &&
         table->entries[value_id].known_divisor != 0;
}

// Looks up facts for a value. Returns unknown facts if the value ID is out of
// range or the entry is undefined.
static inline loom_value_facts_t loom_value_fact_table_lookup(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  if (!loom_value_fact_table_has_entry(table, value_id)) {
    return loom_value_facts_unknown();
  }
  return table->entries[value_id];
}

// Defines (or updates) facts for a value, growing the table if needed.
iree_status_t loom_value_fact_table_define(loom_value_fact_table_t* table,
                                           loom_value_id_t value_id,
                                           loom_value_facts_t facts);

// Clones |facts| from |source| into |target|, re-interning any context-local
// extension payloads in the target table. The returned facts are valid for
// |target|'s fact context and preserve scalar range/divisibility fields.
iree_status_t loom_value_fact_table_clone_fact(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    loom_value_facts_t facts, loom_value_facts_t* out_facts);

// Clones |facts| using the fact domain implied by |type|. Prefer this typed
// form when the value/type is available; extension payloads are type-owned.
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
iree_status_t loom_value_fact_table_widen_for_type(
    loom_value_fact_table_t* target, const loom_module_t* module,
    loom_type_t type, const loom_value_fact_table_t* previous_table,
    loom_value_facts_t previous, const loom_value_fact_table_t* next_table,
    loom_value_facts_t next, uint32_t iteration, loom_value_facts_t* out_facts);

// Clones all defined source entries into |target|. When |module| is provided,
// extension payloads use the type-owned domain implied by each value ID.
// Undefined entries remain unset in |target| so normal block-argument and op
// fact seeding can fill them.
iree_status_t loom_value_fact_table_clone_defined_facts(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    const loom_module_t* module);

// Computes facts for a single op by calling its vtable fact inference function.
// Gathers operand facts from the table, calls the callback, and defines result
// facts. No-op if the op has no inference function.
iree_status_t loom_value_fact_table_compute_op(loom_value_fact_table_t* table,
                                               const loom_module_t* module,
                                               const loom_op_t* op);

// Computes facts for a single op and reports whether any result facts changed
// relative to the table's previous entries. No-op ops report false.
iree_status_t loom_value_fact_table_compute_op_and_report(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, bool* out_changed);

// Seeds the table by running a forward pass over |region| and its nested
// regions. |function| supplies the logical function context for op fact
// inference, and may be empty for detached regions. |parent_op| is the op that
// owns |region|; it is used to seed entry-block argument facts such as loop IV
// ranges. Visits ops in dominance order so operand facts are available before
// use.
iree_status_t loom_value_fact_table_compute_region(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_func_like_t function, loom_region_t* region, loom_op_t* parent_op);

// Seeds the table by running a forward pass over |function|'s root regions and
// nested regions. Non-body root regions are analyzed before the body so source
// containers can compute launch/configuration facts that body ops query through
// the same fact table.
iree_status_t loom_value_fact_table_compute(loom_value_fact_table_t* table,
                                            const loom_module_t* module,
                                            loom_func_like_t function);

// Returns a facts scratch buffer with at least |count| entries.
// The returned pointer is valid until the next call. Grows the
// allocation if needed; never shrinks.
iree_status_t loom_value_fact_table_facts_scratch(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_facts_t** out);

// Returns a value ID scratch buffer with at least |count| entries.
// Same lifetime and growth semantics as facts_scratch.
iree_status_t loom_value_fact_table_value_id_scratch(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_id_t** out);

// Creates facts for a vector whose every lane has |element| facts.
iree_status_t loom_value_facts_make_uniform_element(
    loom_fact_context_t* context, loom_value_facts_t element,
    loom_value_facts_t* out);

// Returns true and populates |out| when |facts| is a uniform-element vector
// extension in |context|.
bool loom_value_facts_query_uniform_element(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_uniform_element_t* out);

// Creates facts for a small all-static vector with explicit per-lane facts.
// Vectors with more than LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT lanes degrade
// to unknown facts.
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
}
#endif

#endif  // LOOM_UTIL_FACT_TABLE_H_
