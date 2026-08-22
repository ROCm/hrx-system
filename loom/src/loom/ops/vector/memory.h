// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vector memory access decomposition.
//
// Vector loads and stores operate on typed views, not on hidden transfer
// pseudo-semantics. This helper describes the regular mapping from a full-rank
// view origin plus a trailing vector lane coordinate to an element or byte
// offset relative to the view base. Callers use it for verification,
// scalarization, reference lowering, and future alias-region derivation.

#ifndef LOOM_OPS_VECTOR_MEMORY_H_
#define LOOM_OPS_VECTOR_MEMORY_H_

#include "iree/base/api.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/op_defs.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// The address-layout operation shape visible through a view type.
typedef enum loom_vector_memory_layout_kind_e {
  LOOM_VECTOR_MEMORY_LAYOUT_UNKNOWN = 0,
  LOOM_VECTOR_MEMORY_LAYOUT_DENSE = 1,
  LOOM_VECTOR_MEMORY_LAYOUT_STRIDED = 2,
} loom_vector_memory_layout_kind_t;

// A decomposed view/vector memory access.
typedef struct loom_vector_memory_access_t {
  // Logical view being accessed. The dimensions bound valid origins.
  loom_type_t view_type;

  // Register lane grid loaded from or stored into the view.
  loom_type_t vector_type;

  // Rank of view_type.
  uint8_t view_rank;

  // Rank of vector_type.
  uint8_t vector_rank;

  // First view axis covered by vector lanes; vector axes map to trailing view
  // axes.
  uint8_t first_vector_axis;

  // Bit count of one addressed view element, or 0 if the element type is
  // malformed.
  int32_t element_bit_count;

  // Byte count of one addressed view element, or -1 when the element is not
  // byte-aligned or the element type is malformed.
  int64_t static_element_byte_count;

  // Resolved address-layout kind for the view type.
  loom_vector_memory_layout_kind_t layout_kind;

  // Address-layout summary decoded from static encodings or fact-bearing SSA
  // encoding values.
  loom_value_fact_address_layout_t layout_summary;

  // Authored operands that materialize a non-exact explicit strided layout.
  // Empty for dense/exact layouts and fact-only layouts without SSA stride
  // provenance.
  loom_encoding_address_layout_operands_t layout_operands;

  // Inline stride fact storage backing layout_summary.
  loom_value_facts_t layout_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
} loom_vector_memory_access_t;

// Semantic memory-footprint family for vector MemoryAccess ops.
typedef enum loom_vector_memory_footprint_kind_e {
  // Not a vector memory footprint.
  LOOM_VECTOR_MEMORY_FOOTPRINT_NONE = 0,
  // Ordinary vector.load/store footprint over trailing view axes.
  LOOM_VECTOR_MEMORY_FOOTPRINT_DENSE = 1,
  // Masked vector.load.mask/store.mask footprint over trailing view axes.
  LOOM_VECTOR_MEMORY_FOOTPRINT_MASKED_DENSE = 2,
  // Expand/compress footprint whose memory element count depends on mask
  // population.
  LOOM_VECTOR_MEMORY_FOOTPRINT_COMPRESS_EXPAND = 3,
  // Gather/scatter footprint selected by per-lane element offsets.
  LOOM_VECTOR_MEMORY_FOOTPRINT_PER_LANE_OFFSET = 4,
  // Masked gather/scatter footprint selected by active per-lane offsets.
  LOOM_VECTOR_MEMORY_FOOTPRINT_MASKED_PER_LANE_OFFSET = 5,
  // Atomic gather/scatter footprint selected by per-lane offsets.
  LOOM_VECTOR_MEMORY_FOOTPRINT_ATOMIC_PER_LANE = 6,
  // Masked atomic gather/scatter footprint selected by active per-lane offsets.
  LOOM_VECTOR_MEMORY_FOOTPRINT_MASKED_ATOMIC_PER_LANE = 7,
  // Target-shaped fragment movement; the payload vector is not the logical
  // memory footprint.
  LOOM_VECTOR_MEMORY_FOOTPRINT_FRAGMENT = 8,
} loom_vector_memory_footprint_kind_t;

enum loom_vector_memory_footprint_flag_bits_e {
  // The footprint reads memory.
  LOOM_VECTOR_MEMORY_FOOTPRINT_READS = 1u << 0,
  // The footprint writes memory.
  LOOM_VECTOR_MEMORY_FOOTPRINT_WRITES = 1u << 1,
};
// Bitfield of loom_vector_memory_footprint_flag_bits_e values.
typedef uint32_t loom_vector_memory_footprint_flags_t;

// Physical storage scaling applied to one logical vector footprint axis.
typedef struct loom_vector_memory_footprint_axis_scale_t {
  // Vector axis whose logical extent is scaled, or UINT8_MAX when absent.
  uint8_t vector_axis;

  // Physical elements stored for each logical group.
  uint16_t storage_element_count;

  // Logical elements represented by each physical group.
  uint16_t logical_element_count;
} loom_vector_memory_footprint_axis_scale_t;

// Classified vector memory footprint. This is a semantic description of legal
// vector memory IR, not a second verifier for malformed operations.
typedef struct loom_vector_memory_footprint_t {
  // Footprint family.
  loom_vector_memory_footprint_kind_t kind;

  // Read/write effects for the footprint.
  loom_vector_memory_footprint_flags_t flags;

  // Borrowed MemoryAccess interface reference.
  loom_memory_access_t access;

  // View being accessed.
  loom_value_id_t view;

  // Written/update value when the op shape has one.
  loom_value_id_t value;

  // Lane/activity mask when the op shape has one.
  loom_value_id_t mask;

  // Passthrough value when the op shape has one.
  loom_value_id_t passthrough;

  // Per-lane offset vector when the op shape has one.
  loom_value_id_t offsets;

  // Dynamic logical origin indices.
  loom_value_slice_t dynamic_indices;

  // Static logical origin indices.
  loom_attribute_t static_indices;

  // Typed view operand.
  loom_type_t view_type;

  // Vector payload type that describes the logical footprint. Fragment memory
  // ops use the logical matrix shape, not the physical fragment payload shape.
  loom_type_t vector_type;

  // Inline backing for synthetic rank-3 fragment footprint dimensions.
  loom_overflow_dim_t fragment_dimensions[3];

  // Optional physical storage scaling for one logical footprint axis.
  loom_vector_memory_footprint_axis_scale_t axis_scale;

  // Decomposed view/vector relationship for the logical footprint.
  loom_vector_memory_access_t vector_access;
} loom_vector_memory_footprint_t;

// Generated-builder flags shared by vector and view memory ops.
typedef enum loom_vector_memory_cache_policy_build_flag_e {
  // cache_scope is present in the optional attribute dictionary.
  LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_SCOPE = 1u << 0,
  // cache_temporal is present in the optional attribute dictionary.
  LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_TEMPORAL = 1u << 1,
} loom_vector_memory_cache_policy_build_flag_t;

// Optional cache policy attached to a vector memory op.
typedef struct loom_vector_memory_cache_policy_t {
  // Optional-attribute flags for forwarding this policy through builders.
  uint32_t build_flags;

  // cache_scope enum payload when the corresponding build flag is set.
  uint8_t cache_scope;

  // cache_temporal enum payload when the corresponding build flag is set.
  uint8_t cache_temporal;
} loom_vector_memory_cache_policy_t;

// Describes a view/vector access. |context| supplies SSA encoding facts; pass
// NULL only from structural verification paths that intentionally cannot depend
// on analysis. Returns false when the types are not a view and a rank-1+
// vector, or when the vector rank exceeds the view rank. Unknown layouts still
// produce a valid access with layout_kind UNKNOWN.
bool loom_vector_memory_access_describe(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t view_type, loom_type_t vector_type,
    loom_vector_memory_access_t* out_access);

// Describes a vector memory operation footprint using the MemoryAccess
// interface and vector memory shape contracts. Returns false for non-vector
// memory ops or malformed vector memory IR.
bool loom_vector_memory_footprint_describe(
    const loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, loom_vector_memory_footprint_t* out_footprint);

// Returns the semantic vector memory footprint family for |op| without
// decomposing view/vector layout details. Non-vector memory ops return NONE.
loom_vector_memory_footprint_kind_t loom_vector_memory_op_footprint_kind(
    const loom_module_t* module, const loom_op_t* op);

// Copies full-rank static logical extents for the footprint into |out_extents|.
// Returns false when any footprint axis is dynamic or when |capacity| is
// smaller than the view rank.
bool loom_vector_memory_footprint_static_extents(
    const loom_vector_memory_footprint_t* footprint, int64_t* out_extents,
    iree_host_size_t capacity);

// Extracts the optional cache policy from a vector memory op. Returns false
// for non-memory ops or malformed cache attrs so callers do not rewrite away
// verifier-owned diagnostics.
bool loom_vector_memory_cache_policy_from_op(
    const loom_module_t* module, const loom_op_t* op,
    loom_vector_memory_cache_policy_t* out_policy);

// Extracts an optional cache policy from the two generated optional cache
// attributes shared by vector and view memory ops.
bool loom_vector_memory_cache_policy_from_attrs(
    loom_attribute_t cache_scope_attr, loom_attribute_t cache_temporal_attr,
    loom_vector_memory_cache_policy_t* out_policy);

// Returns the static element extent touched along a view axis. Leading view
// axes not covered by the vector have extent 1. Trailing axes use the
// corresponding vector dimension and return false when that dimension is
// dynamic.
bool loom_vector_memory_access_static_axis_extent(
    const loom_vector_memory_access_t* access, uint8_t view_axis,
    int64_t* out_extent);

// Returns the static element stride for a view axis. Dense row-major strides
// are derived from the view dimensions. Strided layouts read the corresponding
// static stride entry from encoding.layout.strided. Returns false for unknown
// layouts, dynamic strides, dynamic dense suffix dimensions, and malformed
// axes.
bool loom_vector_memory_access_static_axis_stride(
    const loom_vector_memory_access_t* access, uint8_t view_axis,
    int64_t* out_stride);

// Computes the static element offset for one lane relative to the view base.
// |static_indices| contains one full-rank logical origin index per view axis,
// using INT64_MIN for dynamic origins. |lane_indices| contains one non-negative
// coordinate per vector axis. Returns false when any required origin, stride,
// or arithmetic result is dynamic or not representable.
bool loom_vector_memory_access_static_lane_element_offset(
    const loom_vector_memory_access_t* access, loom_attribute_t static_indices,
    const int64_t* lane_indices, uint8_t lane_index_count,
    int64_t* out_element_offset);

// Computes the static byte offset for one lane relative to the view base. This
// is the element offset multiplied by the byte size of one addressed view
// element. Returns false for sub-byte element types such as i1.
bool loom_vector_memory_access_static_lane_byte_offset(
    const loom_vector_memory_access_t* access, loom_attribute_t static_indices,
    const int64_t* lane_indices, uint8_t lane_index_count,
    int64_t* out_byte_offset);

// Computes facts for the exclusive linear element end of a rectangular access.
// |axis_exclusive_end_facts| contains one view coordinate per view axis and
// must have at least access->view_rank entries. Returns false when the address
// layout does not provide exact element strides.
bool loom_vector_memory_access_linear_element_end_facts(
    const loom_vector_memory_access_t* access,
    const loom_value_facts_t* axis_exclusive_end_facts,
    loom_value_facts_t* out_element_end_facts);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_VECTOR_MEMORY_H_
