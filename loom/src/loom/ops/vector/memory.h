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

  // Inline stride fact storage backing layout_summary.
  loom_value_facts_t layout_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
} loom_vector_memory_access_t;

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

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_VECTOR_MEMORY_H_
