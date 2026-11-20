// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exact logical coordinates and physical element offsets for shaped storage.

#ifndef LOOM_ANALYSIS_STORAGE_GEOMETRY_H_
#define LOOM_ANALYSIS_STORAGE_GEOMETRY_H_

#include "loom/ir/module.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_storage_geometry_axis_t {
  // Number of logical coordinates on this axis.
  uint64_t extent;
  // Physical element offset between consecutive coordinates on this axis.
  uint64_t element_stride;
} loom_storage_geometry_axis_t;

// An exact dense or nonnegative strided address layout. This value owns its
// axes; retaining it does not retain the source IR or borrow fact-table
// storage. It describes physical scalar storage, not storage-schema decoding.
// Root, origin, allocation extent and synchronization identity belong to its
// owner.
typedef struct loom_storage_geometry_t {
  // Resolved encoding family. Dense layouts have explicit row-major axes here.
  loom_value_fact_address_layout_kind_t layout_kind;
  // Bit width of one addressed physical scalar element, including subbyte
  // types.
  uint16_t element_bit_count;
  // Number of axes, in outermost-to-innermost logical order.
  uint8_t rank;
  // Exact coordinate extents and physical strides for the first rank entries.
  loom_storage_geometry_axis_t axes[LOOM_TYPE_MAX_RANK];
} loom_storage_geometry_t;

typedef struct loom_storage_geometry_span_t {
  // Logical elements visited, counting repeated addresses independently.
  uint64_t element_count;
  // Address envelope from the origin through the last addressed element.
  // Padding contributes to this span; repeated addresses do not enlarge it.
  uint64_t element_span;
  // Number of consecutive elements in the dense innermost logical axes.
  // Subsequent groups of this size may be padded, permuted, or repeated.
  uint64_t contiguous_element_count;
} loom_storage_geometry_span_t;

static inline bool loom_storage_geometry_span_is_dense(
    const loom_storage_geometry_span_t* span) {
  return span->contiguous_element_count == span->element_count;
}

// Resolves a shaped type using static dimensions/encodings and already-retained
// SSA facts. No producer or call-graph traversal occurs. Returns false when the
// element width, layout, extents or nonnegative strides are not exact, or dense
// strides exceed the 64-bit element-offset domain. This is exact-realization
// admission, not source verification: unresolved runtime geometry remains valid
// source IR.
bool loom_storage_geometry_query(const loom_fact_context_t* context,
                                 const loom_module_t* module, loom_type_t type,
                                 loom_storage_geometry_t* out_geometry);

// Prepends one outer axis to an already measured inner span. An empty axis or
// span produces an empty span. Overflow returns false without changing the
// span. A scalar has one logical, addressed, and contiguous element.
// Consumers that already visit axes can accumulate a footprint without building
// a temporary geometry or visiting the coordinates again.
bool loom_storage_geometry_span_prepend(loom_storage_geometry_axis_t axis,
                                        loom_storage_geometry_span_t* span);

// Measures the trailing axes beginning at first_axis. Preceding coordinates
// select an origin and do not affect this relative span. An empty axis produces
// zero count/span; no axes describe one scalar element. Returns false if the
// count or address envelope cannot be represented in 64 bits. The geometry
// owner establishes rank, first_axis and nonnegative axis invariants.
bool loom_storage_geometry_measure(const loom_storage_geometry_t* geometry,
                                   uint8_t first_axis,
                                   loom_storage_geometry_span_t* out_span);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_STORAGE_GEOMETRY_H_
