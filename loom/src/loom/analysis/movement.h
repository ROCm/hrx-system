// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent memory movement requests.
//
// Movement requests classify source IR operations before any target selects an
// addressing mode, packet, intrinsic, or scalar fallback. The analysis owns the
// common facts about endpoints, byte footprints, storage schema, masks, and
// async eligibility so target lowerings and legality passes do not each
// rediscover the same view and vector memory shape.

#ifndef LOOM_ANALYSIS_MOVEMENT_H_
#define LOOM_ANALYSIS_MOVEMENT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/view_regions.h"
#include "loom/ir/ir.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/vector/memory.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Operation-level movement category.
typedef enum loom_movement_kind_e {
  LOOM_MOVEMENT_KIND_UNKNOWN = 0,
  LOOM_MOVEMENT_KIND_VECTOR_LOAD = 1,
  LOOM_MOVEMENT_KIND_VECTOR_STORE = 2,
  LOOM_MOVEMENT_KIND_VECTOR_LOAD_MASK = 3,
  LOOM_MOVEMENT_KIND_VECTOR_STORE_MASK = 4,
  LOOM_MOVEMENT_KIND_VECTOR_LOAD_EXPAND = 5,
  LOOM_MOVEMENT_KIND_VECTOR_STORE_COMPRESS = 6,
  LOOM_MOVEMENT_KIND_VECTOR_GATHER = 7,
  LOOM_MOVEMENT_KIND_VECTOR_SCATTER = 8,
  LOOM_MOVEMENT_KIND_VECTOR_GATHER_MASK = 9,
  LOOM_MOVEMENT_KIND_VECTOR_SCATTER_MASK = 10,
  LOOM_MOVEMENT_KIND_KERNEL_ASYNC_COPY = 11,
  LOOM_MOVEMENT_KIND_KERNEL_ASYNC_COPY_MASK = 12,
  LOOM_MOVEMENT_KIND_KERNEL_ASYNC_GATHER = 13,
  LOOM_MOVEMENT_KIND_KERNEL_ASYNC_GATHER_MASK = 14,
  LOOM_MOVEMENT_KIND_KERNEL_ASYNC_CLUSTER_GATHER = 15,
  LOOM_MOVEMENT_KIND_KERNEL_ASYNC_CLUSTER_GATHER_MASK = 16,
  LOOM_MOVEMENT_KIND_KERNEL_ASYNC_TENSOR_LOAD_TO_LDS = 17,
  LOOM_MOVEMENT_KIND_KERNEL_ASYNC_TENSOR_STORE_FROM_LDS = 18,
} loom_movement_kind_t;

// Endpoint storage category.
typedef enum loom_movement_endpoint_kind_e {
  LOOM_MOVEMENT_ENDPOINT_NONE = 0,
  LOOM_MOVEMENT_ENDPOINT_VIEW = 1,
  LOOM_MOVEMENT_ENDPOINT_REGISTER = 2,
} loom_movement_endpoint_kind_t;

// Movement layout category visible before target selection.
typedef enum loom_movement_layout_kind_e {
  LOOM_MOVEMENT_LAYOUT_UNKNOWN = 0,
  LOOM_MOVEMENT_LAYOUT_DENSE = 1,
  LOOM_MOVEMENT_LAYOUT_STATIC_STRIDED = 2,
  LOOM_MOVEMENT_LAYOUT_COMPRESS_EXPAND = 3,
  LOOM_MOVEMENT_LAYOUT_GATHER_SCATTER = 4,
  LOOM_MOVEMENT_LAYOUT_SUBGROUP_GATHER = 5,
  LOOM_MOVEMENT_LAYOUT_BYTE_RANGE = 6,
  LOOM_MOVEMENT_LAYOUT_CLUSTER_GATHER = 7,
  LOOM_MOVEMENT_LAYOUT_TENSOR_TILE = 8,
} loom_movement_layout_kind_t;

// Relationship between byte movement and typed storage interpretation.
typedef enum loom_movement_schema_kind_e {
  LOOM_MOVEMENT_SCHEMA_UNKNOWN = 0,
  LOOM_MOVEMENT_SCHEMA_TYPED_ELEMENT = 1,
  LOOM_MOVEMENT_SCHEMA_BYTE_PRESERVING = 2,
} loom_movement_schema_kind_t;

enum loom_movement_endpoint_flag_bits_e {
  // The endpoint is read by the movement.
  LOOM_MOVEMENT_ENDPOINT_READ = 1u << 0,

  // The endpoint is written by the movement.
  LOOM_MOVEMENT_ENDPOINT_WRITE = 1u << 1,

  // The static_begin_byte_offset field is exact.
  LOOM_MOVEMENT_ENDPOINT_STATIC_BEGIN = 1u << 2,

  // The static_byte_length field is exact.
  LOOM_MOVEMENT_ENDPOINT_STATIC_LENGTH = 1u << 3,
};
typedef uint32_t loom_movement_endpoint_flags_t;

enum loom_movement_request_flag_bits_e {
  // The movement is predicated and may touch no memory for false lanes.
  LOOM_MOVEMENT_REQUEST_MASKED = 1u << 0,

  // The movement participates in the kernel async-token stream.
  LOOM_MOVEMENT_REQUEST_ASYNC = 1u << 1,

  // The movement uses per-lane logical offsets.
  LOOM_MOVEMENT_REQUEST_IRREGULAR_OFFSETS = 1u << 2,

  // The transferred_byte_count field is exact.
  LOOM_MOVEMENT_REQUEST_STATIC_TRANSFER = 1u << 3,

  // The request is eligible for optimized async lowering.
  LOOM_MOVEMENT_REQUEST_ASYNC_ELIGIBLE = 1u << 4,

  // The direction field is present.
  LOOM_MOVEMENT_REQUEST_HAS_DIRECTION = 1u << 5,
};
typedef uint32_t loom_movement_request_flags_t;

typedef uint32_t loom_movement_rejection_flags_t;

#define LOOM_MOVEMENT_REJECTION_UNSUPPORTED_OP ((uint32_t)1u << 0)
#define LOOM_MOVEMENT_REJECTION_ENDPOINT ((uint32_t)1u << 1)
#define LOOM_MOVEMENT_REJECTION_VECTOR_ACCESS ((uint32_t)1u << 2)
#define LOOM_MOVEMENT_REJECTION_LAYOUT ((uint32_t)1u << 3)
#define LOOM_MOVEMENT_REJECTION_ELEMENT_WIDTH ((uint32_t)1u << 4)
#define LOOM_MOVEMENT_REJECTION_LANE_COUNT ((uint32_t)1u << 5)
#define LOOM_MOVEMENT_REJECTION_FOOTPRINT ((uint32_t)1u << 6)
#define LOOM_MOVEMENT_REJECTION_CACHE_POLICY ((uint32_t)1u << 7)

// Diagnostic summary for request construction.
typedef struct loom_movement_diagnostic_t {
  // Bitset of loom_movement_rejection_flags_t values.
  loom_movement_rejection_flags_t rejection_bits;
} loom_movement_diagnostic_t;

// One source or destination endpoint of a movement request.
typedef struct loom_movement_endpoint_t {
  // Storage category for value_id.
  loom_movement_endpoint_kind_t kind;

  // Bitset of loom_movement_endpoint_flag_bits_e.
  loom_movement_endpoint_flags_t flags;

  // SSA value naming this endpoint.
  loom_value_id_t value_id;

  // Type of value_id.
  loom_type_t type;

  // Target-independent memory space for view endpoints.
  loom_value_fact_memory_space_t memory_space;

  // Storage root identity for view endpoints.
  loom_value_id_t root_value_id;

  // Symbolic begin byte offset relative to root_value_id.
  loom_symbolic_expr_t begin_byte_offset;

  // Symbolic byte footprint length for this endpoint.
  loom_symbolic_expr_t byte_length;

  // Symbolic byte offset one past the endpoint footprint.
  loom_symbolic_expr_t end_byte_offset;

  // Minimum provable alignment of begin_byte_offset relative to root_value_id.
  uint64_t minimum_alignment;

  // Minimum provable byte alignment of the root storage base.
  uint64_t root_minimum_alignment;

  // Exact begin byte offset when LOOM_MOVEMENT_ENDPOINT_STATIC_BEGIN is set.
  int64_t static_begin_byte_offset;

  // Exact byte footprint length when LOOM_MOVEMENT_ENDPOINT_STATIC_LENGTH is
  // set.
  int64_t static_byte_length;

  // Precision inherited from the underlying view-region analysis.
  loom_view_region_precision_flags_t precision_flags;
} loom_movement_endpoint_t;

// A target-independent request extracted from one source IR operation.
typedef struct loom_movement_request_t {
  // Source operation being classified.
  const loom_op_t* op;

  // Operation-level movement category.
  loom_movement_kind_t kind;

  // Bitset of loom_movement_request_flag_bits_e.
  loom_movement_request_flags_t flags;

  // Layout category before target-specific addressing selection.
  loom_movement_layout_kind_t layout_kind;

  // Storage interpretation preserved by the movement.
  loom_movement_schema_kind_t schema_kind;

  // Read/source endpoint.
  loom_movement_endpoint_t source;

  // Written/destination endpoint.
  loom_movement_endpoint_t dest;

  // Exact transferred byte count when LOOM_MOVEMENT_REQUEST_STATIC_TRANSFER is
  // set.
  int64_t transferred_byte_count;

  // Optional mask or predicate SSA value, or LOOM_VALUE_ID_INVALID.
  loom_value_id_t mask_value_id;

  // Optional per-lane offset SSA value, or LOOM_VALUE_ID_INVALID.
  loom_value_id_t offsets_value_id;

  // Optional hardware descriptor SSA value, or LOOM_VALUE_ID_INVALID.
  loom_value_id_t descriptor_value_id;

  // Optional cluster mask SSA value, or LOOM_VALUE_ID_INVALID.
  loom_value_id_t cluster_mask_value_id;

  // Optional kernel direction enum when LOOM_MOVEMENT_REQUEST_HAS_DIRECTION is
  // set.
  uint8_t direction;

  // Optional cache policy carried by the source operation.
  loom_vector_memory_cache_policy_t cache_policy;

  // Vector access summary for vector memory operations.
  loom_vector_memory_access_t vector_access;
} loom_movement_request_t;

// Per-run analysis state. The caller owns arena and fact_table.
typedef struct loom_movement_analysis_t {
  // Module containing operations and values being classified.
  const loom_module_t* module;

  // Current value facts consumed by endpoint and layout queries.
  loom_value_fact_table_t* fact_table;

  // Arena backing view-region and symbolic-expression storage.
  iree_arena_allocator_t* arena;

  // View-region analysis reused for endpoint roots and byte intervals.
  loom_view_region_table_t view_regions;
} loom_movement_analysis_t;

// Initializes movement analysis from a caller-owned active local value domain.
// The domain must remain active until the analysis is dead.
iree_status_t loom_movement_analysis_initialize(
    loom_value_fact_table_t* fact_table,
    const loom_local_value_domain_t* value_domain,
    iree_arena_allocator_t* arena, loom_movement_analysis_t* out_analysis);

// Pre-analyzes all view regions reachable from the analysis value domain.
iree_status_t loom_movement_analysis_analyze(
    loom_movement_analysis_t* analysis);

// Returns true when op_kind is represented by the async movement table.
bool loom_movement_op_kind_is_async(loom_op_kind_t op_kind);

// Classifies one source IR operation as a movement request.
//
// Unsupported or unrepresentable operations return OK with |out_described| set
// to false and details in |out_diagnostic|. Status failures are reserved for
// analysis infrastructure failures such as arena allocation.
iree_status_t loom_movement_request_describe_op(
    loom_movement_analysis_t* analysis, const loom_op_t* op,
    loom_movement_request_t* out_request,
    loom_movement_diagnostic_t* out_diagnostic, bool* out_described);

// Returns the minimum provable byte alignment for a memory endpoint address.
//
// The result combines root storage alignment with the endpoint begin offset.
// Non-memory endpoints or unknown alignment facts return 0.
uint64_t loom_movement_endpoint_minimum_byte_alignment(
    const loom_movement_endpoint_t* endpoint);

// Returns a diagnostic detail string for movement request rejection flags.
iree_string_view_t loom_movement_rejection_detail(
    loom_movement_rejection_flags_t rejection_bits);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_MOVEMENT_H_
