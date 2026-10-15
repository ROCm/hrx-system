// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_VECTOR_PACKET_PLAN_H_
#define LOOM_TRANSFORMS_VECTOR_PACKET_PLAN_H_

#include "iree/base/api.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/vector/memory.h"
#include "loom/target/legalization.h"
#include "loom/util/fact_vector_partition.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum source packet operations cloned by the alias-preserving static
// fallback. Larger producer graphs stage through one private vector so compile
// time and emitted code remain proportional to the graph instead of the
// product of graph size and packet count.
#define LOOM_VECTOR_PACKET_STATIC_OP_LIMIT 64u

typedef struct loom_vector_packet_shape_t {
  // Number of logical lanes in every vector carried by the producer graph.
  uint32_t lane_count;
  // Number of matching logical lanes materialized per native packet.
  uint32_t chunk_lane_count;
  // Number of chunks needed to cover lane_count.
  uint32_t chunk_count;
} loom_vector_packet_shape_t;

typedef enum loom_vector_packet_partition_reduce_mode_e {
  // The retained partition does not expose a terminal mixed component plan.
  LOOM_VECTOR_PACKET_PARTITION_REDUCE_MODE_NONE = 0,
  // Reduces exact vector components and scalar lanes without rebuilding the
  // aggregate carrier.
  LOOM_VECTOR_PACKET_PARTITION_REDUCE_MODE_HYBRID = 1,
  // Normalizes exact components to one target-native leaf packet before one
  // packed reduction.
  LOOM_VECTOR_PACKET_PARTITION_REDUCE_MODE_PACKED = 2,
} loom_vector_packet_partition_reduce_mode_t;

typedef struct loom_vector_packet_partition_reduce_plan_t {
  // Selected component-consumption strategy.
  loom_vector_packet_partition_reduce_mode_t mode;
  // Common leaf packet shape when |mode| is PACKED.
  loom_vector_packet_shape_t leaf_shape;
  // Number of leaf-width values materialized before the balanced fold.
  uint32_t packed_value_count;
  // Number of scalar components grouped into each idempotent leaf packet.
  // Zero when the plan has no idempotent scalar-component packets.
  uint32_t scalar_pack_lane_count;
  // Whether the target accepts a reduction of one leaf packet.
  bool leaf_reduction_is_native;
} loom_vector_packet_partition_reduce_plan_t;

typedef enum loom_vector_packet_source_mode_e {
  // Rebuilds an ordinary load or pure producer for each packet.
  LOOM_VECTOR_PACKET_SOURCE_MODE_STREAMED = 0,
  // Rebuilds a vector.from_elements operand range for a static packet offset.
  LOOM_VECTOR_PACKET_SOURCE_MODE_STATIC_ELEMENTS = 1,
  // Legalizes a load at its original position before rewriting its consumer.
  LOOM_VECTOR_PACKET_SOURCE_MODE_PRESERVE_LOAD = 2,
  // Reads packets from a retained regular vector partition.
  LOOM_VECTOR_PACKET_SOURCE_MODE_SNAPSHOT = 3,
  // Reads static slices from an existing SSA value owned outside the graph.
  LOOM_VECTOR_PACKET_SOURCE_MODE_CAPTURED = 4,
} loom_vector_packet_source_mode_t;

typedef enum loom_vector_packet_selection_flag_bits_e {
  // Allows block arguments and loop results to terminate a producer graph.
  LOOM_VECTOR_PACKET_SELECTION_FLAG_CAPTURE_EXTERNAL = 1u << 0,
  // Allows an operation owned outside decomposable packetization to terminate
  // a producer graph. The operation retains its authored result and its own
  // legalizer remains responsible for realizing that value.
  LOOM_VECTOR_PACKET_SELECTION_FLAG_CAPTURE_NON_DECOMPOSABLE = 1u << 1,
} loom_vector_packet_selection_flag_bits_t;
typedef uint8_t loom_vector_packet_selection_flags_t;

typedef struct loom_vector_packet_value_t {
  // Target-legal vector value for the materialized packet.
  loom_value_id_t vector;
  // Direct scalar value for a one-lane packet, or invalid when unavailable.
  loom_value_id_t scalar;
} loom_vector_packet_value_t;

typedef struct loom_vector_packetized_value_t {
  // Original oversized source value represented by a native packet.
  loom_value_id_t source;
  // One-based child indices for source-ID bits above the root directory bits.
  // Zero denotes an absent branch; indices survive record-array growth.
  uint32_t children[2];
  // Original oversized source type.
  loom_type_t source_type;
  // Packet materialization mode for this source value.
  loom_vector_packet_source_mode_t source_mode;
  // Authored packet tuple retained for snapshot materialization.
  loom_value_fact_vector_partition_t source_partition;
  // Native packet materialized for the current physical loop iteration.
  loom_vector_packet_value_t packet;
} loom_vector_packetized_value_t;

// A 64-byte directory selects the first four bits without tree traversal.
#define LOOM_VECTOR_PACKET_INDEX_ROOT_BITS 4u

typedef struct loom_vector_packetization_t {
  // Target legalization context that owns the rewrite.
  loom_target_legalization_context_t* context;
  // Target-native packet widths controlling this rewrite.
  const loom_target_vector_packet_policy_t* policy;
  // Captured-value boundaries accepted while selecting the producer graph.
  loom_vector_packet_selection_flags_t selection_flags;
  // Source location assigned to slices of captured graph leaves.
  loom_location_id_t captured_value_location;
  // One-based roots selected by the low source-ID bits; zero is an empty root.
  uint32_t roots[1u << LOOM_VECTOR_PACKET_INDEX_ROOT_BITS];
  // Arena-backed packetized values materialized for this root rewrite.
  loom_vector_packetized_value_t* values;
  // Number of populated values.
  uint32_t value_count;
  // Number of allocated value entries.
  uint32_t value_capacity;
} loom_vector_packetization_t;

int32_t loom_vector_packet_element_bit_count(
    const loom_target_vector_packet_policy_t* policy,
    loom_scalar_type_t element_type);

bool loom_vector_packet_static_lane_count(loom_type_t vector_type,
                                          uint32_t* out_lane_count);

bool loom_vector_packet_maximum_chunk_lane_count(
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    uint32_t* out_chunk_lane_count);

bool loom_vector_packet_shape_from_type(
    const loom_target_legalization_context_t* context,
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    loom_vector_packet_shape_t* out_shape);

// Returns target-native lane candidates for a shape-preserving decomposable
// operation. |out_producer_closure_candidate_bits| identifies the subset whose
// target contract requires packet-local producer provenance.
iree_status_t loom_vector_packet_component_candidate_bits(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    uint32_t source_lane_count, uint64_t* out_candidate_bits,
    uint64_t* out_producer_closure_candidate_bits);

iree_status_t loom_vector_packet_memory_shape_constrain(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    loom_vector_packet_shape_t* inout_shape, bool* out_supported);

// Selects the widest target-native memory packet no larger than
// |maximum_lane_count|. The query projects the authored vector width directly
// to each candidate so terminal packet selection does not depend on creating
// and later legalizing an intermediate tail load. Returns zero when the target
// contract handled the candidates but rejected all that fit.
iree_status_t loom_vector_packet_memory_chunk_lane_count_select(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    uint32_t maximum_lane_count, uint32_t* out_lane_count);

// Returns exact load packet widths as a bitset over
// |policy->native_lane_counts|. A width is admitted only when both its regular
// packets and optional tail can be loaded without regrouping.
iree_status_t loom_vector_packet_load_candidate_bits(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    uint64_t* out_candidate_bits);

iree_status_t loom_vector_packet_reduce_shape_constrain(
    loom_target_legalization_context_t* context, const loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    loom_vector_packet_shape_t* inout_shape, bool* out_supported);

// Constrains a geometric packet shape to widths whose projected lanewise
// combining operation is target-native. This is the packet-fold fallback for
// reductions without a target-native horizontal reduction.
iree_status_t loom_vector_packet_reduce_combine_shape_constrain(
    loom_target_legalization_context_t* context, const loom_op_t* reduce_op,
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    loom_vector_packet_shape_t* inout_shape, bool* out_supported);

// Selects a reduction plan for an exact retained component partition. Packed
// plans use the widest legal native lane count that can represent every vector
// component through exact slicing, identity padding, or idempotent repetition.
// Terminal partitions without such a width retain a hybrid plan only when
// every vector component has a native horizontal reduction.
iree_status_t loom_vector_packet_partition_reduce_plan_select(
    loom_target_legalization_context_t* context, const loom_op_t* reduce_op,
    const loom_target_vector_packet_policy_t* policy,
    loom_value_fact_vector_partition_t partition,
    loom_vector_packet_partition_reduce_plan_t* out_plan);

bool loom_vector_packet_table_lookup_shape(
    const loom_target_vector_packet_policy_t* policy, loom_type_t index_type,
    loom_type_t result_type, loom_vector_packet_shape_t* out_shape);

loom_type_t loom_vector_packet_type(loom_type_t vector_type,
                                    uint32_t lane_count);

bool loom_vector_packet_memory_can_build_chunk_origins(
    const loom_target_legalization_context_t* context,
    const loom_vector_memory_footprint_t* footprint,
    const loom_vector_packet_shape_t* shape);

iree_status_t loom_vector_packet_memory_build_chunk_origin(
    loom_target_legalization_context_t* context,
    const loom_vector_memory_footprint_t* footprint, const loom_op_t* memory_op,
    uint32_t lane_offset, const loom_value_id_t** out_dynamic_indices,
    iree_host_size_t* out_dynamic_index_count,
    const int64_t** out_static_indices,
    iree_host_size_t* out_static_index_count, bool* out_built);

iree_status_t loom_vector_packet_memory_build_dynamic_chunk_origin(
    loom_target_legalization_context_t* context,
    const loom_vector_memory_footprint_t* footprint, const loom_op_t* memory_op,
    loom_value_id_t dynamic_lane_offset,
    const loom_value_id_t** out_dynamic_indices,
    iree_host_size_t* out_dynamic_index_count,
    const int64_t** out_static_indices,
    iree_host_size_t* out_static_index_count);

const loom_fact_context_t* loom_vector_packet_fact_context(
    const loom_target_legalization_context_t* context);

void loom_vector_packetization_initialize(
    loom_target_legalization_context_t* context,
    const loom_target_vector_packet_policy_t* policy,
    loom_vector_packetization_t* out_packetization);

loom_vector_packetized_value_t* loom_vector_packet_find(
    loom_vector_packetization_t* packetization, loom_value_id_t source);

iree_status_t loom_vector_packet_select_value_shape(
    loom_vector_packetization_t* packetization, loom_value_id_t source,
    loom_vector_packet_shape_t* inout_shape, bool* out_selected);

bool loom_vector_packet_shared_snapshot_requires_staging(
    loom_vector_packetization_t* packetization,
    const loom_vector_packet_shape_t* shape, const loom_op_t* root_op);

// Classifies selected load leaves at their authored observation point. When
// |allow_streaming| is true, exclusive ordinary loads may instead replay at
// the root when indexed motion analysis proves that move safe.
iree_status_t loom_vector_packet_classify_memory_loads(
    loom_vector_packetization_t* packetization, const loom_op_t* root_op,
    bool allow_streaming);

bool loom_vector_packet_can_materialize(
    const loom_vector_packetization_t* packetization,
    const loom_vector_packet_shape_t* shape);

bool loom_vector_packet_store_can_interleave(
    const loom_vector_packetization_t* packetization,
    const loom_vector_memory_footprint_t* store_footprint);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_PACKET_PLAN_H_
