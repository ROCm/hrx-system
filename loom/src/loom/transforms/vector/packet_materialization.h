// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_VECTOR_PACKET_MATERIALIZATION_H_
#define LOOM_TRANSFORMS_VECTOR_PACKET_MATERIALIZATION_H_

#include "iree/base/api.h"
#include "loom/ops/vector/memory.h"
#include "loom/transforms/vector/packet_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_vector_packet_slice_t {
  // Dynamic logical lane offset, or LOOM_VALUE_ID_INVALID for a static slice.
  loom_value_id_t dynamic_lane_offset;
  // Static logical lane offset when dynamic_lane_offset is invalid.
  uint32_t static_lane_offset;
  // Number of logical lanes in this packet.
  uint32_t lane_count;
} loom_vector_packet_slice_t;

iree_status_t loom_vector_packet_rewrite_memory_load(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_vector_memory_footprint_t* footprint,
    const loom_vector_packet_shape_t* shape,
    loom_vector_memory_cache_policy_t cache_policy);

// Rewrites a load into exact target-legal vector packets and a scalar terminal
// suffix when no legal memory packet fits the remaining lanes.
iree_status_t loom_vector_packet_rewrite_memory_load_components(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy,
    const loom_vector_memory_footprint_t* footprint,
    const loom_vector_packet_shape_t* shape,
    loom_vector_memory_cache_policy_t cache_policy, bool* out_rewritten);

iree_status_t loom_vector_packet_preserve_memory_loads(
    loom_vector_packetization_t* packetization,
    const loom_vector_packet_shape_t* shape, bool* out_rewritten);

// Preserves loads for a consumer that can combine exact scalar/vector leaves.
// Legal memory packets are retained directly; a terminal suffix smaller than
// every legal packet is published as scalar components.
iree_status_t loom_vector_packet_preserve_memory_load_components(
    loom_vector_packetization_t* packetization,
    const loom_vector_packet_shape_t* shape, bool* out_rewritten);

iree_status_t loom_vector_packet_staging_load(
    loom_vector_packetization_t* packetization, loom_value_id_t staging_view,
    loom_type_t vector_type, const loom_vector_packet_slice_t* slice,
    const loom_op_t* source_op, loom_value_id_t* out_packet);

iree_status_t loom_vector_packet_materialize(
    loom_vector_packetization_t* packetization, loom_value_id_t source,
    const loom_vector_packet_slice_t* slice,
    loom_vector_packetized_value_t** out_value);

// Computes the exact number of operations needed to materialize one dynamic
// packet from every selected producer. Returns false when the offset facts do
// not intersect the source domain or the operation count overflows.
bool loom_vector_packet_dynamic_materialization_operation_count(
    const loom_vector_packetization_t* packetization,
    loom_value_facts_t lane_offset_facts, uint32_t lane_count,
    iree_host_size_t* out_operation_count);

iree_status_t loom_vector_packet_store(
    loom_vector_packetization_t* packetization,
    const loom_vector_memory_footprint_t* store_footprint,
    const loom_vector_packet_slice_t* slice, loom_value_id_t packet,
    loom_vector_memory_cache_policy_t store_cache_policy, loom_op_t* store_op);

bool loom_vector_packet_static_operation_count_is_bounded(
    iree_host_size_t operations_per_chunk,
    const loom_vector_packet_shape_t* shape);

iree_status_t loom_vector_packet_store_captured_value(
    loom_vector_packetization_t* packetization,
    const loom_vector_memory_footprint_t* store_footprint,
    const loom_vector_packet_shape_t* shape,
    loom_vector_memory_cache_policy_t store_cache_policy, loom_op_t* store_op);

bool loom_vector_packet_static_expansion_is_bounded(
    const loom_vector_packetization_t* packetization,
    const loom_vector_packet_shape_t* shape);

iree_status_t loom_vector_packet_static_interleaved_store(
    loom_vector_packetization_t* packetization,
    const loom_vector_memory_footprint_t* store_footprint,
    const loom_vector_packet_shape_t* shape,
    loom_vector_memory_cache_policy_t store_cache_policy, loom_op_t* store_op);

iree_status_t loom_vector_packet_static_store(
    loom_vector_packetization_t* packetization,
    const loom_vector_memory_footprint_t* store_footprint,
    const loom_vector_packet_shape_t* shape,
    loom_vector_memory_cache_policy_t store_cache_policy, loom_op_t* store_op);

iree_status_t loom_vector_packet_build_staging_view(
    loom_vector_packetization_t* packetization, loom_type_t vector_type,
    const loom_vector_packet_shape_t* shape, const loom_op_t* source_op,
    loom_value_id_t* out_staging_view);

iree_status_t loom_vector_packet_staging_store(
    loom_vector_packetization_t* packetization, loom_value_id_t staging_view,
    const loom_vector_packet_slice_t* slice, loom_value_id_t packet,
    const loom_op_t* source_op);

iree_status_t loom_vector_packet_canonicalize_snapshots(
    loom_vector_packetization_t* packetization,
    const loom_vector_packet_shape_t* shape);

iree_status_t loom_vector_packet_staged_store(
    loom_vector_packetization_t* packetization,
    const loom_vector_memory_footprint_t* store_footprint,
    const loom_vector_packet_shape_t* shape,
    loom_vector_memory_cache_policy_t store_cache_policy, loom_op_t* store_op);

bool loom_vector_packet_has_snapshots(
    const loom_vector_packetization_t* packetization);

bool loom_vector_packet_has_captured_values(
    const loom_vector_packetization_t* packetization);

bool loom_vector_packet_requires_static_slices(
    const loom_vector_packetization_t* packetization);

iree_status_t loom_vector_packet_static_reduce(
    loom_vector_packetization_t* packetization, loom_value_id_t input,
    const loom_vector_packet_shape_t* shape, loom_op_t* reduce_op,
    bool fold_packet_inputs, loom_value_id_t* out_accumulator);

// Reduces an exact scalar/vector component partition without materializing its
// logical aggregate carrier.
iree_status_t loom_vector_packet_partition_reduce(
    loom_target_legalization_context_t* context,
    const loom_vector_packet_partition_reduce_plan_t* plan,
    loom_value_fact_vector_partition_t partition, loom_op_t* reduce_op,
    loom_value_id_t* out_accumulator);

iree_status_t loom_vector_packet_erase_dead_sources(
    loom_vector_packetization_t* packetization, const loom_op_t* excluded_op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_PACKET_MATERIALIZATION_H_
