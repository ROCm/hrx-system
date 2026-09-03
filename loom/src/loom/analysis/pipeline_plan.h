// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Concrete pipeline graph planning.
//
// This analysis resolves specialized group cardinalities and buffer capacities,
// expands group lanes into callable instances, and records the point-to-point
// flow graph once. Target materializers consume the immutable result without
// rescanning pipeline IR or rediscovering SSA relationships.

#ifndef LOOM_ANALYSIS_PIPELINE_PLAN_H_
#define LOOM_ANALYSIS_PIPELINE_PLAN_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_pipeline_binding_access_flag_bits_e {
  // The pipeline reads records from the binding.
  LOOM_PIPELINE_BINDING_ACCESS_FLAG_READ = 1u << 0,

  // The pipeline writes records to the binding.
  LOOM_PIPELINE_BINDING_ACCESS_FLAG_WRITE = 1u << 1,
} loom_pipeline_binding_access_flag_bits_t;

typedef uint32_t loom_pipeline_binding_access_flags_t;

typedef enum loom_pipeline_endpoint_kind_e {
  // An external launch binding endpoint.
  LOOM_PIPELINE_ENDPOINT_KIND_BINDING = 0,

  // A resident callable lane instance endpoint.
  LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE = 1,
} loom_pipeline_endpoint_kind_t;

typedef struct loom_pipeline_plan_binding_t {
  // Combined access required by every flow using this binding.
  loom_pipeline_binding_access_flags_t access;
} loom_pipeline_plan_binding_t;

typedef struct loom_pipeline_plan_group_t {
  // Source SSA identity naming the scheduling group.
  loom_value_id_t source_value;

  // Exact specialized lane count.
  uint32_t lane_count;

  // First callable instance belonging to the group.
  uint32_t instance_start;
} loom_pipeline_plan_group_t;

typedef struct loom_pipeline_plan_instance_t {
  // Scheduling group containing this instance.
  uint32_t group_index;

  // Lane ordinal within the scheduling group.
  uint32_t lane;

  // Callable implementing one record firing.
  loom_symbol_ref_t entry;
} loom_pipeline_plan_instance_t;

typedef struct loom_pipeline_plan_flow_t {
  // Source SSA identity naming this flow version.
  loom_value_id_t source_value;

  // Tile record type transferred by the flow.
  loom_type_t tile_type;

  // Scheduling group whose lanes produce or consume records pointwise.
  uint32_t group_index;

  // Minimum record capacity required by authored buffering.
  uint32_t minimum_capacity;

  // Kind of endpoint producing this flow.
  loom_pipeline_endpoint_kind_t producer_kind;

  // Binding ordinal when |producer_kind| is BINDING.
  uint32_t binding_index;

  // First producer instance when |producer_kind| is INSTANCE.
  uint32_t instance_start;

  // Number of producer instances, or zero for a binding producer.
  uint32_t instance_count;

  // Callable output port or first external binding port.
  uint32_t producer_port;

  // Full source-view tile type for a lane-partitioned binding producer.
  loom_type_t partition_source_type;

  // True when the binding producer partitions its leading dimension by lane.
  bool partitioned;
} loom_pipeline_plan_flow_t;

typedef struct loom_pipeline_plan_edge_t {
  // Flow supplying the edge and its record contract.
  uint32_t flow_index;

  // Kind of concrete producer endpoint.
  loom_pipeline_endpoint_kind_t source_kind;

  // Binding or instance index selected by |source_kind|.
  uint32_t source_index;

  // Producer endpoint port.
  uint32_t source_port;

  // Partition lane for a lane-partitioned binding source.
  uint32_t partition_lane;

  // Kind of concrete consumer endpoint.
  loom_pipeline_endpoint_kind_t target_kind;

  // Binding or instance index selected by |target_kind|.
  uint32_t target_index;

  // Consumer endpoint port.
  uint32_t target_port;
} loom_pipeline_plan_edge_t;

typedef struct loom_pipeline_plan_limits_t {
  // Maximum resident callable instances accepted by the materializer.
  uint32_t instance_count;
} loom_pipeline_plan_limits_t;

typedef struct loom_pipeline_plan_t {
  // Source pipeline function represented by this plan.
  loom_func_like_t pipeline;

  // Launch bindings indexed by source ABI ordinal.
  const loom_pipeline_plan_binding_t* bindings;

  // Number of launch binding slots.
  uint32_t binding_count;

  // Scheduling groups in source definition order.
  const loom_pipeline_plan_group_t* groups;

  // Number of scheduling groups.
  uint32_t group_count;

  // Resident callable instances in stage and lane order.
  const loom_pipeline_plan_instance_t* instances;

  // Number of resident callable instances.
  uint32_t instance_count;

  // Typed logical flows in source definition order.
  const loom_pipeline_plan_flow_t* flows;

  // Number of typed logical flows.
  uint32_t flow_count;

  // Concrete point-to-point flow edges.
  const loom_pipeline_plan_edge_t* edges;

  // Number of concrete flow edges.
  uint32_t edge_count;
} loom_pipeline_plan_t;

// Builds one immutable concrete plan for a verified pipeline definition.
//
// Exact group cardinalities, dynamic tile dimensions, view offsets, and
// buffering capacities are read from |facts|. |limits| is supplied by the
// materializer after target specialization and bounds planning allocations.
// All plan storage is allocated from |arena|.
iree_status_t loom_pipeline_plan_build(const loom_module_t* module,
                                       loom_func_like_t pipeline,
                                       const loom_value_fact_table_t* facts,
                                       loom_pipeline_plan_limits_t limits,
                                       iree_arena_allocator_t* arena,
                                       loom_pipeline_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_PLAN_H_
