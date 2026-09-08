// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Concrete pipeline graph planning.
//
// This analysis resolves specialized group cardinalities and buffer capacities,
// expands group lanes into callable instances, and records the point-to-point
// flow graph once. Multiple uses of one flow become edges sharing the same
// producer endpoint. Target materializers consume the immutable result without
// rescanning pipeline IR or rediscovering SSA relationships.

#ifndef LOOM_ANALYSIS_PIPELINE_PLAN_H_
#define LOOM_ANALYSIS_PIPELINE_PLAN_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/combining.h"
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

typedef struct loom_pipeline_plan_binding_view_t {
  // Full refined binding tile type before selecting record dimensions.
  loom_type_t binding_type;

  // Exact byte offset from the launch binding base.
  uint64_t byte_offset;
} loom_pipeline_plan_binding_view_t;

typedef struct loom_pipeline_plan_group_t {
  // Source SSA identity naming the scheduling group.
  loom_value_id_t source_value;

  // Exact specialized lane count.
  uint32_t lane_count;

  // First callable instance belonging to the group.
  uint32_t instance_start;

  // Number of logical stages scheduled onto each group lane.
  uint32_t stage_count;
} loom_pipeline_plan_group_t;

typedef struct loom_pipeline_plan_stage_t {
  // Scheduling group executing this logical stage.
  uint32_t group_index;

  // Callable implementing one logical stage firing.
  loom_symbol_ref_t entry;

  // First entry in the plan-wide stage-port table.
  uint32_t port_start;

  // Number of input buffer ports in the callable ABI.
  uint16_t input_count;

  // Number of output buffer ports in the callable ABI.
  uint16_t output_count;

  // Number of source records folded into one output record, or zero when the
  // stage fires independently for every record.
  uint32_t fold_record_count;

  // Callable output port folded by the worker when fold_record_count is set.
  uint32_t fold_output_port;

  // Elementwise combining operation used by the temporal fold.
  loom_combining_kind_t fold_kind;

  // Floating-point permissions applied by the temporal fold.
  uint8_t fold_fast_math_flags;
} loom_pipeline_plan_stage_t;

typedef struct loom_pipeline_plan_stage_port_t {
  // Logical flow supplying or defined by this callable argument.
  uint32_t flow_index;

  // Exact producer lane for expanded reduction inputs, or UINT32_MAX for a
  // pointwise input or output.
  uint32_t source_lane;
} loom_pipeline_plan_stage_port_t;

typedef enum loom_pipeline_plan_group_port_direction_e {
  // Records enter the resident group through this port.
  LOOM_PIPELINE_PLAN_GROUP_PORT_DIRECTION_RECEIVE = 0,

  // Records leave the resident group through this port.
  LOOM_PIPELINE_PLAN_GROUP_PORT_DIRECTION_SEND = 1,
} loom_pipeline_plan_group_port_direction_t;

typedef struct loom_pipeline_plan_group_port_t {
  // Resident scheduling group owning this physical port.
  uint32_t group_index;

  // Logical flow carried by the physical port.
  uint32_t flow_index;

  // Exact producer lane selected by a reduction receive port, or UINT32_MAX
  // for pointwise receives and all send ports.
  uint32_t source_lane;

  // Dense physical port ordinal in the resident worker ABI.
  uint32_t port;

  // Direction of the physical port.
  loom_pipeline_plan_group_port_direction_t direction;
} loom_pipeline_plan_group_port_t;

typedef struct loom_pipeline_plan_instance_t {
  // Scheduling group containing this instance.
  uint32_t group_index;

  // Lane ordinal within the scheduling group.
  uint32_t lane;

  // Callable implementing one record firing, or null when the target must
  // compose several logical stages scheduled onto this physical instance.
  loom_symbol_ref_t entry;

  // Number of source records folded into one output record, or zero when the
  // worker fires independently for every record.
  uint32_t fold_record_count;

  // Callable output port folded by the worker when fold_record_count is set.
  uint32_t fold_output_port;

  // Elementwise combining operation used by the temporal fold.
  loom_combining_kind_t fold_kind;

  // Floating-point permissions applied by the temporal fold.
  uint8_t fold_fast_math_flags;
} loom_pipeline_plan_instance_t;

typedef struct loom_pipeline_plan_record_shape_t {
  // Exact outer-to-inner temporal dimensions of one lane's record sequence.
  const uint32_t* dimensions;

  // Number of dimensions in |dimensions|. Rank zero denotes one record.
  uint8_t rank;
} loom_pipeline_plan_record_shape_t;

typedef struct loom_pipeline_plan_flow_t {
  // Source SSA identity naming this flow version.
  loom_value_id_t source_value;

  // Tile record type transferred by the flow.
  loom_type_t tile_type;

  // Ordered temporal shape transferred per lane and activation.
  loom_pipeline_plan_record_shape_t record_shape;

  // Number of ordered records transferred per lane and activation.
  uint32_t record_count;

  // Scheduling group defining the flow lane cardinality.
  uint32_t group_index;

  // Minimum record capacity required by authored buffering.
  uint32_t minimum_capacity;

  // Canonical flow storage identity. Transparent flow refinements preserve
  // this index so same-group stages share one local buffer.
  uint32_t storage_flow_index;

  // Kind of endpoint producing this flow.
  loom_pipeline_endpoint_kind_t producer_kind;

  // Binding ordinal when |producer_kind| is BINDING.
  uint32_t binding_index;

  // First producer instance when |producer_kind| is INSTANCE.
  uint32_t instance_start;

  // Number of producer instances, or zero for a binding producer.
  uint32_t instance_count;

  // Logical producer stage when resident, or UINT32_MAX for a binding.
  uint32_t producer_stage_index;

  // Callable output argument ordinal on the logical producer stage.
  uint32_t producer_stage_port;

  // Callable output port or external binding port.
  uint32_t producer_port;

  // Binding-view table index when produced by an external binding, or
  // UINT32_MAX when produced by a resident instance.
  uint32_t binding_view_index;
} loom_pipeline_plan_flow_t;

typedef struct loom_pipeline_plan_edge_t {
  // Flow supplying the edge and its record contract.
  uint32_t flow_index;

  // Binding-view table index for the edge's external endpoint, or UINT32_MAX
  // when both endpoints are resident instances.
  uint32_t binding_view_index;

  // Kind of concrete producer endpoint.
  loom_pipeline_endpoint_kind_t source_kind;

  // Binding or instance index selected by |source_kind|.
  uint32_t source_index;

  // Producer endpoint port.
  uint32_t source_port;

  // Selected leading-dimension lane when the binding view is partitioned.
  uint32_t binding_view_lane;

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

  // Typed launch-binding views referenced by concrete edges.
  const loom_pipeline_plan_binding_view_t* binding_views;

  // Number of binding view records.
  uint32_t binding_view_count;

  // Scheduling groups in source definition order.
  const loom_pipeline_plan_group_t* groups;

  // Number of scheduling groups.
  uint32_t group_count;

  // Resident callable instances in group and lane order.
  const loom_pipeline_plan_instance_t* instances;

  // Number of resident callable instances.
  uint32_t instance_count;

  // Logical stages in source graph order.
  const loom_pipeline_plan_stage_t* stages;

  // Number of logical stages.
  uint32_t stage_count;

  // Contiguous callable-port flow mappings referenced by logical stages.
  const loom_pipeline_plan_stage_port_t* stage_ports;

  // Number of logical stage-port mappings.
  uint32_t stage_port_count;

  // Physical resident-worker boundary ports.
  const loom_pipeline_plan_group_port_t* group_ports;

  // Number of physical resident-worker boundary ports.
  uint32_t group_port_count;

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
