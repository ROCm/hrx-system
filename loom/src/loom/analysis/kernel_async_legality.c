// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/kernel_async_legality.h"

#include "iree/base/internal/arena.h"
#include "loom/analysis/control_uniformity.h"
#include "loom/analysis/movement.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// State and helpers
//===----------------------------------------------------------------------===//

typedef struct loom_kernel_async_legality_state_t {
  // Module whose function body is being checked.
  const loom_module_t* module;

  // Function-like symbol whose body is being checked.
  loom_func_like_t function;

  // Caller-owned analysis options.
  const loom_kernel_async_legality_options_t* options;

  // Call-scoped arena owning all legality analysis storage.
  iree_arena_allocator_t* arena;

  // Result object receiving diagnostic and stream counters.
  loom_kernel_async_legality_result_t* result;

  // Caller-owned function-local value facts.
  loom_value_fact_table_t* fact_table;

  // Target-independent movement analysis for async token producers.
  loom_movement_analysis_t movement_analysis;

  // Scope-aware control analysis for cluster collectives.
  loom_control_uniformity_info_t control_uniformity;

  // True once fact_table and movement_analysis have been initialized.
  bool movement_analysis_ready;

  // True once a user legality diagnostic was emitted and the walk should stop.
  bool failed;
} loom_kernel_async_legality_state_t;

typedef struct loom_kernel_async_legality_group_t {
  // SSA value produced by kernel.async.group.
  loom_value_id_t group_id;

  // Op that committed the group into the current straight-line stream.
  const loom_op_t* group_op;

  // True once a wait has completed this group directly or indirectly.
  bool completed;
} loom_kernel_async_legality_group_t;

typedef struct loom_kernel_async_legality_endpoint_t {
  // Group index owning this endpoint, or IREE_HOST_SIZE_MAX before commit.
  iree_host_size_t group_index;

  // Token produced by producer_op and consumed by the owning group.
  loom_value_id_t token_id;

  // Async transfer op that started the endpoint lifetime.
  const loom_op_t* producer_op;

  // Movement request recorded for the producer op.
  loom_movement_request_t request;
} loom_kernel_async_legality_endpoint_t;

typedef struct loom_kernel_async_legality_stream_t {
  // Groups committed in the current block, in program order.
  loom_kernel_async_legality_group_t* groups;

  // Number of committed groups in groups.
  iree_host_size_t count;

  // Async endpoints issued by transfer ops, in program order.
  loom_kernel_async_legality_endpoint_t* endpoints;

  // Number of issued entries in endpoints.
  iree_host_size_t endpoint_count;

  // Prefix of endpoints committed to groups.
  iree_host_size_t committed_endpoint_count;
} loom_kernel_async_legality_stream_t;

#define LOOM_KERNEL_ASYNC_LEGALITY_INITIAL_REGION_CAPACITY 16

typedef struct loom_kernel_async_legality_region_worklist_t {
  // Region pointers waiting to be checked.
  loom_region_t** regions;

  // Index of the next queued region to check.
  iree_host_size_t next_index;

  // Number of queued region pointers.
  iree_host_size_t count;

  // Allocated region pointer capacity.
  iree_host_size_t capacity;
} loom_kernel_async_legality_region_worklist_t;

static iree_status_t loom_kernel_async_legality_region_worklist_initialize(
    iree_arena_allocator_t* arena,
    loom_kernel_async_legality_region_worklist_t* worklist) {
  worklist->next_index = 0;
  worklist->count = 0;
  worklist->capacity = LOOM_KERNEL_ASYNC_LEGALITY_INITIAL_REGION_CAPACITY;
  return iree_arena_allocate_array(arena, worklist->capacity,
                                   sizeof(loom_region_t*),
                                   (void**)&worklist->regions);
}

static iree_status_t loom_kernel_async_legality_region_worklist_push(
    iree_arena_allocator_t* arena,
    loom_kernel_async_legality_region_worklist_t* worklist,
    loom_region_t* region) {
  if (!region || region->block_count == 0) {
    return iree_ok_status();
  }
  if (worklist->count >= worklist->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, worklist->count, worklist->count + 1, sizeof(loom_region_t*),
        &worklist->capacity, (void**)&worklist->regions));
  }
  worklist->regions[worklist->count++] = region;
  return iree_ok_status();
}

static loom_region_t* loom_kernel_async_legality_region_worklist_pop(
    loom_kernel_async_legality_region_worklist_t* worklist) {
  if (worklist->next_index >= worklist->count) {
    return NULL;
  }
  return worklist->regions[worklist->next_index++];
}

static void loom_kernel_async_legality_add_blocks_checked(
    loom_kernel_async_legality_state_t* state, uint64_t delta) {
  state->result->blocks_checked += delta;
}

static void loom_kernel_async_legality_add_groups_checked(
    loom_kernel_async_legality_state_t* state, uint64_t delta) {
  state->result->groups_checked += delta;
}

static void loom_kernel_async_legality_add_waits_checked(
    loom_kernel_async_legality_state_t* state, uint64_t delta) {
  state->result->waits_checked += delta;
}

static iree_string_view_t loom_kernel_async_legality_op_name(
    const loom_module_t* module, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable) {
    return IREE_SV("<unknown>");
  }
  return loom_op_vtable_name(vtable);
}

static iree_string_view_t loom_kernel_async_legality_phase_name(
    const loom_kernel_async_legality_state_t* state) {
  if (!iree_string_view_is_empty(state->options->phase_name)) {
    return state->options->phase_name;
  }
  return IREE_SV("kernel-async-legality");
}

static iree_status_t loom_kernel_async_legality_emit(
    loom_kernel_async_legality_state_t* state, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  state->failed = true;
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  ++state->result->error_count;
  return iree_diagnostic_emit(state->options->emitter, &emission);
}

static iree_status_t loom_kernel_async_legality_fail(
    loom_kernel_async_legality_state_t* state, const loom_op_t* op,
    const loom_error_def_t* error) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_kernel_async_legality_op_name(state->module, op)),
      loom_param_string(loom_kernel_async_legality_phase_name(state)),
  };
  return loom_kernel_async_legality_emit(state, op, error, params,
                                         IREE_ARRAYSIZE(params));
}

static iree_status_t loom_kernel_async_legality_fail_movement_rejection(
    loom_kernel_async_legality_state_t* state, const loom_op_t* op,
    uint64_t rejection_bits) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_kernel_async_legality_op_name(state->module, op)),
      loom_param_string(loom_kernel_async_legality_phase_name(state)),
      loom_param_u64(rejection_bits),
  };
  return loom_kernel_async_legality_emit(state, op, LOOM_ERR_LOWERING_024,
                                         params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_kernel_async_legality_fail_newer_groups(
    loom_kernel_async_legality_state_t* state, const loom_op_t* op,
    int64_t actual_newer_groups, iree_host_size_t expected_newer_groups) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_kernel_async_legality_op_name(state->module, op)),
      loom_param_string(loom_kernel_async_legality_phase_name(state)),
      loom_param_i64(actual_newer_groups),
      loom_param_u64((uint64_t)expected_newer_groups),
  };
  return loom_kernel_async_legality_emit(state, op, LOOM_ERR_LOWERING_031,
                                         params, IREE_ARRAYSIZE(params));
}

static bool loom_kernel_async_legality_use_is_wait(const loom_use_t* use) {
  return loom_kernel_async_wait_isa(loom_use_user_op(*use));
}

static iree_status_t loom_kernel_async_legality_check_group_uses(
    loom_kernel_async_legality_state_t* state, const loom_op_t* op,
    loom_value_id_t group_id) {
  const loom_value_t* group_value = loom_module_value(state->module, group_id);
  if (group_value->use_count == 0) {
    return loom_kernel_async_legality_fail(state, op, LOOM_ERR_LOWERING_022);
  }
  const loom_use_t* use = NULL;
  loom_value_for_each_use(group_value, use) {
    if (!loom_kernel_async_legality_use_is_wait(use)) {
      return loom_kernel_async_legality_fail(state, op, LOOM_ERR_LOWERING_023);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_async_legality_ensure_movement_analysis(
    loom_kernel_async_legality_state_t* state) {
  if (state->movement_analysis_ready) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_movement_analysis_initialize(
      state->fact_table, state->options->value_domain, state->arena,
      &state->movement_analysis));
  IREE_RETURN_IF_ERROR(
      loom_movement_analysis_analyze(&state->movement_analysis));
  state->movement_analysis_ready = true;
  return iree_ok_status();
}

static bool loom_kernel_async_legality_exact_value(
    const loom_kernel_async_legality_state_t* state, loom_value_id_t value_id,
    int64_t expected_value) {
  if (value_id == LOOM_VALUE_ID_INVALID) return false;
  int64_t actual_value = 0;
  return loom_value_facts_as_exact_i64(
             loom_value_fact_table_lookup(state->fact_table, value_id),
             &actual_value) &&
         actual_value == expected_value;
}

static bool loom_kernel_async_legality_cluster_shape(
    const loom_kernel_async_legality_state_t* state,
    loom_target_workgroup_cluster_size_t* out_size, uint32_t* out_volume) {
  *out_size = (loom_target_workgroup_cluster_size_t){0};
  *out_volume = 0;
  if (!loom_kernel_def_isa(state->function.op) ||
      !loom_kernel_def_static_workgroup_cluster_size_from_facts(
          state->module, state->function.op, state->fact_table, out_size)) {
    return false;
  }

  // gfx1250 encodes at most sixteen participant ranks in the semantic set.
  // Reject each dimension before multiplication so malformed large extents
  // cannot overflow while proving the bounded product.
  if (out_size->x > 16 || out_size->y > 16 || out_size->z > 16) return false;
  const uint32_t volume = out_size->x * out_size->y * out_size->z;
  if (volume <= 1 || volume > 16) return false;
  *out_volume = volume;
  return true;
}

static uint32_t loom_kernel_async_legality_cluster_axis_extent(
    const loom_target_workgroup_cluster_size_t* size,
    loom_value_fact_topology_axis_t axis) {
  switch (axis) {
    case LOOM_VALUE_FACT_TOPOLOGY_AXIS_X:
      return size->x;
    case LOOM_VALUE_FACT_TOPOLOGY_AXIS_Y:
      return size->y;
    case LOOM_VALUE_FACT_TOPOLOGY_AXIS_Z:
      return size->z;
    case LOOM_VALUE_FACT_TOPOLOGY_AXIS_LANE:
    case LOOM_VALUE_FACT_TOPOLOGY_AXIS_COUNT_:
      return 0;
  }
  return 0;
}

static bool loom_kernel_async_legality_cluster_expression_agrees(
    const loom_kernel_async_legality_state_t* state,
    const loom_target_workgroup_cluster_size_t* cluster_size,
    const loom_symbolic_expr_t* expression) {
  if (!loom_symbolic_expr_is_linear(expression)) return false;
  for (iree_host_size_t i = 0; i < expression->term_count; ++i) {
    const loom_symbolic_term_t* term = &expression->terms[i];
    const loom_value_id_t relation_value_id =
        term->relation_value_id != LOOM_VALUE_ID_INVALID
            ? term->relation_value_id
            : term->value_id;
    const loom_value_facts_t facts =
        loom_value_fact_table_lookup(state->fact_table, relation_value_id);
    if (loom_value_facts_is_uniform_at_scope(
            facts, LOOM_VALUE_FACT_UNIFORM_SCOPE_CLUSTER)) {
      continue;
    }

    const loom_value_fact_topology_domain_t* topology =
        loom_value_facts_topology_domain(facts);
    if (!topology) return false;
    switch (topology->value_kind) {
      case LOOM_VALUE_FACT_TOPOLOGY_VALUE_WORKITEM_ID:
      case LOOM_VALUE_FACT_TOPOLOGY_VALUE_SUBGROUP_LANE_ID:
        // Corresponding lanes have the same local coordinates in every
        // participating workgroup.
        continue;
      case LOOM_VALUE_FACT_TOPOLOGY_VALUE_WORKGROUP_ID:
      case LOOM_VALUE_FACT_TOPOLOGY_VALUE_CLUSTER_WORKGROUP_ID:
        // A workgroup coordinate agrees only on an unclustered axis.
        if (loom_kernel_async_legality_cluster_axis_extent(
                cluster_size, topology->axis) == 1) {
          continue;
        }
        return false;
      case LOOM_VALUE_FACT_TOPOLOGY_VALUE_CLUSTER_ID:
      case LOOM_VALUE_FACT_TOPOLOGY_VALUE_NONE:
      case LOOM_VALUE_FACT_TOPOLOGY_VALUE_COUNT_:
        return false;
    }
  }
  return true;
}

static iree_status_t loom_kernel_async_legality_check_cluster_request(
    loom_kernel_async_legality_state_t* state, const loom_op_t* producer_op,
    const loom_movement_request_t* request) {
  if (request->layout_kind != LOOM_MOVEMENT_LAYOUT_CLUSTER_GATHER) {
    return iree_ok_status();
  }

  loom_target_workgroup_cluster_size_t cluster_size = {0};
  uint32_t cluster_volume = 0;
  if (!loom_kernel_async_legality_cluster_shape(state, &cluster_size,
                                                &cluster_volume)) {
    return loom_kernel_async_legality_fail_movement_rejection(
        state, producer_op, LOOM_MOVEMENT_REJECTION_CLUSTER_SHAPE);
  }

  const uint32_t expected_participants = (UINT32_C(1) << cluster_volume) - 1;
  if (!loom_kernel_async_legality_exact_value(
          state, request->cluster_mask_value_id, expected_participants)) {
    return loom_kernel_async_legality_fail_movement_rejection(
        state, producer_op, LOOM_MOVEMENT_REJECTION_CLUSTER_PARTICIPANTS);
  }
  if (iree_any_bit_set(request->flags, LOOM_MOVEMENT_REQUEST_MASKED) &&
      !loom_kernel_async_legality_exact_value(state, request->mask_value_id,
                                              1)) {
    return loom_kernel_async_legality_fail_movement_rejection(
        state, producer_op, LOOM_MOVEMENT_REJECTION_CLUSTER_PREDICATE);
  }

  bool cluster_uniform_execution = false;
  IREE_RETURN_IF_ERROR(loom_control_uniformity_prove_execution(
      &state->control_uniformity, producer_op,
      LOOM_VALUE_FACT_UNIFORM_SCOPE_CLUSTER, NULL, &cluster_uniform_execution));
  if (!cluster_uniform_execution) {
    return loom_kernel_async_legality_fail_movement_rejection(
        state, producer_op, LOOM_MOVEMENT_REJECTION_CLUSTER_CONTROL);
  }
  if (!loom_kernel_async_legality_cluster_expression_agrees(
          state, &cluster_size, &request->source.begin_byte_offset)) {
    return loom_kernel_async_legality_fail_movement_rejection(
        state, producer_op, LOOM_MOVEMENT_REJECTION_CLUSTER_SOURCE_AGREEMENT);
  }
  if (!loom_kernel_async_legality_cluster_expression_agrees(
          state, &cluster_size, &request->dest.begin_byte_offset)) {
    return loom_kernel_async_legality_fail_movement_rejection(
        state, producer_op, LOOM_MOVEMENT_REJECTION_CLUSTER_DEST_AGREEMENT);
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_async_legality_endpoints_overlap(
    loom_kernel_async_legality_state_t* state,
    const loom_movement_endpoint_t* pending_endpoint,
    const loom_view_region_t* access_region, bool* out_overlap) {
  *out_overlap = false;
  if (pending_endpoint->kind != LOOM_MOVEMENT_ENDPOINT_VIEW || !access_region) {
    *out_overlap = true;
    return iree_ok_status();
  }
  if (pending_endpoint->root_value_id == LOOM_VALUE_ID_INVALID ||
      access_region->root_value_id == LOOM_VALUE_ID_INVALID) {
    *out_overlap = true;
    return iree_ok_status();
  }
  loom_view_region_t pending_region = {0};
  if (!loom_movement_endpoint_as_view_region(pending_endpoint,
                                             &pending_region)) {
    *out_overlap = true;
    return iree_ok_status();
  }
  bool no_overlap = false;
  IREE_RETURN_IF_ERROR(loom_view_regions_prove_no_overlap(
      &state->movement_analysis.view_regions, &pending_region, access_region,
      &no_overlap));
  *out_overlap = !no_overlap;
  return iree_ok_status();
}

static bool loom_kernel_async_legality_endpoint_is_pending(
    const loom_kernel_async_legality_stream_t* stream,
    const loom_kernel_async_legality_endpoint_t* endpoint) {
  return endpoint->group_index == IREE_HOST_SIZE_MAX ||
         !stream->groups[endpoint->group_index].completed;
}

static iree_status_t loom_kernel_async_legality_pending_dest_overlaps(
    loom_kernel_async_legality_state_t* state,
    const loom_kernel_async_legality_stream_t* stream,
    const loom_view_region_t* access_region, bool* out_overlaps) {
  *out_overlaps = false;
  for (iree_host_size_t i = 0; i < stream->endpoint_count; ++i) {
    const loom_kernel_async_legality_endpoint_t* endpoint =
        &stream->endpoints[i];
    if (!loom_kernel_async_legality_endpoint_is_pending(stream, endpoint)) {
      continue;
    }

    bool overlaps = false;
    IREE_RETURN_IF_ERROR(loom_kernel_async_legality_endpoints_overlap(
        state, &endpoint->request.dest, access_region, &overlaps));
    if (!overlaps) {
      continue;
    }
    *out_overlaps = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_async_legality_pending_source_overlaps(
    loom_kernel_async_legality_state_t* state,
    const loom_kernel_async_legality_stream_t* stream,
    const loom_view_region_t* access_region, bool* out_overlaps) {
  *out_overlaps = false;
  for (iree_host_size_t i = 0; i < stream->endpoint_count; ++i) {
    const loom_kernel_async_legality_endpoint_t* endpoint =
        &stream->endpoints[i];
    if (!loom_kernel_async_legality_endpoint_is_pending(stream, endpoint)) {
      continue;
    }
    if (endpoint->request.source.kind != LOOM_MOVEMENT_ENDPOINT_VIEW) continue;

    bool overlaps = false;
    IREE_RETURN_IF_ERROR(loom_kernel_async_legality_endpoints_overlap(
        state, &endpoint->request.source, access_region, &overlaps));
    if (!overlaps) continue;
    *out_overlaps = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_async_legality_check_pending_dest_hazard(
    loom_kernel_async_legality_state_t* state,
    const loom_kernel_async_legality_stream_t* stream, const loom_op_t* op,
    const loom_view_region_t* access_region, loom_operand_flags_t flags) {
  bool overlaps = false;
  IREE_RETURN_IF_ERROR(loom_kernel_async_legality_pending_dest_overlaps(
      state, stream, access_region, &overlaps));
  if (overlaps) {
    if (iree_any_bit_set(flags, LOOM_OPERAND_WRITES)) {
      return loom_kernel_async_legality_fail(state, op, LOOM_ERR_LOWERING_027);
    }
    if (iree_any_bit_set(flags, LOOM_OPERAND_READS)) {
      return loom_kernel_async_legality_fail(state, op, LOOM_ERR_LOWERING_028);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_async_legality_check_pending_source_hazard(
    loom_kernel_async_legality_state_t* state,
    const loom_kernel_async_legality_stream_t* stream, const loom_op_t* op,
    const loom_view_region_t* access_region, loom_operand_flags_t flags) {
  if (!iree_any_bit_set(flags, LOOM_OPERAND_WRITES)) return iree_ok_status();
  bool overlaps = false;
  IREE_RETURN_IF_ERROR(loom_kernel_async_legality_pending_source_overlaps(
      state, stream, access_region, &overlaps));
  if (overlaps) {
    return loom_kernel_async_legality_fail(state, op, LOOM_ERR_LOWERING_037);
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_async_legality_check_op_memory_accesses(
    loom_kernel_async_legality_state_t* state,
    const loom_kernel_async_legality_stream_t* stream, const loom_op_t* op) {
  if (stream->endpoint_count == 0) {
    return iree_ok_status();
  }

  const loom_op_vtable_t* vtable = loom_op_vtable(state->module, op);
  if (!vtable || !vtable->operand_descriptors) {
    return iree_ok_status();
  }
  const uint16_t descriptor_count =
      op->operand_count < vtable->fixed_operand_count
          ? op->operand_count
          : vtable->fixed_operand_count;
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < descriptor_count; ++i) {
    const loom_operand_flags_t flags = vtable->operand_descriptors[i].flags;
    if (!iree_any_bit_set(flags, LOOM_OPERAND_READS | LOOM_OPERAND_WRITES)) {
      continue;
    }

    const loom_view_region_t* access_region = NULL;
    IREE_RETURN_IF_ERROR(loom_view_region_table_get(
        &state->movement_analysis.view_regions, operands[i], &access_region));
    if (!access_region) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_kernel_async_legality_check_pending_dest_hazard(
        state, stream, op, access_region, flags));
    if (state->failed) return iree_ok_status();
    IREE_RETURN_IF_ERROR(loom_kernel_async_legality_check_pending_source_hazard(
        state, stream, op, access_region, flags));
    if (state->failed) return iree_ok_status();
  }
  return iree_ok_status();
}

static bool loom_kernel_async_legality_transfer_token_use_is_local_group(
    const loom_kernel_async_legality_state_t* state,
    const loom_op_t* producer_op, loom_value_id_t token_id) {
  if (token_id >= state->module->values.count) return false;
  const loom_value_t* token_value = loom_module_value(state->module, token_id);
  if (token_value->use_count != 1) return false;
  const loom_use_t* use = NULL;
  loom_value_for_each_use(token_value, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    return loom_kernel_async_group_isa(user_op) &&
           user_op->parent_block == producer_op->parent_block;
  }
  return false;
}

static iree_status_t loom_kernel_async_legality_append_transfer(
    loom_kernel_async_legality_state_t* state,
    loom_kernel_async_legality_stream_t* stream, const loom_op_t* producer_op) {
  IREE_RETURN_IF_ERROR(
      loom_kernel_async_legality_ensure_movement_analysis(state));
  loom_movement_request_t request = {0};
  loom_movement_diagnostic_t diagnostic = {0};
  bool described = false;
  IREE_RETURN_IF_ERROR(
      loom_movement_request_describe_op(&state->movement_analysis, producer_op,
                                        &request, &diagnostic, &described));
  if (!described) {
    return loom_kernel_async_legality_fail_movement_rejection(
        state, producer_op, diagnostic.rejection_bits);
  }
  if (!iree_all_bits_set(request.flags, LOOM_MOVEMENT_REQUEST_ASYNC) ||
      request.dest.kind != LOOM_MOVEMENT_ENDPOINT_VIEW ||
      producer_op->result_count != 1) {
    return loom_kernel_async_legality_fail(state, producer_op,
                                           LOOM_ERR_LOWERING_025);
  }
  IREE_RETURN_IF_ERROR(loom_kernel_async_legality_check_cluster_request(
      state, producer_op, &request));
  if (state->failed) return iree_ok_status();

  const loom_value_id_t token_id = loom_op_const_results(producer_op)[0];
  if (!loom_kernel_async_legality_transfer_token_use_is_local_group(
          state, producer_op, token_id)) {
    return loom_kernel_async_legality_fail(state, producer_op,
                                           LOOM_ERR_LOWERING_040);
  }

  if (request.source.kind == LOOM_MOVEMENT_ENDPOINT_VIEW) {
    loom_view_region_t source_region = {0};
    if (!loom_movement_endpoint_as_view_region(&request.source,
                                               &source_region)) {
      return loom_kernel_async_legality_fail(state, producer_op,
                                             LOOM_ERR_LOWERING_025);
    }
    IREE_RETURN_IF_ERROR(loom_kernel_async_legality_check_pending_dest_hazard(
        state, stream, producer_op, &source_region, LOOM_OPERAND_READS));
    if (state->failed) return iree_ok_status();
  }

  loom_view_region_t dest_region = {0};
  if (!loom_movement_endpoint_as_view_region(&request.dest, &dest_region)) {
    return loom_kernel_async_legality_fail(state, producer_op,
                                           LOOM_ERR_LOWERING_025);
  }
  bool overlaps = false;
  IREE_RETURN_IF_ERROR(loom_kernel_async_legality_pending_dest_overlaps(
      state, stream, &dest_region, &overlaps));
  if (overlaps) {
    return loom_kernel_async_legality_fail(state, producer_op,
                                           LOOM_ERR_LOWERING_026);
  }
  IREE_RETURN_IF_ERROR(loom_kernel_async_legality_check_pending_source_hazard(
      state, stream, producer_op, &dest_region, LOOM_OPERAND_WRITES));
  if (state->failed) return iree_ok_status();

  stream->endpoints[stream->endpoint_count++] =
      (loom_kernel_async_legality_endpoint_t){
          .group_index = IREE_HOST_SIZE_MAX,
          .token_id = token_id,
          .producer_op = producer_op,
          .request = request,
      };
  return iree_ok_status();
}

static bool loom_kernel_async_legality_group_commits_pending_transfers(
    const loom_kernel_async_legality_stream_t* stream, const loom_op_t* op) {
  loom_value_slice_t tokens = loom_kernel_async_group_tokens(op);
  const iree_host_size_t pending_count =
      stream->endpoint_count - stream->committed_endpoint_count;
  if (tokens.count != pending_count) return false;
  for (uint16_t i = 0; i < tokens.count; ++i) {
    const loom_kernel_async_legality_endpoint_t* endpoint =
        &stream->endpoints[stream->committed_endpoint_count + i];
    if (loom_value_slice_get(tokens, i) != endpoint->token_id) return false;
  }
  return true;
}

static iree_status_t loom_kernel_async_legality_append_group(
    loom_kernel_async_legality_state_t* state,
    loom_kernel_async_legality_stream_t* stream, const loom_op_t* op) {
  loom_value_id_t group_id = loom_kernel_async_group_group(op);
  IREE_RETURN_IF_ERROR(
      loom_kernel_async_legality_check_group_uses(state, op, group_id));
  if (state->failed) {
    return iree_ok_status();
  }
  if (!loom_kernel_async_legality_group_commits_pending_transfers(stream, op)) {
    return loom_kernel_async_legality_fail(state, op, LOOM_ERR_LOWERING_038);
  }
  const iree_host_size_t group_index = stream->count++;
  stream->groups[group_index] = (loom_kernel_async_legality_group_t){
      .group_id = group_id,
      .group_op = op,
      .completed = false,
  };
  for (iree_host_size_t i = stream->committed_endpoint_count;
       i < stream->endpoint_count; ++i) {
    stream->endpoints[i].group_index = group_index;
  }
  stream->committed_endpoint_count = stream->endpoint_count;
  loom_kernel_async_legality_add_groups_checked(state, 1);
  return iree_ok_status();
}

static iree_host_size_t loom_kernel_async_legality_find_group(
    const loom_kernel_async_legality_stream_t* stream,
    loom_value_id_t group_id) {
  for (iree_host_size_t i = 0; i < stream->count; ++i) {
    if (stream->groups[i].group_id == group_id) {
      return i;
    }
  }
  return IREE_HOST_SIZE_MAX;
}

static iree_host_size_t loom_kernel_async_legality_newer_uncompleted_count(
    const loom_kernel_async_legality_stream_t* stream,
    iree_host_size_t group_index) {
  iree_host_size_t newer_groups = 0;
  for (iree_host_size_t i = group_index + 1; i < stream->count; ++i) {
    if (!stream->groups[i].completed) {
      ++newer_groups;
    }
  }
  return newer_groups;
}

static void loom_kernel_async_legality_complete_through(
    loom_kernel_async_legality_stream_t* stream, iree_host_size_t group_index) {
  for (iree_host_size_t i = 0; i <= group_index; ++i) {
    stream->groups[i].completed = true;
  }
}

static iree_status_t loom_kernel_async_legality_check_wait(
    loom_kernel_async_legality_state_t* state,
    loom_kernel_async_legality_stream_t* stream, const loom_op_t* op) {
  if (stream->committed_endpoint_count != stream->endpoint_count) {
    const loom_kernel_async_legality_endpoint_t* endpoint =
        &stream->endpoints[stream->committed_endpoint_count];
    return loom_kernel_async_legality_fail(state, endpoint->producer_op,
                                           LOOM_ERR_LOWERING_039);
  }
  loom_value_id_t group_id = loom_kernel_async_wait_group(op);
  iree_host_size_t group_index =
      loom_kernel_async_legality_find_group(stream, group_id);
  if (group_index == IREE_HOST_SIZE_MAX) {
    return loom_kernel_async_legality_fail(state, op, LOOM_ERR_LOWERING_029);
  }
  if (stream->groups[group_index].completed) {
    return loom_kernel_async_legality_fail(state, op, LOOM_ERR_LOWERING_030);
  }

  int64_t actual_newer_groups = loom_kernel_async_wait_newer_groups(op);
  iree_host_size_t expected_newer_groups =
      loom_kernel_async_legality_newer_uncompleted_count(stream, group_index);
  if ((uint64_t)actual_newer_groups != (uint64_t)expected_newer_groups) {
    return loom_kernel_async_legality_fail_newer_groups(
        state, op, actual_newer_groups, expected_newer_groups);
  }

  loom_kernel_async_legality_complete_through(stream, group_index);
  loom_kernel_async_legality_add_waits_checked(state, 1);
  return iree_ok_status();
}

static iree_status_t loom_kernel_async_legality_check_uncompleted_groups(
    loom_kernel_async_legality_state_t* state,
    const loom_kernel_async_legality_stream_t* stream) {
  for (iree_host_size_t i = 0; i < stream->count; ++i) {
    if (stream->groups[i].completed) {
      continue;
    }
    return loom_kernel_async_legality_fail(state, stream->groups[i].group_op,
                                           LOOM_ERR_LOWERING_032);
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_async_legality_check_uncommitted_transfers(
    loom_kernel_async_legality_state_t* state,
    const loom_kernel_async_legality_stream_t* stream) {
  if (stream->committed_endpoint_count == stream->endpoint_count) {
    return iree_ok_status();
  }
  const loom_kernel_async_legality_endpoint_t* endpoint =
      &stream->endpoints[stream->committed_endpoint_count];
  return loom_kernel_async_legality_fail(state, endpoint->producer_op,
                                         LOOM_ERR_LOWERING_039);
}

static iree_host_size_t loom_kernel_async_legality_block_endpoint_capacity(
    loom_block_t* block) {
  iree_host_size_t endpoint_capacity = 0;
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (loom_movement_op_kind_is_async(op->kind)) ++endpoint_capacity;
  }
  return endpoint_capacity;
}

static iree_status_t loom_kernel_async_legality_check_block(
    loom_kernel_async_legality_state_t* state, loom_block_t* block) {
  if (block->op_count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t group_capacity = 0;
  bool has_wait = false;
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (loom_kernel_async_group_isa(op)) {
      ++group_capacity;
    } else if (loom_kernel_async_wait_isa(op)) {
      has_wait = true;
    }
  }
  const iree_host_size_t endpoint_capacity =
      loom_kernel_async_legality_block_endpoint_capacity(block);
  if (group_capacity == 0 && !has_wait && endpoint_capacity == 0) {
    return iree_ok_status();
  }

  loom_kernel_async_legality_stream_t stream = {0};
  if (group_capacity > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, group_capacity,
        sizeof(loom_kernel_async_legality_group_t), (void**)&stream.groups));
  }
  if (endpoint_capacity > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->arena, endpoint_capacity,
                                  sizeof(loom_kernel_async_legality_endpoint_t),
                                  (void**)&stream.endpoints));
  }

  loom_block_for_each_op(block, op) {
    if (loom_kernel_async_group_isa(op)) {
      IREE_RETURN_IF_ERROR(
          loom_kernel_async_legality_append_group(state, &stream, op));
    } else if (loom_kernel_async_wait_isa(op)) {
      IREE_RETURN_IF_ERROR(
          loom_kernel_async_legality_check_wait(state, &stream, op));
    } else if (loom_movement_op_kind_is_async(op->kind)) {
      IREE_RETURN_IF_ERROR(
          loom_kernel_async_legality_append_transfer(state, &stream, op));
    } else {
      IREE_RETURN_IF_ERROR(loom_kernel_async_legality_check_op_memory_accesses(
          state, &stream, op));
    }
    if (state->failed) {
      return iree_ok_status();
    }
  }

  loom_kernel_async_legality_add_blocks_checked(state, 1);
  IREE_RETURN_IF_ERROR(
      loom_kernel_async_legality_check_uncommitted_transfers(state, &stream));
  if (state->failed) return iree_ok_status();
  return loom_kernel_async_legality_check_uncompleted_groups(state, &stream);
}

static iree_status_t loom_kernel_async_legality_check_regions(
    loom_kernel_async_legality_state_t* state, loom_region_t* root_region) {
  loom_kernel_async_legality_region_worklist_t worklist;
  IREE_RETURN_IF_ERROR(loom_kernel_async_legality_region_worklist_initialize(
      state->arena, &worklist));
  IREE_RETURN_IF_ERROR(loom_kernel_async_legality_region_worklist_push(
      state->arena, &worklist, root_region));

  loom_region_t* region = NULL;
  while ((region = loom_kernel_async_legality_region_worklist_pop(&worklist))) {
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      IREE_RETURN_IF_ERROR(
          loom_kernel_async_legality_check_block(state, block));
      if (state->failed) {
        return iree_ok_status();
      }
    }

    loom_region_for_each_block(region, block) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        loom_region_t** regions = loom_op_regions(op);
        for (uint8_t i = 0; i < op->region_count; ++i) {
          IREE_RETURN_IF_ERROR(loom_kernel_async_legality_region_worklist_push(
              state->arena, &worklist, regions[i]));
        }
      }
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

iree_status_t loom_kernel_async_legality_verify_function(
    const loom_module_t* module, loom_func_like_t function,
    const loom_kernel_async_legality_options_t* options,
    loom_kernel_async_legality_result_t* out_result) {
  *out_result = (loom_kernel_async_legality_result_t){0};

  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(options->value_domain->region, body);

  iree_arena_allocator_t arena;
  iree_arena_initialize(module->arena.block_pool, &arena);
  loom_kernel_async_legality_state_t state = {
      .module = module,
      .function = function,
      .options = options,
      .arena = &arena,
      .result = out_result,
      .fact_table = options->fact_table,
  };
  loom_control_uniformity_info_initialize(module, options->fact_table, &arena,
                                          &state.control_uniformity);
  iree_status_t status = loom_kernel_async_legality_check_regions(&state, body);
  iree_arena_deinitialize(&arena);
  return status;
}
