// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/kernel_async_legality.h"

#include "loom/analysis/movement.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
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

  // Result object receiving diagnostic and stream counters.
  loom_kernel_async_legality_result_t* result;

  // Caller-owned function-local value facts.
  loom_value_fact_table_t* fact_table;

  // Target-independent movement analysis for async token producers.
  loom_movement_analysis_t movement_analysis;

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
  // Group index owning this async endpoint.
  iree_host_size_t group_index;

  // Movement request recorded for the producer op.
  loom_movement_request_t request;
} loom_kernel_async_legality_endpoint_t;

typedef struct loom_kernel_async_legality_stream_t {
  // Groups committed in the current block, in program order.
  loom_kernel_async_legality_group_t* groups;

  // Number of committed groups in groups.
  iree_host_size_t count;

  // Async endpoints committed by group tokens, in program order.
  loom_kernel_async_legality_endpoint_t* endpoints;

  // Number of committed entries in endpoints.
  iree_host_size_t endpoint_count;
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
      state->fact_table, state->options->value_domain, state->options->arena,
      &state->movement_analysis));
  IREE_RETURN_IF_ERROR(
      loom_movement_analysis_analyze(&state->movement_analysis));
  state->movement_analysis_ready = true;
  return iree_ok_status();
}

static const loom_op_t* loom_kernel_async_legality_token_producer(
    loom_kernel_async_legality_state_t* state, loom_value_id_t token_id) {
  if (token_id >= state->module->values.count) {
    return NULL;
  }
  const loom_value_t* token_value = loom_module_value(state->module, token_id);
  if (loom_value_is_block_arg(token_value)) {
    return NULL;
  }
  return loom_value_def_op(token_value);
}

static void loom_kernel_async_legality_endpoint_region(
    const loom_movement_endpoint_t* endpoint, loom_view_region_t* out_region) {
  *out_region = (loom_view_region_t){
      .view_value_id = endpoint->value_id,
      .root_value_id = endpoint->root_value_id,
      .begin_byte_offset = endpoint->begin_byte_offset,
      .byte_length = endpoint->byte_length,
      .end_byte_offset = endpoint->end_byte_offset,
      .minimum_alignment = endpoint->minimum_alignment,
      .root_minimum_alignment = endpoint->root_minimum_alignment,
      .memory_space = endpoint->memory_space,
      .precision_flags = endpoint->precision_flags,
  };
}

static iree_status_t loom_kernel_async_legality_endpoints_overlap(
    loom_kernel_async_legality_state_t* state,
    const loom_movement_endpoint_t* pending_dest,
    const loom_view_region_t* access_region, bool* out_overlap) {
  *out_overlap = false;
  if (pending_dest->kind != LOOM_MOVEMENT_ENDPOINT_VIEW || !access_region) {
    *out_overlap = true;
    return iree_ok_status();
  }
  if (pending_dest->root_value_id == LOOM_VALUE_ID_INVALID ||
      access_region->root_value_id == LOOM_VALUE_ID_INVALID) {
    *out_overlap = true;
    return iree_ok_status();
  }
  if (pending_dest->root_value_id != access_region->root_value_id) {
    return iree_ok_status();
  }

  loom_view_region_t pending_region = {0};
  loom_kernel_async_legality_endpoint_region(pending_dest, &pending_region);
  bool no_overlap = false;
  IREE_RETURN_IF_ERROR(loom_view_regions_prove_no_overlap(
      &state->movement_analysis.view_regions, &pending_region, access_region,
      &no_overlap));
  *out_overlap = !no_overlap;
  return iree_ok_status();
}

static iree_status_t loom_kernel_async_legality_pending_dest_overlaps(
    loom_kernel_async_legality_state_t* state,
    const loom_kernel_async_legality_stream_t* stream,
    const loom_view_region_t* access_region, bool* out_overlaps) {
  *out_overlaps = false;
  for (iree_host_size_t i = 0; i < stream->endpoint_count; ++i) {
    const loom_kernel_async_legality_endpoint_t* endpoint =
        &stream->endpoints[i];
    if (stream->groups[endpoint->group_index].completed) {
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

static iree_status_t loom_kernel_async_legality_check_op_memory_accesses(
    loom_kernel_async_legality_state_t* state,
    const loom_kernel_async_legality_stream_t* stream, const loom_op_t* op) {
  if (stream->endpoint_count == 0 || loom_movement_op_kind_is_async(op->kind)) {
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
    if (state->failed) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_async_legality_append_endpoint(
    loom_kernel_async_legality_state_t* state,
    loom_kernel_async_legality_stream_t* stream, const loom_op_t* group_op,
    iree_host_size_t group_index, const loom_op_t* producer_op) {
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
        state, group_op, diagnostic.rejection_bits);
  }
  if (!iree_all_bits_set(request.flags, LOOM_MOVEMENT_REQUEST_ASYNC) ||
      request.dest.kind != LOOM_MOVEMENT_ENDPOINT_VIEW) {
    return loom_kernel_async_legality_fail(state, group_op,
                                           LOOM_ERR_LOWERING_025);
  }

  loom_view_region_t dest_region = {0};
  loom_kernel_async_legality_endpoint_region(&request.dest, &dest_region);
  bool overlaps = false;
  IREE_RETURN_IF_ERROR(loom_kernel_async_legality_pending_dest_overlaps(
      state, stream, &dest_region, &overlaps));
  if (overlaps) {
    return loom_kernel_async_legality_fail(state, group_op,
                                           LOOM_ERR_LOWERING_026);
  }

  stream->endpoints[stream->endpoint_count++] =
      (loom_kernel_async_legality_endpoint_t){
          .group_index = group_index,
          .request = request,
      };
  return iree_ok_status();
}

static iree_status_t loom_kernel_async_legality_append_group_endpoints(
    loom_kernel_async_legality_state_t* state,
    loom_kernel_async_legality_stream_t* stream, const loom_op_t* op,
    iree_host_size_t group_index) {
  loom_value_slice_t tokens = loom_kernel_async_group_tokens(op);
  for (uint16_t i = 0; i < tokens.count; ++i) {
    const loom_op_t* producer_op = loom_kernel_async_legality_token_producer(
        state, loom_value_slice_get(tokens, i));
    if (!producer_op) {
      return loom_kernel_async_legality_fail(state, op, LOOM_ERR_LOWERING_033);
    }
    IREE_RETURN_IF_ERROR(loom_kernel_async_legality_append_endpoint(
        state, stream, op, group_index, producer_op));
    if (state->failed) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
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
  const iree_host_size_t group_index = stream->count++;
  stream->groups[group_index] = (loom_kernel_async_legality_group_t){
      .group_id = group_id,
      .group_op = op,
      .completed = false,
  };
  IREE_RETURN_IF_ERROR(loom_kernel_async_legality_append_group_endpoints(
      state, stream, op, group_index));
  if (state->failed) {
    return iree_ok_status();
  }
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

static iree_host_size_t loom_kernel_async_legality_block_endpoint_capacity(
    loom_block_t* block) {
  iree_host_size_t endpoint_capacity = 0;
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (loom_kernel_async_group_isa(op)) {
      endpoint_capacity += loom_kernel_async_group_tokens(op).count;
    }
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
  if (group_capacity == 0 && !has_wait) {
    return iree_ok_status();
  }

  loom_kernel_async_legality_stream_t stream = {0};
  if (group_capacity > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->options->arena, group_capacity,
        sizeof(loom_kernel_async_legality_group_t), (void**)&stream.groups));
  }
  const iree_host_size_t endpoint_capacity =
      loom_kernel_async_legality_block_endpoint_capacity(block);
  if (endpoint_capacity > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->options->arena, endpoint_capacity,
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
    } else {
      IREE_RETURN_IF_ERROR(loom_kernel_async_legality_check_op_memory_accesses(
          state, &stream, op));
    }
    if (state->failed) {
      return iree_ok_status();
    }
  }

  loom_kernel_async_legality_add_blocks_checked(state, 1);
  return loom_kernel_async_legality_check_uncompleted_groups(state, &stream);
}

static iree_status_t loom_kernel_async_legality_check_regions(
    loom_kernel_async_legality_state_t* state, loom_region_t* root_region) {
  loom_kernel_async_legality_region_worklist_t worklist;
  IREE_RETURN_IF_ERROR(loom_kernel_async_legality_region_worklist_initialize(
      state->options->arena, &worklist));
  IREE_RETURN_IF_ERROR(loom_kernel_async_legality_region_worklist_push(
      state->options->arena, &worklist, root_region));

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
              state->options->arena, &worklist, regions[i]));
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

  loom_kernel_async_legality_state_t state = {
      .module = module,
      .function = function,
      .options = options,
      .result = out_result,
      .fact_table = options->fact_table,
  };
  return loom_kernel_async_legality_check_regions(&state, body);
}
