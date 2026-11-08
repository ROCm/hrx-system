// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdlib.h>

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

typedef struct loom_aie2p_native_join_scope_t {
  // Canonical execution schedule row retained for boundary lookup.
  iree_host_size_t ordinal;
  // Membership by physical engine, independent of representative worker IDs.
  bool* participants;
  // Number of engines in the membership row.
  iree_host_size_t count;
} loom_aie2p_native_join_scope_t;

static int loom_aie2p_native_join_scope_compare(const void* lhs,
                                                const void* rhs) {
  const loom_aie2p_native_join_scope_t* a = lhs;
  const loom_aie2p_native_join_scope_t* b = rhs;
  return memcmp(a->participants, b->participants,
                a->count * sizeof(*a->participants));
}

iree_status_t loom_aie2p_native_select_joins(
    loom_aie2p_native_context_t* context, bool* out_valid) {
  *out_valid = true;
  const loom_pipeline_execution_t* execution = &context->execution;
  if (!execution->schedule_count) {
    return iree_ok_status();
  }
  iree_arena_allocator_t* arena = context->pass->arena;
  loom_aie2p_native_join_scope_t* scopes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, execution->schedule_count, sizeof(*scopes), (void**)&scopes));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, execution->schedule_count, sizeof(*context->joins),
      (void**)&context->joins));
  bool* participants = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, execution->schedule_count * execution->engine_count,
      sizeof(*participants), (void**)&participants));
  for (iree_host_size_t i = 0; i < execution->schedule_count; ++i) {
    bool* row = &participants[i * execution->engine_count];
    for (iree_host_size_t e = 0; e < execution->engine_count; ++e) {
      row[e] = execution->schedules[i].workers[e] != IREE_HOST_SIZE_MAX;
    }
    scopes[i] = (loom_aie2p_native_join_scope_t){
        .ordinal = i, .participants = row, .count = execution->engine_count};
  }
  qsort(scopes, execution->schedule_count, sizeof(*scopes),
        loom_aie2p_native_join_scope_compare);
  uint16_t* allocated_locks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, execution->engine_count,
                                                 sizeof(*allocated_locks),
                                                 (void**)&allocated_locks));
  for (iree_host_size_t i = 0; i < execution->engine_count; ++i) {
    allocated_locks[i] = context->tiles[i].next_lock;
  }
  for (iree_host_size_t i = 0; i < execution->schedule_count; ++i) {
    if (i &&
        !loom_aie2p_native_join_scope_compare(&scopes[i - 1], &scopes[i])) {
      // All members execute the same nested boundary sequence. Separate
      // arrival/departure credits make this tree reusable across those visits.
      context->joins[scopes[i].ordinal] = context->joins[scopes[i - 1].ordinal];
      continue;
    }
    loom_aie2p_native_join_t* join = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, sizeof(*join), (void**)&join));
    *join = (loom_aie2p_native_join_t){.next = context->first_join};
    IREE_RETURN_IF_ERROR(loom_aie2p_join_plan_build(
        context->family, scopes[i].participants, allocated_locks, arena,
        &join->plan, out_valid));
    if (!*out_valid) {
      return loom_aie2p_native_reject(
          context,
          execution->schedules[scopes[i].ordinal].source->source->declaration,
          IREE_SV("a connected phase lock graph with two available semaphores "
                  "per edge"));
    }
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, execution->engine_count, sizeof(*join->functions),
        (void**)&join->functions));
    for (iree_host_size_t e = 0; e < execution->engine_count; ++e) {
      join->functions[e] =
          (loom_aie2p_native_join_function_t){.symbol = loom_symbol_ref_null()};
    }
    context->first_join = context->joins[scopes[i].ordinal] = join;
  }
  for (iree_host_size_t i = 0; i < execution->engine_count; ++i) {
    context->tiles[i].next_lock = allocated_locks[i];
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_join_boundary(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    iree_host_size_t engine, const loom_pipeline_invocation_schedule_t* scope) {
  const loom_pipeline_execution_schedule_t* schedule =
      loom_pipeline_execution_schedule(&context->execution, scope);
  loom_aie2p_native_join_t* join =
      context->joins[schedule - context->execution.schedules];
  const loom_aie2p_join_program_t* program = &join->plan.programs[engine];
  if (!program->count) {
    return iree_ok_status();
  }
  loom_aie2p_native_join_function_t* emitted = &join->functions[engine];
  loom_symbol_ref_t* symbol = &emitted->symbol;
  if (!loom_symbol_ref_is_valid(*symbol)) {
    loom_aie2p_worker_builder_t* code = &context->code;
    code->worker_index = context->execution.workers[engine];
    loom_builder_t helper;
    loom_op_t* function = NULL;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(code, NULL, 0, NULL, 0,
                                                  &helper, symbol, &function));
    for (iree_host_size_t i = 0; i < program->count; ++i) {
      const loom_aie2p_join_action_t* action = &program->actions[i];
      IREE_RETURN_IF_ERROR(loom_aie2p_worker_lock(
          code, &helper,
          action->acquire ? AIE2P_CORE_DESCRIPTOR_REF_LOCK_ACQUIRE_IMMEDIATE
                          : AIE2P_CORE_DESCRIPTOR_REF_LOCK_RELEASE_IMMEDIATE,
          action->selector, action->acquire ? -1 : 1));
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_return(&helper, NULL, 0));
  }
  if (loom_aie2p_join_program_prefers_sharing(program, ++emitted->use_count)) {
    const loom_func_like_t callable = loom_func_like_cast(
        context->code.module,
        context->code.module->symbols.entries[symbol->symbol_id].defining_op);
    loom_op_attrs(callable.op)[callable.vtable->inline_policy_attr_index] =
        loom_attr_enum(LOOM_INLINE_POLICY_NOINLINE);
  }
  loom_op_t* call = NULL;
  return loom_low_invoke_build(builder, 0, 0, 0, 0, *symbol, NULL, 0, NULL, 0,
                               NULL, 0, LOOM_LOCATION_UNKNOWN, &call);
}

static uint32_t loom_aie2p_native_endpoint_register(
    const loom_aie2p_native_context_t* context,
    const loom_aie2p_native_distribution_endpoint_t* endpoint,
    loom_xdna_register_field_id_t field, uint16_t index) {
  const loom_xdna_tile_coordinate_t tile = endpoint->tile->coordinate;
  const uint64_t origin = (uint64_t)tile.column
                              << context->family->column_shift |
                          (uint64_t)tile.row << context->family->row_shift;
  return (uint32_t)(loom_xdna_register_field_address(context->family, field,
                                                     tile, &index) -
                    origin);
}

static iree_status_t loom_aie2p_native_endpoint_start(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    const loom_aie2p_native_distribution_command_t* command, uint8_t packet) {
  const loom_aie2p_native_distribution_endpoint_t* endpoint = command->endpoint;
  const loom_aie2p_native_memory_layout_t* layout =
      endpoint->tile->memory_layout;
  const loom_aie2p_native_dma_sequence_t* sequence = &endpoint->sequence;
  if (sequence->rewind_iteration) {
    const uint32_t address = loom_aie2p_native_endpoint_register(
        context, endpoint, layout->iteration.current, sequence->descriptor);
    loom_value_id_t word;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
        &context->code, builder, sequence->words[0][layout->iteration_word],
        &word));
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
        &context->code, builder, packet, address, &word, 1));
  }
  const uint32_t address = loom_aie2p_native_endpoint_register(
      context, endpoint, layout->directions[command->direction - 1].start,
      endpoint->engine);
  for (uint8_t i = 0; i < sequence->task_count; ++i) {
    loom_value_id_t task;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(&context->code, builder,
                                                    sequence->tasks[i], &task));
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
        &context->code, builder, packet, address, &task, 1));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_endpoint_retire_body(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    loom_region_t* region,
    const loom_aie2p_native_distribution_command_t* command, uint8_t packet,
    uint8_t reply) {
  const loom_aie2p_native_distribution_endpoint_t* endpoint = command->endpoint;
  const loom_aie2p_native_queue_status_t status =
      loom_aie2p_native_queue_status(context, endpoint->tile, endpoint->engine,
                                     command->direction);
  const uint64_t origin = (uint64_t)endpoint->tile->coordinate.column
                              << context->family->column_shift |
                          (uint64_t)endpoint->tile->coordinate.row
                              << context->family->row_shift;
  return loom_aie2p_worker_control_wait_clear(
      &context->code, builder, region, packet, reply,
      (uint32_t)(status.address - origin), status.pending_mask);
}

static iree_status_t loom_aie2p_native_endpoint_retire(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    loom_region_t* region,
    const loom_aie2p_native_distribution_command_t* command, uint8_t packet,
    uint8_t reply) {
  loom_aie2p_native_retirement_t* group = command->retirement;
  if (group->use_count == 1) {
    return loom_aie2p_native_endpoint_retire_body(context, builder, region,
                                                  command, packet, reply);
  }
  if (!loom_symbol_ref_is_valid(group->function)) {
    loom_builder_t helper;
    loom_op_t* function;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(&context->code, NULL, 0, NULL,
                                                  0, &helper, &group->function,
                                                  &function));
    const loom_func_like_t callable =
        loom_func_like_cast(context->code.module, function);
    loom_op_attrs(function)[callable.vtable->inline_policy_attr_index] =
        loom_attr_enum(LOOM_INLINE_POLICY_NOINLINE);
    IREE_RETURN_IF_ERROR(loom_aie2p_native_endpoint_retire_body(
        context, &helper, loom_low_func_def_body(function), group->source,
        packet, reply));
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_return(&helper, NULL, 0));
  }
  loom_op_t* call;
  return loom_low_func_call_build(builder, 0, 0, 0, 0, group->function, NULL, 0,
                                  NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN,
                                  &call);
}

iree_status_t loom_aie2p_native_distribution_boundary(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    iree_host_size_t worker_index, const loom_pipeline_invocation_t* invocation,
    bool before) {
  const loom_aie2p_native_worker_t* worker = &context->workers[worker_index];
  if (!worker->distribution_commands) {
    return iree_ok_status();
  }
  const loom_aie2p_native_worker_t* owner =
      &context->workers[context->execution
                            .workers[context->core_engines[worker_index]]];
  loom_aie2p_worker_builder_t* code = &context->code;
  code->worker_index = worker_index;
  loom_builder_t helper;
  loom_symbol_ref_t symbol;
  loom_op_t* function;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(code, NULL, 0, NULL, 0, &helper,
                                                &symbol, &function));
  for (const loom_aie2p_native_distribution_command_t* command =
           worker->distribution_commands;
       command; command = command->next) {
    if (command->invocation != invocation) {
      continue;
    }
    const iree_host_size_t tile = command->endpoint->tile - context->tiles;
    if (before) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_endpoint_start(
          context, &helper, command, owner->control_packets[tile]));
    } else {
      // All async groups in the child have retired before its return. No
      // distribution task emits tokens, so only this explicit register-read
      // response can reach the scalar stream until the next child starts.
      IREE_RETURN_IF_ERROR(loom_aie2p_native_endpoint_retire(
          context, &helper, loom_low_func_def_body(function), command,
          owner->control_packets[tile], owner->control_replies[tile]));
    }
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_return(&helper, NULL, 0));
  loom_op_t* call = NULL;
  return loom_low_invoke_build(builder, 0, 0, 0, 0, symbol, NULL, 0, NULL, 0,
                               NULL, 0, LOOM_LOCATION_UNKNOWN, &call);
}
