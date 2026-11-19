// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdlib.h>

#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

typedef struct loom_aie2p_native_join_scope_t {
  // Canonical execution schedule row retained for boundary lookup.
  iree_host_size_t ordinal;
  // Shared membership, compared independently of representative worker IDs.
  const loom_pipeline_participants_t* participants;
} loom_aie2p_native_join_scope_t;

static int loom_aie2p_native_join_scope_compare(const void* lhs,
                                                const void* rhs) {
  const loom_aie2p_native_join_scope_t* a = lhs;
  const loom_aie2p_native_join_scope_t* b = rhs;
  return loom_pipeline_participants_compare_members(a->participants,
                                                    b->participants);
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
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, execution->engine_count,
                                                 sizeof(*participants),
                                                 (void**)&participants));
  for (iree_host_size_t i = 0; i < execution->schedule_count; ++i) {
    scopes[i] = (loom_aie2p_native_join_scope_t){
        .ordinal = i, .participants = execution->schedules[i].participants};
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
    memset(participants, 0, execution->engine_count * sizeof(*participants));
    loom_pipeline_participant_iterator_t iterator;
    loom_pipeline_participants_begin(scopes[i].participants, &iterator);
    loom_pipeline_participant_t participant;
    while (loom_pipeline_participants_next(&iterator, &participant)) {
      participants[participant.engine] = true;
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_join_plan_build(
        context->family, participants, allocated_locks, arena, &join->plan,
        out_valid));
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

iree_status_t loom_aie2p_native_queue_i32(loom_builder_t* builder,
                                          loom_value_id_t value,
                                          loom_value_id_t* out_value) {
  const loom_type_t from = loom_module_value_type(builder->module, value);
  const loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_scalar_type_t scalar = loom_type_element_type(from);
  loom_op_t* cast;
  if (scalar == LOOM_SCALAR_TYPE_I32) {
    *out_value = value;
    return iree_ok_status();
  }
  if (!loom_scalar_type_is_integer(scalar)) {
    IREE_RETURN_IF_ERROR(loom_index_cast_build(builder, value, from, i32,
                                               LOOM_LOCATION_UNKNOWN, &cast));
  } else if (scalar == LOOM_SCALAR_TYPE_I64) {
    IREE_RETURN_IF_ERROR(loom_scalar_trunci_build(
        builder, value, from, i32, LOOM_LOCATION_UNKNOWN, &cast));
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_extui_build(builder, value, from, i32,
                                                 LOOM_LOCATION_UNKNOWN, &cast));
  }
  *out_value = loom_op_results(cast)[0];
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_skip_empty(loom_aie2p_worker_builder_t* code,
                                           loom_builder_t* builder,
                                           loom_op_t* function,
                                           loom_value_id_t count) {
  loom_region_t* body = loom_low_func_def_body(function);
  loom_block_t *active, *empty;
  IREE_RETURN_IF_ERROR(loom_region_append_block(code->module, body, &active));
  IREE_RETURN_IF_ERROR(loom_region_append_block(code->module, body, &empty));
  loom_op_t* branch;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(builder, count, active, empty,
                                              LOOM_LOCATION_UNKNOWN, &branch));
  loom_builder_set_block(builder, empty);
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_return(builder, NULL, 0));
  loom_builder_set_block(builder, active);
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_counted_task(
    loom_aie2p_worker_builder_t* code, loom_builder_t* builder,
    loom_value_id_t count, uint32_t start,
    loom_xdna_register_field_id_t repeat_field, loom_value_id_t* out_value) {
  loom_value_id_t one, encoded, shift, base;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(code, builder, 1, &one));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_binary(
      code, builder, AIE2P_CORE_DESCRIPTOR_REF_SUB_I32, count, one, &encoded));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
      code, builder,
      loom_xdna_register_field_info(repeat_field).least_significant_bit,
      &shift));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_binary(
      code, builder, AIE2P_CORE_DESCRIPTOR_REF_LSHL_I32, encoded, shift,
      &encoded));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(code, builder, start, &base));
  return loom_aie2p_worker_binary(code, builder,
                                  AIE2P_CORE_DESCRIPTOR_REF_OR_I32, base,
                                  encoded, out_value);
}

iree_status_t loom_aie2p_native_endpoint_start(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    const loom_aie2p_native_distribution_endpoint_t* endpoint,
    loom_xdna_dma_direction_t direction, uint8_t packet,
    loom_value_id_t count) {
  const loom_aie2p_native_memory_layout_t* layout =
      endpoint->tile->memory_layout;
  const loom_aie2p_native_dma_sequence_t* sequence = &endpoint->sequence;
  const uint32_t address = loom_aie2p_native_endpoint_register(
      context, endpoint, layout->directions[direction - 1].start,
      endpoint->engine);
  if (count != LOOM_VALUE_ID_INVALID) {
    loom_value_id_t task;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_counted_task(
        &context->code, builder, count,
        loom_xdna_register_field_encode(layout->directions[direction - 1].start,
                                        sequence->descriptor),
        layout->directions[direction - 1].repeat, &task));
    return loom_aie2p_worker_control_write(&context->code, builder, packet,
                                           address, &task, 1);
  }
  for (uint8_t i = 0; i < sequence->task_count; ++i) {
    loom_value_id_t task;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(&context->code, builder,
                                                    sequence->tasks[i], &task));
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
        &context->code, builder, packet, address, &task, 1));
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_queue_retire(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    loom_region_t* region, const loom_aie2p_native_tile_t* tile,
    uint16_t engine, loom_xdna_dma_direction_t direction, uint8_t packet,
    uint8_t reply) {
  const loom_aie2p_native_queue_status_t status =
      loom_aie2p_native_queue_status(context, tile, engine, direction);
  const uint64_t origin =
      (uint64_t)tile->coordinate.column << context->family->column_shift |
      (uint64_t)tile->coordinate.row << context->family->row_shift;
  return loom_aie2p_worker_control_wait_clear(
      &context->code, builder, region, packet, reply,
      (uint32_t)(status.address - origin), status.pending_mask);
}

static iree_status_t loom_aie2p_native_endpoint_retire_body(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    loom_region_t* region,
    const loom_aie2p_native_distribution_command_t* command, uint8_t packet,
    uint8_t reply) {
  const loom_aie2p_native_distribution_endpoint_t* endpoint = command->endpoint;
  return loom_aie2p_native_queue_retire(context, builder, region,
                                        endpoint->tile, endpoint->engine,
                                        command->direction, packet, reply);
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

static const loom_aie2p_native_distribution_command_t*
loom_aie2p_native_next_boundary_command(
    const loom_aie2p_native_distribution_command_t* command,
    const loom_pipeline_invocation_t* invocation, bool before) {
  while (command &&
         (command->invocation != invocation || (!before && !command->retire))) {
    command = command->next;
  }
  return command;
}

static bool loom_aie2p_native_same_repeat(
    const loom_pipeline_channel_repeat_t* lhs,
    const loom_pipeline_channel_repeat_t* rhs) {
  return lhs == rhs || (lhs && rhs && lhs->count == rhs->count);
}

iree_status_t loom_aie2p_native_distribution_boundary(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    iree_host_size_t worker_index, const loom_pipeline_invocation_t* invocation,
    bool before, loom_value_slice_t captures) {
  const loom_aie2p_native_worker_t* worker = &context->workers[worker_index];
  if (!worker->distribution_commands) {
    return iree_ok_status();
  }
  const loom_aie2p_native_worker_t* owner =
      &context->workers[context->execution
                            .workers[context->core_engines[worker_index]]];
  loom_aie2p_worker_builder_t* code = &context->code;
  code->worker_index = worker_index;
  const loom_aie2p_native_distribution_command_t* command =
      loom_aie2p_native_next_boundary_command(worker->distribution_commands,
                                              invocation, before);
  while (command) {
    // Keep adjacent writes using the same capture in one helper and preserve
    // task publication order when independently counted streams share a core.
    const loom_pipeline_channel_repeat_t* repeat =
        before ? command->endpoint->channel->repeat : NULL;
    loom_builder_t helper;
    loom_symbol_ref_t symbol;
    loom_op_t* function;
    const uint16_t arguments = repeat ? 1 : 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(code, &code->scalar_type,
                                                  arguments, NULL, 0, &helper,
                                                  &symbol, &function));
    loom_value_id_t count = LOOM_VALUE_ID_INVALID;
    loom_value_id_t actual = LOOM_VALUE_ID_INVALID;
    if (repeat) {
      count =
          loom_region_entry_block(loom_low_func_def_body(function))->arg_ids[0];
      const uint16_t formal =
          loom_value_def_index(loom_module_value(code->module, repeat->count));
      IREE_RETURN_IF_ERROR(loom_aie2p_native_queue_i32(
          builder, captures.values[formal], &actual));
      if (!repeat->minimum) {
        IREE_RETURN_IF_ERROR(
            loom_aie2p_native_skip_empty(code, &helper, function, count));
      }
    }
    do {
      const iree_host_size_t tile = command->endpoint->tile - context->tiles;
      if (before) {
        IREE_RETURN_IF_ERROR(loom_aie2p_native_endpoint_start(
            context, &helper, command->endpoint, command->direction,
            owner->control_packets[tile], count));
      } else {
        IREE_RETURN_IF_ERROR(loom_aie2p_native_endpoint_retire(
            context, &helper, loom_low_func_def_body(function), command,
            owner->control_packets[tile], owner->control_replies[tile]));
      }
      command = loom_aie2p_native_next_boundary_command(command->next,
                                                        invocation, before);
    } while (command &&
             (!before || loom_aie2p_native_same_repeat(
                             repeat, command->endpoint->channel->repeat)));
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_return(&helper, NULL, 0));
    loom_op_t* call;
    IREE_RETURN_IF_ERROR(
        loom_low_invoke_build(builder, 0, 0, 0, 0, symbol, &actual, arguments,
                              NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &call));
  }
  return iree_ok_status();
}
