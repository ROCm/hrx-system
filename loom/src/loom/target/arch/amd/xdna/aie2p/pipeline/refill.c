// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/refill.h"

#include <stdlib.h>

#include "loom/analysis/pipeline/progress.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/input.h"

static uint64_t loom_aie2p_native_progress_distance(void* user_data,
                                                    uint32_t source,
                                                    uint32_t reader) {
  const loom_aie2p_native_context_t* context = user_data;
  const loom_xdna_tile_coordinate_t origin =
      context->pool_tiles[context->channels[source].pool_index]->coordinate;
  const loom_xdna_tile_coordinate_t target =
      context->workers[context->channels[reader].cursor.reader.worker]
          .tile->coordinate;
  return (uint32_t)abs((int)origin.column - (int)target.column) +
         (uint32_t)abs((int)origin.row - (int)target.row);
}

static uint64_t loom_aie2p_native_input_distance(void* user_data,
                                                 uint32_t source,
                                                 uint32_t reader) {
  const loom_aie2p_native_context_t* context = user_data;
  const loom_xdna_tile_coordinate_t origin =
      context->workers[source].tile->coordinate;
  const loom_xdna_tile_coordinate_t target =
      context->workers[reader].tile->coordinate;
  return (uint32_t)abs((int)origin.column - (int)target.column) + target.row;
}

iree_status_t loom_aie2p_native_select_progress(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, bool* out_valid) {
  *out_valid = false;
  const iree_host_size_t count = realization->invocations.channel_count;
  bool needs_progress = false;
  iree_host_size_t input_count = 0, phase_count = 0;
  for (iree_host_size_t i = 0; i < realization->invocations.worker_count; ++i) {
    needs_progress |= realization->workers[i].transport.prefix_cycles != 0;
    loom_aie2p_native_input_t* input = context->workers[i].input;
    if (!input) {
      continue;
    }
    const loom_pipeline_transport_input_phase_t* tail =
        &input->source->phases[input->source->phase_count - 1];
    if (input->source->admissions != 1 || tail->period != tail->transfers) {
      ++input_count;
      phase_count += input->source->phase_count;
      continue;
    }
    // Complete stable input has no external readiness or credit dependency.
    // An existing participant starts it at entry and retires its queues at
    // exit. The retained execution row supplies the complete participant set.
    const loom_pipeline_participants_t* participants =
        context->execution.invocations[input->invocation->ordinal];
    uint64_t best_cost = UINT64_MAX;
    loom_pipeline_participant_iterator_t iterator;
    loom_pipeline_participants_begin(participants, &iterator);
    loom_pipeline_participant_t participant;
    while (loom_pipeline_participants_next(&iterator, &participant)) {
      const iree_host_size_t candidate = participant.worker;
      if (realization->invocations.workers[candidate].invocation !=
          input->invocation) {
        continue;
      }
      const uint64_t cost = loom_aie2p_native_input_distance(
          context, (uint32_t)i, (uint32_t)candidate);
      if (cost < best_cost) {
        best_cost = cost;
        input->controller = (uint32_t)candidate;
      }
    }
    if (input->controller == UINT32_MAX) {
      return loom_aie2p_native_reject(
          context, realization->workers[i].source->declaration,
          IREE_SV("an existing instruction worker in the finite input's "
                  "invocation"));
    }
    loom_aie2p_native_worker_t* owner = &context->workers[input->controller];
    input->next = owner->inputs;
    owner->inputs = input;
  }

  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_aie2p_native_channel_t* channel = &context->channels[i];
    const loom_xdna_dma_facts_t* dma =
        &context->pool_tiles[channel->pool_index]->facts->dma;
    needs_progress |=
        channel->cursor.reader.maximum_admissions >
        (uint32_t)dma->maximum_task_repeat_count * dma->task_queue_depth;
  }
  if (!needs_progress && !input_count) {
    *out_valid = true;
    return iree_ok_status();
  }
  loom_pipeline_progress_endpoint_t* endpoints;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, count, sizeof(*endpoints), (void**)&endpoints));
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_aie2p_native_channel_t* channel = &context->channels[i];
    const uint32_t reader = channel->cursor.reader.worker;
    endpoints[i] = (loom_pipeline_progress_endpoint_t){
        .reads =
            (channel->replay.record_count || channel->cursor.reader.shared ||
             channel->cursor.writer.shared)
                ? 0
                : channel->cursor.reader.maximum_admissions,
        .writes = channel->cursor.writer.shared
                      ? 0
                      : channel->cursor.writer.maximum_admissions,
        .observer = !channel->replay.record_count &&
                    !channel->cursor.reader.shared &&
                    !channel->cursor.writer.shared && reader != UINT32_MAX &&
                    context->workers[reader].execution ==
                        LOOM_AIE2P_NATIVE_EXECUTION_CORE &&
                    channel->cursor.reader.exact_admissions};
  }
  loom_pipeline_progress_site_t* sites;
  uint32_t *inputs, *phases;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(context->pass->arena, input_count ? count : 0,
                                sizeof(*sites), (void**)&sites));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, input_count, sizeof(*inputs), (void**)&inputs));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, phase_count, sizeof(*phases), (void**)&phases));
  if (input_count) {
    memset(sites, 0xff, count * sizeof(*sites));
    iree_host_size_t input_index = 0, phase_index = 0;
    for (iree_host_size_t i = 0; i < realization->invocations.worker_count;
         ++i) {
      const loom_pipeline_worker_t* worker = &realization->workers[i];
      if (context->workers[i].input &&
          context->workers[i].input->controller == UINT32_MAX) {
        inputs[input_index++] = i;
        for (iree_host_size_t j = 0; j < worker->transport.input->phase_count;
             ++j) {
          phases[phase_index++] = loom_pipeline_invocations_channel_index(
              &realization->invocations,
              worker->transport.input->phases[j].reservation->channel);
        }
      }
      for (iree_host_size_t j = 0; j < worker->channels.action_count; ++j) {
        const loom_channel_plan_action_t* action = &worker->channels.actions[j];
        if (!loom_channel_acquire_isa(action->op)) {
          continue;
        }
        const iree_host_size_t channel =
            loom_pipeline_invocations_channel_index(&realization->invocations,
                                                    action->channel);
        if (!endpoints[channel].observer) {
          continue;
        }
        if (sites[channel].worker != UINT32_MAX) {
          endpoints[channel].observer = false;
        } else {
          sites[channel] =
              (loom_pipeline_progress_site_t){(uint32_t)i, (uint32_t)j};
        }
      }
    }
  }
  const loom_pipeline_progress_callback_t callback = {
      .cost = loom_aie2p_native_progress_distance, .user_data = context};
  uint32_t* readers;
  loom_pipeline_progress_observations_t* observations;
  IREE_RETURN_IF_ERROR(loom_pipeline_progress_select(
      &realization->invocations, realization->workers, endpoints, callback,
      phases, phase_count, context->pass->arena, &readers, &observations));
  for (iree_host_size_t i = 0; i < count; ++i) {
    context->channels[i].progress =
        readers[i] != UINT32_MAX ? &context->channels[readers[i]] : NULL;
  }
  const loom_pipeline_progress_callback_t continuation_cost = {
      .cost = loom_aie2p_native_input_distance, .user_data = context};
  loom_pipeline_continuation_t* continuations;
  IREE_RETURN_IF_ERROR(loom_pipeline_continuations_select(
      &realization->invocations, realization->workers, observations, sites,
      inputs, input_count, continuation_cost, context->pass->arena,
      &continuations));
  for (iree_host_size_t i = 0; i < input_count; ++i) {
    loom_aie2p_native_input_t* input = context->workers[inputs[i]].input;
    if (continuations[i].worker == UINT32_MAX) {
      return loom_aie2p_native_reject(
          context, realization->workers[inputs[i]].source->declaration,
          IREE_SV("an existing reader observing every stable input phase in "
                  "one lifetime"));
    }
    input->controller = continuations[i].worker;
    input->continuation = &continuations[i];
    if (input->invocation->ordinal &&
        realization->invocations.workers[input->controller].invocation !=
            input->invocation) {
      return loom_aie2p_native_reject(
          context, realization->workers[inputs[i]].source->declaration,
          IREE_SV("a finite input continuation within its owning invocation"));
    }
    loom_aie2p_native_worker_t* owner =
        &context->workers[continuations[i].worker];
    input->next = owner->inputs;
    owner->inputs = input;
  }
  *out_valid = true;
  return iree_ok_status();
}

uint32_t loom_aie2p_native_phased_window(
    const loom_aie2p_native_context_t* context,
    const loom_aie2p_native_channel_t* channel, iree_host_size_t prefix_count,
    uint32_t cycles, uint32_t records) {
  const loom_xdna_dma_facts_t* dma =
      &context->pool_tiles[channel->pool_index]->facts->dma;
  const uint32_t capacity = channel->source->capacity;
  if (!channel->progress || prefix_count + 2 > dma->task_queue_depth ||
      capacity > dma->maximum_task_repeat_count ||
      channel->progress->cursor.reader.maximum_admissions !=
          (uint64_t)cycles * records) {
    return 0;
  }
  uint32_t window =
      iree_min(records, dma->maximum_task_repeat_count / capacity * capacity);
  // Repeated cycles end on a complete bulk window and ring turn. A single
  // enclosing lifetime can retain the ordinary final partial-window task.
  if (cycles > 1) {
    if (records % capacity) {
      return 0;
    }
    // The largest hardware window need not divide the cycle. Search complete
    // ring turns within that fixed task-repeat bound, keeping the largest
    // divisor to minimize continuation traffic.
    while (records % window) {
      window -= capacity;
    }
  }
  return window;
}

iree_status_t loom_aie2p_native_select_refill(
    loom_aie2p_native_context_t* context, loom_aie2p_native_tile_t* tile,
    uint8_t engine, loom_xdna_dma_direction_t direction,
    const loom_aie2p_native_channel_t* channel, uint32_t repetitions,
    uint32_t capacity, uint32_t byte_offset, uint32_t byte_stride,
    uint32_t credit_interval, uint16_t first_descriptor,
    uint16_t available_descriptors, const uint32_t words[8],
    loom_aie2p_native_dma_sequence_t* sequence, iree_string_view_t* reason,
    bool* out_valid) {
  const loom_aie2p_native_memory_layout_t* layout = tile->memory_layout;
  const loom_xdna_dma_facts_t* dma = &tile->facts->dma;
  if (!layout || !channel || !channel->progress || credit_interval != 1 ||
      channel->progress->cursor.reader.maximum_admissions != repetitions ||
      dma->task_queue_depth < 2 || capacity > dma->maximum_task_repeat_count) {
    return iree_ok_status();
  }
  const uint32_t window = dma->maximum_task_repeat_count / capacity * capacity;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_build_sequence(
      dma, layout, first_descriptor, available_descriptors, direction, window,
      capacity, byte_offset, byte_stride, 1, words, context->pass->arena,
      sequence, reason, out_valid));
  if (!*out_valid) {
    return iree_ok_status();
  }
  sequence->rewind_iteration = repetitions % capacity != 0;
  loom_aie2p_native_refill_queue_t* queue;
  return loom_aie2p_native_add_refill(context, tile, engine, direction, channel,
                                      repetitions, window, sequence, &queue);
}

iree_status_t loom_aie2p_native_add_refill(
    loom_aie2p_native_context_t* context, loom_aie2p_native_tile_t* tile,
    uint8_t engine, loom_xdna_dma_direction_t direction,
    const loom_aie2p_native_channel_t* channel, uint32_t repetitions,
    uint32_t window, const loom_aie2p_native_dma_sequence_t* sequence,
    loom_aie2p_native_refill_queue_t** out_queue) {
  const loom_aie2p_native_memory_layout_t* layout = tile->memory_layout;
  loom_aie2p_native_worker_t* worker =
      &context->workers[channel->progress->cursor.reader.worker];
  loom_aie2p_native_refill_t* group = worker->refills;
  while (group &&
         (group->progress != channel->progress || group->window != window)) {
    group = group->next;
  }
  if (!group) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(context->pass->arena,
                                             sizeof(*group), (void**)&group));
    *group = (loom_aie2p_native_refill_t){
        .next = worker->refills,
        .progress = channel->progress,
        .window = window,
        .remaining_tasks = (repetitions - 1) / window};
    worker->refills = group;
  }
  const uint16_t queue_index = engine;
  const uint64_t base =
      (uint64_t)tile->coordinate.column << context->family->column_shift |
      (uint64_t)tile->coordinate.row << context->family->row_shift;
  const uint32_t address =
      (uint32_t)(loom_xdna_register_field_address(
                     context->family, layout->directions[direction - 1].start,
                     tile->coordinate, &queue_index) -
                 base);
  const uint32_t tail = repetitions % window;
  const uint32_t final_task =
      loom_xdna_register_field_encode(layout->directions[direction - 1].start,
                                      sequence->descriptor) |
      loom_xdna_register_field_encode(layout->directions[direction - 1].repeat,
                                      (tail ? tail : window) - 1);
  loom_aie2p_native_refill_queue_t* queue;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(context->pass->arena, sizeof(*queue),
                                           (void**)&queue));
  *queue = (loom_aie2p_native_refill_queue_t){.next = group->queues,
                                              .tile = tile,
                                              .engine = engine,
                                              .direction = direction,
                                              .address = address,
                                              .task = sequence->tasks[0],
                                              .final_task = final_task};
  group->queues = queue;
  *out_queue = queue;
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_route_refills(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, uint32_t* next_source,
    bool* out_valid) {
  for (iree_host_size_t i = 0; i < realization->invocations.worker_count; ++i) {
    loom_aie2p_native_worker_t* worker = &context->workers[i];
    for (loom_aie2p_native_refill_t* group = worker->refills; group;
         group = group->next) {
      for (loom_aie2p_native_refill_queue_t* queue = group->queues; queue;
           queue = queue->next) {
        IREE_RETURN_IF_ERROR(loom_aie2p_native_select_control_route(
            context, realization, i, queue->tile, false, next_source, NULL,
            out_valid));
        if (!*out_valid) {
          return iree_ok_status();
        }
        const loom_aie2p_native_worker_t* owner =
            &context->workers[context->execution
                                  .workers[context->core_engines[i]]];
        queue->packet = owner->control_packets[queue->tile - context->tiles];
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_prepare_refills(
    loom_aie2p_native_context_t* context, loom_aie2p_native_refill_t* refills) {
  loom_aie2p_worker_builder_t* code = &context->code;
  for (loom_aie2p_native_refill_t* group = refills; group;
       group = group->next) {
    for (loom_aie2p_native_refill_queue_t* queue = group->queues; queue;
         queue = queue->next) {
      loom_builder_t builder;
      loom_op_t* function;
      IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(code, &code->scalar_type, 1,
                                                    NULL, 0, &builder,
                                                    &queue->submit, &function));
      const loom_value_id_t argument =
          loom_region_entry_block(loom_low_func_def_body(function))->arg_ids[0];
      IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
          code, &builder, queue->packet, queue->address, &argument, 1));
      IREE_RETURN_IF_ERROR(loom_aie2p_worker_return(&builder, NULL, 0));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_refill_constant(
    loom_builder_t* builder, uint32_t value, loom_value_id_t* result) {
  loom_op_t* constant;
  IREE_RETURN_IF_ERROR(
      loom_scalar_constant_build(builder, loom_attr_i64((int32_t)value),
                                 loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                                 LOOM_LOCATION_UNKNOWN, &constant));
  *result = loom_scalar_constant_result(constant);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_refill_prefixes(
    loom_builder_t* builder, const loom_aie2p_native_refill_queue_t* queue,
    loom_value_id_t remaining, loom_value_id_t zero) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t period;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_refill_constant(
      builder, queue->cycle_windows, &period));
  loom_op_t *position, *boundary, *prefix;
  IREE_RETURN_IF_ERROR(loom_scalar_remui_build(
      builder, remaining, period, type, LOOM_LOCATION_UNKNOWN, &position));
  IREE_RETURN_IF_ERROR(
      loom_scalar_cmpi_build(builder, LOOM_SCALAR_CMPI_PREDICATE_EQ,
                             loom_scalar_remui_result(position), zero,
                             LOOM_LOCATION_UNKNOWN, &boundary));
  IREE_RETURN_IF_ERROR(
      loom_scf_if_build(builder, 0, loom_scalar_cmpi_result(boundary), NULL, 0,
                        NULL, 0, LOOM_LOCATION_UNKNOWN, &prefix));
  const loom_builder_ip_t outer = loom_builder_enter_region(
      builder, prefix, loom_scf_if_then_region(prefix));
  for (uint32_t i = 0; i < queue->prefix_count; ++i) {
    loom_value_id_t task;
    IREE_RETURN_IF_ERROR(
        loom_aie2p_native_refill_constant(builder, queue->prefixes[i], &task));
    loom_op_t* call;
    IREE_RETURN_IF_ERROR(
        loom_low_invoke_build(builder, 0, 0, 0, 0, queue->submit, &task, 1,
                              NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &call));
  }
  loom_op_t* yield;
  IREE_RETURN_IF_ERROR(
      loom_scf_yield_build(builder, NULL, 0, LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(builder, outer);
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_emit_refill(
    loom_builder_t* builder, const loom_aie2p_native_refill_t* refill,
    loom_value_id_t* state) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_type_t types[] = {type, type};
  const loom_value_id_t countdown = state[refill->state_index];
  const loom_value_id_t remaining = state[refill->state_index + 1];
  loom_value_id_t zero, one, reset;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_refill_constant(builder, 0, &zero));
  IREE_RETURN_IF_ERROR(loom_aie2p_native_refill_constant(builder, 1, &one));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_native_refill_constant(builder, refill->window - 1, &reset));
  loom_op_t *boundary, *window;
  IREE_RETURN_IF_ERROR(
      loom_scalar_cmpi_build(builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, countdown,
                             zero, LOOM_LOCATION_UNKNOWN, &boundary));
  IREE_RETURN_IF_ERROR(
      loom_scf_if_build(builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                        loom_scalar_cmpi_result(boundary), types, 2, NULL, 0,
                        LOOM_LOCATION_UNKNOWN, &window));
  const loom_builder_ip_t outer = loom_builder_enter_region(
      builder, window, loom_scf_if_then_region(window));
  loom_op_t *pending, *submit;
  IREE_RETURN_IF_ERROR(
      loom_scalar_cmpi_build(builder, LOOM_SCALAR_CMPI_PREDICATE_NE, remaining,
                             zero, LOOM_LOCATION_UNKNOWN, &pending));
  IREE_RETURN_IF_ERROR(
      loom_scf_if_build(builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                        loom_scalar_cmpi_result(pending), &type, 1, NULL, 0,
                        LOOM_LOCATION_UNKNOWN, &submit));
  const loom_builder_ip_t inner = loom_builder_enter_region(
      builder, submit, loom_scf_if_then_region(submit));
  loom_op_t* last;
  IREE_RETURN_IF_ERROR(
      loom_scalar_cmpi_build(builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, remaining,
                             one, LOOM_LOCATION_UNKNOWN, &last));
  for (const loom_aie2p_native_refill_queue_t* queue = refill->queues; queue;
       queue = queue->next) {
    if (queue->prefix_count) {
      IREE_RETURN_IF_ERROR(
          loom_aie2p_native_refill_prefixes(builder, queue, remaining, zero));
    }
    loom_value_id_t full, tail;
    IREE_RETURN_IF_ERROR(
        loom_aie2p_native_refill_constant(builder, queue->task, &full));
    IREE_RETURN_IF_ERROR(
        loom_aie2p_native_refill_constant(builder, queue->final_task, &tail));
    loom_op_t* selected;
    IREE_RETURN_IF_ERROR(
        loom_scf_select_build(builder, loom_scalar_cmpi_result(last), tail,
                              full, type, LOOM_LOCATION_UNKNOWN, &selected));
    const loom_value_id_t argument = loom_scf_select_result(selected);
    loom_op_t* call;
    IREE_RETURN_IF_ERROR(
        loom_low_invoke_build(builder, 0, 0, 0, 0, queue->submit, &argument, 1,
                              NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &call));
  }
  loom_op_t *next, *yield;
  IREE_RETURN_IF_ERROR(loom_scalar_subi_build(builder, 0, remaining, one, type,
                                              LOOM_LOCATION_UNKNOWN, &next));
  const loom_value_id_t next_remaining = loom_scalar_subi_result(next);
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(builder, &next_remaining, 1,
                                            LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(builder, inner);
  loom_builder_enter_region(builder, submit, loom_scf_if_else_region(submit));
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(builder, &remaining, 1,
                                            LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(builder, inner);
  const loom_value_id_t updated[] = {reset, loom_op_results(submit)[0]};
  IREE_RETURN_IF_ERROR(
      loom_scf_yield_build(builder, updated, 2, LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(builder, outer);
  loom_builder_enter_region(builder, window, loom_scf_if_else_region(window));
  IREE_RETURN_IF_ERROR(loom_scalar_subi_build(builder, 0, countdown, one, type,
                                              LOOM_LOCATION_UNKNOWN, &next));
  const loom_value_id_t continued[] = {loom_scalar_subi_result(next),
                                       remaining};
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(builder, continued, 2,
                                            LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(builder, outer);
  state[refill->state_index] = loom_op_results(window)[0];
  state[refill->state_index + 1] = loom_op_results(window)[1];
  return iree_ok_status();
}
