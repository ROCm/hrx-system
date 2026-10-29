// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline_workers.h"

#include <stdlib.h>

#include "loom/analysis/storage_access.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/util/fact_cfg.h"
#include "loom/util/fact_extensions.h"

static iree_status_t loom_pipeline_workers_reject(
    const loom_op_t* op, iree_string_view_t requirement,
    iree_diagnostic_emitter_t diagnostic_emitter) {
  const loom_diagnostic_param_t params[] = {loom_param_string(requirement)};
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_LOWERING_066,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(diagnostic_emitter, &emission);
}

static bool loom_pipeline_worker_coordinate(
    int64_t literal, loom_value_slice_t values, uint16_t* dynamic_index,
    const loom_value_fact_table_t* facts, uint64_t* out_value) {
  int64_t value = literal;
  if (literal == INT64_MIN && !loom_value_facts_as_exact_i64(
                                  loom_value_fact_table_lookup(
                                      facts, values.values[(*dynamic_index)++]),
                                  &value)) {
    return false;
  }
  if (value < 0) {
    return false;
  }
  *out_value = (uint64_t)value;
  return true;
}

static const loom_pipeline_channel_binding_t*
loom_pipeline_worker_lookup_incoming(
    const loom_pipeline_channel_binding_t* incoming,
    iree_host_size_t incoming_count, loom_value_id_t value_id) {
  iree_host_size_t begin = 0, end = incoming_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    const loom_pipeline_channel_binding_t* binding = &incoming[middle];
    if (binding->value_id == value_id) {
      return binding;
    }
    if (binding->value_id < value_id) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return NULL;
}

typedef struct loom_pipeline_transport_block_t {
  // Canonical asynchronous stream in this block, or NULL.
  const loom_kernel_async_stream_t* stream;
  // First channel action in this block's contiguous source-order span.
  iree_host_size_t channel_begin;
  // Number of channel actions in the span.
  iree_host_size_t channel_count;
} loom_pipeline_transport_block_t;

// Retains complete channel-copy ownership without assigning a physical
// transport. This recognizes the semantic family from one source obligation to
// one or more independently publishing destinations. Anything outside that
// family remains ordinary instruction-executing channel use.
static iree_status_t loom_pipeline_transport_cohorts(
    loom_module_t* module, iree_arena_allocator_t* arena,
    const loom_value_fact_cfg_region_t* structure,
    const loom_pipeline_worker_t* worker,
    const loom_pipeline_transport_step_t* steps, iree_host_size_t step_count,
    const loom_pipeline_transport_cohort_t** out_cohorts,
    iree_host_size_t* out_cohort_count,
    const loom_pipeline_transport_copy_t** out_copies,
    iree_host_size_t* out_copy_count) {
  *out_cohorts = NULL;
  *out_cohort_count = 0;
  *out_copies = NULL;
  *out_copy_count = 0;
  const loom_channel_plan_t* plan = &worker->channels;
  if (!plan->action_count) {
    return iree_ok_status();
  }
  loom_pipeline_transport_cohort_t* cohorts;
  loom_pipeline_transport_destination_t* destinations;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, plan->action_count, sizeof(*cohorts), (void**)&cohorts));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, plan->action_count, sizeof(*destinations), (void**)&destinations));
  uint32_t* cohort_indices;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, plan->value_count,
                                                 sizeof(*cohort_indices),
                                                 (void**)&cohort_indices));
  iree_host_size_t cohort_count = 0;
  uint32_t admission_sequence = 0;
  bool consumed_since_admission = false;
  for (iree_host_size_t i = 0; i < step_count; ++i) {
    const loom_channel_plan_action_t* action = steps[i].source.channel;
    const loom_op_t* op = action->op;
    if (loom_channel_acquire_isa(op) || loom_channel_accept_isa(op)) {
      admission_sequence += consumed_since_admission;
      consumed_since_admission = false;
      if (loom_channel_acquire_isa(op) &&
          !loom_value_has_no_uses(
              loom_module_value(module, loom_channel_acquire_view(op)))) {
        return iree_ok_status();
      }
      const loom_value_id_t read = loom_op_const_results(op)[0];
      cohort_indices[loom_local_value_domain_ordinal(
          plan->value_domain, read)] = (uint32_t)cohort_count;
      const uint64_t admissions =
          worker->block_execution_counts
              ? worker->block_execution_counts[op->parent_block->region_index]
              : 1;
      if (!admissions) {
        return iree_ok_status();
      }
      cohorts[cohort_count++] = (loom_pipeline_transport_cohort_t){
          .admission = action,
          .admissions = admissions,
          .admission_scope = loom_cfg_loop_nest_innermost(
              &structure->loops, op->parent_block->region_index),
          .copy_scope = LOOM_CFG_LOOP_NEST_NONE,
          .admission_sequence = admission_sequence};
    } else if (loom_channel_reserve_isa(op)) {
      if (!loom_value_has_no_uses(
              loom_module_value(module, loom_channel_reserve_view(op)))) {
        return iree_ok_status();
      }
    } else if (loom_channel_copy_isa(op) || loom_channel_release_isa(op)) {
      consumed_since_admission = true;
    } else if (!loom_channel_fanout_isa(op)) {
      return iree_ok_status();
    }
  }
  // Membership owns forwarding through CFG arguments and nested fanouts.
  // A common source site alone does not establish one dynamic record: copies
  // must stay in its admission scope or in nested replay loops, with final
  // release back in that same admission scope.
  for (iree_host_size_t i = 0; i < step_count; ++i) {
    const loom_op_t* op = steps[i].op;
    const bool release = loom_channel_release_isa(op);
    if (!release && !loom_channel_copy_isa(op)) {
      continue;
    }
    const loom_value_id_t source_admission = loom_channel_plan_admission(
        plan,
        release ? loom_channel_release_read(op) : loom_channel_copy_source(op));
    if (source_admission == LOOM_VALUE_ID_INVALID) {
      return iree_ok_status();
    }
    const uint32_t index = cohort_indices[loom_local_value_domain_ordinal(
        plan->value_domain, source_admission)];
    loom_pipeline_transport_cohort_t* cohort = &cohorts[index];
    const uint16_t block = op->parent_block->region_index;
    const uint16_t admission_loop = cohort->admission_scope;
    const uint16_t operation_loop =
        loom_cfg_loop_nest_innermost(&structure->loops, block);
    const uint64_t executions = worker->block_execution_counts
                                    ? worker->block_execution_counts[block]
                                    : 1;
    if (release) {
      if (cohort->release || admission_loop != operation_loop ||
          executions != cohort->admissions) {
        return iree_ok_status();
      }
      cohort->release = op;
      continue;
    }
    if (executions < cohort->admissions || executions % cohort->admissions ||
        (admission_loop != LOOM_CFG_LOOP_NEST_NONE &&
         !loom_cfg_loop_nest_contains(&structure->loops, admission_loop,
                                      block))) {
      return iree_ok_status();
    }
    const loom_value_id_t destination_admission =
        loom_channel_plan_admission(plan, loom_channel_copy_destination(op));
    if (destination_admission == LOOM_VALUE_ID_INVALID) {
      return iree_ok_status();
    }
    const loom_op_t* reservation =
        loom_value_def_op(loom_module_value(module, destination_admission));
    if (loom_cfg_loop_nest_innermost(&structure->loops,
                                     reservation->parent_block->region_index) !=
        operation_loop) {
      return iree_ok_status();
    }
    if (!cohort->destination_count) {
      cohort->transfers = executions;
      cohort->copy_scope = operation_loop;
    } else if (cohort->transfers != executions ||
               cohort->copy_scope != operation_loop) {
      cohort->transfers = 0;
    }
    ++cohort->destination_count;
  }
  iree_host_size_t destination_count = 0;
  for (iree_host_size_t i = 0; i < cohort_count; ++i) {
    if (!cohorts[i].destination_count ||
        (cohorts[i].transfers != cohorts[i].admissions &&
         !cohorts[i].release)) {
      return iree_ok_status();
    }
    cohorts[i].destinations = &destinations[destination_count];
    destination_count += cohorts[i].destination_count;
    cohorts[i].destination_count = 0;
  }
  loom_pipeline_transport_copy_t* copies;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, destination_count, sizeof(*copies), (void**)&copies));
  iree_host_size_t copy_count = 0;
  for (iree_host_size_t i = 0; i < step_count; ++i) {
    const loom_channel_plan_action_t* action = steps[i].source.channel;
    const loom_op_t* copy = action->op;
    if (!loom_channel_copy_isa(copy)) {
      continue;
    }
    const loom_value_id_t source_admission =
        loom_channel_plan_admission(plan, loom_channel_copy_source(copy));
    const loom_value_id_t destination_admission =
        loom_channel_plan_admission(plan, loom_channel_copy_destination(copy));
    const uint32_t index = cohort_indices[loom_local_value_domain_ordinal(
        plan->value_domain, source_admission)];
    loom_pipeline_transport_cohort_t* cohort = &cohorts[index];
    const iree_host_size_t destination_index =
        (cohort->destinations - destinations) + cohort->destination_count++;
    destinations[destination_index] = (loom_pipeline_transport_destination_t){
        .reservation =
            loom_value_def_op(loom_module_value(module, destination_admission)),
        .copy = copy,
        .channel = action->destination_channel};
    const uint16_t block = copy->parent_block->region_index;
    copies[copy_count++] = (loom_pipeline_transport_copy_t){
        .source = cohort,
        .destination = &destinations[destination_index],
        .executions = worker->block_execution_counts
                          ? worker->block_execution_counts[block]
                          : 1,
        .scope = loom_cfg_loop_nest_innermost(&structure->loops, block)};
  }
  *out_cohorts = cohorts;
  *out_cohort_count = cohort_count;
  *out_copies = copies;
  *out_copy_count = copy_count;
  return iree_ok_status();
}

static int loom_pipeline_transport_compare_channels(const void* lhs,
                                                    const void* rhs) {
  const uintptr_t a = (uintptr_t)*(const loom_channel_identity_t* const*)lhs;
  const uintptr_t b = (uintptr_t)*(const loom_channel_identity_t* const*)rhs;
  return (a > b) - (a < b);
}

// Group adjacent admissions by source lifetime. The ordered copy table retains
// visits independently; uniform replay is admitted only when the whole batch
// has one common destination order and copy scope.
static iree_status_t loom_pipeline_transport_batches(
    iree_arena_allocator_t* arena,
    const loom_pipeline_transport_cohort_t* cohorts,
    iree_host_size_t cohort_count, const loom_pipeline_transport_copy_t* copies,
    iree_host_size_t copy_count,
    const loom_pipeline_transport_batch_t** out_batches,
    iree_host_size_t* out_count) {
  loom_pipeline_transport_batch_t* batches;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, cohort_count, sizeof(*batches), (void**)&batches));
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < cohort_count; ++i) {
    const loom_pipeline_transport_cohort_t* record = &cohorts[i];
    bool append = count != 0;
    if (append) {
      const loom_pipeline_transport_cohort_t* first =
          batches[count - 1].records;
      append = first->admission->channel == record->admission->channel &&
               first->admissions == record->admissions &&
               first->admission_scope == record->admission_scope &&
               first->admission_sequence == record->admission_sequence &&
               (first->release != NULL) == (record->release != NULL);
      // Without a retained obligation, each copy itself releases its source.
      // Such records coalesce only with the same one-pass destinations.
      if (append && !first->release) {
        append = first->transfers == record->transfers &&
                 first->copy_scope == record->copy_scope &&
                 first->destination_count == record->destination_count;
        for (iree_host_size_t j = 0; append && j < first->destination_count;
             ++j) {
          append =
              first->destinations[j].channel == record->destinations[j].channel;
        }
      }
    }
    if (append) {
      ++batches[count - 1].record_count;
    } else {
      batches[count++] = (loom_pipeline_transport_batch_t){
          .records = record,
          .record_count = 1,
          .replays = record->transfers / record->admissions};
    }
  }
  iree_host_size_t copy_index = 0;
  iree_host_size_t maximum_destinations = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    maximum_destinations =
        iree_max(maximum_destinations, batches[i].records->destination_count);
  }
  const loom_channel_identity_t** destinations;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, maximum_destinations,
                                                 sizeof(*destinations),
                                                 (void**)&destinations));
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_pipeline_transport_batch_t* batch = &batches[i];
    const loom_pipeline_transport_cohort_t* first = batch->records;
    batch->copies = &copies[copy_index];
    const loom_pipeline_transport_cohort_t* previous = first;
    while (copy_index < copy_count && copies[copy_index].source >= first &&
           copies[copy_index].source < first + batch->record_count) {
      if (copies[copy_index].source < previous) {
        batch->replays = 0;
      }
      previous = copies[copy_index++].source;
      ++batch->copy_count;
    }
    for (iree_host_size_t j = 0; batch->replays && j < batch->record_count;
         ++j) {
      const loom_pipeline_transport_cohort_t* record = &batch->records[j];
      if (record->transfers != first->transfers ||
          record->copy_scope != first->copy_scope ||
          record->destination_count != first->destination_count) {
        batch->replays = 0;
        break;
      }
      for (iree_host_size_t k = 0; k < first->destination_count; ++k) {
        if (first->destinations[k].channel != record->destinations[k].channel) {
          batch->replays = 0;
          break;
        }
      }
    }
    if (batch->replays && first->destination_count > 1) {
      for (iree_host_size_t j = 0; j < first->destination_count; ++j) {
        destinations[j] = first->destinations[j].channel;
      }
      qsort(destinations, first->destination_count, sizeof(*destinations),
            loom_pipeline_transport_compare_channels);
      for (iree_host_size_t j = 1; j < first->destination_count; ++j) {
        if (destinations[j] == destinations[j - 1]) {
          batch->replays = 0;
          break;
        }
      }
    }
  }
  // Interleaved source lifetimes require a transport that can retain more
  // than one batch at a time. They remain ordinary instruction execution.
  if (copy_index != copy_count) {
    *out_batches = NULL;
    *out_count = 0;
    return iree_ok_status();
  }
  *out_batches = batches;
  *out_count = count;
  return iree_ok_status();
}

// Retains inclusive copy spans in the CFG owner's child-before-parent order.
// Each copy and each natural loop contributes once; no target walks the CFG.
static iree_status_t loom_pipeline_transport_copy_loops(
    const loom_cfg_loop_nest_t* nest, const uint64_t* trip_counts,
    const loom_pipeline_transport_copy_t* copies, iree_host_size_t copy_count,
    iree_arena_allocator_t* arena,
    const loom_pipeline_transport_loop_t** out_loops) {
  loom_pipeline_transport_loop_t* loops;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, nest->loop_count, sizeof(*loops), (void**)&loops));
  for (iree_host_size_t i = 0; i < nest->loop_count; ++i) {
    loops[i] = (loom_pipeline_transport_loop_t){
        .iterations = trip_counts[i],
        .parent = nest->loops[i].parent_loop_index};
  }
  for (iree_host_size_t i = 0; i < copy_count; ++i) {
    if (copies[i].scope == LOOM_CFG_LOOP_NEST_NONE) {
      continue;
    }
    loom_pipeline_transport_loop_t* loop = &loops[copies[i].scope];
    if (!loop->begin) {
      loop->begin = &copies[i];
    }
    loop->end = &copies[i + 1];
  }
  for (iree_host_size_t i = 0; i < nest->loop_count; ++i) {
    const loom_pipeline_transport_loop_t* loop = &loops[i];
    if (!loop->begin || loop->parent == LOOM_CFG_LOOP_NEST_NONE) {
      continue;
    }
    loom_pipeline_transport_loop_t* parent = &loops[loop->parent];
    if (!parent->begin || loop->begin < parent->begin) {
      parent->begin = loop->begin;
    }
    if (!parent->end || loop->end > parent->end) {
      parent->end = loop->end;
    }
  }
  *out_loops = loops;
  return iree_ok_status();
}

iree_status_t loom_pipeline_transport_copy_cursor_initialize(
    const loom_pipeline_transport_t* transport,
    const loom_pipeline_transport_batch_t* batch, iree_arena_allocator_t* arena,
    loom_pipeline_transport_copy_cursor_t* out_cursor) {
  *out_cursor = (loom_pipeline_transport_copy_cursor_t){
      .next = batch->copies,
      .end = batch->copies + batch->copy_count,
      .loops = transport->copy_loops,
      .scope = batch->records->admission_scope};
  if (transport->copy_loop_count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, transport->copy_loop_count, sizeof(*out_cursor->remaining),
        (void**)&out_cursor->remaining));
    for (iree_host_size_t i = 0; i < transport->copy_loop_count; ++i) {
      out_cursor->remaining[i] = transport->copy_loops[i].iterations;
    }
  }
  return iree_ok_status();
}

const loom_pipeline_transport_copy_t* loom_pipeline_transport_copy_cursor_next(
    loom_pipeline_transport_copy_cursor_t* cursor) {
  const loom_pipeline_transport_copy_t* copy = cursor->next;
  if (copy == cursor->end) {
    return NULL;
  }
  cursor->next = copy + 1;
  for (uint16_t scope = copy->scope; scope != cursor->scope;
       scope = cursor->loops[scope].parent) {
    const loom_pipeline_transport_loop_t* loop = &cursor->loops[scope];
    if (copy + 1 != loop->end) {
      break;
    }
    if (--cursor->remaining[scope]) {
      cursor->next = loop->begin;
      break;
    }
    cursor->remaining[scope] = loop->iterations;
  }
  return copy;
}

static iree_status_t loom_pipeline_worker_transport(
    loom_module_t* module, iree_arena_allocator_t* arena,
    const loom_value_fact_cfg_region_t* structure,
    loom_pipeline_worker_t* worker) {
  const loom_cfg_graph_t* graph = worker->graph;
  if ((!worker->asynchronous.streams && !worker->channels.action_count) ||
      (graph->has_cycles && !worker->block_execution_counts)) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < graph->reverse_postorder.count; ++i) {
    const uint16_t block = graph->reverse_postorder.values[i];
    const uint16_t loop =
        loom_cfg_loop_nest_innermost(&structure->loops, block);
    if (graph->blocks[block].successor_count > 1 &&
        (loop == LOOM_CFG_LOOP_NEST_NONE ||
         structure->loops.loops[loop].header_index != block)) {
      return iree_ok_status();
    }
  }
  loom_pipeline_transport_block_t* blocks;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, graph->block_count, sizeof(*blocks), (void**)&blocks));
  memset(blocks, 0, graph->block_count * sizeof(*blocks));
  iree_host_size_t capacity = worker->channels.action_count;
  for (iree_host_size_t i = 0; i < worker->channels.action_count; ++i) {
    loom_pipeline_transport_block_t* block =
        &blocks[worker->channels.actions[i].op->parent_block->region_index];
    if (block->channel_count++ == 0) {
      block->channel_begin = i;
    }
  }
  for (const loom_kernel_async_stream_t* stream = worker->asynchronous.streams;
       stream; stream = stream->next) {
    blocks[stream->block->region_index].stream = stream;
    capacity += stream->transfer_count + stream->wait_count;
  }
  loom_pipeline_transport_step_t* steps;
  const loom_kernel_async_stream_t** streams;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity, sizeof(*steps), (void**)&steps));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, graph->block_count, sizeof(*streams), (void**)&streams));
  const uint16_t* order = graph->reverse_postorder.values;
  if (graph->has_cycles) {
    uint16_t* loop_order;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, graph->reverse_postorder.count,
                                  sizeof(*loop_order), (void**)&loop_order));
    loom_cfg_loop_nest_order_blocks(&structure->loops, loop_order);
    order = loop_order;
  }
  iree_host_size_t count = 0, stream_count = 0;
  for (iree_host_size_t b = 0; b < graph->reverse_postorder.count; ++b) {
    const uint16_t block_index = order[b];
    const loom_pipeline_transport_block_t* block = &blocks[block_index];
    const loom_kernel_async_stream_t* stream = block->stream;
    if (stream) {
      streams[stream_count++] = stream;
    }
    iree_host_size_t channel = block->channel_begin, transfer = 0, wait = 0;
    const iree_host_size_t channel_end = channel + block->channel_count;
    for (const loom_op_t* op = graph->blocks[block_index].block->first_op; op;
         op = op->next_op) {
      if (channel < channel_end && worker->channels.actions[channel].op == op) {
        steps[count++] = (loom_pipeline_transport_step_t){
            .op = op,
            .kind = LOOM_PIPELINE_TRANSPORT_STEP_CHANNEL,
            .source.channel = &worker->channels.actions[channel++]};
      } else if (stream && transfer < stream->transfer_count &&
                 stream->transfers[transfer].request.op == op) {
        steps[count++] = (loom_pipeline_transport_step_t){
            .op = op,
            .kind = LOOM_PIPELINE_TRANSPORT_STEP_TRANSFER,
            .source.transfer = &stream->transfers[transfer++]};
      } else if (stream && wait < stream->wait_count &&
                 stream->waits[wait] == op) {
        steps[count++] = (loom_pipeline_transport_step_t){
            .op = op, .kind = LOOM_PIPELINE_TRANSPORT_STEP_WAIT};
        ++wait;
      } else if (!loom_func_return_isa(op) &&
                 !loom_kernel_async_group_isa(op) && !loom_cfg_br_isa(op) &&
                 !loom_cfg_cond_br_isa(op)) {
        const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
        if (!iree_any_bit_set(traits, LOOM_TRAIT_PURE) ||
            loom_traits_may_read(traits) || loom_traits_may_write(traits) ||
            iree_any_bit_set(
                traits, LOOM_TRAIT_OBSERVABLE_EFFECT | LOOM_TRAIT_CONVERGENT)) {
          return iree_ok_status();
        }
      }
    }
  }
  const loom_value_fact_induction_t* induction = NULL;
  uint64_t repetitions = 1;
  if (graph->has_cycles && stream_count) {
    if (structure->loops.loop_count != 1) {
      return iree_ok_status();
    }
    induction = &structure->inductions[0];
    for (iree_host_size_t i = 0; i < count; ++i) {
      const uint16_t block = steps[i].op->parent_block->region_index;
      if (!loom_cfg_loop_nest_contains(&structure->loops, 0, block) ||
          block == structure->loops.loops[0].header_index) {
        return iree_ok_status();
      }
    }
    repetitions =
        worker->block_execution_counts[steps[0].op->parent_block->region_index];
  }
  const loom_pipeline_transport_cohort_t* cohorts = NULL;
  iree_host_size_t cohort_count = 0;
  const loom_pipeline_transport_copy_t* copies = NULL;
  iree_host_size_t copy_count = 0;
  if (!stream_count) {
    IREE_RETURN_IF_ERROR(loom_pipeline_transport_cohorts(
        module, arena, structure, worker, steps, count, &cohorts, &cohort_count,
        &copies, &copy_count));
    if (!cohort_count) {
      return iree_ok_status();
    }
  }
  const loom_pipeline_transport_batch_t* batches = NULL;
  iree_host_size_t batch_count = 0;
  if (cohort_count) {
    IREE_RETURN_IF_ERROR(
        loom_pipeline_transport_batches(arena, cohorts, cohort_count, copies,
                                        copy_count, &batches, &batch_count));
    if (!batch_count) {
      return iree_ok_status();
    }
  }
  const loom_pipeline_transport_loop_t* copy_loops = NULL;
  iree_host_size_t copy_loop_count = 0;
  for (iree_host_size_t i = 0; i < batch_count; ++i) {
    if (!batches[i].replays && structure->loops.loop_count) {
      IREE_RETURN_IF_ERROR(loom_pipeline_transport_copy_loops(
          &structure->loops, worker->loop_trip_counts, copies, copy_count,
          arena, &copy_loops));
      copy_loop_count = structure->loops.loop_count;
      break;
    }
  }
  worker->transport =
      (loom_pipeline_transport_t){.steps = steps,
                                  .count = count,
                                  .streams = streams,
                                  .stream_count = stream_count,
                                  .cohorts = cohorts,
                                  .cohort_count = cohort_count,
                                  .batches = batches,
                                  .batch_count = batch_count,
                                  .copy_loops = copy_loops,
                                  .copy_loop_count = copy_loop_count,
                                  .repetitions = repetitions,
                                  .induction = induction};
  return iree_ok_status();
}

static iree_status_t loom_pipeline_worker_execution_counts(
    loom_module_t* module, iree_arena_allocator_t* arena,
    const loom_value_fact_cfg_region_t* structure,
    loom_pipeline_worker_t* worker) {
  uint64_t* trip_counts = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, structure->loops.loop_count,
                                sizeof(*trip_counts), (void**)&trip_counts));
  for (iree_host_size_t i = 0; i < structure->loops.loop_count; ++i) {
    const loom_loop_recurrence_facts_t recurrence =
        loom_value_fact_induction_facts(&worker->facts, module,
                                        &structure->inductions[i]);
    if (recurrence.trip_count_kind != LOOM_LOOP_TRIP_COUNT_EXACT) {
      return iree_ok_status();
    }
    trip_counts[i] = recurrence.trip_count;
  }
  uint64_t* counts;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, structure->graph.block_count, sizeof(*counts), (void**)&counts));
  if (loom_cfg_loop_nest_calculate_block_execution_counts(
          &structure->loops, trip_counts, counts)) {
    worker->block_execution_counts = counts;
    worker->loop_trip_counts = trip_counts;
  }
  return iree_ok_status();
}

static iree_status_t loom_pipeline_worker_build(
    loom_module_t* module, const loom_pipeline_resources_t* resources,
    const loom_value_fact_table_t* construction_facts,
    const loom_pipeline_channel_binding_t* incoming,
    iree_host_size_t incoming_count,
    const loom_channel_plan_callable_t* const* callables,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_storage_access_scope_t* storage_access, loom_pipeline_worker_t* worker,
    bool* out_valid) {
  *out_valid = false;
  const loom_op_t* declaration = worker->source->declaration;
  const loom_attribute_t origins =
      loom_pipeline_strand_static_origins(declaration);
  const loom_attribute_t counts =
      loom_pipeline_strand_static_counts(declaration);
  const loom_attribute_t strides =
      loom_pipeline_strand_static_strides(declaration);
  loom_pipeline_worker_axis_t* axes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, origins.count,
                                                 sizeof(*axes), (void**)&axes));
  uint16_t origin_index = 0, count_index = 0, stride_index = 0;
  for (uint32_t i = 0; i < origins.count; ++i) {
    if (!loom_pipeline_worker_coordinate(
            origins.i64_array[i], loom_pipeline_strand_origins(declaration),
            &origin_index, construction_facts, &axes[i].origin) ||
        !loom_pipeline_worker_coordinate(
            counts.i64_array[i], loom_pipeline_strand_counts(declaration),
            &count_index, construction_facts, &axes[i].count) ||
        !loom_pipeline_worker_coordinate(
            strides.i64_array[i], loom_pipeline_strand_strides(declaration),
            &stride_index, construction_facts, &axes[i].stride) ||
        axes[i].stride == 0) {
      return loom_pipeline_workers_reject(
          declaration, IREE_SV("specialized finite worker geometry"),
          diagnostic_emitter);
    }
  }
  worker->axes = axes;
  worker->rank = origins.count;
  const loom_symbol_ref_t callee = loom_func_call_callee(worker->source->call);
  worker->function = loom_func_like_cast(
      module, module->symbols.entries[callee.symbol_id].defining_op);
  loom_region_t* body = loom_func_like_body(worker->function);
  if (!body) {
    return loom_pipeline_workers_reject(worker->source->call,
                                        IREE_SV("a visible strand body"),
                                        diagnostic_emitter);
  }
  const loom_value_slice_t actuals =
      loom_func_call_operands(worker->source->call);
  uint16_t formal_count = 0;
  const loom_value_id_t* formals =
      loom_func_like_arg_ids(worker->function, &formal_count);
  loom_pipeline_channel_binding_t* bindings = NULL;
  loom_channel_plan_binding_t* channel_bindings = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, formal_count, sizeof(*bindings), (void**)&bindings));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, formal_count,
                                                 sizeof(*channel_bindings),
                                                 (void**)&channel_bindings));
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_initialize(&worker->facts, arena, 0));
  worker->facts.context.resolve_type_domain =
      construction_facts->context.resolve_type_domain;
  worker->facts.context.resolve_region_target =
      construction_facts->context.resolve_region_target;
  worker->facts.context.target_facts = loom_value_fact_table_block_target_facts(
      construction_facts,
      loom_region_const_entry_block(loom_pipeline_strand_body(declaration)));
  // Worker inference may refine argument domains, while the construction's
  // captured inputs remain immutable for the retained worker analysis.
  loom_value_fact_table_t* inputs = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*inputs), (void**)&inputs));
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(inputs, arena, 0));
  inputs->context.resolve_type_domain =
      construction_facts->context.resolve_type_domain;
  for (uint16_t i = 0; i < formal_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, formals[i]);
    loom_value_facts_t facts;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_for_type(
        inputs, construction_facts, module, type,
        loom_value_fact_table_lookup(construction_facts, actuals.values[i]),
        &facts));
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_define(inputs, formals[i], facts));
    if (!loom_channel_type_isa(type)) {
      continue;
    }
    loom_pipeline_channel_binding_t binding = {
        .value_id = formals[i],
        .resources = resources,
        .channel = loom_pipeline_resources_lookup_channel(resources,
                                                          actuals.values[i]),
    };
    if (!binding.channel) {
      const loom_pipeline_channel_binding_t* parent =
          loom_pipeline_worker_lookup_incoming(incoming, incoming_count,
                                               actuals.values[i]);
      if (parent) {
        binding.resources = parent->resources;
        binding.channel = parent->channel;
      }
    }
    if (!binding.channel) {
      return loom_pipeline_workers_reject(
          worker->source->call,
          IREE_SV("an incoming channel construction binding"),
          diagnostic_emitter);
    }
    bindings[worker->binding_count] = binding;
    channel_bindings[worker->binding_count++] = (loom_channel_plan_binding_t){
        .value_id = formals[i], .channel = &binding.channel->identity};
  }
  worker->bindings = bindings;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_values(
      &worker->facts,
      (loom_value_fact_table_view_t){inputs, formals, formal_count}, module));
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_compute(&worker->facts, module, worker->function));
  const loom_value_fact_cfg_region_t* structure = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_get_or_build_cfg_region(
      &worker->facts, module, body, &structure));
  worker->graph = &structure->graph;
  IREE_RETURN_IF_ERROR(loom_local_value_domain_acquire_for_region(
      module, body, arena, &worker->value_domain));
  loom_channel_plan_rejection_t rejection;
  iree_status_t status = loom_channel_plan_build(
      &worker->value_domain, channel_bindings, worker->binding_count, callables,
      arena, &worker->channels, &rejection);
  if (iree_status_is_ok(status) &&
      rejection.kind == LOOM_CHANNEL_PLAN_REJECTION_NONE) {
    if (worker->channels.action_count) {
      status = loom_pipeline_worker_execution_counts(module, arena, structure,
                                                     worker);
    }
  }
  if (iree_status_is_ok(status) &&
      rejection.kind == LOOM_CHANNEL_PLAN_REJECTION_NONE) {
    const loom_kernel_synchronization_legality_options_t options = {
        .storage_access = storage_access,
        .value_domain = &worker->value_domain,
        .fact_table = &worker->facts,
        .emitter = diagnostic_emitter,
        .phase_name = IREE_SV("pipeline-worker"),
    };
    status = loom_kernel_synchronization_legality_analyze_function(
        module, worker->function, &options, arena, &worker->asynchronous);
  }
  if (iree_status_is_ok(status) &&
      rejection.kind == LOOM_CHANNEL_PLAN_REJECTION_NONE &&
      worker->asynchronous.error_count == 0) {
    const loom_view_region_table_t* view_regions =
        worker->asynchronous.streams
            ? &worker->asynchronous.movement.view_regions
            : NULL;
    status = loom_channel_completion_analyze(
        &worker->channels, worker->graph, view_regions,
        worker->block_execution_counts, arena, &worker->completion);
  }
  if (iree_status_is_ok(status) &&
      rejection.kind == LOOM_CHANNEL_PLAN_REJECTION_NONE &&
      worker->asynchronous.error_count == 0) {
    status = loom_pipeline_worker_transport(module, arena, structure, worker);
  }
  loom_local_value_domain_release(&worker->value_domain);
  if (!iree_status_is_ok(status)) {
    return status;
  }
  if (rejection.kind != LOOM_CHANNEL_PLAN_REJECTION_NONE) {
    return loom_pipeline_workers_reject(
        rejection.op ? rejection.op : declaration,
        IREE_SV("statically bound channels and visible channel-aware calls"),
        diagnostic_emitter);
  }
  *out_valid = worker->asynchronous.error_count == 0;
  return iree_ok_status();
}

typedef struct loom_pipeline_storage_reference_t {
  // Canonical backing identity in the pipeline construction scope.
  loom_value_id_t root_value_id;
  // Comparable noalias partition, or NONE when roots may alias.
  loom_value_fact_alias_scope_id_t alias_scope_id;
  // Target-independent storage space of the backing allocation.
  loom_value_fact_memory_space_t memory_space;
} loom_pipeline_storage_reference_t;

static loom_pipeline_storage_reference_t
loom_pipeline_storage_reference_from_facts(const loom_value_fact_table_t* facts,
                                           loom_value_id_t value_id) {
  loom_pipeline_storage_reference_t reference = {
      .root_value_id = value_id,
      .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
  };
  const loom_value_facts_t value_facts =
      loom_value_fact_table_lookup(facts, value_id);
  loom_value_fact_buffer_reference_t buffer;
  if (loom_value_facts_query_buffer_reference(&facts->context, value_facts,
                                              &buffer)) {
    reference.root_value_id =
        loom_value_fact_buffer_reference_resolve_root_value(buffer, value_id);
    reference.alias_scope_id = buffer.alias_scope_id;
    reference.memory_space = buffer.memory_space;
    return reference;
  }
  loom_value_fact_view_reference_t view;
  if (loom_value_facts_query_view_reference(&facts->context, value_facts,
                                            &view)) {
    reference.root_value_id =
        loom_value_fact_view_reference_resolve_root_value(view, value_id);
    reference.alias_scope_id = view.alias_scope_id;
    reference.memory_space = view.memory_space;
  }
  return reference;
}

static bool loom_pipeline_storage_reference_may_modify(
    loom_storage_access_effects_t effects,
    const loom_pipeline_storage_reference_t* reference) {
  // Constant memory remains immutable across opaque escapes. Keep explicit
  // writes conservative here so their eventual legality does not affect this
  // proof.
  if (iree_any_bit_set(effects, LOOM_STORAGE_ACCESS_WRITE)) {
    return true;
  }
  return iree_any_bit_set(effects, LOOM_STORAGE_ACCESS_ESCAPE) &&
         reference->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT;
}

static bool loom_pipeline_storage_references_may_alias(
    const loom_view_region_t* source,
    const loom_pipeline_storage_reference_t* access) {
  if (source->root_value_id == access->root_value_id) {
    return true;
  }
  if (loom_view_memory_spaces_are_disjoint(source->memory_space,
                                           access->memory_space)) {
    return false;
  }
  return source->alias_scope_id == LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE ||
         access->alias_scope_id == LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE ||
         source->alias_scope_id == access->alias_scope_id;
}

static iree_status_t loom_pipeline_workers_prove_transport_sources(
    loom_module_t* module, const loom_pipeline_resources_t* resources,
    iree_host_size_t incoming_count, iree_arena_allocator_t* arena,
    loom_storage_access_scope_t* storage_access,
    loom_pipeline_worker_t* workers) {
  if (incoming_count || resources->composition_count) {
    return iree_ok_status();
  }
  bool repeated_transport = false;
  for (iree_host_size_t i = 0; i < resources->strand_count; ++i) {
    repeated_transport |= workers[i].transport.induction != NULL;
  }
  if (!repeated_transport) {
    return iree_ok_status();
  }
  const loom_storage_access_function_t** accesses = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, resources->strand_count, sizeof(*accesses), (void**)&accesses));
  iree_host_size_t modifying_capacity = 0;
  for (iree_host_size_t i = 0; i < resources->strand_count; ++i) {
    loom_pipeline_worker_t* worker = &workers[i];
    loom_movement_analysis_t* movement = &worker->asynchronous.movement;
    if (!movement->view_regions.value_domain) {
      loom_local_value_domain_restore(&worker->value_domain);
      iree_status_t status = loom_movement_analysis_initialize(
          &worker->facts, &worker->value_domain, arena, movement);
      if (iree_status_is_ok(status)) {
        status = loom_movement_analysis_analyze(movement);
      }
      loom_local_value_domain_release(&worker->value_domain);
      IREE_RETURN_IF_ERROR(status);
    }
    loom_local_value_domain_restore(&worker->value_domain);
    iree_status_t status = loom_storage_access_require_function(
        storage_access, worker->function, &worker->value_domain, &accesses[i]);
    loom_local_value_domain_release(&worker->value_domain);
    IREE_RETURN_IF_ERROR(status);
    for (const loom_storage_access_use_t* use = accesses[i]->uses; use;
         use = use->next_in_function) {
      ++modifying_capacity;
    }
    if (accesses[i]->has_unknown_memory_access) {
      for (const loom_storage_reference_t* reference = accesses[i]->references;
           reference; reference = reference->next) {
        ++modifying_capacity;
      }
    }
  }
  loom_pipeline_storage_reference_t* modifying = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, modifying_capacity, sizeof(*modifying), (void**)&modifying));
  iree_host_size_t modifying_count = 0;
  for (iree_host_size_t i = 0; i < resources->strand_count; ++i) {
    const loom_pipeline_worker_t* worker = &workers[i];
    const loom_storage_access_function_t* access = accesses[i];
    for (const loom_storage_access_use_t* use = access->uses; use;
         use = use->next_in_function) {
      const loom_storage_access_effects_t effects =
          loom_storage_access_use_effects(use);
      const loom_pipeline_storage_reference_t reference =
          loom_pipeline_storage_reference_from_facts(&worker->facts,
                                                     use->value_id);
      if (loom_pipeline_storage_reference_may_modify(effects, &reference)) {
        modifying[modifying_count++] = reference;
      }
    }
    if (!access->has_unknown_memory_access) {
      continue;
    }
    for (const loom_storage_reference_t* reference = access->references;
         reference; reference = reference->next) {
      const loom_pipeline_storage_reference_t storage_reference =
          loom_pipeline_storage_reference_from_facts(&worker->facts,
                                                     reference->value_id);
      if (loom_pipeline_storage_reference_may_modify(LOOM_STORAGE_ACCESS_ESCAPE,
                                                     &storage_reference)) {
        modifying[modifying_count++] = storage_reference;
      }
    }
  }
  // Closed channel membership and admitted local async streams account for
  // their own accesses and completion. A source is safe to read ahead only
  // when every other write or unbounded reference use is disjoint by storage
  // space or by the construction's retained noalias partition.
  for (iree_host_size_t i = 0; i < resources->strand_count; ++i) {
    loom_pipeline_worker_t* worker = &workers[i];
    if (!worker->transport.induction) {
      continue;
    }
    bool stable = true;
    loom_local_value_domain_restore(&worker->value_domain);
    for (iree_host_size_t j = 0; j < worker->transport.stream_count; ++j) {
      const loom_kernel_async_stream_t* stream = worker->transport.streams[j];
      for (iree_host_size_t k = 0; k < stream->transfer_count; ++k) {
        const loom_movement_request_t* request = &stream->transfers[k].request;
        const loom_view_region_t* source = NULL;
        if (!loom_view_region_table_try_lookup(
                &worker->asynchronous.movement.view_regions,
                request->source.value_id, &source) ||
            source->alias_scope_id == LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE ||
            loom_local_value_domain_ordinal(&worker->value_domain,
                                            source->root_value_id) <
                worker->value_domain.definition_count) {
          stable = false;
          continue;
        }
        for (iree_host_size_t l = 0; stable && l < modifying_count; ++l) {
          stable = !loom_pipeline_storage_references_may_alias(source,
                                                               &modifying[l]);
        }
      }
    }
    loom_local_value_domain_release(&worker->value_domain);
    worker->transport.sources_stable = stable;
  }
  return iree_ok_status();
}

iree_status_t loom_pipeline_workers_build(
    loom_module_t* module, const loom_pipeline_resources_t* resources,
    const loom_value_fact_table_t* construction_facts,
    const loom_pipeline_channel_binding_t* incoming,
    iree_host_size_t incoming_count,
    const loom_channel_plan_callable_t* const* callables,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_pipeline_worker_t** out_workers, bool* out_valid) {
  *out_workers = NULL;
  *out_valid = false;
  loom_pipeline_worker_t* workers = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, resources->strand_count, sizeof(*workers), (void**)&workers));
  loom_storage_access_scope_t storage_access;
  loom_storage_access_scope_initialize(module, arena, &storage_access);
  for (iree_host_size_t i = 0; i < resources->strand_count; ++i) {
    workers[i] = (loom_pipeline_worker_t){.source = &resources->strands[i]};
    bool valid = false;
    IREE_RETURN_IF_ERROR(loom_pipeline_worker_build(
        module, resources, construction_facts, incoming, incoming_count,
        callables, diagnostic_emitter, arena, &storage_access, &workers[i],
        &valid));
    if (!valid) {
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(loom_pipeline_workers_prove_transport_sources(
      module, resources, incoming_count, arena, &storage_access, workers));
  *out_workers = workers;
  *out_valid = true;
  return iree_ok_status();
}
