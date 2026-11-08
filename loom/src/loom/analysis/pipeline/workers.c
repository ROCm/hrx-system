// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/workers.h"

#include <stdlib.h>

#include "loom/analysis/storage_access.h"
#include "loom/analysis/symbolic_projection.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/util/adaptive_sort.h"
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
    if (action->kind == LOOM_CHANNEL_PLAN_ACTION_KIND_ACQUIRE ||
        action->kind == LOOM_CHANNEL_PLAN_ACTION_KIND_ACCEPT) {
      admission_sequence += consumed_since_admission;
      consumed_since_admission = false;
      if (action->kind == LOOM_CHANNEL_PLAN_ACTION_KIND_ACQUIRE &&
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
    } else if (action->kind == LOOM_CHANNEL_PLAN_ACTION_KIND_RESERVE) {
      if (!loom_value_has_no_uses(
              loom_module_value(module, loom_channel_reserve_view(op)))) {
        return iree_ok_status();
      }
    } else if (action->kind == LOOM_CHANNEL_PLAN_ACTION_KIND_COPY ||
               action->kind == LOOM_CHANNEL_PLAN_ACTION_KIND_RELEASE) {
      consumed_since_admission = true;
    } else if (action->kind != LOOM_CHANNEL_PLAN_ACTION_KIND_FANOUT) {
      return iree_ok_status();
    }
  }
  // Membership owns forwarding through CFG arguments and nested fanouts.
  // A common source site alone does not establish one dynamic record: copies
  // must stay in its admission scope or in nested replay loops, with final
  // release back in that same admission scope.
  for (iree_host_size_t i = 0; i < step_count; ++i) {
    const loom_channel_plan_action_t* action = steps[i].source.channel;
    const loom_op_t* op = action->op;
    const bool release = action->kind == LOOM_CHANNEL_PLAN_ACTION_KIND_RELEASE;
    if (!release && action->kind != LOOM_CHANNEL_PLAN_ACTION_KIND_COPY) {
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
    if (action->kind != LOOM_CHANNEL_PLAN_ACTION_KIND_COPY) {
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

static uint64_t loom_pipeline_transport_prefix_cycles(
    const loom_cfg_loop_nest_t* loops,
    const loom_pipeline_transport_batch_t* batches, iree_host_size_t count) {
  if (count < 2) {
    return 0;
  }
  const loom_pipeline_transport_cohort_t* first = batches[0].records;
  const loom_pipeline_transport_cohort_t* bulk = batches[count - 1].records;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_pipeline_transport_cohort_t* cohort = batches[i].records;
    if (batches[i].record_count != 1 || batches[i].replays != 1 ||
        !loom_channel_acquire_isa(cohort->admission->op) || cohort->release ||
        cohort->copy_scope != cohort->admission_scope ||
        (i + 1 < count && (cohort->admission_scope != first->admission_scope ||
                           cohort->admissions != first->admissions))) {
      return 0;
    }
  }
  if (bulk->admissions % first->admissions ||
      (first->admission_scope != LOOM_CFG_LOOP_NEST_NONE &&
       !loom_cfg_loop_nest_contains(
           loops, first->admission_scope,
           bulk->admission->op->parent_block->region_index))) {
    return 0;
  }
  return first->admissions;
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

// Resolves one retained endpoint expression over its controlling recurrence.
// A complete address period restarts at the same offset; partial or rotated
// periods require a more general sequencer than this compact representation.
static bool loom_pipeline_transport_input_address(
    const loom_pipeline_worker_t* worker,
    const loom_value_fact_cfg_region_t* structure, uint16_t scope,
    const loom_symbolic_expr_t* offset,
    loom_pipeline_transport_input_phase_t* phase) {
  phase->source_byte_offset = offset->constant;
  phase->period = phase->transfers;
  if (!offset->term_count) {
    return loom_symbolic_expr_is_constant(offset);
  }
  if (scope == LOOM_CFG_LOOP_NEST_NONE ||
      !loom_symbolic_expr_is_linear(offset) || offset->term_count != 1 ||
      phase->transfers % worker->loop_trip_counts[scope]) {
    return false;
  }
  phase->period = worker->loop_trip_counts[scope];
  const loom_value_fact_induction_t* induction = &structure->inductions[scope];
  int64_t initial, increment, displacement;
  if (!loom_value_facts_as_exact_i64(
          loom_value_fact_recurrence_operand_facts(&worker->facts,
                                                   induction->initial_value),
          &initial) ||
      !loom_value_facts_as_exact_i64(loom_value_fact_recurrence_operand_facts(
                                         &worker->facts, induction->step),
                                     &increment)) {
    return false;
  }
  if (offset->terms[0].value_id != induction->value) {
    loom_symbolic_expr_summary_t summary;
    if (!loom_symbolic_expr_context_try_lookup_summary(
            worker->asynchronous.movement.view_regions.expression_context,
            offset->terms[0].value_id, &summary) ||
        !summary.projection ||
        summary.projection->value_id != induction->value) {
      return false;
    }
    const loom_symbolic_projection_t* projection = summary.projection;
    if (!iree_checked_mul_i64(projection->scale, initial, &initial) ||
        !iree_checked_add_i64(initial, projection->offset, &initial) ||
        !iree_checked_mul_i64(projection->scale, increment, &increment) ||
        increment < 0 || increment % projection->divisor) {
      return false;
    }
    initial /= projection->divisor;
    increment /= projection->divisor;
    if (projection->modulus) {
      initial %= projection->modulus;
      increment %= projection->modulus;
      if (increment &&
          phase->period - 1 >
              (uint64_t)((projection->modulus - 1 - initial) / increment)) {
        if (initial >= increment || projection->modulus % increment) {
          return false;
        }
        const uint64_t period = (uint64_t)(projection->modulus / increment);
        if (phase->period % period) {
          return false;
        }
        phase->period = period;
      }
    }
  }
  return iree_checked_mul_i64(offset->terms[0].coefficient, initial,
                              &displacement) &&
         iree_checked_add_i64(phase->source_byte_offset, displacement,
                              &phase->source_byte_offset) &&
         iree_checked_mul_i64(offset->terms[0].coefficient, increment,
                              &phase->source_byte_stride);
}

// Each phase completes its copy before publishing the destination. The source
// lease remains independent of the phase's repetition and address period.
static bool loom_pipeline_transport_input_phase(
    const loom_pipeline_worker_t* worker,
    const loom_value_fact_cfg_region_t* structure,
    const loom_pipeline_transport_block_t* blocks,
    const loom_pipeline_transport_step_t* body,
    const loom_pipeline_transport_input_t* input,
    loom_pipeline_transport_input_phase_t* phase) {
  if (!loom_channel_reserve_isa(body[0].op) ||
      body[1].kind != LOOM_PIPELINE_TRANSPORT_STEP_TRANSFER ||
      body[2].kind != LOOM_PIPELINE_TRANSPORT_STEP_WAIT ||
      !loom_channel_publish_isa(body[3].op) ||
      loom_channel_publish_write(body[3].op) !=
          loom_channel_reserve_write(body[0].op)) {
    return false;
  }
  const uint16_t block = body[1].op->parent_block->region_index;
  const loom_kernel_async_stream_t* stream = blocks[block].stream;
  const loom_kernel_async_transfer_t* transfer = body[1].source.transfer;
  if (transfer->request.kind != LOOM_MOVEMENT_KIND_KERNEL_ASYNC_COPY ||
      stream->groups[transfer->group_index].completion != body[2].op) {
    return false;
  }
  const uint64_t executions = worker->block_execution_counts[block];
  if (!executions || executions % input->admissions) {
    return false;
  }
  for (iree_host_size_t i = 0; i < 4; ++i) {
    if (body[i].scope != body[1].scope ||
        worker->block_execution_counts[body[i]
                                           .op->parent_block->region_index] !=
            executions) {
      return false;
    }
  }
  if (input->acquisition) {
    const uint16_t admission_scope = loom_cfg_loop_nest_innermost(
        &structure->loops, input->acquisition->op->parent_block->region_index);
    if (admission_scope != LOOM_CFG_LOOP_NEST_NONE &&
        !loom_cfg_loop_nest_contains(&structure->loops, admission_scope,
                                     block)) {
      return false;
    }
  }
  *phase = (loom_pipeline_transport_input_phase_t){
      .reservation = body[0].source.channel,
      .publication = body[3].source.channel,
      .transfer = transfer,
      .transfers = executions / input->admissions,
      .scope = body[1].scope};
  const loom_view_region_t *source, *destination;
  loom_view_region_table_try_lookup(&worker->asynchronous.movement.view_regions,
                                    transfer->request.source.value_id, &source);
  loom_view_region_table_try_lookup(&worker->asynchronous.movement.view_regions,
                                    transfer->request.dest.value_id,
                                    &destination);
  if (destination->base_view_value_id !=
          loom_channel_reserve_view(body[0].op) ||
      !loom_symbolic_expr_is_constant(&destination->projection_byte_offset)) {
    return false;
  }
  phase->destination_byte_offset = destination->projection_byte_offset.constant;
  const loom_symbolic_expr_t* offset = &source->begin_byte_offset;
  if (input->acquisition) {
    if (source->base_view_value_id !=
        loom_channel_acquire_view(input->acquisition->op)) {
      return false;
    }
    offset = &source->projection_byte_offset;
  }
  return loom_pipeline_transport_input_address(worker, structure, phase->scope,
                                               offset, phase);
}

static iree_status_t loom_pipeline_transport_input(
    iree_arena_allocator_t* arena,
    const loom_value_fact_cfg_region_t* structure,
    const loom_pipeline_transport_block_t* blocks,
    loom_pipeline_worker_t* worker) {
  const loom_pipeline_transport_t* transport = &worker->transport;
  if (transport->count < 4 || worker->completion.requirement !=
                                  LOOM_CHANNEL_COMPLETION_REQUIREMENT_NONE) {
    return iree_ok_status();
  }
  const loom_pipeline_transport_step_t* steps = transport->steps;
  const bool acquired = loom_channel_acquire_isa(steps[0].op);
  const iree_host_size_t phase_sites = transport->count - (acquired ? 2 : 0);
  if (!phase_sites || phase_sites % 4) {
    return iree_ok_status();
  }
  loom_pipeline_transport_input_t input = {.admissions = 1,
                                           .phase_count = phase_sites / 4};
  if (acquired) {
    const loom_pipeline_transport_step_t* final = &steps[transport->count - 1];
    if (!loom_channel_release_isa(final->op) ||
        steps[0].scope != final->scope ||
        loom_channel_plan_admission(&worker->channels,
                                    loom_channel_release_read(final->op)) !=
            loom_channel_acquire_read(steps[0].op)) {
      return iree_ok_status();
    }
    input.admissions =
        worker->block_execution_counts[steps[0].op->parent_block->region_index];
    if (!input.admissions ||
        worker->block_execution_counts[final->op->parent_block->region_index] !=
            input.admissions) {
      return iree_ok_status();
    }
    input.acquisition = steps[0].source.channel;
    input.release = final->source.channel;
  }
  const loom_pipeline_transport_step_t* body = steps + acquired;
  loom_pipeline_transport_input_phase_t first;
  if (!loom_pipeline_transport_input_phase(worker, structure, blocks, body,
                                           &input, &first)) {
    return iree_ok_status();
  }
  loom_pipeline_transport_input_t* retained;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      arena, sizeof(*retained) + input.phase_count * sizeof(first),
      (void**)&retained));
  loom_pipeline_transport_input_phase_t* phases =
      (loom_pipeline_transport_input_phase_t*)(retained + 1);
  phases[0] = first;
  for (iree_host_size_t i = 1; i < input.phase_count; ++i) {
    if (!loom_pipeline_transport_input_phase(
            worker, structure, blocks, &body[i * 4], &input, &phases[i])) {
      return iree_ok_status();
    }
  }
  input.phases = phases;
  *retained = input;
  worker->transport.input = retained;
  return iree_ok_status();
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
  const uint16_t* order = worker->block_order;
  iree_host_size_t count = 0, stream_count = 0;
  for (iree_host_size_t b = 0; b < graph->reverse_postorder.count; ++b) {
    const uint16_t block_index = order[b];
    const uint16_t scope =
        loom_cfg_loop_nest_innermost(&structure->loops, block_index);
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
            .scope = scope,
            .source.channel = &worker->channels.actions[channel++]};
      } else if (stream && transfer < stream->transfer_count &&
                 stream->transfers[transfer].request.op == op) {
        steps[count++] = (loom_pipeline_transport_step_t){
            .op = op,
            .kind = LOOM_PIPELINE_TRANSPORT_STEP_TRANSFER,
            .scope = scope,
            .source.transfer = &stream->transfers[transfer++]};
      } else if (stream && wait < stream->wait_count &&
                 stream->waits[wait] == op) {
        steps[count++] = (loom_pipeline_transport_step_t){
            .op = op,
            .kind = LOOM_PIPELINE_TRANSPORT_STEP_WAIT,
            .scope = scope};
        ++wait;
      } else if (!iree_any_bit_set(op->traits, LOOM_TRAIT_TERMINATOR) &&
                 !loom_kernel_async_group_isa(op)) {
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
    repetitions = 0;
    if (structure->loops.loop_count == 1) {
      bool uniform = true;
      for (iree_host_size_t i = 0; i < count; ++i) {
        const uint16_t block = steps[i].op->parent_block->region_index;
        uniform &= steps[i].scope == 0 &&
                   block != structure->loops.loops[0].header_index;
      }
      if (uniform && structure->inductions[0].update ==
                         LOOM_VALUE_FACT_INDUCTION_UPDATE_ADDITIVE) {
        induction = &structure->inductions[0];
        const uint16_t block = steps[0].op->parent_block->region_index;
        repetitions = worker->block_execution_counts[block];
      }
    }
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
  worker->transport = (loom_pipeline_transport_t){
      .steps = steps,
      .count = count,
      .streams = streams,
      .stream_count = stream_count,
      .cohorts = cohorts,
      .cohort_count = cohort_count,
      .batches = batches,
      .batch_count = batch_count,
      .prefix_cycles = loom_pipeline_transport_prefix_cycles(
          &structure->loops, batches, batch_count),
      .copy_loops = copy_loops,
      .copy_loop_count = copy_loop_count,
      .repetitions = repetitions,
      .induction = induction};
  if (stream_count) {
    IREE_RETURN_IF_ERROR(
        loom_pipeline_transport_input(arena, structure, blocks, worker));
  }
  return iree_ok_status();
}

static iree_status_t loom_pipeline_worker_execution_counts(
    loom_module_t* module, iree_arena_allocator_t* arena,
    const loom_value_fact_cfg_region_t* structure,
    loom_pipeline_worker_t* worker, const uint64_t** out_execution_bounds) {
  uint64_t* trip_counts = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, structure->loops.loop_count,
                                sizeof(*trip_counts), (void**)&trip_counts));
  bool exact = true;
  for (iree_host_size_t i = 0; i < structure->loops.loop_count; ++i) {
    const loom_loop_recurrence_facts_t recurrence =
        loom_value_fact_induction_facts(&worker->facts, module,
                                        &structure->inductions[i]);
    exact &= recurrence.trip_count_kind == LOOM_LOOP_TRIP_COUNT_EXACT;
    trip_counts[i] = recurrence.trip_count_kind == LOOM_LOOP_TRIP_COUNT_UNKNOWN
                         ? UINT64_MAX
                         : recurrence.trip_count;
  }
  uint64_t* counts;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, structure->graph.block_count, sizeof(*counts), (void**)&counts));
  if (exact) {
    IREE_RETURN_IF_ERROR(loom_cfg_execution_counts_calculate(
        &structure->loops, trip_counts, &structure->control_structure, arena,
        counts, &worker->execution_counts));
  } else {
    loom_cfg_loop_nest_calculate_block_execution_bounds(&structure->loops,
                                                        trip_counts, counts);
  }
  *out_execution_bounds = counts;
  if (worker->execution_counts.values) {
    worker->loop_trip_counts = trip_counts;
  }
  if (worker->execution_counts.single_path) {
    worker->block_execution_counts = counts;
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
  worker->axes = worker->source->axes;
  worker->rank = worker->source->rank;
  loom_region_t* body = loom_func_like_body(worker->function);
  if (!body) {
    return loom_pipeline_workers_reject(worker->source->call,
                                        IREE_SV("a visible strand body"),
                                        diagnostic_emitter);
  }
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_initialize(&worker->facts, arena, 0));
  worker->facts.context.resolve_type_domain =
      construction_facts->context.resolve_type_domain;
  worker->facts.context.resolve_region_target =
      construction_facts->context.resolve_region_target;
  worker->facts.context.target_facts = loom_value_fact_table_block_target_facts(
      construction_facts,
      loom_region_const_entry_block(loom_pipeline_strand_body(declaration)));
  loom_pipeline_arguments_t arguments;
  bool arguments_valid = false;
  IREE_RETURN_IF_ERROR(loom_pipeline_arguments_bind(
      module, worker->source->call, worker->function, resources,
      construction_facts, incoming, incoming_count, &worker->facts,
      diagnostic_emitter, arena, &arguments, &arguments_valid));
  if (!arguments_valid) {
    return iree_ok_status();
  }
  worker->literal_captures = arguments.literal_captures;
  worker->bindings = arguments.channels;
  worker->binding_count = arguments.channel_count;
  loom_channel_plan_binding_t* channel_bindings = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, worker->binding_count,
                                                 sizeof(*channel_bindings),
                                                 (void**)&channel_bindings));
  for (iree_host_size_t i = 0; i < worker->binding_count; ++i) {
    channel_bindings[i] = (loom_channel_plan_binding_t){
        .value_id = worker->bindings[i].value_id,
        .channel = &worker->bindings[i].channel->identity};
  }
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_compute(&worker->facts, module, worker->function));
  const loom_value_fact_cfg_region_t* structure = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_get_or_build_cfg_region(
      &worker->facts, module, body, &structure));
  worker->graph = &structure->graph;
  IREE_RETURN_IF_ERROR(loom_local_value_domain_acquire_for_region(
      module, body, arena, &worker->value_domain));
  loom_channel_plan_rejection_t rejection;
  const uint64_t* execution_bounds = NULL;
  iree_status_t status = loom_channel_plan_build(
      &worker->value_domain, channel_bindings, worker->binding_count, callables,
      arena, &worker->channels, &rejection);
  if (iree_status_is_ok(status) &&
      rejection.kind == LOOM_CHANNEL_PLAN_REJECTION_NONE) {
    if (worker->channels.action_count) {
      status = loom_pipeline_worker_execution_counts(module, arena, structure,
                                                     worker, &execution_bounds);
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
    status = loom_channel_completion_analyze(&worker->channels, worker->graph,
                                             view_regions, execution_bounds,
                                             arena, &worker->completion);
  }
  if (iree_status_is_ok(status) &&
      rejection.kind == LOOM_CHANNEL_PLAN_REJECTION_NONE &&
      worker->asynchronous.error_count == 0) {
    if (!worker->graph->has_cycles) {
      worker->block_order = worker->graph->reverse_postorder.values;
    } else if (worker->execution_counts.values) {
      uint16_t* order = NULL;
      status = iree_arena_allocate_array(arena,
                                         worker->graph->reverse_postorder.count,
                                         sizeof(*order), (void**)&order);
      if (iree_status_is_ok(status)) {
        status =
            loom_cfg_loop_nest_order_blocks(&structure->loops, arena, order);
        worker->block_order = order;
      }
    }
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

// The root index preserves same-storage conflicts even when two projections
// carry different alias facts. The partition index unions conflicting memory
// spaces, so a source queries aggregate effects instead of visiting every
// write.
typedef struct loom_pipeline_modified_partition_t {
  // Comparable noalias partition, or NONE when it can overlap every partition.
  loom_value_fact_alias_scope_id_t identity;
  // Source memory spaces that may overlap a modification in this partition.
  loom_memory_space_mask_t conflicting_spaces;
} loom_pipeline_modified_partition_t;

static bool loom_pipeline_storage_root_less(const loom_value_id_t* lhs,
                                            const loom_value_id_t* rhs) {
  return *lhs < *rhs;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_pipeline_storage_sort_roots, loom_value_id_t,
                          loom_pipeline_storage_root_less)

static bool loom_pipeline_storage_partition_less(
    const loom_pipeline_modified_partition_t* lhs,
    const loom_pipeline_modified_partition_t* rhs) {
  return lhs->identity < rhs->identity;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_pipeline_storage_sort_partitions,
                          loom_pipeline_modified_partition_t,
                          loom_pipeline_storage_partition_less)

static bool loom_pipeline_storage_root_is_modified(
    const loom_value_id_t* roots, iree_host_size_t count,
    loom_value_id_t root_value_id) {
  iree_host_size_t begin = 0, end = count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (roots[middle] == root_value_id) {
      return true;
    }
    if (roots[middle] < root_value_id) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return false;
}

static loom_memory_space_mask_t loom_pipeline_storage_partition_conflicts(
    const loom_pipeline_modified_partition_t* partitions,
    iree_host_size_t count, loom_value_fact_alias_scope_id_t identity) {
  iree_host_size_t begin = 0, end = count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (partitions[middle].identity == identity) {
      return partitions[middle].conflicting_spaces;
    }
    if (partitions[middle].identity < identity) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return 0;
}

iree_status_t loom_pipeline_workers_prove_sources(
    loom_storage_access_scope_t* storage_access,
    loom_pipeline_worker_t* workers, iree_host_size_t worker_count) {
  iree_arena_allocator_t* arena = storage_access->arena;
  bool repeated_transport = false;
  for (iree_host_size_t i = 0; i < worker_count; ++i) {
    repeated_transport |=
        workers[i].transport.stream_count && workers[i].graph->has_cycles;
  }
  if (!repeated_transport) {
    return iree_ok_status();
  }
  const loom_storage_access_function_t** accesses = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, worker_count, sizeof(*accesses), (void**)&accesses));
  iree_host_size_t modifying_capacity = 0;
  loom_memory_space_mask_t unbounded_write_spaces = 0;
  for (iree_host_size_t i = 0; i < worker_count; ++i) {
    loom_pipeline_worker_t* worker = &workers[i];
    loom_local_value_domain_restore(&worker->value_domain);
    iree_status_t status = loom_storage_access_require_function(
        storage_access, worker->function, &worker->value_domain, &accesses[i]);
    loom_local_value_domain_release(&worker->value_domain);
    IREE_RETURN_IF_ERROR(status);
    unbounded_write_spaces |= accesses[i]->unbounded_write_spaces;
    for (const loom_storage_access_use_t* use = accesses[i]->uses; use;
         use = use->next_in_function) {
      ++modifying_capacity;
    }
  }
  loom_value_id_t* modified_roots = NULL;
  loom_pipeline_modified_partition_t* modified_partitions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, modifying_capacity,
                                                 sizeof(*modified_roots),
                                                 (void**)&modified_roots));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, modifying_capacity,
                                                 sizeof(*modified_partitions),
                                                 (void**)&modified_partitions));
  loom_memory_space_mask_t
      memory_conflicts[LOOM_VALUE_FACT_MEMORY_SPACE_COUNT_] = {0};
  for (uint32_t i = 0; i < LOOM_VALUE_FACT_MEMORY_SPACE_COUNT_; ++i) {
    for (uint32_t j = 0; j < LOOM_VALUE_FACT_MEMORY_SPACE_COUNT_; ++j) {
      if (!loom_view_memory_spaces_are_disjoint(i, j)) {
        memory_conflicts[i] |= (loom_memory_space_mask_t)(1u << j);
      }
    }
  }
  iree_host_size_t modifying_count = 0;
  for (iree_host_size_t i = 0; i < worker_count; ++i) {
    const loom_pipeline_worker_t* worker = &workers[i];
    const loom_storage_access_function_t* access = accesses[i];
    for (const loom_storage_access_use_t* use = access->uses; use;
         use = use->next_in_function) {
      const loom_storage_access_effects_t effects =
          loom_storage_access_use_effects(use);
      if (!iree_any_bit_set(effects, LOOM_STORAGE_ACCESS_WRITE |
                                         LOOM_STORAGE_ACCESS_ESCAPE)) {
        continue;
      }
      const loom_pipeline_storage_reference_t reference =
          loom_pipeline_storage_reference_from_facts(&worker->facts,
                                                     use->value_id);
      if (loom_pipeline_storage_reference_may_modify(effects, &reference)) {
        modified_roots[modifying_count] = reference.root_value_id;
        modified_partitions[modifying_count++] =
            (loom_pipeline_modified_partition_t){
                .identity = reference.alias_scope_id,
                .conflicting_spaces = memory_conflicts[reference.memory_space]};
      }
    }
  }
  loom_pipeline_storage_sort_roots(modified_roots, modifying_count);
  loom_pipeline_storage_sort_partitions(modified_partitions, modifying_count);
  iree_host_size_t partition_count = 0;
  for (iree_host_size_t i = 0; i < modifying_count; ++i) {
    if (partition_count && modified_partitions[partition_count - 1].identity ==
                               modified_partitions[i].identity) {
      modified_partitions[partition_count - 1].conflicting_spaces |=
          modified_partitions[i].conflicting_spaces;
    } else {
      modified_partitions[partition_count++] = modified_partitions[i];
    }
  }
  const loom_memory_space_mask_t unknown_conflicts =
      loom_pipeline_storage_partition_conflicts(
          modified_partitions, partition_count,
          LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE);
  // Closed channel membership and admitted local async streams account for
  // their own accesses and completion. A source is safe to read ahead only
  // when every other write or reference escape is disjoint by storage space
  // or by the construction's retained noalias partition. Unbounded effects
  // can reach storage absent from call operands; their retained space bounds
  // exclude disjoint storage without implying purity.
  for (iree_host_size_t i = 0; i < worker_count; ++i) {
    loom_pipeline_worker_t* worker = &workers[i];
    if (!worker->transport.stream_count || !worker->graph->has_cycles) {
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
        if (source->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT) {
          for (uint32_t space = 0;
               stable && space < LOOM_VALUE_FACT_MEMORY_SPACE_COUNT_; ++space) {
            if (unbounded_write_spaces & (1u << space)) {
              stable = loom_view_memory_spaces_are_disjoint(
                  source->memory_space, (loom_value_fact_memory_space_t)space);
            }
          }
        }
        const loom_memory_space_mask_t conflicts =
            unknown_conflicts |
            loom_pipeline_storage_partition_conflicts(
                modified_partitions, partition_count, source->alias_scope_id);
        stable &= !(conflicts & (1u << source->memory_space)) &&
                  !loom_pipeline_storage_root_is_modified(
                      modified_roots, modifying_count, source->root_value_id);
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
    iree_host_size_t incoming_count, const loom_func_like_t* functions,
    const loom_channel_plan_callable_t* const* callables,
    loom_storage_access_scope_t* storage_access,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_pipeline_worker_t* workers, bool* out_valid) {
  *out_valid = false;
  for (iree_host_size_t i = 0; i < resources->strand_count; ++i) {
    const loom_pipeline_resource_strand_t* strand = &resources->strands[i];
    const loom_symbol_ref_t callee = loom_func_call_callee(strand->call);
    workers[i] = (loom_pipeline_worker_t){
        .source = strand,
        .function =
            functions
                ? functions[i]
                : loom_func_like_cast(
                      module,
                      module->symbols.entries[callee.symbol_id].defining_op)};
    bool valid = false;
    IREE_RETURN_IF_ERROR(loom_pipeline_worker_build(
        module, resources, construction_facts, incoming, incoming_count,
        callables, diagnostic_emitter, arena, storage_access, &workers[i],
        &valid));
    if (!valid) {
      return iree_ok_status();
    }
  }
  *out_valid = true;
  return iree_ok_status();
}
