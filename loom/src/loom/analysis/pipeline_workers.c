// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline_workers.h"

#include "loom/error/error_catalog.h"
#include "loom/ops/cfg/ops.h"
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

static iree_status_t loom_pipeline_worker_transport(
    loom_module_t* module, iree_arena_allocator_t* arena,
    const loom_value_fact_cfg_region_t* structure,
    loom_pipeline_worker_t* worker) {
  const loom_cfg_graph_t* graph = worker->graph;
  if (!worker->asynchronous.streams ||
      (graph->has_cycles &&
       (!worker->block_execution_counts || structure->loops.loop_count != 1))) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < graph->reverse_postorder.count; ++i) {
    const uint16_t block = graph->reverse_postorder.values[i];
    if (graph->blocks[block].successor_count > 1 &&
        (!graph->has_cycles ||
         structure->loops.loops[0].header_index != block)) {
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
  iree_host_size_t count = 0, stream_count = 0;
  for (iree_host_size_t b = 0; b < graph->reverse_postorder.count; ++b) {
    const uint16_t block_index = graph->reverse_postorder.values[b];
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
  if (stream_count == 0) {
    return iree_ok_status();
  }
  const loom_value_fact_induction_t* induction = NULL;
  uint64_t repetitions = 1;
  if (graph->has_cycles) {
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
  worker->transport = (loom_pipeline_transport_t){.steps = steps,
                                                  .count = count,
                                                  .streams = streams,
                                                  .stream_count = stream_count,
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
    loom_pipeline_worker_t* worker, bool* out_valid) {
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
  for (uint16_t i = 0; i < formal_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, formals[i]);
    loom_value_facts_t facts;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_for_type(
        &worker->facts, construction_facts, module, type,
        loom_value_fact_table_lookup(construction_facts, actuals.values[i]),
        &facts));
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_define(&worker->facts, formals[i], facts));
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
    status = loom_channel_completion_analyze(&worker->channels, worker->graph,
                                             worker->block_execution_counts,
                                             arena, &worker->completion);
  }
  if (iree_status_is_ok(status) &&
      rejection.kind == LOOM_CHANNEL_PLAN_REJECTION_NONE) {
    const loom_kernel_async_legality_options_t options = {
        .value_domain = &worker->value_domain,
        .fact_table = &worker->facts,
        .emitter = diagnostic_emitter,
        .phase_name = IREE_SV("pipeline-worker"),
    };
    status = loom_kernel_async_legality_analyze_function(
        module, worker->function, &options, arena, &worker->asynchronous);
    if (iree_status_is_ok(status) && worker->asynchronous.error_count == 0) {
      status = loom_pipeline_worker_transport(module, arena, structure, worker);
    }
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

static bool loom_pipeline_root_less(const loom_value_id_t* left,
                                    const loom_value_id_t* right) {
  return *left < *right;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_pipeline_sort_roots, loom_value_id_t,
                          loom_pipeline_root_less)

static bool loom_pipeline_root_written(const loom_value_id_t* roots,
                                       iree_host_size_t count,
                                       loom_value_id_t root) {
  iree_host_size_t begin = 0, end = count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (roots[middle] < root) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin < count && roots[begin] == root;
}

static iree_status_t loom_pipeline_workers_prove_transport_sources(
    const loom_pipeline_resources_t* resources, iree_host_size_t incoming_count,
    iree_arena_allocator_t* arena, loom_pipeline_worker_t* workers) {
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
  // Closed channel membership and admitted local async streams account for
  // communication acquisition through the complete execution's writes. Other
  // synchronization and unknown effects still preclude early external reads.
  uint32_t interference = 0;
  iree_host_size_t capacity = 0;
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
    interference |= movement->view_regions.interference_memory_spaces;
    capacity += worker->value_domain.value_count;
  }
  loom_value_id_t* written_roots;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity, sizeof(*written_roots), (void**)&written_roots));
  iree_host_size_t written_count = 0;
  for (iree_host_size_t i = 0; i < resources->strand_count; ++i) {
    const loom_pipeline_worker_t* worker = &workers[i];
    const uint8_t* access = worker->asynchronous.movement.view_regions
                                .storage_flags_by_value_ordinal;
    for (loom_value_ordinal_t j = 0; j < worker->value_domain.value_count;
         ++j) {
      if (iree_any_bit_set(access[j], LOOM_VIEW_ACCESS_WRITE)) {
        written_roots[written_count++] = worker->value_domain.value_ids[j];
      }
    }
  }
  loom_pipeline_sort_roots(written_roots, written_count);
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
                worker->value_domain.definition_count ||
            (interference & (1u << source->memory_space)) ||
            loom_pipeline_root_written(written_roots, written_count,
                                       source->root_value_id)) {
          stable = false;
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
  for (iree_host_size_t i = 0; i < resources->strand_count; ++i) {
    workers[i] = (loom_pipeline_worker_t){.source = &resources->strands[i]};
    bool valid = false;
    IREE_RETURN_IF_ERROR(loom_pipeline_worker_build(
        module, resources, construction_facts, incoming, incoming_count,
        callables, diagnostic_emitter, arena, &workers[i], &valid));
    if (!valid) {
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(loom_pipeline_workers_prove_transport_sources(
      resources, incoming_count, arena, workers));
  *out_workers = workers;
  *out_valid = true;
  return iree_ok_status();
}
