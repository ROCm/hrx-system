// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>

#include "loom/analysis/pipeline/invocations.h"
#include "loom/analysis/pipeline/resources.h"
#include "loom/analysis/pipeline/schedule.h"
#include "loom/analysis/pipeline/workers.h"
#include "loom/analysis/source_storage_packing.h"
#include "loom/ir/context.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/test/analysis/providers.h"

static bool loom_check_test_pipeline_transport_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("pipeline-transport")) ||
         iree_string_view_equal(target_name, IREE_SV("pipeline-invocations"));
}

static iree_status_t loom_check_test_pipeline_schedule_print(
    const loom_module_t* module, const loom_pipeline_schedule_t* schedule,
    int indentation, iree_string_builder_t* output) {
  static const char* const names[] = {
      "dataflow",   "compose", "serial",     "memory",
      "allocation", "channel", "collective", "collective-member",
      "branch",     "yield",
  };
  const loom_cfg_graph_t* graph = &schedule->control->graph;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      "%*sschedule: blocks=%" PRIhsz " loops=%" PRIhsz
      " single_child_choice=%s\n",
      indentation, "", graph->block_count, schedule->control->loops.loop_count,
      schedule->single_child_choice ? "yes" : "no"));
  for (iree_host_size_t b = 0; b < graph->block_count; ++b) {
    const loom_cfg_block_info_t* control = &graph->blocks[b];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output, "%*sblock %" PRIhsz ": successors=", indentation + 2, "", b));
    for (iree_host_size_t e = 0; e < control->successor_count; ++e) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          output, "%s%u", e ? "," : "",
          graph->successor_indices[control->successor_start + e]));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\n"));
    const loom_pipeline_schedule_block_t* block = &schedule->blocks[b];
    for (iree_host_size_t i = 0; i < block->step_count; ++i) {
      const loom_pipeline_schedule_step_t* step = &block->steps[i];
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          output, "%*s%s", indentation + 4, "", names[step->kind]));
      if (step->kind == LOOM_PIPELINE_SCHEDULE_STEP_COMPOSITION) {
        const loom_symbol_ref_t callee =
            loom_pipeline_compose_callee(step->operation);
        const iree_string_view_t name = loom_string_table_get(
            &module->strings,
            module->symbols.entries[callee.symbol_id].name_id);
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            output, " @%.*s arguments=", (int)name.size, name.data));
        const loom_value_id_t* operands =
            loom_op_const_operands(step->operation);
        for (uint16_t a = 0; a < step->operation->operand_count; ++a) {
          const iree_string_view_t argument =
              loom_module_value_name(module, operands[a]);
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              output, "%s%%%.*s", a ? "," : "", (int)argument.size,
              argument.data));
        }
      } else if (step->kind == LOOM_PIPELINE_SCHEDULE_STEP_COLLECTIVE) {
        const loom_pipeline_collective_t* collective = step->collective;
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            output,
            " operations=%" PRIhsz " results=", collective->operation_count));
        for (iree_host_size_t r = 0; r < collective->result_count; ++r) {
          const loom_pipeline_collective_result_t* result =
              &collective->results[r];
          const iree_string_view_t name = loom_module_value_name(
              module,
              loom_op_const_results(
                  collective->operations[result->operation])[result->result]);
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              output, "%s%%%.*s", r ? "," : "", (int)name.size, name.data));
        }
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\n"));
      if (step->nested) {
        IREE_RETURN_IF_ERROR(loom_check_test_pipeline_schedule_print(
            module, step->nested, indentation + 6, output));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_check_test_pipeline_transport_print(
    const loom_module_t* module, const loom_pipeline_worker_t* worker,
    iree_host_size_t worker_index, iree_arena_allocator_t* arena,
    iree_string_builder_t* output) {
  const loom_pipeline_transport_t* transport = &worker->transport;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      "worker %" PRIhsz
      ": transport=%s execution_counts=%s "
      "batches=%" PRIhsz " streams=%" PRIhsz "\n",
      worker_index,
      transport->cohort_count ? "channel-copy"
      : transport->count      ? "borrowed-movement"
                              : "unproven",
      worker->block_execution_counts ? "exact" : "unknown",
      transport->batch_count, transport->stream_count));
  if (transport->induction) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output, "  repetitions=%" PRIu64 " sources_stable=%s\n",
        transport->repetitions, transport->sources_stable ? "yes" : "no"));
  }
  for (iree_host_size_t i = 0; i < transport->batch_count; ++i) {
    const loom_pipeline_transport_batch_t* batch = &transport->batches[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output,
        "  batch %" PRIhsz ": records=%" PRIhsz " replays=%" PRIu64 "\n", i,
        batch->record_count, batch->replays));
    for (iree_host_size_t j = 0; j < batch->record_count; ++j) {
      const loom_pipeline_transport_cohort_t* record = &batch->records[j];
      const iree_string_view_t admission = loom_module_value_name(
          module, loom_op_const_results(record->admission->op)[0]);
      const iree_string_view_t source =
          loom_module_value_name(module, record->admission->channel->value_id);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          output,
          "    record %" PRIhsz
          ": admission=%%%.*s source=%%%.*s "
          "admissions=%" PRIu64 " transfers=%" PRIu64
          " final_release=%s destinations=",
          j, (int)admission.size, admission.data, (int)source.size, source.data,
          record->admissions, record->transfers,
          record->release ? "yes" : "no"));
      for (iree_host_size_t k = 0; k < record->destination_count; ++k) {
        const iree_string_view_t destination = loom_module_value_name(
            module, record->destinations[k].channel->value_id);
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            output, "%s%%%.*s", k ? "," : "", (int)destination.size,
            destination.data));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\n"));
    }
    if (!batch->replays) {
      loom_pipeline_transport_copy_cursor_t cursor;
      IREE_RETURN_IF_ERROR(loom_pipeline_transport_copy_cursor_initialize(
          transport, batch, arena, &cursor));
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(output, "    copy_order:"));
      const loom_pipeline_transport_copy_t* copy = NULL;
      while ((copy = loom_pipeline_transport_copy_cursor_next(&cursor))) {
        const iree_string_view_t source = loom_module_value_name(
            module, loom_op_const_results(copy->source->admission->op)[0]);
        const iree_string_view_t destination = loom_module_value_name(
            module, copy->destination->channel->value_id);
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            output, " %%%.*s->%%%.*s", (int)source.size, source.data,
            (int)destination.size, destination.data));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\n"));
    }
  }
  return iree_ok_status();
}

typedef struct loom_check_test_pipeline_invocations_t {
  // Source module for names and formal argument types.
  const loom_module_t* module;
  // Canonical construction inventory and protocol identity index.
  const loom_pipeline_invocations_t* invocations;
  // Bound execution plans in the construction inventory's worker order.
  const loom_pipeline_worker_t* workers;
  // Exact owner-level result under test.
  iree_string_builder_t* output;
} loom_check_test_pipeline_invocations_t;

static iree_status_t loom_check_test_pipeline_invocation_print(
    loom_check_test_pipeline_invocations_t* state,
    const loom_pipeline_invocation_t* invocation);

static iree_status_t loom_check_test_pipeline_invocation_schedule_print(
    loom_check_test_pipeline_invocations_t* state,
    const loom_pipeline_invocation_schedule_t* schedule) {
  for (iree_host_size_t b = 0; b < schedule->source->control->graph.block_count;
       ++b) {
    const loom_pipeline_schedule_block_t* block = &schedule->source->blocks[b];
    for (iree_host_size_t i = 0; i < block->step_count; ++i) {
      const loom_pipeline_schedule_step_t* step = &block->steps[i];
      if (step->kind == LOOM_PIPELINE_SCHEDULE_STEP_COMPOSITION) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            state->output, "  joined block=%" PRIhsz " step=%" PRIhsz "\n", b,
            i));
        IREE_RETURN_IF_ERROR(loom_check_test_pipeline_invocation_print(
            state, schedule->blocks[b][i].child));
      } else if (step->kind == LOOM_PIPELINE_SCHEDULE_STEP_SERIAL) {
        IREE_RETURN_IF_ERROR(loom_check_test_pipeline_invocation_schedule_print(
            state, schedule->blocks[b][i].nested));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_check_test_pipeline_invocation_print(
    loom_check_test_pipeline_invocations_t* state,
    const loom_pipeline_invocation_t* invocation) {
  const iree_host_size_t index = invocation->ordinal;
  const loom_symbol_ref_t symbol = loom_func_like_callee(invocation->function);
  const iree_string_view_t name = loom_string_table_get(
      &state->module->strings,
      state->module->symbols.entries[symbol.symbol_id].name_id);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      state->output,
      "invocation %" PRIhsz " @%.*s strands=%" PRIhsz " channels=%" PRIhsz "\n",
      index, (int)name.size, name.data, invocation->resources->strand_count,
      invocation->resources->channel_count));
  for (iree_host_size_t i = 0; i < invocation->arguments.channel_count; ++i) {
    const loom_pipeline_channel_binding_t* binding =
        &invocation->arguments.channels[i];
    const iree_string_view_t formal =
        loom_module_value_name(state->module, binding->value_id);
    const iree_string_view_t actual = loom_module_value_name(
        state->module, binding->channel->identity.value_id);
    const iree_host_size_t channel = loom_pipeline_invocations_channel_index(
        state->invocations, &binding->channel->identity);
    const iree_host_size_t owner =
        state->invocations->channels[channel].invocation->ordinal;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        state->output,
        "  channel %%%.*s = %" PRIhsz ":%%%.*s capacity=%" PRIu64
        " inventory=%" PRIhsz "\n",
        (int)formal.size, formal.data, owner, (int)actual.size, actual.data,
        binding->channel->capacity, channel));
  }
  if (invocation->call) {
    uint16_t count = 0;
    const loom_value_id_t* formals =
        loom_func_like_arg_ids(invocation->function, &count);
    for (uint16_t i = 0; i < count; ++i) {
      const loom_type_t type =
          loom_module_value_type(state->module, formals[i]);
      if (!loom_type_is_scalar(type)) {
        continue;
      }
      const iree_string_view_t formal =
          loom_module_value_name(state->module, formals[i]);
      int64_t value = 0;
      if (loom_value_facts_as_exact_i64(
              loom_value_fact_table_lookup(invocation->facts, formals[i]),
              &value)) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            state->output, "  scalar %%%.*s = %" PRId64 "\n", (int)formal.size,
            formal.data, value));
      } else {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            state->output, "  scalar %%%.*s = runtime\n", (int)formal.size,
            formal.data));
      }
    }
  }
  for (iree_host_size_t i = 0; i < invocation->resources->strand_count; ++i) {
    const loom_pipeline_worker_t* worker =
        &state->workers[invocation->first_worker + i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        state->output, "  worker %" PRIhsz " geometry=", i));
    for (iree_host_size_t a = 0; a < worker->rank; ++a) {
      const loom_pipeline_worker_axis_t* axis = &worker->axes[a];
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          state->output, "%s%" PRIu64 ":%" PRIu64 ":%" PRIu64, a ? "," : "",
          axis->origin, axis->count, axis->stride));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        state->output, " actions=%" PRIhsz "\n",
        worker->channels.action_count));
    if (worker->transport.induction) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          state->output, "    repetitions=%" PRIu64 " sources_stable=%s\n",
          worker->transport.repetitions,
          worker->transport.sources_stable ? "yes" : "no"));
    }
    for (const loom_kernel_async_stream_t* stream =
             worker->asynchronous.streams;
         stream; stream = stream->next) {
      for (iree_host_size_t g = 0; g < stream->group_count; ++g) {
        const loom_kernel_async_group_t* group = &stream->groups[g];
        const iree_string_view_t name = loom_module_value_name(
            state->module, loom_op_const_results(group->op)[0]);
        const iree_string_view_t completion = loom_module_value_name(
            state->module, loom_kernel_async_wait_group(group->completion));
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            state->output,
            "    group %%%.*s transfers=%" PRIhsz " completed-by=%%%.*s\n",
            (int)name.size, name.data, group->transfer_count,
            (int)completion.size, completion.data));
      }
    }
  }
  for (iree_host_size_t i = 0; i < invocation->resources->composition_count;
       ++i) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        state->output, "  concurrent %" PRIhsz "\n", i));
    IREE_RETURN_IF_ERROR(loom_check_test_pipeline_invocation_print(
        state, invocation->compositions[i]));
  }
  for (iree_host_size_t i = 0; i < invocation->resources->schedule_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_check_test_pipeline_invocation_schedule_print(
        state, &invocation->schedules[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_check_test_pipeline_transport_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request,
    const loom_check_test_native_module_t* native_module) {
  (void)provider;
  iree_string_view_t selection = iree_string_view_trim(request->target_options);
  const bool admissions =
      iree_string_view_consume_suffix(&selection, IREE_SV(" admissions"));
  if (!iree_string_view_starts_with_char(selection, '@') ||
      selection.size < 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "%.*s requires @pipeline",
        (int)request->target_name.size, request->target_name.data);
  }
  loom_module_t* module = native_module->module;
  const loom_string_id_t name = loom_module_lookup_string(
      module, iree_string_view_substr(selection, 1, IREE_HOST_SIZE_MAX));
  const uint16_t symbol = loom_module_find_symbol(module, name);
  if (symbol == LOOM_SYMBOL_ID_INVALID ||
      !loom_pipeline_def_isa(module->symbols.entries[symbol].defining_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "%.*s requires a pipeline definition",
        (int)request->target_name.size, request->target_name.data);
  }
  const loom_func_like_t function =
      loom_func_like_cast(module, module->symbols.entries[symbol].defining_op);
  iree_arena_allocator_t* arena = request->case_arena;
  loom_value_fact_table_t facts;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_initialize(&facts, arena, module->values.count));
  loom_type_registry_configure_fact_context(&facts.context);
  IREE_RETURN_IF_ERROR(loom_value_fact_table_compute(&facts, module, function));

  loom_check_test_native_diagnostic_emitter_capture_t diagnostic_capture = {
      .context = native_module->diagnostic_context,
      .module = module,
      .source_resolver = native_module->source_resolver,
      .emitter = LOOM_EMITTER_PASS,
  };
  const iree_diagnostic_emitter_t emitter = {
      .fn = loom_check_test_native_diagnostic_emitter_capture_emit,
      .user_data = &diagnostic_capture,
  };
  // Each imported pool names a distinct test backing. Realization supplies
  // the same inventory contract from its selected machine; these inspections
  // exercise construction and binding independently of a physical target.
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  loom_pipeline_resource_pool_t* pools = NULL;
  loom_pipeline_resource_pool_binding_t* bindings = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, argument_count, sizeof(*pools), (void**)&pools));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, argument_count, sizeof(*bindings), (void**)&bindings));
  iree_host_size_t pool_count = 0;
  for (uint16_t i = 0; i < argument_count; ++i) {
    if (!loom_type_is_pool(loom_module_value_type(module, arguments[i]))) {
      continue;
    }
    pools[pool_count] = (loom_pipeline_resource_pool_t){
        .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
        .byte_capacity = 65536};
    IREE_RETURN_IF_ERROR(loom_source_storage_packing_allocate(
        (loom_source_storage_packing_interference_callback_t){0}, NULL, 0,
        arena, &pools[pool_count].packing));
    bindings[pool_count] = (loom_pipeline_resource_pool_binding_t){
        .value_id = arguments[i], .pool_index = (uint32_t)pool_count};
    ++pool_count;
  }
  loom_pipeline_resources_t resources;
  bool valid = false;
  IREE_RETURN_IF_ERROR(loom_pipeline_resources_build(
      module, function, &facts, pools, pool_count, NULL, 0, bindings,
      pool_count, emitter, arena, &resources, &valid));
  if (!valid) {
    return iree_ok_status();
  }
  if (iree_string_view_equal(request->target_name,
                             IREE_SV("pipeline-invocations"))) {
    loom_pipeline_invocations_t invocations = {0};
    IREE_RETURN_IF_ERROR(loom_pipeline_invocations_build(
        module, function, &facts, &resources, NULL, 0,
        (loom_pipeline_invocation_environment_t){0}, emitter, arena,
        &invocations, &valid));
    if (!valid) {
      return iree_ok_status();
    }
    loom_pipeline_worker_t* workers = NULL;
    IREE_RETURN_IF_ERROR(loom_pipeline_invocations_analyze_workers(
        module, &invocations, NULL, emitter, arena, &workers, &valid));
    if (!valid) {
      return iree_ok_status();
    }
    loom_check_test_pipeline_invocations_t state = {
        .module = module,
        .invocations = &invocations,
        .workers = workers,
        .output = &request->result->actual_output};
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        state.output,
        "inventory invocations=%" PRIhsz " workers=%" PRIhsz
        " channels=%" PRIhsz "\n",
        invocations.count, invocations.worker_count,
        invocations.channel_count));
    for (iree_host_size_t i = 0; i < invocations.worker_count; ++i) {
      const loom_pipeline_invocation_worker_t* worker = &invocations.workers[i];
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          state.output, "  execution %" PRIhsz " = %" PRIhsz ":%" PRIhsz "\n",
          i, worker->invocation->ordinal,
          i - worker->invocation->first_worker));
    }
    for (iree_host_size_t i = 0; i < invocations.channel_count; ++i) {
      const loom_pipeline_invocation_channel_t* channel =
          &invocations.channels[i];
      const iree_string_view_t name =
          loom_module_value_name(module, channel->source->identity.value_id);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          state.output, "  protocol %" PRIhsz " = %" PRIhsz ":%%%.*s", i,
          channel->invocation->ordinal, (int)name.size, name.data));
      if (channel->allocation) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            state.output, " pool=%u offset=%" PRIu64 " bytes=%" PRIu64 "\n",
            channel->allocation->pool_index, channel->allocation->byte_offset,
            channel->allocation->byte_length));
      } else {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(state.output, " external\n"));
      }
    }
    return loom_check_test_pipeline_invocation_print(&state,
                                                     invocations.entries[0]);
  }
  loom_storage_access_scope_t storage_access;
  loom_storage_access_scope_initialize(module, arena, &storage_access);
  loom_pipeline_worker_t* workers = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, resources.strand_count, sizeof(*workers), (void**)&workers));
  IREE_RETURN_IF_ERROR(loom_pipeline_workers_build(
      module, &resources, &facts, NULL, 0, NULL, NULL, &storage_access, emitter,
      arena, workers, &valid));
  if (!valid) {
    return iree_ok_status();
  }
  if (!resources.schedule_count && !resources.composition_count) {
    IREE_RETURN_IF_ERROR(loom_pipeline_workers_prove_sources(
        &storage_access, workers, resources.strand_count));
  }
  if (resources.schedule_count) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        &request->result->actual_output,
        "concurrent serial scopes=%" PRIhsz "\n", resources.schedule_count));
    for (iree_host_size_t i = 0; i < resources.schedule_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_check_test_pipeline_schedule_print(
          module, resources.schedules[i], 0, &request->result->actual_output));
    }
  }
  for (iree_host_size_t i = 0; i < resources.strand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_check_test_pipeline_transport_print(
        module, &workers[i], i, arena, &request->result->actual_output));
    if (admissions && workers[i].completion.requirement ==
                          LOOM_CHANNEL_COMPLETION_REQUIREMENT_NONE) {
      for (iree_host_size_t j = 0; j < workers[i].channels.action_count; ++j) {
        const loom_channel_plan_action_t* action =
            &workers[i].channels.actions[j];
        const iree_string_view_t name = loom_op_name(module, action->op);
        const iree_string_view_t channel =
            action->channel
                ? loom_module_value_name(module, action->channel->value_id)
                : iree_string_view_empty();
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            &request->result->actual_output,
            "  %.*s %%%.*s: maximum_admissions=%u\n", (int)name.size, name.data,
            (int)channel.size, channel.data,
            workers[i].completion.actions[j].maximum_admissions));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_check_test_pipeline_transport_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(
      builder, "pipeline-transport, pipeline-invocations");
}

const loom_check_test_native_emit_provider_t
    loom_check_test_pipeline_transport_provider = {
        .base =
            {
                .name = IREE_SVL("pipeline-transport"),
                .match = loom_check_test_pipeline_transport_matches,
                .execute = loom_check_test_native_emit_provider_execute,
                .append_names = loom_check_test_pipeline_transport_append_names,
            },
        .execute_native = loom_check_test_pipeline_transport_execute,
};
