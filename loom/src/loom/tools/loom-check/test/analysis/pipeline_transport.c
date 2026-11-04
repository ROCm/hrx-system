// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>

#include "loom/analysis/pipeline_resources.h"
#include "loom/analysis/pipeline_workers.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/test/analysis/providers.h"

static bool loom_check_test_pipeline_transport_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("pipeline-transport"));
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

static iree_status_t loom_check_test_pipeline_transport_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request,
    const loom_check_emit_native_module_t* native_module) {
  (void)provider;
  const iree_string_view_t selection =
      iree_string_view_trim(request->target_options);
  if (!iree_string_view_starts_with_char(selection, '@') ||
      selection.size < 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipeline-transport requires @pipeline");
  }
  loom_module_t* module = native_module->module;
  const loom_string_id_t name = loom_module_lookup_string(
      module, iree_string_view_substr(selection, 1, IREE_HOST_SIZE_MAX));
  const uint16_t symbol = loom_module_find_symbol(module, name);
  if (symbol == LOOM_SYMBOL_ID_INVALID ||
      !loom_pipeline_def_isa(module->symbols.entries[symbol].defining_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipeline-transport requires a pipeline definition");
  }
  const loom_func_like_t function =
      loom_func_like_cast(module, module->symbols.entries[symbol].defining_op);
  iree_arena_allocator_t* arena = request->case_arena;
  loom_value_fact_table_t facts;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_initialize(&facts, arena, module->values.count));
  loom_type_registry_configure_fact_context(&facts.context);
  IREE_RETURN_IF_ERROR(loom_value_fact_table_compute(&facts, module, function));

  loom_check_diagnostic_emitter_capture_t diagnostic_capture = {
      .diagnostic_collector = request->diagnostic_collector,
      .module = module,
      .source_resolver = native_module->source_resolver,
      .emitter = LOOM_EMITTER_PASS,
  };
  const iree_diagnostic_emitter_t emitter = {
      .fn = loom_check_diagnostic_emitter_capture_emit,
      .user_data = &diagnostic_capture,
  };
  loom_pipeline_resources_t resources;
  bool valid = false;
  IREE_RETURN_IF_ERROR(loom_pipeline_resources_build(
      module, function, &facts, NULL, 0, NULL, 0, NULL, 0, emitter, arena,
      &resources, &valid));
  if (!valid) {
    return iree_ok_status();
  }
  loom_pipeline_worker_t* workers = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_workers_build(module, &resources, &facts,
                                                   NULL, 0, NULL, emitter,
                                                   arena, &workers, &valid));
  if (!valid) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < resources.strand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_check_test_pipeline_transport_print(
        module, &workers[i], i, arena, &request->result->actual_output));
  }
  return iree_ok_status();
}

static iree_status_t loom_check_test_pipeline_transport_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "pipeline-transport");
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
