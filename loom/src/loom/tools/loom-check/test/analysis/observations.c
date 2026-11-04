// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Passive analysis observations for authored compiler tests.

#include "loom/analysis/liveness.h"
#include "loom/analysis/liveness_json.h"
#include "loom/analysis/storage_access.h"
#include "loom/analysis/storage_interference.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/type_registry.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/test/analysis/providers.h"
#include "loom/util/fact_table.h"

typedef enum loom_check_test_analysis_observation_e {
  LOOM_CHECK_TEST_ANALYSIS_OBSERVATION_LIVENESS = 0,
  LOOM_CHECK_TEST_ANALYSIS_OBSERVATION_STORAGE_INTERFERENCE,
} loom_check_test_analysis_observation_t;

static bool loom_check_test_analysis_observation_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("liveness-json")) ||
         iree_string_view_equal(target_name, IREE_SV("liveness")) ||
         iree_string_view_equal(target_name, IREE_SV("storage-interference"));
}

static iree_status_t loom_check_test_analysis_observation_parse(
    const loom_check_emit_provider_request_t* request,
    loom_check_test_analysis_observation_t* out_observation,
    iree_string_view_t* out_symbol_name) {
  if (iree_string_view_equal(request->target_name,
                             IREE_SV("storage-interference"))) {
    *out_observation =
        LOOM_CHECK_TEST_ANALYSIS_OBSERVATION_STORAGE_INTERFERENCE;
  } else {
    *out_observation = LOOM_CHECK_TEST_ANALYSIS_OBSERVATION_LIVENESS;
  }
  iree_string_view_t selection = iree_string_view_trim(request->target_options);
  if (!iree_string_view_consume_prefix(&selection, IREE_SV("@")) ||
      iree_string_view_is_empty(selection)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "%.*s requires a function symbol name",
        (int)request->target_name.size, request->target_name.data);
  }
  *out_symbol_name = selection;
  return iree_ok_status();
}

static iree_status_t loom_check_test_analysis_observation_symbol_not_found(
    const loom_check_emit_provider_request_t* request,
    iree_string_view_t symbol_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(symbol_name),
  };
  return loom_check_diagnostic_collector_emit_case_source(
      request->diagnostic_collector, request->test_case, request->filename,
      LOOM_EMITTER_PASS, LOOM_ERR_SYMBOL_002, params, IREE_ARRAYSIZE(params));
}

static iree_string_view_t
loom_check_test_analysis_observation_symbol_actual_kind(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  if (symbol->definition) {
    return loom_symbol_definition_descriptor_name(symbol->definition);
  }
  if (symbol->defining_op) {
    return loom_op_name(module, symbol->defining_op);
  }
  return IREE_SV("<unknown>");
}

static iree_status_t loom_check_test_analysis_observation_find_function(
    const loom_check_emit_provider_request_t* request,
    const loom_check_emit_native_module_t* native_module,
    iree_string_view_t symbol_name, loom_func_like_t* out_function) {
  *out_function = (loom_func_like_t){0};
  loom_module_t* module = native_module->module;
  const loom_string_id_t name_id =
      loom_module_lookup_string(module, symbol_name);
  if (name_id == LOOM_STRING_ID_INVALID) {
    return loom_check_test_analysis_observation_symbol_not_found(request,
                                                                 symbol_name);
  }
  const uint16_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return loom_check_test_analysis_observation_symbol_not_found(request,
                                                                 symbol_name);
  }

  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  const loom_func_like_t function =
      loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(function) || !loom_func_like_body(function)) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(symbol_name),
        loom_param_string(
            loom_check_test_analysis_observation_symbol_actual_kind(module,
                                                                    symbol)),
        loom_param_string(IREE_SV("function body")),
    };
    const loom_diagnostic_emission_t emission = {
        .op = symbol->defining_op,
        .error = LOOM_ERR_SYMBOL_003,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    loom_check_diagnostic_emitter_capture_t capture = {
        .diagnostic_collector = request->diagnostic_collector,
        .module = module,
        .source_resolver = native_module->source_resolver,
        .emitter = LOOM_EMITTER_PASS,
    };
    const iree_diagnostic_emitter_t emitter = {
        .fn = loom_check_diagnostic_emitter_capture_emit,
        .user_data = &capture,
    };
    return iree_diagnostic_emit(emitter, &emission);
  }
  *out_function = function;
  return iree_ok_status();
}

static iree_status_t loom_check_test_analysis_observe_liveness(
    loom_module_t* module, loom_func_like_t function,
    iree_arena_allocator_t* arena, iree_string_builder_t* output) {
  loom_liveness_analysis_t analysis = {0};
  IREE_RETURN_IF_ERROR(loom_liveness_analyze_region(
      module, loom_func_like_body(function), arena, &analysis));
  return loom_liveness_format_json(&analysis, NULL, output);
}

static iree_status_t loom_check_test_analysis_observe_storage_interference(
    loom_module_t* module, loom_func_like_t function,
    iree_arena_allocator_t* arena, iree_string_builder_t* output) {
  loom_storage_access_scope_t storage_access;
  loom_storage_access_scope_initialize(module, arena, &storage_access);
  loom_value_fact_table_t facts;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_initialize(&facts, arena, module->values.count));
  loom_type_registry_configure_fact_context(&facts.context);
  IREE_RETURN_IF_ERROR(loom_value_fact_table_compute(&facts, module, function));
  loom_local_value_domain_t domain;
  IREE_RETURN_IF_ERROR(loom_local_value_domain_acquire_for_region_tree(
      module, loom_func_like_body(function), arena, &domain));
  loom_storage_interference_t* analysis = NULL;
  loom_value_id_t* roots = NULL;
  iree_status_t status = iree_arena_allocate_array(
      arena, domain.definition_count, sizeof(*roots), (void**)&roots);
  if (iree_status_is_ok(status)) {
    status = loom_storage_interference_analyze_function(
        module, &facts, &domain, function, &storage_access, arena, &analysis);
  }
  iree_host_size_t root_count = 0;
  for (loom_value_ordinal_t i = 0;
       i < domain.definition_count && iree_status_is_ok(status); ++i) {
    const loom_value_id_t value_id = domain.value_ids[i];
    const loom_value_t* value = loom_module_value(module, value_id);
    if (loom_value_is_block_arg(value) ||
        !loom_buffer_alloca_isa(loom_value_def_op(value))) {
      continue;
    }
    roots[root_count++] = value_id;
    const iree_string_view_t name =
        loom_low_diagnostic_value_name(module, value_id);
    status = iree_string_builder_append_format(
        output, "%.*s: accessed=%s single_instance=%s\n", (int)name.size,
        name.data,
        loom_storage_interference_root_may_be_accessed(analysis, value_id)
            ? "yes"
            : "no",
        loom_storage_interference_root_has_single_live_instance(analysis,
                                                                value_id)
            ? "yes"
            : "unproven");
  }
  for (iree_host_size_t i = 0; i < root_count && iree_status_is_ok(status);
       ++i) {
    for (iree_host_size_t j = i + 1;
         j < root_count && iree_status_is_ok(status); ++j) {
      bool nonoverlap = false;
      status = loom_storage_interference_prove_workgroup_nonoverlap(
          analysis, roots[i], roots[j], &nonoverlap);
      if (iree_status_is_ok(status)) {
        const iree_string_view_t lhs =
            loom_low_diagnostic_value_name(module, roots[i]);
        const iree_string_view_t rhs =
            loom_low_diagnostic_value_name(module, roots[j]);
        status = iree_string_builder_append_format(
            output, "%.*s / %.*s: workgroup_nonoverlap=%s\n", (int)lhs.size,
            lhs.data, (int)rhs.size, rhs.data, nonoverlap ? "yes" : "unproven");
      }
    }
  }
  loom_local_value_domain_release(&domain);
  return status;
}

static iree_status_t loom_check_test_analysis_observation_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request,
    const loom_check_emit_native_module_t* native_module) {
  (void)provider;
  loom_check_test_analysis_observation_t observation =
      LOOM_CHECK_TEST_ANALYSIS_OBSERVATION_LIVENESS;
  iree_string_view_t symbol_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_check_test_analysis_observation_parse(
      request, &observation, &symbol_name));
  loom_func_like_t function = {0};
  IREE_RETURN_IF_ERROR(loom_check_test_analysis_observation_find_function(
      request, native_module, symbol_name, &function));
  if (!loom_func_like_isa(function)) {
    return iree_ok_status();
  }
  if (observation ==
      LOOM_CHECK_TEST_ANALYSIS_OBSERVATION_STORAGE_INTERFERENCE) {
    return loom_check_test_analysis_observe_storage_interference(
        native_module->module, function, request->case_arena,
        &request->result->actual_output);
  }
  return loom_check_test_analysis_observe_liveness(
      native_module->module, function, request->case_arena,
      &request->result->actual_output);
}

static iree_status_t loom_check_test_analysis_observation_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(
      builder, "liveness-json, liveness, storage-interference");
}

const loom_check_emit_provider_t loom_check_test_analysis_observation_provider =
    {
        .name = IREE_SVL("analysis-observation"),
        .match = loom_check_test_analysis_observation_matches,
        .execute_native = loom_check_test_analysis_observation_execute,
        .append_names = loom_check_test_analysis_observation_append_names,
};
