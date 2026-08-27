// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/check/bank_service.h"

#include "loom/target/reporting/report.h"
#include "loom/tools/loom-check/diagnostics.h"

static bool loom_amdgpu_bank_service_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("amdgpu-bank-service"));
}

static iree_string_view_t loom_amdgpu_bank_service_check_non_empty(
    iree_string_view_t value) {
  return iree_string_view_is_empty(value) ? IREE_SV("-") : value;
}

static iree_status_t loom_amdgpu_bank_service_check_append_row(
    const loom_target_compile_report_source_low_memory_row_t* row,
    iree_string_builder_t* builder) {
  const loom_target_compile_report_bank_service_t* bank_service =
      &row->bank_service;
  const iree_string_view_t source_root =
      loom_amdgpu_bank_service_check_non_empty(row->source_root_name);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "bank-service function=%.*s source-op=%.*s root=%.*s packet=%.*s "
      "model=%.*s proof=%.*s lane-address=%.*s active-lanes=%.*s "
      "base-residue=%.*s",
      (int)row->function_name.size, row->function_name.data,
      (int)row->source_op_name.size, row->source_op_name.data,
      (int)source_root.size, source_root.data, (int)row->packet_key.size,
      row->packet_key.data, (int)bank_service->model_key.size,
      bank_service->model_key.data, (int)bank_service->proof.size,
      bank_service->proof.data, (int)bank_service->lane_address_proof.size,
      bank_service->lane_address_proof.data,
      (int)bank_service->active_lane_proof.size,
      bank_service->active_lane_proof.data,
      (int)bank_service->base_residue_proof.size,
      bank_service->base_residue_proof.data));
  if (!iree_string_view_is_empty(bank_service->classification)) {
    return iree_string_builder_append_format(
        builder,
        " classification=%.*s required-rounds=%u uncontended-rounds=%u "
        "extra-rounds=%u maximum-multiplicity=%u\n",
        (int)bank_service->classification.size,
        bank_service->classification.data, bank_service->required_rounds,
        bank_service->uncontended_rounds, bank_service->extra_rounds,
        bank_service->maximum_request_multiplicity);
  }
  return iree_string_builder_append_format(
      builder, " unknown-reason=%.*s\n", (int)bank_service->unknown_reason.size,
      bank_service->unknown_reason.data);
}

static iree_status_t loom_amdgpu_bank_service_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  if (!iree_string_view_is_empty(
          iree_string_view_trim(request->target_options))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdgpu-bank-service accepts no options");
  }

  loom_target_compile_report_t report = {0};
  loom_target_compile_report_initialize(
      &report, iree_arena_allocator(request->case_arena));
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;

  loom_check_prepare_source_low_options_t prepare_options = {0};
  loom_check_prepare_source_low_options_initialize(&prepare_options);
  prepare_options.report = &report;
  IREE_RETURN_IF_ERROR(loom_check_prepare_source_low_module(
      request->module, &prepare_options, request->low_registry,
      request->environment, request->source_resolver,
      request->diagnostic_collector, request->block_pool));
  if (request->diagnostic_collector->count != 0) {
    return iree_ok_status();
  }

  for (const loom_target_compile_report_vec_t* vec =
           report.source_low_memory_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_target_compile_report_source_low_memory_row_t* rows =
        (const loom_target_compile_report_source_low_memory_row_t*)
            loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      if (iree_string_view_is_empty(rows[i].bank_service.model_key)) continue;
      IREE_RETURN_IF_ERROR(loom_amdgpu_bank_service_check_append_row(
          &rows[i], &request->result->actual_output));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_bank_service_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "amdgpu-bank-service");
}

const loom_check_emit_provider_t
    loom_amdgpu_bank_service_loom_check_emit_provider = {
        .name = IREE_SVL("amdgpu-bank-service"),
        .match = loom_amdgpu_bank_service_check_emit_provider_matches,
        .execute = loom_amdgpu_bank_service_check_emit_provider_execute,
        .append_names =
            loom_amdgpu_bank_service_check_emit_provider_append_names,
};
