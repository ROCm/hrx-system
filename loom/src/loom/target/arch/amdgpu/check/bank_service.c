// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/check/bank_service.h"

#include <inttypes.h>

#include "iree/base/internal/json.h"
#include "loom/tools/loom-check/compile.h"
#include "loomc/artifact.h"
#include "loomc/byte_sequence.h"
#include "loomc/iree.h"

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

static iree_status_t loom_amdgpu_bank_service_check_lookup_string(
    iree_string_view_t object, iree_string_view_t key,
    iree_arena_allocator_t* arena, iree_string_view_t* out_value) {
  iree_string_view_t encoded = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_try_lookup_object_value(object, key, &encoded));
  if (iree_string_view_equal(encoded, IREE_SV("null"))) {
    *out_value = iree_string_view_empty();
    return iree_ok_status();
  }
  *out_value = encoded;
  if (iree_string_view_is_empty(encoded) ||
      iree_string_view_find_char(encoded, '\\', 0) == IREE_STRING_VIEW_NPOS) {
    return iree_ok_status();
  }
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_json_unescape_string(encoded, 0, NULL, &length));
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, length, (void**)&storage));
  IREE_RETURN_IF_ERROR(
      iree_json_unescape_string(encoded, length, storage, &length));
  *out_value = iree_make_string_view(storage, length);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_bank_service_check_lookup_uint64(
    iree_string_view_t object, iree_string_view_t key, uint64_t* out_value) {
  iree_string_view_t encoded = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(object, key, &encoded));
  return iree_json_parse_uint64(encoded, out_value);
}

typedef struct loom_amdgpu_bank_service_check_row_context_t {
  // Arena owning any unescaped report strings for this case.
  iree_arena_allocator_t* arena;
  // Output receiving one concise line per bank-service row.
  iree_string_builder_t* builder;
} loom_amdgpu_bank_service_check_row_context_t;

static iree_status_t loom_amdgpu_bank_service_check_append_row(
    void* user_data, iree_host_size_t row_index, iree_string_view_t row) {
  (void)row_index;
  loom_amdgpu_bank_service_check_row_context_t* context = user_data;
  iree_string_view_t bank_service = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      row, IREE_SV("bank_service"), &bank_service));
  if (iree_string_view_is_empty(bank_service)) {
    return iree_ok_status();
  }

  iree_string_view_t function_name = iree_string_view_empty();
  iree_string_view_t source_op_name = iree_string_view_empty();
  iree_string_view_t source_root_name = iree_string_view_empty();
  iree_string_view_t packet_key = iree_string_view_empty();
  iree_string_view_t proof = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_bank_service_check_lookup_string(
      row, IREE_SV("function"), context->arena, &function_name));
  IREE_RETURN_IF_ERROR(loom_amdgpu_bank_service_check_lookup_string(
      row, IREE_SV("source_op"), context->arena, &source_op_name));
  IREE_RETURN_IF_ERROR(loom_amdgpu_bank_service_check_lookup_string(
      row, IREE_SV("source_root"), context->arena, &source_root_name));
  IREE_RETURN_IF_ERROR(loom_amdgpu_bank_service_check_lookup_string(
      row, IREE_SV("packet"), context->arena, &packet_key));
  IREE_RETURN_IF_ERROR(loom_amdgpu_bank_service_check_lookup_string(
      bank_service, IREE_SV("proof"), context->arena, &proof));
  const iree_string_view_t source_root =
      loom_amdgpu_bank_service_check_non_empty(source_root_name);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      context->builder,
      "bank-service function=%.*s source-op=%.*s root=%.*s packet=%.*s",
      (int)function_name.size, function_name.data, (int)source_op_name.size,
      source_op_name.data, (int)source_root.size, source_root.data,
      (int)packet_key.size, packet_key.data));

  iree_string_view_t model = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(bank_service, IREE_SV("model"), &model));
  iree_string_view_t unknown_reason = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_bank_service_check_lookup_string(
      bank_service, IREE_SV("unknown_reason"), context->arena,
      &unknown_reason));
  if (iree_string_view_equal(model, IREE_SV("null"))) {
    uint64_t wave_size = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_bank_service_check_lookup_uint64(
        bank_service, IREE_SV("wave_size"), &wave_size));
    return iree_string_builder_append_format(
        context->builder,
        " proof=%.*s wave-size=%" PRIu64 " unknown-reason=%.*s\n",
        (int)proof.size, proof.data, wave_size, (int)unknown_reason.size,
        unknown_reason.data);
  }

  iree_string_view_t model_key = iree_string_view_empty();
  iree_string_view_t address = iree_string_view_empty();
  iree_string_view_t lane_address_proof = iree_string_view_empty();
  iree_string_view_t active_lane_proof = iree_string_view_empty();
  iree_string_view_t base_residue_proof = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_bank_service_check_lookup_string(
      model, IREE_SV("key"), context->arena, &model_key));
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      bank_service, IREE_SV("address"), &address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_bank_service_check_lookup_string(
      address, IREE_SV("lane_address_proof"), context->arena,
      &lane_address_proof));
  IREE_RETURN_IF_ERROR(loom_amdgpu_bank_service_check_lookup_string(
      address, IREE_SV("active_lane_proof"), context->arena,
      &active_lane_proof));
  IREE_RETURN_IF_ERROR(loom_amdgpu_bank_service_check_lookup_string(
      address, IREE_SV("base_residue_proof"), context->arena,
      &base_residue_proof));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      context->builder,
      " model=%.*s proof=%.*s lane-address=%.*s active-lanes=%.*s "
      "base-residue=%.*s",
      (int)model_key.size, model_key.data, (int)proof.size, proof.data,
      (int)lane_address_proof.size, lane_address_proof.data,
      (int)active_lane_proof.size, active_lane_proof.data,
      (int)base_residue_proof.size, base_residue_proof.data));

  iree_string_view_t classification = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_bank_service_check_lookup_string(
      bank_service, IREE_SV("classification"), context->arena,
      &classification));
  if (!iree_string_view_is_empty(classification)) {
    uint64_t required_rounds = 0;
    uint64_t uncontended_rounds = 0;
    uint64_t extra_rounds = 0;
    uint64_t maximum_multiplicity = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_bank_service_check_lookup_uint64(
        bank_service, IREE_SV("required_rounds"), &required_rounds));
    IREE_RETURN_IF_ERROR(loom_amdgpu_bank_service_check_lookup_uint64(
        bank_service, IREE_SV("uncontended_rounds"), &uncontended_rounds));
    IREE_RETURN_IF_ERROR(loom_amdgpu_bank_service_check_lookup_uint64(
        bank_service, IREE_SV("extra_rounds"), &extra_rounds));
    IREE_RETURN_IF_ERROR(loom_amdgpu_bank_service_check_lookup_uint64(
        bank_service, IREE_SV("maximum_request_multiplicity"),
        &maximum_multiplicity));
    return iree_string_builder_append_format(
        context->builder,
        " classification=%.*s required-rounds=%" PRIu64
        " uncontended-rounds=%" PRIu64 " extra-rounds=%" PRIu64
        " maximum-multiplicity=%" PRIu64 "\n",
        (int)classification.size, classification.data, required_rounds,
        uncontended_rounds, extra_rounds, maximum_multiplicity);
  }
  return iree_string_builder_append_format(
      context->builder, " unknown-reason=%.*s\n", (int)unknown_reason.size,
      unknown_reason.data);
}

static iree_status_t loom_amdgpu_bank_service_check_write_output(
    const loom_check_emit_provider_request_t* request,
    const loomc_result_t* result) {
  const loomc_artifact_t* artifact = NULL;
  IREE_RETURN_IF_ERROR(loom_check_compile_select_artifact(
      result, LOOMC_ARTIFACT_KIND_REPORT,
      IREE_SV(LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON), &artifact));

  loomc_byte_span_t bytes = {0};
  const loomc_allocator_t allocator =
      loomc_allocator_from_iree(iree_arena_allocator(request->case_arena));
  const bool cloned =
      !loomc_byte_sequence_try_get_contiguous_span(artifact->contents, &bytes);
  iree_status_t status = iree_ok_status();
  if (cloned) {
    status = iree_status_from_loomc(
        loomc_byte_sequence_clone(artifact->contents, allocator, &bytes));
  }
  iree_string_view_t source_low = iree_string_view_empty();
  iree_string_view_t memory_rows = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    const iree_string_view_t report_json = iree_make_string_view(
        (const char*)bytes.data, (iree_host_size_t)bytes.data_length);
    status = iree_json_lookup_object_value(report_json, IREE_SV("source_low"),
                                           &source_low);
  }
  if (iree_status_is_ok(status)) {
    status = iree_json_lookup_object_value(source_low, IREE_SV("memory_rows"),
                                           &memory_rows);
  }
  if (iree_status_is_ok(status)) {
    loom_amdgpu_bank_service_check_row_context_t context = {
        .arena = request->case_arena,
        .builder = &request->result->actual_output,
    };
    status = iree_json_enumerate_array(
        memory_rows, loom_amdgpu_bank_service_check_append_row, &context);
  }
  if (cloned) {
    loomc_allocator_free(allocator, (void*)bytes.data);
  }
  return status;
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

  const loomc_compile_report_options_t report_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
      .structure_size = sizeof(report_options),
      .mode = LOOMC_COMPILE_REPORT_MODE_DETAILS,
      .format = LOOMC_COMPILE_REPORT_FORMAT_JSON,
  };
  const loom_check_compile_source_low_options_t compile_options = {
      .pipeline_kind = LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW,
      .report = &report_options,
  };
  bool compiled = false;
  loomc_result_t* result = NULL;
  iree_status_t status = loom_check_compile_source_low(
      request, &compile_options, &compiled, &result);
  if (iree_status_is_ok(status) && compiled) {
    status = loom_amdgpu_bank_service_check_write_output(request, result);
  }
  loomc_result_release(result);
  return status;
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
