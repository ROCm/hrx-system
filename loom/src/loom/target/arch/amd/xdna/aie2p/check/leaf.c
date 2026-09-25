// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/check/leaf.h"

#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_compile.h"
#include "loom/target/arch/amd/xdna/aie2p/machine/machine.h"
#include "loom/target/reporting/report.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/low_emit.h"

typedef enum loom_aie2p_leaf_check_report_e {
  LOOM_AIE2P_LEAF_CHECK_REPORT_NONE = 0,
  LOOM_AIE2P_LEAF_CHECK_REPORT_EMISSION,
  LOOM_AIE2P_LEAF_CHECK_REPORT_CODE,
} loom_aie2p_leaf_check_report_t;

static bool loom_aie2p_leaf_check_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("aie2p-leaf"));
}

static iree_status_t loom_aie2p_leaf_check_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  iree_string_view_t remaining = iree_string_view_trim(request->target_options);
  iree_string_view_t symbol;
  iree_string_view_split(remaining, ' ', &symbol, &remaining);
  if (!iree_string_view_starts_with(symbol, IREE_SV("@")) || symbol.size == 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "aie2p-leaf requires a Low function symbol");
  }
  loom_check_low_emit_fixed_value_spec_t
      fixed_specs[LOOM_CHECK_LOW_EMIT_MAX_ALLOCATION_FIXED_VALUES];
  iree_host_size_t fixed_spec_count = 0;
  iree_string_view_t registers = iree_string_view_empty();
  loom_aie2p_leaf_check_report_t report_kind =
      LOOM_AIE2P_LEAF_CHECK_REPORT_NONE;
  while (!iree_string_view_is_empty(iree_string_view_trim(remaining))) {
    iree_string_view_t token;
    iree_string_view_split(iree_string_view_trim(remaining), ' ', &token,
                           &remaining);
    iree_string_view_t name, value;
    iree_string_view_split(token, '=', &name, &value);
    if (iree_string_view_equal(name, IREE_SV("fixed"))) {
      IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_fixed_value_spec(
          value, IREE_SV("aie2p-leaf"), fixed_specs,
          IREE_ARRAYSIZE(fixed_specs), &fixed_spec_count));
    } else if (iree_string_view_equal(name, IREE_SV("registers"))) {
      registers = value;
    } else if (iree_string_view_equal(name, IREE_SV("report")) &&
               iree_string_view_equal(value, IREE_SV("emission"))) {
      report_kind = LOOM_AIE2P_LEAF_CHECK_REPORT_EMISSION;
    } else if (iree_string_view_equal(name, IREE_SV("report")) &&
               iree_string_view_equal(value, IREE_SV("code"))) {
      report_kind = LOOM_AIE2P_LEAF_CHECK_REPORT_CODE;
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown aie2p-leaf option '%.*s'",
                              (int)name.size, name.data);
    }
  }
  loom_check_diagnostic_emitter_capture_t diagnostic_capture = {
      .diagnostic_collector = request->diagnostic_collector,
      .module = request->module,
      .source_resolver = request->source_resolver,
      .emitter = LOOM_EMITTER_PASS,
  };
  const iree_diagnostic_emitter_t emitter = {
      .fn = loom_check_diagnostic_emitter_capture_emit,
      .user_data = &diagnostic_capture,
  };
  loom_op_t* function = NULL;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_find_low_function_def(
      request->module, iree_string_view_substr(symbol, 1, IREE_HOST_SIZE_MAX),
      request->test_case, request->filename, request->diagnostic_collector,
      emitter, &function));
  if (!function) {
    return iree_ok_status();
  }
  loom_aie2p_leaf_compile_options_t options = {
      .descriptor_registry = &request->low_registry->registry,
      .diagnostic_emitter = emitter,
  };
  bool resolved = false;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_resolve_fixed_value_specs(
      request->module, function, fixed_specs, fixed_spec_count, emitter,
      &options.allocation_fixed_values, &options.allocation_fixed_value_count,
      &resolved, request->case_arena));
  if (!resolved) {
    return iree_ok_status();
  }
  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  options.compile_report =
      report_kind == LOOM_AIE2P_LEAF_CHECK_REPORT_EMISSION ? &report : NULL;
  loom_aie2p_leaf_contribution_t contribution = {0};
  bool compiled = false;
  iree_status_t status =
      loom_aie2p_leaf_compile(request->module, function, &options,
                              request->case_arena, &compiled, &contribution);
  if (iree_status_is_ok(status) && compiled &&
      report_kind == LOOM_AIE2P_LEAF_CHECK_REPORT_EMISSION) {
    status = iree_string_builder_append_format(
        &request->result->actual_output,
        "issue cycles: %" PRIu64 "\ncode bytes: %" PRIu64
        "\ncoissued bundles: %" PRIu64 "\ncoissued components: %" PRIu64 "\n",
        report.emitted_instruction_count, report.emitted_code_byte_count,
        report.emission_breakdown.coissued_instruction_count,
        report.emission_breakdown.coissued_component_count);
  }
  loom_target_compile_report_deinitialize(&report);
  IREE_RETURN_IF_ERROR(status);
  if (!compiled) {
    return iree_ok_status();
  }
  if (report_kind == LOOM_AIE2P_LEAF_CHECK_REPORT_CODE) {
    const loom_native_object_symbol_t* entry =
        &contribution.object
             .symbols[contribution.realization.entry_symbol_index];
    const iree_const_byte_span_t code =
        contribution.object.sections[entry->section_contribution_index]
            .contents;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        &request->result->actual_output, "code:"));
    for (iree_host_size_t i = 0; i < code.data_length; ++i) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          &request->result->actual_output, " %02x", (unsigned)code.data[i]));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
        &request->result->actual_output, "\n"));
  }
  while (!iree_string_view_is_empty(registers)) {
    iree_string_view_t name;
    iree_string_view_split(registers, ',', &name, &registers);
    const loom_aie2p_physical_register_id_t physical_register =
        loom_aie2p_machine_find_physical_register(name);
    if (physical_register == LOOM_AIE2P_PHYSICAL_REGISTER_ID_INVALID) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown AIE2P physical register '%.*s'",
                              (int)name.size, name.data);
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        &request->result->actual_output, "%.*s: %s\n", (int)name.size,
        name.data,
        loom_aie2p_leaf_may_write_register(&contribution.realization,
                                           physical_register)
            ? "written"
            : "preserved"));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_leaf_check_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "aie2p-leaf\n");
}

const loom_check_emit_provider_t loom_aie2p_leaf_check_emit_provider = {
    .name = IREE_SVL("aie2p-leaf"),
    .match = loom_aie2p_leaf_check_matches,
    .execute = loom_aie2p_leaf_check_execute,
    .append_names = loom_aie2p_leaf_check_append_names,
};
