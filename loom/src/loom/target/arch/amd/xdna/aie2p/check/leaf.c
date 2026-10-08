// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/check/leaf.h"

#include "loom/codegen/low/diagnostics.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
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

static iree_status_t loom_aie2p_leaf_check_append_value_reference(
    const loom_module_t* module, loom_value_id_t value_id,
    iree_string_builder_t* builder) {
  const iree_string_view_t value_name =
      loom_low_diagnostic_value_name(module, value_id);
  return iree_string_builder_append_format(
      builder, "%%%.*s", (int)value_name.size, value_name.data);
}

static iree_string_view_t loom_aie2p_leaf_check_operation_name(
    const loom_module_t* module, const loom_op_t* op) {
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (loom_low_op_isa(op)) {
    descriptor_ordinal = loom_low_op_descriptor(op);
  } else if (loom_low_const_isa(op)) {
    descriptor_ordinal = loom_low_const_descriptor(op);
  }
  if (descriptor_ordinal != LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    const loom_low_descriptor_set_t* descriptor_set =
        loom_aie2p_core_descriptor_set();
    const loom_low_descriptor_t* descriptor =
        loom_low_descriptor_set_descriptor_at(descriptor_set,
                                              descriptor_ordinal);
    if (descriptor != NULL) {
      return loom_low_descriptor_set_string(descriptor_set,
                                            descriptor->mnemonic_string_ref);
    }
  }
  return loom_op_name(module, op);
}

static iree_status_t loom_aie2p_leaf_check_append_value_uses(
    const loom_module_t* module, const loom_op_t* function,
    iree_string_view_t selected_value_names, iree_string_builder_t* builder) {
  while (!iree_string_view_is_empty(selected_value_names)) {
    iree_string_view_t selected_value_name;
    iree_string_view_split(selected_value_names, ',', &selected_value_name,
                           &selected_value_names);
    selected_value_name = iree_string_view_trim(selected_value_name);
    if (iree_string_view_is_empty(selected_value_name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "aie2p-leaf option 'uses' contains an empty value selector");
    }

    loom_check_low_emit_value_resolution_t resolution;
    loom_check_low_emit_resolve_function_value(
        module, function, selected_value_name, &resolution);
    if (resolution.kind == LOOM_CHECK_LOW_EMIT_VALUE_RESOLUTION_NOT_FOUND) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "aie2p-leaf use selector '%%%.*s' does not name a value in the "
          "selected low function",
          (int)selected_value_name.size, selected_value_name.data);
    }
    if (resolution.kind == LOOM_CHECK_LOW_EMIT_VALUE_RESOLUTION_AMBIGUOUS) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "aie2p-leaf use selector '%%%.*s' is ambiguous in the selected low "
          "function",
          (int)selected_value_name.size, selected_value_name.data);
    }

    const loom_value_t* value = loom_module_value(module, resolution.value_id);
    if (value->use_count == 0) {
      IREE_RETURN_IF_ERROR(loom_aie2p_leaf_check_append_value_reference(
          module, resolution.value_id, builder));
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, " -> <no uses>\n"));
      continue;
    }
    const loom_use_t* uses = loom_value_uses(value);
    for (uint32_t use_index = 0; use_index < value->use_count; ++use_index) {
      const loom_use_t use = uses[use_index];
      const loom_op_t* user_op = loom_use_user_op(use);
      IREE_RETURN_IF_ERROR(loom_aie2p_leaf_check_append_value_reference(
          module, resolution.value_id, builder));
      const iree_string_view_t user_op_name =
          loom_aie2p_leaf_check_operation_name(module, user_op);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " -> %.*s operand=%u (", (int)user_op_name.size,
          user_op_name.data, (unsigned)loom_use_operand_index(use)));
      const loom_value_id_t* operands = loom_op_const_operands(user_op);
      for (uint16_t operand_index = 0; operand_index < user_op->operand_count;
           ++operand_index) {
        if (operand_index > 0) {
          IREE_RETURN_IF_ERROR(
              iree_string_builder_append_cstring(builder, ", "));
        }
        IREE_RETURN_IF_ERROR(loom_aie2p_leaf_check_append_value_reference(
            module, operands[operand_index], builder));
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ")\n"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_leaf_check_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request,
    const loom_check_emit_native_module_t* native_module) {
  (void)provider;
  iree_string_view_t remaining = iree_string_view_trim(request->target_options);
  iree_string_view_t symbol;
  iree_string_view_split(remaining, ' ', &symbol, &remaining);
  if (!iree_string_view_starts_with(symbol, IREE_SV("@")) || symbol.size == 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "aie2p-leaf requires a Low function symbol");
  }
  loom_check_low_emit_fixed_value_spec_list_t fixed_specs;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_fixed_value_spec_list_initialize(
      remaining, request->case_arena, &fixed_specs));
  iree_string_view_t registers = iree_string_view_empty();
  iree_string_view_t selected_value_names = iree_string_view_empty();
  loom_low_allocation_diagnostic_flags_t allocation_diagnostic_flags = 0;
  bool has_allocation_diagnostics_option = false;
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
          value, IREE_SV("aie2p-leaf"), &fixed_specs));
    } else if (iree_string_view_equal(name, IREE_SV("registers"))) {
      registers = value;
    } else if (iree_string_view_equal(name, IREE_SV("uses"))) {
      if (!iree_string_view_is_empty(selected_value_names)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate aie2p-leaf option 'uses'");
      }
      if (iree_string_view_is_empty(value)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "aie2p-leaf option 'uses' requires at least "
                                "one value selector");
      }
      selected_value_names = value;
    } else if (iree_string_view_equal(name, IREE_SV("diagnostics"))) {
      if (has_allocation_diagnostics_option) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate aie2p-leaf option 'diagnostics'");
      }
      IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_allocation_diagnostics(
          value, IREE_SV("aie2p-leaf"), &allocation_diagnostic_flags));
      has_allocation_diagnostics_option = true;
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
      .module = native_module->module,
      .source_resolver = native_module->source_resolver,
      .emitter = LOOM_EMITTER_PASS,
  };
  const iree_diagnostic_emitter_t emitter = {
      .fn = loom_check_diagnostic_emitter_capture_emit,
      .user_data = &diagnostic_capture,
  };
  loom_op_t* function = NULL;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_find_low_function_def(
      native_module->module,
      iree_string_view_substr(symbol, 1, IREE_HOST_SIZE_MAX),
      request->test_case, request->filename, request->diagnostic_collector,
      emitter, &function));
  if (!function) {
    return iree_ok_status();
  }
  // This emitter accepts only core functions, even when the test runner links
  // other representation contracts for parsing and verification.
  const loom_low_descriptor_set_provider_t descriptor_set_providers[] = {
      loom_aie2p_core_descriptor_set,
  };
  const loom_low_descriptor_registry_t descriptor_registry = {
      .descriptor_set_providers = descriptor_set_providers,
      .descriptor_set_provider_count = IREE_ARRAYSIZE(descriptor_set_providers),
  };
  loom_aie2p_leaf_compile_options_t options = {
      .descriptor_registry = &descriptor_registry,
      .allocation_diagnostic_flags = allocation_diagnostic_flags,
      .diagnostic_emitter = emitter,
  };
  bool resolved = false;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_resolve_fixed_value_specs(
      native_module->module, function, fixed_specs.specs, fixed_specs.count,
      emitter, &options.allocation_fixed_values,
      &options.allocation_fixed_value_count, &resolved, request->case_arena));
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
      loom_aie2p_leaf_compile(native_module->module, function, &options,
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
  IREE_RETURN_IF_ERROR(loom_aie2p_leaf_check_append_value_uses(
      native_module->module, function, selected_value_names,
      &request->result->actual_output));
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
    .execute_native = loom_aie2p_leaf_check_execute,
    .append_names = loom_aie2p_leaf_check_append_names,
};
