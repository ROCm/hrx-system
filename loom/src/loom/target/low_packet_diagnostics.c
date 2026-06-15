// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/low_packet_diagnostics.h"

#include "loom/codegen/low/diagnostics.h"
#include "loom/error/error_catalog.h"

struct loom_target_low_packet_diagnostic_context_t {
  // Emission frame being diagnosed.
  const loom_low_emission_frame_t* frame;
  // Caller-owned diagnostic options.
  const loom_target_low_packet_diagnostics_options_t* options;
  // Result object receiving counters.
  loom_target_low_packet_diagnostics_result_t* result;
};

static iree_status_t loom_target_low_packet_diagnostics_verify_options(
    const loom_low_emission_frame_t* frame,
    const loom_target_low_packet_diagnostics_options_t* options,
    loom_target_low_packet_diagnostics_result_t* out_result) {
  const loom_target_low_packet_diagnostic_flags_t unknown_flags =
      options->diagnostic_flags & ~LOOM_TARGET_LOW_PACKET_DIAGNOSTIC_ALL;
  if (unknown_flags != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown target-low packet diagnostic flags 0x%08x",
                            unknown_flags);
  }
  return loom_low_packet_validate_tables(&frame->schedule, &frame->allocation);
}

static iree_status_t loom_target_low_packet_diagnostics_emit(
    loom_target_low_packet_diagnostic_context_t* context,
    const loom_low_packet_view_t* packet, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count) {
  switch (error->severity) {
    case LOOM_DIAGNOSTIC_ERROR:
      ++context->result->error_count;
      break;
    case LOOM_DIAGNOSTIC_WARNING:
      ++context->result->warning_count;
      break;
    case LOOM_DIAGNOSTIC_REMARK:
      ++context->result->remark_count;
      break;
    default:
      break;
  }
  loom_diagnostic_emission_t emission = {
      .op = packet->node->op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(context->options->emitter, &emission);
}

static iree_status_t loom_target_low_packet_diagnostics_packet_key(
    loom_target_low_packet_diagnostic_context_t* context,
    const loom_low_packet_view_t* packet, iree_string_view_t* out_key) {
  if (packet->descriptor == NULL) {
    *out_key = IREE_SV("<structural>");
    return iree_ok_status();
  }
  *out_key = loom_low_descriptor_set_string(
      context->frame->schedule.target.descriptor_set,
      packet->descriptor->key_string_offset);
  return iree_ok_status();
}

iree_status_t loom_target_low_packet_diagnostics_record_packet(
    loom_target_low_packet_diagnostic_context_t* context,
    const loom_low_packet_view_t* packet, iree_string_view_t packet_category,
    iree_string_view_t decision) {
  iree_string_view_t packet_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_target_low_packet_diagnostics_packet_key(
      context, packet, &packet_key));
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  const uint32_t operand_count = descriptor ? descriptor->operand_count : 0;
  const uint32_t result_count = descriptor ? descriptor->result_count : 0;
  const loom_low_schedule_table_t* schedule = &context->frame->schedule;
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&schedule->target)),
      loom_param_string(loom_low_diagnostic_export_name(&schedule->target)),
      loom_param_string(loom_low_diagnostic_config_key(&schedule->target)),
      loom_param_string(loom_low_diagnostic_function_name(
          schedule->module, schedule->function_op)),
      loom_param_string(packet_category),
      loom_param_string(packet_key),
      loom_param_string(decision),
      loom_param_u32(operand_count),
      loom_param_u32(result_count),
  };
  return loom_target_low_packet_diagnostics_emit(
      context, packet, LOOM_ERR_BACKEND_018, params, IREE_ARRAYSIZE(params));
}

const loom_module_t* loom_target_low_packet_diagnostics_module(
    const loom_target_low_packet_diagnostic_context_t* context) {
  return context->frame->schedule.module;
}

const loom_low_schedule_table_t* loom_target_low_packet_diagnostics_schedule(
    const loom_target_low_packet_diagnostic_context_t* context) {
  return &context->frame->schedule;
}

const loom_low_allocation_table_t*
loom_target_low_packet_diagnostics_allocation(
    const loom_target_low_packet_diagnostic_context_t* context) {
  return &context->frame->allocation;
}

loom_target_low_packet_diagnostic_flags_t
loom_target_low_packet_diagnostics_diagnostic_flags(
    const loom_target_low_packet_diagnostic_context_t* context) {
  return context->options->diagnostic_flags;
}

iree_status_t loom_target_low_packet_diagnostics_emit_function(
    const loom_low_emission_frame_t* frame,
    const loom_target_low_packet_diagnostics_options_t* options,
    loom_target_low_packet_diagnostics_result_t* out_result) {
  IREE_RETURN_IF_ERROR(loom_target_low_packet_diagnostics_verify_options(
      frame, options, out_result));
  *out_result = (loom_target_low_packet_diagnostics_result_t){0};
  if (options->diagnostic_flags == 0 ||
      loom_target_low_packet_diagnostic_provider_list_is_empty(
          options->provider_list)) {
    return iree_ok_status();
  }

  loom_target_low_packet_diagnostic_context_t context = {
      .frame = frame,
      .options = options,
      .result = out_result,
  };
  const iree_host_size_t packet_count = loom_low_packet_count(&frame->schedule);
  for (iree_host_size_t packet_index = 0; packet_index < packet_count;
       ++packet_index) {
    loom_low_packet_view_t packet = {0};
    IREE_RETURN_IF_ERROR(loom_low_packet_view_at(
        &frame->schedule, &frame->allocation, packet_index, &packet));
    for (iree_host_size_t provider_index = 0;
         provider_index < options->provider_list.count; ++provider_index) {
      const loom_target_low_packet_diagnostic_provider_t* provider =
          options->provider_list.values[provider_index];
      bool handled = false;
      IREE_RETURN_IF_ERROR(
          provider->try_diagnose_packet(provider, &context, &packet, &handled));
      if (handled) {
        break;
      }
    }
  }
  return iree_ok_status();
}
