// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/check/loom_check.h"

#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/planning/address_state.h"
#include "loom/target/arch/amdgpu/planning/packet_plan.h"
#include "loom/target/arch/amdgpu/planning/storage_lease.h"
#include "loom/target/arch/amdgpu/planning/vopd_plan.h"
#include "loom/target/emit/native/amdgpu/assembly.h"
#include "loom/target/emit/native/amdgpu/encoding.h"
#include "loom/target/emit/native/amdgpu/spill_lowering.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/low_emit.h"

typedef enum loom_amdgpu_loom_check_wait_mode_e {
  LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_AUTO = 0,
  LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_NONE = 1,
} loom_amdgpu_loom_check_wait_mode_t;

typedef struct loom_amdgpu_loom_check_emit_options_t {
  // Module-local target-low function symbol selected by the RUN line.
  iree_string_view_t function_symbol_name;
  // Candidate selection strategy used by low frame.
  loom_low_schedule_strategy_t schedule_strategy;
  // True once a strategy option has been parsed.
  bool has_schedule_strategy_option;
  // Wait-packet materialization mode for the emitted fragment.
  loom_amdgpu_loom_check_wait_mode_t wait_mode;
  // True once a waits option has been parsed.
  bool has_wait_mode_option;
  // Low allocation budget overrides parsed from target options.
  loom_low_allocation_budget_t
      allocation_budgets[LOOM_CHECK_LOW_EMIT_MAX_ALLOCATION_BUDGETS];
  // Number of entries in |allocation_budgets|.
  iree_host_size_t allocation_budget_count;
  // Fixed low allocation requests parsed from target options.
  loom_check_low_emit_fixed_value_spec_t allocation_fixed_value_specs
      [LOOM_CHECK_LOW_EMIT_MAX_ALLOCATION_FIXED_VALUES];
  // Number of entries in |allocation_fixed_value_specs|.
  iree_host_size_t allocation_fixed_value_spec_count;
} loom_amdgpu_loom_check_emit_options_t;

typedef struct loom_amdgpu_loom_check_spill_lowering_context_t {
  // Target-low descriptor registry visible to this check runner.
  const loom_low_descriptor_registry_t* descriptor_registry;
} loom_amdgpu_loom_check_spill_lowering_context_t;

static bool loom_amdgpu_loom_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("amdgpu-assembly")) ||
         iree_string_view_equal(target_name, IREE_SV("amdgpu-asm")) ||
         iree_string_view_equal(target_name,
                                IREE_SV("amdgpu-vopd-plan-json")) ||
         iree_string_view_equal(target_name,
                                IREE_SV("amdgpu-wait-counter-plan-json")) ||
         iree_string_view_equal(target_name,
                                IREE_SV("amdgpu-wait-state-plan")) ||
         iree_string_view_equal(target_name,
                                IREE_SV("amdgpu-wait-state-plan-json")) ||
         iree_string_view_equal(target_name, IREE_SV("amdgpu-native"));
}

static iree_status_t loom_amdgpu_loom_check_parse_key_value_option(
    iree_string_view_t token, loom_amdgpu_loom_check_emit_options_t* options,
    bool* out_matched) {
  *out_matched = false;
  iree_string_view_t name = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_string_view_split(token, '=', &name, &value);
  name = iree_string_view_trim(name);
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(name, IREE_SV("strategy"))) {
    if (options->has_schedule_strategy_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate AMDGPU assembly option 'strategy'");
    }
    IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_schedule_strategy(
        value, IREE_SV("AMDGPU assembly"), &options->schedule_strategy));
    options->has_schedule_strategy_option = true;
    *out_matched = true;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("waits"))) {
    if (options->has_wait_mode_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate AMDGPU assembly option 'waits'");
    }
    if (iree_string_view_equal(value, IREE_SV("auto"))) {
      options->wait_mode = LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_AUTO;
    } else if (iree_string_view_equal(value, IREE_SV("none"))) {
      options->wait_mode = LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_NONE;
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU assembly option 'waits' expected 'auto' or 'none', got "
          "'%.*s'",
          (int)value.size, value.data);
    }
    options->has_wait_mode_option = true;
    *out_matched = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_loom_check_parse_option(
    iree_string_view_t token, loom_amdgpu_loom_check_emit_options_t* options) {
  bool matched = false;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_loom_check_parse_key_value_option(token, options, &matched));
  if (matched) {
    return iree_ok_status();
  }
  return loom_check_low_emit_parse_allocation_option(
      token, IREE_SV("AMDGPU assembly"), options->allocation_budgets,
      IREE_ARRAYSIZE(options->allocation_budgets),
      &options->allocation_budget_count, options->allocation_fixed_value_specs,
      IREE_ARRAYSIZE(options->allocation_fixed_value_specs),
      &options->allocation_fixed_value_spec_count);
}

static iree_status_t loom_amdgpu_loom_check_parse_emit_options(
    const loom_check_emit_provider_request_t* request,
    loom_amdgpu_loom_check_emit_options_t* out_options) {
  *out_options = (loom_amdgpu_loom_check_emit_options_t){
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING,
      .wait_mode = LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_AUTO,
  };

  iree_string_view_t symbol_name = iree_string_view_empty();
  iree_string_view_t option_text = iree_string_view_empty();
  iree_string_view_split(request->target_options, ' ', &symbol_name,
                         &option_text);
  symbol_name = iree_string_view_trim(symbol_name);
  option_text = iree_string_view_trim(option_text);
  if (!iree_string_view_starts_with(symbol_name, IREE_SV("@"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU assembly requires a low function symbol "
                            "name");
  }
  out_options->function_symbol_name =
      iree_string_view_substr(symbol_name, 1, IREE_HOST_SIZE_MAX);
  if (iree_string_view_is_empty(out_options->function_symbol_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU assembly low function symbol name is "
                            "required");
  }

  while (!iree_string_view_is_empty(option_text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(option_text, ' ', &token, &remaining);
    token = iree_string_view_trim(token);
    if (!iree_string_view_is_empty(token)) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_loom_check_parse_option(token, out_options));
    }
    option_text = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_loom_check_emit_assembly(
    const loom_low_emission_frame_t* frame,
    const loom_amdgpu_loom_check_emit_options_t* options,
    iree_string_builder_t* builder, iree_arena_allocator_t* arena) {
  if (options->wait_mode == LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_NONE) {
    return loom_amdgpu_emit_assembly_fragment(
        &frame->schedule, &frame->allocation, builder, arena);
  }

  loom_amdgpu_packet_plan_t packet_plan = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_plan_build(
      &frame->schedule, &frame->allocation, arena, &packet_plan));
  const loom_amdgpu_assembly_fragment_options_t assembly_options = {
      .packet_plan = &packet_plan,
  };
  return loom_amdgpu_emit_assembly_fragment_with_options(
      &frame->schedule, &frame->allocation, &assembly_options, builder, arena);
}

static iree_status_t loom_amdgpu_loom_check_emit_native(
    const loom_low_emission_frame_t* frame,
    const loom_amdgpu_loom_check_emit_options_t* options,
    iree_arena_allocator_t* arena) {
  iree_const_byte_span_t text = iree_const_byte_span_empty();
  if (options->wait_mode == LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_NONE) {
    return loom_amdgpu_encode_instruction_stream(
        &frame->schedule, &frame->allocation, &text, arena);
  }

  loom_amdgpu_packet_plan_t packet_plan = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_plan_build(
      &frame->schedule, &frame->allocation, arena, &packet_plan));
  const loom_amdgpu_encode_instruction_stream_options_t encoding_options = {
      .packet_plan = &packet_plan,
  };
  return loom_amdgpu_encode_instruction_stream_with_options(
      &frame->schedule, &frame->allocation, &encoding_options, &text, arena);
}

static iree_status_t loom_amdgpu_loom_check_emit_wait_state_plan_json(
    const loom_low_emission_frame_t* frame, iree_string_builder_t* builder,
    iree_arena_allocator_t* arena) {
  loom_amdgpu_packet_plan_t packet_plan = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_plan_build(
      &frame->schedule, &frame->allocation, arena, &packet_plan));
  return loom_amdgpu_wait_state_plan_format_json(&packet_plan.wait_states,
                                                 builder);
}

static iree_status_t loom_amdgpu_loom_check_emit_wait_state_plan(
    const loom_low_emission_frame_t* frame, iree_string_builder_t* builder,
    iree_arena_allocator_t* arena) {
  loom_amdgpu_packet_plan_t packet_plan = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_plan_build(
      &frame->schedule, &frame->allocation, arena, &packet_plan));
  return loom_amdgpu_wait_state_plan_format_text(&packet_plan.wait_states,
                                                 builder);
}

static iree_status_t loom_amdgpu_loom_check_emit_wait_counter_plan_json(
    const loom_low_emission_frame_t* frame, iree_string_builder_t* builder,
    iree_arena_allocator_t* arena) {
  loom_amdgpu_packet_plan_t packet_plan = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_plan_build(
      &frame->schedule, &frame->allocation, arena, &packet_plan));
  return loom_amdgpu_wait_plan_format_json(&packet_plan.wait_plan, builder);
}

static iree_status_t loom_amdgpu_loom_check_emit_vopd_plan_json(
    const loom_low_emission_frame_t* frame, iree_string_builder_t* builder,
    iree_arena_allocator_t* arena) {
  loom_amdgpu_packet_plan_t packet_plan = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_plan_build(
      &frame->schedule, &frame->allocation, arena, &packet_plan));
  return loom_amdgpu_vopd_plan_format_json(&packet_plan.vopd_plan, builder);
}

static bool loom_amdgpu_loom_check_needs_storage_leases(
    iree_string_view_t target_name,
    const loom_amdgpu_loom_check_emit_options_t* options) {
  if (iree_string_view_equal(target_name, IREE_SV("amdgpu-vopd-plan-json")) ||
      iree_string_view_equal(target_name,
                             IREE_SV("amdgpu-wait-counter-plan-json")) ||
      iree_string_view_equal(target_name, IREE_SV("amdgpu-wait-state-plan")) ||
      iree_string_view_equal(target_name,
                             IREE_SV("amdgpu-wait-state-plan-json"))) {
    return true;
  }
  return options->wait_mode == LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_AUTO;
}

static iree_status_t loom_amdgpu_loom_check_materialize_address_state(
    void* user_data, loom_module_t* module, loom_op_t* low_function_op,
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    loom_low_emission_frame_materialize_address_state_result_t* out_result) {
  (void)user_data;
  return loom_amdgpu_materialize_address_state(module, low_function_op, frame,
                                               arena, out_result);
}

static iree_status_t loom_amdgpu_loom_check_lower_spill_traffic(
    void* user_data, loom_module_t* module, loom_op_t* low_function_op,
    iree_arena_allocator_t* arena) {
  const loom_amdgpu_loom_check_spill_lowering_context_t* context =
      (const loom_amdgpu_loom_check_spill_lowering_context_t*)user_data;
  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, low_function_op, context->descriptor_registry,
      (loom_target_selection_t){0}, (iree_diagnostic_emitter_t){0}, &target));
  if (target.descriptor_set == NULL) {
    return iree_ok_status();
  }
  return loom_amdgpu_lower_spill_traffic(module, low_function_op,
                                         target.descriptor_set, arena);
}

static iree_status_t loom_amdgpu_loom_check_build_schedule_pair_affinities(
    const loom_check_emit_provider_request_t* request,
    iree_string_view_t function_symbol_name,
    loom_low_schedule_pair_affinity_list_t* out_affinities) {
  *out_affinities = loom_low_schedule_pair_affinity_list_empty();
  loom_check_diagnostic_emitter_capture_t diagnostic_capture = {
      .diagnostic_collector = request->diagnostic_collector,
      .module = request->module,
      .source_resolver = request->source_resolver,
      .emitter = LOOM_EMITTER_PASS,
  };
  iree_diagnostic_emitter_t emitter = {0};
  if (request->diagnostic_collector != NULL) {
    emitter = (iree_diagnostic_emitter_t){
        .fn = loom_check_diagnostic_emitter_capture_emit,
        .user_data = &diagnostic_capture,
    };
  }
  loom_op_t* low_function = NULL;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_find_low_function_def(
      request->module, function_symbol_name, request->test_case,
      request->filename, request->diagnostic_collector, emitter,
      &low_function));
  if (!low_function) {
    return iree_ok_status();
  }
  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      request->module, low_function, &request->low_registry->registry,
      loom_target_selection_empty(), emitter, &target));
  if (target.descriptor_set == NULL) {
    return iree_ok_status();
  }
  return loom_amdgpu_vopd_build_schedule_pair_affinities(
      target.descriptor_set, request->case_arena, out_affinities);
}

static iree_status_t loom_amdgpu_loom_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  loom_amdgpu_loom_check_emit_options_t options;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_loom_check_parse_emit_options(request, &options));
  loom_low_emission_frame_t frame = {0};
  loom_amdgpu_loom_check_spill_lowering_context_t spill_lowering_context = {
      .descriptor_registry = &request->low_registry->registry,
  };
  const loom_low_emission_frame_spill_free_options_t spill_free_options = {
      .materialization_options =
          {
              .has_supported_storage_spaces = true,
              .supported_storage_spaces = LOOM_LOW_STORAGE_SPACE_SET_SCRATCH |
                                          LOOM_LOW_STORAGE_SPACE_SET_PRIVATE,
          },
      .lower_spill_traffic = loom_amdgpu_loom_check_lower_spill_traffic,
      .lower_spill_traffic_user_data = &spill_lowering_context,
      .materialize_address_state =
          loom_amdgpu_loom_check_materialize_address_state,
  };
  loom_low_storage_lease_provider_t storage_lease_provider = {0};
  loom_amdgpu_storage_lease_provider(&storage_lease_provider);
  const loom_low_storage_lease_provider_t* selected_storage_lease_provider =
      loom_amdgpu_loom_check_needs_storage_leases(request->target_name,
                                                  &options)
          ? &storage_lease_provider
          : NULL;
  loom_low_schedule_pair_affinity_list_t schedule_pair_affinities =
      loom_low_schedule_pair_affinity_list_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_loom_check_build_schedule_pair_affinities(
      request, options.function_symbol_name, &schedule_pair_affinities));
  IREE_RETURN_IF_ERROR(loom_check_low_emit_packetize_function(
      request, options.function_symbol_name, options.schedule_strategy,
      options.allocation_budgets, options.allocation_budget_count,
      options.allocation_fixed_value_specs,
      options.allocation_fixed_value_spec_count, schedule_pair_affinities,
      selected_storage_lease_provider, &spill_free_options, &frame));
  if (request->diagnostic_collector != NULL &&
      request->diagnostic_collector->count != 0) {
    return iree_ok_status();
  }
  if (iree_string_view_equal(request->target_name, IREE_SV("amdgpu-native"))) {
    return loom_amdgpu_loom_check_emit_native(&frame, &options,
                                              request->case_arena);
  }
  if (iree_string_view_equal(request->target_name,
                             IREE_SV("amdgpu-wait-state-plan-json"))) {
    return loom_amdgpu_loom_check_emit_wait_state_plan_json(
        &frame, &request->result->actual_output, request->case_arena);
  }
  if (iree_string_view_equal(request->target_name,
                             IREE_SV("amdgpu-wait-state-plan"))) {
    return loom_amdgpu_loom_check_emit_wait_state_plan(
        &frame, &request->result->actual_output, request->case_arena);
  }
  if (iree_string_view_equal(request->target_name,
                             IREE_SV("amdgpu-wait-counter-plan-json"))) {
    return loom_amdgpu_loom_check_emit_wait_counter_plan_json(
        &frame, &request->result->actual_output, request->case_arena);
  }
  if (iree_string_view_equal(request->target_name,
                             IREE_SV("amdgpu-vopd-plan-json"))) {
    return loom_amdgpu_loom_check_emit_vopd_plan_json(
        &frame, &request->result->actual_output, request->case_arena);
  }
  return loom_amdgpu_loom_check_emit_assembly(
      &frame, &options, &request->result->actual_output, request->case_arena);
}

static iree_status_t loom_amdgpu_loom_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(
      builder,
      "amdgpu-assembly, amdgpu-asm, amdgpu-vopd-plan-json, "
      "amdgpu-wait-counter-plan-json, amdgpu-wait-state-plan, "
      "amdgpu-wait-state-plan-json, amdgpu-native");
}

const loom_check_emit_provider_t loom_amdgpu_native_loom_check_emit_provider = {
    .name = IREE_SVL("amdgpu-native"),
    .match = loom_amdgpu_loom_check_emit_provider_matches,
    .execute = loom_amdgpu_loom_check_emit_provider_execute,
    .append_names = loom_amdgpu_loom_check_emit_provider_append_names,
};
