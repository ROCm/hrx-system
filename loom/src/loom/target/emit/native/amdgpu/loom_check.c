// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/loom_check.h"

#include <inttypes.h>

#include "iree/base/alignment.h"
#include "loom/codegen/low/allocation_json.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/packet_json.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/planning/descriptor_semantics.h"
#include "loom/target/arch/amdgpu/planning/occupancy.h"
#include "loom/target/arch/amdgpu/planning/packet_plan.h"
#include "loom/target/arch/amdgpu/planning/storage_lease.h"
#include "loom/target/arch/amdgpu/planning/vopd_plan.h"
#include "loom/target/arch/amdgpu/planning/wait_states_format.h"
#include "loom/target/emit/native/amdgpu/assembly.h"
#include "loom/target/emit/native/amdgpu/encoding.h"
#include "loom/target/emit/native/amdgpu/spill_lowering.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/test/low/native_emit.h"

enum loom_amdgpu_loom_check_option_flag_bits_e {
  LOOM_AMDGPU_LOOM_CHECK_OPTION_FLAG_SCHEDULE_DIAGNOSTICS = 1u << 0,
  LOOM_AMDGPU_LOOM_CHECK_OPTION_FLAG_ALLOCATION_DIAGNOSTICS = 1u << 1,
};
typedef uint8_t loom_amdgpu_loom_check_option_flags_t;

typedef struct loom_amdgpu_loom_check_emit_options_t {
  // Module-local target-low function symbol selected by the RUN line.
  iree_string_view_t function_symbol_name;
  // Low scheduler diagnostic feedback requested by the RUN line.
  loom_low_schedule_diagnostic_flags_t schedule_diagnostic_flags;
  // Low allocator diagnostic feedback requested by the RUN line.
  loom_low_allocation_diagnostic_flags_t allocation_diagnostic_flags;
  // Parsed loom_amdgpu_loom_check_option_flag_bits_e bits.
  loom_amdgpu_loom_check_option_flags_t option_flags;
} loom_amdgpu_loom_check_emit_options_t;

typedef struct loom_amdgpu_loom_check_spill_lowering_context_t {
  // Target-low descriptor registry visible to this check runner.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Case-scoped facts reused while spill lowering preserves target bindings.
  loom_symbol_fact_table_t* symbol_facts;
} loom_amdgpu_loom_check_spill_lowering_context_t;

static bool loom_amdgpu_loom_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("amdgpu-assembly")) ||
         iree_string_view_equal(target_name, IREE_SV("amdgpu-asm")) ||
         iree_string_view_equal(target_name,
                                IREE_SV("amdgpu-allocation-json")) ||
         iree_string_view_equal(target_name, IREE_SV("amdgpu-packet-json")) ||
         iree_string_view_equal(target_name,
                                IREE_SV("amdgpu-vopd-plan-json")) ||
         iree_string_view_equal(target_name,
                                IREE_SV("amdgpu-wait-counter-plan-json")) ||
         iree_string_view_equal(target_name,
                                IREE_SV("amdgpu-wait-state-plan")) ||
         iree_string_view_equal(target_name,
                                IREE_SV("amdgpu-wait-state-plan-json")) ||
         iree_string_view_equal(target_name, IREE_SV("amdgpu-native")) ||
         iree_string_view_equal(target_name, IREE_SV("amdgpu-native-words"));
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
  if (iree_string_view_equal(name, IREE_SV("diagnostics"))) {
    if (iree_any_bit_set(
            options->option_flags,
            LOOM_AMDGPU_LOOM_CHECK_OPTION_FLAG_SCHEDULE_DIAGNOSTICS)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate AMDGPU assembly option 'diagnostics'");
    }
    IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_schedule_diagnostics(
        value, IREE_SV("AMDGPU assembly"),
        &options->schedule_diagnostic_flags));
    options->option_flags |=
        LOOM_AMDGPU_LOOM_CHECK_OPTION_FLAG_SCHEDULE_DIAGNOSTICS;
    *out_matched = true;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("allocation-diagnostics"))) {
    if (iree_any_bit_set(
            options->option_flags,
            LOOM_AMDGPU_LOOM_CHECK_OPTION_FLAG_ALLOCATION_DIAGNOSTICS)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "duplicate AMDGPU assembly option 'allocation-diagnostics'");
    }
    IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_allocation_diagnostics(
        value, IREE_SV("AMDGPU assembly"),
        &options->allocation_diagnostic_flags));
    options->option_flags |=
        LOOM_AMDGPU_LOOM_CHECK_OPTION_FLAG_ALLOCATION_DIAGNOSTICS;
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
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown AMDGPU assembly option '%.*s'",
                          (int)token.size, token.data);
}

static iree_status_t loom_amdgpu_loom_check_parse_emit_options(
    const loom_check_emit_provider_request_t* request,
    loom_amdgpu_loom_check_emit_options_t* out_options) {
  *out_options = (loom_amdgpu_loom_check_emit_options_t){0};

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
    const loom_low_emission_frame_t* frame, iree_string_builder_t* builder,
    iree_arena_allocator_t* arena, iree_diagnostic_emitter_t emitter) {
  loom_amdgpu_packet_plan_t packet_plan = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_plan_build(
      &frame->schedule, &frame->allocation, /*options=*/NULL, arena,
      /*out_vopd_rejections=*/NULL, &packet_plan));
  const loom_amdgpu_encode_instruction_stream_options_t encoding_options = {
      .packet_plan = &packet_plan,
  };
  loom_amdgpu_encoded_instruction_stream_t stream = {0};
  bool accepted = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encode_instruction_stream(
      &frame->schedule, &frame->allocation, &encoding_options, arena, emitter,
      &accepted, &stream));
  if (!accepted) {
    return iree_ok_status();
  }
  const loom_amdgpu_assembly_fragment_options_t assembly_options = {
      .packet_plan = &packet_plan,
      .instruction_layout = &stream.layout,
  };
  return loom_amdgpu_emit_assembly_fragment(
      &frame->schedule, &frame->allocation, &assembly_options, builder, arena,
      emitter, &accepted);
}

static iree_status_t loom_amdgpu_loom_check_encode_native(
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* arena,
    iree_diagnostic_emitter_t emitter, bool* out_accepted,
    iree_const_byte_span_t* out_text) {
  loom_amdgpu_packet_plan_t packet_plan = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_plan_build(
      &frame->schedule, &frame->allocation, /*options=*/NULL, arena,
      /*out_vopd_rejections=*/NULL, &packet_plan));
  const loom_amdgpu_encode_instruction_stream_options_t encoding_options = {
      .packet_plan = &packet_plan,
  };
  loom_amdgpu_encoded_instruction_stream_t stream = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_encode_instruction_stream(
      &frame->schedule, &frame->allocation, &encoding_options, arena, emitter,
      out_accepted, &stream));
  *out_text = stream.text;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_loom_check_format_native_words(
    iree_const_byte_span_t text, iree_string_builder_t* builder) {
  IREE_ASSERT((text.data_length % sizeof(uint32_t)) == 0);
  for (iree_host_size_t offset = 0; offset < text.data_length;
       offset += sizeof(uint32_t)) {
    const uint32_t word = iree_unaligned_load_le_u32(text.data + offset);
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(builder, "0x%08" PRIX32 "\n", word));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_loom_check_emit_wait_state_plan_json(
    const loom_low_emission_frame_t* frame, iree_string_builder_t* builder,
    iree_arena_allocator_t* arena) {
  loom_amdgpu_packet_plan_t packet_plan = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_plan_build(
      &frame->schedule, &frame->allocation, /*options=*/NULL, arena,
      /*out_vopd_rejections=*/NULL, &packet_plan));
  return loom_amdgpu_wait_state_plan_format_json(&packet_plan.wait_states,
                                                 builder);
}

static iree_status_t loom_amdgpu_loom_check_emit_wait_state_plan(
    const loom_low_emission_frame_t* frame, iree_string_builder_t* builder,
    iree_arena_allocator_t* arena) {
  loom_amdgpu_packet_plan_t packet_plan = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_plan_build(
      &frame->schedule, &frame->allocation, /*options=*/NULL, arena,
      /*out_vopd_rejections=*/NULL, &packet_plan));
  return loom_amdgpu_wait_state_plan_format_text(&packet_plan.wait_states,
                                                 builder);
}

static iree_status_t loom_amdgpu_loom_check_emit_wait_counter_plan_json(
    const loom_low_emission_frame_t* frame, iree_string_builder_t* builder,
    iree_arena_allocator_t* arena) {
  loom_amdgpu_packet_plan_t packet_plan = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_plan_build(
      &frame->schedule, &frame->allocation, /*options=*/NULL, arena,
      /*out_vopd_rejections=*/NULL, &packet_plan));
  return loom_amdgpu_wait_plan_format_json(&packet_plan.wait_plan, builder);
}

static iree_status_t loom_amdgpu_loom_check_emit_vopd_plan_json(
    const loom_low_emission_frame_t* frame, iree_string_builder_t* builder,
    iree_arena_allocator_t* arena) {
  loom_amdgpu_packet_plan_t packet_plan = {0};
  loom_amdgpu_vopd_rejection_table_t rejections = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_plan_build(
      &frame->schedule, &frame->allocation, /*options=*/NULL, arena,
      &rejections, &packet_plan));
  return loom_amdgpu_vopd_plan_format_json(&packet_plan.vopd_plan, &rejections,
                                           builder);
}

static iree_status_t loom_amdgpu_loom_check_lower_spill_traffic(
    void* user_data, loom_module_t* module, loom_op_t* low_function_op,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    loom_low_emission_frame_lower_spill_traffic_result_t* out_result) {
  const loom_amdgpu_loom_check_spill_lowering_context_t* context =
      (const loom_amdgpu_loom_check_spill_lowering_context_t*)user_data;
  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, context->symbol_facts, low_function_op,
      /*function_target_facts=*/NULL, context->descriptor_registry, emitter,
      &target));
  if (target.descriptor_set == NULL) {
    return iree_ok_status();
  }
  loom_amdgpu_spill_lowering_result_t result = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_lower_spill_traffic(
      module, low_function_op, target.descriptor_set,
      (loom_amdgpu_storage_layout_segment_sizes_t){0}, emitter, arena,
      &result));
  out_result->error_count = result.error_count;
  out_result->lowered_traffic = result.lowered_traffic;
  out_result->required_register_value_ids = result.required_register_value_ids;
  out_result->required_register_value_count =
      result.required_register_value_count;
  out_result->expanded_execution_value_ids =
      result.expanded_execution_value_ids;
  out_result->expanded_execution_value_count =
      result.expanded_execution_value_count;
  out_result->per_user_placement_value_ids =
      result.per_user_placement_value_ids;
  out_result->per_user_placement_value_count =
      result.per_user_placement_value_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_loom_check_build_schedule_models(
    const loom_check_emit_provider_request_t* request,
    const loom_check_test_native_module_t* native_module,
    loom_symbol_fact_table_t* symbol_facts,
    iree_string_view_t function_symbol_name, iree_diagnostic_emitter_t emitter,
    loom_low_schedule_pair_affinity_list_t* out_affinities,
    loom_low_schedule_structural_state_read_list_t* out_state_reads,
    loom_low_schedule_structural_model_t* out_storage_address_model) {
  *out_affinities = loom_low_schedule_pair_affinity_list_empty();
  *out_state_reads = loom_low_schedule_structural_state_read_list_empty();
  loom_op_t* low_function = NULL;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_find_low_function_def(
      native_module->module, function_symbol_name, request->test_case,
      request->filename, request->diagnostic_collector, emitter,
      &low_function));
  if (!low_function) {
    return iree_ok_status();
  }
  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      native_module->module, symbol_facts, low_function,
      /*function_target_facts=*/NULL, &native_module->low_registry.registry,
      emitter, &target));
  if (target.descriptor_set == NULL) {
    return iree_ok_status();
  }
  if (target.target_facts == NULL) {
    return loom_low_diagnostic_emit_missing_target(native_module->module,
                                                   low_function, emitter);
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_build_schedule_pair_affinities(
      &target, request->case_arena, out_affinities));
  *out_state_reads = loom_amdgpu_descriptor_structural_state_reads();
  *out_storage_address_model =
      loom_amdgpu_descriptor_storage_address_model(target.descriptor_set);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_loom_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request,
    const loom_check_test_native_module_t* native_module) {
  (void)provider;
  loom_check_test_native_diagnostic_emitter_capture_t diagnostic_capture = {
      .context = native_module->diagnostic_context,
      .module = native_module->module,
      .source_resolver = native_module->source_resolver,
      .emitter = LOOM_EMITTER_PASS,
  };
  iree_diagnostic_emitter_t emitter = {0};
  if (request->diagnostic_collector != NULL) {
    emitter = (iree_diagnostic_emitter_t){
        .fn = loom_check_test_native_diagnostic_emitter_capture_emit,
        .user_data = &diagnostic_capture,
    };
  }
  loom_amdgpu_loom_check_emit_options_t options;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_loom_check_parse_emit_options(request, &options));
  loom_low_emission_frame_t frame = {0};
  bool frame_accepted = false;
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, request->case_arena);
  loom_amdgpu_loom_check_spill_lowering_context_t spill_lowering_context = {
      .descriptor_registry = &native_module->low_registry.registry,
      .symbol_facts = &symbol_facts,
  };
  loom_amdgpu_storage_layout_segment_sizes_t segment_prefix = {0};
  const loom_low_emission_frame_spill_free_options_t spill_free_options = {
      .materialization_options =
          {
              .has_supported_storage_spaces = true,
              .supported_storage_spaces = LOOM_LOW_STORAGE_SPACE_SET_SCRATCH |
                                          LOOM_LOW_STORAGE_SPACE_SET_PRIVATE,
              .plan_traffic = loom_amdgpu_plan_spill_materialization_traffic,
              .plan_traffic_user_data = &segment_prefix,
          },
      .lower_spill_traffic = loom_amdgpu_loom_check_lower_spill_traffic,
      .lower_spill_traffic_user_data = &spill_lowering_context,
  };
  loom_low_storage_lease_provider_t storage_lease_provider = {0};
  loom_amdgpu_storage_lease_provider(&storage_lease_provider);
  loom_low_schedule_pair_affinity_list_t schedule_pair_affinities =
      loom_low_schedule_pair_affinity_list_empty();
  loom_low_schedule_structural_state_read_list_t schedule_state_reads =
      loom_low_schedule_structural_state_read_list_empty();
  loom_low_schedule_structural_model_t storage_address_model = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_loom_check_build_schedule_models(
      request, native_module, &symbol_facts, options.function_symbol_name,
      emitter, &schedule_pair_affinities, &schedule_state_reads,
      &storage_address_model));
  if (request->diagnostic_collector != NULL &&
      loom_check_diagnostic_collector_has_error(
          request->diagnostic_collector)) {
    return iree_ok_status();
  }
  const loom_low_emission_frame_options_t frame_options = {
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
      .schedule_flags =
          LOOM_LOW_SCHEDULE_FLAG_RETAIN_VALUE_PRODUCER_NODES |
          LOOM_LOW_SCHEDULE_FLAG_RETAIN_ACYCLIC_MEMORY_COMPLETIONS,
      .allocation_flags =
          LOOM_LOW_ALLOCATION_FLAG_RETAIN_COALESCED_INCOMING_INDEX |
          LOOM_LOW_ALLOCATION_FLAG_RETAIN_STORAGE_RELEASE_ACTION_INDEX,
      .schedule_diagnostic_flags = options.schedule_diagnostic_flags,
      .allocation_diagnostic_flags = options.allocation_diagnostic_flags,
      .residency_query = loom_amdgpu_occupancy_residency_view,
      .schedule_pair_affinities = schedule_pair_affinities,
      .schedule_structural_state_reads = schedule_state_reads,
      .schedule_structural_models = {.values = &storage_address_model,
                                     .count = 1},
      .storage_lease_provider = &storage_lease_provider,
  };
  IREE_RETURN_IF_ERROR(loom_check_low_emit_packetize_function(
      request, native_module, options.function_symbol_name, &frame_options,
      &spill_free_options, &frame, &frame_accepted));
  if (request->diagnostic_collector != NULL &&
      loom_check_diagnostic_collector_has_error(
          request->diagnostic_collector)) {
    return iree_ok_status();
  }
  if (!frame_accepted) {
    return iree_ok_status();
  }
  if (iree_string_view_equal(request->target_name, IREE_SV("amdgpu-native"))) {
    iree_const_byte_span_t text = iree_const_byte_span_empty();
    bool accepted = false;
    return loom_amdgpu_loom_check_encode_native(&frame, request->case_arena,
                                                emitter, &accepted, &text);
  }
  if (iree_string_view_equal(request->target_name,
                             IREE_SV("amdgpu-native-words"))) {
    iree_const_byte_span_t text = iree_const_byte_span_empty();
    bool accepted = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_loom_check_encode_native(
        &frame, request->case_arena, emitter, &accepted, &text));
    if (!accepted) {
      return iree_ok_status();
    }
    return loom_amdgpu_loom_check_format_native_words(
        text, &request->result->actual_output);
  }
  if (iree_string_view_equal(request->target_name,
                             IREE_SV("amdgpu-allocation-json"))) {
    return loom_low_allocation_format_json(&frame.allocation,
                                           &request->result->actual_output);
  }
  if (iree_string_view_equal(request->target_name,
                             IREE_SV("amdgpu-packet-json"))) {
    return loom_low_packet_format_json(&frame.schedule, &frame.allocation,
                                       &request->result->actual_output);
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
      &frame, &request->result->actual_output, request->case_arena, emitter);
}

static iree_status_t loom_amdgpu_loom_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(
      builder,
      "amdgpu-assembly, amdgpu-asm, amdgpu-allocation-json, "
      "amdgpu-packet-json, amdgpu-vopd-plan-json, "
      "amdgpu-wait-counter-plan-json, amdgpu-wait-state-plan, "
      "amdgpu-wait-state-plan-json, amdgpu-native, amdgpu-native-words");
}

const loom_check_test_native_emit_provider_t
    loom_amdgpu_native_loom_check_emit_provider = {
        .base =
            {
                .name = IREE_SVL("amdgpu-native"),
                .match = loom_amdgpu_loom_check_emit_provider_matches,
                .execute = loom_check_test_native_emit_provider_execute,
                .append_names =
                    loom_amdgpu_loom_check_emit_provider_append_names,
            },
        .execute_native = loom_amdgpu_loom_check_emit_provider_execute,
};
