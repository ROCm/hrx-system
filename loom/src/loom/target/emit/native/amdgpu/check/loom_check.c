// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/check/loom_check.h"

#include <inttypes.h>

#include "iree/base/alignment.h"
#include "loom/codegen/low/allocation_json.h"
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
#include "loom/target/emit/native/amdgpu/assembly.h"
#include "loom/target/emit/native/amdgpu/encoding.h"
#include "loom/target/emit/native/amdgpu/hal_kernel_library.h"
#include "loom/target/emit/native/amdgpu/spill_lowering.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/low_emit.h"

typedef enum loom_amdgpu_loom_check_wait_mode_e {
  LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_AUTO = 0,
  LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_NONE = 1,
} loom_amdgpu_loom_check_wait_mode_t;

enum loom_amdgpu_loom_check_option_flag_bits_e {
  LOOM_AMDGPU_LOOM_CHECK_OPTION_FLAG_SCHEDULE_STRATEGY = 1u << 0,
  LOOM_AMDGPU_LOOM_CHECK_OPTION_FLAG_SCHEDULE_DIAGNOSTICS = 1u << 1,
  LOOM_AMDGPU_LOOM_CHECK_OPTION_FLAG_WAIT_MODE = 1u << 2,
  LOOM_AMDGPU_LOOM_CHECK_OPTION_FLAG_ALLOCATION_DIAGNOSTICS = 1u << 3,
};
typedef uint8_t loom_amdgpu_loom_check_option_flags_t;

typedef struct loom_amdgpu_loom_check_emit_options_t {
  // Module-local target-low function symbol selected by the RUN line.
  iree_string_view_t function_symbol_name;
  // Candidate selection strategy used by low frame.
  loom_low_schedule_strategy_t schedule_strategy;
  // Low scheduler diagnostic feedback requested by the RUN line.
  loom_low_schedule_diagnostic_flags_t schedule_diagnostic_flags;
  // Low allocator diagnostic feedback requested by the RUN line.
  loom_low_allocation_diagnostic_flags_t allocation_diagnostic_flags;
  // Wait-packet materialization mode for the emitted fragment.
  loom_amdgpu_loom_check_wait_mode_t wait_mode;
  // Parsed loom_amdgpu_loom_check_option_flag_bits_e bits.
  loom_amdgpu_loom_check_option_flags_t option_flags;
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
                                IREE_SV("amdgpu-kernel-assembly")) ||
         iree_string_view_equal(target_name, IREE_SV("amdgpu-kernel-asm")) ||
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
  if (iree_string_view_equal(name, IREE_SV("strategy"))) {
    if (iree_any_bit_set(
            options->option_flags,
            LOOM_AMDGPU_LOOM_CHECK_OPTION_FLAG_SCHEDULE_STRATEGY)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate AMDGPU assembly option 'strategy'");
    }
    IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_schedule_strategy(
        value, IREE_SV("AMDGPU assembly"), &options->schedule_strategy));
    options->option_flags |=
        LOOM_AMDGPU_LOOM_CHECK_OPTION_FLAG_SCHEDULE_STRATEGY;
    *out_matched = true;
    return iree_ok_status();
  }
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
  if (iree_string_view_equal(name, IREE_SV("waits"))) {
    if (iree_any_bit_set(options->option_flags,
                         LOOM_AMDGPU_LOOM_CHECK_OPTION_FLAG_WAIT_MODE)) {
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
    options->option_flags |= LOOM_AMDGPU_LOOM_CHECK_OPTION_FLAG_WAIT_MODE;
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

static iree_status_t loom_amdgpu_loom_check_encode_native(
    const loom_low_emission_frame_t* frame,
    const loom_amdgpu_loom_check_emit_options_t* options,
    iree_arena_allocator_t* arena, iree_const_byte_span_t* out_text) {
  if (options->wait_mode == LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_NONE) {
    return loom_amdgpu_encode_instruction_stream(
        &frame->schedule, &frame->allocation, out_text, arena);
  }

  loom_amdgpu_packet_plan_t packet_plan = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_plan_build(
      &frame->schedule, &frame->allocation, arena, &packet_plan));
  const loom_amdgpu_encode_instruction_stream_options_t encoding_options = {
      .packet_plan = &packet_plan,
  };
  return loom_amdgpu_encode_instruction_stream_with_options(
      &frame->schedule, &frame->allocation, &encoding_options, out_text, arena);
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
      iree_string_view_equal(target_name, IREE_SV("amdgpu-allocation-json")) ||
      iree_string_view_equal(target_name,
                             IREE_SV("amdgpu-wait-counter-plan-json")) ||
      iree_string_view_equal(target_name, IREE_SV("amdgpu-wait-state-plan")) ||
      iree_string_view_equal(target_name,
                             IREE_SV("amdgpu-wait-state-plan-json"))) {
    return true;
  }
  return options->wait_mode == LOOM_AMDGPU_LOOM_CHECK_WAIT_MODE_AUTO;
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
      module, low_function_op, target.descriptor_set, emitter, &result, arena));
  out_result->error_count = result.error_count;
  out_result->required_register_value_ids = result.required_register_value_ids;
  out_result->required_register_value_count =
      result.required_register_value_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_loom_check_build_schedule_models(
    const loom_check_emit_provider_request_t* request,
    loom_symbol_fact_table_t* symbol_facts,
    iree_string_view_t function_symbol_name,
    const loom_target_residency_model_t** out_residency_model,
    loom_low_schedule_pair_affinity_list_t* out_affinities,
    loom_low_schedule_structural_state_read_list_t* out_state_reads) {
  *out_residency_model = NULL;
  *out_affinities = loom_low_schedule_pair_affinity_list_empty();
  *out_state_reads = loom_low_schedule_structural_state_read_list_empty();
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
      request->module, symbol_facts, low_function,
      /*function_target_facts=*/NULL, &request->low_registry->registry, emitter,
      &target));
  if (target.descriptor_set == NULL) {
    return iree_ok_status();
  }
  *out_residency_model = loom_amdgpu_occupancy_residency_model(&target);
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_build_schedule_pair_affinities(
      &target, request->case_arena, out_affinities));
  *out_state_reads = loom_amdgpu_descriptor_structural_state_reads();
  return iree_ok_status();
}

static bool loom_amdgpu_loom_check_is_kernel_assembly_target(
    iree_string_view_t target_name) {
  return iree_string_view_equal(target_name,
                                IREE_SV("amdgpu-kernel-assembly")) ||
         iree_string_view_equal(target_name, IREE_SV("amdgpu-kernel-asm"));
}

static iree_status_t loom_amdgpu_loom_check_emit_hal_kernel_assembly(
    const loom_check_emit_provider_request_t* request) {
  const loom_amdgpu_hal_kernel_library_options_t options = {
      .diagnostic_sink =
          {
              .fn = loom_check_diagnostic_collector_sink,
              .user_data = request->diagnostic_collector,
          },
      .source_resolver = request->source_resolver,
      .max_errors = 20,
      .capture_target_listing = true,
  };
  bool emitted = false;
  loom_amdgpu_hal_kernel_library_t library = {0};
  iree_status_t status = loom_amdgpu_emit_hal_kernel_library(
      request->module, &options, request->host_allocator, &emitted, &library);
  if (iree_status_is_ok(status) && emitted) {
    if (!iree_string_view_equal(library.target_listing_format,
                                IREE_SV("amdgpu-assembly")) ||
        library.target_listing_data == NULL) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "AMDGPU HAL kernel library omitted its requested assembly listing");
    } else {
      status = iree_string_builder_append_string(
          &request->result->actual_output,
          iree_make_string_view(library.target_listing_data,
                                library.target_listing_data_length));
    }
  } else if (iree_status_is_ok(status) &&
             (request->diagnostic_collector == NULL ||
              request->diagnostic_collector->count == 0)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU HAL kernel library did not select an emittable kernel");
  }
  loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                              request->host_allocator);
  return status;
}

static iree_status_t loom_amdgpu_loom_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  if (loom_amdgpu_loom_check_is_kernel_assembly_target(request->target_name)) {
    return loom_amdgpu_loom_check_emit_hal_kernel_assembly(request);
  }
  loom_amdgpu_loom_check_emit_options_t options;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_loom_check_parse_emit_options(request, &options));
  loom_low_emission_frame_t frame = {0};
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, request->case_arena);
  loom_amdgpu_loom_check_spill_lowering_context_t spill_lowering_context = {
      .descriptor_registry = &request->low_registry->registry,
      .symbol_facts = &symbol_facts,
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
  };
  loom_low_storage_lease_provider_t storage_lease_provider = {0};
  loom_amdgpu_storage_lease_provider(&storage_lease_provider);
  const loom_low_storage_lease_provider_t* selected_storage_lease_provider =
      loom_amdgpu_loom_check_needs_storage_leases(request->target_name,
                                                  &options)
          ? &storage_lease_provider
          : NULL;
  const loom_target_residency_model_t* residency_model = NULL;
  loom_low_schedule_pair_affinity_list_t schedule_pair_affinities =
      loom_low_schedule_pair_affinity_list_empty();
  loom_low_schedule_structural_state_read_list_t schedule_state_reads =
      loom_low_schedule_structural_state_read_list_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_loom_check_build_schedule_models(
      request, &symbol_facts, options.function_symbol_name, &residency_model,
      &schedule_pair_affinities, &schedule_state_reads));
  if (request->diagnostic_collector != NULL &&
      request->diagnostic_collector->count != 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_check_low_emit_packetize_function(
      request, options.function_symbol_name, options.schedule_strategy,
      options.schedule_diagnostic_flags, options.allocation_diagnostic_flags,
      options.allocation_budgets, options.allocation_budget_count,
      options.allocation_fixed_value_specs,
      options.allocation_fixed_value_spec_count, residency_model,
      schedule_pair_affinities, schedule_state_reads,
      selected_storage_lease_provider, &spill_free_options, &frame));
  if (request->diagnostic_collector != NULL &&
      request->diagnostic_collector->count != 0) {
    return iree_ok_status();
  }
  if (frame.schedule.error_count != 0 || frame.allocation.error_count != 0) {
    return iree_ok_status();
  }
  if (iree_string_view_equal(request->target_name, IREE_SV("amdgpu-native"))) {
    iree_const_byte_span_t text = iree_const_byte_span_empty();
    return loom_amdgpu_loom_check_encode_native(&frame, &options,
                                                request->case_arena, &text);
  }
  if (iree_string_view_equal(request->target_name,
                             IREE_SV("amdgpu-native-words"))) {
    iree_const_byte_span_t text = iree_const_byte_span_empty();
    IREE_RETURN_IF_ERROR(loom_amdgpu_loom_check_encode_native(
        &frame, &options, request->case_arena, &text));
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
      &frame, &options, &request->result->actual_output, request->case_arena);
}

static iree_status_t loom_amdgpu_loom_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(
      builder,
      "amdgpu-assembly, amdgpu-asm, amdgpu-kernel-assembly, "
      "amdgpu-kernel-asm, amdgpu-allocation-json, "
      "amdgpu-packet-json, amdgpu-vopd-plan-json, "
      "amdgpu-wait-counter-plan-json, amdgpu-wait-state-plan, "
      "amdgpu-wait-state-plan-json, amdgpu-native");
}

const loom_check_emit_provider_t loom_amdgpu_native_loom_check_emit_provider = {
    .name = IREE_SVL("amdgpu-native"),
    .match = loom_amdgpu_loom_check_emit_provider_matches,
    .execute = loom_amdgpu_loom_check_emit_provider_execute,
    .append_names = loom_amdgpu_loom_check_emit_provider_append_names,
};
