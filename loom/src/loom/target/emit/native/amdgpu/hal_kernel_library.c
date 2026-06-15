// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/hal_kernel_library.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/allocation_materialization.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/verify.h"
#include "loom/ir/module.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/low/kernel.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/hal/kernel_abi.h"
#include "loom/target/arch/amdgpu/ops/ops.h"
#include "loom/target/arch/amdgpu/ops/target.h"
#include "loom/target/arch/amdgpu/planning/address_state.h"
#include "loom/target/arch/amdgpu/planning/occupancy.h"
#include "loom/target/arch/amdgpu/planning/packet_plan.h"
#include "loom/target/arch/amdgpu/planning/storage_lease.h"
#include "loom/target/arch/amdgpu/planning/vopd_plan.h"
#include "loom/target/arch/amdgpu/provider.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/compile_report_low.h"
#include "loom/target/emit/native/amdgpu/kernel_assembly.h"
#include "loom/target/emit/native/amdgpu/kernel_hsaco.h"
#include "loom/target/emit/native/amdgpu/preflight.h"
#include "loom/target/emit/native/amdgpu/spill_lowering.h"
#include "loom/target/entry_selection.h"
#include "loom/target/function_contract.h"
#include "loom/target/launch.h"
#include "loom/target/provider.h"

#define LOOM_AMDGPU_HAL_KERNEL_LIBRARY_DEFAULT_MAX_ERRORS 20u

static bool loom_amdgpu_hal_kernel_library_bundle_is_compatible(
    void* user_data, const loom_target_entry_t* entry) {
  if (!loom_low_kernel_def_isa(entry->func.op)) {
    return false;
  }
  const loom_target_bundle_t* bundle = &entry->bundle_storage.bundle;
  return bundle && bundle->snapshot && bundle->export_plan &&
         loom_amdgpu_target_isa(entry->target_op) &&
         bundle->snapshot->codegen_format ==
             LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE &&
         bundle->snapshot->artifact_format == LOOM_TARGET_ARTIFACT_FORMAT_ELF &&
         bundle->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL;
}

typedef struct loom_amdgpu_hal_kernel_library_kernel_plan_t {
  // Target-resolved function entry used to build this kernel.
  loom_target_entry_t* entry;
  // Selected prepared low.kernel.def op for frame.
  loom_op_t* low_function_op;
  // ABI layout derived from prepared target-low IR.
  loom_amdgpu_hal_kernel_abi_layout_t abi_layout;
  // Fixed allocator values derived from the HAL ABI live-ins.
  const loom_low_allocation_fixed_value_t* fixed_values;
  // Number of entries in |fixed_values|.
  iree_host_size_t fixed_value_count;
} loom_amdgpu_hal_kernel_library_kernel_plan_t;

static iree_status_t loom_amdgpu_hal_kernel_library_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref,
    iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "AMDGPU symbol reference is invalid");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id == LOOM_STRING_ID_INVALID ||
      symbol->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "AMDGPU symbol has no module string");
  }
  *out_name = module->strings.entries[symbol->name_id];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_library_emit(
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    const loom_op_t* op, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count) {
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(loom_target_entry_emitter(diagnostic_emitter),
                              &emission);
}

static const loom_op_t* loom_amdgpu_hal_kernel_library_target_record_op(
    const loom_target_entry_t* entry) {
  return entry->target_op ? entry->target_op : entry->func.op;
}

static iree_status_t loom_amdgpu_hal_kernel_library_emit_unknown_processor(
    const loom_target_entry_t* entry,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_string_view_t processor_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(processor_name),
  };
  return loom_amdgpu_hal_kernel_library_emit(
      diagnostic_emitter,
      loom_amdgpu_hal_kernel_library_target_record_op(entry),
      LOOM_ERR_AMDGPU_003, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_hal_kernel_library_emit_no_descriptor_set(
    const loom_target_entry_t* entry,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_string_view_t processor_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(processor_name),
  };
  return loom_amdgpu_hal_kernel_library_emit(
      diagnostic_emitter,
      loom_amdgpu_hal_kernel_library_target_record_op(entry),
      LOOM_ERR_AMDGPU_004, params, IREE_ARRAYSIZE(params));
}

static iree_status_t
loom_amdgpu_hal_kernel_library_emit_descriptor_set_mismatch(
    const loom_target_entry_t* entry,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    const loom_amdgpu_processor_info_t* processor,
    iree_string_view_t target_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(processor->name),
      loom_param_string(processor->descriptor_set.key),
      loom_param_string(target_name),
      loom_param_string(entry->bundle_storage.config.contract_set_key),
  };
  return loom_amdgpu_hal_kernel_library_emit(
      diagnostic_emitter,
      loom_amdgpu_hal_kernel_library_target_record_op(entry),
      LOOM_ERR_AMDGPU_005, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_amdgpu_hal_kernel_library_apply_processor(
    loom_module_t* module, loom_target_entry_t* entry,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_string_view_t processor_name) {
  if (iree_string_view_is_empty(processor_name)) {
    return iree_ok_status();
  }
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_find_processor(processor_name);
  if (processor == NULL) {
    return loom_amdgpu_hal_kernel_library_emit_unknown_processor(
        entry, diagnostic_emitter, processor_name);
  }
  if (processor->descriptor_set.ordinal ==
          LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE ||
      iree_string_view_is_empty(processor->descriptor_set.key)) {
    return loom_amdgpu_hal_kernel_library_emit_no_descriptor_set(
        entry, diagnostic_emitter, processor->name);
  }
  if (!iree_string_view_equal(processor->descriptor_set.key,
                              entry->bundle_storage.config.contract_set_key)) {
    return loom_amdgpu_hal_kernel_library_emit_descriptor_set_mismatch(
        entry, diagnostic_emitter, processor,
        entry->bundle_storage.bundle.name);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_target_record_set_processor(
      module, entry->target_op, processor));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_library_read_stream_contents(
    iree_io_stream_t* stream, iree_allocator_t allocator,
    iree_const_byte_span_t* out_contents) {
  *out_contents = iree_const_byte_span_empty();
  const iree_io_stream_pos_t stream_length = iree_io_stream_length(stream);
  if (stream_length < 0 || (uint64_t)stream_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO stream length is out of range");
  }
  uint8_t* data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, (iree_host_size_t)stream_length, (void**)&data));
  iree_status_t status =
      iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0);
  if (iree_status_is_ok(status)) {
    status = iree_io_stream_read(stream, (iree_host_size_t)stream_length, data,
                                 /*out_buffer_length=*/NULL);
  }
  if (iree_status_is_ok(status)) {
    *out_contents =
        iree_make_const_byte_span(data, (iree_host_size_t)stream_length);
  } else {
    iree_allocator_free(allocator, data);
  }
  return status;
}

static iree_status_t loom_amdgpu_hal_kernel_library_build_hsaco_contribution(
    const loom_low_emission_frame_t* frame,
    const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout,
    const loom_amdgpu_native_preflight_t* preflight,
    iree_string_builder_t* target_listing,
    loom_amdgpu_kernel_hsaco_contribution_t* out_contribution,
    iree_arena_allocator_t* table_arena) {
  loom_amdgpu_packet_plan_t packet_plan = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_plan_build(
      &frame->schedule, &frame->allocation, table_arena, &packet_plan));

  if (target_listing != NULL) {
    if (iree_string_builder_size(target_listing) != 0) {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(target_listing, "\n\n"));
    }
    const loom_amdgpu_kernel_assembly_options_t assembly_options = {
        .abi_layout = abi_layout,
        .packet_plan = &packet_plan,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_kernel_assembly_with_options(
        &frame->schedule, &frame->allocation, &assembly_options, target_listing,
        table_arena));
  }

  const loom_amdgpu_kernel_hsaco_options_t hsaco_options = {
      .abi_layout = abi_layout,
      .preflight = preflight,
      .packet_plan = &packet_plan,
  };
  return loom_amdgpu_build_kernel_hsaco_contribution(
      &frame->schedule, &frame->allocation, &hsaco_options, out_contribution,
      table_arena);
}

static iree_status_t loom_amdgpu_hal_kernel_library_write_hsaco(
    const loom_amdgpu_kernel_hsaco_contribution_t* contributions,
    iree_host_size_t contribution_count, iree_const_byte_span_t* out_hsaco,
    iree_arena_allocator_t* table_arena, iree_allocator_t allocator) {
  *out_hsaco = iree_const_byte_span_empty();

  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      32 * 1024, allocator, &stream));
  iree_status_t status = loom_amdgpu_write_kernel_hsaco_contributions(
      contributions, contribution_count, stream, table_arena);
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_hal_kernel_library_read_stream_contents(
        stream, allocator, out_hsaco);
  }
  iree_io_stream_release(stream);
  return status;
}

void loom_amdgpu_hal_kernel_library_deinitialize(
    loom_amdgpu_hal_kernel_library_t* library, iree_allocator_t allocator) {
  if (library == NULL) {
    return;
  }
  iree_allocator_free(allocator, (void*)library->executable_format.data);
  iree_allocator_free(allocator, library->hsaco_data);
  iree_allocator_free(allocator, library->target_listing_data);
  loom_target_artifact_manifest_json_t artifact_manifest_json = {
      .contents = library->artifact_manifest.contents,
  };
  loom_target_artifact_manifest_json_release(&artifact_manifest_json,
                                             allocator);
  *library = (loom_amdgpu_hal_kernel_library_t){0};
}

static iree_status_t loom_amdgpu_hal_kernel_library_set_contents(
    iree_string_view_t executable_format, iree_const_byte_span_t hsaco,
    iree_allocator_t allocator, loom_amdgpu_hal_kernel_library_t* out_library) {
  *out_library = (loom_amdgpu_hal_kernel_library_t){0};

  void* executable_format_data = NULL;
  iree_status_t status = iree_allocator_clone(
      allocator,
      iree_make_const_byte_span(executable_format.data, executable_format.size),
      &executable_format_data);
  if (iree_status_is_ok(status)) {
    out_library->executable_format =
        iree_make_string_view(executable_format_data, executable_format.size);
    out_library->hsaco_data = (uint8_t*)hsaco.data;
    out_library->hsaco_data_length = hsaco.data_length;
  }
  if (!iree_status_is_ok(status)) {
    out_library->hsaco_data = NULL;
    out_library->hsaco_data_length = 0;
    loom_amdgpu_hal_kernel_library_deinitialize(out_library, allocator);
  }
  return status;
}

static iree_status_t loom_amdgpu_hal_kernel_library_lookup_func_facts(
    const loom_module_t* module, loom_symbol_ref_t func_ref,
    iree_arena_allocator_t* table_arena,
    const loom_func_symbol_facts_t** out_func_facts) {
  *out_func_facts = NULL;
  loom_symbol_fact_table_t fact_table = {0};
  loom_symbol_fact_table_initialize(&fact_table, table_arena);
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      &fact_table, module, func_ref, &base_facts));
  *out_func_facts = loom_func_symbol_facts_cast(base_facts);
  if (*out_func_facts == NULL) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU HAL kernel-library entry must resolve to func symbol facts");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_library_apply_low_kernel_contract(
    const loom_module_t* module, const loom_op_t* low_function_op,
    loom_target_entry_t* entry,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_arena_allocator_t* table_arena, bool* out_valid) {
  *out_valid = true;
  loom_target_workgroup_size_t workgroup_size = {0};
  if (!loom_low_kernel_def_static_workgroup_size(low_function_op,
                                                 &workgroup_size)) {
    return iree_ok_status();
  }

  const loom_func_symbol_facts_t* func_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_library_lookup_func_facts(
      module, entry->func_ref, table_arena, &func_facts));

  return loom_target_function_contract_apply_hal_workgroup_size(
      func_facts, entry->bundle_storage.bundle.name, &workgroup_size,
      loom_target_entry_emitter(diagnostic_emitter), &entry->bundle_storage,
      out_valid);
}

static iree_status_t loom_amdgpu_hal_kernel_library_prepare_kernel_plan(
    loom_module_t* module, loom_target_entry_t* entry,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_arena_allocator_t* table_arena, loom_target_compile_report_t* report,
    loom_amdgpu_hal_kernel_library_kernel_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_hal_kernel_library_kernel_plan_t){
      .entry = entry,
      .low_function_op = entry->func.op,
  };

  bool kernel_contract_valid = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_library_apply_low_kernel_contract(
      module, out_plan->low_function_op, entry, diagnostic_emitter, table_arena,
      &kernel_contract_valid));
  if (!kernel_contract_valid) {
    return iree_ok_status();
  }
  if (report != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_library_symbol_name(
        module, entry->func_ref, &report->lowered_symbol));
  }

  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_library_verify_kernel_abi(
    const loom_module_t* module,
    const loom_target_low_descriptor_registry_t* low_registry,
    const loom_amdgpu_hal_kernel_library_kernel_plan_t* plan,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    uint32_t max_errors, bool* out_failed,
    iree_arena_allocator_t* table_arena) {
  *out_failed = false;
  const loom_low_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(loom_target_low_descriptor_set_select_for_bundle(
      &low_registry->registry, &plan->entry->bundle_storage.bundle,
      &descriptor_set));
  loom_amdgpu_hal_kernel_abi_verify_result_t result = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_verify_low(
      module, plan->low_function_op, descriptor_set, max_errors,
      loom_target_entry_emitter(diagnostic_emitter), &result, table_arena));
  *out_failed = result.error_count != 0;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_library_compute_kernel_fixed_values(
    const loom_module_t* module,
    loom_amdgpu_hal_kernel_library_kernel_plan_t* plan,
    iree_arena_allocator_t* table_arena) {
  if (loom_amdgpu_hal_kernel_abi_has_layout_attr(plan->low_function_op)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_layout_from_attr(
        module, plan->low_function_op, &plan->abi_layout, table_arena));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_layout_from_low(
        module, plan->low_function_op, &plan->abi_layout, table_arena));
  }
  return loom_amdgpu_hal_kernel_abi_fixed_values_from_low(
      module, plan->low_function_op, &plan->fixed_values,
      &plan->fixed_value_count, table_arena);
}

typedef struct loom_amdgpu_hal_kernel_library_spill_lowering_context_t {
  // Selected descriptor set used to rewrite structural spill traffic.
  const loom_low_descriptor_set_t* descriptor_set;
} loom_amdgpu_hal_kernel_library_spill_lowering_context_t;

static iree_status_t loom_amdgpu_hal_kernel_library_lower_spill_traffic(
    void* user_data, loom_module_t* module, loom_op_t* low_function_op,
    iree_arena_allocator_t* table_arena) {
  const loom_amdgpu_hal_kernel_library_spill_lowering_context_t* context =
      (const loom_amdgpu_hal_kernel_library_spill_lowering_context_t*)user_data;
  return loom_amdgpu_lower_spill_traffic(module, low_function_op,
                                         context->descriptor_set, table_arena);
}

static iree_status_t loom_amdgpu_hal_kernel_library_materialize_address_state(
    void* user_data, loom_module_t* module, loom_op_t* low_function_op,
    const loom_low_emission_frame_t* frame, iree_arena_allocator_t* table_arena,
    loom_low_emission_frame_materialize_address_state_result_t* out_result) {
  (void)user_data;
  return loom_amdgpu_materialize_address_state(module, low_function_op, frame,
                                               table_arena, out_result);
}

static iree_status_t loom_amdgpu_hal_kernel_library_build_kernel_contribution(
    loom_module_t* module,
    const loom_target_low_descriptor_registry_t* low_registry,
    const loom_amdgpu_hal_kernel_library_kernel_plan_t* plan,
    loom_target_selection_t target_selection,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_arena_allocator_t* table_arena, iree_string_builder_t* target_listing,
    loom_target_compile_report_t* report,
    loom_amdgpu_kernel_hsaco_contribution_t* out_contribution) {
  *out_contribution = (loom_amdgpu_kernel_hsaco_contribution_t){0};

  const loom_low_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(loom_target_low_descriptor_set_select_for_bundle(
      &low_registry->registry, &plan->entry->bundle_storage.bundle,
      &descriptor_set));
  loom_low_schedule_pressure_cliff_list_t schedule_pressure_cliffs =
      loom_low_schedule_pressure_cliff_list_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_occupancy_build_schedule_pressure_cliffs(
      descriptor_set, table_arena, &schedule_pressure_cliffs));
  loom_low_schedule_pair_affinity_list_t schedule_pair_affinities =
      loom_low_schedule_pair_affinity_list_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_build_schedule_pair_affinities(
      descriptor_set, table_arena, &schedule_pair_affinities));

  loom_low_emission_frame_t frame = {0};
  loom_low_storage_lease_provider_t storage_lease_provider = {0};
  loom_amdgpu_storage_lease_provider(&storage_lease_provider);
  const loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = &low_registry->registry,
      .target_selection =
          {
              .bundle = &plan->entry->bundle_storage.bundle,
              .data = target_selection.data,
          },
      .schedule_pressure_cliffs = schedule_pressure_cliffs,
      .schedule_pair_affinities = schedule_pair_affinities,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
      .memory_access_table = loom_low_memory_access_table_empty(),
      .allocation_fixed_values = plan->fixed_values,
      .allocation_fixed_value_count = plan->fixed_value_count,
      .storage_lease_provider = &storage_lease_provider,
      .emitter = loom_target_entry_emitter(diagnostic_emitter),
  };
  loom_amdgpu_native_preflight_t preflight = {0};
  loom_amdgpu_hal_kernel_library_spill_lowering_context_t
      spill_lowering_context = {
          .descriptor_set = descriptor_set,
      };
  const loom_low_emission_frame_spill_free_options_t spill_free_options = {
      .materialization_options =
          {
              .has_supported_storage_spaces = true,
              .supported_storage_spaces = LOOM_LOW_STORAGE_SPACE_SET_SCRATCH |
                                          LOOM_LOW_STORAGE_SPACE_SET_PRIVATE,
              .emit_spill_diagnostics = true,
              .emitter = frame_options.emitter,
          },
      .lower_spill_traffic = loom_amdgpu_hal_kernel_library_lower_spill_traffic,
      .lower_spill_traffic_user_data = &spill_lowering_context,
      .materialize_address_state =
          loom_amdgpu_hal_kernel_library_materialize_address_state,
      .materialize_address_state_user_data = NULL,
  };
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_build_spill_free(
      module, plan->low_function_op, &frame_options, &spill_free_options,
      table_arena, &frame));
  if (diagnostic_emitter->error_count != 0) {
    return iree_ok_status();
  }
  const loom_amdgpu_native_preflight_options_t preflight_options = {
      .emitter = frame_options.emitter,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_native_preflight_analyze(
      &frame.schedule, &frame.allocation, &preflight_options, &preflight));
  if (preflight.error_count != 0) {
    return iree_ok_status();
  }
  if (report != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_record_low_emission_frame(report, &frame));
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_library_build_hsaco_contribution(
      &frame, &plan->abi_layout, &preflight, target_listing, out_contribution,
      table_arena));
  if (report != NULL) {
    loom_target_compile_report_record_emission(
        report, out_contribution->summary.instruction_count,
        out_contribution->summary.text_byte_count,
        out_contribution->summary.text_storage_byte_count);
    loom_target_compile_report_record_memory(
        report, out_contribution->summary.private_segment_fixed_size,
        out_contribution->summary.group_segment_fixed_size);
  }

  const loom_target_hal_kernel_abi_t* hal_kernel =
      &plan->entry->bundle_storage.bundle.export_plan->hal_kernel;
  IREE_RETURN_IF_ERROR(loom_target_require_concrete_hal_kernel_launch(
      hal_kernel, IREE_SV("AMDGPU HAL kernel-library entry")));
  return iree_ok_status();
}

static void loom_amdgpu_hal_kernel_library_deinitialize_entry_reports(
    uint16_t report_count, loom_target_compile_report_t* reports) {
  for (uint16_t i = 0; i < report_count && reports != NULL; ++i) {
    loom_target_compile_report_deinitialize(&reports[i]);
  }
}

static iree_status_t loom_amdgpu_hal_kernel_library_entries(
    loom_module_t* module, const loom_target_entry_options_t* target_options,
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_target_entry_list_t entries, loom_target_selection_t target_selection,
    loom_target_entry_diagnostic_emitter_t* diagnostic_emitter,
    iree_arena_allocator_t* table_arena, loom_target_compile_report_t* report,
    const loom_amdgpu_hal_kernel_library_options_t* options, bool* out_emitted,
    loom_amdgpu_hal_kernel_library_t* out_library, iree_allocator_t allocator) {
  *out_emitted = false;
  if (entries.count == 0 || entries.values == NULL) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "AMDGPU HAL kernel-library emission requires at "
                            "least one entry");
  }

  const uint32_t max_errors = loom_target_entry_max_errors(
      target_options, LOOM_AMDGPU_HAL_KERNEL_LIBRARY_DEFAULT_MAX_ERRORS);
  const bool capture_target_listing =
      options ? options->capture_target_listing : false;
  loom_target_compile_report_t* entry_reports = NULL;
  if (report != NULL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(table_arena, entries.count,
                                                   sizeof(*entry_reports),
                                                   (void**)&entry_reports));
    memset(entry_reports, 0, entries.count * sizeof(*entry_reports));
    for (uint16_t i = 0; i < entries.count; ++i) {
      loom_target_compile_report_initialize(&entry_reports[i],
                                            report->allocator);
      entry_reports[i].requested_detail_flags = report->requested_detail_flags;
      entry_reports[i].artifact_kind =
          LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_KERNEL_LIBRARY;
      entry_reports[i].backend_name = report->backend_name;
      entry_reports[i].target_family_name = report->target_family_name;
      entry_reports[i].target_key = report->target_key;
      entry_reports[i].executable_format = report->executable_format;
    }
  }
  loom_amdgpu_hal_kernel_library_kernel_plan_t* plans = NULL;
  iree_status_t status = iree_arena_allocate_array(
      table_arena, entries.count, sizeof(*plans), (void**)&plans);
  bool diagnostics_failed = false;
  for (uint16_t i = 0;
       i < entries.count && iree_status_is_ok(status) && !diagnostics_failed;
       ++i) {
    status = loom_amdgpu_hal_kernel_library_prepare_kernel_plan(
        module, &entries.values[i], diagnostic_emitter, table_arena,
        entry_reports != NULL ? &entry_reports[i] : NULL, &plans[i]);
    if (iree_status_is_ok(status) && plans[i].low_function_op == NULL) {
      diagnostics_failed = true;
    }
  }

  bool abi_failed = false;
  for (uint16_t i = 0;
       i < entries.count && iree_status_is_ok(status) && !diagnostics_failed;
       ++i) {
    bool plan_failed = false;
    status = loom_amdgpu_hal_kernel_library_verify_kernel_abi(
        module, low_registry, &plans[i], diagnostic_emitter, max_errors,
        &plan_failed, table_arena);
    if (iree_status_is_ok(status)) {
      abi_failed |= plan_failed;
    }
  }
  if (abi_failed) {
    diagnostics_failed = true;
  }

  for (uint16_t i = 0;
       i < entries.count && iree_status_is_ok(status) && !diagnostics_failed;
       ++i) {
    status = loom_amdgpu_hal_kernel_library_compute_kernel_fixed_values(
        module, &plans[i], table_arena);
  }

  loom_verify_result_t verify_result = {0};
  if (iree_status_is_ok(status) && !diagnostics_failed) {
    status = loom_target_entry_verify_module(
        module, target_options,
        LOOM_AMDGPU_HAL_KERNEL_LIBRARY_DEFAULT_MAX_ERRORS, &verify_result);
    if (iree_status_is_ok(status) && verify_result.error_count != 0) {
      diagnostics_failed = true;
    }
  }
  loom_low_verify_result_t low_verify_result = {0};
  loom_low_verify_scratch_t low_verify_scratch =
      loom_low_verify_scratch_for_module(module);
  if (iree_status_is_ok(status) && !diagnostics_failed) {
    status = loom_target_entry_verify_low_module(
        module, low_registry, diagnostic_emitter, target_selection, max_errors,
        loom_low_verify_provider_list_empty(), &low_verify_scratch,
        &low_verify_result);
    if (iree_status_is_ok(status) && low_verify_result.error_count != 0) {
      diagnostics_failed = true;
    }
  }

  loom_amdgpu_kernel_hsaco_contribution_t* contributions = NULL;
  if (iree_status_is_ok(status) && !diagnostics_failed) {
    status = iree_arena_allocate_array(table_arena, entries.count,
                                       sizeof(*contributions),
                                       (void**)&contributions);
  }
  iree_string_builder_t target_listing;
  bool target_listing_initialized = false;
  if (iree_status_is_ok(status) && !diagnostics_failed &&
      capture_target_listing) {
    iree_string_builder_initialize(allocator, &target_listing);
    target_listing_initialized = true;
  }
  for (uint16_t i = 0;
       i < entries.count && iree_status_is_ok(status) && !diagnostics_failed;
       ++i) {
    status = loom_amdgpu_hal_kernel_library_build_kernel_contribution(
        module, low_registry, &plans[i], target_selection, diagnostic_emitter,
        table_arena, target_listing_initialized ? &target_listing : NULL,
        entry_reports != NULL ? &entry_reports[i] : NULL, &contributions[i]);
  }
  if (iree_status_is_ok(status) && !diagnostics_failed &&
      diagnostic_emitter->error_count == 0) {
    for (uint16_t i = 0; i < entries.count && entry_reports != NULL &&
                         iree_status_is_ok(status);
         ++i) {
      status = loom_target_compile_report_record_entry_report(
          report, &entry_reports[i]);
    }
  }
  if (iree_status_is_ok(status) && !diagnostics_failed &&
      diagnostic_emitter->error_count == 0) {
    iree_const_byte_span_t hsaco = iree_const_byte_span_empty();
    status = loom_amdgpu_hal_kernel_library_write_hsaco(
        contributions, entries.count, &hsaco, table_arena, allocator);
    if (iree_status_is_ok(status)) {
      status = loom_amdgpu_hal_kernel_library_set_contents(
          contributions[0].target, hsaco, allocator, out_library);
    }
    if (iree_status_is_ok(status)) {
      hsaco = iree_const_byte_span_empty();
      if (capture_target_listing &&
          iree_string_builder_size(&target_listing) != 0) {
        out_library->target_listing_format = IREE_SV("amdgpu-assembly");
        out_library->target_listing_data_length =
            iree_string_builder_size(&target_listing);
        out_library->target_listing_data =
            iree_string_builder_take_storage(&target_listing);
      }
      if (options != NULL && options->artifact_manifest.mode !=
                                 LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
        loom_target_artifact_manifest_collect_options_t manifest_options =
            options->artifact_manifest;
        manifest_options.artifact_name = options->artifact_name;
        manifest_options.artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
        manifest_options.flags =
            LOOM_TARGET_ARTIFACT_MANIFEST_COLLECT_FLAG_ARTIFACT_BYTE_LENGTH;
        manifest_options.artifact_byte_length = out_library->hsaco_data_length;
        loom_target_artifact_manifest_json_t artifact_manifest_json = {0};
        status = loom_target_artifact_manifest_collect_json_from_entries(
            module, entries, &manifest_options, table_arena, allocator,
            &artifact_manifest_json);
        if (iree_status_is_ok(status) &&
            artifact_manifest_json.contents.data != NULL) {
          out_library->artifact_manifest =
              (loom_target_emit_sidecar_artifact_t){
                  .kind =
                      LOOM_TARGET_EMIT_SIDECAR_ARTIFACT_KIND_ARTIFACT_MANIFEST,
                  .identifier = options->artifact_manifest_identifier,
                  .contents = artifact_manifest_json.contents,
              };
        }
      }
      if (iree_status_is_ok(status)) {
        *out_emitted = true;
      }
    }
    iree_allocator_free(allocator, (void*)hsaco.data);
  }
  if (target_listing_initialized) {
    iree_string_builder_deinitialize(&target_listing);
  }
  loom_amdgpu_hal_kernel_library_deinitialize_entry_reports(entries.count,
                                                            entry_reports);
  return status;
}

iree_status_t loom_amdgpu_emit_hal_kernel_library(
    loom_module_t* module,
    const loom_amdgpu_hal_kernel_library_options_t* options,
    iree_allocator_t allocator, bool* out_emitted,
    loom_amdgpu_hal_kernel_library_t* out_library) {
  *out_emitted = false;
  *out_library = (loom_amdgpu_hal_kernel_library_t){0};
  loom_target_compile_report_t* report = options ? options->report : NULL;
  if (report != NULL) {
    loom_target_compile_report_initialize_if_empty(report, allocator);
    report->artifact_kind =
        LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_KERNEL_LIBRARY;
  }
  const loom_target_selection_t target_selection =
      options ? options->target_selection : loom_target_selection_empty();
  const loom_target_entry_options_t target_options = {
      .diagnostic_sink =
          options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
      .source_resolver =
          options ? options->source_resolver : (loom_source_resolver_t){0},
      .max_errors = options ? options->max_errors : 0,
      .effective_target_bundle = target_selection.bundle,
  };
  loom_target_environment_t target_environment = {0};
  iree_status_t status = loom_target_environment_initialize(
      &loom_amdgpu_target_provider_set, &target_environment);
  loom_target_low_descriptor_registry_t low_registry = {0};
  if (iree_status_is_ok(status)) {
    status = loom_target_environment_initialize_low_descriptor_registry(
        &target_environment, &low_registry);
  }
  loom_target_entry_diagnostic_emitter_t diagnostic_emitter = {0};
  loom_target_entry_diagnostic_emitter_initialize(
      module, &target_options, LOOM_EMITTER_VERIFIER, &diagnostic_emitter);
  const loom_target_entry_predicate_t entry_predicate = {
      .fn = loom_amdgpu_hal_kernel_library_bundle_is_compatible,
      .user_data = NULL,
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, allocator, &block_pool);
  iree_arena_allocator_t table_arena;
  iree_arena_initialize(&block_pool, &table_arena);

  loom_target_entry_list_t entries = {0};
  loom_verify_result_t verify_result = {0};
  if (iree_status_is_ok(status)) {
    status = loom_target_entry_verify_module(
        module, &target_options,
        LOOM_AMDGPU_HAL_KERNEL_LIBRARY_DEFAULT_MAX_ERRORS, &verify_result);
  }
  bool selected = false;
  if (iree_status_is_ok(status) && verify_result.error_count == 0 &&
      diagnostic_emitter.error_count == 0) {
    status = loom_target_entry_select_all_entries(
        module, &target_options, entry_predicate, &diagnostic_emitter,
        IREE_SV("AMDGPU HAL-native"), &table_arena, &selected, &entries);
  }
  if (iree_status_is_ok(status) && selected && options != NULL) {
    for (uint16_t i = 0; i < entries.count && iree_status_is_ok(status); ++i) {
      status = loom_amdgpu_hal_kernel_library_apply_processor(
          module, &entries.values[i], &diagnostic_emitter, options->processor);
    }
  }
  if (iree_status_is_ok(status) && selected &&
      diagnostic_emitter.error_count == 0 && report != NULL) {
    if (entries.count == 1) {
      loom_target_compile_report_record_target_bundle(
          report, &entries.values[0].bundle_storage.bundle);
    } else if (entries.count > 0) {
      report->target_bundle_name = entries.values[0].bundle_storage.bundle.name;
      if (entries.values[0].bundle_storage.bundle.snapshot != NULL) {
        report->target_snapshot_name =
            entries.values[0].bundle_storage.bundle.snapshot->name;
      }
    }
  }
  if (iree_status_is_ok(status) && selected &&
      diagnostic_emitter.error_count == 0) {
    status = loom_amdgpu_hal_kernel_library_entries(
        module, &target_options, &low_registry, entries, target_selection,
        &diagnostic_emitter, &table_arena, report, options, out_emitted,
        out_library, allocator);
  }
  if (iree_status_is_ok(status) && *out_emitted && report != NULL) {
    loom_target_compile_report_record_artifact_size(
        report, out_library->hsaco_data_length);
  }

  if (!iree_status_is_ok(status)) {
    loom_amdgpu_hal_kernel_library_deinitialize(out_library, allocator);
  }
  if (report != NULL) {
    loom_target_compile_report_record_status(report, iree_status_code(status));
  }
  iree_arena_deinitialize(&table_arena);
  iree_arena_block_pool_deinitialize(&block_pool);
  loom_target_environment_deinitialize(&target_environment);
  return status;
}
