// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/module.h"

#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/error_catalog.h"
#include "loom/target/emit/native/x86/abi.h"
#include "loom/target/emit/native/x86/function.h"

static iree_status_t loom_x86_module_reject_callable(
    const loom_target_emit_request_t* request, const loom_target_entry_t* entry,
    iree_string_view_t constraint) {
  const loom_target_bundle_t* bundle = loom_target_entry_bundle(entry);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(bundle->snapshot->name),
      loom_param_string(bundle->export_plan->name),
      loom_param_string(bundle->config->name),
      loom_param_string(entry->func_name),
      loom_param_string(loom_op_name(request->module, entry->func.op)),
      loom_param_string(constraint),
  };
  return iree_diagnostic_emit(request->diagnostic_emitter,
                              &(loom_diagnostic_emission_t){
                                  .op = entry->func.op,
                                  .error = LOOM_ERR_TARGET_032,
                                  .params = params,
                                  .param_count = IREE_ARRAYSIZE(params),
                              });
}

static iree_status_t loom_x86_module_prepare(
    const loom_target_emit_request_t* request,
    const loom_target_entry_list_t* entries, iree_arena_allocator_t* arena,
    bool* out_accepted, void** out_context) {
  *out_accepted = false;
  *out_context = NULL;
  loom_x86_module_abi_t* module_abi = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*module_abi), (void**)&module_abi));
  IREE_RETURN_IF_ERROR(loom_x86_module_abi_initialize(
      request->module->symbols.count, entries->count, arena, module_abi));
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, arena);
  bool accepted = true;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < entries->count && iree_status_is_ok(status);
       ++i) {
    const loom_target_entry_t* entry = &entries->values[i];
    const iree_string_view_t convention =
        loom_target_entry_bundle(entry)->export_plan->calling_convention;
    if (!iree_string_view_is_empty(convention) &&
        !iree_string_view_equal(convention, IREE_SV("sysv"))) {
      accepted = false;
      status = loom_x86_module_reject_callable(
          request, entry,
          IREE_SV("native x86 supports the sysv calling convention"));
      continue;
    }
    loom_low_resolved_target_t target = {0};
    status = loom_low_resolve_function_target(
        request->module, &symbol_facts, entry->func.op, entry->target_facts,
        request->low_descriptor_registry, request->diagnostic_emitter, &target);
    if (!iree_status_is_ok(status)) {
      continue;
    }
    if (target.descriptor_set == NULL) {
      accepted = false;
      continue;
    }
    bool function_supported = false;
    iree_string_view_t constraint = iree_string_view_empty();
    loom_x86_function_abi_t* function_abi = &module_abi->functions[i];
    status = loom_x86_function_abi_prepare(request->module, entry->func,
                                           &target, arena, &function_supported,
                                           &constraint, function_abi);
    if (!iree_status_is_ok(status)) {
      continue;
    }
    if (!function_supported) {
      accepted = false;
      status = loom_x86_module_reject_callable(request, entry, constraint);
      continue;
    }
    loom_x86_module_abi_bind(module_abi, i, entry->func_ref.symbol_id);
  }
  if (iree_status_is_ok(status)) {
    *out_accepted = accepted;
    *out_context = module_abi;
  }
  return status;
}

static iree_status_t loom_x86_module_encode_function(
    const loom_target_emit_request_t* request, const loom_target_entry_t* entry,
    const void* context, const uint32_t* symbol_indices,
    iree_host_size_t section_index, loom_native_module_fixups_t* fixups,
    iree_arena_allocator_t* function_arena, iree_io_stream_t* stream,
    bool* out_accepted) {
  const loom_x86_module_abi_t* module_abi = context;
  const loom_x86_function_abi_t* abi =
      loom_x86_module_abi_lookup(module_abi, entry->func_ref);
  IREE_ASSERT_NE(abi, NULL);
  loom_low_allocation_reserved_range_t reserved_ranges[3] = {{{0}}};
  const iree_host_size_t reserved_range_count =
      loom_x86_function_reserved_ranges(abi, reserved_ranges);
  const loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = request->low_descriptor_registry,
      .resolved_target = &abi->target,
      .memory_accesses = entry->function_version
                             ? entry->function_version->memory_accesses
                             : NULL,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
      .allocation_entry_locations = abi->call_contract.arguments,
      .allocation_entry_location_count = abi->call_contract.argument_count,
      .allocation_exit_locations = abi->call_contract.results,
      .allocation_exit_location_count = abi->call_contract.result_count,
      .call_contracts =
          {
              .validate = loom_x86_function_call_contract_validate,
              .query = loom_x86_function_call_contract,
              .query_common_clobbers = loom_x86_function_common_call_clobbers,
              .user_data = (void*)module_abi,
          },
      .synchronous_storage_spaces = LOOM_LOW_STORAGE_SPACE_SET_STACK |
                                    LOOM_LOW_STORAGE_SPACE_SET_PRIVATE |
                                    LOOM_LOW_STORAGE_SPACE_SET_SCRATCH,
      .allocation_reserved_ranges = reserved_ranges,
      .allocation_reserved_range_count = reserved_range_count,
      .emitter = request->diagnostic_emitter,
  };
  const loom_low_emission_frame_spill_free_options_t spill_options = {0};
  loom_low_emission_frame_t frame = {0};
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_build_spill_free(
      request->module, (loom_op_t*)entry->func.op, &frame_options,
      &spill_options, function_arena, &frame, out_accepted));
  if (!*out_accepted) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_native_module_check_calls(
      request, &frame.schedule, symbol_indices, out_accepted));
  if (!*out_accepted) {
    return iree_ok_status();
  }
  loom_x86_function_t function;
  IREE_RETURN_IF_ERROR(loom_x86_function_prepare(
      &frame, abi, module_abi, request->diagnostic_emitter, function_arena,
      out_accepted, &function));
  if (!*out_accepted) {
    return iree_ok_status();
  }
  if (function.symbol_fixup_count) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        request->scratch_arena, fixups->count,
        fixups->count + function.symbol_fixup_count, sizeof(*fixups->values),
        &fixups->capacity, (void**)&fixups->values));
  }
  IREE_RETURN_IF_ERROR(loom_x86_function_write(
      &function, symbol_indices, section_index,
      function.symbol_fixup_count ? fixups->values + fixups->count : NULL,
      stream, function_arena));
  fixups->count += function.symbol_fixup_count;
  return iree_ok_status();
}

iree_status_t loom_x86_module_emit(
    const loom_target_emit_request_t* request,
    const loom_target_fact_type_t* target_fact_type,
    loom_native_module_format_t format, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact) {
  static const loom_native_elf_relocation_t relocations[] = {
      [LOOM_X86_RELOCATION_CALL] =
          {
              .type = 4,  // R_X86_64_PLT32.
              .image_encoding = LOOM_NATIVE_ELF_FIXUP_PC_RELATIVE_32,
          },
      [LOOM_X86_RELOCATION_ADDRESS] =
          {
              .type = 2,  // R_X86_64_PC32.
              .image_encoding = LOOM_NATIVE_ELF_FIXUP_PC_RELATIVE_32,
          },
      [LOOM_X86_RELOCATION_POINTER] =
          {
              .type = 1,  // R_X86_64_64.
              .image_encoding = LOOM_NATIVE_ELF_FIXUP_ABSOLUTE_64,
          },
  };
  static const loom_native_module_elf_target_t target = {
      .prepare_module = loom_x86_module_prepare,
      .emit_function = loom_x86_module_encode_function,
      .function_alignment = 16,
      .pointer_relocation_kind = LOOM_X86_RELOCATION_POINTER,
      .image_options =
          {
              .machine = LOOM_NATIVE_ELF_MACHINE_X86_64,
              .page_alignment = 4096,
              .relative_relocation_type = 8,  // R_X86_64_RELATIVE.
              .relocations = relocations,
          },
  };
  return loom_native_module_emit_elf64le(request, target_fact_type, format,
                                         &target, out_emitted, out_artifact);
}
