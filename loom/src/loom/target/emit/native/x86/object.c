// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/object.h"

#include <string.h>

#include "iree/io/vec_stream.h"
#include "loom/codegen/low/frame.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/x86/ops/ops.h"
#include "loom/target/arch/x86/sysv_frame.h"
#include "loom/target/emit/native/elf_object.h"
#include "loom/target/emit/native/x86/function.h"
#include "loom/target/function_version.h"
#include "loom/target/reporting/low.h"

typedef enum loom_x86_native_relocation_e {
  // Direct PC-relative reference to a module-internal definition.
  LOOM_X86_NATIVE_RELOCATION_PC32 = 1,
  // PC-relative reference preserving ELF symbol interposition.
  LOOM_X86_NATIVE_RELOCATION_PLT32 = 2,
} loom_x86_native_relocation_t;

typedef struct loom_x86_object_function_t {
  // Module symbol defining this function.
  loom_symbol_id_t symbol_id;
  // Function target facts retained by the shared specialization pipeline.
  const loom_target_facts_t* target_facts;
  // Encoded function body and direct-call fixups.
  loom_x86_function_encoding_t encoding;
} loom_x86_object_function_t;

typedef struct loom_x86_object_plan_t {
  // Encoded functions in module symbol order.
  loom_x86_object_function_t* functions;
  // Number of entries in |functions|.
  uint32_t function_count;
  // Total direct-call fixups across all encoded functions.
  uint32_t fixup_count;
  // True when frame construction emitted a user-facing diagnostic.
  bool has_errors;
} loom_x86_object_plan_t;

static bool loom_x86_object_function_is_x86(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_facts_t* target_facts) {
  if (target_facts != NULL) {
    return target_facts->fact_type == &loom_x86_target_fact_type;
  }
  const loom_string_id_t contract_id = loom_func_like_repr_contract(function);
  return contract_id < module->strings.count &&
         iree_string_view_starts_with(module->strings.entries[contract_id],
                                      IREE_SV("x86."));
}

static iree_status_t loom_x86_object_collect_functions(
    const loom_target_emit_request_t* request,
    const loom_target_function_version_snapshot_t* versions,
    loom_x86_object_plan_t* out_plan) {
  *out_plan = (loom_x86_object_plan_t){0};
  uint32_t function_count = 0;
  for (loom_symbol_id_t symbol_id = 0;
       symbol_id < request->module->symbols.count; ++symbol_id) {
    const loom_target_function_version_t* version =
        loom_target_function_version_snapshot_at(versions, symbol_id);
    if (version == NULL) {
      continue;
    }
    loom_func_like_t function = version->base.function;
    if (!loom_low_func_def_isa(function.op) ||
        !loom_x86_object_function_is_x86(request->module, function,
                                         version->function_target_facts)) {
      continue;
    }
    ++function_count;
  }
  if (function_count == 0) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "x86 object emission requires a concrete x86 function definition");
  }

  loom_x86_object_function_t* functions = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(request->scratch_arena, function_count,
                                sizeof(*functions), (void**)&functions));
  uint32_t function_index = 0;
  for (loom_symbol_id_t symbol_id = 0;
       symbol_id < request->module->symbols.count; ++symbol_id) {
    const loom_target_function_version_t* version =
        loom_target_function_version_snapshot_at(versions, symbol_id);
    if (version == NULL) {
      continue;
    }
    loom_func_like_t function = version->base.function;
    if (!loom_low_func_def_isa(function.op) ||
        !loom_x86_object_function_is_x86(request->module, function,
                                         version->function_target_facts)) {
      continue;
    }
    functions[function_index++] = (loom_x86_object_function_t){
        .symbol_id = symbol_id,
        .target_facts = version->function_target_facts,
    };
  }
  IREE_ASSERT_EQ(function_index, function_count);
  out_plan->functions = functions;
  out_plan->function_count = function_count;
  return iree_ok_status();
}

static iree_status_t loom_x86_object_encode_functions(
    const loom_target_emit_request_t* request, loom_x86_object_plan_t* plan) {
  iree_arena_allocator_t function_arena;
  iree_arena_initialize(request->scratch_arena->block_pool, &function_arena);
  iree_status_t status = iree_ok_status();
  uint64_t fixup_count = 0;
  for (uint32_t i = 0; i < plan->function_count && iree_status_is_ok(status);
       ++i) {
    const loom_x86_object_function_t* function = &plan->functions[i];
    const loom_symbol_t* symbol =
        &request->module->symbols.entries[function->symbol_id];
    const loom_low_allocation_reserved_range_t stack_pointer_reservation =
        loom_x86_sysv_frame_stack_pointer_reservation();
    loom_low_planning_statistics_t planning_statistics = {0};
    const loom_low_emission_frame_options_t frame_options = {
        .descriptor_registry = request->low_descriptor_registry,
        .function_target_facts = function->target_facts,
        .memory_access_table = loom_low_memory_access_table_empty(),
        .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
        .allocation_reserved_ranges = &stack_pointer_reservation,
        .allocation_reserved_range_count = 1,
        .emitter = request->diagnostic_emitter,
        .statistics =
            request->compile_report != NULL ? &planning_statistics : NULL,
    };
    const loom_low_emission_frame_spill_free_options_t spill_free_options = {
        .materialization_options =
            {
                .has_supported_storage_spaces = true,
                .supported_storage_spaces = LOOM_LOW_STORAGE_SPACE_SET_STACK,
                .emit_spill_diagnostics = true,
                .record_materialized_spills =
                    loom_target_compile_report_wants_details(
                        request->compile_report,
                        LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS),
                .emitter = request->diagnostic_emitter,
            },
    };
    loom_low_emission_frame_t frame = {0};
    status = loom_low_emission_frame_build_spill_free(
        request->module, symbol->defining_op, &frame_options,
        &spill_free_options, &function_arena, &frame);
    if (iree_status_is_ok(status) &&
        (frame.function_op == NULL || frame.schedule.error_count != 0 ||
         frame.allocation.error_count != 0 ||
         frame.allocation.spill_count != 0 ||
         frame.allocation.spill_plan_count != 0)) {
      plan->has_errors = true;
      break;
    }
    if (iree_status_is_ok(status) && request->compile_report != NULL) {
      loom_target_compile_report_record_low_planning(request->compile_report,
                                                     &planning_statistics);
      status = loom_target_compile_report_record_low_emission_frame(
          request->compile_report, &frame);
    }
    if (iree_status_is_ok(status)) {
      status = loom_x86_encode_sysv_function(&frame, request->scratch_arena,
                                             &plan->functions[i].encoding);
    }
    if (iree_status_is_ok(status)) {
      fixup_count += plan->functions[i].encoding.call_fixup_count;
      if (fixup_count > UINT32_MAX) {
        status = iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "x86 object direct-call fixup count exceeds u32 range");
      }
    }
    iree_arena_reset(&function_arena);
  }
  iree_arena_deinitialize(&function_arena);
  if (iree_status_is_ok(status)) {
    plan->fixup_count = (uint32_t)fixup_count;
  }
  return status;
}

static iree_string_view_t loom_x86_object_module_symbol_name(
    const loom_module_t* module, loom_symbol_id_t symbol_id) {
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  IREE_ASSERT_NE(symbol->name_id, LOOM_STRING_ID_INVALID);
  IREE_ASSERT_LT(symbol->name_id, module->strings.count);
  return module->strings.entries[symbol->name_id];
}

static iree_string_view_t loom_x86_object_definition_name(
    const loom_module_t* module, loom_symbol_id_t symbol_id) {
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  const loom_func_like_t function =
      loom_func_like_const_cast(module, symbol->defining_op);
  const loom_string_id_t export_name = loom_func_like_export_symbol(function);
  return export_name < module->strings.count
             ? module->strings.entries[export_name]
             : loom_x86_object_module_symbol_name(module, symbol_id);
}

static iree_status_t loom_x86_object_import_name(const loom_module_t* module,
                                                 loom_symbol_id_t symbol_id,
                                                 iree_string_view_t* out_name) {
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  const loom_op_t* declaration = symbol->defining_op;
  if (!loom_low_func_decl_isa(declaration)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "x86 direct call target %u is not an emitted definition or native "
        "function declaration",
        (unsigned)symbol_id);
  }
  const loom_attribute_t import_kind_attr = loom_op_const_attrs(
      declaration)[loom_low_func_decl_import_kind_ATTR_INDEX];
  const loom_attribute_t code_symbol_attr = loom_op_const_attrs(
      declaration)[loom_low_func_decl_code_symbol_ATTR_INDEX];
  if (loom_attr_is_absent(import_kind_attr) ||
      loom_attr_as_enum(import_kind_attr) !=
          LOOM_LOW_FUNC_DECL_IMPORT_KIND_NATIVE ||
      loom_attr_is_absent(code_symbol_attr)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "x86 object emission requires native function imports");
  }
  const loom_string_id_t code_symbol = loom_attr_as_string_id(code_symbol_attr);
  if (code_symbol >= module->strings.count ||
      iree_string_view_is_empty(module->strings.entries[code_symbol])) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "x86 native import has no code symbol");
  }
  *out_name = module->strings.entries[code_symbol];
  return iree_ok_status();
}

static iree_status_t loom_x86_object_build_contribution(
    const loom_target_emit_request_t* request,
    const loom_x86_object_plan_t* plan,
    loom_native_object_contribution_t* out_object) {
  *out_object = (loom_native_object_contribution_t){0};
  loom_native_section_contribution_t* sections = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(request->scratch_arena, plan->function_count,
                                sizeof(*sections), (void**)&sections));

  uint32_t* symbol_indices_by_module_symbol = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      request->scratch_arena, request->module->symbols.count,
      sizeof(*symbol_indices_by_module_symbol),
      (void**)&symbol_indices_by_module_symbol));
  memset(symbol_indices_by_module_symbol, 0xFF,
         request->module->symbols.count *
             sizeof(*symbol_indices_by_module_symbol));

  uint32_t symbol_count = 0;
  for (uint32_t i = 0; i < plan->function_count; ++i) {
    const loom_symbol_id_t symbol_id = plan->functions[i].symbol_id;
    IREE_ASSERT_EQ(symbol_indices_by_module_symbol[symbol_id], UINT32_MAX);
    symbol_indices_by_module_symbol[symbol_id] = symbol_count++;
  }
  for (uint32_t i = 0; i < plan->function_count; ++i) {
    const loom_x86_function_encoding_t* encoding = &plan->functions[i].encoding;
    for (iree_host_size_t j = 0; j < encoding->call_fixup_count; ++j) {
      const loom_symbol_ref_t target = encoding->call_fixups[j].target;
      if (!loom_symbol_ref_is_valid(target) || target.module_id != 0 ||
          target.symbol_id >= request->module->symbols.count) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "x86 direct call fixup has no module-local target");
      }
      if (symbol_indices_by_module_symbol[target.symbol_id] == UINT32_MAX) {
        symbol_indices_by_module_symbol[target.symbol_id] = symbol_count++;
      }
    }
  }

  loom_native_object_symbol_t* symbols = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(request->scratch_arena,
                                                 symbol_count, sizeof(*symbols),
                                                 (void**)&symbols));
  memset(symbols, 0, symbol_count * sizeof(*symbols));
  for (uint32_t i = 0; i < plan->function_count; ++i) {
    const loom_x86_object_function_t* function = &plan->functions[i];
    const loom_symbol_t* module_symbol =
        &request->module->symbols.entries[function->symbol_id];
    const loom_func_like_t function_like =
        loom_func_like_const_cast(request->module, module_symbol->defining_op);
    sections[i] = (loom_native_section_contribution_t){
        .section_name = IREE_SV(".text"),
        .kind = LOOM_NATIVE_SECTION_KIND_BYTES,
        .flags = LOOM_NATIVE_SECTION_FLAG_ALLOCATED |
                 LOOM_NATIVE_SECTION_FLAG_EXECUTABLE,
        .contribution_alignment = 16,
        .contents = function->encoding.text,
    };
    symbols[symbol_indices_by_module_symbol[function->symbol_id]] =
        (loom_native_object_symbol_t){
            .name = loom_x86_object_definition_name(request->module,
                                                    function->symbol_id),
            .section_contribution_index = i,
            .size = function->encoding.text.data_length,
            .binding = loom_func_like_is_module_internal(function_like)
                           ? LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL
                           : LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
            .visibility = LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
            .kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
            .definition = LOOM_NATIVE_OBJECT_SYMBOL_DEFINITION_SECTION,
        };
  }

  loom_native_object_fixup_t* fixups = NULL;
  if (plan->fixup_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(request->scratch_arena, plan->fixup_count,
                                  sizeof(*fixups), (void**)&fixups));
  }
  uint32_t fixup_index = 0;
  for (uint32_t i = 0; i < plan->function_count; ++i) {
    const loom_x86_function_encoding_t* encoding = &plan->functions[i].encoding;
    for (iree_host_size_t j = 0; j < encoding->call_fixup_count; ++j) {
      const loom_x86_function_call_fixup_t* function_fixup =
          &encoding->call_fixups[j];
      const loom_symbol_id_t target_id = function_fixup->target.symbol_id;
      const uint32_t target_symbol_index =
          symbol_indices_by_module_symbol[target_id];
      loom_native_object_symbol_t* target_symbol =
          &symbols[target_symbol_index];
      if (iree_string_view_is_empty(target_symbol->name)) {
        iree_string_view_t import_name = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_x86_object_import_name(
            request->module, target_id, &import_name));
        *target_symbol = (loom_native_object_symbol_t){
            .name = import_name,
            .binding = LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
            .visibility = LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
            .kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
            .definition = LOOM_NATIVE_OBJECT_SYMBOL_DEFINITION_UNDEFINED,
        };
      }
      const bool is_internal_definition =
          target_symbol->definition ==
              LOOM_NATIVE_OBJECT_SYMBOL_DEFINITION_SECTION &&
          target_symbol->binding == LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL;
      fixups[fixup_index++] = (loom_native_object_fixup_t){
          .section_contribution_index = i,
          .section_offset = function_fixup->text_offset,
          .relocation_kind = is_internal_definition
                                 ? LOOM_X86_NATIVE_RELOCATION_PC32
                                 : LOOM_X86_NATIVE_RELOCATION_PLT32,
          .target_symbol_index = target_symbol_index,
          .addend = -4,
      };
    }
  }
  IREE_ASSERT_EQ(fixup_index, plan->fixup_count);
  *out_object = (loom_native_object_contribution_t){
      .sections = sections,
      .section_count = plan->function_count,
      .symbols = symbols,
      .symbol_count = symbol_count,
      .fixups = fixups,
      .fixup_count = plan->fixup_count,
  };
  return iree_ok_status();
}

static uint32_t loom_x86_object_map_elf_relocation(
    const void* user_data, uint32_t native_relocation_kind) {
  (void)user_data;
  switch ((loom_x86_native_relocation_t)native_relocation_kind) {
    case LOOM_X86_NATIVE_RELOCATION_PC32:
      return 2;  // R_X86_64_PC32
    case LOOM_X86_NATIVE_RELOCATION_PLT32:
      return 4;  // R_X86_64_PLT32
  }
  IREE_ASSERT_UNREACHABLE("unknown x86 native relocation kind");
  return 0;
}

static iree_status_t loom_x86_elf_object_emit(
    const loom_target_emit_request_t* request,
    loom_target_emit_artifact_t* out_artifact) {
  *out_artifact = (loom_target_emit_artifact_t){0};
  if (request->artifact_manifest.mode !=
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "ordinary x86 object files do not contain kernel metadata");
  }

  loom_target_function_version_snapshot_t versions = {0};
  IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
      request->module, request->function_versions, request->scratch_arena,
      &versions));
  loom_x86_object_plan_t plan = {0};
  IREE_RETURN_IF_ERROR(
      loom_x86_object_collect_functions(request, &versions, &plan));
  IREE_RETURN_IF_ERROR(loom_x86_object_encode_functions(request, &plan));
  if (plan.has_errors) {
    return iree_ok_status();
  }

  loom_native_object_contribution_t object = {0};
  IREE_RETURN_IF_ERROR(
      loom_x86_object_build_contribution(request, &plan, &object));
  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE, 32 * 1024, request->allocator, &stream));
  const loom_native_elf64le_relocatable_options_t options = {
      .machine = LOOM_NATIVE_ELF_MACHINE_X86_64,
      .os_abi = LOOM_NATIVE_ELF_OS_ABI_NONE,
      .abi_version = LOOM_NATIVE_ELF_ABI_VERSION_NONE,
      .relocation_mapper =
          {
              .map = loom_x86_object_map_elf_relocation,
          },
  };
  iree_status_t status = loom_native_elf64le_write_relocatable_object(
      &object, &options, stream, request->scratch_arena);
  if (iree_status_is_ok(status)) {
    status = iree_io_vec_stream_move_contents(stream, &out_artifact->contents);
  }
  if (iree_status_is_ok(status)) {
    out_artifact->target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
  }
  iree_io_stream_release(stream);
  return status;
}

const loom_target_emitter_t loom_x86_elf_object_emitter = {
    .name = IREE_SVL("x86-elf-object"),
    .public_artifact_format = IREE_SVL("x86-elf-object"),
    .default_identifier = IREE_SVL("module.o"),
    .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .default_pipeline_options =
        {
            .control_flow_lowering = LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG,
        },
    .emit = loom_x86_elf_object_emit,
};
