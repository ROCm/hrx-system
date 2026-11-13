// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/module.h"

#include <stdlib.h>

#include "iree/io/vec_stream.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/abi/task/state_layout.h"
#include "loom/target/artifact_diagnostics.h"
#include "loom/target/emit/native/image_elf.h"
#include "loom/target/emit/native/object_elf.h"
#include "loom/target/emit/native/task_library.h"
#include "loom/target/reporting/report.h"

typedef struct loom_native_module_selection_t {
  // Architecture identity supplied by the composing provider.
  const loom_target_fact_type_t* fact_type;
  // Native artifact and ABI family requested by the caller.
  loom_native_module_format_t format;
} loom_native_module_selection_t;

static bool loom_native_module_accept_entry(void* user_data,
                                            const loom_target_entry_t* entry) {
  const loom_target_bundle_t* bundle = loom_target_entry_bundle(entry);
  const loom_native_module_selection_t* selection = user_data;
  return entry->target_facts->fact_type == selection->fact_type &&
         (bundle->export_plan->abi_kind == LOOM_TARGET_ABI_OBJECT_FUNCTION ||
          bundle->export_plan->abi_kind == LOOM_TARGET_ABI_UNKNOWN ||
          (selection->format == LOOM_NATIVE_MODULE_FORMAT_HAL_LIBRARY &&
           bundle->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL)) &&
         bundle->snapshot->artifact_format == LOOM_TARGET_ARTIFACT_FORMAT_ELF;
}

static int loom_native_module_compare_names(const void* lhs, const void* rhs) {
  const loom_native_object_symbol_t* left =
      *(const loom_native_object_symbol_t* const*)lhs;
  const loom_native_object_symbol_t* right =
      *(const loom_native_object_symbol_t* const*)rhs;
  const int order = iree_string_view_compare(left->name, right->name);
  // Both pointers are into the same symbol array. Preserve source order among
  // equal names so the later definition receives the collision diagnostic.
  return order ? order : (int)(left - right);
}

// Export names are authored input. Validate the final native namespace before
// encoding; the object writer consumes trusted symbol records.
static iree_status_t loom_native_module_symbols(
    const loom_module_t* module, const loom_target_entry_list_t* entries,
    loom_target_entry_diagnostic_emitter_t* diagnostics, uint32_t max_errors,
    iree_arena_allocator_t* arena, loom_native_object_symbol_t* symbols,
    iree_host_size_t* out_section_count) {
  const iree_diagnostic_emitter_t emitter =
      loom_target_entry_emitter(diagnostics);
  // Symbol pointers retain both the final spelling and its entry ordinal.
  const loom_native_object_symbol_t** names = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, entries->count, sizeof(*names), (void**)&names));
  uint16_t export_count = 0;
  iree_host_size_t section_count = 0;
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0;
       i < entries->count && diagnostics->error_count < max_errors &&
       iree_status_is_ok(status);
       ++i) {
    const loom_target_entry_t* entry = &entries->values[i];
    const loom_target_export_plan_t* export_plan =
        loom_target_entry_bundle(entry)->export_plan;
    const bool declaration = loom_low_func_decl_isa(entry->func.op);
    const bool exported =
        !declaration && loom_func_like_is_exported(entry->func);
    iree_string_view_t name =
        !exported || iree_string_view_is_empty(export_plan->export_symbol)
            ? entry->func_name
            : export_plan->export_symbol;
    if (declaration) {
      const uint8_t import_kind =
          loom_low_func_decl_import_kind(entry->func.op);
      const loom_string_id_t import_module =
          loom_func_like_import_module(entry->func);
      if ((import_kind &&
           import_kind != LOOM_LOW_FUNC_DECL_IMPORT_KIND_NATIVE &&
           import_kind != LOOM_LOW_FUNC_DECL_IMPORT_KIND_OBJECT) ||
          (import_module != LOOM_STRING_ID_INVALID &&
           !iree_string_view_is_empty(
               loom_string_table_get(&module->strings, import_module)))) {
        status = loom_target_artifact_emit_requirement(
            emitter, entry->func.op, IREE_SV("native"),
            IREE_SV("unqualified native or object imports"));
        continue;
      }
      const loom_string_id_t import_symbol =
          loom_func_like_import_symbol(entry->func);
      if (import_symbol != LOOM_STRING_ID_INVALID) {
        name = loom_string_table_get(&module->strings, import_symbol);
      }
    }
    if (exported) {
      names[export_count++] = &symbols[i];
    }
    if (iree_string_view_is_empty(name) ||
        iree_string_view_find_char(name, '\0', 0) != IREE_STRING_VIEW_NPOS) {
      status = loom_target_artifact_emit_requirement(
          emitter, entry->func.op, IREE_SV("native"),
          IREE_SV("nonempty symbol names without NUL bytes"));
    }
    symbols[i] = (loom_native_object_symbol_t){
        .name = name,
        .section_contribution_index =
            declaration ? IREE_HOST_SIZE_MAX : section_count++,
        .binding = exported || declaration
                       ? LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL
                       : LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL,
        .visibility = LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
        .kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
    };
  }
  if (!iree_status_is_ok(status)) {
    return status;
  }
  *out_section_count = section_count;
  qsort(names, export_count, sizeof(*names), loom_native_module_compare_names);
  for (uint16_t i = 1;
       i < export_count && diagnostics->error_count < max_errors &&
       iree_status_is_ok(status);
       ++i) {
    if (iree_string_view_equal(names[i - 1]->name, names[i]->name)) {
      status = loom_target_artifact_emit_duplicate_export(
          emitter, entries->values[names[i] - symbols].func.op,
          entries->values[names[i - 1] - symbols].func.op, IREE_SV("native"),
          names[i]->name);
    }
  }
  return status;
}

// Retain the data declarations while constructing the module symbol map. Later
// contribution construction consumes these records without rescanning the IR.
static iree_status_t loom_native_module_collect_rodata(
    const loom_module_t* module, iree_host_size_t first_symbol_index,
    uint32_t* symbol_indices, iree_arena_allocator_t* arena,
    const loom_symbol_t*** out_symbols, iree_host_size_t* out_count) {
  *out_symbols = NULL;
  *out_count = 0;
  iree_host_size_t capacity = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < module->symbols.count && iree_status_is_ok(status); ++i) {
    symbol_indices[i] = UINT32_MAX;
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (!symbol->defining_op ||
        !loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_RODATA)) {
      continue;
    }
    status = iree_arena_grow_array(arena, *out_count, *out_count + 1,
                                   sizeof(**out_symbols), &capacity,
                                   (void**)out_symbols);
    if (iree_status_is_ok(status)) {
      symbol_indices[i] = (uint32_t)(first_symbol_index + *out_count);
      (*out_symbols)[(*out_count)++] = symbol;
    }
  }
  return status;
}

static iree_status_t loom_native_module_rodata(
    const loom_module_t* module, const loom_symbol_t* source,
    iree_diagnostic_emitter_t emitter, loom_native_object_symbol_t* symbol,
    loom_native_section_contribution_t* sections,
    iree_host_size_t* section_count) {
  const loom_op_t* op = source->defining_op;
  const bool declaration = loom_global_rodata_decl_isa(op);
  const iree_string_view_t name =
      loom_string_table_get(&module->strings, source->name_id);
  if (iree_string_view_is_empty(name) ||
      iree_string_view_find_char(name, '\0', 0) != IREE_STRING_VIEW_NPOS) {
    return loom_target_artifact_emit_requirement(
        emitter, op, IREE_SV("native"),
        IREE_SV("nonempty symbol names without NUL bytes"));
  }
  *symbol = (loom_native_object_symbol_t){
      .name = name,
      .section_contribution_index =
          declaration ? IREE_HOST_SIZE_MAX : *section_count,
      .binding = declaration ? LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL
                             : LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL,
      .visibility = LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
      .kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_DATA,
  };
  if (declaration) {
    return iree_ok_status();
  }
  if (loom_global_rodata_def_has_bank_conflicts(op)) {
    return loom_target_artifact_emit_requirement(
        emitter, op, IREE_SV("native"),
        IREE_SV("readonly data without bank placement constraints"));
  }
  const iree_const_byte_span_t contents = loom_global_rodata_def_contents(op);
  symbol->size = contents.data_length;
  sections[(*section_count)++] = (loom_native_section_contribution_t){
      .section_name = IREE_SV(".rodata"),
      .storage = LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
      .access = LOOM_NATIVE_SECTION_ACCESS_READ,
      .contribution_alignment = loom_global_rodata_def_has_alignment(op)
                                    ? loom_global_rodata_def_alignment(op)
                                    : 1,
      .contents = contents,
  };
  return iree_ok_status();
}

iree_status_t loom_native_module_check_calls(
    const loom_target_emit_request_t* request,
    const loom_low_schedule_table_t* schedule, const uint32_t* symbol_indices,
    bool* out_accepted) {
  *out_accepted = true;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < schedule->call_node_count && *out_accepted &&
                               iree_status_is_ok(status);
       ++i) {
    const loom_op_t* call = schedule->nodes[schedule->call_node_indices[i]].op;
    const loom_symbol_ref_t callee = loom_low_func_call_callee(call);
    if (symbol_indices[callee.symbol_id] == UINT32_MAX) {
      const iree_string_view_t name = loom_string_table_get(
          &request->module->strings,
          request->module->symbols.entries[callee.symbol_id].name_id);
      *out_accepted = false;
      status = loom_target_artifact_emit_symbol_requirement(
          request->diagnostic_emitter, call, IREE_SV("native"), name,
          IREE_SV("a native object binding"));
    }
  }
  return status;
}

// Keep optional report storage off the ordinary emission stack even under LTO.
IREE_ATTRIBUTE_NOINLINE static iree_status_t loom_native_module_report_function(
    const loom_target_emit_request_t* request, const loom_target_entry_t* entry,
    const loom_native_module_elf_target_t* target, const void* target_context,
    const uint32_t* symbol_indices, iree_string_view_t symbol_name,
    iree_host_size_t section_index, loom_native_module_fixups_t* fixups,
    iree_arena_allocator_t* function_arena, iree_io_stream_t* stream,
    bool* out_accepted) {
  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(request->allocator, &report);
  report.requested_detail_flags =
      request->compile_report->requested_detail_flags;
  report.lowered_symbol = entry->func_name;
  loom_target_compile_report_record_target_bundle(
      &report, loom_target_entry_bundle(entry));
  loom_target_emit_request_t entry_request = *request;
  entry_request.compile_report = &report;
  iree_status_t status = target->emit_function(
      &entry_request, entry, target_context, symbol_indices, section_index,
      fixups, function_arena, stream, out_accepted);
  report.function_name = symbol_name;
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_record_entry_report(
        request->compile_report, &report);
  }
  loom_target_compile_report_deinitialize(&report);
  return status;
}

static iree_status_t loom_native_module_function(
    const loom_target_emit_request_t* module_request,
    const loom_target_entry_t* entry,
    const loom_native_module_elf_target_t* target, const void* target_context,
    const uint32_t* symbol_indices, iree_string_view_t symbol_name,
    iree_host_size_t section_index, loom_native_module_fixups_t* fixups,
    bool* out_accepted, loom_native_section_contribution_t* out_section) {
  *out_section = (loom_native_section_contribution_t){0};
  loom_diagnostic_export_scope_t diagnostic_scope = {
      .export_name = loom_target_facts_export_name(entry->target_facts),
      .target = module_request->diagnostic_emitter,
  };
  loom_target_emit_request_t entry_request = *module_request;
  entry_request.diagnostic_emitter =
      loom_diagnostic_export_scope_emitter(&diagnostic_scope);
  const loom_target_emit_request_t* request = &entry_request;
  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      4096, request->allocator, &stream));
  // Reports copy compact rows before function scratch retires. The schedule,
  // allocation, and instruction storage stays bounded by the largest function.
  iree_arena_allocator_t function_arena;
  iree_arena_initialize(request->scratch_arena->block_pool, &function_arena);
  iree_status_t status;
  if (request->compile_report) {
    status = loom_native_module_report_function(
        request, entry, target, target_context, symbol_indices, symbol_name,
        section_index, fixups, &function_arena, stream, out_accepted);
  } else {
    status = target->emit_function(request, entry, target_context,
                                   symbol_indices, section_index, fixups,
                                   &function_arena, stream, out_accepted);
  }
  iree_arena_deinitialize(&function_arena);
  const iree_host_size_t length =
      (iree_host_size_t)iree_io_stream_length(stream);
  uint8_t* contents = NULL;
  if (iree_status_is_ok(status) && *out_accepted) {
    status =
        iree_arena_allocate(request->scratch_arena, length, (void**)&contents);
  }
  if (iree_status_is_ok(status) && *out_accepted) {
    status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0);
  }
  if (iree_status_is_ok(status) && *out_accepted) {
    status = iree_io_stream_read(stream, length, contents, NULL);
  }
  if (iree_status_is_ok(status) && *out_accepted) {
    *out_section = (loom_native_section_contribution_t){
        .section_name = IREE_SV(".text"),
        .storage = LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
        .access = LOOM_NATIVE_SECTION_ACCESS_READ |
                  LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
        .contribution_alignment = target->function_alignment,
        .contents = iree_make_const_byte_span(contents, length),
    };
  }
  iree_io_stream_release(stream);
  return status;
}

static iree_status_t loom_native_module_build_artifact(
    const loom_target_emit_request_t* source_request,
    const loom_target_fact_type_t* target_fact_type,
    loom_native_module_format_t format,
    const loom_native_module_elf_target_t* target, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact) {
  loom_target_entry_diagnostic_emitter_t diagnostics = {
      .forwarding_emitter = source_request->diagnostic_emitter,
  };
  loom_target_emit_request_t counted_request = *source_request;
  counted_request.diagnostic_emitter = loom_target_entry_emitter(&diagnostics);
  const loom_target_emit_request_t* request = &counted_request;
  *out_emitted = false;
  *out_artifact = (loom_target_emit_artifact_t){0};
  if (request->artifact_manifest.mode !=
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return loom_target_artifact_emit_requirement(
        request->diagnostic_emitter, NULL, IREE_SV("native"),
        IREE_SV("an artifact request without a manifest"));
  }
  const loom_target_entry_options_t options = {
      .flags = LOOM_TARGET_ENTRY_SELECTION_INCLUDE_PRIVATE |
               LOOM_TARGET_ENTRY_SELECTION_INCLUDE_DECLARATIONS,
      .function_versions = request->function_versions,
      .max_errors = request->max_errors,
  };
  const uint32_t max_errors = loom_target_entry_max_errors(&options, 20);
  loom_target_entry_list_t entries = {0};
  bool accepted = false;
  loom_native_module_selection_t selection = {
      .fact_type = target_fact_type,
      .format = format,
  };
  IREE_RETURN_IF_ERROR(loom_target_entry_select_all_entries(
      request->module, &options,
      (loom_target_entry_predicate_t){
          .fn = loom_native_module_accept_entry,
          .user_data = &selection,
      },
      &diagnostics, IREE_SV("native artifact"), request->scratch_arena,
      &accepted, &entries));
  if (!accepted || diagnostics.error_count) {
    return iree_ok_status();
  }
  if (request->compile_report) {
    request->compile_report->target_family_name = target_fact_type->name;
  }
  void* target_context = NULL;
  IREE_RETURN_IF_ERROR(target->prepare_module(
      request, &entries, request->scratch_arena, &accepted, &target_context));
  if (!accepted || diagnostics.error_count) {
    return iree_ok_status();
  }
  const loom_target_bundle_t* artifact_bundle =
      format == LOOM_NATIVE_MODULE_FORMAT_HAL_LIBRARY
          ? NULL
          : loom_target_entry_bundle(&entries.values[0]);
  uint32_t* symbol_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      request->scratch_arena, request->module->symbols.count,
      sizeof(*symbol_indices), (void**)&symbol_indices));
  const loom_symbol_t** rodata_symbols = NULL;
  iree_host_size_t rodata_count = 0;
  IREE_RETURN_IF_ERROR(loom_native_module_collect_rodata(
      request->module, entries.count, symbol_indices, request->scratch_arena,
      &rodata_symbols, &rodata_count));
  const iree_host_size_t symbol_count = entries.count + rodata_count;
  loom_native_section_contribution_t* sections = NULL;
  loom_native_object_symbol_t* symbols = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(request->scratch_arena, symbol_count,
                                sizeof(*sections), (void**)&sections));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(request->scratch_arena,
                                                 symbol_count, sizeof(*symbols),
                                                 (void**)&symbols));
  iree_host_size_t section_count = 0;
  IREE_RETURN_IF_ERROR(loom_native_module_symbols(
      request->module, &entries, &diagnostics, max_errors,
      request->scratch_arena, symbols, &section_count));
  if (diagnostics.error_count) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < entries.count; ++i) {
    symbol_indices[entries.values[i].func_ref.symbol_id] = i;
  }
  loom_native_module_fixups_t fixups = {0};
  iree_host_size_t library_symbol_index = IREE_HOST_SIZE_MAX;
  loom_native_task_library_entry_t* library_entries = NULL;
  uint16_t library_entry_count = 0;
  iree_host_size_t library_entry_capacity = 0;
  bool has_library_query = false;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < rodata_count && diagnostics.error_count < max_errors &&
       iree_status_is_ok(status);
       ++i) {
    const uint32_t prior_errors = diagnostics.error_count;
    status = loom_native_module_rodata(
        request->module, rodata_symbols[i], request->diagnostic_emitter,
        &symbols[entries.count + i], sections, &section_count);
    if (iree_status_is_ok(status) && diagnostics.error_count == prior_errors &&
        format == LOOM_NATIVE_MODULE_FORMAT_HAL_LIBRARY &&
        iree_string_view_equal(symbols[entries.count + i].name,
                               IREE_SV(LOOM_TASK_LIBRARY_SYMBOL))) {
      if (!loom_global_rodata_decl_isa(rodata_symbols[i]->defining_op)) {
        status = loom_target_artifact_emit_requirement(
            request->diagnostic_emitter, rodata_symbols[i]->defining_op,
            IREE_SV("native"),
            IREE_SV(
                "a readonly data declaration for the task HAL library symbol"));
      } else {
        library_symbol_index = entries.count + i;
      }
    }
  }
  IREE_RETURN_IF_ERROR(status);
  if (diagnostics.error_count) {
    return iree_ok_status();
  }
  for (uint16_t i = 0;
       i < entries.count && diagnostics.error_count < max_errors &&
       iree_status_is_ok(status);
       ++i) {
    accepted = true;
    const bool exported = loom_func_like_is_exported(entries.values[i].func);
    has_library_query |=
        format == LOOM_NATIVE_MODULE_FORMAT_HAL_LIBRARY && exported &&
        iree_string_view_equal(
            symbols[i].name, IREE_SV(IREE_HAL_EXECUTABLE_LIBRARY_EXPORT_NAME));
    if (format == LOOM_NATIVE_MODULE_FORMAT_HAL_LIBRARY && exported &&
        loom_target_entry_bundle(&entries.values[i])->export_plan->abi_kind ==
            LOOM_TARGET_ABI_HAL_KERNEL) {
      if (library_entry_count == 0) {
        artifact_bundle = loom_target_entry_bundle(&entries.values[i]);
      }
      status = iree_arena_grow_array(
          request->scratch_arena, library_entry_count, library_entry_count + 1,
          sizeof(*library_entries), &library_entry_capacity,
          (void**)&library_entries);
      if (iree_status_is_ok(status)) {
        library_entries[library_entry_count] =
            (loom_native_task_library_entry_t){
                .name = symbols[i].name,
                .symbol_index = i,
            };
        if (!loom_low_func_def_isa(entries.values[i].func.op)) {
          accepted = false;
          status = loom_target_artifact_emit_requirement(
              request->diagnostic_emitter, entries.values[i].func.op,
              IREE_SV("native"),
              IREE_SV(
                  "a physical task function with a logical parameter layout"));
        } else {
          status = loom_task_parameter_layout_parse(
              request->module, entries.values[i].func.op,
              loom_low_func_def_abi_layout(entries.values[i].func.op),
              request->diagnostic_emitter, request->scratch_arena, &accepted,
              &library_entries[library_entry_count].abi);
        }
      }
      if (iree_status_is_ok(status) && accepted) {
        ++library_entry_count;
      }
    }
    if (!iree_status_is_ok(status) || !accepted) {
      continue;
    }
    if (!loom_low_func_decl_isa(entries.values[i].func.op)) {
      const iree_host_size_t section_index =
          symbols[i].section_contribution_index;
      const iree_host_size_t first_fixup = fixups.count;
      status = loom_native_module_function(
          request, &entries.values[i], target, target_context, symbol_indices,
          symbols[i].name, section_index, &fixups, &accepted,
          &sections[section_index]);
      symbols[i].size = sections[section_index].contents.data_length;
      // The prepared fixups are the exact referenced symbols. Unused imports
      // remain legal; a self-contained image requires definitions only for
      // references that survived compilation. The HAL library declaration is
      // filled by the library builder below.
      for (iree_host_size_t j = first_fixup;
           format != LOOM_NATIVE_MODULE_FORMAT_OBJECT && j < fixups.count &&
           diagnostics.error_count < max_errors && iree_status_is_ok(status);
           ++j) {
        const iree_host_size_t target_index =
            fixups.values[j].target_symbol_index;
        if (symbols[target_index].section_contribution_index ==
                IREE_HOST_SIZE_MAX &&
            target_index != library_symbol_index) {
          status = loom_target_artifact_emit_symbol_requirement(
              request->diagnostic_emitter, entries.values[i].func.op,
              IREE_SV("native"), symbols[target_index].name,
              IREE_SV("a definition in this image; use a relocatable object "
                      "for external linking"));
        }
      }
    }
  }
  accepted = diagnostics.error_count == 0;
  if (iree_status_is_ok(status) && accepted &&
      format == LOOM_NATIVE_MODULE_FORMAT_HAL_LIBRARY) {
    if (!library_entry_count || !has_library_query ||
        library_symbol_index == IREE_HOST_SIZE_MAX) {
      accepted = false;
      status = loom_target_artifact_emit_requirement(
          request->diagnostic_emitter, entries.values[0].func.op,
          IREE_SV("native"),
          IREE_SV("task dispatch entries, the library query, and its readonly "
                  "library declaration"));
    } else {
      loom_native_task_library_data_t data;
      status = loom_native_task_library_build_64le(
          request->identifier, library_entries, library_entry_count,
          library_symbol_index, section_count, target->pointer_relocation_kind,
          request->scratch_arena, &data);
      if (iree_status_is_ok(status)) {
        status = iree_arena_grow_array(request->scratch_arena, fixups.count,
                                       fixups.count + data.fixup_count,
                                       sizeof(*fixups.values), &fixups.capacity,
                                       (void**)&fixups.values);
      }
      if (iree_status_is_ok(status)) {
        memcpy(fixups.values + fixups.count, data.fixups,
               data.fixup_count * sizeof(*fixups.values));
        fixups.count += data.fixup_count;
        loom_native_object_symbol_t* symbol = &symbols[library_symbol_index];
        symbol->section_contribution_index = section_count;
        symbol->binding = LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL;
        symbol->size = data.section.contents.data_length;
        sections[section_count++] = data.section;
      }
    }
  }
  iree_io_stream_t* stream = NULL;
  if (iree_status_is_ok(status) && accepted) {
    status = iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_RESIZABLE, 4096,
        request->allocator, &stream);
  }
  if (iree_status_is_ok(status) && accepted) {
    const loom_native_object_contribution_t object = {
        .sections = sections,
        .section_count = section_count,
        .symbols = symbols,
        .symbol_count = symbol_count,
        .fixups = fixups.values,
        .fixup_count = fixups.count,
    };
    if (format != LOOM_NATIVE_MODULE_FORMAT_OBJECT) {
      status = loom_native_image_write_elf64le(&object, &target->image_options,
                                               stream, request->scratch_arena);
    } else {
      status = loom_native_object_write_elf64le(
          &object, target->image_options.machine,
          target->image_options.relocations, stream, request->scratch_arena);
    }
  }
  if (iree_status_is_ok(status) && accepted) {
    status = iree_io_vec_stream_move_contents(stream, &out_artifact->contents);
  }
  if (iree_status_is_ok(status) && accepted &&
      iree_any_bit_set(request->flags,
                       LOOM_TARGET_EMIT_REQUEST_FLAG_RETAIN_TARGET_BUNDLE)) {
    status = loom_target_emit_artifact_retain_metadata(
        artifact_bundle, 0, NULL, request->allocator, out_artifact);
  }
  if (iree_status_is_ok(status) && accepted) {
    out_artifact->target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
    *out_emitted = true;
  }
  iree_io_stream_release(stream);
  return status;
}

iree_status_t loom_native_module_emit_elf64le(
    const loom_target_emit_request_t* request,
    const loom_target_fact_type_t* target_fact_type,
    loom_native_module_format_t format,
    const loom_native_module_elf_target_t* target, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact) {
  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(request->scratch_arena);
  iree_status_t status = loom_native_module_build_artifact(
      request, target_fact_type, format, target, out_emitted, out_artifact);
  if (!iree_status_is_ok(status)) {
    loom_target_emit_artifact_release(out_artifact);
  }
  iree_arena_checkpoint_restore(&checkpoint);
  return status;
}
