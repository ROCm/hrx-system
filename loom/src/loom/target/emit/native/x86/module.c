// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/module.h"

#include <stdlib.h>

#include "iree/io/vec_stream.h"
#include "loom/target/emit/native/object_elf.h"
#include "loom/target/emit/native/x86/abi.h"
#include "loom/target/emit/native/x86/function.h"

static bool loom_x86_module_accept_entry(void* user_data,
                                         const loom_target_entry_t* entry) {
  const loom_target_bundle_t* bundle = loom_target_entry_bundle(entry);
  return entry->target_facts->fact_type == user_data &&
         bundle->export_plan->abi_kind == LOOM_TARGET_ABI_OBJECT_FUNCTION &&
         bundle->snapshot->artifact_format == LOOM_TARGET_ARTIFACT_FORMAT_ELF;
}

static int loom_x86_module_compare_names(const void* lhs, const void* rhs) {
  return iree_string_view_compare(*(const iree_string_view_t*)lhs,
                                  *(const iree_string_view_t*)rhs);
}

// Export names are authored input. Validate the final native namespace before
// encoding; the object writer consumes trusted symbol records.
static iree_status_t loom_x86_module_symbols(
    const loom_target_entry_list_t* entries, iree_arena_allocator_t* arena,
    loom_native_object_symbol_t* symbols) {
  iree_string_view_t* names = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, entries->count, sizeof(*names), (void**)&names));
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < entries->count && iree_status_is_ok(status); ++i) {
    const loom_target_entry_t* entry = &entries->values[i];
    const loom_target_export_plan_t* export_plan =
        loom_target_entry_bundle(entry)->export_plan;
    names[i] = iree_string_view_is_empty(export_plan->export_symbol)
                   ? entry->func_name
                   : export_plan->export_symbol;
    if (iree_string_view_find_char(names[i], '\0', 0) !=
        IREE_STRING_VIEW_NPOS) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "native symbol names cannot contain NUL");
    }
    symbols[i] = (loom_native_object_symbol_t){
        .name = names[i],
        .section_contribution_index = i,
        .binding = LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
        .visibility = LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
        .kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
    };
  }
  if (!iree_status_is_ok(status)) {
    return status;
  }
  qsort(names, entries->count, sizeof(*names), loom_x86_module_compare_names);
  for (uint16_t i = 1; i < entries->count && iree_status_is_ok(status); ++i) {
    if (iree_string_view_equal(names[i - 1], names[i])) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate native export '%.*s'",
                                (int)names[i].size, names[i].data);
    }
  }
  return status;
}

static iree_status_t loom_x86_module_encode_function(
    const loom_target_emit_request_t* request, const loom_target_entry_t* entry,
    iree_io_stream_t* stream, bool* out_accepted) {
  loom_x86_function_abi_t abi;
  IREE_RETURN_IF_ERROR(loom_x86_function_abi_prepare(
      request->module, entry, request->diagnostic_emitter, out_accepted, &abi));
  if (!*out_accepted) {
    return iree_ok_status();
  }
  const loom_low_allocation_reserved_range_t stack_pointer = {
      .register_class = IREE_SV("x86.gpr64"),
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location_base = 4,
      .location_count = 1,
  };
  const loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = request->low_descriptor_registry,
      .function_target_facts = entry->target_facts,
      .memory_accesses = entry->function_version->memory_accesses,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
      .allocation_fixed_values = abi.fixed_values,
      .allocation_fixed_value_count = abi.fixed_value_count,
      .allocation_reserved_ranges = &stack_pointer,
      .allocation_reserved_range_count = 1,
      .emitter = request->diagnostic_emitter,
  };
  const loom_low_emission_frame_spill_free_options_t spill_options = {0};
  loom_low_emission_frame_t frame = {0};
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_build_spill_free(
      request->module, (loom_op_t*)entry->func.op, &frame_options,
      &spill_options, request->scratch_arena, &frame, out_accepted));
  if (!*out_accepted) {
    return iree_ok_status();
  }
  loom_x86_function_t function;
  IREE_RETURN_IF_ERROR(
      loom_x86_function_prepare(&frame, request->scratch_arena, &function));
  return loom_x86_function_write(&function, stream, request->scratch_arena);
}

static iree_status_t loom_x86_module_function(
    const loom_target_emit_request_t* request, const loom_target_entry_t* entry,
    bool* out_accepted, loom_native_section_contribution_t* out_section) {
  *out_section = (loom_native_section_contribution_t){0};
  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      4096, request->allocator, &stream));
  // Retain only emitted bytes between functions. Schedule, allocation, and
  // instruction preparation storage is bounded by the largest function.
  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(request->scratch_arena);
  iree_status_t status =
      loom_x86_module_encode_function(request, entry, stream, out_accepted);
  iree_arena_checkpoint_restore(&checkpoint);
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
        .contribution_alignment = 16,
        .contents = iree_make_const_byte_span(contents, length),
    };
  }
  iree_io_stream_release(stream);
  return status;
}

static iree_status_t loom_x86_module_build_object(
    const loom_target_emit_request_t* request,
    const loom_target_fact_type_t* target_fact_type, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact) {
  *out_emitted = false;
  *out_artifact = (loom_target_emit_artifact_t){0};
  if (request->artifact_manifest.mode !=
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "x86 objects do not support artifact manifests");
  }
  const loom_target_entry_options_t options = {
      .function_versions = request->function_versions,
  };
  loom_target_entry_diagnostic_emitter_t diagnostics = {
      .forwarding_emitter = request->diagnostic_emitter,
  };
  loom_target_entry_list_t entries = {0};
  bool accepted = false;
  IREE_RETURN_IF_ERROR(loom_target_entry_select_all_entries(
      request->module, &options,
      (loom_target_entry_predicate_t){
          .fn = loom_x86_module_accept_entry,
          .user_data = (void*)target_fact_type,
      },
      &diagnostics, IREE_SV("x86 native object"), request->scratch_arena,
      &accepted, &entries));
  if (!accepted) {
    return iree_ok_status();
  }
  loom_native_section_contribution_t* sections = NULL;
  loom_native_object_symbol_t* symbols = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(request->scratch_arena, entries.count,
                                sizeof(*sections), (void**)&sections));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(request->scratch_arena, entries.count,
                                sizeof(*symbols), (void**)&symbols));
  IREE_RETURN_IF_ERROR(
      loom_x86_module_symbols(&entries, request->scratch_arena, symbols));
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0;
       i < entries.count && accepted && iree_status_is_ok(status); ++i) {
    status = loom_x86_module_function(request, &entries.values[i], &accepted,
                                      &sections[i]);
    symbols[i].size = sections[i].contents.data_length;
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
        .section_count = entries.count,
        .symbols = symbols,
        .symbol_count = entries.count,
    };
    status = loom_native_object_write_elf64le(&object,
                                              LOOM_NATIVE_ELF_MACHINE_X86_64,
                                              stream, request->scratch_arena);
  }
  if (iree_status_is_ok(status) && accepted) {
    status = iree_io_vec_stream_move_contents(stream, &out_artifact->contents);
  }
  if (iree_status_is_ok(status) && accepted) {
    out_artifact->target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
    *out_emitted = true;
  }
  iree_io_stream_release(stream);
  return status;
}

iree_status_t loom_x86_module_emit(
    const loom_target_emit_request_t* request,
    const loom_target_fact_type_t* target_fact_type, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact) {
  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(request->scratch_arena);
  iree_status_t status = loom_x86_module_build_object(
      request, target_fact_type, out_emitted, out_artifact);
  iree_arena_checkpoint_restore(&checkpoint);
  return status;
}
