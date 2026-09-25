// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/module.h"

#include <stdlib.h>

#include "iree/io/vec_stream.h"
#include "iree/vm/bytecode/wire/core.h"
#include "iree/vm/bytecode/wire/module.h"
#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/vm/function.h"
#include "loom/target/arch/vm/types.h"
#include "loom/target/function_version.h"

// A signature is ordered by argument count/types then result
// count/types, exactly as the wire callable table requires. Ordinals are
// assigned by sorting once; the runtime performs no hashing or interning.
static int loom_vm_signature_compare(const void* lhs_ptr, const void* rhs_ptr) {
  const loom_vm_module_callable_t* lhs =
      *(const loom_vm_module_callable_t* const*)lhs_ptr;
  const loom_vm_module_callable_t* rhs =
      *(const loom_vm_module_callable_t* const*)rhs_ptr;
  int comparison = (int)lhs->argument_count - (int)rhs->argument_count;
  if (comparison) {
    return comparison;
  }
  for (uint16_t i = 0; i < lhs->argument_count; ++i) {
    comparison = (int)lhs->signature.fields[i].kind_u16 -
                 (int)rhs->signature.fields[i].kind_u16;
    if (comparison) {
      return comparison;
    }
    comparison = (int)lhs->signature.fields[i].type_ordinal_u16 -
                 (int)rhs->signature.fields[i].type_ordinal_u16;
    if (comparison) {
      return comparison;
    }
  }
  comparison = (int)lhs->results.count - (int)rhs->results.count;
  if (comparison) {
    return comparison;
  }
  for (uint16_t i = 0; i < lhs->results.count; ++i) {
    comparison = (int)lhs->signature.fields[lhs->argument_count + i].kind_u16 -
                 (int)rhs->signature.fields[rhs->argument_count + i].kind_u16;
    if (comparison) {
      return comparison;
    }
    comparison =
        (int)lhs->signature.fields[lhs->argument_count + i].type_ordinal_u16 -
        (int)rhs->signature.fields[rhs->argument_count + i].type_ordinal_u16;
    if (comparison) {
      return comparison;
    }
  }
  return 0;
}

static int loom_vm_export_compare(const void* lhs_ptr, const void* rhs_ptr) {
  const loom_vm_module_callable_t* lhs =
      *(const loom_vm_module_callable_t* const*)lhs_ptr;
  const loom_vm_module_callable_t* rhs =
      *(const loom_vm_module_callable_t* const*)rhs_ptr;
  return iree_string_view_compare(lhs->export_name, rhs->export_name);
}

// Import rows are ordered by module, symbol, then structural callable type.
// Equal rows share one runtime binding even when authored under several
// aliases.
static int loom_vm_import_compare(const void* lhs_ptr, const void* rhs_ptr) {
  const loom_vm_module_callable_t* lhs =
      *(const loom_vm_module_callable_t* const*)lhs_ptr;
  const loom_vm_module_callable_t* rhs =
      *(const loom_vm_module_callable_t* const*)rhs_ptr;
  int comparison = iree_string_view_compare(lhs->import.module_name,
                                            rhs->import.module_name);
  if (comparison) {
    return comparison;
  }
  comparison = iree_string_view_compare(lhs->import.symbol_name,
                                        rhs->import.symbol_name);
  if (comparison) {
    return comparison;
  }
  return (int)lhs->callable_ordinal - (int)rhs->callable_ordinal;
}

static iree_status_t loom_vm_signature_type(loom_type_t type,
                                            uint16_t* out_kind) {
  // Logical scalar tags are stable, small, and independent of cell width.
  // Predicates cross the ABI as canonical zero/one i32 values.
  static const uint8_t kScalarKinds[LOOM_SCALAR_TYPE_COUNT_] = {
      [LOOM_SCALAR_TYPE_I1] = IREE_VM_BYTECODE_SIGNATURE_KIND_I32,
      [LOOM_SCALAR_TYPE_I8] = IREE_VM_BYTECODE_SIGNATURE_KIND_I8,
      [LOOM_SCALAR_TYPE_I16] = IREE_VM_BYTECODE_SIGNATURE_KIND_I16,
      [LOOM_SCALAR_TYPE_I32] = IREE_VM_BYTECODE_SIGNATURE_KIND_I32,
      [LOOM_SCALAR_TYPE_I64] = IREE_VM_BYTECODE_SIGNATURE_KIND_I64,
      [LOOM_SCALAR_TYPE_F8E4M3] = IREE_VM_BYTECODE_SIGNATURE_KIND_F8E4M3FN,
      [LOOM_SCALAR_TYPE_F8E5M2] = IREE_VM_BYTECODE_SIGNATURE_KIND_F8E5M2,
      [LOOM_SCALAR_TYPE_F16] = IREE_VM_BYTECODE_SIGNATURE_KIND_F16,
      [LOOM_SCALAR_TYPE_BF16] = IREE_VM_BYTECODE_SIGNATURE_KIND_BF16,
      [LOOM_SCALAR_TYPE_F32] = IREE_VM_BYTECODE_SIGNATURE_KIND_F32,
      [LOOM_SCALAR_TYPE_F64] = IREE_VM_BYTECODE_SIGNATURE_KIND_F64,
  };
  const loom_type_t* value_type = loom_type_register_value_type(type);
  if (value_type) {
    if (loom_type_is_buffer(*value_type) ||
        (loom_type_is_dialect(*value_type) &&
         loom_type_dialect_param_count(*value_type) == 0)) {
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_REF;
      return iree_ok_status();
    }
    const uint8_t kind = loom_type_is_scalar(*value_type)
                             ? kScalarKinds[loom_type_element_type(*value_type)]
                             : 0;
    if (kind) {
      *out_kind = kind;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "VM signature requires a supported typed register");
}

// Only top-level symbol definitions are collected. Callgraph specialization
// and library composition belong to the shared compiler, not this writer.
static iree_status_t loom_vm_module_collect(
    const loom_target_emit_request_t* request,
    loom_vm_module_plan_t* out_functions) {
  const loom_module_t* module = request->module;
  loom_target_function_version_snapshot_t versions = {0};
  IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
      module, request->function_versions, request->scratch_arena, &versions));
  loom_vm_module_callable_t* functions = NULL;
  iree_host_size_t storage_size = 0;
  iree_host_size_t bindings_offset = 0;
  iree_host_size_t ordinals_offset = 0;
  iree_host_size_t rodata_offset = 0;
  iree_host_size_t rodata_symbols_offset = 0;
  // Pointer-bearing arrays precede the two-byte symbol and ordinal arrays so
  // each field keeps its native alignment without inter-array padding.
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &storage_size,
      IREE_STRUCT_FIELD(module->symbols.count, loom_vm_module_callable_t, NULL),
      IREE_STRUCT_FIELD(module->symbols.count, loom_vm_module_callable_t*,
                        &bindings_offset),
      IREE_STRUCT_FIELD(module->symbols.count, const loom_op_t*,
                        &rodata_offset),
      IREE_STRUCT_FIELD(module->symbols.count, uint16_t, &ordinals_offset),
      IREE_STRUCT_FIELD(module->symbols.count, loom_symbol_id_t,
                        &rodata_symbols_offset)));
  IREE_RETURN_IF_ERROR(iree_arena_allocate(request->scratch_arena, storage_size,
                                           (void**)&functions));
  loom_vm_module_callable_t** bindings_by_symbol =
      (loom_vm_module_callable_t**)((uint8_t*)functions + bindings_offset);
  uint16_t* ordinals_by_symbol =
      (uint16_t*)((uint8_t*)functions + ordinals_offset);
  uint32_t count = 0;
  uint32_t definition_count = 0;
  const loom_op_t** rodata =
      (const loom_op_t**)((uint8_t*)functions + rodata_offset);
  loom_symbol_id_t* rodata_symbols =
      (loom_symbol_id_t*)((uint8_t*)functions + rodata_symbols_offset);
  uint32_t rodata_count = 0;
  iree_host_size_t descriptor_count = 0;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < module->symbols.count && iree_status_is_ok(status);
       ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    loom_op_t* op = symbol->defining_op;
    bindings_by_symbol[i] = NULL;
    ordinals_by_symbol[i] = UINT16_MAX;
    if (loom_global_rodata_def_isa(op)) {
      rodata_symbols[rodata_count++] = (loom_symbol_id_t)i;
      continue;
    }
    if (!loom_low_func_def_isa(op) && !loom_low_func_decl_isa(op)) {
      continue;
    }
    // Unresolved IR declarations have no executable binding. A call to one is
    // diagnosed when the function's call schedule is validated.
    if (loom_low_func_decl_isa(op) && !loom_low_func_decl_has_import_kind(op)) {
      continue;
    }
    loom_func_like_t function = loom_func_like_cast(module, op);
    const loom_string_id_t contract = loom_func_like_repr_contract(function);
    if (contract == LOOM_STRING_ID_INVALID ||
        !iree_string_view_equal(
            loom_string_table_get(&module->strings, contract),
            IREE_SV("vm.core"))) {
      continue;
    }
    loom_vm_module_callable_t* entry = &functions[count];
    bindings_by_symbol[i] = entry;
    *entry = (loom_vm_module_callable_t){
        .function = function,
        .function_version =
            loom_target_function_version_snapshot_at(&versions, i),
        .results = {loom_op_results(op), op->result_count},
    };
    entry->arguments = loom_func_like_arg_ids(function, &entry->argument_count);
    if (loom_low_func_decl_isa(op)) {
      const loom_string_id_t import_module =
          loom_func_like_import_module(function);
      if (loom_low_func_decl_import_kind(op) !=
              LOOM_LOW_FUNC_DECL_IMPORT_KIND_NATIVE ||
          import_module == LOOM_STRING_ID_INVALID) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "VM import requires a native callable with a module namespace");
        continue;
      }
      entry->target_kind = IREE_VM_BYTECODE_CONTROL_CALL_TARGET_REQUIRED_IMPORT;
      entry->import.module_name =
          loom_string_table_get(&module->strings, import_module);
      entry->import.symbol_name = loom_string_table_get(
          &module->strings, loom_func_like_import_symbol(function));
    } else {
      entry->target_kind = IREE_VM_BYTECODE_CONTROL_CALL_TARGET_LOCAL;
      entry->ordinal = (uint16_t)definition_count++;
    }
    const loom_string_id_t export_name = loom_func_like_export_symbol(function);
    if (loom_low_func_def_isa(op) && loom_func_like_is_exported(function)) {
      entry->export_name = loom_string_table_get(
          &module->strings, export_name != LOOM_STRING_ID_INVALID
                                ? export_name
                                : symbol->name_id);
      if (iree_string_view_is_empty(entry->export_name)) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "VM export names must not be empty");
      }
    }
    descriptor_count += entry->argument_count + entry->results.count;
    ++count;
  }
  IREE_RETURN_IF_ERROR(status);
  iree_vm_bytecode_v0_signature_descriptor_row_t* descriptors = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      request->scratch_arena, iree_max(descriptor_count, 1),
      sizeof(*descriptors), (void**)&descriptors));
  for (uint32_t i = 0; i < count; ++i) {
    functions[i].signature.fields = descriptors;
    descriptors += functions[i].argument_count + functions[i].results.count;
  }
  *out_functions = (loom_vm_module_plan_t){
      .values = functions,
      .bindings_by_symbol = bindings_by_symbol,
      .count = count,
      .definition_count = definition_count,
      .rodata = {.ordinals_by_symbol = ordinals_by_symbol,
                 .symbols = rodata_symbols,
                 .symbol_count = rodata_count,
                 .values = rodata,
                 .alignment = IREE_VM_BYTECODE_IMAGE_ALIGNMENT},
  };
  return iree_ok_status();
}

// Signature fields own provisional canonical ordinals while metadata is built.
// No per-occurrence reference records or source-sized lookup arrays are needed.
static iree_status_t loom_vm_module_resolve_signatures(
    const loom_target_emit_request_t* request, loom_vm_module_plan_t* functions,
    loom_vm_reference_plan_t* references) {
  const loom_module_t* module = request->module;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < functions->count && iree_status_is_ok(status); ++i) {
    loom_vm_module_callable_t* entry = &functions->values[i];
    iree_vm_bytecode_v0_signature_descriptor_row_t* descriptors =
        entry->signature.fields;
    const uint32_t field_count = entry->argument_count + entry->results.count;
    for (uint32_t j = 0; j < field_count && iree_status_is_ok(status); ++j) {
      const loom_value_id_t value =
          j < entry->argument_count
              ? entry->arguments[j]
              : entry->results.values[j - entry->argument_count];
      const loom_type_t type = loom_module_value_type(module, value);
      descriptors[j].type_ordinal_u16 = 0;
      status = loom_vm_signature_type(type, &descriptors[j].kind_u16);
      if (iree_status_is_ok(status)) {
        iree_vm_bytecode_v0_signature_row_t* row = &entry->signature.row;
        uint16_t* bank_count;
        if (descriptors[j].kind_u16 == IREE_VM_BYTECODE_SIGNATURE_KIND_REF) {
          status = loom_vm_reference_plan_bind(
              references, module, *loom_type_register_value_type(type),
              request->scratch_arena, &descriptors[j].type_ordinal_u16);
          bank_count = j < entry->argument_count ? &row->argument_ref_count_u16
                                                 : &row->result_ref_count_u16;
        } else {
          bank_count = j < entry->argument_count
                           ? &row->argument_value_count_u16
                           : &row->result_value_count_u16;
        }
        ++(*bank_count);
      }
    }
  }
  if (iree_status_is_ok(status) &&
      loom_vm_reference_plan_finalize(references)) {
    for (uint32_t i = 0; i < functions->count; ++i) {
      loom_vm_module_callable_t* entry = &functions->values[i];
      if (!entry->signature.row.argument_ref_count_u16 &&
          !entry->signature.row.result_ref_count_u16) {
        continue;
      }
      const uint32_t field_count = entry->argument_count + entry->results.count;
      for (uint32_t j = 0; j < field_count; ++j) {
        iree_vm_bytecode_v0_signature_descriptor_row_t* field =
            &entry->signature.fields[j];
        if (field->kind_u16 == IREE_VM_BYTECODE_SIGNATURE_KIND_REF) {
          field->type_ordinal_u16 = loom_vm_reference_plan_ordinal(
              references, field->type_ordinal_u16);
        }
      }
    }
  }
  return status;
}

static iree_status_t loom_vm_section_begin(
    iree_io_stream_t* stream, iree_vm_bytecode_section_type_t type,
    iree_vm_bytecode_v0_section_directory_row_t* row,
    iree_io_stream_pos_t* out_start) {
  const uint8_t zero = 0;
  const iree_io_stream_pos_t padding =
      -iree_io_stream_offset(stream) & (IREE_VM_BYTECODE_IMAGE_ALIGNMENT - 1);
  IREE_RETURN_IF_ERROR(iree_io_stream_fill(stream, padding, &zero, 1));
  *out_start = iree_io_stream_offset(stream);
  *row = (iree_vm_bytecode_v0_section_directory_row_t){
      .section_type_u16 = type,
      .payload_alignment_u32 = IREE_VM_BYTECODE_IMAGE_ALIGNMENT,
  };
  return iree_ok_status();
}

// Patches reserved bytes without changing the append position or requiring
// contiguous stream storage.
static iree_status_t loom_vm_stream_patch(iree_io_stream_t* stream,
                                          iree_io_stream_pos_t offset,
                                          iree_const_byte_span_t contents) {
  const iree_io_stream_pos_t end = iree_io_stream_offset(stream);
  IREE_RETURN_IF_ERROR(
      iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, offset));
  IREE_RETURN_IF_ERROR(
      iree_io_stream_write(stream, contents.data_length, contents.data));
  return iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, end);
}

static iree_status_t loom_vm_module_write_metadata(
    const loom_target_emit_request_t* request, loom_vm_module_plan_t* functions,
    iree_io_stream_t* stream,
    iree_vm_bytecode_v0_section_directory_row_t* directory,
    uint16_t* out_section_count) {
  loom_vm_reference_plan_t references = {0};
  IREE_RETURN_IF_ERROR(
      loom_vm_module_resolve_signatures(request, functions, &references));
  // Sorted views retain direct call bindings and definition order while
  // interning signatures and imports for canonical runtime tables.
  loom_vm_module_callable_t** sorted = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(request->scratch_arena, 3 * functions->count,
                                sizeof(*sorted), (void**)&sorted));
  loom_vm_module_callable_t** exports = sorted + functions->count;
  loom_vm_module_callable_t** imports = exports + functions->count;
  uint32_t export_count = 0;
  uint32_t import_count = 0;
  for (uint32_t i = 0; i < functions->count; ++i) {
    loom_vm_module_callable_t* entry = &functions->values[i];
    sorted[i] = entry;
    if (entry->target_kind ==
        IREE_VM_BYTECODE_CONTROL_CALL_TARGET_REQUIRED_IMPORT) {
      imports[import_count++] = entry;
    } else if (!iree_string_view_is_empty(entry->export_name)) {
      exports[export_count++] = entry;
    }
  }
  qsort(sorted, functions->count, sizeof(*sorted), loom_vm_signature_compare);
  uint32_t callable_count = 0;
  for (uint32_t i = 0; i < functions->count; ++i) {
    loom_vm_module_callable_t* entry = sorted[i];
    if (!callable_count ||
        loom_vm_signature_compare(&sorted[callable_count - 1], &entry)) {
      sorted[callable_count++] = entry;
    }
    entry->callable_ordinal = (uint16_t)(callable_count - 1);
  }
  qsort(imports, import_count, sizeof(*imports), loom_vm_import_compare);
  uint32_t unique_import_count = 0;
  for (uint32_t i = 0; i < import_count; ++i) {
    loom_vm_module_callable_t* entry = imports[i];
    if (!unique_import_count ||
        loom_vm_import_compare(&imports[unique_import_count - 1], &entry)) {
      imports[unique_import_count++] = entry;
    }
    entry->ordinal = (uint16_t)(unique_import_count - 1);
  }
  import_count = unique_import_count;
  qsort(exports, export_count, sizeof(*exports), loom_vm_export_compare);

  iree_string_view_t* strings = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      request->scratch_arena,
      iree_max(1, export_count + references.count + references.group_count +
                      2 * import_count),
      sizeof(*strings), (void**)&strings));
  uint32_t string_count = 0;
  for (uint32_t i = 0; i < export_count; ++i) {
    if (i && iree_string_view_equal(exports[i - 1]->export_name,
                                    exports[i]->export_name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "duplicate VM export '%.*s'",
          (int)exports[i]->export_name.size, exports[i]->export_name.data);
    }
    strings[string_count++] = exports[i]->export_name;
  }
  iree_vm_bytecode_v0_ref_type_group_row_t* reference_groups = NULL;
  iree_vm_bytecode_v0_ref_type_entry_row_t* reference_entries = NULL;
  if (references.count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        request->scratch_arena, references.group_count,
        sizeof(*reference_groups), (void**)&reference_groups));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        request->scratch_arena, references.count, sizeof(*reference_entries),
        (void**)&reference_entries));
  }
  uint32_t reference_group_count = 0;
  for (uint32_t i = 0; i < references.count; ++i) {
    const loom_type_reference_key_t* key =
        loom_vm_reference_plan_key(&references, (uint16_t)i);
    if (!i || !iree_string_view_equal(
                  loom_vm_reference_plan_key(&references, (uint16_t)(i - 1))
                      ->namespace_name,
                  key->namespace_name)) {
      reference_groups[reference_group_count++] =
          (iree_vm_bytecode_v0_ref_type_group_row_t){
              .namespace_string_u16 = (uint16_t)string_count};
      strings[string_count++] = key->namespace_name;
    }
    ++reference_groups[reference_group_count - 1].entry_count_u32;
    reference_entries[i] = (iree_vm_bytecode_v0_ref_type_entry_row_t){
        .type_name_string_u16 = (uint16_t)string_count};
    strings[string_count++] = key->type_name;
  }
  iree_vm_bytecode_v0_import_group_row_t* import_groups = NULL;
  iree_vm_bytecode_v0_import_entry_row_t* import_entries = NULL;
  uint32_t import_group_count = 0;
  if (import_count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        request->scratch_arena, import_count, sizeof(*import_groups),
        (void**)&import_groups));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        request->scratch_arena, import_count, sizeof(*import_entries),
        (void**)&import_entries));
  }
  for (uint32_t i = 0; i < import_count; ++i) {
    const loom_vm_module_callable_t* entry = imports[i];
    if (!i || !iree_string_view_equal(imports[i - 1]->import.module_name,
                                      entry->import.module_name)) {
      import_groups[import_group_count++] =
          (iree_vm_bytecode_v0_import_group_row_t){.module_name_string_u16 =
                                                       (uint16_t)string_count};
      strings[string_count++] = entry->import.module_name;
    }
    ++import_groups[import_group_count - 1].entry_count_u32;
    import_entries[i] = (iree_vm_bytecode_v0_import_entry_row_t){
        .symbol_name_string_u16 = (uint16_t)string_count,
        .callable_type_ordinal_u16 = entry->callable_ordinal,
    };
    strings[string_count++] = entry->import.symbol_name;
  }
  if (string_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM string count exceeds the u16 ordinal space");
  }
  const iree_vm_bytecode_v0_image_header_t header = {
      .magic_u8 = {'I', 'R', 'E', 'E', 'V', 'M', 0, 0},
      .core_major_u16 = IREE_VM_BYTECODE_CORE_MAJOR,
      .core_required_minor_u16 = IREE_VM_BYTECODE_CORE_MINOR,
      .section_count_u16 =
          2 * (callable_count != 0) + (functions->definition_count != 0) +
          (export_count != 0) + (string_count != 0) + (references.count != 0) +
          (import_count != 0) + (functions->rodata.symbol_count != 0),
  };
  IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(header), &header));
  IREE_RETURN_IF_ERROR(iree_io_stream_write(
      stream, header.section_count_u16 * sizeof(directory[0]), directory));
  uint16_t section = 0;
  iree_io_stream_pos_t start = 0;
  iree_status_t status = iree_ok_status();
  if (string_count) {
    IREE_RETURN_IF_ERROR(loom_vm_section_begin(
        stream, IREE_VM_BYTECODE_SECTION_STRINGS, &directory[section], &start));
    const iree_vm_bytecode_v0_strings_header_t strings_header = {string_count};
    IREE_RETURN_IF_ERROR(
        iree_io_stream_write(stream, sizeof(strings_header), &strings_header));
    uint32_t offset = 0;
    IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(offset), &offset));
    for (uint32_t i = 0; i < string_count && iree_status_is_ok(status); ++i) {
      if (strings[i].size > UINT32_MAX - offset) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "VM string bytes exceed u32");
      } else {
        offset += (uint32_t)strings[i].size;
        status = iree_io_stream_write(stream, sizeof(offset), &offset);
      }
    }
    for (uint32_t i = 0; i < string_count && iree_status_is_ok(status); ++i) {
      status = iree_io_stream_write_string(stream, strings[i]);
    }
    IREE_RETURN_IF_ERROR(status);
    directory[section++].byte_length_u64 =
        iree_io_stream_offset(stream) - start;
  }

  if (references.count) {
    IREE_RETURN_IF_ERROR(
        loom_vm_section_begin(stream, IREE_VM_BYTECODE_SECTION_REF_TYPES,
                              &directory[section], &start));
    const iree_vm_bytecode_v0_ref_types_header_t types_header = {
        .group_count_u32 = references.group_count};
    IREE_RETURN_IF_ERROR(
        iree_io_stream_write(stream, sizeof(types_header), &types_header));
    IREE_RETURN_IF_ERROR(iree_io_stream_write(
        stream, references.group_count * sizeof(*reference_groups),
        reference_groups));
    IREE_RETURN_IF_ERROR(iree_io_stream_write(
        stream, references.count * sizeof(*reference_entries),
        reference_entries));
    directory[section++].byte_length_u64 =
        iree_io_stream_offset(stream) - start;
  }

  if (callable_count) {
    IREE_RETURN_IF_ERROR(
        loom_vm_section_begin(stream, IREE_VM_BYTECODE_SECTION_SIGNATURES,
                              &directory[section], &start));
    const iree_vm_bytecode_v0_signatures_header_t signatures_header = {
        callable_count};
    IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(signatures_header),
                                              &signatures_header));
    uint32_t descriptor_base = 0;
    for (uint32_t i = 0; i < callable_count && iree_status_is_ok(status); ++i) {
      const loom_vm_module_callable_t* entry = sorted[i];
      iree_vm_bytecode_v0_signature_row_t signature = entry->signature.row;
      signature.descriptor_base_u32 = descriptor_base;
      status = iree_io_stream_write(stream, sizeof(signature), &signature);
      descriptor_base += entry->argument_count + entry->results.count;
    }
    for (uint32_t i = 0; i < callable_count && iree_status_is_ok(status); ++i) {
      const loom_vm_module_callable_t* entry = sorted[i];
      status =
          iree_io_stream_write(stream,
                               (entry->argument_count + entry->results.count) *
                                   sizeof(*entry->signature.fields),
                               entry->signature.fields);
    }
    IREE_RETURN_IF_ERROR(status);
    directory[section++].byte_length_u64 =
        iree_io_stream_offset(stream) - start;

    IREE_RETURN_IF_ERROR(
        loom_vm_section_begin(stream, IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES,
                              &directory[section], &start));
    const iree_vm_bytecode_v0_callable_types_header_t callables_header = {
        callable_count};
    IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(callables_header),
                                              &callables_header));
    for (uint32_t i = 0; i < callable_count && iree_status_is_ok(status); ++i) {
      const iree_vm_bytecode_v0_callable_type_row_t callable = {
          .signature_ordinal_u16 = (uint16_t)i};
      status = iree_io_stream_write(stream, sizeof(callable), &callable);
    }
    IREE_RETURN_IF_ERROR(status);
    directory[section++].byte_length_u64 =
        iree_io_stream_offset(stream) - start;
  }

  if (import_count) {
    IREE_RETURN_IF_ERROR(loom_vm_section_begin(
        stream, IREE_VM_BYTECODE_SECTION_IMPORTS, &directory[section], &start));
    const iree_vm_bytecode_v0_imports_header_t imports_header = {
        .group_count_u32 = import_group_count};
    IREE_RETURN_IF_ERROR(
        iree_io_stream_write(stream, sizeof(imports_header), &imports_header));
    IREE_RETURN_IF_ERROR(iree_io_stream_write(
        stream, import_group_count * sizeof(*import_groups), import_groups));
    IREE_RETURN_IF_ERROR(iree_io_stream_write(
        stream, import_count * sizeof(*import_entries), import_entries));
    directory[section++].byte_length_u64 =
        iree_io_stream_offset(stream) - start;
  }
  if (export_count) {
    IREE_RETURN_IF_ERROR(loom_vm_section_begin(
        stream, IREE_VM_BYTECODE_SECTION_EXPORTS, &directory[section], &start));
    const iree_vm_bytecode_v0_exports_header_t exports_header = {export_count};
    IREE_RETURN_IF_ERROR(
        iree_io_stream_write(stream, sizeof(exports_header), &exports_header));
    for (uint32_t i = 0; i < export_count && iree_status_is_ok(status); ++i) {
      const loom_vm_module_callable_t* entry = exports[i];
      const iree_vm_bytecode_v0_export_row_t row = {
          .name_string_u16 = (uint16_t)i,
          .callable_type_ordinal_u16 = entry->callable_ordinal,
          .function_ordinal_u16 = entry->ordinal,
      };
      status = iree_io_stream_write(stream, sizeof(row), &row);
    }
    IREE_RETURN_IF_ERROR(status);
    directory[section++].byte_length_u64 =
        iree_io_stream_offset(stream) - start;
  }
  *out_section_count = section;
  return iree_ok_status();
}

static iree_status_t loom_vm_module_write(
    const loom_target_emit_request_t* request, loom_vm_module_plan_t functions,
    iree_io_stream_t* stream, bool* out_emitted) {
  *out_emitted = false;
  iree_vm_bytecode_v0_section_directory_row_t directory[8] = {0};
  uint16_t section = 0;
  // Keys, hash buckets and sorted metadata views die before function scratch.
  // Final descriptors and direct call bindings were allocated before this
  // scope.
  const iree_arena_checkpoint_t metadata_checkpoint =
      iree_arena_checkpoint_save(request->scratch_arena);
  iree_status_t status = loom_vm_module_write_metadata(
      request, &functions, stream, directory, &section);
  iree_arena_checkpoint_restore(&metadata_checkpoint);
  IREE_RETURN_IF_ERROR(status);
  const uint32_t function_count = functions.definition_count;
  iree_io_stream_pos_t start = 0;
  const uint8_t zero = 0;
  if (function_count) {
    IREE_RETURN_IF_ERROR(
        loom_vm_section_begin(stream, IREE_VM_BYTECODE_SECTION_FUNCTIONS,
                              &directory[section], &start));
    iree_vm_bytecode_v0_functions_header_t functions_header = {
        .function_count_u32 = function_count};
    IREE_RETURN_IF_ERROR(iree_io_stream_write(stream, sizeof(functions_header),
                                              &functions_header));
    IREE_RETURN_IF_ERROR(iree_io_stream_fill(
        stream, function_count * sizeof(iree_vm_bytecode_v0_function_row_t),
        &zero, 1));
    const iree_io_stream_pos_t bytecode_base = iree_io_stream_offset(stream);
    // Only emitted bytes, scalar row fields, and module-owned rodata references
    // survive each function. Reuse its planning storage across the module.
    iree_arena_allocator_t function_arena;
    iree_arena_initialize(request->scratch_arena->block_pool, &function_arena);
    loom_target_emit_request_t function_request = *request;
    function_request.scratch_arena = &function_arena;
    bool functions_emitted = true;
    for (uint32_t i = 0;
         i < functions.count && iree_status_is_ok(status) && functions_emitted;
         ++i) {
      const loom_vm_module_callable_t* entry = &functions.values[i];
      if (entry->target_kind != IREE_VM_BYTECODE_CONTROL_CALL_TARGET_LOCAL) {
        continue;
      }
      const iree_io_stream_pos_t offset =
          iree_io_stream_offset(stream) - bytecode_base;
      if (offset > UINT32_MAX) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "VM bytecode offset exceeds u32");
        continue;
      }
      iree_vm_bytecode_v0_function_row_t row = {
          .callable_type_ordinal_u16 = entry->callable_ordinal,
          .bytecode_offset_u32 = (uint32_t)offset,
      };
      bool function_emitted = false;
      status = loom_vm_function_emit(
          &function_request, entry->function, entry->function_version,
          &entry->signature, &functions, stream, &function_emitted, &row);
      iree_arena_reset(&function_arena);
      if (iree_status_is_ok(status) && function_emitted) {
        functions_header.maximum_block_count_u32 = iree_max(
            functions_header.maximum_block_count_u32, row.block_count_u32);
        status = loom_vm_stream_patch(
            stream,
            start + sizeof(functions_header) + entry->ordinal * sizeof(row),
            iree_make_const_byte_span(&row, sizeof(row)));
      } else if (iree_status_is_ok(status)) {
        functions_emitted = false;
      }
    }
    iree_arena_deinitialize(&function_arena);
    IREE_RETURN_IF_ERROR(status);
    if (!functions_emitted) {
      return iree_ok_status();
    }
    directory[section++].byte_length_u64 =
        iree_io_stream_offset(stream) - start;
    IREE_RETURN_IF_ERROR(
        loom_vm_stream_patch(stream, start,
                             iree_make_const_byte_span(
                                 &functions_header, sizeof(functions_header))));
  }
  if (functions.rodata.symbol_count) {
    // Section and block alignment are relative to the serialized image, not
    // the addresses of segmented output chunks. The loader handles a host
    // image allocation whose base does not satisfy a stronger block alignment.
    IREE_RETURN_IF_ERROR(iree_io_stream_fill(
        stream,
        -iree_io_stream_offset(stream) & (functions.rodata.alignment - 1),
        &zero, 1));
    IREE_RETURN_IF_ERROR(loom_vm_section_begin(
        stream, IREE_VM_BYTECODE_SECTION_RODATA, &directory[section], &start));
    directory[section].payload_alignment_u32 = functions.rodata.alignment;
    const iree_vm_bytecode_v0_rodata_header_t rodata_header = {
        .block_count_u32 = functions.rodata.count};
    IREE_RETURN_IF_ERROR(
        iree_io_stream_write(stream, sizeof(rodata_header), &rodata_header));
    for (uint32_t i = 0;
         i < functions.rodata.count && iree_status_is_ok(status); ++i) {
      const loom_op_t* definition = functions.rodata.values[i];
      const iree_vm_bytecode_v0_rodata_block_descriptor_t row = {
          .byte_length_u64 =
              loom_global_rodata_def_contents(definition).data_length,
          .minimum_alignment_u32 = (uint32_t)iree_max(
              1, loom_global_rodata_def_alignment(definition)),
      };
      status = iree_io_stream_write(stream, sizeof(row), &row);
    }
    for (uint32_t i = 0;
         i < functions.rodata.count && iree_status_is_ok(status); ++i) {
      const loom_op_t* definition = functions.rodata.values[i];
      const uint32_t alignment =
          (uint32_t)iree_max(1, loom_global_rodata_def_alignment(definition));
      status = iree_io_stream_fill(
          stream, -(iree_io_stream_offset(stream) - start) & (alignment - 1),
          &zero, 1);
      if (iree_status_is_ok(status)) {
        const iree_const_byte_span_t contents =
            loom_global_rodata_def_contents(definition);
        status =
            iree_io_stream_write(stream, contents.data_length, contents.data);
      }
    }
    IREE_RETURN_IF_ERROR(status);
    directory[section++].byte_length_u64 =
        iree_io_stream_offset(stream) - start;
  }
  IREE_RETURN_IF_ERROR(loom_vm_stream_patch(
      stream, sizeof(iree_vm_bytecode_v0_image_header_t),
      iree_make_const_byte_span(directory, section * sizeof(directory[0]))));
  *out_emitted = true;
  return iree_ok_status();
}

iree_status_t loom_vm_module_emit(const loom_target_emit_request_t* request,
                                  bool* out_emitted,
                                  loom_target_emit_artifact_t* out_artifact) {
  *out_emitted = false;
  *out_artifact = (loom_target_emit_artifact_t){0};
  const iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(request->scratch_arena);
  loom_vm_module_plan_t functions = {0};
  iree_status_t status = loom_vm_module_collect(request, &functions);
  iree_io_stream_t* stream = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE, 32 * 1024,
        request->allocator, &stream);
  }
  bool module_emitted = false;
  if (iree_status_is_ok(status)) {
    status = loom_vm_module_write(request, functions, stream, &module_emitted);
  }
  if (iree_status_is_ok(status) && module_emitted) {
    status = iree_io_vec_stream_move_contents(stream, &out_artifact->contents);
  }
  if (iree_status_is_ok(status) && module_emitted) {
    out_artifact->target_artifact_format =
        LOOM_TARGET_ARTIFACT_FORMAT_VM_BINARY;
    *out_emitted = true;
  }
  iree_io_stream_release(stream);
  iree_arena_checkpoint_restore(&checkpoint);
  return status;
}
