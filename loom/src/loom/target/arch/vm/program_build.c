// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/program_build.h"

#include <stdlib.h>
#include <string.h>

#include "iree/io/vec_stream.h"
#include "iree/vm/bytecode/wire/core.h"
#include "iree/vm/native_call.h"
#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/vm/descriptors/descriptors.h"
#include "loom/target/arch/vm/function_plan.h"
#include "loom/target/arch/vm/reference_plan.h"

// A signature is ordered by argument count/types then result count/types,
// exactly as the wire callable table requires. Ordinals are assigned by
// sorting once; the runtime performs no hashing or interning.
static int loom_vm_signature_compare(const void* lhs_ptr, const void* rhs_ptr) {
  const loom_vm_program_callable_t* lhs =
      *(const loom_vm_program_callable_t* const*)lhs_ptr;
  const loom_vm_program_callable_t* rhs =
      *(const loom_vm_program_callable_t* const*)rhs_ptr;
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
  const loom_vm_program_callable_t* lhs =
      *(const loom_vm_program_callable_t* const*)lhs_ptr;
  const loom_vm_program_callable_t* rhs =
      *(const loom_vm_program_callable_t* const*)rhs_ptr;
  return iree_string_view_compare(lhs->export_name, rhs->export_name);
}

// Import rows are ordered by module, symbol, then structural callable type.
// Equal rows share one runtime binding even when authored under several
// aliases.
static int loom_vm_import_compare(const void* lhs_ptr, const void* rhs_ptr) {
  const loom_vm_program_callable_t* lhs =
      *(const loom_vm_program_callable_t* const*)lhs_ptr;
  const loom_vm_program_callable_t* rhs =
      *(const loom_vm_program_callable_t* const*)rhs_ptr;
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
  const loom_type_t* value_type = loom_type_register_value_type(type);
  if (value_type) {
    if (loom_type_is_buffer(*value_type) ||
        (loom_type_is_dialect(*value_type) &&
         loom_type_dialect_param_count(*value_type) == 0)) {
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_REF;
      return iree_ok_status();
    }
    const uint16_t kind = loom_vm_type_scalar_kind(*value_type);
    if (kind) {
      *out_kind = kind;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "VM signature requires a supported typed register");
}

// Only top-level symbol definitions are collected. Callgraph specialization
// and library composition belong to the shared compiler, not binary emission.
static iree_status_t loom_vm_program_collect(
    loom_module_t* module,
    const loom_function_version_list_t* function_versions,
    iree_arena_allocator_t* plan_arena, iree_arena_allocator_t* scratch_arena,
    loom_vm_program_build_t* out_program) {
  loom_target_function_version_snapshot_t versions = {0};
  IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
      module, function_versions, scratch_arena, &versions));
  loom_vm_program_callable_t* functions = NULL;
  iree_host_size_t storage_size = 0;
  iree_host_size_t bindings_offset = 0;
  iree_host_size_t ordinals_offset = 0;
  iree_host_size_t rodata_offset = 0;
  iree_host_size_t rodata_symbols_offset = 0;
  // Pointer-bearing arrays precede the two-byte symbol and ordinal arrays so
  // each field keeps its native alignment without inter-array padding.
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &storage_size,
      IREE_STRUCT_FIELD(module->symbols.count, loom_vm_program_callable_t,
                        NULL),
      IREE_STRUCT_FIELD(module->symbols.count, loom_vm_program_callable_t*,
                        &bindings_offset),
      IREE_STRUCT_FIELD(module->symbols.count, const loom_op_t*,
                        &rodata_offset),
      IREE_STRUCT_FIELD(module->symbols.count, uint16_t, &ordinals_offset),
      IREE_STRUCT_FIELD(module->symbols.count, loom_symbol_id_t,
                        &rodata_symbols_offset)));
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      plan_arena, iree_max(storage_size, 1), (void**)&functions));
  loom_vm_program_callable_t** bindings_by_symbol =
      (loom_vm_program_callable_t**)((uint8_t*)functions + bindings_offset);
  uint16_t* ordinals_by_symbol =
      (uint16_t*)((uint8_t*)functions + ordinals_offset);
  const loom_op_t** rodata =
      (const loom_op_t**)((uint8_t*)functions + rodata_offset);
  loom_symbol_id_t* rodata_symbols =
      (loom_symbol_id_t*)((uint8_t*)functions + rodata_symbols_offset);

  uint32_t count = 0;
  uint32_t definition_count = 0;
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
    // diagnosed when the function's call schedule is prepared.
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

    loom_vm_program_callable_t* entry = &functions[count];
    bindings_by_symbol[i] = entry;
    *entry = (loom_vm_program_callable_t){
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
      if (loom_func_like_abi(function) == LOOM_TARGET_ABI_VM_NATIVE) {
        status = loom_vm_native_call_template_build(
            module, loom_low_func_decl_abi_layout(op), entry->argument_count,
            entry->results.count, plan_arena, &entry->native);
      }
    } else {
      if (loom_func_like_abi(function) == LOOM_TARGET_ABI_VM_NATIVE) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "vm_native ABI requires an import declaration");
        continue;
      }
      if (definition_count == (uint32_t)UINT16_MAX + 1) {
        status =
            iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "VM function count exceeds the u16 ordinal space");
        continue;
      }
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
    const iree_host_size_t field_count =
        (iree_host_size_t)entry->argument_count + entry->results.count;
    if (!iree_host_size_checked_add(descriptor_count, field_count,
                                    &descriptor_count)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "VM signature descriptor count overflows");
    }
    ++count;
  }
  IREE_RETURN_IF_ERROR(status);

  iree_vm_bytecode_v0_signature_descriptor_row_t* descriptors = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(plan_arena, iree_max(descriptor_count, 1),
                                sizeof(*descriptors), (void**)&descriptors));
  loom_low_allocation_abi_location_t* registers = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(plan_arena, iree_max(descriptor_count, 1),
                                sizeof(*registers), (void**)&registers));
  for (uint32_t i = 0; i < count; ++i) {
    functions[i].signature.fields = descriptors;
    functions[i].signature.registers = registers;
    if (!functions[i].native) {
      functions[i].call_contract = (loom_low_call_contract_t){
          .arguments = registers,
          .argument_count = functions[i].argument_count,
          .results = registers + functions[i].argument_count,
          .result_count = functions[i].results.count,
          .clobbers = {functions[i].call_clobbers, 2},
      };
    }
    descriptors += functions[i].argument_count + functions[i].results.count;
    registers += functions[i].argument_count + functions[i].results.count;
  }
  *out_program = (loom_vm_program_build_t){
      .values = functions,
      .bindings_by_symbol = bindings_by_symbol,
      .count = count,
      .definition_count = definition_count,
      .native_calls = {.arena = plan_arena},
      .rodata = {.ordinals_by_symbol = ordinals_by_symbol,
                 .symbols = rodata_symbols,
                 .symbol_count = rodata_count,
                 .values = rodata,
                 .alignment = IREE_VM_BYTECODE_IMAGE_ALIGNMENT},
  };
  return iree_ok_status();
}

// Signature fields own provisional canonical ordinals while metadata is
// prepared. No per-occurrence reference records or source-sized lookup arrays
// are retained in the final program.
static iree_status_t loom_vm_program_resolve_signatures(
    const loom_module_t* module, iree_arena_allocator_t* scratch_arena,
    loom_vm_program_build_t* program, loom_vm_reference_plan_t* references) {
  iree_status_t status = iree_ok_status();
  loom_vm_native_layout_plan_t** layouts_by_type = NULL;
  for (uint32_t i = 0; i < program->count && iree_status_is_ok(status); ++i) {
    loom_vm_program_callable_t* entry = &program->values[i];
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
        uint16_t* bank_count = NULL;
        if (descriptors[j].kind_u16 == IREE_VM_BYTECODE_SIGNATURE_KIND_REF) {
          status = loom_vm_reference_plan_bind(
              references, module, *loom_type_register_value_type(type),
              scratch_arena, &descriptors[j].type_ordinal_u16);
          bank_count = j < entry->argument_count ? &row->argument_ref_count_u16
                                                 : &row->result_ref_count_u16;
        } else {
          bank_count = j < entry->argument_count
                           ? &row->argument_value_count_u16
                           : &row->result_value_count_u16;
        }
        entry->signature.registers[j] = (loom_low_allocation_abi_location_t){
            .location_kind =
                *bank_count < 16
                    ? LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER
                    : LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED,
            .descriptor_reg_class_id =
                descriptors[j].kind_u16 == IREE_VM_BYTECODE_SIGNATURE_KIND_REF
                    ? VM_CORE_REG_CLASS_ID_REF
                    : VM_CORE_REG_CLASS_ID_VALUE,
            .location_base = *bank_count,
        };
        ++(*bank_count);
      }
    }
    const iree_vm_bytecode_v0_signature_row_t* row = &entry->signature.row;
    entry->call_clobbers[VM_CORE_REG_CLASS_ID_VALUE] =
        (loom_low_call_clobber_t){
            .register_class = VM_CORE_REG_CLASS_ID_VALUE,
            .count = iree_min(16, iree_max(row->argument_value_count_u16,
                                           row->result_value_count_u16)),
        };
    entry->call_clobbers[VM_CORE_REG_CLASS_ID_REF] = (loom_low_call_clobber_t){
        .register_class = VM_CORE_REG_CLASS_ID_REF,
        .count = iree_min(16, iree_max(row->argument_ref_count_u16,
                                       row->result_ref_count_u16)),
    };
    if (iree_status_is_ok(status) && entry->native) {
      if (!layouts_by_type) {
        status = iree_arena_allocate_array(scratch_arena, module->types.count,
                                           sizeof(*layouts_by_type),
                                           (void**)&layouts_by_type);
        if (iree_status_is_ok(status)) {
          memset(layouts_by_type, 0,
                 module->types.count * sizeof(*layouts_by_type));
        }
      }
      if (iree_status_is_ok(status)) {
        status = loom_vm_native_call_layout_bind(&program->native_calls, module,
                                                 entry->native, layouts_by_type,
                                                 references, scratch_arena);
      }
    }
  }
  if (iree_status_is_ok(status) &&
      loom_vm_reference_plan_finalize(references)) {
    for (loom_vm_native_layout_plan_t* layout = program->native_calls.layouts;
         layout; layout = layout->next) {
      for (uint32_t j = 0; j < layout->row.field_count_u16; ++j) {
        iree_vm_bytecode_v0_native_field_row_t* field = &layout->fields[j];
        if ((field->kind_u16 & ~IREE_VM_NATIVE_CALL_FIELD_RESULT) ==
            IREE_VM_NATIVE_CALL_FIELD_REF) {
          field->type_ordinal_u16 = loom_vm_reference_plan_ordinal(
              references, field->type_ordinal_u16);
        }
      }
    }
    for (uint32_t i = 0; i < program->count; ++i) {
      loom_vm_program_callable_t* entry = &program->values[i];
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

static iree_status_t loom_vm_program_build_metadata(
    const loom_module_t* module, iree_arena_allocator_t* plan_arena,
    iree_arena_allocator_t* scratch_arena, loom_vm_program_build_t* program,
    loom_vm_program_plan_t* out_plan) {
  loom_vm_reference_plan_t references = {0};
  IREE_RETURN_IF_ERROR(loom_vm_program_resolve_signatures(
      module, scratch_arena, program, &references));
  if (program->native_calls.layout_count) {
    const loom_vm_native_layout_plan_t** layouts = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, program->native_calls.layout_count, sizeof(*layouts),
        (void**)&layouts));
    for (const loom_vm_native_layout_plan_t* layout =
             program->native_calls.layouts;
         layout; layout = layout->next) {
      layouts[layout->ordinal] = layout;
    }
    for (uint32_t i = 0; i < program->count; ++i) {
      const loom_vm_program_callable_t* entry = &program->values[i];
      if (entry->native) {
        IREE_RETURN_IF_ERROR(loom_vm_native_call_template_validate(
            entry->native, layouts, entry->signature.fields,
            entry->argument_count, scratch_arena));
      }
    }
  }

  // Sorted views retain direct call bindings and definition order while
  // interning signatures and imports for canonical runtime tables.
  loom_vm_program_callable_t** sorted = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena,
      iree_max((iree_host_size_t)3 * program->count, (iree_host_size_t)1),
      sizeof(*sorted), (void**)&sorted));
  loom_vm_program_callable_t** exports = sorted + program->count;
  loom_vm_program_callable_t** imports = exports + program->count;
  uint32_t export_count = 0;
  uint32_t import_count = 0;
  uint32_t ordinary_count = 0;
  uint32_t native_count = 0;
  for (uint32_t i = 0; i < program->count; ++i) {
    loom_vm_program_callable_t* entry = &program->values[i];
    if (entry->native) {
      ++native_count;
      continue;
    }
    sorted[ordinary_count++] = entry;
    if (entry->target_kind ==
        IREE_VM_BYTECODE_CONTROL_CALL_TARGET_REQUIRED_IMPORT) {
      imports[import_count++] = entry;
    } else if (!iree_string_view_is_empty(entry->export_name)) {
      exports[export_count++] = entry;
    }
  }

  qsort(sorted, ordinary_count, sizeof(*sorted), loom_vm_signature_compare);
  uint32_t callable_count = 0;
  for (uint32_t i = 0; i < ordinary_count; ++i) {
    loom_vm_program_callable_t* entry = sorted[i];
    if (!callable_count ||
        loom_vm_signature_compare(&sorted[callable_count - 1], &entry)) {
      if (callable_count == (uint32_t)UINT16_MAX + 1) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "VM callable type count exceeds the u16 ordinal space");
      }
      sorted[callable_count++] = entry;
    }
    entry->callable_ordinal = (uint16_t)(callable_count - 1);
  }

  qsort(imports, import_count, sizeof(*imports), loom_vm_import_compare);
  uint32_t unique_import_count = 0;
  for (uint32_t i = 0; i < import_count; ++i) {
    loom_vm_program_callable_t* entry = imports[i];
    if (!unique_import_count ||
        loom_vm_import_compare(&imports[unique_import_count - 1], &entry)) {
      if (unique_import_count == (uint32_t)UINT16_MAX + 1) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "VM import count exceeds the u16 ordinal space");
      }
      imports[unique_import_count++] = entry;
    }
    entry->ordinal = (uint16_t)(unique_import_count - 1);
  }
  import_count = unique_import_count;
  qsort(exports, export_count, sizeof(*exports), loom_vm_export_compare);

  uint32_t planned_import_group_count = 0;
  for (uint32_t i = 0; i < import_count; ++i) {
    if (!i || !iree_string_view_equal(imports[i - 1]->import.module_name,
                                      imports[i]->import.module_name)) {
      ++planned_import_group_count;
    }
  }
  const uint64_t string_capacity =
      (uint64_t)export_count + references.count + references.group_count +
      import_count + planned_import_group_count + 2 * native_count;
  if (string_capacity > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM string count exceeds the u16 ordinal space");
  }
  iree_string_view_t* strings = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan_arena, iree_max(string_capacity, UINT64_C(1)), sizeof(*strings),
      (void**)&strings));
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
        plan_arena, references.group_count, sizeof(*reference_groups),
        (void**)&reference_groups));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan_arena, references.count,
                                                   sizeof(*reference_entries),
                                                   (void**)&reference_entries));
  }
  uint32_t reference_group_count = 0;
  const loom_type_reference_key_t* previous_reference_key = NULL;
  for (uint32_t i = 0; i < references.count; ++i) {
    const loom_type_reference_key_t* key =
        loom_vm_reference_plan_key(&references, (uint16_t)i);
    if (previous_reference_key == NULL ||
        !iree_string_view_equal(previous_reference_key->namespace_name,
                                key->namespace_name)) {
      reference_groups[reference_group_count++] =
          (iree_vm_bytecode_v0_ref_type_group_row_t){
              .namespace_string_u16 = (uint16_t)string_count};
      strings[string_count++] = key->namespace_name;
    }
    previous_reference_key = key;
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
        plan_arena, planned_import_group_count, sizeof(*import_groups),
        (void**)&import_groups));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan_arena, import_count,
                                                   sizeof(*import_entries),
                                                   (void**)&import_entries));
  }
  for (uint32_t i = 0; i < import_count; ++i) {
    const loom_vm_program_callable_t* entry = imports[i];
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
  iree_vm_bytecode_v0_native_import_row_t* native_imports = NULL;
  if (native_count) {
    if (native_count > 65536) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM native import count exceeds u16 ordinals");
    }
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan_arena, native_count,
                                                   sizeof(*native_imports),
                                                   (void**)&native_imports));
  }
  uint32_t native_import_count = 0;
  for (uint32_t i = 0; i < program->count; ++i) {
    loom_vm_program_callable_t* entry = &program->values[i];
    if (entry->native) {
      entry->native->import_ordinal = (uint16_t)native_import_count;
      native_imports[native_import_count++] =
          (iree_vm_bytecode_v0_native_import_row_t){
              .module_name_u16 = (uint16_t)string_count,
              .export_name_u16 = (uint16_t)(string_count + 1),
              .layout_ordinal_u16 = entry->native->layout_ordinal,
              .version_u32 = entry->native->version,
          };
      strings[string_count++] = entry->import.module_name;
      strings[string_count++] = entry->import.symbol_name;
    }
  }
  out_plan->native_imports = native_imports;
  out_plan->native_import_count = native_import_count;
  out_plan->native_layouts = program->native_calls.layouts;
  out_plan->native_layout_count = program->native_calls.layout_count;
  out_plan->native_field_count = program->native_calls.field_count;
  IREE_ASSERT_EQ(string_count, string_capacity);
  uint32_t string_byte_length = 0;
  for (uint32_t i = 0; i < string_count; ++i) {
    if (strings[i].size > UINT32_MAX - string_byte_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM string bytes exceed u32");
    }
    string_byte_length += (uint32_t)strings[i].size;
  }

  uint64_t descriptor_count = 0;
  for (uint32_t i = 0; i < callable_count; ++i) {
    descriptor_count +=
        (uint64_t)sorted[i]->argument_count + sorted[i]->results.count;
  }
  if (descriptor_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VM signature descriptor count exceeds the u32 index space");
  }
  iree_vm_bytecode_v0_signature_row_t* signatures = NULL;
  iree_vm_bytecode_v0_signature_descriptor_row_t* signature_descriptors = NULL;
  if (callable_count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan_arena, callable_count, sizeof(*signatures), (void**)&signatures));
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan_arena, iree_max(descriptor_count, UINT64_C(1)),
      sizeof(*signature_descriptors), (void**)&signature_descriptors));
  uint32_t descriptor_base = 0;
  for (uint32_t i = 0; i < callable_count; ++i) {
    const loom_vm_program_callable_t* entry = sorted[i];
    signatures[i] = entry->signature.row;
    signatures[i].descriptor_base_u32 = descriptor_base;
    const uint32_t field_count = entry->argument_count + entry->results.count;
    if (field_count) {
      memcpy(&signature_descriptors[descriptor_base], entry->signature.fields,
             field_count * sizeof(*signature_descriptors));
    }
    descriptor_base += field_count;
  }

  iree_vm_bytecode_v0_export_row_t* export_rows = NULL;
  if (export_count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan_arena, export_count, sizeof(*export_rows), (void**)&export_rows));
  }
  for (uint32_t i = 0; i < export_count; ++i) {
    const loom_vm_program_callable_t* entry = exports[i];
    export_rows[i] = (iree_vm_bytecode_v0_export_row_t){
        .name_string_u16 = (uint16_t)i,
        .callable_type_ordinal_u16 = entry->callable_ordinal,
        .function_ordinal_u16 = entry->ordinal,
    };
  }

  out_plan->strings = strings;
  out_plan->string_count = string_count;
  out_plan->reference_groups = reference_groups;
  out_plan->reference_group_count = reference_group_count;
  out_plan->reference_entries = reference_entries;
  out_plan->reference_count = references.count;
  out_plan->signatures = signatures;
  out_plan->signature_count = callable_count;
  out_plan->signature_descriptors = signature_descriptors;
  out_plan->signature_descriptor_count = (uint32_t)descriptor_count;
  out_plan->import_groups = import_groups;
  out_plan->import_group_count = import_group_count;
  out_plan->imports = import_entries;
  out_plan->import_count = import_count;
  out_plan->exports = export_rows;
  out_plan->export_count = export_count;
  return iree_ok_status();
}

static iree_status_t loom_vm_program_build_functions(
    loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* plan_arena, iree_arena_allocator_t* scratch_arena,
    loom_vm_program_build_t* program, iree_allocator_t bytecode_allocator,
    bool* out_accepted, loom_vm_program_plan_t* out_plan) {
  *out_accepted = false;
  if (!program->definition_count) {
    *out_accepted = true;
    return iree_ok_status();
  }

  iree_vm_bytecode_v0_function_row_t* rows = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan_arena, program->definition_count, sizeof(*rows), (void**)&rows));
  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE, 32 * 1024,
      bytecode_allocator, &stream));

  uint32_t maximum_block_count = 0;
  bool functions_accepted = true;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < program->count && iree_status_is_ok(status) && functions_accepted;
       ++i) {
    const loom_vm_program_callable_t* entry = &program->values[i];
    if (entry->target_kind != IREE_VM_BYTECODE_CONTROL_CALL_TARGET_LOCAL) {
      continue;
    }
    const iree_io_stream_pos_t offset = iree_io_stream_offset(stream);
    if (offset > UINT32_MAX) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "VM bytecode offset exceeds u32");
      continue;
    }
    iree_vm_bytecode_v0_function_row_t row = {
        .callable_type_ordinal_u16 = entry->callable_ordinal,
        .bytecode_offset_u32 = (uint32_t)offset,
    };
    bool function_accepted = false;
    status = loom_vm_function_plan_write(
        module, entry->function, entry->function_version, descriptor_registry,
        diagnostic_emitter, &entry->signature, entry->ordinal, program,
        scratch_arena, stream, &function_accepted, &row);
    iree_arena_reset(scratch_arena);
    if (iree_status_is_ok(status) && function_accepted &&
        row.bytecode_length_u32 > UINT32_MAX - (uint32_t)offset) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "VM aggregate function bytecode exceeds the u32 extent");
    }
    if (iree_status_is_ok(status) && function_accepted) {
      rows[entry->ordinal] = row;
      maximum_block_count = iree_max(maximum_block_count, row.block_count_u32);
    } else if (iree_status_is_ok(status)) {
      functions_accepted = false;
    }
  }

  iree_byte_sequence_t* bytecode = NULL;
  if (iree_status_is_ok(status) && functions_accepted) {
    status = iree_io_vec_stream_move_contents(stream, &bytecode);
  }
  iree_io_stream_release(stream);
  if (iree_status_is_ok(status) && functions_accepted) {
    out_plan->functions = rows;
    out_plan->function_count = program->definition_count;
    out_plan->function_bytecode = bytecode;
    out_plan->maximum_block_count = maximum_block_count;
    *out_accepted = true;
  }
  return status;
}

static iree_status_t loom_vm_program_build_rodata(
    iree_arena_allocator_t* plan_arena, const loom_vm_program_build_t* program,
    loom_vm_program_plan_t* out_plan) {
  loom_vm_rodata_plan_t* rodata = NULL;
  if (program->rodata.count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan_arena, program->rodata.count, sizeof(*rodata), (void**)&rodata));
  }
  for (uint32_t i = 0; i < program->rodata.count; ++i) {
    const loom_op_t* definition = program->rodata.values[i];
    const int64_t source_alignment =
        loom_global_rodata_def_alignment(definition);
    rodata[i] = (loom_vm_rodata_plan_t){
        .contents = loom_global_rodata_def_contents(definition),
        .alignment = (uint32_t)iree_max(source_alignment, INT64_C(1)),
    };
  }
  out_plan->rodata = rodata;
  out_plan->rodata_count = program->rodata.count;
  out_plan->rodata_alignment = program->rodata.alignment;
  out_plan->has_rodata_section = program->rodata.symbol_count != 0;
  return iree_ok_status();
}

iree_status_t loom_vm_program_plan_build(
    loom_module_t* module,
    const loom_function_version_list_t* function_versions,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    iree_allocator_t bytecode_allocator, bool* out_accepted,
    loom_vm_program_plan_t* out_plan) {
  *out_accepted = false;
  *out_plan = (loom_vm_program_plan_t){
      .rodata_alignment = IREE_VM_BYTECODE_IMAGE_ALIGNMENT,
  };

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  loom_vm_program_build_t program = {0};
  iree_status_t status = loom_vm_program_collect(
      module, function_versions, arena, &scratch_arena, &program);
  if (iree_status_is_ok(status)) {
    status = loom_vm_program_build_metadata(module, arena, &scratch_arena,
                                            &program, out_plan);
  }
  iree_arena_reset(&scratch_arena);
  bool functions_accepted = false;
  if (iree_status_is_ok(status)) {
    status = loom_vm_program_build_functions(
        module, descriptor_registry, diagnostic_emitter, arena, &scratch_arena,
        &program, bytecode_allocator, &functions_accepted, out_plan);
  }
  if (iree_status_is_ok(status) && functions_accepted) {
    status = loom_vm_program_build_rodata(arena, &program, out_plan);
  }
  if (iree_status_is_ok(status) && functions_accepted) {
    *out_accepted = true;
    out_plan->native_calls = program.native_calls.head;
    out_plan->native_call_count = program.native_calls.count;
  } else {
    loom_vm_program_plan_deinitialize(out_plan);
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
