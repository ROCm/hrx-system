// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/vm/module_layout.h"

#include <stdlib.h>

#include "iree/vm/bytecode/wire/core/selectors.h"
#include "iree/vm/bytecode/wire/module_format.h"
#include "loom/codegen/low/function.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/vm/abi/layout.h"

static iree_string_view_t loom_vm_module_layout_string_or_empty(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_string_view_t loom_vm_module_layout_export_name(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_op_t* function_op) {
  loom_func_like_t function = loom_func_like_cast(module, function_op);
  if (!loom_func_like_isa(function)) return iree_string_view_empty();
  iree_string_view_t export_name = loom_vm_module_layout_string_or_empty(
      module, loom_func_like_export_symbol(function));
  if (!iree_string_view_is_empty(export_name)) return export_name;
  if (loom_func_like_visibility(function) != 0 ||
      iree_any_bit_set(symbol->flags, LOOM_SYMBOL_FLAG_PUBLIC)) {
    return loom_vm_module_layout_string_or_empty(module, symbol->name_id);
  }
  return iree_string_view_empty();
}

static int loom_vm_module_layout_compare_export_names(const void* lhs_ptr,
                                                      const void* rhs_ptr) {
  const loom_vm_module_function_layout_t* lhs =
      *(loom_vm_module_function_layout_t* const*)lhs_ptr;
  const loom_vm_module_function_layout_t* rhs =
      *(loom_vm_module_function_layout_t* const*)rhs_ptr;
  return iree_string_view_compare(lhs->export_name, rhs->export_name);
}

static int loom_vm_module_layout_compare_imports(const void* lhs_ptr,
                                                 const void* rhs_ptr) {
  const loom_vm_module_import_layout_t* lhs =
      *(loom_vm_module_import_layout_t* const*)lhs_ptr;
  const loom_vm_module_import_layout_t* rhs =
      *(loom_vm_module_import_layout_t* const*)rhs_ptr;
  int comparison = iree_string_view_compare(lhs->module_name, rhs->module_name);
  if (comparison != 0) return comparison;
  comparison = iree_string_view_compare(lhs->symbol_name, rhs->symbol_name);
  if (comparison != 0) return comparison;
  if (lhs->callable_type_ordinal < rhs->callable_type_ordinal) return -1;
  if (lhs->callable_type_ordinal > rhs->callable_type_ordinal) return 1;
  return 0;
}

static bool loom_vm_module_layout_imports_equal(
    const loom_vm_module_import_layout_t* lhs,
    const loom_vm_module_import_layout_t* rhs) {
  return iree_string_view_equal(lhs->module_name, rhs->module_name) &&
         iree_string_view_equal(lhs->symbol_name, rhs->symbol_name) &&
         lhs->callable_type_ordinal == rhs->callable_type_ordinal;
}

static iree_status_t loom_vm_module_layout_count_switch_targets(
    const loom_op_t* function_op, uint32_t* out_count) {
  *out_count = 0;
  const loom_region_t* body = loom_low_function_const_body(function_op);
  if (body == NULL) return iree_ok_status();
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(body->blocks[block_index], op) {
      if (!loom_low_switch_isa(op)) continue;
      const uint32_t target_count = loom_low_switch_target_dests(op).count;
      if (target_count > UINT32_MAX - *out_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "VM function switch-target entry count exceeds u32");
      }
      *out_count += target_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_module_layout_count(
    const loom_module_t* module, iree_host_size_t* out_function_count,
    iree_host_size_t* out_import_count) {
  *out_function_count = 0;
  *out_import_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_op_t* defining_op = module->symbols.entries[i].defining_op;
    if (defining_op == NULL) continue;
    if (loom_low_func_decl_isa(defining_op)) {
      if (loom_low_func_decl_import_kind(defining_op) != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "VM function declarations cannot import physical target code");
      }
      if (loom_low_func_decl_import_module(defining_op) ==
          LOOM_STRING_ID_INVALID) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "plain function declaration must be resolved before VM emission");
      }
      if (*out_import_count == 65536u) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "VM import declaration count exceeds the u16 ordinal domain");
      }
      ++*out_import_count;
      continue;
    }
    if (loom_low_kernel_def_isa(defining_op)) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "VM kernel entry points are not implemented");
    }
    if (!loom_low_func_def_isa(defining_op)) continue;
    if (*out_function_count == 65536u) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "VM function count exceeds the u16 ordinal domain");
    }
    ++*out_function_count;
  }
  if (*out_function_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "VM module emission requires low.func.def");
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_module_layout_resolve_logical_signature(
    loom_module_t* module, iree_arena_allocator_t* arena,
    loom_op_t* function_op, loom_type_t* out_signature) {
  const loom_named_attr_slice_t abi_layout =
      loom_low_func_def_isa(function_op)
          ? loom_low_func_def_abi_layout(function_op)
          : loom_low_func_decl_abi_layout(function_op);
  if (abi_layout.count != 0) {
    return loom_vm_call_abi_layout_resolve_signature(module, abi_layout,
                                                     out_signature);
  }

  loom_func_like_t function_like = loom_func_like_cast(module, function_op);
  uint16_t argument_count = 0;
  loom_func_like_arg_ids(function_like, &argument_count);
  if (argument_count != 0 || function_op->result_count != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "nonempty VM callable is missing its preserved logical ABI signature");
  }
  loom_func_type_data_t* empty_signature = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, sizeof(*empty_signature),
                                           (void**)&empty_signature));
  *empty_signature = (loom_func_type_data_t){0};
  *out_signature = loom_type_function(empty_signature);
  return iree_ok_status();
}

static iree_status_t loom_vm_module_layout_populate_functions(
    loom_module_t* module, iree_arena_allocator_t* arena,
    loom_vm_module_layout_t* layout) {
  iree_host_size_t function_index = 0;
  for (iree_host_size_t symbol_index = 0; symbol_index < module->symbols.count;
       ++symbol_index) {
    const loom_symbol_t* symbol = &module->symbols.entries[symbol_index];
    loom_op_t* function_op = symbol->defining_op;
    if (function_op == NULL || !loom_low_func_def_isa(function_op)) continue;

    loom_vm_module_function_layout_t* function =
        &layout->functions[function_index++];
    function->function_op = function_op;
    layout->call_targets_by_symbol[symbol_index] =
        (loom_vm_module_call_target_t){
            .kind = IREE_VM_ISA_CONTROL_CALL_TARGET_LOCAL,
            .ordinal = (uint16_t)(function_index - 1),
        };
    function->export_name =
        loom_vm_module_layout_export_name(module, symbol, function_op);
    if (!iree_string_view_is_empty(function->export_name)) {
      ++layout->export_count;
    }
    IREE_RETURN_IF_ERROR(loom_vm_module_layout_resolve_logical_signature(
        module, arena, function_op, &function->logical_signature));

    IREE_RETURN_IF_ERROR(loom_vm_module_layout_count_switch_targets(
        function_op, &function->switch_target_entry_count));
    if (function->switch_target_entry_count >
        UINT32_MAX - layout->switch_target_entry_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "VM module switch-target entry count exceeds u32");
    }
    layout->switch_target_entry_count += function->switch_target_entry_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_module_layout_populate_imports(
    loom_module_t* module, iree_arena_allocator_t* arena,
    loom_vm_module_layout_t* layout) {
  iree_host_size_t import_index = 0;
  for (iree_host_size_t symbol_index = 0; symbol_index < module->symbols.count;
       ++symbol_index) {
    const loom_symbol_t* symbol = &module->symbols.entries[symbol_index];
    loom_op_t* declaration_op = symbol->defining_op;
    if (declaration_op == NULL || !loom_low_func_decl_isa(declaration_op)) {
      continue;
    }

    loom_vm_module_import_layout_t* import =
        &layout->import_declarations[import_index++];
    const loom_string_id_t module_name_id =
        loom_low_func_decl_import_module(declaration_op);
    const loom_string_id_t symbol_name_id =
        loom_low_func_decl_import_symbol(declaration_op);
    *import = (loom_vm_module_import_layout_t){
        .declaration_op = declaration_op,
        .symbol_id = (loom_symbol_id_t)symbol_index,
        .module_name =
            loom_vm_module_layout_string_or_empty(module, module_name_id),
        .symbol_name = loom_vm_module_layout_string_or_empty(
            module, symbol_name_id != LOOM_STRING_ID_INVALID ? symbol_name_id
                                                             : symbol->name_id),
        .module_name_string_ordinal = UINT16_MAX,
        .symbol_name_string_ordinal = UINT16_MAX,
        .import_ordinal = UINT16_MAX,
        .flags = loom_low_func_decl_import_policy(declaration_op) ==
                         LOOM_LOW_FUNC_DECL_IMPORT_POLICY_OPTIONAL
                     ? IREE_VM_BYTECODE_IMPORT_FLAG_OPTIONAL
                     : 0,
    };
    if (iree_string_view_is_empty(import->module_name) ||
        iree_string_view_is_empty(import->symbol_name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VM runtime import module and symbol names must be nonempty");
    }
    IREE_RETURN_IF_ERROR(loom_vm_module_layout_resolve_logical_signature(
        module, arena, declaration_op, &import->logical_signature));
    layout->imports[import_index - 1] = import;
  }
  IREE_ASSERT_EQ(import_index, layout->import_declaration_count);
  return iree_ok_status();
}

static iree_status_t loom_vm_module_layout_assign_exports(
    iree_arena_allocator_t* arena, loom_vm_module_layout_t* layout) {
  for (iree_host_size_t i = 0; i < layout->function_count; ++i) {
    layout->functions[i].function_ordinal = (uint16_t)i;
  }

  if (layout->export_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, layout->export_count,
                                                 sizeof(*layout->exports),
                                                 (void**)&layout->exports));
  iree_host_size_t export_index = 0;
  for (iree_host_size_t i = 0; i < layout->function_count; ++i) {
    if (!iree_string_view_is_empty(layout->functions[i].export_name)) {
      layout->exports[export_index++] = &layout->functions[i];
    }
  }
  qsort(layout->exports, layout->export_count, sizeof(*layout->exports),
        loom_vm_module_layout_compare_export_names);
  for (iree_host_size_t i = 1; i < layout->export_count; ++i) {
    if (iree_string_view_equal(layout->exports[i - 1]->export_name,
                               layout->exports[i]->export_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VM export names must be unique");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_module_layout_assign_imports(
    iree_arena_allocator_t* arena, loom_vm_module_layout_t* layout) {
  if (layout->import_declaration_count == 0) return iree_ok_status();

  qsort(layout->imports, layout->import_declaration_count,
        sizeof(*layout->imports), loom_vm_module_layout_compare_imports);
  iree_host_size_t unique_count = 0;
  iree_host_size_t declaration_index = 0;
  while (declaration_index < layout->import_declaration_count) {
    const iree_host_size_t run_begin = declaration_index;
    iree_host_size_t run_end = run_begin + 1;
    while (run_end < layout->import_declaration_count &&
           loom_vm_module_layout_imports_equal(layout->imports[run_begin],
                                               layout->imports[run_end])) {
      ++run_end;
    }

    // A required alias dominates optional aliases: any valid linked program
    // must resolve the shared target, so all aliases can use one required row.
    uint16_t flags = layout->imports[run_begin]->flags;
    for (iree_host_size_t i = run_begin + 1; i < run_end; ++i) {
      flags &= layout->imports[i]->flags;
    }
    const uint16_t import_ordinal = (uint16_t)unique_count;
    const uint8_t target_kind =
        iree_any_bit_set(flags, IREE_VM_BYTECODE_IMPORT_FLAG_OPTIONAL)
            ? IREE_VM_ISA_CONTROL_CALL_TARGET_OPTIONAL_IMPORT
            : IREE_VM_ISA_CONTROL_CALL_TARGET_REQUIRED_IMPORT;
    for (iree_host_size_t i = run_begin; i < run_end; ++i) {
      loom_vm_module_import_layout_t* import = layout->imports[i];
      import->flags = flags;
      import->import_ordinal = import_ordinal;
      layout->call_targets_by_symbol[import->symbol_id] =
          (loom_vm_module_call_target_t){
              .kind = target_kind,
              .ordinal = import_ordinal,
          };
    }
    layout->imports[unique_count++] = layout->imports[run_begin];
    declaration_index = run_end;
  }
  layout->import_count = unique_count;

  iree_host_size_t group_count = 0;
  for (iree_host_size_t i = 0; i < unique_count; ++i) {
    if (i == 0 || !iree_string_view_equal(layout->imports[i - 1]->module_name,
                                          layout->imports[i]->module_name)) {
      ++group_count;
    }
  }
  if (group_count > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "VM import group count exceeds the u16 ordinal domain");
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, group_count, sizeof(*layout->import_groups),
      (void**)&layout->import_groups));
  layout->import_group_count = group_count;

  iree_host_size_t group_index = 0;
  iree_host_size_t import_index = 0;
  while (import_index < unique_count) {
    const iree_host_size_t group_begin = import_index;
    const iree_string_view_t module_name =
        layout->imports[group_begin]->module_name;
    while (import_index < unique_count &&
           iree_string_view_equal(module_name,
                                  layout->imports[import_index]->module_name)) {
      ++import_index;
    }
    layout->import_groups[group_index++] =
        (loom_vm_module_import_group_layout_t){
            .module_name_string_ordinal =
                layout->imports[group_begin]->module_name_string_ordinal,
            .import_count = (uint32_t)(import_index - group_begin),
        };
  }
  IREE_ASSERT_EQ(group_index, group_count);
  return iree_ok_status();
}

iree_status_t loom_vm_module_layout_build(loom_module_t* module,
                                          iree_arena_allocator_t* arena,
                                          loom_vm_module_layout_t* out_layout) {
  *out_layout = (loom_vm_module_layout_t){
      .module = module,
  };
  IREE_RETURN_IF_ERROR(
      loom_vm_module_layout_count(module, &out_layout->function_count,
                                  &out_layout->import_declaration_count));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, out_layout->function_count, sizeof(*out_layout->functions),
      (void**)&out_layout->functions));
  if (out_layout->import_declaration_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, out_layout->import_declaration_count,
                                  sizeof(*out_layout->import_declarations),
                                  (void**)&out_layout->import_declarations));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, out_layout->import_declaration_count,
        sizeof(*out_layout->imports), (void**)&out_layout->imports));
  }
  if (module->symbols.count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, module->symbols.count,
                                  sizeof(*out_layout->call_targets_by_symbol),
                                  (void**)&out_layout->call_targets_by_symbol));
    for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
      out_layout->call_targets_by_symbol[i] =
          (loom_vm_module_call_target_t){.ordinal = UINT16_MAX};
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_vm_module_layout_populate_functions(module, arena, out_layout));
  IREE_RETURN_IF_ERROR(
      loom_vm_module_layout_populate_imports(module, arena, out_layout));
  IREE_RETURN_IF_ERROR(loom_vm_module_layout_assign_exports(arena, out_layout));
  IREE_RETURN_IF_ERROR(loom_vm_module_type_tables_build(arena, out_layout));
  return loom_vm_module_layout_assign_imports(arena, out_layout);
}

bool loom_vm_module_layout_try_resolve_call_target(
    const loom_vm_module_layout_t* layout, loom_symbol_ref_t symbol_ref,
    loom_vm_module_call_target_t* out_target) {
  *out_target = (loom_vm_module_call_target_t){0};
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= layout->module->symbols.count) {
    return false;
  }
  const loom_vm_module_call_target_t target =
      layout->call_targets_by_symbol[symbol_ref.symbol_id];
  if (target.ordinal == UINT16_MAX) return false;
  *out_target = target;
  return true;
}
