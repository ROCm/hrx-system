// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/vm/module_layout.h"

#include <stdlib.h>

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
    const loom_module_t* module, iree_host_size_t* out_function_count) {
  *out_function_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_op_t* defining_op = module->symbols.entries[i].defining_op;
    if (defining_op == NULL) continue;
    if (loom_low_func_decl_isa(defining_op)) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "VM function imports are not implemented");
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
    function->export_name =
        loom_vm_module_layout_export_name(module, symbol, function_op);
    if (!iree_string_view_is_empty(function->export_name)) {
      ++layout->export_count;
    }
    const loom_named_attr_slice_t abi_layout =
        loom_low_func_def_abi_layout(function_op);
    if (abi_layout.count == 0) {
      loom_func_like_t function_like = loom_func_like_cast(module, function_op);
      uint16_t argument_count = 0;
      loom_func_like_arg_ids(function_like, &argument_count);
      if (argument_count != 0 ||
          loom_low_func_def_results(function_op).count != 0) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "nonempty VM function is missing its preserved "
                                "logical ABI signature");
      }
      loom_func_type_data_t* empty_signature = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, sizeof(*empty_signature),
                                               (void**)&empty_signature));
      *empty_signature = (loom_func_type_data_t){0};
      function->logical_signature = loom_type_function(empty_signature);
    } else {
      IREE_RETURN_IF_ERROR(loom_vm_call_abi_layout_resolve_signature(
          module, abi_layout, &function->logical_signature));
    }

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

iree_status_t loom_vm_module_layout_build(loom_module_t* module,
                                          iree_arena_allocator_t* arena,
                                          loom_vm_module_layout_t* out_layout) {
  *out_layout = (loom_vm_module_layout_t){
      .module = module,
  };
  IREE_RETURN_IF_ERROR(
      loom_vm_module_layout_count(module, &out_layout->function_count));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, out_layout->function_count, sizeof(*out_layout->functions),
      (void**)&out_layout->functions));
  IREE_RETURN_IF_ERROR(
      loom_vm_module_layout_populate_functions(module, arena, out_layout));
  IREE_RETURN_IF_ERROR(loom_vm_module_layout_assign_exports(arena, out_layout));
  return loom_vm_module_type_tables_build(arena, out_layout);
}
