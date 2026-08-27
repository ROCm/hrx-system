// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/vm/module_layout.h"

#include <stdlib.h>

#include "iree/vm/bytecode/wire/module_format.h"
#include "loom/codegen/low/function.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/target/registers.h"

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

static iree_status_t loom_vm_module_layout_signature_kind(
    const loom_module_t* module, loom_value_id_t value_id, uint16_t* out_kind) {
  const loom_type_t register_type = loom_module_value_type(module, value_id);
  const loom_type_t* value_type = loom_type_register_value_type(register_type);
  if (!loom_low_type_is_register(register_type) ||
      loom_low_register_type_descriptor_set_stable_id(register_type) !=
          VM_CORE_DESCRIPTOR_SET_ID ||
      loom_low_register_type_class_id(register_type) !=
          VM_CORE_REG_CLASS_ID_VALUE ||
      loom_low_register_type_unit_count(register_type) != 1 ||
      value_type == NULL || !loom_type_is_scalar(*value_type)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "VM function signature value %u is not one scalar vm.value register",
        (unsigned)value_id);
  }

  switch (loom_type_element_type(*value_type)) {
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
    case LOOM_SCALAR_TYPE_I64:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_I64;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I1:
    case LOOM_SCALAR_TYPE_I8:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_I8;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I16:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_I16;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_I32:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_I32;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F8E4M3:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_F8E4M3FN;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F8E5M2:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_F8E5M2;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F16:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_F16;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_BF16:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_BF16;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F32:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_F32;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F64:
      *out_kind = IREE_VM_BYTECODE_SIGNATURE_KIND_F64;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "scalar type is not supported by the VM ABI");
  }
}

static int loom_vm_module_layout_compare_kind_lists(const uint16_t* lhs,
                                                    const uint16_t* rhs,
                                                    uint16_t count) {
  for (uint16_t i = 0; i < count; ++i) {
    if (lhs[i] < rhs[i]) return -1;
    if (lhs[i] > rhs[i]) return 1;
  }
  return 0;
}

static int loom_vm_module_layout_compare_callable_types(const void* lhs_ptr,
                                                        const void* rhs_ptr) {
  const loom_vm_module_function_layout_t* lhs =
      *(loom_vm_module_function_layout_t* const*)lhs_ptr;
  const loom_vm_module_function_layout_t* rhs =
      *(loom_vm_module_function_layout_t* const*)rhs_ptr;
  if (lhs->argument_count < rhs->argument_count) return -1;
  if (lhs->argument_count > rhs->argument_count) return 1;
  int comparison = loom_vm_module_layout_compare_kind_lists(
      lhs->argument_kinds, rhs->argument_kinds, lhs->argument_count);
  if (comparison != 0) return comparison;
  if (lhs->result_count < rhs->result_count) return -1;
  if (lhs->result_count > rhs->result_count) return 1;
  return loom_vm_module_layout_compare_kind_lists(
      lhs->result_kinds, rhs->result_kinds, lhs->result_count);
}

static int loom_vm_module_layout_compare_export_names(const void* lhs_ptr,
                                                      const void* rhs_ptr) {
  const loom_vm_module_function_layout_t* lhs =
      *(loom_vm_module_function_layout_t* const*)lhs_ptr;
  const loom_vm_module_function_layout_t* rhs =
      *(loom_vm_module_function_layout_t* const*)rhs_ptr;
  return iree_string_view_compare(lhs->export_name, rhs->export_name);
}

static iree_status_t loom_vm_module_layout_count(
    const loom_module_t* module, iree_host_size_t* out_function_count,
    iree_host_size_t* out_signature_descriptor_count) {
  *out_function_count = 0;
  *out_signature_descriptor_count = 0;
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
    loom_func_like_t function =
        loom_func_like_cast(module, (loom_op_t*)defining_op);
    uint16_t argument_count = 0;
    (void)loom_func_like_arg_ids(function, &argument_count);
    const loom_value_slice_t results = loom_low_func_def_results(defining_op);
    iree_host_size_t descriptor_count = 0;
    if (!iree_host_size_checked_add(argument_count, results.count,
                                    &descriptor_count) ||
        !iree_host_size_checked_add(*out_signature_descriptor_count,
                                    descriptor_count,
                                    out_signature_descriptor_count) ||
        *out_signature_descriptor_count > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM signature descriptor count exceeds u32");
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
    loom_module_t* module, uint16_t* signature_kinds,
    loom_vm_module_layout_t* layout) {
  iree_host_size_t function_index = 0;
  iree_host_size_t signature_kind_index = 0;
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

    loom_func_like_t function_like = loom_func_like_cast(module, function_op);
    const loom_value_id_t* arguments =
        loom_func_like_arg_ids(function_like, &function->argument_count);
    function->argument_kinds = function->argument_count != 0
                                   ? signature_kinds + signature_kind_index
                                   : NULL;
    for (uint16_t i = 0; i < function->argument_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_vm_module_layout_signature_kind(
          module, arguments[i], &signature_kinds[signature_kind_index++]));
    }

    const loom_value_slice_t results = loom_low_func_def_results(function_op);
    if (results.count > UINT16_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM function result count exceeds u16");
    }
    function->result_count = (uint16_t)results.count;
    function->result_kinds = function->result_count != 0
                                 ? signature_kinds + signature_kind_index
                                 : NULL;
    for (uint16_t i = 0; i < function->result_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_vm_module_layout_signature_kind(
          module, results.values[i], &signature_kinds[signature_kind_index++]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_module_layout_assign_ordinals(
    iree_arena_allocator_t* arena, loom_vm_module_layout_t* layout) {
  for (iree_host_size_t i = 0; i < layout->function_count; ++i) {
    layout->functions[i].function_ordinal = (uint16_t)i;
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, layout->function_count, sizeof(*layout->callable_types),
      (void**)&layout->callable_types));
  for (iree_host_size_t i = 0; i < layout->function_count; ++i) {
    layout->callable_types[i] = &layout->functions[i];
  }
  qsort(layout->callable_types, layout->function_count,
        sizeof(*layout->callable_types),
        loom_vm_module_layout_compare_callable_types);

  loom_vm_module_function_layout_t* previous = NULL;
  for (iree_host_size_t i = 0; i < layout->function_count; ++i) {
    loom_vm_module_function_layout_t* function = layout->callable_types[i];
    if (previous == NULL || loom_vm_module_layout_compare_callable_types(
                                &previous, &function) != 0) {
      if (layout->callable_type_count == 65536u) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "VM callable type count exceeds the u16 ordinal domain");
      }
      layout->callable_types[layout->callable_type_count++] = function;
      layout->signature_descriptor_count +=
          function->argument_count + function->result_count;
      previous = function;
    }
    function->callable_type_ordinal =
        (uint16_t)(layout->callable_type_count - 1);
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
  iree_host_size_t signature_descriptor_count = 0;
  IREE_RETURN_IF_ERROR(loom_vm_module_layout_count(
      module, &out_layout->function_count, &signature_descriptor_count));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, out_layout->function_count, sizeof(*out_layout->functions),
      (void**)&out_layout->functions));
  uint16_t* signature_kinds = NULL;
  if (signature_descriptor_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, signature_descriptor_count, sizeof(*signature_kinds),
        (void**)&signature_kinds));
  }
  IREE_RETURN_IF_ERROR(loom_vm_module_layout_populate_functions(
      module, signature_kinds, out_layout));
  return loom_vm_module_layout_assign_ordinals(arena, out_layout);
}
