// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/lower.h"

#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/vm/contracts/core.h"
#include "loom/target/arch/vm/contracts/core_lower_rules.h"
#include "loom/target/arch/vm/descriptors/descriptors.h"
#include "loom/target/arch/vm/hal_lower.h"
#include "loom/target/arch/vm/types.h"

static bool loom_vm_source_type_supported(void* user_data,
                                          const loom_module_t* module,
                                          loom_type_t type) {
  return loom_vm_type_reference_key(module, type) != NULL ||
         (loom_type_is_scalar(type) &&
          loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_ADDRESS |
                                            LOOM_SCALAR_TYPE_SET_INTEGER |
                                            LOOM_SCALAR_TYPE_SET_FLOAT,
                                        loom_type_element_type(type)));
}

static iree_status_t loom_vm_map_type(void* user_data,
                                      loom_low_lower_context_t* context,
                                      const loom_op_t* source_op,
                                      loom_type_t source_type,
                                      loom_type_t* out_low_type) {
  (void)user_data;
  // Both address domains use the target's 64-bit value carrier. Signedness is
  // expressed by the selected operations, not a distinct physical register.
  if (loom_type_is_scalar(source_type) &&
      (loom_type_element_type(source_type) == LOOM_SCALAR_TYPE_INDEX ||
       loom_type_element_type(source_type) == LOOM_SCALAR_TYPE_OFFSET)) {
    source_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  }
  if (loom_vm_source_type_supported(
          user_data, loom_low_lower_context_module(context), source_type)) {
    return loom_low_lower_make_typed_register_type(
        context,
        loom_type_is_scalar(source_type) ? VM_CORE_REG_CLASS_ID_VALUE
                                         : VM_CORE_REG_CLASS_ID_REF,
        1, source_type, out_low_type);
  }
  return iree_ok_status();
}

#include "loom/target/arch/vm/contracts/tables.inl"

// The source symbol distinguishes immutable bytes from process value globals.
// Keep that symbol in Low; the module writer assigns its final data ordinal.
static iree_status_t loom_vm_select_op(void* user_data,
                                       loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  if (loom_op_dialect_id(source_op->kind) == LOOM_DIALECT_HAL) {
    return loom_vm_hal_select(context, source_op, out_plan);
  }
  if (!loom_global_load_isa(source_op)) {
    return iree_ok_status();
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_symbol_ref_t symbol = loom_global_load_global(source_op);
  const loom_op_t* definition =
      module->symbols.entries[symbol.symbol_id].defining_op;
  if (loom_global_rodata_def_isa(definition)) {
    loom_type_t result_type;
    IREE_RETURN_IF_ERROR(loom_low_lower_make_typed_register_type(
        context, VM_CORE_REG_CLASS_ID_REF, 1, loom_type_buffer(),
        &result_type));
    IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
        context, loom_global_load_result(source_op).values[0], result_type));
    *out_plan = loom_low_lower_plan_make(
        VM_CORE_DESCRIPTOR_REF_BUFFER_RODATA_LOAD, NULL);
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_emit_op(void* user_data,
                                     loom_low_lower_context_t* context,
                                     const loom_op_t* source_op,
                                     loom_low_lower_plan_t plan) {
  if (loom_op_dialect_id(source_op->kind) == LOOM_DIALECT_HAL) {
    return loom_vm_hal_emit(context, source_op, plan);
  }
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_string_id_t name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_builder_intern_string(builder, IREE_SV("rodata_u16"), &name));
  const loom_named_attr_t attr = {
      .name_id = name,
      .value = loom_attr_symbol(loom_global_load_global(source_op)),
  };
  const loom_type_t result_type = loom_low_lower_value_binding_type(
      context, loom_global_load_result(source_op).values[0]);
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor =
          &loom_low_lower_context_descriptor_set(context)->descriptors[plan.id],
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, NULL, 0,
      (loom_named_attr_slice_t){.entries = &attr, .count = 1}, &result_type, 1,
      NULL, 0, source_op->location, &low_op));
  return loom_low_lower_bind_value(context,
                                   loom_global_load_result(source_op).values[0],
                                   loom_op_results(low_op)[0]);
}

static const loom_low_lower_policy_t kPolicy = {
    .name = IREE_SVL("vm-lower"),
    .import_decl_kind = LOOM_LOW_FUNC_DECL_IMPORT_KIND_NATIVE,
    .error_catalog = &loom_error_catalog_core,
    .source_type_supported = {.fn = loom_vm_source_type_supported},
    .map_type = {.fn = loom_vm_map_type},
    .contract = LOOM_VM_CORE_CONTRACT,
    .select_op = {.fn = loom_vm_select_op},
    .emit_op = {.fn = loom_vm_emit_op},
};

void loom_vm_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {.contract_set_key = IREE_SVL("vm.core"), .policy = &kPolicy},
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
