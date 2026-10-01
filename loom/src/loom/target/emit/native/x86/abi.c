// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/abi.h"

#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/x86/register_classes.h"

// SysV AMD64 integer-class arguments use RDI, RSI, RDX, RCX, R8, and R9.
// Unused source parameters consume positions even when they need no interval.
static const uint8_t kSysvArgumentRegisters[] = {7, 6, 2, 1, 8, 9};

static iree_status_t loom_x86_callable_reject(
    const loom_module_t* module, const loom_target_entry_t* entry,
    iree_string_view_t constraint, iree_diagnostic_emitter_t emitter) {
  const loom_target_bundle_t* bundle = loom_target_entry_bundle(entry);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(bundle->snapshot->name),
      loom_param_string(bundle->export_plan->name),
      loom_param_string(bundle->config->name),
      loom_param_string(entry->func_name),
      loom_param_string(loom_op_name(module, entry->func.op)),
      loom_param_string(constraint),
  };
  const loom_diagnostic_emission_t emission = {
      .op = entry->func.op,
      .error = LOOM_ERR_TARGET_032,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static bool loom_x86_callable_payload_matches_register(
    loom_type_t type, uint16_t register_class) {
  if (loom_type_is_buffer(type)) {
    return register_class == LOOM_X86_REGISTER_CLASS_GPR64;
  }
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  switch (loom_type_element_type(type)) {
    case LOOM_SCALAR_TYPE_I32:
      return register_class == LOOM_X86_REGISTER_CLASS_GPR32;
    case LOOM_SCALAR_TYPE_I64:
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
      return register_class == LOOM_X86_REGISTER_CLASS_GPR64;
    default:
      return false;
  }
}

static bool loom_x86_callable_type_supported(loom_type_t type) {
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_unit_count(type) != 1) {
    return false;
  }
  uint16_t register_class = loom_low_register_type_class_id(type);
  if (register_class != LOOM_X86_REGISTER_CLASS_GPR32 &&
      register_class != LOOM_X86_REGISTER_CLASS_GPR64) {
    return false;
  }
  const loom_type_t* value_type = loom_type_register_value_type(type);
  return value_type == NULL || loom_x86_callable_payload_matches_register(
                                   *value_type, register_class);
}

// The ABI signature states the logical boundary when raw physical carriers
// cannot distinguish it. Absence selects the register-width boundary, as for
// authored Low GPR32/GPR64 functions. This admission also applies to reparsed
// Low and therefore does not rely on the source-lowering invocation surviving.
static bool loom_x86_callable_layout_supported(const loom_module_t* module,
                                               loom_func_like_t function) {
  const loom_op_t* function_op = function.op;
  const loom_named_attr_slice_t layout =
      loom_low_func_def_abi_layout(function_op);
  if (layout.count == 0) {
    return true;
  }
  if (layout.count != 1 || layout.entries[0].value.kind != LOOM_ATTR_TYPE ||
      !iree_string_view_equal(
          loom_string_table_get(&module->strings, layout.entries[0].name_id),
          IREE_SV("signature"))) {
    return false;
  }
  const loom_type_t signature =
      loom_type_table_get(&module->types, layout.entries[0].value.type_id);
  if (!loom_type_is_function(signature)) {
    return false;
  }
  const loom_func_type_data_t* data = loom_type_func_data(signature);
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  if (data->arg_count != argument_count ||
      data->result_count != function_op->result_count) {
    return false;
  }
  const loom_value_id_t* results = loom_op_const_results(function_op);
  const iree_host_size_t count =
      (iree_host_size_t)data->arg_count + data->result_count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_value_id_t value =
        i < argument_count ? arguments[i] : results[i - argument_count];
    const loom_type_t carrier = loom_module_value_type(module, value);
    if (!loom_low_type_is_register(carrier) ||
        loom_low_register_type_unit_count(carrier) != 1 ||
        !loom_x86_callable_payload_matches_register(
            data->types[i], loom_low_register_type_class_id(carrier))) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_x86_function_abi_prepare(loom_module_t* module,
                                            const loom_target_entry_t* entry,
                                            iree_diagnostic_emitter_t emitter,
                                            bool* out_accepted,
                                            loom_x86_function_abi_t* out_abi) {
  *out_accepted = false;
  *out_abi = (loom_x86_function_abi_t){0};
  if (!iree_string_view_equal(
          loom_target_entry_bundle(entry)->export_plan->calling_convention,
          IREE_SV("sysv"))) {
    return loom_x86_callable_reject(
        module, entry,
        IREE_SV("native x86 supports the sysv calling convention"), emitter);
  }
  if (!loom_x86_callable_layout_supported(module, entry->func)) {
    return loom_x86_callable_reject(
        module, entry,
        IREE_SV("native x86 ABI signature requires scalar i32, i64, or "
                "pointers matching the register boundary"),
        emitter);
  }
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(entry->func, &argument_count);
  if (argument_count > IREE_ARRAYSIZE(kSysvArgumentRegisters) ||
      entry->func.op->result_count > 1) {
    return loom_x86_callable_reject(
        module, entry,
        IREE_SV("native x86 supports at most six register arguments and one "
                "result"),
        emitter);
  }
  iree_host_size_t fixed_value_count = 0;
  bool arguments_supported = true;
  for (uint16_t i = 0; i < argument_count && arguments_supported; ++i) {
    arguments_supported = loom_x86_callable_type_supported(
        loom_module_value_type(module, arguments[i]));
    if (arguments_supported &&
        !loom_value_has_no_uses(loom_module_value(module, arguments[i]))) {
      out_abi->fixed_values[fixed_value_count++] =
          (loom_low_allocation_fixed_value_t){
              .value_id = arguments[i],
              .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
              .location_base = kSysvArgumentRegisters[i],
              .location_count = 1,
          };
    }
  }
  if (!arguments_supported) {
    return loom_x86_callable_reject(
        module, entry,
        IREE_SV("native x86 arguments require scalar i32, i64, or pointers"),
        emitter);
  }
  if (entry->func.op->result_count &&
      !loom_x86_callable_type_supported(loom_module_value_type(
          module, loom_op_const_results(entry->func.op)[0]))) {
    return loom_x86_callable_reject(
        module, entry,
        IREE_SV("native x86 results require scalar i32, i64, or pointers"),
        emitter);
  }
  out_abi->fixed_value_count = fixed_value_count;
  *out_accepted = true;
  return iree_ok_status();
}
