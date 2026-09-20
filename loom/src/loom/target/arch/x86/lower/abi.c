// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/lower/abi.h"

#include "loom/target/arch/x86/sysv_abi.h"
#include "loom/target/facts.h"

static bool loom_x86_source_type_has_scalar_sysv_abi(loom_type_t type) {
  if (loom_type_is_buffer(type) || loom_type_is_view(type)) {
    return true;
  }
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  switch (loom_type_element_type(type)) {
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_I64:
      return true;
    default:
      return false;
  }
}

static bool loom_x86_source_signature_has_scalar_sysv_abi(
    loom_low_lower_context_t* context, iree_host_size_t argument_count,
    iree_host_size_t result_count) {
  loom_func_like_t source_function =
      loom_low_lower_context_source_function(context);
  uint16_t source_argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(source_function, &source_argument_count);
  IREE_ASSERT_EQ(source_argument_count, argument_count);
  IREE_ASSERT_EQ(source_function.op->result_count, result_count);
  const loom_module_t* module = loom_low_lower_context_module(context);
  for (uint16_t i = 0; i < source_argument_count; ++i) {
    if (!loom_x86_source_type_has_scalar_sysv_abi(
            loom_module_value_type(module, source_arguments[i]))) {
      return false;
    }
  }
  const loom_value_id_t* source_results =
      loom_op_const_results(source_function.op);
  for (uint16_t i = 0; i < source_function.op->result_count; ++i) {
    if (!loom_x86_source_type_has_scalar_sysv_abi(
            loom_module_value_type(module, source_results[i]))) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_x86_map_abi_layout(
    void* user_data, loom_low_lower_context_t* context,
    loom_low_lower_abi_layout_kind_t layout_kind,
    const loom_type_t* argument_types, iree_host_size_t argument_count,
    const loom_type_t* result_types, iree_host_size_t result_count,
    loom_named_attr_slice_t* out_abi_layout) {
  (void)user_data;
  *out_abi_layout = loom_named_attr_slice_empty();
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  if (layout_kind != LOOM_LOW_LOWER_ABI_LAYOUT_KIND_FUNC ||
      bundle->export_plan->abi_kind != LOOM_TARGET_ABI_OBJECT_FUNCTION) {
    return iree_ok_status();
  }
  if (!loom_x86_source_signature_has_scalar_sysv_abi(context, argument_count,
                                                     result_count)) {
    return iree_ok_status();
  }

  loom_x86_sysv_abi_layout_t layout = {0};
  bool supported = false;
  IREE_RETURN_IF_ERROR(loom_x86_sysv_abi_layout_build(
      loom_low_lower_context_descriptor_set(context), argument_types,
      argument_count, result_types, result_count,
      loom_low_lower_context_emission_arena(context), &layout, &supported));
  if (!supported) {
    return iree_ok_status();
  }

  loom_attribute_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_x86_sysv_abi_layout_make_attr(
      loom_low_lower_context_module(context), &layout, &attr));
  *out_abi_layout = loom_attr_as_dict(attr);
  return iree_ok_status();
}
