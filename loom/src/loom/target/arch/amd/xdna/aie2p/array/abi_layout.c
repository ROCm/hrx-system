// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/array/abi_layout.h"

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

static const iree_string_view_t loom_aie2p_array_binding_count_key(void) {
  return IREE_SV("binding_count");
}

bool loom_aie2p_array_abi_layout_validate(
    const loom_module_t* module, loom_named_attr_slice_t attributes,
    loom_aie2p_array_abi_layout_issue_t* out_issue) {
  *out_issue = (loom_aie2p_array_abi_layout_issue_t){0};
  for (iree_host_size_t i = 0; i < attributes.count; ++i) {
    const loom_named_attr_t* entry = &attributes.entries[i];
    const iree_string_view_t field_name =
        loom_string_table_get(&module->strings, entry->name_id);
    if (!iree_string_view_equal(field_name,
                                loom_aie2p_array_binding_count_key())) {
      *out_issue = (loom_aie2p_array_abi_layout_issue_t){
          .kind = LOOM_AIE2P_ARRAY_ABI_LAYOUT_ISSUE_UNEXPECTED_FIELD,
          .field_name = field_name,
      };
      return false;
    }
    if (entry->value.kind != LOOM_ATTR_I64) {
      *out_issue = (loom_aie2p_array_abi_layout_issue_t){
          .kind = LOOM_AIE2P_ARRAY_ABI_LAYOUT_ISSUE_BINDING_COUNT_KIND,
          .field_name = field_name,
          .attribute = entry->value,
      };
      return false;
    }
    if (entry->value.i64 < 0 ||
        entry->value.i64 > LOOM_AIE2P_ARRAY_MAX_BINDING_SLOT_COUNT) {
      *out_issue = (loom_aie2p_array_abi_layout_issue_t){
          .kind = LOOM_AIE2P_ARRAY_ABI_LAYOUT_ISSUE_BINDING_COUNT_RANGE,
          .field_name = field_name,
          .attribute = entry->value,
      };
      return false;
    }
  }
  return true;
}

loom_aie2p_array_abi_layout_t loom_aie2p_array_abi_layout_from_verified(
    const loom_op_t* function_op) {
  if (!loom_low_func_def_has_abi_layout(function_op)) {
    return (loom_aie2p_array_abi_layout_t){0};
  }
  const loom_named_attr_slice_t attributes =
      loom_low_func_def_abi_layout(function_op);
  if (attributes.count == 0) {
    return (loom_aie2p_array_abi_layout_t){0};
  }

  const loom_named_attr_t* entry = &attributes.entries[0];
  return (loom_aie2p_array_abi_layout_t){
      .binding_slot_count = (uint32_t)entry->value.i64,
      .has_binding_slot_count = true,
  };
}

iree_status_t loom_aie2p_array_abi_layout_make_attr(
    loom_module_t* module, uint32_t binding_slot_count,
    loom_attribute_t* out_attr) {
  *out_attr = loom_attr_absent();

  loom_string_id_t binding_count_key = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, loom_aie2p_array_binding_count_key(), &binding_count_key));
  const loom_named_attr_t entry = {
      .name_id = binding_count_key,
      .value = loom_attr_i64(binding_slot_count),
  };
  return loom_module_make_canonical_attr_dict(
      module, loom_make_named_attr_slice(&entry, 1), out_attr);
}
