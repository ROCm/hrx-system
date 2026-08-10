// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/abi_layout.h"

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/types.h"

typedef struct loom_cmd_abi_layout_attr_keys_t {
  loom_string_id_t fixed_buffer_count;
  loom_string_id_t rebindable_binding_count;
  loom_string_id_t executable_count;
  loom_string_id_t entry_count;
} loom_cmd_abi_layout_attr_keys_t;

static iree_status_t loom_cmd_abi_layout_intern_attr_keys(
    loom_module_t* module, loom_cmd_abi_layout_attr_keys_t* out_keys) {
  *out_keys = (loom_cmd_abi_layout_attr_keys_t){0};
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("fixed_buffer_count"), &out_keys->fixed_buffer_count));
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, IREE_SV("rebindable_binding_count"),
                                &out_keys->rebindable_binding_count));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("executable_count"), &out_keys->executable_count));
  return loom_module_intern_string(module, IREE_SV("entry_count"),
                                   &out_keys->entry_count);
}

iree_status_t loom_cmd_abi_layout_make_attr(loom_module_t* module,
                                            const loom_cmd_abi_layout_t* layout,
                                            loom_attribute_t* out_attr) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(layout);
  IREE_ASSERT_ARGUMENT(out_attr);
  *out_attr = loom_attr_absent();

  loom_cmd_abi_layout_attr_keys_t keys = {0};
  IREE_RETURN_IF_ERROR(loom_cmd_abi_layout_intern_attr_keys(module, &keys));
  const loom_named_attr_t entries[] = {
      {.name_id = keys.fixed_buffer_count,
       .value = loom_attr_i64(layout->fixed_buffer_count)},
      {.name_id = keys.rebindable_binding_count,
       .value = loom_attr_i64(layout->rebindable_binding_count)},
      {.name_id = keys.executable_count,
       .value = loom_attr_i64(layout->executable_count)},
      {.name_id = keys.entry_count,
       .value = loom_attr_i64(layout->entry_count)},
  };
  return loom_module_make_canonical_attr_dict(
      module, loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)),
      out_attr);
}

static const loom_attribute_t* loom_cmd_abi_layout_find_attr(
    loom_named_attr_slice_t attrs, loom_string_id_t key_id) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    if (attrs.entries[i].name_id == key_id) return &attrs.entries[i].value;
  }
  return NULL;
}

static iree_status_t loom_cmd_abi_layout_decode_count(
    loom_named_attr_slice_t attrs, loom_string_id_t key_id,
    iree_string_view_t key, uint32_t* out_value) {
  const loom_attribute_t* attr = loom_cmd_abi_layout_find_attr(attrs, key_id);
  if (attr == NULL || attr->kind != LOOM_ATTR_I64) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command ABI layout has a missing or malformed '%.*s' field",
        (int)key.size, key.data);
  }
  const int64_t value = loom_attr_as_i64(*attr);
  if (value < 0 || (uint64_t)value > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command ABI layout field '%.*s' is outside the u32 range",
        (int)key.size, key.data);
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

iree_status_t loom_cmd_abi_layout_from_low(const loom_module_t* module,
                                           const loom_op_t* function_op,
                                           loom_cmd_abi_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(function_op);
  IREE_ASSERT_ARGUMENT(out_layout);
  *out_layout = (loom_cmd_abi_layout_t){0};
  if (!loom_low_func_def_isa(function_op) ||
      loom_low_func_def_abi(function_op) != LOOM_TARGET_ABI_COMMAND_PROGRAM) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected a command_program low.func.def");
  }

  const loom_named_attr_slice_t attrs =
      loom_low_func_def_abi_layout(function_op);
  if (attrs.count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command program is missing its ABI layout");
  }
  const loom_cmd_abi_layout_attr_keys_t keys = {
      .fixed_buffer_count =
          loom_module_lookup_string(module, IREE_SV("fixed_buffer_count")),
      .rebindable_binding_count = loom_module_lookup_string(
          module, IREE_SV("rebindable_binding_count")),
      .executable_count =
          loom_module_lookup_string(module, IREE_SV("executable_count")),
      .entry_count = loom_module_lookup_string(module, IREE_SV("entry_count")),
  };
  IREE_RETURN_IF_ERROR(loom_cmd_abi_layout_decode_count(
      attrs, keys.fixed_buffer_count, IREE_SV("fixed_buffer_count"),
      &out_layout->fixed_buffer_count));
  IREE_RETURN_IF_ERROR(loom_cmd_abi_layout_decode_count(
      attrs, keys.rebindable_binding_count, IREE_SV("rebindable_binding_count"),
      &out_layout->rebindable_binding_count));
  IREE_RETURN_IF_ERROR(loom_cmd_abi_layout_decode_count(
      attrs, keys.executable_count, IREE_SV("executable_count"),
      &out_layout->executable_count));
  return loom_cmd_abi_layout_decode_count(attrs, keys.entry_count,
                                          IREE_SV("entry_count"),
                                          &out_layout->entry_count);
}
