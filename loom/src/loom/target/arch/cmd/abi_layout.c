// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/abi_layout.h"

#include "loom/ir/module.h"

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
