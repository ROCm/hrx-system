// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/abi/layout.h"

#include "loom/ir/module.h"

iree_status_t loom_vm_call_abi_layout_make_attr(
    loom_module_t* module, const loom_type_t* argument_types,
    iree_host_size_t argument_count, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_attribute_t* out_attr) {
  *out_attr = loom_attr_absent();
  if (argument_count > UINT16_MAX || result_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM logical call signature exceeds u16");
  }

  loom_type_t signature = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_module_intern_function_type(
      module, argument_types, (uint16_t)argument_count, result_types,
      (uint16_t)result_count, &signature));
  loom_type_id_t signature_id = LOOM_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_type_id(module, signature, &signature_id));
  loom_string_id_t signature_key = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, IREE_SV("signature"), &signature_key));
  const loom_named_attr_t entries[] = {
      {
          .name_id = signature_key,
          .value = loom_attr_type(signature_id),
      },
  };
  return loom_module_make_canonical_attr_dict(
      module, loom_make_named_attr_slice(entries, IREE_ARRAYSIZE(entries)),
      out_attr);
}
