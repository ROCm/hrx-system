// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/module/hal/native_abi.h"

#include "iree/module/hal/native_schema.h"

static iree_status_t iree_hal_vm_native_verify_node(const uint32_t* root,
                                                    const uint32_t* record,
                                                    uint16_t ordinal) {
  const iree_hal_vm_native_node_t* node = &kIreeHalNativeNodes[ordinal];
  if (!node->has_optional) {
    return iree_ok_status();
  }
  if (node->kind == IREE_HAL_VM_NATIVE_OPTIONAL && record[0] > 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL optional group contains %u records",
                            record[0]);
  }
  const uint16_t* children = &kIreeHalNativeChildren[node->child_base];
  iree_status_t status = iree_ok_status();
  if (node->kind == IREE_HAL_VM_NATIVE_RECORD) {
    for (uint16_t i = 0; iree_status_is_ok(status) && i < node->child_count;
         ++i) {
      status = iree_hal_vm_native_verify_node(root, record, children[i]);
      record += kIreeHalNativeNodes[children[i]].word_count;
    }
  } else {
    const uint32_t* elements = root + record[1];
    const uint16_t width = kIreeHalNativeNodes[children[0]].word_count;
    for (uint32_t i = 0; iree_status_is_ok(status) && i < record[0]; ++i) {
      status = iree_hal_vm_native_verify_node(root, elements + i * width,
                                              children[0]);
    }
  }
  return status;
}

iree_status_t iree_hal_vm_native_verify_optionals(const uint32_t* record,
                                                  uint16_t root_node) {
  return iree_hal_vm_native_verify_node(record, record, root_node);
}
