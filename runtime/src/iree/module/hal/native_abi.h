// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULE_HAL_NATIVE_ABI_H_
#define IREE_MODULE_HAL_NATIVE_ABI_H_

#include "iree/vm/native_call.h"

// Semantic grouping over an admitted native record. This cold metadata adds
// optional cardinality constraints to the VM's physical layout declaration.
// Neither source op ordinals nor these nodes are consulted during execution.
enum iree_hal_vm_native_node_kind_e {
  IREE_HAL_VM_NATIVE_VALUE = 0,
  IREE_HAL_VM_NATIVE_RECORD = 1,
  IREE_HAL_VM_NATIVE_RANGE = 2,
  IREE_HAL_VM_NATIVE_OPTIONAL = 3,
};

typedef struct iree_hal_vm_native_node_t {
  // Semantic value, fixed record, range, or zero-or-one optional group.
  uint8_t kind;
  // True if this node or a descendant has an optional cardinality constraint.
  bool has_optional;
  // Number of immediate semantic children.
  uint16_t child_count;
  // First child ordinal in the shared generated edge table.
  uint16_t child_base;
  // Fixed physical words occupied by this node in its containing record.
  uint16_t word_count;
} iree_hal_vm_native_node_t;

#ifdef __cplusplus
extern "C" {
#endif

// Checks optional presence after generic native record admission. Geometry,
// physical selectors and exact field types have already passed VM validation.
iree_status_t iree_hal_vm_native_verify_optionals(const uint32_t* record,
                                                  uint16_t root_node);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_MODULE_HAL_NATIVE_ABI_H_
