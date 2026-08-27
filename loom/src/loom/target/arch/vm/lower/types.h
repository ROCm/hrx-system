// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// VM target-Low type helpers.

#ifndef LOOM_TARGET_ARCH_VM_LOWER_TYPES_H_
#define LOOM_TARGET_ARCH_VM_LOWER_TYPES_H_

#include "loom/ir/types.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/target/registers.h"

// Returns true and assigns the carried scalar type when |type| is exactly one
// Core VM value register.
static inline bool loom_vm_value_register_scalar_type(
    loom_type_t type, loom_scalar_type_t* out_scalar_type) {
  if (!loom_low_type_is_register(type) ||
      loom_low_register_type_descriptor_set_stable_id(type) !=
          VM_CORE_DESCRIPTOR_SET_ID ||
      loom_low_register_type_class_id(type) != VM_CORE_REG_CLASS_ID_VALUE ||
      loom_low_register_type_unit_count(type) != 1) {
    return false;
  }
  const loom_type_t* value_type = loom_type_register_value_type(type);
  if (value_type == NULL || !loom_type_is_scalar(*value_type)) return false;
  *out_scalar_type = loom_type_element_type(*value_type);
  return true;
}

#endif  // LOOM_TARGET_ARCH_VM_LOWER_TYPES_H_
