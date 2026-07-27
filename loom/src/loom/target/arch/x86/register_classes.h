// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Stable x86 register-class projection for descriptor-set-local IDs.
//
// x86 descriptor views can share physical register classes while assigning
// different dense row IDs inside each descriptor set. This header owns the
// target-level logical register-class identities and maps them to the active
// descriptor view by stable register-class names.

#ifndef LOOM_TARGET_ARCH_X86_REGISTER_CLASSES_H_
#define LOOM_TARGET_ARCH_X86_REGISTER_CLASSES_H_

#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_x86_register_class_e {
  LOOM_X86_REGISTER_CLASS_GPR32 = 0,
  LOOM_X86_REGISTER_CLASS_GPR64 = 1,
  LOOM_X86_REGISTER_CLASS_XMM = 2,
  LOOM_X86_REGISTER_CLASS_YMM = 3,
  LOOM_X86_REGISTER_CLASS_ZMM = 4,
  LOOM_X86_REGISTER_CLASS_K = 5,
} loom_x86_register_class_t;

// Returns the stable descriptor register-class name for |register_class|.
iree_status_t loom_x86_register_class_name(
    loom_x86_register_class_t register_class, iree_string_view_t* out_name);

// Maps a SIMD/vector architectural width to its logical x86 register class.
// Returns false for unsupported widths and stores GPR32 in |out_register_class|
// so callers never observe an uninitialized value.
bool loom_x86_register_class_for_vector_bit_width(
    uint32_t vector_bit_width, loom_x86_register_class_t* out_register_class);

// Looks up |register_class| in |descriptor_set| and returns the
// descriptor-set-local register-class ID.
iree_status_t loom_x86_descriptor_set_register_class_id(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_x86_register_class_t register_class,
    uint16_t* out_descriptor_reg_class_id);

// Resolves a descriptor-set-local register-class ID to a logical x86 class.
iree_status_t loom_x86_descriptor_set_logical_register_class(
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t descriptor_reg_class_id,
    loom_x86_register_class_t* out_register_class);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_REGISTER_CLASSES_H_
