// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared x86 register-class identities.
//
// All x86 descriptor views retain register-class IDs from the shared storage
// namespace. A view leaves unavailable classes absent at their storage slots;
// register type construction checks availability in the selected view.

#ifndef LOOM_TARGET_ARCH_X86_REGISTER_CLASSES_H_
#define LOOM_TARGET_ARCH_X86_REGISTER_CLASSES_H_

#include "loom/ir/scalar_type.h"
#include "loom/target/arch/x86/descriptors/avx512_features_descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_x86_register_class_e {
  LOOM_X86_REGISTER_CLASS_REF = X86_AVX512_FEATURES_CORE_REG_CLASS_ID_REF,
  LOOM_X86_REGISTER_CLASS_GPR32 = X86_AVX512_FEATURES_CORE_REG_CLASS_ID_GPR32,
  LOOM_X86_REGISTER_CLASS_GPR64 = X86_AVX512_FEATURES_CORE_REG_CLASS_ID_GPR64,
  LOOM_X86_REGISTER_CLASS_XMM = X86_AVX512_FEATURES_CORE_REG_CLASS_ID_XMM,
  LOOM_X86_REGISTER_CLASS_YMM = X86_AVX512_FEATURES_CORE_REG_CLASS_ID_YMM,
  LOOM_X86_REGISTER_CLASS_ZMM = X86_AVX512_FEATURES_CORE_REG_CLASS_ID_ZMM,
  LOOM_X86_REGISTER_CLASS_K = X86_AVX512_FEATURES_CORE_REG_CLASS_ID_K,
} loom_x86_register_class_t;

// Returns the declared width of a scalar element carried in an x86 vector
// register. Predicate lanes have representation-specific widths and return
// false here.
bool loom_x86_register_vector_element_bit_width(loom_scalar_type_t scalar_type,
                                                uint32_t* out_bit_width);

// Maps a logical vector payload width to its physical x86 register class.
// Payloads narrower than 128 bits occupy the corresponding low XMM bits.
// Returns false for unsupported widths and stores GPR32 in
// |out_register_class| so callers never observe an uninitialized value.
bool loom_x86_register_class_for_vector_bit_width(
    uint32_t vector_bit_width, loom_x86_register_class_t* out_register_class);

// Returns the logical class for a verified x86 descriptor register-class ID.
// Ordinary classes preserve their identity; fixed GPR aliases use GPR32 or
// GPR64 according to their width.
loom_x86_register_class_t loom_x86_logical_register_class(
    uint16_t descriptor_reg_class_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_REGISTER_CLASSES_H_
