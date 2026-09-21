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

#include "loom/target/arch/x86/descriptors/avx512_packed_dot_descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_x86_register_class_e {
  LOOM_X86_REGISTER_CLASS_GPR32 = X86_AVX512_PACKED_DOT_CORE_REG_CLASS_ID_GPR32,
  LOOM_X86_REGISTER_CLASS_GPR64 = X86_AVX512_PACKED_DOT_CORE_REG_CLASS_ID_GPR64,
  LOOM_X86_REGISTER_CLASS_XMM = X86_AVX512_PACKED_DOT_CORE_REG_CLASS_ID_XMM,
  LOOM_X86_REGISTER_CLASS_YMM = X86_AVX512_PACKED_DOT_CORE_REG_CLASS_ID_YMM,
  LOOM_X86_REGISTER_CLASS_ZMM = X86_AVX512_PACKED_DOT_CORE_REG_CLASS_ID_ZMM,
  LOOM_X86_REGISTER_CLASS_K = X86_AVX512_PACKED_DOT_CORE_REG_CLASS_ID_K,
} loom_x86_register_class_t;

// Maps a SIMD/vector architectural width to its logical x86 register class.
// Returns false for unsupported widths and stores GPR32 in |out_register_class|
// so callers never observe an uninitialized value.
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
