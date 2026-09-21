// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/register_classes.h"

bool loom_x86_register_class_for_vector_bit_width(
    uint32_t vector_bit_width, loom_x86_register_class_t* out_register_class) {
  switch (vector_bit_width) {
    case 128:
      *out_register_class = LOOM_X86_REGISTER_CLASS_XMM;
      return true;
    case 256:
      *out_register_class = LOOM_X86_REGISTER_CLASS_YMM;
      return true;
    case 512:
      *out_register_class = LOOM_X86_REGISTER_CLASS_ZMM;
      return true;
    default:
      *out_register_class = LOOM_X86_REGISTER_CLASS_GPR32;
      return false;
  }
}

loom_x86_register_class_t loom_x86_logical_register_class(
    uint16_t descriptor_reg_class_id) {
  static const uint16_t logical_classes[] = {
      [X86_AVX512_PACKED_DOT_CORE_REG_CLASS_ID_GPR32] =
          LOOM_X86_REGISTER_CLASS_GPR32,
      [X86_AVX512_PACKED_DOT_CORE_REG_CLASS_ID_GPR64] =
          LOOM_X86_REGISTER_CLASS_GPR64,
      [X86_AVX512_PACKED_DOT_CORE_REG_CLASS_ID_RAX] =
          LOOM_X86_REGISTER_CLASS_GPR64,
      [X86_AVX512_PACKED_DOT_CORE_REG_CLASS_ID_RDX] =
          LOOM_X86_REGISTER_CLASS_GPR64,
      [X86_AVX512_PACKED_DOT_CORE_REG_CLASS_ID_ECX] =
          LOOM_X86_REGISTER_CLASS_GPR32,
      [X86_AVX512_PACKED_DOT_CORE_REG_CLASS_ID_RCX] =
          LOOM_X86_REGISTER_CLASS_GPR64,
      [X86_AVX512_PACKED_DOT_CORE_REG_CLASS_ID_XMM] =
          LOOM_X86_REGISTER_CLASS_XMM,
      [X86_AVX512_PACKED_DOT_CORE_REG_CLASS_ID_YMM] =
          LOOM_X86_REGISTER_CLASS_YMM,
      [X86_AVX512_PACKED_DOT_CORE_REG_CLASS_ID_ZMM] =
          LOOM_X86_REGISTER_CLASS_ZMM,
      [X86_AVX512_PACKED_DOT_CORE_REG_CLASS_ID_K] = LOOM_X86_REGISTER_CLASS_K,
  };
  return (loom_x86_register_class_t)logical_classes[descriptor_reg_class_id];
}
