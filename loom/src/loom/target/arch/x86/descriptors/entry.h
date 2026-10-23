// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Instruction identities used by x86 entry-point materialization. Descriptor
// ordinals belong to a representation view, even when views share storage.

#ifndef LOOM_TARGET_ARCH_X86_DESCRIPTORS_ENTRY_H_
#define LOOM_TARGET_ARCH_X86_DESCRIPTORS_ENTRY_H_

#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_x86_entry_descriptors_t {
  // Largest SIMD carrier import supported by this representation, in bytes.
  uint8_t maximum_simd_byte_length;
  // Unsigned byte load to a GPR32 carrier.
  uint32_t load_u8;
  // Unsigned halfword load to a GPR32 carrier.
  uint32_t load_u16;
  // Word load to a GPR32 carrier.
  uint32_t load_u32;
  // Pointer or doubleword load to a GPR64 carrier.
  uint32_t load_u64;
  // Byte store from a GPR32 carrier.
  uint32_t store_u8;
  // Halfword store from a GPR32 carrier.
  uint32_t store_u16;
  // Word store from a GPR32 carrier.
  uint32_t store_u32;
  // Pointer or doubleword store from a GPR64 carrier.
  uint32_t store_u64;
  // Pointer-sized address addition.
  uint32_t add_u64;
  // Word bit masking with an immediate.
  uint32_t mask_u32;
  // Zero extension of a word to a pointer-sized carrier.
  uint32_t widen_u32;
  // Truncation of a pointer-sized carrier to its low word.
  uint32_t truncate_u64;
  // Pointer-sized immediate left shift.
  uint32_t shift_left_u64;
  // Pointer-sized immediate logical right shift.
  uint32_t shift_right_u64;
  // Pointer-sized bitwise union.
  uint32_t or_u64;
  // Word immediate materialization.
  uint32_t constant_u32;
  // Pointer-sized immediate materialization.
  uint32_t constant_u64;
  // Unsigned word comparison with an immediate lower bound.
  uint32_t compare_uge_u32;
  // RIP-relative address of a native symbol.
  uint32_t symbol_address;
  // Four-byte bit move from a GPR32 carrier into XMM.
  uint32_t move_u32_xmm;
  // Eight-byte bit move from a GPR64 carrier into XMM.
  uint32_t move_u64_xmm;
  // Four-byte bit move from XMM into a GPR32 carrier.
  uint32_t move_xmm_u32;
  // Eight-byte bit move from XMM into a GPR64 carrier.
  uint32_t move_xmm_u64;
  // Four-byte memory load into the low XMM bits.
  uint32_t load_32_xmm;
  // Eight-byte memory load into the low XMM bits.
  uint32_t load_64_xmm;
  // Full 16-byte XMM memory load.
  uint32_t load_128_xmm;
  // Full 32-byte YMM memory load.
  uint32_t load_256_ymm;
  // Full 64-byte ZMM memory load.
  uint32_t load_512_zmm;
  // Full 16-byte XMM memory store.
  uint32_t store_128_xmm;
  // Full 32-byte YMM memory store.
  uint32_t store_256_ymm;
  // Full 64-byte ZMM memory store.
  uint32_t store_512_zmm;
  // Zero-valued XMM materialization.
  uint32_t zero_xmm;
  // Zero-valued YMM materialization.
  uint32_t zero_ymm;
  // Zero-valued ZMM materialization.
  uint32_t zero_zmm;
  // Packed byte subtraction in XMM.
  uint32_t subtract_i8_xmm;
  // Packed byte subtraction in YMM.
  uint32_t subtract_i8_ymm;
  // Packed byte subtraction in ZMM.
  uint32_t subtract_i8_zmm;
  // Signed widening of two low bytes into two XMM qwords.
  uint32_t widen_i8_i64_xmm;
  // Signed widening of four low bytes into four XMM dwords.
  uint32_t widen_i8_i32_xmm;
  // Signed widening of eight low bytes into eight XMM words.
  uint32_t widen_i8_i16_xmm;
} loom_x86_entry_descriptors_t;

// Returns the entry vocabulary of an x86 representation contract. Feature-only
// fragments without scalar control/addressing instructions return NULL.
const loom_x86_entry_descriptors_t* loom_x86_entry_descriptors(
    const loom_low_descriptor_set_t* descriptor_set);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_DESCRIPTORS_ENTRY_H_
