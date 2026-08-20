// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Natural-layout instruction records for core.family.conversion.
// Multi-byte fields are little-endian and naturally aligned.
// clang-format off
#ifndef IREE_VM_BYTECODE_GENERATED_WIRE_CORE_CONVERSION_H_
#define IREE_VM_BYTECODE_GENERATED_WIRE_CORE_CONVERSION_H_

#include <stdint.h>

// Selects an exact low-bit integer truncation or extension. Results clear every
// cell bit above their declared destination width.
typedef uint8_t iree_vm_isa_integer_convert_t;
enum {
  // Sign-extends the low 8 bits to i32.
  IREE_VM_ISA_INTEGER_CONVERT_S8_TO_I32 = 0x00,
  // Zero-extends the low 8 bits to i32.
  IREE_VM_ISA_INTEGER_CONVERT_U8_TO_I32 = 0x01,
  // Sign-extends the low 16 bits to i32.
  IREE_VM_ISA_INTEGER_CONVERT_S16_TO_I32 = 0x02,
  // Zero-extends the low 16 bits to i32.
  IREE_VM_ISA_INTEGER_CONVERT_U16_TO_I32 = 0x03,
  // Sign-extends the low 32 bits through the cell.
  IREE_VM_ISA_INTEGER_CONVERT_S32_TO_I64 = 0x04,
  // Zero-extends the low 32 bits through the cell.
  IREE_VM_ISA_INTEGER_CONVERT_U32_TO_I64 = 0x05,
  // Preserves the low 8 bits and clears all higher bits.
  IREE_VM_ISA_INTEGER_CONVERT_I32_TO_I8 = 0x06,
  // Preserves the low 16 bits and clears all higher bits.
  IREE_VM_ISA_INTEGER_CONVERT_I32_TO_I16 = 0x07,
  // Preserves the low 32 bits and clears the high 32 bits.
  IREE_VM_ISA_INTEGER_CONVERT_I64_TO_I32 = 0x08,
};

enum {
  IREE_VM_ISA_INTEGER_CONVERT_S8_TO_I32_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_CONVERT_U8_TO_I32_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_CONVERT_S16_TO_I32_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_CONVERT_U16_TO_I32_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_CONVERT_S32_TO_I64_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_CONVERT_U32_TO_I64_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_CONVERT_I32_TO_I8_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_CONVERT_I32_TO_I16_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_CONVERT_I64_TO_I32_SINCE_MINOR = 0,
};

// Selects one narrow source format to extend structurally to its exact f32
// value; NaNs produce a quiet f32 arithmetic NaN.
typedef uint8_t iree_vm_isa_float_extend_t;
enum {
  // Extends low E4M3FN bits exactly to f32.
  IREE_VM_ISA_FLOAT_EXTEND_F8E4M3_TO_F32 = 0x00,
  // Extends low E5M2 bits exactly to f32.
  IREE_VM_ISA_FLOAT_EXTEND_F8E5M2_TO_F32 = 0x01,
  // Extends low IEEE binary16 bits exactly to f32.
  IREE_VM_ISA_FLOAT_EXTEND_F16_TO_F32 = 0x02,
  // Extends low bfloat16 bits exactly to f32.
  IREE_VM_ISA_FLOAT_EXTEND_BF16_TO_F32 = 0x03,
};

enum {
  IREE_VM_ISA_FLOAT_EXTEND_F8E4M3_TO_F32_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_EXTEND_F8E5M2_TO_F32_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_EXTEND_F16_TO_F32_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_EXTEND_BF16_TO_F32_SINCE_MINOR = 0,
};

// Selects source and narrow destination formats for one direct nearest-even
// conversion with gradual subnormals. E4M3FN overflow and infinity saturate to
// signed 448; other overflow produces infinity.
typedef uint8_t iree_vm_isa_float_truncate_t;
enum {
  // Rounds f32 directly to E4M3FN.
  IREE_VM_ISA_FLOAT_TRUNCATE_F32_TO_F8E4M3 = 0x00,
  // Rounds f32 directly to E5M2.
  IREE_VM_ISA_FLOAT_TRUNCATE_F32_TO_F8E5M2 = 0x01,
  // Rounds f32 directly to IEEE binary16.
  IREE_VM_ISA_FLOAT_TRUNCATE_F32_TO_F16 = 0x02,
  // Rounds f32 directly to bfloat16.
  IREE_VM_ISA_FLOAT_TRUNCATE_F32_TO_BF16 = 0x03,
  // Rounds f64 directly to E4M3FN without f32 staging.
  IREE_VM_ISA_FLOAT_TRUNCATE_F64_TO_F8E4M3 = 0x04,
  // Rounds f64 directly to E5M2 without f32 staging.
  IREE_VM_ISA_FLOAT_TRUNCATE_F64_TO_F8E5M2 = 0x05,
  // Rounds f64 directly to IEEE binary16 without f32 staging.
  IREE_VM_ISA_FLOAT_TRUNCATE_F64_TO_F16 = 0x06,
  // Rounds f64 directly to bfloat16 without f32 staging.
  IREE_VM_ISA_FLOAT_TRUNCATE_F64_TO_BF16 = 0x07,
};

enum {
  IREE_VM_ISA_FLOAT_TRUNCATE_F32_TO_F8E4M3_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_TRUNCATE_F32_TO_F8E5M2_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_TRUNCATE_F32_TO_F16_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_TRUNCATE_F32_TO_BF16_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_TRUNCATE_F64_TO_F8E4M3_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_TRUNCATE_F64_TO_F8E5M2_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_TRUNCATE_F64_TO_F16_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_TRUNCATE_F64_TO_BF16_SINCE_MINOR = 0,
};

// Selects one nearest-even conversion between IEEE binary32 and binary64.
typedef uint8_t iree_vm_isa_float_width_t;
enum {
  // Extends every finite f32 exactly to f64.
  IREE_VM_ISA_FLOAT_WIDTH_F32_TO_F64 = 0x00,
  // Rounds f64 to f32 with gradual subnormals and infinity on overflow.
  IREE_VM_ISA_FLOAT_WIDTH_F64_TO_F32 = 0x01,
};

enum {
  IREE_VM_ISA_FLOAT_WIDTH_F32_TO_F64_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_WIDTH_F64_TO_F32_SINCE_MINOR = 0,
};

// Selects source signedness/width and destination float format. The exact
// integer is rounded directly nearest-even without intermediate staging.
typedef uint8_t iree_vm_isa_integer_to_float_t;
enum {
  // Rounds signed low i32 directly to f32.
  IREE_VM_ISA_INTEGER_TO_FLOAT_S32_TO_F32 = 0x00,
  // Rounds unsigned low i32 directly to f32.
  IREE_VM_ISA_INTEGER_TO_FLOAT_U32_TO_F32 = 0x01,
  // Rounds signed low i32 directly to f64.
  IREE_VM_ISA_INTEGER_TO_FLOAT_S32_TO_F64 = 0x02,
  // Rounds unsigned low i32 directly to f64.
  IREE_VM_ISA_INTEGER_TO_FLOAT_U32_TO_F64 = 0x03,
  // Rounds signed i64 directly to f32.
  IREE_VM_ISA_INTEGER_TO_FLOAT_S64_TO_F32 = 0x04,
  // Rounds unsigned i64 directly to f32.
  IREE_VM_ISA_INTEGER_TO_FLOAT_U64_TO_F32 = 0x05,
  // Rounds signed i64 directly to f64.
  IREE_VM_ISA_INTEGER_TO_FLOAT_S64_TO_F64 = 0x06,
  // Rounds unsigned i64 directly to f64.
  IREE_VM_ISA_INTEGER_TO_FLOAT_U64_TO_F64 = 0x07,
  // Rounds signed low i32 directly to bfloat16.
  IREE_VM_ISA_INTEGER_TO_FLOAT_S32_TO_BF16 = 0x08,
  // Rounds unsigned low i32 directly to bfloat16.
  IREE_VM_ISA_INTEGER_TO_FLOAT_U32_TO_BF16 = 0x09,
  // Rounds signed i64 directly to bfloat16.
  IREE_VM_ISA_INTEGER_TO_FLOAT_S64_TO_BF16 = 0x0A,
  // Rounds unsigned i64 directly to bfloat16.
  IREE_VM_ISA_INTEGER_TO_FLOAT_U64_TO_BF16 = 0x0B,
};

enum {
  IREE_VM_ISA_INTEGER_TO_FLOAT_S32_TO_F32_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_TO_FLOAT_U32_TO_F32_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_TO_FLOAT_S32_TO_F64_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_TO_FLOAT_U32_TO_F64_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_TO_FLOAT_S64_TO_F32_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_TO_FLOAT_U64_TO_F32_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_TO_FLOAT_S64_TO_F64_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_TO_FLOAT_U64_TO_F64_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_TO_FLOAT_S32_TO_BF16_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_TO_FLOAT_U32_TO_BF16_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_TO_FLOAT_S64_TO_BF16_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_TO_FLOAT_U64_TO_BF16_SINCE_MINOR = 0,
};

// Selects a finite f32/f64 source and integer destination. Successful values
// truncate toward zero; NaN fails invalid_argument and values outside the
// destination's strict source interval fail out_of_range.
typedef uint8_t iree_vm_isa_float_to_integer_t;
enum {
  // Truncates f32 in (-2^31-1, 2^31) to signed i32.
  IREE_VM_ISA_FLOAT_TO_INTEGER_F32_TO_S32 = 0x00,
  // Truncates f32 in (-1, 2^32) to unsigned i32.
  IREE_VM_ISA_FLOAT_TO_INTEGER_F32_TO_U32 = 0x01,
  // Truncates f32 in (-2^63-1, 2^63) to signed i64.
  IREE_VM_ISA_FLOAT_TO_INTEGER_F32_TO_S64 = 0x02,
  // Truncates f32 in (-1, 2^64) to unsigned i64.
  IREE_VM_ISA_FLOAT_TO_INTEGER_F32_TO_U64 = 0x03,
  // Truncates f64 in (-2^31-1, 2^31) to signed i32.
  IREE_VM_ISA_FLOAT_TO_INTEGER_F64_TO_S32 = 0x04,
  // Truncates f64 in (-1, 2^32) to unsigned i32.
  IREE_VM_ISA_FLOAT_TO_INTEGER_F64_TO_U32 = 0x05,
  // Truncates f64 in (-2^63-1, 2^63) to signed i64.
  IREE_VM_ISA_FLOAT_TO_INTEGER_F64_TO_S64 = 0x06,
  // Truncates f64 in (-1, 2^64) to unsigned i64.
  IREE_VM_ISA_FLOAT_TO_INTEGER_F64_TO_U64 = 0x07,
};

enum {
  IREE_VM_ISA_FLOAT_TO_INTEGER_F32_TO_S32_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_TO_INTEGER_F32_TO_U32_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_TO_INTEGER_F32_TO_S64_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_TO_INTEGER_F32_TO_U64_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_TO_INTEGER_F64_TO_S32_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_TO_INTEGER_F64_TO_U32_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_TO_INTEGER_F64_TO_S64_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_TO_INTEGER_F64_TO_U64_SINCE_MINOR = 0,
};

// Page 0x00, opcode 0xA0: Truncates or extends one integer bit pattern.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination value-register ordinal.
  uint8_t dst_v8;
  // Source value-register ordinal.
  uint8_t src_v8;
  // Closed integer.convert operation selector.
  uint8_t selector_u8;
} iree_vm_isa_conversion_integer_record_t;

// Page 0x00, opcode 0xA1: Exactly extends a narrow floating encoding to f32.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination value-register ordinal.
  uint8_t dst_v8;
  // Source value-register ordinal.
  uint8_t src_v8;
  // Closed float.extend operation selector.
  uint8_t selector_u8;
} iree_vm_isa_conversion_float_extend_record_t;

// Page 0x00, opcode 0xA2: Rounds f32 or f64 directly to one narrow floating
// encoding.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination value-register ordinal.
  uint8_t dst_v8;
  // Source value-register ordinal.
  uint8_t src_v8;
  // Closed float.truncate operation selector.
  uint8_t selector_u8;
} iree_vm_isa_conversion_float_truncate_record_t;

// Page 0x00, opcode 0xA3: Converts exactly between f32 and f64 widths.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination value-register ordinal.
  uint8_t dst_v8;
  // Source value-register ordinal.
  uint8_t src_v8;
  // Closed float.width operation selector.
  uint8_t selector_u8;
} iree_vm_isa_conversion_float_width_record_t;

// Page 0x00, opcode 0xA4: Rounds a signed or unsigned integer directly to a
// float.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination value-register ordinal.
  uint8_t dst_v8;
  // Source value-register ordinal.
  uint8_t src_v8;
  // Closed integer.to.float operation selector.
  uint8_t selector_u8;
} iree_vm_isa_conversion_integer_to_float_record_t;

// Page 0x00, opcode 0xA5: Truncates an in-range finite float to a selected
// integer width.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination value-register ordinal.
  uint8_t dst_v8;
  // Source value-register ordinal.
  uint8_t src_v8;
  // Closed float.to.integer operation selector.
  uint8_t selector_u8;
} iree_vm_isa_conversion_float_to_integer_record_t;

#endif  // IREE_VM_BYTECODE_GENERATED_WIRE_CORE_CONVERSION_H_
// clang-format on
