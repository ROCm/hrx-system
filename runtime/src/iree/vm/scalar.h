// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_SCALAR_H_
#define IREE_VM_SCALAR_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Append-only structural scalar types shared by signatures and logical values.
// The numeric values intentionally match the bytecode module signature
// encodings. Loom types such as index, offset, and tensor are lowered before
// reaching this boundary.
enum iree_vm_scalar_type_e {
  // No scalar type.
  IREE_VM_SCALAR_TYPE_NONE = 0x00,
  // Low eight integer bits.
  IREE_VM_SCALAR_TYPE_I8 = 0x01,
  // Low 16 integer bits.
  IREE_VM_SCALAR_TYPE_I16 = 0x02,
  // Low 32 integer bits.
  IREE_VM_SCALAR_TYPE_I32 = 0x03,
  // Complete 64 integer bits.
  IREE_VM_SCALAR_TYPE_I64 = 0x04,
  // Low eight floating-point bits in E4M3FN format.
  IREE_VM_SCALAR_TYPE_F8E4M3FN = 0x05,
  // Low eight floating-point bits in E5M2 format.
  IREE_VM_SCALAR_TYPE_F8E5M2 = 0x06,
  // Low 16 IEEE binary16 bits.
  IREE_VM_SCALAR_TYPE_F16 = 0x07,
  // Low 16 bfloat16 bits.
  IREE_VM_SCALAR_TYPE_BF16 = 0x08,
  // Low 32 IEEE binary32 bits.
  IREE_VM_SCALAR_TYPE_F32 = 0x09,
  // Complete 64 IEEE binary64 bits.
  IREE_VM_SCALAR_TYPE_F64 = 0x0A,
};
typedef uint8_t iree_vm_scalar_type_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_SCALAR_H_
