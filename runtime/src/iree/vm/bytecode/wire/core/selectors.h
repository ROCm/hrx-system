// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// CORE ISA selectors shared by multiple families.
// Multi-byte fields are little-endian and naturally aligned.
// clang-format off
#ifndef IREE_VM_BYTECODE_GENERATED_WIRE_CORE_SELECTORS_H_
#define IREE_VM_BYTECODE_GENERATED_WIRE_CORE_SELECTORS_H_

#include <stdint.h>

// Selects the integer lane width and lane count transferred between one
// value-register run and consecutive little-endian bytes.
typedef uint8_t iree_vm_isa_memory_format_t;
enum {
  // Transfers one 8-bit lane spanning one byte.
  IREE_VM_ISA_MEMORY_FORMAT_I8_X1 = 0x00,
  // Transfers two 8-bit lanes spanning two bytes.
  IREE_VM_ISA_MEMORY_FORMAT_I8_X2 = 0x01,
  // Transfers four 8-bit lanes spanning four bytes.
  IREE_VM_ISA_MEMORY_FORMAT_I8_X4 = 0x02,
  // Transfers eight 8-bit lanes spanning eight bytes.
  IREE_VM_ISA_MEMORY_FORMAT_I8_X8 = 0x03,
  // Transfers one 16-bit lane spanning two bytes.
  IREE_VM_ISA_MEMORY_FORMAT_I16_X1 = 0x04,
  // Transfers two 16-bit lanes spanning four bytes.
  IREE_VM_ISA_MEMORY_FORMAT_I16_X2 = 0x05,
  // Transfers four 16-bit lanes spanning eight bytes.
  IREE_VM_ISA_MEMORY_FORMAT_I16_X4 = 0x06,
  // Transfers eight 16-bit lanes spanning 16 bytes.
  IREE_VM_ISA_MEMORY_FORMAT_I16_X8 = 0x07,
  // Transfers one 32-bit lane spanning four bytes.
  IREE_VM_ISA_MEMORY_FORMAT_I32_X1 = 0x08,
  // Transfers two 32-bit lanes spanning eight bytes.
  IREE_VM_ISA_MEMORY_FORMAT_I32_X2 = 0x09,
  // Transfers four 32-bit lanes spanning 16 bytes.
  IREE_VM_ISA_MEMORY_FORMAT_I32_X4 = 0x0A,
  // Transfers eight 32-bit lanes spanning 32 bytes.
  IREE_VM_ISA_MEMORY_FORMAT_I32_X8 = 0x0B,
  // Transfers one 64-bit lane spanning eight bytes.
  IREE_VM_ISA_MEMORY_FORMAT_I64_X1 = 0x0C,
  // Transfers two 64-bit lanes spanning 16 bytes.
  IREE_VM_ISA_MEMORY_FORMAT_I64_X2 = 0x0D,
  // Transfers four 64-bit lanes spanning 32 bytes.
  IREE_VM_ISA_MEMORY_FORMAT_I64_X4 = 0x0E,
  // Transfers eight 64-bit lanes spanning 64 bytes.
  IREE_VM_ISA_MEMORY_FORMAT_I64_X8 = 0x0F,
};

enum {
  IREE_VM_ISA_MEMORY_FORMAT_I8_X1_SINCE_MINOR = 0,
  IREE_VM_ISA_MEMORY_FORMAT_I8_X2_SINCE_MINOR = 0,
  IREE_VM_ISA_MEMORY_FORMAT_I8_X4_SINCE_MINOR = 0,
  IREE_VM_ISA_MEMORY_FORMAT_I8_X8_SINCE_MINOR = 0,
  IREE_VM_ISA_MEMORY_FORMAT_I16_X1_SINCE_MINOR = 0,
  IREE_VM_ISA_MEMORY_FORMAT_I16_X2_SINCE_MINOR = 0,
  IREE_VM_ISA_MEMORY_FORMAT_I16_X4_SINCE_MINOR = 0,
  IREE_VM_ISA_MEMORY_FORMAT_I16_X8_SINCE_MINOR = 0,
  IREE_VM_ISA_MEMORY_FORMAT_I32_X1_SINCE_MINOR = 0,
  IREE_VM_ISA_MEMORY_FORMAT_I32_X2_SINCE_MINOR = 0,
  IREE_VM_ISA_MEMORY_FORMAT_I32_X4_SINCE_MINOR = 0,
  IREE_VM_ISA_MEMORY_FORMAT_I32_X8_SINCE_MINOR = 0,
  IREE_VM_ISA_MEMORY_FORMAT_I64_X1_SINCE_MINOR = 0,
  IREE_VM_ISA_MEMORY_FORMAT_I64_X2_SINCE_MINOR = 0,
  IREE_VM_ISA_MEMORY_FORMAT_I64_X4_SINCE_MINOR = 0,
  IREE_VM_ISA_MEMORY_FORMAT_I64_X8_SINCE_MINOR = 0,
};

// Selects the ordinal table used by a direct control.call record.
typedef uint8_t iree_vm_isa_control_call_target_t;
enum {
  // The ordinal names a function defined by the current module.
  IREE_VM_ISA_CONTROL_CALL_TARGET_LOCAL = 0x00,
  // The ordinal names an import that must resolve while linking.
  IREE_VM_ISA_CONTROL_CALL_TARGET_REQUIRED_IMPORT = 0x01,
  // The ordinal names a weak import; calling it unresolved fails not_found.
  IREE_VM_ISA_CONTROL_CALL_TARGET_OPTIONAL_IMPORT = 0x02,
};

enum {
  IREE_VM_ISA_CONTROL_CALL_TARGET_LOCAL_SINCE_MINOR = 0,
  IREE_VM_ISA_CONTROL_CALL_TARGET_REQUIRED_IMPORT_SINCE_MINOR = 0,
  IREE_VM_ISA_CONTROL_CALL_TARGET_OPTIONAL_IMPORT_SINCE_MINOR = 0,
};

#endif  // IREE_VM_BYTECODE_GENERATED_WIRE_CORE_SELECTORS_H_
// clang-format on
