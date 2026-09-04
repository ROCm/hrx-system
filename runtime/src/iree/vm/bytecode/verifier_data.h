// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_VERIFIER_DATA_H_
#define IREE_VM_BYTECODE_VERIFIER_DATA_H_

#include <stdint.h>

#include "iree/vm/bytecode/wire/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Structural continuation kind used by function-level verification.
enum iree_vm_bytecode_control_flow_e {
  IREE_VM_BYTECODE_CONTROL_FLOW_INVALID = 0,
  IREE_VM_BYTECODE_CONTROL_FLOW_SEQUENTIAL = 1,
  IREE_VM_BYTECODE_CONTROL_FLOW_BLOCK = 2,
  IREE_VM_BYTECODE_CONTROL_FLOW_RETURN = 3,
  IREE_VM_BYTECODE_CONTROL_FLOW_YIELD = 4,
  IREE_VM_BYTECODE_CONTROL_FLOW_BRANCH = 5,
  IREE_VM_BYTECODE_CONTROL_FLOW_CONDITIONAL_BRANCH = 6,
  IREE_VM_BYTECODE_CONTROL_FLOW_SWITCH = 7,
  IREE_VM_BYTECODE_CONTROL_FLOW_CALL = 8,
  IREE_VM_BYTECODE_CONTROL_FLOW_FAIL = 9,
};
typedef uint8_t iree_vm_bytecode_control_flow_t;

// Packed instruction validation descriptors indexed directly by u8 opcode.
// Zero denotes an unknown opcode. Nonzero words encode control_flow:u4,
// reserved_zero:u4, and byte_length:u8.
extern const uint16_t iree_vm_bytecode_instruction_verification[256];

// Packed known-section descriptors indexed by dense Core section type. Words
// encode since_minor:u16 and required_flags:u16.
extern const uint32_t iree_vm_bytecode_module_section_verification
    [IREE_VM_BYTECODE_SECTION_METADATA + 1];

// Returns the encoded instruction byte length from a packed descriptor.
static inline uint8_t iree_vm_bytecode_verification_byte_length(
    uint16_t descriptor) {
  return (uint8_t)descriptor;
}

// Returns the structural continuation kind from an instruction descriptor.
static inline iree_vm_bytecode_control_flow_t
iree_vm_bytecode_instruction_verification_control_flow(uint16_t descriptor) {
  return (iree_vm_bytecode_control_flow_t)((descriptor >> 12) & 0xFu);
}

// Returns the required flags from a packed known-section descriptor.
static inline uint16_t iree_vm_bytecode_section_verification_required_flags(
    uint32_t descriptor) {
  return (uint16_t)descriptor;
}

// Returns the introducing Core minor from a packed known-section descriptor.
static inline uint16_t iree_vm_bytecode_section_verification_since_minor(
    uint32_t descriptor) {
  return (uint16_t)(descriptor >> 16);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_VERIFIER_DATA_H_
