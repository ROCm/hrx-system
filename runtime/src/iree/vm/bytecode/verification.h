// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compact generated validation plans consumed by the bytecode verifier.

#ifndef IREE_VM_BYTECODE_VERIFICATION_H_
#define IREE_VM_BYTECODE_VERIFICATION_H_

#include <stdint.h>

#include "iree/vm/bytecode/wire/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Validation operation applied to one field or complete record.
enum iree_vm_bytecode_verification_rule_kind_e {
  IREE_VM_BYTECODE_VERIFICATION_RULE_NONE = 0,
  IREE_VM_BYTECODE_VERIFICATION_RULE_ZERO = 1,
  IREE_VM_BYTECODE_VERIFICATION_RULE_REGISTER_VALUE = 2,
  IREE_VM_BYTECODE_VERIFICATION_RULE_CONTROL_TARGET_S16 = 3,
  IREE_VM_BYTECODE_VERIFICATION_RULE_CONTROL_TARGET_S32 = 4,
  IREE_VM_BYTECODE_VERIFICATION_RULE_RETURN_SIGNATURE = 5,
  IREE_VM_BYTECODE_VERIFICATION_RULE_SWITCH_TARGETS = 6,
  IREE_VM_BYTECODE_VERIFICATION_RULE_EXACT_BYTES = 7,
  IREE_VM_BYTECODE_VERIFICATION_RULE_CORE_MAJOR = 8,
  IREE_VM_BYTECODE_VERIFICATION_RULE_CORE_REQUIRED_MINOR = 9,
  IREE_VM_BYTECODE_VERIFICATION_RULE_ALLOWED_RANGE = 10,
  IREE_VM_BYTECODE_VERIFICATION_RULE_SIGNATURE_DESCRIPTOR = 11,
  IREE_VM_BYTECODE_VERIFICATION_RULE_ALLOWED_BITS = 12,
  IREE_VM_BYTECODE_VERIFICATION_RULE_MULTIPLE = 13,
  IREE_VM_BYTECODE_VERIFICATION_RULE_BYTE_ALIGNMENT = 14,
  IREE_VM_BYTECODE_VERIFICATION_RULE_NONCORE_PAGE = 15,
  IREE_VM_BYTECODE_VERIFICATION_RULE_ORDINAL = 16,
  IREE_VM_BYTECODE_VERIFICATION_RULE_ORDINAL_OR_NULL = 17,
  IREE_VM_BYTECODE_VERIFICATION_RULE_PAGE_MAJOR = 18,
  IREE_VM_BYTECODE_VERIFICATION_RULE_PAGE_REQUIRED_MINOR = 19,
  IREE_VM_BYTECODE_VERIFICATION_RULE_SECTION_BYTE_LENGTH = 20,
  IREE_VM_BYTECODE_VERIFICATION_RULE_SECTION_FLAGS = 21,
  IREE_VM_BYTECODE_VERIFICATION_RULE_SECTION_TYPE = 22,
  IREE_VM_BYTECODE_VERIFICATION_RULE_STRING_OFFSET = 23,
  IREE_VM_BYTECODE_VERIFICATION_RULE_SWITCH_TARGET = 24,
  IREE_VM_BYTECODE_VERIFICATION_RULE_REGISTER_REF = 25,
  IREE_VM_BYTECODE_VERIFICATION_RULE_REGISTER_FUNCTION = 26,
  IREE_VM_BYTECODE_VERIFICATION_RULE_SELECTOR = 27,
  IREE_VM_BYTECODE_VERIFICATION_RULE_CONSTANT_POOL_ORDINAL = 28,
  IREE_VM_BYTECODE_VERIFICATION_RULE_FUNCTION_LOCAL_ORDINAL = 29,
  IREE_VM_BYTECODE_VERIFICATION_RULE_IMPORT_ORDINAL_OPTIONAL = 30,
  IREE_VM_BYTECODE_VERIFICATION_RULE_CALL = 31,
  IREE_VM_BYTECODE_VERIFICATION_RULE_CALL_INDIRECT = 32,
  IREE_VM_BYTECODE_VERIFICATION_RULE_FUNCTION_ADDRESS = 33,
};
typedef uint8_t iree_vm_bytecode_verification_rule_kind_t;

// Module-local ordinal domain interpreted by ordinal validation rules.
enum iree_vm_bytecode_ordinal_domain_e {
  IREE_VM_BYTECODE_ORDINAL_DOMAIN_INVALID = 0,
  IREE_VM_BYTECODE_ORDINAL_DOMAIN_STRING = 1,
  IREE_VM_BYTECODE_ORDINAL_DOMAIN_STRING_NONEMPTY = 2,
  IREE_VM_BYTECODE_ORDINAL_DOMAIN_REF_TYPE = 3,
  IREE_VM_BYTECODE_ORDINAL_DOMAIN_SIGNATURE = 4,
  IREE_VM_BYTECODE_ORDINAL_DOMAIN_CALLABLE_TYPE = 5,
  IREE_VM_BYTECODE_ORDINAL_DOMAIN_FUNCTION = 6,
};
typedef uint8_t iree_vm_bytecode_ordinal_domain_t;

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

// One validation operation over a field or complete record.
typedef struct iree_vm_bytecode_verification_rule_t {
  // Rule-specific immediate or base in |verification_parameters|.
  uint32_t parameter;
  // Operation from iree_vm_bytecode_verification_rule_kind_e.
  iree_vm_bytecode_verification_rule_kind_t kind;
  // Primary byte offset within the record, when used by |kind|.
  uint8_t field_offset;
  // Primary field byte length within the record, when used by |kind|.
  uint8_t field_length;
  // Secondary byte offset or parameter count, when used by |kind|.
  uint8_t auxiliary;
} iree_vm_bytecode_verification_rule_t;

// Packed instruction validation descriptors indexed directly by u8 opcode.
// Zero denotes an unknown opcode. Nonzero words encode, from most- to
// least-significant bits, rule_base:u16, control_flow:u4, rule_count:u4, and
// byte_length:u8.
extern const uint32_t iree_vm_bytecode_instruction_verification[256];

// Packed module validation descriptors indexed by module record ordinal.
// Words encode rule_base:u16, rule_count:u8, and byte_length:u8.
extern const uint32_t iree_vm_bytecode_module_record_verification
    [IREE_VM_BYTECODE_MODULE_RECORD_COUNT];

// Validation rules referenced by the packed instruction and module indices.
extern const iree_vm_bytecode_verification_rule_t
    iree_vm_bytecode_verification_rules[];

// Number of rows in |iree_vm_bytecode_verification_rules|.
extern const uint32_t iree_vm_bytecode_verification_rule_count;

// Rule-specific literal and range words referenced by validation rules.
extern const uint32_t iree_vm_bytecode_verification_parameters[];

// Number of words in |iree_vm_bytecode_verification_parameters|.
extern const uint32_t iree_vm_bytecode_verification_parameter_count;

// Returns the encoded record byte length from a packed descriptor.
static inline uint8_t iree_vm_bytecode_verification_byte_length(
    uint32_t descriptor) {
  return (uint8_t)descriptor;
}

// Returns the rule count from an instruction validation descriptor.
static inline uint8_t iree_vm_bytecode_instruction_verification_rule_count(
    uint32_t descriptor) {
  return (uint8_t)((descriptor >> 8) & 0xFu);
}

// Returns the structural continuation kind from an instruction descriptor.
static inline iree_vm_bytecode_control_flow_t
iree_vm_bytecode_instruction_verification_control_flow(uint32_t descriptor) {
  return (iree_vm_bytecode_control_flow_t)((descriptor >> 12) & 0xFu);
}

// Returns the rule count from a module-record validation descriptor.
static inline uint8_t iree_vm_bytecode_module_verification_rule_count(
    uint32_t descriptor) {
  return (uint8_t)(descriptor >> 8);
}

// Returns the first rule ordinal from any packed validation descriptor.
static inline uint16_t iree_vm_bytecode_verification_rule_base(
    uint32_t descriptor) {
  return (uint16_t)(descriptor >> 16);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_VERIFICATION_H_
