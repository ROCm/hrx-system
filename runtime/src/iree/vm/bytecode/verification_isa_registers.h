// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_VERIFICATION_ISA_REGISTERS_H_
#define IREE_VM_BYTECODE_VERIFICATION_ISA_REGISTERS_H_

#include "iree/vm/bytecode/module_reader.h"

// Verifies that |ordinal| names a value register in the function bank.
iree_status_t iree_vm_bytecode_verify_value_register(uint8_t ordinal,
                                                     uint16_t register_count);

// Verifies that |ordinal| names a ref register in the function bank.
iree_status_t iree_vm_bytecode_verify_ref_register(uint8_t ordinal,
                                                   uint16_t register_count);

// Verifies that |ordinal| names a function register in the function bank.
iree_status_t iree_vm_bytecode_verify_function_register(
    uint8_t ordinal, uint16_t register_count);

// Verifies that |ordinal| names a function in local overflow storage.
iree_status_t iree_vm_bytecode_verify_function_local(uint16_t ordinal,
                                                     uint32_t local_count);

// Verifies that |ordinal| names a ref in local overflow storage.
iree_status_t iree_vm_bytecode_verify_ref_slot(uint16_t ordinal,
                                               uint32_t slot_count);

// Verifies that [|base|, |base| + |count|) is in the value register bank.
iree_status_t iree_vm_bytecode_verify_value_register_range(
    uint8_t base, uint8_t count, uint16_t register_count);

// Verifies that [|base|, |base| + |length|) is in local byte storage.
iree_status_t iree_vm_bytecode_verify_local_range(uint16_t base,
                                                  uint32_t length,
                                                  uint16_t local_byte_length);

// Verifies a memory-format lane range in the function value register bank.
iree_status_t iree_vm_bytecode_verify_lane_register_range(
    uint8_t register_base, uint8_t format,
    const iree_vm_bytecode_v0_function_row_t* function);

#endif  // IREE_VM_BYTECODE_VERIFICATION_ISA_REGISTERS_H_
