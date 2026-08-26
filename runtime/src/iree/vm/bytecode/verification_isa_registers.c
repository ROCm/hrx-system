// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/verification_isa_registers.h"

#include "iree/vm/bytecode/wire/core/selectors.h"

iree_status_t iree_vm_bytecode_verify_value_register(uint8_t ordinal,
                                                     uint16_t register_count) {
  if (ordinal >= register_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value register ordinal is out of range");
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_verify_ref_register(uint8_t ordinal,
                                                   uint16_t register_count) {
  if (ordinal >= register_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ref register ordinal is out of range");
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_verify_function_register(
    uint8_t ordinal, uint16_t register_count) {
  if (ordinal >= register_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function register ordinal is out of range");
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_verify_function_local(uint16_t ordinal,
                                                     uint32_t local_count) {
  if (ordinal >= local_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local function ordinal is out of range");
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_verify_ref_slot(uint16_t ordinal,
                                               uint32_t slot_count) {
  if (ordinal >= slot_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local ref slot ordinal is out of range");
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_verify_value_register_range(
    uint8_t base, uint8_t count, uint16_t register_count) {
  if (base > register_count || count > register_count - base) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value register range is out of bounds");
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_verify_local_range(uint16_t base,
                                                  uint32_t length,
                                                  uint16_t local_byte_length) {
  if (base > local_byte_length || length > local_byte_length - base) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local byte range is out of bounds");
  }
  return iree_ok_status();
}

iree_status_t iree_vm_bytecode_verify_lane_register_range(
    uint8_t register_base, uint8_t format,
    const iree_vm_bytecode_v0_function_row_t* function) {
  if (format > IREE_VM_ISA_MEMORY_FORMAT_I64_X8) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "memory format is invalid");
  }
  const uint8_t lane_count = (uint8_t)(1u << (format & 3u));
  return iree_vm_bytecode_verify_value_register_range(
      register_base, lane_count, function->value_register_count_u16);
}
