// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_PROCESS_H_
#define IREE_VM_BYTECODE_PROCESS_H_

#include "iree/vm/bytecode/image.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum iree_vm_bytecode_process_construction_state_e {
  // The unpublished process may initialize set-once globals.
  IREE_VM_BYTECODE_PROCESS_CONSTRUCTION_STATE_OPEN = 0u,
  // The published process can no longer initialize set-once globals.
  IREE_VM_BYTECODE_PROCESS_CONSTRUCTION_STATE_SEALED = 1u,
};
typedef uint32_t iree_vm_bytecode_process_construction_state_t;

// Returns the private construction header of nonempty |process_storage|.
static inline iree_vm_bytecode_process_header_t*
iree_vm_bytecode_process_header(void* process_storage) {
  return (iree_vm_bytecode_process_header_t*)process_storage;
}

// Returns the first value-global cell in nonempty |process_storage|.
static inline uint64_t* iree_vm_bytecode_process_values(
    const iree_vm_bytecode_image_t* image, void* process_storage) {
  return (uint64_t*)((uint8_t*)process_storage +
                     image->process_layout.values_offset);
}

// Returns the first ref-global cell in nonempty |process_storage|.
static inline iree_vm_ref_t* iree_vm_bytecode_process_refs(
    const iree_vm_bytecode_image_t* image, void* process_storage) {
  return (iree_vm_ref_t*)((uint8_t*)process_storage +
                          image->process_layout.refs_offset);
}

// Returns the first function-global cell in nonempty |process_storage|.
static inline iree_vm_function_ref_t* iree_vm_bytecode_process_functions(
    const iree_vm_bytecode_image_t* image, void* process_storage) {
  return (iree_vm_function_ref_t*)((uint8_t*)process_storage +
                                   image->process_layout.functions_offset);
}

// Returns the first immutable value-global set-bit word.
static inline uint64_t* iree_vm_bytecode_process_value_set_bits(
    const iree_vm_bytecode_image_t* image, void* process_storage) {
  return (uint64_t*)((uint8_t*)process_storage +
                     image->process_layout.value_set_bits_offset);
}

// Returns the first immutable ref-global set-bit word.
static inline uint64_t* iree_vm_bytecode_process_ref_set_bits(
    const iree_vm_bytecode_image_t* image, void* process_storage) {
  return (uint64_t*)((uint8_t*)process_storage +
                     image->process_layout.ref_set_bits_offset);
}

// Returns the first immutable function-global set-bit word.
static inline uint64_t* iree_vm_bytecode_process_function_set_bits(
    const iree_vm_bytecode_image_t* image, void* process_storage) {
  return (uint64_t*)((uint8_t*)process_storage +
                     image->process_layout.function_set_bits_offset);
}

// Returns whether one valid packed bit is set.
static inline bool iree_vm_bytecode_process_bit_test(const uint64_t* bits,
                                                     uint32_t ordinal) {
  return (bits[ordinal / 64u] & ((uint64_t)1u << (ordinal % 64u))) != 0;
}

// Sets one valid packed bit.
static inline void iree_vm_bytecode_process_bit_set(uint64_t* bits,
                                                    uint32_t ordinal) {
  bits[ordinal / 64u] |= (uint64_t)1u << (ordinal % 64u);
}

// Validates initialized globals and closes construction for publication.
iree_status_t iree_vm_bytecode_process_seal_state(iree_vm_module_t* module,
                                                  iree_byte_span_t storage);

// Releases every owning ref global in an attached or sealed process slice.
void iree_vm_bytecode_process_detach_state(iree_vm_module_t* module,
                                           iree_byte_span_t storage);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_PROCESS_H_
