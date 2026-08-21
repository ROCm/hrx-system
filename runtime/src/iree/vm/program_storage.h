// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_PROGRAM_STORAGE_H_
#define IREE_VM_PROGRAM_STORAGE_H_

#include "iree/vm/program.h"

// Private immutable program representation shared with process and invocation
// implementations. No declaration in this file is part of the public ABI.

enum iree_vm_program_target_layout_e {
  // Two zero tag bits preserve the 16-byte function carrier encoding.
  IREE_VM_PROGRAM_TARGET_MODULE_SHIFT = 2,
  // Module and function ordinals each occupy 16 bits.
  IREE_VM_PROGRAM_TARGET_FUNCTION_SHIFT = 18,
  // The structural callable token occupies bits 34 through 62.
  IREE_VM_PROGRAM_TARGET_CALLABLE_TOKEN_SHIFT = 34,
};

// Low callable mapping bits containing the nonzero structural token.
#define IREE_VM_PROGRAM_CALLABLE_TOKEN_MASK UINT32_C(0x1FFFFFFF)
// Callable mapping bit permitting suspension.
#define IREE_VM_PROGRAM_CALLABLE_MAY_YIELD (UINT32_C(1) << 29)

// Exact physical bank counts for one source-ordered argument signature.
typedef struct iree_vm_program_argument_counts_t {
  // Number of value argument cells.
  uint16_t value_count;
  // Number of ref argument cells.
  uint16_t ref_count;
  // Number of function argument cells.
  uint16_t function_count;
} iree_vm_program_argument_counts_t;

// Cached executable initializer plan.
typedef struct iree_vm_program_initializer_t {
  // Complete packed target bits, or zero when no initializer is exported.
  uint64_t target_bits;
  // Stable module-owned source-ordered argument contract.
  iree_vm_module_signature_type_span_t argument_types;
  // Exact physical argument bank counts.
  iree_vm_program_argument_counts_t argument_counts;
} iree_vm_program_initializer_t;

// Immutable program-local view of one retained generic module.
struct iree_vm_linked_module_t {
  // Retained immutable module implementation.
  iree_vm_module_t* module;
  // Immutable module-local resolved import target words.
  const uint64_t* import_target_bits;
  // First entry in the program's flat callable mapping array.
  uint32_t callable_mapping_base;
  // Max-aligned process-state offset, or UINT32_MAX for no state.
  uint32_t process_storage_offset;
};

// One allocation containing the complete immutable linked program.
struct iree_vm_program_t {
  // Intrusive program owner count.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning this program slab.
  iree_allocator_t host_allocator;
  // Name-sorted linked module directory.
  iree_vm_linked_module_t* linked_modules;
  // Number of linked modules in |linked_modules|.
  uint32_t linked_module_count;
  // Program ordinal of the caller-selected executable module.
  uint16_t executable_module_ordinal;
  // Selected executable initializer contract.
  iree_vm_program_initializer_t initializer;
  // Flat module-local callable mapping words.
  uint32_t* callable_mappings;
  // Total max-aligned opaque process-state bytes.
  iree_host_size_t process_storage_size;
};

static_assert(sizeof(void*) != 8 || sizeof(iree_vm_linked_module_t) == 24,
              "64-bit linked modules must remain 24 bytes");
static_assert(sizeof(void*) != 4 || sizeof(iree_vm_linked_module_t) == 16,
              "32-bit linked modules must remain 16 bytes");
static_assert(sizeof(void*) != 8 || sizeof(iree_vm_program_initializer_t) == 32,
              "64-bit initializer plans must remain 32 bytes");

// Returns the nonzero structural token from one callable mapping.
static inline uint32_t iree_vm_program_callable_token(uint32_t mapping) {
  return mapping & IREE_VM_PROGRAM_CALLABLE_TOKEN_MASK;
}

// Returns whether one callable mapping permits suspension.
static inline bool iree_vm_program_callable_may_yield(uint32_t mapping) {
  return iree_any_bit_set(mapping, IREE_VM_PROGRAM_CALLABLE_MAY_YIELD);
}

// Packs one complete target word with its two low tag bits clear.
static inline uint64_t iree_vm_program_pack_target_bits(
    uint16_t module_ordinal, uint16_t function_ordinal,
    uint32_t callable_mapping) {
  return ((uint64_t)module_ordinal << IREE_VM_PROGRAM_TARGET_MODULE_SHIFT) |
         ((uint64_t)function_ordinal << IREE_VM_PROGRAM_TARGET_FUNCTION_SHIFT) |
         ((uint64_t)iree_vm_program_callable_token(callable_mapping)
          << IREE_VM_PROGRAM_TARGET_CALLABLE_TOKEN_SHIFT) |
         (iree_vm_program_callable_may_yield(callable_mapping)
              ? UINT64_C(1) << 63
              : 0);
}

// Returns the module ordinal encoded in nonzero target bits.
static inline uint16_t iree_vm_program_target_module_ordinal(
    uint64_t target_bits) {
  return (uint16_t)(target_bits >> IREE_VM_PROGRAM_TARGET_MODULE_SHIFT);
}

// Returns the function ordinal encoded in nonzero target bits.
static inline uint16_t iree_vm_program_target_function_ordinal(
    uint64_t target_bits) {
  return (uint16_t)(target_bits >> IREE_VM_PROGRAM_TARGET_FUNCTION_SHIFT);
}

// Returns the structural callable token encoded in nonzero target bits.
static inline uint32_t iree_vm_program_target_callable_token(
    uint64_t target_bits) {
  return (uint32_t)(target_bits >>
                    IREE_VM_PROGRAM_TARGET_CALLABLE_TOKEN_SHIFT) &
         IREE_VM_PROGRAM_CALLABLE_TOKEN_MASK;
}

// Returns whether nonzero target bits name a possibly yielding function.
static inline bool iree_vm_program_target_may_yield(uint64_t target_bits) {
  return (target_bits >> 63) != 0;
}

#endif  // IREE_VM_PROGRAM_STORAGE_H_
