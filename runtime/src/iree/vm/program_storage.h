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

// Exact physical bank counts for one source-ordered signature side.
typedef struct iree_vm_program_bank_counts_t {
  // Number of value cells.
  uint16_t value_count;
  // Number of ref cells.
  uint16_t ref_count;
  // Number of function cells.
  uint16_t function_count;
} iree_vm_program_bank_counts_t;

// One scalar field mapped between source order and the physical value bank.
typedef struct iree_vm_program_scalar_field_abi_t {
  // Canonical payload bits permitted by the scalar type.
  uint64_t payload_mask;
  // Exact scalar variant metadata expected or published at the host boundary.
  uint32_t variant_metadata;
  // Byte offset of the source-ordered host variant.
  uint32_t variant_offset;
} iree_vm_program_scalar_field_abi_t;

// One ref field mapped between source order and the physical ref bank.
typedef struct iree_vm_program_ref_field_abi_t {
  // Canonical resolved ref type expected at the host boundary.
  iree_vm_ref_type_t type;
  // Byte offset of the source-ordered host variant.
  uint32_t variant_offset;
} iree_vm_program_ref_field_abi_t;

// One function field mapped between source order and the physical function
// bank.
typedef struct iree_vm_program_function_field_abi_t {
  // Expected canonical callable mapping and suspension permission.
  uint32_t callable_mapping;
  // Byte offset of the source-ordered host variant.
  uint32_t variant_offset;
} iree_vm_program_function_field_abi_t;

// Byte offsets for one contiguous root bank and its provider-visible overflow
// window. An unused window points at offset zero and is never accessed for a
// valid signature ordinal.
typedef struct iree_vm_program_root_bank_layout_t {
  // First cell in the complete contiguous bank.
  uint32_t direct_offset;
  // Cell 16 when present, otherwise the unused offset-zero sentinel.
  uint32_t overflow_offset;
} iree_vm_program_root_bank_layout_t;

// Precomputed placement of every root bank in invocation-owned storage.
typedef struct iree_vm_program_root_layout_t {
  // Value argument bank placement.
  iree_vm_program_root_bank_layout_t value_arguments;
  // Ref argument bank placement.
  iree_vm_program_root_bank_layout_t ref_arguments;
  // Value result bank placement.
  iree_vm_program_root_bank_layout_t value_results;
  // Ref result bank placement.
  iree_vm_program_root_bank_layout_t ref_results;
  // Function argument bank placement.
  iree_vm_program_root_bank_layout_t function_arguments;
  // Function result bank placement.
  iree_vm_program_root_bank_layout_t function_results;
  // Exact invocation-owned bytes occupied by all banks.
  uint32_t storage_size;
} iree_vm_program_root_layout_t;

// Program-linked ABI for one unique structural callable signature. A dense
// nonzero callable token directly indexes this table without a module walk,
// provider query, or signature scan.
typedef struct iree_vm_program_callable_abi_t {
  // Scalar argument fields in physical value-bank order.
  const iree_vm_program_scalar_field_abi_t* value_arguments;
  // Ref argument fields in physical ref-bank order.
  const iree_vm_program_ref_field_abi_t* ref_arguments;
  // Function argument fields in physical function-bank order.
  const iree_vm_program_function_field_abi_t* function_arguments;
  // Scalar result fields in physical value-bank order.
  const iree_vm_program_scalar_field_abi_t* value_results;
  // Ref result fields in physical ref-bank order.
  const iree_vm_program_ref_field_abi_t* ref_results;
  // Function result fields in physical function-bank order.
  const iree_vm_program_function_field_abi_t* function_results;
  // Exact physical argument bank counts.
  iree_vm_program_bank_counts_t argument_counts;
  // Exact physical result bank counts.
  iree_vm_program_bank_counts_t result_counts;
  // Precomputed root-bank placement for invocation marshalling.
  iree_vm_program_root_layout_t root_layout;
} iree_vm_program_callable_abi_t;

// Cached executable initializer plan.
typedef struct iree_vm_program_initializer_t {
  // Complete packed target bits, or zero when no initializer is exported.
  uint64_t target_bits;
  // Direct canonical callable ABI, or null when no initializer is exported.
  const iree_vm_program_callable_abi_t* callable_abi;
} iree_vm_program_initializer_t;

// Immutable program-local view of one retained generic module.
struct iree_vm_linked_module_t {
  // Retained immutable module implementation.
  iree_vm_module_t* module;
  // Immutable module-local resolved import target words.
  const uint64_t* import_target_bits;
  // First entry in the program's flat callable mapping array.
  uint32_t callable_base;
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
  // Flat four-byte mapping words for module-local callable declarations.
  uint32_t* callable_mappings;
  // Program-linked ABIs indexed by dense nonzero structural token.
  iree_vm_program_callable_abi_t* callable_abis;
  // Total number of module-local entries in |callable_mappings|.
  uint32_t callable_mapping_count;
  // Number of initialized unique entries in |callable_abis|.
  uint32_t callable_abi_count;
  // Total max-aligned opaque process-state bytes.
  iree_host_size_t process_storage_size;
};

static_assert(sizeof(void*) != 8 || sizeof(iree_vm_linked_module_t) == 24,
              "64-bit linked modules must remain 24 bytes");
static_assert(sizeof(void*) != 4 || sizeof(iree_vm_linked_module_t) == 16,
              "32-bit linked modules must remain 16 bytes");
static_assert(sizeof(uint32_t) == 4,
              "callable mappings must remain four bytes");
static_assert(sizeof(void*) != 8 ||
                  sizeof(iree_vm_program_scalar_field_abi_t) == 16,
              "64-bit scalar field ABIs must remain 16 bytes");
static_assert(sizeof(void*) != 8 ||
                  sizeof(iree_vm_program_ref_field_abi_t) == 16,
              "64-bit ref field ABIs must remain 16 bytes");
static_assert(sizeof(iree_vm_program_function_field_abi_t) == 8,
              "function field ABIs must remain 8 bytes");
static_assert(sizeof(iree_vm_program_root_bank_layout_t) == 8,
              "root bank layouts must remain 8 bytes");
static_assert(sizeof(iree_vm_program_root_layout_t) == 52,
              "root layouts must remain 52 bytes");
static_assert(sizeof(void*) != 8 ||
                  sizeof(iree_vm_program_callable_abi_t) == 112,
              "64-bit callable ABIs must remain 112 bytes");
static_assert(sizeof(void*) != 8 || sizeof(iree_vm_program_initializer_t) == 16,
              "64-bit initializer records must remain 16 bytes");
static_assert(sizeof(void*) != 8 || sizeof(iree_vm_program_t) <= 128,
              "64-bit program headers must fit the persistent size budget");

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

// Returns the logical argument count of one callable ABI.
static inline iree_host_size_t iree_vm_program_callable_abi_argument_count(
    const iree_vm_program_callable_abi_t* callable_abi) {
  return (iree_host_size_t)callable_abi->argument_counts.value_count +
         callable_abi->argument_counts.ref_count +
         callable_abi->argument_counts.function_count;
}

// Returns the logical result count of one callable ABI.
static inline iree_host_size_t iree_vm_program_callable_abi_result_count(
    const iree_vm_program_callable_abi_t* callable_abi) {
  return (iree_host_size_t)callable_abi->result_counts.value_count +
         callable_abi->result_counts.ref_count +
         callable_abi->result_counts.function_count;
}

// Returns whether one token names a linked callable ABI.
static inline bool iree_vm_program_callable_token_is_valid(
    const iree_vm_program_t* program, uint32_t callable_token) {
  return callable_token != 0 && callable_token <= program->callable_abi_count;
}

// Resolves one canonical token with a direct indexed load.
static inline const iree_vm_program_callable_abi_t*
iree_vm_program_resolve_callable_abi(const iree_vm_program_t* program,
                                     uint32_t callable_token) {
  return iree_vm_program_callable_token_is_valid(program, callable_token)
             ? &program->callable_abis[callable_token - 1]
             : NULL;
}

// Returns whether |function_ref| is null or satisfies one linked callable
// mapping. The check is structural and performs no provider calls, name lookup,
// allocation, or retention.
static inline bool iree_vm_program_function_ref_matches_mapping(
    const iree_vm_program_t* program, iree_vm_function_ref_t function_ref,
    uint32_t expected_mapping) {
  if (iree_vm_function_ref_is_null(function_ref)) return true;
  if (function_ref.program_bits != (uint64_t)(uintptr_t)program ||
      (function_ref.target_bits & 3u) != 0) {
    return false;
  }
  const uint16_t module_ordinal =
      iree_vm_program_target_module_ordinal(function_ref.target_bits);
  const uint16_t function_ordinal =
      iree_vm_program_target_function_ordinal(function_ref.target_bits);
  if (module_ordinal >= program->linked_module_count ||
      function_ordinal >= program->linked_modules[module_ordinal]
                              .module->descriptor->counts.function_count) {
    return false;
  }
  const uint32_t target_token =
      iree_vm_program_target_callable_token(function_ref.target_bits);
  if (!iree_vm_program_callable_token_is_valid(program, target_token)) {
    return false;
  }
  if (target_token != iree_vm_program_callable_token(expected_mapping)) {
    return false;
  }
  return !iree_vm_program_target_may_yield(function_ref.target_bits) ||
         iree_vm_program_callable_may_yield(expected_mapping);
}

// Returns whether |function_ref| satisfies one module-local callable type.
static inline bool iree_vm_program_function_ref_matches(
    const iree_vm_program_t* program, iree_vm_function_ref_t function_ref,
    const iree_vm_linked_module_t* signature_module,
    uint16_t expected_callable_type_ordinal) {
  const uint32_t expected_mapping =
      program->callable_mappings[signature_module->callable_base +
                                 expected_callable_type_ordinal];
  return iree_vm_program_function_ref_matches_mapping(program, function_ref,
                                                      expected_mapping);
}

// Finds one exact retained module by implementation identity.
const iree_vm_linked_module_t* iree_vm_program_find_linked_module(
    const iree_vm_program_t* program, const iree_vm_module_t* module,
    iree_host_size_t* out_ordinal);

// Finds one exact retained module by link name.
const iree_vm_linked_module_t* iree_vm_program_lookup_linked_module(
    const iree_vm_program_t* program, iree_string_view_t name,
    iree_host_size_t* out_ordinal);

#endif  // IREE_VM_PROGRAM_STORAGE_H_
