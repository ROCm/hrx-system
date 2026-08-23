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

// Program-linked execution plan for one module-local callable declaration.
// Structurally equal declarations carry the same token; that token directly
// indexes the first equal entry without a module walk or provider query.
typedef struct iree_vm_program_callable_t {
  // Stable source-ordered argument types in the representative module.
  const iree_vm_module_signature_type_t* argument_types;
  // Stable source-ordered result types in the representative module.
  const iree_vm_module_signature_type_t* result_types;
  // Representative linked module resolving ref and nested callable ordinals.
  const iree_vm_linked_module_t* signature_module;
  // Nonzero canonical token and this declaration's suspension permission.
  uint32_t mapping;
  // Exact physical argument bank counts.
  iree_vm_program_bank_counts_t argument_counts;
  // Exact physical result bank counts.
  iree_vm_program_bank_counts_t result_counts;
  // Shared scalar result type, or INVALID for empty or heterogeneous results.
  iree_vm_scalar_type_t uniform_result_scalar_type;
} iree_vm_program_callable_t;

// Cached executable initializer plan.
typedef struct iree_vm_program_initializer_t {
  // Complete packed target bits, or zero when no initializer is exported.
  uint64_t target_bits;
  // Direct canonical callable plan, or null when no initializer is exported.
  const iree_vm_program_callable_t* callable;
} iree_vm_program_initializer_t;

// Immutable program-local view of one retained generic module.
struct iree_vm_linked_module_t {
  // Retained immutable module implementation.
  iree_vm_module_t* module;
  // Immutable module-local resolved import target words.
  const uint64_t* import_target_bits;
  // First entry in the program's flat callable plan array.
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
  // Flat module-local callable plans.
  iree_vm_program_callable_t* callables;
  // Total number of callable plans in |callables|.
  uint32_t callable_count;
  // Total max-aligned opaque process-state bytes.
  iree_host_size_t process_storage_size;
};

static_assert(sizeof(void*) != 8 || sizeof(iree_vm_linked_module_t) == 24,
              "64-bit linked modules must remain 24 bytes");
static_assert(sizeof(void*) != 4 || sizeof(iree_vm_linked_module_t) == 16,
              "32-bit linked modules must remain 16 bytes");
static_assert(sizeof(void*) != 8 || sizeof(iree_vm_program_callable_t) == 48,
              "64-bit callable plans must remain 48 bytes");
static_assert(sizeof(void*) != 8 || sizeof(iree_vm_program_initializer_t) == 16,
              "64-bit initializer plans must remain 16 bytes");

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

// Returns the source-ordered signature of one callable plan.
static inline iree_vm_module_signature_t iree_vm_program_callable_signature(
    const iree_vm_program_callable_t* callable) {
  const iree_vm_module_signature_t signature = {
      {callable->argument_types,
       (iree_host_size_t)callable->argument_counts.value_count +
           callable->argument_counts.ref_count +
           callable->argument_counts.function_count},
      {callable->result_types,
       (iree_host_size_t)callable->result_counts.value_count +
           callable->result_counts.ref_count +
           callable->result_counts.function_count},
  };
  return signature;
}

// Returns true when both signature sides use only value cells.
static inline bool iree_vm_program_callable_is_scalar_only(
    const iree_vm_program_callable_t* callable) {
  return callable->argument_counts.ref_count == 0 &&
         callable->argument_counts.function_count == 0 &&
         callable->result_counts.ref_count == 0 &&
         callable->result_counts.function_count == 0;
}

// Resolves one nonzero canonical token with a direct indexed load.
static inline const iree_vm_program_callable_t*
iree_vm_program_resolve_callable(const iree_vm_program_t* program,
                                 uint32_t callable_token) {
  if (callable_token == 0 || callable_token > program->callable_count) {
    return NULL;
  }
  const iree_vm_program_callable_t* callable =
      &program->callables[callable_token - 1];
  return iree_vm_program_callable_token(callable->mapping) == callable_token
             ? callable
             : NULL;
}

// Returns whether |function_ref| is null or satisfies one linked callable
// declaration in |signature_module|. The check is structural and performs no
// provider calls, name lookup, allocation, or retention.
static inline bool iree_vm_program_function_ref_matches(
    const iree_vm_program_t* program, iree_vm_function_ref_t function_ref,
    const iree_vm_linked_module_t* signature_module,
    uint16_t expected_callable_type_ordinal) {
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
  if (!iree_vm_program_resolve_callable(program, target_token)) return false;
  const uint32_t expected_mapping =
      program
          ->callables[signature_module->callable_base +
                      expected_callable_type_ordinal]
          .mapping;
  if (target_token != iree_vm_program_callable_token(expected_mapping)) {
    return false;
  }
  return !iree_vm_program_target_may_yield(function_ref.target_bits) ||
         iree_vm_program_callable_may_yield(expected_mapping);
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
