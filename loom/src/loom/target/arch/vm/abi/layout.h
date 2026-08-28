// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Logical and physical VM call ABI layout.

#ifndef LOOM_TARGET_ARCH_VM_ABI_LAYOUT_H_
#define LOOM_TARGET_ARCH_VM_ABI_LAYOUT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/vm/scalar.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Physical register bank used by one logical callable field.
typedef uint8_t loom_vm_call_abi_bank_t;
enum loom_vm_call_abi_bank_e {
  // Invalid or unclassified callable field.
  LOOM_VM_CALL_ABI_BANK_NONE = 0,
  // Exact scalar bits carried in a vm.value register.
  LOOM_VM_CALL_ABI_BANK_VALUE = 1,
  // Managed object state carried in a vm.ref register.
  LOOM_VM_CALL_ABI_BANK_REF = 2,
  // Non-owning callable state carried in a vm.function register.
  LOOM_VM_CALL_ABI_BANK_FUNCTION = 3,
};

// Counts of logical fields in each independent physical register bank.
typedef struct loom_vm_call_abi_bank_counts_t {
  // Number of value-register fields.
  uint16_t value;
  // Number of ref-register fields.
  uint16_t ref;
  // Number of function-register fields.
  uint16_t function;
} loom_vm_call_abi_bank_counts_t;

// Physical placement of one source-ordered logical callable field.
typedef struct loom_vm_call_abi_field_layout_t {
  // Independent physical register bank carrying the field.
  loom_vm_call_abi_bank_t bank;
  // Zero-based field ordinal within |bank|.
  uint16_t bank_ordinal;
} loom_vm_call_abi_field_layout_t;

// Source-ordered layout of one side of a callable signature.
typedef struct loom_vm_call_abi_side_layout_t {
  // Arena-owned field layouts in logical source order.
  const loom_vm_call_abi_field_layout_t* fields;
  // Number of entries in |fields|.
  uint16_t field_count;
  // Independent physical-bank field counts.
  loom_vm_call_abi_bank_counts_t bank_counts;
} loom_vm_call_abi_side_layout_t;

// Complete source-ordered logical callable layout.
typedef struct loom_vm_call_abi_layout_t {
  // Structural function type from the function ABI layout attribute.
  loom_type_t signature;
  // Logical argument fields and their physical-bank ordinals.
  loom_vm_call_abi_side_layout_t arguments;
  // Logical result fields and their physical-bank ordinals.
  loom_vm_call_abi_side_layout_t results;
} loom_vm_call_abi_layout_t;

// Canonical caller-local packet layout for one logical callsite.
//
// Every call in a function reuses the same local-storage prefixes. Value
// overflow cells occupy naturally aligned 64-bit bytes, followed by result
// cells. Ref and function overflow slots use their corresponding typed local
// arrays with arguments followed by results. Direct ref arguments are fresh
// staging values and are moved into the child packet.
typedef struct loom_vm_call_abi_packet_layout_t {
  // Logical argument counts in each independent register bank.
  loom_vm_call_abi_bank_counts_t arguments;
  // Logical result counts in each independent register bank.
  loom_vm_call_abi_bank_counts_t results;
  // Ownership-transfer mask covering every direct ref argument.
  uint16_t direct_ref_move_mask;
  // Complete caller-local byte prefix required by value overflow cells.
  uint32_t local_byte_length;
  // Complete caller-local ref prefix required by ref overflow slots.
  uint32_t local_ref_count;
  // Complete caller-local function prefix required by function overflow slots.
  uint32_t local_function_count;
} loom_vm_call_abi_packet_layout_t;

// Maps one Loom scalar to its exact VM callable scalar type.
//
// Index and offset use the version-zero 64-bit host-program ABI. I1 uses the
// I8 carrier with canonical zero/one bits. Returns NONE for a non-scalar or a
// scalar unavailable in the Core ABI.
iree_vm_scalar_type_t loom_vm_call_abi_scalar_type(
    loom_scalar_type_t scalar_type);

// Returns the number of fields beyond the direct register prefix.
uint16_t loom_vm_call_abi_overflow_count(uint16_t count);

// Classifies one logical source type into its VM callable register bank.
//
// Returns false when the type is not a scalar, an exact managed reference, or
// a structurally valid func.ref type.
bool loom_vm_call_abi_try_classify_logical_type(
    const loom_module_t* module, loom_type_t type,
    loom_vm_call_abi_bank_t* out_bank);

// Classifies one typed Core VM register used at a callable boundary.
//
// Version zero requires exactly one physical register unit carrying a scalar,
// an exact managed-reference type, or a structurally valid func.ref type.
iree_status_t loom_vm_call_abi_classify_type(const loom_module_t* module,
                                             loom_type_t type,
                                             loom_vm_call_abi_bank_t* out_bank);

// Resolves the required structural signature from an abi_layout dictionary.
iree_status_t loom_vm_call_abi_layout_resolve_signature(
    const loom_module_t* module, loom_named_attr_slice_t abi_layout,
    loom_type_t* out_signature);

// Builds deterministic per-bank ordinals for a structural VM signature.
//
// The field arrays are owned by |arena| and remain in source order. Value,
// ref, and function ordinals advance independently from zero on each side.
iree_status_t loom_vm_call_abi_layout_build(
    const loom_module_t* module, loom_type_t signature,
    iree_arena_allocator_t* arena, loom_vm_call_abi_layout_t* out_layout);

// Builds the canonical caller-local packet layout for one logical callsite.
//
// The argument and result values remain in source order. The returned bank
// counts determine direct register ordinals and overflow slot offsets without
// retaining a per-field mapping.
iree_status_t loom_vm_call_abi_packet_layout_build(
    const loom_module_t* module, const loom_value_id_t* arguments,
    iree_host_size_t argument_count, const loom_value_id_t* results,
    iree_host_size_t result_count,
    loom_vm_call_abi_packet_layout_t* out_layout);

// Builds a canonical abi_layout dictionary containing the logical mapped Low
// function signature. The signature is independent of the physical function
// boundary and remains unchanged as boundary values are materialized.
iree_status_t loom_vm_call_abi_layout_make_attr(
    loom_module_t* module, const loom_type_t* argument_types,
    iree_host_size_t argument_count, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_attribute_t* out_attr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_ABI_LAYOUT_H_
