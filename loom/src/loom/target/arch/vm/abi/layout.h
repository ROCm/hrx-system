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

// Physical register representation of one logical callable field.
typedef struct loom_vm_call_abi_register_layout_t {
  // Independent physical register bank carrying the field.
  loom_vm_call_abi_bank_t bank;
  // Number of contiguous register units carrying the field.
  uint16_t unit_count;
} loom_vm_call_abi_register_layout_t;

// Counts of register units in each independent physical register bank.
typedef struct loom_vm_call_abi_bank_counts_t {
  // Number of value-register units.
  uint16_t value;
  // Number of ref-register units.
  uint16_t ref;
  // Number of function-register units.
  uint16_t function;
} loom_vm_call_abi_bank_counts_t;

enum {
  // Maximum physical units carried by one logical callable field.
  LOOM_VM_CALL_ABI_MAX_FIELD_UNIT_COUNT = 256,
};

// Physical placement of one source-ordered logical callable field.
typedef struct loom_vm_call_abi_field_layout_t {
  // Independent physical register bank carrying the field.
  loom_vm_call_abi_bank_t bank;
  // Number of contiguous register units carrying the field.
  uint16_t unit_count;
  // Zero-based register-unit ordinal within |bank|.
  uint16_t bank_ordinal;
} loom_vm_call_abi_field_layout_t;

// Source-ordered layout of one side of a callable signature.
typedef struct loom_vm_call_abi_side_layout_t {
  // Arena-owned field layouts in logical source order.
  const loom_vm_call_abi_field_layout_t* fields;
  // Number of entries in |fields|.
  uint16_t field_count;
  // Independent physical-bank register-unit counts.
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

// Logical source fields used to construct one side of an ABI layout.
typedef struct loom_vm_call_abi_source_fields_t {
  // Borrowed target-mapped logical field types.
  const loom_type_t* types;
  // Borrowed source values supplying optional presentation names.
  const loom_value_id_t* values;
  // Borrowed explicit presentation names when no source values exist.
  const iree_string_view_t* presentation_names;
  // Number of entries in |types| and the selected presentation source.
  iree_host_size_t count;
} loom_vm_call_abi_source_fields_t;

// Canonical caller-local packet layout for one logical callsite.
//
// Every call in a function reuses the same local-storage prefixes. Value
// overflow cells occupy naturally aligned 64-bit bytes, followed by result
// cells. Ref and function overflow slots use their corresponding typed local
// arrays with arguments followed by results. Direct ref arguments are fresh
// staging values and are moved into the child packet.
typedef struct loom_vm_call_abi_packet_layout_t {
  // Argument register-unit counts in each independent register bank.
  loom_vm_call_abi_bank_counts_t arguments;
  // Result register-unit counts in each independent register bank.
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

// Returns the number of register units beyond the direct prefix.
uint16_t loom_vm_call_abi_overflow_count(uint16_t count);

// Classifies one logical source type into its VM register representation.
//
// Static vectors use one value-register unit per logical element. Returns false
// when the type has no exact Core VM representation.
bool loom_vm_call_abi_try_classify_logical_type(
    const loom_module_t* module, loom_type_t type,
    loom_vm_call_abi_register_layout_t* out_layout);

// Classifies one typed Core VM register used at a callable boundary.
//
// The physical register class and unit count must exactly match the retained
// logical type.
iree_status_t loom_vm_call_abi_classify_type(
    const loom_module_t* module, loom_type_t type,
    loom_vm_call_abi_register_layout_t* out_layout);

// Resolves the required structural signature from an abi_layout dictionary.
iree_status_t loom_vm_call_abi_layout_resolve_signature(
    const loom_module_t* module, loom_named_attr_slice_t abi_layout,
    loom_type_t* out_signature);

// Resolves the optional source-authored signature from an abi_layout.
//
// When present, the authored arguments and results must each be exact prefixes
// of |abi_signature|. When absent, |abi_signature| is returned unchanged.
iree_status_t loom_vm_call_abi_layout_resolve_authored_signature(
    const loom_module_t* module, loom_named_attr_slice_t abi_layout,
    loom_type_t abi_signature, loom_type_t* out_authored_signature);

// Resolves optional source field names from an abi_layout dictionary.
//
// Present name tables contain exactly one entry per logical field, including
// empty string values for anonymous fields. The returned slices borrow the
// immutable layout dictionary. Malformed tables fail rather than silently
// dropping presentation data.
iree_status_t loom_vm_call_abi_layout_resolve_presentation_names(
    const loom_module_t* module, loom_named_attr_slice_t abi_layout,
    uint16_t argument_count, uint16_t result_count,
    loom_named_attr_slice_t* out_argument_names,
    loom_named_attr_slice_t* out_result_names);

// Adds missing source field-name tables to an existing ABI layout.
//
// Existing tables and all unrelated layout entries are preserved. A side with
// no authored names does not create a table. |out_changed| is true only when
// |out_attr| contains a newly materialized dictionary.
iree_status_t loom_vm_call_abi_layout_preserve_presentation_names(
    loom_module_t* module, loom_named_attr_slice_t abi_layout,
    loom_vm_call_abi_source_fields_t arguments,
    loom_vm_call_abi_source_fields_t results,
    iree_arena_allocator_t* scratch_arena, bool* out_changed,
    loom_attribute_t* out_attr);

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

// Builds a canonical abi_layout dictionary containing the complete logical Low
// ABI signature, an optional source-authored prefix signature, and optional
// source field names.
//
// The signature is independent of the physical function boundary and remains
// unchanged as boundary values are materialized. |authored_signature| may be
// none when the complete ABI signature is also the authored signature. Exactly
// one of |values| and |presentation_names| may supply names; when both are NULL
// the side retains its types without presentation names.
iree_status_t loom_vm_call_abi_layout_make_attr(
    loom_module_t* module, loom_vm_call_abi_source_fields_t arguments,
    loom_vm_call_abi_source_fields_t results, loom_type_t authored_signature,
    iree_arena_allocator_t* scratch_arena, loom_attribute_t* out_attr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_ABI_LAYOUT_H_
