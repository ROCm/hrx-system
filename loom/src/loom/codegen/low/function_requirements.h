// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Declared interfaces and storage of one immutable, verified Low function.

#ifndef LOOM_CODEGEN_LOW_FUNCTION_REQUIREMENTS_H_
#define LOOM_CODEGEN_LOW_FUNCTION_REQUIREMENTS_H_

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/storage_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

// One read-only data symbol referenced by a Low descriptor immediate.
typedef struct loom_low_read_only_data_requirement_t {
  // Module-local symbol retained by the descriptor immediate.
  loom_symbol_ref_t symbol;
  // Source definition supplying location and symbol ownership.
  const loom_op_t* definition;
  // Immutable definition bytes.
  iree_const_byte_span_t contents;
  // Required power-of-two byte alignment.
  uint64_t minimum_alignment;
  // Definitions that must occupy disjoint target-defined storage banks.
  loom_symbol_ref_array_t bank_conflicts;
} loom_low_read_only_data_requirement_t;

// Requirements available before scheduling, allocation, or native emission.
//
// Resource operations borrow the source function. Storage records belong to
// the supplied arena and use source value IDs. Read-only data records borrow
// their source definitions and payloads; their symbol index belongs to the same
// immutable module snapshot. All requirements must be rebuilt after the
// function or its declarations change. Storage includes explicit reservations,
// not spills that allocation may introduce.
typedef struct loom_low_function_requirements_t {
  // Last entry live-in/resource declaration, or NULL for an empty preamble.
  const loom_op_t* entry_preamble_end;
  // Last live-in/resource/spill in the contiguous entry traffic prefix, or
  // NULL when ordinary traffic begins immediately.
  const loom_op_t* entry_traffic_end;
  // Low resource declarations in function body order.
  const loom_op_t* const* resources;
  // Number of resource declarations.
  iree_host_size_t resource_count;
  // Packed function-local storage reservations.
  loom_low_storage_layout_t storage_layout;
  // Referenced read-only data in first-reference order.
  const loom_low_read_only_data_requirement_t* read_only_data;
  // Number of referenced read-only data records.
  iree_host_size_t read_only_data_count;
  // Direct Low callees in first-call order, with duplicate calls collapsed.
  loom_symbol_ref_array_t direct_callees;
  // Open-addressed read-only data ordinals plus one; zero marks an empty slot.
  // Kept NULL for at most four records, which use a bounded linear lookup.
  const uint16_t* read_only_data_ordinal_slots;
  // Power-of-two number of entries in read_only_data_ordinal_slots.
  iree_host_size_t read_only_data_ordinal_slot_count;
  // Number of operations in the immediate function body.
  iree_host_size_t node_count;
  // Number of Low returns that transfer control back to a caller.
  uint32_t return_count;
  // Number of Low operations materializing function-local storage addresses.
  uint32_t storage_address_count;
} loom_low_function_requirements_t;

// Returns the retained read-only data ordinal for |symbol|, or UINT32_MAX when
// the function does not reference a matching module-local definition.
uint32_t loom_low_function_requirements_read_only_data_ordinal(
    const loom_low_function_requirements_t* requirements,
    loom_symbol_ref_t symbol);

// Inventories one verified Low function body using its resolved representation
// contract. The function's immediate region owns its ABI imports and storage;
// nested regions are not entry scopes.
iree_status_t loom_low_function_requirements_build(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, const loom_region_t* body,
    iree_arena_allocator_t* arena,
    loom_low_function_requirements_t* out_requirements);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_FUNCTION_REQUIREMENTS_H_
