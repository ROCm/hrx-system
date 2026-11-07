// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Declared interfaces and storage of one immutable, verified Low function.

#ifndef LOOM_CODEGEN_LOW_FUNCTION_REQUIREMENTS_H_
#define LOOM_CODEGEN_LOW_FUNCTION_REQUIREMENTS_H_

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/read_only_data.h"
#include "loom/codegen/low/storage_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

// One unique direct Low callee retained in first-call order.
typedef struct loom_low_direct_callee_requirement_t {
  // Directly called symbol.
  loom_symbol_ref_t symbol;
  // First call operation naming |symbol|.
  const loom_op_t* first_call_op;
} loom_low_direct_callee_requirement_t;

// Ordered unique direct Low callees of one function.
typedef struct loom_low_direct_callee_requirement_list_t {
  // Callee records in first-call order.
  const loom_low_direct_callee_requirement_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
} loom_low_direct_callee_requirement_list_t;

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
  loom_low_read_only_data_set_t read_only_data;
  // Direct Low callees in first-call order, with duplicate calls collapsed.
  loom_low_direct_callee_requirement_list_t direct_callees;
  // Number of operations in the immediate function body.
  iree_host_size_t node_count;
  // Number of Low returns that transfer control back to a caller.
  uint32_t return_count;
  // Number of Low operations materializing function-local storage addresses.
  uint32_t storage_address_count;
  // Source-relative managed borrow operations requiring protection decisions.
  uint32_t borrow_count;
} loom_low_function_requirements_t;

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
