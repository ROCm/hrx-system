// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Physical effects of a retained call boundary.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_CALL_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_CALL_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_low_descriptor_set_t;

typedef struct loom_low_call_clobber_t {
  // Register class in the caller's resolved descriptor set.
  uint16_t register_class;
  // Physical register view, or first linear unit in this class.
  uint32_t location;
  // Number of linear units; an explicit register view always uses one row.
  uint32_t count;
} loom_low_call_clobber_t;

typedef struct loom_low_call_clobber_list_t {
  // Borrowed immutable physical writes performed by the callee or transport.
  // Rows are ordered by target storage identity and then location so consumers
  // can merge aliasing intervals without sorting or allocating.
  const loom_low_call_clobber_t* values;
  // Number of rows in |values|.
  iree_host_size_t count;
} loom_low_call_clobber_list_t;

// One ABI location at invocation entry, outgoing call, or incoming result.
// The boundary supplies its own register class, width, and location; it does
// not constrain storage during the value's SSA lifetime.
typedef struct loom_low_allocation_abi_location_t {
  // Register-like allocation kind, or UNASSIGNED for a storage ABI operand.
  uint16_t location_kind;
  // Register class in the allocation's resolved descriptor set.
  uint16_t descriptor_reg_class_id;
  // Physical view or first linear register unit in the boundary register class.
  uint32_t location_base;
} loom_low_allocation_abi_location_t;

static_assert(sizeof(loom_low_allocation_abi_location_t) == 8,
              "ABI locations must remain compact");

typedef struct loom_low_call_contract_t {
  // ABI registers indexed by logical argument. A missing suffix or UNASSIGNED
  // entry is stored through the target's storage ABI before register transport
  // at the same call. Those consumed inputs may then supply cycle scratch.
  const loom_low_allocation_abi_location_t* arguments;
  // Length of |arguments|; queries may return a convention's maximum prefix.
  uint16_t argument_count;
  // ABI registers indexed by logical result, with the same suffix convention.
  const loom_low_allocation_abi_location_t* results;
  // Length of |results|.
  uint16_t result_count;
  // Physical storage overwritten between argument consumption and results.
  loom_low_call_clobber_list_t clobbers;
} loom_low_call_contract_t;

// The target resolves and validates the callee's convention before allocation.
// The query reads that retained binding; it does not inspect bodies or infer
// effects, and must return a non-NULL contract after optional validation.
// Clobbers occur between argument consumption and result production: values
// live across the boundary require preserved storage, while dying arguments
// and new results may use these registers. Asynchronous execution contracts
// must additionally account for outstanding physical storage leases.
typedef struct loom_low_call_contract_provider_t {
  // Optional target-boundary validation performed once per retained call.
  iree_status_t (*validate)(void* user_data, loom_symbol_ref_t callee);
  // Optional infallible query; NULL when calls have no allocation contract.
  const loom_low_call_contract_t* (*query)(void* user_data,
                                           loom_symbol_ref_t callee);
  // Optional physical writes shared by every call in the resolved caller
  // profile. Allocation queries this once and combines it with each callee's
  // retained clobbers without multiplying contracts by caller profile.
  loom_low_call_clobber_list_t (*query_common_clobbers)(
      void* user_data, const struct loom_low_descriptor_set_t* descriptor_set);
  // Borrowed target convention bindings, valid across allocation repair.
  void* user_data;
} loom_low_call_contract_provider_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_CALL_H_
