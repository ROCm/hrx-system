// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_EXECUTION_H_
#define IREE_VM_EXECUTION_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/function.h"
#include "iree/vm/ref.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_vm_frame_t iree_vm_frame_t;
typedef struct iree_vm_invocation_t iree_vm_invocation_t;
typedef struct iree_vm_linked_module_t iree_vm_linked_module_t;
typedef struct iree_vm_module_t iree_vm_module_t;

//===----------------------------------------------------------------------===//
// Execution State
//===----------------------------------------------------------------------===//

// Successful execution-driving outcomes. Suspension is ordinary control flow
// and never encoded as an iree_status_t.
enum iree_vm_execution_outcome_e {
  IREE_VM_EXECUTION_OUTCOME_COMPLETED = 0u,
  IREE_VM_EXECUTION_OUTCOME_SUSPENDED = 1u,
};
typedef uint32_t iree_vm_execution_outcome_t;

// Transient execution view supplied to one module callback. The process state
// pointer is stable across suspension and null exactly for zero-sized state.
typedef struct iree_vm_module_execution_t {
  // Active invocation and composite frame stack.
  iree_vm_invocation_t* invocation;
  // Immutable program linkage for this module slot.
  const iree_vm_linked_module_t* linked_module;
  // Opaque process-owned state visible only to this module.
  void* process_storage;
} iree_vm_module_execution_t;

// Private module-local function behavior flags.
enum iree_vm_module_function_flag_bits_e {
  IREE_VM_MODULE_FUNCTION_FLAG_NONE = 0u,
  // The function may suspend before completing.
  IREE_VM_MODULE_FUNCTION_FLAG_MAY_YIELD = 1u << 0,
};
typedef uint32_t iree_vm_module_function_flags_t;

// Correlated private facts for materializing one local function value.
typedef struct iree_vm_module_local_function_t {
  // Module-local executable function ordinal.
  uint16_t function_ordinal;
  // Module-local callable type defining the structural contract.
  uint16_t callable_type_ordinal;
  // Actual function behavior flags.
  iree_vm_module_function_flags_t flags;
} iree_vm_module_local_function_t;
static_assert(sizeof(iree_vm_module_local_function_t) == 8,
              "local function descriptors must remain eight bytes");

//===----------------------------------------------------------------------===//
// Physical Call Packets
//===----------------------------------------------------------------------===//

// Number of directly addressed registers in each physical packet bank.
// Higher logical ordinals use the corresponding zero-based overflow bank.
enum { IREE_VM_CALL_DIRECT_REGISTER_COUNT = 16 };

typedef struct iree_vm_call_value_arguments_t {
  // Read-only arguments below IREE_VM_CALL_DIRECT_REGISTER_COUNT.
  const uint64_t* direct;
  // Read-only arguments at or above the direct-register count.
  const uint64_t* overflow;
} iree_vm_call_value_arguments_t;

typedef struct iree_vm_call_ref_arguments_t {
  // Arguments below the direct-register count, mutable for explicit moves.
  iree_vm_ref_t* direct;
  // Remaining arguments, mutable only for explicit moves.
  iree_vm_ref_t* overflow;
} iree_vm_call_ref_arguments_t;

typedef struct iree_vm_call_function_arguments_t {
  // Read-only arguments below the direct-register count.
  const iree_vm_function_ref_t* direct;
  // Read-only arguments at or above the direct-register count.
  const iree_vm_function_ref_t* overflow;
} iree_vm_call_function_arguments_t;

typedef struct iree_vm_call_value_results_t {
  // Write-only results below the direct-register count.
  uint64_t* direct;
  // Write-only results at or above the direct-register count.
  uint64_t* overflow;
} iree_vm_call_value_results_t;

typedef struct iree_vm_call_ref_results_t {
  // Results below the direct-register count.
  iree_vm_ref_t* direct;
  // Results at or above the direct-register count.
  iree_vm_ref_t* overflow;
} iree_vm_call_ref_results_t;

typedef struct iree_vm_call_function_results_t {
  // Write-only results below the direct-register count.
  iree_vm_function_ref_t* direct;
  // Write-only results at or above the direct-register count.
  iree_vm_function_ref_t* overflow;
} iree_vm_call_function_results_t;

// Implementer-only physical call packet. The exact logical signature derives
// every bank extent. Nonnull bases remain stable until the call completes or
// unwinds; a yielding implementation preserves any needed bases in its frame.
typedef struct iree_vm_call_packet_t {
  // Physical value argument banks.
  iree_vm_call_value_arguments_t value_arguments;
  // Physical ref argument banks.
  iree_vm_call_ref_arguments_t ref_arguments;
  // Physical value result banks.
  iree_vm_call_value_results_t value_results;
  // Physical ref result banks.
  iree_vm_call_ref_results_t ref_results;
  // Physical function argument banks.
  iree_vm_call_function_arguments_t function_arguments;
  // Physical function result banks.
  iree_vm_call_function_results_t function_results;
} iree_vm_call_packet_t;

// Loads exact bits from one valid physical value argument.
static inline uint64_t iree_vm_call_value_argument_load(
    const iree_vm_call_packet_t* call, uint16_t ordinal) {
  return ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT
             ? call->value_arguments.direct[ordinal]
             : call->value_arguments
                   .overflow[ordinal - IREE_VM_CALL_DIRECT_REGISTER_COUNT];
}

// Stores exact bits to one valid physical value result.
static inline void iree_vm_call_value_result_store(
    const iree_vm_call_packet_t* call, uint16_t ordinal, uint64_t value) {
  uint64_t* slot =
      ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT
          ? &call->value_results.direct[ordinal]
          : &call->value_results
                 .overflow[ordinal - IREE_VM_CALL_DIRECT_REGISTER_COUNT];
  *slot = value;
}

// Loads one complete valid function argument.
static inline iree_vm_function_ref_t iree_vm_call_function_argument_load(
    const iree_vm_call_packet_t* call, uint16_t ordinal) {
  return ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT
             ? call->function_arguments.direct[ordinal]
             : call->function_arguments
                   .overflow[ordinal - IREE_VM_CALL_DIRECT_REGISTER_COUNT];
}

// Stores one complete valid function result.
static inline void iree_vm_call_function_result_store(
    const iree_vm_call_packet_t* call, uint16_t ordinal,
    iree_vm_function_ref_t function_ref) {
  iree_vm_function_ref_t* slot =
      ordinal < IREE_VM_CALL_DIRECT_REGISTER_COUNT
          ? &call->function_results.direct[ordinal]
          : &call->function_results
                 .overflow[ordinal - IREE_VM_CALL_DIRECT_REGISTER_COUNT];
  *slot = function_ref;
}

// Loads one valid ref argument as an internal borrow, replacing |inout_ref|
// without changing the disjoint packet source.
IREE_API_EXPORT void iree_vm_call_ref_argument_load_borrow(
    const iree_vm_call_packet_t* call, uint16_t ordinal,
    iree_vm_ref_t* inout_ref);

// Moves one valid ref argument into |inout_ref| and clears the disjoint packet
// source. An internal borrowed source stays borrowed because its anchor lives.
IREE_API_EXPORT void iree_vm_call_ref_argument_load_move(
    const iree_vm_call_packet_t* call, uint16_t ordinal,
    iree_vm_ref_t* inout_ref);

// Moves |inout_ref| into a disjoint valid result and clears it. An internal
// borrow is promoted before it crosses the caller boundary.
IREE_API_EXPORT void iree_vm_call_ref_result_store_move(
    const iree_vm_call_packet_t* call, uint16_t ordinal,
    iree_vm_ref_t* inout_ref);

//===----------------------------------------------------------------------===//
// Module Function Callbacks
//===----------------------------------------------------------------------===//

// Correlated inputs for starting one module-local function.
typedef struct iree_vm_module_function_start_params_t {
  // Active invocation, linked module, and opaque process state.
  iree_vm_module_execution_t execution;
  // Module-local executable function ordinal.
  uint16_t function_ordinal;
  // Physical arguments and private result staging.
  iree_vm_call_packet_t call;
} iree_vm_module_function_start_params_t;

// Correlated inputs for resuming one module-owned top frame.
typedef struct iree_vm_module_function_resume_params_t {
  // Active invocation, linked module, and opaque process state.
  iree_vm_module_execution_t execution;
  // Exact top frame owned by the executing module.
  iree_vm_frame_t* frame;
} iree_vm_module_function_resume_params_t;

// Starts one module-local function. On OK, COMPLETED means all exact results
// are ready and SUSPENDED means continuation state is durable; terminal
// failure leaves |out_outcome| untouched and never represents control flow.
typedef iree_status_t(IREE_API_PTR* iree_vm_module_function_start_fn_t)(
    iree_vm_module_t* module,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome);

// Resumes one module-owned top frame with the same transactional outcome
// contract as a function start.
typedef iree_status_t(IREE_API_PTR* iree_vm_module_function_resume_fn_t)(
    iree_vm_module_t* module,
    const iree_vm_module_function_resume_params_t* params,
    iree_vm_execution_outcome_t* out_outcome);

// Fails if a module whose functions cannot yield is incorrectly resumed.
// Reaching this callback is a trusted module/core ABI violation.
IREE_API_EXPORT iree_status_t iree_vm_module_function_resume_unreachable(
    iree_vm_module_t* module,
    const iree_vm_module_function_resume_params_t* params,
    iree_vm_execution_outcome_t* out_outcome);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_EXECUTION_H_
