// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Retained x86 platform-ABI classification.

#ifndef LOOM_TARGET_EMIT_NATIVE_X86_ABI_H_
#define LOOM_TARGET_EMIT_NATIVE_X86_ABI_H_

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/call.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/x86/call_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Target-specific facts parallel to one generic argument or result location.
typedef struct loom_x86_abi_value_t {
  // Caller-RSP-relative byte offset, or UINT32_MAX for a register value.
  uint32_t stack_offset;
  // Significant bytes transferred to or from a stack slot.
  uint16_t byte_length;
  // Required stack address alignment in bytes.
  uint8_t byte_alignment;
  // loom_x86_call_abi_value_action_t applied after inbound transport.
  uint8_t action;
} loom_x86_abi_value_t;

static_assert(sizeof(loom_x86_abi_value_t) == 8,
              "x86 ABI value rows must remain compact");

// One immutable SysV classification retained for a function symbol.
typedef struct loom_x86_function_abi_t {
  // Resolved representation and profile shared by classification and emission.
  loom_low_resolved_target_t target;
  // Generic allocation contract over |arguments| and |results|.
  loom_low_call_contract_t call_contract;
  // Target stack and normalization facts indexed by logical argument.
  loom_x86_abi_value_t* arguments;
  // Target stack and normalization facts indexed by logical result.
  loom_x86_abi_value_t* results;
  // Bytes occupied by all stack arguments, including inter-argument padding.
  uint32_t stack_argument_bytes;
  // Bytes occupied by arguments and caller-owned result storage.
  uint32_t call_storage_bytes;
  // Required caller-RSP alignment, or zero when no argument uses the stack.
  uint8_t stack_argument_alignment;
  // Required caller-RSP alignment for arguments and indirect results.
  uint8_t call_storage_alignment;
  // True when an incoming SIMD stack value needs a stable scalar address base.
  bool has_simd_stack_argument;
  // True when module-internal results overflow their register banks.
  bool has_indirect_results;
  // True when an argument retains live YMM/ZMM0-15 state through CALL.
  bool has_upper_vector_register_argument;
  // True when the result returns live YMM/ZMM0-15 state to the caller.
  bool has_upper_vector_result;
} loom_x86_function_abi_t;

// Module-lifetime ABI plans with O(1) symbol lookup. The dense pointer index is
// proportional to all symbols while full records exist only for selected
// functions.
typedef struct loom_x86_module_abi_t {
  // Arena-owned contiguous plans in selection order.
  loom_x86_function_abi_t* functions;
  // Number of initialized plans in |functions|.
  iree_host_size_t function_count;
  // Borrowed plan pointers indexed by module symbol ID.
  loom_x86_function_abi_t** functions_by_symbol;
  // Number of entries in |functions_by_symbol|.
  iree_host_size_t symbol_count;
} loom_x86_module_abi_t;

// Allocates storage for a module ABI with exactly |function_count| plans.
iree_status_t loom_x86_module_abi_initialize(
    iree_host_size_t symbol_count, iree_host_size_t function_count,
    iree_arena_allocator_t* arena, loom_x86_module_abi_t* out_module_abi);

// Classifies one verified Low function into caller/callee SysV locations. The
// logical signature comes from abi_layout.signature for ordinary functions when
// present; otherwise the Low carrier defines the boundary. HAL task layouts
// describe dispatch parameters rather than their fixed physical adapter and do
// not override its carrier signature. |resolved_target| is the function's
// resolved representation and prevents publishing unavailable register
// classes. Unsupported user signatures return OK with |out_supported| false
// and a borrowed diagnostic constraint. Allocated rows remain owned by |arena|.
iree_status_t loom_x86_function_abi_prepare(
    const loom_module_t* module, loom_func_like_t function,
    const loom_low_resolved_target_t* resolved_target,
    iree_arena_allocator_t* arena, bool* out_supported,
    iree_string_view_t* out_constraint, loom_x86_function_abi_t* out_abi);

// Binds an initialized function plan to its module-local symbol ID.
void loom_x86_module_abi_bind(loom_x86_module_abi_t* module_abi,
                              iree_host_size_t function_index,
                              loom_symbol_id_t symbol_id);

// Returns a retained function plan, or NULL when the symbol is not part of this
// native module.
const loom_x86_function_abi_t* loom_x86_module_abi_lookup(
    const loom_x86_module_abi_t* module_abi, loom_symbol_ref_t function);

// Validates that a retained call resolves inside the prepared native module.
iree_status_t loom_x86_function_call_contract_validate(
    void* user_data, loom_symbol_ref_t callee);

// Returns the retained allocation contract for an admitted callee.
const loom_low_call_contract_t* loom_x86_function_call_contract(
    void* user_data, loom_symbol_ref_t callee);

// Returns SysV registers clobbered by every call for the resolved x86 profile.
// The widest available SIMD class covers all of its narrower aliases.
loom_low_call_clobber_list_t loom_x86_function_common_call_clobbers(
    void* user_data, const loom_low_descriptor_set_t* descriptor_set);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_X86_ABI_H_
