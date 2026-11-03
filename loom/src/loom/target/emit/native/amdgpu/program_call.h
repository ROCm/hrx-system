// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU native-program call contracts and compiled helper records.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_CALL_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_CALL_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/call.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/amdgpu/abi/call_abi.h"
#include "loom/target/emit/native/amdgpu/function_compile.h"

#ifdef __cplusplus
extern "C" {
#endif

// One module-internal helper prepared for a fixed AMDGPU call convention.
typedef struct loom_amdgpu_program_callable_t {
  // Module-local function symbol naming this helper.
  loom_symbol_ref_t function;
  // Dense helper ordinal in final program text order.
  iree_host_size_t helper_ordinal;
  // Resolved helper target retained for native emission and diagnostics.
  loom_low_resolved_target_t target;
  // ABI carriers for arguments followed by results.
  loom_low_allocation_abi_location_t* locations;
  // Physical ranges overwritten by ABI transport before or after the body.
  loom_low_call_clobber_t* transport_clobbers;
  // Number of rows in |transport_clobbers|.
  iree_host_size_t transport_clobber_count;
  // Number of argument carriers in |locations|.
  uint16_t argument_count;
  // Number of result carriers following the arguments.
  uint16_t result_count;
  // Register-bank demand of the complete argument tuple.
  loom_amdgpu_call_abi_requirements_t argument_requirements;
  // Register-bank demand of the complete result tuple.
  loom_amdgpu_call_abi_requirements_t result_requirements;
  // Body and transport clobbers in allocation storage order.
  loom_low_call_clobber_t* clobbers;
  // Number of rows in |clobbers|.
  iree_host_size_t clobber_count;
  // Immutable call contract borrowing carriers and clobbers above.
  loom_low_call_contract_t call_contract;
  // Native function compilation paired with this exact contract.
  loom_amdgpu_function_compilation_t compilation;
} loom_amdgpu_program_callable_t;

// Symbol-indexed helper contracts visible while allocating one entry function.
typedef struct loom_amdgpu_program_call_table_t {
  // Descriptor set shared by the entry and every helper.
  const loom_low_descriptor_set_t* descriptor_set;
  // Published helpers indexed directly by module-local symbol ID.
  loom_amdgpu_program_callable_t** callables_by_symbol;
  // Number of entries in |callables_by_symbol|.
  iree_host_size_t symbol_count;
  // Architectural writes performed by every direct call.
  loom_low_call_clobber_list_t common_clobbers;
} loom_amdgpu_program_call_table_t;

// Initializes an empty call table and derives target-wide call clobbers.
iree_status_t loom_amdgpu_program_call_table_initialize(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_host_size_t symbol_count, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* scratch_arena,
    loom_amdgpu_program_call_table_t* out_table);

// Selects fixed SGPR/VGPR carriers for one helper signature.
//
// Unsupported signatures emit a structured diagnostic and leave
// |out_prepared| false. Infrastructure failures return a status.
iree_status_t loom_amdgpu_program_callable_prepare_abi(
    const loom_module_t* module, loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    bool* out_prepared, loom_amdgpu_program_callable_t* callable);

// Derives the complete helper-body and ABI-transport clobber contract after
// |callable->compilation| has been accepted.
iree_status_t loom_amdgpu_program_callable_prepare_clobbers(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch_arena,
    loom_amdgpu_program_callable_t* callable);

// Publishes one fully prepared helper at its module-local symbol ID.
void loom_amdgpu_program_call_table_publish(
    loom_amdgpu_program_call_table_t* table,
    loom_amdgpu_program_callable_t* callable);

// Returns an allocation provider borrowing the published table.
loom_low_call_contract_provider_t loom_amdgpu_program_call_table_provider(
    loom_amdgpu_program_call_table_t* table);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_CALL_H_
