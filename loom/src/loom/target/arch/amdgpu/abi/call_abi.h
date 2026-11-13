// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU module-internal direct-call ABI classification.

#ifndef LOOM_TARGET_ARCH_AMDGPU_ABI_CALL_ABI_H_
#define LOOM_TARGET_ARCH_AMDGPU_ABI_CALL_ABI_H_

#include "loom/codegen/low/allocation/call.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

// A callable helper returns with the exact EXEC mask it received. Structured
// AMDGPU control lowering restores the incoming mask at every function exit;
// authored target assembly follows the same callee-preserved ABI contract.
enum {
  // SGPR4-SGPR29 carry scalar arguments and results. The remaining low SGPRs
  // are available to the helper body or reserved for call control below.
  LOOM_AMDGPU_CALL_ABI_SGPR_BASE = 4,
  LOOM_AMDGPU_CALL_ABI_SGPR_COUNT = 26,
  // VGPR0-VGPR31 carry lane-varying arguments and results.
  LOOM_AMDGPU_CALL_ABI_VGPR_BASE = 0,
  LOOM_AMDGPU_CALL_ABI_VGPR_COUNT = 32,
  // SGPR30:31 hold the return address written by S_SWAPPC_B64.
  LOOM_AMDGPU_CALL_ABI_RETURN_ADDRESS_BASE = 30,
  // SGPR32:33 hold the resolved callee PC at a direct call.
  LOOM_AMDGPU_CALL_ABI_CALL_TARGET_BASE = 32,
  // Contiguous SGPR range unavailable to ordinary allocation in every
  // function participating in a direct-call program.
  LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_BASE = 30,
  LOOM_AMDGPU_CALL_ABI_RESERVED_SGPR_COUNT = 4,
  // Native scalar instructions before one retained direct callee: form the
  // callee PC and transfer control. EXEC is callee-preserved.
  LOOM_AMDGPU_CALL_ABI_CALL_CONTROL_INSTRUCTION_COUNT = 4,
};

// Register-bank demand for one side of a direct-call boundary.
typedef struct loom_amdgpu_call_abi_requirements_t {
  // Number of contiguous 32-bit SGPR units.
  uint16_t sgpr_count;
  // Number of contiguous 32-bit VGPR units.
  uint16_t vgpr_count;
} loom_amdgpu_call_abi_requirements_t;

// Independently planned fixed-state families that a callable may observe or
// perturb through its physical register footprint.
//
// Every family except DELAY_ALU is quiescent at the callable boundary.
// DELAY_ALU identifies optional scheduling-hint history instead: an opaque
// callee invalidates the caller's producer identities without requiring the
// caller or callee to wait.
typedef uint8_t loom_amdgpu_call_fixed_state_mask_t;
enum {
  LOOM_AMDGPU_CALL_FIXED_STATE_SHORT_WAIT = 1u << 0,
  LOOM_AMDGPU_CALL_FIXED_STATE_MATRIX_RESULT = 1u << 1,
  LOOM_AMDGPU_CALL_FIXED_STATE_MATRIX_COEXECUTION = 1u << 2,
  LOOM_AMDGPU_CALL_FIXED_STATE_DELAY_ALU = 1u << 3,
  LOOM_AMDGPU_CALL_FIXED_STATE_MASK_WRITE = 1u << 4,
  LOOM_AMDGPU_CALL_FIXED_STATE_SGPR_READ = 1u << 5,
  LOOM_AMDGPU_CALL_FIXED_STATE_ALL =
      LOOM_AMDGPU_CALL_FIXED_STATE_SHORT_WAIT |
      LOOM_AMDGPU_CALL_FIXED_STATE_MATRIX_RESULT |
      LOOM_AMDGPU_CALL_FIXED_STATE_MATRIX_COEXECUTION |
      LOOM_AMDGPU_CALL_FIXED_STATE_DELAY_ALU |
      LOOM_AMDGPU_CALL_FIXED_STATE_MASK_WRITE |
      LOOM_AMDGPU_CALL_FIXED_STATE_SGPR_READ,
};

// Returns the fixed-state families that can observe accesses to
// |register_class| with the generated target-owned |register_class_traits|.
loom_amdgpu_call_fixed_state_mask_t
loom_amdgpu_call_fixed_states_for_register_class(
    uint16_t register_class,
    loom_amdgpu_reg_class_traits_t register_class_traits);

// Machine state contract for one independently planned callable helper.
//
// The caller establishes every named quiescent state before transferring
// control, and the helper restores it before returning. DELAY_ALU history is
// invalidated by the opaque call instead of restored. State absent from this
// summary passes through the call unchanged.
typedef struct loom_amdgpu_call_state_t {
  // Logical wait-counter bits the helper may issue. Each named counter is
  // empty on helper entry and return.
  uint8_t counter_mask;
  // Fixed physical state families touched by the helper. Boundary-state
  // families are quiescent on helper entry and return; DELAY_ALU invalidates
  // optional caller hint history.
  loom_amdgpu_call_fixed_state_mask_t fixed_state_mask;
} loom_amdgpu_call_state_t;

// Symbol-indexed call-state summaries available while planning one caller.
typedef struct loom_amdgpu_call_state_provider_t {
  // Returns the immutable state contract for |callee|.
  const loom_amdgpu_call_state_t* (*query)(void* user_data,
                                           loom_symbol_ref_t callee);
  // Opaque value passed to |query|.
  void* user_data;
} loom_amdgpu_call_state_provider_t;

// Resolves the immutable state contract for |callee|. An absent provider
// conservatively returns every representable counter and fixed-state family.
loom_amdgpu_call_state_t loom_amdgpu_call_state_resolve(
    loom_amdgpu_call_state_provider_t provider, loom_symbol_ref_t callee);

// Adds the physical direct-call carrier required by one target-low type.
// Returns false unless the type is an SGPR or VGPR range from
// |descriptor_set_id| and the accumulated unit count remains representable.
bool loom_amdgpu_call_abi_add_low_type(
    uint64_t descriptor_set_id, loom_type_t low_type,
    loom_amdgpu_call_abi_requirements_t* inout_requirements);

// Returns true when one argument or result side fits the fixed register ABI.
bool loom_amdgpu_call_abi_requirements_fit(
    loom_amdgpu_call_abi_requirements_t requirements);

// Adds one lowered carrier to |inout_requirements| and selects its fixed
// physical location. Requirements start empty independently for arguments and
// results.
iree_status_t loom_amdgpu_call_abi_select_location(
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t low_type,
    loom_amdgpu_call_abi_requirements_t* inout_requirements,
    loom_low_allocation_abi_location_t* out_location);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_ABI_CALL_ABI_H_
