// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native entry/return instructions for one allocated AIE2P function.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_FUNCTION_ENVELOPE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_FUNCTION_ENVELOPE_H_

#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_alignment.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/stack_frame.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_aie2p_return_address_kind_e {
  // Leaf or nonreturning function; no incoming link needs preservation.
  LOOM_AIE2P_RETURN_ADDRESS_NONE = 0,
  // Returning entry owns the shadow register across its helper calls.
  LOOM_AIE2P_RETURN_ADDRESS_REGISTER,
  // Returning helper also preserves its caller's shadow in its own frame.
  LOOM_AIE2P_RETURN_ADDRESS_FRAME,
} loom_aie2p_return_address_kind_t;

typedef struct loom_aie2p_return_address_t {
  // Lifetime of the incoming link and the caller's shadow register.
  loom_aie2p_return_address_kind_t kind;
  // Reserved scalar register holding the function's incoming link.
  loom_aie2p_physical_register_id_t shadow_register;
} loom_aie2p_return_address_t;

enum {
  // The scalar shadow occupies the first word below the acquired frame's SP.
  // Ordinary storage and allocation-repair slots are placed below this word.
  LOOM_AIE2P_RETURN_ADDRESS_FRAME_BYTES = 4,
};

typedef struct loom_aie2p_function_envelope_instruction_t {
  // Encoded instruction and provenance consumed by ordinary physical issue.
  loom_aie2p_planned_slot_t slot;
  // Descriptor or move binding for that exact instruction.
  loom_aie2p_slot_realization_t realization;
} loom_aie2p_function_envelope_instruction_t;

typedef struct loom_aie2p_function_envelope_sequence_t {
  // Stack adjustment, optional shadow transfer, and optional link transfer.
  loom_aie2p_function_envelope_instruction_t instructions[3];
  // Number of instructions in semantic execution order.
  uint8_t count;
} loom_aie2p_function_envelope_sequence_t;

typedef struct loom_aie2p_function_envelope_t {
  // Executed once before entry argument transport and the body CFG.
  loom_aie2p_function_envelope_sequence_t entry;
  // Executed on every return after result transport and before the return.
  loom_aie2p_function_envelope_sequence_t exit;
} loom_aie2p_function_envelope_t;

// Encodes the admitted calling convention once. These instructions enter the
// same register-event and resource admission as body instructions. A FRAME
// return address requires its word to have been reserved in |stack| before
// allocation; its position is independent of later spill growth.
void loom_aie2p_function_envelope_build(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_aie2p_stack_frame_t* stack,
    loom_aie2p_return_address_t return_address,
    loom_aie2p_function_envelope_t* out_envelope);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_FUNCTION_ENVELOPE_H_
