// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU instruction-constraint classification.

#ifndef LOOM_TARGET_ARCH_AMDGPU_INSTRUCTION_CONSTRAINTS_H_
#define LOOM_TARGET_ARCH_AMDGPU_INSTRUCTION_CONSTRAINTS_H_

#include "loom/codegen/low/descriptors.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_instruction_constraint_kind_e {
  // An operand or encoded field must be rewritten.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KIND_OPERAND_ENCODING = 0,
  // The instruction form must be replaced by another packet sequence.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KIND_INSTRUCTION_FORM = 1,
  // A required packet sequence must surround the constrained instruction.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KIND_SURROUNDING_SEQUENCE = 2,
  // Scheduling must prevent unsafe coexecution with nearby instructions.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_KIND_COEXECUTION_HAZARD = 3,
} loom_amdgpu_instruction_constraint_kind_t;

typedef enum loom_amdgpu_instruction_constraint_resolution_e {
  // Replace a paired DS packet with independently addressed packets.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_SPLIT_DS_PAIR = 0,
  // Materialize the implicit DS ADDTID address as an explicit operand.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_MATERIALIZE_ADDTID_ADDRESS = 1,
  // Clear and restore the cluster multicast mask around the packet.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_PRESERVE_CLUSTER_MASK = 2,
  // Clear and restore the tensor multicast mask around the packet.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_PRESERVE_TENSOR_MULTICAST_MASK =
      3,
  // Establish a neutral regular scale before the constrained WMMA packet.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_PREFIX_NEUTRAL_REGULAR_SCALE =
      4,
  // Replace one K128 WMMA packet with legal K64 packets.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_SPLIT_WMMA_K128 = 5,
  // Replace one 32x16 WMMA packet with legal 16x16 packets.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_SPLIT_WMMA_32X16 = 6,
  // Rewrite scaled WMMA operands into a legal encoding.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_LEGALIZE_SCALED_WMMA = 7,
  // Replace low-precision SWMMAC with a legal instruction sequence.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_LOWER_LOW_PRECISION_SWMMAC = 8,
  // Schedule integer matrix packets with the required separation.
  LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_RESOLUTION_SPACE_INTEGER_MATRIX = 9,
} loom_amdgpu_instruction_constraint_resolution_t;

typedef struct loom_amdgpu_instruction_constraint_info_t {
  // Singular semantic constraint bit represented by this record.
  loom_amdgpu_instruction_constraint_bit_t constraint;
  // Structural class of the restriction.
  loom_amdgpu_instruction_constraint_kind_t kind;
  // Structured legalization or hazard-handling consequence.
  loom_amdgpu_instruction_constraint_resolution_t resolution;
  // Stable diagnostic key identifying the semantic restriction.
  iree_string_view_t constraint_key;
} loom_amdgpu_instruction_constraint_info_t;

// Returns the semantic constraints that may apply to |descriptor|.
loom_amdgpu_instruction_constraint_bits_t
loom_amdgpu_instruction_constraints_for_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns the structured metadata for singular known |constraint|.
loom_amdgpu_instruction_constraint_info_t
loom_amdgpu_instruction_constraint_info(
    loom_amdgpu_instruction_constraint_bit_t constraint);

// Returns the stable diagnostic spelling of |kind|.
iree_string_view_t loom_amdgpu_instruction_constraint_kind_name(
    loom_amdgpu_instruction_constraint_kind_t kind);

// Returns the stable diagnostic spelling of |resolution|.
iree_string_view_t loom_amdgpu_instruction_constraint_resolution_name(
    loom_amdgpu_instruction_constraint_resolution_t resolution);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_INSTRUCTION_CONSTRAINTS_H_
