// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared AMDGPU descriptor semantic predicates.
//
// AMDGPU target-low descriptors carry compact target facts generated from the
// Python descriptor tables. This layer centralizes backend predicates that need
// to stay consistent across wait-state and wait-counter planning.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_DESCRIPTOR_SEMANTICS_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_DESCRIPTOR_SEMANTICS_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_descriptor_trait_bit_e {
  // Descriptor issues on an AMDGPU vector ALU pipeline.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU = 1u << 0,
  // Descriptor issues on an AMDGPU scalar ALU pipeline.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_ALU = 1u << 1,
  // Descriptor issues on an AMDGPU vector-memory pipeline.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY = 1u << 2,
  // Descriptor is a transcendental VALU packet.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_TRANSCENDENTAL = 1u << 3,
  // Descriptor is a DPP lane-crossing packet.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_DPP = 1u << 4,
  // Descriptor reads the first active lane of a VGPR into an SGPR.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_READFIRSTLANE = 1u << 5,
  // Descriptor uses an SDWA packet encoding.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_SDWA = 1u << 6,
} loom_amdgpu_descriptor_trait_bit_t;
typedef uint32_t loom_amdgpu_descriptor_traits_t;

// Returns target-owned descriptor semantic facts as a compact bitfield.
loom_amdgpu_descriptor_traits_t loom_amdgpu_descriptor_traits(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns true when |descriptor| issues on a resource with |kind|.
bool loom_amdgpu_descriptor_uses_resource_kind(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_low_resource_kind_t kind);

// Returns true when |descriptor| issues on the vector ALU.
bool loom_amdgpu_descriptor_uses_vector_alu(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns true when |descriptor| issues on the scalar ALU.
bool loom_amdgpu_descriptor_uses_scalar_alu(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns true when |descriptor| issues on an AMDGPU vector-memory pipeline.
bool loom_amdgpu_descriptor_uses_vector_memory(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns true when |descriptor| is a transcendental VALU packet.
bool loom_amdgpu_descriptor_is_transcendental(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns true when |descriptor| is a DPP lane-crossing packet.
bool loom_amdgpu_descriptor_is_dpp(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns true when |descriptor| is a readfirstlane packet.
bool loom_amdgpu_descriptor_is_readfirstlane(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns true when |descriptor| uses an SDWA packet encoding.
bool loom_amdgpu_descriptor_is_sdwa(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_DESCRIPTOR_SEMANTICS_H_
