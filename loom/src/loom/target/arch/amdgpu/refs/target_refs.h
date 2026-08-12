// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target-low descriptor reference constants, facts, and lookup helpers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_REFS_TARGET_REFS_H_
#define LOOM_TARGET_ARCH_AMDGPU_REFS_TARGET_REFS_H_

#include <stdint.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/target/arch/amdgpu/descriptors/target_refs_tables.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint16_t loom_amdgpu_descriptor_ref_t;

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
  // Descriptor reads one VGPR lane into an SGPR.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_READFIRSTLANE = 1u << 5,
  // Descriptor uses an SDWA packet encoding.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_SDWA = 1u << 6,
  // Descriptor implicitly drains gfx125x XCNT before it executes.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_XCNT_IMPLICIT_DRAIN = 1u << 7,
  // Descriptor issues on an AMDGPU matrix pipeline.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX = 1u << 8,
  // Descriptor advances the vector issue stream used by matrix coexecution.
  // This includes ordinary VALU, WMMA/SWMMAC, and tensor-to-LDS packets even
  // when their primary schedule resources are modeled separately.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ISSUE = 1u << 9,
  // Descriptor establishes a matrix/vector coexecution retention window.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_COEXECUTION_SOURCE = 1u << 10,
  // Descriptor writes a sub-DWORD result through destination-selection
  // forwarding rather than an ordinary full-register write.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_DESTINATION_SELECTION_FORWARDING = 1u << 11,
} loom_amdgpu_descriptor_trait_bit_t;
typedef uint32_t loom_amdgpu_descriptor_traits_t;

// Relative completion order of VMEM instructions that write vector-register
// results. Distinct classes may complete out of order even when they share a
// native wait counter.
typedef enum loom_amdgpu_vmem_result_order_class_e {
  // Descriptor does not write an asynchronous VMEM result.
  LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE = 0,
  // Descriptor writes a VMEM result whose completion class is not known.
  LOOM_AMDGPU_VMEM_RESULT_ORDER_UNKNOWN = 1,
  // Buffer, flat, global, or scratch VMEM result.
  LOOM_AMDGPU_VMEM_RESULT_ORDER_NOSAMPLER = 2,
  // Image sampling VMEM result.
  LOOM_AMDGPU_VMEM_RESULT_ORDER_SAMPLER = 3,
  // Bounding-volume hierarchy VMEM result.
  LOOM_AMDGPU_VMEM_RESULT_ORDER_BVH = 4,
  // Number of VMEM result-order classes, including NONE.
  LOOM_AMDGPU_VMEM_RESULT_ORDER_CLASS_COUNT = 5,
} loom_amdgpu_vmem_result_order_class_t;

typedef enum loom_amdgpu_reg_class_trait_bit_e {
  // Register class is the CDNA accumulator file.
  LOOM_AMDGPU_REG_CLASS_TRAIT_AGPR = 1u << 0,
  // Register class is the M0 scalar special register.
  LOOM_AMDGPU_REG_CLASS_TRAIT_M0 = 1u << 1,
  // Register class is the VCC vector condition mask.
  LOOM_AMDGPU_REG_CLASS_TRAIT_VCC = 1u << 2,
} loom_amdgpu_reg_class_trait_bit_t;
typedef uint8_t loom_amdgpu_reg_class_traits_t;

typedef struct loom_amdgpu_descriptor_immediate_slots_t {
  // Descriptor-local SDWA destination-selector immediate, or LOOM_LOW_ID_NONE.
  uint16_t sdwa_dst_sel;
  // Descriptor-local literal payload immediate, or LOOM_LOW_ID_NONE.
  uint16_t literal;
  // Descriptor-local address offset immediate, or LOOM_LOW_ID_NONE.
  uint16_t address_offset;
} loom_amdgpu_descriptor_immediate_slots_t;

uint32_t loom_amdgpu_descriptor_ref_ordinal(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref);

const loom_low_descriptor_t* loom_amdgpu_descriptor_ref_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref);

// Returns the generated stable descriptor ref for |descriptor|, or NONE when
// |descriptor| is not in the selected AMDGPU descriptor-ref domain.
loom_amdgpu_descriptor_ref_t loom_amdgpu_descriptor_ref_for_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns generated target-owned semantic trait bits for |descriptor|.
loom_amdgpu_descriptor_traits_t loom_amdgpu_descriptor_traits(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns the generated VMEM result completion-order class for |descriptor|.
loom_amdgpu_vmem_result_order_class_t
loom_amdgpu_descriptor_vmem_result_order_class(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns generated target-owned descriptor-local immediate semantic slots for
// |descriptor|.
loom_amdgpu_descriptor_immediate_slots_t loom_amdgpu_descriptor_immediate_slots(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor);

// Returns generated target-owned semantic trait bits for |reg_class_id|.
loom_amdgpu_reg_class_traits_t loom_amdgpu_reg_class_traits(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t reg_class_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_REFS_TARGET_REFS_H_
