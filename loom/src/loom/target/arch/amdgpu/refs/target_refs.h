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
  // A packet address operand remains physically live after the packet issues.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_ADDRESS_SOURCE_RETAINED = 1u << 12,
  // Descriptor is a PERMLANE lane-crossing packet.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_PERMLANE = 1u << 13,
  // Descriptor writes architectural EXEC state.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_WRITES_EXEC = 1u << 14,
  // Packed F32 arithmetic spells input register pairs even when both result
  // lanes select the first dword. Physical read footprints remain independent.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_PACKED_F32 = 1u << 15,
  // Wide shift operand zero needs its next VGPR allocated when the count
  // occupies the last register of an eight-register allocation block.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_SHIFT64_ALLOCATION = 1u << 16,
  // MFMA executes on the F64 matrix pipeline.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_DGEMM = 1u << 22,
  // MFMA executes on the narrow/integer/XF32 matrix pipeline.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_XDL = 1u << 23,
  // VMEM, DS, or export reads use the matrix-to-memory forwarding window.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_MEMORY_CONSUMER = 1u << 24,
  // DIV_FMAS reads its VCC condition through a separate forwarding path.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_DIV_FMAS = 1u << 25,
  // F64 FMA/FMAC shares a pipeline release dependency with DGEMM.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_FMA64 = 1u << 29,
  // VMEM and DS reads depend on the VALU-to-memory hardware interlock.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY_INTERLOCK = 1u << 30,
  // Descriptor belongs to a scalar-memory replay clause.
  LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_MEMORY = 1u << 31,
} loom_amdgpu_descriptor_trait_bit_t;
typedef uint32_t loom_amdgpu_descriptor_traits_t;

enum {
  // MFMA execution passes occupy five bits in the descriptor's existing trait
  // word. Zero identifies descriptors without software MFMA result waits.
  LOOM_AMDGPU_DESCRIPTOR_MATRIX_PASSES_SHIFT = 17,
  LOOM_AMDGPU_DESCRIPTOR_MATRIX_PASSES_MASK = 31u << 17,
  // One-based native CDNA DOT opcode identity; zero means no fixed DOT wait.
  // Matching opcode accumulator reads and overwrites forward immediately.
  LOOM_AMDGPU_DESCRIPTOR_DOT_IDENTITY_SHIFT = 26,
  LOOM_AMDGPU_DESCRIPTOR_DOT_IDENTITY_MASK = 7u << 26,
};

// Returns the hardware MFMA execution passes, independently of result width
// and the scheduler's estimated latency.
static inline uint16_t loom_amdgpu_descriptor_matrix_pass_count(
    loom_amdgpu_descriptor_traits_t traits) {
  return (uint16_t)((traits & LOOM_AMDGPU_DESCRIPTOR_MATRIX_PASSES_MASK) >>
                    LOOM_AMDGPU_DESCRIPTOR_MATRIX_PASSES_SHIFT);
}

// Returns the native DOT identity's one-hot bit, or zero for other packets.
static inline uint8_t loom_amdgpu_descriptor_dot_identity_mask(
    loom_amdgpu_descriptor_traits_t traits) {
  const unsigned identity =
      (traits & LOOM_AMDGPU_DESCRIPTOR_DOT_IDENTITY_MASK) >>
      LOOM_AMDGPU_DESCRIPTOR_DOT_IDENTITY_SHIFT;
  return (uint8_t)((1u << identity) >> 1);
}

// Relative completion order of VMEM loads and cache invalidations. Distinct
// classes may complete out of order even when they share a native wait counter.
typedef enum loom_amdgpu_vmem_result_order_class_e {
  // Descriptor does not participate in VMEM load-result ordering.
  LOOM_AMDGPU_VMEM_RESULT_ORDER_NONE = 0,
  // Descriptor writes a VMEM result whose completion class is not known.
  LOOM_AMDGPU_VMEM_RESULT_ORDER_UNKNOWN = 1,
  // Buffer, flat, global, or scratch VMEM result, or ordered cache
  // invalidation.
  LOOM_AMDGPU_VMEM_RESULT_ORDER_NOSAMPLER = 2,
  // Image sampling VMEM result.
  LOOM_AMDGPU_VMEM_RESULT_ORDER_SAMPLER = 3,
  // Bounding-volume hierarchy VMEM result.
  LOOM_AMDGPU_VMEM_RESULT_ORDER_BVH = 4,
  // Number of VMEM result-order classes, including NONE.
  LOOM_AMDGPU_VMEM_RESULT_ORDER_CLASS_COUNT = 5,
} loom_amdgpu_vmem_result_order_class_t;

// Compact memory properties use three bits for result order and three for the
// one-based retained payload operand. A retained payload requires one VALU
// slot; the remaining bits independently add one non-VALU slot and a second
// VALU slot.
typedef struct loom_amdgpu_store_data_wait_t {
  // Descriptor operand carrying the retained VGPR/AGPR payload, when
  // valu_cycles is nonzero.
  uint8_t operand_index;
  // Intervening issue slots before a VALU or matrix overwrite, at most two.
  uint8_t valu_cycles : 2;
  // Intervening issue slots before another physical overwrite, at most one.
  uint8_t other_cycles : 1;
} loom_amdgpu_store_data_wait_t;

static_assert(sizeof(loom_amdgpu_store_data_wait_t) == 2,
              "store retention bindings must remain compact by-value results");

typedef enum loom_amdgpu_reg_class_trait_bit_e {
  // Register class is the CDNA accumulator file.
  LOOM_AMDGPU_REG_CLASS_TRAIT_AGPR = 1u << 0,
  // Register class is the M0 scalar special register.
  LOOM_AMDGPU_REG_CLASS_TRAIT_M0 = 1u << 1,
  // Register class is the VCC vector condition mask.
  LOOM_AMDGPU_REG_CLASS_TRAIT_VCC = 1u << 2,
} loom_amdgpu_reg_class_trait_bit_t;
typedef uint8_t loom_amdgpu_reg_class_traits_t;

#define LOOM_AMDGPU_DESCRIPTOR_IMMEDIATE_SLOT_NONE UINT8_MAX

typedef struct loom_amdgpu_descriptor_immediate_slots_t {
  // Descriptor-local SDWA destination-selector immediate, or the NONE
  // sentinel.
  uint8_t sdwa_dst_sel;
  // Descriptor-local literal payload immediate, or the NONE sentinel.
  uint8_t literal;
  // Descriptor-local address offset immediate, or the NONE sentinel.
  uint8_t address_offset;
  // Descriptor-local vector-memory coherence immediate, or the NONE
  // sentinel. This names GLC on GFX11/CDNA2 and SCOPE on GFX12.
  uint8_t cache_coherence;
} loom_amdgpu_descriptor_immediate_slots_t;

static_assert(sizeof(loom_amdgpu_descriptor_immediate_slots_t) == 4,
              "descriptor immediate slots must remain compact");

uint32_t loom_amdgpu_descriptor_ref_ordinal(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref);

const loom_low_descriptor_t* loom_amdgpu_descriptor_ref_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref);

// Returns the generated target ref for |descriptor|, or NONE when |descriptor|
// is not in the selected AMDGPU descriptor-ref domain.
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

// Returns the wide-store source retention contract, or zero counts when absent.
loom_amdgpu_store_data_wait_t loom_amdgpu_descriptor_store_data_wait(
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

// Returns the accumulator class in a resolved AMDGPU descriptor set, or
// UINT8_MAX when absent. MFMA descriptor sets always have an accumulator class.
uint8_t loom_amdgpu_agpr_reg_class_id(
    const loom_low_descriptor_set_t* descriptor_set);

// Returns the native one-unit accumulator transfer for the physical register
// banks, or NONE when the pair is not an accumulator transfer. The shared move
// planner owns tuple ordering, overlap, and cycle scratch.
loom_amdgpu_descriptor_ref_t loom_amdgpu_accumulator_copy_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t destination_reg_class_id, uint16_t source_reg_class_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_REFS_TARGET_REFS_H_
