// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared AMDGPU lowering vocabulary used by plans and generated tables.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_KINDS_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_KINDS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of 32-bit lanes supported by direct memory descriptors.
#define LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES 4u

// Maximum number of scalarized 32-bit vector lanes the source-to-low path will
// keep live as individual VGPRs. Tile-level register fragments commonly use
// one lane per workitem in a wave, so this must cover a full 32-lane fragment.
#define LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES 32u

// Maximum number of direct memory packets needed to move one scalarized source
// vector payload. Packed 16-bit vectors with an odd lane count may need one
// final sub-dword tail packet after the whole-register packets.
#define LOOM_AMDGPU_MAX_MEMORY_PACKET_COUNT     \
  (((LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES +   \
     LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES - 1u) / \
    LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES) +       \
   1u)

// Maximum number of packed f16/bf16 lanes accepted for packed-half payloads.
#define LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES \
  (LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES * 2u)

// Maximum number of opaque packed 32-bit registers accepted as a register
// payload. This covers wide matrix operands that are passed through as an
// already-packed VGPR tuple, not scalarized vector arithmetic.
#define LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS 16u

// Maximum number of 32-bit registers in one target matrix fragment. Wide CDNA
// MFMA result fragments use a full 32-register tuple.
#define LOOM_AMDGPU_MAX_MATRIX_FRAGMENT_32BIT_REGISTERS 32u

// Number of packed i8 lanes carried by one 32-bit register.
#define LOOM_AMDGPU_PACKED_I8_LANES_PER_REGISTER 4u

// Maximum number of packed i8 lanes accepted by opaque packed-register
// helpers.
#define LOOM_AMDGPU_MAX_PACKED_I8_LANES     \
  (LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS * \
   LOOM_AMDGPU_PACKED_I8_LANES_PER_REGISTER)

// Maximum number of packed i16 lanes accepted by opaque packed-register
// helpers.
#define LOOM_AMDGPU_MAX_PACKED_I16_LANES \
  (LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS * 2u)

typedef enum loom_amdgpu_memory_address_form_e {
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT = 0,
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO = 1,
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR = 2,
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR = 3,
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID = 4,
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT = 5,
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SMEM = 6,
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_SCRATCH_VADDR = 7,
  LOOM_AMDGPU_MEMORY_ADDRESS_FORM_COUNT_,
} loom_amdgpu_memory_address_form_t;

typedef enum loom_amdgpu_atomic_operation_kind_e {
  LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE = 0,
  LOOM_AMDGPU_ATOMIC_OPERATION_RMW = 1,
  LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG = 2,
  LOOM_AMDGPU_ATOMIC_OPERATION_COUNT_,
} loom_amdgpu_atomic_operation_kind_t;

typedef enum loom_amdgpu_fma_mix_source_kind_e {
  // Source operand is interpreted as an f32 lane.
  LOOM_AMDGPU_FMA_MIX_SOURCE_F32 = 0,
  // Source operand is interpreted as the low f16 lane in a 32-bit register.
  LOOM_AMDGPU_FMA_MIX_SOURCE_F16LO = 1,
  // Source operand is interpreted as the high f16 lane in a 32-bit register.
  LOOM_AMDGPU_FMA_MIX_SOURCE_F16HI = 2,
  // Number of FMA-mix source interpretation kinds.
  LOOM_AMDGPU_FMA_MIX_SOURCE_KIND_COUNT_ = 3,
} loom_amdgpu_fma_mix_source_kind_t;

enum {
  // Number of source operands in mixed-FMA packet order.
  LOOM_AMDGPU_FMA_MIX_SOURCE_COUNT = 3,
  // Number of multiplicand operands in mixed-multiply source order.
  LOOM_AMDGPU_MULF_MIX_SOURCE_COUNT = 2,
  // Number of source operands in packed ternary packet order.
  LOOM_AMDGPU_PACKED_TERNARY_SOURCE_COUNT = 3,
};

typedef uint32_t loom_amdgpu_packed_ternary_flags_t;

enum {
  // The descriptor ties its result to the first source operand.
  LOOM_AMDGPU_PACKED_TERNARY_FLAG_TIED_ACCUMULATOR = 1u << 0,
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_KINDS_H_
