// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared AMDGPU lowering vocabulary used by plans and generated tables.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_KINDS_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_KINDS_H_

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
// vector payload.
#define LOOM_AMDGPU_MAX_MEMORY_PACKET_COUNT    \
  ((LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES +   \
    LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES - 1u) / \
   LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES)

// Maximum number of packed f16/bf16 lanes accepted for packed-half payloads.
#define LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES \
  (LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES * 2u)

// Maximum number of opaque packed 32-bit registers accepted as a register
// payload. This covers wide matrix operands that are passed through as an
// already-packed VGPR tuple, not scalarized vector arithmetic.
#define LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS 16u

// Maximum number of packed i8 lanes accepted by opaque packed-register
// helpers.
#define LOOM_AMDGPU_MAX_PACKED_I8_LANES \
  (LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS * 4u)

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
} loom_amdgpu_memory_address_form_t;

typedef enum loom_amdgpu_atomic_operation_kind_e {
  LOOM_AMDGPU_ATOMIC_OPERATION_REDUCE = 0,
  LOOM_AMDGPU_ATOMIC_OPERATION_RMW = 1,
  LOOM_AMDGPU_ATOMIC_OPERATION_CMPXCHG = 2,
} loom_amdgpu_atomic_operation_kind_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_KINDS_H_
