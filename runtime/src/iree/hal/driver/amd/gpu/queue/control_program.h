// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_CONTROL_PROGRAM_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_CONTROL_PROGRAM_H_

#include "iree/hal/driver/amd/gpu/image/image.h"
#include "iree/hal/driver/amd/gpu/queue/control.h"
#include "iree/hal/driver/amd/gpu/target/identity.h"

#ifdef __cplusplus
extern "C" {
#endif

// One scalar invocation; the enclosing queue/recording owns argument lifetime.
typedef struct iree_hal_amd_gpu_control_program_arguments_t {
  // Naturally aligned native target address.
  uint64_t target;
  // Wait reference, store value, or modular RMW operand.
  uint64_t value;
  // Mask applied only to a wait's loaded value.
  uint64_t mask;
  // EQ/NE/UGE waits (0..2), STORE (3), or ADD/SUB/AND/OR/XOR (4..8).
  uint32_t opcode;
  // Zero padding to the native kernarg alignment.
  uint32_t reserved;
} iree_hal_amd_gpu_control_program_arguments_t;
static_assert(sizeof(iree_hal_amd_gpu_control_program_arguments_t) == 32,
              "control arguments match the compiled ABI");

// Four entries in device32/device64/system32/system64 order.
#define IREE_HAL_AMD_GPU_CONTROL_PROGRAM_COUNT 4

// Public command validation has established the width, kind, and operation.
// Timestamps always use the command processor and never enter these helpers.
static inline uint32_t iree_hal_amd_gpu_control_program_index(
    iree_hal_amd_gpu_control_t command) {
  return (command.width == IREE_HAL_ATOMIC_WIDTH_64 ? 1u : 0u) |
         (command.flags & IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE ? 2u : 0u);
}
static inline iree_hal_amd_gpu_control_program_arguments_t
iree_hal_amd_gpu_control_program_arguments(uint64_t target,
                                           iree_hal_amd_gpu_control_t command) {
  return (iree_hal_amd_gpu_control_program_arguments_t){
      .target = target,
      .value = command.value,
      .mask = command.mask,
      .opcode = command.kind == IREE_HAL_AMD_GPU_CONTROL_WAIT
                    ? command.operation
                : command.kind == IREE_HAL_AMD_GPU_CONTROL_STORE
                    ? 3u
                    : 4u + command.operation,
  };
}

// Relocation-free code and descriptor for one qualified scope/width recipe.
// Cold owners place all entries with their other builtins, without a parser.
iree_status_t iree_hal_amd_gpu_control_program_query(
    iree_hal_amd_gpu_gfxip_version_t gfxip, uint32_t index,
    iree_const_byte_span_t* out_code,
    iree_hal_amd_gpu_image_kernel_t* out_kernel);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_CONTROL_PROGRAM_H_
