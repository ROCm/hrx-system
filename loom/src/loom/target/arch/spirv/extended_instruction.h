// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V extended-instruction set and instruction vocabulary.
//
// Set ordinals are compact Loom-internal packet values. Instruction enumerants
// are the wire values assigned by their corresponding Khronos instruction-set
// specification.

#ifndef LOOM_TARGET_ARCH_SPIRV_EXTENDED_INSTRUCTION_H_
#define LOOM_TARGET_ARCH_SPIRV_EXTENDED_INSTRUCTION_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_spirv_extended_instruction_set_e {
  LOOM_SPIRV_EXTENDED_INSTRUCTION_SET_UNKNOWN = 0,
  LOOM_SPIRV_EXTENDED_INSTRUCTION_SET_GLSL_STD_450 = 1,
  LOOM_SPIRV_EXTENDED_INSTRUCTION_SET_COUNT = 2,
} loom_spirv_extended_instruction_set_t;

typedef enum loom_spirv_glsl_std_450_instruction_e {
  LOOM_SPIRV_GLSL_STD_450_ROUND_EVEN = 2,
  LOOM_SPIRV_GLSL_STD_450_TRUNC = 3,
  LOOM_SPIRV_GLSL_STD_450_FLOOR = 8,
  LOOM_SPIRV_GLSL_STD_450_CEIL = 9,
  LOOM_SPIRV_GLSL_STD_450_SIN = 13,
  LOOM_SPIRV_GLSL_STD_450_COS = 14,
  LOOM_SPIRV_GLSL_STD_450_EXP = 27,
  LOOM_SPIRV_GLSL_STD_450_LOG = 28,
  LOOM_SPIRV_GLSL_STD_450_EXP2 = 29,
  LOOM_SPIRV_GLSL_STD_450_LOG2 = 30,
  LOOM_SPIRV_GLSL_STD_450_SQRT = 31,
  LOOM_SPIRV_GLSL_STD_450_INVERSE_SQRT = 32,
  LOOM_SPIRV_GLSL_STD_450_FIND_UMSB = 75,
  LOOM_SPIRV_GLSL_STD_450_NMIN = 79,
  LOOM_SPIRV_GLSL_STD_450_NMAX = 80,
} loom_spirv_glsl_std_450_instruction_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_EXTENDED_INSTRUCTION_H_
