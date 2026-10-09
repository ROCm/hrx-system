// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU runtime support globals emitted into Loom-native HSACO artifacts.
//
// The compiler treats these as named code-object data symbols. Runtime-side
// structure layouts and HSA symbol lookup are owned by the AMDGPU HAL loader;
// this header deliberately does not include HSA or HAL runtime ABI headers.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_RUNTIME_GLOBALS_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_RUNTIME_GLOBALS_H_

#include "iree/base/api.h"
#include "loom/target/arch/amdgpu/abi/asan.h"
#include "loom/target/arch/amdgpu/abi/feedback.h"
#include "loom/target/arch/amdgpu/abi/tsan.h"
#include "loom/target/emit/native/amdgpu/hsaco_build.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_runtime_global_flag_bits_e {
  // No AMDGPU runtime support globals are emitted.
  LOOM_AMDGPU_RUNTIME_GLOBAL_NONE = 0u,
  // Emits the common kernel-to-runtime feedback channel configuration global.
  LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG = 1u << 0,
  // Emits the AMDGPU ASAN configuration global consumed by sanitizer reports.
  LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG = 1u << 1,
  // Emits the AMDGPU TSAN configuration global consumed by race observers.
  LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG = 1u << 2,
} loom_amdgpu_runtime_global_flag_bits_t;

// Bitset of loom_amdgpu_runtime_global_flag_bits_t values.
typedef uint32_t loom_amdgpu_runtime_global_flags_t;

enum {
  // Maximum number of symbols produced by the current runtime-global bitset.
  LOOM_AMDGPU_RUNTIME_GLOBAL_SYMBOL_CAPACITY = 4u,
  // Writable configuration globals use pointer-granularity alignment.
  LOOM_AMDGPU_RUNTIME_GLOBAL_CONFIG_ALIGNMENT = 8u,
  // Byte length of the AMDGPU HAL feedback-channel configuration global.
  LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG_BYTE_LENGTH = 64u,
  // Byte length of the AMDGPU HAL ASAN configuration global.
  LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG_BYTE_LENGTH = 96u,
  // Byte length of the ASAN global-layout marker object.
  LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_LAYOUT_MARKER_BYTE_LENGTH = 1u,
  // Byte length of the AMDGPU HAL TSAN configuration global.
  LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG_BYTE_LENGTH = 96u,
};

#define LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG_NAME \
  IREE_SVL(LOOM_AMDGPU_FEEDBACK_CONFIG_GLOBAL_NAME)
#define LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG_NAME \
  IREE_SVL(LOOM_AMDGPU_ASAN_CONFIG_GLOBAL_NAME)
#define LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_LAYOUT_MARKER_NAME \
  IREE_SVL(LOOM_AMDGPU_ASAN_GLOBAL_LAYOUT_V0_MARKER_NAME)
#define LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG_NAME \
  IREE_SVL(LOOM_AMDGPU_TSAN_CONFIG_GLOBAL_NAME)

// Returns the number of HSACO data symbols required for |flags|.
iree_host_size_t loom_amdgpu_runtime_global_count(
    loom_amdgpu_runtime_global_flags_t flags);

// Writes HSACO data symbols for |flags| into |out_symbols|.
//
// |out_symbols| must have capacity for
// LOOM_AMDGPU_RUNTIME_GLOBAL_SYMBOL_CAPACITY entries. The returned symbols
// reference static string storage and contain zero initial contents so the HAL
// loader can populate them after executable load.
void loom_amdgpu_runtime_global_symbols(
    loom_amdgpu_runtime_global_flags_t flags,
    loom_amdgpu_hsaco_data_symbol_t* out_symbols,
    iree_host_size_t* out_symbol_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_RUNTIME_GLOBALS_H_
