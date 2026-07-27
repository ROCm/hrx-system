// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_AMDGPU_EMIT_H_
#define LOOMC_TARGET_AMDGPU_EMIT_H_

#include "loomc/emit.h"
#include "loomc/target/amdgpu/base.h"

/// @file
/// AMDGPU HSACO emission option space.
///
/// Link the AMDGPU target binding package to make the generic
/// `loomc_emit_module` operation capable of producing
/// `LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO` artifacts.

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loomc_amdgpu_runtime_global_flag_bits_e {
  /// No AMDGPU runtime support globals are emitted.
  LOOMC_AMDGPU_RUNTIME_GLOBAL_NONE = 0u,

  /// Emits the common kernel-to-runtime feedback-channel configuration global.
  LOOMC_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG = 1u << 0,

  /// Emits the AMDGPU ASAN configuration global consumed by access checks.
  LOOMC_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG = 1u << 1,
} loomc_amdgpu_runtime_global_flag_bits_t;

/// Bitset of `loomc_amdgpu_runtime_global_flag_bits_t` values.
typedef uint32_t loomc_amdgpu_runtime_global_flags_t;

#define LOOMC_AMDGPU_RUNTIME_GLOBALS_KNOWN                                             \
  ((loomc_amdgpu_runtime_global_flags_t)(LOOMC_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG | \
                                         LOOMC_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG))

/// AMDGPU HSACO emission options.
///
/// AMDGPU emission uses the generic artifact format, artifact identifier, and
/// target selection settings on `loomc_emit_options_t` and shared invocation
/// descriptors. The target-specific descriptor carries AMDGPU code-object data
/// globals required by prepared-low IR that references HAL runtime support.
typedef struct loomc_amdgpu_emit_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Next invocation option extension.
  const void* next;

  /// AMDGPU runtime support globals to emit into the HSACO.
  loomc_amdgpu_runtime_global_flags_t runtime_globals;
} loomc_amdgpu_emit_options_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_AMDGPU_EMIT_H_
