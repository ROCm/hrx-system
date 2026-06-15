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

/// AMDGPU HSACO emission options.
///
/// AMDGPU emission uses the generic artifact format, artifact identifier, and
/// target selection settings on `loomc_emit_options_t` and shared invocation
/// descriptors. This target-specific descriptor carries no additional stable
/// fields today.
typedef struct loomc_amdgpu_emit_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Next invocation option extension.
  const void* next;
} loomc_amdgpu_emit_options_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_AMDGPU_EMIT_H_
